#!/usr/bin/env python3
"""
Transform and Embed

Apply a reaction SMARTS transform to input molecules and generate optimized
3D conformers for each unique product. Demonstrates the §4 workflow:
reaction transform -> sanitize/dedupe products -> AddHs -> ETKDG embed ->
MMFF/UFF optimization.

Usage:
    python transform_and_embed.py molecules.smi \\
        --reaction "[C:1](=[O:2])[OH]>>[C:1](=[O:2])[O][CH3]" \\
        --num-confs 5 --output products.sdf

    # Read SMILES from stdin, write best conformer per product to SDF
    echo "OC(=O)c1ccccc1" | python transform_and_embed.py - \\
        --reaction "[C:1](=[O:2])[OH]>>[C:1](=[O:2])[OCC]"
"""

import argparse
import sys
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    print("Error: RDKit not installed. Install with: conda install -c conda-forge rdkit")
    sys.exit(1)


def read_smiles(source):
    """Yield (name, smiles) from a .smi file, an .sdf file, or stdin ('-')."""
    if source == "-":
        for i, line in enumerate(sys.stdin):
            smi = line.strip().split()[0] if line.strip() else ""
            if smi:
                yield f"stdin_{i}", smi
        return

    path = Path(source)
    if path.suffix.lower() == ".sdf":
        for i, mol in enumerate(Chem.SDMolSupplier(str(path))):
            if mol is not None:
                yield mol.GetProp("_Name") or f"mol_{i}", Chem.MolToSmiles(mol)
        return

    with open(path) as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if parts:
                name = parts[1] if len(parts) > 1 else f"mol_{i}"
                yield name, parts[0]


def apply_reaction(rxn, mol):
    """Run a reaction on one mol, returning sanitized, deduped product Mols."""
    seen = set()
    products = []
    for product_set in rxn.RunReactants((mol,)):
        for product in product_set:
            try:
                Chem.SanitizeMol(product)
            except (Chem.AtomValenceException, Chem.KekulizeException, ValueError):
                continue
            smi = Chem.MolToSmiles(product)
            if smi not in seen:
                seen.add(smi)
                products.append(product)
    return products


def embed_3d(mol, num_confs=1, seed=42):
    """Add Hs, embed conformers with ETKDG, optimize with MMFF (UFF fallback)."""
    molh = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    conf_ids = list(AllChem.EmbedMultipleConfs(molh, numConfs=num_confs, params=params))
    if not conf_ids:
        return None, []

    energies = []
    use_mmff = AllChem.MMFFHasAllMoleculeParams(molh)
    for cid in conf_ids:
        if use_mmff:
            props = AllChem.MMFFGetMoleculeProperties(molh)
            ff = AllChem.MMFFGetMoleculeForceField(molh, props, confId=cid)
        else:
            ff = AllChem.UFFGetMoleculeForceField(molh, confId=cid)
        ff.Minimize()
        energies.append((cid, ff.CalcEnergy()))

    energies.sort(key=lambda x: x[1])
    return molh, energies


def main():
    parser = argparse.ArgumentParser(description="Apply a reaction SMARTS and embed 3D products.")
    parser.add_argument("input", help="Input .smi/.sdf file, or '-' for stdin")
    parser.add_argument("--reaction", required=True, help="Reaction SMARTS (reactants>>products)")
    parser.add_argument("--num-confs", type=int, default=1, help="Conformers per product (default: 1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for embedding")
    parser.add_argument("--output", help="Output SDF (best conformer per product); default: print SMILES")
    args = parser.parse_args()

    rxn = AllChem.ReactionFromSmarts(args.reaction)
    if rxn is None:
        print(f"Error: could not parse reaction SMARTS: {args.reaction}")
        sys.exit(1)
    rxn.Initialize()

    writer = Chem.SDWriter(args.output) if args.output else None
    n_in = n_products = n_embedded = 0

    for name, smi in read_smiles(args.input):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"Warning: failed to parse {name}: {smi}", file=sys.stderr)
            continue
        n_in += 1

        for j, product in enumerate(apply_reaction(rxn, mol)):
            n_products += 1
            prod_smi = Chem.MolToSmiles(product)

            if writer is None:
                print(f"{prod_smi}\t{name}_p{j}")
                continue

            molh, energies = embed_3d(product, args.num_confs, args.seed)
            if not energies:
                print(f"Warning: embedding failed for {name}_p{j}: {prod_smi}", file=sys.stderr)
                continue
            n_embedded += 1
            best_cid, best_e = energies[0]
            molh.SetProp("_Name", f"{name}_p{j}")
            molh.SetProp("SMILES", prod_smi)
            molh.SetProp("MMFF_or_UFF_energy", f"{best_e:.3f}")
            writer.write(molh, confId=best_cid)

    if writer is not None:
        writer.close()

    print(
        f"\nProcessed {n_in} input mol(s) -> {n_products} product(s)"
        + (f", embedded {n_embedded} to '{args.output}'" if writer is not None else ""),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
