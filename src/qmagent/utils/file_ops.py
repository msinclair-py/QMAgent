import numpy as np
from pathlib import Path
from pydantic import BaseModel, BeforeValidator, ConfigDict, PlainSerializer
from rdkit import Chem
from typing import Annotated, Any, Self


def _coerce_ndarray(value: Any) -> np.ndarray:
    """Accept lists (e.g. from JSON) as well as arrays for ndarray fields."""
    return value if isinstance(value, np.ndarray) else np.asarray(value)


# A numpy array field that round-trips through JSON: lists are coerced to arrays
# on validation and arrays are emitted as (nested) lists on serialization. This
# lets models with array fields both ``save`` and ``load`` cleanly.
NDArray = Annotated[
    np.ndarray,
    BeforeValidator(_coerce_ndarray),
    PlainSerializer(lambda v: np.asarray(v).tolist(), return_type=list),
]


class XYZContents(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    elements: list[str]
    coords: NDArray
    comment: str=''

    @classmethod
    def from_xyz(cls,
                 path: Path,
                 grid_level: int=5) -> Self:
        """Helper function which reads in an XYZ file and returns an XYZContents
        object which contains the index-ordered list of elements, coordinates, grid
        definition and any relevant comments.
    
        Arguments:
            path (Path): Path to the xyz file
            grid_level (int): Defaults to 5. PySCF grid level (equivalent to 
                ultrafine in Gaussian)
    
        Returns:
            (XYZContents): The contents of the xyz file.
        """
        with open(path) as f:
            natom = int(f.readline().strip())
            comment = f.readline().strip()
            elements = []
            coords = []
            for _ in range(natom):
                parts = f.readline().split()
                elements.append(parts[0])
                coords.append([float(x) for x in parts[1:4]])
                
            return XYZContents(elements=elements, coords=coords, comment=comment)

    @classmethod
    def from_mol2(cls,
                  path: Path,
                  grid_level: int=5) -> Self:
        """Helper function which reads in a tripos mol2 file and returns an XYZContents
        object.

        Arguments:
            path (Path): Path to the mol2 file.
            grid_level (int): Defaults to 5. PySCF grid level (equivalent to 
                ultrafine in Gaussian)

        Returns:
            (XYZContents): Contents for an xyz file
        """
        symbols, coords = [], []
        with open(path) as f:
            in_atom = False
            for line in f:
                if line.startswith('@<TRIPOS>'):
                    in_atom = line.startswith('@<TRIPOS>ATOM')
                    continue
                if in_atom and line.strip():
                    parts = line.split()
                    # cols: id name x y z sybyl_type subst_id subst_name charge
                    sybyl = parts[5]
                    element = sybyl.split('.')[0] # 'C.3' -> 'C', 'N.am' -> 'N'
                    symbols.append(element)
                    coords.append([float(parts[2]), float(parts[3]), float(parts[4])])

        coords = np.array(coords)

        return XYZContents(elements=symbols, coords=coords, comment='')

def write_xyz(path: Path,
              contents: XYZContents) -> None:
    """Helper function to write an XYZ file from an XYZContents object.

    Arguments:
        path (Path): Path to xyz file to be written
        contents (XYZContents): Object containing data for xyz file.

    Returns:
        None
    """
    with open(path, 'w') as f:
        f.write(f'{len(contents.elements)}\n{contents.comment}\n')
        for elem, (x, y, z) in zip(contents.elements, contents.coords, strict=True):
            f.write(f'{elem:2s}  {x:14.8f}  {y:14.8f}  {z:14.8f}\n')

def write_mol2(mol: Chem.Mol,
               path: Path,
               resname: str) -> None:
    """Helper function for writing mol2 files from rdkit Mol objects.

    Arguments:
        mol (Chem.Mol): rdkit Mol object
        path (Path): Path to the mol2 file to be written
        resname (str): Name of the residue to be written
    """
    conf = mol.GetConformer()
    atoms = mol.GetAtoms()

    lines = [
        '@<TRIPOS>MOLECULE',
        resname,
        f'{mol.GetNumAtoms()} {mol.GetNumBonds()} 1 0 0',
        'SMALL', 'USER_CHARGES', '',
        '@<TRIPOS>ATOM'
    ]

    for i, a in enumerate(atoms, 1):
        p = conf.GetAtomPosition(a.GetIdx())
        name = a.GetPDBResidueInfo().GetName().strip() if a.GetPDBResidueInfo() else f'{a.GetSymbol()}{i}'
        # Sybyl type: start with element; antechamber will retype with -at gaff2 anyway
        sybyl = a.GetSymbol()
        q = a.GetDoubleProp('_TriposPartialCharge') if a.HasProp('_TriposPartialCharge') else 0.0
        lines.append(
            f'{i:>7} {name:<8} {p.x:>9.4f} {p.y:>9.4f} {p.z:>9.4f} '
            f'{sybyl:<6} 1 {resname:<4} {q:>9.4f}'
        )

    lines.append('@<TRIPOS>BOND')

    for j, b in enumerate(mol.GetBonds(), 1):
        bt = {1.0: '1', 2.0: '2', 3.0: '3', 1.5: 'ar'}.get(b.GetBondTypeAsDouble(), '1')
        lines.append(f'{j:>6} {b.GetBeginAtomIdx() + 1:>5} {b.GetEndAtomIdx()+1:>5} {bt:>4}')

    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

def write_charge_file(charges: np.ndarray,
                      path: Path) -> None:
    """Write partial charges in antechamber's free-format -cf layout (8 per line).

    antechamber reads pre-computed charges from this file via the -cf flag (with
    -c rc), so it only assigns atom types and keeps our charges untouched.

    Arguments:
        charges (np.ndarray): Partial charges in atom (mol2) order.
        path (Path): Path to the charge file to be written.
    """
    vals = [f'{q:10.6f}' for q in np.asarray(charges).ravel()]
    lines = [''.join(vals[i:i + 8]) for i in range(0, len(vals), 8)]
    path.write_text('\n'.join(lines) + '\n')

def mol2_to_sdf(mol2: Path,
                sdf: Path) -> None:
    """Convert a Tripos mol2 file to sdf, preserving atom order.

    Used to produce antechamber's sdf input from the build geometry; preserving
    atom order keeps it consistent with the RESP charge file written for the same
    molecule.

    Arguments:
        mol2 (Path): Path to the input mol2 file.
        sdf (Path): Path to the sdf file to be written.
    """
    mol = Chem.MolFromMol2File(str(mol2), removeHs=False, sanitize=True)
    if mol is None:
        raise ValueError(f'RDKit could not parse "{mol2}" into a molecule for SDF export.')
    writer = Chem.SDWriter(str(sdf))
    writer.write(mol)
    writer.close()
