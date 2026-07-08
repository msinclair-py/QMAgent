#!/usr/bin/env python
"""Build a molecule from an XYZ file and run a DFT single point or geometry
optimization, printing the total energy in Hartree.

    python run_dft.py molecule.xyz                       # single point
    python run_dft.py molecule.xyz --opt                 # optimize geometry
    python run_dft.py molecule.xyz --opt --gpu \
        --xc wb97x-d3bj --basis def2-svp --grid 99 590

Geometry optimization requires the optional `geometric` package.

Key reminders this script encodes:
  * spin = 2S = multiplicity - 1
  * symmetry=False for optimizations (a relaxing geometry can break symmetry)
  * optimize() returns a NEW Mole, not an energy -> rebuild the mean-field
  * always check mf.converged
"""
from __future__ import annotations

import argparse
import numpy as np


def read_xyz(path):
    with open(path) as f:
        n = int(f.readline())
        f.readline()
        elements, coords = [], []
        for _ in range(n):
            parts = f.readline().split()
            elements.append(parts[0])
            coords.append([float(x) for x in parts[1:4]])
    return elements, np.array(coords)


def geom_string(elements, coords):
    return "\n".join(
        f"{e}  {c[0]:.8f}  {c[1]:.8f}  {c[2]:.8f}"
        for e, c in zip(elements, coords, strict=True)
    )


def build_mf(geom, *, basis, xc, disp, grid, charge, spin, gpu, max_memory, symmetry):
    from pyscf import gto
    if gpu:
        from gpu4pyscf import dft
    else:
        from pyscf import dft

    mol = gto.M(atom=geom, basis=basis, charge=charge, spin=spin,
                symmetry=symmetry, max_memory=max_memory, verbose=4)
    mf = dft.RKS(mol)
    mf.xc = xc
    if disp:
        mf.disp = disp
    mf.grids.atom_grid = grid
    return mf


def main():
    ap = argparse.ArgumentParser(description="DFT single point or optimization for an XYZ file.")
    ap.add_argument("xyz")
    ap.add_argument("--basis", default="def2-tzvp")
    ap.add_argument("--xc", default="wb97x-d3bj")
    ap.add_argument("--disp", default=None, help="e.g. d3bj; omit if functional includes dispersion")
    ap.add_argument("--grid", nargs=2, type=int, default=[99, 590], metavar=("NRAD", "NANG"))
    ap.add_argument("--charge", type=int, default=0)
    ap.add_argument("--spin", type=int, default=0, help="2S = multiplicity - 1")
    ap.add_argument("--opt", action="store_true", help="optimize geometry (needs `geometric`)")
    ap.add_argument("--maxsteps", type=int, default=200)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--max-memory", type=int, default=4000)
    args = ap.parse_args()

    elements, coords = read_xyz(args.xyz)
    grid = tuple(args.grid)

    mf = build_mf(
        geom_string(elements, coords),
        basis=args.basis, xc=args.xc, disp=args.disp, grid=grid,
        charge=args.charge, spin=args.spin, gpu=args.gpu,
        max_memory=args.max_memory,
        symmetry=not args.opt,   # symmetry off for optimization
    )

    if args.opt:
        from pyscf.geomopt.geometric_solver import optimize
        mol_eq = optimize(mf, maxsteps=args.maxsteps)
        opt_coords = mol_eq.atom_coords(unit="Angstrom")
        # optimize() returns a Mole, not an energy -> rebuild mf for the final energy
        mf = build_mf(
            geom_string(elements, opt_coords),
            basis=args.basis, xc=args.xc, disp=args.disp, grid=grid,
            charge=args.charge, spin=args.spin, gpu=args.gpu,
            max_memory=args.max_memory, symmetry=False,
        )

    e_tot = mf.kernel()
    if not mf.converged:
        mf = mf.newton()
        e_tot = mf.kernel()
    status = "converged" if mf.converged else "NOT CONVERGED"
    print(f"E_tot = {float(e_tot):.8f} Hartree  ({status})")


if __name__ == "__main__":
    main()
