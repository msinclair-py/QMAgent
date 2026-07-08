#!/usr/bin/env python
"""Evaluate the molecular electrostatic potential (ESP) on a set of grid points.

    V(r) = sum_A Z_A / |r - R_A|  -  integral rho(r') / |r - r'| dr'

The electronic term is built from the SCF density matrix and 3-center Coulomb
integrals against point charges placed at the grid locations (fakemol). This is
the QM half of a RESP charge-fitting workflow.

Usage as a library:

    from compute_esp import compute_esp
    esp = compute_esp(mf, grid_points_angstrom)   # ndarray (npts,), Hartree/e

Usage as a script (single-point ESP on an auto-generated MK grid):

    python compute_esp.py molecule.xyz --basis def2-tzvp --xc wb97x-d3bj
"""
from __future__ import annotations

import argparse
import numpy as np

BOHR_PER_ANGSTROM = 1.8897259886


def compute_esp(mf, grid_points_angstrom: np.ndarray, *, batch_size: int = 500) -> np.ndarray:
    """Compute the ESP at `grid_points_angstrom` from a converged mean-field `mf`.

    Arguments:
        mf: A converged PySCF (or gpu4pyscf) mean-field object. `mf.kernel()`
            must already have been run.
        grid_points_angstrom (np.ndarray): (npts, 3) evaluation points in Angstrom.
        batch_size (int): Grid points per integral batch (memory vs. speed).

    Returns:
        (np.ndarray): ESP values (npts,) in Hartree/e (atomic units).
    """
    from pyscf import df, gto

    mol = mf.mol
    dm = mf.make_rdm1()
    # gpu4pyscf may hand back a CuPy array; bring it to NumPy for the einsum below.
    if hasattr(dm, "get"):
        dm = dm.get()

    grid_bohr = np.asarray(grid_points_angstrom, dtype=float) * BOHR_PER_ANGSTROM
    coords_bohr = mol.atom_coords()              # PySCF returns Bohr by default
    npts = len(grid_bohr)

    # Nuclear contribution: sum_A Z_A / |r - R_A|
    Z = mol.atom_charges()
    esp_nuc = np.zeros(npts)
    for A in range(mol.natm):
        r = np.linalg.norm(grid_bohr - coords_bohr[A], axis=1)
        esp_nuc += Z[A] / r

    # Electronic contribution: -<mu| 1/|r-R| |nu> contracted with the density.
    esp_elec = np.zeros(npts)
    for start in range(0, npts, batch_size):
        pts = grid_bohr[start:start + batch_size]
        fakemol = gto.fakemol_for_charges(pts)
        ints = df.incore.aux_e2(mol, fakemol, intor="int3c2e")   # (nao, nao, nbatch)
        for k in range(ints.shape[2]):
            esp_elec[start + k] = -np.einsum("ij,ij->", dm, ints[:, :, k])

    return esp_nuc + esp_elec


def mk_grid(elements, coords_angstrom, density: float = 1.0) -> np.ndarray:
    """Merz-Kollman ESP grid: nested Connolly shells at 1.4/1.6/1.8/2.0x vdW radius,
    excluding points that fall inside a neighbor's innermost shell.

    Arguments:
        elements (list[str]): Element symbols in index order.
        coords_angstrom (np.ndarray): (natm, 3) coordinates in Angstrom.
        density (float): Scales point count per atom.

    Returns:
        (np.ndarray): (npts, 3) grid points in Angstrom.
    """
    from collections import defaultdict

    common_radii = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52,
                    "S": 1.80, "F": 1.47, "Cl": 1.75, "Br": 1.85}
    vdw = defaultdict(lambda: 1.70, common_radii)
    golden = (1 + np.sqrt(5)) / 2
    shells = [1.4, 1.6, 1.8, 2.0]
    coords = np.asarray(coords_angstrom, dtype=float)

    out = []
    for factor in shells:
        for i, (elem, center) in enumerate(zip(elements, coords, strict=True)):
            radius = vdw[elem] * factor
            npoints = max(int(4.0 * np.pi * radius ** 2 * density), 50)
            idx = np.arange(npoints)
            theta = 2 * np.pi * idx / golden
            phi = np.arccos(1 - 2 * (idx + 0.5) / npoints)
            pts = center + radius * np.column_stack(
                [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]
            )
            keep = np.ones(len(pts), dtype=bool)
            for j, center_j in enumerate(coords):
                if j == i:
                    continue
                r_excl = vdw[elem] * shells[0]
                keep &= np.linalg.norm(pts - center_j, axis=1) > r_excl
            out.append(pts[keep])
    return np.vstack(out)


def _read_xyz(path):
    with open(path) as f:
        n = int(f.readline())
        f.readline()  # comment
        elements, coords = [], []
        for _ in range(n):
            parts = f.readline().split()
            elements.append(parts[0])
            coords.append([float(x) for x in parts[1:4]])
    return elements, np.array(coords)


def main():
    ap = argparse.ArgumentParser(description="Compute ESP on an MK grid for an XYZ file.")
    ap.add_argument("xyz")
    ap.add_argument("--basis", default="def2-tzvp")
    ap.add_argument("--xc", default="wb97x-d3bj")
    ap.add_argument("--charge", type=int, default=0)
    ap.add_argument("--spin", type=int, default=0, help="2S = multiplicity - 1")
    ap.add_argument("--gpu", action="store_true", help="use gpu4pyscf for SCF")
    args = ap.parse_args()

    from pyscf import gto
    if args.gpu:
        from gpu4pyscf import dft
    else:
        from pyscf import dft

    elements, coords = _read_xyz(args.xyz)
    geom = "\n".join(f"{e} {x:.8f} {y:.8f} {z:.8f}" for e, (x, y, z) in zip(elements, coords))
    mol = gto.M(atom=geom, basis=args.basis, charge=args.charge, spin=args.spin, verbose=3)

    mf = dft.RKS(mol)
    mf.xc = args.xc
    mf.kernel()
    if not mf.converged:
        raise SystemExit("SCF did not converge")

    grid = mk_grid(elements, coords)
    esp = compute_esp(mf, grid)
    print(f"grid points: {len(grid)}   ESP range: [{esp.min():.4f}, {esp.max():.4f}] Hartree/e")


if __name__ == "__main__":
    main()
