"""Tests for qmagent.utils.file_ops: XYZ/mol2 parsing and file writers."""

import numpy as np
import pytest
from rdkit import Chem

from qmagent.utils.file_ops import (
    XYZContents,
    mol2_to_sdf,
    write_charge_file,
    write_mol2,
    write_xyz,
)


# --------------------------------------------------------------------------- #
# XYZContents.from_xyz
# --------------------------------------------------------------------------- #

def test_from_xyz_parses_elements_coords_and_comment(tmp_path):
    xyz = tmp_path / "h2o.xyz"
    xyz.write_text(
        "3\n"
        "a water molecule\n"
        "O   0.00000  0.00000  0.00000\n"
        "H   0.75700  0.58600  0.00000\n"
        "H  -0.75700  0.58600  0.00000\n"
    )

    contents = XYZContents.from_xyz(xyz)

    assert contents.elements == ["O", "H", "H"]
    assert contents.comment == "a water molecule"
    assert contents.coords.shape == (3, 3)
    np.testing.assert_allclose(contents.coords[1], [0.757, 0.586, 0.0])


def test_from_xyz_ignores_extra_columns(tmp_path):
    # Some XYZ variants carry trailing data after x/y/z; only cols 1..4 are read.
    xyz = tmp_path / "extra.xyz"
    xyz.write_text("1\ncomment\nC 1.0 2.0 3.0 99.0 extra\n")

    contents = XYZContents.from_xyz(xyz)

    assert contents.elements == ["C"]
    np.testing.assert_allclose(contents.coords, [[1.0, 2.0, 3.0]])


def test_xyz_write_then_read_roundtrip(tmp_path):
    original = XYZContents(
        elements=["C", "O"],
        coords=np.array([[0.0, 0.0, 0.0], [1.16, 0.0, 0.0]]),
        comment="carbon monoxide",
    )
    path = tmp_path / "co.xyz"

    write_xyz(path, original)
    reloaded = XYZContents.from_xyz(path)

    assert reloaded.elements == original.elements
    np.testing.assert_allclose(reloaded.coords, original.coords, atol=1e-8)


def test_write_xyz_header_has_atom_count_and_comment(tmp_path):
    contents = XYZContents(
        elements=["H", "H"],
        coords=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
        comment="dihydrogen",
    )
    path = tmp_path / "h2.xyz"

    write_xyz(path, contents)

    lines = path.read_text().splitlines()
    assert lines[0] == "2"
    assert lines[1] == "dihydrogen"
    assert len(lines) == 4  # 2 header + 2 atoms


# --------------------------------------------------------------------------- #
# XYZContents.from_mol2
# --------------------------------------------------------------------------- #

def test_from_mol2_strips_sybyl_subtype(water_mol2):
    contents = XYZContents.from_mol2(water_mol2)

    # 'O.3' -> 'O', bare 'H' -> 'H'
    assert contents.elements == ["O", "H", "H"]
    assert contents.coords.shape == (3, 3)
    np.testing.assert_allclose(contents.coords[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(contents.coords[1], [0.757, 0.586, 0.0])


def test_from_mol2_only_reads_atom_block(water_mol2):
    # The BOND records must not leak into the parsed atoms.
    contents = XYZContents.from_mol2(water_mol2)
    assert len(contents.elements) == 3


# --------------------------------------------------------------------------- #
# write_charge_file
# --------------------------------------------------------------------------- #

def test_write_charge_file_wraps_at_eight_per_line(tmp_path):
    charges = np.arange(1, 11, dtype=float) / 10.0  # 10 values
    path = tmp_path / "charges.dat"

    write_charge_file(charges, path)

    lines = path.read_text().splitlines()
    assert len(lines) == 2  # 8 + 2
    # Each value occupies the fixed 10-char field.
    assert len(lines[0]) == 80
    assert len(lines[1]) == 20


def test_write_charge_file_values_are_recoverable(tmp_path):
    charges = np.array([-0.834, 0.417, 0.417])
    path = tmp_path / "q.dat"

    write_charge_file(charges, path)

    parsed = [float(tok) for tok in path.read_text().split()]
    np.testing.assert_allclose(parsed, charges, atol=1e-6)


def test_write_charge_file_accepts_2d_input(tmp_path):
    # ravel() flattens column vectors written by the fitter.
    charges = np.array([[0.1], [0.2], [0.3]])
    path = tmp_path / "q2d.dat"

    write_charge_file(charges, path)

    parsed = [float(tok) for tok in path.read_text().split()]
    assert parsed == [0.1, 0.2, 0.3]


# --------------------------------------------------------------------------- #
# write_mol2 / mol2_to_sdf
# --------------------------------------------------------------------------- #

def test_write_mol2_is_readable_by_rdkit(ethane_mol2):
    mol = Chem.MolFromMol2File(str(ethane_mol2), removeHs=False, sanitize=True)
    assert mol is not None
    assert mol.GetNumAtoms() == 8  # C2H6


def test_write_mol2_has_required_sections(ethane_mol2):
    text = ethane_mol2.read_text()
    assert "@<TRIPOS>MOLECULE" in text
    assert "@<TRIPOS>ATOM" in text
    assert "@<TRIPOS>BOND" in text


def test_mol2_to_sdf_roundtrips_atom_count(tmp_path, ethane_mol2):
    sdf = tmp_path / "ethane.sdf"

    mol2_to_sdf(ethane_mol2, sdf)

    assert sdf.exists()
    mol = Chem.SDMolSupplier(str(sdf), removeHs=False)[0]
    assert mol is not None
    assert mol.GetNumAtoms() == 8


def test_mol2_to_sdf_raises_on_unparseable_input(tmp_path):
    bad = tmp_path / "bad.mol2"
    bad.write_text("this is not a mol2 file\n")
    sdf = tmp_path / "out.sdf"

    with pytest.raises(ValueError):
        mol2_to_sdf(bad, sdf)
