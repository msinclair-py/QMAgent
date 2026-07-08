"""Tests for the pure/static helpers on qmagent.agents.qm_agent.QMAgent.

The agent's @action coroutines drive PySCF/AmberTools/parsl and are out of
scope here; these target the deterministic helpers that need no runtime.
"""

import numpy as np
import pytest
from pathlib import Path

from qmagent.agents.qm_agent import QMAgent
from qmagent.utils.pydantic_models import TorsionFitResult


# --------------------------------------------------------------------------- #
# formulate_geometry_string
# --------------------------------------------------------------------------- #

def test_formulate_geometry_string_format():
    s = QMAgent.formulate_geometry_string(
        ["O", "H"], np.array([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0]])
    )
    lines = s.splitlines()
    assert len(lines) == 2
    assert lines[0].split() == ["O", "0.00000000", "0.00000000", "0.00000000"]
    assert lines[1].startswith("H")
    assert "0.95720000" in lines[1]


def test_formulate_geometry_string_length_mismatch_raises():
    # zip(..., strict=True) guards against ragged element/coord inputs.
    with pytest.raises(ValueError):
        QMAgent.formulate_geometry_string(["O", "H"], np.array([[0.0, 0.0, 0.0]]))


# --------------------------------------------------------------------------- #
# generate_mk_grid
# --------------------------------------------------------------------------- #

def test_mk_grid_single_atom_sits_on_expected_shells():
    grid = QMAgent.generate_mk_grid(["H"], np.array([[0.0, 0.0, 0.0]]))

    assert grid.ndim == 2 and grid.shape[1] == 3
    radii = np.linalg.norm(grid, axis=1)
    # H vdW radius 1.20 scaled by the four Connolly shell factors.
    expected = np.array([1.20 * f for f in (1.4, 1.6, 1.8, 2.0)])
    np.testing.assert_allclose(np.unique(np.round(radii, 6)), expected)


def test_mk_grid_density_increases_point_count():
    elements = ["C", "O"]
    coords = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])

    sparse = QMAgent.generate_mk_grid(elements, coords, density=1.0)
    dense = QMAgent.generate_mk_grid(elements, coords, density=5.0)

    assert dense.shape[0] > sparse.shape[0]
    assert sparse.shape[1] == 3


def test_mk_grid_mismatched_inputs_raise():
    with pytest.raises(ValueError):
        QMAgent.generate_mk_grid(["C", "O"], np.array([[0.0, 0.0, 0.0]]))


# --------------------------------------------------------------------------- #
# find_symmetry_pairs
# --------------------------------------------------------------------------- #

def test_find_symmetry_pairs_returns_valid_equivalent_atoms(methanol_mol2):
    from rdkit import Chem

    pairs = QMAgent.find_symmetry_pairs(methanol_mol2)
    mol = Chem.MolFromMol2File(str(methanol_mol2), removeHs=False, sanitize=True)
    n = mol.GetNumAtoms()

    # Methanol's three methyl hydrogens are symmetry-equivalent, so the
    # automorphism is non-trivial and must yield at least one pair.
    assert pairs, "expected symmetry-equivalent atoms in methanol"
    for i, j in pairs:
        assert i < j
        assert 0 <= i < n and 0 <= j < n
        # Equivalent atoms must be the same element.
        assert mol.GetAtomWithIdx(i).GetSymbol() == mol.GetAtomWithIdx(j).GetSymbol()


def test_find_symmetry_pairs_excludes_identity(ethane_mol2):
    # The identity automorphism must be skipped; any returned pair must move an
    # atom (i != j is guaranteed by the i < j filter, so just assert distinctness).
    pairs = QMAgent.find_symmetry_pairs(ethane_mol2)
    assert all(i != j for i, j in pairs)


# --------------------------------------------------------------------------- #
# merge_frcmods
# --------------------------------------------------------------------------- #

FRCMOD_TEMPLATE = """\
remark goes here
MASS

BOND

ANGLE

DIHE
{dihe_line}
IMPROPER

NONBON
"""


def _write_frcmod(path: Path, dihe_line: str) -> None:
    path.write_text(FRCMOD_TEMPLATE.format(dihe_line=dihe_line))


def test_merge_frcmods_collects_dihe_blocks(tmp_path):
    f1 = tmp_path / "t1.frcmod"
    f2 = tmp_path / "t2.frcmod"
    _write_frcmod(f1, "c3-c3-c3-c3   1    0.156   0.000   3.000")
    _write_frcmod(f2, "c3-c3-os-c3   1    0.383 180.000  -2.000")

    fits = [
        TorsionFitResult(torsion=(0, 1, 2, 3), atom_types=("c3", "c3", "c3", "c3"), frcmod_file=f1),
        TorsionFitResult(torsion=(1, 2, 3, 4), atom_types=("c3", "c3", "os", "c3"), frcmod_file=f2),
    ]
    out = tmp_path / "refined.frcmod"

    QMAgent.merge_frcmods(fits, out)

    text = out.read_text()
    assert text.startswith("Refined dihedrals from QM torsion scans")
    assert "DIHE" in text
    assert "c3-c3-c3-c3   1    0.156   0.000   3.000" in text
    assert "c3-c3-os-c3   1    0.383 180.000  -2.000" in text
    # Per-torsion provenance comments use the joined atom types.
    assert "# c3-c3-c3-c3" in text
    assert "# c3-c3-os-c3" in text


def test_merge_frcmods_skips_failed_fits(tmp_path):
    f1 = tmp_path / "ok.frcmod"
    _write_frcmod(f1, "c3-c3-c3-c3   1    0.156   0.000   3.000")

    fits = [
        TorsionFitResult(torsion=(0, 1, 2, 3), atom_types=("c3", "c3", "c3", "c3"), frcmod_file=f1),
        TorsionFitResult(torsion=(1, 2, 3, 4), atom_types=("ca", "ca", "ca", "ca"), frcmod_file=None),
    ]
    out = tmp_path / "refined.frcmod"

    QMAgent.merge_frcmods(fits, out)

    text = out.read_text()
    assert "c3-c3-c3-c3" in text
    assert "# ca-ca-ca-ca" not in text  # the None-frcmod fit contributed nothing


def test_merge_frcmods_with_no_valid_fits_writes_empty_dihe(tmp_path):
    out = tmp_path / "refined.frcmod"

    QMAgent.merge_frcmods(
        [TorsionFitResult(torsion=(0, 1, 2, 3), atom_types=("c3",) * 4, frcmod_file=None)],
        out,
    )

    text = out.read_text()
    assert "DIHE" in text
    assert "#" not in text  # no per-torsion blocks emitted
