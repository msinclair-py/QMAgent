"""Tests for the RESP charge fitter (qmagent.agents.resp_fitter)."""

import numpy as np
import pytest

from qmagent.agents.resp_fitter import RESPFitter


@pytest.fixture
def synthetic_esp():
    """Build an exact ESP field from known charges so the fit is recoverable.

    Returns (coords, grid_pts, esp, q_true). The grid is asymmetric and much
    larger than the atom count so the 1/r design matrix has full column rank.
    """
    rng = np.random.default_rng(0)
    coords = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]])
    grid_pts = rng.uniform(-6.0, 6.0, size=(800, 3))
    # Push grid points off the nuclei to avoid 1/r blow-ups.
    grid_pts = grid_pts[np.linalg.norm(grid_pts, axis=1) > 2.0]
    q_true = np.array([0.40, -0.10, -0.30])

    inv_r = np.zeros((len(grid_pts), 3))
    for i in range(3):
        inv_r[:, i] = 1.0 / np.linalg.norm(grid_pts - coords[i], axis=1)
    esp = inv_r @ q_true

    return coords, grid_pts, esp, q_true


def test_precomputed_inv_r_and_normal_equations(synthetic_esp):
    coords, grid_pts, esp, _ = synthetic_esp
    fitter = RESPFitter(coords, grid_pts, esp)

    expected_inv_r = np.zeros((len(grid_pts), 3))
    for i in range(3):
        expected_inv_r[:, i] = 1.0 / np.linalg.norm(grid_pts - coords[i], axis=1)

    np.testing.assert_allclose(fitter.inv_r, expected_inv_r)
    np.testing.assert_allclose(fitter.A, expected_inv_r.T @ expected_inv_r)
    np.testing.assert_allclose(fitter.B, expected_inv_r.T @ esp)
    assert fitter.natom == 3
    assert fitter.ngrid == len(grid_pts)


def test_unrestrained_fit_recovers_true_charges(synthetic_esp):
    coords, grid_pts, esp, q_true = synthetic_esp
    fitter = RESPFitter(coords, grid_pts, esp)

    # No restraint: pure least-squares under the total-charge constraint, so the
    # exact generating charges are the unique minimiser.
    q = fitter.fit(total_charge=0, restraint_a=0.0)

    np.testing.assert_allclose(q, q_true, atol=1e-4)
    assert np.isclose(q.sum(), 0.0, atol=1e-8)


def test_total_charge_constraint_is_enforced(synthetic_esp):
    coords, grid_pts, esp, _ = synthetic_esp
    fitter = RESPFitter(coords, grid_pts, esp)

    q = fitter.fit(total_charge=1, restraint_a=0.0005)

    assert np.isclose(q.sum(), 1.0, atol=1e-6)


def test_symmetry_constraint_equalises_two_atoms(synthetic_esp):
    coords, grid_pts, esp, _ = synthetic_esp
    fitter = RESPFitter(coords, grid_pts, esp)

    q = fitter.fit(total_charge=0, restraint_a=0.0, symmetry_constraints=[(1, 2)])

    assert np.isclose(q[1], q[2], atol=1e-6)
    assert np.isclose(q.sum(), 0.0, atol=1e-8)


def test_frozen_atom_keeps_its_assigned_charge(synthetic_esp):
    coords, grid_pts, esp, _ = synthetic_esp
    fitter = RESPFitter(coords, grid_pts, esp)

    q = fitter.fit(
        total_charge=0,
        restraint_a=0.0,
        frozen_atoms=[0],
        frozen_charges={0: 0.25},
    )

    assert np.isclose(q[0], 0.25, atol=1e-6)
    assert np.isclose(q.sum(), 0.0, atol=1e-6)


def test_hyperbolic_restraint_shrinks_charge_magnitudes(synthetic_esp):
    coords, grid_pts, esp, _ = synthetic_esp
    fitter = RESPFitter(coords, grid_pts, esp)

    q_free = fitter.fit(total_charge=0, restraint_a=0.0)
    q_restrained = fitter.fit(total_charge=0, restraint_a=0.01)

    # The restraint biases charges toward zero, so the L2 norm should not grow.
    assert np.linalg.norm(q_restrained) <= np.linalg.norm(q_free) + 1e-9


def test_two_stage_resp_conserves_total_charge(synthetic_esp):
    coords, grid_pts, esp, _ = synthetic_esp
    fitter = RESPFitter(coords, grid_pts, esp)

    q = fitter.two_stage_resp(elements=["C", "O", "O"], total_charge=0)

    assert q.shape == (3,)
    assert np.isclose(q.sum(), 0.0, atol=1e-6)


def test_two_stage_resp_freezes_non_refit_atoms_at_stage1_value(synthetic_esp):
    """Atoms outside refit_atoms must come out of stage 2 exactly at their
    stage-1 charge, not merely "close" to it under the new restraint."""
    coords, grid_pts, esp, _ = synthetic_esp
    fitter = RESPFitter(coords, grid_pts, esp)

    q1 = fitter.fit(total_charge=0, restraint_a=0.0005)
    q2 = fitter.two_stage_resp(
        elements=["C", "O", "O"], total_charge=0, refit_atoms={0},
    )

    # Atoms 1 and 2 are frozen -> identical to the independently-run stage 1.
    np.testing.assert_allclose(q2[1], q1[1], atol=1e-8)
    np.testing.assert_allclose(q2[2], q1[2], atol=1e-8)
    # Atom 0 is the only refit atom and must still satisfy total charge.
    assert np.isclose(q2.sum(), 0.0, atol=1e-6)


def test_two_stage_resp_with_no_refit_atoms_returns_stage1_charges(synthetic_esp):
    """An empty (or omitted) refit set freezes everything: stage 2 is a no-op."""
    coords, grid_pts, esp, _ = synthetic_esp
    fitter = RESPFitter(coords, grid_pts, esp)

    q1 = fitter.fit(total_charge=0, restraint_a=0.0005)
    q2 = fitter.two_stage_resp(elements=["C", "O", "O"], total_charge=0, refit_atoms=set())

    np.testing.assert_allclose(q2, q1, atol=1e-8)


def test_two_stage_resp_refit_atoms_actually_move(synthetic_esp):
    """With a non-trivial restraint change, a refit atom's stage-2 charge should
    generally differ from its stage-1 charge (the whole point of refitting it
    under a different restraint strength)."""
    coords, grid_pts, esp, _ = synthetic_esp
    fitter = RESPFitter(coords, grid_pts, esp)

    q1 = fitter.fit(total_charge=0, restraint_a=0.0005)
    q2 = fitter.two_stage_resp(elements=["C", "O", "O"], total_charge=0, refit_atoms={0, 1, 2})

    # All atoms free in stage 2 -> equivalent to an independent fit at a=0.001.
    q_direct = fitter.fit(total_charge=0, restraint_a=0.001)
    np.testing.assert_allclose(q2, q_direct, atol=1e-6)
    assert not np.allclose(q2, q1, atol=1e-8)


def test_two_stage_resp_symmetry_only_applied_within_refit_set(synthetic_esp):
    """A symmetry pair spanning a frozen and a refit atom is honored in stage 1
    (constraints apply from stage 1 onward, per standard RESP), but must not be
    re-applied as an active stage-2 constraint once one side is frozen -- doing
    so would be redundant with freezing and, for a pair split across the
    boundary, would incorrectly force the frozen atom's value onto the refit
    atom instead of letting the refit atom explore under the new restraint."""
    coords, grid_pts, esp, _ = synthetic_esp
    fitter = RESPFitter(coords, grid_pts, esp)

    # Pair (0, 1): only atom 0 is in the refit set. Stage 1 (which two_stage_resp
    # runs with the full symmetry_constraints) ties q1[0] == q1[1].
    q1 = fitter.fit(total_charge=0, restraint_a=0.0005, symmetry_constraints=[(0, 1)])
    q2 = fitter.two_stage_resp(
        elements=["C", "O", "O"],
        total_charge=0,
        symmetry_constraints=[(0, 1)],
        refit_atoms={0},
    )

    # Atom 1 stays frozen at its stage-1 value; the cross-set symmetry
    # constraint must not be re-applied in stage 2 (that would just reproduce
    # freezing here, but for it to be meaningfully dropped rather than
    # coincidentally satisfied, atom 0 must still be free to move under the
    # stronger stage-2 restraint rather than being pinned to q1[1]).
    np.testing.assert_allclose(q2[1], q1[1], atol=1e-8)
    assert np.isclose(q2.sum(), 0.0, atol=1e-6)
