import numpy as np
from scipy.optimize import minimize

class RESPFitter:
    """
    Two-stage RESP charge fitting with arbitrary constraints.

    Implements the Bayly et al. (1993) RESP algorithm:
        Stage 1: Fit all atoms with weak hyperbolic restraint (a=0.0005)
        Stage 2: Freeze every atom not in the caller-supplied refit set (see
                 ``two_stage_resp``'s ``refit_atoms``; typically aliphatic
                 CH/CH2/CH3 carbons and their hydrogens) at its stage-1 charge,
                 and refit only the remaining atoms with stronger restraint
                 (a=0.001).

    The ESP fitting functional:
        chi^2 = sum_k [ V_QM(r_k) - sum_i q_i / |r_k - R_i| ]^2
              + a * sum_i [ (q_i^2 + b^2)^0.5 - b ]   (hyperbolic restraint)

    Subject to constraints:
        - Total charge = target (usually 0)
        - Fragment charge sums
        - Symmetry equivalences
    """

    def __init__(self, 
                 coords_bohr: np.ndarray, 
                 grid_points_bohr: np.ndarray, 
                 esp_values_au: np.ndarray):
        """
        Parameters
        ----------
        coords_bohr : ndarray (N, 3)
            Atomic positions in Bohr
        grid_points_bohr : ndarray (M, 3)
            ESP grid points in Bohr
        esp_values_au : ndarray (M,)
            QM ESP values in atomic units (Hartree/e)
        """
        self.coords = coords_bohr
        self.grid_pts = grid_points_bohr
        self.esp = esp_values_au
        self.natom = len(coords_bohr)
        self.ngrid = len(grid_points_bohr)

        # Precompute 1/r matrix: inv_r[k, i] = 1 / |grid_k - atom_i|
        self.inv_r = np.zeros((self.ngrid, self.natom))
        for i in range(self.natom):
            diff = self.grid_pts - self.coords[i]
            self.inv_r[:, i] = 1.0 / np.linalg.norm(diff, axis=1)

        # Precompute A matrix and B vector for the linear system
        # A_ij = sum_k 1/r_ki * 1/r_kj
        # B_i  = sum_k V_k * 1/r_ki

        self.A = self.inv_r.T @ self.inv_r
        self.B = self.inv_r.T @ self.esp

    def fit(
        self,
        total_charge: int=0,
        restraint_a: float=0.0005,
        restraint_b: float=0.1,
        charge_constraints: list[tuple[int, float]] | None=None,
        symmetry_constraints: list[tuple[int, int]] | None=None,
        frozen_atoms: list[int]=None,
        frozen_charges: dict[int, float]=None,
        unrestrained_atoms: set[int] | None=None,
    ):
        """
        Fit RESP charges.

        Arguments:
            total_charge : float
                Total molecular charge
            restraint_a : float
                Hyperbolic restraint strength
            restraint_b : float
                Hyperbolic restraint tightness
            charge_constraints : list of (atom_indices, target_charge)
                Fragment charge sum constraints
            symmetry_constraints : list of (atom_i, atom_j)
                Pairs of atoms that must have equal charges
            frozen_atoms : list of int
                Atom indices with fixed charges
            frozen_charges : dict {atom_idx: charge}
                Fixed charge values for frozen atoms
            unrestrained_atoms : set of int
                Atom indices exempt from the hyperbolic restraint (in addition to
                frozen atoms). Standard RESP (ihfree=1) leaves hydrogens
                unrestrained; ``two_stage_resp`` passes the hydrogens here.

        Returns:
            (np.ndarray): Charge array
        """
        if charge_constraints is None:
            charge_constraints = []
        if symmetry_constraints is None:
            symmetry_constraints = []
        if frozen_atoms is None:
            frozen_atoms = []
        if frozen_charges is None:
            frozen_charges = {}

        # Atoms that carry no hyperbolic restraint: frozen atoms (their charge is
        # fixed) plus the caller-supplied exemptions (hydrogens, per ihfree=1).
        restraint_skip = set(frozen_atoms) | (set(unrestrained_atoms) if unrestrained_atoms else set())

        # Initial guess: distribute charge evenly
        q0 = np.full(self.natom, total_charge / self.natom)
        for idx, charge in frozen_charges.items():
            q0[idx] = charge

        # Build constraint list for scipy
        constraints = []

        # Total charge constraint
        constraints.append({
            'type': 'eq',
            'fun': lambda q: q.sum() - total_charge,
            'jac': lambda q: np.ones(self.natom),
        })

        # Fragment charge constraints
        for atom_indices, target in charge_constraints:
            indices = list(atom_indices)
            def make_frag_constraint(idx, tgt):
                def frag_fun(q):
                    return q[idx].sum() - tgt
                def frag_jac(q):
                    j = np.zeros(self.natom)
                    j[idx] = 1.0
                    return j
                return {'type': 'eq', 'fun': frag_fun, 'jac': frag_jac}
            constraints.append(make_frag_constraint(indices, target))

        # Symmetry constraints
        for i, j in symmetry_constraints:
            def make_sym_constraint(ii, jj):
                def sym_fun(q):
                    return q[ii] - q[jj]
                def sym_jac(q):
                    j = np.zeros(self.natom)
                    j[ii] = 1.0
                    j[jj] = -1.0
                    return j
                return {'type': 'eq', 'fun': sym_fun, 'jac': sym_jac}
            constraints.append(make_sym_constraint(i, j))

        # Frozen atom constraints
        for idx in frozen_atoms:
            if idx in frozen_charges:
                def make_freeze(ii, val):
                    def freeze_fun(q):
                        return q[ii] - val
                    def freeze_jac(q):
                        j = np.zeros(self.natom)
                        j[ii] = 1.0
                        return j
                    return {'type': 'eq', 'fun': freeze_fun, 'jac': freeze_jac}
                constraints.append(make_freeze(idx, frozen_charges[idx]))

        # Objective: chi^2 + restraint
        def objective(q):
            # ESP residual
            esp_calc = self.inv_r @ q
            residual = self.esp - esp_calc
            chi2 = np.dot(residual, residual)

            # Hyperbolic restraint (not on frozen atoms)
            restraint = 0.0
            for i in range(self.natom):
                if i not in restraint_skip:
                    restraint += np.sqrt(q[i]**2 + restraint_b**2) - restraint_b

            return chi2 + restraint_a * restraint

        def gradient(q):
            esp_calc = self.inv_r @ q
            residual = self.esp - esp_calc
            grad = -2.0 * (self.A @ q - self.B)  # d(chi2)/dq = -2*(B - A*q)
            # Actually: grad = 2 * (A @ q - B)
            grad = 2.0 * (self.A @ q - self.B)

            # Restraint gradient
            for i in range(self.natom):
                if i not in restraint_skip:
                    grad[i] += restraint_a * q[i] / np.sqrt(q[i]**2 + restraint_b**2)

            return grad

        result = minimize(
            objective,
            q0,
            jac=gradient,
            method='SLSQP',
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-12},
        )

        if not result.success:
            print(f'  WARNING: RESP optimization did not fully converge: {result.message}')

        return result.x

    def two_stage_resp(
        self,
        elements,
        total_charge=0,
        charge_constraints=None,
        symmetry_constraints=None,
        refit_atoms=None,
    ):
        """
        Full two-stage RESP fitting.

        Stage 1: Fit all atoms, a=0.0005
        Stage 2: Freeze every atom not in ``refit_atoms`` at its stage-1 charge
                 and refit only ``refit_atoms`` with a=0.001.

        Arguments:
            elements : list of str
                Element symbols in atom-index order (kept for interface
                compatibility / future per-element heuristics).
            total_charge : float
                Total molecular charge.
            charge_constraints, symmetry_constraints :
                Passed through to both stages (see ``fit``).
            refit_atoms : set[int] | None
                Atom indices eligible to move in stage 2 (typically aliphatic
                CH/CH2/CH3 carbons and their hydrogens; see
                ``QMAgent.find_resp_refit_atoms``). Every other atom is frozen
                at its stage-1 charge. If None or empty, every atom is frozen
                and stage 2 trivially returns the stage-1 charges.
        """
        # Hydrogens are exempt from the hyperbolic restraint (standard RESP
        # ihfree=1): the restraint targets the poorly-determined heavy-atom
        # charges, while hydrogen charges are left to the ESP fit and symmetry
        # equivalencing. This is the one place ``elements`` is used.
        hydrogens = {
            i for i, el in enumerate(elements)
            if str(el).strip().upper() == 'H'
        }

        # Stage 1
        print('  Stage 1: weak restraint (a=0.0005), all atoms free')
        q1 = self.fit(
            total_charge=total_charge,
            restraint_a=0.0005,
            charge_constraints=charge_constraints,
            symmetry_constraints=symmetry_constraints,
            unrestrained_atoms=hydrogens,
        )

        print(f'  Stage 1 charges: sum = {q1.sum():.6f}')
        print(f'  Stage 1 range: [{q1.min():.4f}, {q1.max():.4f}]')

        # Stage 2: freeze everything except the refit set (aliphatic CH/CH2/CH3
        # carbons and their hydrogens by convention) at its stage-1 charge, and
        # refit only the refit set with a stronger restraint. Frozen atoms are
        # excluded from the symmetry/charge constraints too, since a constraint
        # coupling a frozen and a free atom would either be trivially satisfied
        # (both frozen) or over-determine the free atom.
        refit = set(refit_atoms) if refit_atoms else set()
        frozen_atoms = [i for i in range(self.natom) if i not in refit]
        frozen_charges = {i: float(q1[i]) for i in frozen_atoms}

        active_symmetry = [
            (i, j) for (i, j) in (symmetry_constraints or [])
            if i in refit and j in refit
        ]
        active_charge_constraints = [
            (idx, tgt) for (idx, tgt) in (charge_constraints or [])
            if all(a in refit for a in idx)
        ]

        print(f'  Stage 2: stronger restraint (a=0.001), refitting {len(refit)}/{self.natom} atoms')
        q2 = self.fit(
            total_charge=total_charge,
            restraint_a=0.001,
            charge_constraints=active_charge_constraints,
            symmetry_constraints=active_symmetry,
            frozen_atoms=frozen_atoms,
            frozen_charges=frozen_charges,
            unrestrained_atoms=hydrogens,
        )

        print(f'  Stage 2 charges: sum = {q2.sum():.6f}')

        return q2
