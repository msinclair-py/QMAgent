import numpy as np
from parsl import python_app
from pathlib import Path
from ..utils.file_ops import XYZContents
from ..utils.pydantic_models import (
    ESPCalculation,
    OptimizationResult,
    QMConfig,
    TorsionFitResult,
    TorsionScanResult
)

@python_app(executors=['cpu'])
def build_app(smiles: str,
              mol2_file: Path,
              resname: str,
              num_threads: int,
              max_iters: int) -> XYZContents:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from ..utils.file_ops import write_mol2, XYZContents

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'RDKit could not parse SMILES {smiles!r}')
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.numThreads = num_threads
    status = AllChem.EmbedMolecule(mol, params)

    # EmbedMolecule returns 0 on success and -1 on failure
    if status != 0:
        raise RuntimeError(f'RDKit failed to embed a conformer for {smiles!r} (status {status})')

    AllChem.MMFFOptimizeMolecule(mol, maxIters=max_iters)

    write_mol2(mol, mol2_file, resname)

    return XYZContents.from_mol2(mol2_file)

@python_app(executors=['gpu'])
def geomopt_app(geom_str: str,
                qm_config: QMConfig,
                log_file: Path,
                verbose: int,
                max_steps: int,
                constraints: Path | None,
                num_threads: int,
                max_memory: int,
                gpu: bool=True) -> OptimizationResult:
    if gpu:
        from gpu4pyscf import dft
    else:
        from pyscf import dft
    from pyscf import lib
    from pyscf.geomopt.geometric_solver import optimize
    from .distributed import load_dft
    from ..utils.pydantic_models import OptimizationResult

    lib.num_threads(num_threads)

    mf = load_dft(
        geom_str=geom_str,
        qm_config=qm_config,
        verbose=verbose,
        max_memory=max_memory,
        symmetry=False,
        gpu=gpu
    )

    mol_eq = optimize(
        mf,
        maxsteps=max_steps,
        constraints=constraints
    )

    opt_coords = mol_eq.atom_coords(unit='Angstrom')

    mf_final = dft.RKS(mol_eq)
    mf_final.xc = qm_config.functional
    mf_final.disp = qm_config.dispersion
    mf_final.grids.atom_grid = qm_config.grid_level
    e_final = mf_final.kernel()

    return OptimizationResult(e_final=e_final, coords=opt_coords)

@python_app(executors=['gpu'])
def esp_app(geom_str: str,
            qm_config: QMConfig,
            log_file: Path,
            solvated: bool,
            verbose: int,
            grid_pts: np.ndarray,
            num_threads: int,
            max_memory: int,
            gpu: bool=True) -> ESPCalculation:
    """Computes ESP as grid points.
    ESP = Nuclear contribution + Electronic contribution
    V(r) = sum_A Z_A / |r - R_A| - integral rho(r') / |r - r'| dr'
    """
    import numpy as np
    from pyscf import df, gto, lib
    from .distributed import load_dft
    from ..utils.pydantic_models import ESPCalculation

    lib.num_threads(num_threads)

    # symmetry=False is required here: with symmetry=True PySCF reorients the
    # molecule into its standard symmetry frame, so mol.atom_coords() (used for
    # the nuclear term) would no longer align with the MK grid, which is built
    # externally from the input coordinates. That misalignment silently corrupts
    # the ESP -- worst on the symmetric systems (e.g. phosphate, trimethyl).
    mf = load_dft(
        geom_str=geom_str,
        qm_config=qm_config,
        verbose=verbose,
        max_memory=max_memory,
        symmetry=False,
        gpu=gpu
    )

    mol = mf.mol

    if solvated:
        mf = mf.PCM()
        mf.with_solvent.method = 'C-PCM'
        mf.with_solvent.eps = 78.3553

    energy = mf.kernel()

    dm = mf.make_rdm1()
    bohr_per_angstrom = 1.8897259886
    grid_pts_bohr = grid_pts * bohr_per_angstrom
    coords_bohr = mol.atom_coords()  # pyscf returns coordinates in bohr

    # Nuclear contribution
    nuc_charges = mol.atom_charges()
    esp_nuc = np.zeros(len(grid_pts_bohr))

    for j in range(mol.natm):
        r = np.linalg.norm(grid_pts_bohr - coords_bohr[j], axis=1)
        esp_nuc += nuc_charges[j] / r

    # Electronic contribution using 1-electron integrals
    # This computes <mu| 1/|r-R| |nu> for each grid point R
    ngrids = len(grid_pts_bohr)
    esp_elec = np.zeros(ngrids)

    batch_size = 500
    for ibatch in range(0, ngrids, batch_size):
        batch_end = min(ibatch + batch_size, ngrids)
        batch_pts = grid_pts_bohr[ibatch:batch_end]

        # Create a "fake" molecule with point charges at grid locations
        fakemol = gto.fakemol_for_charges(batch_pts)

        # 3-center integrals (AO | 1/r | point)
        integrals = df.incore.aux_e2(mol, fakemol, intor='int3c2e')

        for k in range(batch_end - ibatch):
            esp_elec[ibatch + k] = -np.einsum('ij,ij->', dm, integrals[:, :, k])

    esp_total = esp_nuc + esp_elec # Hartree/e

    return ESPCalculation(esp_total=esp_total, energy=energy, solvated=solvated)

@python_app(executors=['gpu'])
def scan_torsions_app(xyz: XYZContents,
                      qm_config: QMConfig,
                      output_dir: Path,
                      target_angles: list[float],
                      torsion: tuple[int, int, int, int],
                      verbose: int,
                      num_threads: int,
                      max_memory: int=12000,
                      gpu: bool=True) -> TorsionScanResult:
    if gpu:
        from gpu4pyscf import dft
    else:
        from pyscf import dft
    import numpy as np
    from pyscf import lib
    from pyscf.geomopt.geometric_solver import optimize
    from .distributed import load_dft
    from .qm_agent import QMAgent
    from ..utils.file_ops import XYZContents, write_xyz
    from ..utils.pydantic_models import ScanPoint, TorsionScanResult

    lib.num_threads(num_threads)

    name = 'T' + '_'.join(str(idx) for idx in torsion)
    scan_dir = output_dir / name
    scan_dir.mkdir(parents=True, exist_ok=True)

    # geomeTRIC constraints are 1-indexed
    i, j, k, l = [idx + 1 for idx in torsion]

    results = []
    for angle in target_angles:
        geom_str = QMAgent.formulate_geometry_string(xyz.elements, xyz.coords)

        # symmetry=False: a constrained (frozen-dihedral) optimization can lower
        # the molecular point group mid-scan, so symmetry detection at build time
        # is unsafe here and can crash or silently corrupt the geometry.
        mf = load_dft(
            geom_str=geom_str,
            qm_config=qm_config,
            verbose=verbose,
            max_memory=max_memory,
            symmetry=False,
            gpu=gpu
        )

        # geomeTRIC constraints dictionary
        constraints = {
            'set': [
                {
                    'type': 'dihedral',
                    'indices': [i, j, k, l],
                    'value': angle,
                }
            ]
        }

        # Dispersion is applied once, via mf.disp set in load_dft (consistent with
        # geomopt_app/esp_app). Do NOT also wrap with dftd3.dftd3(mf): that would
        # double-count the correction and corrupt the very torsion energy surface
        # paramfit is fit against.
        try:
            mol_opt = optimize(
                mf,
                maxsteps=150,
                constraints=constraints,
            )

            mf2 = dft.RKS(mol_opt)

            mf2.xc = qm_config.functional
            mf2.disp = qm_config.dispersion  # single dispersion source; see note above
            mf2.grids.atom_grid = qm_config.grid_level

            energy = mf2.kernel()

            coords = mol_opt.atom_coords(unit='Angstrom')
            
            output_file = scan_dir / f'opt_{angle:06.1f}.xyz'
            contents = XYZContents(
                elements=xyz.elements,
                coords=coords,
                comment=''
            )

            write_xyz(output_file, contents)

            results.append(ScanPoint(xyz_file=output_file, energy=energy, angle=angle))

        except Exception as e:
            print(f'  WARNING: Optimization failed at {angle}°: {e}')
            break

    return TorsionScanResult(torsion=torsion, points=results)

@python_app(executors=['cpu'])
def resp_app(xyz: XYZContents,
             qm_config: QMConfig,
             esp: np.ndarray,
             grid_pts: np.ndarray,
             charge_constraints: list[tuple[list[int], float]] | None,
             symmetry_pairs: list[tuple[int, int]] | None,
             refit_atoms: set[int] | None=None) -> np.ndarray:
    import numpy as np
    from .resp_fitter import RESPFitter

    # RESPFitter expects atomic positions and grid points in bohr
    bohr_per_angstrom = 1.8897259886
    coords_bohr = xyz.coords * bohr_per_angstrom
    grid_bohr = grid_pts * bohr_per_angstrom

    fitter = RESPFitter(coords_bohr, grid_bohr, esp)
    q = fitter.two_stage_resp(
        xyz.elements,
        total_charge=qm_config.charge,
        charge_constraints=charge_constraints,
        symmetry_constraints=symmetry_pairs,
        refit_atoms=refit_atoms,
    )

    return q

@python_app(executors=['cpu'])
def fit_torsions_app(scan: TorsionScanResult,
                     prmtop: Path,
                     output_dir: Path,
                     amberhome: Path,
                     max_periodicity: int=4) -> TorsionFitResult:
    """Fit AMBER dihedral parameters to a single QM torsion scan via paramfit.

    Builds an AMBER trajectory of the scan's optimized geometries plus a matching
    QM energy file, then runs paramfit twice: once with K_ONLY to fit the QM/MM
    energy offset, then with LOAD to fit V_n / phase for n = 1..max_periodicity.
    The fitted DIHE terms are written to a per-torsion frcmod.

    Arguments:
        scan (TorsionScanResult): Completed scan over one rotatable bond.
        prmtop (Path): Topology for the parameterized residue (for atom types).
        output_dir (Path): Parent directory for per-torsion paramfit output.
        amberhome (Path): Path to where the ambertools binaries are.
        max_periodicity (int): Defaults to 4. Highest dihedral periodicity to fit.

    Returns:
        (TorsionFitResult): Fitted frcmod path and GAFF2 atom-type quartet.
    """
    import parmed
    from .amber_apps import run_paramfit
    from ..utils.file_ops import XYZContents
    from ..utils.pydantic_models import TorsionFitResult

    name = 'T_' + '_'.join(str(idx) for idx in scan.torsion)
    work = output_dir / f'paramfit_{name}'
    work.mkdir(parents=True, exist_ok=True)

    # AMBER text trajectory: title + 10F8.3 coordinates per frame, no box
    mdcrd = work / 'scan.mdcrd'
    with open(mdcrd, 'w') as f:
        f.write(f'{name}\n')
        for point in scan.points:
            flat = XYZContents.from_xyz(point.xyz_file).coords.flatten()
            for c in range(0, len(flat), 10):
                f.write(''.join(f'{x:8.3f}' for x in flat[c:c + 10]) + '\n')

    qm_energies = work / 'qm_energies.dat'
    qm_energies.write_text(
        '\n'.join(f'{point.energy:.10f}' for point in scan.points) + '\n'
    )

    # GAFF2 atom-type quartet for this dihedral, read from the topology
    parm = parmed.load_file(str(prmtop))
    a, b, c, d = (parm.atoms[idx].type for idx in scan.torsion)

    def write_job_ctrl(path: Path, *, param_file: Path | None=None,
                       frcmod_out: Path | None=None) -> None:
        lines = [
            'RUNTYPE=FIT',
            f'NSTRUCTURES={len(scan.points)}',
            'COORDINATE_FORMAT=TRAJECTORY',
            'QM_FILE_FORMAT=NUMERIC',
            'QM_ENERGY_UNITS=HARTREE',
            'ALGORITHM=BOTH',
            'OPTIMIZATIONS=200',
            'MAX_GENERATIONS=10000',
            'FUNC_TO_FIT=SUM_SQUARES_AMBER_STANDARD',
        ]
        if param_file is not None:
            lines += ['PARAMETERS_TO_FIT=LOAD',
                      f'PARAMETER_FILE_NAME={param_file}']
        else:
            lines.append('PARAMETERS_TO_FIT=K_ONLY')
        if frcmod_out is not None:
            lines.append(f'WRITE_FRCMOD={frcmod_out}')
        path.write_text('\n'.join(lines) + '\n')

    # Pass 1: K_ONLY - fit the QM/MM energy offset
    fit_k = work / 'fit_K.in'
    write_job_ctrl(fit_k)
    if not run_paramfit(fit_k, prmtop, mdcrd, qm_energies, work / 'fit_K.log', amberhome):
        return TorsionFitResult(torsion=scan.torsion, atom_types=(a, b, c, d))

    # Pass 2: LOAD - fit V_n / phase for each periodicity
    param_file = work / 'params.in'
    param_lines = [f'NDIHEDRALS {max_periodicity}']
    for n in range(1, max_periodicity + 1):
        param_lines.append(
            f'DIHED  {a:<2}-{b:<2}-{c:<2}-{d:<2}  KP=yes  PHASE=yes  PERIODICITY={n}'
        )
    param_file.write_text('\n'.join(param_lines) + '\n')

    frcmod_out = work / 'fit.frcmod'
    fit_in = work / 'fit.in'
    write_job_ctrl(fit_in, param_file=param_file, frcmod_out=frcmod_out)
    success = run_paramfit(fit_in, prmtop, mdcrd, qm_energies, work / 'fit.log', amberhome)

    return TorsionFitResult(
        torsion=scan.torsion,
        atom_types=(a, b, c, d),
        frcmod_file=frcmod_out if success else None,
    )

def load_dft(
    geom_str: str,
    qm_config: QMConfig,
    verbose: int,
    max_memory: int,
    symmetry: bool=True,
    gpu: bool=True
):
    from pyscf import gto

    mol = gto.M(
        atom=geom_str,
        basis=qm_config.basis,
        charge=qm_config.charge,
        spin=qm_config.multiplicity - 1,
        verbose=verbose,
        max_memory=max_memory,
        symmetry=symmetry,
    )

    if gpu:
        from gpu4pyscf import dft
    else:
        from pyscf import dft

    mf = dft.RKS(mol)

    mf.xc = qm_config.functional
    mf.disp = qm_config.dispersion
    mf.grids.atom_grid = qm_config.grid_level

    return mf
