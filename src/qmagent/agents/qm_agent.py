import asyncio
import numpy as np
import os
import parsl
from academy.agent import Agent, action
from collections import defaultdict
from parsl.config import Config
from parsl.executors import ThreadPoolExecutor
from pathlib import Path
from rdkit import Chem
import sys
import tempfile

from .amber_apps import (
    get_lib_files,
    run_antechamber,
    run_parmchk2
)
from .distributed import (
    build_app,
    geomopt_app,
    esp_app,
    scan_torsions_app,
    resp_app,
    fit_torsions_app
)
from ..utils.pydantic_models import (
    AMBERConfig,
    AMBERResultSet,
    ESPResult,
    GeomOptResult,
    QMConfig,
    RESPCharges,
    TorsionFitResult,
    TorsionFitSet,
    TorsionScanResult,
    TorsionScanSet,
)
from ..utils import file_ops
from ..utils.file_ops import (
    XYZContents,
    write_xyz
)

Torsion = tuple[int, int, int, int]
Torsions = set[Torsion]


def _local_parsl_config(num_threads: int) -> Config:
    """Minimal local parsl config so the @python_app workflow steps can run.

    Defines 'cpu' and 'gpu' executors (both plain CPU thread pools) to match the
    executor labels requested by the apps in ``distributed.py``. This only
    satisfies parsl's task routing -- the 'gpu'-labelled apps (geomopt/esp/scan)
    still require ``gpu4pyscf`` + CUDA at runtime, so on a non-GPU host they fail
    at import. For HPC/GPU deployments pass an explicit ``parsl_config`` built
    from ``HeterogeneousSettings.config_factory`` instead.
    """
    return Config(executors=[
        ThreadPoolExecutor(label='cpu', max_threads=num_threads),
        ThreadPoolExecutor(label='gpu', max_threads=1),
    ])


class QMAgent(Agent):
    def __init__(self,
                 num_threads: int,
                 max_memory: int=160000,
                 parsl_config: Config | None=None,
                 use_gpu: bool=True,):
        super().__init__()
        self.num_threads = num_threads
        self.max_memory = max_memory
        # Loaded in agent_on_startup; defaults to a local CPU config.
        self._parsl_config = parsl_config
        # When False the DFT apps import ``pyscf`` (CPU) instead of ``gpu4pyscf``,
        # so the whole QM pipeline runs on a machine with no GPU/CUDA (the live
        # demo path). HPC/GPU deployments keep the default True.
        self.use_gpu = use_gpu

    async def agent_on_startup(self) -> None:
        """Load a parsl DataFlowKernel so the workflow apps can execute.

        Without this, the first @python_app call raises NoDataFlowKernelError.
        """
        config = self._parsl_config or _local_parsl_config(self.num_threads)
        parsl.load(config)

    async def agent_on_shutdown(self) -> None:
        """Tear the parsl runtime down cleanly on agent shutdown."""
        try:
            parsl.dfk().cleanup()
        except Exception:
            pass
        parsl.clear()

    @action
    async def build_compound(self,
                             smiles: str,
                             mol2_file: Path,
                             resname: str,
                             max_iters: int=2000) -> None:
        """Builds a smiles compound into a mol2 file with explicit hydrogens.

        Currently utilizes ETKDGv3 for conformer generation and MMFF force field.

        Arguments:
            smiles (str): A valid smiles string. RDKit will crash if this is invalid.
            mol2_file (Path): Path to output mol2 file.
            resname (str): Residue name to write into the mol2 file.
            max_iters (int): Defaults to 2000. Number of max iterations of MMFF optimization.

        Returns:
            None
        """
        future = asyncio.wrap_future(
            build_app(
                smiles=smiles,
                mol2_file=mol2_file,
                resname=resname,
                num_threads=self.num_threads,
                max_iters=max_iters
            )
        )

        await future
    
    @action
    async def geometry_optimization(self,
                                    mol2_file: Path,
                                    output_path: Path,
                                    optimization_stages: list[QMConfig],
                                    constraints: str | None=None,
                                    max_steps: int=200) -> GeomOptResult:
        """Performs a series of geometry optimization simulations defined by the 
        coordinator agent. 

        Stages run in order, each starting from the previous stage's optimized
        geometry (e.g. a cheap pre-optimization followed by a higher-level
        refinement). The final stage's geometry is written to disk and returned
        with its energy.

        Arguments:
            mol2_file (Path): Input compound to be optimized.
            output_path (Path): Output path for optimized compound.
            optimization_stages (list[QMConfig]): List of optimization stages
                stored in a pydantic model. Attributes include basis, functional, 
                dispersion, charge and multiplicity. Must be non-empty.
            constraints (str | None): Defaults to None. Path to constraints file
                which defines nuclei constraints for constrained optimization.

        Returns:
            (GeomOptResult): The optimized geometry (xyz file) and final energy
                from the last stage.

        Raises:
            ValueError: If ``optimization_stages`` is empty.
        """
        if not optimization_stages:
            raise ValueError('optimization_stages must contain at least one QMConfig stage.')

        contents = XYZContents.from_mol2(mol2_file)
        elements = contents.elements
        coords = contents.coords

        e_final = None
        for optimization_stage in optimization_stages:
            geom_str = self.formulate_geometry_string(elements, coords)

            future = asyncio.wrap_future(
                geomopt_app(
                    geom_str=geom_str,
                    qm_config=optimization_stage,
                    log_file = str(output_path / f'pyscf_{optimization_stage.basis}.log'),
                    verbose=4,
                    max_steps=max_steps,
                    constraints = constraints,
                    num_threads = self.num_threads,
                    max_memory = self.max_memory,
                    gpu = self.use_gpu
                )
            )

            result = await future
            e_final, coords = result.e_final, result.coords

        final_xyz = output_path / f'optimized_{optimization_stages[-1].basis}.xyz'

        write_xyz(final_xyz, XYZContents(elements=elements, coords=coords))

        return GeomOptResult(xyz_file=final_xyz, energy=e_final)

    @action
    async def compute_ESP_charges(self,
                                  contents: XYZContents,
                                  output_path: Path,
                                  qm_config: QMConfig) -> ESPResult:
        """Compute ESP at MK grid points.

        For solvated calculation, PySCF supports PCM/CPCM via the pyscf.solvent module.

        Arguments:
            xyz_file (Path): Input compound to be optimized.
            output_path (Path): Output path for optimized compound.
            qm_config (QMConfig): Pydantic model whose attributes include basis, functional, 
                dispersion, charge and multiplicity.

        Returns:
            (list[ESPResult]): Results for vacuum and solvated ESP charge calculation. Includes,
                label for solvation, total electrostatic potential and energy.
        """
        grid_points = self.generate_mk_grid(contents.elements, contents.coords)
        geom_str = self.formulate_geometry_string(contents.elements, contents.coords)

        futures = []
        for phase, solvated in [('gas', False), ('solvent', True)]:
            futures.append(
                asyncio.wrap_future(
                    esp_app(
                        geom_str=geom_str,
                        qm_config=qm_config,
                        log_file=str(output_path / f'esp_{phase}.dat'),
                        solvated=solvated,
                        verbose=4,
                        grid_pts=grid_points,
                        num_threads=self.num_threads,
                        max_memory=self.max_memory,
                        gpu=self.use_gpu
                    )
                )
            )

        # NOTE: error handling goes here
        calculations = await asyncio.gather(*futures)

        result = ESPResult(calculations=calculations, metadata=qm_config.model_dump())
        result.save(output_path / 'esp_results.json')

        return result

    @action
    async def scan_torsions(self,
                            contents: XYZContents,
                            qm_config: QMConfig,
                            output_dir: Path,
                            torsions: Torsions,
                            scan_step: int=15) -> tuple[TorsionScanSet, Torsions]:
        num_angles = 360 // scan_step
        target_angles = [i * scan_step for i in range(num_angles)]

        futures = []
        for torsion in torsions:
            futures.append(
                asyncio.wrap_future(
                    scan_torsions_app(
                        xyz=contents,
                        qm_config=qm_config,
                        output_dir=output_dir,
                        target_angles=target_angles,
                        torsion=torsion,
                        verbose=4,
                        num_threads=self.num_threads,
                        max_memory=self.max_memory,
                        gpu=self.use_gpu,
                    )
                )
            )

        scans = await asyncio.gather(*futures)

        failed = set()
        for scan in scans:
            if scan.angles.shape[0] != num_angles:
                failed.add(scan.torsion)

        dataset = TorsionScanSet(scans=scans, metadata=qm_config.model_dump())
        dataset.save(output_dir / 'torsion_scan.json')

        return dataset, failed

    @action
    async def fit_RESP_charges(self,
                               molecule: XYZContents,
                               mol2_file: Path,
                               esp_results: ESPResult,
                               qm_config: QMConfig,
                               delta_resp2: float=0.5) -> RESPCharges:
        """Fit RESP2 charges from the gas- and solvent-phase ESP calculations.

        RESP2(delta) interpolates between the two phases as
        delta * q_solv + (1 - delta) * q_gas, with each phase fit to its QM ESP
        on a shared Merz-Kollman grid under the molecule's symmetry equivalences.
        Each phase's fit is itself the standard two-stage RESP: stage 1 fits every
        atom under a weak restraint, then stage 2 freezes everything except
        aliphatic (sp3) CH/CH2/CH3 carbons and their hydrogens -- identified from
        the mol2 topology via ``find_resp_refit_atoms`` -- and refits only those
        under a stronger restraint.

        Arguments:
            molecule (XYZContents): The optimized geometry the ESP was computed on;
                supplies the element ordering and coordinates for the MK grid.
            mol2_file (Path): The model compound's mol2 file, used to detect
                symmetry-equivalent atoms whose charges must be constrained equal.
            esp_results (ESPResult): Gas- and solvent-phase ESP calculations. The
                phases are selected explicitly by each calculation's ``solvated``
                flag, so their order within the set does not matter; exactly one
                gas-phase and one solvent-phase calculation must be present.
            qm_config (QMConfig): QM settings; supplies the net charge.
            delta_resp2 (float): Defaults to 0.5. RESP2 solvent weighting delta.

        Returns:
            (RESPCharges): The interpolated RESP2 charges with metadata.

        Raises:
            ValueError: If the ESP set does not contain exactly one gas-phase and
                one solvent-phase calculation.
        """
        # Select phases by the solvated flag rather than positional order: the
        # RESP2 interpolation q = delta*q_solv + (1-delta)*q_gas is meaningless
        # if the two phases are swapped, and gather() preserves submission order,
        # not semantic order.
        gas = [c for c in esp_results if not c.solvated]
        solv = [c for c in esp_results if c.solvated]
        if len(gas) != 1 or len(solv) != 1:
            raise ValueError(
                'RESP2 requires exactly one gas-phase and one solvent-phase ESP '
                f'calculation, got {len(gas)} gas and {len(solv)} solvent. '
                'Re-run compute_ESP_charges to produce both phases.'
            )
        gas_esp, solv_esp = gas[0], solv[0]

        symmetry_pairs = self.find_symmetry_pairs(mol2_file)
        refit_atoms = self.find_resp_refit_atoms(mol2_file)
        grid_pts = self.generate_mk_grid(molecule.elements, molecule.coords)

        futures = {}
        for phase, calc in (('gas', gas_esp), ('solv', solv_esp)):
            futures[phase] = asyncio.wrap_future(
                resp_app(
                    xyz=molecule,
                    qm_config=qm_config,
                    esp=calc.esp_total,
                    grid_pts=grid_pts,
                    charge_constraints=None, # how do we propagate this information?
                    symmetry_pairs=symmetry_pairs,
                    refit_atoms=refit_atoms,
                )
            )

        q_gas = await futures['gas']
        q_solv = await futures['solv']

        q_resp2 = delta_resp2 * q_solv + (1 - delta_resp2) * q_gas
        assert np.isclose(q_resp2.sum(), qm_config.charge)

        metadata = {'method': 'RESP2', 'delta': delta_resp2, 'total_charge': float(q_resp2.sum())}
        resp_charges = RESPCharges(elements=molecule.elements, charges=q_resp2, metadata=metadata)

        return resp_charges

    @action
    async def integrate_AMBER_ff(self,
                                 amber_config: AMBERConfig) -> AMBERResultSet:
        """Integrate the QM-derived RESP2 charges into an AMBER/GAFF2 force field.

        Runs the AmberTools pipeline: antechamber assigns GAFF2 atom types while
        keeping our pre-computed RESP2 charges, parmchk2 fills in any missing
        parameters, and tleap validates that the residue builds while writing the
        library (.lib) and topology (prmtop/inpcrd) files.

        Arguments:
            amber_config (AMBERConfig): AMBER settings including input sdf, the
                RESP charge file, output paths, residue name, net charge and
                amberhome.

        Returns:
            (AMBERResultSet): The generated mol2, frcmod, lib and topology files.
        """
        # assign GAFF2 atom types while keeping our RESP2 charges
        future = asyncio.wrap_future(
            run_antechamber(
                sdf_file=amber_config.sdf_file,
                resp_charges=amber_config.resp_charges,
                mol2_output=amber_config.mol2_file,
                resname=amber_config.resname,
                amberhome=amber_config.amberhome,
                charge=amber_config.charge
            )
        )

        if not await future:
            raise RuntimeError('antechamber failed during GAFF2 atom typing')

        # fill in any missing GAFF2 parameters
        future = asyncio.wrap_future(
            run_parmchk2(
                mol2=amber_config.mol2_file,
                frcmod_output=amber_config.frcmod_file,
                amberhome=amber_config.amberhome
            )
        )

        if not await future:
            raise RuntimeError('parmchk2 failed during missing-parameter check')

        # validate the residue builds and write the lib + topology files
        future = asyncio.wrap_future(
            get_lib_files(
                mol2=amber_config.mol2_file,
                frcmod_file=amber_config.frcmod_file,
                prmtop=amber_config.prmtop,
                resname=amber_config.resname,
                amberhome=amber_config.amberhome
            )
        )

        if not await future:
            raise RuntimeError('tleap failed to build the residue')

        return AMBERResultSet(
            mol2_file=amber_config.mol2_file,
            frcmod_file=amber_config.frcmod_file,
            lib_file=amber_config.lib_files,
            prmtop=amber_config.prmtop,
            inpcrd=amber_config.prmtop.with_suffix('.inpcrd'),
            metadata={'resname': amber_config.resname}
        )

    @action
    async def fit_torsions(self,
                           torsion_scans: TorsionScanSet,
                           amber_config: AMBERConfig,
                           output_dir: Path,
                           max_periodicity: int=4) -> TorsionFitSet:
        """Fit AMBER dihedral parameters to the QM torsion scans via paramfit.

        Each scanned torsion is fit independently (in parallel) against its QM
        energy surface, then the per-torsion DIHE blocks are merged into a single
        refined frcmod to be loaded after the base GAFF2 parameters.

        Arguments:
            torsion_scans (TorsionScanSet): Completed scans for every rotatable bond.
            amber_config (AMBERConfig): AMBER settings supplying the prmtop and
                amberhome.
            output_dir (Path): Output directory for paramfit work and the refined
                frcmod.
            max_periodicity (int): Defaults to 4. Highest dihedral periodicity to fit.

        Returns:
            (TorsionFitSet): Per-torsion fits and the merged refined frcmod path.
        """
        futures = []
        for scan in torsion_scans:
            futures.append(
                asyncio.wrap_future(
                    fit_torsions_app(
                        scan=scan,
                        prmtop=amber_config.prmtop,
                        output_dir=output_dir,
                        amberhome=amber_config.amberhome,
                        max_periodicity=max_periodicity
                    )
                )
            )

        fits = await asyncio.gather(*futures)

        refined_frcmod = output_dir / 'refined.frcmod'
        self.merge_frcmods(fits, refined_frcmod)

        dataset = TorsionFitSet(fits=fits, refined_frcmod=refined_frcmod)
        dataset.save(output_dir / 'torsion_fits.json')

        return dataset

    @action
    async def find_rotatable_torsions(self,
                                      mol2_file: Path) -> Torsions:
        """Detect a representative dihedral quartet for each rotatable bond.

        A rotatable bond is a single, acyclic bond between two non-terminal
        heavy-atom centres; for each one a neighbour is chosen on either side to
        define the (i, j, k, l) dihedral to scan. Runs agent-side as it reads the
        mol2 off the agent's filesystem.

        Arguments:
            mol2_file (Path): Path to the input mol2 file.

        Returns:
            (Torsions): Set of 0-indexed dihedral atom quartets, one per
                rotatable bond.
        """
        mol = Chem.MolFromMol2File(str(mol2_file), removeHs=False, sanitize=True)

        pattern = Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')

        torsions: Torsions = set()
        for b, c in mol.GetSubstructMatches(pattern):
            b_atom, c_atom = mol.GetAtomWithIdx(b), mol.GetAtomWithIdx(c)
            a = next((n.GetIdx() for n in b_atom.GetNeighbors() if n.GetIdx() != c), None)
            d = next((n.GetIdx() for n in c_atom.GetNeighbors() if n.GetIdx() != b), None)
            if a is not None and d is not None:
                torsions.add((a, b, c, d))

        return torsions

    @action
    async def prepare_amber_inputs(self,
                                   charges: list[float],
                                   mol2_file: Path,
                                   charge_file: Path,
                                   sdf_file: Path) -> None:
        """Write the antechamber inputs (RESP charge file + sdf) agent-side.

        antechamber reads the pre-computed charges from a free-format -cf file (with
        -c rc, so only atom typing happens) and takes the build geometry as an sdf.
        Both are written here, on the agent's filesystem, in a single round trip.

        Arguments:
            charges (list[float]): Partial charges in atom (mol2) order.
            mol2_file (Path): Path to the build mol2 to convert to sdf.
            charge_file (Path): Path to the -cf charge file to be written.
            sdf_file (Path): Path to the sdf file to be written.
        """
        file_ops.write_charge_file(np.asarray(charges), charge_file)
        file_ops.mol2_to_sdf(mol2_file, sdf_file)

    @action
    async def execute_code(self,
                           code_snippet: str,
                           workdir: Path | None = None,
                           extra_paths: list[Path] | None = None,
                           timeout: float = 300.0) -> dict[str, str]:
        """Code execution action. Used for performing bespoke analysis or simulation,
        entirely LLM agent driven. Initially here to allow the reasoning to sidestep
        the above curated actions if need be.

        The snippet is written to a temporary file and executed in a fresh Python
        subprocess (the same interpreter running the agent, so it shares the
        agent's environment: rdkit, numpy, pyscf, ambertools bindings, etc.).
        Running out-of-process isolates the agent from hard crashes (segfaults,
        ``sys.exit``, C-extension aborts) in the generated code and lets us reap
        the whole thing on timeout.

        The subprocess runs with ``workdir`` as its current directory (so relative
        paths the model writes land in the run's output directory, and it can read
        the artifacts produced by earlier steps), and with the qmagent package root
        plus any ``extra_paths`` (e.g. a skill's ``scripts/`` directory) prepended
        to ``PYTHONPATH``. That lets generated code both ``import qmagent...`` and
        import/execute the helper scripts shipped with the project skills.

        stdout and stderr are captured separately. On any failure -- a raised
        exception, a non-zero exit, or a timeout -- the reason (Python traceback,
        exit status, or timeout notice) lands in ``stderr`` so the LLM agent can
        read why the code failed and revise it. On success ``stdout`` carries the
        printed output (possibly empty) and ``stderr`` is whatever the snippet
        wrote there (usually empty).

        Arguments:
            code_snippet (str): A multi-line string containing the python code to be
                executed by the QMAgent.
            workdir (Path | None): Defaults to None. Directory to run the snippet
                in (typically the run's output directory). Created if missing;
                falls back to the process cwd when None.
            extra_paths (list[Path] | None): Defaults to None. Additional directories
                to prepend to the subprocess ``PYTHONPATH`` (e.g. skill ``scripts/``
                dirs) so their modules are importable from the snippet.
            timeout (float): Defaults to 300.0. Wall-clock seconds before the
                subprocess is killed and reported as a timeout.

        Returns:
            (dict[str, str]): ``{'stdout': ..., 'stderr': ..., 'returncode': ...}``.
                A non-zero ``returncode`` signals failure; any traceback is placed
                in ``stderr``.
        """
        if workdir is not None:
            workdir = Path(workdir)
            workdir.mkdir(parents=True, exist_ok=True)

        # Build PYTHONPATH: qmagent package root (parent of the `qmagent` package
        # dir, i.e. the `src` root) first, then any skill script dirs, then
        # whatever the agent already had, so generated code can import the
        # project and the skill helpers.
        pkg_root = Path(__file__).resolve().parents[2]  # .../src
        path_parts = [str(pkg_root)]
        if extra_paths:
            path_parts.extend(str(Path(p)) for p in extra_paths)
        existing = os.environ.get('PYTHONPATH')
        if existing:
            path_parts.append(existing)

        env = os.environ.copy()
        env['PYTHONPATH'] = os.pathsep.join(path_parts)

        # Write to a real temp file rather than passing via ``-c`` so tracebacks
        # carry a stable filename and correct line numbers for the agent to read.
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', prefix='qm_exec_', delete=False,
        ) as f:
            f.write(code_snippet)
            script_path = f.name

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(workdir) if workdir is not None else None,
                env=env,
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                # Drain the pipes so the killed child doesn't leave a zombie.
                stdout_b, stderr_b = await proc.communicate()
                stdout = stdout_b.decode('utf-8', errors='replace')
                stderr = stderr_b.decode('utf-8', errors='replace')
                timeout_msg = f'Execution timed out after {timeout:g}s and was killed.'
                stderr = f'{stderr}\n{timeout_msg}'.strip() if stderr else timeout_msg
                return {'stdout': stdout, 'stderr': stderr, 'returncode': '-1'}

            stdout = stdout_b.decode('utf-8', errors='replace')
            stderr = stderr_b.decode('utf-8', errors='replace')
            return {
                'stdout': stdout,
                'stderr': stderr,
                'returncode': str(proc.returncode),
            }
        finally:
            Path(script_path).unlink(missing_ok=True)

    @staticmethod
    def merge_frcmods(fits: list[TorsionFitResult],
                      output_file: Path) -> None:
        """Merge the per-torsion paramfit DIHE blocks into one refined frcmod.

        The result is loaded with loadamberparams after the base GAFF2 frcmod so
        the refined dihedral terms override the analogy-based GAFF2 estimates.

        Arguments:
            fits (list[TorsionFitResult]): Per-torsion fit results. Entries whose
                frcmod_file is None (failed fits) are skipped.
            output_file (Path): Path to the merged refined frcmod to be written.
        """
        body = ['Refined dihedrals from QM torsion scans', '', 'DIHE']

        for fit in fits:
            if fit.frcmod_file is None:
                continue

            in_dihe = False
            block = []
            for line in fit.frcmod_file.read_text().splitlines():
                stripped = line.strip()
                if stripped == 'DIHE':
                    in_dihe = True
                    continue
                if in_dihe:
                    if not stripped or stripped in {'IMPROPER', 'NONBON', 'MASS', 'BOND', 'ANGLE'}:
                        break
                    block.append(line)

            if block:
                body.append(f'# {"-".join(fit.atom_types)}')
                body.extend(block)

        body += ['', '']
        output_file.write_text('\n'.join(body) + '\n')

    @staticmethod
    def find_symmetry_pairs(mol2: Path) -> list[tuple[int, int]]:
        """Find symmetry-equivalent atom pairs in the model compound.
        (E.g. the 3 hydrogens on a methyl group which must each have the same charge)

        Topologically equivalent atoms share a canonical rank when ties are left
        unbroken (``Chem.CanonicalRankAtoms(mol, breakTies=False)``); every atom in
        such a class must carry the same RESP charge. For each class we chain
        consecutive members -- (a, b), (b, c), ... -- so transitivity through the
        equal-charge constraints equalizes the *whole* class. The previous
        implementation took a single graph automorphism and would miss members not
        moved by that one permutation (e.g. it could equate only two of a methyl's
        three hydrogens).

        Arguments:
            mol2 (Path): Path to the input mol2 file for checking symmetry

        Returns:
            (list[tuple[int, int]]): 0-indexed (i, j) pairs with i < j whose
                charges must be constrained equal.
        """
        mol = Chem.MolFromMol2File(str(mol2), removeHs=False, sanitize=True)

        ranks = Chem.CanonicalRankAtoms(mol, breakTies=False)

        classes: dict[int, list[int]] = defaultdict(list)
        for idx, rank in enumerate(ranks):
            classes[rank].append(idx)

        symmetry_pairs: list[tuple[int, int]] = []
        for members in classes.values():
            members.sort()
            for a, b in zip(members, members[1:]):
                symmetry_pairs.append((a, b))

        return symmetry_pairs

    @staticmethod
    def find_resp_refit_atoms(mol2: Path) -> set[int]:
        """Identify the atoms RESP stage 2 is allowed to refit.

        Standard two-stage RESP (Bayly et al. 1993, and its common implementations
        e.g. Antechamber/resp) fits every atom in stage 1 under a weak restraint,
        then in stage 2 freezes everything *except* aliphatic (sp3) CH/CH2/CH3
        carbons and their attached hydrogens, which are refit under a stronger
        restraint. Methyl/methylene charges are the most poorly determined by the
        ESP (they sit in the molecular interior, far from most grid points) and
        the least chemically informative, so they alone are allowed to move again
        while every other charge (heteroatoms, sp2/aromatic carbons, polar
        hydrogens) is locked at its stage-1 value.

        Arguments:
            mol2 (Path): Path to the input mol2 file for topology/hybridization.

        Returns:
            (set[int]): 0-indexed atoms (sp3 aliphatic carbons and their bonded
                hydrogens) eligible for the stage-2 refit. All other atoms should
                be frozen at their stage-1 charges.
        """
        mol = Chem.MolFromMol2File(str(mol2), removeHs=False, sanitize=True)

        refit_atoms: set[int] = set()
        for atom in mol.GetAtoms():
            if (
                atom.GetSymbol() == 'C'
                and not atom.GetIsAromatic()
                and atom.GetHybridization() == Chem.HybridizationType.SP3
            ):
                refit_atoms.add(atom.GetIdx())
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetSymbol() == 'H':
                        refit_atoms.add(neighbor.GetIdx())

        return refit_atoms

    @staticmethod
    def generate_mk_grid(elements: list[str],
                         coords: list[str],
                         density: float=1.) -> np.ndarray:
        """Generate Merz-Kollman ESP grid points.

        The MK scheme places points on nested Connolly surfaces at 1.4, 1.6, 1.8, and 2.0
        times the vdW radius of each atom.

        Arguments:
            elements (list[str]): List of element names in index order (e.g. [H, C, O, S]).
            coords (np.ndarray): Coordinates of molecular system.
            density (float): Defaults to 1.0. Scales the number of points for a given atom.

        Returns:
            (np.ndarray): MK grid array.
        """
        common_radii = {'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52,
                        'S': 1.80, 'F': 1.47, 'Cl': 1.75, 'Br': 1.85}
        vdw_radii = defaultdict(lambda: 1.70, common_radii)

        golden_ratio = (1 + np.sqrt(5)) / 2

        grid_points = []
        shell_factors = [1.4, 1.6, 1.8, 2.0]
        for factor in shell_factors:
            for i, (elem, center) in enumerate(zip(elements, coords, strict=True)):
                radius = vdw_radii[elem] * factor
                area = 4.0 * np.pi * radius**2
                npoints = max(int(area * density), 50)
                indices = np.arange(npoints)
                theta = 2 * np.pi * indices / golden_ratio
                phi = np.arccos(1 - 2 * (indices + 0.5) / npoints)

                x = center[0] + radius * np.sin(phi) * np.cos(theta)
                y = center[1] + radius * np.sin(phi) * np.sin(theta)
                z = center[2] + radius * np.cos(phi)

                shell_pts = np.column_stack([x, y, z])

                keep = np.ones(len(shell_pts), dtype=bool)
                for j, (elem_j, center_j) in enumerate(zip(elements, coords, strict=True)):
                    if j == i:
                        continue

                    r_excl = vdw_radii[elem] * shell_factors[0]
                    dists = np.linalg.norm(shell_pts - center_j, axis=1)
                    keep &= dists > r_excl

                grid_points.append(shell_pts[keep])

        return np.vstack(grid_points)

    @staticmethod
    def formulate_geometry_string(elements: list[str],
                                  coords: np.ndarray) -> str:
        """Generates a geometry string for PySCF based on the input element identities
        and the coordinates for each atom.

        Arguments:
            elements (list[str]): List of element names in index order (e.g. [H, C, O, S]).
            coords (np.ndarray): Coordinates of molecular system.

        Returns:
            (str): Geometry string ready for PySCF.
        """
        return '\n'.join(
            f'{e}  {c[0]:.8f}  {c[1]:.8f}  {c[2]:.8f}'
            for e, c in zip(elements, coords, strict=True)
        )
