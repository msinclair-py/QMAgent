"""Harness-agnostic QM tool layer -- the execution layer's public surface.

``QMToolkit`` holds one method per curated QM step. Each dispatches to the
distributed ``QMAgent`` over its academy ``Handle`` and returns a short summary
string, stashing large results as keyed *artifacts* on the run state so a caller
chains steps by passing short keys rather than echoing big structured objects.

Nothing here imports an LLM harness, which is what lets the same tools be driven
two ways:

* ``llm_interface`` wraps these bound methods in a pydantic-ai ``FunctionToolset``
  (mapping ``QMToolError`` -> ``ModelRetry``) for the self-managed harness.
* ``mcp_server`` registers the same bound methods on a FastMCP server (mapping
  ``QMToolError`` -> ``ToolError``) for an externally managed one.

Both adapters build their schemas by reading these signatures and docstrings, so
a tool is defined, documented and fixed exactly once. Keep that property: put
behaviour here, not in an adapter.
"""

import functools
import numpy as np
from academy.handle import Handle
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from .utils.file_ops import XYZContents
from .utils.pydantic_models import (
    AMBERConfig,
    ESPResult,
    GeomOptResult,
    QMConfig,
    QMExperiment,
    RESPCharges,
    TorsionScanSet,
)

T = TypeVar('T')
Torsion = tuple[int, int, int, int]

# Cap on the stdout/stderr a single run_code call puts into the conversation.
#
# Tool output is the dominant cost in a QM agent, and it compounds: whatever a
# tool returns is re-sent to the model on every subsequent step, so one verbose
# call is paid for once per remaining turn -- and this bites either harness, which
# is why the cap lives in the shared tool layer, not in one adapter. The output is
# not small: a *tiny* geomeTRIC optimization (sto-3g water, 5 steps) prints ~1,800
# tokens of banner and per-iteration tables, and a real def2-TZVP saddle-point
# search prints several times that. At ~30 run_code calls per task, returning it
# all verbatim is what took a CH4 + .OH run to 1.53M input tokens, 99% re-sent.
#
# 12000 chars is ~3k tokens: enough to see a traceback or a results block whole,
# while refusing to let one chatty optimizer log ride along for the rest of the
# run. Head and tail are kept because the interesting parts of QM output live at
# both ends (what was set up; what it converged to).
MAX_TOOL_OUTPUT_CHARS = 12_000


def _clip(text: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    """Clip tool output to ``limit`` chars, keeping the head and the tail.

    QM logs are front- and back-loaded: the setup (geometry, basis, functional)
    is at the top and the answer (converged energy, final table, traceback) is at
    the bottom, with hundreds of lines of per-iteration noise between. Middle-out
    clipping keeps both ends and says plainly how much it dropped, so the model
    can tell the difference between "that is all of it" and "there was more".

    Arguments:
        text (str): The raw captured output.
        limit (int): Defaults to MAX_TOOL_OUTPUT_CHARS. Maximum characters kept.

    Returns:
        (str): ``text`` unchanged when short enough, else head + marker + tail.
    """
    if len(text) <= limit:
        return text

    half = limit // 2
    dropped = len(text) - 2 * half
    return (
        f'{text[:half]}\n'
        f'\n... [{dropped:,} characters clipped from the middle of this output. '
        f'Tool output is re-sent to the model on every later step, so print only '
        f'what you need -- set verbose=0, or write bulk output to a file in the '
        f'working directory and read back just the part you want.] ...\n\n'
        f'{text[-half:]}'
    )


class QMToolError(Exception):
    """A problem the caller can fix and retry: a missing prerequisite, an unknown
    artifact key, config the run never supplied.

    Harness-neutral on purpose. Each adapter translates it into the idiom its
    harness understands -- ``ModelRetry`` for pydantic-ai, ``ToolError`` for MCP
    -- both of which hand the message back to the model so it can correct itself.
    Anything raised that is *not* a ``QMToolError`` is a genuine fault and is left
    to propagate.
    """


def translate_qm_errors(fn: Callable[..., Awaitable[str]],
                        into: type[Exception]) -> Callable[..., Awaitable[str]]:
    """Wrap a toolkit method so a ``QMToolError`` re-raises as ``into``.

    The single seam both harnesses share: pydantic-ai passes ``ModelRetry`` and
    MCP passes ``ToolError``, each of which hands the message back to the model to
    correct itself. Only ``QMToolError`` is translated -- any other exception is a
    genuine fault and propagates untouched. ``functools.wraps`` keeps the method's
    signature and docstring so each harness still derives the tool's schema and
    description from the one definition in this module.
    """
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs) -> str:
        try:
            return await fn(*args, **kwargs)
        except QMToolError as e:
            raise into(str(e)) from e

    return wrapper


@dataclass
class QMRunState:
    """One parameterization run's state.

    Carries the academy ``Handle`` (how the tools reach the distributed
    ``QMAgent``), the run's output directory, and the run-level identifiers a
    model must not be allowed to hallucinate (residue name, AMBERHOME).
    Intermediate results are stashed in ``artifacts`` so callers pass short keys
    around instead of large structured objects.

    Concurrency. One run scope is shared across concurrent tool calls so a harness
    can fan out independent QM work (a basis/functional sweep, many torsion scans).
    That is safe because a tool touches this state only *between* its awaits, on the
    single event-loop thread: ``put``/``get`` contain no await, so key allocation
    can never interleave and two fanned-out steps always get distinct keys. The
    ``smiles``/``mol2_file``/``amber_config`` fields are the exception -- they are
    written once by the convergence steps (``build_compound`` at the start,
    ``integrate_amber_ff`` near the end) and only read by the steps that fan out, so
    the fan-out never races a write. Sequencing dependent steps is the driving
    harness's job; referencing an artifact that does not exist yet is a correctable
    error, not a corruption.
    """
    qm: Handle
    output_path: Path
    resname: str = 'LIG'
    amberhome: Path | None = None
    # Where run_code looks for the skills' helper scripts to put on the snippet's
    # import path. Kept here so it stays in step with whatever directory a harness
    # actually surfaces (the MCP server serves this same directory as resources).
    skills_root: Path = Path('./skills')
    smiles: str | None = None
    mol2_file: Path | None = None
    amber_config: AMBERConfig | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)  # id -> result object

    def put(self, kind: str, obj: Any) -> str:
        # No await here (nor in get): runs to completion on the event-loop thread,
        # so concurrent fanned-out tool calls allocate distinct keys without a lock.
        key = f'{kind}_{len([k for k in self.artifacts if k.startswith(kind)]) + 1}'
        self.artifacts[key] = obj
        return key

    def get(self, key: str, kind: type[T]) -> T:
        artifact = self.artifacts.get(key)
        if artifact is None:
            raise QMToolError(f'No artifact "{key}". Produce one first.')
        if not isinstance(artifact, kind):
            raise QMToolError(
                f'Artifact "{key}" is a {type(artifact).__name__}, but this step needs a '
                f'{kind.__name__}. Run the step that produces a {kind.__name__} first.'
            )

        return artifact


class QMToolkit:
    """The curated QM steps, bound to one run's state.

    Adapters register the bound methods directly (``toolkit.build_compound``),
    which keeps ``self`` out of the generated schema while preserving each
    method's signature and docstring as the tool's contract.
    """

    def __init__(self, state: QMRunState) -> None:
        self.state = state

    def tool_functions(self) -> tuple[Callable[..., Awaitable[str]], ...]:
        """The bound methods to expose, in the order a harness should see them.

        Adapters iterate this rather than introspecting the class, so adding a
        tool is a deliberate act and a helper method never leaks onto the wire.
        """
        return (
            self.run_code,
            self.build_compound,
            self.geometry_optimization,
            self.compute_esp,
            self.scan_torsions,
            self.fit_resp_charges,
            self.integrate_amber_ff,
            self.fit_torsions,
            self.run_parameterization_pipeline,
        )

    async def run_code(self, code: str, timeout: float = 1800.0) -> str:
        """Execute an arbitrary Python snippet on the QMAgent for bespoke analysis.

        The snippet runs in an isolated subprocess on the agent (sharing its
        scientific environment: rdkit, numpy, pyscf, ambertools, etc.), so a crash
        or hang in the code cannot take down the agent. Use this to sidestep the
        curated tools for one-off inspection, quick calculations, or glue work that
        no dedicated tool covers -- ``print`` whatever you need to see.

        The snippet runs with the run's output directory as its working directory, so
        relative paths resolve there and it can read artifacts produced by earlier
        steps. The qmagent package and the ``scripts/`` directories of the project
        skills are importable, so you can ``import qmagent...`` or reuse a skill's
        helper scripts (e.g. ``from run_dft import ...``) directly.

        Arguments:
            code (str): A multi-line Python snippet. Anything it prints to stdout is
                returned to you; if it raises, the traceback comes back in stderr.
            timeout (float): Defaults to 1800 (30 min). Wall-clock seconds before the
                snippet is killed. Raise it for genuinely long work -- a saddle-point
                search, a Hessian, a multi-point scan -- rather than trying to move
                the job off this tool; this is the only sandbox available, and there
                is no background execution to escape to.

        Returns:
            (str): The captured stdout and stderr, clipped to keep the head and tail
                if very long. On failure the Python traceback is in the STDERR
                section -- read it to see why the code failed and revise the snippet.
        """
        # Make the project skills' helper scripts importable from the snippet, so the
        # code-gen path can lean on the same vetted helpers the harness surfaces
        # (skills/<name>/scripts/*.py). Uses the run's skills_root so the import path
        # tracks the directory the harness actually serves, not a fixed guess.
        skills_root = self.state.skills_root
        extra_paths = sorted(
            p for p in skills_root.glob('*/scripts') if p.is_dir()
        ) if skills_root.is_dir() else []

        result = await self.state.qm.execute_code(
            code,
            workdir=self.state.output_path,
            extra_paths=extra_paths,
            timeout=timeout,
        )

        stdout = result.get('stdout', '')
        stderr = result.get('stderr', '')
        returncode = result.get('returncode', '')

        # returncode is the authoritative success signal: a raised exception exits
        # non-zero with its traceback on stderr. stderr alone is not treated as
        # failure -- libraries write benign warnings there on a clean (0) run.
        # Clip both paths: a failing snippet's stdout is just as capable of burying
        # the traceback (and the context) as a successful one's.
        if returncode not in ('', '0'):
            raise QMToolError(
                f'Code execution failed (returncode {returncode}).\n'
                f'--- STDOUT ---\n{_clip(stdout) or "(empty)"}\n'
                f'--- STDERR ---\n{_clip(stderr) or "(empty)"}\n'
                'Read the traceback above, fix the snippet, and try again.'
            )

        out = (f'Code executed successfully (returncode {returncode}).\n'
               f'--- STDOUT ---\n{_clip(stdout) or "(empty)"}')
        if stderr.strip():
            out += f'\n--- STDERR (warnings) ---\n{_clip(stderr)}'
        return out

    async def build_compound(self, smiles: str, max_iters: int = 2000) -> str:
        """Embed a SMILES string into a 3D conformer and write a model-compound mol2.

        Uses ETKDGv3 for conformer generation followed by MMFF refinement. The mol2 is
        written into the run's output directory under the configured residue name.

        Arguments:
            smiles (str): A valid SMILES string (RDKit will fail on invalid input).
            max_iters (int): Defaults to 2000. Max iterations of MMFF optimization.

        Returns:
            (str): Confirmation with the mol2 filename.
        """
        state = self.state
        mol2_file = state.output_path / f'{state.resname}.mol2'
        await state.qm.build_compound(
            smiles=smiles, mol2_file=mol2_file, resname=state.resname, max_iters=max_iters,
        )
        state.smiles = smiles
        state.mol2_file = mol2_file
        return f'Built {smiles} -> {mol2_file.name} (resname {state.resname})'

    async def geometry_optimization(self,
                                    stages: list[QMConfig],
                                    constraints: str | None = None,
                                    max_steps: int = 200) -> str:
        """Geometry-optimize the built compound through one or more QM stages.

        Each stage in ``stages`` is applied in order (e.g. a cheap pre-optimization
        followed by a higher-level refinement); the final stage's geometry and energy
        are returned.

        Arguments:
            stages (list[QMConfig]): Ordered optimization stages (basis/functional/etc).
            constraints (str | None): Optional path to a geomeTRIC constraints file.
            max_steps (int): Defaults to 200. Max optimizer steps per stage.

        Returns:
            (str): Artifact key, final energy and optimized xyz filename.
        """
        state = self.state
        if state.mol2_file is None:
            raise QMToolError('No compound has been built yet. Call build_compound first.')
        result = await state.qm.geometry_optimization(
            mol2_file=state.mol2_file,
            output_path=state.output_path,
            optimization_stages=stages,
            constraints=constraints,
            max_steps=max_steps,
        )
        gid = state.put('geomopt', result)
        return f'{gid}: optimized, final energy {result.energy:.6f} Ha, xyz={result.xyz_file.name}'

    async def compute_esp(self, geomopt_key: str, qm_config: QMConfig) -> str:
        """Compute the QM electrostatic potential (gas + solvent) on an MK grid.

        Operates on the optimized geometry referenced by ``geomopt_key``.

        Arguments:
            geomopt_key (str): Artifact key of a prior geometry_optimization result.
            qm_config (QMConfig): QM settings for the single-point ESP calculation.

        Returns:
            (str): Artifact key and a short description of the ESP calculations.
        """
        state = self.state
        geom = state.get(geomopt_key, GeomOptResult)
        molecule = XYZContents.from_xyz(geom.xyz_file)
        result = await state.qm.compute_ESP_charges(
            contents=molecule,
            output_path=state.output_path,
            qm_config=qm_config,
        )
        eid = state.put('esp', result)
        return f'{eid}: {len(result)} ESP calculations (gas + solvent) on the MK grid'

    async def scan_torsions(self,
                            geomopt_key: str,
                            qm_config: QMConfig,
                            torsions: list[Torsion],
                            scan_step: int = 15) -> str:
        """Relaxed QM dihedral scan over one or more rotatable bonds.

        Each torsion is a 0-indexed atom quartet (i, j, k, l); the j-k bond is rotated
        in ``scan_step`` degree increments through 360 degrees with constrained
        optimization at each point.

        Arguments:
            geomopt_key (str): Artifact key of a prior geometry_optimization result.
            qm_config (QMConfig): QM settings for each constrained optimization.
            torsions (list[tuple[int, int, int, int]]): Dihedral atom quartets to scan.
            scan_step (int): Defaults to 15. Angular step in degrees.

        Returns:
            (str): Artifact key, number of scans, and any incomplete torsions.
        """
        state = self.state
        geom = state.get(geomopt_key, GeomOptResult)
        molecule = XYZContents.from_xyz(geom.xyz_file)
        dataset, failed = await state.qm.scan_torsions(
            contents=molecule,
            qm_config=qm_config,
            output_dir=state.output_path,
            torsions=set(torsions),
            scan_step=scan_step,
        )
        sid = state.put('torsionscan', dataset)
        msg = f'{sid}: {len(dataset)} torsion scan(s) at {scan_step} deg steps'
        if failed:
            msg += f'; {len(failed)} incomplete (geometry failed mid-scan): {sorted(failed)}'
        return msg

    async def fit_resp_charges(self,
                               geomopt_key: str,
                               esp_key: str,
                               qm_config: QMConfig,
                               delta_resp2: float = 0.5) -> str:
        """Fit RESP2 charges from the gas- and solvent-phase ESP calculations.

        Interpolates as delta * q_solv + (1 - delta) * q_gas under the molecule's
        symmetry equivalences.

        Arguments:
            geomopt_key (str): Artifact key of the optimized geometry the ESP used.
            esp_key (str): Artifact key of a prior compute_esp result.
            qm_config (QMConfig): QM settings; supplies the net charge constraint.
            delta_resp2 (float): Defaults to 0.5. RESP2 solvent weighting delta.

        Returns:
            (str): Artifact key and the fitted net charge.
        """
        state = self.state
        if state.mol2_file is None:
            raise QMToolError('No compound has been built yet. Call build_compound first.')
        geom = state.get(geomopt_key, GeomOptResult)
        esp = state.get(esp_key, ESPResult)
        molecule = XYZContents.from_xyz(geom.xyz_file)
        result = await state.qm.fit_RESP_charges(
            molecule=molecule,
            mol2_file=state.mol2_file,
            esp_results=esp,
            qm_config=qm_config,
            delta_resp2=delta_resp2,
        )
        rid = state.put('resp', result)
        total = float(np.asarray(result.charges).sum())
        return f'{rid}: RESP2(delta={delta_resp2}) charges fit, total charge {total:+.4f} e'

    async def integrate_amber_ff(self, resp_key: str, charge: int | None = None) -> str:
        """Integrate the RESP2 charges into an AMBER/GAFF2 force field.

        Runs antechamber (GAFF2 atom typing, keeping our charges), parmchk2 (missing
        parameters) and tleap (residue build + lib/topology). The resulting AMBERConfig
        is stashed on the run state so fit_torsions can reuse the topology.

        Arguments:
            resp_key (str): Artifact key of a prior fit_resp_charges result.
            charge (int | None): Net residue charge. Defaults to the rounded RESP total.

        Returns:
            (str): Artifact key and the generated lib / frcmod / prmtop filenames.
        """
        state = self.state
        if state.mol2_file is None:
            raise QMToolError('No compound has been built yet. Call build_compound first.')
        if state.amberhome is None:
            raise QMToolError('AMBERHOME is not set for this run; cannot run the AmberTools pipeline.')
        resp = state.get(resp_key, RESPCharges)
        if charge is None:
            charge = int(round(float(np.asarray(resp.charges).sum())))

        # antechamber's inputs (the -cf charge file and the build-geometry sdf) are
        # written agent-side (where output_path lives) in a single round trip.
        resp_file = state.output_path / f'{state.resname}_resp.dat'
        sdf_file = state.output_path / f'{state.resname}.sdf'
        await state.qm.prepare_amber_inputs(
            charges=np.asarray(resp.charges).ravel().tolist(),
            mol2_file=state.mol2_file,
            charge_file=resp_file,
            sdf_file=sdf_file,
        )

        config = AMBERConfig(
            sdf_file=sdf_file,
            mol2_file=state.output_path / f'{state.resname}_gaff2.mol2',  # antechamber output (typed)
            frcmod_file=state.output_path / f'{state.resname}.frcmod',
            lib_files=state.output_path / f'{state.resname}.lib',
            resp_charges=resp_file,
            prmtop=state.output_path / f'{state.resname}.prmtop',
            amberhome=state.amberhome,
            resname=state.resname,
            charge=charge,
        )
        result = await state.qm.integrate_AMBER_ff(amber_config=config)
        state.amber_config = config
        aid = state.put('amber', result)
        return (f'{aid}: GAFF2 integration done (charge {charge:+d}); '
                f'lib={result.lib_file.name}, frcmod={result.frcmod_file.name}, '
                f'prmtop={result.prmtop.name}')

    async def fit_torsions(self, torsionscan_key: str, max_periodicity: int = 4) -> str:
        """Fit AMBER dihedral parameters to the QM torsion scans via paramfit.

        Requires that integrate_amber_ff has already run (it supplies the prmtop the
        fit reads atom types from).

        Arguments:
            torsionscan_key (str): Artifact key of a prior scan_torsions result.
            max_periodicity (int): Defaults to 4. Highest dihedral periodicity to fit.

        Returns:
            (str): Artifact key, number of fitted torsions and the refined frcmod path.
        """
        state = self.state
        if state.amber_config is None:
            raise QMToolError('No AMBER topology yet. Run integrate_amber_ff before fitting torsions.')
        scans = state.get(torsionscan_key, TorsionScanSet)
        result = await state.qm.fit_torsions(
            torsion_scans=scans,
            amber_config=state.amber_config,
            output_dir=state.output_path,
            max_periodicity=max_periodicity,
        )
        fid = state.put('torsionfit', result)
        n_fit = sum(1 for f in result.fits if f.frcmod_file is not None)
        return (f'{fid}: {n_fit}/{len(result.fits)} torsions fit; '
                f'refined frcmod={result.refined_frcmod.name}')

    async def run_parameterization_pipeline(self,
                                            smiles: str,
                                            optimization_stages: list[QMConfig],
                                            esp_config: QMConfig,
                                            scan_config: QMConfig,
                                            scan_step: int = 15,
                                            delta_resp2: float = 0.5,
                                            max_periodicity: int = 4,
                                            torsions: list[Torsion] | None = None) -> str:
        """Run the full parameterization pipeline end to end for one compound.

        Executes the fixed dependency chain deterministically:
        build -> geometry optimization -> ESP -> RESP2 -> torsion scan ->
        GAFF2 integration -> torsion fit. Rotatable torsions are auto-detected from the
        built geometry unless ``torsions`` is given. A full ``QMExperiment`` record is
        written to ``experiment.json`` in the output directory.

        Use this when the request is "parameterize compound X" and the QM settings are
        decided; use the individual tools when you need to inspect or branch between
        steps.

        Arguments:
            smiles (str): SMILES of the compound to parameterize.
            optimization_stages (list[QMConfig]): Ordered geometry-optimization stages.
            esp_config (QMConfig): QM settings for the ESP single points (also used as
                the charge reference for the RESP fit).
            scan_config (QMConfig): QM settings for the constrained torsion scans.
            scan_step (int): Defaults to 15. Torsion scan angular step (degrees).
            delta_resp2 (float): Defaults to 0.5. RESP2 solvent weighting delta.
            max_periodicity (int): Defaults to 4. Highest dihedral periodicity to fit.
            torsions (list[tuple[int, int, int, int]] | None): Explicit dihedral
                quartets to scan; auto-detected from rotatable bonds when omitted.

        Returns:
            (str): A multi-line summary of every produced artifact and output file.
        """
        state = self.state
        if state.amberhome is None:
            raise QMToolError('AMBERHOME is not set for this run; cannot run the AmberTools pipeline.')

        # build
        mol2_file = state.output_path / f'{state.resname}.mol2'
        await state.qm.build_compound(smiles=smiles, mol2_file=mol2_file, resname=state.resname)
        state.smiles = smiles
        state.mol2_file = mol2_file

        # geometry optimization
        geomopt = await state.qm.geometry_optimization(
            mol2_file=mol2_file,
            output_path=state.output_path,
            optimization_stages=optimization_stages,
        )
        molecule = XYZContents.from_xyz(geomopt.xyz_file)

        # electrostatic potential (gas + solvent)
        esp = await state.qm.compute_ESP_charges(
            contents=molecule,
            output_path=state.output_path,
            qm_config=esp_config,
        )

        # torsion scans (auto-detect rotatable bonds unless given)
        if torsions is None:
            torsions = sorted(await state.qm.find_rotatable_torsions(mol2_file))
        scans, failed = await state.qm.scan_torsions(
            contents=molecule,
            qm_config=scan_config,
            output_dir=state.output_path,
            torsions=set(torsions),
            scan_step=scan_step,
        )

        # RESP2 charges
        resp = await state.qm.fit_RESP_charges(
            molecule=molecule,
            mol2_file=mol2_file,
            esp_results=esp,
            qm_config=esp_config,
            delta_resp2=delta_resp2,
        )
        charge = int(round(float(np.asarray(resp.charges).sum())))

        # GAFF2 integration (charge file + sdf written agent-side via the handle)
        resp_file = state.output_path / f'{state.resname}_resp.dat'
        sdf_file = state.output_path / f'{state.resname}.sdf'
        await state.qm.prepare_amber_inputs(
            charges=np.asarray(resp.charges).ravel().tolist(),
            mol2_file=mol2_file,
            charge_file=resp_file,
            sdf_file=sdf_file,
        )
        amber_config = AMBERConfig(
            sdf_file=sdf_file,
            mol2_file=state.output_path / f'{state.resname}_gaff2.mol2',
            frcmod_file=state.output_path / f'{state.resname}.frcmod',
            lib_files=state.output_path / f'{state.resname}.lib',
            resp_charges=resp_file,
            prmtop=state.output_path / f'{state.resname}.prmtop',
            amberhome=state.amberhome,
            resname=state.resname,
            charge=charge,
        )
        amber_result = await state.qm.integrate_AMBER_ff(amber_config=amber_config)
        state.amber_config = amber_config

        # torsion fit
        fits = await state.qm.fit_torsions(
            torsion_scans=scans,
            amber_config=amber_config,
            output_dir=state.output_path,
            max_periodicity=max_periodicity,
        )

        experiment = QMExperiment(
            smiles=smiles,
            mol2_file=mol2_file,
            molecule=molecule,
            geometry_optimizations=[geomopt],
            electrostatic_potential=esp,
            resp_charges=resp,
            torsion_scan=scans,
        )
        experiment_json = state.output_path / 'experiment.json'
        experiment_json.write_text(experiment.model_dump_json(indent=2))

        n_fit = sum(1 for f in fits.fits if f.frcmod_file is not None)
        notes = []
        if failed:
            notes.append(f'{len(failed)} torsion scan(s) incomplete: {sorted(failed)}')
        if n_fit < len(fits.fits):
            notes.append(f'{len(fits.fits) - n_fit} torsion fit(s) failed')

        return (
            f'Parameterized {smiles} as residue {state.resname}.\n'
            f'  final energy : {geomopt.energy:.6f} Ha\n'
            f'  net charge   : {charge:+d} e\n'
            f'  torsions     : {n_fit}/{len(fits.fits)} fit ({len(scans)} scanned)\n'
            f'  lib          : {amber_result.lib_file}\n'
            f'  frcmod       : {amber_result.frcmod_file}\n'
            f'  refined frcmod: {fits.refined_frcmod}\n'
            f'  prmtop/inpcrd: {amber_result.prmtop} / {amber_result.inpcrd}\n'
            f'  experiment   : {experiment_json}\n'
            + (f'  notes        : {"; ".join(notes)}\n' if notes else '')
        )
