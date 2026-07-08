import numpy as np
from academy.handle import Handle
from dataclasses import dataclass, field
from pathlib import Path
from pydantic import BaseModel
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai import ModelSettings, ModelRetry, RunContext
from pydantic_ai.capabilities import Thinking, ToolSearch, WebSearch
from pydantic_ai_backends import ConsoleCapability, LocalBackend, ensure_async
from pydantic_ai_backends.permissions import READONLY_RULESET
from pydantic_ai_shields import CostTracking, InputGuard, SecretRedaction, ToolGuard
from pydantic_ai_skills import SkillsCapability
from pydantic_ai_summarization import ContextManagerCapability
from pydantic_ai_todo import TodoCapability, AsyncMemoryStorage
from pydantic_deep import MemoryCapability, StuckLoopDetection
from subagents_pydantic_ai import SubAgentCapability, SubAgentConfig
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

T = TypeVar("T")
Torsion = tuple[int, int, int, int]

# Context-window threshold at which the summarization capability compacts history.
# Distinct from ModelSettings.max_tokens, which caps a single response's output.
context_max_tokens = 120_000


@dataclass
class QMDeps:
    """Run-scoped state injected into every tool.

    Carries the academy ``Handle`` (how tools reach the distributed ``QMAgent``),
    the run's output directory, and run-level identifiers/config that the model
    must not be allowed to hallucinate (residue name, AMBERHOME). Intermediate
    results are stashed in ``artifacts`` so the model passes around short keys
    instead of large structured objects.
    """
    qm: Handle
    output_path: Path
    resname: str = 'LIG'
    amberhome: Path | None = None
    smiles: str | None = None
    mol2_file: Path | None = None
    amber_config: AMBERConfig | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)  # id -> result object
    # Filesystem backend shared by ConsoleCapability (grep/glob/ls/read) and
    # MemoryCapability. Must be async: the memory toolset awaits read_bytes/read
    # directly, while the console toolset wraps with ensure_async either way.
    # Read-only enforcement lives in the capability's READONLY_RULESET, not here.
    backend: Any = field(default_factory=lambda: ensure_async(LocalBackend()))

    def put(self, kind: str, obj: Any) -> str:
        key = f'{kind}_{len([k for k in self.artifacts if k.startswith(kind)]) + 1}'
        self.artifacts[key] = obj
        return key

    def get(self, key: str, kind: type[T]) -> T:
        artifact = self.artifacts.get(key)
        if artifact is None:
            raise ModelRetry(f'No artifact "{key}". Produce one first.')
        if not isinstance(artifact, kind):
            raise ModelRetry(
                f'Artifact "{key}" is a {type(artifact).__name__}, but this step needs a '
                f'{kind.__name__}. Run the step that produces a {kind.__name__} first.'
            )

        return artifact


class ParameterizationSummary(BaseModel):
    """Final deliverables of a parameterization run."""
    resname: str
    smiles: str
    final_energy_ha: float
    net_charge: int
    n_torsions_fit: int
    lib_file: Path
    frcmod_file: Path
    refined_frcmod: Path
    prmtop: Path
    experiment_json: Path
    notes: str = ''


model = 'openai:gpt-5.5'

system_prompt = (
    'You are a computational chemist responsible for parameterizing novel biomolecules '
    'and post translational modifications of amino acids, nucleic acids and other such species. '
    'You have access to a suite of modern QM tools and workflows, utilizing the pyscf and gpu4pyscf '
    'ecosystems, as well as python libraries including but not limited to rdkit, openbabel and ambertools.'
)

research_subagent_prompt = (
    'You are a thorough researcher that has deep expertise in '
    'chemistry. You have strong literature parsing and synthesis '
    'skills and are able to identify the optimal quantum chemistry '
    'workflows and pipelines based on previous experiments reported '
    'in the scientific literature.'
)

orchestrator = PydanticAgent[QMDeps, ParameterizationSummary](
    model,
    deps_type=QMDeps,
    output_type=ParameterizationSummary,
    model_settings=ModelSettings(temperature=0.8, max_tokens=10000),
    instructions=system_prompt,  # instructions (not system_prompt) so only the current agent's prompt reaches the model
    capabilities=[
        ToolSearch(),
        Thinking('xhigh'),
        ContextManagerCapability(max_tokens=context_max_tokens),
        WebSearch(),
        ConsoleCapability(permissions=READONLY_RULESET),  # grep, glob, ls, read
        MemoryCapability(agent_name='quantum-agent'),
        SkillsCapability(directories=['./skills']),
        SubAgentCapability(subagents=[
            SubAgentConfig(
                name='researcher',
                description='Deep research on a topic',
                instructions=research_subagent_prompt
            ),
        ]),
        TodoCapability(enable_subtasks=True, async_storage=AsyncMemoryStorage()),
        InputGuard(guard=lambda p: 'ignore previous instructions' not in p.lower()),
        ToolGuard(blocked=['rm']),
        SecretRedaction(),
        StuckLoopDetection(),
    ]
)


# --------------------------------------------------------------------------- #
# Tools — one per QMAgent @action. Each returns a short summary string (and an
# artifact key where it produces a result) so the model can chain steps without
# echoing large structured objects.
# --------------------------------------------------------------------------- #

@orchestrator.tool
async def run_code(ctx: RunContext[QMDeps], code: str) -> str:
    """"""
    if dangerous_pattern := ctx.deps.qm.examine_code(code):
        return dangerous_pattern

    return_str = await ctx.deps.qm.execute_code(code)

    if 'Traceback' in return_str:
        pass

    return return_str

@orchestrator.tool
async def build_compound(ctx: RunContext[QMDeps], smiles: str, max_iters: int = 2000) -> str:
    """Embed a SMILES string into a 3D conformer and write a model-compound mol2.

    Uses ETKDGv3 for conformer generation followed by MMFF refinement. The mol2 is
    written into the run's output directory under the configured residue name.

    Arguments:
        smiles (str): A valid SMILES string (RDKit will fail on invalid input).
        max_iters (int): Defaults to 2000. Max iterations of MMFF optimization.

    Returns:
        (str): Confirmation with the mol2 filename.
    """
    deps = ctx.deps
    mol2_file = deps.output_path / f'{deps.resname}.mol2'
    await deps.qm.build_compound(smiles=smiles, mol2_file=mol2_file, resname=deps.resname, max_iters=max_iters)
    deps.smiles = smiles
    deps.mol2_file = mol2_file
    return f'Built {smiles} -> {mol2_file.name} (resname {deps.resname})'


@orchestrator.tool
async def geometry_optimization(ctx: RunContext[QMDeps],
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
    deps = ctx.deps
    if deps.mol2_file is None:
        raise ModelRetry('No compound has been built yet. Call build_compound first.')
    result = await deps.qm.geometry_optimization(
        mol2_file=deps.mol2_file,
        output_path=deps.output_path,
        optimization_stages=stages,
        constraints=constraints,
        max_steps=max_steps,
    )
    gid = deps.put('geomopt', result)
    return f'{gid}: optimized, final energy {result.energy:.6f} Ha, xyz={result.xyz_file.name}'


@orchestrator.tool
async def compute_esp(ctx: RunContext[QMDeps], geomopt_key: str, qm_config: QMConfig) -> str:
    """Compute the QM electrostatic potential (gas + solvent) on an MK grid.

    Operates on the optimized geometry referenced by ``geomopt_key``.

    Arguments:
        geomopt_key (str): Artifact key of a prior geometry_optimization result.
        qm_config (QMConfig): QM settings for the single-point ESP calculation.

    Returns:
        (str): Artifact key and a short description of the ESP calculations.
    """
    deps = ctx.deps
    geom = deps.get(geomopt_key, GeomOptResult)
    molecule = XYZContents.from_xyz(geom.xyz_file)
    result = await deps.qm.compute_ESP_charges(
        contents=molecule,
        output_path=deps.output_path,
        qm_config=qm_config,
    )
    eid = deps.put('esp', result)
    return f'{eid}: {len(result)} ESP calculations (gas + solvent) on the MK grid'


@orchestrator.tool
async def scan_torsions(ctx: RunContext[QMDeps],
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
    deps = ctx.deps
    geom = deps.get(geomopt_key, GeomOptResult)
    molecule = XYZContents.from_xyz(geom.xyz_file)
    dataset, failed = await deps.qm.scan_torsions(
        contents=molecule,
        qm_config=qm_config,
        output_dir=deps.output_path,
        torsions=set(torsions),
        scan_step=scan_step,
    )
    sid = deps.put('torsionscan', dataset)
    msg = f'{sid}: {len(dataset)} torsion scan(s) at {scan_step} deg steps'
    if failed:
        msg += f'; {len(failed)} incomplete (geometry failed mid-scan): {sorted(failed)}'
    return msg


@orchestrator.tool
async def fit_resp_charges(ctx: RunContext[QMDeps],
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
    deps = ctx.deps
    if deps.mol2_file is None:
        raise ModelRetry('No compound has been built yet. Call build_compound first.')
    geom = deps.get(geomopt_key, GeomOptResult)
    esp = deps.get(esp_key, ESPResult)
    molecule = XYZContents.from_xyz(geom.xyz_file)
    result = await deps.qm.fit_RESP_charges(
        molecule=molecule,
        mol2_file=deps.mol2_file,
        esp_results=esp,
        qm_config=qm_config,
        delta_resp2=delta_resp2,
    )
    rid = deps.put('resp', result)
    total = float(np.asarray(result.charges).sum())
    return f'{rid}: RESP2(delta={delta_resp2}) charges fit, total charge {total:+.4f} e'


@orchestrator.tool
async def integrate_amber_ff(ctx: RunContext[QMDeps],
                             resp_key: str,
                             charge: int | None = None) -> str:
    """Integrate the RESP2 charges into an AMBER/GAFF2 force field.

    Runs antechamber (GAFF2 atom typing, keeping our charges), parmchk2 (missing
    parameters) and tleap (residue build + lib/topology). The resulting AMBERConfig
    is stashed on deps so fit_torsions can reuse the topology.

    Arguments:
        resp_key (str): Artifact key of a prior fit_resp_charges result.
        charge (int | None): Net residue charge. Defaults to the rounded RESP total.

    Returns:
        (str): Artifact key and the generated lib / frcmod / prmtop filenames.
    """
    deps = ctx.deps
    if deps.mol2_file is None:
        raise ModelRetry('No compound has been built yet. Call build_compound first.')
    if deps.amberhome is None:
        raise ModelRetry('QMDeps.amberhome is not set; cannot run the AmberTools pipeline.')
    resp = deps.get(resp_key, RESPCharges)
    if charge is None:
        charge = int(round(float(np.asarray(resp.charges).sum())))

    # antechamber's inputs (the -cf charge file and the build-geometry sdf) are
    # written agent-side (where deps.output_path lives) in a single round trip.
    resp_file = deps.output_path / f'{deps.resname}_resp.dat'
    sdf_file = deps.output_path / f'{deps.resname}.sdf'
    await deps.qm.prepare_amber_inputs(
        charges=np.asarray(resp.charges).ravel().tolist(),
        mol2_file=deps.mol2_file,
        charge_file=resp_file,
        sdf_file=sdf_file,
    )

    config = AMBERConfig(
        sdf_file=sdf_file,
        mol2_file=deps.output_path / f'{deps.resname}_gaff2.mol2',  # antechamber output (typed)
        frcmod_file=deps.output_path / f'{deps.resname}.frcmod',
        lib_files=deps.output_path / f'{deps.resname}.lib',
        resp_charges=resp_file,
        prmtop=deps.output_path / f'{deps.resname}.prmtop',
        amberhome=deps.amberhome,
        resname=deps.resname,
        charge=charge,
    )
    result = await deps.qm.integrate_AMBER_ff(amber_config=config)
    deps.amber_config = config
    aid = deps.put('amber', result)
    return (f'{aid}: GAFF2 integration done (charge {charge:+d}); '
            f'lib={result.lib_file.name}, frcmod={result.frcmod_file.name}, '
            f'prmtop={result.prmtop.name}')


@orchestrator.tool
async def fit_torsions(ctx: RunContext[QMDeps],
                       torsionscan_key: str,
                       max_periodicity: int = 4) -> str:
    """Fit AMBER dihedral parameters to the QM torsion scans via paramfit.

    Requires that integrate_amber_ff has already run (it supplies the prmtop the
    fit reads atom types from).

    Arguments:
        torsionscan_key (str): Artifact key of a prior scan_torsions result.
        max_periodicity (int): Defaults to 4. Highest dihedral periodicity to fit.

    Returns:
        (str): Artifact key, number of fitted torsions and the refined frcmod path.
    """
    deps = ctx.deps
    if deps.amber_config is None:
        raise ModelRetry('No AMBER topology yet. Run integrate_amber_ff before fitting torsions.')
    scans = deps.get(torsionscan_key, TorsionScanSet)
    result = await deps.qm.fit_torsions(
        torsion_scans=scans,
        amber_config=deps.amber_config,
        output_dir=deps.output_path,
        max_periodicity=max_periodicity,
    )
    fid = deps.put('torsionfit', result)
    n_fit = sum(1 for f in result.fits if f.frcmod_file is not None)
    return (f'{fid}: {n_fit}/{len(result.fits)} torsions fit; '
            f'refined frcmod={result.refined_frcmod.name}')


@orchestrator.tool
async def run_parameterization_pipeline(ctx: RunContext[QMDeps],
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
    deps = ctx.deps
    if deps.amberhome is None:
        raise ModelRetry('QMDeps.amberhome is not set; cannot run the AmberTools pipeline.')

    # build
    mol2_file = deps.output_path / f'{deps.resname}.mol2'
    await deps.qm.build_compound(smiles=smiles, mol2_file=mol2_file, resname=deps.resname)
    deps.smiles = smiles
    deps.mol2_file = mol2_file

    # geometry optimization
    geomopt = await deps.qm.geometry_optimization(
        mol2_file=mol2_file,
        output_path=deps.output_path,
        optimization_stages=optimization_stages,
    )
    molecule = XYZContents.from_xyz(geomopt.xyz_file)

    # electrostatic potential (gas + solvent)
    esp = await deps.qm.compute_ESP_charges(
        contents=molecule,
        output_path=deps.output_path,
        qm_config=esp_config,
    )

    # torsion scans (auto-detect rotatable bonds unless given)
    if torsions is None:
        torsions = sorted(await deps.qm.find_rotatable_torsions(mol2_file))
    scans, failed = await deps.qm.scan_torsions(
        contents=molecule,
        qm_config=scan_config,
        output_dir=deps.output_path,
        torsions=set(torsions),
        scan_step=scan_step,
    )

    # RESP2 charges
    resp = await deps.qm.fit_RESP_charges(
        molecule=molecule,
        mol2_file=mol2_file,
        esp_results=esp,
        qm_config=esp_config,
        delta_resp2=delta_resp2,
    )
    charge = int(round(float(np.asarray(resp.charges).sum())))

    # GAFF2 integration (charge file + sdf written agent-side via the handle)
    resp_file = deps.output_path / f'{deps.resname}_resp.dat'
    sdf_file = deps.output_path / f'{deps.resname}.sdf'
    await deps.qm.prepare_amber_inputs(
        charges=np.asarray(resp.charges).ravel().tolist(),
        mol2_file=mol2_file,
        charge_file=resp_file,
        sdf_file=sdf_file,
    )
    amber_config = AMBERConfig(
        sdf_file=sdf_file,
        mol2_file=deps.output_path / f'{deps.resname}_gaff2.mol2',
        frcmod_file=deps.output_path / f'{deps.resname}.frcmod',
        lib_files=deps.output_path / f'{deps.resname}.lib',
        resp_charges=resp_file,
        prmtop=deps.output_path / f'{deps.resname}.prmtop',
        amberhome=deps.amberhome,
        resname=deps.resname,
        charge=charge,
    )
    amber_result = await deps.qm.integrate_AMBER_ff(amber_config=amber_config)
    deps.amber_config = amber_config

    # torsion fit
    fits = await deps.qm.fit_torsions(
        torsion_scans=scans,
        amber_config=amber_config,
        output_dir=deps.output_path,
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
    experiment_json = deps.output_path / 'experiment.json'
    experiment_json.write_text(experiment.model_dump_json(indent=2))

    n_fit = sum(1 for f in fits.fits if f.frcmod_file is not None)
    notes = []
    if failed:
        notes.append(f'{len(failed)} torsion scan(s) incomplete: {sorted(failed)}')
    if n_fit < len(fits.fits):
        notes.append(f'{len(fits.fits) - n_fit} torsion fit(s) failed')

    return (
        f'Parameterized {smiles} as residue {deps.resname}.\n'
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
