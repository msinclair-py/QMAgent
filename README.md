# QMAgent

[![CI](https://github.com/msinclair-py/QMAgent/actions/workflows/ci.yml/badge.svg)](https://github.com/msinclair-py/QMAgent/actions/workflows/ci.yml)

Agentic harness for constructing quantum-chemistry (QM) workflows, distributing
them across HPC resources, and automatically exploring protocols. Its primary
target is **parameterizing novel biomolecules — post-translational modifications,
non-canonical residues, small-molecule ligands — for the AMBER molecular-dynamics
force field**, but the same building blocks (plus an LLM-driven code-execution
escape hatch) generalize to arbitrary QM calculations.

The QM engine is [PySCF](https://pyscf.org/) and its GPU-accelerated drop-in
[gpu4pyscf](https://github.com/pyscf/gpu4pyscf); cheminformatics is
[RDKit](https://www.rdkit.org/); force-field integration is
[AmberTools](https://ambermd.org/AmberTools.php).

---

## What it does

Given a SMILES string, QMAgent runs the standard QM-derived RESP/GAFF2
parameterization pipeline end to end and emits AMBER-ready force-field files
(`.lib`, `.frcmod`, `.prmtop`/`.inpcrd`) plus a full `experiment.json` record:

```
SMILES
  │  build_compound          RDKit ETKDGv3 embed + MMFF  ──▶ mol2
  ▼
geometry_optimization        PySCF/gpu4pyscf DFT (staged)  ──▶ optimized xyz + energy
  │
  ├─▶ compute_esp            ESP on a Merz–Kollman grid, gas + C-PCM solvent
  │     └─▶ fit_resp_charges RESP2(δ) two-stage fit (δ·q_solv + (1−δ)·q_gas)
  │
  └─▶ scan_torsions          relaxed constrained dihedral scans (per rotatable bond)
        │
        ▼
integrate_amber_ff           antechamber (GAFF2 typing, our charges) → parmchk2 → tleap
        │
        ▼
fit_torsions                 paramfit fits Vₙ/phase to the QM torsion surface ──▶ refined frcmod
```

Each step is exposed both as an **individual tool** (inspect/branch between
steps) and as a single deterministic **`run_parameterization_pipeline`** tool
("parameterize compound X" when the QM settings are already decided).

---

## Architecture

QMAgent is three layers, and the top one is swappable.

**1. Harness — pick one.** Something has to decide *which* QM step to run next
and with what settings. That job is either ours or someone else's:

* **Self-managed** (`llm_interface.py`, `main.py`) — a
  [`pydantic-ai`](https://ai.pydantic.dev/) agent playing "computational chemist",
  equipped with a capability stack (tool search, extended thinking, context
  summarization, web search, a read-only filesystem console, persistent memory, a
  `researcher` subagent, a TODO planner, input/tool/secret shields) and returning a
  typed `ParameterizationSummary`.
* **Externally managed** (`mcp_server.py`) — an
  [MCP](https://modelcontextprotocol.io) server exposing the same tools to a
  commercial or open-source harness (Claude Code, Codex, …), which brings its own
  model, context management and UI. This path imports no pydantic-ai at all: no
  orchestrator, no model config, no API key.

**2. QM tools** (`src/qmagent/tools.py`)
`QMToolkit` — one method per QM step, dispatching to the distributed agent and
returning a short summary string while stashing large results as keyed *artifacts*
on the run state, so a model chains steps by passing short keys rather than echoing
big structured objects. It imports neither harness; both adapt the same bound
methods, deriving identical schemas and descriptions from these signatures and
docstrings. A tool is defined, documented and fixed exactly once.

**3. Distributed execution** (`src/qmagent/agents/`, `src/qmagent/utils/`)
An [`academy`](https://github.com/proxystore/academy) `QMAgent` whose `@action`
methods dispatch [`parsl`](https://parsl-project.org/) `@python_app`s. Parsl
routes work to two executor labels — **`gpu`** (PySCF/gpu4pyscf DFT: geometry
optimization, ESP, torsion scans) and **`cpu`** (RDKit build, RESP fitting,
AmberTools) — so a single run fans QM tasks across a cluster while the CPU-bound
glue runs alongside. On a laptop the same labels map to local thread pools
(see *Local vs. HPC* below).

```
   self-managed                        externally managed
pydantic-ai agent                 Claude Code / Codex / …
  (llm_interface)                          │  MCP (stdio or http)
        │                                  ▼
        │                            mcp_server.py
        │                                  │
        └──────────────┬───────────────────┘
                       ▼
             QMToolkit  (tools.py)
                       │  academy Handle
                       ▼
              QMAgent (academy Agent)
                       │  parsl @python_app
        ├── gpu executor ── PySCF / gpu4pyscf   (geomopt, ESP, torsion scan)
        └── cpu executor ── RDKit, RESP fit, AmberTools (antechamber/parmchk2/tleap/paramfit)
```

---

## Installation

The Python package and its orchestration dependencies install with
[`uv`](https://docs.astral.sh/uv/):

```bash
uv sync                     # runtime deps
uv sync --extra dev         # + pytest
```

Requires **Python ≥ 3.12**.

### Runtime scientific stack (provided by the environment)

The heavy QM/MM engines are **deliberately not declared in `pyproject.toml`** —
they are platform-specific (CUDA builds, conda-only packages, site module
systems) and expected to be present in the environment the `QMAgent` runs in:

| Component            | Provides                                   | Notes |
|----------------------|--------------------------------------------|-------|
| `pyscf`              | CPU quantum chemistry                      | pip/conda |
| `gpu4pyscf` + CUDA   | GPU-accelerated DFT (the `gpu` executor)   | required for the GPU path |
| `geometric`          | geometry optimizer used by PySCF           | pip |
| `parmed`             | reads GAFF2 atom types from the prmtop     | pip/conda |
| **AmberTools**       | antechamber, parmchk2, tleap, paramfit     | conda (`conda install -c conda-forge ambertools`) |

Set `AMBERHOME` to your AmberTools install root before running the AMBER steps:

```bash
export AMBERHOME=/path/to/amber          # binaries are read from $AMBERHOME/bin
```

> The pure-Python logic (RESP fitter, MK grid, file I/O, data models) and the
> test suite need none of the above — only `numpy`, `scipy`, `rdkit`, `pydantic`.

---

## Quickstart

Parameterize a single compound (hosted Globus exchange, GPU agent):

```bash
export AMBERHOME=/path/to/amber
uv run python -m qmagent.main
```

`main.py` launches a `QMAgent`, binds a `QMToolkit` to it, then asks the
orchestrator to *"generate parameters for this compound: CCCCCC"*, writing results
under `./qm_output/`. Edit the SMILES / residue name there, or drive the
orchestrator from your own script:

```python
toolkit = QMToolkit(QMRunState(qm=qm_handle, output_path=Path('./qm_output'),
                               resname='LIG', amberhome=Path(os.environ['AMBERHOME'])))
result = await orchestrator.run(prompt, deps=QMDeps(), toolsets=[qm_toolset(toolkit)])
```

### Under an external harness (MCP)

To hand the driving over to Claude Code, Codex or any other MCP client instead,
run the execution layer as an MCP server. The same nine tools, plus the chemist
prompt our own agent uses and the `skills/` directory as readable resources:

```bash
uv run python -m qmagent.mcp_server                          # stdio
uv run python -m qmagent.mcp_server --transport http --port 8000
uv run python -m qmagent.mcp_server --cpu                    # CPU pyscf, no GPU
```

One server process is one run scope, so it is configured at startup rather than
per call — an MCP client spawns the server as a subprocess and reaches it through
`env`/`args`:

```json
{"mcpServers": {"qmagent": {
  "command": "uv",
  "args": ["run", "python", "-m", "qmagent.mcp_server"],
  "env": {"QMAGENT_OUTPUT": "./qm_output",
          "QMAGENT_RESNAME": "LIG",
          "AMBERHOME": "/path/to/amber"}}}}
```

| variable | meaning | default |
|----------|---------|---------|
| `QMAGENT_OUTPUT`   | run output directory                       | `./qm_output` |
| `QMAGENT_RESNAME`  | residue name / output basename             | `LIG` |
| `AMBERHOME`        | AmberTools root (AMBER steps error without) | — |
| `QMAGENT_GPU`      | `0` to import CPU pyscf instead of gpu4pyscf | on |
| `QMAGENT_THREADS`  | agent worker threads                        | `os.cpu_count()` |
| `QMAGENT_EXCHANGE` | `local`, or an academy exchange http(s) URL | `local` |
| `QMAGENT_SKILLS`   | skills directory served as resources        | `./skills` |

Every flag has a `--flag` equivalent (`--output`, `--resname`, `--cpu`, …) that
overrides the environment; see `--help`.

### Live demo — CPU only, no GPU, no AmberTools, no API key

`qmagent.demo` runs the **quantum core** (build → geometry optimization → ESP →
RESP2 charges) on plain CPU PySCF and prints the fitted partial charges. It
drives the `QMAgent` actions directly through a local in-process exchange, so it
needs neither an LLM/API key nor a GPU nor AmberTools — only `pyscf` + `geometric`:

```bash
pip install pyscf geometric          # no gpu4pyscf / CUDA required
uv run python -m qmagent.demo                      # methanol (default)
uv run python -m qmagent.demo --smiles CC(=O)NC    # N-methylacetamide
```

Under the hood this is `QMAgent(use_gpu=False)`, which swaps the `gpu4pyscf`
import for CPU `pyscf`; the ESP grid, two-stage RESP fit and symmetry handling
are identical to the GPU path. Keep molecules small and the basis modest for a
snappy live run. The LLM-driven reference ladder can also run CPU-only with the
`--cpu` flag (below).

### The reference ladder

`test_systems.py` runs a ladder of small model compounds with **known published
AMBER parameters** (from `Forcefield_PTM`), so the agent's charges, GAFF2 types
and torsion barriers are checkable against literature. It uses a local in-process
`academy` exchange (no Globus needed):

```bash
uv run python -m qmagent.test_systems              # HEX, NMA, ALY, SEP, M3L
uv run python -m qmagent.test_systems --only NMA   # a subset by resname
```

| resname | compound                     | charge | exercises |
|---------|------------------------------|:------:|-----------|
| HEX     | hexane                       |  0     | plumbing smoke test |
| NMA     | N-methylacetamide            |  0     | smallest real chemistry (one amide) |
| ALY     | acetyl-lysine side chain     |  0     | neutral PTM, several rotatable bonds |
| SEP     | methylphosphate (pSer model) | −2     | charged path + phosphate typing |
| M3L     | trimethyllysine side chain   | +1     | cationic path, symmetric methyls |

---

## Local vs. HPC execution

**Local (default).** `QMAgent` with no `parsl_config` loads a local thread-pool
config exposing `cpu` and `gpu` labels (both CPU thread pools). The `gpu`-labelled
apps still require `gpu4pyscf` + CUDA at import, so on a non-GPU host the GPU
steps fail at import time — the CPU-only helpers and tests run fine.

**HPC / multi-GPU.** Build a parsl config from
`qmagent.utils.parsl_settings.HeterogeneousSettings` and pass it to the agent.
It provisions `HighThroughputExecutor`s with an MPI launcher, one worker per GPU
via `available_accelerators`, and CPU-affinity binding:

```python
from qmagent.utils.parsl_settings import HeterogeneousSettings
from qmagent.agents.qm_agent import QMAgent

settings = HeterogeneousSettings(available_accelerators=4, worker_init="module load cuda; ...")
agent = QMAgent(num_threads=8, parsl_config=settings.config_factory(run_dir))
```

Settings round-trip to/from YAML (`dump_yaml` / `from_yaml`) for reproducible
deployments.

---

## Skills

`skills/` holds curated, model-loadable domain knowledge — surfaced to the
self-managed orchestrator via the skills capability, served to an external
harness as `skill://` MCP resources, and reusable from generated code either way:

- **`skills/pyscf/`** — the PySCF / gpu4pyscf workflow this project relies on:
  molecule construction, DFT/SCF, geometry optimization & constrained torsion
  scans, ESP-on-a-grid for RESP, PCM solvent, and CPU↔GPU switching — with the
  unit (Bohr/Å), spin/charge, and dispersion gotchas spelled out.
- **`skills/rdkit/`** — molecular I/O & parsing, SMARTS substructure search, and
  2D/3D coordinate/conformer generation.

Each ships reference docs and runnable helper scripts under `references/` and
`scripts/`.

---

## Testing

```bash
uv run pytest
```

The suite (`tests/`) exercises the deterministic, dependency-light logic —
file parsing/writing, the pydantic data models, the two-stage RESP fitter, the
`run_code` subprocess sandbox, and the static `QMAgent` helpers (MK grid,
symmetry classes, frcmod merging). It deliberately avoids PySCF/GPU, AmberTools,
the LLM orchestrator and the parsl/academy runtime, so it runs anywhere.

---

## Project layout

```
src/qmagent/
  main.py                 single-compound entry point (hosted exchange)
  test_systems.py         reference ladder with known parameters
  tools.py                QMToolkit: the QM tools, shared by both harnesses
  llm_interface.py        self-managed harness: pydantic-ai orchestrator
  mcp_server.py           externally managed harness: MCP server (no pydantic-ai)
  prompts.py              chemist framing shared by both harnesses
  agents/
    qm_agent.py           academy QMAgent: @action per pipeline step + helpers
    distributed.py        parsl @python_app QM kernels (PySCF/gpu4pyscf, RESP)
    amber_apps.py         antechamber / parmchk2 / tleap / paramfit wrappers
    resp_fitter.py        two-stage RESP charge fitter (SLSQP + restraints)
  utils/
    pydantic_models.py    typed results (QMConfig, ESPResult, TorsionScanSet, …)
    file_ops.py           xyz/mol2/sdf/charge-file I/O, NDArray JSON round-trip
    parsl_settings.py     HPC parsl config factory (HeterogeneousSettings)
skills/                   loadable PySCF and RDKit skills
tests/                    dependency-light unit tests
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
