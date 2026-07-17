"""FastMCP server exposing the QM execution layer to an externally managed harness.

Run this and the curated QM tools -- plus the code-execution escape hatch --
become available to any MCP client (Claude Code, Codex, ...), which then plays the
orchestrator role that ``llm_interface``'s pydantic-ai agent plays in the
self-managed path. The two are alternatives, not layers: **nothing in this module
imports pydantic-ai**, so an externally managed run never constructs the
orchestrator, needs no model configuration and needs no API key. Keep it that way.

The tools themselves live in ``tools.QMToolkit`` and are shared verbatim with the
self-managed harness -- this module only adapts them (errors, stdout) and owns the
process lifecycle.

Beyond tools, the server also serves the domain context an external harness would
otherwise lack: the chemist ``instructions``/prompt our own agent runs with, and
the project ``skills/`` as readable resources.

Configuration comes from the environment or flags, because an MCP client launches
the server as a subprocess and can only reach it through argv/env:

    QMAGENT_OUTPUT     run output directory            (default ./qm_output)
    QMAGENT_RESNAME    residue name / output basename  (default LIG)
    AMBERHOME          AmberTools root; the AMBER steps error without it
    QMAGENT_GPU        0 to import CPU pyscf instead of gpu4pyscf
    QMAGENT_THREADS    agent worker threads            (default os.cpu_count())
    QMAGENT_EXCHANGE   'local' (default) or an academy exchange http(s) URL
    QMAGENT_SKILLS     skills directory to serve       (default ./skills)

One server process is one run scope: output directory and residue name are fixed
at startup (one compound per server), matching the way a client spawns a server
per session. Within that scope tool calls run concurrently, so a harness can fan
out independent QM work -- a basis/functional sweep of geometry optimizations, or
many torsion scans -- and the run state stays consistent (see ``tools.QMRunState``).
Ordering dependent steps correctly is the driving harness's job; a step that
references an artifact not produced yet gets a correctable error back.

Examples
--------
    uv run python -m qmagent.mcp_server                          # stdio
    uv run python -m qmagent.mcp_server --transport http --port 8000
    uv run python -m qmagent.mcp_server --cpu --resname LIG      # CPU-only pyscf

Registering with an MCP client (Claude Code's ``.mcp.json``)::

    {"mcpServers": {"qmagent": {
      "command": "uv",
      "args": ["run", "python", "-m", "qmagent.mcp_server"],
      "env": {"QMAGENT_OUTPUT": "./qm_output", "AMBERHOME": "/opt/amber"}}}}
"""

import argparse
import os
import sys
from academy.exchange import HttpExchangeFactory, LocalExchangeFactory
from academy.manager import Manager
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.prompts import Prompt
from fastmcp.resources import FileResource
from pathlib import Path
from typing import Any

from .agents.qm_agent import QMAgent
from .prompts import SYSTEM_PROMPT
from .tools import QMRunState, QMToolkit, translate_qm_errors

SERVER_NAME = 'qmagent'

# Text the model can act on: a skill's markdown/plain-text references, its helper
# scripts (which run_code can import by name), and the structured data those
# references point at. Mirrors what the self-managed SkillsCapability surfaces, so
# an external harness reading a skill does not miss an asset the file references.
_SKILL_MIME_TYPES = {
    '.md': 'text/markdown',
    '.txt': 'text/plain',
    '.py': 'text/x-python',
    '.json': 'application/json',
    '.yaml': 'application/yaml',
    '.yml': 'application/yaml',
    '.csv': 'text/csv',
    '.xml': 'application/xml',
}


@dataclass(frozen=True)
class ServerConfig:
    """Everything fixed for the lifetime of one server process."""
    output_path: Path = Path('./qm_output')
    resname: str = 'LIG'
    amberhome: Path | None = None
    use_gpu: bool = True
    num_threads: int = 8
    exchange: str = 'local'
    skills_root: Path | None = Path('./skills')


def _computational_chemist() -> str:
    """The framing the self-managed QMAgent harness runs with.

    Adopt it before driving the QM tools, so an externally managed run reasons
    about basis sets, charges and torsions the way our own agent does.
    """
    return SYSTEM_PROMPT


def _register_skills(server: FastMCP, skills_root: Path) -> int:
    """Serve ``skills/<name>/**`` as ``skill://<name>/<path>`` resources.

    The self-managed harness gets these through its skills capability; an external
    harness has no such thing, so they are exposed as resources it can read.
    """
    if not skills_root.is_dir():
        return 0

    count = 0
    for path in sorted(skills_root.rglob('*')):
        mime_type = _SKILL_MIME_TYPES.get(path.suffix)
        if mime_type is None or not path.is_file() or '__pycache__' in path.parts:
            continue
        relative = path.relative_to(skills_root)
        server.add_resource(FileResource(
            uri=f'skill://{relative.as_posix()}',
            path=path.resolve(),  # FileResource requires an absolute path
            name=relative.as_posix(),
            description=f'{relative.parts[0]} skill: {relative.as_posix()}',
            mime_type=mime_type,
        ))
        count += 1

    return count


def _register(server: FastMCP, toolkit: QMToolkit, skills_root: Path | None) -> None:
    for fn in toolkit.tool_functions():
        # Register the bound method with QMToolError -> ToolError translation (so a
        # correctable problem reaches the client verbatim while genuine faults stay
        # masked). No serialization: tool calls run concurrently so a harness can
        # fan out independent QM work -- a basis/functional sweep of geometry
        # optimizations, many torsion scans at once. The shared run scope stays
        # consistent under that concurrency; see QMRunState. Stdout isolation is
        # process-wide (_isolate_stdout_for_stdio), so tools need no per-call guard.
        server.add_tool(translate_qm_errors(fn, ToolError))
    server.add_prompt(Prompt.from_function(_computational_chemist, name='computational_chemist'))
    if skills_root is not None:
        _register_skills(server, skills_root)


def create_server(toolkit: QMToolkit, *, skills_root: Path | None = None) -> FastMCP:
    """A server bound to an already-running toolkit.

    For embedding the QM tools in a process that owns its own ``QMAgent`` (and for
    tests, which drive this in-process against a stub handle). The CLI path wants
    ``create_managed_server`` instead.
    """
    server = FastMCP(name=SERVER_NAME, instructions=SYSTEM_PROMPT)
    _register(server, toolkit, skills_root)
    return server


@asynccontextmanager
async def _managed_toolkit(config: ServerConfig) -> AsyncIterator[QMToolkit]:
    """Launch a QMAgent for the life of the server and bind a toolkit to it."""
    config.output_path.mkdir(parents=True, exist_ok=True)
    factory = (
        LocalExchangeFactory() if config.exchange == 'local'
        else HttpExchangeFactory(config.exchange, auth_method='globus')
    )
    async with await Manager.from_exchange_factory(
        factory=factory,
        executors=ThreadPoolExecutor(),
    ) as manager:
        handle = await manager.launch(
            QMAgent(num_threads=config.num_threads, use_gpu=config.use_gpu)
        )
        yield QMToolkit(QMRunState(
            qm=handle,
            output_path=config.output_path,
            resname=config.resname,
            amberhome=config.amberhome,
            # run_code's helper-import path tracks the same directory served as
            # resources, so a snippet can import a skill script it was shown.
            skills_root=config.skills_root or Path('./skills'),
        ))


def create_managed_server(config: ServerConfig) -> FastMCP:
    """A server that owns its QMAgent: the lifespan launches it and tears it down.

    Tools are registered inside the lifespan because they are bound to the agent
    handle, which does not exist until it is launched. Clients only list tools
    after startup, so they see the full set.
    """
    @asynccontextmanager
    async def lifespan(server: FastMCP) -> AsyncIterator[QMToolkit]:
        async with _managed_toolkit(config) as toolkit:
            _register(server, toolkit, config.skills_root)
            yield toolkit

    return FastMCP(
        name=SERVER_NAME,
        instructions=SYSTEM_PROMPT,
        lifespan=lifespan,
        on_duplicate='replace',
    )


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    # An unset OR blank value means "use the default"; only an explicit token flips
    # it. Treating '' as false would let a launcher that exports QMAGENT_GPU= (a
    # common way to blank a var) silently force CPU against the documented default.
    if raw is None or raw.strip() == '':
        return default
    return raw.strip().lower() not in ('0', 'false', 'no')


class _StdoutShim:
    """A ``sys.stdout`` replacement that splits the two ways stdout is used.

    On stdio transport the JSON-RPC channel *is* stdout, but the MCP transport
    reaches it only through ``sys.stdout.buffer`` (it wraps the binary buffer),
    whereas stray text output -- ``print``, geomeTRIC progress, a library banner --
    goes through ``write``. So ``buffer`` stays the real client stream while every
    ``write`` is diverted to stderr. Combined with an fd-level ``dup2`` of stderr
    onto fd 1 (for native/subprocess writers), nothing but framed protocol reaches
    the client.
    """

    def __init__(self, real_buffer: Any) -> None:
        self.buffer = real_buffer  # the MCP transport wraps this -> reaches client

    def write(self, s: str) -> int:
        return sys.stderr.write(s)

    def flush(self) -> None:
        sys.stderr.flush()

    def __getattr__(self, name: str) -> Any:
        # Anything the shim does not override (encoding, isatty, ...) answers as
        # stderr would, so callers introspecting sys.stdout see a consistent stream.
        return getattr(sys.stderr, name)


def _isolate_stdout_for_stdio() -> None:
    """Keep every non-protocol writer in this process off the JSON-RPC channel.

    The QM apps run in-process (parsl thread pools) and are noisy -- PySCF and
    geomeTRIC on stdout, AmberTools subprocesses on fd 1 -- and academy/parsl are
    noisy at startup too. Done once, before the server runs, so it also covers the
    lifespan launch and needs no per-call guard (no save/restore race):

    * ``dup(1)`` preserves the real client stdout for the transport;
    * ``dup2(2, 1)`` repoints fd 1 at stderr, so native code and subprocesses that
      write to the file descriptor land on stderr;
    * ``sys.stdout`` becomes a :class:`_StdoutShim` whose ``buffer`` is the
      preserved client stream but whose ``write`` goes to stderr.
    """
    real_stdout = os.fdopen(os.dup(1), 'wb')  # preserved BEFORE the dup2 below
    os.dup2(sys.stderr.fileno(), 1)
    sys.stdout = _StdoutShim(real_stdout)


def _build_parser() -> argparse.ArgumentParser:
    """Flags default to the environment, so a client can configure the server
    either through ``env`` in its MCP config or through ``args``."""
    parser = argparse.ArgumentParser(
        prog='qmagent-mcp',
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    amberhome = os.environ.get('AMBERHOME')
    parser.add_argument('--output', type=Path, default=Path(os.environ.get('QMAGENT_OUTPUT', './qm_output')),
                        help='Run output directory (env QMAGENT_OUTPUT).')
    parser.add_argument('--resname', default=os.environ.get('QMAGENT_RESNAME', 'LIG'),
                        help='Residue name / output basename (env QMAGENT_RESNAME).')
    parser.add_argument('--amberhome', type=Path, default=Path(amberhome) if amberhome else None,
                        help='AmberTools install root (env AMBERHOME).')
    parser.add_argument('--cpu', dest='use_gpu', action='store_false',
                        default=_env_flag('QMAGENT_GPU', True),
                        help='Import CPU pyscf instead of gpu4pyscf (env QMAGENT_GPU=0).')
    parser.add_argument('--threads', type=int, default=int(os.environ.get('QMAGENT_THREADS', os.cpu_count() or 8)),
                        help='Agent worker threads (env QMAGENT_THREADS).')
    parser.add_argument('--exchange', default=os.environ.get('QMAGENT_EXCHANGE', 'local'),
                        help="academy exchange: 'local' or an http(s) URL (env QMAGENT_EXCHANGE).")
    parser.add_argument('--skills', type=Path, default=Path(os.environ.get('QMAGENT_SKILLS', './skills')),
                        help='Skills directory served as resources (env QMAGENT_SKILLS).')
    parser.add_argument('--transport', choices=('stdio', 'http'), default=os.environ.get('QMAGENT_TRANSPORT', 'stdio'),
                        help='stdio for a client-spawned server; http to serve a remote one.')
    parser.add_argument('--host', default='127.0.0.1', help='Bind host for --transport http.')
    parser.add_argument('--port', type=int, default=8000, help='Bind port for --transport http.')
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    config = ServerConfig(
        output_path=args.output,
        resname=args.resname,
        amberhome=args.amberhome,
        use_gpu=args.use_gpu,
        num_threads=args.threads,
        exchange=args.exchange,
        skills_root=args.skills,
    )
    if args.transport == 'stdio':
        # Must happen before the server (and its lifespan) runs, so no QM/parsl/
        # academy output can reach the protocol channel. http has a real stdout.
        _isolate_stdout_for_stdio()

    server = create_managed_server(config)
    if args.transport == 'stdio':
        # show_banner=False: the banner is decoration a client would only have to
        # skip past, and this path is about keeping the channel clean.
        server.run(transport='stdio', show_banner=False)
    else:
        server.run(transport='http', host=args.host, port=args.port)


if __name__ == '__main__':
    main()
