"""MCP front-end for a persistent Academy-backed QMAgent.

Run this on the machine that owns the QM environment/GPU allocation.  An MCP
client on a laptop/login node can then call the tools here; the tools dispatch to
one long-lived Academy ``QMAgent`` handle instead of spawning a fresh execution
layer for each calculation.
"""

from __future__ import annotations

import argparse
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

from academy.exchange import LocalExchangeFactory
from academy.manager import Manager
from concurrent.futures import ThreadPoolExecutor
from mcp.server.fastmcp import FastMCP

from .agents.qm_agent import QMAgent
from .agents.parsl_configs import multi_gpu_config


@dataclass
class QMAgentMCPState:
    """Runtime state owned by the MCP server lifespan."""

    manager_cm: Any | None = None
    manager: Manager | None = None
    executor: ThreadPoolExecutor | None = None
    qm_handle: Any | None = None
    num_threads: int = os.cpu_count() or 8
    max_memory: int = 160_000
    use_gpu: bool = True
    # When non-empty, GPU-labelled apps fan out one-per-device across these
    # accelerator indices (relative to the process's CUDA_VISIBLE_DEVICES) via a
    # HighThroughputExecutor. Empty means the single-slot default.
    gpu_ids: list[str] = field(default_factory=list)


STATE = QMAgentMCPState()


def _handle() -> Any:
    """Return the live Academy handle or raise a clear server-side error."""
    if STATE.qm_handle is None:
        raise RuntimeError("QMAgent handle is not available; MCP lifespan did not start cleanly")
    return STATE.qm_handle


def _path(value: str | None) -> Path | None:
    return Path(value).expanduser().resolve() if value else None


@asynccontextmanager
async def lifespan(_: FastMCP) -> AsyncIterator[QMAgentMCPState]:
    """Launch one persistent QMAgent for the life of the MCP server."""
    STATE.executor = ThreadPoolExecutor()
    STATE.manager_cm = await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
        executors=STATE.executor,
    )
    STATE.manager = await STATE.manager_cm.__aenter__()
    if STATE.manager is None:  # defensive; __aenter__ should always return a Manager
        raise RuntimeError("Academy Manager did not start")

    # Multi-GPU fan-out: when --gpus is given, build a HighThroughputExecutor
    # that pins one worker process per device, so N torsion scans (one parsl
    # task each) run on N GPUs at once instead of queueing on a single slot.
    # QMAgent.num_threads then becomes the *per-worker* PySCF thread count so N
    # workers don't oversubscribe the node's cores. With no --gpus, parsl_config
    # is None and QMAgent falls back to its single-slot local config unchanged.
    parsl_config = None
    agent_threads = STATE.num_threads
    if STATE.use_gpu and STATE.gpu_ids:
        n = len(STATE.gpu_ids)
        agent_threads = max(1, STATE.num_threads // n)
        parsl_config = multi_gpu_config(
            gpu_ids=STATE.gpu_ids,
            cpu_threads_per_worker=agent_threads,
        )

    STATE.qm_handle = await STATE.manager.launch(
        QMAgent(
            num_threads=agent_threads,
            max_memory=STATE.max_memory,
            use_gpu=STATE.use_gpu,
            parsl_config=parsl_config,
        )
    )
    try:
        yield STATE
    finally:
        if STATE.manager_cm is not None:
            await STATE.manager_cm.__aexit__(None, None, None)
        if STATE.executor is not None:
            STATE.executor.shutdown(wait=True)
        STATE.qm_handle = None
        STATE.manager = None
        STATE.manager_cm = None
        STATE.executor = None


mcp = FastMCP(
    "qmagent-academy",
    instructions=(
        "Persistent MCP facade for an Academy QMAgent. The MCP server should run "
        "on the compute resource; clients call these tools from a login node or laptop."
    ),
    lifespan=lifespan,
)


@mcp.tool()
async def health() -> dict[str, Any]:
    """Report whether the persistent Academy QMAgent handle is available."""
    return {
        "status": "ready" if STATE.qm_handle is not None else "not_ready",
        "num_threads": STATE.num_threads,
        "max_memory_mb": STATE.max_memory,
        "use_gpu": STATE.use_gpu,
        "gpu_ids": STATE.gpu_ids,
        "gpu_fanout": len(STATE.gpu_ids) if (STATE.use_gpu and STATE.gpu_ids) else 1,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "amberhome": os.environ.get("AMBERHOME"),
        "pyscf_config_file": os.environ.get("PYSCF_CONFIG_FILE"),
        "handle_type": type(STATE.qm_handle).__name__ if STATE.qm_handle is not None else None,
    }


@mcp.tool()
async def execute_code(
    code: str,
    workdir: str | None = None,
    timeout: float = 1800.0,
    extra_paths: list[str] | None = None,
) -> dict[str, str]:
    """Execute Python through the persistent Academy QMAgent handle.

    The snippet runs on the QMAgent side with the same interpreter/environment as
    the worker, so it sees PySCF/gpu4pyscf/AmberTools exactly as the compute node
    does. Use ``workdir`` for experiment output isolation.
    """
    return await _handle().execute_code(
        code_snippet=code,
        workdir=_path(workdir),
        extra_paths=[Path(p).expanduser().resolve() for p in (extra_paths or [])],
        timeout=timeout,
    )


@mcp.tool()
async def shutdown_note() -> str:
    """Explain how to stop the long-lived server cleanly."""
    return "Stop the MCP server process (Ctrl-C or scheduler cancellation); lifespan cleanup tears down the Academy Manager and QMAgent."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", choices=["stdio", "sse", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1", help="Host for HTTP/SSE transports")
    parser.add_argument("--port", type=int, default=8000, help="Port for HTTP/SSE transports")
    parser.add_argument("--num-threads", type=int, default=os.cpu_count() or 8)
    parser.add_argument("--max-memory", type=int, default=160_000, help="PySCF max memory in MB")
    parser.add_argument("--cpu", action="store_true", help="Run QMAgent with use_gpu=False")
    parser.add_argument(
        "--gpus", default=None,
        help=(
            "Comma-separated ABSOLUTE physical GPU IDs to fan the server-side "
            "QMAgent's GPU tasks across, one task per device (e.g. '3,5,6,7'). "
            "parsl pins each worker to that physical device verbatim. NOTE: in "
            "code-execution-only mode this has no effect on execute_code (which "
            "runs in a subprocess with its own parsl) and only wastes GPUs -- "
            "build the fan-out inside the executed snippet instead. Omit for the "
            "single-slot default. --num-threads is divided across the workers."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    STATE.num_threads = args.num_threads
    STATE.max_memory = args.max_memory
    STATE.use_gpu = not args.cpu
    STATE.gpu_ids = (
        [g.strip() for g in args.gpus.split(",") if g.strip()] if args.gpus else []
    )

    # FastMCP stores host/port on the instance settings; set them from CLI before run.
    mcp.settings.host = args.host
    mcp.settings.port = args.port
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
