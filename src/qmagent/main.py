"""Entry point for the self-managed harness.

Launches a QMAgent, binds the shared QM toolkit to it, and hands both to our own
pydantic-ai orchestrator. To drive the same execution layer from an externally
managed harness instead (Claude Code, Codex, ...), run ``qmagent.mcp_server`` --
which launches the agent itself and never touches this module.
"""

import asyncio
import os
from pathlib import Path
from academy.exchange import HttpExchangeFactory
from academy.manager import Manager
from concurrent.futures import ThreadPoolExecutor
from .agents.qm_agent import QMAgent
from .llm_interface import ParameterizationSummary, QMDeps, orchestrator, qm_toolset
from .tools import QMRunState, QMToolkit


async def main() -> ParameterizationSummary:
    output_path = Path('./qm_output')
    output_path.mkdir(parents=True, exist_ok=True)

    async with await Manager.from_exchange_factory(
        factory=HttpExchangeFactory('https://exchange.academy-agents.org', auth_method='globus'),
        executors=ThreadPoolExecutor(),
    ) as manager:
        qm_handle = await manager.launch(QMAgent(num_threads=os.cpu_count() or 8))
        toolkit = QMToolkit(QMRunState(
            qm=qm_handle,
            output_path=output_path,
            resname='LIG',
            amberhome=Path(os.environ['AMBERHOME']),
        ))
        result = await orchestrator.run(
            'Can you generate parameters for this compound: CCCCCC',
            deps=QMDeps(),
            toolsets=[qm_toolset(toolkit)],
        )

        return result.output

if __name__ == '__main__':
    asyncio.run(main())
