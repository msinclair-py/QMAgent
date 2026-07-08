import asyncio
import os
from pathlib import Path
from academy.exchange import HttpExchangeFactory
from academy.manager import Manager
from concurrent.futures import ThreadPoolExecutor
from .llm_interface import orchestrator, QMDeps
from .agents.qm_agent import QMAgent

async def main() -> str:
    output_path = Path('./qm_output')
    output_path.mkdir(parents=True, exist_ok=True)

    async with await Manager.from_exchange_factory(
        factory=HttpExchangeFactory('https://exchange.academy-agents.org', auth_method='globus'),
        executors=ThreadPoolExecutor(),
    ) as manager:
        qm_handle = await manager.launch(QMAgent(num_threads=os.cpu_count() or 8))
        result = await orchestrator.run(
            'Can you generate parameters for this compound: CCCCCC',
            deps=QMDeps(
                qm=qm_handle,
                output_path=output_path,
                resname='LIG',
                amberhome=Path(os.environ['AMBERHOME']),
            ),
        )

        return result.output

if __name__ == '__main__':
    asyncio.run(main())
