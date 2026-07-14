"""Test systems for exercising the parameterization agent end to end.

Each entry is a small *model compound* (a truncated side-chain analog, the same
way the stapled-linker work models CYL as a methyl-thioether) chosen because it
already has published AMBER parameters in the ``Forcefield_PTM`` reference under
``../parameterization``. That makes the agent's output checkable: charges,
GAFF2 atom types, and torsion barriers can be compared against literature.

The ladder runs neutral -> charged and small -> larger:

    HEX  hexane                      q= 0  pure-plumbing smoke test
    NMA  N-methylacetamide           q= 0  smallest real chemistry (one amide)
    ALY  acetyl-lysine side chain    q= 0  neutral PTM, a few rotatable bonds
    SEP  methylphosphate (pSer)      q=-2  exercises the charged path + P typing
    M3L  trimethyllysine side chain  q=+1  cationic path, symmetric methyls

Run the whole ladder (from the repo root, as a module so relative imports work):

    python -m qmagent.test_systems
    python -m qmagent.test_systems --only NMA ALY     # subset by resname
"""

import argparse
import asyncio
import os
import traceback
from dataclasses import dataclass
from pathlib import Path

from academy.exchange import LocalExchangeFactory
from academy.manager import Manager
from concurrent.futures import ThreadPoolExecutor

from .llm_interface import orchestrator, QMDeps
from .agents.qm_agent import QMAgent


@dataclass(frozen=True)
class TestSystem:
    """A single parameterization test case with a known reference."""
    resname: str          # 3-letter AMBER residue name (also the output subdir)
    name: str             # human-readable label
    smiles: str           # model-compound SMILES
    expected_charge: int  # net formal charge the RESP fit should round to
    reference: str         # where published parameters can be checked
    notes: str = ''


TEST_SYSTEMS: list[TestSystem] = [
    TestSystem(
        resname='HEX',
        name='hexane',
        smiles='CCCCCC',
        expected_charge=0,
        reference='trivial / GAFF2 alkane',
        notes='Pure plumbing smoke test (matches the original main.py run).',
    ),
    TestSystem(
        resname='NMA',
        name='N-methylacetamide',
        smiles='CC(=O)NC',
        expected_charge=0,
        reference='GAFF2 / standard amide; canonical backbone model',
        notes='Smallest case with real chemistry: a single amide torsion.',
    ),
    TestSystem(
        resname='ALY',
        name='acetyl-lysine side-chain model',
        smiles='CC(=O)NCCCC',
        expected_charge=0,
        reference='Forcefield_PTM (acetyl-lysine, ALY)',
        notes='Neutral PTM with several rotatable bonds.',
    ),
    TestSystem(
        resname='SEP',
        name='methylphosphate (phospho-serine model)',
        smiles='COP(=O)([O-])[O-]',
        expected_charge=-2,
        reference='Forcefield_PTM (phospho-serine, SEP); Steinbrecher 2012',
        notes='Exercises the charged path and GAFF2 phosphate atom typing.',
    ),
    TestSystem(
        resname='M3L',
        name='trimethyllysine side-chain model',
        smiles='CCCC[N+](C)(C)C',
        expected_charge=+1,
        reference='Forcefield_PTM (trimethyl-lysine, M3L)',
        notes='Cationic path with symmetry-equivalent methyls for the RESP fit.',
    ),
]


async def run_one(manager: Manager, qm_handle, system: TestSystem,
                  base_output: Path) -> tuple[TestSystem, object | None, str | None]:
    """Run the agent on one system. Returns (system, summary, error)."""
    output_path = base_output / system.resname
    output_path.mkdir(parents=True, exist_ok=True)
    try:
        result = await orchestrator.run(
            f'Can you generate parameters for this compound: {system.smiles}',
            deps=QMDeps(
                qm=qm_handle,
                output_path=output_path,
                resname=system.resname,
                amberhome=Path(os.environ['AMBERHOME']),
            ),
        )
        return system, result.output, None
    except Exception as exc:  # keep going so one failure doesn't sink the ladder
        return system, None, f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}'


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--only', nargs='+', metavar='RESNAME',
        help='Restrict to these resnames (e.g. --only NMA ALY).',
    )
    parser.add_argument(
        '--output', type=Path, default=Path('./qm_output'),
        help='Base output directory (default ./qm_output); each system gets a subdir.',
    )
    parser.add_argument(
        '--cpu', action='store_true',
        help='Run the QM steps on CPU PySCF (use_gpu=False) instead of gpu4pyscf, '
             'for a GPU-less machine. Slower; keep to the smallest systems.',
    )
    args = parser.parse_args()

    systems = TEST_SYSTEMS
    if args.only:
        wanted = {r.upper() for r in args.only}
        systems = [s for s in TEST_SYSTEMS if s.resname in wanted]
        missing = wanted - {s.resname for s in systems}
        if missing:
            parser.error(f'unknown resname(s): {", ".join(sorted(missing))}')

    args.output.mkdir(parents=True, exist_ok=True)

    # Test harness uses a local in-process exchange instead of the hosted
    # Globus exchange (main.py): the QMAgent already runs locally in the thread
    # pool, so the exchange is just the message bus. This also sidesteps the
    # hosted exchange running an academy build newer than the pinned client.
    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
        executors=ThreadPoolExecutor(),
    ) as manager:
        qm_handle = await manager.launch(
            QMAgent(num_threads=os.cpu_count() or 8, use_gpu=not args.cpu)
        )

        results = []
        for system in systems:
            print(f'\n=== {system.resname}: {system.name}  '
                  f'(SMILES {system.smiles}, expected q={system.expected_charge:+d}) ===')
            system, summary, error = await run_one(manager, qm_handle, system, args.output)
            results.append((system, summary, error))
            if error is None:
                print(summary)
                if summary.net_charge != system.expected_charge:
                    print(f'CHARGE MISMATCH: agent fit q={summary.net_charge:+d}, '
                          f'expected q={system.expected_charge:+d}')
            else:
                print(f'FAILED: {error}')

    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)
    n_fail = 0
    for system, summary, error in results:
        if error is not None:
            status, detail = 'FAIL', f'  ({error})'
        elif summary.net_charge != system.expected_charge:
            status = 'CHARGE'
            detail = f'  (got q={summary.net_charge:+d}, expected {system.expected_charge:+d})'
        else:
            status, detail = 'ok', ''
        if status != 'ok':
            n_fail += 1
        print(f'  [{status:6s}] {system.resname:4s} {system.name}{detail}')

    if n_fail:
        raise SystemExit(f'{n_fail}/{len(results)} system(s) failed or mismatched charge.')


if __name__ == '__main__':
    asyncio.run(main())
