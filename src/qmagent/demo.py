"""CPU-only, laptop-friendly demo of the QM charge-derivation core.

This runs the *quantum* half of the parameterization pipeline end to end on a
machine with **no GPU and no AmberTools**:

    build_compound -> geometry_optimization -> compute_ESP_charges -> fit_RESP_charges

and prints the fitted RESP2 partial charges. It drives the ``QMAgent`` actions
directly through a local in-process ``academy`` exchange, so it needs **no LLM /
API key** either -- ideal for a live talk. The only runtime requirements beyond
the base install are CPU PySCF and its optimizer:

    pip install pyscf geometric        # (no gpu4pyscf / CUDA needed)

The agent is constructed with ``use_gpu=False``, which makes the DFT apps import
``pyscf`` (CPU) instead of ``gpu4pyscf``. Everything else -- the ESP grid, the
two-stage RESP fit, the symmetry handling -- is identical to the GPU/HPC path.

Examples
--------
    uv run python -m qmagent.demo                         # methanol, HF-ish DFT
    uv run python -m qmagent.demo --smiles CC(=O)NC       # N-methylacetamide
    uv run python -m qmagent.demo --smiles "O" --basis 6-31g --functional b3lyp
"""

import argparse
import asyncio
import os
from pathlib import Path

import numpy as np
from academy.exchange import LocalExchangeFactory
from academy.manager import Manager
from concurrent.futures import ThreadPoolExecutor

from .agents.qm_agent import QMAgent
from .utils.file_ops import XYZContents
from .utils.pydantic_models import QMConfig


async def run_demo(smiles: str, resname: str, output: Path, *,
                   functional: str, basis: str, dispersion: str,
                   charge: int, multiplicity: int, grid_level: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    qm_config = QMConfig(
        functional=functional,
        basis=basis,
        dispersion=dispersion,
        charge=charge,
        multiplicity=multiplicity,
        grid_level=grid_level,
    )

    # Local in-process exchange + a CPU-only QMAgent (use_gpu=False).
    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
        executors=ThreadPoolExecutor(),
    ) as manager:
        qm = await manager.launch(
            QMAgent(num_threads=os.cpu_count() or 4, use_gpu=False)
        )

        mol2_file = output / f'{resname}.mol2'
        print(f'[1/4] build_compound        {smiles!r} -> {mol2_file.name}')
        await qm.build_compound(smiles=smiles, mol2_file=mol2_file, resname=resname)

        print(f'[2/4] geometry_optimization {functional}/{basis} (CPU)')
        geom = await qm.geometry_optimization(
            mol2_file=mol2_file,
            output_path=output,
            optimization_stages=[qm_config],
        )
        print(f'       final energy: {geom.energy:.6f} Ha  ->  {geom.xyz_file.name}')

        molecule = XYZContents.from_xyz(geom.xyz_file)

        print('[3/4] compute_ESP_charges   gas + C-PCM solvent on the MK grid')
        esp = await qm.compute_ESP_charges(
            contents=molecule, output_path=output, qm_config=qm_config,
        )

        print('[4/4] fit_RESP_charges      two-stage RESP2(delta=0.5)')
        resp = await qm.fit_RESP_charges(
            molecule=molecule, mol2_file=mol2_file, esp_results=esp,
            qm_config=qm_config, delta_resp2=0.5,
        )

    charges = np.asarray(resp.charges).ravel()
    print('\nRESP2 partial charges')
    print('  idx  elem      charge (e)')
    for i, (elem, q) in enumerate(zip(resp.elements, charges, strict=True)):
        print(f'  {i:>3}  {elem:<4}  {q:>12.5f}')
    print(f'  {"":>3}  {"sum":<4}  {charges.sum():>12.5f}  (target {charge:+d})')
    print(f'\nWrote intermediates to {output}/')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--smiles', default='CO', help='SMILES to parameterize (default: methanol).')
    parser.add_argument('--resname', default='LIG', help='Residue name / output basename (default: LIG).')
    parser.add_argument('--output', type=Path, default=Path('./qm_demo'),
                        help='Output directory (default: ./qm_demo).')
    parser.add_argument('--functional', default='b3lyp', help='DFT exchange-correlation functional.')
    parser.add_argument('--basis', default='6-31g*', help='Basis set (small = fast; default 6-31g*).')
    parser.add_argument('--dispersion', default='d3bj', help='Empirical dispersion (must not double-count the functional).')
    parser.add_argument('--charge', type=int, default=0, help='Net molecular charge.')
    parser.add_argument('--multiplicity', type=int, default=1, help='Spin multiplicity (2S+1).')
    parser.add_argument('--grid-level', type=int, default=3, help='PySCF integration grid level (higher = finer/slower).')
    args = parser.parse_args()

    asyncio.run(run_demo(
        smiles=args.smiles, resname=args.resname, output=args.output,
        functional=args.functional, basis=args.basis, dispersion=args.dispersion,
        charge=args.charge, multiplicity=args.multiplicity, grid_level=args.grid_level,
    ))


if __name__ == '__main__':
    main()
