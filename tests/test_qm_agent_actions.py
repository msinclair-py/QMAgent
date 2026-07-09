"""Tests for the async @action input-validation guards on QMAgent that need no
QM/parsl runtime.

The heavy @action coroutines (geometry_optimization, ESP, RESP, torsion scans)
dispatch parsl apps that require PySCF/GPU/AmberTools, so they stay out of
scope. What *is* testable without any of that: the guards that raise *before*
any app is dispatched -- ``geometry_optimization``'s empty-stages guard and
``fit_RESP_charges``'s gas/solvent phase-count guard.

These call the coroutines directly via ``asyncio.run`` (no pytest-asyncio
plugin dependency); the @action decorator leaves the method awaitable as-is.
"""

import asyncio

import numpy as np
import pytest

from qmagent.agents.qm_agent import QMAgent
from qmagent.utils.file_ops import XYZContents
from qmagent.utils.pydantic_models import (
    ESPCalculation,
    ESPResult,
    QMConfig,
)


@pytest.fixture
def agent():
    """A QMAgent instance. These guards touch no parsl runtime, so no
    agent_on_startup / DataFlowKernel is needed."""
    return QMAgent(num_threads=1)


def _run(coro):
    """Drive an async @action to completion without a pytest-asyncio plugin."""
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# geometry_optimization -- empty-stages guard (raises before any app dispatch)
# --------------------------------------------------------------------------- #

def test_geometry_optimization_empty_stages_raises(agent, tmp_path):
    with pytest.raises(ValueError, match="at least one"):
        _run(
            agent.geometry_optimization(
                mol2_file=tmp_path / "nope.mol2",
                output_path=tmp_path,
                optimization_stages=[],
            )
        )


# --------------------------------------------------------------------------- #
# fit_RESP_charges -- gas/solvent phase-count guard
# --------------------------------------------------------------------------- #

def _qm_config():
    return QMConfig(
        functional="b3lyp",
        basis="sto-3g",
        dispersion="d3bj",
        charge=0,
        multiplicity=1,
        grid_level=3,
    )


def _esp_result(*solvated_flags):
    return ESPResult(
        calculations=[
            ESPCalculation(esp_total=np.zeros(3), energy=0.0, solvated=flag)
            for flag in solvated_flags
        ]
    )


@pytest.mark.parametrize(
    "flags",
    [
        (False, False),   # two gas, no solvent
        (True, True),     # two solvent, no gas
        (False,),         # only gas
        (False, True, True),  # duplicate solvent
    ],
)
def test_fit_resp_charges_bad_phase_counts_raise(agent, tmp_path, flags):
    molecule = XYZContents(elements=["H"], coords=np.zeros((1, 3)))
    with pytest.raises(ValueError, match="gas-phase and one solvent-phase"):
        _run(
            agent.fit_RESP_charges(
                molecule=molecule,
                mol2_file=tmp_path / "nope.mol2",
                esp_results=_esp_result(*flags),
                qm_config=_qm_config(),
            )
        )
