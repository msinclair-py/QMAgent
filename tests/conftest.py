"""Shared fixtures for the qmagent test suite.

These tests deliberately avoid the heavy runtime dependencies (PySCF/GPU,
AmberTools, the LLM orchestrator and the parsl/academy distributed layer).
They exercise the pure-Python logic: file parsing/writing, the pydantic data
models, the RESP charge fitter, the QM tool layer and the static helpers on
``QMAgent``.
"""

import asyncio

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from qmagent.utils.file_ops import write_mol2


def _run(coro):
    """Drive an async coroutine to completion without a pytest-asyncio plugin.

    The @action methods and tool coroutines are awaitable as-is, so a bare
    ``asyncio.run`` per test is enough and keeps the suite plugin-free.
    """
    return asyncio.run(coro)


class StubHandle:
    """Stands in for the academy Handle to the distributed QMAgent.

    Every attribute resolves to an async action that records its call and returns
    whatever was registered for that name, so a test can assert on what the tool
    layer dispatched without a parsl/academy runtime behind it.
    """

    def __init__(self, **returns):
        self.calls: list[tuple[str, dict]] = []
        self._returns = returns

    def __getattr__(self, name: str):
        async def action(*args, **kwargs):
            self.calls.append((name, kwargs))
            return self._returns.get(name)
        return action

    def kwargs_for(self, name: str) -> dict:
        """The keyword arguments the toolkit dispatched to ``name``."""
        return next(kwargs for called, kwargs in self.calls if called == name)


@pytest.fixture
def ethane_mol():
    """An RDKit ethane molecule with explicit Hs and a single 3D conformer."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CC"))
    # Deterministic embedding so coordinate-dependent assertions are stable.
    assert AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE) == 0
    return mol


@pytest.fixture
def ethane_mol2(tmp_path, ethane_mol):
    """A written ethane .mol2 file that RDKit can read back."""
    path = tmp_path / "ethane.mol2"
    write_mol2(ethane_mol, path, "ETH")
    return path


@pytest.fixture
def methanol_mol2(tmp_path):
    """A methanol .mol2 (has a symmetry-equivalent methyl-H set)."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CO"))
    assert AllChem.EmbedMolecule(mol, randomSeed=1234) == 0
    path = tmp_path / "methanol.mol2"
    write_mol2(mol, path, "MOH")
    return path


# A hand-written Tripos mol2 with known Sybyl types and coordinates, used to
# pin XYZContents.from_mol2's element extraction (sybyl 'C.3' -> 'C') and
# coordinate parsing independent of any RDKit round-tripping.
WATER_MOL2 = """\
@<TRIPOS>MOLECULE
WAT
3 2 1 0 0
SMALL
USER_CHARGES

@<TRIPOS>ATOM
      1 O1      0.0000  0.0000  0.0000 O.3    1 WAT  -0.8000
      2 H1      0.7570  0.5860  0.0000 H      1 WAT   0.4000
      3 H2     -0.7570  0.5860  0.0000 H      1 WAT   0.4000
@<TRIPOS>BOND
     1    1    2 1
     2    1    3 1
"""


@pytest.fixture
def water_mol2(tmp_path):
    path = tmp_path / "water.mol2"
    path.write_text(WATER_MOL2)
    return path
