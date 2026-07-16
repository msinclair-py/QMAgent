"""Tests for the async @action methods on QMAgent that need no QM/parsl runtime.

The heavy @action coroutines (geometry_optimization, ESP, RESP, torsion scans)
dispatch parsl apps that require PySCF/GPU/AmberTools, so they stay out of scope.
What *is* testable without any of that:

  * ``execute_code`` -- runs a snippet in a plain Python subprocess (no parsl),
    so its I/O capture, working-directory, PYTHONPATH injection and timeout
    behaviour can be exercised directly.
  * the input-validation guards that raise *before* any app is dispatched:
    ``geometry_optimization``'s empty-stages guard and ``fit_RESP_charges``'s
    gas/solvent phase-count guard.

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
    """A QMAgent instance. execute_code and the guards touch no parsl runtime,
    so no agent_on_startup / DataFlowKernel is needed."""
    return QMAgent(num_threads=1)


def _run(coro):
    """Drive an async @action to completion without a pytest-asyncio plugin."""
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# execute_code
# --------------------------------------------------------------------------- #

def test_execute_code_success_captures_stdout(agent):
    result = _run(agent.execute_code("print(6 * 7)"))
    assert result["returncode"] == "0"
    assert result["stdout"].strip() == "42"
    assert result["stderr"] == ""


def test_execute_code_success_with_no_output(agent):
    result = _run(agent.execute_code("x = 1 + 1"))
    assert result["returncode"] == "0"
    assert result["stdout"] == ""
    assert result["stderr"] == ""


def test_execute_code_separates_stdout_and_stderr(agent, tmp_path):
    # Write via a file to avoid shell-escaping newlines in the snippet.
    snippet = (
        "import sys\n"
        "print('on stdout')\n"
        "sys.stderr.write('on stderr')\n"
    )
    result = _run(agent.execute_code(snippet, workdir=tmp_path))
    assert result["returncode"] == "0"
    assert result["stdout"].strip() == "on stdout"
    assert result["stderr"].strip() == "on stderr"


def test_execute_code_exception_puts_traceback_in_stderr(agent):
    result = _run(agent.execute_code("raise ValueError('boom')"))
    # A raised exception must be a non-zero exit with the traceback on stderr.
    assert result["returncode"] == "1"
    assert "Traceback" in result["stderr"]
    assert "ValueError" in result["stderr"]
    assert "boom" in result["stderr"]
    assert result["stdout"] == ""


def test_execute_code_timeout_is_reported(agent):
    result = _run(agent.execute_code("import time; time.sleep(30)", timeout=0.5))
    assert result["returncode"] == "-1"
    assert "timed out" in result["stderr"].lower()


def test_execute_code_workdir_is_cwd(agent, tmp_path):
    # A relative path written by the snippet must land inside workdir.
    _run(agent.execute_code("open('artifact.txt', 'w').write('hi')", workdir=tmp_path))
    assert (tmp_path / "artifact.txt").read_text() == "hi"


def test_execute_code_creates_missing_workdir(agent, tmp_path):
    target = tmp_path / "nested" / "run"
    assert not target.exists()
    result = _run(agent.execute_code("print('ok')", workdir=target))
    assert result["returncode"] == "0"
    assert target.is_dir()


def test_execute_code_qmagent_importable(agent, tmp_path):
    # The package root is injected into PYTHONPATH, so generated code can
    # import the project even when run from an unrelated working directory.
    snippet = (
        "import qmagent.utils.file_ops as f\n"
        "print('has_write_xyz', hasattr(f, 'write_xyz'))\n"
    )
    result = _run(agent.execute_code(snippet, workdir=tmp_path))
    assert result["returncode"] == "0", result["stderr"]
    assert "has_write_xyz True" in result["stdout"]


def test_execute_code_extra_paths_importable(agent, tmp_path):
    # A directory passed via extra_paths must be importable from the snippet.
    helper_dir = tmp_path / "helpers"
    helper_dir.mkdir()
    (helper_dir / "my_helper.py").write_text("VALUE = 123\n")

    result = _run(
        agent.execute_code(
            "import my_helper; print('value', my_helper.VALUE)",
            workdir=tmp_path,
            extra_paths=[helper_dir],
        )
    )
    assert result["returncode"] == "0", result["stderr"]
    assert "value 123" in result["stdout"]


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


# --------------------------------------------------------------------------- #
# find_rotatable_torsions
# --------------------------------------------------------------------------- #

@pytest.fixture
def butane_mol2(tmp_path):
    """n-butane as a mol2 with explicit hydrogens.

    The discriminating case: three bonds match the rotatable SMARTS, but only
    the central C-C joins two non-terminal heavy atoms. The other two are
    methyl rotors.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from qmagent.utils.file_ops import write_mol2

    mol = Chem.AddHs(Chem.MolFromSmiles("CCCC"))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    path = tmp_path / "butane.mol2"
    write_mol2(mol, path, "BUT")
    return path


def test_find_rotatable_torsions_skips_terminal_methyl_rotors(butane_mol2):
    # n-butane has exactly one rotatable bond: the central C1-C2. The two
    # terminal C-C bonds are methyl rotors -- the SMARTS' own `!D1` is meant to
    # reject them, but D1 counts explicit hydrogens, so a methyl carbon (degree
    # 4) slips through and they were being scanned too.
    agent = QMAgent(num_threads=1)
    torsions = asyncio.run(agent.find_rotatable_torsions(mol2_file=butane_mol2))

    assert len(torsions) == 1, f"expected 1 rotatable bond in butane, got {torsions}"

    from rdkit import Chem
    mol = Chem.MolFromMol2File(str(butane_mol2), removeHs=False, sanitize=True)
    (_, b, c, _), = torsions
    # Both central atoms must be carbons with two heavy neighbours.
    for idx in (b, c):
        atom = mol.GetAtomWithIdx(idx)
        heavy = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() > 1)
        assert atom.GetSymbol() == "C" and heavy == 2


def test_find_rotatable_torsions_ethane_has_none(ethane_mol2):
    # Ethane is two methyls: no bond joins two non-terminal heavy atoms, so
    # there is nothing worth scanning (RDKit's own NumRotatableBonds agrees).
    agent = QMAgent(num_threads=1)
    assert asyncio.run(agent.find_rotatable_torsions(mol2_file=ethane_mol2)) == set()
