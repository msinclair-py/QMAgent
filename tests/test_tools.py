"""Tests for the harness-agnostic QM tool layer.

``QMToolkit`` is where the orchestration logic lives -- artifact bookkeeping,
prerequisite guards, AMBER config assembly -- and none of it needs a harness or a
QM runtime to exercise. The academy ``Handle`` is the only boundary the toolkit
talks through, so a stub that records calls and returns canned results is enough
to pin the behaviour both harnesses depend on.
"""

import numpy as np
import pytest

from qmagent.tools import QMRunState, QMToolError, QMToolkit
from qmagent.utils.pydantic_models import (
    AMBERResultSet,
    GeomOptResult,
    QMConfig,
    RESPCharges,
)

from conftest import StubHandle, _run


def _qm_config() -> QMConfig:
    return QMConfig(
        functional='b3lyp', basis='6-31g*', dispersion='d3bj',
        charge=0, multiplicity=1, grid_level=3,
    )


@pytest.fixture
def xyz_file(tmp_path):
    """A minimal two-atom xyz the toolkit can read back as XYZContents."""
    path = tmp_path / 'opt.xyz'
    path.write_text('2\nH2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n')
    return path


@pytest.fixture
def state(tmp_path):
    return QMRunState(qm=StubHandle(), output_path=tmp_path, resname='LIG')


# --------------------------------------------------------------------------- #
# Artifact table
# --------------------------------------------------------------------------- #

def test_put_returns_incrementing_keys_per_kind(state):
    assert state.put('geomopt', 'a') == 'geomopt_1'
    assert state.put('geomopt', 'b') == 'geomopt_2'
    assert state.put('esp', 'c') == 'esp_1'


def test_get_returns_the_stored_artifact(state):
    result = GeomOptResult(xyz_file='x.xyz', energy=-1.0)
    key = state.put('geomopt', result)
    assert state.get(key, GeomOptResult) is result


def test_get_unknown_key_is_correctable(state):
    with pytest.raises(QMToolError, match='No artifact "geomopt_9"'):
        state.get('geomopt_9', GeomOptResult)


def test_get_wrong_type_names_both_types(state):
    key = state.put('esp', 'not a geomopt')
    with pytest.raises(QMToolError, match='is a str, but this step needs a GeomOptResult'):
        state.get(key, GeomOptResult)


# --------------------------------------------------------------------------- #
# Prerequisite guards -- these must raise *before* dispatching to the agent
# --------------------------------------------------------------------------- #

def test_geometry_optimization_requires_a_built_compound(state):
    toolkit = QMToolkit(state)
    with pytest.raises(QMToolError, match='Call build_compound first'):
        _run(toolkit.geometry_optimization(stages=[_qm_config()]))
    assert state.qm.calls == []


def test_fit_resp_charges_requires_a_built_compound(state):
    toolkit = QMToolkit(state)
    with pytest.raises(QMToolError, match='Call build_compound first'):
        _run(toolkit.fit_resp_charges(geomopt_key='geomopt_1', esp_key='esp_1',
                                      qm_config=_qm_config()))
    assert state.qm.calls == []


def test_integrate_amber_ff_requires_amberhome(state):
    state.mol2_file = state.output_path / 'LIG.mol2'
    toolkit = QMToolkit(state)
    with pytest.raises(QMToolError, match='AMBERHOME is not set'):
        _run(toolkit.integrate_amber_ff(resp_key='resp_1'))
    assert state.qm.calls == []


def test_fit_torsions_requires_an_amber_topology(state):
    toolkit = QMToolkit(state)
    with pytest.raises(QMToolError, match='Run integrate_amber_ff before fitting torsions'):
        _run(toolkit.fit_torsions(torsionscan_key='torsionscan_1'))
    assert state.qm.calls == []


# --------------------------------------------------------------------------- #
# Dispatch + state threading
# --------------------------------------------------------------------------- #

def test_build_compound_records_smiles_and_mol2_on_the_run_state(state):
    toolkit = QMToolkit(state)
    summary = _run(toolkit.build_compound(smiles='CCO'))

    assert state.smiles == 'CCO'
    assert state.mol2_file == state.output_path / 'LIG.mol2'
    # The residue name is the run's, never the caller's to choose.
    assert state.qm.kwargs_for('build_compound')['resname'] == 'LIG'
    assert 'LIG.mol2' in summary


def test_geometry_optimization_stashes_an_artifact_and_reports_its_key(state, xyz_file):
    state.mol2_file = state.output_path / 'LIG.mol2'
    state.qm = StubHandle(geometry_optimization=GeomOptResult(xyz_file=xyz_file, energy=-40.5))
    toolkit = QMToolkit(state)

    summary = _run(toolkit.geometry_optimization(stages=[_qm_config()], max_steps=50))

    assert 'geomopt_1' in summary and '-40.5' in summary
    assert isinstance(state.artifacts['geomopt_1'], GeomOptResult)
    assert state.qm.kwargs_for('geometry_optimization')['max_steps'] == 50


def test_integrate_amber_ff_defaults_the_charge_to_the_rounded_resp_total(state, tmp_path):
    state.mol2_file = tmp_path / 'LIG.mol2'
    state.amberhome = tmp_path / 'amber'
    state.put('resp', RESPCharges(elements=['C', 'O'], charges=np.array([-0.62, -0.35])))
    state.qm = StubHandle(integrate_AMBER_ff=AMBERResultSet(
        mol2_file=tmp_path / 'LIG_gaff2.mol2',
        frcmod_file=tmp_path / 'LIG.frcmod',
        lib_file=tmp_path / 'LIG.lib',
        prmtop=tmp_path / 'LIG.prmtop',
        inpcrd=tmp_path / 'LIG.inpcrd',
    ))
    toolkit = QMToolkit(state)

    summary = _run(toolkit.integrate_amber_ff(resp_key='resp_1'))

    # -0.97 e of fitted charge is a -1 residue, not 0.
    assert state.qm.kwargs_for('integrate_AMBER_ff')['amber_config'].charge == -1
    assert 'charge -1' in summary
    # The topology is stashed so fit_torsions can reuse it.
    assert state.amber_config is not None


# --------------------------------------------------------------------------- #
# run_code
# --------------------------------------------------------------------------- #

def test_run_code_returns_stdout_on_success(state):
    state.qm = StubHandle(execute_code={'stdout': 'hello', 'stderr': '', 'returncode': '0'})
    toolkit = QMToolkit(state)

    out = _run(toolkit.run_code(code='print("hello")'))

    assert 'hello' in out
    assert state.qm.kwargs_for('execute_code')['workdir'] == state.output_path


def test_run_code_surfaces_the_traceback_as_correctable(state):
    state.qm = StubHandle(execute_code={
        'stdout': '', 'stderr': 'ZeroDivisionError: division by zero', 'returncode': '1',
    })
    toolkit = QMToolkit(state)

    with pytest.raises(QMToolError, match='ZeroDivisionError'):
        _run(toolkit.run_code(code='1/0'))


def test_run_code_uses_the_runs_configured_skills_root(tmp_path):
    """The helper-import path must follow the run's skills_root, so a snippet can
    import a skill script the harness actually served -- not a fixed ./skills."""
    skills = tmp_path / 'custom_skills'
    (skills / 'pyscf' / 'scripts').mkdir(parents=True)
    state = QMRunState(
        qm=StubHandle(execute_code={'stdout': '', 'stderr': '', 'returncode': '0'}),
        output_path=tmp_path,
        skills_root=skills,
    )

    _run(QMToolkit(state).run_code(code='pass'))

    extra_paths = state.qm.kwargs_for('execute_code')['extra_paths']
    assert skills / 'pyscf' / 'scripts' in extra_paths


def test_run_code_treats_stderr_alone_as_warnings_not_failure(state):
    # Libraries write benign warnings to stderr on a clean run; only the
    # returncode decides success.
    state.qm = StubHandle(execute_code={
        'stdout': 'ok', 'stderr': 'UserWarning: deprecated', 'returncode': '0',
    })
    toolkit = QMToolkit(state)

    out = _run(toolkit.run_code(code='...'))

    assert 'ok' in out and 'UserWarning' in out


def test_run_code_forwards_its_timeout_to_the_agent(state):
    """run_code must let the model choose a timeout and pass it explicitly.

    execute_code's own default is 300s, which does not fit real QM: an open-shell
    TS search plus a Hessian blows straight through it. The tool owns the policy
    and always forwards it, so the model's escape from a too-short default is a
    bigger timeout, not a reach for background execution."""
    state.qm = StubHandle(execute_code={'stdout': 'ok', 'stderr': '', 'returncode': '0'})

    _run(QMToolkit(state).run_code(code='print(1)', timeout=5400.0))

    assert state.qm.kwargs_for('execute_code')['timeout'] == 5400.0


def test_run_code_default_timeout_fits_real_qm():
    """The default must be long enough for a saddle-point search plus a Hessian."""
    import inspect

    default = inspect.signature(QMToolkit.run_code).parameters['timeout'].default
    # 300s (execute_code's own default) is empirically too short: an HCN TS
    # search alone took ~175s at def2-SVP, and CH4 + .OH is far bigger.
    assert default >= 1800.0


def test_run_code_clips_what_it_returns(state):
    """A chatty snippet must not put its whole log into the conversation.

    Tool output is re-sent to the model on every later step, so one verbose
    optimizer log is paid for once per remaining turn -- the compounding that took
    a real run to 1.53M input tokens, 99% of it re-sent context."""
    from qmagent.tools import MAX_TOOL_OUTPUT_CHARS

    state.qm = StubHandle(execute_code={
        'stdout': 'A' + ('geomeTRIC iteration noise ' * 20_000) + 'Z',
        'stderr': '', 'returncode': '0',
    })

    out = _run(QMToolkit(state).run_code(code='print("noisy")'))

    assert len(out) < MAX_TOOL_OUTPUT_CHARS + 2000, f'returned {len(out):,} chars unclipped'
    assert 'clipped' in out


# --------------------------------------------------------------------------- #
# Tool-output clipping (_clip)
# --------------------------------------------------------------------------- #

def test_clip_leaves_short_output_untouched():
    from qmagent.tools import _clip

    assert _clip('E = -76.358207 Ha') == 'E = -76.358207 Ha'


def test_clip_keeps_both_ends_and_says_how_much_it_dropped():
    """QM logs are front- and back-loaded: setup at the top, answer at the bottom."""
    from qmagent.tools import _clip

    text = 'SETUP-MARKER' + ('x' * 100_000) + 'CONVERGED-MARKER'
    out = _clip(text, limit=1000)

    assert 'SETUP-MARKER' in out, 'lost the head of the output'
    assert 'CONVERGED-MARKER' in out, 'lost the tail -- where the answer lives'
    assert 'clipped' in out
    assert len(out) < 2000
    # The model must be able to tell "that was all" from "there was more".
    assert '100,0' in out or 'characters clipped' in out


def test_clip_preserves_a_traceback_tail():
    """A failing snippet's traceback must survive: it is the whole point of the retry."""
    from qmagent.tools import _clip

    noise = '\n'.join(f'  cycle {i}  E = -76.35' for i in range(5000))
    text = noise + '\nTraceback (most recent call last):\nValueError: bad basis'
    out = _clip(text, limit=1000)

    assert 'ValueError: bad basis' in out


# --------------------------------------------------------------------------- #
# The exposed surface
# --------------------------------------------------------------------------- #

def test_tool_functions_are_documented_and_uniquely_named(state):
    functions = QMToolkit(state).tool_functions()

    names = [fn.__name__ for fn in functions]
    assert len(names) == len(set(names))
    # Both adapters build their schemas from these docstrings, so a tool without
    # one would reach a model undescribed.
    assert all(fn.__doc__ for fn in functions)
