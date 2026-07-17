"""Tests for the MCP server that fronts the execution layer.

These drive a real ``FastMCP`` server in-process over a real client session, with
a stub handle standing in for the distributed QMAgent -- so the wiring under test
(schemas derived from the shared toolkit, error translation, the stdout guard,
prompt and skill resources) is the same wiring an external harness would meet.
"""

import sys

import pytest
from fastmcp import Client

from qmagent.mcp_server import ServerConfig, create_server
from qmagent.prompts import SYSTEM_PROMPT
from qmagent.tools import QMRunState, QMToolkit

from conftest import StubHandle, _run


EXPECTED_TOOLS = {
    'run_code',
    'build_compound',
    'geometry_optimization',
    'compute_esp',
    'scan_torsions',
    'fit_resp_charges',
    'integrate_amber_ff',
    'fit_torsions',
    'run_parameterization_pipeline',
}


@pytest.fixture
def toolkit(tmp_path):
    return QMToolkit(QMRunState(qm=StubHandle(), output_path=tmp_path, resname='LIG'))


@pytest.fixture
def server(toolkit):
    return create_server(toolkit)


def test_server_exposes_the_whole_toolkit(server):
    async def go():
        async with Client(server) as client:
            return await client.list_tools()

    tools = _run(go())
    assert {t.name for t in tools} == EXPECTED_TOOLS


def test_tools_carry_their_docstrings_as_descriptions(server):
    async def go():
        async with Client(server) as client:
            return await client.list_tools()

    tools = {t.name: t for t in _run(go())}
    # The description is the shared docstring from tools.py, not a second copy.
    assert tools['build_compound'].description.startswith(
        'Embed a SMILES string into a 3D conformer'
    )
    assert all(t.description for t in tools.values())


def test_schemas_expose_only_the_tool_arguments(server):
    """`self` is bound and the run's config is server-level; neither is the
    caller's to pass."""
    async def go():
        async with Client(server) as client:
            return await client.list_tools()

    tools = {t.name: t for t in _run(go())}

    build = tools['build_compound'].inputSchema['properties']
    assert set(build) == {'smiles', 'max_iters'}

    geomopt = tools['geometry_optimization'].inputSchema['properties']
    assert set(geomopt) == {'stages', 'constraints', 'max_steps'}
    assert 'self' not in geomopt and 'state' not in geomopt


def test_pydantic_arguments_round_trip_over_the_wire(server, toolkit, tmp_path):
    """QMConfig reaches the toolkit as a model, not a dict."""
    xyz = tmp_path / 'opt.xyz'
    xyz.write_text('2\nH2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n')
    from qmagent.utils.pydantic_models import GeomOptResult, QMConfig

    toolkit.state.mol2_file = tmp_path / 'LIG.mol2'
    toolkit.state.qm = StubHandle(
        geometry_optimization=GeomOptResult(xyz_file=xyz, energy=-40.5),
    )

    async def go():
        async with Client(server) as client:
            return await client.call_tool('geometry_optimization', {
                'stages': [{
                    'functional': 'b3lyp', 'basis': '6-31g*', 'dispersion': 'd3bj',
                    'charge': 0, 'multiplicity': 1, 'grid_level': 3,
                }],
                'max_steps': 50,
            })

    result = _run(go())

    assert 'geomopt_1' in result.content[0].text
    (stage,) = toolkit.state.qm.kwargs_for('geometry_optimization')['optimization_stages']
    assert isinstance(stage, QMConfig)
    assert stage.functional == 'b3lyp'


def test_tool_calls_fan_out_concurrently_with_distinct_artifacts(server, toolkit, tmp_path):
    """A harness can fan out independent QM work (here a basis/functional sweep of
    geometry optimizations): the calls run concurrently AND each gets its own
    artifact key, with no lost writes to the shared run scope."""
    import asyncio
    from qmagent.utils.pydantic_models import GeomOptResult

    xyz = tmp_path / 'opt.xyz'
    xyz.write_text('2\nH2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n')
    toolkit.state.mol2_file = tmp_path / 'LIG.mol2'  # built once, read by every call

    live = 0
    max_live = 0

    async def tracked_geomopt(*args, **kwargs):
        nonlocal live, max_live
        live += 1
        max_live = max(max_live, live)
        await asyncio.sleep(0.05)  # keep every call in flight at once
        live -= 1
        return GeomOptResult(xyz_file=xyz, energy=-40.0)

    toolkit.state.qm.geometry_optimization = tracked_geomopt

    stage = {'functional': 'b3lyp', 'basis': '6-31g*', 'dispersion': 'd3bj',
             'charge': 0, 'multiplicity': 1, 'grid_level': 3}

    async def go():
        async with Client(server) as client:
            return await asyncio.gather(*(
                client.call_tool('geometry_optimization', {'stages': [stage]})
                for _ in range(4)
            ))

    results = _run(go())

    assert max_live == 4  # all four ran concurrently -- not serialized
    keys = {r.content[0].text.split(':')[0] for r in results}
    assert keys == {'geomopt_1', 'geomopt_2', 'geomopt_3', 'geomopt_4'}  # distinct, no loss


def test_qm_tool_error_reaches_the_client_as_a_correctable_message(server):
    """QMToolError must arrive verbatim: it is how a harness learns what to fix."""
    async def go():
        async with Client(server) as client:
            return await client.call_tool('geometry_optimization', {'stages': []})

    with pytest.raises(Exception, match='Call build_compound first'):
        _run(go())


def test_stdout_shim_splits_protocol_from_chatter():
    """The transport reaches the client through sys.stdout.buffer, while stray
    text writes (print, geomeTRIC progress) must be diverted to stderr. The shim
    keeps buffer pointed at the real client stream and sends write() to stderr."""
    import io
    from qmagent.mcp_server import _StdoutShim

    client_channel = io.BytesIO()
    shim = _StdoutShim(client_channel)

    # The MCP transport writes framed protocol through .buffer -> the client.
    shim.buffer.write(b'{"jsonrpc":"2.0"}')
    assert client_channel.getvalue() == b'{"jsonrpc":"2.0"}'

    # A stray print() reaches .write() -> stderr, never the client channel.
    err = io.StringIO()
    old_stderr = sys.stderr
    sys.stderr = err
    try:
        print('SCF cycle 1  E = -40.5', file=shim)
    finally:
        sys.stderr = old_stderr

    assert 'SCF cycle' in err.getvalue()
    assert b'SCF cycle' not in client_channel.getvalue()


def test_server_serves_the_chemist_prompt(server):
    """An external harness has no system prompt of ours unless we hand it one."""
    async def go():
        async with Client(server) as client:
            prompts = await client.list_prompts()
            rendered = await client.get_prompt('computational_chemist')
            return prompts, rendered

    prompts, rendered = _run(go())

    assert [p.name for p in prompts] == ['computational_chemist']
    assert rendered.messages[0].content.text == SYSTEM_PROMPT


def test_skills_are_served_as_resources(toolkit, tmp_path):
    skills = tmp_path / 'skills'
    (skills / 'pyscf' / 'references').mkdir(parents=True)
    (skills / 'pyscf' / 'SKILL.md').write_text('# pyscf skill')
    (skills / 'pyscf' / 'references' / 'api.md').write_text('# api')
    # A data asset a reference points at: served too, so the external harness sees
    # the same inputs the self-managed SkillsCapability would.
    (skills / 'pyscf' / 'references' / 'basis.json').write_text('{"basis": "6-31g*"}')
    (skills / 'pyscf' / 'scripts').mkdir()
    (skills / 'pyscf' / 'scripts' / 'run_dft.py').write_text('# helper')
    # Compiled bytecode is not domain context.
    (skills / 'pyscf' / 'scripts' / '__pycache__').mkdir()
    (skills / 'pyscf' / 'scripts' / '__pycache__' / 'run_dft.pyc').write_text('junk')
    # A binary asset with no text mime is not served.
    (skills / 'pyscf' / 'references' / 'diagram.png').write_bytes(b'\x89PNG\r\n')

    server = create_server(toolkit, skills_root=skills)

    async def go():
        async with Client(server) as client:
            resources = await client.list_resources()
            body = await client.read_resource('skill://pyscf/SKILL.md')
            return resources, body

    resources, body = _run(go())

    uris = {str(r.uri) for r in resources}
    assert uris == {
        'skill://pyscf/SKILL.md',
        'skill://pyscf/references/api.md',
        'skill://pyscf/references/basis.json',
        'skill://pyscf/scripts/run_dft.py',
    }
    assert body[0].text == '# pyscf skill'


def test_missing_skills_directory_is_not_fatal(toolkit, tmp_path):
    server = create_server(toolkit, skills_root=tmp_path / 'nope')

    async def go():
        async with Client(server) as client:
            return await client.list_resources()

    assert _run(go()) == []


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

def test_cli_flags_override_the_environment(monkeypatch):
    from qmagent.mcp_server import _build_parser

    monkeypatch.setenv('QMAGENT_RESNAME', 'ENV')
    args = _build_parser().parse_args(['--resname', 'FLG'])
    assert args.resname == 'FLG'


def test_config_falls_back_to_the_environment(monkeypatch):
    from qmagent.mcp_server import _build_parser

    monkeypatch.setenv('QMAGENT_RESNAME', 'ENV')
    monkeypatch.setenv('QMAGENT_GPU', '0')
    args = _build_parser().parse_args([])
    assert args.resname == 'ENV'
    assert args.use_gpu is False


def test_blank_gpu_env_keeps_the_default(monkeypatch):
    """QMAGENT_GPU='' is a blanked var, not a request for CPU: the documented GPU
    default must survive it."""
    from qmagent.mcp_server import _build_parser

    monkeypatch.setenv('QMAGENT_GPU', '')
    args = _build_parser().parse_args([])
    assert args.use_gpu is True


def test_server_config_defaults_are_laptop_safe():
    config = ServerConfig()
    assert config.exchange == 'local'
    assert config.resname == 'LIG'
