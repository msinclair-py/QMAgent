"""Tests for the orchestrator's run-level robustness.

These use pydantic-ai's FunctionModel, so they exercise the real agent -- its
real capability stack and output contract -- without a provider credential or a
network call.
"""

import os


# Importing the module needs no credential, but this file imports `orchestrator`
# itself, which builds the agent and so infers the model -- that needs a key to
# *exist*, not to be valid. Supply a dummy so the suite runs with none
# configured. (test_module_imports_without_a_provider_credential asserts the
# import-only path in a subprocess with the key genuinely absent.)
os.environ.setdefault('OPENAI_API_KEY', 'test-key-not-used')

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart  # noqa: E402
from pydantic_ai.models.function import AgentInfo, FunctionModel  # noqa: E402

from qmagent.llm_interface import QMDeps, orchestrator  # noqa: E402


def test_approval_required_tool_call_does_not_kill_the_run(tmp_path):
    """A tool that demands approval must be denied, not crash the whole run.

    Any toolset may register a tool with requires_approval=True. Without a
    deferred-call handler, the first such call makes pydantic-ai raise UserError
    ("a deferred tool call was present, but DeferredToolRequests is not among
    output types") and the run dies, discarding every result and every token
    already spent. That is exactly how the CH4 + .OH barrier task died, twice,
    ~1.5M tokens in, when it reached for ConsoleCapability's run_in_background.

    ConsoleCapability has since been removed, so nothing in the stock capability
    list demands approval any more -- which would make a test that leans on it
    vacuous (it would pass with the handler deleted). Register an
    approval-required tool explicitly instead, so this keeps guarding the
    handler itself rather than a capability that happens to be absent.
    """
    from pydantic_ai import RunContext
    from pydantic_ai.toolsets import FunctionToolset

    extra = FunctionToolset()

    @extra.tool(requires_approval=True)
    def launch_detached_job(ctx: RunContext[QMDeps], command: str) -> str:
        """Stand-in for any tool a toolset marks requires_approval=True."""
        raise AssertionError('a denied tool must never actually execute')

    calls: list[str] = []

    def respond(messages, info: AgentInfo) -> ModelResponse:
        if not calls:
            calls.append('launch_detached_job')
            return ModelResponse(parts=[ToolCallPart(
                tool_name='launch_detached_job',
                args={'command': 'python long_running_qm.py'},
            )])
        # Second turn: having been denied, answer normally.
        return ModelResponse(parts=[TextPart('understood, using run_code instead')])

    deps = QMDeps(qm=None, output_path=tmp_path, resname='LIG')

    with orchestrator.override(model=FunctionModel(respond)):
        # The assertion is that this does not raise UserError.
        result = orchestrator.run_sync(
            'start a long job in the background', deps=deps,
            output_type=str, toolsets=[extra],
        )

    assert calls == ['launch_detached_job'], 'the approval-required tool was never called'
    assert isinstance(result.output, str)


def test_run_code_forwards_its_timeout_to_the_agent(tmp_path):
    """run_code must let the model choose a timeout, and default generously.

    execute_code's own default is 300s, which does not fit real QM: an
    open-shell TS search plus a Hessian blows straight through it. When the tool
    hardcoded that default, the model's only way out was to look for background
    execution -- which is what tripped the approval-required crash. The tool
    therefore owns the policy and always passes it explicitly.
    """
    seen: dict[str, object] = {}

    class _FakeAgent:
        async def execute_code(self, code, workdir=None, extra_paths=None,
                               timeout=None):
            seen.update(code=code, timeout=timeout)
            return {'stdout': 'ok', 'stderr': '', 'returncode': '0'}

    def respond(messages, info: AgentInfo) -> ModelResponse:
        if 'timeout' not in seen:
            return ModelResponse(parts=[ToolCallPart(
                tool_name='run_code',
                args={'code': 'print(1)', 'timeout': 5400.0},
            )])
        return ModelResponse(parts=[TextPart('done')])

    deps = QMDeps(qm=_FakeAgent(), output_path=tmp_path, resname='LIG')
    with orchestrator.override(model=FunctionModel(respond)):
        orchestrator.run_sync('run some code', deps=deps, output_type=str)

    assert seen['timeout'] == 5400.0, 'run_code did not forward the timeout'


def test_run_code_default_timeout_fits_real_qm():
    """The default must be long enough for a saddle-point search plus a Hessian."""
    import inspect

    from qmagent.llm_interface import run_code

    default = inspect.signature(run_code).parameters['timeout'].default
    # 300s (execute_code's own default) is empirically too short: an HCN TS
    # search alone took ~175s at def2-SVP, and CH4 + .OH is far bigger.
    assert default >= 1800.0


def test_denial_message_points_the_model_at_run_code(tmp_path):
    """The denial must be actionable: tell the model what to use instead.

    A bare "denied" teaches the model nothing and invites a retry loop; the
    message has to name the escape hatch that actually works.
    """
    from pydantic_ai.tools import DeferredToolRequests

    from qmagent.llm_interface import _resolve_deferred

    call = ToolCallPart(tool_name='run_in_background', args={'command': 'x'},
                        tool_call_id='abc123')
    results = _resolve_deferred(None, DeferredToolRequests(approvals=[call]))

    denial = results.approvals['abc123']
    assert 'run_code' in denial.message
    assert 'run_in_background' in denial.message


def test_module_imports_without_a_provider_credential():
    """Importing llm_interface must not require an API key.

    The orchestrator used to be constructed at module scope, so `import
    qmagent.llm_interface` called infer_model -> OpenAIProvider() and raised
    UserError on any machine without OPENAI_API_KEY. That made the data models,
    QMDeps and the tool functions untestable offline, and meant even reading the
    configured model name needed a credential. Run in a subprocess so the
    module is imported fresh with the key genuinely absent.
    """
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k != 'OPENAI_API_KEY'}
    proc = subprocess.run(
        [sys.executable, '-c',
         'from qmagent.llm_interface import QMDeps, ParameterizationSummary, model; '
         'print(model)'],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert proc.returncode == 0, f'import needed a credential:\n{proc.stderr[-600:]}'
    assert 'gpt' in proc.stdout


def test_all_tools_are_registered_on_the_built_agent():
    """Every tool must survive the move off @orchestrator.tool decorators."""
    from qmagent.llm_interface import orchestrator

    registered = set(orchestrator._function_toolset.tools)
    expected = {
        'run_code', 'build_compound', 'geometry_optimization', 'compute_esp',
        'scan_torsions', 'fit_resp_charges', 'integrate_amber_ff',
        'fit_torsions', 'run_parameterization_pipeline',
    }
    assert expected <= registered, f'missing: {sorted(expected - registered)}'


def test_orchestrator_is_cached_not_rebuilt():
    """__getattr__ must cache: rebuilding per access would be slow and lose overrides."""
    import qmagent.llm_interface as li

    assert li.orchestrator is li.orchestrator
