"""Tests for the orchestrator's run-level robustness.

These use pydantic-ai's FunctionModel, so they exercise the real agent -- its
real capability stack and output contract -- without a provider credential or a
network call.
"""

import os


# The orchestrator is constructed at import time and infers its model, which
# needs a provider credential to exist (not to be valid). Supply a dummy before
# importing so the suite runs on a machine with no API key configured.
os.environ.setdefault('OPENAI_API_KEY', 'test-key-not-used')

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart  # noqa: E402
from pydantic_ai.models.function import AgentInfo, FunctionModel  # noqa: E402

from qmagent.llm_interface import QMDeps, orchestrator  # noqa: E402


def test_approval_required_tool_call_does_not_kill_the_run(tmp_path):
    """A tool that demands approval must be denied, not crash the whole run.

    ConsoleCapability registers run_in_background/execute/write_file with
    requires_approval=True. With no deferred-call handler, pydantic-ai raises
    UserError ("a deferred tool call was present, but DeferredToolRequests is
    not among output types") and the run dies -- discarding every result and
    every token already spent. Regression guard for that crash.
    """
    calls: list[str] = []

    def respond(messages, info: AgentInfo) -> ModelResponse:
        if not calls:
            # First turn: reach for an approval-required console tool, exactly
            # as the model did on the CH4 + .OH barrier task.
            calls.append('run_in_background')
            return ModelResponse(parts=[ToolCallPart(
                tool_name='run_in_background',
                args={'command': 'python long_running_qm.py'},
            )])
        # Second turn: having been denied, answer normally.
        return ModelResponse(parts=[TextPart('understood, using run_code instead')])

    deps = QMDeps(qm=None, output_path=tmp_path, resname='LIG')

    with orchestrator.override(model=FunctionModel(respond)):
        # The assertion is that this does not raise UserError. The output type
        # is irrelevant here, so take free-form text.
        result = orchestrator.run_sync(
            'start a long job in the background', deps=deps, output_type=str,
        )

    assert calls == ['run_in_background'], 'the approval-required tool was never called'
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
