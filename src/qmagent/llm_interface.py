"""Self-managed harness: the pydantic-ai orchestrator.

This is one of the two ways to drive the execution layer. Here the harness is
ours: a pydantic-ai agent playing "computational chemist", equipped with a
capability stack (tool search, extended thinking, summarization, web search,
memory, skills, a TODO planner and input/tool/secret shields) and returning a
typed ``ParameterizationSummary``. The alternative is ``mcp_server``, where an
external harness (Claude Code, Codex, ...) plays that role instead and none of
this module is involved.

The QM tools are deliberately *not* defined here. They live in ``tools.QMToolkit``
and are shared verbatim with the MCP server; ``qm_toolset`` only wraps them so
that a ``QMToolError`` becomes a ``ModelRetry`` the model can act on. What this
module owns is what belongs to pydantic-ai alone: the capability stack, the
deferred-tool-call guard, and lazy construction of the agent.
"""

from dataclasses import dataclass, field
from pathlib import Path
from pydantic import BaseModel
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai import ModelSettings, ModelRetry, RunContext
from pydantic_ai.capabilities import (
    HandleDeferredToolCalls,
    Thinking,
    ToolSearch,
    WebSearch,
)
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolDenied
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai_backends import LocalBackend, ensure_async
from pydantic_ai_shields import InputGuard, SecretRedaction, ToolGuard
from pydantic_ai_skills import SkillsCapability
from pydantic_ai_summarization import ContextManagerCapability
from pydantic_ai_todo import TodoCapability, AsyncMemoryStorage
from pydantic_deep import MemoryCapability, StuckLoopDetection
from typing import Any

from .prompts import SYSTEM_PROMPT
from .tools import QMToolkit, translate_qm_errors

# Context-window threshold at which the summarization capability compacts history.
# Distinct from ModelSettings.max_tokens, which caps a single response's output.
#
# This is an *overflow guard*, not a cost control, and it is worth being clear
# about which. Measured across three real runs it never fired once -- the most
# expensive task (28 requests, 1.53M input tokens) peaked around 97k on its final
# request, comfortably under the threshold. Compaction is what stops a long
# parameterization run walking off the end of the context window; it is not what
# makes a run cheap. The thing that actually drives the token bill -- clipping
# verbose run_code output -- lives in tools.py (MAX_TOOL_OUTPUT_CHARS), because it
# is a tool-layer concern both harnesses share.
context_max_tokens = 120_000


@dataclass
class QMDeps:
    """Run-scoped state injected into every pydantic-ai tool call.

    The QM run's state is *not* here -- it lives in ``tools.QMRunState``, which the
    MCP server shares. What remains is the filesystem backend that the memory
    capability reads off ``ctx.deps.backend`` by convention. Must be async: the
    memory toolset awaits read_bytes/read directly.
    """
    backend: Any = field(default_factory=lambda: ensure_async(LocalBackend()))


class ParameterizationSummary(BaseModel):
    """Final deliverables of a parameterization run."""
    resname: str
    smiles: str
    final_energy_ha: float
    net_charge: int
    n_torsions_fit: int
    lib_file: Path
    frcmod_file: Path
    refined_frcmod: Path
    prmtop: Path
    experiment_json: Path
    notes: str = ''


model = 'openai:gpt-5.5'


def _resolve_deferred(ctx: RunContext[QMDeps],
                      requests: DeferredToolRequests) -> DeferredToolResults:
    """Deny approval-required tool calls instead of letting them kill the run.

    When a toolset registers a tool with ``requires_approval=True`` and nothing
    can answer the approval request, pydantic-ai raises ``UserError`` ("a
    deferred tool call was present, but DeferredToolRequests is not among output
    types") and the entire run dies, discarding all work and spend. That is not
    a hypothetical: a CH4 + .OH barrier task died this way twice, ~1.5M tokens in
    each time, after reaching for ConsoleCapability's ``run_in_background``.

    ConsoleCapability has since been dropped, so nothing in the capability list
    below demands approval today. This stays as a floor: the failure mode is
    catastrophic and silent-until-fatal, the guard costs nothing (it adds no
    tool schema), and any capability added later could reintroduce it.

    This agent runs unattended, so there is no human to approve anything.
    Denying with an explanatory message keeps the run alive and points the model
    at ``run_code``, which covers the legitimate uses.

    Arguments:
        ctx (RunContext[QMDeps]): The active run context.
        requests (DeferredToolRequests): Calls awaiting approval or external
            execution.

    Returns:
        (DeferredToolResults): A denial for every pending call.
    """
    results = DeferredToolResults()

    for call in requests.approvals:
        results.approvals[call.tool_call_id] = ToolDenied(
            message=(
                f'{call.tool_name!r} needs approval, but this agent runs '
                f'unattended with no one to grant it, and its filesystem '
                f'permissions are read-only. Use run_code for anything that '
                f'must execute code or write files -- it runs in a sandboxed '
                f'subprocess in the run output directory.'
            )
        )

    # Externally-executed tools: nothing here can execute them, so return the
    # reason as the tool's result rather than stalling the run.
    for call in requests.calls:
        results.calls[call.tool_call_id] = (
            f'{call.tool_name!r} requires external execution, which this '
            f'deployment does not provide. Use run_code instead.'
        )

    return results


def _build_orchestrator() -> PydanticAgent[QMDeps, ParameterizationSummary]:
    """Construct the coordinator agent (capabilities only; tools come per run).

    Called lazily by the module's ``__getattr__`` the first time ``orchestrator``
    is accessed, and cached thereafter. Constructing an Agent resolves its model
    string through ``infer_model``, which builds the provider and therefore
    demands a credential *exist*. Doing that at module import made ``import
    qmagent.llm_interface`` fail outright with a ``UserError`` on any machine
    without ``OPENAI_API_KEY`` -- so the data models, the deps dataclass and the
    tool functions could not be imported or tested offline, and even ``--help``
    needed an API key.

    Deferring it means a credential is required when you actually build the
    agent (i.e. when you are about to run it), not when you import the module
    that defines it.

    The QM tools are *not* registered here. They are bound to a run's ``QMAgent``
    handle, which does not exist at construction time, so they are supplied per
    call via ``orchestrator.run(..., toolsets=[qm_toolset(toolkit)])``.

    Returns:
        (PydanticAgent): The configured coordinator agent.
    """
    return PydanticAgent[QMDeps, ParameterizationSummary](
        model,
        deps_type=QMDeps,
        output_type=ParameterizationSummary,
        # No temperature: `model` is a reasoning model, and providers reject or
        # drop sampling parameters when reasoning is enabled ("Sampling
        # parameters ['temperature'] are not supported when reasoning is
        # enabled. These settings will be ignored."). Setting it read as a
        # deliberate diversity knob while doing nothing. Re-add it only
        # alongside a non-reasoning model.
        model_settings=ModelSettings(max_tokens=10000),
        # instructions (not system_prompt) so only the current agent's prompt
        # reaches the model
        instructions=SYSTEM_PROMPT,
        capabilities=[
            ToolSearch(),
            Thinking('xhigh'),
            ContextManagerCapability(max_tokens=context_max_tokens),
            WebSearch(),
            # Deliberately NOT here (measured over three real runs, see below):
            #
            # ConsoleCapability -- 23 calls, all ls/read_file/grep/glob, which
            #   run_code already does from inside the run's output directory. It
            #   also advertised write_file/execute/run_in_background despite the
            #   READONLY_RULESET name, and the one run_in_background call killed
            #   a run outright (~1.5M tokens). ~1,400 tokens of schema per
            #   request to duplicate run_code and carry a trapdoor.
            #
            # SubAgentCapability -- 0 calls. Never once delegated to the
            #   'researcher' subagent. It cost ~865 tokens of task-management
            #   schema per request, and compiling its sub-agent at import is
            #   what made this module unimportable without a credential.
            MemoryCapability(agent_name='quantum-agent'),
            SkillsCapability(directories=['./skills']),
            TodoCapability(enable_subtasks=True, async_storage=AsyncMemoryStorage()),
            InputGuard(guard=lambda p: 'ignore previous instructions' not in p.lower()),
            ToolGuard(blocked=['rm']),
            SecretRedaction(),
            StuckLoopDetection(),
            # Must be present whenever a toolset registers approval-required tools.
            # Without it the first such call raises UserError and kills the run
            # outright -- see _resolve_deferred.
            HandleDeferredToolCalls(_resolve_deferred),
        ],
    )


_orchestrator: PydanticAgent[QMDeps, ParameterizationSummary] | None = None


def __getattr__(name: str) -> Any:
    """Build ``orchestrator`` on first access, then cache it (PEP 562).

    Keeps ``from qmagent.llm_interface import QMDeps`` (and ``qm_toolset``, and
    ParameterizationSummary) working with no provider credential, while
    ``orchestrator`` itself still resolves normally for anyone about to run it.
    """
    if name == 'orchestrator':
        global _orchestrator
        if _orchestrator is None:
            _orchestrator = _build_orchestrator()
        return _orchestrator
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def qm_toolset(toolkit: QMToolkit) -> FunctionToolset[QMDeps]:
    """Expose the shared QM tools to the orchestrator.

    Built per run rather than at import, because a toolkit is bound to that run's
    QMAgent handle. Each tool's ``QMToolError`` becomes a ``ModelRetry`` so the
    model can self-correct. Pass the result to
    ``orchestrator.run(..., toolsets=[...])``.
    """
    return FunctionToolset(
        tools=[translate_qm_errors(fn, ModelRetry) for fn in toolkit.tool_functions()]
    )
