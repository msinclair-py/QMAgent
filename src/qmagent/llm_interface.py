"""Self-managed harness: the pydantic-ai orchestrator.

This is one of the two ways to drive the execution layer. Here the harness is
ours: a pydantic-ai agent playing "computational chemist", equipped with a
capability stack (tool search, extended thinking, summarization, web search,
filesystem console, memory, skills, a researcher subagent, a TODO planner and
input/tool/secret shields) and returning a typed ``ParameterizationSummary``. The
alternative is ``mcp_server``, where an external harness (Claude Code, Codex, ...)
plays that role instead and none of this module is involved.

The QM tools are deliberately *not* defined here. They live in ``tools.QMToolkit``
and are shared verbatim with the MCP server; ``qm_toolset`` only wraps them so
that a ``QMToolError`` becomes a ``ModelRetry`` the model can act on. What this
module owns is what belongs to pydantic-ai alone.
"""

from dataclasses import dataclass, field
from pathlib import Path
from pydantic import BaseModel
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai import ModelSettings, ModelRetry
from pydantic_ai.capabilities import Thinking, ToolSearch, WebSearch
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai_backends import ConsoleCapability, LocalBackend, ensure_async
from pydantic_ai_backends.permissions import READONLY_RULESET
from pydantic_ai_shields import InputGuard, SecretRedaction, ToolGuard
from pydantic_ai_skills import SkillsCapability
from pydantic_ai_summarization import ContextManagerCapability
from pydantic_ai_todo import TodoCapability, AsyncMemoryStorage
from pydantic_deep import MemoryCapability, StuckLoopDetection
from subagents_pydantic_ai import SubAgentCapability, SubAgentConfig
from typing import Any

from .prompts import RESEARCH_SUBAGENT_PROMPT, SYSTEM_PROMPT
from .tools import QMToolkit, translate_qm_errors

# Context-window threshold at which the summarization capability compacts history.
# Distinct from ModelSettings.max_tokens, which caps a single response's output.
context_max_tokens = 120_000


@dataclass
class QMDeps:
    """Run-scoped state injected into every pydantic-ai tool call.

    The QM run's state is *not* here -- it lives in ``tools.QMRunState``, which the
    MCP server shares. What remains is the filesystem backend that the console and
    memory capabilities read off ``ctx.deps.backend`` by convention. Must be async:
    the memory toolset awaits read_bytes/read directly, while the console toolset
    wraps with ensure_async either way. Read-only enforcement lives in the
    capability's READONLY_RULESET, not here.
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

orchestrator = PydanticAgent[QMDeps, ParameterizationSummary](
    model,
    deps_type=QMDeps,
    output_type=ParameterizationSummary,
    model_settings=ModelSettings(temperature=0.8, max_tokens=10000),
    instructions=SYSTEM_PROMPT,  # instructions (not system_prompt) so only the current agent's prompt reaches the model
    capabilities=[
        ToolSearch(),
        Thinking('xhigh'),
        ContextManagerCapability(max_tokens=context_max_tokens),
        WebSearch(),
        ConsoleCapability(permissions=READONLY_RULESET),  # grep, glob, ls, read
        MemoryCapability(agent_name='quantum-agent'),
        SkillsCapability(directories=['./skills']),
        SubAgentCapability(subagents=[
            SubAgentConfig(
                name='researcher',
                description='Deep research on a topic',
                instructions=RESEARCH_SUBAGENT_PROMPT
            ),
        ]),
        TodoCapability(enable_subtasks=True, async_storage=AsyncMemoryStorage()),
        InputGuard(guard=lambda p: 'ignore previous instructions' not in p.lower()),
        ToolGuard(blocked=['rm']),
        SecretRedaction(),
        StuckLoopDetection(),
    ]
)


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
