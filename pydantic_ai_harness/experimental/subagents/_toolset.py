"""Sub-agent toolset: a single delegate tool that runs named child agents."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Literal

from pydantic_ai.agent import AbstractAgent, EventStreamHandler
from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.exceptions import ModelRetry, UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset

# Private import: pydantic-ai has no public way to tell capability-contributed
# toolsets apart from the agent's own in `agent.toolsets`.
from pydantic_ai.toolsets._capability_owned import CapabilityOwnedToolset
from pydantic_ai.usage import UsageLimits


@dataclass(frozen=True)
class SubAgent(Generic[AgentDepsT]):
    """One delegate: a child agent plus its per-delegate run controls.

    Pass a sequence of these as `SubAgents(agents=[...])`. The delegate's name --
    how the parent model refers to it, and how it is listed in the system prompt --
    is `name` when set, otherwise the agent's own `name`. An agent with neither is
    rejected by `SubAgents`.

    Every control below is optional; an unset field leaves the corresponding
    behaviour at the `SubAgents` default.
    """

    agent: AbstractAgent[AgentDepsT, Any]
    """The agent that runs when this delegate is invoked."""

    name: str | None = None
    """Name the parent model uses to delegate to this agent. Defaults to the
    agent's own `name` when unset."""

    description: str | None = None
    """Description for the system-prompt listing. Defaults to the agent's own
    `description` when unset; a delegate with neither is listed by name alone."""

    usage_limits: UsageLimits | None = None
    """Request/token budget for one delegation. When set, the child runs with
    its own usage accounting so the budget counts only the child's own requests
    and tokens (not the parent's or siblings'), even when `forward_usage=True`.
    The tradeoff: that child's tokens no longer aggregate into the parent's
    `usage`. Hitting this budget is a soft outcome (steering message), not a
    run-stopping `UsageLimitExceeded`."""

    timeout_seconds: float | None = None
    """Wall-clock budget for one delegation. When the child exceeds it, the run
    is cancelled and the parent gets a soft steering message instead of hanging
    on the child."""

    max_calls: int | None = None
    """Maximum number of delegations to this sub-agent per parent run. Once
    reached, further delegations return a soft budget-exhausted message without
    running the child."""

    on_failure: str | None = None
    """Steering message returned to the parent for any soft degradation of this
    delegate (timeout, child failure, usage budget reached, call budget
    exhausted), in place of the built-in default. Setting it also makes child
    failures soft: a child error returns this message as a normal tool result
    instead of raising a parent `ModelRetry`."""

    workspace: Literal['shared', 'branch'] = 'branch'
    """Which durable-environment workspace this delegate's run sees, when the
    parent run holds a `DurableEnvironment` lease.

    - `'branch'` (default): the sub-agent works on its own git branch, forked
      from the parent's current state, merged back after the run completes.
      Isolates the delegation from concurrent edits (the parent's own, or a
      sibling delegation's) at the cost of a merge step. A merge conflict
      triggers the self-heal loop: the parent's head is materialized into the
      sub-agent's branch as ordinary conflict markers, the sub-agent is
      relaunched to resolve them, and the merge is retried, up to
      `max_merge_retries` times before falling back to reporting the conflict
      to the parent model with the parent's workspace left untouched.
    - `'shared'`: the sub-agent works directly in the parent's live workspace,
      same branch, no fork or merge. Concurrent writes are serialized, not
      isolated -- the sub-agent can see and be affected by edits that aren't
      its own.

    Has no effect when the parent run holds no `DurableEnvironment` lease: the
    sub-agent just runs normally regardless of this setting.

    Least privilege: `SubAgent` has no `allowed_patterns`/`protected_patterns`
    of its own. `'branch'` already gives a delegate its own root for free (its
    own git branch, checked out to its own workspace directory) -- to bound
    which paths within that root it can touch, give that delegate's own
    `Agent` its own `FileSystem(allowed_patterns=..., protected_patterns=...)`
    capability, not something injected per-call via `shared_capabilities`
    (`DurableEnvironmentPlugin` only wires a toolset's root/durability at
    Worker-registration time, from each agent's own construction-time
    toolsets).

    Under Temporal, a delegate's run (via `nested_agent_run`) executes inside
    its own child workflow, but that child workflow does not support
    continue-as-new: if the delegate's own `Agent` carries
    `TemporalDurability(continue_as_new='auto')` (the default) and its history
    grows enough to trigger a pause, that exception propagates uncaught and
    genuinely fails the child workflow instead of continuing it gracefully
    (a known pydantic-ai limitation -- see `ToolCallWorkflow`'s docstring).
    This matters more for `'branch'` than `'shared'`: the self-heal loop can
    relaunch the delegate's run multiple times within the same child workflow.
    Set `continue_as_new=False` on a delegate's own `TemporalDurability` and
    bound its runs with `usage_limits`/`max_merge_retries` instead. See also
    `inherit_model` if this delegate also needs to run with the parent's actual
    model rather than its own construction-time placeholder.
    """

    max_merge_retries: int = 1
    """`'branch'`-workspace only: how many self-heal rounds (materialize
    conflict markers into the sub-agent's branch, relaunch it to resolve them,
    retry the merge) to attempt before giving up and reporting the conflict to
    the parent model instead, leaving the parent's workspace unchanged."""

    inherit_model: bool = False
    """Force this delegation to use the parent run's model (`ctx.model`) even though
    `agent.model` is set. Off by default -- a delegate's own model is normally a deliberate
    choice (a cheaper model for extraction, a disk-loaded agent's inherited default) and should
    be kept.

    Needed specifically when the delegate's `Agent` also carries `TemporalDurability`: that
    capability requires a concrete `model` at construction (`agent.model` can never be `None`
    for such an agent), which would otherwise always win over inheriting the parent's actual
    runtime-resolved model -- exactly the model a per-run/per-tenant `provider_factory` exists
    to vary. The delegate's construction-time model still has to be *something* concrete to
    satisfy `TemporalDurability`; with `inherit_model=True` it's never actually used, just a
    placeholder satisfying that requirement.
    """

    @property
    def resolved_name(self) -> str | None:
        """The delegate's name: `name` if set, else the agent's own `name`."""
        return self.name or self.agent.name


def _is_capability_contributed(toolset: AbstractToolset[AgentDepsT]) -> bool:
    """Whether `toolset`'s tree contains a `CapabilityOwnedToolset`."""
    found = False

    def visit(node: AbstractToolset[AgentDepsT]) -> None:
        nonlocal found
        if isinstance(node, CapabilityOwnedToolset):
            found = True

    toolset.apply(visit)
    return found


class SubAgentToolset(FunctionToolset[AgentDepsT]):
    """Exposes one delegate tool that dispatches a task to a named sub-agent.

    Each delegation runs the child agent in a fresh run with its own message
    history, so the sub-agent never sees the parent conversation. The parent's
    `deps` are forwarded; its `usage` is shared when enabled; its tools are
    inherited when enabled; any `shared_capabilities` are applied to every
    sub-agent run; and sub-agent events are streamed to `event_stream_handler`
    when one is set. Per-delegate run controls come from each `SubAgent`.
    """

    def __init__(
        self,
        *,
        agents: Mapping[str, SubAgent[AgentDepsT]],
        forward_usage: bool,
        inherit_tools: bool,
        shared_capabilities: Sequence[AgentCapability[AgentDepsT]],
        event_stream_handler: EventStreamHandler[AgentDepsT] | None,
        tool_name: str,
        tool_retries: int | None,
        call_counts: dict[str, dict[str, int]],
        # TODO: Non toolset ne devrait pas savoir task queue qui est uniquement temporal coupled
        host_task_queue: str | None = None,
    ) -> None:
        # An explicit `id` is required for this toolset to be usable with any durable-execution
        # engine (Temporal, DBOS, Prefect): they identify a toolset's activities/tasks/workflows
        # by it. `tool_name` is already unique per `SubAgents` capability on an agent, so it
        # doubles as a stable id without asking the caller for another name.
        super().__init__(id=tool_name)
        self._agents: dict[str, SubAgent[AgentDepsT]] = dict(agents)
        self._forward_usage = forward_usage
        self._inherit_tools = inherit_tools
        self._shared_capabilities = list(shared_capabilities)
        self._event_stream_handler = event_stream_handler
        self._tool_name = tool_name
        self._host_task_queue = host_task_queue
        # Run-scoped delegation counts, keyed by run_id then sub-agent name.
        # Shared with the capability, which clears each run's entry in wrap_run.
        self._call_counts = call_counts
        self.add_function(
            self.delegate_task,
            name=tool_name,
            retries=tool_retries,
            # Engine-neutral fact: this tool's body runs another agent. A durability
            # capability that knows what to do with it (e.g. pydantic-ai's Temporal
            # `TemporalDurability`, which runs the tool as a child workflow instead of
            # collapsing the whole delegation into one activity) reads this tag; this
            # toolset stays unaware of Temporal or any other specific engine.
            metadata={'nested_agent_run': True},
        )

    def _inherited_toolsets(self, ctx: RunContext[AgentDepsT]) -> list[AbstractToolset[AgentDepsT]] | None:
        """The parent agent's own toolsets, excluding capability-contributed ones.

        Capability toolsets are bound to capability instances registered in the
        parent run; carrying them into the sub-agent's run (where their owner is
        not registered) fails `CapabilityOwnedToolset`'s ownership resolution, and
        the tools would arrive without the hooks and instructions that make them
        work. Use `shared_capabilities` to share a capability with sub-agents.
        The delegate tool itself is also filtered out by name, so delegation
        cannot recurse. When this toolset was registered via the `SubAgents`
        capability the capability filter already drops it; the name filter covers
        direct registration in `Agent(toolsets=[...])`, where nothing wraps it in
        `CapabilityOwnedToolset`.
        """
        agent = ctx.agent
        if agent is None:  # pragma: no cover - the running agent is always set during a run
            return None
        # Capability toolsets surface as `CombinedToolset(CapabilityOwnedToolset(...))`
        # entries, so ownership is detected by walking each tree. Only core's capability
        # assembly constructs `CapabilityOwnedToolset`, so a tree containing one is
        # capability-contributed in its entirety.
        return [
            toolset.filtered(lambda _ctx, tool_def: tool_def.name != self._tool_name)
            for toolset in agent.toolsets
            if not _is_capability_contributed(toolset)
        ]

    def _budget_exhausted(self, ctx: RunContext[AgentDepsT], agent_name: str, max_calls: int) -> bool:
        """Increment this run's delegation count for `agent_name` and report whether it is over budget.

        Runs synchronously before any await, so concurrent delegations in one run
        count without a lock.
        """
        counts = self._call_counts.setdefault(ctx.run_id or '', {})
        counts[agent_name] = counts.get(agent_name, 0) + 1
        return counts[agent_name] > max_calls

    async def delegate_task(self, ctx: RunContext[AgentDepsT], agent_name: str, task: str) -> str:
        """Delegate a self-contained task to a named sub-agent and return its result.

        The sub-agent runs in its own fresh context and does not see this
        conversation, so `task` must contain everything it needs.

        Args:
            ctx: The run context (provides the parent's deps, usage, and tools).
            agent_name: Name of the sub-agent to run. Must be one of the agents
                listed in the instructions.
            task: The complete, self-contained instruction for the sub-agent.
        """
        sub_agent = self._agents.get(agent_name)
        if sub_agent is None:
            available = ', '.join(sorted(self._agents))
            raise ModelRetry(f'Unknown sub-agent {agent_name!r}. Available sub-agents: {available}.')

        if sub_agent.max_calls is not None and self._budget_exhausted(ctx, agent_name, sub_agent.max_calls):
            return self._steer(
                sub_agent.on_failure,
                f'Delegate budget for {agent_name!r} is exhausted for this run '
                f'({sub_agent.max_calls} call(s)). Synthesize from existing evidence and '
                f'choose the next action; do not delegate to {agent_name!r} again.',
            )

        toolsets = self._inherited_toolsets(ctx) if self._inherit_tools else None
        capabilities = self._shared_capabilities or None
        usage_limits: UsageLimits | None
        if sub_agent.usage_limits is not None:
            # Isolated accounting so the per-child budget counts only this child.
            own_budget = True
            usage = None
            usage_limits = sub_agent.usage_limits
        else:
            own_budget = False
            usage = ctx.usage if self._forward_usage else None
            usage_limits = None

        # A sub-agent with no model of its own (e.g. one loaded from disk) inherits
        # the parent run's model; one that brought its own keeps it -- unless
        # inherit_model forces the parent's model regardless (see its docstring:
        # a TemporalDurability-bound delegate always has a concrete model, so this
        # is the only way such a delegate can still track a per-run model).
        if sub_agent.inherit_model:
            model = ctx.model
        else:
            model = None if sub_agent.agent.model is not None else ctx.model
        timeout = sub_agent.timeout_seconds

        async def run_once(task_text: str) -> tuple[str, bool]:
            """Run the sub-agent once.

            `completed=False` means a soft degradation (timeout, usage budget, or a
            swallowed `on_failure` error) with nothing to merge; the text is a
            steering message, not real sub-agent output.
            """
            run = sub_agent.agent.run(
                task_text,
                deps=ctx.deps,
                model=model,
                usage=usage,
                usage_limits=usage_limits,
                toolsets=toolsets,
                capabilities=capabilities,
                event_stream_handler=self._event_stream_handler,
            )
            try:
                result = await (asyncio.wait_for(run, timeout) if timeout is not None else run)
            except asyncio.TimeoutError:
                return (
                    self._steer(
                        sub_agent.on_failure,
                        f'Sub-agent {agent_name!r} exceeded its {timeout}s time budget. '
                        f'Treat this as a recoverable observation and decide from existing evidence.',
                    ),
                    False,
                )
            except UsageLimitExceeded:
                if own_budget:
                    return (
                        self._steer(
                            sub_agent.on_failure,
                            f'Sub-agent {agent_name!r} reached its usage budget. '
                            f'Treat this as a recoverable observation and decide from existing evidence.',
                        ),
                        False,
                    )
                # A shared/parent usage limit means the whole tree is out of budget.
                raise
            except (ModelRetry, UnexpectedModelBehavior) as exc:
                if sub_agent.on_failure is not None:
                    return sub_agent.on_failure, False
                # Soft sub-agent failures come back to the parent as a retry it can react to.
                raise ModelRetry(f'Sub-agent {agent_name!r} failed: {exc}') from exc
            return str(result.output), True

        durable_env = ctx.metadata.get('durable_env') if ctx.metadata else None
        if sub_agent.workspace == 'branch':
            # Lazy import: this toolset must stay importable without `temporalio`
            # installed (see `_branch_delegation`'s own docstring for why).
            from pydantic_ai_harness.durable._branch_delegation import (
                resolve_parent_environment_lease,
                run_with_self_heal,
            )
            from pydantic_ai_harness.durable._store import EnvironmentLease

            parent_lease: EnvironmentLease | None = None
            if durable_env is not None:
                parent_lease = EnvironmentLease.model_validate(durable_env)
            elif self._host_task_queue is not None:
                parent_lease = await resolve_parent_environment_lease(
                    host_task_queue=self._host_task_queue
                )

            if parent_lease is not None:
                return await run_with_self_heal(
                    parent_lease=parent_lease,
                    run_once=run_once,
                    task=task,
                    max_merge_retries=sub_agent.max_merge_retries,
                    host_task_queue=self._host_task_queue,
                )

        output, _completed = await run_once(task)
        return output

    @staticmethod
    def _steer(on_failure: str | None, default: str) -> str:
        """A soft steering message: the delegate's `on_failure` override, else `default`."""
        if on_failure is not None:
            return on_failure
        return default
