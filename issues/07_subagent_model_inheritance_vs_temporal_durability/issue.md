# A `TemporalDurability`-bound sub-agent can never inherit the parent's runtime model

> Repo: **pydantic-ai-harness** -- suggested labels: `bug`, `capability`
> Found while: wiring a real `SubAgents` delegate that also needs `TemporalDurability`
> (durable environments, Phase 5 of a consumer's harness migration)

## Problem

`SubAgentToolset.delegate_task` decides which model a delegation uses with:

```python
# A sub-agent with no model of its own (e.g. one loaded from disk) inherits
# the parent run's model; one that brought its own keeps it.
model = None if sub_agent.agent.model is not None else ctx.model
```

The intent (per the comment) is reasonable: a disk-loaded delegate with no model of its own
should inherit the parent's; a delegate constructed with its own deliberate model choice (e.g. a
cheaper model for extraction) should keep it.

But `TemporalDurability.for_agent` independently *requires* a concrete `Model` at construction:

```python
if not isinstance(agent.model, Model):
    raise UserError(
        "An agent needs to have a concrete `model` in order to be used with Temporal, "
        "it cannot be set at agent run time."
    )
```

`pydantic_ai.Agent.__init__` also resolves *any* model argument -- including the special `'test'`
string -- into a concrete `Model` instance immediately, so `agent.model` is `None` only when no
model was passed to `Agent(...)` at all. Combining these two facts: **every delegate that carries
`TemporalDurability` unavoidably has `agent.model is not None`**, so `delegate_task`'s heuristic
always takes the "keep your own model" branch for it -- there is no way, through the public API,
for such a delegate to inherit the parent's actual runtime-resolved model instead.

This isn't a narrow edge case: any consumer needing durable sub-agent delegation with a
per-run/per-tenant model (a different provider or API key depending on which user's workflow is
running -- exactly what `TemporalDurability(provider_factory=...)` exists to support) hits this.
The delegate's construction-time model becomes a permanent placeholder (in the reproducing case,
literally pydantic-ai's `'test'` string, i.e. `TestModel()`) that silently never gets overridden,
in production as much as in tests.

## Why the model round-trips correctly once inherited (confirmed)

Passing a concrete `Model` instance as `model=` to a `TemporalDurability`-bound agent's `.run()`
already works today: the model is round-tripped by its `model_id` string across the activity
boundary (`_find_model_id`/`_resolve_model_id`, `durable_exec/temporal/_durability.py`) and rebuilt
worker-side via the *bound agent's own* `provider_factory` if the id isn't in its local registry.
A delegate sharing the parent's `deps` (the default, `SubAgentToolset` always forwards `ctx.deps`
unchanged) and the same shape of `provider_factory` rebuilds the identical provider/API key. The
gap is purely in `delegate_task` never being told to pass the parent's model through in this case
-- not in whether the round-trip itself works.

## Proposal

Add `SubAgent.inherit_model: bool = False` (a new field, defaulting to preserve today's behavior)
that forces `delegate_task` to use `ctx.model` regardless of whether `sub_agent.agent.model` is
set:

```python
model = ctx.model if sub_agent.inherit_model else (None if sub_agent.agent.model is not None else ctx.model)
```

Documented as the knob to set when a delegate needs `TemporalDurability` (hence a concrete
construction-time model to satisfy that requirement) but should still run with whatever model the
parent is actually configured with for this run, not its own placeholder.

## Acceptance

- A `SubAgent(durable_agent, inherit_model=True)` delegation runs with `ctx.model`, verified by a
  parent/child pair using different `FunctionModel`s where the child's actual behavior proves
  which model executed (not just an assertion on `sub_agent.agent.model`).
- Existing behavior (`inherit_model` unset/`False`) is unchanged: disk-loaded delegates still
  inherit when they have no model, delegates with a deliberate model choice still keep it.
- Docs for `SubAgent.workspace`'s existing "Under Temporal" note gain a cross-reference to this
  field, since both concern the same "delegate needs `TemporalDurability`" scenario.
