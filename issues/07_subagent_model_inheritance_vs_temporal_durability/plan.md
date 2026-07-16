# Plan -- `SubAgent.inherit_model` for `TemporalDurability`-bound delegates

Target: `pydantic_ai_harness/experimental/subagents/_toolset.py`.

## Step 1 -- new field

`SubAgent` (frozen dataclass), after `workspace`/`max_merge_retries` (keeps the durability-related
fields grouped):

```python
inherit_model: bool = False
"""Force this delegation to use the parent run's model (`ctx.model`) even though
`agent.model` is set. Off by default -- a delegate's own model is normally a deliberate choice
(a cheaper model for extraction, a disk-loaded agent's inherited default) and should be kept.

Needed specifically when the delegate's `Agent` also carries `TemporalDurability`: that capability
requires a concrete `model` at construction (`agent.model` can never be `None` for such an agent),
which would otherwise always win over inheriting the parent's actual runtime-resolved model --
exactly the model a per-run/per-tenant `provider_factory` exists to vary. The delegate's
construction-time model still has to be *something* concrete to satisfy `TemporalDurability`; with
`inherit_model=True` it's never actually used, just a placeholder satisfying that requirement.
"""
```

## Step 2 -- `delegate_task`'s model selection

`_toolset.py`, replace:

```python
model = None if sub_agent.agent.model is not None else ctx.model
```

with:

```python
model = ctx.model if sub_agent.inherit_model else (None if sub_agent.agent.model is not None else ctx.model)
```

No other change needed in `delegate_task` -- `model` flows into `sub_agent.agent.run(task_text,
deps=ctx.deps, model=model, ...)` exactly as today; a concrete `Model` instance round-trips
through the delegate's own `TemporalDurability` via `_resolve_model_id`'s existing "unregistered
model -> `provider_factory`" path (verified while writing the issue -- no core change needed
there).

## Step 3 -- docs

- `SubAgent.inherit_model`'s own docstring (above) is the main documentation.
- Add a one-line cross-reference in `SubAgent.workspace`'s "Under Temporal" paragraph: "See also
  `inherit_model` if this delegate also needs to run with the parent's actual model rather than
  its own construction-time placeholder."
- `experimental/subagents/README.md`: extend the existing "Under Temporal" section with a short
  example showing `inherit_model=True` alongside `continue_as_new=False` for a durable delegate.

## Step 4 -- tests

`tests/subagents/test_toolset.py` (or wherever `delegate_task`'s model selection is already
covered):

- `inherit_model=False` (default), delegate has its own model: delegate keeps it (regression,
  matches current behavior).
- `inherit_model=False`, delegate has no model (disk-loaded style): delegate inherits `ctx.model`
  (regression).
- `inherit_model=True`, delegate has its own model: delegate uses `ctx.model` instead -- assert via
  two distinct `FunctionModel`s (parent's and delegate's own) where only the parent's model's
  behavior appears in the result, proving the override actually took effect and isn't just a
  passthrough that happened to look the same.
- A `TemporalDurability`-bound delegate (concrete model forced by construction) with
  `inherit_model=True` still uses the parent's model when delegated to under a real `Worker` --
  the scenario that motivated this issue. Model round-trip via `_resolve_model_id`/
  `provider_factory` on the delegate's own `TemporalDurability` is what's actually being proven
  here, not just that `model=ctx.model` was passed.

## Gates

`make lint && make typecheck && make test` before considering this issue done.
