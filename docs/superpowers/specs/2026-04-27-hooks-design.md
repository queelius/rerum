# Rerum Hooks System: Design Spec

**Date:** 2026-04-27
**Status:** Draft
**Scope:** Engine-level hook system. DSL/metadata work is a follow-up spec; MCP integration is a separate downstream concern.

## Problem

Today the rerum engine has one hook: the `RewriteTrace` listener (task G), an
observer that receives `RewriteStep` events. Several use cases want more:

1. **LLM rule inference.** When the engine reaches an expression where no rule
   matches (or a `(! op ...)` form with an unknown op), call out to an LLM to
   infer a rule or value, install it, and continue rewriting from where it
   left off.
2. **Pretty-printing and progress.** Watchers that observe each step and
   render it for a UI or stream.
3. **Decision-layer policies.** Predicates beyond DSL `when` clauses: budget
   enforcement, A/B testing, debugger breakpoints, conditional firing based
   on engine state the rule itself cannot see.

These all want the same primitive: a way to attach external code to specific
points in the engine's execution and influence behavior at those points.
xtk's earlier attempt at this is informative but predates the structural
cleanup we now have (listener pattern, RuleSet view, EquivalenceClass value
object), so a redesign on the cleaner foundation is the right call.

## Goals

1. **Principled categorization.** Hooks fall into a small fixed set of
   categories whose composition policies are determined by the category, not
   by per-event special cases.
2. **Curated event taxonomy.** A small number of named events at the
   engine's natural decision points. No string-keyed dynamic registry; events
   are stable engine API.
3. **Sync hook contract.** Hooks are plain Python callables. LLM resolvers
   block; downstream callers (the MCP) wrap `engine.simplify()` in a thread
   executor when they need not to block.
4. **Engine-controlled mutation.** Hooks influence engine state through
   structured `Resolution` returns, not direct mutation. The engine decides
   when to apply the mutation, in a way that preserves invariants
   (priority sort, cache validity).
5. **Backward compatibility.** Existing `RewriteTrace`, `simplify(trace=True)`,
   and `condition`/`when` clauses keep working unchanged.

## Non-goals

- Async/await native engine. (Wrap in `asyncio.to_thread()` at the boundary.)
- Transformation hooks (rewriting expressions outside the rule system). The
  Resolver `value` and `rules` paths cover the legitimate use cases without
  blurring the line between "rule" and "code that mutates expressions."
- Mid-pass rule mutation. Rules added by resolvers take effect at the next
  outer-loop iteration, not partway through a single rule application pass.
- Custom user-defined event names. The 8 events are stable engine API. A
  future revision can add a generic registry if a real use case demands it.

## Architecture

### Three categories, three composition policies

Each event belongs to exactly one of three categories. The category determines
how multiple registered hooks for the same event interact.

| Category | Composition policy | Events |
|----------|--------------------|--------|
| **Observer** | broadcast (all run, in registration order, return values ignored) | `rule_applied`, `fixpoint` |
| **Resolver** | chain of responsibility (first non-None Resolution wins; rest skipped) | `no_match`, `undefined_op`, `fold_error`, `max_depth`, `cycle` |
| **Decision** | AND-gate (every hook must return True; first False short-circuits) | `should_fire` |

This means the engine never has to special-case a specific event for
composition. Knowing the category, the engine knows the policy. Adding a new
event in the future is a matter of declaring its category, not writing new
composition logic.

The `should_fire` decision layer sits *on top of* the existing DSL `condition`
mechanism: rule-author guards stay in the DSL (where they are data), while
engine-user predicates go in `should_fire` (where they are code). They
compose via AND. A rule fires iff its `condition` passes AND every
`should_fire` hook returns True.

## Components

### `Resolution`

The structured return type from a Resolver hook. Frozen dataclass.

```python
@dataclass(frozen=True)
class Resolution:
    """Returned by a Resolver hook to override engine default behavior at a
    dead-end. Exactly one of value/rules/fold_funcs/allow_more must be set;
    abort and metadata are orthogonal flags that can co-exist."""
    value: Optional[ExprType] = None
    rules: Optional[List["Rule"]] = None
    fold_funcs: Optional[Dict[str, "FoldHandler"]] = None
    allow_more: Optional[bool] = None
    abort: bool = False
    metadata: Optional[Dict[str, Any]] = None
```

The engine reacts to whichever field is set:

- **`value`**: use this expression as the rewrite output for the current
  position. No retry.
- **`rules`**: append to the engine's rule set (with provenance metadata if
  `metadata` was provided), invalidate the simplifier cache, retry the match
  at the same position.
- **`fold_funcs`**: install in the prelude, retry the `(! op ...)` evaluation.
- **`allow_more=True`**: only meaningful for `max_depth`. Engine doubles the
  budget once and re-fires if still exhausted.
- **`abort=True`**: stop the entire rewrite and propagate up. Engine returns
  whatever it has so far.
- **`metadata`**: free-form dict copied onto the rules' `RuleMetadata`
  (provenance, model name, confidence, derivation explanation, etc.).

Conflicts (e.g. both `value` and `rules` set) raise `ResolutionError` at the
hook return boundary, before the engine acts on the Resolution.

### `HookContext`

Read access to engine state plus controlled mutation primitives. Constructed
by the engine for each hook invocation; not user-instantiable.

```python
class HookContext:
    @property
    def engine(self) -> RuleEngine: ...
    @property
    def expr_path(self) -> List[ExprType]: ...
    @property
    def depth(self) -> int: ...
    @property
    def step_count(self) -> int: ...
    @property
    def event_name(self) -> str: ...
    def cancel(self) -> None: ...
```

`expr_path` gives ancestry from the root expression to the current position,
useful for resolvers that need context-sensitive logic. `depth` is the
recursion depth into the expression tree. `step_count` is the total successful
rule applications in the current `simplify` (or `equivalents`, etc.) call.
`event_name` lets a single hook function attach to multiple events and
dispatch internally.

`cancel()` sets the engine's abort flag, equivalent to returning
`Resolution(abort=True)` from a Resolver. Decisions and Observers also have
access; they cannot return a Resolution but can still cancel.

### Registration API

Each event has its own `on_<event>` and `off_<event>` method on the engine.
`on_<event>` is overloaded as both decorator and method.

```python
# Decorator form (recommended for readability)
@engine.on_no_match
def llm_resolver(expr: ExprType, ctx: HookContext) -> Optional[Resolution]:
    rules = llm_infer_rules(expr, model="claude-opus-4-7")
    return Resolution(rules=rules,
                      metadata={"provenance": "llm-inferred",
                                "model": "claude-opus-4-7"})

# Method form
engine.on_no_match(llm_resolver)

# Removal
engine.off_no_match(llm_resolver)
engine.clear_hooks("no_match")
engine.clear_hooks()  # all events
```

There is no `engine.hooks` registry object. Events are stable engine API,
not a string-keyed dynamic table. (A future revision can add a generic
`hooks[event_name].register()` if a real custom-event use case appears.)

## Events

Each event is documented with: signature, when it fires, what payload it
carries, and how the engine reacts to a Resolution.

### `rule_applied(step: RewriteStep, ctx)`, observer

Fires after every successful rule application. The engine's existing
`RewriteTrace` is implemented as an `on_rule_applied` hook (registered
transparently when `simplify(trace=True)` is called). External observers
(logging, progress, pretty-print) attach here.

### `should_fire(rule, expr, bindings, ctx) -> bool`, decision

Fires before applying a rule, after the DSL pattern match has succeeded and
the rule's `condition`/`when` clause has passed. Every registered hook must
return True for the firing to proceed. First False vetoes. Use cases:
budget enforcement (`return ctx.step_count < N`), A/B testing rule subsets,
debugger breakpoints.

### `no_match(expr, ctx) -> Optional[Resolution]`, resolver

Fires at a position where the engine has tried every active rule and none
matched. The engine *would* return `expr` unchanged at this point. The hook
can:

- Return `Resolution(rules=[...])`: engine drains the pending-rule queue
  (see "Mid-rewrite mutation" below), invalidates the simplifier cache,
  retries the match at the same position.
- Return `Resolution(value=...)`: engine substitutes the returned expression
  in for the current position. Normal rewriting continues from the new value
  (the outer `simplify` loop will try to apply rules to it). The resolver is
  not consulted again at the same position unless a later iteration brings
  the engine back to a no-match state there.
- Return `None`: engine continues with `expr` unchanged.

### `undefined_op(op: str, args: List, ctx) -> Optional[Resolution]`, resolver

Fires when `instantiate` sees `["!", op, ...]` and `op` is not in
`fold_funcs`. The hook can return `Resolution(fold_funcs={op: handler})`
(installed and retried), `Resolution(value=...)` (used as the result), or
`None` (engine emits `[op, *args]`, the existing fallback).

### `fold_error(op, args, exception, ctx) -> Optional[Resolution]`, resolver

Fires when an installed fold handler raises. Hook can return
`Resolution(value=...)` for a fallback or `None` to fall through to the
existing "leave as compound" behavior.

### `max_depth(expr, depth, ctx) -> Optional[Resolution]`, resolver

Fires when `equivalents`, `prove_equal`, `minimize`, or `random_walk` exhaust
their depth budget. Hook can return `Resolution(allow_more=True)` to grant
another batch (engine doubles the budget once and re-fires if still
exhausted), or `None` to stop normally.

### `cycle(expr, visited_path, ctx) -> Optional[Resolution]`, resolver

Fires when the visited-set cycle detection (from task B) catches a repeat.
By default the engine breaks out. Hook can return `Resolution(abort=True)`
to also propagate the abort to the caller, or `None` to keep the default
behavior. Useful primarily for instrumentation ("which rules cycle?") rather
than behavior change.

### `fixpoint(expr, ctx)`, observer

Fires once when `simplify` converges (no rule fires anywhere in the tree).
Useful for "engine is stuck, here's the final form" hooks. A downstream
resolver wanting "ask LLM to simplify further" should *not* re-enter
`simplify` from inside a fixpoint observer (that would loop). Instead the
hook should add rules and let the next outer call see them.

## Mid-rewrite mutation policy

When a Resolver returns `Resolution(rules=[r1, r2])`:

1. Rules are appended to a *pending* queue, not directly to `engine._rules`.
2. After the current event handler returns control to the engine, the engine
   drains the pending queue: appends to `_rules`/`_metadata`, applies any
   `metadata` from the Resolution to each rule's `RuleMetadata`, calls
   `_sort_by_priority()`, sets `_simplifier = None` (cache invalidate).
3. The engine retries the operation that triggered the resolver (e.g.
   `no_match` retries match at `expr` with the now-extended rule set).

This sequencing keeps the priority sort and cache invariants stable: rules
never appear partway through a pass. The engine controls when mutation
happens; hooks cannot reach into `_rules` directly.

For `Resolution(fold_funcs={...})`, the same sequencing applies: install
into `_fold_funcs`, then retry the `(! op ...)` evaluation.

## Error handling

**Hook raises an exception:**
- The engine catches it, wraps in
  `HookError(hook=callable, event="no_match", cause=original)`, re-raises at
  the boundary of the public method the user called.
- The engine does not silently swallow hook errors. A buggy hook fails loudly.
- A try/finally ensures any pending-rule queue from the same call is
  discarded if the hook raised.

**Resolution validation:**
- `Resolution(value=..., rules=...)` (multiple primary fields set) raises
  `ResolutionError("ambiguous: set exactly one of value/rules/fold_funcs/allow_more")`.
- `Resolution(allow_more=True)` returned from any event other than
  `max_depth` raises `ResolutionError("allow_more is only valid for max_depth")`.

**Resolver re-entry protection:**
- If a `no_match` hook returns `Resolution(rules=[...])` and after retry the
  same expression *still* has no match, the engine fires `no_match` again
  with the same `expr`. The resolver author is responsible for detecting
  re-entry (e.g., via `ctx.metadata` carrying retry count) to avoid infinite
  LLM calls.
- A safety cap on retries per top-level call (default 100) is enforced.
  Exceeding raises `ResolverLoopError` to the caller.

**Hook removal during firing:**
- A hook calling `engine.off_<event>(self)` from inside its own handler is
  allowed. Other hooks queued for the same event still run (broadcast and
  decision categories) or are skipped if a prior resolver already won.

## Backward compatibility

- `RewriteTrace` and `simplify(trace=True)` continue to work unchanged.
  Internally `simplify(trace=True)` is implemented as: register a
  `RewriteTrace` instance as an `on_rule_applied` hook, run, deregister,
  return `(result, trace)`.
- Existing `_simplify_with_trace` (collapsed in task G to a thin wrapper)
  becomes `with engine._temporary_hook("rule_applied", trace_obj): ...`. No
  parallel rule loops anywhere.
- `condition`/`when` clauses keep working unchanged. `should_fire` decisions
  layer on top.
- Existing `RewriteStep`, `RewriteTrace`, `TraceListener` types in
  `rerum/trace.py` are reused as-is; the hook system is built on top of
  them, not in place of them.

## Testing strategy

The hook system has a clean test surface:

- **Per-event firing tests**: build small engines, register a recording
  observer, run a known input, assert the event log matches expectations
  (e.g., `no_match` fires exactly at positions with no match).
- **Composition tests**:
  - Observer broadcast: register two `on_rule_applied` hooks; assert both
    receive every step.
  - Resolver chain: register two `on_no_match` resolvers; assert the second
    only fires when the first returns `None`.
  - Decision AND-gate: register two `on_should_fire` hooks; one returns
    False; assert the firing is vetoed.
- **Resolution path tests**: register a resolver returning
  `Resolution(rules=[r])`; assert `engine._rules` grows after the call,
  with `metadata.provenance` set as the resolver specified; assert subsequent
  calls see the new rule.
- **Error path tests**: hook raises; assert `HookError` propagates with
  `event` and `cause` populated.
- **Cycle/loop tests**: resolver always returns the same rule; assert
  `ResolverLoopError` after the retry cap.
- **MCP integration tests**: mock the LLM call inside resolvers (e.g.,
  `llm_call = lambda expr: parse_rule_line("@stub: ...")`) so the LLM-rule-
  inference flow is exercised without a real network call.

## Implementation notes

The hook system lives primarily in `rerum/engine.py` (registration methods
on `RuleEngine`, the dispatch logic) and a new module `rerum/hooks.py`
(the `Resolution` dataclass, `HookContext`, error types, the per-category
composition helpers `_run_observers`, `_run_resolvers`, `_run_decisions`).

The dispatch logic is intentionally thin. Each event in the engine becomes
a single line that calls into `hooks.py`:

```python
# Inside _all_single_rewrites or wherever no_match is detected:
resolution = self._hooks.run_resolvers("no_match", expr, ctx)
if resolution is not None:
    # Process resolution per the rules above
```

The composition logic is centralized in `_HookRegistry`, which knows the
category of each event and runs the appropriate policy. The engine's role
is reduced to "fire the event at the right place; honor the Resolution
that comes back."

## Open questions for follow-up specs

- **DSL metadata**: How should `provenance`, `confidence`, `category`,
  `references` etc. surface in the rule DSL vs the JSON serialization? This
  is the next spec.
- **MCP surface**: The MCP server will expose hook registration via tools
  (e.g., `register_llm_resolver(model_name)`). Spec to follow once the
  hook system is implemented and the DSL metadata is settled.
