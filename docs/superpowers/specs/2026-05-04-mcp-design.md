# Rerum MCP Server: Design Spec

**Date:** 2026-05-04
**Status:** Draft
**Scope:** v0.8 MCP server. Exposes the rerum engine to LLM agents via MCP. Builds on v0.6 (hooks) for verifiable traces and v0.7 (metadata) for LLM-citable labels and reasoning. The eventual evaluation harness against a real LLM is a separate downstream effort.

## Problem

The rerum engine reduces expressions deterministically against a rule library. The hook system (v0.6) emits a structured trace of every rule application; the metadata layer (v0.7) attaches a category, reasoning, and examples to each rule. Together they give an LLM the substrate to:

1. Use the engine as a tool to solve rewriting problems.
2. Author or extend the rule library.
3. Explain a solution grounded in the engine's actual execution, citing rule names, categories, and reasoning rather than hallucinating reasoning.

What's missing is the connecting tissue: an MCP server that exposes the engine's capabilities as LLM-callable tools, returns structured traces the LLM can cite from, and supports an agentic loop where the LLM can supply rules mid-solve when the engine hits a dead end.

This spec designs that server.

## Goals

1. **Solver-first workflow.** The LLM uses the engine to simplify, prove, minimize, or enumerate equivalents, with traces it can narrate from.
2. **Server-side LLM resolver.** When the engine reaches an expression with no matching rule, a resolver hook calls back to the connected LLM via MCP sampling and asks for a rule. The LLM's reply is parsed, validated, and installed; the engine retries.
3. **Curated tool surface.** Twelve tools spanning authoring, solving, auditing, the agentic loop, and admin. Discoverable; not overwhelming.
4. **Persistent engine per session.** Tool calls accumulate state. The LLM builds a working set across many turns.
5. **Verifiable explanations.** Every rewriting tool returns a structured trace identical in shape across tools. The LLM cites from the trace; the trace was emitted by the engine, not by the LLM.

## Non-goals

- Multi-engine sessions. One engine per session. (A future revision could add named engines.)
- Streaming MCP responses. Responses are batched: tool returns when the rewrite finishes.
- A high-level "explain this trace" tool. Explanation is the LLM's job; the MCP gives it the data.
- Stateless tool calls (rebuild engine each call). Persistent state is more efficient and matches how an agentic LLM works.
- Real-LLM end-to-end testing in this spec; covered by a separate evaluation effort.
- A web UI or REPL frontend. The MCP is the only entry point for LLMs. Humans use the existing CLI.

## Architecture

### Layers

```
+----------------------------------+
|  MCP tool handlers (12 tools)    |   thin orchestration; no business logic
+----------------------------------+
|  trace_recorder context manager  |   wraps engine ops with on_rule_applied hook
|  solver callback                 |   wraps no_match resolver around mcp.sample
+----------------------------------+
|  rerum.RuleEngine                |   unchanged; the engine itself
+----------------------------------+
```

### Lifecycle

- One `RuleEngine` per MCP session.
- Engine created lazily on first use with default settings (no rules, no fold_funcs).
- `reset_engine` discards and recreates.
- The `solve` tool installs and removes a server-side `no_match` resolver around its call. Other tools do not modify hook registrations.

### Module layout

New code lives in `rerum/mcp/`:

- `rerum/mcp/__init__.py`: public entry, `run_server()`.
- `rerum/mcp/server.py`: server setup, session lifecycle.
- `rerum/mcp/tools.py`: the 12 tool handlers.
- `rerum/mcp/trace.py`: trace_recorder context manager and JSON serialization.
- `rerum/mcp/solver.py`: the `solve` tool's resolver-callback logic.

Tests:

- `rerum/tests/test_mcp_trace.py`: trace serialization unit tests.
- `rerum/tests/test_mcp_tools.py`: per-tool happy and error paths.
- `rerum/tests/test_mcp_solve.py`: solver flow with mocked sampling.

### Dependencies

The MCP Python SDK (`mcp` package) is added to `pyproject.toml` as an optional dependency: `pip install rerum[mcp]`. Importing `rerum.mcp` without the SDK installed raises an informative error.

### CLI

A new console entry point `rerum-mcp` starts the server. Default transport is stdio; `--transport http --port 8765` enables HTTP.

## Trace shape

The trace is the load-bearing data structure. Every rewriting tool (`simplify`, `apply_once`, `equivalents`, `prove_equal`, `minimize`, `solve`) returns a result whose `trace` field has this shape:

```json
{
  "result": "x",
  "converged": true,
  "trace": {
    "initial": "(+ x 0)",
    "final": "x",
    "steps": [
      {
        "rule_name": "add-zero",
        "category": "identity",
        "reasoning": "Zero is the additive identity element of the integers.",
        "before": "(+ x 0)",
        "after": "x",
        "rule_index": 0,
        "step_count": 1,
        "depth": 0,
        "provenance": null
      }
    ],
    "total_steps": 1,
    "summary": "Applied 1 rule (add-zero, identity)."
  },
  "stats": {
    "duration_ms": 0.4,
    "rules_in_engine": 5
  }
}
```

### Field semantics

- `initial`, `final`, `before`, `after`: s-expression *strings* via `format_sexpr`. The MCP never sends raw nested-list expressions over JSON.
- `rule_name`, `category`, `reasoning`: from `RuleMetadata`. The LLM cites these.
- `provenance`: from `RuleMetadata.extra.provenance`. For LLM-inferred rules added during `solve`, this is `"llm-inferred"`.
- `step_count`, `depth`: from `HookContext` (post-F2 these are reliable across strategies).
- `rule_index`: engine-internal index. Lets the LLM call `get_rule(rule_index=...)` to fetch full metadata.
- `summary`: pre-computed via `RewriteTrace.summary()`. One-line digest.
- `total_steps`: `len(steps)`.
- `stats`: timing and engine size for budget reasoning.

### Termination signaling

When a rewrite stops for a non-convergence reason, the response gains a `termination` field:

```json
{
  "result": "(+ a b)",
  "converged": false,
  "termination": {"reason": "cycle", "detail": "..."},
  "trace": {...}
}
```

`reason` is one of: `"cycle"`, `"max_steps"`, `"max_depth"`, `"cancelled"`, `"resolver_loop"`, `"resolver_timeout"`, `"resolver_budget_exhausted"`. The `"converged"` reason is implicit (omitted when true).

### Bidirectional steps

When a `<=>` rule fires, the step uses the storage name (`commute-fwd` or `commute-rev`) and adds a `direction_label` field with the rule's `fwd_label` or `rev_label`:

```json
{
  "rule_name": "assoc-fwd",
  "category": "associativity",
  "direction_label": "regroup-right",
  "before": "(+ (+ a b) c)",
  "after": "(+ a (+ b c))"
}
```

### Trace truncation

Long traces would blow LLM context. The MCP truncates at a configurable limit (default 200 steps); when truncated, the response carries `trace_truncated: {original_length: N}` and the steps are the *first 100 + last 100* with a `<...elided N steps...>` marker between. The LLM still sees start and convergence; the middle is summarized.

## The 12 tools

### Authoring (4)

```
load_rules(text: str, format: "dsl"|"json" = "dsl",
           validate_examples: bool = True) -> {ok, rules_added}
```
Bulk load. `format` auto-detected from leading char (`{` for json) but overridable. Validates examples by default.

```
add_rule(pattern: str, skeleton: str, name?: str, description?: str,
         category?: str, reasoning?: str, examples?: list,
         priority: int = 0, condition?: str, tags?: list,
         validate_examples: bool = True) -> {ok, rule_index}
```
One rule, structured. S-expression strings parsed via `parse_sexpr`. Returns the engine-internal index for follow-up queries.

```
list_rules(category?: str, tag?: str)
           -> [{rule_index, name, category, description, bidirectional, ...}]
```
Lightweight summary. Optional filters.

```
get_rule(rule_index?: int, name?: str) -> {full RuleMetadata + pattern + skeleton}
```
Full details. Pass either index or name.

### Solving (4), all return `{result, converged, trace, stats}`

```
simplify(expr: str, strategy: "exhaustive"|"once"|"bottomup"|"topdown" = "exhaustive",
         max_steps: int = 1000, groups?: list)
         -> {result, converged, trace, stats}
```

```
equivalents(expr: str, max_depth: int = 10, max_count: int = 100,
            strategy: "bfs"|"dfs" = "bfs", include_unidirectional: bool = false,
            groups?: list)
            -> {forms: [str], total_count, stats}
```
Trace omitted (BFS doesn't naturally produce a single trace).

```
prove_equal(expr_a: str, expr_b: str, max_depth: int = 10,
            max_expressions?: int, include_unidirectional: bool = false,
            trace: bool = True, groups?: list)
            -> {proven, common_form?, depth_a?, depth_b?, path_a?, path_b?, stats}
```
On `proven=true`, returns meeting expression and both paths (each path is a list of step dicts in the same shape as the trace section).

```
minimize(expr: str, metric: "size"|"depth"|"ops"|"atoms" = "size",
         op_costs?: dict, max_depth: int = 10, max_count: int = 10000,
         include_unidirectional: bool = true, groups?: list)
         -> {original, original_cost, best, best_cost, improvement_ratio,
             expressions_checked, stats}
```

### Auditing (1)

```
validate_examples() -> {ok, errors: [{rule_name, example, message}]}
```
Walks every rule with examples; returns errors as data, does not raise. The LLM can read the failures and decide what to fix.

### LLM-driven (1)

```
solve(expr: str, max_depth: int = 20, max_resolver_calls: int = 10,
      strategy: "exhaustive"|"bottomup"|"topdown" = "exhaustive",
      goal?: str)
      -> {result, converged, trace, inferred_rules, resolver_calls, stats}
```
The agentic flow. See "Solve flow" below.

### Admin (2)

```
reset_engine(fold_funcs?: "arithmetic"|"math"|"full"|"predicate"|"none" = "none")
             -> {ok}
```
Clears all rules, hooks, fold_funcs. Optionally configures a built-in prelude.

```
get_status() -> {rules_count, fold_funcs?, hooks: {event: count},
                 categories: [str], groups: [str], engine_version,
                 protocol_version}
```
Inspection. Lets the LLM orient itself.

## Solve flow

`solve` is the load-bearing agentic primitive. The server drives the engine; the engine pulls the LLM in only when stuck.

### Sequence

1. **Tool entry.** LLM calls `solve(expr, goal?, max_resolver_calls=10)`.
2. **Resolver setup.** Server registers a temporary `on_no_match` resolver. The closure captures: the MCP sampling channel, the `goal` string, a per-solve call counter, the cap.
3. **Engine runs.** Server calls `engine.simplify(expr, strategy=...)`. The engine drives normally.
4. **Dead end.** Engine reaches a position with no matching rule. Fires `no_match`. The resolver wakes up.
5. **LLM round trip.** Resolver builds a structured prompt (see below) and calls `mcp.sample(prompt)`.
6. **Parse response.**
   - **Parses to a valid rule:** wrap as `Resolution(rules=[...], metadata={"provenance": "llm-inferred", "via_solve": True, "round": call_count})`. Engine installs and retries.
   - **`NONE` or no parseable rule:** return `None`. Engine treats the position as terminal.
   - **Parses but examples-validation fails:** retry the LLM call once with the validation error in the prompt. If still failing, return `None`.
7. **Cap enforcement.** When `call_count` exceeds `max_resolver_calls`, the resolver returns `None` immediately on every subsequent firing without calling the LLM. Per-solve. Engine's existing T14 retry cap (100) is the outer safety net.
8. **Cleanup.** When `engine.simplify()` returns, the server deregisters the resolver and assembles the response.

### Prompt format

```
The rewrite engine is stuck. Propose ONE rewrite rule that
would help.

Goal: <goal text>
Stuck at: (foo bar)
Path so far: 3 rules applied (add-zero, mul-one, distrib).
Rules currently in engine: 47 (categories: identity,
distributivity, commutativity, associativity).

Reply with a single rule in DSL format, e.g.:
  @my-rule {category=identity}: (foo ?x) => :x

If you cannot propose a useful rule, reply: NONE
```

### Response

```json
{
  "result": "...",
  "converged": true,
  "trace": {... with provenance: "llm-inferred" on inferred-rule steps ...},
  "inferred_rules": [
    {
      "rule_index": 47,
      "name": "my-rule",
      "category": "identity",
      "dsl": "@my-rule {category=identity}: (foo ?x) => :x",
      "round": 3
    }
  ],
  "resolver_calls": 3,
  "stats": {...}
}
```

`inferred_rules` lists every rule the LLM contributed, grouped at the top level for review. After the call, those rules persist in the engine; `reset_engine` clears them.

### Failure paths

| Path | Behavior |
|---|---|
| MCP sampling unsupported by client | Tool returns error `code: "sampling_unsupported"` with suggestion to use the manual loop (`simplify` then `add_rule` then retry). |
| Sampling times out | Resolver returns `None`; response includes `termination: {reason: "resolver_timeout"}`. |
| LLM repeatedly proposes invalid rules | Cap fires; response `termination: {reason: "resolver_budget_exhausted"}`. |
| Engine's T14 ResolverLoopError | Tool returns error `code: "resolver_loop"` with the rule count and last expression. |

### Concurrency

The resolver's `mcp.sample` blocks the engine's rewrite (engine is sync). One rewrite per session can be in flight. The MCP server enforces this by serializing tool calls; concurrent calls return `code: "engine_busy"`. Documented constraint, not a bug.

## Error handling

Engine exceptions caught at the tool boundary, mapped to MCP tool-error codes:

| Code | When |
|---|---|
| `parse_error` | s-expression input fails `parse_sexpr` |
| `unknown_rule` | `get_rule(name=...)` for non-existent name |
| `validation_error` | Examples mismatch, malformed annotation, unknown annotation key |
| `sampling_unsupported` | `solve` called but MCP client doesn't support sampling |
| `resolver_loop` | `ResolverLoopError` from engine |
| `engine_busy` | concurrent rewrite attempted |
| `internal_error` | unexpected exception |

Error responses include a human-readable `message` and structured `details` (e.g., for `unknown_rule`, the requested name and a list of currently-loaded names). Tracebacks are not sent to the LLM; with `RERUM_MCP_DEBUG=1` the server includes a sanitized `_traceback` field.

The server never crashes from a malformed tool input. Engine internals stay engine-internal.

## Testing strategy

Three layers:

1. **Trace serialization unit tests** (`test_mcp_trace.py`). Register a `RewriteTrace` listener on a fresh engine, run a rewrite, deregister, assert the resulting JSON matches the schema. No MCP transport; tests just the serialization shape and field population.

2. **Tool tests via in-process MCP client** (`test_mcp_tools.py`). Use `mcp.client.session.ClientSession` against the server running in a thread (or via stdio). Each of 12 tools gets at least one happy-path test plus its main error path. Roughly 36 tests.

3. **Solve integration tests** (`test_mcp_solve.py`). Replace `mcp.sample` with a stub returning canned LLM responses. Drive a known dead-end and assert: rule installs, trace shows `provenance: llm-inferred`, cap fires correctly when responses are empty, validation-fail retry works, sampling-timeout produces the right termination reason.

Real-LLM end-to-end testing is a separate evaluation effort outside this spec.

## Versioning

This is a v0.8.0 minor bump. The MCP server is a new surface; no existing API changes.

`get_status()` returns a `protocol_version` field so future MCP schema changes can be negotiated.

CHANGELOG `[0.8.0]` `Added` section will describe the new `rerum.mcp` module, the `rerum-mcp` console entry point, the optional `[mcp]` install extra, and the 12 tools.

## Open questions for follow-up specs

- **Real-LLM evaluation harness.** Run `solve` against a small problem set (algebra, simple proofs, code rewriting) using a real LLM and measure: convergence rate, mean inferred-rules-per-solve, hallucination rate (LLM-cited rules that don't exist in the trace). Drives quality bar for v0.9.
- **Rule library curation tools.** As the LLM accumulates inferred rules, the engine grows. Tools like `consolidate_rules` (merge duplicates), `prune_rules` (remove rarely-used), `export_inferred` (save the LLM's contributions as a `.json`) belong to a follow-up cycle.
- **Multi-step examples.** `examples` test one rule application. Some rules only make sense in chains; a future field could let an example specify a multi-step expected sequence.
- **Negative examples** ("this rule should not fire on input X") for the metadata layer.
