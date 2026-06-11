# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RERUM (Rewriting Expressions via Rules Using Morphisms) is a pattern matching and term rewriting library for symbolic computation in Python. It exposes a DSL for defining rewrite rules and applies them to s-expression-style nested lists. Beyond reduction-to-normal-form, it supports bidirectional rules (`<=>`) and equivalence-class reasoning: `equivalents`, `prove_equal`, and cost-directed `minimize`.

For end-user features, DSL syntax, CLI flags, and worked examples, see `README.md` and `docs/`. This file is for working *on* the codebase.

## Build and Test Commands

```bash
pip install -e ".[dev]"                                # pytest, pytest-cov
pip install -e ".[docs]"                               # mkdocs, mkdocs-material
pytest                                                 # all tests (config in pyproject.toml)
pytest --cov=rerum --cov-report=term-missing           # with branch coverage
pytest rerum/tests/test_guards.py -v                   # one file
pytest rerum/tests/test_guards.py::TestGuardParsing::test_parse_when_clause -v  # one test
mkdocs serve                                            # local docs at 127.0.0.1:8000
mkdocs build                                            # static build to site/
python experiments/features_benchmark.py               # empirical perf probes (NOT pytest)
python experiments/scaling.py                          # validate enumeration class sizes
```

Python 3.9+. CI tests on 3.9 through 3.13.

## Architecture

The package is split into a *pure functional core* (`rewriter.py`) and a *high-level OO API* (`engine.py`). The line-count ratio (~960 vs ~2,650) is a useful map of where complexity lives. Pattern matching is small and pure; rule loading, group management, equivalence reasoning, and tracing are where surface area grew.

### `rewriter.py`, the pure functional core
- `match(pattern, expr, bindings) -> Bindings | NoMatch`: structural pattern match.
- `instantiate(skeleton, bindings, fold_funcs)`: skeleton expansion with `(! op ...)` evaluation.
- `rewriter(rules, fold_funcs) -> callable`: factory producing a fixpoint simplifier.
- `Bindings` (dict wrapper) and `NoMatch` (falsy singleton). The codebase migrated away from a stringly-typed `"failed"` sentinel in 0.5.0; use `wrap_bindings()` and truthiness checks rather than string comparisons.
- Prelude constants: `ARITHMETIC_PRELUDE`, `MATH_PRELUDE`, `PREDICATE_PRELUDE`, `FULL_PRELUDE`, `MINIMAL_PRELUDE`, `NO_PRELUDE`.
- Fold-handler builders: `nary_fold`, `unary_only`, `binary_only`, `special_minus`, `safe_div`.

### `engine.py`, the DSL parser and high-level API
- `RuleEngine`: load DSL/JSON/files, manage groups, apply with strategies, trace, sequence.
- DSL parsing: `parse_rule_line`, `load_rules_from_dsl`, `load_rules_from_file`, `load_rules_from_json`.
- S-expression I/O: `parse_sexpr`, `format_sexpr`. Expression builder singleton: `E`.
- Equivalence-class reasoning: `equivalents` (lazy generator), `enumerate_equivalents`, `prove_equal` (bidirectional BFS), `are_equal`, `EqualityProof`.
- Cost minimization: `minimize`, `OptimizationResult`, `expr_size`, `expr_depth`, `expr_ops`, `expr_atoms`, `make_op_cost_fn`, `COST_METRICS`.
- Stochastic: `random_equivalent`, `sample_equivalents`, `random_walk`.
- Tracing: `RewriteTrace`, `RewriteStep`. Engine sequencing: `SequencedEngine` via `>>`.
- Metadata layer (v0.7): ``RuleMetadata`` carries ``category`` (free-form
  string, populated via DSL ``{category=X}`` annotation or JSON),
  ``reasoning`` (JSON only), ``examples`` (validated at load time, may
  carry an optional ``direction`` field for bidirectional rules), and
  ``fwd_label``/``rev_label`` (JSON only). ``ExampleValidationError``
  raised on validation failure. ``engine.load_metadata_json()`` merges
  a sidecar JSON file onto already-loaded rules.

### `cli.py`, the command-line interface
- `RerumREPL` (interactive), `ScriptRunner` (`.rerum` files), one-shot via `-e`, pipe via stdin with `-q`.
- Custom prelude: any `.py` file exporting a `PRELUDE` dict, loaded with `-p path.py`.

### `hooks.py`, the engine extension points

- ``Resolution`` (frozen dataclass), ``HookContext`` (engine state view),
  exception types (``HooksError`` base; ``HookError``, ``ResolutionError``,
  ``ResolverLoopError`` subclasses), and ``_HookRegistry`` (per-category
  composition).
- Eight named events: ``rule_applied`` and ``fixpoint`` (observers,
  broadcast); ``no_match``, ``undefined_op``, ``fold_error``, ``max_depth``,
  ``cycle`` (resolvers, chain); ``should_fire`` (decision, AND-gate).
- LLM rule inference: register an ``on_no_match`` resolver that returns
  ``Resolution(rules=[...])``; the engine installs the rules with
  provenance metadata and retries. Default retry cap of 100 catches
  resolver loops.
- ``simplify(trace=True)`` registers a temporary ``on_rule_applied``
  hook; the trace is fully integrated with the hook system.

### `rerum/mcp/`, the agent-facing MCP server (v0.9)

The MCP layer is thin orchestration over the general engine: tools marshal
JSON in and out, the engine does the rewriting. No tool holds domain logic;
rules, theories, and caller goals all arrive as DATA. ``test_mcp_no_domain.py``
locks this in.

- ``registry.py``: the SINGLE SOURCE OF TRUTH. Discovery (every ``tool_*``
  callable), dependency injection (positional params = injected deps:
  engine/store/sampler), the typed JSON input schemas (keyword-only
  params' annotations; ``Literal`` -> enum; ``Args:`` docstring section ->
  per-param descriptions), and dispatch validation/coercion all DERIVE
  from the handler signatures. Adding a tool = writing one annotated,
  docstringed ``tool_*`` function; never edit a parallel table.
- ``tools.py``: the 18 handlers (authoring, persistence, applying, goal
  solving, agentic loop, admin). ``prose`` is a TOP-LEVEL response field;
  ``converged`` is truthful (fixpoint event); inputs are strictly
  validated at the boundary.
- ``server.py``: ``RerumMCPServer`` per-session engine + ``RuleStore`` +
  sampler; registry-driven ``call_tool`` with a lock-guarded
  ``engine_busy`` flag (dispatch runs in a worker thread).
- ``__init__.py``: importable WITHOUT the ``mcp`` SDK; ``run_server()``
  (stdio) and ``_build_sdk_server()`` (used by the in-memory wire tests)
  require it. ``list_tools`` advertises the registry schemas (the SDK then
  VALIDATES calls against them). The capability-gated SAMPLING BRIDGE
  wires ``solve_assisted`` to the client's LLM via
  ``sampling/createMessage``; without the capability the tool refuses
  with ``sampling_unsupported``.
- ``trace.py``: situated ``step_to_dict``, ``assemble_trace`` (global
  roots + truncation; no prose inside), ``render_prose(trace)``,
  ``trace_recorder`` (rule_applied + fixpoint observers).
- ``persistence.py``: ``RuleStore`` (``.rerum/rules/<name>.json`` and
  ``<name>.theory.json``); git-friendly, rejects path traversal. A loaded
  theory is CONSUMED by ``solve_goal``'s ``normalize_between``.
- ``solver.py``: LLM-resolver factory used by ``solve_assisted``.
- ``errors.py``: ``MCPToolError`` (stable codes) + ``map_exception`` (the
  single mapping point; unwraps ``HookError`` causes, wires tool context).
- ``utils.py``: ``json_safe``, the transport sanitizer (Fraction +
  non-finite floats).
- Naming: engine ``solve()`` (search) vs ``solve_goal`` (its MCP wrapper)
  vs ``solve_assisted`` (LLM-resolver loop) are three distinct things.

### Key design boundaries
- **Rules are data; preludes are code.** Rules can only invoke operations the developer has explicitly enabled in the prelude. That is the security boundary for loading rules from untrusted sources, so preserve it when adding features that touch rule evaluation.
- **Pure core, mutable engine.** `rewriter.py` does not allocate engine state. Rule storage, group enable/disable, and trace accumulation all live in `engine.py`.
- **Bidirectional desugaring is eager.** A `<=>` rule produces two `RuleMetadata` entries (`-fwd` and `-rev`) at parse time. Tests asserting rule counts must account for the doubling; `engine.list_rules()` shows post-desugar names.

## Tests and experiments

- `rerum/tests/` is one file per feature area. Core: `test_bidirectional`, `test_bindings`, `test_cli`, `test_engine_methods`, `test_equivalents`, `test_expr_builder`, `test_groups`, `test_guards`, `test_includes`, `test_normalize`, `test_numeval`, `test_optimization`, `test_priorities`, `test_prove_equal`, `test_rationals`, `test_rewriter`, `test_sequencing`, `test_solve`, `test_strategies`, `test_trace`, `test_trace_situated`, `test_training`, plus the `test_mcp_*` family (tools, registry, wire, smoke, trace, errors, persistence, solve_goal, solve_assisted, no_domain). Example-content suites drive the GENERAL engine through `examples/` data: `test_differentiation`, `test_integration`, `test_limits`, `test_calculus_checker_d2`, `test_boolean`, `test_sets`, `test_peano`, `test_ski`. When adding a feature, add a parallel file rather than extending an unrelated one.
- `experiments/` is for *empirical probes*: timings, scaling validation against the theoretical `n! × Catalan(n-1)` class size for associative-commutative sums. These are runnable scripts, not pytest, and they catch regressions in the equivalence/proof/minimize pipeline that unit tests miss. Run them after changes to `equivalents`, `prove_equal`, or `minimize`.
- `examples/` is the DOMAIN CONTENT library, each domain pure data + an optional checker: calculus (`differentiation.rules`, `integration.rules`, `limits.rules` + `.metadata.json` example sidecars, `calculus_checker.py` numeric verification on the general `numeval`, `limits_fold_ops.py` example fold ops, `arithmetic.theory.json`), `boolean.rules`/`sets.rules` (+ theories; truth-table / Venn property tests certify every rule), `peano.rules` and `ski.rules` (NO prelude: computation from pure rewriting; SKI demonstrates honest non-termination budgets), legacy `algebra.rules`/`calculus.rules`/`number_theory.rules`, a custom prelude, `demo.py`, `demo.rerum`. Every domain's loading contract (prelude bundles, theory, driver+goal) currently lives in file comments and per-test loader helpers.

## Footguns (non-obvious behavior)

- **`improvement_ratio` semantics flipped in 0.5.0.** It now reports the *fractional reduction* (`1 - cost/original_cost`): `0.0` is no improvement, `1.0` is fully eliminated. The retention ratio is `cost_ratio`. Pre-0.5 callers that read `improvement_ratio` as "fraction kept" must switch to `cost_ratio`.
- **`minimize()` default flipped in 0.5.0.** `include_unidirectional` now defaults to `True`. The other equivalence methods (`equivalents`, `enumerate_equivalents`, `prove_equal`, `are_equal`, `random_equivalent`, `sample_equivalents`, `random_walk`) still default to `False`. That asymmetry is intentional, since minimize is for users with mixed `=>` and `<=>` rules while the rest are for reasoning over strict equivalence classes, but it is an easy place to write a bug.
- **`prove_equal(..., max_expressions=N)` is the work budget.** Without it, un-provable queries exhaust the depth-bounded reachable set on both BFS frontiers and can run for tens of seconds at modest depths on rich bidirectional rule sets.
- **Equivalence-class size grows as `n! × Catalan(n-1)`** under associative-commutative `+`. Enumeration is fine to `n=5` (1,680 forms); `n=6` is roughly 30k and impractical. Past `n=5` use `prove_equal` with a budget over `enumerate_equivalents`.
- **Strategy default is `"exhaustive"`** (apply rules to fixpoint). For one-shot rule application that also returns the matched rule's metadata, use `engine.apply_once(expr)` rather than `engine(expr, strategy="once")`.
- **Hook fast-path bypass**: when any engine-fired hook is registered,
  ``simplify`` skips the cached ``rewriter()`` fast path and uses
  ``_simplify_exhaustive``. Hooks need engine context that the
  pure-function rewriter does not have. This is correct behavior, not
  a bug, but explains why a heavily-hooked engine is slower than an
  unhooked one on the same rule set.
- **Resolver retry cap**: a resolver returning rules that don't match
  triggers ``ResolverLoopError`` after 100 retries per top-level call.
  Named rules deduplicate (so the same named rule installed twice
  doesn't increment the counter). Anonymous rules without progress
  trigger the cap. Default cap is not currently configurable; future
  work may expose it.
- **Examples validation needs the prelude**: rules whose examples use
  ``(! op ...)`` compute forms require ``fold_funcs`` to be set before
  the example is validated. Load with ``validate_examples=False`` if
  loading rules before configuring the prelude, then call
  ``engine.validate_examples()`` after the prelude is set. Default DSL
  loaders validate eagerly, so an unconfigured prelude with examples
  using fold ops will raise ``ExampleValidationError`` at load.
- **Cancellation propagation**: ``ctx.cancel()`` from within a hook
  sets ``self._cancel_requested``. Strategy drivers
  (``_simplify_exhaustive``, ``_simplify_bottomup``, ``_simplify_topdown``)
  check this flag at strategic points; ``equivalents`` and ``prove_equal``
  also honor it. ``apply_once`` is one-shot so it doesn't need to check.
  Cancellation from a ``fixpoint`` observer is silently ignored because
  the engine has already converged.
- **MCP solve naming**: ``solve_goal`` is goal-directed search (engine
  ``solve()``); ``solve_assisted`` is the LLM-resolver agentic loop (the
  old ``solve`` tool). They do not share an implementation, and no tool is
  named bare ``solve``.
- **MCP is domain-agnostic by test**: ``test_mcp_no_domain.py`` fails if any
  ``rerum/mcp/`` file hardcodes a domain operator literal as code (it strips
  docstrings/comments first, so caller-data examples are fine). The server
  loads rules and theories as data; keep it that way.
- **MCP responses must be JSON-safe**: every tool response is ``json.dumps``-ed
  by the transport (with ``allow_nan=False``), but engine values are not all
  JSON-native -- a ``fractions.Fraction`` in particular has broken
  serialization repeatedly, and non-finite floats emit non-spec JSON. The
  MCP layer renders expressions to s-expr strings via ``format_sexpr`` and
  sanitizes structured values through ``rerum.mcp.utils.json_safe`` before
  returning. New tools must route any expression or numeric-bearing field
  through the same path.
- **The MCP tool surface IS the tool_* signatures**: the registry derives
  discovery, dependency injection, the typed input schema, and validation
  from each handler's signature and ``Args:`` docstring. Changing a
  keyword-only parameter (name, annotation, default) changes the
  client-facing schema -- and the SDK VALIDATES calls against it, so a
  stale client breaks loudly. Positional (pre-``*``) params are injected
  dependencies and never appear in the schema; the provider map in
  ``server.py`` must know their names (engine/store/sampler).
- **solve_assisted requires a sampling channel**: without one (client
  lacks the MCP sampling capability, or no ``set_sampler``) it raises
  ``sampling_unsupported`` rather than silently degrading to plain
  ``simplify``. The stdio bridge is capability-gated per call in
  ``_build_sdk_server``.

## Expression representation

Expressions are nested Python lists in prefix notation: `["+", "x", ["*", 2, "y"]]` represents `x + 2*y`. Atoms are strings (variables) or numbers (constants). The DSL pattern syntax uses `?x` (match-any-bind), `?x:const` (number only), `?x:var` (symbol only), `?x:free(v)` (does not contain `v`), and `?x...` (rest pattern). Skeleton uses `:x` (substitute), `:x...` (splice list), and `(! op args...)` (compute via prelude). See `README.md` and `docs/dsl-reference.md` for the full reference.
