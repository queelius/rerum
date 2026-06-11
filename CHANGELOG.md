# Changelog

All notable changes to RERUM are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning is
[SemVer](https://semver.org/) with the caveat that while `0.x`, minor bumps
may include breaking changes.

## [0.9.0]

A comprehensive review-driven redesign of the MCP layer (code review +
design review + simplification audit). Breaking changes throughout the
MCP surface; the engine API is extended, not broken.

### Breaking (MCP)
- Tool input schemas are now TYPED and derived from the handler
  signatures (the registry): real parameter names, types, Literal enums,
  defaults, required lists, and docstring-sourced descriptions. The SDK
  validates calls against them (additionalProperties: false), so a
  misspelled or mistyped argument is rejected at the protocol layer.
- Response envelopes standardized: ``prose`` is a TOP-LEVEL field on
  every derivation-bearing response (no longer embedded in the trace
  dict); ``list_rules`` returns ``{rules, count}`` instead of a bare
  list; the redundant ``trace_truncated`` field is gone (``total_steps``
  plus the in-stream ``_elided`` marker encode it).
- ``solve_assisted``: ``max_depth`` renamed ``max_steps`` (it always fed
  the simplify step budget); without a client sampling channel the tool
  now refuses with ``sampling_unsupported`` instead of silently behaving
  like plain ``simplify``; ``sampler`` is an injected dependency, not a
  caller parameter.
- ``solve_goal``: the no-op ``fresh_vars`` parameter is gone;
  ``normalize_between`` is now REAL (a theory loaded via ``load_theory``
  is threaded into the search and canonicalizes nodes).
- ``reset_engine`` delegates to the new public ``engine.reset()``.
- Unknown tool names map to ``unknown_tool`` (was ``parse_error``).
- ``rerum.mcp`` imports WITHOUT the optional ``mcp`` SDK; only
  ``run_server`` requires it.
- RATIONAL LITERALS (Scheme-style): ``parse_sexpr("1/3")`` now yields the
  exact ``Fraction(1, 3)`` atom (was: the symbol ``"1/3"``), and
  ``format_sexpr(Fraction(1, 3))`` renders ``"1/3"`` (was: the division
  EXPRESSION ``"(/ 1 3)"``, which re-parsed to a different structure --
  the round-trip was lossy). ``parse(format(x)) == x`` is now exact for
  Fraction atoms; an int-valued literal (``4/2``) narrows to the int; a
  zero denominator or non-integer parts stay plain symbols. Forced by the
  first real consumer: load-validated examples of rational-producing rules
  could not express their expected output. Matches the corpus encoder's
  existing rendering. MCP responses now carry ``"1/3"`` where they carried
  ``"(/ 1 3)"``.

### Added
- Example DOMAIN LIBRARY: calculus (differentiation D1; integration,
  limits D2 -- solve-driven, numerically certified via
  ``examples/calculus_checker.py``), boolean algebra and set algebra
  (truth-table / Venn property tests certify every rule; theory JSON
  drives the general normalize), Peano arithmetic and SKI combinators
  (NO prelude: computation from pure rewriting; SKI demonstrates honest
  non-termination budgets). All content under ``examples/``; the engine
  names no domain operator.
- ``PRELUDE_BUNDLES``: ONE registry of named prelude bundles in the core,
  consumed by both the CLI and MCP (the two tables had drifted;
  ``minimal`` was CLI-only). ``engine.fold_op_names()`` joins the public
  state API; MCP ``get_status`` lists ``available_preludes`` + installed
  ``fold_ops``.
- ``solve_goal`` goal kinds (DATA, AND-composed): ``op_free``,
  ``is_numeric`` (evaluation-style domains), ``matches`` (any caller
  pattern); plus ``op_costs`` per-operator weights steering best-first
  search.
- ``check_numeric_equiv`` MCP tool: general numeric verification over
  caller-supplied sampling ranges (domain errors skip; all-skipped is
  False, never vacuous-True; ``prelude="session"`` verifies under the
  session's fold ops).
- ``rerum/mcp/registry.py``: the single source of truth -- discovery,
  dependency injection, JSON schemas, and dispatch validation all derive
  from the ``tool_*`` signatures. Adding a tool is writing one annotated,
  docstringed function.
- MCP SAMPLING BRIDGE: when the connected client advertises the sampling
  capability, ``solve_assisted`` round-trips rule proposals through the
  client's LLM (``sampling/createMessage``); end-to-end protocol tests
  run over the SDK's in-memory streams.
- Truthful status: ``converged`` reflects the engine's fixpoint event
  (budget exhaustion is False; one-shot strategy is None);
  ``inferred_rules`` reports exactly the rules actually installed;
  ``apply_once`` surfaces ``matched``/``rule``.
- Strict inputs: empty/garbage expressions and malformed goals are clear
  ``parse_error``s (previously a ``None`` atom poisoned the engine).
- Error model: ``map_exception`` wires the tool name into details,
  unwraps ``HookError`` to the real cause, and adds ``domain_error`` /
  ``eval_error`` codes; error payload details are JSON-sanitized.
- Engine public state API: ``iter_rules()``, ``hook_counts()``,
  ``has_fold_funcs()``, ``has_theory()``, ``reset()``; ``_theory`` is an
  initialized session slot.
- ``Theory.from_json`` validates the JSON shape (clean ``ValueError``).

### Fixed
- ATOMIC rule loading: a mid-batch example-validation failure previously
  committed the earlier rules with a stale name index (they fired in
  ``simplify`` but were invisible to ``get_rule``; a retry duplicated
  them). ``_install`` now validates everything before committing
  anything.
- ``prove_equal``/``minimize`` prose: the merged forward-plus-reversed
  chain mis-narrated; ``prove_equal`` now narrates the two sides
  separately to the common form.
- Non-finite floats can no longer emit non-spec JSON (``json_safe``
  renders them as strings; the transport dumps with ``allow_nan=False``).
- ``solve``-built steps now carry ``rationale`` (metadata.reasoning or
  category), matching the engine's own emit sites -- solve-driven
  training corpora previously lost the sidecar ``reasoning`` field
  written precisely for them.
- ``RewriteTrace.inverse()`` / ``RewriteStep.inverse()``: a pure
  reverse-trace primitive (swap before/after, flip direction, keep path).
  ``minimize``'s derivation now inverts its reversed ``path_b`` steps, so
  the derivation chains correctly under ``to_global_sequence`` (the Phase 1
  limitation: it previously ended at the common form via a phantom no-op,
  never reaching best) and the MCP ``minimize`` prose narrates the real
  original->best moves.

## [0.8.0]

### Added (MCP server)
- ``rerum/mcp/`` submodule: the GENERAL agent surface over the rewriting
  engine. 18 tools across authoring, persistence, applying, goal solving,
  the agentic loop, and admin. No domain logic; rules and theories load as
  data. ``test_mcp_no_domain.py`` mechanizes this guarantee: it fails if any
  ``rerum/mcp/`` file hardcodes a domain operator literal as code.
- New console entry point ``rerum-mcp`` and optional install extra
  ``pip install rerum[mcp]``; ``run_server()`` over stdio.
- Authoring: ``load_rules``, ``add_rule``, ``list_rules``, ``get_rule``,
  ``validate_examples``.
- Persistence (file-backed, git-friendly, default ``.rerum/rules/``):
  ``save_ruleset``, ``load_ruleset``, ``list_rulesets`` (``<name>.json``)
  and ``load_theory`` (``<name>.theory.json``). Path traversal is rejected.
- Applying (return result + situated trace + ``prose``): ``simplify``,
  ``apply_once``, ``equivalents``, ``prove_equal``, ``minimize``. The trace
  is the Phase 1 situated trace (rule_id, direction, bindings, path, kind,
  guard, rationale, whole-expression before_root/after_root) and every
  response carries a natural-language ``prose`` rendering via
  ``rerum.training.to_prose``.
- Goal solving: ``solve_goal`` wraps engine ``solve()`` over a
  caller-described goal (e.g. ``{"op_free": ["int","lim"]}``); the goal is
  DATA, so the tool special-cases no operator.
- Agentic loop: ``solve_assisted`` (renamed from the earlier ``solve``)
  runs directed simplify with an ``on_no_match`` LLM resolver via MCP
  sampling; inferred rules install with provenance ``llm-inferred``.
- Admin: ``reset_engine`` (computation-bundle prelude or combination; no
  domain bundle) and ``get_status``. ``MCPToolError`` stable codes;
  ``engine_busy`` guard serializes concurrent calls.
- Every tool response is ``json.dumps``-able: expressions render to s-expr
  strings via ``format_sexpr`` and structured values route through a
  ``_json_safe`` sanitizer, so a ``fractions.Fraction`` in a result never
  breaks serialization.

## [0.7.0]

### Added (metadata layer)
- ``RuleMetadata`` gains four new fields: ``category`` (free-form string,
  for LLM paraphrasing), ``reasoning`` (free text justification),
  ``examples`` (list of ``{in, out}`` s-expression strings, validated on
  load), and ``fwd_label``/``rev_label`` (direction semantics for
  ``<=>`` rules).
- DSL annotation ``{category=X}`` between description and colon.
  Multi-line form supported. Closing-brace and quote handling robust to
  edge cases.
- JSON schema extends with the four new fields. Bidirectional rules carry
  ``fwd_label``/``rev_label`` once on the source-rule entry; the loader
  routes to the appropriate ``-fwd``/``-rev`` half. JSON roundtrip
  preserves all four fields.
- Examples validation at load time. Each engine loader (``load_dsl``,
  ``load_file``, ``load_rules_from_json``, ``add_rule``) accepts a
  ``validate_examples=True`` kwarg. ``engine.validate_examples()`` runs
  on demand (useful after a prelude change).
- ``ExampleValidationError`` raised on pattern mismatch, condition
  failure, or output mismatch. Carries ``rule_name`` and the offending
  example.
- ``engine.load_metadata_json(text)`` merges a metadata-only sidecar
  (shape: ``{rule_name: {field: value}}``) onto already-loaded rules.
  Sidecar fills missing fields only; conflicts raise ``ValueError``.
- ``add_rule`` extended to accept ``category``, ``reasoning``, ``examples``,
  ``priority``, ``condition``, ``tags``, plus the ``validate_examples``
  kwarg.

### Fixed
- **Critical**: `simplify()` no longer raises `RecursionError` on bidirectional
  rules. The fast-path `try_rules` previously called `simplify(result)`
  recursively, which reset the iteration guard on each recursion; with a
  `<=>` rule that forms a 2-cycle (fwd output matches rev pattern, rev
  output matches fwd pattern), this exhausted the call stack. The fix
  drops the recursive call (the outer fixpoint loop drives convergence)
  and adds visited-set cycle detection to all four fixpoint paths
  (`exhaustive` fast/slow, `bottomup`, `topdown`, traced).
- **Critical**: `<=>` desugaring now preserves type and free-variable
  constraints (`?x:const`, `?x:var`, `?x:free(v)`) on the auto-derived
  reverse pattern. Previously, `(f ?x:const) <=> (g :x)` produced a
  reverse rule `(g ?x) => (f :x)` that fired on non-constant arguments,
  silently producing unsound equivalence classes.
- **Critical**: `<=>` desugaring now validates the auto-derived reverse
  pattern at parse/load time. Multiple `?...` rest patterns at the same
  compound level and `(! ...)` compute forms in pattern position are
  rejected with clear errors instead of failing lazily inside the rule
  application loop.
- `to_dsl()` now emits paired `-fwd`/`-rev` rules as a single `<=>` rule;
  `to_json()`/`to_dict()` write `"bidirectional": true` and
  `load_rules_from_json()` reads it back. JSON and DSL roundtrips now
  preserve bidirectional-rule semantics; previously the flag was silently
  lost and `equivalents()` after roundtrip returned only the trivial
  result.

### Changed
- Internal bindings pipeline migrated from the stringly-typed `"failed"`
  sentinel to `Bindings | None`. Public `match()`, `extend_bindings()`,
  and `lookup()` accept both the legacy form and the new form for
  backward compatibility. The eight ``wrap_bindings`` call sites in
  ``engine.py`` collapse into direct truthiness checks.
- ``_simplify_with_trace`` is now a thin wrapper around
  ``_simplify_exhaustive`` with a ``RewriteTrace`` listener. The duplicate
  rule loop is removed (~50 lines), and the listener mechanism unblocks
  tracing on other simplification strategies.

### Added (hooks system)
- ``rerum/hooks.py``: new module with the ``Resolution`` dataclass,
  ``HookContext``, ``HooksError``/``HookError``/``ResolutionError``/
  ``ResolverLoopError`` exception types, and the internal ``_HookRegistry``.
- ``RuleEngine`` exposes eight ``on_<event>`` / ``off_<event>`` methods
  for ``rule_applied``, ``fixpoint``, ``no_match``, ``undefined_op``,
  ``fold_error``, ``max_depth``, ``cycle``, and ``should_fire``. Each
  ``on_<event>`` works as both decorator and method form.
  ``clear_hooks(event=None)`` removes one or all events.
- ``Resolution`` is the structured return type from Resolver hooks.
  Setting ``rules=...`` causes the engine to install the rules (with
  provenance metadata if provided) and retry the operation.
  ``value=...`` substitutes an expression in. ``fold_funcs={op: handler}``
  installs prelude handlers. ``allow_more=True`` extends ``max_depth``
  budgets. ``abort=True`` returns early with whatever the engine has.
- Three composition policies, locally determined by event category:
  observers broadcast, resolvers chain (first non-None wins), decisions
  AND-gate.
- Default resolver retry cap of 100 per top-level call; exceeding raises
  ``ResolverLoopError`` to the caller rather than hanging.
- ``simplify(trace=True)`` is now implemented as a temporary
  ``on_rule_applied`` hook; the legacy ``_simplify_with_trace`` duplicate
  rule loop is fully retired.
- ``RuleMetadata.extra`` dict for resolver-supplied metadata
  (``provenance``, model name, confidence). Inferred rules carry their
  origin via this field.
- Strategy parity for resolver and observer events: ``no_match``,
  ``cycle``, and ``fixpoint`` now fire under ``strategy="bottomup"`` and
  ``strategy="topdown"`` in addition to the default exhaustive strategy.
  An ``on_no_match`` resolver registered for LLM rule inference works
  uniformly across all strategies.
- ``HookContext.step_count`` is an engine-level counter (``self._step_count``)
  reset at every top-level call (``simplify`` variants, ``equivalents``,
  ``prove_equal``, ``random_walk``, ``random_equivalent``, ``apply_once``).
  It accumulates across recursive descent and across pass methods, giving
  hooks a reliable count of successful rule applications. Recursive
  internal calls accept a ``_top_level=False`` kwarg to avoid resetting
  the counter mid-call.
- ``HookContext.depth`` is threaded through ``_bottomup_pass`` and
  ``_topdown_pass`` recursive descent. Hooks at compound positions
  observe the actual tree depth (root = 0; child = 1; etc.).
- ``RuleEngine`` instances are explicitly documented as not thread-safe.
  ``_step_count``, ``_cancel_requested``, and rule storage are mutable
  per-call instance state; concurrent ``simplify()`` calls on the same
  engine race.

### Added
- ``rerum/optimize.py``: extracted ``expr_size``, ``expr_depth``,
  ``expr_ops``, ``expr_atoms``, ``make_op_cost_fn``, ``COST_METRICS``, and
  ``OptimizationResult``. Pure-function utilities now stand on their own.
- ``rerum/trace.py``: extracted ``RewriteStep`` and ``RewriteTrace``, plus
  the ``TraceListener`` callable contract. ``RewriteTrace`` instances are
  now directly callable as listeners (they append the received step).
- ``rerum/expr.py``: extracted ``parse_sexpr``, ``format_sexpr``,
  ``expr_to_tuple``, and the ``E`` builder. Eliminates lazy-import cycles
  in ``optimize.py`` and ``trace.py``.
- ``RuleEngine.rule_set()`` returns a ``RuleSet`` view object with
  chainable filters (``bidirectional_only()``, ``unidirectional_only()``,
  ``in_groups()``). Equivalence-class methods now accept
  ``rules: Optional[RuleSet] = None`` to opt into this API; the legacy
  ``include_unidirectional`` and ``groups`` kwargs continue to work.
- ``RuleEngine.equivalence_class(expr) -> EquivalenceClass`` returns a
  value object capturing the starting expression and rule set. Provides
  ``iter``, ``enumerate``, ``contains``, ``prove``, ``minimum``,
  ``sample``, ``random``, ``walk``, plus ``__contains__`` and ``__iter__``.
  The eight engine-level methods become thin shims; the cleaner API for
  new code is the value object.
- ``RuleEngine.source_rules()`` iterates over ``BidirectionalRule`` and
  ``UnidirectionalRule`` value objects, collapsing adjacent ``-fwd``/``-rev``
  storage entries back into a single logical rule. Useful for listing,
  exporting, or counting "rules I wrote" rather than "rules in storage."

## [0.5.0]

### Added
- `prove_equal(..., max_expressions=N)`: optional total-work budget across
  both bidirectional-BFS frontiers. Un-provable queries that previously
  exhausted the full depth-bounded ball now return `None` in bounded time.
- `experiments/` directory with runnable benchmarks for the equivalence,
  proof, minimize, and random-walk features. Includes a scaling study that
  validates enumeration against the theoretical `n! × Catalan(n-1)` class
  size for associative-commutative sums.
- New documentation page: Equivalence & Proof
  (`docs/equivalence.md`), linked from `mkdocs.yml`.
- README section covering bidirectional rules, `prove_equal`, `minimize`,
  and random sampling.
- Expanded `docs/api-reference.md` with equivalence, cost optimization,
  and random sampling sections.
- New examples in `docs/examples.md`: proving boolean identities,
  minimizing algebraic expressions, enumerating equivalence classes.

### Changed
- **Breaking**: `minimize()` now defaults `include_unidirectional=True`.
  The previous default silently ignored every `=>` rule, producing
  "no improvement found" on expressions with obvious unidirectional
  simplifications. If you relied on the old behavior to restrict to strict
  reversible equivalences, pass `include_unidirectional=False` explicitly.

### Fixed
- Removed dead `check_intersection()` helper inside `prove_equal()` that
  was defined but never called.
- Collapsed a dead `if/else` in `equivalents()` where both branches
  performed identical `frontier.append` operations.
- Replaced 7 stringly-typed `"failed"` sentinel checks with
  `wrap_bindings()` truthiness, consistent with `engine.match()`.

### API
- `OptimizationResult.improvement_ratio` now returns a true fractional
  improvement (`1 - cost/original_cost`): `0.0` means no improvement,
  `0.5` means cost was halved, `1.0` means eliminated. Prior versions
  returned the retention ratio `cost/original_cost`.
- `OptimizationResult.cost_ratio` (new): the retention ratio, matching
  the semantics of the old `improvement_ratio`. Callers who depended on
  the prior definition should switch to `cost_ratio`.

## [0.4.0]

### Added
- Bidirectional rules: `@name: pattern <=> skeleton` expands to forward
  (`-fwd`) and reverse (`-rev`) unidirectional rules.
- `equivalents()` and `enumerate_equivalents()`: enumerate expressions
  reachable via the engine's rules.
- `prove_equal()` and `are_equal()`: bidirectional BFS proof of equality.
- `EqualityProof` class with `common`, `depth_a`, `depth_b`, `path_a`,
  `path_b`, and `format()` for readable output.
- `minimize()`: cost-directed search for the minimum-cost equivalent
  expression.
- `OptimizationResult` class with cost reporting and improvement metrics.
- Built-in cost functions: `expr_size`, `expr_depth`, `expr_ops`,
  `expr_atoms`.
- `make_op_cost_fn()` for per-operator costs.
- `random_equivalent()`, `sample_equivalents()`, `random_walk()` for
  stochastic exploration of equivalence classes.

## [0.2.2]

### Fixed
- Python 3.9 / 3.10 type annotation compatibility.

## [0.2.1]

### Changed
- PyPI publishing workflow supports API token in addition to trusted
  publishing.

## [0.2.0]

### Added
- Initial release with CI, PyPI publishing, and mkdocs documentation.

## [0.1.0]

### Added
- Initial public release.

[0.5.0]: https://github.com/queelius/rerum/releases/tag/v0.5.0
[0.4.0]: https://github.com/queelius/rerum/releases/tag/v0.4.0
[0.2.2]: https://github.com/queelius/rerum/releases/tag/v0.2.2
[0.2.1]: https://github.com/queelius/rerum/releases/tag/v0.2.1
[0.2.0]: https://github.com/queelius/rerum/releases/tag/v0.2.0
[0.1.0]: https://github.com/queelius/rerum/releases/tag/v0.1.0
