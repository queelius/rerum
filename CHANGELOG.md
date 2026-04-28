# Changelog

All notable changes to RERUM are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning is
[SemVer](https://semver.org/) with the caveat that while `0.x`, minor bumps
may include breaking changes.

## [Unreleased]

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
