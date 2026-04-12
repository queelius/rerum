# Changelog

All notable changes to RERUM are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning is
[SemVer](https://semver.org/) with the caveat that while `0.x`, minor bumps
may include breaking changes.

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
