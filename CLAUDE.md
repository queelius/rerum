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

### `cli.py`, the command-line interface
- `RerumREPL` (interactive), `ScriptRunner` (`.rerum` files), one-shot via `-e`, pipe via stdin with `-q`.
- Custom prelude: any `.py` file exporting a `PRELUDE` dict, loaded with `-p path.py`.

### Key design boundaries
- **Rules are data; preludes are code.** Rules can only invoke operations the developer has explicitly enabled in the prelude. That is the security boundary for loading rules from untrusted sources, so preserve it when adding features that touch rule evaluation.
- **Pure core, mutable engine.** `rewriter.py` does not allocate engine state. Rule storage, group enable/disable, and trace accumulation all live in `engine.py`.
- **Bidirectional desugaring is eager.** A `<=>` rule produces two `RuleMetadata` entries (`-fwd` and `-rev`) at parse time. Tests asserting rule counts must account for the doubling; `engine.list_rules()` shows post-desugar names.

## Tests and experiments

- `rerum/tests/` is one file per feature area: `test_bidirectional`, `test_bindings`, `test_cli`, `test_engine_methods`, `test_equivalents`, `test_expr_builder`, `test_groups`, `test_guards`, `test_includes`, `test_optimization`, `test_priorities`, `test_prove_equal`, `test_rewriter`, `test_sequencing`, `test_strategies`, `test_trace`. When adding a feature, add a parallel file rather than extending an unrelated one.
- `experiments/` is for *empirical probes*: timings, scaling validation against the theoretical `n! × Catalan(n-1)` class size for associative-commutative sums. These are runnable scripts, not pytest, and they catch regressions in the equivalence/proof/minimize pipeline that unit tests miss. Run them after changes to `equivalents`, `prove_equal`, or `minimize`.
- `examples/` contains reference rule files (`algebra.rules`, `calculus.rules`, `number_theory.rules`), a custom prelude (`custom_prelude.py`), a Python demo (`demo.py`), and a `.rerum` script (`demo.rerum`). Use these as integration smoke tests when the DSL parser changes.

## Footguns (non-obvious behavior)

- **`improvement_ratio` semantics flipped in 0.5.0.** It now reports the *fractional reduction* (`1 - cost/original_cost`): `0.0` is no improvement, `1.0` is fully eliminated. The retention ratio is `cost_ratio`. Pre-0.5 callers that read `improvement_ratio` as "fraction kept" must switch to `cost_ratio`.
- **`minimize()` default flipped in 0.5.0.** `include_unidirectional` now defaults to `True`. The other equivalence methods (`equivalents`, `enumerate_equivalents`, `prove_equal`, `are_equal`, `random_equivalent`, `sample_equivalents`, `random_walk`) still default to `False`. That asymmetry is intentional, since minimize is for users with mixed `=>` and `<=>` rules while the rest are for reasoning over strict equivalence classes, but it is an easy place to write a bug.
- **`prove_equal(..., max_expressions=N)` is the work budget.** Without it, un-provable queries exhaust the depth-bounded reachable set on both BFS frontiers and can run for tens of seconds at modest depths on rich bidirectional rule sets.
- **Equivalence-class size grows as `n! × Catalan(n-1)`** under associative-commutative `+`. Enumeration is fine to `n=5` (1,680 forms); `n=6` is roughly 30k and impractical. Past `n=5` use `prove_equal` with a budget over `enumerate_equivalents`.
- **Strategy default is `"exhaustive"`** (apply rules to fixpoint). For one-shot rule application that also returns the matched rule's metadata, use `engine.apply_once(expr)` rather than `engine(expr, strategy="once")`.

## Expression representation

Expressions are nested Python lists in prefix notation: `["+", "x", ["*", 2, "y"]]` represents `x + 2*y`. Atoms are strings (variables) or numbers (constants). The DSL pattern syntax uses `?x` (match-any-bind), `?x:const` (number only), `?x:var` (symbol only), `?x:free(v)` (does not contain `v`), and `?x...` (rest pattern). Skeleton uses `:x` (substitute), `:x...` (splice list), and `(! op args...)` (compute via prelude). See `README.md` and `docs/dsl-reference.md` for the full reference.
