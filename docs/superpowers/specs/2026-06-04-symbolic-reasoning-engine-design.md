# RERUM as a Traceable Symbolic-Reasoning Engine

**Date:** 2026-06-04
**Status:** Design approved, pending spec review
**Scope:** One unified vision spec; implementation is phased (plans per phase).

## 1. Motivation

RERUM is a term-rewriting engine over s-expression nested lists. Today it can
differentiate basic expressions and simplify algebra, but three limitations
block it from being a vehicle for high-quality symbolic-reasoning training data:

1. **Traces lose the reasoning.** A `RewriteStep` keeps only
   `(rule_index, metadata, before, after)`, where `before`/`after` are the
   *local subtree*, not the whole expression. The matching `bindings` (the
   substitution that justifies a step) and the redex `path` are computed and
   discarded. `prove_equal`/`equivalents`/`minimize` return bare expression
   chains with no rule labels, so the sequence of rules that proves an equality
   is unrecoverable. None of this is usable as "show your work" data.

2. **Simplification is weak.** Matching is strictly positional (verified:
   `(+ a ?y)` does not match `(+ b a)`); there is no normal form. Derivative
   output is verbose cruft that cannot be cleaned up without combinatorial
   `minimize`.

3. **Only differentiation exists, and partially.** No integration, no limits.
   Two correctness bugs and a corruption footgun sit in the calculus-relevant
   paths.

The goal is a single coherent system where differentiation, integration, and
limits are all solved by one goal-directed search over the rewrite graph, every
step is self-justifying and situated, and the solution path renders directly to
both machine-checkable and natural-language training records.

## 2. Goals and Non-Goals

**Goals**

- A trace model where each step is self-contained (rule identity, direction,
  bindings, redex path, guard result, rationale) and the whole-expression
  derivation is reconstructible.
- A canonical normal form that makes commuted/associated forms converge, with
  the normalization steps themselves visible in the trace.
- A unified goal-directed search that reduces `dd`, `int`, and `lim` operators.
- Complete single- and multi-variable differentiation; table+search-driven
  integration; substitution/L'Hopital/known-limit evaluation of limits.
- Exact rational arithmetic.
- A training-data emitter producing verifiable JSONL and natural-language
  chain-of-thought, both projected from one structured trace, with a numeric
  verifier as a quality filter.

**Non-Goals (this spec)**

- Symbolic linear algebra, ODEs, series expansions beyond what limits need.
- A general CAS-grade simplifier (Groebner bases, factorization over fields).
- Performance tuning beyond budgeted termination; correctness and trace quality
  come first.
- A UI. Output is files (JSONL, prose) and the Python API.

## 3. Architecture Overview

Four layers over a unified search spine:

```
Layer 4  Training-data emitter   (structured JSONL + NL prose + numeric verify)
Layer 3  Calculus rule sets      (differentiation / integration / limits)
Layer 2  Engine extensions       (normal form, search, fresh vars, rationals,
                                   free-of?, prelude, bug fixes)
Layer 1  Trace foundation        (situated self-contained steps; labeled paths)
                                  |
                          single-step rewrite
                       (rule + direction + bindings + path)
```

The single-step rewrite is the first-class unit. `_all_single_rewrites` already
generates whole-expression neighbors carrying `rule_idx`, `bindings`, and the
redex position, then throws the labels away. Keeping those labels is what powers
both rich tracing and search-as-solving.

## 4. Layer 1: Trace Foundation

### 4.1 Step model

`RewriteStep` gains fields (additive; existing `before`/`after` retained for
back-compat and continue to mean the local redex):

| Field | Meaning |
|-------|---------|
| `rule_id` | Stable identity: the rule `name`, else a content hash of `(pattern, skeleton)`. Robust to the post-desugar index churn that makes `rule_index` brittle. |
| `direction` | `"fwd"`/`"rev"`/`None`, surfaced for bidirectional rules. |
| `bindings` | The match substitution, serialized via `Bindings.to_dict()`. |
| `path` | List of child indices locating the redex within the root expression. |
| `before_redex`, `after_redex` | The local subtree before/after (aliases of legacy `before`/`after`). |
| `kind` | `"rule"` (a named rewrite), `"normalize"` (flatten/sort/collect), or `"fold"` (constant folding). |
| `guard` | When the rule had a condition: the instantiated condition and its boolean result. |
| `rationale` | `RuleMetadata.reasoning`/`category`, carried for the NL renderer. |

### 4.2 Path threading and global reconstruction

The strategy drivers (`_simplify_exhaustive`, `_bottomup_pass`,
`_topdown_pass`) already recurse child-by-child. Thread an accumulating `path`
(parent path + child index) into the recursion and stamp it on each emitted
step. `expr_path` already exists on `HookContext` but is always passed empty;
populate it from the same source.

The whole-expression sequence is reconstructed, not stored per step:
`RewriteTrace.to_global_sequence()` starts from `initial` and, for each step,
splices `after_redex` at `path` to yield `before_root`/`after_root`. This is
lossless and avoids threading the root through every recursion frame.

### 4.3 Labeled search paths

`_all_single_rewrites` returns labeled edges (expression plus the
`rule_id`/`direction`/`bindings`/`path` that produced it). The search methods
record the edge label on the parent pointer:

- `prove_equal`: `visited_a`/`visited_b` store the producing edge, so
  `reconstruct_path` returns labeled steps. `EqualityProof.path_a`/`path_b`
  become lists of `RewriteStep`, and `to_dict()` emits the rule sequence.
- `equivalents`/`enumerate_equivalents`/`minimize`/`random_walk`: each can
  report how a member was reached.
- `minimize`: `OptimizationResult` gains the derivation path from original to
  minimum.

### 4.4 Serialization

`RewriteStep.to_dict()` emits the full situated record (including
`rule_id`, `direction`, `bindings`, `path`, `kind`, `guard`, `rationale`).
`RewriteTrace.to_dict()` gains a `global_sequence` option. A step is then
independently checkable by an external verifier without holding the rule set,
provided `rule_id` resolves (the emitter can optionally inline the rule's
pattern/skeleton).

## 5. Layer 2: Engine Extensions

### 5.1 Canonical normal form (`normalize.py`)

A separate, traceable normalization pass:

- **Flatten** nested associative operators: `(+ (+ a b) c)` becomes
  `(+ a b c)`; likewise `*`. The representation becomes n-ary for `+`/`*`.
  Linearity rules use the existing `?rest...` rest-pattern and `:rest...`
  splice (e.g. `(dd (+ ?f ?rest...) ?v) => (+ (dd :f :v) (dd (+ :rest...) :v))`).
- **Sort** commutative operands by a total order (constants first, then
  variables lexicographically, then compounds by a structural key). Deterministic.
- **Collect like terms**: in a flattened sorted `+`, combine `x + x` into
  `(* 2 x)` and `c1*x + c2*x` into `((c1+c2)*x)`; in `*`, combine `x * x` into
  `(^ x 2)` and `x^a * x^b` into `x^(a+b)`.
- **Constant fold** remaining literal subexpressions.

Each sub-step emits a `kind="normalize"` trace step so simplification is
explained rather than opaque. Normalization is idempotent and confluent by
construction (sort + collect yields a unique representative).

### 5.2 Goal-directed search (`solve.py`)

Generalize the existing bidirectional BFS into a best-first search:

```
solve(expr, goal_predicate, *, cost_fn=expr_size, max_nodes=10000,
      fresh_vars=True) -> SolveResult(solution, derivation) | None
```

- `goal_predicate(expr) -> bool`: e.g. `lambda e: not contains_op(e, {"int", "lim"})`.
- Best-first by `cost_fn` (reuses `optimize.py` metrics), budgeted by
  `max_nodes` expanded (mirrors `prove_equal(max_expressions=...)`, with the
  same default-budget caution); emits `max_depth` on exhaustion and returns
  `None` rather than a partial result.
- Returns the labeled derivation (a `RewriteTrace`), including, optionally, the
  branches explored but not on the solution path (valuable as "search" training
  data).

Differentiation typically reduces deterministically (the `dd` rules are
near-confluent once normalized); integration and limits use the search with
backtracking.

### 5.3 Fresh variables

A deterministic skeleton form `(fresh u)` resolves to the smallest gensym of the
form `u`, `u1`, `u2`, ... not free in the current expression. Being a pure
function of the expression keeps the rewrite deterministic and therefore
traceable. Used by u-substitution and integration-by-parts rules.

### 5.4 Exact rationals

Introduce `fractions.Fraction` as a first-class numeric value:

- `safe_div`/`nary_fold` return `Fraction` when the exact result is non-integral
  over integer inputs; `Fraction` collapses to `int` when whole.
- `parse_sexpr`/`format_sexpr` render a rational as `(/ p q)` (still valid
  s-expr), and recognize an integer-ratio compound as a literal where useful.
- The float-renarrowing in `instantiate`/`try_constant_fold` is extended to keep
  `Fraction` exact rather than coercing to float.

### 5.5 Predicates, prelude, and bug fixes

- Add a `free-of?` guard predicate: `(! free-of? :f v)` is true when symbol `v`
  does not occur in `:f`. This replaces the fragile pattern tag for guard use.
- Fix the `?x:free(v)` binding-order bug: evaluate free-ness against the fully
  resolved bindings (so `v` bound to the right of the `?free` pattern is seen),
  or deprecate the tag in favor of the guard predicate, documenting the change.
- Fix the guard-on-undefined-op footgun: a guard that references an op absent
  from the active prelude raises (guards must be decidable) rather than
  evaluating to a truthy unfolded compound. This closes the silent
  list-concatenation corruption path.
- Ship a combined `CALCULUS_PRELUDE` = math functions + predicates + `free-of?`
  + `fresh`, so calculus rule files load with one correct prelude.

### 5.6 The two pre-existing bugs

The `?x:free(v)` binding-order bug and the guard-on-undefined-op corruption are
fixed in Phase 0 (they are independent of the larger work and unblock safe rule
authoring).

## 6. Layer 3: Calculus Rule Sets

Each rule set ships with curated, load-validated `examples` metadata (using the
v0.7 metadata layer), so rules are self-documenting and machine-checked.

### 6.1 Differentiation (`differentiation.rules`)

Extends the current `dd` set to completeness:

- Inverse trig: `asin`, `acos`, `atan`.
- Hyperbolic: `sinh`, `cosh`, `tanh`.
- `sec`, `csc`, `cot` (and consume the `sec` that `tan`'s derivative emits).
- `sqrt` as a first-class form.
- General power `f^g` (exponent contains the variable) via logarithmic
  differentiation: `d/dx f^g = f^g * (g' * ln f + g * f'/f)`.
- `a^x` (constant base), `log_b`.
- Partial derivatives w.r.t. multiple variables via the `free-of?` guard
  (a subexpression free of the differentiation variable differentiates to 0).

### 6.2 Integration (`integration.rules`)

`(int f x)` operator, reduced by the Layer 2 search:

- Linearity (sum, constant multiple), power rule (`int x^n = x^(n+1)/(n+1)`,
  n != -1), `int 1/x = ln|x|`, `int e^x`, `int sin/cos`, and a table of standard
  forms.
- u-substitution and integration-by-parts as rules that introduce `(fresh u)`
  and rely on the search to close (reduce to an int-free form) or backtrack.
- Guards gate applicability (e.g. by-parts only when the integrand is a product).

### 6.3 Limits (`limits.rules`)

`(lim f x a)` operator:

- Direct substitution when continuous at `a`.
- Indeterminate-form detection (0/0, inf/inf) gating L'Hopital, which reuses the
  differentiator: `lim f/g = lim f'/g'`.
- Known limits (e.g. `lim_{x->0} sin x / x = 1`).
- Algebraic manipulation (factor/cancel) via the search.

## 7. Layer 4: Training-Data Emitter

- `trace.to_training_record()`: structured JSONL
  `{problem, operator, steps:[{kind, rule_id, rationale, before_root,
  after_root, bindings, path, guard}], answer, verified}`. Each step is
  independently re-checkable. The per-step `before_root`/`after_root` are
  supplied by the emitter joining each redex-local step with the whole-expression
  states from `to_global_sequence()` (Section 4.2); the in-memory step itself
  stores only the redex-local edit plus its path.
- `trace.to_prose()`: chain-of-thought natural language, a projection of the
  structured trace using rule rationale/category plus a per-`kind` template
  (e.g. "Applying the product rule d/dx(f g) = f' g + f g', we get ..."). Because
  prose is derived from the structured form, the two cannot drift.
- **Numeric verifier**: differentiation checked by evaluating df/dx numerically
  at sampled points and comparing; integration by differentiating the result and
  matching the integrand; limits by numeric approach to `a`. Each derivation is
  tagged `verified`, giving a quality filter for a training corpus.
- A `corpus` generator: given a set of problems, produce verified derivation
  records (structured + prose).

## 8. Module and File Plan

New modules:

- `rerum/normalize.py`: canonical normal form (flatten, sort, collect, fold).
- `rerum/solve.py`: goal-directed best-first search and `SolveResult`.
- `rerum/training.py`: emitter (structured record, prose renderer, corpus).
- `rerum/verify.py`: numeric verification of derivations.

Extended modules:

- `rerum/trace.py`: richer `RewriteStep`, `to_global_sequence`, fuller `to_dict`.
- `rerum/engine.py`: path threading, labeled `_all_single_rewrites`, labeled
  proof/equivalence paths, `solve` entry point, prelude wiring, bug fixes.
- `rerum/rewriter.py`: `free-of?` predicate, `fresh` form, `Fraction` support in
  fold builders and renarrowing, `?x:free(v)` fix.
- `rerum/expr.py`: `Fraction` parse/format.

New rule/example files:

- `examples/differentiation.rules`, `examples/integration.rules`,
  `examples/limits.rules`, and a `examples/calculus_prelude.py`.

New tests (one file per area, matching the existing convention):

- `test_normalize.py`, `test_solve.py`, `test_trace_situated.py`,
  `test_rationals.py`, `test_free_of.py`, `test_differentiation.py`,
  `test_integration.py`, `test_limits.py`, `test_training.py`, `test_verify.py`.

## 9. Testing Strategy

- **Unit**: each new module tested in isolation (normal form idempotence and
  confluence; search termination and budget; fresh-var determinism; rational
  exactness; free-of correctness).
- **Rule-set**: every calculus rule carries `examples` validated at load; plus
  derivation tests asserting both the final answer and key trace properties
  (e.g. "the product rule fired", "the path is reconstructible").
- **Property/numeric**: differentiate-then-verify-numerically and
  integrate-then-differentiate-back as property checks over generated problems.
- **Trace integrity**: `to_global_sequence` round-trips; each serialized step
  re-checks against its rule.
- **Experiments**: `experiments/` scripts for search scaling (integration node
  counts vs problem size) and trace-corpus generation timing.

## 10. Risks and Mitigations

- **Search blowup (integration/limits).** Mitigate with strict node budgets,
  cost-guided ordering, and honest failure (return None + `max_depth`), never
  silent truncation.
- **n-ary representation ripple.** Flattening `+`/`*` changes shapes that binary
  rules assumed. Mitigate by rewriting calculus/algebra rules to rest-pattern
  form and testing normalization confluence first (Phase 2 precedes the rule
  sets).
- **Fraction coercion edge cases.** Centralize numeric coercion so int/Fraction/
  float narrowing has one definition; test exhaustively.
- **Trace size.** Situated steps plus search branches can be large; the emitter
  streams records and the in-memory trace stores redex-local data with on-demand
  global reconstruction.
- **Scope.** The spec is one vision; implementation is strictly phased so each
  phase is independently shippable and reviewable.

## 11. Implementation Phasing

Each phase becomes its own implementation plan.

0. **Foundation fixes**: `?x:free(v)` bug, guard-on-undefined-op, combined
   `CALCULUS_PRELUDE`. Small, unblocks safe rule authoring.
1. **Trace foundation**: situated `RewriteStep`, path threading, labeled search
   paths, `to_global_sequence`, fuller serialization.
2. **Canonical normal form**: `normalize.py` (flatten/sort/collect/fold) with
   confluence/idempotence tests.
3. **Search + fresh vars + rationals**: `solve.py`, `(fresh u)`, `Fraction`.
4. **Differentiation + verifier**: complete `differentiation.rules`,
   `verify.py`.
5. **Integration**: `integration.rules` over the search.
6. **Limits**: `limits.rules` (L'Hopital reuses Phase 4).
7. **Training emitter**: `training.py` (structured + prose + corpus).

## 12. Success Criteria

- A differentiation problem yields a clean, fully simplified answer and a
  reconstructible whole-expression derivation whose every step names a rule and
  carries its bindings.
- `prove_equal` returns the rule sequence, not just an expression chain.
- An integration problem requiring u-substitution is solved by the search, and
  the derivation (including the substitution) renders to both JSONL and prose.
- A limit requiring L'Hopital is solved by reusing the differentiator.
- The emitter produces a verified corpus where each record is numerically
  checked and independently re-checkable.
