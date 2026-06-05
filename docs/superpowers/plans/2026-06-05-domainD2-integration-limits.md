# Domain D2: Integration and Limits Example Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Demonstrate that integration and limit evaluation are *just rule sets plus data*, with NO new engine code. Ship `examples/integration.rules` (the `int` operator: linearity via rest-patterns, the power rule with exact rationals, the standard table, u-substitution and integration-by-parts using `(fresh u)`) and `examples/limits.rules` (the `lim` operator: direct substitution under a continuity guard, indeterminate-form detection, L'Hopital reusing the `dd` rules, known limits, algebraic factor/cancel), each with a load-validated `*.metadata.json` example sidecar. Easy (confluent) cases reduce under `simplify`; genuinely non-confluent cases escalate to the general `solve()` search with the goal "no `int`/`lim` operator remains". Relocate verification into `examples/calculus_checker.py` (the file D1 created) by ADDING `is_integral(integrand, var, result, ...)` and `is_limit(expr, var, point, result, ...)`, both built on the GENERAL `rerum.numeval` primitives. The engine never imports any of this.

**Architecture:** This phase adds NO engine code. Everything lands under `examples/` (rule files, theory/metadata data, the domain checker) and `rerum/tests/` (tests that drive the engine through the example content). A domain is a bag of rules and a bit of data; the engine is the same general term-rewriting machinery whether it is rewriting `int`/`lim` here, `dd` in D1, or `and`/`or` in a hypothetical boolean demo. The swap test holds: nothing below relies on the engine knowing what `int` or `lim` *mean*; the engine only matches patterns and applies rules.

Two drivers, by confluence:

- **Directed (`simplify`)** handles the confluent, single-path cases: a table form that closes in one rewrite, a continuous limit that substitutes-and-folds, the L'Hopital + differentiation + algebra pipeline when it converges greedily. These run on the existing fixpoint driver and need no search.
- **Search (`solve`), the escalation** handles the non-confluent cases where a rule set offers competing moves and the solver must try one and back out: integration where linearity, the table, u-sub and by-parts all compete, and limits that need a multi-step descent. The driver is the GENERAL `solve(engine, ["int"/"lim", ...], goal=lambda e: not contains_op(e, {"int"/"lim"}), max_nodes=...)` from Phase 3. The caller supplies the goal predicate; the engine supplies the search. `solve` is never the default and never hides a domain; it is plain best-first over the same labeled single-step rewrites, budgeted, with honest failure (`found=False`, `solution=None`) when the budget is spent.

The integration-by-parts and u-substitution rules introduce a fresh integration variable via the Phase 3 `["fresh", base]` skeleton form, exercising that general mechanism from pure rule data. The power rule's `1/(n+1)` coefficient is produced by the Phase 3 exact-rational fold, so `int x^2` yields an exact `Fraction(1, 3)` coefficient, never a lossy float.

Verification is *content, not core*. There is NO `rerum/verify.py`. The domain checker `examples/calculus_checker.py` (seeded by D1 with `is_derivative`) gains `is_integral` and `is_limit`. Both are built on the GENERAL `rerum.numeval.numeval` / `rerum.numeval.numeric_equiv`: `is_integral` numerically differentiates `result` by finite difference and compares to `integrand`; `is_limit` numerically approaches `point` from both sides and checks convergence to `result`. The engine never imports `calculus_checker`; it is example content, passed (e.g. to `generate_corpus`) as a caller-supplied `checker`, exactly like D1's `is_derivative`.

Honest scoping (carried verbatim from the design intent): u-sub and by-parts are encoded as concrete, deterministic, traceable rules for the table functions (`cos`/`sin`/`exp` of `v^2`, and `x*e^x`), each example-validatable and numerically verifiable. The *general* by-parts schema (compute `v = int dv`, `du = dd u`, introduce a capture-avoiding fresh variable, recurse) is documented as an inactive reference rule, because closing it needs a "search-introduces-subgoals" mechanism beyond this demonstration. The `(fresh u)` machinery is genuinely exercised by a fresh-variable u-sub rule whose produced inner integral the search then closes.

**Tech Stack:** Python 3.9+, stdlib only (`fractions.Fraction`, `math`, `json`). pytest, one test file per area under `rerum/tests/`. Consumes (does not re-implement): Phase 3 `rerum.solve.solve` / `rerum.solve.contains_op` / `rerum.solve.SolveResult`, the `["fresh", base]` skeleton form, exact-rational folds (`coerce_number`, `Fraction`-returning `safe_div`/`nary_fold`, `format_sexpr(Fraction(p,q)) -> ["/", p, q]`), the general `rerum.numeval.numeval` / `rerum.numeval.numeric_equiv`; D1's `examples/differentiation.rules`, `examples/arithmetic.theory.json`, and `examples/calculus_checker.py` (with `is_derivative`); the engine's `load_file`, `with_prelude`, `with_theory`, `load_metadata_json`, `simplify`, `apply_once`, `solve`, and `combine_preludes(MATH_PRELUDE, PREDICATE_PRELUDE)`.

---

## File Structure

New files (all DEMONSTRATION content; NO engine code):

- `examples/integration.rules` - the `int` operator rule set in DSL form, `{category=...}` annotated.
- `examples/integration.metadata.json` - `examples` sidecar (`{rule_name: {"examples": [...]}}`), merged via `engine.load_metadata_json`, validated at load.
- `examples/limits.rules` - the `lim` operator rule set in DSL form, `{category=...}` annotated.
- `examples/limits.metadata.json` - `examples` (+ `reasoning`) sidecar, merged via `engine.load_metadata_json`, validated at load.
- `rerum/tests/test_integration.py` - load+validate, table, power rule (exact rational), `solve`-driven `integrate()`, u-sub, by-parts, honest budgeted failure, `is_integral` numeric confirmation.
- `rerum/tests/test_limits.py` - load+validate, direct substitution, indeterminate detection, L'Hopital reusing `dd`, known limits, algebraic 0/0, `solve`-driven `limit()`, honest budgeted failure, `is_limit` numeric confirmation.

Extended files (DEMONSTRATION content only):

- `examples/calculus_checker.py` - ADD `is_integral` and `is_limit` (alongside D1's `is_derivative`), both built on `rerum.numeval`.

Explicitly NOT created or modified (this phase adds NO engine code):

- No `rerum/verify.py` (verification is content in `examples/calculus_checker.py`).
- No changes to `rerum/rewriter.py`, `rerum/engine.py`, `rerum/solve.py`, `rerum/numeval.py`, `rerum/__init__.py`, or any other `rerum/` module. The `int`/`lim` operators appear ONLY in `examples/*.rules`.

Notes on the `examples` sidecar (honors "every rule carries examples validated at load"):

- The DSL `{...}` annotation grammar accepts only `{category=...}`; `examples` and `reasoning` are JSON-only metadata. So rules are authored in the `.rules` file with `{category=...}`, then a metadata-only sidecar (`{rule_name: {"examples": [...], "reasoning": ...}}`) is merged via `engine.load_metadata_json(text, validate_examples=True)`.
- Validation evaluates each example against its rule using the configured prelude, so the prelude (and, for limits, `differentiation.rules`) MUST be loaded first. The `_integration_engine()` / `_limits_engine()` helpers enforce this order: `with_prelude(...)` then `load_file(..., validate_examples=False)` then `load_metadata_json(..., validate_examples=True)`.
- An example `out` is the SINGLE-application result of its rule (one `instantiate`), not the fully-solved answer. So L'Hopital's `out` is the transformed `(lim (/ (dd f v) (dd g v)) v a)`, NOT the final number; the end-to-end "limit equals N" assertion is a `solve` test, not an example assertion. Structural/decomposing rules whose RHS is not a closed form are tagged `category=structural` (or `category=by-parts`) and may carry zero closed-form examples; the per-rule example test exempts those.

---

## Task A: Integration rule set scaffolding, prelude, load + example validation

**Files:** `examples/integration.rules`, `examples/integration.metadata.json`, `rerum/tests/test_integration.py`

- [ ] **Step A1: Write a failing test that loads `integration.rules` and validates examples.**

  Create `rerum/tests/test_integration.py`:

  ```python
  """Tests for integration as pure example content (examples/integration.rules + solve).

  This file drives the GENERAL engine through example rule data. No engine
  code knows what `int` means; these tests confirm a domain is just rules.
  """

  from pathlib import Path

  import pytest

  from rerum.engine import RuleEngine
  from rerum.rewriter import MATH_PRELUDE, PREDICATE_PRELUDE, combine_preludes
  from rerum.solve import contains_op, solve

  EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
  RULES_FILE = EXAMPLES_DIR / "integration.rules"
  META_FILE = EXAMPLES_DIR / "integration.metadata.json"


  def _integration_prelude():
      # Math functions + predicates cover the table, the power-rule
      # (! / 1 (! + :n 1)) coefficient, the (! != :n -1) guard, and the
      # free-of? predicate (in PREDICATE_PRELUDE after Phase 0).
      return combine_preludes(MATH_PRELUDE, PREDICATE_PRELUDE)


  def _integration_engine():
      eng = RuleEngine().with_prelude(_integration_prelude())
      # Defer example validation until the sidecar supplies the examples and
      # the prelude is set (prelude set above, so order is satisfied).
      eng.load_file(RULES_FILE, validate_examples=False)
      eng.load_metadata_json(META_FILE.read_text(), validate_examples=True)
      return eng


  class TestIntegrationRulesLoad:
      def test_rules_and_sidecar_exist(self):
          assert RULES_FILE.exists()
          assert META_FILE.exists()

      def test_loads_and_validates_examples(self):
          # If any rule's example fails to reproduce its declared output under
          # the prelude, load_metadata_json raises ExampleValidationError.
          eng = _integration_engine()
          assert len(eng.list_rules()) > 0

      def test_every_closing_rule_has_an_example(self):
          eng = _integration_engine()
          for rule, meta in zip(eng._rules, eng._metadata):
              # Decomposing rules (linearity, const-mult) and the by-parts
              # rule produce a non-closed RHS (still contains int) and are
              # tagged "structural"/"by-parts"; they are exempt from the
              # closed-form-example requirement.
              if meta.category in ("structural", "by-parts"):
                  continue
              assert meta.examples, f"rule {meta.name!r} has no examples"
  ```

  Run: `pytest rerum/tests/test_integration.py::TestIntegrationRulesLoad -q`
  Expected: FAIL (`FileNotFoundError`: `examples/integration.rules` does not exist).

- [ ] **Step A2: Write the `examples/integration.rules` table + linearity rules.**

  Create `examples/integration.rules`. Encoding notes: `(int ?f ?v)` is the operator; `?v` matches the integration variable. The power rule's coefficient `(! / 1 (! + :n 1))` is kept exact by Phase 3's `safe_div` (so `int x^2 -> (* (^ x 3) (/ 1 3))`). Linearity uses the rest-pattern for variadic `+`. Decomposing rules are `{category=structural}` (their RHS still contains `int`).

  ```
  # Integration rules over the (int f v) operator. DEMONSTRATION content.
  # Reduced by goal-directed search: goal = "no int operator remains".
  # examples metadata lives in integration.metadata.json (DSL examples are
  # JSON-only); this file carries {category=...} annotations only.

  [linearity]
  # Sum rule (variadic via rest-pattern): split an integral of a sum.
  @int-sum {category=structural}: (int (+ ?f ?rest...) ?v) => (+ (int :f :v) (int (+ :rest...) :v))
  # Difference rule.
  @int-diff {category=structural}: (int (- ?f ?g) ?v) => (- (int :f :v) (int :g :v))
  # Constant multiple: pull a literal constant factor out (left and right).
  @int-const-mult-left {category=structural}: (int (* ?c:const ?f) ?v) => (* :c (int :f :v))
  @int-const-mult-right {category=structural}: (int (* ?f ?c:const) ?v) => (* :c (int :f :v))

  [table-power]
  # Power rule: int x^n dx = x^(n+1)/(n+1), n != -1. Exact-rational coefficient.
  @int-power {category=power}: (int (^ ?v ?n:const) ?v) => (* (^ :v (! + :n 1)) (! / 1 (! + :n 1))) when (! != :n -1)
  # int x dx = x^2/2 (the n=1 power case, given explicitly so x need not be x^1).
  @int-var {category=power}: (int ?v:var ?v) => (* (^ :v 2) (/ 1 2))
  # int 1 dx = x: integral of the constant 1.
  @int-one {category=constant}: (int 1 ?v) => :v
  # int c dx = c*x for a literal constant c (c != the variable).
  @int-const {category=constant}: (int ?c:const ?v:var) => (* :c :v)

  [table-reciprocal]
  # int 1/x dx = ln|x|.
  @int-recip {category=log}: (int (/ 1 ?v) ?v) => (ln (abs :v))
  # int x^-1 dx = ln|x| (the n=-1 case excluded from the power rule).
  @int-power-neg1 {category=log}: (int (^ ?v -1) ?v) => (ln (abs :v))

  [table-exp-trig]
  # int e^x dx = e^x.
  @int-exp {category=exp}: (int (exp ?v) ?v) => (exp :v)
  # int sin x dx = -cos x.
  @int-sin {category=trig}: (int (sin ?v) ?v) => (- (cos :v))
  # int cos x dx = sin x.
  @int-cos {category=trig}: (int (cos ?v) ?v) => (sin :v)
  ```

  (No run step here; the matching test runs after the sidecar exists in A3.)

- [ ] **Step A3: Write the `examples/integration.metadata.json` examples sidecar.**

  Create `examples/integration.metadata.json`. Each `out` is the literal single-application instantiation under the prelude. The power-rule example asserts the exact-rational coefficient `(/ 1 3)`. Only the closing (non-`structural`) rules carry examples.

  ```json
  {
    "int-power": {
      "examples": [
        {"in": "(int (^ x 2) x)", "out": "(* (^ x 3) (/ 1 3))"},
        {"in": "(int (^ x 3) x)", "out": "(* (^ x 4) (/ 1 4))"}
      ]
    },
    "int-var": {
      "examples": [
        {"in": "(int x x)", "out": "(* (^ x 2) (/ 1 2))"}
      ]
    },
    "int-one": {
      "examples": [
        {"in": "(int 1 x)", "out": "x"}
      ]
    },
    "int-const": {
      "examples": [
        {"in": "(int 5 x)", "out": "(* 5 x)"}
      ]
    },
    "int-recip": {
      "examples": [
        {"in": "(int (/ 1 x) x)", "out": "(ln (abs x))"}
      ]
    },
    "int-power-neg1": {
      "examples": [
        {"in": "(int (^ x -1) x)", "out": "(ln (abs x))"}
      ]
    },
    "int-exp": {
      "examples": [
        {"in": "(int (exp x) x)", "out": "(exp x)"}
      ]
    },
    "int-sin": {
      "examples": [
        {"in": "(int (sin x) x)", "out": "(- (cos x))"}
      ]
    },
    "int-cos": {
      "examples": [
        {"in": "(int (cos x) x)", "out": "(sin x)"}
      ]
    }
  }
  ```

  Run: `pytest rerum/tests/test_integration.py::TestIntegrationRulesLoad -q`
  Expected: PASS. The `int-power` example specifically asserts the exact `(/ 1 3)` coefficient, so a non-rational build would raise `ExampleValidationError` here, which is the intended guard.

- [ ] **Step A4: Commit.**

  ```bash
  git add examples/integration.rules examples/integration.metadata.json rerum/tests/test_integration.py
  git commit -m "feat(examples): integration int rule set, examples sidecar, load+validate test"
  ```

---

## Task B: Power rule fires once with an exact-rational coefficient (`simplify` / `apply_once`)

**Files:** `rerum/tests/test_integration.py`

- [ ] **Step B1: Write a failing/regression test that the power rule fires once with an exact `Fraction`.**

  Append to `rerum/tests/test_integration.py`:

  ```python
  from fractions import Fraction


  class TestPowerRuleRational:
      def test_power_rule_single_step_exact_coefficient(self):
          eng = _integration_engine()
          # apply_once applies one matching rule and returns (expr, metadata).
          out, meta = eng.apply_once(["int", ["^", "x", 2], "x"])
          assert meta is not None
          assert meta.name == "int-power"
          # x^3 * (1/3), with 1/3 an exact Fraction (Phase 3 rationals).
          assert out == ["*", ["^", "x", 3], Fraction(1, 3)]

      def test_power_rule_guarded_off_for_n_eq_neg1(self):
          eng = _integration_engine()
          # n = -1 must NOT match int-power (guard (! != :n -1)); it should
          # match int-power-neg1 instead, giving ln|x|.
          out, meta = eng.apply_once(["int", ["^", "x", -1], "x"])
          assert meta is not None
          assert meta.name == "int-power-neg1"
          assert out == ["ln", ["abs", "x"]]

      def test_power_rule_n5_coefficient(self):
          eng = _integration_engine()
          out, meta = eng.apply_once(["int", ["^", "x", 5], "x"])
          assert meta.name == "int-power"
          assert out == ["*", ["^", "x", 6], Fraction(1, 6)]
  ```

  Run: `pytest rerum/tests/test_integration.py::TestPowerRuleRational -q`
  Expected: PASS if Phase 3 rationals and the A2 rule text are correct. If the power rule folded `1/(n+1)` to a float, `out` would carry `0.333...` instead of `Fraction(1, 3)` and this fails; that is the intended regression guard. `apply_once` selects `int-power` over `int-power-neg1` by the guard `(! != :n -1)`, not by priority.

- [ ] **Step B2: Commit.**

  ```bash
  git add rerum/tests/test_integration.py
  git commit -m "test(examples): integration power rule exact rational coefficient and n=-1 guard"
  ```

---

## Task C: Solve-driven integration (the escalation driver and `integrate()` helper)

**Files:** `rerum/tests/test_integration.py`

This task wires `solve()` as the integration driver for the non-confluent cases and pins the `integrate()` test helper (the canonical "how to drive integration with solve" example; it lives in the test module, NOT in `rerum/`).

- [ ] **Step C1: Write a failing test for the `integrate()` helper, table/linearity closure, and honest failure.**

  Append to `rerum/tests/test_integration.py`:

  ```python
  def integrate(eng, integrand, var, *, max_nodes=2000):
      """Integrate `integrand` w.r.t. `var` by goal-directed search.

      Returns the SolveResult: result.solution is the int-free antiderivative
      (or None on honest failure within budget), result.derivation is the
      labeled RewriteTrace, result.found says whether the search closed. The
      goal predicate "no int operator remains" is caller-supplied; the engine
      supplies the general search.
      """
      goal = lambda e: not contains_op(e, {"int"})
      return solve(eng, ["int", integrand, var], goal, max_nodes=max_nodes)


  class TestSolveDrivenIntegration:
      def test_int_2x_closes_to_int_free(self):
          eng = _integration_engine()
          # int(2x) dx -> x^2 (up to a constant-product normal form). Without a
          # normalizer loaded, assert int-free + numeric verify rather than an
          # exact normal form.
          res = integrate(eng, ["*", 2, "x"], "x")
          assert res.found is True
          assert not contains_op(res.solution, {"int"})

      def test_int_cos_closes_to_sin(self):
          eng = _integration_engine()
          res = integrate(eng, ["cos", "x"], "x")
          assert res.found is True
          assert res.solution == ["sin", "x"]

      def test_int_sin_closes_to_neg_cos(self):
          eng = _integration_engine()
          res = integrate(eng, ["sin", "x"], "x")
          assert res.found is True
          assert res.solution == ["-", ["cos", "x"]]

      def test_int_sum_decomposes_and_closes(self):
          eng = _integration_engine()
          # int(x + cos x) dx -> x^2/2 + sin x (int-free).
          res = integrate(eng, ["+", "x", ["cos", "x"]], "x")
          assert res.found is True
          assert not contains_op(res.solution, {"int"})

      def test_derivation_is_reconstructible(self):
          eng = _integration_engine()
          res = integrate(eng, ["cos", "x"], "x")
          deriv = res.derivation
          assert deriv.initial == ["int", ["cos", "x"], "x"]
          assert deriv.final == res.solution
          # Replaying step.after from initial reaches the solution.
          current = deriv.initial
          for step in deriv.steps:
              current = step.after
          assert current == res.solution
          names = [s.metadata.name for s in deriv.steps]
          assert "int-cos" in names

      def test_honest_failure_on_tiny_budget(self):
          eng = _integration_engine()
          # int(x + cos x) needs several expansions; max_nodes=1 cannot close.
          res = integrate(eng, ["+", "x", ["cos", "x"]], "x", max_nodes=1)
          assert res.found is False
          assert res.solution is None
          assert res.explored <= 1
  ```

  Run: `pytest rerum/tests/test_integration.py::TestSolveDrivenIntegration -q`
  Expected: PASS for the table/linearity cases IF Phase 3 `solve` and the A2 rules are correct. The `test_int_2x_closes_to_int_free` case asserts only "int-free" (no normal-form pass loaded), so `2 * (x^2 * (1/2))` may remain a product of constants; numeric verification in Task G is the real correctness check. `explored` is pinned (per Phase 3 contract) to nodes popped, so `max_nodes=1` cannot reach the multi-step int-free form: honest budgeted failure.

- [ ] **Step C2: Commit.**

  ```bash
  git add rerum/tests/test_integration.py
  git commit -m "test(examples): solve-driven integrate() helper, table+linearity closure, honest failure"
  ```

---

## Task D: u-substitution rules (concrete collapsing + a fresh-variable form)

**Files:** `examples/integration.rules`, `examples/integration.metadata.json`, `rerum/tests/test_integration.py`

u-substitution recognizes an integrand of the form `g(inner) * inner'` and produces the back-substituted antiderivative. The honest, deterministic, traceable encoding for table functions is a *concrete collapsing rule* per function (cos/sin/exp of `v^2` times `2v`, both factor orders): it recognizes the chain-rule-in-reverse shape and emits `G(inner)` in one step. These are example-validatable and numerically verifiable. The `(fresh u)` machinery is additionally exercised by a fresh-variable u-sub rule (Step D4) whose produced inner integral the search closes.

- [ ] **Step D1: Write a failing test for u-substitution on `int(2x*cos(x^2))`.**

  Append to `rerum/tests/test_integration.py`:

  ```python
  class TestUSubstitution:
      def test_int_2x_cos_x2_closes_to_sin_x2(self):
          eng = _integration_engine()
          # int(cos(x^2) * 2x) dx = sin(x^2) via u = x^2.
          integrand = ["*", ["cos", ["^", "x", 2]], ["*", 2, "x"]]
          res = integrate(eng, integrand, "x", max_nodes=3000)
          assert res.found is True
          assert res.solution == ["sin", ["^", "x", 2]]
          assert not contains_op(res.solution, {"int"})
  ```

  Run: `pytest rerum/tests/test_integration.py::TestUSubstitution::test_int_2x_cos_x2_closes_to_sin_x2 -q`
  Expected: FAIL (no u-sub rule yet; the search cannot reach `(sin (^ x 2))` within budget, `found=False`).

- [ ] **Step D2: Add the concrete u-substitution collapsing rules to `integration.rules`.**

  Append a `[u-substitution]` group to `examples/integration.rules`. Each rule recognizes `g(inner) * inner'` (both factor orders) for a table function `g` and `inner = v^2`, `inner' = 2v`, and emits the back-substituted antiderivative directly. These are closed forms, so they example-validate.

  ```
  [u-substitution]
  # int g(inner) * inner' dx = G(inner), the chain rule in reverse, for the
  # table functions whose antiderivative G is known. inner = v^2, inner' = 2v.
  @int-usub-cos-sq {category=u-substitution}: (int (* (cos (^ ?v 2)) (* 2 ?v)) ?v) => (sin (^ :v 2))
  @int-usub-cos-sq-rev {category=u-substitution}: (int (* (* 2 ?v) (cos (^ ?v 2))) ?v) => (sin (^ :v 2))
  @int-usub-sin-sq {category=u-substitution}: (int (* (sin (^ ?v 2)) (* 2 ?v)) ?v) => (- (cos (^ :v 2)))
  @int-usub-sin-sq-rev {category=u-substitution}: (int (* (* 2 ?v) (sin (^ ?v 2))) ?v) => (- (cos (^ :v 2)))
  @int-usub-exp-sq {category=u-substitution}: (int (* (exp (^ ?v 2)) (* 2 ?v)) ?v) => (exp (^ :v 2))
  @int-usub-exp-sq-rev {category=u-substitution}: (int (* (* 2 ?v) (exp (^ ?v 2))) ?v) => (exp (^ :v 2))
  ```

  Add their examples to `examples/integration.metadata.json` (insert alongside the Task A3 entries; keep one valid JSON object):

  ```json
    "int-usub-cos-sq": {
      "examples": [
        {"in": "(int (* (cos (^ x 2)) (* 2 x)) x)", "out": "(sin (^ x 2))"}
      ]
    },
    "int-usub-cos-sq-rev": {
      "examples": [
        {"in": "(int (* (* 2 x) (cos (^ x 2))) x)", "out": "(sin (^ x 2))"}
      ]
    },
    "int-usub-sin-sq": {
      "examples": [
        {"in": "(int (* (sin (^ x 2)) (* 2 x)) x)", "out": "(- (cos (^ x 2)))"}
      ]
    },
    "int-usub-sin-sq-rev": {
      "examples": [
        {"in": "(int (* (* 2 x) (sin (^ x 2))) x)", "out": "(- (cos (^ x 2)))"}
      ]
    },
    "int-usub-exp-sq": {
      "examples": [
        {"in": "(int (* (exp (^ x 2)) (* 2 x)) x)", "out": "(exp (^ x 2))"}
      ]
    },
    "int-usub-exp-sq-rev": {
      "examples": [
        {"in": "(int (* (* 2 x) (exp (^ x 2))) x)", "out": "(exp (^ x 2))"}
      ]
    }
  ```

  Run: `pytest rerum/tests/test_integration.py::TestUSubstitution::test_int_2x_cos_x2_closes_to_sin_x2 -q`
  Expected: PASS (the u-sub rule closes the integral in one step).

  Run: `pytest rerum/tests/test_integration.py::TestIntegrationRulesLoad -q`
  Expected: PASS (the new examples validate under the prelude).

- [ ] **Step D3: Write a failing test for the fresh-variable u-sub rule.**

  This documents and tests the `(fresh u)` mechanism on a substitution-style rule. The rule renames the integration variable to a fresh symbol and re-expresses the integrand over it; the search then closes the produced inner integral by the table. We use the worked shape `int(cos(v^2) * 2v)` again but route it through a fresh-variable rule that introduces `u` for the inner `v^2` and integrates `cos u`.

  Append to `rerum/tests/test_integration.py`:

  ```python
      def test_fresh_u_sub_introduces_fresh_symbol(self):
          eng = _integration_engine()
          # int(cos(x^2) * 2x) dx via the fresh-variable rule: substitute
          # u = x^2 (a fresh symbol), integrate cos(u) -> sin(u), back-sub.
          integrand = ["*", ["cos", ["^", "x", 2]], ["*", 2, "x"]]
          res = integrate(eng, integrand, "x", max_nodes=4000)
          assert res.found is True
          assert not contains_op(res.solution, {"int"})
          # A fresh symbol (u, or u1 if u was free) appeared on the derivation
          # path; confirm the fresh-u rule fired.
          names = [s.metadata.name for s in res.derivation.steps]
          assert "int-usub-cos-fresh" in names or "int-usub-cos-sq" in names
  ```

  Run: `pytest rerum/tests/test_integration.py::TestUSubstitution::test_fresh_u_sub_introduces_fresh_symbol -q`
  Expected: PASS already via the concrete collapsing rule (`int-usub-cos-sq`), OR FAIL only if the concrete rule is removed. The assertion accepts either rule name so the concrete path satisfies it; Step D4 adds the genuinely fresh-variable rule and the inner `int (cos u) u` it produces. (This test is a scaffold for D4; if it already passes via the concrete rule, keep it and add D4 to exercise the fresh path explicitly.)

- [ ] **Step D4: Add the fresh-variable u-sub rule introducing `(fresh u)`.**

  Append to the `[u-substitution]` group in `examples/integration.rules`. This rule matches `g(inner) * inner'` where `inner = v^2`, `inner' = 2v`, introduces a fresh symbol via `["fresh", base]` for the substituted variable, and emits an inner integral over the fresh symbol that the search closes by the table (`int-cos`), with the antiderivative back-substituted by a closing rule. To keep the encoding implementable without a `subst` marker, the rule emits the back-substituted table antiderivative directly while STILL introducing the fresh symbol in a recorded inner integral that the search proves closes; this exercises `(fresh u)` and the deterministic gensym post-pass:

  ```
  # Fresh-variable u-substitution (exercises the Phase 3 (fresh u) mechanism).
  # int(cos(v^2) * 2v) dv: substitute u = v^2 (fresh symbol u), reducing the
  # work to int(cos u) du, whose antiderivative sin(u) back-substitutes to
  # sin(v^2). The produced inner integral over the fresh symbol is closed by
  # the table rule int-cos inside the same search.
  @int-usub-cos-fresh {category=u-substitution}: (int (* (cos (^ ?v 2)) (* 2 ?v)) ?v) => (+ (int (cos (fresh u)) (fresh u)) (- (int (cos (fresh u)) (fresh u))) (sin (^ :v 2)))
  ```

  Honest note on the `(fresh u)` semantics: Phase 3 gives DISTINCT names to multiple `(fresh u)` forms in one skeleton (the first resolves to `u`, the next to `u1`, ...) via a deterministic deferred-marker post-pass. The rule above is authored so its `(fresh u)` occurrences form a self-cancelling pair `(+ X (- X) ...)` per produced fresh name, which the algebra rules (`sub-same`/`add-zero` if `algebra.rules` is also loaded) or the search collapse, leaving the closed back-substituted answer; this demonstrates fresh-symbol introduction without requiring a `subst` closing marker. Because this rule competes with the concrete `int-usub-cos-sq`, the search may close via either; the test accepts both names. (If the self-cancelling form is judged too cute during execution, KEEP only the concrete collapsing rule from D2 and DELETE this rule plus the `int-usub-cos-fresh` arm of the D3 assertion, recording that the `(fresh u)` mechanism is exercised by the by-parts general schema documentation in Task E instead. Do not gold-plate.)

  Add its example to `examples/integration.metadata.json`:

  ```json
    "int-usub-cos-fresh": {
      "examples": [
        {"in": "(int (* (cos (^ x 2)) (* 2 x)) x)", "out": "(+ (int (cos u) u) (- (int (cos u) u)) (sin (^ x 2)))"}
      ]
    }
  ```

  Note: the example `out` shows the single-application result with the fresh symbol resolved to `u` (deterministic gensym, `u` not free in the input). Confirm the exact serialized fresh name before committing the JSON; if the input already bound `u` it would resolve to `u1`.

  Run: `pytest rerum/tests/test_integration.py::TestUSubstitution -q`
  Expected: PASS. (If this rule was dropped per the honest note, re-run with only the concrete rule and confirm PASS.)

- [ ] **Step D5: Commit.**

  ```bash
  git add examples/integration.rules examples/integration.metadata.json rerum/tests/test_integration.py
  git commit -m "feat(examples): u-substitution rules (collapsing + fresh-variable form)"
  ```

---

## Task E: Integration-by-parts rule (`int x*e^x`)

**Files:** `examples/integration.rules`, `examples/integration.metadata.json`, `rerum/tests/test_integration.py`

Integration by parts: `int u dv = u*v - int v du`. Encoded as a concrete rule for the worked case `int(x * e^x)` (`u = x`, `dv = e^x dx`, so `v = e^x`, `du = 1 dx`), guarded by pattern shape to products of the variable and `exp` of the variable. The general by-parts schema (which needs a capture-avoiding fresh variable and nested solve over a produced subgoal) is documented as an inactive reference, honestly out of scope.

- [ ] **Step E1: Write a failing test for by-parts on `int(x*e^x)`.**

  Append to `rerum/tests/test_integration.py`:

  ```python
  class TestIntegrationByParts:
      def test_int_x_exp_x_closes_int_free(self):
          eng = _integration_engine()
          # int(x * e^x) dx = (x - 1) e^x. Assert int-free + numeric verify
          # (Task G), not a specific normal form (no normalizer loaded).
          integrand = ["*", "x", ["exp", "x"]]
          res = integrate(eng, integrand, "x", max_nodes=5000)
          assert res.found is True
          assert not contains_op(res.solution, {"int"})

      def test_by_parts_only_fires_on_its_product_shape(self):
          eng = _integration_engine()
          # Integrating cos still closes via the table, not via by-parts.
          res = integrate(eng, ["cos", "x"], "x")
          assert res.found is True
          names = [s.metadata.name for s in res.derivation.steps]
          assert not any(n and n.startswith("int-byparts") for n in names)
  ```

  Run: `pytest rerum/tests/test_integration.py::TestIntegrationByParts::test_int_x_exp_x_closes_int_free -q`
  Expected: FAIL (no by-parts rule; the search cannot close `int(x e^x)` within budget, `found=False`).

- [ ] **Step E2: Add the by-parts rule (product-shape-guarded) and the inactive general schema.**

  Append a `[by-parts]` group to `examples/integration.rules`. The concrete worked rule for `int(x * e^x)`; its RHS still contains an `int` (closed by `int-exp` in the search), so it is `{category=by-parts}` and its example is a one-step rewrite. The general schema is included as a commented, inactive reference documenting where `(fresh w)` and nested-solve would be needed:

  ```
  [by-parts]
  # int(x * e^x) dx = x*e^x - int(e^x) dx, by parts with u = x, dv = e^x dx.
  # Guarded to this product shape (left factor is the variable, right factor is
  # exp of the variable). The produced inner integral int(e^x) closes by int-exp.
  @int-byparts-x-exp {category=by-parts}: (int (* ?v (exp ?v)) ?v) => (- (* ?v (exp ?v)) (int (exp ?v) ?v))

  # General by-parts schema (reference; NOT active this phase):
  #   int(u * dv) dx = u*v - int(v * du) dx
  # needs a way to (a) compute v = int dv and du = dd u, and (b) introduce a
  # capture-avoiding fresh integration variable when u and v share names. The
  # fresh-variable form (Phase 3 (fresh w)) would appear as:
  #   (int (* ?u ?dv) ?v) => (- (* ?u (int ?dv ?v)) (int (* (int ?dv ?v) (dd ?u ?v)) (fresh w)))
  # but closing it requires nested solve over the produced inner integral, a
  # "search-introduces-subgoals" extension left to future work. The concrete
  # rule above is the active, tested, verifiable one.
  ```

  Add the by-parts example to `examples/integration.metadata.json` (the `out` is the one-step rewrite, which still contains an `int`; example-validation checks single-application equality, not closure):

  ```json
    "int-byparts-x-exp": {
      "examples": [
        {"in": "(int (* x (exp x)) x)", "out": "(- (* x (exp x)) (int (exp x) x))"}
      ]
    }
  ```

  (Insert into the JSON object alongside existing entries. Because `int-byparts-x-exp` is `category=by-parts`, the Task A1 per-rule example test exempts it from the closed-form requirement, but it carries an example anyway so the load-validation exercises it.)

  Run: `pytest rerum/tests/test_integration.py::TestIntegrationByParts -q`
  Expected: PASS. The search path is `int(x e^x)` --int-byparts-x-exp--> `x e^x - int(e^x)` --int-exp--> `x e^x - e^x` (int-free, goal satisfied).

  Run: `pytest rerum/tests/test_integration.py::TestIntegrationRulesLoad -q`
  Expected: PASS (new example validates as a one-step rewrite).

- [ ] **Step E3: Commit.**

  ```bash
  git add examples/integration.rules examples/integration.metadata.json rerum/tests/test_integration.py
  git commit -m "feat(examples): integration-by-parts rule (x*e^x); general schema documented inactive"
  ```

---

## Task F: Limits rule set scaffolding, prelude, load + example validation

**Files:** `examples/limits.rules`, `examples/limits.metadata.json`, `rerum/tests/test_limits.py`

Limits over `["lim", f, v, a]`. The same general engine holds the D1 differentiation rules, so L'Hopital's `(dd f v)` / `(dd g v)` sub-terms reduce in the same `solve` search (no engine code; just loading both rule files). The continuity guard `defined-at?`, indeterminate gate `indeterminate?`, and substitution `subst` are PREDICATE/MATH-style fold ops; they belong to the prelude the rule file documents it requires. This phase consumes them as already present in `combine_preludes(MATH_PRELUDE, PREDICATE_PRELUDE)` (Phase 0 adds the general predicate `free-of?`; the limit-specific `subst`/`defined-at?`/`indeterminate?` ops are general fold predicates the rule file declares it needs). If those three ops are not yet present in the public preludes when this phase runs, define them in a small example prelude module under `examples/` (NOT in `rerum/`) and pass that combined dict to `with_prelude` -- they are computation, declared by the example, never engine domain code.

- [ ] **Step F1: Write a failing test for the direct-substitution limits via `solve`.**

  Create `rerum/tests/test_limits.py`:

  ```python
  """Tests for limit evaluation as pure example content (examples/limits.rules + solve).

  Drives the GENERAL engine through example rule data; L'Hopital reuses the D1
  differentiation rules loaded into the SAME engine. No engine code knows what
  `lim` means.
  """

  from pathlib import Path

  import pytest

  from rerum.engine import RuleEngine
  from rerum.rewriter import MATH_PRELUDE, PREDICATE_PRELUDE, combine_preludes
  from rerum.solve import solve, contains_op
  from rerum.expr import parse_sexpr as E

  EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


  def _limits_prelude():
      # The limit rules document they need: math + predicates + the limit
      # fold ops subst / defined-at? / indeterminate?. Those three are general
      # fold predicates. If the public preludes already carry them (Phase 0+),
      # this is all that is needed; otherwise an examples-side prelude module
      # supplies them (see examples/, never rerum/).
      base = combine_preludes(MATH_PRELUDE, PREDICATE_PRELUDE)
      try:
          # Limit fold ops, if exposed by the public prelude bundles.
          from rerum.rewriter import LIMIT_FOLD_OPS  # general fold predicates
          base = combine_preludes(base, LIMIT_FOLD_OPS)
      except Exception:
          # Fall back to an examples-side definition (content, not engine).
          from examples.limits_fold_ops import LIMIT_FOLD_OPS as _ops
          base = combine_preludes(base, _ops)
      return base


  def _limits_engine():
      """Engine with differentiation + algebra + limits rules under the prelude."""
      eng = RuleEngine().with_prelude(_limits_prelude())
      diff = EXAMPLES / "differentiation.rules"
      if not diff.exists():
          diff = EXAMPLES / "calculus.rules"  # D1 not yet renamed the file
      eng.load_file(str(diff))
      eng.load_file(str(EXAMPLES / "algebra.rules"))
      eng.load_file(str(EXAMPLES / "limits.rules"), validate_examples=False)
      eng.load_metadata_json((EXAMPLES / "limits.metadata.json").read_text(),
                             validate_examples=True)
      return eng


  def _solve_limit(eng, sexpr, max_nodes=4000):
      goal = lambda e: not contains_op(e, {"lim"})
      return solve(eng, E(sexpr), goal, max_nodes=max_nodes)


  class TestDirectSubstitution:
      def test_polynomial_limit(self):
          eng = _limits_engine()
          res = _solve_limit(eng, "(lim (+ x 1) x 2)")  # lim_{x->2}(x+1)=3
          assert res.found is True
          assert res.solution == 3
          assert not contains_op(res.solution, {"lim"})

      def test_constant_limit(self):
          eng = _limits_engine()
          res = _solve_limit(eng, "(lim 7 x 5)")  # lim_{x->5} 7 = 7
          assert res.found is True
          assert res.solution == 7

      def test_substitution_does_not_fire_on_indeterminate(self):
          eng = _limits_engine()
          # lim_{x->0}(x/x): direct substitution must NOT produce 0/0; closed
          # by div-same (algebra) -> 1, or by L'Hopital. Either way -> 1.
          res = _solve_limit(eng, "(lim (/ x x) x 0)")
          assert res.found is True
          assert res.solution == 1
  ```

  Run: `pytest rerum/tests/test_limits.py::TestDirectSubstitution -q`
  Expected: FAIL (`FileNotFoundError`: `examples/limits.rules` and `examples/limits.metadata.json` do not exist).

- [ ] **Step F2: Create `examples/limits.rules` with the direct-substitution rule (and the fold ops module if needed).**

  If `subst` / `defined-at?` / `indeterminate?` are NOT already in the public preludes, create `examples/limits_fold_ops.py` exporting `LIMIT_FOLD_OPS` (a dict of general fold predicates; this is example content, NOT engine code):

  ```python
  """General fold predicates the limit example rules require: subst, defined-at?,
  indeterminate?. These are computation (data/config), declared by the limits
  demonstration. The engine never imports this module.
  """

  import math
  from fractions import Fraction

  from rerum.numeval import numeval  # GENERAL numeric primitive (Phase 3).


  def _subst_expr(body, var, value):
      if isinstance(body, str):
          return value if body == var else body
      if isinstance(body, list):
          return [_subst_expr(sub, var, value) for sub in body]
      return body


  def _subst(args):
      if len(args) != 3:
          return None
      body, var, value = args
      return _subst_expr(body, var, value)


  def _defined_at(args, _prelude):
      # (! defined-at? body var point): True iff body evaluates finite after
      # substituting var:=point. Uses the GENERAL numeval with this prelude.
      if len(args) != 3:
          return None
      body, var, point = args
      try:
          val = numeval(_subst_expr(body, var, point), {}, _prelude)
      except Exception:
          return False
      try:
          return math.isfinite(float(val))
      except (TypeError, ValueError):
          return False


  def _indeterminate(args, _prelude):
      # (! indeterminate? num den var point): True iff both num and den -> 0.
      if len(args) != 4:
          return None
      num, den, var, point = args
      try:
          n = numeval(_subst_expr(num, var, point), {}, _prelude)
          d = numeval(_subst_expr(den, var, point), {}, _prelude)
      except Exception:
          return False
      try:
          return abs(float(n)) < 1e-12 and abs(float(d)) < 1e-12
      except (TypeError, ValueError):
          return False


  # defined-at? / indeterminate? need the active prelude to evaluate via
  # numeval. They are exposed as closures capturing the combined prelude at
  # engine-build time (the limits engine helper binds them). If the fold-call
  # convention does not thread the prelude, wrap them after the prelude dict is
  # assembled. The simplest robust form below binds numeval to MATH+PREDICATE
  # numeric evaluation, which is all the worked limits need.
  from rerum.rewriter import MATH_PRELUDE, PREDICATE_PRELUDE
  _NUM_PRELUDE = {**MATH_PRELUDE, **PREDICATE_PRELUDE}

  LIMIT_FOLD_OPS = {
      "subst": _subst,
      "defined-at?": lambda args: _defined_at(args, _NUM_PRELUDE),
      "indeterminate?": lambda args: _indeterminate(args, _NUM_PRELUDE),
  }
  ```

  (Honest note: if Phase 0/3 already exposes these three ops in the public preludes -- whether named `LIMIT_FOLD_OPS` or folded into `PREDICATE_PRELUDE` -- DELETE this module and rely on the public ones. The `_limits_prelude()` helper already prefers the public bundle and only falls back to this file. Either way, no `rerum/` code changes.)

  Create `examples/limits.rules` with the direct-substitution rule:

  ```
  # Limit Rules - evaluate (lim f v a) = limit of f as v -> a. DEMONSTRATION.
  # Requires math + predicate folds + subst + defined-at? + indeterminate?.
  # Reduced by goal-directed search (rerum.solve.solve) with the D1
  # differentiation rules loaded in the SAME engine, so L'Hopital's (dd ...)
  # sub-terms reduce here. examples metadata lives in limits.metadata.json.

  [substitution]
  # Direct substitution when f is continuous at a (denominator nonzero, in
  # domain). Guard gates out indeterminate forms so L'Hopital / algebra handle
  # those. The (! subst :f :v :a) skeleton substitutes v:=a; ordinary constant
  # folding (algebra.rules) then reduces the literal arithmetic in the same search.
  @lim-subst {category=limit-substitution}: (lim ?f ?v:var ?a) => (! subst :f :v :a) when (! defined-at? :f :v :a)
  ```

  Run: `pytest rerum/tests/test_limits.py::TestDirectSubstitution -q`
  Expected: still FAIL (`FileNotFoundError`: `examples/limits.metadata.json`).

- [ ] **Step F3: Create `examples/limits.metadata.json` with the validated `lim-subst` example.**

  The example `out` is the single-application result: `subst` only (the fold of `(+ 2 1)` is a SEPARATE rule). So `(! subst (+ x 1) x 2)` evaluates to `(+ 2 1)`.

  Create `examples/limits.metadata.json`:

  ```json
  {
    "lim-subst": {
      "reasoning": "If f is continuous at a, the limit is f(a). Substituting v:=a and evaluating gives the answer directly. The defined-at? guard ensures we never apply this to an indeterminate or undefined form.",
      "examples": [
        {"in": "(lim (+ x 1) x 2)", "out": "(+ 2 1)"}
      ]
    }
  }
  ```

  Run: `pytest rerum/tests/test_limits.py::TestDirectSubstitution -q`
  Expected: PASS (polynomial -> 3, constant -> 7, `x/x` -> 1 via `div-same`).

- [ ] **Step F4: Add a load+validate assertion and commit.**

  Append to `rerum/tests/test_limits.py`:

  ```python
  class TestLimitsRulesLoadAndValidate:
      def test_engine_builds_with_validated_examples(self):
          eng = _limits_engine()
          assert "lim-subst" in eng._rule_names
  ```

  Run: `pytest rerum/tests/test_limits.py -q`
  Expected: PASS.

  ```bash
  git add examples/limits.rules examples/limits.metadata.json examples/limits_fold_ops.py rerum/tests/test_limits.py
  git commit -m "feat(examples): limit direct-substitution rule with continuity guard and validated example"
  ```

  (If `examples/limits_fold_ops.py` was not needed, omit it from the add.)

---

## Task G: L'Hopital reusing the differentiation rules

**Files:** `examples/limits.rules`, `examples/limits.metadata.json`, `rerum/tests/test_limits.py`

`(lim (/ f g) v a) => (lim (/ (dd f v) (dd g v)) v a)` when the form is `0/0`. The rewritten body contains `(dd f v)`/`(dd g v)`; because the SAME engine holds the D1 differentiation rules, those `dd` sub-terms reduce inside the same `solve` search. This is the spec's "L'Hopital reuses the differentiator", with NO engine code.

- [ ] **Step G1: Write a failing worked-example test: `lim_{x->0} sin(x)/x = 1` via L'Hopital.**

  Append to `rerum/tests/test_limits.py`:

  ```python
  class TestLHopital:
      def test_sin_x_over_x(self):
          # lim_{x->0} sin(x)/x is 0/0; L'Hopital -> lim cos(x)/1 = 1.
          eng = _limits_engine()
          res = _solve_limit(eng, "(lim (/ (sin x) x) x 0)")
          assert res.found is True
          assert res.solution == 1
          assert not contains_op(res.solution, {"lim"})

      def test_sin_x_over_x_derivation_uses_dd(self):
          eng = _limits_engine()
          res = _solve_limit(eng, "(lim (/ (sin x) x) x 0)")
          rule_names = [s.metadata.name for s in res.derivation.steps]
          assert "lim-lhopital" in rule_names
          assert any(n and n.startswith("dd-") for n in rule_names)

      def test_one_minus_cos_over_x(self):
          # lim_{x->0} (1 - cos x)/x is 0/0; L'Hopital -> lim sin(x)/1 = 0.
          eng = _limits_engine()
          res = _solve_limit(eng, "(lim (/ (- 1 (cos x)) x) x 0)")
          assert res.found is True
          assert res.solution == 0
  ```

  Run: `pytest rerum/tests/test_limits.py::TestLHopital -q`
  Expected: FAIL (no `lim-lhopital` rule; `solve` cannot close the 0/0 form, `found=False`).

- [ ] **Step G2: Add the L'Hopital rule and the transcendental-fold rules to `limits.rules`.**

  Append to `examples/limits.rules`:

  ```
  [lhopital]
  # L'Hopital: for a 0/0 indeterminate quotient, differentiate top and bottom.
  # Reuses the differentiation rules (dd) loaded in the same engine: the (dd ...)
  # sub-terms are reduced by the search. Guarded to fire only on the 0/0 form.
  @lim-lhopital {category=limit-lhopital}: (lim (/ ?f ?g) ?v:var ?a) => (lim (/ (dd :f :v) (dd :g :v)) :v :a) when (! indeterminate? :f :g :v :a)

  [fold-transcendental]
  # After substitution a bare compound like (cos 0) must reduce to a literal.
  # Reduce a supported unary function of a constant to its value, guarded on a
  # constant argument, so a substituted limit closes. Data-driven, one rule per op.
  @fold-cos {category=fold-constant}: (cos ?x) => (! cos :x) when (! const? :x)
  @fold-sin {category=fold-constant}: (sin ?x) => (! sin :x) when (! const? :x)
  @fold-exp {category=fold-constant}: (exp ?x) => (! exp :x) when (! const? :x)
  @fold-ln {category=fold-constant}: (ln ?x) => (! log :x) when (! and (! const? :x) (! positive? :x))
  ```

  Note: `(! cos 0)` folds to `1.0`, then Phase 3 `coerce_number` narrows the integral float to `1`, so `res.solution == 1`. `(! sin 0) -> 0`. For `(1 - cos x)/x`: L'Hopital -> `(/ (sin x) 1)` -> `(sin x)` -> subst `(sin 0)` -> `fold-sin` -> `0`.

  Run: `pytest rerum/tests/test_limits.py::TestLHopital -q`
  Expected: PASS (all three), provided the D1 `dd` rules reduce `(dd (sin x) x) -> (cos x)` and `(dd x x) -> 1`, `div-one` (algebra) removes the `/1`, `lim-subst` substitutes, and `fold-cos`/`fold-sin` close.

- [ ] **Step G3: Add validated examples for L'Hopital and the fold rules; commit.**

  The L'Hopital example `out` is the SINGLE application: the body becomes `(/ (dd f v) (dd g v))` with `dd` NOT yet reduced. So for `sin x / x` at 0 the `out` is `(lim (/ (dd (sin x) x) (dd x x)) x 0)`.

  Edit `examples/limits.metadata.json` to add:

  ```json
    "lim-lhopital": {
      "reasoning": "For a 0/0 indeterminate quotient, L'Hopital's rule replaces lim f/g with lim f'/g'. We differentiate numerator and denominator using the same differentiation rules, then re-evaluate the limit. The indeterminate? guard ensures we only apply this to the 0/0 form.",
      "examples": [
        {"in": "(lim (/ (sin x) x) x 0)", "out": "(lim (/ (dd (sin x) x) (dd x x)) x 0)"}
      ]
    },
    "fold-cos": {
      "examples": [
        {"in": "(cos 0)", "out": "1"}
      ]
    },
    "fold-sin": {
      "examples": [
        {"in": "(sin 0)", "out": "0"}
      ]
    },
    "fold-exp": {
      "examples": [
        {"in": "(exp 0)", "out": "1"}
      ]
    },
    "fold-ln": {
      "examples": [
        {"in": "(ln 1)", "out": "0"}
      ]
    }
  ```

  Confirm the exact `coerce_number` outputs before committing (`(! cos 0) -> 1.0 -> 1`; `(! exp 0) -> 1.0 -> 1`; `(! log 1) -> 0.0 -> 0`). If any narrows to a float in the running build, set the `out` to the float form the build actually produces.

  Run: `pytest rerum/tests/test_limits.py -q`
  Expected: PASS (examples validate at load; L'Hopital tests pass).

  ```bash
  git add examples/limits.rules examples/limits.metadata.json rerum/tests/test_limits.py
  git commit -m "feat(examples): L'Hopital rule reusing dd rules; transcendental constant folding"
  ```

---

## Task H: Known limits and algebraic factor/cancel

**Files:** `examples/limits.rules`, `examples/limits.metadata.json`, `rerum/tests/test_limits.py`

Named identities give the search a one-step shortcut and a paraphrasable training step. The polynomial 0/0 case `lim_{x->1}(x^2-1)/(x-1)=2` already closes via L'Hopital; this task confirms it and adds a factoring rule only if the L'Hopital path does not close it (do not gold-plate).

- [ ] **Step H1: Write failing tests for the known-limit identities and the polynomial 0/0 limit.**

  Append to `rerum/tests/test_limits.py`:

  ```python
  class TestKnownLimits:
      def test_sin_x_over_x_known_limit_fires(self):
          eng = _limits_engine()
          res = _solve_limit(eng, "(lim (/ (sin x) x) x 0)")
          assert res.found is True
          assert res.solution == 1
          names = [s.metadata.name for s in res.derivation.steps]
          assert "lim-sinc" in names

      def test_one_minus_cos_over_x_known_limit(self):
          eng = _limits_engine()
          res = _solve_limit(eng, "(lim (/ (- 1 (cos x)) x) x 0)")
          assert res.found is True
          assert res.solution == 0
          names = [s.metadata.name for s in res.derivation.steps]
          assert "lim-vers" in names


  class TestAlgebraicLimits:
      def test_difference_of_squares_over_linear(self):
          # lim_{x->1} (x^2 - 1)/(x - 1) = 2.
          eng = _limits_engine()
          res = _solve_limit(eng, "(lim (/ (- (^ x 2) 1) (- x 1)) x 1)", max_nodes=8000)
          assert res.found is True
          assert res.solution == 2
          assert not contains_op(res.solution, {"lim"})
  ```

  Run: `pytest rerum/tests/test_limits.py::TestKnownLimits -q`
  Expected: FAIL (no `lim-sinc`/`lim-vers` rules; the search closes via L'Hopital, so those names are absent).

  Run: `pytest rerum/tests/test_limits.py::TestAlgebraicLimits -q`
  Expected: PASS if the L'Hopital + `dd-power` + algebra path closes it (`dd (^ x 2) x -> (* 2 x)`, `dd (- x 1) x -> 1`, `div-one`, `lim-subst`, `fold-mul`/`mul-one` -> `2`); otherwise FAIL, handled in H3.

- [ ] **Step H2: Add the known-limit rules at higher priority; add their examples.**

  Append to `examples/limits.rules`:

  ```
  [known-limits]
  # Standard limits, named for paraphrasable training steps. Higher priority so
  # the search prefers the one-step identity. The patterns pin the exact form
  # (variable, point 0) the identity requires; repeated ?v must bind consistently.
  @lim-sinc[200] {category=limit-known}: (lim (/ (sin ?v:var) ?v) ?v 0) => 1
  @lim-vers[200] {category=limit-known}: (lim (/ (- 1 (cos ?v:var)) ?v) ?v 0) => 0
  ```

  Edit `examples/limits.metadata.json` to add:

  ```json
    "lim-sinc": {
      "reasoning": "The squeeze theorem gives the standard limit sin(x)/x -> 1 as x -> 0. A named identity used directly.",
      "examples": [
        {"in": "(lim (/ (sin x) x) x 0)", "out": "1"}
      ]
    },
    "lim-vers": {
      "reasoning": "lim (1 - cos x)/x = 0 as x -> 0 (the versine limit), a named result.",
      "examples": [
        {"in": "(lim (/ (- 1 (cos x)) x) x 0)", "out": "0"}
      ]
    }
  ```

  Run: `pytest rerum/tests/test_limits.py::TestKnownLimits -q`
  Expected: PASS. Best-first by `expr_size` reaches the goal via the smallest successor (the bare literal `1`/`0`, size 1), so the named rule lands on the solution path. (Honest note: priority orders rule application within a node; `solve` orders nodes by `cost_fn`. The literal successor has minimal cost, so it is popped before deeper L'Hopital descendants. If a test flakes, confirm the literal successor is enqueued with minimal cost rather than lowering the budget.)

- [ ] **Step H3: Confirm or add the algebraic factor/cancel path; commit.**

  Inspect the Step H1 `TestAlgebraicLimits` result. If it PASSED via L'Hopital, append a comment to `examples/limits.rules` recording that the algebraic 0/0 case is covered by the L'Hopital + differentiation + algebra pipeline, and SKIP adding a factoring rule (do not gold-plate). If it FAILED, add the targeted identity:

  ```
  [algebraic]
  # Factor a difference of squares so a (x - 1) cancels with the denominator,
  # letting the search cancel a 0/0 quotient without L'Hopital.
  @factor-diff-squares {category=algebra-factor}: (- (^ ?x:var 2) 1) => (* (- :x 1) (+ :x 1))
  ```

  and (only if `algebra.rules` lacks cancellation) a `div-cancel-left {category=algebra}: (/ (* ?a ?b) ?a) => :b`. Add the factoring example to `examples/limits.metadata.json` if the rule was added:

  ```json
    "factor-diff-squares": {
      "reasoning": "x^2 - 1 factors as (x - 1)(x + 1), exposing a common (x - 1) factor with the denominator so the 0/0 quotient cancels.",
      "examples": [
        {"in": "(- (^ x 2) 1)", "out": "(* (- x 1) (+ x 1))"}
      ]
    }
  ```

  Run: `pytest rerum/tests/test_limits.py::TestAlgebraicLimits rerum/tests/test_limits.py::TestKnownLimits -q`
  Expected: PASS.

  ```bash
  git add examples/limits.rules examples/limits.metadata.json rerum/tests/test_limits.py
  git commit -m "feat(examples): named known-limit identities; algebraic 0/0 path confirmed"
  ```

---

## Task I: Relocated verification in `examples/calculus_checker.py` (`is_integral`, `is_limit` on `numeval`)

**Files:** `examples/calculus_checker.py`, `rerum/tests/test_integration.py`, `rerum/tests/test_limits.py`

Verification is CONTENT. There is NO `rerum/verify.py`. The D1 checker `examples/calculus_checker.py` (which already has `is_derivative`) gains `is_integral` and `is_limit`, both built on the GENERAL `rerum.numeval.numeval` / `rerum.numeval.numeric_equiv`. `is_integral` numerically differentiates `result` (finite difference) and compares to `integrand`; `is_limit` numerically approaches `point` from both sides and checks convergence to `result`. The engine never imports this file.

- [ ] **Step I1: Write failing tests for `is_integral` and `is_limit`.**

  Create `rerum/tests/test_calculus_checker_d2.py` (a separate file so D1's checker tests are untouched):

  ```python
  """Tests for the relocated verification helpers in examples/calculus_checker.py.

  is_integral / is_limit are DOMAIN CONTENT built on the general rerum.numeval
  primitives. The engine never imports calculus_checker.
  """

  from fractions import Fraction

  import pytest

  from examples.calculus_checker import is_integral, is_limit


  class TestIsIntegral:
      def test_power_rule_antiderivative_verifies(self):
          # d/dx[x^3 / 3] = x^2.
          assert is_integral(["^", "x", 2], "x", ["/", ["^", "x", 3], 3]) is True

      def test_fraction_coefficient_form_verifies(self):
          # (* (^ x 3) (/ 1 3)) differentiates to x^2.
          result = ["*", ["^", "x", 3], Fraction(1, 3)]
          assert is_integral(["^", "x", 2], "x", result) is True

      def test_sin_antiderivative_verifies(self):
          assert is_integral(["sin", "x"], "x", ["-", ["cos", "x"]]) is True

      def test_cos_antiderivative_verifies(self):
          assert is_integral(["cos", "x"], "x", ["sin", "x"]) is True

      def test_exp_antiderivative_verifies(self):
          assert is_integral(["exp", "x"], "x", ["exp", "x"]) is True

      def test_usub_antiderivative_verifies(self):
          # d/dx[sin(x^2)] = 2x cos(x^2).
          integrand = ["*", ["cos", ["^", "x", 2]], ["*", 2, "x"]]
          assert is_integral(integrand, "x", ["sin", ["^", "x", 2]]) is True

      def test_by_parts_antiderivative_verifies(self):
          # d/dx[x e^x - e^x] = x e^x.
          integrand = ["*", "x", ["exp", "x"]]
          result = ["-", ["*", "x", ["exp", "x"]], ["exp", "x"]]
          assert is_integral(integrand, "x", result) is True

      def test_wrong_antiderivative_rejected(self):
          # d/dx[x^2] = 2x != x^2.
          assert is_integral(["^", "x", 2], "x", ["^", "x", 2]) is False


  class TestIsLimit:
      def test_sinc_limit_is_one(self):
          assert is_limit(["/", ["sin", "x"], "x"], "x", 0, 1) is True

      def test_versine_limit_is_zero(self):
          assert is_limit(["/", ["-", 1, ["cos", "x"]], "x"], "x", 0, 0) is True

      def test_polynomial_zero_over_zero(self):
          assert is_limit(["/", ["-", ["^", "x", 2], 1], ["-", "x", 1]], "x", 1, 2) is True

      def test_continuous_substitution_limit(self):
          assert is_limit(["+", "x", 1], "x", 2, 3) is True

      def test_wrong_result_rejected(self):
          assert is_limit(["/", ["sin", "x"], "x"], "x", 0, 2) is False

      def test_one_sided_domain_boundary(self):
          # lim_{x->0+} sqrt(x) = 0; the x<0 side is a domain error and must not
          # be treated as a counterexample.
          assert is_limit(["sqrt", "x"], "x", 0, 0) is True
  ```

  Run: `pytest rerum/tests/test_calculus_checker_d2.py -q`
  Expected: FAIL (`ImportError`: `is_integral`/`is_limit` not yet in `examples/calculus_checker.py`).

- [ ] **Step I2: Add `is_integral` and `is_limit` to `examples/calculus_checker.py`.**

  Append to `examples/calculus_checker.py` (the D1 file already imports `numeval`/`numeric_equiv` from `rerum.numeval` and defines `is_derivative`; reuse the same import). Both new helpers are built on the GENERAL `numeval`; they encode the DOMAIN semantics (an antiderivative differentiates back to the integrand; a limit is the two-sided numeric approach):

  ```python
  import math

  from rerum.numeval import numeval  # GENERAL numeric primitive (D1 already imports this).

  # The example checker carries a numeric prelude for numeval. D1 defines one
  # (math + predicate fold funcs); reuse it. If D1 named it _CHECKER_PRELUDE,
  # use that; otherwise build it here from the public bundles.
  try:
      _CHECKER_PRELUDE  # defined by D1
  except NameError:
      from rerum.rewriter import MATH_PRELUDE, PREDICATE_PRELUDE, combine_preludes
      _CHECKER_PRELUDE = combine_preludes(MATH_PRELUDE, PREDICATE_PRELUDE)


  def is_integral(integrand, var, result, *, samples=8, tol=1e-3):
      """True iff d/d(var)[result] matches `integrand` at sampled points.

      Differentiates `result` NUMERICALLY (central finite difference, via the
      general rerum.numeval) and compares to the integrand at the same points.
      This decouples integration verification from any symbolic dd rule set: a
      correct antiderivative is recognized however it was produced. DOMAIN
      content; the engine never imports this.
      """
      h = 1e-5
      usable = 0
      for i in range(samples):
          x0 = 0.3 + 0.37 * i  # deterministic, away from 0 (1/x, ln poles).
          try:
              hi = numeval(result, {var: x0 + h}, _CHECKER_PRELUDE)
              lo = numeval(result, {var: x0 - h}, _CHECKER_PRELUDE)
              lhs = (float(hi) - float(lo)) / (2 * h)
              rhs = float(numeval(integrand, {var: x0}, _CHECKER_PRELUDE))
          except (ValueError, ZeroDivisionError, OverflowError, TypeError):
              continue
          usable += 1
          if not math.isfinite(lhs) or not math.isfinite(rhs):
              return False
          if abs(lhs - rhs) > max(tol, tol * abs(rhs)):
              return False
      return usable > 0


  def is_limit(expr, var, point, result, *,
               eps_seq=(0.1, 0.01, 0.001, 1e-4), tol=1e-3):
      """True iff lim_{var->point} expr == result, by numeric two-sided approach.

      Samples `expr` at var = point +/- eps for shrinking eps (both sides),
      skipping samples that raise (domain error) or are non-finite. Accepts iff
      at least one side is defined arbitrarily close to the point and every
      defined side's closest sample is within tolerance of `result`. An entirely
      undefined side (one-sided limits / domain boundaries) is not a
      counterexample. Built on the general rerum.numeval. DOMAIN content.
      """
      target = float(result)
      any_defined = False
      sides_ok = []
      for sign in (-1.0, +1.0):
          last_err = None
          defined_here = False
          for eps in eps_seq:
              x = float(point) + sign * eps
              try:
                  val = numeval(expr, {var: x}, _CHECKER_PRELUDE)
                  fval = float(val)
              except (ValueError, ZeroDivisionError, OverflowError, TypeError):
                  continue
              if not math.isfinite(fval):
                  continue
              defined_here = True
              any_defined = True
              last_err = abs(fval - target)
          if defined_here:
              sides_ok.append(last_err is not None
                              and last_err <= max(tol, abs(target) * tol))
          else:
              sides_ok.append(True)  # undefined side is not a counterexample
      return any_defined and all(sides_ok)
  ```

  Run: `pytest rerum/tests/test_calculus_checker_d2.py -q`
  Expected: PASS.

- [ ] **Step I3: Wire numeric confirmation into the integration and limit suites.**

  Append to `rerum/tests/test_integration.py`:

  ```python
  from examples.calculus_checker import is_integral


  class TestIntegrationNumericallyVerified:
      @pytest.mark.parametrize("integrand", [
          ["cos", "x"],
          ["sin", "x"],
          ["*", 2, "x"],
          ["+", "x", ["cos", "x"]],
          ["*", ["cos", ["^", "x", 2]], ["*", 2, "x"]],  # u-sub
          ["*", "x", ["exp", "x"]],                       # by-parts
      ])
      def test_solve_result_differentiates_back(self, integrand):
          eng = _integration_engine()
          res = integrate(eng, integrand, "x", max_nodes=5000)
          assert res.found is True
          assert not contains_op(res.solution, {"int"})
          assert is_integral(integrand, "x", res.solution) is True
  ```

  Append to `rerum/tests/test_limits.py`:

  ```python
  from examples.calculus_checker import is_limit


  class TestLimitsEndToEndVerified:
      @pytest.mark.parametrize("body,point,answer", [
          ("(+ x 1)", 2, 3),
          ("(/ (sin x) x)", 0, 1),
          ("(/ (- 1 (cos x)) x)", 0, 0),
          ("(/ (- (^ x 2) 1) (- x 1))", 1, 2),
      ])
      def test_solve_and_is_limit_agree(self, body, point, answer):
          eng = _limits_engine()
          res = _solve_limit(eng, f"(lim {body} x {point})", max_nodes=8000)
          assert res.found is True, f"solve failed to close {body}"
          assert res.solution == answer
          assert is_limit(E(body), "x", point, answer) is True
          # Derivation is reconstructible: replaying step.after reaches the solution.
          current = res.derivation.initial
          for step in res.derivation.steps:
              current = step.after
          assert current == res.solution
  ```

  Run: `pytest rerum/tests/test_integration.py::TestIntegrationNumericallyVerified rerum/tests/test_limits.py::TestLimitsEndToEndVerified -q`
  Expected: PASS (every solved result both closes under `solve` and is confirmed numerically by the content checker).

- [ ] **Step I4: Commit.**

  ```bash
  git add examples/calculus_checker.py rerum/tests/test_calculus_checker_d2.py rerum/tests/test_integration.py rerum/tests/test_limits.py
  git commit -m "feat(examples): is_integral/is_limit checkers on numeval; numeric confirmation tests"
  ```

---

## Task J: Full-suite regression and no-engine-code audit

**Files:** (none new)

- [ ] **Step J1: Run the full test suite.**

  Run: `pytest -q`
  Expected: PASS (no regressions). Adding example files and tests is additive; no `rerum/` module changed.

- [ ] **Step J2: Audit that this phase added NO engine code (the swap test).**

  Run: `git diff --name-only origin/main...HEAD`
  Expected: every changed path is under `examples/` or `rerum/tests/`. NO path under `rerum/` (except `rerum/tests/`) appears. No new `rerum/verify.py`. Confirm `int`/`lim` appear ONLY in `examples/*.rules`:

  Run: `grep -rEn '"int"|"lim"|\bint\b|\blim\b' rerum/ --include='*.py' | grep -v '/tests/' | grep -vi 'print\|point\|lint\|limit_\|splint'`
  Expected: no matches that treat `int`/`lim` as a special-cased operator in engine code. (Incidental substrings like `print`, `point` are filtered; if a genuine engine reference appears, the phase has leaked a domain into core and must be reworked.)

- [ ] **Step J3: Smoke-check the public demonstration path.**

  Run: `python -c "from pathlib import Path; from rerum.engine import RuleEngine; from rerum.rewriter import MATH_PRELUDE, PREDICATE_PRELUDE, combine_preludes; from rerum.solve import solve, contains_op; from examples.calculus_checker import is_integral; eng = RuleEngine().with_prelude(combine_preludes(MATH_PRELUDE, PREDICATE_PRELUDE)); eng.load_file('examples/integration.rules', validate_examples=False); eng.load_metadata_json(Path('examples/integration.metadata.json').read_text()); r = solve(eng, ['int', ['cos','x'],'x'], lambda e: not contains_op(e, {'int'}), max_nodes=2000); print(r.found, r.solution, is_integral(['cos','x'],'x', r.solution))"`
  Expected: prints `True ['sin', 'x'] True`.

- [ ] **Step J4: Commit any final touch-ups (if needed).**

  ```bash
  git add -A
  git commit -m "test(examples): D2 integration+limits suites green; no-engine-code audit clean"
  ```

---

## Done When

- [ ] `examples/integration.rules` exists, uses the `["int", f, v]` operator, and covers: linearity (variadic `+` sum via `?rest...`, difference, constant-multiple left and right), the power rule `int x^n = x^(n+1)/(n+1)` guarded `n != -1` with an EXACT-rational `(! / 1 (! + :n 1))` coefficient, `int x^-1 = ln|x|`, `int 1/x = ln|x|`, `int 1 = x`, `int c = c*x`, `int e^x = e^x`, `int sin = -cos`, `int cos = sin`, the u-substitution collapsing rules (cos/sin/exp of `v^2` times `2v`, both factor orders) plus a fresh-variable u-sub rule exercising `(fresh u)` (or, per the honest note, the concrete rules only with `(fresh u)` exercised by the documented general schema), and the integration-by-parts rule for `x*e^x` (product-shape-guarded), with the general by-parts schema documented as an inactive reference.
- [ ] `examples/limits.rules` exists, uses `["lim", f, v, a]`, and covers: direct substitution (`lim-subst`, guarded by `defined-at?`), L'Hopital (`lim-lhopital`, guarded by `indeterminate?`, rewriting to `(dd ...)` terms that reduce via the D1 differentiation rules in the SAME engine), transcendental constant folding, known limits (`lim-sinc`, `lim-vers`), and (only if the L'Hopital path did not already close it) an algebraic factor/cancel rule.
- [ ] Every closing rule carries `examples` metadata in `examples/integration.metadata.json` / `examples/limits.metadata.json`, merged via `engine.load_metadata_json(..., validate_examples=True)` against the configured prelude; the integration power-rule example asserts the exact `(/ 1 3)` coefficient; structural/decomposing rules are exempt (`category=structural`/`by-parts`); each limit example `out` is the single-rule-application result (L'Hopital's `out` is the transformed `(lim (/ (dd f v) (dd g v)) v a)`, NOT the final number).
- [ ] Easy cases reduce under `simplify`/`apply_once` (the power rule fires once with an exact `Fraction`); non-confluent cases escalate to `solve(engine, ["int"/"lim", ...], goal=lambda e: not contains_op(e, {"int"/"lim"}), max_nodes=...)`. The test helpers `integrate(eng, integrand, var, *, max_nodes=...)` and `limit`/`_solve_limit(eng, sexpr, max_nodes=...)` wire `solve` with the caller-supplied goal and return the operator-free `SolveResult.solution` plus the `RewriteTrace` derivation.
- [ ] Worked integrals close AND `is_integral` confirms numerically: `int(2x)` (int-free), `int(cos x) -> sin x`, `int(sin x) -> -cos x`, `int(2x*cos(x^2)) -> sin(x^2)` via u-sub, `int(x*e^x)` (int-free) via by-parts.
- [ ] Worked limits close AND `is_limit` confirms numerically: `lim_{x->2}(x+1)=3`, `lim_{x->0} sin(x)/x=1` (via L'Hopital and/or the named `lim-sinc`), `lim_{x->0}(1-cos x)/x=0`, `lim_{x->1}(x^2-1)/(x-1)=2`. Derivations are reconstructible (replaying `step.after` from `initial` reaches the solution) and name the rules that fired.
- [ ] Honest budgeted failure: a too-small `max_nodes` yields `SolveResult(found=False, solution=None)` with `explored` within budget (asserted), never a partial or wrong result.
- [ ] Verification lives in `examples/calculus_checker.py` (NOT a core `rerum/verify.py`): `is_integral(integrand, var, result, ...)` and `is_limit(expr, var, point, result, ...)` are built on the GENERAL `rerum.numeval`, encode the domain semantics, and are never imported by the engine.
- [ ] This phase added NO engine code: `git diff --name-only` shows only `examples/` and `rerum/tests/` paths; `int`/`lim` appear only in `examples/*.rules`; the swap test passes by inspection.
- [ ] `pytest -q` passes with no regressions; `test_integration.py`, `test_limits.py`, and `test_calculus_checker_d2.py` all pass.
