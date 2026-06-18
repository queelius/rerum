# C1: General By-Parts Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether general integration-by-parts can be made robust with cheap content-level mechanisms (pattern-restriction, theory-normalization, cost-shaping) and decide whether C3 (AND/OR subgoal search) is needed.

**Architecture:** An `experiments/byparts_search.py` harness builds engine variants (each a "mechanism": which by-parts rules, whether the theory is threaded, which cost function) and measures `solve` on a battery of integrals (found / nodes-explored / wall-time). The harness is a runnable script (the `experiments/` convention), NOT pytest. One small pytest file pins the deterministic BASELINE premise (raw schema is fragile) so the experiment's motivation stays regression-protected. If a mechanism clears the success bar, its tamed rule ships as content in `examples/integration.rules` with end-to-end + numeric-verification tests.

**Tech Stack:** Python 3.9+, `rerum.solve.solve`, `rerum.optimize.make_op_cost_fn`, `rerum.normalize.Theory`, the existing `examples/integration.rules` + `examples/differentiation.rules` + `examples/arithmetic.theory.json`. Spec: `docs/superpowers/specs/2026-06-18-c1-byparts-experiment-design.md`.

**Mechanism reference (used throughout):**
- General by-parts schema (baseline): `(int (* ?u ?dv) ?v) => (- (* :u (int :dv :v)) (int (* (int :dv :v) (dd :u :v)) :v))`
- Pattern-restricted by-parts, two rules:
  - `(int (* ?v ?g) ?v) => (- (* :v (int :g :v)) (int (* (int :g :v) (dd :v :v)) :v))`
  - `(int (* (^ ?v ?n:const) ?g) ?v) => (- (* (^ :v :n) (int :g :v)) (int (* (int :g :v) (dd (^ :v :n) :v)) :v))`
- **Negation linearity** (the prime suspect from analysis): `(int (- ?f) ?v) => (- (int :f :v))`. A correct, fully general linearity rule that no domain knowledge -- it is missing from the current table, which is WHY `int(x*sin x)` explodes while `int(x*cos x)` closes: by-parts on `int(x*sin x)` produces the sub-integral `int(-cos x)` (a unary-negated function) that nothing in the table reduces, so the branch strands; `int(x*cos x)`'s sub-integral is `int(sin x)`, which closes via `int-sin`. Adding this rule is the cheapest candidate for making by-parts robust and MUST be measured.

(Note: in the pattern-restricted n=1 rule, `(dd :v :v)` reduces to `1` via the differentiation rules co-loaded in the same engine, collapsing the second integral's `u' = 1` factor. In a SKELETON, substitution uses `:g`/`:v`, never `?g`/`?v` -- a `?`-form in a skeleton is a literal pattern node, not a substitution.)

---

## File Structure

- `experiments/byparts_search.py` (NEW) -- the measurement harness + matrix + printed findings/decision. Runnable script, not pytest.
- `rerum/tests/test_byparts_experiment.py` (NEW) -- pins the deterministic baseline premise (raw general schema is fragile) so the experiment's starting point is regression-protected.
- `examples/integration.rules` (MODIFY, CONDITIONAL) -- ship the winning tamed by-parts rule(s) only if a mechanism clears the bar.
- `examples/integration.metadata.json` (MODIFY, CONDITIONAL) -- sidecar example(s) for the shipped rule(s).
- `rerum/tests/test_integration.py` (MODIFY, CONDITIONAL) -- end-to-end + `is_integral` tests for the shipped rule(s).

---

### Task 1: The measurement harness

**Files:**
- Create: `experiments/byparts_search.py`

- [ ] **Step 1: Write the harness script.**

Create `experiments/byparts_search.py`:

```python
"""C1 experiment: can general integration-by-parts be made ROBUST cheaply?

Runnable measurement harness (NOT pytest -- the experiments/ convention).
For each MECHANISM it runs `solve` over a battery of integrals and reports
found / CORRECT (is_integral) / nodes-explored / wall-time.

CRITICAL: "found" (int-free) is NOT success -- CORRECTNESS is. By-parts
spawns sub-integrals; a cost that rewards int-elimination makes best-first
find the SHORTEST int-free path, which can be WRONG (e.g. int(x*sin x)
"closes" to -x*cos x, missing +sin x; is_integral=False). So every cell
reports whether the found solution differentiates back to the integrand.
The printed findings record the C3 decision: which cases a cheap mechanism
closes CORRECTLY, which remain frontier.

Run from the repo root:  python experiments/byparts_search.py
"""
from __future__ import annotations

import importlib.util
import time
from pathlib import Path

from rerum.engine import RuleEngine, format_sexpr
from rerum.rewriter import MATH_PRELUDE, PREDICATE_PRELUDE, combine_preludes
from rerum.normalize import Theory
from rerum.solve import solve, contains_op
from rerum.optimize import make_op_cost_fn

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
THEORY = Theory.from_json((EXAMPLES / "arithmetic.theory.json").read_text())


def _load_checker():
    """examples/calculus_checker.py is example content (imported by path)."""
    spec = importlib.util.spec_from_file_location(
        "calculus_checker", EXAMPLES / "calculus_checker.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CHECKER = _load_checker()  # provides is_integral(integrand, var, result)

# By-parts rule sets per mechanism (DSL strings loaded into the engine).
BYPARTS_GENERAL = (
    "@bp-gen: (int (* ?u ?dv) ?v) => "
    "(- (* :u (int :dv :v)) (int (* (int :dv :v) (dd :u :v)) :v))"
)
BYPARTS_PATTERN = (
    "@bp-v: (int (* ?v ?g) ?v) => "
    "(- (* :v (int :g :v)) (int (* (int :g :v) (dd :v :v)) :v))\n"
    "@bp-pow: (int (* (^ ?v ?n:const) ?g) ?v) => "
    "(- (* (^ :v :n) (int :g :v)) "
    "(int (* (int :g :v) (dd (^ :v :n) :v)) :v))"
)
# Negation linearity -- the prime suspect: by-parts produces int(-cos x)
# sub-integrals that nothing reduces, stranding the branch. This closes them.
INT_NEG = "@int-neg: (int (- ?f) ?v) => (- (int :f :v))"

# The case battery: name -> integrand (w.r.t. x).
CASES = {
    "int(x*e^x)":       ["*", "x", ["exp", "x"]],          # concrete control
    "int(x*cos x)":     ["*", "x", ["cos", "x"]],
    "int(x*sin x)":     ["*", "x", ["sin", "x"]],
    "int(x^2*e^x)":     ["*", ["^", "x", 2], ["exp", "x"]],
    "int(x*ln x)":      ["*", "x", ["ln", "x"]],
    "int(e^x*sin x)":   ["*", ["exp", "x"], ["sin", "x"]],  # boomerang
}


def build_engine(byparts_dsl: str) -> RuleEngine:
    """Integration + differentiation rules + the chosen by-parts rule set."""
    eng = RuleEngine().with_prelude(
        combine_preludes(MATH_PRELUDE, PREDICATE_PRELUDE))
    eng.load_file(EXAMPLES / "integration.rules", validate_examples=False)
    eng.load_file(EXAMPLES / "differentiation.rules", validate_examples=False)
    if byparts_dsl:
        eng.load_dsl(byparts_dsl, validate_examples=False)
    return eng


def measure(byparts_dsl, integrand, *, use_theory, cost, budget,
            verified_goal=False):
    """Return (found, correct, explored, seconds, r) for one cell.

    correct = is_integral(integrand, x, solution) when found, else None.
    verified_goal=True puts is_integral INTO the goal predicate, so the
    search rejects wrong int-free nodes (a mechanism under test)."""
    eng = build_engine(byparts_dsl)
    kw = {}
    if use_theory:
        kw.update(theory=THEORY, normalize_between=True)
    if cost is not None:
        kw["cost_fn"] = cost
    if verified_goal:
        goal = lambda e: (not contains_op(e, {"int"})
                          and CHECKER.is_integral(integrand, "x", e))
    else:
        goal = lambda e: not contains_op(e, {"int"})
    t = time.time()
    r = solve(eng, ["int", integrand, "x"], goal, max_nodes=budget, **kw)
    correct = (CHECKER.is_integral(integrand, "x", r.solution)
               if r.found else None)
    return r.found, correct, r.explored, time.time() - t, r


def run_mechanism(name, byparts_dsl, *, use_theory, cost, budget=500,
                  verified_goal=False):
    """Run the full battery under one mechanism; print a row per case.
    Returns the set of cases closed CORRECTLY (found AND is_integral)."""
    print(f"\n=== {name} (theory={use_theory}, verified_goal={verified_goal}, "
          f"budget={budget}) ===")
    closed_correct = []
    for case, integrand in CASES.items():
        found, correct, explored, secs, _ = measure(
            byparts_dsl, integrand, use_theory=use_theory, cost=cost,
            budget=budget, verified_goal=verified_goal)
        flag = "CORRECT" if correct else ("WRONG" if found else "-")
        print(f"  {case:16s} found={found!s:5s} {flag:7s} "
              f"explored={explored:5d} {secs:6.2f}s")
        if found and correct:
            closed_correct.append(case)
    return closed_correct


COST_INT_HIGH = make_op_cost_fn({"int": 50.0})


def main():
    results = {}
    results["baseline (general schema)"] = run_mechanism(
        "baseline (general schema)", BYPARTS_GENERAL,
        use_theory=False, cost=COST_INT_HIGH)
    results["pattern-restricted"] = run_mechanism(
        "pattern-restricted", BYPARTS_PATTERN,
        use_theory=False, cost=COST_INT_HIGH)
    results["pattern + theory"] = run_mechanism(
        "pattern + theory", BYPARTS_PATTERN,
        use_theory=True, cost=COST_INT_HIGH)
    # The prime-suspect mechanism: pattern by-parts + the negation linearity
    # rule that closes the int(-cos x) sub-integrals. Also test the GENERAL
    # schema + int-neg, to separate "by-parts needs restriction" from
    # "by-parts just needed a missing linearity rule".
    results["pattern + theory + int-neg"] = run_mechanism(
        "pattern + theory + int-neg", BYPARTS_PATTERN + "\n" + INT_NEG,
        use_theory=True, cost=COST_INT_HIGH)
    results["general + theory + int-neg"] = run_mechanism(
        "general + theory + int-neg", BYPARTS_GENERAL + "\n" + INT_NEG,
        use_theory=True, cost=COST_INT_HIGH)
    # CORRECTNESS-AWARE GOAL: is_integral is part of the goal, so the search
    # cannot stop at a fast-wrong int-free node. Tests whether soundness is
    # restored AND whether it stays cheap. (is_integral is numeric sampling,
    # so this is more expensive per node -- watch the time column.)
    results["pattern + int-neg + verified-goal"] = run_mechanism(
        "pattern + int-neg + verified-goal", BYPARTS_PATTERN + "\n" + INT_NEG,
        use_theory=True, cost=COST_INT_HIGH, verified_goal=True)
    results["general + int-neg + verified-goal"] = run_mechanism(
        "general + int-neg + verified-goal", BYPARTS_GENERAL + "\n" + INT_NEG,
        use_theory=True, cost=COST_INT_HIGH, verified_goal=True)
    print("\n=== DECISION ===")
    classic = ["int(x*cos x)", "int(x*sin x)", "int(x^2*e^x)", "int(x*ln x)"]
    for mech, closed in results.items():
        got = [c for c in classic if c in closed]
        print(f"  {mech:34s} closed CORRECTLY: {got}")
    print("  boomerang int(e^x*sin x): expected frontier (a wrong int-free "
          "'solution' under a plain goal; needs a verified goal or C3)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the harness to confirm it executes and produces a table.**

Run: `python experiments/byparts_search.py`
Expected: it runs to completion (budget=500 keeps each cell bounded) and prints seven mechanism tables (each row showing found + CORRECT/WRONG + explored + time) + a DECISION block. Do NOT interpret results yet -- Task 2+ analyze them. If any cell hangs, the budget guard failed; lower `budget` to 300 and note it. (The verified-goal mechanisms are slower per node because `is_integral` samples numerically -- expect higher times there.)

- [ ] **Step 3: Commit.**

```bash
git add experiments/byparts_search.py
git commit -m "experiment(c1): by-parts search measurement harness

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Pin the baseline premise (deterministic)

**Files:**
- Create: `rerum/tests/test_byparts_experiment.py`

- [ ] **Step 1: Write the baseline-fragility test.**

The experiment's whole premise is "raw general schema + op_costs is fragile." That specific fact is deterministic and worth protecting from regression (if a future engine change made the raw schema robust, this experiment's motivation would change and we'd want to know).

Create `rerum/tests/test_byparts_experiment.py`:

```python
"""Pins the deterministic premise of the C1 by-parts experiment: the raw
general by-parts schema + op_costs is FRAGILE (closes some classic cases,
explodes on others within a tight budget). The full matrix lives in
experiments/byparts_search.py (a script, not pytest)."""

import importlib.util
from pathlib import Path

import pytest

HARNESS = (Path(__file__).resolve().parents[2]
           / "experiments" / "byparts_search.py")


def _harness():
    spec = importlib.util.spec_from_file_location("byparts_search", HARNESS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestBaselineFragility:
    def test_raw_schema_closes_x_cos_x_correctly(self):
        h = _harness()
        found, correct, _explored, _secs, _r = h.measure(
            h.BYPARTS_GENERAL, h.CASES["int(x*cos x)"],
            use_theory=False, cost=h.COST_INT_HIGH, budget=500)
        assert found is True and correct is True  # the lucky case is correct

    def test_raw_schema_does_not_close_x_sin_x(self):
        h = _harness()
        found, _correct, _explored, _secs, _r = h.measure(
            h.BYPARTS_GENERAL, h.CASES["int(x*sin x)"],
            use_theory=False, cost=h.COST_INT_HIGH, budget=500)
        # Fragile: the structurally-identical sin case does NOT close (its
        # int(-cos x) sub-integral strands without the int-neg rule).
        assert found is False

    def test_op_costs_goal_is_UNSOUND_finds_fast_wrong_answer(self):
        # THE central finding: with int-neg added, int(x*sin x) "closes"
        # FAST under the plain int-free goal -- to a WRONG answer (the
        # search prefers the shortest int-free path, which spuriously zeroes
        # a sub-integral). found=True but is_integral=False. This is why the
        # success bar requires CORRECTNESS, not int-freeness.
        h = _harness()
        found, correct, _explored, _secs, _r = h.measure(
            h.BYPARTS_GENERAL + "\n" + h.INT_NEG, h.CASES["int(x*sin x)"],
            use_theory=True, cost=h.COST_INT_HIGH, budget=500)
        assert found is True
        assert correct is False  # FAST but WRONG -- the unsoundness
```

- [ ] **Step 2: Run the test, expect PASS (it pins current behavior).**

Run: `python -m pytest rerum/tests/test_byparts_experiment.py -v`
Expected: all THREE PASS -- the cos case closes correctly, the sin case does not close under the bare schema, and (the central finding) the schema+int-neg "closes" sin FAST but WRONG under the plain goal. If `test_op_costs_goal_is_UNSOUND_finds_fast_wrong_answer` does NOT see found=True/correct=False (e.g. the search now finds the correct answer, or finds nothing), the unsoundness premise has shifted -- record it, because it changes the whole experiment's conclusion and the C3 decision.

- [ ] **Step 3: Commit.**

```bash
git add rerum/tests/test_byparts_experiment.py
git commit -m "test(c1): pin the baseline by-parts fragility premise

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Measure pattern-restriction (the primary refinement) and record findings

**Files:**
- Modify: `experiments/byparts_search.py` (the `main()` already runs it; this task ANALYZES and records the result into the module docstring)

- [ ] **Step 1: Run the harness and read ALL mechanism tables.**

Run: `python experiments/byparts_search.py`
Read every mechanism table -- baseline, pattern-restricted, pattern+theory, and especially the two int-neg mechanisms (`pattern + theory + int-neg`, `general + theory + int-neg`). For each classic case (`int(x*cos x)`, `int(x*sin x)`, `int(x^2*e^x)`, `int(x*ln x)`), note found + explored. The success bar (spec): a mechanism WORKS if it closes the classic battery within `budget=500`. PRIME SUSPECT: the int-neg mechanisms, since the analysis shows the sin/cos asymmetry is the missing negation-linearity rule, not the by-parts schema itself. If `general + theory + int-neg` closes the battery, the SIMPLER general schema + one linearity rule suffices (no pattern-restriction needed) -- record that, it is the cleanest content to ship.

- [ ] **Step 2: Record the measured findings into the harness module docstring.**

Edit the top docstring of `experiments/byparts_search.py` to append a "## Findings (measured YYYY-MM-DD)" section stating, per mechanism, which classic cases closed and the node counts, e.g.:

```
## Findings (measured 2026-06-18)
- baseline (general schema):   closed {int(x*cos x)}; the rest exploded.
- pattern-restricted:          closed {...}; explored counts {...}.
- pattern + theory:            closed {...}; explored counts {...}.
- boomerang int(e^x*sin x):    not closed under any mechanism (frontier).
```

(Fill in the ACTUAL measured sets and counts -- this is the experiment's data, not a guess.)

- [ ] **Step 3: Commit.**

```bash
git add experiments/byparts_search.py
git commit -m "experiment(c1): record pattern-restriction + theory measurements

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Cost-shaping + combinations; characterize the boomerang

**Files:**
- Modify: `experiments/byparts_search.py`

- [ ] **Step 1: Add a size-aware cost mechanism and a boomerang-at-larger-budget probe.**

Append to `experiments/byparts_search.py`, and call them from `main()` after the existing mechanisms:

```python
from rerum.optimize import expr_size


def cost_int_and_size(expr):
    """Price int nodes high AND penalize raw expression size, so best-first
    prefers smaller int-free descendants over large ones."""
    return 50.0 * _count_int(expr) + expr_size(expr)


def _count_int(expr):
    if isinstance(expr, list):
        n = 1 if (expr and expr[0] == "int") else 0
        return n + sum(_count_int(e) for e in expr)
    return 0
```

In `main()`, after the existing `run_mechanism` calls, add:

```python
    results["pattern + theory + size-cost"] = run_mechanism(
        "pattern + theory + size-cost", BYPARTS_PATTERN,
        use_theory=True, cost=cost_int_and_size)
    # Boomerang at a larger budget: confirm it is genuinely out of reach
    # CORRECTLY, not merely under-budgeted (under a plain goal it may "find"
    # a fast WRONG answer; under a verified goal it should find nothing).
    print("\n=== boomerang at budget=3000 ===")
    for label, vg in (("plain goal", False), ("verified goal", True)):
        found, correct, explored, secs, _ = measure(
            BYPARTS_GENERAL + "\n" + INT_NEG, CASES["int(e^x*sin x)"],
            use_theory=True, cost=cost_int_and_size, budget=3000,
            verified_goal=vg)
        flag = "CORRECT" if correct else ("WRONG" if found else "-")
        print(f"  int(e^x*sin x) [{label:13s}] found={found} {flag} "
              f"explored={explored} {secs:.2f}s")
```

- [ ] **Step 2: Run the harness, read the new rows.**

Run: `python experiments/byparts_search.py`
Expected: the size-cost mechanism's table prints; the boomerang row at budget=3000 prints. Record whether size-cost closes any case the pure int-cost did not, and confirm the boomerang stays `found=False` even at 3000 nodes (the spec's expectation -- it needs the algebraic `I = A - I` step that pure rewriting cannot do).

- [ ] **Step 3: Append the cost-shaping + boomerang findings to the module docstring.** Add their measured results to the "## Findings" section.

- [ ] **Step 4: Commit.**

```bash
git add experiments/byparts_search.py
git commit -m "experiment(c1): cost-shaping mechanism + boomerang characterization

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Write the C3 decision

**Files:**
- Modify: `experiments/byparts_search.py` (docstring) and the C1 spec's status

- [ ] **Step 1: Write the explicit C3 decision into the harness docstring.**

Append a "## C3 decision" section to the module docstring stating, grounded in the measured findings:
- which classic cases a CHEAP mechanism (name it) closes within budget -> these are shippable as content (Task 6);
- which cases remain frontier (at minimum the boomerang family) -> these are exactly what C3 would need to cover;
- the verdict: C3 is NEEDED-for-boomerang / AVOIDABLE-for-the-classic-battery (or, if no cheap mechanism worked, C3 is needed for general by-parts entirely).

Example shape (fill with real data):

```
## C3 decision (2026-06-18)
- Cheap mechanism "pattern + theory" closes the classic battery
  {int(x*cos x), int(x*sin x), int(x^2*e^x), int(x*ln x)} within <N> nodes.
  -> SHIP as content (Task 6).
- The boomerang family (int(e^x*sin x)) stays unclosed at 3000 nodes; it
  needs the algebraic I = A - I step, which is beyond pure rewriting.
  -> C3 (AND/OR subgoal search) is needed ONLY for the boomerang family,
     not for the textbook polynomial-times-transcendental battery.
```

- [ ] **Step 2: Update the C1 spec status line.**

Edit `docs/superpowers/specs/2026-06-18-c1-byparts-experiment-design.md` line 3 (`**Status:**`) to append `; EXECUTED <date>: <one-line verdict>` so the spec records the outcome.

- [ ] **Step 3: Commit.**

```bash
git add experiments/byparts_search.py docs/superpowers/specs/2026-06-18-c1-byparts-experiment-design.md
git commit -m "experiment(c1): record the C3 decision from the measured findings

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6 (CONDITIONAL): Ship the tamed by-parts rule as content

**Run this task ONLY IF a cheap mechanism closes the classic battery CORRECTLY (is_integral, not merely int-free) within budget=500.** If no cheap mechanism produced correct answers, SKIP this task; the deliverable is the findings + decision, and general by-parts waits for C3.

**Two shipping shapes, by which mechanism won:**
- If a PLAIN-GOAL mechanism (e.g. pattern + int-neg, no verified goal) closes the battery correctly: ship the rule content (below); the normal int-free goal suffices at runtime. This is the clean win.
- If ONLY the VERIFIED-GOAL mechanism closes correctly (the plain goal finds fast-wrong answers): the fix is a SOLVER RECIPE, not just rule content. Ship the rule(s) AND expose a verified-goal integrate helper, e.g. in `rerum/tests/test_integration.py`:
  ```python
  def integrate_verified(eng, integrand, var, *, max_nodes=500):
      from rerum.normalize import Theory
      checker = _load_checker()
      theory = Theory.from_json((EXAMPLES_DIR / "arithmetic.theory.json").read_text())
      goal = lambda e: (not contains_op(e, {"int"})
                        and checker.is_integral(integrand, var, e))
      return solve(eng, ["int", integrand, var], goal, max_nodes=max_nodes,
                   theory=theory, normalize_between=True,
                   cost_fn=make_op_cost_fn({"int": 50.0}))
  ```
  and write the e2e tests against `integrate_verified`. Document in the rule comment that general by-parts requires the verified goal because the plain int-free goal is unsound for it. (This is itself a valuable finding: by-parts is searchable cheaply ONLY with a correctness-checking goal.)

**Files:**
- Modify: `examples/integration.rules`
- Modify: `examples/integration.metadata.json`
- Modify: `rerum/tests/test_integration.py`

- [ ] **Step 1: Write the failing end-to-end test for a classic case the CURRENT rules do not close.**

Append to `rerum/tests/test_integration.py` (the `integrate()` helper threads the theory + `is_integral` already imported via `_load_checker`):

```python
class TestGeneralByParts:
    def test_x_sin_x_closes_and_verifies(self):
        eng = _integration_engine()
        checker = _load_checker()
        out = integrate(eng, ["*", "x", ["sin", "x"]], "x", max_nodes=500)
        assert out.found is True
        assert checker.is_integral(["*", "x", ["sin", "x"]], "x",
                                   out.solution) is True

    def test_x_ln_x_closes_and_verifies(self):
        eng = _integration_engine()
        checker = _load_checker()
        out = integrate(eng, ["*", "x", ["ln", "x"]], "x", max_nodes=500)
        assert out.found is True
        assert checker.is_integral(["*", "x", ["ln", "x"]], "x",
                                   out.solution) is True
```

- [ ] **Step 2: Run the test, expect FAIL.**

Run: `python -m pytest rerum/tests/test_integration.py::TestGeneralByParts -v`
Expected: FAIL (`found is False`) -- the current integration rules lack general by-parts, so `int(x sin x)` does not close in 500 nodes.

- [ ] **Step 3: Add the winning rule set to `examples/integration.rules`.**

Ship whatever the experiment found robust. IMPORTANT: if the winning mechanism included `INT_NEG`, ship the negation-linearity rule TOO (it is a correct general linearity rule and likely the actual fix). If `general + theory + int-neg` won, ship the SIMPLER general schema + int-neg (no pattern-restriction). The produced `(dd ...)` sub-term needs the differentiation rules co-loaded (Step 4). Add to the `[by-parts]`/`[linearity]` groups, e.g. for the pattern + int-neg winner:

```
# Negation linearity: int(-f) = -int(f). General; closes the int(-cos x)
# sub-integrals that by-parts produces. (Add to the [linearity] group.)
@int-neg {category=structural}: (int (- ?f) ?v) => (- (int :f :v))

# General by-parts, polynomial factor (the C1 experiment found this robust
# under theory-normalized solve WITH int-neg). Requires the differentiation
# rules co-loaded so the produced (dd ...) reduces. (Add to [by-parts].)
@int-byparts-v {category=by-parts}: (int (* ?v ?g) ?v) => (- (* :v (int :g :v)) (int (* (int :g :v) (dd :v :v)) :v))
@int-byparts-pow {category=by-parts}: (int (* (^ ?v ?n:const) ?g) ?v) => (- (* (^ :v :n) (int :g :v)) (int (* (int :g :v) (dd (^ :v :n) :v)) :v))
```

If the GENERAL schema + int-neg won, ship `@int-byparts-gen: (int (* ?u ?dv) ?v) => (- (* :u (int :dv :v)) (int (* (int :dv :v) (dd :u :v)) :v))` plus `@int-neg` instead of the two pattern rules.

- [ ] **Step 4: Co-load the differentiation rules in the integration test engine.**

Modify `_integration_engine()` in `rerum/tests/test_integration.py` to add, after the `integration.rules` load:

```python
    eng.load_file(EXAMPLES_DIR / "differentiation.rules",
                  validate_examples=False)
```

- [ ] **Step 5: Add sidecar examples for the new rules to `examples/integration.metadata.json`.**

Add (single-step rewrites; the inner `dd`/`int` stay unreduced at the example level), inside the JSON object:

```json
  "int-byparts-v": {
    "category": "by-parts",
    "examples": [{"in": "(int (* x (sin x)) x)", "out": "(- (* x (int (sin x) x)) (int (* (int (sin x) x) (dd x x)) x))"}]
  },
  "int-byparts-pow": {
    "category": "by-parts",
    "examples": [{"in": "(int (* (^ x 2) (exp x)) x)", "out": "(- (* (^ x 2) (int (exp x) x)) (int (* (int (exp x) x) (dd (^ x 2) x)) x))"}]
  }
```

(Confirm each `out` against the actual single-step `apply_once` output before committing; the byparts rules are `category=by-parts`, exempt from the closed-form requirement but carrying an example anyway.)

- [ ] **Step 6: Run the new tests + the full integration suite.**

Run: `python -m pytest rerum/tests/test_integration.py -v`
Expected: `TestGeneralByParts` passes AND every pre-existing integration test still passes (no regression -- the new rules must not slow the other cases past their budgets or change their solutions). If a pre-existing case now fails/slows, the new rule is over-firing; tighten its pattern or REVERT this task and record that the cheap mechanism is not regression-safe (an important finding -- it means C3 is needed after all).

- [ ] **Step 7: Update the every-rule-fires coverage map.**

`test_cases_cover_every_named_rule` asserts `all_names == set(CASES)` exactly, so add a `CASES` entry for EVERY rule you shipped in Step 3 (the test fails otherwise). For the pattern + int-neg winner that is three rules:

```python
        "int-byparts-v": ["*", "x", ["sin", "x"]],
        "int-byparts-pow": ["*", ["^", "x", 2], ["exp", "x"]],
        "int-neg": ["-", ["sin", "x"]],   # (int (- (sin x)) x) fires int-neg
```

If a different rule set shipped (e.g. `int-byparts-gen` instead of the two pattern rules), use those names. Run `python -m pytest rerum/tests/test_integration.py::TestEveryRuleFiresEndToEnd -q` and confirm `test_cases_cover_every_named_rule` passes (the set matches exactly).

- [ ] **Step 8: Run the full suite + commit.**

Run: `python -m pytest -q`
Expected: all green.

```bash
git add examples/integration.rules examples/integration.metadata.json rerum/tests/test_integration.py
git commit -m "feat(examples): general by-parts rule (C1 found it robust under theory-normalized solve)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Mechanism 4 (poly-in? predicate) -- conditional follow-on, NOT a task here

The spec lists a fallback mechanism 4: a general structural predicate
`poly-in?(u, v)` gating the unrestricted `(* ?u ?dv)` schema. It is NOT
planned as a task because it is doubly-conditional: pursue it ONLY if Task 3
shows pattern-restriction (mechanism 1) is too narrow in a way that matters
(e.g. it closes monomials `x^k * g` but the experiment shows a real need for
non-monomial polynomial factors like `(x+1) * e^x`, and the regression in
Task 6 traces to over-firing rather than to the polynomial shape). If that
specific situation arises, mechanism 4 gets its OWN brainstorm -> spec ->
plan cycle (it adds engine code -- a `PREDICATE_PRELUDE` predicate that must
name no domain and keep `test_mcp_no_domain` green). Record the trigger in
the findings if it occurs; do not build it speculatively.

## Done When

- `experiments/byparts_search.py` runs to completion and its module docstring records the measured findings and an explicit C3 decision.
- `rerum/tests/test_byparts_experiment.py` pins the baseline fragility premise.
- The C1 spec status line records the executed verdict.
- IF a cheap mechanism cleared the bar: the tamed by-parts rule ships in `examples/integration.rules` with sidecar + e2e + `is_integral` tests, every-rule coverage updated, full suite green. ELSE: no content ships and the decision records that C3 is required for general by-parts.
- The boomerang `int(e^x*sin x)` is characterized as out of reach under every cheap mechanism (the precise family C3 would target).
