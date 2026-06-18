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

## Findings (measured 2026-06-18, budget=500 unless noted)

NOTE: the int-neg linearity rule (int(-f) = -int(f)) was SHIPPED into
examples/integration.rules as a result of this experiment, so it is now in
the BASE table that build_engine loads. The numbers below reflect that.

Classic polynomial x transcendental cases closed CORRECTLY (found AND
is_integral):

- baseline (general schema, NO theory, plain goal):  {x*cos x, x*sin x}
  (7-9 nodes). With int-neg in the table the int(-cos x) sub-integrals now
  close, so the plain no-theory goal finds the CORRECT antiderivatives for
  the single-by-parts cases. int(x^2*e^x) (parts twice) is not reached in
  500 nodes; int(x*ln x) is not reached (u-choice).
- pattern-restricted (NO theory, plain goal):        {x*cos x, x*sin x}.
- pattern/general + THEORY + int-neg (plain goal):   {} -- and WRONG, fast.
  THIS is the unsoundness: theory-normalization exposes a path where a
  sub-integral spuriously collapses, and the int-eliminating cost steers
  best-first to that SHORTEST (wrong) int-free node (~4-6 nodes). So the
  bug is specifically PLAIN-GOAL + THEORY-NORMALIZATION.
- pattern + int-neg + VERIFIED goal:                 {x*cos x, x*sin x}
  (9-10 nodes). bp-pow does not reach int(x^2*e^x) in 500 nodes.
- general + int-neg + VERIFIED goal:                 {x*cos x, x*sin x,
  x^2*e^x} -- the WINNER for completeness. 9 / 10 / 67 nodes, all CORRECT.
  The verified goal is robust WITH the theory on (it rejects the wrong
  int-free nodes the theory exposes) and reaches the parts-twice case.
- size-cost did not improve on the int-cost under the verified goal.

Boomerang int(e^x*sin x):
- plain goal + theory:  found in ~6 nodes -- WRONG (a false int-free answer).
- verified goal:        found=False at budget=3000 (~35s). Correctly NOT
  closed: it needs the algebraic I = A - I step, beyond pure rewriting.

## C3 decision (2026-06-18)

- The plain "no int remains" goal is UNSOUND for by-parts UNDER
  theory-normalization: cheap cost-steered search finds FAST WRONG answers.
  Correctness must be IN the goal (or the theory off, but then the search
  is incomplete on parts-twice cases).
- With a VERIFIED goal (is_integral in the goal predicate), the general
  by-parts schema + the int-neg linearity rule closes the textbook
  polynomial x transcendental battery {x*cos x, x*sin x, x^2*e^x} CORRECTLY
  and cheaply (<= 67 nodes). SHIPPED from this experiment: int-neg (a safe,
  general table improvement). NOT shipped into the default rule set: the
  general by-parts schema -- it over-fires on every product and is only
  sound under the verified-goal RECIPE, so it stays a recipe-gated
  capability (demonstrated in test_byparts_experiment), not a default rule.
- Two cases remain FRONTIER, pinning what a richer mechanism would need:
    * int(x*ln x): needs the u/dv CHOICE (LIATE: log should be u). A
      u-selection mechanism (or trying both splits) would address it --
      cheaper than full AND/OR search; a candidate for a small follow-on.
    * int(e^x*sin x) [boomerang]: needs the algebraic I = A - I resolution,
      genuinely beyond pure rewriting. This is the precise, named situation
      that justifies C3 (search-introduces-subgoals + an algebraic closer).
- VERDICT: C3 is AVOIDABLE for the textbook by-parts battery (a verified
  goal + one linearity rule suffices). C3 is NEEDED only for the boomerang
  family; int(x*ln x) wants a smaller u-selection follow-on, not C3.
"""
from __future__ import annotations

import importlib.util
import time
from pathlib import Path

from rerum.engine import RuleEngine, format_sexpr
from rerum.rewriter import MATH_PRELUDE, PREDICATE_PRELUDE, combine_preludes
from rerum.normalize import Theory
from rerum.solve import solve, contains_op
from rerum.optimize import make_op_cost_fn, expr_size

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

COST_INT_HIGH = make_op_cost_fn({"int": 50.0})


def cost_int_and_size(expr):
    """Price int nodes high AND penalize raw expression size, so best-first
    prefers smaller int-free descendants over large ones."""
    return 50.0 * _count_int(expr) + expr_size(expr)


def _count_int(expr):
    if isinstance(expr, list):
        n = 1 if (expr and expr[0] == "int") else 0
        return n + sum(_count_int(e) for e in expr)
    return 0


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
    # Cost-shaping complement on the best rule set.
    results["pattern + int-neg + verified + size-cost"] = run_mechanism(
        "pattern + int-neg + verified + size-cost",
        BYPARTS_PATTERN + "\n" + INT_NEG,
        use_theory=True, cost=cost_int_and_size, verified_goal=True)

    # Boomerang at a larger budget: confirm it is genuinely out of reach
    # CORRECTLY, not merely under-budgeted (under a plain goal it may "find"
    # a fast WRONG answer; under a verified goal it should find nothing).
    print("\n=== boomerang int(e^x*sin x) at budget=3000 ===")
    for label, vg in (("plain goal", False), ("verified goal", True)):
        found, correct, explored, secs, _ = measure(
            BYPARTS_GENERAL + "\n" + INT_NEG, CASES["int(e^x*sin x)"],
            use_theory=True, cost=cost_int_and_size, budget=3000,
            verified_goal=vg)
        flag = "CORRECT" if correct else ("WRONG" if found else "-")
        print(f"  [{label:13s}] found={found} {flag} "
              f"explored={explored} {secs:.2f}s")

    print("\n=== DECISION ===")
    classic = ["int(x*cos x)", "int(x*sin x)", "int(x^2*e^x)", "int(x*ln x)"]
    for mech, closed in results.items():
        got = [c for c in classic if c in closed]
        print(f"  {mech:42s} closed CORRECTLY: {got}")
    print("  boomerang int(e^x*sin x): expected frontier (a wrong int-free "
          "'solution' under a plain goal; needs a verified goal or C3)")


if __name__ == "__main__":
    main()
