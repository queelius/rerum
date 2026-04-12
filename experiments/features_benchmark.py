"""
Experiments on rerum's equivalence/proof/minimize/random-walk features.

Four domains:
1. Boolean logic       - finite equivalence classes; ground truth clear
2. Linear algebra      - infinite class via associativity + commutativity
3. Derivative calculus - directional minimization (symbolic simplification)
4. Cycle stress test   - pathological no-op bidirectional rule

For each, we measure:
  - wall-clock time
  - expressions examined (when available)
  - whether the expected result was found
  - diversity of random sampling (unique fraction)
"""
from __future__ import annotations
import time
import random
from contextlib import contextmanager

from rerum import (
    RuleEngine,
    format_sexpr,
    parse_sexpr,
    expr_size,
    expr_depth,
)


@contextmanager
def timed(label: str):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    print(f"    [time] {label}: {dt*1000:.2f} ms")


def hr(title: str):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


# ---------------------------------------------------------------------------
# Experiment 1: Boolean logic — prove De Morgan, double negation
# ---------------------------------------------------------------------------
def exp_boolean():
    hr("Experiment 1: Boolean logic — prove_equal on De Morgan, double neg")

    # Bidirectional axioms for boolean algebra (just enough to prove our targets)
    dsl = """
    # De Morgan (both forms, bidirectional)
    @demorgan-and: (not (and ?x ?y)) <=> (or (not :x) (not :y))
    @demorgan-or:  (not (or  ?x ?y)) <=> (and (not :x) (not :y))

    # Double negation
    @double-neg:   (not (not ?x)) <=> :x

    # Commutativity
    @comm-and: (and ?x ?y) <=> (and :y :x)
    @comm-or:  (or  ?x ?y) <=> (or  :y :x)
    """
    engine = RuleEngine.from_dsl(dsl)

    cases = [
        # (a, b, label, expected-provable)
        (["not", ["and", "p", "q"]],
         ["or", ["not", "p"], ["not", "q"]],
         "De Morgan (not-and = or-of-nots)", True),

        (["not", ["not", ["not", "p"]]],
         ["not", "p"],
         "Triple negation = single negation", True),

        (["and", ["or", "a", "b"], "c"],
         ["and", "c", ["or", "b", "a"]],
         "Comm of both and/or together", True),

        (["not", ["and", "p", "q"]],
         ["and", "p", "q"],
         "not(p and q) == (p and q)  [SHOULD FAIL]", False),
    ]

    for a, b, label, expected in cases:
        print(f"\n  {label}")
        print(f"    a = {format_sexpr(a)}")
        print(f"    b = {format_sexpr(b)}")
        # A work budget keeps the un-provable case tractable.
        with timed("prove_equal"):
            proof = engine.prove_equal(
                a, b, max_depth=6, max_expressions=2000, trace=True
            )
        if proof:
            print(f"    PROVED: common = {format_sexpr(proof.common)}")
            print(f"    depth_a={proof.depth_a}, depth_b={proof.depth_b}")
            if proof.path_a:
                print(f"    path_a length = {len(proof.path_a)}")
        else:
            print(f"    not provable within max_depth/budget")
        ok = (proof is not None) == expected
        print(f"    => {'OK' if ok else 'UNEXPECTED'}")


# ---------------------------------------------------------------------------
# Experiment 2: Equivalence class size under assoc + commute
# ---------------------------------------------------------------------------
def exp_equivalence_class_size():
    hr("Experiment 2: Size of equivalence class under + assoc/commute")

    dsl = """
    @assoc: (+ (+ ?x ?y) ?z) <=> (+ :x (+ :y :z))
    @comm:  (+ ?x ?y) <=> (+ :y :x)
    """
    engine = RuleEngine.from_dsl(dsl)

    # a + b  => equivalence class should be {a+b, b+a} = 2
    # a+b+c  (parenthesized) has 2 × 3! = 12 shapes if we count all
    #        parenthesizations times permutations; but under full closure
    #        should be exactly 3! × Catalan(2) = 6 × 2 = 12
    # Actually: for n leaves, # of distinct (assoc+comm) expressions is
    #   n! × Catalan(n-1) / symmetries. For n=3: 3! × 2 = 12.
    # For n=4: 4! × 5 = 120.

    tests = [
        (["+", "a", "b"], "a+b", 2),
        (["+", ["+", "a", "b"], "c"], "(a+b)+c", 12),
        (["+", ["+", "a", "b"], ["+", "c", "d"]], "(a+b)+(c+d)", 120),
    ]

    for expr, label, expected_count in tests:
        print(f"\n  {label}  (expected {expected_count} forms)")
        with timed("enumerate_equivalents"):
            forms = engine.enumerate_equivalents(
                expr, max_depth=15, max_count=500
            )
        print(f"    found {len(forms)} distinct forms")
        verdict = "OK" if len(forms) == expected_count else (
            "TRUNCATED" if len(forms) < expected_count else "OVERSHOOT"
        )
        print(f"    => {verdict}")


# ---------------------------------------------------------------------------
# Experiment 3: prove_equal on non-obvious algebraic identity
# ---------------------------------------------------------------------------
def exp_algebraic_proof():
    hr("Experiment 3: prove_equal on (a+b)+(c+d) vs (d+c)+(b+a)")

    dsl = """
    @assoc: (+ (+ ?x ?y) ?z) <=> (+ :x (+ :y :z))
    @comm:  (+ ?x ?y) <=> (+ :y :x)
    """
    engine = RuleEngine.from_dsl(dsl)

    a = ["+", ["+", "a", "b"], ["+", "c", "d"]]
    b = ["+", ["+", "d", "c"], ["+", "b", "a"]]
    print(f"    a = {format_sexpr(a)}")
    print(f"    b = {format_sexpr(b)}")

    for max_depth in (4, 6, 8, 10):
        with timed(f"prove_equal(max_depth={max_depth})"):
            proof = engine.prove_equal(a, b, max_depth=max_depth)
        if proof:
            print(f"    depth={max_depth}: PROVED via {format_sexpr(proof.common)} "
                  f"(a:{proof.depth_a} + b:{proof.depth_b} steps)")
            break
        else:
            print(f"    depth={max_depth}: not found")


# ---------------------------------------------------------------------------
# Experiment 4: minimize() — does cost-directed search actually reduce?
# ---------------------------------------------------------------------------
def exp_minimize():
    hr("Experiment 4: minimize() on expressions with built-in slack")

    # Bidirectional identity+assoc/comm rules, plus unidirectional simplifications
    dsl = """
    # Simplifications (unidirectional; only fire when useful)
    @add-zero:  (+ ?x 0) => :x
    @add-zeroL: (+ 0 ?x) => :x
    @mul-one:   (* ?x 1) => :x
    @mul-oneL:  (* 1 ?x) => :x
    @mul-zero:  (* ?x 0) => 0
    @mul-zeroL: (* 0 ?x) => 0

    # Bidirectional (to reach simplifiable forms via commutation)
    @comm-add:  (+ ?x ?y) <=> (+ :y :x)
    @comm-mul:  (* ?x ?y) <=> (* :y :x)
    """
    engine = RuleEngine.from_dsl(dsl)

    cases = [
        (["+", "x", 0], "x + 0", "x"),
        (["*", 1, ["+", "x", 0]], "1 * (x + 0)", "x"),
        (["+", ["*", "x", 0], ["+", "y", 0]], "(x*0) + (y+0)", "y"),
        (["*", ["+", "x", 0], ["+", "y", 0]], "(x+0) * (y+0)", "(* x y)"),
    ]

    print("\n  --- Run A: minimize() with v0.4 default (unidirectional enabled) ---")
    for expr, label, expected in cases:
        print(f"\n  {label}  (expected minimal: {expected})")
        with timed("minimize default"):
            result = engine.minimize(expr, metric="size", max_depth=8)
        if result:
            print(f"    original cost={result.original_cost}  "
                  f"minimum cost={result.cost}  "
                  f"checked={result.expressions_checked}")
            print(f"    => {format_sexpr(result.expr)}")
        else:
            print(f"    no improvement found; original size={expr_size(expr)}")

    print("\n  --- Run B: minimize(include_unidirectional=False) [strict mode] ---")
    print("  (Only <=> rules apply. Useful when equivalence must be reversible.)")
    for expr, label, expected in cases:
        print(f"\n  {label}  (expected minimal: {expected})")
        with timed("minimize strict"):
            result = engine.minimize(
                expr, metric="size", max_depth=8, include_unidirectional=False
            )
        if result:
            print(f"    original cost={result.original_cost}  "
                  f"minimum cost={result.cost}  "
                  f"checked={result.expressions_checked}")
            print(f"    => {format_sexpr(result.expr)}")
        else:
            print(f"    no improvement found; original size={expr_size(expr)}")


# ---------------------------------------------------------------------------
# Experiment 5: Random walk / sampling — diversity test
# ---------------------------------------------------------------------------
def exp_random_walk():
    hr("Experiment 5: random_walk diversity on assoc+commute (a+b+c+d)")

    dsl = """
    @assoc: (+ (+ ?x ?y) ?z) <=> (+ :x (+ :y :z))
    @comm:  (+ ?x ?y) <=> (+ :y :x)
    """
    engine = RuleEngine.from_dsl(dsl)
    start = ["+", ["+", "a", "b"], ["+", "c", "d"]]

    for n_samples in (10, 50, 200):
        rng = random.Random(42)
        samples = engine.sample_equivalents(
            start, n=n_samples, steps=50, unique=False, rng=rng
        )
        # count unique
        unique = set()
        for s in samples:
            unique.add(str(s))
        print(f"  n={n_samples}: {len(unique)} unique / {len(samples)} sampled "
              f"({len(unique)/len(samples)*100:.0f}% diversity)")


# ---------------------------------------------------------------------------
# Experiment 6: Cycle stress — identity bidirectional rule
# ---------------------------------------------------------------------------
def exp_cycle_stress():
    hr("Experiment 6: Cycle stress — rule where LHS == RHS")

    # Pathological: a rule where lhs and rhs are the same shape
    dsl = """
    @id-plus: (+ ?x ?y) <=> (+ :x :y)
    """
    engine = RuleEngine.from_dsl(dsl)
    start = ["+", "a", "b"]

    with timed("enumerate_equivalents on no-op rule"):
        forms = engine.enumerate_equivalents(start, max_depth=10, max_count=100)
    print(f"    found {len(forms)} forms (expected 1)")
    print(f"    => {'OK (terminates)' if len(forms) <= 2 else 'EXPLODED'}")


# ---------------------------------------------------------------------------
# Experiment 7: Bidirectional BFS scaling — does it beat one-sided?
# ---------------------------------------------------------------------------
def exp_bfs_vs_enumerate():
    hr("Experiment 7: prove_equal vs enumerate_equivalents — which is cheaper?")

    dsl = """
    @assoc: (+ (+ ?x ?y) ?z) <=> (+ :x (+ :y :z))
    @comm:  (+ ?x ?y) <=> (+ :y :x)
    """
    engine = RuleEngine.from_dsl(dsl)

    # Two forms near opposite ends of the equivalence class
    a = ["+", ["+", ["+", "a", "b"], "c"], "d"]
    b = ["+", "d", ["+", "c", ["+", "b", "a"]]]

    print(f"    a = {format_sexpr(a)}")
    print(f"    b = {format_sexpr(b)}")

    # Strategy 1: enumerate all equivalents of a and check membership
    import sys
    with timed("enumerate then membership-check"):
        all_forms = engine.enumerate_equivalents(a, max_depth=10, max_count=500)
        found = any(f == b for f in all_forms)
    print(f"    enumerate: {len(all_forms)} forms examined, found={found}")

    # Strategy 2: prove_equal (bidirectional BFS)
    with timed("prove_equal"):
        proof = engine.prove_equal(a, b, max_depth=10)
    print(f"    prove_equal: found={proof is not None}")
    if proof:
        print(f"    depths: a={proof.depth_a}, b={proof.depth_b}")


if __name__ == "__main__":
    exp_boolean()
    exp_equivalence_class_size()
    exp_algebraic_proof()
    exp_minimize()
    exp_random_walk()
    exp_cycle_stress()
    exp_bfs_vs_enumerate()
    print("\nDone.")
