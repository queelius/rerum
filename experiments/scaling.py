"""Scaling experiment: equivalence class size under assoc+commute vs n-leaf sum."""
import time
import math
from rerum import RuleEngine

dsl = """
@assoc: (+ (+ ?x ?y) ?z) <=> (+ :x (+ :y :z))
@comm:  (+ ?x ?y) <=> (+ :y :x)
"""
engine = RuleEngine.from_dsl(dsl)

def left_comb(leaves):
    """Left-leaning comb: (((a+b)+c)+d)..."""
    e = leaves[0]
    for x in leaves[1:]:
        e = ["+", e, x]
    return e

def catalan(n):
    return math.comb(2*n, n) // (n+1)

print(f"{'n':>3} {'theory':>10} {'found':>10} {'time_ms':>10} {'per_form':>10}")
print("-" * 50)
for n in (2, 3, 4, 5):
    leaves = [chr(ord('a')+i) for i in range(n)]
    expr = left_comb(leaves)
    # Theoretical class size: n! * C(n-1)
    theory = math.factorial(n) * catalan(n-1)
    t0 = time.perf_counter()
    forms = engine.enumerate_equivalents(expr, max_depth=30, max_count=10000)
    dt = (time.perf_counter() - t0) * 1000
    per = dt / len(forms) if forms else 0
    print(f"{n:>3} {theory:>10} {len(forms):>10} {dt:>10.1f} {per:>10.3f}")
