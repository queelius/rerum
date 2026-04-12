# Equivalence, Proof, and Optimization

RERUM's rewriting machinery is powerful enough to treat rules as a theory
over expressions. Once you have bidirectional rules (rules that can fire in
either direction), you can:

- **Enumerate** all expressions equivalent to a given one under your theory.
- **Prove** two expressions are equivalent by searching for a common form.
- **Minimize** an expression by finding the lowest-cost member of its
  equivalence class.
- **Sample** from an equivalence class stochastically.

These features share a common substrate: search over the rewrite graph.

## Bidirectional Rules

A bidirectional rule is written with `<=>`:

```python
from rerum import RuleEngine

engine = RuleEngine.from_dsl('''
    @comm-add: (+ ?x ?y) <=> (+ :y :x)
    @assoc:    (+ (+ ?x ?y) ?z) <=> (+ :x (+ :y :z))
''')
```

Each `<=>` rule is expanded into two unidirectional rules internally
(`-fwd` and `-rev` variants), so the equivalence class of an expression
is closed under both directions.

You can mix bidirectional and unidirectional rules in one engine. By
convention, use `<=>` for true equivalences (commutativity, associativity,
De Morgan) and `=>` for directional simplifications (`(+ ?x 0) => :x`).

## Enumerating Equivalents

`equivalents()` is a lazy generator over the reachable equivalence class:

```python
from rerum import format_sexpr

engine = RuleEngine.from_dsl('''
    @comm:  (+ ?x ?y) <=> (+ :y :x)
    @assoc: (+ (+ ?x ?y) ?z) <=> (+ :x (+ :y :z))
''')

for eq in engine.equivalents(["+", ["+", "a", "b"], "c"], max_depth=6):
    print(format_sexpr(eq))
# (+ (+ b a) c)
# (+ c (+ a b))
# (+ a (+ b c))
# ... 12 distinct forms total
```

`enumerate_equivalents()` is the eager variant returning a list. Both
accept:

- `max_depth` (default 10): rewrite steps from the start.
- `max_count`: cap on results.
- `strategy`: `"bfs"` (default) or `"dfs"`.
- `include_unidirectional` (default `False`): if `True`, `=>` rules
  participate in the search.
- `groups`: restrict to a subset of rule groups.

### Class sizes grow fast

Under `assoc + commute`, the class of an `n`-term sum has exactly
`n! × Catalan(n-1)` members:

| n | forms |
|---|-------|
| 2 | 2     |
| 3 | 12    |
| 4 | 120   |
| 5 | 1680  |
| 6 | ~30k  |

Past `n=5`, full enumeration is impractical. Use `prove_equal` or
`minimize` with sensible depth caps instead.

## Proving Equality

`prove_equal()` uses **bidirectional BFS**, expanding from both expressions
simultaneously and looking for a meeting point. This is dramatically cheaper
than enumerating one side and checking membership.

```python
proof = engine.prove_equal(
    ["+", ["+", "a", "b"], "c"],
    ["+", "c", ["+", "b", "a"]]
)
if proof:
    print(f"Common form: {format_sexpr(proof.common)}")
    print(f"Depths: a={proof.depth_a}, b={proof.depth_b}")
```

### With a work budget

Un-provable queries otherwise exhaust the entire depth-bounded ball from
both sides. The `max_expressions` parameter caps total work:

```python
# Returns None quickly if budget exhausted.
proof = engine.prove_equal(a, b, max_depth=8, max_expressions=5000)
```

Provable queries generally finish well under the budget because
bidirectional BFS meets in the middle. The budget is your safety net for
queries where the answer is "probably not equal, but I don't want to
wait forever to find out."

### With a trace

Pass `trace=True` to get the actual rewrite paths:

```python
proof = engine.prove_equal(a, b, trace=True)
print(proof.format("full"))
# prints the step-by-step path from a and from b to the common form
```

### Boolean check

If you only need yes/no:

```python
if engine.are_equal(a, b):
    ...
```

## Cost Optimization

`minimize()` finds the minimum-cost equivalent expression under a cost
function you supply:

```python
from rerum import expr_size, expr_depth

engine = RuleEngine.from_dsl('''
    @add-zero: (+ ?x 0) => :x
    @mul-one:  (* ?x 1) => :x
    @mul-zero: (* ?x 0) => 0
    @comm:     (+ ?x ?y) <=> (+ :y :x)
''')

result = engine.minimize(
    ["+", ["*", "x", 0], ["+", "y", 0]],
    metric="size",
)
print(result.expr)            # "y"
print(result.cost)            # 1
print(result.improvement_ratio)   # 0.857
print(result.cost_ratio)          # 0.143
```

### Cost options

You can pick any of three ways to specify cost:

```python
# Built-in metric
result = engine.minimize(expr, metric="size")   # or "depth", "ops", "atoms"

# Custom cost function
result = engine.minimize(expr, cost=lambda e: expr_size(e) + 2*expr_depth(e))

# Per-operator cost
result = engine.minimize(expr, op_costs={"+": 1, "*": 2, "/": 5, "^": 10})
```

### The `include_unidirectional` default

`minimize()` defaults `include_unidirectional=True` because most
simplification rules are written unidirectionally (`=>`). If you set it to
`False`, only `<=>` rules are explored, which is useful when you want to
preserve equivalence strictly under a reversible theory but rarely what you
want for general simplification.

`prove_equal()` and `equivalents()` default to `False` because those
features encode a symmetric question (is X equivalent to Y?) and using
unidirectional rules breaks the symmetry assumption.

## Stochastic Exploration

For large equivalence classes, random sampling can surface interesting
forms faster than breadth-first enumeration:

```python
import random

# Random walk from start (revisits allowed)
equiv = engine.random_equivalent(expr, steps=20)

# Sample distinct equivalents
samples = engine.sample_equivalents(expr, n=10, unique=True)

# Reproducible with a seeded RNG
rng = random.Random(42)
samples = engine.sample_equivalents(expr, n=5, rng=rng)

# Lazy walk
for eq in engine.random_walk(expr, max_steps=100, rng=rng):
    if interesting(eq):
        break
```

Diversity drops as sample size grows relative to class size (this is the
birthday paradox showing up). For a 120-member class, 200 random draws
yield roughly 100 unique forms.

## Choosing the right tool

| Question | Method |
|----------|--------|
| Are these two expressions equivalent? | `prove_equal` / `are_equal` |
| What is the simplest form? | `minimize` |
| What forms are equivalent? | `equivalents` / `enumerate_equivalents` |
| Pick a random equivalent | `random_equivalent` |
| Sample many equivalents | `sample_equivalents` |
| Walk through the class | `random_walk` |

For asymmetric simplification (normal form), keep using `engine(expr)` with
unidirectional rules. The features in this page are for reasoning *over*
equivalence classes, not computing canonical forms.

## Benchmarks

The `experiments/` directory in the repository contains runnable probes
that validate correctness and measure scaling:

- `features_benchmark.py` covers proof, enumeration, minimize, and
  random-walk on boolean, algebraic, and stress-test rule sets.
- `scaling.py` measures enumeration cost against the theoretical
  `n! × Catalan(n-1)` class size.
