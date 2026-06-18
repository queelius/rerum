# Getting Started

This guide introduces RERUM's core concepts through examples.

## Expressions

RERUM works with s-expressions - nested lists in prefix notation:

```python
from rerum import E

# Parse s-expression strings
expr = E("(+ x (* 2 y))")  # => ["+", "x", ["*", 2, "y"]]

# Build programmatically
expr = E.op("+", "x", E.op("*", 2, "y"))

# Create variables
x, y = E.vars("x", "y")
```

## Rules

Rules transform expressions. A rule has a **pattern** (what to match) and a **skeleton** (what to produce):

```python
from rerum import RuleEngine

engine = RuleEngine.from_dsl('''
    @add-zero: (+ ?x 0) => :x
''')

engine(E("(+ y 0)"))  # => "y"
```

- `?x` matches any expression and binds it to `x`
- `:x` substitutes the bound value

## Pattern Types

| Pattern | Matches |
|---------|---------|
| `?x` | Any expression |
| `?x:const` | Numbers only |
| `?x:var` | Symbols only |
| `?x:free(v)` | Expressions not containing `v` |
| `?x...` | Rest of arguments |

## Computation

The `(! op args...)` form evaluates operations:

```python
from rerum import RuleEngine, ARITHMETIC_PRELUDE

engine = (RuleEngine()
    .with_prelude(ARITHMETIC_PRELUDE)
    .load_dsl('''
        @fold: (+ ?a:const ?b:const) => (! + :a :b)
    '''))

engine(E("(+ 2 3)"))  # => 5
```

!!! note
    `(! ...)` only evaluates when the prelude defines the operation.
    Without a prelude, it remains symbolic.

## Conditional Guards

Rules can have conditions:

```python
from rerum import FULL_PRELUDE

engine = (RuleEngine()
    .with_prelude(FULL_PRELUDE)
    .load_dsl('''
        @abs-pos: (abs ?x) => :x when (! > :x 0)
        @abs-neg: (abs ?x) => (! - 0 :x) when (! < :x 0)
    '''))

engine(E("(abs -5)"))  # => 5
```

## Priorities

Higher priority rules fire first:

```python
engine = RuleEngine.from_dsl('''
    @general: (f ?x) => (g :x)
    @specific[100]: (f 0) => zero
''')

engine(E("(f 0)"))  # => "zero" (specific wins)
```

## Groups

Organize rules into named groups:

```python
engine = RuleEngine.from_dsl('''
    [algebra]
    @add-zero: (+ ?x 0) => :x

    [expand]
    @square: (square ?x) => (* :x :x)
''')

# Use only algebra rules
engine(E("(+ x 0)"), groups=["algebra"])

# Disable a group
engine.disable_group("expand")
```

## Strategies

Control how rules are applied:

```python
# exhaustive (default): repeat until no rules match
engine(expr, strategy="exhaustive")

# once: apply at most one rule
engine(expr, strategy="once")

# bottomup: simplify children first
engine(expr, strategy="bottomup")

# topdown: try parent first
engine(expr, strategy="topdown")
```

## Tracing

See which rules fire:

```python
result, trace = engine(expr, trace=True)
print(trace)
print(trace.format("compact"))
print(trace.rules_applied())
```

## Bidirectional Rules and Equivalence

So far rules reduce an expression toward a normal form. A `<=>` rule declares
a reversible equivalence and lets you reason over an equivalence class instead:

```python
engine = RuleEngine.from_dsl('''
    @comm-add: (+ ?x ?y) <=> (+ :y :x)
    @assoc:    (+ (+ ?x ?y) ?z) <=> (+ :x (+ :y :z))
''')

# Prove two forms equal by bidirectional search
proof = engine.prove_equal(["+", ["+", "a", "b"], "c"],
                           ["+", "c", ["+", "b", "a"]], max_depth=10)
bool(proof)   # True

# Minimize by a cost metric
result = engine.minimize(["+", ["+", "a", "b"], "c"], metric="size")
```

See [Equivalence & Proof](equivalence.md) for `equivalents`, `prove_equal`,
and `minimize`.

## Goal-Directed Search

Confluent rule sets reduce to a fixpoint; non-confluent ones (where rules
COMPETE -- integration is the classic case) need search with backtracking.
`solve` does best-first search toward a caller-supplied goal:

```python
from rerum.solve import solve, contains_op

engine = RuleEngine.from_dsl("@int-cos: (int (cos ?x) ?x) => (sin :x)")
result = solve(engine, ["int", ["cos", "x"], "x"],
               lambda e: not contains_op(e, {"int"}), max_nodes=2000)
result.found      # True
result.solution   # ["sin", "x"]
```

The goal is YOURS; the engine just searches. Failure is honest
(`found=False` within budget), never a hang.

## Theories and Canonical Forms

Operator signatures are DATA. Declare which operators are
associative-commutative, with identities and annihilators, in JSON, and get
canonical forms with no engine changes per domain:

```python
from rerum.normalize import Theory, normalize

theory = Theory.from_json('{"+": {"ac": true, "identity": 0}}')
normalize(["+", "b", "a", 0], theory)   # ["+", "a", "b"]
```

The same machinery serves arithmetic, boolean algebra, and set algebra -- the
engine never knows which.

## Numeric Verification

Evaluate ground terms over a prelude, and check whether two forms are
equivalent by sampling:

```python
from rerum.numeval import numeval, numeric_equiv
from rerum import MATH_PRELUDE

numeval(["+", "x", 1], {"x": 2}, MATH_PRELUDE)             # 3
numeric_equiv(["*", "x", 2], ["+", "x", "x"],
              {"x": (0.5, 2.0)}, MATH_PRELUDE)             # True
```

Sample points come from ranges (DATA); domain errors skip a point, and an
all-skipped check is False, never a vacuous True.

## Rule-set Manifests

A *manifest* is a self-describing rule file: its `:`-directives declare the
preludes, theory, and metadata a domain needs, so the whole domain assembles
from one file. `RuleEngine.from_manifest` wires it all up and fails loud if a
required fold op is missing:

```python
from rerum import RuleEngine
from rerum.engine import format_sexpr

engine = RuleEngine.from_manifest("examples/differentiation.manifest")
format_sexpr(engine(["dd", "x", "x"]))   # "1"
```

See the [manifest section of the DSL Reference](dsl-reference.md#rule-set-manifests)
for the six directives.

## The Domain Library and MCP Server

Every domain under `examples/` is rules + data the engine loads -- the
standing proof that the engine is general:

- **Calculus**: differentiation, integration, limits (numerically certified)
- **Boolean algebra** and **set algebra** (rule-for-rule isomorphic)
- **Peano arithmetic** (computation from PURE rewriting, empty prelude)
- **SKI combinators** (Turing-complete in 3 rules)

The MCP server (`pip install "rerum[mcp]"`, then `rerum-mcp`) exposes the
engine to LLM agents through typed tools, including an agentic loop where the
model proposes rules when the engine is stuck. The server holds NO domain
logic.

## Next Steps

- [DSL Reference](dsl-reference.md) - Complete syntax, manifests included
- [Equivalence & Proof](equivalence.md) - Bidirectional rules, proof,
  minimize
- [CLI](cli.md) - Interactive use
- [Examples](examples.md) - Real-world patterns and the domain library

For the training corpora layer and the full MCP tool surface, see the
[README](https://github.com/queelius/rerum#readme).
