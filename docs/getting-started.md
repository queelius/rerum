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

## Next Steps

- [DSL Reference](dsl-reference.md) - Complete syntax
- [CLI](cli.md) - Interactive use
- [Examples](examples.md) - Real-world patterns
