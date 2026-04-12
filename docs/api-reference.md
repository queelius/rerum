# API Reference

## RuleEngine

The main class for loading and applying rules.

### Construction

```python
from rerum import RuleEngine, ARITHMETIC_PRELUDE

# Empty engine
engine = RuleEngine()

# With prelude
engine = RuleEngine(fold_funcs=ARITHMETIC_PRELUDE)

# From DSL
engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")

# From file
engine = RuleEngine.from_file("rules.rules")

# From Python list
engine = RuleEngine.from_rules([[["+", ["?", "x"], 0], [":", "x"]]])
```

### Fluent API

```python
engine = (RuleEngine()
    .with_prelude(ARITHMETIC_PRELUDE)
    .load_dsl("@add-zero: (+ ?x 0) => :x")
    .load_file("more.rules")
    .disable_group("experimental"))
```

### Simplification

```python
# Basic
result = engine(expr)
result = engine.simplify(expr)

# With options
result = engine(expr, strategy="bottomup", groups=["algebra"])

# With tracing
result, trace = engine(expr, trace=True)
```

### Pattern Matching

```python
from rerum import Bindings, NoMatch

# Match returns Bindings or NoMatch
if bindings := engine.match("(+ ?a ?b)", expr):
    print(bindings["a"], bindings["b"])
```

### Rule Application

```python
# Apply one rule
result, metadata = engine.apply_once(expr)

# Find matching rules
for metadata, bindings in engine.rules_matching(expr):
    print(f"{metadata.name}: {bindings.to_dict()}")
```

### Inspection

```python
len(engine)                    # Number of rules
"add-zero" in engine           # Check if rule exists
rule, meta = engine["add-zero"] # Get by name
engine.list_rules()            # DSL format strings
engine.groups()                # All group names

for rule, meta in engine:
    print(meta.name)
```

### Groups

```python
engine.groups()                # Get all group names
engine.disable_group("name")   # Disable a group
engine.enable_group("name")    # Enable a group

# Use specific groups
engine(expr, groups=["algebra", "folding"])
```

### Combining Engines

```python
# Union
combined = engine1 | engine2
engine1 |= engine2

# Sequencing (phase1 to fixpoint, then phase2)
pipeline = engine1 >> engine2
result = pipeline(expr)
```

### Equivalence and Proof

```python
from rerum import format_sexpr

# Lazy generator of equivalent expressions
for eq in engine.equivalents(expr, max_depth=10):
    print(format_sexpr(eq))

# Eager list
forms = engine.enumerate_equivalents(expr, max_depth=10, max_count=1000)

# Prove two expressions are equivalent (bidirectional BFS)
proof = engine.prove_equal(a, b, max_depth=10, max_expressions=5000)
if proof:
    proof.common       # common form
    proof.depth_a      # steps from a
    proof.depth_b      # steps from b
    proof.path_a       # full path (if trace=True)
    proof.path_b
    print(proof.format("full"))

# Boolean convenience
engine.are_equal(a, b)
```

### Cost Optimization

```python
from rerum import expr_size, expr_depth, expr_ops, expr_atoms, make_op_cost_fn

# Built-in metric: "size", "depth", "ops", or "atoms"
result = engine.minimize(expr, metric="size")

# Custom cost function
result = engine.minimize(expr, cost=lambda e: expr_size(e) + 2*expr_depth(e))

# Per-operator costs
result = engine.minimize(expr, op_costs={"+": 1, "*": 2, "^": 10})

# OptimizationResult attributes
result.expr               # minimum-cost expression found
result.cost               # its cost
result.original_cost
result.improvement        # absolute cost reduction
result.improvement_ratio  # fractional improvement (0.0 = none)
result.cost_ratio         # retained cost ratio (1.0 = no change)
result.expressions_checked
bool(result)              # True iff any improvement found
```

### Random Sampling

```python
import random
rng = random.Random(42)

# Single random equivalent via random walk
equiv = engine.random_equivalent(expr, steps=20, rng=rng)

# Sample multiple (unique by default)
samples = engine.sample_equivalents(expr, n=10, unique=True, rng=rng)

# Lazy infinite walk
for eq in engine.random_walk(expr, max_steps=100, rng=rng):
    if interesting(eq):
        break
```

## Expression Builder (E)

```python
from rerum import E

# Parse s-expression
expr = E("(+ x (* 2 y))")

# Build programmatically
expr = E.op("+", "x", E.op("*", 2, "y"))

# Variables
x, y = E.vars("x", "y")
expr = E.op("+", x, y)

# Constants
n = E.const(42)
```

## Bindings

```python
from rerum import Bindings, NoMatch

# Dict-like access
bindings["x"]
bindings.get("x", default)
"x" in bindings
bindings.to_dict()

# NoMatch is falsy
if bindings := engine.match(pattern, expr):
    # matched
```

## RewriteTrace

```python
result, trace = engine(expr, trace=True)

# Properties
trace.initial    # Initial expression
trace.final      # Final expression
trace.steps      # List of RewriteStep

# Formatting
str(trace)                  # Verbose
trace.format("compact")     # Single line
trace.format("rules")       # Just rule names
trace.format("chain")       # Step-by-step

# Statistics
len(trace)                  # Number of steps
trace.summary()             # Brief summary
trace.rule_counts()         # Dict of rule -> count
trace.rules_applied()       # List of rule names

# Iteration
for step in trace:
    print(step.metadata.name)

# Serialization
trace.to_dict()             # JSON-serializable dict

# Boolean
if trace:  # True if any rules applied
    ...
```

## Preludes

```python
from rerum import (
    ARITHMETIC_PRELUDE,  # +, -, *, /, ^
    MATH_PRELUDE,        # arithmetic + sin, cos, exp, log, sqrt, abs
    PREDICATE_PRELUDE,   # >, <, =, const?, var?, list?, and, or, not
    FULL_PRELUDE,        # arithmetic + predicates
    MINIMAL_PRELUDE,     # +, * only
    NO_PRELUDE,          # empty
)

# Helpers for custom preludes
from rerum import nary_fold, binary_only, unary_only

my_prelude = {
    "+": nary_fold(0, lambda a, b: a + b),
    "max": binary_only(max),
    "neg": unary_only(lambda x: -x),
}
```

## Low-Level Functions

```python
from rerum import (
    match,        # Pattern matching
    instantiate,  # Skeleton instantiation
    rewriter,     # Create simplifier function
    parse_sexpr,  # Parse s-expression string
    format_sexpr, # Format to s-expression string
)

# Direct pattern matching: wrap_bindings() returns falsy Bindings on failure
from rerum import wrap_bindings
raw = match(pattern, expr, [])
if wrap_bindings(raw):
    result = instantiate(skeleton, raw, fold_funcs)

# Create rewriter function
simplify = rewriter(rules, fold_funcs=ARITHMETIC_PRELUDE)
result = simplify(expr)

# S-expression I/O
expr = parse_sexpr("(+ x 1)")
text = format_sexpr(expr)
```
