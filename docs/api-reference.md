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

# Direct pattern matching
bindings = match(pattern, expr, [])
if bindings != "failed":
    result = instantiate(skeleton, bindings, fold_funcs)

# Create rewriter function
simplify = rewriter(rules, fold_funcs=ARITHMETIC_PRELUDE)
result = simplify(expr)

# S-expression I/O
expr = parse_sexpr("(+ x 1)")
text = format_sexpr(expr)
```
