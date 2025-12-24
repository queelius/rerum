# RERUM

[![PyPI version](https://badge.fury.io/py/rerum.svg)](https://badge.fury.io/py/rerum)
[![CI](https://github.com/queelius/rerum/actions/workflows/ci.yml/badge.svg)](https://github.com/queelius/rerum/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Rewriting Expressions via Rules Using Morphisms**

A pattern matching and term rewriting library for symbolic computation in Python.

## Installation

```bash
pip install rerum
```

## Quick Start

```python
from rerum import RuleEngine, E

# Create an engine with rules
engine = RuleEngine.from_dsl('''
    @add-zero "x + 0 = x": (+ ?x 0) => :x
    @mul-one: (* ?x 1) => :x
    @mul-zero: (* ?x 0) => 0
''')

# Simplify expressions using E() to parse s-expressions
engine(E("(+ y 0)"))           # => "y"
engine(E("(* x 1)"))           # => "x"
engine(E("(* (+ a 0) 0)"))     # => 0

# Or use raw lists
engine(["+", "y", 0])          # => "y"
```

## DSL Syntax

Rules use a simple, readable syntax:

```
# Comments start with #
@rule-name: (pattern) => (skeleton)
@rule-name "Description": (pattern) => (skeleton)
@rule-name[100]: (pattern) => (skeleton)           # With priority
@rule-name: (pattern) => (skeleton) when (cond)    # With guard
```

### Pattern Syntax

| Syntax | Meaning |
|--------|---------|
| `?x` or `?x:expr` | Match any expression, bind to x |
| `?x:const` | Match constant (number) only |
| `?x:var` | Match variable (symbol) only |
| `?x:free(v)` | Match expression not containing v |
| `?x...` | Match zero or more remaining args (rest pattern) |

### Skeleton Syntax

| Syntax | Meaning |
|--------|---------|
| `:x` | Substitute bound value of x |
| `:x...` | Splice list bound to x |
| `(! op args...)` | Compute: evaluate op with args using prelude |

## Expression Builder

The `E` builder provides convenient expression construction:

```python
from rerum import E

# Parse s-expression strings
expr = E("(+ x (* 2 y))")      # => ["+", "x", ["*", 2, "y"]]

# Build programmatically
expr = E.op("+", "x", E.op("*", 2, "y"))

# Create variables
x, y = E.vars("x", "y")
expr = E.op("+", x, E.op("*", 2, y))
```

## Conditional Guards

Rules can have conditions that must be satisfied:

```python
from rerum import RuleEngine, E, FULL_PRELUDE

engine = (RuleEngine()
    .with_prelude(FULL_PRELUDE)
    .load_dsl('''
        @abs-pos: (abs ?x) => :x when (! > :x 0)
        @abs-neg: (abs ?x) => (! - 0 :x) when (! < :x 0)
        @abs-zero: (abs ?x) => 0 when (! = :x 0)
    '''))

engine(E("(abs 5)"))   # => 5
engine(E("(abs -5)"))  # => 5
engine(E("(abs 0)"))   # => 0
```

Guards use the `(! ...)` compute syntax and have access to type predicates:
- `const?` - true for numbers
- `var?` - true for symbols/variables
- `list?` - true for compound expressions
- Comparison: `>`, `<`, `>=`, `<=`, `=`, `!=`
- Logical: `and`, `or`, `not`

## Rule Priorities

Higher priority rules fire first:

```python
engine = RuleEngine.from_dsl('''
    @general: (+ ?x ?y) => (add :x :y)
    @specific[100]: (+ 0 ?x) => :x        # Fires first
    @specific2[100]: (+ ?x 0) => :x       # Fires first
''')

engine(E("(+ 0 y)"))  # => "y" (specific rule wins)
engine(E("(+ a b)"))  # => ["add", "a", "b"] (general rule)
```

## Named Rulesets (Groups)

Organize rules into groups and selectively enable them:

```python
engine = RuleEngine.from_dsl('''
    [algebra]
    @add-zero: (+ ?x 0) => :x
    @mul-one: (* ?x 1) => :x

    [calculus]
    @dd-const: (dd ?c:const ?v:var) => 0
    @dd-var: (dd ?x:var ?x) => 1
''')

# Use only algebra rules
engine(E("(+ x 0)"), groups=["algebra"])

# Disable a group
engine.disable_group("calculus")

# Get all group names
engine.groups()  # => {"algebra", "calculus"}
```

## Rewriting Strategies

Control how rules are applied:

```python
# exhaustive (default): Apply rules repeatedly until fixpoint
result = engine(expr, strategy="exhaustive")

# once: Apply at most one rule anywhere
result = engine(expr, strategy="once")

# bottomup: Simplify children first, then parent
result = engine(expr, strategy="bottomup")

# topdown: Try parent first, then children
result = engine(expr, strategy="topdown")
```

## Tracing

See which rules are applied:

```python
result, trace = engine(expr, trace=True)
print(trace)  # Verbose multi-line format

# Different formats
print(trace.format("compact"))  # Single line
print(trace.format("rules"))    # Just rule names
print(trace.format("chain"))    # Step-by-step chain

# Statistics
print(trace.summary())          # Brief summary
print(trace.rule_counts())      # Rule usage counts
print(trace.rules_applied())    # List of rules in order

# Serialization
import json
json.dumps(trace.to_dict())
```

## Preludes and Constant Folding

Preludes define computational primitives for the `(! op ...)` compute form:

```python
from rerum import RuleEngine, ARITHMETIC_PRELUDE, MATH_PRELUDE, FULL_PRELUDE

# Fluent construction with prelude
engine = (RuleEngine()
    .with_prelude(ARITHMETIC_PRELUDE)
    .load_dsl('''
        @fold: (+ ?a:const ?b:const) => (! + :a :b)
    '''))

engine(E("(+ 1 2)"))  # => 3
```

### Available Preludes

| Prelude | Operations |
|---------|------------|
| `ARITHMETIC_PRELUDE` | `+`, `-`, `*`, `/`, `^` |
| `MATH_PRELUDE` | Arithmetic + `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `abs` |
| `PREDICATE_PRELUDE` | `>`, `<`, `=`, `const?`, `var?`, `list?`, `and`, `or`, `not` |
| `FULL_PRELUDE` | Arithmetic + Predicates |
| `MINIMAL_PRELUDE` | `+`, `*` only |
| `NO_PRELUDE` | Empty (pure symbolic rewriting) |

### Custom Preludes

```python
from rerum import RuleEngine, nary_fold, unary_only, binary_only

my_prelude = {
    "+": nary_fold(0, lambda a, b: a + b),      # n-ary with identity
    "max": binary_only(max),                     # binary only
    "neg": unary_only(lambda x: -x),            # unary only
    "gcd": binary_only(math.gcd),               # custom function
}

engine = RuleEngine().with_prelude(my_prelude).load_dsl(rules)
```

## Engine Sequencing

Apply engines in phases:

```python
expand = RuleEngine.from_dsl("@square: (square ?x) => (* :x :x)")
simplify = RuleEngine.from_dsl("@fold: (* ?a:const ?b:const) => (! * :a :b)",
                                fold_funcs=ARITHMETIC_PRELUDE)

# Sequence with >>
normalize = expand >> simplify
normalize(E("(square 3)"))  # => 9

# Chain multiple phases
pipeline = expand >> simplify >> another_engine
```

## Fluent API

```python
from rerum import RuleEngine, FULL_PRELUDE

engine = (RuleEngine()
    .with_prelude(FULL_PRELUDE)
    .load_dsl('''
        @add-zero: (+ ?x 0) => :x
    ''')
    .load_file("more_rules.rules")
    .add_rule(
        pattern=E.op("+", ["?", "x"], ["?", "x"]),
        skeleton=E.op("*", 2, [":", "x"]),
        name="double"
    )
    .disable_group("experimental"))

# Pattern matching
if bindings := engine.match("(+ ?a ?b)", expr):
    print(bindings["a"], bindings["b"])

# Apply single rule
result, meta = engine.apply_once(expr)

# Find matching rules
for meta, bindings in engine.rules_matching(expr):
    print(f"Rule {meta.name} matches")
```

## Variadic Patterns

Rest patterns (`?x...`) capture remaining arguments:

```python
engine = RuleEngine.from_dsl('''
    # Flatten nested additions
    @flatten-add: (+ (+ ?a ?b) ?rest...) => (+ :a :b :rest...)

    # Constant folding with rest
    @fold-add: (+ ?a:const ?b:const ?rest...) => (+ (! + :a :b) :rest...)
''', fold_funcs=ARITHMETIC_PRELUDE)

engine(E("(+ (+ 1 2) 3 4)"))  # => ["+", 1, 2, 3, 4] => 10
```

## API Reference

### Creating Engines

```python
# From DSL text
engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")

# From file
engine = RuleEngine.from_file("rules.rules")  # DSL format
engine = RuleEngine.from_file("rules.json")   # JSON format

# From Python lists
rules = [[["+", ["?", "x"], 0], [":", "x"]]]
engine = RuleEngine.from_rules(rules)

# With prelude
engine = RuleEngine.from_dsl(dsl_text, fold_funcs=ARITHMETIC_PRELUDE)
```

### Using Engines

```python
# Simplify (callable shorthand)
result = engine(expr)

# With options
result = engine(expr, strategy="bottomup", groups=["algebra"])

# With tracing
result, trace = engine(expr, trace=True)
```

### Inspecting Engines

```python
len(engine)                  # Number of rules
"add-zero" in engine         # Check if rule exists
rule, meta = engine["add-zero"]  # Get by name
engine.list_rules()          # DSL format strings
engine.groups()              # All group names

for rule, meta in engine:    # Iterate
    print(meta.name, meta.description)
```

### Combining Engines

```python
algebra = RuleEngine.from_file("algebra.rules")
calculus = RuleEngine.from_file("calculus.rules")

combined = algebra | calculus  # Union
algebra |= calculus            # In-place union
phased = algebra >> calculus   # Sequence (algebra first, then calculus)
```

## JSON Format

Rules can also be loaded from JSON:

```json
{
    "name": "algebra",
    "description": "Basic algebraic rules",
    "rules": [
        {
            "name": "add-zero",
            "description": "x + 0 = x",
            "pattern": ["+", ["?", "x"], 0],
            "skeleton": [":", "x"]
        }
    ]
}
```

## Architecture

```
+-------------------------------------+
|  Rules (DSL/JSON) - Serializable    |
|  - Pattern matching                 |
|  - Symbolic transformation          |
|  - (! op ...) references operations |
|  - Conditions reference predicates  |
+------------------+------------------+
                   | references
                   v
+-------------------------------------+
|  Prelude (Python) - Trusted Code    |
|  - Defines what operations do       |
|  - Provided by developer            |
|  - Cannot be injected from rules    |
+-------------------------------------+
```

Rules loaded from untrusted sources cannot execute arbitrary code - they can only invoke operations explicitly enabled in the prelude.

## Low-Level API

For advanced use cases:

```python
from rerum import rewriter, match, instantiate, parse_sexpr, format_sexpr

# Create a rewriter function directly
simplify = rewriter(rules, fold_funcs=ARITHMETIC_PRELUDE)
result = simplify(expr)

# Pattern matching
bindings = match(pattern, expr, [])
if bindings != "failed":
    result = instantiate(skeleton, bindings, fold_funcs)

# S-expression parsing
expr = parse_sexpr("(+ x (* 2 y))")  # => ["+", "x", ["*", 2, "y"]]
text = format_sexpr(expr)             # => "(+ x (* 2 y))"
```

## Command-Line Interface

RERUM includes a CLI for interactive use and scripting.

### REPL Mode

```bash
$ rerum
rerum> @add-zero: (+ ?x 0) => :x
Added 1 rule(s)
rerum> (+ y 0)
y
rerum> :quit
```

### Script Mode

Create a `.rerum` file:

```bash
#!/usr/bin/env rerum
:prelude full

# Symbolic rule: transforms structure, no computation
@square: (square ?x) => (* :x :x)

# Computation rule: (!) evaluates when args are constants
@fold-mul: (* ?a ?b) => (! * :a :b) when (! and (! const? :a) (! const? :b))

# (square x) => (* x x)  (symbolic, x stays as x)
(square x)

# (square 5) => (* 5 5) => 25  (fold-mul computes the result)
(square 5)
```

Output:
```
(* x x)
25
```

Run scripts:
```bash
$ rerum script.rerum
$ chmod +x script.rerum && ./script.rerum  # With shebang
```

### One-Shot Mode

```bash
$ rerum -r rules.rules -p full -e "(+ x 0)"
x
```

### Pipe Mode

```bash
$ echo "(+ x 0)" | rerum -r rules.rules -p full -q
x
```

### CLI Options

```
rerum [script]              Run a script or start REPL
  -r, --rules FILE          Load rules (can repeat)
  -e, --expr EXPR           Evaluate single expression
  -p, --prelude NAME        Set prelude (arithmetic, math, full, none, or path.py)
  -t, --trace               Enable tracing
  -s, --strategy NAME       Strategy: exhaustive, once, bottomup, topdown
  -q, --quiet               Suppress non-essential output
```

### REPL Commands

```
:help              Show help
:load FILE         Load rules from file
:rules             List loaded rules
:clear             Clear all rules
:prelude NAME      Set prelude
:trace on|off      Toggle tracing
:strategy NAME     Set rewriting strategy
:groups            Show groups
:enable GROUP      Enable a group
:disable GROUP     Disable a group
:quit              Exit
```

### Custom Preludes

Create a Python file with a `PRELUDE` dict:

```python
# my_prelude.py
from rerum import binary_only, unary_only
import math

PRELUDE = {
    "gcd": binary_only(math.gcd),
    "factorial": unary_only(math.factorial),
}
```

Use it:
```bash
$ rerum -p my_prelude.py -r rules.rules
```

## License

MIT
