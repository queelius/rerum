# RERUM

**Rewriting Expressions via Rules Using Morphisms**

A pattern matching and term rewriting library for symbolic computation in Python.

## Features

- **Declarative Rules** - Define rewrite rules in a simple DSL
- **Pattern Matching** - Match expressions with variables, constants, and rest patterns
- **Conditional Guards** - Rules with `when` conditions
- **Rule Priorities** - Control which rules fire first
- **Named Groups** - Organize rules into selectable groups
- **Multiple Strategies** - exhaustive, once, bottomup, topdown
- **CLI & REPL** - Interactive exploration and scripting
- **Extensible** - Custom preludes for new operations

## Quick Start

```python
from rerum import RuleEngine, E

engine = RuleEngine.from_dsl('''
    @add-zero: (+ ?x 0) => :x
    @mul-one: (* ?x 1) => :x
''')

engine(E("(+ y 0)"))  # => "y"
engine(E("(* x 1)"))  # => "x"
```

## Installation

```bash
pip install rerum
```

## Command Line

```bash
$ rerum
rerum> @add-zero: (+ ?x 0) => :x
Added 1 rule(s)
rerum> (+ y 0)
y
```

## Next Steps

- [Getting Started](getting-started.md) - Tutorial introduction
- [DSL Reference](dsl-reference.md) - Complete syntax guide
- [CLI](cli.md) - Command-line interface
- [API Reference](api-reference.md) - Python API
- [Examples](examples.md) - Real-world examples
