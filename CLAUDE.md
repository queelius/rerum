# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RERUM (Rewriting Expressions via Rules Using Morphisms) is a pattern matching and term rewriting library for symbolic computation in Python. It provides a DSL for defining rewrite rules and applies them to s-expression-style nested lists.

## Build and Test Commands

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run tests with coverage
pytest --cov=rerum --cov-report=term-missing

# Run a single test file
pytest rerum/tests/test_rewriter.py

# Run a specific test
pytest rerum/tests/test_rewriter.py::test_function_name -v
```

## Architecture

The library has two main modules:

### rewriter.py - Core Pattern Matching Engine
- `match(pattern, expr, bindings)` - Pattern matching with binding extraction
- `instantiate(skeleton, bindings, fold_funcs)` - Skeleton instantiation with optional compute evaluation
- `rewriter(rules, fold_funcs)` - Factory that creates a simplifier function
- Pattern syntax uses lists: `["?", "x"]` for any expr, `["?c", "x"]` for constants, `["?v", "x"]` for variables
- Preludes (`ARITHMETIC_PRELUDE`, `MATH_PRELUDE`, etc.) define operations for the `(! op ...)` compute form

### engine.py - DSL Parser and Rule Engine
- `RuleEngine` - High-level API for loading and applying rules
- `parse_sexpr(s)` / `format_sexpr(expr)` - S-expression parsing/formatting
- `parse_rule_line(line)` - Parses DSL rule syntax: `@name: (pattern) => (skeleton)`
- DSL pattern syntax: `?x` or `?x:expr`, `?x:const`, `?x:var`, `?x:free(v)`, `?x...`
- DSL skeleton syntax: `:x` for substitution, `:x...` for splice, `(! op args)` for compute

## Key Design Patterns

- **Separation of rules and preludes**: Rules are pure data (serializable as DSL/JSON); preludes are Python code that defines what operations actually do. Rules can only invoke operations explicitly enabled in the prelude.
- **Bindings as dict-like wrapper**: `Bindings` class provides dict-like access (`bindings["x"]`); `NoMatch` singleton is falsy for pattern-match failure.
- **Recursive simplification**: The rewriter applies rules exhaustively, recursing into subexpressions until no rule matches.
- **Fluent API**: Methods like `.with_prelude()`, `.load_dsl()` return `self` for chaining.

## Fluent API

```python
from rerum import RuleEngine, E, ARITHMETIC_PRELUDE, Bindings, NoMatch

# Build engines with method chaining
engine = (RuleEngine()
    .with_prelude(ARITHMETIC_PRELUDE)
    .load_dsl('''
        @add-zero: (+ ?x 0) => :x
        @fold: (+ ?a:const ?b:const) => (! + :a :b)
    '''))

# Expression builder E - generic, doesn't privilege any operators
expr = E("(+ (* x 1) 0)")           # Parse s-expression string
expr = E.op("+", "x", E.op("*", 2, "y"))  # Build programmatically
x, y = E.vars("x", "y")             # Create variable symbols

# Pattern matching with walrus operator
if bindings := engine.match("(+ ?a ?b)", expr):
    print(bindings["a"], bindings["b"])
else:
    print("no match")

# Single-step rewriting
result, meta = engine.apply_once(expr)
if meta:
    print(f"Applied rule: {meta.name}")

# Find all rules that could fire
for meta, bindings in engine.rules_matching(expr):
    print(f"{meta.name} would bind {dict(bindings)}")

# Rewriting strategies
engine(expr)                      # default: exhaustive
engine(expr, strategy="once")     # single rule application
engine(expr, strategy="bottomup") # children before parent
engine(expr, strategy="topdown")  # parent before children

# Phased rewriting with >> operator
expand = RuleEngine.from_dsl("@sq: (square ?x) => (* :x :x)")
simplify = RuleEngine.from_dsl("@fold: (* ?a:const ?b:const) => (! * :a :b)")
    .with_prelude(ARITHMETIC_PRELUDE)
pipeline = expand >> simplify
pipeline(E("(square 5)"))  # => 25

# Union with | operator
both = expand | simplify
```

## Expression Representation

Expressions are Python nested lists in prefix notation:
```python
["+", "x", ["*", 2, "y"]]  # represents: x + 2*y
```
Atoms are strings (variables) or numbers (constants). This matches s-expression semantics.
