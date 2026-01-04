# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RERUM (Rewriting Expressions via Rules Using Morphisms) is a pattern matching and term rewriting library for symbolic computation in Python. It provides a DSL for defining rewrite rules and applies them to s-expression-style nested lists.

## Build and Test Commands

```bash
pip install -e ".[dev]"                    # Install with dev dependencies
pip install -e ".[docs]"                   # Install with docs dependencies
pytest                                      # Run all tests
pytest --cov=rerum --cov-report=term-missing  # With coverage
pytest rerum/tests/test_guards.py -v       # Single test file
pytest rerum/tests/test_guards.py::TestGuardParsing::test_parse_when_clause -v  # Single test
mkdocs serve                               # Local docs server
mkdocs build                               # Build docs to site/
```

## Architecture

### Core Modules

**rewriter.py** - Low-level pattern matching engine
- `match(pattern, expr, bindings)` - Pattern matching with binding extraction
- `instantiate(skeleton, bindings, fold_funcs)` - Skeleton instantiation with `(! ...)` compute evaluation
- `rewriter(rules, fold_funcs)` - Factory that creates a simplifier function
- Preludes: `ARITHMETIC_PRELUDE`, `MATH_PRELUDE`, `PREDICATE_PRELUDE`, `FULL_PRELUDE`

**engine.py** - High-level DSL and rule engine
- `RuleEngine` - Main API: load rules, apply strategies, manage groups
- `RuleMetadata` - Rule name, description, priority, condition (guard)
- `RewriteTrace` / `RewriteStep` - Tracing infrastructure
- `parse_sexpr()` / `format_sexpr()` - S-expression I/O
- `E` - Expression builder singleton

**cli.py** - Command-line interface
- `RerumREPL` - Interactive REPL with readline support
- `ScriptRunner` - Executes `.rerum` scripts
- Supports pipe mode, expression mode, custom prelude loading

### Key Design Patterns

- **Rules vs Preludes**: Rules are pure data (DSL/JSON serializable); preludes are Python code defining what `(! op ...)` operations do. Security boundary: rules can only invoke prelude-enabled operations.
- **Fluent API**: Methods return `self` for chaining: `.with_prelude()`, `.load_dsl()`, `.disable_group()`
- **Bindings**: `Bindings` class wraps raw bindings with dict-like access; `NoMatch` singleton is falsy
- **Strategies**: exhaustive (default), once, bottomup, topdown - control rule application order

### DSL Features

```
@name: (pattern) => (skeleton)                    # Basic rule
@name: (pattern) <=> (skeleton)                   # Bidirectional (creates -fwd and -rev)
@name[100]: (pattern) => (skeleton)               # With priority (higher fires first)
@name: (pattern) => (skeleton) when (condition)   # With guard
[groupname]                                        # Start a named group
```

Pattern syntax: `?x`, `?x:const`, `?x:var`, `?x:free(v)`, `?x...`
Skeleton syntax: `:x`, `:x...`, `(! op args...)`

Bidirectional rules (`<=>`) create two rules automatically:
```
@commute: (+ ?x ?y) <=> (+ :y :x)
# Creates: @commute-fwd: (+ ?x ?y) => (+ :y :x)
#          @commute-rev: (+ ?y ?x) => (+ :x :y)
```

### Equivalence Enumeration

Use `equivalents()` to explore all equivalent forms of an expression:
```python
engine = RuleEngine.from_dsl('''
    @commute: (+ ?x ?y) <=> (+ :y :x)
    @assoc: (+ (+ ?x ?y) ?z) <=> (+ :x (+ :y :z))
''')

# Enumerate equivalent forms (lazy generator)
for expr in engine.equivalents(["+", ["+", "a", "b"], "c"], max_depth=3):
    print(format_sexpr(expr))

# Or collect all at once
all_forms = engine.enumerate_equivalents(expr, max_count=100)
```

### Proving Equality

Use `prove_equal()` to prove two expressions are equivalent:
```python
proof = engine.prove_equal(expr_a, expr_b, max_depth=10)
if proof:
    print(f"Equal via: {format_sexpr(proof.common)}")
    print(f"Distance: {proof.depth_a} + {proof.depth_b} steps")

# Simple boolean check
if engine.are_equal(expr_a, expr_b):
    print("Expressions are equivalent!")

# With full trace
proof = engine.prove_equal(a, b, trace=True)
print(proof.format("full"))  # Shows paths from both expressions
```

### Expression Representation

Expressions are nested Python lists in prefix notation:
```python
["+", "x", ["*", 2, "y"]]  # represents: x + 2*y
```
Atoms are strings (variables) or numbers (constants).

## CLI Usage

```bash
rerum                              # Start REPL
rerum script.rerum                 # Run script
rerum -r rules.rules -p full -e "(+ x 0)"  # One-shot evaluation
echo "(+ x 0)" | rerum -r rules.rules -q   # Pipe mode
```

REPL commands: `:help`, `:load FILE`, `:rules`, `:clear`, `:prelude NAME`, `:trace on/off`, `:strategy NAME`, `:groups`, `:enable GROUP`, `:disable GROUP`, `:quit`
