# RERUM

**Rewriting Expressions via Rules Using Morphisms**

A pattern matching and term rewriting library for symbolic computation in
Python. One GENERAL engine -- rules are data, domains are content.

## Features

- **Declarative Rules** - Define rewrite rules in a simple DSL (guards,
  priorities, groups, categories, `(fresh u)` gensym)
- **Pattern Matching** - Match expressions with variables, constants, type
  constraints, and rest patterns
- **Bidirectional Rules** - Use `<=>` for reversible equivalences
- **Equivalence & Proof** - Enumerate equivalent forms, prove equality by
  bidirectional search, minimize by cost
- **Goal-Directed Search** (`solve`) - Best-first search with caller-supplied
  goals and per-operator costs, for non-confluent rule sets
- **Theories** (`normalize`) - Declare operators associative-commutative with
  identities/annihilators as JSON data; get canonical forms with no engine
  changes per domain
- **Numeric Verification** (`numeval`, `numeric_equiv`) - Evaluate ground
  terms over a prelude; check expression equivalence by sampling
- **Rule-set Manifests** - Self-describing rule files that assemble a whole
  domain (preludes + theory + metadata + rules) from one file
- **MCP Server** (`rerum-mcp`) - Typed tools exposing the engine to LLM agents
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
pip install "rerum[mcp]"   # with the MCP server
```

## A General Engine

The engine names no domain operator. Every domain under `examples/` --
calculus (differentiation, integration, limits, numerically certified),
boolean algebra, set algebra, Peano arithmetic, SKI combinators -- is rules +
data the engine loads. Beyond reduction to a normal form, RERUM does
goal-directed search, equivalence-class reasoning, theory-based
canonicalization, and numeric verification.

```python
# Goal-directed search for a non-confluent rule set
from rerum.solve import solve, contains_op
result = solve(engine, ["int", ["cos", "x"], "x"],
               lambda e: not contains_op(e, {"int"}), max_nodes=2000)

# Theory-based canonical forms (operator signatures are DATA)
from rerum.normalize import Theory, normalize
theory = Theory.from_json('{"+": {"ac": true, "identity": 0}}')
normalize(["+", "b", "a", 0], theory)   # ["+", "a", "b"]

# Numeric verification
from rerum.numeval import numeric_equiv
from rerum import MATH_PRELUDE
numeric_equiv(["*", "x", 2], ["+", "x", "x"], {"x": (0.5, 2.0)}, MATH_PRELUDE)
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
- [DSL Reference](dsl-reference.md) - Complete syntax guide, including
  manifests
- [Equivalence & Proof](equivalence.md) - Bidirectional rules, `prove_equal`,
  `minimize`
- [CLI](cli.md) - Command-line interface
- [API Reference](api-reference.md) - Python API
- [Examples](examples.md) - Real-world examples and the `examples/` domain
  library

For goal-directed search (`solve`), theories (`normalize`), numeric
verification (`numeval`/`numeric_equiv`), the training corpora layer, and the
MCP server, see the [README](https://github.com/queelius/rerum#readme), which
carries runnable sections for each.
