# DSL Reference

Complete reference for RERUM's rule definition language.

## Rule Syntax

```
@name: (pattern) => (skeleton)
@name[priority]: (pattern) => (skeleton)
@name "description": (pattern) => (skeleton)
@name {category=label}: (pattern) => (skeleton)
@name[priority] "description" {category=label}: (pattern) => (skeleton)
@name: (pattern) => (skeleton) when (condition)
@name: (lhs) <=> (rhs)
```

### Examples

```
# Simple rule
@add-zero: (+ ?x 0) => :x

# With priority (higher fires first)
@specific[100]: (+ 0 ?x) => :x

# With description
@add-zero "x + 0 = x": (+ ?x 0) => :x

# With category annotation (a free-form semantic label)
@add-zero {category=identity}: (+ ?x 0) => :x

# With guard condition
@abs-pos: (abs ?x) => :x when (! > :x 0)

# Bidirectional (reversible) rule
@comm-add: (+ ?x ?y) <=> (+ :y :x)
```

### Category Annotation

`{category=label}` attaches a free-form semantic label (`identity`,
`commutativity`, ...) to a rule. The label is carried as
`RuleMetadata.category` and is the same field JSON rules populate via a
`"category"` key. It is descriptive metadata: the engine does not act on it,
but tracing and the training layer surface it, and `engine.to_dsl()`
round-trips it.

```python
from rerum import RuleEngine

engine = RuleEngine.from_dsl("@add-zero {category=identity}: (+ ?x 0) => :x")
rule, meta = engine["add-zero"]
meta.category            # "identity"
engine.to_dsl()          # '@add-zero {category=identity}: (+ ?x 0) => :x'
```

Note: `engine.list_rules()` shows a compact view that omits the category;
`engine.to_dsl()` is the round-trippable serialization that preserves it.

### Bidirectional Rules

A `<=>` rule declares a reversible equivalence. It desugars eagerly at parse
time into TWO unidirectional rules named `<name>-fwd` and `<name>-rev`, so the
equivalence class is closed under both directions. Rule-count assertions and
`engine.list_rules()` see the post-desugar pair.

```python
from rerum import RuleEngine

engine = RuleEngine.from_dsl("@comm-add: (+ ?x ?y) <=> (+ :y :x)")
engine.list_rules()
# ['@comm-add-fwd: (+ ?x ?y) => (+ :y :x)',
#  '@comm-add-rev: (+ ?y ?x) => (+ :x :y)']
```

Bidirectional rules drive the equivalence-class reasoning (`equivalents`,
`prove_equal`, `minimize`). See the
[Equivalence & Proof](equivalence.md) guide.

## Pattern Syntax

### Basic Patterns

| Syntax | Meaning | Example Match |
|--------|---------|---------------|
| `?x` | Match anything | `(+ ?x 0)` matches `(+ y 0)` |
| `?x:expr` | Same as `?x` | - |
| `literal` | Match exact value | `0` matches `0` |
| `(op ...)` | Match compound | `(+ ?x ?y)` matches `(+ a b)` |

### Type Constraints

| Syntax | Matches | Example |
|--------|---------|---------|
| `?x:const` | Numbers only | `?n:const` matches `42` |
| `?x:var` | Symbols only | `?v:var` matches `x` |
| `?x:free(v)` | Not containing `v` | `?f:free(x)` matches `(+ y z)` |

### Rest Patterns

| Syntax | Meaning | Example |
|--------|---------|---------|
| `?xs...` | Match remaining args | `(+ ?a ?rest...)` |
| `?xs:const...` | Rest, all constants | `(+ ?nums:const...)` |
| `?xs:var...` | Rest, all variables | `(list ?vars:var...)` |

## Skeleton Syntax

### Substitution

| Syntax | Meaning |
|--------|---------|
| `:x` | Substitute bound value |
| `:xs...` | Splice list into position |
| `(fresh base)` | Gensym: a fresh name derived from `base` |
| `literal` | Use as-is |

### Computation

```
(! op arg1 arg2 ...)
```

Evaluates `op` with arguments using the prelude:

```
@fold-add: (+ ?a:const ?b:const) => (! + :a :b)
```

### Fresh Variables (Gensym)

`(fresh base)` produces a generated name that does not collide. Two important
guarantees:

1. The fresh name is NOT free in the expression being built, so it cannot
   capture a variable already present in the result.
2. Multiple `(fresh base)` markers with the SAME base resolve to DISTINCT
   names within a single instantiation: the first is `base`, the next `base1`,
   then `base2`, ... All markers are resolved together against one shared
   avoid-set, in one post-pass.

```python
from rerum import RuleEngine
from rerum.engine import format_sexpr

# Two same-base markers -> distinct names u, u1
engine = RuleEngine.from_dsl("@two-fresh: (gen ?x) => (pair (fresh u) (fresh u))")
format_sexpr(engine(["gen", "z"]))   # "(pair u u1)"

# Avoids a name already free in the result: body already contains u, so the
# fresh marker resolves to u1 instead.
engine2 = RuleEngine.from_dsl("@bind: (bind ?body) => (lam (fresh u) :body)")
format_sexpr(engine2(["bind", "u"]))  # "(lam u1 u)"
```

This is what makes rules that introduce bound variables (lambda-binders,
substitution targets) safe under repeated rewriting.

## Groups

```
[groupname]
@rule1: ...
@rule2: ...

[another-group]
@rule3: ...
```

Rules after `[groupname]` belong to that group until another group starts.

## Directives

A `.rules`/`.manifest` file may carry `:`-prefixed directives. They live in a
namespace distinct from rules and groups. The loader skips them when scanning
for rules, so a directive never mis-parses as a rule.

### `:include`

`:include <path>` splices another rule file in place. Paths are resolved
relative to the including file. Circular includes are detected and rejected.

```
# main.rules
:include base.rules
@mul-one: (* ?x 1) => :x
```

```python
from rerum import RuleEngine
engine = RuleEngine.from_file("main.rules")   # base.rules is loaded too
```

## Rule-set Manifests

A *manifest* is a DSL file whose `:`-directives declare a domain's full
loading contract -- which prelude bundles it needs, which fold ops, which
theory and metadata sidecar, and how it is meant to be driven. Assemble the
whole domain from one file with `RuleEngine.from_manifest(path)`: it installs
the preludes, sets the theory, loads the rules (typically via an `:include`
body), merges the metadata sidecar, and FAILS LOUD if a required fold op is
missing (the silent-junk footgun where a skeleton `(! op ...)` with an unknown
`op` survives as a literal compound).

### Manifest Directives

| Directive | Meaning |
|-----------|---------|
| `:requires <bundle...>` | Named prelude bundles to combine left-to-right: `none`, `minimal`, `arithmetic`, `math`, `predicate`, `full` |
| `:requires-ops <op...>` | Fold-op names the rules need; the assembler verifies each is present after preludes combine |
| `:theory <path>` | A theory JSON file (relative), parsed via `Theory.from_json` and installed as the session theory |
| `:metadata <path>` | A metadata sidecar JSON (relative), merged via `load_metadata_json` after rules load |
| `:driver simplify\|solve` | A HINT (data only): how the domain is meant to be driven. The engine stores it; a caller may read it |
| `:goal <json>` | A goal-description HINT (data only), e.g. `{"op_free": ["int"]}` |

All are optional and may appear alongside `:include`. The list-valued
directives (`:requires`, `:requires-ops`) accumulate across repeats; the
single-valued ones (`:theory`, `:metadata`, `:driver`, `:goal`) raise on a
duplicate. `:driver`/`:goal` are stored as data; the engine does not act on
them in v1.

### Worked Example

`examples/differentiation.manifest` assembles the whole differentiation
pipeline in one call:

```
# Differentiation domain manifest.
:requires math predicate
:theory arithmetic.theory.json
:metadata differentiation.metadata.json
:driver simplify
:include differentiation.rules
```

```python
from rerum import RuleEngine
from rerum.engine import format_sexpr

engine = RuleEngine.from_manifest("examples/differentiation.manifest")
format_sexpr(engine(["dd", "x", "x"]))   # "1"  (d/dx of x)

engine.manifest.driver                   # "simplify"  (the stored hint)
```

A no-prelude domain looks like `examples/boolean.manifest`
(`:requires none`, a theory + metadata sidecar, `:include boolean.rules`):
every rewrite is structural.

### Parsing and Auditing

`parse_manifest(text)` is the pure parse half (no engine state). It raises
`ValueError` on a malformed directive: an unknown bundle in `:requires`, an
unknown `:driver` value, non-JSON `:goal`, a duplicate single-valued
directive, or an unknown `:`-directive head (so a typo like `:requies` is
caught, not silently ignored).

```python
from rerum.manifest import parse_manifest

m = parse_manifest('''
:requires math predicate
:requires-ops dd
:driver solve
:goal {"op_free": ["int"]}
:include differentiation.rules
''')
m.requires       # ('math', 'predicate')
m.requires_ops   # ('dd',)
m.driver         # 'solve'
m.goal           # {'op_free': ['int']}
```

The fail-loud audit is also available standalone:
`engine.missing_fold_ops()` returns the names of `(! op ...)` heads (in
skeletons and guards) plus any `:requires-ops` declaration that the installed
prelude does not provide.

```python
from rerum import RuleEngine, ARITHMETIC_PRELUDE

engine = (RuleEngine().with_prelude(ARITHMETIC_PRELUDE)
          .load_dsl("@r: (f ?x) => (! frobnicate :x)", validate_examples=False))
engine.missing_fold_ops()    # ['frobnicate']
```

Plain `RuleEngine.from_file` of a manifest applies NOTHING from it (no
prelude/theory/sidecar) -- loading a file must not silently mutate the
engine's prelude -- but it does store the parsed contract on
`engine.manifest` so a caller can inspect it. Assembly happens only via the
explicit `from_manifest`.

## Atoms and Literals

- Integers (`3`, `-7`) and floats (`2.5`) parse as Python numbers.
- Rational literals `p/q` (integer numerator and denominator) parse to an
  exact `Fraction` atom, Scheme-style: `1/3` is `Fraction(1, 3)`, and
  `format_sexpr(Fraction(1, 3))` renders `1/3`, so the round-trip is
  exact. An int-valued literal (`4/2`) narrows to the int `2`. A zero
  denominator (`1/0`) or non-integer parts (`x/y`, `1/x`) stay plain
  symbols. Note `(/ 1 3)` is something different: a division EXPRESSION.
- Everything else is a symbol (including `true`/`false` -- there is no
  boolean literal; rule sets use symbol constants, see
  `examples/boolean.rules`).

## Comments

```
# This is a comment
@rule: (+ ?x 0) => :x  # Inline comments work too
```

## Available Preludes

### ARITHMETIC_PRELUDE

Operations: `+`, `-`, `*`, `/`, `^`

### MATH_PRELUDE

Arithmetic plus: `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `abs`

### PREDICATE_PRELUDE

Comparisons: `>`, `<`, `>=`, `<=`, `=`, `!=`
Type checks: `const?`, `var?`, `list?`
Logical: `and`, `or`, `not`

### FULL_PRELUDE

Combines ARITHMETIC_PRELUDE and PREDICATE_PRELUDE.

## Complete Example

```
# algebra.rules - Algebraic simplification

[identity]
@add-zero[100]: (+ ?x 0) => :x
@add-zero-left[100]: (+ 0 ?x) => :x
@mul-one[100]: (* ?x 1) => :x
@mul-zero[100]: (* ?x 0) => 0

[folding]
@fold-add: (+ ?a ?b) => (! + :a :b) when (! and (! const? :a) (! const? :b))
@fold-mul: (* ?a ?b) => (! * :a :b) when (! and (! const? :a) (! const? :b))

[simplify]
@add-same: (+ ?x ?x) => (* 2 :x)
@sub-same: (- ?x ?x) => 0
```
