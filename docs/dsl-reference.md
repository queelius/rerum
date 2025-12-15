# DSL Reference

Complete reference for RERUM's rule definition language.

## Rule Syntax

```
@name: (pattern) => (skeleton)
@name[priority]: (pattern) => (skeleton)
@name "description": (pattern) => (skeleton)
@name[priority] "description": (pattern) => (skeleton)
@name: (pattern) => (skeleton) when (condition)
```

### Examples

```
# Simple rule
@add-zero: (+ ?x 0) => :x

# With priority (higher fires first)
@specific[100]: (+ 0 ?x) => :x

# With description
@add-zero "x + 0 = x": (+ ?x 0) => :x

# With guard condition
@abs-pos: (abs ?x) => :x when (! > :x 0)
```

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
| `literal` | Use as-is |

### Computation

```
(! op arg1 arg2 ...)
```

Evaluates `op` with arguments using the prelude:

```
@fold-add: (+ ?a:const ?b:const) => (! + :a :b)
```

## Groups

```
[groupname]
@rule1: ...
@rule2: ...

[another-group]
@rule3: ...
```

Rules after `[groupname]` belong to that group until another group starts.

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
