# Examples

Real-world examples of RERUM usage.

## Algebraic Simplification

```python
from rerum import RuleEngine, E, FULL_PRELUDE

algebra = (RuleEngine()
    .with_prelude(FULL_PRELUDE)
    .load_dsl('''
        [identity]
        @add-zero[100]: (+ ?x 0) => :x
        @add-zero-left[100]: (+ 0 ?x) => :x
        @mul-one[100]: (* ?x 1) => :x
        @mul-one-left[100]: (* 1 ?x) => :x
        @mul-zero[100]: (* ?x 0) => 0

        [folding]
        @fold-add: (+ ?a ?b) => (! + :a :b) when (! and (! const? :a) (! const? :b))
        @fold-mul: (* ?a ?b) => (! * :a :b) when (! and (! const? :a) (! const? :b))

        [simplify]
        @add-same: (+ ?x ?x) => (* 2 :x)
        @sub-same: (- ?x ?x) => 0
    '''))

algebra(E("(+ (* x 1) 0)"))      # => "x"
algebra(E("(+ 2 3)"))            # => 5
algebra(E("(+ x x)"))            # => (* 2 x)
```

## Symbolic Differentiation

```python
calculus = RuleEngine.from_dsl('''
    [basic]
    @dd-const[100]: (dd ?c:const ?v:var) => 0
    @dd-var-same[100]: (dd ?x:var ?x) => 1
    @dd-var-diff[90]: (dd ?y:var ?x:var) => 0

    [linear]
    @dd-sum: (dd (+ ?f ?g) ?v:var) => (+ (dd :f :v) (dd :g :v))
    @dd-const-mult: (dd (* ?c:const ?f) ?v:var) => (* :c (dd :f :v))

    [product]
    @dd-product: (dd (* ?f ?g) ?v:var) => (+ (* (dd :f :v) :g) (* :f (dd :g :v)))

    [power]
    @dd-power: (dd (^ ?f ?n:const) ?v:var) => (* :n (* (^ :f (- :n 1)) (dd :f :v)))
''')

calculus(E("(dd x x)"))              # => 1
calculus(E("(dd (^ x 2) x)"))        # => (* 2 (* (^ x (- 2 1)) 1))
calculus(E("(dd (+ x y) x)"))        # => (+ 1 0)
```

## Phased Processing

```python
# Expand then simplify
expand = RuleEngine.from_dsl('''
    @square: (square ?x) => (* :x :x)
    @cube: (cube ?x) => (* :x (* :x :x))
''')

simplify = (RuleEngine()
    .with_prelude(FULL_PRELUDE)
    .load_dsl('''
        @fold: (* ?a ?b) => (! * :a :b) when (! and (! const? :a) (! const? :b))
    '''))

pipeline = expand >> simplify

pipeline(E("(square 5)"))   # => 25
pipeline(E("(cube 3)"))     # => 27
```

## Boolean Logic

```python
logic = RuleEngine.from_dsl('''
    [identity]
    @and-true[100]: (and true ?x) => :x
    @and-false[100]: (and false ?x) => false
    @or-true[100]: (or true ?x) => true
    @or-false[100]: (or false ?x) => :x
    @not-not[100]: (not (not ?x)) => :x

    [deMorgan]
    @demorgan-and: (not (and ?x ?y)) => (or (not :x) (not :y))
    @demorgan-or: (not (or ?x ?y)) => (and (not :x) (not :y))

    [simplify]
    @and-same: (and ?x ?x) => :x
    @or-same: (or ?x ?x) => :x
''')

logic(E("(and true x)"))                # => x
logic(E("(not (not p))"))               # => p
logic(E("(not (and a b))"))             # => (or (not a) (not b))
```

## Lambda Calculus

```python
lambda_calc = RuleEngine.from_dsl('''
    # Beta reduction (simplified - assumes no capture)
    @beta: (app (lam ?x ?body) ?arg) => (subst :body :x :arg)

    # Substitution rules
    @subst-var-same: (subst ?x ?x ?e) => :e
    @subst-var-diff: (subst ?y:var ?x:var ?e) => :y
    @subst-const: (subst ?c:const ?x ?e) => :c
    @subst-lam: (subst (lam ?y ?body) ?x ?e) => (lam :y (subst :body :x :e))
    @subst-app: (subst (app ?f ?a) ?x ?e) => (app (subst :f :x :e) (subst :a :x :e))
''')

# (λx.x) y => y
lambda_calc(E("(app (lam x x) y)"))

# (λx.(λy.x)) a b => a
expr = E("(app (app (lam x (lam y x)) a) b)")
lambda_calc(expr)
```

## Selective Simplification

```python
engine = RuleEngine.from_dsl('''
    [expand]
    @square: (square ?x) => (* :x :x)

    [collect]
    @add-same: (+ ?x ?x) => (* 2 :x)
''')

expr = E("(+ (square x) (square x))")

# Only expand
engine(expr, groups=["expand"])     # => (+ (* x x) (* x x))

# Only collect
engine(expr, groups=["collect"])    # => (* 2 (square x))

# Both
engine(expr)                        # => (* 2 (* x x))
```

## Custom Operations

```python
# my_prelude.py
from rerum import binary_only, unary_only
import math

PRELUDE = {
    "gcd": binary_only(math.gcd),
    "lcm": binary_only(math.lcm),
    "factorial": unary_only(math.factorial),
    "even?": unary_only(lambda x: x % 2 == 0),
    "odd?": unary_only(lambda x: x % 2 == 1),
}
```

```python
from my_prelude import PRELUDE

number_theory = (RuleEngine()
    .with_prelude(PRELUDE)
    .load_dsl('''
        @eval-gcd: (gcd ?a ?b) => (! gcd :a :b) when (! and (! const? :a) (! const? :b))
        @eval-lcm: (lcm ?a ?b) => (! lcm :a :b) when (! and (! const? :a) (! const? :b))
        @eval-factorial: (factorial ?n) => (! factorial :n) when (! const? :n)

        @gcd-same: (gcd ?x ?x) => :x
        @lcm-same: (lcm ?x ?x) => :x
    '''))

number_theory(E("(gcd 12 8)"))      # => 4
number_theory(E("(factorial 5)"))   # => 120
```

## Proving Boolean Identities

Bidirectional rules let you prove expressions are equivalent by searching
for a common form from both sides.

```python
from rerum import RuleEngine, format_sexpr

logic = RuleEngine.from_dsl('''
    @demorgan-and: (not (and ?x ?y)) <=> (or (not :x) (not :y))
    @demorgan-or:  (not (or  ?x ?y)) <=> (and (not :x) (not :y))
    @double-neg:   (not (not ?x)) <=> :x
    @comm-and:     (and ?x ?y) <=> (and :y :x)
    @comm-or:      (or  ?x ?y) <=> (or  :y :x)
''')

proof = logic.prove_equal(
    ["not", ["and", "p", "q"]],
    ["or", ["not", "p"], ["not", "q"]],
)
print(format_sexpr(proof.common))
# (or (not p) (not q))
```

## Minimizing Algebraic Expressions

`minimize()` finds the smallest equivalent expression under a cost metric.
It applies both `=>` and `<=>` rules by default, so you can throw in
commutativity and it will explore permutations to find simplifiable forms.

```python
algebra = RuleEngine.from_dsl('''
    @add-zero: (+ ?x 0) => :x
    @mul-one:  (* ?x 1) => :x
    @mul-zero: (* ?x 0) => 0
    @comm-add: (+ ?x ?y) <=> (+ :y :x)
    @comm-mul: (* ?x ?y) <=> (* :y :x)
''')

result = algebra.minimize(
    ["*", ["+", "x", 0], ["+", "y", 0]],
    metric="size",
)
print(format_sexpr(result.expr))   # (* x y)
print(f"{result.improvement_ratio:.0%} smaller")   # 57%
```

## Enumerating Equivalence Classes

Under assoc + commute, `a + b + c + d` has exactly 120 distinct forms:

```python
assoc_comm = RuleEngine.from_dsl('''
    @assoc: (+ (+ ?x ?y) ?z) <=> (+ :x (+ :y :z))
    @comm:  (+ ?x ?y) <=> (+ :y :x)
''')

forms = assoc_comm.enumerate_equivalents(
    ["+", ["+", "a", "b"], ["+", "c", "d"]],
    max_depth=10,
)
print(len(forms))   # 120
```

For larger sums, use `prove_equal` with a `max_expressions` budget rather
than full enumeration.

## Tracing Derivations

```python
engine = RuleEngine.from_dsl('''
    @add-zero: (+ ?x 0) => :x
    @mul-one: (* ?x 1) => :x
    @mul-zero: (* ?x 0) => 0
''')

result, trace = engine(E("(+ (* (* x 0) 1) 0)"), trace=True)

print(trace)
# Initial: (+ (* (* x 0) 1) 0)
#   1. mul-zero: (* x 0) → 0
#   2. mul-one: (* 0 1) → 0
#   3. add-zero: (+ 0 0) → 0
# Final: 0

print(trace.format("compact"))
# (+ (* (* x 0) 1) 0) --[mul-zero, mul-one, add-zero]--> 0

print(trace.summary())
# 3 steps using 3 unique rules. Most used: mul-zero (1x)
```
