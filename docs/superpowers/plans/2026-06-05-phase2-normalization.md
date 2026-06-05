# Phase 2: Theory-Driven Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Add `rerum/normalize.py`, a traceable normalization pass that turns commuted, re-associated, and like-term-laden expressions into a single canonical representative. Crucially, the pass hardcodes NO domain: which operators are associative-commutative (AC), what their identity and annihilator elements are, and how repeated operands combine, are all DATA carried by a `Theory`. Every public function is reparameterized by a `theory`. With the arithmetic theory the pass flattens nested `+`/`*` to n-ary form, sorts commutative operands by a domain-free structural order, collects like terms (`x + x` to `(* 2 x)`, `x * x` to `(^ x 2)`), folds constants using the theory units, and runs to fixpoint. With an EMPTY theory it is the identity function. With a boolean theory declaring `and`/`or` AC it flattens and sorts boolean expressions through the SAME machinery, with no engine change. It is idempotent (`normalize(normalize(e, t), t) == normalize(e, t)`) and confluent. When given a `listener`, it emits a `kind="normalize"` `RewriteStep` per transformation so simplification is explained rather than opaque.

**Architecture:** Pure functional module, no engine state. It depends only on `rerum.rewriter` (predicates `compound`, `constant`, `variable`; `ExprType`) and, for the listener path, `rerum.trace` (`RewriteStep`). It imports NOTHING from `engine.py` (avoids the circular-import constraint that `expr.py` already documents); the constant fold is local and uses the theory's identity/annihilator units rather than a prelude. The general-engine principle (spec Section 0) is the hard constraint here: the swap test (replace "arithmetic" with "boolean"; if engine code changes, it is wrong) must pass. The engine ships NO built-in theory naming `+`/`*`. The `Theory` is constructed from a plain dict or loaded from a `*.theory.json` string (data, shipped under `examples/` for the calculus/algebra demo). `ORDER_KEY` needs no theory: it is a structural total order embedding zero domain knowledge. The n-ary representation this produces is what later phases (calculus rule sets using `?rest...`) build on, so confluence here precedes the rule sets.

**Tech Stack:** Python 3.9+, pytest (config in `pyproject.toml`). `fractions.Fraction` is tolerated as a numeric atom (Phase 3 introduces it as a fold result) but Phase 2 does not require it; numeric ordering is written to accept int/float/Fraction uniformly so Phase 3 needs no change here.

**Dependency note (read before starting):** Tasks 1 through 6 (the `Theory` class and the pure transforms) have NO dependency on other phases and can proceed immediately. Task 7's listener emission assumes Phase 1 has already extended `rerum/trace.py` so `RewriteStep.__init__` accepts the keyword `kind` (default `"rule"`). If Phase 1 is not yet merged, Task 7's test fails at `RewriteStep(..., kind="normalize")` construction; complete Phases 0 and 1 first, or land Tasks 1 through 6 now and return to Task 7 after Phase 1.

---

## File Structure

```
rerum/
  normalize.py            (NEW - this phase)
  rewriter.py             (read-only reference: compound/constant/variable, ExprType)
  trace.py                (read-only reference: RewriteStep with Phase-1 kind= kwarg)
  tests/
    test_normalize.py     (NEW - this phase)
  __init__.py             (edited in Task 8: export Theory + normalize symbols)
examples/
  arithmetic.theory.json  (NEW - this phase: data, the worked-example theory)
```

`normalize.py` public surface (contract-verbatim):

```
class Theory:
    def is_ac(self, op) -> bool
    def identity(self, op)         # or None
    def annihilator(self, op)      # or None
    @classmethod
    def from_dict(cls, d) -> "Theory"
    @classmethod
    def from_json(cls, text) -> "Theory"

flatten(expr, theory) -> ExprType            # n-ary only for theory.is_ac(op)
ORDER_KEY(expr) -> tuple                     # STRUCTURAL total order, domain-free
canonical_sort(expr, theory) -> ExprType     # sort operands of ac ops by ORDER_KEY
collect_like_terms(expr, theory) -> ExprType # theory-driven repeat combination
normalize(expr, theory, *, listener=None) -> ExprType  # flatten->sort->collect->fold, fixpoint
```

**How `collect_like_terms` stays theory-driven (the substantive change).** The old
draft hardcoded "in `+` combine into `(* count x)`, in `*` combine into
`(^ x exp)`." That bakes `+`/`*`/`^` into the engine and FAILS the swap test.
Instead, the `Theory` declares, per AC operator, an OPTIONAL `repeat` rule as
data: the operator that expresses "n copies combined" and that operator's unit.
For the arithmetic theory:

```
"+": {"ac": true, "identity": 0, "repeat": {"op": "*", "via": "count"}}
"*": {"ac": true, "identity": 1, "annihilator": 0, "repeat": {"op": "^", "via": "exp"}}
```

`collect_like_terms(expr, theory)` walks each AC operator `op`, groups its
operands by structural base (via `ORDER_KEY`), and combines repeats using
`theory.repeat(op)`:
- `via="count"`: `k` copies of base `b` become `(repeat_op k b)` (so `x+x` to `(* 2 x)`),
  reading existing coefficients from `(repeat_op c b)` operands.
- `via="exp"`: `k` copies of base `b` become `(repeat_op b k)` (so `x*x` to `(^ x 2)`),
  reading existing exponents from `(repeat_op b e)` operands.
- no `repeat` declared (e.g. an idempotent boolean `and`/`or`): repeats simply
  COLLAPSE to a single copy (`(and a a)` to `(and a)` to `a`), the correct
  behavior for idempotent operators, again read from the theory not from a literal.

There is no `if op == "+"` / `if op == "*"` anywhere. The arithmetic behavior
emerges entirely from `arithmetic.theory.json`; a boolean theory drives the same
function with different (or absent) `repeat` declarations. The boolean
generality test (Task 9) proves this.

---

### Task 1: `Theory` (the data carrier)

**Files:** Create `rerum/normalize.py`; Test `rerum/tests/test_normalize.py`

Goal: a `Theory` class constructed from a dict (or JSON text) that answers
`is_ac(op)`, `identity(op)`, `annihilator(op)`, and `repeat(op)`. The engine
ships no built-in instance naming `+`/`*`; every test builds its own small
theory. An empty theory (`Theory.from_dict({})`) makes every operator non-AC.

- [ ] **Step 1: Write the failing test for `Theory`.**
  Create `rerum/tests/test_normalize.py`:

  ```python
  """Tests for theory-driven normalization (rerum/normalize.py).

  Every test constructs its own small Theory. The engine ships NO built-in
  theory naming +/*; arithmetic is just one data instance, boolean is another.
  """

  import pytest

  from rerum.normalize import (
      Theory, flatten, ORDER_KEY, canonical_sort, collect_like_terms, normalize,
  )

  # Arithmetic theory built IN-TEST from data (not from the engine).
  ARITH = Theory.from_dict({
      "+": {"ac": True, "identity": 0, "repeat": {"op": "*", "via": "count"}},
      "*": {"ac": True, "identity": 1, "annihilator": 0,
            "repeat": {"op": "^", "via": "exp"}},
  })

  # Boolean theory: a DIFFERENT data instance, same machinery.
  BOOL = Theory.from_dict({
      "and": {"ac": True, "identity": True, "annihilator": False},
      "or": {"ac": True, "identity": False, "annihilator": True},
  })

  EMPTY = Theory.from_dict({})


  class TestTheory:
      def test_is_ac_reads_data(self):
          assert ARITH.is_ac("+") is True
          assert ARITH.is_ac("*") is True
          assert ARITH.is_ac("-") is False
          assert ARITH.is_ac("dd") is False

      def test_is_ac_for_boolean(self):
          assert BOOL.is_ac("and") is True
          assert BOOL.is_ac("or") is True
          assert BOOL.is_ac("+") is False  # arithmetic ops unknown to a boolean theory

      def test_empty_theory_has_no_ac_ops(self):
          assert EMPTY.is_ac("+") is False
          assert EMPTY.is_ac("*") is False
          assert EMPTY.is_ac("and") is False

      def test_identity(self):
          assert ARITH.identity("+") == 0
          assert ARITH.identity("*") == 1
          assert ARITH.identity("-") is None
          assert BOOL.identity("and") is True

      def test_annihilator(self):
          assert ARITH.annihilator("*") == 0
          assert ARITH.annihilator("+") is None
          assert BOOL.annihilator("or") is True

      def test_repeat(self):
          assert ARITH.repeat("+") == {"op": "*", "via": "count"}
          assert ARITH.repeat("*") == {"op": "^", "via": "exp"}
          # boolean ops declare no repeat (idempotent): None.
          assert BOOL.repeat("and") is None
          assert ARITH.repeat("-") is None

      def test_from_json(self):
          import json
          t = Theory.from_json(json.dumps({"+": {"ac": True, "identity": 0}}))
          assert t.is_ac("+") is True
          assert t.identity("+") == 0
          assert t.annihilator("+") is None
  ```

- [ ] **Step 2: Run the test, expect FAIL (module missing).**
  ```bash
  pytest rerum/tests/test_normalize.py::TestTheory -v
  ```
  Expected: collection error or `ModuleNotFoundError: No module named 'rerum.normalize'`.

- [ ] **Step 3: Implement `Theory` and the module header.**
  Create `rerum/normalize.py`:

  ```python
  """Theory-driven canonical normalization: flatten, sort, collect, fold.

  A traceable normalization pass (spec Section 5.2). Pure functions over the
  nested-list ``ExprType`` from ``rewriter.py``. WHICH operators are
  associative-commutative, their identity/annihilator units, and how repeated
  operands combine, are all DATA carried by a ``Theory``; this module hardcodes
  no domain. The engine ships no built-in theory naming ``+``/``*``. The same
  functions normalize arithmetic, boolean algebra, or any AC theory with no
  change. With an empty ``Theory`` ``normalize`` is the identity.

  Idempotent: ``normalize(normalize(e, t), t) == normalize(e, t)``.
  Confluent: two orderings of the same operand multiset normalize equal.

  Imports nothing from ``engine.py`` (circular-import constraint); the constant
  fold is local and uses the theory's units.
  """

  import json as _json
  from typing import Any, Callable, Dict, List, Optional

  from .rewriter import ExprType, compound, constant, variable


  class Theory:
      """Operator signature: which ops are AC, their units, repeat rule.

      Constructed from a dict of the shape::

          {"+": {"ac": True, "identity": 0,
                 "repeat": {"op": "*", "via": "count"}},
           "*": {"ac": True, "identity": 1, "annihilator": 0,
                 "repeat": {"op": "^", "via": "exp"}}}

      An absent key means the operator is not AC and has no units. ``repeat``
      declares (as data) how to combine k copies of an operand under an AC op:
      ``via="count"`` builds ``(op k base)`` (coefficient form), ``via="exp"``
      builds ``(op base k)`` (power form). No ``repeat`` means repeats collapse
      to one copy (idempotent operators).
      """

      __slots__ = ("_sig",)

      def __init__(self, sig: Dict[str, Dict[str, Any]]):
          self._sig = dict(sig or {})

      @classmethod
      def from_dict(cls, d: Dict[str, Dict[str, Any]]) -> "Theory":
          return cls(d)

      @classmethod
      def from_json(cls, text: str) -> "Theory":
          return cls(_json.loads(text))

      def is_ac(self, op) -> bool:
          entry = self._sig.get(op)
          return bool(entry) and bool(entry.get("ac", False))

      def identity(self, op):
          entry = self._sig.get(op)
          return entry.get("identity") if entry else None

      def annihilator(self, op):
          entry = self._sig.get(op)
          return entry.get("annihilator") if entry else None

      def repeat(self, op) -> Optional[Dict[str, Any]]:
          entry = self._sig.get(op)
          return entry.get("repeat") if entry else None

      def __repr__(self) -> str:
          return f"Theory({sorted(self._sig)})"


  def _is_number(x) -> bool:
      """True for any numeric atom (int, float, Fraction). bool excluded."""
      return constant(x) and not isinstance(x, bool)
  ```

- [ ] **Step 4: Run the test, expect PASS.**
  ```bash
  pytest rerum/tests/test_normalize.py::TestTheory -v
  ```
  Expected: all `TestTheory` tests pass.

- [ ] **Step 5: Commit.**
  ```bash
  git add rerum/normalize.py rerum/tests/test_normalize.py
  git commit -m "feat(normalize): Theory data carrier (is_ac/identity/annihilator/repeat)"
  ```

---

### Task 2: `flatten(expr, theory)` (theory-driven n-ary association)

**Files:** Edit `rerum/normalize.py`; Test `rerum/tests/test_normalize.py`

Goal: `(+ (+ a b) c)` becomes `(+ a b c)` ONLY because the theory declares `+`
AC. Flattening is recursive (children flattened first) and merges a nested
operator into its parent only when the heads match AND `theory.is_ac(head)`.
Non-AC operators keep their arity; their children are still flattened
recursively. With an empty theory, nothing flattens.

- [ ] **Step 1: Write the failing test for `flatten`.**
  Append to `rerum/tests/test_normalize.py`:

  ```python
  class TestFlatten:
      def test_flatten_nested_plus(self):
          assert flatten(["+", ["+", "a", "b"], "c"], ARITH) == ["+", "a", "b", "c"]

      def test_flatten_nested_times(self):
          assert flatten(["*", ["*", "a", "b"], "c"], ARITH) == ["*", "a", "b", "c"]

      def test_flatten_right_nested(self):
          assert flatten(["+", "a", ["+", "b", "c"]], ARITH) == ["+", "a", "b", "c"]

      def test_flatten_deep(self):
          expr = ["+", ["+", ["+", "a", "b"], "c"], "d"]
          assert flatten(expr, ARITH) == ["+", "a", "b", "c", "d"]

      def test_flatten_does_not_merge_mixed_ops(self):
          assert flatten(["+", ["*", "a", "b"], "c"], ARITH) == \
              ["+", ["*", "a", "b"], "c"]

      def test_flatten_recurses_into_non_ac_ops(self):
          assert flatten(["-", ["+", ["+", "a", "b"], "c"], "d"], ARITH) == \
              ["-", ["+", "a", "b", "c"], "d"]

      def test_flatten_atom_unchanged(self):
          assert flatten("x", ARITH) == "x"
          assert flatten(5, ARITH) == 5

      def test_flatten_idempotent(self):
          once = flatten(["+", ["+", "a", "b"], "c"], ARITH)
          assert flatten(once, ARITH) == once

      def test_flatten_empty_theory_no_change(self):
          # Empty theory: no operator is AC, so no flattening happens.
          expr = ["+", ["+", "a", "b"], "c"]
          assert flatten(expr, EMPTY) == expr
  ```

- [ ] **Step 2: Run the test, expect FAIL.**
  ```bash
  pytest rerum/tests/test_normalize.py::TestFlatten -v
  ```
  Expected: `ImportError`/`AttributeError` for `flatten` (not yet defined), or
  the import line at the top fails because `flatten` is missing. Confirm RED.

- [ ] **Step 3: Implement `flatten`.**
  Add to `rerum/normalize.py` (after the `Theory` class and `_is_number`):

  ```python
  def flatten(expr: ExprType, theory: Theory) -> ExprType:
      """Recursively make AC operators n-ary, per the theory.

      ``(+ (+ a b) c)`` becomes ``(+ a b c)`` when ``theory.is_ac("+")``. A
      nested operand is merged into its parent only when their heads match and
      the head is AC under ``theory``. Children of every compound are flattened
      first. Atoms are returned unchanged. With an empty theory nothing merges.
      """
      if not compound(expr) or not expr:
          return expr

      head = expr[0]
      flat_args = [flatten(a, theory) for a in expr[1:]]

      if theory.is_ac(head):
          merged: List[ExprType] = []
          for a in flat_args:
              if compound(a) and a and a[0] == head:
                  merged.extend(a[1:])
              else:
                  merged.append(a)
          return [head] + merged

      return [head] + flat_args
  ```

- [ ] **Step 4: Run the test, expect PASS.**
  ```bash
  pytest rerum/tests/test_normalize.py::TestFlatten -v
  ```
  Expected: all `TestFlatten` tests pass.

- [ ] **Step 5: Commit.**
  ```bash
  git add rerum/normalize.py rerum/tests/test_normalize.py
  git commit -m "feat(normalize): flatten(expr, theory) n-ary association via is_ac"
  ```

---

### Task 3: `ORDER_KEY` (domain-free structural total order)

**Files:** Edit `rerum/normalize.py`; Test `rerum/tests/test_normalize.py`

Goal: a key function giving a STRICT, domain-free total order so sorting is
canonical and deterministic. It takes NO theory (it embeds no domain knowledge).
The order is: **numbers first, then symbols lexicographically, then compounds**.
Within compounds, order by `(head, then args recursively)`.

`ORDER_KEY(expr)` returns a `(rank, payload)` tuple per the contract:
- rank `0` for numbers: payload `(float(value), typename)`. The `float(value)`
  keeps int / float / `Fraction` mutually comparable; the `typename` tiebreak
  avoids ties between e.g. `int 1` and `float 1.0` while never crossing ranks.
- rank `1` for symbols (strings): payload `(name,)`.
- rank `2` for compounds: payload `(head, tuple(ORDER_KEY(arg) for arg in args))`
  where `head` is keyed the same way an atom would be, so structural comparison
  is `(head, arg0, arg1, ...)`.

Because every key begins with the integer `rank`, keys of different ranks
compare by rank alone and never reach a payload comparison between incompatible
Python types. Keys of the same rank compare by payloads built only from
(int, float, str, tuple), all mutually comparable. This makes `ORDER_KEY` a
total order with no `TypeError` and no incomparable pairs.

- [ ] **Step 1: Write the failing test for `ORDER_KEY`.**
  Append to `rerum/tests/test_normalize.py`:

  ```python
  class TestOrderKey:
      def test_numbers_before_symbols(self):
          assert ORDER_KEY(2) < ORDER_KEY("x")
          assert ORDER_KEY(0) < ORDER_KEY("a")

      def test_symbols_before_compounds(self):
          assert ORDER_KEY("z") < ORDER_KEY(["+", "a", "b"])
          assert ORDER_KEY("x") < ORDER_KEY(["*", 2, "y"])

      def test_numbers_before_compounds(self):
          assert ORDER_KEY(100) < ORDER_KEY(["+", "a", "b"])

      def test_numbers_sorted_by_value(self):
          assert ORDER_KEY(1) < ORDER_KEY(2) < ORDER_KEY(10)
          assert ORDER_KEY(-5) < ORDER_KEY(0)

      def test_symbols_lexicographic(self):
          assert ORDER_KEY("a") < ORDER_KEY("b") < ORDER_KEY("z")
          assert ORDER_KEY("x") < ORDER_KEY("xy")

      def test_compounds_by_head_then_args(self):
          assert ORDER_KEY(["+", "a", "b"]) < ORDER_KEY(["+", "b", "b"])
          assert ORDER_KEY(["*", "a"]) < ORDER_KEY(["^", "a"]) or \
              ORDER_KEY(["^", "a"]) < ORDER_KEY(["*", "a"])

      def test_total_order_no_typeerror(self):
          items = [["+", "a", "b"], "x", 3, 1, "a", ["*", 2, "y"], -1]
          ordered = sorted(items, key=ORDER_KEY)
          assert ordered[:3] == [-1, 1, 3]
          assert ordered[3:5] == ["a", "x"]
          assert all(isinstance(e, list) for e in ordered[5:])

      def test_no_theory_argument(self):
          # ORDER_KEY is domain-free: it takes only an expression.
          import inspect
          params = list(inspect.signature(ORDER_KEY).parameters)
          assert params == ["expr"]

      def test_key_is_strict(self):
          exprs = [1, 2, "a", "b", ["+", "a", "b"], ["*", "a", "b"]]
          keys = [ORDER_KEY(e) for e in exprs]
          assert len(set(keys)) == len(exprs)
  ```

- [ ] **Step 2: Run the test, expect FAIL.**
  ```bash
  pytest rerum/tests/test_normalize.py::TestOrderKey -v
  ```
  Expected: `AttributeError`/`NameError` for `ORDER_KEY` (not yet defined), or
  the top import fails. Confirm RED.

- [ ] **Step 3: Implement `ORDER_KEY`.**
  Add to `rerum/normalize.py` (after `flatten`):

  ```python
  # Rank constants: the primary sort key. Strictly increasing so that
  # numbers < symbols < compounds, regardless of payload contents.
  _RANK_NUMBER = 0
  _RANK_SYMBOL = 1
  _RANK_COMPOUND = 2


  def ORDER_KEY(expr: ExprType) -> tuple:
      """Domain-free structural total-order key for canonical sorting.

      Numbers sort before symbols before compounds. Numbers order by value
      (int/float/Fraction comparable via ``float``), symbols lexicographically,
      compounds by ``(head, then args recursively)``. The leading integer rank
      makes keys of different shapes always comparable without ``TypeError``.
      Takes no theory: this is pure structure, no domain knowledge.
      """
      if _is_number(expr):
          return (_RANK_NUMBER, (float(expr), type(expr).__name__))
      if variable(expr):
          return (_RANK_SYMBOL, (expr,))
      # compound: key by head (recursively, the normal head is a string) then args.
      head = expr[0] if expr else ""
      head_key = ORDER_KEY(head)
      arg_keys = tuple(ORDER_KEY(a) for a in expr[1:])
      return (_RANK_COMPOUND, (head_key, arg_keys))
  ```

- [ ] **Step 4: Run the test, expect PASS.**
  ```bash
  pytest rerum/tests/test_normalize.py::TestOrderKey -v
  ```
  Expected: all `TestOrderKey` tests pass.

- [ ] **Step 5: Commit.**
  ```bash
  git add rerum/normalize.py rerum/tests/test_normalize.py
  git commit -m "feat(normalize): ORDER_KEY domain-free structural total order"
  ```

---

### Task 4: `canonical_sort(expr, theory)` (sort AC operands)

**Files:** Edit `rerum/normalize.py`; Test `rerum/tests/test_normalize.py`

Goal: in operators the theory declares AC, sort operands by `ORDER_KEY`. Recurse
into all children. Non-AC heads keep operand order but their children are still
recursively sorted. `canonical_sort` assumes (but does not require) flattened
input; it sorts whatever operands are present. With an empty theory nothing is
reordered.

- [ ] **Step 1: Write the failing test for `canonical_sort`.**
  Append to `rerum/tests/test_normalize.py`:

  ```python
  class TestCanonicalSort:
      def test_sort_numbers_first_then_vars(self):
          # The contract's worked example: (+ x 2 y) -> (+ 2 x y).
          assert canonical_sort(["+", "x", 2, "y"], ARITH) == ["+", 2, "x", "y"]

      def test_sort_times(self):
          assert canonical_sort(["*", "y", "x", 3], ARITH) == ["*", 3, "x", "y"]

      def test_sort_is_stable_on_already_sorted(self):
          assert canonical_sort(["+", 2, "x", "y"], ARITH) == ["+", 2, "x", "y"]

      def test_sort_recurses(self):
          expr = ["+", ["*", "b", "a"], 1]
          assert canonical_sort(expr, ARITH) == ["+", 1, ["*", "a", "b"]]

      def test_sort_preserves_non_commutative_order(self):
          # subtraction is not AC: operands not reordered; children still sorted.
          assert canonical_sort(["-", ["*", "b", "a"], "c"], ARITH) == \
              ["-", ["*", "a", "b"], "c"]

      def test_sort_atom(self):
          assert canonical_sort("x", ARITH) == "x"
          assert canonical_sort(7, ARITH) == 7

      def test_sort_empty_theory_no_change(self):
          assert canonical_sort(["+", "x", 2, "y"], EMPTY) == ["+", "x", 2, "y"]

      def test_sort_confluent_over_permutations(self):
          import itertools
          base = ["+", "c", "a", "b", 2, 1]
          ref = canonical_sort(base, ARITH)
          for perm in itertools.permutations(["a", "b", "c", 1, 2]):
              assert canonical_sort(["+", *perm], ARITH) == ref
  ```

- [ ] **Step 2: Run the test, expect FAIL.**
  ```bash
  pytest rerum/tests/test_normalize.py::TestCanonicalSort -v
  ```
  Expected: `NameError`/`AttributeError` for `canonical_sort`. Confirm RED.

- [ ] **Step 3: Implement `canonical_sort`.**
  Add to `rerum/normalize.py`:

  ```python
  def canonical_sort(expr: ExprType, theory: Theory) -> ExprType:
      """Sort operands of AC operators by ``ORDER_KEY``, per the theory.

      Recurses into every child. Operands of AC heads are reordered into
      ``ORDER_KEY`` order; non-AC heads keep operand order. Atoms unchanged.
      """
      if not compound(expr) or not expr:
          return expr

      head = expr[0]
      sorted_args = [canonical_sort(a, theory) for a in expr[1:]]

      if theory.is_ac(head):
          sorted_args = sorted(sorted_args, key=ORDER_KEY)

      return [head] + sorted_args
  ```

- [ ] **Step 4: Run the test, expect PASS.**
  ```bash
  pytest rerum/tests/test_normalize.py::TestCanonicalSort -v
  ```
  Expected: all `TestCanonicalSort` tests pass.

- [ ] **Step 5: Commit.**
  ```bash
  git add rerum/normalize.py rerum/tests/test_normalize.py
  git commit -m "feat(normalize): canonical_sort(expr, theory) AC operands by ORDER_KEY"
  ```

---

### Task 5: `collect_like_terms(expr, theory)` (theory-driven repeat combination)

**Files:** Edit `rerum/normalize.py`; Test `rerum/tests/test_normalize.py`

Goal: in a flattened, sorted AC operator, combine repeated operands using the
theory's `repeat` rule. This is the substantive theory-driven change: there is
no `+`/`*`/`^` literal. The `repeat` declaration (data) decides whether repeats
become a count form `(repeat_op k base)`, a power form `(repeat_op base k)`, or
collapse (idempotent, no `repeat` declared).

Behavior under an AC operator `op` with `r = theory.repeat(op)`:
- Group operands by structural base via `ORDER_KEY` (input is sorted so equal
  bases are adjacent, but grouping by key is order-robust).
- Read an existing repeat-multiplicity from each operand:
  - `r["via"] == "count"`: an operand `(repeat_op k base)` with numeric `k`
    contributes count `k` to `base`; a bare `b` contributes count `1` to `b`.
  - `r["via"] == "exp"`: an operand `(repeat_op base e)` with numeric `e`
    contributes exponent `e` to `base`; a bare `b` contributes `1` to `b`.
- Re-emit each group:
  - `via == "count"`: total `k` => `base` if `k == 1`, drop if `k == 0`,
    else `[repeat_op, k, base]`.
  - `via == "exp"`: total `e` => `base` if `e == 1`, else `[repeat_op, base, e]`.
  - `r is None` (idempotent op): any group of one-or-more copies => a single
    `base` (so `(and a a)` collapses to `(and a)`).
- A head left with a single operand unwraps to that operand.
- Numeric literal operands are NOT folded here (Task 6's constant fold does
  that); `collect_like_terms` only merges repeats so it stays one explainable
  transformation. It recurses into children first.

- [ ] **Step 1: Write the failing test for `collect_like_terms`.**
  Append to `rerum/tests/test_normalize.py`:

  ```python
  class TestCollectLikeTerms:
      def test_collect_x_plus_x(self):
          # x + x -> (* 2 x) via repeat {op:*, via:count}
          assert collect_like_terms(["+", "x", "x"], ARITH) == ["*", 2, "x"]

      def test_collect_coeff_terms(self):
          # (* 2 x) + (* 3 x) -> (* 5 x)
          assert collect_like_terms(["+", ["*", 2, "x"], ["*", 3, "x"]], ARITH) == \
              ["*", 5, "x"]

      def test_collect_mixed_coeff_and_bare(self):
          # x + (* 2 x) -> (* 3 x)
          assert collect_like_terms(["+", "x", ["*", 2, "x"]], ARITH) == \
              ["*", 3, "x"]

      def test_collect_keeps_distinct_terms(self):
          assert collect_like_terms(["+", "x", "y"], ARITH) == ["+", "x", "y"]

      def test_collect_x_times_x(self):
          # x * x -> (^ x 2) via repeat {op:^, via:exp}
          assert collect_like_terms(["*", "x", "x"], ARITH) == ["^", "x", 2]

      def test_collect_power_factors(self):
          # (^ x 2) * (^ x 3) -> (^ x 5)
          assert collect_like_terms(["*", ["^", "x", 2], ["^", "x", 3]], ARITH) == \
              ["^", "x", 5]

      def test_collect_mixed_power_and_bare(self):
          # x * (^ x 2) -> (^ x 3)
          assert collect_like_terms(["*", "x", ["^", "x", 2]], ARITH) == \
              ["^", "x", 3]

      def test_collect_distinct_factors(self):
          assert collect_like_terms(["*", "x", "y"], ARITH) == ["*", "x", "y"]

      def test_collect_recurses(self):
          assert collect_like_terms(["-", ["+", "x", "x"], "y"], ARITH) == \
              ["-", ["*", 2, "x"], "y"]

      def test_collect_single_operand_unwraps(self):
          assert collect_like_terms(["+", "x"], ARITH) == "x"
          assert collect_like_terms(["*", "x"], ARITH) == "x"

      def test_collect_idempotent_op_collapses(self):
          # No repeat declared (boolean and is idempotent): (and a a) -> a.
          assert collect_like_terms(["and", "a", "a"], BOOL) == "a"
          assert collect_like_terms(["or", "x", "x", "y"], BOOL) == ["or", "x", "y"]

      def test_collect_empty_theory_no_change(self):
          # No AC op: nothing collected.
          assert collect_like_terms(["+", "x", "x"], EMPTY) == ["+", "x", "x"]
  ```

- [ ] **Step 2: Run the test, expect FAIL.**
  ```bash
  pytest rerum/tests/test_normalize.py::TestCollectLikeTerms -v
  ```
  Expected: `NameError`/`AttributeError` for `collect_like_terms`. Confirm RED.

- [ ] **Step 3: Implement `collect_like_terms` and its theory-driven helpers.**
  Add to `rerum/normalize.py`:

  ```python
  def _read_multiplicity(operand: ExprType, repeat: Dict[str, Any]):
      """Read (base, count) from an operand given the theory's repeat rule.

      ``via="count"``: ``(repeat_op k base)`` -> ``(base, k)``; bare ``b`` ->
      ``(b, 1)``. ``via="exp"``: ``(repeat_op base e)`` -> ``(base, e)``; bare
      ``b`` -> ``(b, 1)``. The shape is read from ``repeat``, never hardcoded.
      """
      rop = repeat["op"]
      via = repeat["via"]
      if compound(operand) and operand and operand[0] == rop:
          if via == "count" and len(operand) == 3 and _is_number(operand[1]):
              return operand[2], operand[1]
          if via == "exp" and len(operand) == 3 and _is_number(operand[2]):
              return operand[1], operand[2]
      return operand, 1


  def _emit_group(base: ExprType, total, repeat: Optional[Dict[str, Any]]):
      """Re-emit a collected group as an operand (or None to drop it).

      ``repeat is None`` (idempotent op): a single ``base``. ``via="count"``:
      ``base`` if total 1, ``None`` if total 0, else ``(repeat_op total base)``.
      ``via="exp"``: ``base`` if total 1, else ``(repeat_op base total)``.
      """
      if repeat is None:
          return base
      rop = repeat["op"]
      via = repeat["via"]
      if via == "count":
          if total == 0:
              return None
          if total == 1:
              return base
          return [rop, total, base]
      # via == "exp"
      if total == 1:
          return base
      return [rop, base, total]


  def _collect_ac(args: List[ExprType], op: str, theory: Theory) -> List[ExprType]:
      """Combine repeated operands under AC operator ``op`` using the theory."""
      repeat = theory.repeat(op)
      order: List[tuple] = []          # first-seen base keys
      groups: Dict[tuple, list] = {}   # key -> [total, base_expr]
      for operand in args:
          if repeat is None:
              base, count = operand, 1
          else:
              base, count = _read_multiplicity(operand, repeat)
          k = ORDER_KEY(base)
          if k not in groups:
              groups[k] = [count, base]
              order.append(k)
          else:
              groups[k][0] = groups[k][0] + count
      out: List[ExprType] = []
      for k in order:
          total, base = groups[k]
          emitted = _emit_group(base, total, repeat)
          if emitted is not None:
              out.append(emitted)
      return out


  def collect_like_terms(expr: ExprType, theory: Theory) -> ExprType:
      """Combine repeated operands of AC operators using the theory's repeat rule.

      Theory-driven, no ``+``/``*``/``^`` literal: for ``+`` (repeat ``*`` count)
      ``x + x`` -> ``(* 2 x)``; for ``*`` (repeat ``^`` exp) ``x * x`` ->
      ``(^ x 2)``; for an idempotent boolean ``and`` (no repeat) ``(and a a)`` ->
      ``a``. Recurses into children first. A head left with a single operand
      unwraps. Non-AC heads keep their operands.
      """
      if not compound(expr) or not expr:
          return expr

      head = expr[0]
      args = [collect_like_terms(a, theory) for a in expr[1:]]

      if theory.is_ac(head):
          args = _collect_ac(args, head, theory)
          if len(args) == 1:
              return args[0]
          return [head] + args

      return [head] + args
  ```

- [ ] **Step 4: Run the test, expect PASS.**
  ```bash
  pytest rerum/tests/test_normalize.py::TestCollectLikeTerms -v
  ```
  Expected: all `TestCollectLikeTerms` tests pass, including the idempotent
  boolean collapse and the empty-theory no-op.

- [ ] **Step 5: Commit.**
  ```bash
  git add rerum/normalize.py rerum/tests/test_normalize.py
  git commit -m "feat(normalize): theory-driven collect_like_terms (no +/*/^ literals)"
  ```

---

### Task 6: `normalize(expr, theory)` (flatten -> sort -> collect -> fold, to fixpoint)

**Files:** Edit `rerum/normalize.py`; Test `rerum/tests/test_normalize.py`

Goal: compose the transforms and a theory-driven constant fold, iterating to
fixpoint. The motivating example: `(+ (* 1 x) (* x 1))` normalizes to `(* 2 x)`
under `ARITH` (the inner `(* 1 x)` and `(* x 1)` fold to `x` using `*`'s identity
`1`, then `x + x` collects to `(* 2 x)`). The constant fold is local and uses the
theory's identity (and annihilator) units, so Phase 2 depends on neither
`engine.py` nor a prelude. With an EMPTY theory, `normalize` is the identity.

- [ ] **Step 1: Write the failing test for `normalize` (including the motivating example and empty-theory identity).**
  Append to `rerum/tests/test_normalize.py`:

  ```python
  class TestNormalize:
      def test_motivating_example(self):
          # (* 1 x) -> x and (* x 1) -> x by AC fold, then x + x -> (* 2 x).
          assert normalize(["+", ["*", 1, "x"], ["*", "x", 1]], ARITH) == \
              ["*", 2, "x"]

      def test_flatten_sort_collect_pipeline(self):
          # (+ (+ x 1) x) -> flatten (+ x 1 x) -> sort (+ 1 x x) -> collect (+ 1 (* 2 x))
          assert normalize(["+", ["+", "x", 1], "x"], ARITH) == \
              ["+", 1, ["*", 2, "x"]]

      def test_constant_fold_plus(self):
          assert normalize(["+", 2, 3], ARITH) == 5
          assert normalize(["+", 1, 2, 3], ARITH) == 6

      def test_constant_fold_times(self):
          assert normalize(["*", 2, 3, 4], ARITH) == 24

      def test_annihilator_zeroes_product(self):
          # 0 is the * annihilator (declared in the theory).
          assert normalize(["*", "x", 0, "y"], ARITH) == 0

      def test_commuted_forms_converge(self):
          assert normalize(["+", "x", "y"], ARITH) == normalize(["+", "y", "x"], ARITH)

      def test_associated_forms_converge(self):
          a = normalize(["+", ["+", "a", "b"], "c"], ARITH)
          b = normalize(["+", "a", ["+", "b", "c"]], ARITH)
          assert a == b == ["+", "a", "b", "c"]

      def test_power_collection(self):
          assert normalize(["*", "x", "x", "x"], ARITH) == ["^", "x", 3]

      def test_atom_unchanged(self):
          assert normalize("x", ARITH) == "x"
          assert normalize(5, ARITH) == 5

      def test_zero_drops_term(self):
          # x + 0 -> x (0 is the + identity).
          assert normalize(["+", "x", 0], ARITH) == "x"

      def test_one_drops_factor(self):
          # x * 1 -> x (1 is the * identity).
          assert normalize(["*", "x", 1], ARITH) == "x"

      def test_empty_theory_is_identity(self):
          # THE empty-theory identity guarantee.
          for e in ["x", 5, ["+", ["+", "a", "b"], "c"],
                    ["*", "x", "x"], ["+", "x", "x"]]:
              assert normalize(e, EMPTY) == e
  ```

- [ ] **Step 2: Run the test, expect FAIL.**
  ```bash
  pytest rerum/tests/test_normalize.py::TestNormalize -v
  ```
  Expected: `NameError`/`AttributeError` for `normalize`. Confirm RED.

- [ ] **Step 3: Implement the theory-driven AC constant fold and `normalize` (no listener yet).**
  Add to `rerum/normalize.py`:

  ```python
  def _fold_constants(expr: ExprType, theory: Theory) -> ExprType:
      """Fold numeric operands of AC operators using the theory's units.

      Local fold so Phase 2 needs no prelude. For each AC head, combine numeric
      operands using the operator's group op (sum for an additive identity 0,
      product for a multiplicative identity 1) inferred from the identity unit:
      we accumulate with ``+`` when ``identity == 0`` and with ``*`` when
      ``identity == 1``. A declared annihilator short-circuits the whole
      operator. Drop a result equal to the identity; unwrap a single operand.
      Recurses into children first. Non-AC heads return folded children.
      """
      if not compound(expr) or not expr:
          return expr

      head = expr[0]
      args = [_fold_constants(a, theory) for a in expr[1:]]

      if theory.is_ac(head):
          identity = theory.identity(head)
          annihilator = theory.annihilator(head)
          numbers = [a for a in args if _is_number(a)]
          rest = [a for a in args if not _is_number(a)]
          # Annihilator present among operands: the whole operator collapses.
          if annihilator is not None and any(a == annihilator for a in args):
              return annihilator
          acc = identity
          for n in numbers:
              # The accumulation op is derived from the identity unit:
              # additive identity 0 -> sum; multiplicative identity 1 -> product.
              if identity == 0:
                  acc = acc + n
              elif identity == 1:
                  acc = acc * n
              else:
                  # No numeric folding rule for this identity; keep numbers as-is.
                  rest.append(n)
                  acc = identity
          new_args: List[ExprType] = []
          if identity in (0, 1):
              if acc != identity or not rest:
                  new_args.append(acc)
              new_args.extend(rest)
          else:
              new_args = args
          if len(new_args) == 1:
              return new_args[0]
          if not new_args:
              return identity
          return [head] + new_args

      return [head] + args


  def _normalize_once(expr: ExprType, theory: Theory) -> ExprType:
      """One full pass: flatten -> sort -> collect -> fold."""
      e = flatten(expr, theory)
      e = canonical_sort(e, theory)
      e = collect_like_terms(e, theory)
      e = _fold_constants(e, theory)
      return e


  def normalize(expr: ExprType, theory: Theory, *,
                listener: Optional[Callable] = None) -> ExprType:
      """Canonical normal form under ``theory``: flatten->sort->collect->fold, to fixpoint.

      Idempotent and confluent. With an empty ``Theory`` it is the identity.
      When ``listener`` is provided, a ``kind="normalize"`` ``RewriteStep`` is
      emitted per changed sub-transformation (see Task 7).
      """
      current = expr
      for _ in range(1000):  # fixpoint bound; mirrors rewriter's iteration cap
          nxt = _normalize_once(current, theory)
          if nxt == current:
              break
          current = nxt
      return current
  ```

  Note the order: `collect_like_terms` runs before `_fold_constants` so
  `(* 1 x)` is still a product when the fold collapses it to `x`, and the next
  fixpoint pass collects `x + x`. The fixpoint loop guarantees convergence. The
  fold derives its accumulation operation from the identity unit (`0` -> sum,
  `1` -> product), so it remains theory-driven, not `+`/`*`-keyed.

- [ ] **Step 4: Run the test, expect PASS.**
  ```bash
  pytest rerum/tests/test_normalize.py::TestNormalize -v
  ```
  Expected: all `TestNormalize` tests pass, including `test_motivating_example`
  and `test_empty_theory_is_identity`.

- [ ] **Step 5: Commit.**
  ```bash
  git add rerum/normalize.py rerum/tests/test_normalize.py
  git commit -m "feat(normalize): normalize(expr, theory) fixpoint + theory-unit fold"
  ```

---

### Task 7: Listener / trace emission (`kind="normalize"`)

**Files:** Edit `rerum/normalize.py`; Test `rerum/tests/test_normalize.py`

Goal: when `normalize(expr, theory, listener=cb)` is given a callback, emit a
`RewriteStep` with `kind="normalize"` for each sub-transformation that changes
the expression. A `RewriteTrace` (itself a callable listener) is the canonical
consumer. Each emitted step carries `before`/`after` (the whole-expression edit
for that sub-step) and a `_NormalizeMeta` stub naming which sub-step fired
(`normalize:flatten`, `normalize:sort`, `normalize:collect`, `normalize:fold`),
so the trace explains the normalization rather than being opaque.

**Phase 1 dependency:** this task constructs `RewriteStep(..., kind="normalize")`.
That requires the Phase 1 extension to `RewriteStep.__init__` (additive `kind`
kwarg, default `"rule"`). If Phase 1 is unmerged, the construction raises a
`TypeError`; that is the signal to land Phase 1 first (see the dependency note).

- [ ] **Step 1: Write the failing test for listener emission.**
  Append to `rerum/tests/test_normalize.py`:

  ```python
  class TestNormalizeListener:
      def _collect_steps(self, expr, theory=ARITH):
          steps = []
          normalize(expr, theory, listener=steps.append)
          return steps

      def test_listener_receives_normalize_steps(self):
          steps = self._collect_steps(["+", ["*", 1, "x"], ["*", "x", 1]])
          assert len(steps) >= 1
          assert all(s.kind == "normalize" for s in steps)

      def test_listener_steps_have_before_after(self):
          steps = self._collect_steps(["+", ["+", "a", "b"], "c"])
          assert steps  # flatten changes it
          first = steps[0]
          assert first.before == ["+", ["+", "a", "b"], "c"]
          assert first.after is not None

      def test_listener_names_substeps(self):
          steps = self._collect_steps(["+", ["+", "a", "b"], "c"])
          names = {s.metadata.name for s in steps}
          assert "normalize:flatten" in names

      def test_listener_noop_emits_nothing(self):
          steps = self._collect_steps("x")
          assert steps == []

      def test_listener_empty_theory_emits_nothing(self):
          # Empty theory is the identity, so nothing changes, nothing emits.
          steps = self._collect_steps(["+", ["+", "a", "b"], "c"], theory=EMPTY)
          assert steps == []

      def test_rewrite_trace_consumes_steps(self):
          from rerum.trace import RewriteTrace
          trace = RewriteTrace()
          normalize(["+", "x", "x"], ARITH, listener=trace)
          assert len(trace) >= 1
          assert all(s.kind == "normalize" for s in trace)

      def test_no_listener_still_works(self):
          assert normalize(["+", "x", "x"], ARITH) == ["*", 2, "x"]
  ```

- [ ] **Step 2: Run the test, expect FAIL.**
  ```bash
  pytest rerum/tests/test_normalize.py::TestNormalizeListener -v
  ```
  Expected: failures because `_normalize_once` ignores `listener` (steps stay
  empty), or a `TypeError` at `RewriteStep(..., kind=...)` if Phase 1 is
  unmerged (see dependency note). Confirm RED.

- [ ] **Step 3: Implement listener emission.**
  Add to `rerum/normalize.py` (a tiny metadata stub and listener-aware passes),
  and route `normalize` through them:

  ```python
  class _NormalizeMeta:
      """Minimal metadata stand-in for normalize steps.

      ``RewriteStep`` reads ``metadata.name`` and ``metadata.description`` for
      its repr/``to_dict``; ``rationale`` derives from ``reasoning``/``category``.
      A full ``RuleMetadata`` is not needed for normalization steps.
      """

      __slots__ = ("name", "description", "reasoning", "category")

      def __init__(self, name: str, description: str):
          self.name = name
          self.description = description
          self.reasoning = None
          self.category = "normalize"


  def _emit(listener, name, description, before, after):
      """Emit a kind="normalize" RewriteStep if before != after."""
      if listener is None or before == after:
          return
      from .trace import RewriteStep
      step = RewriteStep(
          rule_index=-1,
          metadata=_NormalizeMeta(name, description),
          before=before,
          after=after,
          kind="normalize",
      )
      listener(step)


  def _normalize_once(expr: ExprType, theory: Theory,
                      listener: Optional[Callable] = None) -> ExprType:
      """One full pass, emitting a step per changed sub-transformation."""
      flat = flatten(expr, theory)
      _emit(listener, "normalize:flatten", "Flatten AC operators", expr, flat)

      srt = canonical_sort(flat, theory)
      _emit(listener, "normalize:sort", "Sort AC operands", flat, srt)

      col = collect_like_terms(srt, theory)
      _emit(listener, "normalize:collect", "Collect like terms", srt, col)

      fld = _fold_constants(col, theory)
      _emit(listener, "normalize:fold", "Fold constants", col, fld)

      return fld
  ```

  Then replace the Task 6 `normalize` body so it threads the listener:

  ```python
  def normalize(expr: ExprType, theory: Theory, *,
                listener: Optional[Callable] = None) -> ExprType:
      current = expr
      for _ in range(1000):
          nxt = _normalize_once(current, theory, listener=listener)
          if nxt == current:
              break
          current = nxt
      return current
  ```

  (Delete the Task 6 `_normalize_once` and `normalize` definitions, replacing
  them with these two.)

- [ ] **Step 4: Run the test, expect PASS.**
  ```bash
  pytest rerum/tests/test_normalize.py::TestNormalizeListener -v
  ```
  Expected: all `TestNormalizeListener` tests pass. Re-run `TestNormalize` to
  confirm no regression:
  ```bash
  pytest rerum/tests/test_normalize.py::TestNormalize -v
  ```

- [ ] **Step 5: Commit.**
  ```bash
  git add rerum/normalize.py rerum/tests/test_normalize.py
  git commit -m "feat(normalize): emit kind=normalize RewriteStep via listener"
  ```

---

### Task 8: `arithmetic.theory.json` data file + package exports

**Files:** Create `examples/arithmetic.theory.json`; Edit `rerum/normalize.py` (no code), `rerum/__init__.py`; Test `rerum/tests/test_normalize.py`

Goal: ship the worked-example arithmetic theory as DATA under `examples/`
(loaded via `Theory.from_json`), and export the public `normalize` API from the
package. The data file is the proof that the engine names no operators: `+`/`*`
appear only in this JSON, never in `rerum/`.

- [ ] **Step 1: Write the failing test for the data file and exports.**
  Append to `rerum/tests/test_normalize.py`:

  ```python
  class TestTheoryDataFile:
      def test_load_arithmetic_theory_from_examples(self):
          import os
          from rerum.normalize import Theory
          path = os.path.join(
              os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
              "examples", "arithmetic.theory.json",
          )
          with open(path) as f:
              theory = Theory.from_json(f.read())
          assert theory.is_ac("+") and theory.is_ac("*")
          assert theory.identity("+") == 0 and theory.identity("*") == 1
          assert theory.annihilator("*") == 0
          # The same machinery normalizes via the file-loaded theory.
          assert normalize(["+", ["*", 1, "x"], ["*", "x", 1]], theory) == \
              ["*", 2, "x"]


  class TestExports:
      def test_top_level_imports(self):
          import rerum
          assert hasattr(rerum, "Theory")
          assert hasattr(rerum, "normalize")
          assert hasattr(rerum, "flatten")
          assert hasattr(rerum, "canonical_sort")
          assert hasattr(rerum, "collect_like_terms")
          assert hasattr(rerum, "ORDER_KEY")
  ```

- [ ] **Step 2: Run the tests, expect FAIL.**
  ```bash
  pytest rerum/tests/test_normalize.py::TestTheoryDataFile rerum/tests/test_normalize.py::TestExports -v
  ```
  Expected: `TestTheoryDataFile` FAILS (`FileNotFoundError` for
  `examples/arithmetic.theory.json`); `TestExports` FAILS (symbols not
  re-exported at package top level). Confirm RED.

- [ ] **Step 3: Create the data file and add the exports.**
  Create `examples/arithmetic.theory.json`:

  ```json
  {
    "+": {"ac": true, "identity": 0, "repeat": {"op": "*", "via": "count"}},
    "*": {"ac": true, "identity": 1, "annihilator": 0, "repeat": {"op": "^", "via": "exp"}}
  }
  ```

  Edit `rerum/__init__.py`. After the `from .engine import (...)` block, add:

  ```python
  # Theory-driven normalization (Phase 2)
  from .normalize import (
      Theory,
      normalize,
      flatten,
      canonical_sort,
      collect_like_terms,
      ORDER_KEY,
  )
  ```

  Then add the six names to the `__all__` list (same style as existing entries):

  ```python
      # Theory-driven normalization
      "Theory",
      "normalize",
      "flatten",
      "canonical_sort",
      "collect_like_terms",
      "ORDER_KEY",
  ```

- [ ] **Step 4: Run the export tests, then the whole normalize file, then the full suite.**
  ```bash
  pytest rerum/tests/test_normalize.py::TestTheoryDataFile rerum/tests/test_normalize.py::TestExports -v
  pytest rerum/tests/test_normalize.py -v
  pytest
  ```
  Expected: both new classes PASS; the whole `test_normalize.py` PASS; the full
  suite green (no regressions).

- [ ] **Step 5: Commit.**
  ```bash
  git add examples/arithmetic.theory.json rerum/__init__.py rerum/tests/test_normalize.py
  git commit -m "feat(normalize): ship arithmetic.theory.json data; export Theory + API"
  ```

---

### Task 9: Generality (boolean theory) + idempotence + confluence properties

**Files:** Edit `rerum/tests/test_normalize.py`

Goal: prove the engine is domain-agnostic by driving the SAME functions with a
DIFFERENT theory (boolean `and`/`or` AC), and lock in idempotence and confluence
over many inputs. This generality test is REQUIRED: it is the executable form of
the spec Section 0 swap test.

- [ ] **Step 1: Write the failing generality and property tests.**
  Append to `rerum/tests/test_normalize.py`:

  ```python
  import itertools


  class TestBooleanGenerality:
      """The SAME machinery, driven by a boolean theory, with no engine change."""

      def test_flatten_boolean(self):
          # (and (and a b) c) flattens to (and a b c) via is_ac("and").
          assert flatten(["and", ["and", "a", "b"], "c"], BOOL) == \
              ["and", "a", "b", "c"]

      def test_canonical_sort_boolean(self):
          # (or y x) sorts to (or x y) by the same ORDER_KEY.
          assert canonical_sort(["or", "y", "x"], BOOL) == ["or", "x", "y"]

      def test_normalize_boolean_pipeline(self):
          # flatten + sort + idempotent collapse, all from the boolean theory.
          assert normalize(["and", ["and", "b", "a"], "a"], BOOL) == \
              ["and", "a", "b"]

      def test_boolean_does_not_touch_arithmetic_ops(self):
          # + is unknown to the boolean theory: left structurally intact.
          assert flatten(["+", ["+", "a", "b"], "c"], BOOL) == \
              ["+", ["+", "a", "b"], "c"]
          assert canonical_sort(["+", "y", "x"], BOOL) == ["+", "y", "x"]


  def _sample_exprs():
      """A spread of arithmetic expressions: atoms, sums, products, nesting, powers."""
      return [
          "x", 3, ["+", "x", "x"], ["*", "x", "x"],
          ["+", ["*", 1, "x"], ["*", "x", 1]],
          ["+", "c", "a", "b", 1, 2],
          ["*", "y", "x", 2, 3],
          ["+", ["+", "a", "b"], ["+", "c", "d"]],
          ["*", ["^", "x", 2], ["^", "x", 3]],
          ["-", ["+", "x", "x"], "y"],
          ["+", ["*", 2, "x"], ["*", 3, "x"], "y"],
      ]


  class TestIdempotence:
      def test_idempotent_arithmetic(self):
          for e in _sample_exprs():
              once = normalize(e, ARITH)
              assert normalize(once, ARITH) == once, f"not idempotent on {e}"

      def test_idempotent_boolean(self):
          for e in [["and", ["and", "a", "b"], "c"], ["or", "x", "x", "y"],
                    ["and", "b", "a", "a"]]:
              once = normalize(e, BOOL)
              assert normalize(once, BOOL) == once, f"not idempotent on {e}"


  class TestConfluence:
      def test_permutations_of_sum_converge(self):
          operands = ["a", "b", "c", 1, 2]
          ref = normalize(["+", *operands], ARITH)
          for perm in itertools.permutations(operands):
              assert normalize(["+", *perm], ARITH) == ref

      def test_permutations_of_product_converge(self):
          operands = ["x", "y", "z", 2]
          ref = normalize(["*", *operands], ARITH)
          for perm in itertools.permutations(operands):
              assert normalize(["*", *perm], ARITH) == ref

      def test_reassociation_converges(self):
          forms = [
              ["+", ["+", ["+", "a", "b"], "c"], "d"],
              ["+", "a", ["+", "b", ["+", "c", "d"]]],
              ["+", ["+", "a", "b"], ["+", "c", "d"]],
              ["+", "a", ["+", ["+", "b", "c"], "d"]],
          ]
          ref = normalize(forms[0], ARITH)
          for f in forms[1:]:
              assert normalize(f, ARITH) == ref
          assert ref == ["+", "a", "b", "c", "d"]

      def test_boolean_permutations_converge(self):
          # Confluence is theory-agnostic: the boolean theory converges too.
          operands = ["a", "b", "c"]
          ref = normalize(["or", *operands], BOOL)
          for perm in itertools.permutations(operands):
              assert normalize(["or", *perm], BOOL) == ref
  ```

- [ ] **Step 2: Run the tests, expect PASS (properties already hold from Tasks 1 through 8).**
  ```bash
  pytest rerum/tests/test_normalize.py::TestBooleanGenerality rerum/tests/test_normalize.py::TestIdempotence rerum/tests/test_normalize.py::TestConfluence -v
  ```
  Expected: all PASS. The boolean generality tests should pass with ZERO change
  to `rerum/normalize.py` (that is the point). If any property test fails, STOP
  and fix the underlying transform; do not weaken the test. If a boolean test
  fails, the function is leaking an arithmetic assumption (a `+`/`*`/`^` literal)
  and must be made theory-driven.

- [ ] **Step 3: Run the whole normalize file and the full suite.**
  ```bash
  pytest rerum/tests/test_normalize.py -v
  pytest
  ```
  Expected: the whole `test_normalize.py` PASS; the full suite green.

- [ ] **Step 4: Run the AC scaling experiment as a smoke check.**
  ```bash
  python experiments/scaling.py
  ```
  Expected: it runs without error (it does not import `normalize` but confirms
  the package still imports and the existing AC equivalence machinery is intact).

- [ ] **Step 5: Commit.**
  ```bash
  git add rerum/tests/test_normalize.py
  git commit -m "test(normalize): boolean-theory generality, idempotence, confluence"
  ```

---

## Phase 2 Done When

- [ ] `rerum/normalize.py` exists with a `Theory` class (`is_ac`, `identity`,
  `annihilator`, `from_dict`, `from_json`) and `flatten(expr, theory)`,
  `ORDER_KEY(expr)`, `canonical_sort(expr, theory)`,
  `collect_like_terms(expr, theory)`, `normalize(expr, theory, *, listener=None)`,
  matching the contract signatures verbatim.
- [ ] The engine ships NO built-in theory naming `+`/`*`; the arithmetic theory
  lives only in `examples/arithmetic.theory.json` (data). No `dd`/`int`/`lim`/
  `and`/`or` and no `+`/`*`/`^` literal appears as an engine special-case in
  `rerum/normalize.py` (the swap test passes by inspection).
- [ ] `collect_like_terms` is THEORY-DRIVEN: repeats combine via the theory's
  `repeat` declaration (`{op,via}`), not via `+`/`*`/`^` literals; idempotent
  operators with no `repeat` collapse repeats. The boolean generality test
  proves it works for a different AC pair.
- [ ] `ORDER_KEY` is a STRICT, DOMAIN-FREE total order taking no theory: numbers
  first (by value), then symbols lexicographically, then compounds by
  `(head, args recursively)`; `sorted(mixed, key=ORDER_KEY)` never raises.
- [ ] `flatten(["+", ["+", "a", "b"], "c"], ARITH) == ["+", "a", "b", "c"]`.
- [ ] `canonical_sort(["+", "x", 2, "y"], ARITH) == ["+", 2, "x", "y"]`.
- [ ] `collect_like_terms(["+", "x", "x"], ARITH) == ["*", 2, "x"]` and
  `collect_like_terms(["*", "x", "x"], ARITH) == ["^", "x", 2]`.
- [ ] THE motivating example holds:
  `normalize(["+", ["*", 1, "x"], ["*", "x", 1]], ARITH) == ["*", 2, "x"]`.
- [ ] EMPTY-THEORY IDENTITY: `normalize(e, Theory.from_dict({})) == e` for every
  expression in the sample set (`test_empty_theory_is_identity`).
- [ ] GENERALITY: with a boolean theory (`and`/`or` AC) and NO change to
  `normalize.py`, `flatten(["and", ["and", "a", "b"], "c"], BOOL)` flattens to
  `["and", "a", "b", "c"]` and `canonical_sort(["or", "y", "x"], BOOL) ==
  ["or", "x", "y"]` (`TestBooleanGenerality`).
- [ ] `normalize` is idempotent for both arithmetic and boolean theories
  (`TestIdempotence`).
- [ ] `normalize` is confluent: all permutations of a sum/product multiset and
  all re-associations normalize to the identical representative, for both
  arithmetic and boolean theories (`TestConfluence`).
- [ ] With a `listener`, `normalize` emits `kind="normalize"` `RewriteStep`
  events (one per changed sub-transformation), and a `RewriteTrace` consumes
  them; with no listener, behavior is unchanged.
- [ ] `Theory`, `normalize`, `flatten`, `canonical_sort`, `collect_like_terms`,
  `ORDER_KEY` are importable from the top-level `rerum` package (`TestExports`).
- [ ] `pytest` (full suite) is green; `experiments/scaling.py` runs without error.
