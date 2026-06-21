# F3: AC-Matching Proper -- Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make rule matching work modulo associative-commutative operators so an AC rule fires on any arrangement of the subject, closing the F1 soundness boundary pinned in `test_theory_reasoning.py`.

**Architecture:** A new pure module `rerum/acmatch.py` exports a multi-valued `ac_match` generator (atom/variable/non-AC cases plus an AC multiset-partition backtracker with a work-budget fail-safe). The engine gains one seam, `_match_lhs`, that yields `match()`'s single result when no AC theory is loaded and `ac_match`'s several results under an AC theory; `apply_once` (first match) and `_all_single_rewrites` (all matches) consume it. `match()` and the no-theory fast path are untouched.

**Tech Stack:** Python 3.9+, pytest. Reuses `rerum/rewriter.py` matcher predicates and `Bindings`, and `rerum/normalize.py` `flatten`/`ORDER_KEY`/`normalize`/`Theory`.

---

## File Structure

- **Create** `rerum/acmatch.py` -- pure AC matcher: `MatchBudget`, `ac_match`, internal helpers (`_bind`, `_canon_eq`, `_ac_assignments`). Names NO domain operator; keys on `theory.is_ac`.
- **Create** `rerum/tests/test_acmatch.py` -- unit + property tests for the pure matcher.
- **Modify** `rerum/normalize.py` -- add `Theory.has_ac()`.
- **Modify** `rerum/engine.py` -- add `_match_lhs`, the `_ac_match_truncated` flag + property, rewire `apply_once` and `_all_single_rewrites` to loop over `_match_lhs`, add the AC condition to `simplify`'s fast-path gate.
- **Modify** `rerum/__init__.py` -- re-export `ac_match`, `MatchBudget`.
- **Modify** `rerum/tests/test_theory_reasoning.py` -- flip the pinned boundary test to assert a real proof.
- **Create** `examples/ac_demo.rules` + `examples/ac_demo.theory.json` -- light demonstration that one AC rule fires across arrangements (driven through the general engine in a test).

## Conventions all tasks follow

- **ASCII only** in every file write (a commit hook rejects non-ASCII such as em-dashes or curly quotes). Use `->`, `<=`, plain hyphens.
- **Never stage `.mcp.json`** (untracked local file). Stage only the files each task names.
- Commit messages end with the trailer:
  ```
  Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
  ```
- DSL skeletons use `:x` (substitute), patterns use `?x`. Expressions are nested lists in prefix form: `["+", "a", "b"]`.
- The general-engine principle: `acmatch.py` must contain NO domain operator literal as code. `test_mcp_no_domain.py` checks `rerum/mcp/` specifically, but keep `acmatch.py` clean by the same standard.

---

### Task 1: `MatchBudget` and `Theory.has_ac()`

The two small primitives the matcher and the seam need: a fail-safe work budget, and a way to ask a `Theory` whether any operator is AC.

**Files:**
- Create: `rerum/acmatch.py`
- Create: `rerum/tests/test_acmatch.py`
- Modify: `rerum/normalize.py` (add method to the `Theory` class, after `is_ac` around line 62)

- [ ] **Step 1: Write the failing tests**

Create `rerum/tests/test_acmatch.py`:

```python
"""F3: AC-matching proper (matching modulo associativity/commutativity)."""

from rerum import acmatch as am
from rerum.normalize import Theory


class TestMatchBudget:
    def test_spend_decrements_and_flags_truncation(self):
        b = am.MatchBudget(steps=2)
        assert b.spend() is True       # 2 -> 1, still has budget
        assert b.spend() is True       # 1 -> 0, this call consumes the last
        assert b.spend() is False      # exhausted
        assert b.truncated is True

    def test_unbounded_when_none_steps(self):
        b = am.MatchBudget(steps=None)
        for _ in range(1000):
            assert b.spend() is True
        assert b.truncated is False


class TestTheoryHasAC:
    def test_has_ac_true_when_any_ac_op(self):
        assert Theory.from_dict({"+": {"ac": True}}).has_ac() is True

    def test_has_ac_false_when_no_ac_op(self):
        assert Theory.from_dict({"-": {"identity": 0}}).has_ac() is False
        assert Theory.from_dict({}).has_ac() is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest rerum/tests/test_acmatch.py -v`
Expected: FAIL (`No module named 'rerum.acmatch'`, and `Theory has no attribute 'has_ac'`).

- [ ] **Step 3: Add `Theory.has_ac()`**

In `rerum/normalize.py`, in the `Theory` class immediately after the `is_ac` method (around line 62), add:

```python
    def has_ac(self) -> bool:
        """True if any operator in this theory is declared AC."""
        return any(bool(entry.get("ac", False)) for entry in self._sig.values())
```

- [ ] **Step 4: Create `rerum/acmatch.py` with `MatchBudget`**

Create `rerum/acmatch.py`:

```python
"""F3: AC-matching proper -- matching modulo associativity and commutativity.

This module is the multiset-partition matcher that F1's normalize pass cannot
provide (canonical order is not pattern unification). It is PURE: it takes the
``Theory`` carrier (which declares, as DATA, which operators are AC) and reuses
the matcher predicates from ``rewriter.py`` plus ``flatten``/``ORDER_KEY``/
``normalize`` from ``normalize.py``. It names NO domain operator.

``ac_match`` is MULTI-VALUED: an AC pattern can match a subject several ways, so
it yields each consistent ``Bindings`` lazily. ``match()`` in ``rewriter.py`` is
unchanged; the engine routes to ``ac_match`` only when an AC theory is loaded.

Scope: MATCHING only (pattern has variables, subject is a concrete term). NOT
AC-unification (both sides have variables), which F2/F5 would need. NOT ACU
(matching modulo identity).
"""

from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass
class MatchBudget:
    """Fail-safe work budget for AC enumeration.

    ``steps`` counts assignment attempts; each ``spend()`` decrements it. When it
    reaches zero, further ``spend()`` calls return False and ``truncated`` is set,
    so the matcher stops enumerating. ``steps=None`` means unbounded. Truncation
    bounds COMPLETENESS only: every match already yielded is still valid.
    """

    steps: Optional[int] = None
    truncated: bool = False

    def spend(self) -> bool:
        """Consume one unit. Return True if budget remained, False if exhausted."""
        if self.steps is None:
            return True
        if self.steps <= 0:
            self.truncated = True
            return False
        self.steps -= 1
        return True
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest rerum/tests/test_acmatch.py -v`
Expected: PASS (4 passed).

- [ ] **Step 6: ASCII check and commit**

Run: `LC_ALL=C grep -n '[^[:print:][:space:]]' rerum/acmatch.py rerum/normalize.py rerum/tests/test_acmatch.py && echo "FOUND non-ASCII" || echo "clean"`
Expected: `clean`.

```bash
git add rerum/acmatch.py rerum/normalize.py rerum/tests/test_acmatch.py
git commit -m "feat(f3): MatchBudget fail-safe + Theory.has_ac

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `ac_match` -- non-AC cases (multi-valued generalization of `match`)

Build `ac_match` for everything EXCEPT the AC multiset case: atoms, single-variable patterns, and non-AC compounds (positional, in lockstep, but multi-valued because a child may later be an AC node). On non-AC patterns this must agree with `match()`.

**Files:**
- Modify: `rerum/acmatch.py`
- Test: `rerum/tests/test_acmatch.py`

- [ ] **Step 1: Write the failing tests**

Append to `rerum/tests/test_acmatch.py`:

```python
from rerum.rewriter import Bindings


def _matches(pat, exp, theory):
    """All binding dicts ac_match yields, as a list of plain dicts."""
    return [b.to_dict() for b in am.ac_match(pat, exp, theory)]


NO_AC = Theory.from_dict({})


class TestNonACCases:
    def test_literal_match_and_mismatch(self):
        assert _matches("a", "a", NO_AC) == [{}]
        assert _matches("a", "b", NO_AC) == []

    def test_single_variable_binds_whole_expr(self):
        assert _matches(["?", "x"], ["f", "a"], NO_AC) == [{"x": ["f", "a"]}]

    def test_typed_variable_constraints(self):
        assert _matches(["?c", "n"], "3", NO_AC) == [{"n": "3"}]
        assert _matches(["?c", "n"], "x", NO_AC) == []       # x is not constant
        assert _matches(["?v", "s"], "x", NO_AC) == [{"s": "x"}]
        assert _matches(["?v", "s"], "3", NO_AC) == []       # 3 is not a variable

    def test_non_ac_compound_positional(self):
        # (f ?x ?y) against (f a b): exactly one match, positional.
        assert _matches(["f", ["?", "x"], ["?", "y"]], ["f", "a", "b"], NO_AC) == \
            [{"x": "a", "y": "b"}]

    def test_non_ac_head_mismatch(self):
        assert _matches(["f", ["?", "x"]], ["g", "a"], NO_AC) == []

    def test_non_linear_consistency(self):
        # (f ?x ?x) matches (f a a) but not (f a b).
        assert _matches(["f", ["?", "x"], ["?", "x"]], ["f", "a", "a"], NO_AC) == \
            [{"x": "a"}]
        assert _matches(["f", ["?", "x"], ["?", "x"]], ["f", "a", "b"], NO_AC) == []

    def test_agrees_with_syntactic_match_on_non_ac(self):
        from rerum.rewriter import match
        pat = ["f", ["?", "x"], ["g", ["?", "y"]]]
        exp = ["f", "a", ["g", "b"]]
        syntactic = match(pat, exp)
        ac = list(am.ac_match(pat, exp, NO_AC))
        assert len(ac) == 1 and ac[0].to_dict() == syntactic.to_dict()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest rerum/tests/test_acmatch.py::TestNonACCases -v`
Expected: FAIL (`ac_match` not defined).

- [ ] **Step 3: Implement the non-AC dispatch**

In `rerum/acmatch.py`, add imports at the top (after the existing `from typing import ...`):

```python
from .rewriter import (
    Bindings,
    atom,
    compound,
    car,
    arbitrary_constant,
    arbitrary_variable,
    arbitrary_expression,
    arbitrary_free,
    arbitrary_rest,
    variable_name,
    constant,
    variable,
    free_in,
)
from .normalize import flatten, ORDER_KEY, normalize
```

Then add the binding helpers and the matcher core:

```python
def _canon_eq(a, b, theory) -> bool:
    """Equality of two terms modulo the theory's canonical form."""
    if a == b:
        return True
    return normalize(a, theory) == normalize(b, theory)


def _bind(bindings: Bindings, name: str, value, theory) -> Optional[Bindings]:
    """Extend ``bindings`` with name->value, consistent MODULO the theory.

    Returns the (possibly same) Bindings on success, or None on a conflicting
    re-binding. Unlike ``Bindings.extend``, re-binding compares modulo AC so
    ``?x`` bound to ``(+ a b)`` still matches ``(+ b a)``.
    """
    if name in bindings:
        return bindings if _canon_eq(bindings[name], value, theory) else None
    return bindings.extend(name, value)


def _match_one(pat, exp, theory, bindings: Bindings) -> Optional[Bindings]:
    """Single-result match for a NON-rest, NON-AC-node pattern element.

    Handles atoms, the four single-variable forms, and delegates compounds to
    the first ac_match yield (a compound element has at most one match here in
    the non-AC path, but may be AC and multi-valued; callers that need every
    match use ac_match directly). Returns Bindings or None.
    """
    if atom(pat):
        return bindings if (atom(exp) and pat == exp) else None
    if arbitrary_constant(pat):
        return _bind(bindings, variable_name(pat), exp, theory) if constant(exp) else None
    if arbitrary_variable(pat):
        return _bind(bindings, variable_name(pat), exp, theory) if variable(exp) else None
    if arbitrary_expression(pat):
        return _bind(bindings, variable_name(pat), exp, theory)
    if arbitrary_free(pat):
        # Optimistic bind; the excluded var's containment is validated inline:
        # if it is already bound to a symbol present in exp, fail now.
        excluded = bindings.lookup(pat[2])
        if isinstance(excluded, str) and excluded != pat[2] and free_in(excluded, exp):
            return None
        return _bind(bindings, variable_name(pat), exp, theory)
    # Compound, non-AC element: take the unique first ac_match yield, if any.
    for b in ac_match(pat, exp, theory, bindings):
        return b
    return None


def ac_match(pat, exp, theory, bindings: Optional[Bindings] = None,
             budget: Optional["MatchBudget"] = None) -> Iterator[Bindings]:
    """Yield every Bindings under which ``pat`` matches ``exp`` modulo the AC
    operators declared in ``theory``. Multi-valued and lazy.

    ``bindings`` seeds the match (default: empty). ``budget`` is an optional
    ``MatchBudget`` fail-safe for the AC enumeration.
    """
    if bindings is None:
        bindings = Bindings.empty()

    # Atom / single-variable / ?free: zero or one result.
    if (atom(pat) or arbitrary_constant(pat) or arbitrary_variable(pat)
            or arbitrary_expression(pat) or arbitrary_free(pat)):
        result = _match_one(pat, exp, theory, bindings)
        if result is not None:
            yield result
        return

    # A bare rest pattern matched directly against a list binds the whole list.
    if arbitrary_rest(pat):
        if isinstance(exp, list):
            extended = _bind(bindings, variable_name(pat), exp, theory)
            if extended is not None:
                yield extended
        return

    # Compound pattern. AC multiset case is added in a later task; for now,
    # only the non-AC positional case is handled.
    if not compound(pat) or not isinstance(exp, list) or not exp:
        return
    if car(pat) != car(exp):
        return
    yield from _match_positional(pat[1:], exp[1:], theory, bindings, budget)


def _match_positional(pats, exps, theory, bindings, budget) -> Iterator[Bindings]:
    """Match a list of sub-patterns against a list of sub-expressions in
    lockstep (the non-AC compound case). Multi-valued: a child may be an AC
    node. A trailing ``?...`` rest captures the positional tail as a list.
    """
    if not pats:
        if not exps:
            yield bindings
        return
    head, tail = pats[0], pats[1:]
    if arbitrary_rest(head):
        # Rest must be last; capture the positional remainder as a list.
        if tail:
            raise ValueError("Rest pattern (?...) must be last")
        extended = _bind(bindings, variable_name(head), list(exps), theory)
        if extended is not None:
            yield extended
        return
    if not exps:
        return
    for b in ac_match(head, exps[0], theory, bindings, budget):
        yield from _match_positional(tail, exps[1:], theory, b, budget)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest rerum/tests/test_acmatch.py -v`
Expected: PASS (all of Task 1 + TestNonACCases).

- [ ] **Step 5: ASCII check and commit**

Run: `LC_ALL=C grep -n '[^[:print:][:space:]]' rerum/acmatch.py rerum/tests/test_acmatch.py && echo "FOUND non-ASCII" || echo "clean"`
Expected: `clean`.

```bash
git add rerum/acmatch.py rerum/tests/test_acmatch.py
git commit -m "feat(f3): ac_match non-AC cases (multi-valued generalization of match)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `ac_match` -- AC multiset case, exhaust form (no rest) + budget

Add the heart: when the pattern head is AC and the expression shares that head, flatten both sides and enumerate assignments of the explicit sub-patterns to distinct expression elements. This task handles the case with NO trailing rest (the chosen elements must EXHAUST the multiset). The work budget is spent per assignment attempt.

**Files:**
- Modify: `rerum/acmatch.py`
- Test: `rerum/tests/test_acmatch.py`

- [ ] **Step 1: Write the failing tests**

Append to `rerum/tests/test_acmatch.py`:

```python
AC_PLUS = Theory.from_dict({"+": {"ac": True, "identity": 0}})


def _dictset(pat, exp, theory, budget=None):
    """Yielded bindings as a set of frozenset(items) for order-insensitive compare."""
    out = []
    for b in am.ac_match(pat, exp, theory, budget=budget):
        out.append(frozenset((k, _freeze(v)) for k, v in b.to_dict().items()))
    return out


def _freeze(v):
    return tuple(_freeze(x) for x in v) if isinstance(v, list) else v


class TestACMultisetExhaust:
    def test_two_vars_two_elements_two_matches(self):
        # (+ ?x ?y) against (+ a b): {x=a,y=b} and {x=b,y=a}.
        got = _dictset(["+", ["?", "x"], ["?", "y"]], ["+", "a", "b"], AC_PLUS)
        assert len(got) == 2
        assert frozenset({("x", "a"), ("y", "b")}) in got
        assert frozenset({("x", "b"), ("y", "a")}) in got

    def test_three_vars_three_elements_six_matches(self):
        got = _dictset(
            ["+", ["?", "x"], ["?", "y"], ["?", "z"]],
            ["+", "a", "b", "c"], AC_PLUS)
        assert len(got) == 6

    def test_exhaust_required_without_rest(self):
        # (+ ?x ?y) against (+ a b c): no rest -> must exhaust -> no match.
        assert _dictset(["+", ["?", "x"], ["?", "y"]], ["+", "a", "b", "c"], AC_PLUS) == []

    def test_literal_element_in_ac_node(self):
        # (+ 2 ?x) against (+ 2 a): 2 matches the literal, ?x=a.
        got = _dictset(["+", "2", ["?", "x"]], ["+", "2", "a"], AC_PLUS)
        assert got == [frozenset({("x", "a")})]
        # (+ 2 ?x) against (+ 3 a): no 2 present -> no match.
        assert _dictset(["+", "2", ["?", "x"]], ["+", "3", "a"], AC_PLUS) == []

    def test_flatten_before_match(self):
        # Nested sum is seen flat: (+ ?x ?y) against (+ a (+ b)) -> a, b.
        got = _dictset(["+", ["?", "x"], ["?", "y"]], ["+", "a", ["+", "b"]], AC_PLUS)
        assert len(got) == 2

    def test_non_linear_under_ac(self):
        # (+ ?x ?x) against (+ a a) matches (x=a); against (+ a b) does not.
        assert _dictset(["+", ["?", "x"], ["?", "x"]], ["+", "a", "a"], AC_PLUS) == \
            [frozenset({("x", "a")})]
        assert _dictset(["+", ["?", "x"], ["?", "x"]], ["+", "a", "b"], AC_PLUS) == []

    def test_budget_truncates_but_yields_are_valid(self):
        # A tiny budget over (+ ?x ?y ?z) vs (+ a b c): the 6-assignment
        # enumeration is cut short, but each binding yielded is a real match
        # (soundness under truncation). With steps=3 exactly one assignment
        # completes before the budget runs out.
        budget = am.MatchBudget(steps=3)
        pat = ["+", ["?", "x"], ["?", "y"], ["?", "z"]]
        exp = ["+", "a", "b", "c"]
        got = list(am.ac_match(pat, exp, AC_PLUS, budget=budget))
        assert budget.truncated is True
        assert 0 < len(got) < 6        # some, but not all 6, assignments
        for b in got:
            vals = [b["x"], b["y"], b["z"]]
            assert len(set(vals)) == 3
            assert set(vals) == {"a", "b", "c"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest rerum/tests/test_acmatch.py::TestACMultisetExhaust -v`
Expected: FAIL (the AC branch is not implemented; `(+ ?x ?y)` currently falls into `_match_positional` and yields the single positional match or nothing).

- [ ] **Step 3: Implement the AC multiset branch (exhaust form)**

In `rerum/acmatch.py`, modify `ac_match`: replace the final block

```python
    if not compound(pat) or not isinstance(exp, list) or not exp:
        return
    if car(pat) != car(exp):
        return
    yield from _match_positional(pat[1:], exp[1:], theory, bindings, budget)
```

with:

```python
    if not compound(pat) or not isinstance(exp, list) or not exp:
        return

    head = car(pat)
    if theory.is_ac(head) and isinstance(exp, list) and exp and exp[0] == head:
        # AC multiset case: flatten both sides, enumerate assignments.
        sub_pats = pat[1:]
        elements = flatten(exp, theory)[1:]
        # Split explicit sub-patterns from an optional trailing rest.
        rest = None
        explicit = sub_pats
        if sub_pats and arbitrary_rest(sub_pats[-1]):
            rest = sub_pats[-1]
            explicit = sub_pats[:-1]
        yield from _match_ac(explicit, rest, elements, theory,
                             bindings, budget)
        return

    if car(pat) != car(exp):
        return
    yield from _match_positional(pat[1:], exp[1:], theory, bindings, budget)
```

Then add the AC backtracker (the rest branch is added in Task 4; here `rest is None`):

```python
def _match_ac(explicit, rest, elements, theory, bindings, budget) -> Iterator[Bindings]:
    """Assign each pattern in ``explicit`` to a distinct element of the multiset
    ``elements`` (a list), backtracking with bindings threaded. ``elements`` is
    iterated in canonical (ORDER_KEY) order for determinism. With ``rest`` None,
    the chosen elements must EXHAUST ``elements``.
    """
    ordered = sorted(range(len(elements)), key=lambda i: ORDER_KEY(elements[i]))

    def recurse(pat_idx, used, binds):
        if pat_idx == len(explicit):
            leftover = [elements[i] for i in ordered if i not in used]
            if rest is None:
                if not leftover:
                    yield binds
                return
            # rest handling added in Task 4
            return
        p = explicit[pat_idx]
        for i in ordered:
            if i in used:
                continue
            if budget is not None and not budget.spend():
                return
            for b in ac_match(p, elements[i], theory, binds, budget):
                yield from recurse(pat_idx + 1, used | {i}, b)

    yield from recurse(0, frozenset(), bindings)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest rerum/tests/test_acmatch.py::TestACMultisetExhaust -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Run the whole acmatch file**

Run: `pytest rerum/tests/test_acmatch.py -v`
Expected: PASS (Tasks 1-3 all green).

- [ ] **Step 6: ASCII check and commit**

Run: `LC_ALL=C grep -n '[^[:print:][:space:]]' rerum/acmatch.py rerum/tests/test_acmatch.py && echo "FOUND non-ASCII" || echo "clean"`
Expected: `clean`.

```bash
git add rerum/acmatch.py rerum/tests/test_acmatch.py
git commit -m "feat(f3): ac_match AC multiset (exhaust form) + work budget

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: `ac_match` -- AC `?rest...` leftover capture + property test

Add the trailing-rest branch: the explicit sub-patterns consume a sub-multiset and `?rest...` binds the LEFTOVER as a list (canonical order). Then a property test pins soundness (every yield re-applies to an AC-equal term) and small-case completeness (count matches a brute-force oracle).

**Files:**
- Modify: `rerum/acmatch.py`
- Test: `rerum/tests/test_acmatch.py`

- [ ] **Step 1: Write the failing tests**

Append to `rerum/tests/test_acmatch.py`:

```python
import itertools


class TestACRest:
    def test_rest_captures_leftover_list(self):
        # (+ ?x ?rest...) against (+ a b c): x picks one, rest is the other two.
        got = []
        for b in am.ac_match(["+", ["?", "x"], ["?...", "rest"]],
                             ["+", "a", "b", "c"], AC_PLUS):
            got.append((b["x"], b["rest"]))
        # Three choices of x; rest is the remaining two in canonical order.
        xs = sorted(g[0] for g in got)
        assert xs == ["a", "b", "c"]
        for x, rest in got:
            assert isinstance(rest, list)
            assert sorted([x] + rest) == ["a", "b", "c"]

    def test_rest_empty_when_explicit_exhausts(self):
        # (+ ?x ?y ?rest...) against (+ a b): rest = [].
        got = [b["rest"] for b in am.ac_match(
            ["+", ["?", "x"], ["?", "y"], ["?...", "rest"]],
            ["+", "a", "b"], AC_PLUS)]
        assert got and all(r == [] for r in got)

    def test_rest_singleton(self):
        got = [b["rest"] for b in am.ac_match(
            ["+", ["?", "x"], ["?...", "rest"]], ["+", "a", "b"], AC_PLUS)]
        assert all(len(r) == 1 for r in got)

    def test_cancellation_idiom(self):
        # (+ ?x (- ?x) ?rest...) against (+ a (- a) b): x=a, rest=[b].
        pat = ["+", ["?", "x"], ["-", ["?", "x"]], ["?...", "rest"]]
        got = list(am.ac_match(pat, ["+", "a", ["-", "a"], "b"], AC_PLUS))
        assert any(b["x"] == "a" and b["rest"] == ["b"] for b in got)

    def test_cancellation_no_pair_no_match(self):
        pat = ["+", ["?", "x"], ["-", ["?", "x"]], ["?...", "rest"]]
        assert list(am.ac_match(pat, ["+", "a", "b", "c"], AC_PLUS)) == []


class TestACSoundnessProperty:
    def _ac_equal(self, t1, t2):
        return normalize_eq(t1, t2)

    def test_every_yield_is_a_real_match(self):
        from rerum.rewriter import instantiate
        from rerum.normalize import normalize
        # Pattern with explicit + rest; verify each yield reconstructs an
        # AC-equal subject when substituted back.
        pat = ["+", ["?", "x"], ["?", "y"], ["?...", "rest"]]
        exp = ["+", "a", "b", "c", "d"]
        for b in am.ac_match(pat, exp, AC_PLUS):
            # Rebuild (+ x y rest...) from bindings and check AC-equality to exp.
            rebuilt = ["+", b["x"], b["y"]] + list(b["rest"])
            assert normalize(rebuilt, AC_PLUS) == normalize(exp, AC_PLUS)

    def test_completeness_matches_brute_force_small(self):
        # (+ ?x ?y) over (+ a b c d): count equals ordered pairs of distinct
        # elements = 4*3 = 12 (no rest -> must exhaust, so actually 0). Use the
        # rest form for a non-trivial count.
        pat = ["+", ["?", "x"], ["?", "y"], ["?...", "rest"]]
        exp = ["+", "a", "b", "c", "d"]
        got = list(am.ac_match(pat, exp, AC_PLUS))
        # Brute force: ordered pairs (x,y) of distinct elements; rest = the rest.
        elems = ["a", "b", "c", "d"]
        expected = [(x, y) for x, y in itertools.permutations(elems, 2)]
        assert len(got) == len(expected)


def normalize_eq(t1, t2):
    from rerum.normalize import normalize
    return normalize(t1, AC_PLUS) == normalize(t2, AC_PLUS)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest rerum/tests/test_acmatch.py::TestACRest rerum/tests/test_acmatch.py::TestACSoundnessProperty -v`
Expected: FAIL (rest branch returns nothing; `_match_ac` has `# rest handling added in Task 4`).

- [ ] **Step 3: Implement the rest branch**

In `rerum/acmatch.py`, in `_match_ac`, replace the body of `recurse` after the `pat_idx == len(explicit)` guard:

```python
        if pat_idx == len(explicit):
            leftover = [elements[i] for i in ordered if i not in used]
            if rest is None:
                if not leftover:
                    yield binds
                return
            # rest handling added in Task 4
            return
```

with:

```python
        if pat_idx == len(explicit):
            leftover = [elements[i] for i in ordered if i not in used]
            if rest is None:
                if not leftover:
                    yield binds
                return
            extended = _bind(binds, variable_name(rest), leftover, theory)
            if extended is not None:
                yield extended
            return
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest rerum/tests/test_acmatch.py::TestACRest rerum/tests/test_acmatch.py::TestACSoundnessProperty -v`
Expected: PASS.

- [ ] **Step 5: Run the whole acmatch file**

Run: `pytest rerum/tests/test_acmatch.py -v`
Expected: PASS (all Tasks 1-4).

- [ ] **Step 6: ASCII check and commit**

Run: `LC_ALL=C grep -n '[^[:print:][:space:]]' rerum/acmatch.py rerum/tests/test_acmatch.py && echo "FOUND non-ASCII" || echo "clean"`
Expected: `clean`.

```bash
git add rerum/acmatch.py rerum/tests/test_acmatch.py
git commit -m "feat(f3): ac_match ?rest leftover capture + soundness property test

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Engine seam -- `_match_lhs`, rewire consumers, fast-path gate, truncation flag

Wire `ac_match` into the engine behind one helper, so `simplify`/`apply_once` (first match) and the equational methods via `_all_single_rewrites` (all matches) become AC-aware under an AC theory. The no-theory path is byte-identical.

**Files:**
- Modify: `rerum/engine.py`
- Test: `rerum/tests/test_acmatch.py`

- [ ] **Step 1: Write the failing tests**

Append to `rerum/tests/test_acmatch.py`:

```python
from rerum.engine import RuleEngine


class TestEngineACMatching:
    def test_simplify_fires_ac_rule_across_arrangements(self):
        # Cancellation rule fires no matter where the cancelling pair sits.
        eng = RuleEngine.from_dsl(
            "@cancel: (+ ?x (- ?x) ?rest...) => (+ :rest...)"
        )
        eng.with_theory(Theory.from_dict({"+": {"ac": True, "identity": 0}}))
        # (- a) is not adjacent to a; positional matching would miss it.
        result = eng.simplify(["+", "a", "b", ["-", "a"]])
        # After cancelling a and (- a), only b remains; (+ b) collapses to b.
        assert result == "b"

    def test_no_theory_simplify_unchanged(self):
        # Without a theory, the same rule only fires on the exact arrangement.
        eng = RuleEngine.from_dsl(
            "@cancel: (+ ?x (- ?x) ?rest...) => (+ :rest...)"
        )
        # Positional: a and (- a) ARE adjacent here, so it fires syntactically.
        assert eng.simplify(["+", "a", ["-", "a"], "b"]) == ["+", "b"]
        # But not when separated -- no AC theory, no reordering.
        assert eng.simplify(["+", "a", "b", ["-", "a"]]) == ["+", "a", "b", ["-", "a"]]

    def test_apply_once_takes_first_ac_match(self):
        eng = RuleEngine.from_dsl("@r: (+ ?x ?y) => (pair :x :y)")
        eng.with_theory(Theory.from_dict({"+": {"ac": True, "identity": 0}}))
        result, meta = eng.apply_once(["+", "a", "b"])
        # First canonical assignment fires; result is a (pair ...).
        assert meta is not None and result[0] == "pair"

    def test_truncation_flag_exposed(self):
        eng = RuleEngine.from_dsl("@r: (+ ?x ?y ?z) => done")
        eng.with_theory(Theory.from_dict({"+": {"ac": True, "identity": 0}}))
        eng.set_ac_match_budget(2)        # tiny budget
        eng.simplify(["+", "a", "b", "c", "d", "e", "f"])
        assert eng.ac_match_truncated is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest rerum/tests/test_acmatch.py::TestEngineACMatching -v`
Expected: FAIL (`with_theory` exists but AC matching is not wired; `set_ac_match_budget`/`ac_match_truncated` undefined).

- [ ] **Step 3: Add the seam to `RuleEngine.__init__`**

In `rerum/engine.py`, find `self._theory = None` (around line 1617) and immediately after it add:

```python
        self._ac_match_budget = 10000     # fail-safe cap for AC enumeration
        self._ac_match_truncated = False  # set when any AC match truncated
```

- [ ] **Step 4: Add `_match_lhs`, the budget setter, and the truncation property**

In `rerum/engine.py`, immediately AFTER the `_canonicalize` method (ends around line 1871 with `return _normalize(expr, self._theory)`), add:

```python
    def _theory_has_ac(self) -> bool:
        return self._theory is not None and self._theory.has_ac()

    def _match_lhs(self, pattern, expr):
        """Yield each Bindings under which ``pattern`` matches ``expr``.

        Without an AC theory this yields ``match(pattern, expr)``'s single
        result (0 or 1) and is byte-identical to the prior single-match path.
        Under an AC theory it yields ``ac_match``'s several results and records
        budget truncation on ``self._ac_match_truncated``.
        """
        if not self._theory_has_ac():
            b = _match_internal(pattern, expr)
            if b is not None:
                yield b
            return
        from .acmatch import ac_match, MatchBudget
        budget = MatchBudget(steps=self._ac_match_budget)
        try:
            for b in ac_match(pattern, expr, self._theory, budget=budget):
                yield b
        except RecursionError:
            return
        finally:
            if budget.truncated:
                self._ac_match_truncated = True

    def set_ac_match_budget(self, steps):
        """Set the per-match AC enumeration budget (None = unbounded)."""
        self._ac_match_budget = steps

    @property
    def ac_match_truncated(self) -> bool:
        """True if an AC match hit its budget since the last top-level call."""
        return self._ac_match_truncated
```

- [ ] **Step 5: Rewire `apply_once` to loop over `_match_lhs`**

In `rerum/engine.py`, in `apply_once` (around line 2366), replace:

```python
            pattern, skeleton = rule
            bindings = _match_internal(pattern, expr)
            if bindings is not None:
                # Check condition if present
                if not self._check_condition(metadata.condition, bindings):
                    continue
                if not self._check_should_fire(rule, metadata, expr, bindings):
                    continue
                result = instantiate(skeleton, bindings, self._fold_funcs,
                                     undefined_op_resolver=self._undefined_op_resolver,
                                     fold_error_resolver=self._fold_error_resolver)
                if result != expr:
                    guard = self._evaluate_guard(metadata.condition, bindings)
                    step = self._build_step(
                        rule_idx, rule, metadata, expr, result, bindings,
                        path=path, guard=guard,
                    )
                    self._fire_rule_applied(step, expr_path=path)
                return result, metadata
```

with:

```python
            pattern, skeleton = rule
            for bindings in self._match_lhs(pattern, expr):
                # Check condition if present
                if not self._check_condition(metadata.condition, bindings):
                    continue
                if not self._check_should_fire(rule, metadata, expr, bindings):
                    continue
                result = instantiate(skeleton, bindings, self._fold_funcs,
                                     undefined_op_resolver=self._undefined_op_resolver,
                                     fold_error_resolver=self._fold_error_resolver)
                if result != expr:
                    guard = self._evaluate_guard(metadata.condition, bindings)
                    step = self._build_step(
                        rule_idx, rule, metadata, expr, result, bindings,
                        path=path, guard=guard,
                    )
                    self._fire_rule_applied(step, expr_path=path)
                return result, metadata
```

Note: the `return result, metadata` is INSIDE the loop, so it returns on the first binding that passes condition + should_fire -- byte-identical to the old behavior when `_match_lhs` yields a single result.

- [ ] **Step 6: Rewire `_all_single_rewrites` to loop over `_match_lhs`**

In `rerum/engine.py`, in `_all_single_rewrites` (around line 3645), replace:

```python
            pattern, skeleton = rule
            bindings = _match_internal(pattern, expr)
            if bindings is not None:
                if not self._check_condition(metadata.condition, bindings):
                    continue
                if not self._check_should_fire(rule, metadata, expr, bindings):
                    continue
                result = instantiate(skeleton, bindings, self._fold_funcs,
                                     undefined_op_resolver=self._undefined_op_resolver,
                                     fold_error_resolver=self._fold_error_resolver)
                if result != expr:
                    if labeled:
                        from .trace import rule_identity
                        label = {
                            "rule_id": rule_identity(metadata, pattern, skeleton),
                            "direction": metadata.direction,
                            "bindings": bindings.to_dict(),
                            "path": list(_path),
                        }
                    else:
                        label = None
                    add_if_new(result, label)
```

with:

```python
            pattern, skeleton = rule
            for bindings in self._match_lhs(pattern, expr):
                if not self._check_condition(metadata.condition, bindings):
                    continue
                if not self._check_should_fire(rule, metadata, expr, bindings):
                    continue
                result = instantiate(skeleton, bindings, self._fold_funcs,
                                     undefined_op_resolver=self._undefined_op_resolver,
                                     fold_error_resolver=self._fold_error_resolver)
                if result != expr:
                    if labeled:
                        from .trace import rule_identity
                        label = {
                            "rule_id": rule_identity(metadata, pattern, skeleton),
                            "direction": metadata.direction,
                            "bindings": bindings.to_dict(),
                            "path": list(_path),
                        }
                    else:
                        label = None
                    add_if_new(result, label)
```

(The `for` replaces the `if bindings is not None`; the body is otherwise unchanged. The `seen` set dedups distinct bindings that yield the same result.)

- [ ] **Step 7: Gate `simplify`'s fast path on the AC theory + reset the flag**

In `rerum/engine.py`, in `simplify` (around line 2462), find:

```python
        # Check if we need slow path (conditions or groups)
        has_conditions = any(m.condition is not None for m in self._metadata)
        has_groups = groups is not None or self._disabled_groups
```

and replace with:

```python
        # Reset the AC-truncation flag at the top of every top-level simplify.
        self._ac_match_truncated = False
        # Check if we need slow path (conditions, groups, or an AC theory).
        has_conditions = any(m.condition is not None for m in self._metadata)
        has_groups = groups is not None or self._disabled_groups
        has_ac = self._theory_has_ac()
```

Then change the `exhaustive` branch condition from:

```python
        if strategy == "exhaustive":
            if has_conditions or has_groups:
                return self._simplify_exhaustive(expr, max_steps, groups=groups)
```

to:

```python
        if strategy == "exhaustive":
            if has_conditions or has_groups or has_ac:
                return self._simplify_exhaustive(expr, max_steps, groups=groups)
```

This bypasses the cached `rewriter()` fast path under an AC theory (the same reason hooks bypass it: the pure cached rewriter cannot carry theory context). With no theory, `has_ac` is False and the fast path is unchanged.

- [ ] **Step 8: Rewire `_simplify_exhaustive` to use `_match_lhs` and canonicalize AC results**

This is the driver the AC gate routes to. It has its OWN top-level match loop (NOT shared with `apply_once`), so it must be rewired too, and AC rewrite results (e.g. a residual `(+ b)`) must be canonicalized so the fixpoint converges to `b`.

In `rerum/engine.py`, in `_simplify_exhaustive` (around line 2899), find the top of the step loop:

```python
        for _ in range(max_steps):
            if self._cancel_requested:
                return current
            key = _expr_to_tuple(current)
```

and replace with (insert one canonicalization line under an AC theory):

```python
        ac_active = self._theory_has_ac()
        for _ in range(max_steps):
            if self._cancel_requested:
                return current
            if ac_active:
                # Rewriting modulo AC keeps terms canonical: this collapses a
                # residual (+ b) -> b and (+) -> identity, and lets the
                # visited-set cycle check compare canonical forms. _canonicalize
                # is the identity when no theory is set, but ac_active gates it.
                current = self._canonicalize(current)
            key = _expr_to_tuple(current)
```

Then find the top-level rule loop inside `_simplify_exhaustive` (around line 2911):

```python
            # Try rules at top level
            for rule_idx, rule in enumerate(self._rules):
                metadata = self._metadata[rule_idx]
                # Check group filter
                if not self._is_rule_active(metadata, groups):
                    continue
                pattern, skeleton = rule
                bindings = _match_internal(pattern, current)
                if bindings is not None:
                    if not self._check_condition(metadata.condition, bindings):
                        continue
                    if not self._check_should_fire(rule, metadata, current, bindings):
                        continue
                    new_expr = instantiate(skeleton, bindings, self._fold_funcs,
                                           undefined_op_resolver=self._undefined_op_resolver,
                                           fold_error_resolver=self._fold_error_resolver)
                    if new_expr != current:
                        guard = self._evaluate_guard(metadata.condition, bindings)
                        step = self._build_step(
                            rule_idx, rule, metadata, current, new_expr, bindings,
                            path=path, guard=guard,
                        )
                        if self._fire_rule_applied(step, expr_path=path):
                            return new_expr  # Hook requested cancellation after this step.
                        current = new_expr
                        changed = True
                        break
```

and replace with (loop over `_match_lhs`; fire the first productive binding):

```python
            # Try rules at top level
            for rule_idx, rule in enumerate(self._rules):
                metadata = self._metadata[rule_idx]
                # Check group filter
                if not self._is_rule_active(metadata, groups):
                    continue
                pattern, skeleton = rule
                fired = False
                for bindings in self._match_lhs(pattern, current):
                    if not self._check_condition(metadata.condition, bindings):
                        continue
                    if not self._check_should_fire(rule, metadata, current, bindings):
                        continue
                    new_expr = instantiate(skeleton, bindings, self._fold_funcs,
                                           undefined_op_resolver=self._undefined_op_resolver,
                                           fold_error_resolver=self._fold_error_resolver)
                    if new_expr != current:
                        guard = self._evaluate_guard(metadata.condition, bindings)
                        step = self._build_step(
                            rule_idx, rule, metadata, current, new_expr, bindings,
                            path=path, guard=guard,
                        )
                        if self._fire_rule_applied(step, expr_path=path):
                            return new_expr  # Hook requested cancellation after this step.
                        current = new_expr
                        changed = True
                        fired = True
                        break
                if fired:
                    break
```

With no AC theory, `_match_lhs` yields the single `_match_internal` result, so this loop runs at most once and is behavior-identical to the original. Under AC it tries each binding and fires the first productive one (`ORDER_KEY` order makes "first" deterministic).

Note: `_simplify_with_trace` (used by `simplify(trace=True)`) drives the same `_simplify_exhaustive`, so trace mode inherits AC matching. The non-default `bottomup`/`topdown` strategies and the introspection helper `rules_matching` keep their syntactic loops in v1 -- a documented limitation, since the default `exhaustive` strategy is the AC path.

- [ ] **Step 9: Run tests to verify they pass**

Run: `pytest rerum/tests/test_acmatch.py::TestEngineACMatching -v`
Expected: PASS (4 passed).

- [ ] **Step 10: Regression -- the no-theory world is unchanged**

Run: `pytest rerum/tests/test_rewriter.py rerum/tests/test_engine_methods.py rerum/tests/test_strategies.py rerum/tests/test_theory_reasoning.py -q`
Expected: PASS (no regressions; the no-theory path is byte-identical). The one currently-passing boundary test in `test_theory_reasoning.py` still passes here -- it is flipped in Task 6.

- [ ] **Step 11: ASCII check and commit**

Run: `LC_ALL=C grep -n '[^[:print:][:space:]]' rerum/engine.py rerum/tests/test_acmatch.py && echo "FOUND non-ASCII" || echo "clean"`
Expected: `clean`.

```bash
git add rerum/engine.py rerum/tests/test_acmatch.py
git commit -m "feat(f3): engine seam _match_lhs; AC-aware simplify + equational methods

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Flip the pinned boundary, equational AC tests, re-exports, examples demo, full gate

Close the documented F1 soundness boundary, prove the equational methods are AC-aware, re-export the public surface, ship a light data-only demo, and run the full gate.

**Files:**
- Modify: `rerum/tests/test_theory_reasoning.py`
- Modify: `rerum/__init__.py`
- Create: `examples/ac_demo.rules`, `examples/ac_demo.theory.json`
- Test: `rerum/tests/test_acmatch.py`

- [ ] **Step 1: Flip the pinned boundary test**

In `rerum/tests/test_theory_reasoning.py`, find `test_position_pinning_rule_is_not_reached_under_ac_theory` (around line 190). Replace the whole method with:

```python
    def test_position_pinning_rule_IS_reached_under_ac_theory(self):
        # F3 (AC-matching) closes the former soundness boundary: a distribute
        # rule that pins the (+ ...) factor as the FIRST operand of * now fires
        # under an AC * theory, because the matcher tries every arrangement.
        eng = RuleEngine.from_dsl(
            "@distrib: (* (+ ?a ?b) ?c) => (+ (* :a :c) (* :b :c))"
        )
        eng.with_theory(Theory.from_dict({"*": {"ac": True}}))
        target = ["+", ["*", "a", "c"], ["*", "b", "c"]]
        proof = eng.prove_equal(["*", ["+", "a", "b"], "c"], target,
                                include_unidirectional=True)
        assert proof is not None   # boundary closed by F3
```

Keep `test_position_pinning_rule_DOES_fire_without_theory` (the no-theory control) unchanged.

- [ ] **Step 2: Run the flipped test (and its control)**

Run: `pytest rerum/tests/test_theory_reasoning.py -k position_pinning -v`
Expected: PASS (the boundary test now finds a proof; the control still passes).

- [ ] **Step 3: Add equational-method AC tests**

Append to `rerum/tests/test_acmatch.py`:

```python
class TestEquationalAC:
    def _ac_engine(self, dsl, sig):
        eng = RuleEngine.from_dsl(dsl)
        eng.with_theory(Theory.from_dict(sig))
        return eng

    def test_prove_equal_ac_distributivity(self):
        eng = self._ac_engine(
            "@distrib: (* (+ ?a ?b) ?c) => (+ (* :a :c) (* :b :c))",
            {"*": {"ac": True}})
        proof = eng.prove_equal(["*", ["+", "a", "b"], "c"],
                                ["+", ["*", "a", "c"], ["*", "b", "c"]],
                                include_unidirectional=True)
        assert proof is not None

    def test_equivalents_includes_ac_rewrite(self):
        eng = self._ac_engine(
            "@cancel: (+ ?x (- ?x) ?rest...) => (+ :rest...)",
            {"+": {"ac": True, "identity": 0}})
        forms = list(eng.equivalents(["+", "a", "b", ["-", "a"]],
                                     include_unidirectional=True, max_depth=3))
        # The cancelled form (b, possibly as (+ b)) is reachable.
        assert any(f == "b" or f == ["+", "b"] for f in forms)
```

- [ ] **Step 4: Add the re-exports**

In `rerum/__init__.py`, find the `# Termination analysis (F4)` import block and the `# Completion (F5)` block (added earlier). Immediately AFTER the completion import block, add:

```python
# AC-matching (F3)
from .acmatch import (
    ac_match,
    MatchBudget,
)
```

In the `__all__` list, after the `# Completion` entries (`"complete"`, `"CompletionResult"`), add:

```python
    # AC-matching
    "ac_match",
    "MatchBudget",
```

- [ ] **Step 5: Create the data-only examples demo**

Create `examples/ac_demo.theory.json`:

```json
{"+": {"ac": true, "identity": 0}}
```

Create `examples/ac_demo.rules`:

```
# AC-matching demo: one rule cancels a term wherever it sits in a sum.
# Load with the paired ac_demo.theory.json so + is associative-commutative.
@cancel: (+ ?x (- ?x) ?rest...) => (+ :rest...)
```

Append a test that drives them through the GENERAL engine to `rerum/tests/test_acmatch.py`:

```python
class TestExamplesDemo:
    def test_ac_demo_cancels_across_arrangements(self):
        import json
        import os
        root = os.path.join(os.path.dirname(__file__), "..", "..", "examples")
        eng = RuleEngine.from_file(os.path.join(root, "ac_demo.rules"))
        with open(os.path.join(root, "ac_demo.theory.json")) as fh:
            eng.with_theory(Theory.from_json(fh.read()))
        # The cancelling pair is separated by an unrelated term.
        assert eng.simplify(["+", "a", "b", ["-", "a"]]) == "b"
```

Note: `RuleEngine.from_file(path)` is a real classmethod (`engine.py:3549`). The relative path resolves from `rerum/tests/`: `examples/` is at the repo root, so from `rerum/tests/__file__` it is `../../examples`.

- [ ] **Step 6: Run the new tests**

Run: `pytest rerum/tests/test_acmatch.py::TestEquationalAC rerum/tests/test_acmatch.py::TestExamplesDemo -v`
Expected: PASS.

- [ ] **Step 7: Full suite + domain guard + ASCII**

Run: `pytest -q`
Expected: PASS. Report the total. Baseline before F3 was 1647; this adds the new `test_acmatch.py` tests and flips one boundary test (no net deletion).

Run: `pytest rerum/tests/test_mcp_no_domain.py -q`
Expected: PASS (12). `acmatch.py` is core, not under `rerum/mcp/`, but it must still name no operator literal -- the guard plus review enforce this.

Run: `LC_ALL=C grep -n '[^[:print:][:space:]]' rerum/__init__.py rerum/tests/test_theory_reasoning.py rerum/tests/test_acmatch.py examples/ac_demo.rules examples/ac_demo.theory.json && echo "FOUND non-ASCII" || echo "clean"`
Expected: `clean`.

- [ ] **Step 8: Commit**

```bash
git add rerum/__init__.py rerum/tests/test_theory_reasoning.py rerum/tests/test_acmatch.py examples/ac_demo.rules examples/ac_demo.theory.json
git commit -m "feat(f3): close pinned boundary, equational AC tests, re-exports, demo

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Post-implementation

After all six tasks: dispatch the Opus final holistic review (soundness of the AC matcher: every yield is a real match; no-theory path byte-identical; general-engine principle held; the pinned boundary genuinely closed), then `superpowers:finishing-a-development-branch` to present the push decision (on `main`, per the per-feature rhythm).

## Notes for the implementer

- **Determinism:** `ORDER_KEY` gives the canonical iteration order so `simplify`'s first AC match is reproducible. Do not iterate `elements` in raw order.
- **Soundness vs completeness:** the budget only bounds enumeration. Never let a budget hit fabricate or skip a CONSISTENCY check -- a yielded binding must always be a real match.
- **No-theory invariant:** `_match_lhs` with no AC theory must call `_match_internal` and yield its single result, and every rewired loop (`apply_once`, `_simplify_exhaustive`, `_all_single_rewrites`) must be behavior-identical to its original when `_match_lhs` yields a single result. Any divergence ripples through the whole suite -- Task 5 Step 10 is the guard.
- **Three match loops, not one:** `_simplify_exhaustive` (the AC `simplify` path), `apply_once` (the `once` strategy + the public method), and `_all_single_rewrites` (every equational method) each have their OWN match call site. All three are rewired in Task 5. The `bottomup`/`topdown` strategies and `rules_matching` keep syntactic matching in v1 (documented limitation).
- **`?free` inside AC nodes:** validated inline in `_match_one` (bindings are known per branch), not via a positional post-pass.
- **AC result cleanup is deterministic, verified:** `normalize(["+","b"], theory)` returns `"b"` and `normalize(["+"], theory)` returns the identity (confirmed). Task 5 Step 8 canonicalizes `current` at the top of each `_simplify_exhaustive` pass under an AC theory, so a residual `(+ b)` collapses to `b` before the next pass and the fixpoint converges. The equational methods already canonicalize neighbors via the F1 `_canonicalize` seam, so `_all_single_rewrites` needs no separate cleanup. `apply_once` is one-shot and returns its single step uncleaned by design.
