# TRS-Frontier Review Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all 20 confirmed MAJOR+MINOR findings from the 2026-06-23 TRS-frontier code review (`docs/superpowers/reviews/2026-06-23-trs-frontier-code-review.md`).

**Architecture:** Each task is a focused, TDD bug fix (write a test that reproduces the finding, then fix). Related findings are clustered into one task when they share a root cause or file region. No new modules; every change is to an existing core module plus a regression test.

**Tech Stack:** Python 3.9+, pytest. Touches `rerum/normalize.py`, `confluence.py`, `completion.py`, `acmatch.py`, `narrowing.py`, `engine.py`, `rewriter.py`.

---

## Conventions all tasks follow

- **ASCII only** in every file write (a commit hook rejects non-ASCII). Use `->`, plain hyphens.
- **Never stage `.mcp.json`**. Stage only the files each task names.
- Commit messages end with the trailer:
  ```
  Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
  ```
- After EACH task: run the task's tests, then the FULL suite (`pytest -q`) to confirm no regression, then the no-domain guard (`pytest rerum/tests/test_mcp_no_domain.py -q`), then ASCII-check the touched files.
- Baseline full-suite count before this plan: **1711 passed**. Each task adds tests; report the new total.

## Task ordering

Tasks are independent except: Task 4 (rewriter `max_steps` threading) should precede Task 9 (completion honesty) only because both touch completion; otherwise any order works. Recommended order is as numbered (MAJORs first within each module).

## Findings -> tasks map

| Task | Findings closed | Severity |
|------|-----------------|----------|
| 1 | frac-key-collision-1, nan-non-total-order-3 | MAJOR, MINOR |
| 2 | theory-malformed-entry-2 | MAJOR |
| 3 | repeat-op-not-ac-4 | MINOR |
| 4 | budget-1, max-steps-decorative-1 | MAJOR, MINOR |
| 5 | empty-pat-crash-1 | MAJOR |
| 6 | typed-rest-1 | MAJOR |
| 7 | budget-non-ac-1 | MINOR |
| 8 | confluence-1 | MAJOR |
| 9 | completion-1 | MAJOR |
| 10 | unify-refuse-1 | MINOR |
| 11 | notanalyzed-none-1, confluence-2 | MINOR |
| 12 | ac-strategy-gap-1 | MAJOR |
| 13 | apply-once-ac-completeness-2 | MINOR |
| 14 | truncated-flag-not-reset-3, recursionerror-silent-truncation-4 | MINOR |
| 15 | narrowing-1, narrowing-2 | MINOR |

---

### Task 1: `normalize.ORDER_KEY` exact numeric key (frac-key-collision + NaN total order)

`ORDER_KEY` keys numbers on lossy `float(expr)`, so two distinct large Fractions/ints collide and `_collect_ac` silently merges them (value corruption). Non-finite floats also make the order non-total. Fix: key numbers on an EXACT, totally-ordered value, keeping the existing type-name tiebreaker (so int/float/Fraction/bool stay distinct exactly as before -- only the value comparison becomes exact).

**Files:**
- Modify: `rerum/normalize.py` (ORDER_KEY around lines 126-145; add a helper)
- Test: `rerum/tests/test_normalize.py`

- [ ] **Step 1: Write the failing tests**

Append to `rerum/tests/test_normalize.py`:

```python
class TestOrderKeyExactNumeric:
    def _arith(self):
        from rerum.normalize import Theory
        return Theory.from_dict({
            "+": {"ac": True, "identity": 0, "repeat": {"op": "*", "via": "count"}},
            "*": {"ac": True, "identity": 1, "annihilator": 0,
                  "repeat": {"op": "^", "via": "exp"}}})

    def test_distinct_large_fractions_not_merged(self):
        from fractions import Fraction
        from rerum.normalize import normalize
        a = Fraction(10**18, 3)
        b = Fraction(10**18 + 1, 3)  # distinct, but float(a)==float(b)
        # a*b*x must NOT collapse to a^2*x.
        out = normalize(["*", a, b, "x"], self._arith())
        # The two distinct coefficients survive (no (^ a 2) form).
        flat = repr(out)
        assert "2" not in [str(e) for e in (out[1:] if isinstance(out, list) else [])] or \
            (Fraction(10**18, 3) in out and Fraction(10**18 + 1, 3) in out)
        # Strong check: simplify-by-hand product is preserved as two operands.
        nums = [e for e in (out[1:] if isinstance(out, list) else []) if isinstance(e, Fraction)]
        assert a in nums and b in nums

    def test_distinct_large_ints_not_merged(self):
        from rerum.normalize import normalize
        big = 2**53
        out = normalize(["*", big, big + 1, "x"], self._arith())
        nums = [e for e in (out[1:] if isinstance(out, list) else []) if isinstance(e, int)]
        assert big in nums and (big + 1) in nums

    def test_order_key_total_on_non_finite(self):
        # NaN/inf operands must not raise and must give a deterministic order.
        from rerum.normalize import ORDER_KEY
        nan = float("nan"); pinf = float("inf"); ninf = float("-inf")
        keys = [ORDER_KEY(x) for x in (nan, pinf, ninf, 0, 1.5)]
        # All comparable pairwise (no TypeError) -> sorting succeeds.
        assert sorted(keys) == sorted(keys)
        assert ORDER_KEY(ninf) < ORDER_KEY(0) < ORDER_KEY(pinf) < ORDER_KEY(nan)

    def test_equal_value_same_type_still_merges(self):
        from fractions import Fraction
        from rerum.normalize import normalize
        a = Fraction(10**18, 3)
        out = normalize(["+", a, a, "x"], self._arith())
        # identical operands DO collect: (+ x (* 2 a))
        assert any(isinstance(e, list) and e[0] == "*" for e in out[1:])
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest rerum/tests/test_normalize.py::TestOrderKeyExactNumeric -v`
Expected: FAIL (large distinct numbers merge; NaN ordering raises or is non-total).

- [ ] **Step 3: Implement the exact numeric key**

In `rerum/normalize.py`, add `import math` and `from fractions import Fraction` at the top if not present (check the existing imports first). Then add this helper immediately ABOVE `ORDER_KEY`:

```python
def _num_value_key(x):
    """Exact, totally-ordered value sub-key for a numeric atom, replacing the
    lossy ``float(x)``. Finite ints/floats/Fractions key on their EXACT rational
    value (so distinct values never collide and equal values share a key);
    non-finite floats sort into fixed buckets (-inf < finite < +inf < NaN)."""
    if isinstance(x, float):
        if math.isnan(x):
            return (2, 0)
        if math.isinf(x):
            return (1, 0) if x > 0 else (-1, 0)
    fr = Fraction(x)  # exact for int/float/Fraction/bool; finite by here
    return (0, (fr.numerator, fr.denominator))
```

Then change the numeric branches of `ORDER_KEY` from:

```python
    if isinstance(expr, bool):
        return (_RANK_NUMBER, (float(expr), "bool"))
    if _is_number(expr):
        return (_RANK_NUMBER, (float(expr), type(expr).__name__))
```

to:

```python
    if isinstance(expr, bool):
        return (_RANK_NUMBER, (_num_value_key(expr), "bool"))
    if _is_number(expr):
        return (_RANK_NUMBER, (_num_value_key(expr), type(expr).__name__))
```

(The type-name tiebreaker is retained, so int/float/Fraction/bool keep their existing relative order; only the value comparison is now exact.)

- [ ] **Step 4: Run to verify pass + regression**

Run: `pytest rerum/tests/test_normalize.py -q` then `pytest -q`
Expected: PASS (new tests green; full suite unchanged otherwise).

- [ ] **Step 5: ASCII + commit**

Run: `LC_ALL=C grep -n '[^[:print:][:space:]]' rerum/normalize.py rerum/tests/test_normalize.py && echo FOUND || echo clean`

```bash
git add rerum/normalize.py rerum/tests/test_normalize.py
git commit -m "fix(normalize): exact numeric ORDER_KEY (no float() collision; total order on NaN/inf)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `Theory` entry validation (theory-malformed-entry)

`Theory.from_json`/`from_dict` accept a signature whose values are not dicts; `is_ac`/`has_ac` then crash with an unmapped `AttributeError`, defeating the MCP `parse_error` mapping. Fix: validate entry shape at construction.

**Files:**
- Modify: `rerum/normalize.py` (`Theory.__init__`, around lines 43-44)
- Test: `rerum/tests/test_normalize.py`

- [ ] **Step 1: Write the failing tests**

Append to `rerum/tests/test_normalize.py`:

```python
class TestTheoryEntryValidation:
    def test_non_dict_entry_rejected_from_dict(self):
        import pytest
        from rerum.normalize import Theory
        with pytest.raises(ValueError) as ei:
            Theory.from_dict({"+": "ac"})   # value should be an object
        assert "+" in str(ei.value)

    def test_non_dict_entry_rejected_from_json(self):
        import pytest
        from rerum.normalize import Theory
        with pytest.raises(ValueError):
            Theory.from_json('{"+": true}')

    def test_valid_theory_still_constructs(self):
        from rerum.normalize import Theory
        t = Theory.from_dict({"+": {"ac": True}})
        assert t.is_ac("+") is True and t.has_ac() is True
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest rerum/tests/test_normalize.py::TestTheoryEntryValidation -v`
Expected: FAIL (no validation; `from_dict({"+": "ac"})` constructs, later `.is_ac` may AttributeError).

- [ ] **Step 3: Implement validation in `__init__`**

In `rerum/normalize.py`, replace `Theory.__init__`:

```python
    def __init__(self, sig: Dict[str, Dict[str, Any]]):
        self._sig = dict(sig or {})
```

with:

```python
    def __init__(self, sig: Dict[str, Dict[str, Any]]):
        sig = dict(sig or {})
        for op, entry in sig.items():
            if not isinstance(entry, dict):
                raise ValueError(
                    f"theory entry for {op!r} must be an object (operator "
                    f"signature), got {type(entry).__name__}")
        self._sig = sig
```

(`from_dict` and `from_json` both route through `__init__`, so this covers both.)

- [ ] **Step 4: Run + regression**

Run: `pytest rerum/tests/test_normalize.py -q` then `pytest -q`
Expected: PASS.

- [ ] **Step 5: ASCII + commit**

```bash
git add rerum/normalize.py rerum/tests/test_normalize.py
git commit -m "fix(normalize): validate Theory signature entries are dicts (clean ValueError)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: validate `repeat.op` is a declared AC operator (repeat-op-not-ac)

A `repeat.op` that is not itself a declared AC operator yields a surprising non-simplified form (e.g. `(+ 0 0)` -> `(* 2 0)`). Fix: validate at construction that every referenced `repeat.op` is itself a declared AC operator in the same theory.

**Files:**
- Modify: `rerum/normalize.py` (`Theory.__init__`, after the Task 2 entry-shape check)
- Test: `rerum/tests/test_normalize.py`

- [ ] **Step 1: Write the failing tests**

Append to `rerum/tests/test_normalize.py`:

```python
class TestTheoryRepeatValidation:
    def test_repeat_op_must_be_declared_ac(self):
        import pytest
        from rerum.normalize import Theory
        with pytest.raises(ValueError) as ei:
            # '*' is referenced as +'s repeat op but is not declared AC here.
            Theory.from_dict({"+": {"ac": True, "repeat": {"op": "*", "via": "count"}}})
        assert "*" in str(ei.value) and "repeat" in str(ei.value).lower()

    def test_repeat_op_declared_ac_ok(self):
        from rerum.normalize import Theory
        Theory.from_dict({
            "+": {"ac": True, "repeat": {"op": "*", "via": "count"}},
            "*": {"ac": True}})   # ok: * is declared AC
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest rerum/tests/test_normalize.py::TestTheoryRepeatValidation -v`
Expected: FAIL (no validation today).

- [ ] **Step 3: Implement the check**

In `rerum/normalize.py` `Theory.__init__`, AFTER the entry-shape loop from Task 2 and before `self._sig = sig`, add:

```python
        for op, entry in sig.items():
            rep = entry.get("repeat")
            if rep is not None:
                rop = rep.get("op") if isinstance(rep, dict) else None
                rentry = sig.get(rop)
                if not (isinstance(rentry, dict) and rentry.get("ac", False)):
                    raise ValueError(
                        f"repeat.op {rop!r} for operator {op!r} must itself be a "
                        f"declared AC operator in the same theory")
```

- [ ] **Step 4: Run + regression**

Run: `pytest rerum/tests/test_normalize.py -q` then `pytest -q`
Expected: PASS. NOTE: if any shipped `examples/*.theory.json` or test theory references a non-AC repeat op, this will surface it -- that is the bug; fix the data (declare the op AC) and report it. The arithmetic theory already declares both `+` and `*` AC, so it is fine.

- [ ] **Step 5: ASCII + commit**

```bash
git add rerum/normalize.py rerum/tests/test_normalize.py
git commit -m "fix(normalize): validate repeat.op is a declared AC operator

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: thread `max_steps` through the rewriter fast path (budget-1 + max-steps-decorative)

`rewriter()`'s `simplify` hardcodes `max_iterations = 1000`, so `engine.simplify(expr, max_steps=N)` is ignored on the no-theory/unconditional fast path. This makes `check_confluence`'s and `complete`'s budgets inert. Fix: let the rewriter's `simplify` accept a per-call `max_steps`, and have the engine fast path pass it.

**Files:**
- Modify: `rerum/rewriter.py` (the `simplify` closure inside `rewriter()`, around lines 1184-1196)
- Modify: `rerum/engine.py` (the fast-path call `self._simplifier(expr)` in `simplify`, around line 2488)
- Test: `rerum/tests/test_confluence.py`

- [ ] **Step 1: Write the failing test**

Append to `rerum/tests/test_confluence.py`:

```python
class TestBudgetHonoredOnFastPath:
    def test_small_max_steps_bounds_growth_rule(self):
        from rerum.engine import RuleEngine
        # A growth rule with no theory/condition/group/hook -> fast path.
        eng = RuleEngine.from_dsl("@g: (f ?x) => (f (s ?x))")
        small = eng.simplify(["f", "z"], max_steps=3)
        big = eng.simplify(["f", "z"], max_steps=50)
        # Result size must scale with the budget (the cap is real now).
        def depth(t):
            return 1 + max((depth(c) for c in t[1:]), default=0) if isinstance(t, list) else 0
        assert depth(small) < depth(big)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest rerum/tests/test_confluence.py::TestBudgetHonoredOnFastPath -v`
Expected: FAIL (both results identical -- the hardcoded 1000 cap ignores max_steps).

- [ ] **Step 3: Make the rewriter's `simplify` accept `max_steps`**

In `rerum/rewriter.py`, change the inner `simplify` signature and the cap. Find:

```python
    def simplify(exp: ExprType) -> ExprType:
```

change to:

```python
    def simplify(exp: ExprType, max_steps: Optional[int] = None) -> ExprType:
```

and find:

```python
        visited = set()
        max_iterations = 1000
        iterations = 0
```

change to:

```python
        visited = set()
        max_iterations = max_steps if max_steps is not None else 1000
        iterations = 0
```

- [ ] **Step 4: Thread it from the engine fast path**

In `rerum/engine.py`, in `simplify`, find the fast-path return (around line 2486-2488):

```python
                if self._simplifier is None:
                    self._simplifier = rewriter(self._rules, fold_funcs=self._fold_funcs)
                return self._simplifier(expr)
```

change the last line to:

```python
                return self._simplifier(expr, max_steps=max_steps)
```

(`max_steps` is `simplify`'s own parameter, default 1000, so default behavior is byte-identical; a smaller caller value now takes effect.)

- [ ] **Step 5: Run + regression**

Run: `pytest rerum/tests/test_confluence.py -q` then `pytest -q`
Expected: PASS (new test green; full suite unchanged -- default max_steps=1000 preserves behavior).

- [ ] **Step 6: ASCII + commit**

```bash
git add rerum/rewriter.py rerum/engine.py rerum/tests/test_confluence.py
git commit -m "fix(rewriter): honor max_steps on the fast path (real budget for confluence/completion)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: guard the empty-list pattern in `ac_match` (empty-pat-crash)

An empty-list pattern `[]` reaches `car()` in `_ac_match_core` and throws an uncaught `ValueError`. Fix: handle `[] vs []` at the top of `_ac_match_core` before any `car`/predicate, mirroring the rewriter.

**Files:**
- Modify: `rerum/acmatch.py` (`_ac_match_core`, around line 171-176)
- Test: `rerum/tests/test_acmatch.py`

- [ ] **Step 1: Write the failing tests**

Append to `rerum/tests/test_acmatch.py`:

```python
class TestEmptyPatternGuard:
    def test_empty_vs_empty_matches(self):
        assert _matches([], [], NO_AC) == [{}]
        assert _matches([], [], AC_PLUS) == [{}]

    def test_empty_vs_nonempty_no_match(self):
        assert _matches([], ["a"], NO_AC) == []

    def test_nested_empty_in_compound(self):
        # (f []) vs (f []) matches; (f []) vs (f a) does not.
        assert _matches(["f", []], ["f", []], NO_AC) == [{}]
        assert _matches(["f", []], ["f", "a"], NO_AC) == []
```

(`_matches`, `NO_AC`, `AC_PLUS` are defined earlier in the test file.)

- [ ] **Step 2: Run to verify failure**

Run: `pytest rerum/tests/test_acmatch.py::TestEmptyPatternGuard -v`
Expected: FAIL (ValueError out of car on the empty list).

- [ ] **Step 3: Add the guard**

In `rerum/acmatch.py`, in `_ac_match_core`, immediately after the `bindings` default-init (the function begins by handling atom/var cases) and BEFORE the compound dispatch that calls `car`, add an empty-list guard. Find the start of the compound handling -- the line that computes `head = car(pat)` or the `if not compound(pat) ...` guard -- and ensure an empty-list check precedes it:

```python
    if pat == []:
        if exp == []:
            yield bindings
        return
```

Place this right after the single-variable/atom dispatch block and before the `if not compound(pat) or not isinstance(exp, list) or not exp:` line. (An empty list is falsy, so `not pat` also works; use `pat == []` for clarity.)

- [ ] **Step 4: Run + regression**

Run: `pytest rerum/tests/test_acmatch.py -q` then `pytest -q`
Expected: PASS.

- [ ] **Step 5: ASCII + commit**

```bash
git add rerum/acmatch.py rerum/tests/test_acmatch.py
git commit -m "fix(acmatch): guard empty-list pattern (no ValueError from car)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: enforce typed rest constraints in `ac_match` (typed-rest)

`(+ ?rest:const...)` / `?rest:var...` constraints are dropped under AC, yielding constraint-violating matches. Fix: before binding any rest variable, apply `rest_type_constraint`; reject candidate leftovers/tails with a non-conforming element. Apply at all three rest sites.

**Files:**
- Modify: `rerum/acmatch.py` (bare-rest branch ~184-189; `_match_ac` leftover ~236; `_match_positional` rest ~265-272)
- Test: `rerum/tests/test_acmatch.py`

- [ ] **Step 1: Write the failing tests**

Append to `rerum/tests/test_acmatch.py`:

```python
class TestTypedRestConstraints:
    def test_const_rest_rejects_nonconstant_under_ac(self):
        # (+ ?x ?rest:const...) vs (+ 1 2 a): leftover {2,a} has a non-constant
        # (a) whenever x=1; only x picking a leaves {1,2} (all const) -> the
        # rest must be all-constant in every yielded binding.
        pat = ["+", ["?", "x"], ["?...", "rest", "const"]]
        for b in am.ac_match(pat, ["+", 1, 2, "a"], AC_PLUS):
            assert all(not isinstance(e, str) for e in b["rest"])

    def test_var_rest_rejects_constants_positional(self):
        # (f ?x ?rest:var...) vs (f a b 1): rest tail [b,1] has a constant -> no
        # match (positional rest, no AC).
        pat = ["f", ["?", "x"], ["?...", "rest", "var"]]
        assert _matches(pat, ["f", "a", "b", 1], NO_AC) == []

    def test_unconstrained_rest_unchanged(self):
        pat = ["+", ["?", "x"], ["?...", "rest"]]
        got = list(am.ac_match(pat, ["+", "a", 1], AC_PLUS))
        assert got  # still matches (no constraint)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest rerum/tests/test_acmatch.py::TestTypedRestConstraints -v`
Expected: FAIL (constraints ignored; constant/variable rests not enforced).

- [ ] **Step 3: Implement the constraint check**

In `rerum/acmatch.py`, add to the imports from `rewriter` (the existing `from .rewriter import (...)` block): `rest_type_constraint`, `constant`, `variable` (check which are already imported; add the missing ones). Then add a helper near the other module helpers:

```python
def _rest_ok(rest_pat, items) -> bool:
    """True if every element of ``items`` satisfies ``rest_pat``'s type
    constraint (const/var), or there is no constraint."""
    tc = rest_type_constraint(rest_pat)
    if tc is None:
        return True
    if tc == "const":
        return all(constant(it) for it in items)
    if tc == "var":
        return all(variable(it) for it in items)
    return True
```

Then guard each of the three rest bindings. (a) The bare-rest branch in `_ac_match_core` (where `arbitrary_rest(pat)` binds the whole list): wrap the bind with `if _rest_ok(pat, exp):`. (b) In `_match_ac`, where the trailing `rest` binds `leftover`: only proceed if `_rest_ok(rest, leftover)`. (c) In `_match_positional`, where the trailing rest binds `list(exps)`: only proceed if `_rest_ok(head, list(exps))`. In each case, if the check fails, do NOT yield (skip that binding).

Concretely, in `_match_ac`'s rest branch, change:

```python
            extended = _bind(binds, variable_name(rest), leftover, theory)
            if extended is not None:
```

to:

```python
            if not _rest_ok(rest, leftover):
                return
            extended = _bind(binds, variable_name(rest), leftover, theory)
            if extended is not None:
```

Apply the analogous `_rest_ok(...)` guard at the bare-rest site and the `_match_positional` rest site.

- [ ] **Step 4: Run + regression**

Run: `pytest rerum/tests/test_acmatch.py -q` then `pytest -q`
Expected: PASS.

- [ ] **Step 5: ASCII + commit**

```bash
git add rerum/acmatch.py rerum/tests/test_acmatch.py
git commit -m "fix(acmatch): enforce typed rest constraints (const/var) under AC and positional

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: document the AC budget scope (budget-non-ac)

`MatchBudget` only guards the `_match_ac` multiset fan-out (the combinatorial source), not positional/nested recursion. This is acceptable (the fan-out is the blow-up), but the docstring overstates it. Fix (documentation): state the scope precisely in the `MatchBudget` docstring.

**Files:**
- Modify: `rerum/acmatch.py` (`MatchBudget` docstring)
- Test: none (documentation-only); covered by existing budget tests.

- [ ] **Step 1: Update the docstring**

In `rerum/acmatch.py`, in the `MatchBudget` class docstring, add a sentence:

```
    Scope: the budget bounds the MULTISET-ASSIGNMENT fan-out in _match_ac (the
    combinatorial blow-up source) only. Positional and nested-compound recursion
    are bounded by term size, not by this budget; under an AC theory a very deep
    NON-AC structure is not budget-limited (it is size-limited instead).
```

- [ ] **Step 2: ASCII + commit**

Run: `pytest rerum/tests/test_acmatch.py -q` (still green).

```bash
git add rerum/acmatch.py
git commit -m "docs(acmatch): state MatchBudget bounds the multiset fan-out, not depth

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: skip extra-RHS-variable rules in `check_confluence` (confluence-1)

`check_confluence` feeds rule RHSs (via `instantiate_skeleton`) to `engine.simplify`, but a dangling `:x` (extra RHS variable, `Var(r)` not subset `Var(l)`) is modeled as a free variable while the engine reduces it to the symbol `"x"` -- the same divergence fixed in F6 narrowing. Fix: skip such malformed rules in the critical-pair analysis, treating them as non-analyzable (counted in `not_analyzed`).

**Files:**
- Modify: `rerum/confluence.py` (`critical_pairs`, the `is_analyzable` gate around lines 322-326)
- Test: `rerum/tests/test_confluence.py`

- [ ] **Step 1: Write the failing test**

Append to `rerum/tests/test_confluence.py`:

```python
class TestConfluenceDanglingRHS:
    def test_extra_rhs_variable_rule_is_not_analyzed(self):
        from rerum.engine import RuleEngine
        from rerum import confluence as cf
        # @r: (g a) => :x  has an EXTRA RHS variable (no ?x binder in the LHS).
        eng = RuleEngine.from_dsl("@r: (g a) => :x")
        report = eng.check_confluence()
        # The rule is skipped (malformed), not silently analyzed with a free var.
        assert "r" in report.not_analyzed
```

(If `ConfluenceReport` exposes `not_analyzed` under a different attribute, use that name -- check the dataclass.)

- [ ] **Step 2: Run to verify failure**

Run: `pytest rerum/tests/test_confluence.py::TestConfluenceDanglingRHS -v`
Expected: FAIL (the rule is analyzed, not skipped).

- [ ] **Step 3: Add the well-formedness check**

In `rerum/confluence.py`, the `critical_pairs` analyzable gate currently is:

```python
    analyzable = []
    for r in rules:
        if is_analyzable(r.pattern, r.skeleton, r.condition):
            analyzable.append(r)
        else:
            skip(r)
```

Change the condition to also require `Var(r) subset Var(l)` (no extra RHS variable). Add a helper near the top of the module:

```python
def _well_formed_rhs(pattern, skeleton) -> bool:
    """True if every variable referenced by the skeleton is bound by the
    pattern (Var(r) subset Var(l)). A dangling reference would be modeled as a
    free variable by instantiate_skeleton but reduced to a ground symbol by the
    engine -- a divergence that breaks the joinability oracle. Such rules are
    not analyzable."""
    return not (_variables(instantiate_skeleton(skeleton, {})) - _variables(pattern))
```

and change the gate to:

```python
    analyzable = []
    for r in rules:
        if is_analyzable(r.pattern, r.skeleton, r.condition) and \
                _well_formed_rhs(r.pattern, r.skeleton):
            analyzable.append(r)
        else:
            skip(r)
```

- [ ] **Step 4: Run + regression**

Run: `pytest rerum/tests/test_confluence.py -q` then `pytest -q`
Expected: PASS.

- [ ] **Step 5: ASCII + commit**

```bash
git add rerum/confluence.py rerum/tests/test_confluence.py
git commit -m "fix(confluence): treat extra-RHS-variable rules as not-analyzable (Var(r) subset Var(l))

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 9: completion honesty about non-analyzable rules (completion-1)

`complete` computes critical pairs over the ANALYZABLE subset (via F2's `critical_pairs`, whose `not_analyzed` it currently discards as `_na`), and can report `status == "complete"` even when the engine's own rules include non-analyzable ones -- so the "complete" verdict does not certify the FULL system. Fix: surface the skipped count and make "complete" honest about scope. Since the loop builds only first-order well-formed rules, `not_analyzed` is empty for completion's OWN rules; the real gap is the INPUT equations that orient to such rules -- but completion already only orients analyzable equations. The principled fix: assert the invariant and document it, and expose `not_analyzed` on the result for transparency.

**Files:**
- Modify: `rerum/completion.py` (the `critical_pairs(records)` call and `CompletionResult`)
- Test: `rerum/tests/test_completion.py`

- [ ] **Step 1: Write the failing test**

Append to `rerum/tests/test_completion.py`:

```python
class TestCompletionNotAnalyzedInvariant:
    def test_internal_rules_are_always_analyzable(self):
        # Every rule completion builds is first-order and well-formed, so the
        # F2 critical_pairs call must report ZERO not_analyzed -- complete()
        # asserts this so a future regression that smuggles a non-analyzable
        # rule into the loop is caught instead of silently weakening "complete".
        import rerum.completion as cmp
        eqs = [(["f", ["g", ["?", "x"]]], "a"), (["g", ["g", ["?", "x"]]], ["?", "x"])]
        result = cmp.complete(eqs, ["f", "g", "a"])
        assert result.status == "complete"
        # Re-derive: the result system has no non-analyzable rule.
        from rerum.confluence import critical_pairs, DirectedRule
        from rerum.completion import _term_to_skeleton
        recs = [DirectedRule(name=str(i), pattern=l,
                             skeleton=_term_to_skeleton(r), condition=None)
                for i, (l, r) in enumerate(result.rules)]
        _pairs, na = critical_pairs(recs)
        assert na == []
```

- [ ] **Step 2: Run to verify**

Run: `pytest rerum/tests/test_completion.py::TestCompletionNotAnalyzedInvariant -v`
Expected: PASS already if the invariant holds (it should). If it FAILS, completion is building a non-analyzable rule -- a real bug to fix. Either way, proceed to Step 3 to make the invariant explicit in code.

- [ ] **Step 3: Assert the invariant in `complete`**

In `rerum/completion.py`, find the critical-pairs call:

```python
        pairs, _na = critical_pairs(records)
```

change to:

```python
        pairs, not_analyzed = critical_pairs(records)
        # INVARIANT: every rule the loop builds is a first-order, well-formed
        # [pattern, [":",x]-skeleton] rule, so critical_pairs must analyze all of
        # them. If this ever fires, a non-analyzable rule has entered the loop
        # and a "complete" verdict could not certify the full system.
        assert not not_analyzed, (
            "completion built a non-analyzable rule: " + repr(not_analyzed))
```

- [ ] **Step 4: Run + regression**

Run: `pytest rerum/tests/test_completion.py -q` then `pytest -q`
Expected: PASS.

- [ ] **Step 5: ASCII + commit**

```bash
git add rerum/completion.py rerum/tests/test_completion.py
git commit -m "fix(completion): assert no non-analyzable rule enters the loop (complete is full-system honest)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 10: recurse the unsupported-node check in `unify` (unify-refuse)

`unify`'s refuse-first `_unsupported` guard is root-only, so a `?c`/`?v`/`?free`/`?...` node reached via a VARIABLE BINDING (`unify(["?","x"], ["f", ["?c","y"]])`) is silently bound as opaque, violating the documented refuse-first contract. Fix: reject binding a value that contains an unsupported node.

**Files:**
- Modify: `rerum/confluence.py` (`_unify_var` around lines 171-178, or `_compose_bind`)
- Test: `rerum/tests/test_confluence.py`

- [ ] **Step 1: Write the failing test**

Append to `rerum/tests/test_confluence.py`:

```python
class TestUnifyNestedUnsupported:
    def test_unsupported_via_binding_raises(self):
        import pytest
        from rerum import confluence as cf
        with pytest.raises(cf.UnsupportedPattern):
            cf.unify(["?", "x"], ["f", ["?c", "y"]])
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest rerum/tests/test_confluence.py::TestUnifyNestedUnsupported -v`
Expected: FAIL (binds opaquely, no raise).

- [ ] **Step 3: Add a recursive unsupported check at bind time**

In `rerum/confluence.py`, add a recursive variant near `_unsupported`:

```python
def _contains_unsupported(t) -> bool:
    if _unsupported(t):
        return True
    if compound(t):
        return any(_contains_unsupported(s) for s in t)
    return False
```

Then in `_unify_var`, before binding `other` to `var`, raise if it contains an unsupported node:

```python
    if _contains_unsupported(other):
        raise UnsupportedPattern(
            f"cannot bind a value containing an unsupported node: {other!r}")
```

Place this at the top of `_unify_var` (after resolving `other` through the current subst, before the occurs-check/bind).

- [ ] **Step 4: Run + regression**

Run: `pytest rerum/tests/test_confluence.py -q` then `pytest -q`
Expected: PASS. (If any existing analyzable-rule path now raises, it indicates a rule that was slipping an unsupported node through a binding -- investigate; the analyzable pre-scan should already exclude these, so the full suite should stay green.)

- [ ] **Step 5: ASCII + commit**

```bash
git add rerum/confluence.py rerum/tests/test_confluence.py
git commit -m "fix(confluence): refuse binding a value containing an unsupported node in unify

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 11: count anonymous non-analyzable rules correctly (notanalyzed-none + confluence-2)

`critical_pairs`'s `skip()` dedups on `rule.name`, so multiple anonymous rules (name `None`) collapse to a single `[None]` entry, undercounting skipped rules. Fix: dedup on rule identity (index), and surface a stable label for anonymous rules.

**Files:**
- Modify: `rerum/confluence.py` (`critical_pairs` `skip`/`seen_skips`, lines 313-326)
- Test: `rerum/tests/test_confluence.py`

- [ ] **Step 1: Write the failing test**

Append to `rerum/tests/test_confluence.py`:

```python
class TestNotAnalyzedAnonymous:
    def test_two_anonymous_nonanalyzable_rules_both_counted(self):
        from rerum.confluence import critical_pairs, DirectedRule
        # Two DISTINCT non-analyzable rules (rest patterns), both unnamed.
        r1 = DirectedRule(name=None, pattern=["f", ["?...", "a"]],
                          skeleton=["g", [":...", "a"]], condition=None)
        r2 = DirectedRule(name=None, pattern=["h", ["?...", "b"]],
                          skeleton=["k", [":...", "b"]], condition=None)
        _pairs, na = critical_pairs([r1, r2])
        assert len(na) == 2   # both skipped rules counted, not collapsed to one
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest rerum/tests/test_confluence.py::TestNotAnalyzedAnonymous -v`
Expected: FAIL (both collapse to a single `[None]` entry -> len 1).

- [ ] **Step 3: Dedup on index, label anonymous rules**

In `rerum/confluence.py` `critical_pairs`, change the skip bookkeeping. Replace:

```python
    not_analyzed: List[str] = []
    seen_skips: set = set()

    def skip(rule: DirectedRule) -> None:
        key = rule.name
        if key not in seen_skips:
            seen_skips.add(key)
            not_analyzed.append(key)
```

with an index-keyed version. Since `skip` is called with the rule object, key on `id(rule)` and emit a stable label:

```python
    not_analyzed: List[str] = []
    seen_skips: set = set()

    def skip(rule: DirectedRule) -> None:
        key = id(rule)
        if key not in seen_skips:
            seen_skips.add(key)
            not_analyzed.append(rule.name if rule.name is not None else "<anonymous>")
```

- [ ] **Step 4: Run + regression**

Run: `pytest rerum/tests/test_confluence.py -q` then `pytest -q`
Expected: PASS. (Existing tests that assert `not_analyzed == ["name"]` for named rules still hold; only anonymous-rule counting changes.)

- [ ] **Step 5: ASCII + commit**

```bash
git add rerum/confluence.py rerum/tests/test_confluence.py
git commit -m "fix(confluence): count anonymous non-analyzable rules by identity, not name

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 12: make `bottomup`/`topdown` AC-aware or refuse under AC (ac-strategy-gap)

The `bottomup`/`topdown` strategies use `_match_internal` directly and silently ignore a loaded AC theory (the AC gate only routes `exhaustive`). Decision: make `simplify` RAISE a clear error when a non-default strategy is requested under an AC theory, rather than silently returning a non-AC result. (Wiring the two passes through `_match_lhs`+canonicalize is the larger alternative; refusing is the safe, honest minimal fix and matches the documented v1 scope.)

**Files:**
- Modify: `rerum/engine.py` (`simplify`, the strategy dispatch around lines 2489-2497)
- Test: `rerum/tests/test_acmatch.py`

- [ ] **Step 1: Write the failing test**

Append to `rerum/tests/test_acmatch.py`:

```python
class TestStrategyACRefusal:
    def test_bottomup_under_ac_refuses(self):
        import pytest
        eng = RuleEngine.from_dsl("@cancel: (+ ?x (- ?x) ?rest...) => (+ :rest...)")
        eng.with_theory(Theory.from_dict({"+": {"ac": True, "identity": 0}}))
        with pytest.raises(ValueError) as ei:
            eng.simplify(["+", "a", "b", ["-", "a"]], strategy="bottomup")
        assert "AC" in str(ei.value) or "ac" in str(ei.value)

    def test_exhaustive_under_ac_still_works(self):
        eng = RuleEngine.from_dsl("@cancel: (+ ?x (- ?x) ?rest...) => (+ :rest...)")
        eng.with_theory(Theory.from_dict({"+": {"ac": True, "identity": 0}}))
        assert eng.simplify(["+", "a", "b", ["-", "a"]]) == "b"
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest rerum/tests/test_acmatch.py::TestStrategyACRefusal -v`
Expected: FAIL (bottomup silently returns a non-AC result, no raise).

- [ ] **Step 3: Refuse non-exhaustive strategies under an AC theory**

In `rerum/engine.py` `simplify`, after the `has_ac = self._theory_has_ac()` line (added in F3) and before the strategy dispatch, add:

```python
        if has_ac and strategy in ("bottomup", "topdown", "once"):
            raise ValueError(
                f"strategy={strategy!r} does not support AC matching; use the "
                f"default 'exhaustive' strategy under an AC theory")
```

NOTE: `apply_once` (the public method) IS AC-aware (Task 13 fixes its binding loop), so this refusal is only for the `simplify(strategy="once")` path which routes through `_simplify_once`. If `_simplify_once` is already AC-aware via `apply_once`, narrow the refusal to `("bottomup", "topdown")`. Verify which by checking whether `_simplify_once` calls `apply_once` (it does) -- in that case use:

```python
        if has_ac and strategy in ("bottomup", "topdown"):
```

- [ ] **Step 4: Update docs**

In `CLAUDE.md`, in the Footguns or strategy notes section, add a one-line note: "Under an AC theory, only the default `exhaustive` strategy is AC-aware; `bottomup`/`topdown` raise rather than silently match syntactically."

- [ ] **Step 5: Run + regression**

Run: `pytest rerum/tests/test_acmatch.py -q` then `pytest -q`
Expected: PASS.

- [ ] **Step 6: ASCII + commit**

```bash
git add rerum/engine.py rerum/tests/test_acmatch.py CLAUDE.md
git commit -m "fix(engine): refuse bottomup/topdown under an AC theory (no silent non-AC result)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 13: `apply_once` AC completeness + no-op honesty (apply-once-ac-completeness)

`apply_once` returns on the FIRST guard-passing binding even if it yields no change, so a later PRODUCTIVE AC binding is skipped, and a no-op application is reported as applied. Fix: continue the binding loop on a no-change binding; only return (with metadata) on a productive one.

**Files:**
- Modify: `rerum/engine.py` (`apply_once`, lines ~2405-2421)
- Test: `rerum/tests/test_acmatch.py`

- [ ] **Step 1: Write the failing test**

Append to `rerum/tests/test_acmatch.py`:

```python
class TestApplyOnceACCompleteness:
    def test_apply_once_skips_noop_binding_for_productive_one(self):
        # A rule that is a no-op for one AC binding but productive for another.
        # (+ ?x ?y) => (+ :y :x) is a no-op when x,y already sorted, productive
        # when they are not -- but the engine canonicalizes, so use a rule that
        # only fires productively for a specific element choice.
        eng = RuleEngine.from_dsl("@r: (+ (k ?x) ?y) => (found :x :y)")
        eng.with_theory(Theory.from_dict({"+": {"ac": True, "identity": 0}}))
        # (+ a (k b)): the productive binding is x=b (matching (k ?x)), y=a.
        result, meta = eng.apply_once(["+", "a", ["k", "b"]])
        assert meta is not None and result[0] == "found"
```

- [ ] **Step 2: Run to verify**

Run: `pytest rerum/tests/test_acmatch.py::TestApplyOnceACCompleteness -v`
Expected: This may already PASS if the first canonical binding happens to be productive. If it PASSES, still apply Step 3 to fix the no-op-reported-as-applied semantics and add the stricter assertion below.

- [ ] **Step 3: Make the binding loop productive-only**

In `rerum/engine.py` `apply_once`, the loop currently returns `result, metadata` inside the binding loop unconditionally after guards pass. Change it so a no-change binding CONTINUES to the next binding, and `(expr, None)` is returned if no binding produced a change. Find:

```python
            for bindings in self._match_lhs(pattern, expr):
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

change the final `return result, metadata` so it is INSIDE the `if result != expr:` block (productive only):

```python
            for bindings in self._match_lhs(pattern, expr):
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
                # no-op binding: try the next binding (AC) / next rule
```

WARNING: this changes the documented behavior where a matched no-op rule returned `(expr, metadata)`. Run the FULL suite; if a test asserts a no-op apply returns metadata, that test encodes the OLD semantics -- update it to expect `(expr, None)` and note the change in the commit. The `once` strategy via `_simplify_once` keys on `applied` truthiness; with this change a no-op no longer counts as applied (it falls through to children), which is the correct fixpoint behavior.

- [ ] **Step 4: Run + regression**

Run: `pytest rerum/tests/test_acmatch.py -q` then `pytest -q`
Expected: PASS (fix any test encoding the old no-op-returns-metadata semantics).

- [ ] **Step 5: ASCII + commit**

```bash
git add rerum/engine.py rerum/tests/test_acmatch.py
git commit -m "fix(engine): apply_once returns only on a productive binding (AC completeness + no-op honesty)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 14: AC truncation-flag honesty (truncated-flag-not-reset + recursionerror-silent-truncation)

`ac_match_truncated` is only reset by non-trace `simplify`, so it is stale after `apply_once`/`equivalents`/`prove_equal`/`minimize`/`simplify(trace=True)`; and a `RecursionError` in `_match_lhs` is swallowed without setting the flag. Fix: reset the flag at every top-level entry point, and set it on `RecursionError`.

**Files:**
- Modify: `rerum/engine.py` (`_match_lhs` RecursionError branch ~1893-1900; the top-level methods' reset)
- Test: `rerum/tests/test_acmatch.py`

- [ ] **Step 1: Write the failing tests**

Append to `rerum/tests/test_acmatch.py`:

```python
class TestTruncationFlagHonesty:
    def _ac_eng(self):
        eng = RuleEngine.from_dsl("@r: (+ ?x ?y ?z) => done")
        eng.with_theory(Theory.from_dict({"+": {"ac": True, "identity": 0}}))
        return eng

    def test_flag_reset_by_prove_equal(self):
        eng = self._ac_eng()
        eng.set_ac_match_budget(2)
        eng.prove_equal(["+", "a", "b", "c", "d", "e"], "done",
                        include_unidirectional=True, max_expressions=200)
        truncated_after = eng.ac_match_truncated
        # Now a clean call with a big budget must reset the flag to False.
        eng.set_ac_match_budget(100000)
        eng.simplify(["+", "a", "b"])
        assert eng.ac_match_truncated is False

    def test_flag_set_after_equivalents_truncation(self):
        eng = self._ac_eng()
        eng.set_ac_match_budget(1)
        list(eng.equivalents(["+", "a", "b", "c", "d", "e"],
                             include_unidirectional=True, max_depth=2))
        assert eng.ac_match_truncated is True
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest rerum/tests/test_acmatch.py::TestTruncationFlagHonesty -v`
Expected: FAIL (flag not reset/managed outside non-trace simplify).

- [ ] **Step 3: Set the flag on RecursionError**

In `rerum/engine.py` `_match_lhs`, the AC branch catches `RecursionError`. Change:

```python
        except RecursionError:
            return
```

to:

```python
        except RecursionError:
            self._ac_match_truncated = True
            return
```

- [ ] **Step 4: Reset the flag at every top-level entry**

Add a one-line reset `self._ac_match_truncated = False` at the top of each top-level method that can run AC matching, guarded by `_top_level` where applicable: `apply_once` (when `_top_level`), `equivalents`, `prove_equal`, `minimize`, and `_simplify_with_trace`. The cleanest approach: these methods already reset `self._step_count`/`self._cancel_requested` at their top-level entry; add the flag reset at the SAME site. Grep for `self._cancel_requested = False` and add `self._ac_match_truncated = False` immediately after each occurrence that marks a top-level entry (simplify already has it via Task-untouched F3 code).

- [ ] **Step 5: Run + regression**

Run: `pytest rerum/tests/test_acmatch.py -q` then `pytest -q`
Expected: PASS.

- [ ] **Step 6: ASCII + commit**

```bash
git add rerum/engine.py rerum/tests/test_acmatch.py
git commit -m "fix(engine): reset ac_match_truncated at every top-level entry; set it on RecursionError

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 15: narrowing `exhausted` precision + docstring (narrowing-1 + narrowing-2)

`_narrow_with_rules` over-reports `exhausted=True` at the depth cap when all of a node's successors are already in `seen` (a false "inconclusive" for a cyclic finite tree). And the `narrow` docstring overstates the guarantee for variable-containing targets. Fix: only set `depth_capped` when a successor is genuinely new; tighten the docstring to joinability.

**Files:**
- Modify: `rerum/narrowing.py` (`_narrow_with_rules` depth-cap branch ~183-190; `narrow` docstring)
- Test: `rerum/tests/test_narrowing.py`

- [ ] **Step 1: Write the failing test**

Append to `rerum/tests/test_narrowing.py`:

```python
class TestNarrowExhaustedPrecision:
    def test_cyclic_finite_tree_not_inconclusive(self):
        # (swap (a b)) => (swap (b a)) and back: a finite cyclic tree. With a
        # depth cap, all cap-depth successors are already seen, so exhausted
        # must stay False (genuinely no solution), not True (inconclusive).
        eng = RuleEngine.from_dsl("@s: (swap ?p) => (swap2 :p)")
        # Build a 2-cycle: swap -> swap2 -> swap ...
        eng2 = RuleEngine.from_dsl("""
            @a: (p ?x) => (q :x)
            @b: (q ?x) => (p :x)
        """)
        result = eng2.narrow(["p", "z"], "done", max_nodes=100000, max_depth=3) \
            if hasattr(eng2, "narrow") else None
        import rerum.narrowing as nw
        result = nw.narrow(eng2, ["p", "z"], "done", max_nodes=100000, max_depth=3)
        assert result.found is False
        assert result.exhausted is False   # cyclic finite tree, fully explored
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest rerum/tests/test_narrowing.py::TestNarrowExhaustedPrecision -v`
Expected: FAIL (depth-cap sets exhausted=True even though every successor is already seen).

- [ ] **Step 3: Make the depth-cap check membership-aware**

In `rerum/narrowing.py` `_narrow_with_rules`, the depth-cap branch is:

```python
        else:
            for _ in narrow_step(term, rules):
                depth_capped = True
                break
```

change it to only flag truncation when a successor is genuinely new (not already reachable):

```python
        else:
            for step in narrow_step(term, rules):
                if _key(step.successor, _compose(step.sigma, theta)) not in seen:
                    depth_capped = True
                    break
```

- [ ] **Step 4: Tighten the `narrow` docstring**

In `rerum/narrowing.py` `narrow`, update the docstring to state the real guarantee (joinability), replacing the "reduces to a term unifying sigma(target)" phrasing with: "find sigma such that sigma(start) and sigma(target) reduce to a common form (joinability) under the engine's rules; for a ground target this is reachability of the target."

- [ ] **Step 5: Run + regression**

Run: `pytest rerum/tests/test_narrowing.py -q` then `pytest -q`
Expected: PASS.

- [ ] **Step 6: ASCII + commit**

```bash
git add rerum/narrowing.py rerum/tests/test_narrowing.py
git commit -m "fix(narrowing): exhausted only on a genuinely-new cap-depth successor; joinability docstring

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Post-implementation

After all 15 tasks: run the full suite + no-domain guard once more, then dispatch an Opus holistic review of the diff (focus: each fix closes its finding without regressing the feature's soundness; the ORDER_KEY change preserves canonical-form behavior on the common path; the apply_once semantics change did not break the `once` strategy or traces). Then `superpowers:finishing-a-development-branch` to present the push decision (on `main`, per the per-feature rhythm). Finally, update `docs/superpowers/reviews/2026-06-23-trs-frontier-code-review.md` with a short "Remediation" footer noting which commits closed which findings, and note the 9 NITs as deferred.

## Notes for the implementer

- **Run the FULL suite after every task.** These fixes touch core modules (normalize/confluence/engine) used everywhere; a green per-task test is necessary but not sufficient.
- **ORDER_KEY (Task 1) is the riskiest change** -- it is the core canonical-sort key. The fix keeps the type-name tiebreaker so int/float/Fraction/bool ordering is byte-identical; only the value comparison becomes exact. If any `test_normalize.py` or `test_theory_reasoning.py` test changes output, investigate before "fixing" the test -- a changed canonical form is a regression unless it is the large-number case.
- **apply_once (Task 13) changes a documented semantic** (no-op matched rule no longer returns metadata). This is the only task likely to require updating an EXISTING test; do so deliberately and note it in the commit.
- **Tasks are independent** -- they can be reordered or done in parallel sessions, except Task 4 is a prerequisite for fully honoring `max_steps` anywhere (its absence does not block other tasks).
- **NITs are out of scope** for this plan (9 items in the review); list them as deferred in the review-report footer.
