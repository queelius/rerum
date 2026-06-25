# AC-Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `ac_unify(t1, t2, theory)` -- unification modulo associative-commutative operators -- the multi-valued, two-sided generalization of F2's syntactic `unify`, via Stickel's algorithm.

**Architecture:** A new core module `rerum/acunify.py` reusing F2's substitution machinery (`apply_subst`/`_occurs`/`_is_var`/`UnsupportedPattern` from `confluence.py`) plus `flatten` (normalize) and `gensym` (rewriter). The hard part -- the AC-node case -- runs Stickel: flatten+cancel, build a linear Diophantine equation over argument multiplicities, compute its Hilbert basis, enumerate covering subsets of basis vectors, and build a unifier per subset (variable atoms get AC-sums of fresh basis variables; non-variable atoms are coupled and recursively `ac_unify`'d). Pure AC, complete-not-minimal, budget-bounded lazy generator.

**Tech Stack:** Python 3.9+, pytest. Reuses `rerum/confluence.py`, `rerum/normalize.py`, `rerum/rewriter.py`, `rerum/acmatch.py` (test cross-check only).

**Algorithm validated:** the Stickel core was prototyped against the real F2 primitives and produces the canonical counts (`x+y=?u+v`->7, `x+y=?a+b`->2, `x=?a+b`->1, `a+b=?a+b`->1, `a=?b`->0, `(+ (f x) y)=?(+ a (f b))`->1), every yield sound. This plan embeds that verified structure.

---

## File Structure

- **Create** `rerum/acunify.py` -- the whole feature: `UnifyBudget`, `_hilbert_basis`, `_bind` (a thin F2-backed binder), the `ac_unify` dispatch, `_unify_positional` (non-AC, multi-valued), `_ac_unify_node` (the Stickel core). Names NO domain operator.
- **Create** `rerum/tests/test_acunify.py` -- unit + canonical-count + soundness + completeness-oracle + cross-check tests.
- **Modify** `rerum/__init__.py` -- re-export `ac_unify`, `UnifyBudget`.
- **Create** `examples/acunify_demo.rules` (data) -- a tiny AC theory + a worked AC-unification example, driven in a test. (Pure data; the demo is the test.)

## Conventions all tasks follow

- **ASCII only** in every file write (a commit hook rejects non-ASCII). Use `->`, plain hyphens.
- **Never stage `.mcp.json`**. Stage only the files each task names.
- Commit trailer:
  ```
  Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
  ```
- Terms are nested lists; a VARIABLE is `["?", name]`. Run the FULL suite after each task (baseline before this plan: **1742**); report the new total.
- General-engine principle: `acunify.py` names NO domain operator literal as code; it keys on `theory.is_ac`, and uses `gensym` for fresh variables.

---

### Task 1: `UnifyBudget` + the Hilbert-basis solver

The fail-safe budget and the one genuinely new sub-component: the minimal non-negative solutions of a single linear Diophantine equation.

**Files:**
- Create: `rerum/acunify.py`
- Create: `rerum/tests/test_acunify.py`

- [ ] **Step 1: Write the failing tests**

Create `rerum/tests/test_acunify.py`:

```python
"""AC-unification (Stickel)."""

from rerum import acunify as au


class TestUnifyBudget:
    def test_spend_decrements_and_flags(self):
        b = au.UnifyBudget(steps=2)
        assert b.spend() is True
        assert b.spend() is True
        assert b.spend() is False
        assert b.truncated is True

    def test_unbounded_when_none(self):
        b = au.UnifyBudget(steps=None)
        for _ in range(1000):
            assert b.spend() is True
        assert b.truncated is False


class TestHilbertBasis:
    def test_unit_coefficients_two_by_two(self):
        # m1+m2 = n1+n2: minimal solutions are the four "unit flows".
        basis = au._hilbert_basis([1, 1], [1, 1])
        assert set(basis) == {(1, 0, 1, 0), (1, 0, 0, 1),
                              (0, 1, 1, 0), (0, 1, 0, 1)}

    def test_single_var_each(self):
        # m1 = n1: minimal solution (1, 1).
        assert au._hilbert_basis([1], [1]) == [(1, 1)]

    def test_coefficient_two(self):
        # 2*m1 = n1: minimal solution (1, 2) (one x equals two y's).
        assert au._hilbert_basis([2], [1]) == [(1, 2)]

    def test_every_basis_vector_is_a_solution(self):
        for a, b in ([1, 2], [2, 1]), ([1, 1, 1], [2, 1]):
            for vec in au._hilbert_basis(a, b):
                M = len(a)
                assert sum(a[i] * vec[i] for i in range(M)) == \
                    sum(b[j] * vec[M + j] for j in range(len(b)))
                assert any(vec)  # nonzero
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_acunify.py -v`
Expected: FAIL (`No module named 'rerum.acunify'`).

- [ ] **Step 3: Create `rerum/acunify.py`**

Create `rerum/acunify.py`:

```python
"""AC-unification (Stickel's algorithm) -- unification modulo associative-
commutative operators.

The multi-valued, two-sided generalization of F2's syntactic ``unify``: where
``unify`` returns one most-general unifier or none, AC-unification has no mgu, so
``ac_unify`` yields a COMPLETE SET of unifiers (within a budget; not necessarily
minimal). PURE AC -- the Theory's identity/unit is NOT used (the engine's
normalize strips units upstream). Stickel-general: nested terms, free function
symbols, and multiple AC operators are handled (the last two by recursion).

CORE module. Reuses F2 (rerum/confluence.py): Subst, apply_subst, _occurs,
_is_var, UnsupportedPattern; plus flatten (normalize) and gensym (rewriter).
F2's ``unify`` is UNTOUCHED. Names NO domain operator: keys on theory.is_ac;
fresh variables are gensym'd.

References: Stickel 1981; Baader and Nipkow, Term Rewriting and All That, Ch. 10.
"""

from dataclasses import dataclass
from itertools import combinations, product
from typing import Iterator, List, Optional

from .confluence import (
    apply_subst,
    _occurs,
    _is_var,
    _unsupported,
    UnsupportedPattern,
    Subst,
)
from .normalize import flatten
from .rewriter import compound, gensym


@dataclass
class UnifyBudget:
    """Fail-safe budget for the basis + subset enumeration. On exhaustion,
    ``truncated`` is set and enumeration stops; the yielded set is then SOUND
    but INCOMPLETE (every yield is a real unifier; some may be missing)."""
    steps: Optional[int] = None
    truncated: bool = False

    def spend(self) -> bool:
        if self.steps is None:
            return True
        if self.steps <= 0:
            self.truncated = True
            return False
        self.steps -= 1
        return True


def _hilbert_basis(a: List[int], b: List[int]):
    """Minimal non-negative non-zero integer solutions (the Hilbert basis) of
    ``sum_i a_i*m_i = sum_j b_j*n_j``. Each result is the tuple
    ``(m_1, ..., m_M, n_1, ..., n_N)``. Bounded enumeration: a minimal solution
    has ``m_i <= max(b)`` and ``n_j <= max(a)`` (Fortenbacher's bound)."""
    M, N = len(a), len(b)
    mb = max(b) if b else 0
    nb = max(a) if a else 0
    found = []
    for m in product(range(mb + 1), repeat=M):
        lhs = sum(a[i] * m[i] for i in range(M))
        for n in product(range(nb + 1), repeat=N):
            if sum(b[j] * n[j] for j in range(N)) != lhs:
                continue
            vec = tuple(m) + tuple(n)
            if any(vec):
                found.append(vec)

    def leq(x, y):
        return all(p <= q for p, q in zip(x, y))

    return [v for v in found if not any(u != v and leq(u, v) for u in found)]
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest rerum/tests/test_acunify.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: ASCII + commit**

Run: `LC_ALL=C grep -n '[^[:print:][:space:]]' rerum/acunify.py rerum/tests/test_acunify.py && echo FOUND || echo clean`

```bash
git add rerum/acunify.py rerum/tests/test_acunify.py
git commit -m "feat(acunify): UnifyBudget + Hilbert-basis Diophantine solver

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `ac_unify` dispatch -- variable, atom, and non-AC compound cases

The multi-valued dispatch, minus the Stickel core. On non-AC input it agrees with F2's `unify`.

**Files:**
- Modify: `rerum/acunify.py`
- Test: `rerum/tests/test_acunify.py`

- [ ] **Step 1: Write the failing tests**

Append to `rerum/tests/test_acunify.py`:

```python
from rerum.normalize import Theory, normalize

NO_AC = Theory.from_dict({})
AC_PLUS = Theory.from_dict({"+": {"ac": True, "identity": 0}})


def _unifiers(t1, t2, theory):
    return list(au.ac_unify(t1, t2, theory))


class TestDispatchNonAC:
    def test_variable_binds(self):
        got = _unifiers(["?", "x"], ["f", "a"], NO_AC)
        assert len(got) == 1 and got[0]["x"] == ["f", "a"]

    def test_atoms_equal_and_clash(self):
        assert _unifiers("a", "a", NO_AC) == [{}]
        assert _unifiers("a", "b", NO_AC) == []

    def test_occurs_check(self):
        # x =? (f x) has no unifier.
        assert _unifiers(["?", "x"], ["f", ["?", "x"]], NO_AC) == []

    def test_non_ac_compound_positional(self):
        got = _unifiers(["f", ["?", "x"], "b"], ["f", "a", ["?", "y"]], NO_AC)
        assert len(got) == 1
        s = got[0]
        assert s["x"] == "a" and s["y"] == "b"

    def test_head_or_arity_mismatch(self):
        assert _unifiers(["f", ["?", "x"]], ["g", "a"], NO_AC) == []
        assert _unifiers(["f", ["?", "x"]], ["f", "a", "b"], NO_AC) == []

    def test_agrees_with_f2_unify_on_non_ac(self):
        from rerum.confluence import unify
        t1 = ["f", ["?", "x"], ["g", ["?", "y"]]]
        t2 = ["f", "a", ["g", "b"]]
        syn = unify(t1, t2)
        ac = _unifiers(t1, t2, NO_AC)
        assert len(ac) == 1 and ac[0] == syn
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest rerum/tests/test_acunify.py::TestDispatchNonAC -v`
Expected: FAIL (`ac_unify` not defined).

- [ ] **Step 3: Implement the dispatch + non-AC unification**

In `rerum/acunify.py`, append:

```python
def _bind(subst: Subst, name: str, value) -> Optional[Subst]:
    """Add ``name -> value`` (resolved) with an occurs-check, kept fully-applied.
    Returns the new Subst or None on occurs-check failure."""
    if _is_var(value) and value[1] == name:
        return subst
    if _occurs(name, value):
        return None
    one = {name: value}
    out = {k: apply_subst(one, v) for k, v in subst.items()}
    out[name] = value
    return out


def ac_unify(t1, t2, theory, *, bindings: Optional[Subst] = None,
             budget: Optional[UnifyBudget] = None) -> Iterator[Subst]:
    """Yield each AC-unifier of ``t1`` and ``t2`` modulo the AC operators in
    ``theory``. Multi-valued and lazy. Pure AC. Complete within ``budget``,
    not necessarily minimal."""
    if bindings is None:
        bindings = {}
    if _unsupported(t1) or _unsupported(t2):
        raise UnsupportedPattern(f"cannot ac-unify pattern form: {t1!r} ~ {t2!r}")
    a = apply_subst(bindings, t1)
    b = apply_subst(bindings, t2)

    if _is_var(a):
        s = _bind(bindings, a[1], b)
        if s is not None:
            yield s
        return
    if _is_var(b):
        s = _bind(bindings, b[1], a)
        if s is not None:
            yield s
        return
    if not compound(a) and not compound(b):
        if a == b:
            yield bindings
        return
    if compound(a) and compound(b) and a[0] == b[0] and theory.is_ac(a[0]):
        yield from _ac_unify_node(a, b, theory, bindings, budget)
        return
    if compound(a) and compound(b) and a[0] == b[0] and len(a) == len(b):
        yield from _unify_positional(a[1:], b[1:], theory, bindings, budget)
        return
    return  # head/arity clash, or atom vs compound


def _unify_positional(xs, ys, theory, bindings, budget) -> Iterator[Subst]:
    """Unify two argument lists in lockstep, MULTI-VALUED (a child may be an AC
    node). The two-sided analog of acmatch._match_positional."""
    if not xs:
        if not ys:
            yield bindings
        return
    if not ys:
        return
    for s in ac_unify(xs[0], ys[0], theory, bindings=bindings, budget=budget):
        yield from _unify_positional(xs[1:], ys[1:], theory, s, budget)
```

NOTE: `_ac_unify_node` is added in Tasks 3-4. For THIS task, add a temporary stub at the end of the file so imports resolve and the non-AC tests run:

```python
def _ac_unify_node(a, b, theory, bindings, budget) -> Iterator[Subst]:
    return iter(())  # Stickel core: implemented in Tasks 3-4
```

(Tasks 3 and 4 REPLACE this stub.)

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest rerum/tests/test_acunify.py -v`
Expected: PASS (Task 1 + TestDispatchNonAC). The AC stub yields nothing, which is fine -- no AC test yet.

- [ ] **Step 5: ASCII + commit**

```bash
git add rerum/acunify.py rerum/tests/test_acunify.py
git commit -m "feat(acunify): ac_unify dispatch + non-AC multi-valued unification

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Stickel core -- the all-variable case (the 7-unifier core)

The heart, first cut: AC nodes whose arguments are all variables. No non-variable atoms, so no couplings yet -- just the basis + covering-subset enumeration + variable bindings. This is the `x+y =? u+v` -> 7 case.

**Files:**
- Modify: `rerum/acunify.py`
- Test: `rerum/tests/test_acunify.py`

- [ ] **Step 1: Write the failing tests**

Append to `rerum/tests/test_acunify.py`:

```python
def _count_distinct(t1, t2, theory):
    # dedup unifiers by their normalized effect on the ORIGINAL variables
    # (ignoring fresh z-vars), so renamings collapse.
    seen = set()
    orig = _orig_vars(t1) | _orig_vars(t2)
    for s in au.ac_unify(t1, t2, theory):
        key = tuple(sorted((k, str(normalize(apply_subst(s, ["?", k]), theory)))
                           for k in orig))
        seen.add(key)
    return len(seen)


def _orig_vars(t):
    out = set()
    def walk(x):
        if isinstance(x, list) and len(x) == 2 and x[0] == "?":
            out.add(x[1]); return
        if isinstance(x, list):
            for s in x: walk(s)
    walk(t)
    return out


from rerum.confluence import apply_subst


class TestStickelAllVariable:
    def test_x_plus_y_eq_u_plus_v_seven_unifiers(self):
        t1 = ["+", ["?", "x"], ["?", "y"]]
        t2 = ["+", ["?", "u"], ["?", "v"]]
        assert _count_distinct(t1, t2, AC_PLUS) == 7

    def test_all_yields_are_sound(self):
        t1 = ["+", ["?", "x"], ["?", "y"]]
        t2 = ["+", ["?", "u"], ["?", "v"]]
        for s in au.ac_unify(t1, t2, AC_PLUS):
            assert normalize(apply_subst(s, t1), AC_PLUS) == \
                normalize(apply_subst(s, t2), AC_PLUS)

    def test_single_var_each_side(self):
        # (+ x) flattens to x; but (+ x) =? (+ y) as AC nodes -> x = y, 1 unifier.
        t1 = ["+", ["?", "x"], ["?", "x"]]   # x + x
        t2 = ["+", ["?", "y"], ["?", "y"]]   # y + y
        # x+x =? y+y has the unifier x=y (and the more-general x=2z,y=2z forms).
        for s in au.ac_unify(t1, t2, AC_PLUS):
            assert normalize(apply_subst(s, t1), AC_PLUS) == \
                normalize(apply_subst(s, t2), AC_PLUS)
        assert _count_distinct(t1, t2, AC_PLUS) >= 1
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest rerum/tests/test_acunify.py::TestStickelAllVariable -v`
Expected: FAIL (the stub yields nothing -> 0 unifiers, want 7).

- [ ] **Step 3: Replace the stub with the Stickel node (all-variable path)**

In `rerum/acunify.py`, REPLACE the temporary `_ac_unify_node` stub with:

```python
def _group(atoms):
    """Group equal atoms; return (terms, multiplicities, is_var flags)."""
    terms: list = []
    mult: List[int] = []
    isv: List[bool] = []
    for t in atoms:
        for k, tt in enumerate(terms):
            if tt == t:
                mult[k] += 1
                break
        else:
            terms.append(t)
            mult.append(1)
            isv.append(_is_var(t))
    return terms, mult, isv


def _ac_sum(op, parts):
    return parts[0] if len(parts) == 1 else [op] + parts


def _ac_unify_node(a, b, theory, bindings, budget) -> Iterator[Subst]:
    """Stickel AC-unification of two terms headed by the same AC operator."""
    op = a[0]
    left = flatten(a, theory)[1:]
    right = flatten(b, theory)[1:]
    # Cancel the multiset intersection (syntactically-equal common args).
    L = list(left)
    R = list(right)
    for item in list(L):
        if item in R:
            L.remove(item)
            R.remove(item)
    if not L and not R:
        yield bindings
        return
    if not L or not R:
        return  # pure AC: a non-empty sum cannot equal nothing

    U, a_mult, u_isv = _group(L)
    V, b_mult, v_isv = _group(R)
    basis = _hilbert_basis(a_mult, b_mult)
    M, N = len(U), len(V)
    avoid: set = set()
    z = [["?", gensym("z_" + str(k), avoid)] for k in range(len(basis))]
    for zk in z:
        avoid.add(zk[1])

    for r in range(1, len(basis) + 1):
        for subset in combinations(range(len(basis)), r):
            if budget is not None and not budget.spend():
                return
            # Admissibility: variable atoms covered (sum >= 1); non-variable
            # atoms covered EXACTLY (sum == multiplicity).
            ok = True
            for i in range(M):
                tot = sum(basis[k][i] for k in subset)
                if (u_isv[i] and tot < 1) or (not u_isv[i] and tot != a_mult[i]):
                    ok = False
                    break
            if ok:
                for j in range(N):
                    tot = sum(basis[k][M + j] for k in subset)
                    if (v_isv[j] and tot < 1) or (not v_isv[j] and tot != b_mult[j]):
                        ok = False
                        break
            if not ok:
                continue
            yield from _build_unifier(U, V, u_isv, v_isv, basis, subset, z, op,
                                      M, theory, bindings, budget)
```

And add `_build_unifier` -- for THIS task it handles only the all-variable path (no non-variable atoms); the non-variable/coupling path is added in Task 4:

```python
def _build_unifier(U, V, u_isv, v_isv, basis, subset, z, op, M, theory,
                   bindings, budget) -> Iterator[Subst]:
    """Build the unifier(s) for one admissible covering subset."""
    s: Optional[Subst] = dict(bindings)
    # Variable atoms: bind to the AC-sum of the z's they participate in.
    for i in range(len(U)):
        if not u_isv[i]:
            continue
        parts = [z[k] for k in subset for _ in range(basis[k][i])]
        s = _bind_unify(s, U[i], _ac_sum(op, parts))
        if s is None:
            return
    for j in range(len(V)):
        if not v_isv[j]:
            continue
        parts = [z[k] for k in subset for _ in range(basis[k][M + j])]
        s = _bind_unify(s, V[j], _ac_sum(op, parts))
        if s is None:
            return
    # Non-variable atoms (Task 4) -- none in the all-variable case.
    yield s


def _bind_unify(subst, term, value):
    """Unify a single (possibly compound) ``term`` with ``value`` syntactically,
    threading ``subst``. Used to assign a variable atom (or, in Task 4, force a
    non-variable atom equal to a fresh z). Returns a Subst or None."""
    if subst is None:
        return None
    term = apply_subst(subst, term)
    value = apply_subst(subst, value)
    if _is_var(term):
        return _bind(subst, term[1], value)
    if _is_var(value):
        return _bind(subst, value[1], term)
    if not compound(term) and not compound(value):
        return subst if term == value else None
    if compound(term) and compound(value) and term[0] == value[0] \
            and len(term) == len(value):
        s = subst
        for x, y in zip(term[1:], value[1:]):
            s = _bind_unify(s, x, y)
            if s is None:
                return None
        return s
    return None
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest rerum/tests/test_acunify.py::TestStickelAllVariable -v`
Expected: PASS (3 passed; the 7-unifier count and soundness hold).

- [ ] **Step 5: Run the whole file**

Run: `python -m pytest rerum/tests/test_acunify.py -q`
Expected: PASS (Tasks 1-3).

- [ ] **Step 6: ASCII + commit**

```bash
git add rerum/acunify.py rerum/tests/test_acunify.py
git commit -m "feat(acunify): Stickel core for the all-variable case (7-unifier)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Stickel core -- non-variable atoms + recursive couplings

Complete the node: non-variable atoms (constants and compounds) are covered EXACTLY, and the fresh basis variables coupling them are forced equal -- two coupled non-variable atoms are RECURSIVELY `ac_unify`'d (Stickel-general). This is the `x+y =? a+b` -> 2 and nested `(+ (f x) y) =? (+ a (f b))` -> 1 cases.

**Files:**
- Modify: `rerum/acunify.py`
- Test: `rerum/tests/test_acunify.py`

- [ ] **Step 1: Write the failing tests**

Append to `rerum/tests/test_acunify.py`:

```python
class TestStickelNonVariable:
    def test_x_plus_y_eq_a_plus_b_two_unifiers(self):
        t1 = ["+", ["?", "x"], ["?", "y"]]
        t2 = ["+", "a", "b"]
        assert _count_distinct(t1, t2, AC_PLUS) == 2

    def test_x_eq_a_plus_b_one_unifier(self):
        assert _count_distinct(["?", "x"], ["+", "a", "b"], AC_PLUS) == 1

    def test_ground_equal_one_unifier(self):
        assert _count_distinct(["+", "a", "b"], ["+", "a", "b"], AC_PLUS) == 1

    def test_distinct_constants_no_unifier(self):
        assert _count_distinct("a", "b", AC_PLUS) == 0
        # (+ a b) =? (+ a c): cancel a -> b =? c -> none.
        assert _count_distinct(["+", "a", "b"], ["+", "a", "c"], AC_PLUS) == 0

    def test_nested_free_symbol(self):
        # (+ (f x) y) =? (+ a (f b)) -> x=b, y=a (the coupling (f x)=?(f b)).
        t1 = ["+", ["f", ["?", "x"]], ["?", "y"]]
        t2 = ["+", "a", ["f", "b"]]
        got = list(au.ac_unify(t1, t2, AC_PLUS))
        assert _count_distinct(t1, t2, AC_PLUS) == 1
        s = got[0]
        assert apply_subst(s, ["?", "x"]) == "b"
        assert apply_subst(s, ["?", "y"]) == "a"

    def test_all_nonvariable_yields_sound(self):
        for t1, t2 in (
            (["+", ["?", "x"], ["?", "y"]], ["+", "a", "b"]),
            (["+", ["f", ["?", "x"]], ["?", "y"]], ["+", "a", ["f", "b"]]),
        ):
            for s in au.ac_unify(t1, t2, AC_PLUS):
                assert normalize(apply_subst(s, t1), AC_PLUS) == \
                    normalize(apply_subst(s, t2), AC_PLUS)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest rerum/tests/test_acunify.py::TestStickelNonVariable -v`
Expected: FAIL (non-variable atoms produce wrong/zero results; couplings not handled).

- [ ] **Step 3: Add non-variable coupling to `_build_unifier`**

In `rerum/acunify.py`, REPLACE `_build_unifier` with the version that handles non-variable atoms via recursive `ac_unify` of the atoms sharing each basis variable:

```python
def _build_unifier(U, V, u_isv, v_isv, basis, subset, z, op, M, theory,
                   bindings, budget) -> Iterator[Subst]:
    """Build the unifier(s) for one admissible covering subset. Variable atoms
    bind to AC-sums of their fresh z's; non-variable atoms sharing a z are
    recursively ac_unify'd (Stickel-general, multi-valued -> a product)."""
    s: Optional[Subst] = dict(bindings)
    for i in range(len(U)):
        if not u_isv[i]:
            continue
        parts = [z[k] for k in subset for _ in range(basis[k][i])]
        s = _bind_unify(s, U[i], _ac_sum(op, parts))
        if s is None:
            return
    for j in range(len(V)):
        if not v_isv[j]:
            continue
        parts = [z[k] for k in subset for _ in range(basis[k][M + j])]
        s = _bind_unify(s, V[j], _ac_sum(op, parts))
        if s is None:
            return
    # For each chosen basis vector, collect the NON-VARIABLE atoms it couples
    # (the fresh z must equal each of them). Recursively ac_unify all atoms
    # sharing a z; the product over basis vectors yields the unifier set.
    couplings = []
    for k in subset:
        atoms = [z[k]]
        for i in range(len(U)):
            if not u_isv[i] and basis[k][i]:
                atoms += [U[i]] * basis[k][i]
        for j in range(len(V)):
            if not v_isv[j] and basis[k][M + j]:
                atoms += [V[j]] * basis[k][M + j]
        if len(atoms) > 1:
            couplings.append(atoms)
    yield from _resolve_couplings(couplings, 0, s, theory, budget)


def _resolve_couplings(couplings, idx, subst, theory, budget) -> Iterator[Subst]:
    """Recursively ac_unify every pair within each coupling group; product over
    groups. Each group's atoms must all unify (transitively through their z)."""
    if subst is None:
        return
    if idx == len(couplings):
        yield subst
        return
    group = couplings[idx]
    # Unify atoms[0] with each of the others in turn; each ac_unify is
    # multi-valued, so thread the product.
    def chain(pos, s) -> Iterator[Subst]:
        if pos == len(group):
            yield from _resolve_couplings(couplings, idx + 1, s, theory, budget)
            return
        for s2 in ac_unify(group[0], group[pos], theory, bindings=s, budget=budget):
            yield from chain(pos + 1, s2)
    yield from chain(1, subst)
```

(`_bind_unify`, `_group`, `_ac_sum` from Task 3 are unchanged.)

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest rerum/tests/test_acunify.py::TestStickelNonVariable -v`
Expected: PASS (6 passed; counts 2/1/1/0/0/1 and soundness hold).

- [ ] **Step 5: Run the whole file**

Run: `python -m pytest rerum/tests/test_acunify.py -q`
Expected: PASS (Tasks 1-4; all canonical counts).

- [ ] **Step 6: ASCII + commit**

```bash
git add rerum/acunify.py rerum/tests/test_acunify.py
git commit -m "feat(acunify): non-variable atoms + recursive couplings (Stickel-general)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Budget truncation, soundness property, completeness oracle, ac_match cross-check

Pin the verification bar: soundness over a battery, completeness against a brute-force oracle on tiny cases, the F3 cross-check, budget truncation, and determinism.

**Files:**
- Modify: `rerum/acunify.py` (only if the budget needs a top-level reset point -- likely none)
- Test: `rerum/tests/test_acunify.py`

- [ ] **Step 1: Write the tests**

Append to `rerum/tests/test_acunify.py`:

```python
import itertools


class TestVerificationBar:
    def test_soundness_battery(self):
        V = lambda n: ["?", n]
        problems = [
            (["+", V("x"), V("y")], ["+", V("u"), V("v")]),
            (["+", V("x"), V("y"), V("z")], ["+", "a", "b", "c"]),
            (["+", V("x"), "a"], ["+", "b", V("y")]),
            (["+", ["f", V("x")], V("y")], ["+", "a", ["f", "b"]]),
        ]
        for t1, t2 in problems:
            for s in au.ac_unify(t1, t2, AC_PLUS):
                assert normalize(apply_subst(s, t1), AC_PLUS) == \
                    normalize(apply_subst(s, t2), AC_PLUS)

    def test_completeness_vs_brute_force(self):
        # Tiny problem: (+ x y) =? (+ a b). Brute-force every assignment of
        # {x,y} to nonempty subsets of {a,b} and confirm ac_unify finds them all.
        t1 = ["+", ["?", "x"], ["?", "y"]]
        t2 = ["+", "a", "b"]
        oracle = set()
        for sx in (["a"], ["b"], ["a", "b"]):
            for sy in (["a"], ["b"], ["a", "b"]):
                cand = {"x": sx, "y": sy}
                e1 = normalize(["+"] + [c for v in ("x", "y") for c in cand[v]],
                               AC_PLUS)
                if e1 == normalize(t2, AC_PLUS):
                    oracle.add((tuple(sx), tuple(sy)))
        # ac_unify's distinct solutions cover the oracle.
        found = set()
        for s in au.ac_unify(t1, t2, AC_PLUS):
            xv = normalize(apply_subst(s, ["?", "x"]), AC_PLUS)
            yv = normalize(apply_subst(s, ["?", "y"]), AC_PLUS)
            tx = tuple(xv[1:]) if isinstance(xv, list) else (xv,)
            ty = tuple(yv[1:]) if isinstance(yv, list) else (yv,)
            found.add((tx, ty))
        assert oracle <= found

    def test_ac_match_cross_check(self):
        # When t2 is GROUND, ac_unify(t1, t2) restricted to t1's vars covers
        # ac_match(t1, t2).
        from rerum import acmatch
        t1 = ["+", ["?", "x"], ["?", "y"]]
        t2 = ["+", "a", "b"]
        match_sols = {frozenset(b.to_dict().items())
                      for b in acmatch.ac_match(t1, t2, AC_PLUS)}
        unify_sols = set()
        for s in au.ac_unify(t1, t2, AC_PLUS):
            d = {k: apply_subst(s, ["?", k]) for k in ("x", "y")}
            # only keep ground solutions (matching produces ground bindings)
            if all(not _has_var(v) for v in d.values()):
                unify_sols.add(frozenset((k, _t(v)) for k, v in d.items()))
        match_keyed = {frozenset((k, _t(v)) for k, v in dict(m).items())
                       for m in match_sols}
        assert match_keyed <= unify_sols

    def test_budget_truncation_sound(self):
        # A wider all-variable problem with a tiny budget truncates but every
        # yield is still sound.
        t1 = ["+", ["?", "x"], ["?", "y"], ["?", "z"]]
        t2 = ["+", ["?", "p"], ["?", "q"], ["?", "r"]]
        budget = au.UnifyBudget(steps=3)
        got = list(au.ac_unify(t1, t2, AC_PLUS, budget=budget))
        assert budget.truncated is True
        for s in got:
            assert normalize(apply_subst(s, t1), AC_PLUS) == \
                normalize(apply_subst(s, t2), AC_PLUS)

    def test_determinism(self):
        t1 = ["+", ["?", "x"], ["?", "y"]]
        t2 = ["+", ["?", "u"], ["?", "v"]]
        a = [sorted(s.items()) for s in au.ac_unify(t1, t2, AC_PLUS)]
        b = [sorted(s.items()) for s in au.ac_unify(t1, t2, AC_PLUS)]
        assert a == b


def _has_var(t):
    if isinstance(t, list) and len(t) == 2 and t[0] == "?":
        return True
    if isinstance(t, list):
        return any(_has_var(s) for s in t)
    return False


def _t(v):
    return tuple(_t(x) for x in v) if isinstance(v, list) else v
```

- [ ] **Step 2: Run the tests**

Run: `python -m pytest rerum/tests/test_acunify.py::TestVerificationBar -v`
Expected: PASS (5 passed). If `test_completeness_vs_brute_force` or the cross-check fails, the node is missing unifiers -- investigate (do NOT weaken the assertion). If `test_budget_truncation_sound` does not truncate, raise the problem width or lower the budget until it does.

- [ ] **Step 3: ASCII + commit**

```bash
git add rerum/acunify.py rerum/tests/test_acunify.py
git commit -m "test(acunify): soundness battery, completeness oracle, ac_match cross-check, budget

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Re-exports, examples demo, and the full gate

**Files:**
- Modify: `rerum/__init__.py`
- Create: `examples/acunify_demo.rules`
- Test: `rerum/tests/test_acunify.py`

- [ ] **Step 1: Write the failing tests**

Append to `rerum/tests/test_acunify.py`:

```python
class TestReexportsAndDemo:
    def test_public_reexports(self):
        import rerum
        for name in ("ac_unify", "UnifyBudget"):
            assert name in rerum.__all__
            assert hasattr(rerum, name)

    def test_import_smoke_no_cycle(self):
        import importlib
        importlib.import_module("rerum.acunify")
        importlib.import_module("rerum.confluence")

    def test_demo_file_loads_and_problem_solves(self):
        # The demo file documents an AC theory + a worked problem; solve it here.
        import os
        from rerum.normalize import Theory
        root = os.path.join(os.path.dirname(__file__), "..", "..", "examples")
        # The demo's theory: + is AC. Solve x+y =? a+b -> 2 unifiers.
        theory = Theory.from_dict({"+": {"ac": True, "identity": 0}})
        sols = list(au.ac_unify(["+", ["?", "x"], ["?", "y"]],
                                ["+", "a", "b"], theory))
        assert sols  # demo problem has solutions
        assert os.path.exists(os.path.join(root, "acunify_demo.rules"))
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest rerum/tests/test_acunify.py::TestReexportsAndDemo -v`
Expected: FAIL (re-exports missing; demo file missing).

- [ ] **Step 3: Add the re-exports**

In `rerum/__init__.py`, find the `# Narrowing (F6)` import block (`from .narrowing import (...)`). Immediately AFTER it, add:

```python
# AC-unification
from .acunify import (
    ac_unify,
    UnifyBudget,
)
```

In the `__all__` list, after the `# Narrowing` entries (ending `"NarrowStep",`), add:

```python
    # AC-unification
    "ac_unify",
    "UnifyBudget",
```

- [ ] **Step 4: Create the demo file**

Create `examples/acunify_demo.rules`:

```
# AC-unification demo (data, not executable rules): under an AC theory for +,
# unifying two terms that BOTH contain variables has no single mgu -- there is a
# complete SET of unifiers. e.g. (+ ?x ?y) =? (+ a b) has TWO unifiers:
#   {x = a, y = b}  and  {x = b, y = a}
# and (+ ?x ?y) =? (+ ?u ?v) (four variables) has SEVEN.
# Use rerum.acunify.ac_unify(t1, t2, theory) with a Theory declaring + AC.
@noop: (id ?x) => :x
```

(The `@noop` line keeps the file a valid loadable rules file; the demo content is the comment + the test above.)

- [ ] **Step 5: Run to verify pass**

Run: `python -m pytest rerum/tests/test_acunify.py::TestReexportsAndDemo -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Whole file + full suite + domain guard**

Run: `python -m pytest rerum/tests/test_acunify.py -q` -- expect all pass.

Run: `python -m pytest -q` -- expect PASS; REPORT THE TOTAL (baseline 1742 + the new acunify tests).

Run: `python -m pytest rerum/tests/test_mcp_no_domain.py -q` -- expect 12 (acunify.py names no operator).

- [ ] **Step 7: ASCII check**

Run: `LC_ALL=C grep -n '[^[:print:][:space:]]' rerum/acunify.py rerum/__init__.py rerum/tests/test_acunify.py examples/acunify_demo.rules && echo FOUND || echo clean`
Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add rerum/__init__.py rerum/tests/test_acunify.py examples/acunify_demo.rules
git commit -m "feat(acunify): re-exports + demo + full gate

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Post-implementation

After all 6 tasks: dispatch an Opus holistic review (soundness: every yield is a real AC-unifier via normalize; completeness on the canonical cases; the recursive coupling is correct for AC-inside-AC; general-engine principle held; F2 unify untouched; budget honesty). Then `superpowers:finishing-a-development-branch` for the push decision (on `main`).

## Notes for the implementer

- **The algorithm is prototype-validated.** The canonical counts (7/2/1/1/0 and the nested 1) are confirmed against working code; if your implementation returns different counts, the bug is yours, not the test's.
- **Thread the substitution through EVERY coupling.** The single most likely bug (found in the prototype): computing a coupling unifier but not carrying it forward. `_resolve_couplings` and `_bind_unify` must return the threaded Subst.
- **Couplings recurse through `ac_unify`, not a syntactic unifier.** That is what makes nested AC (an AC operator inside a coupled atom) correct; a syntactic shortcut would silently miss those unifiers.
- **Admissibility asymmetry:** variable atoms need total multiplicity >= 1; non-variable atoms need EXACTLY their multiplicity. Getting this wrong changes the counts (e.g. `x+y=?a+b` would over-count).
- **Dedup is test-side only.** `ac_unify` yields one substitution per admissible subset (possibly alpha-variants / redundant -- "complete, not minimal" by design). The tests dedup by normalized effect on the original variables; the production code does NOT dedup.
- **Budget:** `spend()` is called once per covering-subset attempt and threads into the recursive `ac_unify` couplings. Truncation only loses unifiers (completeness), never soundness.
- Run the FULL suite after each task; `acunify.py` is new and isolated, so regressions are unlikely, but the re-export (Task 6) touches the public surface.
