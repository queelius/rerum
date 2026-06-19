# F2: Confluence + Critical-Pair Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a read-only confluence diagnostic: compute a rule set's critical pairs (overlaps between rule LHSs), check each for joinability, and report a local-confluence verdict naming any non-joinable overlap.

**Architecture:** A new PURE module `rerum/confluence.py` (first-order unification + critical-pair superposition + joinability), plus thin `engine.critical_pairs()` / `engine.check_confluence()` wrappers and `rerum` re-exports. Joinability reuses F1's `engine._canonicalize` to compare reduced forms modulo the loaded theory. Nothing about reduction changes.

**Tech Stack:** Python 3.9+, pytest. Pure-list `ExprType` expressions; structured pattern nodes (`["?", name]` etc.); `rerum.rewriter` predicates (`arbitrary_expression`/`arbitrary_constant`/`arbitrary_variable`/`arbitrary_free`/`arbitrary_rest`, `compound`, `free_symbols`, `gensym`, `skeleton_evaluation`).

**Spec:** `docs/superpowers/specs/2026-06-18-f2-confluence-diagnostics-design.md` (read it first; this plan implements it exactly, including the verification-driven fixes: recursive normal-form oracle via `_simplify_once` NOT root-only `apply_once`; `engine.rule_set()` as the directed accessor; trivial-overlap exclusion `i==j and p==()`; canonical-equality checked FIRST; `locally_confluent` = no False and no None).

**Constraints (CLAUDE.md + spec):**
- ASCII only in all files (a commit hook rejects non-ASCII).
- General-engine principle: no domain operator hardcoded in `rerum/`. `confluence.py` analyzes whatever rules the engine holds.
- Read-only: F2 changes no reduction behavior. The no-theory and theory paths of existing methods are untouched.
- Do NOT commit `.mcp.json`. Stage only files named in each task.

---

## Key code facts (verified against the codebase)

- A plain pattern variable is the node `["?", name]`. Typed/sequence forms: `["?c", name]`, `["?v", name]`, `["?free", name, v]`, `["?...", name]`. A RHS (skeleton) references a variable as `[":", name]` (NOT `["?", name]`); other skeleton forms are `[":...", name]` (splice), `["!", op, ...]` (compute), `["fresh", base]`.
- `rerum.rewriter` exports: `compound(x)` (is a non-empty list), `arbitrary_expression`/`arbitrary_constant`/`arbitrary_variable`/`arbitrary_free`/`arbitrary_rest` (head checks), `skeleton_evaluation(s)` (`car(s)==":"`), `free_symbols(expr)` (all string leaves incl. heads/markers), `gensym(base, avoid)` (deterministic fresh name), `ExprType`.
- `engine.rule_set()` returns a `RuleSet` already excluding disabled groups; iterating yields `(idx, rule, meta)` where `rule == [pattern, skeleton]` and `meta` has `.name`, `.condition`, `.bidirectional`, `.tags`. A `<=>` rule appears as two directed entries (`-fwd`, `-rev`).
- `engine.simplify(expr, max_steps=N)` reduces by rules (NOT theory-aware). `engine._simplify_once(expr)` applies at most one rule ANYWHERE (recursive) and returns the expr, so `engine._simplify_once(u) == u` is a sound "no redex anywhere" test. `engine.apply_once` is ROOT-ONLY and returns a tuple -- DO NOT use it for normal-form testing.
- `engine._canonicalize(expr)` (from F1) returns `expr` when no theory, else `normalize(expr, theory)`.

---

## File Structure

- **Create** `rerum/confluence.py`: positions/term-surgery, unification, renaming, critical-pair computation, confluence checking, the dataclasses. ONE responsibility: confluence analysis. Pure except `check_confluence`, which only READS the engine.
- **Modify** `rerum/engine.py`: add `RuleEngine.critical_pairs()` and `RuleEngine.check_confluence()` (thin wrappers).
- **Modify** `rerum/__init__.py`: re-export the public confluence surface.
- **Create** `rerum/tests/test_confluence.py`: the test suite.

---

### Task 1: Positions and term surgery

**Files:**
- Create: `rerum/confluence.py`
- Test: `rerum/tests/test_confluence.py` (create)

- [ ] **Step 1: Create the test file with position/surgery tests**

Create `rerum/tests/test_confluence.py`:

```python
"""F2: confluence and critical-pair diagnostics."""

from rerum import confluence as cf
from rerum.engine import RuleEngine
from rerum.normalize import Theory


class TestTermSurgery:
    def test_subterm_at_root_and_paths(self):
        t = ["+", ["*", "a", "b"], "c"]
        assert cf.subterm_at(t, ()) == t
        assert cf.subterm_at(t, (1,)) == ["*", "a", "b"]
        assert cf.subterm_at(t, (1, 2)) == "b"

    def test_replace_at(self):
        t = ["+", ["*", "a", "b"], "c"]
        assert cf.replace_at(t, (1, 2), "Z") == ["+", ["*", "a", "Z"], "c"]
        assert cf.replace_at(t, (), "Z") == "Z"
        # The original is not mutated.
        assert t == ["+", ["*", "a", "b"], "c"]

    def test_positions_are_non_variable_operand_paths(self):
        # (f (g ?x) a): non-variable positions are the root, the (g ?x) operand
        # and its operator-applied subterm, and the constant a -- NOT the ?x
        # variable node and NOT the operator-head index 0.
        t = ["f", ["g", ["?", "x"]], "a"]
        ps = set(cf.positions(t))
        assert () in ps          # whole term
        assert (1,) in ps        # (g ?x)
        assert (2,) in ps        # constant a
        assert (1, 1) not in ps  # ?x is a variable position -> excluded
        assert (0,) not in ps    # operator head -> not a position
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest rerum/tests/test_confluence.py::TestTermSurgery -v`
Expected: FAIL with `ModuleNotFoundError` / `AttributeError` (no `rerum.confluence`).

- [ ] **Step 3: Create `rerum/confluence.py` with the surgery layer**

Create `rerum/confluence.py`:

```python
"""F2: confluence and critical-pair diagnostics (read-only analysis).

Computes the CRITICAL PAIRS of a rule set (overlaps between rule left-hand
sides) and checks each for JOINABILITY, reporting a LOCAL confluence verdict.
This is analysis OVER the rewrite relation; it changes no rewrite behavior.

GENERAL ENGINE: hardcodes no operator. Patterns and rules are DATA. FIRST-ORDER
unification only; richer pattern forms (?c/?v/?free/?...) and non-trivial
skeleton forms (:.../!/fresh) and conditional rules are REFUSED -- the affected
rule is reported not-analyzed rather than silently mis-analyzed, so the verdict
is never a false "locally confluent".
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .rewriter import (
    ExprType,
    compound,
    free_symbols,
    gensym,
    skeleton_evaluation,
    arbitrary_expression,
    arbitrary_constant,
    arbitrary_variable,
    arbitrary_free,
    arbitrary_rest,
)

# A position is a tuple of child indices from the root; () is the root.
Position = Tuple[int, ...]


def _is_pattern_node(t: ExprType) -> bool:
    """True if t is any structured pattern node (?/?c/?v/?free/?...)."""
    return (
        arbitrary_expression(t)
        or arbitrary_constant(t)
        or arbitrary_variable(t)
        or arbitrary_free(t)
        or arbitrary_rest(t)
    )


def subterm_at(term: ExprType, p: Position) -> ExprType:
    """The subterm of ``term`` at position ``p`` (a path of child indices)."""
    cur = term
    for i in p:
        cur = cur[i]
    return cur


def replace_at(term: ExprType, p: Position, new: ExprType) -> ExprType:
    """A copy of ``term`` with the subterm at ``p`` replaced by ``new``."""
    if not p:
        return new
    i, rest = p[0], p[1:]
    return term[:i] + [replace_at(term[i], rest, new)] + term[i + 1:]


def positions(term: ExprType) -> List[Position]:
    """All NON-VARIABLE positions of ``term``.

    A non-variable position is one whose subterm is a compound application or a
    constant atom -- NOT a pattern node and NOT the operator head (index 0).
    Overlaps are computed only at these positions.
    """
    out: List[Position] = []

    def walk(t: ExprType, p: Position) -> None:
        if _is_pattern_node(t):
            return  # variable position: skip it (and its subtree)
        out.append(p)
        if compound(t):
            for i in range(1, len(t)):  # operands only; index 0 is the operator
                walk(t[i], p + (i,))

    walk(term, ())
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest rerum/tests/test_confluence.py::TestTermSurgery -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add rerum/confluence.py rerum/tests/test_confluence.py
git commit -m "feat(f2): positions and term surgery for confluence analysis

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: First-order unification, substitution, and skeleton instantiation

**Files:**
- Modify: `rerum/confluence.py`
- Test: `rerum/tests/test_confluence.py`

- [ ] **Step 1: Write the failing tests**

Append to `rerum/tests/test_confluence.py`:

```python
import pytest  # noqa: E402


class TestUnify:
    def test_two_distinct_variables_unify(self):
        s = cf.unify(["?", "x"], ["?", "y"])
        assert s is not None
        # x resolves to the y-variable.
        assert cf.apply_subst(s, ["?", "x"]) == ["?", "y"]

    def test_variable_binds_compound(self):
        s = cf.unify(["?", "x"], ["g", "a"])
        assert s is not None and cf.apply_subst(s, ["?", "x"]) == ["g", "a"]

    def test_occurs_check_fails(self):
        assert cf.unify(["?", "x"], ["f", ["?", "x"]]) is None

    def test_head_and_arity_clashes_fail(self):
        assert cf.unify(["f", "a"], ["g", "a"]) is None
        assert cf.unify(["f", "a"], ["f", "a", "b"]) is None

    def test_atoms(self):
        assert cf.unify("a", "a") == {}
        assert cf.unify("a", "b") is None
        assert cf.unify(["f", "a"], "a") is None  # compound vs atom

    def test_compound_unifies_pairwise(self):
        s = cf.unify(["f", ["?", "x"], "b"], ["f", "a", ["?", "y"]])
        assert s is not None
        assert cf.apply_subst(s, ["?", "x"]) == "a"
        assert cf.apply_subst(s, ["?", "y"]) == "b"


class TestUnifyRefusal:
    @pytest.mark.parametrize("bad", [
        ["?c", "x"], ["?v", "x"], ["?free", "x", "y"], ["?...", "r"],
    ])
    def test_unsupported_node_raises(self, bad):
        with pytest.raises(cf.UnsupportedPattern):
            cf.unify(["?", "x"], bad)

    def test_mixed_var_vs_typed_raises_not_binds(self):
        # Refuse-FIRST: the typed node must raise, not be bound as opaque.
        with pytest.raises(cf.UnsupportedPattern):
            cf.unify(["?", "x"], ["?c", "y"])

    def test_nested_unsupported_raises(self):
        with pytest.raises(cf.UnsupportedPattern):
            cf.unify(["f", ["?c", "x"]], ["f", "a"])


class TestApplySubstAndInstantiate:
    def test_apply_subst_recurses_and_leaves_free(self):
        s = {"x": "a"}
        assert cf.apply_subst(s, ["f", ["?", "x"], ["?", "y"]]) == ["f", "a", ["?", "y"]]

    def test_apply_subst_single_pass_on_resolved_subst(self):
        # A fully-applied subst: x -> (g y), y -> b. One pass resolves x fully.
        s = cf.unify(["f", ["?", "x"], ["?", "y"]], ["f", ["g", ["?", "y"]], "b"])
        assert s is not None
        assert cf.apply_subst(s, ["?", "x"]) == ["g", "b"]

    def test_instantiate_skeleton_substitutes_colon_vars(self):
        # skeleton (h :x) under {x -> (k z)} becomes (h (k z)); free :w stays ?w.
        sk = ["h", [":", "x"], [":", "w"]]
        s = {"x": ["k", ["?", "z"]]}
        assert cf.instantiate_skeleton(sk, s) == ["h", ["k", ["?", "z"]], ["?", "w"]]
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest rerum/tests/test_confluence.py::TestUnify rerum/tests/test_confluence.py::TestUnifyRefusal rerum/tests/test_confluence.py::TestApplySubstAndInstantiate -v`
Expected: FAIL (`unify`/`apply_subst`/`instantiate_skeleton`/`UnsupportedPattern` not defined).

- [ ] **Step 3: Add unification + substitution to `rerum/confluence.py`**

Append to `rerum/confluence.py` (after `positions`):

```python
class UnsupportedPattern(Exception):
    """Raised by ``unify`` on a pattern form outside the first-order fragment
    (?c/?v/?free/?... or a skeleton-only marker). The caller records the rule
    as not-analyzed, keeping the confluence verdict conservative."""


# A substitution {var_name: term}, maintained fully-applied (idempotent).
Subst = Dict[str, ExprType]

_UNSUPPORTED_HEADS = {"?c", "?v", "?free", "?...", "!", "fresh"}


def _unsupported(t: ExprType) -> bool:
    return compound(t) and isinstance(t[0], str) and t[0] in _UNSUPPORTED_HEADS


def _is_var(t: ExprType) -> bool:
    return arbitrary_expression(t)  # ["?", name]


def apply_subst(subst: Subst, term: ExprType) -> ExprType:
    """Replace each ``["?", name]`` whose ``name`` is bound. Single pass --
    ``subst`` is kept fully-applied -- and recurses into compound arguments."""
    if _is_var(term):
        name = term[1]
        return subst[name] if name in subst else term
    if compound(term):
        return [apply_subst(subst, sub) for sub in term]
    return term


def _occurs(name: str, term: ExprType) -> bool:
    if _is_var(term):
        return term[1] == name
    if compound(term):
        return any(_occurs(name, sub) for sub in term)
    return False


def _compose_bind(subst: Subst, name: str, value: ExprType) -> Subst:
    """Add ``name -> value`` (value already resolved), substituting it through
    every existing range term so the result stays fully-applied."""
    one = {name: value}
    updated = {k: apply_subst(one, v) for k, v in subst.items()}
    updated[name] = value
    return updated


def unify(t1: ExprType, t2: ExprType,
          subst: Optional[Subst] = None) -> Optional[Subst]:
    """First-order syntactic unification of two structured pattern terms.

    Returns the mgu (a fully-applied ``Subst``) or ``None`` on a normal failure
    (clash / occurs-check / arity / head mismatch). Raises ``UnsupportedPattern``
    on any ?c/?v/?free/?... or skeleton-only node, checked BEFORE the
    variable/compound branches so a typed node is never bound as opaque.
    """
    if subst is None:
        subst = {}
    if _unsupported(t1) or _unsupported(t2):
        raise UnsupportedPattern(f"cannot unify pattern form: {t1!r} ~ {t2!r}")
    a = apply_subst(subst, t1)
    b = apply_subst(subst, t2)
    if _is_var(a):
        return _unify_var(a, b, subst)
    if _is_var(b):
        return _unify_var(b, a, subst)
    if not compound(a) and not compound(b):
        return subst if a == b else None
    if compound(a) and compound(b):
        if a[0] != b[0] or len(a) != len(b):
            return None
        s: Optional[Subst] = subst
        for x, y in zip(a[1:], b[1:]):
            s = unify(x, y, s)
            if s is None:
                return None
        return s
    return None  # atom vs compound


def _unify_var(var: ExprType, other: ExprType, subst: Subst) -> Optional[Subst]:
    name = var[1]
    other = apply_subst(subst, other)
    if _is_var(other) and other[1] == name:
        return subst
    if _occurs(name, other):
        return None
    return _compose_bind(subst, name, other)


def instantiate_skeleton(skeleton: ExprType, sigma: Subst) -> ExprType:
    """Turn a rule RHS (skeleton) into a term under ``sigma``.

    A ``[":", name]`` reference becomes ``sigma``'s value for ``name`` (or the
    variable ``["?", name]`` itself if unbound -- it remains a free variable of
    the critical pair). Compounds recurse; literal atoms/operators are returned
    as-is. Non-trivial skeleton forms (:.../!/fresh) never reach here because
    such rules are refused upstream (see ``is_analyzable``).
    """
    if skeleton_evaluation(skeleton):  # [":", name]
        return apply_subst(sigma, ["?", skeleton[1]])
    if compound(skeleton):
        return [instantiate_skeleton(sub, sigma) for sub in skeleton]
    return skeleton
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest rerum/tests/test_confluence.py::TestUnify rerum/tests/test_confluence.py::TestUnifyRefusal rerum/tests/test_confluence.py::TestApplySubstAndInstantiate -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rerum/confluence.py rerum/tests/test_confluence.py
git commit -m "feat(f2): first-order unification + skeleton instantiation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Renaming-apart and the analyzability pre-scan

**Files:**
- Modify: `rerum/confluence.py`
- Test: `rerum/tests/test_confluence.py`

- [ ] **Step 1: Write the failing tests**

Append to `rerum/tests/test_confluence.py`:

```python
class TestRenameAndAnalyzable:
    def test_rename_apart_makes_fresh_variables(self):
        pat, sk = cf.rename_apart(["f", ["?", "x"]], ["g", [":", "x"]], {"x"})
        # The pattern variable is renamed away from the avoided "x".
        assert pat != ["f", ["?", "x"]]
        new_name = pat[1][1]
        assert new_name != "x"
        # The skeleton's [":", x] reference is renamed to the SAME new name.
        assert sk == ["g", [":", new_name]]

    def test_is_analyzable_accepts_first_order(self):
        assert cf.is_analyzable(["f", ["?", "x"]], ["g", [":", "x"]], None) is True

    def test_is_analyzable_refuses_conditional(self):
        assert cf.is_analyzable(["f", ["?", "x"]], [":", "x"], ["pos", [":", "x"]]) is False

    def test_is_analyzable_refuses_bad_pattern_forms(self):
        assert cf.is_analyzable(["f", ["?...", "r"]], [":", "r"], None) is False
        assert cf.is_analyzable(["f", ["?c", "x"]], [":", "x"], None) is False

    def test_is_analyzable_refuses_bad_skeleton_forms(self):
        assert cf.is_analyzable(["f", ["?", "x"]], ["!", "+", [":", "x"], 1], None) is False
        assert cf.is_analyzable(["f", ["?", "x"]], [":...", "x"], None) is False
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest rerum/tests/test_confluence.py::TestRenameAndAnalyzable -v`
Expected: FAIL (`rename_apart`/`is_analyzable` not defined).

- [ ] **Step 3: Add renaming + pre-scan to `rerum/confluence.py`**

Append to `rerum/confluence.py`:

```python
_PATTERN_BAD = {"?c", "?v", "?free", "?..."}
_SKELETON_BAD = {":...", "!", "fresh"}


def _variables(term: ExprType) -> set:
    """The set of ``["?", name]`` variable names occurring in ``term``."""
    out: set = set()

    def walk(t: ExprType) -> None:
        if _is_var(t):
            out.add(t[1])
        elif compound(t):
            for sub in t:
                walk(sub)

    walk(term)
    return out


def _rename(term: ExprType, mapping: Dict[str, str]) -> ExprType:
    """Rename ``["?", name]`` and ``[":", name]`` nodes per ``mapping``."""
    if _is_var(term):  # ["?", name]
        return ["?", mapping.get(term[1], term[1])]
    if skeleton_evaluation(term):  # [":", name]
        return [":", mapping.get(term[1], term[1])]
    if compound(term):
        return [_rename(sub, mapping) for sub in term]
    return term


def rename_apart(pattern: ExprType, skeleton: ExprType,
                 avoid: set) -> Tuple[ExprType, ExprType]:
    """Return ``(pattern', skeleton')`` with every rule variable renamed to a
    fresh name not in ``avoid`` (and distinct from each other). Renames both
    the pattern's ``["?", name]`` binders and the skeleton's ``[":", name]``
    references with the SAME mapping."""
    names = _variables(pattern) | _variables(skeleton)
    mapping: Dict[str, str] = {}
    used = set(avoid)
    for n in sorted(names):
        fresh = gensym(n, used)
        mapping[n] = fresh
        used.add(fresh)
    return _rename(pattern, mapping), _rename(skeleton, mapping)


def _has_marker(term: ExprType, heads: set) -> bool:
    if compound(term):
        if isinstance(term[0], str) and term[0] in heads:
            return True
        return any(_has_marker(sub, heads) for sub in term)
    return False


def is_analyzable(pattern: ExprType, skeleton: ExprType,
                  condition: Optional[ExprType]) -> bool:
    """True iff F2 can soundly analyze this directed rule: unconditional, a
    first-order pattern (no ?c/?v/?free/?...), and a plain-substitution
    skeleton (no :.../!/fresh)."""
    if condition is not None:
        return False
    if _has_marker(pattern, _PATTERN_BAD):
        return False
    if _has_marker(skeleton, _SKELETON_BAD):
        return False
    return True
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest rerum/tests/test_confluence.py::TestRenameAndAnalyzable -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rerum/confluence.py rerum/tests/test_confluence.py
git commit -m "feat(f2): rename-apart + analyzability pre-scan

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Critical-pair computation

**Files:**
- Modify: `rerum/confluence.py`
- Test: `rerum/tests/test_confluence.py`

- [ ] **Step 1: Write the failing tests**

Append to `rerum/tests/test_confluence.py`:

```python
def _rule(name, pattern, skeleton, condition=None):
    return cf.DirectedRule(name=name, pattern=pattern, skeleton=skeleton,
                           condition=condition)


class TestCriticalPairs:
    def test_overlap_between_two_rules(self):
        # r2 (g ?x)=>(k ?x) overlaps r1 (f (g ?x))=>(h ?x) at position (1,).
        r1 = _rule("r1", ["f", ["g", ["?", "x"]]], ["h", [":", "x"]])
        r2 = _rule("r2", ["g", ["?", "x"]], ["k", [":", "x"]])
        pairs, not_analyzed = cf.critical_pairs([r1, r2])
        assert not_analyzed == []
        # One overlap (r2 into r1 at (1,)) plus possibly others; find it.
        found = [cp for cp in pairs
                 if cp.rule_left == "r1" and cp.rule_right == "r2"
                 and cp.position == (1,)]
        assert len(found) == 1

    def test_distinct_rule_variables_are_kept_apart(self):
        # Both rules use ?x; the overlap must not conflate them.
        r1 = _rule("r1", ["f", ["g", ["?", "x"]]], ["h", [":", "x"]])
        r2 = _rule("r2", ["g", ["?", "x"]], ["k", [":", "x"]])
        pairs, _ = cf.critical_pairs([r1, r2])
        cp = [c for c in pairs if c.rule_left == "r1" and c.rule_right == "r2"
              and c.position == (1,)][0]
        # left is (h ?something); right wraps a (k ?something); a single shared
        # variable name remains -- not two conflated copies. Just assert the
        # construction produced well-formed terms with one free variable each.
        assert cp.left[0] == "h"
        assert cp.right[0] == "f"

    def test_self_overlap_excludes_trivial_root(self):
        # (f (f ?x)) => (f ?x): the only critical pair is the NON-root self
        # overlap at (1,); the trivial root overlap (i==j, p==()) is excluded.
        r = _rule("r", ["f", ["f", ["?", "x"]]], ["f", [":", "x"]])
        pairs, _ = cf.critical_pairs([r])
        positions_seen = {cp.position for cp in pairs}
        assert () not in positions_seen
        assert (1,) in positions_seen

    def test_conditional_and_bad_rules_are_not_analyzed(self):
        good = _rule("good", ["f", ["?", "x"]], [":", "x"])
        cond = _rule("cond", ["g", ["?", "x"]], [":", "x"], condition=["p", [":", "x"]])
        rest = _rule("rest", ["h", ["?...", "r"]], [":", "r"])
        pairs, not_analyzed = cf.critical_pairs([good, cond, rest])
        assert "cond" in not_analyzed
        assert "rest" in not_analyzed
        # The good rule is still analyzed (it just may yield no overlaps here).
        assert "good" not in not_analyzed
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest rerum/tests/test_confluence.py::TestCriticalPairs -v`
Expected: FAIL (`DirectedRule`/`critical_pairs`/`CriticalPair` not defined).

- [ ] **Step 3: Add the dataclasses + `critical_pairs` to `rerum/confluence.py`**

Append to `rerum/confluence.py`:

```python
@dataclass(frozen=True)
class DirectedRule:
    """A single directed reduction rule (post-desugar)."""
    name: Optional[str]
    pattern: ExprType
    skeleton: ExprType
    condition: Optional[ExprType] = None


@dataclass(frozen=True)
class CriticalPair:
    """An overlap between two rule LHSs. ``left``/``right`` are the two reducts
    of the overlapped term; ``joinable`` is filled by ``check_confluence``
    (None until then, or when undecidable)."""
    left: ExprType
    right: ExprType
    rule_left: Optional[str]
    rule_right: Optional[str]
    position: Position
    joinable: Optional[bool] = None


def critical_pairs(
    rules: List[DirectedRule],
) -> Tuple[List[CriticalPair], List[str]]:
    """Compute the critical pairs of ``rules`` (the standard superposition).

    Returns ``(pairs, not_analyzed)`` where ``not_analyzed`` lists the names of
    rules skipped (conditional or non-first-order), deduplicated in order.
    """
    pairs: List[CriticalPair] = []
    not_analyzed: List[str] = []
    seen_skips: set = set()

    def skip(rule: DirectedRule) -> None:
        key = rule.name
        if key not in seen_skips:
            seen_skips.add(key)
            not_analyzed.append(key)

    analyzable = []
    for r in rules:
        if is_analyzable(r.pattern, r.skeleton, r.condition):
            analyzable.append(r)
        else:
            skip(r)

    for i, ri in enumerate(analyzable):
        avoid = free_symbols(ri.pattern) | free_symbols(ri.skeleton)
        for j, rj in enumerate(analyzable):
            rj_pat, rj_sk = rename_apart(rj.pattern, rj.skeleton, avoid)
            for p in positions(ri.pattern):
                if i == j and p == ():
                    continue  # trivial root self-overlap
                try:
                    sigma = unify(subterm_at(ri.pattern, p), rj_pat)
                except UnsupportedPattern:
                    skip(ri)
                    continue
                if sigma is None:
                    continue
                u = apply_subst(sigma, ri.pattern)
                left = instantiate_skeleton(ri.skeleton, sigma)
                rj_rhs = instantiate_skeleton(rj_sk, sigma)
                right = replace_at(u, p, rj_rhs)
                pairs.append(CriticalPair(
                    left=left, right=right,
                    rule_left=ri.name, rule_right=rj.name, position=p,
                ))

    return pairs, not_analyzed
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest rerum/tests/test_confluence.py::TestCriticalPairs -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rerum/confluence.py rerum/tests/test_confluence.py
git commit -m "feat(f2): critical-pair computation (superposition)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Joinability and the confluence report

**Files:**
- Modify: `rerum/confluence.py`
- Test: `rerum/tests/test_confluence.py`

- [ ] **Step 1: Write the failing tests**

Append to `rerum/tests/test_confluence.py`:

```python
class TestCheckConfluence:
    def test_confluent_set_is_locally_confluent(self):
        # r2 makes a critical pair with r1; r3 joins it. Locally confluent.
        eng = RuleEngine.from_dsl("""
            @r1: (f (g ?x)) => (h ?x)
            @r2: (g ?x) => (k ?x)
            @r3: (f (k ?x)) => (h ?x)
        """)
        report = cf.check_confluence(eng)
        assert report.locally_confluent is True
        assert report.analyzed_pair_count >= 1
        assert report.non_joinable == []
        assert report.unknown == []

    def test_non_confluent_set_is_flagged(self):
        eng = RuleEngine.from_dsl("""
            @l: (f (f ?x)) => a
            @r: (f ?x) => b
        """)
        report = cf.check_confluence(eng)
        assert report.locally_confluent is False
        assert len(report.non_joinable) >= 1
        names = {cp.rule_left for cp in report.non_joinable} | \
                {cp.rule_right for cp in report.non_joinable}
        assert "l" in names and "r" in names

    def test_no_overlap_is_vacuously_confluent(self):
        eng = RuleEngine.from_dsl("""
            @p: (f ?x) => (g ?x)
            @q: (h ?x) => (k ?x)
        """)
        report = cf.check_confluence(eng)
        assert report.locally_confluent is True
        assert report.analyzed_pair_count == 0
        assert report.not_analyzed == []

    def test_not_analyzed_is_surfaced(self):
        # A rest-pattern rule is refused; with it the ONLY rule, there are no
        # analyzable pairs. The verdict is vacuously True, but not_analyzed +
        # analyzed_pair_count==0 make clear nothing was actually checked.
        eng = RuleEngine.from_dsl("@rest: (f ?x...) => (g :x...)")
        report = cf.check_confluence(eng)
        assert report.not_analyzed == ["rest"]
        assert report.analyzed_pair_count == 0
        assert report.locally_confluent is True

    def test_unknown_pair_blocks_verdict_and_returns(self):
        # r1 and r2 share LHS (a); one reduct (b) is a normal form, the other
        # (s c) grows forever via @grow, so the critical pair never joins and is
        # UNKNOWN (not a false verdict). The call must RETURN, not hang.
        eng = RuleEngine.from_dsl("""
            @r1: (a) => (b)
            @r2: (a) => (s c)
            @grow: (s ?x) => (s (s ?x))
        """)
        report = cf.check_confluence(eng, max_steps=20)
        assert len(report.unknown) >= 1
        assert report.locally_confluent is False

    def test_joinability_modulo_theory(self):
        # Overlap whose two sides are equal only after AC-normalization.
        eng = RuleEngine.from_dsl("""
            @l: (m ?x) => (+ ?x c)
            @r: (m ?x) => (+ c ?x)
        """)
        ac = Theory.from_dict({"+": {"ac": True}})
        # Without a theory: (+ ?x c) and (+ c ?x) are distinct normal forms.
        rep_no = cf.check_confluence(eng)
        assert rep_no.locally_confluent is False
        # With the AC theory: they canonicalize equal -> joinable.
        eng.with_theory(ac)
        rep_ac = cf.check_confluence(eng)
        assert rep_ac.locally_confluent is True

    def test_general_boolean_and_arithmetic(self):
        # Same code analyzes a non-arithmetic rule set.
        boolean = RuleEngine.from_dsl("@dn: (not (not ?x)) => ?x")
        arith = RuleEngine.from_dsl("@z: (+ ?x 0) => ?x")
        assert cf.check_confluence(boolean).locally_confluent is True
        assert cf.check_confluence(arith).locally_confluent is True
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest rerum/tests/test_confluence.py::TestCheckConfluence -v`
Expected: FAIL (`check_confluence`/`ConfluenceReport` not defined).

- [ ] **Step 3: Add joinability + report to `rerum/confluence.py`**

Append to `rerum/confluence.py`:

```python
@dataclass(frozen=True)
class ConfluenceReport:
    """Result of a local-confluence check.

    ``locally_confluent`` is True iff NO critical pair is non-joinable and NONE
    is undecidable (an empty critical-pair set is vacuously True). This is LOCAL
    confluence only: global confluence additionally requires termination
    (Newman's Lemma; that is roadmap F4). A ``joinable is False`` means "not
    joinable by the engine's own reduction (this strategy)", the right notion
    for a confluence DEFECT report. ``unknown`` pairs (reduction hit the budget
    or a cycle) are never counted as joinable.
    """
    locally_confluent: bool
    critical_pairs: List[CriticalPair]
    non_joinable: List[CriticalPair]
    unknown: List[CriticalPair]
    not_analyzed: List[str]
    analyzed_pair_count: int


def _is_normal_form(engine, term: ExprType, max_steps: int) -> bool:
    """A term is a normal form iff one recursive single-rule pass changes
    nothing. Uses ``_simplify_once`` (recursive), NOT the root-only
    ``apply_once``."""
    return engine._simplify_once(term) == term


def _decide_joinable(engine, cp: CriticalPair, max_steps: int) -> Optional[bool]:
    s2 = engine.simplify(cp.left, max_steps=max_steps)
    t2 = engine.simplify(cp.right, max_steps=max_steps)
    if engine._canonicalize(s2) == engine._canonicalize(t2):
        return True  # common reduct (modulo theory) -- checked FIRST
    if _is_normal_form(engine, s2, max_steps) and _is_normal_form(engine, t2, max_steps):
        return False  # distinct normal forms under the engine's reduction
    return None  # undecided within the budget


def check_confluence(engine, *, max_steps: int = 1000) -> ConfluenceReport:
    """Compute the engine's critical pairs, decide joinability of each, and
    return a local-confluence report. Read-only: mutates nothing."""
    records = [
        DirectedRule(name=meta.name, pattern=rule[0], skeleton=rule[1],
                     condition=meta.condition)
        for _idx, rule, meta in engine.rule_set()
    ]
    raw_pairs, not_analyzed = critical_pairs(records)

    decided: List[CriticalPair] = []
    non_joinable: List[CriticalPair] = []
    unknown: List[CriticalPair] = []
    analyzed = 0
    for cp in raw_pairs:
        verdict = _decide_joinable(engine, cp, max_steps)
        cp2 = CriticalPair(left=cp.left, right=cp.right,
                           rule_left=cp.rule_left, rule_right=cp.rule_right,
                           position=cp.position, joinable=verdict)
        decided.append(cp2)
        if verdict is True:
            analyzed += 1
        elif verdict is False:
            analyzed += 1
            non_joinable.append(cp2)
        else:
            unknown.append(cp2)

    locally_confluent = not non_joinable and not unknown
    return ConfluenceReport(
        locally_confluent=locally_confluent,
        critical_pairs=decided,
        non_joinable=non_joinable,
        unknown=unknown,
        not_analyzed=not_analyzed,
        analyzed_pair_count=analyzed,
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest rerum/tests/test_confluence.py::TestCheckConfluence -v`
Expected: PASS. If `test_joinability_modulo_theory` fails, confirm `Theory.from_dict({"+": {"ac": True}})` canonicalizes `(+ ?x c)` and `(+ c ?x)` to the same form (the variable node and the constant sort to a canonical order); if the chosen example does not normalize equal, adjust the example operands so the two sides are AC-variants. Do NOT weaken the assertion.

- [ ] **Step 5: Commit**

```bash
git add rerum/confluence.py rerum/tests/test_confluence.py
git commit -m "feat(f2): joinability + confluence report (modulo-theory via F1)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Engine methods, re-exports, and the full-suite gate

**Files:**
- Modify: `rerum/engine.py` (add two methods)
- Modify: `rerum/__init__.py` (re-exports)
- Test: `rerum/tests/test_confluence.py`

- [ ] **Step 1: Write the failing tests**

Append to `rerum/tests/test_confluence.py`:

```python
class TestEngineWrappersAndExports:
    def test_engine_check_confluence_delegates(self):
        eng = RuleEngine.from_dsl("""
            @l: (f (f ?x)) => a
            @r: (f ?x) => b
        """)
        method = eng.check_confluence()
        direct = cf.check_confluence(eng)
        assert method.locally_confluent == direct.locally_confluent is False

    def test_engine_critical_pairs_delegates(self):
        eng = RuleEngine.from_dsl("@r: (f (f ?x)) => (f ?x)")
        pairs = eng.critical_pairs()
        assert isinstance(pairs, list)
        assert all(isinstance(cp, cf.CriticalPair) for cp in pairs)

    def test_disabled_group_contributes_no_overlap(self):
        # DSL groups are [section] headers; rules under [grp] carry that tag.
        eng = RuleEngine.from_dsl("""
            [grp]
            @l: (f (f ?x)) => a
            @r: (f ?x) => b
        """)
        eng.disable_group("grp")
        report = eng.check_confluence()
        # With the only rules disabled, no overlaps -> vacuously confluent.
        assert report.locally_confluent is True
        assert report.analyzed_pair_count == 0

    def test_public_reexports(self):
        import rerum
        for name in ("check_confluence", "critical_pairs", "CriticalPair",
                     "ConfluenceReport", "unify", "UnsupportedPattern"):
            assert name in rerum.__all__
            assert hasattr(rerum, name)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest rerum/tests/test_confluence.py::TestEngineWrappersAndExports -v`
Expected: FAIL (`eng.check_confluence`/`eng.critical_pairs` not defined; re-exports missing). (The DSL group syntax is verified: `[grp]` is a section header and rules under it carry the `grp` tag; `disable_group("grp")` is the engine method.)

- [ ] **Step 3: Add the engine methods**

In `rerum/engine.py`, immediately AFTER the `minimize` method (it ends with the `return OptimizationResult(...)` block), insert:

```python
    def critical_pairs(self) -> "List[CriticalPair]":
        """The critical pairs (LHS overlaps) of this engine's enabled rules.

        Read-only confluence analysis (roadmap F2). See ``rerum.confluence``.
        """
        from .confluence import critical_pairs as _critical_pairs, DirectedRule
        records = [
            DirectedRule(name=meta.name, pattern=rule[0], skeleton=rule[1],
                         condition=meta.condition)
            for _idx, rule, meta in self.rule_set()
        ]
        pairs, _not_analyzed = _critical_pairs(records)
        return pairs

    def check_confluence(self, *, max_steps: int = 1000) -> "ConfluenceReport":
        """Local-confluence diagnostic for this engine's enabled rules.

        Computes critical pairs and checks joinability (modulo the loaded
        theory). Read-only. See ``rerum.confluence.ConfluenceReport``.
        """
        from .confluence import check_confluence as _check_confluence
        return _check_confluence(self, max_steps=max_steps)
```

(The annotations are STRING forms (`"List[CriticalPair]"`, `"ConfluenceReport"`), so no top-level `from .confluence import ...` is added to `engine.py`. This avoids any import-order fragility: `confluence.py` imports only from `rewriter`, never from `engine`, so the only edge is `engine -> confluence` at call time via the lazy method imports. The public types are re-exported from `rerum/__init__.py` in Step 4.)

- [ ] **Step 4: Add the re-exports**

In `rerum/__init__.py`, add an import block (near the other submodule imports, e.g. after the `from .normalize import (...)` block):

```python
from .confluence import (
    unify,
    apply_subst,
    critical_pairs,
    check_confluence,
    CriticalPair,
    ConfluenceReport,
    UnsupportedPattern,
)
```

And add these names to the `__all__` list (in the confluence-appropriate section; add a `# Confluence analysis` comment line):

```python
    # Confluence analysis
    "unify",
    "apply_subst",
    "critical_pairs",
    "check_confluence",
    "CriticalPair",
    "ConfluenceReport",
    "UnsupportedPattern",
```

- [ ] **Step 5: Run to verify pass**

Run: `pytest rerum/tests/test_confluence.py::TestEngineWrappersAndExports -v`
Expected: PASS (4 passed).

- [ ] **Step 6: Run the WHOLE confluence file + the full suite**

Run: `pytest rerum/tests/test_confluence.py -q`
Expected: PASS (all confluence tests).

Run: `pytest -q`
Expected: PASS with NO new failures vs. baseline (F2 adds a module + methods + re-exports; it changes no existing behavior). Report the total count.

Run: `pytest rerum/tests/test_mcp_no_domain.py -q`
Expected: PASS (F2 added no domain operator literal to `rerum/mcp/`).

- [ ] **Step 7: ASCII check**

Run: `LC_ALL=C grep -n '[^[:print:][:space:]]' rerum/confluence.py rerum/tests/test_confluence.py && echo "FOUND non-ASCII" || echo "clean"`
Expected: `clean`.

- [ ] **Step 8: Commit**

```bash
git add rerum/engine.py rerum/__init__.py rerum/tests/test_confluence.py
git commit -m "feat(f2): engine.check_confluence/critical_pairs + re-exports

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Post-Implementation

F2 is complete: `engine.check_confluence()` reports local confluence and names non-joinable overlaps; first-order unification is sound; unanalyzable rules are surfaced, not assumed joinable; the verdict is honestly LOCAL (global needs F4). Follow-ups (tracked in the spec): F3 AC-matching (the `?...`/AC overlaps F2 refuses, and the matching that would let position-pinning rules fire), F4 termination ordering (upgrades "locally confluent" to "confluent"), F5 Knuth-Bendix completion (consumes F2's critical pairs).

Use superpowers:finishing-a-development-branch to complete the work. Note: this work is on `main` directly; present the push decision to the user (pushing requires explicit authorization).
