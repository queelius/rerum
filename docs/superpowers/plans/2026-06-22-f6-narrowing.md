# F6: Narrowing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add narrowing -- unification-driven backward rewriting -- so rerum can solve goals (find sigma such that sigma(start) reduces to a target) and equations (E-unification), the principled in-family counterpart to the demoted best-first `solve`.

**Architecture:** A new core module `rerum/narrowing.py` that reuses F2's confluence primitives wholesale (`unify`, `apply_subst`, `rename_apart`, `is_analyzable`, `instantiate_skeleton`, `_variables`, `_is_var`) plus `gensym`/`free_symbols` from the rewriter. A narrowing step is a rewrite step with `match` swapped for `unify`. `narrow` runs budget-bounded BFS over the narrowing tree; `solve_equation` wraps it via a gensym'd reflexivity rule. Operates on pattern terms (variables are `["?", name]`), over analyzable first-order rules only, syntactically (no AC).

**Tech Stack:** Python 3.9+, pytest. Reuses `rerum/confluence.py` (F2) and `rerum/rewriter.py`.

---

## File Structure

- **Create** `rerum/narrowing.py` -- the whole feature: pure helpers (`_positions`, `_term_at`, `_replace_at`, `_compose`, `_freeze`/`_key`), `NarrowStep`, `narrow_step`, `NarrowResult`, `_extract_rules`, `_narrow_with_rules`, `narrow`, `solve_equation`. Names NO domain operator.
- **Create** `rerum/tests/test_narrowing.py` -- unit + reachability + E-unification + soundness + budget tests.
- **Modify** `rerum/__init__.py` -- re-export `narrow`, `solve_equation`, `narrow_step`, `NarrowResult`, `NarrowStep`.
- **Create** `examples/narrowing_demo.rules` -- a data-only Peano demo, driven through the general engine in a test.

## Conventions all tasks follow

- **ASCII only** in every file write (a commit hook rejects non-ASCII em-dashes / curly quotes). Use `->`, plain hyphens.
- **Never stage `.mcp.json`** (untracked local file). Stage only the files each task names.
- Commit messages end with the trailer:
  ```
  Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
  ```
- Terms are nested lists in prefix form. A VARIABLE is `["?", name]` (rerum's `?x`). `s(z)` is `["s", "z"]`; zero is the symbol `"z"`. A rule LHS uses `["?", name]`; a rule RHS skeleton uses `[":", name]`.
- General-engine principle: `narrowing.py` must name NO domain operator literal as code. `eq`/`true` for `solve_equation` are `gensym`'d; domains arrive as engine rules (data).

---

### Task 1: Pure helpers -- positions, term navigation, substitution composition

The position/navigation/compose primitives the matcher and BFS build on.

**Files:**
- Create: `rerum/narrowing.py`
- Create: `rerum/tests/test_narrowing.py`

- [ ] **Step 1: Write the failing tests**

Create `rerum/tests/test_narrowing.py`:

```python
"""F6: narrowing (unification-driven backward rewriting)."""

from rerum import narrowing as nw


class TestPureHelpers:
    def test_positions_non_variable_only(self):
        # (add ?x (s z)): root [], the (s z) at [2], the z at [2,1].
        # The variable ?x at [1] is NOT a position.
        term = ["add", ["?", "x"], ["s", "z"]]
        assert sorted(nw._positions(term)) == sorted([[], [2], [2, 1]])

    def test_positions_atom(self):
        assert list(nw._positions("z")) == [[]]

    def test_positions_bare_variable(self):
        assert list(nw._positions(["?", "x"])) == []

    def test_term_at(self):
        term = ["add", ["?", "x"], ["s", "z"]]
        assert nw._term_at(term, []) == term
        assert nw._term_at(term, [2]) == ["s", "z"]
        assert nw._term_at(term, [2, 1]) == "z"

    def test_replace_at(self):
        term = ["add", ["?", "x"], ["s", "z"]]
        assert nw._replace_at(term, [2, 1], "q") == ["add", ["?", "x"], ["s", "q"]]
        assert nw._replace_at(term, [], "done") == "done"
        # original is unchanged (functional)
        assert term == ["add", ["?", "x"], ["s", "z"]]

    def test_compose_applies_second_through_first(self):
        # compose(s2, s1) = s2 . s1 : apply s1 first, then s2.
        s1 = {"x": ["s", ["?", "y"]]}
        s2 = {"y": "z"}
        out = nw._compose(s2, s1)
        assert out["x"] == ["s", "z"]   # s2 applied through s1's range
        assert out["y"] == "z"          # s2's own binding kept

    def test_compose_first_wins_on_overlap(self):
        s1 = {"x": "a"}
        s2 = {"x": "b"}
        assert nw._compose(s2, s1)["x"] == "a"   # s1 binding survives
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest rerum/tests/test_narrowing.py -v`
Expected: FAIL (`No module named 'rerum.narrowing'`).

- [ ] **Step 3: Create `rerum/narrowing.py` with the helpers**

Create `rerum/narrowing.py`:

```python
"""F6: narrowing -- unification-driven backward rewriting.

Narrowing is rewriting with the matcher swapped for a unifier: a rewrite step
asks "does rule LHS l MATCH subterm t|p?" (only l's variables bind); a narrowing
step asks "can l and t|p be UNIFIED?" (both sides' variables bind). Iterating
narrowing solves goals -- find sigma such that sigma(start) reduces to a target --
and equations (E-unification). Sound and complete for confluent terminating
systems; the demoted best-first `solve` is neither.

CORE module. Reuses F2 (rerum/confluence.py) wholesale: unify, apply_subst,
rename_apart, is_analyzable, instantiate_skeleton, _variables, _is_var. Operates
on PATTERN TERMS (variables are ["?", name]). Scope: analyzable first-order rules
only (unify refuses ?c/?v/?free/?...); SYNTACTIC (a loaded AC theory is not used --
true AC-narrowing needs AC-unification). Names NO domain operator: solve_equation's
eq/true symbols are gensym'd; domains arrive as engine rules (data).
"""

from collections import deque
from dataclasses import dataclass
from typing import Iterator, Optional

from .confluence import (
    unify,
    apply_subst,
    rename_apart,
    is_analyzable,
    instantiate_skeleton,
    _variables,
    _is_var,
    UnsupportedPattern,
)
from .rewriter import compound, gensym, free_symbols, ExprType


def _positions(term) -> Iterator[list]:
    """Yield the path (list of indices) to every NON-VARIABLE subterm of
    ``term``. Index 0 (the operator head) is not a position; a ``["?", _]``
    node is not a position."""
    if not _is_var(term):
        yield []
    if compound(term):
        for i in range(1, len(term)):
            for sub in _positions(term[i]):
                yield [i] + sub


def _term_at(term, path):
    """The subterm of ``term`` at integer-index ``path``."""
    for i in path:
        term = term[i]
    return term


def _replace_at(term, path, new):
    """A copy of ``term`` with the subterm at ``path`` replaced by ``new``.
    Functional: ``term`` is not mutated."""
    if not path:
        return new
    i = path[0]
    return term[:i] + [_replace_at(term[i], path[1:], new)] + term[i + 1:]


def _compose(s2: dict, s1: dict) -> dict:
    """Substitution composition ``s2 . s1``: apply ``s1`` first, then ``s2``.
    For x in dom(s1): s2(s1(x)); for x in dom(s2)\\dom(s1): s2(x)."""
    out = {name: apply_subst(s2, val) for name, val in s1.items()}
    for name, val in s2.items():
        out.setdefault(name, val)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest rerum/tests/test_narrowing.py -v`
Expected: PASS (7 passed).

- [ ] **Step 5: ASCII check and commit**

Run: `LC_ALL=C grep -n '[^[:print:][:space:]]' rerum/narrowing.py rerum/tests/test_narrowing.py && echo "FOUND non-ASCII" || echo "clean"`
Expected: `clean`.

```bash
git add rerum/narrowing.py rerum/tests/test_narrowing.py
git commit -m "feat(f6): narrowing pure helpers (positions, navigation, compose)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `narrow_step` -- one-step narrowing successors

The narrowing step: for each non-variable position and each rule, rename apart, unify the subterm with the rule LHS, and apply the mgu to the term with the RHS spliced in.

**Files:**
- Modify: `rerum/narrowing.py`
- Test: `rerum/tests/test_narrowing.py`

- [ ] **Step 1: Write the failing tests**

Append to `rerum/tests/test_narrowing.py`:

```python
# Peano add as (l_pattern, r_term, rule_id) triples (r already a TERM: [":",n]->["?",n]).
ADD0 = (["add", "z", ["?", "y"]], ["?", "y"], "add0")
ADDS = (["add", ["s", ["?", "x"]], ["?", "y"]],
        ["s", ["add", ["?", "x"], ["?", "y"]]], "addS")
PEANO = [ADD0, ADDS]


class TestNarrowStep:
    def test_root_successors(self):
        # narrow (add ?a (s z)) at the root with both rules.
        term = ["add", ["?", "a"], ["s", "z"]]
        steps = list(nw.narrow_step(term, PEANO))
        succs = [s.successor for s in steps]
        # add0: ?a=z, ?y=(s z) -> successor (s z)
        assert ["s", "z"] in succs
        # addS: ?a=(s ?x'), ?y=(s z) -> successor (s (add ?x' (s z)))
        assert any(s[0] == "s" and s[1][0] == "add" for s in succs)

    def test_step_carries_sigma_and_position(self):
        term = ["add", ["?", "a"], ["s", "z"]]
        steps = list(nw.narrow_step(term, PEANO))
        add0_step = next(s for s in steps if s.successor == ["s", "z"])
        assert add0_step.sigma["a"] == "z"
        assert add0_step.position == []
        assert add0_step.rule_id == "add0"

    def test_variable_position_yields_nothing_extra(self):
        # The ?a at [1] is a variable position; no rule narrows there.
        term = ["add", ["?", "a"], ["s", "z"]]
        steps = list(nw.narrow_step(term, PEANO))
        assert all(s.position != [1] for s in steps)

    def test_no_match_when_no_rule_unifies(self):
        # (foo ?a) unifies no Peano LHS.
        assert list(nw.narrow_step(["foo", ["?", "a"]], PEANO)) == []

    def test_rename_apart_prevents_capture(self):
        # term reuses the rule's own variable name ?y; the rule must be renamed
        # apart so its ?y does not capture the term's ?y.
        term = ["add", ["?", "y"], "z"]   # term has ?y
        steps = list(nw.narrow_step(term, [ADD0]))
        # add0 (add z ?y) vs (add ?y z): ?y(term)=z and rule-?y'=z; successor z.
        assert any(s.successor == "z" for s in steps)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest rerum/tests/test_narrowing.py::TestNarrowStep -v`
Expected: FAIL (`narrow_step`/`NarrowStep` not defined).

- [ ] **Step 3: Implement `NarrowStep` and `narrow_step`**

In `rerum/narrowing.py`, after `_compose`, add:

```python
@dataclass
class NarrowStep:
    """One narrowing successor: ``successor`` is the new term, ``sigma`` the
    step mgu, ``position`` the redex path, ``rule_id`` the rule's name."""
    successor: ExprType
    sigma: dict
    position: list
    rule_id: str


def narrow_step(term, rules) -> Iterator[NarrowStep]:
    """Yield every one-step narrowing successor of ``term`` under ``rules``
    (a list of ``(l_pattern, r_term, rule_id)`` triples). For each non-variable
    position p and each rule, rename the rule apart from ``term``, unify
    ``term|p`` with the rule LHS, and apply the mgu to ``term`` with the RHS
    spliced at p."""
    avoid = _variables(term)
    for p in _positions(term):
        sub = _term_at(term, p)
        if _is_var(sub):
            continue
        for (l, r, rule_id) in rules:
            l_r, r_r = rename_apart(l, r, avoid)
            try:
                mgu = unify(sub, l_r)
            except UnsupportedPattern:
                continue
            if mgu is None:
                continue
            successor = apply_subst(mgu, _replace_at(term, p, r_r))
            yield NarrowStep(successor=successor, sigma=mgu,
                             position=p, rule_id=rule_id)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest rerum/tests/test_narrowing.py -v`
Expected: PASS (Task 1 + TestNarrowStep).

- [ ] **Step 5: ASCII check and commit**

Run: `LC_ALL=C grep -n '[^[:print:][:space:]]' rerum/narrowing.py rerum/tests/test_narrowing.py && echo "FOUND non-ASCII" || echo "clean"`
Expected: `clean`.

```bash
git add rerum/narrowing.py rerum/tests/test_narrowing.py
git commit -m "feat(f6): narrow_step one-step narrowing successors

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `narrow` -- the reachability driver (BFS + NarrowResult)

The budget-bounded breadth-first search over the narrowing tree, plus rule extraction from an engine.

**Files:**
- Modify: `rerum/narrowing.py`
- Test: `rerum/tests/test_narrowing.py`

- [ ] **Step 1: Write the failing tests**

Append to `rerum/tests/test_narrowing.py`:

```python
from rerum.engine import RuleEngine


def _peano_engine():
    return RuleEngine.from_dsl("""
        @add0: (add z ?y) => :y
        @addS: (add (s ?x) ?y) => (s (add :x :y))
    """)


class TestNarrowReachability:
    def test_solves_for_the_missing_addend(self):
        # find ?x such that add(?x, s(z)) reduces to s(s(z)) -> ?x = s(z).
        eng = _peano_engine()
        result = nw.narrow(eng, ["add", ["?", "x"], ["s", "z"]],
                           ["s", ["s", "z"]])
        assert result.found is True
        assert result.substitution["x"] == ["s", "z"]

    def test_immediate_goal_when_start_unifies_target(self):
        eng = _peano_engine()
        result = nw.narrow(eng, ["?", "x"], ["s", "z"])
        # ?x unifies the target directly: ?x = (s z), zero narrowing steps.
        assert result.found is True
        assert result.substitution["x"] == ["s", "z"]
        assert result.derivation == []

    def test_budget_exhaustion(self):
        # A non-terminating rule (loop ?x) => (loop (s ?x)) never reaches done.
        eng = RuleEngine.from_dsl("@loop: (loop ?x) => (loop (s ?x))")
        result = nw.narrow(eng, ["loop", "z"], "done", max_nodes=20)
        assert result.found is False
        assert result.exhausted is True

    def test_no_solution_finite_tree(self):
        # add(z, z) -> z, never s(z); finite tree, no solution.
        eng = _peano_engine()
        result = nw.narrow(eng, ["add", "z", "z"], ["s", "z"], max_nodes=1000)
        assert result.found is False
        assert result.exhausted is False

    def test_determinism(self):
        eng = _peano_engine()
        start, target = ["add", ["?", "x"], ["s", "z"]], ["s", ["s", "z"]]
        a = nw.narrow(eng, start, target).substitution
        b = nw.narrow(eng, start, target).substitution
        assert a == b
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest rerum/tests/test_narrowing.py::TestNarrowReachability -v`
Expected: FAIL (`narrow`/`NarrowResult` not defined).

- [ ] **Step 3: Implement `NarrowResult`, `_extract_rules`, `_freeze`/`_key`, `_narrow_with_rules`, `narrow`**

In `rerum/narrowing.py`, after `narrow_step`, add:

```python
@dataclass(frozen=True)
class NarrowResult:
    """Outcome of a narrowing search. ``substitution`` is the answer (a dict
    {name: term} restricted to the original variables) when ``found``;
    ``derivation`` is the list of NarrowStep witnesses; ``exhausted`` is True
    when the node budget was hit (vs a genuinely finite exhausted tree)."""
    found: bool
    substitution: Optional[dict]
    derivation: list
    nodes_expanded: int
    exhausted: bool


def _freeze(t):
    return tuple(_freeze(x) for x in t) if isinstance(t, list) else t


def _key(term, theta):
    return (_freeze(term),
            frozenset((k, _freeze(v)) for k, v in theta.items()))


def _extract_rules(engine):
    """Analyzable first-order rules from ``engine`` as (l, r_term, rule_id)
    triples; the RHS skeleton is converted to a term via instantiate_skeleton.
    Non-analyzable rules (?c/?v/?free/?.../skeleton-compute) are skipped."""
    rules = []
    for _idx, rule, meta in engine.rule_set():
        l, skel = rule[0], rule[1]
        if not is_analyzable(l, skel, meta.condition):
            continue
        rules.append((l, instantiate_skeleton(skel, {}), meta.name))
    return rules


def _narrow_with_rules(rules, start, target, *,
                       max_nodes=1000, max_depth=20) -> NarrowResult:
    """Budget-bounded BFS: find sigma such that sigma(start) narrows to a term
    unifying sigma(target). Returns the FIRST solution."""
    keep = _variables(start) | _variables(target)
    frontier = deque([(start, {}, 0, [])])
    seen = {_key(start, {})}
    nodes = 0
    while frontier:
        if nodes >= max_nodes:
            return NarrowResult(False, None, [], nodes, True)
        term, theta, depth, deriv = frontier.popleft()
        try:
            tau = unify(term, apply_subst(theta, target))
        except UnsupportedPattern:
            tau = None
        if tau is not None:
            sigma = {k: v for k, v in _compose(tau, theta).items() if k in keep}
            return NarrowResult(True, sigma, deriv, nodes, False)
        nodes += 1
        if depth < max_depth:
            for step in narrow_step(term, rules):
                theta2 = _compose(step.sigma, theta)
                k = _key(step.successor, theta2)
                if k not in seen:
                    seen.add(k)
                    frontier.append((step.successor, theta2, depth + 1,
                                     deriv + [step]))
    return NarrowResult(False, None, [], nodes, False)


def narrow(engine, start, target, *, max_nodes=1000, max_depth=20) -> NarrowResult:
    """Reachability narrowing over ``engine``'s analyzable rules: find sigma
    such that sigma(start) reduces to a term unifying sigma(target). Read-only;
    SYNTACTIC (ignores any loaded theory). See module docstring."""
    return _narrow_with_rules(_extract_rules(engine), start, target,
                              max_nodes=max_nodes, max_depth=max_depth)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest rerum/tests/test_narrowing.py::TestNarrowReachability -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Run the whole file**

Run: `pytest rerum/tests/test_narrowing.py -v`
Expected: PASS (Tasks 1-3).

- [ ] **Step 6: ASCII check and commit**

Run: `LC_ALL=C grep -n '[^[:print:][:space:]]' rerum/narrowing.py rerum/tests/test_narrowing.py && echo "FOUND non-ASCII" || echo "clean"`
Expected: `clean`.

```bash
git add rerum/narrowing.py rerum/tests/test_narrowing.py
git commit -m "feat(f6): narrow reachability driver (budget-bounded BFS)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: `solve_equation` -- E-unification wrapper

Solve `s =? t` modulo the rules via a gensym'd reflexivity rule, returning answer substitutions. The headline capability.

**Files:**
- Modify: `rerum/narrowing.py`
- Test: `rerum/tests/test_narrowing.py`

- [ ] **Step 1: Write the failing tests**

Append to `rerum/tests/test_narrowing.py`:

```python
def _append_engine():
    return RuleEngine.from_dsl("""
        @app0: (app nil ?ys) => :ys
        @appC: (app (cons ?x ?xs) ?ys) => (cons :x (app :xs :ys))
    """)


def _lst(*items):
    out = "nil"
    for it in reversed(items):
        out = ["cons", it, out]
    return out


class TestSolveEquation:
    def test_append_solves_the_prefix(self):
        # solve app(?xs, [c]) =? [a, b, c]  ->  ?xs = [a, b].
        eng = _append_engine()
        result = nw.solve_equation(eng,
                                   ["app", ["?", "xs"], _lst("c")],
                                   _lst("a", "b", "c"))
        assert result.found is True
        assert result.substitution["xs"] == _lst("a", "b")

    def test_already_equal_solves_trivially(self):
        eng = _peano_engine()
        result = nw.solve_equation(eng, ["s", "z"], ["s", "z"])
        assert result.found is True
        assert result.substitution == {}

    def test_answer_substitution_is_sound(self):
        # The returned sigma must re-derive: sigma(s) and sigma(t) join under
        # the engine's simplify.
        eng = _append_engine()
        s = ["app", ["?", "xs"], _lst("c")]
        t = _lst("a", "b", "c")
        result = nw.solve_equation(eng, s, t)
        sigma = result.substitution
        s_sub = nw.apply_subst(sigma, s)
        assert eng.simplify(s_sub) == eng.simplify(t)

    def test_no_solution_returns_not_found(self):
        # app(?xs, [c]) can never equal [a] (length mismatch); finite search.
        eng = _append_engine()
        result = nw.solve_equation(eng, ["app", ["?", "xs"], _lst("c")],
                                   _lst("a"), max_nodes=500)
        assert result.found is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest rerum/tests/test_narrowing.py::TestSolveEquation -v`
Expected: FAIL (`solve_equation` not defined; note `nw.apply_subst` is re-exported from confluence via the module import, so it resolves).

- [ ] **Step 3: Implement `solve_equation`**

In `rerum/narrowing.py`, after `narrow`, add:

```python
def solve_equation(engine, s, t, *, max_nodes=1000, max_depth=20) -> NarrowResult:
    """E-unification: solve ``s =? t`` modulo ``engine``'s rules. Returns the
    FIRST answer substitution sigma with sigma(s) and sigma(t) joinable.

    Reduces to reachability: with fresh (gensym'd) ``eq``/``true`` symbols and a
    reflexivity rule ``(eq ?x ?x) -> true``, narrow ``(eq s t)`` toward ``true``.
    Unrestricted narrowing explores positions inside s and t, and reflexivity
    fires at the root exactly when the two narrowed sides unify. Fresh symbols
    keep this domain-free."""
    rules = _extract_rules(engine)
    avoid = set(free_symbols(s)) | set(free_symbols(t))
    for (l, r, _rid) in rules:
        avoid |= set(free_symbols(l)) | set(free_symbols(r))
    eq = gensym("eq", avoid)
    true_ = gensym("true", avoid | {eq})
    refl = ([eq, ["?", "x"], ["?", "x"]], true_, "refl")
    result = _narrow_with_rules(rules + [refl], [eq, s, t], true_,
                                max_nodes=max_nodes, max_depth=max_depth)
    if not result.found:
        return result
    keep = _variables(s) | _variables(t)
    sub = {k: v for k, v in result.substitution.items() if k in keep}
    return NarrowResult(True, sub, result.derivation,
                        result.nodes_expanded, False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest rerum/tests/test_narrowing.py::TestSolveEquation -v`
Expected: PASS (4 passed).

- [ ] **Step 5: ASCII check and commit**

Run: `LC_ALL=C grep -n '[^[:print:][:space:]]' rerum/narrowing.py rerum/tests/test_narrowing.py && echo "FOUND non-ASCII" || echo "clean"`
Expected: `clean`.

```bash
git add rerum/narrowing.py rerum/tests/test_narrowing.py
git commit -m "feat(f6): solve_equation E-unification wrapper (gensym reflexivity)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Re-exports, examples demo, and the full gate

Make the public surface importable from `rerum`, ship a data-only demo, and run the full suite + guards.

**Files:**
- Modify: `rerum/__init__.py`
- Create: `examples/narrowing_demo.rules`
- Test: `rerum/tests/test_narrowing.py`

- [ ] **Step 1: Write the failing tests**

Append to `rerum/tests/test_narrowing.py`:

```python
class TestReexportsAndDemo:
    def test_public_reexports(self):
        import rerum
        for name in ("narrow", "solve_equation", "narrow_step",
                     "NarrowResult", "NarrowStep"):
            assert name in rerum.__all__
            assert hasattr(rerum, name)

    def test_demo_solves_via_general_engine(self):
        import os
        root = os.path.join(os.path.dirname(__file__), "..", "..", "examples")
        eng = RuleEngine.from_file(os.path.join(root, "narrowing_demo.rules"))
        # find ?x with add(?x, s(z)) = s(s(z)) -> ?x = s(z).
        result = nw.narrow(eng, ["add", ["?", "x"], ["s", "z"]],
                           ["s", ["s", "z"]])
        assert result.found and result.substitution["x"] == ["s", "z"]

    def test_import_smoke_no_cycle(self):
        import importlib
        importlib.import_module("rerum.narrowing")
        importlib.import_module("rerum.confluence")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest rerum/tests/test_narrowing.py::TestReexportsAndDemo -v`
Expected: FAIL (re-exports missing; demo file missing).

- [ ] **Step 3: Add the re-exports**

In `rerum/__init__.py`, find the `# AC-matching (F3)` import block (added by F3, `from .acmatch import (...)`). Immediately AFTER that block, add:

```python
# Narrowing (F6)
from .narrowing import (
    narrow,
    solve_equation,
    narrow_step,
    NarrowResult,
    NarrowStep,
)
```

In the `__all__` list, after the `# AC-matching` entries (`"ac_match"`, `"MatchBudget"`), add:

```python
    # Narrowing
    "narrow",
    "solve_equation",
    "narrow_step",
    "NarrowResult",
    "NarrowStep",
```

- [ ] **Step 4: Create the data-only demo**

Create `examples/narrowing_demo.rules`:

```
# Narrowing demo: Peano addition as DATA. With narrow(), the engine runs these
# rules BACKWARDS to solve for a missing addend, e.g. find ?x with
# add(?x, s(z)) = s(s(z)) -> ?x = s(z).
@add0: (add z ?y) => :y
@addS: (add (s ?x) ?y) => (s (add :x :y))
```

- [ ] **Step 5: Run to verify pass**

Run: `pytest rerum/tests/test_narrowing.py::TestReexportsAndDemo -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Whole file + full suite + domain guard**

Run: `pytest rerum/tests/test_narrowing.py -q`
Expected: PASS (all narrowing tests).

Run: `pytest -q`
Expected: PASS. Report the total. Baseline before F6 was 1684; this adds the new `test_narrowing.py` tests (no deletions).

Run: `pytest rerum/tests/test_mcp_no_domain.py -q`
Expected: PASS (12). `narrowing.py` is core, not under `rerum/mcp/`, but must still name no operator literal -- `eq`/`true` are gensym'd, domains are data.

- [ ] **Step 7: ASCII check**

Run: `LC_ALL=C grep -n '[^[:print:][:space:]]' rerum/narrowing.py rerum/__init__.py rerum/tests/test_narrowing.py examples/narrowing_demo.rules && echo "FOUND non-ASCII" || echo "clean"`
Expected: `clean`.

- [ ] **Step 8: Commit**

```bash
git add rerum/__init__.py rerum/tests/test_narrowing.py examples/narrowing_demo.rules
git commit -m "feat(f6): re-exports + data-only narrowing demo + full gate

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Post-implementation

After all five tasks: dispatch an Opus final holistic review (soundness: every answer sigma re-derives via simplify; rename-apart prevents capture; budget honesty; general-engine principle held; reuse of F2 is faithful), then `superpowers:finishing-a-development-branch` to present the push decision (on `main`, per the per-feature rhythm).

## Notes for the implementer

- **Variables are `["?", name]` everywhere.** `start`/`target` and rule LHS/RHS-as-term all use this form. `unify`, `apply_subst`, `_variables`, `rename_apart` all key on it.
- **RHS conversion:** a rule's stored RHS is a SKELETON (`[":", name]`); `instantiate_skeleton(skel, {})` turns it into a term (`[":",n] -> ["?",n]`). `_extract_rules` does this once. `solve_equation`'s reflexivity RHS (`true_`) is already a term.
- **rename_apart per step is load-bearing.** Without fresh rule variables each step, a rule reused down a derivation captures and produces unsound bindings. `narrow_step` renames apart from `_variables(term)` before every `unify`.
- **The `seen` key is `(frozen term, frozen theta)`**, NOT term alone -- two paths reaching the same term under different substitutions are distinct narrowing states; collapsing them by term would lose solutions.
- **Soundness check is the keystone test** (`test_answer_substitution_is_sound`): the returned sigma must make sigma(s) and sigma(t) join under `simplify`. If it fails, the bug is real (capture, bad compose, or wrong restriction) -- do not weaken it.
- **`compose(s2, s1)` order:** `s2 . s1` means apply s1 first. In the BFS, `theta2 = _compose(step.sigma, theta)` (the step's mgu applied on top of the accumulated theta); the answer is `_compose(tau, theta)` (the goal-unifier on top of theta). Getting the order backwards yields wrong/empty answers -- the soundness test guards it.
- **Syntactic only:** do not call `engine._canonicalize` or use the theory anywhere in narrowing. A loaded AC theory is intentionally ignored (documented non-goal).
