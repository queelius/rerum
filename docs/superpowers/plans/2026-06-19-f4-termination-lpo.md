# F4: Termination via LPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Certify that a rule set terminates (and orient an equation toward a terminating direction) via the lexicographic path order, and use it to upgrade F2's "locally confluent" to a genuine "confluent" verdict (Newman).

**Architecture:** A new PURE module `rerum/termination.py` (LPO + orient + check_termination), reusing F2's `DirectedRule`/`is_analyzable`/`instantiate_skeleton`/`_is_var`. Plus an additive Newman wiring into `confluence.check_confluence` (optional `precedence`), thin engine wrappers, and `rerum` re-exports. Read-only; reduction behavior unchanged.

**Tech Stack:** Python 3.9+, pytest. Structured terms (`["?",name]` variables, constant atoms as 0-ary symbols, compounds). Precedence = a list of function symbols in decreasing order.

**Spec:** `docs/superpowers/specs/2026-06-19-f4-termination-lpo-design.md` (read it first; this plan implements it exactly, including the verification fixes: constants treated as 0-ary symbols and rankable by precedence; the variadic `n==m` guard; `confluent=False` keyed on a genuine `non_joinable` witness, `unknown`-only failure -> `confluent=None`).

**Constraints:** ASCII only (commit hook). No domain operator hardcoded in `rerum/`. Do NOT commit `.mcp.json`.

---

## Key code facts (verified)

- `rerum.confluence` exports `DirectedRule(name, pattern, skeleton, condition=None)`, `is_analyzable(pattern, skeleton, condition)`, `instantiate_skeleton(skeleton, sigma)`, `_is_var(t)` (True for `["?",name]`). `instantiate_skeleton(skeleton, {})` converts a RHS skeleton to a term: `[":",x] -> ["?",x]`, bare `["?",x]` left literal, compounds recurse. So an analyzable rule's RHS-as-term shares the LHS's `["?",name]` variable representation.
- `rerum.rewriter` exports `compound(x)` (True for a non-empty list) and `ExprType`.
- `engine.rule_set()` yields `(idx, [pattern, skeleton], meta)`, post-desugar, disabled groups excluded; `meta.name`, `meta.condition` are attributes.
- `ConfluenceReport` is a frozen dataclass with fields `locally_confluent, critical_pairs, non_joinable, unknown, not_analyzed, analyzed_pair_count` (none have defaults). `confluence.check_confluence(engine, *, max_steps=1000)` builds it; `engine.check_confluence(self, *, max_steps=1000)` (engine.py ~4279) delegates. `engine.critical_pairs()` (~4265). Lazy-import-inside-method is the established cycle-avoidance idiom.
- `rerum/__init__.py` has a `# Confluence analysis (F2)` block (`from .confluence import (...)`) and an `__all__` `# Confluence analysis` section.

---

## File Structure

- **Create** `rerum/termination.py`: `_prec_gt`, `_head_args`, `_occurs`, `_lex_gt`, `lpo_greater`, `orient`, `TerminationReport`, `check_termination`. ONE responsibility: termination analysis. Pure except `check_termination`, which only READS the engine.
- **Modify** `rerum/confluence.py`: add `terminating`/`confluent` fields to `ConfluenceReport`; extend `check_confluence` with `precedence=None` + the Newman block (lazy-imports `check_termination`).
- **Modify** `rerum/engine.py`: add `RuleEngine.check_termination`; extend `RuleEngine.check_confluence` to pass `precedence` through.
- **Modify** `rerum/__init__.py`: re-export `lpo_greater`, `orient`, `check_termination`, `TerminationReport`.
- **Create** `rerum/tests/test_termination.py`.

---

### Task 1: The lexicographic path order

**Files:**
- Create: `rerum/termination.py`
- Test: `rerum/tests/test_termination.py` (create)

- [ ] **Step 1: Create the test file with the LPO tests**

Create `rerum/tests/test_termination.py`:

```python
"""F4: termination via the lexicographic path order."""

from rerum import termination as tm
from rerum.engine import RuleEngine


class TestPrecedenceAndLPO:
    def test_prec_gt(self):
        assert tm._prec_gt("*", "+", ["*", "+"]) is True
        assert tm._prec_gt("+", "*", ["*", "+"]) is False
        assert tm._prec_gt("z", "+", ["*", "+"]) is False   # unlisted
        assert tm._prec_gt("+", "z", ["*", "+"]) is False   # unlisted
        assert tm._prec_gt("+", "+", ["+"]) is False        # not > itself

    def test_variable_cases(self):
        assert tm.lpo_greater(["?", "x"], ["f", "a"], []) is False
        assert tm.lpo_greater(["f", ["?", "x"]], ["?", "x"], []) is True
        assert tm.lpo_greater(["f", ["?", "x"]], ["?", "y"], []) is False

    def test_constant_cases(self):
        assert tm.lpo_greater("a", "b", ["a", "b"]) is True
        assert tm.lpo_greater("b", "a", ["a", "b"]) is False
        assert tm.lpo_greater(["f", "a"], "a", ["f", "a"]) is True   # subterm/head
        assert tm.lpo_greater("a", ["f", "a"], ["f", "a"]) is False

    def test_case1_subterm(self):
        assert tm.lpo_greater(["f", ["g", "a"]], ["g", "a"], []) is True

    def test_case2_precedence(self):
        prec = ["*", "+", "a", "b"]
        assert tm.lpo_greater(["*", "a", "b"], ["+", "a", "b"], prec) is True
        rev = ["+", "*", "a", "b"]
        assert tm.lpo_greater(["*", "a", "b"], ["+", "a", "b"], rev) is False

    def test_case3_lexicographic(self):
        assert tm.lpo_greater(["f", "b", "a"], ["f", "a", "a"], ["b", "a"]) is True

    def test_variadic_same_head_different_arity(self):
        # Case 3 is SKIPPED (arity mismatch); only case 1 may fire. The bigger
        # term dominates via the subterm case; neither call raises.
        assert tm.lpo_greater(["+", "a", "b", ["+", "a", "b"]],
                              ["+", "a", "b"], []) is True   # t is a subterm
        assert tm.lpo_greater(["+", "a", "b"],
                              ["+", "a", "b", "c"], []) is False
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest rerum/tests/test_termination.py::TestPrecedenceAndLPO -v`
Expected: FAIL with `ModuleNotFoundError`/`AttributeError` (no `rerum.termination`).

- [ ] **Step 3: Create `rerum/termination.py`**

Create `rerum/termination.py`:

```python
"""F4: termination via the lexicographic path order (read-only analysis).

Certifies that a rule set TERMINATES and ORIENTS an equation toward a
terminating direction, using the lexicographic path order (LPO) derived from a
PRECEDENCE on function symbols supplied as DATA. If every rule l -> r satisfies
l >_lpo r, every rewrite step strictly decreases the term in the well-founded
order >_lpo, so the system cannot loop.

GENERAL ENGINE: the precedence is DATA (a list of function symbols -- operators
AND constants -- in decreasing order). No operator is hardcoded. First-order
only; non-first-order rules are refused by reusing confluence.is_analyzable.
A constant atom is treated as a 0-ARY function symbol (standard LPO).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .rewriter import ExprType, compound
from .confluence import (
    DirectedRule,
    is_analyzable,
    instantiate_skeleton,
    _is_var,
)

# A precedence: function symbols in DECREASING precedence (head = greatest).
Precedence = List


def _prec_gt(f, g, precedence: Precedence) -> bool:
    """True iff f > g: both listed and f earlier (greater) than g. Symbols not
    both in the list are incomparable (False both ways)."""
    if f not in precedence or g not in precedence:
        return False
    return precedence.index(f) < precedence.index(g)


def _head_args(t: ExprType) -> Tuple:
    """(head, args), treating a constant atom as a 0-ary symbol."""
    if compound(t):
        return t[0], t[1:]
    return t, []


def _occurs(t: ExprType, term: ExprType) -> bool:
    """True iff ``t`` occurs anywhere inside ``term`` (structural equality)."""
    if t == term:
        return True
    if compound(term):
        return any(_occurs(t, sub) for sub in term)
    return False


def _lex_gt(sargs: List, targs: List, precedence: Precedence) -> bool:
    """Lexicographic compare of equal-length tuples: the first DIFFERING
    position must have its s-arg >_lpo its t-arg (earlier positions equal)."""
    for si, ti in zip(sargs, targs):
        if si == ti:
            continue
        return lpo_greater(si, ti, precedence)
    return False  # identical


def lpo_greater(s: ExprType, t: ExprType, precedence: Precedence) -> bool:
    """The lexicographic path order: True iff ``s`` strictly dominates ``t``."""
    if s == t:
        return False
    if _is_var(s):
        return False  # a variable dominates nothing
    if _is_var(t):
        return _occurs(t, s)  # s != t, so t is a proper subterm of s
    f, sargs = _head_args(s)
    g, targs = _head_args(t)
    # Case 1 (subterm): some argument of s is >= t.
    if any(si == t or lpo_greater(si, t, precedence) for si in sargs):
        return True
    # Case 2 (precedence): f outranks g and s beats every argument of t.
    if _prec_gt(f, g, precedence) and all(
            lpo_greater(s, tj, precedence) for tj in targs):
        return True
    # Case 3 (lexicographic): same head AND arity, s beats every tj, args >lex.
    if (f == g and len(sargs) == len(targs)
            and all(lpo_greater(s, tj, precedence) for tj in targs)
            and _lex_gt(sargs, targs, precedence)):
        return True
    return False
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest rerum/tests/test_termination.py::TestPrecedenceAndLPO -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add rerum/termination.py rerum/tests/test_termination.py
git commit -m "feat(f4): lexicographic path order (lpo_greater)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: orient

**Files:**
- Modify: `rerum/termination.py`
- Test: `rerum/tests/test_termination.py`

- [ ] **Step 1: Write the failing tests**

Append to `rerum/tests/test_termination.py`:

```python
class TestOrient:
    def test_associativity_orients_lr_precedence_independent(self):
        # Right-associativity decreases lexicographically on the shared + head,
        # so it orients regardless of precedence (use []).
        l = ["+", ["+", ["?", "x"], ["?", "y"]], ["?", "z"]]
        r = ["+", ["?", "x"], ["+", ["?", "y"], ["?", "z"]]]
        assert tm.orient(l, r, []) == "lr"

    def test_commutativity_orients_none(self):
        l = ["+", ["?", "x"], ["?", "y"]]
        r = ["+", ["?", "y"], ["?", "x"]]
        assert tm.orient(l, r, ["+"]) is None

    def test_lr_and_rl(self):
        big = ["f", ["g", ["?", "x"]]]
        small = ["g", ["?", "x"]]
        assert tm.orient(big, small, ["f", "g"]) == "lr"
        assert tm.orient(small, big, ["f", "g"]) == "rl"
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest rerum/tests/test_termination.py::TestOrient -v`
Expected: FAIL (`orient` not defined).

- [ ] **Step 3: Add `orient`**

Append to `rerum/termination.py`:

```python
def orient(l: ExprType, r: ExprType,
           precedence: Precedence) -> Optional[str]:
    """Pick the terminating direction for the equation ``l = r``.

    Returns "lr" if ``l >_lpo r`` (rule ``l -> r`` decreases), "rl" if
    ``r >_lpo l``, or ``None`` if this LPO/precedence orients neither (e.g. a
    commutativity axiom, which no reduction order can orient). The orientation
    oracle Knuth-Bendix completion (F5) needs.
    """
    if lpo_greater(l, r, precedence):
        return "lr"
    if lpo_greater(r, l, precedence):
        return "rl"
    return None
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest rerum/tests/test_termination.py::TestOrient -v`
Expected: PASS (3 passed). If `test_associativity_orients_lr_precedence_independent` fails, hand-trace: the two sides share head `+` with arity 2; case 3 requires `s >_lpo` each `tj` and the args decrease lexicographically at the first differing position (`(+ ?x ?y)` vs `?x`), where `lpo_greater(["+",["?","x"],["?","y"]], ["?","x"])` is True (proper subterm). Do NOT weaken the assertion; if it genuinely fails, report it (it indicates an LPO bug from Task 1).

- [ ] **Step 5: Commit**

```bash
git add rerum/termination.py rerum/tests/test_termination.py
git commit -m "feat(f4): orient (the F5 orientation oracle)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: check_termination

**Files:**
- Modify: `rerum/termination.py`
- Test: `rerum/tests/test_termination.py`

- [ ] **Step 1: Write the failing tests**

Append to `rerum/tests/test_termination.py`:

```python
class TestCheckTermination:
    def test_terminating_set(self):
        # f > g > h: each rule's LHS dominates its RHS.
        eng = RuleEngine.from_dsl("""
            @r1: (f (g ?x)) => (g (g :x))
            @r2: (g (h ?x)) => (h :x)
        """)
        report = tm.check_termination(eng, ["f", "g", "h"])
        assert report.terminating is True
        assert report.unoriented == []
        assert report.not_analyzed == []

    def test_commutativity_is_unoriented(self):
        eng = RuleEngine.from_dsl("@c: (+ ?x ?y) => (+ :y :x)")
        report = tm.check_termination(eng, ["+"])
        assert "c" in report.unoriented
        assert report.terminating is False

    def test_not_analyzed_blocks(self):
        eng = RuleEngine.from_dsl("@rest: (f ?x...) => (g :x...)")
        report = tm.check_termination(eng, ["f", "g"])
        assert "rest" in report.not_analyzed
        assert report.terminating is False

    def test_general_boolean(self):
        eng = RuleEngine.from_dsl("@dn: (not (not ?x)) => :x")
        report = tm.check_termination(eng, ["not"])
        assert report.terminating is True   # (not (not x)) >_lpo x (subterm)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest rerum/tests/test_termination.py::TestCheckTermination -v`
Expected: FAIL (`check_termination`/`TerminationReport` not defined).

- [ ] **Step 3: Add `check_termination` + `TerminationReport`**

Append to `rerum/termination.py`:

```python
@dataclass(frozen=True)
class TerminationReport:
    """Result of an LPO termination check.

    ``terminating`` is True iff EVERY rule is analyzable AND oriented
    ``l >_lpo r`` -- a PROOF of termination by this LPO. False means "not proven
    by this precedence" (the rule may be reversed, incomparable, or genuinely
    non-terminating), with ``unoriented``/``not_analyzed`` explaining why. Like
    F2's ``unknown``, it is honest about the limit -- it never claims
    "non-terminating", only "not proven".
    """
    terminating: bool
    oriented: List[Tuple[str, str]]   # (rule name, direction "lr")
    unoriented: List[str]
    not_analyzed: List[str]


def check_termination(engine, precedence: Precedence) -> TerminationReport:
    """LPO termination diagnostic for ``engine``'s enabled rules under
    ``precedence``. Read-only: mutates nothing."""
    oriented: List[Tuple[str, str]] = []
    unoriented: List[str] = []
    not_analyzed: List[str] = []
    for _idx, rule, meta in engine.rule_set():
        pattern, skeleton = rule[0], rule[1]
        if not is_analyzable(pattern, skeleton, meta.condition):
            not_analyzed.append(meta.name)
            continue
        r_term = instantiate_skeleton(skeleton, {})  # [":",x] -> ["?",x]
        if lpo_greater(pattern, r_term, precedence):
            oriented.append((meta.name, "lr"))
        else:
            unoriented.append(meta.name)
    terminating = (not unoriented) and (not not_analyzed)
    return TerminationReport(
        terminating=terminating, oriented=oriented,
        unoriented=unoriented, not_analyzed=not_analyzed,
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest rerum/tests/test_termination.py::TestCheckTermination -v`
Expected: PASS (4 passed). If `test_terminating_set` fails, hand-trace each rule under `["f","g","h"]`: `(f (g ?x)) >_lpo (g (g ?x))` via case 1 (subterm `(g ?x)` dominates `(g (g ?x))`? -- actually via case 2, `f > g` and `(f (g ?x))` beats each `g`-arg) and `(g (h ?x)) >_lpo (h ?x)` via case 1 (`(h ?x)` is a proper subterm). If a rule does not orient under this precedence, adjust the rule set to one whose every rule provably decreases (do NOT weaken `terminating is True`); the point is a genuinely terminating example.

- [ ] **Step 5: Commit**

```bash
git add rerum/termination.py rerum/tests/test_termination.py
git commit -m "feat(f4): check_termination + TerminationReport

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Newman integration into check_confluence

**Files:**
- Modify: `rerum/confluence.py` (`ConfluenceReport`, `check_confluence`)
- Test: `rerum/tests/test_termination.py`

- [ ] **Step 1: Write the failing tests**

Append to `rerum/tests/test_termination.py`:

```python
from rerum import confluence as cf  # noqa: E402


class TestInstantiateSkeletonReuse:
    def test_empty_subst_converts_colon_to_var(self):
        assert cf.instantiate_skeleton([":", "x"], {}) == ["?", "x"]
        assert cf.instantiate_skeleton(["+", [":", "x"], "0"], {}) == \
            ["+", ["?", "x"], "0"]


class TestNewman:
    def test_locally_confluent_and_terminating_is_confluent(self):
        eng = RuleEngine.from_dsl("""
            @r1: (f (g ?x)) => (h :x)
            @r2: (g ?x) => (k :x)
            @r3: (f (k ?x)) => (h :x)
        """)
        report = cf.check_confluence(eng, precedence=["f", "g", "k", "h"])
        assert report.terminating is True
        assert report.confluent is True

    def test_non_joinable_is_not_confluent(self):
        eng = RuleEngine.from_dsl("""
            @l: (f (f ?x)) => a
            @r: (f ?x) => b
        """)
        report = cf.check_confluence(eng, precedence=["f", "a", "b"])
        assert report.confluent is False   # a genuine non_joinable witness

    def test_locally_confluent_but_unorientable_is_unknown(self):
        # Single overlap-free rule that LPO cannot orient (RHS bigger, big<...).
        eng = RuleEngine.from_dsl("@up: (small ?x) => (big (big :x))")
        report = cf.check_confluence(eng, precedence=["big", "small"])
        assert report.non_joinable == [] and report.unknown == []
        assert report.locally_confluent is True
        assert report.terminating is False
        assert report.confluent is None

    def test_backward_compat_no_precedence(self):
        eng = RuleEngine.from_dsl("""
            @l: (f (f ?x)) => a
            @r: (f ?x) => b
        """)
        report = cf.check_confluence(eng)   # no precedence
        assert report.terminating is None
        assert report.confluent is None
        assert report.locally_confluent is False
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest rerum/tests/test_termination.py::TestNewman -v`
Expected: FAIL (`check_confluence` has no `precedence` kwarg / `ConfluenceReport` has no `terminating`/`confluent`). `TestInstantiateSkeletonReuse` passes already.

- [ ] **Step 3: Extend `ConfluenceReport`**

In `rerum/confluence.py`, find the `ConfluenceReport` field block:

```python
    locally_confluent: bool
    critical_pairs: List[CriticalPair]
    non_joinable: List[CriticalPair]
    unknown: List[CriticalPair]
    not_analyzed: List[str]
    analyzed_pair_count: int
```

Replace with (add two defaulted fields at the END -- additive, frozen-safe):

```python
    locally_confluent: bool
    critical_pairs: List[CriticalPair]
    non_joinable: List[CriticalPair]
    unknown: List[CriticalPair]
    not_analyzed: List[str]
    analyzed_pair_count: int
    terminating: Optional[bool] = None
    confluent: Optional[bool] = None
```

- [ ] **Step 4: Extend `check_confluence` with the Newman block**

In `rerum/confluence.py`, change the `check_confluence` signature:

```python
def check_confluence(engine, *, max_steps: int = 1000) -> ConfluenceReport:
```

to:

```python
def check_confluence(engine, *, max_steps: int = 1000,
                     precedence=None) -> ConfluenceReport:
```

Then find the tail of `check_confluence`:

```python
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

Replace with:

```python
    locally_confluent = not non_joinable and not unknown

    # Newman integration (F4). With a precedence, decide global confluence.
    # confluent=False keys on a GENUINE non_joinable witness (distinct
    # irreducible normal forms), NOT bare not-locally-confluent: an unknown-only
    # failure is UNDECIDED (confluent=None), never a false negative.
    terminating = None
    confluent = None
    if precedence is not None:
        from .termination import check_termination as _check_termination
        terminating = _check_termination(engine, precedence).terminating
        if non_joinable:
            confluent = False
        elif unknown:
            confluent = None
        elif terminating:
            confluent = True  # locally confluent + terminating (Newman)
        else:
            confluent = None  # locally confluent but termination not proven

    return ConfluenceReport(
        locally_confluent=locally_confluent,
        critical_pairs=decided,
        non_joinable=non_joinable,
        unknown=unknown,
        not_analyzed=not_analyzed,
        analyzed_pair_count=analyzed,
        terminating=terminating,
        confluent=confluent,
    )
```

(`Optional` is already imported in `confluence.py`.)

- [ ] **Step 5: Run to verify pass**

Run: `pytest rerum/tests/test_termination.py::TestNewman rerum/tests/test_termination.py::TestInstantiateSkeletonReuse -v`
Expected: PASS (5 passed). If `test_locally_confluent_and_terminating_is_confluent` fails on `terminating`, hand-check that each rule orients under `["f","g","k","h"]` (r1: `f>g`; r2: `g>k`; r3: `f>h`); if `confluent` is wrong, re-check the Newman branch order.

- [ ] **Step 6: Confluence regression guard**

Run: `pytest rerum/tests/test_confluence.py -q`
Expected: PASS (the two new defaulted fields and the `precedence=None` default leave all F2 behavior unchanged).

- [ ] **Step 7: Commit**

```bash
git add rerum/confluence.py rerum/tests/test_termination.py
git commit -m "feat(f4): Newman integration -- check_confluence(precedence) -> confluent

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Engine wrappers, re-exports, and the full-suite gate

**Files:**
- Modify: `rerum/engine.py`, `rerum/__init__.py`
- Test: `rerum/tests/test_termination.py`

- [ ] **Step 1: Write the failing tests**

Append to `rerum/tests/test_termination.py`:

```python
class TestEngineAndExports:
    def test_engine_check_termination_delegates(self):
        eng = RuleEngine.from_dsl("@dn: (not (not ?x)) => :x")
        report = eng.check_termination(["not"])
        assert report.terminating is True

    def test_engine_check_confluence_precedence_passthrough(self):
        eng = RuleEngine.from_dsl("""
            @l: (f (f ?x)) => a
            @r: (f ?x) => b
        """)
        assert eng.check_confluence(precedence=["f", "a", "b"]).confluent is False
        assert eng.check_confluence().confluent is None  # no precedence

    def test_public_reexports(self):
        import rerum
        for name in ("lpo_greater", "orient", "check_termination",
                     "TerminationReport"):
            assert name in rerum.__all__
            assert hasattr(rerum, name)
        assert "_prec_gt" not in rerum.__all__  # private stays private

    def test_import_smoke_no_cycle(self):
        # Both import orders succeed (pins the lazy-import boundary).
        import importlib
        importlib.import_module("rerum.termination")
        importlib.import_module("rerum.confluence")
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest rerum/tests/test_termination.py::TestEngineAndExports -v`
Expected: FAIL (`eng.check_termination` undefined; `eng.check_confluence` has no `precedence`; re-exports missing).

- [ ] **Step 3: Add the engine methods**

In `rerum/engine.py`, find the existing `check_confluence` method (in the "Confluence Analysis (F2)" section, ~line 4279):

```python
    def check_confluence(self, *, max_steps: int = 1000) -> "ConfluenceReport":
        """Local-confluence diagnostic for this engine's enabled rules.

        Computes critical pairs and checks joinability (modulo the loaded
        theory). Read-only. See ``rerum.confluence.ConfluenceReport``.
        """
        from .confluence import check_confluence as _check_confluence
        return _check_confluence(self, max_steps=max_steps)
```

Replace with (add `precedence` passthrough + a new `check_termination` method):

```python
    def check_confluence(self, *, max_steps: int = 1000,
                         precedence=None) -> "ConfluenceReport":
        """Local-confluence diagnostic for this engine's enabled rules.

        Computes critical pairs and checks joinability (modulo the loaded
        theory). With ``precedence`` (a list of function symbols in decreasing
        order, roadmap F4), also reports global ``confluent`` via Newman's
        Lemma. Read-only. See ``rerum.confluence.ConfluenceReport``.
        """
        from .confluence import check_confluence as _check_confluence
        return _check_confluence(self, max_steps=max_steps,
                                 precedence=precedence)

    def check_termination(self, precedence) -> "TerminationReport":
        """LPO termination diagnostic for this engine's enabled rules under
        ``precedence`` (roadmap F4). Read-only. See
        ``rerum.termination.TerminationReport``."""
        from .termination import check_termination as _check_termination
        return _check_termination(self, precedence)
```

- [ ] **Step 4: Add the re-exports**

In `rerum/__init__.py`, find the `# Confluence analysis (F2)` import block:

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

Immediately AFTER it, add:

```python
# Termination analysis (F4)
from .termination import (
    lpo_greater,
    orient,
    check_termination,
    TerminationReport,
)
```

And in the `__all__` list, after the `# Confluence analysis` section's entries, add:

```python
    # Termination analysis
    "lpo_greater",
    "orient",
    "check_termination",
    "TerminationReport",
```

- [ ] **Step 5: Run to verify pass**

Run: `pytest rerum/tests/test_termination.py::TestEngineAndExports -v`
Expected: PASS (4 passed).

- [ ] **Step 6: Whole termination file + full suite**

Run: `pytest rerum/tests/test_termination.py -q`
Expected: PASS (all termination tests).

Run: `pytest -q`
Expected: PASS with NO new failures vs baseline. Report the total count.

Run: `pytest rerum/tests/test_mcp_no_domain.py -q`
Expected: PASS (no domain operator literal in `rerum/mcp/`).

- [ ] **Step 7: ASCII check**

Run: `LC_ALL=C grep -n '[^[:print:][:space:]]' rerum/termination.py rerum/tests/test_termination.py && echo "FOUND non-ASCII" || echo "clean"`
Expected: `clean`.

- [ ] **Step 8: Commit**

```bash
git add rerum/engine.py rerum/__init__.py rerum/tests/test_termination.py
git commit -m "feat(f4): engine.check_termination + check_confluence(precedence) + re-exports

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Post-Implementation

F4 is complete: `engine.check_termination(precedence)` proves termination by LPO; `orient` is the F5 orientation oracle; `engine.check_confluence(precedence=...)` upgrades "locally confluent" to a genuine `confluent` verdict via Newman. Follow-up (the roadmap capstone): F5 Knuth-Bendix completion, which drives an orient-and-add loop over F2's critical pairs using F4's `orient` + `check_termination`. F3 (AC-matching) remains the way to handle commutativity-style axioms no plain reduction order can orient.

Use superpowers:finishing-a-development-branch to complete the work. Note: this work is on `main` directly; present the push decision to the user.
