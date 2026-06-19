# F5: Knuth-Bendix Completion (basic) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a set of equations into a confluent + terminating rewrite system via the basic Knuth-Bendix completion loop (orient + critical-pairs + normalize + add), self-validated by `check_confluence`.

**Architecture:** A new pure module `rerum/completion.py` orchestrating F4 `orient` + F2 `critical_pairs` + the engine's `simplify` in an orient-and-add fixpoint. Plus a thin `engine.complete(...)` wrapper and `rerum` re-exports. Read-only; builds fresh engines internally for normalization.

**Tech Stack:** Python 3.9+, pytest. Terms are nested lists (`["?", name]` variables, constants, compounds). Equations/rules are `(l, r)` term pairs.

**Spec:** `docs/superpowers/specs/2026-06-19-f5-completion-design.md` (read it first; this plan implements it exactly, including the verification fixes: unified signature with `max_steps`; `l==r` filtered BEFORE orient; consistent failure-path `rules`; `CompletionResult` defaults; `iterations` = pass count; the syntactic `s==t` join test). The pinned test examples below were empirically verified by simulating the loop over the existing F2/F4 primitives.

**Constraints:** ASCII only (commit hook). No domain operator hardcoded in `rerum/`. Do NOT commit `.mcp.json`.

---

## Key code facts (verified)

- `RuleEngine.from_rules(rules)` builds an engine from a list of `[pattern, skeleton]` pairs. `engine.simplify(expr, max_steps=N)` reduces by rules. `engine.rule_set()` yields `(idx, [pattern, skeleton], meta)` (post-desugar, disabled excluded; `meta.name`/`meta.condition`).
- `confluence.critical_pairs(records)` takes a list of `DirectedRule(name, pattern, skeleton, condition)` and returns `(pairs, not_analyzed)`. A `CriticalPair`'s `.left`/`.right` are TERMS in `["?", name]` representation (with gensym-renamed variable names -- harmless). `confluence.is_analyzable(pattern, skeleton, condition)`. `confluence.instantiate_skeleton(skeleton, {})` maps `[":", x] -> ["?", x]`.
- `termination.orient(l, r, precedence)` returns `"lr"`/`"rl"`/`None` on `["?", name]` terms.
- `confluence.check_confluence(engine, *, max_steps=1000, precedence=None)` (F4) reports `.confluent` (`True` iff locally confluent + terminating, given a precedence).
- Import direction (no load-time cycle): `engine`/`confluence`/`termination` do NOT import a `completion` module; so `completion.py` may TOP-import `RuleEngine` from `.engine` and the F2/F4 functions; `engine.complete` lazy-imports `complete`.
- `rerum/__init__.py` has `# Confluence analysis (F2)` and `# Termination analysis (F4)` import blocks + matching `__all__` sections.

**Verified test values (from simulating the loop):**
- Associativity `(* (* x y) z) = (* x (* y z))` (precedence `["*"]`) -> `"complete"`, 1 rule, `iterations == 1`, `rules[0] == (l, r)`.
- `[(["f",["g",["?","x"]]], "a"), (["g",["g",["?","x"]]], ["?","x"])]` (precedence `["f","g","a"]`) -> `"complete"`, 3 rules, `iterations == 2`; `to_engine()` reduces `(f (g a))` and `(g (g a))` to `"a"`; `check_confluence(to_engine(), precedence=["f","g","a"]).confluent is True`.
- Same set, `max_iterations=1` -> `"max_iterations"`, `iterations == 1`.
- Commutativity `(+ ?x ?y) = (+ ?y ?x)` (precedence `["+"]`) -> `"failed"`, `rules == []`.

---

## File Structure

- **Create** `rerum/completion.py`: `_term_to_skeleton`, `_dedup`, `CompletionResult` (+ `to_engine`), `complete`. ONE responsibility: completion. Pure-ish (builds fresh engines for normalization; reads nothing global).
- **Modify** `rerum/engine.py`: add `RuleEngine.complete`.
- **Modify** `rerum/__init__.py`: re-export `complete`, `CompletionResult`.
- **Create** `rerum/tests/test_completion.py`.

---

### Task 1: Bridge helpers (`_term_to_skeleton`, `_dedup`)

**Files:**
- Create: `rerum/completion.py`
- Test: `rerum/tests/test_completion.py` (create)

- [ ] **Step 1: Create the test file with the helper tests**

Create `rerum/tests/test_completion.py`:

```python
"""F5: basic Knuth-Bendix completion."""

from rerum import completion as cmp
from rerum import confluence as cf
from rerum.engine import RuleEngine


def _v(n):
    return ["?", n]


class TestBridgeHelpers:
    def test_term_to_skeleton_variable(self):
        assert cmp._term_to_skeleton(["?", "x"]) == [":", "x"]

    def test_term_to_skeleton_inverse_of_instantiate(self):
        # _term_to_skeleton is the inverse of instantiate_skeleton(.., {}).
        t = ["+", ["?", "x"], ["g", ["?", "y"]], "0"]
        skel = cmp._term_to_skeleton(t)
        assert skel == ["+", [":", "x"], ["g", [":", "y"]], "0"]
        assert cf.instantiate_skeleton(skel, {}) == t

    def test_dedup_preserves_order_drops_structural_duplicates(self):
        a = (["a"], "b")
        c = (["c"], "d")
        a2 = (["a"], "b")  # equal by value, distinct identity
        assert cmp._dedup([a, c, a2]) == [(["a"], "b"), (["c"], "d")]
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest rerum/tests/test_completion.py::TestBridgeHelpers -v`
Expected: FAIL with `ModuleNotFoundError`/`AttributeError` (no `rerum.completion`).

- [ ] **Step 3: Create `rerum/completion.py` with the helpers**

Create `rerum/completion.py`:

```python
"""F5: basic Knuth-Bendix completion (read-only analysis).

Turns a set of EQUATIONS into a CONFLUENT + TERMINATING rewrite system by the
basic completion loop: orient each equation into a rule (F4 orient), compute
critical pairs (F2 critical_pairs), normalize both sides with the current rules
(engine.simplify), and add any un-joined pair as a new oriented rule, until
every critical pair joins. Pure ORCHESTRATION of F2 + F4 + the engine; almost
no new math.

GENERAL ENGINE: the precedence and equations are DATA. First-order only. The
join test is SYNTACTIC (s == t) -- sound here because the internal normalization
engines (built by RuleEngine.from_rules) carry NO theory, so _canonicalize is
the identity and s == t coincides with F2's join test. A modulo-theory (AC)
extension must switch to _canonicalize-based comparison, as F2 does.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .rewriter import ExprType
from .engine import RuleEngine
from .confluence import critical_pairs, DirectedRule
from .termination import orient


def _term_to_skeleton(term: ExprType) -> ExprType:
    """Convert a TERM (``["?", name]`` variables) to a SKELETON
    (``[":", name]`` references) -- the forward of
    ``instantiate_skeleton(.., {})``. Recurses compounds; atoms unchanged."""
    if isinstance(term, list) and len(term) == 2 and term[0] == "?":
        return [":", term[1]]
    if isinstance(term, list):
        return [_term_to_skeleton(sub) for sub in term]
    return term


def _dedup(rules: list) -> list:
    """Drop structurally-duplicate ``(l, r)`` pairs, preserving first-occurrence
    order (O(n^2) list membership; n is small)."""
    out: list = []
    for rule in rules:
        if rule not in out:
            out.append(rule)
    return out
```

(Note: `dataclass`, `Optional`, `Tuple`, `RuleEngine`, `critical_pairs`, `DirectedRule`, `orient` are imported now but used by Task 2's `CompletionResult`/`complete` -- intentional forward declarations for the module being built up. Leave them. `engine.complete` uses `is_analyzable`/`instantiate_skeleton`, but it lazy-imports those itself, so `completion.py` does NOT import them.)

- [ ] **Step 4: Run to verify pass**

Run: `pytest rerum/tests/test_completion.py::TestBridgeHelpers -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add rerum/completion.py rerum/tests/test_completion.py
git commit -m "feat(f5): completion bridge helpers (_term_to_skeleton, _dedup)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `CompletionResult` + the `complete` loop

**Files:**
- Modify: `rerum/completion.py`
- Test: `rerum/tests/test_completion.py`

- [ ] **Step 1: Write the failing tests**

Append to `rerum/tests/test_completion.py`:

```python
class TestComplete:
    def test_associativity_completes_to_one_rule(self):
        l = ["*", ["*", _v("x"), _v("y")], _v("z")]
        r = ["*", _v("x"), ["*", _v("y"), _v("z")]]
        result = cmp.complete([(l, r)], ["*"])
        assert result.status == "complete"
        assert result.iterations == 1
        assert len(result.rules) == 1
        assert result.rules[0] == (l, r)   # oriented to the right

    def test_add_a_rule_converges(self):
        eqs = [
            (["f", ["g", _v("x")]], "a"),
            (["g", ["g", _v("x")]], _v("x")),
        ]
        result = cmp.complete(eqs, ["f", "g", "a"])
        assert result.status == "complete"
        assert result.iterations == 2
        assert len(result.rules) == 3   # adds (f ?v) -> a
        # The derived rule sends f-of-anything to a.
        assert any(lhs[0] == "f" and len(lhs) == 2 and lhs[1][0] == "?"
                   and rhs == "a" for (lhs, rhs) in result.rules)

    def test_failed_on_unorientable_input(self):
        eqs = [(["+", _v("x"), _v("y")], ["+", _v("y"), _v("x")])]
        result = cmp.complete(eqs, ["+"])
        assert result.status == "failed"
        assert result.failed_equation is not None
        assert result.rules == []

    def test_max_iterations_returns_not_hangs(self):
        # The add-a-rule set needs 2 passes; cap at 1 -> max_iterations.
        eqs = [
            (["f", ["g", _v("x")]], "a"),
            (["g", ["g", _v("x")]], _v("x")),
        ]
        result = cmp.complete(eqs, ["f", "g", "a"], max_iterations=1)
        assert result.status == "max_iterations"
        assert result.iterations == 1

    def test_trivial_equation_filtered_not_failed(self):
        # l == r is dropped BEFORE orient, so it is not a spurious "failed".
        eqs = [(_v("x"), _v("x"))]
        result = cmp.complete(eqs, [])
        assert result.status == "complete"
        assert result.rules == []
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest rerum/tests/test_completion.py::TestComplete -v`
Expected: FAIL (`complete`/`CompletionResult` not defined).

- [ ] **Step 3: Add `CompletionResult` + `complete`**

Append to `rerum/completion.py`:

```python
@dataclass(frozen=True)
class CompletionResult:
    """Result of a basic Knuth-Bendix completion run.

    ``status`` and ``rules`` are always set; ``failed_equation`` and
    ``iterations`` carry defaults so every return path constructs cleanly.
    A ``"complete"`` result is genuinely confluent + terminating (every critical
    pair joined; every rule oriented ``l >_lpo r``, hence terminating; Newman).
    ``"failed"`` means an equation no reduction order can orient (e.g.
    commutativity). ``"max_iterations"`` means basic completion did not converge
    within the budget (it is only a semi-decision procedure).
    """
    status: str
    rules: List[Tuple[ExprType, ExprType]]
    failed_equation: Optional[Tuple[ExprType, ExprType]] = None
    iterations: int = 0

    def to_engine(self) -> "RuleEngine":
        """Build a fresh ``RuleEngine`` loaded with the completed rules -- the
        ergonomic way to USE the result."""
        return RuleEngine.from_rules(
            [[l, _term_to_skeleton(r)] for (l, r) in self.rules]
        )


def complete(equations, precedence, *, max_iterations: int = 100,
             max_steps: int = 1000) -> CompletionResult:
    """Basic Knuth-Bendix completion of ``equations`` (a list of ``(l, r)`` term
    pairs in ``["?", name]`` form) under ``precedence``. Read-only."""
    # 1. Orient the input. Drop trivial l == r BEFORE orient (orient returns
    #    None on structurally-equal terms, which would be a spurious "failed").
    rules: List[Tuple[ExprType, ExprType]] = []
    for (l, r) in equations:
        if l == r:
            continue
        d = orient(l, r, precedence)
        if d is None:
            return CompletionResult(status="failed", rules=list(rules),
                                    failed_equation=(l, r), iterations=0)
        rules.append((l, r) if d == "lr" else (r, l))
    rules = _dedup(rules)

    # 2. Fixpoint: orient-and-add until no critical pair is un-joined.
    for iteration in range(max_iterations):
        passes = iteration + 1
        records = [
            DirectedRule(name=str(i), pattern=l,
                         skeleton=_term_to_skeleton(r), condition=None)
            for i, (l, r) in enumerate(rules)
        ]
        eng = RuleEngine.from_rules(
            [[l, _term_to_skeleton(r)] for (l, r) in rules]
        )
        pairs, _na = critical_pairs(records)
        new_rules: List[Tuple[ExprType, ExprType]] = []
        for cp in pairs:
            try:
                s = eng.simplify(cp.left, max_steps=max_steps)
                t = eng.simplify(cp.right, max_steps=max_steps)
            except RecursionError:
                s, t = cp.left, cp.right  # treat as NOT joining (conservative)
            if s == t:
                continue
            d = orient(s, t, precedence)
            if d is None:
                return CompletionResult(status="failed", rules=list(rules),
                                        failed_equation=(s, t),
                                        iterations=passes)
            new = (s, t) if d == "lr" else (t, s)
            if new not in rules and new not in new_rules:
                new_rules.append(new)
        if not new_rules:
            return CompletionResult(status="complete", rules=list(rules),
                                    iterations=passes)
        rules = _dedup(rules + new_rules)

    return CompletionResult(status="max_iterations", rules=list(rules),
                            iterations=max_iterations)
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest rerum/tests/test_completion.py::TestComplete -v`
Expected: PASS (5 passed). If `test_add_a_rule_converges` fails on the count/iterations, STOP and report -- the example was empirically verified, so a failure indicates a bug in the loop or in F2/F4, not the data.

- [ ] **Step 5: Commit**

```bash
git add rerum/completion.py rerum/tests/test_completion.py
git commit -m "feat(f5): CompletionResult + the orient-and-add completion loop

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Self-validation, `to_engine` reduction, generality (tests)

**Files:**
- Test: `rerum/tests/test_completion.py` (these pass on Task 2's code; they lock in the self-validation + generality contracts)

- [ ] **Step 1: Write the validation tests**

Append to `rerum/tests/test_completion.py`:

```python
class TestSelfValidationAndGenerality:
    def test_complete_result_is_confluent(self):
        # The capstone validates itself: a "complete" system is confluent under
        # the precedence, via F2+F4's check_confluence. Same max_steps.
        eqs = [
            (["f", ["g", _v("x")]], "a"),
            (["g", ["g", _v("x")]], _v("x")),
        ]
        prec = ["f", "g", "a"]
        result = cmp.complete(eqs, prec, max_steps=1000)
        report = cf.check_confluence(result.to_engine(), precedence=prec,
                                     max_steps=1000)
        assert report.confluent is True
        assert report.terminating is True

    def test_to_engine_reduces(self):
        eqs = [
            (["f", ["g", _v("x")]], "a"),
            (["g", ["g", _v("x")]], _v("x")),
        ]
        eng = cmp.complete(eqs, ["f", "g", "a"]).to_engine()
        assert eng.simplify(["f", ["g", "a"]]) == "a"
        assert eng.simplify(["g", ["g", "a"]]) == "a"

    def test_general_boolean(self):
        # Same code completes a non-arithmetic equation set.
        eqs = [(["not", ["not", _v("x")]], _v("x"))]
        result = cmp.complete(eqs, ["not"])
        assert result.status == "complete"
        assert len(result.rules) == 1

    def test_general_arithmetic(self):
        eqs = [(["+", _v("x"), "0"], _v("x"))]
        result = cmp.complete(eqs, ["+", "0"])
        assert result.status == "complete"
        assert len(result.rules) == 1
```

- [ ] **Step 2: Run to verify pass (no new code)**

Run: `pytest rerum/tests/test_completion.py::TestSelfValidationAndGenerality -v`
Expected: PASS (4 passed). These exercise Task 2's `complete`/`to_engine` plus F2/F4. If `test_complete_result_is_confluent` reports `confluent is None`, the `check_confluence` budget is too small -- it already matches `complete`'s `max_steps=1000`, so a `None` indicates a real issue; STOP and report (do NOT weaken the assertion).

- [ ] **Step 3: Commit**

```bash
git add rerum/tests/test_completion.py
git commit -m "test(f5): self-validation (confluent), to_engine reduction, generality

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: `engine.complete`, re-exports, and the full-suite gate

**Files:**
- Modify: `rerum/engine.py`, `rerum/__init__.py`
- Test: `rerum/tests/test_completion.py`

- [ ] **Step 1: Write the failing tests**

Append to `rerum/tests/test_completion.py`:

```python
class TestEngineAndExports:
    def test_engine_complete_extracts_and_matches(self):
        # Engine rules become equations; engine.complete matches the pure call.
        eng = RuleEngine.from_dsl("""
            @r1: (f (g ?x)) => a
            @r2: (g (g ?x)) => :x
        """)
        report = eng.complete(["f", "g", "a"])
        assert report.status == "complete"
        assert len(report.rules) == 3

    def test_engine_complete_skips_non_analyzable(self):
        # A ?... rule is not analyzable -> excluded from the extracted equations.
        eng = RuleEngine.from_dsl("@rest: (f ?x...) => (g :x...)")
        report = eng.complete(["f", "g"])
        # No analyzable equations -> trivially complete with no rules.
        assert report.status == "complete"
        assert report.rules == []

    def test_public_reexports(self):
        import rerum
        for name in ("complete", "CompletionResult"):
            assert name in rerum.__all__
            assert hasattr(rerum, name)

    def test_import_smoke_no_cycle(self):
        import importlib
        importlib.import_module("rerum.completion")
        importlib.import_module("rerum.engine")
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest rerum/tests/test_completion.py::TestEngineAndExports -v`
Expected: FAIL (`eng.complete` undefined; re-exports missing).

- [ ] **Step 3: Add the engine method**

In `rerum/engine.py`, find the `check_termination` method (added by F4, in the confluence/termination area). Immediately AFTER it, insert:

```python
    def complete(self, precedence, *, max_iterations: int = 100,
                 max_steps: int = 1000) -> "CompletionResult":
        """Basic Knuth-Bendix completion (roadmap F5) of this engine's
        ANALYZABLE rules, treated as equations, under ``precedence``. Returns a
        ``CompletionResult`` (the completed rule system, or failed /
        max_iterations). Read-only -- builds fresh engines internally; mutates
        nothing. See ``rerum.completion``."""
        from .completion import complete as _complete
        from .confluence import (
            is_analyzable as _is_analyzable,
            instantiate_skeleton as _instantiate,
        )
        equations = [
            (rule[0], _instantiate(rule[1], {}))
            for _idx, rule, meta in self.rule_set()
            if _is_analyzable(rule[0], rule[1], meta.condition)
        ]
        return _complete(equations, precedence, max_iterations=max_iterations,
                         max_steps=max_steps)
```

(STRING annotation `"CompletionResult"`; lazy imports; no top-level completion import in engine.py.)

- [ ] **Step 4: Add the re-exports**

In `rerum/__init__.py`, find the `# Termination analysis (F4)` import block. Immediately AFTER it, add:

```python
# Completion (F5)
from .completion import (
    complete,
    CompletionResult,
)
```

And in the `__all__` list, after the `# Termination analysis` section's entries, add:

```python
    # Completion
    "complete",
    "CompletionResult",
```

- [ ] **Step 5: Run to verify pass**

Run: `pytest rerum/tests/test_completion.py::TestEngineAndExports -v`
Expected: PASS (4 passed).

- [ ] **Step 6: Whole completion file + full suite + guards**

Run: `pytest rerum/tests/test_completion.py -q`
Expected: PASS (all completion tests).

Run: `pytest -q`
Expected: PASS with NO new failures vs baseline. Report the total count.

Run: `pytest rerum/tests/test_mcp_no_domain.py -q`
Expected: PASS (no domain operator literal in `rerum/mcp/`).

- [ ] **Step 7: ASCII check**

Run: `LC_ALL=C grep -n '[^[:print:][:space:]]' rerum/completion.py rerum/tests/test_completion.py && echo "FOUND non-ASCII" || echo "clean"`
Expected: `clean`.

- [ ] **Step 8: Commit**

```bash
git add rerum/engine.py rerum/__init__.py rerum/tests/test_completion.py
git commit -m "feat(f5): engine.complete + re-exports

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Post-Implementation

F5 is complete: `engine.complete(precedence)` turns a rule set's equations into a confluent + terminating system, self-validated by `check_confluence`. The TRS-frontier stack is now end-to-end: F1 (normalize modulo a theory) -> F2 (critical pairs + joinability) -> F4 (termination + Newman) -> F5 (completion). Follow-ups (tracked in the spec): F5b inter-reduction (COMPOSE/COLLAPSE -- convergence on group axioms, minimal systems), unfailing/ordered completion, and F3 AC-matching (for commutativity-style axioms no plain order can orient).

Use superpowers:finishing-a-development-branch to complete the work. Note: this work is on `main` directly; present the push decision to the user.
