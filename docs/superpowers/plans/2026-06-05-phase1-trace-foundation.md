# Phase 1: Trace Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Make every rewrite step self-contained (rule identity, direction, bindings, redex path, guard result, rationale, kind) and make labeled whole-expression derivations reconstructible, without breaking the legacy redex-local `before`/`after` contract.

**Architecture:** Extend the trace data model in `rerum/trace.py` (pure helpers plus richer `RewriteStep`/`RewriteTrace`), thread an accumulating redex `path` through the three strategy drivers in `rerum/engine.py`, populate the new step fields at the four emit sites, and propagate labeled single-step edges from `_all_single_rewrites` into `prove_equal`/`EqualityProof` and `minimize`/`OptimizationResult`.

**Tech Stack:** Python 3.9+, pytest (config in `pyproject.toml`), no new runtime dependencies (`hashlib` is stdlib).

---

## File Structure

Files touched (all under `/home/spinoza/github/repos/rerum`):

- `rerum/trace.py` (EXTEND): `splice_at`, `rule_identity` helpers; `RewriteStep` new keyword fields + `__slots__` + `to_dict`; `RewriteTrace.to_global_sequence` + `to_dict(global_sequence=...)`; `RewriteStep.__eq__`/`__hash__` for expression-endpoint comparison.
- `rerum/engine.py` (EDIT): path threading in `_simplify_exhaustive` (line ~2497), `_bottomup_pass` (line ~2650), `_topdown_pass` (line ~2755), `apply_once` (line ~2023); populate fields at the four `RewriteStep(...)` sites (lines ~2066, ~2550, ~2692, ~2791); `_fire_rule_applied` `expr_path` (line ~2208); labeled `_all_single_rewrites` (line ~3181); labeled `prove_equal` (line ~3375) with synthetic-initial steps and `reconstruct_path`; `EqualityProof.path_a`/`path_b` doc (line ~1214); `minimize`/`OptimizationResult.derivation` (line ~3599).
- `rerum/optimize.py` (EDIT): `OptimizationResult` gains `.derivation` (class at line ~101).
- `rerum/__init__.py` (EDIT): export `rule_identity`, `splice_at` from `.trace`.
- `rerum/tests/test_trace_situated.py` (NEW): tests for the situated step model, helpers, global reconstruction, path threading, populated fields.
- `rerum/tests/test_prove_equal.py` (EXTEND): tests for labeled proof paths (new class `TestProveEqualLabeledPaths`).
- `rerum/tests/test_optimization.py` (EXTEND): tests for `OptimizationResult.derivation`.

Backward-compat invariants that MUST hold at every task:

- `RewriteStep(rule_index=..., metadata=..., before=..., after=...)` positional/keyword construction keeps working; `before`/`after` keep aliasing the redex-local edit.
- `RewriteStep.to_dict()` keeps emitting `rule_index`, `rule_name`, `description`, `before`, `after` (only ADD keys).
- `RewriteTrace.to_dict()` keeps emitting `initial`, `final`, `steps`, `step_count`.
- `EqualityProof.path_a`/`path_b` keep accepting plain-expression lists (the directly-constructed `EqualityProof` in `test_prove_equal.py` lines 91-126), AND, for `prove_equal(trace=True)`, the path elements compare equal to the node expression at their endpoints (lines 262-293 assert `path_a[0] == start`, `path_a[-1] == common`, `len == 1` for identical exprs).

Resolution of the contract-vs-test tension (see final report): `reconstruct_path` returns `List[RewriteStep]` where each step's `after` holds the node expression and the path begins with a synthetic initial step (`rule_index=-1`, `kind="initial"`, `before == after == start_expr`). `RewriteStep.__eq__` compares against a non-`RewriteStep` operand by `self.after == operand`, so `path_a[0] == start` and `path_a[-1] == common` still hold. Directly-constructed `EqualityProof` paths that are plain lists stay plain lists (no conversion), so `to_dict_with_paths` still emits the raw expressions.

---

### Task 1: Pure helpers `splice_at` and `rule_identity` in trace.py

Adds the two pure, dependency-free module helpers the rest of the phase builds on. No behavior change anywhere else yet.

**Files:**
- `rerum/trace.py` (EDIT): add `import hashlib`; add `splice_at` and `rule_identity` after the imports (around line 18, before `class RewriteStep`).
- `rerum/__init__.py` (EDIT): export both from `.trace`.
- `rerum/tests/test_trace_situated.py` (NEW).

- [ ] **Step 1: Failing test for `splice_at` and `rule_identity`.**

Create `rerum/tests/test_trace_situated.py`:

```python
"""Tests for the situated (self-contained) trace model: helpers, fields,
global reconstruction, path threading."""

import json

import pytest

from rerum import RuleEngine, E, RewriteStep, RewriteTrace
from rerum.trace import splice_at, rule_identity
from rerum.engine import RuleMetadata


class TestSpliceAt:
    """splice_at(root, path, subtree) replaces the subtree at a child path."""

    def test_empty_path_replaces_root(self):
        assert splice_at(["+", "a", "b"], [], ["*", "c", "d"]) == ["*", "c", "d"]

    def test_single_index(self):
        # path [1] addresses the first operand of (+ a b)
        assert splice_at(["+", "a", "b"], [1], "z") == ["+", "z", "b"]

    def test_nested_index(self):
        root = ["+", ["*", "a", "b"], "c"]
        # path [1, 2] addresses 'b' inside (* a b)
        assert splice_at(root, [1, 2], "Z") == ["+", ["*", "a", "Z"], "c"]

    def test_does_not_mutate_root(self):
        root = ["+", "a", "b"]
        out = splice_at(root, [1], "z")
        assert root == ["+", "a", "b"]
        assert out is not root


class TestRuleIdentity:
    """rule_identity prefers metadata.name, else hashes (pattern, skeleton)."""

    def test_named_rule_uses_name(self):
        meta = RuleMetadata(name="add-zero")
        assert rule_identity(meta, ["+", "?x", 0], ":x") == "add-zero"

    def test_anonymous_rule_uses_hash(self):
        meta = RuleMetadata(name=None)
        rid = rule_identity(meta, ["+", "?x", 0], ":x")
        assert rid.startswith("#")
        assert len(rid) == 13  # "#" + 12 hex chars

    def test_anonymous_hash_is_stable_and_content_addressed(self):
        meta = RuleMetadata(name=None)
        a = rule_identity(meta, ["+", "?x", 0], ":x")
        b = rule_identity(meta, ["+", "?x", 0], ":x")
        c = rule_identity(meta, ["*", "?x", 1], ":x")
        assert a == b
        assert a != c
```

- [ ] **Step 2: Run the failing test (expect ImportError / fail).**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_trace_situated.py::TestSpliceAt rerum/tests/test_trace_situated.py::TestRuleIdentity -v
```

Expected FAIL: `ImportError: cannot import name 'splice_at' from 'rerum.trace'` (and `rule_identity`).

- [ ] **Step 3: Implement the helpers in `rerum/trace.py`.**

Change the import block (line 16-18) to add `hashlib`:

```python
import hashlib
from typing import Any, Callable, Dict, List, Optional

from .rewriter import ExprType
```

Then, immediately after the imports and before `class RewriteStep`, add:

```python
def splice_at(root: ExprType, path: List[int], subtree: ExprType) -> ExprType:
    """Return a copy of ``root`` with the subtree at ``path`` replaced.

    ``path`` is a list of child indices: ``[]`` addresses the root itself,
    ``[1]`` the element at index 1 of a list expression, ``[1, 2]`` the
    element at index 2 of the element at index 1, and so on. Pure: ``root``
    is never mutated and the returned structure shares no mutable nodes on
    the spliced path with ``root``.
    """
    if not path:
        return subtree
    if not isinstance(root, list):
        raise ValueError(f"cannot splice into non-list at path {path}: {root!r}")
    i = path[0]
    if i < 0 or i >= len(root):
        raise IndexError(f"path index {i} out of range for {root!r}")
    new_child = splice_at(root[i], path[1:], subtree)
    return root[:i] + [new_child] + root[i + 1:]


def rule_identity(metadata: Any, pattern: ExprType, skeleton: ExprType) -> str:
    """Stable identity for a rule.

    Returns ``metadata.name`` when set, else ``"#"`` followed by the first 12
    hex chars of the sha1 of the rule's ``(pattern)(skeleton)`` rendering.
    Robust to the post-desugar rule-index churn that makes ``rule_index``
    brittle as an identity.
    """
    name = getattr(metadata, "name", None)
    if name:
        return name
    payload = f"({pattern!r})({skeleton!r})".encode("utf-8")
    return "#" + hashlib.sha1(payload).hexdigest()[:12]
```

In `rerum/__init__.py`, extend the `.trace`-sourced exports. The trace names currently come through `.engine` re-export (`RewriteStep`, `RewriteTrace` at lines 78-79). Add a direct import after the engine import block (after line ~90) and add to `__all__`:

```python
from .trace import rule_identity, splice_at
```

and add `"rule_identity"` and `"splice_at"` to the `__all__` list near `"RewriteStep"` / `"RewriteTrace"`.

- [ ] **Step 4: Run the test (expect PASS).**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_trace_situated.py::TestSpliceAt rerum/tests/test_trace_situated.py::TestRuleIdentity -v
```

Expected PASS: all 8 tests green.

- [ ] **Step 5: Commit.**

```bash
git add rerum/trace.py rerum/__init__.py rerum/tests/test_trace_situated.py
git commit -m "feat(trace): pure splice_at and rule_identity helpers

splice_at(root, path, subtree) replaces a redex subtree at a child-index
path (pure, non-mutating). rule_identity(metadata, pattern, skeleton)
gives a name-or-content-hash stable id robust to post-desugar index churn.
Exported from rerum.trace.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `RewriteStep` new keyword fields, `__slots__`, `to_dict`, endpoint equality

Extends the step model additively. Existing positional construction and existing `to_dict` keys are preserved; new fields default to `None`/`"rule"`. Endpoint equality (`__eq__` against an expression) is added so labeled proof paths can satisfy the existing `prove_equal(trace=True)` assertions in Task 7.

**Files:**
- `rerum/trace.py` (EDIT): `RewriteStep` `__slots__` (line ~29), `__init__` (line ~31-41), `to_dict` (line ~50-58); add `before_redex`/`after_redex` properties, `__eq__`/`__hash__`.
- `rerum/tests/test_trace_situated.py` (EXTEND).

- [ ] **Step 1: Failing test for the new fields and serialization.**

Append to `rerum/tests/test_trace_situated.py`:

```python
class TestRewriteStepFields:
    """RewriteStep gains additive situated fields; legacy construction works."""

    def _meta(self):
        return RuleMetadata(name="add-zero", description="x+0=x",
                            reasoning="additive identity", category="identity")

    def test_legacy_positional_construction_still_works(self):
        step = RewriteStep(0, self._meta(), ["+", "x", 0], "x")
        assert step.before == ["+", "x", 0]
        assert step.after == "x"
        # New fields default sensibly.
        assert step.rule_id is None
        assert step.direction is None
        assert step.bindings is None
        assert step.path is None
        assert step.kind == "rule"
        assert step.guard is None
        assert step.rationale is None

    def test_redex_aliases(self):
        step = RewriteStep(0, self._meta(), ["+", "x", 0], "x")
        assert step.before_redex == step.before == ["+", "x", 0]
        assert step.after_redex == step.after == "x"

    def test_new_fields_round_trip(self):
        step = RewriteStep(
            0, self._meta(), ["+", "x", 0], "x",
            rule_id="add-zero", direction="fwd",
            bindings={"x": "x"}, path=[1],
            kind="rule", guard={"condition": ["true"], "result": True},
            rationale="additive identity",
        )
        assert step.rule_id == "add-zero"
        assert step.direction == "fwd"
        assert step.bindings == {"x": "x"}
        assert step.path == [1]
        assert step.guard == {"condition": ["true"], "result": True}
        assert step.rationale == "additive identity"

    def test_to_dict_keeps_legacy_keys(self):
        step = RewriteStep(0, self._meta(), ["+", "x", 0], "x")
        d = step.to_dict()
        for k in ("rule_index", "rule_name", "description", "before", "after"):
            assert k in d

    def test_to_dict_emits_all_situated_keys(self):
        step = RewriteStep(
            0, self._meta(), ["+", "x", 0], "x",
            rule_id="add-zero", direction="fwd", bindings={"x": "x"},
            path=[1], kind="rule",
            guard={"condition": ["true"], "result": True},
            rationale="additive identity",
        )
        d = step.to_dict()
        for k in ("rule_index", "rule_id", "rule_name", "direction",
                  "description", "kind", "before", "after", "path",
                  "bindings", "guard", "rationale"):
            assert k in d, f"missing key {k}"
        assert json.dumps(d) is not None

    def test_eq_against_expression_compares_after(self):
        # A step equals the expression it produces (its 'after'/redex result).
        step = RewriteStep(0, self._meta(), ["+", "x", 0], "x")
        assert step == "x"
        assert step != ["+", "x", 0]

    def test_eq_against_step_is_identity(self):
        m = self._meta()
        s1 = RewriteStep(0, m, ["+", "x", 0], "x")
        s2 = RewriteStep(0, m, ["+", "x", 0], "x")
        assert s1 == s1
        assert s1 != s2  # distinct objects
```

- [ ] **Step 2: Run the failing test (expect fail).**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_trace_situated.py::TestRewriteStepFields -v
```

Expected FAIL: `TypeError: __init__() got an unexpected keyword argument 'rule_id'` (and missing attributes / keys).

- [ ] **Step 3: Implement the extended `RewriteStep`.**

Replace `__slots__` (line 29) and `__init__` (lines 31-41) in `rerum/trace.py`:

```python
    __slots__ = (
        "rule_index", "metadata", "before", "after",
        "rule_id", "direction", "bindings", "path", "kind", "guard",
        "rationale",
    )

    def __init__(
        self,
        rule_index: int,
        metadata: Any,
        before: ExprType,
        after: ExprType,
        *,
        rule_id: Optional[str] = None,
        direction: Optional[str] = None,
        bindings: Optional[dict] = None,
        path: Optional[List[int]] = None,
        kind: str = "rule",
        guard: Optional[dict] = None,
        rationale: Optional[str] = None,
    ):
        self.rule_index = rule_index
        self.metadata = metadata
        self.before = before
        self.after = after
        self.rule_id = rule_id
        self.direction = direction
        self.bindings = bindings
        self.path = path
        self.kind = kind
        self.guard = guard
        self.rationale = rationale

    @property
    def before_redex(self) -> ExprType:
        """Alias of ``before``: the redex-local subtree before the edit."""
        return self.before

    @property
    def after_redex(self) -> ExprType:
        """Alias of ``after``: the redex-local subtree after the edit."""
        return self.after

    def __eq__(self, other: Any) -> bool:
        """Identity for step-vs-step; endpoint match for step-vs-expression.

        Comparing a step against another ``RewriteStep`` is object identity
        (steps are not value-equal). Comparing against any other operand
        (an expression: list/str/number) tests ``self.after == other`` so a
        reconstructed proof path element equals the node expression it
        represents. This keeps legacy assertions like ``path_a[0] == start``
        working when paths are lists of steps.
        """
        if isinstance(other, RewriteStep):
            return self is other
        return self.after == other

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return id(self)
```

Update `to_dict` (lines 50-58) to emit all fields additively (legacy keys preserved):

```python
    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary for serialization.

        Emits the legacy keys (rule_index, rule_name, description, before,
        after) plus the situated keys (rule_id, direction, kind, path,
        bindings, guard, rationale). ``rule_id`` falls back to
        ``rule_identity`` of the metadata's pattern/skeleton when not
        pre-populated and the metadata exposes them; otherwise it is the
        plain ``rule_id`` field (possibly None).
        """
        return {
            "rule_index": self.rule_index,
            "rule_id": self.rule_id,
            "rule_name": self.metadata.name,
            "direction": self.direction,
            "description": self.metadata.description,
            "kind": self.kind,
            "before": self.before,
            "after": self.after,
            "path": self.path,
            "bindings": self.bindings,
            "guard": self.guard,
            "rationale": self.rationale,
        }
```

Note: `_name()` and `__repr__` keep referencing `self.metadata.name`/`self.rule_index`, which still exist, so they are unchanged.

- [ ] **Step 4: Run the test plus the legacy trace suite (expect PASS).**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_trace_situated.py::TestRewriteStepFields rerum/tests/test_trace.py -v
```

Expected PASS: new field tests pass; all existing `test_trace.py` tests still pass (legacy `to_dict` keys preserved).

- [ ] **Step 5: Commit.**

```bash
git add rerum/trace.py rerum/tests/test_trace_situated.py
git commit -m "feat(trace): additive situated fields on RewriteStep

rule_id, direction, bindings, path, kind, guard, rationale added as
keyword-only fields (defaults preserve legacy positional construction).
before_redex/after_redex alias before/after. to_dict emits all fields
while keeping legacy keys. __eq__ compares step-vs-expression by 'after'
(endpoint match) and step-vs-step by identity; __hash__ kept.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `RewriteTrace.to_global_sequence` and `to_dict(global_sequence=...)`

Reconstructs the whole-expression derivation by replaying redex-local edits at their paths, starting from `self.initial`. Lossless and stored on demand rather than per-step.

**Files:**
- `rerum/trace.py` (EDIT): add `to_global_sequence` method on `RewriteTrace`; extend `to_dict` signature (line ~143).
- `rerum/tests/test_trace_situated.py` (EXTEND).

- [ ] **Step 1: Failing test for global reconstruction.**

Append to `rerum/tests/test_trace_situated.py`:

```python
class TestGlobalSequence:
    """to_global_sequence replays redex edits at their paths from initial."""

    def _trace(self):
        # Build a trace by hand: two steps editing different redexes of a root.
        meta = RuleMetadata(name="add-zero")
        t = RewriteTrace()
        t.initial = ["+", ["+", "x", 0], 0]
        # Step 1: inner (+ x 0) at path [1] -> x.  Root becomes (+ x 0).
        t(RewriteStep(0, meta, ["+", "x", 0], "x", path=[1]))
        # Step 2: outer (+ x 0) at path [] -> x.  Root becomes x.
        t(RewriteStep(0, meta, ["+", "x", 0], "x", path=[]))
        t.final = "x"
        return t

    def test_global_sequence_roots(self):
        t = self._trace()
        seq = t.to_global_sequence()
        assert len(seq) == 2
        assert seq[0]["before_root"] == ["+", ["+", "x", 0], 0]
        assert seq[0]["after_root"] == ["+", "x", 0]
        assert seq[1]["before_root"] == ["+", "x", 0]
        assert seq[1]["after_root"] == "x"

    def test_global_sequence_carries_step(self):
        t = self._trace()
        seq = t.to_global_sequence()
        assert seq[0]["step"] is t.steps[0]
        assert seq[1]["step"] is t.steps[1]

    def test_global_sequence_final_matches(self):
        t = self._trace()
        seq = t.to_global_sequence()
        assert seq[-1]["after_root"] == t.final

    def test_to_dict_global_sequence_flag(self):
        t = self._trace()
        d_plain = t.to_dict()
        assert "global_sequence" not in d_plain
        d_glob = t.to_dict(global_sequence=True)
        assert "global_sequence" in d_glob
        assert len(d_glob["global_sequence"]) == 2
        # Each global entry is JSON-serializable (step rendered via to_dict).
        assert json.dumps(d_glob) is not None

    def test_to_dict_keeps_legacy_keys(self):
        t = self._trace()
        d = t.to_dict()
        for k in ("initial", "final", "steps", "step_count"):
            assert k in d
```

- [ ] **Step 2: Run the failing test (expect fail).**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_trace_situated.py::TestGlobalSequence -v
```

Expected FAIL: `AttributeError: 'RewriteTrace' object has no attribute 'to_global_sequence'` and `to_dict()` rejecting `global_sequence=`.

- [ ] **Step 3: Implement `to_global_sequence` and extend `to_dict`.**

In `rerum/trace.py`, add a method on `RewriteTrace` (place it just before the existing `to_dict`, around line 142):

```python
    def to_global_sequence(self) -> List[Dict[str, Any]]:
        """Replay the trace from ``self.initial`` as whole-expression states.

        Each entry is ``{"before_root", "after_root", "step"}``. The running
        root starts at ``self.initial``; for each step, ``after_root`` is the
        running root with ``step.after`` (the redex result) spliced in at
        ``step.path`` (``[]`` when a step predates path threading, meaning a
        root-level edit). The new ``after_root`` becomes the next
        ``before_root``. Lossless: the redex-local edits plus paths fully
        determine the global derivation, so it need not be stored per step.
        """
        sequence: List[Dict[str, Any]] = []
        root = self.initial
        for step in self.steps:
            path = step.path if step.path is not None else []
            before_root = root
            after_root = splice_at(root, path, step.after)
            sequence.append({
                "before_root": before_root,
                "after_root": after_root,
                "step": step,
            })
            root = after_root
        return sequence
```

Replace the existing `to_dict` (lines 143-150) with the `global_sequence`-aware form:

```python
    def to_dict(self, global_sequence: bool = False) -> Dict[str, Any]:
        """Convert trace to JSON-serializable dictionary.

        When ``global_sequence`` is True, include a ``global_sequence`` key:
        the whole-expression replay from ``to_global_sequence()`` with each
        step rendered via ``RewriteStep.to_dict()``.
        """
        d: Dict[str, Any] = {
            "initial": self.initial,
            "final": self.final,
            "steps": [step.to_dict() for step in self.steps],
            "step_count": len(self.steps),
        }
        if global_sequence:
            d["global_sequence"] = [
                {
                    "before_root": entry["before_root"],
                    "after_root": entry["after_root"],
                    "step": entry["step"].to_dict(),
                }
                for entry in self.to_global_sequence()
            ]
        return d
```

- [ ] **Step 4: Run the test plus legacy trace serialization (expect PASS).**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_trace_situated.py::TestGlobalSequence rerum/tests/test_trace.py::TestTraceToDict -v
```

Expected PASS: global-sequence tests pass; `TestTraceToDict` (legacy keys, serializability) still passes.

- [ ] **Step 5: Commit.**

```bash
git add rerum/trace.py rerum/tests/test_trace_situated.py
git commit -m "feat(trace): RewriteTrace.to_global_sequence and global_sequence to_dict

Replay redex-local edits at their paths from self.initial to reconstruct
the whole-expression derivation on demand. to_dict(global_sequence=True)
embeds it; default keys unchanged.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Path threading through the strategy drivers

Threads an accumulating redex `path` (parent path plus child index) through `_simplify_exhaustive`, `_bottomup_pass`, and `_topdown_pass`, stamps it on each emitted `RewriteStep`, and populates `HookContext.expr_path` from the same source. This task only sets `path` (other situated fields are populated in Task 5) so it is independently verifiable.

**Files:**
- `rerum/engine.py` (EDIT): `_simplify_exhaustive` (line ~2497, recursion at ~2606), `_bottomup_pass` (line ~2650, recursion at ~2664), `_topdown_pass` (line ~2755, recursion at ~2828), and `_fire_rule_applied` (line ~2208, `expr_path=[]` at ~2230). The emit sites are at lines ~2550, ~2692, ~2791.
- `rerum/tests/test_trace_situated.py` (EXTEND).

- [ ] **Step 1: Failing test asserting step paths under each strategy.**

Append to `rerum/tests/test_trace_situated.py`:

```python
class TestPathThreading:
    """Emitted steps carry the redex path under each strategy."""

    def _engine(self):
        return RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")

    def test_root_redex_has_empty_path(self):
        eng = self._engine()
        _, trace = eng.simplify(E("(+ x 0)"), trace=True)
        assert trace.steps, "expected at least one step"
        assert trace.steps[0].path == []

    def test_child_redex_carries_path_exhaustive(self):
        eng = self._engine()
        # (* (+ x 0) y): the redex (+ x 0) is at child index 1 of the root.
        _, trace = eng.simplify(E("(* (+ x 0) y)"), trace=True,
                                strategy="exhaustive")
        paths = [s.path for s in trace.steps]
        assert [1] in paths, f"expected redex path [1] among {paths}"

    def test_child_redex_carries_path_bottomup(self):
        eng = self._engine()
        _, trace = eng.simplify(E("(* (+ x 0) y)"), trace=True,
                                strategy="bottomup")
        paths = [s.path for s in trace.steps]
        assert [1] in paths, f"expected redex path [1] among {paths}"

    def test_child_redex_carries_path_topdown(self):
        eng = self._engine()
        _, trace = eng.simplify(E("(* (+ x 0) y)"), trace=True,
                                strategy="topdown")
        paths = [s.path for s in trace.steps]
        assert [1] in paths, f"expected redex path [1] among {paths}"

    def test_global_sequence_roundtrips_after_threading(self):
        eng = self._engine()
        result, trace = eng.simplify(E("(* (+ x 0) y)"), trace=True)
        seq = trace.to_global_sequence()
        # The reconstructed final root equals the engine's final result.
        assert seq[-1]["after_root"] == result
        assert seq[0]["before_root"] == trace.initial

    def test_hook_context_expr_path_populated(self):
        eng = self._engine()
        seen = []

        def observer(step, ctx):
            seen.append(tuple(ctx.expr_path))

        eng.on_rule_applied(observer)
        eng.simplify(E("(* (+ x 0) y)"))
        # The child redex fired with a non-empty expr_path matching its path.
        assert (1,) in seen, f"expected (1,) among {seen}"
```

- [ ] **Step 2: Run the failing test (expect fail).**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_trace_situated.py::TestPathThreading -v
```

Expected FAIL: `step.path` is `None` (paths never stamped); `ctx.expr_path` always empty so `(1,) not in seen`.

- [ ] **Step 3: Thread `path` through the drivers.**

`_fire_rule_applied` (line 2208): add an `expr_path` parameter and forward it to the `HookContext`:

```python
    def _fire_rule_applied(self, step: RewriteStep, *, depth: int = 0,
                           expr_path: Optional[List[int]] = None) -> bool:
```

and change the `HookContext` construction (line 2228-2234) so `expr_path` uses the threaded value:

```python
        ctx = HookContext(
            engine=self,
            expr_path=list(expr_path) if expr_path is not None else [],
            depth=depth,
            step_count=self._step_count,
            event_name="rule_applied",
        )
```

`_simplify_exhaustive` (line 2497): add a `path` keyword that defaults to the root path, stamp it on the emitted step, pass it to `_fire_rule_applied`, and extend it on the child recursion. Change the signature:

```python
    def _simplify_exhaustive(self, expr: ExprType, max_steps: int,
                              groups: Optional[List[str]] = None,
                              _top_level: bool = True,
                              path: Optional[List[int]] = None) -> ExprType:
```

At the top of the method body, normalize `path`:

```python
        if path is None:
            path = []
```

At the emit site (lines 2550-2557), stamp `path` and forward it:

```python
                    if new_expr != current:
                        step = RewriteStep(
                            rule_index=rule_idx,
                            metadata=metadata,
                            before=current,
                            after=new_expr,
                            path=list(path),
                        )
                        if self._fire_rule_applied(step, expr_path=path):
                            return new_expr  # Hook requested cancellation after this step.
                        current = new_expr
                        changed = True
                        break
```

At the child recursion (lines 2605-2608), extend the path with the child index:

```python
                    for idx, child in enumerate(current):
                        new_child = self._simplify_exhaustive(
                            child, max_steps // 10 or 1, groups=groups,
                            _top_level=False, path=path + [idx],
                        )
                        new_children.append(new_child)
                        if new_child != child:
                            subexpr_changed = True
```

(The existing `for child in current:` at line 2605 becomes `for idx, child in enumerate(current):`.)

`_bottomup_pass` (line 2650): the signature already carries `depth`; add `path`:

```python
    def _bottomup_pass(self, expr: ExprType, groups: Optional[List[str]] = None,
                        depth: int = 0, path: Optional[List[int]] = None) -> ExprType:
```

Normalize at the top (after the docstring, before the base-case check):

```python
        if path is None:
            path = []
```

Children are simplified before the parent (line 2664); thread the child index:

```python
        new_children = [
            self._bottomup_pass(child, groups=groups, depth=depth + 1,
                                path=path + [i])
            for i, child in enumerate(expr)
        ]
```

At the emit site (lines 2692-2699):

```python
                if result != current:
                    step = RewriteStep(
                        rule_index=rule_idx,
                        metadata=metadata,
                        before=current,
                        after=result,
                        path=list(path),
                    )
                    self._fire_rule_applied(step, depth=depth, expr_path=path)
                    return result
```

`_topdown_pass` (line 2755): add `path`:

```python
    def _topdown_pass(self, expr: ExprType, groups: Optional[List[str]] = None,
                      depth: int = 0, path: Optional[List[int]] = None) -> ExprType:
```

Normalize at the top of the method body (before `current = expr`):

```python
        if path is None:
            path = []
```

At the emit site (lines 2791-2798):

```python
                if result != current:
                    step = RewriteStep(
                        rule_index=rule_idx,
                        metadata=metadata,
                        before=current,
                        after=result,
                        path=list(path),
                    )
                    self._fire_rule_applied(step, depth=depth, expr_path=path)
                    return result  # Return immediately - will be called again
```

At the child recursion (line 2828), thread the child index:

```python
        if isinstance(current, list) and len(current) > 0:
            new_children = [
                self._topdown_pass(child, groups=groups, depth=depth + 1,
                                   path=path + [i])
                for i, child in enumerate(current)
            ]
            if new_children != list(current):
                return new_children
```

- [ ] **Step 4: Run the test plus all strategy and hook tests (expect PASS).**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_trace_situated.py::TestPathThreading rerum/tests/test_strategies.py rerum/tests/test_trace.py -v
```

Expected PASS: path threading tests pass; existing strategy and trace tests unaffected.

- [ ] **Step 5: Commit.**

```bash
git add rerum/engine.py rerum/tests/test_trace_situated.py
git commit -m "feat(engine): thread redex path through strategy drivers

_simplify_exhaustive, _bottomup_pass, _topdown_pass carry an accumulating
path (parent path + child index) and stamp it on each emitted RewriteStep.
HookContext.expr_path is populated from the same source via
_fire_rule_applied(expr_path=...). to_global_sequence now round-trips.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Populate the remaining situated fields at emit sites

Fills `rule_id`, `direction`, `bindings`, `kind`, `guard`, and `rationale` at the four `RewriteStep` construction sites (`apply_once` and the three pass emit sites). Path was set in Task 4; this task adds the rest.

**Files:**
- `rerum/engine.py` (EDIT): `apply_once` emit (line ~2066), `_simplify_exhaustive` emit (line ~2550), `_bottomup_pass` emit (line ~2692), `_topdown_pass` emit (line ~2791). A small private helper `_build_step(...)` centralizes field derivation.
- `rerum/tests/test_trace_situated.py` (EXTEND).

- [ ] **Step 1: Failing test asserting populated fields.**

Append to `rerum/tests/test_trace_situated.py`:

```python
class TestPopulatedFields:
    """Situated fields are populated at the emit sites during simplify."""

    def test_named_rule_id_and_rationale(self):
        eng = RuleEngine.from_dsl(
            '@add-zero {category=identity}: (+ ?x 0) => :x'
        )
        _, trace = eng.simplify(E("(+ x 0)"), trace=True)
        step = trace.steps[0]
        assert step.rule_id == "add-zero"
        assert step.kind == "rule"
        # rationale comes from reasoning or category.
        assert step.rationale == "identity"

    def test_bindings_captured(self):
        eng = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")
        _, trace = eng.simplify(E("(+ y 0)"), trace=True)
        step = trace.steps[0]
        assert step.bindings is not None
        assert step.bindings.get("x") == "y"

    def test_direction_for_bidirectional(self):
        eng = RuleEngine.from_dsl("@commute: (+ ?x ?y) <=> (+ :y :x)")
        # apply_once gives us the matched metadata; direction is fwd or rev.
        _, trace = eng.simplify(E("(+ a b)"), trace=True, max_steps=1)
        if trace.steps:
            assert trace.steps[0].direction in ("fwd", "rev")

    def test_guard_recorded_when_condition_checked(self):
        eng = RuleEngine.from_dsl(
            "@pos: (abs ?x) => :x when (! > :x 0)",
            fold_funcs=None,
        )
        # Use a prelude that defines '>'; here assert the field shape only
        # when a guarded rule fires.
        # (Guard population is asserted structurally: presence of dict.)
        # Fall back: a rule without a guard records guard=None.
        eng2 = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")
        _, trace2 = eng2.simplify(E("(+ x 0)"), trace=True)
        assert trace2.steps[0].guard is None

    def test_apply_once_populates_fields(self):
        eng = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")
        captured = []
        eng.on_rule_applied(lambda step, ctx: captured.append(step))
        result, meta = eng.apply_once(E("(+ x 0)"))
        assert result == "x"
        assert captured
        assert captured[0].rule_id == "add-zero"
        assert captured[0].bindings is not None
```

For the guarded-rule guard-dict assertion, add a focused test using a prelude that supplies `>`:

```python
class TestGuardField:
    """A checked condition is recorded in step.guard."""

    def test_guard_dict_present(self):
        from rerum.rewriter import PREDICATE_PRELUDE
        eng = RuleEngine.from_dsl(
            "@drop-abs: (abs ?x) => :x when (! >= :x 0)",
            fold_funcs=PREDICATE_PRELUDE,
        )
        _, trace = eng.simplify(E("(abs 5)"), trace=True)
        if trace.steps:
            g = trace.steps[0].guard
            assert g is not None
            assert g["result"] is True
            assert "condition" in g
```

- [ ] **Step 2: Run the failing test (expect fail).**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_trace_situated.py::TestPopulatedFields rerum/tests/test_trace_situated.py::TestGuardField -v
```

Expected FAIL: `rule_id`, `bindings`, `direction`, `rationale`, and `guard` are `None` at emit sites.

- [ ] **Step 3: Centralize field derivation and populate emit sites.**

Add a private helper on `RuleEngine` near the strategy drivers (place it just above `_simplify_exhaustive`, around line 2496). It derives the situated fields from the rule, the match bindings, and the condition result:

```python
    def _build_step(self, rule_idx, rule, metadata, before, after, bindings,
                    *, path=None, guard=None):
        """Construct a fully-populated situated RewriteStep.

        Derives rule_id (rule_identity over pattern/skeleton), direction
        (metadata.direction for bidirectional halves), serialized bindings,
        and rationale (reasoning, else category). ``guard`` is a
        {"condition", "result"} dict when a condition was evaluated, else
        None. ``path`` is the redex child-index path (defaults to []).
        """
        from .trace import rule_identity
        pattern, skeleton = rule
        return RewriteStep(
            rule_index=rule_idx,
            metadata=metadata,
            before=before,
            after=after,
            rule_id=rule_identity(metadata, pattern, skeleton),
            direction=metadata.direction,
            bindings=bindings.to_dict() if bindings is not None else None,
            path=list(path) if path is not None else [],
            kind="rule",
            guard=guard,
            rationale=metadata.reasoning or metadata.category,
        )
```

Add a small condition-evaluation variant that returns both the truthiness and the instantiated condition so emit sites can build the guard dict without double-instantiating. Extend `_check_condition` with an internal sibling (add just after `_check_condition`, line ~1926):

```python
    def _evaluate_guard(self, condition, bindings):
        """Return a guard dict for a checked condition, or None.

        ``{"condition": <instantiated condition expr>, "result": <bool>}``
        when ``condition`` is not None, else None. Reuses the same
        instantiation/truthiness path as ``_check_condition``.
        """
        if condition is None:
            return None
        instantiated = instantiate(
            condition, bindings, self._fold_funcs,
            undefined_op_resolver=self._undefined_op_resolver,
            fold_error_resolver=self._fold_error_resolver,
        )
        return {"condition": instantiated, "result": _condition_truthy(instantiated)}
```

Now update the four emit sites to use `_build_step` and capture the guard.

`apply_once` (lines 2066-2071). The bindings are already in scope as `bindings`; build the guard from `metadata.condition`:

```python
                if result != expr:
                    guard = self._evaluate_guard(metadata.condition, bindings)
                    step = self._build_step(
                        rule_idx, rule, metadata, expr, result, bindings,
                        guard=guard,
                    )
                    self._fire_rule_applied(step)
                return result, metadata
```

`_simplify_exhaustive` emit (lines 2550-2557). `bindings` and `path` are in scope:

```python
                    if new_expr != current:
                        guard = self._evaluate_guard(metadata.condition, bindings)
                        step = self._build_step(
                            rule_idx, rule, metadata, current, new_expr,
                            bindings, path=path, guard=guard,
                        )
                        if self._fire_rule_applied(step, expr_path=path):
                            return new_expr
                        current = new_expr
                        changed = True
                        break
```

`_bottomup_pass` emit (lines 2692-2699):

```python
                if result != current:
                    guard = self._evaluate_guard(metadata.condition, bindings)
                    step = self._build_step(
                        rule_idx, rule, metadata, current, result, bindings,
                        path=path, guard=guard,
                    )
                    self._fire_rule_applied(step, depth=depth, expr_path=path)
                    return result
```

`_topdown_pass` emit (lines 2791-2798):

```python
                if result != current:
                    guard = self._evaluate_guard(metadata.condition, bindings)
                    step = self._build_step(
                        rule_idx, rule, metadata, current, result, bindings,
                        path=path, guard=guard,
                    )
                    self._fire_rule_applied(step, depth=depth, expr_path=path)
                    return result  # Return immediately - will be called again
```

- [ ] **Step 4: Run the test plus the full trace and hooks suites (expect PASS).**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_trace_situated.py rerum/tests/test_trace.py rerum/tests/test_guards.py -v
```

Expected PASS: populated-field tests pass; existing trace and guard tests unaffected (guard evaluation is read-only and matches `_check_condition`).

- [ ] **Step 5: Commit.**

```bash
git add rerum/engine.py rerum/tests/test_trace_situated.py
git commit -m "feat(engine): populate situated step fields at emit sites

_build_step centralizes rule_id (rule_identity), direction, serialized
bindings, kind, and rationale (reasoning|category). _evaluate_guard records
{condition, result} when a rule condition is checked. Wired into apply_once
and the three pass emit sites.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Labeled `_all_single_rewrites`

Makes `_all_single_rewrites` return labeled edges (neighbor expression plus the `rule_id`/`direction`/`bindings`/`path` that produced it) while keeping a backward-compatible expression-only view for current callers.

**Files:**
- `rerum/engine.py` (EDIT): `_all_single_rewrites` (line ~3181). Callers at lines ~3331 (`equivalents`), ~3496/3536 (`prove_equal`), ~3734/3844 (random walk paths) consume the plain list; keep them working by adding a `labeled` keyword that defaults to the legacy shape.
- `rerum/tests/test_trace_situated.py` (EXTEND).

- [ ] **Step 1: Failing test for labeled edges.**

Append to `rerum/tests/test_trace_situated.py`:

```python
class TestLabeledSingleRewrites:
    """_all_single_rewrites(labeled=True) returns (expr, label) edges."""

    def test_default_is_legacy_expr_list(self):
        eng = RuleEngine.from_dsl("@commute: (+ ?x ?y) <=> (+ :y :x)")
        outs = eng._all_single_rewrites(["+", "a", "b"])
        # Legacy shape: a list of plain expressions.
        assert ["+", "b", "a"] in outs

    def test_labeled_returns_edges(self):
        eng = RuleEngine.from_dsl("@commute: (+ ?x ?y) <=> (+ :y :x)")
        edges = eng._all_single_rewrites(["+", "a", "b"], labeled=True)
        assert edges, "expected at least one labeled edge"
        expr, label = edges[0]
        assert "rule_id" in label
        assert "direction" in label
        assert "bindings" in label
        assert "path" in label

    def test_labeled_edge_records_path_for_child_redex(self):
        eng = RuleEngine.from_dsl("@commute: (+ ?x ?y) <=> (+ :y :x)")
        edges = eng._all_single_rewrites(["*", ["+", "a", "b"], "c"],
                                         labeled=True)
        # The (+ a b) -> (+ b a) edge is at child path [1].
        target = ["*", ["+", "b", "a"], "c"]
        match = [lbl for ex, lbl in edges if ex == target]
        assert match, f"missing edge to {target}"
        assert match[0]["path"] == [1]

    def test_labeled_rule_id_is_named(self):
        eng = RuleEngine.from_dsl("@commute: (+ ?x ?y) <=> (+ :y :x)")
        edges = eng._all_single_rewrites(["+", "a", "b"], labeled=True)
        # commute desugars to commute-fwd / commute-rev; rule_id is the name.
        rule_ids = {lbl["rule_id"] for _, lbl in edges}
        assert any(rid.startswith("commute") for rid in rule_ids)
```

- [ ] **Step 2: Run the failing test (expect fail).**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_trace_situated.py::TestLabeledSingleRewrites -v
```

Expected FAIL: `_all_single_rewrites() got an unexpected keyword argument 'labeled'`.

- [ ] **Step 3: Implement labeled edges.**

Replace `_all_single_rewrites` (lines 3181-3239) with a `labeled`-aware, path-carrying version. When `labeled=False` (default), it returns the legacy `List[ExprType]`; when `labeled=True`, it returns `List[Tuple[ExprType, dict]]` where the dict is `{"rule_id", "direction", "bindings", "path"}`:

```python
    def _all_single_rewrites(
        self,
        expr: ExprType,
        bidirectional_only: bool = True,
        groups: Optional[List[str]] = None,
        rules: Optional[RuleSet] = None,
        labeled: bool = False,
        _path: Optional[List[int]] = None,
    ):
        """Find all expressions reachable by applying exactly one rule.

        Tries every rule at every position in the expression tree.

        When ``labeled`` is False (default), returns a list of distinct
        one-step rewrite expressions (legacy shape). When ``labeled`` is
        True, returns a list of ``(new_expr, label)`` edges where ``label``
        is ``{"rule_id", "direction", "bindings", "path"}`` describing the
        rule application that produced ``new_expr``.
        """
        from .trace import rule_identity
        if rules is None:
            rules = self.rule_set(groups=groups, bidirectional_only=bidirectional_only)
        if _path is None:
            _path = []

        results = []
        seen: Set[tuple] = set()

        def add_if_new(new_expr: ExprType, label: Optional[dict]) -> None:
            key = _expr_to_tuple(new_expr)
            if key not in seen:
                seen.add(key)
                if labeled:
                    results.append((new_expr, label))
                else:
                    results.append(new_expr)

        # Try rules at top level
        for rule_idx, rule, metadata in rules:
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
                    label = {
                        "rule_id": rule_identity(metadata, pattern, skeleton),
                        "direction": metadata.direction,
                        "bindings": bindings.to_dict(),
                        "path": list(_path),
                    } if labeled else None
                    add_if_new(result, label)

        # Recursively try rules in subexpressions
        if isinstance(expr, list) and len(expr) > 0:
            for i, child in enumerate(expr):
                child_rewrites = self._all_single_rewrites(
                    child, rules=rules, labeled=labeled, _path=_path + [i]
                )
                if labeled:
                    for new_child, label in child_rewrites:
                        new_expr = expr[:i] + [new_child] + expr[i+1:]
                        add_if_new(new_expr, label)
                else:
                    for new_child in child_rewrites:
                        new_expr = expr[:i] + [new_child] + expr[i+1:]
                        add_if_new(new_expr, None)

        return results
```

Existing callers (`equivalents` line ~3331, `prove_equal` lines ~3496/3536, random-walk lines ~3734/3844) call without `labeled`, so they keep the legacy list shape unchanged.

- [ ] **Step 4: Run the test plus equivalence/minimize suites (expect PASS).**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_trace_situated.py::TestLabeledSingleRewrites rerum/tests/test_equivalents.py rerum/tests/test_optimization.py rerum/tests/test_prove_equal.py -v
```

Expected PASS: labeled-edge tests pass; existing equivalence, optimization, and proof tests unaffected (default shape preserved).

- [ ] **Step 5: Commit.**

```bash
git add rerum/engine.py rerum/tests/test_trace_situated.py
git commit -m "feat(engine): labeled edges from _all_single_rewrites

Add labeled=True to return (expr, {rule_id, direction, bindings, path})
edges with an accumulating redex path; default keeps the legacy
expression-only list so equivalents/prove_equal/minimize are unchanged.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Labeled `prove_equal` paths and `EqualityProof` step paths

Carries the producing edge label on `prove_equal`'s visited parent pointers so `reconstruct_path` returns labeled `RewriteStep`s. `EqualityProof.path_a`/`path_b` become lists of `RewriteStep` (for the `trace=True` path), while directly-constructed plain-expression paths and the existing assertions keep working via `RewriteStep.__eq__` (Task 2) and `EqualityProof` rendering that tolerates both shapes.

**Files:**
- `rerum/engine.py` (EDIT): `prove_equal` (line ~3375): `visited_a`/`visited_b` tuple gains a label slot; `reconstruct_path` (line ~3461) builds `RewriteStep`s; the two intersection branches (lines ~3505-3523, ~3545-3563) pass labeled paths. `EqualityProof.format`/`to_dict` (lines ~1265-1316) tolerate `RewriteStep` elements.
- `rerum/tests/test_prove_equal.py` (EXTEND): new class `TestProveEqualLabeledPaths`.

- [ ] **Step 1: Failing test for labeled proof paths.**

Append to `rerum/tests/test_prove_equal.py`:

```python
from rerum import RewriteStep, E


class TestProveEqualLabeledPaths:
    """prove_equal(trace=True) returns RewriteStep-labeled paths."""

    def _engine(self):
        return RuleEngine.from_dsl("@commute: (+ ?x ?y) <=> (+ :y :x)")

    def test_path_elements_are_steps(self):
        eng = self._engine()
        proof = eng.prove_equal(["+", "a", "b"], ["+", "b", "a"], trace=True)
        assert proof is not None
        # Non-trivial path elements are RewriteStep (the synthetic-initial
        # step plus edge steps).
        all_steps = (proof.path_a or []) + (proof.path_b or [])
        assert any(isinstance(s, RewriteStep) for s in all_steps)

    def test_steps_carry_rule_ids(self):
        eng = self._engine()
        proof = eng.prove_equal(["+", "a", "b"], ["+", "b", "a"], trace=True)
        ids = [s.rule_id for s in (proof.path_a or []) if isinstance(s, RewriteStep)]
        ids += [s.rule_id for s in (proof.path_b or []) if isinstance(s, RewriteStep)]
        # At least one edge step names the commute rule.
        assert any(rid and rid.startswith("commute") for rid in ids)

    def test_to_dict_emits_step_dicts(self):
        eng = self._engine()
        proof = eng.prove_equal(["+", "a", "b"], ["+", "b", "a"], trace=True)
        d = proof.to_dict()
        import json
        assert json.dumps(d) is not None
        # Step-shaped path entries serialize with rule_id keys.
        if "path_a" in d:
            for entry in d["path_a"]:
                if isinstance(entry, dict):
                    assert "rule_id" in entry
```

- [ ] **Step 2: Run the failing test (expect fail).**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_prove_equal.py::TestProveEqualLabeledPaths -v
```

Expected FAIL: `path_a` elements are plain expressions, not `RewriteStep`; no `rule_id` attribute.

- [ ] **Step 3: Carry labels and reconstruct labeled steps.**

In `prove_equal` (line ~3375), the visited maps currently store `(expr, depth, parent_key)`. Extend them to `(expr, depth, parent_key, label)` where `label` is the producing edge dict (`None` for the start node). Update the two dict initializers (lines 3444-3449):

```python
        visited_a: Dict[tuple, Tuple[ExprType, int, Optional[tuple], Optional[dict]]] = {
            key_a: (expr_a, 0, None, None)
        }
        visited_b: Dict[tuple, Tuple[ExprType, int, Optional[tuple], Optional[dict]]] = {
            key_b: (expr_b, 0, None, None)
        }
```

Replace `reconstruct_path` (lines 3461-3473) so it returns `List[RewriteStep]`, with a synthetic initial step at the start and one labeled step per edge:

```python
        from .trace import RewriteStep

        def reconstruct_path(
            visited: Dict[tuple, Tuple[ExprType, int, Optional[tuple], Optional[dict]]],
            target_key: tuple
        ) -> List[RewriteStep]:
            """Reconstruct the labeled path from start to target as steps.

            The first element is a synthetic initial step (rule_index=-1,
            kind="initial") whose before/after both equal the start
            expression; each subsequent element is a step whose ``after`` is
            the node expression and whose label fields come from the edge
            that produced it. Step ``__eq__`` against an expression compares
            ``after``, so ``path[0] == start`` and ``path[-1] == target``.
            """
            chain = []  # list of (expr, label) from start..target
            current_key = target_key
            while current_key is not None:
                expr, depth, parent_key, label = visited[current_key]
                chain.append((expr, label))
                current_key = parent_key
            chain.reverse()

            steps: List[RewriteStep] = []
            start_expr = chain[0][0]
            steps.append(RewriteStep(
                rule_index=-1, metadata=RuleMetadata(name=None),
                before=start_expr, after=start_expr, kind="initial",
            ))
            for expr, label in chain[1:]:
                if label is None:
                    steps.append(RewriteStep(
                        rule_index=-1, metadata=RuleMetadata(name=None),
                        before=expr, after=expr, kind="initial",
                    ))
                else:
                    steps.append(RewriteStep(
                        rule_index=-1,
                        metadata=RuleMetadata(name=label.get("rule_id")),
                        before=expr, after=expr,
                        rule_id=label.get("rule_id"),
                        direction=label.get("direction"),
                        bindings=label.get("bindings"),
                        path=label.get("path"),
                        kind="rule",
                    ))
            return steps
```

The identical-expression early return (lines 3430-3440) must produce single-element step paths so `len(path_a) == 1` (test line 292) holds. Replace it:

```python
        if key_a == key_b:
            if trace:
                from .trace import RewriteStep
                init = [RewriteStep(
                    rule_index=-1, metadata=RuleMetadata(name=None),
                    before=expr_a, after=expr_a, kind="initial",
                )]
            else:
                init = None
            return EqualityProof(
                expr_a=expr_a, expr_b=expr_b, common=expr_a,
                depth_a=0, depth_b=0,
                path_a=init, path_b=init,
            )
```

Both expansion frontiers must store the edge label. Switch the A-side expansion (lines 3494-3503) to labeled edges:

```python
                if depth < max_depth_a:
                    current_key = _expr_to_tuple(current)
                    edges = self._all_single_rewrites(
                        current, bidirectional_only, groups, rules=rules,
                        labeled=True,
                    )
                    for new_expr, label in edges:
                        new_key = _expr_to_tuple(new_expr)
                        if new_key not in visited_a:
                            visited_a[new_key] = (new_expr, depth + 1, current_key, label)
                            frontier_a.append((new_expr, depth + 1))
                            if new_key in visited_b:
                                _, depth_b, _, _ = visited_b[new_key]
                                if trace:
                                    path_a = reconstruct_path(visited_a, new_key)
                                    path_b = reconstruct_path(visited_b, new_key)
                                else:
                                    path_a = None
                                    path_b = None
                                return EqualityProof(
                                    expr_a=expr_a, expr_b=expr_b,
                                    common=new_expr,
                                    depth_a=depth + 1, depth_b=depth_b,
                                    path_a=path_a, path_b=path_b,
                                )
```

Apply the symmetric change to the B-side expansion (lines 3534-3563), unpacking the 4-tuple (`_, depth_a_val, _, _ = visited_a[new_key]`) and using `labeled=True`.

`EqualityProof.format` (lines 1290-1298) iterates path elements with `format_sexpr(expr)`. Make it tolerate `RewriteStep`:

```python
            if self.path_a:
                lines.append(f"\nPath from A ({self.depth_a} steps):")
                for i, item in enumerate(self.path_a):
                    rendered = (format_sexpr(item.after)
                                if isinstance(item, RewriteStep) else format_sexpr(item))
                    lines.append(f"  {i}. {rendered}")

            if self.path_b:
                lines.append(f"\nPath from B ({self.depth_b} steps):")
                for i, item in enumerate(self.path_b):
                    rendered = (format_sexpr(item.after)
                                if isinstance(item, RewriteStep) else format_sexpr(item))
                    lines.append(f"  {i}. {rendered}")
```

`EqualityProof.to_dict` (lines 1312-1315) must serialize step paths as step dicts while leaving plain-expression paths as-is (so `test_to_dict_with_paths` at line 116 still emits `["a", "c"]`):

```python
        if self.path_a:
            result["path_a"] = [
                item.to_dict() if isinstance(item, RewriteStep) else item
                for item in self.path_a
            ]
        if self.path_b:
            result["path_b"] = [
                item.to_dict() if isinstance(item, RewriteStep) else item
                for item in self.path_b
            ]
```

Add `from .trace import RewriteStep` at the top of `engine.py`'s trace import (line 1211 already imports `RewriteStep, RewriteTrace`), so `EqualityProof` (defined at line 1214, after that import) can reference it. Confirm the import precedes the class; line 1211 is immediately above line 1214, so it does.

Update the `EqualityProof` docstring (lines 1227-1228) to note the dual shape:

```python
        path_a: From expr_a to common. List[RewriteStep] when produced by
            prove_equal(trace=True) (a synthetic initial step plus one
            labeled step per edge); may also be a plain expression list when
            constructed directly. Step __eq__ compares to an expression by
            its ``after``, so endpoint assertions still hold.
        path_b: From expr_b to common; same shape as path_a.
```

- [ ] **Step 4: Run the test plus the full prove_equal suite (expect PASS).**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_prove_equal.py -v
```

Expected PASS: new labeled-path tests pass; existing `TestProveEqualWithTrace` (lines 236-293: `path_a[0] == start`, `path_a[-1] == common`, `len == 1` for identical) still pass via step endpoint equality; `TestEqualityProofClass` direct-construction tests (plain lists) still pass.

- [ ] **Step 5: Commit.**

```bash
git add rerum/engine.py rerum/tests/test_prove_equal.py
git commit -m "feat(engine): labeled prove_equal paths and EqualityProof steps

visited_a/visited_b carry the producing edge label; reconstruct_path
returns List[RewriteStep] (synthetic initial step + labeled edge steps).
EqualityProof.format/to_dict tolerate both step paths and legacy
expression paths. Step __eq__ keeps path[0]==start / path[-1]==common
assertions working.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: `OptimizationResult.derivation`

Adds the labeled derivation path from the original expression to the minimum found, reusing `prove_equal(trace=True)` to recover the rule sequence between the two endpoints. Backward compatible: `derivation` defaults to `None` and `to_dict`/truthiness are unchanged for callers that ignore it.

**Files:**
- `rerum/optimize.py` (EDIT): `OptimizationResult.__slots__`/`__init__` (lines ~115-129), `to_dict` (lines ~168-177).
- `rerum/engine.py` (EDIT): `minimize` (line ~3599), populate `derivation` before returning.
- `rerum/tests/test_optimization.py` (EXTEND): new class `TestOptimizationDerivation`.

- [ ] **Step 1: Failing test for the derivation.**

Append to `rerum/tests/test_optimization.py`:

```python
from rerum import RewriteTrace, RewriteStep


class TestOptimizationDerivation:
    """minimize records a labeled derivation from original to minimum."""

    def test_derivation_defaults_none_when_no_improvement(self):
        # A rule set with no applicable rule leaves expr at its original.
        eng = RuleEngine.from_dsl("@commute: (+ ?x ?y) <=> (+ :y :x)")
        result = eng.minimize(["z"], metric="size")
        assert result.derivation is None or len(result.derivation) == 0

    def test_derivation_present_when_improved(self):
        eng = RuleEngine.from_dsl("""
            @add-zero: (+ ?x 0) <=> :x
        """)
        result = eng.minimize(["+", "x", 0], metric="size")
        assert result.expr == "x"
        assert isinstance(result.derivation, RewriteTrace)
        assert result.derivation.initial == ["+", "x", 0]
        assert result.derivation.final == "x"
        assert len(result.derivation) >= 1

    def test_derivation_steps_are_labeled(self):
        eng = RuleEngine.from_dsl("@add-zero: (+ ?x 0) <=> :x")
        result = eng.minimize(["+", "x", 0], metric="size")
        steps = [s for s in result.derivation
                 if isinstance(s, RewriteStep) and s.kind == "rule"]
        assert any(s.rule_id and s.rule_id.startswith("add-zero") for s in steps)

    def test_to_dict_includes_derivation_when_present(self):
        eng = RuleEngine.from_dsl("@add-zero: (+ ?x 0) <=> :x")
        result = eng.minimize(["+", "x", 0], metric="size")
        d = result.to_dict()
        import json
        assert json.dumps(d) is not None
        assert "derivation" in d
```

- [ ] **Step 2: Run the failing test (expect fail).**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_optimization.py::TestOptimizationDerivation -v
```

Expected FAIL: `TypeError: __init__() got an unexpected keyword argument 'derivation'` / `AttributeError: 'OptimizationResult' object has no attribute 'derivation'`.

- [ ] **Step 3: Add `derivation` to `OptimizationResult` and populate it in `minimize`.**

In `rerum/optimize.py`, extend `__slots__` (line 115) and `__init__` (lines 117-129):

```python
    __slots__ = ("expr", "cost", "original", "original_cost",
                 "expressions_checked", "derivation")

    def __init__(
        self,
        expr: ExprType,
        cost: float,
        original: ExprType,
        original_cost: float,
        expressions_checked: int = 0,
        derivation: Optional["RewriteTrace"] = None,
    ):
        self.expr = expr
        self.cost = cost
        self.original = original
        self.original_cost = original_cost
        self.expressions_checked = expressions_checked
        self.derivation = derivation
```

Add the typing import at the top of `optimize.py` if not present (`from typing import Optional` likely already imported; verify and add `Optional` if missing). `RewriteTrace` is only referenced as a forward string annotation, so no runtime import is required.

Extend `to_dict` (lines 168-177) to include the derivation when present:

```python
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        d = {
            "expr": self.expr,
            "cost": self.cost,
            "original": self.original,
            "original_cost": self.original_cost,
            "improvement": self.improvement,
            "expressions_checked": self.expressions_checked,
        }
        if self.derivation is not None:
            d["derivation"] = self.derivation.to_dict()
        else:
            d["derivation"] = None
        return d
```

In `rerum/engine.py` `minimize` (lines 3680-3686), build the derivation by proving the original equal to the best expression with a trace, then assembling a `RewriteTrace` from the A-side labeled steps (original -> common) followed by the reversed B-side steps (common -> best). When no improvement was found (`best_expr == expr`) or the proof is unavailable, leave `derivation=None`:

```python
        derivation = None
        if _expr_to_tuple(best_expr) != _expr_to_tuple(expr):
            proof = self.prove_equal(
                expr, best_expr, max_depth=max_depth,
                max_expressions=max_count, trace=True,
                include_unidirectional=include_unidirectional,
                groups=groups, rules=rules,
            )
            if proof is not None and proof.path_a is not None \
                    and proof.path_b is not None:
                trace = RewriteTrace()
                trace.initial = expr
                trace.final = best_expr
                # path_a: original -> common (skip the synthetic initial step).
                for step in proof.path_a[1:]:
                    trace(step)
                # path_b: best -> common; reverse to get common -> best.
                # Skip the synthetic initial (best) step, reverse the rest.
                for step in reversed(proof.path_b[1:]):
                    trace(step)
                derivation = trace

        return OptimizationResult(
            expr=best_expr,
            cost=best_cost,
            original=expr,
            original_cost=original_cost,
            expressions_checked=count,
            derivation=derivation,
        )
```

Note: the per-step `path`/`before`/`after` on the reused proof steps describe single-step neighbor edges, sufficient for the labeled rule sequence; whole-expression roots for the derivation come from `RewriteTrace.to_global_sequence()` when the emitter (Phase 7) needs them. This task only guarantees the labeled rule sequence and the `initial`/`final` endpoints.

- [ ] **Step 4: Run the test plus the optimization suite (expect PASS).**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_optimization.py rerum/tests/test_prove_equal.py -v
```

Expected PASS: derivation tests pass; existing optimization tests (cost, ratios, `to_dict` keys) unaffected since `derivation` is additive and defaults to `None`.

- [ ] **Step 5: Commit.**

```bash
git add rerum/optimize.py rerum/engine.py rerum/tests/test_optimization.py
git commit -m "feat(optimize): OptimizationResult.derivation labeled path

minimize records a RewriteTrace from original to minimum (additive,
default None) by reusing prove_equal(trace=True): A-side steps forward
plus reversed B-side steps. to_dict emits the derivation when present.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Phase 1 Done When

- [ ] `rerum/trace.py` exports pure `splice_at(root, path, subtree)` and `rule_identity(metadata, pattern, skeleton)`; both re-exported from `rerum/__init__.py`.
- [ ] `RewriteStep` carries `rule_id`, `direction`, `bindings`, `path`, `kind`, `guard`, `rationale` as keyword fields with `__slots__` updated; legacy positional construction and `before`/`after` aliases (`before_redex`/`after_redex`) still work.
- [ ] `RewriteStep.to_dict()` emits all situated keys while retaining the legacy `rule_index`, `rule_name`, `description`, `before`, `after` keys.
- [ ] `RewriteTrace.to_global_sequence()` reconstructs whole-expression `before_root`/`after_root` by splicing each redex `after` at its `path` from `self.initial`; `to_dict(global_sequence=True)` embeds it; default `to_dict()` keys unchanged.
- [ ] `_simplify_exhaustive`, `_bottomup_pass`, `_topdown_pass`, and `apply_once` stamp the redex `path` and populate `rule_id`/`direction`/`bindings`/`kind`/`guard`/`rationale` on every emitted step; `HookContext.expr_path` is populated from the same path source.
- [ ] `_all_single_rewrites(..., labeled=True)` returns `(expr, {rule_id, direction, bindings, path})` edges; default shape unchanged.
- [ ] `prove_equal(trace=True)` returns `EqualityProof` whose `path_a`/`path_b` are `List[RewriteStep]`; `EqualityProof.format`/`to_dict` tolerate both step paths and legacy expression-list paths.
- [ ] `OptimizationResult` gains `.derivation` (additive, default `None`); `minimize` populates it with the labeled original-to-minimum trace.
- [ ] `python -m pytest rerum/tests/test_trace_situated.py rerum/tests/test_trace.py rerum/tests/test_prove_equal.py rerum/tests/test_optimization.py rerum/tests/test_strategies.py rerum/tests/test_equivalents.py rerum/tests/test_guards.py -v` is fully green (no regressions; new situated tests pass).
- [ ] `python -m pytest` (full suite) is green.
- [ ] `python experiments/scaling.py` still validates equivalence-class sizes (run after the `_all_single_rewrites`/`prove_equal` changes per the project footguns guidance).
