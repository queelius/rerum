# RewriteTrace.inverse() Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pure `inverse()` to `RewriteStep` and `RewriteTrace` that turns a forward derivation around (final -> initial), and use it to orient the engine's `minimize` derivation correctly.

**Architecture:** `RewriteStep.inverse()` swaps the redex-local `before`/`after`, flips `direction` (fwd<->rev), keeps `path`/`kind`/identity, and nulls the forward-match `bindings`/`guard`. `RewriteTrace.inverse()` swaps `initial`/`final` and maps `inverse()` over the steps in reverse order. `minimize` then inverts (not just reverses) its `path_b` steps so the derivation chains correctly under `to_global_sequence`.

**Tech Stack:** Python 3.9+, stdlib only, pytest. Files: `rerum/trace.py`, `rerum/engine.py`, tests in `rerum/tests/test_trace.py`, `rerum/tests/test_optimization.py`, `rerum/tests/test_mcp_tools.py`.

Design spec: `docs/superpowers/specs/2026-06-10-rewritetrace-inverse-design.md`.

---

### Task 1: `RewriteStep.inverse()`

**Files:**
- Modify: `rerum/trace.py` (add a method to `class RewriteStep`, after `after_redex` property near line 107)
- Test: `rerum/tests/test_trace.py`

- [ ] **Step 1: Write the failing tests**

Append to `rerum/tests/test_trace.py`:

```python
class TestRewriteStepInverse:
    """RewriteStep.inverse(): swap before/after, flip direction, keep path,
    null the forward-match bindings/guard."""

    def _meta(self, name="r"):
        from rerum.engine import RuleMetadata
        return RuleMetadata(name=name)

    def test_inverse_swaps_and_flips(self):
        meta = self._meta()
        step = RewriteStep(
            0, meta, ["foo", "a"], ["bar", "a"], rule_id="r",
            direction="fwd", bindings={"x": "a"}, path=[1], kind="rule",
            guard={"condition": ["?", "p"], "result": True},
            rationale="why")
        inv = step.inverse()
        assert inv.before == ["bar", "a"]      # was after
        assert inv.after == ["foo", "a"]       # was before
        assert inv.direction == "rev"          # flipped
        assert inv.path == [1]                 # unchanged
        assert inv.kind == "rule"              # unchanged
        assert inv.rule_id == "r"              # unchanged
        assert inv.metadata is meta            # same object
        assert inv.rationale == "why"          # unchanged
        assert inv.bindings is None            # cleared
        assert inv.guard is None               # cleared

    def test_inverse_none_direction_stays_none(self):
        step = RewriteStep(0, self._meta(), ["a"], ["b"], direction=None)
        assert step.inverse().direction is None

    def test_inverse_rev_becomes_fwd(self):
        step = RewriteStep(0, self._meta(), ["a"], ["b"], direction="rev")
        assert step.inverse().direction == "fwd"

    def test_inverse_is_structural_involution(self):
        step = RewriteStep(
            3, self._meta(), ["+", "a", 0], "a", rule_id="add-zero",
            direction="rev", path=[2], kind="normalize", rationale="why")
        twice = step.inverse().inverse()
        assert twice.before == step.before
        assert twice.after == step.after
        assert twice.direction == step.direction
        assert twice.path == step.path
        assert twice.kind == step.kind
        assert twice.rule_id == step.rule_id
        assert twice.rationale == step.rationale

    def test_inverse_path_is_a_copy(self):
        p = [1, 2]
        step = RewriteStep(0, self._meta(), ["a"], ["b"], path=p)
        inv = step.inverse()
        assert inv.path == [1, 2]
        assert inv.path is not p  # pure: not the same list object

    def test_inverse_preserves_kind_for_all_kinds(self):
        # kind is a passthrough string; inverting a normalize/fold/initial
        # step keeps its kind (a normalize/fold step inverts to its specific
        # reverse edit; an initial step is a no-op).
        for kind in ("rule", "normalize", "fold", "initial"):
            before, after = ("x", "x") if kind == "initial" else (["a"], ["b"])
            step = RewriteStep(0, self._meta(), before, after, kind=kind)
            inv = step.inverse()
            assert inv.kind == kind
            assert inv.before == after and inv.after == before
```

- [ ] **Step 2: Run the tests, expect FAIL**

Run: `python -m pytest rerum/tests/test_trace.py::TestRewriteStepInverse -v`
Expected: FAIL with `AttributeError: 'RewriteStep' object has no attribute 'inverse'`.

- [ ] **Step 3: Implement `RewriteStep.inverse()`**

In `rerum/trace.py`, inside `class RewriteStep`, add this method immediately after the `after_redex` property (around line 107, before `__eq__`):

```python
    def inverse(self) -> "RewriteStep":
        """Return the reverse of this step.

        Swaps the redex-local ``before``/``after``, flips ``direction``
        (``fwd`` <-> ``rev``; ``None`` unchanged), and keeps ``path`` (the
        same child-index path locates the post-step subtree, because
        ``splice_at`` preserves indices), ``kind``, and the rule identity.
        ``bindings`` and ``guard`` are CLEARED: they describe the forward
        match, and a pure trace transform cannot know the reverse
        application's match. So ``inverse()`` is a STRUCTURAL involution
        (before/after/direction/path/kind round-trip), not a field-identical
        one. See docs/superpowers/specs/2026-06-10-rewritetrace-inverse-design.md.
        """
        flipped = {"fwd": "rev", "rev": "fwd"}.get(self.direction,
                                                   self.direction)
        return RewriteStep(
            self.rule_index,
            self.metadata,
            self.after,
            self.before,
            rule_id=self.rule_id,
            direction=flipped,
            bindings=None,
            path=(list(self.path) if self.path is not None else None),
            kind=self.kind,
            guard=None,
            rationale=self.rationale,
        )
```

- [ ] **Step 4: Run the tests, expect PASS**

Run: `python -m pytest rerum/tests/test_trace.py::TestRewriteStepInverse -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add rerum/trace.py rerum/tests/test_trace.py
git commit -m "feat(trace): RewriteStep.inverse() -- swap before/after, flip direction

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `RewriteTrace.inverse()`

**Files:**
- Modify: `rerum/trace.py` (add a method to `class RewriteTrace`, after `add_step` near line 189)
- Test: `rerum/tests/test_trace.py`

Depends on Task 1 (`RewriteStep.inverse`).

- [ ] **Step 1: Write the failing tests**

Append to `rerum/tests/test_trace.py`:

```python
class TestRewriteTraceInverse:
    """RewriteTrace.inverse(): swap initial/final, steps reversed + inverted.
    The load-bearing property is replay correctness on a NESTED redex."""

    def test_inverse_replays_final_to_initial_nested(self):
        # The redex (foo a) sits at path [1] of (top (foo a)) -- a nested
        # edit, so path preservation is exercised.
        eng = RuleEngine.from_dsl("@r: (foo ?x) => (bar :x)")
        result, trace = eng.simplify(["top", ["foo", "a"]], trace=True)
        assert result == ["top", ["bar", "a"]]

        inv = trace.inverse()
        assert inv.initial == ["top", ["bar", "a"]]  # was final
        assert inv.final == ["top", ["foo", "a"]]     # was initial

        seq = inv.to_global_sequence()
        assert seq[0]["before_root"] == ["top", ["bar", "a"]]
        assert seq[-1]["after_root"] == ["top", ["foo", "a"]]
        for k in range(len(seq) - 1):
            assert seq[k]["after_root"] == seq[k + 1]["before_root"]

    def test_inverse_multistep_chains(self):
        eng = RuleEngine.from_dsl(
            "@r: (foo ?x) => (bar :x)\n@s: (bar ?x) => (baz :x)")
        result, trace = eng.simplify(["wrap", ["foo", "a"]], trace=True)
        assert len(trace.steps) == 2

        inv = trace.inverse()
        seq = inv.to_global_sequence()
        assert seq[0]["before_root"] == result
        assert seq[-1]["after_root"] == ["wrap", ["foo", "a"]]
        for k in range(len(seq) - 1):
            assert seq[k]["after_root"] == seq[k + 1]["before_root"]

    def test_inverse_inverse_is_structural_identity(self):
        eng = RuleEngine.from_dsl("@r: (foo ?x) => (bar :x)")
        _result, trace = eng.simplify(["top", ["foo", "a"]], trace=True)
        twice = trace.inverse().inverse()
        assert twice.initial == trace.initial
        assert twice.final == trace.final
        assert len(twice.steps) == len(trace.steps)
        for a, b in zip(twice.steps, trace.steps):
            assert a.before == b.before
            assert a.after == b.after
            assert a.direction == b.direction
            assert a.path == b.path
            assert a.kind == b.kind

    def test_inverse_empty_trace(self):
        t = RewriteTrace()
        t.initial = "x"
        t.final = "x"
        inv = t.inverse()
        assert inv.initial == "x" and inv.final == "x"
        assert inv.steps == []
```

- [ ] **Step 2: Run the tests, expect FAIL**

Run: `python -m pytest rerum/tests/test_trace.py::TestRewriteTraceInverse -v`
Expected: FAIL with `AttributeError: 'RewriteTrace' object has no attribute 'inverse'`.

- [ ] **Step 3: Implement `RewriteTrace.inverse()`**

In `rerum/trace.py`, inside `class RewriteTrace`, add this method immediately after `add_step` (around line 189, before `format`):

```python
    def inverse(self) -> "RewriteTrace":
        """Return the reverse trace: from ``final`` back to ``initial``.

        Steps are reversed in order and each is ``RewriteStep.inverse()``-d,
        so ``self.inverse().to_global_sequence()`` replays ``final ->
        initial`` with the chain intact. Pure: a new trace is returned.
        """
        out = RewriteTrace()
        out.initial = self.final
        out.final = self.initial
        out.steps = [s.inverse() for s in reversed(self.steps)]
        return out
```

- [ ] **Step 4: Run the tests, expect PASS**

Run: `python -m pytest rerum/tests/test_trace.py::TestRewriteTraceInverse -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add rerum/trace.py rerum/tests/test_trace.py
git commit -m "feat(trace): RewriteTrace.inverse() -- reverse + invert the derivation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: orient the `minimize` derivation with `step.inverse()`

**Files:**
- Modify: `rerum/engine.py:4001-4002` (the reversed `path_b` loop inside `minimize`)
- Test: `rerum/tests/test_optimization.py`

Depends on Task 1.

- [ ] **Step 1: Write the failing test**

Append to `rerum/tests/test_optimization.py`:

```python
class TestMinimizeDerivationChains:
    """minimize's derivation must chain correctly under to_global_sequence:
    the reversed path_b steps must be INVERTED, not merely reordered (the
    Phase 1 limitation). Endpoint-correct but not chain-correct was the bug."""

    def test_derivation_global_sequence_chains(self):
        from rerum import RuleEngine
        engine = RuleEngine.from_dsl(
            "@az: (+ ?x 0) <=> :x\n@comm: (+ ?x ?y) <=> (+ :y :x)")
        opt = engine.minimize(["+", 0, "a"])
        assert opt.expr == "a"
        deriv = opt.derivation
        assert deriv is not None

        seq = deriv.to_global_sequence()
        assert seq[0]["before_root"] == ["+", 0, "a"]
        assert seq[-1]["after_root"] == "a"
        for k in range(len(seq) - 1):
            assert seq[k]["after_root"] == seq[k + 1]["before_root"], (
                "minimize derivation does not chain: reversed path_b steps "
                "were not inverted")
```

- [ ] **Step 2: Run the test, expect FAIL**

Run: `python -m pytest rerum/tests/test_optimization.py::TestMinimizeDerivationChains -v`
Expected: FAIL on `seq[-1]["after_root"] == "a"`. With the current (reverse-but-not-invert) code, the reversed `path_b` step is misoriented and replays as a PHANTOM NO-OP, so the derivation ends at the common form `["+", "a", 0]` and never reaches the best form `"a"` (verified: the sequence is `["+",0,"a"] -> ["+","a",0]` then `["+","a",0] -> ["+","a",0]`). The adjacent-join assertion happens to hold trivially because the misoriented step is a no-op; the ENDPOINT assertion is the one that catches the bug.

- [ ] **Step 3: Apply the one-line fix**

In `rerum/engine.py`, find the reversed `path_b` loop (around line 4001-4002) inside `minimize`. It currently reads:

```python
                for step in reversed(proof.path_b[1:]):  # common -> best
                    trace(step)
```

Change the body to invert each step:

```python
                for step in reversed(proof.path_b[1:]):  # common -> best
                    # Reverse the ORDER and INVERT each step: path_b is a
                    # forward path (best -> common), so each step must be
                    # turned around (swap before/after, flip direction) to
                    # read common -> best. Reordering alone leaves the steps
                    # oriented best -> common, breaking the global-sequence
                    # chain (the Phase 1 minimize-derivation limitation).
                    trace(step.inverse())
```

- [ ] **Step 4: Run the test, expect PASS**

Run: `python -m pytest rerum/tests/test_optimization.py::TestMinimizeDerivationChains -v`
Expected: PASS.

- [ ] **Step 5: Run the full optimization + trace suites for regressions**

Run: `python -m pytest rerum/tests/test_optimization.py rerum/tests/test_trace.py rerum/tests/test_prove_equal.py -q`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add rerum/engine.py rerum/tests/test_optimization.py
git commit -m "fix(engine): invert reversed path_b steps in minimize derivation

Closes the Phase 1 minimize-derivation limitation: the derivation was
endpoint-correct but not chain-correct under to_global_sequence because the
reversed path_b steps kept their forward orientation. inverse() turns each
around, so original -> common -> best chains.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: pin the MCP minimize prose payoff + CHANGELOG

**Files:**
- Test: `rerum/tests/test_mcp_tools.py`
- Modify: `CHANGELOG.md`

Depends on Task 3 (the engine fix that the MCP layer consumes).

- [ ] **Step 1: Write the test for the MCP prose payoff**

Append to `rerum/tests/test_mcp_tools.py` (the module already imports `pytest`):

```python
class TestMinimizeProseNoPhantomSteps:
    """After the inverse() fix, the MCP minimize prose narrates real moves:
    no phantom no-op step (before == after) and the answer is the best form.
    Pins the 0.9.0 review's minimize-prose finding once inverse() lands."""

    def test_prose_has_no_no_op_steps(self):
        import re
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_minimize
        engine = RuleEngine.from_dsl(
            "@az: (+ ?x 0) <=> :x\n@comm: (+ ?x ?y) <=> (+ :y :x)")
        result = tool_minimize(engine, expr="(+ 0 a)")
        assert result["best"] == "a"
        assert result["prose"].splitlines()[-1] == "Answer: a."
        step_line = re.compile(
            r"^(?:Applying|Simplifying with|Computing with) .*?: "
            r"(.+) becomes (.+)\.$")
        for line in result["prose"].splitlines():
            m = step_line.match(line)
            if m:
                assert m.group(1) != m.group(2), (
                    f"phantom no-op step in prose: {line!r}")
```

- [ ] **Step 2: Run the test, expect PASS**

Run: `python -m pytest rerum/tests/test_mcp_tools.py::TestMinimizeProseNoPhantomSteps -v`
Expected: PASS (the engine fix from Task 3 propagates to the MCP layer, which renders prose from `opt.derivation`).

- [ ] **Step 3: Add a CHANGELOG note**

In `CHANGELOG.md`, under the existing `## [0.9.0]` section's `### Fixed` list (or add `### Fixed` if absent), add this bullet:

```markdown
- `RewriteTrace.inverse()` / `RewriteStep.inverse()`: a pure reverse-trace
  primitive (swap before/after, flip direction, keep path). `minimize`'s
  derivation now inverts its reversed `path_b` steps, so the derivation
  chains correctly under `to_global_sequence` (the Phase 1 limitation) and
  the MCP `minimize` prose narrates the real original->best moves.
```

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest -q`
Expected: all PASS (the prior baseline plus the new tests from Tasks 1-4).

- [ ] **Step 5: Commit**

```bash
git add rerum/tests/test_mcp_tools.py CHANGELOG.md
git commit -m "test(mcp): pin minimize prose has no phantom steps post-inverse; changelog

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Done When

- `RewriteStep.inverse()` and `RewriteTrace.inverse()` exist, pure, with the documented structural-involution semantics (bindings/guard cleared).
- `RewriteTrace.inverse().to_global_sequence()` replays `final -> initial` with the chain intact, verified on a nested-redex trace.
- `minimize`'s `OptimizationResult.derivation` chains correctly under `to_global_sequence`.
- The MCP `minimize` prose narrates real moves (no phantom no-op step) and closes with the true best form.
- `python -m pytest -q` is green.
