# Phase 4: Trace-to-Text and Trace-to-Record Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Ship `rerum/training.py`, the general trace-to-text and trace-to-record layer. It turns ONE structured `RewriteTrace` (Phase 1) into two faithful projections: a machine-checkable JSONL record (`to_training_record`) and a deterministic natural-language chain-of-thought (`to_prose`); plus a streaming corpus generator (`generate_corpus`) that is PARAMETERIZED by a caller-supplied `driver` and an optional caller-supplied `checker`. Per the general-engine principle (spec Section 0), `training.py` contains NO domain operator names (`dd`/`int`/`lim`/`and`/`or`): it never decides how to run a problem and never decides how to validate an answer. The driver (which search goal, `simplify` vs `solve`) and the checker (numeric validator) are DATA supplied by the caller and demonstrated in the domain phases (D1/D2), not here. Every record's per-step `before_root`/`after_root` come from `RewriteTrace.to_global_sequence()` (Phase 1), so the structured record, the prose, and the global-sequence replay can never drift.

**Architecture:** `training.py` is a pure projection layer over already-built artifacts. It imports ONLY `format_sexpr` from `rerum.expr` (rendering) and `ExprType` for typing. It does NOT import `rerum.solve`, `rerum.verify`, or anything domain-shaped, and it references no operator literal. `to_training_record` and `to_prose` read `trace.to_global_sequence()` and the situated step fields `step.kind`/`step.rule_id`/`step.rationale`/`step.bindings`/`step.path`/`step.guard`. The prose renderer is a per-`kind` template dispatch (`rule` / `normalize` / `fold`) keyed on the SAME fields the structured record exposes, which is what makes the prose a projection rather than an independent narration. `generate_corpus(engine, problems, *, driver, checker=None)` calls `driver(engine, problem) -> (answer, trace)` for each problem (the caller decides per-problem how to run it: `simplify` for a confluent rule set, `solve` for a non-confluent one), builds the record with `to_training_record`, and when `checker` is supplied stamps `verified = checker(problem, answer)`. The swap test (spec Section 0) holds: replace "calculus" with "boolean algebra" and `training.py` does not change, because it names no domain.

**Tech Stack:** Python 3.9+, stdlib only (`json`, `typing`). pytest with plain asserts (config in `pyproject.toml`: `testpaths = ["rerum/tests"]`). One new test file: `rerum/tests/test_training.py`. The corpus tests run the REAL engine on a domain-free TOY rule set (an algebra-style simplification, NO calculus), so the trace under test is genuine and `training.py` is demonstrated to be domain-agnostic.

**Dependency note (read before starting):** This phase consumes Phase 1 (`rerum/trace.py`) only:
- situated `RewriteStep` constructed as `RewriteStep(rule_index, metadata, before, after, *, rule_id=..., direction=..., bindings=..., path=..., kind=..., guard=..., rationale=...)` with `.kind`/`.rule_id`/`.rationale`/`.bindings`/`.path`/`.guard`/`.before`/`.after`/`.metadata`;
- `RewriteTrace.to_global_sequence() -> List[{"before_root", "after_root", "step"}]` (replays redex edits at their paths from `trace.initial`);
- `trace.initial`/`trace.final`.

These are the load-bearing inputs. `training.py` depends on NOTHING from Phase 2 (`normalize`), Phase 3 (`solve`/`numeval`), or any domain phase. The corpus tests supply their own trivial `driver` and `checker`, so this phase is self-contained: it can land as soon as Phase 1 is merged.

---

## File Structure

```
rerum/
  training.py                        (NEW - this phase: to_training_record, to_prose, generate_corpus)
  trace.py                           (read-only: RewriteStep situated fields, to_global_sequence - Phase 1)
  expr.py                            (read-only: format_sexpr)
  engine.py                          (read-only: RuleEngine.from_dsl, simplify(trace=True))
  __init__.py                        (edited in Task 3: export to_training_record, to_prose, generate_corpus)
  tests/
    test_training.py                 (NEW - this phase)
```

`training.py` public surface (contract-verbatim):

```
to_training_record(trace, *, problem, operator, answer, verified=None) -> dict
to_prose(trace) -> str
generate_corpus(engine, problems, *, driver, checker=None) -> Iterator[dict]
    # driver(engine, problem) -> (answer, trace)   [caller-supplied, domain adapter]
    # checker(problem, answer) -> bool             [caller-supplied domain validator, optional]
```

Record schema (contract-verbatim):

```
{
  "problem": <str>,                    # the original problem, caller-rendered label or s-expr
  "operator": <str>,                   # a free-form label supplied by the caller (NOT interpreted here)
  "steps": [                           # one entry per global-sequence step (kind in {rule, normalize, fold})
    {
      "kind": <str>,                   # "rule" | "normalize" | "fold"
      "rule_id": <str | None>,
      "rationale": <str | None>,
      "before_root": <expr>,           # whole-expression state BEFORE this step (from to_global_sequence)
      "after_root": <expr>,            # whole-expression state AFTER this step (from to_global_sequence)
      "bindings": <dict | None>,
      "path": <list[int] | None>,
      "guard": <dict | None>
    },
    ...
  ],
  "answer": <expr>,                    # the final simplified/solved expression
  "verified": <bool | None>           # checker verdict, None if not run
}
```

Note on `operator`: it is a caller-supplied free-form label that `to_training_record` stores verbatim and `training.py` never inspects. The domain phases pass an operator symbol there (e.g. the head of the problem); the engine layer does not.

Projection invariant (the property the tests pin): for adjacent record steps `k` and `k+1`,
`steps[k+1]["before_root"] == steps[k]["after_root"]` (the global-sequence join is a chain),
`steps[0]["before_root"] == trace.initial`, and `steps[-1]["after_root"] == trace.final`.

`to_prose` projection invariant: every line of prose is derived from a record step's
`kind` + `rule_id` + `rationale` + rendered `before_root`/`after_root`, plus a leading
problem line and a trailing answer line. No field is invented; the same structured trace
produces the same prose every call (deterministic).

---

### Task 1: `to_training_record` (+ the global-sequence join)

The structured JSONL row. Reads `trace.to_global_sequence()` and joins each redex-local step with its whole-expression `before_root`/`after_root`. Pure: no engine, no rewriting, no verification (the caller passes `verified`). The `operator` argument is a free-form label, stored verbatim, never interpreted, so `to_training_record` is domain-agnostic.

**Files:**
- Create: `rerum/training.py`
- Test: `rerum/tests/test_training.py` (create)

- [ ] **Step 1: Write the failing test for `to_training_record` and the global-sequence join.**

  Create `rerum/tests/test_training.py`:

  ```python
  """Tests for the trace-to-text and trace-to-record layer (rerum/training.py).

  to_training_record / to_prose are projections of a single RewriteTrace;
  generate_corpus drives the engine via a CALLER-SUPPLIED driver and stamps
  verification via a CALLER-SUPPLIED checker. training.py names no domain
  operator: the end-to-end tasks run the engine on a domain-free TOY algebra
  rule set so the records under test are genuine AND domain-agnostic.
  """

  import json

  import pytest

  from rerum import RuleEngine, E, RewriteStep, RewriteTrace
  from rerum.engine import RuleMetadata
  from rerum.training import to_training_record


  def _hand_trace():
      """A two-step trace built by hand, editing two redexes of one root.

      Root: (+ (+ x 0) 0).
      Step 1 rewrites the inner (+ x 0) at path [1] -> x; root -> (+ x 0).
      Step 2 rewrites the outer (+ x 0) at path [] -> x; root -> x.
      The global-sequence join must chain: after_root of step 1 equals
      before_root of step 2.
      """
      meta = RuleMetadata(name="add-zero", category="identity",
                          reasoning="additive identity")
      t = RewriteTrace()
      t.initial = ["+", ["+", "x", 0], 0]
      t(RewriteStep(0, meta, ["+", "x", 0], "x", path=[1],
                    rule_id="add-zero", kind="rule",
                    bindings={"x": "x"}, rationale="additive identity"))
      t(RewriteStep(0, meta, ["+", "x", 0], "x", path=[],
                    rule_id="add-zero", kind="rule",
                    bindings={"x": "x"}, rationale="additive identity"))
      t.final = "x"
      return t


  class TestToTrainingRecordSchema:
      def test_top_level_keys(self):
          rec = to_training_record(_hand_trace(), problem="(+ (+ x 0) 0)",
                                   operator="simplify", answer="x")
          for k in ("problem", "operator", "steps", "answer", "verified"):
              assert k in rec, f"missing top-level key {k}"

      def test_operator_and_problem_and_answer_passthrough(self):
          rec = to_training_record(_hand_trace(), problem="(+ (+ x 0) 0)",
                                   operator="simplify", answer="x")
          assert rec["operator"] == "simplify"
          assert rec["problem"] == "(+ (+ x 0) 0)"
          assert rec["answer"] == "x"

      def test_operator_is_free_form_label(self):
          # operator is stored verbatim and never interpreted by training.py.
          rec = to_training_record(_hand_trace(), problem="p",
                                   operator="anything-goes", answer="x")
          assert rec["operator"] == "anything-goes"

      def test_verified_defaults_to_none(self):
          rec = to_training_record(_hand_trace(), problem="p",
                                   operator="simplify", answer="x")
          assert rec["verified"] is None

      def test_verified_passthrough(self):
          rec = to_training_record(_hand_trace(), problem="p",
                                   operator="simplify", answer="x", verified=True)
          assert rec["verified"] is True

      def test_step_keys(self):
          rec = to_training_record(_hand_trace(), problem="p",
                                   operator="simplify", answer="x")
          assert len(rec["steps"]) == 2
          for step in rec["steps"]:
              for k in ("kind", "rule_id", "rationale", "before_root",
                        "after_root", "bindings", "path", "guard"):
                  assert k in step, f"missing step key {k}"

      def test_step_field_values(self):
          rec = to_training_record(_hand_trace(), problem="p",
                                   operator="simplify", answer="x")
          s0 = rec["steps"][0]
          assert s0["kind"] == "rule"
          assert s0["rule_id"] == "add-zero"
          assert s0["rationale"] == "additive identity"
          assert s0["bindings"] == {"x": "x"}
          assert s0["path"] == [1]
          assert s0["guard"] is None


  class TestGlobalSequenceJoin:
      """before_root/after_root come from to_global_sequence and chain."""

      def test_first_before_root_is_initial(self):
          rec = to_training_record(_hand_trace(), problem="p",
                                   operator="simplify", answer="x")
          assert rec["steps"][0]["before_root"] == ["+", ["+", "x", 0], 0]

      def test_last_after_root_is_final(self):
          rec = to_training_record(_hand_trace(), problem="p",
                                   operator="simplify", answer="x")
          assert rec["steps"][-1]["after_root"] == "x"

      def test_adjacent_steps_chain(self):
          rec = to_training_record(_hand_trace(), problem="p",
                                   operator="simplify", answer="x")
          steps = rec["steps"]
          for k in range(len(steps) - 1):
              assert steps[k + 1]["before_root"] == steps[k]["after_root"], (
                  f"join broken between step {k} and {k + 1}")

      def test_intermediate_root(self):
          rec = to_training_record(_hand_trace(), problem="p",
                                   operator="simplify", answer="x")
          # After collapsing the inner (+ x 0), the root is (+ x 0).
          assert rec["steps"][0]["after_root"] == ["+", "x", 0]
          assert rec["steps"][1]["before_root"] == ["+", "x", 0]

      def test_record_is_json_serializable(self):
          rec = to_training_record(_hand_trace(), problem="p",
                                   operator="simplify", answer="x", verified=False)
          assert json.dumps(rec) is not None
  ```

- [ ] **Step 2: Run the test, expect FAIL (module missing).**
  ```bash
  cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_training.py::TestToTrainingRecordSchema rerum/tests/test_training.py::TestGlobalSequenceJoin -v
  ```
  Expected: collection error / `ModuleNotFoundError: No module named 'rerum.training'`.

- [ ] **Step 3: Implement `to_training_record` in `rerum/training.py`.**
  Create `rerum/training.py`:

  ```python
  """The trace-to-text and trace-to-record layer (general, domain-agnostic).

  Projects ONE structured ``RewriteTrace`` (Phase 1) into two faithful,
  non-drifting views:

  - ``to_training_record``: a machine-checkable JSONL row. Per-step
    ``before_root``/``after_root`` come from ``trace.to_global_sequence()``,
    so the structured record IS the whole-expression replay; the in-memory
    step stores only the redex-local edit plus its path.
  - ``to_prose``: a deterministic natural-language chain-of-thought. It is a
    projection of the SAME structured trace (per-``kind`` template plus
    ``step.rationale``), so prose and record cannot diverge.

  ``generate_corpus`` streams records: for each problem it calls a
  CALLER-SUPPLIED ``driver(engine, problem) -> (answer, trace)`` (the caller
  decides whether to run ``simplify`` or ``solve``), builds the record, and
  when a CALLER-SUPPLIED ``checker(problem, answer) -> bool`` is given stamps
  ``verified``.

  General-engine principle (spec Section 0): this module names NO domain
  operator (no ``dd``/``int``/``lim``/``and``/``or``). It never decides how to
  run a problem and never decides how to validate an answer; both are data the
  caller supplies. The ``operator`` field of a record is a free-form label
  stored verbatim and never interpreted here. Swap "calculus" for "boolean
  algebra" and this module does not change.

  Pure projection layer: no rewriting, search, or numeric checks are
  implemented here; those live in ``engine.py``/``solve.py`` and the domain
  content under ``examples/``.
  """

  from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

  from .expr import ExprType, format_sexpr


  def to_training_record(trace, *, problem, operator, answer,
                         verified: Optional[bool] = None) -> Dict[str, Any]:
      """Build the structured JSONL training record from one ``RewriteTrace``.

      The record schema is::

          {problem, operator,
           steps: [{kind, rule_id, rationale, before_root, after_root,
                    bindings, path, guard}, ...],
           answer, verified}

      ``before_root``/``after_root`` are NOT read off the redex-local step;
      they are supplied by joining each step with the whole-expression states
      from ``trace.to_global_sequence()`` (Phase 1). Every step the global
      sequence yields is included (kind in ``{"rule", "normalize", "fold"}``).
      ``operator`` is a free-form caller-supplied label, stored verbatim and
      never interpreted. ``verified`` is passed through unchanged (the caller,
      e.g. ``generate_corpus``, runs a checker and supplies the verdict).
      """
      steps: List[Dict[str, Any]] = []
      for entry in trace.to_global_sequence():
          step = entry["step"]
          steps.append({
              "kind": step.kind,
              "rule_id": step.rule_id,
              "rationale": step.rationale,
              "before_root": entry["before_root"],
              "after_root": entry["after_root"],
              "bindings": step.bindings,
              "path": step.path,
              "guard": step.guard,
          })
      return {
          "problem": problem,
          "operator": operator,
          "steps": steps,
          "answer": answer,
          "verified": verified,
      }
  ```

- [ ] **Step 4: Run the test, expect PASS.**
  ```bash
  cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_training.py::TestToTrainingRecordSchema rerum/tests/test_training.py::TestGlobalSequenceJoin -v
  ```
  Expected: all schema and global-sequence-join tests pass. The join is exactly
  `trace.to_global_sequence()`, so the chain property holds by construction
  (each entry's `after_root` becomes the next entry's `before_root`).

- [ ] **Step 5: Commit.**
  ```bash
  git add rerum/training.py rerum/tests/test_training.py
  git commit -m "feat(training): to_training_record structured JSONL row + global-sequence join

  Projects a RewriteTrace into the schema {problem, operator, steps[...],
  answer, verified}. Each step's before_root/after_root come from
  trace.to_global_sequence() (Phase 1), so the record is the whole-expression
  replay; the in-memory step keeps only the redex-local edit plus its path.
  Adjacent steps chain (after_root[k] == before_root[k+1]); first before_root
  is trace.initial, last after_root is trace.final. operator is a free-form
  label stored verbatim; training.py names no domain operator.

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task 2: `to_prose` renderer (deterministic per-kind projection)

The natural-language chain-of-thought. A PROJECTION of the structured trace: it walks the SAME global sequence, dispatches on `step.kind` to a template, fills `rule_id`/`rationale`/rendered roots, and brackets the lines with a problem line and an answer line. Deterministic (no RNG, no LLM call), so repeated calls on the same trace produce identical text. The templates are keyed on `kind`, never on an operator, so `to_prose` is domain-agnostic.

**Files:**
- Edit: `rerum/training.py` (add `to_prose` and two private helpers)
- Test: `rerum/tests/test_training.py` (extend)

- [ ] **Step 1: Write the failing test for `to_prose`.**
  Append to `rerum/tests/test_training.py`:

  ```python
  from rerum.training import to_prose


  def _mixed_kind_trace():
      """A trace with one rule step and one normalize step (domain-free).

      Root: (+ (* 2 x) (+ y 0)).
      Step 1 (rule): rewrite inner (+ y 0) at path [2] -> y; root ->
                     (+ (* 2 x) y).
      Step 2 (normalize): reorder operands of the + at path [] ->
                     (+ y (* 2 x)).
      """
      meta_rule = RuleMetadata(name="add-zero", category="identity",
                               reasoning="additive identity")
      meta_norm = RuleMetadata(name="canonical-sort", category="normalize",
                               reasoning="commutative ordering")
      t = RewriteTrace()
      t.initial = ["+", ["*", 2, "x"], ["+", "y", 0]]
      t(RewriteStep(0, meta_rule, ["+", "y", 0], "y", path=[2],
                    rule_id="add-zero", kind="rule",
                    rationale="additive identity"))
      t(RewriteStep(0, meta_norm, ["+", ["*", 2, "x"], "y"],
                    ["+", "y", ["*", 2, "x"]], path=[],
                    rule_id="canonical-sort", kind="normalize",
                    rationale="commutative ordering"))
      t.final = ["+", "y", ["*", 2, "x"]]
      return t


  class TestToProseProjection:
      def test_starts_with_problem(self):
          prose = to_prose(_mixed_kind_trace())
          first = prose.splitlines()[0]
          assert "(+ (* 2 x) (+ y 0))" in first

      def test_ends_with_answer(self):
          prose = to_prose(_mixed_kind_trace())
          last = prose.splitlines()[-1]
          assert "(+ y (* 2 x))" in last

      def test_rule_step_mentions_rule_id_and_rationale(self):
          prose = to_prose(_mixed_kind_trace())
          assert "add-zero" in prose
          assert "additive identity" in prose
          # The rule template reads "becomes".
          assert "becomes" in prose

      def test_rule_step_renders_before_and_after_roots(self):
          prose = to_prose(_mixed_kind_trace())
          # The rule step's whole-expression before/after roots appear.
          assert "(+ (* 2 x) (+ y 0))" in prose   # before_root of step 1
          assert "(+ (* 2 x) y)" in prose          # after_root of step 1

      def test_normalize_step_uses_simplifying_template(self):
          prose = to_prose(_mixed_kind_trace())
          assert "Simplifying" in prose
          assert "canonical-sort" in prose

      def test_deterministic(self):
          t = _mixed_kind_trace()
          assert to_prose(t) == to_prose(t)

      def test_anonymous_rule_id_is_handled(self):
          # A step with rule_id=None renders without crashing.
          t = RewriteTrace()
          t.initial = ["+", "x", 0]
          t(RewriteStep(0, RuleMetadata(), ["+", "x", 0], "x", path=[],
                        rule_id=None, kind="rule"))
          t.final = "x"
          prose = to_prose(t)
          assert "x" in prose

      def test_empty_trace_is_problem_then_answer(self):
          t = RewriteTrace()
          t.initial = ["+", "x", "y"]
          t.final = ["+", "x", "y"]
          prose = to_prose(t)
          lines = prose.splitlines()
          # No steps: just the problem framing and the (unchanged) answer.
          assert "(+ x y)" in lines[0]
          assert "(+ x y)" in lines[-1]


  class TestToProseFoldTemplate:
      """A fold step reads as 'Computing with ...'."""

      def _fold_trace(self):
          meta_fold = RuleMetadata(name="fold-add", category="fold",
                                   reasoning="constant folding")
          t = RewriteTrace()
          t.initial = ["+", ["+", 2, 3], "x"]
          t(RewriteStep(0, meta_fold, ["+", 2, 3], 5, path=[1],
                        rule_id="fold-add", kind="fold",
                        rationale="constant folding"))
          t.final = ["+", 5, "x"]
          return t

      def test_fold_step_uses_computing_template(self):
          prose = to_prose(self._fold_trace())
          assert "Computing" in prose
          assert "fold-add" in prose
          # before_root -> after_root for the fold.
          assert "(+ (+ 2 3) x)" in prose
          assert "(+ 5 x)" in prose
  ```

- [ ] **Step 2: Run the test, expect FAIL.**
  ```bash
  cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_training.py::TestToProseProjection rerum/tests/test_training.py::TestToProseFoldTemplate -v
  ```
  Expected: `ImportError: cannot import name 'to_prose'` (then, once added but
  before templates are wired, assertion failures).

- [ ] **Step 3: Implement `to_prose` and its helpers in `rerum/training.py`.**
  Append to `rerum/training.py`:

  ```python
  def _render(expr: ExprType) -> str:
      """Render an expression to a DSL s-expr string for prose/labels."""
      return format_sexpr(expr)


  def _prose_line(entry: Dict[str, Any]) -> str:
      """Render ONE global-sequence entry as a chain-of-thought line.

      Dispatches on the step's ``kind`` to a fixed template, filling the
      ``rule_id``, the ``rationale`` (when present), and the rendered
      whole-expression ``before_root``/``after_root``. The line is a pure
      function of the structured fields, which is what makes the prose a
      projection of the record (no field is invented). The dispatch is on
      ``kind`` only; no operator name is consulted, so this is domain-agnostic.

      Templates (kind -> shape):
        rule      -> "Applying <rule_id> (<rationale>): <before> becomes <after>."
        normalize -> "Simplifying with <rule_id> (<rationale>): <before> becomes <after>."
        fold      -> "Computing with <rule_id> (<rationale>): <before> becomes <after>."
      The "(<rationale>)" clause is omitted when ``rationale`` is None/empty.
      """
      step = entry["step"]
      before = _render(entry["before_root"])
      after = _render(entry["after_root"])
      rule_id = step.rule_id if step.rule_id is not None else "(anonymous rule)"
      reason = f" ({step.rationale})" if step.rationale else ""
      if step.kind == "normalize":
          verb = "Simplifying with"
      elif step.kind == "fold":
          verb = "Computing with"
      else:  # "rule" and any future kind default to the rule template
          verb = "Applying"
      return f"{verb} {rule_id}{reason}: {before} becomes {after}."


  def to_prose(trace) -> str:
      """Render a deterministic natural-language chain-of-thought.

      A PROJECTION of the structured trace: it begins with the problem
      (``trace.initial``), emits one per-``kind`` templated line for each
      global-sequence step (rule / normalize / fold), and ends with the
      answer (``trace.final``). No LLM call, no randomness: the same trace
      yields the same prose every time. Because the lines are derived from
      the same fields ``to_training_record`` exposes, the prose and the
      structured record cannot drift. Templates key on ``kind`` only, so this
      is domain-agnostic.
      """
      lines: List[str] = []
      lines.append(f"Problem: {_render(trace.initial)}.")
      for entry in trace.to_global_sequence():
          lines.append(_prose_line(entry))
      lines.append(f"Answer: {_render(trace.final)}.")
      return "\n".join(lines)
  ```

- [ ] **Step 4: Run the test, expect PASS.**
  ```bash
  cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_training.py::TestToProseProjection rerum/tests/test_training.py::TestToProseFoldTemplate -v
  ```
  Expected: all pass. The problem line is `trace.initial`, the answer line is
  `trace.final`, rule steps read "Applying ... becomes ...", normalize steps
  read "Simplifying with ...", fold steps read "Computing with ...". Each line's
  before/after are the whole-expression roots, so the prose narrates the global
  derivation, not the redex-local edit.

- [ ] **Step 5: Commit.**
  ```bash
  git add rerum/training.py rerum/tests/test_training.py
  git commit -m "feat(training): to_prose deterministic per-kind chain-of-thought

  Projects the same RewriteTrace as to_training_record into natural language:
  a problem line, one templated line per global-sequence step (rule ->
  'Applying', normalize -> 'Simplifying with', fold -> 'Computing with',
  each naming the rule_id and rationale and showing the whole-expression
  before/after roots), and an answer line. Deterministic; no LLM call.
  Templates key on kind only, so no domain operator is named. The prose
  cannot drift from the record because both read the same structured fields.

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task 3: `generate_corpus` parameterized by `driver` and `checker`

The streaming corpus generator. For each problem it calls the CALLER-SUPPLIED `driver(engine, problem) -> (answer, trace)`, builds the record with `to_training_record`, sets `verified` from the CALLER-SUPPLIED `checker(problem, answer) -> bool` when `checker` is provided, and yields the record. It is a generator (streams, does not accumulate). `generate_corpus` references no operator name, no `simplify`/`solve` choice, and no numeric verifier: those decisions live entirely in the caller's `driver`/`checker`. The end-to-end tests use a domain-free TOY algebra rule set, demonstrating that `training.py` is domain-agnostic.

**Files:**
- Edit: `rerum/training.py` (add `generate_corpus`)
- Edit: `rerum/__init__.py` (export `to_training_record`, `to_prose`, `generate_corpus`)
- Test: `rerum/tests/test_training.py` (extend; this is the END-TO-END toy task)

- [ ] **Step 1: Write the failing end-to-end test for `generate_corpus` over a TOY rule set.**
  Append to `rerum/tests/test_training.py`:

  ```python
  import types

  from rerum.training import generate_corpus


  # A domain-free TOY rule set: ordinary algebra simplification, NO calculus.
  # This proves training.py is domain-agnostic (no dd/int/lim anywhere).
  TOY_RULES = """
      @add-zero {category=identity}: (+ ?x 0) => :x
      @mul-one {category=identity}: (* ?x 1) => :x
      @mul-zero {category=annihilator}: (* ?x 0) => 0
  """


  def _toy_engine():
      """Engine over the toy algebra rules. No prelude needed (no folds)."""
      return RuleEngine.from_dsl(TOY_RULES)


  def _simplify_driver(engine, problem):
      """Caller-supplied adapter: run the confluent toy rules via simplify.

      A problem is an (label, expr) pair. Returns (answer, trace) where the
      trace is the Phase-1 RewriteTrace from simplify(trace=True). This adapter
      is the CALLER's responsibility; training.py never picks simplify vs solve.
      """
      _label, expr = problem
      result, trace = engine.simplify(expr, trace=True)
      return result, trace


  def _is_atom_checker(problem, answer):
      """Caller-supplied validator: the toy answer should reduce to an atom.

      Domain-free: for the toy rules every problem collapses to a single atom.
      Returns True iff the answer is a non-list (a symbol or number).
      """
      return not isinstance(answer, list)


  class TestGenerateCorpusToy:
      def test_yields_a_record_per_problem(self):
          engine = _toy_engine()
          problems = [("p1", ["+", "x", 0]),
                      ("p2", ["*", "y", 1])]
          records = list(generate_corpus(engine, problems,
                                         driver=_simplify_driver,
                                         checker=_is_atom_checker))
          assert len(records) == 2

      def test_is_a_streaming_generator(self):
          engine = _toy_engine()
          gen = generate_corpus(engine, [("p", ["+", "x", 0])],
                                driver=_simplify_driver)
          assert isinstance(gen, types.GeneratorType)

      def test_answer_is_the_simplified_result(self):
          engine = _toy_engine()
          recs = list(generate_corpus(engine, [("p", ["+", "x", 0])],
                                      driver=_simplify_driver))
          assert recs[0]["answer"] == "x"

      def test_checker_stamps_verified_true(self):
          engine = _toy_engine()
          recs = list(generate_corpus(engine, [("p", ["+", "x", 0])],
                                      driver=_simplify_driver,
                                      checker=_is_atom_checker))
          assert recs[0]["verified"] is True

      def test_checker_can_stamp_false(self):
          # A checker that rejects everything stamps verified=False.
          engine = _toy_engine()
          recs = list(generate_corpus(engine, [("p", ["+", "x", 0])],
                                      driver=_simplify_driver,
                                      checker=lambda prob, ans: False))
          assert recs[0]["verified"] is False

      def test_no_checker_leaves_verified_none(self):
          engine = _toy_engine()
          recs = list(generate_corpus(engine, [("p", ["+", "x", 0])],
                                      driver=_simplify_driver))
          assert recs[0]["verified"] is None

      def test_operator_label_comes_from_driver_free_problem(self):
          # The caller controls the operator label via the problem; here the
          # driver passes the problem label through (see implementation note).
          engine = _toy_engine()
          recs = list(generate_corpus(engine, [("simplify", ["*", "z", 1])],
                                      driver=_simplify_driver))
          assert recs[0]["operator"] == "simplify"

      def test_record_chain_property_holds_end_to_end(self):
          # The global-sequence join must chain for a REAL engine trace.
          engine = _toy_engine()
          rec = next(generate_corpus(engine, [("p", ["+", ["+", "x", 0], 0])],
                                     driver=_simplify_driver))
          steps = rec["steps"]
          assert steps, "expected a non-empty derivation"
          for k in range(len(steps) - 1):
              assert steps[k + 1]["before_root"] == steps[k]["after_root"]


  class TestGenerateCorpusExport:
      def test_exports(self):
          import rerum
          assert rerum.to_training_record is to_training_record
          assert rerum.to_prose is to_prose
          assert rerum.generate_corpus is generate_corpus
  ```

  Implementation note for the `operator` label: `generate_corpus` does not invent
  an operator name (that would violate the general principle). It stores the
  caller's per-problem `operator` label. To keep `generate_corpus` from naming a
  domain, the operator label is taken from the problem itself: a problem is an
  `(operator_label, expr)` pair, and `generate_corpus` passes `operator_label`
  through to `to_training_record`. The `driver` receives the WHOLE `(label, expr)`
  pair and decides how to run it. This keeps the engine layer free of operator
  literals while still letting the caller tag records.

- [ ] **Step 2: Run the test, expect FAIL.**
  ```bash
  cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_training.py::TestGenerateCorpusToy rerum/tests/test_training.py::TestGenerateCorpusExport -v
  ```
  Expected: `ImportError: cannot import name 'generate_corpus'` (and the export
  test fails until `__init__.py` re-exports the three names).

- [ ] **Step 3: Implement `generate_corpus` in `rerum/training.py`.**
  Append to `rerum/training.py`:

  ```python
  # A problem is an (operator_label, expr) pair. ``operator_label`` is a
  # free-form string the caller chooses (NOT interpreted by this module); it is
  # stamped verbatim into the record's ``operator`` field. ``expr`` is the
  # expression to drive. The ``driver`` receives the whole pair and decides how
  # to run it (simplify for a confluent rule set, solve for a non-confluent
  # one); that decision is the CALLER's, never this module's.
  ProblemType = Tuple[str, ExprType]

  DriverType = Callable[[Any, ProblemType], Tuple[Any, Any]]
  CheckerType = Callable[[ProblemType, Any], bool]


  def generate_corpus(engine, problems: Sequence[ProblemType], *,
                      driver: DriverType,
                      checker: Optional[CheckerType] = None,
                      ) -> Iterator[Dict[str, Any]]:
      """Stream training records, one per problem, via a caller-supplied driver.

      ``generate_corpus`` is GENERAL: it names no domain operator and makes no
      ``simplify``-vs-``solve`` choice. For each problem it calls::

          answer, trace = driver(engine, problem)

      where ``driver`` is the caller's domain adapter (it runs ``simplify`` for
      a confluent rule set, ``solve`` for a non-confluent one, and returns the
      Phase-1 ``RewriteTrace``). The record is built with
      ``to_training_record`` (the problem's ``operator_label`` becomes the
      record's ``operator``, the s-expr rendering of the problem expression
      becomes ``problem``, and ``answer`` is the driver's result). When
      ``checker`` is supplied, ``verified = checker(problem, answer)``;
      otherwise ``verified`` stays ``None``. Yields the record (a generator:
      records stream out, nothing is accumulated).

      ``problems`` is a sequence of ``(operator_label, expr)`` pairs. The
      operator label is the caller's free-form tag; this module stores it
      verbatim and never inspects it.
      """
      for problem in problems:
          operator_label, expr = problem
          answer, trace = driver(engine, problem)
          verified: Optional[bool] = None
          if checker is not None:
              verified = checker(problem, answer)
          record = to_training_record(
              trace,
              problem=format_sexpr(expr),
              operator=operator_label,
              answer=answer,
              verified=verified,
          )
          yield record
  ```

  In `rerum/__init__.py`, add the export (after the existing engine/trace import
  group) and extend `__all__`:

  ```python
  from .training import generate_corpus, to_prose, to_training_record
  ```

  and add `"generate_corpus"`, `"to_prose"`, `"to_training_record"` to `__all__`.

- [ ] **Step 4: Run the test, expect PASS.**
  ```bash
  cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_training.py::TestGenerateCorpusToy rerum/tests/test_training.py::TestGenerateCorpusExport -v
  ```
  Expected: all pass. The records come from REAL engine traces over the TOY
  algebra rules (no calculus), the caller's `driver` chooses `simplify`, the
  caller's `checker` stamps `verified`, the chain property holds end to end, and
  the three names are exported from the package. `training.py` mentions no
  operator symbol.

- [ ] **Step 5: Commit.**
  ```bash
  git add rerum/training.py rerum/__init__.py rerum/tests/test_training.py
  git commit -m "feat(training): generate_corpus parameterized by driver + checker

  generate_corpus(engine, problems, *, driver, checker=None) yields one record
  per problem. The caller-supplied driver(engine, problem) -> (answer, trace)
  decides how to run each problem (simplify for confluent rule sets, solve for
  non-confluent ones); the optional caller-supplied checker(problem, answer)
  -> bool stamps verified. Streams (generator), does not accumulate. No domain
  operator name and no simplify-vs-solve choice live in training.py. Exported
  from rerum.

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task 4: End-to-end corpus smoke test (JSONL round trip + record/prose agreement)

Proves the full pipeline as a file artifact, still domain-free: generate a corpus of TOY algebra problems with a caller-supplied driver and checker, serialize each record as one JSON line to a tmp file, read the file back, and assert the records validate against the schema, are verified, and that the prose projection (rendered from the same traces) mentions the rule names. This is the capstone integration smoke test and the explicit demonstration that `training.py` is domain-agnostic.

**Files:**
- Test: `rerum/tests/test_training.py` (extend)

- [ ] **Step 1: Write the failing end-to-end JSONL round-trip test.**
  Append to `rerum/tests/test_training.py`:

  ```python
  from rerum.engine import format_sexpr


  class TestCorpusJsonlRoundTrip:
      """Generate a JSONL corpus over the TOY rules, write it, read it back."""

      REQUIRED_TOP = ("problem", "operator", "steps", "answer", "verified")
      REQUIRED_STEP = ("kind", "rule_id", "rationale", "before_root",
                       "after_root", "bindings", "path", "guard")

      def _problems(self):
          return [
              ("simplify", ["+", ["+", "x", 0], 0]),  # -> x, two add-zero steps
              ("simplify", ["*", ["* ", "y", 1] if False else ["*", "y", 1], 1]),
          ]

      def test_write_and_read_back_jsonl(self, tmp_path):
          engine = _toy_engine()
          out = tmp_path / "corpus.jsonl"
          problems = [
              ("simplify", ["+", ["+", "x", 0], 0]),
              ("simplify", ["*", ["*", "y", 1], 1]),
          ]
          # Stream records to disk, one JSON object per line.
          with out.open("w") as fh:
              for rec in generate_corpus(engine, problems,
                                         driver=_simplify_driver,
                                         checker=_is_atom_checker):
                  fh.write(json.dumps(rec) + "\n")

          # Read back and validate.
          lines = out.read_text().splitlines()
          assert len(lines) == 2
          records = [json.loads(line) for line in lines]

          for rec in records:
              for k in self.REQUIRED_TOP:
                  assert k in rec, f"record missing {k}"
              assert rec["operator"] == "simplify"
              assert rec["verified"] is True, "every record passes the checker"
              assert rec["steps"], "non-empty derivation"
              for step in rec["steps"]:
                  for k in self.REQUIRED_STEP:
                      assert k in step, f"step missing {k}"
              # Chain property survives JSON round-trip.
              steps = rec["steps"]
              for i in range(len(steps) - 1):
                  assert steps[i + 1]["before_root"] == steps[i]["after_root"]

      def test_prose_projection_mentions_rule_names(self):
          # The prose, rendered from the same trace, names the rules that fired.
          engine = _toy_engine()
          _, trace = engine.simplify(["+", ["+", "x", 0], 0], trace=True)
          prose = to_prose(trace)
          # The prose begins with the problem and ends with the answer.
          assert "(+ (+ x 0) 0)" in prose.splitlines()[0]
          # At least one rule id appears (the rule lines name the fired rule).
          rule_ids = {s.rule_id for s in trace.steps if s.rule_id}
          assert rule_ids, "expected named rules in the derivation"
          assert any(rid in prose for rid in rule_ids), (
              "prose must name a rule that fired")

      def test_record_and_prose_agree_on_step_count(self):
          # The record's step list and the prose's per-step lines both derive
          # from to_global_sequence, so their counts match.
          engine = _toy_engine()
          full = ["+", ["+", "x", 0], 0]
          _, trace = engine.simplify(full, trace=True)
          rec = to_training_record(trace, problem=format_sexpr(full),
                                   operator="simplify", answer=trace.final)
          prose_lines = to_prose(trace).splitlines()
          # prose = 1 problem line + N step lines + 1 answer line.
          assert len(prose_lines) == len(rec["steps"]) + 2

      def test_training_module_names_no_domain_operator(self):
          # Guardrail for the general-engine principle: training.py must not
          # contain any domain operator literal (dd/int/lim/and/or as words).
          import re
          from pathlib import Path
          import rerum.training as _t
          src = Path(_t.__file__).read_text()
          # Strip string/quote noise is unnecessary; we look for the operator
          # tokens as standalone identifiers in code or docstring.
          for op in ("dd", "int(", "lim", " and ", " or "):
              # ' and '/' or ' would match Python keywords; restrict to the
              # operator-symbol sense by requiring quotes around them.
              pass
          # Concrete check: the calculus operator symbols must not appear as
          # quoted string literals (how they would be referenced as ops).
          for sym in ('"dd"', "'dd'", '"int"', "'int'", '"lim"', "'lim'"):
              assert sym not in src, f"training.py must not name operator {sym}"
  ```

  Note: `test_training_module_names_no_domain_operator` is a guardrail asserting
  the general-engine principle by inspecting the module source for quoted
  operator-symbol literals (`"dd"`, `"int"`, `"lim"`). Keep it; it fails loudly
  if a future edit smuggles a domain operator into `training.py`.

- [ ] **Step 2: Run the test, expect FAIL (or PASS if Tasks 1-3 fully wired).**
  ```bash
  cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_training.py::TestCorpusJsonlRoundTrip -v
  ```
  Expected: FAIL only if a wiring gap remains (e.g. prose/record step-count
  mismatch, or a record not verified). If Tasks 1-3 are correct, the only NEW
  surface here is the tmp-file round trip, the record/prose agreement, and the
  source guardrail, which should pass directly; treat any failure as a real
  integration bug, not a test to weaken.

- [ ] **Step 3: No new source needed; if a test fails, fix `training.py`, not the test.**
  This task adds no production code: it exercises the Task 1-3 surface as a file
  artifact. If `test_record_and_prose_agree_on_step_count` fails, the cause is a
  mismatch between the number of global-sequence entries the record uses and the
  number of prose step lines: both MUST iterate `trace.to_global_sequence()`
  exactly once. If `test_training_module_names_no_domain_operator` fails, a
  domain operator literal leaked into `training.py`; remove it (the driver and
  checker are the caller's, never the module's). (Per Phase 1,
  `to_global_sequence()` yields one entry per recorded step, so the record and
  prose step counts coincide by construction.)

- [ ] **Step 4: Run the test, expect PASS.**
  ```bash
  cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_training.py::TestCorpusJsonlRoundTrip -v
  ```
  Expected: all pass. The corpus serializes to JSONL, reads back, every record
  validates against the schema and is `verified=True`, the chain property
  survives the JSON round trip, the prose names a rule that fired with a per-step
  line count matching the record, and the source guardrail confirms `training.py`
  names no domain operator.

- [ ] **Step 5: Run the full training suite plus a regression check.**
  ```bash
  cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_training.py -v
  cd /home/spinoza/github/repos/rerum && python -m pytest -q
  ```
  Expected: `test_training.py` all green; the full suite green (this phase adds
  only `rerum/training.py` and three exports, all additive).

- [ ] **Step 6: Commit.**
  ```bash
  git add rerum/tests/test_training.py
  git commit -m "test(training): end-to-end JSONL corpus round trip on a domain-free toy rule set

  Generate a TOY algebra corpus (no calculus) with a caller-supplied driver
  and checker, stream it to a tmp JSONL file, read it back, and assert every
  record validates against the schema, is checker-verified, preserves the
  global-sequence chain across the JSON round trip, and that the prose
  projection names the rules that fired with a per-step line count matching
  the record. A source guardrail asserts training.py names no domain operator
  literal, enforcing the general-engine principle.

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

## Done When

- [ ] `rerum/training.py` exists and exposes the contract-verbatim surface:
  `to_training_record(trace, *, problem, operator, answer, verified=None) -> dict`,
  `to_prose(trace) -> str`, and
  `generate_corpus(engine, problems, *, driver, checker=None) -> Iterator[dict]`.
- [ ] `training.py` contains NO domain operator name: no `"dd"`/`"int"`/`"lim"`/
  `"and"`/`"or"` operator literal, no `simplify`-vs-`solve` choice, no numeric
  verifier. The `operator` field is a free-form caller-supplied label stored
  verbatim. The source guardrail test asserts this. (General-engine principle,
  spec Section 0.)
- [ ] Records validate against the schema
  `{problem, operator, steps:[{kind, rule_id, rationale, before_root,
  after_root, bindings, path, guard}], answer, verified}`; every step's
  `before_root`/`after_root` come from `trace.to_global_sequence()` (Phase 1),
  steps whose `kind` is in `{rule, normalize, fold}` are all included, and the
  global-sequence join chains (`steps[k+1].before_root == steps[k].after_root`,
  `steps[0].before_root == trace.initial`, `steps[-1].after_root == trace.final`),
  surviving a JSON round trip.
- [ ] `to_prose` is a faithful, deterministic PROJECTION of the structured
  trace: a problem line (`trace.initial`), one per-`kind` templated line per
  global-sequence step (`rule` -> "Applying", `normalize` -> "Simplifying with",
  `fold` -> "Computing with", each naming the `rule_id` and `rationale` and
  rendering the whole-expression before/after via `format_sexpr`), and an answer
  line (`trace.final`); repeated calls on the same trace are byte-identical; the
  prose's step-line count equals the record's step count (both project
  `to_global_sequence`). Templates key on `kind` only, so `to_prose` is
  domain-agnostic.
- [ ] `generate_corpus` STREAMS (is a generator, does not accumulate) and is
  PARAMETERIZED by a caller-supplied `driver(engine, problem) -> (answer, trace)`
  and an optional caller-supplied `checker(problem, answer) -> bool`. It makes no
  domain choice itself: the driver decides `simplify` vs `solve`, the checker
  decides validation. `verified` is `checker(problem, answer)` when `checker` is
  given, else `None`.
- [ ] The end-to-end tests build traces by ACTUALLY running the engine on a
  DOMAIN-FREE TOY algebra rule set (no calculus): `generate_corpus` with a
  trivial `_simplify_driver` and `_is_atom_checker` yields verified records,
  streams (is a generator), and a JSONL corpus round-trips through a tmp file.
- [ ] `to_training_record`, `to_prose`, `generate_corpus` are exported from
  `rerum/__init__.py` (`rerum.to_training_record is to_training_record`, etc.).
- [ ] `python -m pytest -q` passes with no regressions; `rerum/tests/test_training.py`
  is fully green.
