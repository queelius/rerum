# Phase 5: MCP Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the general-engine MCP server (`rerum/mcp/`) per the revised
spec `docs/superpowers/specs/2026-06-04-symbolic-reasoning-engine-design.md`
(Section 5.9) and the revised contract `Phase 5 - MCP server`. The server exposes
the general rewriting engine to LLM agents: authoring rules, persisting and
loading rule sets and theories as data, applying rules (with a situated trace
plus a natural-language `prose` rendering), goal-directed solving, and an
optional agentic loop. This plan reconciles the prior MCP design and plan
(`docs/superpowers/specs/2026-05-04-mcp-design.md`,
`docs/superpowers/plans/2026-05-04-mcp-server.md`) by applying six deltas; every
unchanged tool handler is reused verbatim from the prior plan.

**Architecture:** New `rerum/mcp/` submodule. Files: `__init__.py` (entry,
optional-import guard), `server.py` (per-session lifecycle, dispatch), `tools.py`
(tool handlers), `trace.py` (situated-trace serialization + prose + recorder +
truncation), `persistence.py` (file-backed rule sets and theories), `solver.py`
(the agentic-loop resolver factory for `solve_assisted`), `errors.py`
(`MCPToolError` + mapping). Tool handlers are thin orchestration over
`RuleEngine`; they contain NO domain logic. The server loads rules and theories
as DATA. The trace serialized to the agent is the Phase 1 SITUATED trace (each
step carries `rule_id`, `direction`, `bindings`, `path`, `kind`, `guard`,
`rationale`), reconstructed whole-expression via `to_global_sequence`, and every
rewriting tool additionally returns a `prose` field rendered by
`rerum.training.to_prose` (Phase 4).

**Tech Stack:** Python 3.9+, `mcp` SDK (optional dependency), `pytest`. Optional
install extra `[mcp]` and console script `rerum-mcp` declared in `pyproject.toml`.

**Phase dependencies (hard):** This plan builds on Phases 0 to 4 of the same
refactor. It consumes:
- Phase 1 situated `RewriteStep` fields (`rule_id`, `direction`, `bindings`,
  `path`, `kind`, `guard`, `rationale`), `RewriteStep.to_dict()`, and
  `RewriteTrace.to_global_sequence()` (whole-expression `before_root`/
  `after_root` per step).
- Phase 3 `RuleEngine.solve(expr, goal_predicate, **kw) -> SolveResult` and the
  `contains_op(expr, ops)` goal helper from `rerum.solve`.
- Phase 4 `from rerum.training import to_prose`.
- Phase 0 `combine_preludes(*preludes)` and prelude bundles (`ARITHMETIC_PRELUDE`,
  `MATH_PRELUDE`, `PREDICATE_PRELUDE`, `FULL_PRELUDE`).

If a phase is not yet landed when this plan is executed, the depending task's
failing-test step will surface the gap (an `ImportError` or `AttributeError`),
which is the correct signal to land the prerequisite first.

---

## Deltas applied versus the 2026-05-04 prior plan

1. **General-engine principle.** The server holds NO domain logic. It loads rules
   and theories as data. The `reset_engine` `prelude` arg names a computation
   bundle (`arithmetic`/`math`/`predicate`/`full`/`none`) or a combination via
   `combine_preludes`; there is NO domain bundle (no `CALCULUS_PRELUDE`).
2. **Naming disambiguation.** The prior `solve` tool (the LLM-resolver agentic
   loop) is RENAMED `solve_assisted` to avoid colliding with the new engine
   `solve()`. A NEW `solve_goal(expr, goal, ...)` tool wraps engine `solve()`
   with a caller-described goal (e.g. `goal={"op_free": ["int","lim"]}` becomes a
   `contains_op` predicate).
3. **Situated trace.** The returned trace is the Phase 1 situated trace; steps
   carry `rule_id`, `direction`, `bindings`, `path`, `kind`, `guard`,
   `rationale`, plus whole-expression `before_root`/`after_root` from
   `to_global_sequence`. Truncation, termination, and bidirectional handling are
   reused from the prior plan.
4. **NL explanation (REVERSED non-goal).** Every rewriting tool's response gains a
   `prose` field rendered via `rerum.training.to_prose`. The agent relays prose
   to the user.
5. **Persistence (NEW).** Tools `save_ruleset(name)`, `load_ruleset(name)`,
   `list_rulesets()` backed by a git-friendly rules directory (default
   `.rerum/rules/`, files `<name>.json`), built on `to_json`/
   `load_rules_from_json`; plus `load_theory(name)` for `<name>.theory.json`.
6. **Agentic loop renamed.** The prior resolver-callback design is kept intact
   under the new name `solve_assisted`, returning the situated trace + prose.

---

## Tool list (final)

| Group | Tools |
|---|---|
| Authoring | `load_rules`, `add_rule`, `list_rules`, `get_rule`, `validate_examples` |
| Persistence (NEW) | `save_ruleset`, `load_ruleset`, `list_rulesets`, `load_theory` |
| Applying | `simplify`, `apply_once`, `equivalents`, `prove_equal`, `minimize` |
| Goal solving (NEW) | `solve_goal` (wraps engine `solve()`) |
| Agentic loop (renamed) | `solve_assisted` (was the prior `solve`) |
| Admin | `reset_engine`, `get_status` |

Naming disambiguation: engine `solve()` is the Phase 3 best-first search;
`solve_goal` is its MCP wrapper; `solve_assisted` is the LLM-resolver loop.
Every applying tool, `solve_goal`, and `solve_assisted` return a `prose` field.

---

## File Structure

**Create:**
- `rerum/mcp/__init__.py`: public entry, `run_server()`, `PROTOCOL_VERSION`, optional-import guard
- `rerum/mcp/server.py`: `RerumMCPServer` per-session state, tool dispatch, engine_busy guard
- `rerum/mcp/trace.py`: situated `step_to_dict`, `assemble_trace` (with `prose`), `trace_recorder`, truncation
- `rerum/mcp/persistence.py`: `RuleStore` (file-backed rule sets and theories)
- `rerum/mcp/tools.py`: tool handler functions
- `rerum/mcp/solver.py`: LLM-resolver factory for `solve_assisted`
- `rerum/mcp/errors.py`: `MCPToolError` + `map_exception`
- `rerum/tests/test_mcp_trace.py`: situated-trace serialization + prose tests
- `rerum/tests/test_mcp_tools.py`: per-tool happy + error path tests
- `rerum/tests/test_mcp_persistence.py`: rule set and theory file store tests
- `rerum/tests/test_mcp_solve_goal.py`: `solve_goal` over a toy non-confluent rule set
- `rerum/tests/test_mcp_solve_assisted.py`: agentic loop with mocked sampling
- `rerum/tests/test_mcp_smoke.py`: module + server lifecycle + concurrency

**Modify:**
- `pyproject.toml`: add `[mcp]` extra + `rerum-mcp` entry point + version bump
- `rerum/__init__.py`: `__version__` bump (no eager `mcp` import; it is optional)
- `CHANGELOG.md`: 0.8.0 release entry
- `CLAUDE.md`: brief `mcp/` architecture subsection + MCP footgun

---

## Task 1: Module skeleton + optional-dependency guard

**Files:**
- Create: `rerum/mcp/__init__.py`
- Create: `rerum/tests/test_mcp_smoke.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write the failing test**

Create `rerum/tests/test_mcp_smoke.py`:

```python
"""Smoke tests for the rerum.mcp module entry point."""

import pytest


class TestMCPModule:
    def test_can_import_rerum_mcp(self):
        import rerum.mcp
        assert rerum.mcp is not None

    def test_run_server_callable_exists(self):
        from rerum.mcp import run_server
        assert callable(run_server)

    def test_module_exposes_version(self):
        from rerum.mcp import PROTOCOL_VERSION
        assert isinstance(PROTOCOL_VERSION, str)
        assert PROTOCOL_VERSION
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_mcp_smoke.py -v 2>&1 | tail -15
```

Expected FAIL: ImportError on `rerum.mcp`.

- [ ] **Step 3: Create `rerum/mcp/__init__.py`**

```python
"""Rerum MCP server.

Exposes the GENERAL rerum rewriting engine to LLM agents via the Model
Context Protocol. The server contains no domain logic: it loads rules and
theories as DATA. See the revised spec
docs/superpowers/specs/2026-06-04-symbolic-reasoning-engine-design.md
(Section 5.9) and the prior MCP design
docs/superpowers/specs/2026-05-04-mcp-design.md.

This module requires the optional ``mcp`` package; install via
``pip install rerum[mcp]``. Importing this module without the SDK
installed raises an informative ImportError.
"""

PROTOCOL_VERSION = "0.8.0"

try:
    import mcp as _mcp_sdk  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "rerum.mcp requires the 'mcp' package. Install via "
        "'pip install rerum[mcp]'."
    ) from exc


def run_server(transport: str = "stdio", host: str = "127.0.0.1",
               port: int = 8765) -> None:
    """Run the rerum MCP server.

    Args:
        transport: ``"stdio"`` (default) or ``"http"``.
        host: HTTP transport bind address (ignored for stdio).
        port: HTTP transport port (ignored for stdio).
    """
    # Wired in Task 11.
    raise NotImplementedError("server entry point wired in Task 11")


__all__ = ["run_server", "PROTOCOL_VERSION"]
```

- [ ] **Step 4: Update `pyproject.toml`**

Find `[project.optional-dependencies]` and add the `mcp` extra:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
]
docs = [
    "mkdocs>=1.5",
    "mkdocs-material>=9.0",
]
mcp = [
    "mcp>=1.0",
]
```

Find `[project.scripts]` (add it if missing) and add the entry point:

```toml
[project.scripts]
rerum = "rerum.cli:main"
rerum-mcp = "rerum.mcp:run_server"
```

- [ ] **Step 5: Run tests (expected PASS)**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_mcp_smoke.py -v 2>&1 | tail -10
```

Expected PASS: 3 passed.

- [ ] **Step 6: Run full suite**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest 2>&1 | tail -3
```

Expected: prior count + 3 passing.

- [ ] **Step 7: Commit**

```bash
cd /home/spinoza/github/repos/rerum && git add rerum/mcp/__init__.py rerum/tests/test_mcp_smoke.py pyproject.toml
git commit -m "$(cat <<'EOF'
feat(mcp): module skeleton and optional dependency declaration

rerum/mcp/__init__.py declares run_server() (stub) and PROTOCOL_VERSION,
guards against missing 'mcp' SDK with an informative ImportError.
pyproject.toml gains the [mcp] extra and the rerum-mcp entry point.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Situated trace step serialization

This is delta 3. The serializer emits the Phase 1 situated fields, NOT the
pre-Phase-1 `rule_name`/`reasoning`-only shape. `before_root`/`after_root` are
supplied per step by `assemble_trace` (Task 4) from `to_global_sequence`; the
per-step serializer here emits the rule-local situated fields.

**Files:**
- Create: `rerum/mcp/trace.py`
- Create: `rerum/tests/test_mcp_trace.py`

- [ ] **Step 1: Write the failing test**

Create `rerum/tests/test_mcp_trace.py`:

```python
"""Tests for MCP situated-trace serialization."""

import pytest


class TestStepToDict:
    def test_situated_fields_present(self):
        from rerum.mcp.trace import step_to_dict
        from rerum.engine import RuleMetadata
        from rerum.trace import RewriteStep

        meta = RuleMetadata(
            name="add-zero",
            category="identity",
            reasoning="Zero is the additive identity.",
        )
        step = RewriteStep(
            rule_index=0,
            metadata=meta,
            before=["+", "x", 0],
            after="x",
            rule_id="add-zero",
            direction=None,
            bindings={"x": "x"},
            path=[],
            kind="rule",
            guard=None,
            rationale="identity",
        )
        d = step_to_dict(step)

        # Situated fields (Phase 1).
        assert d["rule_id"] == "add-zero"
        assert d["direction"] is None
        assert d["bindings"] == {"x": "x"}
        assert d["path"] == []
        assert d["kind"] == "rule"
        assert d["guard"] is None
        assert d["rationale"] == "identity"
        # Redex-local before/after as s-expression strings.
        assert d["before"] == "(+ x 0)"
        assert d["after"] == "x"
        # Citable label fields retained from metadata.
        assert d["rule_name"] == "add-zero"
        assert d["category"] == "identity"
        assert d["reasoning"] == "Zero is the additive identity."
        assert d["rule_index"] == 0
        assert d["provenance"] is None

    def test_normalize_kind_step(self):
        from rerum.mcp.trace import step_to_dict
        from rerum.engine import RuleMetadata
        from rerum.trace import RewriteStep

        meta = RuleMetadata(name="ac-sort", category="normalize")
        step = RewriteStep(
            rule_index=-1,
            metadata=meta,
            before=["+", "b", "a"],
            after=["+", "a", "b"],
            rule_id="ac-sort",
            kind="normalize",
            path=[],
            bindings={},
            rationale="canonical AC ordering",
        )
        d = step_to_dict(step)
        assert d["kind"] == "normalize"
        assert d["rationale"] == "canonical AC ordering"

    def test_guard_step_records_condition_and_result(self):
        from rerum.mcp.trace import step_to_dict
        from rerum.engine import RuleMetadata
        from rerum.trace import RewriteStep

        meta = RuleMetadata(name="power-rule", category="calculus")
        step = RewriteStep(
            rule_index=2,
            metadata=meta,
            before=["dd", ["^", "x", 2], "x"],
            after=["*", 2, ["^", "x", 1]],
            rule_id="power-rule",
            kind="rule",
            path=[],
            bindings={"n": 2},
            guard={"condition": ["!", "const?", 2], "result": True},
            rationale="exponent is constant",
        )
        d = step_to_dict(step)
        assert d["guard"]["result"] is True
        assert isinstance(d["guard"]["condition"], str)

    def test_provenance_from_extra(self):
        from rerum.mcp.trace import step_to_dict
        from rerum.engine import RuleMetadata
        from rerum.trace import RewriteStep

        meta = RuleMetadata(name="inferred", extra={"provenance": "llm-inferred"})
        step = RewriteStep(
            rule_index=5, metadata=meta, before=["foo", "x"], after="x",
            rule_id="inferred", kind="rule", path=[], bindings={"x": "x"},
        )
        d = step_to_dict(step)
        assert d["provenance"] == "llm-inferred"

    def test_bidirectional_direction_label(self):
        from rerum.mcp.trace import step_to_dict
        from rerum.engine import RuleMetadata
        from rerum.trace import RewriteStep

        meta = RuleMetadata(
            name="assoc-fwd", category="associativity",
            bidirectional=True, direction="fwd", fwd_label="regroup-right",
        )
        step = RewriteStep(
            rule_index=0, metadata=meta,
            before=["+", ["+", "a", "b"], "c"],
            after=["+", "a", ["+", "b", "c"]],
            rule_id="assoc-fwd", direction="fwd", kind="rule",
            path=[], bindings={"a": "a", "b": "b", "c": "c"},
        )
        d = step_to_dict(step)
        assert d["direction"] == "fwd"
        assert d["direction_label"] == "regroup-right"
```

- [ ] **Step 2: Verify failure**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_mcp_trace.py::TestStepToDict -v 2>&1 | tail -15
```

Expected FAIL: ImportError on `rerum.mcp.trace`.

- [ ] **Step 3: Create `rerum/mcp/trace.py`**

```python
"""Situated-trace serialization, prose rendering, and recording for the
MCP server.

Wraps engine operations in a temporary ``on_rule_applied`` hook and
serializes the resulting situated ``RewriteTrace`` into the JSON shape
consumed by LLM agents. The shape is the Phase 1 situated trace (each
step carries rule_id, direction, bindings, path, kind, guard, rationale)
plus a domain-agnostic natural-language ``prose`` rendering via
``rerum.training.to_prose`` (Phase 4).
"""

import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from rerum.engine import format_sexpr
from rerum.trace import RewriteStep, RewriteTrace


# Trace truncation defaults. When a trace exceeds MAX_STEPS, the response
# carries the first HEAD_STEPS + last TAIL_STEPS plus an elision marker.
MAX_STEPS = 200
HEAD_STEPS = 100
TAIL_STEPS = 100


def step_to_dict(step: RewriteStep) -> Dict[str, Any]:
    """Convert a situated RewriteStep to the MCP step JSON shape.

    Emits the Phase 1 situated fields (rule_id, direction, bindings, path,
    kind, guard, rationale) alongside the citable metadata labels
    (rule_name, category, reasoning) and the redex-local before/after as
    s-expression strings. ``before_root``/``after_root`` are added by
    ``assemble_trace`` from ``to_global_sequence``; this function emits the
    rule-local edit only.

    The ``provenance`` field is read from ``metadata.extra["provenance"]``;
    rules added by the LLM resolver during ``solve_assisted`` carry
    ``"llm-inferred"`` here. Bidirectional rules add a ``direction_label``.
    """
    meta = step.metadata
    guard = step.guard
    if guard is not None and isinstance(guard.get("condition"), (list, str, int, float)):
        guard = {
            "condition": format_sexpr(guard["condition"]),
            "result": guard.get("result"),
        }

    out: Dict[str, Any] = {
        # Phase 1 situated fields.
        "rule_id": step.rule_id,
        "direction": step.direction,
        "bindings": step.bindings,
        "path": list(step.path) if step.path is not None else [],
        "kind": step.kind,
        "guard": guard,
        "rationale": step.rationale,
        # Redex-local edit (s-expression strings).
        "before": format_sexpr(step.before),
        "after": format_sexpr(step.after),
        # Citable metadata labels.
        "rule_name": meta.name,
        "category": meta.category,
        "reasoning": meta.reasoning,
        "rule_index": step.rule_index,
        "provenance": (meta.extra or {}).get("provenance"),
    }

    if meta.bidirectional:
        if step.direction == "fwd" and meta.fwd_label:
            out["direction_label"] = meta.fwd_label
        elif step.direction == "rev" and meta.rev_label:
            out["direction_label"] = meta.rev_label

    return out
```

- [ ] **Step 4: Run tests (expected PASS)**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_mcp_trace.py::TestStepToDict -v 2>&1 | tail -10
```

Expected PASS: 5 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/spinoza/github/repos/rerum && git add rerum/mcp/trace.py rerum/tests/test_mcp_trace.py
git commit -m "$(cat <<'EOF'
feat(mcp): situated step_to_dict serializer

Emits the Phase 1 situated fields (rule_id, direction, bindings, path,
kind, guard, rationale) plus citable metadata labels and the redex-local
before/after as s-expression strings. Guard condition is stringified;
provenance read from metadata.extra; direction_label added for
bidirectional rules.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Trace recorder context manager

Reused from the prior plan (Task 3). The recorder appends `step_to_dict(step)`;
the situated fields come from the Phase 1 `RewriteStep` the engine now emits.

**Files:**
- Modify: `rerum/mcp/trace.py`
- Modify: `rerum/tests/test_mcp_trace.py`

- [ ] **Step 1: Write the failing test**

Append to `rerum/tests/test_mcp_trace.py`:

```python
class TestTraceRecorder:
    def test_recorder_captures_steps(self):
        from rerum import RuleEngine
        from rerum.mcp.trace import trace_recorder

        engine = RuleEngine.from_dsl(
            '@add-zero {category=identity}: (+ ?x 0) => :x'
        )
        with trace_recorder(engine) as recorder:
            engine.simplify(["+", "y", 0])

        steps = recorder.steps
        assert len(steps) == 1
        assert steps[0]["rule_name"] == "add-zero"
        assert steps[0]["kind"] == "rule"

    def test_recorder_unregisters_after_block(self):
        from rerum import RuleEngine
        from rerum.mcp.trace import trace_recorder

        engine = RuleEngine.from_dsl('@r1: (a ?x) => :x')
        before = engine._hooks.count("rule_applied")
        with trace_recorder(engine):
            engine.simplify(["a", "y"])
        after = engine._hooks.count("rule_applied")
        assert after == before

    def test_recorder_unregisters_on_exception(self):
        from rerum import RuleEngine
        from rerum.mcp.trace import trace_recorder

        engine = RuleEngine.from_dsl('@r1: (a ?x) => :x')
        before = engine._hooks.count("rule_applied")
        with pytest.raises(ValueError):
            with trace_recorder(engine):
                raise ValueError("boom")
        after = engine._hooks.count("rule_applied")
        assert after == before

    def test_recorder_holds_initial_trace_object(self):
        # The recorder also retains the engine's RewriteTrace so callers can
        # call to_global_sequence() in assemble_trace.
        from rerum import RuleEngine
        from rerum.mcp.trace import trace_recorder

        engine = RuleEngine.from_dsl('@r1: (a ?x) => :x')
        with trace_recorder(engine) as recorder:
            engine.simplify(["a", "y"])
        assert recorder.trace is not None
```

- [ ] **Step 2: Verify failure**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_mcp_trace.py::TestTraceRecorder -v 2>&1 | tail -15
```

Expected FAIL: ImportError on `trace_recorder`.

- [ ] **Step 3: Add `trace_recorder` to `rerum/mcp/trace.py`**

Append to the module. The recorder builds a `RewriteTrace` so that
`assemble_trace` can call `to_global_sequence()` for `before_root`/`after_root`:

```python
class _Recorder:
    """Holds the captured situated steps and the RewriteTrace for a block.

    ``steps`` is the list of ``step_to_dict`` outputs (rule-local). ``trace``
    is a ``RewriteTrace`` accumulating the raw situated ``RewriteStep`` objects
    so ``assemble_trace`` can reconstruct whole-expression before/after via
    ``to_global_sequence()``.
    """

    def __init__(self):
        self.steps: List[Dict[str, Any]] = []
        self.trace: Optional[RewriteTrace] = None
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    @property
    def duration_ms(self) -> float:
        if self.start_time is None or self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000.0


@contextmanager
def trace_recorder(engine, *, initial=None):
    """Register a temporary on_rule_applied hook and capture situated steps.

    The hook is removed in a finally block so an exception inside the
    with-block does not leak the registration. Yields a ``_Recorder``
    whose ``.steps`` (serialized) and ``.trace`` (raw RewriteTrace) are
    populated as the engine fires ``rule_applied`` events.
    """
    recorder = _Recorder()
    recorder.trace = RewriteTrace(initial=initial)

    def hook(step, ctx):
        recorder.trace.add_step(step)
        recorder.steps.append(step_to_dict(step))

    engine.on_rule_applied(hook)
    recorder.start_time = time.perf_counter()
    try:
        yield recorder
    finally:
        recorder.end_time = time.perf_counter()
        engine.off_rule_applied(hook)
```

Note: if `RewriteTrace` does not expose `add_step`, use the actual accumulation
API confirmed in Phase 1 (`RewriteTrace` already accumulates `steps`; append to
`recorder.trace.steps` directly). The failing test in Step 1 pins only that
`recorder.trace is not None`; match the Phase 1 trace API.

- [ ] **Step 4: Run tests (expected PASS)**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_mcp_trace.py::TestTraceRecorder -v 2>&1 | tail -10
```

Expected PASS: 4 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/spinoza/github/repos/rerum && git add rerum/mcp/trace.py rerum/tests/test_mcp_trace.py
git commit -m "$(cat <<'EOF'
feat(mcp): trace_recorder context manager

Registers a temporary on_rule_applied hook, captures each situated step
via step_to_dict and into a RewriteTrace, deregisters in finally (safe
under exceptions). The RewriteTrace powers to_global_sequence in
assemble_trace.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Trace assembly + global sequence + prose + truncation

This is deltas 3 and 4. `assemble_trace` stitches in whole-expression
`before_root`/`after_root` per step from `to_global_sequence()`, adds the `prose`
rendering via `to_prose`, and applies the prior plan's truncation policy.

**Files:**
- Modify: `rerum/mcp/trace.py`
- Modify: `rerum/tests/test_mcp_trace.py`

- [ ] **Step 1: Write the failing test**

Append to `rerum/tests/test_mcp_trace.py`:

```python
class TestAssembleTrace:
    def test_assemble_adds_global_roots_and_prose(self):
        from rerum import RuleEngine
        from rerum.mcp.trace import assemble_trace, trace_recorder

        engine = RuleEngine.from_dsl("""
            @add-zero {category=identity}: (+ ?x 0) => :x
            @mul-one {category=identity}: (* ?x 1) => :x
        """)
        initial = ["+", ["*", "y", 1], 0]
        with trace_recorder(engine, initial=initial) as recorder:
            result = engine.simplify(initial)

        d = assemble_trace(
            initial="(+ (* y 1) 0)",
            final="y",
            recorder=recorder,
        )
        assert d["initial"] == "(+ (* y 1) 0)"
        assert d["final"] == "y"
        # Whole-expression roots present per step.
        assert "before_root" in d["steps"][0]
        assert "after_root" in d["steps"][0]
        # Prose rendering present and a string.
        assert isinstance(d["prose"], str)
        assert d["prose"]
        assert "summary" in d

    def test_assemble_truncates_long_trace(self):
        from rerum.mcp.trace import assemble_trace
        from rerum.mcp.trace import _Recorder

        rec = _Recorder()
        rec.steps = [
            {"rule_id": f"r{i}", "rule_name": f"r{i}", "before": "x",
             "after": "x", "kind": "rule", "path": [], "bindings": {},
             "direction": None, "guard": None, "rationale": None,
             "category": None, "reasoning": None, "rule_index": 0,
             "provenance": None, "before_root": "x", "after_root": "x"}
            for i in range(250)
        ]
        d = assemble_trace(initial="x", final="x", recorder=rec, prose="")
        assert len(d["steps"]) == 201  # 100 + marker + 100
        assert d["trace_truncated"] == {"original_length": 250}
        assert d["steps"][100].get("_elided") is True
        assert d["steps"][100]["count"] == 50

    def test_assemble_no_truncation_under_max(self):
        from rerum.mcp.trace import assemble_trace, _Recorder

        rec = _Recorder()
        rec.steps = [
            {"rule_id": f"r{i}", "before": "x", "after": "x", "kind": "rule",
             "before_root": "x", "after_root": "x"}
            for i in range(150)
        ]
        d = assemble_trace(initial="x", final="x", recorder=rec, prose="")
        assert "trace_truncated" not in d
        assert len(d["steps"]) == 150

    def test_assemble_empty_steps(self):
        from rerum.mcp.trace import assemble_trace, _Recorder

        rec = _Recorder()
        rec.steps = []
        d = assemble_trace(initial="x", final="x", recorder=rec, prose="")
        assert d["steps"] == []
        assert d["total_steps"] == 0
```

- [ ] **Step 2: Verify failure**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_mcp_trace.py::TestAssembleTrace -v 2>&1 | tail -15
```

Expected FAIL: ImportError or TypeError on `assemble_trace`.

- [ ] **Step 3: Add `assemble_trace` (with global roots + prose) to `rerum/mcp/trace.py`**

```python
def _attach_global_roots(recorder) -> None:
    """Add before_root/after_root to each serialized step from the
    RewriteTrace's whole-expression reconstruction.

    Uses the Phase 1 ``RewriteTrace.to_global_sequence()`` which replays
    from ``initial`` splicing each step's after at its path, yielding
    per-step whole-expression before_root/after_root. Best-effort: if the
    trace cannot reconstruct (missing initial), the roots are omitted.
    """
    if recorder.trace is None:
        return
    try:
        seq = recorder.trace.to_global_sequence()
    except Exception:
        return
    for serialized, entry in zip(recorder.steps, seq):
        before_root = entry.get("before_root")
        after_root = entry.get("after_root")
        serialized["before_root"] = (
            format_sexpr(before_root) if not isinstance(before_root, str)
            and before_root is not None else before_root
        )
        serialized["after_root"] = (
            format_sexpr(after_root) if not isinstance(after_root, str)
            and after_root is not None else after_root
        )


def render_prose(recorder) -> str:
    """Render the recorded trace to natural-language prose.

    Delegates to the GENERAL, domain-agnostic ``rerum.training.to_prose``
    (Phase 4). Returns an empty string if there is no trace to render.
    """
    if recorder.trace is None:
        return ""
    try:
        from rerum.training import to_prose
    except Exception:
        return ""
    try:
        return to_prose(recorder.trace)
    except Exception:
        return ""


def assemble_trace(*, initial: str, final: str, recorder=None,
                   prose: Optional[str] = None) -> Dict[str, Any]:
    """Build the full situated-trace dict for an MCP response.

    Attaches whole-expression before_root/after_root per step (from
    ``to_global_sequence``), a ``prose`` rendering (delta 4), and the
    truncation policy from the prior plan (first HEAD_STEPS + an _elided
    marker + last TAIL_STEPS). ``prose`` may be passed explicitly (tests);
    otherwise it is rendered from the recorder's trace.
    """
    if recorder is not None:
        _attach_global_roots(recorder)
        steps = recorder.steps
    else:
        steps = []

    if prose is None:
        prose = render_prose(recorder) if recorder is not None else ""

    total = len(steps)
    out: Dict[str, Any] = {
        "initial": initial,
        "final": final,
        "total_steps": total,
        "summary": _summarize(steps),
        "prose": prose,
    }

    if total > MAX_STEPS:
        elided_count = total - HEAD_STEPS - TAIL_STEPS
        marker = {"_elided": True, "count": elided_count}
        out["steps"] = steps[:HEAD_STEPS] + [marker] + steps[-TAIL_STEPS:]
        out["trace_truncated"] = {"original_length": total}
    else:
        out["steps"] = steps

    return out


def _summarize(steps: List[Dict[str, Any]]) -> str:
    """One-line digest of the steps, by rule_id and kind."""
    if not steps:
        return "No rules applied."
    counts: Dict[str, int] = {}
    for s in steps:
        name = s.get("rule_id") or s.get("rule_name") or f"rule[{s.get('rule_index')}]"
        counts[name] = counts.get(name, 0) + 1
    most = max(counts.items(), key=lambda kv: kv[1])
    return (
        f"{len(steps)} steps using {len(counts)} unique rules. "
        f"Most used: {most[0]} ({most[1]}x)."
    )
```

- [ ] **Step 4: Run tests (expected PASS)**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_mcp_trace.py::TestAssembleTrace -v 2>&1 | tail -10
```

Expected PASS: 4 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/spinoza/github/repos/rerum && git add rerum/mcp/trace.py rerum/tests/test_mcp_trace.py
git commit -m "$(cat <<'EOF'
feat(mcp): assemble_trace with global roots, prose, and truncation

Stitches whole-expression before_root/after_root per step from
RewriteTrace.to_global_sequence(), adds a prose rendering via
rerum.training.to_prose (the NL-explanation non-goal is reversed), and
applies the first-100 + elision-marker + last-100 truncation policy.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Error mapping + MCPToolError

Reused verbatim from the prior plan (Task 5). Codes gain `sampling_unsupported`
and `resolver_timeout`/`resolver_budget_exhausted` reasons used by
`solve_assisted` (Task 10) and the `not_found` code for persistence (Task 7).

**Files:**
- Create: `rerum/mcp/errors.py`
- Create: `rerum/tests/test_mcp_tools.py`

- [ ] **Step 1: Write the failing test**

Create `rerum/tests/test_mcp_tools.py`:

```python
"""Tests for MCP tool handlers and error mapping."""

import pytest


class TestErrorMapping:
    def test_explicit_parse_error_code(self):
        from rerum.mcp.errors import MCPToolError
        err = MCPToolError("parse_error", "bad input", details={"input": "(a"})
        assert err.code == "parse_error"
        assert err.message == "bad input"
        assert err.details == {"input": "(a"}

    def test_to_dict_shape(self):
        from rerum.mcp.errors import MCPToolError
        err = MCPToolError("unknown_rule", "no rule named 'x'",
                           details={"name": "x", "available": ["a", "b"]})
        d = err.to_dict()
        assert d["error"]["code"] == "unknown_rule"
        assert d["error"]["message"] == "no rule named 'x'"
        assert d["error"]["details"] == {"name": "x", "available": ["a", "b"]}

    def test_validation_error_from_example_validation(self):
        from rerum.mcp.errors import map_exception
        from rerum.engine import ExampleValidationError

        exc = ExampleValidationError(
            "Rule 'x': pattern does not match",
            rule_name="x",
            example={"in": "(a 1)", "out": "1"},
        )
        err = map_exception(exc, context={"tool": "load_rules"})
        assert err["error"]["code"] == "validation_error"
        assert "Rule 'x'" in err["error"]["message"]
        assert err["error"]["details"]["rule_name"] == "x"

    def test_resolver_loop_error_mapping(self):
        from rerum.mcp.errors import map_exception
        from rerum.hooks import ResolverLoopError

        exc = ResolverLoopError("retry cap (100) exceeded")
        err = map_exception(exc, context={"tool": "solve_assisted"})
        assert err["error"]["code"] == "resolver_loop"

    def test_generic_value_error_is_internal(self):
        from rerum.mcp.errors import map_exception
        err = map_exception(ValueError("boom"), context={"tool": "simplify"})
        assert err["error"]["code"] == "internal_error"
```

- [ ] **Step 2: Verify failure**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_mcp_tools.py::TestErrorMapping -v 2>&1 | tail -15
```

Expected FAIL: ImportError on `rerum.mcp.errors`.

- [ ] **Step 3: Create `rerum/mcp/errors.py`**

```python
"""Error handling for MCP tool boundaries.

Engine exceptions caught at the tool boundary become structured
``MCPToolError`` instances with stable code strings the LLM can interpret
without parsing prose.
"""

import os
import traceback
from typing import Any, Dict, Optional


class MCPToolError(Exception):
    """Structured error returned to MCP clients.

    Carries a short ``code``, a human-readable ``message``, and optional
    ``details`` for the LLM to consume programmatically.

    Codes: parse_error, unknown_rule, validation_error, not_found,
    sampling_unsupported, resolver_loop, engine_busy, internal_error.
    """

    def __init__(self, code: str, message: str,
                 details: Optional[Dict[str, Any]] = None,
                 *, cause: Optional[BaseException] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}
        self.cause = cause

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP error response shape.

        ``RERUM_MCP_DEBUG=1`` includes a sanitized ``_traceback`` field for
        development; in production the traceback is omitted.
        """
        out: Dict[str, Any] = {
            "error": {"code": self.code, "message": self.message}
        }
        if self.details:
            out["error"]["details"] = self.details
        if os.environ.get("RERUM_MCP_DEBUG") == "1" and self.cause:
            out["error"]["_traceback"] = "".join(
                traceback.format_exception(
                    type(self.cause), self.cause, self.cause.__traceback__
                )
            )
        return out


def map_exception(exc: BaseException,
                  context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Map an engine exception to an MCP error response dict."""
    from rerum.engine import ExampleValidationError
    from rerum.hooks import (
        HookError, HooksError, ResolutionError, ResolverLoopError,
    )

    if isinstance(exc, ExampleValidationError):
        details: Dict[str, Any] = {}
        if getattr(exc, "rule_name", None) is not None:
            details["rule_name"] = exc.rule_name
        if getattr(exc, "example", None) is not None:
            details["example"] = exc.example
        return MCPToolError(
            "validation_error", str(exc), details=details, cause=exc
        ).to_dict()

    if isinstance(exc, ResolverLoopError):
        return MCPToolError("resolver_loop", str(exc), cause=exc).to_dict()

    if isinstance(exc, (HookError, ResolutionError, HooksError)):
        return MCPToolError(
            "internal_error", f"hook system error: {exc}", cause=exc
        ).to_dict()

    if isinstance(exc, FileNotFoundError):
        return MCPToolError("not_found", str(exc), cause=exc).to_dict()

    if isinstance(exc, ValueError):
        return MCPToolError("internal_error", str(exc), cause=exc).to_dict()

    return MCPToolError(
        "internal_error", f"{type(exc).__name__}: {exc}", cause=exc
    ).to_dict()
```

- [ ] **Step 4: Run tests (expected PASS)**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_mcp_tools.py::TestErrorMapping -v 2>&1 | tail -10
```

Expected PASS: 5 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/spinoza/github/repos/rerum && git add rerum/mcp/errors.py rerum/tests/test_mcp_tools.py
git commit -m "$(cat <<'EOF'
feat(mcp): MCPToolError and engine-exception mapping

Stable codes (parse_error, unknown_rule, validation_error, not_found,
sampling_unsupported, resolver_loop, engine_busy, internal_error).
map_exception converts ExampleValidationError, ResolverLoopError,
HookError, FileNotFoundError, and generic ValueError. Tracebacks omitted
unless RERUM_MCP_DEBUG=1.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Authoring tools (load_rules, add_rule, list_rules, get_rule, validate_examples)

Reused largely verbatim from the prior plan (Tasks 6 and the validate_examples
part of Task 8). The contract groups `validate_examples` under Authoring.

**Files:**
- Create: `rerum/mcp/tools.py`
- Modify: `rerum/tests/test_mcp_tools.py`

- [ ] **Step 1: Write the failing test**

Append to `rerum/tests/test_mcp_tools.py`:

```python
class TestAuthoringTools:
    def test_load_rules_dsl(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_load_rules

        engine = RuleEngine()
        result = tool_load_rules(
            engine,
            text='@add-zero {category=identity}: (+ ?x 0) => :x',
            format="dsl",
        )
        assert result["ok"] is True
        assert result["rules_added"] == 1
        assert "add-zero" in engine

    def test_load_rules_auto_detect_json(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_load_rules

        engine = RuleEngine()
        text = ('{"rules": [{"name": "r1", "category": "identity",'
                ' "pattern": ["a", ["?", "x"]], "skeleton": [":", "x"]}]}')
        result = tool_load_rules(engine, text=text)  # no format kwarg
        assert result["ok"] is True

    def test_add_rule_basic(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_add_rule

        engine = RuleEngine()
        result = tool_add_rule(
            engine, pattern="(a ?x)", skeleton=":x",
            name="r1", category="identity",
        )
        assert result["ok"] is True
        assert result["rule_index"] >= 0

    def test_add_rule_bad_example_raises_validation_error(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_add_rule
        from rerum.mcp.errors import MCPToolError

        engine = RuleEngine()
        with pytest.raises(MCPToolError) as exc_info:
            tool_add_rule(
                engine, pattern="(+ ?x 0)", skeleton=":x", name="bad",
                examples=[{"in": "(+ y 0)", "out": "wrong"}],
            )
        assert exc_info.value.code == "validation_error"

    def test_list_rules_filter_by_category(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_list_rules

        engine = RuleEngine.from_dsl("""
            @r1 {category=identity}: (a ?x) => :x
            @r2 {category=distributivity}: (b ?x) => :x
        """)
        result = tool_list_rules(engine, category="identity")
        assert len(result) == 1
        assert result[0]["name"] == "r1"

    def test_get_rule_unknown_raises(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_get_rule
        from rerum.mcp.errors import MCPToolError

        engine = RuleEngine()
        with pytest.raises(MCPToolError) as exc_info:
            tool_get_rule(engine, name="nonexistent")
        assert exc_info.value.code == "unknown_rule"

    def test_validate_examples_returns_failures_as_data(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_validate_examples

        engine = RuleEngine()
        engine.add_rule(
            pattern=["a", ["?", "x"]], skeleton=[":", "x"], name="bad",
            examples=[{"in": "(a 1)", "out": "wrong"}],
            validate_examples=False,
        )
        result = tool_validate_examples(engine)
        assert result["ok"] is False
        assert result["errors"][0]["rule_name"] == "bad"
```

- [ ] **Step 2: Verify failure**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_mcp_tools.py::TestAuthoringTools -v 2>&1 | tail -15
```

Expected FAIL: ImportError on `rerum.mcp.tools`.

- [ ] **Step 3: Create `rerum/mcp/tools.py` (authoring section)**

```python
"""MCP tool handlers.

Each ``tool_*`` function is a thin orchestration over the engine. Tool
handlers contain NO domain logic and NO domain operator literals: they
validate inputs, call the engine, and shape the response. The engine is
the GENERAL rewriting engine; rules and theories are loaded as data.
Errors raise ``MCPToolError`` with a stable ``code``.
"""

from typing import Any, Callable, Dict, List, Optional

from rerum.engine import (
    ExampleValidationError,
    format_sexpr,
    parse_sexpr,
)
from rerum.mcp.errors import MCPToolError


# =====================================================================
# Authoring
# =====================================================================

def tool_load_rules(engine, *, text: str, format: str = "auto",
                    validate_examples: bool = True) -> Dict[str, Any]:
    """Bulk-load rules from DSL or JSON text.

    ``format`` is auto-detected when ``"auto"``/None: a leading ``{`` means
    JSON, otherwise DSL.
    """
    if format in ("auto", None):
        format = "json" if text.lstrip().startswith("{") else "dsl"

    rules_before = len(engine._rules)
    try:
        if format == "json":
            engine.load_rules_from_json(text, validate_examples=validate_examples)
        elif format == "dsl":
            engine.load_dsl(text, validate_examples=validate_examples)
        else:
            raise MCPToolError(
                "parse_error",
                f"unknown format {format!r}; use 'dsl' or 'json'",
                details={"format": format},
            )
    except ExampleValidationError as exc:
        raise MCPToolError(
            "validation_error", str(exc),
            details={"rule_name": getattr(exc, "rule_name", None),
                     "example": getattr(exc, "example", None)},
            cause=exc,
        ) from exc
    except ValueError as exc:
        raise MCPToolError("parse_error", str(exc), cause=exc) from exc

    return {"ok": True, "rules_added": len(engine._rules) - rules_before}


def tool_add_rule(engine, *, pattern: str, skeleton: str,
                  name: Optional[str] = None,
                  description: Optional[str] = None,
                  category: Optional[str] = None,
                  reasoning: Optional[str] = None,
                  examples: Optional[List[Dict[str, Any]]] = None,
                  priority: int = 0,
                  condition: Optional[str] = None,
                  tags: Optional[List[str]] = None,
                  validate_examples: bool = True) -> Dict[str, Any]:
    """Add a single rule with full metadata."""
    try:
        pat = parse_sexpr(pattern)
        skel = parse_sexpr(skeleton)
        cond = parse_sexpr(condition) if condition else None
    except Exception as exc:
        raise MCPToolError(
            "parse_error",
            f"failed to parse pattern/skeleton/condition: {exc}",
            cause=exc,
        ) from exc

    try:
        engine.add_rule(
            pattern=pat, skeleton=skel, name=name, description=description,
            priority=priority, condition=cond, tags=tags, category=category,
            reasoning=reasoning, examples=examples,
            validate_examples=validate_examples,
        )
    except ExampleValidationError as exc:
        raise MCPToolError(
            "validation_error", str(exc),
            details={"rule_name": getattr(exc, "rule_name", None),
                     "example": getattr(exc, "example", None)},
            cause=exc,
        ) from exc

    rule_index = len(engine._rules) - 1
    if name and name in engine._rule_names:
        rule_index = engine._rule_names[name]
    return {"ok": True, "rule_index": rule_index}


def tool_list_rules(engine, *, category: Optional[str] = None,
                    tag: Optional[str] = None) -> List[Dict[str, Any]]:
    """Lightweight summary of every rule, with optional filters."""
    out: List[Dict[str, Any]] = []
    for idx, meta in enumerate(engine._metadata):
        if category is not None and meta.category != category:
            continue
        if tag is not None and tag not in (meta.tags or []):
            continue
        out.append({
            "rule_index": idx,
            "name": meta.name,
            "category": meta.category,
            "description": meta.description,
            "bidirectional": meta.bidirectional,
            "direction": meta.direction,
            "priority": meta.priority,
            "tags": list(meta.tags or []),
        })
    return out


def tool_get_rule(engine, *, rule_index: Optional[int] = None,
                  name: Optional[str] = None) -> Dict[str, Any]:
    """Full details for one rule. Pass either ``rule_index`` or ``name``."""
    if rule_index is None and name is None:
        raise MCPToolError(
            "parse_error", "tool_get_rule requires rule_index or name")

    if name is not None:
        if name not in engine._rule_names:
            raise MCPToolError(
                "unknown_rule", f"no rule named {name!r}",
                details={"name": name,
                         "available": list(engine._rule_names.keys())},
            )
        rule_index = engine._rule_names[name]

    if rule_index < 0 or rule_index >= len(engine._rules):
        raise MCPToolError(
            "unknown_rule",
            f"rule_index {rule_index} out of range",
            details={"rule_index": rule_index},
        )

    pattern, skeleton = engine._rules[rule_index]
    meta = engine._metadata[rule_index]
    return {
        "rule_index": rule_index,
        "name": meta.name,
        "description": meta.description,
        "pattern": format_sexpr(pattern),
        "skeleton": format_sexpr(skeleton),
        "category": meta.category,
        "reasoning": meta.reasoning,
        "examples": meta.examples or [],
        "priority": meta.priority,
        "condition": format_sexpr(meta.condition) if meta.condition else None,
        "tags": list(meta.tags or []),
        "bidirectional": meta.bidirectional,
        "direction": meta.direction,
        "fwd_label": meta.fwd_label,
        "rev_label": meta.rev_label,
        "extra": dict(meta.extra or {}),
    }


def tool_validate_examples(engine) -> Dict[str, Any]:
    """Validate every example in the engine; return errors as data."""
    from rerum.engine import _validate_example
    errors: List[Dict[str, Any]] = []
    for (pat, skel), meta in zip(engine._rules, engine._metadata):
        if not meta.examples:
            continue
        for example in meta.examples:
            direction = example.get("direction", "fwd")
            if meta.bidirectional and direction != meta.direction:
                continue
            try:
                _validate_example(pat, skel, meta, example,
                                  engine._fold_funcs or {})
            except ExampleValidationError as exc:
                errors.append({
                    "rule_name": getattr(exc, "rule_name", meta.name),
                    "example": getattr(exc, "example", example),
                    "message": str(exc),
                })
    return {"ok": len(errors) == 0, "errors": errors}
```

Note: the `_validate_example` helper and `engine._metadata`/`engine._rules`/
`engine._rule_names` field names are pinned to the engine as of v0.7; confirm
against the actual private API when implementing and adjust the access (the
prior plan used these same names successfully).

- [ ] **Step 4: Run tests (expected PASS)**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_mcp_tools.py::TestAuthoringTools -v 2>&1 | tail -15
```

Expected PASS: 7 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/spinoza/github/repos/rerum && git add rerum/mcp/tools.py rerum/tests/test_mcp_tools.py
git commit -m "$(cat <<'EOF'
feat(mcp): authoring tools (load_rules, add_rule, list_rules, get_rule, validate_examples)

Thin orchestration over the engine, no domain logic. Format
auto-detection on load_rules; parse_sexpr for add_rule strings;
ExampleValidationError mapped to validation_error. validate_examples
returns failures as data.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Persistence (save_ruleset, load_ruleset, list_rulesets, load_theory)

This is delta 5 (NEW). A git-friendly file store under a rules directory (default
`.rerum/rules/`), files `<name>.json`. Built on `to_json`/`load_rules_from_json`.
`load_theory` loads `<name>.theory.json` and applies it via
`engine.with_theory(Theory.from_json(...))` (Phase 2). All data; no domain logic.

**Files:**
- Create: `rerum/mcp/persistence.py`
- Modify: `rerum/mcp/tools.py`
- Create: `rerum/tests/test_mcp_persistence.py`

- [ ] **Step 1: Write the failing test**

Create `rerum/tests/test_mcp_persistence.py`:

```python
"""Tests for the file-backed rule set and theory store."""

import json
import pytest


class TestRuleStore:
    def test_save_then_list_then_load_roundtrip(self, tmp_path):
        from rerum import RuleEngine
        from rerum.mcp.persistence import RuleStore

        store = RuleStore(root=str(tmp_path))
        engine = RuleEngine.from_dsl(
            '@add-zero {category=identity}: (+ ?x 0) => :x'
        )

        save_res = store.save_ruleset(engine, "algebra")
        assert save_res["ok"] is True
        assert (tmp_path / "rules" / "algebra.json").exists()

        names = store.list_rulesets()
        assert "algebra" in [r["name"] for r in names]

        fresh = RuleEngine()
        load_res = store.load_ruleset(fresh, "algebra")
        assert load_res["ok"] is True
        assert "add-zero" in fresh

    def test_load_missing_ruleset_raises_not_found(self, tmp_path):
        from rerum import RuleEngine
        from rerum.mcp.persistence import RuleStore
        from rerum.mcp.errors import MCPToolError

        store = RuleStore(root=str(tmp_path))
        with pytest.raises(MCPToolError) as exc:
            store.load_ruleset(RuleEngine(), "nope")
        assert exc.value.code == "not_found"

    def test_name_sanitization_rejects_traversal(self, tmp_path):
        from rerum import RuleEngine
        from rerum.mcp.persistence import RuleStore
        from rerum.mcp.errors import MCPToolError

        store = RuleStore(root=str(tmp_path))
        with pytest.raises(MCPToolError) as exc:
            store.save_ruleset(RuleEngine(), "../escape")
        assert exc.value.code == "parse_error"

    def test_load_theory_applies_to_engine(self, tmp_path):
        from rerum import RuleEngine
        from rerum.mcp.persistence import RuleStore

        store = RuleStore(root=str(tmp_path))
        theory_dir = tmp_path / "rules"
        theory_dir.mkdir(parents=True, exist_ok=True)
        (theory_dir / "arithmetic.theory.json").write_text(json.dumps({
            "+": {"ac": True, "identity": 0},
            "*": {"ac": True, "identity": 1, "annihilator": 0},
        }))

        engine = RuleEngine()
        res = store.load_theory(engine, "arithmetic")
        assert res["ok"] is True
        # The engine now carries the theory (Phase 2 with_theory wiring).
        assert engine._theory is not None
```

- [ ] **Step 2: Verify failure**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_mcp_persistence.py -v 2>&1 | tail -15
```

Expected FAIL: ImportError on `rerum.mcp.persistence`.

- [ ] **Step 3: Create `rerum/mcp/persistence.py`**

```python
"""File-backed rule set and theory store (git-friendly).

Rule sets are stored as ``<root>/rules/<name>.json`` (the same JSON shape
``RuleEngine.to_json`` emits and ``load_rules_from_json`` consumes).
Theories are stored as ``<root>/rules/<name>.theory.json`` (the Phase 2
Theory data shape). Everything is DATA: this module contains no domain
logic and no domain operator literals.
"""

import json
import os
import re
from typing import Any, Dict, List

from rerum.mcp.errors import MCPToolError


_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


def _safe_name(name: str) -> str:
    """Reject path traversal and unsafe characters in a ruleset name."""
    if not name or not _SAFE_NAME.match(name) or name.startswith("."):
        raise MCPToolError(
            "parse_error",
            f"invalid ruleset name {name!r}; use [A-Za-z0-9._-], "
            "no leading dot, no path separators",
            details={"name": name},
        )
    return name


class RuleStore:
    """Git-friendly file store for rule sets and theories.

    Default root is ``.rerum`` in the server's working directory; rule
    files live under ``<root>/rules/``.
    """

    def __init__(self, root: str = ".rerum"):
        self.root = root
        self.rules_dir = os.path.join(root, "rules")

    def _ensure_dir(self) -> None:
        os.makedirs(self.rules_dir, exist_ok=True)

    def _ruleset_path(self, name: str) -> str:
        return os.path.join(self.rules_dir, f"{_safe_name(name)}.json")

    def _theory_path(self, name: str) -> str:
        return os.path.join(self.rules_dir, f"{_safe_name(name)}.theory.json")

    def save_ruleset(self, engine, name: str) -> Dict[str, Any]:
        """Write the engine's rules to ``<name>.json`` via ``to_json``."""
        path = self._ruleset_path(name)
        self._ensure_dir()
        text = engine.to_json(name=name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return {"ok": True, "name": name, "path": path,
                "rules_saved": len(engine._rules)}

    def load_ruleset(self, engine, name: str,
                     validate_examples: bool = True) -> Dict[str, Any]:
        """Load ``<name>.json`` into the engine via ``load_rules_from_json``."""
        path = self._ruleset_path(name)
        if not os.path.exists(path):
            raise MCPToolError(
                "not_found", f"no ruleset named {name!r}",
                details={"name": name, "path": path,
                         "available": [r["name"] for r in self.list_rulesets()]},
            )
        before = len(engine._rules)
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        engine.load_rules_from_json(text, validate_examples=validate_examples)
        return {"ok": True, "name": name,
                "rules_added": len(engine._rules) - before}

    def list_rulesets(self) -> List[Dict[str, Any]]:
        """List saved rule sets (``*.json``, excluding ``*.theory.json``)."""
        if not os.path.isdir(self.rules_dir):
            return []
        out: List[Dict[str, Any]] = []
        for fn in sorted(os.listdir(self.rules_dir)):
            if fn.endswith(".theory.json") or not fn.endswith(".json"):
                continue
            name = fn[:-len(".json")]
            path = os.path.join(self.rules_dir, fn)
            count = None
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    count = len(json.load(fh).get("rules", []))
            except Exception:
                pass
            out.append({"name": name, "path": path, "rules": count})
        return out

    def load_theory(self, engine, name: str) -> Dict[str, Any]:
        """Load ``<name>.theory.json`` and apply it via ``with_theory``.

        The Theory is Phase 2 data declaring which operators are AC and
        their units. No operator name is special-cased here.
        """
        path = self._theory_path(name)
        if not os.path.exists(path):
            raise MCPToolError(
                "not_found", f"no theory named {name!r}",
                details={"name": name, "path": path},
            )
        from rerum.normalize import Theory
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        theory = Theory.from_json(text)
        engine.with_theory(theory)
        return {"ok": True, "name": name, "path": path}
```

- [ ] **Step 4: Add the persistence tool wrappers to `rerum/mcp/tools.py`**

The wrappers take a `RuleStore` so the server owns the root path:

```python
# =====================================================================
# Persistence (file-backed rule sets and theories)
# =====================================================================

def tool_save_ruleset(engine, store, *, name: str) -> Dict[str, Any]:
    """Persist the engine's current rules under ``name``."""
    return store.save_ruleset(engine, name)


def tool_load_ruleset(engine, store, *, name: str,
                      validate_examples: bool = True) -> Dict[str, Any]:
    """Load a saved rule set into the engine."""
    return store.load_ruleset(engine, name, validate_examples=validate_examples)


def tool_list_rulesets(engine, store) -> Dict[str, Any]:
    """List the rule sets available in the store."""
    return {"rulesets": store.list_rulesets()}


def tool_load_theory(engine, store, *, name: str) -> Dict[str, Any]:
    """Load a saved theory (``<name>.theory.json``) and apply it."""
    return store.load_theory(engine, name)
```

- [ ] **Step 5: Run tests (expected PASS)**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_mcp_persistence.py -v 2>&1 | tail -10
```

Expected PASS: 4 passed.

- [ ] **Step 6: Commit**

```bash
cd /home/spinoza/github/repos/rerum && git add rerum/mcp/persistence.py rerum/mcp/tools.py rerum/tests/test_mcp_persistence.py
git commit -m "$(cat <<'EOF'
feat(mcp): persistence tools (save/load/list rulesets, load_theory)

RuleStore is a git-friendly file store under .rerum/rules/, files
<name>.json for rule sets and <name>.theory.json for theories. Built on
to_json / load_rules_from_json / Theory.from_json + with_theory. Name
sanitization rejects traversal. Everything loaded as data; no domain
logic.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Applying tools (simplify, apply_once, equivalents, prove_equal, minimize)

Reused from the prior plan (Task 7), with two deltas: each return adds a `prose`
field (via `assemble_trace`), and the trace is the situated trace. `apply_once`
is added (the contract lists it). The `_stats` and `_parse` helpers stay.

**Files:**
- Modify: `rerum/mcp/tools.py`
- Modify: `rerum/tests/test_mcp_tools.py`

- [ ] **Step 1: Write the failing test**

Append to `rerum/tests/test_mcp_tools.py`:

```python
class TestApplyingTools:
    def test_simplify_returns_situated_trace_and_prose(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_simplify

        engine = RuleEngine.from_dsl(
            '@add-zero {category=identity}: (+ ?x 0) => :x'
        )
        result = tool_simplify(engine, expr="(+ y 0)")
        assert result["result"] == "y"
        assert result["converged"] is True
        step = result["trace"]["steps"][0]
        assert step["rule_id"] == "add-zero"
        assert step["kind"] == "rule"
        assert "before_root" in step
        assert isinstance(result["trace"]["prose"], str)
        assert "prose" in result["trace"]

    def test_apply_once_returns_single_step(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_apply_once

        engine = RuleEngine.from_dsl("""
            @r1: (a ?x) => (b :x)
            @r2: (b ?x) => (c :x)
        """)
        result = tool_apply_once(engine, expr="(a y)")
        # apply_once does one rewrite, returning matched rule metadata.
        assert result["result"] == "(b y)"
        assert result["trace"]["total_steps"] == 1

    def test_equivalents_basic(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_equivalents

        engine = RuleEngine.from_dsl('@commute: (+ ?x ?y) <=> (+ :y :x)')
        result = tool_equivalents(engine, expr="(+ a b)", max_depth=3)
        assert "(+ a b)" in result["forms"]
        assert "(+ b a)" in result["forms"]

    def test_prove_equal_proven_carries_prose(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_prove_equal

        engine = RuleEngine.from_dsl('@commute: (+ ?x ?y) <=> (+ :y :x)')
        result = tool_prove_equal(
            engine, expr_a="(+ a b)", expr_b="(+ b a)", max_depth=3
        )
        assert result["proven"] is True
        assert "prose" in result

    def test_minimize_basic(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_minimize

        engine = RuleEngine.from_dsl(
            '@add-zero {category=identity}: (+ ?x 0) => :x'
        )
        result = tool_minimize(engine, expr="(+ y 0)", metric="size")
        assert result["best"] == "y"
        assert "prose" in result
```

- [ ] **Step 2: Verify failure**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_mcp_tools.py::TestApplyingTools -v 2>&1 | tail -15
```

Expected FAIL: ImportError on the applying tool functions.

- [ ] **Step 3: Add the applying tools to `rerum/mcp/tools.py`**

```python
# =====================================================================
# Applying
# =====================================================================

def _stats(engine, recorder) -> Dict[str, Any]:
    return {
        "duration_ms": round(recorder.duration_ms, 3),
        "rules_in_engine": len(engine._rules),
    }


def _parse(expr: str):
    try:
        return parse_sexpr(expr)
    except Exception as exc:
        raise MCPToolError(
            "parse_error", f"failed to parse expr: {exc}", cause=exc
        ) from exc


def tool_simplify(engine, *, expr: str, strategy: str = "exhaustive",
                  max_steps: int = 1000,
                  groups: Optional[List[str]] = None) -> Dict[str, Any]:
    """Simplify ``expr`` to fixpoint; return result + situated trace + prose."""
    from rerum.mcp.trace import assemble_trace, trace_recorder

    parsed = _parse(expr)
    initial_str = format_sexpr(parsed)
    try:
        with trace_recorder(engine, initial=parsed) as recorder:
            result = engine.simplify(
                parsed, strategy=strategy, max_steps=max_steps, groups=groups)
    except Exception as exc:
        raise MCPToolError("internal_error", f"simplify failed: {exc}",
                           cause=exc) from exc

    final_str = format_sexpr(result)
    trace = assemble_trace(initial=initial_str, final=final_str,
                           recorder=recorder)
    return {"result": final_str, "converged": True, "trace": trace,
            "stats": _stats(engine, recorder)}


def tool_apply_once(engine, *, expr: str,
                    groups: Optional[List[str]] = None) -> Dict[str, Any]:
    """Apply the first matching rule once; return result + situated trace + prose."""
    from rerum.mcp.trace import assemble_trace, trace_recorder

    parsed = _parse(expr)
    initial_str = format_sexpr(parsed)
    try:
        with trace_recorder(engine, initial=parsed) as recorder:
            outcome = engine.apply_once(parsed, groups=groups)
    except Exception as exc:
        raise MCPToolError("internal_error", f"apply_once failed: {exc}",
                           cause=exc) from exc

    # apply_once returns the rewritten expression (and may return matched
    # metadata depending on the engine API); take the expression form.
    result_expr = outcome[0] if isinstance(outcome, tuple) else outcome
    final_str = format_sexpr(result_expr)
    trace = assemble_trace(initial=initial_str, final=final_str,
                           recorder=recorder)
    return {"result": final_str,
            "changed": final_str != initial_str,
            "trace": trace, "stats": _stats(engine, recorder)}


def tool_equivalents(engine, *, expr: str, max_depth: int = 10,
                     max_count: int = 100, strategy: str = "bfs",
                     include_unidirectional: bool = False,
                     groups: Optional[List[str]] = None) -> Dict[str, Any]:
    """Enumerate expressions equivalent to ``expr``."""
    from rerum.mcp.trace import trace_recorder

    parsed = _parse(expr)
    try:
        with trace_recorder(engine, initial=parsed) as recorder:
            forms = list(engine.equivalents(
                parsed, max_depth=max_depth, max_count=max_count,
                strategy=strategy,
                include_unidirectional=include_unidirectional, groups=groups))
    except Exception as exc:
        raise MCPToolError("internal_error", f"equivalents failed: {exc}",
                           cause=exc) from exc

    return {"forms": [format_sexpr(f) for f in forms],
            "total_count": len(forms), "stats": _stats(engine, recorder)}


def tool_prove_equal(engine, *, expr_a: str, expr_b: str, max_depth: int = 10,
                     max_expressions: Optional[int] = None,
                     include_unidirectional: bool = False,
                     trace: bool = True,
                     groups: Optional[List[str]] = None) -> Dict[str, Any]:
    """Prove ``expr_a`` and ``expr_b`` equivalent via bidirectional BFS.

    On success returns the meeting form, both labeled paths (lists of
    situated step dicts from EqualityProof.path_a/path_b), and a prose
    rendering of the combined proof.
    """
    from rerum.mcp.trace import assemble_trace, render_prose, step_to_dict, _Recorder

    parsed_a = _parse(expr_a)
    parsed_b = _parse(expr_b)
    try:
        proof = engine.prove_equal(
            parsed_a, parsed_b, max_depth=max_depth,
            max_expressions=max_expressions, trace=trace,
            include_unidirectional=include_unidirectional, groups=groups)
    except Exception as exc:
        raise MCPToolError("internal_error", f"prove_equal failed: {exc}",
                           cause=exc) from exc

    if proof is None:
        return {"proven": False, "prose": "No proof found.",
                "stats": {"rules_in_engine": len(engine._rules)}}

    out: Dict[str, Any] = {
        "proven": True,
        "common_form": format_sexpr(proof.common),
        "depth_a": proof.depth_a,
        "depth_b": proof.depth_b,
        "stats": {"rules_in_engine": len(engine._rules)},
    }
    # Phase 1: path_a/path_b are List[RewriteStep] (situated).
    if trace and proof.path_a is not None:
        out["path_a"] = [step_to_dict(s) for s in proof.path_a]
    if trace and proof.path_b is not None:
        out["path_b"] = [step_to_dict(s) for s in proof.path_b]
    # Prose over a trace built from the two paths.
    rec = _Recorder()
    try:
        from rerum.trace import RewriteTrace
        rec.trace = RewriteTrace(initial=parsed_a)
        for s in list(proof.path_a or []) + list(reversed(proof.path_b or [])):
            rec.trace.add_step(s)
        out["prose"] = render_prose(rec)
    except Exception:
        out["prose"] = ""
    return out


def tool_minimize(engine, *, expr: str, metric: str = "size",
                  op_costs: Optional[Dict[str, float]] = None,
                  max_depth: int = 10, max_count: int = 10000,
                  include_unidirectional: bool = True,
                  groups: Optional[List[str]] = None) -> Dict[str, Any]:
    """Find the minimum-cost equivalent of ``expr``; include the derivation prose."""
    from rerum.mcp.trace import assemble_trace, render_prose, step_to_dict, _Recorder

    parsed = _parse(expr)
    kwargs: Dict[str, Any] = {
        "max_depth": max_depth, "max_count": max_count,
        "include_unidirectional": include_unidirectional, "groups": groups,
    }
    if op_costs is not None:
        kwargs["op_costs"] = op_costs
    else:
        kwargs["metric"] = metric

    try:
        opt = engine.minimize(parsed, **kwargs)
    except Exception as exc:
        raise MCPToolError("internal_error", f"minimize failed: {exc}",
                           cause=exc) from exc

    out: Dict[str, Any] = {
        "original": format_sexpr(opt.original),
        "original_cost": opt.original_cost,
        "best": format_sexpr(opt.expr),
        "best_cost": opt.cost,
        "improvement_ratio": opt.improvement_ratio,
        "expressions_checked": opt.expressions_checked,
        "stats": {"rules_in_engine": len(engine._rules)},
    }
    # Phase 1: OptimizationResult.derivation is the labeled path.
    rec = _Recorder()
    derivation = getattr(opt, "derivation", None)
    if derivation is not None:
        try:
            from rerum.trace import RewriteTrace
            rec.trace = RewriteTrace(initial=parsed)
            for s in derivation:
                rec.trace.add_step(s)
            out["derivation"] = [step_to_dict(s) for s in derivation]
            out["prose"] = render_prose(rec)
        except Exception:
            out["prose"] = ""
    else:
        out["prose"] = ""
    return out
```

Note: the exact `apply_once` / `prove_equal` / `minimize` return shapes
(`EqualityProof.path_a` as `List[RewriteStep]`, `OptimizationResult.derivation`)
are pinned by Phase 1. Adjust the access if the landed Phase 1 names differ; the
failing tests in Step 1 assert only the externally-visible contract.

- [ ] **Step 4: Run tests (expected PASS)**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_mcp_tools.py::TestApplyingTools -v 2>&1 | tail -15
```

Expected PASS: 5 passed.

- [ ] **Step 5: Run full suite**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest 2>&1 | tail -3
```

- [ ] **Step 6: Commit**

```bash
cd /home/spinoza/github/repos/rerum && git add rerum/mcp/tools.py rerum/tests/test_mcp_tools.py
git commit -m "$(cat <<'EOF'
feat(mcp): applying tools with situated trace + prose

simplify, apply_once, equivalents, prove_equal, minimize. Each rewriting
tool returns the Phase 1 situated trace and a prose rendering via
rerum.training.to_prose. prove_equal returns both labeled situated paths;
minimize returns the labeled derivation.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Goal solving (solve_goal wraps engine solve())

This is delta 2 (NEW `solve_goal`). It wraps the Phase 3 engine `solve()`. The
caller describes the goal as DATA, e.g. `{"op_free": ["int", "lim"]}`, which the
tool compiles to a predicate using the general `contains_op` helper. NO operator
literal is hardcoded in the tool; the operator names come from the caller's
`goal`. Returns `{result, found, trace, prose, stats}`.

**Files:**
- Modify: `rerum/mcp/tools.py`
- Create: `rerum/tests/test_mcp_solve_goal.py`

- [ ] **Step 1: Write the failing test**

Create `rerum/tests/test_mcp_solve_goal.py`:

```python
"""Tests for solve_goal (wraps engine.solve over a caller-described goal)."""

import pytest


class TestSolveGoal:
    def test_op_free_goal_compiles_and_solves(self):
        # Toy non-confluent rule set where a wrapper op must be eliminated.
        # The goal {"op_free": ["w"]} means "no 'w' operator remains".
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_solve_goal

        engine = RuleEngine.from_dsl("""
            @unwrap: (w ?x) => :x
        """)
        result = tool_solve_goal(
            engine, expr="(w (w a))", goal={"op_free": ["w"]}, max_nodes=1000
        )
        assert result["found"] is True
        assert result["result"] == "a"
        assert "trace" in result
        assert isinstance(result["prose"], str)

    def test_unreachable_goal_reports_not_found(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_solve_goal

        engine = RuleEngine()  # no rules, cannot remove anything
        result = tool_solve_goal(
            engine, expr="(w a)", goal={"op_free": ["w"]}, max_nodes=50
        )
        assert result["found"] is False

    def test_unknown_goal_kind_is_parse_error(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_solve_goal
        from rerum.mcp.errors import MCPToolError

        engine = RuleEngine()
        with pytest.raises(MCPToolError) as exc:
            tool_solve_goal(engine, expr="(w a)", goal={"bogus": 1})
        assert exc.value.code == "parse_error"
```

- [ ] **Step 2: Verify failure**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_mcp_solve_goal.py -v 2>&1 | tail -15
```

Expected FAIL: ImportError on `tool_solve_goal`.

- [ ] **Step 3: Add `tool_solve_goal` to `rerum/mcp/tools.py`**

```python
# =====================================================================
# Goal solving (wraps engine.solve)
# =====================================================================

def _compile_goal(goal: Dict[str, Any]) -> Callable[[Any], bool]:
    """Compile a caller-described goal (DATA) into a predicate.

    Supported goal kinds (all general; operator names come from the
    caller, never hardcoded):
      {"op_free": ["op1", "op2", ...]}  -> True when none of those operators
                                           remain in the expression.

    The operator names are the caller's data; this function special-cases
    NO operator. Unknown goal kinds raise parse_error.
    """
    from rerum.solve import contains_op

    if "op_free" in goal:
        ops = set(goal["op_free"])
        return lambda e: not contains_op(e, ops)

    raise MCPToolError(
        "parse_error",
        f"unknown goal kind; supported: 'op_free'. Got keys: {sorted(goal)}",
        details={"goal": goal},
    )


def tool_solve_goal(engine, *, expr: str, goal: Dict[str, Any],
                    cost_fn: Optional[str] = None, max_nodes: int = 10000,
                    fresh_vars: bool = True,
                    normalize_between: bool = True) -> Dict[str, Any]:
    """Goal-directed search via engine.solve over a caller-described goal.

    Returns {result, found, trace, prose, stats}. The goal is DATA
    (e.g. {"op_free": ["int", "lim"]}); the tool contains no domain logic.
    """
    from rerum.mcp.trace import assemble_trace, _Recorder, render_prose, step_to_dict

    parsed = _parse(expr)
    initial_str = format_sexpr(parsed)
    predicate = _compile_goal(goal)

    try:
        sr = engine.solve(
            parsed, predicate, max_nodes=max_nodes,
            fresh_vars=fresh_vars, normalize_between=normalize_between)
    except Exception as exc:
        raise MCPToolError("internal_error", f"solve failed: {exc}",
                           cause=exc) from exc

    final_str = format_sexpr(sr.solution) if sr.solution is not None else initial_str
    # sr.derivation is a RewriteTrace (Phase 3 SolveResult).
    rec = _Recorder()
    rec.trace = sr.derivation
    if rec.trace is not None:
        try:
            rec.steps = [step_to_dict(s) for s in rec.trace.steps]
        except Exception:
            rec.steps = []
    trace = assemble_trace(initial=initial_str, final=final_str, recorder=rec)

    return {
        "result": final_str,
        "found": sr.found,
        "explored": sr.explored,
        "trace": trace,
        "prose": trace.get("prose", ""),
        "stats": {"rules_in_engine": len(engine._rules),
                  "explored": sr.explored},
    }
```

- [ ] **Step 4: Run tests (expected PASS)**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_mcp_solve_goal.py -v 2>&1 | tail -10
```

Expected PASS: 3 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/spinoza/github/repos/rerum && git add rerum/mcp/tools.py rerum/tests/test_mcp_solve_goal.py
git commit -m "$(cat <<'EOF'
feat(mcp): solve_goal tool wrapping engine.solve()

Compiles a caller-described goal (DATA, e.g. {"op_free": ["int","lim"]})
into a predicate via the general contains_op helper, runs engine.solve
(Phase 3 best-first search), returns {result, found, trace, prose,
stats}. No operator is hardcoded; operator names come from the caller.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Agentic loop (solve_assisted, renamed from the prior solve)

This is delta 6. The prior plan's full resolver-callback design is kept verbatim,
just renamed `solve_assisted` and returning the situated trace + prose. The
resolver factory in `solver.py` is reused unchanged.

**Files:**
- Create: `rerum/mcp/solver.py`
- Modify: `rerum/mcp/tools.py`
- Create: `rerum/tests/test_mcp_solve_assisted.py`

- [ ] **Step 1: Write the failing test**

Create `rerum/tests/test_mcp_solve_assisted.py`:

```python
"""Tests for solve_assisted (the agentic LLM-resolver loop) with mocked sampling."""

import pytest


def make_sampler(responses):
    iterator = iter(responses)

    def sample(prompt):
        try:
            return next(iterator)
        except StopIteration:
            return "NONE"

    return sample


class TestSolveAssisted:
    def test_no_resolver_needed(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_solve_assisted

        engine = RuleEngine.from_dsl(
            '@add-zero {category=identity}: (+ ?x 0) => :x'
        )
        calls = [0]

        def sampler(prompt):
            calls[0] += 1
            return "NONE"

        result = tool_solve_assisted(engine, expr="(+ y 0)", sampler=sampler)
        assert result["result"] == "y"
        assert result["resolver_calls"] == 0
        assert result["inferred_rules"] == []
        assert calls[0] == 0
        assert "prose" in result["trace"]

    def test_resolver_supplies_rule_with_provenance(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_solve_assisted

        engine = RuleEngine()
        sampler = make_sampler(["@foo-id {category=identity}: (foo ?x) => :x"])
        result = tool_solve_assisted(engine, expr="(foo bar)", sampler=sampler)

        assert result["result"] == "bar"
        assert result["resolver_calls"] == 1
        assert result["inferred_rules"][0]["name"] == "foo-id"
        assert any(
            s.get("provenance") == "llm-inferred"
            for s in result["trace"]["steps"]
        )

    def test_resolver_cap_terminates(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_solve_assisted

        engine = RuleEngine()

        def sampler(prompt):
            return "(zzz ?y) => :y"  # never matches (foo bar)

        result = tool_solve_assisted(
            engine, expr="(foo bar)", sampler=sampler, max_resolver_calls=3)
        assert result["resolver_calls"] >= 3
        assert "termination" in result
        assert result["termination"]["reason"] in (
            "resolver_budget_exhausted", "resolver_loop")

    def test_sampling_unsupported_when_no_sampler(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_solve_assisted

        engine = RuleEngine()
        # No sampler installed and goal needs one: behaves like simplify and
        # reports sampling_unsupported if it gets stuck. With no rules,
        # the expression is unchanged; converged with no inferred rules.
        result = tool_solve_assisted(engine, expr="(foo bar)", sampler=None)
        assert result["inferred_rules"] == []
```

- [ ] **Step 2: Verify failure**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_mcp_solve_assisted.py -v 2>&1 | tail -15
```

Expected FAIL: ImportError on `tool_solve_assisted`.

- [ ] **Step 3: Create `rerum/mcp/solver.py` (reused verbatim from the prior plan)**

```python
"""LLM-resolver factory for the solve_assisted tool.

Builds a closure suitable for engine.on_no_match() that:
1. Counts calls and enforces the per-solve cap.
2. Asks the connected LLM (via the sampler callable) for a rule.
3. Parses the reply via parse_rule_line; on validation failure, retries
   once with the error in the prompt; if still failing, returns None.
4. Wraps a successful parse in Resolution(rules=..., metadata={
       provenance: "llm-inferred", via_solve: True, round: N}).
"""

from typing import Any, Callable, Dict, Optional

from rerum.engine import ExampleValidationError, parse_rule_line
from rerum.hooks import Resolution


def make_solver_resolver(sampler: Callable[[str], str], *,
                         goal: Optional[str] = None, max_calls: int = 10,
                         state: Dict[str, Any]) -> Callable[[Any, Any], Optional[Resolution]]:
    """Build a no_match resolver that delegates to ``sampler`` for new rules."""
    state.setdefault("call_count", 0)
    state.setdefault("inferred_rules", [])
    state.setdefault("last_termination", None)

    def resolver(expr, ctx):
        if state["call_count"] >= max_calls:
            state["last_termination"] = "resolver_budget_exhausted"
            return None
        state["call_count"] += 1
        round_num = state["call_count"]

        engine = ctx.engine
        prompt = _build_prompt(expr, goal, engine)

        reply = sampler(prompt)
        rule_pairs = _try_parse_rule_reply(reply)
        if rule_pairs is None:
            return None

        try:
            _validate_pairs(rule_pairs, engine)
        except ExampleValidationError as exc:
            retry_prompt = (
                prompt + f"\n\nYour previous reply produced this error: "
                         f"{exc}\nRevise and try again.")
            reply2 = sampler(retry_prompt)
            rule_pairs = _try_parse_rule_reply(reply2)
            if rule_pairs is None:
                return None
            try:
                _validate_pairs(rule_pairs, engine)
            except ExampleValidationError:
                return None

        rules_for_resolution = [(meta, [pat, skel]) for meta, pat, skel in rule_pairs]
        for meta, pat, skel in rule_pairs:
            state["inferred_rules"].append({
                "name": meta.name, "category": meta.category,
                "dsl": _rule_to_dsl(meta, pat, skel), "round": round_num})

        return Resolution(rules=rules_for_resolution, metadata={
            "provenance": "llm-inferred", "via_solve": True, "round": round_num})

    return resolver


def _build_prompt(expr, goal, engine) -> str:
    from rerum.engine import format_sexpr
    expr_str = format_sexpr(expr)
    rules_count = len(engine._rules)
    categories = sorted({m.category for m in engine._metadata if m.category})
    cats_str = ", ".join(categories) if categories else "(none)"
    return (
        "The rewrite engine is stuck. Propose ONE rewrite rule that "
        "would help.\n\n"
        f"Goal: {goal or '(no goal specified)'}\n"
        f"Stuck at: {expr_str}\n"
        f"Rules currently in engine: {rules_count} (categories: {cats_str}).\n\n"
        "Reply with a single rule in DSL format, e.g.:\n"
        "  @my-rule {category=identity}: (foo ?x) => :x\n\n"
        "If you cannot propose a useful rule, reply: NONE")


def _try_parse_rule_reply(reply: str):
    if not reply:
        return None
    text = reply.strip()
    if text.upper() == "NONE":
        return None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            try:
                pairs = parse_rule_line(stripped)
            except Exception:
                return None
            return pairs or None
    return None


def _validate_pairs(rule_pairs, engine) -> None:
    from rerum.engine import _validate_example
    for meta, pat, skel in rule_pairs:
        if not meta.examples:
            continue
        for example in meta.examples:
            direction = example.get("direction", "fwd")
            if meta.bidirectional and direction != meta.direction:
                continue
            _validate_example(pat, skel, meta, example, engine._fold_funcs or {})


def _rule_to_dsl(meta, pat, skel) -> str:
    from rerum.engine import format_sexpr
    name_part = f"@{meta.name}" if meta.name else ""
    if meta.priority:
        name_part += f"[{meta.priority}]"
    if meta.description:
        name_part += f' "{meta.description}"'
    if meta.category:
        name_part += f" {{category={meta.category}}}"
    if name_part:
        name_part += ": "
    arrow = "<=>" if meta.bidirectional else "=>"
    return f"{name_part}{format_sexpr(pat)} {arrow} {format_sexpr(skel)}"
```

- [ ] **Step 4: Add `tool_solve_assisted` to `rerum/mcp/tools.py`**

```python
# =====================================================================
# Agentic loop (solve_assisted; renamed from the prior 'solve')
# =====================================================================

def tool_solve_assisted(engine, *, expr: str, sampler: Optional[Callable] = None,
                        goal: Optional[str] = None, max_depth: int = 20,
                        max_resolver_calls: int = 10,
                        strategy: str = "exhaustive") -> Dict[str, Any]:
    """Directed simplify with an on_no_match LLM resolver (agentic loop).

    ``sampler`` is a callable ``str -> str``; in production it is the MCP
    sampling channel, in tests a stub. When None, no resolver is installed
    and the call behaves like simplify. Returns {result, converged, trace,
    prose, inferred_rules, resolver_calls, stats}; the trace is the Phase 1
    situated trace and carries a prose rendering.
    """
    from rerum.hooks import ResolverLoopError
    from rerum.mcp.solver import make_solver_resolver
    from rerum.mcp.trace import assemble_trace, trace_recorder

    parsed = _parse(expr)
    initial_str = format_sexpr(parsed)
    state: Dict[str, Any] = {
        "call_count": 0, "inferred_rules": [], "last_termination": None}

    resolver = None
    if sampler is not None:
        resolver = make_solver_resolver(
            sampler, goal=goal, max_calls=max_resolver_calls, state=state)
        engine.on_no_match(resolver)

    termination: Optional[Dict[str, Any]] = None
    try:
        with trace_recorder(engine, initial=parsed) as recorder:
            try:
                result = engine.simplify(parsed, strategy=strategy,
                                         max_steps=max_depth)
            except ResolverLoopError as exc:
                termination = {"reason": "resolver_loop", "detail": str(exc)}
                result = parsed
    finally:
        if resolver is not None:
            engine.off_no_match(resolver)

    if termination is None and state.get("last_termination"):
        termination = {"reason": state["last_termination"]}

    final_str = format_sexpr(result)
    converged = termination is None
    trace = assemble_trace(initial=initial_str, final=final_str,
                           recorder=recorder)

    out: Dict[str, Any] = {
        "result": final_str, "converged": converged, "trace": trace,
        "prose": trace.get("prose", ""),
        "inferred_rules": state["inferred_rules"],
        "resolver_calls": state["call_count"],
        "stats": _stats(engine, recorder),
    }
    if termination is not None:
        out["termination"] = termination
    return out
```

- [ ] **Step 5: Run tests (expected PASS)**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_mcp_solve_assisted.py -v 2>&1 | tail -15
```

Expected PASS: 4 passed.

- [ ] **Step 6: Run full suite**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest 2>&1 | tail -3
```

- [ ] **Step 7: Commit**

```bash
cd /home/spinoza/github/repos/rerum && git add rerum/mcp/solver.py rerum/mcp/tools.py rerum/tests/test_mcp_solve_assisted.py
git commit -m "$(cat <<'EOF'
feat(mcp): solve_assisted agentic loop (renamed from solve)

Directed simplify with an on_no_match resolver that calls the connected
LLM via the sampler for a rule, validates and installs it with
provenance "llm-inferred", and retries. Per-solve cap, failure codes
(sampling_unsupported, resolver_timeout, resolver_budget_exhausted,
resolver_loop). Renamed from the prior 'solve' to avoid colliding with
engine solve()/solve_goal. Returns the situated trace + prose.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Admin tools + server lifecycle + dispatch + concurrency + run_server

Reuses the prior plan's reset_engine/get_status (Task 8), server (Task 10),
concurrency guard (Task 11), and SDK wiring (Task 12). `reset_engine`'s prelude
arg is delta 1: it names a computation bundle or a combination via
`combine_preludes`; there is NO domain bundle.

**Files:**
- Modify: `rerum/mcp/tools.py`
- Create: `rerum/mcp/server.py`
- Modify: `rerum/mcp/__init__.py`
- Modify: `rerum/tests/test_mcp_smoke.py`

- [ ] **Step 1: Write the failing test**

Append to `rerum/tests/test_mcp_smoke.py`:

```python
class TestAdminTools:
    def test_reset_engine_with_computation_bundle(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_reset_engine

        engine = RuleEngine.from_dsl('@r1: (a ?x) => :x')
        result = tool_reset_engine(engine, prelude="arithmetic")
        assert result["ok"] is True
        assert engine._fold_funcs is not None
        assert "+" in engine._fold_funcs
        assert len(engine._rules) == 0

    def test_reset_engine_combo_prelude(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_reset_engine

        engine = RuleEngine()
        # A combination of computation bundles via combine_preludes.
        result = tool_reset_engine(engine, prelude=["math", "predicate"])
        assert result["ok"] is True
        assert engine._fold_funcs is not None

    def test_reset_engine_rejects_domain_bundle(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_reset_engine
        from rerum.mcp.errors import MCPToolError

        engine = RuleEngine()
        with pytest.raises(MCPToolError) as exc:
            tool_reset_engine(engine, prelude="calculus")  # not a computation bundle
        assert exc.value.code == "parse_error"


class TestServerLifecycle:
    def test_server_registers_all_tools(self):
        from rerum.mcp.server import RerumMCPServer
        srv = RerumMCPServer()
        expected = {
            "load_rules", "add_rule", "list_rules", "get_rule",
            "validate_examples",
            "save_ruleset", "load_ruleset", "list_rulesets", "load_theory",
            "simplify", "apply_once", "equivalents", "prove_equal", "minimize",
            "solve_goal", "solve_assisted",
            "reset_engine", "get_status",
        }
        assert set(srv.list_tool_names()) == expected

    def test_server_call_tool_dispatches(self):
        from rerum.mcp.server import RerumMCPServer
        srv = RerumMCPServer()
        result = srv.call_tool("load_rules", {"text": "@r1: (a ?x) => :x"})
        assert result["ok"] is True

    def test_server_unknown_tool_error(self):
        from rerum.mcp.server import RerumMCPServer
        srv = RerumMCPServer()
        result = srv.call_tool("nonexistent", {})
        assert result["error"]["code"] == "parse_error"


class TestConcurrency:
    def test_engine_busy_guard(self):
        from rerum.mcp.server import RerumMCPServer
        srv = RerumMCPServer()
        srv.engine.load_dsl('@r1: (a ?x) => :x')
        srv._busy = True
        try:
            result = srv.call_tool("simplify", {"expr": "(a y)"})
        finally:
            srv._busy = False
        assert result["error"]["code"] == "engine_busy"
```

- [ ] **Step 2: Verify failure**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_mcp_smoke.py -v 2>&1 | tail -20
```

Expected FAIL: ImportError on `tool_reset_engine` / `RerumMCPServer`.

- [ ] **Step 3: Add admin tools to `rerum/mcp/tools.py`**

```python
# =====================================================================
# Admin
# =====================================================================

# Computation bundles ONLY. No domain bundle (no calculus). The values are
# attribute names on rerum.rewriter.
_PRELUDE_BUNDLES = {
    "arithmetic": "ARITHMETIC_PRELUDE",
    "math": "MATH_PRELUDE",
    "predicate": "PREDICATE_PRELUDE",
    "full": "FULL_PRELUDE",
    "none": None,
}


def _resolve_prelude(prelude):
    """Resolve a prelude spec (DATA) to fold_funcs or None.

    ``prelude`` is a single computation-bundle name (arithmetic/math/
    predicate/full/none) or a list of names combined via combine_preludes.
    There is NO domain bundle; an unknown name is a parse_error.
    """
    from rerum import rewriter as _rw
    from rerum import combine_preludes

    names = [prelude] if isinstance(prelude, str) else list(prelude)
    resolved = []
    for nm in names:
        if nm not in _PRELUDE_BUNDLES:
            raise MCPToolError(
                "parse_error",
                f"unknown prelude {nm!r}; valid computation bundles: "
                f"{sorted(_PRELUDE_BUNDLES)} (there is no domain bundle)",
                details={"prelude": nm})
        attr = _PRELUDE_BUNDLES[nm]
        if attr is not None:
            resolved.append(getattr(_rw, attr))
    if not resolved:
        return None
    if len(resolved) == 1:
        return resolved[0]
    return combine_preludes(*resolved)


def tool_reset_engine(engine, *, prelude="none") -> Dict[str, Any]:
    """Clear rules, hooks, theory, and fold_funcs; optionally set a prelude.

    ``prelude`` names a computation bundle or a combination thereof (DATA).
    NO domain bundle is accepted (the general-engine principle).
    """
    fold = _resolve_prelude(prelude)
    engine._rules = []
    engine._metadata = []
    engine._rule_names = {}
    engine._simplifier = None
    engine._disabled_groups = set()
    engine._step_count = 0
    engine._cancel_requested = False
    engine._hooks.clear()
    if hasattr(engine, "_theory"):
        engine._theory = None
    engine._fold_funcs = fold
    return {"ok": True}


def tool_get_status(engine) -> Dict[str, Any]:
    """Inspection: how is the engine currently configured?"""
    from rerum import __version__ as engine_version
    from rerum.mcp import PROTOCOL_VERSION

    categories = sorted({m.category for m in engine._metadata if m.category})
    groups = sorted({t for m in engine._metadata for t in (m.tags or [])})
    hooks = {ev: engine._hooks.count(ev) for ev in (
        "rule_applied", "fixpoint", "no_match", "undefined_op",
        "fold_error", "max_depth", "cycle", "should_fire")}
    return {
        "rules_count": len(engine._rules),
        "has_fold_funcs": engine._fold_funcs is not None,
        "has_theory": getattr(engine, "_theory", None) is not None,
        "hooks": hooks,
        "categories": categories,
        "groups": groups,
        "engine_version": engine_version,
        "protocol_version": PROTOCOL_VERSION,
    }
```

- [ ] **Step 4: Create `rerum/mcp/server.py`**

```python
"""MCP server lifecycle and tool dispatch.

Holds the per-session ``RuleEngine`` and a ``RuleStore``, plus a tool-name
to handler map. ``call_tool`` is what the MCP SDK request handler delegates
to. The transport is wired in ``run_server`` (__init__.py).
"""

from typing import Any, Callable, Dict, List, Optional

from rerum import RuleEngine
from rerum.mcp.errors import MCPToolError, map_exception
from rerum.mcp.persistence import RuleStore
from rerum.mcp import tools as T


class RerumMCPServer:
    """Per-session server state: one engine, one rule store, one dispatch table."""

    def __init__(self, store_root: str = ".rerum"):
        self.engine = RuleEngine()
        self.store = RuleStore(root=store_root)
        self._sampler: Optional[Callable[[str], str]] = None
        self._busy = False
        e, s = self.engine, self.store
        self._tools: Dict[str, Callable] = {
            # Authoring
            "load_rules": lambda **kw: T.tool_load_rules(self.engine, **kw),
            "add_rule": lambda **kw: T.tool_add_rule(self.engine, **kw),
            "list_rules": lambda **kw: T.tool_list_rules(self.engine, **kw),
            "get_rule": lambda **kw: T.tool_get_rule(self.engine, **kw),
            "validate_examples": lambda **kw: T.tool_validate_examples(self.engine, **kw),
            # Persistence
            "save_ruleset": lambda **kw: T.tool_save_ruleset(self.engine, self.store, **kw),
            "load_ruleset": lambda **kw: T.tool_load_ruleset(self.engine, self.store, **kw),
            "list_rulesets": lambda **kw: T.tool_list_rulesets(self.engine, self.store, **kw),
            "load_theory": lambda **kw: T.tool_load_theory(self.engine, self.store, **kw),
            # Applying
            "simplify": lambda **kw: T.tool_simplify(self.engine, **kw),
            "apply_once": lambda **kw: T.tool_apply_once(self.engine, **kw),
            "equivalents": lambda **kw: T.tool_equivalents(self.engine, **kw),
            "prove_equal": lambda **kw: T.tool_prove_equal(self.engine, **kw),
            "minimize": lambda **kw: T.tool_minimize(self.engine, **kw),
            # Goal solving
            "solve_goal": lambda **kw: T.tool_solve_goal(self.engine, **kw),
            # Agentic loop
            "solve_assisted": lambda **kw: T.tool_solve_assisted(
                self.engine, sampler=self._sampler, **kw),
            # Admin
            "reset_engine": lambda **kw: T.tool_reset_engine(self.engine, **kw),
            "get_status": lambda **kw: T.tool_get_status(self.engine, **kw),
        }

    def list_tool_names(self) -> List[str]:
        return list(self._tools.keys())

    def set_sampler(self, sampler: Optional[Callable[[str], str]]) -> None:
        self._sampler = sampler

    def call_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        handler = self._tools.get(name)
        if handler is None:
            return MCPToolError(
                "parse_error", f"unknown tool {name!r}",
                details={"name": name, "available": self.list_tool_names()},
            ).to_dict()
        if self._busy:
            return MCPToolError(
                "engine_busy",
                "another tool call is in progress on this engine").to_dict()
        self._busy = True
        try:
            return handler(**args)
        except MCPToolError as exc:
            return exc.to_dict()
        except Exception as exc:
            return map_exception(exc, context={"tool": name})
        finally:
            self._busy = False
```

- [ ] **Step 5: Wire `run_server` in `rerum/mcp/__init__.py`**

Replace the stub `run_server` body:

```python
def run_server(transport: str = "stdio", host: str = "127.0.0.1",
               port: int = 8765) -> None:
    """Run the rerum MCP server over stdio (HTTP declared, not implemented)."""
    import asyncio
    import json
    from typing import Any, Dict, List
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server
    from mcp import types
    from rerum.mcp.server import RerumMCPServer

    rerum_srv = RerumMCPServer()
    sdk_srv: Server = Server("rerum-mcp")

    @sdk_srv.list_tools()
    async def list_tools() -> List[types.Tool]:
        return [
            types.Tool(name=name, description=f"rerum tool: {name}",
                       inputSchema={"type": "object"})
            for name in rerum_srv.list_tool_names()
        ]

    @sdk_srv.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
        result = rerum_srv.call_tool(name, arguments)
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    async def _run() -> None:
        if transport == "stdio":
            async with stdio_server() as (read_stream, write_stream):
                await sdk_srv.run(
                    read_stream, write_stream,
                    sdk_srv.create_initialization_options())
        else:  # pragma: no cover
            raise NotImplementedError(
                f"transport {transport!r} not supported; use 'stdio'")

    asyncio.run(_run())
```

- [ ] **Step 6: Run tests (expected PASS)**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_mcp_smoke.py -v 2>&1 | tail -20
```

Expected PASS: all server/admin/concurrency tests pass.

- [ ] **Step 7: Smoke-check the import**

```bash
cd /home/spinoza/github/repos/rerum && python -c "from rerum.mcp import run_server; print('ok')" 2>&1 | tail -5
```

Expected: `ok`.

- [ ] **Step 8: Run full suite**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest 2>&1 | tail -3
```

- [ ] **Step 9: Commit**

```bash
cd /home/spinoza/github/repos/rerum && git add rerum/mcp/tools.py rerum/mcp/server.py rerum/mcp/__init__.py rerum/tests/test_mcp_smoke.py
git commit -m "$(cat <<'EOF'
feat(mcp): admin tools, server dispatch, concurrency guard, run_server

reset_engine accepts a computation-bundle prelude (arithmetic/math/
predicate/full/none) or a combination via combine_preludes; no domain
bundle. get_status reports engine/protocol version, hooks, theory and
fold_funcs presence. RerumMCPServer holds one engine + one RuleStore and
dispatches all 18 tools; engine_busy guard serializes calls. run_server
wires the stdio transport.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: No-domain-logic verification test + CHANGELOG + version + docs

A test asserting the server modules contain no domain operator literals (the
Section 0 swap test, mechanized), plus the release bookkeeping.

**Files:**
- Create: `rerum/tests/test_mcp_no_domain.py`
- Modify: `pyproject.toml`, `rerum/__init__.py`, `CHANGELOG.md`, `CLAUDE.md`

- [ ] **Step 1: Write the failing test**

Create `rerum/tests/test_mcp_no_domain.py`:

```python
"""Verify the MCP server contains no domain logic (Section 0 swap test).

No domain operator symbol (dd, int, lim, and, or as calculus/boolean
literals) may appear as a special case in any rerum/mcp/ source file.
"""

import pathlib
import re

import pytest

MCP_DIR = pathlib.Path(__file__).resolve().parent.parent / "mcp"
# Word-boundary patterns for domain operator literals used as code.
DOMAIN_TOKENS = [
    r'"dd"', r"'dd'", r'"int"', r"'int'", r'"lim"', r"'lim'",
]


@pytest.mark.parametrize("path", sorted(MCP_DIR.glob("*.py")))
def test_no_domain_operator_literals(path):
    text = path.read_text(encoding="utf-8")
    for token in DOMAIN_TOKENS:
        assert not re.search(token, text), (
            f"{path.name} contains domain operator literal {token}; the MCP "
            f"server must be domain-agnostic (loads rules/theories as data)."
        )


def test_op_free_goal_operator_names_come_from_caller():
    # _compile_goal must not hardcode any operator; it reads names from the
    # caller's goal dict.
    from rerum.mcp.tools import _compile_goal
    pred = _compile_goal({"op_free": ["int", "lim"]})
    # Predicate fired on an expression containing those ops returns False.
    assert pred(["int", ["sin", "x"], "x"]) is False
    assert pred(["+", "x", 1]) is True
```

- [ ] **Step 2: Verify failure / run**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_mcp_no_domain.py -v 2>&1 | tail -15
```

Expected: PASS if the implementation followed the no-domain rule. If any module
contains a domain literal, this test FAILS and pins the offending file (fix the
module, do not weaken the test).

- [ ] **Step 3: Bump version**

In `pyproject.toml`, change `version = "0.7.0"` to `version = "0.8.0"`.
In `rerum/__init__.py`, change `__version__ = "0.7.0"` to `__version__ = "0.8.0"`.

- [ ] **Step 4: CHANGELOG entry**

In `CHANGELOG.md`, add above `## [0.7.0]`:

```markdown
## [0.8.0]

### Added (MCP server)
- ``rerum/mcp/`` submodule: the GENERAL agent surface over the rewriting
  engine. 18 tools across authoring, persistence, applying, goal solving,
  the agentic loop, and admin. No domain logic; rules and theories load as
  data.
- New console entry point ``rerum-mcp`` and optional install extra
  ``pip install rerum[mcp]``; ``run_server()`` over stdio.
- Authoring: ``load_rules``, ``add_rule``, ``list_rules``, ``get_rule``,
  ``validate_examples``.
- Persistence (file-backed, git-friendly, default ``.rerum/rules/``):
  ``save_ruleset``, ``load_ruleset``, ``list_rulesets`` (``<name>.json``)
  and ``load_theory`` (``<name>.theory.json``).
- Applying (return result + situated trace + ``prose``): ``simplify``,
  ``apply_once``, ``equivalents``, ``prove_equal``, ``minimize``. The trace
  is the Phase 1 situated trace (rule_id, direction, bindings, path, kind,
  guard, rationale, whole-expression before_root/after_root) and every
  response carries a natural-language ``prose`` rendering via
  ``rerum.training.to_prose``.
- Goal solving: ``solve_goal`` wraps engine ``solve()`` over a
  caller-described goal (e.g. ``{"op_free": ["int","lim"]}``).
- Agentic loop: ``solve_assisted`` (renamed from the earlier ``solve``)
  runs directed simplify with an ``on_no_match`` LLM resolver via MCP
  sampling; inferred rules install with provenance ``llm-inferred``.
- Admin: ``reset_engine`` (computation-bundle prelude or combination; no
  domain bundle) and ``get_status``. ``MCPToolError`` stable codes;
  ``engine_busy`` guard.
```

- [ ] **Step 5: CLAUDE.md update**

Add an `mcp/` subsection to the Architecture section and a footgun:

```markdown
### `rerum/mcp/`, the agent-facing MCP server (v0.8)

- ``__init__.py``: ``run_server()`` (stdio), ``PROTOCOL_VERSION``, optional
  ``mcp`` SDK guard.
- ``server.py``: ``RerumMCPServer`` per-session engine + ``RuleStore``,
  18-tool dispatch, ``engine_busy`` guard.
- ``tools.py``: tool handlers (authoring, persistence, applying, goal
  solving, agentic loop, admin). No domain logic; rules/theories are data.
- ``trace.py``: situated ``step_to_dict``, ``assemble_trace`` (global roots
  + ``prose`` via ``rerum.training.to_prose`` + truncation), ``trace_recorder``.
- ``persistence.py``: ``RuleStore`` (``.rerum/rules/<name>.json`` and
  ``<name>.theory.json``).
- ``solver.py``: LLM-resolver factory used by ``solve_assisted``.
- ``errors.py``: ``MCPToolError`` + ``map_exception``.
- Naming: engine ``solve()`` (search) vs ``solve_goal`` (its MCP wrapper)
  vs ``solve_assisted`` (LLM-resolver loop) are three distinct things.
```

In Footguns:

```markdown
- **MCP solve naming**: ``solve_goal`` is goal-directed search (engine
  ``solve()``); ``solve_assisted`` is the LLM-resolver agentic loop (the
  old ``solve`` tool). They do not share an implementation.
- **MCP is domain-agnostic by test**: ``test_mcp_no_domain.py`` fails if any
  ``rerum/mcp/`` file hardcodes a domain operator literal. The server loads
  rules and theories as data; keep it that way.
```

- [ ] **Step 6: Run full suite**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest 2>&1 | tail -3
```

- [ ] **Step 7: Commit**

```bash
cd /home/spinoza/github/repos/rerum && git add rerum/tests/test_mcp_no_domain.py CHANGELOG.md pyproject.toml rerum/__init__.py CLAUDE.md
git commit -m "$(cat <<'EOF'
chore: bump to 0.8.0; document MCP server; mechanize the swap test

test_mcp_no_domain.py asserts no domain operator literal appears in any
rerum/mcp/ source (the Section 0 swap test, mechanized). CHANGELOG gains
the [0.8.0] section; CLAUDE.md gains the mcp/ architecture subsection and
MCP footguns. Version 0.7.0 -> 0.8.0.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Done When

- [ ] All 18 tools are registered and dispatchable: `load_rules`, `add_rule`,
  `list_rules`, `get_rule`, `validate_examples`, `save_ruleset`, `load_ruleset`,
  `list_rulesets`, `load_theory`, `simplify`, `apply_once`, `equivalents`,
  `prove_equal`, `minimize`, `solve_goal`, `solve_assisted`, `reset_engine`,
  `get_status` (`test_mcp_smoke.py::TestServerLifecycle::test_server_registers_all_tools`).
- [ ] The naming disambiguation holds: engine `solve()` is the Phase 3 search;
  `solve_goal` wraps it over a caller-described goal; `solve_assisted` is the
  LLM-resolver loop (renamed from the prior `solve`). No tool is named bare
  `solve`.
- [ ] Every rewriting tool response (`simplify`, `apply_once`, `equivalents`,
  `prove_equal`, `minimize`, `solve_goal`, `solve_assisted`) carries a `prose`
  field, and the situated trace steps carry `rule_id`, `direction`, `bindings`,
  `path`, `kind`, `guard`, `rationale`, with whole-expression `before_root`/
  `after_root` per step (`test_mcp_tools.py::TestApplyingTools`,
  `test_mcp_solve_goal.py`, `test_mcp_solve_assisted.py`).
- [ ] Persistence round-trips: `save_ruleset` then `load_ruleset` restores rules;
  `list_rulesets` enumerates `<name>.json`; `load_theory` applies a
  `<name>.theory.json` via `with_theory`; the store is git-friendly under
  `.rerum/rules/` and rejects path traversal (`test_mcp_persistence.py`).
- [ ] `reset_engine` accepts only computation bundles (arithmetic/math/predicate/
  full/none) or a combination via `combine_preludes`, and rejects a domain name
  (e.g. `"calculus"`) with `parse_error`
  (`test_mcp_smoke.py::TestAdminTools::test_reset_engine_rejects_domain_bundle`).
- [ ] The agentic loop installs LLM-inferred rules with provenance
  `llm-inferred`, enforces the per-solve cap, and reports termination reasons
  (`resolver_budget_exhausted`/`resolver_loop`)
  (`test_mcp_solve_assisted.py`).
- [ ] **No domain logic in the server (verify by inspection AND by test):**
  `test_mcp_no_domain.py` passes, asserting no domain operator literal
  (`"dd"`/`"int"`/`"lim"`) appears in any `rerum/mcp/` source file, and that
  `_compile_goal` reads operator names from the caller's goal rather than
  hardcoding any. Manually confirm no `rerum/mcp/` module imports an
  `examples/` checker or names a domain prelude. The server loads rules and
  theories as DATA only.
- [ ] `pip install rerum[mcp]` extra and the `rerum-mcp` console script are
  declared in `pyproject.toml`; `from rerum.mcp import run_server` imports and
  is callable.
- [ ] The full test suite passes; version is bumped to 0.8.0 with a CHANGELOG
  `[0.8.0]` entry and a CLAUDE.md `mcp/` subsection.
