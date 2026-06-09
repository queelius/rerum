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
