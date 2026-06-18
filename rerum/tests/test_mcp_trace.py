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
        # Prose is NOT embedded in the trace dict (it is a top-level
        # response field); render it from the recorder's trace instead.
        assert "prose" not in d
        from rerum.mcp.trace import render_prose
        prose = render_prose(recorder.trace)
        assert isinstance(prose, str) and prose
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
        d = assemble_trace(initial="x", final="x", recorder=rec)
        assert len(d["steps"]) == 201  # 100 + marker + 100
        assert d["total_steps"] == 250  # full count survives truncation
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
        d = assemble_trace(initial="x", final="x", recorder=rec)
        assert not any(isinstance(s, dict) and s.get("_elided")
                       for s in d["steps"])
        assert len(d["steps"]) == 150

    def test_assemble_empty_steps(self):
        from rerum.mcp.trace import assemble_trace, _Recorder

        rec = _Recorder()
        rec.steps = []
        d = assemble_trace(initial="x", final="x", recorder=rec)
        assert d["steps"] == []
        assert d["total_steps"] == 0


class TestJsonSafety:
    """The MCP response is JSON. Steps can carry Fraction atoms (a pattern
    var binds one, a guard computes one) now that Fraction is a numeric atom;
    the assembled response must stay json.dumps-serializable. Regression for
    the Group 1 review's blocking finding (raw bindings/guard.result)."""

    def _fraction_trace(self):
        from rerum.engine import RuleEngine
        from rerum.rewriter import ARITHMETIC_PRELUDE
        from rerum.mcp.trace import trace_recorder
        eng = RuleEngine(fold_funcs=ARITHMETIC_PRELUDE)
        eng.load_dsl("@mkhalf: (mk) => (wrap (! / 1 2))\n@unwrap: (wrap ?x) => :x")
        with trace_recorder(eng, initial=["mk"]) as rec:
            eng.simplify(["mk"])
        return rec

    def test_fraction_binding_assembled_response_serializes(self):
        import json
        from rerum.mcp.trace import assemble_trace
        rec = self._fraction_trace()
        d = assemble_trace(initial="(mk)", final="(/ 1 2)", recorder=rec)
        text = json.dumps(d)  # must not raise
        back = json.loads(text)
        # The Fraction binding was rendered to its exact rational literal.
        unwrap = [s for s in back["steps"]
                  if isinstance(s, dict) and "unwrap" in str(s.get("rule_id"))]
        assert unwrap and unwrap[0]["bindings"]["x"] == "1/2"

    def test_guard_result_fraction_serializes(self):
        import json
        from fractions import Fraction
        from rerum.mcp.trace import step_to_dict
        from rerum.engine import RuleMetadata
        from rerum.trace import RewriteStep
        meta = RuleMetadata(name="g", category="test")
        step = RewriteStep(
            0, meta, ["a"], ["b"], kind="rule",
            guard={"condition": [">", "x", 0], "result": Fraction(3, 4)},
        )
        d = step_to_dict(step)
        text = json.dumps(d)  # must not raise
        assert json.loads(text)["guard"]["result"] == "3/4"

    def test_prose_answer_line_reflects_final_not_none(self):
        from rerum.mcp.trace import assemble_trace
        rec = self._fraction_trace()
        from rerum.mcp.trace import render_prose
        assemble_trace(initial="(mk)", final="(/ 1 2)", recorder=rec)
        # The recorder never set trace.final; assemble_trace must, so the
        # prose answer line is the result, not "None".
        prose = render_prose(rec.trace)
        assert prose.splitlines()[-1] == "Answer: (/ 1 2)."

    def test_json_safe_preserves_native_and_bool(self):
        from rerum.mcp.trace import _json_safe
        # bool stays bool (checked before int); native structure unchanged.
        assert _json_safe(True) is True
        assert _json_safe({"a": ["+", "x", 1], "b": "y"}) == {
            "a": ["+", "x", 1], "b": "y"}


class TestJsonSafeFloats:
    def test_non_finite_floats_render_as_strings(self):
        import json
        from rerum.mcp.utils import json_safe
        out = json_safe({"a": float("inf"), "b": float("-inf"),
                         "c": float("nan"), "d": 1.5})
        assert out["a"] == "inf" and out["b"] == "-inf" and out["c"] == "nan"
        assert out["d"] == 1.5
        json.dumps(out, allow_nan=False)  # strictly valid JSON

    def test_trace_alias_is_shared_helper(self):
        from rerum.mcp import trace, utils
        assert trace._json_safe is utils.json_safe


class TestJsonSafeSetsAndKeys:
    """json_safe is the single transport sanitizer; it must handle the
    non-JSON-native shapes that can reach it (sets, non-string dict keys)
    rather than relying on the server's last-resort TypeError guard."""

    def test_set_becomes_list(self):
        import json
        from rerum.mcp.utils import json_safe
        out = json_safe({"ops": {"a", "b", "c"}})
        assert sorted(out["ops"]) == ["a", "b", "c"]
        json.dumps(out)  # must not raise

    def test_frozenset_becomes_list(self):
        import json
        from rerum.mcp.utils import json_safe
        json.dumps(json_safe(frozenset({1, 2})))

    def test_non_string_dict_key_coerced(self):
        import json
        from fractions import Fraction
        from rerum.mcp.utils import json_safe
        # (1 and True collide as dict keys in Python -- True==1 -- so keep
        # each coercion in its own dict.)
        assert json_safe({1: "a"}) == {"1": "a"}
        assert json_safe({Fraction(1, 2): "b"}) == {"1/2": "b"}
        assert json_safe({True: "c"}) == {"True": "c"}
        json.dumps(json_safe({7: "x", Fraction(3, 4): "y"}))  # no raise

    def test_nested_set_in_list(self):
        import json
        from rerum.mcp.utils import json_safe
        json.dumps(json_safe([{"k": {1, 2}}, ({3, 4},)]))
