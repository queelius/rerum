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


class TestAuthoringJsonSafety:
    """Every authoring response must survive json.dumps, even when a rule
    or example carries an exact rational (a Fraction atom).

    Fraction is not JSON-native; expression fields must render to s-expr
    strings via format_sexpr and structured fields must pass through
    _json_safe. These tests are the proof.
    """

    def _rational_engine(self):
        from rerum import RuleEngine
        from fractions import Fraction
        engine = RuleEngine()
        # A Fraction atom in both pattern and skeleton: this is the value
        # that breaks raw json.dumps and must be rendered via format_sexpr.
        # The example uses s-expr STRINGS (the validator parses them), and
        # rewriting (half (/ 1 2)) -> (/ 1 1) holds, so it validates clean.
        engine.add_rule(
            pattern=["half", Fraction(1, 2)],
            skeleton=Fraction(1, 1),
            name="frac",
            category="arith",
            examples=[{"in": "(half (/ 1 2))", "out": "(/ 1 1)"}],
            validate_examples=False,
        )
        return engine

    def test_get_rule_json_dumps_clean_with_rational(self):
        import json
        from rerum.mcp.tools import tool_get_rule

        engine = self._rational_engine()
        result = tool_get_rule(engine, name="frac")
        # Must not raise. The expr fields are strings; examples are sanitized.
        json.dumps(result)
        assert result["pattern"] == "(half (/ 1 2))"
        assert result["skeleton"] == "(/ 1 1)"

    def test_list_rules_json_dumps_clean_with_rational(self):
        import json
        from rerum.mcp.tools import tool_list_rules

        engine = self._rational_engine()
        result = tool_list_rules(engine)
        json.dumps(result)
        assert any(r["name"] == "frac" for r in result)

    def test_validate_examples_json_dumps_clean_with_rational(self):
        import json
        from fractions import Fraction
        from rerum.mcp.tools import tool_validate_examples

        engine = self._rational_engine()
        # A second rule that carries a Fraction atom AND a wrong example, so
        # the failure-as-data path reports an example referencing a rational.
        # (half (/ 1 3)) rewrites to (/ 1 1), not (/ 1 3), so this fails.
        engine.add_rule(
            pattern=["half", Fraction(1, 2)],
            skeleton=Fraction(1, 1),
            name="frac-bad",
            examples=[{"in": "(half (/ 1 3))", "out": "(/ 1 3)"}],
            validate_examples=False,
        )
        result = tool_validate_examples(engine)
        # The response must json.dumps regardless of pass/fail, and the bad
        # rule must surface as data (not a raised traceback).
        json.dumps(result)
        assert result["ok"] is False
        assert any(e["rule_name"] == "frac-bad" for e in result["errors"])
