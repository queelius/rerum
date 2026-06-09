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
