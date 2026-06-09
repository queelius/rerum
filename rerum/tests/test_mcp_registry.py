"""Tests for the tool registry (schema-from-signature, the 0.9.0 core).

The registry derives the whole MCP tool surface from the ``tool_*`` handler
signatures and docstrings: discovery, dependency injection, JSON input
schemas, and dispatch-time validation. These tests pin that derivation.
"""

import pytest

from rerum.mcp.errors import MCPToolError
from rerum.mcp.registry import (
    build_registry,
    get_registry,
    validate_and_coerce,
)


@pytest.fixture(scope="module")
def registry():
    return get_registry()


class TestRegistryBuild:
    def test_discovers_exactly_the_tool_functions(self, registry):
        import rerum.mcp.tools as T
        expected = {n[len("tool_"):] for n in vars(T)
                    if n.startswith("tool_") and callable(getattr(T, n))}
        assert set(registry) == expected
        assert len(registry) == 18

    def test_deps_extracted_from_positional_params(self, registry):
        assert registry["simplify"].deps == ("engine",)
        assert registry["save_ruleset"].deps == ("engine", "store")
        assert registry["list_rulesets"].deps == ("store",)
        assert registry["solve_assisted"].deps == ("engine", "sampler")

    def test_injected_deps_never_appear_in_schema(self, registry):
        schema = registry["solve_assisted"].input_schema
        assert "sampler" not in schema["properties"]
        assert "engine" not in schema["properties"]

    def test_required_derived_from_defaults(self, registry):
        schema = registry["simplify"].input_schema
        assert schema["required"] == ["expr"]
        assert "strategy" in schema["properties"]  # optional, present

    def test_scalar_annotations_map_to_json_types(self, registry):
        props = registry["simplify"].input_schema["properties"]
        assert props["expr"]["type"] == "string"
        assert props["max_steps"]["type"] == "integer"
        props_eq = registry["equivalents"].input_schema["properties"]
        assert props_eq["include_unidirectional"]["type"] == "boolean"

    def test_literal_annotation_becomes_enum(self, registry):
        strategy = registry["simplify"].input_schema["properties"]["strategy"]
        assert strategy["type"] == "string"
        assert set(strategy["enum"]) == {
            "exhaustive", "once", "bottomup", "topdown"}

    def test_optional_literal_keeps_enum_not_required(self, registry):
        spec = registry["minimize"]
        metric = spec.input_schema["properties"]["metric"]
        assert set(metric["enum"]) == {"size", "depth", "ops", "atoms"}
        assert "metric" not in spec.input_schema.get("required", [])

    def test_union_str_or_list_is_oneof(self, registry):
        prelude = registry["reset_engine"].input_schema["properties"]["prelude"]
        assert "oneOf" in prelude
        types = {frozenset(s.items()) if False else s.get("type")
                 for s in prelude["oneOf"]}
        assert types == {"string", "array"}

    def test_list_annotation_is_array(self, registry):
        groups = registry["simplify"].input_schema["properties"]["groups"]
        assert groups["type"] == "array"
        assert groups["items"]["type"] == "string"

    def test_dict_annotation_is_object(self, registry):
        goal = registry["solve_goal"].input_schema["properties"]["goal"]
        assert goal["type"] == "object"

    def test_defaults_included_when_json_native(self, registry):
        props = registry["simplify"].input_schema["properties"]
        assert props["max_steps"]["default"] == 1000
        assert props["strategy"]["default"] == "exhaustive"

    def test_zero_param_tools_have_empty_schema(self, registry):
        for name in ("validate_examples", "get_status", "list_rulesets"):
            schema = registry[name].input_schema
            assert schema["properties"] == {}
            assert "required" not in schema

    def test_descriptions_from_docstrings(self, registry):
        spec = registry["simplify"]
        assert spec.description  # summary line present
        assert "Args:" not in spec.description
        # Param description parsed from the Args: section.
        assert spec.input_schema["properties"]["expr"]["description"]

    def test_multiline_arg_description_joined(self, registry):
        # load_rules' validate_examples doc wraps across two lines.
        desc = registry["load_rules"].input_schema[
            "properties"]["validate_examples"]["description"]
        assert "prelude" in desc  # continuation line was joined

    def test_additional_properties_false_everywhere(self, registry):
        for spec in registry.values():
            assert spec.input_schema["additionalProperties"] is False

    def test_build_registry_is_pure_over_module(self):
        import rerum.mcp.tools as T
        assert set(build_registry(T)) == set(get_registry())


class TestValidateAndCoerce:
    def test_missing_required_is_parse_error(self, registry):
        with pytest.raises(MCPToolError) as exc_info:
            validate_and_coerce(registry["simplify"], {})
        assert exc_info.value.code == "parse_error"
        assert "expr" in str(exc_info.value)

    def test_unknown_param_is_parse_error(self, registry):
        with pytest.raises(MCPToolError) as exc_info:
            validate_and_coerce(registry["simplify"],
                                {"expr": "x", "max_deph": 6})
        assert exc_info.value.code == "parse_error"
        assert "max_deph" in str(exc_info.value)

    def test_string_int_coerced(self, registry):
        out = validate_and_coerce(registry["simplify"],
                                  {"expr": "x", "max_steps": "50"})
        assert out["max_steps"] == 50

    def test_string_bool_coerced(self, registry):
        out = validate_and_coerce(
            registry["equivalents"],
            {"expr": "x", "include_unidirectional": "true"})
        assert out["include_unidirectional"] is True

    def test_correct_types_pass_through(self, registry):
        out = validate_and_coerce(registry["simplify"],
                                  {"expr": "x", "max_steps": 50})
        assert out["max_steps"] == 50

    def test_uncoercible_string_left_for_handler(self, registry):
        out = validate_and_coerce(registry["simplify"],
                                  {"expr": "x", "max_steps": "lots"})
        assert out["max_steps"] == "lots"

    def test_enum_violation_is_parse_error(self, registry):
        with pytest.raises(MCPToolError) as exc_info:
            validate_and_coerce(registry["simplify"],
                                {"expr": "x", "strategy": "magic"})
        assert exc_info.value.code == "parse_error"
        assert "allowed" in str(exc_info.value.details)

    def test_optional_literal_accepts_none(self, registry):
        out = validate_and_coerce(registry["minimize"],
                                  {"expr": "x", "metric": None})
        assert out["metric"] is None

    def test_optional_int_string_coerced(self, registry):
        out = validate_and_coerce(
            registry["prove_equal"],
            {"expr_a": "a", "expr_b": "b", "max_expressions": "500"})
        assert out["max_expressions"] == 500
