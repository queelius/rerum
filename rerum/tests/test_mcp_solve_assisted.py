"""Tests for solve_assisted (the agentic LLM-resolver loop) with mocked sampling."""

import json

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

    def test_response_is_json_serializable(self):
        """Every solve_assisted field must json.dumps cleanly (a proposed
        rule may compute a Fraction at the boundary)."""
        from rerum import RuleEngine, ARITHMETIC_PRELUDE
        from rerum.mcp.tools import tool_solve_assisted

        engine = RuleEngine(fold_funcs=ARITHMETIC_PRELUDE)
        # The inferred rule computes (/ 1 3), a Fraction, via the prelude.
        sampler = make_sampler(["@third: (third ?x) => (! / 1 3)"])
        result = tool_solve_assisted(engine, expr="(third q)", sampler=sampler)
        # Must not raise; a Fraction would break json.dumps if it leaked.
        json.dumps(result)
        assert result["result"] == "(/ 1 3)"
