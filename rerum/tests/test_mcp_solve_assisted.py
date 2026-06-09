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

        result = tool_solve_assisted(engine, sampler, expr="(+ y 0)")
        assert result["result"] == "y"
        assert result["resolver_calls"] == 0
        assert result["inferred_rules"] == []
        assert calls[0] == 0
        assert "prose" in result and "prose" not in result["trace"]
        assert result["converged"] is True  # natural fixpoint, truthful

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

        from rerum.mcp.errors import MCPToolError

        engine = RuleEngine()
        # No sampler: the tool REFUSES (honest) rather than silently
        # degrading to plain simplify.
        with pytest.raises(MCPToolError) as exc_info:
            tool_solve_assisted(engine, None, expr="(foo bar)")
        assert exc_info.value.code == "sampling_unsupported"

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

    def test_inferred_rule_cannot_execute_op_outside_prelude(self):
        # THE security boundary for pattern #2 (rules from an untrusted LLM):
        # a proposed rule is DATA. Its (! op ...) compute forms can ONLY
        # invoke operators already in the prelude. An op the LLM names that is
        # NOT in the prelude (here a scary __import__) is never executed -- it
        # is left as an inert unfolded compound, not run as Python, and is not
        # added to the prelude. "Rules are data; preludes are code."
        from rerum import RuleEngine, ARITHMETIC_PRELUDE
        from rerum.mcp.tools import tool_solve_assisted

        engine = RuleEngine(fold_funcs=ARITHMETIC_PRELUDE)
        assert "__import__" not in engine._fold_funcs
        sampler = make_sampler(["@evil: (danger ?x) => (! __import__ os)"])
        result = tool_solve_assisted(engine, expr="(danger q)", sampler=sampler)
        json.dumps(result)  # JSON-safe
        # The dangerous compute was NOT folded/executed: it survives as an
        # inert compound in the output, and the prelude is unchanged.
        assert "__import__" in result["result"]
        assert "__import__" not in engine._fold_funcs

    def test_resolver_hook_removed_after_solve_assisted(self):
        # The agentic resolver is temporary: after solve_assisted (on both the
        # success and the resolver-cap paths) the no_match hook is gone, so it
        # cannot leak into later engine operations on the same session.
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_solve_assisted

        engine = RuleEngine()
        before = engine._hooks.count("no_match")
        tool_solve_assisted(
            engine, expr="(foo bar)",
            sampler=make_sampler(["@foo-id: (foo ?x) => :x"]))
        assert engine._hooks.count("no_match") == before
        # Resolver-loop path (a never-matching rule) must also clean up.
        tool_solve_assisted(
            engine, expr="(stuck x)",
            sampler=lambda prompt: "(zzz ?y) => :y", max_resolver_calls=2)
        assert engine._hooks.count("no_match") == before


class TestTruthfulInferredRules:
    def test_reproposed_named_rule_counted_once(self):
        # The engine dedupes a re-proposed named rule; inferred_rules must
        # report what was ACTUALLY installed (1), not one entry per round.
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_solve_assisted

        engine = RuleEngine()

        def sampler(prompt):
            return "@foo-id: (foo ?x) => :x"

        result = tool_solve_assisted(engine, sampler, expr="(foo bar)",
                                     max_resolver_calls=3)
        assert result["result"] == "bar"
        assert len(result["inferred_rules"]) == 1
        assert result["inferred_rules"][0]["name"] == "foo-id"
        # The resolver may have been consulted again on the stuck atom;
        # the report still reflects installs, not proposals.
        assert result["resolver_calls"] >= 1
