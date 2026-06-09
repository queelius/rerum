"""Tests for solve_goal (wraps engine.solve over a caller-described goal).

The goal is DATA from the caller (e.g. {"op_free": ["w"]}), compiled to a
predicate via the general contains_op helper. No domain operator literal
appears in the tool: the operator names are the caller's data.
"""

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

    def test_budget_exhaustion_reports_not_found(self):
        # A solvable-in-principle goal under a tiny node budget must return
        # found=False -- the caller's budget is honored, no hang, no partial.
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_solve_goal

        engine = RuleEngine.from_dsl("@unwrap: (w ?x) => :x")
        result = tool_solve_goal(
            engine, expr="(w (w (w (w (w a)))))",
            goal={"op_free": ["w"]}, max_nodes=1,
        )
        assert result["found"] is False
        # The result is the start expr (no solution reached), still a string.
        assert isinstance(result["result"], str)

    def test_unknown_goal_kind_is_parse_error(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_solve_goal
        from rerum.mcp.errors import MCPToolError

        engine = RuleEngine()
        with pytest.raises(MCPToolError) as exc:
            tool_solve_goal(engine, expr="(w a)", goal={"bogus": 1})
        assert exc.value.code == "parse_error"

    def test_goal_operator_names_are_caller_data(self):
        # The SAME engine solves two different goals depending only on the
        # caller's operator-name list -- proving the goal is data, not a
        # hardcoded predicate. With goal eliminating "w", (w a) reduces to a;
        # with a goal that names an op already absent, the start satisfies it.
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_solve_goal

        engine = RuleEngine.from_dsl("@unwrap: (w ?x) => :x")
        r1 = tool_solve_goal(engine, expr="(w a)", goal={"op_free": ["w"]})
        assert r1["found"] is True and r1["result"] == "a"
        # "z" never appears, so (w a) already satisfies op_free=["z"].
        r2 = tool_solve_goal(engine, expr="(w a)", goal={"op_free": ["z"]})
        assert r2["found"] is True and r2["result"] == "(w a)"


class TestSolveGoalJsonSafety:
    """solve_goal's response must survive json.dumps even when the
    derivation reaches a Fraction atom.

    The solution and every derivation step expression must render via
    format_sexpr; structured step fields route through _json_safe. A
    Fraction is not JSON-native, so this is the proof.
    """

    def test_solve_goal_json_dumps_clean_with_rational(self):
        import json
        from rerum import RuleEngine
        from rerum.rewriter import ARITHMETIC_PRELUDE
        from rerum.mcp.tools import tool_solve_goal

        # (third ?x) computes (! / 1 3) -> Fraction(1, 3). The goal
        # eliminates the "third" operator; the solution is the Fraction
        # atom and the derivation carries a Fraction-bearing step.
        engine = RuleEngine.from_dsl(
            "@third: (third ?x) => (! / 1 3)",
            fold_funcs=ARITHMETIC_PRELUDE,
        )
        result = tool_solve_goal(
            engine, expr="(third a)", goal={"op_free": ["third"]},
            max_nodes=100,
        )
        assert result["found"] is True
        assert result["result"] == "(/ 1 3)"
        json.dumps(result)
