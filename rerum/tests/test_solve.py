"""Tests for goal-directed best-first search (solve)."""

import pytest

from rerum.engine import RuleEngine
from rerum.solve import SolveResult, contains_op, solve
from rerum.optimize import expr_size
from rerum.trace import RewriteTrace


class TestContainsOp:
    def test_atom_has_no_op(self):
        assert contains_op("x", {"foo"}) is False
        assert contains_op(42, {"foo"}) is False

    def test_top_level_op(self):
        assert contains_op(["foo", "x"], {"foo"}) is True

    def test_nested_op(self):
        assert contains_op(["+", "x", ["foo", "y"]], {"foo"}) is True

    def test_absent_op(self):
        assert contains_op(["+", "x", ["bar", "y"]], {"foo"}) is False

    def test_multiple_ops(self):
        assert contains_op(["aaa", "x", "x"], {"aaa", "bbb"}) is True
        assert contains_op(["bbb", "x", "x", 0], {"aaa", "bbb"}) is True


# The toy problem: a `foo` operator that rewrites to `+`, and `double` that
# rewrites to `foo`. The goal is "no `foo`/`double` op remains". `solve` must
# find the foo-free form and the derivation must replay to it. The toy
# operators are deliberately nonsense names so no domain leaks in.
#
# NOTE (deviation from plan): the plan's TOY_DSL wrote the `@double` skeleton
# as ``(foo ?x)``. In this codebase ``?x`` is a literal pattern marker in
# skeleton position (it emits ``["?", "x"]``); substitution is ``:x``. The
# correct skeleton that yields ``(foo x)`` is ``(foo :x)``, used here.
TOY_DSL = """
@unfoo: (foo ?x) => (+ :x :x)
@double: (double ?x) => (foo :x)
"""


def _toy_engine():
    from rerum import ARITHMETIC_PRELUDE
    eng = RuleEngine(fold_funcs=ARITHMETIC_PRELUDE)
    eng.load_dsl(TOY_DSL)
    return eng


class TestSolveToy:
    def test_finds_foo_free_form(self):
        eng = _toy_engine()
        goal = lambda e: not contains_op(e, {"foo", "double"})
        result = solve(eng, ["double", "x"], goal, max_nodes=200)
        assert result.found is True
        assert isinstance(result, SolveResult)
        # (double x) -> (foo x) -> (+ x x)
        assert result.solution == ["+", "x", "x"]
        assert goal(result.solution)

    def test_derivation_is_a_trace_that_replays(self):
        eng = _toy_engine()
        goal = lambda e: not contains_op(e, {"foo", "double"})
        result = solve(eng, ["double", "x"], goal, max_nodes=200)
        deriv = result.derivation
        assert isinstance(deriv, RewriteTrace)
        assert deriv.initial == ["double", "x"]
        assert deriv.final == result.solution
        # Replaying the step `after` fields from initial reaches solution.
        current = deriv.initial
        for step in deriv.steps:
            current = step.after
        assert current == result.solution
        # Each step names the rule that produced it.
        names = [s.metadata.name for s in deriv.steps]
        assert names == ["double", "unfoo"]

    def test_already_satisfied_returns_zero_step_trace(self):
        eng = _toy_engine()
        goal = lambda e: not contains_op(e, {"foo", "double"})
        result = solve(eng, ["+", "x", "x"], goal, max_nodes=50)
        assert result.found is True
        assert result.solution == ["+", "x", "x"]
        assert len(result.derivation.steps) == 0

    def test_budget_exhaustion_reports_not_found(self):
        eng = _toy_engine()
        # An impossible goal: no rule can ever make this true.
        impossible = lambda e: contains_op(e, {"never"})
        result = solve(eng, ["double", "x"], impossible, max_nodes=25)
        assert result.found is False
        assert result.solution is None
        assert result.explored <= 25

    def test_explored_count_is_positive_on_success(self):
        eng = _toy_engine()
        goal = lambda e: not contains_op(e, {"foo", "double"})
        result = solve(eng, ["double", "x"], goal, max_nodes=200)
        assert result.explored >= 1


class TestEngineSolveWrapper:
    def test_engine_method_delegates(self):
        eng = _toy_engine()
        goal = lambda e: not contains_op(e, {"foo", "double"})
        result = eng.solve(["double", "x"], goal, max_nodes=200)
        assert result.found is True
        assert result.solution == ["+", "x", "x"]


class TestTopLevelImports:
    def test_exports(self):
        import rerum
        assert rerum.solve is solve
        assert rerum.SolveResult is SolveResult
        assert rerum.contains_op is contains_op
