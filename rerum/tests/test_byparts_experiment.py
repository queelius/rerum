"""Pins the deterministic premise of the C1 by-parts experiment: the
plain int-free goal is UNSOUND for by-parts (best-first finds fast WRONG
int-free answers), and a verified goal (is_integral in the goal predicate)
restores correctness cheaply. The full matrix lives in
experiments/byparts_search.py (a script, not pytest)."""

import importlib.util
from pathlib import Path

import pytest

HARNESS = (Path(__file__).resolve().parents[2]
           / "experiments" / "byparts_search.py")


def _harness():
    spec = importlib.util.spec_from_file_location("byparts_search", HARNESS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestBaselineFragility:
    def test_raw_schema_closes_x_cos_x_correctly(self):
        h = _harness()
        found, correct, _explored, _secs, _r = h.measure(
            h.BYPARTS_GENERAL, h.CASES["int(x*cos x)"],
            use_theory=False, cost=h.COST_INT_HIGH, budget=500)
        assert found is True and correct is True  # the lucky case is correct

    def test_raw_schema_does_not_close_x_sin_x(self):
        h = _harness()
        found, _correct, _explored, _secs, _r = h.measure(
            h.BYPARTS_GENERAL, h.CASES["int(x*sin x)"],
            use_theory=False, cost=h.COST_INT_HIGH, budget=500)
        # Fragile: the structurally-identical sin case does NOT close (its
        # int(-cos x) sub-integral strands without the int-neg rule).
        assert found is False

    def test_op_costs_goal_is_UNSOUND_finds_fast_wrong_answer(self):
        # THE central finding: with int-neg added, int(x*sin x) "closes"
        # FAST under the plain int-free goal -- to a WRONG answer (the
        # search prefers the shortest int-free path, which spuriously zeroes
        # a sub-integral). found=True but is_integral=False.
        h = _harness()
        found, correct, _explored, _secs, _r = h.measure(
            h.BYPARTS_GENERAL + "\n" + h.INT_NEG, h.CASES["int(x*sin x)"],
            use_theory=True, cost=h.COST_INT_HIGH, budget=500)
        assert found is True
        assert correct is False  # FAST but WRONG -- the unsoundness


class TestVerifiedGoalRestoresSoundness:
    def test_verified_goal_closes_x_sin_x_correctly_and_cheaply(self):
        # The fix: is_integral in the goal predicate rejects wrong int-free
        # nodes, so the search reaches the CORRECT antiderivative -- cheaply.
        h = _harness()
        found, correct, explored, _secs, _r = h.measure(
            h.BYPARTS_GENERAL + "\n" + h.INT_NEG, h.CASES["int(x*sin x)"],
            use_theory=True, cost=h.COST_INT_HIGH, budget=500,
            verified_goal=True)
        assert found is True and correct is True
        assert explored < 100  # cheap

    def test_verified_goal_handles_parts_twice(self):
        # int(x^2*e^x) needs by-parts twice; the verified goal still closes
        # it correctly within budget.
        h = _harness()
        found, correct, _explored, _secs, _r = h.measure(
            h.BYPARTS_GENERAL + "\n" + h.INT_NEG, h.CASES["int(x^2*e^x)"],
            use_theory=True, cost=h.COST_INT_HIGH, budget=500,
            verified_goal=True)
        assert found is True and correct is True

    def test_boomerang_correctly_unreached_under_verified_goal(self):
        # The boomerang needs the algebraic I = A - I step (beyond pure
        # rewriting); the verified goal correctly refuses to "close" it with
        # a wrong answer. This is the frontier C3 would target.
        h = _harness()
        found, _correct, _explored, _secs, _r = h.measure(
            h.BYPARTS_GENERAL + "\n" + h.INT_NEG, h.CASES["int(e^x*sin x)"],
            use_theory=True, cost=h.COST_INT_HIGH, budget=800,
            verified_goal=True)
        assert found is False
