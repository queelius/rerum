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

    def test_no_theory_plain_goal_finds_x_sin_x_correctly(self):
        # With int-neg now in the base table, the plain goal WITHOUT theory
        # closes the single-by-parts case CORRECTLY (the int(-cos x)
        # sub-integral now reduces). It is the THEORY that breaks soundness
        # (next test), not the plain goal per se.
        h = _harness()
        found, correct, _explored, _secs, _r = h.measure(
            h.BYPARTS_GENERAL, h.CASES["int(x*sin x)"],
            use_theory=False, cost=h.COST_INT_HIGH, budget=500)
        assert found is True and correct is True

    def test_theory_plus_plain_goal_is_UNSOUND_finds_fast_wrong_answer(self):
        # THE central finding: PLAIN goal + THEORY-normalization is unsound.
        # Theory exposes a path where a sub-integral spuriously collapses;
        # the int-eliminating cost steers best-first to that shortest (WRONG)
        # int-free node. found=True but is_integral=False.
        h = _harness()
        found, correct, _explored, _secs, _r = h.measure(
            h.BYPARTS_GENERAL, h.CASES["int(x*sin x)"],
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
