"""Tests for the general numeric evaluator (numeval / numeric_equiv)."""

import math

import pytest

from rerum.numeval import numeval, numeric_equiv, NumevalError
from rerum.rewriter import ARITHMETIC_PRELUDE, MATH_PRELUDE


class TestNumevalAtoms:
    def test_number_returns_itself(self):
        assert numeval(7, {}, ARITHMETIC_PRELUDE) == 7
        assert numeval(3.5, {}, ARITHMETIC_PRELUDE) == 3.5

    def test_symbol_looks_up_env(self):
        assert numeval("x", {"x": 2.0}, ARITHMETIC_PRELUDE) == 2.0

    def test_unbound_symbol_raises(self):
        with pytest.raises(NumevalError):
            numeval("y", {"x": 1.0}, ARITHMETIC_PRELUDE)


class TestNumevalCompounds:
    def test_nested_arithmetic(self):
        # (+ 1 (* 2 3)) == 7
        expr = ["+", 1, ["*", 2, 3]]
        assert numeval(expr, {}, ARITHMETIC_PRELUDE) == 7

    def test_env_substitution_then_evaluate(self):
        # (* x x) with x=2.0 == 4.0
        expr = ["*", "x", "x"]
        assert numeval(expr, {"x": 2.0}, ARITHMETIC_PRELUDE) == 4.0

    def test_math_prelude_operator(self):
        # (sin 0) == 0.0 via MATH_PRELUDE
        assert numeval(["sin", 0], {}, MATH_PRELUDE) == pytest.approx(0.0)

    def test_undefined_op_raises(self):
        # `quux` is in no prelude.
        with pytest.raises(NumevalError):
            numeval(["quux", 1, 2], {}, ARITHMETIC_PRELUDE)

    def test_domain_error_raises(self):
        # (log -1) is a math domain error under MATH_PRELUDE.
        with pytest.raises(NumevalError):
            numeval(["log", -1], {}, MATH_PRELUDE)


def _fixed_sampler(points):
    """A deterministic sampler: cycles through a fixed list of envs."""
    it = iter(points)

    def sample():
        return next(it)

    return sample


class TestNumericEquiv:
    def test_equivalent_expressions_agree(self):
        # (+ x x) and (* 2 x) agree everywhere.
        a = ["+", "x", "x"]
        b = ["*", 2, "x"]
        sampler = lambda: {"x": 1.0}  # constant sampler is fine
        assert numeric_equiv(a, b, sampler, ARITHMETIC_PRELUDE, samples=8) is True

    def test_inequivalent_expressions_disagree(self):
        # (* x x) and (+ x 1) differ (e.g. at x=3: 9 vs 4).
        a = ["*", "x", "x"]
        b = ["+", "x", 1]
        # Sample a spread of points so at least one separates them.
        pts = [{"x": float(v)} for v in (0, 1, 2, 3, 4, 5, 6, 7)]
        sampler = _fixed_sampler(pts)
        assert numeric_equiv(a, b, sampler, ARITHMETIC_PRELUDE, samples=8) is False

    def test_dict_of_ranges_sampler(self):
        # A dict {var: (lo, hi)} is accepted as a sampler spec.
        a = ["+", "x", "x"]
        b = ["*", 2, "x"]
        assert numeric_equiv(a, b, {"x": (-5.0, 5.0)},
                             MATH_PRELUDE, samples=16) is True

    def test_domain_error_points_are_skipped_not_crashed(self):
        # (log x) vs (log x): identical, so always equal where defined.
        # Sampling includes x <= 0 (domain error for log); those points
        # must be skipped, not crash, and the verdict stays True.
        a = ["log", "x"]
        b = ["log", "x"]
        pts = [{"x": float(v)} for v in (-2, -1, 0, 1, 2, 3, 4, 5)]
        sampler = _fixed_sampler(pts)
        assert numeric_equiv(a, b, sampler, MATH_PRELUDE, samples=8) is True

    def test_within_tolerance(self):
        # Two expressions equal up to floating slack are equivalent.
        a = ["*", "x", 1]
        b = "x"
        assert numeric_equiv(a, b, lambda: {"x": 0.1},
                             ARITHMETIC_PRELUDE, samples=4, tol=1e-9) is True


class TestNumevalExports:
    def test_exports(self):
        import rerum
        assert rerum.numeval is numeval
        assert rerum.numeric_equiv is numeric_equiv
        assert rerum.NumevalError is NumevalError
