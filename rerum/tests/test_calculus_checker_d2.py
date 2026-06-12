"""Tests for the D2 verification helpers in examples/calculus_checker.py.

is_integral / is_limit are DOMAIN CONTENT built on the general rerum.numeval
primitives. The engine never imports calculus_checker; these tests load it
by path (examples/ is content, not a package).
"""

import importlib.util
from fractions import Fraction
from pathlib import Path

import pytest

CHECKER_PATH = (Path(__file__).resolve().parents[2]
                / "examples" / "calculus_checker.py")


def _load_checker():
    spec = importlib.util.spec_from_file_location("calculus_checker",
                                                  CHECKER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestIsIntegral:
    def test_power_rule_antiderivative_verifies(self):
        c = _load_checker()
        # d/dx[x^3 / 3] = x^2.
        assert c.is_integral(["^", "x", 2], "x",
                             ["/", ["^", "x", 3], 3]) is True

    def test_fraction_coefficient_form_verifies(self):
        c = _load_checker()
        # (* (^ x 3) 1/3) differentiates to x^2.
        result = ["*", ["^", "x", 3], Fraction(1, 3)]
        assert c.is_integral(["^", "x", 2], "x", result) is True

    def test_sin_antiderivative_verifies(self):
        c = _load_checker()
        assert c.is_integral(["sin", "x"], "x", ["-", ["cos", "x"]]) is True

    def test_cos_antiderivative_verifies(self):
        c = _load_checker()
        assert c.is_integral(["cos", "x"], "x", ["sin", "x"]) is True

    def test_exp_antiderivative_verifies(self):
        c = _load_checker()
        assert c.is_integral(["exp", "x"], "x", ["exp", "x"]) is True

    def test_usub_antiderivative_verifies(self):
        c = _load_checker()
        # d/dx[sin(x^2)] = 2x cos(x^2).
        integrand = ["*", ["cos", ["^", "x", 2]], ["*", 2, "x"]]
        assert c.is_integral(integrand, "x", ["sin", ["^", "x", 2]]) is True

    def test_by_parts_antiderivative_verifies(self):
        c = _load_checker()
        # d/dx[x e^x - e^x] = x e^x.
        integrand = ["*", "x", ["exp", "x"]]
        result = ["-", ["*", "x", ["exp", "x"]], ["exp", "x"]]
        assert c.is_integral(integrand, "x", result) is True

    def test_wrong_antiderivative_rejected(self):
        c = _load_checker()
        # d/dx[x^2] = 2x != x^2.
        assert c.is_integral(["^", "x", 2], "x", ["^", "x", 2]) is False

    def test_leftover_int_head_rejected(self):
        c = _load_checker()
        # A structurally-unclosed result (an int head survives) is a real
        # failure, not a domain skip.
        assert c.is_integral(["cos", "x"], "x",
                             ["int", ["cos", "x"], "x"]) is False


class TestIsLimit:
    def test_sinc_limit_is_one(self):
        c = _load_checker()
        assert c.is_limit(["/", ["sin", "x"], "x"], "x", 0, 1) is True

    def test_versine_limit_is_zero(self):
        c = _load_checker()
        assert c.is_limit(["/", ["-", 1, ["cos", "x"]], "x"], "x", 0, 0) is True

    def test_polynomial_zero_over_zero(self):
        c = _load_checker()
        assert c.is_limit(["/", ["-", ["^", "x", 2], 1], ["-", "x", 1]],
                          "x", 1, 2) is True

    def test_continuous_substitution_limit(self):
        c = _load_checker()
        assert c.is_limit(["+", "x", 1], "x", 2, 3) is True

    def test_wrong_result_rejected(self):
        c = _load_checker()
        assert c.is_limit(["/", ["sin", "x"], "x"], "x", 0, 2) is False

    def test_one_sided_domain_boundary(self):
        c = _load_checker()
        # lim_{x->0+} sqrt(x) = 0; the x<0 side is a domain error and must
        # not be treated as a counterexample.
        assert c.is_limit(["sqrt", "x"], "x", 0, 0) is True


class TestIsLimitCancellationRobustness:
    def test_second_order_limit_survives_cancellation(self):
        # lim_{x->0} (1 - cos x)/x^2 = 1/2: at eps=1e-8 the numerator
        # underflows to 0 (catastrophic cancellation), so judging by the
        # smallest-eps error alone wrongly rejected this correct limit.
        # The verdict uses the MEDIAN of the smallest defined errors.
        c = _load_checker()
        expr = ["/", ["-", 1, ["cos", "x"]], ["^", "x", 2]]
        assert c.is_limit(expr, "x", 0, 0.5) is True

    def test_wrong_target_still_rejected_despite_cancellation(self):
        # The same expression with the WRONG target 0: cancellation at
        # 1e-8 makes that single sample agree, but the median over the
        # approach does not.
        c = _load_checker()
        expr = ["/", ["-", 1, ["cos", "x"]], ["^", "x", 2]]
        assert c.is_limit(expr, "x", 0, 0) is False
