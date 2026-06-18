"""Tests for the general numeric evaluator (numeval / numeric_equiv)."""

from fractions import Fraction

import pytest

from rerum.numeval import (
    numeval, numeric_equiv, NumevalError, NumevalDomainError,
)
from rerum.rewriter import ARITHMETIC_PRELUDE, MATH_PRELUDE, PREDICATE_PRELUDE


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


class TestNumevalGenerality:
    """The swap test: numeval interprets an operator it has never heard of,
    because semantics come ONLY from the supplied prelude."""

    def test_made_up_operator_from_custom_prelude(self):
        blorp = {"blorp": lambda args: args[0] + 2 * args[1]}
        assert numeval(["blorp", 3, 4], {}, blorp) == 11
        assert numeval(["blorp", "a", "b"], {"a": 1, "b": 1}, blorp) == 3


class TestNumevalErrorTaxonomy:
    """Structural failures (point-independent) vs domain failures
    (point-dependent) are distinct types, so numeric_equiv can skip the
    latter while propagating the former."""

    def test_handler_raise_is_domain_error(self):
        with pytest.raises(NumevalDomainError):
            numeval(["log", -1], {}, MATH_PRELUDE)

    def test_cannot_fold_is_domain_error(self):
        # safe_div returns None for division by zero: a domain failure.
        with pytest.raises(NumevalDomainError):
            numeval(["/", 1, 0], {}, ARITHMETIC_PRELUDE)

    def test_undefined_operator_is_structural_not_domain(self):
        with pytest.raises(NumevalError) as exc:
            numeval(["quux", 1], {}, ARITHMETIC_PRELUDE)
        assert not isinstance(exc.value, NumevalDomainError)

    def test_unbound_symbol_is_structural_not_domain(self):
        with pytest.raises(NumevalError) as exc:
            numeval("zzz", {}, ARITHMETIC_PRELUDE)
        assert not isinstance(exc.value, NumevalDomainError)


class TestNumericEquivEdgeCases:
    def test_all_points_skipped_is_not_equivalent(self):
        # Every sampled point is a domain error for `a` (log of a
        # non-positive), so no point is defined. With no supporting
        # evidence, the verdict is False -- NOT a vacuous True (even though
        # `b` is wildly different from `a`).
        a = ["log", "x"]
        b = ["+", ["*", 7, "x"], 100]
        pts = [{"x": float(v)} for v in (-5, -4, -3, -2, -1)]
        assert numeric_equiv(a, b, _fixed_sampler(pts),
                             MATH_PRELUDE, samples=5) is False

    def test_division_by_zero_point_is_skipped(self):
        # (/ x x) == 1 wherever x != 0; the x=0 point is a domain failure
        # (None fold) and is skipped, leaving defined points that agree.
        a = ["/", "x", "x"]
        b = 1
        pts = [{"x": float(v)} for v in (0, 1, 2, 3)]
        assert numeric_equiv(a, b, _fixed_sampler(pts),
                             ARITHMETIC_PRELUDE, samples=4) is True

    def test_missing_operator_propagates(self):
        # An undefined operator is structural: it must propagate, surfacing
        # the malformed query rather than being skipped to a vacuous verdict.
        a = ["notanop", "x"]
        b = ["+", "x", 1]
        with pytest.raises(NumevalError):
            numeric_equiv(a, b, lambda: {"x": 1.0},
                          ARITHMETIC_PRELUDE, samples=4)

    def test_unbound_variable_propagates(self):
        # The sampler env omits `y`, which the expressions use: structural
        # mismatch, propagates rather than skipping to a vacuous True.
        a = ["+", "x", "y"]
        b = ["+", "y", "x"]
        with pytest.raises(NumevalError):
            numeric_equiv(a, b, lambda: {"x": 1.0},
                          ARITHMETIC_PRELUDE, samples=4)

    def test_bool_result_does_not_match_number(self):
        # (> x 0) returns a bool; comparing it to the number 1 is False even
        # though Python makes True == 1.
        a = [">", "x", 0]
        b = 1
        assert numeric_equiv(a, b, lambda: {"x": 5.0},
                             PREDICATE_PRELUDE, samples=3) is False

    def test_fraction_result_is_exact_and_compares(self):
        # (/ 1 3) evaluates to an exact Fraction and compares True against a
        # float approximation within tolerance.
        assert numeval(["/", 1, 3], {}, ARITHMETIC_PRELUDE) == Fraction(1, 3)
        assert numeric_equiv(["/", 1, 3], 0.3333333333, lambda: {},
                             ARITHMETIC_PRELUDE, samples=2, tol=1e-6) is True


class TestNumevalExports:
    def test_non_core_but_importable_from_submodule(self):
        # numeval is the OPTIONAL numeric-evaluation layer: not in the
        # `rerum` core API, imported explicitly from rerum.numeval.
        import rerum
        # Not part of the public core API (the submodule rerum.numeval is
        # always accessible, as for any package).
        assert "numeval" not in rerum.__all__
        assert "numeric_equiv" not in rerum.__all__
        from rerum.numeval import (numeval as ne, numeric_equiv as nq,
                                   NumevalError as err, NumevalDomainError as derr)
        assert ne is numeval and nq is numeric_equiv
        assert err is NumevalError and derr is NumevalDomainError
