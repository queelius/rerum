"""Tests for exact rational arithmetic (Fraction folds and formatting)."""

from fractions import Fraction

import pytest

from rerum.rewriter import (
    coerce_number, safe_div, nary_fold, instantiate, Bindings,
    rewriter, ARITHMETIC_PRELUDE,
)
from rerum.expr import format_sexpr, parse_sexpr


class TestCoerceNumber:
    def test_int_passthrough(self):
        assert coerce_number(5) == 5
        assert isinstance(coerce_number(5), int)

    def test_float_integral_narrows_to_int(self):
        out = coerce_number(4.0)
        assert out == 4
        assert isinstance(out, int)

    def test_float_non_integral_stays_float(self):
        out = coerce_number(1.5)
        assert out == 1.5
        assert isinstance(out, float)

    def test_fraction_whole_collapses_to_int(self):
        out = coerce_number(Fraction(6, 3))
        assert out == 2
        assert isinstance(out, int)

    def test_fraction_non_integral_stays_fraction(self):
        out = coerce_number(Fraction(1, 3))
        assert out == Fraction(1, 3)
        assert isinstance(out, Fraction)

    def test_fraction_is_never_silently_floated(self):
        out = coerce_number(Fraction(1, 3))
        assert not isinstance(out, float)


class TestCoerceNumberBoolSafety:
    """bool/int footgun: Python ``True == 1`` and ``isinstance(True, int)``.

    ``coerce_number`` must guard ``bool`` FIRST and return the SAME bool
    object: it must never narrow ``True`` -> ``1`` or ``False`` -> ``0``.
    """

    def test_true_stays_true_object(self):
        assert coerce_number(True) is True

    def test_false_stays_false_object(self):
        assert coerce_number(False) is False

    def test_bool_not_narrowed_to_int(self):
        # Identity is the strong check, but pin the type too: a bool stays
        # a bool, it does not become a plain int 1/0.
        assert isinstance(coerce_number(True), bool)
        assert isinstance(coerce_number(False), bool)


class TestSafeDivFraction:
    def test_non_integral_int_division_returns_fraction(self):
        handler = safe_div()
        out = handler([1, 3])
        assert out == Fraction(1, 3)
        assert isinstance(out, Fraction)

    def test_integral_int_division_collapses_to_int(self):
        handler = safe_div()
        out = handler([6, 3])
        assert out == 2
        assert isinstance(out, int)

    def test_division_by_zero_returns_none(self):
        handler = safe_div()
        assert handler([1, 0]) is None

    def test_float_division_stays_float(self):
        handler = safe_div()
        out = handler([1.0, 4.0])
        assert out == 0.25
        assert isinstance(out, float)


class TestNaryFoldFraction:
    def test_sum_of_fractions_is_exact(self):
        add = nary_fold(0, lambda a, b: a + b)
        out = add([Fraction(1, 3), Fraction(1, 6)])
        assert out == Fraction(1, 2)
        assert isinstance(out, Fraction)

    def test_product_collapsing_to_int(self):
        mul = nary_fold(1, lambda a, b: a * b)
        out = mul([Fraction(2, 3), 3])
        assert out == 2
        assert isinstance(out, int)

    def test_plain_int_sum_unchanged(self):
        add = nary_fold(0, lambda a, b: a + b)
        out = add([1, 2, 3])
        assert out == 6
        assert isinstance(out, int)


class TestFractionFormat:
    def test_format_fraction_as_div(self):
        assert format_sexpr(Fraction(1, 3)) == "(/ 1 3)"

    def test_format_negative_fraction(self):
        assert format_sexpr(Fraction(-1, 3)) == "(/ -1 3)"

    def test_format_roundtrip_parses_back_to_div_form(self):
        s = format_sexpr(Fraction(2, 5))
        assert s == "(/ 2 5)"
        assert parse_sexpr(s) == ["/", 2, 5]

    def test_format_inside_compound(self):
        assert format_sexpr(["+", "x", Fraction(1, 2)]) == "(+ x (/ 1 2))"

    def test_format_whole_fraction_after_coercion_is_int(self):
        # coerce_number(Fraction(4,2)) -> 2; format is the int, not a div.
        assert format_sexpr(coerce_number(Fraction(4, 2))) == "2"


class TestInstantiateRationals:
    def test_compute_keeps_fraction_exact(self):
        # (! / 1 3) must stay Fraction(1, 3), not collapse to float.
        skel = ["!", "/", 1, 3]
        out = instantiate(skel, Bindings.empty(), ARITHMETIC_PRELUDE)
        assert out == Fraction(1, 3)
        assert isinstance(out, Fraction)

    def test_compute_collapses_whole_division_to_int(self):
        skel = ["!", "/", 6, 3]
        out = instantiate(skel, Bindings.empty(), ARITHMETIC_PRELUDE)
        assert out == 2
        assert isinstance(out, int)

    def test_compute_fraction_sum(self):
        skel = ["!", "+", ["!", "/", 1, 3], ["!", "/", 1, 6]]
        out = instantiate(skel, Bindings.empty(), ARITHMETIC_PRELUDE)
        assert out == Fraction(1, 2)
        assert isinstance(out, Fraction)


class TestRewriterFastPathRationals:
    """Pin the second renarrowing site: ``try_constant_fold`` inside the
    ``rewriter()`` fast-path simplifier, reached for constant operands.
    """

    def test_fastpath_division_stays_fraction(self):
        simplify = rewriter([], fold_funcs=ARITHMETIC_PRELUDE)
        out = simplify(["/", 1, 3])
        assert out == Fraction(1, 3)
        assert isinstance(out, Fraction)

    def test_fastpath_division_whole_collapses_to_int(self):
        simplify = rewriter([], fold_funcs=ARITHMETIC_PRELUDE)
        out = simplify(["/", 6, 3])
        assert out == 2
        assert isinstance(out, int)

    def test_fastpath_folds_fraction_operands(self):
        # A Fraction already present as an operand must be admitted by the
        # numeric-constant guard and folded exactly.
        simplify = rewriter([], fold_funcs=ARITHMETIC_PRELUDE)
        out = simplify(["+", Fraction(1, 3), Fraction(1, 6)])
        assert out == Fraction(1, 2)
        assert isinstance(out, Fraction)


class TestRationalExports:
    def test_export(self):
        import rerum
        assert rerum.coerce_number is coerce_number
