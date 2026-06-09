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
