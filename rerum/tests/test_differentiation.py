"""Tests for examples/differentiation.rules + examples/calculus_checker.py.

These tests exercise the GENERAL engine through example content: they load
the example rule files, theory, and metadata sidecar, differentiate concrete
expressions on the existing ``simplify`` driver with a ``normalize`` finishing
pass, and confirm each answer numerically with the example checker
``is_derivative`` (built on the general ``rerum.numeval``). They would be
deleted or swapped if the example changed; the engine would not.
"""

import importlib.util
from pathlib import Path

import pytest

from rerum import RuleEngine, MATH_PRELUDE, PREDICATE_PRELUDE, combine_preludes
from rerum.engine import parse_sexpr, format_sexpr
from rerum.normalize import normalize, Theory

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
DIFF_RULES = EXAMPLES_DIR / "differentiation.rules"
DIFF_SIDECAR = EXAMPLES_DIR / "differentiation.metadata.json"
ALGEBRA_RULES = EXAMPLES_DIR / "algebra.rules"
ARITH_THEORY = EXAMPLES_DIR / "arithmetic.theory.json"
CHECKER_PATH = EXAMPLES_DIR / "calculus_checker.py"


def _load_checker():
    """Import examples/calculus_checker.py by path (it is example content,
    never importable as a package module)."""
    spec = importlib.util.spec_from_file_location("calculus_checker",
                                                  CHECKER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCheckerElementary:
    """is_derivative on hand-computed elementary derivatives."""

    def test_constant_derivative_is_zero(self):
        checker = _load_checker()
        assert checker.is_derivative("5", "x", "0") is True

    def test_identity_derivative_is_one(self):
        checker = _load_checker()
        assert checker.is_derivative("x", "x", "1") is True

    def test_power_rule(self):
        checker = _load_checker()
        assert checker.is_derivative("(^ x 3)", "x", "(* 3 (^ x 2))") is True

    def test_product_x_squared(self):
        checker = _load_checker()
        assert checker.is_derivative("(* x x)", "x", "(* 2 x)") is True

    def test_sum_derivative(self):
        checker = _load_checker()
        assert checker.is_derivative("(+ (^ x 2) x)", "x",
                                     "(+ (* 2 x) 1)") is True

    def test_wrong_derivative_rejected(self):
        checker = _load_checker()
        assert checker.is_derivative("(^ x 3)", "x",
                                     "(* 2 (^ x 2))") is False


class TestCheckerTranscendental:
    """is_derivative across the transcendental families and domain skipping."""

    def test_sin(self):
        checker = _load_checker()
        assert checker.is_derivative("(sin x)", "x", "(cos x)") is True

    def test_cos(self):
        checker = _load_checker()
        assert checker.is_derivative("(cos x)", "x", "(- (sin x))") is True

    def test_tan(self):
        checker = _load_checker()
        assert checker.is_derivative("(tan x)", "x", "(^ (sec x) 2)") is True

    def test_exp(self):
        checker = _load_checker()
        assert checker.is_derivative("(exp x)", "x", "(exp x)") is True

    def test_ln(self):
        checker = _load_checker()
        assert checker.is_derivative("(ln x)", "x", "(/ 1 x)") is True

    def test_sqrt(self):
        checker = _load_checker()
        assert checker.is_derivative("(sqrt x)", "x",
                                     "(/ 1 (* 2 (sqrt x)))") is True

    def test_asin(self):
        checker = _load_checker()
        assert checker.is_derivative("(asin x)", "x",
                                     "(/ 1 (sqrt (- 1 (^ x 2))))") is True

    def test_atan(self):
        checker = _load_checker()
        assert checker.is_derivative("(atan x)", "x",
                                     "(/ 1 (+ 1 (^ x 2)))") is True

    def test_sinh(self):
        checker = _load_checker()
        assert checker.is_derivative("(sinh x)", "x", "(cosh x)") is True

    def test_tanh(self):
        checker = _load_checker()
        assert checker.is_derivative("(tanh x)", "x",
                                     "(- 1 (^ (tanh x) 2))") is True

    def test_log_base(self):
        checker = _load_checker()
        assert checker.is_derivative("(log 2 x)", "x",
                                     "(/ 1 (* x (ln 2)))") is True

    def test_wrong_transcendental_rejected(self):
        checker = _load_checker()
        assert checker.is_derivative("(sin x)", "x", "(sin x)") is False
