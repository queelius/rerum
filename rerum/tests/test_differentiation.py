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


def make_diff_engine():
    """Load differentiation + algebra under combine_preludes(MATH_PRELUDE,
    PREDICATE_PRELUDE), examples validated via the metadata sidecar. This is
    the motivating pipeline, exercised end to end through example content.
    (The engine has no with_theory builder; the theory drives the EXPLICIT
    normalize finishing pass in differentiate() below, which is what produces
    clean output.)"""
    engine = (
        RuleEngine()
        .with_prelude(combine_preludes(MATH_PRELUDE, PREDICATE_PRELUDE))
        .load_file(ALGEBRA_RULES)
        .load_file(DIFF_RULES)
    )
    engine.load_metadata_json(DIFF_SIDECAR.read_text(), validate_examples=True)
    return engine


def differentiate(engine, src):
    """Run the existing simplify driver (engine call) then a normalize
    finishing pass under the arithmetic theory."""
    theory = Theory.from_json(ARITH_THEORY.read_text())
    simplified = engine(parse_sexpr(src))
    return normalize(simplified, theory)


class TestTheoryAndLoad:
    def test_theory_declares_plus_times_ac(self):
        # Reuses the Phase 2 examples/arithmetic.theory.json (its repeat
        # declarations are what collect (+ x x) into (* 2 x)).
        theory = Theory.from_json(ARITH_THEORY.read_text())
        assert theory.is_ac("+") is True
        assert theory.is_ac("*") is True
        assert theory.identity("+") == 0
        assert theory.identity("*") == 1
        assert theory.annihilator("*") == 0

    def test_rules_load_and_examples_validate(self):
        # Loading the sidecar with validate_examples=True must not raise:
        # every rule's example is a correct single-step rewrite.
        engine = make_diff_engine()
        assert len(engine) > 0


class TestBasicsAndLinearity:
    def test_constant(self):
        engine = make_diff_engine()
        assert differentiate(engine, "(dd 5 x)") == 0

    def test_variable_same(self):
        engine = make_diff_engine()
        assert differentiate(engine, "(dd x x)") == 1

    def test_variable_other(self):
        engine = make_diff_engine()
        assert differentiate(engine, "(dd y x)") == 0

    def test_sum(self):
        engine = make_diff_engine()
        checker = _load_checker()
        out = differentiate(engine, "(dd (+ x x) x)")
        assert out == 2
        assert checker.is_derivative("(+ x x)", "x", format_sexpr(out)) is True

    def test_difference(self):
        engine = make_diff_engine()
        checker = _load_checker()
        out = differentiate(engine, "(dd (- x 5) x)")
        assert out == 1
        assert checker.is_derivative("(- x 5)", "x", format_sexpr(out)) is True

    def test_constant_multiple(self):
        engine = make_diff_engine()
        checker = _load_checker()
        out = differentiate(engine, "(dd (* 3 x) x)")
        assert out == 3
        assert checker.is_derivative("(* 3 x)", "x", format_sexpr(out)) is True

    def test_nary_sum(self):
        engine = make_diff_engine()
        # d/dx(x + y + x) = 2 (y free of x -> 0); rest-pattern linearity.
        out = differentiate(engine, "(dd (+ x y x) x)")
        assert out == 2


class TestProductsQuotientPower:
    def test_product_x_times_x(self):
        engine = make_diff_engine()
        checker = _load_checker()
        # THE motivating example: d/dx(x*x) = 2x
        out = differentiate(engine, "(dd (* x x) x)")
        assert out == ["*", 2, "x"]
        assert checker.is_derivative("(* x x)", "x", format_sexpr(out)) is True

    def test_product_general(self):
        engine = make_diff_engine()
        checker = _load_checker()
        # d/dx(x^2 * x^3) via the product rule; verify numerically (shape may
        # vary). Transcendental-composed products are tested with the trig
        # family (the sin rule does not exist yet at this point in the file).
        out = differentiate(engine, "(dd (* (^ x 2) (^ x 3)) x)")
        assert checker.is_derivative("(* (^ x 2) (^ x 3))", "x",
                                     format_sexpr(out)) is True

    def test_quotient(self):
        engine = make_diff_engine()
        checker = _load_checker()
        # d/dx(x / (1 + x^2)); quotient rule with a polynomial denominator.
        out = differentiate(engine, "(dd (/ x (+ 1 (^ x 2))) x)")
        assert checker.is_derivative("(/ x (+ 1 (^ x 2)))", "x",
                                     format_sexpr(out)) is True

    def test_power_constant_exponent(self):
        engine = make_diff_engine()
        checker = _load_checker()
        # THE motivating example: d/dx(x^3) = 3 x^2
        out = differentiate(engine, "(dd (^ x 3) x)")
        assert out == ["*", 3, ["^", "x", 2]]
        assert checker.is_derivative("(^ x 3)", "x", format_sexpr(out)) is True

    def test_power_quadratic(self):
        engine = make_diff_engine()
        checker = _load_checker()
        out = differentiate(engine, "(dd (^ x 2) x)")
        assert out == ["*", 2, "x"]
        assert checker.is_derivative("(^ x 2)", "x", format_sexpr(out)) is True


class TestExpLogSqrt:
    def test_exp(self):
        engine = make_diff_engine()
        checker = _load_checker()
        out = differentiate(engine, "(dd (exp x) x)")
        assert checker.is_derivative("(exp x)", "x", format_sexpr(out)) is True

    def test_ln(self):
        engine = make_diff_engine()
        checker = _load_checker()
        out = differentiate(engine, "(dd (ln x) x)")
        assert checker.is_derivative("(ln x)", "x", format_sexpr(out)) is True

    def test_sqrt(self):
        engine = make_diff_engine()
        checker = _load_checker()
        out = differentiate(engine, "(dd (sqrt x) x)")
        assert checker.is_derivative("(sqrt x)", "x", format_sexpr(out)) is True

    def test_log_base(self):
        engine = make_diff_engine()
        checker = _load_checker()
        out = differentiate(engine, "(dd (log 2 x) x)")
        assert checker.is_derivative("(log 2 x)", "x",
                                     format_sexpr(out)) is True

    def test_a_to_the_x(self):
        engine = make_diff_engine()
        checker = _load_checker()
        # d/dx(2^x) = 2^x ln 2 (constant base, variable exponent)
        out = differentiate(engine, "(dd (^ 2 x) x)")
        assert checker.is_derivative("(^ 2 x)", "x", format_sexpr(out)) is True


class TestTrig:
    def test_sin(self):
        engine = make_diff_engine()
        checker = _load_checker()
        out = differentiate(engine, "(dd (sin x) x)")
        assert checker.is_derivative("(sin x)", "x", format_sexpr(out)) is True

    def test_cos(self):
        engine = make_diff_engine()
        checker = _load_checker()
        out = differentiate(engine, "(dd (cos x) x)")
        assert checker.is_derivative("(cos x)", "x", format_sexpr(out)) is True

    def test_tan(self):
        engine = make_diff_engine()
        checker = _load_checker()
        out = differentiate(engine, "(dd (tan x) x)")
        assert checker.is_derivative("(tan x)", "x", format_sexpr(out)) is True

    def test_sec(self):
        engine = make_diff_engine()
        checker = _load_checker()
        out = differentiate(engine, "(dd (sec x) x)")
        assert checker.is_derivative("(sec x)", "x", format_sexpr(out)) is True

    def test_csc(self):
        engine = make_diff_engine()
        checker = _load_checker()
        out = differentiate(engine, "(dd (csc x) x)")
        assert checker.is_derivative("(csc x)", "x", format_sexpr(out)) is True

    def test_cot(self):
        engine = make_diff_engine()
        checker = _load_checker()
        out = differentiate(engine, "(dd (cot x) x)")
        assert checker.is_derivative("(cot x)", "x", format_sexpr(out)) is True

    def test_chain_sin_of_square(self):
        engine = make_diff_engine()
        checker = _load_checker()
        # d/dx(sin(x^2)) = cos(x^2) * 2x ; verify numerically.
        out = differentiate(engine, "(dd (sin (^ x 2)) x)")
        assert checker.is_derivative("(sin (^ x 2))", "x",
                                     format_sexpr(out)) is True

    def test_product_with_sin(self):
        engine = make_diff_engine()
        checker = _load_checker()
        # Deferred from the products task (plan sequencing: needed the sin
        # rule): d/dx(x * sin x) = sin x + x cos x.
        out = differentiate(engine, "(dd (* x (sin x)) x)")
        assert checker.is_derivative("(* x (sin x))", "x",
                                     format_sexpr(out)) is True

    def test_quotient_with_sin(self):
        engine = make_diff_engine()
        checker = _load_checker()
        # Deferred from the products task: d/dx(x / sin x).
        out = differentiate(engine, "(dd (/ x (sin x)) x)")
        assert checker.is_derivative("(/ x (sin x))", "x",
                                     format_sexpr(out)) is True


class TestInverseTrig:
    def test_asin(self):
        engine = make_diff_engine()
        checker = _load_checker()
        out = differentiate(engine, "(dd (asin x) x)")
        assert checker.is_derivative("(asin x)", "x", format_sexpr(out)) is True

    def test_acos(self):
        engine = make_diff_engine()
        checker = _load_checker()
        out = differentiate(engine, "(dd (acos x) x)")
        assert checker.is_derivative("(acos x)", "x", format_sexpr(out)) is True

    def test_atan(self):
        engine = make_diff_engine()
        checker = _load_checker()
        out = differentiate(engine, "(dd (atan x) x)")
        assert checker.is_derivative("(atan x)", "x", format_sexpr(out)) is True


class TestHyperbolic:
    def test_sinh(self):
        engine = make_diff_engine()
        checker = _load_checker()
        out = differentiate(engine, "(dd (sinh x) x)")
        assert checker.is_derivative("(sinh x)", "x", format_sexpr(out)) is True

    def test_cosh(self):
        engine = make_diff_engine()
        checker = _load_checker()
        out = differentiate(engine, "(dd (cosh x) x)")
        assert checker.is_derivative("(cosh x)", "x", format_sexpr(out)) is True

    def test_tanh(self):
        engine = make_diff_engine()
        checker = _load_checker()
        out = differentiate(engine, "(dd (tanh x) x)")
        assert checker.is_derivative("(tanh x)", "x", format_sexpr(out)) is True


class TestGeneralPower:
    def test_x_to_the_x(self):
        engine = make_diff_engine()
        checker = _load_checker()
        # d/dx(x^x) = x^x (ln x + 1) ; verify numerically (domain x>0 by sampling).
        out = differentiate(engine, "(dd (^ x x) x)")
        assert checker.is_derivative("(^ x x)", "x", format_sexpr(out)) is True

    def test_x_to_the_sin_x(self):
        engine = make_diff_engine()
        checker = _load_checker()
        # d/dx(x^(sin x)) via log-diff ; verify numerically.
        out = differentiate(engine, "(dd (^ x (sin x)) x)")
        assert checker.is_derivative("(^ x (sin x))", "x",
                                     format_sexpr(out)) is True

    def test_constant_exponent_still_uses_power_rule(self):
        engine = make_diff_engine()
        # Regression: a numeric exponent must STILL take the clean power rule,
        # not log-diff. d/dx(x^3) = 3 x^2 exactly.
        out = differentiate(engine, "(dd (^ x 3) x)")
        assert out == ["*", 3, ["^", "x", 2]]

    def test_constant_base_still_uses_aexp(self):
        engine = make_diff_engine()
        checker = _load_checker()
        # Regression: a numeric base must STILL take the a^x rule.
        out = differentiate(engine, "(dd (^ 2 x) x)")
        assert checker.is_derivative("(^ 2 x)", "x", format_sexpr(out)) is True


class TestPartialDerivatives:
    def test_partial_treats_other_var_as_constant(self):
        engine = make_diff_engine()
        checker = _load_checker()
        # d/dx(x * y) = y (y free of x; product rule + free-of? -> y*1 + x*0)
        out = differentiate(engine, "(dd (* x y) x)")
        assert out == "y"
        assert checker.is_derivative("(* x y)", "x", format_sexpr(out)) is True

    def test_partial_wrt_y(self):
        engine = make_diff_engine()
        checker = _load_checker()
        out = differentiate(engine, "(dd (* x y) y)")
        assert out == "x"
        assert checker.is_derivative("(* x y)", "y", format_sexpr(out)) is True

    def test_partial_sum_of_two_vars(self):
        engine = make_diff_engine()
        checker = _load_checker()
        # d/dx(x^2 + y^2) = 2x
        out = differentiate(engine, "(dd (+ (^ x 2) (^ y 2)) x)")
        assert out == ["*", 2, "x"]
        assert checker.is_derivative("(+ (^ x 2) (^ y 2))", "x",
                                     format_sexpr(out)) is True

    def test_free_subexpression_is_zero(self):
        engine = make_diff_engine()
        # d/dx(sin y) = 0 directly (free-of? guard at top priority).
        assert differentiate(engine, "(dd (sin y) x)") == 0


class TestMotivatingCleanOutput:
    def test_motivating_examples_are_clean(self):
        engine = make_diff_engine()
        # The two spec/contract motivating cases, exact forms.
        assert differentiate(engine, "(dd (* x x) x)") == ["*", 2, "x"]
        assert differentiate(engine, "(dd (^ x 3) x)") == ["*", 3, ["^", "x", 2]]


class TestEveryRuleHasExample:
    def test_every_named_rule_carries_an_example(self):
        engine = make_diff_engine()
        # After the sidecar merge, every dd-* rule carries at least one
        # validated example (iter_rules is the public state API).
        seen = 0
        for _idx, _rule, meta in engine.iter_rules():
            if meta.name and meta.name.startswith("dd-"):
                assert meta.examples, f"{meta.name} has no example"
                seen += 1
        assert seen >= 25  # every family is present
