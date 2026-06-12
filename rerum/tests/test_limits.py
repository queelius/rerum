"""Tests for limit evaluation as pure example content (examples/limits.rules + solve).

Drives the GENERAL engine through example rule data; L'Hopital reuses the D1
differentiation rules loaded into the SAME engine. No engine code knows what
`lim` means. The goal predicate here is "a numeric atom remains" -- limit
EVALUATION is solved when a number is reached -- which differs from
integration's "no int operator remains": same engine, same search, different
caller-supplied goals.
"""

import importlib.util
from pathlib import Path

import pytest

from rerum.engine import RuleEngine
from rerum.expr import parse_sexpr as E
from rerum.rewriter import (MATH_PRELUDE, NUMERIC_TYPES, PREDICATE_PRELUDE,
                            combine_preludes)
from rerum.solve import contains_op, solve

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _limit_fold_ops():
    """Import examples/limits_fold_ops.py by path (example content, not a
    package module)."""
    spec = importlib.util.spec_from_file_location(
        "limits_fold_ops", EXAMPLES / "limits_fold_ops.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LIMIT_FOLD_OPS


def _limits_prelude():
    # math + predicates + the limit fold ops (subst / defined-at? /
    # indeterminate?), supplied by the example module (they are content,
    # not engine code; the public preludes do not carry them).
    return combine_preludes(MATH_PRELUDE, PREDICATE_PRELUDE,
                            _limit_fold_ops())


def _limits_engine():
    """Engine with differentiation + algebra + limits rules under the prelude."""
    eng = RuleEngine().with_prelude(_limits_prelude())
    eng.load_file(EXAMPLES / "differentiation.rules")
    eng.load_file(EXAMPLES / "algebra.rules")
    eng.load_file(EXAMPLES / "limits.rules", validate_examples=False)
    eng.load_metadata_json((EXAMPLES / "limits.metadata.json").read_text(),
                           validate_examples=True)
    return eng


def _solve_limit(eng, sexpr, max_nodes=4000):
    # The answer to a limit query is a NUMBER: searching merely for a
    # lim-free expression would stop at an unfolded (+ 2 1).
    goal = lambda e: isinstance(e, NUMERIC_TYPES) and not isinstance(e, bool)
    return solve(eng, E(sexpr), goal, max_nodes=max_nodes)


class TestDirectSubstitution:
    def test_polynomial_limit(self):
        eng = _limits_engine()
        res = _solve_limit(eng, "(lim (+ x 1) x 2)")  # lim_{x->2}(x+1)=3
        assert res.found is True
        assert res.solution == 3
        assert not contains_op(res.solution, {"lim"})

    def test_constant_limit(self):
        eng = _limits_engine()
        res = _solve_limit(eng, "(lim 7 x 5)")  # lim_{x->5} 7 = 7
        assert res.found is True
        assert res.solution == 7

    def test_substitution_does_not_fire_on_indeterminate(self):
        eng = _limits_engine()
        # lim_{x->0}(x/x): direct substitution must NOT produce 0/0; closed
        # by div-same (algebra) -> 1, or by L'Hopital. Either way -> 1.
        res = _solve_limit(eng, "(lim (/ x x) x 0)")
        assert res.found is True
        assert res.solution == 1


class TestLimitsRulesLoadAndValidate:
    def test_engine_builds_with_validated_examples(self):
        eng = _limits_engine()
        names = [meta.name for _i, _r, meta in eng.iter_rules()]
        assert "lim-subst" in names


class TestLHopital:
    def test_sin_x_over_x(self):
        # lim_{x->0} sin(x)/x is 0/0; L'Hopital -> lim cos(x)/1 = 1.
        eng = _limits_engine()
        res = _solve_limit(eng, "(lim (/ (sin x) x) x 0)")
        assert res.found is True
        assert res.solution == 1
        assert not contains_op(res.solution, {"lim"})

    def test_lhopital_derivation_uses_dd(self):
        # A 0/0 form with NO named identity, so the search must take the
        # L'Hopital path: lim_{x->0} (e^x - 1)/x = 1. (sin x / x closes via
        # the one-step lim-sinc identity by design once known-limits load.)
        eng = _limits_engine()
        res = _solve_limit(eng, "(lim (/ (- (exp x) 1) x) x 0)")
        assert res.found is True
        assert res.solution == 1
        rule_names = [s.metadata.name for s in res.derivation.steps]
        assert "lim-lhopital" in rule_names
        assert any(n and n.startswith("dd-") for n in rule_names)

    def test_one_minus_cos_over_x(self):
        # lim_{x->0} (1 - cos x)/x is 0/0; L'Hopital -> lim sin(x)/1 = 0.
        eng = _limits_engine()
        res = _solve_limit(eng, "(lim (/ (- 1 (cos x)) x) x 0)")
        assert res.found is True
        assert res.solution == 0


class TestKnownLimits:
    def test_sin_x_over_x_known_limit_fires(self):
        eng = _limits_engine()
        res = _solve_limit(eng, "(lim (/ (sin x) x) x 0)")
        assert res.found is True
        assert res.solution == 1
        names = [s.metadata.name for s in res.derivation.steps]
        assert "lim-sinc" in names

    def test_one_minus_cos_over_x_known_limit(self):
        eng = _limits_engine()
        res = _solve_limit(eng, "(lim (/ (- 1 (cos x)) x) x 0)")
        assert res.found is True
        assert res.solution == 0
        names = [s.metadata.name for s in res.derivation.steps]
        assert "lim-vers" in names


class TestAlgebraicLimits:
    def test_difference_of_squares_over_linear(self):
        # lim_{x->1} (x^2 - 1)/(x - 1) = 2.
        eng = _limits_engine()
        res = _solve_limit(eng, "(lim (/ (- (^ x 2) 1) (- x 1)) x 1)",
                           max_nodes=8000)
        assert res.found is True
        assert res.solution == 2
        assert not contains_op(res.solution, {"lim"})


def _load_checker():
    spec = importlib.util.spec_from_file_location(
        "calculus_checker", EXAMPLES / "calculus_checker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestLimitsEndToEndVerified:
    @pytest.mark.parametrize("body,point,answer", [
        ("(+ x 1)", 2, 3),
        ("(/ (sin x) x)", 0, 1),
        ("(/ (- 1 (cos x)) x)", 0, 0),
        ("(/ (- (^ x 2) 1) (- x 1))", 1, 2),
    ])
    def test_solve_and_is_limit_agree(self, body, point, answer):
        eng = _limits_engine()
        checker = _load_checker()
        res = _solve_limit(eng, f"(lim {body} x {point})", max_nodes=8000)
        assert res.found is True, f"solve failed to close {body}"
        assert res.solution == answer
        assert checker.is_limit(E(body), "x", point, answer) is True
        # Derivation is reconstructible: replaying step.after reaches the
        # solution.
        current = res.derivation.initial
        for step in res.derivation.steps:
            current = step.after
        assert current == res.solution


class TestLnLimits:
    def test_ln_limit_solves(self):
        # lim_{x->1} ln(x) = 0. defined-at? evaluates via numeval, whose
        # prelude must know the rule files' op NAME ln (MATH_PRELUDE's
        # natural log is named log) -- without the alias this limit was
        # silently unsolvable (found=False) and fold-ln was dead.
        eng = _limits_engine()
        res = _solve_limit(eng, "(lim (ln x) x 1)")
        assert res.found is True
        assert res.solution == 0
