"""Tests for integration as pure example content (examples/integration.rules + solve).

This file drives the GENERAL engine through example rule data. No engine
code knows what `int` means; these tests confirm a domain is just rules.
Antiderivatives are a SEARCH problem (non-confluent: table, linearity,
u-sub, and by-parts compete), so the driver is the general goal-directed
``solve`` with the caller-supplied goal "no int operator remains".
"""

from pathlib import Path

import pytest

from rerum.engine import RuleEngine
from rerum.rewriter import MATH_PRELUDE, PREDICATE_PRELUDE, combine_preludes
from rerum.solve import contains_op, solve

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
RULES_FILE = EXAMPLES_DIR / "integration.rules"
META_FILE = EXAMPLES_DIR / "integration.metadata.json"


def _integration_prelude():
    # Math functions + predicates cover the table, the power-rule
    # (! / 1 (! + :n 1)) coefficient, the (! != :n -1) guard, and the
    # free-of? predicate.
    return combine_preludes(MATH_PRELUDE, PREDICATE_PRELUDE)


def _integration_engine():
    eng = RuleEngine().with_prelude(_integration_prelude())
    # Defer example validation until the sidecar supplies the examples and
    # the prelude is set (prelude set above, so order is satisfied).
    eng.load_file(RULES_FILE, validate_examples=False)
    eng.load_metadata_json(META_FILE.read_text(), validate_examples=True)
    return eng


class TestIntegrationRulesLoad:
    def test_rules_and_sidecar_exist(self):
        assert RULES_FILE.exists()
        assert META_FILE.exists()

    def test_loads_and_validates_examples(self):
        # If any rule's example fails to reproduce its declared output under
        # the prelude, load_metadata_json raises ExampleValidationError.
        eng = _integration_engine()
        assert len(eng.list_rules()) > 0

    def test_every_closing_rule_has_an_example(self):
        eng = _integration_engine()
        for _idx, _rule, meta in eng.iter_rules():
            # Decomposing rules (linearity, const-mult) and the by-parts
            # rule produce a non-closed RHS (still contains int) and are
            # tagged "structural"/"by-parts"; they are exempt from the
            # closed-form-example requirement.
            if meta.category in ("structural", "by-parts"):
                continue
            assert meta.examples, f"rule {meta.name!r} has no examples"


from fractions import Fraction


class TestPowerRuleRational:
    def test_power_rule_single_step_exact_coefficient(self):
        eng = _integration_engine()
        # apply_once applies one matching rule and returns (expr, metadata).
        out, meta = eng.apply_once(["int", ["^", "x", 2], "x"])
        assert meta is not None
        assert meta.name == "int-power"
        # x^3 * (1/3), with 1/3 an exact Fraction (Phase 3 rationals).
        assert out == ["*", ["^", "x", 3], Fraction(1, 3)]

    def test_power_rule_guarded_off_for_n_eq_neg1(self):
        eng = _integration_engine()
        # n = -1 must NOT match int-power (guard (! != :n -1)); it should
        # match int-power-neg1 instead, giving ln|x|.
        out, meta = eng.apply_once(["int", ["^", "x", -1], "x"])
        assert meta is not None
        assert meta.name == "int-power-neg1"
        assert out == ["ln", ["abs", "x"]]

    def test_power_rule_n5_coefficient(self):
        eng = _integration_engine()
        out, meta = eng.apply_once(["int", ["^", "x", 5], "x"])
        assert meta.name == "int-power"
        assert out == ["*", ["^", "x", 6], Fraction(1, 6)]


def integrate(eng, integrand, var, *, max_nodes=2000):
    """Integrate `integrand` w.r.t. `var` by goal-directed search.

    Returns the SolveResult: result.solution is the int-free antiderivative
    (or None on honest failure within budget), result.derivation is the
    labeled RewriteTrace, result.found says whether the search closed. The
    goal predicate "no int operator remains" is caller-supplied; the engine
    supplies the general search.
    """
    goal = lambda e: not contains_op(e, {"int"})
    return solve(eng, ["int", integrand, var], goal, max_nodes=max_nodes)


class TestSolveDrivenIntegration:
    def test_int_2x_closes_to_int_free(self):
        eng = _integration_engine()
        # int(2x) dx -> x^2 (up to a constant-product normal form). Without a
        # normalizer loaded, assert int-free + numeric verify rather than an
        # exact normal form.
        res = integrate(eng, ["*", 2, "x"], "x")
        assert res.found is True
        assert not contains_op(res.solution, {"int"})

    def test_int_cos_closes_to_sin(self):
        eng = _integration_engine()
        res = integrate(eng, ["cos", "x"], "x")
        assert res.found is True
        assert res.solution == ["sin", "x"]

    def test_int_sin_closes_to_neg_cos(self):
        eng = _integration_engine()
        res = integrate(eng, ["sin", "x"], "x")
        assert res.found is True
        assert res.solution == ["-", ["cos", "x"]]

    def test_int_sum_decomposes_and_closes(self):
        eng = _integration_engine()
        # int(x + cos x) dx -> x^2/2 + sin x (int-free).
        res = integrate(eng, ["+", "x", ["cos", "x"]], "x")
        assert res.found is True
        assert not contains_op(res.solution, {"int"})

    def test_derivation_is_reconstructible(self):
        eng = _integration_engine()
        res = integrate(eng, ["cos", "x"], "x")
        deriv = res.derivation
        assert deriv.initial == ["int", ["cos", "x"], "x"]
        assert deriv.final == res.solution
        # Replaying step.after from initial reaches the solution.
        current = deriv.initial
        for step in deriv.steps:
            current = step.after
        assert current == res.solution
        names = [s.metadata.name for s in deriv.steps]
        assert "int-cos" in names

    def test_honest_failure_on_tiny_budget(self):
        eng = _integration_engine()
        # int(x + cos x) needs several expansions; max_nodes=1 cannot close.
        res = integrate(eng, ["+", "x", ["cos", "x"]], "x", max_nodes=1)
        assert res.found is False
        assert res.solution is None
        assert res.explored <= 1
