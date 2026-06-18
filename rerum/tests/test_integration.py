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
    supplies the general search. The arithmetic THEORY is threaded so
    normalize_between canonicalizes nodes -- the u-sub rules match the one
    canonical factor order instead of spelling every order out.
    """
    from rerum.normalize import Theory
    theory = Theory.from_json(
        (EXAMPLES_DIR / "arithmetic.theory.json").read_text())
    goal = lambda e: not contains_op(e, {"int"})
    return solve(eng, ["int", integrand, var], goal, max_nodes=max_nodes,
                 theory=theory, normalize_between=True)


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


class TestUSubstitution:
    def test_int_2x_cos_x2_closes_to_sin_x2(self):
        eng = _integration_engine()
        # int(cos(x^2) * 2x) dx = sin(x^2) via u = x^2.
        integrand = ["*", ["cos", ["^", "x", 2]], ["*", 2, "x"]]
        res = integrate(eng, integrand, "x", max_nodes=3000)
        assert res.found is True
        assert res.solution == ["sin", ["^", "x", 2]]
        assert not contains_op(res.solution, {"int"})

    def test_usub_reversed_factor_order_also_closes(self):
        eng = _integration_engine()
        integrand = ["*", ["*", 2, "x"], ["cos", ["^", "x", 2]]]
        res = integrate(eng, integrand, "x", max_nodes=3000)
        assert res.found is True
        assert res.solution == ["sin", ["^", "x", 2]]


class TestIntegrationByParts:
    def test_int_x_exp_x_closes_int_free(self):
        eng = _integration_engine()
        # int(x * e^x) dx = (x - 1) e^x. Assert int-free + numeric verify
        # (Task I), not a specific normal form (no normalizer loaded).
        integrand = ["*", "x", ["exp", "x"]]
        res = integrate(eng, integrand, "x", max_nodes=5000)
        assert res.found is True
        assert not contains_op(res.solution, {"int"})

    def test_by_parts_only_fires_on_its_product_shape(self):
        eng = _integration_engine()
        # Integrating cos still closes via the table, not via by-parts.
        res = integrate(eng, ["cos", "x"], "x")
        assert res.found is True
        names = [s.metadata.name for s in res.derivation.steps]
        assert not any(n and n.startswith("int-byparts") for n in names)


def _load_checker():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "calculus_checker", EXAMPLES_DIR / "calculus_checker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestIntegrationNumericallyVerified:
    @pytest.mark.parametrize("integrand", [
        ["cos", "x"],
        ["sin", "x"],
        ["*", 2, "x"],
        ["+", "x", ["cos", "x"]],
        ["*", ["cos", ["^", "x", 2]], ["*", 2, "x"]],  # u-sub
        ["*", "x", ["exp", "x"]],                       # by-parts
    ])
    def test_solve_result_differentiates_back(self, integrand):
        eng = _integration_engine()
        checker = _load_checker()
        res = integrate(eng, integrand, "x", max_nodes=5000)
        assert res.found is True
        assert not contains_op(res.solution, {"int"})
        assert checker.is_integral(integrand, "x", res.solution) is True


class TestEveryRuleFiresEndToEnd:
    """Coverage finding: several rules were only exercised by single-step
    sidecar validation, never end-to-end through solve. This drives EVERY
    integration rule via a representative integrand and asserts it appears in
    a solve derivation -- proving none are dead. Some rules (int-sum-one,
    int-const-mult-right) fire only under the NON-theory driver, where
    normalize does not pre-canonicalize them away; integrate_plain() omits
    the theory so those paths are exercised."""

    def _plain(self, integrand, max_nodes=4000):
        from rerum.solve import contains_op, solve
        eng = _integration_engine()
        return solve(eng, ["int", integrand, "x"],
                     lambda e: not contains_op(e, {"int"}),
                     max_nodes=max_nodes)

    # (rule name -> a plain-driver integrand that makes it fire)
    CASES = {
        "int-sum": ["+", "x", ["sin", "x"], ["cos", "x"]],
        "int-sum-one": ["+", ["sin", "x"], ["cos", "x"]],
        "int-diff": ["-", ["sin", "x"], ["cos", "x"]],
        "int-const-mult-left": ["*", 3, ["sin", "x"]],
        "int-const-mult-right": ["*", ["sin", "x"], 3],
        "int-power": ["^", "x", 2],
        "int-var": "x",
        "int-one": 1,
        "int-const": 5,
        "int-recip": ["/", 1, "x"],
        "int-power-neg1": ["^", "x", -1],
        "int-exp": ["exp", "x"],
        "int-sin": ["sin", "x"],
        "int-cos": ["cos", "x"],
        "int-usub-cos-sq": ["*", 2, "x", ["cos", ["^", "x", 2]]],
        "int-usub-sin-sq": ["*", 2, "x", ["sin", ["^", "x", 2]]],
        "int-usub-exp-sq": ["*", 2, "x", ["exp", ["^", "x", 2]]],
        "int-byparts-x-exp": ["*", "x", ["exp", "x"]],
        "int-neg": ["-", ["cos", "x"]],
    }

    @pytest.mark.parametrize("rule,integrand", sorted(CASES.items()))
    def test_rule_fires(self, rule, integrand):
        res = self._plain(integrand)
        assert res.found is True, f"{rule}: solve did not close {integrand}"
        names = [s.metadata.name for s in res.derivation.steps]
        assert rule in names, f"{rule} did not fire; got {names}"

    def test_cases_cover_every_named_rule(self):
        # Guard against a rule being added without coverage here.
        eng = _integration_engine()
        all_names = {m.name for _i, _r, m in eng.iter_rules()}
        assert all_names == set(self.CASES), (
            f"uncovered rules: {all_names - set(self.CASES)}")


class TestNegationLinearity:
    """int-neg (int(-f) = -int(f)) fills a real table gap surfaced by the
    C1 by-parts experiment. Pin it directly + numerically."""

    def test_int_neg_cos_closes_and_verifies(self):
        eng = _integration_engine()
        checker = _load_checker()
        out = integrate(eng, ["-", ["cos", "x"]], "x")
        assert out.found is True
        assert out.solution == ["-", ["sin", "x"]]
        assert checker.is_integral(["-", ["cos", "x"]], "x",
                                   out.solution) is True

    def test_int_neg_fires_in_trace(self):
        eng = _integration_engine()
        out = integrate(eng, ["-", ["sin", "x"]], "x")
        assert out.found is True
        names = [s.metadata.name for s in out.derivation.steps]
        assert "int-neg" in names
