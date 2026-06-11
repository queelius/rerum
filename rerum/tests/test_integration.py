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
