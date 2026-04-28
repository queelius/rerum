"""Tests for the RuleSet view abstraction.

A RuleSet is a filtered view over an engine's rules. Equivalence-class
methods (equivalents, prove_equal, minimize, etc.) accept ``rules=`` to
restrict the rule subset without per-call ``include_unidirectional`` /
``groups`` kwargs.
"""

import pytest
from rerum import RuleEngine
from rerum.engine import RuleSet


class TestRuleSetBasics:
    def test_rule_set_returns_full_view_by_default(self):
        engine = RuleEngine.from_dsl(
            """
            @r1: (a ?x) => :x
            @r2: (b ?x) => :x
            """
        )
        rs = engine.rule_set()
        assert len(rs) == 2

    def test_bidirectional_only_filters_unidirectional(self):
        engine = RuleEngine.from_dsl(
            """
            @r1: (a ?x) => :x
            @comm: (+ ?x ?y) <=> (+ :y :x)
            """
        )
        # 1 unidirectional + 2 bidirectional pairs = 3 rules total in storage
        assert len(engine.rule_set()) == 3
        bidi = engine.rule_set().bidirectional_only()
        assert len(bidi) == 2  # only the -fwd and -rev pair
        for _, _, meta in bidi:
            assert meta.bidirectional is True

    def test_unidirectional_only_filters_bidirectional(self):
        engine = RuleEngine.from_dsl(
            """
            @r1: (a ?x) => :x
            @comm: (+ ?x ?y) <=> (+ :y :x)
            """
        )
        uni = engine.rule_set().unidirectional_only()
        assert len(uni) == 1
        for _, _, meta in uni:
            assert meta.bidirectional is False

    def test_in_groups_filters_by_tag(self):
        engine = RuleEngine.from_dsl(
            """
            [algebra]
            @r1: (a ?x) => :x

            [calculus]
            @r2: (b ?x) => :x
            """
        )
        algebra = engine.rule_set().in_groups(["algebra"])
        names = [m.name for _, _, m in algebra]
        assert "r1" in names
        assert "r2" not in names

    def test_filters_chain(self):
        engine = RuleEngine.from_dsl(
            """
            [algebra]
            @r1: (a ?x) => :x
            @comm: (+ ?x ?y) <=> (+ :y :x)
            """
        )
        view = engine.rule_set().in_groups(["algebra"]).bidirectional_only()
        assert len(view) == 2  # comm-fwd, comm-rev


class TestRuleSetParameterPropagation:
    def test_equivalents_accepts_rules_kwarg(self):
        engine = RuleEngine.from_dsl(
            """
            @add-zero: (+ ?x 0) => :x
            @comm: (+ ?x ?y) <=> (+ :y :x)
            """
        )
        # Strict equivalence (only <=>): commute alone gives 2 forms.
        rules = engine.rule_set().bidirectional_only()
        forms = list(engine.equivalents(["+", "a", "b"], rules=rules, max_depth=3))
        assert set(map(tuple, forms)) == {("+", "a", "b"), ("+", "b", "a")}

    def test_minimize_accepts_rules_kwarg(self):
        engine = RuleEngine.from_dsl(
            """
            @add-zero: (+ ?x 0) => :x
            @comm: (+ ?x ?y) <=> (+ :y :x)
            """
        )
        # Including unidirectional: minimize finds the (+ a 0) -> a reduction.
        result = engine.minimize(
            ["+", "a", 0],
            metric="size",
            rules=engine.rule_set(),  # full set including unidirectional
        )
        assert result.expr == "a"

    def test_prove_equal_accepts_rules_kwarg(self):
        engine = RuleEngine.from_dsl(
            """
            @comm: (+ ?x ?y) <=> (+ :y :x)
            """
        )
        proof = engine.prove_equal(
            ["+", "a", "b"], ["+", "b", "a"],
            rules=engine.rule_set().bidirectional_only(),
        )
        assert proof is not None

    def test_random_equivalent_accepts_rules_kwarg(self):
        import random
        engine = RuleEngine.from_dsl(
            "@comm: (+ ?x ?y) <=> (+ :y :x)"
        )
        rng = random.Random(42)
        result = engine.random_equivalent(
            ["+", "a", "b"], steps=5, rng=rng,
            rules=engine.rule_set().bidirectional_only(),
        )
        # Result is in the equivalence class
        assert result in (["+", "a", "b"], ["+", "b", "a"])


class TestRuleSetTruthiness:
    def test_empty_rule_set_is_falsy(self):
        engine = RuleEngine()
        assert not engine.rule_set()

    def test_non_empty_rule_set_is_truthy(self):
        engine = RuleEngine.from_dsl("@r: (a ?x) => :x")
        assert engine.rule_set()
