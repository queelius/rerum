"""Tests for the EquivalenceClass value object.

EquivalenceClass captures a starting expression and rule set, then exposes
``iter``, ``enumerate``, ``contains``, ``prove``, ``minimum``, ``sample``,
``random``, ``walk`` as methods. Replaces the eight engine-level methods
with one constructor + an object that knows the question.
"""

import random
import pytest
from rerum import RuleEngine
from rerum.engine import EquivalenceClass


@pytest.fixture
def commute_engine():
    return RuleEngine.from_dsl("@commute: (+ ?x ?y) <=> (+ :y :x)")


@pytest.fixture
def algebra_engine():
    return RuleEngine.from_dsl(
        """
        @add-zero: (+ ?x 0) => :x
        @comm: (+ ?x ?y) <=> (+ :y :x)
        @assoc: (+ (+ ?x ?y) ?z) <=> (+ :x (+ :y :z))
        """
    )


class TestEquivalenceClassConstruction:
    def test_engine_equivalence_class_returns_object(self, commute_engine):
        cls = commute_engine.equivalence_class(["+", "a", "b"])
        assert isinstance(cls, EquivalenceClass)
        assert cls.expr == ["+", "a", "b"]

    def test_default_rules_is_bidirectional_only(self, algebra_engine):
        cls = algebra_engine.equivalence_class(["+", "a", "b"])
        # add-zero is unidirectional; should not be in default rule set
        for _, _, meta in cls.rules:
            assert meta.bidirectional is True

    def test_explicit_rules_overrides_default(self, algebra_engine):
        cls = algebra_engine.equivalence_class(
            ["+", "a", 0],
            rules=algebra_engine.rule_set(),  # full set
        )
        # add-zero now in scope
        result = cls.minimum(metric="size")
        assert result.expr == "a"


class TestEquivalenceClassIteration:
    def test_iter_yields_original_first(self, commute_engine):
        cls = commute_engine.equivalence_class(["+", "a", "b"])
        forms = list(cls.iter(max_depth=2))
        assert forms[0] == ["+", "a", "b"]

    def test_enumerate_collects_to_list(self, commute_engine):
        cls = commute_engine.equivalence_class(["+", "a", "b"])
        forms = cls.enumerate(max_depth=2)
        assert {tuple(f) for f in forms} == {("+", "a", "b"), ("+", "b", "a")}

    def test_iter_via_default_iter_dunder(self, commute_engine):
        cls = commute_engine.equivalence_class(["+", "a", "b"])
        forms = list(cls)
        assert ["+", "a", "b"] in forms
        assert ["+", "b", "a"] in forms


class TestEquivalenceClassMembership:
    def test_contains_true_for_member(self, commute_engine):
        cls = commute_engine.equivalence_class(["+", "a", "b"])
        assert cls.contains(["+", "b", "a"])

    def test_contains_false_for_non_member(self, commute_engine):
        cls = commute_engine.equivalence_class(["+", "a", "b"])
        assert not cls.contains(["*", "a", "b"])

    def test_in_operator_works(self, commute_engine):
        cls = commute_engine.equivalence_class(["+", "a", "b"])
        assert ["+", "b", "a"] in cls
        assert ["*", "a", "b"] not in cls


class TestEquivalenceClassProof:
    def test_prove_returns_proof_for_member(self, commute_engine):
        cls = commute_engine.equivalence_class(["+", "a", "b"])
        proof = cls.prove(["+", "b", "a"])
        assert proof is not None
        assert proof.expr_a == ["+", "a", "b"]
        assert proof.expr_b == ["+", "b", "a"]

    def test_prove_returns_none_for_non_member(self, commute_engine):
        cls = commute_engine.equivalence_class(["+", "a", "b"])
        assert cls.prove(["*", "a", "b"]) is None


class TestEquivalenceClassMinimum:
    def test_minimum_with_metric(self, algebra_engine):
        cls = algebra_engine.equivalence_class(
            ["+", "a", 0],
            rules=algebra_engine.rule_set(),
        )
        result = cls.minimum(metric="size")
        assert result.expr == "a"
        assert result.cost == 1


class TestEquivalenceClassSampling:
    def test_random_returns_member(self, commute_engine):
        rng = random.Random(42)
        cls = commute_engine.equivalence_class(["+", "a", "b"])
        result = cls.random(steps=5, rng=rng)
        assert result in (["+", "a", "b"], ["+", "b", "a"])

    def test_sample_returns_n_members(self, algebra_engine):
        rng = random.Random(42)
        cls = algebra_engine.equivalence_class(["+", ["+", "a", "b"], "c"])
        samples = cls.sample(n=3, steps=5, rng=rng, unique=True)
        # All samples should be in the equivalence class
        for s in samples:
            assert cls.contains(s)

    def test_walk_yields_lazily(self, commute_engine):
        rng = random.Random(42)
        cls = commute_engine.equivalence_class(["+", "a", "b"])
        # Take 3 steps from the walk
        walker = cls.walk(max_steps=10, rng=rng)
        steps = []
        for i, step in enumerate(walker):
            steps.append(step)
            if i >= 2:
                break
        assert len(steps) >= 1


class TestEquivalenceClassRepr:
    def test_repr_includes_expr(self, commute_engine):
        cls = commute_engine.equivalence_class(["+", "a", "b"])
        r = repr(cls)
        assert "EquivalenceClass" in r
        assert "(+" in r  # the formatted expression
