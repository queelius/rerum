"""Tests for the source_rules() iterator and BidirectionalRule/UnidirectionalRule
value objects.

Where ``len(engine)`` counts storage entries (a `<=>` rule contributes 2),
``source_rules()`` collapses paired -fwd/-rev entries into a single
``BidirectionalRule``, matching how ``to_dsl``/``to_dict`` already serialize.
"""

import pytest
from rerum import RuleEngine
from rerum.engine import BidirectionalRule, UnidirectionalRule


class TestSourceRulesIteration:
    def test_unidirectional_yields_unidirectional_rule(self):
        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")
        sources = list(engine.source_rules())
        assert len(sources) == 1
        assert isinstance(sources[0], UnidirectionalRule)
        assert sources[0].name == "add-zero"

    def test_bidirectional_yields_one_bidirectional_rule(self):
        engine = RuleEngine.from_dsl("@commute: (+ ?x ?y) <=> (+ :y :x)")
        sources = list(engine.source_rules())
        # Storage has 2 entries (-fwd and -rev), source_rules collapses to 1.
        assert len(engine) == 2
        assert len(sources) == 1
        assert isinstance(sources[0], BidirectionalRule)
        assert sources[0].name == "commute"

    def test_mixed_engine(self):
        engine = RuleEngine.from_dsl(
            """
            @add-zero: (+ ?x 0) => :x
            @comm: (+ ?x ?y) <=> (+ :y :x)
            @assoc: (+ (+ ?x ?y) ?z) <=> (+ :x (+ :y :z))
            """
        )
        sources = list(engine.source_rules())
        assert len(engine) == 5  # 1 + 2 + 2
        assert len(sources) == 3  # 1 uni + 2 bi
        kinds = [type(s).__name__ for s in sources]
        assert kinds == ["UnidirectionalRule", "BidirectionalRule", "BidirectionalRule"]

    def test_anonymous_bidirectional_handled(self):
        engine = RuleEngine.from_dsl("(+ ?x ?y) <=> (+ :y :x)")
        sources = list(engine.source_rules())
        assert len(sources) == 1
        assert isinstance(sources[0], BidirectionalRule)
        assert sources[0].name is None


class TestSourceRuleAttributes:
    def test_bidirectional_preserves_priority(self):
        engine = RuleEngine.from_dsl("@swap[42]: (a ?x ?y) <=> (b :y :x)")
        source = next(engine.source_rules())
        assert source.priority == 42

    def test_bidirectional_preserves_description(self):
        engine = RuleEngine.from_dsl(
            '@assoc "Associativity": (+ (+ ?x ?y) ?z) <=> (+ :x (+ :y :z))'
        )
        source = next(engine.source_rules())
        # source_rules() strips the " (forward)" suffix from the fwd metadata
        assert source.description == "Associativity"

    def test_bidirectional_preserves_both_directions(self):
        engine = RuleEngine.from_dsl("@swap: (a ?x ?y) <=> (b :y :x)")
        source = next(engine.source_rules())
        assert source.fwd_pattern == ["a", ["?", "x"], ["?", "y"]]
        assert source.fwd_skeleton == ["b", [":", "y"], [":", "x"]]
        assert source.rev_pattern == ["b", ["?", "y"], ["?", "x"]]
        assert source.rev_skeleton == ["a", [":", "x"], [":", "y"]]

    def test_unidirectional_preserves_pattern_and_skeleton(self):
        engine = RuleEngine.from_dsl("@r: (foo ?x) => (bar :x)")
        source = next(engine.source_rules())
        assert source.pattern == ["foo", ["?", "x"]]
        assert source.skeleton == ["bar", [":", "x"]]
