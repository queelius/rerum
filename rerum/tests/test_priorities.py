"""Tests for rule priorities."""

import pytest
from rerum import RuleEngine, E


class TestPriorityParsing:
    """Tests for parsing rules with priority."""

    def test_parse_priority(self):
        """Priority is parsed correctly."""
        engine = RuleEngine.from_dsl("@rule[100]: (f ?x) => :x")

        assert engine._metadata[0].priority == 100

    def test_parse_priority_with_description(self):
        """Priority works with description."""
        engine = RuleEngine.from_dsl('@rule[50] "A rule": (f ?x) => :x')

        assert engine._metadata[0].name == "rule"
        assert engine._metadata[0].priority == 50
        assert engine._metadata[0].description == "A rule"

    def test_parse_no_priority(self):
        """Rules without priority default to 0."""
        engine = RuleEngine.from_dsl("@rule: (f ?x) => :x")

        assert engine._metadata[0].priority == 0

    def test_priority_repr(self):
        """RuleMetadata repr includes priority when non-zero."""
        engine = RuleEngine.from_dsl('''
            @high[100]: (f ?x) => :x
            @low: (g ?x) => :x
        ''')

        # High priority shows in repr
        assert "[100]" in repr(engine._metadata[0])

        # Zero priority doesn't show
        assert "[0]" not in repr(engine._metadata[1])


class TestPriorityOrdering:
    """Tests for priority-based rule ordering."""

    def test_higher_priority_first(self):
        """Higher priority rules fire first."""
        engine = RuleEngine.from_dsl('''
            @low: (f ?x) => low
            @high[100]: (f ?x) => high
        ''')

        # High priority should fire even though low was defined first
        result = engine(E("(f a)"))
        assert result == "high"

    def test_priority_ordering(self):
        """Rules are sorted by priority descending."""
        engine = RuleEngine.from_dsl('''
            @p0: (f ?x) => p0
            @p100[100]: (f ?x) => p100
            @p50[50]: (f ?x) => p50
        ''')

        # Check order: p100, p50, p0
        assert engine._metadata[0].name == "p100"
        assert engine._metadata[1].name == "p50"
        assert engine._metadata[2].name == "p0"

    def test_equal_priority_maintains_order(self):
        """Rules with equal priority maintain definition order."""
        engine = RuleEngine.from_dsl('''
            @first: (f ?x) => first
            @second: (f ?x) => second
            @third: (f ?x) => third
        ''')

        # All priority 0, should maintain order
        assert engine._metadata[0].name == "first"
        assert engine._metadata[1].name == "second"
        assert engine._metadata[2].name == "third"

    def test_equal_priority_with_mixed(self):
        """Stable sort: equal priorities keep relative order."""
        engine = RuleEngine.from_dsl('''
            @first[50]: (f ?x) => first
            @second[50]: (f ?x) => second
            @high[100]: (f ?x) => high
            @third[50]: (f ?x) => third
        ''')

        # high first, then first/second/third in original order
        names = [m.name for m in engine._metadata]
        assert names == ["high", "first", "second", "third"]


class TestPriorityWithStrategies:
    """Tests for priorities with different strategies."""

    def test_priority_with_once(self):
        """Priorities work with 'once' strategy."""
        engine = RuleEngine.from_dsl('''
            @low: (f ?x) => low
            @high[100]: (f ?x) => high
        ''')

        result = engine(E("(f a)"), strategy="once")
        assert result == "high"

    def test_priority_with_bottomup(self):
        """Priorities work with 'bottomup' strategy."""
        engine = RuleEngine.from_dsl('''
            @low: (inner) => low
            @high[100]: (inner) => high
        ''')

        result = engine(E("(outer (inner))"), strategy="bottomup")
        assert result == ["outer", "high"]

    def test_priority_with_topdown(self):
        """Priorities work with 'topdown' strategy."""
        engine = RuleEngine.from_dsl('''
            @low: (outer ?x) => (result :x)
            @high[100]: (outer ?x) => (priority :x)
        ''')

        result = engine(E("(outer a)"), strategy="topdown")
        assert result == ["priority", "a"]


class TestPriorityWithApplyOnce:
    """Tests for priorities with apply_once."""

    def test_apply_once_uses_priority(self):
        """apply_once respects priority order."""
        engine = RuleEngine.from_dsl('''
            @low: (f ?x) => (low :x)
            @high[100]: (f ?x) => (high :x)
        ''')

        result, meta = engine.apply_once(E("(f a)"))
        assert meta.name == "high"
        assert result == ["high", "a"]


class TestPriorityWithGuards:
    """Tests for priorities combined with guards."""

    def test_priority_with_guards(self):
        """Priority takes precedence, then guards checked."""
        from rerum import FULL_PRELUDE

        engine = (RuleEngine()
            .with_prelude(FULL_PRELUDE)
            .load_dsl('''
                @low: (f ?x) => fallback
                @high[100]: (f ?x) => high when (! > :x 0)
            '''))

        # Positive: high priority and guard passes
        assert engine(E("(f 5)")) == "high"

        # Negative: high priority but guard fails, falls to low
        assert engine(E("(f -5)")) == "fallback"

    def test_priority_ordering_with_multiple_guards(self):
        """Multiple guarded rules respect priority."""
        from rerum import FULL_PRELUDE

        engine = (RuleEngine()
            .with_prelude(FULL_PRELUDE)
            .load_dsl('''
                @low[10]: (f ?x) => low-guarded when (! > :x 0)
                @high[100]: (f ?x) => high-guarded when (! < :x 0)
                @fallback: (f ?x) => fallback
            '''))

        # Positive: low-guarded wins (high's guard fails)
        assert engine(E("(f 5)")) == "low-guarded"

        # Negative: high-guarded wins
        assert engine(E("(f -5)")) == "high-guarded"

        # Zero: fallback wins (both guards fail)
        assert engine(E("(f 0)")) == "fallback"


class TestRealWorldPriority:
    """Real-world use cases for priorities."""

    def test_specific_before_general(self):
        """Specific rules can have higher priority than general."""
        engine = RuleEngine.from_dsl('''
            @general: (+ ?x ?y) => (add :x :y)
            @specific[100]: (+ 0 ?x) => :x
            @specific2[100]: (+ ?x 0) => :x
        ''')

        # Specific rules fire first
        assert engine(E("(+ 0 y)")) == "y"
        assert engine(E("(+ x 0)")) == "x"

        # General rule for non-zero cases
        result = engine(E("(+ a b)"))
        assert result == ["add", "a", "b"]

    def test_catch_all_rule(self):
        """Low priority catch-all rule."""
        from rerum import FULL_PRELUDE

        engine = (RuleEngine()
            .with_prelude(FULL_PRELUDE)
            .load_dsl('''
                @fold[100]: (+ ?a ?b) => (! + :a :b) when (! and (! const? :a) (! const? :b))
                @simplify[50]: (+ ?x 0) => :x
                @catch-all: (+ ?x ?y) => (sum :x :y)
            '''))

        # Fold when both constants
        assert engine(E("(+ 1 2)")) == 3

        # Simplify when adding zero
        assert engine(E("(+ x 0)")) == "x"

        # Catch-all for other cases
        result = engine(E("(+ a b)"))
        assert result == ["sum", "a", "b"]
