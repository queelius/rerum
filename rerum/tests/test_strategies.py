"""Tests for rewriting strategies."""

import pytest
from rerum import RuleEngine, E, ARITHMETIC_PRELUDE


class TestExhaustiveStrategy:
    """Tests for the default exhaustive strategy."""

    def test_exhaustive_is_default(self):
        """Exhaustive strategy is the default."""
        engine = RuleEngine.from_dsl('''
            @add-zero: (+ ?x 0) => :x
            @mul-one: (* ?x 1) => :x
        ''')

        expr = E("(+ (* x 1) 0)")
        # Both rules should apply
        assert engine(expr) == "x"
        assert engine(expr, strategy="exhaustive") == "x"

    def test_exhaustive_applies_all_rules(self):
        """Exhaustive strategy applies rules until fixpoint."""
        engine = RuleEngine.from_dsl('''
            @step1: (a) => (b)
            @step2: (b) => (c)
            @step3: (c) => done
        ''')

        assert engine(E("(a)")) == "done"


class TestOnceStrategy:
    """Tests for the 'once' strategy."""

    def test_once_applies_single_rule(self):
        """Once strategy applies at most one rule."""
        engine = RuleEngine.from_dsl('''
            @step1: (a) => (b)
            @step2: (b) => (c)
        ''')

        # Only one step
        result = engine(E("(a)"), strategy="once")
        assert result == ["b"]

    def test_once_finds_rule_in_subexpression(self):
        """Once strategy searches subexpressions."""
        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")

        expr = E("(f (+ y 0))")
        result = engine(expr, strategy="once")
        assert result == ["f", "y"]

    def test_once_returns_unchanged_if_no_match(self):
        """Once strategy returns original if nothing matches."""
        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")

        expr = E("(* y 1)")
        result = engine(expr, strategy="once")
        assert result == expr

    def test_once_prefers_top_level(self):
        """Once strategy tries top level before diving into children."""
        engine = RuleEngine.from_dsl('''
            @outer: (f ?x) => (g :x)
            @inner: (+ ?a 0) => :a
        ''')

        # Outer rule should fire first
        expr = E("(f (+ y 0))")
        result = engine(expr, strategy="once")
        assert result == ["g", ["+", "y", 0]]


class TestBottomUpStrategy:
    """Tests for the 'bottomup' strategy."""

    def test_bottomup_simplifies_children_first(self):
        """Bottom-up processes children before parent."""
        engine = RuleEngine.from_dsl('''
            @add-zero: (+ ?x 0) => :x
        ''')

        # Inner (+ y 0) should simplify first
        expr = E("(f (+ y 0))")
        result = engine(expr, strategy="bottomup")
        assert result == ["f", "y"]

    def test_bottomup_then_parent(self):
        """After children simplify, parent can match."""
        engine = RuleEngine.from_dsl('''
            @inner: (a) => x
            @outer: (f x) => done
        ''')

        # First (a) => x, then (f x) => done
        expr = E("(f (a))")
        result = engine(expr, strategy="bottomup")
        assert result == "done"

    def test_bottomup_multiple_passes(self):
        """Bottom-up repeats until fixpoint."""
        engine = RuleEngine.from_dsl('''
            @step1: (a) => (b)
            @step2: (f (b)) => (c)
        ''')

        expr = E("(f (a))")
        result = engine(expr, strategy="bottomup")
        assert result == ["c"]

    def test_bottomup_nested(self):
        """Bottom-up handles deeply nested expressions."""
        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")

        expr = E("(+ (+ (+ x 0) 0) 0)")
        result = engine(expr, strategy="bottomup")
        assert result == "x"


class TestTopDownStrategy:
    """Tests for the 'topdown' strategy."""

    def test_topdown_tries_parent_first(self):
        """Top-down tries to match parent before children."""
        engine = RuleEngine.from_dsl('''
            @outer: (f ?x) => done
            @inner: (a) => x
        ''')

        # Outer rule fires first, inner never gets a chance
        expr = E("(f (a))")
        result = engine(expr, strategy="topdown")
        assert result == "done"

    def test_topdown_recurses_after_no_match(self):
        """If parent doesn't match, top-down processes children."""
        engine = RuleEngine.from_dsl("@inner: (a) => x")

        # No rule matches (g ...), so process children
        expr = E("(g (a))")
        result = engine(expr, strategy="topdown")
        assert result == ["g", "x"]

    def test_topdown_multiple_passes(self):
        """Top-down repeats until fixpoint."""
        engine = RuleEngine.from_dsl('''
            @step1: (f (g ?x)) => (h :x)
            @step2: (h ?x) => :x
        ''')

        expr = E("(f (g (f (g y))))")
        result = engine(expr, strategy="topdown")
        assert result == "y"

    def test_topdown_expansion_then_simplification(self):
        """Top-down is good for expansion-style rules."""
        engine = RuleEngine.from_dsl('''
            @expand: (square ?x) => (* :x :x)
        ''')

        expr = E("(square (square a))")
        result = engine(expr, strategy="topdown")
        # (square (square a)) => (* (square a) (square a)) => (* (* a a) (* a a))
        assert result == ["*", ["*", "a", "a"], ["*", "a", "a"]]


class TestStrategyComparison:
    """Tests comparing different strategies."""

    def test_bottomup_vs_topdown_order(self):
        """Bottom-up and top-down apply rules in different order during each pass."""
        # Use rules where the order matters for the INTERMEDIATE results
        # Both strategies run until fixpoint, so final results may be the same

        # This rule only works if the child is already simplified
        engine = RuleEngine.from_dsl('''
            @needs-simple-child: (f simplified) => success
            @simplify-child: (complex) => simplified
        ''')

        expr = E("(f (complex))")

        # Bottom-up: child simplifies first, then parent can match
        bottomup_result = engine(expr, strategy="bottomup")
        assert bottomup_result == "success"

        # Top-down: parent doesn't match first (child not yet simplified),
        # then child simplifies, then on second pass parent matches
        topdown_result = engine(expr, strategy="topdown")
        assert topdown_result == "success"  # Both reach same fixpoint

    def test_once_shows_order_difference(self):
        """Once strategy clearly shows which rule would fire first."""
        engine = RuleEngine.from_dsl('''
            @outer: (f ?x) => (outer-fired :x)
            @inner: (a) => inner-fired
        ''')

        expr = E("(f (a))")

        # Once with "once" strategy - outer fires first (top-level preference)
        once_result = engine(expr, strategy="once")
        assert once_result == ["outer-fired", ["a"]]

    def test_all_strategies_reach_same_fixpoint(self):
        """For confluent rules, all strategies reach same result."""
        engine = RuleEngine.from_dsl('''
            @add-zero-r: (+ ?x 0) => :x
            @add-zero-l: (+ 0 ?x) => :x
            @mul-one-r: (* ?x 1) => :x
            @mul-one-l: (* 1 ?x) => :x
        ''')

        expr = E("(+ (* 1 x) 0)")

        assert engine(expr, strategy="exhaustive") == "x"
        assert engine(expr, strategy="bottomup") == "x"
        assert engine(expr, strategy="topdown") == "x"


class TestInvalidStrategy:
    """Tests for invalid strategy handling."""

    def test_invalid_strategy_raises(self):
        """Invalid strategy name raises ValueError."""
        engine = RuleEngine.from_dsl("@rule: (a) => (b)")

        with pytest.raises(ValueError) as exc_info:
            engine(E("(a)"), strategy="invalid")

        assert "invalid" in str(exc_info.value).lower()
        assert "exhaustive" in str(exc_info.value)


class TestStrategyWithPrelude:
    """Tests that strategies work with preludes."""

    def test_bottomup_with_constant_folding(self):
        """Bottom-up strategy works with constant folding."""
        engine = (RuleEngine()
            .with_prelude(ARITHMETIC_PRELUDE)
            .load_dsl("@fold: (+ ?a:const ?b:const) => (! + :a :b)"))

        expr = E("(f (+ 1 (+ 2 3)))")
        # Inner (+ 2 3) => 5, then (+ 1 5) => 6
        result = engine(expr, strategy="bottomup")
        assert result == ["f", 6]

    def test_topdown_with_constant_folding(self):
        """Top-down strategy works with constant folding."""
        engine = (RuleEngine()
            .with_prelude(ARITHMETIC_PRELUDE)
            .load_dsl("@fold: (+ ?a:const ?b:const) => (! + :a :b)"))

        # Can't fold at top level, so recurses
        expr = E("(f (+ 1 2))")
        result = engine(expr, strategy="topdown")
        assert result == ["f", 3]
