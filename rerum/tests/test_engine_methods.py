"""Tests for new RuleEngine methods: with_prelude, match, apply_once, rules_matching."""

import pytest
from rerum import (
    RuleEngine, E, Bindings, NoMatch,
    ARITHMETIC_PRELUDE, MATH_PRELUDE,
)


class TestWithPrelude:
    """Tests for RuleEngine.with_prelude()."""

    def test_with_prelude_fluent(self):
        """with_prelude() returns self for chaining."""
        engine = RuleEngine()
        result = engine.with_prelude(ARITHMETIC_PRELUDE)
        assert result is engine

    def test_with_prelude_enables_folding(self):
        """with_prelude() enables constant folding."""
        engine = (RuleEngine()
            .with_prelude(ARITHMETIC_PRELUDE)
            .load_dsl("@fold: (+ ?a:const ?b:const) => (! + :a :b)"))

        result = engine(E("(+ 1 2)"))
        assert result == 3

    def test_with_prelude_chain_with_load(self):
        """with_prelude() chains naturally with load methods."""
        engine = (RuleEngine()
            .with_prelude(ARITHMETIC_PRELUDE)
            .load_dsl("@add-zero: (+ ?x 0) => :x"))

        assert engine(E("(+ y 0)")) == "y"

    def test_with_prelude_after_load(self):
        """with_prelude() can be called after loading rules."""
        engine = (RuleEngine()
            .load_dsl("@fold: (+ ?a:const ?b:const) => (! + :a :b)")
            .with_prelude(ARITHMETIC_PRELUDE))

        result = engine(E("(+ 1 2)"))
        assert result == 3

    def test_without_prelude_no_folding(self):
        """Without prelude, (! +) doesn't evaluate to a number."""
        # Use a rule that doesn't match its own output to avoid recursion
        engine = RuleEngine().load_dsl(
            "@fold: (fold-add ?a:const ?b:const) => (! + :a :b)")

        # Without prelude, (! + 1 2) becomes ["+", 1, 2] (not evaluated)
        result = engine(E("(fold-add 1 2)"))
        assert result == ["+", 1, 2]


class TestMatch:
    """Tests for RuleEngine.match()."""

    def test_match_success(self):
        """match() returns Bindings on success."""
        engine = RuleEngine()
        bindings = engine.match("(+ ?a ?b)", E("(+ x 1)"))

        assert bindings
        assert isinstance(bindings, Bindings)
        assert bindings["a"] == "x"
        assert bindings["b"] == 1

    def test_match_failure(self):
        """match() returns NoMatch on failure."""
        engine = RuleEngine()
        result = engine.match("(+ ?a ?b)", E("(* x 1)"))

        assert result is NoMatch
        assert not result

    def test_match_string_pattern(self):
        """match() accepts string patterns."""
        engine = RuleEngine()
        bindings = engine.match("(dd ?f ?x:var)", E("(dd (^ x 2) x)"))

        assert bindings
        assert bindings["f"] == ["^", "x", 2]
        assert bindings["x"] == "x"

    def test_match_list_pattern(self):
        """match() accepts list patterns."""
        engine = RuleEngine()
        pattern = ["+", ["?", "a"], ["?", "b"]]
        bindings = engine.match(pattern, ["+", "x", 1])

        assert bindings
        assert bindings["a"] == "x"
        assert bindings["b"] == 1

    def test_match_walrus_pattern(self):
        """match() works with walrus operator."""
        engine = RuleEngine()
        expr = E("(+ x y)")

        if bindings := engine.match("(+ ?a ?b)", expr):
            assert bindings["a"] == "x"
            assert bindings["b"] == "y"
        else:
            pytest.fail("Should have matched")

    def test_match_with_const_constraint(self):
        """match() respects type constraints."""
        engine = RuleEngine()

        # Should match - 5 is a constant
        assert engine.match("(f ?n:const)", E("(f 5)"))

        # Should not match - x is a variable
        assert engine.match("(f ?n:const)", E("(f x)")) is NoMatch


class TestApplyOnce:
    """Tests for RuleEngine.apply_once()."""

    def test_apply_once_success(self):
        """apply_once() applies first matching rule."""
        engine = RuleEngine.from_dsl('''
            @add-zero: (+ ?x 0) => :x
            @mul-one: (* ?x 1) => :x
        ''')

        result, meta = engine.apply_once(E("(+ y 0)"))
        assert result == "y"
        assert meta.name == "add-zero"

    def test_apply_once_no_match(self):
        """apply_once() returns original when no rule matches."""
        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")

        expr = E("(* y 1)")
        result, meta = engine.apply_once(expr)
        assert result == expr
        assert meta is None

    def test_apply_once_first_rule(self):
        """apply_once() applies the first matching rule only."""
        engine = RuleEngine.from_dsl('''
            @rule1: (f ?x) => (g :x)
            @rule2: (f ?x) => (h :x)
        ''')

        result, meta = engine.apply_once(E("(f a)"))
        assert result == ["g", "a"]
        assert meta.name == "rule1"

    def test_apply_once_no_recursion(self):
        """apply_once() doesn't recurse into subexpressions."""
        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")

        # The inner (+ y 0) should NOT be simplified
        expr = E("(f (+ y 0))")
        result, meta = engine.apply_once(expr)
        assert result == expr  # unchanged
        assert meta is None

    def test_apply_once_with_prelude(self):
        """apply_once() uses prelude for skeleton evaluation."""
        engine = (RuleEngine()
            .with_prelude(ARITHMETIC_PRELUDE)
            .load_dsl("@fold: (+ ?a:const ?b:const) => (! + :a :b)"))

        result, meta = engine.apply_once(E("(+ 1 2)"))
        assert result == 3
        assert meta.name == "fold"


class TestRulesMatching:
    """Tests for RuleEngine.rules_matching()."""

    def test_rules_matching_single(self):
        """rules_matching() finds a single matching rule."""
        engine = RuleEngine.from_dsl('''
            @add-zero: (+ ?x 0) => :x
            @mul-one: (* ?x 1) => :x
        ''')

        matches = engine.rules_matching(E("(+ y 0)"))
        assert len(matches) == 1
        assert matches[0][0].name == "add-zero"
        assert matches[0][1]["x"] == "y"

    def test_rules_matching_multiple(self):
        """rules_matching() finds all matching rules."""
        engine = RuleEngine.from_dsl('''
            @rule1: (f ?x) => (g :x)
            @rule2: (f ?x) => (h :x)
            @rule3: (other ?x) => :x
        ''')

        matches = engine.rules_matching(E("(f a)"))
        assert len(matches) == 2
        names = {m[0].name for m in matches}
        assert names == {"rule1", "rule2"}

    def test_rules_matching_none(self):
        """rules_matching() returns empty list when nothing matches."""
        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")

        matches = engine.rules_matching(E("(* y 1)"))
        assert matches == []

    def test_rules_matching_bindings(self):
        """rules_matching() returns proper Bindings objects."""
        engine = RuleEngine.from_dsl("@rule: (+ ?a ?b) => :a")

        matches = engine.rules_matching(E("(+ x y)"))
        assert len(matches) == 1
        meta, bindings = matches[0]
        assert isinstance(bindings, Bindings)
        assert bindings["a"] == "x"
        assert bindings["b"] == "y"

    def test_rules_matching_debug_use(self):
        """rules_matching() is useful for debugging."""
        engine = RuleEngine.from_dsl('''
            @add-zero-r: (+ ?x 0) => :x
            @add-zero-l: (+ 0 ?x) => :x
        ''')

        # This expression should match add-zero-l
        expr = E("(+ 0 y)")
        matches = engine.rules_matching(expr)

        assert len(matches) == 1
        assert matches[0][0].name == "add-zero-l"


class TestIntegration:
    """Integration tests combining multiple new features."""

    def test_fluent_construction(self):
        """Full fluent construction with new methods."""
        engine = (RuleEngine()
            .with_prelude(ARITHMETIC_PRELUDE)
            .load_dsl('''
                @add-zero: (+ ?x 0) => :x
                @mul-one: (* ?x 1) => :x
                @fold-add: (+ ?a:const ?b:const) => (! + :a :b)
            '''))

        # Test match
        assert engine.match("(+ ?x 0)", E("(+ y 0)"))

        # Test apply_once
        result, _ = engine.apply_once(E("(+ 1 2)"))
        assert result == 3

        # Test full simplification
        expr = E("(+ (* x 1) 0)")
        assert engine(expr) == "x"

    def test_expression_builder_with_match(self):
        """E and match work together naturally."""
        engine = RuleEngine()

        x, y = E.vars("x", "y")
        expr = E.op("+", x, E.op("*", 2, y))

        if bindings := engine.match("(+ ?a (* ?n ?b))", expr):
            assert bindings["a"] == "x"
            assert bindings["n"] == 2
            assert bindings["b"] == "y"
        else:
            pytest.fail("Should have matched")
