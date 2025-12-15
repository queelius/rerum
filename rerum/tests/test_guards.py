"""Tests for conditional guards (when clause)."""

import pytest
from rerum import RuleEngine, E, FULL_PRELUDE, PREDICATE_PRELUDE, ARITHMETIC_PRELUDE


class TestGuardParsing:
    """Tests for parsing rules with when clauses."""

    def test_parse_simple_guard(self):
        """Basic when clause is parsed correctly."""
        engine = RuleEngine.from_dsl("@rule: (f ?x) => :x when (! > :x 0)")

        # Check that condition was parsed
        assert engine._metadata[0].condition is not None
        assert engine._metadata[0].condition == ["!", ">", [":", "x"], 0]

    def test_parse_guard_with_description(self):
        """When clause works with description."""
        engine = RuleEngine.from_dsl(
            '@rule "Only positive": (f ?x) => :x when (! positive? :x)'
        )

        assert engine._metadata[0].name == "rule"
        assert engine._metadata[0].description == "Only positive"
        assert engine._metadata[0].condition is not None

    def test_parse_no_guard(self):
        """Rules without when clause have None condition."""
        engine = RuleEngine.from_dsl("@rule: (f ?x) => (g :x)")

        assert engine._metadata[0].condition is None

    def test_parse_guard_complex_condition(self):
        """Complex conditions are parsed correctly."""
        engine = RuleEngine.from_dsl(
            "@rule: (f ?x ?y) => :x when (! and (! > :x 0) (! < :y 10))"
        )

        cond = engine._metadata[0].condition
        assert cond[0] == "!"
        assert cond[1] == "and"

    def test_guard_repr(self):
        """RuleMetadata repr includes condition."""
        engine = RuleEngine.from_dsl("@abs-pos: (abs ?x) => :x when (! > :x 0)")

        meta = engine._metadata[0]
        assert "when" in repr(meta)
        assert "abs-pos" in repr(meta)


class TestGuardEvaluation:
    """Tests for evaluating conditional guards."""

    def test_guard_blocks_rule(self):
        """Guard condition blocks rule when false."""
        engine = (RuleEngine()
            .with_prelude(FULL_PRELUDE)
            .load_dsl("@positive: (abs ?x) => :x when (! > :x 0)"))

        # Positive number: guard passes
        assert engine(E("(abs 5)")) == 5

        # Negative number: guard fails, rule doesn't apply
        assert engine(E("(abs -5)")) == ["abs", -5]

    def test_guard_with_fallback_rule(self):
        """Multiple rules with guards act as if-then-else."""
        engine = (RuleEngine()
            .with_prelude(FULL_PRELUDE)
            .load_dsl('''
                @abs-pos: (abs ?x) => :x when (! > :x 0)
                @abs-zero: (abs ?x) => 0 when (! = :x 0)
                @abs-neg: (abs ?x) => (! - 0 :x) when (! < :x 0)
            '''))

        assert engine(E("(abs 5)")) == 5
        assert engine(E("(abs 0)")) == 0
        assert engine(E("(abs -5)")) == 5

    def test_guard_with_type_predicate(self):
        """Type predicates work in guards."""
        engine = (RuleEngine()
            .with_prelude(FULL_PRELUDE)
            .load_dsl("@fold: (+ ?a ?b) => (! + :a :b) when (! and (! const? :a) (! const? :b))"))

        # Both constants: folds
        assert engine(E("(+ 2 3)")) == 5

        # Variable present: doesn't fold
        assert engine(E("(+ x 3)")) == ["+", "x", 3]

    def test_guard_in_nested_expression(self):
        """Guards work during recursive simplification."""
        engine = (RuleEngine()
            .with_prelude(FULL_PRELUDE)
            .load_dsl('''
                @add-zero: (+ ?x 0) => :x
                @fold: (+ ?a ?b) => (! + :a :b) when (! and (! const? :a) (! const? :b))
            '''))

        # Inner expression matches fold guard, outer uses add-zero
        result = engine(E("(+ (+ 1 2) 0)"))
        assert result == 3


class TestGuardWithStrategies:
    """Tests for guards with different rewriting strategies."""

    def test_guard_with_once_strategy(self):
        """Guards work with 'once' strategy."""
        engine = (RuleEngine()
            .with_prelude(FULL_PRELUDE)
            .load_dsl('''
                @guarded: (f ?x) => (g :x) when (! > :x 0)
                @unguarded: (f ?x) => (h :x)
            '''))

        # Guard passes: first rule applies
        result = engine(E("(f 5)"), strategy="once")
        assert result == ["g", 5]

        # Guard fails: second rule applies
        result = engine(E("(f -5)"), strategy="once")
        assert result == ["h", -5]

    def test_guard_with_bottomup_strategy(self):
        """Guards work with 'bottomup' strategy."""
        engine = (RuleEngine()
            .with_prelude(FULL_PRELUDE)
            .load_dsl("@fold: (+ ?a ?b) => (! + :a :b) when (! and (! const? :a) (! const? :b))"))

        result = engine(E("(f (+ 1 (+ 2 3)))"), strategy="bottomup")
        assert result == ["f", 6]

    def test_guard_with_topdown_strategy(self):
        """Guards work with 'topdown' strategy."""
        engine = (RuleEngine()
            .with_prelude(FULL_PRELUDE)
            .load_dsl("@fold: (+ ?a ?b) => (! + :a :b) when (! and (! const? :a) (! const? :b))"))

        result = engine(E("(f (+ 1 2))"), strategy="topdown")
        assert result == ["f", 3]


class TestGuardWithApplyOnce:
    """Tests for guards with apply_once method."""

    def test_apply_once_respects_guard(self):
        """apply_once skips rules with failing guards."""
        engine = (RuleEngine()
            .with_prelude(FULL_PRELUDE)
            .load_dsl('''
                @guarded: (f ?x) => (passed :x) when (! > :x 0)
                @fallback: (f ?x) => (fallback :x)
            '''))

        # Guard passes
        result, meta = engine.apply_once(E("(f 5)"))
        assert result == ["passed", 5]
        assert meta.name == "guarded"

        # Guard fails, fallback applies
        result, meta = engine.apply_once(E("(f -5)"))
        assert result == ["fallback", -5]
        assert meta.name == "fallback"


class TestGuardWithRulesMatching:
    """Tests for guards with rules_matching method."""

    def test_rules_matching_respects_guards(self):
        """rules_matching by default only returns rules with passing guards."""
        engine = (RuleEngine()
            .with_prelude(FULL_PRELUDE)
            .load_dsl('''
                @pos: (f ?x) => (pos :x) when (! > :x 0)
                @neg: (f ?x) => (neg :x) when (! < :x 0)
                @any: (f ?x) => (any :x)
            '''))

        # Positive: pos and any match
        matches = engine.rules_matching(E("(f 5)"))
        names = {m[0].name for m in matches}
        assert names == {"pos", "any"}

        # Negative: neg and any match
        matches = engine.rules_matching(E("(f -5)"))
        names = {m[0].name for m in matches}
        assert names == {"neg", "any"}

    def test_rules_matching_can_ignore_guards(self):
        """rules_matching with check_conditions=False ignores guards."""
        engine = (RuleEngine()
            .with_prelude(FULL_PRELUDE)
            .load_dsl('''
                @pos: (f ?x) => (pos :x) when (! > :x 0)
                @neg: (f ?x) => (neg :x) when (! < :x 0)
            '''))

        # With check_conditions=False, both rules match structurally
        matches = engine.rules_matching(E("(f 5)"), check_conditions=False)
        names = {m[0].name for m in matches}
        assert names == {"pos", "neg"}


class TestPredicatePrelude:
    """Tests for the predicate prelude."""

    def test_comparison_operators(self):
        """Comparison operators work correctly."""
        engine = RuleEngine().with_prelude(PREDICATE_PRELUDE)

        assert engine.match("(! > 5 3)", E("(! > 5 3)")) is not None

        # Use a rule to test evaluation
        test_engine = (RuleEngine()
            .with_prelude(PREDICATE_PRELUDE)
            .load_dsl("@test: (check) => (! > 5 3)"))

        assert test_engine(E("(check)")) == True

    def test_type_predicates(self):
        """Type predicates identify types correctly."""
        engine = (RuleEngine()
            .with_prelude(PREDICATE_PRELUDE)
            .load_dsl('''
                @is-const: (type ?x) => const when (! const? :x)
                @is-var: (type ?x) => var when (! var? :x)
                @is-list: (type ?x) => list when (! list? :x)
            '''))

        assert engine(E("(type 5)")) == "const"
        assert engine(E("(type x)")) == "var"
        assert engine(E("(type (+ 1 2))")) == "list"

    def test_logical_operators(self):
        """Logical operators work correctly."""
        engine = (RuleEngine()
            .with_prelude(FULL_PRELUDE)
            .load_dsl('''
                @both-pos: (check ?x ?y) => yes when (! and (! > :x 0) (! > :y 0))
                @default: (check ?x ?y) => no
            '''))

        assert engine(E("(check 1 2)")) == "yes"
        assert engine(E("(check 1 -2)")) == "no"
        assert engine(E("(check -1 2)")) == "no"


class TestFullPrelude:
    """Tests for FULL_PRELUDE combining arithmetic and predicates."""

    def test_full_prelude_has_arithmetic(self):
        """FULL_PRELUDE includes arithmetic operations."""
        engine = (RuleEngine()
            .with_prelude(FULL_PRELUDE)
            .load_dsl("@fold: (+ ?a ?b) => (! + :a :b) when (! and (! const? :a) (! const? :b))"))

        assert engine(E("(+ 2 3)")) == 5

    def test_full_prelude_has_predicates(self):
        """FULL_PRELUDE includes predicates."""
        engine = (RuleEngine()
            .with_prelude(FULL_PRELUDE)
            .load_dsl("@check: (test) => (! > 5 3)"))

        assert engine(E("(test)")) == True


class TestRealWorldGuards:
    """Real-world use cases for conditional guards."""

    def test_sign_function(self):
        """Implement sign function with guards."""
        engine = (RuleEngine()
            .with_prelude(FULL_PRELUDE)
            .load_dsl('''
                @sign-pos: (sign ?x) => 1 when (! > :x 0)
                @sign-zero: (sign ?x) => 0 when (! = :x 0)
                @sign-neg: (sign ?x) => -1 when (! < :x 0)
            '''))

        assert engine(E("(sign 42)")) == 1
        assert engine(E("(sign 0)")) == 0
        assert engine(E("(sign -17)")) == -1

    def test_factorial_base_case(self):
        """Guards for factorial base case."""
        engine = (RuleEngine()
            .with_prelude(FULL_PRELUDE)
            .load_dsl('''
                @fact-zero: (fact ?n) => 1 when (! = :n 0)
                @fact-one: (fact ?n) => 1 when (! = :n 1)
            '''))

        assert engine(E("(fact 0)")) == 1
        assert engine(E("(fact 1)")) == 1
        # Non-base case unchanged (would need recursive rule)
        assert engine(E("(fact 5)")) == ["fact", 5]

    def test_conditional_simplification(self):
        """Simplify only when beneficial."""
        engine = (RuleEngine()
            .with_prelude(FULL_PRELUDE)
            .load_dsl('''
                # Only fold multiplication if result would be a constant
                @mul-const: (* ?a ?b) => (! * :a :b) when (! and (! const? :a) (! const? :b))
                # Don't expand x*x to x^2 if x is already simple
                @square: (* ?x ?x) => (^ :x 2) when (! var? :x)
            '''))

        # Constants fold
        assert engine(E("(* 3 4)")) == 12

        # Variable squared gets rewritten
        assert engine(E("(* x x)")) == ["^", "x", 2]

        # Complex expression doesn't get squared (not a simple var)
        assert engine(E("(* (+ a b) (+ a b))")) == ["*", ["+", "a", "b"], ["+", "a", "b"]]
