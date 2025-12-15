"""Tests for named rulesets (groups)."""

import pytest
from rerum import RuleEngine, E


class TestGroupParsing:
    """Tests for parsing rules with group syntax."""

    def test_parse_group_declaration(self):
        """Group declaration assigns tags to subsequent rules."""
        engine = RuleEngine.from_dsl('''
            [algebra]
            @add-zero: (+ ?x 0) => :x
            @mul-one: (* ?x 1) => :x
        ''')

        assert "algebra" in engine._metadata[0].tags
        assert "algebra" in engine._metadata[1].tags

    def test_multiple_groups(self):
        """Multiple group declarations work correctly."""
        engine = RuleEngine.from_dsl('''
            [algebra]
            @add-zero: (+ ?x 0) => :x

            [calculus]
            @dd-const: (dd ?c:const ?v) => 0
        ''')

        assert "algebra" in engine._metadata[0].tags
        assert "calculus" in engine._metadata[1].tags
        assert "algebra" not in engine._metadata[1].tags

    def test_rules_without_group(self):
        """Rules before any group have no group tags."""
        engine = RuleEngine.from_dsl('''
            @ungrouped: (f ?x) => :x

            [grouped]
            @grouped-rule: (g ?x) => :x
        ''')

        assert engine._metadata[0].tags == []
        assert "grouped" in engine._metadata[1].tags

    def test_groups_method(self):
        """groups() returns all group names."""
        engine = RuleEngine.from_dsl('''
            [algebra]
            @add-zero: (+ ?x 0) => :x

            [calculus]
            @dd-const: (dd ?c:const ?v) => 0

            [algebra]
            @mul-one: (* ?x 1) => :x
        ''')

        assert engine.groups() == {"algebra", "calculus"}


class TestGroupDisabling:
    """Tests for disabling and enabling groups."""

    def test_disable_group(self):
        """Disabling a group prevents its rules from firing."""
        engine = RuleEngine.from_dsl('''
            [algebra]
            @add-zero: (+ ?x 0) => :x

            [other]
            @f-rule: (f ?x) => (g :x)
        ''')

        # Both work before disabling
        assert engine(E("(+ y 0)")) == "y"
        assert engine(E("(f a)")) == ["g", "a"]

        # Disable algebra
        engine.disable_group("algebra")

        # algebra rule no longer fires
        assert engine(E("(+ y 0)")) == ["+", "y", 0]
        # other group still works
        assert engine(E("(f a)")) == ["g", "a"]

    def test_enable_group(self):
        """Re-enabling a group restores its rules."""
        engine = RuleEngine.from_dsl('''
            [algebra]
            @add-zero: (+ ?x 0) => :x
        ''')

        engine.disable_group("algebra")
        assert engine(E("(+ y 0)")) == ["+", "y", 0]

        engine.enable_group("algebra")
        assert engine(E("(+ y 0)")) == "y"

    def test_disable_multiple_groups(self):
        """Multiple groups can be disabled."""
        engine = RuleEngine.from_dsl('''
            [algebra]
            @add-zero: (+ ?x 0) => :x

            [calculus]
            @dd-const: (dd 1 x) => 0
        ''')

        engine.disable_group("algebra")
        engine.disable_group("calculus")

        assert engine(E("(+ y 0)")) == ["+", "y", 0]
        assert engine(E("(dd 1 x)")) == ["dd", 1, "x"]

    def test_fluent_disable_enable(self):
        """disable_group and enable_group support chaining."""
        engine = (RuleEngine.from_dsl('''
            [a]
            @rule-a: (a) => 1
            [b]
            @rule-b: (b) => 2
        ''')
        .disable_group("a")
        .disable_group("b")
        .enable_group("a"))

        # a is enabled, b is disabled
        assert engine(E("(a)")) == 1
        assert engine(E("(b)")) == ["b"]


class TestExplicitGroups:
    """Tests for the explicit groups parameter."""

    def test_simplify_with_explicit_groups(self):
        """simplify() can specify which groups to use."""
        engine = RuleEngine.from_dsl('''
            [algebra]
            @add-zero: (+ ?x 0) => :x

            [calculus]
            @dd-const: (dd ?c:const ?v) => 0
        ''')

        # Only use algebra group
        result = engine(E("(+ y 0)"), groups=["algebra"])
        assert result == "y"

        # Only use calculus group
        result = engine(E("(dd 5 x)"), groups=["calculus"])
        assert result == 0

        # calculus group doesn't have algebra rules
        result = engine(E("(+ y 0)"), groups=["calculus"])
        assert result == ["+", "y", 0]

    def test_explicit_groups_overrides_disabled(self):
        """Explicit groups parameter overrides disabled_groups."""
        engine = RuleEngine.from_dsl('''
            [algebra]
            @add-zero: (+ ?x 0) => :x
        ''')

        engine.disable_group("algebra")

        # Disabled by default
        assert engine(E("(+ y 0)")) == ["+", "y", 0]

        # But explicit groups parameter overrides
        assert engine(E("(+ y 0)"), groups=["algebra"]) == "y"

    def test_ungrouped_rules_always_active_with_explicit(self):
        """Rules without groups are always active when explicit groups specified."""
        engine = RuleEngine.from_dsl('''
            @always-on: (f ?x) => :x

            [optional]
            @optional-rule: (g ?x) => :x
        ''')

        # Only specify "optional" group - ungrouped rules still fire
        result = engine(E("(f a)"), groups=["optional"])
        assert result == "a"


class TestGroupsWithStrategies:
    """Tests for groups with different strategies."""

    def test_groups_with_once_strategy(self):
        """Groups work with 'once' strategy."""
        engine = RuleEngine.from_dsl('''
            [algebra]
            @add-zero: (+ ?x 0) => :x

            [other]
            @other-rule: (+ ?x 0) => (other :x)
        ''')

        # With algebra group
        result = engine(E("(+ y 0)"), strategy="once", groups=["algebra"])
        assert result == "y"

        # With other group
        result = engine(E("(+ y 0)"), strategy="once", groups=["other"])
        assert result == ["other", "y"]

    def test_groups_with_bottomup_strategy(self):
        """Groups work with 'bottomup' strategy."""
        engine = RuleEngine.from_dsl('''
            [inner]
            @inner: (inner) => simplified

            [outer]
            @outer: (outer ?x) => (done :x)
        ''')

        # Both groups
        result = engine(E("(outer (inner))"), strategy="bottomup")
        assert result == ["done", "simplified"]

        # Only outer group - inner not simplified
        result = engine(E("(outer (inner))"), strategy="bottomup", groups=["outer"])
        assert result == ["done", ["inner"]]

    def test_groups_with_topdown_strategy(self):
        """Groups work with 'topdown' strategy."""
        engine = RuleEngine.from_dsl('''
            [outer]
            @outer: (outer ?x) => (result :x)

            [inner]
            @inner: (a) => b
        ''')

        # Both groups
        result = engine(E("(outer (a))"), strategy="topdown")
        assert result == ["result", "b"]

        # Only outer group
        result = engine(E("(outer (a))"), strategy="topdown", groups=["outer"])
        assert result == ["result", ["a"]]


class TestGroupsWithApplyOnce:
    """Tests for groups with apply_once method."""

    def test_apply_once_with_groups(self):
        """apply_once respects groups parameter."""
        engine = RuleEngine.from_dsl('''
            [algebra]
            @add-zero: (+ ?x 0) => :x

            [other]
            @other-rule: (+ ?x 0) => (other :x)
        ''')

        # With algebra group
        result, meta = engine.apply_once(E("(+ y 0)"), groups=["algebra"])
        assert result == "y"
        assert meta.name == "add-zero"

        # With other group
        result, meta = engine.apply_once(E("(+ y 0)"), groups=["other"])
        assert result == ["other", "y"]
        assert meta.name == "other-rule"

    def test_apply_once_with_disabled_groups(self):
        """apply_once respects disabled groups."""
        engine = RuleEngine.from_dsl('''
            [algebra]
            @add-zero: (+ ?x 0) => :x
        ''')

        engine.disable_group("algebra")

        result, meta = engine.apply_once(E("(+ y 0)"))
        assert result == ["+", "y", 0]
        assert meta is None


class TestGroupsWithRulesMatching:
    """Tests for groups with rules_matching method."""

    def test_rules_matching_with_groups(self):
        """rules_matching respects groups parameter."""
        engine = RuleEngine.from_dsl('''
            [algebra]
            @add-zero: (+ ?x 0) => :x

            [other]
            @other-rule: (+ ?x 0) => (other :x)
        ''')

        # Both groups match
        matches = engine.rules_matching(E("(+ y 0)"))
        names = {m[0].name for m in matches}
        assert names == {"add-zero", "other-rule"}

        # Only algebra group
        matches = engine.rules_matching(E("(+ y 0)"), groups=["algebra"])
        names = {m[0].name for m in matches}
        assert names == {"add-zero"}

    def test_rules_matching_with_disabled_groups(self):
        """rules_matching respects disabled groups."""
        engine = RuleEngine.from_dsl('''
            # Ungrouped rule must come BEFORE any group declaration
            @ungrouped: (+ ?x 0) => other

            [algebra]
            @add-zero: (+ ?x 0) => :x
        ''')

        engine.disable_group("algebra")

        matches = engine.rules_matching(E("(+ y 0)"))
        names = {m[0].name for m in matches}
        assert names == {"ungrouped"}


class TestGroupsWithTrace:
    """Tests for groups with tracing."""

    def test_trace_with_groups(self):
        """Tracing works with groups parameter."""
        engine = RuleEngine.from_dsl('''
            [algebra]
            @add-zero: (+ ?x 0) => :x

            [other]
            @mul-one: (* ?x 1) => :x
        ''')

        # Only algebra group - add-zero matches (+ (* y 1) 0) with x=(* y 1)
        # But mul-one won't fire since it's in 'other' group
        result, trace = engine(E("(+ (* y 1) 0)"), trace=True, groups=["algebra"])
        assert result == ["*", "y", 1]  # Simplifies outer +, inner * stays
        assert len(trace.steps) == 1
        assert trace.steps[0].metadata.name == "add-zero"

    def test_trace_shows_only_active_rules(self):
        """Trace only shows rules that could fire given group settings."""
        engine = RuleEngine.from_dsl('''
            [algebra]
            @add-zero: (+ ?x 0) => :x
        ''')

        # With algebra group enabled
        result, trace = engine(E("(+ y 0)"), trace=True, groups=["algebra"])
        assert result == "y"
        assert len(trace.steps) == 1
        assert trace.steps[0].metadata.name == "add-zero"


class TestRealWorldGroups:
    """Real-world use cases for groups."""

    def test_phased_simplification(self):
        """Groups can implement phased simplification."""
        engine = RuleEngine.from_dsl('''
            [expand]
            @square: (square ?x) => (* :x :x)

            [simplify]
            @add-zero: (+ ?x 0) => :x
            @mul-one: (* ?x 1) => :x
        ''')

        expr = E("(+ (square x) 0)")

        # Phase 1: expand
        result = engine(expr, groups=["expand"])
        assert result == ["+", ["*", "x", "x"], 0]

        # Phase 2: simplify
        result = engine(result, groups=["simplify"])
        assert result == ["*", "x", "x"]

    def test_conditional_rules_by_context(self):
        """Groups can provide context-specific rules."""
        engine = RuleEngine.from_dsl('''
            [positive-domain]
            @sqrt-square: (sqrt (^ ?x 2)) => :x

            [general]
            @sqrt-square: (sqrt (^ ?x 2)) => (abs :x)
        ''')

        # In positive domain, sqrt(x^2) = x
        result = engine(E("(sqrt (^ x 2))"), groups=["positive-domain"])
        assert result == "x"

        # In general, sqrt(x^2) = |x|
        result = engine(E("(sqrt (^ x 2))"), groups=["general"])
        assert result == ["abs", "x"]

    def test_selective_constant_folding(self):
        """Groups can control which operations fold constants."""
        from rerum import FULL_PRELUDE

        engine = (RuleEngine()
            .with_prelude(FULL_PRELUDE)
            .load_dsl('''
                [fold-add]
                @fold-add: (+ ?a ?b) => (! + :a :b) when (! and (! const? :a) (! const? :b))

                [fold-mul]
                @fold-mul: (* ?a ?b) => (! * :a :b) when (! and (! const? :a) (! const? :b))
            '''))

        # Only fold addition
        result = engine(E("(+ (* 2 3) (+ 1 2))"), groups=["fold-add"])
        assert result == ["+", ["*", 2, 3], 3]

        # Only fold multiplication
        result = engine(E("(+ (* 2 3) (+ 1 2))"), groups=["fold-mul"])
        assert result == ["+", 6, ["+", 1, 2]]

