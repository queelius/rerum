"""Tests for bidirectional rules (<=> syntax)."""

import pytest
from rerum.engine import (
    parse_rule_line, load_rules_from_dsl, RuleEngine,
    _convert_skeleton_to_pattern, _convert_pattern_to_skeleton,
    format_sexpr, parse_sexpr,
)


class TestConversionHelpers:
    """Tests for pattern/skeleton conversion functions."""

    def test_skeleton_to_pattern_simple(self):
        """Convert simple substitution to pattern variable."""
        # (: x) -> (? x)
        skeleton = [":", "x"]
        result = _convert_skeleton_to_pattern(skeleton)
        assert result == ["?", "x"]

    def test_skeleton_to_pattern_splice(self):
        """Convert splice to rest pattern."""
        # (:... xs) -> (?... xs)
        skeleton = [":...", "xs"]
        result = _convert_skeleton_to_pattern(skeleton)
        assert result == ["?...", "xs"]

    def test_skeleton_to_pattern_compound(self):
        """Convert compound skeleton to pattern."""
        # (+ :x :y) -> (+ ?x ?y)
        skeleton = ["+", [":", "x"], [":", "y"]]
        result = _convert_skeleton_to_pattern(skeleton)
        assert result == ["+", ["?", "x"], ["?", "y"]]

    def test_skeleton_to_pattern_nested(self):
        """Convert nested skeleton to pattern."""
        # (+ :x (* :y :z)) -> (+ ?x (* ?y ?z))
        skeleton = ["+", [":", "x"], ["*", [":", "y"], [":", "z"]]]
        result = _convert_skeleton_to_pattern(skeleton)
        assert result == ["+", ["?", "x"], ["*", ["?", "y"], ["?", "z"]]]

    def test_skeleton_to_pattern_with_constants(self):
        """Constants are preserved during conversion."""
        # (+ :x 1) -> (+ ?x 1)
        skeleton = ["+", [":", "x"], 1]
        result = _convert_skeleton_to_pattern(skeleton)
        assert result == ["+", ["?", "x"], 1]

    def test_pattern_to_skeleton_simple(self):
        """Convert simple pattern variable to substitution."""
        # (? x) -> (: x)
        pattern = ["?", "x"]
        result = _convert_pattern_to_skeleton(pattern)
        assert result == [":", "x"]

    def test_pattern_to_skeleton_const(self):
        """Type constraint is dropped when converting."""
        # (?c n) -> (: n)
        pattern = ["?c", "n"]
        result = _convert_pattern_to_skeleton(pattern)
        assert result == [":", "n"]

    def test_pattern_to_skeleton_var(self):
        """Var constraint is dropped when converting."""
        # (?v v) -> (: v)
        pattern = ["?v", "v"]
        result = _convert_pattern_to_skeleton(pattern)
        assert result == [":", "v"]

    def test_pattern_to_skeleton_free(self):
        """Free constraint is dropped when converting."""
        # (?free e v) -> (: e)
        pattern = ["?free", "e", "v"]
        result = _convert_pattern_to_skeleton(pattern)
        assert result == [":", "e"]

    def test_pattern_to_skeleton_rest(self):
        """Convert rest pattern to splice."""
        # (?... xs) -> (:... xs)
        pattern = ["?...", "xs"]
        result = _convert_pattern_to_skeleton(pattern)
        assert result == [":...", "xs"]

    def test_pattern_to_skeleton_compound(self):
        """Convert compound pattern to skeleton."""
        # (+ ?x ?y) -> (+ :x :y)
        pattern = ["+", ["?", "x"], ["?", "y"]]
        result = _convert_pattern_to_skeleton(pattern)
        assert result == ["+", [":", "x"], [":", "y"]]


class TestBidirectionalParsing:
    """Tests for parsing bidirectional rules."""

    def test_parse_bidirectional_basic(self):
        """Parse basic bidirectional rule."""
        results = parse_rule_line("@commute: (+ ?x ?y) <=> (+ :y :x)")
        assert len(results) == 2

        fwd_meta, fwd_pattern, fwd_skeleton = results[0]
        rev_meta, rev_pattern, rev_skeleton = results[1]

        # Forward rule
        assert fwd_meta.name == "commute-fwd"
        assert fwd_meta.bidirectional == True
        assert fwd_meta.direction == "fwd"
        assert fwd_pattern == ["+", ["?", "x"], ["?", "y"]]
        assert fwd_skeleton == ["+", [":", "y"], [":", "x"]]

        # Reverse rule
        assert rev_meta.name == "commute-rev"
        assert rev_meta.bidirectional == True
        assert rev_meta.direction == "rev"
        assert rev_pattern == ["+", ["?", "y"], ["?", "x"]]
        assert rev_skeleton == ["+", [":", "x"], [":", "y"]]

    def test_parse_bidirectional_with_priority(self):
        """Parse bidirectional rule with priority."""
        results = parse_rule_line("@assoc[100]: (+ (+ ?x ?y) ?z) <=> (+ :x (+ :y :z))")
        assert len(results) == 2

        fwd_meta, _, _ = results[0]
        rev_meta, _, _ = results[1]

        assert fwd_meta.priority == 100
        assert rev_meta.priority == 100

    def test_parse_bidirectional_with_description(self):
        """Parse bidirectional rule with description."""
        results = parse_rule_line('@assoc "Associativity": (+ (+ ?x ?y) ?z) <=> (+ :x (+ :y :z))')
        assert len(results) == 2

        fwd_meta, _, _ = results[0]
        rev_meta, _, _ = results[1]

        assert fwd_meta.description == "Associativity (forward)"
        assert rev_meta.description == "Associativity (reverse)"

    def test_parse_bidirectional_anonymous(self):
        """Parse anonymous bidirectional rule."""
        results = parse_rule_line("(+ ?x ?y) <=> (+ :y :x)")
        assert len(results) == 2

        fwd_meta, _, _ = results[0]
        rev_meta, _, _ = results[1]

        assert fwd_meta.name is None
        assert rev_meta.name is None
        assert fwd_meta.bidirectional == True
        assert rev_meta.bidirectional == True

    def test_parse_unidirectional_still_works(self):
        """Unidirectional rules still parse correctly."""
        results = parse_rule_line("@add-zero: (+ ?x 0) => :x")
        assert len(results) == 1

        meta, pattern, skeleton = results[0]
        assert meta.name == "add-zero"
        assert meta.bidirectional == False
        assert meta.direction is None

    def test_associativity_reverse(self):
        """Test associativity produces correct reverse rule."""
        results = parse_rule_line("@assoc: (+ (+ ?x ?y) ?z) <=> (+ :x (+ :y :z))")

        fwd_meta, fwd_pattern, fwd_skeleton = results[0]
        rev_meta, rev_pattern, rev_skeleton = results[1]

        # Forward: (+ (+ ?x ?y) ?z) => (+ :x (+ :y :z))
        assert fwd_pattern == ["+", ["+", ["?", "x"], ["?", "y"]], ["?", "z"]]
        assert fwd_skeleton == ["+", [":", "x"], ["+", [":", "y"], [":", "z"]]]

        # Reverse: (+ ?x (+ ?y ?z)) => (+ (+ :x :y) :z)
        assert rev_pattern == ["+", ["?", "x"], ["+", ["?", "y"], ["?", "z"]]]
        assert rev_skeleton == ["+", ["+", [":", "x"], [":", "y"]], [":", "z"]]


class TestBidirectionalEngine:
    """Tests for RuleEngine with bidirectional rules."""

    def test_engine_loads_bidirectional(self):
        """Engine correctly loads bidirectional rules."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)

        assert len(engine) == 2
        assert "commute-fwd" in engine
        assert "commute-rev" in engine

    def test_engine_forward_direction(self):
        """Forward rule applies correctly."""
        engine = RuleEngine.from_dsl("""
            @assoc: (+ (+ ?x ?y) ?z) <=> (+ :x (+ :y :z))
        """)

        # Only enable forward direction
        engine.disable_group("all")  # This won't do anything since no groups

        # Forward: (+ (+ a b) c) -> (+ a (+ b c))
        result = engine.simplify(["+", ["+", "a", "b"], "c"], strategy="once")
        assert result == ["+", "a", ["+", "b", "c"]]

    def test_engine_reverse_direction(self):
        """Reverse rule also exists and works."""
        # Create engine with only the reverse rule by disabling fwd via groups
        engine = RuleEngine.from_dsl("""
            [assoc]
            @assoc: (+ (+ ?x ?y) ?z) <=> (+ :x (+ :y :z))
        """)

        # Verify both rules were loaded
        assert "assoc-fwd" in engine
        assert "assoc-rev" in engine

        # The reverse rule: (+ ?x (+ ?y ?z)) => (+ (+ :x :y) :z)
        # should match (+ a (+ b c)) and produce (+ (+ a b) c)

        # With strategy="once", we get the first matching rule
        # Since fwd is loaded first, it won't match (+ a (+ b c))
        # So rev should fire
        result = engine.simplify(["+", "a", ["+", "b", "c"]], strategy="once")
        assert result == ["+", ["+", "a", "b"], "c"]

    def test_commute_applies_both_ways(self):
        """Commutativity applies in both directions."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)

        # Forward: (+ a b) -> (+ b a)
        result = engine.simplify(["+", "a", "b"], strategy="once")
        assert result == ["+", "b", "a"]

        # The same rule can apply again (since commute-rev matches the output)
        result2 = engine.simplify(result, strategy="once")
        assert result2 == ["+", "a", "b"]

    def test_list_rules_shows_bidirectional(self):
        """list_rules() shows both directions of bidirectional rules."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)

        rules = engine.list_rules()
        assert len(rules) == 2

        # Both rules should be present
        assert any("commute-fwd" in r for r in rules)
        assert any("commute-rev" in r for r in rules)

    def test_to_dsl_exports_both_rules(self):
        """to_dsl() exports both directions as separate rules."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)

        dsl = engine.to_dsl()
        assert "commute-fwd" in dsl
        assert "commute-rev" in dsl

    def test_bidirectional_with_when_clause(self):
        """Bidirectional rules with conditions."""
        engine = RuleEngine.from_dsl("""
            @swap: (pair ?x ?y) <=> (pair :y :x) when (! > :x :y)
        """)

        assert len(engine) == 2

        # Both rules should have the condition
        fwd_rule, fwd_meta = engine["swap-fwd"]
        rev_rule, rev_meta = engine["swap-rev"]

        assert fwd_meta.condition is not None
        assert rev_meta.condition is not None


class TestBidirectionalWithGroups:
    """Tests for bidirectional rules with groups."""

    def test_group_applies_to_both_directions(self):
        """Group tag applies to both forward and reverse rules."""
        engine = RuleEngine.from_dsl("""
            [algebra]
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)

        fwd_rule, fwd_meta = engine["commute-fwd"]
        rev_rule, rev_meta = engine["commute-rev"]

        assert "algebra" in fwd_meta.tags
        assert "algebra" in rev_meta.tags

    def test_disable_group_disables_both(self):
        """Disabling a group disables both directions."""
        engine = RuleEngine.from_dsl("""
            [algebra]
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)

        engine.disable_group("algebra")

        # Neither rule should fire
        result = engine.simplify(["+", "a", "b"], strategy="once")
        assert result == ["+", "a", "b"]  # No change


class TestBidirectionalMetadata:
    """Tests for bidirectional rule metadata."""

    def test_metadata_bidirectional_flag(self):
        """Bidirectional flag is set correctly."""
        results = parse_rule_line("@foo: (a ?x) <=> (b :x)")

        for meta, _, _ in results:
            assert meta.bidirectional == True

    def test_metadata_direction(self):
        """Direction is set correctly."""
        results = parse_rule_line("@foo: (a ?x) <=> (b :x)")

        assert results[0][0].direction == "fwd"
        assert results[1][0].direction == "rev"

    def test_unidirectional_no_bidirectional_flag(self):
        """Unidirectional rules don't have bidirectional flag."""
        results = parse_rule_line("@foo: (a ?x) => :x")

        meta, _, _ = results[0]
        assert meta.bidirectional == False
        assert meta.direction is None
