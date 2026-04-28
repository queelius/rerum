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

    def test_to_dsl_emits_bidirectional_form(self):
        """to_dsl() collapses adjacent -fwd/-rev pairs back into a single
        `<=>` rule for roundtrip-safe serialization."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)

        dsl = engine.to_dsl()
        assert "<=>" in dsl
        # Source name, not internal -fwd/-rev split
        assert "@commute:" in dsl
        assert "commute-fwd" not in dsl
        assert "commute-rev" not in dsl

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


class TestBidirectionalPatternValidation:
    """The reverse direction of a `<=>` rule is auto-derived from the
    original skeleton. If that derivation produces a structurally invalid
    pattern (multiple rest patterns, compute forms in pattern position),
    the rule must be rejected at load time, not lazily during rewriting
    (review finding C-3)."""

    def test_two_rest_patterns_at_same_level_rejected(self):
        """`(+ (+ ?x...) ?y...) <=> (+ :x... :y...)` desugars to a rev
        pattern with two `?...` at the same compound level. Reject."""
        with pytest.raises(ValueError, match="rest pattern"):
            parse_rule_line("@flatten: (+ (+ ?x...) ?y...) <=> (+ :x... :y...)")

    def test_compute_form_on_rhs_of_bidirectional_rejected(self):
        """`(+ ?x ?y) <=> (! + :x :y)` would create a rev pattern with
        a `(! ...)` compute form in pattern position. Reject."""
        with pytest.raises(ValueError, match="compute form"):
            parse_rule_line("@bad: (+ ?x ?y) <=> (! + :x :y)")

    def test_well_formed_bidirectional_with_rest_pattern_accepted(self):
        """A `<=>` rule whose rev pattern keeps the rest at the end is fine."""
        # Forward: (f ?x ?xs...) <=> (g :x :xs...)
        # Rev pattern: (g ?x ?xs...) -- single rest at end, OK
        results = parse_rule_line("@foo: (f ?x ?xs...) <=> (g :x :xs...)")
        assert len(results) == 2

    def test_unidirectional_rule_with_compute_in_skeleton_unchanged(self):
        """Compute forms in the skeleton of a `=>` rule are fine."""
        results = parse_rule_line("@fold: (+ ?a:const ?b:const) => (! + :a :b)")
        assert len(results) == 1


class TestBidirectionalRoundtrip:
    """Bidirectional rules must survive serialize/deserialize cycles. Before
    the fix, to_dsl/to_json emitted -fwd/-rev as plain `=>` rules and the
    `bidirectional` flag was silently lost (review finding C-4)."""

    def test_to_dsl_roundtrips_preserves_bidirectional_metadata(self):
        engine1 = RuleEngine.from_dsl("@commute: (+ ?x ?y) <=> (+ :y :x)")
        engine2 = RuleEngine.from_dsl(engine1.to_dsl())
        assert "commute-fwd" in engine2
        assert "commute-rev" in engine2
        _, fwd_meta = engine2["commute-fwd"]
        _, rev_meta = engine2["commute-rev"]
        assert fwd_meta.bidirectional is True
        assert rev_meta.bidirectional is True

    def test_to_dsl_roundtrip_preserves_equivalents(self):
        """The equivalence class of a roundtripped engine must match the
        original. Pre-fix, to_dsl lost the `bidirectional` flag, so
        `equivalents()` returned only the trivial result after roundtrip."""
        engine1 = RuleEngine.from_dsl("@commute: (+ ?x ?y) <=> (+ :y :x)")
        engine2 = RuleEngine.from_dsl(engine1.to_dsl())
        forms1 = set(map(tuple, engine1.enumerate_equivalents(["+", "a", "b"])))
        forms2 = set(map(tuple, engine2.enumerate_equivalents(["+", "a", "b"])))
        assert forms1 == forms2
        assert len(forms1) == 2  # (+ a b) and (+ b a)

    def test_to_dsl_anonymous_bidirectional_roundtrip(self):
        """Anonymous `<=>` rules also roundtrip safely."""
        engine1 = RuleEngine.from_dsl("(+ ?x ?y) <=> (+ :y :x)")
        engine2 = RuleEngine.from_dsl(engine1.to_dsl())
        assert len(engine2) == 2
        # Both should be bidirectional after roundtrip
        for _, meta in engine2:
            assert meta.bidirectional is True

    def test_to_dsl_with_priority_and_description_roundtrips(self):
        engine1 = RuleEngine.from_dsl(
            '@assoc[100] "Associativity": (+ (+ ?x ?y) ?z) <=> (+ :x (+ :y :z))'
        )
        dsl = engine1.to_dsl()
        engine2 = RuleEngine.from_dsl(dsl)
        _, fwd_meta = engine2["assoc-fwd"]
        assert fwd_meta.priority == 100
        # Description should be on both halves after roundtrip
        assert fwd_meta.description == "Associativity (forward)"

    def test_to_json_roundtrip_preserves_bidirectional_flag(self):
        engine1 = RuleEngine.from_dsl("@commute: (+ ?x ?y) <=> (+ :y :x)")
        json_str = engine1.to_json()
        # The JSON should mark the rule as bidirectional, not duplicate it
        import json as _json
        parsed = _json.loads(json_str)
        assert len(parsed["rules"]) == 1
        assert parsed["rules"][0].get("bidirectional") is True

        engine2 = RuleEngine.from_dsl(json_str) if json_str.lstrip().startswith("{") else None
        # Use the proper loader entry for JSON
        from rerum.engine import load_rules_from_json
        loaded = load_rules_from_json(json_str)
        # Loaded should produce two rules (fwd and rev)
        assert len(loaded) == 2
        names = [m.name for m, _ in loaded]
        assert "commute-fwd" in names
        assert "commute-rev" in names

    def test_to_dict_emits_bidirectional_flag(self):
        engine = RuleEngine.from_dsl("@commute: (+ ?x ?y) <=> (+ :y :x)")
        d = engine.to_dict()
        assert len(d["rules"]) == 1
        assert d["rules"][0]["bidirectional"] is True
        assert d["rules"][0]["name"] == "commute"


class TestBidirectionalConstraintPreservation:
    """The reverse direction of a `<=>` rule must carry the same type and
    free-variable constraints as the forward direction. Otherwise the rev
    rule fires on expressions the fwd rule would have rejected, producing
    unsound equivalence classes (see review finding C-2)."""

    def test_const_constraint_preserved_in_rev_pattern(self):
        results = parse_rule_line("@foo: (f ?x:const) <=> (g :x)")
        _, rev_pattern, rev_skeleton = results[1]
        assert rev_pattern == ["g", ["?c", "x"]]
        assert rev_skeleton == ["f", [":", "x"]]

    def test_var_constraint_preserved_in_rev_pattern(self):
        results = parse_rule_line("@foo: (f ?x:var) <=> (g :x)")
        _, rev_pattern, _ = results[1]
        assert rev_pattern == ["g", ["?v", "x"]]

    def test_free_constraint_preserved_in_rev_pattern(self):
        results = parse_rule_line("@foo: (f ?x:free(y)) <=> (g :x)")
        _, rev_pattern, _ = results[1]
        assert rev_pattern == ["g", ["?free", "x", "y"]]

    def test_mixed_constraints_in_multi_var_rule(self):
        """Constraints follow the variable name, not the position."""
        results = parse_rule_line("@foo: (f ?x:const ?y) <=> (g :y :x)")
        _, rev_pattern, _ = results[1]
        # In the rev pattern, x retains its const constraint even though
        # its position changed in the rule body.
        assert rev_pattern == ["g", ["?", "y"], ["?c", "x"]]

    def test_unconstrained_vars_unchanged(self):
        """Unconstrained variables don't gain a spurious constraint."""
        results = parse_rule_line("@foo: (f ?x ?y) <=> (g :y :x)")
        _, rev_pattern, _ = results[1]
        assert rev_pattern == ["g", ["?", "y"], ["?", "x"]]

    def test_rev_rule_const_constraint_actually_filters(self):
        """Soundness: rev rule with const constraint must reject non-const inputs."""
        engine = RuleEngine.from_dsl("@cancel: (f ?x:const) <=> (g :x)")
        # Forward: (f 5) -> (g 5)
        assert engine.simplify(["f", 5], strategy="once") == ["g", 5]
        # Reverse: (g 5) -> (f 5) because 5 is const
        assert engine.simplify(["g", 5], strategy="once") == ["f", 5]
        # Reverse with non-const variable: (g a) must NOT fire
        assert engine.simplify(["g", "a"], strategy="once") == ["g", "a"]

    def test_rev_rule_free_constraint_actually_filters(self):
        """Soundness: rev rule with free constraint must reject inputs containing the var."""
        engine = RuleEngine.from_dsl("@foo: (f ?x:free(y)) <=> (g :x)")
        # Reverse on (g a) -- a is free of y, so rev fires
        assert engine.simplify(["g", "a"], strategy="once") == ["f", "a"]
        # Reverse on (g (h y)) -- contains y, rev must NOT fire
        assert engine.simplify(["g", ["h", "y"]], strategy="once") == ["g", ["h", "y"]]


class TestSimplifyTerminatesOnBidirectionalCycle:
    """Regression: bidirectional rules form a cycle (fwd output matches rev,
    rev output matches fwd). simplify() must terminate, not raise
    RecursionError. The fast path in rewriter.try_rules previously called
    simplify(result) recursively without cycle detection."""

    def test_simplify_default_strategy_with_commute_only(self):
        """Default strategy (exhaustive) on a single <=> rule must not crash.
        The result is in the equivalence class of the input."""
        engine = RuleEngine.from_dsl("@commute: (+ ?x ?y) <=> (+ :y :x)")
        result = engine.simplify(["+", "a", "b"])
        assert result in (["+", "a", "b"], ["+", "b", "a"])

    def test_simplify_default_strategy_with_assoc_only(self):
        """Default strategy on a <=> associativity rule must terminate."""
        engine = RuleEngine.from_dsl(
            "@assoc: (+ (+ ?x ?y) ?z) <=> (+ :x (+ :y :z))"
        )
        result = engine.simplify(["+", ["+", "a", "b"], "c"])
        assert result in (
            ["+", ["+", "a", "b"], "c"],
            ["+", "a", ["+", "b", "c"]],
        )

    def test_simplify_default_with_mixed_uni_and_bidirectional(self):
        """Mixed => and <=> rules: simplify must not loop forever and the
        unidirectional simplification still applies where reachable."""
        engine = RuleEngine.from_dsl(
            """
            @add-zero: (+ ?x 0) => :x
            @commute: (+ ?x ?y) <=> (+ :y :x)
            """
        )
        # add-zero is declared first, so on (+ a 0) it fires immediately to a.
        assert engine.simplify(["+", "a", 0]) == "a"

    def test_simplify_pure_unidirectional_unchanged(self):
        """Cycle-detection fix must not regress unidirectional simplification."""
        engine = RuleEngine.from_dsl(
            """
            @add-zero: (+ ?x 0) => :x
            @mul-one: (* ?x 1) => :x
            """
        )
        assert engine.simplify(["+", ["*", "a", 1], 0]) == "a"

    def test_simplify_with_groups_active_terminates_on_bidirectional(self):
        """The slow-path (_simplify_exhaustive) must also terminate on cycles."""
        engine = RuleEngine.from_dsl(
            """
            [algebra]
            @commute: (+ ?x ?y) <=> (+ :y :x)
            """
        )
        result = engine.simplify(["+", "a", "b"], groups=["algebra"])
        assert result in (["+", "a", "b"], ["+", "b", "a"])

    def test_simplify_with_trace_terminates_on_bidirectional(self):
        """Traced simplification must also terminate, not loop max_steps."""
        engine = RuleEngine.from_dsl("@commute: (+ ?x ?y) <=> (+ :y :x)")
        result, trace = engine.simplify(["+", "a", "b"], trace=True)
        assert result in (["+", "a", "b"], ["+", "b", "a"])
        # Trace should be bounded - not max_steps long
        assert len(trace.steps) < 10
