"""Tests for equivalence enumeration."""

import pytest
from rerum.engine import (
    RuleEngine, _expr_to_tuple, format_sexpr,
)


class TestExprToTuple:
    """Tests for the _expr_to_tuple helper."""

    def test_simple_atom(self):
        """Atoms are returned as-is."""
        assert _expr_to_tuple("x") == "x"
        assert _expr_to_tuple(42) == 42
        assert _expr_to_tuple(3.14) == 3.14

    def test_simple_list(self):
        """Simple list becomes tuple."""
        assert _expr_to_tuple(["+", "x", 1]) == ("+", "x", 1)

    def test_nested_list(self):
        """Nested lists become nested tuples."""
        expr = ["+", ["*", "a", "b"], "c"]
        assert _expr_to_tuple(expr) == ("+", ("*", "a", "b"), "c")

    def test_deeply_nested(self):
        """Deeply nested structures are handled."""
        expr = ["+", ["+", ["+", "a", "b"], "c"], "d"]
        expected = ("+", ("+", ("+", "a", "b"), "c"), "d")
        assert _expr_to_tuple(expr) == expected

    def test_empty_list(self):
        """Empty list becomes empty tuple."""
        assert _expr_to_tuple([]) == ()

    def test_hashable(self):
        """Result is hashable (can be used in sets)."""
        result = _expr_to_tuple(["+", "x", 1])
        assert hash(result)  # Should not raise
        s = {result}  # Should work
        assert result in s


class TestAllSingleRewrites:
    """Tests for _all_single_rewrites method."""

    def test_top_level_rewrite(self):
        """Finds rewrites at top level."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)

        rewrites = engine._all_single_rewrites(["+", "a", "b"])
        assert len(rewrites) == 1
        assert ["+", "b", "a"] in rewrites

    def test_nested_rewrite(self):
        """Finds rewrites in subexpressions."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)

        # (+ (+ a b) c) - can rewrite inner (+ a b) to (+ b a)
        expr = ["+", ["+", "a", "b"], "c"]
        rewrites = engine._all_single_rewrites(expr)

        assert ["+", ["+", "b", "a"], "c"] in rewrites  # inner commute
        assert ["+", "c", ["+", "a", "b"]] in rewrites  # outer commute

    def test_bidirectional_only_default(self):
        """By default, only uses bidirectional rules."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
            @add-zero: (+ ?x 0) => :x
        """)

        # (+ a 0) - only commute should apply by default
        rewrites = engine._all_single_rewrites(["+", "a", 0])
        assert ["+", 0, "a"] in rewrites  # commute
        assert "a" not in rewrites  # add-zero is unidirectional

    def test_include_unidirectional(self):
        """Can include unidirectional rules."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
            @add-zero: (+ ?x 0) => :x
        """)

        rewrites = engine._all_single_rewrites(
            ["+", "a", 0], bidirectional_only=False
        )
        assert ["+", 0, "a"] in rewrites  # commute
        assert "a" in rewrites  # add-zero now included

    def test_no_duplicates(self):
        """Doesn't return duplicate rewrites."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)

        # (+ a a) - commute produces same expression
        rewrites = engine._all_single_rewrites(["+", "a", "a"])
        # Should be empty since (+ a a) commuted is still (+ a a)
        assert len(rewrites) == 0

    def test_atom_no_rewrites(self):
        """Atoms have no rewrites."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)

        assert engine._all_single_rewrites("x") == []
        assert engine._all_single_rewrites(42) == []


class TestEquivalentsBasic:
    """Basic tests for equivalents() method."""

    def test_yields_original_first(self):
        """First yielded expression is the original."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)

        equivs = list(engine.equivalents(["+", "a", "b"], max_depth=1))
        assert equivs[0] == ["+", "a", "b"]

    def test_simple_commutativity(self):
        """Finds commuted form."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)

        equivs = list(engine.equivalents(["+", "a", "b"], max_depth=1))
        assert ["+", "a", "b"] in equivs
        assert ["+", "b", "a"] in equivs
        assert len(equivs) == 2

    def test_associativity(self):
        """Explores associative rewrites."""
        engine = RuleEngine.from_dsl("""
            @assoc: (+ (+ ?x ?y) ?z) <=> (+ :x (+ :y :z))
        """)

        # (+ (+ a b) c) should produce (+ a (+ b c))
        equivs = list(engine.equivalents(
            ["+", ["+", "a", "b"], "c"], max_depth=1
        ))
        assert ["+", ["+", "a", "b"], "c"] in equivs
        assert ["+", "a", ["+", "b", "c"]] in equivs

    def test_max_depth_limits(self):
        """max_depth limits exploration."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)

        # With depth 0, only original
        equivs = list(engine.equivalents(["+", "a", "b"], max_depth=0))
        assert len(equivs) == 1
        assert equivs[0] == ["+", "a", "b"]

    def test_max_count_limits(self):
        """max_count limits number of results."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
            @assoc: (+ (+ ?x ?y) ?z) <=> (+ :x (+ :y :z))
        """)

        equivs = list(engine.equivalents(
            ["+", ["+", "a", "b"], "c"],
            max_depth=5,
            max_count=3
        ))
        assert len(equivs) == 3

    def test_no_duplicates(self):
        """Doesn't yield duplicate expressions."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)

        # Applying commute twice gets back to original
        equivs = list(engine.equivalents(["+", "a", "b"], max_depth=3))

        # Convert to tuples for comparison
        seen = set()
        for e in equivs:
            key = _expr_to_tuple(e)
            assert key not in seen, f"Duplicate: {e}"
            seen.add(key)


class TestEquivalentsStrategies:
    """Tests for BFS vs DFS strategies."""

    def test_invalid_strategy_raises(self):
        """Invalid strategy raises ValueError."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)

        with pytest.raises(ValueError, match="Unknown strategy"):
            list(engine.equivalents(["+", "a", "b"], strategy="invalid"))

    def test_bfs_yields_closer_first(self):
        """BFS yields expressions in order of distance."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)

        equivs = list(engine.equivalents(
            ["+", ["+", "a", "b"], "c"],
            max_depth=2,
            strategy="bfs"
        ))

        # First is original (depth 0)
        assert equivs[0] == ["+", ["+", "a", "b"], "c"]

        # Second batch is depth 1 (one rewrite away)
        # All depth-1 expressions before depth-2

    def test_dfs_explores_deep_first(self):
        """DFS explores deeply before breadth."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)

        # Both strategies should find the same equivalents
        bfs_equivs = set(_expr_to_tuple(e) for e in engine.equivalents(
            ["+", "a", "b"], max_depth=3, strategy="bfs"
        ))
        dfs_equivs = set(_expr_to_tuple(e) for e in engine.equivalents(
            ["+", "a", "b"], max_depth=3, strategy="dfs"
        ))

        assert bfs_equivs == dfs_equivs


class TestEquivalentsComplex:
    """Tests with more complex rule sets."""

    def test_commute_and_associativity(self):
        """Combined commutativity and associativity."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
            @assoc: (+ (+ ?x ?y) ?z) <=> (+ :x (+ :y :z))
        """)

        # (+ (+ a b) c) has many equivalent forms
        equivs = list(engine.equivalents(
            ["+", ["+", "a", "b"], "c"],
            max_depth=3,
            max_count=20
        ))

        # Should include various rearrangements
        equiv_set = set(_expr_to_tuple(e) for e in equivs)

        # Some expected forms
        assert ("+", ("+", "a", "b"), "c") in equiv_set
        assert ("+", "a", ("+", "b", "c")) in equiv_set
        assert ("+", "c", ("+", "a", "b")) in equiv_set

    def test_no_bidirectional_rules(self):
        """With no bidirectional rules, only yields original."""
        engine = RuleEngine.from_dsl("""
            @add-zero: (+ ?x 0) => :x
        """)

        equivs = list(engine.equivalents(["+", "a", 0], max_depth=5))
        assert len(equivs) == 1
        assert equivs[0] == ["+", "a", 0]

    def test_include_unidirectional(self):
        """Can include unidirectional rules in exploration."""
        engine = RuleEngine.from_dsl("""
            @add-zero: (+ ?x 0) => :x
        """)

        equivs = list(engine.equivalents(
            ["+", "a", 0],
            max_depth=5,
            include_unidirectional=True
        ))
        assert ["+", "a", 0] in equivs
        assert "a" in equivs


class TestEquivalentsWithGroups:
    """Tests for equivalents with group filtering."""

    def test_filter_by_group(self):
        """Can filter rules by group."""
        engine = RuleEngine.from_dsl("""
            [algebra]
            @commute: (+ ?x ?y) <=> (+ :y :x)

            [rearrange]
            @assoc: (+ (+ ?x ?y) ?z) <=> (+ :x (+ :y :z))
        """)

        # Only use algebra group (commutativity)
        equivs = list(engine.equivalents(
            ["+", ["+", "a", "b"], "c"],
            max_depth=2,
            groups=["algebra"]
        ))

        # Should have commuted forms but not associative rearrangements
        equiv_set = set(_expr_to_tuple(e) for e in equivs)

        # Commuted outer: (+ c (+ a b))
        assert ("+", "c", ("+", "a", "b")) in equiv_set

        # Associative form should NOT be present (wrong group)
        assert ("+", "a", ("+", "b", "c")) not in equiv_set


class TestEnumerateEquivalents:
    """Tests for enumerate_equivalents convenience method."""

    def test_returns_list(self):
        """enumerate_equivalents returns a list."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)

        result = engine.enumerate_equivalents(["+", "a", "b"])
        assert isinstance(result, list)

    def test_default_max_count(self):
        """Has a default max_count of 1000."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)

        # For a simple case, should get all equivalents
        result = engine.enumerate_equivalents(["+", "a", "b"], max_depth=2)
        assert len(result) == 2  # original + commuted


class TestEquivalentsEdgeCases:
    """Edge case tests."""

    def test_empty_engine(self):
        """Engine with no rules yields only original."""
        engine = RuleEngine()

        equivs = list(engine.equivalents(["+", "a", "b"], max_depth=5))
        assert len(equivs) == 1
        assert equivs[0] == ["+", "a", "b"]

    def test_non_matching_rules(self):
        """Rules that don't match yield only original."""
        engine = RuleEngine.from_dsl("""
            @commute-mul: (* ?x ?y) <=> (* :y :x)
        """)

        # Addition expression with multiplication rule
        equivs = list(engine.equivalents(["+", "a", "b"], max_depth=5))
        assert len(equivs) == 1

    def test_atom_expression(self):
        """Atoms yield only themselves."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)

        equivs = list(engine.equivalents("x", max_depth=5))
        assert equivs == ["x"]

        equivs = list(engine.equivalents(42, max_depth=5))
        assert equivs == [42]

    def test_self_symmetric_expression(self):
        """Expression that commutes to itself."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)

        # (+ a a) commuted is still (+ a a)
        equivs = list(engine.equivalents(["+", "a", "a"], max_depth=3))
        assert len(equivs) == 1
        assert equivs[0] == ["+", "a", "a"]

    def test_generator_is_lazy(self):
        """equivalents() is a lazy generator."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
            @assoc: (+ (+ ?x ?y) ?z) <=> (+ :x (+ :y :z))
        """)

        # Get iterator but don't consume it
        gen = engine.equivalents(
            ["+", ["+", "a", "b"], "c"],
            max_depth=100  # Would be huge if not lazy
        )

        # Just get first 3
        first = next(gen)
        second = next(gen)
        third = next(gen)

        assert first == ["+", ["+", "a", "b"], "c"]
        # Don't need to consume the whole generator


class TestEquivalentsPractical:
    """Practical usage tests."""

    def test_find_simplified_form(self):
        """Can find simplified forms among equivalents."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
            @assoc: (+ (+ ?x ?y) ?z) <=> (+ :x (+ :y :z))
        """)

        # Find all equivalent forms
        equivs = engine.enumerate_equivalents(
            ["+", ["+", "a", "b"], "c"],
            max_depth=3
        )

        # Could analyze for "best" form by some metric
        # (e.g., smallest depth, alphabetically first, etc.)
        formatted = [format_sexpr(e) for e in equivs]
        assert "(+ (+ a b) c)" in formatted
        assert "(+ a (+ b c))" in formatted

    def test_prove_equivalence(self):
        """Can prove two expressions are equivalent by finding common form."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)

        expr1 = ["+", "a", "b"]
        expr2 = ["+", "b", "a"]

        equivs1 = set(_expr_to_tuple(e) for e in engine.equivalents(expr1, max_depth=2))
        equivs2 = set(_expr_to_tuple(e) for e in engine.equivalents(expr2, max_depth=2))

        # They should share expressions (proving equivalence)
        assert equivs1 == equivs2
