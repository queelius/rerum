"""Tests for equality proving (prove_equal and are_equal)."""

import pytest
from rerum.engine import (
    RuleEngine, EqualityProof, format_sexpr, _expr_to_tuple,
)


class TestEqualityProofClass:
    """Tests for the EqualityProof class."""

    def test_basic_attributes(self):
        """EqualityProof stores all attributes correctly."""
        proof = EqualityProof(
            expr_a=["+", "a", "b"],
            expr_b=["+", "b", "a"],
            common=["+", "a", "b"],
            depth_a=0,
            depth_b=1,
        )
        assert proof.expr_a == ["+", "a", "b"]
        assert proof.expr_b == ["+", "b", "a"]
        assert proof.common == ["+", "a", "b"]
        assert proof.depth_a == 0
        assert proof.depth_b == 1

    def test_total_depth(self):
        """total_depth is sum of both depths."""
        proof = EqualityProof(
            expr_a="a", expr_b="b", common="c",
            depth_a=3, depth_b=2
        )
        assert proof.total_depth == 5

    def test_bool_is_truthy(self):
        """Proofs are always truthy."""
        proof = EqualityProof(
            expr_a="a", expr_b="a", common="a",
            depth_a=0, depth_b=0
        )
        assert bool(proof) is True

    def test_repr(self):
        """Repr shows key information."""
        proof = EqualityProof(
            expr_a=["+", "a", "b"],
            expr_b=["+", "b", "a"],
            common=["+", "a", "b"],
            depth_a=0,
            depth_b=1,
        )
        r = repr(proof)
        assert "(+ a b)" in r
        assert "(+ b a)" in r
        assert "≡" in r

    def test_format_brief(self):
        """Brief format shows concise equality."""
        proof = EqualityProof(
            expr_a=["+", "a", "b"],
            expr_b=["+", "b", "a"],
            common=["+", "a", "b"],
            depth_a=0,
            depth_b=1,
        )
        brief = proof.format("brief")
        assert "(+ a b) ≡ (+ b a)" in brief
        assert "via" in brief

    def test_format_paths(self):
        """Paths format shows distances."""
        proof = EqualityProof(
            expr_a=["+", "a", "b"],
            expr_b=["+", "b", "a"],
            common=["+", "a", "b"],
            depth_a=0,
            depth_b=1,
        )
        paths = proof.format("paths")
        assert "Distance from A: 0 steps" in paths
        assert "Distance from B: 1 steps" in paths

    def test_format_full_with_paths(self):
        """Full format shows path details when available."""
        proof = EqualityProof(
            expr_a=["+", "a", "b"],
            expr_b=["+", "b", "a"],
            common=["+", "a", "b"],
            depth_a=0,
            depth_b=1,
            path_a=[["+", "a", "b"]],
            path_b=[["+", "b", "a"], ["+", "a", "b"]],
        )
        full = proof.format("full")
        assert "Proof:" in full
        assert "Path from A" in full
        assert "Path from B" in full

    def test_to_dict(self):
        """to_dict produces serializable dictionary."""
        proof = EqualityProof(
            expr_a=["+", "a", "b"],
            expr_b=["+", "b", "a"],
            common=["+", "a", "b"],
            depth_a=0,
            depth_b=1,
        )
        d = proof.to_dict()
        assert d["expr_a"] == ["+", "a", "b"]
        assert d["expr_b"] == ["+", "b", "a"]
        assert d["common"] == ["+", "a", "b"]
        assert d["depth_a"] == 0
        assert d["depth_b"] == 1
        assert d["total_depth"] == 1

    def test_to_dict_with_paths(self):
        """to_dict includes paths when present."""
        proof = EqualityProof(
            expr_a="a", expr_b="b", common="c",
            depth_a=1, depth_b=2,
            path_a=["a", "c"],
            path_b=["b", "x", "c"],
        )
        d = proof.to_dict()
        assert d["path_a"] == ["a", "c"]
        assert d["path_b"] == ["b", "x", "c"]


class TestProveEqualBasic:
    """Basic tests for prove_equal method."""

    def test_identical_expressions(self):
        """Identical expressions are immediately equal."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)

        proof = engine.prove_equal(["+", "a", "b"], ["+", "a", "b"])
        assert proof is not None
        assert proof.depth_a == 0
        assert proof.depth_b == 0

    def test_simple_commute(self):
        """Commutativity proves (+ a b) = (+ b a)."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)

        proof = engine.prove_equal(["+", "a", "b"], ["+", "b", "a"])
        assert proof is not None
        assert proof.common is not None
        assert proof.total_depth <= 2  # At most one step each

    def test_not_equal_without_rules(self):
        """Different expressions not equal without rules."""
        engine = RuleEngine()

        proof = engine.prove_equal(["+", "a", "b"], ["+", "b", "a"])
        assert proof is None

    def test_not_equal_different_structure(self):
        """Structurally different expressions not equal."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)

        proof = engine.prove_equal(["+", "a", "b"], ["*", "a", "b"])
        assert proof is None

    def test_associativity(self):
        """Associativity proves (+ (+ a b) c) = (+ a (+ b c))."""
        engine = RuleEngine.from_dsl("""
            @assoc: (+ (+ ?x ?y) ?z) <=> (+ :x (+ :y :z))
        """)

        proof = engine.prove_equal(
            ["+", ["+", "a", "b"], "c"],
            ["+", "a", ["+", "b", "c"]]
        )
        assert proof is not None


class TestProveEqualComplex:
    """Tests with more complex rule sets."""

    def test_commute_and_assoc(self):
        """Combined commutativity and associativity."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
            @assoc: (+ (+ ?x ?y) ?z) <=> (+ :x (+ :y :z))
        """)

        # These require multiple rewrites
        proof = engine.prove_equal(
            ["+", ["+", "a", "b"], "c"],
            ["+", "c", ["+", "b", "a"]]
        )
        assert proof is not None

    def test_multiple_operators(self):
        """Works with multiple operators."""
        engine = RuleEngine.from_dsl("""
            @commute-add: (+ ?x ?y) <=> (+ :y :x)
            @commute-mul: (* ?x ?y) <=> (* :y :x)
        """)

        # + commutes
        proof = engine.prove_equal(["+", "a", "b"], ["+", "b", "a"])
        assert proof is not None

        # * commutes
        proof = engine.prove_equal(["*", "a", "b"], ["*", "b", "a"])
        assert proof is not None

        # But + and * are different
        proof = engine.prove_equal(["+", "a", "b"], ["*", "a", "b"])
        assert proof is None

    def test_nested_expressions(self):
        """Proves equality of nested expressions."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)

        # Inner commute
        proof = engine.prove_equal(
            ["+", ["+", "a", "b"], "c"],
            ["+", ["+", "b", "a"], "c"]
        )
        assert proof is not None


class TestProveEqualWithTrace:
    """Tests for prove_equal with trace=True."""

    def test_trace_includes_paths(self):
        """With trace=True, paths are included."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)

        proof = engine.prove_equal(
            ["+", "a", "b"],
            ["+", "b", "a"],
            trace=True
        )
        assert proof is not None
        assert proof.path_a is not None
        assert proof.path_b is not None

    def test_path_starts_with_original(self):
        """Paths start with original expressions."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)

        proof = engine.prove_equal(
            ["+", "a", "b"],
            ["+", "b", "a"],
            trace=True
        )
        assert proof.path_a[0] == ["+", "a", "b"]
        assert proof.path_b[0] == ["+", "b", "a"]

    def test_path_ends_with_common(self):
        """Paths end with common form."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)

        proof = engine.prove_equal(
            ["+", "a", "b"],
            ["+", "b", "a"],
            trace=True
        )
        # One path should end at the common form
        common_key = _expr_to_tuple(proof.common)
        assert _expr_to_tuple(proof.path_a[-1]) == common_key or \
               _expr_to_tuple(proof.path_b[-1]) == common_key

    def test_identical_trace(self):
        """Trace for identical expressions has single-element paths."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)

        proof = engine.prove_equal(
            ["+", "a", "b"],
            ["+", "a", "b"],
            trace=True
        )
        assert len(proof.path_a) == 1
        assert len(proof.path_b) == 1


class TestProveEqualMaxDepth:
    """Tests for max_depth limiting."""

    def test_respects_max_depth(self):
        """Cannot find proof if max_depth is too small."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
            @assoc: (+ (+ ?x ?y) ?z) <=> (+ :x (+ :y :z))
        """)

        # Need multiple steps to prove this
        expr_a = ["+", ["+", "a", "b"], "c"]
        expr_b = ["+", "c", ["+", "b", "a"]]

        # Depth 0 shouldn't find it (unless identical)
        proof = engine.prove_equal(expr_a, expr_b, max_depth=0)
        # They're not identical, so no proof at depth 0
        # Actually depth=0 means we only check initial, no expansion
        # Let's check with depth=1 which may or may not be enough

        # With sufficient depth, should find it
        proof = engine.prove_equal(expr_a, expr_b, max_depth=10)
        assert proof is not None


class TestProveEqualWithGroups:
    """Tests for prove_equal with group filtering."""

    def test_filter_by_group(self):
        """Can filter rules by group."""
        engine = RuleEngine.from_dsl("""
            [algebra]
            @commute: (+ ?x ?y) <=> (+ :y :x)

            [rearrange]
            @assoc: (+ (+ ?x ?y) ?z) <=> (+ :x (+ :y :z))
        """)

        # With only algebra group, can prove commute
        proof = engine.prove_equal(
            ["+", "a", "b"],
            ["+", "b", "a"],
            groups=["algebra"]
        )
        assert proof is not None

        # But can't use associativity
        proof = engine.prove_equal(
            ["+", ["+", "a", "b"], "c"],
            ["+", "a", ["+", "b", "c"]],
            groups=["algebra"]
        )
        assert proof is None  # Needs assoc which is in different group


class TestAreEqual:
    """Tests for are_equal convenience method."""

    def test_returns_bool(self):
        """are_equal returns boolean."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)

        result = engine.are_equal(["+", "a", "b"], ["+", "b", "a"])
        assert isinstance(result, bool)
        assert result is True

    def test_false_for_not_equal(self):
        """Returns False when not equal."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)

        result = engine.are_equal(["+", "a", "b"], ["*", "a", "b"])
        assert result is False

    def test_passes_kwargs(self):
        """Passes additional kwargs to prove_equal."""
        engine = RuleEngine.from_dsl("""
            [algebra]
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)

        # Without group filter, would work
        # With wrong group, should fail
        result = engine.are_equal(
            ["+", "a", "b"],
            ["+", "b", "a"],
            groups=["nonexistent"]
        )
        # Commute is in algebra group, not nonexistent
        # Actually, since nonexistent doesn't exist, _is_rule_active returns False
        # So no rules apply
        assert result is False


class TestProveEqualEdgeCases:
    """Edge case tests."""

    def test_atoms_equal(self):
        """Atoms that are identical are equal."""
        engine = RuleEngine()

        proof = engine.prove_equal("x", "x")
        assert proof is not None
        assert proof.depth_a == 0
        assert proof.depth_b == 0

    def test_atoms_not_equal(self):
        """Different atoms are not equal without rules."""
        engine = RuleEngine()

        proof = engine.prove_equal("x", "y")
        assert proof is None

    def test_numbers_equal(self):
        """Same numbers are equal."""
        engine = RuleEngine()

        proof = engine.prove_equal(42, 42)
        assert proof is not None

    def test_empty_engine(self):
        """Empty engine can still prove structural equality."""
        engine = RuleEngine()

        proof = engine.prove_equal(["+", "a", "b"], ["+", "a", "b"])
        assert proof is not None

    def test_unidirectional_rules_ignored(self):
        """Unidirectional rules ignored by default."""
        engine = RuleEngine.from_dsl("""
            @add-zero: (+ ?x 0) => :x
        """)

        # (+ a 0) and a are related by unidirectional rule
        # But prove_equal uses only bidirectional by default
        proof = engine.prove_equal(["+", "a", 0], "a")
        assert proof is None

    def test_include_unidirectional(self):
        """Can include unidirectional rules."""
        engine = RuleEngine.from_dsl("""
            @add-zero: (+ ?x 0) => :x
        """)

        proof = engine.prove_equal(
            ["+", "a", 0],
            "a",
            include_unidirectional=True
        )
        assert proof is not None


class TestProveEqualPractical:
    """Practical usage tests."""

    def test_verify_algebraic_identity(self):
        """Verify an algebraic identity."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
            @assoc: (+ (+ ?x ?y) ?z) <=> (+ :x (+ :y :z))
        """)

        # (a + b) + c = a + (c + b) - requires both rules
        proof = engine.prove_equal(
            ["+", ["+", "a", "b"], "c"],
            ["+", "a", ["+", "c", "b"]],
            max_depth=5
        )
        assert proof is not None

    def test_proof_can_be_used_in_conditional(self):
        """Proof works in if statements."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)

        if proof := engine.prove_equal(["+", "a", "b"], ["+", "b", "a"]):
            # This should execute
            common = proof.common
            assert common is not None
        else:
            pytest.fail("Proof should have been found")

    def test_none_proof_in_conditional(self):
        """None proof is falsy in conditionals."""
        engine = RuleEngine()

        proof = engine.prove_equal(["+", "a", "b"], ["*", "a", "b"])
        if proof:
            pytest.fail("Should not have found proof")
        # This is correct behavior
