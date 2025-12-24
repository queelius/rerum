"""Tests for core rewriter functions."""

import pytest
from rerum.rewriter import (
    match, instantiate, rewriter,
    free_in, extend_bindings, lookup,
    ARITHMETIC_PRELUDE, MATH_PRELUDE, FULL_PRELUDE,
    constant, variable, compound,
)


class TestFreeIn:
    """Tests for the free_in function (checks if variable appears in expression)."""

    def test_free_in_constant(self):
        """Variable is not free in a constant."""
        assert free_in("x", 42) == False
        assert free_in("x", 3.14) == False

    def test_free_in_same_variable(self):
        """Variable is free in itself."""
        assert free_in("x", "x") == True

    def test_free_in_different_variable(self):
        """Variable is not free in a different variable."""
        assert free_in("x", "y") == False

    def test_free_in_compound(self):
        """Variable free in compound expression."""
        assert free_in("x", ["+", "x", 1]) == True
        assert free_in("y", ["+", "x", 1]) == False

    def test_free_in_nested(self):
        """Variable free in nested expression."""
        assert free_in("x", ["+", ["*", "x", 2], 1]) == True
        assert free_in("z", ["+", ["*", "x", 2], 1]) == False

    def test_free_in_empty_list(self):
        """Empty list returns False."""
        assert free_in("x", []) == False


class TestExtendBindings:
    """Tests for extend_bindings function."""

    def test_extend_empty(self):
        """Extend empty bindings."""
        result = extend_bindings(["?", "x"], 42, [])
        assert result == [["x", 42]]

    def test_extend_existing(self):
        """Extend with new binding."""
        result = extend_bindings(["?", "y"], 10, [["x", 5]])
        assert result == [["x", 5], ["y", 10]]

    def test_extend_consistent(self):
        """Consistent rebinding returns same bindings."""
        result = extend_bindings(["?", "x"], 5, [["x", 5]])
        assert result == [["x", 5]]

    def test_extend_inconsistent(self):
        """Inconsistent rebinding returns failed."""
        result = extend_bindings(["?", "x"], 10, [["x", 5]])
        assert result == "failed"

    def test_extend_failed_bindings(self):
        """Extending failed bindings returns failed."""
        result = extend_bindings(["?", "x"], 42, "failed")
        assert result == "failed"


class TestLookup:
    """Tests for lookup function."""

    def test_lookup_found(self):
        """Lookup returns bound value."""
        assert lookup("x", [["x", 42], ["y", 10]]) == 42

    def test_lookup_not_found(self):
        """Lookup returns variable name if not bound."""
        assert lookup("z", [["x", 42]]) == "z"

    def test_lookup_failed_bindings(self):
        """Lookup on failed returns variable name."""
        assert lookup("x", "failed") == "x"


class TestHelperPredicates:
    """Tests for helper predicates."""

    def test_constant(self):
        """constant() identifies numbers."""
        assert constant(42) == True
        assert constant(3.14) == True
        assert constant("x") == False
        assert constant(["+", 1, 2]) == False

    def test_variable(self):
        """variable() identifies strings."""
        assert variable("x") == True
        assert variable("foo") == True
        assert variable(42) == False
        assert variable(["+", 1, 2]) == False

    def test_compound(self):
        """compound() identifies lists."""
        assert compound(["+", 1, 2]) == True
        assert compound([]) == True  # empty list is still a list
        assert compound("x") == False
        assert compound(42) == False


class TestMatch:
    """Tests for the match function."""

    def test_match_constant(self):
        """Match constant to constant."""
        result = match(5, 5, [])
        assert result == []

    def test_match_constant_fail(self):
        """Match different constants fails."""
        result = match(5, 10, [])
        assert result == "failed"

    def test_match_variable_pattern(self):
        """Match with variable pattern."""
        result = match(["?", "x"], 42, [])
        assert result == [["x", 42]]

    def test_match_const_pattern(self):
        """Match with const pattern."""
        result = match(["?c", "n"], 42, [])
        assert result == [["n", 42]]

        result = match(["?c", "n"], "x", [])
        assert result == "failed"

    def test_match_var_pattern(self):
        """Match with var pattern."""
        result = match(["?v", "v"], "x", [])
        assert result == [["v", "x"]]

        result = match(["?v", "v"], 42, [])
        assert result == "failed"

    def test_match_compound(self):
        """Match compound expression."""
        result = match(["+", ["?", "x"], 1], ["+", "y", 1], [])
        assert result == [["x", "y"]]

    def test_match_rest_pattern(self):
        """Match rest pattern captures remaining args."""
        result = match(["+", ["?...", "xs"]], ["+", 1, 2, 3], [])
        assert result == [["xs", [1, 2, 3]]]


class TestInstantiate:
    """Tests for the instantiate function."""

    def test_instantiate_constant(self):
        """Instantiate constant returns constant."""
        result = instantiate(42, [["x", 10]])
        assert result == 42

    def test_instantiate_substitution(self):
        """Instantiate substitution returns bound value."""
        result = instantiate([":", "x"], [["x", 42]])
        assert result == 42

    def test_instantiate_compound(self):
        """Instantiate compound with substitutions."""
        result = instantiate(["+", [":", "x"], 1], [["x", 5]])
        assert result == ["+", 5, 1]

    def test_instantiate_splice(self):
        """Instantiate splice expands into parent."""
        result = instantiate(["+", [":...", "xs"]], [["xs", [1, 2, 3]]])
        assert result == ["+", 1, 2, 3]

    def test_instantiate_compute(self):
        """Instantiate compute evaluates with prelude."""
        result = instantiate(["!", "+", 1, 2], [], ARITHMETIC_PRELUDE)
        assert result == 3

    def test_instantiate_nested_compute(self):
        """Nested compute expressions."""
        result = instantiate(
            ["!", "+", ["!", "*", 2, 3], 4],
            [],
            ARITHMETIC_PRELUDE
        )
        assert result == 10


class TestRewriter:
    """Tests for the rewriter factory function."""

    def test_rewriter_basic(self):
        """Basic rewriter applies rules."""
        rules = [
            [["+" , ["?", "x"], 0], [":", "x"]],
        ]
        simplify = rewriter(rules)
        assert simplify(["+", "y", 0]) == "y"

    def test_rewriter_multiple_rules(self):
        """Rewriter with multiple rules."""
        rules = [
            [["+", ["?", "x"], 0], [":", "x"]],
            [["*", ["?", "x"], 1], [":", "x"]],
        ]
        simplify = rewriter(rules)
        assert simplify(["+", "a", 0]) == "a"
        assert simplify(["*", "b", 1]) == "b"

    def test_rewriter_recursive(self):
        """Rewriter applies rules recursively."""
        rules = [
            [["+", ["?", "x"], 0], [":", "x"]],
        ]
        simplify = rewriter(rules)
        # Nested expression should simplify
        assert simplify(["+", ["+", "x", 0], 0]) == "x"

    def test_rewriter_with_fold_funcs(self):
        """Rewriter with fold functions for evaluation."""
        rules = [
            [["+", ["?c", "a"], ["?c", "b"]], ["!", "+", [":", "a"], [":", "b"]]],
        ]
        simplify = rewriter(rules, fold_funcs=ARITHMETIC_PRELUDE)
        assert simplify(["+", 3, 4]) == 7


class TestPreludeOperations:
    """Tests for prelude operations."""

    def test_arithmetic_prelude_operations(self):
        """Arithmetic prelude has basic operations."""
        assert "+" in ARITHMETIC_PRELUDE
        assert "-" in ARITHMETIC_PRELUDE
        assert "*" in ARITHMETIC_PRELUDE
        assert "/" in ARITHMETIC_PRELUDE
        assert "^" in ARITHMETIC_PRELUDE

    def test_math_prelude_operations(self):
        """Math prelude has trig/exp operations."""
        assert "sin" in MATH_PRELUDE
        assert "cos" in MATH_PRELUDE
        assert "exp" in MATH_PRELUDE
        assert "log" in MATH_PRELUDE

    def test_full_prelude_has_predicates(self):
        """Full prelude has predicates."""
        assert "const?" in FULL_PRELUDE
        assert "var?" in FULL_PRELUDE
        assert "and" in FULL_PRELUDE
        assert "or" in FULL_PRELUDE

    def test_arithmetic_operations(self):
        """Test arithmetic operation results."""
        result = instantiate(["!", "+", 10, 5], [], ARITHMETIC_PRELUDE)
        assert result == 15

        result = instantiate(["!", "-", 10, 3], [], ARITHMETIC_PRELUDE)
        assert result == 7

        result = instantiate(["!", "*", 4, 5], [], ARITHMETIC_PRELUDE)
        assert result == 20

        result = instantiate(["!", "/", 20, 4], [], ARITHMETIC_PRELUDE)
        assert result == 5

        result = instantiate(["!", "^", 2, 3], [], ARITHMETIC_PRELUDE)
        assert result == 8

    def test_predicate_operations(self):
        """Test predicate operation results."""
        result = instantiate(["!", "const?", 42], [], FULL_PRELUDE)
        assert result == True

        result = instantiate(["!", "const?", "x"], [], FULL_PRELUDE)
        assert result == False

        result = instantiate(["!", "var?", "x"], [], FULL_PRELUDE)
        assert result == True

        result = instantiate(["!", "and", True, True], [], FULL_PRELUDE)
        assert result == True

        result = instantiate(["!", "and", True, False], [], FULL_PRELUDE)
        assert result == False

    def test_comparison_operations(self):
        """Test comparison operation results."""
        result = instantiate(["!", ">", 5, 3], [], FULL_PRELUDE)
        assert result == True

        result = instantiate(["!", "<", 5, 3], [], FULL_PRELUDE)
        assert result == False

        result = instantiate(["!", "=", 5, 5], [], FULL_PRELUDE)
        assert result == True
