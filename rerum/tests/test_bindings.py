"""Tests for the Bindings class and NoMatch singleton."""

import pytest
from rerum import Bindings, NoMatch, wrap_bindings, match, parse_sexpr


class TestBindings:
    """Tests for Bindings class."""

    def test_creation(self):
        """Bindings can be created from list of pairs."""
        bindings = Bindings([["x", 1], ["y", 2]])
        assert bindings["x"] == 1
        assert bindings["y"] == 2

    def test_getitem(self):
        """Bindings support bracket access."""
        bindings = Bindings([["x", 1]])
        assert bindings["x"] == 1

    def test_getitem_missing_raises(self):
        """Accessing missing key raises KeyError."""
        bindings = Bindings([["x", 1]])
        with pytest.raises(KeyError):
            _ = bindings["y"]

    def test_get_with_default(self):
        """get() returns default for missing keys."""
        bindings = Bindings([["x", 1]])
        assert bindings.get("x") == 1
        assert bindings.get("y") is None
        assert bindings.get("y", default=42) == 42

    def test_contains(self):
        """'in' operator works for checking bindings."""
        bindings = Bindings([["x", 1], ["y", 2]])
        assert "x" in bindings
        assert "y" in bindings
        assert "z" not in bindings

    def test_len(self):
        """len() returns number of bindings."""
        assert len(Bindings([])) == 0
        assert len(Bindings([["x", 1]])) == 1
        assert len(Bindings([["x", 1], ["y", 2]])) == 2

    def test_bool_always_true(self):
        """Bindings are always truthy (even if empty)."""
        assert bool(Bindings([]))
        assert bool(Bindings([["x", 1]]))

    def test_keys(self):
        """keys() returns bound variable names."""
        bindings = Bindings([["x", 1], ["y", 2]])
        assert set(bindings.keys()) == {"x", "y"}

    def test_values(self):
        """values() returns bound values."""
        bindings = Bindings([["x", 1], ["y", 2]])
        assert set(bindings.values()) == {1, 2}

    def test_items(self):
        """items() returns (name, value) pairs."""
        bindings = Bindings([["x", 1], ["y", 2]])
        assert set(bindings.items()) == {("x", 1), ("y", 2)}

    def test_iter(self):
        """Iterating over Bindings yields keys."""
        bindings = Bindings([["x", 1], ["y", 2]])
        assert set(bindings) == {"x", "y"}

    def test_to_dict(self):
        """to_dict() returns a plain dictionary."""
        bindings = Bindings([["x", 1], ["y", 2]])
        d = bindings.to_dict()
        assert d == {"x": 1, "y": 2}
        assert isinstance(d, dict)

    def test_equality(self):
        """Bindings equality based on contents."""
        b1 = Bindings([["x", 1], ["y", 2]])
        b2 = Bindings([["x", 1], ["y", 2]])
        b3 = Bindings([["x", 1]])
        assert b1 == b2
        assert b1 != b3

    def test_repr(self):
        """Bindings have a sensible repr."""
        bindings = Bindings([["x", 1]])
        assert "Bindings" in repr(bindings)
        assert "x" in repr(bindings)

    def test_complex_values(self):
        """Bindings can hold complex values (lists, etc.)."""
        bindings = Bindings([["f", ["+", "x", 1]], ["v", "x"]])
        assert bindings["f"] == ["+", "x", 1]
        assert bindings["v"] == "x"


class TestNoMatch:
    """Tests for NoMatch singleton."""

    def test_singleton(self):
        """NoMatch is a singleton."""
        assert NoMatch is NoMatch
        from rerum.rewriter import _NoMatch
        assert _NoMatch() is NoMatch

    def test_bool_false(self):
        """NoMatch is falsy."""
        assert not NoMatch
        assert bool(NoMatch) is False

    def test_getitem_raises(self):
        """Accessing any key raises KeyError."""
        with pytest.raises(KeyError):
            _ = NoMatch["x"]

    def test_get_returns_default(self):
        """get() always returns default."""
        assert NoMatch.get("x") is None
        assert NoMatch.get("x", default=42) == 42

    def test_contains_always_false(self):
        """Nothing is 'in' NoMatch."""
        assert "x" not in NoMatch
        assert "anything" not in NoMatch

    def test_len_zero(self):
        """len(NoMatch) is 0."""
        assert len(NoMatch) == 0

    def test_iter_empty(self):
        """Iterating over NoMatch yields nothing."""
        assert list(NoMatch) == []

    def test_repr(self):
        """NoMatch has a sensible repr."""
        assert repr(NoMatch) == "NoMatch"


class TestWrapBindings:
    """Tests for wrap_bindings function."""

    def test_wrap_success(self):
        """wrap_bindings converts successful match to Bindings."""
        result = wrap_bindings([["x", 1], ["y", 2]])
        assert isinstance(result, Bindings)
        assert result["x"] == 1
        assert result["y"] == 2

    def test_wrap_failure(self):
        """wrap_bindings converts 'failed' to NoMatch."""
        result = wrap_bindings("failed")
        assert result is NoMatch

    def test_wrap_empty_success(self):
        """wrap_bindings handles empty bindings (still success)."""
        result = wrap_bindings([])
        assert isinstance(result, Bindings)
        assert len(result) == 0
        assert bool(result)  # empty Bindings is still truthy


class TestBindingsIntegration:
    """Integration tests with actual pattern matching."""

    def test_with_match_function(self):
        """Bindings work with match() function via wrap_bindings."""
        pattern = parse_sexpr("(+ ?a ?b)")
        expr = parse_sexpr("(+ x 1)")

        raw_result = match(pattern, expr, [])
        bindings = wrap_bindings(raw_result)

        assert bindings
        assert bindings["a"] == "x"
        assert bindings["b"] == 1

    def test_with_failed_match(self):
        """NoMatch returned for failed matches."""
        pattern = parse_sexpr("(+ ?a ?b)")
        expr = parse_sexpr("(* x 1)")  # different operator

        raw_result = match(pattern, expr, [])
        result = wrap_bindings(raw_result)

        assert result is NoMatch
        assert not result

    def test_conditional_usage(self):
        """Bindings work naturally in conditionals."""
        pattern = parse_sexpr("(+ ?a ?b)")
        expr = parse_sexpr("(+ x 1)")

        raw_result = match(pattern, expr, [])
        bindings = wrap_bindings(raw_result)

        # This is the intended usage pattern
        if bindings:
            assert bindings["a"] == "x"
        else:
            pytest.fail("Should have matched")

    def test_walrus_operator_pattern(self):
        """Bindings work with walrus operator."""
        pattern = parse_sexpr("(+ ?a ?b)")
        expr = parse_sexpr("(+ x 1)")

        raw_result = match(pattern, expr, [])

        if bindings := wrap_bindings(raw_result):
            assert bindings["a"] == "x"
            assert bindings["b"] == 1
        else:
            pytest.fail("Should have matched")
