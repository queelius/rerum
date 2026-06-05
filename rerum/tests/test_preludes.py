"""Tests for the general combine_preludes helper (no domain bundle)."""

import pytest


class TestCombinePreludes:
    """combine_preludes merges fold dicts left-to-right, later-wins, fresh dict."""

    def test_combine_preludes_importable(self):
        """combine_preludes is importable from the package root."""
        from rerum import combine_preludes
        assert callable(combine_preludes)

    def test_combine_two_dicts_merges_keys(self):
        """The result contains every key from each input prelude."""
        from rerum import combine_preludes
        a = {"f": 1, "g": 2}
        b = {"h": 3}
        merged = combine_preludes(a, b)
        assert merged == {"f": 1, "g": 2, "h": 3}

    def test_combine_later_wins_on_conflict(self):
        """When a key appears in more than one prelude, the later prelude wins."""
        from rerum import combine_preludes
        a = {"f": 1, "g": 2}
        b = {"g": 99}
        merged = combine_preludes(a, b)
        assert merged["g"] == 99

    def test_combine_returns_fresh_dict_no_mutation(self):
        """combine_preludes returns a new dict and does not mutate its inputs."""
        from rerum import combine_preludes
        a = {"f": 1}
        b = {"g": 2}
        merged = combine_preludes(a, b)
        assert merged is not a
        assert merged is not b
        assert a == {"f": 1}
        assert b == {"g": 2}
        merged["z"] = 0
        assert "z" not in a and "z" not in b

    def test_combine_empty_returns_empty_dict(self):
        """combine_preludes() with no args returns a fresh empty dict."""
        from rerum import combine_preludes
        assert combine_preludes() == {}

    def test_combine_math_and_predicates(self):
        """combine_preludes(MATH_PRELUDE, PREDICATE_PRELUDE) has sin, const?, and free-of?."""
        from rerum import combine_preludes, MATH_PRELUDE, PREDICATE_PRELUDE
        merged = combine_preludes(MATH_PRELUDE, PREDICATE_PRELUDE)
        assert "sin" in merged
        assert "const?" in merged
        assert "free-of?" in merged

    def test_combine_math_and_predicates_usable_in_engine(self):
        """A guarded rule loads and fires under the combined prelude."""
        from rerum import combine_preludes, MATH_PRELUDE, PREDICATE_PRELUDE
        from rerum import RuleEngine, E
        prelude = combine_preludes(MATH_PRELUDE, PREDICATE_PRELUDE)
        engine = (RuleEngine()
            .with_prelude(prelude)
            .load_dsl("@free: (dd ?f ?v) => 0 when (! free-of? :f :v)"))
        assert engine(E("(dd (sin y) x)")) == 0
