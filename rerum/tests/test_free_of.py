"""Tests for the free-of? predicate and the ?free binding-order fix."""

import pytest
from rerum import RuleEngine, E, PREDICATE_PRELUDE


class TestFreeOfPredicate:
    """The free-of? fold op: (! free-of? f v) is true iff symbol v does not occur in f."""

    def test_free_of_in_prelude(self):
        """PREDICATE_PRELUDE exposes the free-of? operator."""
        assert "free-of?" in PREDICATE_PRELUDE

    def test_free_of_true_when_absent(self):
        """free-of? returns True when v does not occur in f."""
        engine = (RuleEngine()
            .with_prelude(PREDICATE_PRELUDE)
            .load_dsl("@check: (q ?f ?v) => yes when (! free-of? :f :v)"))
        assert engine(E("(q (sin y) x)")) == "yes"

    def test_free_of_false_when_present(self):
        """free-of? returns False when v occurs in f, so a guarded rule does not fire."""
        engine = (RuleEngine()
            .with_prelude(PREDICATE_PRELUDE)
            .load_dsl("@check: (q ?f ?v) => yes when (! free-of? :f :v)"))
        assert engine(E("(q (sin x) x)")) == ["q", ["sin", "x"], "x"]

    def test_free_of_atom_self(self):
        """A symbol is not free of itself."""
        engine = (RuleEngine()
            .with_prelude(PREDICATE_PRELUDE)
            .load_dsl("@check: (q ?f ?v) => yes when (! free-of? :f :v)"))
        assert engine(E("(q x x)")) == ["q", "x", "x"]

    def test_free_of_constant_always_free(self):
        """A constant contains no variable, so free-of? is always True for it."""
        engine = (RuleEngine()
            .with_prelude(PREDICATE_PRELUDE)
            .load_dsl("@check: (q ?f ?v) => yes when (! free-of? :f :v)"))
        assert engine(E("(q 7 x)")) == "yes"
