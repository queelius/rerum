"""Tests for fresh-variable generation (gensym / free_symbols / fresh form)."""

import pytest

from rerum.rewriter import (
    free_symbols, gensym, instantiate, Bindings,
)


class TestFreeSymbols:
    def test_atom_variable(self):
        assert free_symbols("x") == {"x"}

    def test_constant_has_no_symbols(self):
        assert free_symbols(42) == set()
        assert free_symbols(3.14) == set()

    def test_compound_collects_leaves_not_operator_position(self):
        # Operator heads are symbols too, but free_symbols collects ALL
        # symbol leaves including the head, which is the conservative
        # choice for "names already in use".
        syms = free_symbols(["+", "x", ["*", 2, "y"]])
        assert "x" in syms and "y" in syms

    def test_empty_list(self):
        assert free_symbols([]) == set()


class TestGensym:
    def test_base_when_free(self):
        assert gensym("u", set()) == "u"

    def test_skips_occupied(self):
        assert gensym("u", {"u"}) == "u1"
        assert gensym("u", {"u", "u1"}) == "u2"

    def test_deterministic(self):
        avoid = {"u", "u1", "u3"}
        assert gensym("u", avoid) == "u2"
        assert gensym("u", avoid) == "u2"  # same inputs -> same output
