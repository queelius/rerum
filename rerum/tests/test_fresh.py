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


class TestFreshSkeleton:
    def test_fresh_resolves_to_base_when_free(self):
        # Skeleton (let (fresh u) (sin u)); but minimal: just the fresh
        # symbol spliced into a compound that does NOT already use it.
        skel = ["g", ["fresh", "u"], "y"]
        out = instantiate(skel, Bindings.empty())
        assert out == ["g", "u", "y"]

    def test_fresh_avoids_occurring_name(self):
        # The whole expression being built already contains `u`, so the
        # fresh form must pick `u1`.
        skel = ["g", "u", ["fresh", "u"]]
        out = instantiate(skel, Bindings.empty())
        assert out == ["g", "u", "u1"]

    def test_fresh_avoids_bound_substituted_name(self):
        # A bound variable :v resolves to `u`, occupying the name, so the
        # fresh form picks `u1`.
        b = Bindings([["v", "u"]])
        skel = ["g", [":", "v"], ["fresh", "u"]]
        out = instantiate(skel, b)
        assert out == ["g", "u", "u1"]

    def test_fresh_is_deterministic(self):
        skel = ["g", "u", ["fresh", "u"]]
        out1 = instantiate(skel, Bindings.empty())
        out2 = instantiate(skel, Bindings.empty())
        assert out1 == out2 == ["g", "u", "u1"]

    def test_two_fresh_in_same_expr_get_distinct_names(self):
        # Both ask for base `u`; the first takes `u`, the second must see
        # it occupied and take `u1`. Determinism requires left-to-right
        # resolution against the partially-built expression.
        skel = ["g", ["fresh", "u"], ["fresh", "u"]]
        out = instantiate(skel, Bindings.empty())
        assert out == ["g", "u", "u1"]


class TestFreshTopLevel:
    def test_fresh_as_whole_skeleton(self):
        # A bare ["fresh", "u"] skeleton resolves to "u" (nothing to avoid).
        out = instantiate(["fresh", "u"], Bindings.empty())
        assert out == "u"

    def test_non_fresh_compound_unaffected(self):
        # A normal two-element compound whose head is not "fresh" is
        # untouched.
        out = instantiate(["g", "u"], Bindings.empty())
        assert out == ["g", "u"]


class TestFreshExports:
    def test_exports(self):
        import rerum
        assert rerum.gensym is gensym
        assert rerum.free_symbols is free_symbols
