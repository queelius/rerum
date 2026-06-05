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


class TestFreeBindingOrder:
    """The ?x:free(v) tag must be checked against the FINAL resolved bindings."""

    def test_free_left_of_var_does_not_match(self):
        """The documented failing case: (dd ?f:free(v) ?v:var) must NOT match (dd (sin x) x)."""
        from rerum.rewriter import match
        from rerum.engine import parse_sexpr
        pat = parse_sexpr("(dd ?f:free(v) ?v:var)")
        exp = parse_sexpr("(dd (sin x) x)")
        assert match(pat, exp) is None

    def test_free_left_of_var_matches_when_truly_free(self):
        """The same pattern still matches when f is genuinely free of the bound v."""
        from rerum.rewriter import match
        from rerum.engine import parse_sexpr
        pat = parse_sexpr("(dd ?f:free(v) ?v:var)")
        exp = parse_sexpr("(dd (sin y) x)")
        b = match(pat, exp)
        assert b is not None
        assert b.to_dict() == {"f": ["sin", "y"], "v": "x"}

    def test_free_right_of_var_still_works(self):
        """When v is bound to the LEFT of ?free, the legacy ordering still rejects/accepts correctly."""
        from rerum.rewriter import match
        from rerum.engine import parse_sexpr
        pat = parse_sexpr("(g ?v:var ?f:free(v))")
        assert match(pat, parse_sexpr("(g x (sin x))")) is None
        b = match(pat, parse_sexpr("(g x (sin y))"))
        assert b is not None
        assert b.to_dict() == {"v": "x", "f": ["sin", "y"]}

    def test_free_of_compound_excluded_var(self):
        """If the excluded var resolves to a non-symbol, free-of is treated structurally."""
        from rerum.rewriter import match
        from rerum.engine import parse_sexpr
        pat = parse_sexpr("(dd ?f:free(v) ?v:var)")
        b = match(pat, parse_sexpr("(dd 5 x)"))
        assert b is not None
        assert b.to_dict() == {"f": 5, "v": "x"}

    def test_free_excluded_var_name_in_subexpr_still_matches(self):
        """If the matched subexpression contains the excluded var's LITERAL name
        but the var binds elsewhere, the match must still succeed (the free
        check is deferred to the final bindings, not the literal name).

        (dd ?f:free(v) ?v:var) vs (dd (sin v) x): v binds to x, and (sin v) is
        free of x, so this matches with f=(sin v), v=x.
        """
        from rerum.rewriter import match
        from rerum.engine import parse_sexpr
        pat = parse_sexpr("(dd ?f:free(v) ?v:var)")
        b = match(pat, parse_sexpr("(dd (sin v) x)"))
        assert b is not None
        assert b.to_dict() == {"f": ["sin", "v"], "v": "x"}

    def test_free_excluded_var_bound_to_nonsymbol_matches(self):
        """When the excluded var is bound to a non-symbol (e.g. a const), the
        free constraint is vacuously satisfied: nothing can contain a number.

        (op ?v:const ?f:free(v)) vs (op 3 (sin x)): v binds to 3, (sin x) is
        trivially free of 3, so this matches with f=(sin x), v=3.
        """
        from rerum.rewriter import match
        from rerum.engine import parse_sexpr
        pat = parse_sexpr("(op ?v:const ?f:free(v))")
        b = match(pat, parse_sexpr("(op 3 (sin x))"))
        assert b is not None
        assert b.to_dict() == {"v": 3, "f": ["sin", "x"]}
