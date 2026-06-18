"""Behavior tests for typed and untyped rest-patterns.

Covers the rest-pattern DSL forms that previously had ZERO coverage:

  ?xs...          - untyped rest (control): matches any tail
  ?xs:const...    - constant-rest: tail must be all numeric constants
  ?xs:var...      - variable-rest: tail must be all symbols (variables)

A rest-pattern binds the matched tail as a list; the bound list is spliced
back into the skeleton via ``:xs...``. The typed variants add a per-element
type guard: if any element of the tail violates the constraint, the pattern
does NOT match and the rule simply does not fire (the expression is left
unchanged at that position).

The matching logic lives in ``rerum.rewriter.match_compound`` /
``rest_type_constraint``; the DSL parse + render of these forms lives in
``rerum.expr`` (``parse_sexpr`` / ``format_sexpr``). These are behavior
tests over the public engine API, not white-box tests of those internals.
"""

import pytest

from rerum import RuleEngine, parse_sexpr, format_sexpr


def _engine(dsl):
    """Build an engine with a single DSL rule loaded."""
    e = RuleEngine()
    e.load_dsl(dsl)
    return e


# ============================================================
# ?xs:const... - constant-rest
# ============================================================

class TestConstRest:
    """The ?xs:const... typed rest matches an all-numeric tail only."""

    def test_const_rest_matches_all_numeric_tail(self):
        """A const-rest matches a tail of only numbers and rewrites it.

        The matched tail [1, 2, 3] is bound to ``xs`` and spliced into the
        skeleton via ``:xs...``.
        """
        e = _engine("(lst ?xs:const...) => (allnum :xs...)")
        assert e(parse_sexpr("(lst 1 2 3)")) == ["allnum", 1, 2, 3]

    def test_const_rest_does_not_fire_on_nonnumber_in_tail(self):
        """A non-number anywhere in the tail makes the rule NOT fire.

        The expression is returned unchanged because the pattern fails to
        match (the type guard rejects the symbol ``x``).
        """
        e = _engine("(lst ?xs:const...) => (allnum :xs...)")
        expr = parse_sexpr("(lst 1 x 3)")
        assert e(expr) == ["lst", 1, "x", 3]

    def test_const_rest_does_not_fire_when_tail_is_all_symbols(self):
        """An entirely non-numeric tail also fails to match a const-rest."""
        e = _engine("(lst ?xs:const...) => (allnum :xs...)")
        expr = parse_sexpr("(lst a b c)")
        assert e(expr) == ["lst", "a", "b", "c"]

    def test_const_rest_matches_single_number(self):
        """A const-rest matches a one-element numeric tail."""
        e = _engine("(lst ?xs:const...) => (allnum :xs...)")
        assert e(parse_sexpr("(lst 7)")) == ["allnum", 7]

    def test_const_rest_matches_empty_tail(self):
        """PIN: a const-rest matches a ZERO-length tail.

        The per-element type guard never runs on an empty tail, so
        ``(lst)`` matches with ``xs`` bound to ``[]`` and the rule fires,
        producing ``(allnum)``. This is the actual current behavior; a
        typed rest does NOT require at least one element.
        """
        e = _engine("(lst ?xs:const...) => (allnum :xs...)")
        assert e(parse_sexpr("(lst)")) == ["allnum"]

    def test_const_rest_fires_with_metadata_via_apply_once(self):
        """apply_once returns the rewritten expr and the firing rule's metadata."""
        e = _engine("(lst ?xs:const...) => (allnum :xs...)")
        result, meta = e.apply_once(parse_sexpr("(lst 1 2 3)"))
        assert result == ["allnum", 1, 2, 3]
        assert meta is not None

    def test_const_rest_no_fire_returns_no_metadata(self):
        """apply_once reports no metadata when the const-rest rule does not fire."""
        e = _engine("(lst ?xs:const...) => (allnum :xs...)")
        result, meta = e.apply_once(parse_sexpr("(lst 1 x 3)"))
        assert result == ["lst", 1, "x", 3]
        assert meta is None

    def test_const_rest_after_fixed_prefix(self):
        """A const-rest may follow a fixed prefix pattern; the prefix binds
        normally and the typed tail still guards each remaining element."""
        e = _engine("(lst ?head ?xs:const...) => (got :head :xs...)")
        assert e(parse_sexpr("(lst a 1 2 3)")) == ["got", "a", 1, 2, 3]
        # A symbol in the const-tail still blocks the match.
        assert e(parse_sexpr("(lst a 1 x 3)")) == ["lst", "a", 1, "x", 3]


# ============================================================
# ?xs:var... - variable-rest
# ============================================================

class TestVarRest:
    """The ?xs:var... typed rest matches an all-symbol tail only."""

    def test_var_rest_matches_all_symbol_tail(self):
        """A var-rest matches a tail of only symbols and rewrites it."""
        e = _engine("(lst ?xs:var...) => (allvar :xs...)")
        assert e(parse_sexpr("(lst a b c)")) == ["allvar", "a", "b", "c"]

    def test_var_rest_does_not_fire_on_number_in_tail(self):
        """A number anywhere in the tail makes a var-rest NOT fire."""
        e = _engine("(lst ?xs:var...) => (allvar :xs...)")
        expr = parse_sexpr("(lst a 2 c)")
        assert e(expr) == ["lst", "a", 2, "c"]

    def test_var_rest_does_not_fire_when_tail_is_all_numbers(self):
        """An entirely numeric tail fails to match a var-rest."""
        e = _engine("(lst ?xs:var...) => (allvar :xs...)")
        expr = parse_sexpr("(lst 1 2 3)")
        assert e(expr) == ["lst", 1, 2, 3]

    def test_var_rest_matches_single_symbol(self):
        """A var-rest matches a one-element symbol tail."""
        e = _engine("(lst ?xs:var...) => (allvar :xs...)")
        assert e(parse_sexpr("(lst z)")) == ["allvar", "z"]

    def test_var_rest_matches_empty_tail(self):
        """PIN: a var-rest matches a ZERO-length tail (binds xs to [])."""
        e = _engine("(lst ?xs:var...) => (allvar :xs...)")
        assert e(parse_sexpr("(lst)")) == ["allvar"]


# ============================================================
# ?xs... - untyped rest (control)
# ============================================================

class TestUntypedRest:
    """The plain ?xs... untyped rest matches ANY tail; the control."""

    def test_untyped_rest_matches_mixed_tail(self):
        """An untyped rest matches a mixed numeric/symbol tail (no guard)."""
        e = _engine("(lst ?xs...) => (any :xs...)")
        assert e(parse_sexpr("(lst 1 x 3)")) == ["any", 1, "x", 3]

    def test_untyped_rest_matches_all_numeric_tail(self):
        e = _engine("(lst ?xs...) => (any :xs...)")
        assert e(parse_sexpr("(lst 1 2 3)")) == ["any", 1, 2, 3]

    def test_untyped_rest_matches_all_symbol_tail(self):
        e = _engine("(lst ?xs...) => (any :xs...)")
        assert e(parse_sexpr("(lst a b c)")) == ["any", "a", "b", "c"]

    def test_untyped_rest_matches_empty_tail(self):
        """An untyped rest matches a zero-length tail."""
        e = _engine("(lst ?xs...) => (any :xs...)")
        assert e(parse_sexpr("(lst)")) == ["any"]


# ============================================================
# Round-trip: parse_sexpr / format_sexpr preserve rest forms
# ============================================================

class TestRestRoundTrip:
    """parse_sexpr/format_sexpr of patterns containing rest forms preserves
    them, so a rule can survive a text -> structure -> text cycle."""

    @pytest.mark.parametrize("text", [
        "(lst ?xs:const...)",
        "(lst ?xs:var...)",
        "(lst ?xs...)",
    ])
    def test_pattern_round_trip(self, text):
        """Parsing then formatting a rest pattern is the identity on text."""
        assert format_sexpr(parse_sexpr(text)) == text

    @pytest.mark.parametrize("text", [
        "(allnum :xs...)",
        "(got :head :xs...)",
    ])
    def test_skeleton_splice_round_trip(self, text):
        """The skeleton splice form ``:xs...`` round-trips through text."""
        assert format_sexpr(parse_sexpr(text)) == text

    def test_const_rest_round_trip_structure(self):
        """The parsed structure of a typed const-rest is the documented
        ``["?...", name, "const"]`` form, and re-rendering recovers the text."""
        parsed = parse_sexpr("?xs:const...")
        assert parsed == ["?...", "xs", "const"]
        assert format_sexpr(parsed) == "?xs:const..."

    def test_var_rest_round_trip_structure(self):
        parsed = parse_sexpr("?xs:var...")
        assert parsed == ["?...", "xs", "var"]
        assert format_sexpr(parsed) == "?xs:var..."

    def test_untyped_rest_round_trip_structure(self):
        parsed = parse_sexpr("?xs...")
        assert parsed == ["?...", "xs"]
        assert format_sexpr(parsed) == "?xs..."


# ============================================================
# Splicing the bound rest into the skeleton via :xs...
# ============================================================

class TestRestSplice:
    """The bound rest list is spliced (not nested) into the skeleton."""

    def test_const_rest_splices_in_place(self):
        """``:xs...`` splices the bound tail elements into the parent list,
        rather than inserting a single nested list."""
        e = _engine("(lst ?xs:const...) => (wrap (inner :xs...) tail)")
        # The inner list receives the spliced numbers in place.
        assert e(parse_sexpr("(lst 1 2 3)")) == [
            "wrap", ["inner", 1, 2, 3], "tail"
        ]

    def test_splice_preserves_order(self):
        """Splicing preserves the original element order of the tail."""
        e = _engine("(lst ?xs:const...) => (rev :xs...)")
        assert e(parse_sexpr("(lst 3 1 2)")) == ["rev", 3, 1, 2]

    def test_splice_around_other_args(self):
        """A spliced rest can sit between other skeleton arguments."""
        e = _engine("(lst ?xs:var...) => (mid before :xs... after)")
        assert e(parse_sexpr("(lst a b)")) == [
            "mid", "before", "a", "b", "after"
        ]


# ============================================================
# format_sexpr boolean behavior (documented lossy round-trip)
# ============================================================

class TestBoolFormatting:
    """PIN the documented lossy formatting of Python bool atoms.

    The text layer has NO boolean literal, so ``format_sexpr(True)`` renders
    the Python repr ``"True"`` (and ``False`` -> ``"False"``). Re-parsing that
    text yields the *symbol* ``"True"``, not the bool ``True`` -- a lossy
    round-trip. This is a known LOW finding pinned here so any future change
    to bool handling is a deliberate, test-visible decision rather than an
    accident. (Booleans normally enter expressions only as predicate-fold
    results, not as authored literals.)
    """

    def test_bool_true_formats_to_string_true(self):
        assert format_sexpr(True) == "True"

    def test_bool_false_formats_to_string_false(self):
        assert format_sexpr(False) == "False"

    def test_bool_round_trip_is_lossy(self):
        """Re-parsing the formatted bool yields a symbol, not the bool."""
        round_tripped = parse_sexpr(format_sexpr(True))
        assert round_tripped == "True"
        assert round_tripped is not True
