"""Tests for exact rational arithmetic (Fraction folds and formatting)."""

from fractions import Fraction

import pytest

from rerum.rewriter import (
    coerce_number, safe_div, nary_fold, instantiate, Bindings,
    rewriter, ARITHMETIC_PRELUDE, PREDICATE_PRELUDE,
    match, atom, constant, NUMERIC_TYPES,
)
from rerum.expr import format_sexpr, parse_sexpr
from rerum.engine import RuleEngine


class TestCoerceNumber:
    def test_int_passthrough(self):
        assert coerce_number(5) == 5
        assert isinstance(coerce_number(5), int)

    def test_float_integral_narrows_to_int(self):
        out = coerce_number(4.0)
        assert out == 4
        assert isinstance(out, int)

    def test_float_non_integral_stays_float(self):
        out = coerce_number(1.5)
        assert out == 1.5
        assert isinstance(out, float)

    def test_fraction_whole_collapses_to_int(self):
        out = coerce_number(Fraction(6, 3))
        assert out == 2
        assert isinstance(out, int)

    def test_fraction_non_integral_stays_fraction(self):
        out = coerce_number(Fraction(1, 3))
        assert out == Fraction(1, 3)
        assert isinstance(out, Fraction)

    def test_fraction_is_never_silently_floated(self):
        out = coerce_number(Fraction(1, 3))
        assert not isinstance(out, float)


class TestCoerceNumberBoolSafety:
    """bool/int footgun: Python ``True == 1`` and ``isinstance(True, int)``.

    ``coerce_number`` must guard ``bool`` FIRST and return the SAME bool
    object: it must never narrow ``True`` -> ``1`` or ``False`` -> ``0``.
    """

    def test_true_stays_true_object(self):
        assert coerce_number(True) is True

    def test_false_stays_false_object(self):
        assert coerce_number(False) is False

    def test_bool_not_narrowed_to_int(self):
        # Identity is the strong check, but pin the type too: a bool stays
        # a bool, it does not become a plain int 1/0.
        assert isinstance(coerce_number(True), bool)
        assert isinstance(coerce_number(False), bool)


class TestSafeDivFraction:
    def test_non_integral_int_division_returns_fraction(self):
        handler = safe_div()
        out = handler([1, 3])
        assert out == Fraction(1, 3)
        assert isinstance(out, Fraction)

    def test_integral_int_division_collapses_to_int(self):
        handler = safe_div()
        out = handler([6, 3])
        assert out == 2
        assert isinstance(out, int)

    def test_division_by_zero_returns_none(self):
        handler = safe_div()
        assert handler([1, 0]) is None

    def test_float_division_stays_float(self):
        handler = safe_div()
        out = handler([1.0, 4.0])
        assert out == 0.25
        assert isinstance(out, float)


class TestNaryFoldFraction:
    def test_sum_of_fractions_is_exact(self):
        add = nary_fold(0, lambda a, b: a + b)
        out = add([Fraction(1, 3), Fraction(1, 6)])
        assert out == Fraction(1, 2)
        assert isinstance(out, Fraction)

    def test_product_collapsing_to_int(self):
        mul = nary_fold(1, lambda a, b: a * b)
        out = mul([Fraction(2, 3), 3])
        assert out == 2
        assert isinstance(out, int)

    def test_plain_int_sum_unchanged(self):
        add = nary_fold(0, lambda a, b: a + b)
        out = add([1, 2, 3])
        assert out == 6
        assert isinstance(out, int)


class TestInstantiateRationals:
    def test_compute_keeps_fraction_exact(self):
        # (! / 1 3) must stay Fraction(1, 3), not collapse to float.
        skel = ["!", "/", 1, 3]
        out = instantiate(skel, Bindings.empty(), ARITHMETIC_PRELUDE)
        assert out == Fraction(1, 3)
        assert isinstance(out, Fraction)

    def test_compute_collapses_whole_division_to_int(self):
        skel = ["!", "/", 6, 3]
        out = instantiate(skel, Bindings.empty(), ARITHMETIC_PRELUDE)
        assert out == 2
        assert isinstance(out, int)

    def test_compute_fraction_sum(self):
        skel = ["!", "+", ["!", "/", 1, 3], ["!", "/", 1, 6]]
        out = instantiate(skel, Bindings.empty(), ARITHMETIC_PRELUDE)
        assert out == Fraction(1, 2)
        assert isinstance(out, Fraction)


class TestRewriterFastPathRationals:
    """Pin the second renarrowing site: ``try_constant_fold`` inside the
    ``rewriter()`` fast-path simplifier, reached for constant operands.
    """

    def test_fastpath_division_stays_fraction(self):
        simplify = rewriter([], fold_funcs=ARITHMETIC_PRELUDE)
        out = simplify(["/", 1, 3])
        assert out == Fraction(1, 3)
        assert isinstance(out, Fraction)

    def test_fastpath_division_whole_collapses_to_int(self):
        simplify = rewriter([], fold_funcs=ARITHMETIC_PRELUDE)
        out = simplify(["/", 6, 3])
        assert out == 2
        assert isinstance(out, int)

    def test_fastpath_folds_fraction_operands(self):
        # A Fraction already present as an operand must be admitted by the
        # numeric-constant guard and folded exactly.
        simplify = rewriter([], fold_funcs=ARITHMETIC_PRELUDE)
        out = simplify(["+", Fraction(1, 3), Fraction(1, 6)])
        assert out == Fraction(1, 2)
        assert isinstance(out, Fraction)


class TestFractionIsANumericAtom:
    """A Fraction must be recognized as a numeric atom everywhere a number is.

    Regression for the Phase 3 gap where exact rationals were added as a value
    type but the type predicates still tested only (int, float): the matcher
    crashed (car of a non-list) when a compound pattern met a Fraction atom,
    and the rational predicates / fold gates silently excluded Fractions.
    """

    def test_atom_and_constant_accept_fraction(self):
        assert atom(Fraction(1, 3)) is True
        assert constant(Fraction(1, 3)) is True
        assert Fraction in NUMERIC_TYPES

    def test_compound_pattern_vs_fraction_is_no_match_not_crash(self):
        # The bug: match(["foo","?x"], Fraction(1,3)) raised TypeError via
        # car() instead of returning None. A Fraction is an atom, so a
        # compound pattern simply does not match it.
        assert match(["foo", "?x"], Fraction(1, 3)) is None

    def test_simplify_over_fraction_producing_rule_does_not_crash(self):
        # A compound-pattern rule that computes an exact rational. The engine
        # re-matches all rules against the Fraction result at fixpoint; before
        # the fix that re-match crashed.
        eng = RuleEngine(fold_funcs=ARITHMETIC_PRELUDE)
        eng.load_dsl("@third: (third ?x) => (! / :x 3)")
        result, _trace = eng.simplify(["third", 1], trace=True)
        assert result == Fraction(1, 3)

    def test_predicates_accept_fraction(self):
        assert PREDICATE_PRELUDE["const?"]([Fraction(1, 3)]) is True
        assert PREDICATE_PRELUDE["positive?"]([Fraction(1, 3)]) is True
        assert PREDICATE_PRELUDE["negative?"]([Fraction(-1, 3)]) is True
        assert PREDICATE_PRELUDE["positive?"]([Fraction(-1, 3)]) is False

    def test_const_type_constraint_matches_fraction(self):
        # ?x:const should bind a Fraction (it is a numeric constant). The
        # pattern must be the PARSED form (?x:const -> ["?c", "x"]); match
        # operates on parsed expressions, not raw DSL strings.
        pat = parse_sexpr("(f ?x:const)")
        b = match(pat, ["f", Fraction(1, 3)])
        assert b is not None
        assert b.lookup("x") == Fraction(1, 3)

    def test_fold_gate_folds_fraction_operands(self):
        # (+ (/ 1 3) (/ 1 6)) must fold to the exact 1/2, not be left
        # unfolded because the args are Fractions.
        eng = RuleEngine(fold_funcs=ARITHMETIC_PRELUDE)
        out = eng.simplify(["+", Fraction(1, 3), Fraction(1, 6)])
        assert out == Fraction(1, 2)


class TestRationalExports:
    def test_export(self):
        import rerum
        assert rerum.coerce_number is coerce_number


class TestRationalLiterals:
    """Rational literals: p/q parses to a Fraction ATOM and a Fraction
    formats as p/q, making parse(format(x)) == x exact. (Scheme-style;
    replaces the lossy 0.9 contract where Fraction rendered to the division
    EXPRESSION (/ p q) that parsed back to a different structure.) Forced by
    the first real consumer: load-validated examples of rational-producing
    rules could not express their expected output."""

    def test_parse_rational_literal(self):
        assert parse_sexpr("1/3") == Fraction(1, 3)
        assert isinstance(parse_sexpr("1/3"), Fraction)

    def test_parse_negative_rational(self):
        assert parse_sexpr("-1/3") == Fraction(-1, 3)

    def test_parse_int_valued_rational_narrows(self):
        # 4/2 narrows through coerce_number to the int 2 (engine invariant:
        # no int-valued Fraction atoms).
        out = parse_sexpr("4/2")
        assert out == 2 and isinstance(out, int)

    def test_parse_rational_inside_compound(self):
        assert parse_sexpr("(* x 1/3)") == ["*", "x", Fraction(1, 3)]

    def test_zero_denominator_stays_symbol(self):
        # 1/0 is not a number; it falls through to a plain symbol.
        assert parse_sexpr("1/0") == "1/0"

    def test_non_numeric_slash_tokens_stay_symbols(self):
        for tok in ("x/y", "1/x", "x/2", "/", "1/", "/3", "1/2/3"):
            assert parse_sexpr(tok) == tok

    def test_format_fraction_as_rational_literal(self):
        assert format_sexpr(Fraction(1, 3)) == "1/3"
        assert format_sexpr(Fraction(-1, 3)) == "-1/3"

    def test_roundtrip_is_exact(self):
        f = Fraction(2, 5)
        assert parse_sexpr(format_sexpr(f)) == f
        assert isinstance(parse_sexpr(format_sexpr(f)), Fraction)

    def test_format_inside_compound_literal(self):
        assert format_sexpr(["+", "x", Fraction(1, 2)]) == "(+ x 1/2)"

    def test_sidecar_example_can_express_rational_output(self):
        # THE motivating case: a rational-producing rule's example validates.
        import json
        from rerum.engine import RuleEngine
        eng = RuleEngine().with_prelude(ARITHMETIC_PRELUDE)
        eng.load_dsl(
            "@p: (intp (^ ?v ?n:const) ?v) =>"
            " (* (^ :v (! + :n 1)) (! / 1 (! + :n 1)))",
            validate_examples=False)
        sidecar = json.dumps({"p": {"examples": [
            {"in": "(intp (^ x 2) x)", "out": "(* (^ x 3) 1/3)"}]}})
        eng.load_metadata_json(sidecar, validate_examples=True)  # must not raise


class TestFractionJsonRoundTrip:
    """Persistence parity for rational literals: a Fraction atom in a rule
    must survive to_json -> load_rules_from_json exactly (it previously
    CRASHED to_json, and a 'p/q' string in JSON stayed an inert symbol
    that rendered identically to the real Fraction but computed nothing)."""

    def test_to_json_with_fraction_skeleton_does_not_crash(self):
        from rerum.engine import RuleEngine
        eng = RuleEngine.from_dsl("@third: (third ?x) => (* :x 1/3)")
        js = eng.to_json()
        assert '"1/3"' in js  # encoded as the rational-literal string

    def test_round_trip_restores_the_fraction_atom(self):
        from rerum.engine import RuleEngine, load_rules_from_json
        eng = RuleEngine.from_dsl("@third: (third ?x) => (* :x 1/3)")
        rules = load_rules_from_json(eng.to_json())
        (_meta, (pattern, skeleton)), = rules
        assert skeleton == ["*", [":", "x"], Fraction(1, 3)]
        assert isinstance(skeleton[2], Fraction)

    def test_reloaded_rule_computes(self):
        from rerum.engine import RuleEngine
        eng = RuleEngine.from_dsl("@third: (third ?x) => (* :x 1/3)")
        eng2 = RuleEngine().load_rules_from_json(eng.to_json())
        out = eng2(["third", 6])
        # (* 6 1/3): stays symbolic without a prelude, but the Fraction
        # ATOM must be there (not an inert string).
        assert out == ["*", 6, Fraction(1, 3)]
        assert isinstance(out[2], Fraction)

    def test_rulestore_round_trip(self, tmp_path):
        from rerum.engine import RuleEngine
        from rerum.mcp.persistence import RuleStore
        eng = RuleEngine.from_dsl("@third: (third ?x) => (* :x 1/3)")
        store = RuleStore(root=str(tmp_path))
        store.save_ruleset(eng, "rat")
        eng2 = RuleEngine()
        store.load_ruleset(eng2, "rat")
        out = eng2(["third", 6])
        assert out == ["*", 6, Fraction(1, 3)]
