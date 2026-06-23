"""F3: AC-matching proper (matching modulo associativity/commutativity)."""

from rerum import acmatch as am
from rerum.normalize import Theory


class TestMatchBudget:
    def test_spend_decrements_and_flags_truncation(self):
        b = am.MatchBudget(steps=2)
        assert b.spend() is True       # 2 -> 1, still has budget
        assert b.spend() is True       # 1 -> 0, this call consumes the last
        assert b.spend() is False      # exhausted
        assert b.truncated is True

    def test_unbounded_when_none_steps(self):
        b = am.MatchBudget(steps=None)
        for _ in range(1000):
            assert b.spend() is True
        assert b.truncated is False


class TestTheoryHasAC:
    def test_has_ac_true_when_any_ac_op(self):
        assert Theory.from_dict({"+": {"ac": True}}).has_ac() is True

    def test_has_ac_false_when_no_ac_op(self):
        assert Theory.from_dict({"-": {"identity": 0}}).has_ac() is False
        assert Theory.from_dict({}).has_ac() is False


from rerum.rewriter import Bindings


def _matches(pat, exp, theory):
    """All binding dicts ac_match yields, as a list of plain dicts."""
    return [b.to_dict() for b in am.ac_match(pat, exp, theory)]


NO_AC = Theory.from_dict({})


class TestNonACCases:
    def test_literal_match_and_mismatch(self):
        assert _matches("a", "a", NO_AC) == [{}]
        assert _matches("a", "b", NO_AC) == []

    def test_single_variable_binds_whole_expr(self):
        assert _matches(["?", "x"], ["f", "a"], NO_AC) == [{"x": ["f", "a"]}]

    def test_typed_variable_constraints(self):
        # Constants are NUMBERS in rerum; symbols (strings) are variables.
        assert _matches(["?c", "n"], 3, NO_AC) == [{"n": 3}]
        assert _matches(["?c", "n"], "x", NO_AC) == []       # x is not constant
        assert _matches(["?v", "s"], "x", NO_AC) == [{"s": "x"}]
        assert _matches(["?v", "s"], 3, NO_AC) == []         # 3 is not a variable

    def test_non_ac_compound_positional(self):
        # (f ?x ?y) against (f a b): exactly one match, positional.
        assert _matches(["f", ["?", "x"], ["?", "y"]], ["f", "a", "b"], NO_AC) == \
            [{"x": "a", "y": "b"}]

    def test_non_ac_head_mismatch(self):
        assert _matches(["f", ["?", "x"]], ["g", "a"], NO_AC) == []

    def test_non_linear_consistency(self):
        # (f ?x ?x) matches (f a a) but not (f a b).
        assert _matches(["f", ["?", "x"], ["?", "x"]], ["f", "a", "a"], NO_AC) == \
            [{"x": "a"}]
        assert _matches(["f", ["?", "x"], ["?", "x"]], ["f", "a", "b"], NO_AC) == []

    def test_agrees_with_syntactic_match_on_non_ac(self):
        from rerum.rewriter import match
        pat = ["f", ["?", "x"], ["g", ["?", "y"]]]
        exp = ["f", "a", ["g", "b"]]
        syntactic = match(pat, exp)
        ac = list(am.ac_match(pat, exp, NO_AC))
        assert len(ac) == 1 and ac[0].to_dict() == syntactic.to_dict()


AC_PLUS = Theory.from_dict({"+": {"ac": True, "identity": 0}})


def _freeze(v):
    return tuple(_freeze(x) for x in v) if isinstance(v, list) else v


def _dictset(pat, exp, theory, budget=None):
    """Yielded bindings as a list of frozenset(items) for order-insensitive compare."""
    out = []
    for b in am.ac_match(pat, exp, theory, budget=budget):
        out.append(frozenset((k, _freeze(v)) for k, v in b.to_dict().items()))
    return out


class TestACMultisetExhaust:
    def test_two_vars_two_elements_two_matches(self):
        # (+ ?x ?y) against (+ a b): {x=a,y=b} and {x=b,y=a}.
        got = _dictset(["+", ["?", "x"], ["?", "y"]], ["+", "a", "b"], AC_PLUS)
        assert len(got) == 2
        assert frozenset({("x", "a"), ("y", "b")}) in got
        assert frozenset({("x", "b"), ("y", "a")}) in got

    def test_three_vars_three_elements_six_matches(self):
        got = _dictset(
            ["+", ["?", "x"], ["?", "y"], ["?", "z"]],
            ["+", "a", "b", "c"], AC_PLUS)
        assert len(got) == 6

    def test_exhaust_required_without_rest(self):
        # (+ ?x ?y) against (+ a b c): no rest -> must exhaust -> no match.
        assert _dictset(["+", ["?", "x"], ["?", "y"]], ["+", "a", "b", "c"], AC_PLUS) == []

    def test_literal_element_in_ac_node(self):
        # (+ 2 ?x) against (+ 2 a): 2 matches the literal, ?x=a.
        got = _dictset(["+", 2, ["?", "x"]], ["+", 2, "a"], AC_PLUS)
        assert got == [frozenset({("x", "a")})]
        # (+ 2 ?x) against (+ 3 a): no 2 present -> no match.
        assert _dictset(["+", 2, ["?", "x"]], ["+", 3, "a"], AC_PLUS) == []

    def test_flatten_before_match(self):
        # Nested sum is seen flat: (+ ?x ?y) against (+ a (+ b)) -> a, b.
        got = _dictset(["+", ["?", "x"], ["?", "y"]], ["+", "a", ["+", "b"]], AC_PLUS)
        assert len(got) == 2

    def test_non_linear_under_ac(self):
        # (+ ?x ?x) against (+ a a) matches (x=a); against (+ a b) does not.
        assert _dictset(["+", ["?", "x"], ["?", "x"]], ["+", "a", "a"], AC_PLUS) == \
            [frozenset({("x", "a")})]
        assert _dictset(["+", ["?", "x"], ["?", "x"]], ["+", "a", "b"], AC_PLUS) == []

    def test_budget_truncates_but_yields_are_valid(self):
        # A tiny budget over (+ ?x ?y ?z) vs (+ a b c): the 6-assignment
        # enumeration is cut short, but each binding yielded is a real match
        # (soundness under truncation). With steps=3 exactly one assignment
        # completes before the budget runs out.
        budget = am.MatchBudget(steps=3)
        pat = ["+", ["?", "x"], ["?", "y"], ["?", "z"]]
        exp = ["+", "a", "b", "c"]
        got = list(am.ac_match(pat, exp, AC_PLUS, budget=budget))
        assert budget.truncated is True
        assert 0 < len(got) < 6        # some, but not all 6, assignments
        for b in got:
            vals = [b["x"], b["y"], b["z"]]
            assert len(set(vals)) == 3
            assert set(vals) == {"a", "b", "c"}


import itertools


class TestACRest:
    def test_rest_captures_leftover_list(self):
        # (+ ?x ?rest...) against (+ a b c): x picks one, rest is the other two.
        got = []
        for b in am.ac_match(["+", ["?", "x"], ["?...", "rest"]],
                             ["+", "a", "b", "c"], AC_PLUS):
            got.append((b["x"], b["rest"]))
        # Three choices of x; rest is the remaining two in canonical order.
        xs = sorted(g[0] for g in got)
        assert xs == ["a", "b", "c"]
        for x, rest in got:
            assert isinstance(rest, list)
            assert sorted([x] + rest) == ["a", "b", "c"]

    def test_rest_empty_when_explicit_exhausts(self):
        # (+ ?x ?y ?rest...) against (+ a b): rest = [].
        got = [b["rest"] for b in am.ac_match(
            ["+", ["?", "x"], ["?", "y"], ["?...", "rest"]],
            ["+", "a", "b"], AC_PLUS)]
        assert got and all(r == [] for r in got)

    def test_rest_singleton(self):
        got = [b["rest"] for b in am.ac_match(
            ["+", ["?", "x"], ["?...", "rest"]], ["+", "a", "b"], AC_PLUS)]
        assert all(len(r) == 1 for r in got)

    def test_cancellation_idiom(self):
        # (+ ?x (- ?x) ?rest...) against (+ a (- a) b): x=a, rest=[b].
        pat = ["+", ["?", "x"], ["-", ["?", "x"]], ["?...", "rest"]]
        got = list(am.ac_match(pat, ["+", "a", ["-", "a"], "b"], AC_PLUS))
        assert any(b["x"] == "a" and b["rest"] == ["b"] for b in got)

    def test_cancellation_no_pair_no_match(self):
        pat = ["+", ["?", "x"], ["-", ["?", "x"]], ["?...", "rest"]]
        assert list(am.ac_match(pat, ["+", "a", "b", "c"], AC_PLUS)) == []


class TestACFreeConstraints:
    # Regression for the ?free binding-order soundness bug: ac_match validates
    # ?free against the COMPLETE binding (a top-level post-pass), not only
    # inline. Without that, a ?free node LEFT of its excluded variable passes
    # vacuously and fires spuriously under an AC theory.

    def test_free_left_of_excluded_var_rejects_spurious(self):
        # (+ ?z:free(w) ?w) vs (+ x x): every split has z containing w -> none.
        pat = ["+", ["?free", "z", "w"], ["?", "w"]]
        assert _dictset(pat, ["+", "x", "x"], AC_PLUS) == []

    def test_free_right_of_excluded_var_rejects_spurious(self):
        # Excluded var first; was already correct, pinned for symmetry.
        pat = ["+", ["?", "w"], ["?free", "z", "w"]]
        assert _dictset(pat, ["+", "x", "x"], AC_PLUS) == []

    def test_free_satisfied_when_excluded_absent(self):
        # (+ ?z:free(w) ?w) vs (+ a b): a free of b and b free of a -> both.
        pat = ["+", ["?free", "z", "w"], ["?", "w"]]
        assert len(_dictset(pat, ["+", "a", "b"], AC_PLUS)) == 2

    def test_free_excluded_inside_compound_element(self):
        # (+ ?z:free(w) ?w) vs (+ (g y) y): z=(g y),w=y rejected ((g y) holds y);
        # z=y,w=(g y) kept (y does not hold (g y)).
        pat = ["+", ["?free", "z", "w"], ["?", "w"]]
        got = list(am.ac_match(pat, ["+", ["g", "y"], "y"], AC_PLUS))
        assert len(got) == 1
        assert got[0]["z"] == "y" and got[0]["w"] == ["g", "y"]

    def test_engine_equivalents_excludes_spurious_free(self):
        # End-to-end: the spurious (got (g y)) must not appear; only (got y).
        eng = RuleEngine.from_dsl("@r: (+ ?z:free(w) ?w) => (got :z)")
        eng.with_theory(Theory.from_dict({"+": {"ac": True, "identity": 0}}))
        forms = list(eng.equivalents(["+", ["g", "y"], "y"],
                                     include_unidirectional=True, max_depth=2))
        assert ["got", "y"] in forms
        assert ["got", ["g", "y"]] not in forms


class TestACSoundnessProperty:
    def test_every_yield_is_a_real_match(self):
        from rerum.normalize import normalize
        # Pattern with explicit + rest; verify each yield reconstructs an
        # AC-equal subject when substituted back.
        pat = ["+", ["?", "x"], ["?", "y"], ["?...", "rest"]]
        exp = ["+", "a", "b", "c", "d"]
        for b in am.ac_match(pat, exp, AC_PLUS):
            rebuilt = ["+", b["x"], b["y"]] + list(b["rest"])
            assert normalize(rebuilt, AC_PLUS) == normalize(exp, AC_PLUS)

    def test_completeness_matches_brute_force_small(self):
        # (+ ?x ?y ?rest...) over (+ a b c d): one yield per ordered pair (x,y)
        # of distinct elements; rest is the remaining two.
        pat = ["+", ["?", "x"], ["?", "y"], ["?...", "rest"]]
        exp = ["+", "a", "b", "c", "d"]
        got = list(am.ac_match(pat, exp, AC_PLUS))
        elems = ["a", "b", "c", "d"]
        expected = list(itertools.permutations(elems, 2))
        assert len(got) == len(expected)


from rerum.engine import RuleEngine


class TestEngineACMatching:
    def test_simplify_fires_ac_rule_across_arrangements(self):
        # Cancellation rule fires no matter where the cancelling pair sits.
        eng = RuleEngine.from_dsl(
            "@cancel: (+ ?x (- ?x) ?rest...) => (+ :rest...)"
        )
        eng.with_theory(Theory.from_dict({"+": {"ac": True, "identity": 0}}))
        # (- a) is not adjacent to a; positional matching would miss it.
        result = eng.simplify(["+", "a", "b", ["-", "a"]])
        # After cancelling a and (- a), only b remains; (+ b) collapses to b.
        assert result == "b"

    def test_no_theory_simplify_unchanged(self):
        # Without a theory, the same rule only fires on the exact arrangement.
        eng = RuleEngine.from_dsl(
            "@cancel: (+ ?x (- ?x) ?rest...) => (+ :rest...)"
        )
        # Positional: a and (- a) ARE adjacent here, so it fires syntactically.
        assert eng.simplify(["+", "a", ["-", "a"], "b"]) == ["+", "b"]
        # But not when separated -- no AC theory, no reordering.
        assert eng.simplify(["+", "a", "b", ["-", "a"]]) == ["+", "a", "b", ["-", "a"]]

    def test_apply_once_takes_first_ac_match(self):
        eng = RuleEngine.from_dsl("@r: (+ ?x ?y) => (pair :x :y)")
        eng.with_theory(Theory.from_dict({"+": {"ac": True, "identity": 0}}))
        result, meta = eng.apply_once(["+", "a", "b"])
        # First canonical assignment fires; result is a (pair ...).
        assert meta is not None and result[0] == "pair"

    def test_truncation_flag_exposed(self):
        eng = RuleEngine.from_dsl("@r: (+ ?x ?y ?z) => done")
        eng.with_theory(Theory.from_dict({"+": {"ac": True, "identity": 0}}))
        eng.set_ac_match_budget(2)        # tiny budget
        eng.simplify(["+", "a", "b", "c", "d", "e", "f"])
        assert eng.ac_match_truncated is True


class TestEquationalAC:
    def _ac_engine(self, dsl, sig):
        eng = RuleEngine.from_dsl(dsl)
        eng.with_theory(Theory.from_dict(sig))
        return eng

    def test_prove_equal_ac_distributivity(self):
        eng = self._ac_engine(
            "@distrib: (* (+ ?a ?b) ?c) => (+ (* :a :c) (* :b :c))",
            {"*": {"ac": True}})
        proof = eng.prove_equal(["*", ["+", "a", "b"], "c"],
                                ["+", ["*", "a", "c"], ["*", "b", "c"]],
                                include_unidirectional=True)
        assert proof is not None

    def test_equivalents_includes_ac_rewrite(self):
        eng = self._ac_engine(
            "@cancel: (+ ?x (- ?x) ?rest...) => (+ :rest...)",
            {"+": {"ac": True, "identity": 0}})
        forms = list(eng.equivalents(["+", "a", "b", ["-", "a"]],
                                     include_unidirectional=True, max_depth=3))
        # The cancelled form (b, possibly as (+ b)) is reachable.
        assert any(f == "b" or f == ["+", "b"] for f in forms)


class TestExamplesDemo:
    def test_ac_demo_cancels_across_arrangements(self):
        import os
        root = os.path.join(os.path.dirname(__file__), "..", "..", "examples")
        eng = RuleEngine.from_file(os.path.join(root, "ac_demo.rules"))
        with open(os.path.join(root, "ac_demo.theory.json")) as fh:
            eng.with_theory(Theory.from_json(fh.read()))
        # The cancelling pair is separated by an unrelated term.
        assert eng.simplify(["+", "a", "b", ["-", "a"]]) == "b"


class TestTypedRestConstraints:
    def test_const_rest_rejects_nonconstant_under_ac(self):
        # (+ ?x ?rest:const...) vs (+ 1 2 a): in every yielded binding the rest
        # list must contain only constants (numbers), never the symbol 'a'.
        pat = ["+", ["?", "x"], ["?...", "rest", "const"]]
        for b in am.ac_match(pat, ["+", 1, 2, "a"], AC_PLUS):
            assert all(not isinstance(e, str) for e in b["rest"])

    def test_var_rest_rejects_constants_positional(self):
        # (f ?x ?rest:var...) vs (f a b 1): rest tail [b,1] has a constant -> no match.
        pat = ["f", ["?", "x"], ["?...", "rest", "var"]]
        assert _matches(pat, ["f", "a", "b", 1], NO_AC) == []

    def test_unconstrained_rest_unchanged(self):
        pat = ["+", ["?", "x"], ["?...", "rest"]]
        got = list(am.ac_match(pat, ["+", "a", 1], AC_PLUS))
        assert got  # still matches (no constraint)


class TestEmptyPatternGuard:
    def test_empty_vs_empty_matches(self):
        assert _matches([], [], NO_AC) == [{}]
        assert _matches([], [], AC_PLUS) == [{}]

    def test_empty_vs_nonempty_no_match(self):
        assert _matches([], ["a"], NO_AC) == []

    def test_nested_empty_in_compound(self):
        assert _matches(["f", []], ["f", []], NO_AC) == [{}]
        assert _matches(["f", []], ["f", "a"], NO_AC) == []


class TestStrategyACRefusal:
    def test_bottomup_under_ac_refuses(self):
        import pytest
        eng = RuleEngine.from_dsl("@cancel: (+ ?x (- ?x) ?rest...) => (+ :rest...)")
        eng.with_theory(Theory.from_dict({"+": {"ac": True, "identity": 0}}))
        with pytest.raises(ValueError) as ei:
            eng.simplify(["+", "a", "b", ["-", "a"]], strategy="bottomup")
        assert "AC" in str(ei.value) or "ac" in str(ei.value)

    def test_topdown_under_ac_refuses(self):
        import pytest
        eng = RuleEngine.from_dsl("@cancel: (+ ?x (- ?x) ?rest...) => (+ :rest...)")
        eng.with_theory(Theory.from_dict({"+": {"ac": True, "identity": 0}}))
        with pytest.raises(ValueError) as ei:
            eng.simplify(["+", "a", "b", ["-", "a"]], strategy="topdown")
        assert "AC" in str(ei.value) or "ac" in str(ei.value)

    def test_exhaustive_under_ac_still_works(self):
        eng = RuleEngine.from_dsl("@cancel: (+ ?x (- ?x) ?rest...) => (+ :rest...)")
        eng.with_theory(Theory.from_dict({"+": {"ac": True, "identity": 0}}))
        assert eng.simplify(["+", "a", "b", ["-", "a"]]) == "b"

    def test_once_under_ac_allowed(self):
        eng = RuleEngine.from_dsl("@cancel: (+ ?x (- ?x) ?rest...) => (+ :rest...)")
        eng.with_theory(Theory.from_dict({"+": {"ac": True, "identity": 0}}))
        # "once" routes through apply_once which is AC-aware; must not raise.
        result = eng.simplify(["+", "a", "b", ["-", "a"]], strategy="once")
        assert result is not None  # no exception raised


class TestApplyOnceACCompleteness:
    def test_apply_once_finds_productive_ac_binding(self):
        # (+ (k ?x) ?y) => (found :x :y); subject (+ a (k b)): the productive
        # binding is x=b (matching (k ?x)), y=a -- the matcher must try bindings
        # until a productive one is found.
        eng = RuleEngine.from_dsl("@r: (+ (k ?x) ?y) => (found :x :y)")
        eng.with_theory(Theory.from_dict({"+": {"ac": True, "identity": 0}}))
        result, meta = eng.apply_once(["+", "a", ["k", "b"]])
        assert meta is not None and result[0] == "found"

    def test_apply_once_noop_rule_reports_not_applied(self):
        # A rule that matches but reproduces the input unchanged is a no-op:
        # apply_once must report (expr, None), not (expr, metadata).
        eng = RuleEngine.from_dsl("@id: (f ?x) => (f :x)")
        result, meta = eng.apply_once(["f", "a"])
        assert result == ["f", "a"]
        assert meta is None
