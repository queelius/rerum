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
