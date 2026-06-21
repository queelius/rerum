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
