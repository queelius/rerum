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
