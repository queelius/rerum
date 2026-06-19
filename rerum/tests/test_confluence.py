"""F2: confluence and critical-pair diagnostics."""

from rerum import confluence as cf
from rerum.engine import RuleEngine
from rerum.normalize import Theory


class TestTermSurgery:
    def test_subterm_at_root_and_paths(self):
        t = ["+", ["*", "a", "b"], "c"]
        assert cf.subterm_at(t, ()) == t
        assert cf.subterm_at(t, (1,)) == ["*", "a", "b"]
        assert cf.subterm_at(t, (1, 2)) == "b"

    def test_replace_at(self):
        t = ["+", ["*", "a", "b"], "c"]
        assert cf.replace_at(t, (1, 2), "Z") == ["+", ["*", "a", "Z"], "c"]
        assert cf.replace_at(t, (), "Z") == "Z"
        # The original is not mutated.
        assert t == ["+", ["*", "a", "b"], "c"]

    def test_positions_are_non_variable_operand_paths(self):
        # (f (g ?x) a): non-variable positions are the root, the (g ?x) operand
        # and its operator-applied subterm, and the constant a -- NOT the ?x
        # variable node and NOT the operator-head index 0.
        t = ["f", ["g", ["?", "x"]], "a"]
        ps = set(cf.positions(t))
        assert () in ps          # whole term
        assert (1,) in ps        # (g ?x)
        assert (2,) in ps        # constant a
        assert (1, 1) not in ps  # ?x is a variable position -> excluded
        assert (0,) not in ps    # operator head -> not a position
