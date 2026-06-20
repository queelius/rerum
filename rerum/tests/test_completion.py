"""F5: basic Knuth-Bendix completion."""

from rerum import completion as cmp
from rerum import confluence as cf
from rerum.engine import RuleEngine


def _v(n):
    return ["?", n]


class TestBridgeHelpers:
    def test_term_to_skeleton_variable(self):
        assert cmp._term_to_skeleton(["?", "x"]) == [":", "x"]

    def test_term_to_skeleton_inverse_of_instantiate(self):
        # _term_to_skeleton is the inverse of instantiate_skeleton(.., {}).
        t = ["+", ["?", "x"], ["g", ["?", "y"]], "0"]
        skel = cmp._term_to_skeleton(t)
        assert skel == ["+", [":", "x"], ["g", [":", "y"]], "0"]
        assert cf.instantiate_skeleton(skel, {}) == t

    def test_dedup_preserves_order_drops_structural_duplicates(self):
        a = (["a"], "b")
        c = (["c"], "d")
        a2 = (["a"], "b")  # equal by value, distinct identity
        assert cmp._dedup([a, c, a2]) == [(["a"], "b"), (["c"], "d")]
