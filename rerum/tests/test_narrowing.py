"""F6: narrowing (unification-driven backward rewriting)."""

from rerum import narrowing as nw


class TestPureHelpers:
    def test_positions_non_variable_only(self):
        # (add ?x (s z)): root [], the (s z) at [2], the z at [2,1].
        # The variable ?x at [1] is NOT a position.
        term = ["add", ["?", "x"], ["s", "z"]]
        assert sorted(nw._positions(term)) == sorted([[], [2], [2, 1]])

    def test_positions_atom(self):
        assert list(nw._positions("z")) == [[]]

    def test_positions_bare_variable(self):
        assert list(nw._positions(["?", "x"])) == []

    def test_term_at(self):
        term = ["add", ["?", "x"], ["s", "z"]]
        assert nw._term_at(term, []) == term
        assert nw._term_at(term, [2]) == ["s", "z"]
        assert nw._term_at(term, [2, 1]) == "z"

    def test_replace_at(self):
        term = ["add", ["?", "x"], ["s", "z"]]
        assert nw._replace_at(term, [2, 1], "q") == ["add", ["?", "x"], ["s", "q"]]
        assert nw._replace_at(term, [], "done") == "done"
        # original is unchanged (functional)
        assert term == ["add", ["?", "x"], ["s", "z"]]

    def test_compose_applies_second_through_first(self):
        # compose(s2, s1) = s2 . s1 : apply s1 first, then s2.
        s1 = {"x": ["s", ["?", "y"]]}
        s2 = {"y": "z"}
        out = nw._compose(s2, s1)
        assert out["x"] == ["s", "z"]   # s2 applied through s1's range
        assert out["y"] == "z"          # s2's own binding kept

    def test_compose_first_wins_on_overlap(self):
        s1 = {"x": "a"}
        s2 = {"x": "b"}
        assert nw._compose(s2, s1)["x"] == "a"   # s1 binding survives
