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


# Peano add as (l_pattern, r_term, rule_id) triples (r already a TERM: [":",n]->["?",n]).
ADD0 = (["add", "z", ["?", "y"]], ["?", "y"], "add0")
ADDS = (["add", ["s", ["?", "x"]], ["?", "y"]],
        ["s", ["add", ["?", "x"], ["?", "y"]]], "addS")
PEANO = [ADD0, ADDS]


class TestNarrowStep:
    def test_root_successors(self):
        # narrow (add ?a (s z)) at the root with both rules.
        term = ["add", ["?", "a"], ["s", "z"]]
        steps = list(nw.narrow_step(term, PEANO))
        succs = [s.successor for s in steps]
        # add0: ?a=z, ?y=(s z) -> successor (s z)
        assert ["s", "z"] in succs
        # addS: ?a=(s ?x'), ?y=(s z) -> successor (s (add ?x' (s z)))
        assert any(s[0] == "s" and s[1][0] == "add" for s in succs)

    def test_step_carries_sigma_and_position(self):
        term = ["add", ["?", "a"], ["s", "z"]]
        steps = list(nw.narrow_step(term, PEANO))
        add0_step = next(s for s in steps if s.successor == ["s", "z"])
        assert add0_step.sigma["a"] == "z"
        assert add0_step.position == []
        assert add0_step.rule_id == "add0"

    def test_variable_position_yields_nothing_extra(self):
        # The ?a at [1] is a variable position; no rule narrows there.
        term = ["add", ["?", "a"], ["s", "z"]]
        steps = list(nw.narrow_step(term, PEANO))
        assert all(s.position != [1] for s in steps)

    def test_no_match_when_no_rule_unifies(self):
        # (foo ?a) unifies no Peano LHS.
        assert list(nw.narrow_step(["foo", ["?", "a"]], PEANO)) == []

    def test_rename_apart_prevents_capture(self):
        # term reuses the rule's own variable name ?y; the rule must be renamed
        # apart so its ?y does not capture the term's ?y.
        term = ["add", ["?", "y"], "z"]   # term has ?y
        steps = list(nw.narrow_step(term, [ADD0]))
        # add0 (add z ?y) vs (add ?y z): ?y(term)=z and rule-?y'=z; successor z.
        assert any(s.successor == "z" for s in steps)
