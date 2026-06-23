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


from rerum.engine import RuleEngine


def _peano_engine():
    return RuleEngine.from_dsl("""
        @add0: (add z ?y) => :y
        @addS: (add (s ?x) ?y) => (s (add :x :y))
    """)


class TestNarrowReachability:
    def test_solves_for_the_missing_addend(self):
        # find ?x such that add(?x, s(z)) reduces to s(s(z)) -> ?x = s(z).
        eng = _peano_engine()
        result = nw.narrow(eng, ["add", ["?", "x"], ["s", "z"]],
                           ["s", ["s", "z"]])
        assert result.found is True
        assert result.substitution["x"] == ["s", "z"]

    def test_immediate_goal_when_start_unifies_target(self):
        eng = _peano_engine()
        result = nw.narrow(eng, ["?", "x"], ["s", "z"])
        # ?x unifies the target directly: ?x = (s z), zero narrowing steps.
        assert result.found is True
        assert result.substitution["x"] == ["s", "z"]
        assert result.derivation == []

    def test_budget_exhaustion(self):
        # A non-terminating rule (loop ?x) => (loop (s ?x)) never reaches done.
        eng = RuleEngine.from_dsl("@loop: (loop ?x) => (loop (s ?x))")
        result = nw.narrow(eng, ["loop", "z"], "done", max_nodes=20)
        assert result.found is False
        assert result.exhausted is True

    def test_no_solution_finite_tree(self):
        # add(z, z) -> z, never s(z); finite tree, no solution.
        eng = _peano_engine()
        result = nw.narrow(eng, ["add", "z", "z"], ["s", "z"], max_nodes=1000)
        assert result.found is False
        assert result.exhausted is False

    def test_determinism(self):
        eng = _peano_engine()
        start, target = ["add", ["?", "x"], ["s", "z"]], ["s", ["s", "z"]]
        a = nw.narrow(eng, start, target).substitution
        b = nw.narrow(eng, start, target).substitution
        assert a == b


def _append_engine():
    return RuleEngine.from_dsl("""
        @app0: (app nil ?ys) => :ys
        @appC: (app (cons ?x ?xs) ?ys) => (cons :x (app :xs :ys))
    """)


def _lst(*items):
    out = "nil"
    for it in reversed(items):
        out = ["cons", it, out]
    return out


class TestSolveEquation:
    def test_append_solves_the_prefix(self):
        # solve app(?xs, [c]) =? [a, b, c]  ->  ?xs = [a, b].
        eng = _append_engine()
        result = nw.solve_equation(eng,
                                   ["app", ["?", "xs"], _lst("c")],
                                   _lst("a", "b", "c"))
        assert result.found is True
        assert result.substitution["xs"] == _lst("a", "b")

    def test_already_equal_solves_trivially(self):
        eng = _peano_engine()
        result = nw.solve_equation(eng, ["s", "z"], ["s", "z"])
        assert result.found is True
        assert result.substitution == {}

    def test_answer_substitution_is_sound(self):
        # The returned sigma must re-derive: sigma(s) and sigma(t) join under
        # the engine's simplify.
        eng = _append_engine()
        s = ["app", ["?", "xs"], _lst("c")]
        t = _lst("a", "b", "c")
        result = nw.solve_equation(eng, s, t)
        sigma = result.substitution
        s_sub = nw.apply_subst(sigma, s)
        assert eng.simplify(s_sub) == eng.simplify(t)

    def test_no_solution_returns_not_found(self):
        # app(?xs, [c]) can never equal [a] (length mismatch); finite search.
        eng = _append_engine()
        result = nw.solve_equation(eng, ["app", ["?", "xs"], _lst("c")],
                                   _lst("a"), max_nodes=500)
        assert result.found is False


class TestReexportsAndDemo:
    def test_public_reexports(self):
        import rerum
        for name in ("narrow", "solve_equation", "narrow_step",
                     "NarrowResult", "NarrowStep"):
            assert name in rerum.__all__
            assert hasattr(rerum, name)

    def test_demo_solves_via_general_engine(self):
        import os
        root = os.path.join(os.path.dirname(__file__), "..", "..", "examples")
        eng = RuleEngine.from_file(os.path.join(root, "narrowing_demo.rules"))
        # find ?x with add(?x, s(z)) = s(s(z)) -> ?x = s(z).
        result = nw.narrow(eng, ["add", ["?", "x"], ["s", "z"]],
                           ["s", ["s", "z"]])
        assert result.found and result.substitution["x"] == ["s", "z"]

    def test_import_smoke_no_cycle(self):
        import importlib
        importlib.import_module("rerum.narrowing")
        importlib.import_module("rerum.confluence")


class TestReviewFixes:
    # Opus holistic review found a soundness hole (dangling skeleton refs) and a
    # budget-honesty gap (max_depth truncation). These pin the fixes.

    def test_dangling_skeleton_ref_rule_is_skipped(self):
        # @r: (g a) => :x has an EXTRA RHS var (:x with no ?x binder, so
        # Var(r) not subset Var(l)). instantiate_skeleton models it as a free
        # var but the engine reduces it to the symbol "x"; narrowing must SKIP
        # the rule. The former spurious answer (u=a "solving" (g ?u) -> (h c))
        # must be gone.
        eng = RuleEngine.from_dsl("@r: (g a) => :x")
        result = nw.narrow(eng, ["g", ["?", "u"]], ["h", "c"], max_nodes=50)
        assert result.found is False

    def test_dangling_ref_does_not_break_wellformed_rules(self):
        # A well-formed rule alongside a dangling one still narrows.
        eng = RuleEngine.from_dsl("""
            @bad: (g a) => :x
            @good: (f ?y) => :y
        """)
        result = nw.narrow(eng, ["f", ["?", "u"]], "c")
        assert result.found is True and result.substitution["u"] == "c"

    def test_max_depth_truncation_sets_exhausted(self):
        # (loop ?x) => (loop (s ?x)) grows without bound; max_depth (NOT
        # max_nodes) truncates -> exhausted=True (inconclusive), not a finite
        # exhausted tree.
        eng = RuleEngine.from_dsl("@loop: (loop ?x) => (loop (s ?x))")
        result = nw.narrow(eng, ["loop", "z"], "done",
                           max_nodes=100000, max_depth=5)
        assert result.found is False
        assert result.exhausted is True


class TestNarrowExhaustedPrecision:
    def test_cyclic_finite_tree_not_inconclusive(self):
        # p <-> q 2-cycle: a finite cyclic tree.  With max_depth=2 the BFS
        # reaches the state (p z)/theta={x:z} at depth=2.  Its only narrowing
        # successor is (q z)/theta={x:z}, which is already in `seen`.  The old
        # depth-cap branch fires depth_capped=True without checking membership,
        # incorrectly reporting exhausted=True (inconclusive).  The fix checks
        # membership first; since every cap-depth successor here IS already in
        # `seen`, depth_capped stays False and exhausted is correctly False.
        eng = RuleEngine.from_dsl("""
            @a: (p ?x) => (q :x)
            @b: (q ?x) => (p :x)
        """)
        result = nw.narrow(eng, ["p", "z"], "done", max_nodes=100000, max_depth=2)
        assert result.found is False
        assert result.exhausted is False
