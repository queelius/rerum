"""F2: confluence and critical-pair diagnostics."""

import pytest

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


class TestUnify:
    def test_two_distinct_variables_unify(self):
        s = cf.unify(["?", "x"], ["?", "y"])
        assert s is not None
        # x resolves to the y-variable.
        assert cf.apply_subst(s, ["?", "x"]) == ["?", "y"]

    def test_variable_binds_compound(self):
        s = cf.unify(["?", "x"], ["g", "a"])
        assert s is not None and cf.apply_subst(s, ["?", "x"]) == ["g", "a"]

    def test_occurs_check_fails(self):
        assert cf.unify(["?", "x"], ["f", ["?", "x"]]) is None

    def test_head_and_arity_clashes_fail(self):
        assert cf.unify(["f", "a"], ["g", "a"]) is None
        assert cf.unify(["f", "a"], ["f", "a", "b"]) is None

    def test_atoms(self):
        assert cf.unify("a", "a") == {}
        assert cf.unify("a", "b") is None
        assert cf.unify(["f", "a"], "a") is None  # compound vs atom

    def test_compound_unifies_pairwise(self):
        s = cf.unify(["f", ["?", "x"], "b"], ["f", "a", ["?", "y"]])
        assert s is not None
        assert cf.apply_subst(s, ["?", "x"]) == "a"
        assert cf.apply_subst(s, ["?", "y"]) == "b"

    def test_repeated_variable_in_one_pattern(self):
        # (f ?x ?x): a repeated binder (e.g. idempotence f(x,x)=>x). The two
        # occurrences must take the SAME value, so distinct operands clash.
        assert cf.unify(["f", ["?", "x"], ["?", "x"]], ["f", "a", "a"]) is not None
        assert cf.apply_subst(
            cf.unify(["f", ["?", "x"], ["?", "x"]], ["f", "a", "a"]),
            ["?", "x"]) == "a"
        assert cf.unify(["f", ["?", "x"], ["?", "x"]], ["f", "a", "b"]) is None

    def test_variable_chain_resolves_in_one_pass(self):
        # (f ?x ?y) ~ (f ?y a): x and y both resolve to a (the fully-applied
        # invariant must hold across the pairwise steps).
        s = cf.unify(["f", ["?", "x"], ["?", "y"]], ["f", ["?", "y"], "a"])
        assert s is not None
        assert cf.apply_subst(s, ["?", "x"]) == "a"
        assert cf.apply_subst(s, ["?", "y"]) == "a"


class TestUnifyRefusal:
    @pytest.mark.parametrize("bad", [
        ["?c", "x"], ["?v", "x"], ["?free", "x", "y"], ["?...", "r"],
    ])
    def test_unsupported_node_raises(self, bad):
        with pytest.raises(cf.UnsupportedPattern):
            cf.unify(["?", "x"], bad)

    def test_mixed_var_vs_typed_raises_not_binds(self):
        # Refuse-FIRST: the typed node must raise, not be bound as opaque.
        with pytest.raises(cf.UnsupportedPattern):
            cf.unify(["?", "x"], ["?c", "y"])

    def test_nested_unsupported_raises(self):
        with pytest.raises(cf.UnsupportedPattern):
            cf.unify(["f", ["?c", "x"]], ["f", "a"])


class TestApplySubstAndInstantiate:
    def test_apply_subst_recurses_and_leaves_free(self):
        s = {"x": "a"}
        assert cf.apply_subst(s, ["f", ["?", "x"], ["?", "y"]]) == ["f", "a", ["?", "y"]]

    def test_apply_subst_single_pass_on_resolved_subst(self):
        # A fully-applied subst: x -> (g y), y -> b. One pass resolves x fully.
        s = cf.unify(["f", ["?", "x"], ["?", "y"]], ["f", ["g", ["?", "y"]], "b"])
        assert s is not None
        assert cf.apply_subst(s, ["?", "x"]) == ["g", "b"]

    def test_instantiate_skeleton_substitutes_colon_vars(self):
        # skeleton (h :x) under {x -> (k z)} becomes (h (k z)); free :w stays ?w.
        sk = ["h", [":", "x"], [":", "w"]]
        s = {"x": ["k", ["?", "z"]]}
        assert cf.instantiate_skeleton(sk, s) == ["h", ["k", ["?", "z"]], ["?", "w"]]

    def test_instantiate_skeleton_bare_root_colon_var(self):
        # A skeleton that is just :x at the root (rule RHS is a single var).
        assert cf.instantiate_skeleton([":", "x"], {"x": ["g", "a"]}) == ["g", "a"]
        assert cf.instantiate_skeleton([":", "x"], {}) == ["?", "x"]


class TestRenameAndAnalyzable:
    def test_rename_apart_makes_fresh_variables(self):
        pat, sk = cf.rename_apart(["f", ["?", "x"]], ["g", [":", "x"]], {"x"})
        # The pattern variable is renamed away from the avoided "x".
        assert pat != ["f", ["?", "x"]]
        new_name = pat[1][1]
        assert new_name != "x"
        # The skeleton's [":", x] reference is renamed to the SAME new name.
        assert sk == ["g", [":", new_name]]

    def test_is_analyzable_accepts_first_order(self):
        assert cf.is_analyzable(["f", ["?", "x"]], ["g", [":", "x"]], None) is True

    def test_is_analyzable_refuses_conditional(self):
        assert cf.is_analyzable(["f", ["?", "x"]], [":", "x"], ["pos", [":", "x"]]) is False

    def test_is_analyzable_refuses_bad_pattern_forms(self):
        assert cf.is_analyzable(["f", ["?...", "r"]], [":", "r"], None) is False
        assert cf.is_analyzable(["f", ["?c", "x"]], [":", "x"], None) is False

    def test_is_analyzable_refuses_bad_skeleton_forms(self):
        assert cf.is_analyzable(["f", ["?", "x"]], ["!", "+", [":", "x"], 1], None) is False
        assert cf.is_analyzable(["f", ["?", "x"]], [":...", "x"], None) is False


def _rule(name, pattern, skeleton, condition=None):
    return cf.DirectedRule(name=name, pattern=pattern, skeleton=skeleton,
                           condition=condition)


class TestCriticalPairs:
    def test_overlap_between_two_rules(self):
        # r2 (g ?x)=>(k ?x) overlaps r1 (f (g ?x))=>(h ?x) at position (1,).
        r1 = _rule("r1", ["f", ["g", ["?", "x"]]], ["h", [":", "x"]])
        r2 = _rule("r2", ["g", ["?", "x"]], ["k", [":", "x"]])
        pairs, not_analyzed = cf.critical_pairs([r1, r2])
        assert not_analyzed == []
        # One overlap (r2 into r1 at (1,)) plus possibly others; find it.
        found = [cp for cp in pairs
                 if cp.rule_left == "r1" and cp.rule_right == "r2"
                 and cp.position == (1,)]
        assert len(found) == 1

    def test_distinct_rule_variables_are_kept_apart(self):
        # Both rules use ?x; the overlap must not conflate them.
        r1 = _rule("r1", ["f", ["g", ["?", "x"]]], ["h", [":", "x"]])
        r2 = _rule("r2", ["g", ["?", "x"]], ["k", [":", "x"]])
        pairs, _ = cf.critical_pairs([r1, r2])
        cp = [c for c in pairs if c.rule_left == "r1" and c.rule_right == "r2"
              and c.position == (1,)][0]
        # left = (h ?v); right = (f (k ?v)). The CORE apartness invariant:
        # the unified variable appears as ONE shared renamed node in both legs
        # (left's argument equals right's inner (k ...) argument), proving the
        # two rules' ?x were renamed apart and then unified coherently.
        assert cp.left[0] == "h" and cp.right[0] == "f"
        shared = cp.left[1]
        assert cf._is_var(shared)            # ["?", <fresh>]
        assert cp.right[1][0] == "k"
        assert cp.right[1][1] == shared      # same renamed variable node

    def test_self_overlap_excludes_trivial_root(self):
        # (f (f ?x)) => (f ?x): the only critical pair is the NON-root self
        # overlap at (1,); the trivial root overlap (i==j, p==()) is excluded.
        r = _rule("r", ["f", ["f", ["?", "x"]]], ["f", [":", "x"]])
        pairs, _ = cf.critical_pairs([r])
        positions_seen = {cp.position for cp in pairs}
        assert () not in positions_seen
        assert (1,) in positions_seen

    def test_conditional_and_bad_rules_are_not_analyzed(self):
        good = _rule("good", ["f", ["?", "x"]], [":", "x"])
        cond = _rule("cond", ["g", ["?", "x"]], [":", "x"], condition=["p", [":", "x"]])
        rest = _rule("rest", ["h", ["?...", "r"]], [":", "r"])
        pairs, not_analyzed = cf.critical_pairs([good, cond, rest])
        assert "cond" in not_analyzed
        assert "rest" in not_analyzed
        # The good rule is still analyzed (it just may yield no overlaps here).
        assert "good" not in not_analyzed


class TestCheckConfluence:
    def test_confluent_set_is_locally_confluent(self):
        # r2 makes a critical pair with r1; r3 joins it. Locally confluent.
        # (RHS uses :x -- the rerum skeleton substitution syntax.)
        eng = RuleEngine.from_dsl("""
            @r1: (f (g ?x)) => (h :x)
            @r2: (g ?x) => (k :x)
            @r3: (f (k ?x)) => (h :x)
        """)
        report = cf.check_confluence(eng)
        assert report.locally_confluent is True
        assert report.analyzed_pair_count >= 1
        assert report.non_joinable == []
        assert report.unknown == []

    def test_non_confluent_set_is_flagged(self):
        eng = RuleEngine.from_dsl("""
            @l: (f (f ?x)) => a
            @r: (f ?x) => b
        """)
        report = cf.check_confluence(eng)
        assert report.locally_confluent is False
        assert len(report.non_joinable) >= 1
        names = {cp.rule_left for cp in report.non_joinable} | \
                {cp.rule_right for cp in report.non_joinable}
        assert "l" in names and "r" in names

    def test_no_overlap_is_vacuously_confluent(self):
        eng = RuleEngine.from_dsl("""
            @p: (f ?x) => (g :x)
            @q: (h ?x) => (k :x)
        """)
        report = cf.check_confluence(eng)
        assert report.locally_confluent is True
        assert report.analyzed_pair_count == 0
        assert report.not_analyzed == []

    def test_not_analyzed_is_surfaced(self):
        # A rest-pattern rule is refused; with it the ONLY rule, there are no
        # analyzable pairs. Vacuously True, but not_analyzed + count==0 make
        # clear nothing was actually checked.
        eng = RuleEngine.from_dsl("@rest: (f ?x...) => (g :x...)")
        report = cf.check_confluence(eng)
        assert report.not_analyzed == ["rest"]
        assert report.analyzed_pair_count == 0
        assert report.locally_confluent is True

    def test_unknown_pair_blocks_verdict_and_returns(self):
        # r1 and r2 share LHS (a); one reduct (b) is a normal form, the other
        # (s c) grows forever via @grow, so the critical pair never joins and is
        # UNKNOWN (not a false verdict). The call must RETURN, not hang.
        eng = RuleEngine.from_dsl("""
            @r1: (a) => (b)
            @r2: (a) => (s c)
            @grow: (s ?x) => (s (s :x))
        """)
        report = cf.check_confluence(eng, max_steps=20)
        assert len(report.unknown) >= 1
        assert report.locally_confluent is False

    def test_joinability_modulo_theory(self):
        # Overlap whose two sides are equal only after AC-normalization.
        eng = RuleEngine.from_dsl("""
            @l: (m ?x) => (+ :x c)
            @r: (m ?x) => (+ c :x)
        """)
        ac = Theory.from_dict({"+": {"ac": True}})
        # Without a theory: (+ ?x c) and (+ c ?x) are distinct normal forms.
        rep_no = cf.check_confluence(eng)
        assert rep_no.locally_confluent is False
        # With the AC theory: they canonicalize equal -> joinable.
        eng.with_theory(ac)
        rep_ac = cf.check_confluence(eng)
        assert rep_ac.locally_confluent is True

    def test_general_boolean_and_arithmetic(self):
        # Same code analyzes a non-arithmetic rule set.
        boolean = RuleEngine.from_dsl("@dn: (not (not ?x)) => :x")
        arith = RuleEngine.from_dsl("@z: (+ ?x 0) => :x")
        assert cf.check_confluence(boolean).locally_confluent is True
        assert cf.check_confluence(arith).locally_confluent is True


class TestEngineWrappersAndExports:
    def test_engine_check_confluence_delegates(self):
        eng = RuleEngine.from_dsl("""
            @l: (f (f ?x)) => a
            @r: (f ?x) => b
        """)
        method = eng.check_confluence()
        direct = cf.check_confluence(eng)
        assert method.locally_confluent == direct.locally_confluent is False

    def test_engine_critical_pairs_delegates(self):
        eng = RuleEngine.from_dsl("@r: (f (f ?x)) => (f :x)")
        pairs = eng.critical_pairs()
        assert isinstance(pairs, list)
        assert all(isinstance(cp, cf.CriticalPair) for cp in pairs)

    def test_disabled_group_contributes_no_overlap(self):
        # DSL groups are [section] headers; rules under [grp] carry that tag.
        eng = RuleEngine.from_dsl("""
            [grp]
            @l: (f (f ?x)) => a
            @r: (f ?x) => b
        """)
        eng.disable_group("grp")
        report = eng.check_confluence()
        # With the only rules disabled, no overlaps -> vacuously confluent.
        assert report.locally_confluent is True
        assert report.analyzed_pair_count == 0

    def test_public_reexports(self):
        import rerum
        for name in ("check_confluence", "critical_pairs", "CriticalPair",
                     "ConfluenceReport", "unify", "apply_subst",
                     "UnsupportedPattern"):
            assert name in rerum.__all__
            assert hasattr(rerum, name)


class TestRecursionGuard:
    def test_recursion_error_maps_to_unknown(self):
        # If reduction blows the recursion limit on a pathological growth rule,
        # the pair is UNDECIDED (None), not an exception -- so the call returns.
        class _Boom:
            def simplify(self, expr, max_steps=1000):
                raise RecursionError("deep term")

            def _simplify_once(self, expr):
                return expr

            def _canonicalize(self, expr):
                return expr

        cp = cf.CriticalPair(left=["a"], right=["b"], rule_left="x",
                             rule_right="y", position=())
        assert cf._decide_joinable(_Boom(), cp, 10) is None


class TestBudgetHonoredOnFastPath:
    def test_small_max_steps_bounds_growth_rule(self):
        from rerum.engine import RuleEngine
        eng = RuleEngine.from_dsl("@g: (f ?x) => (f (s :x))")
        small = eng.simplify(["f", "z"], max_steps=3)
        big = eng.simplify(["f", "z"], max_steps=50)
        def depth(t):
            return 1 + max((depth(c) for c in t[1:]), default=0) if isinstance(t, list) else 0
        assert depth(small) < depth(big)
