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
