"""AC-unification (Stickel)."""

from rerum import acunify as au


class TestUnifyBudget:
    def test_spend_decrements_and_flags(self):
        b = au.UnifyBudget(steps=2)
        assert b.spend() is True
        assert b.spend() is True
        assert b.spend() is False
        assert b.truncated is True

    def test_unbounded_when_none(self):
        b = au.UnifyBudget(steps=None)
        for _ in range(1000):
            assert b.spend() is True
        assert b.truncated is False


class TestHilbertBasis:
    def test_unit_coefficients_two_by_two(self):
        basis = au._hilbert_basis([1, 1], [1, 1])
        assert set(basis) == {(1, 0, 1, 0), (1, 0, 0, 1),
                              (0, 1, 1, 0), (0, 1, 0, 1)}

    def test_single_var_each(self):
        assert au._hilbert_basis([1], [1]) == [(1, 1)]

    def test_coefficient_two(self):
        assert au._hilbert_basis([2], [1]) == [(1, 2)]

    def test_every_basis_vector_is_a_solution(self):
        for a, b in ([1, 2], [2, 1]), ([1, 1, 1], [2, 1]):
            for vec in au._hilbert_basis(a, b):
                M = len(a)
                assert sum(a[i] * vec[i] for i in range(M)) == \
                    sum(b[j] * vec[M + j] for j in range(len(b)))
                assert any(vec)


from rerum.normalize import Theory, normalize

NO_AC = Theory.from_dict({})
# ac_unify implements PURE AC (multiset semantics: a+a != a). To verify against
# `normalize`, the test theory must be MULTISET-PRESERVING -- i.e. carry a
# `repeat` clause so a+a normalizes to (* 2 a) rather than collapsing to a. A
# bare {"ac": True} theory makes `normalize` IDEMPOTENT (a+a -> a, i.e. ACI),
# which has strictly more unifiers than pure AC and would mismatch ac_unify.
AC_PLUS = Theory.from_dict({
    "+": {"ac": True, "identity": 0, "repeat": {"op": "*", "via": "count"}},
    "*": {"ac": True, "identity": 1}})


def _unifiers(t1, t2, theory):
    return list(au.ac_unify(t1, t2, theory))


class TestDispatchNonAC:
    def test_variable_binds(self):
        got = _unifiers(["?", "x"], ["f", "a"], NO_AC)
        assert len(got) == 1 and got[0]["x"] == ["f", "a"]

    def test_atoms_equal_and_clash(self):
        assert _unifiers("a", "a", NO_AC) == [{}]
        assert _unifiers("a", "b", NO_AC) == []

    def test_occurs_check(self):
        assert _unifiers(["?", "x"], ["f", ["?", "x"]], NO_AC) == []

    def test_non_ac_compound_positional(self):
        got = _unifiers(["f", ["?", "x"], "b"], ["f", "a", ["?", "y"]], NO_AC)
        assert len(got) == 1
        s = got[0]
        assert s["x"] == "a" and s["y"] == "b"

    def test_head_or_arity_mismatch(self):
        assert _unifiers(["f", ["?", "x"]], ["g", "a"], NO_AC) == []
        assert _unifiers(["f", ["?", "x"]], ["f", "a", "b"], NO_AC) == []

    def test_agrees_with_f2_unify_on_non_ac(self):
        from rerum.confluence import unify
        t1 = ["f", ["?", "x"], ["g", ["?", "y"]]]
        t2 = ["f", "a", ["g", "b"]]
        syn = unify(t1, t2)
        ac = _unifiers(t1, t2, NO_AC)
        assert len(ac) == 1 and ac[0] == syn


from rerum.confluence import apply_subst


def _orig_vars(t):
    out = set()
    def walk(x):
        if isinstance(x, list) and len(x) == 2 and x[0] == "?":
            out.add(x[1]); return
        if isinstance(x, list):
            for s in x: walk(s)
    walk(t)
    return out


def _count_distinct(t1, t2, theory):
    seen = set()
    orig = _orig_vars(t1) | _orig_vars(t2)
    for s in au.ac_unify(t1, t2, theory):
        key = tuple(sorted((k, str(normalize(apply_subst(s, ["?", k]), theory)))
                           for k in orig))
        seen.add(key)
    return len(seen)


class TestStickelAllVariable:
    def test_x_plus_y_eq_u_plus_v_seven_unifiers(self):
        t1 = ["+", ["?", "x"], ["?", "y"]]
        t2 = ["+", ["?", "u"], ["?", "v"]]
        assert _count_distinct(t1, t2, AC_PLUS) == 7

    def test_all_yields_are_sound(self):
        t1 = ["+", ["?", "x"], ["?", "y"]]
        t2 = ["+", ["?", "u"], ["?", "v"]]
        for s in au.ac_unify(t1, t2, AC_PLUS):
            assert normalize(apply_subst(s, t1), AC_PLUS) == \
                normalize(apply_subst(s, t2), AC_PLUS)

    def test_x_plus_x_eq_y_plus_y_sound(self):
        t1 = ["+", ["?", "x"], ["?", "x"]]
        t2 = ["+", ["?", "y"], ["?", "y"]]
        for s in au.ac_unify(t1, t2, AC_PLUS):
            assert normalize(apply_subst(s, t1), AC_PLUS) == \
                normalize(apply_subst(s, t2), AC_PLUS)
        assert _count_distinct(t1, t2, AC_PLUS) >= 1


class TestStickelNonVariable:
    def test_x_plus_y_eq_a_plus_b_two_unifiers(self):
        t1 = ["+", ["?", "x"], ["?", "y"]]
        t2 = ["+", "a", "b"]
        assert _count_distinct(t1, t2, AC_PLUS) == 2

    def test_x_eq_a_plus_b_one_unifier(self):
        assert _count_distinct(["?", "x"], ["+", "a", "b"], AC_PLUS) == 1

    def test_ground_equal_one_unifier(self):
        assert _count_distinct(["+", "a", "b"], ["+", "a", "b"], AC_PLUS) == 1

    def test_distinct_constants_no_unifier(self):
        assert _count_distinct("a", "b", AC_PLUS) == 0
        assert _count_distinct(["+", "a", "b"], ["+", "a", "c"], AC_PLUS) == 0

    def test_nested_free_symbol(self):
        t1 = ["+", ["f", ["?", "x"]], ["?", "y"]]
        t2 = ["+", "a", ["f", "b"]]
        got = list(au.ac_unify(t1, t2, AC_PLUS))
        assert _count_distinct(t1, t2, AC_PLUS) == 1
        s = got[0]
        assert apply_subst(s, ["?", "x"]) == "b"
        assert apply_subst(s, ["?", "y"]) == "a"

    def test_all_nonvariable_yields_sound(self):
        for t1, t2 in (
            (["+", ["?", "x"], ["?", "y"]], ["+", "a", "b"]),
            (["+", ["f", ["?", "x"]], ["?", "y"]], ["+", "a", ["f", "b"]]),
        ):
            for s in au.ac_unify(t1, t2, AC_PLUS):
                assert normalize(apply_subst(s, t1), AC_PLUS) == \
                    normalize(apply_subst(s, t2), AC_PLUS)


import itertools


class TestVerificationBar:
    def test_soundness_battery(self):
        V = lambda n: ["?", n]
        problems = [
            (["+", V("x"), V("y")], ["+", V("u"), V("v")]),
            (["+", V("x"), V("y"), V("z")], ["+", "a", "b", "c"]),
            (["+", V("x"), "a"], ["+", "b", V("y")]),
            (["+", ["f", V("x")], V("y")], ["+", "a", ["f", "b"]]),
        ]
        for t1, t2 in problems:
            for s in au.ac_unify(t1, t2, AC_PLUS):
                assert normalize(apply_subst(s, t1), AC_PLUS) == \
                    normalize(apply_subst(s, t2), AC_PLUS)

    def test_completeness_vs_brute_force(self):
        t1 = ["+", ["?", "x"], ["?", "y"]]
        t2 = ["+", "a", "b"]
        oracle = set()
        for sx in (["a"], ["b"], ["a", "b"]):
            for sy in (["a"], ["b"], ["a", "b"]):
                cand = {"x": sx, "y": sy}
                e1 = normalize(["+"] + [c for v in ("x", "y") for c in cand[v]],
                               AC_PLUS)
                if e1 == normalize(t2, AC_PLUS):
                    oracle.add((tuple(sx), tuple(sy)))
        found = set()
        for s in au.ac_unify(t1, t2, AC_PLUS):
            xv = normalize(apply_subst(s, ["?", "x"]), AC_PLUS)
            yv = normalize(apply_subst(s, ["?", "y"]), AC_PLUS)
            tx = tuple(xv[1:]) if isinstance(xv, list) else (xv,)
            ty = tuple(yv[1:]) if isinstance(yv, list) else (yv,)
            found.add((tx, ty))
        assert oracle <= found

    def test_ac_match_cross_check(self):
        from rerum import acmatch
        t1 = ["+", ["?", "x"], ["?", "y"]]
        t2 = ["+", "a", "b"]
        match_keyed = {frozenset((k, _t(v)) for k, v in b.to_dict().items())
                       for b in acmatch.ac_match(t1, t2, AC_PLUS)}
        unify_sols = set()
        for s in au.ac_unify(t1, t2, AC_PLUS):
            d = {k: apply_subst(s, ["?", k]) for k in ("x", "y")}
            if all(not _has_var(v) for v in d.values()):
                unify_sols.add(frozenset((k, _t(v)) for k, v in d.items()))
        assert match_keyed <= unify_sols

    def test_budget_truncation_sound(self):
        t1 = ["+", ["?", "x"], ["?", "y"], ["?", "z"]]
        t2 = ["+", ["?", "p"], ["?", "q"], ["?", "r"]]
        budget = au.UnifyBudget(steps=3)
        got = list(au.ac_unify(t1, t2, AC_PLUS, budget=budget))
        assert budget.truncated is True
        for s in got:
            assert normalize(apply_subst(s, t1), AC_PLUS) == \
                normalize(apply_subst(s, t2), AC_PLUS)

    def test_determinism(self):
        t1 = ["+", ["?", "x"], ["?", "y"]]
        t2 = ["+", ["?", "u"], ["?", "v"]]
        a = [sorted(s.items()) for s in au.ac_unify(t1, t2, AC_PLUS)]
        b = [sorted(s.items()) for s in au.ac_unify(t1, t2, AC_PLUS)]
        assert a == b


def _has_var(t):
    if isinstance(t, list) and len(t) == 2 and t[0] == "?":
        return True
    if isinstance(t, list):
        return any(_has_var(s) for s in t)
    return False


def _t(v):
    return tuple(_t(x) for x in v) if isinstance(v, list) else v


class TestReexportsAndDemo:
    def test_public_reexports(self):
        import rerum
        for name in ("ac_unify", "UnifyBudget"):
            assert name in rerum.__all__
            assert hasattr(rerum, name)

    def test_import_smoke_no_cycle(self):
        import importlib
        importlib.import_module("rerum.acunify")
        importlib.import_module("rerum.confluence")

    def test_demo_file_loads_and_problem_solves(self):
        import os
        from rerum.normalize import Theory
        root = os.path.join(os.path.dirname(__file__), "..", "..", "examples")
        theory = Theory.from_dict({
            "+": {"ac": True, "identity": 0, "repeat": {"op": "*", "via": "count"}},
            "*": {"ac": True, "identity": 1}})
        sols = list(au.ac_unify(["+", ["?", "x"], ["?", "y"]],
                                ["+", "a", "b"], theory))
        assert sols  # demo problem has solutions
        assert os.path.exists(os.path.join(root, "acunify_demo.rules"))


class TestReviewFixes:
    # Opus holistic review found two BLOCKING soundness bugs the suite missed.

    def test_repeated_nonvar_atom_sound_and_complete(self):
        # (+ ?x ?y) =? (+ a a): the ONLY unifier is {x:a, y:a}; previously this
        # yielded 3 wholly-unsound unifiers and dropped the real one. Root cause:
        # a non-variable atom of multiplicity 2 was merged into one Diophantine
        # column instead of two weight-1 columns.
        t1 = ["+", ["?", "x"], ["?", "y"]]
        t2 = ["+", "a", "a"]
        for s in au.ac_unify(t1, t2, AC_PLUS):
            assert normalize(apply_subst(s, t1), AC_PLUS) == \
                normalize(apply_subst(s, t2), AC_PLUS)
        assert _count_distinct(t1, t2, AC_PLUS) == 1

    def test_repeated_compound_atom_sound(self):
        t1 = ["+", ["?", "x"], ["?", "y"]]
        t2 = ["+", ["*", "a", "b"], ["*", "a", "b"]]
        for s in au.ac_unify(t1, t2, AC_PLUS):
            assert normalize(apply_subst(s, t1), AC_PLUS) == \
                normalize(apply_subst(s, t2), AC_PLUS)
        assert _count_distinct(t1, t2, AC_PLUS) == 1

    def test_ac_inside_ac_coupling(self):
        # (+ (* ?x b) c) =? (+ (* a ?y) c): the * coupling must unify modulo AC.
        # A gensym variable capture across the recursive AC node previously made
        # this yield NOTHING. Fix: seed the avoid set from bindings + free vars.
        t1 = ["+", ["*", ["?", "x"], "b"], "c"]
        t2 = ["+", ["*", "a", ["?", "y"]], "c"]
        sols = list(au.ac_unify(t1, t2, AC_PLUS))
        assert sols  # non-empty (was [] before the fix)
        for s in sols:
            assert normalize(apply_subst(s, t1), AC_PLUS) == \
                normalize(apply_subst(s, t2), AC_PLUS)
        assert any(apply_subst(s, ["?", "x"]) == "a" and
                   apply_subst(s, ["?", "y"]) == "b" for s in sols)
