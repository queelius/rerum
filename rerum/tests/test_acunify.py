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
AC_PLUS = Theory.from_dict({"+": {"ac": True, "identity": 0}})


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
