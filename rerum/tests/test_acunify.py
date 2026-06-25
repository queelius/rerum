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
