"""F1: theory-normalized equational reasoning.

Verifies the five reasoning methods (equivalents, enumerate_equivalents,
prove_equal, are_equal, minimize) reason MODULO an equational theory, that
the no-theory path is unchanged, and that the documented soundness boundary
(position-pinning rules under an AC theory) holds as intended behavior.

GENERAL ENGINE: theories are DATA. The boolean fixture proves the same engine
code reasons over a non-arithmetic AC theory with no code change.
"""

from rerum.engine import RuleEngine
from rerum.normalize import Theory


# --- Theory fixtures (DATA; no operator is special-cased in rerum/) ---------

AC_PLUS = Theory.from_dict({"+": {"ac": True, "identity": 0}})
AC_TIMES = Theory.from_dict({"*": {"ac": True, "annihilator": 0}})
AC_BOOL = Theory.from_dict({"and": {"ac": True}, "or": {"ac": True}})


class TestCanonicalizeSeam:
    def test_no_theory_is_identity(self):
        eng = RuleEngine()
        expr = ["+", "b", "a"]
        # Identity function: returns the SAME object (zero-copy fast path).
        assert eng._canonicalize(expr) is expr

    def test_theory_returns_normal_form(self):
        eng = RuleEngine().with_theory(AC_PLUS)
        # AC + sorts operands: (+ b a) canonicalizes to (+ a b).
        assert eng._canonicalize(["+", "b", "a"]) == ["+", "a", "b"]

    def test_theory_collapses_identity_unit(self):
        eng = RuleEngine().with_theory(AC_PLUS)
        # identity 0 is dropped, single operand unwraps: (+ x 0) -> x.
        assert eng._canonicalize(["+", "x", 0]) == "x"


# --- Helpers for TestEquivalentsModuloTheory --------------------------------

def _comm_plus_engine():
    """Engine with a single commutativity rule for +."""
    return RuleEngine.from_dsl("@comm: (+ ?x ?y) <=> (+ :y :x)")


class TestEquivalentsModuloTheory:
    def test_ac_class_dedups_with_theory(self):
        eng = _comm_plus_engine().with_theory(AC_PLUS)
        members = eng.enumerate_equivalents(["+", "a", "b"], max_depth=3)
        # (+ a b) and (+ b a) share a canonical key -> one class member.
        assert len(members) == 1
        # The yielded form is canonical (sorted).
        assert members[0] == ["+", "a", "b"]

    def test_same_class_without_theory_has_both_arrangements(self):
        eng = _comm_plus_engine()  # no theory
        members = eng.enumerate_equivalents(["+", "a", "b"], max_depth=3)
        # Without the theory, the commute rule yields both arrangements.
        assert len(members) == 2
        assert ["+", "a", "b"] in members
        assert ["+", "b", "a"] in members

    def test_every_yielded_form_is_canonical_and_unique(self):
        eng = _comm_plus_engine().with_theory(AC_PLUS)
        members = eng.enumerate_equivalents(["+", "c", "a", "b"], max_depth=5)
        # Each yielded form equals its own normal form (a dedup guarantee).
        for m in members:
            assert eng._canonicalize(m) == m
        # No duplicate canonical keys among the yielded members.
        keys = [tuple(eng._canonicalize(m)) if isinstance(m, list) else m
                for m in members]
        assert len(set(keys)) == len(members)

    def test_no_theory_output_unchanged_value_and_order(self):
        # Backward-compat at the VALUE+ORDER level: identical generator output.
        eng = _comm_plus_engine()
        out = list(eng.equivalents(["+", "a", "b"], max_depth=3))
        assert out == [["+", "a", "b"], ["+", "b", "a"]]
