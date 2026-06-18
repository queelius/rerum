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
