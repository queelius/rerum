"""F1: theory-normalized equational reasoning.

Verifies the five reasoning methods (equivalents, enumerate_equivalents,
prove_equal, are_equal, minimize) reason MODULO an equational theory, that
the no-theory path is unchanged, and that the documented soundness boundary
(position-pinning rules under an AC theory) holds as intended behavior.

GENERAL ENGINE: theories are DATA. The boolean fixture proves the same engine
code reasons over a non-arithmetic AC theory with no code change.
"""

from rerum.engine import RuleEngine, expr_size
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


class TestProveEqualModuloTheory:
    def test_commute_holds_instantly_with_theory(self):
        # No commute rule loaded: equality holds ONLY via the theory.
        eng = RuleEngine().with_theory(AC_PLUS)
        proof = eng.prove_equal(["+", "x", "y"], ["+", "y", "x"])
        assert proof is not None
        # Zero-step: the canonical keys match, so the quick check fires.
        assert proof.depth_a == 0 and proof.depth_b == 0

    def test_zero_step_proof_common_is_canonical(self):
        # The quick-check branch must report the CANONICAL common form, not a
        # raw input: (+ y x) and (+ x y) meet at the canonical (+ x y).
        eng = RuleEngine().with_theory(AC_PLUS)
        proof = eng.prove_equal(["+", "y", "x"], ["+", "x", "y"])
        assert proof is not None
        assert proof.common == ["+", "x", "y"]
        assert eng._canonicalize(proof.common) == proof.common

    def test_commute_not_provable_without_theory(self):
        eng = RuleEngine()  # no theory, no commute rule
        proof = eng.prove_equal(["+", "x", "y"], ["+", "y", "x"])
        assert proof is None

    def test_associativity_and_commutativity_modulo_theory(self):
        eng = RuleEngine().with_theory(AC_PLUS)
        proof = eng.prove_equal(["+", ["+", "a", "b"], "c"],
                                ["+", "a", ["+", "c", "b"]])
        assert proof is not None

    def test_proof_path_states_are_canonical_no_normalize_steps(self):
        # A real proof under a theory: every step.after is the canonical state,
        # and no step is a kind="normalize" micro-step. Distinct operands a, b
        # avoid the idempotent-collapse of (+ a a) (AC_PLUS has no "repeat"
        # rule, so it is a join-semilattice on repeated operands).
        eng = RuleEngine.from_dsl("@f: (f ?x ?y) => (+ :x :y)").with_theory(AC_PLUS)
        proof = eng.prove_equal(["f", "b", "a"], ["+", "a", "b"],
                                include_unidirectional=True, trace=True)
        assert proof is not None
        for step in (proof.path_a or []):
            assert step.kind != "normalize"
            if isinstance(step.after, list):
                assert eng._canonicalize(step.after) == step.after


class TestInheritedAndUnits:
    def test_are_equal_true_for_ac_variants(self):
        eng = RuleEngine().with_theory(AC_PLUS)
        assert eng.are_equal(["+", "x", "y"], ["+", "y", "x"]) is True

    def test_are_equal_false_without_theory(self):
        eng = RuleEngine()  # no theory, no commute rule
        assert eng.are_equal(["+", "x", "y"], ["+", "y", "x"]) is False

    def test_identity_unit_collapses_into_class(self):
        eng = RuleEngine().with_theory(AC_PLUS)
        # (+ x 0) and x are the same modulo the identity unit.
        assert eng.are_equal(["+", "x", 0], "x") is True

    def test_annihilator_unit_collapses_into_class(self):
        eng = RuleEngine().with_theory(AC_TIMES)
        # (* x 0) and 0 are the same modulo the annihilator unit.
        assert eng.are_equal(["*", "x", 0], 0) is True

    def test_domain_free_boolean_theory(self):
        # Same engine code, a non-arithmetic AC theory: (and p q) == (and q p).
        eng = RuleEngine().with_theory(AC_BOOL)
        assert eng.are_equal(["and", "p", "q"], ["and", "q", "p"]) is True
        assert eng.are_equal(["or", "p", "q"], ["or", "q", "p"]) is True


class TestMinimizeModuloTheory:
    def test_returns_canonical_representative(self):
        eng = RuleEngine.from_dsl("@comm: (+ ?x ?y) <=> (+ :y :x)")
        eng.with_theory(AC_PLUS)
        result = eng.minimize(["+", "b", "a"], metric="size")
        # The result is canonical (sorted), not the raw (+ b a).
        assert eng._canonicalize(result.expr) == result.expr
        # The result is canonical-consistent: original is the canonical input
        # and original_cost is the cost of THAT, not of the raw input.
        assert result.original == eng._canonicalize(result.original)
        assert result.original_cost == expr_size(result.original)

    def test_no_spurious_improvement_on_normalization_only(self):
        # Input (+ a 0) has raw size 3 but canonical size 1 (a). Without the
        # canonical seed, minimize would measure improvement against the RAW
        # baseline and report ~0.667 -- counting normalization as a win. With
        # the seed, the baseline is the canonical form, so the ratio is 0.0.
        eng = RuleEngine().with_theory(AC_PLUS)
        result = eng.minimize(["+", "a", 0], metric="size")
        assert result.improvement_ratio == 0.0
        assert result.expr == "a"

    def test_mixed_directional_rules_under_theory(self):
        # A => simplification rule that genuinely fires under the theory:
        # @unwrap reduces (id a) to a, then the AC sort canonicalizes the sum.
        # The result is canonical and the derivation is well-formed or None.
        eng = RuleEngine.from_dsl("@unwrap: (id ?x) => :x")
        eng.with_theory(AC_PLUS)
        result = eng.minimize(["+", ["id", "a"], "b"], metric="size",
                              include_unidirectional=True)
        assert eng._canonicalize(result.expr) == result.expr
        assert result.cost <= expr_size(["+", ["id", "a"], "b"])
        # Derivation contract: present-and-wellformed or None, never malformed.
        assert result.derivation is None or result.derivation.final == result.expr


class TestSoundnessBoundary:
    def test_position_pinning_rule_is_not_reached_under_ac_theory(self):
        # DOCUMENTED INCOMPLETENESS (spec "Soundness boundary"): a distribute
        # rule pins the (+ ...) factor as the FIRST operand of *. Under an AC *
        # theory, canonical_sort moves it, so the syntactic matcher never sees
        # the arrangement and the distributed form is NOT reached. This is
        # intended behavior until F3 (AC-matching). Pinned here so a future
        # reader does not mistake it for completeness.
        eng = RuleEngine.from_dsl(
            "@distrib: (* (+ ?a ?b) ?c) => (+ (* :a :c) (* :b :c))"
        )
        eng.with_theory(Theory.from_dict({"*": {"ac": True}}))
        target = ["+", ["*", "a", "c"], ["*", "b", "c"]]
        proof = eng.prove_equal(["*", ["+", "a", "b"], "c"], target,
                                include_unidirectional=True)
        assert proof is None  # boundary: F3 is expected to make this succeed

    def test_position_pinning_rule_DOES_fire_without_theory(self):
        # Control: the same rule reaches the distributed form with no theory.
        eng = RuleEngine.from_dsl(
            "@distrib: (* (+ ?a ?b) ?c) => (+ (* :a :c) (* :b :c))"
        )
        target = ["+", ["*", "a", "c"], ["*", "b", "c"]]
        proof = eng.prove_equal(["*", ["+", "a", "b"], "c"], target,
                                include_unidirectional=True)
        assert proof is not None


class TestBackwardCompatValueLevel:
    def test_equivalents_identical_with_no_theory(self):
        eng = RuleEngine.from_dsl("@comm: (+ ?x ?y) <=> (+ :y :x)")
        out = list(eng.equivalents(["+", "a", "b"], max_depth=3))
        # Exact value AND order, byte-for-byte with pre-F1 behavior.
        assert out == [["+", "a", "b"], ["+", "b", "a"]]

    def test_prove_equal_path_identical_with_no_theory(self):
        eng = RuleEngine.from_dsl("@comm: (+ ?x ?y) <=> (+ :y :x)")
        proof = eng.prove_equal(["+", "a", "b"], ["+", "b", "a"],
                                include_unidirectional=True, trace=True)
        assert proof is not None
        assert proof.common == ["+", "b", "a"]
