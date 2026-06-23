"""F5: basic Knuth-Bendix completion."""

from rerum import completion as cmp
from rerum import confluence as cf
from rerum.engine import RuleEngine


def _v(n):
    return ["?", n]


class TestBridgeHelpers:
    def test_term_to_skeleton_variable(self):
        assert cmp._term_to_skeleton(["?", "x"]) == [":", "x"]

    def test_term_to_skeleton_inverse_of_instantiate(self):
        # _term_to_skeleton is the inverse of instantiate_skeleton(.., {}).
        t = ["+", ["?", "x"], ["g", ["?", "y"]], "0"]
        skel = cmp._term_to_skeleton(t)
        assert skel == ["+", [":", "x"], ["g", [":", "y"]], "0"]
        assert cf.instantiate_skeleton(skel, {}) == t

    def test_dedup_preserves_order_drops_structural_duplicates(self):
        a = (["a"], "b")
        c = (["c"], "d")
        a2 = (["a"], "b")  # equal by value, distinct identity
        assert cmp._dedup([a, c, a2]) == [(["a"], "b"), (["c"], "d")]


class TestComplete:
    def test_associativity_completes_to_one_rule(self):
        l = ["*", ["*", _v("x"), _v("y")], _v("z")]
        r = ["*", _v("x"), ["*", _v("y"), _v("z")]]
        result = cmp.complete([(l, r)], ["*"])
        assert result.status == "complete"
        assert result.iterations == 1
        assert len(result.rules) == 1
        assert result.rules[0] == (l, r)   # oriented to the right

    def test_add_a_rule_converges(self):
        eqs = [
            (["f", ["g", _v("x")]], "a"),
            (["g", ["g", _v("x")]], _v("x")),
        ]
        result = cmp.complete(eqs, ["f", "g", "a"])
        assert result.status == "complete"
        assert result.iterations == 2
        assert len(result.rules) == 3   # adds (f ?v) -> a
        # The derived rule sends f-of-anything to a.
        assert any(lhs[0] == "f" and len(lhs) == 2 and lhs[1][0] == "?"
                   and rhs == "a" for (lhs, rhs) in result.rules)

    def test_failed_on_unorientable_input(self):
        eqs = [(["+", _v("x"), _v("y")], ["+", _v("y"), _v("x")])]
        result = cmp.complete(eqs, ["+"])
        assert result.status == "failed"
        assert result.failed_equation is not None
        assert result.rules == []

    def test_max_iterations_returns_not_hangs(self):
        # The add-a-rule set needs 2 passes; cap at 1 -> max_iterations.
        eqs = [
            (["f", ["g", _v("x")]], "a"),
            (["g", ["g", _v("x")]], _v("x")),
        ]
        result = cmp.complete(eqs, ["f", "g", "a"], max_iterations=1)
        assert result.status == "max_iterations"
        assert result.iterations == 1

    def test_trivial_equation_filtered_not_failed(self):
        # l == r is dropped BEFORE orient, so it is not a spurious "failed".
        eqs = [(_v("x"), _v("x"))]
        result = cmp.complete(eqs, [])
        assert result.status == "complete"
        assert result.rules == []


class TestSelfValidationAndGenerality:
    def test_complete_result_is_confluent(self):
        # The capstone validates itself: a "complete" system is confluent under
        # the precedence, via F2+F4's check_confluence. Same max_steps.
        eqs = [
            (["f", ["g", _v("x")]], "a"),
            (["g", ["g", _v("x")]], _v("x")),
        ]
        prec = ["f", "g", "a"]
        result = cmp.complete(eqs, prec, max_steps=1000)
        report = cf.check_confluence(result.to_engine(), precedence=prec,
                                     max_steps=1000)
        assert report.confluent is True
        assert report.terminating is True

    def test_to_engine_reduces(self):
        eqs = [
            (["f", ["g", _v("x")]], "a"),
            (["g", ["g", _v("x")]], _v("x")),
        ]
        eng = cmp.complete(eqs, ["f", "g", "a"]).to_engine()
        assert eng.simplify(["f", ["g", "a"]]) == "a"
        assert eng.simplify(["g", ["g", "a"]]) == "a"

    def test_general_boolean(self):
        # Same code completes a non-arithmetic equation set.
        eqs = [(["not", ["not", _v("x")]], _v("x"))]
        result = cmp.complete(eqs, ["not"])
        assert result.status == "complete"
        assert len(result.rules) == 1

    def test_general_arithmetic(self):
        eqs = [(["+", _v("x"), "0"], _v("x"))]
        result = cmp.complete(eqs, ["+", "0"])
        assert result.status == "complete"
        assert len(result.rules) == 1


class TestEngineAndExports:
    def test_engine_complete_extracts_and_matches(self):
        # Engine rules become equations; engine.complete matches the pure call.
        eng = RuleEngine.from_dsl("""
            @r1: (f (g ?x)) => a
            @r2: (g (g ?x)) => :x
        """)
        report = eng.complete(["f", "g", "a"])
        assert report.status == "complete"
        assert len(report.rules) == 3

    def test_engine_complete_skips_non_analyzable(self):
        # A ?... rule is not analyzable -> excluded from the extracted equations.
        eng = RuleEngine.from_dsl("@rest: (f ?x...) => (g :x...)")
        report = eng.complete(["f", "g"])
        # No analyzable equations -> trivially complete with no rules.
        assert report.status == "complete"
        assert report.rules == []

    def test_public_reexports(self):
        import rerum
        for name in ("complete", "CompletionResult"):
            assert name in rerum.__all__
            assert hasattr(rerum, name)

    def test_import_smoke_no_cycle(self):
        import importlib
        importlib.import_module("rerum.completion")
        importlib.import_module("rerum.engine")


class TestCompletionNotAnalyzedInvariant:
    def test_internal_rules_are_always_analyzable(self):
        import rerum.completion as cmp
        from rerum.confluence import critical_pairs, DirectedRule
        from rerum.completion import _term_to_skeleton
        eqs = [(["f", ["g", ["?", "x"]]], "a"),
               (["g", ["g", ["?", "x"]]], ["?", "x"])]
        result = cmp.complete(eqs, ["f", "g", "a"])
        assert result.status == "complete"
        recs = [DirectedRule(name=str(i), pattern=l,
                             skeleton=_term_to_skeleton(r), condition=None)
                for i, (l, r) in enumerate(result.rules)]
        _pairs, na = critical_pairs(recs)
        assert na == []
