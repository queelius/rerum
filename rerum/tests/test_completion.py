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
