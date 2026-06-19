"""F4: termination via the lexicographic path order."""

from rerum import termination as tm
from rerum.engine import RuleEngine


class TestPrecedenceAndLPO:
    def test_prec_gt(self):
        assert tm._prec_gt("*", "+", ["*", "+"]) is True
        assert tm._prec_gt("+", "*", ["*", "+"]) is False
        assert tm._prec_gt("z", "+", ["*", "+"]) is False   # unlisted
        assert tm._prec_gt("+", "z", ["*", "+"]) is False   # unlisted
        assert tm._prec_gt("+", "+", ["+"]) is False        # not > itself

    def test_variable_cases(self):
        assert tm.lpo_greater(["?", "x"], ["f", "a"], []) is False
        assert tm.lpo_greater(["f", ["?", "x"]], ["?", "x"], []) is True
        assert tm.lpo_greater(["f", ["?", "x"]], ["?", "y"], []) is False

    def test_constant_cases(self):
        assert tm.lpo_greater("a", "b", ["a", "b"]) is True
        assert tm.lpo_greater("b", "a", ["a", "b"]) is False
        assert tm.lpo_greater(["f", "a"], "a", ["f", "a"]) is True   # subterm/head
        assert tm.lpo_greater("a", ["f", "a"], ["f", "a"]) is False

    def test_case1_subterm(self):
        assert tm.lpo_greater(["f", ["g", "a"]], ["g", "a"], []) is True

    def test_case2_precedence(self):
        prec = ["*", "+", "a", "b"]
        assert tm.lpo_greater(["*", "a", "b"], ["+", "a", "b"], prec) is True
        rev = ["+", "*", "a", "b"]
        assert tm.lpo_greater(["*", "a", "b"], ["+", "a", "b"], rev) is False

    def test_case3_lexicographic(self):
        assert tm.lpo_greater(["f", "b", "a"], ["f", "a", "a"], ["b", "a"]) is True

    def test_variadic_same_head_different_arity(self):
        # Case 3 is SKIPPED (arity mismatch); only case 1 may fire. The bigger
        # term dominates via the subterm case; neither call raises.
        assert tm.lpo_greater(["+", "a", "b", ["+", "a", "b"]],
                              ["+", "a", "b"], []) is True   # t is a subterm
        assert tm.lpo_greater(["+", "a", "b"],
                              ["+", "a", "b", "c"], []) is False

    def test_all_guard_blocks_case2(self):
        # (* a) vs (+ (* a)) with * > +: case 2 needs (* a) >_lpo every arg of
        # (+ ...), but that arg IS (* a), and (* a) is not >_lpo itself, so the
        # all-guard blocks case 2. The order is False -- a term never dominates
        # a context that contains it (well-foundedness).
        assert tm.lpo_greater(["*", "a"], ["+", ["*", "a"]], ["*", "+"]) is False

    def test_asymmetry(self):
        # s >_lpo t implies not (t >_lpo s), and irreflexivity on a compound.
        s = ["f", ["g", ["?", "x"]]]
        t = ["g", ["?", "x"]]
        prec = ["f", "g"]
        assert tm.lpo_greater(s, t, prec) is True
        assert tm.lpo_greater(t, s, prec) is False
        assert tm.lpo_greater(s, s, prec) is False  # irreflexive


class TestOrient:
    def test_associativity_orients_lr_precedence_independent(self):
        # Right-associativity decreases lexicographically on the shared + head,
        # so it orients regardless of precedence (use []).
        l = ["+", ["+", ["?", "x"], ["?", "y"]], ["?", "z"]]
        r = ["+", ["?", "x"], ["+", ["?", "y"], ["?", "z"]]]
        assert tm.orient(l, r, []) == "lr"

    def test_commutativity_orients_none(self):
        l = ["+", ["?", "x"], ["?", "y"]]
        r = ["+", ["?", "y"], ["?", "x"]]
        assert tm.orient(l, r, ["+"]) is None

    def test_lr_and_rl(self):
        big = ["f", ["g", ["?", "x"]]]
        small = ["g", ["?", "x"]]
        assert tm.orient(big, small, ["f", "g"]) == "lr"
        assert tm.orient(small, big, ["f", "g"]) == "rl"


class TestCheckTermination:
    def test_terminating_set(self):
        # f > g > h: each rule's LHS dominates its RHS.
        eng = RuleEngine.from_dsl("""
            @r1: (f (g ?x)) => (g (g :x))
            @r2: (g (h ?x)) => (h :x)
        """)
        report = tm.check_termination(eng, ["f", "g", "h"])
        assert report.terminating is True
        assert report.unoriented == []
        assert report.not_analyzed == []
        # Both rules are positively classified as oriented "lr".
        assert {name for name, _dir in report.oriented} == {"r1", "r2"}
        assert all(d == "lr" for _n, d in report.oriented)

    def test_commutativity_is_unoriented(self):
        eng = RuleEngine.from_dsl("@c: (+ ?x ?y) => (+ :y :x)")
        report = tm.check_termination(eng, ["+"])
        assert "c" in report.unoriented
        assert report.terminating is False

    def test_not_analyzed_blocks(self):
        eng = RuleEngine.from_dsl("@rest: (f ?x...) => (g :x...)")
        report = tm.check_termination(eng, ["f", "g"])
        assert report.not_analyzed == ["rest"]
        assert report.terminating is False

    def test_general_boolean(self):
        eng = RuleEngine.from_dsl("@dn: (not (not ?x)) => :x")
        report = tm.check_termination(eng, ["not"])
        assert report.terminating is True   # (not (not x)) >_lpo x (subterm)
