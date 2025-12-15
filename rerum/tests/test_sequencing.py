"""Tests for the >> sequencing operator."""

import pytest
from rerum import RuleEngine, SequencedEngine, E, ARITHMETIC_PRELUDE


class TestSequencedEngine:
    """Tests for SequencedEngine and >> operator."""

    def test_basic_sequencing(self):
        """>> creates a SequencedEngine."""
        phase1 = RuleEngine.from_dsl("@step1: (a) => (b)")
        phase2 = RuleEngine.from_dsl("@step2: (b) => (c)")

        sequenced = phase1 >> phase2
        assert isinstance(sequenced, SequencedEngine)

    def test_sequenced_applies_in_order(self):
        """SequencedEngine applies engines in sequence."""
        phase1 = RuleEngine.from_dsl("@step1: (a) => (b)")
        phase2 = RuleEngine.from_dsl("@step2: (b) => (c)")

        sequenced = phase1 >> phase2
        result = sequenced(E("(a)"))
        assert result == ["c"]

    def test_each_phase_to_fixpoint(self):
        """Each phase runs until its fixpoint."""
        # Phase 1: simplify nested additions
        phase1 = RuleEngine.from_dsl('''
            @flatten: (+ (+ ?a ?b) ?c) => (+ :a :b :c)
        ''')
        # Phase 2: fold constants
        phase2 = (RuleEngine()
            .with_prelude(ARITHMETIC_PRELUDE)
            .load_dsl("@fold: (+ ?a:const ?b:const ?rest...) => (+ (! + :a :b) :rest...)"))

        pipeline = phase1 >> phase2

        expr = E("(+ (+ 1 2) (+ 3 4))")
        # Phase 1: (+ (+ 1 2) (+ 3 4)) => (+ 1 2 (+ 3 4)) => (+ 1 2 3 4)
        # Phase 2: (+ 1 2 3 4) => (+ 3 3 4) => (+ 6 4) => 10
        result = pipeline(expr)
        assert result == 10

    def test_chained_sequencing(self):
        """Multiple >> operators chain correctly."""
        phase1 = RuleEngine.from_dsl("@step1: (a) => (b)")
        phase2 = RuleEngine.from_dsl("@step2: (b) => (c)")
        phase3 = RuleEngine.from_dsl("@step3: (c) => done")

        pipeline = phase1 >> phase2 >> phase3
        assert len(pipeline) == 3

        result = pipeline(E("(a)"))
        assert result == "done"

    def test_sequenced_with_strategy(self):
        """SequencedEngine passes kwargs to each engine."""
        phase1 = RuleEngine.from_dsl("@outer: (f ?x) => (g :x)")
        phase2 = RuleEngine.from_dsl("@inner: (a) => done")

        pipeline = phase1 >> phase2

        # With strategy="once", only one rule fires per phase
        result = pipeline(E("(f (a))"), strategy="once")
        # Phase 1: (f (a)) => (g (a))
        # Phase 2: (g (a)) - outer doesn't match, inner matches inside => (g done)
        assert result == ["g", "done"]

    def test_sequenced_repr(self):
        """SequencedEngine has a sensible repr."""
        phase1 = RuleEngine.from_dsl("@step1: (a) => (b)")
        phase2 = RuleEngine.from_dsl("@step2: (b) => (c)")

        pipeline = phase1 >> phase2
        assert "2 phases" in repr(pipeline)

    def test_sequenced_len(self):
        """len() returns number of phases."""
        phase1 = RuleEngine.from_dsl("@step1: (a) => (b)")
        phase2 = RuleEngine.from_dsl("@step2: (b) => (c)")
        phase3 = RuleEngine.from_dsl("@step3: (c) => (d)")

        assert len(phase1 >> phase2) == 2
        assert len(phase1 >> phase2 >> phase3) == 3

    def test_sequenced_iter(self):
        """Can iterate over phases."""
        phase1 = RuleEngine.from_dsl("@step1: (a) => (b)")
        phase2 = RuleEngine.from_dsl("@step2: (b) => (c)")

        pipeline = phase1 >> phase2
        phases = list(pipeline)
        assert len(phases) == 2
        assert phases[0] is phase1
        assert phases[1] is phase2


class TestSequencingVsUnion:
    """Tests comparing >> (sequence) with | (union)."""

    def test_sequence_vs_union_difference(self):
        """Sequence and union behave differently."""
        # Rules that could interfere
        expand = RuleEngine.from_dsl("@expand: (f ?x) => (g :x :x)")
        simplify = RuleEngine.from_dsl("@simplify: (g ?x ?x) => (h :x)")

        expr = E("(f a)")

        # Union: all rules in one engine - might not work as expected
        # depending on order
        union = expand | simplify
        union_result = union(expr)
        # expand fires first: (f a) => (g a a)
        # then simplify: (g a a) => (h a)
        assert union_result == ["h", "a"]

        # Sequence: same result in this case, but semantically clearer
        sequence = expand >> simplify
        seq_result = sequence(expr)
        assert seq_result == ["h", "a"]

    def test_sequence_isolation(self):
        """Sequence keeps phases isolated - no rule interference."""
        # Phase 1 has a rule that would loop in union
        phase1 = RuleEngine.from_dsl("@step: (process ?x) => (result :x)")
        phase2 = RuleEngine.from_dsl("@wrap: (result ?x) => (done :x)")

        # Sequence works correctly
        pipeline = phase1 >> phase2
        result = pipeline(E("(process value)"))
        assert result == ["done", "value"]


class TestRealWorldSequencing:
    """Real-world use cases for sequencing."""

    def test_expand_then_simplify(self):
        """Common pattern: expand, then simplify."""
        expand = RuleEngine.from_dsl('''
            @square: (square ?x) => (* :x :x)
            @cube: (cube ?x) => (* :x :x :x)
        ''')

        simplify = (RuleEngine()
            .with_prelude(ARITHMETIC_PRELUDE)
            .load_dsl('''
                @mul-one: (* ?x 1) => :x
                @mul-zero: (* ?x 0) => 0
                @fold: (* ?a:const ?b:const ?rest...) => (* (! * :a :b) :rest...)
            '''))

        normalize = expand >> simplify

        assert normalize(E("(square 3)")) == 9
        assert normalize(E("(cube 2)")) == 8
        assert normalize(E("(* (square 2) (cube 3))")) == 108

    def test_multi_phase_compiler(self):
        """Simulate a multi-phase compiler transformation."""
        # Phase 1: Desugar
        desugar = RuleEngine.from_dsl('''
            @let-to-lambda: (let ?name ?val ?body) => ((lambda :name :body) :val)
        ''')

        # Phase 2: Inline trivial lambdas
        inline = RuleEngine.from_dsl('''
            @beta: ((lambda ?name ?body) ?val) => (subst :body :name :val)
        ''')

        # Phase 3: Clean up subst markers
        cleanup = RuleEngine.from_dsl('''
            @subst-var: (subst ?name ?name ?val) => :val
        ''')

        compiler = desugar >> inline >> cleanup

        # (let x 5 x) => ((lambda x x) 5) => (subst x x 5) => 5
        result = compiler(E("(let x 5 x)"))
        assert result == 5
