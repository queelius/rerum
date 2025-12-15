"""Tests for trace formatting."""

import pytest
from rerum import RuleEngine, E, RewriteTrace, RewriteStep


class TestTraceFormatting:
    """Tests for trace format() method."""

    def setup_method(self):
        """Set up test engine."""
        self.engine = RuleEngine.from_dsl('''
            @add-zero: (+ ?x 0) => :x
            @mul-one: (* ?x 1) => :x
            @mul-zero: (* ?x 0) => 0
        ''')

    def test_format_verbose(self):
        """Verbose format shows full details."""
        result, trace = self.engine(E("(+ (* x 1) 0)"), trace=True)

        verbose = trace.format("verbose")
        assert "Initial:" in verbose
        assert "Final:" in verbose
        assert "add-zero" in verbose or "mul-one" in verbose

    def test_format_compact(self):
        """Compact format shows single line."""
        result, trace = self.engine(E("(+ (* x 1) 0)"), trace=True)

        compact = trace.format("compact")
        assert "--[" in compact
        assert "]-->" in compact
        # Should be a single line
        assert compact.count("\n") == 0

    def test_format_rules(self):
        """Rules format shows just rule names."""
        result, trace = self.engine(E("(+ (* x 1) 0)"), trace=True)

        rules = trace.format("rules")
        # Should show rule names joined by ->
        assert " -> " in rules or "(no rules applied)" in rules

    def test_format_chain(self):
        """Chain format shows step-by-step transformations."""
        result, trace = self.engine(E("(+ x 0)"), trace=True)

        chain = trace.format("chain")
        assert "(+ x 0)" in chain
        if trace.steps:
            assert "--(" in chain
            assert ")-->" in chain

    def test_format_empty_trace(self):
        """Empty trace formats correctly."""
        # Expression that doesn't match any rules
        result, trace = self.engine(E("(+ x y)"), trace=True)

        assert trace.format("rules") == "(no rules applied)"
        assert trace.format("compact").endswith("(+ x y)")
        assert trace.format("chain") == "(+ x y)"


class TestTraceIteration:
    """Tests for iterating over trace steps."""

    def test_iter_trace(self):
        """Can iterate over trace steps."""
        engine = RuleEngine.from_dsl('''
            @add-zero: (+ ?x 0) => :x
        ''')

        result, trace = engine(E("(+ (+ y 0) 0)"), trace=True)

        step_count = 0
        for step in trace:
            assert isinstance(step, RewriteStep)
            step_count += 1

        assert step_count == len(trace)

    def test_bool_nonempty(self):
        """Non-empty trace is truthy."""
        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")
        result, trace = engine(E("(+ x 0)"), trace=True)

        assert bool(trace) == True
        assert trace  # Direct truthiness test

    def test_bool_empty(self):
        """Empty trace is falsy."""
        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")
        result, trace = engine(E("(+ x y)"), trace=True)

        assert bool(trace) == False
        assert not trace


class TestTraceToDict:
    """Tests for trace serialization."""

    def test_to_dict_structure(self):
        """to_dict() returns correct structure."""
        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")
        result, trace = engine(E("(+ x 0)"), trace=True)

        d = trace.to_dict()
        assert "initial" in d
        assert "final" in d
        assert "steps" in d
        assert "step_count" in d
        assert isinstance(d["steps"], list)

    def test_step_to_dict(self):
        """Step to_dict() returns correct structure."""
        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")
        result, trace = engine(E("(+ x 0)"), trace=True)

        if trace.steps:
            step_dict = trace.steps[0].to_dict()
            assert "rule_index" in step_dict
            assert "rule_name" in step_dict
            assert "before" in step_dict
            assert "after" in step_dict

    def test_to_dict_serializable(self):
        """to_dict() output is JSON-serializable."""
        import json

        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")
        result, trace = engine(E("(+ (+ y 0) 0)"), trace=True)

        d = trace.to_dict()
        # Should not raise
        json_str = json.dumps(d)
        assert json_str is not None


class TestTraceStatistics:
    """Tests for trace statistics methods."""

    def test_rule_counts(self):
        """rule_counts() returns correct counts."""
        engine = RuleEngine.from_dsl('''
            @add-zero: (+ ?x 0) => :x
            @mul-one: (* ?x 1) => :x
        ''')

        # Expression that uses add-zero twice
        result, trace = engine(E("(+ (+ x 0) 0)"), trace=True)

        counts = trace.rule_counts()
        assert counts.get("add-zero", 0) >= 1

    def test_rules_applied(self):
        """rules_applied() returns list of rule names."""
        engine = RuleEngine.from_dsl('''
            @add-zero: (+ ?x 0) => :x
        ''')

        result, trace = engine(E("(+ (+ x 0) 0)"), trace=True)

        rules = trace.rules_applied()
        assert isinstance(rules, list)
        for rule in rules:
            assert isinstance(rule, str)

    def test_summary_with_steps(self):
        """summary() describes non-empty trace."""
        engine = RuleEngine.from_dsl('''
            @add-zero: (+ ?x 0) => :x
        ''')

        result, trace = engine(E("(+ (+ x 0) 0)"), trace=True)

        summary = trace.summary()
        assert "steps" in summary
        assert "rules" in summary

    def test_summary_empty(self):
        """summary() describes empty trace."""
        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")
        result, trace = engine(E("(+ x y)"), trace=True)

        summary = trace.summary()
        assert "No rewriting" in summary


class TestTraceRepr:
    """Tests for trace __repr__."""

    def test_repr_includes_initial(self):
        """Repr shows initial expression."""
        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")
        result, trace = engine(E("(+ x 0)"), trace=True)

        assert "Initial:" in repr(trace)
        assert "(+ x 0)" in repr(trace)

    def test_repr_includes_final(self):
        """Repr shows final expression."""
        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")
        result, trace = engine(E("(+ x 0)"), trace=True)

        assert "Final:" in repr(trace)
        assert "x" in repr(trace)

    def test_repr_includes_steps(self):
        """Repr shows step numbers."""
        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")
        result, trace = engine(E("(+ (+ x 0) 0)"), trace=True)

        r = repr(trace)
        if trace.steps:
            assert "1." in r


class TestStepRepr:
    """Tests for RewriteStep __repr__."""

    def test_step_repr_shows_rule_name(self):
        """Step repr shows rule name."""
        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")
        result, trace = engine(E("(+ x 0)"), trace=True)

        if trace.steps:
            step = trace.steps[0]
            assert "add-zero" in repr(step)

    def test_step_repr_shows_transformation(self):
        """Step repr shows before → after."""
        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")
        result, trace = engine(E("(+ x 0)"), trace=True)

        if trace.steps:
            step = trace.steps[0]
            assert "→" in repr(step)


class TestTraceWithComplexRewriting:
    """Tests for traces with complex rewriting scenarios."""

    def test_trace_multi_step(self):
        """Trace captures multiple steps."""
        engine = RuleEngine.from_dsl('''
            @expand: (square ?x) => (* :x :x)
            @add-zero: (+ ?x 0) => :x
        ''')

        result, trace = engine(E("(+ (square 3) 0)"), trace=True)

        # Should have at least one step
        assert len(trace) >= 1

    def test_trace_with_priority(self):
        """Trace works with prioritized rules."""
        engine = RuleEngine.from_dsl('''
            @low: (f ?x) => low
            @high[100]: (f ?x) => high
        ''')

        result, trace = engine(E("(f a)"), trace=True)

        assert result == "high"
        if trace.steps:
            assert trace.steps[0].metadata.name == "high"

    def test_trace_with_guards(self):
        """Trace works with guarded rules."""
        from rerum import FULL_PRELUDE

        engine = (RuleEngine()
            .with_prelude(FULL_PRELUDE)
            .load_dsl('''
                @pos: (abs ?x) => :x when (! > :x 0)
                @neg: (abs ?x) => (! - 0 :x) when (! < :x 0)
            '''))

        result, trace = engine(E("(abs -5)"), trace=True)

        assert result == 5
        if trace.steps:
            assert trace.steps[0].metadata.name == "neg"

