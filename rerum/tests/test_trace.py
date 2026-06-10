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



class TestRewriteStepInverse:
    """RewriteStep.inverse(): swap before/after, flip direction, keep path,
    null the forward-match bindings/guard."""

    def _meta(self, name="r"):
        from rerum.engine import RuleMetadata
        return RuleMetadata(name=name)

    def test_inverse_swaps_and_flips(self):
        meta = self._meta()
        step = RewriteStep(
            0, meta, ["foo", "a"], ["bar", "a"], rule_id="r",
            direction="fwd", bindings={"x": "a"}, path=[1], kind="rule",
            guard={"condition": ["?", "p"], "result": True},
            rationale="why")
        inv = step.inverse()
        assert inv.before == ["bar", "a"]      # was after
        assert inv.after == ["foo", "a"]       # was before
        assert inv.direction == "rev"          # flipped
        assert inv.path == [1]                 # unchanged
        assert inv.kind == "rule"              # unchanged
        assert inv.rule_id == "r"              # unchanged
        assert inv.metadata is meta            # same object
        assert inv.rationale == "why"          # unchanged
        assert inv.bindings is None            # cleared
        assert inv.guard is None               # cleared

    def test_inverse_none_direction_stays_none(self):
        step = RewriteStep(0, self._meta(), ["a"], ["b"], direction=None)
        assert step.inverse().direction is None

    def test_inverse_rev_becomes_fwd(self):
        step = RewriteStep(0, self._meta(), ["a"], ["b"], direction="rev")
        assert step.inverse().direction == "fwd"

    def test_inverse_is_structural_involution(self):
        step = RewriteStep(
            3, self._meta(), ["+", "a", 0], "a", rule_id="add-zero",
            direction="rev", path=[2], kind="normalize", rationale="why")
        twice = step.inverse().inverse()
        assert twice.before == step.before
        assert twice.after == step.after
        assert twice.direction == step.direction
        assert twice.path == step.path
        assert twice.kind == step.kind
        assert twice.rule_id == step.rule_id
        assert twice.rationale == step.rationale

    def test_inverse_path_is_a_copy(self):
        p = [1, 2]
        step = RewriteStep(0, self._meta(), ["a"], ["b"], path=p)
        inv = step.inverse()
        assert inv.path == [1, 2]
        assert inv.path is not p  # pure: not the same list object

    def test_inverse_preserves_kind_for_all_kinds(self):
        for kind in ("rule", "normalize", "fold", "initial"):
            before, after = ("x", "x") if kind == "initial" else (["a"], ["b"])
            step = RewriteStep(0, self._meta(), before, after, kind=kind)
            inv = step.inverse()
            assert inv.kind == kind
            assert inv.before == after and inv.after == before


class TestRewriteTraceInverse:
    """RewriteTrace.inverse(): swap initial/final, steps reversed + inverted.
    The load-bearing property is replay correctness on a NESTED redex."""

    def test_inverse_replays_final_to_initial_nested(self):
        eng = RuleEngine.from_dsl("@r: (foo ?x) => (bar :x)")
        result, trace = eng.simplify(["top", ["foo", "a"]], trace=True)
        assert result == ["top", ["bar", "a"]]

        inv = trace.inverse()
        assert inv.initial == ["top", ["bar", "a"]]  # was final
        assert inv.final == ["top", ["foo", "a"]]     # was initial

        seq = inv.to_global_sequence()
        assert seq[0]["before_root"] == ["top", ["bar", "a"]]
        assert seq[-1]["after_root"] == ["top", ["foo", "a"]]
        for k in range(len(seq) - 1):
            assert seq[k]["after_root"] == seq[k + 1]["before_root"]

    def test_inverse_multistep_chains(self):
        eng = RuleEngine.from_dsl(
            "@r: (foo ?x) => (bar :x)\n@s: (bar ?x) => (baz :x)")
        result, trace = eng.simplify(["wrap", ["foo", "a"]], trace=True)
        assert len(trace.steps) == 2

        inv = trace.inverse()
        seq = inv.to_global_sequence()
        assert seq[0]["before_root"] == result
        assert seq[-1]["after_root"] == ["wrap", ["foo", "a"]]
        for k in range(len(seq) - 1):
            assert seq[k]["after_root"] == seq[k + 1]["before_root"]

    def test_inverse_inverse_is_structural_identity(self):
        eng = RuleEngine.from_dsl("@r: (foo ?x) => (bar :x)")
        _result, trace = eng.simplify(["top", ["foo", "a"]], trace=True)
        twice = trace.inverse().inverse()
        assert twice.initial == trace.initial
        assert twice.final == trace.final
        assert len(twice.steps) == len(trace.steps)
        for a, b in zip(twice.steps, trace.steps):
            assert a.before == b.before
            assert a.after == b.after
            assert a.direction == b.direction
            assert a.path == b.path
            assert a.kind == b.kind

    def test_inverse_empty_trace(self):
        t = RewriteTrace()
        t.initial = "x"
        t.final = "x"
        inv = t.inverse()
        assert inv.initial == "x" and inv.final == "x"
        assert inv.steps == []
