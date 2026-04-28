"""Integration tests: hooks attached to a RuleEngine."""

import pytest
from rerum import RuleEngine
from rerum.hooks import Resolution, HookContext


class TestEngineHookRegistration:
    def test_engine_starts_with_no_hooks(self):
        engine = RuleEngine()
        assert engine._hooks.count("rule_applied") == 0
        assert engine._hooks.count("no_match") == 0

    def test_on_rule_applied_registers(self):
        engine = RuleEngine()
        log = []

        @engine.on_rule_applied
        def observer(step, ctx):
            log.append(step)

        assert engine._hooks.count("rule_applied") == 1

    def test_on_rule_applied_can_be_used_as_method(self):
        engine = RuleEngine()
        cb = lambda step, ctx: None
        engine.on_rule_applied(cb)
        assert engine._hooks.count("rule_applied") == 1

    def test_off_rule_applied_unregisters(self):
        engine = RuleEngine()
        cb = lambda step, ctx: None
        engine.on_rule_applied(cb)
        assert engine._hooks.count("rule_applied") == 1
        engine.off_rule_applied(cb)
        assert engine._hooks.count("rule_applied") == 0

    def test_clear_hooks_for_one_event(self):
        engine = RuleEngine()
        engine.on_rule_applied(lambda step, ctx: None)
        engine.on_no_match(lambda expr, ctx: None)
        engine.clear_hooks("rule_applied")
        assert engine._hooks.count("rule_applied") == 0
        assert engine._hooks.count("no_match") == 1

    def test_clear_hooks_all(self):
        engine = RuleEngine()
        engine.on_rule_applied(lambda step, ctx: None)
        engine.on_no_match(lambda expr, ctx: None)
        engine.clear_hooks()
        assert engine._hooks.count("rule_applied") == 0
        assert engine._hooks.count("no_match") == 0

    def test_all_eight_events_registerable(self):
        engine = RuleEngine()
        cb = lambda *a: None
        for event in (
            "rule_applied", "fixpoint", "no_match", "undefined_op",
            "fold_error", "max_depth", "cycle", "should_fire",
        ):
            getattr(engine, f"on_{event}")(cb)
            assert engine._hooks.count(event) == 1
            getattr(engine, f"off_{event}")(cb)
            assert engine._hooks.count(event) == 0

    def test_copy_preserves_hooks(self):
        engine = RuleEngine()
        cb = lambda *a: None
        engine.on_rule_applied(cb)
        engine.on_no_match(cb)

        copied = engine.copy()
        assert copied._hooks.count("rule_applied") == 1
        assert copied._hooks.count("no_match") == 1

        # But hooks dicts are independent (modifying copy doesn't affect original).
        copied.off_rule_applied(cb)
        assert engine._hooks.count("rule_applied") == 1
        assert copied._hooks.count("rule_applied") == 0

    def test_union_preserves_hooks_from_left(self):
        e1 = RuleEngine()
        e2 = RuleEngine()
        cb = lambda *a: None
        e1.on_rule_applied(cb)

        union = e1 | e2
        assert union._hooks.count("rule_applied") == 1

    def test_on_event_decorator_returns_callback(self):
        engine = RuleEngine()

        @engine.on_rule_applied
        def cb(step, ctx):
            pass

        # Decorator must return the callback unchanged.
        assert cb.__name__ == "cb"
        # And the engine actually registered it.
        assert engine._hooks.count("rule_applied") == 1


class TestPublicReexports:
    def test_resolution_exported(self):
        from rerum import Resolution
        r = Resolution(value=42)
        assert r.value == 42

    def test_hookcontext_exported(self):
        from rerum import HookContext
        # Verify it's actually the right class with expected attributes.
        assert hasattr(HookContext, "engine")  # via __slots__
        assert hasattr(HookContext, "cancel")  # method
        assert hasattr(HookContext, "expr_path")  # property

    def test_error_types_exported(self):
        from rerum import HookError, ResolutionError, ResolverLoopError, HooksError
        # All hook error types share the HooksError base.
        assert issubclass(HookError, HooksError)
        assert issubclass(ResolutionError, HooksError)
        assert issubclass(ResolverLoopError, HooksError)
        # All ultimately Exception.
        for cls in (HookError, ResolutionError, ResolverLoopError, HooksError):
            assert issubclass(cls, Exception)

    def test_hooks_error_base_exported(self):
        from rerum import HooksError, HookError, ResolutionError, ResolverLoopError
        assert issubclass(HookError, HooksError)
        assert issubclass(ResolutionError, HooksError)
        assert issubclass(ResolverLoopError, HooksError)


class TestRuleAppliedEventSlowPath:
    """Tests for rule_applied in _simplify_exhaustive directly.

    These tests call the private method because the public ``simplify()``
    fast path bypasses engine-level hooks. Task 6 will wire the fast path
    and other strategies; once that lands, these tests should be migrated
    to use the public API and this class can be deleted.
    """

    def test_rule_applied_fires_on_each_step(self):
        engine = RuleEngine.from_dsl("""
            @add-zero: (+ ?x 0) => :x
            @mul-one: (* ?x 1) => :x
        """)
        steps = []

        @engine.on_rule_applied
        def observer(step, ctx):
            steps.append(step.metadata.name)

        engine._simplify_exhaustive(["+", ["*", "a", 1], 0], 100)
        assert "mul-one" in steps
        assert "add-zero" in steps

    def test_observers_broadcast_to_all(self):
        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")
        log_a = []
        log_b = []
        engine.on_rule_applied(lambda step, ctx: log_a.append(step.metadata.name))
        engine.on_rule_applied(lambda step, ctx: log_b.append(step.metadata.name))

        engine._simplify_exhaustive(["+", "x", 0], 100)
        assert log_a == ["add-zero"]
        assert log_b == ["add-zero"]

    def test_ctx_passed_to_observer(self):
        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")
        contexts = []

        @engine.on_rule_applied
        def observer(step, ctx):
            contexts.append(ctx)

        engine._simplify_exhaustive(["+", "x", 0], 100)
        assert len(contexts) == 1
        assert contexts[0].engine is engine
        assert contexts[0].event_name == "rule_applied"

    def test_observer_can_cancel_via_ctx(self):
        """Observer calling ctx.cancel() halts the rewrite immediately."""
        engine = RuleEngine.from_dsl("""
            @r1: (a ?x) => (b :x)
            @r2: (b ?x) => (c :x)
        """)

        @engine.on_rule_applied
        def cancelling(step, ctx):
            if step.metadata.name == "r1":
                ctx.cancel()

        # The first rule fires, then cancel triggers; r2 should not fire.
        result = engine._simplify_exhaustive(["a", "x"], 100)
        assert result == ["b", "x"]  # r1 applied; r2 blocked by cancel.

    def test_observer_receives_step_count(self):
        engine = RuleEngine.from_dsl("""
            @r1: (a ?x) => (b :x)
            @r2: (b ?x) => (c :x)
        """)
        counts = []

        @engine.on_rule_applied
        def observer(step, ctx):
            counts.append(ctx.step_count)

        engine._simplify_exhaustive(["a", "x"], 100)
        # First rule firing has step_count=1, second has step_count=2.
        assert counts == [1, 2]


class TestRuleAppliedAcrossStrategies:
    def test_bottomup_fires_rule_applied(self):
        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")
        steps = []
        engine.on_rule_applied(lambda step, ctx: steps.append(step.metadata.name))
        engine.simplify(["+", "x", 0], strategy="bottomup")
        assert steps == ["add-zero"]

    def test_topdown_fires_rule_applied(self):
        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")
        steps = []
        engine.on_rule_applied(lambda step, ctx: steps.append(step.metadata.name))
        engine.simplify(["+", "x", 0], strategy="topdown")
        assert steps == ["add-zero"]

    def test_apply_once_fires_rule_applied(self):
        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")
        steps = []
        engine.on_rule_applied(lambda step, ctx: steps.append(step.metadata.name))
        engine.apply_once(["+", "x", 0])
        assert steps == ["add-zero"]

    def test_hooks_bypass_fast_path_and_fire_rule_applied(self):
        # No conditions, no groups: would normally hit fast path. With hooks
        # registered, the bailout makes simplify use _simplify_exhaustive,
        # which fires the event.
        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")
        steps = []
        engine.on_rule_applied(lambda step, ctx: steps.append(step.metadata.name))
        # Default strategy.
        engine.simplify(["+", "x", 0])
        assert steps == ["add-zero"]

    def test_cancel_propagates_through_bottomup_driver(self):
        engine = RuleEngine.from_dsl("""
            @r1: (a ?x) => (b :x)
            @r2: (b ?x) => (c :x)
        """)
        seen = []

        @engine.on_rule_applied
        def cancelling(step, ctx):
            seen.append(step.metadata.name)
            if step.metadata.name == "r1":
                ctx.cancel()

        result = engine.simplify(["a", "x"], strategy="bottomup")
        # Only r1 fires; cancellation breaks out of the bottomup driver loop.
        assert seen == ["r1"]
        assert result == ["b", "x"]

    def test_cancel_propagates_through_topdown_driver(self):
        engine = RuleEngine.from_dsl("""
            @r1: (a ?x) => (b :x)
            @r2: (b ?x) => (c :x)
        """)
        seen = []

        @engine.on_rule_applied
        def cancelling(step, ctx):
            seen.append(step.metadata.name)
            if step.metadata.name == "r1":
                ctx.cancel()

        result = engine.simplify(["a", "x"], strategy="topdown")
        assert seen == ["r1"]
        assert result == ["b", "x"]

    def test_apply_once_does_not_fire_for_noop_rewrite(self):
        # A rule whose RHS equals its LHS structurally is a no-op.
        engine = RuleEngine.from_dsl("@noop: (foo ?x) => (foo :x)")
        seen = []
        engine.on_rule_applied(lambda step, ctx: seen.append(step.metadata.name))
        engine.apply_once(["foo", "y"])
        assert seen == []


class TestShouldFireDecision:
    def test_false_vetoes_rule(self):
        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")

        @engine.on_should_fire
        def veto_all(rule, metadata, expr, bindings, ctx):
            return False

        result = engine.simplify(["+", "x", 0])
        assert result == ["+", "x", 0]

    def test_true_allows_rule(self):
        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")
        engine.on_should_fire(lambda rule, metadata, expr, bindings, ctx: True)
        result = engine.simplify(["+", "x", 0])
        assert result == "x"

    def test_decision_and_gate(self):
        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")
        engine.on_should_fire(lambda *a: True)
        engine.on_should_fire(lambda *a: False)
        result = engine.simplify(["+", "x", 0])
        assert result == ["+", "x", 0]

    def test_decision_receives_rule_and_metadata(self):
        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")
        seen_names = []

        @engine.on_should_fire
        def record(rule, metadata, expr, bindings, ctx):
            seen_names.append(metadata.name)
            return True

        engine.simplify(["+", "x", 0])
        assert "add-zero" in seen_names

    def test_should_fire_only_called_when_condition_passes(self):
        # The DSL condition is checked first; should_fire is only consulted
        # when the condition has already passed. This locks in the order so
        # a regression that moves the calls won't slip through.
        engine = (RuleEngine()
                  .with_prelude(__import__("rerum").FULL_PRELUDE)
                  .load_dsl("@only-positive: (foo ?x) => :x when (! > :x 0)"))

        called = []

        @engine.on_should_fire
        def record(rule, metadata, expr, bindings, ctx):
            called.append(True)
            return True

        # condition fails: x = -5 is not > 0. should_fire should NOT be called.
        engine.simplify(["foo", -5])
        assert called == []

        # condition passes: should_fire IS called.
        engine.simplify(["foo", 5])
        assert called == [True]

    def test_should_fire_can_cancel_via_ctx(self):
        engine = RuleEngine.from_dsl("""
            @r1: (a ?x) => (b :x)
            @r2: (b ?x) => (c :x)
        """)
        seen = []

        @engine.on_should_fire
        def cancelling(rule, metadata, expr, bindings, ctx):
            seen.append(metadata.name)
            if metadata.name == "r1":
                ctx.cancel()
                return False  # Veto this firing too.
            return True

        result = engine.simplify(["a", "x"])
        # r1 vetoed and triggers cancel; engine bails out before r2.
        assert seen == ["r1"]
        # The result is unchanged because r1 was vetoed (no rewrite happened).
        assert result == ["a", "x"]


class TestNoMatchResolverValuePath:
    def test_value_resolution_substitutes_expression(self):
        engine = RuleEngine()  # no rules

        @engine.on_no_match
        def resolver(expr, ctx):
            from rerum.hooks import Resolution
            if expr == ["foo", "bar"]:
                return Resolution(value="baz")
            return None

        result = engine.simplify(["foo", "bar"])
        assert result == "baz"

    def test_none_resolution_means_no_change(self):
        engine = RuleEngine()
        engine.on_no_match(lambda expr, ctx: None)
        result = engine.simplify(["foo", "bar"])
        assert result == ["foo", "bar"]

    def test_resolver_only_fires_at_compound_positions(self):
        # Atoms (constants, variables) are not "positions where rules could
        # have fired"; resolver should not be called for them.
        engine = RuleEngine.from_dsl("@id: (foo ?x) => :x")
        called = []
        engine.on_no_match(lambda expr, ctx: called.append(expr))

        engine.simplify(["foo", "bar"])
        # foo-rule fires on (foo bar) -> bar; bar is an atom, no_match
        # should not fire there. The compound (foo bar) had a matching rule
        # so no_match shouldn't fire either.
        assert called == []

    def test_no_match_fires_when_no_rule_matches(self):
        engine = RuleEngine.from_dsl("@id: (foo ?x) => :x")
        called_with = []

        @engine.on_no_match
        def resolver(expr, ctx):
            called_with.append(expr)
            return None

        # (bar baz) has no matching rule; no_match should fire.
        engine.simplify(["bar", "baz"])
        assert called_with == [["bar", "baz"]]

    def test_resolution_abort_returns_current_expression(self):
        engine = RuleEngine()

        @engine.on_no_match
        def resolver(expr, ctx):
            from rerum.hooks import Resolution
            return Resolution(abort=True)

        result = engine.simplify(["foo", "bar"])
        # Abort short-circuits; engine returns the unchanged expression.
        assert result == ["foo", "bar"]
