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
