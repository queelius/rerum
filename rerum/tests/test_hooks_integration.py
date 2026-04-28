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


class TestPublicReexports:
    def test_resolution_exported(self):
        from rerum import Resolution
        r = Resolution(value=42)
        assert r.value == 42

    def test_hookcontext_exported(self):
        from rerum import HookContext
        assert HookContext is not None

    def test_error_types_exported(self):
        from rerum import HookError, ResolutionError, ResolverLoopError
        assert issubclass(HookError, Exception)
        assert issubclass(ResolutionError, Exception)
        assert issubclass(ResolverLoopError, Exception)
