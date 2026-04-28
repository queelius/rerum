"""Tests for the hooks module: Resolution, HookContext, error types,
the _HookRegistry, and per-event firing."""

import pytest


class TestResolutionValidation:
    def test_value_only_is_valid(self):
        from rerum.hooks import Resolution
        r = Resolution(value=["+", "a", "b"])
        assert r.value == ["+", "a", "b"]
        assert r.rules is None

    def test_rules_only_is_valid(self):
        from rerum.hooks import Resolution
        r = Resolution(rules=[(["+", ["?", "x"], 0], [":", "x"])])
        assert r.rules is not None
        assert r.value is None

    def test_value_and_rules_is_invalid(self):
        from rerum.hooks import Resolution, ResolutionError
        with pytest.raises(ResolutionError, match="exactly one"):
            Resolution(value=42, rules=[(["+", ["?", "x"], 0], [":", "x"])])

    def test_value_and_fold_funcs_is_invalid(self):
        from rerum.hooks import Resolution, ResolutionError
        with pytest.raises(ResolutionError, match="exactly one"):
            Resolution(value=42, fold_funcs={"+": lambda a: a})

    def test_abort_alone_is_valid(self):
        from rerum.hooks import Resolution
        r = Resolution(abort=True)
        assert r.abort is True

    def test_metadata_alone_is_invalid(self):
        # metadata is orthogonal but a Resolution must carry at least one
        # primary action (value/rules/fold_funcs/allow_more/abort).
        from rerum.hooks import Resolution, ResolutionError
        with pytest.raises(ResolutionError, match="empty"):
            Resolution(metadata={"foo": "bar"})

    def test_abort_with_metadata_is_valid(self):
        from rerum.hooks import Resolution
        r = Resolution(abort=True, metadata={"reason": "stuck"})
        assert r.abort is True
        assert r.metadata == {"reason": "stuck"}

    def test_resolution_is_frozen(self):
        from rerum.hooks import Resolution
        r = Resolution(value=42)
        with pytest.raises(AttributeError):
            r.value = 99

    def test_value_zero_is_valid(self):
        from rerum.hooks import Resolution
        r = Resolution(value=0)
        assert r.value == 0

    def test_value_false_is_valid(self):
        from rerum.hooks import Resolution
        r = Resolution(value=False)
        assert r.value is False

    def test_value_empty_list_is_valid(self):
        from rerum.hooks import Resolution
        r = Resolution(value=[])
        assert r.value == []

    def test_abort_with_value_is_valid(self):
        # Spec: abort is orthogonal, can co-exist with primary actions.
        from rerum.hooks import Resolution
        r = Resolution(abort=True, value="fallback")
        assert r.abort is True
        assert r.value == "fallback"

    def test_empty_rules_is_invalid(self):
        from rerum.hooks import Resolution, ResolutionError
        with pytest.raises(ResolutionError, match="non-empty"):
            Resolution(rules=[])

    def test_empty_fold_funcs_is_invalid(self):
        from rerum.hooks import Resolution, ResolutionError
        with pytest.raises(ResolutionError, match="non-empty"):
            Resolution(fold_funcs={})

    def test_allow_more_false_is_invalid(self):
        from rerum.hooks import Resolution, ResolutionError
        with pytest.raises(ResolutionError, match="allow_more=False"):
            Resolution(allow_more=False)


class TestHookContext:
    def test_context_exposes_engine(self):
        from rerum.hooks import HookContext

        sentinel_engine = object()
        ctx = HookContext(
            engine=sentinel_engine,
            expr_path=[],
            depth=0,
            step_count=0,
            event_name="rule_applied",
        )
        assert ctx.engine is sentinel_engine
        assert ctx.event_name == "rule_applied"
        assert ctx.depth == 0
        assert ctx.step_count == 0

    def test_cancel_sets_flag(self):
        from rerum.hooks import HookContext

        ctx = HookContext(
            engine=None,
            expr_path=[],
            depth=0,
            step_count=0,
            event_name="rule_applied",
        )
        assert ctx.cancelled is False
        ctx.cancel()
        assert ctx.cancelled is True

    def test_expr_path_is_tuple_not_list(self):
        from rerum.hooks import HookContext

        path = [["+", "a", "b"], ["a"]]
        ctx = HookContext(
            engine=None,
            expr_path=path,
            depth=2,
            step_count=5,
            event_name="no_match",
        )
        # The container is a tuple, so hooks can't reassign or resize the path.
        assert isinstance(ctx.expr_path, tuple)
        assert ctx.expr_path == tuple(path)

    def test_expr_path_container_cannot_be_modified(self):
        from rerum.hooks import HookContext

        path = [["+", "a", "b"], ["a"]]
        ctx = HookContext(
            engine=None,
            expr_path=path,
            depth=2,
            step_count=5,
            event_name="no_match",
        )
        with pytest.raises(TypeError):
            ctx.expr_path[0] = "replaced"


class TestHookRegistryObservers:
    """Observer events: broadcast, all run in registration order."""

    def test_register_and_run_observer(self):
        from rerum.hooks import _HookRegistry, HookContext

        reg = _HookRegistry()
        log = []
        reg.register("rule_applied", "observer", lambda payload, ctx: log.append(("a", payload)))
        reg.register("rule_applied", "observer", lambda payload, ctx: log.append(("b", payload)))

        ctx = HookContext(None, [], 0, 0, "rule_applied")
        reg.run_observers("rule_applied", "STEP", ctx)
        assert log == [("a", "STEP"), ("b", "STEP")]

    def test_run_observers_with_no_hooks_does_nothing(self):
        from rerum.hooks import _HookRegistry, HookContext
        reg = _HookRegistry()
        ctx = HookContext(None, [], 0, 0, "rule_applied")
        reg.run_observers("rule_applied", "STEP", ctx)  # should not raise

    def test_observer_exceptions_wrapped_in_hook_error(self):
        from rerum.hooks import _HookRegistry, HookContext, HookError

        def bad(payload, ctx):
            raise ValueError("oops")

        reg = _HookRegistry()
        reg.register("rule_applied", "observer", bad)
        ctx = HookContext(None, [], 0, 0, "rule_applied")
        with pytest.raises(HookError) as exc_info:
            reg.run_observers("rule_applied", "STEP", ctx)
        assert exc_info.value.event == "rule_applied"
        assert isinstance(exc_info.value.cause, ValueError)


class TestHookRegistryResolvers:
    """Resolver events: chain of responsibility, first non-None wins."""

    def test_first_resolver_returning_resolution_wins(self):
        from rerum.hooks import _HookRegistry, HookContext, Resolution

        reg = _HookRegistry()
        reg.register("no_match", "resolver", lambda expr, ctx: None)
        reg.register("no_match", "resolver", lambda expr, ctx: Resolution(value="HIT"))
        reg.register("no_match", "resolver", lambda expr, ctx: Resolution(value="LOSE"))

        ctx = HookContext(None, [], 0, 0, "no_match")
        result = reg.run_resolvers("no_match", ["foo"], ctx)
        assert result is not None
        assert result.value == "HIT"

    def test_all_none_returns_none(self):
        from rerum.hooks import _HookRegistry, HookContext

        reg = _HookRegistry()
        reg.register("no_match", "resolver", lambda expr, ctx: None)
        reg.register("no_match", "resolver", lambda expr, ctx: None)

        ctx = HookContext(None, [], 0, 0, "no_match")
        assert reg.run_resolvers("no_match", ["foo"], ctx) is None

    def test_no_resolvers_returns_none(self):
        from rerum.hooks import _HookRegistry, HookContext
        reg = _HookRegistry()
        ctx = HookContext(None, [], 0, 0, "no_match")
        assert reg.run_resolvers("no_match", ["foo"], ctx) is None


class TestHookRegistryDecisions:
    """Decision events: AND-gate, every hook must return True."""

    def test_all_true_passes(self):
        from rerum.hooks import _HookRegistry, HookContext

        reg = _HookRegistry()
        reg.register("should_fire", "decision", lambda *args: True)
        reg.register("should_fire", "decision", lambda *args: True)

        ctx = HookContext(None, [], 0, 0, "should_fire")
        assert reg.run_decisions("should_fire", "rule", "expr", "bindings", ctx) is True

    def test_any_false_short_circuits(self):
        from rerum.hooks import _HookRegistry, HookContext

        called = []

        def first(*args):
            called.append("first")
            return False

        def second(*args):
            called.append("second")
            return True

        reg = _HookRegistry()
        reg.register("should_fire", "decision", first)
        reg.register("should_fire", "decision", second)

        ctx = HookContext(None, [], 0, 0, "should_fire")
        assert reg.run_decisions("should_fire", "rule", "expr", "bindings", ctx) is False
        assert called == ["first"]  # short-circuited

    def test_no_decisions_passes(self):
        from rerum.hooks import _HookRegistry, HookContext
        reg = _HookRegistry()
        ctx = HookContext(None, [], 0, 0, "should_fire")
        assert reg.run_decisions("should_fire", ctx) is True


class TestHookRegistryUnregister:
    def test_unregister_removes_hook(self):
        from rerum.hooks import _HookRegistry, HookContext

        reg = _HookRegistry()
        log = []
        h1 = lambda payload, ctx: log.append("h1")
        h2 = lambda payload, ctx: log.append("h2")
        reg.register("rule_applied", "observer", h1)
        reg.register("rule_applied", "observer", h2)

        reg.unregister("rule_applied", h1)

        ctx = HookContext(None, [], 0, 0, "rule_applied")
        reg.run_observers("rule_applied", "STEP", ctx)
        assert log == ["h2"]

    def test_clear_event(self):
        from rerum.hooks import _HookRegistry, HookContext
        reg = _HookRegistry()
        reg.register("rule_applied", "observer", lambda *a: None)
        reg.clear("rule_applied")

        ctx = HookContext(None, [], 0, 0, "rule_applied")
        reg.run_observers("rule_applied", "STEP", ctx)  # no-op, no exception

    def test_clear_all(self):
        from rerum.hooks import _HookRegistry
        reg = _HookRegistry()
        reg.register("rule_applied", "observer", lambda *a: None)
        reg.register("no_match", "resolver", lambda *a: None)
        reg.clear()
        # both empty
        assert reg.count("rule_applied") == 0
        assert reg.count("no_match") == 0
