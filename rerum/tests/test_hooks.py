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

    def test_expr_path_is_immutable_view(self):
        from rerum.hooks import HookContext

        path = [["+", "a", "b"], ["a"]]
        ctx = HookContext(
            engine=None,
            expr_path=path,
            depth=2,
            step_count=5,
            event_name="no_match",
        )
        # ctx.expr_path returns a tuple (immutable) so hooks can't mutate the
        # engine's internal stack.
        assert isinstance(ctx.expr_path, tuple)
        assert ctx.expr_path == tuple(path)
