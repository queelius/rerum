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

    def test_resolver_does_not_fire_for_atom_children(self):
        """The atom guard in _fire_no_match prevents firing on leaf nodes
        (strings, numbers). The resolver should be called only at compound
        positions where rules could have applied."""
        engine = RuleEngine()  # No rules, every position will exhaust rules.
        seen = []

        @engine.on_no_match
        def resolver(expr, ctx):
            seen.append(expr)
            return None

        engine.simplify(["foo", "bar", 42])
        # Only the top-level compound expression. Atoms ("foo", "bar", 42)
        # are leaves and don't fire no_match.
        assert seen == [["foo", "bar", 42]]

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

    def test_resolver_can_cancel_via_ctx(self):
        engine = RuleEngine()

        @engine.on_no_match
        def resolver(expr, ctx):
            ctx.cancel()
            return None  # Decline the chain; cancellation propagates separately.

        result = engine.simplify(["foo", "bar"])
        # Cancellation halts the rewrite; the unchanged expression is returned.
        assert result == ["foo", "bar"]


class TestNoMatchResolverRulesPath:
    def test_rules_resolution_adds_rule_and_retries(self):
        from rerum.engine import parse_rule_line
        from rerum.hooks import Resolution

        engine = RuleEngine()

        @engine.on_no_match
        def resolver(expr, ctx):
            if expr == ["foo", "bar"]:
                pairs = parse_rule_line("@foo-id: (foo ?x) => :x")
                rules = [(meta, [pat, skel]) for meta, pat, skel in pairs]
                return Resolution(rules=rules, metadata={"provenance": "test"})
            return None

        result = engine.simplify(["foo", "bar"])
        assert result == "bar"

    def test_added_rule_persists_after_call(self):
        from rerum.engine import parse_rule_line
        from rerum.hooks import Resolution

        engine = RuleEngine()
        called = [0]

        @engine.on_no_match
        def resolver(expr, ctx):
            called[0] += 1
            if isinstance(expr, list) and expr and expr[0] == "foo":
                pairs = parse_rule_line("@foo-id: (foo ?x) => :x")
                rules = [(m, [p, s]) for m, p, s in pairs]
                return Resolution(rules=rules)
            return None

        engine.simplify(["foo", "a"])  # adds rule, resolver called
        first_call_count = called[0]
        engine.simplify(["foo", "b"])  # rule already there
        # Resolver should not fire again because foo-id matches now.
        assert called[0] == first_call_count

    def test_rule_provenance_metadata_in_extra(self):
        from rerum.engine import parse_rule_line
        from rerum.hooks import Resolution

        engine = RuleEngine()

        @engine.on_no_match
        def resolver(expr, ctx):
            if isinstance(expr, list) and expr and expr[0] == "foo":
                pairs = parse_rule_line("@foo-id: (foo ?x) => :x")
                rules = [(m, [p, s]) for m, p, s in pairs]
                return Resolution(rules=rules,
                                  metadata={"provenance": "llm-inferred",
                                            "model": "test-model"})
            return None

        engine.simplify(["foo", "a"])
        # Find the rule's metadata.
        rule_meta = engine._metadata[-1]
        assert rule_meta.extra.get("provenance") == "llm-inferred"
        assert rule_meta.extra.get("model") == "test-model"

    def test_added_rule_visible_in_subsequent_simplify(self):
        from rerum.engine import parse_rule_line
        from rerum.hooks import Resolution

        engine = RuleEngine()

        @engine.on_no_match
        def resolver(expr, ctx):
            if isinstance(expr, list) and expr and expr[0] == "foo":
                pairs = parse_rule_line("@foo-id: (foo ?x) => :x")
                rules = [(m, [p, s]) for m, p, s in pairs]
                return Resolution(rules=rules)
            return None

        engine.simplify(["foo", "a"])
        # Subsequent simplify uses the added rule directly.
        assert engine.simplify(["foo", "z"]) == "z"
        # The rule is now in the engine.
        assert "foo-id" in engine

    def test_resolver_returning_nonmatching_rules_terminates(self):
        from rerum.engine import parse_rule_line
        from rerum.hooks import Resolution

        engine = RuleEngine()
        calls = [0]

        @engine.on_no_match
        def resolver(expr, ctx):
            calls[0] += 1
            # Return rules that match a different expression entirely.
            pairs = parse_rule_line("@never: (zzz ?x) => :x")
            rules = [(m, [p, s]) for m, p, s in pairs]
            return Resolution(rules=rules)

        result = engine.simplify(["foo", "bar"], max_steps=100)
        # Expression unchanged because no rule matches (foo bar).
        assert result == ["foo", "bar"]
        # Resolver called at most twice: once for the first install (rules
        # added), once on retry where the rules don't match and aren't
        # re-installed (deduplication by name). Then the engine falls
        # through to default no-match behavior.
        assert calls[0] <= 2
        # Only one copy of @never in the engine, not max_steps copies.
        assert len(engine._rules) == 1

    def test_resolver_returning_anonymous_rules_dedups_via_progress_check(self):
        # Anonymous rules can't dedupe by name. The "added > 0" check still
        # caps work because each install genuinely adds rules. T14 will add
        # the retry cap as a final safeguard for this case.
        from rerum.engine import parse_rule_line
        from rerum.hooks import Resolution

        engine = RuleEngine()

        @engine.on_no_match
        def resolver(expr, ctx):
            # Anonymous rule (no @name).
            pairs = parse_rule_line("(zzz ?x) => :x")
            rules = [(m, [p, s]) for m, p, s in pairs]
            return Resolution(rules=rules)

        # With max_steps=5, the engine will accumulate up to 5 anonymous
        # rules in this case. T14 will fix this entirely with a retry cap.
        result = engine.simplify(["foo", "bar"], max_steps=5)
        assert result == ["foo", "bar"]
        # Bound is max_steps; T14 will tighten this.
        assert len(engine._rules) <= 5


class TestUndefinedOpResolver:
    def test_value_resolution_used_as_fold_result(self):
        from rerum.hooks import Resolution
        engine = RuleEngine.from_dsl("@compute: (foo ?x) => (! my-op :x)")

        @engine.on_undefined_op
        def resolver(op, args, ctx):
            if op == "my-op":
                return Resolution(value=args[0])  # identity
            return None

        result = engine.simplify(["foo", 42])
        assert result == 42

    def test_fold_funcs_resolution_installs_handler(self):
        from rerum.hooks import Resolution
        engine = RuleEngine.from_dsl("@compute: (square ?x) => (! my-square :x)")
        call_count = [0]

        @engine.on_undefined_op
        def resolver(op, args, ctx):
            if op == "my-square":
                call_count[0] += 1
                return Resolution(fold_funcs={"my-square": lambda xs: xs[0] * xs[0]})
            return None

        # First call installs handler.
        result1 = engine.simplify(["square", 3])
        assert result1 == 9
        assert call_count[0] == 1

        # Second call: handler already installed, resolver should NOT fire.
        result2 = engine.simplify(["square", 5])
        assert result2 == 25
        assert call_count[0] == 1  # Still 1; not 2.

    def test_no_resolver_falls_through_to_compound(self):
        engine = RuleEngine.from_dsl("@compute: (foo ?x) => (! my-op :x)")
        # No resolver registered. Existing behavior: (! my-op 42) becomes [my-op, 42].
        result = engine.simplify(["foo", 42])
        assert result == ["my-op", 42]

    def test_resolver_returning_none_falls_through(self):
        engine = RuleEngine.from_dsl("@compute: (foo ?x) => (! my-op :x)")
        engine.on_undefined_op(lambda op, args, ctx: None)
        result = engine.simplify(["foo", 42])
        assert result == ["my-op", 42]

    def test_undefined_op_resolver_can_cancel_via_ctx(self):
        engine = RuleEngine.from_dsl("""
            @r1: (foo ?x) => (! op :x)
            @r2: (bar ?x) => :x
        """)
        seen = []

        @engine.on_undefined_op
        def resolver(op, args, ctx):
            seen.append(op)
            ctx.cancel()
            return None

        # The compute (! op 42) triggers the resolver, which cancels.
        # The cancellation should propagate and halt the rewrite.
        result = engine.simplify(["foo", 42])
        # The original expression rewrote to (! op 42) which can't be folded;
        # resolver cancelled. Engine should bail out.
        assert engine._cancel_requested  # The cancel signal was set.


class TestFoldErrorResolver:
    def test_fold_raises_resolver_provides_fallback(self):
        from rerum.hooks import Resolution

        def explode(args):
            raise ValueError("boom")

        engine = RuleEngine(fold_funcs={"bomb": explode})
        engine.load_dsl("@detonate: (foo ?x) => (! bomb :x)")

        @engine.on_fold_error
        def resolver(op, args, exception, ctx):
            return Resolution(value="defused")

        result = engine.simplify(["foo", 42])
        assert result == "defused"

    def test_no_resolver_keeps_existing_silent_fallback(self):
        def explode(args):
            raise ValueError("boom")

        engine = RuleEngine(fold_funcs={"bomb": explode})
        engine.load_dsl("@detonate: (foo ?x) => (! bomb :x)")
        # No resolver: existing behavior leaves [bomb, 42] in place.
        result = engine.simplify(["foo", 42])
        assert result == ["bomb", 42]

    def test_resolver_returning_none_falls_through(self):
        from rerum.hooks import Resolution

        def explode(args):
            raise ValueError("boom")

        engine = RuleEngine(fold_funcs={"bomb": explode})
        engine.load_dsl("@detonate: (foo ?x) => (! bomb :x)")

        @engine.on_fold_error
        def resolver(op, args, exception, ctx):
            return None  # decline

        result = engine.simplify(["foo", 42])
        assert result == ["bomb", 42]

    def test_resolver_receives_exception(self):
        from rerum.hooks import Resolution

        def explode(args):
            raise ValueError("boom")

        engine = RuleEngine(fold_funcs={"bomb": explode})
        engine.load_dsl("@detonate: (foo ?x) => (! bomb :x)")

        seen = []

        @engine.on_fold_error
        def resolver(op, args, exception, ctx):
            seen.append((op, args, type(exception).__name__, str(exception)))
            return Resolution(value="ok")

        engine.simplify(["foo", 42])
        assert seen == [("bomb", [42], "ValueError", "boom")]

    def test_fold_error_resolver_can_cancel_via_ctx(self):
        from rerum.hooks import Resolution

        def explode(args):
            raise ValueError("boom")

        engine = RuleEngine(fold_funcs={"bomb": explode})
        engine.load_dsl("@detonate: (foo ?x) => (! bomb :x)")

        @engine.on_fold_error
        def resolver(op, args, exception, ctx):
            ctx.cancel()
            return None

        # Cancellation propagates via _cancel_requested.
        engine.simplify(["foo", 42])
        assert engine._cancel_requested

    def test_fold_error_resolver_can_abort_via_resolution(self):
        from rerum.hooks import Resolution

        def explode(args):
            raise ValueError("boom")

        engine = RuleEngine(fold_funcs={"bomb": explode})
        engine.load_dsl("@detonate: (foo ?x) => (! bomb :x)")

        @engine.on_fold_error
        def resolver(op, args, exception, ctx):
            return Resolution(abort=True)

        engine.simplify(["foo", 42])
        # Resolution(abort=True) and ctx.cancel() are equivalent: both set
        # _cancel_requested.
        assert engine._cancel_requested


class TestCycleAndFixpointEvents:
    def test_cycle_event_fires_on_bidirectional(self):
        engine = RuleEngine.from_dsl("@commute: (+ ?x ?y) <=> (+ :y :x)")
        cycles = []
        engine.on_cycle(lambda expr, path, ctx: cycles.append(expr))
        engine.simplify(["+", "a", "b"])
        # The cycle detection in simplify fires when the engine returns
        # to a visited state.
        assert len(cycles) >= 1

    def test_fixpoint_event_fires_at_convergence(self):
        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")
        finals = []
        engine.on_fixpoint(lambda expr, ctx: finals.append(expr))
        result = engine.simplify(["+", "x", 0])
        assert result == "x"
        assert finals == ["x"]

    def test_cycle_resolver_can_abort(self):
        from rerum.hooks import Resolution

        engine = RuleEngine.from_dsl("@commute: (+ ?x ?y) <=> (+ :y :x)")

        @engine.on_cycle
        def aborter(expr, path, ctx):
            return Resolution(abort=True)

        result = engine.simplify(["+", "a", "b"])
        # Abort propagates; engine returns whatever it has at that point.
        assert result in (["+", "a", "b"], ["+", "b", "a"])

    def test_fixpoint_observer_receives_final_expr(self):
        engine = RuleEngine.from_dsl("""
            @add-zero: (+ ?x 0) => :x
            @mul-one: (* ?x 1) => :x
        """)
        finals = []
        engine.on_fixpoint(lambda expr, ctx: finals.append(expr))
        result = engine.simplify(["+", ["*", "y", 1], 0])
        assert result == "y"
        assert finals == ["y"]

    def test_fixpoint_observer_receives_ctx(self):
        engine = RuleEngine.from_dsl("@id: (foo ?x) => :x")
        ctxs = []
        engine.on_fixpoint(lambda expr, ctx: ctxs.append(ctx))
        engine.simplify(["foo", "y"])
        assert len(ctxs) == 1
        assert ctxs[0].engine is engine
        assert ctxs[0].event_name == "fixpoint"

    def test_fixpoint_does_not_fire_on_max_steps_exhaustion(self):
        from rerum.engine import parse_rule_line

        # A rule that keeps producing different output forever (no convergence).
        # If max_steps caps without convergence, fixpoint should NOT fire.
        engine = RuleEngine()
        # Rule that wraps in another layer each time: (foo X) -> (foo (foo X)),
        # never converges.
        pairs = parse_rule_line("@grow: (foo ?x) => (foo (foo :x))")
        for meta, pat, skel in pairs:
            engine._rules.append([pat, skel])
            engine._metadata.append(meta)
            if meta.name:
                engine._rule_names[meta.name] = len(engine._rules) - 1

        finals = []
        engine.on_fixpoint(lambda expr, ctx: finals.append(expr))

        # max_steps=3 exhausts before convergence; visited set won't break early
        # because each step grows the expression to a new form.
        engine.simplify(["foo", "x"], max_steps=3, strategy="bottomup")
        # No fixpoint fired because we never converged naturally.
        assert finals == []

        # Same check for topdown.
        finals.clear()
        engine.simplify(["foo", "x"], max_steps=3, strategy="topdown")
        assert finals == []

        # Same check for exhaustive.
        finals.clear()
        engine.simplify(["foo", "x"], max_steps=3)
        assert finals == []


class TestMaxDepthResolver:
    def test_max_depth_event_fires_when_budget_exhausted(self):
        engine = RuleEngine.from_dsl("@commute: (+ ?x ?y) <=> (+ :y :x)")
        called = [0]

        @engine.on_max_depth
        def resolver(expr, depth, ctx):
            called[0] += 1
            return None  # decline

        # Tight depth budget triggers max_depth.
        list(engine.equivalents(["+", "a", "b"], max_depth=1))
        assert called[0] >= 1

    def test_no_resolver_means_no_event(self):
        engine = RuleEngine.from_dsl("@commute: (+ ?x ?y) <=> (+ :y :x)")
        # No resolver registered. equivalents just stops at max_depth.
        forms = list(engine.equivalents(["+", "a", "b"], max_depth=0))
        # At max_depth=0 with the original yielded, then stop.
        assert forms == [["+", "a", "b"]]

    def test_allow_more_extends_budget_once(self):
        from rerum.hooks import Resolution

        engine = RuleEngine.from_dsl("@commute: (+ ?x ?y) <=> (+ :y :x)")

        # Baseline: with no resolver, max_depth=0 yields only the original.
        baseline = list(engine.equivalents(["+", "a", "b"], max_depth=0))
        assert len(baseline) == 1
        assert baseline[0] == ["+", "a", "b"]

        # With allow_more, the budget extends so commute can fire.
        @engine.on_max_depth
        def resolver(expr, depth, ctx):
            return Resolution(allow_more=True)

        forms = list(engine.equivalents(["+", "a", "b"], max_depth=0))
        keys = {tuple(f) for f in forms}
        assert ("+", "a", "b") in keys
        assert ("+", "b", "a") in keys  # commute fired thanks to extension


class TestResolverRetryCap:
    def test_resolver_loop_error_raised_after_cap(self):
        from rerum import ResolverLoopError
        from rerum.engine import parse_rule_line
        from rerum.hooks import Resolution

        engine = RuleEngine()

        @engine.on_no_match
        def looping(expr, ctx):
            # Anonymous rule that doesn't match anything in the input.
            pairs = parse_rule_line("(zzz ?y) => :y")
            rules = [(m, [p, s]) for m, p, s in pairs]
            return Resolution(rules=rules)

        with pytest.raises(ResolverLoopError, match="retry"):
            engine.simplify(["foo", "bar"], max_steps=200)

    def test_named_rules_dedup_avoids_loop_error(self):
        # Named rules dedupe so the same rule isn't re-installed; T9 fix
        # protects against this case without needing T14 to fire.
        from rerum.engine import parse_rule_line
        from rerum.hooks import Resolution

        engine = RuleEngine()

        @engine.on_no_match
        def stable(expr, ctx):
            pairs = parse_rule_line("@same: (zzz ?y) => :y")
            rules = [(m, [p, s]) for m, p, s in pairs]
            return Resolution(rules=rules)

        # Should NOT raise: the named rule is deduped after first install,
        # and the engine falls through after that without resolver loops.
        result = engine.simplify(["foo", "bar"])
        assert result == ["foo", "bar"]

    def test_cap_does_not_fire_under_normal_use(self):
        # A resolver that returns a useful rule on first call shouldn't
        # trigger the cap.
        from rerum.engine import parse_rule_line
        from rerum.hooks import Resolution

        engine = RuleEngine()

        @engine.on_no_match
        def helpful(expr, ctx):
            if isinstance(expr, list) and expr and expr[0] == "foo":
                pairs = parse_rule_line("@foo-id: (foo ?x) => :x")
                rules = [(m, [p, s]) for m, p, s in pairs]
                return Resolution(rules=rules)
            return None

        # No exception; rule fires.
        result = engine.simplify(["foo", "x"])
        assert result == "x"


class TestTraceMigration:
    def test_simplify_trace_still_returns_pair(self):
        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")
        result, trace = engine.simplify(["+", "x", 0], trace=True)
        assert result == "x"
        assert len(trace.steps) == 1

    def test_simplify_trace_captures_multiple_steps(self):
        engine = RuleEngine.from_dsl("""
            @add-zero: (+ ?x 0) => :x
            @mul-one: (* ?x 1) => :x
        """)
        result, trace = engine.simplify(["+", ["*", "y", 1], 0], trace=True)
        assert result == "y"
        names = [s.metadata.name for s in trace.steps]
        assert "mul-one" in names
        assert "add-zero" in names

    def test_external_observer_does_not_pollute_returned_trace(self):
        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")
        external_log = []
        engine.on_rule_applied(lambda step, ctx: external_log.append(step))

        result, trace = engine.simplify(["+", "x", 0], trace=True)
        # Both observed the step, but the returned trace contains only the
        # internal accumulation; the external observer captured the same step
        # independently.
        assert len(external_log) == 1
        assert len(trace.steps) == 1
        # The trace step is the same RewriteStep instance that the external
        # observer received (both go through _fire_rule_applied).

    def test_trace_observer_unregistered_after_call(self):
        # The trace registers a temporary observer for the duration of the
        # call. After the call returns, that observer should be gone.
        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")
        engine.simplify(["+", "x", 0], trace=True)
        # No external observer was registered, so count should be 0.
        assert engine._hooks.count("rule_applied") == 0


class TestCancelRequestedIsolation:
    """A cancellation from one top-level call must not leak into the next."""

    def test_cancel_from_simplify_does_not_affect_subsequent_equivalents(self):
        engine = RuleEngine.from_dsl("@commute: (+ ?x ?y) <=> (+ :y :x)")

        @engine.on_rule_applied
        def cancel_first(step, ctx):
            ctx.cancel()

        # First call: cancellation fires, simplify returns early.
        engine.simplify(["+", "a", "b"])
        # Now _cancel_requested is True. Without a reset, equivalents
        # would bail immediately.
        engine.off_rule_applied(cancel_first)

        forms = list(engine.equivalents(["+", "a", "b"], max_depth=2))
        # Both equivalents should be reachable; the cancellation flag from
        # the previous call must have been reset.
        keys = {tuple(f) for f in forms}
        assert ("+", "a", "b") in keys
        assert ("+", "b", "a") in keys

    def test_cancel_isolation_between_simplify_calls(self):
        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")

        @engine.on_rule_applied
        def cancel_first(step, ctx):
            ctx.cancel()

        # First call: cancellation fires.
        engine.simplify(["+", "x", 0])
        # Even with the hook still registered, the next call must reset
        # the flag at entry, fire the rule, then cancel again. The
        # contract is that each top-level call starts fresh.
        result = engine.simplify(["+", "y", 0])
        # Cancellation fired again, but the rule had already produced "y".
        assert result == "y"


class TestStrategyParity:
    """no_match, cycle, and fixpoint must fire across all strategies, not
    just exhaustive. The hook-driven LLM rule inference flow needs this
    parity to work uniformly."""

    def test_no_match_fires_in_bottomup(self):
        engine = RuleEngine()  # no rules
        seen = []
        engine.on_no_match(lambda expr, ctx: seen.append(expr))
        engine.simplify(["foo", "bar"], strategy="bottomup")
        # The compound (foo bar) had no matching rule.
        assert ["foo", "bar"] in seen

    def test_no_match_fires_in_topdown(self):
        engine = RuleEngine()
        seen = []
        engine.on_no_match(lambda expr, ctx: seen.append(expr))
        engine.simplify(["foo", "bar"], strategy="topdown")
        assert ["foo", "bar"] in seen

    def test_no_match_value_resolution_in_bottomup(self):
        from rerum.hooks import Resolution
        engine = RuleEngine()

        @engine.on_no_match
        def resolver(expr, ctx):
            if expr == ["foo", "bar"]:
                return Resolution(value="baz")
            return None

        result = engine.simplify(["foo", "bar"], strategy="bottomup")
        assert result == "baz"

    def test_no_match_value_resolution_in_topdown(self):
        from rerum.hooks import Resolution
        engine = RuleEngine()

        @engine.on_no_match
        def resolver(expr, ctx):
            if expr == ["foo", "bar"]:
                return Resolution(value="baz")
            return None

        result = engine.simplify(["foo", "bar"], strategy="topdown")
        assert result == "baz"

    def test_no_match_rules_resolution_in_bottomup(self):
        from rerum.engine import parse_rule_line
        from rerum.hooks import Resolution

        engine = RuleEngine()

        @engine.on_no_match
        def resolver(expr, ctx):
            if isinstance(expr, list) and expr and expr[0] == "foo":
                pairs = parse_rule_line("@foo-id: (foo ?x) => :x")
                rules = [(m, [p, s]) for m, p, s in pairs]
                return Resolution(rules=rules)
            return None

        result = engine.simplify(["foo", "y"], strategy="bottomup")
        assert result == "y"

    def test_cycle_fires_in_bottomup(self):
        engine = RuleEngine.from_dsl("@commute: (+ ?x ?y) <=> (+ :y :x)")
        cycles = []
        engine.on_cycle(lambda expr, path, ctx: cycles.append(expr))
        engine.simplify(["+", "a", "b"], strategy="bottomup")
        assert len(cycles) >= 1

    def test_cycle_fires_in_topdown(self):
        engine = RuleEngine.from_dsl("@commute: (+ ?x ?y) <=> (+ :y :x)")
        cycles = []
        engine.on_cycle(lambda expr, path, ctx: cycles.append(expr))
        engine.simplify(["+", "a", "b"], strategy="topdown")
        assert len(cycles) >= 1

    def test_fixpoint_fires_in_bottomup(self):
        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")
        finals = []
        engine.on_fixpoint(lambda expr, ctx: finals.append(expr))
        result = engine.simplify(["+", "y", 0], strategy="bottomup")
        assert result == "y"
        assert finals == ["y"]

    def test_fixpoint_fires_in_topdown(self):
        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")
        finals = []
        engine.on_fixpoint(lambda expr, ctx: finals.append(expr))
        result = engine.simplify(["+", "y", 0], strategy="topdown")
        assert result == "y"
        assert finals == ["y"]

    def test_no_match_does_not_fire_when_rule_matches(self):
        """Sanity: no_match should NOT fire if a rule matched at that
        position, regardless of strategy."""
        engine = RuleEngine.from_dsl("@id: (foo ?x) => :x")
        seen = []
        engine.on_no_match(lambda expr, ctx: seen.append(expr))
        engine.simplify(["foo", "y"], strategy="bottomup")
        # foo-id matched; no_match should not fire on the (foo y) compound.
        assert ["foo", "y"] not in seen


class TestContextFieldConsistency:
    """step_count and depth in HookContext should be populated correctly
    across all strategies, not just _simplify_exhaustive."""

    def test_step_count_in_bottomup(self):
        engine = RuleEngine.from_dsl("""
            @r1: (a ?x) => (b :x)
            @r2: (b ?x) => (c :x)
        """)
        counts = []
        engine.on_rule_applied(lambda step, ctx: counts.append(ctx.step_count))
        engine.simplify(["a", "x"], strategy="bottomup")
        # Two rule firings: r1 -> step_count=1, r2 -> step_count=2.
        assert counts == [1, 2]

    def test_step_count_in_topdown(self):
        engine = RuleEngine.from_dsl("""
            @r1: (a ?x) => (b :x)
            @r2: (b ?x) => (c :x)
        """)
        counts = []
        engine.on_rule_applied(lambda step, ctx: counts.append(ctx.step_count))
        engine.simplify(["a", "x"], strategy="topdown")
        assert counts == [1, 2]

    def test_step_count_resets_between_calls(self):
        engine = RuleEngine.from_dsl("@r: (a ?x) => :x")
        counts = []
        engine.on_rule_applied(lambda step, ctx: counts.append(ctx.step_count))
        engine.simplify(["a", "x"])
        engine.simplify(["a", "y"])
        # Each top-level call resets; both observed step_count=1.
        assert counts == [1, 1]

    def test_depth_in_no_match_for_nested_expression(self):
        engine = RuleEngine()
        seen = []
        engine.on_no_match(lambda expr, ctx: seen.append((expr, ctx.depth)))
        engine.simplify(["foo", ["bar", "baz"]], strategy="bottomup")
        # Both compounds should fire no_match. The inner one has depth > 0;
        # the outer has depth = 0.
        depths = sorted({d for _, d in seen})
        # At least depth 0 (outer) and depth 1 (inner).
        assert 0 in depths
        assert 1 in depths

    def test_step_count_cumulative_across_subexpressions(self):
        engine = RuleEngine.from_dsl("""
            @r1: (foo ?x) => :x
            @r2: (bar ?x) => :x
        """)
        counts = []
        engine.on_rule_applied(lambda step, ctx: counts.append(ctx.step_count))
        # Compound with nested compounds: bottomup fires r2 on inner first,
        # then r1 on outer. Both step_count values must be distinct.
        engine.simplify(["foo", ["bar", "y"]], strategy="bottomup")
        # Two rule firings, increasing counts.
        assert counts == [1, 2] or counts == [2, 1]  # order depends on traversal
        # The set is {1, 2}.
        assert set(counts) == {1, 2}

    def test_depth_in_no_match_for_three_level_tree(self):
        engine = RuleEngine()
        seen = []
        engine.on_no_match(lambda expr, ctx: seen.append((expr, ctx.depth)))
        # Three levels of nesting.
        engine.simplify(["a", ["b", ["c", "d"]]], strategy="bottomup")
        depths = sorted({d for _, d in seen})
        # depth 0 (outer), depth 1 (middle), depth 2 (innermost compound).
        assert depths == [0, 1, 2]

    def test_step_count_in_exhaustive_with_recursive_descent(self):
        """The exhaustive strategy recurses into children when no top-level
        rule matches. The step counter must accumulate across the entire
        top-level call, not reset per recursion level."""
        engine = RuleEngine.from_dsl("""
            @r1: (foo ?x) => :x
            @r2: (bar ?x) => :x
        """)
        counts = []
        engine.on_rule_applied(lambda step, ctx: counts.append(ctx.step_count))
        # Two rules fire: r1 on the outer (foo (bar y)) -> (bar y), then
        # r2 on (bar y) -> y. (Or r2 first if exhaustive descends into the
        # child before the parent rule loop finds a match.)
        engine.simplify(["foo", ["bar", "y"]])
        assert sorted(counts) == [1, 2]
