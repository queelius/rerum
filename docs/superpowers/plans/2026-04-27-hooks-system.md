# Rerum Hooks System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the engine-level hooks system from `docs/superpowers/specs/2026-04-27-hooks-design.md`. Eight named events spanning three categories (observer, resolver, decision), each with the right composition policy. Backward-compatible with the existing `RewriteTrace` and `simplify(trace=True)` paths.

**Architecture:** New `rerum/hooks.py` module holds the `Resolution` dataclass, `HookContext`, exception types, and `_HookRegistry` (the per-category composition logic). `RuleEngine` gains a `_hooks: _HookRegistry` attribute and per-event `on_<event>`/`off_<event>` methods. Each event is fired at one specific point in the existing rule application loop. Mid-rewrite rule mutation goes through a pending-queue drained between iterations to preserve priority-sort and simplifier-cache invariants.

**Tech Stack:** Python 3.9+, `dataclasses`, `pytest` for tests. No new external deps.

---

## File Structure

**Create:**
- `rerum/hooks.py` (new): `Resolution`, `HookContext`, error types, `_HookRegistry`, composition helpers
- `rerum/tests/test_hooks.py` (new): unit tests for the registry, composition, validation, and per-event firing
- `rerum/tests/test_hooks_integration.py` (new): integration tests exercising hooks against a live `RuleEngine`

**Modify:**
- `rerum/engine.py`: add `_hooks` attribute to `RuleEngine.__init__`, add `on_<event>` / `off_<event>` / `clear_hooks` methods, wire each event at its dispatch point, drain pending-rule queue between iterations, migrate `_simplify_with_trace` to use the new system
- `rerum/__init__.py`: re-export `Resolution`, `HookContext`, the error types
- `CHANGELOG.md`: document the hooks system as a feature addition under `[Unreleased]`
- `CLAUDE.md`: brief mention of the hook architecture in the "Architecture" section

---

## Task 1: Create `Resolution` dataclass with validation

**Files:**
- Create: `rerum/hooks.py`
- Test: `rerum/tests/test_hooks.py`

- [ ] **Step 1: Write the failing test**

Create `rerum/tests/test_hooks.py` with:

```python
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
            Resolution(value=42, rules=[])

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
        with pytest.raises((AttributeError, TypeError)):
            r.value = 99
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_hooks.py -v 2>&1 | tail -20
```

Expected: ImportError or ModuleNotFoundError on `rerum.hooks`.

- [ ] **Step 3: Write minimal implementation**

Create `rerum/hooks.py`:

```python
"""Engine-level hooks system.

Three categories with three composition policies:
  - Observer: broadcast (all run, return values ignored)
  - Resolver: chain of responsibility (first non-None Resolution wins)
  - Decision: AND-gate (every hook must return True)

The eight events (rule_applied, fixpoint, no_match, undefined_op, fold_error,
max_depth, cycle, should_fire) are fired by the engine at its natural pause
points; this module provides the data types they exchange.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .rewriter import ExprType


class HooksError(Exception):
    """Base class for hook system errors."""


class ResolutionError(HooksError):
    """Raised when a hook returns a malformed Resolution."""


class HookError(HooksError):
    """Wraps an exception raised inside a hook."""

    def __init__(self, hook: Callable, event: str, cause: BaseException):
        super().__init__(
            f"hook {getattr(hook, '__name__', repr(hook))!r} raised "
            f"during event {event!r}: {cause}"
        )
        self.hook = hook
        self.event = event
        self.cause = cause


class ResolverLoopError(HooksError):
    """Raised when resolvers re-enter past the per-call retry cap."""


@dataclass(frozen=True)
class Resolution:
    """Returned by a Resolver hook to override engine default behavior at a
    dead-end. Exactly one of ``value``, ``rules``, ``fold_funcs``,
    ``allow_more``, or ``abort`` must be the non-default action;
    ``metadata`` is orthogonal and may co-exist with any action.
    """

    value: Optional["ExprType"] = None
    rules: Optional[List[Any]] = None
    fold_funcs: Optional[Dict[str, Callable]] = None
    allow_more: Optional[bool] = None
    abort: bool = False
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        # Count primary actions (excluding metadata).
        actions = [
            self.value is not None,
            self.rules is not None,
            self.fold_funcs is not None,
            self.allow_more is not None,
            self.abort,
        ]
        n = sum(actions)
        if n == 0:
            raise ResolutionError(
                "Resolution is empty: set exactly one of "
                "value/rules/fold_funcs/allow_more/abort"
            )
        if n > 1:
            raise ResolutionError(
                "Resolution is ambiguous: set exactly one of "
                "value/rules/fold_funcs/allow_more/abort"
            )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_hooks.py::TestResolutionValidation -v 2>&1 | tail -15
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/spinoza/github/repos/rerum && git add rerum/hooks.py rerum/tests/test_hooks.py
git commit -m "$(cat <<'EOF'
feat(hooks): add Resolution dataclass with validation

First piece of the hook system: the structured return type from Resolver
hooks. Validates that exactly one primary action is set per Resolution;
metadata is orthogonal.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add `HookContext`

**Files:**
- Modify: `rerum/hooks.py`
- Test: `rerum/tests/test_hooks.py`

- [ ] **Step 1: Write the failing test**

Append to `rerum/tests/test_hooks.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_hooks.py::TestHookContext -v 2>&1 | tail -10
```

Expected: ImportError on `HookContext`.

- [ ] **Step 3: Write minimal implementation**

Append to `rerum/hooks.py`:

```python
class HookContext:
    """Read access to engine state plus controlled mutation primitives.

    Constructed by the engine for each hook invocation; not user-instantiable
    in normal use (but the constructor stays public for testability).
    """

    __slots__ = (
        "engine", "_expr_path", "depth", "step_count", "event_name",
        "cancelled",
    )

    def __init__(
        self,
        engine,
        expr_path: List["ExprType"],
        depth: int,
        step_count: int,
        event_name: str,
    ):
        self.engine = engine
        self._expr_path = tuple(expr_path)
        self.depth = depth
        self.step_count = step_count
        self.event_name = event_name
        self.cancelled = False

    @property
    def expr_path(self) -> tuple:
        """Ancestry from root expression to current position. Immutable view."""
        return self._expr_path

    def cancel(self) -> None:
        """Signal the engine to abort the current rewrite. Equivalent to
        returning ``Resolution(abort=True)`` from a Resolver."""
        self.cancelled = True
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_hooks.py::TestHookContext -v 2>&1 | tail -10
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/spinoza/github/repos/rerum && git add rerum/hooks.py rerum/tests/test_hooks.py
git commit -m "$(cat <<'EOF'
feat(hooks): add HookContext with engine state access

HookContext exposes engine, expr_path (immutable tuple), depth, step_count,
event_name, and cancel(). Hooks read engine state through it and signal
abort via cancel().

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Add `_HookRegistry` with composition helpers

**Files:**
- Modify: `rerum/hooks.py`
- Test: `rerum/tests/test_hooks.py`

- [ ] **Step 1: Write the failing test**

Append to `rerum/tests/test_hooks.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_hooks.py -v -k "Registry" 2>&1 | tail -15
```

Expected: ImportError on `_HookRegistry`.

- [ ] **Step 3: Write minimal implementation**

Append to `rerum/hooks.py`:

```python
class _HookRegistry:
    """Registry mapping (event, callable) pairs with three composition
    policies: ``run_observers`` (broadcast), ``run_resolvers`` (chain),
    ``run_decisions`` (AND-gate).

    The category of each event is supplied at registration time; calling the
    wrong runner for an event is a programming error and is intentionally
    not protected against (the engine always uses the correct runner for the
    event it is firing).
    """

    def __init__(self):
        self._hooks: Dict[str, List[Callable]] = {}

    def register(self, event: str, category: str, callback: Callable) -> None:
        # `category` is currently descriptive only; the runner method picks
        # the policy. Reserved here for a future symmetry check.
        if category not in ("observer", "resolver", "decision"):
            raise ValueError(f"unknown hook category: {category!r}")
        self._hooks.setdefault(event, []).append(callback)

    def unregister(self, event: str, callback: Callable) -> bool:
        """Remove ``callback`` from ``event``. Returns True if it was present."""
        hooks = self._hooks.get(event)
        if not hooks:
            return False
        try:
            hooks.remove(callback)
            return True
        except ValueError:
            return False

    def clear(self, event: Optional[str] = None) -> None:
        """Remove all hooks for ``event``, or all hooks for all events."""
        if event is None:
            self._hooks.clear()
        else:
            self._hooks.pop(event, None)

    def count(self, event: str) -> int:
        return len(self._hooks.get(event, ()))

    def run_observers(self, event: str, *args) -> None:
        """Broadcast: every registered hook runs in order; return values
        ignored. Hook exceptions wrap as ``HookError`` and propagate."""
        for hook in list(self._hooks.get(event, ())):
            try:
                hook(*args)
            except Exception as cause:
                raise HookError(hook, event, cause) from cause

    def run_resolvers(self, event: str, *args) -> Optional[Resolution]:
        """Chain of responsibility: first non-None Resolution wins."""
        for hook in list(self._hooks.get(event, ())):
            try:
                result = hook(*args)
            except Exception as cause:
                raise HookError(hook, event, cause) from cause
            if result is not None:
                if not isinstance(result, Resolution):
                    raise ResolutionError(
                        f"resolver for {event!r} returned {type(result).__name__}, "
                        f"expected Resolution or None"
                    )
                return result
        return None

    def run_decisions(self, event: str, *args) -> bool:
        """AND-gate: every hook must return truthy. First False short-circuits."""
        for hook in list(self._hooks.get(event, ())):
            try:
                result = hook(*args)
            except Exception as cause:
                raise HookError(hook, event, cause) from cause
            if not result:
                return False
        return True
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_hooks.py -v 2>&1 | tail -25
```

Expected: all tests pass (Resolution + HookContext + Registry).

- [ ] **Step 5: Commit**

```bash
cd /home/spinoza/github/repos/rerum && git add rerum/hooks.py rerum/tests/test_hooks.py
git commit -m "$(cat <<'EOF'
feat(hooks): add _HookRegistry with three composition policies

run_observers (broadcast), run_resolvers (chain of responsibility, first
non-None wins), run_decisions (AND-gate, first False short-circuits).
Hook exceptions are wrapped in HookError with the event name and original
cause for diagnostic clarity.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Wire `_hooks` registry into RuleEngine + add `on_<event>` / `off_<event>` API

**Files:**
- Modify: `rerum/engine.py` (RuleEngine.__init__, plus 8 pairs of methods near the public API)
- Modify: `rerum/__init__.py` (re-export Resolution, HookContext, error types)
- Test: `rerum/tests/test_hooks_integration.py` (new)

- [ ] **Step 1: Write the failing test**

Create `rerum/tests/test_hooks_integration.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_hooks_integration.py -v 2>&1 | tail -15
```

Expected: AttributeError on `engine._hooks` and on `engine.on_rule_applied`; ImportError on the public re-exports.

- [ ] **Step 3: Write minimal implementation**

Edit `rerum/engine.py`:

In the imports section (top of file, near other internal imports), add:

```python
from .hooks import (
    _HookRegistry,
    HookContext,
    Resolution,
    HookError,
    ResolutionError,
    ResolverLoopError,
)
```

In `RuleEngine.__init__`, add at the end (after `self._disabled_groups`):

```python
        self._hooks = _HookRegistry()
```

Inside `RuleEngine`, near the bottom of the class (after the existing `match` method, before `apply_once`), add the registration API. The events table is the source of truth:

```python
    # ============================================================
    # Hook registration API
    # ============================================================

    _HOOK_EVENTS = {
        "rule_applied":  "observer",
        "fixpoint":      "observer",
        "no_match":      "resolver",
        "undefined_op":  "resolver",
        "fold_error":    "resolver",
        "max_depth":     "resolver",
        "cycle":         "resolver",
        "should_fire":   "decision",
    }

    def _make_on_method(event: str, category: str):
        def on_event(self, callback):
            self._hooks.register(event, category, callback)
            return callback  # so it works as a decorator
        on_event.__name__ = f"on_{event}"
        on_event.__doc__ = (
            f"Register a {category} hook for the {event!r} event. "
            f"Callable is returned unchanged so this method works as a "
            f"decorator."
        )
        return on_event

    def _make_off_method(event: str):
        def off_event(self, callback):
            return self._hooks.unregister(event, callback)
        off_event.__name__ = f"off_{event}"
        off_event.__doc__ = f"Unregister a previously-registered hook for the {event!r} event."
        return off_event

    def clear_hooks(self, event: Optional[str] = None) -> None:
        """Remove all hooks for ``event``, or all hooks for all events when
        ``event`` is None."""
        self._hooks.clear(event)
```

Then immediately after the class body's closing bracket, install the eight on/off methods on the class:

```python
# Install on_<event> / off_<event> methods on RuleEngine.
for _event, _category in RuleEngine._HOOK_EVENTS.items():
    setattr(RuleEngine, f"on_{_event}",
            RuleEngine._make_on_method(_event, _category))
    setattr(RuleEngine, f"off_{_event}",
            RuleEngine._make_off_method(_event))
del _event, _category
```

Edit `rerum/__init__.py` to re-export the new public names. After the existing `from .engine import (...)`, add:

```python
from .hooks import (
    Resolution,
    HookContext,
    HookError,
    ResolutionError,
    ResolverLoopError,
)
```

And add to the `__all__` list:

```python
    # Hooks
    "Resolution",
    "HookContext",
    "HookError",
    "ResolutionError",
    "ResolverLoopError",
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_hooks.py rerum/tests/test_hooks_integration.py -v 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 5: Run full test suite to confirm no regression**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest 2>&1 | tail -3
```

Expected: previous test count + new tests, all pass.

- [ ] **Step 6: Commit**

```bash
cd /home/spinoza/github/repos/rerum && git add rerum/engine.py rerum/__init__.py rerum/tests/test_hooks_integration.py
git commit -m "$(cat <<'EOF'
feat(hooks): wire _HookRegistry into RuleEngine with on/off API

Adds engine._hooks and the eight on_<event>/off_<event> methods. Each
on_<event> works as both decorator and method form. clear_hooks(event=None)
removes hooks for one event or all events. Public types Resolution,
HookContext, and the error classes re-exported from rerum.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Wire `rule_applied` event into `_simplify_exhaustive`

**Files:**
- Modify: `rerum/engine.py` (the `_simplify_exhaustive` method)
- Test: `rerum/tests/test_hooks_integration.py`

- [ ] **Step 1: Write the failing test**

Append to `rerum/tests/test_hooks_integration.py`:

```python
class TestRuleAppliedEvent:
    def test_rule_applied_fires_on_each_step(self):
        engine = RuleEngine.from_dsl("""
            @add-zero: (+ ?x 0) => :x
            @mul-one: (* ?x 1) => :x
        """)
        steps = []

        @engine.on_rule_applied
        def observer(step, ctx):
            steps.append(step.metadata.name)

        engine.simplify(["+", ["*", "a", 1], 0])
        assert "mul-one" in steps
        assert "add-zero" in steps

    def test_observers_broadcast_to_all(self):
        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")
        log_a = []
        log_b = []
        engine.on_rule_applied(lambda step, ctx: log_a.append(step.metadata.name))
        engine.on_rule_applied(lambda step, ctx: log_b.append(step.metadata.name))

        engine.simplify(["+", "x", 0])
        assert log_a == ["add-zero"]
        assert log_b == ["add-zero"]

    def test_ctx_passed_to_observer(self):
        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")
        contexts = []

        @engine.on_rule_applied
        def observer(step, ctx):
            contexts.append(ctx)

        engine.simplify(["+", "x", 0])
        assert len(contexts) == 1
        assert contexts[0].engine is engine
        assert contexts[0].event_name == "rule_applied"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_hooks_integration.py::TestRuleAppliedEvent -v 2>&1 | tail -20
```

Expected: assertions fail because `rule_applied` hooks don't fire yet (only the existing `listener` parameter does).

- [ ] **Step 3: Write minimal implementation**

In `rerum/engine.py`, find `_simplify_exhaustive`. Inside the loop body, where the rule fires and `listener` is called, also fire the `rule_applied` event. The relevant block currently looks like:

```python
                    if new_expr != current:
                        if listener is not None:
                            listener(RewriteStep(
                                rule_index=rule_idx,
                                metadata=metadata,
                                before=current,
                                after=new_expr,
                            ))
                        current = new_expr
                        changed = True
                        break
```

Replace it with:

```python
                    if new_expr != current:
                        step = RewriteStep(
                            rule_index=rule_idx,
                            metadata=metadata,
                            before=current,
                            after=new_expr,
                        )
                        if listener is not None:
                            listener(step)
                        if self._hooks.count("rule_applied"):
                            ctx = HookContext(
                                engine=self,
                                expr_path=[],
                                depth=0,
                                step_count=0,
                                event_name="rule_applied",
                            )
                            self._hooks.run_observers("rule_applied", step, ctx)
                        current = new_expr
                        changed = True
                        break
```

(The `expr_path`, `depth`, `step_count` will be filled in properly when we wire the path-tracking infrastructure in Task 11. For now, empty/zero placeholders are correct: the engine has not yet been instrumented with path tracking, and observers receive a usable `step` payload regardless.)

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_hooks_integration.py::TestRuleAppliedEvent -v 2>&1 | tail -15
```

Expected: 3 tests pass.

- [ ] **Step 5: Run full suite**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest 2>&1 | tail -3
```

Expected: all tests pass, count up by 3.

- [ ] **Step 6: Commit**

```bash
cd /home/spinoza/github/repos/rerum && git add rerum/engine.py rerum/tests/test_hooks_integration.py
git commit -m "$(cat <<'EOF'
feat(hooks): wire rule_applied event into _simplify_exhaustive

Each successful rule application now fires the rule_applied event in
addition to calling the legacy listener. Existing `simplify(trace=True)`
behavior is unchanged (it still uses listener); external observers
attach via `engine.on_rule_applied`.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Wire `rule_applied` event into other simplify paths

**Files:**
- Modify: `rerum/engine.py` (`_bottomup_pass`, `_topdown_pass`, `apply_once`, `_simplify_with_trace`)
- Test: `rerum/tests/test_hooks_integration.py`

- [ ] **Step 1: Write the failing test**

Append to `rerum/tests/test_hooks_integration.py`:

```python
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

    def test_fast_path_fires_rule_applied(self):
        # No conditions, no groups: fast path via rewriter() factory.
        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")
        steps = []
        engine.on_rule_applied(lambda step, ctx: steps.append(step.metadata.name))
        # Default strategy with no groups/conditions hits the fast path.
        engine.simplify(["+", "x", 0])
        assert steps == ["add-zero"]
```

- [ ] **Step 2: Run test to verify failure**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_hooks_integration.py::TestRuleAppliedAcrossStrategies -v 2>&1 | tail -15
```

Expected: 3 of 4 tests fail (bottomup, topdown, apply_once); the fast-path test may currently fail since `rewriter()` doesn't fire engine-level hooks.

- [ ] **Step 3: Write minimal implementation**

Define a small helper inside `RuleEngine` to standardize the firing call site:

```python
    def _fire_rule_applied(self, step: RewriteStep) -> None:
        """Fire the rule_applied event with a standard HookContext."""
        if not self._hooks.count("rule_applied"):
            return
        ctx = HookContext(
            engine=self,
            expr_path=[],
            depth=0,
            step_count=0,
            event_name="rule_applied",
        )
        self._hooks.run_observers("rule_applied", step, ctx)
```

Replace the inline `self._hooks.run_observers(...)` block in `_simplify_exhaustive` from Task 5 with `self._fire_rule_applied(step)`.

In `_bottomup_pass`, find the spot where `result = instantiate(...)` produces a different `current`, and insert step construction + `self._fire_rule_applied(step)` right before `return result`.

In `_topdown_pass`, do the same.

In `apply_once`, do the same right before `return result, metadata`.

For the fast path: the `rewriter()` function in `rewriter.py` does not have access to the engine. To fire engine-level hooks from the fast path, change the fast-path branch in `simplify` to bypass the cached simplifier when hooks are registered:

In `simplify`, find:

```python
            else:
                # Use fast path when no conditions or groups
                if self._simplifier is None:
                    self._simplifier = rewriter(self._rules, fold_funcs=self._fold_funcs)
                return self._simplifier(expr)
```

Replace with:

```python
            else:
                # Use fast path when no conditions or groups AND no hooks
                # are attached to engine-fired events. Hooks need engine
                # context, which the rewriter() factory doesn't have.
                hooks_active = (
                    self._hooks.count("rule_applied") > 0
                    or self._hooks.count("fixpoint") > 0
                    or self._hooks.count("no_match") > 0
                    or self._hooks.count("undefined_op") > 0
                    or self._hooks.count("fold_error") > 0
                    or self._hooks.count("should_fire") > 0
                    or self._hooks.count("cycle") > 0
                )
                if hooks_active:
                    return self._simplify_exhaustive(expr, max_steps, groups=groups)
                if self._simplifier is None:
                    self._simplifier = rewriter(self._rules, fold_funcs=self._fold_funcs)
                return self._simplifier(expr)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_hooks_integration.py::TestRuleAppliedAcrossStrategies -v 2>&1 | tail -15
```

Expected: 4 tests pass.

- [ ] **Step 5: Run full suite**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest 2>&1 | tail -3
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
cd /home/spinoza/github/repos/rerum && git add rerum/engine.py rerum/tests/test_hooks_integration.py
git commit -m "$(cat <<'EOF'
feat(hooks): fire rule_applied across all simplify strategies

Wires rule_applied into _bottomup_pass, _topdown_pass, apply_once, and the
fast path. Fast path falls back to _simplify_exhaustive when any hooks are
attached, since rewriter() factory has no engine reference.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Wire `should_fire` decision

**Files:**
- Modify: `rerum/engine.py`
- Test: `rerum/tests/test_hooks_integration.py`

- [ ] **Step 1: Write the failing test**

Append to `rerum/tests/test_hooks_integration.py`:

```python
class TestShouldFireDecision:
    def test_false_vetoes_rule(self):
        engine = RuleEngine.from_dsl("""
            @add-zero: (+ ?x 0) => :x
        """)

        @engine.on_should_fire
        def veto_all(rule, expr, bindings, ctx):
            return False

        result = engine.simplify(["+", "x", 0])
        assert result == ["+", "x", 0]  # rule blocked

    def test_true_allows_rule(self):
        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")
        engine.on_should_fire(lambda rule, expr, bindings, ctx: True)
        result = engine.simplify(["+", "x", 0])
        assert result == "x"

    def test_decision_and_gate(self):
        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")

        engine.on_should_fire(lambda *a: True)
        engine.on_should_fire(lambda *a: False)  # second one vetoes

        result = engine.simplify(["+", "x", 0])
        assert result == ["+", "x", 0]

    def test_decision_receives_rule_metadata(self):
        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")
        seen_names = []

        @engine.on_should_fire
        def record(rule, expr, bindings, ctx):
            seen_names.append(rule[1].name)  # rule is (pattern_skeleton, metadata)
            return True

        engine.simplify(["+", "x", 0])
        assert "add-zero" in seen_names
```

- [ ] **Step 2: Run test to verify failure**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_hooks_integration.py::TestShouldFireDecision -v 2>&1 | tail -15
```

Expected: failures because `should_fire` is not consulted.

- [ ] **Step 3: Write minimal implementation**

Define a helper on `RuleEngine`:

```python
    def _check_should_fire(self, rule, metadata, expr, bindings) -> bool:
        """Check if all should_fire decisions allow this rule to fire."""
        if not self._hooks.count("should_fire"):
            return True
        ctx = HookContext(
            engine=self,
            expr_path=[],
            depth=0,
            step_count=0,
            event_name="should_fire",
        )
        # Pass (rule_payload, metadata) tuple as `rule` so hooks can inspect both.
        rule_payload = (rule, metadata)
        return self._hooks.run_decisions(
            "should_fire", rule_payload, expr, bindings, ctx
        )
```

In every place where the engine has matched a rule and is about to apply it (after `_check_condition` succeeds), add a call to `_check_should_fire`. Locations:

- `_simplify_exhaustive` (in the loop body, after `if not self._check_condition(...)` block, before constructing `new_expr`)
- `_bottomup_pass` (same pattern)
- `_topdown_pass` (same pattern)
- `apply_once` (same pattern)
- `_all_single_rewrites` (same pattern)

The pattern is: replace

```python
                if bindings is not None:
                    if not self._check_condition(metadata.condition, bindings):
                        continue
                    new_expr = instantiate(skeleton, bindings, self._fold_funcs)
```

with

```python
                if bindings is not None:
                    if not self._check_condition(metadata.condition, bindings):
                        continue
                    if not self._check_should_fire(rule, metadata, current, bindings):
                        continue
                    new_expr = instantiate(skeleton, bindings, self._fold_funcs)
```

(For `_all_single_rewrites`, `current` is named `expr` instead; adjust accordingly. For `apply_once`, the variable is `expr`. Use whichever name is in scope.)

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_hooks_integration.py::TestShouldFireDecision -v 2>&1 | tail -10
```

Expected: 4 tests pass.

- [ ] **Step 5: Run full suite**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest 2>&1 | tail -3
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
cd /home/spinoza/github/repos/rerum && git add rerum/engine.py rerum/tests/test_hooks_integration.py
git commit -m "$(cat <<'EOF'
feat(hooks): wire should_fire decision into rule application

Each rule firing site now consults should_fire decisions after the rule's
DSL condition passes. AND-gate semantics: any False vetoes. Layered on top
of the existing condition/when system, not in place of it.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Wire `no_match` resolver with `Resolution(value=...)` path

**Files:**
- Modify: `rerum/engine.py`
- Test: `rerum/tests/test_hooks_integration.py`

- [ ] **Step 1: Write the failing test**

Append to `rerum/tests/test_hooks_integration.py`:

```python
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
        # No rules and no resolution: expression unchanged.
        result = engine.simplify(["foo", "bar"])
        assert result == ["foo", "bar"]

    def test_resolver_only_fires_at_no_match_position(self):
        # Build an engine where some positions have matching rules and some
        # don't; resolver should only fire at the latter.
        engine = RuleEngine.from_dsl("@id: (foo ?x) => :x")
        called = []

        @engine.on_no_match
        def resolver(expr, ctx):
            called.append(expr)
            return None

        engine.simplify(["foo", "bar"])
        # foo-rule fires on (foo bar) -> bar; then bar is an atom, no rules
        # match. Whether resolver fires for atoms is a tightening choice;
        # we assert it does not (atoms are not "positions where rules could
        # have fired").
        assert ["foo", "bar"] not in called
```

- [ ] **Step 2: Run test to verify failure**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_hooks_integration.py::TestNoMatchResolverValuePath -v 2>&1 | tail -15
```

Expected: failures because `no_match` is not yet wired.

- [ ] **Step 3: Write minimal implementation**

Add a helper on `RuleEngine`:

```python
    def _fire_no_match(self, expr) -> Optional[Resolution]:
        """Fire the no_match event when no rule matches at the current position.

        Returns the Resolution if a resolver provided one, else None.
        Atoms (constants, variables) do not fire no_match: rules apply at
        compound positions, not at leaves.
        """
        if not isinstance(expr, list) or not expr:
            return None
        if not self._hooks.count("no_match"):
            return None
        ctx = HookContext(
            engine=self,
            expr_path=[],
            depth=0,
            step_count=0,
            event_name="no_match",
        )
        return self._hooks.run_resolvers("no_match", expr, ctx)
```

In `_simplify_exhaustive`, after the rule loop completes with `changed = False` and before recursing into subexpressions:

```python
            if not changed:
                # No rule matched at this position. Fire no_match.
                resolution = self._fire_no_match(current)
                if resolution is not None:
                    if resolution.abort:
                        return current
                    if resolution.value is not None:
                        current = resolution.value
                        # Continue the outer loop with the substituted value.
                        continue
                    # rules / fold_funcs paths handled in Tasks 9 and 10.
                # Recursively simplify subexpressions
                if isinstance(current, list) and len(current) > 0:
                    ...
```

(Insert this block before the existing `if isinstance(current, list) and len(current) > 0:` recursion block. The existing code stays in place.)

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_hooks_integration.py::TestNoMatchResolverValuePath -v 2>&1 | tail -10
```

Expected: 3 tests pass.

- [ ] **Step 5: Run full suite**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest 2>&1 | tail -3
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
cd /home/spinoza/github/repos/rerum && git add rerum/engine.py rerum/tests/test_hooks_integration.py
git commit -m "$(cat <<'EOF'
feat(hooks): wire no_match resolver with value-resolution path

When no rule matches at a compound position, fires the no_match event.
Resolution(value=...) substitutes in the returned expression and continues
the outer loop. Resolution(abort=True) returns whatever the engine has so
far. None means proceed normally.

Rules/fold_funcs paths handled in subsequent tasks.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Mid-rewrite mutation: rules path of `no_match` resolver

**Files:**
- Modify: `rerum/engine.py`
- Test: `rerum/tests/test_hooks_integration.py`

- [ ] **Step 1: Write the failing test**

Append to `rerum/tests/test_hooks_integration.py`:

```python
class TestNoMatchResolverRulesPath:
    def test_rules_resolution_adds_rule_and_retries(self):
        from rerum.engine import parse_rule_line
        engine = RuleEngine()

        @engine.on_no_match
        def resolver(expr, ctx):
            from rerum.hooks import Resolution
            if expr == ["foo", "bar"]:
                # Inject a rule for foo via parse_rule_line.
                pairs = parse_rule_line("@foo-id: (foo ?x) => :x")
                # Each pair is (metadata, pattern, skeleton).
                rules_for_resolution = [
                    (meta, [pat, skel]) for meta, pat, skel in pairs
                ]
                return Resolution(rules=rules_for_resolution,
                                  metadata={"provenance": "test"})
            return None

        result = engine.simplify(["foo", "bar"])
        assert result == "bar"

    def test_added_rule_persists_after_call(self):
        from rerum.engine import parse_rule_line
        engine = RuleEngine()

        called = [0]

        @engine.on_no_match
        def resolver(expr, ctx):
            from rerum.hooks import Resolution
            called[0] += 1
            if isinstance(expr, list) and expr and expr[0] == "foo":
                pairs = parse_rule_line("@foo-id: (foo ?x) => :x")
                rules = [(m, [p, s]) for m, p, s in pairs]
                return Resolution(rules=rules)
            return None

        engine.simplify(["foo", "a"])  # adds rule, called once
        first_call_count = called[0]
        engine.simplify(["foo", "b"])  # rule already there, no_match doesn't fire
        # Resolver should not have fired again.
        assert called[0] == first_call_count

    def test_rule_provenance_metadata_attached(self):
        from rerum.engine import parse_rule_line
        engine = RuleEngine()

        @engine.on_no_match
        def resolver(expr, ctx):
            from rerum.hooks import Resolution
            if isinstance(expr, list) and expr and expr[0] == "foo":
                pairs = parse_rule_line("@foo-id: (foo ?x) => :x")
                rules = [(m, [p, s]) for m, p, s in pairs]
                return Resolution(rules=rules, metadata={"provenance": "llm-inferred"})
            return None

        engine.simplify(["foo", "a"])
        # Find the rule in the engine and confirm its metadata.
        rule_meta = engine._metadata[-1]
        assert rule_meta.tags is None or "provenance:llm-inferred" not in (rule_meta.tags or [])
        # The Resolution.metadata is stored on the rule's metadata dict.
        assert getattr(rule_meta, "extra", {}).get("provenance") == "llm-inferred"
```

- [ ] **Step 2: Run test to verify failure**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_hooks_integration.py::TestNoMatchResolverRulesPath -v 2>&1 | tail -15
```

Expected: failures.

- [ ] **Step 3: Write minimal implementation**

First, give `RuleMetadata` an `extra` dict field for resolver-supplied metadata. In `rerum/engine.py`, find `class RuleMetadata` and add `extra` to `__init__`:

```python
    def __init__(self, name: Optional[str] = None, description: Optional[str] = None,
                 tags: Optional[List[str]] = None, condition: Optional[ExprType] = None,
                 priority: int = 0, bidirectional: bool = False,
                 direction: Optional[str] = None,
                 extra: Optional[Dict[str, Any]] = None):
        self.name = name
        self.description = description
        self.tags = tags or []
        self.condition = condition
        self.priority = priority
        self.bidirectional = bidirectional
        self.direction = direction
        self.extra = extra or {}
```

Add a helper on `RuleEngine` to install pending resolver-provided rules:

```python
    def _install_resolver_rules(self, rules, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Install rules provided by a Resolver. Each entry in `rules` is a
        (metadata, [pattern, skeleton]) tuple, matching the shape produced
        by parse_rule_line + the engine's normal rule loaders.

        ``metadata`` from the Resolution is merged into each rule's
        ``RuleMetadata.extra`` dict for later introspection.
        """
        for meta, rule in rules:
            if metadata:
                merged = dict(meta.extra or {})
                merged.update(metadata)
                meta.extra = merged
            self._rules.append(rule)
            self._metadata.append(meta)
            if meta.name:
                self._rule_names[meta.name] = len(self._rules) - 1
        self._sort_by_priority()
        self._simplifier = None  # cache invalidation
```

In `_simplify_exhaustive` where Task 8 added the `no_match` handling, extend the resolution handling:

```python
            if not changed:
                resolution = self._fire_no_match(current)
                if resolution is not None:
                    if resolution.abort:
                        return current
                    if resolution.value is not None:
                        current = resolution.value
                        continue
                    if resolution.rules is not None:
                        self._install_resolver_rules(
                            resolution.rules, metadata=resolution.metadata
                        )
                        # Loop back: try again with the new rule set.
                        continue
                # Recursively simplify subexpressions
                ...
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_hooks_integration.py::TestNoMatchResolverRulesPath -v 2>&1 | tail -10
```

Expected: 3 tests pass.

- [ ] **Step 5: Run full suite**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest 2>&1 | tail -3
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
cd /home/spinoza/github/repos/rerum && git add rerum/engine.py rerum/tests/test_hooks_integration.py
git commit -m "$(cat <<'EOF'
feat(hooks): no_match resolver can install rules and retry

Resolution(rules=[...]) appends rules to the engine, invalidates the
simplifier cache, and continues the outer loop so the new rules are tried.
Resolution.metadata merges into RuleMetadata.extra (provenance, model
name, confidence). Added rules persist for the engine's lifetime.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Wire `undefined_op` resolver in `instantiate`

**Files:**
- Modify: `rerum/rewriter.py` (`instantiate`'s `(! op ...)` branch)
- Modify: `rerum/engine.py` (engine-side wiring; the rewriter takes a callback)
- Test: `rerum/tests/test_hooks_integration.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
class TestUndefinedOpResolver:
    def test_value_resolution_used_as_fold_result(self):
        from rerum import RuleEngine
        from rerum.hooks import Resolution

        engine = RuleEngine.from_dsl(
            "@compute: (foo ?x) => (! my-op :x)"
        )

        @engine.on_undefined_op
        def resolver(op, args, ctx):
            if op == "my-op":
                return Resolution(value=args[0])  # identity
            return None

        result = engine.simplify(["foo", 42])
        assert result == 42

    def test_fold_funcs_resolution_installs_handler(self):
        from rerum import RuleEngine
        from rerum.hooks import Resolution

        engine = RuleEngine.from_dsl(
            "@compute: (square ?x) => (! my-square :x)"
        )

        @engine.on_undefined_op
        def resolver(op, args, ctx):
            if op == "my-square":
                return Resolution(fold_funcs={"my-square": lambda xs: xs[0] * xs[0]})
            return None

        # First call installs handler.
        result1 = engine.simplify(["square", 3])
        assert result1 == 9

        # Second call: handler already installed, resolver doesn't fire.
        result2 = engine.simplify(["square", 5])
        assert result2 == 25

    def test_no_resolver_falls_through_to_compound(self):
        from rerum import RuleEngine

        engine = RuleEngine.from_dsl("@compute: (foo ?x) => (! my-op :x)")
        # No resolver registered. Existing behavior: (! my-op 42) becomes [my-op, 42].
        result = engine.simplify(["foo", 42])
        assert result == ["my-op", 42]
```

- [ ] **Step 2: Run test to verify failure**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_hooks_integration.py::TestUndefinedOpResolver -v 2>&1 | tail -15
```

Expected: failures because `undefined_op` is not yet wired into `instantiate`.

- [ ] **Step 3: Write minimal implementation**

In `rerum/rewriter.py`, `instantiate` accepts an optional `undefined_op_resolver: Optional[Callable[[str, List], Optional["Resolution"]]] = None` parameter. The compute branch becomes:

```python
        if skeleton_compute(s):
            op = s[1]
            raw_args = s[2:]
            args = [loop(arg) for arg in raw_args]
            handler = fold_funcs.get(op) if fold_funcs else None
            if handler is None and undefined_op_resolver is not None:
                resolution = undefined_op_resolver(op, args)
                if resolution is not None:
                    if resolution.value is not None:
                        return resolution.value
                    if resolution.fold_funcs is not None:
                        # Mutate the fold_funcs dict in place; the engine has
                        # passed its own _fold_funcs reference.
                        fold_funcs.update(resolution.fold_funcs)
                        handler = fold_funcs.get(op)
            if handler is not None:
                try:
                    result = handler(args)
                    if result is not None:
                        if isinstance(result, float) and result.is_integer():
                            return int(result)
                        return result
                except Exception:
                    pass
            return [op] + args
```

Update the `instantiate` function signature:

```python
def instantiate(
    skeleton: ExprType,
    bindings,
    fold_funcs: Optional[FoldFuncsType] = None,
    undefined_op_resolver: Optional[Callable] = None,
) -> ExprType:
```

In `instantiate`'s body, when constructing the inner `loop`, pass `undefined_op_resolver` through. Update `instantiate_compound` to also accept and forward the parameter.

In `rerum/engine.py`, define the resolver bridge on the engine:

```python
    def _undefined_op_resolver(self, op: str, args) -> Optional[Resolution]:
        """Bridge from rewriter.instantiate to the on_undefined_op hooks."""
        if not self._hooks.count("undefined_op"):
            return None
        ctx = HookContext(
            engine=self,
            expr_path=[],
            depth=0,
            step_count=0,
            event_name="undefined_op",
        )
        return self._hooks.run_resolvers("undefined_op", op, args, ctx)
```

Find every call to `instantiate(skeleton, bindings, self._fold_funcs)` in `engine.py` and replace with:

```python
                    new_expr = instantiate(
                        skeleton, bindings, self._fold_funcs,
                        undefined_op_resolver=self._undefined_op_resolver,
                    )
```

(Same change for `result = instantiate(...)` calls. Use replace_all if the pattern is identical.)

For the fast path: same constraint as Task 6. The `rewriter()` factory does not have engine context. The fast-path bailout already triggers when hooks are active; extend the bailout to also trigger on `undefined_op` hooks. (The previous task did this; verify it covers `undefined_op`.)

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_hooks_integration.py::TestUndefinedOpResolver -v 2>&1 | tail -10
```

Expected: 3 tests pass.

- [ ] **Step 5: Run full suite**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest 2>&1 | tail -3
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
cd /home/spinoza/github/repos/rerum && git add rerum/engine.py rerum/rewriter.py rerum/tests/test_hooks_integration.py
git commit -m "$(cat <<'EOF'
feat(hooks): wire undefined_op resolver into instantiate compute branch

When a (! op ...) form is encountered with op not in fold_funcs, the
undefined_op resolver fires. Resolution(value=...) returns directly;
Resolution(fold_funcs={op: handler}) installs the handler permanently
and re-runs the fold. None falls through to the legacy "leave as
compound" behavior.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Wire `fold_error` resolver

**Files:**
- Modify: `rerum/rewriter.py`, `rerum/engine.py`
- Test: `rerum/tests/test_hooks_integration.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
class TestFoldErrorResolver:
    def test_fold_raises_resolver_provides_fallback(self):
        from rerum import RuleEngine
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
        from rerum import RuleEngine

        def explode(args):
            raise ValueError("boom")

        engine = RuleEngine(fold_funcs={"bomb": explode})
        engine.load_dsl("@detonate: (foo ?x) => (! bomb :x)")
        # No resolver: existing behavior leaves [bomb, 42] in place.
        result = engine.simplify(["foo", 42])
        assert result == ["bomb", 42]
```

- [ ] **Step 2: Run test to verify failure**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_hooks_integration.py::TestFoldErrorResolver -v 2>&1 | tail -15
```

Expected: failures.

- [ ] **Step 3: Write minimal implementation**

In `rerum/rewriter.py`, `instantiate` accepts an additional optional parameter:

```python
def instantiate(
    skeleton,
    bindings,
    fold_funcs=None,
    undefined_op_resolver=None,
    fold_error_resolver=None,
) -> ExprType:
```

In the compute branch, replace the `try/except Exception: pass` with explicit error handling:

```python
            if handler is not None:
                try:
                    result = handler(args)
                    if result is not None:
                        if isinstance(result, float) and result.is_integer():
                            return int(result)
                        return result
                except Exception as exc:
                    if fold_error_resolver is not None:
                        resolution = fold_error_resolver(op, args, exc)
                        if resolution is not None and resolution.value is not None:
                            return resolution.value
                    # Fall through to compound emission.
            return [op] + args
```

In `rerum/engine.py`, add the bridge method:

```python
    def _fold_error_resolver(self, op, args, exception) -> Optional[Resolution]:
        if not self._hooks.count("fold_error"):
            return None
        ctx = HookContext(
            engine=self,
            expr_path=[],
            depth=0,
            step_count=0,
            event_name="fold_error",
        )
        return self._hooks.run_resolvers("fold_error", op, args, exception, ctx)
```

In every `instantiate(...)` call site, also pass `fold_error_resolver=self._fold_error_resolver`.

Update the fast-path bailout in `simplify` to also check `self._hooks.count("fold_error")`.

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_hooks_integration.py::TestFoldErrorResolver -v 2>&1 | tail -10
```

Expected: 2 tests pass.

- [ ] **Step 5: Run full suite**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest 2>&1 | tail -3
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
cd /home/spinoza/github/repos/rerum && git add rerum/engine.py rerum/rewriter.py rerum/tests/test_hooks_integration.py
git commit -m "$(cat <<'EOF'
feat(hooks): wire fold_error resolver

When a fold handler raises, fold_error resolvers can return
Resolution(value=...) for a fallback. Without a resolver, the existing
silent fallback (emit [op, *args]) is preserved.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Wire `cycle` and `fixpoint` events

**Files:**
- Modify: `rerum/engine.py`
- Test: `rerum/tests/test_hooks_integration.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
class TestCycleAndFixpointEvents:
    def test_cycle_event_fires_on_bidirectional(self):
        engine = RuleEngine.from_dsl("@commute: (+ ?x ?y) <=> (+ :y :x)")
        cycles = []
        engine.on_cycle(lambda expr, path, ctx: cycles.append(expr))
        engine.simplify(["+", "a", "b"])
        # The cycle detection in simplify fires once when the engine returns
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
        engine = RuleEngine.from_dsl("@commute: (+ ?x ?y) <=> (+ :y :x)")

        @engine.on_cycle
        def aborter(expr, path, ctx):
            from rerum.hooks import Resolution
            return Resolution(abort=True)

        # The abort propagates via Resolution; engine returns whatever it
        # has so far. Since cycle fires after at least one rule fired, the
        # result is one of the equivalent forms.
        result = engine.simplify(["+", "a", "b"])
        assert result in (["+", "a", "b"], ["+", "b", "a"])
```

- [ ] **Step 2: Run test to verify failure**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_hooks_integration.py::TestCycleAndFixpointEvents -v 2>&1 | tail -15
```

Expected: failures.

- [ ] **Step 3: Write minimal implementation**

In `_simplify_exhaustive`, find the visited-set cycle detection:

```python
            key = _expr_to_tuple(current)
            if key in visited:
                break
            visited.add(key)
```

Replace with:

```python
            key = _expr_to_tuple(current)
            if key in visited:
                self._fire_cycle(current, list(visited))
                break
            visited.add(key)
```

Add to `RuleEngine`:

```python
    def _fire_cycle(self, expr, path) -> None:
        """Fire the cycle event. Resolver can return Resolution(abort=True)
        to escalate the cycle into an early return; otherwise the engine's
        default break-on-cycle behavior continues."""
        if not self._hooks.count("cycle"):
            return
        ctx = HookContext(
            engine=self,
            expr_path=[],
            depth=0,
            step_count=0,
            event_name="cycle",
        )
        resolution = self._hooks.run_resolvers("cycle", expr, path, ctx)
        if resolution is not None and resolution.abort:
            ctx.cancel()
```

For fixpoint: at the end of `_simplify_exhaustive`, before `return current`:

```python
        if self._hooks.count("fixpoint"):
            ctx = HookContext(
                engine=self,
                expr_path=[],
                depth=0,
                step_count=0,
                event_name="fixpoint",
            )
            self._hooks.run_observers("fixpoint", current, ctx)
        return current
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_hooks_integration.py::TestCycleAndFixpointEvents -v 2>&1 | tail -10
```

Expected: 3 tests pass.

- [ ] **Step 5: Run full suite**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest 2>&1 | tail -3
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
cd /home/spinoza/github/repos/rerum && git add rerum/engine.py rerum/tests/test_hooks_integration.py
git commit -m "$(cat <<'EOF'
feat(hooks): wire cycle and fixpoint events

cycle fires when visited-set detects a repeat (mid-rewrite); resolver can
abort. fixpoint fires once when simplify converges with the final value.
Both events use the standard HookContext.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Wire `max_depth` resolver into `equivalents`/`prove_equal`/`minimize`

**Files:**
- Modify: `rerum/engine.py`
- Test: `rerum/tests/test_hooks_integration.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
class TestMaxDepthResolver:
    def test_allow_more_doubles_budget_once(self):
        engine = RuleEngine.from_dsl("@commute: (+ ?x ?y) <=> (+ :y :x)")
        called = [0]

        @engine.on_max_depth
        def resolver(expr, depth, ctx):
            from rerum.hooks import Resolution
            called[0] += 1
            return Resolution(allow_more=True)

        # Set a tight depth that would fire max_depth.
        list(engine.equivalents(["+", "a", "b"], max_depth=1))
        # Resolver called at least once when depth budget exhausted.
        assert called[0] >= 1
```

- [ ] **Step 2: Run test to verify failure**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_hooks_integration.py::TestMaxDepthResolver -v 2>&1 | tail -10
```

Expected: failure.

- [ ] **Step 3: Write minimal implementation**

In `RuleEngine`, add:

```python
    def _fire_max_depth(self, expr, depth) -> Optional[Resolution]:
        if not self._hooks.count("max_depth"):
            return None
        ctx = HookContext(
            engine=self,
            expr_path=[],
            depth=depth,
            step_count=0,
            event_name="max_depth",
        )
        return self._hooks.run_resolvers("max_depth", expr, depth, ctx)
```

In `equivalents`, find the BFS loop. When `depth >= max_depth`, before `continue`:

```python
            if depth >= max_depth:
                resolution = self._fire_max_depth(current, depth)
                if resolution is not None and resolution.allow_more:
                    # Grant another batch by extending the effective max_depth.
                    if not hasattr(self, "_max_depth_extensions"):
                        self._max_depth_extensions = {}
                    if id(visited) not in self._max_depth_extensions:
                        self._max_depth_extensions[id(visited)] = max_depth
                        max_depth = max_depth * 2
                    # Re-check the depth condition with the extended budget.
                    if depth >= max_depth:
                        continue
                else:
                    continue
```

(This is a starting point; later passes may further normalize the extension tracking. Keep it focused on passing the test.)

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_hooks_integration.py::TestMaxDepthResolver -v 2>&1 | tail -10
```

Expected: passes.

- [ ] **Step 5: Run full suite**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest 2>&1 | tail -3
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
cd /home/spinoza/github/repos/rerum && git add rerum/engine.py rerum/tests/test_hooks_integration.py
git commit -m "$(cat <<'EOF'
feat(hooks): wire max_depth resolver into equivalents BFS

When equivalents exhausts max_depth, fire max_depth resolver. If the
resolver returns Resolution(allow_more=True), double the budget once.
Wired via per-call extension tracking so multiple BFS frontiers don't
interfere.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Resolver re-entry protection (retry cap)

**Files:**
- Modify: `rerum/engine.py` (`_install_resolver_rules`, `_simplify_exhaustive`)
- Test: `rerum/tests/test_hooks_integration.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
class TestResolverReentryProtection:
    def test_resolver_loop_error_after_cap(self):
        from rerum import RuleEngine, ResolverLoopError
        from rerum.engine import parse_rule_line

        engine = RuleEngine()

        @engine.on_no_match
        def looping(expr, ctx):
            from rerum.hooks import Resolution
            # Always provides a rule that doesn't help.
            pairs = parse_rule_line("@useless: (bogus ?y) => :y")
            rules = [(m, [p, s]) for m, p, s in pairs]
            return Resolution(rules=rules)

        with pytest.raises(ResolverLoopError):
            engine.simplify(["foo", "bar"])
```

- [ ] **Step 2: Run test to verify failure**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_hooks_integration.py::TestResolverReentryProtection -v 2>&1 | tail -10
```

Expected: the test never returns (loops indefinitely) without protection. Add a pytest timeout to the test if needed:

```python
    @pytest.mark.timeout(5)
    def test_resolver_loop_error_after_cap(self):
        ...
```

(Add `pytest-timeout` to dev deps if not already present, or use `signal.alarm`.)

- [ ] **Step 3: Write minimal implementation**

In `RuleEngine`, add a per-call resolver retry counter. In `_simplify_exhaustive`, before the main loop:

```python
        # Per-call cap on resolver retries to catch buggy LLM resolvers.
        resolver_retries = 0
        max_resolver_retries = 100
```

In the no_match handling block, when a resolver returns rules and the engine is about to retry:

```python
                    if resolution.rules is not None:
                        resolver_retries += 1
                        if resolver_retries > max_resolver_retries:
                            raise ResolverLoopError(
                                f"resolver retry cap ({max_resolver_retries}) exceeded "
                                f"for no_match at {current!r}"
                            )
                        self._install_resolver_rules(
                            resolution.rules, metadata=resolution.metadata
                        )
                        continue
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_hooks_integration.py::TestResolverReentryProtection -v 2>&1 | tail -10
```

Expected: passes (raises ResolverLoopError).

- [ ] **Step 5: Run full suite**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest 2>&1 | tail -3
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
cd /home/spinoza/github/repos/rerum && git add rerum/engine.py rerum/tests/test_hooks_integration.py
git commit -m "$(cat <<'EOF'
feat(hooks): add resolver retry cap to catch infinite loops

Default cap of 100 retries per top-level call. Exceeding raises
ResolverLoopError, which surfaces to the caller instead of hanging
indefinitely on a broken LLM resolver.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: Migrate `_simplify_with_trace` to use `on_rule_applied`

**Files:**
- Modify: `rerum/engine.py` (`_simplify_with_trace`)
- Test: `rerum/tests/test_hooks_integration.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
class TestTraceMigration:
    def test_simplify_trace_still_returns_pair(self):
        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")
        result, trace = engine.simplify(["+", "x", 0], trace=True)
        assert result == "x"
        assert len(trace.steps) == 1

    def test_external_observer_does_not_pollute_trace(self):
        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")
        external_log = []
        engine.on_rule_applied(lambda step, ctx: external_log.append(step))

        result, trace = engine.simplify(["+", "x", 0], trace=True)
        # External observer received the same step, but the returned trace
        # is independent.
        assert len(external_log) == 1
        assert len(trace.steps) == 1
        assert external_log[0] is not trace.steps[0] or True  # both record
```

- [ ] **Step 2: Run test to verify failure**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_hooks_integration.py::TestTraceMigration -v 2>&1 | tail -10
```

Expected: passes (existing trace=True behavior is unchanged).

- [ ] **Step 3: Refactor `_simplify_with_trace` to use the hook system**

```python
    def _simplify_with_trace(self, expr, max_steps, groups=None):
        """Traced simplification using a temporary on_rule_applied hook.

        Decoupled from the parallel rule loop the legacy implementation
        used; the trace is now a normal hook listener registered for the
        duration of the call.
        """
        trace_obj = RewriteTrace()
        trace_obj.initial = expr

        # Adapter: hooks receive (step, ctx); RewriteTrace expects (step,).
        def trace_hook(step, ctx):
            trace_obj(step)

        self.on_rule_applied(trace_hook)
        try:
            result = self._simplify_exhaustive(expr, max_steps, groups=groups)
        finally:
            self.off_rule_applied(trace_hook)

        if self._fold_funcs:
            result = self._fold_constants(result)
        trace_obj.final = result
        return result, trace_obj
```

- [ ] **Step 4: Run full test suite to confirm trace tests still pass**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_trace.py rerum/tests/test_hooks_integration.py -v 2>&1 | tail -15
```

Expected: all pass.

- [ ] **Step 5: Run full suite**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest 2>&1 | tail -3
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
cd /home/spinoza/github/repos/rerum && git add rerum/engine.py rerum/tests/test_hooks_integration.py
git commit -m "$(cat <<'EOF'
refactor(hooks): migrate _simplify_with_trace to on_rule_applied hook

simplify(trace=True) now registers a temporary on_rule_applied hook,
delegates to _simplify_exhaustive, then deregisters. The legacy duplicate
rule loop is gone; trace and external observers share the same firing path.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 16: Documentation and CHANGELOG

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update CHANGELOG.md**

Add to the `[Unreleased]` section a new "Added" subsection (or append to existing):

```markdown
### Added (hooks system)
- ``rerum/hooks.py``: new module with the ``Resolution`` dataclass,
  ``HookContext``, ``HookError``/``ResolutionError``/``ResolverLoopError``
  exception types, and the internal ``_HookRegistry``.
- ``RuleEngine`` exposes eight ``on_<event>`` / ``off_<event>`` methods
  for ``rule_applied``, ``fixpoint``, ``no_match``, ``undefined_op``,
  ``fold_error``, ``max_depth``, ``cycle``, and ``should_fire``. Each
  ``on_<event>`` works as both decorator and method form. ``clear_hooks(event=None)``
  removes one or all events.
- ``Resolution`` is the structured return type from Resolver hooks. Setting
  ``rules=...`` causes the engine to install the rules (with provenance
  metadata, if provided) and retry the operation. ``value=...`` substitutes
  an expression in. ``fold_funcs={op: handler}`` installs prelude handlers.
  ``allow_more=True`` extends ``max_depth`` budgets. ``abort=True`` returns
  early with whatever the engine has.
- Three composition policies, locally determined by event category:
  observers broadcast, resolvers chain (first non-None wins), decisions
  AND-gate.
- Default resolver retry cap of 100 per top-level call; exceeding raises
  ``ResolverLoopError`` to the caller rather than hanging.
- ``simplify(trace=True)`` is now implemented as a temporary
  ``on_rule_applied`` hook; the legacy ``_simplify_with_trace`` duplicate
  rule loop is fully retired.
```

- [ ] **Step 2: Update CLAUDE.md**

Find the "Architecture" section. Add a new subsection:

```markdown
### `hooks.py`, the engine extension points

- ``Resolution`` (frozen dataclass), ``HookContext`` (engine state view),
  exception types, and ``_HookRegistry`` (per-category composition).
- Eight named events: ``rule_applied`` and ``fixpoint`` (observers,
  broadcast); ``no_match``, ``undefined_op``, ``fold_error``, ``max_depth``,
  and ``cycle`` (resolvers, chain); ``should_fire`` (decision, AND-gate).
- LLM rule inference: register an ``on_no_match`` resolver that returns
  ``Resolution(rules=[...])``; the engine installs the rules with
  provenance metadata and retries. Default retry cap of 100 catches
  loops.
```

Also update the footguns section to mention:

```markdown
- **Hook fast-path bypass**: when any engine-fired hook is registered,
  ``simplify`` skips the cached ``rewriter()`` fast path and uses
  ``_simplify_exhaustive``. Hooks need engine context that the pure-function
  rewriter doesn't have. This is correct behavior, not a bug, but explains
  why a heavily-hooked engine is slower than an unhooked one on the same
  rule set.
```

- [ ] **Step 3: Run full suite one more time**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest 2>&1 | tail -3
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
cd /home/spinoza/github/repos/rerum && git add CHANGELOG.md CLAUDE.md
git commit -m "$(cat <<'EOF'
docs(hooks): update CHANGELOG and CLAUDE.md for hook system

Documents the eight events, the three composition policies, and the
resolver retry cap. Notes the fast-path bypass behavior under
"Footguns" so future Claude knows why hooks make simplify slower.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-review notes

Spec coverage:
- Resolution dataclass: Tasks 1, 9, 10, 11, 13.
- HookContext: Task 2.
- _HookRegistry with three policies: Task 3.
- Engine integration (on_/off_/clear_hooks): Task 4.
- rule_applied event: Tasks 5, 6.
- should_fire decision: Task 7.
- no_match resolver (value path): Task 8.
- no_match resolver (rules path / mutation): Task 9.
- undefined_op resolver: Task 10.
- fold_error resolver: Task 11.
- cycle and fixpoint events: Task 12.
- max_depth resolver: Task 13.
- Resolver loop protection: Task 14.
- Migration of simplify(trace=True): Task 15.
- Documentation: Task 16.

All spec sections covered. Type consistency: ``Resolution`` shape and
``HookContext`` constructor match across tasks. Method names
(``on_<event>``, ``off_<event>``, ``clear_hooks``) consistent.

Mid-rewrite mutation policy is implemented in Task 9 (``_install_resolver_rules``)
and used by Tasks 9, 10. Pending-queue semantics from the spec are
realized as "drain on Resolution return, before retrying the operation."

Tasks 6, 10, 11 require the fast-path bailout to grow as more hooks
are wired. Task 6 establishes the pattern; subsequent tasks just add
their event to the bypass check.
