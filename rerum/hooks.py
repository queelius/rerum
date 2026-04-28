"""Engine-level hooks system.

Three categories with three composition policies:
  - Observer: broadcast (all run, return values ignored)
  - Resolver: chain of responsibility (first non-None Resolution wins)
  - Decision: AND-gate (every hook must return True)

The eight events (rule_applied, fixpoint, no_match, undefined_op, fold_error,
max_depth, cycle, should_fire) are fired by the engine at its natural pause
points; this module provides the data types they exchange.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

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
    dead-end. Exactly one of ``value``, ``rules``, ``fold_funcs``, or
    ``allow_more`` must be set as the primary action; ``abort`` and
    ``metadata`` are orthogonal flags that may co-exist with any primary
    action (or with each other when no primary action is set).
    """

    value: Optional["ExprType"] = None
    rules: Optional[List[Any]] = None
    fold_funcs: Optional[Dict[str, Callable]] = None
    allow_more: Optional[bool] = None
    abort: bool = False
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        # Count primary actions (abort is orthogonal, not a primary action).
        primary = [
            self.value is not None,
            self.rules is not None,
            self.fold_funcs is not None,
            self.allow_more is not None,
        ]
        n = sum(primary)
        if n == 0 and not self.abort:
            raise ResolutionError(
                "Resolution is empty: set exactly one of "
                "value/rules/fold_funcs/allow_more, or set abort=True"
            )
        if n > 1:
            raise ResolutionError(
                "Resolution is ambiguous: set exactly one of "
                "value/rules/fold_funcs/allow_more"
            )
        if self.allow_more is False:
            raise ResolutionError(
                "allow_more=False is not a valid Resolution; "
                "return None from the resolver to decline"
            )
        if self.rules is not None and len(self.rules) == 0:
            raise ResolutionError("rules must be non-empty (an empty list adds nothing)")
        if self.fold_funcs is not None and len(self.fold_funcs) == 0:
            raise ResolutionError("fold_funcs must be non-empty")


class HookContext:
    """Read access to engine state plus controlled mutation primitives.

    Constructed by the engine for each hook invocation; not user-instantiable
    in normal use (but the constructor stays public for testability).

    Hooks should treat all attributes as read-only and call ``cancel()`` to
    signal abort rather than mutating ``cancelled`` directly. The engine
    only checks ``cancelled`` after the hook returns.
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
    def expr_path(self) -> "Tuple[ExprType, ...]":
        """Ancestry from root expression to current position.

        Returned as a tuple so the path itself cannot be re-assigned or
        re-sized from inside a hook. Note that the elements (expression
        nodes) are shared references; the engine treats expressions as
        immutable values, and hooks must do the same. Mutating an
        expression node from a hook is undefined behavior.
        """
        return self._expr_path

    def cancel(self) -> None:
        """Signal the engine to abort the current rewrite. Equivalent to
        returning ``Resolution(abort=True)`` from a Resolver."""
        self.cancelled = True
