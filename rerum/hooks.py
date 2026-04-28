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
