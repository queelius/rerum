"""Error handling for MCP tool boundaries.

Engine exceptions caught at the dispatch boundary become structured
``MCPToolError`` payloads with stable code strings the LLM can interpret
without parsing prose. ``map_exception`` is the SINGLE mapping point: tool
handlers raise (or let engine exceptions propagate) and the dispatcher maps.

Codes: parse_error, unknown_tool, unknown_rule, validation_error,
not_found, domain_error, eval_error, sampling_unsupported, resolver_loop,
engine_busy, internal_error.
"""

import os
import traceback
from typing import Any, Dict, Optional

from rerum.mcp.utils import json_safe


class MCPToolError(Exception):
    """Structured error returned to MCP clients.

    Carries a short ``code``, a human-readable ``message``, and optional
    ``details`` for the LLM to consume programmatically. ``to_dict``
    sanitizes ``details`` through ``json_safe``, so an error payload is
    JSON-serializable regardless of construction site (an example dict or a
    guard value can carry a Fraction).
    """

    def __init__(self, code: str, message: str,
                 details: Optional[Dict[str, Any]] = None,
                 *, cause: Optional[BaseException] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}
        self.cause = cause

    def to_dict(self) -> Dict[str, Any]:
        """Convert to the MCP error response shape.

        ``RERUM_MCP_DEBUG=1`` includes a ``_traceback`` field for
        development; in production the traceback is omitted.
        """
        out: Dict[str, Any] = {
            "error": {"code": self.code, "message": self.message}
        }
        if self.details:
            out["error"]["details"] = json_safe(self.details)
        if os.environ.get("RERUM_MCP_DEBUG") == "1" and self.cause:
            out["error"]["_traceback"] = "".join(
                traceback.format_exception(
                    type(self.cause), self.cause, self.cause.__traceback__
                )
            )
        return out


def map_exception(exc: BaseException,
                  context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Map an exception to an MCP error response dict.

    ``context`` (e.g. ``{"tool": name}``) is merged into the error details
    so the client can see which call failed. A ``HookError`` is UNWRAPPED:
    the engine's hook runner wraps a raising hook as ``HookError(...) from
    cause``, and the cause is what the client needs (an ``MCPToolError``
    raised inside a resolver -- e.g. sampling_unsupported from the sampling
    bridge -- keeps its own code instead of degrading to internal_error).
    """
    from rerum.engine import ExampleValidationError
    from rerum.hooks import (
        HookError, HooksError, ResolutionError, ResolverLoopError,
    )
    from rerum.numeval import NumevalDomainError, NumevalError

    def _err(code: str, message: str,
             details: Optional[Dict[str, Any]] = None,
             cause: BaseException = exc) -> Dict[str, Any]:
        merged = dict(details or {})
        if context:
            merged["context"] = context
        return MCPToolError(code, message, details=merged or None,
                            cause=cause).to_dict()

    if isinstance(exc, MCPToolError):
        # Already structured (e.g. unwrapped from a HookError below).
        if context and "context" not in exc.details:
            exc.details = {**exc.details, "context": context}
        return exc.to_dict()

    if isinstance(exc, ExampleValidationError):
        details: Dict[str, Any] = {}
        if getattr(exc, "rule_name", None) is not None:
            details["rule_name"] = exc.rule_name
        if getattr(exc, "example", None) is not None:
            details["example"] = exc.example
        return _err("validation_error", str(exc), details)

    if isinstance(exc, ResolverLoopError):
        return _err("resolver_loop", str(exc))

    if isinstance(exc, HookError):
        cause = exc.__cause__
        if cause is not None and cause is not exc:
            via = {**(context or {}),
                   "via_hook": getattr(exc, "event", None)}
            return map_exception(cause, context=via)
        return _err("internal_error", f"hook system error: {exc}")

    if isinstance(exc, (ResolutionError, HooksError)):
        return _err("internal_error", f"hook system error: {exc}")

    if isinstance(exc, NumevalDomainError):
        # Point-dependent numeric failure (e.g. log of a negative, 1/0):
        # the expression is undefined there, not a bug.
        return _err("domain_error", str(exc))

    if isinstance(exc, NumevalError):
        # Structural numeric-evaluation failure (undefined operator,
        # unbound symbol): a malformed query, distinct from a domain edge.
        return _err("eval_error", str(exc))

    if isinstance(exc, FileNotFoundError):
        return _err("not_found", str(exc))

    if isinstance(exc, ValueError):
        return _err("internal_error", str(exc))

    return _err("internal_error", f"{type(exc).__name__}: {exc}")
