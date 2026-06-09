"""Error handling for MCP tool boundaries.

Engine exceptions caught at the tool boundary become structured
``MCPToolError`` instances with stable code strings the LLM can interpret
without parsing prose.
"""

import os
import traceback
from typing import Any, Dict, Optional


class MCPToolError(Exception):
    """Structured error returned to MCP clients.

    Carries a short ``code``, a human-readable ``message``, and optional
    ``details`` for the LLM to consume programmatically.

    Codes: parse_error, unknown_rule, validation_error, not_found,
    sampling_unsupported, resolver_loop, engine_busy, internal_error.
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
        """Convert to MCP error response shape.

        ``RERUM_MCP_DEBUG=1`` includes a sanitized ``_traceback`` field for
        development; in production the traceback is omitted.
        """
        out: Dict[str, Any] = {
            "error": {"code": self.code, "message": self.message}
        }
        if self.details:
            out["error"]["details"] = self.details
        if os.environ.get("RERUM_MCP_DEBUG") == "1" and self.cause:
            out["error"]["_traceback"] = "".join(
                traceback.format_exception(
                    type(self.cause), self.cause, self.cause.__traceback__
                )
            )
        return out


def map_exception(exc: BaseException,
                  context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Map an engine exception to an MCP error response dict."""
    from rerum.engine import ExampleValidationError
    from rerum.hooks import (
        HookError, HooksError, ResolutionError, ResolverLoopError,
    )

    if isinstance(exc, ExampleValidationError):
        details: Dict[str, Any] = {}
        if getattr(exc, "rule_name", None) is not None:
            details["rule_name"] = exc.rule_name
        if getattr(exc, "example", None) is not None:
            details["example"] = exc.example
        return MCPToolError(
            "validation_error", str(exc), details=details, cause=exc
        ).to_dict()

    if isinstance(exc, ResolverLoopError):
        return MCPToolError("resolver_loop", str(exc), cause=exc).to_dict()

    if isinstance(exc, (HookError, ResolutionError, HooksError)):
        return MCPToolError(
            "internal_error", f"hook system error: {exc}", cause=exc
        ).to_dict()

    if isinstance(exc, FileNotFoundError):
        return MCPToolError("not_found", str(exc), cause=exc).to_dict()

    if isinstance(exc, ValueError):
        return MCPToolError("internal_error", str(exc), cause=exc).to_dict()

    return MCPToolError(
        "internal_error", f"{type(exc).__name__}: {exc}", cause=exc
    ).to_dict()
