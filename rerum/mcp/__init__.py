"""Rerum MCP server.

Exposes the GENERAL rerum rewriting engine to LLM agents via the Model
Context Protocol. The server contains no domain logic: it loads rules and
theories as DATA. See the revised spec
docs/superpowers/specs/2026-06-04-symbolic-reasoning-engine-design.md
(Section 5.9) and the prior MCP design
docs/superpowers/specs/2026-05-04-mcp-design.md.

This module requires the optional ``mcp`` package; install via
``pip install rerum[mcp]``. Importing this module without the SDK
installed raises an informative ImportError.
"""

PROTOCOL_VERSION = "0.8.0"

try:
    import mcp as _mcp_sdk  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "rerum.mcp requires the 'mcp' package. Install via "
        "'pip install rerum[mcp]'."
    ) from exc


def run_server(transport: str = "stdio", host: str = "127.0.0.1",
               port: int = 8765) -> None:
    """Run the rerum MCP server.

    Args:
        transport: ``"stdio"`` (default) or ``"http"``.
        host: HTTP transport bind address (ignored for stdio).
        port: HTTP transport port (ignored for stdio).
    """
    # Wired in Task 11.
    raise NotImplementedError("server entry point wired in Task 11")


__all__ = ["run_server", "PROTOCOL_VERSION"]
