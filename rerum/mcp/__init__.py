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
    """Run the rerum MCP server over stdio (HTTP declared, not implemented).

    Wires the installed ``mcp`` SDK low-level ``Server`` (mcp 1.26): a
    ``@list_tools()`` handler advertising the dispatch table and a
    ``@call_tool()`` handler delegating to ``RerumMCPServer.call_tool``.
    Every tool result is ``json.dumps``'d into a ``TextContent`` block; the
    dispatcher already returns JSON-safe dicts (errors mapped, Fractions
    rendered to strings), so the dump never raises.

    Args:
        transport: ``"stdio"`` (default). ``"http"`` is declared but not
            implemented and raises ``NotImplementedError``.
        host: HTTP transport bind address (ignored for stdio).
        port: HTTP transport port (ignored for stdio).
    """
    import asyncio
    import json
    from typing import Any, Dict, List
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server
    from mcp import types
    from rerum.mcp.server import RerumMCPServer

    rerum_srv = RerumMCPServer()
    sdk_srv: Server = Server("rerum-mcp")

    @sdk_srv.list_tools()
    async def list_tools() -> List[types.Tool]:
        return [
            types.Tool(name=name, description=f"rerum tool: {name}",
                       inputSchema={"type": "object"})
            for name in rerum_srv.list_tool_names()
        ]

    @sdk_srv.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
        result = rerum_srv.call_tool(name, arguments)
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    async def _run() -> None:
        if transport == "stdio":
            async with stdio_server() as (read_stream, write_stream):
                await sdk_srv.run(
                    read_stream, write_stream,
                    sdk_srv.create_initialization_options())
        else:  # pragma: no cover
            raise NotImplementedError(
                f"transport {transport!r} not supported; use 'stdio'")

    asyncio.run(_run())


__all__ = ["run_server", "PROTOCOL_VERSION"]
