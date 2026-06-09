"""Rerum MCP server.

Exposes the GENERAL rerum rewriting engine to LLM agents via the Model
Context Protocol. The server contains no domain logic: it loads rules and
theories as DATA. See
docs/superpowers/specs/2026-06-04-symbolic-reasoning-engine-design.md
(Section 5.9).

This package imports WITHOUT the optional ``mcp`` SDK (handlers, registry,
persistence, and errors are plain Python); only the transport entry points
(``run_server`` / ``_build_sdk_server``) require it, and raise an
informative ImportError when it is absent. Install via
``pip install rerum[mcp]``.
"""

PROTOCOL_VERSION = "0.9.0"


def _require_sdk() -> None:
    try:
        import mcp  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "rerum.mcp's server transport requires the optional 'mcp' "
            "package. Install via 'pip install rerum[mcp]'."
        ) from exc


def _build_sdk_server(rerum_srv=None):
    """Construct the SDK ``Server`` wired to a ``RerumMCPServer``.

    Factored out of ``run_server`` (no transport attached) so in-process
    tests can drive the REAL protocol over memory streams. Returns
    ``(sdk_server, rerum_server)``.

    Wiring:
    - ``list_tools`` advertises the registry-derived TYPED schemas (real
      parameter names, types, enums, defaults, and docstring-sourced
      descriptions) -- the client knows the contract.
    - ``call_tool`` runs dispatch in a worker thread and, when the
      connected client advertises the SAMPLING capability, installs a
      bridge sampler so ``solve_assisted`` can round-trip rule proposals
      through the client's LLM (``sampling/createMessage``). Without the
      capability no sampler is installed and ``solve_assisted`` refuses
      with ``sampling_unsupported``.
    - Responses are ``json.dumps``-ed with ``allow_nan=False``: the
      sanitizers upstream keep payloads JSON-safe, and a leak fails loudly
      into a mapped error instead of emitting non-spec JSON.
    """
    _require_sdk()
    import asyncio
    import json

    from mcp import types
    from mcp.server.lowlevel import Server

    from rerum.mcp.errors import MCPToolError
    from rerum.mcp.registry import get_registry
    from rerum.mcp.server import RerumMCPServer

    rerum_srv = rerum_srv if rerum_srv is not None else RerumMCPServer()
    sdk_srv = Server("rerum-mcp")

    @sdk_srv.list_tools()
    async def list_tools():
        return [
            types.Tool(
                name=spec.name,
                description=spec.description,
                inputSchema=spec.input_schema,
            )
            for spec in get_registry().values()
        ]

    def _make_bridge(session, loop):
        """A synchronous ``str -> str`` sampler over the MCP sampling channel.

        Dispatch runs in a worker thread (the loop stays free), so the
        bridge posts ``create_message`` onto the loop and blocks on the
        future. Any failure surfaces as ``sampling_unsupported`` -- the
        hook runner wraps it in HookError and ``map_exception`` unwraps it
        back to its own code.
        """

        def bridge(prompt: str) -> str:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    session.create_message(
                        messages=[types.SamplingMessage(
                            role="user",
                            content=types.TextContent(type="text",
                                                      text=prompt),
                        )],
                        max_tokens=2000,
                    ),
                    loop,
                )
                result = future.result(timeout=300)
            except Exception as exc:
                raise MCPToolError(
                    "sampling_unsupported",
                    f"sampling request failed: {exc}",
                    cause=exc,
                ) from exc
            text = getattr(result.content, "text", None)
            return text if isinstance(text, str) else ""

        return bridge

    @sdk_srv.call_tool()
    async def call_tool(name, arguments):
        # Capability-gated sampling bridge: installed per-call from the
        # live session, so solve_assisted works exactly when the client
        # can serve sampling requests.
        loop = asyncio.get_running_loop()
        sampler = None
        try:
            session = sdk_srv.request_context.session
            params = getattr(session, "client_params", None)
            caps = getattr(params, "capabilities", None)
            if getattr(caps, "sampling", None) is not None:
                sampler = _make_bridge(session, loop)
        except Exception:
            sampler = None
        rerum_srv.set_sampler(sampler)

        # Worker thread keeps the event loop free for the bridge round-trip.
        result = await asyncio.to_thread(rerum_srv.call_tool, name, arguments)
        try:
            text = json.dumps(result, indent=2, allow_nan=False)
        except ValueError as exc:  # a non-finite float escaped a sanitizer
            text = json.dumps(MCPToolError(
                "internal_error",
                f"non-finite number leaked into the response: {exc}",
            ).to_dict(), indent=2)
        return [types.TextContent(type="text", text=text)]

    return sdk_srv, rerum_srv


def run_server(transport: str = "stdio") -> None:
    """Run the rerum MCP server.

    Args:
        transport: ``"stdio"`` (the only supported transport).
    """
    _require_sdk()
    import asyncio

    from mcp.server.stdio import stdio_server

    if transport != "stdio":
        raise NotImplementedError(
            f"transport {transport!r} not supported; use 'stdio'")

    sdk_srv, _ = _build_sdk_server()

    async def _run() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await sdk_srv.run(
                read_stream, write_stream,
                sdk_srv.create_initialization_options())

    asyncio.run(_run())


__all__ = ["run_server", "PROTOCOL_VERSION"]
