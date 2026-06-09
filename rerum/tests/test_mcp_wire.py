"""End-to-end MCP protocol tests over in-memory streams.

These drive the REAL SDK wiring (initialize handshake, typed tool schemas,
call_tool dispatch, and the sampling bridge) without a subprocess, via the
SDK's memory-stream test harness.
"""

import json

import pytest

pytest.importorskip("mcp")

from mcp import types  # noqa: E402
from mcp.shared.memory import (  # noqa: E402
    create_connected_server_and_client_session,
)

from rerum.mcp import _build_sdk_server  # noqa: E402


def _payload(call_result):
    return json.loads(call_result.content[0].text)


@pytest.mark.asyncio
async def test_list_tools_advertises_typed_schemas():
    sdk_srv, _ = _build_sdk_server()
    async with create_connected_server_and_client_session(sdk_srv) as client:
        tools = (await client.list_tools()).tools
        by_name = {t.name: t for t in tools}
        assert len(by_name) == 18
        simplify = by_name["simplify"]
        assert simplify.description  # real docstring summary, not a stub
        schema = simplify.inputSchema
        assert schema["required"] == ["expr"]
        assert schema["properties"]["max_steps"]["type"] == "integer"
        assert set(schema["properties"]["strategy"]["enum"]) == {
            "exhaustive", "once", "bottomup", "topdown"}
        assert schema["properties"]["expr"]["description"]
        assert schema["additionalProperties"] is False


@pytest.mark.asyncio
async def test_typed_schema_enforced_over_the_wire():
    # The original live-MCP bug class (max_depth arriving as the STRING "6")
    # is now impossible over the wire: the SDK validates call arguments
    # against the declared typed schema and rejects with a crisp message.
    # Properly typed arguments flow through.
    sdk_srv, _ = _build_sdk_server()
    async with create_connected_server_and_client_session(sdk_srv) as client:
        await client.call_tool("load_rules",
                               {"text": "@az: (+ ?x 0) => :x"})
        bad = await client.call_tool(
            "simplify", {"expr": "(+ y 0)", "max_steps": "50"})
        assert bad.isError is True
        assert "not of type 'integer'" in bad.content[0].text

        good = _payload(await client.call_tool(
            "simplify", {"expr": "(+ y 0)", "max_steps": 50}))
        assert good.get("error") is None
        assert good["result"] == "y"
        assert good["converged"] is True


@pytest.mark.asyncio
async def test_unknown_parameter_rejected_over_the_wire():
    # additionalProperties: false in the declared schema makes the SDK
    # reject a misspelled parameter at the protocol layer.
    sdk_srv, _ = _build_sdk_server()
    async with create_connected_server_and_client_session(sdk_srv) as client:
        res = await client.call_tool("simplify", {"expr": "x", "max_deph": 6})
        assert res.isError is True
        assert "max_deph" in res.content[0].text


@pytest.mark.asyncio
async def test_unknown_tool_over_the_wire():
    sdk_srv, _ = _build_sdk_server()
    async with create_connected_server_and_client_session(sdk_srv) as client:
        res = _payload(await client.call_tool("blorp", {}))
        assert res["error"]["code"] == "unknown_tool"


@pytest.mark.asyncio
async def test_solve_assisted_without_sampling_capability_refuses():
    # No sampling_callback: the client does NOT advertise the sampling
    # capability, so no bridge is installed and the tool refuses honestly.
    sdk_srv, _ = _build_sdk_server()
    async with create_connected_server_and_client_session(sdk_srv) as client:
        res = _payload(await client.call_tool(
            "solve_assisted", {"expr": "(foo bar)"}))
        assert res["error"]["code"] == "sampling_unsupported"


@pytest.mark.asyncio
async def test_solve_assisted_bridges_sampling_to_client():
    # The headline agentic loop, end to end over the REAL protocol: the
    # engine gets stuck, the server sends sampling/createMessage to the
    # client, the client's 'LLM' proposes a rule, the engine installs it
    # with provenance and finishes the rewrite.
    async def sampling_callback(context, params):
        prompt = params.messages[0].content.text
        assert "stuck" in prompt.lower()
        return types.CreateMessageResult(
            role="assistant",
            content=types.TextContent(type="text",
                                      text="@foo-id: (foo ?x) => :x"),
            model="canned-test-llm",
            stopReason="endTurn",
        )

    sdk_srv, _ = _build_sdk_server()
    async with create_connected_server_and_client_session(
            sdk_srv, sampling_callback=sampling_callback) as client:
        res = _payload(await client.call_tool(
            "solve_assisted", {"expr": "(foo bar)"}))
        assert res.get("error") is None, res
        assert res["result"] == "bar"
        assert len(res["inferred_rules"]) == 1
        assert res["inferred_rules"][0]["name"] == "foo-id"
        steps = res["trace"]["steps"]
        assert any(s.get("provenance") == "llm-inferred" for s in steps)
        assert res["converged"] is True
