"""Smoke tests for the rerum.mcp module entry point."""

import pytest


class TestMCPModule:
    def test_can_import_rerum_mcp(self):
        import rerum.mcp
        assert rerum.mcp is not None

    def test_run_server_callable_exists(self):
        from rerum.mcp import run_server
        assert callable(run_server)

    def test_module_exposes_version(self):
        from rerum.mcp import PROTOCOL_VERSION
        assert isinstance(PROTOCOL_VERSION, str)
        assert PROTOCOL_VERSION


class TestAdminTools:
    def test_reset_engine_with_computation_bundle(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_reset_engine

        engine = RuleEngine.from_dsl('@r1: (a ?x) => :x')
        result = tool_reset_engine(engine, prelude="arithmetic")
        assert result["ok"] is True
        assert engine._fold_funcs is not None
        assert "+" in engine._fold_funcs
        assert len(engine._rules) == 0

    def test_reset_engine_combo_prelude(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_reset_engine

        engine = RuleEngine()
        # A combination of computation bundles via combine_preludes.
        result = tool_reset_engine(engine, prelude=["math", "predicate"])
        assert result["ok"] is True
        assert engine._fold_funcs is not None

    def test_reset_engine_rejects_domain_bundle(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_reset_engine
        from rerum.mcp.errors import MCPToolError

        engine = RuleEngine()
        with pytest.raises(MCPToolError) as exc:
            tool_reset_engine(engine, prelude="calculus")  # not a computation bundle
        assert exc.value.code == "parse_error"

    def test_get_status_reports_configuration(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_get_status

        engine = RuleEngine.from_dsl('@r1 {category=identity}: (a ?x) => :x')
        status = tool_get_status(engine)
        assert status["rules_count"] == 1
        assert status["has_fold_funcs"] is False
        assert "identity" in status["categories"]
        assert isinstance(status["hooks"], dict)
        assert isinstance(status["engine_version"], str)
        assert isinstance(status["protocol_version"], str)


class TestServerLifecycle:
    def test_server_registers_all_tools(self):
        from rerum.mcp.server import RerumMCPServer
        srv = RerumMCPServer()
        expected = {
            "load_rules", "add_rule", "list_rules", "get_rule",
            "validate_examples",
            "save_ruleset", "load_ruleset", "list_rulesets", "load_theory",
            "simplify", "apply_once", "equivalents", "prove_equal", "minimize",
            "solve_goal", "solve_assisted",
            "reset_engine", "get_status",
        }
        assert set(srv.list_tool_names()) == expected

    def test_server_call_tool_dispatches(self):
        from rerum.mcp.server import RerumMCPServer
        srv = RerumMCPServer()
        result = srv.call_tool("load_rules", {"text": "@r1: (a ?x) => :x"})
        assert result["ok"] is True

    def test_server_unknown_tool_error(self):
        from rerum.mcp.server import RerumMCPServer
        srv = RerumMCPServer()
        result = srv.call_tool("nonexistent", {})
        assert result["error"]["code"] == "parse_error"

    def test_dispatch_error_is_json_safe(self):
        """A tool exception (e.g. a Fraction-bearing parse error) must come
        back as a JSON-safe error dict, never a raw traceback to transport."""
        import json
        from rerum.mcp.server import RerumMCPServer
        srv = RerumMCPServer()
        # An invalid prelude name raises MCPToolError inside reset_engine;
        # dispatch must catch it and return a JSON-serializable error dict.
        result = srv.call_tool("reset_engine", {"prelude": "calculus"})
        assert result["error"]["code"] == "parse_error"
        json.dumps(result)  # must not raise

    def test_solve_assisted_dispatch_uses_session_sampler(self):
        from rerum.mcp.server import RerumMCPServer
        srv = RerumMCPServer()
        srv.set_sampler(lambda prompt: "@foo-id: (foo ?x) => :x")
        result = srv.call_tool("solve_assisted", {"expr": "(foo bar)"})
        assert result["result"] == "bar"
        assert result["resolver_calls"] == 1


class TestConcurrency:
    def test_engine_busy_guard(self):
        from rerum.mcp.server import RerumMCPServer
        srv = RerumMCPServer()
        srv.engine.load_dsl('@r1: (a ?x) => :x')
        srv._busy = True
        try:
            result = srv.call_tool("simplify", {"expr": "(a y)"})
        finally:
            srv._busy = False
        assert result["error"]["code"] == "engine_busy"

    def test_busy_flag_cleared_after_call(self):
        from rerum.mcp.server import RerumMCPServer
        srv = RerumMCPServer()
        srv.call_tool("load_rules", {"text": "@r1: (a ?x) => :x"})
        # The guard must release after a successful call so the next one runs.
        assert srv._busy is False
        result = srv.call_tool("simplify", {"expr": "(a y)"})
        assert result["result"] == "y"


class TestRunServerWiring:
    def test_run_server_builds_sdk_server_without_transport(self):
        """run_server must wire the real installed SDK. Smoke-test the wiring
        by patching the stdio transport so no live process is needed."""
        import rerum.mcp as mcpmod

        # Patch asyncio.run so run_server returns after building the SDK
        # server and registering tools, without opening a real stdio loop.
        import asyncio
        called = {"ran": False}
        orig_run = asyncio.run

        def fake_run(coro):
            called["ran"] = True
            coro.close()  # don't actually await the stdio server

        asyncio.run = fake_run
        try:
            mcpmod.run_server(transport="stdio")
        finally:
            asyncio.run = orig_run
        assert called["ran"] is True
