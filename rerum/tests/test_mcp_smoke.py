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
            "check_numeric_equiv",
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
        assert result["error"]["code"] == "unknown_tool"
        assert "available" in result["error"]["details"]

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

    def test_busy_flag_cleared_when_handler_raises(self):
        # The finally-clear must hold even when a handler raises a
        # non-MCPToolError: a raising sampler propagates through solve_assisted,
        # the dispatch maps it to a JSON-safe error, and _busy is released so
        # the engine is not wedged for the next call.
        import json
        from rerum.mcp.server import RerumMCPServer
        srv = RerumMCPServer()

        def boom(prompt):
            raise RuntimeError("kaboom")

        srv.set_sampler(boom)
        result = srv.call_tool("solve_assisted", {"expr": "(foo bar)"})
        assert "error" in result
        json.dumps(result)  # mapped error is JSON-safe, no traceback leak
        assert srv._busy is False  # released despite the raise
        # The engine is not wedged: a normal call still runs.
        srv.set_sampler(None)
        srv.call_tool("load_rules", {"text": "@r: (a ?x) => :x"})
        ok = srv.call_tool("simplify", {"expr": "(a y)"})
        assert ok["result"] == "y"


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


class TestArgCoercion:
    """An MCP client given a permissive input schema may send numeric/bool
    arguments as strings. The dispatch coerces them to the handler's annotated
    types, so a tool does not crash on e.g. max_depth='6'. Regression for a
    bug found by driving prove_equal over a live MCP connection:
    "'>=' not supported between instances of 'int' and 'str'"."""

    def test_string_numeric_and_bool_args_are_coerced(self):
        from rerum.mcp.server import RerumMCPServer
        srv = RerumMCPServer()
        srv.call_tool("reset_engine", {"prelude": "full"})
        srv.call_tool("load_rules", {"format": "dsl", "text":
            "@distribute: (* ?a (+ ?b ?c)) <=> (+ (* :a :b) (* :a :c))\n"
            "@mul-one: (* ?x 1) <=> :x"})
        # Every argument sent as a string, as a bare-schema client would.
        result = srv.call_tool("prove_equal", {
            "expr_a": "(* x (+ y 1))", "expr_b": "(+ (* x y) x)",
            "max_depth": "6", "include_unidirectional": "false",
            "trace": "true"})
        assert "error" not in result, result
        assert result["proven"] is True

    def test_string_max_steps_coerced_for_simplify(self):
        from rerum.mcp.server import RerumMCPServer
        srv = RerumMCPServer()
        srv.call_tool("load_rules", {"format": "dsl",
                                     "text": "@az: (+ ?x 0) => :x"})
        result = srv.call_tool("simplify", {"expr": "(+ y 0)",
                                            "max_steps": "50"})
        assert "error" not in result
        assert result["result"] == "y"

    def test_uncoercible_string_left_for_handler_to_validate(self):
        # A non-numeric string for a numeric param is left untouched (not
        # silently zeroed); the handler surfaces a clean error, not a crash.
        from rerum.mcp.server import RerumMCPServer
        srv = RerumMCPServer()
        srv.call_tool("load_rules", {"format": "dsl",
                                     "text": "@az: (+ ?x 0) => :x"})
        result = srv.call_tool("simplify", {"expr": "(+ y 0)",
                                            "max_steps": "lots"})
        assert "error" in result  # mapped error, not an unhandled traceback


class TestOptionalSdkBoundary:
    def test_package_imports_without_sdk_only_transport_needs_it(self):
        # 0.9.0: rerum.mcp imports WITHOUT the mcp SDK (handlers, registry,
        # persistence, errors are plain Python); only the transport entry
        # points require it. Simulate absence and reimport the package.
        import importlib
        import subprocess
        import sys
        code = (
            "import sys; sys.modules['mcp'] = None\n"
            "import rerum.mcp\n"
            "import rerum.mcp.tools, rerum.mcp.registry\n"
            "print('import-ok')\n"
            "try:\n"
            "    rerum.mcp.run_server()\n"
            "except ImportError as e:\n"
            "    print('run-server-raises' if 'rerum[mcp]' in str(e) else 'wrong-error')\n"
        )
        out = subprocess.run([sys.executable, "-c", code],
                             capture_output=True, text=True)
        assert "import-ok" in out.stdout, out.stderr
        assert "run-server-raises" in out.stdout, out.stdout + out.stderr


class TestConcurrencyReal:
    """The busy guard's only genuine proof: two real threads, one holding the
    engine mid-handler while the other arrives. (The other TestConcurrency
    cases set _busy by hand; this exercises the lock under contention.)"""

    def test_busy_guard_under_thread_contention(self):
        import threading
        from rerum.mcp.server import RerumMCPServer

        srv = RerumMCPServer()
        srv.engine.load_dsl("@s: (foo ?x) => :x")  # does NOT match (stuck x)

        entered = threading.Event()
        proceed = threading.Event()

        def blocking_sampler(prompt):
            entered.set()
            proceed.wait(timeout=5)
            return "NONE"  # propose nothing; solve_assisted then converges

        srv.set_sampler(blocking_sampler)
        results = {}

        def call_a():
            # Gets stuck on (stuck x), fires the (blocking) resolver -> holds
            # the engine until proceed is set.
            results["a"] = srv.call_tool("solve_assisted", {"expr": "(stuck x)"})

        ta = threading.Thread(target=call_a)
        ta.start()
        assert entered.wait(timeout=5)  # A is now inside, holding the engine

        # B arrives while A holds the engine.
        results["b"] = srv.call_tool("simplify", {"expr": "(foo bar)"})

        proceed.set()
        ta.join(timeout=5)

        assert results["b"]["error"]["code"] == "engine_busy"
        assert srv._busy is False  # released after A finished
        # The engine is usable again (not wedged).
        assert srv.call_tool("simplify", {"expr": "(foo bar)"})["result"] == "bar"
