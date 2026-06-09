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
