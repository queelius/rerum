"""Shared pytest fixtures for the rerum test suite.

Before this file, the MCP ``RuleStore`` and ``RerumMCPServer`` were
constructed inline in many tests (the store rooted at ``tmp_path`` ~10x in
test_mcp_persistence alone). These fixtures give them one home; a test that
also needs the underlying ``tmp_path`` can request both (pytest's
``tmp_path`` is function-scoped and shared, so the store is rooted in the
same directory the test inspects).
"""

import pytest


@pytest.fixture
def store(tmp_path):
    """A fresh file-backed RuleStore rooted at the test's tmp_path."""
    from rerum.mcp.persistence import RuleStore
    return RuleStore(root=str(tmp_path))


@pytest.fixture
def mcp_server():
    """A fresh per-session MCP server (engine + store + sampler slot)."""
    from rerum.mcp.server import RerumMCPServer
    return RerumMCPServer()
