"""Tests for the file-backed rule set and theory store."""

import json
import pytest


class TestRuleStore:
    def test_save_then_list_then_load_roundtrip(self, tmp_path):
        from rerum import RuleEngine
        from rerum.mcp.persistence import RuleStore

        store = RuleStore(root=str(tmp_path))
        engine = RuleEngine.from_dsl(
            '@add-zero {category=identity}: (+ ?x 0) => :x'
        )

        save_res = store.save_ruleset(engine, "algebra")
        assert save_res["ok"] is True
        assert (tmp_path / "rules" / "algebra.json").exists()

        names = store.list_rulesets()
        assert "algebra" in [r["name"] for r in names]

        fresh = RuleEngine()
        load_res = store.load_ruleset(fresh, "algebra")
        assert load_res["ok"] is True
        assert "add-zero" in fresh

    def test_roundtrip_preserves_rule_semantics(self, tmp_path):
        # A stronger round-trip: the loaded engine rewrites identically.
        from rerum import RuleEngine
        from rerum.engine import format_sexpr, parse_sexpr
        from rerum.mcp.persistence import RuleStore

        store = RuleStore(root=str(tmp_path))
        src = RuleEngine.from_dsl(
            '@add-zero {category=identity}: (+ ?x 0) => :x'
        )
        store.save_ruleset(src, "algebra")

        fresh = RuleEngine()
        store.load_ruleset(fresh, "algebra")
        expr = parse_sexpr("(+ y 0)")
        assert format_sexpr(fresh.simplify(expr)) == "y"

    def test_load_missing_ruleset_raises_not_found(self, tmp_path):
        from rerum import RuleEngine
        from rerum.mcp.persistence import RuleStore
        from rerum.mcp.errors import MCPToolError

        store = RuleStore(root=str(tmp_path))
        with pytest.raises(MCPToolError) as exc:
            store.load_ruleset(RuleEngine(), "nope")
        assert exc.value.code == "not_found"

    def test_name_sanitization_rejects_traversal(self, tmp_path):
        from rerum import RuleEngine
        from rerum.mcp.persistence import RuleStore
        from rerum.mcp.errors import MCPToolError

        store = RuleStore(root=str(tmp_path))
        with pytest.raises(MCPToolError) as exc:
            store.save_ruleset(RuleEngine(), "../escape")
        assert exc.value.code == "parse_error"

    def test_name_sanitization_rejects_separators_and_dotfiles(self, tmp_path):
        from rerum import RuleEngine
        from rerum.mcp.persistence import RuleStore
        from rerum.mcp.errors import MCPToolError

        store = RuleStore(root=str(tmp_path))
        for bad in ("a/b", "sub/../x", ".hidden", "", "a\\b"):
            with pytest.raises(MCPToolError) as exc:
                store.save_ruleset(RuleEngine(), bad)
            assert exc.value.code == "parse_error", bad

    def test_load_theory_applies_to_engine(self, tmp_path):
        from rerum import RuleEngine
        from rerum.mcp.persistence import RuleStore

        store = RuleStore(root=str(tmp_path))
        theory_dir = tmp_path / "rules"
        theory_dir.mkdir(parents=True, exist_ok=True)
        (theory_dir / "arithmetic.theory.json").write_text(json.dumps({
            "+": {"ac": True, "identity": 0},
            "*": {"ac": True, "identity": 1, "annihilator": 0},
        }))

        engine = RuleEngine()
        res = store.load_theory(engine, "arithmetic")
        assert res["ok"] is True
        # The engine now carries the theory as caller-supplied DATA. (The
        # engine here exposes no with_theory wiring, so the store stashes the
        # loaded Theory on engine._theory; no operator is special-cased.)
        assert engine._theory is not None
        # The response reports the operator signature as data (JSON-safe).
        json.dumps(res)
        assert set(res["operators"]) == {"+", "*"}
        # The loaded theory is the real Theory object and reads back as data.
        assert engine._theory.is_ac("+") is True
        assert engine._theory.identity("*") == 1

    def test_load_missing_theory_raises_not_found(self, tmp_path):
        from rerum import RuleEngine
        from rerum.mcp.persistence import RuleStore
        from rerum.mcp.errors import MCPToolError

        store = RuleStore(root=str(tmp_path))
        with pytest.raises(MCPToolError) as exc:
            store.load_theory(RuleEngine(), "nope")
        assert exc.value.code == "not_found"

    def test_theory_name_sanitization_rejects_traversal(self, tmp_path):
        from rerum import RuleEngine
        from rerum.mcp.persistence import RuleStore
        from rerum.mcp.errors import MCPToolError

        store = RuleStore(root=str(tmp_path))
        with pytest.raises(MCPToolError) as exc:
            store.load_theory(RuleEngine(), "../escape")
        assert exc.value.code == "parse_error"


class TestPersistenceToolWrappers:
    def test_tool_wrappers_roundtrip(self, tmp_path):
        from rerum import RuleEngine
        from rerum.mcp.persistence import RuleStore
        from rerum.mcp.tools import (
            tool_save_ruleset, tool_load_ruleset, tool_list_rulesets,
        )

        store = RuleStore(root=str(tmp_path))
        engine = RuleEngine.from_dsl(
            '@add-zero {category=identity}: (+ ?x 0) => :x'
        )
        save = tool_save_ruleset(engine, store, name="algebra")
        assert save["ok"] is True

        listing = tool_list_rulesets(store)
        assert "algebra" in [r["name"] for r in listing["rulesets"]]

        fresh = RuleEngine()
        load = tool_load_ruleset(fresh, store, name="algebra")
        assert load["ok"] is True
        assert "add-zero" in fresh

    def test_tool_load_theory_wrapper(self, tmp_path):
        import json as _json
        from rerum import RuleEngine
        from rerum.mcp.persistence import RuleStore
        from rerum.mcp.tools import tool_load_theory

        store = RuleStore(root=str(tmp_path))
        theory_dir = tmp_path / "rules"
        theory_dir.mkdir(parents=True, exist_ok=True)
        (theory_dir / "arith.theory.json").write_text(_json.dumps(
            {"+": {"ac": True, "identity": 0}}))

        engine = RuleEngine()
        res = tool_load_theory(engine, store, name="arith")
        assert res["ok"] is True
        assert engine._theory is not None
