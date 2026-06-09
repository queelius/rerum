"""File-backed rule set and theory store (git-friendly).

Rule sets are stored as ``<root>/rules/<name>.json`` (the same JSON shape
``RuleEngine.to_json`` emits and ``load_rules_from_json`` consumes).
Theories are stored as ``<root>/rules/<name>.theory.json`` (the Theory
data shape from ``rerum.normalize``). Everything is DATA: this module
contains no domain logic and no domain operator literals. WHICH operators
are AC and their units come entirely from the caller's theory file.
"""

import json
import os
import re
from typing import Any, Dict, List

from rerum.mcp.errors import MCPToolError


_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


def _safe_name(name: str) -> str:
    """Reject path traversal and unsafe characters in a ruleset name.

    A name must be a single path component drawn from ``[A-Za-z0-9._-]``
    with no leading dot. This rejects ``..``, any path separator (``/`` or
    ``\\``), and dotfiles, so a name can never escape the store directory.
    """
    if not name or not _SAFE_NAME.match(name) or name.startswith("."):
        raise MCPToolError(
            "parse_error",
            f"invalid ruleset name {name!r}; use [A-Za-z0-9._-], "
            "no leading dot, no path separators",
            details={"name": name},
        )
    return name


class RuleStore:
    """Git-friendly file store for rule sets and theories.

    Default root is ``.rerum`` in the server's working directory; rule
    files live under ``<root>/rules/``.
    """

    def __init__(self, root: str = ".rerum"):
        self.root = root
        self.rules_dir = os.path.join(root, "rules")

    def _ensure_dir(self) -> None:
        os.makedirs(self.rules_dir, exist_ok=True)

    def _ruleset_path(self, name: str) -> str:
        return os.path.join(self.rules_dir, f"{_safe_name(name)}.json")

    def _theory_path(self, name: str) -> str:
        return os.path.join(self.rules_dir, f"{_safe_name(name)}.theory.json")

    def save_ruleset(self, engine, name: str) -> Dict[str, Any]:
        """Write the engine's rules to ``<name>.json`` via ``to_json``."""
        path = self._ruleset_path(name)
        self._ensure_dir()
        text = engine.to_json(name=name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return {"ok": True, "name": name, "path": path,
                "rules_saved": len(engine._rules)}

    def load_ruleset(self, engine, name: str,
                     validate_examples: bool = True) -> Dict[str, Any]:
        """Load ``<name>.json`` into the engine via ``load_rules_from_json``.

        A missing file raises a mapped ``not_found`` error rather than a raw
        traceback.
        """
        path = self._ruleset_path(name)
        if not os.path.exists(path):
            raise MCPToolError(
                "not_found", f"no ruleset named {name!r}",
                details={"name": name, "path": path,
                         "available": [r["name"]
                                       for r in self.list_rulesets()]},
            )
        before = len(engine._rules)
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        engine.load_rules_from_json(text, validate_examples=validate_examples)
        return {"ok": True, "name": name,
                "rules_added": len(engine._rules) - before}

    def list_rulesets(self) -> List[Dict[str, Any]]:
        """List saved rule sets (``*.json``, excluding ``*.theory.json``)."""
        if not os.path.isdir(self.rules_dir):
            return []
        out: List[Dict[str, Any]] = []
        for fn in sorted(os.listdir(self.rules_dir)):
            if fn.endswith(".theory.json") or not fn.endswith(".json"):
                continue
            name = fn[:-len(".json")]
            path = os.path.join(self.rules_dir, fn)
            count = None
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    count = len(json.load(fh).get("rules", []))
            except Exception:
                pass
            out.append({"name": name, "path": path, "rules": count})
        return out

    def load_theory(self, engine, name: str) -> Dict[str, Any]:
        """Load ``<name>.theory.json`` and attach it to the engine as data.

        The Theory is operator-signature DATA declaring which operators are
        AC and their units. No operator name is special-cased here.

        This worktree's engine exposes no ``with_theory`` wiring, so the
        loaded Theory is stashed on ``engine._theory`` (a plain attribute,
        forward-compatible with a future ``with_theory``). NOTE: nothing in
        this worktree yet CONSUMES ``engine._theory`` -- it is stored for
        forward-compat, not applied to any engine operation. The response
        reports the loaded operator signature but does not claim the theory
        is active. A missing file raises a mapped ``not_found`` error.
        """
        path = self._theory_path(name)
        if not os.path.exists(path):
            raise MCPToolError(
                "not_found", f"no theory named {name!r}",
                details={"name": name, "path": path},
            )
        from rerum.normalize import Theory
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        theory = Theory.from_json(text)

        # Prefer a real attachment method if one exists; otherwise stash the
        # theory on the engine without inventing engine behavior.
        attach = getattr(engine, "with_theory", None)
        if callable(attach):
            attach(theory)
        else:
            engine._theory = theory

        operators = sorted(getattr(theory, "_sig", {}))
        return {"ok": True, "name": name, "path": path,
                "operators": operators}
