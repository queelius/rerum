"""MCP server lifecycle and tool dispatch.

Holds the per-session ``RuleEngine`` and a ``RuleStore``, plus a tool-name
to handler map. ``call_tool`` is what the MCP SDK request handler delegates
to. The transport is wired in ``run_server`` (__init__.py).

This module holds NO domain logic: it dispatches to the GENERAL tool
handlers, which load rules and theories as DATA. Every dispatch path
returns a JSON-safe dict -- a tool exception is caught and mapped to a
structured error dict (never a raw traceback escaping to the transport,
and never a Fraction leaking into ``json.dumps``).
"""

import inspect
import typing
from typing import Any, Callable, Dict, List, Optional

from rerum import RuleEngine
from rerum.mcp.errors import MCPToolError, map_exception
from rerum.mcp.persistence import RuleStore
from rerum.mcp import tools as T

_COERCE_MISS = object()


def _numeric_target(annotation):
    """Return int/float/bool if the annotation is one (or Optional of one)."""
    if annotation in (int, float, bool):
        return annotation
    for arg in typing.get_args(annotation):  # Optional[int] -> (int, NoneType)
        if arg in (int, float, bool):
            return arg
    return None


def _coerce_scalar(value: str, target):
    """Coerce a string to ``target`` (int/float/bool); _COERCE_MISS if it can't."""
    if target is bool:
        low = value.strip().lower()
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no", ""):
            return False
        return _COERCE_MISS
    try:
        return target(value)  # int("6") -> 6, float("1.5") -> 1.5
    except (TypeError, ValueError):
        return _COERCE_MISS


def _coerce_args(func, args: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce string args to the handler's annotated numeric/bool types.

    An MCP client given a permissive input schema may send every argument as
    a string (e.g. ``max_depth="6"``), which then crashes a handler that
    compares it numerically (``depth >= "6"`` -> TypeError). The tool
    signatures annotate their numeric/boolean parameters, so a string arg is
    coerced to ``int``/``float``/``bool`` when the annotation says so. A
    value that cannot be coerced is left untouched (the handler validates it).
    """
    if func is None:
        return args
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return args
    out = dict(args)
    for key, value in args.items():
        if not isinstance(value, str) or key not in params:
            continue
        target = _numeric_target(params[key].annotation)
        if target is None:
            continue
        coerced = _coerce_scalar(value, target)
        if coerced is not _COERCE_MISS:
            out[key] = coerced
    return out


class RerumMCPServer:
    """Per-session server state: one engine, one rule store, one dispatch table.

    Concurrency: a single ``_busy`` flag serializes engine operations within
    a session. ``call_tool`` rejects a concurrent call with an
    ``engine_busy`` error rather than letting two operations mutate shared
    engine state at once. The flag is released in a ``finally`` so an
    exception never wedges the engine.
    """

    def __init__(self, store_root: str = ".rerum"):
        self.engine = RuleEngine()
        self.store = RuleStore(root=store_root)
        self._sampler: Optional[Callable[[str], str]] = None
        self._busy = False
        self._tools: Dict[str, Callable] = {
            # Authoring
            "load_rules": lambda **kw: T.tool_load_rules(self.engine, **kw),
            "add_rule": lambda **kw: T.tool_add_rule(self.engine, **kw),
            "list_rules": lambda **kw: T.tool_list_rules(self.engine, **kw),
            "get_rule": lambda **kw: T.tool_get_rule(self.engine, **kw),
            "validate_examples": lambda **kw: T.tool_validate_examples(self.engine, **kw),
            # Persistence
            "save_ruleset": lambda **kw: T.tool_save_ruleset(self.engine, self.store, **kw),
            "load_ruleset": lambda **kw: T.tool_load_ruleset(self.engine, self.store, **kw),
            "list_rulesets": lambda **kw: T.tool_list_rulesets(self.store, **kw),
            "load_theory": lambda **kw: T.tool_load_theory(self.engine, self.store, **kw),
            # Applying
            "simplify": lambda **kw: T.tool_simplify(self.engine, **kw),
            "apply_once": lambda **kw: T.tool_apply_once(self.engine, **kw),
            "equivalents": lambda **kw: T.tool_equivalents(self.engine, **kw),
            "prove_equal": lambda **kw: T.tool_prove_equal(self.engine, **kw),
            "minimize": lambda **kw: T.tool_minimize(self.engine, **kw),
            # Goal solving
            "solve_goal": lambda **kw: T.tool_solve_goal(self.engine, **kw),
            # Agentic loop
            "solve_assisted": lambda **kw: T.tool_solve_assisted(
                self.engine, sampler=self._sampler, **kw),
            # Admin
            "reset_engine": lambda **kw: T.tool_reset_engine(self.engine, **kw),
            "get_status": lambda **kw: T.tool_get_status(self.engine, **kw),
        }

    def list_tool_names(self) -> List[str]:
        return list(self._tools.keys())

    def set_sampler(self, sampler: Optional[Callable[[str], str]]) -> None:
        self._sampler = sampler

    def call_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        handler = self._tools.get(name)
        if handler is None:
            return MCPToolError(
                "parse_error", f"unknown tool {name!r}",
                details={"name": name, "available": self.list_tool_names()},
            ).to_dict()
        if self._busy:
            return MCPToolError(
                "engine_busy",
                "another tool call is in progress on this engine").to_dict()
        # Coerce string args to the handler's annotated numeric/bool types
        # (an MCP client with a permissive schema may send numbers as strings).
        # The underlying handler is T.tool_<name>; use it for the annotations.
        args = _coerce_args(getattr(T, "tool_" + name, None), args)
        self._busy = True
        try:
            return handler(**args)
        except MCPToolError as exc:
            return exc.to_dict()
        except Exception as exc:
            return map_exception(exc, context={"tool": name})
        finally:
            self._busy = False
