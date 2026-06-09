"""MCP server lifecycle and registry-driven tool dispatch.

Holds the per-session ``RuleEngine`` and ``RuleStore``. Dispatch is driven
entirely by the tool REGISTRY (``rerum.mcp.registry``): the tool set, each
tool's injected dependencies, its input schema, and its validation all
derive from the ``tool_*`` handler signatures. There is no hand-maintained
dispatch table.

This module holds NO domain logic: it dispatches to the GENERAL tool
handlers, which load rules and theories as DATA. Every dispatch path
returns a JSON-safe dict -- a tool exception is mapped to a structured
error payload (never a raw traceback escaping to the transport, never a
Fraction leaking into ``json.dumps``).
"""

import threading
from typing import Any, Callable, Dict, List, Optional

from rerum import RuleEngine
from rerum.mcp.errors import MCPToolError, map_exception
from rerum.mcp.persistence import RuleStore
from rerum.mcp.registry import get_registry, validate_and_coerce


class RerumMCPServer:
    """Per-session server state: one engine, one rule store, one sampler.

    Concurrency: a ``_busy`` flag serializes engine operations within a
    session; the check-and-set is guarded by a ``threading.Lock`` because
    the transport layer runs dispatch in a worker thread (the sampling
    bridge needs the event loop free for the client round-trip), so two
    requests CAN race on the flag. A concurrent call is rejected with an
    ``engine_busy`` error rather than letting two operations mutate shared
    engine state at once. The flag is released in a ``finally`` so an
    exception never wedges the engine.
    """

    def __init__(self, store_root: str = ".rerum"):
        self.engine = RuleEngine()
        self.store = RuleStore(root=store_root)
        self._sampler: Optional[Callable[[str], str]] = None
        self._busy = False
        self._busy_lock = threading.Lock()
        # Dependency providers, looked up by the dep names the registry
        # extracted from each handler's positional parameters. Late-bound
        # so reset/set_sampler take effect on the next call.
        self._providers: Dict[str, Callable[[], Any]] = {
            "engine": lambda: self.engine,
            "store": lambda: self.store,
            "sampler": lambda: self._sampler,
        }

    def list_tool_names(self) -> List[str]:
        return sorted(get_registry())

    def set_sampler(self, sampler: Optional[Callable[[str], str]]) -> None:
        self._sampler = sampler

    def call_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        spec = get_registry().get(name)
        if spec is None:
            return MCPToolError(
                "unknown_tool", f"unknown tool {name!r}",
                details={"name": name, "available": self.list_tool_names()},
            ).to_dict()

        with self._busy_lock:
            if self._busy:
                return MCPToolError(
                    "engine_busy",
                    "another tool call is in progress on this engine",
                ).to_dict()
            self._busy = True
        try:
            validated = validate_and_coerce(spec, args)
            deps = [self._providers[dep]() for dep in spec.deps]
            return spec.handler(*deps, **validated)
        except Exception as exc:
            # map_exception handles MCPToolError instances directly and
            # unwraps HookError causes; context names the failing tool.
            return map_exception(exc, context={"tool": name})
        finally:
            with self._busy_lock:
                self._busy = False
