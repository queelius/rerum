"""MCP tool handlers.

Each ``tool_*`` function is a thin orchestration over the engine. Tool
handlers contain NO domain logic and NO domain operator literals: they
validate inputs, call the engine, and shape the response. The engine is
the GENERAL rewriting engine; rules and theories are loaded as data.
Errors raise ``MCPToolError`` with a stable ``code``.

Every response is serialized with ``json.dumps`` at the server boundary.
Expression fields are rendered to s-expr STRINGS via ``format_sexpr`` and
any structured field that could carry a ``fractions.Fraction`` atom is
passed through ``_json_safe`` (the shared Group 1 helper). A Fraction is
not JSON-native, so both routes are load-bearing.
"""

from typing import Any, Dict, List, Optional

from rerum.engine import (
    ExampleValidationError,
    format_sexpr,
    parse_sexpr,
)
from rerum.mcp.errors import MCPToolError
from rerum.mcp.trace import _json_safe


# =====================================================================
# Authoring
# =====================================================================

def tool_load_rules(engine, *, text: str, format: str = "auto",
                    validate_examples: bool = True) -> Dict[str, Any]:
    """Bulk-load rules from DSL or JSON text.

    ``format`` is auto-detected when ``"auto"``/None: a leading ``{`` means
    JSON, otherwise DSL.
    """
    if format in ("auto", None):
        format = "json" if text.lstrip().startswith("{") else "dsl"

    rules_before = len(engine._rules)
    try:
        if format == "json":
            engine.load_rules_from_json(
                text, validate_examples=validate_examples)
        elif format == "dsl":
            engine.load_dsl(text, validate_examples=validate_examples)
        else:
            raise MCPToolError(
                "parse_error",
                f"unknown format {format!r}; use 'dsl' or 'json'",
                details={"format": format},
            )
    except ExampleValidationError as exc:
        raise MCPToolError(
            "validation_error", str(exc),
            details={"rule_name": getattr(exc, "rule_name", None),
                     "example": _json_safe(getattr(exc, "example", None))},
            cause=exc,
        ) from exc
    except ValueError as exc:
        raise MCPToolError("parse_error", str(exc), cause=exc) from exc

    return {"ok": True, "rules_added": len(engine._rules) - rules_before}


def tool_add_rule(engine, *, pattern: str, skeleton: str,
                  name: Optional[str] = None,
                  description: Optional[str] = None,
                  category: Optional[str] = None,
                  reasoning: Optional[str] = None,
                  examples: Optional[List[Dict[str, Any]]] = None,
                  priority: int = 0,
                  condition: Optional[str] = None,
                  tags: Optional[List[str]] = None,
                  validate_examples: bool = True) -> Dict[str, Any]:
    """Add a single rule with full metadata.

    ``pattern``/``skeleton``/``condition`` are s-expression strings; they
    are parsed with ``parse_sexpr``. This is a unidirectional rule (the
    engine's ``add_rule`` is unidirectional only); for ``<=>`` rules load
    DSL/JSON via ``tool_load_rules``.

    The returned ``rule_index`` is a SNAPSHOT: ``add_rule`` re-sorts rules by
    priority, so a later add with a different priority can shift indices and
    invalidate a previously-returned ``rule_index``. The durable handle is the
    rule ``name`` -- pass it to ``tool_get_rule`` rather than caching an index.
    """
    try:
        pat = parse_sexpr(pattern)
        skel = parse_sexpr(skeleton)
        cond = parse_sexpr(condition) if condition else None
    except Exception as exc:
        raise MCPToolError(
            "parse_error",
            f"failed to parse pattern/skeleton/condition: {exc}",
            cause=exc,
        ) from exc

    try:
        engine.add_rule(
            pattern=pat, skeleton=skel, name=name, description=description,
            priority=priority, condition=cond, tags=tags, category=category,
            reasoning=reasoning, examples=examples,
            validate_examples=validate_examples,
        )
    except ExampleValidationError as exc:
        raise MCPToolError(
            "validation_error", str(exc),
            details={"rule_name": getattr(exc, "rule_name", None),
                     "example": _json_safe(getattr(exc, "example", None))},
            cause=exc,
        ) from exc

    # add_rule re-sorts by priority, so the new rule is not necessarily last.
    # Resolve its storage index by name when named; otherwise report the tail.
    rule_index = len(engine._rules) - 1
    if name and name in engine._rule_names:
        rule_index = engine._rule_names[name]
    return {"ok": True, "rule_index": rule_index}


def tool_list_rules(engine, *, category: Optional[str] = None,
                    tag: Optional[str] = None) -> List[Dict[str, Any]]:
    """Lightweight summary of every rule, with optional filters.

    Returned fields are all JSON-native (strings, ints, bools, lists of
    strings); no expression objects appear here.
    """
    out: List[Dict[str, Any]] = []
    for idx, meta in enumerate(engine._metadata):
        if category is not None and meta.category != category:
            continue
        if tag is not None and tag not in (meta.tags or []):
            continue
        out.append({
            "rule_index": idx,
            "name": meta.name,
            "category": meta.category,
            "description": meta.description,
            "bidirectional": meta.bidirectional,
            "direction": meta.direction,
            "priority": meta.priority,
            "tags": list(meta.tags or []),
        })
    return out


def tool_get_rule(engine, *, rule_index: Optional[int] = None,
                  name: Optional[str] = None) -> Dict[str, Any]:
    """Full details for one rule. Pass either ``rule_index`` or ``name``.

    ``pattern``/``skeleton``/``condition`` are rendered to s-expr strings;
    ``examples`` and ``extra`` are passed through ``_json_safe`` because an
    example input/output or an extra field may carry a Fraction atom.
    """
    if rule_index is None and name is None:
        raise MCPToolError(
            "parse_error", "tool_get_rule requires rule_index or name")

    if name is not None:
        if name not in engine._rule_names:
            raise MCPToolError(
                "unknown_rule", f"no rule named {name!r}",
                details={"name": name,
                         "available": list(engine._rule_names.keys())},
            )
        rule_index = engine._rule_names[name]

    if rule_index < 0 or rule_index >= len(engine._rules):
        raise MCPToolError(
            "unknown_rule",
            f"rule_index {rule_index} out of range",
            details={"rule_index": rule_index},
        )

    pattern, skeleton = engine._rules[rule_index]
    meta = engine._metadata[rule_index]
    return {
        "rule_index": rule_index,
        "name": meta.name,
        "description": meta.description,
        "pattern": format_sexpr(pattern),
        "skeleton": format_sexpr(skeleton),
        "category": meta.category,
        "reasoning": meta.reasoning,
        "examples": _json_safe(meta.examples or []),
        "priority": meta.priority,
        "condition": format_sexpr(meta.condition) if meta.condition else None,
        "tags": list(meta.tags or []),
        "bidirectional": meta.bidirectional,
        "direction": meta.direction,
        "fwd_label": meta.fwd_label,
        "rev_label": meta.rev_label,
        "extra": _json_safe(dict(meta.extra or {})),
    }


def tool_validate_examples(engine) -> Dict[str, Any]:
    """Validate every example in the engine; return errors as data.

    Delegates per-rule validation to the engine's own
    ``_validate_rule_examples`` (which honours the bidirectional
    direction-skip and threads the engine's resolvers). The reported
    ``example`` passes through ``_json_safe`` so a rational-bearing example
    keeps the response JSON-serializable.
    """
    errors: List[Dict[str, Any]] = []
    for rule, meta in zip(engine._rules, engine._metadata):
        if not meta.examples:
            continue
        try:
            engine._validate_rule_examples(rule, meta)
        except ExampleValidationError as exc:
            errors.append({
                "rule_name": getattr(exc, "rule_name", meta.name),
                "example": _json_safe(getattr(exc, "example", None)),
                "message": str(exc),
            })
    return {"ok": len(errors) == 0, "errors": errors}


# =====================================================================
# Persistence (file-backed rule sets and theories)
# =====================================================================

def tool_save_ruleset(engine, store, *, name: str) -> Dict[str, Any]:
    """Persist the engine's current rules under ``name``."""
    return store.save_ruleset(engine, name)


def tool_load_ruleset(engine, store, *, name: str,
                      validate_examples: bool = True) -> Dict[str, Any]:
    """Load a saved rule set into the engine."""
    return store.load_ruleset(
        engine, name, validate_examples=validate_examples)


def tool_list_rulesets(engine, store) -> Dict[str, Any]:
    """List the rule sets available in the store."""
    return {"rulesets": store.list_rulesets()}


def tool_load_theory(engine, store, *, name: str) -> Dict[str, Any]:
    """Load a saved theory (``<name>.theory.json``) and apply it."""
    return store.load_theory(engine, name)
