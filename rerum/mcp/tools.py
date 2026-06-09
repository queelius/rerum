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

from typing import Any, Callable, Dict, List, Optional

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


# =====================================================================
# Applying
# =====================================================================

def _stats(engine, recorder=None) -> Dict[str, Any]:
    """Lightweight, JSON-native run stats.

    ``recorder`` is an optional ``_Recorder`` carrying a wall-clock
    duration; reasoning tools (equivalents/prove_equal/minimize) call this
    without one and report just the rule count.
    """
    out: Dict[str, Any] = {"rules_in_engine": len(engine._rules)}
    if recorder is not None:
        out["duration_ms"] = round(recorder.duration_ms, 3)
    return out


def _parse(expr: str):
    try:
        return parse_sexpr(expr)
    except Exception as exc:
        raise MCPToolError(
            "parse_error", f"failed to parse expr: {exc}", cause=exc
        ) from exc


def _path_prose(initial, steps) -> str:
    """Render prose for a hand-assembled situated path.

    ``prove_equal`` and ``minimize`` build their step list out of proof
    paths / a derivation trace rather than via the live ``trace_recorder``
    hook, so this helper wires those raw ``RewriteStep`` objects into a
    fresh ``_Recorder`` and delegates to the domain-agnostic
    ``render_prose`` (``rerum.training.to_prose``). Best-effort: any
    reconstruction failure yields an empty string rather than raising.
    """
    from rerum.mcp.trace import _Recorder, render_prose
    from rerum.trace import RewriteTrace

    rec = _Recorder()
    try:
        trace = RewriteTrace()
        trace.initial = initial
        for s in steps:
            trace.add_step(s)
        rec.trace = trace
        return render_prose(rec)
    except Exception:
        return ""


def tool_simplify(engine, *, expr: str, strategy: str = "exhaustive",
                  max_steps: int = 1000,
                  groups: Optional[List[str]] = None) -> Dict[str, Any]:
    """Simplify ``expr`` to fixpoint; return result + situated trace + prose.

    The situated trace is captured via ``trace_recorder`` (a temporary
    ``on_rule_applied`` hook); ``assemble_trace`` renders it JSON-safe with
    a domain-agnostic prose answer line. The result is rendered to an
    s-expr STRING so a Fraction-valued normal form stays serializable.
    """
    from rerum.mcp.trace import assemble_trace, trace_recorder

    parsed = _parse(expr)
    initial_str = format_sexpr(parsed)
    try:
        with trace_recorder(engine, initial=parsed) as recorder:
            result = engine.simplify(
                parsed, strategy=strategy, max_steps=max_steps, groups=groups)
    except MCPToolError:
        raise
    except Exception as exc:
        raise MCPToolError("internal_error", f"simplify failed: {exc}",
                           cause=exc) from exc

    final_str = format_sexpr(result)
    trace = assemble_trace(initial=initial_str, final=final_str,
                           recorder=recorder)
    return {"result": final_str, "converged": True, "trace": trace,
            "stats": _stats(engine, recorder)}


def tool_apply_once(engine, *, expr: str,
                    groups: Optional[List[str]] = None) -> Dict[str, Any]:
    """Apply the first matching rule once; return result + situated trace + prose.

    ``engine.apply_once`` returns ``(result, metadata)``; this takes the
    expression form and renders it to a string. The trace is at most one
    step. ``changed`` reports whether the one-shot rewrite altered the
    expression.
    """
    from rerum.mcp.trace import assemble_trace, trace_recorder

    parsed = _parse(expr)
    initial_str = format_sexpr(parsed)
    try:
        with trace_recorder(engine, initial=parsed) as recorder:
            outcome = engine.apply_once(parsed, groups=groups)
    except MCPToolError:
        raise
    except Exception as exc:
        raise MCPToolError("internal_error", f"apply_once failed: {exc}",
                           cause=exc) from exc

    # apply_once returns (result, metadata); take the expression form.
    result_expr = outcome[0] if isinstance(outcome, tuple) else outcome
    final_str = format_sexpr(result_expr)
    trace = assemble_trace(initial=initial_str, final=final_str,
                           recorder=recorder)
    return {"result": final_str,
            "changed": final_str != initial_str,
            "trace": trace, "stats": _stats(engine, recorder)}


def tool_equivalents(engine, *, expr: str, max_depth: int = 10,
                     max_count: int = 100, strategy: str = "bfs",
                     include_unidirectional: bool = False,
                     groups: Optional[List[str]] = None) -> Dict[str, Any]:
    """Enumerate expressions equivalent to ``expr``.

    Every form is rendered to an s-expr STRING via ``format_sexpr`` so a
    rational-bearing equivalent (e.g. ``(/ 1 3)``) stays JSON-serializable.
    """
    parsed = _parse(expr)
    try:
        forms = list(engine.equivalents(
            parsed, max_depth=max_depth, max_count=max_count,
            strategy=strategy,
            include_unidirectional=include_unidirectional, groups=groups))
    except MCPToolError:
        raise
    except Exception as exc:
        raise MCPToolError("internal_error", f"equivalents failed: {exc}",
                           cause=exc) from exc

    return {"forms": [format_sexpr(f) for f in forms],
            "total_count": len(forms), "stats": _stats(engine)}


def tool_prove_equal(engine, *, expr_a: str, expr_b: str, max_depth: int = 10,
                     max_expressions: Optional[int] = None,
                     include_unidirectional: bool = False,
                     trace: bool = True,
                     groups: Optional[List[str]] = None) -> Dict[str, Any]:
    """Prove ``expr_a`` and ``expr_b`` equivalent via bidirectional BFS.

    On success returns the meeting form, both labeled paths (lists of
    situated step dicts from ``EqualityProof.path_a``/``path_b``), and a
    prose rendering of the combined proof. On budget exhaustion (no common
    form within ``max_depth``/``max_expressions``) returns ``proven=False``
    with a clear message -- never a partial or a hang.

    Each path step is serialized via ``step_to_dict`` (already JSON-safe:
    bindings and any Fraction-valued ``after`` are routed through
    ``_json_safe``/``format_sexpr``).
    """
    from rerum.mcp.trace import step_to_dict

    parsed_a = _parse(expr_a)
    parsed_b = _parse(expr_b)
    try:
        proof = engine.prove_equal(
            parsed_a, parsed_b, max_depth=max_depth,
            max_expressions=max_expressions, trace=trace,
            include_unidirectional=include_unidirectional, groups=groups)
    except MCPToolError:
        raise
    except Exception as exc:
        raise MCPToolError("internal_error", f"prove_equal failed: {exc}",
                           cause=exc) from exc

    if proof is None:
        # Budget exhausted (max_depth / max_expressions) with no common
        # form. Surface a clear not-found result, not a partial.
        return {"proven": False, "prose": "No proof found within budget.",
                "stats": _stats(engine)}

    out: Dict[str, Any] = {
        "proven": True,
        "common_form": format_sexpr(proof.common),
        "depth_a": proof.depth_a,
        "depth_b": proof.depth_b,
        "stats": _stats(engine),
    }
    # path_a/path_b are List[RewriteStep] (situated) when trace=True.
    path_a = proof.path_a if trace else None
    path_b = proof.path_b if trace else None
    if path_a is not None:
        out["path_a"] = [step_to_dict(s) for s in path_a]
    if path_b is not None:
        out["path_b"] = [step_to_dict(s) for s in path_b]
    # Prose over the combined path: A -> common -> B. The reversed B path
    # walks common -> B. Each is a situated RewriteStep; reconstruction is
    # best-effort.
    combined = list(path_a or []) + list(reversed(path_b or []))
    out["prose"] = _path_prose(parsed_a, combined)
    return out


def tool_minimize(engine, *, expr: str, metric: str = "size",
                  op_costs: Optional[Dict[str, float]] = None,
                  max_depth: int = 10, max_count: int = 10000,
                  include_unidirectional: bool = True,
                  groups: Optional[List[str]] = None) -> Dict[str, Any]:
    """Find the minimum-cost equivalent of ``expr``; include the derivation prose.

    ``OptimizationResult.derivation`` is a ``RewriteTrace`` from the
    original to the best form (or ``None`` when the original is already
    optimal). Its steps are serialized via ``step_to_dict`` (JSON-safe) and
    rendered to prose. Costs are JSON-native numbers; the best expression
    is rendered to an s-expr STRING.
    """
    from rerum.mcp.trace import step_to_dict

    parsed = _parse(expr)
    kwargs: Dict[str, Any] = {
        "max_depth": max_depth, "max_count": max_count,
        "include_unidirectional": include_unidirectional, "groups": groups,
    }
    if op_costs is not None:
        kwargs["op_costs"] = op_costs
    else:
        kwargs["metric"] = metric

    try:
        opt = engine.minimize(parsed, **kwargs)
    except MCPToolError:
        raise
    except Exception as exc:
        raise MCPToolError("internal_error", f"minimize failed: {exc}",
                           cause=exc) from exc

    out: Dict[str, Any] = {
        "original": format_sexpr(opt.original),
        "original_cost": opt.original_cost,
        "best": format_sexpr(opt.expr),
        "best_cost": opt.cost,
        "improvement_ratio": opt.improvement_ratio,
        "expressions_checked": opt.expressions_checked,
        "stats": _stats(engine),
    }
    # derivation is a RewriteTrace (iterable of RewriteStep) or None.
    derivation = getattr(opt, "derivation", None)
    if derivation is not None:
        steps = list(derivation)
        out["derivation"] = [step_to_dict(s) for s in steps]
        out["prose"] = _path_prose(opt.original, steps)
    else:
        out["prose"] = ""
    return out
