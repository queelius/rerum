"""MCP tool handlers.

Each ``tool_*`` function is a thin orchestration over the engine. Tool
handlers contain NO domain logic and NO domain operator literals: they
validate inputs, call the engine, and shape the response. The engine is
the GENERAL rewriting engine; rules, theories, and goals arrive as DATA.

CONVENTIONS (the registry derives the client-facing contract from these):
- A handler's POSITIONAL (pre-``*``) parameters are server-injected
  dependencies (``engine``, ``store``, ``sampler``); its KEYWORD-ONLY
  parameters are the caller's inputs and become the tool's JSON schema
  (types from the annotations, descriptions from the ``Args:`` docstring
  section).
- Every response is a JSON object. Expression fields are rendered to
  s-expr STRINGS via ``format_sexpr`` and structured values pass through
  ``json_safe`` (a Fraction is not JSON-native). ``prose`` is a TOP-LEVEL
  field on every derivation-bearing response.
- Handlers raise ``MCPToolError`` for caller-contract violations and let
  engine exceptions propagate; the dispatch layer maps everything through
  ``map_exception`` (single mapping point).
"""

from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Union

try:  # Python 3.9 compatibility: Literal is in typing since 3.8.
    from typing import Literal
except ImportError:  # pragma: no cover
    Literal = None  # type: ignore

from rerum import (
    ARITHMETIC_PRELUDE,
    FULL_PRELUDE,
    MATH_PRELUDE,
    PREDICATE_PRELUDE,
    combine_preludes,
)
from rerum.engine import (
    ExampleValidationError,
    format_sexpr,
    parse_sexpr,
)
from rerum.mcp.errors import MCPToolError
from rerum.mcp.trace import (
    _Recorder,
    assemble_trace,
    render_prose,
    step_to_dict,
    trace_recorder,
)
from rerum.mcp.utils import json_safe


# =====================================================================
# Shared helpers
# =====================================================================

def _parse(expr: str, *, what: str = "expr"):
    """Parse an s-expr string, rejecting empty/garbage input.

    ``parse_sexpr`` is lenient ("" -> None, "(" -> []); accepting those
    would poison the engine with a ``None`` atom and produce a confident
    "Answer: None." Reject them at the boundary instead.
    """
    try:
        parsed = parse_sexpr(expr)
    except Exception as exc:
        raise MCPToolError(
            "parse_error", f"failed to parse {what}: {exc}", cause=exc
        ) from exc
    if parsed is None or parsed == [] or parsed == "":
        raise MCPToolError(
            "parse_error", f"empty or unparseable {what}",
            details={what: expr},
        )
    return parsed


def _validation_error(exc: ExampleValidationError) -> MCPToolError:
    """The single ExampleValidationError -> MCPToolError mapping."""
    return MCPToolError(
        "validation_error", str(exc),
        details={"rule_name": getattr(exc, "rule_name", None),
                 "example": json_safe(getattr(exc, "example", None))},
        cause=exc,
    )


def _stats(engine, recorder=None) -> Dict[str, Any]:
    """Lightweight, JSON-native run stats."""
    out: Dict[str, Any] = {"rules_in_engine": len(engine)}
    if recorder is not None:
        out["duration_ms"] = round(recorder.duration_ms, 3)
    return out


def _find_rule(engine, name: str):
    """Locate a rule by name via the public iter_rules surface."""
    for idx, rule, meta in engine.iter_rules():
        if meta.name == name:
            return idx, rule, meta
    return None


def _path_prose(initial, steps, final=None) -> str:
    """Render prose for a hand-assembled situated path.

    ``prove_equal`` and ``minimize`` build their step lists from proof
    paths / a derivation trace rather than via the live ``trace_recorder``
    hook. Synthetic anchor steps (kind="initial") and no-op junctions
    (before == after) are dropped so the prose narrates real moves only.

    ``final`` overrides the closing answer line. ``minimize``'s derivation
    is a path through the equivalence graph that is NOT oriented
    original->best, so its last step's ``after`` is the wrong endpoint;
    callers that know the true endpoint pass it explicitly. When omitted,
    the last real step's ``after`` is used. Best-effort: a reconstruction
    failure yields an empty string.
    """
    from rerum.trace import RewriteTrace

    try:
        trace = RewriteTrace()
        trace.initial = initial
        real = [s for s in steps
                if s.kind != "initial" and s.before != s.after]
        for s in real:
            trace.add_step(s)
        if final is not None:
            trace.final = final
        elif real:
            trace.final = real[-1].after
        elif steps:
            trace.final = steps[-1].after
        else:
            trace.final = initial
        return render_prose(trace)
    except Exception:
        return ""


# =====================================================================
# Authoring
# =====================================================================

def tool_load_rules(engine, *, text: str,
                    format: Literal["auto", "dsl", "json"] = "auto",
                    validate_examples: bool = True) -> Dict[str, Any]:
    """Bulk-load rules from DSL or JSON text.

    Loading is ATOMIC: a validation failure anywhere in the batch leaves
    the engine unchanged (no half-loaded rules).

    Args:
        text: The rule definitions, as rerum DSL or rule-set JSON.
        format: Input format; "auto" detects JSON by a leading "{".
        validate_examples: Validate rule examples at load time (needs the
            prelude configured when examples use compute forms).
    """
    if format == "auto":
        format = "json" if text.lstrip().startswith("{") else "dsl"

    rules_before = len(engine)
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
        raise _validation_error(exc) from exc
    except ValueError as exc:
        raise MCPToolError("parse_error", str(exc), cause=exc) from exc

    return {"ok": True, "rules_added": len(engine) - rules_before}


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
    """Add a single unidirectional rule with full metadata.

    For bidirectional (``<=>``) rules, load DSL/JSON via ``load_rules``.
    The returned ``rule_index`` is a SNAPSHOT: a later add with a different
    priority re-sorts storage and can invalidate it. The durable handle is
    the rule ``name``.

    Args:
        pattern: The match pattern, as an s-expr string (e.g. "(+ ?x 0)").
        skeleton: The rewrite skeleton, as an s-expr string (e.g. ":x").
        name: Optional rule name (the durable lookup handle).
        description: Optional human-readable description.
        category: Optional free-form category label.
        reasoning: Optional rationale recorded in metadata.
        examples: Optional list of {"in": sexpr, "out": sexpr} examples.
        priority: Higher-priority rules match first (default 0).
        condition: Optional guard condition, as an s-expr string.
        tags: Optional group tags (enable/disable by group).
        validate_examples: Validate the examples now (default True).
    """
    pat = _parse(pattern, what="pattern")
    skel_raw = skeleton
    try:
        skel = parse_sexpr(skel_raw)
    except Exception as exc:
        raise MCPToolError(
            "parse_error", f"failed to parse skeleton: {exc}", cause=exc
        ) from exc
    if skel is None or skel == "":
        raise MCPToolError("parse_error", "empty or unparseable skeleton",
                           details={"skeleton": skeleton})
    cond = _parse(condition, what="condition") if condition else None

    try:
        engine.add_rule(
            pattern=pat, skeleton=skel, name=name, description=description,
            priority=priority, condition=cond, tags=tags, category=category,
            reasoning=reasoning, examples=examples,
            validate_examples=validate_examples,
        )
    except ExampleValidationError as exc:
        raise _validation_error(exc) from exc

    # add_rule re-sorts by priority; resolve the new rule's index by name.
    rule_index = len(engine) - 1
    if name is not None:
        found = _find_rule(engine, name)
        if found is not None:
            rule_index = found[0]
    return {"ok": True, "rule_index": rule_index}


def tool_list_rules(engine, *, category: Optional[str] = None,
                    tag: Optional[str] = None) -> Dict[str, Any]:
    """Summarize every rule, with optional filters.

    Args:
        category: Only rules whose category equals this label.
        tag: Only rules carrying this group tag.
    """
    rules: List[Dict[str, Any]] = []
    for idx, _rule, meta in engine.iter_rules():
        if category is not None and meta.category != category:
            continue
        if tag is not None and tag not in (meta.tags or []):
            continue
        rules.append({
            "rule_index": idx,
            "name": meta.name,
            "category": meta.category,
            "description": meta.description,
            "bidirectional": meta.bidirectional,
            "direction": meta.direction,
            "priority": meta.priority,
            "tags": list(meta.tags or []),
        })
    return {"rules": rules, "count": len(rules)}


def tool_get_rule(engine, *, rule_index: Optional[int] = None,
                  name: Optional[str] = None) -> Dict[str, Any]:
    """Full details for one rule, by ``rule_index`` or ``name``.

    Args:
        rule_index: The rule's storage index (a snapshot; may shift).
        name: The rule's name (the durable handle).
    """
    if rule_index is None and name is None:
        raise MCPToolError(
            "parse_error", "tool_get_rule requires rule_index or name")

    if name is not None:
        found = _find_rule(engine, name)
        if found is None:
            raise MCPToolError(
                "unknown_rule", f"no rule named {name!r}",
                details={"name": name,
                         "available": [m.name for _, _, m
                                       in engine.iter_rules() if m.name]},
            )
        rule_index, (pattern, skeleton), meta = found
    else:
        if rule_index < 0 or rule_index >= len(engine):
            raise MCPToolError(
                "unknown_rule", f"rule_index {rule_index} out of range",
                details={"rule_index": rule_index},
            )
        pattern, skeleton = None, None
        for idx, rule, m in engine.iter_rules():
            if idx == rule_index:
                (pattern, skeleton), meta = rule, m
                break

    return {
        "rule_index": rule_index,
        "name": meta.name,
        "description": meta.description,
        "pattern": format_sexpr(pattern),
        "skeleton": format_sexpr(skeleton),
        "category": meta.category,
        "reasoning": meta.reasoning,
        "examples": json_safe(meta.examples or []),
        "priority": meta.priority,
        "condition": format_sexpr(meta.condition) if meta.condition else None,
        "tags": list(meta.tags or []),
        "bidirectional": meta.bidirectional,
        "direction": meta.direction,
        "fwd_label": meta.fwd_label,
        "rev_label": meta.rev_label,
        "extra": json_safe(dict(meta.extra or {})),
    }


def tool_validate_examples(engine) -> Dict[str, Any]:
    """Validate every rule example in the engine; return failures as data."""
    errors: List[Dict[str, Any]] = []
    for _idx, rule, meta in engine.iter_rules():
        if not meta.examples:
            continue
        try:
            engine._validate_rule_examples(rule, meta)
        except ExampleValidationError as exc:
            errors.append({
                "rule_name": getattr(exc, "rule_name", meta.name),
                "example": json_safe(getattr(exc, "example", None)),
                "message": str(exc),
            })
    return {"ok": len(errors) == 0, "errors": errors}


# =====================================================================
# Persistence (file-backed rule sets and theories)
# =====================================================================

def tool_save_ruleset(engine, store, *, name: str) -> Dict[str, Any]:
    """Persist the engine's current rules under ``name``.

    Args:
        name: Store name ([A-Za-z0-9._-], no leading dot, no separators).
    """
    return store.save_ruleset(engine, name)


def tool_load_ruleset(engine, store, *, name: str,
                      validate_examples: bool = True) -> Dict[str, Any]:
    """Load a saved rule set into the engine (atomic on failure).

    Args:
        name: The saved rule set's name.
        validate_examples: Validate examples at load time (default True).
    """
    return store.load_ruleset(
        engine, name, validate_examples=validate_examples)


def tool_list_rulesets(store) -> Dict[str, Any]:
    """List the rule sets available in the store."""
    return {"rulesets": store.list_rulesets()}


def tool_load_theory(engine, store, *, name: str) -> Dict[str, Any]:
    """Load a saved theory (``<name>.theory.json``) into the session.

    The theory (operator-signature DATA: which operators are AC, their
    units) is consumed by ``solve_goal``'s ``normalize_between``.

    Args:
        name: The saved theory's name.
    """
    return store.load_theory(engine, name)


# =====================================================================
# Applying
# =====================================================================

_FIXPOINT_STRATEGIES = ("exhaustive", "bottomup", "topdown")


def tool_simplify(engine, *, expr: str,
                  strategy: Literal["exhaustive", "once",
                                    "bottomup", "topdown"] = "exhaustive",
                  max_steps: int = 1000,
                  groups: Optional[List[str]] = None) -> Dict[str, Any]:
    """Simplify ``expr``; return result + situated trace + prose.

    ``converged`` is TRUTHFUL: True only when the engine reached a natural
    fixpoint (its fixpoint event fired), False when the step budget ran out
    first, and None for the one-shot "once" strategy where convergence does
    not apply.

    Args:
        expr: The expression to simplify, as an s-expr string.
        strategy: Rewriting strategy; "exhaustive" applies rules to fixpoint.
        max_steps: Step budget for fixpoint strategies (default 1000).
        groups: Restrict to rules in these group tags.
    """
    parsed = _parse(expr)
    initial_str = format_sexpr(parsed)
    with trace_recorder(engine, initial=parsed) as recorder:
        result = engine.simplify(
            parsed, strategy=strategy, max_steps=max_steps, groups=groups)

    final_str = format_sexpr(result)
    trace = assemble_trace(initial=initial_str, final=final_str,
                           recorder=recorder)
    converged = (recorder.converged
                 if strategy in _FIXPOINT_STRATEGIES else None)
    return {"result": final_str, "converged": converged, "trace": trace,
            "prose": render_prose(recorder.trace),
            "stats": _stats(engine, recorder)}


def tool_apply_once(engine, *, expr: str,
                    groups: Optional[List[str]] = None) -> Dict[str, Any]:
    """Apply the first matching rule once; return result + trace + prose.

    ``matched`` reports whether ANY rule matched; ``rule`` names it.
    ``changed`` reports whether the rewrite altered the expression --
    a no-op match (matched=True, changed=False) is distinguishable from
    no rule matching at all.

    Args:
        expr: The expression, as an s-expr string.
        groups: Restrict to rules in these group tags.
    """
    parsed = _parse(expr)
    initial_str = format_sexpr(parsed)
    with trace_recorder(engine, initial=parsed) as recorder:
        result_expr, meta = engine.apply_once(parsed, groups=groups)

    final_str = format_sexpr(result_expr)
    trace = assemble_trace(initial=initial_str, final=final_str,
                           recorder=recorder)
    return {"result": final_str,
            "changed": final_str != initial_str,
            "matched": meta is not None,
            "rule": meta.name if meta is not None else None,
            "trace": trace, "prose": render_prose(recorder.trace),
            "stats": _stats(engine, recorder)}


def tool_equivalents(engine, *, expr: str, max_depth: int = 10,
                     max_count: int = 100,
                     strategy: Literal["bfs", "dfs"] = "bfs",
                     include_unidirectional: bool = False,
                     groups: Optional[List[str]] = None) -> Dict[str, Any]:
    """Enumerate expressions equivalent to ``expr`` (bounded).

    Args:
        expr: The starting expression, as an s-expr string.
        max_depth: Maximum rewrite depth to explore (default 10).
        max_count: Maximum number of forms to return (default 100;
            equivalence classes grow factorially, keep this bounded).
        strategy: Exploration order, breadth-first or depth-first.
        include_unidirectional: Also traverse one-way (=>) rules.
        groups: Restrict to rules in these group tags.
    """
    parsed = _parse(expr)
    forms = list(engine.equivalents(
        parsed, max_depth=max_depth, max_count=max_count,
        strategy=strategy,
        include_unidirectional=include_unidirectional, groups=groups))
    return {"forms": [format_sexpr(f) for f in forms],
            "total_count": len(forms), "stats": _stats(engine)}


def tool_prove_equal(engine, *, expr_a: str, expr_b: str, max_depth: int = 10,
                     max_expressions: Optional[int] = None,
                     include_unidirectional: bool = False,
                     trace: bool = True,
                     groups: Optional[List[str]] = None) -> Dict[str, Any]:
    """Prove ``expr_a`` and ``expr_b`` equivalent via bidirectional BFS.

    On success returns the meeting form, both labeled paths, and a
    TWO-SIDED prose narration (each side narrated from its own endpoint to
    the common form; the sides are not merged into one chain because the
    reverse path's steps are not direction-inverted). On budget exhaustion
    returns ``proven=False`` -- never a partial or a hang.

    Args:
        expr_a: The left expression, as an s-expr string.
        expr_b: The right expression, as an s-expr string.
        max_depth: Per-side BFS depth bound (default 10).
        max_expressions: Total work budget across both frontiers
            (recommended for rich bidirectional rule sets).
        include_unidirectional: Also traverse one-way (=>) rules.
        trace: Include the labeled proof paths (default True).
        groups: Restrict to rules in these group tags.
    """
    parsed_a = _parse(expr_a, what="expr_a")
    parsed_b = _parse(expr_b, what="expr_b")
    proof = engine.prove_equal(
        parsed_a, parsed_b, max_depth=max_depth,
        max_expressions=max_expressions, trace=trace,
        include_unidirectional=include_unidirectional, groups=groups)

    if proof is None:
        return {"proven": False, "prose": "No proof found within budget.",
                "stats": _stats(engine)}

    common_str = format_sexpr(proof.common)
    out: Dict[str, Any] = {
        "proven": True,
        "common_form": common_str,
        "depth_a": proof.depth_a,
        "depth_b": proof.depth_b,
        "stats": _stats(engine),
    }
    path_a = proof.path_a if trace else None
    path_b = proof.path_b if trace else None
    if path_a is not None:
        out["path_a"] = [step_to_dict(s) for s in path_a]
    if path_b is not None:
        out["path_b"] = [step_to_dict(s) for s in path_b]

    prose_a = _path_prose(parsed_a, list(path_a or []))
    prose_b = _path_prose(parsed_b, list(path_b or []))
    out["prose"] = (
        f"Both sides reach {common_str}.\n"
        f"From {format_sexpr(parsed_a)}:\n{prose_a}\n"
        f"From {format_sexpr(parsed_b)}:\n{prose_b}"
    )
    return out


def tool_minimize(engine, *, expr: str,
                  metric: Optional[Literal["size", "depth",
                                           "ops", "atoms"]] = None,
                  op_costs: Optional[Dict[str, float]] = None,
                  max_depth: int = 10, max_count: int = 10000,
                  include_unidirectional: bool = True,
                  groups: Optional[List[str]] = None) -> Dict[str, Any]:
    """Find the minimum-cost equivalent of ``expr``; include the derivation.

    Cost precedence is the engine's: ``op_costs`` beats ``metric`` beats
    the default size metric. ``improvement_ratio`` is the fractional
    REDUCTION (0.0 = no improvement, 1.0 = fully eliminated).

    Args:
        expr: The expression to minimize, as an s-expr string.
        metric: Built-in cost metric (size/depth/ops/atoms).
        op_costs: Per-operator cost table (overrides metric).
        max_depth: Equivalence-class exploration depth (default 10).
        max_count: Work budget on expressions checked (default 10000).
        include_unidirectional: Also traverse one-way (=>) rules
            (default True; minimize is for mixed rule sets).
        groups: Restrict to rules in these group tags.
    """
    parsed = _parse(expr)
    opt = engine.minimize(
        parsed, metric=metric, op_costs=op_costs,
        max_depth=max_depth, max_count=max_count,
        include_unidirectional=include_unidirectional, groups=groups)

    out: Dict[str, Any] = {
        "original": format_sexpr(opt.original),
        "original_cost": json_safe(opt.original_cost),
        "best": format_sexpr(opt.expr),
        "best_cost": json_safe(opt.cost),
        "improvement_ratio": json_safe(opt.improvement_ratio),
        "expressions_checked": opt.expressions_checked,
        "stats": _stats(engine),
    }
    derivation = getattr(opt, "derivation", None)
    if derivation is not None:
        steps = list(derivation)
        out["derivation"] = [step_to_dict(s) for s in steps]
        out["prose"] = _path_prose(opt.original, steps, final=opt.expr)
    else:
        out["prose"] = ""
    return out


# =====================================================================
# Goal solving (wraps engine.solve)
# =====================================================================

def _compile_goal(goal: Dict[str, Any]) -> Callable[[Any], bool]:
    """Compile a caller-described goal (DATA) into a predicate.

    The goal is the CALLER's data, never a hardcoded domain predicate;
    the tool special-cases NO operator. Supported kinds:
      ``{"op_free": ["op1", ...]}`` -> True when none of those operators
      remain anywhere in the expression.
    """
    from rerum.solve import contains_op

    if not isinstance(goal, dict):
        raise MCPToolError(
            "parse_error",
            'goal must be an object, e.g. {"op_free": ["opname"]}',
            details={"goal": json_safe(goal)},
        )
    if "op_free" in goal:
        spec = goal["op_free"]
        # A string would silently iterate to a set of single CHARACTERS
        # (op_free="neg" -> {'n','e','g'}) -> contains_op finds none -> a
        # false found=True at the start node. Reject it.
        if not isinstance(spec, list) or not all(isinstance(o, str)
                                                 for o in spec):
            raise MCPToolError(
                "parse_error",
                'op_free must be a list of operator-name strings, e.g. '
                '{"op_free": ["opname"]}',
                details={"goal": json_safe(goal)},
            )
        ops = set(spec)
        return lambda e: not contains_op(e, ops)

    raise MCPToolError(
        "parse_error",
        f"unknown goal kind; supported: 'op_free'. Got keys: {sorted(goal)}",
        details={"goal": json_safe(goal)},
    )


def tool_solve_goal(engine, *, expr: str, goal: Dict[str, Any],
                    max_nodes: int = 10000,
                    normalize_between: bool = True) -> Dict[str, Any]:
    """Goal-directed best-first search via ``engine.solve``.

    The goal is DATA (e.g. eliminate the caller's named operators); the
    tool holds no domain logic. A session theory loaded via ``load_theory``
    is threaded into the search, so ``normalize_between`` canonicalizes
    nodes under the caller's operator signature. On budget exhaustion
    returns ``found=False`` -- never a partial or a hang.

    Args:
        expr: The starting expression, as an s-expr string.
        goal: The goal description, e.g. {"op_free": ["opname", ...]}.
        max_nodes: Search budget on expanded nodes (default 10000).
        normalize_between: Canonicalize nodes between steps when a session
            theory is loaded (default True).
    """
    parsed = _parse(expr)
    initial_str = format_sexpr(parsed)
    predicate = _compile_goal(goal)

    sr = engine.solve(
        parsed, predicate, max_nodes=max_nodes,
        normalize_between=normalize_between,
        theory=engine._theory)

    final_str = (format_sexpr(sr.solution)
                 if sr.solution is not None else initial_str)
    rec = _Recorder()
    rec.trace = sr.derivation
    if rec.trace is not None:
        try:
            rec.steps = [step_to_dict(s) for s in rec.trace.steps]
        except Exception:
            rec.steps = []
    trace = assemble_trace(initial=initial_str, final=final_str, recorder=rec)

    return {
        "result": final_str,
        "found": sr.found,
        "explored": sr.explored,
        "trace": trace,
        "prose": render_prose(rec.trace),
        "stats": _stats(engine),
    }


# =====================================================================
# Agentic loop (solve_assisted)
# =====================================================================

@contextmanager
def _temporary_resolver(engine, resolver):
    """Install a no_match resolver for the block; always remove it."""
    engine.on_no_match(resolver)
    try:
        yield
    finally:
        engine.off_no_match(resolver)


def tool_solve_assisted(engine, sampler, *, expr: str,
                        goal: Optional[str] = None,
                        max_steps: int = 20,
                        max_resolver_calls: int = 10,
                        strategy: Literal["exhaustive", "bottomup",
                                          "topdown"] = "exhaustive",
                        ) -> Dict[str, Any]:
    """Simplify with an LLM resolver proposing rules when the engine sticks.

    ``sampler`` is an INJECTED dependency (the session's MCP sampling
    channel). Without one this tool raises ``sampling_unsupported`` rather
    than silently degrading to plain simplify. Proposed rules are installed
    as DATA with provenance "llm-inferred" under the prelude security
    boundary; ``inferred_rules`` reports exactly the rules ACTUALLY
    installed (a deduplicated re-proposal is not double-counted).
    ``converged`` is truthful (the engine's fixpoint event).

    Args:
        expr: The expression to simplify, as an s-expr string.
        goal: Optional natural-language goal passed to the LLM prompt.
        max_steps: Step budget for the underlying simplify (default 20).
        max_resolver_calls: LLM-proposal budget per call (default 10).
        strategy: Rewriting strategy (fixpoint strategies only).
    """
    from rerum.hooks import ResolverLoopError
    from rerum.mcp.solver import make_solver_resolver

    if sampler is None:
        raise MCPToolError(
            "sampling_unsupported",
            "solve_assisted requires an LLM sampling channel; the "
            "connected MCP client does not support sampling (or no "
            "sampler is configured via set_sampler)",
        )

    parsed = _parse(expr)
    initial_str = format_sexpr(parsed)
    state: Dict[str, Any] = {}
    resolver = make_solver_resolver(
        sampler, goal=goal, max_calls=max_resolver_calls, state=state)

    before_meta_ids = {id(meta) for _, _, meta in engine.iter_rules()}

    termination: Optional[Dict[str, Any]] = None
    with _temporary_resolver(engine, resolver):
        with trace_recorder(engine, initial=parsed) as recorder:
            try:
                result = engine.simplify(parsed, strategy=strategy,
                                         max_steps=max_steps)
            except ResolverLoopError as exc:
                termination = {"reason": "resolver_loop", "detail": str(exc)}
                result = parsed

    if termination is None and state.get("last_termination"):
        termination = {"reason": state["last_termination"]}

    # TRUTHFUL inferred_rules: exactly the rules that are NEW since the
    # snapshot and carry the llm-inferred provenance (a deduped re-proposal
    # was never installed, so it does not appear).
    inferred: List[Dict[str, Any]] = []
    for idx, _rule, meta in engine.iter_rules():
        if id(meta) in before_meta_ids:
            continue
        if (meta.extra or {}).get("provenance") == "llm-inferred":
            inferred.append({
                "name": meta.name,
                "category": meta.category,
                "rule_index": idx,
                "round": (meta.extra or {}).get("round"),
            })

    final_str = format_sexpr(result)
    trace = assemble_trace(initial=initial_str, final=final_str,
                           recorder=recorder)
    out: Dict[str, Any] = {
        "result": final_str,
        "converged": recorder.converged,
        "trace": trace,
        "prose": render_prose(recorder.trace),
        "inferred_rules": inferred,
        "resolver_calls": state.get("call_count", 0),
        "stats": _stats(engine, recorder),
    }
    if termination is not None:
        out["termination"] = termination
    return out


# =====================================================================
# Admin
# =====================================================================

# Computation bundles ONLY -- no domain bundle (the general-engine
# principle). Names map DIRECTLY to the prelude dicts (data, not attribute
# indirection).
_PRELUDE_BUNDLES: Dict[str, Optional[Dict[str, Any]]] = {
    "arithmetic": ARITHMETIC_PRELUDE,
    "math": MATH_PRELUDE,
    "predicate": PREDICATE_PRELUDE,
    "full": FULL_PRELUDE,
    "none": None,
}


def _resolve_prelude(prelude):
    """Resolve a prelude spec (DATA) to fold_funcs or None.

    A single computation-bundle name or a list combined via
    ``combine_preludes``. There is NO domain bundle; an unknown name is a
    parse_error.
    """
    names = [prelude] if isinstance(prelude, str) else list(prelude)
    resolved = []
    for nm in names:
        if nm not in _PRELUDE_BUNDLES:
            raise MCPToolError(
                "parse_error",
                f"unknown prelude {nm!r}; valid computation bundles: "
                f"{sorted(_PRELUDE_BUNDLES)} (there is no domain bundle)",
                details={"prelude": nm})
        bundle = _PRELUDE_BUNDLES[nm]
        if bundle is not None:
            resolved.append(bundle)
    if not resolved:
        return None
    if len(resolved) == 1:
        return resolved[0]
    return combine_preludes(*resolved)


def tool_reset_engine(engine, *,
                      prelude: Union[str, List[str]] = "none",
                      ) -> Dict[str, Any]:
    """Reset the session engine; optionally install a computation prelude.

    Clears rules, hooks, theory, groups, and fold functions via the
    engine's public ``reset``.

    Args:
        prelude: A computation bundle name (arithmetic/math/predicate/
            full/none) or a list of names to combine. No domain bundle
            exists; domains are rules loaded as data.
    """
    engine.reset(_resolve_prelude(prelude))
    return {"ok": True}


def tool_get_status(engine) -> Dict[str, Any]:
    """Inspection: how is the session engine currently configured?"""
    from rerum import __version__ as engine_version
    from rerum.mcp import PROTOCOL_VERSION

    metas = [meta for _, _, meta in engine.iter_rules()]
    categories = sorted({m.category for m in metas if m.category})
    groups = sorted({t for m in metas for t in (m.tags or [])})
    return {
        "rules_count": len(engine),
        "has_fold_funcs": engine.has_fold_funcs(),
        "has_theory": engine.has_theory(),
        "hooks": engine.hook_counts(),
        "categories": categories,
        "groups": groups,
        "engine_version": engine_version,
        "protocol_version": PROTOCOL_VERSION,
    }
