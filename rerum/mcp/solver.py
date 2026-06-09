"""LLM-resolver factory for the solve_assisted tool.

Builds a closure suitable for ``engine.on_no_match()`` that:
1. Counts calls and enforces the per-solve cap.
2. Asks the connected LLM (via the ``sampler`` callable) for a rule.
3. Parses the reply via ``parse_rule_line``; on validation failure, retries
   once with the error in the prompt; if still failing, returns None.
4. Wraps a successful parse in ``Resolution(rules=..., metadata={
       provenance: "llm-inferred", via_solve: True, round: N})``.

This is GENERAL: the resolver never special-cases a domain operator. The
LLM's reply is parsed as DATA and installed under the engine's existing
prelude security boundary -- a proposed rule can only invoke operations
already in ``engine._fold_funcs``; the resolver injects no new prelude
code. WHICH operators a proposed rule may use is the engine's prelude, not
anything this module decides.
"""

from typing import Any, Callable, Dict, Optional

from rerum.engine import ExampleValidationError, parse_rule_line
from rerum.hooks import Resolution


def make_solver_resolver(sampler: Callable[[str], str], *,
                         goal: Optional[str] = None, max_calls: int = 10,
                         state: Dict[str, Any]) -> Callable[[Any, Any], Optional[Resolution]]:
    """Build a no_match resolver that delegates to ``sampler`` for new rules.

    ``state`` is a shared dict the caller owns; the resolver records
    ``call_count``, the list of ``inferred_rules`` (JSON-native dicts), and
    a ``last_termination`` code (``"resolver_budget_exhausted"`` when the
    per-solve cap is hit). The caller reads these after the rewrite to shape
    the response.
    """
    state.setdefault("call_count", 0)
    state.setdefault("inferred_rules", [])
    state.setdefault("last_termination", None)

    def resolver(expr, ctx):
        if state["call_count"] >= max_calls:
            state["last_termination"] = "resolver_budget_exhausted"
            return None
        state["call_count"] += 1
        round_num = state["call_count"]

        engine = ctx.engine
        prompt = _build_prompt(expr, goal, engine)

        reply = sampler(prompt)
        rule_pairs = _try_parse_rule_reply(reply)
        if rule_pairs is None:
            return None

        try:
            _validate_pairs(rule_pairs, engine)
        except ExampleValidationError as exc:
            retry_prompt = (
                prompt + "\n\nYour previous reply produced this error: "
                         f"{exc}\nRevise and try again.")
            reply2 = sampler(retry_prompt)
            rule_pairs = _try_parse_rule_reply(reply2)
            if rule_pairs is None:
                return None
            try:
                _validate_pairs(rule_pairs, engine)
            except ExampleValidationError:
                return None

        rules_for_resolution = [(meta, [pat, skel]) for meta, pat, skel in rule_pairs]
        for meta, pat, skel in rule_pairs:
            state["inferred_rules"].append({
                "name": meta.name, "category": meta.category,
                "dsl": _rule_to_dsl(meta, pat, skel), "round": round_num})

        return Resolution(rules=rules_for_resolution, metadata={
            "provenance": "llm-inferred", "via_solve": True, "round": round_num})

    return resolver


def _build_prompt(expr, goal, engine) -> str:
    from rerum.engine import format_sexpr
    expr_str = format_sexpr(expr)
    rules_count = len(engine._rules)
    categories = sorted({m.category for m in engine._metadata if m.category})
    cats_str = ", ".join(categories) if categories else "(none)"
    return (
        "The rewrite engine is stuck. Propose ONE rewrite rule that "
        "would help.\n\n"
        f"Goal: {goal or '(no goal specified)'}\n"
        f"Stuck at: {expr_str}\n"
        f"Rules currently in engine: {rules_count} (categories: {cats_str}).\n\n"
        "Reply with a single rule in DSL format, e.g.:\n"
        "  @my-rule {category=identity}: (foo ?x) => :x\n\n"
        "If you cannot propose a useful rule, reply: NONE")


def _try_parse_rule_reply(reply: str):
    if not reply:
        return None
    text = reply.strip()
    if text.upper() == "NONE":
        return None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            try:
                pairs = parse_rule_line(stripped)
            except Exception:
                return None
            return pairs or None
    return None


def _validate_pairs(rule_pairs, engine) -> None:
    from rerum.engine import _validate_example
    for meta, pat, skel in rule_pairs:
        if not meta.examples:
            continue
        for example in meta.examples:
            direction = example.get("direction", "fwd")
            if meta.bidirectional and direction != meta.direction:
                continue
            _validate_example(pat, skel, meta, example, engine._fold_funcs or {})


def _rule_to_dsl(meta, pat, skel) -> str:
    from rerum.engine import format_sexpr
    name_part = f"@{meta.name}" if meta.name else ""
    if meta.priority:
        name_part += f"[{meta.priority}]"
    if meta.description:
        name_part += f' "{meta.description}"'
    if meta.category:
        name_part += f" {{category={meta.category}}}"
    if name_part:
        name_part += ": "
    arrow = "<=>" if meta.bidirectional else "=>"
    return f"{name_part}{format_sexpr(pat)} {arrow} {format_sexpr(skel)}"
