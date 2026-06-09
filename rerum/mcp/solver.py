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

    ``state`` is a shared dict the caller owns (this factory initializes its
    defaults); the resolver records ``call_count`` and a ``last_termination``
    code (``"resolver_budget_exhausted"`` when the per-solve cap is hit).
    Which rules were ACTUALLY installed is read off the engine afterwards by
    the caller (provenance metadata), not bookkept here -- the engine
    deduplicates re-proposed named rules, so resolver-side accounting would
    over-count.
    """
    state.setdefault("call_count", 0)
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
        return Resolution(rules=rules_for_resolution, metadata={
            "provenance": "llm-inferred", "via_solve": True, "round": round_num})

    return resolver


def _build_prompt(expr, goal, engine) -> str:
    from rerum.engine import format_sexpr
    expr_str = format_sexpr(expr)
    rules_count = len(engine)
    categories = sorted({meta.category for _, _, meta in engine.iter_rules()
                         if meta.category})
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
    """Delegate per-rule example validation to the engine's own validator.

    ``_validate_rule_examples`` honors the bidirectional direction-skip and
    threads the engine's undefined_op/fold_error resolvers; re-implementing
    that loop here had already drifted (it dropped the resolver threading).
    """
    for meta, pat, skel in rule_pairs:
        engine._validate_rule_examples([pat, skel], meta)
