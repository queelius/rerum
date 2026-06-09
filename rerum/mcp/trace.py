"""Situated-trace serialization, prose rendering, and recording for the
MCP server.

Wraps engine operations in a temporary ``on_rule_applied`` hook and
serializes the resulting situated ``RewriteTrace`` into the JSON shape
consumed by LLM agents. The shape is the Phase 1 situated trace (each
step carries rule_id, direction, bindings, path, kind, guard, rationale)
plus a domain-agnostic natural-language ``prose`` rendering via
``rerum.training.to_prose`` (Phase 4).
"""

import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from rerum.engine import format_sexpr
from rerum.trace import RewriteStep, RewriteTrace


# Trace truncation defaults. When a trace exceeds MAX_STEPS, the response
# carries the first HEAD_STEPS + last TAIL_STEPS plus an elision marker.
MAX_STEPS = 200
HEAD_STEPS = 100
TAIL_STEPS = 100


def step_to_dict(step: RewriteStep) -> Dict[str, Any]:
    """Convert a situated RewriteStep to the MCP step JSON shape.

    Emits the Phase 1 situated fields (rule_id, direction, bindings, path,
    kind, guard, rationale) alongside the citable metadata labels
    (rule_name, category, reasoning) and the redex-local before/after as
    s-expression strings. ``before_root``/``after_root`` are added by
    ``assemble_trace`` from ``to_global_sequence``; this function emits the
    rule-local edit only.

    The ``provenance`` field is read from ``metadata.extra["provenance"]``;
    rules added by the LLM resolver during ``solve_assisted`` carry
    ``"llm-inferred"`` here. Bidirectional rules add a ``direction_label``.
    """
    meta = step.metadata
    guard = step.guard
    if guard is not None and isinstance(guard.get("condition"), (list, str, int, float)):
        guard = {
            "condition": format_sexpr(guard["condition"]),
            "result": guard.get("result"),
        }

    out: Dict[str, Any] = {
        # Phase 1 situated fields.
        "rule_id": step.rule_id,
        "direction": step.direction,
        "bindings": step.bindings,
        "path": list(step.path) if step.path is not None else [],
        "kind": step.kind,
        "guard": guard,
        "rationale": step.rationale,
        # Redex-local edit (s-expression strings).
        "before": format_sexpr(step.before),
        "after": format_sexpr(step.after),
        # Citable metadata labels.
        "rule_name": meta.name,
        "category": meta.category,
        "reasoning": meta.reasoning,
        "rule_index": step.rule_index,
        "provenance": (meta.extra or {}).get("provenance"),
    }

    if meta.bidirectional:
        if step.direction == "fwd" and meta.fwd_label:
            out["direction_label"] = meta.fwd_label
        elif step.direction == "rev" and meta.rev_label:
            out["direction_label"] = meta.rev_label

    return out
