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


class _Recorder:
    """Holds the captured situated steps and the RewriteTrace for a block.

    ``steps`` is the list of ``step_to_dict`` outputs (rule-local). ``trace``
    is a ``RewriteTrace`` accumulating the raw situated ``RewriteStep`` objects
    so ``assemble_trace`` can reconstruct whole-expression before/after via
    ``to_global_sequence()``.
    """

    def __init__(self):
        self.steps: List[Dict[str, Any]] = []
        self.trace: Optional[RewriteTrace] = None
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    @property
    def duration_ms(self) -> float:
        if self.start_time is None or self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000.0


@contextmanager
def trace_recorder(engine, *, initial=None):
    """Register a temporary on_rule_applied hook and capture situated steps.

    The hook is removed in a finally block so an exception inside the
    with-block does not leak the registration. Yields a ``_Recorder``
    whose ``.steps`` (serialized) and ``.trace`` (raw RewriteTrace) are
    populated as the engine fires ``rule_applied`` events.
    """
    recorder = _Recorder()
    recorder.trace = RewriteTrace()
    recorder.trace.initial = initial

    def hook(step, ctx):
        recorder.trace.add_step(step)
        recorder.steps.append(step_to_dict(step))

    engine.on_rule_applied(hook)
    recorder.start_time = time.perf_counter()
    try:
        yield recorder
    finally:
        recorder.end_time = time.perf_counter()
        engine.off_rule_applied(hook)
