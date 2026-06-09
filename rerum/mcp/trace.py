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

# The shared sanitizer lives in utils; the old private name is kept as an
# alias for the transition (tools and tests import it from here).
from rerum.mcp.utils import json_safe as _json_safe


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
    if guard is not None:
        # Render the condition as an s-expr string (format_sexpr is
        # Fraction-safe) and JSON-sanitize the computed result (it may be a
        # Fraction). Keeps the whole guard JSON-serializable.
        cond = guard.get("condition")
        guard = {
            "condition": format_sexpr(cond) if cond is not None else None,
            "result": _json_safe(guard.get("result")),
        }

    out: Dict[str, Any] = {
        # Phase 1 situated fields.
        "rule_id": step.rule_id,
        "direction": step.direction,
        "bindings": _json_safe(step.bindings),
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


def _attach_global_roots(recorder) -> None:
    """Add before_root/after_root to each serialized step from the
    RewriteTrace's whole-expression reconstruction.

    Uses the Phase 1 ``RewriteTrace.to_global_sequence()`` which replays
    from ``initial`` splicing each step's after at its path, yielding
    per-step whole-expression before_root/after_root. Best-effort: if the
    trace cannot reconstruct (missing initial), the roots are omitted.
    """
    if recorder.trace is None:
        return
    try:
        seq = recorder.trace.to_global_sequence()
    except Exception:
        return
    for serialized, entry in zip(recorder.steps, seq):
        before_root = entry.get("before_root")
        after_root = entry.get("after_root")
        serialized["before_root"] = (
            format_sexpr(before_root) if not isinstance(before_root, str)
            and before_root is not None else before_root
        )
        serialized["after_root"] = (
            format_sexpr(after_root) if not isinstance(after_root, str)
            and after_root is not None else after_root
        )


def render_prose(recorder) -> str:
    """Render the recorded trace to natural-language prose.

    Delegates to the GENERAL, domain-agnostic ``rerum.training.to_prose``
    (Phase 4). Returns an empty string if there is no trace to render.
    """
    if recorder.trace is None:
        return ""
    try:
        from rerum.training import to_prose
    except Exception:
        return ""
    try:
        return to_prose(recorder.trace)
    except Exception:
        return ""


def assemble_trace(*, initial: str, final: str, recorder=None,
                   prose: Optional[str] = None) -> Dict[str, Any]:
    """Build the full situated-trace dict for an MCP response.

    Attaches whole-expression before_root/after_root per step (from
    ``to_global_sequence``), a ``prose`` rendering (delta 4), and the
    truncation policy from the prior plan (first HEAD_STEPS + an _elided
    marker + last TAIL_STEPS). ``prose`` may be passed explicitly (tests);
    otherwise it is rendered from the recorder's trace.
    """
    if recorder is not None:
        _attach_global_roots(recorder)
        # The recorder accumulates steps + initial but never the final state,
        # so set it here for the prose answer line. ``final`` is the
        # already-rendered s-expr string; ``format_sexpr`` passes a string
        # through, so the prose reads "Answer: <final>." rather than "None".
        if recorder.trace is not None and recorder.trace.final is None:
            recorder.trace.final = final
        steps = recorder.steps
    else:
        steps = []

    if prose is None:
        prose = render_prose(recorder) if recorder is not None else ""

    total = len(steps)
    out: Dict[str, Any] = {
        "initial": initial,
        "final": final,
        "total_steps": total,
        "summary": _summarize(steps),
        "prose": prose,
    }

    if total > MAX_STEPS:
        elided_count = total - HEAD_STEPS - TAIL_STEPS
        marker = {"_elided": True, "count": elided_count}
        out["steps"] = steps[:HEAD_STEPS] + [marker] + steps[-TAIL_STEPS:]
        out["trace_truncated"] = {"original_length": total}
    else:
        out["steps"] = steps

    return out


def _summarize(steps: List[Dict[str, Any]]) -> str:
    """One-line digest of the steps, by rule_id and kind."""
    if not steps:
        return "No rules applied."
    counts: Dict[str, int] = {}
    for s in steps:
        name = s.get("rule_id") or s.get("rule_name") or f"rule[{s.get('rule_index')}]"
        counts[name] = counts.get(name, 0) + 1
    most = max(counts.items(), key=lambda kv: kv[1])
    return (
        f"{len(steps)} steps using {len(counts)} unique rules. "
        f"Most used: {most[0]} ({most[1]}x)."
    )
