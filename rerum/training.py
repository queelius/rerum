"""The trace-to-text and trace-to-record layer (general, domain-agnostic).

Projects ONE structured ``RewriteTrace`` (Phase 1) into two faithful,
non-drifting views:

- ``to_training_record``: a machine-checkable JSONL row. Per-step
  ``before_root``/``after_root`` come from ``trace.to_global_sequence()``,
  so the structured record IS the whole-expression replay; the in-memory
  step stores only the redex-local edit plus its path.
- ``to_prose``: a deterministic natural-language chain-of-thought. It is a
  projection of the SAME structured trace (per-``kind`` template plus
  ``step.rationale``), so prose and record cannot diverge.

``generate_corpus`` streams records: for each problem it calls a
CALLER-SUPPLIED ``driver(engine, problem) -> (answer, trace)`` (the caller
decides whether to run ``simplify`` or ``solve``), builds the record, and
when a CALLER-SUPPLIED ``checker(problem, answer) -> bool`` is given stamps
``verified``.

General-engine principle (spec Section 0): this module names NO domain
operator. It never decides how to run a problem and never decides how to
validate an answer; both are data the caller supplies. The ``operator``
field of a record is a free-form label stored verbatim and never interpreted
here. Swap "calculus" for "boolean algebra" and this module does not change.

Pure projection layer: no rewriting, search, or numeric checks are
implemented here; those live in ``engine.py``/``solve.py`` and the domain
content under ``examples/``.
"""

from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from .expr import ExprType, format_sexpr


def to_training_record(trace, *, problem, operator, answer,
                       verified: Optional[bool] = None) -> Dict[str, Any]:
    """Build the structured JSONL training record from one ``RewriteTrace``.

    The record schema is::

        {problem, operator,
         steps: [{kind, rule_id, rationale, before_root, after_root,
                  bindings, path, guard}, ...],
         answer, verified}

    ``before_root``/``after_root`` are NOT read off the redex-local step;
    they are supplied by joining each step with the whole-expression states
    from ``trace.to_global_sequence()`` (Phase 1). Every step the global
    sequence yields is included (kind in ``{"rule", "normalize", "fold"}``).
    ``operator`` is a free-form caller-supplied label, stored verbatim and
    never interpreted. ``verified`` is passed through unchanged (the caller,
    e.g. ``generate_corpus``, runs a checker and supplies the verdict).
    """
    steps: List[Dict[str, Any]] = []
    for entry in trace.to_global_sequence():
        step = entry["step"]
        steps.append({
            "kind": step.kind,
            "rule_id": step.rule_id,
            "rationale": step.rationale,
            "before_root": entry["before_root"],
            "after_root": entry["after_root"],
            "bindings": step.bindings,
            "path": step.path,
            "guard": step.guard,
        })
    return {
        "problem": problem,
        "operator": operator,
        "steps": steps,
        "answer": answer,
        "verified": verified,
    }
