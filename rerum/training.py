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

from fractions import Fraction
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from .expr import ExprType, format_sexpr


def corpus_json_default(obj: Any) -> str:
    """``json.dumps`` ``default`` hook for serializing a record to JSONL.

    Records store the engine's value atoms verbatim, which keeps the
    structured trace faithful but means a record can contain a
    ``fractions.Fraction`` (the arithmetic prelude's exact division yields
    one for a non-whole quotient). ``Fraction`` is not JSON-native, so write
    a corpus with ``json.dumps(record, default=corpus_json_default)``. A
    Fraction renders as its exact string form (e.g. ``"1/3"``).

    This is domain-free: it dispatches on the Python TYPE, naming no
    operator. It is deliberately TARGETED -- any other non-serializable type
    raises ``TypeError`` rather than being silently coerced, so an unexpected
    value in a record surfaces as a bug instead of corrupting the corpus.
    """
    if isinstance(obj, Fraction):
        return str(obj)
    raise TypeError(
        f"Object of type {type(obj).__name__} is not JSON serializable"
    )


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
    rules_used: Dict[str, Dict[str, Any]] = {}
    for entry in trace.to_global_sequence():
        step = entry["step"]
        meta = getattr(step, "metadata", None)
        category = getattr(meta, "category", None)
        steps.append({
            "kind": step.kind,
            "rule_id": step.rule_id,
            # Per-step provenance: the rule's category (a free-form label).
            # ``rationale`` already carries reasoning-or-category for prose;
            # ``category`` is the explicit, separable field a downstream
            # dataset can group/filter on without parsing prose.
            "category": category,
            "rationale": step.rationale,
            "before_root": entry["before_root"],
            "after_root": entry["after_root"],
            "bindings": step.bindings,
            "path": step.path,
            "guard": step.guard,
        })
        # Record-level provenance: the distinct rules that fired, with their
        # category, fire count, and any sidecar ``extra`` (difficulty,
        # citation, prerequisites). This is the data path a corpus consumer
        # needs to stratify a dataset; it previously never reached the record.
        rid = step.rule_id
        if rid is not None:
            slot = rules_used.get(rid)
            if slot is None:
                extra = getattr(meta, "extra", None) or None
                rules_used[rid] = {
                    "rule_id": rid,
                    "category": category,
                    "count": 1,
                    "extra": dict(extra) if extra else None,
                }
            else:
                slot["count"] += 1
    return {
        "problem": problem,
        "operator": operator,
        "steps": steps,
        "answer": answer,
        "verified": verified,
        "rules_used": list(rules_used.values()),
    }


def _render(expr: ExprType) -> str:
    """Render an expression to a DSL s-expr string for prose/labels."""
    return format_sexpr(expr)


def _prose_line(entry: Dict[str, Any]) -> str:
    """Render ONE global-sequence entry as a chain-of-thought line.

    Dispatches on the step's ``kind`` to a fixed template, filling the
    ``rule_id``, the ``rationale`` (when present), and the rendered
    whole-expression ``before_root``/``after_root``. The line is a pure
    function of the structured fields, which is what makes the prose a
    projection of the record (no field is invented). The dispatch is on
    ``kind`` only; no operator name is consulted, so this is domain-agnostic.

    Templates (kind -> shape):
      rule      -> "Applying <rule_id> (<rationale>): <before> becomes <after>."
      normalize -> "Simplifying with <rule_id> (<rationale>): <before> becomes <after>."
      fold      -> "Computing with <rule_id> (<rationale>): <before> becomes <after>."
    The "(<rationale>)" clause is omitted when ``rationale`` is None/empty.
    """
    step = entry["step"]
    before = _render(entry["before_root"])
    after = _render(entry["after_root"])
    rule_id = step.rule_id if step.rule_id is not None else "(anonymous rule)"
    reason = f" ({step.rationale})" if step.rationale else ""
    if step.kind == "normalize":
        verb = "Simplifying with"
    elif step.kind == "fold":
        verb = "Computing with"
    else:  # "rule" and any future kind default to the rule template
        verb = "Applying"
    return f"{verb} {rule_id}{reason}: {before} becomes {after}."


def to_prose(trace) -> str:
    """Render a deterministic natural-language chain-of-thought.

    A PROJECTION of the structured trace: it begins with the problem
    (``trace.initial``), emits one per-``kind`` templated line for each
    global-sequence step (rule / normalize / fold), and ends with the
    answer (``trace.final``). No LLM call, no randomness: the same trace
    yields the same prose every time. Because the lines are derived from
    the same fields ``to_training_record`` exposes, the prose and the
    structured record cannot drift. Templates key on ``kind`` only, so this
    is domain-agnostic.
    """
    lines: List[str] = []
    lines.append(f"Problem: {_render(trace.initial)}.")
    for entry in trace.to_global_sequence():
        lines.append(_prose_line(entry))
    lines.append(f"Answer: {_render(trace.final)}.")
    return "\n".join(lines)


# A problem is an (operator_label, expr) pair. ``operator_label`` is a
# free-form string the caller chooses (NOT interpreted by this module); it is
# stamped verbatim into the record's ``operator`` field. ``expr`` is the
# expression to drive. The ``driver`` receives the whole pair and decides how
# to run it (simplify for a confluent rule set, solve for a non-confluent
# one); that decision is the CALLER's, never this module's.
ProblemType = Tuple[str, ExprType]

DriverType = Callable[[Any, ProblemType], Tuple[Any, Any]]
CheckerType = Callable[[ProblemType, Any], bool]


def generate_corpus(engine, problems: Sequence[ProblemType], *,
                    driver: DriverType,
                    checker: Optional[CheckerType] = None,
                    ) -> Iterator[Dict[str, Any]]:
    """Stream training records, one per problem, via a caller-supplied driver.

    ``generate_corpus`` is GENERAL: it names no domain operator and makes no
    ``simplify``-vs-``solve`` choice. For each problem it calls::

        answer, trace = driver(engine, problem)

    where ``driver`` is the caller's domain adapter (it runs ``simplify`` for
    a confluent rule set, ``solve`` for a non-confluent one, and returns the
    Phase-1 ``RewriteTrace``). The record is built with
    ``to_training_record`` (the problem's ``operator_label`` becomes the
    record's ``operator``, the s-expr rendering of the problem expression
    becomes ``problem``, and ``answer`` is the driver's result). When
    ``checker`` is supplied, ``verified = checker(problem, answer)``;
    otherwise ``verified`` stays ``None``. Yields the record (a generator:
    records stream out, nothing is accumulated).

    ``problems`` is a sequence of ``(operator_label, expr)`` pairs. The
    operator label is the caller's free-form tag; this module stores it
    verbatim and never inspects it.

    Serialization: records preserve the engine's exact value atoms verbatim,
    including ``fractions.Fraction``. To write the stream to JSONL, pass
    :func:`corpus_json_default` as the ``json.dumps`` ``default`` hook:
    ``json.dumps(record, default=corpus_json_default)``.
    """
    for problem in problems:
        operator_label, expr = problem
        answer, trace = driver(engine, problem)
        verified: Optional[bool] = None
        if checker is not None:
            verified = checker(problem, answer)
        record = to_training_record(
            trace,
            problem=format_sexpr(expr),
            operator=operator_label,
            answer=answer,
            verified=verified,
        )
        yield record
