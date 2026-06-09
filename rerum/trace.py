"""Tracing infrastructure for rewriting steps.

A trace captures the sequence of rule applications during simplification.
The internal API is listener-based: any code that applies rules accepts
an optional ``listener: Optional[Callable[[RewriteStep], None]]``. Calling
the listener after each successful rule application is the only requirement;
how those steps are accumulated, reported, or summarized is the listener's
concern.

``RewriteTrace`` is the canonical listener used by ``simplify(trace=True)``:
it accumulates steps into an ordered list with the original and final
expressions. Other listeners might be progress bars, debugging stoppers, or
custom counters.
"""

import hashlib

from typing import Any, Callable, Dict, List, Optional

from .rewriter import ExprType


def splice_at(root: ExprType, path: List[int], subtree: ExprType) -> ExprType:
    """Return a copy of ``root`` with the subtree at ``path`` replaced.

    ``path`` is a list of child indices: ``[]`` addresses the root itself,
    ``[1]`` the element at index 1 of a list expression, ``[1, 2]`` the
    element at index 2 of the element at index 1, and so on. Pure: ``root``
    is never mutated and the returned structure shares no mutable nodes on
    the spliced path with ``root``.
    """
    if not path:
        return subtree
    if not isinstance(root, list):
        raise ValueError(f"cannot splice into non-list at path {path}: {root!r}")
    i = path[0]
    if i < 0 or i >= len(root):
        raise IndexError(f"path index {i} out of range for {root!r}")
    new_child = splice_at(root[i], path[1:], subtree)
    return root[:i] + [new_child] + root[i + 1:]


def rule_identity(metadata: Any, pattern: ExprType, skeleton: ExprType) -> str:
    """Stable identity for a rule.

    Returns ``metadata.name`` when set, else ``"#"`` followed by the first 12
    hex chars of the sha1 of the rule's ``(pattern)(skeleton)`` rendering.
    Robust to the post-desugar rule-index churn that makes ``rule_index``
    brittle as an identity.
    """
    name = getattr(metadata, "name", None)
    if name:
        return name
    payload = f"({pattern!r})({skeleton!r})".encode("utf-8")
    return "#" + hashlib.sha1(payload).hexdigest()[:12]


class RewriteStep:
    """A single step in a rewriting trace.

    Captures the rule that fired, the expression before, and the expression
    after the rewrite. ``metadata`` is the ``RuleMetadata`` of the rule that
    fired (typed loosely here to avoid a circular import on engine.py).
    """

    __slots__ = (
        "rule_index", "metadata", "before", "after",
        "rule_id", "direction", "bindings", "path", "kind", "guard",
        "rationale",
    )

    def __init__(
        self,
        rule_index: int,
        metadata: Any,
        before: ExprType,
        after: ExprType,
        *,
        rule_id: Optional[str] = None,
        direction: Optional[str] = None,
        bindings: Optional[dict] = None,
        path: Optional[List[int]] = None,
        kind: str = "rule",
        guard: Optional[dict] = None,
        rationale: Optional[str] = None,
    ):
        self.rule_index = rule_index
        self.metadata = metadata
        self.before = before
        self.after = after
        self.rule_id = rule_id
        self.direction = direction
        self.bindings = bindings
        self.path = path
        self.kind = kind
        self.guard = guard
        self.rationale = rationale

    @property
    def before_redex(self) -> ExprType:
        """Alias of ``before``: the redex-local subtree before the edit."""
        return self.before

    @property
    def after_redex(self) -> ExprType:
        """Alias of ``after``: the redex-local subtree after the edit."""
        return self.after

    def __eq__(self, other: Any) -> bool:
        """Identity for step-vs-step; endpoint match for step-vs-expression.

        Comparing against another RewriteStep is object identity (steps are
        not value-equal). Comparing against any other operand (an expression:
        list/str/number) tests ``self.after == other`` so a reconstructed
        proof path element equals the node expression it represents.
        """
        if isinstance(other, RewriteStep):
            return self is other
        return self.after == other

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        # Identity hash, consistent with step-vs-step identity equality.
        # CAVEAT: a step and the expression it equals (its ``after``) hash
        # differently, so do NOT mix steps and raw expressions in the same
        # set/dict (membership would silently miss). Proof/derivation paths
        # keep steps in plain lists, where ``==`` is used, not hashing.
        return id(self)

    def _name(self) -> str:
        return self.metadata.name or f"rule[{self.rule_index}]"

    def __repr__(self) -> str:
        from .expr import format_sexpr
        return f"{self._name()}: {format_sexpr(self.before)} → {format_sexpr(self.after)}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary for serialization.

        Emits the legacy keys (rule_index, rule_name, description, before,
        after) plus the situated keys (rule_id, direction, kind, path,
        bindings, guard, rationale).
        """
        return {
            "rule_index": self.rule_index,
            "rule_id": self.rule_id,
            "rule_name": self.metadata.name,
            "direction": self.direction,
            "description": self.metadata.description,
            "kind": self.kind,
            "before": self.before,
            "after": self.after,
            "path": self.path,
            "bindings": self.bindings,
            "guard": self.guard,
            "rationale": self.rationale,
        }


# Listener protocol: a callable that consumes ``RewriteStep`` events.
TraceListener = Callable[[RewriteStep], None]


class RewriteTrace:
    """Accumulating trace of rewrite steps. Doubles as a ``TraceListener``:
    instances are callable and append the received step to ``self.steps``.

    Provides multiple formatting options:
      - default ``repr`` and ``format("verbose")``: multi-line with before/after
      - ``format("compact")``: single-line rule chain
      - ``format("rules")``: just the rule names
      - ``format("chain")``: stepped expression transformations
      - ``to_dict()``: JSON-serializable dictionary
    """

    def __init__(self):
        self.steps: List[RewriteStep] = []
        self.initial: Optional[ExprType] = None
        self.final: Optional[ExprType] = None

    def __call__(self, step: RewriteStep) -> None:
        """Listener form: receive a step and append it."""
        self.steps.append(step)

    # Legacy explicit append, kept for backward compatibility.
    def add_step(self, step: RewriteStep) -> None:
        self.steps.append(step)

    def format(self, style: str = "verbose") -> str:
        """Format the trace.

        Styles: ``"verbose"`` (default, multi-line), ``"compact"``,
        ``"rules"``, ``"chain"``.
        """
        from .expr import format_sexpr

        if style == "compact":
            rules = [s._name() for s in self.steps]
            return (
                f"{format_sexpr(self.initial)} "
                f"--[{', '.join(rules)}]--> "
                f"{format_sexpr(self.final)}"
            )

        elif style == "rules":
            rules = [s._name() for s in self.steps]
            return " -> ".join(rules) if rules else "(no rules applied)"

        elif style == "chain":
            if not self.steps:
                return format_sexpr(self.initial)
            parts = [format_sexpr(self.initial)]
            for step in self.steps:
                parts.append(f"  --({step._name()})-->")
                parts.append(format_sexpr(step.after))
            return "\n".join(parts)

        else:  # verbose
            return repr(self)

    def __repr__(self) -> str:
        from .expr import format_sexpr
        lines = [f"Initial: {format_sexpr(self.initial)}"]
        for i, step in enumerate(self.steps, 1):
            if step.metadata.description:
                lines.append(f"  {i}. {step.metadata} ({step.metadata.description})")
            else:
                lines.append(f"  {i}. {step}")
        lines.append(f"Final: {format_sexpr(self.final)}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)

    def __bool__(self) -> bool:
        """True if any rewriting was done."""
        return len(self.steps) > 0

    def to_global_sequence(self) -> List[Dict[str, Any]]:
        """Replay the trace from ``self.initial`` as whole-expression states.

        Each entry is ``{"before_root", "after_root", "step"}``. The running
        root starts at ``self.initial``; for each step, ``after_root`` is the
        running root with ``step.after`` (the redex result) spliced in at
        ``step.path`` (``[]`` when a step predates path threading, meaning a
        root-level edit). The new ``after_root`` becomes the next
        ``before_root``. Lossless: the redex-local edits plus paths fully
        determine the global derivation, so it need not be stored per step.

        CONTRACT: each ``step.path`` must be recorded relative to the
        *running root at the moment that step fired*, not relative to
        ``self.initial``. The strategy drivers that emit steps are
        responsible for this (a step firing on a child threads the path to
        that child). A legacy step with ``path`` ``None`` is treated as a
        whole-expression edit (``[]``), which is correct when its
        ``before``/``after`` are themselves whole expressions.
        """
        sequence: List[Dict[str, Any]] = []
        root = self.initial
        for step in self.steps:
            path = step.path if step.path is not None else []
            before_root = root
            after_root = splice_at(root, path, step.after)
            sequence.append({
                "before_root": before_root,
                "after_root": after_root,
                "step": step,
            })
            root = after_root
        return sequence

    def to_dict(self, global_sequence: bool = False) -> Dict[str, Any]:
        """Convert trace to JSON-serializable dictionary.

        When ``global_sequence`` is True, include a ``global_sequence`` key:
        the whole-expression replay from ``to_global_sequence()`` with each
        step rendered via ``RewriteStep.to_dict()``.
        """
        d: Dict[str, Any] = {
            "initial": self.initial,
            "final": self.final,
            "steps": [step.to_dict() for step in self.steps],
            "step_count": len(self.steps),
        }
        if global_sequence:
            d["global_sequence"] = [
                {
                    "before_root": entry["before_root"],
                    "after_root": entry["after_root"],
                    "step": entry["step"].to_dict(),
                }
                for entry in self.to_global_sequence()
            ]
        return d

    def rule_counts(self) -> Dict[str, int]:
        """Count how many times each rule was applied."""
        counts: Dict[str, int] = {}
        for step in self.steps:
            name = step._name()
            counts[name] = counts.get(name, 0) + 1
        return counts

    def rules_applied(self) -> List[str]:
        """Rule names in order of application."""
        return [s._name() for s in self.steps]

    def summary(self) -> str:
        """Brief summary of the rewriting."""
        if not self.steps:
            return "No rewriting performed"
        counts = self.rule_counts()
        most_used = max(counts.items(), key=lambda x: x[1])
        return (
            f"{len(self.steps)} steps using {len(counts)} unique rules. "
            f"Most used: {most_used[0]} ({most_used[1]}x)"
        )
