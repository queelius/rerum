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

from typing import Any, Callable, Dict, List, Optional

from .rewriter import ExprType


class RewriteStep:
    """A single step in a rewriting trace.

    Captures the rule that fired, the expression before, and the expression
    after the rewrite. ``metadata`` is the ``RuleMetadata`` of the rule that
    fired (typed loosely here to avoid a circular import on engine.py).
    """

    __slots__ = ("rule_index", "metadata", "before", "after")

    def __init__(
        self,
        rule_index: int,
        metadata: Any,
        before: ExprType,
        after: ExprType,
    ):
        self.rule_index = rule_index
        self.metadata = metadata
        self.before = before
        self.after = after

    def _name(self) -> str:
        return self.metadata.name or f"rule[{self.rule_index}]"

    def __repr__(self) -> str:
        from .expr import format_sexpr
        return f"{self._name()}: {format_sexpr(self.before)} → {format_sexpr(self.after)}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary for serialization."""
        return {
            "rule_index": self.rule_index,
            "rule_name": self.metadata.name,
            "description": self.metadata.description,
            "before": self.before,
            "after": self.after,
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to JSON-serializable dictionary."""
        return {
            "initial": self.initial,
            "final": self.final,
            "steps": [step.to_dict() for step in self.steps],
            "step_count": len(self.steps),
        }

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
