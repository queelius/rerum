"""Cost functions and optimization result type for cost-directed search.

These utilities are pure functions over ``ExprType`` with no rule-machinery
dependency; they are extracted from ``engine.py`` so that downstream callers
(``RuleEngine.minimize``, ``EquivalenceClass.minimum`` once introduced) can
share them without pulling in the engine's state. ``COST_METRICS`` is the
registry consulted when ``minimize(metric="size")`` and friends select a
named metric.
"""

from typing import Any, Callable, Dict, List, Optional

from .rewriter import ExprType


def expr_size(expr: ExprType) -> int:
    """Total node count.

    Atoms count as 1; compound expressions sum their elements recursively.
    Examples::

        expr_size("x") == 1
        expr_size(["+", "x", 1]) == 3
        expr_size(["+", ["*", "a", "b"], "c"]) == 5
    """
    if isinstance(expr, list):
        return sum(expr_size(e) for e in expr)
    return 1


def expr_depth(expr: ExprType) -> int:
    """Maximum nesting depth.

    Atoms have depth 0; compounds have ``1 + max child depth``. Empty list
    has depth 1.
    """
    if isinstance(expr, list):
        if len(expr) == 0:
            return 1
        return 1 + max(expr_depth(e) for e in expr)
    return 0


def expr_ops(expr: ExprType) -> int:
    """Number of compound (operation) nodes.

    Atoms have 0 ops; each compound expression contributes 1 plus the ops
    of its elements.
    """
    if isinstance(expr, list):
        return 1 + sum(expr_ops(e) for e in expr)
    return 0


def expr_atoms(expr: ExprType) -> int:
    """Number of leaf operands (operators not counted).

    Examples::

        expr_atoms("x") == 1
        expr_atoms(["+", "x", 1]) == 2
        expr_atoms(["+", ["*", "a", "b"], "c"]) == 3
    """
    if isinstance(expr, list):
        return sum(expr_atoms(e) for e in expr[1:])
    return 1


def make_op_cost_fn(
    op_costs: Dict[str, float], default: float = 1.0
) -> Callable[[ExprType], float]:
    """Build a cost function from per-operator weights.

    Returns a callable that sums ``op_costs[op]`` over each compound node,
    using ``default`` for operators not in the table.

    Example::

        cost_fn = make_op_cost_fn({"+": 1, "*": 2, "/": 5, "^": 10})
        cost_fn(["+", ["*", "a", "b"], "c"])  # 1 + 2 == 3
    """

    def cost_fn(expr: ExprType) -> float:
        if isinstance(expr, list) and len(expr) > 0:
            op = expr[0]
            op_cost = op_costs.get(op, default) if isinstance(op, str) else default
            return op_cost + sum(cost_fn(e) for e in expr[1:])
        return 0

    return cost_fn


COST_METRICS: Dict[str, Callable[[ExprType], float]] = {
    "size": expr_size,
    "depth": expr_depth,
    "ops": expr_ops,
    "atoms": expr_atoms,
}


class OptimizationResult:
    """Result of cost-directed search over an equivalence class.

    Attributes:
        expr: The lowest-cost expression found.
        cost: Cost of ``expr``.
        original: The starting expression.
        original_cost: Cost of ``original``.
        expressions_checked: Total number of equivalents evaluated.

    Truthiness: ``True`` when an improvement was found
    (``cost < original_cost``).
    """

    __slots__ = ("expr", "cost", "original", "original_cost", "expressions_checked", "derivation")

    def __init__(
        self,
        expr: ExprType,
        cost: float,
        original: ExprType,
        original_cost: float,
        expressions_checked: int = 0,
        derivation: Optional["RewriteTrace"] = None,
    ):
        self.expr = expr
        self.cost = cost
        self.original = original
        self.original_cost = original_cost
        self.expressions_checked = expressions_checked
        self.derivation = derivation

    @property
    def improvement(self) -> float:
        """Absolute cost reduction (``original_cost - cost``)."""
        return self.original_cost - self.cost

    @property
    def cost_ratio(self) -> float:
        """Retained cost as a ratio of the original (1.0 = no change, 0.5 = halved)."""
        if self.original_cost == 0:
            return 1.0 if self.cost == 0 else float("inf")
        return self.cost / self.original_cost

    @property
    def improvement_ratio(self) -> float:
        """Fractional improvement (0.0 = no improvement, 1.0 = eliminated).

        Note: this is ``1 - cost/original_cost``. The retention ratio is
        ``cost_ratio``. The semantics flipped in 0.5.0; pre-0.5 callers
        that read ``improvement_ratio`` as "fraction kept" should switch
        to ``cost_ratio``.
        """
        if self.original_cost == 0:
            return 0.0 if self.cost == 0 else float("-inf")
        return 1.0 - (self.cost / self.original_cost)

    def __repr__(self) -> str:
        from .expr import format_sexpr
        expr_str = format_sexpr(self.expr)
        return (
            f"OptimizationResult({expr_str}, cost={self.cost}, "
            f"checked={self.expressions_checked})"
        )

    def __bool__(self) -> bool:
        """True if any improvement was made."""
        return self.cost < self.original_cost

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "expr": self.expr,
            "cost": self.cost,
            "original": self.original,
            "original_cost": self.original_cost,
            "improvement": self.improvement,
            "expressions_checked": self.expressions_checked,
            "derivation": self.derivation.to_dict() if self.derivation is not None else None,
        }
