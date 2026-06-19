"""F2: confluence and critical-pair diagnostics (read-only analysis).

Computes the CRITICAL PAIRS of a rule set (overlaps between rule left-hand
sides) and checks each for JOINABILITY, reporting a LOCAL confluence verdict.
This is analysis OVER the rewrite relation; it changes no rewrite behavior.

GENERAL ENGINE: hardcodes no operator. Patterns and rules are DATA. FIRST-ORDER
unification only; richer pattern forms (?c/?v/?free/?...) and non-trivial
skeleton forms (:.../!/fresh) and conditional rules are REFUSED -- the affected
rule is reported not-analyzed rather than silently mis-analyzed, so the verdict
is never a false "locally confluent".
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .rewriter import (
    ExprType,
    compound,
    free_symbols,
    gensym,
    skeleton_evaluation,
    arbitrary_expression,
    arbitrary_constant,
    arbitrary_variable,
    arbitrary_free,
    arbitrary_rest,
)

# A position is a tuple of child indices from the root; () is the root.
Position = Tuple[int, ...]


def _is_pattern_node(t: ExprType) -> bool:
    """True if t is any structured pattern node (?/?c/?v/?free/?...)."""
    return (
        arbitrary_expression(t)
        or arbitrary_constant(t)
        or arbitrary_variable(t)
        or arbitrary_free(t)
        or arbitrary_rest(t)
    )


def subterm_at(term: ExprType, p: Position) -> ExprType:
    """The subterm of ``term`` at position ``p`` (a path of child indices)."""
    cur = term
    for i in p:
        cur = cur[i]
    return cur


def replace_at(term: ExprType, p: Position, new: ExprType) -> ExprType:
    """A copy of ``term`` with the subterm at ``p`` replaced by ``new``."""
    if not p:
        return new
    i, rest = p[0], p[1:]
    return term[:i] + [replace_at(term[i], rest, new)] + term[i + 1:]


def positions(term: ExprType) -> List[Position]:
    """All NON-VARIABLE positions of ``term``.

    A non-variable position is one whose subterm is a compound application or a
    constant atom -- NOT a pattern node and NOT the operator head (index 0).
    Overlaps are computed only at these positions.
    """
    out: List[Position] = []

    def walk(t: ExprType, p: Position) -> None:
        if _is_pattern_node(t):
            return  # variable position: skip it (and its subtree)
        out.append(p)
        if compound(t):
            for i in range(1, len(t)):  # operands only; index 0 is the operator
                walk(t[i], p + (i,))

    walk(term, ())
    return out
