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


class UnsupportedPattern(Exception):
    """Raised by ``unify`` on a pattern form outside the first-order fragment:
    ``?c``, ``?v``, ``?free``, ``?...``, or a skeleton-only marker (``!``,
    ``fresh``). The caller records the rule as not-analyzed, keeping the
    confluence verdict conservative."""


# A substitution {var_name: term}, maintained fully-applied (idempotent).
Subst = Dict[str, ExprType]

_UNSUPPORTED_HEADS = {"?c", "?v", "?free", "?...", "!", "fresh"}


def _unsupported(t: ExprType) -> bool:
    return compound(t) and isinstance(t[0], str) and t[0] in _UNSUPPORTED_HEADS


def _is_var(t: ExprType) -> bool:
    return arbitrary_expression(t)  # ["?", name]


def apply_subst(subst: Subst, term: ExprType) -> ExprType:
    """Replace each ``["?", name]`` whose ``name`` is bound. Single pass --
    ``subst`` is kept fully-applied -- and recurses into compound arguments."""
    if _is_var(term):
        name = term[1]
        return subst[name] if name in subst else term
    if compound(term):
        # Recurses over index 0 (the operator head) too; that is a harmless
        # no-op since heads are operator strings, never ["?", name] nodes.
        return [apply_subst(subst, sub) for sub in term]
    return term


def _occurs(name: str, term: ExprType) -> bool:
    if _is_var(term):
        return term[1] == name
    if compound(term):
        return any(_occurs(name, sub) for sub in term)
    return False


def _compose_bind(subst: Subst, name: str, value: ExprType) -> Subst:
    """Add ``name -> value`` (value already resolved), substituting it through
    every existing range term so the result stays fully-applied."""
    one = {name: value}
    updated = {k: apply_subst(one, v) for k, v in subst.items()}
    updated[name] = value
    return updated


def unify(t1: ExprType, t2: ExprType,
          subst: Optional[Subst] = None) -> Optional[Subst]:
    """First-order syntactic unification of two structured pattern terms.

    Returns the mgu (a fully-applied ``Subst``) or ``None`` on a normal failure
    (clash / occurs-check / arity / head mismatch). Raises ``UnsupportedPattern``
    on any ?c/?v/?free/?... or skeleton-only node, checked BEFORE the
    variable/compound branches so a typed node is never bound as opaque.

    ``subst`` is an INTERNAL accumulator (left unset by external callers); it is
    assumed to contain no unsupported nodes, since every value it stores derives
    from a term that already passed the refuse-first guard. Do not pre-seed it
    with caller data.
    """
    if subst is None:
        subst = {}
    if _unsupported(t1) or _unsupported(t2):
        raise UnsupportedPattern(f"cannot unify pattern form: {t1!r} ~ {t2!r}")
    a = apply_subst(subst, t1)
    b = apply_subst(subst, t2)
    if _is_var(a):
        return _unify_var(a, b, subst)
    if _is_var(b):
        return _unify_var(b, a, subst)
    if not compound(a) and not compound(b):
        return subst if a == b else None
    if compound(a) and compound(b):
        if a[0] != b[0] or len(a) != len(b):
            return None
        s: Optional[Subst] = subst
        for x, y in zip(a[1:], b[1:]):
            s = unify(x, y, s)
            if s is None:
                return None
        return s
    return None  # atom vs compound


def _unify_var(var: ExprType, other: ExprType, subst: Subst) -> Optional[Subst]:
    name = var[1]
    other = apply_subst(subst, other)
    if _is_var(other) and other[1] == name:
        return subst
    if _occurs(name, other):
        return None
    return _compose_bind(subst, name, other)


def instantiate_skeleton(skeleton: ExprType, sigma: Subst) -> ExprType:
    """Turn a rule RHS (skeleton) into a term under ``sigma``.

    A ``[":", name]`` reference becomes ``sigma``'s value for ``name`` (or the
    variable ``["?", name]`` itself if unbound -- it remains a free variable of
    the critical pair). Compounds recurse; literal atoms/operators are returned
    as-is. Non-trivial skeleton forms (:.../!/fresh) never reach here because
    such rules are refused upstream (see ``is_analyzable``).
    """
    if skeleton_evaluation(skeleton):  # [":", name]
        return apply_subst(sigma, ["?", skeleton[1]])
    if compound(skeleton):
        return [instantiate_skeleton(sub, sigma) for sub in skeleton]
    return skeleton
