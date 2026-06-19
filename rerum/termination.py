"""F4: termination via the lexicographic path order (read-only analysis).

Certifies that a rule set TERMINATES and ORIENTS an equation toward a
terminating direction, using the lexicographic path order (LPO) derived from a
PRECEDENCE on function symbols supplied as DATA. If every rule l -> r satisfies
l >_lpo r, every rewrite step strictly decreases the term in the well-founded
order >_lpo, so the system cannot loop.

GENERAL ENGINE: the precedence is DATA (a list of function symbols -- operators
AND constants -- in decreasing order). No operator is hardcoded. First-order
only; non-first-order rules are refused by reusing confluence.is_analyzable.
A constant atom is treated as a 0-ARY function symbol (standard LPO).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .rewriter import ExprType, compound
from .confluence import (
    is_analyzable,
    instantiate_skeleton,
    _is_var,
)

# A precedence: function symbols in DECREASING precedence (head = greatest).
Precedence = List


def _prec_gt(f, g, precedence: Precedence) -> bool:
    """True iff f > g: both listed and f earlier (greater) than g. Symbols not
    both in the list are incomparable (False both ways). The precedence list
    must be duplicate-free (``index`` uses the first occurrence)."""
    if f not in precedence or g not in precedence:
        return False
    return precedence.index(f) < precedence.index(g)


def _head_args(t: ExprType) -> Tuple:
    """(head, args), treating a constant atom as a 0-ary symbol."""
    if compound(t):
        return t[0], t[1:]
    return t, []


def _occurs_node(node: ExprType, term: ExprType) -> bool:
    """True iff the subexpression ``node`` occurs anywhere inside ``term``
    (structural equality). Distinct from ``confluence._occurs``, which tests a
    variable NAME; this tests a whole node (e.g. a ``["?", x]`` variable node)."""
    if node == term:
        return True
    if compound(term):
        return any(_occurs_node(node, sub) for sub in term)
    return False


def _lex_gt(sargs: List, targs: List, precedence: Precedence) -> bool:
    """Lexicographic compare of equal-length tuples (caller guarantees equal
    lengths): the first DIFFERING position must have its s-arg >_lpo its t-arg
    (earlier positions equal)."""
    for si, ti in zip(sargs, targs):
        if si == ti:
            continue
        return lpo_greater(si, ti, precedence)
    return False  # identical


def lpo_greater(s: ExprType, t: ExprType, precedence: Precedence) -> bool:
    """The lexicographic path order: True iff ``s`` strictly dominates ``t``."""
    if s == t:
        return False
    if _is_var(s):
        return False  # a variable dominates nothing
    if _is_var(t):
        return _occurs_node(t, s)  # s != t, so t is a proper subterm of s
    f, sargs = _head_args(s)
    g, targs = _head_args(t)
    # Case 1 (subterm): some argument of s is >= t.
    if any(si == t or lpo_greater(si, t, precedence) for si in sargs):
        return True
    # Case 2 (precedence): f outranks g and s beats every argument of t.
    if _prec_gt(f, g, precedence) and all(
            lpo_greater(s, tj, precedence) for tj in targs):
        return True
    # Case 3 (lexicographic): same head AND arity, s beats every tj, args >lex.
    if (f == g and len(sargs) == len(targs)
            and all(lpo_greater(s, tj, precedence) for tj in targs)
            and _lex_gt(sargs, targs, precedence)):
        return True
    return False


def orient(l: ExprType, r: ExprType,
           precedence: Precedence) -> Optional[str]:
    """Pick the terminating direction for the equation ``l = r``.

    Returns "lr" if ``l >_lpo r`` (rule ``l -> r`` decreases), "rl" if
    ``r >_lpo l``, or ``None`` if this LPO/precedence orients neither (e.g. a
    commutativity axiom, which no reduction order can orient). The orientation
    oracle Knuth-Bendix completion (F5) needs.
    """
    if lpo_greater(l, r, precedence):
        return "lr"
    if lpo_greater(r, l, precedence):
        return "rl"
    return None


@dataclass(frozen=True)
class TerminationReport:
    """Result of an LPO termination check.

    ``terminating`` is True iff EVERY rule is analyzable AND oriented
    ``l >_lpo r`` -- a PROOF of termination by this LPO. False means "not proven
    by this precedence" (the rule may be reversed, incomparable, or genuinely
    non-terminating), with ``unoriented``/``not_analyzed`` explaining why. Like
    F2's ``unknown``, it is honest about the limit -- it never claims
    "non-terminating", only "not proven".
    """
    terminating: bool
    oriented: List[Tuple[str, str]]   # (rule name, direction "lr")
    unoriented: List[str]
    not_analyzed: List[str]


def check_termination(engine, precedence: Precedence) -> TerminationReport:
    """LPO termination diagnostic for ``engine``'s enabled rules under
    ``precedence``. Read-only: mutates nothing."""
    oriented: List[Tuple[str, str]] = []
    unoriented: List[str] = []
    not_analyzed: List[str] = []
    for _idx, rule, meta in engine.rule_set():
        pattern, skeleton = rule[0], rule[1]
        if not is_analyzable(pattern, skeleton, meta.condition):
            not_analyzed.append(meta.name)
            continue
        r_term = instantiate_skeleton(skeleton, {})  # [":",x] -> ["?",x]
        if lpo_greater(pattern, r_term, precedence):
            oriented.append((meta.name, "lr"))
        else:
            unoriented.append(meta.name)
    terminating = (not unoriented) and (not not_analyzed)
    return TerminationReport(
        terminating=terminating, oriented=oriented,
        unoriented=unoriented, not_analyzed=not_analyzed,
    )
