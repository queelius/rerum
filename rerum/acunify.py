"""AC-unification (Stickel's algorithm) -- unification modulo associative-
commutative operators.

The multi-valued, two-sided generalization of F2's syntactic ``unify``: where
``unify`` returns one most-general unifier or none, AC-unification has no mgu, so
``ac_unify`` yields a COMPLETE SET of unifiers (within a budget; not necessarily
minimal). PURE AC -- the Theory's identity/unit is NOT used (the engine's
normalize strips units upstream). Stickel-general: nested terms, free function
symbols, and multiple AC operators are handled (the last two by recursion).

CORE module. Reuses F2 (rerum/confluence.py): Subst, apply_subst, _occurs,
_is_var, UnsupportedPattern; plus flatten (normalize) and gensym (rewriter).
F2's ``unify`` is UNTOUCHED. Names NO domain operator: keys on theory.is_ac;
fresh variables are gensym'd.

References: Stickel 1981; Baader and Nipkow, Term Rewriting and All That, Ch. 10.
"""

from dataclasses import dataclass
from itertools import combinations, product
from typing import Iterator, List, Optional

from .confluence import (
    apply_subst,
    _occurs,
    _is_var,
    _unsupported,
    UnsupportedPattern,
    Subst,
)
from .normalize import flatten
from .rewriter import compound, gensym


@dataclass
class UnifyBudget:
    """Fail-safe budget for the basis + subset enumeration. On exhaustion,
    ``truncated`` is set and enumeration stops; the yielded set is then SOUND
    but INCOMPLETE (every yield is a real unifier; some may be missing)."""
    steps: Optional[int] = None
    truncated: bool = False

    def spend(self) -> bool:
        if self.steps is None:
            return True
        if self.steps <= 0:
            self.truncated = True
            return False
        self.steps -= 1
        return True


def _hilbert_basis(a: List[int], b: List[int]):
    """Minimal non-negative non-zero integer solutions (the Hilbert basis) of
    ``sum_i a_i*m_i = sum_j b_j*n_j``. Each result is the tuple
    ``(m_1, ..., m_M, n_1, ..., n_N)``. Bounded enumeration: a minimal solution
    has ``m_i <= max(b)`` and ``n_j <= max(a)`` (Fortenbacher's bound)."""
    M, N = len(a), len(b)
    mb = max(b) if b else 0
    nb = max(a) if a else 0
    found = []
    for m in product(range(mb + 1), repeat=M):
        lhs = sum(a[i] * m[i] for i in range(M))
        for n in product(range(nb + 1), repeat=N):
            if sum(b[j] * n[j] for j in range(N)) != lhs:
                continue
            vec = tuple(m) + tuple(n)
            if any(vec):
                found.append(vec)

    def leq(x, y):
        return all(p <= q for p, q in zip(x, y))

    return [v for v in found if not any(u != v and leq(u, v) for u in found)]
