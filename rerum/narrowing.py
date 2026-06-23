"""F6: narrowing -- unification-driven backward rewriting.

Narrowing is rewriting with the matcher swapped for a unifier: a rewrite step
asks "does rule LHS l MATCH subterm t|p?" (only l's variables bind); a narrowing
step asks "can l and t|p be UNIFIED?" (both sides' variables bind). Iterating
narrowing solves goals -- find sigma such that sigma(start) reduces to a target --
and equations (E-unification). Sound and complete for confluent terminating
systems; the demoted best-first `solve` is neither.

CORE module. Reuses F2 (rerum/confluence.py) wholesale: unify, apply_subst,
rename_apart, is_analyzable, instantiate_skeleton, _variables, _is_var. Operates
on PATTERN TERMS (variables are ["?", name]). Scope: analyzable first-order rules
only (unify refuses ?c/?v/?free/?...); SYNTACTIC (a loaded AC theory is not used --
true AC-narrowing needs AC-unification). Names NO domain operator: solve_equation's
eq/true symbols are gensym'd; domains arrive as engine rules (data).
"""

from collections import deque
from dataclasses import dataclass
from typing import Iterator, Optional

from .confluence import (
    unify,
    apply_subst,
    rename_apart,
    is_analyzable,
    instantiate_skeleton,
    _variables,
    _is_var,
    UnsupportedPattern,
)
from .rewriter import compound, gensym, free_symbols, ExprType


def _positions(term) -> Iterator[list]:
    """Yield the path (list of indices) to every NON-VARIABLE subterm of
    ``term``. Index 0 (the operator head) is not a position; a ``["?", _]``
    node is not a position (and we do not recurse into it)."""
    if _is_var(term):
        return
    yield []
    if compound(term):
        for i in range(1, len(term)):
            for sub in _positions(term[i]):
                yield [i] + sub


def _term_at(term, path):
    """The subterm of ``term`` at integer-index ``path``."""
    for i in path:
        term = term[i]
    return term


def _replace_at(term, path, new):
    """A copy of ``term`` with the subterm at ``path`` replaced by ``new``.
    Functional: ``term`` is not mutated."""
    if not path:
        return new
    i = path[0]
    return term[:i] + [_replace_at(term[i], path[1:], new)] + term[i + 1:]


def _compose(s2: dict, s1: dict) -> dict:
    """Substitution composition ``s2 . s1``: apply ``s1`` first, then ``s2``.
    For x in dom(s1): s2(s1(x)); for x in dom(s2)\\dom(s1): s2(x)."""
    out = {name: apply_subst(s2, val) for name, val in s1.items()}
    for name, val in s2.items():
        out.setdefault(name, val)
    return out
