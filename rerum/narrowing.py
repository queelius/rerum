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


@dataclass
class NarrowStep:
    """One narrowing successor: ``successor`` is the new term, ``sigma`` the
    step mgu, ``position`` the redex path, ``rule_id`` the rule's name."""
    successor: ExprType
    sigma: dict
    position: list
    rule_id: str


def narrow_step(term, rules) -> Iterator[NarrowStep]:
    """Yield every one-step narrowing successor of ``term`` under ``rules``
    (a list of ``(l_pattern, r_term, rule_id)`` triples). For each non-variable
    position p and each rule, rename the rule apart from ``term``, unify
    ``term|p`` with the rule LHS, and apply the mgu to ``term`` with the RHS
    spliced at p."""
    avoid = _variables(term)
    for p in _positions(term):
        sub = _term_at(term, p)
        if _is_var(sub):
            continue
        for (l, r, rule_id) in rules:
            l_r, r_r = rename_apart(l, r, avoid)
            try:
                mgu = unify(sub, l_r)
            except UnsupportedPattern:
                continue
            if mgu is None:
                continue
            successor = apply_subst(mgu, _replace_at(term, p, r_r))
            yield NarrowStep(successor=successor, sigma=mgu,
                             position=p, rule_id=rule_id)


@dataclass(frozen=True)
class NarrowResult:
    """Outcome of a narrowing search. ``substitution`` is the answer (a dict
    {name: term} restricted to the original variables) when ``found``;
    ``derivation`` is the list of NarrowStep witnesses; ``exhausted`` is True
    when the node budget was hit (vs a genuinely finite exhausted tree)."""
    found: bool
    substitution: Optional[dict]
    derivation: list
    nodes_expanded: int
    exhausted: bool


def _freeze(t):
    return tuple(_freeze(x) for x in t) if isinstance(t, list) else t


def _key(term, theta):
    return (_freeze(term),
            frozenset((k, _freeze(v)) for k, v in theta.items()))


def _extract_rules(engine):
    """Analyzable first-order rules from ``engine`` as (l, r_term, rule_id)
    triples; the RHS skeleton is converted to a term via instantiate_skeleton.
    Non-analyzable rules (?c/?v/?free/?.../skeleton-compute) are skipped."""
    rules = []
    for _idx, rule, meta in engine.rule_set():
        l, skel = rule[0], rule[1]
        if not is_analyzable(l, skel, meta.condition):
            continue
        rules.append((l, instantiate_skeleton(skel, {}), meta.name))
    return rules


def _narrow_with_rules(rules, start, target, *,
                       max_nodes=1000, max_depth=20) -> "NarrowResult":
    """Budget-bounded BFS: find sigma such that sigma(start) narrows to a term
    unifying sigma(target). Returns the FIRST solution."""
    keep = _variables(start) | _variables(target)
    frontier = deque([(start, {}, 0, [])])
    seen = {_key(start, {})}
    nodes = 0
    while frontier:
        if nodes >= max_nodes:
            return NarrowResult(False, None, [], nodes, True)
        term, theta, depth, deriv = frontier.popleft()
        try:
            tau = unify(term, apply_subst(theta, target))
        except UnsupportedPattern:
            tau = None
        if tau is not None:
            sigma = {k: v for k, v in _compose(tau, theta).items() if k in keep}
            return NarrowResult(True, sigma, deriv, nodes, False)
        nodes += 1
        if depth < max_depth:
            for step in narrow_step(term, rules):
                theta2 = _compose(step.sigma, theta)
                k = _key(step.successor, theta2)
                if k not in seen:
                    seen.add(k)
                    frontier.append((step.successor, theta2, depth + 1,
                                     deriv + [step]))
    return NarrowResult(False, None, [], nodes, False)


def narrow(engine, start, target, *, max_nodes=1000, max_depth=20) -> NarrowResult:
    """Reachability narrowing over ``engine``'s analyzable rules: find sigma
    such that sigma(start) reduces to a term unifying sigma(target). Read-only;
    SYNTACTIC (ignores any loaded theory). See module docstring."""
    return _narrow_with_rules(_extract_rules(engine), start, target,
                              max_nodes=max_nodes, max_depth=max_depth)
