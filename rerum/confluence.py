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
    as-is. A bare ``["?", name]`` in a skeleton is left LITERAL (it falls
    through to the compound branch and rebuilds unchanged), matching the
    engine's own ``instantiate``, which substitutes only ``:`` references.
    Non-trivial skeleton forms (:.../!/fresh) never reach here because such
    rules are refused upstream (see ``is_analyzable``).
    """
    if skeleton_evaluation(skeleton):  # [":", name]
        return apply_subst(sigma, ["?", skeleton[1]])
    if compound(skeleton):
        return [instantiate_skeleton(sub, sigma) for sub in skeleton]
    return skeleton


# ---------------------------------------------------------------------------
# Rename-apart and analyzability pre-scan
# ---------------------------------------------------------------------------

_PATTERN_BAD = {"?c", "?v", "?free", "?..."}
_SKELETON_BAD = {":...", "!", "fresh"}


def _variables(term: ExprType) -> set:
    """The set of ``["?", name]`` variable names occurring in ``term``."""
    out: set = set()

    def walk(t: ExprType) -> None:
        if _is_var(t):
            out.add(t[1])
        elif compound(t):
            for sub in t:
                walk(sub)

    walk(term)
    return out


def _rename(term: ExprType, mapping: Dict[str, str]) -> ExprType:
    """Rename ``["?", name]`` and ``[":", name]`` nodes per ``mapping``."""
    if _is_var(term):  # ["?", name]
        return ["?", mapping.get(term[1], term[1])]
    if skeleton_evaluation(term):  # [":", name]
        return [":", mapping.get(term[1], term[1])]
    if compound(term):
        return [_rename(sub, mapping) for sub in term]
    return term


def rename_apart(pattern: ExprType, skeleton: ExprType,
                 avoid: set) -> Tuple[ExprType, ExprType]:
    """Return ``(pattern', skeleton')`` with every rule variable renamed to a
    fresh name not in ``avoid`` (and distinct from each other). Renames both
    the pattern's ``["?", name]`` binders and the skeleton's ``[":", name]``
    references with the SAME mapping."""
    names = _variables(pattern) | _variables(skeleton)
    mapping: Dict[str, str] = {}
    used = set(avoid)
    for n in sorted(names):
        fresh = gensym(n, used)
        mapping[n] = fresh
        used.add(fresh)
    return _rename(pattern, mapping), _rename(skeleton, mapping)


def _has_marker(term: ExprType, heads: set) -> bool:
    if compound(term):
        if isinstance(term[0], str) and term[0] in heads:
            return True
        return any(_has_marker(sub, heads) for sub in term)
    return False


def is_analyzable(pattern: ExprType, skeleton: ExprType,
                  condition: Optional[ExprType]) -> bool:
    """True iff F2 can soundly analyze this directed rule: unconditional, a
    first-order pattern (no ?c/?v/?free/?...), and a plain-substitution
    skeleton (no :.../!/fresh)."""
    if condition is not None:
        return False
    if _has_marker(pattern, _PATTERN_BAD):
        return False
    if _has_marker(skeleton, _SKELETON_BAD):
        return False
    return True


# ---------------------------------------------------------------------------
# Critical-pair computation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DirectedRule:
    """A single directed reduction rule (post-desugar)."""
    name: Optional[str]
    pattern: ExprType
    skeleton: ExprType
    condition: Optional[ExprType] = None


@dataclass(frozen=True)
class CriticalPair:
    """An overlap between two rule LHSs. ``left``/``right`` are the two reducts
    of the overlapped term; ``joinable`` is filled by ``check_confluence``
    (None until then, or when undecidable).

    NOTE: ``frozen=True`` protects against mutation but does NOT make instances
    hashable -- ``left``/``right`` are lists. Do not put a ``CriticalPair`` in a
    set or use it as a dict key.
    """
    left: ExprType
    right: ExprType
    rule_left: Optional[str]
    rule_right: Optional[str]
    position: Position
    joinable: Optional[bool] = None


def critical_pairs(
    rules: List[DirectedRule],
) -> Tuple[List[CriticalPair], List[str]]:
    """Compute the critical pairs of ``rules`` (the standard superposition).

    Returns ``(pairs, not_analyzed)`` where ``not_analyzed`` lists the names of
    rules skipped (conditional or non-first-order), deduplicated in order.
    """
    pairs: List[CriticalPair] = []
    not_analyzed: List[str] = []
    seen_skips: set = set()

    def skip(rule: DirectedRule) -> None:
        key = rule.name
        if key not in seen_skips:
            seen_skips.add(key)
            not_analyzed.append(key)

    analyzable = []
    for r in rules:
        if is_analyzable(r.pattern, r.skeleton, r.condition):
            analyzable.append(r)
        else:
            skip(r)

    # Ordered (i, j) iteration: every directed overlap is generated, including
    # the joinability-symmetric mirror of a root overlap between two DISTINCT
    # rules ((Ri,Rj) at () and (Rj,Ri) at ()). The redundancy is cosmetic --
    # joinability is symmetric -- so the verdict is unaffected; check_confluence
    # does NOT deduplicate (CriticalPair is not hashable). Non-root overlaps are
    # genuinely distinct and must all be kept.
    for i, ri in enumerate(analyzable):
        avoid = free_symbols(ri.pattern) | free_symbols(ri.skeleton)
        for j, rj in enumerate(analyzable):
            # Rj is renamed apart ONCE per (i, j): Ri is fixed across the
            # position loop, so one fresh copy keeps the variables disjoint at
            # every position p.
            rj_pat, rj_sk = rename_apart(rj.pattern, rj.skeleton, avoid)
            for p in positions(ri.pattern):
                if i == j and p == ():
                    continue  # trivial root self-overlap
                try:
                    sigma = unify(subterm_at(ri.pattern, p), rj_pat)
                except UnsupportedPattern:
                    # Defense-in-depth: unreachable while is_analyzable
                    # pre-scans the whole pattern (positions() also skips
                    # variable nodes). Kept so unify staying authoritative on
                    # the first-order fragment is not silently bypassed.
                    skip(ri)
                    continue
                if sigma is None:
                    continue
                u = apply_subst(sigma, ri.pattern)
                left = instantiate_skeleton(ri.skeleton, sigma)
                rj_rhs = instantiate_skeleton(rj_sk, sigma)
                right = replace_at(u, p, rj_rhs)
                pairs.append(CriticalPair(
                    left=left, right=right,
                    rule_left=ri.name, rule_right=rj.name, position=p,
                ))

    return pairs, not_analyzed


# ---------------------------------------------------------------------------
# Joinability and the confluence report
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConfluenceReport:
    """Result of a local-confluence check.

    ``locally_confluent`` is True iff NO critical pair is non-joinable and NONE
    is undecidable (an empty critical-pair set is vacuously True). This is LOCAL
    confluence only: global confluence additionally requires termination
    (Newman's Lemma; that is roadmap F4). A ``joinable is False`` means "not
    joinable by the engine's own reduction (this strategy)", the right notion
    for a confluence DEFECT report. ``unknown`` pairs (reduction hit the budget
    or a cycle) are never counted as joinable.
    """
    locally_confluent: bool
    critical_pairs: List[CriticalPair]
    non_joinable: List[CriticalPair]
    unknown: List[CriticalPair]
    not_analyzed: List[str]
    analyzed_pair_count: int


def _is_normal_form(engine, term: ExprType) -> bool:
    """A term is a normal form iff one recursive single-rule pass changes
    nothing. Uses the engine's ``_simplify_once`` (which applies at most one
    rule ANYWHERE in the tree), NOT the root-only ``apply_once``."""
    return engine._simplify_once(term) == term


def _decide_joinable(engine, cp: CriticalPair, max_steps: int) -> Optional[bool]:
    """Decide joinability of ``cp`` via the ENGINE's own reduction.

    Reduces both legs with ``engine.simplify`` (the real reduction relation --
    all enabled rules, including conditional ones the CP generator could not
    analyze) and compares modulo the loaded theory via ``engine._canonicalize``.
    Canonical-EQUALITY is checked FIRST, so a pair that reaches a common form
    within budget is joinable regardless of normal-form status.

    Returns True (joinable), False (distinct normal forms under the engine's
    reduction), or None (unknown: budget/cycle/recursion limit, not decided).
    """
    try:
        s2 = engine.simplify(cp.left, max_steps=max_steps)
        t2 = engine.simplify(cp.right, max_steps=max_steps)
        if engine._canonicalize(s2) == engine._canonicalize(t2):
            return True  # common reduct (modulo theory) -- checked FIRST
        if _is_normal_form(engine, s2) and _is_normal_form(engine, t2):
            return False  # distinct normal forms under the engine's reduction
        return None  # undecided within the budget
    except RecursionError:
        # A pathological unbounded-growth rule can build a term deep enough to
        # exceed Python's recursion limit during reduction before the step
        # budget trips. That is an UNDECIDED pair, not a defect -- map it to
        # unknown (the sound verdict) so the call returns rather than raising.
        return None


def check_confluence(engine, *, max_steps: int = 1000) -> ConfluenceReport:
    """Compute the engine's critical pairs, decide joinability of each, and
    return a local-confluence report. Read-only: mutates nothing."""
    records = [
        DirectedRule(name=meta.name, pattern=rule[0], skeleton=rule[1],
                     condition=meta.condition)
        for _idx, rule, meta in engine.rule_set()
    ]
    raw_pairs, not_analyzed = critical_pairs(records)

    decided: List[CriticalPair] = []
    non_joinable: List[CriticalPair] = []
    unknown: List[CriticalPair] = []
    analyzed = 0
    for cp in raw_pairs:
        verdict = _decide_joinable(engine, cp, max_steps)
        cp2 = CriticalPair(left=cp.left, right=cp.right,
                           rule_left=cp.rule_left, rule_right=cp.rule_right,
                           position=cp.position, joinable=verdict)
        decided.append(cp2)
        if verdict is True:
            analyzed += 1
        elif verdict is False:
            analyzed += 1
            non_joinable.append(cp2)
        else:
            unknown.append(cp2)

    locally_confluent = not non_joinable and not unknown
    return ConfluenceReport(
        locally_confluent=locally_confluent,
        critical_pairs=decided,
        non_joinable=non_joinable,
        unknown=unknown,
        not_analyzed=not_analyzed,
        analyzed_pair_count=analyzed,
    )
