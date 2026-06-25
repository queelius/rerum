"""AC-unification (Stickel's algorithm) -- unification modulo associative-
commutative operators.

The multi-valued, two-sided generalization of F2's syntactic ``unify``: where
``unify`` returns one most-general unifier or none, AC-unification has no mgu, so
``ac_unify`` yields a COMPLETE SET of unifiers (within a budget; not necessarily
minimal). PURE AC -- the Theory's identity/unit is NOT used (the engine's
normalize strips units upstream). Stickel-general: nested terms, free function
symbols, and multiple AC operators are handled (the last two by recursion).

PURE AC means MULTISET semantics: ``a + a`` is NOT equal to ``a`` (no
idempotence). A Theory with a bare ``{"ac": True}`` (no ``repeat`` clause) makes
``normalize`` treat the operator as IDEMPOTENT (ACI: ``a + a -> a``); ac_unify
does NOT model that idempotence, so against such a theory its unifiers are a
SOUND SUBSET of the (larger) ACI-unifier set. Idempotent / ACI unification is
out of scope here, like ACU (unification modulo the unit). To reason about
ac_unify's results via ``normalize``, use a multiset-preserving theory (one
with a ``repeat`` clause).

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
    unify,
    _compose_bind,
    _occurs,
    _is_var,
    _unsupported,
    _variables,
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


def _bind(subst: Subst, name: str, value) -> Optional[Subst]:
    """Add ``name -> value`` (resolved) with an occurs-check, kept fully-applied
    via F2's ``_compose_bind``. Returns the new Subst or None on occurs-check
    failure."""
    if _is_var(value) and value[1] == name:
        return subst
    if _occurs(name, value):
        return None
    return _compose_bind(subst, name, value)


def ac_unify(t1, t2, theory, *, bindings: Optional[Subst] = None,
             budget: Optional[UnifyBudget] = None) -> Iterator[Subst]:
    """Yield each AC-unifier of ``t1`` and ``t2`` modulo the AC operators in
    ``theory``. Multi-valued and lazy. Pure AC. Complete within ``budget``,
    not necessarily minimal."""
    if bindings is None:
        bindings = {}
    if _unsupported(t1) or _unsupported(t2):
        raise UnsupportedPattern(f"cannot ac-unify pattern form: {t1!r} ~ {t2!r}")
    a = apply_subst(bindings, t1)
    b = apply_subst(bindings, t2)

    if _is_var(a):
        s = _bind(bindings, a[1], b)
        if s is not None:
            yield s
        return
    if _is_var(b):
        s = _bind(bindings, b[1], a)
        if s is not None:
            yield s
        return
    if not compound(a) and not compound(b):
        if a == b:
            yield bindings
        return
    if compound(a) and compound(b) and a[0] == b[0] and theory.is_ac(a[0]):
        yield from _ac_unify_node(a, b, theory, bindings, budget)
        return
    if compound(a) and compound(b) and a[0] == b[0] and len(a) == len(b):
        yield from _unify_positional(a[1:], b[1:], theory, bindings, budget)
        return
    return  # head/arity clash, or atom vs compound


def _unify_positional(xs, ys, theory, bindings, budget) -> Iterator[Subst]:
    """Unify two argument lists in lockstep, MULTI-VALUED (a child may be an AC
    node). The two-sided analog of acmatch._match_positional."""
    if not xs:
        if not ys:
            yield bindings
        return
    if not ys:
        return
    for s in ac_unify(xs[0], ys[0], theory, bindings=bindings, budget=budget):
        yield from _unify_positional(xs[1:], ys[1:], theory, s, budget)


@dataclass
class _Atoms:
    """One side of a Stickel problem after grouping: the parallel arrays
    ``terms`` (the distinct atoms), ``mult`` (each atom's multiplicity /
    Diophantine coefficient), and ``isv`` (whether each atom is a variable).
    Indexed positionally; ``len(terms) == len(mult) == len(isv)``."""
    terms: list
    mult: List[int]
    isv: List[bool]


def _group(atoms) -> "_Atoms":
    """Group equal VARIABLE atoms (accumulating multiplicity); keep each
    NON-VARIABLE occurrence as its OWN weight-1 atom. A non-variable atom (a
    constant or compound) cannot be 'split' in the Stickel encoding -- it
    occupies a single coefficient-1 column per occurrence. Merging duplicates
    (e.g. ``a + a``) into one column of multiplicity 2 lets the variable side
    absorb that weight more than once and yields UNSOUND unifiers. Variables DO
    carry multiplicity (``x + x`` is the coefficient-2 atom ``2x``). Returns an
    ``_Atoms`` bundle (terms, multiplicities, is_var flags)."""
    terms: list = []
    mult: List[int] = []
    isv: List[bool] = []
    for t in atoms:
        if _is_var(t):
            for k, tt in enumerate(terms):
                if isv[k] and tt == t:
                    mult[k] += 1
                    break
            else:
                terms.append(t)
                mult.append(1)
                isv.append(True)
        else:
            terms.append(t)
            mult.append(1)
            isv.append(False)
    return _Atoms(terms, mult, isv)


def _ac_sum(op, parts):
    return parts[0] if len(parts) == 1 else [op] + parts


def _ac_unify_node(a, b, theory, bindings, budget) -> Iterator[Subst]:
    """Stickel AC-unification of two terms headed by the same AC operator."""
    op = a[0]
    left = flatten(a, theory)[1:]
    right = flatten(b, theory)[1:]
    # Cancel the multiset intersection (syntactically-equal common args).
    lhs_args = list(left)
    rhs_args = list(right)
    for item in list(lhs_args):
        if item in rhs_args:
            lhs_args.remove(item)
            rhs_args.remove(item)
    if not lhs_args and not rhs_args:
        yield bindings
        return
    if not lhs_args or not rhs_args:
        return  # pure AC: a non-empty sum cannot equal nothing

    lhs = _group(lhs_args)
    rhs = _group(rhs_args)
    basis = _hilbert_basis(lhs.mult, rhs.mult)
    M, N = len(lhs.terms), len(rhs.terms)
    # Seed the fresh-variable avoid set with every name that could collide: the
    # bound names and the variables in their values, plus the free variables of
    # both terms. Without this, a recursive AC node (a coupling whose atoms are
    # themselves AC) re-gensyms the OUTER node's z names and captures them.
    avoid: set = set(bindings.keys()) | _variables(a) | _variables(b)
    for val in bindings.values():
        avoid |= _variables(val)
    z = [["?", gensym("z_" + str(k), avoid)] for k in range(len(basis))]
    for zk in z:
        avoid.add(zk[1])

    for r in range(1, len(basis) + 1):
        for subset in combinations(range(len(basis)), r):
            if budget is not None and not budget.spend():
                return
            ok = True
            for i in range(M):
                tot = sum(basis[k][i] for k in subset)
                if (lhs.isv[i] and tot < 1) or (not lhs.isv[i] and tot != lhs.mult[i]):
                    ok = False
                    break
            if ok:
                for j in range(N):
                    tot = sum(basis[k][M + j] for k in subset)
                    if (rhs.isv[j] and tot < 1) or (not rhs.isv[j] and tot != rhs.mult[j]):
                        ok = False
                        break
            if not ok:
                continue
            yield from _build_unifier(lhs, rhs, basis, subset, z, op,
                                      theory, bindings, budget)


def _build_unifier(lhs: "_Atoms", rhs: "_Atoms", basis, subset, z, op,
                   theory, bindings, budget) -> Iterator[Subst]:
    """Build the unifier(s) for one admissible covering subset. Variable atoms
    bind to AC-sums of their fresh z's; non-variable atoms sharing a z are
    recursively ac_unify'd (Stickel-general, multi-valued -> a product)."""
    M = len(lhs.terms)
    s: Optional[Subst] = dict(bindings)
    for i in range(len(lhs.terms)):
        if not lhs.isv[i]:
            continue
        parts = [z[k] for k in subset for _ in range(basis[k][i])]
        s = unify(lhs.terms[i], _ac_sum(op, parts), s)
        if s is None:
            return
    for j in range(len(rhs.terms)):
        if not rhs.isv[j]:
            continue
        parts = [z[k] for k in subset for _ in range(basis[k][M + j])]
        s = unify(rhs.terms[j], _ac_sum(op, parts), s)
        if s is None:
            return
    # For each chosen basis vector, collect the NON-VARIABLE atoms it couples
    # (the fresh z must equal each). Recursively ac_unify all atoms sharing a z;
    # the product over basis vectors yields the unifier set.
    couplings = []
    for k in subset:
        atoms = [z[k]]
        for i in range(len(lhs.terms)):
            if not lhs.isv[i] and basis[k][i]:
                atoms += [lhs.terms[i]] * basis[k][i]
        for j in range(len(rhs.terms)):
            if not rhs.isv[j] and basis[k][M + j]:
                atoms += [rhs.terms[j]] * basis[k][M + j]
        if len(atoms) > 1:
            couplings.append(atoms)
    yield from _resolve_couplings(couplings, 0, s, theory, budget)


def _resolve_couplings(couplings, idx, subst, theory, budget) -> Iterator[Subst]:
    """Recursively ac_unify every atom within each coupling group against the
    group's fresh z (atoms[0]); product over groups. Threads the substitution."""
    if subst is None:
        return
    if idx == len(couplings):
        yield subst
        return
    group = couplings[idx]

    def chain(pos, s) -> Iterator[Subst]:
        if pos == len(group):
            yield from _resolve_couplings(couplings, idx + 1, s, theory, budget)
            return
        for s2 in ac_unify(group[0], group[pos], theory, bindings=s, budget=budget):
            yield from chain(pos + 1, s2)
    yield from chain(1, subst)
