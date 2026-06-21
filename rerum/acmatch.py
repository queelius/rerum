"""F3: AC-matching proper -- matching modulo associativity and commutativity.

This module is the multiset-partition matcher that F1's normalize pass cannot
provide (canonical order is not pattern unification). It is PURE: it takes the
``Theory`` carrier (which declares, as DATA, which operators are AC) and reuses
the matcher predicates from ``rewriter.py`` plus ``flatten``/``ORDER_KEY``/
``normalize`` from ``normalize.py``. It names NO domain operator.

``ac_match`` is MULTI-VALUED: an AC pattern can match a subject several ways, so
it yields each consistent ``Bindings`` lazily. ``match()`` in ``rewriter.py`` is
unchanged; the engine routes to ``ac_match`` only when an AC theory is loaded.

Scope: MATCHING only (pattern has variables, subject is a concrete term). NOT
AC-unification (both sides have variables), which F2/F5 would need. NOT ACU
(matching modulo identity).
"""

from dataclasses import dataclass
from typing import Iterator, Optional

from .rewriter import (
    Bindings,
    atom,
    compound,
    car,
    arbitrary_constant,
    arbitrary_variable,
    arbitrary_expression,
    arbitrary_free,
    arbitrary_rest,
    variable_name,
    constant,
    variable,
    free_in,
)
from .normalize import flatten, ORDER_KEY, normalize


@dataclass
class MatchBudget:
    """Fail-safe work budget for AC enumeration.

    ``steps`` counts assignment attempts; each ``spend()`` decrements it. When it
    reaches zero, further ``spend()`` calls return False and ``truncated`` is set,
    so the matcher stops enumerating. ``steps=None`` means unbounded. Truncation
    bounds COMPLETENESS only: every match already yielded is still valid.
    """

    steps: Optional[int] = None
    truncated: bool = False

    def spend(self) -> bool:
        """Consume one unit. Return True if budget remained, False if exhausted."""
        if self.steps is None:
            return True
        if self.steps <= 0:
            self.truncated = True
            return False
        self.steps -= 1
        return True


def _canon_eq(a, b, theory) -> bool:
    """Equality of two terms modulo the theory's canonical form."""
    if a == b:
        return True
    return normalize(a, theory) == normalize(b, theory)


def _bind(bindings: Bindings, name: str, value, theory) -> Optional[Bindings]:
    """Extend ``bindings`` with name->value, consistent MODULO the theory.

    Returns the (possibly same) Bindings on success, or None on a conflicting
    re-binding. Unlike ``Bindings.extend``, re-binding compares modulo AC so
    ``?x`` bound to ``(+ a b)`` still matches ``(+ b a)``.
    """
    if name in bindings:
        return bindings if _canon_eq(bindings[name], value, theory) else None
    return bindings.extend(name, value)


def _match_one(pat, exp, theory, bindings: Bindings) -> Optional[Bindings]:
    """Single-result match for a NON-rest, NON-AC-node pattern element.

    Handles atoms, the four single-variable forms, and delegates compounds to
    the first ac_match yield (a compound element has at most one match here in
    the non-AC path, but may be AC and multi-valued; callers that need every
    match use ac_match directly). Returns Bindings or None.
    """
    if atom(pat):
        return bindings if (atom(exp) and pat == exp) else None
    if arbitrary_constant(pat):
        return _bind(bindings, variable_name(pat), exp, theory) if constant(exp) else None
    if arbitrary_variable(pat):
        return _bind(bindings, variable_name(pat), exp, theory) if variable(exp) else None
    if arbitrary_expression(pat):
        return _bind(bindings, variable_name(pat), exp, theory)
    if arbitrary_free(pat):
        # Optimistic bind; the excluded var's containment is validated inline:
        # if it is already bound to a symbol present in exp, fail now.
        excluded = bindings.lookup(pat[2])
        if isinstance(excluded, str) and excluded != pat[2] and free_in(excluded, exp):
            return None
        return _bind(bindings, variable_name(pat), exp, theory)
    # Compound, non-AC element: take the unique first ac_match yield, if any.
    for b in ac_match(pat, exp, theory, bindings):
        return b
    return None


def ac_match(pat, exp, theory, bindings: Optional[Bindings] = None,
             budget: Optional["MatchBudget"] = None) -> Iterator[Bindings]:
    """Yield every Bindings under which ``pat`` matches ``exp`` modulo the AC
    operators declared in ``theory``. Multi-valued and lazy.

    ``bindings`` seeds the match (default: empty). ``budget`` is an optional
    ``MatchBudget`` fail-safe for the AC enumeration.
    """
    if bindings is None:
        bindings = Bindings.empty()

    # Atom / single-variable / ?free: zero or one result.
    if (atom(pat) or arbitrary_constant(pat) or arbitrary_variable(pat)
            or arbitrary_expression(pat) or arbitrary_free(pat)):
        result = _match_one(pat, exp, theory, bindings)
        if result is not None:
            yield result
        return

    # A bare rest pattern matched directly against a list binds the whole list.
    if arbitrary_rest(pat):
        if isinstance(exp, list):
            extended = _bind(bindings, variable_name(pat), exp, theory)
            if extended is not None:
                yield extended
        return

    # Compound pattern. AC multiset case is added in a later task; for now,
    # only the non-AC positional case is handled.
    if not compound(pat) or not isinstance(exp, list) or not exp:
        return
    if car(pat) != car(exp):
        return
    yield from _match_positional(pat[1:], exp[1:], theory, bindings, budget)


def _match_positional(pats, exps, theory, bindings, budget) -> Iterator[Bindings]:
    """Match a list of sub-patterns against a list of sub-expressions in
    lockstep (the non-AC compound case). Multi-valued: a child may be an AC
    node. A trailing ``?...`` rest captures the positional tail as a list.
    """
    if not pats:
        if not exps:
            yield bindings
        return
    head, tail = pats[0], pats[1:]
    if arbitrary_rest(head):
        # Rest must be last; capture the positional remainder as a list.
        if tail:
            raise ValueError("Rest pattern (?...) must be last")
        extended = _bind(bindings, variable_name(head), list(exps), theory)
        if extended is not None:
            yield extended
        return
    if not exps:
        return
    for b in ac_match(head, exps[0], theory, bindings, budget):
        yield from _match_positional(tail, exps[1:], theory, b, budget)
