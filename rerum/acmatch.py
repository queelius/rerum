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
