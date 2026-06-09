"""Theory-driven canonical normalization: flatten, sort, collect, fold.

A traceable normalization pass (spec Section 5.2). Pure functions over the
nested-list ``ExprType`` from ``rewriter.py``. WHICH operators are
associative-commutative, their identity/annihilator units, and how repeated
operands combine, are all DATA carried by a ``Theory``; this module hardcodes
no domain. The engine ships no built-in theory naming ``+``/``*``. The same
functions normalize arithmetic, boolean algebra, or any AC theory with no
change. With an empty ``Theory`` ``normalize`` is the identity.

Idempotent: ``normalize(normalize(e, t), t) == normalize(e, t)``.
Confluent: two orderings of the same operand multiset normalize equal.

Imports nothing from ``engine.py`` (circular-import constraint); the constant
fold is local and uses the theory's units.
"""

import json as _json
from typing import Any, Callable, Dict, List, Optional

from .rewriter import ExprType, compound, constant, variable


class Theory:
    """Operator signature: which ops are AC, their units, repeat rule.

    Constructed from a dict of the shape::

        {"+": {"ac": True, "identity": 0,
               "repeat": {"op": "*", "via": "count"}},
         "*": {"ac": True, "identity": 1, "annihilator": 0,
               "repeat": {"op": "^", "via": "exp"}}}

    An absent key means the operator is not AC and has no units. ``repeat``
    declares (as data) how to combine k copies of an operand under an AC op:
    ``via="count"`` builds ``(op k base)`` (coefficient form), ``via="exp"``
    builds ``(op base k)`` (power form). No ``repeat`` means repeats collapse
    to one copy (idempotent operators).
    """

    __slots__ = ("_sig",)

    def __init__(self, sig: Dict[str, Dict[str, Any]]):
        self._sig = dict(sig or {})

    @classmethod
    def from_dict(cls, d: Dict[str, Dict[str, Any]]) -> "Theory":
        return cls(d)

    @classmethod
    def from_json(cls, text: str) -> "Theory":
        return cls(_json.loads(text))

    def is_ac(self, op) -> bool:
        entry = self._sig.get(op)
        return bool(entry) and bool(entry.get("ac", False))

    def identity(self, op):
        entry = self._sig.get(op)
        return entry.get("identity") if entry else None

    def annihilator(self, op):
        entry = self._sig.get(op)
        return entry.get("annihilator") if entry else None

    def repeat(self, op) -> Optional[Dict[str, Any]]:
        entry = self._sig.get(op)
        return entry.get("repeat") if entry else None

    def __repr__(self) -> str:
        return f"Theory({sorted(self._sig)})"


def _is_number(x) -> bool:
    """True for any numeric atom (int, float, Fraction). bool excluded."""
    return constant(x) and not isinstance(x, bool)


# ---------------------------------------------------------------------------
# Stubs — replaced task-by-task
# ---------------------------------------------------------------------------

def flatten(expr: ExprType, theory: Theory) -> ExprType:
    """Recursively make AC operators n-ary, per the theory.

    ``(+ (+ a b) c)`` becomes ``(+ a b c)`` when ``theory.is_ac("+")``. A
    nested operand is merged into its parent only when their heads match and
    the head is AC under ``theory``. Children of every compound are flattened
    first. Atoms are returned unchanged. With an empty theory nothing merges.
    """
    if not compound(expr) or not expr:
        return expr

    head = expr[0]
    flat_args = [flatten(a, theory) for a in expr[1:]]

    if theory.is_ac(head):
        merged: List[ExprType] = []
        for a in flat_args:
            if compound(a) and a and a[0] == head:
                merged.extend(a[1:])
            else:
                merged.append(a)
        return [head] + merged

    return [head] + flat_args


def ORDER_KEY(expr: ExprType) -> tuple:
    raise NotImplementedError("ORDER_KEY not yet implemented")


def canonical_sort(expr: ExprType, theory: "Theory") -> ExprType:
    raise NotImplementedError("canonical_sort not yet implemented")


def collect_like_terms(expr: ExprType, theory: "Theory") -> ExprType:
    raise NotImplementedError("collect_like_terms not yet implemented")


def normalize(expr: ExprType, theory: "Theory", *, listener=None) -> ExprType:
    raise NotImplementedError("normalize not yet implemented")
