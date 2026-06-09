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
# Normalization transforms: flatten -> sort -> collect -> fold
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


# Rank constants: the primary sort key. Strictly increasing so that
# numbers < symbols < compounds, regardless of payload contents.
_RANK_NUMBER = 0
_RANK_SYMBOL = 1
_RANK_COMPOUND = 2


def ORDER_KEY(expr: ExprType) -> tuple:
    """Domain-free structural total-order key for canonical sorting.

    Numbers sort before symbols before compounds. Numbers order by value
    (int/float/Fraction comparable via ``float``), symbols lexicographically,
    compounds by ``(head, then args recursively)``. The leading integer rank
    makes keys of different shapes always comparable without ``TypeError``.
    Takes no theory: this is pure structure, no domain knowledge.
    """
    if isinstance(expr, bool):
        return (_RANK_NUMBER, (float(expr), "bool"))
    if _is_number(expr):
        return (_RANK_NUMBER, (float(expr), type(expr).__name__))
    if variable(expr):
        return (_RANK_SYMBOL, (expr,))
    # compound: key by head (recursively, the normal head is a string) then args.
    head = expr[0] if expr else ""
    head_key = ORDER_KEY(head)
    arg_keys = tuple(ORDER_KEY(a) for a in expr[1:])
    return (_RANK_COMPOUND, (head_key, arg_keys))


def canonical_sort(expr: ExprType, theory: Theory) -> ExprType:
    """Sort operands of AC operators by ``ORDER_KEY``, per the theory.

    Recurses into every child. Operands of AC heads are reordered into
    ``ORDER_KEY`` order; non-AC heads keep operand order. Atoms unchanged.
    """
    if not compound(expr) or not expr:
        return expr

    head = expr[0]
    sorted_args = [canonical_sort(a, theory) for a in expr[1:]]

    if theory.is_ac(head):
        sorted_args = sorted(sorted_args, key=ORDER_KEY)

    return [head] + sorted_args


def _read_multiplicity(operand: ExprType, repeat: Dict[str, Any]):
    """Read (base, count) from an operand given the theory's repeat rule.

    ``via="count"``: ``(repeat_op k base)`` -> ``(base, k)``; bare ``b`` ->
    ``(b, 1)``. ``via="exp"``: ``(repeat_op base e)`` -> ``(base, e)``; bare
    ``b`` -> ``(b, 1)``. The shape is read from ``repeat``, never hardcoded.
    """
    rop = repeat["op"]
    via = repeat["via"]
    if compound(operand) and operand and operand[0] == rop:
        if via == "count" and len(operand) == 3 and _is_number(operand[1]):
            return operand[2], operand[1]
        if via == "exp" and len(operand) == 3 and _is_number(operand[2]):
            return operand[1], operand[2]
    return operand, 1


def _emit_group(base: ExprType, total, repeat: Optional[Dict[str, Any]]):
    """Re-emit a collected group as an operand (or None to drop it).

    ``repeat is None`` (idempotent op): a single ``base``. ``via="count"``:
    ``base`` if total 1, ``None`` if total 0, else ``(repeat_op total base)``.
    ``via="exp"``: ``base`` if total 1, else ``(repeat_op base total)``.
    """
    if repeat is None:
        return base
    rop = repeat["op"]
    via = repeat["via"]
    if via == "count":
        if total == 0:
            return None
        if total == 1:
            return base
        return [rop, total, base]
    # via == "exp"
    if total == 1:
        return base
    return [rop, base, total]


def _collect_ac(args: List[ExprType], op: str, theory: Theory) -> List[ExprType]:
    """Combine repeated operands under AC operator ``op`` using the theory."""
    repeat = theory.repeat(op)
    order: List[tuple] = []          # first-seen base keys
    groups: Dict[tuple, list] = {}   # key -> [total, base_expr]
    for operand in args:
        if repeat is None:
            base, count = operand, 1
        else:
            base, count = _read_multiplicity(operand, repeat)
        k = ORDER_KEY(base)
        if k not in groups:
            groups[k] = [count, base]
            order.append(k)
        else:
            groups[k][0] = groups[k][0] + count
    out: List[ExprType] = []
    for k in order:
        total, base = groups[k]
        emitted = _emit_group(base, total, repeat)
        if emitted is not None:
            out.append(emitted)
    return out


def collect_like_terms(expr: ExprType, theory: Theory) -> ExprType:
    """Combine repeated operands of AC operators using the theory's repeat rule.

    Theory-driven, no ``+``/``*``/``^`` literal: for ``+`` (repeat ``*`` count)
    ``x + x`` -> ``(* 2 x)``; for ``*`` (repeat ``^`` exp) ``x * x`` ->
    ``(^ x 2)``; for an idempotent boolean ``and`` (no repeat) ``(and a a)`` ->
    ``a``. Recurses into children first. A head left with a single operand
    unwraps. Non-AC heads keep their operands.
    """
    if not compound(expr) or not expr:
        return expr

    head = expr[0]
    args = [collect_like_terms(a, theory) for a in expr[1:]]

    if theory.is_ac(head):
        args = _collect_ac(args, head, theory)
        if len(args) == 1:
            return args[0]
        return [head] + args

    return [head] + args


def _same_atom(a, b) -> bool:
    """Equality that does not conflate bool with int (True != 1, False != 0).

    Python's ``True == 1`` and ``False == 0``.  This helper keeps bool
    identity/annihilator atoms (``True``, ``False``) distinct from the
    arithmetic integers ``1`` and ``0``.  Two bools compare with ``is``
    (identity), two non-bools compare with ``==`` (value equality, so
    ``int(0) == float(0.0)`` stays true as expected for an arithmetic
    theory).  Mixed bool/non-bool pairs are always unequal.
    """
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    return a == b


def _fold_constants(expr: ExprType, theory: Theory) -> ExprType:
    """Domain-agnostic identity/annihilator removal for AC operators.

    This is a STRUCTURAL/algebraic operation only.  Numeric evaluation
    (``(+ 2 3) -> 5``) is NOT performed here; that belongs to the
    engine's constant-folding prelude.  What this function does:

    1. Recurse into all children first.
    2. For an AC operator (``theory.is_ac(op)`` True):
       - If any arg equals the annihilator (``theory.annihilator(op)``,
         checked via ``_same_atom``), collapse the whole expression to
         the annihilator.
       - Drop any arg that equals the identity (``theory.identity(op)``,
         checked via ``_same_atom``).
       - If no args remain, return the identity (if defined) else
         ``[op]`` (empty application).
       - If exactly one arg remains, unwrap to that arg.
       - Otherwise return ``[op] + remaining_args``.
    3. For a non-AC operator: return ``[op] + folded_children``.

    Uses ``_same_atom`` for all identity/annihilator comparisons so that
    boolean identities ``True``/``False`` are not conflated with integer
    ``1``/``0``.
    """
    if not compound(expr) or not expr:
        return expr

    head = expr[0]
    args = [_fold_constants(a, theory) for a in expr[1:]]

    if theory.is_ac(head):
        identity = theory.identity(head)
        annihilator = theory.annihilator(head)

        # Annihilator present among operands: the whole operator collapses.
        if annihilator is not None and any(_same_atom(a, annihilator) for a in args):
            return annihilator

        # Drop identity elements.
        if identity is not None:
            remaining = [a for a in args if not _same_atom(a, identity)]
        else:
            remaining = list(args)

        if not remaining:
            # All args were the identity (or there were none): return identity
            # if defined, otherwise keep the empty application.
            return identity if identity is not None else [head]
        if len(remaining) == 1:
            return remaining[0]
        return [head] + remaining

    return [head] + args


class _NormalizeMeta:
    """Minimal metadata stand-in for normalize steps.

    ``RewriteStep`` reads ``metadata.name`` and ``metadata.description`` for
    its repr/``to_dict``; ``rationale`` derives from ``reasoning``/``category``.
    A full ``RuleMetadata`` is not needed for normalization steps.
    """

    __slots__ = ("name", "description", "reasoning", "category")

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.reasoning = None
        self.category = "normalize"

    def __repr__(self) -> str:
        # RewriteTrace.__repr__ interpolates the metadata directly; without
        # this a normalize step would print as a bare object address. Mirror
        # the readable repr a full RuleMetadata gives.
        return self.name


def _emit(listener, name, description, before, after):
    """Emit a kind="normalize" RewriteStep if before != after."""
    if listener is None or before == after:
        return
    from .trace import RewriteStep
    step = RewriteStep(
        rule_index=-1,
        metadata=_NormalizeMeta(name, description),
        before=before,
        after=after,
        kind="normalize",
    )
    listener(step)


def _normalize_once(expr: ExprType, theory: Theory,
                    listener: Optional[Callable] = None) -> ExprType:
    """One full pass, emitting a step per changed sub-transformation."""
    flat = flatten(expr, theory)
    _emit(listener, "normalize:flatten", "Flatten AC operators", expr, flat)

    srt = canonical_sort(flat, theory)
    _emit(listener, "normalize:sort", "Sort AC operands", flat, srt)

    col = collect_like_terms(srt, theory)
    _emit(listener, "normalize:collect", "Collect like terms", srt, col)

    fld = _fold_constants(col, theory)
    _emit(listener, "normalize:fold", "Fold constants", col, fld)

    return fld


def normalize(expr: ExprType, theory: Theory, *,
              listener: Optional[Callable] = None) -> ExprType:
    """Canonical normal form under ``theory``: flatten->sort->collect->fold, to fixpoint.

    Idempotent and confluent. With an empty ``Theory`` it is the identity.
    When ``listener`` is provided, a ``kind="normalize"`` ``RewriteStep`` is
    emitted per changed sub-transformation (see Task 7).
    """
    current = expr
    for _ in range(1000):  # fixpoint bound; mirrors rewriter's iteration cap
        nxt = _normalize_once(current, theory, listener=listener)
        if nxt == current:
            break
        current = nxt
    return current
