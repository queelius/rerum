"""General fold predicates the limit example rules require: subst,
defined-at?, indeterminate?. These are computation (data/config), declared
by the limits demonstration. The engine never imports this module.

All handlers follow rerum's fold convention (one evaluated-args list in).
``defined-at?`` and ``indeterminate?`` evaluate via the GENERAL
``rerum.numeval`` over a math prelude; any out-of-domain or non-ground
evaluation (numeval raises) simply answers False, which is the correct
guard semantics (an undefined point is not "defined at", and a
non-evaluable side is not provably indeterminate).
"""

import math

from rerum.numeval import numeval
from rerum.rewriter import MATH_PRELUDE


def _subst_expr(body, var, value):
    if isinstance(body, str):
        return value if body == var else body
    if isinstance(body, list):
        return [_subst_expr(sub, var, value) for sub in body]
    return body


def _subst(args):
    """(! subst body var value): body with var := value. Returns the
    substituted EXPRESSION (folds may return structures, not just numbers)."""
    if len(args) != 3:
        return None
    body, var, value = args
    return _subst_expr(body, var, value)


def _defined_at(args):
    """(! defined-at? body var point): True iff body evaluates to a finite
    number after substituting var := point."""
    if len(args) != 3:
        return False
    body, var, point = args
    try:
        val = numeval(_subst_expr(body, var, point), {}, MATH_PRELUDE)
    except Exception:
        return False
    try:
        return math.isfinite(float(val))
    except (TypeError, ValueError):
        return False


def _indeterminate(args):
    """(! indeterminate? num den var point): True iff both num and den
    evaluate to 0 at the point (the 0/0 form L'Hopital applies to)."""
    if len(args) != 4:
        return False
    num, den, var, point = args
    try:
        n = numeval(_subst_expr(num, var, point), {}, MATH_PRELUDE)
        d = numeval(_subst_expr(den, var, point), {}, MATH_PRELUDE)
    except Exception:
        return False
    try:
        return abs(float(n)) < 1e-12 and abs(float(d)) < 1e-12
    except (TypeError, ValueError):
        return False


LIMIT_FOLD_OPS = {
    "subst": _subst,
    "defined-at?": _defined_at,
    "indeterminate?": _indeterminate,
}
