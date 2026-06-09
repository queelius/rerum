"""General numeric evaluation of ground terms under a prelude.

GENERAL ENGINE PRINCIPLE: this module special-cases NO operator. A term is
interpreted using only the fold functions in the supplied prelude, exactly
the same extension point the rewriter uses for constant folding. Swap the
prelude (arithmetic, math, a boolean prelude) and the same machinery
evaluates a different algebra with no code change.

A prelude is a ``Dict[str, FoldHandler]`` where a ``FoldHandler`` is
``Callable[[List[number]], Optional[number]]`` (see ``rewriter.py``). So a
compound ``[op, a, b, ...]`` is evaluated by recursively evaluating each
argument to a number, then calling ``prelude[op]([va, vb, ...])`` -- the
handler takes a single list of already-evaluated arguments, not varargs.

Domain validators (is-this-the-derivative-of-that, etc.) do NOT live here;
they are example content that CALLS ``numeval``/``numeric_equiv``.
"""

import random
from fractions import Fraction
from typing import Callable, Dict, Mapping, Union

from .rewriter import ExprType

# A numeric value: a Python int/float, or an exact Fraction (safe_div in the
# arithmetic prelude returns a Fraction for an exact rational quotient).
Number = Union[int, float, Fraction]


class NumevalError(Exception):
    """Raised when a ground term cannot be numerically evaluated.

    Covers an unbound symbol, an operator absent from the prelude, a
    handler that returns ``None`` (cannot fold), and a handler that raises
    a domain error (e.g. ``log`` of a negative).
    """


def numeval(expr: ExprType, env: Mapping[str, Number], prelude: Dict) -> Number:
    """Evaluate a ground term to a number using ``prelude``'s fold funcs.

    Args:
        expr: A term. After substituting ``env`` for symbol leaves it must
            be ground (every symbol bound, every operator in ``prelude``).
        env: Maps symbol names to numbers.
        prelude: A ``Dict[str, FoldHandler]`` (e.g. ``ARITHMETIC_PRELUDE``,
            ``MATH_PRELUDE``). The ONLY source of operator semantics; no
            operator is special-cased.

    Returns:
        The numeric value of ``expr``.

    Raises:
        NumevalError: on an unbound symbol, an undefined operator, a
            non-foldable result, or a domain error from a handler.
    """
    # Atoms: a number is itself; a symbol is looked up in env.
    # bool is a subclass of int; preserve it verbatim (no numeric coercion)
    # so a caller passing a boolean env value or a predicate result is not
    # silently widened.
    if isinstance(expr, bool):
        return expr
    if isinstance(expr, (int, float, Fraction)):
        return expr
    if isinstance(expr, str):
        if expr in env:
            return env[expr]
        raise NumevalError(f"unbound symbol: {expr!r}")

    if isinstance(expr, list):
        if not expr or not isinstance(expr[0], str):
            raise NumevalError(f"not an evaluable compound: {expr!r}")
        op = expr[0]
        handler = prelude.get(op)
        if handler is None:
            raise NumevalError(f"undefined operator: {op!r}")
        # Recursively evaluate each argument, then hand the prelude the
        # evaluated-args LIST (single positional list arg, not varargs).
        args = [numeval(arg, env, prelude) for arg in expr[1:]]
        try:
            value = handler(args)
        except NumevalError:
            raise
        except Exception as exc:  # domain error, e.g. log of a negative
            raise NumevalError(
                f"domain error evaluating ({op} ...): {exc}"
            ) from exc
        if value is None:
            raise NumevalError(
                f"operator {op!r} could not fold args {args!r}"
            )
        return value

    raise NumevalError(f"not an evaluable term: {expr!r}")
