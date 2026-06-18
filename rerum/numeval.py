"""General numeric evaluation of ground terms under a prelude.

OPTIONAL, NON-CORE LAYER. This is INTERPRETATION (evaluating a ground term
in a numeric model), NOT term rewriting. It is deliberately not re-exported
from the `rerum` core API; import it explicitly (`from rerum.numeval import
numeval`). It backs numeric verification (e.g. the example calculus
checkers) and the optional search layer, not the rewriting core.

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
    """Base: a ground term cannot be numerically evaluated.

    The taxonomy distinguishes point-INDEPENDENT *structural* failures from
    point-DEPENDENT *domain* failures, because :func:`numeric_equiv` treats
    them differently:

    - Structural (this base class): an unbound symbol, an operator absent
      from the prelude, a malformed term. These do not depend on the sample
      point -- they signal a malformed query -- so ``numeric_equiv`` lets
      them propagate rather than silently skipping every point.
    - Domain (:class:`NumevalDomainError`): a handler that raises (e.g.
      ``log`` of a negative) or cannot fold (e.g. division by zero). These
      are legitimately undefined at *some* points, so ``numeric_equiv``
      skips such a sample point.
    """


class NumevalDomainError(NumevalError):
    """A point-dependent failure: a handler raised a domain error or could
    not fold at this point (e.g. ``log`` of a negative, division by zero).

    :func:`numeric_equiv` treats a point that triggers this as undefined and
    skips it, rather than counting it as disagreement or crashing.
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
        NumevalError: structural failure -- an unbound symbol, an undefined
            operator, or a malformed term.
        NumevalDomainError: point-dependent failure -- a handler raised a
            domain error or could not fold (e.g. division by zero).
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
            raise NumevalDomainError(
                f"domain error evaluating ({op} ...): {exc}"
            ) from exc
        if value is None:
            # A handler returning None is a point-dependent "cannot fold"
            # (e.g. division by zero): a domain failure, not structural.
            raise NumevalDomainError(
                f"operator {op!r} could not fold args {args!r}"
            )
        return value

    raise NumevalError(f"not an evaluable term: {expr!r}")


def _as_sampler(spec) -> Callable[[], Dict[str, Number]]:
    """Normalize a sampler spec into a zero-arg env-producing callable.

    Accepts either:
    - a callable ``() -> {var: number}`` (returned as-is), or
    - a dict ``{var: (lo, hi)}`` of inclusive ranges, turned into a
      callable that draws each var uniformly from its range.
    """
    if callable(spec):
        return spec
    if isinstance(spec, dict):
        ranges = dict(spec)

        def sample() -> Dict[str, Number]:
            return {
                var: random.uniform(lo, hi)
                for var, (lo, hi) in ranges.items()
            }

        return sample
    raise TypeError(f"unsupported sampler spec: {spec!r}")


def _values_agree(va: Number, vb: Number, tol: float) -> bool:
    """Compare two evaluated values for agreement.

    Numbers agree when their absolute difference is within ``tol``. Booleans
    are compared by exact identity (``True`` is NOT silently matched against
    ``1.0`` by the tolerance window, and vice versa): if EITHER side is a
    bool the two must be the same bool, since ``True``/``1.0`` carry
    different meaning even though Python makes ``True == 1``.
    """
    if isinstance(va, bool) or isinstance(vb, bool):
        # Both must be bools and equal; a bool never matches a bare number.
        return type(va) is type(vb) and va == vb
    return abs(va - vb) <= tol


def numeric_equiv(
    a: ExprType,
    b: ExprType,
    sampler,
    prelude: Dict,
    *,
    samples: int = 8,
    tol: float = 1e-6,
) -> bool:
    """True iff ``a`` and ``b`` evaluate equal at every defined sample point.

    Draws ``samples`` variable assignments from ``sampler`` (a callable
    ``() -> env`` or a dict ``{var: (lo, hi)}``), evaluates both
    expressions via :func:`numeval`, and returns True iff they agree within
    ``tol`` at every sampled point where BOTH are defined AND at least one
    such point exists. Points where either expression hits a DOMAIN error
    (a handler raised, or could not fold) are SKIPPED -- legitimately
    undefined there -- not counted as disagreement and not crashed.

    Structural failures (an undefined operator, an unbound symbol the
    sampler never supplies) are NOT skipped: they propagate, surfacing a
    malformed query instead of being silently swallowed into a vacuous
    verdict. And if EVERY point is skipped (no point where both are
    defined), the result is False, not a vacuous True: an equivalence with
    no supporting evidence is not asserted.

    GENERAL: operator semantics come entirely from ``prelude``; no domain
    knowledge here.
    """
    draw = _as_sampler(sampler)
    defined = 0
    for _ in range(samples):
        env = draw()
        try:
            va = numeval(a, env, prelude)
            vb = numeval(b, env, prelude)
        except NumevalDomainError:
            # Point-dependent: at least one side is undefined HERE. Skip it.
            # A structural NumevalError (undefined op / unbound symbol) is
            # NOT caught and propagates.
            continue
        defined += 1
        if not _values_agree(va, vb, tol):
            return False
    # No defined point means no evidence of equivalence -> not equivalent.
    return defined > 0
