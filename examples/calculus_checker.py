"""Domain validators for the calculus example, built on the GENERAL numeric
primitives ``rerum.numeval``.

This file is EXAMPLE CONTENT. The engine never imports it. It encodes the
domain semantics that a derivative result must match the finite-difference of
the input, which is calculus knowledge that has no place in the engine. It is
the kind of validator a caller passes to ``rerum.training.generate_corpus``
as the ``checker`` for a differentiation corpus.

``is_derivative(expr, var, result)`` evaluates the central finite difference
of ``expr`` w.r.t. ``var`` and the claimed ``result`` at random sample points
via ``numeval`` over a numeric prelude, and accepts when they agree at every
usable point.

Error taxonomy (matches numeval's contract):

- ``NumevalDomainError`` -- an op evaluated outside its real domain at THIS
  sample point (``ln`` of a non-positive, ``sqrt`` of a negative, a pole of
  ``sec``/``csc``/``cot``, division by zero). numeval raises it both when a
  prelude function raises and when one returns ``None`` (rerum's "cannot
  fold" convention, e.g. ``safe_div`` by zero). The sampler SKIPS the point
  so a correct derivative is never rejected for an unlucky draw.
- ``NumevalError`` -- structural: an unreduced ``(dd ...)`` head, an unbound
  symbol. A real failure: ``is_derivative`` returns ``False``.

The prelude follows rerum's fold convention (each handler receives the ONE
evaluated-args list), so it is assembled from ``MATH_PRELUDE`` plus a few
additions via the same builders rule preludes use.
"""

import math
import random

from rerum import MATH_PRELUDE, combine_preludes
from rerum.expr import parse_sexpr
from rerum.numeval import NumevalDomainError, NumevalError, numeval
from rerum.rewriter import NUMERIC_TYPES, unary_only


# ------------------------------------------------------------------
# Numeric prelude for the checker (the fold functions numeval interprets).
# GENERAL primitives; the calculus knowledge is only in is_derivative below.
# Domain violations need no hand guards: math.* raises (ValueError,
# ZeroDivisionError, ...) and numeval wraps any handler exception -- or a
# handler returning None -- as NumevalDomainError, which the sampler skips.
# ------------------------------------------------------------------

def _log_any(args):
    """``(log x)`` = natural log; ``(log b x)`` = log base ``b`` of ``x``."""
    if len(args) == 1:
        return math.log(args[0])
    if len(args) == 2:
        return math.log(args[1], args[0])
    return None


NUMERIC_PRELUDE = combine_preludes(MATH_PRELUDE, {
    "ln": unary_only(math.log),
    "log": _log_any,
    "sec": unary_only(lambda x: 1.0 / math.cos(x)),
    "csc": unary_only(lambda x: 1.0 / math.sin(x)),
    "cot": unary_only(lambda x: math.cos(x) / math.sin(x)),
})


# ------------------------------------------------------------------
# The domain validator. THIS is the calculus knowledge.
# ------------------------------------------------------------------

def _free_symbols(expr, acc):
    """Collect variable symbols (non-numeric atoms that are not op heads)."""
    if isinstance(expr, str):
        acc.add(expr)
        return
    if isinstance(expr, list) and expr:
        for a in expr[1:]:
            _free_symbols(a, acc)


def _numeval(expr, env):
    """Evaluate a ground term with the checker's numeric prelude, gating the
    result to a real number. MATH_PRELUDE's ``^`` can return a COMPLEX for a
    negative base with a fractional exponent (numeval does not type-check
    results); a non-real value here is point-dependent, so it is mapped to
    the same domain-skip channel as any other out-of-domain draw."""
    value = numeval(expr, env, NUMERIC_PRELUDE)
    if isinstance(value, bool) or not isinstance(value, NUMERIC_TYPES):
        raise NumevalDomainError(f"non-real result: {value!r}")
    return float(value)


def _central_difference(expr, env, var, h):
    """Central finite difference d expr / d var at the point in ``env``."""
    env_plus = dict(env)
    env_minus = dict(env)
    env_plus[var] = env[var] + h
    env_minus[var] = env[var] - h
    return (_numeval(expr, env_plus) - _numeval(expr, env_minus)) / (2.0 * h)


def is_derivative(expr, var, result, *, samples=8, tol=1e-6) -> bool:
    """Numerically check that ``result`` is d(``expr``)/d(``var``).

    ``expr`` and ``result`` are s-expression strings (or already-parsed nested
    lists). For ``samples`` random points (a fixed-seed RNG for determinism),
    evaluate the central finite difference of ``expr`` w.r.t. ``var`` via the
    general ``numeval`` and compare to ``result`` evaluated at the same point;
    the derivative is accepted when the relative-or-absolute error is within
    ``tol`` at every usable point. Points where either side is out of domain
    (``NumevalDomainError``) are skipped; a structural failure
    (``NumevalError``: an unreduced ``(dd ...)`` head, an unbound symbol) is
    a real failure. Returns ``True`` only if at least one point was usable
    and all usable points agreed.

    Sampling range (0.25, 1.45): strictly positive (keeps ``ln``/``sqrt``/
    ``x^x`` in domain on the NOMINAL variable) and clear of pi/2 ~ 1.5708,
    the nearest ``tan``/``sec`` pole, where the finite difference is too
    ill-conditioned for ``tol`` even at points the domain skip would keep.
    """
    if isinstance(expr, str):
        expr = parse_sexpr(expr)
    if isinstance(result, str):
        result = parse_sexpr(result)

    syms = set()
    _free_symbols(expr, syms)
    _free_symbols(result, syms)
    syms.add(var)
    syms = sorted(syms)

    rng = random.Random(0xD1FF)  # deterministic
    h = 1e-5                     # finite-difference step
    checked = 0

    for _ in range(samples * 4):  # extra draws to absorb skipped points
        if checked >= samples:
            break
        env = {s: rng.uniform(0.25, 1.45) for s in syms}
        try:
            fd = _central_difference(expr, env, var, h)
            claimed = _numeval(result, env)
        except NumevalDomainError:
            continue
        except NumevalError:
            # Structural problems (an unreduced (dd ...) head, an unbound
            # symbol) are real failures, not domain skips.
            return False
        checked += 1
        denom = max(1.0, abs(fd), abs(claimed))
        if abs(fd - claimed) > tol * denom:
            return False

    return checked > 0


# ------------------------------------------------------------------
# D2 validators: antiderivatives and limits. Same foundations as
# is_derivative: the general numeval over NUMERIC_PRELUDE, with
# NumevalDomainError as the skip channel and NumevalError as the
# structural-failure channel.
# ------------------------------------------------------------------

def is_integral(integrand, var, result, *, samples=8, tol=1e-3) -> bool:
    """True iff d/d(var)[result] matches ``integrand`` at sampled points.

    Differentiates ``result`` NUMERICALLY (central finite difference via the
    general numeval) and compares to the integrand at the same points. This
    decouples integration verification from any symbolic dd rule set: a
    correct antiderivative is recognized however it was produced. Sample
    points where either side is out of domain are skipped
    (``NumevalDomainError``); a structural failure (a leftover ``int`` head,
    an unbound symbol -- ``NumevalError``) is a real failure. DOMAIN content;
    the engine never imports this.
    """
    if isinstance(integrand, str):
        integrand = parse_sexpr(integrand)
    if isinstance(result, str):
        result = parse_sexpr(result)

    h = 1e-5
    usable = 0
    for i in range(samples):
        x0 = 0.3 + 0.37 * i  # deterministic, away from 0 (1/x, ln poles)
        try:
            hi = _numeval(result, {var: x0 + h})
            lo = _numeval(result, {var: x0 - h})
            lhs = (hi - lo) / (2.0 * h)
            rhs = _numeval(integrand, {var: x0})
        except NumevalDomainError:
            continue
        except NumevalError:
            return False
        usable += 1
        if abs(lhs - rhs) > max(tol, tol * abs(rhs)):
            return False
    return usable > 0


def is_limit(expr, var, point, result, *,
             eps_seq=(0.1, 0.01, 0.001, 1e-4, 1e-6, 1e-8), tol=1e-3) -> bool:
    """True iff lim_{var->point} expr == result, by numeric two-sided approach.

    Samples ``expr`` at var = point +/- eps for shrinking eps (both sides),
    skipping samples that fail to evaluate (a 0/0 quotient AT the point is a
    domain skip; so is an out-of-domain draw like sqrt of a negative).
    Accepts iff at least one side is defined arbitrarily close to the point
    and every defined side's closest sample is within tolerance of
    ``result``. An entirely undefined side (one-sided limits at a domain
    boundary) is not a counterexample; an expression that NEVER evaluates
    anywhere yields False (any_defined stays False). The default ``eps_seq``
    reaches 1e-8 because a limit converging like sqrt(eps) (e.g.
    lim_{x->0+} sqrt(x)) only lands within ``tol`` for eps well below
    tol**2. DOMAIN content.
    """
    if isinstance(expr, str):
        expr = parse_sexpr(expr)

    target = float(result)
    any_defined = False
    sides_ok = []
    for sign in (-1.0, +1.0):
        last_err = None
        defined_here = False
        for eps in eps_seq:
            x = float(point) + sign * eps
            try:
                fval = _numeval(expr, {var: x})
            except (NumevalDomainError, NumevalError):
                continue
            defined_here = True
            any_defined = True
            last_err = abs(fval - target)
        if defined_here:
            sides_ok.append(last_err is not None
                            and last_err <= max(tol, abs(target) * tol))
        else:
            sides_ok.append(True)  # undefined side is not a counterexample
    return any_defined and all(sides_ok)
