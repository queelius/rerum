"""
Core rewriter module for symbolic expression transformation.

RERUM - Rewriting Expressions via Rules Using Morphisms

This module provides pattern matching, instantiation, and evaluation
capabilities for rule-based expression rewriting.
"""

from typing import Any, List, Union, Optional, Callable, Dict
from copy import deepcopy
from fractions import Fraction
import math

# Type aliases
ExprType = Union[int, float, str, List]
RuleType = List  # [pattern, skeleton]
NumericType = Union[int, float]


def coerce_number(x):
    """Normalize a numeric fold result to the tightest exact type.

    Rules:
    - ``bool`` passes through UNCHANGED (guarded first): Python treats
      ``True == 1`` and ``isinstance(True, int)`` as true, so a bool must
      never be narrowed to ``1``/``0``; the same bool object is returned.
    - ``int`` passes through unchanged.
    - ``float`` that is integral narrows to ``int``; otherwise stays float.
    - ``Fraction`` with denominator 1 collapses to ``int``; otherwise
      stays an exact ``Fraction`` (never silently floated).
    - any other value passes through unchanged.

    This is the single definition of int/float/Fraction narrowing; all
    fold handlers and the renarrowing in ``instantiate`` route through it.
    """
    if isinstance(x, bool):
        return x
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x) if x.is_integer() else x
    if isinstance(x, Fraction):
        return int(x) if x.denominator == 1 else x
    return x
# Internal bindings type: a Bindings object on success or None on failure.
# Public match/lookup/extend_bindings still accept the legacy list-of-pairs
# and "failed" sentinel for backward compatibility (see below).
BindingsType = Optional["Bindings"]


# ============================================================
# Bindings Class - Dict-like interface for match results
# ============================================================

class Bindings:
    """
    Dict-like wrapper for pattern matching bindings.

    Provides convenient access to bound values with a clean interface:

        if bindings := engine.match("(+ ?a ?b)", expr):
            print(bindings["a"], bindings["b"])
            print(bindings.get("c", default=0))

    Bindings objects are truthy when a match succeeded.
    Use NoMatch (which is falsy) to represent failed matches.

    Examples:
        bindings = Bindings([["x", 1], ["y", 2]])
        bindings["x"]      # => 1
        bindings.get("z")  # => None
        "x" in bindings    # => True
        len(bindings)      # => 2
        dict(bindings)     # => {"x": 1, "y": 2}
    """

    __slots__ = ('_dict',)

    def __init__(self, pairs: List[List]):
        """Initialize from list of [name, value] pairs."""
        self._dict = {name: value for name, value in pairs}

    def __bool__(self) -> bool:
        """Bindings are always truthy (use NoMatch for failed matches)."""
        return True

    def __getitem__(self, key: str):
        """Get a bound value: bindings["x"]"""
        return self._dict[key]

    def get(self, key: str, default=None):
        """Get a bound value with optional default."""
        return self._dict.get(key, default)

    def __contains__(self, key: str) -> bool:
        """Check if a variable is bound: "x" in bindings"""
        return key in self._dict

    def keys(self):
        """Return bound variable names."""
        return self._dict.keys()

    def values(self):
        """Return bound values."""
        return self._dict.values()

    def items(self):
        """Return (name, value) pairs."""
        return self._dict.items()

    def __iter__(self):
        """Iterate over bound variable names."""
        return iter(self._dict)

    def __len__(self) -> int:
        """Number of bindings."""
        return len(self._dict)

    def __repr__(self) -> str:
        return f"Bindings({self._dict})"

    def __eq__(self, other):
        if isinstance(other, Bindings):
            return self._dict == other._dict
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dictionary."""
        return self._dict.copy()

    def extend(self, name: str, value: Any) -> Optional["Bindings"]:
        """Return a new Bindings with ``name`` bound to ``value``.

        - If ``name`` is unbound, returns a new Bindings with the binding added.
        - If ``name`` is already bound to ``value`` (consistent), returns ``self``.
        - If ``name`` is already bound to something else (conflict), returns ``None``.

        Functional / non-mutating: ``self`` is never modified.
        """
        if name in self._dict:
            return self if self._dict[name] == value else None
        new = Bindings.__new__(Bindings)
        new._dict = {**self._dict, name: value}
        return new

    def lookup(self, name: str, default=None) -> Any:
        """Return the bound value for ``name``, or ``default`` if unbound.

        When ``default`` is None (the default), returns ``name`` itself,
        matching the legacy ``lookup`` semantics where unbound variables
        pass through as symbols during instantiation.
        """
        if default is None:
            return self._dict.get(name, name)
        return self._dict.get(name, default)

    @classmethod
    def empty(cls) -> "Bindings":
        """Return an empty Bindings (no variables bound)."""
        new = cls.__new__(cls)
        new._dict = {}
        return new


class _NoMatch:
    """
    Singleton representing a failed pattern match.

    NoMatch is falsy, allowing natural use in conditionals:

        if bindings := engine.match(pattern, expr):
            # matched
        else:
            # NoMatch
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __bool__(self) -> bool:
        """NoMatch is falsy."""
        return False

    def __repr__(self) -> str:
        return "NoMatch"

    def __getitem__(self, key: str):
        """Raise KeyError - no bindings exist."""
        raise KeyError(f"NoMatch has no binding for '{key}'")

    def get(self, key: str, default=None):
        """Always return default - no bindings exist."""
        return default

    def __contains__(self, key: str) -> bool:
        """Nothing is bound in NoMatch."""
        return False

    def __len__(self) -> int:
        """NoMatch has no bindings."""
        return 0

    def __iter__(self):
        """Empty iterator."""
        return iter([])


# Singleton instance
NoMatch = _NoMatch()


def wrap_bindings(result) -> Union[Bindings, _NoMatch]:
    """Convert any bindings form to ``Bindings`` or ``NoMatch``.

    Accepts:
      - ``Bindings`` (returned unchanged)
      - ``None`` (new internal sentinel for failure)
      - List of [name, value] pairs (legacy success form)
      - The string ``"failed"`` (legacy failure sentinel)

    .. deprecated::
        Use ``bindings is not None`` (or just truthiness) directly. ``Bindings``
        is truthy and ``None`` is falsy, so the wrapper is no longer needed
        for control flow. Kept only for callers that need a concrete
        ``NoMatch`` object (e.g. existing public APIs returning a
        ``Bindings | NoMatch`` union).
    """
    if result is None or result == "failed":
        return NoMatch
    if isinstance(result, Bindings):
        return result
    return Bindings(result)

# FoldOp handler: receives list of numeric args, returns result or None (can't fold)
FoldHandler = Callable[[List[NumericType]], Optional[NumericType]]
FoldFuncsType = Dict[str, FoldHandler]


# ============================================================
# Fold Operation Builders
# ============================================================

def nary_fold(
    identity: NumericType,
    binary_op: Callable[[NumericType, NumericType], NumericType],
    unary: Optional[Callable[[NumericType], NumericType]] = None,
) -> FoldHandler:
    """Create an n-ary folder with identity element.

    The folded result is normalized via ``coerce_number`` so exact
    ``Fraction`` operands stay exact (and collapse to ``int`` when whole).

    Args:
        identity: Value for 0-arity, e.g., 0 for +, 1 for *
        binary_op: Binary operation for folding
        unary: Optional special unary behavior (defaults to identity)

    Examples:
        nary_fold(0, lambda a, b: a + b)  # (+) = 0, (+ x) = x, (+ x y z) = x+y+z
        nary_fold(1, lambda a, b: a * b)  # (*) = 1, (* x) = x, (* x y z) = x*y*z
    """
    def handler(args: List[NumericType]) -> NumericType:
        if len(args) == 0:
            return identity
        if len(args) == 1:
            return coerce_number(unary(args[0]) if unary else args[0])
        result = args[0]
        for a in args[1:]:
            result = binary_op(result, a)
        return coerce_number(result)
    return handler


def unary_only(f: Callable[[NumericType], NumericType]) -> FoldHandler:
    """Create a unary-only folder (e.g., sin, cos, exp)."""
    def handler(args: List[NumericType]) -> Optional[NumericType]:
        if len(args) != 1:
            return None  # Can't fold non-unary
        return f(args[0])
    return handler


def binary_only(f: Callable[[NumericType, NumericType], NumericType]) -> FoldHandler:
    """Create a binary-only folder (e.g., /, ^, atan2)."""
    def handler(args: List[NumericType]) -> Optional[NumericType]:
        if len(args) != 2:
            return None  # Can't fold non-binary
        return f(args[0], args[1])
    return handler


def special_minus() -> FoldHandler:
    """Special handler for subtraction: (-) = 0, (- x) = -x, (- x y) = x-y."""
    def handler(args: List[NumericType]) -> Optional[NumericType]:
        if len(args) == 0:
            return 0
        if len(args) == 1:
            return -args[0]
        if len(args) == 2:
            return args[0] - args[1]
        return None  # No n-ary subtraction beyond 2
    return handler


def safe_div() -> FoldHandler:
    """Safe division handler that returns None on division by zero.

    For integer operands the quotient is computed exactly via ``Fraction``
    (so ``1/3`` stays ``Fraction(1, 3)`` rather than a lossy float), then
    narrowed by ``coerce_number`` (a whole quotient collapses to ``int``).
    Float operands divide as floats (then narrow integral results to int).
    """
    def handler(args: List[NumericType]) -> Optional[NumericType]:
        if len(args) != 2:
            return None
        a, b = args
        if b == 0:
            return None  # Can't fold division by zero
        if isinstance(a, int) and isinstance(b, int):
            return coerce_number(Fraction(a, b))
        if isinstance(a, Fraction) or isinstance(b, Fraction):
            return coerce_number(Fraction(a) / Fraction(b))
        return coerce_number(a / b)
    return handler


# ============================================================
# Standard Preludes for Constant Folding
# ============================================================

# Arithmetic prelude: basic arithmetic operators
ARITHMETIC_PRELUDE: FoldFuncsType = {
    "+": nary_fold(0, lambda a, b: a + b),
    "*": nary_fold(1, lambda a, b: a * b),
    "-": special_minus(),
    "/": safe_div(),
    "^": binary_only(lambda a, b: a ** b),
}

# Math prelude: standard mathematical functions (unary)
MATH_PRELUDE: FoldFuncsType = {
    **ARITHMETIC_PRELUDE,
    "sin": unary_only(math.sin),
    "cos": unary_only(math.cos),
    "tan": unary_only(math.tan),
    "asin": unary_only(math.asin),
    "acos": unary_only(math.acos),
    "atan": unary_only(math.atan),
    "atan2": binary_only(math.atan2),
    "sinh": unary_only(math.sinh),
    "cosh": unary_only(math.cosh),
    "tanh": unary_only(math.tanh),
    "exp": unary_only(math.exp),
    "log": unary_only(math.log),
    "log10": unary_only(math.log10),
    "log2": unary_only(math.log2),
    "sqrt": unary_only(math.sqrt),
    "abs": unary_only(abs),
    "floor": unary_only(math.floor),
    "ceil": unary_only(math.ceil),
    "round": unary_only(round),
}

# Minimal prelude: just arithmetic + essentials
MINIMAL_PRELUDE: FoldFuncsType = {
    **ARITHMETIC_PRELUDE,
    "sin": unary_only(math.sin),
    "cos": unary_only(math.cos),
    "exp": unary_only(math.exp),
    "log": unary_only(math.log),
    "sqrt": unary_only(math.sqrt),
    "abs": unary_only(abs),
}

# Predicate prelude: comparison and type predicates for conditional guards
PREDICATE_PRELUDE: FoldFuncsType = {
    # Comparison operators
    ">": binary_only(lambda a, b: a > b),
    "<": binary_only(lambda a, b: a < b),
    ">=": binary_only(lambda a, b: a >= b),
    "<=": binary_only(lambda a, b: a <= b),
    "=": binary_only(lambda a, b: a == b),
    "!=": binary_only(lambda a, b: a != b),
    # Type predicates
    "const?": unary_only(lambda x: isinstance(x, (int, float))),
    "var?": unary_only(lambda x: isinstance(x, str)),
    "list?": unary_only(lambda x: isinstance(x, list)),
    "atom?": unary_only(lambda x: not isinstance(x, list)),
    "zero?": unary_only(lambda x: x == 0),
    "positive?": unary_only(lambda x: isinstance(x, (int, float)) and x > 0),
    "negative?": unary_only(lambda x: isinstance(x, (int, float)) and x < 0),
    # Structural predicates
    "free-of?": binary_only(lambda f, v: isinstance(v, str) and not free_in(v, f)),
    # Logical operators
    "not": unary_only(lambda x: not x),
    "and": binary_only(lambda a, b: a and b),
    "or": binary_only(lambda a, b: a or b),
}

# Full prelude: arithmetic + predicates (common choice for conditional rules)
FULL_PRELUDE: FoldFuncsType = {
    **ARITHMETIC_PRELUDE,
    **PREDICATE_PRELUDE,
}

# Empty prelude (no constant folding at all)
NO_PRELUDE: FoldFuncsType = {}


def combine_preludes(*preludes: FoldFuncsType) -> FoldFuncsType:
    """Merge fold-function dicts left-to-right into a fresh dict.

    This is the general way a rule set composes the preludes it needs:
    ``combine_preludes(MATH_PRELUDE, PREDICATE_PRELUDE)`` yields a prelude
    with both math functions and predicates. Later preludes win on key
    conflict. The result is a new dict; inputs are not mutated.

    The engine ships no domain-named bundle. A rule set documents the
    combination it requires and assembles it via this helper as data.
    """
    merged: FoldFuncsType = {}
    for prelude in preludes:
        merged.update(prelude)
    return merged


# ============================================================
# Primitive Operations (Lisp-like list operations)
# ============================================================

def car(lst: List) -> Any:
    """
    Return the first element of a list (head).

    Args:
        lst: A non-empty list

    Returns:
        The first element of the list

    Raises:
        TypeError: If argument is not a list
        ValueError: If list is empty
    """
    if not isinstance(lst, list):
        raise TypeError("car: argument must be a list")
    if not lst:
        raise ValueError("car: argument is an empty list")
    return lst[0]


def cdr(lst: List) -> List:
    """
    Return all but the first element of a list (tail).

    Args:
        lst: A list

    Returns:
        A list containing all elements except the first
    """
    if not isinstance(lst, list):
        raise TypeError("cdr: argument must be a list")
    return lst[1:] if lst else []


def cons(item: Any, lst: List) -> List:
    """
    Construct a new list by prepending an item.

    Args:
        item: The item to prepend
        lst: The list to prepend to

    Returns:
        A new list with item as the first element
    """
    return [item] + lst


def atom(exp: ExprType) -> bool:
    """
    Check if an expression is atomic (constant or variable).

    Args:
        exp: The expression to check

    Returns:
        True if exp is a constant or variable, False otherwise
    """
    return constant(exp) or variable(exp)


def _expr_key(exp: ExprType):
    """Convert an expression to a hashable key for cycle detection."""
    if isinstance(exp, list):
        return tuple(_expr_key(e) for e in exp)
    return exp


def compound(exp: ExprType) -> bool:
    """
    Check if an expression is compound (a list).

    Args:
        exp: The expression to check

    Returns:
        True if exp is a list, False otherwise
    """
    return isinstance(exp, list)


def constant(exp: ExprType) -> bool:
    """
    Check if an expression is a numeric constant.

    Args:
        exp: The expression to check

    Returns:
        True if exp is an int or float, False otherwise
    """
    return isinstance(exp, (int, float))


def variable(exp: ExprType) -> bool:
    """
    Check if an expression is a variable (string).

    Args:
        exp: The expression to check

    Returns:
        True if exp is a string, False otherwise
    """
    return isinstance(exp, str)


def null(s: Any) -> bool:
    """
    Check if an expression is null (empty list).

    Args:
        s: The expression to check

    Returns:
        True if s is an empty list, False otherwise
    """
    return s == []


# ============================================================
# Pattern Matching Helpers
# ============================================================

def arbitrary_constant(pat: ExprType) -> bool:
    """Check if pattern matches any constant (?c)."""
    return compound(pat) and car(pat) == "?c"


def arbitrary_variable(pat: ExprType) -> bool:
    """Check if pattern matches any variable (?v)."""
    return compound(pat) and car(pat) == "?v"


def arbitrary_expression(pat: ExprType) -> bool:
    """Check if pattern matches any expression (?)."""
    return compound(pat) and car(pat) == "?"


def arbitrary_free(pat: ExprType) -> bool:
    """Check if pattern matches expression free of a variable (?free)."""
    return compound(pat) and len(pat) == 3 and car(pat) == "?free"


def arbitrary_rest(pat: ExprType) -> bool:
    """Check if pattern matches remaining expressions (?...).

    Forms:
        ["?...", "name"]           - match remaining args (any type)
        ["?...", "name", "const"]  - match remaining args (each must be constant)
        ["?...", "name", "var"]    - match remaining args (each must be variable)
    """
    return compound(pat) and len(pat) >= 2 and car(pat) == "?..."


def rest_type_constraint(pat: List) -> Optional[str]:
    """Get the type constraint for a rest pattern, if any.

    Returns: "const", "var", or None for unconstrained
    """
    if len(pat) >= 3:
        return pat[2]
    return None


def skeleton_splice(s: ExprType) -> bool:
    """Check if skeleton element should be spliced (:...).

    Form: [":...", "name"] - splice list into parent
    """
    return compound(s) and len(s) == 2 and car(s) == ":..."


def skeleton_compute(s: ExprType) -> bool:
    """Check if skeleton element should be computed (!).

    Form: ["!", "op", arg1, arg2, ...] - evaluate op on args
    """
    return compound(s) and len(s) >= 2 and car(s) == "!"


def skeleton_fresh(s: ExprType) -> bool:
    """Check if skeleton element is a fresh-variable form (fresh base).

    Form: ["fresh", "base"] - resolve to a gensym not free in the result.
    """
    return compound(s) and len(s) == 2 and car(s) == "fresh"


def free_in(var: str, expr: ExprType) -> bool:
    """
    Check if a variable appears free in an expression.

    Args:
        var: Variable name to check for
        expr: Expression to search in

    Returns:
        True if var appears in expr, False otherwise
    """
    if isinstance(expr, (int, float)):
        return False
    if isinstance(expr, str):
        return expr == var
    if isinstance(expr, list) and expr:
        return any(free_in(var, sub) for sub in expr)
    return False


def free_symbols(expr: ExprType) -> set:
    """Return the set of all symbol (string) leaves occurring in ``expr``.

    Includes operator-head symbols; for fresh-variable avoidance the
    conservative superset of "names already present" is exactly what we
    want, so we do not distinguish head from operand position.
    """
    if isinstance(expr, str):
        return {expr}
    if isinstance(expr, list):
        out: set = set()
        for sub in expr:
            out |= free_symbols(sub)
        return out
    return set()


def gensym(base: str, avoid: set) -> str:
    """Smallest of ``base``, ``base+"1"``, ``base+"2"``, ... not in ``avoid``.

    Deterministic: a pure function of ``base`` and ``avoid``.
    """
    if base not in avoid:
        return base
    i = 1
    while f"{base}{i}" in avoid:
        i += 1
    return f"{base}{i}"


def skeleton_evaluation(s: ExprType) -> bool:
    """Check if skeleton element should be evaluated (:)."""
    return compound(s) and car(s) == ":"


def variable_name(pat: List) -> str:
    """Extract the variable name from a pattern element.

    For ["?", "name"], ["?c", "name"], ["?v", "name"]: returns "name"
    For ["?free", "name", "var"]: returns "name"
    """
    return pat[1]  # Second element is always the binding name


def _coerce_bindings(bindings) -> Optional[Bindings]:
    """Accept legacy or new bindings form; return ``Bindings`` or ``None``.

    Legacy callers pass:
      - ``[]`` (initial, empty bindings) or ``[[name, value], ...]``
      - ``"failed"`` (legacy failure sentinel)
    New callers pass:
      - ``Bindings`` instance
      - ``None`` (failure)

    This shim lets the public API accept either form during the transition
    away from the stringly-typed sentinel. Internal helpers should produce
    only ``Bindings | None``.
    """
    if bindings is None or bindings == "failed":
        return None
    if isinstance(bindings, Bindings):
        return bindings
    if isinstance(bindings, list):
        return Bindings(bindings)
    return bindings  # unknown type; let callers fail loudly


def extend_bindings(pat: List, dat: ExprType, bindings) -> Optional[Bindings]:
    """Extend ``bindings`` with the binding implied by ``pat`` and ``dat``.

    Returns a new ``Bindings`` with the binding added, ``bindings`` itself
    if the binding was already consistent, or ``None`` on conflict / when
    starting from a failed match.

    Backward-compat: accepts legacy list-of-pairs and ``"failed"`` for the
    bindings argument.
    """
    coerced = _coerce_bindings(bindings)
    if coerced is None:
        return None
    name = variable_name(pat)
    return coerced.extend(name, dat)


def lookup(var: str, bindings) -> Any:
    """Look up ``var`` in ``bindings``; returns ``var`` itself if unbound.

    Backward-compat: accepts legacy list-of-pairs and ``"failed"`` for the
    bindings argument.
    """
    coerced = _coerce_bindings(bindings)
    if coerced is None:
        return var
    return coerced.lookup(var)


# ============================================================
# Pattern Matching
# ============================================================

def _match_recursive(pat: ExprType, exp: ExprType,
                     bindings: Optional["Bindings"] = None) -> Optional["Bindings"]:
    """Recursive core for structural pattern matching.

    This is the internal recursion entry point. It does NOT run the
    deferred ``?free`` validation; that post-pass is applied once by the
    public ``match`` wrapper after all bindings are fully resolved.

    Pattern syntax:
        ["?", "name"]            - match any expression
        ["?c", "name"]           - match constants only
        ["?v", "name"]           - match variables only
        ["?free", "name", "var"] - match expression not containing var
                                   (constraint deferred; checked by public match)
        ["?...", "name"]         - match remaining args (zero or more)
        ["?...", "name", "const"] - match remaining args (each must be constant)
        literal                  - match exact value

    Args:
        pat: The pattern to match
        exp: The expression to match against
        bindings: Current bindings (``Bindings`` or ``None``). The legacy
            list-of-pairs and ``"failed"`` forms are also accepted. When
            ``None`` (default), starts from an empty Bindings.

    Returns:
        ``Bindings`` on success, ``None`` on failure.
    """
    if bindings is None:
        bindings = Bindings.empty()
    elif bindings == "failed":
        return None
    elif isinstance(bindings, list):
        bindings = Bindings(bindings)
    elif not isinstance(bindings, Bindings):
        return None

    if null(pat):
        return bindings if null(exp) else None

    if atom(pat):
        return bindings if atom(exp) and pat == exp else None

    if arbitrary_constant(pat):
        return bindings.extend(variable_name(pat), exp) if constant(exp) else None

    if arbitrary_variable(pat):
        return bindings.extend(variable_name(pat), exp) if variable(exp) else None

    if arbitrary_expression(pat):
        return bindings.extend(variable_name(pat), exp)

    if arbitrary_free(pat):
        # ["?free", "name", "var"] - optimistically bind; the public match
        # wrapper re-validates this constraint against the final bindings.
        var_to_exclude = pat[2]
        actual_var = bindings.lookup(var_to_exclude)
        # If the excluded variable is already bound to a SYMBOL, we can
        # fail fast when the constraint is already violated. Otherwise (the
        # excluded var is still unbound, or is bound to a non-symbol that no
        # term can "contain"), bind optimistically and let the final-bindings
        # post-pass (_check_free_constraints) decide. This keeps the early
        # exit consistent with the post-pass semantics.
        if isinstance(actual_var, str) and actual_var != var_to_exclude:
            if free_in(actual_var, exp):
                return None
        return bindings.extend(variable_name(pat), exp)

    if arbitrary_rest(pat):
        # Rest patterns are handled in compound matching; when called
        # directly with a list expression, bind the whole list.
        if isinstance(exp, list):
            return bindings.extend(variable_name(pat), exp)
        return None

    if atom(exp):
        return None

    if null(pat) or null(exp):
        return None

    return match_compound(pat, exp, bindings)


def _check_free_constraints(pat: ExprType, exp: ExprType,
                            bindings: "Bindings") -> bool:
    """Re-validate every ``?free`` constraint in ``pat`` against ``exp``
    using the FINAL ``bindings``.

    Returns True if all ``?free`` constraints hold, False otherwise. A
    ``?free`` node ``["?free", name, var]`` requires that the resolved
    value of ``var`` does not occur in the subexpression that ``name``
    matched (which, structurally, is the aligned ``exp`` position).
    """
    if arbitrary_free(pat):
        excluded = bindings.lookup(pat[2])
        if isinstance(excluded, str) and free_in(excluded, exp):
            return False
        return True
    if not isinstance(pat, list) or not pat:
        return True
    head = pat[0]
    if head in ("?", "?c", "?v"):
        return True
    if not isinstance(exp, list):
        return True
    i = 0
    for sub_pat in pat:
        if arbitrary_rest(sub_pat):
            break
        if i >= len(exp):
            break
        if not _check_free_constraints(sub_pat, exp[i], bindings):
            return False
        i += 1
    return True


def match(pat: ExprType, exp: ExprType,
          bindings: Optional["Bindings"] = None) -> Optional["Bindings"]:
    """Structural pattern match (public entry point).

    Delegates to the recursive core, then validates every ``?free``
    constraint against the FINAL resolved bindings. This fixes the
    binding-order bug where a ``?free`` pattern appearing to the LEFT of
    the binding of its excluded variable passed vacuously (the excluded
    var was still unbound when the free check ran). See ``?x:free(v)``.

    Returns ``Bindings`` on success, ``None`` on failure.
    """
    result = _match_recursive(pat, exp, bindings)
    if result is None:
        return None
    if not _check_free_constraints(pat, exp, result):
        return None
    return result


def match_compound(pat: List, exp: List,
                   bindings: Optional["Bindings"]) -> Optional["Bindings"]:
    """Match compound patterns against compound expressions.

    Handles rest patterns (?...) which must appear at the end.
    """
    if bindings is None:
        return None

    if null(pat) and null(exp):
        return bindings

    if null(pat):
        return None

    current_pat = car(pat)
    rest_pat = cdr(pat)

    if arbitrary_rest(current_pat):
        if not null(rest_pat):
            raise ValueError("Rest pattern (?...) must be last in compound pattern")

        remaining = exp if isinstance(exp, list) else [exp] if not null(exp) else []

        type_constraint = rest_type_constraint(current_pat)
        if type_constraint:
            for item in remaining:
                if type_constraint == "const" and not constant(item):
                    return None
                elif type_constraint == "var" and not variable(item):
                    return None

        return bindings.extend(variable_name(current_pat), remaining)

    if null(exp):
        return None

    submatch = _match_recursive(current_pat, car(exp), bindings)
    return match_compound(rest_pat, cdr(exp), submatch)


# ============================================================
# Instantiation
# ============================================================

def instantiate(
    skeleton: ExprType,
    bindings,
    fold_funcs: Optional[FoldFuncsType] = None,
    undefined_op_resolver: Optional[Callable] = None,
    fold_error_resolver: Optional[Callable] = None,
    *,
    _resolve_fresh_markers: bool = True,
) -> ExprType:
    """
    Instantiate a skeleton with bindings.

    Skeleton syntax:
        [":", "name"]           - substitute with bound value
        [":...", "name"]        - splice bound list into parent
        ["!", "op", args...]    - compute op(args) immediately
        ["fresh", "base"]       - a name not free in the expression being built
        literal                 - keep as-is

    ``__fresh__`` is a reserved internal sentinel: the ``["fresh", base]``
    form is emitted as a deferred ``["__fresh__", base]`` marker and resolved
    in one post-pass, so a literal ``["__fresh__", base]`` in user data is
    treated as such a marker (the same way ``"!"``/``":"``/``"fresh"`` are
    reserved skeleton words).

    Args:
        skeleton: The skeleton to instantiate
        bindings: ``Bindings`` instance (legacy list-of-pairs also accepted)
        fold_funcs: Optional fold functions for compute (!) evaluation
        undefined_op_resolver: Optional callable ``(op, args) -> Resolution|None``
            called when ``op`` is not in ``fold_funcs``. A
            ``Resolution(value=v)`` returns ``v`` directly; a
            ``Resolution(fold_funcs={op: handler})`` installs the handler into
            ``fold_funcs`` in place and retries the fold; ``None`` falls
            through to the existing "leave as compound" behavior.
        fold_error_resolver: Optional callable ``(op, args, exc) -> Resolution|None``
            called when an installed fold handler raises an exception. A
            ``Resolution(value=v)`` returns ``v`` as the fallback; ``None``
            falls through to the existing "leave as compound" behavior.
            ``Resolution(value=None)`` is impossible to construct
            (``Resolution.__post_init__`` rejects it as "empty"); fold
            handlers cannot legitimately yield Python None as a result. If
            such a value is needed, return ``Resolution(value=...)`` where
            ``...`` is a sentinel object the caller can interpret.
        _resolve_fresh_markers: Internal flag. When True (the default,
            top-level entry), ``["fresh", base]`` forms are resolved to
            deterministic gensyms after the whole expression is built.
            Recursive/internal calls pass False so the markers survive to
            be resolved once, together, by the outermost call.

    Returns:
        The instantiated expression
    """
    coerced = _coerce_bindings(bindings) or Bindings.empty()

    def loop(s):
        if null(s):
            return []
        if atom(s):
            return s
        if skeleton_evaluation(s):
            return coerced.lookup(car(cdr(s)))
        if skeleton_splice(s):
            return coerced.lookup(car(cdr(s)))
        if skeleton_fresh(s):
            # Defer resolution: emit a unique unresolved marker; the
            # post-pass picks the smallest gensym not already used.
            return ["__fresh__", car(cdr(s))]
        if skeleton_compute(s):
            op = s[1]
            raw_args = s[2:]
            args = [loop(arg) for arg in raw_args]
            handler = fold_funcs.get(op) if fold_funcs else None
            if handler is None and undefined_op_resolver is not None:
                resolution = undefined_op_resolver(op, args)
                if resolution is not None:
                    if resolution.value is not None:
                        return resolution.value
                    if resolution.fold_funcs is not None:
                        # Install into fold_funcs in-place when a dict is
                        # available; the engine passes its own _fold_funcs
                        # reference, so a live dict means permanent install.
                        # When fold_funcs is None (no prelude configured),
                        # the engine bridge is responsible for the permanent
                        # side-effect; here we just grab the handler for this
                        # single call.
                        if fold_funcs is not None:
                            fold_funcs.update(resolution.fold_funcs)
                        handler = resolution.fold_funcs.get(op)
            if handler is not None:
                try:
                    result = handler(args)
                    if result is not None:
                        # Narrow to the tightest exact numeric type
                        # (int/Fraction/float) via the single chokepoint;
                        # keeps (! / 1 3) -> Fraction(1, 3) exact.
                        return coerce_number(result)
                except Exception as exc:
                    if fold_error_resolver is not None:
                        resolution = fold_error_resolver(op, args, exc)
                        if resolution is not None and resolution.value is not None:
                            return resolution.value
                    # Fall through to compound emission.
            return [op] + args
        return instantiate_compound(s, coerced, fold_funcs, undefined_op_resolver,
                                    fold_error_resolver)

    built = loop(skeleton)
    # Resolve ["fresh", base] forms exactly once, at the top-level call,
    # so every marker in the fully-built expression is resolved together
    # against one shared avoid-set (recursive/internal calls pass
    # ``_resolve_fresh_markers=False`` and leave markers intact).
    if _resolve_fresh_markers:
        return _resolve_fresh(built)
    return built


def instantiate_compound(
    skeleton: List,
    bindings,
    fold_funcs: Optional[FoldFuncsType] = None,
    undefined_op_resolver: Optional[Callable] = None,
    fold_error_resolver: Optional[Callable] = None,
) -> List:
    """Instantiate a compound skeleton, handling splice patterns.

    When a splice pattern [":...", "name"] is encountered, its bound list
    is spliced into the result rather than inserted as a single element.
    """
    coerced = _coerce_bindings(bindings) or Bindings.empty()
    if null(skeleton):
        return []

    first = car(skeleton)
    rest = cdr(skeleton)

    if skeleton_splice(first):
        name = car(cdr(first))
        spliced = coerced.lookup(name)
        rest_instantiated = instantiate_compound(rest, coerced, fold_funcs,
                                                 undefined_op_resolver, fold_error_resolver)
        if isinstance(spliced, list):
            return spliced + rest_instantiated
        return [spliced] + rest_instantiated

    first_instantiated = instantiate(first, coerced, fold_funcs, undefined_op_resolver,
                                     fold_error_resolver, _resolve_fresh_markers=False)
    rest_instantiated = instantiate_compound(rest, coerced, fold_funcs, undefined_op_resolver,
                                             fold_error_resolver)
    return [first_instantiated] + rest_instantiated


def _resolve_fresh(expr: ExprType) -> ExprType:
    """Replace ["__fresh__", base] markers with deterministic gensyms.

    Resolution is left-to-right (pre-order). Each resolved name is added
    to the avoid-set so two fresh forms with the same base get distinct
    names, and every already-present symbol in the expression is avoided.
    Pure and deterministic: a fixed input yields a fixed output.
    """
    # Names already present (excluding the markers themselves).
    def present(e: ExprType) -> set:
        if isinstance(e, str):
            return {e}
        if isinstance(e, list):
            if len(e) == 2 and e[0] == "__fresh__":
                return set()
            out: set = set()
            for sub in e:
                out |= present(sub)
            return out
        return set()

    used = present(expr)

    def walk(e: ExprType) -> ExprType:
        if isinstance(e, list):
            if len(e) == 2 and e[0] == "__fresh__":
                name = gensym(e[1], used)
                used.add(name)
                return name
            return [walk(sub) for sub in e]
        return e

    return walk(expr)


# ============================================================
# Rewriter Factory
# ============================================================

def rewriter(
    rules: List[RuleType],
    fold_funcs: Optional[FoldFuncsType] = None,
    numerical_fallback: Optional[Callable[[ExprType], Optional[ExprType]]] = None,
) -> Callable[[ExprType], ExprType]:
    """
    Create a rewriter function using given rules.

    By default, this is a pure rule rewriter with no built-in evaluation.
    To enable constant folding, pass a prelude via fold_funcs.

    Args:
        rules: List of [pattern, skeleton] rules
        fold_funcs: Optional fold functions dict for constant folding.
            Pass ARITHMETIC_PRELUDE for basic +, -, *, /, ^.
            Pass MATH_PRELUDE for arithmetic + trig/exp/log.
            Pass your own dict mapping operators to FoldHandler functions.
            Default: None (no constant folding - pure rule rewriting)
        numerical_fallback: Optional callback for stuck expressions.
            Signature: fallback(expr) -> ExprType or None

    Returns:
        A function that rewrites expressions to simplified form

    Examples:
        # Pure rule rewriting (no constant folding)
        simplify = rewriter(rules)

        # With basic arithmetic folding
        simplify = rewriter(rules, fold_funcs=ARITHMETIC_PRELUDE)

        # With math functions (sin, cos, exp, log, etc.)
        simplify = rewriter(rules, fold_funcs=MATH_PRELUDE)

        # Custom functions
        my_funcs = {**ARITHMETIC_PRELUDE, "sigmoid": unary_only(lambda x: 1/(1+math.exp(-x)))}
        simplify = rewriter(rules, fold_funcs=my_funcs)
    """
    # fold_funcs controls constant folding - None means no folding
    active_fold_funcs: FoldFuncsType = fold_funcs if fold_funcs is not None else {}

    def simplify(exp: ExprType) -> ExprType:
        """Simplify an expression using the rules.

        Uses a visited set to detect cycles. Non-confluent rule sets (such as
        rules with `<=>` semantics, or pairs like `a => b` plus `b => a`) cycle
        back to a previously seen state; the loop terminates at the first
        repeat and returns the current expression deterministically.
        """
        visited = set()
        max_iterations = 1000
        iterations = 0

        while iterations < max_iterations:
            iterations += 1
            key = _expr_key(exp)
            if key in visited:
                break
            visited.add(key)
            old_exp = deepcopy(exp)

            # Try applying rules
            result = try_rules(exp)
            if result != exp:
                exp = result
                continue

            # Try constant folding (only if fold_funcs provided)
            if active_fold_funcs:
                result = try_constant_fold(exp)
                if result != exp:
                    exp = result
                    continue

            # Try numerical fallback for stuck expressions
            if numerical_fallback and compound(exp):
                result = numerical_fallback(exp)
                if result is not None and result != exp:
                    exp = result
                    continue

            # Recursively simplify compound parts
            if compound(exp):
                result = simplify_parts(exp)
                if result != exp:
                    exp = result
                    continue

            # No changes, we're done
            if exp == old_exp:
                break

        return exp

    def simplify_parts(exp: ExprType) -> ExprType:
        """Recursively simplify parts of a compound expression."""
        if null(exp):
            return []
        return cons(simplify(car(exp)), simplify_parts(cdr(exp)))

    def try_constant_fold(exp: ExprType) -> ExprType:
        """Try to evaluate operations on constant operands using active_fold_funcs."""
        if not compound(exp) or null(exp):
            return exp

        op = car(exp)
        args = cdr(exp)

        # Check if we have a fold function for this operator
        if op not in active_fold_funcs:
            return exp

        # Check if all arguments are numeric constants (int/float/Fraction).
        if not all(isinstance(arg, (int, float, Fraction)) for arg in args):
            return exp

        try:
            handler = active_fold_funcs[op]
            result = handler(args)

            # None means can't fold (wrong arity, etc.)
            if result is None:
                return exp

            # Narrow to the tightest exact numeric type (int/Fraction/float)
            # via the single chokepoint, keeping rationals exact.
            return coerce_number(result)
        except Exception:
            return exp

    def try_rules(exp: ExprType) -> ExprType:
        """Apply the first matching rule to an expression.

        Returns the rewritten expression after a single rule application,
        or the input unchanged if no rule matches. The outer fixpoint loop
        in `simplify` drives convergence and detects cycles.
        """
        for rule in rules:
            pat = rule[0]
            skel = rule[1]

            bindings = match(pat, exp)
            if bindings is not None:
                return instantiate(skel, bindings, active_fold_funcs)

        return exp

    return simplify


# Convenience alias
simplifier = rewriter
