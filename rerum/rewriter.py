"""
Core rewriter module for symbolic expression transformation.

RERUM - Rewriting Expressions via Rules Using Morphisms

This module provides pattern matching, instantiation, and evaluation
capabilities for rule-based expression rewriting.
"""

from typing import Any, List, Union, Optional, Callable, Dict
from copy import deepcopy
import math

# Type aliases
ExprType = Union[int, float, str, List]
BindingsType = Union[List[List], str]  # List of [name, value] pairs or "failed"
RuleType = List  # [pattern, skeleton]
NumericType = Union[int, float]


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


def wrap_bindings(result: BindingsType) -> Union[Bindings, _NoMatch]:
    """
    Convert internal bindings representation to Bindings or NoMatch.

    Args:
        result: Either list of [name, value] pairs or "failed"

    Returns:
        Bindings object if matched, NoMatch if failed
    """
    if result == "failed":
        return NoMatch
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
            return unary(args[0]) if unary else args[0]
        result = args[0]
        for a in args[1:]:
            result = binary_op(result, a)
        return result
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
    """Safe division handler that returns None on division by zero."""
    def handler(args: List[NumericType]) -> Optional[NumericType]:
        if len(args) != 2:
            return None
        if args[1] == 0:
            return None  # Can't fold division by zero
        return args[0] / args[1]
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


def skeleton_evaluation(s: ExprType) -> bool:
    """Check if skeleton element should be evaluated (:)."""
    return compound(s) and car(s) == ":"


def variable_name(pat: List) -> str:
    """Extract the variable name from a pattern element.

    For ["?", "name"], ["?c", "name"], ["?v", "name"]: returns "name"
    For ["?free", "name", "var"]: returns "name"
    """
    return pat[1]  # Second element is always the binding name


def extend_bindings(
    pat: List, dat: ExprType, bindings: BindingsType
) -> BindingsType:
    """
    Extend a bindings dictionary with a new binding.

    Args:
        pat: The pattern containing the variable name
        dat: The data to bind
        bindings: The current bindings dictionary

    Returns:
        Extended bindings or "failed" on conflict
    """
    if bindings == "failed":
        return "failed"

    name = variable_name(pat)
    for entry in bindings:
        if entry[0] == name:
            # Check for consistent binding
            if entry[1] == dat:
                return bindings
            else:
                return "failed"

    return bindings + [[name, dat]]


def lookup(var: str, bindings: BindingsType) -> Any:
    """
    Look up a variable in the bindings dictionary.

    Args:
        var: The variable name
        bindings: The bindings dictionary

    Returns:
        The bound value or var if not found
    """
    if bindings == "failed":
        return var
    for entry in bindings:
        if entry[0] == var:
            return entry[1]
    return var


# ============================================================
# Pattern Matching
# ============================================================

def match(pat: ExprType, exp: ExprType, bindings: BindingsType) -> BindingsType:
    """
    Match a pattern against an expression with bindings.

    Pattern syntax:
        ["?", "name"]            - match any expression
        ["?c", "name"]           - match constants only
        ["?v", "name"]           - match variables only
        ["?free", "name", "var"] - match expression not containing var
        ["?...", "name"]         - match remaining args (zero or more)
        ["?...", "name", "const"] - match remaining args (each must be constant)
        literal                  - match exact value

    Args:
        pat: The pattern to match
        exp: The expression to match against
        bindings: Current bindings dictionary

    Returns:
        Updated bindings on success, "failed" on failure
    """
    if bindings == "failed":
        return "failed"

    if null(pat):
        return bindings if null(exp) else "failed"

    if atom(pat):
        return bindings if atom(exp) and pat == exp else "failed"

    if arbitrary_constant(pat):
        return extend_bindings(pat, exp, bindings) if constant(exp) else "failed"

    if arbitrary_variable(pat):
        return extend_bindings(pat, exp, bindings) if variable(exp) else "failed"

    if arbitrary_expression(pat):
        return extend_bindings(pat, exp, bindings)

    if arbitrary_free(pat):
        # ["?free", "name", "var"] - match if exp doesn't contain var
        var_to_exclude = pat[2]
        # Look up var_to_exclude in bindings in case it's bound
        actual_var = lookup(var_to_exclude, bindings)
        if isinstance(actual_var, str) and not free_in(actual_var, exp):
            return extend_bindings(pat, exp, bindings)
        return "failed"

    # Rest pattern can only appear when matching within a compound
    # It's handled specially in the compound matching below
    if arbitrary_rest(pat):
        # This case shouldn't be reached directly - rest patterns are
        # handled in compound matching. But if somehow called directly
        # with a list expression, bind the whole thing.
        if isinstance(exp, list):
            return extend_bindings(pat, exp, bindings)
        return "failed"

    if atom(exp):
        return "failed"

    # Both are compound - structural matching
    if null(pat) or null(exp):
        return "failed"

    return match_compound(pat, exp, bindings)


def match_compound(pat: List, exp: List, bindings: BindingsType) -> BindingsType:
    """
    Match compound patterns against compound expressions.

    Handles rest patterns (?...) which must appear at the end.
    """
    if bindings == "failed":
        return "failed"

    # Base case: both empty
    if null(pat) and null(exp):
        return bindings

    # Pattern empty but expression has more - fail (unless we had a rest pattern)
    if null(pat):
        return "failed"

    # Get the current pattern element
    current_pat = car(pat)
    rest_pat = cdr(pat)

    # Check if current pattern is a rest pattern
    if arbitrary_rest(current_pat):
        # Rest pattern must be last in the pattern list
        if not null(rest_pat):
            raise ValueError("Rest pattern (?...) must be last in compound pattern")

        # Collect remaining expressions
        remaining = exp if isinstance(exp, list) else [exp] if not null(exp) else []

        # Check type constraint if present
        type_constraint = rest_type_constraint(current_pat)
        if type_constraint:
            for item in remaining:
                if type_constraint == "const" and not constant(item):
                    return "failed"
                elif type_constraint == "var" and not variable(item):
                    return "failed"

        # Bind the list of remaining expressions
        return extend_bindings(current_pat, remaining, bindings)

    # Expression empty but pattern has more
    if null(exp):
        # Current pattern must be a rest pattern (which can match empty)
        # Otherwise fail - we need expressions to match non-rest patterns
        return "failed"

    # Normal case: match current elements, then recurse
    submatch = match(current_pat, car(exp), bindings)
    return match_compound(rest_pat, cdr(exp), submatch)


# ============================================================
# Instantiation
# ============================================================

def instantiate(
    skeleton: ExprType,
    bindings: BindingsType,
    fold_funcs: Optional[FoldFuncsType] = None,
) -> ExprType:
    """
    Instantiate a skeleton with bindings.

    Skeleton syntax:
        [":", "name"]           - substitute with bound value
        [":...", "name"]        - splice bound list into parent
        ["!", "op", args...]    - compute op(args) immediately
        literal                 - keep as-is

    Args:
        skeleton: The skeleton to instantiate
        bindings: The bindings dictionary
        fold_funcs: Optional fold functions for compute (!) evaluation

    Returns:
        The instantiated expression
    """

    def loop(s):
        if null(s):
            return []
        if atom(s):
            return s
        if skeleton_evaluation(s):
            name = car(cdr(s))
            return lookup(name, bindings)
        if skeleton_splice(s):
            # Splice is handled at the compound level, not here
            # This shouldn't be reached for well-formed skeletons
            name = car(cdr(s))
            return lookup(name, bindings)
        if skeleton_compute(s):
            # Compute form: ["!", "op", arg1, arg2, ...]
            # First instantiate all args, then evaluate
            op = s[1]
            raw_args = s[2:]
            # Instantiate each arg
            args = [loop(arg) for arg in raw_args]
            # Try to evaluate if we have fold_funcs
            if fold_funcs and op in fold_funcs:
                try:
                    result = fold_funcs[op](args)
                    if result is not None:
                        # Preserve integer type for numeric results
                        if isinstance(result, float) and result.is_integer():
                            return int(result)
                        return result
                except Exception:
                    pass
            # If can't evaluate, return as regular expression
            return [op] + args
        # For compound expressions, use special handling for splice
        return instantiate_compound(s, bindings, fold_funcs)

    return loop(skeleton)


def instantiate_compound(
    skeleton: List,
    bindings: BindingsType,
    fold_funcs: Optional[FoldFuncsType] = None,
) -> List:
    """
    Instantiate a compound skeleton, handling splice patterns.

    When a splice pattern [":...", "name"] is encountered, its bound list
    is spliced into the result rather than inserted as a single element.
    """
    if null(skeleton):
        return []

    first = car(skeleton)
    rest = cdr(skeleton)

    # Check if first element is a splice pattern
    if skeleton_splice(first):
        name = car(cdr(first))
        spliced = lookup(name, bindings)
        # spliced should be a list - splice it in, then continue with rest
        if isinstance(spliced, list):
            rest_instantiated = instantiate_compound(rest, bindings, fold_funcs)
            return spliced + rest_instantiated
        else:
            # If not a list, treat as single element (shouldn't normally happen)
            rest_instantiated = instantiate_compound(rest, bindings, fold_funcs)
            return [spliced] + rest_instantiated

    # Normal element - instantiate it and cons with rest
    first_instantiated = instantiate(first, bindings, fold_funcs)
    rest_instantiated = instantiate_compound(rest, bindings, fold_funcs)
    return [first_instantiated] + rest_instantiated


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
        """Simplify an expression using the rules."""
        max_iterations = 1000
        iterations = 0

        while iterations < max_iterations:
            iterations += 1
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

        # Check if all arguments are numeric constants
        if not all(isinstance(arg, (int, float)) for arg in args):
            return exp

        try:
            handler = active_fold_funcs[op]
            result = handler(args)

            # None means can't fold (wrong arity, etc.)
            if result is None:
                return exp

            # Preserve integer type when possible
            if isinstance(result, float) and result.is_integer():
                result = int(result)
            return result
        except Exception:
            return exp

    def try_rules(exp: ExprType) -> ExprType:
        """Try applying rules to an expression."""
        for rule in rules:
            pat = rule[0]
            skel = rule[1]

            bindings = match(pat, exp, [])
            if bindings != "failed":
                result = instantiate(skel, bindings, active_fold_funcs)
                return simplify(result)

        return exp

    return simplify


# Convenience alias
simplifier = rewriter
