"""Expression model: parsing, formatting, and the ``E`` builder.

These utilities are pure functions and a single namespace object over the
``ExprType`` (nested-list s-expression representation defined in
``rewriter.py``). They are split out so that downstream modules
(``optimize.py``, ``trace.py``, ``equivalence.py``) can format expressions
without importing the engine and without lazy-import workarounds.
"""

from fractions import Fraction
from typing import List, Tuple, Union

from .rewriter import ExprType, coerce_number


# ============================================================
# Expression Builder
# ============================================================

class _ExprBuilder:
    """Expression builder for RERUM.

    Provides convenient ways to construct expressions without privileging
    any particular operators.

    Examples::

        from rerum import E

        # Parse s-expression string
        expr = E("(+ x (* 2 y))")

        # Build programmatically with E.op()
        expr = E.op("+", "x", E.op("*", 2, "y"))

        # Create variables
        x, y = E.vars("x", "y")
        expr = E.op("+", x, E.op("*", 2, y))

        # Any operator works - no privileged operations
        E.op("dd", E.op("^", "x", 2), "x")
        E.op("my-custom-op", "a", "b", "c")
    """

    def __call__(self, s: str) -> ExprType:
        """Parse an s-expression string."""
        return parse_sexpr(s)

    def op(self, name: str, *args) -> List:
        """Build a compound expression with the given operator and arguments."""
        return [name] + list(args)

    def var(self, name: str) -> str:
        """Create a variable. Variables are just strings."""
        return name

    def vars(self, *names: str) -> Tuple[str, ...]:
        """Create multiple variables for unpacking."""
        return names

    def const(self, value: Union[int, float]) -> Union[int, float]:
        """Create a constant. Constants are just numbers."""
        return value

    def __repr__(self) -> str:
        return "E (expression builder)"


# Singleton instance
E = _ExprBuilder()


# ============================================================
# S-expression I/O
# ============================================================

def parse_sexpr(s: str) -> ExprType:
    """Parse an S-expression string into a nested list.

    Examples::

        parse_sexpr("(+ x 1)")        # ["+", "x", 1]
        parse_sexpr("(dd (^ x 2) x)") # ["dd", ["^", "x", 2], "x"]
    """
    s = s.strip()
    if not s:
        return None

    if s.startswith('('):
        depth = 0
        parts = []
        current = ''
        i = 1  # Skip opening paren

        while i < len(s):
            c = s[i]
            if c == '(':
                depth += 1
                current += c
            elif c == ')':
                if depth == 0:
                    if current.strip():
                        parts.append(parse_sexpr(current.strip()))
                    break
                depth -= 1
                current += c
            elif c in ' \t\n' and depth == 0:
                if current.strip():
                    parts.append(parse_sexpr(current.strip()))
                current = ''
            else:
                current += c
            i += 1

        return parts

    # Parse atom: try number first
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            pass

    # Rational literal: p/q (see parse_rational_token).
    rat = parse_rational_token(s)
    if rat is not None:
        return rat

    # Pattern variable syntax conversion
    if s.startswith('?'):
        rest = s[1:]

        # Check for rest pattern (ends with ...)
        is_rest = rest.endswith('...')
        if is_rest:
            rest = rest[:-3]

        # New typed syntax: ?name:type or ?name:free(var)
        if ':' in rest:
            name_part, type_part = rest.split(':', 1)
            name = name_part.strip() or 'x'

            if is_rest:
                if type_part == 'const':
                    return ["?...", name, "const"]
                elif type_part == 'var':
                    return ["?...", name, "var"]
                else:
                    return ["?...", name]
            else:
                if type_part == 'const':
                    return ["?c", name]
                elif type_part == 'var':
                    return ["?v", name]
                elif type_part == 'expr':
                    return ["?", name]
                elif type_part.startswith('free(') and type_part.endswith(')'):
                    var = type_part[5:-1].strip()
                    return ["?free", name, var]
                else:
                    return ["?", name]

        else:
            # Plain pattern variable: ?x or ?x...
            name = rest.strip() or 'x'
            if is_rest:
                return ["?...", name]
            return ["?", name]

    elif s.startswith(':'):
        rest = s[1:].strip()
        if rest.endswith('...'):
            name = rest[:-3].strip()
            return [":...", name]
        else:
            return [":", rest]

    # Plain symbol
    return s


def parse_rational_token(s: str):
    """Parse a rational-literal token ``p/q`` to its exact number, or None.

    The single definition of the rational-literal lexical form, used by
    ``parse_sexpr`` (text atoms) AND the JSON rule loader (string atoms in
    persisted rules -- the JSON encoding of a Fraction is this same token).
    Integer numerator and denominator only; narrowed through
    ``coerce_number`` so an int-valued literal like ``4/2`` becomes the
    int 2. A zero denominator or anything non-integer around the slash
    (``x/y``, ``1/x``, ``1/2/3``) yields None (a plain symbol).
    """
    if not isinstance(s, str) or "/" not in s:
        return None
    num_s, _, den_s = s.partition("/")
    digits = num_s[1:] if num_s.startswith("-") else num_s
    if digits.isdigit() and den_s.isdigit() and int(den_s) != 0:
        return coerce_number(Fraction(int(num_s), int(den_s)))
    return None


def format_sexpr(expr: ExprType, dsl_syntax: bool = True) -> str:
    """Format an expression as an S-expression string.

    Args:
        expr: Expression to format.
        dsl_syntax: If True (default), use DSL syntax for patterns
            (``?x``, ``:x``). If False, use raw list syntax
            (``(? x)``, ``(: x)``).

    Examples::

        format_sexpr(["+", "x", 1])        # "(+ x 1)"
        format_sexpr(["?", "x"])           # "?x"
        format_sexpr([":", "x"])           # ":x"
        format_sexpr(["?c", "n"])          # "?n:const"
        format_sexpr(["?...", "xs"])       # "?xs..."
        format_sexpr([":...", "xs"])       # ":xs..."
    """
    if isinstance(expr, list):
        if not expr:
            return "()"

        # Handle pattern/skeleton DSL syntax
        if dsl_syntax and len(expr) == 2:
            op = expr[0]
            if op == "?":
                return f"?{expr[1]}"
            elif op == ":":
                return f":{expr[1]}"
            elif op == "?c":
                return f"?{expr[1]}:const"
            elif op == "?v":
                return f"?{expr[1]}:var"
            elif op == "?...":
                return f"?{expr[1]}..."
            elif op == ":...":
                return f":{expr[1]}..."

        if dsl_syntax and len(expr) == 3:
            op = expr[0]
            if op == "?free":
                return f"?{expr[1]}:free({expr[2]})"
            elif op == "?...":
                return f"?{expr[1]}:{expr[2]}..."

        parts = [format_sexpr(e, dsl_syntax) for e in expr]
        return "(" + " ".join(parts) + ")"
    elif isinstance(expr, Fraction):
        # Rational literal, the exact round-trip form: parse_sexpr("1/3")
        # gives back this same Fraction atom (a division EXPRESSION would
        # parse to the different structure ["/", 1, 3]). Matches the corpus
        # encoder's rendering (training.corpus_json_default).
        return f"{expr.numerator}/{expr.denominator}"
    elif isinstance(expr, (int, float)):
        return str(expr)
    else:
        return str(expr)


def expr_to_tuple(expr: ExprType) -> tuple:
    """Convert an expression to a hashable tuple form.

    Used for deduplication in equivalence enumeration.

    Examples::

        expr_to_tuple(["+", "x", 1])             # ("+", "x", 1)
        expr_to_tuple(["+", ["*", "a", "b"], "c"]) # ("+", ("*", "a", "b"), "c")
    """
    if isinstance(expr, list):
        return tuple(expr_to_tuple(e) for e in expr)
    return expr
