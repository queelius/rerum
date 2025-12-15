"""
Example custom prelude for RERUM.

This file demonstrates how to create custom preludes that extend
RERUM's computational capabilities.

Usage:
    rerum -p examples/custom_prelude.py -e "(gcd 12 8)"

Or in scripts:
    :prelude examples/custom_prelude.py
    (gcd 12 8)
"""

import math
from rerum import binary_only, unary_only, nary_fold, ARITHMETIC_PRELUDE

# Start with arithmetic prelude and extend it
PRELUDE = {
    **ARITHMETIC_PRELUDE,

    # Number theory
    "gcd": binary_only(math.gcd),
    "lcm": binary_only(math.lcm),
    "mod": binary_only(lambda a, b: a % b),
    "factorial": unary_only(math.factorial),

    # Additional math functions
    "floor": unary_only(math.floor),
    "ceil": unary_only(math.ceil),
    "round": unary_only(round),

    # Min/max
    "min": binary_only(min),
    "max": binary_only(max),

    # Boolean operations (for use in guards)
    "even?": unary_only(lambda x: x % 2 == 0),
    "odd?": unary_only(lambda x: x % 2 == 1),
    "positive?": unary_only(lambda x: x > 0),
    "negative?": unary_only(lambda x: x < 0),
    "zero?": unary_only(lambda x: x == 0),

    # Comparison (for guards)
    ">": binary_only(lambda a, b: a > b),
    "<": binary_only(lambda a, b: a < b),
    ">=": binary_only(lambda a, b: a >= b),
    "<=": binary_only(lambda a, b: a <= b),
    "=": binary_only(lambda a, b: a == b),
    "!=": binary_only(lambda a, b: a != b),

    # Logical
    "and": binary_only(lambda a, b: a and b),
    "or": binary_only(lambda a, b: a or b),
    "not": unary_only(lambda x: not x),

    # Type predicates
    "const?": unary_only(lambda x: isinstance(x, (int, float))),
    "var?": unary_only(lambda x: isinstance(x, str)),
    "list?": unary_only(lambda x: isinstance(x, list)),
}
