"""
RERUM - Rewriting Expressions via Rules Using Morphisms

A pattern matching and term rewriting library for symbolic computation.

Quick Start:
    from rerum import RuleEngine

    engine = RuleEngine.from_dsl('''
        @add-zero: (+ ?x 0) => :x
        @mul-one: (* ?x 1) => :x
    ''')

    result = engine(["+", "y", 0])  # => "y"

DSL Syntax:
    # Comments start with #
    @rule-name: (pattern) => (skeleton)
    @rule-name "Description": (pattern) => (skeleton)

Pattern Syntax:
    ?x or ?x:expr     - match any expression, bind to x
    ?x:const          - match constant only
    ?x:var            - match variable only
    ?x:free(v)        - match expression not containing v
    :x                - substitute bound value

Example Rules File (algebra.rules):
    # Basic algebra
    @add-zero "x + 0 = x": (+ ?x 0) => :x
    @mul-one: (* ?x 1) => :x
    @mul-zero: (* ?x 0) => 0

    # Derivatives
    @dd-const: (dd ?c:const ?v:var) => 0
    @dd-var: (dd ?x:var ?x) => 1
"""

__version__ = "0.1.0"
__author__ = "spinoza"

# Core rewriter components
from .rewriter import (
    rewriter,
    simplifier,
    match,
    instantiate,
    ExprType,
    BindingsType,
    RuleType,
    NumericType,
    FoldHandler,
    FoldFuncsType,
    # Bindings classes
    Bindings,
    NoMatch,
    wrap_bindings,
    # Fold operation builders
    nary_fold,
    unary_only,
    binary_only,
    special_minus,
    safe_div,
    # Standard preludes
    ARITHMETIC_PRELUDE,
    MATH_PRELUDE,
    MINIMAL_PRELUDE,
    PREDICATE_PRELUDE,
    FULL_PRELUDE,
    NO_PRELUDE,
)

# Engine and DSL
from .engine import (
    RuleEngine,
    SequencedEngine,
    RuleMetadata,
    RewriteStep,
    RewriteTrace,
    E,
    parse_sexpr,
    format_sexpr,
    parse_rule_line,
    load_rules_from_dsl,
    load_rules_from_file,
    load_rules_from_json,
)

# Public API
__all__ = [
    # Version
    "__version__",
    # Core
    "rewriter",
    "simplifier",
    "match",
    "instantiate",
    # Types
    "ExprType",
    "BindingsType",
    "RuleType",
    "NumericType",
    "FoldHandler",
    "FoldFuncsType",
    # Bindings
    "Bindings",
    "NoMatch",
    "wrap_bindings",
    # Fold operation builders
    "nary_fold",
    "unary_only",
    "binary_only",
    "special_minus",
    "safe_div",
    # Standard preludes
    "ARITHMETIC_PRELUDE",
    "MATH_PRELUDE",
    "MINIMAL_PRELUDE",
    "PREDICATE_PRELUDE",
    "FULL_PRELUDE",
    "NO_PRELUDE",
    # Engine
    "RuleEngine",
    "SequencedEngine",
    "RuleMetadata",
    "RewriteStep",
    "RewriteTrace",
    # Expression builder
    "E",
    # DSL utilities
    "parse_sexpr",
    "format_sexpr",
    "parse_rule_line",
    "load_rules_from_dsl",
    "load_rules_from_file",
    "load_rules_from_json",
]
