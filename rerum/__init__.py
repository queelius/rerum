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

__version__ = "0.9.0"
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
    coerce_number,
    free_symbols,
    gensym,
    combine_preludes,
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
    EqualityProof,
    OptimizationResult,
    ExampleValidationError,
    E,
    parse_sexpr,
    format_sexpr,
    parse_rule_line,
    load_rules_from_dsl,
    load_rules_from_file,
    load_rules_from_json,
    # Cost functions
    expr_size,
    expr_depth,
    expr_ops,
    expr_atoms,
    make_op_cost_fn,
    COST_METRICS,
)

# Theory-driven normalization (Phase 2)
from .normalize import (
    Theory,
    normalize,
    flatten,
    canonical_sort,
    collect_like_terms,
    ORDER_KEY,
)

# Trace helpers
from .trace import splice_at, rule_identity

# Hooks
from .hooks import (
    HooksError,
    Resolution,
    HookContext,
    HookError,
    ResolutionError,
    ResolverLoopError,
)

# Goal-directed search (escalation driver)
from .solve import (
    solve,
    SolveResult,
    contains_op,
)

# General numeric evaluation
from .numeval import (
    numeval,
    numeric_equiv,
    NumevalError,
    NumevalDomainError,
)

# Trace-to-text / trace-to-record projection layer (Phase 4)
from .training import (
    to_training_record,
    to_prose,
    generate_corpus,
    corpus_json_default,
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
    "coerce_number",
    "free_symbols",
    "gensym",
    "combine_preludes",
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
    "splice_at",
    "rule_identity",
    "EqualityProof",
    "OptimizationResult",
    "ExampleValidationError",
    # Expression builder
    "E",
    # DSL utilities
    "parse_sexpr",
    "format_sexpr",
    "parse_rule_line",
    "load_rules_from_dsl",
    "load_rules_from_file",
    "load_rules_from_json",
    # Cost functions
    "expr_size",
    "expr_depth",
    "expr_ops",
    "expr_atoms",
    "make_op_cost_fn",
    "COST_METRICS",
    # Hooks
    "HooksError",
    "HookContext",
    "HookError",
    "Resolution",
    "ResolutionError",
    "ResolverLoopError",
    # Search
    "solve",
    "SolveResult",
    "contains_op",
    # Numeric evaluation
    "numeval",
    "numeric_equiv",
    "NumevalError",
    "NumevalDomainError",
    # Theory-driven normalization
    "Theory",
    "normalize",
    "flatten",
    "canonical_sort",
    "collect_like_terms",
    "ORDER_KEY",
    # Trace-to-text / trace-to-record (Phase 4)
    "to_training_record",
    "to_prose",
    "generate_corpus",
    "corpus_json_default",
]
