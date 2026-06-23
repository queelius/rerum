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

__version__ = "0.10.0"
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

# Rule-set manifests (self-describing rule files)
from .manifest import RuleSetManifest, parse_manifest

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

# NOTE: `rerum.solve` (goal-directed best-first SEARCH over the rewrite graph)
# and `rerum.numeval` (numeric evaluation of ground terms) are OPTIONAL,
# NON-CORE layers -- they are search and model-interpretation, not term
# rewriting -- and are deliberately NOT re-exported here. Import them
# explicitly when needed:  `from rerum.solve import solve`,
# `from rerum.numeval import numeval`. Keeping them out of the core API keeps
# `import rerum` a pure term-rewriting surface.

# Trace-to-text / trace-to-record projection layer (Phase 4)
from .training import (
    to_training_record,
    to_prose,
    generate_corpus,
    corpus_json_default,
)

# Confluence analysis (F2)
from .confluence import (
    unify,
    apply_subst,
    critical_pairs,
    check_confluence,
    CriticalPair,
    ConfluenceReport,
    UnsupportedPattern,
)

# Termination analysis (F4)
from .termination import (
    lpo_greater,
    orient,
    check_termination,
    TerminationReport,
)

# Completion (F5)
from .completion import (
    complete,
    CompletionResult,
)

# AC-matching (F3)
from .acmatch import (
    ac_match,
    MatchBudget,
)

# Narrowing (F6)
from .narrowing import (
    narrow,
    solve_equation,
    narrow_step,
    NarrowResult,
    NarrowStep,
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
    # Theory-driven normalization
    "Theory",
    "RuleSetManifest",
    "parse_manifest",
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
    # Confluence analysis
    "unify",
    "apply_subst",
    "critical_pairs",
    "check_confluence",
    "CriticalPair",
    "ConfluenceReport",
    "UnsupportedPattern",
    # Termination analysis
    "lpo_greater",
    "orient",
    "check_termination",
    "TerminationReport",
    # Completion
    "complete",
    "CompletionResult",
    # AC-matching
    "ac_match",
    "MatchBudget",
    # Narrowing
    "narrow",
    "solve_equation",
    "narrow_step",
    "NarrowResult",
    "NarrowStep",
]
