#!/usr/bin/env python3
"""
RERUM Feature Demonstration

This script demonstrates all the major features of the RERUM library.
"""

from pathlib import Path
from rerum import (
    RuleEngine, E,
    FULL_PRELUDE, ARITHMETIC_PRELUDE,
    format_sexpr
)


def section(title: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def demo_basic_usage():
    """Demonstrate basic rule engine usage."""
    section("Basic Usage")

    engine = RuleEngine.from_dsl('''
        @add-zero: (+ ?x 0) => :x
        @mul-one: (* ?x 1) => :x
        @mul-zero: (* ?x 0) => 0
    ''')

    examples = [
        "(+ y 0)",
        "(* x 1)",
        "(* (+ a 0) 0)",
    ]

    for expr_str in examples:
        expr = E(expr_str)
        result = engine(expr)
        print(f"  {expr_str} => {format_sexpr(result)}")


def demo_guards():
    """Demonstrate conditional guards."""
    section("Conditional Guards")

    engine = (RuleEngine()
        .with_prelude(FULL_PRELUDE)
        .load_dsl('''
            @abs-pos: (abs ?x) => :x when (! > :x 0)
            @abs-neg: (abs ?x) => (! - 0 :x) when (! < :x 0)
            @abs-zero: (abs ?x) => 0 when (! = :x 0)
        '''))

    examples = [
        ("(abs 5)", "positive number"),
        ("(abs -5)", "negative number"),
        ("(abs 0)", "zero"),
    ]

    for expr_str, desc in examples:
        result = engine(E(expr_str))
        print(f"  {expr_str} ({desc}) => {format_sexpr(result)}")


def demo_priorities():
    """Demonstrate rule priorities."""
    section("Rule Priorities")

    engine = RuleEngine.from_dsl('''
        @general: (+ ?x ?y) => (add :x :y)
        @zero-left[100]: (+ 0 ?x) => :x
        @zero-right[100]: (+ ?x 0) => :x
    ''')

    examples = [
        ("(+ 0 y)", "zero on left (specific rule)"),
        ("(+ x 0)", "zero on right (specific rule)"),
        ("(+ a b)", "general case"),
    ]

    for expr_str, desc in examples:
        result = engine(E(expr_str))
        print(f"  {expr_str} ({desc}) => {format_sexpr(result)}")


def demo_groups():
    """Demonstrate named rulesets (groups)."""
    section("Named Rulesets (Groups)")

    engine = RuleEngine.from_dsl('''
        [algebra]
        @add-zero: (+ ?x 0) => :x
        @mul-one: (* ?x 1) => :x

        [expand]
        @square: (square ?x) => (* :x :x)
    ''')

    print(f"  Available groups: {engine.groups()}")

    expr = E("(+ (square x) 0)")

    # All groups
    result = engine(expr)
    print(f"\n  Expression: (+ (square x) 0)")
    print(f"  All groups: {format_sexpr(result)}")

    # Only algebra
    result = engine(expr, groups=["algebra"])
    print(f"  Only [algebra]: {format_sexpr(result)}")

    # Only expand
    result = engine(expr, groups=["expand"])
    print(f"  Only [expand]: {format_sexpr(result)}")


def demo_strategies():
    """Demonstrate rewriting strategies."""
    section("Rewriting Strategies")

    engine = RuleEngine.from_dsl('''
        @expand: (double ?x) => (+ :x :x)
        @fold: (+ ?x ?x) => (* 2 :x)
    ''')

    expr = E("(double (double x))")
    print(f"  Expression: {format_sexpr(expr)}")

    for strategy in ["exhaustive", "once", "bottomup", "topdown"]:
        result = engine(expr, strategy=strategy)
        print(f"  {strategy:12}: {format_sexpr(result)}")


def demo_tracing():
    """Demonstrate trace formatting."""
    section("Tracing")

    engine = RuleEngine.from_dsl('''
        @add-zero: (+ ?x 0) => :x
        @mul-one: (* ?x 1) => :x
    ''')

    expr = E("(+ (* x 1) 0)")
    result, trace = engine(expr, trace=True)

    print("  Verbose format (default):")
    for line in str(trace).split('\n'):
        print(f"    {line}")

    print(f"\n  Compact: {trace.format('compact')}")
    print(f"  Rules: {trace.format('rules')}")
    print(f"  Summary: {trace.summary()}")


def demo_constant_folding():
    """Demonstrate constant folding with guards."""
    section("Constant Folding")

    engine = (RuleEngine()
        .with_prelude(FULL_PRELUDE)
        .load_dsl('''
            @fold-add: (+ ?a ?b) => (! + :a :b) when (! and (! const? :a) (! const? :b))
            @fold-mul: (* ?a ?b) => (! * :a :b) when (! and (! const? :a) (! const? :b))
        '''))

    examples = [
        ("(+ 1 2)", "both constants"),
        ("(+ x 2)", "mixed - no folding"),
        ("(+ (* 2 3) (* 4 5))", "nested constants"),
    ]

    for expr_str, desc in examples:
        result = engine(E(expr_str))
        print(f"  {expr_str} ({desc}) => {format_sexpr(result)}")


def demo_sequencing():
    """Demonstrate engine sequencing."""
    section("Engine Sequencing")

    expand = RuleEngine.from_dsl('''
        @square: (square ?x) => (* :x :x)
    ''')

    simplify = (RuleEngine()
        .with_prelude(ARITHMETIC_PRELUDE)
        .load_dsl('''
            @fold: (* ?a ?b) => (! * :a :b) when (! and (! const? :a) (! const? :b))
        '''))

    # Sequence engines
    pipeline = expand >> simplify

    expr = E("(square 5)")
    print(f"  Expression: {format_sexpr(expr)}")
    print(f"  After expand only: {format_sexpr(expand(expr))}")
    print(f"  After pipeline (expand >> simplify): {format_sexpr(pipeline(expr))}")


def demo_file_loading():
    """Demonstrate loading rules from files."""
    section("Loading Rules from Files")

    examples_dir = Path(__file__).parent

    # Load algebra rules
    algebra = (RuleEngine()
        .with_prelude(FULL_PRELUDE)
        .load_file(examples_dir / "algebra.rules"))

    print(f"  Loaded {len(algebra)} rules from algebra.rules")
    print(f"  Groups: {algebra.groups()}")

    # Test some simplifications
    examples = [
        "(+ x 0)",
        "(* y 1)",
        "(* z 0)",
        "(+ 2 3)",
    ]

    print("\n  Simplifications:")
    for expr_str in examples:
        result = algebra(E(expr_str))
        print(f"    {expr_str} => {format_sexpr(result)}")


def demo_derivatives():
    """Demonstrate calculus rules."""
    section("Symbolic Differentiation")

    examples_dir = Path(__file__).parent

    # Load calculus rules
    calculus = RuleEngine.from_file(examples_dir / "calculus.rules")

    print(f"  Loaded {len(calculus)} rules from calculus.rules")

    # Test derivatives
    examples = [
        ("(dd 5 x)", "constant"),
        ("(dd x x)", "variable same"),
        ("(dd y x)", "variable different"),
        ("(dd (+ x y) x)", "sum rule"),
        ("(dd (^ x 2) x)", "power rule"),
    ]

    print("\n  Derivatives:")
    for expr_str, desc in examples:
        result = calculus(E(expr_str))
        print(f"    d/dx[{desc}]: {expr_str} => {format_sexpr(result)}")


def main():
    """Run all demonstrations."""
    print("RERUM - Rewriting Expressions via Rules Using Morphisms")
    print("Feature Demonstration")

    demo_basic_usage()
    demo_guards()
    demo_priorities()
    demo_groups()
    demo_strategies()
    demo_tracing()
    demo_constant_folding()
    demo_sequencing()
    demo_file_loading()
    demo_derivatives()

    print("\n" + "="*60)
    print(" Demo complete!")
    print("="*60)


if __name__ == "__main__":
    main()
