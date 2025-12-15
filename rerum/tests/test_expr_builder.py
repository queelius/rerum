"""Tests for the expression builder E."""

import pytest
from rerum import E


class TestExprBuilder:
    """Tests for E expression builder."""

    def test_parse_simple(self):
        """E() parses simple s-expressions."""
        assert E("x") == "x"
        assert E("42") == 42
        assert E("3.14") == 3.14

    def test_parse_compound(self):
        """E() parses compound s-expressions."""
        assert E("(+ x 1)") == ["+", "x", 1]
        assert E("(* 2 y)") == ["*", 2, "y"]

    def test_parse_nested(self):
        """E() parses nested s-expressions."""
        assert E("(+ x (* 2 y))") == ["+", "x", ["*", 2, "y"]]
        assert E("(dd (^ x 2) x)") == ["dd", ["^", "x", 2], "x"]

    def test_op_simple(self):
        """E.op() builds simple compound expressions."""
        assert E.op("+", "x", 1) == ["+", "x", 1]
        assert E.op("*", 2, "y") == ["*", 2, "y"]

    def test_op_nested(self):
        """E.op() builds nested expressions."""
        expr = E.op("+", "x", E.op("*", 2, "y"))
        assert expr == ["+", "x", ["*", 2, "y"]]

    def test_op_variadic(self):
        """E.op() handles any number of arguments."""
        assert E.op("+") == ["+"]
        assert E.op("+", "x") == ["+", "x"]
        assert E.op("+", "x", "y", "z") == ["+", "x", "y", "z"]

    def test_op_custom_operators(self):
        """E.op() works with any operator name."""
        assert E.op("dd", "x", "y") == ["dd", "x", "y"]
        assert E.op("my-custom-op", "a") == ["my-custom-op", "a"]
        assert E.op("∂", "f", "x") == ["∂", "f", "x"]

    def test_var(self):
        """E.var() creates variables."""
        assert E.var("x") == "x"
        assert E.var("foo") == "foo"

    def test_vars(self):
        """E.vars() creates multiple variables for unpacking."""
        x, y, z = E.vars("x", "y", "z")
        assert x == "x"
        assert y == "y"
        assert z == "z"

    def test_vars_single(self):
        """E.vars() works with single variable."""
        (x,) = E.vars("x")
        assert x == "x"

    def test_const(self):
        """E.const() creates constants."""
        assert E.const(5) == 5
        assert E.const(3.14) == 3.14
        assert E.const(-1) == -1

    def test_combined_usage(self):
        """E methods work together naturally."""
        x, y = E.vars("x", "y")
        expr = E.op("+", x, E.op("*", E.const(2), y))
        assert expr == ["+", "x", ["*", 2, "y"]]

    def test_equivalence_parse_and_op(self):
        """E() and E.op() produce equivalent results."""
        parsed = E("(+ x (* 2 y))")
        built = E.op("+", "x", E.op("*", 2, "y"))
        assert parsed == built

    def test_repr(self):
        """E has a sensible repr."""
        assert "expression builder" in repr(E).lower()


class TestExprBuilderWithEngine:
    """Tests that E works well with RuleEngine."""

    def test_e_with_engine(self):
        """Expressions built with E work with RuleEngine."""
        from rerum import RuleEngine

        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")

        # Using E() to parse
        expr = E("(+ y 0)")
        assert engine(expr) == "y"

        # Using E.op() to build
        expr = E.op("+", "y", 0)
        assert engine(expr) == "y"

    def test_e_with_complex_rules(self):
        """E works with more complex rewriting."""
        from rerum import RuleEngine, ARITHMETIC_PRELUDE

        engine = RuleEngine.from_dsl('''
            @add-zero-r: (+ ?x 0) => :x
            @add-zero-l: (+ 0 ?x) => :x
            @mul-one: (* ?x 1) => :x
            @mul-zero: (* ?x 0) => 0
        ''', fold_funcs=ARITHMETIC_PRELUDE)

        x, y = E.vars("x", "y")
        # (* x 0) => 0, (* y 1) => y, (+ 0 y) => y
        expr = E.op("+", E.op("*", x, 0), E.op("*", y, 1))
        result = engine(expr)
        assert result == "y"
