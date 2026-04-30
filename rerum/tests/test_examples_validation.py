"""Tests for examples validation at load time."""

import pytest
from rerum import RuleEngine
from rerum.engine import (
    ExampleValidationError, _validate_example, RuleMetadata,
    parse_sexpr, ARITHMETIC_PRELUDE, parse_rule_line,
)
from rerum.rewriter import FULL_PRELUDE


class TestValidateExampleHelper:
    def test_pass_simple(self):
        # @add-zero: (+ ?x 0) => :x
        # Example: (+ y 0) -> y
        meta = RuleMetadata(name="add-zero")
        pattern = ["+", ["?", "x"], 0]
        skeleton = [":", "x"]
        example = {"in": "(+ y 0)", "out": "y"}
        # Should not raise.
        _validate_example(pattern, skeleton, meta, example, fold_funcs={})

    def test_fail_pattern_mismatch(self):
        meta = RuleMetadata(name="add-zero")
        pattern = ["+", ["?", "x"], 0]
        skeleton = [":", "x"]
        # Input doesn't match pattern.
        example = {"in": "(* y 0)", "out": "y"}
        with pytest.raises(ExampleValidationError, match="pattern does not match"):
            _validate_example(pattern, skeleton, meta, example, fold_funcs={})

    def test_fail_output_mismatch(self):
        meta = RuleMetadata(name="add-zero")
        pattern = ["+", ["?", "x"], 0]
        skeleton = [":", "x"]
        # Wrong expected output.
        example = {"in": "(+ y 0)", "out": "z"}
        with pytest.raises(ExampleValidationError, match="produced"):
            _validate_example(pattern, skeleton, meta, example, fold_funcs={})

    def test_fail_condition_fails(self):
        # Rule with a condition; example input doesn't satisfy it.
        results = parse_rule_line(
            "@fold: (+ ?a ?b) => (! + :a :b) when (! and (! const? :a) (! const? :b))"
        )
        meta, pattern, skeleton = results[0]
        example = {"in": "(+ x 0)", "out": "x"}  # x is not const, condition fails
        with pytest.raises(ExampleValidationError, match="condition fails"):
            _validate_example(pattern, skeleton, meta, example,
                              fold_funcs=FULL_PRELUDE)

    def test_pass_with_fold_funcs(self):
        # @fold: (+ ?a:const ?b:const) => (! + :a :b)
        results = parse_rule_line(
            "@fold: (+ ?a:const ?b:const) => (! + :a :b)"
        )
        meta, pattern, skeleton = results[0]
        example = {"in": "(+ 2 3)", "out": "5"}
        # Should not raise.
        _validate_example(pattern, skeleton, meta, example,
                          fold_funcs=ARITHMETIC_PRELUDE)

    def test_explicit_rev_direction(self):
        # Bidirectional commute. The rev direction is (+ ?y ?x) => (+ :x :y).
        results = parse_rule_line("@commute: (+ ?x ?y) <=> (+ :y :x)")
        rev_meta, rev_pattern, rev_skeleton = results[1]  # -rev
        # rev pattern matches (+ b a); applies (+ :x :y) where :x and :y are
        # bound from the rev pattern.
        example = {"in": "(+ b a)", "out": "(+ a b)"}
        _validate_example(rev_pattern, rev_skeleton, rev_meta, example,
                          fold_funcs={})

    def test_error_carries_rule_name_and_example(self):
        meta = RuleMetadata(name="my-rule")
        pattern = ["+", ["?", "x"], 0]
        skeleton = [":", "x"]
        example = {"in": "(+ y 0)", "out": "z"}
        with pytest.raises(ExampleValidationError) as exc_info:
            _validate_example(pattern, skeleton, meta, example, fold_funcs={})
        assert exc_info.value.rule_name == "my-rule"
        assert exc_info.value.example == example
