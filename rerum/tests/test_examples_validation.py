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

    def test_missing_in_key_raises_validation_error(self):
        meta = RuleMetadata(name="r1")
        pattern = ["a", ["?", "x"]]
        skeleton = [":", "x"]
        with pytest.raises(ExampleValidationError, match="must be a dict"):
            _validate_example(pattern, skeleton, meta, {"out": "x"}, fold_funcs={})

    def test_missing_out_key_raises_validation_error(self):
        meta = RuleMetadata(name="r1")
        pattern = ["a", ["?", "x"]]
        skeleton = [":", "x"]
        with pytest.raises(ExampleValidationError, match="must be a dict"):
            _validate_example(pattern, skeleton, meta, {"in": "(a 1)"}, fold_funcs={})

    def test_non_dict_example_raises_validation_error(self):
        meta = RuleMetadata(name="r1")
        pattern = ["a", ["?", "x"]]
        skeleton = [":", "x"]
        with pytest.raises(ExampleValidationError, match="must be a dict"):
            _validate_example(pattern, skeleton, meta, "not a dict", fold_funcs={})


class TestLoaderValidation:
    def test_load_rules_with_valid_examples_passes(self):
        from rerum import RuleEngine
        text = '''{"rules": [{
            "name": "add-zero",
            "pattern": ["+", ["?", "x"], 0],
            "skeleton": [":", "x"],
            "examples": [{"in": "(+ y 0)", "out": "y"}]
        }]}'''
        engine = RuleEngine()
        engine.load_rules_from_json(text)  # method on engine; should not raise
        _, meta = engine["add-zero"]
        assert meta.examples == [{"in": "(+ y 0)", "out": "y"}]

    def test_load_rules_with_bad_example_raises(self):
        from rerum import RuleEngine
        text = '''{"rules": [{
            "name": "broken",
            "pattern": ["+", ["?", "x"], 0],
            "skeleton": [":", "x"],
            "examples": [{"in": "(+ y 0)", "out": "z"}]
        }]}'''
        engine = RuleEngine()
        with pytest.raises(ExampleValidationError, match="produced"):
            engine.load_rules_from_json(text)

    def test_validate_examples_false_skips_validation(self):
        from rerum import RuleEngine
        text = '''{"rules": [{
            "name": "broken",
            "pattern": ["+", ["?", "x"], 0],
            "skeleton": [":", "x"],
            "examples": [{"in": "(+ y 0)", "out": "z"}]
        }]}'''
        engine = RuleEngine()
        # No validation; should load.
        engine.load_rules_from_json(text, validate_examples=False)
        _, meta = engine["broken"]
        assert meta.examples == [{"in": "(+ y 0)", "out": "z"}]

    def test_first_failing_example_reported_loud(self):
        from rerum import RuleEngine
        text = '''{"rules": [{
            "name": "rule-a",
            "pattern": ["a", ["?", "x"]],
            "skeleton": [":", "x"],
            "examples": [{"in": "(a 1)", "out": "1"}, {"in": "(a 2)", "out": "wrong"}]
        }]}'''
        engine = RuleEngine()
        with pytest.raises(ExampleValidationError) as exc_info:
            engine.load_rules_from_json(text)
        # Second example is the failing one.
        assert exc_info.value.example == {"in": "(a 2)", "out": "wrong"}

    def test_bidirectional_example_with_rev_direction(self):
        from rerum import RuleEngine
        text = '''{"rules": [{
            "name": "commute",
            "bidirectional": true,
            "pattern": ["+", ["?", "x"], ["?", "y"]],
            "skeleton": ["+", [":", "y"], [":", "x"]],
            "examples": [
                {"in": "(+ a b)", "out": "(+ b a)", "direction": "fwd"},
                {"in": "(+ a b)", "out": "(+ b a)", "direction": "rev"}
            ]
        }]}'''
        engine = RuleEngine()
        engine.load_rules_from_json(text)  # both directions valid

    def test_load_dsl_validates_examples_via_sidecar(self):
        # Examples can't appear in DSL syntax, but rules added via
        # add_rule with examples (M10) or via load_rules_from_json
        # validate at load time. This test confirms load_dsl path is
        # not broken by the validation hook (no examples = no validation).
        from rerum import RuleEngine
        engine = RuleEngine()
        engine.load_dsl('@r1 {category=identity}: (a ?x) => :x')
        # Should not raise.
        _, meta = engine["r1"]
        assert meta.category == "identity"


class TestOnDemandValidation:
    def test_validate_examples_walks_all_rules(self):
        from rerum import RuleEngine
        text = '''{"rules": [
            {"name": "r1",
             "pattern": ["a", ["?", "x"]], "skeleton": [":", "x"],
             "examples": [{"in": "(a 5)", "out": "5"}]},
            {"name": "r2",
             "pattern": ["b", ["?", "x"]], "skeleton": [":", "x"],
             "examples": [{"in": "(b 7)", "out": "7"}]}
        ]}'''
        engine = RuleEngine()
        engine.load_rules_from_json(text)
        # Should not raise; both rules' examples are valid.
        engine.validate_examples()

    def test_validate_examples_after_prelude_change(self):
        # Load rules without a prelude; load with validate_examples=False
        # because the example needs folding. Then set the prelude and
        # call validate_examples().
        from rerum import RuleEngine, ARITHMETIC_PRELUDE
        text = '''{"rules": [{
            "name": "fold-add",
            "pattern": ["+", ["?c", "a"], ["?c", "b"]],
            "skeleton": ["!", "+", [":", "a"], [":", "b"]],
            "examples": [{"in": "(+ 2 3)", "out": "5"}]
        }]}'''
        engine = RuleEngine()
        engine.load_rules_from_json(text, validate_examples=False)
        # Without fold_funcs, validation would fail.
        engine._fold_funcs = ARITHMETIC_PRELUDE
        engine.validate_examples()  # should pass now

    def test_validate_examples_raises_on_bad(self):
        from rerum import RuleEngine
        text = '''{"rules": [{
            "name": "bad",
            "pattern": ["a", ["?", "x"]], "skeleton": [":", "x"],
            "examples": [{"in": "(a 1)", "out": "wrong"}]
        }]}'''
        engine = RuleEngine()
        engine.load_rules_from_json(text, validate_examples=False)
        with pytest.raises(ExampleValidationError):
            engine.validate_examples()

    def test_validate_examples_no_rules_with_examples_passes(self):
        from rerum import RuleEngine
        engine = RuleEngine.from_dsl('@r1: (a ?x) => :x')
        # No examples on any rule; should be a no-op.
        engine.validate_examples()
