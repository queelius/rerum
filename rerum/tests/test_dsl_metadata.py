"""Tests for DSL `{category=X}` annotation parsing."""

import pytest
from rerum.engine import parse_rule_line, load_rules_from_dsl


class TestSingleLineAnnotation:
    def test_annotation_with_name_priority_description(self):
        results = parse_rule_line(
            '@add-zero[100] "x + 0 = x" {category=identity}: (+ ?x 0) => :x'
        )
        assert len(results) == 1
        meta, pat, skel = results[0]
        assert meta.name == "add-zero"
        assert meta.priority == 100
        assert meta.description == "x + 0 = x"
        assert meta.category == "identity"

    def test_annotation_with_name_only(self):
        results = parse_rule_line(
            '@distrib {category=distributivity}: (* ?x (+ ?y ?z)) => (+ (* :x :y) (* :x :z))'
        )
        assert len(results) == 1
        meta, _, _ = results[0]
        assert meta.name == "distrib"
        assert meta.category == "distributivity"
        assert meta.description is None

    def test_annotation_with_name_and_description(self):
        results = parse_rule_line(
            '@r1 "desc" {category=cat}: (a ?x) => :x'
        )
        meta, _, _ = results[0]
        assert meta.name == "r1"
        assert meta.description == "desc"
        assert meta.category == "cat"

    def test_annotation_with_name_and_priority(self):
        results = parse_rule_line(
            '@r1[50] {category=cat}: (a ?x) => :x'
        )
        meta, _, _ = results[0]
        assert meta.name == "r1"
        assert meta.priority == 50
        assert meta.category == "cat"

    def test_anonymous_rule_with_annotation(self):
        results = parse_rule_line(
            '{category=fold-constant}: (* ?a:const ?b:const) => (! * :a :b)'
        )
        assert len(results) == 1
        meta, _, _ = results[0]
        assert meta.name is None
        assert meta.category == "fold-constant"

    def test_bidirectional_with_annotation(self):
        results = parse_rule_line(
            '@commute {category=commutativity}: (+ ?x ?y) <=> (+ :y :x)'
        )
        assert len(results) == 2  # fwd and rev
        for meta, _, _ in results:
            assert meta.category == "commutativity"

    def test_quoted_value(self):
        results = parse_rule_line(
            '@r1 {category="multi word"}: (a ?x) => :x'
        )
        meta, _, _ = results[0]
        assert meta.category == "multi word"

    def test_whitespace_in_annotation(self):
        results = parse_rule_line(
            '@r1 { category = identity }: (a ?x) => :x'
        )
        meta, _, _ = results[0]
        assert meta.category == "identity"

    def test_no_annotation_still_parses(self):
        results = parse_rule_line('@r1: (a ?x) => :x')
        meta, _, _ = results[0]
        assert meta.name == "r1"
        assert meta.category is None

    def test_unknown_annotation_key_raises(self):
        with pytest.raises(ValueError, match="unknown annotation key"):
            parse_rule_line('@r1 {ref="paper"}: (a ?x) => :x')

    def test_malformed_annotation_raises(self):
        # Missing closing brace
        with pytest.raises(ValueError, match="malformed annotation"):
            parse_rule_line('@r1 {category=identity: (a ?x) => :x')

    def test_quote_with_close_brace_inside(self):
        # Closing brace inside a quoted value should not terminate the
        # annotation early.
        results = parse_rule_line(
            '@r1 {category="has } in it"}: (a ?x) => :x'
        )
        meta, _, _ = results[0]
        assert meta.category == "has } in it"

    def test_unclosed_quote_raises(self):
        with pytest.raises(ValueError, match="unclosed quote"):
            parse_rule_line('@r1 {category="unclosed}: (a ?x) => :x')

    def test_two_annotation_blocks_raises(self):
        with pytest.raises(ValueError, match="multiple annotation blocks"):
            parse_rule_line('@r1 {category=foo} {category=bar}: (a ?x) => :x')

    def test_annotation_only_in_header(self):
        # An `{` after the arrow is left in the body untouched.
        # The annotation extractor must operate on the header only.
        results = parse_rule_line('@r1 {category=foo}: (a ?x) => :x')
        meta, pat, skel = results[0]
        assert meta.category == "foo"
        assert pat == ["a", ["?", "x"]]
        assert skel == [":", "x"]


class TestMultiLineAnnotation:
    def test_multi_line_annotation(self):
        text = """
@distrib {
  category=distributivity
}: (* ?x (+ ?y ?z)) => (+ (* :x :y) (* :x :z))
"""
        rules = load_rules_from_dsl(text)
        assert len(rules) == 1
        meta, _ = rules[0]
        assert meta.name == "distrib"
        assert meta.category == "distributivity"

    def test_multi_line_with_priority_and_description(self):
        text = """
@assoc[50] "Associativity" {
  category=associativity
}: (+ (+ ?x ?y) ?z) <=> (+ :x (+ :y :z))
"""
        rules = load_rules_from_dsl(text)
        assert len(rules) == 2  # fwd and rev
        for meta, _ in rules:
            assert meta.priority == 50
            assert meta.category == "associativity"

    def test_multi_line_compact_form_still_works(self):
        text = """
@r1 {category=cat}: (a ?x) => :x
"""
        rules = load_rules_from_dsl(text)
        meta, _ = rules[0]
        assert meta.category == "cat"

    def test_multi_line_with_braces_in_quoted_value(self):
        # Multi-line annotation with a `}` inside a quoted value should not
        # terminate the block early.
        text = '''
@r1 {
  category="has } inside"
}: (a ?x) => :x
'''
        rules = load_rules_from_dsl(text)
        meta, _ = rules[0]
        assert meta.category == "has } inside"

    def test_multi_line_with_other_rules(self):
        # Multi-line block doesn't interfere with surrounding rules.
        text = """
@r1: (a ?x) => :x
@r2 {
  category=cat
}: (b ?x) => :x
@r3: (c ?x) => :x
"""
        rules = load_rules_from_dsl(text)
        assert len(rules) == 3
        names = [m.name for m, _ in rules]
        assert names == ["r1", "r2", "r3"]
        # Only r2 has a category.
        cats = [m.category for m, _ in rules]
        assert cats == [None, "cat", None]

    def test_multi_line_with_apostrophe_via_double_quote(self):
        # The documented workaround for apostrophes: use double quotes.
        text = '''
@r1 {
  category="it's fine"
}: (a ?x) => :x
'''
        rules = load_rules_from_dsl(text)
        meta, _ = rules[0]
        assert meta.category == "it's fine"
