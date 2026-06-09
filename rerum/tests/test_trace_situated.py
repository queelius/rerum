"""Tests for the situated (self-contained) trace model: helpers, fields,
global reconstruction, path threading."""

import json

import pytest

from rerum import RuleEngine, E, RewriteStep, RewriteTrace
from rerum.trace import splice_at, rule_identity
from rerum.engine import RuleMetadata


class TestSpliceAt:
    """splice_at(root, path, subtree) replaces the subtree at a child path."""

    def test_empty_path_replaces_root(self):
        assert splice_at(["+", "a", "b"], [], ["*", "c", "d"]) == ["*", "c", "d"]

    def test_single_index(self):
        # path [1] addresses the first operand of (+ a b)
        assert splice_at(["+", "a", "b"], [1], "z") == ["+", "z", "b"]

    def test_nested_index(self):
        root = ["+", ["*", "a", "b"], "c"]
        # path [1, 2] addresses 'b' inside (* a b)
        assert splice_at(root, [1, 2], "Z") == ["+", ["*", "a", "Z"], "c"]

    def test_does_not_mutate_root(self):
        root = ["+", "a", "b"]
        out = splice_at(root, [1], "z")
        assert root == ["+", "a", "b"]
        assert out is not root


class TestRuleIdentity:
    """rule_identity prefers metadata.name, else hashes (pattern, skeleton)."""

    def test_named_rule_uses_name(self):
        meta = RuleMetadata(name="add-zero")
        assert rule_identity(meta, ["+", "?x", 0], ":x") == "add-zero"

    def test_anonymous_rule_uses_hash(self):
        meta = RuleMetadata(name=None)
        rid = rule_identity(meta, ["+", "?x", 0], ":x")
        assert rid.startswith("#")
        assert len(rid) == 13  # "#" + 12 hex chars

    def test_anonymous_hash_is_stable_and_content_addressed(self):
        meta = RuleMetadata(name=None)
        a = rule_identity(meta, ["+", "?x", 0], ":x")
        b = rule_identity(meta, ["+", "?x", 0], ":x")
        c = rule_identity(meta, ["*", "?x", 1], ":x")
        assert a == b
        assert a != c
