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


class TestRewriteStepFields:
    """RewriteStep gains additive situated fields; legacy construction works."""

    def _meta(self):
        return RuleMetadata(name="add-zero", description="x+0=x",
                            reasoning="additive identity", category="identity")

    def test_legacy_positional_construction_still_works(self):
        step = RewriteStep(0, self._meta(), ["+", "x", 0], "x")
        assert step.before == ["+", "x", 0]
        assert step.after == "x"
        assert step.rule_id is None
        assert step.direction is None
        assert step.bindings is None
        assert step.path is None
        assert step.kind == "rule"
        assert step.guard is None
        assert step.rationale is None

    def test_redex_aliases(self):
        step = RewriteStep(0, self._meta(), ["+", "x", 0], "x")
        assert step.before_redex == step.before == ["+", "x", 0]
        assert step.after_redex == step.after == "x"

    def test_new_fields_round_trip(self):
        step = RewriteStep(
            0, self._meta(), ["+", "x", 0], "x",
            rule_id="add-zero", direction="fwd",
            bindings={"x": "x"}, path=[1],
            kind="rule", guard={"condition": ["true"], "result": True},
            rationale="additive identity",
        )
        assert step.rule_id == "add-zero"
        assert step.direction == "fwd"
        assert step.bindings == {"x": "x"}
        assert step.path == [1]
        assert step.guard == {"condition": ["true"], "result": True}
        assert step.rationale == "additive identity"

    def test_to_dict_keeps_legacy_keys(self):
        step = RewriteStep(0, self._meta(), ["+", "x", 0], "x")
        d = step.to_dict()
        for k in ("rule_index", "rule_name", "description", "before", "after"):
            assert k in d

    def test_to_dict_emits_all_situated_keys(self):
        step = RewriteStep(
            0, self._meta(), ["+", "x", 0], "x",
            rule_id="add-zero", direction="fwd", bindings={"x": "x"},
            path=[1], kind="rule",
            guard={"condition": ["true"], "result": True},
            rationale="additive identity",
        )
        d = step.to_dict()
        for k in ("rule_index", "rule_id", "rule_name", "direction",
                  "description", "kind", "before", "after", "path",
                  "bindings", "guard", "rationale"):
            assert k in d, f"missing key {k}"
        assert json.dumps(d) is not None

    def test_eq_against_expression_compares_after(self):
        step = RewriteStep(0, self._meta(), ["+", "x", 0], "x")
        assert step == "x"
        assert step != ["+", "x", 0]

    def test_eq_against_step_is_identity(self):
        m = self._meta()
        s1 = RewriteStep(0, m, ["+", "x", 0], "x")
        s2 = RewriteStep(0, m, ["+", "x", 0], "x")
        assert s1 == s1
        assert s1 != s2  # distinct objects
