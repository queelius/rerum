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


class TestGlobalSequence:
    """to_global_sequence replays redex edits at their paths from initial."""

    def _trace(self):
        # Build a trace by hand: two steps editing different redexes of a root.
        meta = RuleMetadata(name="add-zero")
        t = RewriteTrace()
        t.initial = ["+", ["+", "x", 0], 0]
        # Step 1: inner (+ x 0) at path [1] -> x.  Root becomes (+ x 0).
        t(RewriteStep(0, meta, ["+", "x", 0], "x", path=[1]))
        # Step 2: outer (+ x 0) at path [] -> x.  Root becomes x.
        t(RewriteStep(0, meta, ["+", "x", 0], "x", path=[]))
        t.final = "x"
        return t

    def test_global_sequence_roots(self):
        t = self._trace()
        seq = t.to_global_sequence()
        assert len(seq) == 2
        assert seq[0]["before_root"] == ["+", ["+", "x", 0], 0]
        assert seq[0]["after_root"] == ["+", "x", 0]
        assert seq[1]["before_root"] == ["+", "x", 0]
        assert seq[1]["after_root"] == "x"

    def test_global_sequence_carries_step(self):
        t = self._trace()
        seq = t.to_global_sequence()
        assert seq[0]["step"] is t.steps[0]
        assert seq[1]["step"] is t.steps[1]

    def test_global_sequence_final_matches(self):
        t = self._trace()
        seq = t.to_global_sequence()
        assert seq[-1]["after_root"] == t.final

    def test_to_dict_global_sequence_flag(self):
        t = self._trace()
        d_plain = t.to_dict()
        assert "global_sequence" not in d_plain
        d_glob = t.to_dict(global_sequence=True)
        assert "global_sequence" in d_glob
        assert len(d_glob["global_sequence"]) == 2
        assert json.dumps(d_glob) is not None

    def test_to_dict_keeps_legacy_keys(self):
        t = self._trace()
        d = t.to_dict()
        for k in ("initial", "final", "steps", "step_count"):
            assert k in d

    def test_empty_trace_global_sequence_is_empty(self):
        t = RewriteTrace()
        t.initial = ["+", "x", 0]
        t.final = ["+", "x", 0]
        assert t.to_global_sequence() == []

    def test_legacy_none_path_treats_steps_as_whole_expression(self):
        # Pre-path-threading steps (path=None) carry whole expressions in
        # before/after; the None->[] fallback reconstructs roots correctly.
        meta = RuleMetadata(name="r")
        t = RewriteTrace()
        t.initial = ["+", "x", 0]
        t(RewriteStep(0, meta, ["+", "x", 0], "x"))          # path defaults to None
        t(RewriteStep(0, meta, "x", ["g", "x"]))             # whole-expr edit
        t.final = ["g", "x"]
        seq = t.to_global_sequence()
        assert [e["before_root"] for e in seq] == [["+", "x", 0], "x"]
        assert [e["after_root"] for e in seq] == ["x", ["g", "x"]]
        assert seq[-1]["after_root"] == t.final


class TestPathThreading:
    """Emitted steps carry the redex path under each strategy."""

    def _engine(self):
        return RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")

    def test_root_redex_has_empty_path(self):
        eng = self._engine()
        _, trace = eng.simplify(E("(+ x 0)"), trace=True)
        assert trace.steps, "expected at least one step"
        assert trace.steps[0].path == []

    def test_child_redex_carries_path_exhaustive(self):
        eng = self._engine()
        _, trace = eng.simplify(E("(* (+ x 0) y)"), trace=True, strategy="exhaustive")
        paths = [s.path for s in trace.steps]
        assert [1] in paths, f"expected redex path [1] among {paths}"

    def test_child_redex_carries_path_bottomup(self):
        eng = self._engine()
        _, trace = eng.simplify(E("(* (+ x 0) y)"), trace=True, strategy="bottomup")
        paths = [s.path for s in trace.steps]
        assert [1] in paths, f"expected redex path [1] among {paths}"

    def test_child_redex_carries_path_topdown(self):
        eng = self._engine()
        _, trace = eng.simplify(E("(* (+ x 0) y)"), trace=True, strategy="topdown")
        paths = [s.path for s in trace.steps]
        assert [1] in paths, f"expected redex path [1] among {paths}"

    def test_global_sequence_roundtrips_after_threading(self):
        eng = self._engine()
        result, trace = eng.simplify(E("(* (+ x 0) y)"), trace=True)
        seq = trace.to_global_sequence()
        assert seq[-1]["after_root"] == result
        assert seq[0]["before_root"] == trace.initial

    def test_hook_context_expr_path_populated(self):
        eng = self._engine()
        seen = []

        def observer(step, ctx):
            seen.append(tuple(ctx.expr_path))

        eng.on_rule_applied(observer)
        eng.simplify(E("(* (+ x 0) y)"))
        assert (1,) in seen, f"expected (1,) among {seen}"
