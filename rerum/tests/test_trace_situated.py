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

    def test_once_strategy_with_trace_applies_once(self):
        # simplify(trace=True, strategy="once") must honor 'once' (one step),
        # not silently run exhaustive.  The only redex is (+ x 0) inside
        # (* (+ x 0) y); the engine should take exactly one step.
        eng = self._engine()
        out, trace = eng.simplify(E("(* (+ x 0) y)"), trace=True, strategy="once")
        assert len(trace.steps) == 1
        # Confirm one step was actually taken (not full exhaustive).
        assert out == ["*", "x", "y"]

    def test_unknown_strategy_with_trace_raises(self):
        eng = self._engine()
        with pytest.raises(ValueError):
            eng.simplify(E("(+ x 0)"), trace=True, strategy="sideways")

    def test_multistep_structure_change_running_root_contract(self):
        # Two rules; (+ ?x 0) => :x  and  (* ?x 1) => :x.
        # Exhaustive strategy on (* (+ x 0) 1) fires m1 first (matching the
        # whole expression where ?x = (+ x 0)), yielding (+ x 0), then fires
        # az on that, yielding x.  Both redexes are at the root (path=[]).
        # The running-root contract: seq[i].after_root == seq[i+1].before_root,
        # and seq[-1].after_root == final engine result.
        eng = RuleEngine.from_dsl("@az: (+ ?x 0) => :x\n@m1: (* ?x 1) => :x")
        result, trace = eng.simplify(E("(* (+ x 0) 1)"), trace=True)
        seq = trace.to_global_sequence()
        # Should be exactly 2 steps (m1 then az).
        assert len(seq) == 2
        # Step 1 turns (* (+ x 0) 1) into (+ x 0).
        assert seq[0]["after_root"] == ["+", "x", 0]
        # Step 2's before_root must be what step 1 produced (chain contract).
        assert seq[1]["before_root"] == seq[0]["after_root"]
        # Final result matches engine output.
        assert seq[-1]["after_root"] == result

    def test_deeply_nested_redex_path(self):
        eng = self._engine()
        # (f (g (+ x 0))): redex (+ x 0) is at list path [1, 1].
        _, trace = eng.simplify(E("(f (g (+ x 0)))"), trace=True)
        paths = [s.path for s in trace.steps]
        assert [1, 1] in paths, f"expected [1, 1] among {paths}"


class TestPopulatedFields:
    """Situated fields are populated at the emit sites during simplify."""

    def test_named_rule_id_and_rationale(self):
        eng = RuleEngine.from_dsl('@add-zero {category=identity}: (+ ?x 0) => :x')
        _, trace = eng.simplify(E("(+ x 0)"), trace=True)
        step = trace.steps[0]
        assert step.rule_id == "add-zero"
        assert step.kind == "rule"
        assert step.rationale == "identity"  # reasoning or category

    def test_bindings_captured(self):
        eng = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")
        _, trace = eng.simplify(E("(+ y 0)"), trace=True)
        step = trace.steps[0]
        assert step.bindings is not None
        assert step.bindings.get("x") == "y"

    def test_direction_for_bidirectional(self):
        eng = RuleEngine.from_dsl("@commute: (+ ?x ?y) <=> (+ :y :x)")
        _, trace = eng.simplify(E("(+ a b)"), trace=True, max_steps=1)
        assert trace.steps, "commute rule should fire at least once"
        assert trace.steps[0].direction in ("fwd", "rev")

    def test_unguarded_rule_has_none_guard(self):
        eng = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")
        _, trace = eng.simplify(E("(+ x 0)"), trace=True)
        assert trace.steps[0].guard is None

    def test_apply_once_populates_fields(self):
        eng = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")
        captured = []
        eng.on_rule_applied(lambda step, ctx: captured.append(step))
        result, meta = eng.apply_once(E("(+ x 0)"))
        assert result == "x"
        assert captured
        assert captured[0].rule_id == "add-zero"
        assert captured[0].bindings is not None


class TestGuardField:
    """A checked condition is recorded in step.guard."""

    def test_guard_dict_present(self):
        from rerum.rewriter import PREDICATE_PRELUDE
        eng = RuleEngine.from_dsl(
            "@drop-abs: (abs ?x) => :x when (! >= :x 0)",
            fold_funcs=PREDICATE_PRELUDE,
        )
        _, trace = eng.simplify(E("(abs 5)"), trace=True)
        assert trace.steps, "guarded rule should fire on (abs 5)"
        g = trace.steps[0].guard
        assert g is not None
        assert g["result"] is True
        assert "condition" in g


class TestOncePathRoundTrip:
    def test_once_trace_global_sequence_roundtrips(self):
        eng = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")
        # redex (+ x 0) is at child path [1] of (* (+ x 0) y)
        result, trace = eng.simplify(E("(* (+ x 0) y)"), trace=True, strategy="once")
        assert len(trace.steps) == 1
        assert trace.steps[0].path == [1]
        seq = trace.to_global_sequence()
        assert seq[-1]["after_root"] == result  # reconstructs (* x y), not "x"


class TestLabeledSingleRewrites:
    """_all_single_rewrites(labeled=True) returns (expr, label) edges."""

    def test_default_is_legacy_expr_list(self):
        eng = RuleEngine.from_dsl("@commute: (+ ?x ?y) <=> (+ :y :x)")
        outs = eng._all_single_rewrites(["+", "a", "b"])
        assert ["+", "b", "a"] in outs

    def test_labeled_returns_edges(self):
        eng = RuleEngine.from_dsl("@commute: (+ ?x ?y) <=> (+ :y :x)")
        edges = eng._all_single_rewrites(["+", "a", "b"], labeled=True)
        assert edges, "expected at least one labeled edge"
        expr, label = edges[0]
        assert "rule_id" in label
        assert "direction" in label
        assert "bindings" in label
        assert "path" in label

    def test_labeled_edge_records_path_for_child_redex(self):
        eng = RuleEngine.from_dsl("@commute: (+ ?x ?y) <=> (+ :y :x)")
        edges = eng._all_single_rewrites(["*", ["+", "a", "b"], "c"], labeled=True)
        target = ["*", ["+", "b", "a"], "c"]
        match = [lbl for ex, lbl in edges if ex == target]
        assert match, f"missing edge to {target}"
        assert match[0]["path"] == [1]

    def test_labeled_rule_id_is_named(self):
        eng = RuleEngine.from_dsl("@commute: (+ ?x ?y) <=> (+ :y :x)")
        edges = eng._all_single_rewrites(["+", "a", "b"], labeled=True)
        rule_ids = {lbl["rule_id"] for _, lbl in edges}
        assert any(rid.startswith("commute") for rid in rule_ids)
