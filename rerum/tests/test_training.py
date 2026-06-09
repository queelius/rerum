"""Tests for the trace-to-text and trace-to-record layer (rerum/training.py).

to_training_record / to_prose are projections of a single RewriteTrace;
generate_corpus drives the engine via a CALLER-SUPPLIED driver and stamps
verification via a CALLER-SUPPLIED checker. training.py names no domain
operator: the end-to-end tasks run the engine on a domain-free TOY algebra
rule set so the records under test are genuine AND domain-agnostic.
"""

import json

import pytest

from rerum import RuleEngine, E, RewriteStep, RewriteTrace
from rerum.engine import RuleMetadata
from rerum.training import to_training_record


def _hand_trace():
    """A two-step trace built by hand, editing two redexes of one root.

    Root: (+ (+ x 0) 0).
    Step 1 rewrites the inner (+ x 0) at path [1] -> x; root -> (+ x 0).
    Step 2 rewrites the outer (+ x 0) at path [] -> x; root -> x.
    The global-sequence join must chain: after_root of step 1 equals
    before_root of step 2.
    """
    meta = RuleMetadata(name="add-zero", category="identity",
                        reasoning="additive identity")
    t = RewriteTrace()
    t.initial = ["+", ["+", "x", 0], 0]
    t(RewriteStep(0, meta, ["+", "x", 0], "x", path=[1],
                  rule_id="add-zero", kind="rule",
                  bindings={"x": "x"}, rationale="additive identity"))
    t(RewriteStep(0, meta, ["+", "x", 0], "x", path=[],
                  rule_id="add-zero", kind="rule",
                  bindings={"x": "x"}, rationale="additive identity"))
    t.final = "x"
    return t


class TestToTrainingRecordSchema:
    def test_top_level_keys(self):
        rec = to_training_record(_hand_trace(), problem="(+ (+ x 0) 0)",
                                 operator="simplify", answer="x")
        for k in ("problem", "operator", "steps", "answer", "verified"):
            assert k in rec, f"missing top-level key {k}"

    def test_operator_and_problem_and_answer_passthrough(self):
        rec = to_training_record(_hand_trace(), problem="(+ (+ x 0) 0)",
                                 operator="simplify", answer="x")
        assert rec["operator"] == "simplify"
        assert rec["problem"] == "(+ (+ x 0) 0)"
        assert rec["answer"] == "x"

    def test_operator_is_free_form_label(self):
        # operator is stored verbatim and never interpreted by training.py.
        rec = to_training_record(_hand_trace(), problem="p",
                                 operator="anything-goes", answer="x")
        assert rec["operator"] == "anything-goes"

    def test_verified_defaults_to_none(self):
        rec = to_training_record(_hand_trace(), problem="p",
                                 operator="simplify", answer="x")
        assert rec["verified"] is None

    def test_verified_passthrough(self):
        rec = to_training_record(_hand_trace(), problem="p",
                                 operator="simplify", answer="x", verified=True)
        assert rec["verified"] is True

    def test_step_keys(self):
        rec = to_training_record(_hand_trace(), problem="p",
                                 operator="simplify", answer="x")
        assert len(rec["steps"]) == 2
        for step in rec["steps"]:
            for k in ("kind", "rule_id", "rationale", "before_root",
                      "after_root", "bindings", "path", "guard"):
                assert k in step, f"missing step key {k}"

    def test_step_field_values(self):
        rec = to_training_record(_hand_trace(), problem="p",
                                 operator="simplify", answer="x")
        s0 = rec["steps"][0]
        assert s0["kind"] == "rule"
        assert s0["rule_id"] == "add-zero"
        assert s0["rationale"] == "additive identity"
        assert s0["bindings"] == {"x": "x"}
        assert s0["path"] == [1]
        assert s0["guard"] is None


class TestGlobalSequenceJoin:
    """before_root/after_root come from to_global_sequence and chain."""

    def test_first_before_root_is_initial(self):
        rec = to_training_record(_hand_trace(), problem="p",
                                 operator="simplify", answer="x")
        assert rec["steps"][0]["before_root"] == ["+", ["+", "x", 0], 0]

    def test_last_after_root_is_final(self):
        rec = to_training_record(_hand_trace(), problem="p",
                                 operator="simplify", answer="x")
        assert rec["steps"][-1]["after_root"] == "x"

    def test_adjacent_steps_chain(self):
        rec = to_training_record(_hand_trace(), problem="p",
                                 operator="simplify", answer="x")
        steps = rec["steps"]
        for k in range(len(steps) - 1):
            assert steps[k + 1]["before_root"] == steps[k]["after_root"], (
                f"join broken between step {k} and {k + 1}")

    def test_intermediate_root(self):
        rec = to_training_record(_hand_trace(), problem="p",
                                 operator="simplify", answer="x")
        # After collapsing the inner (+ x 0), the root is (+ x 0).
        assert rec["steps"][0]["after_root"] == ["+", "x", 0]
        assert rec["steps"][1]["before_root"] == ["+", "x", 0]

    def test_record_is_json_serializable(self):
        rec = to_training_record(_hand_trace(), problem="p",
                                 operator="simplify", answer="x", verified=False)
        assert json.dumps(rec) is not None
