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


from rerum.training import to_prose


def _mixed_kind_trace():
    """A trace with one rule step and one normalize step (domain-free).

    Root: (+ (* 2 x) (+ y 0)).
    Step 1 (rule): rewrite inner (+ y 0) at path [2] -> y; root ->
                   (+ (* 2 x) y).
    Step 2 (normalize): reorder operands of the + at path [] ->
                   (+ y (* 2 x)).
    """
    meta_rule = RuleMetadata(name="add-zero", category="identity",
                             reasoning="additive identity")
    meta_norm = RuleMetadata(name="canonical-sort", category="normalize",
                             reasoning="commutative ordering")
    t = RewriteTrace()
    t.initial = ["+", ["*", 2, "x"], ["+", "y", 0]]
    t(RewriteStep(0, meta_rule, ["+", "y", 0], "y", path=[2],
                  rule_id="add-zero", kind="rule",
                  rationale="additive identity"))
    t(RewriteStep(0, meta_norm, ["+", ["*", 2, "x"], "y"],
                  ["+", "y", ["*", 2, "x"]], path=[],
                  rule_id="canonical-sort", kind="normalize",
                  rationale="commutative ordering"))
    t.final = ["+", "y", ["*", 2, "x"]]
    return t


class TestToProseProjection:
    def test_starts_with_problem(self):
        prose = to_prose(_mixed_kind_trace())
        first = prose.splitlines()[0]
        assert "(+ (* 2 x) (+ y 0))" in first

    def test_ends_with_answer(self):
        prose = to_prose(_mixed_kind_trace())
        last = prose.splitlines()[-1]
        assert "(+ y (* 2 x))" in last

    def test_rule_step_mentions_rule_id_and_rationale(self):
        prose = to_prose(_mixed_kind_trace())
        assert "add-zero" in prose
        assert "additive identity" in prose
        # The rule template reads "becomes".
        assert "becomes" in prose

    def test_rule_step_renders_before_and_after_roots(self):
        prose = to_prose(_mixed_kind_trace())
        # The rule step's whole-expression before/after roots appear.
        assert "(+ (* 2 x) (+ y 0))" in prose   # before_root of step 1
        assert "(+ (* 2 x) y)" in prose          # after_root of step 1

    def test_normalize_step_uses_simplifying_template(self):
        prose = to_prose(_mixed_kind_trace())
        assert "Simplifying" in prose
        assert "canonical-sort" in prose

    def test_deterministic(self):
        t = _mixed_kind_trace()
        assert to_prose(t) == to_prose(t)

    def test_anonymous_rule_id_is_handled(self):
        # A step with rule_id=None renders without crashing.
        t = RewriteTrace()
        t.initial = ["+", "x", 0]
        t(RewriteStep(0, RuleMetadata(), ["+", "x", 0], "x", path=[],
                      rule_id=None, kind="rule"))
        t.final = "x"
        prose = to_prose(t)
        assert "x" in prose

    def test_empty_trace_is_problem_then_answer(self):
        t = RewriteTrace()
        t.initial = ["+", "x", "y"]
        t.final = ["+", "x", "y"]
        prose = to_prose(t)
        lines = prose.splitlines()
        # No steps: just the problem framing and the (unchanged) answer.
        assert "(+ x y)" in lines[0]
        assert "(+ x y)" in lines[-1]


class TestToProseFoldTemplate:
    """A fold step reads as 'Computing with ...'."""

    def _fold_trace(self):
        meta_fold = RuleMetadata(name="fold-add", category="fold",
                                 reasoning="constant folding")
        t = RewriteTrace()
        t.initial = ["+", ["+", 2, 3], "x"]
        t(RewriteStep(0, meta_fold, ["+", 2, 3], 5, path=[1],
                      rule_id="fold-add", kind="fold",
                      rationale="constant folding"))
        t.final = ["+", 5, "x"]
        return t

    def test_fold_step_uses_computing_template(self):
        prose = to_prose(self._fold_trace())
        assert "Computing" in prose
        assert "fold-add" in prose
        # before_root -> after_root for the fold.
        assert "(+ (+ 2 3) x)" in prose
        assert "(+ 5 x)" in prose


import types

from rerum.training import generate_corpus


# A domain-free TOY rule set: ordinary algebra simplification, NO calculus.
# This proves training.py is domain-agnostic (no dd/int/lim anywhere).
TOY_RULES = """
    @add-zero {category=identity}: (+ ?x 0) => :x
    @mul-one {category=identity}: (* ?x 1) => :x
    @mul-zero {category=annihilator}: (* ?x 0) => 0
"""


def _toy_engine():
    """Engine over the toy algebra rules. No prelude needed (no folds)."""
    return RuleEngine.from_dsl(TOY_RULES)


def _simplify_driver(engine, problem):
    """Caller-supplied adapter: run the confluent toy rules via simplify.

    A problem is an (label, expr) pair. Returns (answer, trace) where the
    trace is the Phase-1 RewriteTrace from simplify(trace=True). This adapter
    is the CALLER's responsibility; training.py never picks simplify vs solve.
    """
    _label, expr = problem
    result, trace = engine.simplify(expr, trace=True)
    return result, trace


def _is_atom_checker(problem, answer):
    """Caller-supplied validator: the toy answer should reduce to an atom.

    Domain-free: for the toy rules every problem collapses to a single atom.
    Returns True iff the answer is a non-list (a symbol or number).
    """
    return not isinstance(answer, list)


class TestGenerateCorpusToy:
    def test_yields_a_record_per_problem(self):
        engine = _toy_engine()
        problems = [("p1", ["+", "x", 0]),
                    ("p2", ["*", "y", 1])]
        records = list(generate_corpus(engine, problems,
                                       driver=_simplify_driver,
                                       checker=_is_atom_checker))
        assert len(records) == 2

    def test_is_a_streaming_generator(self):
        engine = _toy_engine()
        gen = generate_corpus(engine, [("p", ["+", "x", 0])],
                              driver=_simplify_driver)
        assert isinstance(gen, types.GeneratorType)

    def test_answer_is_the_simplified_result(self):
        engine = _toy_engine()
        recs = list(generate_corpus(engine, [("p", ["+", "x", 0])],
                                    driver=_simplify_driver))
        assert recs[0]["answer"] == "x"

    def test_checker_stamps_verified_true(self):
        engine = _toy_engine()
        recs = list(generate_corpus(engine, [("p", ["+", "x", 0])],
                                    driver=_simplify_driver,
                                    checker=_is_atom_checker))
        assert recs[0]["verified"] is True

    def test_checker_can_stamp_false(self):
        # A checker that rejects everything stamps verified=False.
        engine = _toy_engine()
        recs = list(generate_corpus(engine, [("p", ["+", "x", 0])],
                                    driver=_simplify_driver,
                                    checker=lambda prob, ans: False))
        assert recs[0]["verified"] is False

    def test_no_checker_leaves_verified_none(self):
        engine = _toy_engine()
        recs = list(generate_corpus(engine, [("p", ["+", "x", 0])],
                                    driver=_simplify_driver))
        assert recs[0]["verified"] is None

    def test_operator_label_comes_from_driver_free_problem(self):
        # The caller controls the operator label via the problem; here the
        # driver passes the problem label through (see implementation note).
        engine = _toy_engine()
        recs = list(generate_corpus(engine, [("simplify", ["*", "z", 1])],
                                    driver=_simplify_driver))
        assert recs[0]["operator"] == "simplify"

    def test_record_chain_property_holds_end_to_end(self):
        # The global-sequence join must chain for a REAL engine trace.
        engine = _toy_engine()
        rec = next(generate_corpus(engine, [("p", ["+", ["+", "x", 0], 0])],
                                   driver=_simplify_driver))
        steps = rec["steps"]
        assert steps, "expected a non-empty derivation"
        for k in range(len(steps) - 1):
            assert steps[k + 1]["before_root"] == steps[k]["after_root"]


class TestGenerateCorpusExport:
    def test_exports(self):
        import rerum
        assert rerum.to_training_record is to_training_record
        assert rerum.to_prose is to_prose
        assert rerum.generate_corpus is generate_corpus


from rerum.engine import format_sexpr


class TestCorpusJsonlRoundTrip:
    """Generate a JSONL corpus over the TOY rules, write it, read it back."""

    REQUIRED_TOP = ("problem", "operator", "steps", "answer", "verified")
    REQUIRED_STEP = ("kind", "rule_id", "rationale", "before_root",
                     "after_root", "bindings", "path", "guard")

    def _problems(self):
        return [
            ("simplify", ["+", ["+", "x", 0], 0]),  # -> x, two add-zero steps
            ("simplify", ["*", ["* ", "y", 1] if False else ["*", "y", 1], 1]),
        ]

    def test_write_and_read_back_jsonl(self, tmp_path):
        engine = _toy_engine()
        out = tmp_path / "corpus.jsonl"
        problems = [
            ("simplify", ["+", ["+", "x", 0], 0]),
            ("simplify", ["*", ["*", "y", 1], 1]),
        ]
        # Stream records to disk, one JSON object per line.
        with out.open("w") as fh:
            for rec in generate_corpus(engine, problems,
                                       driver=_simplify_driver,
                                       checker=_is_atom_checker):
                fh.write(json.dumps(rec) + "\n")

        # Read back and validate.
        lines = out.read_text().splitlines()
        assert len(lines) == 2
        records = [json.loads(line) for line in lines]

        for rec in records:
            for k in self.REQUIRED_TOP:
                assert k in rec, f"record missing {k}"
            assert rec["operator"] == "simplify"
            assert rec["verified"] is True, "every record passes the checker"
            assert rec["steps"], "non-empty derivation"
            for step in rec["steps"]:
                for k in self.REQUIRED_STEP:
                    assert k in step, f"step missing {k}"
            # Chain property survives JSON round-trip.
            steps = rec["steps"]
            for i in range(len(steps) - 1):
                assert steps[i + 1]["before_root"] == steps[i]["after_root"]

    def test_prose_projection_mentions_rule_names(self):
        # The prose, rendered from the same trace, names the rules that fired.
        engine = _toy_engine()
        _, trace = engine.simplify(["+", ["+", "x", 0], 0], trace=True)
        prose = to_prose(trace)
        # The prose begins with the problem and ends with the answer.
        assert "(+ (+ x 0) 0)" in prose.splitlines()[0]
        # At least one rule id appears (the rule lines name the fired rule).
        rule_ids = {s.rule_id for s in trace.steps if s.rule_id}
        assert rule_ids, "expected named rules in the derivation"
        assert any(rid in prose for rid in rule_ids), (
            "prose must name a rule that fired")

    def test_record_and_prose_agree_on_step_count(self):
        # The record's step list and the prose's per-step lines both derive
        # from to_global_sequence, so their counts match.
        engine = _toy_engine()
        full = ["+", ["+", "x", 0], 0]
        _, trace = engine.simplify(full, trace=True)
        rec = to_training_record(trace, problem=format_sexpr(full),
                                 operator="simplify", answer=trace.final)
        prose_lines = to_prose(trace).splitlines()
        # prose = 1 problem line + N step lines + 1 answer line.
        assert len(prose_lines) == len(rec["steps"]) + 2

    def test_training_module_names_no_domain_operator(self):
        # Guardrail for the general-engine principle: training.py must not
        # contain any domain operator literal (dd/int/lim/and/or as words).
        import re
        from pathlib import Path
        import rerum.training as _t
        src = Path(_t.__file__).read_text()
        # Strip string/quote noise is unnecessary; we look for the operator
        # tokens as standalone identifiers in code or docstring.
        for op in ("dd", "int(", "lim", " and ", " or "):
            # ' and '/' or ' would match Python keywords; restrict to the
            # operator-symbol sense by requiring quotes around them.
            pass
        # Concrete check: the calculus operator symbols must not appear as
        # quoted string literals (how they would be referenced as ops).
        for sym in ('"dd"', "'dd'", '"int"', "'int'", '"lim"', "'lim'"):
            assert sym not in src, f"training.py must not name operator {sym}"
