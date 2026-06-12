"""Tests for goal-directed best-first search (solve)."""

import pytest

from rerum.engine import RuleEngine
from rerum.solve import SolveResult, contains_op, solve
from rerum.optimize import expr_size
from rerum.trace import RewriteTrace


class TestContainsOp:
    def test_atom_has_no_op(self):
        assert contains_op("x", {"foo"}) is False
        assert contains_op(42, {"foo"}) is False

    def test_top_level_op(self):
        assert contains_op(["foo", "x"], {"foo"}) is True

    def test_nested_op(self):
        assert contains_op(["+", "x", ["foo", "y"]], {"foo"}) is True

    def test_absent_op(self):
        assert contains_op(["+", "x", ["bar", "y"]], {"foo"}) is False

    def test_multiple_ops(self):
        assert contains_op(["aaa", "x", "x"], {"aaa", "bbb"}) is True
        assert contains_op(["bbb", "x", "x", 0], {"aaa", "bbb"}) is True


# The toy problem: a `foo` operator that rewrites to `+`, and `double` that
# rewrites to `foo`. The goal is "no `foo`/`double` op remains". `solve` must
# find the foo-free form and the derivation must replay to it. The toy
# operators are deliberately nonsense names so no domain leaks in.
#
# NOTE (deviation from plan): the plan's TOY_DSL wrote the `@double` skeleton
# as ``(foo ?x)``. In this codebase ``?x`` is a literal pattern marker in
# skeleton position (it emits ``["?", "x"]``); substitution is ``:x``. The
# correct skeleton that yields ``(foo x)`` is ``(foo :x)``, used here.
TOY_DSL = """
@unfoo: (foo ?x) => (+ :x :x)
@double: (double ?x) => (foo :x)
"""


def _toy_engine():
    from rerum import ARITHMETIC_PRELUDE
    eng = RuleEngine(fold_funcs=ARITHMETIC_PRELUDE)
    eng.load_dsl(TOY_DSL)
    return eng


class TestSolveToy:
    def test_finds_foo_free_form(self):
        eng = _toy_engine()
        goal = lambda e: not contains_op(e, {"foo", "double"})
        result = solve(eng, ["double", "x"], goal, max_nodes=200)
        assert result.found is True
        assert isinstance(result, SolveResult)
        # (double x) -> (foo x) -> (+ x x)
        assert result.solution == ["+", "x", "x"]
        assert goal(result.solution)

    def test_derivation_is_a_trace_that_replays(self):
        eng = _toy_engine()
        goal = lambda e: not contains_op(e, {"foo", "double"})
        result = solve(eng, ["double", "x"], goal, max_nodes=200)
        deriv = result.derivation
        assert isinstance(deriv, RewriteTrace)
        assert deriv.initial == ["double", "x"]
        assert deriv.final == result.solution
        # Replaying the step `after` fields from initial reaches solution.
        current = deriv.initial
        for step in deriv.steps:
            current = step.after
        assert current == result.solution
        # Each step names the rule that produced it.
        names = [s.metadata.name for s in deriv.steps]
        assert names == ["double", "unfoo"]

    def test_already_satisfied_returns_zero_step_trace(self):
        eng = _toy_engine()
        goal = lambda e: not contains_op(e, {"foo", "double"})
        result = solve(eng, ["+", "x", "x"], goal, max_nodes=50)
        assert result.found is True
        assert result.solution == ["+", "x", "x"]
        assert len(result.derivation.steps) == 0

    def test_budget_exhaustion_reports_not_found(self):
        eng = _toy_engine()
        # An impossible goal: no rule can ever make this true.
        impossible = lambda e: contains_op(e, {"never"})
        result = solve(eng, ["double", "x"], impossible, max_nodes=25)
        assert result.found is False
        assert result.solution is None
        assert result.explored <= 25

    def test_explored_count_is_positive_on_success(self):
        eng = _toy_engine()
        goal = lambda e: not contains_op(e, {"foo", "double"})
        result = solve(eng, ["double", "x"], goal, max_nodes=200)
        assert result.explored >= 1


class TestEngineSolveWrapper:
    def test_engine_method_delegates(self):
        eng = _toy_engine()
        goal = lambda e: not contains_op(e, {"foo", "double"})
        result = eng.solve(["double", "x"], goal, max_nodes=200)
        assert result.found is True
        assert result.solution == ["+", "x", "x"]


class TestTopLevelImports:
    def test_exports(self):
        import rerum
        assert rerum.solve is solve
        assert rerum.SolveResult is SolveResult
        assert rerum.contains_op is contains_op


def _engine(dsl):
    """Build a RuleEngine with the arithmetic prelude from a DSL string."""
    from rerum import ARITHMETIC_PRELUDE
    eng = RuleEngine(fold_funcs=ARITHMETIC_PRELUDE)
    eng.load_dsl(dsl)
    return eng


class TestSolveSemantics:
    """Lock in the search semantics the Done-When calls out: best-first with
    backtracking, tie safety, budget/max_depth firing, cycle termination, and
    theory-driven normalization between nodes. (Hardening tests added after the
    Task A adversarial review, which had probed these as throwaway cases.)"""

    def test_best_first_backtracks_past_cheap_dead_end(self):
        # start (size 1) has two moves: the CHEAP `trap` (size 1, a dead end)
        # and the COSTLIER `(mid junk junk)` (size 3) that leads to `goal`.
        # A greedy descent driver would commit to `trap` and stick; best-first
        # keeps the trap branch but backtracks once it dead-ends, then takes
        # the uphill move to reach the goal.
        eng = _engine(
            "@a: start => (mid junk junk)\n"
            "@b: (mid ?x ?y) => goal\n"
            "@c: start => trap\n"
        )
        goal = lambda e: e == "goal"
        result = solve(eng, "start", goal, max_nodes=200)
        assert result.found is True
        assert result.solution == "goal"
        names = [s.metadata.name for s in result.derivation.steps]
        assert names == ["a", "b"]  # via the costlier branch, not `trap`

    def test_equal_cost_frontier_does_not_crash(self):
        # A constant cost forces every frontier entry to tie. The counter
        # tiebreak must keep heap entries orderable; raw exprs are never
        # compared (that would raise TypeError on two lists/strings).
        eng = _engine(
            "@l: start => left\n@r: start => right\n"
            "@lg: left => goal\n@rg: right => goal\n"
        )
        goal = lambda e: e == "goal"
        result = solve(eng, "start", goal, cost_fn=lambda e: 0, max_nodes=200)
        assert result.found is True
        assert result.solution == "goal"

    def test_max_depth_hook_fires_once_on_non_goal_exit(self):
        # On any exit without reaching the goal, solve fires the engine's
        # max_depth hook exactly once with depth == explored.
        eng = _toy_engine()
        seen = []

        @eng.on_max_depth
        def resolver(expr, depth, ctx):
            seen.append(depth)
            return None  # decline

        impossible = lambda e: contains_op(e, {"never"})
        result = solve(eng, ["double", "x"], impossible, max_nodes=50)
        assert result.found is False
        assert result.solution is None
        assert len(seen) == 1
        assert seen[0] == result.explored

    def test_bidirectional_cycle_terminates(self):
        # a <=> b is a 2-node cycle. With an impossible goal and a large
        # budget, the visited set must bound exploration to {a, b} rather
        # than oscillating until the budget is spent.
        eng = _engine("@ab: a <=> b\n")
        impossible = lambda e: e == "never"
        result = solve(eng, "a", impossible, max_nodes=10000)
        assert result.found is False
        assert result.explored <= 3  # only a and b are reachable

    def test_rewrites_at_child_position_with_path_label(self):
        # The matching redex is nested: only the child `target` is rewritten,
        # exercising the child-position recursion in the edge generator.
        # CONTRACT (corrected): solve steps carry WHOLE search states as
        # before/after, so path is [] -- the original Phase 3 behavior
        # stamped the redex-local path [1] on whole-expression steps, which
        # made to_global_sequence splice the whole state INTO itself and
        # fabricate nonexistent intermediates.
        eng = _engine("@t: target => done\n")
        goal = lambda e: e == ["wrap", "done"]
        result = solve(eng, ["wrap", "target"], goal, max_nodes=50)
        assert result  # truthy iff found
        assert result.solution == ["wrap", "done"]
        step = result.derivation.steps[-1]
        assert step.path == []  # whole-state step: splices at the root
        assert step.before == ["wrap", "target"]
        assert step.after == ["wrap", "done"]

    def test_normalize_between_with_theory_canonicalizes_nodes(self):
        # Without a theory, normalize_between is a no-op: the produced node
        # (+ b a) never matches a goal stated as (+ a b). With an AC theory,
        # each node is canonicalized to (+ a b) and the goal is reached. The
        # theory is caller-supplied operator-signature DATA, not engine code.
        from rerum.normalize import Theory
        arith = Theory.from_dict({"+": {"ac": True, "identity": 0}})
        eng = _engine("@mk: start => (+ b a)\n")
        goal = lambda e: e == ["+", "a", "b"]
        assert solve(eng, "start", goal, max_nodes=50).found is False
        res = solve(eng, "start", goal, theory=arith, max_nodes=50)
        assert res.found is True
        assert res.solution == ["+", "a", "b"]


class TestSolveStepMetadataParity:
    """Solve-built steps must carry the same situated fields the engine's
    own emit sites stamp -- the corpus layer renders rationale into the
    chain-of-thought, and sidecar 'reasoning' exists precisely for that."""

    def test_solve_steps_carry_rationale_from_reasoning(self):
        import json
        from rerum.engine import RuleEngine
        from rerum.solve import contains_op, solve

        eng = RuleEngine.from_dsl("@collapse: (foo ?x) => :x")
        eng.load_metadata_json(json.dumps({
            "collapse": {"reasoning": "unwrap the foo shell",
                         "examples": [{"in": "(foo a)", "out": "a"}]}}))
        res = solve(eng, ["foo", "a"],
                    lambda e: not contains_op(e, {"foo"}))
        assert res.found is True
        steps = res.derivation.steps
        assert steps, "expected at least one step"
        assert steps[-1].rationale == "unwrap the foo shell"

    def test_solve_steps_fall_back_to_category(self):
        from rerum.engine import RuleEngine
        from rerum.solve import contains_op, solve

        eng = RuleEngine.from_dsl(
            "@collapse {category=structural}: (foo ?x) => :x")
        res = solve(eng, ["foo", "a"],
                    lambda e: not contains_op(e, {"foo"}))
        assert res.found is True
        assert res.derivation.steps[-1].rationale == "structural"

    def test_training_record_carries_solve_rationale(self):
        # The downstream payoff: a solve-driven corpus record names WHY.
        import json
        from rerum.engine import RuleEngine
        from rerum.solve import contains_op, solve
        from rerum.training import to_training_record

        eng = RuleEngine.from_dsl("@collapse: (foo ?x) => :x")
        eng.load_metadata_json(json.dumps({
            "collapse": {"reasoning": "unwrap the foo shell",
                         "examples": [{"in": "(foo a)", "out": "a"}]}}))
        res = solve(eng, ["foo", "a"],
                    lambda e: not contains_op(e, {"foo"}))
        record = to_training_record(
            res.derivation, problem="(foo a)", operator="foo", answer="a")
        rationales = [step.get("rationale") for step in record["steps"]]
        assert "unwrap the foo shell" in rationales


class TestSubPositionDerivationContract:
    """Regression for the path/before-after contract violation: solve steps
    carry WHOLE expressions, so path must be [] -- a redex-local path made
    to_global_sequence fabricate states like (top (top (bar a))) whenever a
    rule fired below the root. All prior tests used root-level rewrites."""

    def test_sub_root_rewrite_global_sequence_is_truthful(self):
        from rerum.engine import RuleEngine
        from rerum.solve import contains_op, solve

        eng = RuleEngine.from_dsl("@inner: (foo ?x) => (bar :x)")
        # The rewrite fires INSIDE (top ...): sub-root position.
        res = solve(eng, ["top", ["foo", "a"]],
                    lambda e: not contains_op(e, {"foo"}))
        assert res.found is True
        assert res.solution == ["top", ["bar", "a"]]
        seq = res.derivation.to_global_sequence()
        assert seq[0]["before_root"] == ["top", ["foo", "a"]]
        assert seq[-1]["after_root"] == ["top", ["bar", "a"]]
        for k in range(len(seq) - 1):
            assert seq[k]["after_root"] == seq[k + 1]["before_root"]

    def test_multi_level_chain(self):
        from rerum.engine import RuleEngine
        from rerum.solve import contains_op, solve

        eng = RuleEngine.from_dsl(
            "@a: (foo ?x) => (bar :x)\n@b: (bar ?x) => :x")
        res = solve(eng, ["wrap", ["wrap", ["foo", "c"]]],
                    lambda e: not contains_op(e, {"foo", "bar"}))
        assert res.found is True
        seq = res.derivation.to_global_sequence()
        assert seq[-1]["after_root"] == res.solution
        for k in range(len(seq) - 1):
            assert seq[k]["after_root"] == seq[k + 1]["before_root"]
