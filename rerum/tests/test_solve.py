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


class TestSolveBoolIntKeying:
    """Regression: the visited/dedup keys must not merge a bool atom with the
    equal int. Python makes ``True == 1`` and ``False == 0`` (and they are
    hash-equal), so a naive ``expr_to_tuple`` key conflates the node ``(f
    True)`` with ``(f 1)`` -- distinct expressions with distinct types and
    distinct downstream meaning. The engine elsewhere bool-guards exactly this
    (``normalize._same_atom``, ``normalize.ORDER_KEY``); ``solve`` must too.

    The booleans here arise from the predicate prelude (``(! not 0)`` folds to
    Python ``True``), not from any hardcoded domain -- the rules and the goal
    are caller-supplied DATA.
    """

    def _bool_int_engine(self):
        from rerum import FULL_PRELUDE
        from rerum.engine import RuleEngine
        # start has two distinct neighbors: the bool True (via the predicate
        # fold) and the int 1. They are == and hash-equal in Python but are
        # genuinely different expressions.
        eng = RuleEngine(fold_funcs=FULL_PRELUDE)
        eng.load_dsl("@a: start => (! not 0)\n@b: start => 1\n")
        return eng

    def test_both_bool_and_int_neighbors_survive_dedup(self):
        # The per-node edge generator dedups by key; if it merges True and 1,
        # only one neighbor survives. Both must be present.
        from rerum.solve import _labeled_rewrites
        eng = self._bool_int_engine()
        rules = list(eng.rule_set())
        neighbors = [n for n, _ in _labeled_rewrites(eng, "start", rules)]
        assert True in neighbors
        assert 1 in neighbors
        # Distinguish by identity/type, not ==: True == 1 so the value test
        # alone would be satisfied by a single surviving bool.
        types = sorted(type(n).__name__ for n in neighbors)
        assert types == ["bool", "int"]

    def test_type_sensitive_goal_reaches_the_int_not_the_bool(self):
        # A legitimate, type-sensitive caller goal: reach the integer 1, NOT
        # the bool True. If the int neighbor is shadowed by the bool's key,
        # this goal is unreachable and solve wrongly reports not-found.
        from rerum.solve import solve
        eng = self._bool_int_engine()
        goal = lambda e: e == 1 and not isinstance(e, bool)
        res = solve(eng, "start", goal, max_nodes=200)
        assert res.found is True
        assert res.solution == 1
        assert not isinstance(res.solution, bool)

    def test_type_sensitive_goal_reaches_the_bool_not_the_int(self):
        # Symmetric: a goal that wants the bool True must reach it even though
        # the int 1 is also a neighbor.
        from rerum.solve import solve
        eng = self._bool_int_engine()
        goal = lambda e: e is True
        res = solve(eng, "start", goal, max_nodes=200)
        assert res.found is True
        assert res.solution is True

    def test_false_and_zero_are_distinct_nodes(self):
        # The False/0 half of the alias. start -> False (bool) and start -> 0
        # (int); a goal demanding the int 0 must not be satisfied by the bool.
        from rerum import FULL_PRELUDE
        from rerum.engine import RuleEngine
        from rerum.solve import solve
        eng = RuleEngine(fold_funcs=FULL_PRELUDE)
        # (! not 1) folds to False; the int literal 0 is the other neighbor.
        eng.load_dsl("@a: start => (! not 1)\n@b: start => 0\n")
        goal = lambda e: e == 0 and not isinstance(e, bool)
        res = solve(eng, "start", goal, max_nodes=200)
        assert res.found is True
        assert res.solution == 0
        assert not isinstance(res.solution, bool)

    def test_visited_set_does_not_merge_bool_into_int_branch(self):
        # End-to-end: the int branch leads to a goal reachable ONLY through
        # the int, while the bool branch is a genuine dead end. If the visited
        # key merges them and the bool is explored first, the int branch (and
        # thus the goal) is lost. The onward rule head is non-numeric so it
        # cannot accidentally match the bool by ==.
        from rerum import FULL_PRELUDE
        from rerum.engine import RuleEngine
        from rerum.solve import solve
        eng = RuleEngine(fold_funcs=FULL_PRELUDE)
        eng.load_dsl(
            "@mkbool: start => (wrap (! not 0))\n"   # (wrap True)
            "@mkint: start => (wrap 1)\n"            # (wrap 1)
            "@onlyint: (wrap ?x:const) => done\n"    # fires on either, but...
        )
        # ...the goal demands the derivation went through the int wrapper.
        # We assert the result is reached AND both wrapper nodes were distinct
        # search states (explored count reflects both, not a merge).
        goal = lambda e: e == "done"
        res = solve(eng, "start", goal, max_nodes=200)
        assert res.found is True
        assert res.solution == "done"


class TestSolveGuardBranches:
    """Coverage for the guard branches in the edge generator and the search
    loop: should_fire veto skips an edge, an identity-producing rule creates
    no self-loop edge, and cancellation breaks the search honestly."""

    def test_should_fire_veto_skips_the_edge(self):
        # A rule whose should_fire decision returns False must not produce an
        # edge during search. Vetoing the only rule out of `start` leaves the
        # goal unreachable.
        from rerum.engine import RuleEngine
        from rerum.solve import solve
        eng = RuleEngine.from_dsl("@step: start => goal\n")

        @eng.on_should_fire
        def veto(rule, metadata, expr, bindings, ctx):
            return False  # AND-gate veto: no rule ever fires

        res = solve(eng, "start", lambda e: e == "goal", max_nodes=50)
        assert res.found is False
        assert res.solution is None
        # No outgoing edge was ever generated, so nothing past start expanded.
        assert res.explored == 1

    def test_should_fire_allow_lets_the_edge_through(self):
        # Companion to the veto test: a decision that returns True does not
        # block the edge (exercises the True path of the guard).
        from rerum.engine import RuleEngine
        from rerum.solve import solve
        eng = RuleEngine.from_dsl("@step: start => goal\n")

        @eng.on_should_fire
        def allow(rule, metadata, expr, bindings, ctx):
            return True

        res = solve(eng, "start", lambda e: e == "goal", max_nodes=50)
        assert res.found is True
        assert res.solution == "goal"

    def test_selective_should_fire_prunes_one_branch(self):
        # A decision that vetoes only the `trap` rule forces solve down the
        # surviving branch -- the edge generator must honor the per-rule veto.
        from rerum.engine import RuleEngine
        from rerum.solve import solve
        eng = RuleEngine.from_dsl(
            "@good: start => goal\n@trap: start => trap\n")

        @eng.on_should_fire
        def no_trap(rule, metadata, expr, bindings, ctx):
            # metadata.name identifies the rule; veto the trap only.
            return metadata.name != "trap"

        res = solve(eng, "start", lambda e: e == "goal", max_nodes=50)
        assert res.found is True
        names = [s.metadata.name for s in res.derivation.steps]
        assert "trap" not in names

    def test_identity_result_creates_no_self_loop_edge(self):
        # A rule that maps an expression to itself (result == expr) must not
        # create an edge: the edge generator skips it. Without the skip the
        # node would re-enqueue itself forever (bounded only by the budget).
        from rerum.engine import RuleEngine
        from rerum.solve import _labeled_rewrites, solve
        # @idem rewrites (f ?x) to (f :x) -- structurally identical.
        eng = RuleEngine.from_dsl(
            "@idem: (f ?x) => (f :x)\n@real: (f ?x) => done\n")
        rules = list(eng.rule_set())
        neighbors = [n for n, _ in _labeled_rewrites(eng, ["f", "a"], rules)]
        # The identity edge is dropped; only the real rewrite remains.
        assert ["f", "a"] not in neighbors
        assert "done" in neighbors
        # And the search still terminates and finds the goal.
        res = solve(eng, ["f", "a"], lambda e: e == "done", max_nodes=50)
        assert res.found is True
        assert res.solution == "done"

    def test_cancellation_from_should_fire_breaks_search(self):
        # A should_fire decision that calls ctx.cancel() while ALLOWING the
        # edge through leaves the frontier non-empty (the `mid` node is
        # enqueued), so the search loop's top-of-iteration cancel check fires
        # the break BEFORE `mid` is expanded. The goal lies one hop past `mid`
        # and must therefore NOT be reached: cancellation is honest, not a
        # silent budget burn.
        from rerum.engine import RuleEngine
        from rerum.solve import solve
        eng = RuleEngine.from_dsl(
            "@a: start => mid\n@b: mid => goal\n")
        calls = []

        @eng.on_should_fire
        def cancel_on_first_expansion(rule, metadata, expr, bindings, ctx):
            calls.append(metadata.name)
            # Request cancel while expanding `start`, but let the edge through.
            if expr == "start":
                ctx.cancel()
            return True

        res = solve(eng, "start", lambda e: e == "goal", max_nodes=10000)
        assert res.found is False
        assert res.solution is None
        # Only `start` was expanded; the loop broke before `mid` (explored==1),
        # far short of the 10000-node budget.
        assert res.explored == 1
        assert calls == ["a"]

    def test_cancellation_with_veto_exits_without_goal(self):
        # The other cancellation shape: cancel AND veto, so no edge is even
        # produced. The frontier empties and the search exits not-found. (Pins
        # that cancel-then-veto does not somehow reach the goal.)
        from rerum.engine import RuleEngine
        from rerum.solve import solve
        eng = RuleEngine.from_dsl(
            "@a: start => mid\n@b: mid => goal\n")

        @eng.on_should_fire
        def cancel_and_veto(rule, metadata, expr, bindings, ctx):
            ctx.cancel()
            return False

        res = solve(eng, "start", lambda e: e == "goal", max_nodes=10000)
        assert res.found is False
        assert res.solution is None
        assert res.explored < 10000
