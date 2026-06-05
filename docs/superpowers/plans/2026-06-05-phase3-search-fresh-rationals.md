# Phase 3: Search, Fresh Variables, Rationals, and Numeval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Add the four general engine capabilities the calculus demonstration layers depend on: (A) a goal-directed best-first search `solve()` over labeled single-step rewrites (the ESCALATION driver above directed `simplify`, used only for non-confluent rule sets), (B) deterministic fresh-variable generation via a `["fresh", base]` skeleton form, (C) exact rational arithmetic so division and folds keep `fractions.Fraction` exact instead of silently floating, and (D) a GENERAL numeric evaluator `numeval`/`numeric_equiv` that interprets a ground term under a prelude. All four are domain-agnostic machinery parameterized by data: `solve`'s goal is a caller-supplied predicate, `numeval`'s operators come entirely from the supplied prelude, and no module names or special-cases any domain operator (`dd`/`int`/`lim`/`and`/`or`).

**Architecture:** A new pure-ish search module `rerum/solve.py` consumes labeled single-step rewrite edges. To stay decoupled from Phase 1 (whose changes to `_all_single_rewrites` may or may not have landed), `solve.py` builds its own labeled edges directly from `engine.rule_set(...)`, `_match_internal`, and `instantiate` (the exact ingredients `_all_single_rewrites` already uses). It produces a `RewriteTrace` derivation by stamping `RewriteStep`s with the edge label. It stays decoupled from Phase 2 (`normalize.py`) by guarding the import and treating normalization as an optional best-effort pass controlled by `normalize_between`. `solve` is the escalation driver: a confluent rule set is solved by directed `simplify`; `solve` exists for the non-confluent case where solving requires trying a move and backing out, so it is best-first with backtracking and a node budget. The goal is supplied by the caller as a predicate, and `contains_op` is a generic, operator-agnostic helper for building "no operator X remains" goals. Fresh variables and rationals are pure additions in `rewriter.py`/`expr.py`: a new `["fresh", base]` branch in `instantiate`, helpers `gensym`/`free_symbols`, a central `coerce_number`, and `Fraction`-aware `safe_div`/`nary_fold`/`format_sexpr` plus renarrowing in `instantiate`/`try_constant_fold`. A new module `rerum/numeval.py` provides the general numeric primitive: `numeval` recursively evaluates a ground term using only the supplied prelude's fold functions (no operator special-cased beyond what the prelude provides), and `numeric_equiv` samples variable assignments and compares two expressions numerically, skipping points where either raises a domain error.

**Tech Stack:** Python 3.9+, stdlib only (`fractions.Fraction`, `heapq`, `inspect`, `random`), pytest. One test file per feature area under `rerum/tests/`.

---

## File Structure

New files:
- `rerum/solve.py` - `SolveResult`, `contains_op`, `solve`.
- `rerum/numeval.py` - `numeval`, `numeric_equiv`.
- `rerum/tests/test_solve.py` - toy-rule-set search tests.
- `rerum/tests/test_rationals.py` - exact-rational fold/format tests.
- `rerum/tests/test_fresh.py` - fresh-variable determinism tests.
- `rerum/tests/test_numeval.py` - numeric evaluation and equivalence tests.

Extended files:
- `rerum/rewriter.py` - `coerce_number`, `gensym`, `free_symbols`, `["fresh", base]` in `instantiate`, `Fraction` in `safe_div`/`nary_fold`, renarrowing in `instantiate`/`try_constant_fold`.
- `rerum/expr.py` - `format_sexpr(Fraction(p,q)) -> "(/ p q)"`.
- `rerum/engine.py` - `RuleEngine.solve(expr, goal_predicate, **kw)` wrapper.
- `rerum/__init__.py` - export `solve`, `SolveResult`, `contains_op`, `coerce_number`, `gensym`, `free_symbols`, `numeval`, `numeric_equiv`.

Notes on phase ordering and decoupling:
- `solve.py` does NOT call `_all_single_rewrites`. It defines a private
  `_labeled_rewrites(engine, expr, rules)` generator that mirrors the
  `_all_single_rewrites` traversal but yields
  `(neighbor_expr, label)` where `label = {"rule_id", "direction",
  "bindings", "path"}`. This is the "reuse Phase 1 labeled edges"
  requirement satisfied by reusing the same primitives Phase 1 labels,
  without depending on Phase 1's not-yet-final return type.
- `RewriteStep` is assumed to already carry the situated keyword fields
  from Phase 1 (`rule_id`, `direction`, `bindings`, `path`, `kind`).
  `solve.py` constructs steps through a small `_make_step` adapter that
  passes those kwargs only if the running `RewriteStep.__init__` accepts
  them (probed once via `inspect.signature`), so the module still imports
  and runs if Phase 1 has not landed yet. This keeps Phase 3 shippable
  in isolation.
- Phase 2 `normalize` is imported inside a `try/except ImportError`. When
  absent or when `normalize_between=False`, nodes are used as-is.
- `numeval.py` is fully general: it interprets a ground term using ONLY
  the fold functions in the supplied prelude. It special-cases no
  operator. Domain validators (such as "is this the derivative of that?")
  are NOT in this module; they are example content under `examples/`
  (domain phases D1/D2) that CALL `numeval`/`numeric_equiv`.

The general-engine swap test (Section 0 of the spec) applies to every file
touched here: replace "calculus" with "boolean algebra" and no engine code
changes. None of `solve.py`, `numeval.py`, the fresh-var form, or the
rational helpers names any domain operator.

---

## Task A: Goal-directed best-first search (`rerum/solve.py`)

`solve` is the ESCALATION driver. Directed `simplify` solves confluent rule
sets; `solve` exists for the non-confluent case where solving requires
trying a move and backing out. The goal is a caller-supplied predicate, and
`contains_op` is a generic, operator-agnostic helper for building goals.

**Files:** `rerum/solve.py`, `rerum/tests/test_solve.py`, `rerum/engine.py`, `rerum/__init__.py`

- [ ] **Step A1: Write a failing test for `contains_op`.**

  Create `rerum/tests/test_solve.py` with:

  ```python
  """Tests for goal-directed best-first search (solve)."""

  import pytest

  from rerum.engine import RuleEngine
  from rerum.solve import SolveResult, contains_op, solve
  from rerum.optimize import expr_size


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
  ```

  Run: `pytest rerum/tests/test_solve.py::TestContainsOp -q`
  Expected: FAIL (ModuleNotFoundError: No module named 'rerum.solve').

- [ ] **Step A2: Implement `contains_op` in a new `rerum/solve.py`.**

  Create `rerum/solve.py`:

  ```python
  """Goal-directed best-first search over the rewrite graph.

  `solve` is the ESCALATION driver above directed `simplify`. A confluent
  rule set is solved greedily by `simplify`; `solve` exists for the
  non-confluent case, where solving requires trying a move and backing out
  of dead ends. It generalizes the bidirectional BFS in `engine.prove_equal`
  into a single-source best-first search: expand labeled single-step
  rewrites, ordered by a cost function, until a CALLER-SUPPLIED goal
  predicate holds or a node budget is spent. The labeled derivation (a
  `RewriteTrace`) is the solution path.

  General-engine principle: `solve` knows no domain. The goal is the
  caller's predicate; `contains_op` is a generic helper for building
  "no operator X remains" goals and is not tied to any operator.

  Decoupling:
  - Labeled edges are built here from the engine's `rule_set`,
    `_match_internal`, and `instantiate` primitives (the same ingredients
    `_all_single_rewrites` uses), so this module does not depend on the
    exact return type of `engine._all_single_rewrites` from Phase 1.
  - Phase 2 `normalize` is optional: imported defensively and only used
    when available and `normalize_between=True`.
  """

  import heapq
  import inspect
  from typing import Callable, List, Optional, Set, Tuple

  from .rewriter import ExprType, instantiate
  from .optimize import expr_size
  from .trace import RewriteStep, RewriteTrace

  try:  # Phase 2 may not have landed; normalization is best-effort.
      from .normalize import normalize as _normalize  # type: ignore
  except Exception:  # pragma: no cover - exercised when normalize.py absent
      _normalize = None


  def contains_op(expr: ExprType, ops: Set[str]) -> bool:
      """True if any compound node in ``expr`` has a head operator in ``ops``.

      A generic, operator-agnostic helper for building goal predicates of the
      form "no operator in ``ops`` remains". Knows no domain; ``ops`` is the
      caller's set of operator symbols.
      """
      if isinstance(expr, list):
          if expr and isinstance(expr[0], str) and expr[0] in ops:
              return True
          return any(contains_op(child, ops) for child in expr)
      return False
  ```

  Run: `pytest rerum/tests/test_solve.py::TestContainsOp -q`
  Expected: PASS.

- [ ] **Step A3: Commit.**

  ```bash
  git add rerum/solve.py rerum/tests/test_solve.py
  git commit -m "feat(solve): contains_op generic goal-builder helper"
  ```

- [ ] **Step A4: Write a failing test for `solve` on a toy rule set.**

  Append to `rerum/tests/test_solve.py`. The toy problem: a `foo` operator
  that rewrites to `+`, and the goal is "no `foo` op remains". `solve`
  must find the foo-free form and the derivation must replay to it. The
  toy operators are deliberately nonsense names so no domain leaks in.

  ```python
  TOY_DSL = """
  @unfoo: (foo ?x) => (+ :x :x)
  @double: (double ?x) => (foo ?x)
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
  ```

  Run: `pytest rerum/tests/test_solve.py::TestSolveToy -q`
  Expected: FAIL (ImportError: cannot import name 'SolveResult' / 'solve').

- [ ] **Step A5: Implement `SolveResult`, `_labeled_rewrites`, `_make_step`, and `solve`.**

  Append to `rerum/solve.py`:

  ```python
  class SolveResult:
      """Outcome of a `solve` search.

      Attributes:
          solution: The goal-satisfying expression, or None if not found.
          derivation: A `RewriteTrace` from the start expression to the
              solution (empty steps when the start already satisfies the
              goal). When not found, a trace whose `initial`/`final` are the
              start expression and whose steps are empty.
          explored: Number of nodes expanded (popped from the frontier).
          found: True iff a goal-satisfying node was reached within budget.

      Truthy iff `found`.
      """

      __slots__ = ("solution", "derivation", "explored", "found")

      def __init__(
          self,
          solution: Optional[ExprType],
          derivation: RewriteTrace,
          explored: int,
          found: bool,
      ):
          self.solution = solution
          self.derivation = derivation
          self.explored = explored
          self.found = found

      def __bool__(self) -> bool:
          return self.found

      def __repr__(self) -> str:
          from .expr import format_sexpr
          sol = format_sexpr(self.solution) if self.found else "<none>"
          return (
              f"SolveResult(found={self.found}, solution={sol}, "
              f"explored={self.explored}, steps={len(self.derivation.steps)})"
          )


  # Probe once: does the running RewriteStep accept the Phase-1 situated
  # keyword fields? Keeps this module importable before Phase 1 lands.
  _STEP_PARAMS = set(inspect.signature(RewriteStep.__init__).parameters)


  def _make_step(metadata, before, after, label) -> RewriteStep:
      """Build a RewriteStep, attaching situated label fields when supported."""
      kwargs = {}
      if "rule_id" in _STEP_PARAMS:
          kwargs["rule_id"] = label.get("rule_id")
      if "direction" in _STEP_PARAMS:
          kwargs["direction"] = label.get("direction")
      if "bindings" in _STEP_PARAMS:
          kwargs["bindings"] = label.get("bindings")
      if "path" in _STEP_PARAMS:
          kwargs["path"] = label.get("path")
      return RewriteStep(
          rule_index=label.get("rule_index", -1),
          metadata=metadata,
          before=before,
          after=after,
          **kwargs,
      )


  def _rule_identity(metadata, rule) -> str:
      """Stable rule id: name if present, else a content hash of (pat, skel).

      Prefers the Phase-1 `rule_identity` helper from trace.py when present;
      falls back to a local definition so this module works standalone.
      """
      try:
          from .trace import rule_identity  # Phase 1 helper, if present.
          return rule_identity(metadata, rule[0], rule[1])
      except Exception:
          import hashlib
          if getattr(metadata, "name", None):
              return metadata.name
          from .expr import format_sexpr
          payload = f"({format_sexpr(rule[0])})({format_sexpr(rule[1])})"
          return "#" + hashlib.sha1(payload.encode()).hexdigest()[:12]


  def _labeled_rewrites(engine, expr, rules, _path=None):
      """Yield (neighbor_expr, label) for every one-step rewrite of ``expr``.

      Mirrors `engine._all_single_rewrites` traversal (top-level rules then
      each child position), but stamps a label
      `{"rule_index","rule_id","direction","bindings","path"}` on each edge.
      Deduplicates by the resulting expression so the search frontier is not
      flooded with structurally identical neighbors.
      """
      from .engine import _match_internal, _expr_to_tuple

      if _path is None:
          _path = []
      seen = set()

      def emit(new_expr, label):
          key = _expr_to_tuple(new_expr)
          if key in seen:
              return
          seen.add(key)
          return (new_expr, label)

      # Top-level rule applications.
      for rule_idx, rule, metadata in rules:
          pattern, skeleton = rule
          bindings = _match_internal(pattern, expr)
          if bindings is None:
              continue
          if not engine._check_condition(metadata.condition, bindings):
              continue
          if not engine._check_should_fire(rule, metadata, expr, bindings):
              continue
          result = instantiate(
              skeleton, bindings, engine._fold_funcs,
              undefined_op_resolver=engine._undefined_op_resolver,
              fold_error_resolver=engine._fold_error_resolver,
          )
          if result == expr:
              continue
          label = {
              "rule_index": rule_idx,
              "rule_id": _rule_identity(metadata, rule),
              "direction": getattr(metadata, "direction", None),
              "bindings": bindings.to_dict() if hasattr(bindings, "to_dict") else None,
              "path": list(_path),
              "_metadata": metadata,
          }
          edge = emit(result, label)
          if edge is not None:
              yield edge

      # Recurse into child positions, extending the path.
      if isinstance(expr, list) and expr:
          for i, child in enumerate(expr):
              for new_child, label in _labeled_rewrites(
                  engine, child, rules, _path + [i]
              ):
                  new_expr = expr[:i] + [new_child] + expr[i + 1:]
                  edge = emit(new_expr, label)
                  if edge is not None:
                      yield edge


  def solve(
      engine,
      expr: ExprType,
      goal_predicate: Callable[[ExprType], bool],
      *,
      cost_fn: Callable[[ExprType], float] = expr_size,
      max_nodes: int = 10000,
      fresh_vars: bool = True,
      normalize_between: bool = True,
  ) -> SolveResult:
      """Best-first search from ``expr`` to a node satisfying ``goal_predicate``.

      The escalation driver for non-confluent rule sets. Knows no domain;
      the goal is the caller's predicate.

      Args:
          engine: A `RuleEngine`. Its `rule_set()` (all active rules,
              both `=>` and `<=>`) defines the edges; `fresh_vars` controls
              whether `["fresh", base]` skeletons are resolved (handled in
              `instantiate`; this flag is threaded for forward-compat and is
              a no-op here since instantiation always resolves the form).
          expr: Start expression.
          goal_predicate: `expr -> bool`; search stops at the first node for
              which this is True. Caller-supplied; domain-free.
          cost_fn: Priority key (lower is expanded first). Defaults to
              `expr_size`.
          max_nodes: Budget on expanded (popped) nodes. On exhaustion the
              search fires `max_depth` on the engine and returns
              `found=False`.
          normalize_between: When True and Phase 2 `normalize` is available,
              normalize each generated node before enqueueing. When False or
              when `normalize` is absent, nodes are used as produced.

      Returns:
          A `SolveResult`.
      """
      from .engine import _expr_to_tuple

      def maybe_normalize(e: ExprType) -> ExprType:
          if normalize_between and _normalize is not None:
              return _normalize(e)
          return e

      rules = engine.rule_set()
      engine._step_count = 0
      engine._cancel_requested = False

      start = maybe_normalize(expr)

      trace = RewriteTrace()
      trace.initial = start

      # Goal already satisfied: zero-step derivation.
      if goal_predicate(start):
          trace.final = start
          return SolveResult(solution=start, derivation=trace,
                             explored=0, found=True)

      start_key = _expr_to_tuple(start)
      # Parent pointers: key -> (parent_key | None, step | None).
      parents = {start_key: (None, None)}
      # Priority queue of (cost, tiebreak, expr). tiebreak keeps it total.
      counter = 0
      frontier: List[Tuple[float, int, ExprType]] = [
          (cost_fn(start), counter, start)
      ]
      explored = 0

      def reconstruct(goal_key) -> List[RewriteStep]:
          steps: List[RewriteStep] = []
          key = goal_key
          while True:
              parent_key, step = parents[key]
              if step is None:
                  break
              steps.append(step)
              key = parent_key
          steps.reverse()
          return steps

      while frontier:
          if engine._cancel_requested:
              break
          if explored >= max_nodes:
              break

          _, _, node = heapq.heappop(frontier)
          explored += 1
          node_key = _expr_to_tuple(node)

          for neighbor, label in _labeled_rewrites(engine, node, rules):
              neighbor = maybe_normalize(neighbor)
              nkey = _expr_to_tuple(neighbor)
              if nkey in parents:
                  continue
              step = _make_step(label["_metadata"], node, neighbor, label)
              parents[nkey] = (node_key, step)

              if goal_predicate(neighbor):
                  trace.steps = reconstruct(nkey)
                  trace.final = neighbor
                  return SolveResult(solution=neighbor, derivation=trace,
                                     explored=explored, found=True)

              counter += 1
              heapq.heappush(frontier, (cost_fn(neighbor), counter, neighbor))

      # Exhausted budget or frontier without reaching the goal.
      engine._fire_max_depth(expr, explored)
      trace.final = start
      return SolveResult(solution=None, derivation=trace,
                         explored=explored, found=False)
  ```

  Run: `pytest rerum/tests/test_solve.py -q`
  Expected: PASS.

- [ ] **Step A6: Add the `RuleEngine.solve` wrapper and a test for it.**

  Append to `rerum/tests/test_solve.py`:

  ```python
  class TestEngineSolveWrapper:
      def test_engine_method_delegates(self):
          eng = _toy_engine()
          goal = lambda e: not contains_op(e, {"foo", "double"})
          result = eng.solve(["double", "x"], goal, max_nodes=200)
          assert result.found is True
          assert result.solution == ["+", "x", "x"]
  ```

  Run: `pytest rerum/tests/test_solve.py::TestEngineSolveWrapper -q`
  Expected: FAIL (AttributeError: 'RuleEngine' object has no attribute 'solve').

  Add the wrapper method to `RuleEngine` in `rerum/engine.py`, placed
  immediately after the `minimize` method (after the line returning
  `OptimizationResult(...)` near line 3686):

  ```python
      def solve(self, expr: ExprType, goal_predicate, **kw):
          """Goal-directed best-first search to a node satisfying
          ``goal_predicate``.

          The escalation driver above directed ``simplify``: use it for
          non-confluent rule sets where solving needs backtracking. Thin
          wrapper over :func:`rerum.solve.solve`. Keyword arguments
          (``cost_fn``, ``max_nodes``, ``fresh_vars``, ``normalize_between``)
          pass through unchanged.

          Returns a :class:`rerum.solve.SolveResult`.
          """
          from .solve import solve as _solve
          return _solve(self, expr, goal_predicate, **kw)
  ```

  Run: `pytest rerum/tests/test_solve.py -q`
  Expected: PASS.

- [ ] **Step A7: Export the search API and add an import smoke test.**

  In `rerum/__init__.py`, insert a new import group after the hooks import
  block (after line ~107):

  ```python
  # Goal-directed search (escalation driver)
  from .solve import (
      solve,
      SolveResult,
      contains_op,
  )
  ```

  And add to `__all__` (after the Hooks entries):

  ```python
      # Search
      "solve",
      "SolveResult",
      "contains_op",
  ```

  Append to `rerum/tests/test_solve.py`:

  ```python
  class TestTopLevelImports:
      def test_exports(self):
          import rerum
          assert rerum.solve is solve
          assert rerum.SolveResult is SolveResult
          assert rerum.contains_op is contains_op
  ```

  Run: `pytest rerum/tests/test_solve.py -q`
  Expected: PASS.

- [ ] **Step A8: Run the full suite for regressions and commit.**

  Run: `pytest -q`
  Expected: PASS (no regressions in existing tests).

  ```bash
  git add rerum/solve.py rerum/tests/test_solve.py rerum/engine.py rerum/__init__.py
  git commit -m "feat(solve): best-first escalation search, SolveResult, RuleEngine.solve wrapper"
  ```

---

## Task B: Fresh variables (`["fresh", base]` skeleton form)

**Files:** `rerum/rewriter.py`, `rerum/tests/test_fresh.py`, `rerum/__init__.py`

- [ ] **Step B1: Write a failing test for `free_symbols` and `gensym`.**

  Create `rerum/tests/test_fresh.py`:

  ```python
  """Tests for fresh-variable generation (gensym / free_symbols / fresh form)."""

  import pytest

  from rerum.rewriter import (
      free_symbols, gensym, instantiate, Bindings,
  )


  class TestFreeSymbols:
      def test_atom_variable(self):
          assert free_symbols("x") == {"x"}

      def test_constant_has_no_symbols(self):
          assert free_symbols(42) == set()
          assert free_symbols(3.14) == set()

      def test_compound_collects_leaves_not_operator_position(self):
          # Operator heads are symbols too, but free_symbols collects ALL
          # symbol leaves including the head, which is the conservative
          # choice for "names already in use".
          syms = free_symbols(["+", "x", ["*", 2, "y"]])
          assert "x" in syms and "y" in syms

      def test_empty_list(self):
          assert free_symbols([]) == set()


  class TestGensym:
      def test_base_when_free(self):
          assert gensym("u", set()) == "u"

      def test_skips_occupied(self):
          assert gensym("u", {"u"}) == "u1"
          assert gensym("u", {"u", "u1"}) == "u2"

      def test_deterministic(self):
          avoid = {"u", "u1", "u3"}
          assert gensym("u", avoid) == "u2"
          assert gensym("u", avoid) == "u2"  # same inputs -> same output
  ```

  Run: `pytest rerum/tests/test_fresh.py -q`
  Expected: FAIL (ImportError: cannot import name 'free_symbols' / 'gensym').

- [ ] **Step B2: Implement `free_symbols` and `gensym` in `rewriter.py`.**

  Add to `rerum/rewriter.py`, just after `free_in` (around line 576):

  ```python
  def free_symbols(expr: ExprType) -> set:
      """Return the set of all symbol (string) leaves occurring in ``expr``.

      Includes operator-head symbols; for fresh-variable avoidance the
      conservative superset of "names already present" is exactly what we
      want, so we do not distinguish head from operand position.
      """
      if isinstance(expr, str):
          return {expr}
      if isinstance(expr, list):
          out: set = set()
          for sub in expr:
              out |= free_symbols(sub)
          return out
      return set()


  def gensym(base: str, avoid: set) -> str:
      """Smallest of ``base``, ``base+"1"``, ``base+"2"``, ... not in ``avoid``.

      Deterministic: a pure function of ``base`` and ``avoid``.
      """
      if base not in avoid:
          return base
      i = 1
      while f"{base}{i}" in avoid:
          i += 1
      return f"{base}{i}"
  ```

  Run: `pytest rerum/tests/test_fresh.py -q`
  Expected: PASS.

- [ ] **Step B3: Commit.**

  ```bash
  git add rerum/rewriter.py rerum/tests/test_fresh.py
  git commit -m "feat(fresh): free_symbols and gensym helpers"
  ```

- [ ] **Step B4: Write a failing test for the `["fresh", base]` skeleton form.**

  Append to `rerum/tests/test_fresh.py`:

  ```python
  class TestFreshSkeleton:
      def test_fresh_resolves_to_base_when_free(self):
          # Skeleton (let (fresh u) (sin u)); but minimal: just the fresh
          # symbol spliced into a compound that does NOT already use it.
          skel = ["g", ["fresh", "u"], "y"]
          out = instantiate(skel, Bindings.empty())
          assert out == ["g", "u", "y"]

      def test_fresh_avoids_occurring_name(self):
          # The whole expression being built already contains `u`, so the
          # fresh form must pick `u1`.
          skel = ["g", "u", ["fresh", "u"]]
          out = instantiate(skel, Bindings.empty())
          assert out == ["g", "u", "u1"]

      def test_fresh_avoids_bound_substituted_name(self):
          # A bound variable :v resolves to `u`, occupying the name, so the
          # fresh form picks `u1`.
          b = Bindings([["v", "u"]])
          skel = ["g", [":", "v"], ["fresh", "u"]]
          out = instantiate(skel, b)
          assert out == ["g", "u", "u1"]

      def test_fresh_is_deterministic(self):
          skel = ["g", "u", ["fresh", "u"]]
          out1 = instantiate(skel, Bindings.empty())
          out2 = instantiate(skel, Bindings.empty())
          assert out1 == out2 == ["g", "u", "u1"]

      def test_two_fresh_in_same_expr_get_distinct_names(self):
          # Both ask for base `u`; the first takes `u`, the second must see
          # it occupied and take `u1`. Determinism requires left-to-right
          # resolution against the partially-built expression.
          skel = ["g", ["fresh", "u"], ["fresh", "u"]]
          out = instantiate(skel, Bindings.empty())
          assert out == ["g", "u", "u1"]
  ```

  Run: `pytest rerum/tests/test_fresh.py::TestFreshSkeleton -q`
  Expected: FAIL (the `["fresh", "u"]` form is currently treated as a
  literal compound and emitted verbatim).

- [ ] **Step B5: Implement the `["fresh", base]` form in `instantiate`.**

  The fresh form must avoid every symbol already present in the
  expression being built. Because `instantiate`'s inner `loop`/
  `instantiate_compound` build the result bottom-up, the simplest correct
  and deterministic approach is a post-pass: instantiate normally with
  `["fresh", base]` carried through as a placeholder marker, then resolve
  all fresh markers left-to-right against the accumulating set of used
  names. Implement as follows.

  Add a recognizer near the other skeleton recognizers in
  `rerum/rewriter.py` (after `skeleton_compute`, around line 556):

  ```python
  def skeleton_fresh(s: ExprType) -> bool:
      """Check if skeleton element is a fresh-variable form (fresh base).

      Form: ["fresh", "base"] - resolve to a gensym not free in the result.
      """
      return compound(s) and len(s) == 2 and car(s) == "fresh"
  ```

  In `instantiate`, the inner `loop` must leave a recognizable marker for
  fresh forms so the post-pass can resolve them deterministically. Add a
  branch in `loop` BEFORE the `skeleton_compute` branch (inside
  `instantiate`, around line 811, right after the `skeleton_evaluation`
  and `skeleton_splice` branches):

  ```python
          if skeleton_fresh(s):
              # Defer resolution: emit a unique unresolved marker; the
              # post-pass picks the smallest gensym not already used.
              return ["__fresh__", car(cdr(s))]
  ```

  Then wrap the body of `instantiate` so the returned expression is
  passed through a resolver. Replace the final `return loop(skeleton)`
  with:

  ```python
      built = loop(skeleton)
      return _resolve_fresh(built)
  ```

  And add the `_resolve_fresh` helper at module scope (after
  `instantiate_compound`, around line 889):

  ```python
  def _resolve_fresh(expr: ExprType) -> ExprType:
      """Replace ["__fresh__", base] markers with deterministic gensyms.

      Resolution is left-to-right (pre-order). Each resolved name is added
      to the avoid-set so two fresh forms with the same base get distinct
      names, and every already-present symbol in the expression is avoided.
      Pure and deterministic: a fixed input yields a fixed output.
      """
      # Names already present (excluding the markers themselves).
      def present(e: ExprType) -> set:
          if isinstance(e, str):
              return {e}
          if isinstance(e, list):
              if len(e) == 2 and e[0] == "__fresh__":
                  return set()
              out: set = set()
              for sub in e:
                  out |= present(sub)
              return out
          return set()

      used = present(expr)

      def walk(e: ExprType) -> ExprType:
          if isinstance(e, list):
              if len(e) == 2 and e[0] == "__fresh__":
                  name = gensym(e[1], used)
                  used.add(name)
                  return name
              return [walk(sub) for sub in e]
          return e

      return walk(expr)
  ```

  Run: `pytest rerum/tests/test_fresh.py -q`
  Expected: PASS.

- [ ] **Step B6: Guard against a top-level fresh form and run the suite.**

  Append to `rerum/tests/test_fresh.py`:

  ```python
  class TestFreshTopLevel:
      def test_fresh_as_whole_skeleton(self):
          # A bare ["fresh", "u"] skeleton resolves to "u" (nothing to avoid).
          out = instantiate(["fresh", "u"], Bindings.empty())
          assert out == "u"

      def test_non_fresh_compound_unaffected(self):
          # A normal two-element compound whose head is not "fresh" is
          # untouched.
          out = instantiate(["g", "u"], Bindings.empty())
          assert out == ["g", "u"]
  ```

  Run: `pytest rerum/tests/test_fresh.py -q`
  Expected: PASS.

  Run: `pytest -q`
  Expected: PASS (no regressions; in particular existing `instantiate`
  tests in `test_rewriter.py` are unaffected because non-fresh expressions
  contain no `__fresh__` markers and `_resolve_fresh` is a structural
  identity on them).

- [ ] **Step B7: Export `gensym` and `free_symbols` and commit.**

  In `rerum/__init__.py`, add to the `from .rewriter import (...)` block
  (after `safe_div` around line 63):

  ```python
      free_symbols,
      gensym,
  ```

  And to `__all__` (after `"safe_div"` around line 134):

  ```python
      "free_symbols",
      "gensym",
  ```

  Append to `rerum/tests/test_fresh.py`:

  ```python
  class TestFreshExports:
      def test_exports(self):
          import rerum
          assert rerum.gensym is gensym
          assert rerum.free_symbols is free_symbols
  ```

  Run: `pytest rerum/tests/test_fresh.py -q && pytest -q`
  Expected: PASS.

  ```bash
  git add rerum/rewriter.py rerum/tests/test_fresh.py rerum/__init__.py
  git commit -m "feat(fresh): deterministic (fresh base) skeleton form in instantiate"
  ```

---

## Task C: Exact rationals (`Fraction` in folds, coerce_number, format)

**Files:** `rerum/rewriter.py`, `rerum/expr.py`, `rerum/tests/test_rationals.py`, `rerum/__init__.py`

- [ ] **Step C1: Write a failing test for `coerce_number`.**

  Create `rerum/tests/test_rationals.py`:

  ```python
  """Tests for exact rational arithmetic (Fraction folds and formatting)."""

  from fractions import Fraction

  import pytest

  from rerum.rewriter import (
      coerce_number, safe_div, nary_fold, instantiate, Bindings,
      ARITHMETIC_PRELUDE,
  )
  from rerum.expr import format_sexpr, parse_sexpr


  class TestCoerceNumber:
      def test_int_passthrough(self):
          assert coerce_number(5) == 5
          assert isinstance(coerce_number(5), int)

      def test_float_integral_narrows_to_int(self):
          out = coerce_number(4.0)
          assert out == 4
          assert isinstance(out, int)

      def test_float_non_integral_stays_float(self):
          out = coerce_number(1.5)
          assert out == 1.5
          assert isinstance(out, float)

      def test_fraction_whole_collapses_to_int(self):
          out = coerce_number(Fraction(6, 3))
          assert out == 2
          assert isinstance(out, int)

      def test_fraction_non_integral_stays_fraction(self):
          out = coerce_number(Fraction(1, 3))
          assert out == Fraction(1, 3)
          assert isinstance(out, Fraction)

      def test_fraction_is_never_silently_floated(self):
          out = coerce_number(Fraction(1, 3))
          assert not isinstance(out, float)
  ```

  Run: `pytest rerum/tests/test_rationals.py::TestCoerceNumber -q`
  Expected: FAIL (ImportError: cannot import name 'coerce_number').

- [ ] **Step C2: Implement `coerce_number` in `rewriter.py`.**

  Add `from fractions import Fraction` to the imports at the top of
  `rerum/rewriter.py` (after `import math`, line 12):

  ```python
  from fractions import Fraction
  ```

  Add the helper after the type aliases (around line 21, before the
  `Bindings` class), so it is available to the fold builders below:

  ```python
  def coerce_number(x):
      """Normalize a numeric fold result to the tightest exact type.

      Rules:
      - ``int`` passes through unchanged.
      - ``float`` that is integral narrows to ``int``; otherwise stays float.
      - ``Fraction`` with denominator 1 collapses to ``int``; otherwise
        stays an exact ``Fraction`` (never silently floated).
      - any other value passes through unchanged.

      This is the single definition of int/float/Fraction narrowing; all
      fold handlers and the renarrowing in ``instantiate`` route through it.
      """
      if isinstance(x, bool):
          return x
      if isinstance(x, int):
          return x
      if isinstance(x, float):
          return int(x) if x.is_integer() else x
      if isinstance(x, Fraction):
          return int(x) if x.denominator == 1 else x
      return x
  ```

  Run: `pytest rerum/tests/test_rationals.py::TestCoerceNumber -q`
  Expected: PASS.

- [ ] **Step C3: Write a failing test for Fraction-returning `safe_div` and `nary_fold`.**

  Append to `rerum/tests/test_rationals.py`:

  ```python
  class TestSafeDivFraction:
      def test_non_integral_int_division_returns_fraction(self):
          handler = safe_div()
          out = handler([1, 3])
          assert out == Fraction(1, 3)
          assert isinstance(out, Fraction)

      def test_integral_int_division_collapses_to_int(self):
          handler = safe_div()
          out = handler([6, 3])
          assert out == 2
          assert isinstance(out, int)

      def test_division_by_zero_returns_none(self):
          handler = safe_div()
          assert handler([1, 0]) is None

      def test_float_division_stays_float(self):
          handler = safe_div()
          out = handler([1.0, 4.0])
          assert out == 0.25
          assert isinstance(out, float)


  class TestNaryFoldFraction:
      def test_sum_of_fractions_is_exact(self):
          add = nary_fold(0, lambda a, b: a + b)
          out = add([Fraction(1, 3), Fraction(1, 6)])
          assert out == Fraction(1, 2)
          assert isinstance(out, Fraction)

      def test_product_collapsing_to_int(self):
          mul = nary_fold(1, lambda a, b: a * b)
          out = mul([Fraction(2, 3), 3])
          assert out == 2
          assert isinstance(out, int)

      def test_plain_int_sum_unchanged(self):
          add = nary_fold(0, lambda a, b: a + b)
          out = add([1, 2, 3])
          assert out == 6
          assert isinstance(out, int)
  ```

  Run: `pytest rerum/tests/test_rationals.py -k "SafeDivFraction or NaryFoldFraction" -q`
  Expected: FAIL (`safe_div([1,3])` currently returns the float `0.3333...`;
  `nary_fold` does not route through `coerce_number`).

- [ ] **Step C4: Make `safe_div` and `nary_fold` exact via `coerce_number`.**

  In `rerum/rewriter.py`, update `safe_div` (around line 279) so integer
  inputs use exact `Fraction` division and the result is coerced:

  ```python
  def safe_div() -> FoldHandler:
      """Safe division handler that returns None on division by zero.

      For integer operands the quotient is computed exactly via ``Fraction``
      (so ``1/3`` stays ``Fraction(1, 3)`` rather than a lossy float), then
      narrowed by ``coerce_number`` (a whole quotient collapses to ``int``).
      Float operands divide as floats (then narrow integral results to int).
      """
      def handler(args: List[NumericType]) -> Optional[NumericType]:
          if len(args) != 2:
              return None
          a, b = args
          if b == 0:
              return None  # Can't fold division by zero
          if isinstance(a, int) and isinstance(b, int):
              return coerce_number(Fraction(a, b))
          if isinstance(a, Fraction) or isinstance(b, Fraction):
              return coerce_number(Fraction(a) / Fraction(b))
          return coerce_number(a / b)
      return handler
  ```

  Update `nary_fold` (around line 220) so the folded result is coerced:

  ```python
  def nary_fold(
      identity: NumericType,
      binary_op: Callable[[NumericType, NumericType], NumericType],
      unary: Optional[Callable[[NumericType], NumericType]] = None,
  ) -> FoldHandler:
      """Create an n-ary folder with identity element.

      The folded result is normalized via ``coerce_number`` so exact
      ``Fraction`` operands stay exact (and collapse to ``int`` when whole).

      Args:
          identity: Value for 0-arity, e.g., 0 for +, 1 for *
          binary_op: Binary operation for folding
          unary: Optional special unary behavior (defaults to identity)

      Examples:
          nary_fold(0, lambda a, b: a + b)  # (+) = 0, (+ x) = x, (+ x y z) = x+y+z
          nary_fold(1, lambda a, b: a * b)  # (*) = 1, (* x) = x, (* x y z) = x*y*z
      """
      def handler(args: List[NumericType]) -> NumericType:
          if len(args) == 0:
              return identity
          if len(args) == 1:
              return coerce_number(unary(args[0]) if unary else args[0])
          result = args[0]
          for a in args[1:]:
              result = binary_op(result, a)
          return coerce_number(result)
      return handler
  ```

  Run: `pytest rerum/tests/test_rationals.py -k "SafeDivFraction or NaryFoldFraction" -q`
  Expected: PASS.

- [ ] **Step C5: Write a failing test for Fraction formatting and instantiate renarrowing.**

  Append to `rerum/tests/test_rationals.py`:

  ```python
  class TestFractionFormat:
      def test_format_fraction_as_div(self):
          assert format_sexpr(Fraction(1, 3)) == "(/ 1 3)"

      def test_format_negative_fraction(self):
          assert format_sexpr(Fraction(-1, 3)) == "(/ -1 3)"

      def test_format_roundtrip_parses_back_to_div_form(self):
          s = format_sexpr(Fraction(2, 5))
          assert s == "(/ 2 5)"
          assert parse_sexpr(s) == ["/", 2, 5]

      def test_format_inside_compound(self):
          assert format_sexpr(["+", "x", Fraction(1, 2)]) == "(+ x (/ 1 2))"

      def test_format_whole_fraction_after_coercion_is_int(self):
          # coerce_number(Fraction(4,2)) -> 2; format is the int, not a div.
          assert format_sexpr(coerce_number(Fraction(4, 2))) == "2"


  class TestInstantiateRationals:
      def test_compute_keeps_fraction_exact(self):
          # (! / 1 3) must stay Fraction(1, 3), not collapse to float.
          skel = ["!", "/", 1, 3]
          out = instantiate(skel, Bindings.empty(), ARITHMETIC_PRELUDE)
          assert out == Fraction(1, 3)
          assert isinstance(out, Fraction)

      def test_compute_collapses_whole_division_to_int(self):
          skel = ["!", "/", 6, 3]
          out = instantiate(skel, Bindings.empty(), ARITHMETIC_PRELUDE)
          assert out == 2
          assert isinstance(out, int)

      def test_compute_fraction_sum(self):
          skel = ["!", "+", ["!", "/", 1, 3], ["!", "/", 1, 6]]
          out = instantiate(skel, Bindings.empty(), ARITHMETIC_PRELUDE)
          assert out == Fraction(1, 2)
          assert isinstance(out, Fraction)
  ```

  Run: `pytest rerum/tests/test_rationals.py -k "FractionFormat or InstantiateRationals" -q`
  Expected: FAIL (`format_sexpr(Fraction(...))` hits the numeric branch and
  returns `"1/3"`; `instantiate`'s renarrowing only handles `float` and
  re-coerces Fraction results away).

- [ ] **Step C6: Add `Fraction` formatting in `expr.py` and renarrowing in `instantiate`.**

  In `rerum/expr.py`, add the import at the top (after the `from typing`
  import, line 10):

  ```python
  from fractions import Fraction
  ```

  In `format_sexpr`, add a `Fraction` branch BEFORE the
  `isinstance(expr, (int, float))` branch (around line 225). Because
  `Fraction` is not a subclass of `int`/`float`, the new branch is
  necessary and must come first for clarity:

  ```python
      elif isinstance(expr, Fraction):
          return f"(/ {expr.numerator} {expr.denominator})"
      elif isinstance(expr, (int, float)):
          return str(expr)
  ```

  In `rerum/rewriter.py`, update the renarrowing inside `instantiate`'s
  `loop` (the `skeleton_compute` success branch around line 838) to route
  through `coerce_number` instead of the float-only narrowing:

  ```python
              if handler is not None:
                  try:
                      result = handler(args)
                      if result is not None:
                          return coerce_number(result)
                  except Exception as exc:
                      if fold_error_resolver is not None:
                          resolution = fold_error_resolver(op, args, exc)
                          if resolution is not None and resolution.value is not None:
                              return resolution.value
                      # Fall through to compound emission.
              return [op] + args
  ```

  Also update `try_constant_fold` inside `rewriter` (around line 1019) so
  the fast-path simplifier keeps Fractions exact. Replace the
  float-only narrowing:

  ```python
              # None means can't fold (wrong arity, etc.)
              if result is None:
                  return exp

              # Narrow to the tightest exact numeric type (int/Fraction/float).
              return coerce_number(result)
  ```

  Note: `try_constant_fold`'s guard `all(isinstance(arg, (int, float)) ...)`
  must also admit `Fraction` so that already-rational subexpressions fold.
  Update that guard (around line 1008):

  ```python
          # Check if all arguments are numeric constants (int/float/Fraction).
          if not all(isinstance(arg, (int, float, Fraction)) for arg in args):
              return exp
  ```

  Run: `pytest rerum/tests/test_rationals.py -q`
  Expected: PASS.

- [ ] **Step C7: Export `coerce_number`, run the full suite, and commit.**

  In `rerum/__init__.py`, add to the `from .rewriter import (...)` block
  (near `safe_div`):

  ```python
      coerce_number,
  ```

  And to `__all__`:

  ```python
      "coerce_number",
  ```

  Append to `rerum/tests/test_rationals.py`:

  ```python
  class TestRationalExports:
      def test_export(self):
          import rerum
          assert rerum.coerce_number is coerce_number
  ```

  Run: `pytest rerum/tests/test_rationals.py -q`
  Expected: PASS.

  Run: `pytest -q`
  Expected: PASS. Pay attention to any existing test that asserted a float
  result from `safe_div`/division folds (e.g. in `test_rewriter.py` or the
  example `.rules` integration tests): integer/integer division that was
  previously a float and happened to be whole still collapses to int via
  `coerce_number`, so only the genuinely non-integral integer-division
  cases change (now `Fraction` instead of float). Update any such assertion
  to expect the exact `Fraction`, since that is the intended Phase 3
  behavior.

  ```bash
  git add rerum/rewriter.py rerum/expr.py rerum/tests/test_rationals.py rerum/__init__.py
  git commit -m "feat(rationals): coerce_number, Fraction-exact folds and (/ p q) formatting"
  ```

---

## Task D: General numeric evaluator (`rerum/numeval.py`)

`numeval`/`numeric_equiv` are GENERAL primitives. `numeval` interprets a
ground term using ONLY the fold functions in the supplied prelude: it
special-cases no operator. A prelude is a `Dict[str, FoldHandler]` where a
`FoldHandler` is `Callable[[List[number]], Optional[number]]` (verified in
`rewriter.py`: `nary_fold`, `unary_only`, `binary_only`, `special_minus`,
`safe_div` all return handlers with that positional-args-list shape). So
`numeval` evaluates a compound by recursively evaluating its arguments, then
calling `prelude[head](evaluated_args)`. A missing operator, a handler that
returns `None` (cannot fold), or a handler that raises (e.g. `math.log` of a
negative, a domain error) all surface as a clear error. `numeric_equiv`
draws variable assignments from a sampler, evaluates both expressions, and
checks agreement within tolerance at every sampled point where both are
defined, skipping points where either raises a domain error.

Domain validators (such as "is this the derivative of that?") are NOT here.
They are example content under `examples/` (domain phases D1/D2) that CALL
these primitives. No domain operator name appears in this module.

**Files:** `rerum/numeval.py`, `rerum/tests/test_numeval.py`, `rerum/__init__.py`

- [ ] **Step D1: Write a failing test for `numeval` over real preludes.**

  Create `rerum/tests/test_numeval.py`:

  ```python
  """Tests for the general numeric evaluator (numeval / numeric_equiv)."""

  import math

  import pytest

  from rerum.numeval import numeval, numeric_equiv, NumevalError
  from rerum.rewriter import ARITHMETIC_PRELUDE, MATH_PRELUDE


  class TestNumevalAtoms:
      def test_number_returns_itself(self):
          assert numeval(7, {}, ARITHMETIC_PRELUDE) == 7
          assert numeval(3.5, {}, ARITHMETIC_PRELUDE) == 3.5

      def test_symbol_looks_up_env(self):
          assert numeval("x", {"x": 2.0}, ARITHMETIC_PRELUDE) == 2.0

      def test_unbound_symbol_raises(self):
          with pytest.raises(NumevalError):
              numeval("y", {"x": 1.0}, ARITHMETIC_PRELUDE)


  class TestNumevalCompounds:
      def test_nested_arithmetic(self):
          # (+ 1 (* 2 3)) == 7
          expr = ["+", 1, ["*", 2, 3]]
          assert numeval(expr, {}, ARITHMETIC_PRELUDE) == 7

      def test_env_substitution_then_evaluate(self):
          # (* x x) with x=2.0 == 4.0
          expr = ["*", "x", "x"]
          assert numeval(expr, {"x": 2.0}, ARITHMETIC_PRELUDE) == 4.0

      def test_math_prelude_operator(self):
          # (sin 0) == 0.0 via MATH_PRELUDE
          assert numeval(["sin", 0], {}, MATH_PRELUDE) == pytest.approx(0.0)

      def test_undefined_op_raises(self):
          # `quux` is in no prelude.
          with pytest.raises(NumevalError):
              numeval(["quux", 1, 2], {}, ARITHMETIC_PRELUDE)

      def test_domain_error_raises(self):
          # (log -1) is a math domain error under MATH_PRELUDE.
          with pytest.raises(NumevalError):
              numeval(["log", -1], {}, MATH_PRELUDE)
  ```

  Run: `pytest rerum/tests/test_numeval.py -q`
  Expected: FAIL (ModuleNotFoundError: No module named 'rerum.numeval').

- [ ] **Step D2: Implement `numeval` and `NumevalError` in a new `rerum/numeval.py`.**

  Create `rerum/numeval.py`:

  ```python
  """General numeric evaluation of ground terms under a prelude.

  GENERAL ENGINE PRINCIPLE: this module special-cases NO operator. A term is
  interpreted using only the fold functions in the supplied prelude, exactly
  the same extension point the rewriter uses for constant folding. Swap the
  prelude (arithmetic, math, a boolean prelude) and the same machinery
  evaluates a different algebra with no code change.

  A prelude is a ``Dict[str, FoldHandler]`` where a ``FoldHandler`` is
  ``Callable[[List[number]], Optional[number]]`` (see ``rewriter.py``). So a
  compound ``[op, a, b, ...]`` is evaluated by recursively evaluating each
  argument to a number, then calling ``prelude[op]([va, vb, ...])``.

  Domain validators (is-this-the-derivative-of-that, etc.) do NOT live here;
  they are example content that CALLS ``numeval``/``numeric_equiv``.
  """

  from typing import Callable, Dict, Mapping, Optional, Union

  from .rewriter import ExprType


  Number = Union[int, float]


  class NumevalError(Exception):
      """Raised when a ground term cannot be numerically evaluated.

      Covers an unbound symbol, an operator absent from the prelude, a
      handler that returns ``None`` (cannot fold), and a handler that raises
      a domain error (e.g. ``log`` of a negative).
      """


  def numeval(expr: ExprType, env: Mapping[str, Number], prelude: Dict) -> Number:
      """Evaluate a ground term to a number using ``prelude``'s fold funcs.

      Args:
          expr: A term. After substituting ``env`` for symbol leaves it must
              be ground (every symbol bound, every operator in ``prelude``).
          env: Maps symbol names to numbers.
          prelude: A ``Dict[str, FoldHandler]`` (e.g. ``ARITHMETIC_PRELUDE``,
              ``MATH_PRELUDE``). The ONLY source of operator semantics; no
              operator is special-cased.

      Returns:
          The numeric value of ``expr``.

      Raises:
          NumevalError: on an unbound symbol, an undefined operator, a
              non-foldable result, or a domain error from a handler.
      """
      # Atoms: a number is itself; a symbol is looked up in env.
      if isinstance(expr, bool):
          return expr
      if isinstance(expr, (int, float)):
          return expr
      if isinstance(expr, str):
          if expr in env:
              return env[expr]
          raise NumevalError(f"unbound symbol: {expr!r}")

      if isinstance(expr, list):
          if not expr or not isinstance(expr[0], str):
              raise NumevalError(f"not an evaluable compound: {expr!r}")
          op = expr[0]
          handler = prelude.get(op)
          if handler is None:
              raise NumevalError(f"undefined operator: {op!r}")
          args = [numeval(arg, env, prelude) for arg in expr[1:]]
          try:
              value = handler(args)
          except NumevalError:
              raise
          except Exception as exc:  # domain error, e.g. log of a negative
              raise NumevalError(
                  f"domain error evaluating ({op} ...): {exc}"
              ) from exc
          if value is None:
              raise NumevalError(
                  f"operator {op!r} could not fold args {args!r}"
              )
          return value

      raise NumevalError(f"not an evaluable term: {expr!r}")
  ```

  Run: `pytest rerum/tests/test_numeval.py -q`
  Expected: PASS (the `numeric_equiv` test is not yet written).

- [ ] **Step D3: Commit.**

  ```bash
  git add rerum/numeval.py rerum/tests/test_numeval.py
  git commit -m "feat(numeval): general numeric evaluation of ground terms under a prelude"
  ```

- [ ] **Step D4: Write a failing test for `numeric_equiv`.**

  Append to `rerum/tests/test_numeval.py`:

  ```python
  def _fixed_sampler(points):
      """A deterministic sampler: cycles through a fixed list of envs."""
      it = iter(points)

      def sample():
          return next(it)

      return sample


  class TestNumericEquiv:
      def test_equivalent_expressions_agree(self):
          # (+ x x) and (* 2 x) agree everywhere.
          a = ["+", "x", "x"]
          b = ["*", 2, "x"]
          sampler = lambda: {"x": 1.0}  # constant sampler is fine
          assert numeric_equiv(a, b, sampler, ARITHMETIC_PRELUDE, samples=8) is True

      def test_inequivalent_expressions_disagree(self):
          # (* x x) and (+ x 1) differ (e.g. at x=3: 9 vs 4).
          a = ["*", "x", "x"]
          b = ["+", "x", 1]
          # Sample a spread of points so at least one separates them.
          pts = [{"x": float(v)} for v in (0, 1, 2, 3, 4, 5, 6, 7)]
          sampler = _fixed_sampler(pts)
          assert numeric_equiv(a, b, sampler, ARITHMETIC_PRELUDE, samples=8) is False

      def test_dict_of_ranges_sampler(self):
          # A dict {var: (lo, hi)} is accepted as a sampler spec.
          a = ["+", "x", "x"]
          b = ["*", 2, "x"]
          assert numeric_equiv(a, b, {"x": (-5.0, 5.0)},
                               MATH_PRELUDE, samples=16) is True

      def test_domain_error_points_are_skipped_not_crashed(self):
          # (log x) vs (log x): identical, so always equal where defined.
          # Sampling includes x <= 0 (domain error for log); those points
          # must be skipped, not crash, and the verdict stays True.
          a = ["log", "x"]
          b = ["log", "x"]
          pts = [{"x": float(v)} for v in (-2, -1, 0, 1, 2, 3, 4, 5)]
          sampler = _fixed_sampler(pts)
          assert numeric_equiv(a, b, sampler, MATH_PRELUDE, samples=8) is True

      def test_within_tolerance(self):
          # Two expressions equal up to floating slack are equivalent.
          a = ["*", "x", 1]
          b = "x"
          assert numeric_equiv(a, b, lambda: {"x": 0.1},
                               ARITHMETIC_PRELUDE, samples=4, tol=1e-9) is True
  ```

  Run: `pytest rerum/tests/test_numeval.py::TestNumericEquiv -q`
  Expected: FAIL (ImportError: cannot import name 'numeric_equiv', since the
  test module imports it at the top; actually the whole module fails to
  collect once `numeric_equiv` is referenced and undefined). Implement next.

- [ ] **Step D5: Implement `numeric_equiv` (and a sampler normalizer).**

  Append to `rerum/numeval.py`:

  ```python
  import random


  def _as_sampler(spec) -> Callable[[], Dict[str, Number]]:
      """Normalize a sampler spec into a zero-arg env-producing callable.

      Accepts either:
      - a callable ``() -> {var: number}`` (returned as-is), or
      - a dict ``{var: (lo, hi)}`` of inclusive ranges, turned into a
        callable that draws each var uniformly from its range.
      """
      if callable(spec):
          return spec
      if isinstance(spec, dict):
          ranges = dict(spec)

          def sample() -> Dict[str, Number]:
              return {
                  var: random.uniform(lo, hi)
                  for var, (lo, hi) in ranges.items()
              }

          return sample
      raise TypeError(f"unsupported sampler spec: {spec!r}")


  def numeric_equiv(
      a: ExprType,
      b: ExprType,
      sampler,
      prelude: Dict,
      *,
      samples: int = 8,
      tol: float = 1e-6,
  ) -> bool:
      """True iff ``a`` and ``b`` evaluate equal at every defined sample point.

      Draws ``samples`` variable assignments from ``sampler`` (a callable
      ``() -> env`` or a dict ``{var: (lo, hi)}``), evaluates both
      expressions via :func:`numeval`, and returns True iff they agree within
      ``tol`` at every sampled point where BOTH are defined. Points where
      either expression raises a domain error (or any NumevalError) are
      SKIPPED, not counted as disagreement and not crashed.

      GENERAL: operator semantics come entirely from ``prelude``; no domain
      knowledge here.
      """
      draw = _as_sampler(sampler)
      for _ in range(samples):
          env = draw()
          try:
              va = numeval(a, env, prelude)
              vb = numeval(b, env, prelude)
          except NumevalError:
              # Skip points where either side is undefined (domain error,
              # unbound symbol from a partial env, etc.).
              continue
          if abs(va - vb) > tol:
              return False
      return True
  ```

  Run: `pytest rerum/tests/test_numeval.py -q`
  Expected: PASS.

- [ ] **Step D6: Export `numeval` and `numeric_equiv` and add an import smoke test.**

  In `rerum/__init__.py`, add a new import group after the search import
  group (after the `from .solve import (...)` block added in Task A):

  ```python
  # General numeric evaluation
  from .numeval import (
      numeval,
      numeric_equiv,
      NumevalError,
  )
  ```

  And add to `__all__` (after the Search entries):

  ```python
      # Numeric evaluation
      "numeval",
      "numeric_equiv",
      "NumevalError",
  ```

  Append to `rerum/tests/test_numeval.py`:

  ```python
  class TestNumevalExports:
      def test_exports(self):
          import rerum
          assert rerum.numeval is numeval
          assert rerum.numeric_equiv is numeric_equiv
          assert rerum.NumevalError is NumevalError
  ```

  Run: `pytest rerum/tests/test_numeval.py -q`
  Expected: PASS.

- [ ] **Step D7: Run the full suite and commit.**

  Run: `pytest -q`
  Expected: PASS (no regressions; `numeval`/`numeric_equiv` are additive
  and depend only on the existing preludes).

  ```bash
  git add rerum/numeval.py rerum/tests/test_numeval.py rerum/__init__.py
  git commit -m "feat(numeval): numeric_equiv sampling-based equivalence over a prelude"
  ```

---

## Done When

- [ ] `rerum/solve.py` exists with `SolveResult`, `contains_op`, and
  `solve(engine, expr, goal_predicate, *, cost_fn=expr_size,
  max_nodes=10000, fresh_vars=True, normalize_between=True)`. `solve` is the
  escalation driver above directed `simplify`; `contains_op` is a generic,
  operator-agnostic goal helper.
- [ ] `solve` is best-first (priority queue keyed by `cost_fn`), stops at
  the first node where the caller-supplied `goal_predicate` holds, budgets
  on `max_nodes` expanded nodes, and on exhaustion returns
  `SolveResult(found=False)` (never a partial result) and fires `max_depth`
  on the engine.
- [ ] `solve`'s derivation is a `RewriteTrace` whose steps carry the
  labeled edge (rule identity/direction/bindings/path when the running
  `RewriteStep` supports them) and whose `after` fields replay from
  `initial` to `solution`.
- [ ] `solve` reuses Phase 1's labeled-edge primitives (it builds labels
  from `rule_set` + `_match_internal` + `instantiate`) without depending on
  the exact return type of `_all_single_rewrites`, and treats Phase 2
  `normalize` as optional (guarded import; `normalize_between` toggle).
- [ ] `RuleEngine.solve(expr, goal_predicate, **kw)` delegates to
  `rerum.solve.solve`.
- [ ] `free_symbols(expr) -> set` and `gensym(base, avoid) -> str` exist,
  are pure and deterministic.
- [ ] The `["fresh", base]` skeleton form resolves in `instantiate` to the
  smallest of `base, base+"1", base+"2", ...` not free in the expression
  being built; two fresh forms with the same base in one skeleton get
  distinct names; same input yields same output.
- [ ] `coerce_number(x)` is the single int/float/Fraction narrowing helper
  (Fraction with denominator 1 -> int; a Fraction is never silently
  floated).
- [ ] `safe_div` returns `Fraction(1, 3)` for `(/ 1 3)` and `2` for
  `(/ 6 3)`; `nary_fold` keeps exact-rational results exact.
- [ ] `format_sexpr(Fraction(p, q))` renders `(/ p q)` and round-trips
  through `parse_sexpr` to `["/", p, q]`.
- [ ] The renarrowing in `instantiate` and `try_constant_fold` routes
  through `coerce_number`, so `(! / 1 3)` stays `Fraction(1, 3)`.
- [ ] `rerum/numeval.py` exists with `numeval(expr, env, prelude) -> number`
  and `numeric_equiv(a, b, sampler, prelude, *, samples=8, tol=1e-6) -> bool`
  and a `NumevalError`. `numeval` of `(+ 1 (* 2 3))` under
  `ARITHMETIC_PRELUDE` is `7`; `numeval` of `(* x x)` with `{"x": 2.0}` is
  `4.0`; `numeric_equiv((+ x x), (* 2 x), ...)` is True and
  `numeric_equiv((* x x), (+ x 1), ...)` is False; domain-error points are
  skipped, not crashed.
- [ ] `numeval`/`numeric_equiv` are GENERAL: operator semantics come ONLY
  from the supplied prelude (no operator special-cased), and the module
  names no domain operator. Domain validators live under `examples/` and
  call these primitives.
- [ ] `solve`, `SolveResult`, `contains_op`, `coerce_number`, `gensym`,
  `free_symbols`, `numeval`, `numeric_equiv`, `NumevalError` are exported
  from `rerum/__init__.py`.
- [ ] `pytest -q` passes with no regressions; `test_solve.py`,
  `test_rationals.py`, `test_fresh.py`, and `test_numeval.py` all pass.
