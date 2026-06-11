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
  when available, a theory is supplied, and `normalize_between=True`.
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
    if "rationale" in _STEP_PARAMS:
        # Parity with the engine's own emit sites (rule_applied stamps
        # rationale=metadata.reasoning or metadata.category): the sidecar
        # 'reasoning' field exists precisely for paraphrasable derivation
        # steps, and without this solve-driven corpora lose the WHY.
        kwargs["rationale"] = (getattr(metadata, "reasoning", None)
                               or getattr(metadata, "category", None))
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
            return None
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
    theory=None,
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
        normalize_between: When True and Phase 2 `normalize` is available
            AND a ``theory`` is supplied, normalize each generated node
            before enqueueing. When False, when `normalize` is absent, or
            when no theory is given, nodes are used as produced. (The real
            `normalize` requires a theory signature, so normalization is a
            no-op without one.)
        theory: Optional `normalize.Theory` driving best-effort
            normalization between nodes. Domain-free: it is the caller's
            operator-signature data, not a hardcoded algebra.

    Returns:
        A `SolveResult`.
    """
    from .engine import _expr_to_tuple

    def maybe_normalize(e: ExprType) -> ExprType:
        if (normalize_between and _normalize is not None
                and theory is not None):
            return _normalize(e, theory)
        return e

    # Materialize the active rule set once: the RuleSet view is re-iterated
    # at every node and every recursive child position.
    rules = list(engine.rule_set())
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
    # Parent pointers double as the visited set: key -> (parent_key, step).
    # The key is _expr_to_tuple (identity on atoms), so node identity tracks
    # Python equality exactly: True/1 and False/0 alias deliberately (they are
    # == and hash-equal everywhere else in the engine) and no != pair is ever
    # merged. A str()-based key would be unsafe here (it would alias "1" and 1).
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
