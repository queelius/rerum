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
