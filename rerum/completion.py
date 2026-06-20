"""F5: basic Knuth-Bendix completion (read-only analysis).

Turns a set of EQUATIONS into a CONFLUENT + TERMINATING rewrite system by the
basic completion loop: orient each equation into a rule (F4 orient), compute
critical pairs (F2 critical_pairs), normalize both sides with the current rules
(engine.simplify), and add any un-joined pair as a new oriented rule, until
every critical pair joins. Pure ORCHESTRATION of F2 + F4 + the engine; almost
no new math.

GENERAL ENGINE: the precedence and equations are DATA. First-order only. The
join test is SYNTACTIC (s == t) -- sound here because the internal normalization
engines (built by RuleEngine.from_rules) carry NO theory, so _canonicalize is
the identity and s == t coincides with F2's join test. A modulo-theory (AC)
extension must switch to _canonicalize-based comparison, as F2 does.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .rewriter import ExprType
from .engine import RuleEngine
from .confluence import critical_pairs, DirectedRule
from .termination import orient


def _term_to_skeleton(term: ExprType) -> ExprType:
    """Convert a TERM (``["?", name]`` variables) to a SKELETON
    (``[":", name]`` references) -- the forward of
    ``instantiate_skeleton(.., {})``. Recurses compounds; atoms unchanged."""
    if isinstance(term, list) and len(term) == 2 and term[0] == "?":
        return [":", term[1]]
    if isinstance(term, list):
        return [_term_to_skeleton(sub) for sub in term]
    return term


def _dedup(rules: list) -> list:
    """Drop structurally-duplicate ``(l, r)`` pairs, preserving first-occurrence
    order (O(n^2) list membership; n is small)."""
    out: list = []
    for rule in rules:
        if rule not in out:
            out.append(rule)
    return out


@dataclass(frozen=True)
class CompletionResult:
    """Result of a basic Knuth-Bendix completion run.

    ``status`` and ``rules`` are always set; ``failed_equation`` and
    ``iterations`` carry defaults so every return path constructs cleanly.
    A ``"complete"`` result is genuinely confluent + terminating (every critical
    pair joined; every rule oriented ``l >_lpo r``, hence terminating; Newman).
    ``"failed"`` means an equation no reduction order can orient (e.g.
    commutativity). ``"max_iterations"`` means basic completion did not converge
    within the budget (it is only a semi-decision procedure).

    ``frozen=True`` protects against attribute reassignment but does NOT make
    instances hashable (``rules`` is a list); do not put a ``CompletionResult``
    in a set.
    """
    status: str
    rules: List[Tuple[ExprType, ExprType]]
    failed_equation: Optional[Tuple[ExprType, ExprType]] = None
    iterations: int = 0

    def to_engine(self) -> "RuleEngine":
        """Build a fresh ``RuleEngine`` loaded with ``rules`` -- the ergonomic
        way to USE the result. The rules are ALWAYS terminating (each was
        LPO-oriented), but the engine is CONFLUENT only when
        ``status == "complete"``; for ``"max_iterations"``/``"failed"`` it loads
        the PARTIAL rule set (terminating, not necessarily confluent)."""
        return RuleEngine.from_rules(
            [[l, _term_to_skeleton(r)] for (l, r) in self.rules]
        )


def complete(equations, precedence, *, max_iterations: int = 100,
             max_steps: int = 1000) -> CompletionResult:
    """Basic Knuth-Bendix completion of ``equations`` (a list of ``(l, r)`` term
    pairs in ``["?", name]`` form) under ``precedence``. Read-only."""
    # 1. Orient the input. Drop trivial l == r BEFORE orient (orient returns
    #    None on structurally-equal terms, which would be a spurious "failed").
    rules: List[Tuple[ExprType, ExprType]] = []
    for (l, r) in equations:
        if l == r:
            continue
        d = orient(l, r, precedence)
        if d is None:
            return CompletionResult(status="failed", rules=list(rules),
                                    failed_equation=(l, r), iterations=0)
        rules.append((l, r) if d == "lr" else (r, l))
    rules = _dedup(rules)

    # 2. Fixpoint: orient-and-add until no critical pair is un-joined.
    for iteration in range(max_iterations):
        passes = iteration + 1
        records = [
            DirectedRule(name=str(i), pattern=l,
                         skeleton=_term_to_skeleton(r), condition=None)
            for i, (l, r) in enumerate(rules)
        ]
        eng = RuleEngine.from_rules(
            [[l, _term_to_skeleton(r)] for (l, r) in rules]
        )
        # not_analyzed is always empty here: every rule the loop builds is a
        # first-order [pattern, [":",x]-skeleton] with no condition, so
        # is_analyzable accepts it.
        pairs, _na = critical_pairs(records)
        new_rules: List[Tuple[ExprType, ExprType]] = []
        for cp in pairs:
            try:
                s = eng.simplify(cp.left, max_steps=max_steps)
                t = eng.simplify(cp.right, max_steps=max_steps)
            except RecursionError:
                # Treat as NOT joining: conservative about the "complete"
                # verdict (a non-normalizing pair never makes s == t, so it
                # cannot cause a false "complete"). It may, in a pathological
                # case, route to an added rule instead of "failed", surfacing
                # as "max_iterations" -- a status-accuracy cost, never a
                # soundness one.
                s, t = cp.left, cp.right
            if s == t:
                continue
            d = orient(s, t, precedence)
            if d is None:
                return CompletionResult(status="failed", rules=list(rules),
                                        failed_equation=(s, t),
                                        iterations=passes)
            new = (s, t) if d == "lr" else (t, s)
            if new not in rules and new not in new_rules:
                new_rules.append(new)
        if not new_rules:
            return CompletionResult(status="complete", rules=list(rules),
                                    iterations=passes)
        rules = _dedup(rules + new_rules)

    return CompletionResult(status="max_iterations", rules=list(rules),
                            iterations=max_iterations)
