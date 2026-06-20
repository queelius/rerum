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
