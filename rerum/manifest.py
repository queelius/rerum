"""Rule-set manifests: self-describing rule files.

A manifest is a DSL file carrying ``:``-directives -- alongside the existing
``:include`` -- that declare a domain's loading contract:

    :requires math predicate          # named prelude bundles to combine
    :requires-ops dd                  # fold ops the rules need (audited)
    :theory arithmetic.theory.json    # session theory (relative path)
    :metadata differentiation.metadata.json   # examples sidecar (relative)
    :driver simplify                  # how to drive it (hint; data only)
    :goal {"op_free": ["int"]}        # goal description (hint; data only)
    :include differentiation.rules    # the rule body

``RuleEngine.from_manifest(path)`` assembles the whole domain from one file
(install preludes, set theory, load rules, merge the sidecar) and FAILS LOUD
when a required fold op is absent -- the silent-junk footgun where a missing
skeleton ``(! op ...)`` survives as a literal compound. This module is the
PARSE + AUDIT half (pure, no engine state); ``RuleEngine.from_manifest`` does
the assembly.
"""

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .rewriter import PRELUDE_BUNDLES

# The driver hints a manifest may declare. Data only -- the engine stores
# the value but does not act on it (a caller may read it).
_DRIVERS = {"simplify", "solve"}

# Recognized directive heads (``:include`` is handled by the rule loader, not
# the manifest; it is allowed here so an unknown-directive typo still raises).
_DIRECTIVES = {
    "requires", "requires-ops", "theory", "metadata", "driver", "goal",
}


@dataclass(frozen=True)
class RuleSetManifest:
    """The declared loading contract of a rule file (all fields optional).

    ``requires``/``requires_ops`` are tuples (immutable, order-preserving);
    ``theory``/``metadata`` are relative paths; ``driver``/``goal`` are hints
    the engine stores but does not act on.
    """

    requires: Tuple[str, ...] = ()
    requires_ops: Tuple[str, ...] = ()
    theory: Optional[str] = None
    metadata: Optional[str] = None
    driver: Optional[str] = None
    goal: Optional[Dict] = None

    @property
    def is_empty(self) -> bool:
        """True when the file declared no manifest directive at all."""
        return (not self.requires and not self.requires_ops
                and self.theory is None and self.metadata is None
                and self.driver is None and self.goal is None)


def parse_manifest(text: str) -> RuleSetManifest:
    """Scan DSL ``text`` for manifest directives (ignoring rules/groups/
    ``:include``) and return a :class:`RuleSetManifest`.

    Raises ``ValueError`` on a malformed directive: an unknown bundle in
    ``:requires``, an unknown ``:driver`` value, non-JSON ``:goal``, a
    duplicate single-valued directive, or an unknown ``:``-directive (so a
    typo like ``:requies`` is caught rather than silently ignored).
    """
    requires: List[str] = []
    requires_ops: List[str] = []
    theory: Optional[str] = None
    metadata: Optional[str] = None
    driver: Optional[str] = None
    goal: Optional[Dict] = None

    for raw in text.split("\n"):
        line = raw.strip()
        if not line.startswith(":"):
            continue
        head, _, rest = line[1:].partition(" ")
        head = head.strip()
        rest = rest.strip()
        if head == "include":
            continue  # the rule loader's directive, not the manifest's
        if head not in _DIRECTIVES:
            raise ValueError(
                f"unknown manifest directive ':{head}'; known: "
                f"{sorted(_DIRECTIVES)} (and :include)")

        if head == "requires":
            for name in rest.split():
                if name not in PRELUDE_BUNDLES:
                    raise ValueError(
                        f":requires names unknown prelude bundle {name!r}; "
                        f"valid: {sorted(PRELUDE_BUNDLES)}")
                requires.append(name)
        elif head == "requires-ops":
            requires_ops.extend(rest.split())
        elif head == "theory":
            if theory is not None:
                raise ValueError(":theory may appear only once")
            theory = rest
        elif head == "metadata":
            if metadata is not None:
                raise ValueError(":metadata may appear only once")
            metadata = rest
        elif head == "driver":
            if driver is not None:
                raise ValueError(":driver may appear only once")
            if rest not in _DRIVERS:
                raise ValueError(
                    f":driver must be one of {sorted(_DRIVERS)}, got {rest!r}")
            driver = rest
        elif head == "goal":
            if goal is not None:
                raise ValueError(":goal may appear only once")
            try:
                parsed = json.loads(rest)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f":goal must be a JSON object, got {rest!r}: {exc}")
            if not isinstance(parsed, dict):
                raise ValueError(
                    f":goal must be a JSON OBJECT, got {type(parsed).__name__}")
            goal = parsed

    return RuleSetManifest(
        requires=tuple(requires),
        requires_ops=tuple(requires_ops),
        theory=theory,
        metadata=metadata,
        driver=driver,
        goal=goal,
    )


def collect_fold_ops(expr) -> set:
    """Every ``(! op ...)`` head reachable in ``expr`` (a pattern, skeleton,
    or guard condition). The op name is at index 1 of an ``['!', op, ...]``
    node; recurse into all children."""
    ops = set()
    if isinstance(expr, list):
        if len(expr) >= 2 and expr[0] == "!" and isinstance(expr[1], str):
            ops.add(expr[1])
        for sub in expr:
            ops |= collect_fold_ops(sub)
    return ops
