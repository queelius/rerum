"""
Rule Engine and DSL Loader for RERUM

RERUM - Rewriting Expressions via Rules Using Morphisms

This module provides facilities for loading rewriting rules from external files,
supporting both a custom DSL format and JSON.

DSL Format (.rules files):
    # Comment
    @rule-name: (pattern) => (skeleton)              # Unidirectional
    @rule-name: (pattern) <=> (skeleton)             # Bidirectional (creates 2 rules)
    @rule-name "Description text": (pattern) => (skeleton)

    Examples:
    @add-zero: (+ ?x 0) => :x
    @dd-sum "Derivative of sum": (dd (+ ?f ?g) ?v:var) => (+ (dd :f :v) (dd :g :v))

    Bidirectional rules create two rules:
    @commute: (+ ?x ?y) <=> (+ :y :x)
    # Creates: @commute-fwd: (+ ?x ?y) => (+ :y :x)
    #          @commute-rev: (+ ?y ?x) => (+ :x :y)

Pattern syntax:
    ?x or ?x:expr      - match any expression, bind to x
    ?x:const           - match constant only, bind to x
    ?x:var             - match variable only, bind to x
    ?x:free(var)       - match expression not containing var
    ?x...              - match rest of arguments (variadic), bind to x

Skeleton syntax:
    :x      - substitute bound value of x
    literal - use as-is

JSON Format:
    {
        "name": "algebra",
        "description": "Algebraic simplification rules",
        "rules": [
            {"name": "add-zero", "description": "...", "pattern": [...], "skeleton": [...]},
            or just [pattern, skeleton]
        ]
    }

Tracing:
    Use RuleEngine.simplify(expr, trace=True) to see which rules are applied.
"""

import json
import random
import re
from collections import deque
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union, Iterator, Set, Callable

from .rewriter import (
    rewriter, match as _match_internal, instantiate, ExprType,
    FoldFuncsType, ARITHMETIC_PRELUDE, Bindings, NoMatch, _NoMatch, wrap_bindings,
)
from .hooks import (
    _HookRegistry,
    HookContext,
    Resolution,
    HookError,
    ResolutionError,
    ResolverLoopError,
)


# ============================================================
# Expression Builder
# ============================================================

# Expression model (parse_sexpr, format_sexpr, expr_to_tuple, E) lives in
# `expr.py`. Re-exported below for backward compatibility.
from .expr import (
    parse_sexpr,
    format_sexpr,
    E,
    _ExprBuilder,
    expr_to_tuple as _expr_to_tuple,
)


class RuleMetadata:
    """Metadata for a rule including name, description, priority, and condition."""

    def __init__(self, name: Optional[str] = None, description: Optional[str] = None,
                 tags: Optional[List[str]] = None, condition: Optional[ExprType] = None,
                 priority: int = 0, bidirectional: bool = False,
                 direction: Optional[str] = None):
        self.name = name
        self.description = description
        self.tags = tags or []
        self.condition = condition  # Optional guard condition
        self.priority = priority  # Higher priority fires first (default: 0)
        self.bidirectional = bidirectional  # True if from a <=> rule
        self.direction = direction  # 'fwd' or 'rev' for bidirectional rules

    def __repr__(self) -> str:
        base = ""
        if self.name:
            if self.priority != 0:
                base = f"@{self.name}[{self.priority}]"
            else:
                base = f"@{self.name}"
            if self.description:
                base += f" \"{self.description}\""
        else:
            base = "<anonymous>"

        if self.condition:
            base += f" when {format_sexpr(self.condition)}"
        return base


class BidirectionalRule:
    """A value object that pairs the ``-fwd`` and ``-rev`` halves of a `<=>`
    rule for inspection and serialization.

    The rule storage in :class:`RuleEngine` keeps the directions as two
    separate entries (so the application loop, group filtering, and priority
    sort all stay direction-symmetric). This wrapper aggregates them for
    contexts where the user wants to think about a single logical rule:
    listing source rules, formatting a DSL line, JSON export, or counting
    "rules I wrote" rather than "rules in storage."

    Construct via :meth:`RuleEngine.source_rules`. Read-only.
    """

    __slots__ = ("name", "description", "priority", "condition", "tags",
                 "fwd_pattern", "fwd_skeleton", "rev_pattern", "rev_skeleton")

    def __init__(self, *, name: Optional[str], description: Optional[str],
                 priority: int, condition: Optional[ExprType],
                 tags: Optional[List[str]],
                 fwd_pattern: ExprType, fwd_skeleton: ExprType,
                 rev_pattern: ExprType, rev_skeleton: ExprType):
        self.name = name
        self.description = description
        self.priority = priority
        self.condition = condition
        self.tags = tags or []
        self.fwd_pattern = fwd_pattern
        self.fwd_skeleton = fwd_skeleton
        self.rev_pattern = rev_pattern
        self.rev_skeleton = rev_skeleton

    @property
    def bidirectional(self) -> bool:
        return True

    def __repr__(self) -> str:
        n = self.name or "<anonymous>"
        return f"BidirectionalRule(@{n}: <=>)"


class UnidirectionalRule:
    """A value object representing a single `=>` rule. Symmetric counterpart
    to :class:`BidirectionalRule` for unified iteration over source rules.
    """

    __slots__ = ("name", "description", "priority", "condition", "tags",
                 "pattern", "skeleton")

    def __init__(self, *, name: Optional[str], description: Optional[str],
                 priority: int, condition: Optional[ExprType],
                 tags: Optional[List[str]],
                 pattern: ExprType, skeleton: ExprType):
        self.name = name
        self.description = description
        self.priority = priority
        self.condition = condition
        self.tags = tags or []
        self.pattern = pattern
        self.skeleton = skeleton

    @property
    def bidirectional(self) -> bool:
        return False

    def __repr__(self) -> str:
        n = self.name or "<anonymous>"
        return f"UnidirectionalRule(@{n}: =>)"


# Cost functions and OptimizationResult live in `optimize.py`.
# Re-exported below for backward compatibility.
from .optimize import (
    expr_size,
    expr_depth,
    expr_ops,
    expr_atoms,
    make_op_cost_fn,
    COST_METRICS,
    OptimizationResult,
)


def _convert_skeleton_to_pattern(expr: ExprType,
                                  constraints: Optional[Dict[str, List]] = None) -> ExprType:
    """
    Convert a skeleton expression to a pattern expression.

    Transforms substitution markers to pattern variables:
    - (: x) -> (? x)        (or restored constraint, e.g. (?c x), (?v x), (?free x v))
    - (:... x) -> (?... x)

    When ``constraints`` is provided, restores type and free-variable
    constraints attached to each variable name in the original pattern.
    This is what makes a `<=>` rule sound: without restoration, the
    reverse direction's pattern matches expressions that the forward
    direction's pattern would have rejected.
    """
    constraints = constraints or {}
    if isinstance(expr, list):
        if len(expr) == 2:
            if expr[0] == ":":
                name = expr[1]
                if name in constraints:
                    return list(constraints[name])
                return ["?", name]
            elif expr[0] == ":...":
                name = expr[1]
                if name in constraints:
                    return list(constraints[name])
                return ["?...", name]
        # Recurse for compound expressions
        return [_convert_skeleton_to_pattern(e, constraints) for e in expr]
    return expr


def _validate_pattern_structure(pattern: ExprType, *, where: str = "pattern") -> None:
    """Raise ``ValueError`` for structurally invalid patterns.

    - At most one ``?...`` rest pattern per compound, and if present it
      must be the last element. Otherwise `match_compound` raises lazily,
      producing a confusing error far from the rule definition.
    - ``(! ...)`` compute forms are not valid in pattern position; they
      belong on the skeleton side. Detected here because the auto-derived
      reverse pattern of a `<=>` rule whose original skeleton contained
      a compute form would otherwise be silently broken.

    The ``where`` argument disambiguates error messages between the
    user-written pattern and the auto-derived reverse pattern.
    """
    if not isinstance(pattern, list):
        return
    if pattern and pattern[0] == "!":
        raise ValueError(
            f"Invalid {where}: compute form `(! ...)` is for skeletons, "
            f"not patterns. Got: {pattern}. If this came from the reverse "
            f"of a `<=>` rule, the original skeleton's compute form does "
            f"not have a meaningful inverse; use two `=>` rules instead."
        )
    rest_seen = False
    for idx, child in enumerate(pattern):
        is_rest = (
            isinstance(child, list)
            and child
            and isinstance(child[0], str)
            and child[0] == "?..."
        )
        if is_rest:
            if rest_seen:
                raise ValueError(
                    f"Invalid {where}: at most one `?...` rest pattern is "
                    f"allowed per compound. Got: {pattern}"
                )
            rest_seen = True
            if idx != len(pattern) - 1:
                raise ValueError(
                    f"Invalid {where}: `?...` rest pattern must be the last "
                    f"element of the compound. Got: {pattern}"
                )
    for child in pattern:
        _validate_pattern_structure(child, where=where)


def _strip_bidirectional_naming(meta: "RuleMetadata") -> Tuple[Optional[str], Optional[str]]:
    """Recover the ``(base_name, base_description)`` of a `<=>` source rule
    from its -fwd metadata entry. Used when emitting paired rules back
    out as a single bidirectional rule for roundtrip-safe serialization.
    """
    base_name = meta.name
    if base_name and base_name.endswith("-fwd"):
        base_name = base_name[:-4]
    base_description = meta.description
    if base_description and base_description.endswith(" (forward)"):
        base_description = base_description[:-len(" (forward)")]
    return base_name, base_description


def _is_bidirectional_pair(metadata: List["RuleMetadata"], i: int) -> bool:
    """True iff entries `i` and `i+1` form a `-fwd`/`-rev` pair (adjacent,
    both bidirectional, with matching base names)."""
    if i + 1 >= len(metadata):
        return False
    fwd, rev = metadata[i], metadata[i + 1]
    if not (fwd.bidirectional and fwd.direction == "fwd"):
        return False
    if not (rev.bidirectional and rev.direction == "rev"):
        return False
    fwd_base, _ = _strip_bidirectional_naming(fwd)
    rev_base = rev.name
    if rev_base and rev_base.endswith("-rev"):
        rev_base = rev_base[:-4]
    # If both anonymous (None == None) accept; if either has a name, they
    # must agree on base.
    return fwd_base == rev_base


def _build_bidirectional_rules(
    base_name: Optional[str],
    description: Optional[str],
    priority: int,
    condition: Optional[ExprType],
    tags: Optional[List[str]],
    pattern: ExprType,
    skeleton: ExprType,
) -> List[Tuple["RuleMetadata", ExprType, ExprType]]:
    """Construct the forward and reverse `RuleMetadata` plus rule structures
    from a single `<=>` rule. Used by both the DSL and JSON loaders so the
    desugaring is identical regardless of source format.
    """
    fwd_metadata = RuleMetadata(
        name=f"{base_name}-fwd" if base_name else None,
        description=f"{description} (forward)" if description else None,
        tags=tags,
        priority=priority,
        condition=condition,
        bidirectional=True,
        direction="fwd",
    )
    rev_metadata = RuleMetadata(
        name=f"{base_name}-rev" if base_name else None,
        description=f"{description} (reverse)" if description else None,
        tags=tags,
        priority=priority,
        condition=condition,
        bidirectional=True,
        direction="rev",
    )
    constraints = _extract_pattern_constraints(pattern)
    rev_pattern = _convert_skeleton_to_pattern(skeleton, constraints)
    rev_skeleton = _convert_pattern_to_skeleton(pattern)
    # Validate the auto-derived reverse pattern eagerly so that ill-formed
    # `<=>` rules fail at parse/load time with a clear message rather than
    # lazily inside the rule-application loop.
    _validate_pattern_structure(rev_pattern, where="reverse pattern")
    return [
        (fwd_metadata, pattern, skeleton),
        (rev_metadata, rev_pattern, rev_skeleton),
    ]


def _extract_pattern_constraints(pattern: ExprType) -> Dict[str, List]:
    """Walk a pattern and return ``{var_name: full_pattern_var_form}``.

    For each `?`-headed pattern variable in the input, records the full
    form (including type constraint or free-of constraint) keyed by name.
    Used by `_convert_skeleton_to_pattern` to restore constraints when
    constructing the reverse direction of a bidirectional rule.
    """
    constraints: Dict[str, List] = {}

    def walk(expr):
        if isinstance(expr, list) and expr:
            head = expr[0]
            if isinstance(head, str) and head.startswith("?"):
                if len(expr) >= 2 and isinstance(expr[1], str):
                    constraints[expr[1]] = list(expr)
                # Pattern variable forms don't have nested patterns to walk.
                return
            for e in expr:
                walk(e)

    walk(pattern)
    return constraints


def _convert_pattern_to_skeleton(expr: ExprType) -> ExprType:
    """
    Convert a pattern expression to a skeleton expression.

    Transforms pattern variables to substitution markers:
    - (? x) -> (: x)
    - (?c x) -> (: x)  (type constraints dropped)
    - (?v x) -> (: x)  (type constraints dropped)
    - (?free x v) -> (: x)  (free constraints dropped)
    - (?... x) -> (:... x)
    """
    if isinstance(expr, list):
        if len(expr) == 2:
            if expr[0] == "?":
                return [":", expr[1]]
            elif expr[0] == "?c":
                return [":", expr[1]]
            elif expr[0] == "?v":
                return [":", expr[1]]
            elif expr[0] == "?...":
                return [":...", expr[1]]
        elif len(expr) == 3:
            if expr[0] == "?free":
                return [":", expr[1]]
            elif expr[0] == "?...":
                # Rest pattern with type constraint: ["?...", "name", "const"]
                return [":...", expr[1]]
        # Recurse for compound expressions
        return [_convert_pattern_to_skeleton(e) for e in expr]
    return expr


def parse_rule_line(line: str) -> Optional[List[Tuple[RuleMetadata, ExprType, ExprType]]]:
    """
    Parse a single rule line, potentially returning multiple rules.

    Formats:
        @name: pattern => skeleton           # Unidirectional rule
        @name: pattern <=> skeleton          # Bidirectional rule (creates 2 rules)
        @name[priority]: pattern => skeleton
        @name "description": pattern => skeleton
        @name[priority] "description": pattern => skeleton
        @name: pattern => skeleton when condition
        pattern => skeleton

    Returns: List of (metadata, pattern, skeleton) tuples, or None if not a rule.
             Bidirectional rules (<=>)  return two rules: forward (-fwd) and reverse (-rev).
    """
    line = line.strip()

    # Skip empty lines and comments
    if not line or line.startswith('#'):
        return None

    # Extract name, optional priority, and optional description if present
    base_name = None
    description = None
    priority = 0

    if line.startswith('@'):
        # Try format: @name[priority] "description": ...
        match_obj = re.match(r'@([\w-]+)\[(\d+)\]\s+"([^"]+)":\s*(.+)', line)
        if match_obj:
            base_name = match_obj.group(1)
            priority = int(match_obj.group(2))
            description = match_obj.group(3)
            line = match_obj.group(4)
        else:
            # Try format: @name[priority]: ...
            match_obj = re.match(r'@([\w-]+)\[(\d+)\]:\s*(.+)', line)
            if match_obj:
                base_name = match_obj.group(1)
                priority = int(match_obj.group(2))
                line = match_obj.group(3)
            else:
                # Try format: @name "description": ...
                match_obj = re.match(r'@([\w-]+)\s+"([^"]+)":\s*(.+)', line)
                if match_obj:
                    base_name = match_obj.group(1)
                    description = match_obj.group(2)
                    line = match_obj.group(3)
                else:
                    # Try format: @name: ...
                    match_obj = re.match(r'@([\w-]+):\s*(.+)', line)
                    if match_obj:
                        base_name = match_obj.group(1)
                        line = match_obj.group(2)

    # Determine if bidirectional (<=>)  or unidirectional (=>)
    is_bidirectional = '<=>' in line

    if is_bidirectional:
        # Split on <=>
        parts = line.split('<=>', 1)
    elif '=>' in line:
        # Split on =>
        parts = line.split('=>', 1)
    else:
        return None

    pattern_str = parts[0].strip()
    rest = parts[1].strip()

    # Check for 'when' clause - need to find top-level 'when' not inside parens
    skeleton_str = rest
    condition = None

    # Find 'when' at top level (not inside parentheses)
    depth = 0
    when_pos = -1
    i = 0
    while i < len(rest):
        if rest[i] == '(':
            depth += 1
        elif rest[i] == ')':
            depth -= 1
        elif depth == 0 and rest[i:i+4] == 'when' and (i == 0 or rest[i-1].isspace()):
            # Check it's followed by space or end
            after = i + 4
            if after >= len(rest) or rest[after].isspace():
                when_pos = i
                break
        i += 1

    if when_pos >= 0:
        skeleton_str = rest[:when_pos].strip()
        condition_str = rest[when_pos + 4:].strip()
        condition = parse_sexpr(condition_str)

    pattern = parse_sexpr(pattern_str)
    skeleton = parse_sexpr(skeleton_str)

    if pattern is None or skeleton is None:
        return None

    if is_bidirectional:
        return _build_bidirectional_rules(
            base_name=base_name,
            description=description,
            priority=priority,
            condition=condition,
            tags=None,
            pattern=pattern,
            skeleton=skeleton,
        )
    else:
        # Single unidirectional rule
        metadata = RuleMetadata(
            name=base_name,
            description=description,
            priority=priority,
            condition=condition
        )
        return [(metadata, pattern, skeleton)]


def load_rules_from_dsl(
    text: str,
    base_path: Optional[Path] = None,
    _included_files: Optional[set] = None
) -> List[Tuple[RuleMetadata, List]]:
    """
    Load rules from DSL text.

    Supports:
    - Named groups: [groupname]
    - File includes: :include path/to/file.rules

    Example:
        [algebra]
        @add-zero: (+ ?x 0) => :x

        :include calculus.rules

        [simplify]
        @fold: (* ?a ?b) => (! * :a :b)

    Args:
        text: DSL text containing rules
        base_path: Base path for resolving relative :include paths
        _included_files: Internal tracking for circular include detection

    Returns:
        List of (metadata, [pattern, skeleton]) tuples
    """
    rules = []
    current_group = None

    # Track included files to prevent circular includes
    if _included_files is None:
        _included_files = set()

    for line in text.split('\n'):
        line_stripped = line.strip()

        # Check for group declaration: [groupname]
        if line_stripped.startswith('[') and line_stripped.endswith(']'):
            current_group = line_stripped[1:-1].strip()
            continue

        # Check for include directive: :include path
        if line_stripped.startswith(':include '):
            include_path_str = line_stripped[9:].strip()
            if include_path_str:
                # Resolve relative to base_path if provided
                if base_path:
                    include_path = base_path / include_path_str
                else:
                    include_path = Path(include_path_str)

                # Resolve to absolute path for cycle detection
                abs_path = include_path.resolve()
                if abs_path in _included_files:
                    raise ValueError(f"Circular include detected: {include_path}")

                if include_path.exists():
                    _included_files.add(abs_path)
                    included_rules = load_rules_from_file(
                        include_path,
                        _included_files=_included_files
                    )
                    # Apply current group to included rules that don't have tags
                    for meta, rule in included_rules:
                        if current_group and not meta.tags:
                            meta.tags.append(current_group)
                    rules.extend(included_rules)
                else:
                    raise FileNotFoundError(f"Include file not found: {include_path}")
            continue

        results = parse_rule_line(line)
        if results:
            for metadata, pattern, skeleton in results:
                # Add current group to tags if set
                if current_group and current_group not in metadata.tags:
                    metadata.tags.append(current_group)
                rules.append((metadata, [pattern, skeleton]))
    return rules


def load_rules_from_file(
    path: Union[str, Path],
    _included_files: Optional[set] = None
) -> List[Tuple[RuleMetadata, List]]:
    """
    Load rules from a .rules or .json file.

    Supports :include directives for DSL files, resolving paths
    relative to the containing file.

    Args:
        path: Path to the rules file
        _included_files: Internal tracking for circular include detection

    Returns:
        List of (metadata, [pattern, skeleton]) tuples
    """
    path = Path(path)
    text = path.read_text()

    if path.suffix == '.json':
        return load_rules_from_json(text)
    else:
        # Pass the parent directory as base_path for resolving includes
        return load_rules_from_dsl(
            text,
            base_path=path.parent,
            _included_files=_included_files
        )


def load_rules_from_json(text: str) -> List[Tuple[RuleMetadata, List]]:
    """
    Load rules from JSON text.

    Expected format:
        {
            "name": "ruleset-name",
            "description": "optional ruleset description",
            "rules": [
                {
                    "name": "rule-name",
                    "description": "...",
                    "pattern": [...],
                    "skeleton": [...],
                    "priority": 100,  # optional
                    "condition": [...],  # optional guard expression
                    "tags": ["group1"]  # optional
                },
                or just [pattern, skeleton]
            ]
        }
    """
    data = json.loads(text)
    rules = []

    for rule in data.get('rules', []):
        if isinstance(rule, dict):
            pattern = rule['pattern']
            skeleton = rule['skeleton']
            # Bidirectional rules are stored once (with bidirectional=true) and
            # expanded back into -fwd/-rev pairs at load time, mirroring the
            # `<=>` desugaring path. This makes JSON roundtrip stable.
            if rule.get('bidirectional'):
                pairs = _build_bidirectional_rules(
                    base_name=rule.get('name'),
                    description=rule.get('description'),
                    priority=rule.get('priority', 0),
                    condition=rule.get('condition'),
                    tags=rule.get('tags'),
                    pattern=pattern,
                    skeleton=skeleton,
                )
                for meta, pat, skel in pairs:
                    rules.append((meta, [pat, skel]))
                continue
            metadata = RuleMetadata(
                name=rule.get('name'),
                description=rule.get('description'),
                tags=rule.get('tags'),
                priority=rule.get('priority', 0),
                condition=rule.get('condition'),
            )
        else:
            metadata = RuleMetadata()
            pattern, skeleton = rule[0], rule[1]
        rules.append((metadata, [pattern, skeleton]))

    return rules


# RewriteStep, RewriteTrace, and the listener protocol live in `trace.py`.
# Re-exported below for backward compatibility.
from .trace import RewriteStep, RewriteTrace, TraceListener


class EqualityProof:
    """
    Proof that two expressions are equivalent.

    Contains the common form found and the paths from each expression
    to that common form.

    Attributes:
        expr_a: First expression
        expr_b: Second expression
        common: The common equivalent form found
        depth_a: Number of rewrite steps from expr_a to common
        depth_b: Number of rewrite steps from expr_b to common
        path_a: List of expressions from expr_a to common (if traced)
        path_b: List of expressions from expr_b to common (if traced)
    """

    def __init__(
        self,
        expr_a: ExprType,
        expr_b: ExprType,
        common: ExprType,
        depth_a: int,
        depth_b: int,
        path_a: Optional[List[ExprType]] = None,
        path_b: Optional[List[ExprType]] = None
    ):
        self.expr_a = expr_a
        self.expr_b = expr_b
        self.common = common
        self.depth_a = depth_a
        self.depth_b = depth_b
        self.path_a = path_a
        self.path_b = path_b

    @property
    def total_depth(self) -> int:
        """Total rewrite steps in the proof."""
        return self.depth_a + self.depth_b

    def __repr__(self) -> str:
        a_str = format_sexpr(self.expr_a)
        b_str = format_sexpr(self.expr_b)
        c_str = format_sexpr(self.common)
        return (f"EqualityProof({a_str} ≡ {b_str} via {c_str}, "
                f"depth={self.depth_a}+{self.depth_b})")

    def __bool__(self) -> bool:
        """Proofs are always truthy (existence means equality proved)."""
        return True

    def format(self, style: str = "brief") -> str:
        """
        Format the proof for display.

        Args:
            style: "brief" (default), "paths", or "full"
        """
        a_str = format_sexpr(self.expr_a)
        b_str = format_sexpr(self.expr_b)
        c_str = format_sexpr(self.common)

        if style == "brief":
            return f"{a_str} ≡ {b_str} (via {c_str})"

        elif style == "paths":
            lines = [f"{a_str} ≡ {b_str}"]
            lines.append(f"Common form: {c_str}")
            lines.append(f"Distance from A: {self.depth_a} steps")
            lines.append(f"Distance from B: {self.depth_b} steps")
            return "\n".join(lines)

        else:  # full
            lines = [f"Proof: {a_str} ≡ {b_str}"]
            lines.append(f"Common form: {c_str}")

            if self.path_a:
                lines.append(f"\nPath from A ({self.depth_a} steps):")
                for i, expr in enumerate(self.path_a):
                    lines.append(f"  {i}. {format_sexpr(expr)}")

            if self.path_b:
                lines.append(f"\nPath from B ({self.depth_b} steps):")
                for i, expr in enumerate(self.path_b):
                    lines.append(f"  {i}. {format_sexpr(expr)}")

            return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        result = {
            "expr_a": self.expr_a,
            "expr_b": self.expr_b,
            "common": self.common,
            "depth_a": self.depth_a,
            "depth_b": self.depth_b,
            "total_depth": self.total_depth,
        }
        if self.path_a:
            result["path_a"] = self.path_a
        if self.path_b:
            result["path_b"] = self.path_b
        return result


class RuleSet:
    """An immutable view over a subset of an engine's rules.

    Filters compose: ``engine.rule_set().bidirectional_only().in_groups(["algebra"])``.
    Iteration yields ``(rule_idx, rule, metadata)`` triples in original
    insertion order.

    A ``RuleSet`` can be passed to equivalence-class methods
    (``equivalents``, ``prove_equal``, ``minimize``, ``random_walk``, etc.)
    via the ``rules=`` keyword to control which rules are considered without
    cluttering each method with separate ``include_unidirectional`` /
    ``groups`` knobs.
    """

    __slots__ = ("_engine", "_predicate")

    def __init__(self, engine: "RuleEngine",
                 predicate: Optional[Callable[["RuleMetadata"], bool]] = None):
        self._engine = engine
        self._predicate = predicate or (lambda m: True)

    def __iter__(self) -> Iterator[Tuple[int, "RuleType", "RuleMetadata"]]:
        for idx, (rule, meta) in enumerate(
            zip(self._engine._rules, self._engine._metadata)
        ):
            if self._predicate(meta):
                yield idx, rule, meta

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __bool__(self) -> bool:
        for _ in self:
            return True
        return False

    def _and(self, extra: Callable[["RuleMetadata"], bool]) -> "RuleSet":
        prev = self._predicate
        return RuleSet(self._engine, lambda m: prev(m) and extra(m))

    def bidirectional_only(self) -> "RuleSet":
        """Restrict to rules originating from `<=>` declarations."""
        return self._and(lambda m: m.bidirectional)

    def unidirectional_only(self) -> "RuleSet":
        """Restrict to plain `=>` rules."""
        return self._and(lambda m: not m.bidirectional)

    def in_groups(self, groups: Optional[List[str]]) -> "RuleSet":
        """Restrict to rules whose tags overlap with ``groups``.

        Rules with no tags are kept (they are universal). When ``groups``
        is None, returns ``self`` unchanged.
        """
        if groups is None:
            return self
        groups_set = set(groups)
        return self._and(
            lambda m: not m.tags or any(t in groups_set for t in m.tags)
        )

    def excluding_disabled(self, disabled_groups: Set[str]) -> "RuleSet":
        """Drop rules whose tags overlap with the engine's disabled groups."""
        if not disabled_groups:
            return self
        return self._and(
            lambda m: not m.tags or not any(t in disabled_groups for t in m.tags)
        )


class EquivalenceClass:
    """The equivalence class of an expression under a rule set.

    Eight engine methods (``equivalents``, ``enumerate_equivalents``,
    ``prove_equal``, ``are_equal``, ``minimize``, ``random_equivalent``,
    ``sample_equivalents``, ``random_walk``) all answered the same
    underlying question: "what is reachable from this expression under
    these rules?" Each one took ``expr`` and ``rules``-related kwargs.
    ``EquivalenceClass`` captures both at construction; the methods then
    operate on the implicit class.

    Construct via :meth:`RuleEngine.equivalence_class`. The starting
    expression and rule subset are immutable on the value object.

    Example::

        cls = engine.equivalence_class(["+", "a", "b"])
        assert cls.contains(["+", "b", "a"])
        result = cls.minimum(metric="size")
        for form in cls.iter(max_depth=3):
            ...
    """

    __slots__ = ("_engine", "_expr", "_rules")

    def __init__(self, engine: "RuleEngine", expr: ExprType,
                 rules: Optional[RuleSet] = None):
        self._engine = engine
        self._expr = expr
        # Default rules: full active set with bidirectional only, matching the
        # historical default of `equivalents` and friends. Callers wanting
        # `=>`-rules-too explicitly pass ``engine.rule_set()``.
        self._rules = rules if rules is not None else engine.rule_set(bidirectional_only=True)

    @property
    def expr(self) -> ExprType:
        return self._expr

    @property
    def rules(self) -> RuleSet:
        return self._rules

    def iter(self, max_depth: int = 10, max_count: Optional[int] = None,
             strategy: str = "bfs") -> Iterator[ExprType]:
        """Yield equivalents lazily; see ``RuleEngine.equivalents``."""
        return self._engine.equivalents(
            self._expr, max_depth=max_depth, max_count=max_count,
            strategy=strategy, rules=self._rules,
        )

    def enumerate(self, max_depth: int = 10,
                   max_count: Optional[int] = None) -> List[ExprType]:
        """Eagerly collect equivalents into a list."""
        return list(self.iter(max_depth=max_depth, max_count=max_count))

    def contains(self, other: ExprType, max_depth: int = 10,
                  max_expressions: Optional[int] = None) -> bool:
        """True iff ``other`` is in this equivalence class."""
        return self._engine.are_equal(
            self._expr, other, max_depth=max_depth,
            max_expressions=max_expressions, rules=self._rules,
        )

    def prove(self, other: ExprType, max_depth: int = 10,
               max_expressions: Optional[int] = None,
               trace: bool = False) -> Optional["EqualityProof"]:
        """Return an ``EqualityProof`` if ``other`` is equivalent, else None."""
        return self._engine.prove_equal(
            self._expr, other, max_depth=max_depth,
            max_expressions=max_expressions, trace=trace, rules=self._rules,
        )

    def minimum(self, cost: Optional[Callable[[ExprType], float]] = None,
                metric: Optional[str] = None,
                op_costs: Optional[Dict[str, float]] = None,
                max_depth: int = 10,
                max_count: Optional[int] = 10000) -> "OptimizationResult":
        """Find the minimum-cost equivalent; see ``RuleEngine.minimize``."""
        return self._engine.minimize(
            self._expr, cost=cost, metric=metric, op_costs=op_costs,
            max_depth=max_depth, max_count=max_count, rules=self._rules,
        )

    def sample(self, n: int = 10, steps: int = 10, unique: bool = True,
               max_attempts: int = 100,
               rng: "Optional[random.Random]" = None) -> List[ExprType]:
        """Sample n random equivalents via random walk."""
        return self._engine.sample_equivalents(
            self._expr, n=n, steps=steps, unique=unique,
            max_attempts=max_attempts, rng=rng, rules=self._rules,
        )

    def walk(self, max_steps: int = 100,
             rng: "Optional[random.Random]" = None) -> Iterator[ExprType]:
        """Lazy random walk through the class."""
        return self._engine.random_walk(
            self._expr, max_steps=max_steps, rng=rng, rules=self._rules,
        )

    def random(self, steps: int = 10,
               rng: "Optional[random.Random]" = None) -> ExprType:
        """Get a single random equivalent via ``steps`` random rewrites.

        Note: defining ``random`` last means earlier methods' annotations
        referring to ``random.Random`` resolve to the module, not this method.
        """
        return self._engine.random_equivalent(
            self._expr, steps=steps, rng=rng, rules=self._rules,
        )

    def __contains__(self, other: ExprType) -> bool:
        """Membership test: ``other in cls`` (uses default depth)."""
        return self.contains(other)

    def __iter__(self) -> Iterator[ExprType]:
        """Default iteration uses BFS with default depth."""
        return self.iter()

    def __repr__(self) -> str:
        return f"EquivalenceClass({format_sexpr(self._expr)}, rules={len(self._rules)} rules)"


class RuleEngine:
    """
    A rule engine that loads and applies rewriting rules.

    Supports loading rules from DSL files, JSON files, or Python lists.
    Provides optional tracing to see which rules are applied.

    By default, this is a pure rule rewriter with no built-in evaluation.
    To enable constant folding, pass a prelude via fold_funcs.

    Example:
        from rerum import RuleEngine, ARITHMETIC_PRELUDE, MATH_PRELUDE

        # Pure rule rewriting (no constant folding)
        engine = RuleEngine.from_dsl('''
            @add-zero "Adding zero has no effect": (+ ?x 0) => :x
            @mul-one: (* ?x 1) => :x
        ''')
        result = engine(expr)

        # With arithmetic constant folding (+, -, *, /, ^)
        engine = RuleEngine.from_dsl(rules, fold_funcs=ARITHMETIC_PRELUDE)

        # With math function folding (sin, cos, exp, log, etc.)
        engine = RuleEngine(fold_funcs=MATH_PRELUDE)
    """

    def __init__(self, fold_funcs: Optional[FoldFuncsType] = None):
        """
        Initialize a RuleEngine.

        Args:
            fold_funcs: Optional fold functions for constant folding.
                Default: None (pure rule rewriting, no constant folding).
                Use ARITHMETIC_PRELUDE for basic +, -, *, /, ^.
                Use MATH_PRELUDE for arithmetic + trig/exp/log.
        """
        self._rules: List[List] = []
        self._metadata: List[RuleMetadata] = []
        self._rule_names: Dict[str, int] = {}  # Maps name -> index
        self._simplifier = None
        self._fold_funcs: Optional[FoldFuncsType] = fold_funcs
        self._disabled_groups: set = set()  # Groups that are disabled
        self._hooks = _HookRegistry()

    def _sort_by_priority(self) -> None:
        """Sort rules by priority (descending). Higher priority fires first.

        Uses stable sort, so rules with equal priority maintain their relative order.
        """
        if not self._rules:
            return

        # Create list of (priority, original_index, rule, metadata)
        indexed = [(self._metadata[i].priority, i, self._rules[i], self._metadata[i])
                   for i in range(len(self._rules))]

        # Sort by priority descending, then by original index ascending (stable)
        indexed.sort(key=lambda x: (-x[0], x[1]))

        # Rebuild lists
        self._rules = [item[2] for item in indexed]
        self._metadata = [item[3] for item in indexed]

        # Rebuild name index
        self._rule_names = {}
        for idx, meta in enumerate(self._metadata):
            if meta.name:
                self._rule_names[meta.name] = idx

    def load_dsl(self, text: str) -> 'RuleEngine':
        """Load rules from DSL text."""
        parsed = load_rules_from_dsl(text)
        for metadata, rule in parsed:
            idx = len(self._rules)
            self._rules.append(rule)
            self._metadata.append(metadata)
            if metadata.name:
                self._rule_names[metadata.name] = idx
        self._sort_by_priority()
        self._simplifier = None  # Invalidate cached simplifier
        return self

    def load_file(self, path: Union[str, Path]) -> 'RuleEngine':
        """Load rules from a file (.rules or .json)."""
        parsed = load_rules_from_file(path)
        for metadata, rule in parsed:
            idx = len(self._rules)
            self._rules.append(rule)
            self._metadata.append(metadata)
            if metadata.name:
                self._rule_names[metadata.name] = idx
        self._sort_by_priority()
        self._simplifier = None
        return self

    def load_rules(self, rules: List[List]) -> 'RuleEngine':
        """Load rules from a Python list (without metadata)."""
        for rule in rules:
            self._rules.append(rule)
            self._metadata.append(RuleMetadata())
        self._sort_by_priority()
        self._simplifier = None
        return self

    def with_prelude(self, fold_funcs: FoldFuncsType) -> 'RuleEngine':
        """
        Set the prelude (fold functions) for constant folding.

        Enables fluent construction:
            engine = RuleEngine().with_prelude(ARITHMETIC_PRELUDE).load_dsl(...)

        Args:
            fold_funcs: Fold functions dict for constant folding.
                Use ARITHMETIC_PRELUDE for basic +, -, *, /, ^.
                Use MATH_PRELUDE for arithmetic + trig/exp/log.
                Use a custom dict for your own operations.

        Returns:
            self for chaining
        """
        self._fold_funcs = fold_funcs
        self._simplifier = None  # Invalidate cached simplifier
        return self

    def add_rule(self, pattern: ExprType, skeleton: ExprType,
                 name: Optional[str] = None,
                 description: Optional[str] = None) -> 'RuleEngine':
        """Add a single rule with optional metadata."""
        rule = [pattern, skeleton]
        idx = len(self._rules)
        self._rules.append(rule)
        metadata = RuleMetadata(name=name, description=description)
        self._metadata.append(metadata)
        if name:
            self._rule_names[name] = idx
        self._simplifier = None
        return self

    def get_rule(self, name: str) -> Optional[Tuple[List, RuleMetadata]]:
        """Get a rule and its metadata by name."""
        if name in self._rule_names:
            idx = self._rule_names[name]
            return self._rules[idx], self._metadata[idx]
        return None

    def get_metadata(self, index: int) -> RuleMetadata:
        """Get metadata for a rule by index."""
        return self._metadata[index] if index < len(self._metadata) else RuleMetadata()

    # ============================================================
    # Group Management
    # ============================================================

    def disable_group(self, group: str) -> 'RuleEngine':
        """Disable all rules in a group."""
        self._disabled_groups.add(group)
        self._simplifier = None
        return self

    def enable_group(self, group: str) -> 'RuleEngine':
        """Enable all rules in a group."""
        self._disabled_groups.discard(group)
        self._simplifier = None
        return self

    def groups(self) -> set:
        """Return all group names used by rules."""
        all_groups = set()
        for meta in self._metadata:
            all_groups.update(meta.tags)
        return all_groups

    def _is_rule_active(self, metadata: RuleMetadata, groups: Optional[List[str]] = None) -> bool:
        """Check if a rule should be applied given current group settings.

        Args:
            metadata: The rule's metadata
            groups: If specified, only rules in these groups are active.
                    If None, use the disabled_groups setting.

        Returns:
            True if the rule should be considered, False if it should be skipped.
        """
        if groups is not None:
            # Explicit groups specified - rule must be in one of them
            # Rules without groups are always active when explicit groups given
            if not metadata.tags:
                return True
            return any(g in groups for g in metadata.tags)
        else:
            # Use disabled_groups - rule is active unless in a disabled group
            if not metadata.tags:
                return True
            return not any(g in self._disabled_groups for g in metadata.tags)

    def match(self, pattern: Union[str, ExprType], expr: ExprType) -> Union[Bindings, _NoMatch]:
        """
        Match a pattern against an expression.

        Returns Bindings if matched, NoMatch if not.

        Args:
            pattern: Pattern to match (string or list). If string, parsed as s-expression.
            expr: Expression to match against.

        Returns:
            Bindings object with dict-like access if matched, NoMatch (falsy) if not.

        Example:
            if bindings := engine.match("(+ ?a ?b)", expr):
                print(bindings["a"], bindings["b"])
        """
        # Parse pattern if string
        if isinstance(pattern, str):
            pattern = parse_sexpr(pattern)

        # Use internal match function. Public API still returns
        # ``Bindings | NoMatch`` for backward compatibility; internally the
        # match returns ``Bindings | None`` and wrap_bindings normalizes
        # the failure case into the public ``NoMatch`` sentinel.
        result = _match_internal(pattern, expr)
        return wrap_bindings(result)

    def _check_condition(self, condition: Optional[ExprType], bindings) -> bool:
        """
        Check if a rule's condition is satisfied.

        Args:
            condition: The condition expression (or None for unconditional rules)
            bindings: The bindings from pattern matching

        Returns:
            True if condition is satisfied (or no condition), False otherwise
        """
        if condition is None:
            return True

        # Instantiate the condition with bindings
        result = instantiate(condition, bindings, self._fold_funcs)

        # Check truthiness
        # Numbers: 0 is falsy
        # Strings: empty is falsy
        # Lists: empty is falsy
        # Booleans: as expected
        if isinstance(result, bool):
            return result
        if isinstance(result, (int, float)):
            return result != 0
        if isinstance(result, str):
            return len(result) > 0
        if isinstance(result, list):
            return len(result) > 0
        # Default: truthy
        return True

    def source_rules(self) -> Iterator[Union["BidirectionalRule", "UnidirectionalRule"]]:
        """Iterate the engine's rules as a sequence of *source* rules.

        Where ``len(engine)`` counts rule storage entries (a ``<=>`` rule
        contributes 2: the ``-fwd`` and ``-rev``), ``source_rules()`` yields
        one :class:`BidirectionalRule` per ``<=>`` declaration and one
        :class:`UnidirectionalRule` per ``=>`` declaration. Useful for
        listing, exporting, or counting "rules I wrote" rather than "rules
        in storage."

        Detection is by adjacency and the ``-fwd``/``-rev`` naming convention,
        matching how :meth:`to_dsl` and :meth:`to_dict` already collapse pairs.
        Rules added programmatically without the convention are emitted as
        unidirectional.
        """
        i = 0
        while i < len(self._rules):
            meta = self._metadata[i]
            rule = self._rules[i]
            pattern, skeleton = rule

            if _is_bidirectional_pair(self._metadata, i):
                base_name, base_description = _strip_bidirectional_naming(meta)
                rev_rule = self._rules[i + 1]
                rev_pattern, rev_skeleton = rev_rule
                yield BidirectionalRule(
                    name=base_name,
                    description=base_description,
                    priority=meta.priority,
                    condition=meta.condition,
                    tags=meta.tags,
                    fwd_pattern=pattern,
                    fwd_skeleton=skeleton,
                    rev_pattern=rev_pattern,
                    rev_skeleton=rev_skeleton,
                )
                i += 2
                continue

            yield UnidirectionalRule(
                name=meta.name,
                description=meta.description,
                priority=meta.priority,
                condition=meta.condition,
                tags=meta.tags,
                pattern=pattern,
                skeleton=skeleton,
            )
            i += 1

    def equivalence_class(
        self,
        expr: ExprType,
        *,
        rules: Optional[RuleSet] = None,
    ) -> "EquivalenceClass":
        """Return an :class:`EquivalenceClass` value object for ``expr``.

        The default ``rules`` is ``self.rule_set(bidirectional_only=True)``,
        matching the historical default of ``equivalents``/``prove_equal``
        and friends (strict equivalence only). Pass ``rules=self.rule_set()``
        for the full set including ``=>`` rules.

        Example::

            cls = engine.equivalence_class(["+", "a", "b"])
            assert cls.contains(["+", "b", "a"])
            cls.minimum(metric="size")
            for form in cls.iter(max_depth=3):
                ...
        """
        return EquivalenceClass(self, expr, rules=rules)

    def rule_set(
        self,
        *,
        groups: Optional[List[str]] = None,
        bidirectional_only: bool = False,
    ) -> RuleSet:
        """Return a ``RuleSet`` view over the engine's currently active rules.

        By default the view excludes rules in disabled groups (matching the
        behavior of strategy methods). Optional filters layer on top.

        Use ``rules=engine.rule_set().bidirectional_only()`` (or
        ``engine.rule_set(bidirectional_only=True)``) to restrict any
        equivalence-class method to strict equivalence rules.
        """
        rs = RuleSet(self).excluding_disabled(self._disabled_groups)
        if groups is not None:
            rs = rs.in_groups(groups)
        if bidirectional_only:
            rs = rs.bidirectional_only()
        return rs

    def apply_once(self, expr: ExprType, groups: Optional[List[str]] = None) -> Tuple[ExprType, Optional[RuleMetadata]]:
        """
        Apply at most one rule to the expression.

        Tries each rule in order and returns after the first successful application.
        Does not recurse into subexpressions. Respects conditional guards and group filters.

        Args:
            expr: Expression to rewrite.
            groups: If specified, only use rules from these groups.
                    If None, use all rules except those in disabled groups.

        Returns:
            Tuple of (result, metadata) where:
                - result: The rewritten expression (or original if no rule applied)
                - metadata: RuleMetadata of applied rule, or None if no rule applied

        Example:
            result, applied = engine.apply_once(expr)
            if applied:
                print(f"Applied rule: {applied.name}")
        """
        for rule_idx, rule in enumerate(self._rules):
            metadata = self._metadata[rule_idx]
            # Check group filter
            if not self._is_rule_active(metadata, groups):
                continue
            pattern, skeleton = rule
            bindings = _match_internal(pattern, expr)
            if bindings is not None:
                # Check condition if present
                if not self._check_condition(metadata.condition, bindings):
                    continue
                result = instantiate(skeleton, bindings, self._fold_funcs)
                return result, metadata

        return expr, None

    def rules_matching(self, expr: ExprType, check_conditions: bool = True,
                        groups: Optional[List[str]] = None) -> List[Tuple[RuleMetadata, Bindings]]:
        """
        Find all rules that could apply to an expression.

        Useful for debugging and understanding why an expression isn't simplifying.

        Args:
            expr: Expression to check.
            check_conditions: If True (default), only return rules whose conditions
                are satisfied. If False, return all pattern-matching rules.
            groups: If specified, only check rules from these groups.
                    If None, use all rules except those in disabled groups.

        Returns:
            List of (metadata, bindings) for each rule that matches.

        Example:
            for meta, bindings in engine.rules_matching(expr):
                print(f"Rule {meta.name} matches with {bindings.to_dict()}")
        """
        matching = []
        for rule_idx, rule in enumerate(self._rules):
            metadata = self._metadata[rule_idx]
            # Check group filter
            if not self._is_rule_active(metadata, groups):
                continue
            pattern, skeleton = rule
            bindings = _match_internal(pattern, expr)
            if bindings is not None:
                # Check condition if requested
                if check_conditions and not self._check_condition(metadata.condition, bindings):
                    continue
                matching.append((metadata, bindings))
        return matching

    @property
    def rules(self) -> List[List]:
        """Get all loaded rules."""
        return self._rules.copy()

    def simplify(
        self,
        expr: ExprType,
        trace: bool = False,
        max_steps: int = 1000,
        strategy: str = "exhaustive",
        groups: Optional[List[str]] = None
    ):
        """
        Simplify an expression using all loaded rules.

        Constant folding is controlled by the fold_funcs parameter passed
        to the constructor. By default, no constant folding is performed.

        Args:
            expr: Expression to simplify
            trace: If True, return (result, trace) tuple
            max_steps: Maximum rewrite steps (default: 1000)
            strategy: Rewriting strategy (default: "exhaustive")
                - "exhaustive": Apply rules repeatedly until no more apply (default)
                - "once": Apply at most one rule anywhere in the expression
                - "bottomup": Simplify children first, then parent, repeat until fixpoint
                - "topdown": Try to simplify parent first, then children, repeat until fixpoint
            groups: If specified, only use rules from these groups.
                    If None, use all rules except those in disabled groups.

        Returns:
            Simplified expression, or (expression, trace) if trace=True
        """
        if trace:
            return self._simplify_with_trace(expr, max_steps, groups=groups)

        # Check if we need slow path (conditions or groups)
        has_conditions = any(m.condition is not None for m in self._metadata)
        has_groups = groups is not None or self._disabled_groups

        if strategy == "exhaustive":
            if has_conditions or has_groups:
                return self._simplify_exhaustive(expr, max_steps, groups=groups)
            else:
                # Use fast path when no conditions or groups
                if self._simplifier is None:
                    self._simplifier = rewriter(self._rules, fold_funcs=self._fold_funcs)
                return self._simplifier(expr)
        elif strategy == "once":
            return self._simplify_once(expr, groups=groups)
        elif strategy == "bottomup":
            return self._simplify_bottomup(expr, max_steps, groups=groups)
        elif strategy == "topdown":
            return self._simplify_topdown(expr, max_steps, groups=groups)
        else:
            raise ValueError(f"Unknown strategy: {strategy}. "
                           f"Valid options: exhaustive, once, bottomup, topdown")

    def _simplify_once(self, expr: ExprType, groups: Optional[List[str]] = None) -> ExprType:
        """Apply at most one rule anywhere in the expression tree."""
        # Try to apply a rule at the top level
        result, applied = self.apply_once(expr, groups=groups)
        if applied:
            return result

        # If no rule applied at top level, try children (depth-first)
        if isinstance(expr, list) and len(expr) > 0:
            for i, child in enumerate(expr):
                new_child = self._simplify_once(child, groups=groups)
                if new_child != child:
                    # Found a rewrite - apply it and return
                    return expr[:i] + [new_child] + expr[i+1:]

        return expr

    def _simplify_exhaustive(self, expr: ExprType, max_steps: int,
                              groups: Optional[List[str]] = None,
                              listener: Optional[TraceListener] = None) -> ExprType:
        """Exhaustive strategy with condition, group, and listener support.

        Uses a visited set to terminate on cycles (e.g. bidirectional rules
        like ``(+ ?x ?y) <=> (+ :y :x)`` that bounce between two equivalent
        forms). Without this, max_steps would simply bound an oscillation.

        If ``listener`` is provided, it is invoked with each successful
        ``RewriteStep``. This is how ``simplify(trace=True)`` accumulates
        a trace without duplicating the rule loop.
        """
        current = expr
        visited = set()
        for _ in range(max_steps):
            key = _expr_to_tuple(current)
            if key in visited:
                break
            visited.add(key)
            changed = False

            # Try rules at top level
            for rule_idx, rule in enumerate(self._rules):
                metadata = self._metadata[rule_idx]
                # Check group filter
                if not self._is_rule_active(metadata, groups):
                    continue
                pattern, skeleton = rule
                bindings = _match_internal(pattern, current)
                if bindings is not None:
                    if not self._check_condition(metadata.condition, bindings):
                        continue
                    new_expr = instantiate(skeleton, bindings, self._fold_funcs)
                    if new_expr != current:
                        if listener is not None:
                            listener(RewriteStep(
                                rule_index=rule_idx,
                                metadata=metadata,
                                before=current,
                                after=new_expr,
                            ))
                        current = new_expr
                        changed = True
                        break

            if not changed:
                # Recursively simplify subexpressions
                if isinstance(current, list) and len(current) > 0:
                    new_children = []
                    subexpr_changed = False
                    for child in current:
                        new_child = self._simplify_exhaustive(
                            child, max_steps // 10 or 1, groups=groups, listener=listener,
                        )
                        new_children.append(new_child)
                        if new_child != child:
                            subexpr_changed = True
                    if subexpr_changed:
                        current = new_children
                        continue
                break

        return current

    def _simplify_bottomup(self, expr: ExprType, max_steps: int, groups: Optional[List[str]] = None) -> ExprType:
        """Bottom-up strategy: simplify children first, then parent.

        Cycle-detected so non-confluent rule sets terminate cleanly.
        """
        visited = set()
        for _ in range(max_steps):
            key = _expr_to_tuple(expr)
            if key in visited:
                break
            visited.add(key)
            new_expr = self._bottomup_pass(expr, groups=groups)
            if new_expr == expr:
                break
            expr = new_expr
        return expr

    def _bottomup_pass(self, expr: ExprType, groups: Optional[List[str]] = None) -> ExprType:
        """Single bottom-up pass: simplify children, then apply rules to parent."""
        # Base case: atoms can't be simplified structurally
        if not isinstance(expr, list) or len(expr) == 0:
            return expr

        # First, recursively simplify all children
        new_children = [self._bottomup_pass(child, groups=groups) for child in expr]
        current = new_children

        # Then try to apply rules to this node
        for rule_idx, rule in enumerate(self._rules):
            metadata = self._metadata[rule_idx]
            # Check group filter
            if not self._is_rule_active(metadata, groups):
                continue
            pattern, skeleton = rule
            bindings = _match_internal(pattern, current)
            if bindings is not None:
                # Check condition if present
                if not self._check_condition(metadata.condition, bindings):
                    continue
                result = instantiate(skeleton, bindings, self._fold_funcs)
                if result != current:
                    return result

        return current

    def _simplify_topdown(self, expr: ExprType, max_steps: int, groups: Optional[List[str]] = None) -> ExprType:
        """Top-down strategy: try parent first, then children.

        Cycle-detected so non-confluent rule sets terminate cleanly.
        """
        visited = set()
        for _ in range(max_steps):
            key = _expr_to_tuple(expr)
            if key in visited:
                break
            visited.add(key)
            new_expr = self._topdown_pass(expr, groups=groups)
            if new_expr == expr:
                break
            expr = new_expr
        return expr

    def _topdown_pass(self, expr: ExprType, groups: Optional[List[str]] = None) -> ExprType:
        """Single top-down pass: apply rules to parent, then simplify children."""
        # Try to apply rules at this node first
        current = expr
        for rule_idx, rule in enumerate(self._rules):
            metadata = self._metadata[rule_idx]
            # Check group filter
            if not self._is_rule_active(metadata, groups):
                continue
            pattern, skeleton = rule
            bindings = _match_internal(pattern, current)
            if bindings is not None:
                # Check condition if present
                if not self._check_condition(metadata.condition, bindings):
                    continue
                result = instantiate(skeleton, bindings, self._fold_funcs)
                if result != current:
                    return result  # Return immediately - will be called again

        # No rule applied - recursively process children
        if isinstance(current, list) and len(current) > 0:
            new_children = [self._topdown_pass(child, groups=groups) for child in current]
            if new_children != list(current):
                return new_children

        return current

    def _simplify_with_trace(self, expr: ExprType, max_steps: int,
                             groups: Optional[List[str]] = None) -> Tuple[ExprType, RewriteTrace]:
        """Traced simplification, implemented as a thin wrapper around
        ``_simplify_exhaustive`` with a ``RewriteTrace`` listener.

        Pre-refactor this was a separate near-duplicate rule loop. Pulling
        the trace into a listener removes the duplication and makes the
        same trace machinery available to other callers
        (``equivalents``, ``minimize``, etc. could pass a listener through).
        """
        trace_obj = RewriteTrace()
        trace_obj.initial = expr
        result = self._simplify_exhaustive(expr, max_steps, groups=groups,
                                            listener=trace_obj)
        if self._fold_funcs:
            result = self._fold_constants(result)
        trace_obj.final = result
        return result, trace_obj

    def _fold_constants(self, expr: ExprType) -> ExprType:
        """Fold constant expressions using the configured fold_funcs."""
        if not isinstance(expr, list) or len(expr) == 0:
            return expr

        # Recursively fold subexpressions
        folded = [expr[0]] + [self._fold_constants(e) for e in expr[1:]]

        # Try to evaluate if all args are constants
        op = folded[0]
        args = folded[1:]

        # Check if we have a fold function for this operator
        if op not in self._fold_funcs:
            return folded

        if all(isinstance(a, (int, float)) for a in args):
            try:
                handler = self._fold_funcs[op]
                result = handler(args)

                # None means can't fold (wrong arity, etc.)
                if result is None:
                    return folded

                # Preserve integer type when possible
                if isinstance(result, float) and result.is_integer():
                    result = int(result)
                return result
            except (ValueError, OverflowError):
                pass

        return folded

    def clear(self) -> 'RuleEngine':
        """Clear all rules."""
        self._rules = []
        self._metadata = []
        self._rule_names = {}
        self._simplifier = None
        return self

    def list_rules(self) -> List[str]:
        """List all rules with their metadata in DSL format."""
        result = []
        for idx, (rule, meta) in enumerate(zip(self._rules, self._metadata)):
            pattern, skeleton = rule
            pattern_str = format_sexpr(pattern)
            skeleton_str = format_sexpr(skeleton)

            # Build rule name with optional priority
            if meta.name:
                if meta.priority != 0:
                    name_part = f"@{meta.name}[{meta.priority}]"
                else:
                    name_part = f"@{meta.name}"
                if meta.description:
                    name_part += f" \"{meta.description}\""
                name_part += ": "
            else:
                name_part = ""

            # Build rule body
            rule_str = f"{name_part}{pattern_str} => {skeleton_str}"

            # Add condition if present
            if meta.condition:
                rule_str += f" when {format_sexpr(meta.condition)}"

            result.append(rule_str)
        return result

    def to_dsl(self, name: Optional[str] = None) -> str:
        """
        Export rules to DSL format string.

        Adjacent ``-fwd``/``-rev`` pairs are collapsed back into a single
        ``<=>`` rule so that ``RuleEngine.from_dsl(engine.to_dsl())`` is a
        roundtrip-stable identity for bidirectional rules.

        Args:
            name: Optional name to include as a comment header

        Returns:
            DSL-formatted string with all rules, organized by groups.
        """
        lines = []
        if name:
            lines.append(f"# {name}")
            lines.append("")

        current_group = None
        i = 0
        while i < len(self._rules):
            meta = self._metadata[i]
            rule = self._rules[i]

            rule_group = meta.tags[0] if meta.tags else None
            if rule_group != current_group:
                if rule_group:
                    if lines and lines[-1] != "":
                        lines.append("")
                    lines.append(f"[{rule_group}]")
                current_group = rule_group

            pattern, skeleton = rule
            pattern_str = format_sexpr(pattern)
            skeleton_str = format_sexpr(skeleton)

            if _is_bidirectional_pair(self._metadata, i):
                base_name, base_description = _strip_bidirectional_naming(meta)
                arrow = "<=>"
                if base_name:
                    name_part = f"@{base_name}"
                    if meta.priority != 0:
                        name_part += f"[{meta.priority}]"
                    if base_description:
                        name_part += f" \"{base_description}\""
                    name_part += ": "
                else:
                    name_part = ""
                rule_str = f"{name_part}{pattern_str} {arrow} {skeleton_str}"
                if meta.condition:
                    rule_str += f" when {format_sexpr(meta.condition)}"
                lines.append(rule_str)
                i += 2
                continue

            if meta.name:
                if meta.priority != 0:
                    name_part = f"@{meta.name}[{meta.priority}]"
                else:
                    name_part = f"@{meta.name}"
                if meta.description:
                    name_part += f" \"{meta.description}\""
                name_part += ": "
            else:
                name_part = ""

            rule_str = f"{name_part}{pattern_str} => {skeleton_str}"
            if meta.condition:
                rule_str += f" when {format_sexpr(meta.condition)}"

            lines.append(rule_str)
            i += 1

        return "\n".join(lines)

    def to_json(self, name: Optional[str] = None, description: Optional[str] = None,
                indent: Optional[int] = 2) -> str:
        """
        Export rules to JSON format string.

        Args:
            name: Optional ruleset name
            description: Optional ruleset description
            indent: JSON indentation (None for compact)

        Returns:
            JSON-formatted string compatible with load_rules_from_json().
        """
        rules_list = self._rules_as_dicts()

        result = {"rules": rules_list}
        if name:
            result["name"] = name
        if description:
            result["description"] = description

        return json.dumps(result, indent=indent)

    def to_dict(self) -> Dict:
        """
        Export rules to a dictionary.

        Adjacent ``-fwd``/``-rev`` pairs are collapsed into a single rule
        with ``"bidirectional": true`` so JSON roundtrip preserves the
        original `<=>` form.

        Returns:
            Dictionary compatible with JSON serialization.
        """
        return {"rules": self._rules_as_dicts()}

    def _rules_as_dicts(self) -> List[Dict]:
        """Build the JSON-shaped list of rule dicts.

        Adjacent ``-fwd``/``-rev`` pairs are emitted as a single dict with
        ``"bidirectional": true``; the loader expands them back into a pair
        via `_build_bidirectional_rules`.
        """
        rules_list: List[Dict] = []
        i = 0
        while i < len(self._rules):
            meta = self._metadata[i]
            rule = self._rules[i]
            pattern, skeleton = rule

            if _is_bidirectional_pair(self._metadata, i):
                base_name, base_description = _strip_bidirectional_naming(meta)
                rule_dict: Dict = {
                    "pattern": pattern,
                    "skeleton": skeleton,
                    "bidirectional": True,
                }
                if base_name:
                    rule_dict["name"] = base_name
                if base_description:
                    rule_dict["description"] = base_description
                if meta.priority != 0:
                    rule_dict["priority"] = meta.priority
                if meta.condition:
                    rule_dict["condition"] = meta.condition
                if meta.tags:
                    rule_dict["tags"] = meta.tags
                rules_list.append(rule_dict)
                i += 2
                continue

            rule_dict = {
                "pattern": pattern,
                "skeleton": skeleton,
            }
            if meta.name:
                rule_dict["name"] = meta.name
            if meta.description:
                rule_dict["description"] = meta.description
            if meta.priority != 0:
                rule_dict["priority"] = meta.priority
            if meta.condition:
                rule_dict["condition"] = meta.condition
            if meta.tags:
                rule_dict["tags"] = meta.tags
            rules_list.append(rule_dict)
            i += 1

        return rules_list

    def __len__(self) -> int:
        return len(self._rules)

    def __repr__(self) -> str:
        return f"RuleEngine({len(self._rules)} rules)"

    def __call__(self, expr: ExprType, **kwargs) -> ExprType:
        """Make engine callable: engine(expr) is shorthand for engine.simplify(expr)."""
        return self.simplify(expr, **kwargs)

    def __iter__(self):
        """Iterate over (rule, metadata) pairs."""
        return iter(zip(self._rules, self._metadata))

    def __contains__(self, name: str) -> bool:
        """Check if a named rule exists: 'add-zero' in engine."""
        return name in self._rule_names

    def __getitem__(self, name: str) -> Tuple[List, RuleMetadata]:
        """Get rule by name: engine['add-zero']."""
        if name not in self._rule_names:
            raise KeyError(f"No rule named '{name}'")
        idx = self._rule_names[name]
        return self._rules[idx], self._metadata[idx]

    # Class method constructors for fluent creation
    @classmethod
    def from_dsl(cls, text: str, fold_funcs: Optional[FoldFuncsType] = None) -> 'RuleEngine':
        """Create engine from DSL text."""
        return cls(fold_funcs=fold_funcs).load_dsl(text)

    @classmethod
    def from_file(cls, path: Union[str, Path], fold_funcs: Optional[FoldFuncsType] = None) -> 'RuleEngine':
        """Create engine from file."""
        return cls(fold_funcs=fold_funcs).load_file(path)

    @classmethod
    def from_rules(cls, rules: List[List], fold_funcs: Optional[FoldFuncsType] = None) -> 'RuleEngine':
        """Create engine from Python rule list."""
        return cls(fold_funcs=fold_funcs).load_rules(rules)

    # Combining engines (rule set algebra)
    def copy(self) -> 'RuleEngine':
        """Create a copy of this engine."""
        new_engine = RuleEngine(fold_funcs=self._fold_funcs)
        new_engine._rules = self._rules.copy()
        new_engine._metadata = self._metadata.copy()
        new_engine._rule_names = self._rule_names.copy()
        return new_engine

    def __or__(self, other: 'RuleEngine') -> 'RuleEngine':
        """Union of two engines: engine1 | engine2."""
        result = self.copy()
        for rule, meta in other:
            result._rules.append(rule)
            result._metadata.append(meta)
            if meta.name:
                result._rule_names[meta.name] = len(result._rules) - 1
        return result

    def __ior__(self, other: 'RuleEngine') -> 'RuleEngine':
        """In-place union: engine1 |= engine2."""
        for rule, meta in other:
            self._rules.append(rule)
            self._metadata.append(meta)
            if meta.name:
                self._rule_names[meta.name] = len(self._rules) - 1
        self._simplifier = None
        return self

    # ============================================================
    # Equivalence Enumeration
    # ============================================================

    def _all_single_rewrites(
        self,
        expr: ExprType,
        bidirectional_only: bool = True,
        groups: Optional[List[str]] = None,
        rules: Optional[RuleSet] = None,
    ) -> List[ExprType]:
        """Find all expressions reachable by applying exactly one rule.

        Tries every rule at every position in the expression tree.

        Args:
            expr: Expression to rewrite.
            bidirectional_only: If True (default), only use rules from `<=>`
                declarations. Ignored when ``rules`` is provided.
            groups: If specified, only use rules from these groups.
                Ignored when ``rules`` is provided.
            rules: A ``RuleSet`` view that supersedes ``bidirectional_only``
                and ``groups``. The recommended way to scope rule subsets.

        Returns:
            List of all distinct one-step rewrites.
        """
        if rules is None:
            rules = self.rule_set(groups=groups, bidirectional_only=bidirectional_only)

        results = []
        seen: Set[tuple] = set()

        def add_if_new(new_expr: ExprType) -> None:
            key = _expr_to_tuple(new_expr)
            if key not in seen:
                seen.add(key)
                results.append(new_expr)

        # Try rules at top level
        for rule_idx, rule, metadata in rules:
            pattern, skeleton = rule
            bindings = _match_internal(pattern, expr)
            if bindings is not None:
                if not self._check_condition(metadata.condition, bindings):
                    continue
                result = instantiate(skeleton, bindings, self._fold_funcs)
                if result != expr:
                    add_if_new(result)

        # Recursively try rules in subexpressions
        if isinstance(expr, list) and len(expr) > 0:
            for i, child in enumerate(expr):
                child_rewrites = self._all_single_rewrites(child, rules=rules)
                for new_child in child_rewrites:
                    new_expr = expr[:i] + [new_child] + expr[i+1:]
                    add_if_new(new_expr)

        return results

    def equivalents(
        self,
        expr: ExprType,
        max_depth: int = 10,
        max_count: Optional[int] = None,
        strategy: str = "bfs",
        include_unidirectional: bool = False,
        groups: Optional[List[str]] = None,
        rules: Optional[RuleSet] = None,
    ) -> Iterator[ExprType]:
        """
        Enumerate all expressions equivalent to the given expression.

        Uses bidirectional rules (<=> declarations) to explore the equivalence
        class. Yields expressions in order of increasing distance from the
        original (when using BFS strategy).

        Args:
            expr: Starting expression
            max_depth: Maximum number of rewrite steps from original (default: 10)
            max_count: Maximum number of equivalents to yield (default: None = unlimited)
            strategy: Exploration strategy - "bfs" (breadth-first) or "dfs" (depth-first)
            include_unidirectional: If True, also use => rules (not just <=>)
            groups: If specified, only use rules from these groups

        Yields:
            Equivalent expressions, starting with the original

        Example:
            engine = RuleEngine.from_dsl('''
                @commute: (+ ?x ?y) <=> (+ :y :x)
                @assoc: (+ (+ ?x ?y) ?z) <=> (+ :x (+ :y :z))
            ''')

            for equiv in engine.equivalents(["+", "a", "b"], max_depth=2):
                print(format_sexpr(equiv))
            # Output:
            # (+ a b)
            # (+ b a)
        """
        if strategy not in ("bfs", "dfs"):
            raise ValueError(f"Unknown strategy: {strategy}. Valid: bfs, dfs")

        bidirectional_only = not include_unidirectional

        # Track visited expressions
        visited: Set[tuple] = set()
        start_key = _expr_to_tuple(expr)
        visited.add(start_key)

        # Count of yielded expressions
        count = 0

        # Yield the starting expression
        yield expr
        count += 1
        if max_count is not None and count >= max_count:
            return

        # Initialize queue/stack with (expression, depth)
        if strategy == "bfs":
            frontier: deque = deque([(expr, 0)])
        else:  # dfs
            frontier: List = [(expr, 0)]

        while frontier:
            if strategy == "bfs":
                current, depth = frontier.popleft()
            else:
                current, depth = frontier.pop()

            if depth >= max_depth:
                continue

            # Find all single-step rewrites
            rewrites = self._all_single_rewrites(
                current, bidirectional_only, groups, rules=rules
            )

            for new_expr in rewrites:
                key = _expr_to_tuple(new_expr)
                if key not in visited:
                    visited.add(key)
                    yield new_expr
                    count += 1

                    if max_count is not None and count >= max_count:
                        return

                    # Add to frontier for further exploration.
                    # BFS vs DFS only differs on the pop side (popleft vs pop);
                    # the append side is symmetric for both structures.
                    frontier.append((new_expr, depth + 1))

    def enumerate_equivalents(
        self,
        expr: ExprType,
        max_depth: int = 10,
        max_count: Optional[int] = 1000,
        **kwargs
    ) -> List[ExprType]:
        """
        Return a list of all equivalent expressions.

        Convenience wrapper around equivalents() that collects all results.

        Args:
            expr: Starting expression
            max_depth: Maximum rewrite steps (default: 10)
            max_count: Maximum results (default: 1000)
            **kwargs: Additional arguments passed to equivalents()

        Returns:
            List of equivalent expressions
        """
        return list(self.equivalents(
            expr, max_depth=max_depth, max_count=max_count, **kwargs
        ))

    def prove_equal(
        self,
        expr_a: ExprType,
        expr_b: ExprType,
        max_depth: int = 10,
        max_expressions: Optional[int] = None,
        trace: bool = False,
        include_unidirectional: bool = False,
        groups: Optional[List[str]] = None,
        rules: Optional[RuleSet] = None,
    ) -> Optional[EqualityProof]:
        """
        Prove two expressions are equivalent by finding a common form.

        Uses bidirectional BFS: expands from both expressions simultaneously
        and checks for intersection. This is more efficient than enumerating
        entire equivalence classes.

        Args:
            expr_a: First expression
            expr_b: Second expression
            max_depth: Maximum rewrite steps from either expression (default: 10)
            max_expressions: Optional total-work budget across both frontiers.
                When the sum of visited states exceeds this, the search
                returns None (same as max_depth exhaustion). Useful for
                bounding un-provable queries, which otherwise exhaust the
                full reachable set under `max_depth`. Default: None (no cap).
            trace: If True, include full paths in the proof
            include_unidirectional: If True, also use => rules (not just <=>)
            groups: If specified, only use rules from these groups

        Returns:
            EqualityProof if expressions are equivalent, None otherwise

        Example:
            engine = RuleEngine.from_dsl('''
                @commute: (+ ?x ?y) <=> (+ :y :x)
            ''')

            proof = engine.prove_equal(["+", "a", "b"], ["+", "b", "a"])
            if proof:
                print(f"Equal! Common form: {format_sexpr(proof.common)}")

            # With a work budget for expensive queries:
            proof = engine.prove_equal(a, b, max_depth=8, max_expressions=5000)
        """
        bidirectional_only = not include_unidirectional

        # Convert to hashable for set operations
        key_a = _expr_to_tuple(expr_a)
        key_b = _expr_to_tuple(expr_b)

        # Quick check: are they already equal?
        if key_a == key_b:
            path = [expr_a] if trace else None
            return EqualityProof(
                expr_a=expr_a,
                expr_b=expr_b,
                common=expr_a,
                depth_a=0,
                depth_b=0,
                path_a=path,
                path_b=path
            )

        # Track visited expressions from each side
        # Maps: hashable_key -> (original_expr, depth, parent_key)
        visited_a: Dict[tuple, Tuple[ExprType, int, Optional[tuple]]] = {
            key_a: (expr_a, 0, None)
        }
        visited_b: Dict[tuple, Tuple[ExprType, int, Optional[tuple]]] = {
            key_b: (expr_b, 0, None)
        }

        # BFS frontiers: (expression, depth)
        frontier_a: deque = deque([(expr_a, 0)])
        frontier_b: deque = deque([(expr_b, 0)])

        def reconstruct_path(
            visited: Dict[tuple, Tuple[ExprType, int, Optional[tuple]]],
            target_key: tuple
        ) -> List[ExprType]:
            """Reconstruct path from start to target."""
            path = []
            current_key = target_key
            while current_key is not None:
                expr, depth, parent_key = visited[current_key]
                path.append(expr)
                current_key = parent_key
            path.reverse()
            return path

        # Alternate expanding from A and B
        while frontier_a or frontier_b:
            # Budget check: total work across both frontiers
            if max_expressions is not None and \
                    len(visited_a) + len(visited_b) >= max_expressions:
                return None

            # Expand from A
            if frontier_a:
                current, depth = frontier_a.popleft()
                if depth < max_depth:
                    current_key = _expr_to_tuple(current)
                    rewrites = self._all_single_rewrites(
                        current, bidirectional_only, groups, rules=rules
                    )
                    for new_expr in rewrites:
                        new_key = _expr_to_tuple(new_expr)
                        if new_key not in visited_a:
                            visited_a[new_key] = (new_expr, depth + 1, current_key)
                            frontier_a.append((new_expr, depth + 1))

                            # Check for intersection
                            if new_key in visited_b:
                                _, depth_b, _ = visited_b[new_key]
                                if trace:
                                    path_a = reconstruct_path(visited_a, new_key)
                                    path_b = reconstruct_path(visited_b, new_key)
                                else:
                                    path_a = None
                                    path_b = None

                                return EqualityProof(
                                    expr_a=expr_a,
                                    expr_b=expr_b,
                                    common=new_expr,
                                    depth_a=depth + 1,
                                    depth_b=depth_b,
                                    path_a=path_a,
                                    path_b=path_b
                                )

            # Expand from B
            if frontier_b:
                current, depth = frontier_b.popleft()
                if depth < max_depth:
                    current_key = _expr_to_tuple(current)
                    rewrites = self._all_single_rewrites(
                        current, bidirectional_only, groups, rules=rules
                    )
                    for new_expr in rewrites:
                        new_key = _expr_to_tuple(new_expr)
                        if new_key not in visited_b:
                            visited_b[new_key] = (new_expr, depth + 1, current_key)
                            frontier_b.append((new_expr, depth + 1))

                            # Check for intersection
                            if new_key in visited_a:
                                _, depth_a_val, _ = visited_a[new_key]
                                if trace:
                                    path_a = reconstruct_path(visited_a, new_key)
                                    path_b = reconstruct_path(visited_b, new_key)
                                else:
                                    path_a = None
                                    path_b = None

                                return EqualityProof(
                                    expr_a=expr_a,
                                    expr_b=expr_b,
                                    common=new_expr,
                                    depth_a=depth_a_val,
                                    depth_b=depth + 1,
                                    path_a=path_a,
                                    path_b=path_b
                                )

        # No common form found within depth limit
        return None

    def are_equal(
        self,
        expr_a: ExprType,
        expr_b: ExprType,
        max_depth: int = 10,
        **kwargs
    ) -> bool:
        """
        Check if two expressions are equivalent.

        Convenience method that returns a boolean instead of a proof.

        Args:
            expr_a: First expression
            expr_b: Second expression
            max_depth: Maximum rewrite steps (default: 10)
            **kwargs: Additional arguments passed to prove_equal()

        Returns:
            True if expressions are equivalent, False otherwise

        Example:
            if engine.are_equal(["+", "a", "b"], ["+", "b", "a"]):
                print("Expressions are equal!")
        """
        return self.prove_equal(expr_a, expr_b, max_depth=max_depth, **kwargs) is not None

    # ============================================================
    # Cost Optimization
    # ============================================================

    def minimize(
        self,
        expr: ExprType,
        cost: Optional[Callable[[ExprType], float]] = None,
        metric: Optional[str] = None,
        op_costs: Optional[Dict[str, float]] = None,
        max_depth: int = 10,
        max_count: Optional[int] = 10000,
        include_unidirectional: bool = True,
        groups: Optional[List[str]] = None,
        rules: Optional[RuleSet] = None,
    ) -> OptimizationResult:
        """
        Find the minimum-cost equivalent expression.

        Explores the equivalence class and returns the expression with lowest cost.
        Cost can be specified via a custom function, a built-in metric, or operator costs.

        Args:
            expr: Expression to optimize
            cost: Custom cost function (expr -> float)
            metric: Built-in metric: "size", "depth", "ops", or "atoms"
            op_costs: Dictionary of operator costs (e.g., {"+": 1, "*": 2, "^": 10})
            max_depth: Maximum rewrite steps from original (default: 10)
            max_count: Maximum expressions to evaluate (default: 10000)
            include_unidirectional: If True (default), also use => rules in
                addition to <=> rules. Simplification rules are typically
                unidirectional, so this default makes minimize useful out
                of the box. Set to False to restrict to bidirectional-only
                exploration (useful when you want to preserve equivalence
                strictly under a reversible theory).
            groups: If specified, only use rules from these groups

        Returns:
            OptimizationResult with the best expression found

        Example:
            # Minimize by size
            result = engine.minimize(expr, metric="size")
            print(f"Best: {format_sexpr(result.expr)}, cost: {result.cost}")

            # Custom cost function
            result = engine.minimize(expr, cost=lambda e: expr_depth(e) * 2 + expr_ops(e))

            # Operator costs
            result = engine.minimize(expr, op_costs={"+": 1, "*": 2, "/": 5, "^": 10})
        """
        # Determine cost function
        if cost is not None:
            cost_fn = cost
        elif metric is not None:
            if metric not in COST_METRICS:
                raise ValueError(f"Unknown metric: {metric}. Valid: {list(COST_METRICS.keys())}")
            cost_fn = COST_METRICS[metric]
        elif op_costs is not None:
            cost_fn = make_op_cost_fn(op_costs)
        else:
            # Default to size
            cost_fn = expr_size

        # Track best found
        best_expr = expr
        best_cost = cost_fn(expr)
        original_cost = best_cost
        count = 0

        # Explore equivalents
        for equiv in self.equivalents(
            expr,
            max_depth=max_depth,
            max_count=max_count,
            include_unidirectional=include_unidirectional,
            groups=groups,
            rules=rules,
        ):
            count += 1
            equiv_cost = cost_fn(equiv)
            if equiv_cost < best_cost:
                best_expr = equiv
                best_cost = equiv_cost

        return OptimizationResult(
            expr=best_expr,
            cost=best_cost,
            original=expr,
            original_cost=original_cost,
            expressions_checked=count
        )

    # ============================================================
    # Random Sampling
    # ============================================================

    def random_equivalent(
        self,
        expr: ExprType,
        steps: int = 10,
        include_unidirectional: bool = False,
        groups: Optional[List[str]] = None,
        rng: Optional[random.Random] = None,
        rules: Optional[RuleSet] = None,
    ) -> ExprType:
        """
        Generate a random equivalent expression via random walk.

        Performs a random walk through the rewrite space, applying
        randomly chosen rules at each step.

        Args:
            expr: Starting expression
            steps: Number of random rewrite steps (default: 10)
            include_unidirectional: If True, also use => rules
            groups: If specified, only use rules from these groups
            rng: Optional random.Random instance for reproducibility

        Returns:
            A randomly chosen equivalent expression

        Example:
            # Get a random equivalent
            rand_expr = engine.random_equivalent(expr, steps=10)

            # Reproducible randomness
            rng = random.Random(42)
            rand_expr = engine.random_equivalent(expr, rng=rng)
        """
        if rng is None:
            rng = random.Random()

        bidirectional_only = not include_unidirectional
        current = expr

        for _ in range(steps):
            rewrites = self._all_single_rewrites(current, bidirectional_only, groups, rules=rules)
            if not rewrites:
                break
            current = rng.choice(rewrites)

        return current

    def sample_equivalents(
        self,
        expr: ExprType,
        n: int = 10,
        steps: int = 10,
        unique: bool = True,
        max_attempts: int = 100,
        include_unidirectional: bool = False,
        groups: Optional[List[str]] = None,
        rng: Optional[random.Random] = None,
        rules: Optional[RuleSet] = None,
    ) -> List[ExprType]:
        """
        Sample multiple random equivalent expressions.

        Args:
            expr: Starting expression
            n: Number of samples to generate (default: 10)
            steps: Random walk steps per sample (default: 10)
            unique: If True, return only unique expressions (default: True)
            max_attempts: Maximum attempts to find unique samples (default: 100)
            include_unidirectional: If True, also use => rules
            groups: If specified, only use rules from these groups
            rng: Optional random.Random instance for reproducibility

        Returns:
            List of sampled equivalent expressions

        Example:
            samples = engine.sample_equivalents(expr, n=5, steps=10)
            for s in samples:
                print(format_sexpr(s))
        """
        if rng is None:
            rng = random.Random()

        if unique:
            seen: Set[tuple] = set()
            samples: List[ExprType] = []
            attempts = 0

            while len(samples) < n and attempts < max_attempts:
                attempts += 1
                sample = self.random_equivalent(
                    expr, steps=steps,
                    include_unidirectional=include_unidirectional,
                    groups=groups, rng=rng, rules=rules,
                )
                key = _expr_to_tuple(sample)
                if key not in seen:
                    seen.add(key)
                    samples.append(sample)

            return samples
        else:
            return [
                self.random_equivalent(
                    expr, steps=steps,
                    include_unidirectional=include_unidirectional,
                    groups=groups, rng=rng, rules=rules,
                )
                for _ in range(n)
            ]

    def random_walk(
        self,
        expr: ExprType,
        max_steps: int = 100,
        include_unidirectional: bool = False,
        groups: Optional[List[str]] = None,
        rng: Optional[random.Random] = None,
        rules: Optional[RuleSet] = None,
    ) -> Iterator[ExprType]:
        """
        Generate a lazy random walk through the equivalence class.

        Yields expressions along a random path through the rewrite space.
        Useful for exploring large equivalence classes without enumerating all.

        Args:
            expr: Starting expression
            max_steps: Maximum steps to take (default: 100)
            include_unidirectional: If True, also use => rules
            groups: If specified, only use rules from these groups
            rng: Optional random.Random instance for reproducibility

        Yields:
            Expressions along the random walk, starting with the original

        Example:
            for i, equiv in enumerate(engine.random_walk(expr, max_steps=20)):
                print(f"Step {i}: {format_sexpr(equiv)}")
        """
        if rng is None:
            rng = random.Random()

        bidirectional_only = not include_unidirectional
        current = expr
        yield current

        for _ in range(max_steps):
            rewrites = self._all_single_rewrites(current, bidirectional_only, groups, rules=rules)
            if not rewrites:
                break
            current = rng.choice(rewrites)
            yield current

    # ============================================================
    # Hook registration API
    # ============================================================

    _HOOK_EVENTS = {
        "rule_applied":  "observer",
        "fixpoint":      "observer",
        "no_match":      "resolver",
        "undefined_op":  "resolver",
        "fold_error":    "resolver",
        "max_depth":     "resolver",
        "cycle":         "resolver",
        "should_fire":   "decision",
    }

    def _make_on_method(event: str, category: str):
        def on_event(self, callback):
            self._hooks.register(event, category, callback)
            return callback  # so it works as a decorator
        on_event.__name__ = f"on_{event}"
        on_event.__doc__ = (
            f"Register a {category} hook for the {event!r} event. "
            f"Callable is returned unchanged so this method works as a "
            f"decorator."
        )
        return on_event

    def _make_off_method(event: str):
        def off_event(self, callback):
            return self._hooks.unregister(event, callback)
        off_event.__name__ = f"off_{event}"
        off_event.__doc__ = f"Unregister a previously-registered hook for the {event!r} event."
        return off_event

    def clear_hooks(self, event: Optional[str] = None) -> None:
        """Remove all hooks for ``event``, or all hooks for all events when
        ``event`` is None."""
        self._hooks.clear(event)

    def __rshift__(self, other: 'RuleEngine') -> 'SequencedEngine':
        """
        Sequence two engines: engine1 >> engine2.

        Returns a SequencedEngine that applies engine1 until fixpoint,
        then applies engine2 until fixpoint.

        Example:
            expand = RuleEngine.from_dsl("@expand: (square ?x) => (* :x :x)")
            simplify = RuleEngine.from_dsl("@fold: (* ?a:const ?b:const) => (! * :a :b)")
            normalize = expand >> simplify

            result = normalize(E("(square 3)"))  # => 9
        """
        return SequencedEngine([self, other])


# Install on_<event> / off_<event> methods on RuleEngine.
for _event, _category in RuleEngine._HOOK_EVENTS.items():
    setattr(RuleEngine, f"on_{_event}",
            RuleEngine._make_on_method(_event, _category))
    setattr(RuleEngine, f"off_{_event}",
            RuleEngine._make_off_method(_event))
del _event, _category
del RuleEngine._make_on_method
del RuleEngine._make_off_method


class SequencedEngine:
    """
    An engine that applies multiple engines in sequence.

    Each engine is run until its fixpoint before moving to the next.
    Created via the >> operator on RuleEngine.

    Example:
        phase1 = RuleEngine.from_dsl("...")
        phase2 = RuleEngine.from_dsl("...")
        phased = phase1 >> phase2
        result = phased(expr)
    """

    def __init__(self, engines: List['RuleEngine']):
        """Initialize with a list of engines to apply in sequence."""
        self._engines = engines

    def __call__(self, expr: ExprType, **kwargs) -> ExprType:
        """Apply all engines in sequence."""
        result = expr
        for engine in self._engines:
            result = engine(result, **kwargs)
        return result

    def __rshift__(self, other: 'RuleEngine') -> 'SequencedEngine':
        """Chain another engine: (a >> b) >> c."""
        if isinstance(other, SequencedEngine):
            return SequencedEngine(self._engines + other._engines)
        return SequencedEngine(self._engines + [other])

    def __repr__(self) -> str:
        return f"SequencedEngine({len(self._engines)} phases)"

    def __len__(self) -> int:
        """Number of phases."""
        return len(self._engines)

    def __iter__(self):
        """Iterate over engines."""
        return iter(self._engines)
