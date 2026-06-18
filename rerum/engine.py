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

import inspect
import json
import random
import re
from collections import deque
from fractions import Fraction
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union, Iterator, Set, Callable, Any

from .rewriter import (
    rewriter, match as _match_internal, instantiate, ExprType,
    FoldFuncsType, ARITHMETIC_PRELUDE, Bindings, NoMatch, _NoMatch, wrap_bindings,
    skeleton_compute, NUMERIC_TYPES,
)
from .hooks import (
    _HookRegistry,
    HooksError,
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
    parse_rational_token,
    parse_sexpr,
    format_sexpr,
    E,
    _ExprBuilder,
    expr_to_tuple as _expr_to_tuple,
)


class RuleMetadata:
    """Metadata for a rule.

    Beyond the original name/description/priority/tags/condition/bidirectional
    fields, this carries v0.7 metadata: ``category`` (free-form label for
    LLM paraphrasing), ``reasoning`` (free-text justification), ``examples``
    (list of {in, out} s-expression strings, validated on load), and
    ``fwd_label``/``rev_label`` (direction semantics for ``<=>`` rules,
    surfaced through JSON only).
    """

    def __init__(self, name: Optional[str] = None,
                 description: Optional[str] = None,
                 tags: Optional[List[str]] = None,
                 condition: Optional[ExprType] = None,
                 priority: int = 0,
                 bidirectional: bool = False,
                 direction: Optional[str] = None,
                 extra: Optional[Dict[str, Any]] = None,
                 category: Optional[str] = None,
                 reasoning: Optional[str] = None,
                 examples: Optional[List[Dict[str, Any]]] = None,
                 fwd_label: Optional[str] = None,
                 rev_label: Optional[str] = None):
        self.name = name
        self.description = description
        self.tags = tags or []
        self.condition = condition  # Optional guard condition
        self.priority = priority  # Higher priority fires first (default: 0)
        self.bidirectional = bidirectional  # True if from a <=> rule
        self.direction = direction  # 'fwd' or 'rev' for bidirectional rules
        self.extra = extra or {}  # Resolver-provided metadata (provenance, model, confidence)
        self.category = category  # Semantic category label (e.g. "identity", "commutativity")
        self.reasoning = reasoning  # Human-readable explanation of why the rule is valid
        self.examples = examples if examples is not None else []  # List of {"in": ..., "out": ...} worked examples
        self.fwd_label = fwd_label  # Direction-label metadata; surfaced via JSON in M5
        self.rev_label = rev_label  # Direction-label metadata; surfaced via JSON in M5

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


# Single source of truth for which keys name a RuleMetadata field. Derived from
# the constructor signature so adding a field to RuleMetadata.__init__ updates
# every loader automatically (no hand-maintained, drift-prone parallel sets).
# ``extra`` is excluded: it is the catch-all bucket for *unknown* keys, not a
# settable named field.
_METADATA_FIELDS = frozenset(
    name for name in inspect.signature(RuleMetadata.__init__).parameters
    if name not in ("self", "extra")
)

# A metadata sidecar (load_metadata_json) may DESCRIBE a rule but not redefine
# its identity or structure: name and the bidirectional/direction pairing are
# off-limits. Everything else in _METADATA_FIELDS is mergeable. (priority is
# mergeable but additionally guarded on bidirectional halves, where changing it
# would split the stored -fwd/-rev pair; see load_metadata_json.)
_SIDECAR_PROTECTED_FIELDS = frozenset({"name", "bidirectional", "direction"})
_SIDECAR_MERGEABLE_FIELDS = _METADATA_FIELDS - _SIDECAR_PROTECTED_FIELDS


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


class ExampleValidationError(ValueError):
    """Raised when an example in a rule's metadata does not match the rule.

    Carries ``rule_name``, ``example``, and a description of the mismatch
    (pattern doesn't match, condition fails, or output mismatch).
    """

    def __init__(self, message: str, *, rule_name: Optional[str] = None,
                 example: Optional[Dict] = None):
        super().__init__(message)
        self.rule_name = rule_name
        self.example = example


def _condition_truthy(result) -> bool:
    """Truthiness rule for condition expressions.

    Mirrors RuleEngine._check_condition's truthiness logic, but as a
    standalone helper so it works without an engine instance.
    """
    if isinstance(result, bool):
        return result
    if isinstance(result, NUMERIC_TYPES):
        return result != 0
    if isinstance(result, str):
        return len(result) > 0
    if isinstance(result, list):
        return len(result) > 0
    return True


def _validate_example(pattern, skeleton, metadata, example, fold_funcs,
                      undefined_op_resolver=None, fold_error_resolver=None):
    """Validate one example against a rule. Raises ExampleValidationError on mismatch.

    ``example`` is a dict with ``in`` (s-expr string), ``out`` (s-expr string),
    and an optional ``direction`` field which is informational only; the
    caller is responsible for selecting the right (pattern, skeleton)
    pair for bidirectional rules.

    ``fold_funcs`` is the engine's prelude; needed for ``(! op ...)`` evaluation
    in skeletons or conditions. ``undefined_op_resolver``/``fold_error_resolver``
    are the engine's hook bridges, threaded through so that validation evaluates
    ``(! op ...)`` exactly as live rewriting would (matching ``_check_condition``).
    """
    if not isinstance(example, dict) or "in" not in example or "out" not in example:
        raise ExampleValidationError(
            f"Rule {metadata.name!r}: example must be a dict with 'in' and 'out' keys; "
            f"got {example!r}",
            rule_name=metadata.name,
            example=example,
        )
    in_expr = parse_sexpr(example["in"])
    expected_out = parse_sexpr(example["out"])

    bindings = _match_internal(pattern, in_expr)
    if bindings is None:
        raise ExampleValidationError(
            f"Rule {metadata.name!r}: pattern does not match input "
            f"{example['in']!r}",
            rule_name=metadata.name,
            example=example,
        )

    if metadata.condition is not None:
        cond_result = instantiate(metadata.condition, bindings, fold_funcs,
                                  undefined_op_resolver=undefined_op_resolver,
                                  fold_error_resolver=fold_error_resolver)
        if not _condition_truthy(cond_result):
            raise ExampleValidationError(
                f"Rule {metadata.name!r}: condition fails on input "
                f"{example['in']!r}",
                rule_name=metadata.name,
                example=example,
            )

    actual = instantiate(skeleton, bindings, fold_funcs,
                         undefined_op_resolver=undefined_op_resolver,
                         fold_error_resolver=fold_error_resolver)
    if actual != expected_out:
        raise ExampleValidationError(
            f"Rule {metadata.name!r}: input {example['in']!r} produced "
            f"{format_sexpr(actual)!r}, expected {example['out']!r}",
            rule_name=metadata.name,
            example=example,
        )


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
    category: Optional[str] = None,
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
        category=category,
    )
    rev_metadata = RuleMetadata(
        name=f"{base_name}-rev" if base_name else None,
        description=f"{description} (reverse)" if description else None,
        tags=tags,
        priority=priority,
        condition=condition,
        bidirectional=True,
        direction="rev",
        category=category,
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


_ANNOTATION_KEYS = frozenset({"category"})


def _split_annotation_pairs(inner: str) -> List[str]:
    """Split ``key=value, key=value`` accounting for quoted values."""
    pairs: List[str] = []
    current: List[str] = []
    in_quote = None
    for c in inner:
        if in_quote is not None:
            if c == in_quote:
                in_quote = None
            current.append(c)
        elif c in ('"', "'"):
            in_quote = c
            current.append(c)
        elif c == ',':
            pair = ''.join(current).strip()
            if pair:
                pairs.append(pair)
            current = []
        else:
            current.append(c)
    pair = ''.join(current).strip()
    if pair:
        pairs.append(pair)
    if in_quote is not None:
        raise ValueError(
            f"unclosed quote ({in_quote!r}) in annotation: {inner!r}"
        )
    return pairs


def _find_arrow(line: str) -> Tuple[int, int]:
    """Find the position and length of the rule arrow (=> or <=>) at paren-depth 0.

    Returns (-1, 0) if no arrow is found at depth 0.
    """
    depth = 0
    in_quote = None
    i = 0
    while i < len(line):
        c = line[i]
        if in_quote is not None:
            if c == in_quote:
                in_quote = None
            i += 1
            continue
        if c in ('"', "'"):
            in_quote = c
            i += 1
            continue
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        elif depth == 0:
            # Check for <=> first (longer match).
            if line[i:i+3] == '<=>':
                return i, 3
            if line[i:i+2] == '=>':
                return i, 2
        i += 1
    return -1, 0


def _extract_annotation(header: str) -> Tuple[str, Dict[str, str]]:
    """Extract a ``{key=value}`` annotation block from *header* (header only).

    The annotation must appear in the header portion of the rule line (before
    the arrow). If no annotation is present returns ``(header, {})``.

    Whitespace is permitted inside the braces. Values may be quoted strings
    (for multi-word values or values containing special characters like ``}``)
    or bare tokens. Unknown keys raise ``ValueError``. A missing closing brace
    raises ``ValueError``.

    Returns the header with the annotation block removed and a dict of parsed
    key-value pairs.

    Limitations:
    - Backslash escapes inside quoted values are not recognized. Use the
      alternate quote character to embed the conflicting quote.
    """
    # Locate a ``{`` that is not inside an s-expression (paren-depth == 0).
    depth = 0
    brace_start = -1
    for i, c in enumerate(header):
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        elif c == '{' and depth == 0:
            brace_start = i
            break

    if brace_start < 0:
        return header, {}

    # Find the matching closing brace at depth 0, respecting quotes.
    brace_end = -1
    in_quote = None
    for j in range(brace_start + 1, len(header)):
        c = header[j]
        if in_quote is not None:
            if c == in_quote:
                in_quote = None
            continue
        if c in ('"', "'"):
            in_quote = c
            continue
        if c == '}':
            brace_end = j
            break
    if brace_end < 0:
        if in_quote is not None:
            raise ValueError(
                f"unclosed quote ({in_quote!r}) in annotation: {header!r}"
            )
        raise ValueError(f"malformed annotation: missing '}}' in {header!r}")

    inner = header[brace_start + 1:brace_end].strip()
    remaining = (header[:brace_start] + header[brace_end + 1:]).strip()

    # Detect a stray second annotation block in the remaining header.
    for ch in remaining:
        if ch == '{':
            raise ValueError(
                f"multiple annotation blocks in {header!r}; only one is allowed"
            )

    annotations: Dict[str, str] = {}
    for pair in _split_annotation_pairs(inner):
        if '=' not in pair:
            raise ValueError(f"malformed annotation pair {pair!r} in {header!r}")
        key, value = pair.split('=', 1)
        key = key.strip()
        value = value.strip()
        if key not in _ANNOTATION_KEYS:
            raise ValueError(
                f"unknown annotation key {key!r} (known: {sorted(_ANNOTATION_KEYS)})"
            )
        # Strip surrounding quotes from value.
        if (len(value) >= 2 and value[0] == value[-1]
                and value[0] in ('"', "'")):
            value = value[1:-1]
        annotations[key] = value

    return remaining, annotations


def _format_annotation_value(value: str) -> str:
    """Render an annotation value so it round-trips through ``_extract_annotation``.

    The DSL annotation grammar splits pairs on ``,`` and terminates the block at
    ``}`` (both respected inside quotes), and trims surrounding whitespace from
    bare values. So a value containing any of those — or surrounding whitespace,
    or a leading quote — must be quoted. The grammar has no escape mechanism, so
    we pick whichever quote character is absent from the value (preferring ``"``).
    A value containing *both* quote characters cannot be represented losslessly;
    we fall back to double quotes (consistent with the documented limitation).
    """
    needs_quote = (
        value == ""
        or value != value.strip()
        or any(ch in value for ch in (",", "{", "}", "\"", "'"))
    )
    if not needs_quote:
        return value
    if '"' not in value:
        return f'"{value}"'
    if "'" not in value:
        return f"'{value}'"
    return f'"{value}"'


def _format_dsl_header(name: Optional[str], priority: int,
                       description: Optional[str],
                       category: Optional[str]) -> str:
    """Build the ``@name[priority] "description" {category=...}: `` DSL header
    prefix shared by the ``=>`` and ``<=>`` serialization branches.

    Returns ``""`` for an anonymous, category-less rule.
    """
    if name:
        hdr = f"@{name}"
        if priority != 0:
            hdr += f"[{priority}]"
        if description:
            hdr += f" \"{description}\""
        if category is not None:
            hdr += f" {{category={_format_annotation_value(category)}}}"
        return hdr + ": "
    if category is not None:
        return f"{{category={_format_annotation_value(category)}}}: "
    return ""


def _emit_metadata_fields(rule_dict: Dict, meta: "RuleMetadata") -> None:
    """Append the v0.7 descriptive metadata (``category``/``reasoning``/
    ``examples``) plus any resolver-provided ``extra`` keys onto a serialized
    rule dict. Shared by the unidirectional and bidirectional branches of
    ``_rules_as_dicts`` so the two stay in lock-step.

    ``extra`` is emitted last via ``setdefault`` so it can never clobber a known
    field (by construction ``extra`` only holds keys outside the known set, but
    ``setdefault`` keeps that guarantee local and obvious).
    """
    if meta.category is not None:
        rule_dict["category"] = meta.category
    if meta.reasoning is not None:
        rule_dict["reasoning"] = meta.reasoning
    if meta.examples:
        rule_dict["examples"] = meta.examples
    for key, value in meta.extra.items():
        rule_dict.setdefault(key, value)


def _count_unbalanced_braces(line: str) -> int:
    """Return ``open - close`` brace count for *line*, ignoring ``{``/``}``
    inside s-expressions (paren depth > 0) and inside quoted strings.

    A positive result means the line opens more braces than it closes; negative
    means it closes more than it opens; zero means balanced (or no braces at all).
    """
    paren_depth = 0
    open_count = 0
    close_count = 0
    in_quote: Optional[str] = None
    for c in line:
        if in_quote is not None:
            if c == in_quote:
                in_quote = None
            continue
        if c in ('"', "'"):
            in_quote = c
            continue
        if c == '(':
            paren_depth += 1
        elif c == ')':
            paren_depth -= 1
        elif paren_depth == 0:
            if c == '{':
                open_count += 1
            elif c == '}':
                close_count += 1
    return open_count - close_count


def _join_multi_line_annotations(lines: List[str]) -> List[str]:
    """Join lines so multi-line ``{ ... }`` annotation blocks become single-line
    entries ready for ``parse_rule_line``.

    A line that opens a ``{`` without a matching ``}`` on the same line (at
    s-expression depth 0 and outside quoted strings) is joined with subsequent
    lines until the matching ``}`` is found. The joined result is a single
    space-separated string.

    Limitations:
    - ``#`` comments are not stripped inside a multi-line block. Use a
      single-line annotation if you want to comment alongside it.
    - Backslash escapes inside quoted values are not recognized. Use the
      alternate quote character (double for values with apostrophes,
      single for values with double-quotes).
    """
    joined: List[str] = []
    buffer: List[str] = []
    open_depth = 0  # net unbalanced open braces accumulated so far

    for line in lines:
        delta = _count_unbalanced_braces(line)
        if buffer:
            buffer.append(line)
            open_depth += delta
            if open_depth <= 0:
                # Block closed: emit the joined line and reset.
                joined.append(' '.join(buffer))
                buffer = []
                open_depth = 0
            continue

        if delta > 0:
            # This line starts a multi-line block.
            buffer = [line]
            open_depth = delta
        else:
            joined.append(line)

    if buffer:
        # Unbalanced at end of input; pass through so the downstream parser
        # raises a clear error.
        joined.append(' '.join(buffer))

    return joined


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

    # Find the arrow position (=> or <=>). Annotation extraction operates
    # only on the header portion (before the arrow); the body is left
    # untouched.
    arrow_pos, arrow_len = _find_arrow(line)
    if arrow_pos < 0:
        # No arrow found at paren-depth 0. If the raw string contains '=>'
        # the most likely cause is an unclosed quote inside an annotation
        # block that consumed the arrow. Attempt annotation extraction on
        # the full line so that _split_annotation_pairs can raise its
        # "unclosed quote" error rather than silently returning None.
        if '=>' in line:
            _extract_annotation(line)  # raises ValueError on malformed input
        return None

    header = line[:arrow_pos]
    body_with_arrow = line[arrow_pos:]

    # Extract optional `{key=val}` annotation from the header only.
    header, annotations = _extract_annotation(header)

    # Reconstruct the line for the existing regex-based header parsing.
    line = header + ' ' + body_with_arrow

    # Extract name, optional priority, and optional description if present
    base_name = None
    description = None
    priority = 0

    if line.startswith('@'):
        # Header regex patterns: `\s*:` allows whitespace between the last token
        # and the colon, which is left over after `_extract_annotation` strips
        # the `{...}` block.
        # Try format: @name[priority] "description": ...
        match_obj = re.match(r'@([\w-]+)\[(\d+)\]\s+"([^"]+)"\s*:\s*(.+)', line)
        if match_obj:
            base_name = match_obj.group(1)
            priority = int(match_obj.group(2))
            description = match_obj.group(3)
            line = match_obj.group(4)
        else:
            # Try format: @name[priority]: ...
            match_obj = re.match(r'@([\w-]+)\[(\d+)\]\s*:\s*(.+)', line)
            if match_obj:
                base_name = match_obj.group(1)
                priority = int(match_obj.group(2))
                line = match_obj.group(3)
            else:
                # Try format: @name "description": ...
                match_obj = re.match(r'@([\w-]+)\s+"([^"]+)"\s*:\s*(.+)', line)
                if match_obj:
                    base_name = match_obj.group(1)
                    description = match_obj.group(2)
                    line = match_obj.group(3)
                else:
                    # Try format: @name: ...
                    match_obj = re.match(r'@([\w-]+)\s*:\s*(.+)', line)
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
            category=annotations.get("category"),
        )
    else:
        # Single unidirectional rule
        metadata = RuleMetadata(
            name=base_name,
            description=description,
            priority=priority,
            condition=condition,
            category=annotations.get("category"),
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

    for line in _join_multi_line_annotations(text.split('\n')):
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


def _expr_to_jsonable(expr):
    """Encode an expression for the JSON rule format.

    JSON has no Fraction: a Fraction atom is encoded as its rational-literal
    token (``Fraction(1, 3)`` -> ``"1/3"``), exactly the form the text layer
    parses back to the same atom. Lists recurse; everything else is already
    JSON-native.
    """
    if isinstance(expr, Fraction):
        return f"{expr.numerator}/{expr.denominator}"
    if isinstance(expr, list):
        return [_expr_to_jsonable(e) for e in expr]
    return expr


def _expr_from_jsonable(expr):
    """Decode an expression from the JSON rule format.

    A string atom shaped like a rational-literal token (``"1/3"``) decodes
    to the exact Fraction atom -- the inverse of ``_expr_to_jsonable``, and
    the same lexical rule the text parser applies. (Post-rational-literals
    a SYMBOL of that shape is not constructible from text, so the token
    unambiguously means the number.)
    """
    if isinstance(expr, str):
        rat = parse_rational_token(expr)
        return rat if rat is not None else expr
    if isinstance(expr, list):
        return [_expr_from_jsonable(e) for e in expr]
    return expr


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
            pattern = _expr_from_jsonable(rule['pattern'])
            skeleton = _expr_from_jsonable(rule['skeleton'])

            # Known keys = the RuleMetadata fields (derived from its signature,
            # see _METADATA_FIELDS) plus the two structural keys a raw rule dict
            # carries. Anything else is routed to `extra`. Adding a field to
            # RuleMetadata.__init__ updates this automatically.
            known = _METADATA_FIELDS | {'pattern', 'skeleton'}
            extra = {k: v for k, v in rule.items() if k not in known}

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
                    category=rule.get('category'),
                )
                fwd_label = rule.get('fwd_label')
                rev_label = rule.get('rev_label')
                reasoning = rule.get('reasoning')
                examples = rule.get('examples')
                for meta, pat, skel in pairs:
                    if meta.direction == 'fwd':
                        meta.fwd_label = fwd_label
                    if meta.direction == 'rev':
                        meta.rev_label = rev_label
                    meta.reasoning = reasoning
                    if examples is not None:
                        # Independent list per half (mirrors the dict(extra)
                        # copy below); a shared list would let mutating one
                        # half's examples silently mutate the other.
                        meta.examples = list(examples)
                    if extra:
                        meta.extra = dict(extra)  # one independent copy per half
                    rules.append((meta, [pat, skel]))
                continue

            # Validate that fwd_label/rev_label are not set on unidirectional rules.
            if rule.get('fwd_label') is not None or rule.get('rev_label') is not None:
                raise ValueError(
                    f"fwd_label/rev_label only valid on bidirectional rules; "
                    f"got rule {rule.get('name')!r}"
                )

            metadata = RuleMetadata(
                name=rule.get('name'),
                description=rule.get('description'),
                tags=rule.get('tags'),
                priority=rule.get('priority', 0),
                condition=rule.get('condition'),
                category=rule.get('category'),
                reasoning=rule.get('reasoning'),
                examples=rule.get('examples'),
                extra=extra,
            )
        else:
            metadata = RuleMetadata()
            pattern = _expr_from_jsonable(rule[0])
            skeleton = _expr_from_jsonable(rule[1])
        rules.append((metadata, [pattern, skeleton]))

    return rules


# RewriteStep, RewriteTrace, and the listener protocol live in `trace.py`.
# Re-exported below for backward compatibility.
from .trace import RewriteStep, RewriteTrace


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
        path_a: From expr_a to common. List[RewriteStep] when produced by
            prove_equal(trace=True) (a synthetic initial step plus one
            labeled step per edge); may also be a plain expression list when
            constructed directly. Step __eq__ compares to an expression by
            its ``after``, so endpoint assertions still hold.
        path_b: From expr_b to common; same shape as path_a.
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

            def _fmt_path_item(item) -> str:
                return format_sexpr(item.after) if isinstance(item, RewriteStep) else format_sexpr(item)

            if self.path_a:
                lines.append(f"\nPath from A ({self.depth_a} steps):")
                for i, item in enumerate(self.path_a):
                    lines.append(f"  {i}. {_fmt_path_item(item)}")

            if self.path_b:
                lines.append(f"\nPath from B ({self.depth_b} steps):")
                for i, item in enumerate(self.path_b):
                    lines.append(f"  {i}. {_fmt_path_item(item)}")

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

        def _serialize_path(path):
            out = []
            for item in path:
                out.append(item.to_dict() if isinstance(item, RewriteStep) else item)
            return out

        if self.path_a:
            result["path_a"] = _serialize_path(self.path_a)
        if self.path_b:
            result["path_b"] = _serialize_path(self.path_b)
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

        Note: ``RuleEngine`` instances are not thread-safe. ``_step_count``,
        ``_cancel_requested``, and the rule-set storage are mutable per-call
        state held on the instance. Concurrent ``simplify()`` calls on the
        same engine will race. Create a separate instance per thread, or
        use a lock at the call boundary.
        """
        self._rules: List[List] = []
        self._metadata: List[RuleMetadata] = []
        self._rule_names: Dict[str, int] = {}  # Maps name -> index
        self._simplifier = None
        self._fold_funcs: Optional[FoldFuncsType] = fold_funcs
        self._disabled_groups: set = set()  # Groups that are disabled
        self._hooks = _HookRegistry()
        self._cancel_requested = False
        self._step_count: int = 0  # successful rule applications in current top-level call
        # Session theory slot: operator-signature DATA (a normalize.Theory),
        # set by callers (e.g. the MCP load_theory tool) and consumed where
        # a theory= argument is threaded (e.g. solve's normalize_between).
        # Always present, so callers never need a hasattr dance.
        self._theory = None
        # The loading contract declared by the last file loaded via
        # ``load_file`` (a RuleSetManifest), or None. ``load_file`` STORES
        # it but applies nothing; ``from_manifest`` both stores and acts.
        self.manifest = None

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

    def _validate_rule_examples(self, rule, metadata) -> None:
        """Validate every example for a rule. Raises ExampleValidationError on
        the first failing example.

        For bidirectional rules, examples may carry a ``direction`` field
        ("fwd" or "rev") to select which pattern/skeleton pair to test
        against. Default is "fwd".
        """
        if not metadata.examples:
            return
        pattern, skeleton = rule
        for example in metadata.examples:
            direction = example.get("direction", "fwd")
            if metadata.bidirectional and direction != metadata.direction:
                # This example is not for this half of the bidirectional pair.
                continue
            _validate_example(
                pattern, skeleton, metadata, example, self._fold_funcs or {},
                undefined_op_resolver=self._undefined_op_resolver,
                fold_error_resolver=self._fold_error_resolver,
            )

    def validate_examples(self) -> None:
        """Validate every example for every rule in the engine.

        Raises ExampleValidationError on the first failing example. Useful
        after a prelude change (rules loaded with validate_examples=False
        when the prelude was not yet configured) or as an explicit audit
        step before relying on rule documentation.
        """
        for rule, metadata in zip(self._rules, self._metadata):
            self._validate_rule_examples(rule, metadata)

    def load_metadata_json(self, text: str,
                           validate_examples: bool = True) -> 'RuleEngine':
        """Merge a metadata-only JSON sidecar onto already-loaded rules.

        The JSON shape is ``{rule_name: {field: value, ...}}``. Each top-level
        key must match an ``@name`` already in the engine. Each inner field
        merges onto the existing RuleMetadata: a field is filled only if it is
        still at its constructor default; trying to overwrite an already-set
        field with a different value raises ValueError (setting the same value
        is harmless). Identity/structure fields (``name``, ``bidirectional``,
        ``direction``) may not be set via a sidecar, and ``priority`` may not be
        set on a bidirectional half (it would split the stored -fwd/-rev pair).

        Unknown fields land in ``RuleMetadata.extra`` and are conflict-checked
        the same way. A filled ``priority`` re-sorts the engine.

        After merging, examples are validated by default (validate_examples
        kwarg).
        """
        import json as _json
        data = _json.loads(text)
        if not isinstance(data, dict):
            raise ValueError(
                "sidecar JSON must be an object mapping name -> metadata"
            )

        reference = RuleMetadata()  # constructor defaults; "unset" == equals this
        priority_touched = False
        changed = False

        for rule_name, fields in data.items():
            if rule_name not in self._rule_names:
                raise ValueError(
                    f"sidecar references no rule named {rule_name!r}"
                )
            idx = self._rule_names[rule_name]
            metadata = self._metadata[idx]

            for field, value in fields.items():
                # B7: identity/structure fields are not sidecar-settable.
                if field in _SIDECAR_PROTECTED_FIELDS:
                    raise ValueError(
                        f"sidecar may not set structural field {field!r} "
                        f"on rule {rule_name!r}"
                    )
                # Direction labels are only meaningful on bidirectional rules
                # (mirrors load_rules_from_json's rejection on => rules).
                if (field in ("fwd_label", "rev_label")
                        and not metadata.bidirectional):
                    raise ValueError(
                        f"{field} only valid on bidirectional rules; "
                        f"got rule {rule_name!r}"
                    )
                # Changing one half's priority would split the stored -fwd/-rev
                # pair under re-sort; set priority on the source rule instead.
                if field == "priority" and metadata.bidirectional:
                    raise ValueError(
                        f"priority cannot be set via sidecar on bidirectional "
                        f"rule {rule_name!r}; set it on the source rule"
                    )

                if field not in _SIDECAR_MERGEABLE_FIELDS:
                    # B6: unknown -> extra, conflict-checked like known fields
                    # so a second sidecar can't silently overwrite.
                    if field in metadata.extra and metadata.extra[field] != value:
                        raise ValueError(
                            f"sidecar conflict on rule {rule_name!r} extra "
                            f"field {field!r}: existing={metadata.extra[field]!r}, "
                            f"sidecar={value!r}"
                        )
                    metadata.extra[field] = value
                    continue

                # B1: a field is "unset" (fillable) iff it still holds its
                # constructor default; a different already-set value conflicts.
                # This makes falsy defaults like priority=0 fillable instead of
                # spuriously conflicting.
                existing = getattr(metadata, field)
                if existing != getattr(reference, field):
                    if existing != value:
                        raise ValueError(
                            f"sidecar conflict on rule {rule_name!r} "
                            f"field {field!r}: existing={existing!r}, "
                            f"sidecar={value!r}"
                        )
                    continue  # same value; harmless
                # Preserve the "examples is always a list" invariant (B5):
                # a sidecar `examples: null` normalises to [] like __init__.
                if field == "examples" and value is None:
                    value = []
                setattr(metadata, field, value)
                changed = True
                if field == "priority":
                    priority_touched = True

            if validate_examples:
                self._validate_rule_examples(self._rules[idx], metadata)

        # priority is the one mergeable field that affects firing order; re-sort
        # when it changed, and drop the cached simplifier on any change.
        if priority_touched:
            self._sort_by_priority()
        if changed:
            self._simplifier = None
        return self

    def _install(self, parsed, validate_examples: bool = False) -> 'RuleEngine':
        """Single insertion point for every loader. ATOMIC.

        When ``validate_examples`` is set, EVERY pair is validated before
        ANY is committed, so a mid-batch ``ExampleValidationError`` leaves
        the engine exactly as it was: no half-loaded rules, no stale name
        index. (Example validation is per-rule-isolated -- it instantiates
        the rule against its own examples -- so validate-then-commit is
        sound.) Then append all pairs, re-sort by priority (which rebuilds
        the name index from scratch), and invalidate the cached simplifier.
        Centralising this keeps the four public loaders and ``add_rule``
        from drifting apart.
        """
        parsed = list(parsed)
        if validate_examples:
            for metadata, rule in parsed:
                self._validate_rule_examples(rule, metadata)
        for metadata, rule in parsed:
            self._rules.append(rule)
            self._metadata.append(metadata)
        self._sort_by_priority()  # rebuilds self._rule_names from self._metadata
        self._simplifier = None   # Invalidate cached simplifier
        return self

    def load_dsl(self, text: str, validate_examples: bool = True) -> 'RuleEngine':
        """Load rules from DSL text. Validates examples by default."""
        return self._install(load_rules_from_dsl(text), validate_examples)

    def load_file(self, path: Union[str, Path],
                  validate_examples: bool = True) -> 'RuleEngine':
        """Load rules from a file (.rules/.json/.manifest).

        Validates examples by default. If the file carries manifest
        directives (``:requires``/``:theory``/...), they are PARSED and
        STORED on ``self.manifest`` for inspection but NOT applied -- a
        plain load never silently installs a prelude, sets a theory, or
        merges a sidecar. Use ``RuleEngine.from_manifest`` for assembly.
        """
        path = Path(path)
        if path.suffix != ".json":
            from .manifest import parse_manifest
            try:
                self.manifest = parse_manifest(path.read_text())
            except ValueError:
                # load_file stays LENIENT: a pre-manifest .rules file may
                # carry a stray ':'-line (a note, a typo, a future
                # directive). The old loader silently ignored such lines, so
                # load_file must not regress to raising on them -- it applies
                # nothing from the manifest anyway. ``from_manifest`` parses
                # strictly and DOES raise, which is where typos are caught.
                self.manifest = None
        return self._install(load_rules_from_file(path), validate_examples)

    def with_theory(self, theory) -> 'RuleEngine':
        """Set the session theory (a ``normalize.Theory``) and return self.

        Public setter for the ``_theory`` slot threaded into
        ``solve(normalize_between=True)`` and available for explicit
        ``normalize`` passes. Invalidates the cached simplifier.
        """
        self._theory = theory
        self._simplifier = None
        return self

    def _canonicalize(self, expr: ExprType) -> ExprType:
        """Canonical form of ``expr`` under the engine's theory, else unchanged.

        Identity function when no theory is set (``self._theory is None``) --
        this is the backward-compat path that keeps the no-theory reasoning
        behavior byte-for-byte unchanged. When a theory is set, returns
        ``normalize(expr, theory)``, which is idempotent and confluent, so the
        result is the single, well-defined IDENTITY of the expression for the
        reasoning layer (equivalents / prove_equal / minimize key on it).

        The theory is DATA; no operator is special-cased here.
        """
        if self._theory is None:
            return expr
        from .normalize import normalize as _normalize
        return _normalize(expr, self._theory)

    def missing_fold_ops(self) -> List[str]:
        """Fold-op names invoked by loaded rules but absent from the prelude.

        Walks every rule's skeleton AND guard condition for ``(! op ...)``
        compute heads and returns, sorted, those not in the installed
        prelude. Empty when no prelude is installed only if no rule computes;
        otherwise a non-empty result is the silent-junk footgun (a missing
        skeleton op survives as a literal compound) made visible.
        """
        from .manifest import collect_fold_ops
        installed = set(self._fold_funcs or {})
        used = set()
        for _idx, rule, meta in self.iter_rules():
            _pattern, skeleton = rule
            used |= collect_fold_ops(skeleton)
            if meta.condition is not None:
                used |= collect_fold_ops(meta.condition)
        return sorted(used - installed)

    @classmethod
    def from_manifest(cls, path: Union[str, Path]) -> 'RuleEngine':
        """Assemble a complete engine from a rule-set manifest file.

        Parses the file's manifest directives, then: installs the combined
        ``:requires`` prelude bundles; sets the ``:theory``; loads the rules
        (the file's ``:include``d body and any inline rules); merges the
        ``:metadata`` sidecar with examples validated against the assembled
        prelude. Finally runs a FAIL-LOUD audit: every ``(! op ...)`` head
        across all skeletons and guards, plus every ``:requires-ops`` name,
        must resolve in the assembled prelude -- otherwise raises
        ``ValueError`` naming the missing ops. The ``:driver``/``:goal``
        hints are stored on ``engine.manifest`` (data only).
        """
        from .manifest import parse_manifest
        from .normalize import Theory
        from .rewriter import PRELUDE_BUNDLES, combine_preludes

        path = Path(path)
        manifest = parse_manifest(path.read_text())

        engine = cls()
        if manifest.requires:
            bundles = [PRELUDE_BUNDLES[name] for name in manifest.requires]
            combined = combine_preludes(*bundles)
            # An empty result (e.g. :requires none) installs NO prelude, so
            # has_fold_funcs() stays False -- a domain that declares it needs
            # no computation should not look like it has an empty one.
            if combined:
                engine.with_prelude(combined)
        if manifest.theory is not None:
            theory_path = path.parent / manifest.theory
            engine.with_theory(Theory.from_json(theory_path.read_text()))

        # Load rules WITHOUT example validation (the sidecar supplies the
        # examples, and the prelude must be set before they validate).
        engine._install(load_rules_from_file(path), validate_examples=False)

        if manifest.metadata is not None:
            meta_path = path.parent / manifest.metadata
            engine.load_metadata_json(meta_path.read_text(),
                                      validate_examples=True)

        # Validate ANY examples now present against the assembled prelude --
        # not only the sidecar's. Rules were installed with
        # validate_examples=False (the prelude was not yet set), and
        # load_metadata_json only validates the names it carries, so examples
        # embedded in the :include'd rules file itself would otherwise never
        # be checked. This makes the spec's "examples validate against the
        # assembled prelude" hold for embedded examples too, matching the
        # validation a plain load_file(validate_examples=True) would do.
        engine.validate_examples()

        # Fail-loud audit: declared + structurally-used ops must all resolve.
        installed = set(engine._fold_funcs or {})
        missing = (set(engine.missing_fold_ops())
                   | (set(manifest.requires_ops) - installed))
        if missing:
            raise ValueError(
                f"manifest {path.name}: rules require fold ops absent from "
                f"the assembled prelude {sorted(manifest.requires) or '[]'}: "
                f"{sorted(missing)}. Add the bundle that provides them to "
                f":requires, or load this domain manually if the ops live in "
                f"a custom code prelude.")

        engine.manifest = manifest
        return engine

    def load_rules(self, rules: List[List]) -> 'RuleEngine':
        """Load rules from a Python list (without metadata)."""
        return self._install([(RuleMetadata(), rule) for rule in rules])

    def load_rules_from_json(self, text: str,
                             validate_examples: bool = True) -> 'RuleEngine':
        """Load rules from JSON text. Validates examples by default.

        The module-level ``load_rules_from_json`` function is pure (no
        validation); this engine method is the validation entry point.
        """
        return self._install(load_rules_from_json(text), validate_examples)

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
                 description: Optional[str] = None,
                 priority: int = 0,
                 condition: Optional[ExprType] = None,
                 tags: Optional[List[str]] = None,
                 category: Optional[str] = None,
                 reasoning: Optional[str] = None,
                 examples: Optional[List[Dict[str, Any]]] = None,
                 validate_examples: bool = True) -> 'RuleEngine':
        """Add a single rule with optional metadata.

        v0.7 fields: ``category``, ``reasoning``, ``examples``. When
        ``examples`` are provided and ``validate_examples`` is True
        (default), each example is checked against the rule before
        installation.

        For bidirectional rule construction, use ``load_dsl``,
        ``load_rules_from_json``, or ``_build_bidirectional_rules`` directly.
        ``add_rule`` is for unidirectional rules only.
        """
        rule = [pattern, skeleton]
        metadata = RuleMetadata(
            name=name,
            description=description,
            priority=priority,
            condition=condition,
            tags=tags,
            category=category,
            reasoning=reasoning,
            examples=examples,
        )
        return self._install([(metadata, rule)], validate_examples)

    def get_rule(self, name: str) -> Optional[Tuple[List, RuleMetadata]]:
        """Get a rule and its metadata by name."""
        if name in self._rule_names:
            idx = self._rule_names[name]
            return self._rules[idx], self._metadata[idx]
        return None

    def get_metadata(self, index: int) -> RuleMetadata:
        """Get metadata for a rule by index."""
        return self._metadata[index] if index < len(self._metadata) else RuleMetadata()

    def iter_rules(self) -> Iterator[Tuple[int, List, RuleMetadata]]:
        """Yield ``(index, [pattern, skeleton], metadata)`` for every stored rule.

        Storage (priority) order; includes rules in disabled groups. Use
        ``rule_set()`` for the active (group-filtered) view. This is the
        public read surface for callers (CLI, MCP tools) that previously
        reached into ``_rules``/``_metadata`` directly.
        """
        for i, (rule, meta) in enumerate(zip(self._rules, self._metadata)):
            yield i, rule, meta

    def hook_counts(self) -> Dict[str, int]:
        """Registered-handler count per hook event.

        Derived from ``_HOOK_EVENTS`` (the canonical event list), so a
        future event is covered automatically.
        """
        return {ev: self._hooks.count(ev) for ev in self._HOOK_EVENTS}

    def has_fold_funcs(self) -> bool:
        """True when a prelude (fold functions) is installed."""
        return self._fold_funcs is not None

    def fold_op_names(self) -> List[str]:
        """The names of the installed fold operations, sorted.

        Public state API (with ``iter_rules``/``hook_counts``/...): lets a
        caller -- the MCP ``get_status`` tool in particular -- discover
        which ``(! op ...)`` computations the loaded rules may invoke,
        without reaching into ``_fold_funcs``. Empty when no prelude is
        installed.
        """
        return sorted(self._fold_funcs) if self._fold_funcs else []

    def has_theory(self) -> bool:
        """True when a session Theory (operator-signature data) is set."""
        return self._theory is not None

    def reset(self, fold_funcs: Optional[FoldFuncsType] = None) -> 'RuleEngine':
        """Reset to a fresh-engine state, optionally installing a prelude.

        Clears rules, metadata, the name index, the cached simplifier,
        disabled groups, counters, cancellation, ALL hooks, and the session
        theory; installs ``fold_funcs`` as the new prelude (None for a pure
        rewriting engine). The single public counterpart of attribute-level
        surgery: any future per-session state field must be cleared here.
        """
        self._rules = []
        self._metadata = []
        self._rule_names = {}
        self._simplifier = None
        self._disabled_groups = set()
        self._step_count = 0
        self._cancel_requested = False
        self._hooks.clear()
        self._theory = None
        self.manifest = None
        self._fold_funcs = fold_funcs
        return self

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

        # Guards must be decidable: every (! op ...) compute node in the
        # condition must reference an op the engine can fold. An op absent
        # from the active prelude (and not supplied by a resolver) would
        # otherwise leave an unfolded compound that _condition_truthy reads
        # as truthy, silently passing a bogus guard. Raise instead.
        undefined = self._undefined_guard_ops(condition)
        if undefined:
            raise ValueError(
                f"guard references undefined op {sorted(undefined)!r}; "
                f"guards must be decidable (add the op to the prelude)"
            )

        # Instantiate the condition with bindings, then apply the shared
        # truthiness rule (bool as-is; 0/""/[] falsy; everything else truthy).
        result = instantiate(condition, bindings, self._fold_funcs,
                             undefined_op_resolver=self._undefined_op_resolver,
                             fold_error_resolver=self._fold_error_resolver)
        return _condition_truthy(result)

    def _undefined_guard_ops(self, condition: ExprType) -> set:
        """Return the set of compute-op names in ``condition`` that the
        engine cannot decide.

        Walks every ``(! op ...)`` node. An op is undecidable when it is
        absent from the active prelude and no ``undefined_op_resolver``
        can supply it. The resolver, when present, is treated as able to
        supply any op (it is consulted lazily during instantiation), so a
        configured resolver suppresses the raise and defers to runtime.
        """
        undefined = set()

        has_resolver = self._hooks.count("undefined_op") > 0

        def walk(node):
            if not isinstance(node, list) or not node:
                return
            if skeleton_compute(node):
                op = node[1]
                if (op not in (self._fold_funcs or {})
                        and not has_resolver):
                    undefined.add(op)
                for arg in node[2:]:
                    walk(arg)
                return
            for child in node:
                walk(child)

        walk(condition)
        return undefined

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

    def apply_once(self, expr: ExprType, groups: Optional[List[str]] = None,
                   _top_level: bool = True,
                   path: Optional[List[int]] = None) -> Tuple[ExprType, Optional[RuleMetadata]]:
        """
        Apply at most one rule to the expression.

        Tries each rule in order and returns after the first successful application.
        Does not recurse into subexpressions. Respects conditional guards and group filters.

        Args:
            expr: Expression to rewrite.
            groups: If specified, only use rules from these groups.
                    If None, use all rules except those in disabled groups.
            path: Optional redex path from the running root ([] for root-level,
                  [i] for a child, etc.). Used to populate RewriteStep.path so
                  traces produced via the "once" strategy reconstruct correctly.

        Returns:
            Tuple of (result, metadata) where:
                - result: The rewritten expression (or original if no rule applied)
                - metadata: RuleMetadata of applied rule, or None if no rule applied

        Example:
            result, applied = engine.apply_once(expr)
            if applied:
                print(f"Applied rule: {applied.name}")
        """
        if _top_level:
            self._step_count = 0
            self._cancel_requested = False
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
                if not self._check_should_fire(rule, metadata, expr, bindings):
                    continue
                result = instantiate(skeleton, bindings, self._fold_funcs,
                                     undefined_op_resolver=self._undefined_op_resolver,
                                     fold_error_resolver=self._fold_error_resolver)
                if result != expr:
                    guard = self._evaluate_guard(metadata.condition, bindings)
                    step = self._build_step(
                        rule_idx, rule, metadata, expr, result, bindings,
                        path=path, guard=guard,
                    )
                    self._fire_rule_applied(step, expr_path=path)
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
            return self._simplify_with_trace(expr, max_steps, groups=groups,
                                             strategy=strategy)

        # Check if we need slow path (conditions or groups)
        has_conditions = any(m.condition is not None for m in self._metadata)
        has_groups = groups is not None or self._disabled_groups

        if strategy == "exhaustive":
            if has_conditions or has_groups:
                return self._simplify_exhaustive(expr, max_steps, groups=groups)
            else:
                # Use fast path when no conditions, no groups, AND no
                # engine-fired hooks are registered. Hooks need engine
                # context, which the rewriter() factory does not have, so
                # we fall back to _simplify_exhaustive when any hook
                # event has subscribers.
                # Derive the bailout list from _HOOK_EVENTS so new events
                # (added by future tasks) are automatically covered. Exclude
                # max_depth which fires only in equivalents/prove_equal/
                # minimize, not in simplify.
                hooks_active = any(
                    self._hooks.count(ev) > 0
                    for ev in RuleEngine._HOOK_EVENTS
                    if ev != "max_depth"
                )
                if hooks_active:
                    return self._simplify_exhaustive(expr, max_steps, groups=groups)
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

    def _simplify_once(self, expr: ExprType, groups: Optional[List[str]] = None,
                       _top_level: bool = True,
                       path: Optional[List[int]] = None) -> ExprType:
        """Apply at most one rule anywhere in the expression tree.

        ``path`` is the integer-index path from the running root to the
        current position being examined ([] at the root, [i] for a child).
        Threading it through to ``apply_once`` ensures that steps emitted
        by the "once" strategy carry the correct redex path for
        ``to_global_sequence`` reconstruction.
        """
        if path is None:
            path = []
        if _top_level:
            self._step_count = 0
            self._cancel_requested = False
        # Try to apply a rule at the top level
        result, applied = self.apply_once(expr, groups=groups, _top_level=False,
                                          path=path)
        if applied:
            return result

        # If no rule applied at top level, try children (depth-first)
        if isinstance(expr, list) and len(expr) > 0:
            for i, child in enumerate(expr):
                new_child = self._simplify_once(child, groups=groups,
                                                _top_level=False,
                                                path=path + [i])
                if new_child != child:
                    # Found a rewrite - apply it and return
                    return expr[:i] + [new_child] + expr[i+1:]

        return expr

    def _fire_rule_applied(self, step: RewriteStep, *, depth: int = 0,
                           expr_path: Optional[List[int]] = None) -> bool:
        """Fire the rule_applied event with a standard HookContext.

        Returns True if any observer requested cancellation via ctx.cancel().
        Also sets ``self._cancel_requested = True`` so callers higher up the
        stack (e.g. strategy drivers calling pass methods in a loop) can
        detect cancellation without threading the return value through.

        Increments ``self._step_count`` before constructing the context so
        that ``ctx.step_count`` reflects the count after this firing (1-based).
        ``self._step_count`` is reset to 0 at the top of every top-level call
        (simplify, equivalents, prove_equal, random_walk, random_equivalent).

        ``depth`` is the recursion depth of the expression position being
        matched (root = 0, root's child = 1, etc.). Pass-methods thread this
        value through their recursive calls.

        ``expr_path`` is the integer-index path from the running root to the
        redex position ([] for a root-level redex, [1] for the first operand,
        etc.). Strategy drivers populate this from their accumulated path;
        callers that do not thread a path leave it None (treated as []).
        """
        self._step_count += 1
        if not self._hooks.count("rule_applied"):
            return False
        ctx = HookContext(
            engine=self,
            expr_path=list(expr_path) if expr_path is not None else [],
            depth=depth,
            step_count=self._step_count,
            event_name="rule_applied",
        )
        self._hooks.run_observers("rule_applied", step, ctx)
        if ctx.cancelled:
            self._cancel_requested = True
            return True
        return False

    def _fire_no_match(self, expr, *, depth: int = 0) -> Optional[Resolution]:
        """Fire the no_match event when no rule matches at the current
        compound position.

        Returns the Resolution if a resolver provided one, else None.
        Atoms (constants, variables, empty list) do not fire no_match:
        rules apply at compound positions, not at leaves.

        ``ctx.cancel()`` from a resolver propagates to ``self._cancel_requested``,
        consistent with the other event helpers.
        Note: Resolution.abort is handled at the call site (returns current).
        Both paths now consistently set _cancel_requested when cancellation
        is requested.
        """
        if not isinstance(expr, list) or not expr:
            return None
        if not self._hooks.count("no_match"):
            return None
        ctx = HookContext(
            engine=self,
            expr_path=[],
            depth=depth,
            step_count=self._step_count,
            event_name="no_match",
        )
        resolution = self._hooks.run_resolvers("no_match", expr, ctx)
        if ctx.cancelled:
            self._cancel_requested = True
        return resolution

    def _fire_cycle(self, expr, visited_states) -> None:
        """Fire the cycle event when visited-set cycle detection catches a repeat.

        ``visited_states`` is the unordered set of expression states seen
        so far in the current simplify call (as a list, snapshotted from
        the engine's visited set). It is *not* an ordered path: the
        engine does not retain order. Resolvers needing reconstruction
        of the cycle must build their own history.

        Resolvers can return Resolution(abort=True) to escalate the cycle
        into a propagated cancellation. ctx.cancel() does the same.
        """
        if not self._hooks.count("cycle"):
            return
        ctx = HookContext(
            engine=self,
            expr_path=[],
            depth=0,
            step_count=self._step_count,
            event_name="cycle",
        )
        resolution = self._hooks.run_resolvers(
            "cycle", expr, list(visited_states), ctx
        )
        if ctx.cancelled or (resolution is not None and resolution.abort):
            self._cancel_requested = True

    def _fire_fixpoint(self, expr) -> None:
        """Fire the fixpoint event when the engine converges (no rule fires).

        Observer-only: there is no Resolution mechanism. ``ctx.cancel()``
        from a fixpoint observer is silently ignored because the engine
        has already finished computing the result before this fires;
        cancellation has nothing to abort. Observers that need to influence
        a downstream operation (e.g., a SequencedEngine pipeline) should
        return their decision through a separate channel (e.g., raising
        an exception).
        """
        if not self._hooks.count("fixpoint"):
            return
        ctx = HookContext(
            engine=self,
            expr_path=[],
            depth=0,
            step_count=self._step_count,
            event_name="fixpoint",
        )
        self._hooks.run_observers("fixpoint", expr, ctx)
        # Note: ctx.cancelled is intentionally not propagated. The engine
        # has already converged; there is nothing to cancel.

    def _fire_max_depth(self, expr, depth) -> Optional[Resolution]:
        """Bridge to on_max_depth hooks.

        Fires when equivalents/prove_equal/minimize exhaust their depth
        budget. Resolvers can return Resolution(allow_more=True) to extend
        the budget once per call.

        random_walk uses max_steps (a different concept) and does not fire
        this event.

        Honors ctx.cancel() and Resolution(abort=True) by setting
        self._cancel_requested.
        """
        if not self._hooks.count("max_depth"):
            return None
        ctx = HookContext(
            engine=self,
            expr_path=[],
            depth=depth,
            step_count=self._step_count,
            event_name="max_depth",
        )
        resolution = self._hooks.run_resolvers("max_depth", expr, depth, ctx)
        if ctx.cancelled or (resolution is not None and resolution.abort):
            self._cancel_requested = True
        return resolution

    def _undefined_op_resolver(self, op: str, args) -> Optional[Resolution]:
        """Bridge from rewriter.instantiate to on_undefined_op hooks.

        Dual-path install for ``Resolution(fold_funcs={op: handler})``:
          1. ``instantiate`` updates ``fold_funcs`` (its parameter) in-place
             when ``fold_funcs is not None``. This makes the handler available
             for the current call.
          2. This bridge also installs into ``self._fold_funcs``, initializing
             it to ``{}`` if it was ``None`` (instantiate cannot do that
             because Python pass-by-object means assigning to its parameter
             doesn't reach the engine's attribute). It also invalidates
             ``self._simplifier`` so the cached fast-path simplifier
             rebuilds with the new fold_funcs on the next call.

        The two paths overlap (idempotent updates) but are both needed: path 1
        for the current call when ``self._fold_funcs`` was already a dict,
        path 2 to handle the fresh-engine case where ``self._fold_funcs`` was
        ``None``. Honors ``ctx.cancel()`` by setting ``self._cancel_requested``.
        """
        if not self._hooks.count("undefined_op"):
            return None
        ctx = HookContext(
            engine=self,
            expr_path=[],
            depth=0,
            step_count=self._step_count,
            event_name="undefined_op",
        )
        resolution = self._hooks.run_resolvers("undefined_op", op, args, ctx)
        if ctx.cancelled or (resolution is not None and resolution.abort):
            self._cancel_requested = True
        if resolution is not None and resolution.fold_funcs is not None:
            # Permanent install at the engine level.
            if self._fold_funcs is None:
                self._fold_funcs = {}
            self._fold_funcs.update(resolution.fold_funcs)
            self._simplifier = None  # Invalidate fast-path cache.
        return resolution

    def _fold_error_resolver(self, op: str, args, exception) -> Optional[Resolution]:
        """Bridge from rewriter.instantiate to on_fold_error hooks.

        Fired when an installed fold handler raises an exception. Resolvers
        can return Resolution(value=...) for a fallback or None to fall
        through to the existing "leave as compound" behavior.

        Resolution(abort=True) and ctx.cancel() both propagate to
        self._cancel_requested, halting the rewrite at the next strategy
        loop checkpoint.
        """
        if not self._hooks.count("fold_error"):
            return None
        ctx = HookContext(
            engine=self,
            expr_path=[],
            depth=0,
            step_count=self._step_count,
            event_name="fold_error",
        )
        resolution = self._hooks.run_resolvers(
            "fold_error", op, args, exception, ctx
        )
        if ctx.cancelled or (resolution is not None and resolution.abort):
            self._cancel_requested = True
        return resolution

    def _install_resolver_rules(self, rules,
                                metadata: Optional[Dict[str, Any]] = None) -> int:
        """Install rules provided by a Resolver mid-rewrite.

        Each entry in ``rules`` is a (RuleMetadata, [pattern, skeleton]) tuple,
        matching the shape produced by parse_rule_line.

        ``metadata`` from the Resolution is merged into each rule's
        ``RuleMetadata.extra`` dict for later introspection.

        Rules whose names already exist in the engine are skipped (idempotent
        installation). This prevents unbounded accumulation when a resolver
        keeps returning the same rules. Anonymous rules (name=None) are
        always added.

        Returns the number of newly-added rules. Callers can use this to
        detect "resolver returned rules but none were new" and avoid
        spinning the outer loop.
        """
        added = 0
        for meta, rule in rules:
            if meta.name and meta.name in self._rule_names:
                continue  # Already installed; skip.
            if metadata:
                # Construct a new RuleMetadata with merged extra rather than
                # mutating the caller's object.
                merged_extra = dict(meta.extra or {})
                merged_extra.update(metadata)
                meta = RuleMetadata(
                    name=meta.name,
                    description=meta.description,
                    tags=list(meta.tags) if meta.tags else None,
                    condition=meta.condition,
                    priority=meta.priority,
                    bidirectional=meta.bidirectional,
                    direction=meta.direction,
                    extra=merged_extra,
                )
            self._rules.append(rule)
            self._metadata.append(meta)
            added += 1
        if added > 0:
            self._sort_by_priority()  # Rebuilds _rule_names correctly.
            self._simplifier = None  # Cache invalidation.
        return added

    def _check_should_fire(self, rule, metadata, expr, bindings) -> bool:
        """Check if all should_fire decisions allow this rule to fire.

        Returns True if no decisions are registered or every registered
        decision returns truthy. AND-gate semantics: any False vetoes.

        Falsy non-True returns (None, 0, "", []) all veto. Decisions that
        forget to return a value will silently veto every rule, which is
        the standard chain-of-decisions semantics. Use ``return True``
        explicitly.

        ``ctx.cancel()`` from a decision sets ``self._cancel_requested``;
        the strategy driver loop reads that flag to abort.

        Layered on top of the DSL ``condition``/``when`` clause: rule-author
        guards stay in the DSL (data), engine-user predicates go in
        ``should_fire`` (code). A rule fires iff its ``condition`` passes
        AND every ``should_fire`` hook returns True.
        """
        if not self._hooks.count("should_fire"):
            return True
        ctx = HookContext(
            engine=self,
            expr_path=[],
            depth=0,
            step_count=self._step_count,
            event_name="should_fire",
        )
        result = self._hooks.run_decisions(
            "should_fire", rule, metadata, expr, bindings, ctx
        )
        if ctx.cancelled:
            self._cancel_requested = True
            return False
        return result

    def _build_step(self, rule_idx, rule, metadata, before, after, bindings,
                    *, path=None, guard=None):
        """Construct a fully-populated situated RewriteStep.

        Derives rule_id (rule_identity over pattern/skeleton), direction
        (metadata.direction), serialized bindings, kind, and rationale
        (reasoning else category). ``guard`` is a {"condition","result"}
        dict when a condition was evaluated, else None. ``path`` defaults to [].
        """
        from .trace import rule_identity
        pattern, skeleton = rule
        return RewriteStep(
            rule_index=rule_idx,
            metadata=metadata,
            before=before,
            after=after,
            rule_id=rule_identity(metadata, pattern, skeleton),
            direction=metadata.direction,
            bindings=bindings.to_dict() if bindings is not None else None,
            path=list(path) if path is not None else [],
            kind="rule",
            guard=guard,
            rationale=metadata.reasoning or metadata.category,
        )

    def _evaluate_guard(self, condition, bindings):
        """Return {"condition": <instantiated>, "result": <bool>} when
        ``condition`` is not None, else None. Same instantiation/truthiness
        path as _check_condition."""
        if condition is None:
            return None
        instantiated = instantiate(
            condition, bindings, self._fold_funcs,
            undefined_op_resolver=self._undefined_op_resolver,
            fold_error_resolver=self._fold_error_resolver,
        )
        return {"condition": instantiated, "result": _condition_truthy(instantiated)}

    def _simplify_exhaustive(self, expr: ExprType, max_steps: int,
                              groups: Optional[List[str]] = None,
                              _top_level: bool = True,
                              path: Optional[List[int]] = None) -> ExprType:
        """Exhaustive strategy with condition and group support.

        Uses a visited set to terminate on cycles (e.g. bidirectional rules
        like ``(+ ?x ?y) <=> (+ :y :x)`` that bounce between two equivalent
        forms). Without this, max_steps would simply bound an oscillation.

        Rule firings are reported via ``_fire_rule_applied``, which broadcasts
        to all registered ``on_rule_applied`` hooks. Tracing is done by
        registering a temporary hook in ``_simplify_with_trace`` rather than
        passing a listener through.

        ``path`` is the integer-index path from the running root to the
        position being simplified: [] at the root, [i] for a direct child,
        etc. Used to stamp ``RewriteStep.path`` and populate
        ``HookContext.expr_path``.
        """
        if path is None:
            path = []
        current = expr
        visited = set()
        if _top_level:
            self._step_count = 0
            self._cancel_requested = False
        _cycle_break = False
        converged = False
        # Per-call cap on resolver-driven rule retries. Catches buggy LLM
        # resolvers that keep returning rules without making progress.
        max_resolver_retries = 100
        resolver_retries = 0
        for _ in range(max_steps):
            if self._cancel_requested:
                return current
            key = _expr_to_tuple(current)
            if key in visited:
                self._fire_cycle(current, visited)
                _cycle_break = True
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
                    if not self._check_should_fire(rule, metadata, current, bindings):
                        continue
                    new_expr = instantiate(skeleton, bindings, self._fold_funcs,
                                           undefined_op_resolver=self._undefined_op_resolver,
                                           fold_error_resolver=self._fold_error_resolver)
                    if new_expr != current:
                        guard = self._evaluate_guard(metadata.condition, bindings)
                        step = self._build_step(
                            rule_idx, rule, metadata, current, new_expr, bindings,
                            path=path, guard=guard,
                        )
                        if self._fire_rule_applied(step, expr_path=path):
                            return new_expr  # Hook requested cancellation after this step.
                        current = new_expr
                        changed = True
                        break

            if not changed:
                # No rule matched at this compound position. Fire no_match.
                resolution = self._fire_no_match(current)
                if self._cancel_requested:
                    return current
                if resolution is not None:
                    if resolution.abort:
                        return current
                    if resolution.value is not None:
                        current = resolution.value
                        # The visited set at the top of the loop catches
                        # the case where a resolver returns a value that
                        # was already seen (or the same value repeatedly),
                        # so the engine terminates within max_steps even
                        # if the resolver is stuck.
                        continue  # Outer loop tries to apply rules to the new value.
                    if resolution.rules is not None:
                        added = self._install_resolver_rules(
                            resolution.rules, metadata=resolution.metadata
                        )
                        if added == 0:
                            # Resolver returned only rules already installed.
                            # No progress is possible; fall through to the
                            # default no-match behavior (recurse into children).
                            pass
                        else:
                            # New rules added; clear current expr from visited
                            # so the retry can re-evaluate with the new rules.
                            resolver_retries += 1
                            if resolver_retries > max_resolver_retries:
                                raise ResolverLoopError(
                                    f"resolver retry cap "
                                    f"({max_resolver_retries}) exceeded "
                                    f"for no_match at {current!r}"
                                )
                            visited.discard(_expr_to_tuple(current))
                            continue
                    # fold_funcs path is T10.

                # Recursively simplify subexpressions
                if isinstance(current, list) and len(current) > 0:
                    new_children = []
                    subexpr_changed = False
                    for idx, child in enumerate(current):
                        new_child = self._simplify_exhaustive(
                            child, max_steps // 10 or 1, groups=groups, _top_level=False,
                            path=path + [idx],
                        )
                        new_children.append(new_child)
                        if new_child != child:
                            subexpr_changed = True
                    if subexpr_changed:
                        current = new_children
                        continue
                converged = True
                break

        if converged and not _cycle_break and not self._cancel_requested:
            self._fire_fixpoint(current)
        return current

    def _simplify_bottomup(self, expr: ExprType, max_steps: int, groups: Optional[List[str]] = None) -> ExprType:
        """Bottom-up strategy: simplify children first, then parent.

        Cycle-detected so non-confluent rule sets terminate cleanly.
        """
        visited = set()
        self._step_count = 0
        self._cancel_requested = False
        cycle_break = False
        converged = False
        for _ in range(max_steps):
            key = _expr_to_tuple(expr)
            if key in visited:
                self._fire_cycle(expr, visited)
                cycle_break = True
                break
            visited.add(key)
            new_expr = self._bottomup_pass(expr, groups=groups)
            if new_expr == expr:
                converged = True
                break  # natural convergence
            expr = new_expr
            if self._cancel_requested:
                break
        if converged and not cycle_break and not self._cancel_requested:
            self._fire_fixpoint(expr)
        return expr

    def _bottomup_pass(self, expr: ExprType, groups: Optional[List[str]] = None,
                        depth: int = 0,
                        path: Optional[List[int]] = None) -> ExprType:
        """Single bottom-up pass: simplify children, then apply rules to parent.

        no_match resolvers may install rules and request retry. Retry loops
        within this pass invocation up to max_resolver_retries times before
        raising ResolverLoopError, matching the safeguard in
        _simplify_exhaustive.

        ``path`` is the integer-index path from the running root to the
        position being simplified: [] at the root, [i] for a direct child,
        etc. Used to stamp ``RewriteStep.path`` and populate
        ``HookContext.expr_path``.
        """
        if path is None:
            path = []
        # Base case: atoms can't be simplified structurally
        if not isinstance(expr, list) or len(expr) == 0:
            return expr

        # First, recursively simplify all children (threading their path)
        new_children = [
            self._bottomup_pass(child, groups=groups, depth=depth + 1, path=path + [i])
            for i, child in enumerate(expr)
        ]
        current = new_children

        # Try to apply rules to the parent. Loop allows a resolver returning
        # rules to retry without unbounded recursion.
        max_resolver_retries = 100
        resolver_retries = 0
        while True:
            for rule_idx, rule in enumerate(self._rules):
                if self._cancel_requested:
                    return current
                metadata = self._metadata[rule_idx]
                if not self._is_rule_active(metadata, groups):
                    continue
                pattern, skeleton = rule
                bindings = _match_internal(pattern, current)
                if bindings is None:
                    continue
                if not self._check_condition(metadata.condition, bindings):
                    continue
                if not self._check_should_fire(rule, metadata, current, bindings):
                    continue
                result = instantiate(
                    skeleton, bindings, self._fold_funcs,
                    undefined_op_resolver=self._undefined_op_resolver,
                    fold_error_resolver=self._fold_error_resolver,
                )
                if result != current:
                    guard = self._evaluate_guard(metadata.condition, bindings)
                    step = self._build_step(
                        rule_idx, rule, metadata, current, result, bindings,
                        path=path, guard=guard,
                    )
                    self._fire_rule_applied(step, depth=depth, expr_path=path)
                    return result

            # No rule fired. Try the no_match resolver.
            resolution = self._fire_no_match(current, depth=depth)
            if self._cancel_requested:
                return current
            if resolution is None:
                return current
            if resolution.abort:
                return current
            if resolution.value is not None:
                return resolution.value
            if resolution.rules is not None:
                added = self._install_resolver_rules(
                    resolution.rules, metadata=resolution.metadata
                )
                if added == 0:
                    return current  # No progress; give up on retry.
                resolver_retries += 1
                if resolver_retries > max_resolver_retries:
                    raise ResolverLoopError(
                        f"resolver retry cap ({max_resolver_retries}) "
                        f"exceeded in _bottomup_pass at {current!r}"
                    )
                # Retry the outer while: scan all rules again with the new set.
                continue
            return current

    def _simplify_topdown(self, expr: ExprType, max_steps: int, groups: Optional[List[str]] = None) -> ExprType:
        """Top-down strategy: try parent first, then children.

        Cycle-detected so non-confluent rule sets terminate cleanly.
        """
        visited = set()
        self._step_count = 0
        self._cancel_requested = False
        cycle_break = False
        converged = False
        for _ in range(max_steps):
            key = _expr_to_tuple(expr)
            if key in visited:
                self._fire_cycle(expr, visited)
                cycle_break = True
                break
            visited.add(key)
            new_expr = self._topdown_pass(expr, groups=groups)
            if new_expr == expr:
                converged = True
                break  # natural convergence
            expr = new_expr
            if self._cancel_requested:
                break
        if converged and not cycle_break and not self._cancel_requested:
            self._fire_fixpoint(expr)
        return expr

    def _topdown_pass(self, expr: ExprType, groups: Optional[List[str]] = None,
                      depth: int = 0,
                      path: Optional[List[int]] = None) -> ExprType:
        """Single top-down pass: apply rules to parent, then simplify children.

        no_match resolvers may install rules and request retry. Retry loops
        within this pass invocation up to max_resolver_retries times before
        raising ResolverLoopError, matching the safeguard in
        _simplify_exhaustive.

        ``path`` is the integer-index path from the running root to the
        position being simplified: [] at the root, [i] for a direct child,
        etc. Used to stamp ``RewriteStep.path`` and populate
        ``HookContext.expr_path``.
        """
        if path is None:
            path = []
        current = expr

        # Try to apply rules at this node first. Loop allows a resolver
        # returning rules to retry without unbounded recursion.
        max_resolver_retries = 100
        resolver_retries = 0
        while True:
            for rule_idx, rule in enumerate(self._rules):
                if self._cancel_requested:
                    return current
                metadata = self._metadata[rule_idx]
                if not self._is_rule_active(metadata, groups):
                    continue
                pattern, skeleton = rule
                bindings = _match_internal(pattern, current)
                if bindings is None:
                    continue
                if not self._check_condition(metadata.condition, bindings):
                    continue
                if not self._check_should_fire(rule, metadata, current, bindings):
                    continue
                result = instantiate(
                    skeleton, bindings, self._fold_funcs,
                    undefined_op_resolver=self._undefined_op_resolver,
                    fold_error_resolver=self._fold_error_resolver,
                )
                if result != current:
                    guard = self._evaluate_guard(metadata.condition, bindings)
                    step = self._build_step(
                        rule_idx, rule, metadata, current, result, bindings,
                        path=path, guard=guard,
                    )
                    self._fire_rule_applied(step, depth=depth, expr_path=path)
                    return result  # Return immediately - will be called again

            # No rule fired. Try the no_match resolver.
            resolution = self._fire_no_match(current, depth=depth)
            if self._cancel_requested:
                return current
            if resolution is None:
                break
            if resolution.abort:
                return current
            if resolution.value is not None:
                return resolution.value
            if resolution.rules is not None:
                added = self._install_resolver_rules(
                    resolution.rules, metadata=resolution.metadata
                )
                if added == 0:
                    break  # No progress; fall through to children.
                resolver_retries += 1
                if resolver_retries > max_resolver_retries:
                    raise ResolverLoopError(
                        f"resolver retry cap ({max_resolver_retries}) "
                        f"exceeded in _topdown_pass at {current!r}"
                    )
                # Retry the outer while: scan all rules again with the new set.
                continue
            break

        # No rule applied at this node - recursively process children
        if isinstance(current, list) and len(current) > 0:
            new_children = [
                self._topdown_pass(child, groups=groups, depth=depth + 1, path=path + [i])
                for i, child in enumerate(current)
            ]
            if new_children != list(current):
                return new_children

        return current

    def _simplify_with_trace(self, expr: ExprType, max_steps: int,
                             groups: Optional[List[str]] = None,
                             strategy: str = "exhaustive") -> Tuple[ExprType, RewriteTrace]:
        """Traced simplification implemented as a temporary on_rule_applied hook.

        A ``RewriteTrace`` instance is registered as an ``on_rule_applied``
        observer for the duration of the call, then deregistered. This means
        ``simplify(trace=True)`` goes through the same code path as any other
        simplify call; no separate listener thread is needed.

        Behavioral compatibility: returns ``(result, RewriteTrace)`` with the
        same trace contents as before.

        ``strategy`` selects the rewriting strategy: "exhaustive" (default),
        "bottomup", or "topdown". The trace receives steps stamped with the
        redex path under whichever strategy is active.
        """
        trace_obj = RewriteTrace()
        trace_obj.initial = expr

        def trace_hook(step, ctx):
            trace_obj(step)

        self.on_rule_applied(trace_hook)
        try:
            if strategy == "exhaustive":
                result = self._simplify_exhaustive(expr, max_steps, groups=groups)
            elif strategy == "once":
                result = self._simplify_once(expr, groups=groups)
            elif strategy == "bottomup":
                result = self._simplify_bottomup(expr, max_steps, groups=groups)
            elif strategy == "topdown":
                result = self._simplify_topdown(expr, max_steps, groups=groups)
            else:
                raise ValueError(f"Unknown strategy: {strategy}. "
                                 f"Valid options: exhaustive, once, bottomup, topdown")
        finally:
            self.off_rule_applied(trace_hook)

        if self._fold_funcs:
            result = self._fold_constants(result)
        trace_obj.final = result
        return result, trace_obj

    def _fold_constants(self, expr: ExprType) -> ExprType:
        """Fold constant expressions using the configured fold_funcs."""
        if self._fold_funcs is None:
            return expr
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

        if all(isinstance(a, NUMERIC_TYPES) for a in args):
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

        Note: ``to_dsl`` serializes only the fields the DSL syntax supports
        (name, priority, description, category, condition, pattern,
        skeleton). ``reasoning`` and ``examples`` are JSON-only metadata
        and are silently omitted on DSL serialization. For lossless
        roundtripping, use ``to_json``/``to_dict``.

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
                hdr = _format_dsl_header(base_name, meta.priority,
                                         base_description, meta.category)
                rule_str = f"{hdr}{pattern_str} {arrow} {skeleton_str}"
                if meta.condition:
                    rule_str += f" when {format_sexpr(meta.condition)}"
                lines.append(rule_str)
                i += 2
                continue

            hdr = _format_dsl_header(meta.name, meta.priority,
                                     meta.description, meta.category)
            rule_str = f"{hdr}{pattern_str} => {skeleton_str}"
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
                    "pattern": _expr_to_jsonable(pattern),
                    "skeleton": _expr_to_jsonable(skeleton),
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
                _emit_metadata_fields(rule_dict, meta)
                # Direction labels (fwd half carries fwd_label; rev half carries rev_label).
                if meta.fwd_label is not None:
                    rule_dict["fwd_label"] = meta.fwd_label
                assert i + 1 < len(self._metadata), (
                    "bidirectional pair invariant violated: fwd at index "
                    f"{i} has no rev half"
                )
                rev_meta = self._metadata[i + 1]
                if rev_meta.rev_label is not None:
                    rule_dict["rev_label"] = rev_meta.rev_label
                rules_list.append(rule_dict)
                i += 2
                continue

            rule_dict = {
                "pattern": _expr_to_jsonable(pattern),
                "skeleton": _expr_to_jsonable(skeleton),
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
            _emit_metadata_fields(rule_dict, meta)
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
        # Copy hooks too so that copy() and `|` are hook-preserving.
        new_engine._hooks._hooks = {k: list(v) for k, v in self._hooks._hooks.items()}
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
        labeled: bool = False,
        _path: Optional[List[int]] = None,
    ):
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
            labeled: If False (default), returns a list of distinct one-step
                rewrite expressions (legacy shape). If True, returns a list of
                ``(new_expr, label)`` edges where ``label`` is a dict with
                keys ``rule_id``, ``direction``, ``bindings``, and ``path``.
            _path: Internal accumulator for the redex position; callers should
                not pass this argument.

        Returns:
            When ``labeled`` is False: list of distinct one-step rewrite
            expressions (legacy shape). When ``labeled`` is True: list of
            ``(new_expr, label)`` tuples.
        """
        if rules is None:
            rules = self.rule_set(groups=groups, bidirectional_only=bidirectional_only)
        if _path is None:
            _path = []

        results = []
        seen: Set[tuple] = set()

        def add_if_new(new_expr: ExprType, label) -> None:
            key = _expr_to_tuple(new_expr)
            if key not in seen:
                seen.add(key)
                if labeled:
                    results.append((new_expr, label))
                else:
                    results.append(new_expr)

        # Try rules at top level
        for rule_idx, rule, metadata in rules:
            pattern, skeleton = rule
            bindings = _match_internal(pattern, expr)
            if bindings is not None:
                if not self._check_condition(metadata.condition, bindings):
                    continue
                if not self._check_should_fire(rule, metadata, expr, bindings):
                    continue
                result = instantiate(skeleton, bindings, self._fold_funcs,
                                     undefined_op_resolver=self._undefined_op_resolver,
                                     fold_error_resolver=self._fold_error_resolver)
                if result != expr:
                    if labeled:
                        from .trace import rule_identity
                        label = {
                            "rule_id": rule_identity(metadata, pattern, skeleton),
                            "direction": metadata.direction,
                            "bindings": bindings.to_dict(),
                            "path": list(_path),
                        }
                    else:
                        label = None
                    add_if_new(result, label)

        # Recursively try rules in subexpressions
        if isinstance(expr, list) and len(expr) > 0:
            for i, child in enumerate(expr):
                child_rewrites = self._all_single_rewrites(
                    child, rules=rules, labeled=labeled, _path=_path + [i]
                )
                if labeled:
                    for new_child, label in child_rewrites:
                        new_expr = expr[:i] + [new_child] + expr[i+1:]
                        add_if_new(new_expr, label)
                else:
                    for new_child in child_rewrites:
                        new_expr = expr[:i] + [new_child] + expr[i+1:]
                        add_if_new(new_expr, None)

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

        When a theory is loaded (see ``with_theory``), each yielded expression
        is the CANONICAL representative of its class under the theory; AC-variants
        that share a normal form are deduplicated. With no theory, the behavior
        is unchanged.

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

        self._step_count = 0
        self._cancel_requested = False
        bidirectional_only = not include_unidirectional

        # Track visited expressions. Under a theory, identity is the NORMAL
        # FORM: the class is the set of canonical representatives. With no
        # theory _canonicalize is the identity, so this is unchanged.
        cexpr = self._canonicalize(expr)
        visited: Set[tuple] = set()
        start_key = _expr_to_tuple(cexpr)
        visited.add(start_key)

        # Count of yielded expressions
        count = 0

        # Yield the starting (canonical) expression
        yield cexpr
        count += 1
        if max_count is not None and count >= max_count:
            return

        # Extension tracker is local to this call; avoids memory leaks and
        # id-reuse hazards from a per-instance dict.
        _depth_extended = False

        # Initialize queue/stack with (canonical expression, depth)
        if strategy == "bfs":
            frontier: deque = deque([(cexpr, 0)])
        else:  # dfs
            frontier: List = [(cexpr, 0)]

        while frontier:
            if self._cancel_requested:
                return

            if strategy == "bfs":
                current, depth = frontier.popleft()
            else:
                current, depth = frontier.pop()

            if depth >= max_depth:
                resolution = self._fire_max_depth(current, depth)
                if (resolution is not None and resolution.allow_more
                        and not _depth_extended):
                    _depth_extended = True
                    max_depth = max_depth * 2 if max_depth > 0 else 1
                if depth >= max_depth:
                    continue

            # Find all single-step rewrites
            rewrites = self._all_single_rewrites(
                current, bidirectional_only, groups, rules=rules
            )

            for new_expr in rewrites:
                cnew = self._canonicalize(new_expr)
                key = _expr_to_tuple(cnew)
                if key not in visited:
                    visited.add(key)
                    yield cnew
                    count += 1

                    if max_count is not None and count >= max_count:
                        return

                    # Add to frontier for further exploration.
                    # BFS vs DFS only differs on the pop side (popleft vs pop);
                    # the append side is symmetric for both structures.
                    frontier.append((cnew, depth + 1))

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

        When a theory is loaded (see ``with_theory``), expressions are compared
        MODULO the theory: AC-variant inputs are proven equal with no search via
        the canonical-key quick check, and the search meets on canonical forms.
        With no theory, the behavior is unchanged.

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
        self._step_count = 0
        self._cancel_requested = False
        bidirectional_only = not include_unidirectional

        # Convert to hashable for set operations. Under a theory the key is
        # the NORMAL FORM, so AC-variant inputs collide here and the quick
        # check below returns a zero-step proof. With no theory _canonicalize
        # is the identity (unchanged behavior).
        ca = self._canonicalize(expr_a)
        cb = self._canonicalize(expr_b)
        key_a = _expr_to_tuple(ca)
        key_b = _expr_to_tuple(cb)

        # Quick check: are they already equal? Under a theory the keys are
        # canonical, so AC-variant inputs land here; report the CANONICAL
        # common form (ca) for consistency with the search branches. With no
        # theory ca is expr_a (identity), so this is byte-for-byte unchanged.
        if key_a == key_b:
            if trace:
                init_step = RewriteStep(
                    rule_index=-1,
                    metadata=RuleMetadata(name=None),
                    before=ca,
                    after=ca,
                    kind="initial",
                )
                path_a: Optional[List] = [init_step]
                path_b: Optional[List] = [init_step]
            else:
                path_a = None
                path_b = None
            return EqualityProof(
                expr_a=expr_a,
                expr_b=expr_b,
                common=ca,
                depth_a=0,
                depth_b=0,
                path_a=path_a,
                path_b=path_b
            )

        # Track visited expressions from each side
        # Maps: hashable_key -> (original_expr, depth, parent_key, label)
        # label is None for the start node, else a dict from _all_single_rewrites(labeled=True)
        visited_a: Dict[tuple, Tuple[ExprType, int, Optional[tuple], Optional[dict]]] = {
            key_a: (ca, 0, None, None)
        }
        visited_b: Dict[tuple, Tuple[ExprType, int, Optional[tuple], Optional[dict]]] = {
            key_b: (cb, 0, None, None)
        }

        # Per-side extension trackers; local to this call to avoid state leaks.
        _depth_extended_a = False
        _depth_extended_b = False
        max_depth_a = max_depth
        max_depth_b = max_depth

        # BFS frontiers carry CANONICAL states (expression, depth).
        frontier_a: deque = deque([(ca, 0)])
        frontier_b: deque = deque([(cb, 0)])

        def reconstruct_path(
            visited: Dict[tuple, Tuple[ExprType, int, Optional[tuple], Optional[dict]]],
            target_key: tuple
        ) -> List[RewriteStep]:
            """Reconstruct path from start to target as a list of RewriteSteps.

            Walks the parent chain collecting (expr, label) pairs, reverses
            them, then builds one RewriteStep per node.  The start node
            (label=None) gets a synthetic "initial" step whose before==after==expr.
            Each subsequent node gets a "rule" step whose .after equals the
            node expression, satisfying the step==expression endpoint invariant
            used by the backward-compat assertions in TestProveEqualWithTrace.
            """
            # Collect raw (expr, label) chain (target → start)
            raw: List[Tuple[ExprType, Optional[dict]]] = []
            current_key = target_key
            while current_key is not None:
                expr, _depth, parent_key, label = visited[current_key]
                raw.append((expr, label))
                current_key = parent_key
            raw.reverse()  # now start → target

            steps: List[RewriteStep] = []
            # Build one RewriteStep per node.
            # The "before" of each step is the expression at the previous node
            # (or the start expression for the first step).
            prev_expr = raw[0][0]
            for expr, label in raw:
                if label is None:
                    # Start node: synthetic initial step
                    step = RewriteStep(
                        rule_index=-1,
                        metadata=RuleMetadata(name=None),
                        before=expr,
                        after=expr,
                        kind="initial",
                    )
                else:
                    step = RewriteStep(
                        rule_index=-1,
                        metadata=RuleMetadata(name=label.get("rule_id")),
                        before=prev_expr,
                        after=expr,
                        rule_id=label.get("rule_id"),
                        direction=label.get("direction"),
                        bindings=label.get("bindings"),
                        # before/after above are WHOLE expressions (the BFS
                        # states), so the path MUST be [] -- the
                        # to_global_sequence contract splices `after` at
                        # `path`, and stamping the redex-local path here
                        # fabricated nonexistent intermediate states for any
                        # sub-root rewrite. The label's redex path is kept
                        # out of the step until steps carry redex-local
                        # before/after.
                        path=[],
                        kind="rule",
                    )
                steps.append(step)
                prev_expr = expr
            return steps

        # Alternate expanding from A and B
        while frontier_a or frontier_b:
            if self._cancel_requested:
                return None

            # Budget check: total work across both frontiers
            if max_expressions is not None and \
                    len(visited_a) + len(visited_b) >= max_expressions:
                return None

            # Expand from A
            if frontier_a:
                current, depth = frontier_a.popleft()
                if depth >= max_depth_a:
                    resolution = self._fire_max_depth(current, depth)
                    if (resolution is not None and resolution.allow_more
                            and not _depth_extended_a):
                        _depth_extended_a = True
                        max_depth_a = max_depth_a * 2 if max_depth_a > 0 else 1
                if depth < max_depth_a:
                    current_key = _expr_to_tuple(current)
                    rewrites = self._all_single_rewrites(
                        current, bidirectional_only, groups, rules=rules, labeled=True
                    )
                    for new_expr, label in rewrites:
                        cnew = self._canonicalize(new_expr)
                        new_key = _expr_to_tuple(cnew)
                        if new_key not in visited_a:
                            visited_a[new_key] = (cnew, depth + 1, current_key, label)
                            frontier_a.append((cnew, depth + 1))

                            # Check for intersection
                            if new_key in visited_b:
                                _, depth_b, _, _ = visited_b[new_key]
                                if trace:
                                    path_a = reconstruct_path(visited_a, new_key)
                                    path_b = reconstruct_path(visited_b, new_key)
                                else:
                                    path_a = None
                                    path_b = None

                                return EqualityProof(
                                    expr_a=expr_a,
                                    expr_b=expr_b,
                                    common=cnew,
                                    depth_a=depth + 1,
                                    depth_b=depth_b,
                                    path_a=path_a,
                                    path_b=path_b
                                )

            # Expand from B
            if frontier_b:
                current, depth = frontier_b.popleft()
                if depth >= max_depth_b:
                    resolution = self._fire_max_depth(current, depth)
                    if (resolution is not None and resolution.allow_more
                            and not _depth_extended_b):
                        _depth_extended_b = True
                        max_depth_b = max_depth_b * 2 if max_depth_b > 0 else 1
                if depth < max_depth_b:
                    current_key = _expr_to_tuple(current)
                    rewrites = self._all_single_rewrites(
                        current, bidirectional_only, groups, rules=rules, labeled=True
                    )
                    for new_expr, label in rewrites:
                        cnew = self._canonicalize(new_expr)
                        new_key = _expr_to_tuple(cnew)
                        if new_key not in visited_b:
                            visited_b[new_key] = (cnew, depth + 1, current_key, label)
                            frontier_b.append((cnew, depth + 1))

                            # Check for intersection
                            if new_key in visited_a:
                                _, depth_a_val, _, _ = visited_a[new_key]
                                if trace:
                                    path_a = reconstruct_path(visited_a, new_key)
                                    path_b = reconstruct_path(visited_b, new_key)
                                else:
                                    path_a = None
                                    path_b = None

                                return EqualityProof(
                                    expr_a=expr_a,
                                    expr_b=expr_b,
                                    common=cnew,
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

        When a theory is loaded (see ``with_theory``), the class is explored
        modulo the theory and the returned expression is the min-cost CANONICAL
        representative; improvement is measured against the canonical form of
        the input. With no theory, the behavior is unchanged.

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

        # Track best found. Seed from the CANONICAL form so the baseline lives
        # in the same normalization state as the reps equivalents() yields;
        # otherwise normalization alone would read as a spurious improvement.
        # ``cexpr`` is also the result's ``original`` and the derivation's
        # ``initial`` so the whole OptimizationResult is canonical-consistent
        # (``original_cost == cost_fn(original)``). With no theory _canonicalize
        # is the identity, so this is byte-for-byte unchanged.
        cexpr = self._canonicalize(expr)
        best_expr = cexpr
        best_cost = cost_fn(best_expr)
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

        derivation = None
        if _expr_to_tuple(best_expr) != _expr_to_tuple(cexpr):
            proof = self.prove_equal(
                expr, best_expr, trace=True,
                max_depth=max_depth,
                max_expressions=max_count,
                include_unidirectional=include_unidirectional,
                groups=groups,
                rules=rules,
            )
            if proof is not None and proof.path_a is not None and proof.path_b is not None:
                trace = RewriteTrace()
                trace.initial = cexpr
                trace.final = best_expr
                for step in proof.path_a[1:]:        # skip synthetic initial
                    trace(step)
                for step in reversed(proof.path_b[1:]):  # common -> best
                    # Reverse the ORDER and INVERT each step: path_b is a
                    # forward path (best -> common), so each step must be
                    # turned around (swap before/after, flip direction) to
                    # read common -> best. Reordering alone leaves the steps
                    # oriented best -> common, breaking the global-sequence
                    # chain (the Phase 1 minimize-derivation limitation).
                    trace(step.inverse())
                derivation = trace

        return OptimizationResult(
            expr=best_expr,
            cost=best_cost,
            original=cexpr,
            original_cost=original_cost,
            expressions_checked=count,
            derivation=derivation,
        )

    # NOTE: goal-directed search is NOT a method on the engine. The engine is
    # a pure term-rewriting engine; goal-directed best-first SEARCH is an
    # optional, non-core layer. Use it explicitly:
    #   from rerum.solve import solve
    #   result = solve(engine, expr, goal_predicate, ...)

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
        self._step_count = 0
        self._cancel_requested = False
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
        self._step_count = 0
        self._cancel_requested = False
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

import types as _types
RuleEngine._HOOK_EVENTS = _types.MappingProxyType(RuleEngine._HOOK_EVENTS)
del _types


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
