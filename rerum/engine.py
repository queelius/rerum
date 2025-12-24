"""
Rule Engine and DSL Loader for RERUM

RERUM - Rewriting Expressions via Rules Using Morphisms

This module provides facilities for loading rewriting rules from external files,
supporting both a custom DSL format and JSON.

DSL Format (.rules files):
    # Comment
    @rule-name: (pattern) => (skeleton)
    @rule-name "Description text": (pattern) => (skeleton)

    Examples:
    @add-zero: (+ ?x 0) => :x
    @dd-sum "Derivative of sum": (dd (+ ?f ?g) ?v:var) => (+ (dd :f :v) (dd :g :v))

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
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union

from .rewriter import (
    rewriter, match as _match_internal, instantiate, ExprType,
    FoldFuncsType, ARITHMETIC_PRELUDE, Bindings, NoMatch, wrap_bindings,
)


# ============================================================
# Expression Builder
# ============================================================

class _ExprBuilder:
    """
    Expression builder for RERUM.

    Provides convenient ways to construct expressions without
    privileging any particular operators.

    Examples:
        from rerum import E

        # Parse s-expression string
        expr = E("(+ x (* 2 y))")

        # Build programmatically with E.op()
        expr = E.op("+", "x", E.op("*", 2, "y"))

        # Create variables
        x, y = E.vars("x", "y")
        expr = E.op("+", x, E.op("*", 2, y))

        # Any operator works - no privileged operations
        E.op("dd", E.op("^", "x", 2), "x")
        E.op("my-custom-op", "a", "b", "c")
    """

    def __call__(self, s: str) -> ExprType:
        """
        Parse an s-expression string.

        Examples:
            E("(+ x 1)") -> ["+", "x", 1]
            E("(dd (^ x 2) x)") -> ["dd", ["^", "x", 2], "x"]
        """
        return parse_sexpr(s)

    def op(self, name: str, *args) -> List:
        """
        Build a compound expression with the given operator and arguments.

        This is the universal constructor for compound expressions.
        No operators are privileged - use this for any operation.

        Examples:
            E.op("+", "x", 1) -> ["+", "x", 1]
            E.op("*", 2, "y") -> ["*", 2, "y"]
            E.op("dd", E.op("^", "x", 2), "x") -> ["dd", ["^", "x", 2], "x"]
            E.op("my-func", "a", "b") -> ["my-func", "a", "b"]
        """
        return [name] + list(args)

    def var(self, name: str) -> str:
        """
        Create a variable.

        Variables are just strings. This method exists for clarity
        and to document intent.

        Example:
            E.var("x") -> "x"
        """
        return name

    def vars(self, *names: str) -> Tuple[str, ...]:
        """
        Create multiple variables for unpacking.

        Example:
            x, y, z = E.vars("x", "y", "z")
            expr = E.op("+", x, E.op("*", y, z))
        """
        return names

    def const(self, value: Union[int, float]) -> Union[int, float]:
        """
        Create a constant.

        Constants are just numbers. This method exists for clarity
        and to document intent in code that constructs expressions.

        Example:
            E.const(5) -> 5
            E.const(3.14) -> 3.14
        """
        return value

    def __repr__(self) -> str:
        return "E (expression builder)"


# Singleton instance
E = _ExprBuilder()


def parse_sexpr(s: str) -> ExprType:
    """
    Parse an S-expression string into a nested list.

    Examples:
        "(+ x 1)" -> ["+", "x", 1]
        "(dd (^ x 2) x)" -> ["dd", ["^", "x", 2], "x"]
    """
    s = s.strip()
    if not s:
        return None

    if s.startswith('('):
        # Parse list
        depth = 0
        parts = []
        current = ''
        i = 1  # Skip opening paren

        while i < len(s):
            c = s[i]
            if c == '(':
                depth += 1
                current += c
            elif c == ')':
                if depth == 0:
                    if current.strip():
                        parts.append(parse_sexpr(current.strip()))
                    break
                depth -= 1
                current += c
            elif c in ' \t\n' and depth == 0:
                if current.strip():
                    parts.append(parse_sexpr(current.strip()))
                current = ''
            else:
                current += c
            i += 1

        return parts

    # Parse atom
    # Try number first
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            pass

    # Pattern variable syntax conversion
    # Support both old syntax (?c name) and new typed syntax (?name:const)
    if s.startswith('?'):
        rest = s[1:]

        # Check for rest pattern (ends with ...)
        is_rest = rest.endswith('...')
        if is_rest:
            rest = rest[:-3]  # Remove trailing ...

        # New typed syntax: ?name:type or ?name:free(var)
        if ':' in rest:
            name_part, type_part = rest.split(':', 1)
            name = name_part.strip() or 'x'

            if is_rest:
                # Rest pattern with type constraint: ?x:const... or ?x:var...
                if type_part == 'const':
                    return ["?...", name, "const"]
                elif type_part == 'var':
                    return ["?...", name, "var"]
                else:
                    # Unknown type constraint, unconstrained rest
                    return ["?...", name]
            else:
                # Single-element typed pattern
                if type_part == 'const':
                    return ["?c", name]
                elif type_part == 'var':
                    return ["?v", name]
                elif type_part == 'expr':
                    return ["?", name]
                elif type_part.startswith('free(') and type_part.endswith(')'):
                    # ?name:free(var)
                    var = type_part[5:-1].strip()
                    return ["?free", name, var]
                else:
                    # Unknown type, treat as plain pattern
                    return ["?", name]

        else:
            # Plain pattern variable: ?x or ?x...
            name = rest.strip() or 'x'
            if is_rest:
                return ["?...", name]
            return ["?", name]

    elif s.startswith(':'):
        rest = s[1:].strip()
        # Check for splice (ends with ...)
        if rest.endswith('...'):
            name = rest[:-3].strip()
            return [":...", name]
        else:
            return [":", rest]

    # Plain symbol
    return s


def format_sexpr(expr: ExprType, dsl_syntax: bool = True) -> str:
    """
    Format an expression as an S-expression string.

    Args:
        expr: Expression to format
        dsl_syntax: If True, use DSL syntax for patterns (?x, :x).
                    If False, use raw list syntax ((? x), (: x)).

    Examples:
        ["+", "x", 1] -> "(+ x 1)"
        ["?", "x"] -> "?x" (with dsl_syntax=True)
        [":", "x"] -> ":x" (with dsl_syntax=True)
        ["?c", "n"] -> "?n:const" (with dsl_syntax=True)
        ["?...", "xs"] -> "?xs..." (with dsl_syntax=True)
        [":...", "xs"] -> ":xs..." (with dsl_syntax=True)
    """
    if isinstance(expr, list):
        if not expr:
            return "()"

        # Handle pattern/skeleton DSL syntax
        if dsl_syntax and len(expr) == 2:
            op = expr[0]
            if op == "?":
                return f"?{expr[1]}"
            elif op == ":":
                return f":{expr[1]}"
            elif op == "?c":
                return f"?{expr[1]}:const"
            elif op == "?v":
                return f"?{expr[1]}:var"
            elif op == "?...":
                return f"?{expr[1]}..."
            elif op == ":...":
                return f":{expr[1]}..."

        if dsl_syntax and len(expr) == 3:
            op = expr[0]
            if op == "?free":
                return f"?{expr[1]}:free({expr[2]})"
            elif op == "?...":
                # Rest pattern with type constraint: ["?...", "name", "const"]
                return f"?{expr[1]}:{expr[2]}..."

        parts = [format_sexpr(e, dsl_syntax) for e in expr]
        return "(" + " ".join(parts) + ")"
    elif isinstance(expr, (int, float)):
        return str(expr)
    else:
        return str(expr)


class RuleMetadata:
    """Metadata for a rule including name, description, priority, and condition."""

    def __init__(self, name: Optional[str] = None, description: Optional[str] = None,
                 tags: Optional[List[str]] = None, condition: Optional[ExprType] = None,
                 priority: int = 0):
        self.name = name
        self.description = description
        self.tags = tags or []
        self.condition = condition  # Optional guard condition
        self.priority = priority  # Higher priority fires first (default: 0)

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


def parse_rule_line(line: str) -> Optional[Tuple[RuleMetadata, ExprType, ExprType]]:
    """
    Parse a single rule line.

    Formats:
        @name: pattern => skeleton
        @name[priority]: pattern => skeleton
        @name "description": pattern => skeleton
        @name[priority] "description": pattern => skeleton
        @name: pattern => skeleton when condition
        pattern => skeleton

    Returns: (metadata, pattern, skeleton) or None if not a rule
    """
    line = line.strip()

    # Skip empty lines and comments
    if not line or line.startswith('#'):
        return None

    # Extract name, optional priority, and optional description if present
    metadata = RuleMetadata()
    if line.startswith('@'):
        # Try format: @name[priority] "description": ...
        match_obj = re.match(r'@([\w-]+)\[(\d+)\]\s+"([^"]+)":\s*(.+)', line)
        if match_obj:
            metadata.name = match_obj.group(1)
            metadata.priority = int(match_obj.group(2))
            metadata.description = match_obj.group(3)
            line = match_obj.group(4)
        else:
            # Try format: @name[priority]: ...
            match_obj = re.match(r'@([\w-]+)\[(\d+)\]:\s*(.+)', line)
            if match_obj:
                metadata.name = match_obj.group(1)
                metadata.priority = int(match_obj.group(2))
                line = match_obj.group(3)
            else:
                # Try format: @name "description": ...
                match_obj = re.match(r'@([\w-]+)\s+"([^"]+)":\s*(.+)', line)
                if match_obj:
                    metadata.name = match_obj.group(1)
                    metadata.description = match_obj.group(2)
                    line = match_obj.group(3)
                else:
                    # Try format: @name: ...
                    match_obj = re.match(r'@([\w-]+):\s*(.+)', line)
                    if match_obj:
                        metadata.name = match_obj.group(1)
                        line = match_obj.group(2)

    # Must have =>
    if '=>' not in line:
        return None

    # Split on =>
    parts = line.split('=>', 1)
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
        metadata.condition = condition

    pattern = parse_sexpr(pattern_str)
    skeleton = parse_sexpr(skeleton_str)

    if pattern is None or skeleton is None:
        return None

    return (metadata, pattern, skeleton)


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

        result = parse_rule_line(line)
        if result:
            metadata, pattern, skeleton = result
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
            metadata = RuleMetadata(
                name=rule.get('name'),
                description=rule.get('description'),
                tags=rule.get('tags'),
                priority=rule.get('priority', 0),
                condition=rule.get('condition')
            )
            pattern = rule['pattern']
            skeleton = rule['skeleton']
        else:
            metadata = RuleMetadata()
            pattern, skeleton = rule[0], rule[1]
        rules.append((metadata, [pattern, skeleton]))

    return rules


class RewriteStep:
    """A single step in a rewriting trace."""

    def __init__(self, rule_index: int, metadata: RuleMetadata,
                 before: ExprType, after: ExprType):
        self.rule_index = rule_index
        self.metadata = metadata
        self.before = before
        self.after = after

    def __repr__(self) -> str:
        name = self.metadata.name or f"rule[{self.rule_index}]"
        return f"{name}: {format_sexpr(self.before)} → {format_sexpr(self.after)}"

    def to_dict(self) -> Dict:
        """Convert step to dictionary for serialization."""
        return {
            "rule_index": self.rule_index,
            "rule_name": self.metadata.name,
            "description": self.metadata.description,
            "before": self.before,
            "after": self.after,
        }


class RewriteTrace:
    """
    A trace of all rewriting steps applied.

    Provides multiple formatting options:
        - Default repr: verbose multi-line format
        - format("compact"): single line showing rule chain
        - format("rules"): just the rule names applied
        - format("verbose"): full details with before/after
        - to_dict(): JSON-serializable dictionary
    """

    def __init__(self):
        self.steps: List[RewriteStep] = []
        self.initial: ExprType = None
        self.final: ExprType = None

    def add_step(self, step: RewriteStep):
        self.steps.append(step)

    def format(self, style: str = "verbose") -> str:
        """
        Format the trace in different styles.

        Args:
            style: One of "verbose", "compact", "rules", "chain"
                - "verbose": Full multi-line format with before/after (default)
                - "compact": Single line summary
                - "rules": Just the list of rule names applied
                - "chain": Show expression transformations as a chain

        Returns:
            Formatted string representation of the trace.
        """
        if style == "compact":
            rules = [s.metadata.name or f"rule[{s.rule_index}]" for s in self.steps]
            return f"{format_sexpr(self.initial)} --[{', '.join(rules)}]--> {format_sexpr(self.final)}"

        elif style == "rules":
            rules = [s.metadata.name or f"rule[{s.rule_index}]" for s in self.steps]
            return " -> ".join(rules) if rules else "(no rules applied)"

        elif style == "chain":
            if not self.steps:
                return format_sexpr(self.initial)
            parts = [format_sexpr(self.initial)]
            for step in self.steps:
                name = step.metadata.name or f"rule[{step.rule_index}]"
                parts.append(f"  --({name})-->")
                parts.append(format_sexpr(step.after))
            return "\n".join(parts)

        else:  # verbose (default)
            return repr(self)

    def __repr__(self) -> str:
        lines = [f"Initial: {format_sexpr(self.initial)}"]
        for i, step in enumerate(self.steps, 1):
            if step.metadata.description:
                lines.append(f"  {i}. {step.metadata} ({step.metadata.description})")
            else:
                lines.append(f"  {i}. {step}")
        lines.append(f"Final: {format_sexpr(self.final)}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        """Iterate over rewrite steps."""
        return iter(self.steps)

    def __bool__(self) -> bool:
        """True if any rewriting was done."""
        return len(self.steps) > 0

    def to_dict(self) -> Dict:
        """Convert trace to dictionary for JSON serialization."""
        return {
            "initial": self.initial,
            "final": self.final,
            "steps": [step.to_dict() for step in self.steps],
            "step_count": len(self.steps),
        }

    def rule_counts(self) -> Dict[str, int]:
        """Count how many times each rule was applied."""
        counts: Dict[str, int] = {}
        for step in self.steps:
            name = step.metadata.name or f"rule[{step.rule_index}]"
            counts[name] = counts.get(name, 0) + 1
        return counts

    def rules_applied(self) -> List[str]:
        """Get list of rule names in order of application."""
        return [s.metadata.name or f"rule[{s.rule_index}]" for s in self.steps]

    def summary(self) -> str:
        """Get a brief summary of the rewriting."""
        if not self.steps:
            return "No rewriting performed"
        counts = self.rule_counts()
        most_used = max(counts.items(), key=lambda x: x[1])
        return (f"{len(self.steps)} steps using {len(counts)} unique rules. "
                f"Most used: {most_used[0]} ({most_used[1]}x)")


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

    def match(self, pattern: Union[str, ExprType], expr: ExprType) -> Union[Bindings, NoMatch]:
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

        # Use internal match function
        result = _match_internal(pattern, expr, [])
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
            bindings = _match_internal(pattern, expr, [])
            if bindings != "failed":
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
            raw_bindings = _match_internal(pattern, expr, [])
            if raw_bindings != "failed":
                # Check condition if requested
                if check_conditions and not self._check_condition(metadata.condition, raw_bindings):
                    continue
                matching.append((metadata, Bindings(raw_bindings)))
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

    def _simplify_exhaustive(self, expr: ExprType, max_steps: int, groups: Optional[List[str]] = None) -> ExprType:
        """Exhaustive strategy with condition and group support."""
        current = expr
        for _ in range(max_steps):
            changed = False

            # Try rules at top level
            for rule_idx, rule in enumerate(self._rules):
                metadata = self._metadata[rule_idx]
                # Check group filter
                if not self._is_rule_active(metadata, groups):
                    continue
                pattern, skeleton = rule
                bindings = _match_internal(pattern, current, [])
                if bindings != "failed":
                    if not self._check_condition(metadata.condition, bindings):
                        continue
                    new_expr = instantiate(skeleton, bindings, self._fold_funcs)
                    if new_expr != current:
                        current = new_expr
                        changed = True
                        break

            if not changed:
                # Recursively simplify subexpressions
                if isinstance(current, list) and len(current) > 0:
                    new_children = []
                    subexpr_changed = False
                    for child in current:
                        new_child = self._simplify_exhaustive(child, max_steps // 10 or 1, groups=groups)
                        new_children.append(new_child)
                        if new_child != child:
                            subexpr_changed = True
                    if subexpr_changed:
                        current = new_children
                        continue
                break

        return current

    def _simplify_bottomup(self, expr: ExprType, max_steps: int, groups: Optional[List[str]] = None) -> ExprType:
        """Bottom-up strategy: simplify children first, then parent."""
        for _ in range(max_steps):
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
            bindings = _match_internal(pattern, current, [])
            if bindings != "failed":
                # Check condition if present
                if not self._check_condition(metadata.condition, bindings):
                    continue
                result = instantiate(skeleton, bindings, self._fold_funcs)
                if result != current:
                    return result

        return current

    def _simplify_topdown(self, expr: ExprType, max_steps: int, groups: Optional[List[str]] = None) -> ExprType:
        """Top-down strategy: try parent first, then children."""
        for _ in range(max_steps):
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
            bindings = _match_internal(pattern, current, [])
            if bindings != "failed":
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
        """Internal method for traced simplification."""
        trace_obj = RewriteTrace()
        trace_obj.initial = expr

        current = expr
        for _ in range(max_steps):
            changed = False
            for rule_idx, rule in enumerate(self._rules):
                metadata = self._metadata[rule_idx]
                # Check group filter
                if not self._is_rule_active(metadata, groups):
                    continue
                pattern, skeleton = rule
                bindings = _match_internal(pattern, current, [])
                if bindings != "failed":
                    # Check condition if present
                    if not self._check_condition(metadata.condition, bindings):
                        continue
                    new_expr = instantiate(skeleton, bindings, self._fold_funcs)
                    if new_expr != current:
                        step = RewriteStep(
                            rule_index=rule_idx,
                            metadata=metadata,
                            before=current,
                            after=new_expr
                        )
                        trace_obj.add_step(step)
                        current = new_expr
                        changed = True
                        break  # Restart from first rule

            if not changed:
                # Try to simplify subexpressions
                if isinstance(current, list) and len(current) > 0:
                    new_list = [current[0]]
                    subexpr_changed = False
                    for sub in current[1:]:
                        sub_result, sub_trace = self._simplify_with_trace(sub, max_steps // 10, groups=groups)
                        new_list.append(sub_result)
                        if sub_trace.steps:
                            trace_obj.steps.extend(sub_trace.steps)
                            subexpr_changed = True
                    if subexpr_changed:
                        current = new_list
                        continue
                break

        # Apply constant folding if fold_funcs provided
        if self._fold_funcs:
            current = self._fold_constants(current)

        trace_obj.final = current
        return current, trace_obj

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

        Args:
            name: Optional name to include as a comment header

        Returns:
            DSL-formatted string with all rules, organized by groups.
        """
        lines = []
        if name:
            lines.append(f"# {name}")
            lines.append("")

        # Group rules by their first tag (group)
        current_group = None
        for rule, meta in zip(self._rules, self._metadata):
            # Check if we need a new group header
            rule_group = meta.tags[0] if meta.tags else None
            if rule_group != current_group:
                if rule_group:
                    if lines and lines[-1] != "":
                        lines.append("")
                    lines.append(f"[{rule_group}]")
                current_group = rule_group

            # Format the rule
            pattern, skeleton = rule
            pattern_str = format_sexpr(pattern)
            skeleton_str = format_sexpr(skeleton)

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
        rules_list = []
        for rule, meta in zip(self._rules, self._metadata):
            pattern, skeleton = rule
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

        result = {"rules": rules_list}
        if name:
            result["name"] = name
        if description:
            result["description"] = description

        return json.dumps(result, indent=indent)

    def to_dict(self) -> Dict:
        """
        Export rules to a dictionary.

        Returns:
            Dictionary compatible with JSON serialization.
        """
        rules_list = []
        for rule, meta in zip(self._rules, self._metadata):
            pattern, skeleton = rule
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

        return {"rules": rules_list}

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
