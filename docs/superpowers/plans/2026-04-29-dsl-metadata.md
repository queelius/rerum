# DSL/Metadata Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement v0.7 metadata layer per `docs/superpowers/specs/2026-04-29-dsl-metadata-design.md`. Adds four `RuleMetadata` fields (`category`, `reasoning`, `examples`, `fwd_label`/`rev_label`), one DSL annotation `{category=X}`, JSON schema extension, examples validation at load time, and sidecar JSON loading.

**Architecture:** All changes land in `rerum/engine.py` plus three new test files. The DSL parser gets an annotation pre-extraction pass; the JSON loader/serializer gain four optional fields; a new `_validate_example` helper runs at every loader. `ExampleValidationError` is a new exception. `add_rule` and a new `load_metadata_json` round out the public API.

**Tech Stack:** Python 3.9+, `dataclasses`, `pytest`. No new external dependencies.

---

## File Structure

**Create:**
- `rerum/tests/test_metadata.py`: RuleMetadata new fields, JSON loader/serializer for new fields, sidecar merge
- `rerum/tests/test_dsl_metadata.py`: DSL `{category=X}` annotation parsing (single-line and multi-line)
- `rerum/tests/test_examples_validation.py`: examples validation at load and on demand

**Modify:**
- `rerum/engine.py`: `RuleMetadata.__init__` (4 new fields), `parse_rule_line` (annotation parsing), `_build_bidirectional_rules` (label propagation), `load_rules_from_dsl` (multi-line annotation pre-processing), `load_rules_from_json` (4 new fields), `to_dsl` / `to_json` / `to_dict` (4 new fields), `add_rule` (4 new kwargs), new helpers `_validate_example` and `_validate_rule_examples`, new public methods `validate_examples` and `load_metadata_json`, new exception `ExampleValidationError`
- `rerum/__init__.py`: re-export `ExampleValidationError`
- `CHANGELOG.md`: 0.7.0 release entry
- `pyproject.toml`: version bump to 0.7.0
- `CLAUDE.md`: brief mention of the metadata layer

---

## Task 1: Extend `RuleMetadata` with four new fields

**Files:**
- Modify: `rerum/engine.py` (`RuleMetadata.__init__`, around line 326)
- Test: `rerum/tests/test_metadata.py` (new file)

- [ ] **Step 1: Write the failing test**

Create `rerum/tests/test_metadata.py`:

```python
"""Tests for RuleMetadata new fields (v0.7) and metadata JSON roundtrip."""

import pytest
from rerum.engine import RuleMetadata


class TestRuleMetadataFields:
    def test_category_defaults_to_none(self):
        m = RuleMetadata()
        assert m.category is None

    def test_reasoning_defaults_to_none(self):
        m = RuleMetadata()
        assert m.reasoning is None

    def test_examples_defaults_to_none(self):
        m = RuleMetadata()
        assert m.examples is None

    def test_fwd_label_defaults_to_none(self):
        m = RuleMetadata()
        assert m.fwd_label is None

    def test_rev_label_defaults_to_none(self):
        m = RuleMetadata()
        assert m.rev_label is None

    def test_category_set(self):
        m = RuleMetadata(category="identity")
        assert m.category == "identity"

    def test_reasoning_set(self):
        m = RuleMetadata(reasoning="Zero is the additive identity.")
        assert m.reasoning == "Zero is the additive identity."

    def test_examples_set(self):
        m = RuleMetadata(examples=[{"in": "(+ x 0)", "out": "x"}])
        assert m.examples == [{"in": "(+ x 0)", "out": "x"}]

    def test_fwd_label_set_on_bidirectional(self):
        m = RuleMetadata(bidirectional=True, fwd_label="regroup-right")
        assert m.fwd_label == "regroup-right"

    def test_rev_label_set_on_bidirectional(self):
        m = RuleMetadata(bidirectional=True, rev_label="regroup-left")
        assert m.rev_label == "regroup-left"

    def test_existing_fields_still_work(self):
        m = RuleMetadata(name="r1", description="d", priority=5,
                         tags=["g1"], bidirectional=True, direction="fwd")
        assert m.name == "r1"
        assert m.description == "d"
        assert m.priority == 5
        assert m.tags == ["g1"]
        assert m.bidirectional is True
        assert m.direction == "fwd"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_metadata.py::TestRuleMetadataFields -v 2>&1 | tail -20
```

Expected: failures on `category`, `reasoning`, `examples`, `fwd_label`, `rev_label` accessors (AttributeError).

- [ ] **Step 3: Add the four new fields to `RuleMetadata`**

Find `RuleMetadata.__init__` in `rerum/engine.py` (around line 326). Replace with:

```python
class RuleMetadata:
    """Metadata for a rule including name, description, priority, condition, and v0.7 fields."""

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
        self.condition = condition
        self.priority = priority
        self.bidirectional = bidirectional
        self.direction = direction
        self.extra = extra or {}
        self.category = category
        self.reasoning = reasoning
        self.examples = examples
        self.fwd_label = fwd_label
        self.rev_label = rev_label
```

- [ ] **Step 4: Run tests**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_metadata.py::TestRuleMetadataFields -v 2>&1 | tail -15
```

Expected: 11 passed.

- [ ] **Step 5: Run full suite to confirm no regression**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest 2>&1 | tail -3
```

Expected: count goes up by 11; all pass.

- [ ] **Step 6: Commit**

```bash
cd /home/spinoza/github/repos/rerum && git add rerum/engine.py rerum/tests/test_metadata.py
git commit -m "$(cat <<'EOF'
feat(metadata): add RuleMetadata fields category/reasoning/examples/labels

Four new optional fields default to None; existing constructor calls and
field accesses remain valid. Sets the foundation for v0.7 metadata layer
(DSL annotation, JSON serialization, examples validation in subsequent
tasks).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Parse `{category=X}` annotation in `parse_rule_line` (single-line)

**Files:**
- Modify: `rerum/engine.py` (`parse_rule_line`, around line 417)
- Test: `rerum/tests/test_dsl_metadata.py` (new)

- [ ] **Step 1: Write the failing test**

Create `rerum/tests/test_dsl_metadata.py`:

```python
"""Tests for DSL `{category=X}` annotation parsing."""

import pytest
from rerum.engine import parse_rule_line, load_rules_from_dsl


class TestSingleLineAnnotation:
    def test_annotation_with_name_priority_description(self):
        results = parse_rule_line(
            '@add-zero[100] "x + 0 = x" {category=identity}: (+ ?x 0) => :x'
        )
        assert len(results) == 1
        meta, pat, skel = results[0]
        assert meta.name == "add-zero"
        assert meta.priority == 100
        assert meta.description == "x + 0 = x"
        assert meta.category == "identity"

    def test_annotation_with_name_only(self):
        results = parse_rule_line(
            '@distrib {category=distributivity}: (* ?x (+ ?y ?z)) => (+ (* :x :y) (* :x :z))'
        )
        assert len(results) == 1
        meta, _, _ = results[0]
        assert meta.name == "distrib"
        assert meta.category == "distributivity"
        assert meta.description is None

    def test_annotation_with_name_and_description(self):
        results = parse_rule_line(
            '@r1 "desc" {category=cat}: (a ?x) => :x'
        )
        meta, _, _ = results[0]
        assert meta.name == "r1"
        assert meta.description == "desc"
        assert meta.category == "cat"

    def test_annotation_with_name_and_priority(self):
        results = parse_rule_line(
            '@r1[50] {category=cat}: (a ?x) => :x'
        )
        meta, _, _ = results[0]
        assert meta.name == "r1"
        assert meta.priority == 50
        assert meta.category == "cat"

    def test_anonymous_rule_with_annotation(self):
        results = parse_rule_line(
            '{category=fold-constant}: (* ?a:const ?b:const) => (! * :a :b)'
        )
        assert len(results) == 1
        meta, _, _ = results[0]
        assert meta.name is None
        assert meta.category == "fold-constant"

    def test_bidirectional_with_annotation(self):
        results = parse_rule_line(
            '@commute {category=commutativity}: (+ ?x ?y) <=> (+ :y :x)'
        )
        assert len(results) == 2  # fwd and rev
        for meta, _, _ in results:
            assert meta.category == "commutativity"

    def test_quoted_value(self):
        results = parse_rule_line(
            '@r1 {category="multi word"}: (a ?x) => :x'
        )
        meta, _, _ = results[0]
        assert meta.category == "multi word"

    def test_whitespace_in_annotation(self):
        results = parse_rule_line(
            '@r1 { category = identity }: (a ?x) => :x'
        )
        meta, _, _ = results[0]
        assert meta.category == "identity"

    def test_no_annotation_still_parses(self):
        results = parse_rule_line('@r1: (a ?x) => :x')
        meta, _, _ = results[0]
        assert meta.name == "r1"
        assert meta.category is None

    def test_unknown_annotation_key_raises(self):
        with pytest.raises(ValueError, match="unknown annotation key"):
            parse_rule_line('@r1 {ref="paper"}: (a ?x) => :x')

    def test_malformed_annotation_raises(self):
        # Missing closing brace
        with pytest.raises(ValueError, match="malformed annotation"):
            parse_rule_line('@r1 {category=identity: (a ?x) => :x')
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_dsl_metadata.py::TestSingleLineAnnotation -v 2>&1 | tail -25
```

Expected: failures (annotation parsing not yet implemented).

- [ ] **Step 3: Implement annotation parsing**

In `rerum/engine.py`, add a helper just above `parse_rule_line`:

```python
import re as _re

_ANNOTATION_KEYS = frozenset({"category"})


def _extract_annotation(line: str) -> Tuple[str, Dict[str, str]]:
    """Extract a `{key=value, key=value}` annotation block from `line`.

    The annotation must appear after the optional name/priority/description
    and before the `:`. If no annotation is present, returns (line, {}).

    Whitespace is permitted inside the braces. Values may be quoted strings
    (for multi-word values) or single tokens. Unknown keys raise
    ValueError. Malformed annotations (e.g., missing closing brace) raise
    ValueError.

    Returns the line with the annotation block removed and a dict of
    parsed key-value pairs.
    """
    # Find a `{` that is not inside an s-expression. Strategy: scan from
    # the start, ignoring `{` inside `(...)` (s-expr depth tracking).
    depth = 0
    brace_start = -1
    for i, c in enumerate(line):
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        elif c == '{' and depth == 0:
            brace_start = i
            break
    if brace_start < 0:
        return line, {}

    # Find the matching closing brace at depth 0.
    brace_end = -1
    for j in range(brace_start + 1, len(line)):
        if line[j] == '}':
            brace_end = j
            break
    if brace_end < 0:
        raise ValueError(f"malformed annotation: missing '}}' in {line!r}")

    inner = line[brace_start + 1:brace_end].strip()
    remaining = (line[:brace_start] + line[brace_end + 1:]).strip()

    # Parse `key=value, key=value`
    annotations: Dict[str, str] = {}
    for pair in _split_annotation_pairs(inner):
        if '=' not in pair:
            raise ValueError(f"malformed annotation pair {pair!r} in {line!r}")
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


def _split_annotation_pairs(inner: str) -> List[str]:
    """Split `key=value, key=value` accounting for quoted values."""
    pairs = []
    current = []
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
    return pairs
```

Now modify `parse_rule_line` to extract the annotation before the existing parsing logic. Find the function (around line 417). At the very start of the function body (after the `line = line.strip()` and comment-skip checks), insert:

```python
    # Extract optional `{key=val}` annotation (v0.7).
    line, annotations = _extract_annotation(line)
```

At every place where `parse_rule_line` constructs a `RuleMetadata` (the `if is_bidirectional:` branch via `_build_bidirectional_rules` and the unidirectional branch), pass `category=annotations.get("category")`.

Specifically:
- The `_build_bidirectional_rules` call already takes most metadata kwargs. Add `category=annotations.get("category")` to its call site inside `parse_rule_line`.
- The unidirectional branch's `RuleMetadata(...)` constructor call needs the same kwarg.

Look at the call site inside `parse_rule_line`. Currently:

```python
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
```

Update to:

```python
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
        metadata = RuleMetadata(
            name=base_name,
            description=description,
            priority=priority,
            condition=condition,
            category=annotations.get("category"),
        )
        return [(metadata, pattern, skeleton)]
```

Then update `_build_bidirectional_rules` (around line 316) to accept `category` and propagate it:

```python
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
    """..."""
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
    _validate_pattern_structure(rev_pattern, where="reverse pattern")
    return [
        (fwd_metadata, pattern, skeleton),
        (rev_metadata, rev_pattern, rev_skeleton),
    ]
```

- [ ] **Step 4: Run tests**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_dsl_metadata.py::TestSingleLineAnnotation -v 2>&1 | tail -15
```

Expected: 11 passed.

- [ ] **Step 5: Run full suite**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest 2>&1 | tail -3
```

Expected: all pass; count up by 11.

- [ ] **Step 6: Commit**

```bash
cd /home/spinoza/github/repos/rerum && git add rerum/engine.py rerum/tests/test_dsl_metadata.py
git commit -m "$(cat <<'EOF'
feat(dsl): parse {category=X} annotation in parse_rule_line

Adds _extract_annotation pre-processing step. Annotation block sits
between the optional description and the colon. v1 has only `category`
as a known key; unknown keys raise ValueError. _build_bidirectional_rules
propagates category to both -fwd and -rev metadata.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Multi-line annotation block

**Files:**
- Modify: `rerum/engine.py` (`load_rules_from_dsl`, around line 596)
- Test: `rerum/tests/test_dsl_metadata.py`

- [ ] **Step 1: Write the failing test**

Append to `rerum/tests/test_dsl_metadata.py`:

```python
class TestMultiLineAnnotation:
    def test_multi_line_annotation(self):
        text = """
@distrib {
  category=distributivity
}: (* ?x (+ ?y ?z)) => (+ (* :x :y) (* :x :z))
"""
        rules = load_rules_from_dsl(text)
        assert len(rules) == 1
        meta, _ = rules[0]
        assert meta.name == "distrib"
        assert meta.category == "distributivity"

    def test_multi_line_with_priority_and_description(self):
        text = """
@assoc[50] "Associativity" {
  category=associativity
}: (+ (+ ?x ?y) ?z) <=> (+ :x (+ :y :z))
"""
        rules = load_rules_from_dsl(text)
        assert len(rules) == 2  # fwd and rev
        for meta, _ in rules:
            assert meta.priority == 50
            assert meta.category == "associativity"

    def test_multi_line_compact_form_still_works(self):
        text = """
@r1 {category=cat}: (a ?x) => :x
"""
        rules = load_rules_from_dsl(text)
        meta, _ = rules[0]
        assert meta.category == "cat"
```

- [ ] **Step 2: Verify failure**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_dsl_metadata.py::TestMultiLineAnnotation -v 2>&1 | tail -15
```

Expected: failures (the line-oriented loader splits the multi-line annotation across lines).

- [ ] **Step 3: Pre-process multi-line blocks in `load_rules_from_dsl`**

Find `load_rules_from_dsl` (around line 596). The function iterates lines; each line is passed to `parse_rule_line`. To support multi-line annotation, fold open `{` blocks onto their start line before calling `parse_rule_line`.

Add a helper at module level (near `_extract_annotation`):

```python
def _join_multi_line_annotations(lines: List[str]) -> List[str]:
    """Join lines so multi-line `{ ... }` annotation blocks become
    single-line entries.

    A line that contains `{` (at s-expr depth 0) without a matching `}` on
    the same line is joined with subsequent lines until the matching `}`
    is found.
    """
    joined: List[str] = []
    buffer: List[str] = []
    open_depth = 0  # depth of unbalanced `{`

    for line in lines:
        # Count depth changes from this line, ignoring `{` inside parens.
        line_open = _count_unbalanced_braces(line)
        if buffer:
            buffer.append(line)
            open_depth += line_open
            if open_depth <= 0:
                # Close: emit joined line, reset buffer.
                joined.append(' '.join(buffer))
                buffer = []
                open_depth = 0
            continue

        if line_open > 0:
            # Opening brace not closed on the same line.
            buffer = [line]
            open_depth = line_open
        else:
            joined.append(line)

    if buffer:
        # Unbalanced at end of input; pass through as-is so the
        # downstream parser raises a clear error.
        joined.append(' '.join(buffer))

    return joined


def _count_unbalanced_braces(line: str) -> int:
    """Return `open - close` brace count, ignoring `{`/`}` inside `(...)`.

    Positive means the line opens more braces than it closes; negative means
    the line closes more than it opens; 0 means balanced (or no braces).
    """
    paren_depth = 0
    open_count = 0
    close_count = 0
    in_quote = None
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
```

In `load_rules_from_dsl`, find the line iteration. Currently something like:

```python
    lines = text.split('\n')
    for line in lines:
        ...
```

Insert the join step before iteration:

```python
    lines = text.split('\n')
    lines = _join_multi_line_annotations(lines)
    for line in lines:
        ...
```

- [ ] **Step 4: Run tests**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_dsl_metadata.py -v 2>&1 | tail -20
```

Expected: all pass.

- [ ] **Step 5: Run full suite**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest 2>&1 | tail -3
```

- [ ] **Step 6: Commit**

```bash
cd /home/spinoza/github/repos/rerum && git add rerum/engine.py rerum/tests/test_dsl_metadata.py
git commit -m "$(cat <<'EOF'
feat(dsl): support multi-line {category=X} annotation blocks

load_rules_from_dsl now joins multi-line `{...}` annotation blocks onto
their start line before parsing. Brace counting respects s-expression
depth and quoted values so it does not match braces inside expression
literals or strings.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: JSON loader reads new fields

**Files:**
- Modify: `rerum/engine.py` (`load_rules_from_json`, around line 656)
- Test: `rerum/tests/test_metadata.py`

- [ ] **Step 1: Write the failing test**

Append to `rerum/tests/test_metadata.py`:

```python
class TestJSONLoaderNewFields:
    def test_load_category(self):
        from rerum.engine import load_rules_from_json
        text = '{"rules": [{"name": "r1", "category": "identity",' \
               ' "pattern": ["a", ["?", "x"]], "skeleton": [":", "x"]}]}'
        rules = load_rules_from_json(text)
        meta, _ = rules[0]
        assert meta.category == "identity"

    def test_load_reasoning(self):
        from rerum.engine import load_rules_from_json
        text = '{"rules": [{"name": "r1", "reasoning": "Because zero",' \
               ' "pattern": ["a", ["?", "x"]], "skeleton": [":", "x"]}]}'
        rules = load_rules_from_json(text)
        meta, _ = rules[0]
        assert meta.reasoning == "Because zero"

    def test_load_examples(self):
        from rerum.engine import load_rules_from_json
        text = '{"rules": [{"name": "r1",' \
               ' "examples": [{"in": "(a 5)", "out": "5"}],' \
               ' "pattern": ["a", ["?", "x"]], "skeleton": [":", "x"]}]}'
        rules = load_rules_from_json(text)
        meta, _ = rules[0]
        assert meta.examples == [{"in": "(a 5)", "out": "5"}]

    def test_load_bidirectional_with_labels(self):
        from rerum.engine import load_rules_from_json
        text = '{"rules": [{"name": "assoc", "bidirectional": true,' \
               ' "fwd_label": "regroup-right", "rev_label": "regroup-left",' \
               ' "pattern": ["+", ["+", ["?", "x"], ["?", "y"]], ["?", "z"]],' \
               ' "skeleton": ["+", [":", "x"], ["+", [":", "y"], [":", "z"]]]}]}'
        rules = load_rules_from_json(text)
        # Bidirectional yields fwd and rev pair.
        assert len(rules) == 2
        # Fwd metadata carries fwd_label; rev carries rev_label.
        fwd_meta = next(m for m, _ in rules if m.direction == "fwd")
        rev_meta = next(m for m, _ in rules if m.direction == "rev")
        assert fwd_meta.fwd_label == "regroup-right"
        assert rev_meta.rev_label == "regroup-left"

    def test_load_unidirectional_with_label_raises(self):
        from rerum.engine import load_rules_from_json
        text = '{"rules": [{"name": "r1",' \
               ' "fwd_label": "x", "bidirectional": false,' \
               ' "pattern": ["a", ["?", "x"]], "skeleton": [":", "x"]}]}'
        with pytest.raises(ValueError, match="fwd_label"):
            load_rules_from_json(text)

    def test_missing_new_fields_default_to_none(self):
        from rerum.engine import load_rules_from_json
        text = '{"rules": [{"name": "r1",' \
               ' "pattern": ["a", ["?", "x"]], "skeleton": [":", "x"]}]}'
        rules = load_rules_from_json(text)
        meta, _ = rules[0]
        assert meta.category is None
        assert meta.reasoning is None
        assert meta.examples is None
        assert meta.fwd_label is None
        assert meta.rev_label is None

    def test_unknown_fields_preserved_in_extra(self):
        from rerum.engine import load_rules_from_json
        text = '{"rules": [{"name": "r1", "weird_field": "value",' \
               ' "pattern": ["a", ["?", "x"]], "skeleton": [":", "x"]}]}'
        rules = load_rules_from_json(text)
        meta, _ = rules[0]
        assert meta.extra.get("weird_field") == "value"
```

- [ ] **Step 2: Verify failure**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_metadata.py::TestJSONLoaderNewFields -v 2>&1 | tail -15
```

Expected: failures (loader does not yet read the new fields).

- [ ] **Step 3: Update `load_rules_from_json`**

Find `load_rules_from_json` (around line 656). It currently constructs `RuleMetadata` from a subset of fields. Update to read the four new fields and to validate that `fwd_label`/`rev_label` only appear on bidirectional rules.

Find the unidirectional branch:

```python
        else:
            # Bidirectional rules are stored once (with bidirectional=true) and
            # expanded back into -fwd/-rev pairs at load time...
            ...
            metadata = RuleMetadata(
                name=rule.get('name'),
                description=rule.get('description'),
                tags=rule.get('tags'),
                priority=rule.get('priority', 0),
                condition=rule.get('condition'),
            )
```

Update the metadata constructor call to include new fields and propagate unknown fields into `extra`:

```python
            # Validate that fwd_label/rev_label are not set on unidirectional rules.
            if rule.get('fwd_label') is not None or rule.get('rev_label') is not None:
                raise ValueError(
                    f"fwd_label/rev_label only valid on bidirectional rules; "
                    f"got rule {rule.get('name')!r}"
                )

            # Known fields; everything else lands in `extra`.
            known = {
                'name', 'description', 'tags', 'priority', 'condition',
                'bidirectional', 'pattern', 'skeleton',
                'category', 'reasoning', 'examples',
                'fwd_label', 'rev_label',
            }
            extra = {k: v for k, v in rule.items() if k not in known}

            metadata = RuleMetadata(
                name=rule.get('name'),
                description=rule.get('description'),
                tags=rule.get('tags'),
                priority=rule.get('priority', 0),
                condition=rule.get('condition'),
                category=rule.get('category'),
                reasoning=rule.get('reasoning'),
                examples=rule.get('examples'),
                extra=extra or None,
            )
```

For the bidirectional branch (the `if rule.get('bidirectional'):` clause), update the `_build_bidirectional_rules` call to pass `category` and additionally set `fwd_label`/`rev_label` on the resulting metadata:

```python
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
                # Set additional metadata that isn't a parameter.
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
                    meta.examples = examples
                    rules.append((meta, [pat, skel]))
                continue
```

- [ ] **Step 4: Run tests**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_metadata.py::TestJSONLoaderNewFields -v 2>&1 | tail -15
```

Expected: 7 passed.

- [ ] **Step 5: Run full suite**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest 2>&1 | tail -3
```

- [ ] **Step 6: Commit**

```bash
cd /home/spinoza/github/repos/rerum && git add rerum/engine.py rerum/tests/test_metadata.py
git commit -m "$(cat <<'EOF'
feat(metadata): JSON loader reads category/reasoning/examples/labels

load_rules_from_json now reads the four new fields. fwd_label/rev_label
on a unidirectional rule raises ValueError. Unknown JSON fields land in
RuleMetadata.extra (preserves forward compatibility). Bidirectional rules
propagate labels to the appropriate -fwd/-rev half.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: JSON/DSL serializers emit new fields

**Files:**
- Modify: `rerum/engine.py` (`to_dsl`, `to_json`, `to_dict`, around lines 2294/2370/2393, plus `_rules_as_dicts`)
- Test: `rerum/tests/test_metadata.py`

- [ ] **Step 1: Write the failing test**

Append to `rerum/tests/test_metadata.py`:

```python
class TestRoundtripNewFields:
    def test_to_dsl_emits_category(self):
        from rerum import RuleEngine
        engine = RuleEngine.from_dsl(
            '@r1 {category=identity}: (a ?x) => :x'
        )
        dsl = engine.to_dsl()
        assert "{category=identity}" in dsl

    def test_to_dsl_no_category_no_annotation(self):
        from rerum import RuleEngine
        engine = RuleEngine.from_dsl('@r1: (a ?x) => :x')
        dsl = engine.to_dsl()
        # No annotation block when category is None.
        assert "{category=" not in dsl

    def test_dsl_roundtrip_preserves_category(self):
        from rerum import RuleEngine
        engine1 = RuleEngine.from_dsl(
            '@r1 {category=identity}: (a ?x) => :x'
        )
        engine2 = RuleEngine.from_dsl(engine1.to_dsl())
        _, meta = engine2["r1"]
        assert meta.category == "identity"

    def test_to_json_emits_all_four_fields(self):
        from rerum import RuleEngine
        from rerum.engine import RuleMetadata
        engine = RuleEngine()
        engine.add_rule(
            pattern=["a", ["?", "x"]],
            skeleton=[":", "x"],
            name="r1",
        )
        # Manually set the new fields (add_rule kwargs come in Task 9).
        engine._metadata[0].category = "identity"
        engine._metadata[0].reasoning = "Because zero"
        engine._metadata[0].examples = [{"in": "(a 5)", "out": "5"}]
        d = engine.to_dict()
        rule = d["rules"][0]
        assert rule["category"] == "identity"
        assert rule["reasoning"] == "Because zero"
        assert rule["examples"] == [{"in": "(a 5)", "out": "5"}]

    def test_to_json_omits_none_fields(self):
        from rerum import RuleEngine
        engine = RuleEngine.from_dsl('@r1: (a ?x) => :x')
        d = engine.to_dict()
        rule = d["rules"][0]
        # Fields with None values are not in the output.
        assert "category" not in rule
        assert "reasoning" not in rule
        assert "examples" not in rule

    def test_bidirectional_roundtrip_preserves_labels(self):
        from rerum import RuleEngine
        from rerum.engine import load_rules_from_json
        import json
        text = json.dumps({"rules": [{
            "name": "assoc", "bidirectional": True,
            "fwd_label": "regroup-right", "rev_label": "regroup-left",
            "pattern": ["+", ["+", ["?", "x"], ["?", "y"]], ["?", "z"]],
            "skeleton": ["+", [":", "x"], ["+", [":", "y"], [":", "z"]]],
        }]})
        engine = RuleEngine()
        for meta, rule in load_rules_from_json(text):
            engine._rules.append(rule)
            engine._metadata.append(meta)
            if meta.name:
                engine._rule_names[meta.name] = len(engine._rules) - 1
        engine._sort_by_priority()
        # Roundtrip: serialize and reload.
        d = engine.to_dict()
        rule = d["rules"][0]
        assert rule["fwd_label"] == "regroup-right"
        assert rule["rev_label"] == "regroup-left"
        assert rule["bidirectional"] is True
```

- [ ] **Step 2: Verify failure**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_metadata.py::TestRoundtripNewFields -v 2>&1 | tail -15
```

Expected: failures.

- [ ] **Step 3: Update `to_dsl`**

Find `to_dsl` (around line 2294). The function emits `@name[priority] "description": pattern => skeleton`. Add the annotation when `category` is set.

Inside the rule emission loop, after building `name_part` and before constructing `rule_str`, add:

```python
            # Annotation block (v0.7).
            annotation_part = ""
            if meta.category is not None:
                annotation_part = f" {{category={meta.category}}}"
```

Then change:

```python
            rule_str = f"{name_part}{pattern_str} {arrow} {skeleton_str}"
```

to insert `annotation_part`. Note that the `name_part` ends with `: ` for named rules; the annotation must come *before* the colon. Restructure:

For *named* rules with annotation:
```python
            if meta.name:
                # Construct: @name[prio] "desc" {category=X}: pattern <arrow> skeleton
                hdr = f"@{meta.name}"
                if meta.priority != 0:
                    hdr += f"[{meta.priority}]"
                if meta.description:
                    hdr += f' "{meta.description}"'
                if meta.category is not None:
                    hdr += f" {{category={meta.category}}}"
                hdr += ": "
            else:
                hdr = ""
                if meta.category is not None:
                    hdr = f"{{category={meta.category}}}: "
            rule_str = f"{hdr}{pattern_str} {arrow} {skeleton_str}"
            if meta.condition:
                rule_str += f" when {format_sexpr(meta.condition)}"
```

This restructure replaces the existing header-building block. Apply consistently to both the bidirectional and unidirectional emission branches in `to_dsl`. (Note: `_strip_bidirectional_naming` remains the source of `base_name` and `base_description` for `<=>` rules; the `category` is already on both halves and is identical.)

- [ ] **Step 4: Update `_rules_as_dicts` and `to_dict`/`to_json`**

Find `_rules_as_dicts` (around line 2400 area, called by `to_json` and `to_dict`). Update both branches (bidirectional and unidirectional) to include the new fields when set. The pattern:

```python
            if meta.category is not None:
                rule_dict["category"] = meta.category
            if meta.reasoning is not None:
                rule_dict["reasoning"] = meta.reasoning
            if meta.examples is not None:
                rule_dict["examples"] = meta.examples
```

For the bidirectional branch, also include direction labels (drawn from the fwd half's `fwd_label` and the rev half's `rev_label`):

```python
                # Direction labels (v0.7).
                if meta.fwd_label is not None:
                    rule_dict["fwd_label"] = meta.fwd_label
                # Look up the rev half (next entry, validated by _is_bidirectional_pair).
                rev_meta = self._metadata[i + 1]
                if rev_meta.rev_label is not None:
                    rule_dict["rev_label"] = rev_meta.rev_label
```

Place this inside the bidirectional emission block, after the existing fields are set.

- [ ] **Step 5: Run tests**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_metadata.py::TestRoundtripNewFields -v 2>&1 | tail -15
```

Expected: 6 passed.

- [ ] **Step 6: Run full suite**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest 2>&1 | tail -3
```

- [ ] **Step 7: Commit**

```bash
cd /home/spinoza/github/repos/rerum && git add rerum/engine.py rerum/tests/test_metadata.py
git commit -m "$(cat <<'EOF'
feat(metadata): serializers emit category/reasoning/examples/labels

to_dsl includes {category=X} annotation when set. to_json/to_dict include
category, reasoning, examples, and (for bidirectional rules) fwd_label
and rev_label. None fields are omitted. DSL and JSON roundtrips preserve
the new fields.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: ExampleValidationError + `_validate_example` helper

**Files:**
- Modify: `rerum/engine.py` (new exception + helper)
- Modify: `rerum/__init__.py` (re-export)
- Test: `rerum/tests/test_examples_validation.py` (new)

- [ ] **Step 1: Write the failing test**

Create `rerum/tests/test_examples_validation.py`:

```python
"""Tests for examples validation at load time."""

import pytest
from rerum import RuleEngine
from rerum.engine import (
    ExampleValidationError, _validate_example, RuleMetadata,
    parse_sexpr, ARITHMETIC_PRELUDE,
)


class TestValidateExampleHelper:
    def test_pass_simple(self):
        # @add-zero: (+ ?x 0) => :x
        # Example: (+ y 0) -> y
        meta = RuleMetadata(name="add-zero")
        pattern = ["+", ["?", "x"], 0]
        skeleton = [":", "x"]
        example = {"in": "(+ y 0)", "out": "y"}
        # Should not raise.
        _validate_example(pattern, skeleton, meta, example, fold_funcs={})

    def test_fail_pattern_mismatch(self):
        meta = RuleMetadata(name="add-zero")
        pattern = ["+", ["?", "x"], 0]
        skeleton = [":", "x"]
        # Input doesn't match pattern.
        example = {"in": "(* y 0)", "out": "y"}
        with pytest.raises(ExampleValidationError, match="pattern does not match"):
            _validate_example(pattern, skeleton, meta, example, fold_funcs={})

    def test_fail_output_mismatch(self):
        meta = RuleMetadata(name="add-zero")
        pattern = ["+", ["?", "x"], 0]
        skeleton = [":", "x"]
        # Wrong expected output.
        example = {"in": "(+ y 0)", "out": "z"}
        with pytest.raises(ExampleValidationError, match="produced"):
            _validate_example(pattern, skeleton, meta, example, fold_funcs={})

    def test_fail_condition_fails(self):
        # Rule with a condition; example input doesn't satisfy it.
        from rerum.engine import parse_rule_line
        results = parse_rule_line(
            "@fold: (+ ?a ?b) => (! + :a :b) when (! and (! const? :a) (! const? :b))"
        )
        meta, pattern, skeleton = results[0]
        example = {"in": "(+ x 0)", "out": "x"}  # x is not const, condition fails
        with pytest.raises(ExampleValidationError, match="condition fails"):
            _validate_example(pattern, skeleton, meta, example,
                              fold_funcs=ARITHMETIC_PRELUDE)

    def test_pass_with_fold_funcs(self):
        # @fold: (+ ?a:const ?b:const) => (! + :a :b)
        from rerum.engine import parse_rule_line
        results = parse_rule_line(
            "@fold: (+ ?a:const ?b:const) => (! + :a :b)"
        )
        meta, pattern, skeleton = results[0]
        example = {"in": "(+ 2 3)", "out": "5"}
        # Should not raise.
        _validate_example(pattern, skeleton, meta, example,
                          fold_funcs=ARITHMETIC_PRELUDE)

    def test_explicit_rev_direction(self):
        # Bidirectional commute. The rev direction is (+ ?y ?x) => (+ :x :y).
        from rerum.engine import parse_rule_line
        results = parse_rule_line("@commute: (+ ?x ?y) <=> (+ :y :x)")
        rev_meta, rev_pattern, rev_skeleton = results[1]  # -rev
        # rev pattern matches (+ b a); applies (+ :x :y) where :x and :y are
        # bound from the rev pattern.
        example = {"in": "(+ b a)", "out": "(+ a b)"}
        _validate_example(rev_pattern, rev_skeleton, rev_meta, example,
                          fold_funcs={})
```

- [ ] **Step 2: Verify failure**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_examples_validation.py::TestValidateExampleHelper -v 2>&1 | tail -15
```

Expected: ImportError on `ExampleValidationError` and `_validate_example`.

- [ ] **Step 3: Add `ExampleValidationError` and `_validate_example`**

In `rerum/engine.py`, near the other exception/error classes (after `_validate_pattern_structure` or similar utility helpers, before `RuleEngine`), add:

```python
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


def _validate_example(pattern, skeleton, metadata, example, fold_funcs):
    """Validate one example against a rule. Raises ExampleValidationError on mismatch.

    `example` is a dict with `in` (s-expr string), `out` (s-expr string),
    and an optional `direction` field which is informational only; the
    caller is responsible for selecting the right (pattern, skeleton)
    pair for bidirectional rules.

    `fold_funcs` is the engine's prelude; needed for `(! op ...)` evaluation
    in skeletons or conditions.
    """
    in_expr = parse_sexpr(example["in"])
    expected_out = parse_sexpr(example["out"])

    bindings = match(pattern, in_expr)
    if bindings is None:
        raise ExampleValidationError(
            f"Rule {metadata.name!r}: pattern does not match input "
            f"{example['in']!r}",
            rule_name=metadata.name,
            example=example,
        )

    if metadata.condition is not None:
        # Use the engine's _check_condition equivalent. Inline simplified
        # version: instantiate condition, check truthiness.
        cond_result = instantiate(metadata.condition, bindings, fold_funcs)
        if not _condition_truthy(cond_result):
            raise ExampleValidationError(
                f"Rule {metadata.name!r}: condition fails on input "
                f"{example['in']!r}",
                rule_name=metadata.name,
                example=example,
            )

    actual = instantiate(skeleton, bindings, fold_funcs)
    if actual != expected_out:
        raise ExampleValidationError(
            f"Rule {metadata.name!r}: input {example['in']!r} produced "
            f"{format_sexpr(actual)!r}, expected {example['out']!r}",
            rule_name=metadata.name,
            example=example,
        )


def _condition_truthy(result) -> bool:
    """Truthiness rule for condition expressions, mirroring _check_condition."""
    if isinstance(result, bool):
        return result
    if isinstance(result, (int, float)):
        return result != 0
    if isinstance(result, str):
        return len(result) > 0
    if isinstance(result, list):
        return len(result) > 0
    return True
```

(The `match` import comes from the existing engine import block; `instantiate` and `parse_sexpr` are also already imported.)

In `rerum/__init__.py`, add `ExampleValidationError` to the imports and `__all__`:

```python
from .engine import (
    ...,
    ExampleValidationError,
)

__all__ = [
    ...,
    "ExampleValidationError",
]
```

- [ ] **Step 4: Run tests**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_examples_validation.py::TestValidateExampleHelper -v 2>&1 | tail -15
```

Expected: 6 passed.

- [ ] **Step 5: Run full suite**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest 2>&1 | tail -3
```

- [ ] **Step 6: Commit**

```bash
cd /home/spinoza/github/repos/rerum && git add rerum/engine.py rerum/__init__.py rerum/tests/test_examples_validation.py
git commit -m "$(cat <<'EOF'
feat(metadata): ExampleValidationError and _validate_example helper

Adds a single-example validator that applies a rule once to its example
input and asserts the output matches. Raises ExampleValidationError on
pattern mismatch, condition failure, or output mismatch. Supports
condition checking via fold_funcs for compute-form predicates.

Public types ExampleValidationError exported from rerum.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Hook examples validation into all loaders

**Files:**
- Modify: `rerum/engine.py` (`load_dsl`, `load_file`, `load_rules`, plus signatures of `load_rules_from_dsl`/`load_rules_from_json` if they grow the kwarg)
- Test: `rerum/tests/test_examples_validation.py`

- [ ] **Step 1: Write the failing test**

Append to `rerum/tests/test_examples_validation.py`:

```python
class TestLoaderValidation:
    def test_load_rules_with_valid_examples_passes(self):
        from rerum import RuleEngine
        from rerum.engine import load_rules_from_json
        text = '''{"rules": [{
            "name": "add-zero",
            "pattern": ["+", ["?", "x"], 0],
            "skeleton": [":", "x"],
            "examples": [{"in": "(+ y 0)", "out": "y"}]
        }]}'''
        engine = RuleEngine()
        engine.load_rules_from_json(text)  # method on engine; should not raise
        _, meta = engine["add-zero"]
        assert meta.examples == [{"in": "(+ y 0)", "out": "y"}]

    def test_load_rules_with_bad_example_raises(self):
        from rerum import RuleEngine
        text = '''{"rules": [{
            "name": "broken",
            "pattern": ["+", ["?", "x"], 0],
            "skeleton": [":", "x"],
            "examples": [{"in": "(+ y 0)", "out": "z"}]
        }]}'''
        engine = RuleEngine()
        with pytest.raises(ExampleValidationError, match="produced"):
            engine.load_rules_from_json(text)

    def test_validate_examples_false_skips_validation(self):
        from rerum import RuleEngine
        text = '''{"rules": [{
            "name": "broken",
            "pattern": ["+", ["?", "x"], 0],
            "skeleton": [":", "x"],
            "examples": [{"in": "(+ y 0)", "out": "z"}]
        }]}'''
        engine = RuleEngine()
        # No validation; should load.
        engine.load_rules_from_json(text, validate_examples=False)
        _, meta = engine["broken"]
        assert meta.examples == [{"in": "(+ y 0)", "out": "z"}]

    def test_first_failing_example_reported_loud(self):
        from rerum import RuleEngine
        text = '''{"rules": [{
            "name": "rule-a",
            "pattern": ["a", ["?", "x"]],
            "skeleton": [":", "x"],
            "examples": [{"in": "(a 1)", "out": "1"}, {"in": "(a 2)", "out": "wrong"}]
        }]}'''
        engine = RuleEngine()
        with pytest.raises(ExampleValidationError) as exc_info:
            engine.load_rules_from_json(text)
        # Second example is the failing one.
        assert exc_info.value.example == {"in": "(a 2)", "out": "wrong"}

    def test_bidirectional_example_with_rev_direction(self):
        from rerum import RuleEngine
        text = '''{"rules": [{
            "name": "commute",
            "bidirectional": true,
            "pattern": ["+", ["?", "x"], ["?", "y"]],
            "skeleton": ["+", [":", "y"], [":", "x"]],
            "examples": [
                {"in": "(+ a b)", "out": "(+ b a)", "direction": "fwd"},
                {"in": "(+ a b)", "out": "(+ b a)", "direction": "rev"}
            ]
        }]}'''
        engine = RuleEngine()
        engine.load_rules_from_json(text)  # both directions valid
```

- [ ] **Step 2: Verify failure**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_examples_validation.py::TestLoaderValidation -v 2>&1 | tail -15
```

Expected: failures (no validation hook yet).

- [ ] **Step 3: Add validation to engine load methods**

Add a helper method on `RuleEngine` (place near `_install_resolver_rules`):

```python
    def _validate_rule_examples(self, rule, metadata) -> None:
        """Validate every example for a rule. Raises ExampleValidationError on
        the first failing example.

        For bidirectional rules, examples may carry a `direction` field
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
                pattern, skeleton, metadata, example, self._fold_funcs or {}
            )
```

Update each engine-level loader to call this after appending each rule. Find `load_dsl`, `load_file`, `load_rules` methods. Each currently looks like:

```python
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
        self._simplifier = None
        return self
```

Update each to accept `validate_examples=True` and run validation:

```python
    def load_dsl(self, text: str, validate_examples: bool = True) -> 'RuleEngine':
        """Load rules from DSL text. Validates examples by default."""
        parsed = load_rules_from_dsl(text)
        for metadata, rule in parsed:
            if validate_examples:
                self._validate_rule_examples(rule, metadata)
            idx = len(self._rules)
            self._rules.append(rule)
            self._metadata.append(metadata)
            if metadata.name:
                self._rule_names[metadata.name] = idx
        self._sort_by_priority()
        self._simplifier = None
        return self
```

Apply the same pattern to `load_file` and `load_rules`. Also add a new engine method `load_rules_from_json` (mirrored to delegate to the module function but accept the kwarg):

```python
    def load_rules_from_json(self, text: str,
                             validate_examples: bool = True) -> 'RuleEngine':
        """Load rules from JSON text. Validates examples by default."""
        parsed = load_rules_from_json(text)
        for metadata, rule in parsed:
            if validate_examples:
                self._validate_rule_examples(rule, metadata)
            idx = len(self._rules)
            self._rules.append(rule)
            self._metadata.append(metadata)
            if metadata.name:
                self._rule_names[metadata.name] = idx
        self._sort_by_priority()
        self._simplifier = None
        return self
```

Note: `load_rules_from_json` (module function) does NOT validate; the engine method does. Keeps the module function pure.

- [ ] **Step 4: Run tests**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_examples_validation.py::TestLoaderValidation -v 2>&1 | tail -15
```

Expected: 5 passed.

- [ ] **Step 5: Run full suite**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest 2>&1 | tail -3
```

- [ ] **Step 6: Commit**

```bash
cd /home/spinoza/github/repos/rerum && git add rerum/engine.py rerum/tests/test_examples_validation.py
git commit -m "$(cat <<'EOF'
feat(metadata): validate examples at load time across all loaders

load_dsl, load_file, load_rules, load_rules_from_json each accept
validate_examples=True (default). When True, each rule with examples is
validated immediately on load; failure raises ExampleValidationError.
Bidirectional examples honor the optional direction field, validating
against the appropriate -fwd or -rev half.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: On-demand `engine.validate_examples()`

**Files:**
- Modify: `rerum/engine.py` (new method on RuleEngine)
- Test: `rerum/tests/test_examples_validation.py`

- [ ] **Step 1: Write the failing test**

Append to `rerum/tests/test_examples_validation.py`:

```python
class TestOnDemandValidation:
    def test_validate_examples_walks_all_rules(self):
        from rerum import RuleEngine
        text = '''{"rules": [
            {"name": "r1",
             "pattern": ["a", ["?", "x"]], "skeleton": [":", "x"],
             "examples": [{"in": "(a 5)", "out": "5"}]},
            {"name": "r2",
             "pattern": ["b", ["?", "x"]], "skeleton": [":", "x"],
             "examples": [{"in": "(b 7)", "out": "7"}]}
        ]}'''
        engine = RuleEngine()
        engine.load_rules_from_json(text)
        # Should not raise; both rules' examples are valid.
        engine.validate_examples()

    def test_validate_examples_after_prelude_change(self):
        # Load rules without a prelude; load with validate_examples=False
        # because the example needs folding. Then set the prelude and
        # call validate_examples().
        from rerum import RuleEngine, ARITHMETIC_PRELUDE
        text = '''{"rules": [{
            "name": "fold-add",
            "pattern": ["+", ["?c", "a"], ["?c", "b"]],
            "skeleton": ["!", "+", [":", "a"], [":", "b"]],
            "examples": [{"in": "(+ 2 3)", "out": "5"}]
        }]}'''
        engine = RuleEngine()
        engine.load_rules_from_json(text, validate_examples=False)
        # Without fold_funcs, validation would fail.
        engine._fold_funcs = ARITHMETIC_PRELUDE
        engine.validate_examples()  # should pass now

    def test_validate_examples_raises_on_bad(self):
        from rerum import RuleEngine
        text = '''{"rules": [{
            "name": "bad",
            "pattern": ["a", ["?", "x"]], "skeleton": [":", "x"],
            "examples": [{"in": "(a 1)", "out": "wrong"}]
        }]}'''
        engine = RuleEngine()
        engine.load_rules_from_json(text, validate_examples=False)
        with pytest.raises(ExampleValidationError):
            engine.validate_examples()
```

- [ ] **Step 2: Verify failure**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_examples_validation.py::TestOnDemandValidation -v 2>&1 | tail -15
```

Expected: AttributeError on `engine.validate_examples`.

- [ ] **Step 3: Add `validate_examples` method on RuleEngine**

In `rerum/engine.py`, near `_validate_rule_examples`:

```python
    def validate_examples(self) -> None:
        """Validate every example for every rule in the engine.

        Raises ExampleValidationError on the first failing example. Useful
        after a prelude change or after loading rules with
        validate_examples=False.
        """
        for rule, metadata in zip(self._rules, self._metadata):
            self._validate_rule_examples(rule, metadata)
```

- [ ] **Step 4: Run tests**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_examples_validation.py::TestOnDemandValidation -v 2>&1 | tail -15
```

Expected: 3 passed.

- [ ] **Step 5: Full suite**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest 2>&1 | tail -3
```

- [ ] **Step 6: Commit**

```bash
cd /home/spinoza/github/repos/rerum && git add rerum/engine.py rerum/tests/test_examples_validation.py
git commit -m "$(cat <<'EOF'
feat(metadata): on-demand engine.validate_examples() method

Walks every rule with examples and validates them. Useful after a prelude
change or when rules were loaded with validate_examples=False. Fail-fast
on the first error.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Sidecar `load_metadata_json`

**Files:**
- Modify: `rerum/engine.py` (new method)
- Test: `rerum/tests/test_metadata.py`

- [ ] **Step 1: Write the failing test**

Append to `rerum/tests/test_metadata.py`:

```python
class TestSidecarLoad:
    def test_sidecar_fills_missing_category(self):
        from rerum import RuleEngine
        engine = RuleEngine.from_dsl('@r1: (a ?x) => :x')
        engine.load_metadata_json('{"r1": {"category": "identity"}}')
        _, meta = engine["r1"]
        assert meta.category == "identity"

    def test_sidecar_fills_reasoning_and_examples(self):
        from rerum import RuleEngine
        engine = RuleEngine.from_dsl('@r1: (a ?x) => :x')
        engine.load_metadata_json('''{
            "r1": {
                "reasoning": "Because identity",
                "examples": [{"in": "(a 5)", "out": "5"}]
            }
        }''')
        _, meta = engine["r1"]
        assert meta.reasoning == "Because identity"
        assert meta.examples == [{"in": "(a 5)", "out": "5"}]

    def test_sidecar_conflict_raises(self):
        from rerum import RuleEngine
        engine = RuleEngine.from_dsl(
            '@r1 {category=existing}: (a ?x) => :x'
        )
        with pytest.raises(ValueError, match="conflict"):
            engine.load_metadata_json('{"r1": {"category": "different"}}')

    def test_sidecar_orphan_raises(self):
        from rerum import RuleEngine
        engine = RuleEngine.from_dsl('@r1: (a ?x) => :x')
        with pytest.raises(ValueError, match="no rule named"):
            engine.load_metadata_json('{"nonexistent": {"category": "x"}}')

    def test_sidecar_validates_examples(self):
        from rerum import RuleEngine
        engine = RuleEngine.from_dsl('@r1: (a ?x) => :x')
        with pytest.raises(ExampleValidationError):
            engine.load_metadata_json('''{
                "r1": {"examples": [{"in": "(a 5)", "out": "wrong"}]}
            }''')
```

- [ ] **Step 2: Verify failure**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_metadata.py::TestSidecarLoad -v 2>&1 | tail -15
```

Expected: AttributeError on `engine.load_metadata_json`.

- [ ] **Step 3: Add `load_metadata_json`**

In `rerum/engine.py`, near `validate_examples`:

```python
    def load_metadata_json(self, text: str,
                           validate_examples: bool = True) -> 'RuleEngine':
        """Merge a metadata-only JSON sidecar onto already-loaded rules.

        The JSON shape is `{rule_name: {field: value, ...}}`. Each top-level
        key must match an `@name` already in the engine. Each inner field
        merges onto the existing RuleMetadata: only fields that are
        currently None on the rule are filled. A conflict (sidecar tries
        to set a field already set on the rule) raises ValueError.

        After merging, examples are validated by default (validate_examples
        kwarg).
        """
        import json as _json
        data = _json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("sidecar JSON must be an object mapping name -> metadata")

        for rule_name, fields in data.items():
            if rule_name not in self._rule_names:
                raise ValueError(
                    f"sidecar references no rule named {rule_name!r}"
                )
            idx = self._rule_names[rule_name]
            metadata = self._metadata[idx]

            for field, value in fields.items():
                # Validate field is recognised.
                if not hasattr(metadata, field):
                    # Unknown field; preserve in extra.
                    metadata.extra[field] = value
                    continue
                existing = getattr(metadata, field)
                if existing is not None and existing != []:
                    if existing != value:
                        raise ValueError(
                            f"sidecar conflict on rule {rule_name!r} "
                            f"field {field!r}: existing={existing!r}, "
                            f"sidecar={value!r}"
                        )
                    # Same value; harmless.
                    continue
                setattr(metadata, field, value)

            if validate_examples:
                self._validate_rule_examples(self._rules[idx], metadata)

        return self
```

- [ ] **Step 4: Run tests**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_metadata.py::TestSidecarLoad -v 2>&1 | tail -15
```

Expected: 5 passed.

- [ ] **Step 5: Full suite**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest 2>&1 | tail -3
```

- [ ] **Step 6: Commit**

```bash
cd /home/spinoza/github/repos/rerum && git add rerum/engine.py rerum/tests/test_metadata.py
git commit -m "$(cat <<'EOF'
feat(metadata): load_metadata_json sidecar loader

Sidecar JSON shape: {rule_name: {field: value, ...}}. Fills missing fields
on already-loaded rules. Conflicts on already-set fields raise ValueError;
sidecar entries referencing non-existent rules raise ValueError. Examples
validate after merge by default.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Extend `add_rule` with new kwargs

**Files:**
- Modify: `rerum/engine.py` (`add_rule`, around line 1151)
- Test: `rerum/tests/test_metadata.py`

- [ ] **Step 1: Write the failing test**

Append to `rerum/tests/test_metadata.py`:

```python
class TestAddRuleExtended:
    def test_add_rule_with_category(self):
        from rerum import RuleEngine
        engine = RuleEngine()
        engine.add_rule(
            pattern=["a", ["?", "x"]],
            skeleton=[":", "x"],
            name="r1",
            category="identity",
        )
        _, meta = engine["r1"]
        assert meta.category == "identity"

    def test_add_rule_with_reasoning(self):
        from rerum import RuleEngine
        engine = RuleEngine()
        engine.add_rule(
            pattern=["a", ["?", "x"]],
            skeleton=[":", "x"],
            name="r1",
            reasoning="Because zero",
        )
        _, meta = engine["r1"]
        assert meta.reasoning == "Because zero"

    def test_add_rule_with_examples_validates(self):
        from rerum import RuleEngine
        engine = RuleEngine()
        engine.add_rule(
            pattern=["a", ["?", "x"]],
            skeleton=[":", "x"],
            name="r1",
            examples=[{"in": "(a 5)", "out": "5"}],
        )
        _, meta = engine["r1"]
        assert meta.examples == [{"in": "(a 5)", "out": "5"}]

    def test_add_rule_bad_examples_raises(self):
        from rerum import RuleEngine
        engine = RuleEngine()
        with pytest.raises(ExampleValidationError):
            engine.add_rule(
                pattern=["a", ["?", "x"]],
                skeleton=[":", "x"],
                name="r1",
                examples=[{"in": "(a 5)", "out": "wrong"}],
            )

    def test_add_rule_validate_examples_false(self):
        from rerum import RuleEngine
        engine = RuleEngine()
        # Should not raise even with bad example.
        engine.add_rule(
            pattern=["a", ["?", "x"]],
            skeleton=[":", "x"],
            name="r1",
            examples=[{"in": "(a 5)", "out": "wrong"}],
            validate_examples=False,
        )
```

- [ ] **Step 2: Verify failure**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_metadata.py::TestAddRuleExtended -v 2>&1 | tail -15
```

Expected: TypeError on the new kwargs.

- [ ] **Step 3: Update `add_rule`**

Find `add_rule` (around line 1151). Replace with:

```python
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
        ``load_rules_from_json``, or pre-built ``_build_bidirectional_rules``
        output. ``add_rule`` is for unidirectional rules only.
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
        if validate_examples:
            self._validate_rule_examples(rule, metadata)
        idx = len(self._rules)
        self._rules.append(rule)
        self._metadata.append(metadata)
        if name:
            self._rule_names[name] = idx
        self._sort_by_priority()
        self._simplifier = None
        return self
```

- [ ] **Step 4: Run tests**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_metadata.py::TestAddRuleExtended -v 2>&1 | tail -15
```

Expected: 5 passed.

- [ ] **Step 5: Full suite**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest 2>&1 | tail -3
```

- [ ] **Step 6: Commit**

```bash
cd /home/spinoza/github/repos/rerum && git add rerum/engine.py rerum/tests/test_metadata.py
git commit -m "$(cat <<'EOF'
feat(metadata): extend add_rule with category/reasoning/examples kwargs

add_rule now accepts category, reasoning, examples, plus priority,
condition, and tags (which were missing from the old signature). When
examples are provided, they validate at insertion time unless
validate_examples=False is passed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: CHANGELOG, version bump, CLAUDE.md

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `pyproject.toml`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Bump version**

In `pyproject.toml`, change `version = "0.5.0"` to `version = "0.7.0"`.

In `rerum/__init__.py`, change `__version__ = "0.5.0"` to `__version__ = "0.7.0"`.

- [ ] **Step 2: CHANGELOG entry**

In `CHANGELOG.md`, promote the existing `[Unreleased]` section to `[0.7.0]` and add a new metadata-layer subsection:

```markdown
## [0.7.0]

### Added (metadata layer)
- ``RuleMetadata`` gains four new fields: ``category`` (free-form string,
  for LLM paraphrasing), ``reasoning`` (free text justification),
  ``examples`` (list of ``{in, out}`` s-expression strings, validated on
  load), and ``fwd_label``/``rev_label`` (direction semantics for
  ``<=>`` rules).
- DSL annotation ``{category=X}`` between description and colon. Multi-line
  form supported.
- JSON schema extends with the four new fields. Bidirectional rules carry
  ``fwd_label``/``rev_label`` once on the source-rule entry; the loader
  routes to the appropriate ``-fwd``/``-rev`` half.
- Examples validation at load time. Each loader (``load_dsl``,
  ``load_file``, ``load_rules_from_json``, ``add_rule``) accepts a
  ``validate_examples=True`` kwarg. ``engine.validate_examples()`` runs
  on demand.
- ``ExampleValidationError`` raised on pattern mismatch, condition
  failure, or output mismatch.
- ``engine.load_metadata_json(text)`` merges a metadata-only sidecar
  onto already-loaded rules. Sidecar fills missing fields only;
  conflicts raise ``ValueError``.

[Existing Unreleased entries from the hooks system release]
```

(Keep the existing hooks-system content; promote everything together to 0.7.0.)

- [ ] **Step 3: CLAUDE.md update**

In `CLAUDE.md` Architecture section, expand the existing description of `engine.py` to mention the metadata layer:

Add to the engine.py bullet list:

```
- Metadata layer (v0.7): RuleMetadata fields ``category``, ``reasoning``,
  ``examples``, ``fwd_label``/``rev_label``. DSL ``{category=X}`` annotation.
  ``ExampleValidationError`` raised when an example does not match its
  rule. ``engine.load_metadata_json()`` merges sidecar metadata onto
  already-loaded rules.
```

In the Footguns section, add:

```
- **Examples validation needs the prelude**: rules whose examples use
  ``(! op ...)`` compute forms require ``fold_funcs`` to be set before
  the example is validated. Load with ``validate_examples=False`` if
  loading rules before configuring the prelude, then call
  ``engine.validate_examples()`` after the prelude is set.
```

- [ ] **Step 4: Full test run**

```bash
cd /home/spinoza/github/repos/rerum && python -m pytest 2>&1 | tail -3
```

- [ ] **Step 5: Commit**

```bash
cd /home/spinoza/github/repos/rerum && git add CHANGELOG.md pyproject.toml rerum/__init__.py CLAUDE.md
git commit -m "$(cat <<'EOF'
chore: bump version to 0.7.0; document metadata layer

CHANGELOG: promotes Unreleased to 0.7.0 and adds metadata-layer entries
covering RuleMetadata new fields, DSL annotation, JSON schema, examples
validation, ExampleValidationError, and load_metadata_json.

CLAUDE.md: notes the metadata layer in engine.py architecture description
and adds an "examples validation needs prelude" footgun.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-review notes

**Spec coverage:**
- Four `RuleMetadata` fields → Task 1.
- DSL `{category=X}` single-line → Task 2.
- DSL multi-line annotation → Task 3.
- JSON loader → Task 4.
- JSON/DSL serializers → Task 5.
- `ExampleValidationError` + `_validate_example` → Task 6.
- Loader validation hooks (kwarg) → Task 7.
- On-demand `validate_examples` → Task 8.
- Sidecar `load_metadata_json` → Task 9.
- `add_rule` extension → Task 10.
- Version bump + CHANGELOG + CLAUDE.md → Task 11.

All spec sections covered. Type consistency: `RuleMetadata` field names match across all tasks (`category`, `reasoning`, `examples`, `fwd_label`, `rev_label`). Method names consistent: `load_metadata_json`, `validate_examples`, `_validate_example`, `_validate_rule_examples`. Exception name consistent: `ExampleValidationError`.

The plan covers v0.7. Future work (negative examples, MCP surface) is out of scope per the spec.
