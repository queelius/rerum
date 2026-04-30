# Rerum DSL/Metadata Layer: Design Spec

**Date:** 2026-04-29
**Status:** Draft
**Scope:** v0.7 metadata layer. Adds four first-class fields to `RuleMetadata`, one DSL annotation, JSON schema extension, and load-time examples validation. The MCP design that consumes this surface is a separate downstream spec.

## Problem

The existing rule metadata is `name`, `description`, `priority`, `tags` (groups), `condition`, plus the recently-added `bidirectional`/`direction` flags and `extra` dict. The MCP design we plan to follow needs more:

1. **LLM paraphrasing of solution steps.** The hooks system (v0.6) emits `RewriteStep` events naming each rule that fired. For an LLM to narrate "I applied associativity here, then commutativity, then constant folding," each rule needs a *categorical* label the LLM can verbalize. `name` alone is not enough; `add-zero` is opaque to a reader without context.

2. **Justification of rule validity.** When the LLM explains *why* a rule is sound, it should not invent the reason from scratch. A free-text `reasoning` field on each rule grounds the explanation in author-supplied claims.

3. **Self-validating rules.** As the rule library grows, especially with LLM-inferred rules, a way for each rule to ship with `(input, output)` test pairs that the engine verifies on load catches typos and drift early. Examples that ship verified are also LLM-trustworthy.

4. **Direction semantics for bidirectional rules.** A `<=>` rule has two orientations. The engine knows them as `-fwd` and `-rev`. An LLM choosing a direction needs labels that convey intent ("regroup-right" vs "regroup-left", "expand" vs "fold").

This spec adds the metadata fields these uses need without bloating the DSL or breaking any existing rule file.

## Goals

1. **Four new first-class fields**: `category`, `reasoning`, `examples`, plus `fwd_label`/`rev_label` for `<=>` rules.
2. **Minimal DSL surface**: only `category` gets DSL syntax; the rest is JSON-only. The DSL stays line-oriented and human-skimmable.
3. **Examples are executable contracts**: validated at load time. A rule whose examples do not match the rule is rejected with a clear error.
4. **Roundtrip stability**: DSL -> JSON -> DSL preserves `category`; JSON -> JSON preserves all four fields.
5. **JSON is the canonical interchange format** for the eventual MCP. Humans write DSL; LLMs read JSON.

## Non-goals

- A formal category vocabulary. `category` is a free-form string; project conventions emerge from usage.
- Multi-step examples. Each example tests one rule application, not a chain.
- Validation against external proof systems. `reasoning` is free text; the engine does not verify mathematical claims.
- Inline JSON or YAML mixed into `.rules` files. The DSL adds one annotation block; richer metadata uses a separate file.
- Negative examples ("this rule should NOT fire on input X"). Future feature.

## Architecture

### New `RuleMetadata` fields

| Field | Type | Surfaces in | Purpose |
|-------|------|-------------|---------|
| `category` | `Optional[str]` (free-form) | DSL inline `{category=X}` and JSON | LLM-paraphrasing label |
| `reasoning` | `Optional[str]` | JSON only | Natural-language justification |
| `examples` | `Optional[List[Dict]]` | JSON only | Concrete I/O pairs (validated on load) |
| `fwd_label` / `rev_label` | `Optional[str]` | JSON only (only on `<=>` rules) | Direction semantics |

Defaults: all `None`. Existing rules parse and load identically.

### DSL syntax

The DSL gains one annotation block: `{category=X}`, placed after the optional description and before the colon.

```
rule         ::= header? body
header       ::= "@" name priority? description? annotation? ":"
                 | annotation? ":"
priority     ::= "[" INT "]"
description  ::= '"' string '"'
annotation   ::= "{" key "=" value ("," key "=" value)* "}"
key          ::= identifier
value        ::= identifier | string
body         ::= pattern ("=>" | "<=>") skeleton when_clause?
when_clause  ::= "when" expr
```

Examples:

```rerum
@add-zero[100] "x + 0 = x" {category=identity}: (+ ?x 0) => :x
@distrib {category=distributivity}: (* ?x (+ ?y ?z)) => (+ (* :x :y) (* :x :z))
@commute {category=commutativity}: (+ ?x ?y) <=> (+ :y :x)
{category=fold-constant}: (* ?a:const ?b:const) => (! * :a :b)
```

Multi-line annotation form is also accepted (forward-compatible with future keys):

```rerum
@distrib {
  category=distributivity
}: (* ?x (+ ?y ?z)) => (+ (* :x :y) (* :x :z))
```

Parser rules:
- The annotation is the *last* optional header element. Order is fixed: `@name [priority] "description" {annotation} :`.
- v1 has only `category` as a known key. Unknown keys raise `ValueError` at parse time.
- Whitespace is tolerated inside the braces.
- Values may be quoted strings (for multi-word categories): `{category="fold-constant"}`. Single-token values do not need quotes.
- `to_dsl()` re-emits the annotation when `category` is set.

### JSON schema

Per-rule shape (unidirectional):

```json
{
  "name": "add-zero",
  "description": "x + 0 = x",
  "priority": 100,
  "tags": ["identity"],
  "condition": null,
  "bidirectional": false,

  "category": "identity",
  "reasoning": "Zero is the additive identity element of the integers.",
  "examples": [
    {"in": "(+ 5 0)", "out": "5"},
    {"in": "(+ x 0)", "out": "x"}
  ],

  "pattern": ["+", ["?", "x"], 0],
  "skeleton": [":", "x"]
}
```

Bidirectional:

```json
{
  "name": "assoc",
  "description": "Associativity of +",
  "category": "associativity",
  "bidirectional": true,
  "fwd_label": "regroup-right",
  "rev_label": "regroup-left",
  "pattern": ["+", ["+", ["?", "x"], ["?", "y"]], ["?", "z"]],
  "skeleton": ["+", [":", "x"], ["+", [":", "y"], [":", "z"]]]
}
```

Notes:

- Each example is `{"in": "<sexpr-string>", "out": "<sexpr-string>"}`. Strings are friendlier for hand-written JSON than nested-list literals. The engine parses them via `parse_sexpr` at validation time.
- An example may carry an optional `"direction": "fwd" | "rev"` field for bidirectional rules. Default is `"fwd"`.
- Direction labels are stored once on the source-rule JSON. The bidirectional desugaring sets each `RuleMetadata` half's labels appropriately, and `to_json()` recombines the pair back into a single JSON entry.
- Unknown JSON fields are preserved in `RuleMetadata.extra` (already implemented).
- Setting `fwd_label` or `rev_label` on a unidirectional rule raises `ValueError` at load time.

### Examples validation

When a rule with `examples` is loaded, the engine validates each example by applying the rule once and asserting the output matches.

```python
def _validate_example(rule_pattern, rule_skeleton, metadata, example, fold_funcs):
    in_expr = parse_sexpr(example["in"])
    expected_out = parse_sexpr(example["out"])

    bindings = match(rule_pattern, in_expr)
    if bindings is None:
        raise ExampleValidationError(
            f"Rule {metadata.name!r}: pattern does not match input {example['in']!r}"
        )

    if metadata.condition is not None:
        if not _check_condition(metadata.condition, bindings, fold_funcs):
            raise ExampleValidationError(
                f"Rule {metadata.name!r}: condition fails on input {example['in']!r}"
            )

    actual = instantiate(rule_skeleton, bindings, fold_funcs)
    if actual != expected_out:
        raise ExampleValidationError(
            f"Rule {metadata.name!r}: input {example['in']!r} produced "
            f"{format_sexpr(actual)!r}, expected {example['out']!r}"
        )
```

Behavior summary:

- **Single rewrite per example**, not fixpoint. Each example tests what *this rule* does in one application, not what the engine as a whole produces.
- **Validation runs at load time** for every loader: `load_rules_from_dsl`, `load_rules_from_file`, `load_rules_from_json`, `add_rule`. Each loader accepts a `validate_examples=True` kwarg (default True). Pass False to skip during iterative development.
- **Fail-fast**: validation raises at the first failing example. The error message names the rule and the example.
- **Bidirectional**: examples test the forward direction by default. `"direction": "rev"` selects the reverse pattern/skeleton.
- **Prelude**: validation uses the engine's current `fold_funcs`. Configure the prelude before loading rules whose examples need folding.
- **On-demand**: `engine.validate_examples()` runs validation on every rule that has examples (e.g., after a prelude change).

### File extensions and load API

| Extension | Purpose | Loader |
|-----------|---------|--------|
| `.rules` | Rule definitions only (DSL syntax) | `load_file` routes to DSL parser |
| `.json` | Rule definitions with full metadata | `load_file` routes to JSON loader |
| `.rerum` | Executable scripts (rules + expressions) | `rerum` CLI |

Recommendation: hand-write `.rules` for the common case where only `category` is needed; switch to `.json` or pair `.rules` with a `.json` sidecar when richer metadata is needed.

New API:

```python
# Sidecar metadata merge:
engine.load_file("algebra.rules")
engine.load_metadata_json("algebra.meta.json")
```

Sidecar semantics:
- JSON entries with a matching `@name` in the engine: metadata fields merged onto the existing `RuleMetadata`.
- JSON entries without a matching rule: `ValueError` (loud over silent).
- Conflicting fields (already-set in the rule): `ValueError`. Sidecar can only *fill* fields, not override.

### `add_rule` API extension

```python
engine.add_rule(
    pattern=...,
    skeleton=...,
    name="add-zero",
    description="x + 0 = x",
    category="identity",
    reasoning="Zero is the additive identity.",
    examples=[{"in": "(+ y 0)", "out": "y"}],
    priority=100,
)

engine.add_rule(
    pattern=...,
    skeleton=...,
    bidirectional=True,
    category="associativity",
    fwd_label="regroup-right",
    rev_label="regroup-left",
)
```

All new kwargs are optional with defaults of `None`. Existing positional callers continue to work.

## Errors

- `ExampleValidationError(HooksError)` (or sibling base in `rerum/exceptions.py`): raised at load time when an example does not match its rule. Carries `rule_name`, `example`, and a description of the mismatch.
- `ValueError` for malformed DSL annotation, unknown annotation key, `fwd_label`/`rev_label` on unidirectional rules, or sidecar-JSON references to non-existent rules.

## Test plan

1. **Parser tests** (`test_dsl_metadata.py`): single-line `{category=X}`, multi-line, with/without name, with/without priority, with/without description, anonymous rules, error on unknown key, error on malformed annotation.
2. **JSON loader tests** (extending `test_engine_methods.py`): all four new fields parse round-trip; missing fields default to None; unknown fields preserved in `extra`; conflicting `fwd_label` on unidirectional raises.
3. **Sidecar merge tests**: sidecar fills missing fields; conflict on already-set field raises; orphaned JSON entry raises.
4. **Examples validation tests** (`test_examples_validation.py`): happy path, pattern-mismatch, condition-fails, output-mismatch, direction `"rev"` on bidirectional, `validate_examples=False` skip, on-demand `engine.validate_examples()` after prelude change.
5. **Roundtrip tests**: DSL -> `to_dsl` -> DSL preserves `category`; JSON -> `to_json` -> JSON preserves all four fields.
6. **Backward compat tests**: existing example rule files in `examples/` continue to parse and behave identically without modification.

## Versioning

This is a 0.7.0 minor bump. The new fields are additive. The new DSL annotation is additive (no token previously starting with `{` was accepted in the header position). No removals or behavior changes for existing rules.

CHANGELOG `[0.7.0]` `Added` section describes:
- Four new `RuleMetadata` fields.
- DSL `{category=X}` annotation.
- Examples validation at load time.
- `validate_examples` kwarg, `engine.validate_examples()` method, `load_metadata_json()` method.
- `ExampleValidationError` exception.

## Open questions for follow-up specs

- **MCP surface**: how the new metadata is exposed via MCP tools. The natural shape is a `get_rule(name)` tool returning the full JSON, plus `find_rules_by_category(cat)`, plus the rule-application trace from the hook system. Spec to follow.
- **Negative examples**: a future extension could allow `{"in": ..., "should_not_match": true}` to assert a rule does *not* fire on a given input.
- **Per-example descriptions**: each example could carry an optional `"note"` field explaining what the case demonstrates ("this case shows the rule with an int constant"). Defer.
