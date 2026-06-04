"""Regression tests for v0.7 metadata-layer fixes.

Covers the bugs fixed alongside the metadata-layer cleanup:
  B2  to_dsl emitted ``category`` unquoted, breaking from_dsl(to_dsl()) roundtrip
      when the value contained ',', '}', or surrounding whitespace.
  B3  to_dict/to_json silently dropped ``extra`` (resolver provenance), so the
      "lossless" JSON roundtrip lost unknown fields.
  B4  load_rules_from_json aliased the SAME ``examples`` list across both
      bidirectional halves; mutating one mutated the other.
  B5  load_metadata_json's sidecar setattr bypassed the None->[] normalization,
      letting ``examples`` become None and breaking the "always a list" invariant.

Also pins behaviour that the cleanup refactor must preserve (install loop
extraction, signature-derived known-field sets, add_rule priority ordering).
"""

import json

import pytest

from rerum.engine import (
    RuleEngine,
    RuleMetadata,
    load_rules_from_json,
)


class TestToDslCategoryRoundtrip:
    """B2: category values must survive from_dsl(to_dsl())."""

    @pytest.mark.parametrize("category", [
        "identity",            # bare token, must stay loss-free (and ideally unquoted)
        "is identity",         # internal whitespace
        "a, b",                # comma — used to split into a bogus second pair
        "x}y",                 # closing brace — used to terminate the block early
        "{weird}",             # both braces
        "  pad  ",             # surrounding whitespace — would be stripped if bare
        'has"dq',              # embedded double quote
    ])
    def test_category_roundtrips_through_dsl(self, category):
        eng = RuleEngine()
        eng.add_rule(["+", ["?", "x"], 0], [":", "x"], name="r", category=category)
        dsl = eng.to_dsl()
        reloaded = RuleEngine().load_dsl(dsl)
        assert reloaded._metadata[0].category == category, (
            f"roundtrip lost category through DSL: emitted {dsl!r}"
        )

    def test_bare_category_is_not_over_quoted(self):
        # A simple identifier should still emit the bare {category=identity} form.
        eng = RuleEngine()
        eng.add_rule(["+", ["?", "x"], 0], [":", "x"], name="r", category="identity")
        dsl = eng.to_dsl()
        assert "{category=identity}" in dsl


class TestToDictExtraRoundtrip:
    """B3: extra (unknown/provenance fields) must survive to_dict -> reload."""

    def test_extra_survives_json_roundtrip_unidirectional(self):
        text = json.dumps({"rules": [{
            "name": "r", "pattern": ["+", ["?", "x"], 0], "skeleton": [":", "x"],
            "provenance": "llm", "confidence": 0.9,
        }]})
        eng = RuleEngine().load_rules_from_json(text)
        assert eng._metadata[0].extra == {"provenance": "llm", "confidence": 0.9}

        d = eng.to_dict()
        rd = d["rules"][0]
        assert rd.get("provenance") == "llm"
        assert rd.get("confidence") == 0.9

        eng2 = RuleEngine().load_rules_from_json(json.dumps(d))
        assert eng2._metadata[0].extra == {"provenance": "llm", "confidence": 0.9}

    def test_extra_survives_json_roundtrip_bidirectional(self):
        text = json.dumps({"rules": [{
            "name": "comm", "bidirectional": True,
            "pattern": ["+", ["?", "x"], ["?", "y"]],
            "skeleton": ["+", [":", "y"], [":", "x"]],
            "provenance": "llm",
        }]})
        eng = RuleEngine().load_rules_from_json(text)
        d = eng.to_dict()
        assert d["rules"][0].get("provenance") == "llm"
        eng2 = RuleEngine().load_rules_from_json(json.dumps(d))
        # both halves carry the provenance after reload
        assert eng2._metadata[0].extra.get("provenance") == "llm"
        assert eng2._metadata[1].extra.get("provenance") == "llm"


class TestBidirectionalExamplesIndependent:
    """B4: parallel to test_bidirectional_extras_are_independent_per_half."""

    def test_examples_lists_are_independent_per_half(self):
        text = json.dumps({"rules": [{
            "name": "comm", "bidirectional": True,
            "pattern": ["+", ["?", "x"], ["?", "y"]],
            "skeleton": ["+", [":", "y"], [":", "x"]],
            "examples": [{"in": "(+ 1 2)", "out": "(+ 2 1)"}],
        }]})
        pairs = load_rules_from_json(text)
        fwd_meta, rev_meta = pairs[0][0], pairs[1][0]
        assert fwd_meta.examples is not rev_meta.examples
        fwd_meta.examples.append({"in": "(+ 3 4)", "out": "(+ 4 3)"})
        assert len(rev_meta.examples) == 1  # rev unaffected


class TestSidecarExamplesNormalization:
    """B5: sidecar examples=null must not break the always-a-list invariant."""

    def test_sidecar_examples_null_normalizes_to_empty_list(self):
        eng = RuleEngine().load_dsl("@r: (+ ?x 0) => :x")
        eng.load_metadata_json(json.dumps({"r": {"examples": None}}),
                               validate_examples=False)
        assert eng._metadata[0].examples == []


class TestCleanupBehaviourPreserved:
    """Pin behaviour the refactor must not regress."""

    def test_add_rule_priority_reorders_firing(self):
        # add_rule must keep rules priority-ordered so the higher-priority rule fires.
        eng = RuleEngine()
        eng.add_rule(["+", ["?", "x"], 0], ["lo"], name="lo", priority=1)
        eng.add_rule(["+", ["?", "x"], 0], ["hi"], name="hi", priority=10)
        out, meta = eng.apply_once(["+", "a", 0])
        assert meta.name == "hi"

    def test_unknown_json_field_routes_to_extra(self):
        text = json.dumps({"rules": [{
            "name": "r", "pattern": ["+", ["?", "x"], 0], "skeleton": [":", "x"],
            "mystery": 42,
        }]})
        eng = RuleEngine().load_rules_from_json(text)
        assert eng._metadata[0].extra == {"mystery": 42}

    def test_unknown_sidecar_field_routes_to_extra(self):
        eng = RuleEngine().load_dsl("@r: (+ ?x 0) => :x")
        eng.load_metadata_json(json.dumps({"r": {"mystery": 42}}))
        assert eng._metadata[0].extra == {"mystery": 42}

    def test_load_dsl_still_validates_examples_by_default(self):
        # A wrong example should still raise at load.
        from rerum.engine import ExampleValidationError
        text = json.dumps({"rules": [{
            "name": "bad", "pattern": ["+", ["?", "x"], 0], "skeleton": [":", "x"],
            "examples": [{"in": "(+ 5 0)", "out": "6"}],  # wrong: should be 5
        }]})
        with pytest.raises(ExampleValidationError):
            RuleEngine().load_rules_from_json(text)


class TestDslHeaderEdges:
    """Cover the anonymous-rule and both-quote header branches of the shared
    _format_dsl_header / _format_annotation_value helpers."""

    def test_anonymous_rule_with_category_roundtrips(self):
        eng = RuleEngine()
        # No name, but a category: exercises the nameless `{category=...}:` arm.
        eng.add_rule(["+", ["?", "x"], 0], [":", "x"], category="identity")
        dsl = eng.to_dsl()
        assert "{category=identity}:" in dsl
        reloaded = RuleEngine().load_dsl(dsl)
        assert reloaded._metadata[0].category == "identity"
        assert reloaded._metadata[0].name is None

    def test_category_with_both_quote_chars_falls_back_to_double(self):
        from rerum.engine import _format_annotation_value
        # Both quote chars present and the grammar has no escapes (documented
        # lossy): we fall back to a double-quoted rendering rather than crash.
        out = _format_annotation_value("a\"b'c")
        assert out == "\"a\"b'c\""


# ---------------------------------------------------------------------------
# Deferred follow-ups B1/B6/B7/B8: load_metadata_json sidecar semantics.
# ---------------------------------------------------------------------------

class TestSidecarPriorityFillable:
    """B1: 'unset' means 'equals the constructor default', so falsy defaults
    (priority=0) are fillable instead of spuriously conflicting, and a filled
    priority re-sorts the engine."""

    def test_sidecar_fills_priority_and_reorders(self):
        eng = RuleEngine()
        eng.add_rule(["+", ["?", "x"], 0], ["lo"], name="lo", priority=1)
        eng.add_rule(["+", ["?", "x"], 0], ["hi"], name="hi")  # default priority 0
        assert eng.apply_once(["+", "a", 0])[1].name == "lo"
        eng.load_metadata_json(json.dumps({"hi": {"priority": 10}}))
        assert eng._metadata[eng._rule_names["hi"]].priority == 10
        assert eng.apply_once(["+", "a", 0])[1].name == "hi"  # re-sorted

    def test_sidecar_priority_same_as_default_is_noop(self):
        eng = RuleEngine().load_dsl("@r: (+ ?x 0) => :x")  # priority 0
        eng.load_metadata_json(json.dumps({"r": {"priority": 0}}))  # no conflict
        assert eng._metadata[0].priority == 0

    def test_sidecar_priority_conflict_on_explicit_value(self):
        eng = RuleEngine().load_dsl("@r[5]: (+ ?x 0) => :x")  # explicit priority 5
        with pytest.raises(ValueError, match="conflict"):
            eng.load_metadata_json(json.dumps({"r": {"priority": 7}}))


class TestSidecarExtraConflict:
    """B6: extra (unknown) fields are conflict-checked like known fields, not
    silently overwritten on a second load."""

    def test_extra_same_value_is_ok(self):
        eng = RuleEngine().load_dsl("@r: (+ ?x 0) => :x")
        eng.load_metadata_json(json.dumps({"r": {"prov": "a"}}))
        eng.load_metadata_json(json.dumps({"r": {"prov": "a"}}))  # same, harmless
        assert eng._metadata[0].extra == {"prov": "a"}

    def test_extra_conflict_raises(self):
        eng = RuleEngine().load_dsl("@r: (+ ?x 0) => :x")
        eng.load_metadata_json(json.dumps({"r": {"prov": "a"}}))
        with pytest.raises(ValueError, match="conflict"):
            eng.load_metadata_json(json.dumps({"r": {"prov": "b"}}))


class TestSidecarProtectedFields:
    """B7: a metadata sidecar may describe a rule but not redefine its
    identity/structure or break storage invariants."""

    @pytest.mark.parametrize("field,value", [
        ("name", "renamed"),
        ("bidirectional", True),
        ("direction", "fwd"),
    ])
    def test_structural_field_rejected(self, field, value):
        eng = RuleEngine().load_dsl("@r: (+ ?x 0) => :x")
        with pytest.raises(ValueError):
            eng.load_metadata_json(json.dumps({"r": {field: value}}))

    def test_priority_on_bidirectional_half_rejected(self):
        # Bumping one half's priority would split the stored -fwd/-rev pair.
        text = json.dumps({"rules": [{
            "name": "comm", "bidirectional": True,
            "pattern": ["+", ["?", "x"], ["?", "y"]],
            "skeleton": ["+", [":", "y"], [":", "x"]],
        }]})
        eng = RuleEngine().load_rules_from_json(text)
        with pytest.raises(ValueError):
            eng.load_metadata_json(json.dumps({"comm-fwd": {"priority": 9}}))

    def test_label_on_unidirectional_rejected(self):
        # Mirrors load_rules_from_json's rejection of labels on => rules.
        eng = RuleEngine().load_dsl("@r: (+ ?x 0) => :x")
        with pytest.raises(ValueError):
            eng.load_metadata_json(json.dumps({"r": {"fwd_label": "x"}}))


class TestValidationUsesResolvers:
    """B8: example validation threads the engine's undefined-op/fold-error
    resolvers so it matches live rewriting behaviour."""

    def test_validation_uses_undefined_op_resolver(self):
        from rerum.hooks import Resolution
        eng = RuleEngine()

        @eng.on_undefined_op
        def resolver(op, args, ctx):
            if op == "my-op":
                return Resolution(value=args[0])  # identity
            return None

        # Skeleton uses my-op (absent from the empty prelude); the resolver
        # supplies it. Without threading the resolver into validation, this
        # example would fail to validate at load.
        text = json.dumps({"rules": [{
            "name": "r", "pattern": ["foo", ["?", "x"]],
            "skeleton": ["!", "my-op", [":", "x"]],
            "examples": [{"in": "(foo 5)", "out": "5"}],
        }]})
        eng.load_rules_from_json(text)  # must not raise
        assert eng._metadata[eng._rule_names["r"]].examples
