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

    def test_examples_defaults_to_empty_list(self):
        m = RuleMetadata()
        assert m.examples == []

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

    def test_missing_new_fields_default_to_none_or_empty(self):
        from rerum.engine import load_rules_from_json
        text = '{"rules": [{"name": "r1",' \
               ' "pattern": ["a", ["?", "x"]], "skeleton": [":", "x"]}]}'
        rules = load_rules_from_json(text)
        meta, _ = rules[0]
        assert meta.category is None
        assert meta.reasoning is None
        # examples normalizes to [] (M1 fix); some callers may still see None
        # if not normalized. Accept either.
        assert meta.examples in (None, [])
        assert meta.fwd_label is None
        assert meta.rev_label is None

    def test_unknown_fields_preserved_in_extra(self):
        from rerum.engine import load_rules_from_json
        text = '{"rules": [{"name": "r1", "weird_field": "value",' \
               ' "pattern": ["a", ["?", "x"]], "skeleton": [":", "x"]}]}'
        rules = load_rules_from_json(text)
        meta, _ = rules[0]
        assert meta.extra.get("weird_field") == "value"

    def test_bidirectional_propagates_category(self):
        from rerum.engine import load_rules_from_json
        text = '{"rules": [{"name": "commute", "bidirectional": true,' \
               ' "category": "commutativity",' \
               ' "pattern": ["+", ["?", "x"], ["?", "y"]],' \
               ' "skeleton": ["+", [":", "y"], [":", "x"]]}]}'
        rules = load_rules_from_json(text)
        # Both fwd and rev carry the category.
        for meta, _ in rules:
            assert meta.category == "commutativity"

    def test_load_unidirectional_with_label_raises_when_bidirectional_absent(self):
        # When bidirectional is omitted entirely, label fields are still rejected.
        from rerum.engine import load_rules_from_json
        text = '{"rules": [{"name": "r1", "fwd_label": "x",' \
               ' "pattern": ["a", ["?", "x"]], "skeleton": [":", "x"]}]}'
        with pytest.raises(ValueError, match="fwd_label"):
            load_rules_from_json(text)

    def test_bidirectional_unknown_fields_preserved_in_extra(self):
        from rerum.engine import load_rules_from_json
        text = '{"rules": [{"name": "commute", "bidirectional": true,' \
               ' "weird_field": "value",' \
               ' "pattern": ["+", ["?", "x"], ["?", "y"]],' \
               ' "skeleton": ["+", [":", "y"], [":", "x"]]}]}'
        rules = load_rules_from_json(text)
        # Both fwd and rev should carry the extra field.
        for meta, _ in rules:
            assert meta.extra.get("weird_field") == "value"

    def test_bidirectional_extras_are_independent_per_half(self):
        # Mutating one half's extra dict should not affect the other.
        from rerum.engine import load_rules_from_json
        text = '{"rules": [{"name": "commute", "bidirectional": true,' \
               ' "weird_field": "value",' \
               ' "pattern": ["+", ["?", "x"], ["?", "y"]],' \
               ' "skeleton": ["+", [":", "y"], [":", "x"]]}]}'
        rules = load_rules_from_json(text)
        rules[0][0].extra["new"] = "fwd-only"
        assert "new" not in rules[1][0].extra


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
        engine = RuleEngine()
        engine.add_rule(
            pattern=["a", ["?", "x"]],
            skeleton=[":", "x"],
            name="r1",
        )
        # Manually set the new fields (add_rule kwargs come in M10).
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
        # examples is normalized to [] in M1; either omit or empty list is fine.
        assert rule.get("examples", []) == []

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
