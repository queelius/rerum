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
