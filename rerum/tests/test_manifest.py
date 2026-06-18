"""Tests for the rule-set manifest: self-describing rule files.

A manifest is a DSL file carrying ``:``-directives that declare a domain's
loading contract (prelude bundles, custom fold ops, theory, metadata
sidecar, driver+goal hints). ``RuleEngine.from_manifest`` assembles the
whole domain from one file and FAILS LOUD when a required fold op is
missing. Plain ``load_file`` parses+stores the manifest but applies nothing.
"""

from pathlib import Path

import pytest

from rerum.engine import RuleEngine
from rerum.manifest import RuleSetManifest, parse_manifest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


class TestParseManifest:
    def test_empty_file_is_empty_manifest(self):
        m = parse_manifest("@r: (f ?x) => :x\n[group]\n:include other.rules")
        assert m.is_empty
        assert m.requires == ()
        assert m.theory is None and m.driver is None and m.goal is None

    def test_all_directives_parsed(self):
        text = (
            ":requires math predicate\n"
            ":requires-ops dd\n"
            ":theory arithmetic.theory.json\n"
            ":metadata differentiation.metadata.json\n"
            ":driver simplify\n"
            ':goal {"op_free": ["int"]}\n'
            ":include differentiation.rules\n"
        )
        m = parse_manifest(text)
        assert m.requires == ("math", "predicate")
        assert m.requires_ops == ("dd",)
        assert m.theory == "arithmetic.theory.json"
        assert m.metadata == "differentiation.metadata.json"
        assert m.driver == "simplify"
        assert m.goal == {"op_free": ["int"]}
        assert not m.is_empty

    def test_requires_accumulates_across_lines(self):
        m = parse_manifest(":requires math\n:requires predicate\n")
        assert m.requires == ("math", "predicate")

    def test_unknown_bundle_raises(self):
        with pytest.raises(ValueError, match="unknown.*bundle|frobnicate"):
            parse_manifest(":requires frobnicate\n")

    def test_unknown_driver_raises(self):
        with pytest.raises(ValueError, match="driver"):
            parse_manifest(":driver teleport\n")

    def test_bad_goal_json_raises(self):
        with pytest.raises(ValueError, match="goal"):
            parse_manifest(":goal not json\n")

    def test_unknown_directive_raises(self):
        # A typo'd directive must not be silently ignored.
        with pytest.raises(ValueError, match="directive|requies"):
            parse_manifest(":requies math\n")


class TestWithTheory:
    def test_with_theory_sets_and_is_chainable(self):
        from rerum.normalize import Theory
        theory = Theory.from_json('{"+": {"ac": true, "identity": 0}}')
        eng = RuleEngine().with_theory(theory)
        assert eng.has_theory() is True
        assert eng._theory is theory


class TestMissingFoldOps:
    def test_reports_skeleton_and_guard_ops(self):
        # No prelude installed: every (! op ...) head is missing.
        eng = RuleEngine.from_dsl(
            "@r: (f ?x) => (! foo :x) when (! bar :x)")
        missing = set(eng.missing_fold_ops())
        assert {"foo", "bar"} <= missing

    def test_empty_when_prelude_covers_ops(self):
        from rerum.rewriter import ARITHMETIC_PRELUDE
        eng = RuleEngine(fold_funcs=ARITHMETIC_PRELUDE)
        eng.load_dsl("@r: (g ?a ?b) => (! + :a :b)", validate_examples=False)
        assert eng.missing_fold_ops() == []


class TestFromManifestExamples:
    def test_differentiation_manifest_assembles_and_differentiates(self):
        eng = RuleEngine.from_manifest(EXAMPLES / "differentiation.manifest")
        from rerum.engine import parse_sexpr, format_sexpr
        from rerum.normalize import normalize
        out = normalize(eng(parse_sexpr("(dd (^ x 3) x)")), eng._theory)
        assert out == ["*", 3, ["^", "x", 2]]
        # The declared driver/goal hints are stored as data.
        assert eng.manifest.driver == "simplify"

    def test_boolean_manifest_no_prelude(self):
        eng = RuleEngine.from_manifest(EXAMPLES / "boolean.manifest")
        from rerum.engine import parse_sexpr
        assert eng.has_fold_funcs() is False
        eng.disable_group("equivalences")
        assert eng(parse_sexpr("(and a (or a b))")) == "a"


class TestFromManifestAudit:
    def _write(self, tmp_path, body, manifest_extra=""):
        (tmp_path / "r.rules").write_text(body)
        man = tmp_path / "d.manifest"
        man.write_text(f":requires math\n{manifest_extra}:include r.rules\n")
        return man

    def test_missing_skeleton_op_fails_loud(self, tmp_path):
        man = self._write(tmp_path, "@r: (f ?x) => (! frobnicate :x)\n")
        with pytest.raises(ValueError, match="frobnicate"):
            RuleEngine.from_manifest(man)

    def test_missing_guard_op_fails_at_load(self, tmp_path):
        man = self._write(
            tmp_path, "@r: (f ?x) => :x when (! frobnicate :x)\n")
        with pytest.raises(ValueError, match="frobnicate"):
            RuleEngine.from_manifest(man)

    def test_requires_ops_absent_fails_even_if_unused(self, tmp_path):
        man = self._write(tmp_path, "@r: (f ?x) => :x\n",
                          manifest_extra=":requires-ops frobnicate\n")
        with pytest.raises(ValueError, match="frobnicate"):
            RuleEngine.from_manifest(man)

    def test_present_ops_assemble_clean(self, tmp_path):
        man = self._write(tmp_path, "@r: (g ?a ?b) => (! + :a :b)\n")
        eng = RuleEngine.from_manifest(man)  # math provides +
        assert eng(["g", 2, 3]) == 5


class TestLoadFileBoundary:
    def test_load_file_stores_manifest_but_installs_nothing(self, tmp_path):
        (tmp_path / "r.rules").write_text("@r: (f ?x) => :x\n")
        man = tmp_path / "d.manifest"
        man.write_text(":requires math\n:driver solve\n:include r.rules\n")
        eng = RuleEngine().load_file(man)
        # Rules loaded, but the prelude was NOT installed (BC: load_file
        # never silently mutates the engine's prelude).
        assert len(eng) == 1
        assert eng.has_fold_funcs() is False
        # The declared contract is inspectable.
        assert eng.manifest is not None
        assert eng.manifest.requires == ("math",)
        assert eng.manifest.driver == "solve"

    def test_directive_free_file_has_empty_manifest(self, tmp_path):
        f = tmp_path / "r.rules"
        f.write_text("@r: (f ?x) => :x\n")
        eng = RuleEngine().load_file(f)
        assert eng.manifest is not None and eng.manifest.is_empty


class TestManifestReviewFixes:
    """Fixes from the manifest adversarial review (PASS_WITH_NOTES)."""

    def test_embedded_rules_examples_are_validated(self, tmp_path):
        # ISSUE 1: from_manifest validated only SIDECAR examples; a WRONG
        # example embedded in the :include'd rules JSON loaded silently.
        # Now from_manifest validates embedded examples too (parity with a
        # plain load_file(validate_examples=True)).
        import json
        (tmp_path / "r.json").write_text(json.dumps({"rules": [{
            "name": "inc", "pattern": ["f", ["?", "x"]],
            "skeleton": ["!", "+", [":", "x"], 1],
            "examples": [{"in": "(f 1)", "out": "999"}],  # wrong: (f 1)->2
        }]}))
        man = tmp_path / "d.manifest"
        man.write_text(":requires math\n:include r.json\n")
        with pytest.raises(Exception):  # ExampleValidationError
            RuleEngine.from_manifest(man)

    def test_load_file_lenient_on_stray_colon_line(self, tmp_path):
        # ISSUE 2: load_file now parses a manifest on every non-.json file;
        # it must NOT regress to raising on a stray ':'-line (a note/typo/
        # future directive) that the old loader silently ignored.
        f = tmp_path / "old.rules"
        f.write_text(":see docs for details\n@r: (f ?x) => :x\n")
        eng = RuleEngine().load_file(f)  # must not raise
        assert len(eng) == 1
        assert eng.manifest is None  # malformed manifest not recorded

    def test_from_manifest_still_strict_on_typo(self, tmp_path):
        # The strict path (from_manifest) still catches the typo load_file
        # now tolerates.
        (tmp_path / "r.rules").write_text("@r: (f ?x) => :x\n")
        man = tmp_path / "d.manifest"
        man.write_text(":requies math\n:include r.rules\n")  # typo
        with pytest.raises(ValueError, match="directive|requies"):
            RuleEngine.from_manifest(man)
