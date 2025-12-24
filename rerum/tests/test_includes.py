"""Tests for :include directive in DSL files."""

import pytest
from pathlib import Path
from rerum import RuleEngine, E
from rerum.engine import load_rules_from_dsl, load_rules_from_file


class TestIncludeDirective:
    """Tests for :include directive in DSL files."""

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create a temporary directory for test files."""
        return tmp_path

    def test_include_basic(self, temp_dir):
        """Basic include of another rules file."""
        # Create included file
        (temp_dir / "base.rules").write_text('''
            @add-zero: (+ ?x 0) => :x
            @mul-one: (* ?x 1) => :x
        ''')

        # Create main file that includes base.rules
        (temp_dir / "main.rules").write_text('''
            :include base.rules

            @mul-zero: (* ?x 0) => 0
        ''')

        engine = RuleEngine.from_file(temp_dir / "main.rules")

        assert len(engine) == 3
        assert "add-zero" in engine
        assert "mul-one" in engine
        assert "mul-zero" in engine

    def test_include_nested(self, temp_dir):
        """Nested includes (file includes file that includes file)."""
        # Level 2
        (temp_dir / "level2.rules").write_text('''
            @level2-rule: (level2 ?x) => :x
        ''')

        # Level 1 includes level 2
        (temp_dir / "level1.rules").write_text('''
            :include level2.rules
            @level1-rule: (level1 ?x) => :x
        ''')

        # Main includes level 1
        (temp_dir / "main.rules").write_text('''
            :include level1.rules
            @main-rule: (main ?x) => :x
        ''')

        engine = RuleEngine.from_file(temp_dir / "main.rules")

        assert len(engine) == 3
        assert "level2-rule" in engine
        assert "level1-rule" in engine
        assert "main-rule" in engine

    def test_include_in_subdirectory(self, temp_dir):
        """Include file from subdirectory."""
        # Create subdirectory
        subdir = temp_dir / "subdir"
        subdir.mkdir()

        # Create file in subdirectory
        (subdir / "sub.rules").write_text('''
            @sub-rule: (sub ?x) => :x
        ''')

        # Main file includes from subdirectory
        (temp_dir / "main.rules").write_text('''
            :include subdir/sub.rules
            @main-rule: (main ?x) => :x
        ''')

        engine = RuleEngine.from_file(temp_dir / "main.rules")

        assert len(engine) == 2
        assert "sub-rule" in engine
        assert "main-rule" in engine

    def test_include_multiple(self, temp_dir):
        """Multiple includes in same file."""
        (temp_dir / "algebra.rules").write_text('''
            @add-zero: (+ ?x 0) => :x
        ''')

        (temp_dir / "calculus.rules").write_text('''
            @dd-const: (dd ?c:const ?v) => 0
        ''')

        (temp_dir / "main.rules").write_text('''
            :include algebra.rules
            :include calculus.rules
            @custom: (custom ?x) => :x
        ''')

        engine = RuleEngine.from_file(temp_dir / "main.rules")

        assert len(engine) == 3
        assert "add-zero" in engine
        assert "dd-const" in engine
        assert "custom" in engine

    def test_include_with_groups(self, temp_dir):
        """Include applies current group to included rules."""
        (temp_dir / "base.rules").write_text('''
            @ungrouped-rule: (f ?x) => :x
        ''')

        (temp_dir / "main.rules").write_text('''
            [mygroup]
            :include base.rules
            @grouped-rule: (g ?x) => :x
        ''')

        engine = RuleEngine.from_file(temp_dir / "main.rules")

        # ungrouped-rule should inherit the current group
        _, meta1 = engine["ungrouped-rule"]
        assert "mygroup" in meta1.tags

        _, meta2 = engine["grouped-rule"]
        assert "mygroup" in meta2.tags

    def test_include_preserves_existing_groups(self, temp_dir):
        """Include doesn't override existing groups in included file."""
        (temp_dir / "base.rules").write_text('''
            [original-group]
            @has-group: (f ?x) => :x
        ''')

        (temp_dir / "main.rules").write_text('''
            [override-attempt]
            :include base.rules
        ''')

        engine = RuleEngine.from_file(temp_dir / "main.rules")

        _, meta = engine["has-group"]
        # Should keep original group, not be overridden
        assert "original-group" in meta.tags

    def test_include_circular_detection(self, temp_dir):
        """Circular includes are detected and raise error."""
        # a.rules includes b.rules
        (temp_dir / "a.rules").write_text('''
            :include b.rules
        ''')

        # b.rules includes a.rules (circular!)
        (temp_dir / "b.rules").write_text('''
            :include a.rules
        ''')

        with pytest.raises(ValueError, match="Circular include"):
            RuleEngine.from_file(temp_dir / "a.rules")

    def test_include_self_detection(self, temp_dir):
        """Self-include is detected and raises error."""
        (temp_dir / "self.rules").write_text('''
            @rule: (f ?x) => :x
            :include self.rules
        ''')

        with pytest.raises(ValueError, match="Circular include"):
            RuleEngine.from_file(temp_dir / "self.rules")

    def test_include_file_not_found(self, temp_dir):
        """Missing include file raises FileNotFoundError."""
        (temp_dir / "main.rules").write_text('''
            :include nonexistent.rules
        ''')

        with pytest.raises(FileNotFoundError, match="Include file not found"):
            RuleEngine.from_file(temp_dir / "main.rules")

    def test_include_empty_path(self, temp_dir):
        """Empty include path is ignored."""
        (temp_dir / "main.rules").write_text('''
            :include
            @rule: (f ?x) => :x
        ''')

        engine = RuleEngine.from_file(temp_dir / "main.rules")
        assert len(engine) == 1

    def test_include_json_file(self, temp_dir):
        """Include JSON rules file."""
        import json
        (temp_dir / "base.json").write_text(json.dumps({
            "rules": [
                {"name": "json-rule", "pattern": ["f", ["?", "x"]], "skeleton": [":", "x"]}
            ]
        }))

        (temp_dir / "main.rules").write_text('''
            :include base.json
            @dsl-rule: (g ?x) => :x
        ''')

        engine = RuleEngine.from_file(temp_dir / "main.rules")

        assert len(engine) == 2
        assert "json-rule" in engine
        assert "dsl-rule" in engine

    def test_include_functional_rules(self, temp_dir):
        """Included rules work correctly for rewriting."""
        (temp_dir / "base.rules").write_text('''
            @add-zero: (+ ?x 0) => :x
            @mul-one: (* ?x 1) => :x
        ''')

        (temp_dir / "main.rules").write_text('''
            :include base.rules
        ''')

        engine = RuleEngine.from_file(temp_dir / "main.rules")

        assert engine(E("(+ x 0)")) == "x"
        assert engine(E("(* y 1)")) == "y"

    def test_load_dsl_without_base_path(self):
        """load_rules_from_dsl works without base_path (no includes)."""
        text = '''
            @rule1: (f ?x) => :x
            @rule2: (g ?x) => :x
        '''

        rules = load_rules_from_dsl(text)
        assert len(rules) == 2

    def test_load_dsl_with_base_path(self, temp_dir):
        """load_rules_from_dsl uses base_path for includes."""
        (temp_dir / "included.rules").write_text('''
            @included: (inc ?x) => :x
        ''')

        text = '''
            :include included.rules
            @main: (main ?x) => :x
        '''

        rules = load_rules_from_dsl(text, base_path=temp_dir)
        assert len(rules) == 2
