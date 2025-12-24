"""Tests for new RuleEngine methods: with_prelude, match, apply_once, rules_matching."""

import pytest
from rerum import (
    RuleEngine, E, Bindings, NoMatch,
    ARITHMETIC_PRELUDE, MATH_PRELUDE,
)


class TestWithPrelude:
    """Tests for RuleEngine.with_prelude()."""

    def test_with_prelude_fluent(self):
        """with_prelude() returns self for chaining."""
        engine = RuleEngine()
        result = engine.with_prelude(ARITHMETIC_PRELUDE)
        assert result is engine

    def test_with_prelude_enables_folding(self):
        """with_prelude() enables constant folding."""
        engine = (RuleEngine()
            .with_prelude(ARITHMETIC_PRELUDE)
            .load_dsl("@fold: (+ ?a:const ?b:const) => (! + :a :b)"))

        result = engine(E("(+ 1 2)"))
        assert result == 3

    def test_with_prelude_chain_with_load(self):
        """with_prelude() chains naturally with load methods."""
        engine = (RuleEngine()
            .with_prelude(ARITHMETIC_PRELUDE)
            .load_dsl("@add-zero: (+ ?x 0) => :x"))

        assert engine(E("(+ y 0)")) == "y"

    def test_with_prelude_after_load(self):
        """with_prelude() can be called after loading rules."""
        engine = (RuleEngine()
            .load_dsl("@fold: (+ ?a:const ?b:const) => (! + :a :b)")
            .with_prelude(ARITHMETIC_PRELUDE))

        result = engine(E("(+ 1 2)"))
        assert result == 3

    def test_without_prelude_no_folding(self):
        """Without prelude, (! +) doesn't evaluate to a number."""
        # Use a rule that doesn't match its own output to avoid recursion
        engine = RuleEngine().load_dsl(
            "@fold: (fold-add ?a:const ?b:const) => (! + :a :b)")

        # Without prelude, (! + 1 2) becomes ["+", 1, 2] (not evaluated)
        result = engine(E("(fold-add 1 2)"))
        assert result == ["+", 1, 2]


class TestMatch:
    """Tests for RuleEngine.match()."""

    def test_match_success(self):
        """match() returns Bindings on success."""
        engine = RuleEngine()
        bindings = engine.match("(+ ?a ?b)", E("(+ x 1)"))

        assert bindings
        assert isinstance(bindings, Bindings)
        assert bindings["a"] == "x"
        assert bindings["b"] == 1

    def test_match_failure(self):
        """match() returns NoMatch on failure."""
        engine = RuleEngine()
        result = engine.match("(+ ?a ?b)", E("(* x 1)"))

        assert result is NoMatch
        assert not result

    def test_match_string_pattern(self):
        """match() accepts string patterns."""
        engine = RuleEngine()
        bindings = engine.match("(dd ?f ?x:var)", E("(dd (^ x 2) x)"))

        assert bindings
        assert bindings["f"] == ["^", "x", 2]
        assert bindings["x"] == "x"

    def test_match_list_pattern(self):
        """match() accepts list patterns."""
        engine = RuleEngine()
        pattern = ["+", ["?", "a"], ["?", "b"]]
        bindings = engine.match(pattern, ["+", "x", 1])

        assert bindings
        assert bindings["a"] == "x"
        assert bindings["b"] == 1

    def test_match_walrus_pattern(self):
        """match() works with walrus operator."""
        engine = RuleEngine()
        expr = E("(+ x y)")

        if bindings := engine.match("(+ ?a ?b)", expr):
            assert bindings["a"] == "x"
            assert bindings["b"] == "y"
        else:
            pytest.fail("Should have matched")

    def test_match_with_const_constraint(self):
        """match() respects type constraints."""
        engine = RuleEngine()

        # Should match - 5 is a constant
        assert engine.match("(f ?n:const)", E("(f 5)"))

        # Should not match - x is a variable
        assert engine.match("(f ?n:const)", E("(f x)")) is NoMatch


class TestApplyOnce:
    """Tests for RuleEngine.apply_once()."""

    def test_apply_once_success(self):
        """apply_once() applies first matching rule."""
        engine = RuleEngine.from_dsl('''
            @add-zero: (+ ?x 0) => :x
            @mul-one: (* ?x 1) => :x
        ''')

        result, meta = engine.apply_once(E("(+ y 0)"))
        assert result == "y"
        assert meta.name == "add-zero"

    def test_apply_once_no_match(self):
        """apply_once() returns original when no rule matches."""
        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")

        expr = E("(* y 1)")
        result, meta = engine.apply_once(expr)
        assert result == expr
        assert meta is None

    def test_apply_once_first_rule(self):
        """apply_once() applies the first matching rule only."""
        engine = RuleEngine.from_dsl('''
            @rule1: (f ?x) => (g :x)
            @rule2: (f ?x) => (h :x)
        ''')

        result, meta = engine.apply_once(E("(f a)"))
        assert result == ["g", "a"]
        assert meta.name == "rule1"

    def test_apply_once_no_recursion(self):
        """apply_once() doesn't recurse into subexpressions."""
        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")

        # The inner (+ y 0) should NOT be simplified
        expr = E("(f (+ y 0))")
        result, meta = engine.apply_once(expr)
        assert result == expr  # unchanged
        assert meta is None

    def test_apply_once_with_prelude(self):
        """apply_once() uses prelude for skeleton evaluation."""
        engine = (RuleEngine()
            .with_prelude(ARITHMETIC_PRELUDE)
            .load_dsl("@fold: (+ ?a:const ?b:const) => (! + :a :b)"))

        result, meta = engine.apply_once(E("(+ 1 2)"))
        assert result == 3
        assert meta.name == "fold"


class TestRulesMatching:
    """Tests for RuleEngine.rules_matching()."""

    def test_rules_matching_single(self):
        """rules_matching() finds a single matching rule."""
        engine = RuleEngine.from_dsl('''
            @add-zero: (+ ?x 0) => :x
            @mul-one: (* ?x 1) => :x
        ''')

        matches = engine.rules_matching(E("(+ y 0)"))
        assert len(matches) == 1
        assert matches[0][0].name == "add-zero"
        assert matches[0][1]["x"] == "y"

    def test_rules_matching_multiple(self):
        """rules_matching() finds all matching rules."""
        engine = RuleEngine.from_dsl('''
            @rule1: (f ?x) => (g :x)
            @rule2: (f ?x) => (h :x)
            @rule3: (other ?x) => :x
        ''')

        matches = engine.rules_matching(E("(f a)"))
        assert len(matches) == 2
        names = {m[0].name for m in matches}
        assert names == {"rule1", "rule2"}

    def test_rules_matching_none(self):
        """rules_matching() returns empty list when nothing matches."""
        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")

        matches = engine.rules_matching(E("(* y 1)"))
        assert matches == []

    def test_rules_matching_bindings(self):
        """rules_matching() returns proper Bindings objects."""
        engine = RuleEngine.from_dsl("@rule: (+ ?a ?b) => :a")

        matches = engine.rules_matching(E("(+ x y)"))
        assert len(matches) == 1
        meta, bindings = matches[0]
        assert isinstance(bindings, Bindings)
        assert bindings["a"] == "x"
        assert bindings["b"] == "y"

    def test_rules_matching_debug_use(self):
        """rules_matching() is useful for debugging."""
        engine = RuleEngine.from_dsl('''
            @add-zero-r: (+ ?x 0) => :x
            @add-zero-l: (+ 0 ?x) => :x
        ''')

        # This expression should match add-zero-l
        expr = E("(+ 0 y)")
        matches = engine.rules_matching(expr)

        assert len(matches) == 1
        assert matches[0][0].name == "add-zero-l"


class TestIntegration:
    """Integration tests combining multiple new features."""

    def test_fluent_construction(self):
        """Full fluent construction with new methods."""
        engine = (RuleEngine()
            .with_prelude(ARITHMETIC_PRELUDE)
            .load_dsl('''
                @add-zero: (+ ?x 0) => :x
                @mul-one: (* ?x 1) => :x
                @fold-add: (+ ?a:const ?b:const) => (! + :a :b)
            '''))

        # Test match
        assert engine.match("(+ ?x 0)", E("(+ y 0)"))

        # Test apply_once
        result, _ = engine.apply_once(E("(+ 1 2)"))
        assert result == 3

        # Test full simplification
        expr = E("(+ (* x 1) 0)")
        assert engine(expr) == "x"

    def test_expression_builder_with_match(self):
        """E and match work together naturally."""
        engine = RuleEngine()

        x, y = E.vars("x", "y")
        expr = E.op("+", x, E.op("*", 2, y))

        if bindings := engine.match("(+ ?a (* ?n ?b))", expr):
            assert bindings["a"] == "x"
            assert bindings["n"] == 2
            assert bindings["b"] == "y"
        else:
            pytest.fail("Should have matched")


class TestExport:
    """Tests for RuleEngine export methods: to_dsl, to_json, to_dict."""

    def test_to_dsl_basic(self):
        """to_dsl() exports simple rules correctly."""
        engine = RuleEngine.from_dsl('''
            @add-zero: (+ ?x 0) => :x
            @mul-one: (* ?x 1) => :x
        ''')

        dsl = engine.to_dsl()
        assert "@add-zero: (+ ?x 0) => :x" in dsl
        assert "@mul-one: (* ?x 1) => :x" in dsl

    def test_to_dsl_with_priority(self):
        """to_dsl() includes priority when non-zero."""
        engine = RuleEngine.from_dsl('''
            @high[100]: (+ ?x 0) => :x
            @low[10]: (* ?x 1) => :x
        ''')

        dsl = engine.to_dsl()
        assert "@high[100]:" in dsl
        assert "@low[10]:" in dsl

    def test_to_dsl_with_condition(self):
        """to_dsl() includes when clause for conditional rules."""
        engine = RuleEngine.from_dsl('''
            @fold: (+ ?a ?b) => (! + :a :b) when (! and (! const? :a) (! const? :b))
        ''')

        dsl = engine.to_dsl()
        assert "when" in dsl
        assert "(! and (! const? :a) (! const? :b))" in dsl

    def test_to_dsl_with_groups(self):
        """to_dsl() organizes rules by groups."""
        engine = RuleEngine.from_dsl('''
            [algebra]
            @add-zero: (+ ?x 0) => :x

            [calculus]
            @dd-const: (dd ?c:const ?v) => 0
        ''')

        dsl = engine.to_dsl()
        assert "[algebra]" in dsl
        assert "[calculus]" in dsl

    def test_to_dsl_with_name(self):
        """to_dsl() includes name as comment header."""
        engine = RuleEngine.from_dsl("@rule: (f ?x) => :x")

        dsl = engine.to_dsl(name="My Ruleset")
        assert "# My Ruleset" in dsl

    def test_to_json_basic(self):
        """to_json() exports rules in JSON format."""
        import json
        engine = RuleEngine.from_dsl('''
            @add-zero: (+ ?x 0) => :x
        ''')

        json_str = engine.to_json()
        data = json.loads(json_str)

        assert "rules" in data
        assert len(data["rules"]) == 1
        assert data["rules"][0]["name"] == "add-zero"
        assert data["rules"][0]["pattern"] == ["+", ["?", "x"], 0]
        assert data["rules"][0]["skeleton"] == [":", "x"]

    def test_to_json_with_metadata(self):
        """to_json() includes name and description."""
        import json
        engine = RuleEngine.from_dsl("@rule: (f ?x) => :x")

        json_str = engine.to_json(name="TestRules", description="A test ruleset")
        data = json.loads(json_str)

        assert data["name"] == "TestRules"
        assert data["description"] == "A test ruleset"

    def test_to_json_with_priority_and_condition(self):
        """to_json() includes priority and condition."""
        import json
        engine = (RuleEngine()
            .with_prelude(ARITHMETIC_PRELUDE)
            .load_dsl('''
                @fold[100]: (+ ?a ?b) => (! + :a :b) when (! and (! const? :a) (! const? :b))
            '''))

        json_str = engine.to_json()
        data = json.loads(json_str)

        rule = data["rules"][0]
        assert rule["priority"] == 100
        assert "condition" in rule

    def test_to_json_compact(self):
        """to_json() supports compact output."""
        engine = RuleEngine.from_dsl("@rule: (f ?x) => :x")

        json_str = engine.to_json(indent=None)
        assert "\n" not in json_str  # No newlines in compact JSON

    def test_to_dict_basic(self):
        """to_dict() returns a dictionary."""
        engine = RuleEngine.from_dsl("@add-zero: (+ ?x 0) => :x")

        d = engine.to_dict()
        assert isinstance(d, dict)
        assert "rules" in d
        assert len(d["rules"]) == 1

    def test_to_dict_complete_fields(self):
        """to_dict() includes all rule fields."""
        engine = RuleEngine.from_dsl('''
            [mygroup]
            @myrule[50] "A description": (+ ?x 0) => :x when (! const? :x)
        ''')

        d = engine.to_dict()
        rule = d["rules"][0]

        assert rule["name"] == "myrule"
        assert rule["description"] == "A description"
        assert rule["priority"] == 50
        assert rule["tags"] == ["mygroup"]
        assert "condition" in rule

    def test_roundtrip_json(self):
        """Rules can be exported to JSON and reimported."""
        import json
        original = RuleEngine.from_dsl('''
            [algebra]
            @add-zero[100]: (+ ?x 0) => :x
            @mul-one[50]: (* ?x 1) => :x
        ''')

        # Export to JSON
        json_str = original.to_json()

        # Reimport
        restored = RuleEngine()
        data = json.loads(json_str)
        from rerum.engine import load_rules_from_json
        parsed = load_rules_from_json(json_str)
        for meta, rule in parsed:
            restored._rules.append(rule)
            restored._metadata.append(meta)
            if meta.name:
                restored._rule_names[meta.name] = len(restored._rules) - 1

        # Verify structure
        assert len(restored) == len(original)
        assert "add-zero" in restored
        assert "mul-one" in restored

        # Verify priority preserved
        _, meta1 = restored["add-zero"]
        assert meta1.priority == 100

        # Verify tags preserved
        assert "algebra" in meta1.tags

    def test_list_rules_format(self):
        """list_rules() returns DSL-formatted strings."""
        engine = RuleEngine.from_dsl('''
            @add-zero[100] "Adding zero": (+ ?x 0) => :x when (! expr? :x)
        ''')

        rules = engine.list_rules()
        assert len(rules) == 1
        assert "@add-zero[100]" in rules[0]
        assert '"Adding zero"' in rules[0]
        assert "when" in rules[0]
