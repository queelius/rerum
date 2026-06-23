"""Tests for MCP tool handlers and error mapping."""

import pytest

from rerum.mcp.errors import MCPToolError


class TestErrorMapping:
    def test_explicit_parse_error_code(self):
        from rerum.mcp.errors import MCPToolError
        err = MCPToolError("parse_error", "bad input", details={"input": "(a"})
        assert err.code == "parse_error"
        assert err.message == "bad input"
        assert err.details == {"input": "(a"}

    def test_to_dict_shape(self):
        from rerum.mcp.errors import MCPToolError
        err = MCPToolError("unknown_rule", "no rule named 'x'",
                           details={"name": "x", "available": ["a", "b"]})
        d = err.to_dict()
        assert d["error"]["code"] == "unknown_rule"
        assert d["error"]["message"] == "no rule named 'x'"
        assert d["error"]["details"] == {"name": "x", "available": ["a", "b"]}

    def test_validation_error_from_example_validation(self):
        from rerum.mcp.errors import map_exception
        from rerum.engine import ExampleValidationError

        exc = ExampleValidationError(
            "Rule 'x': pattern does not match",
            rule_name="x",
            example={"in": "(a 1)", "out": "1"},
        )
        err = map_exception(exc, context={"tool": "load_rules"})
        assert err["error"]["code"] == "validation_error"
        assert "Rule 'x'" in err["error"]["message"]
        assert err["error"]["details"]["rule_name"] == "x"

    def test_resolver_loop_error_mapping(self):
        from rerum.mcp.errors import map_exception
        from rerum.hooks import ResolverLoopError

        exc = ResolverLoopError("retry cap (100) exceeded")
        err = map_exception(exc, context={"tool": "solve_assisted"})
        assert err["error"]["code"] == "resolver_loop"

    def test_generic_value_error_is_internal(self):
        from rerum.mcp.errors import map_exception
        err = map_exception(ValueError("boom"), context={"tool": "simplify"})
        assert err["error"]["code"] == "internal_error"


class TestAuthoringTools:
    def test_load_rules_dsl(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_load_rules

        engine = RuleEngine()
        result = tool_load_rules(
            engine,
            text='@add-zero {category=identity}: (+ ?x 0) => :x',
            format="dsl",
        )
        assert result["ok"] is True
        assert result["rules_added"] == 1
        assert "add-zero" in engine

    def test_load_rules_auto_detect_json(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_load_rules

        engine = RuleEngine()
        text = ('{"rules": [{"name": "r1", "category": "identity",'
                ' "pattern": ["a", ["?", "x"]], "skeleton": [":", "x"]}]}')
        result = tool_load_rules(engine, text=text)  # no format kwarg
        assert result["ok"] is True

    def test_add_rule_basic(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_add_rule

        engine = RuleEngine()
        result = tool_add_rule(
            engine, pattern="(a ?x)", skeleton=":x",
            name="r1", category="identity",
        )
        assert result["ok"] is True
        assert result["rule_index"] >= 0

    def test_add_rule_index_resolved_by_name_under_priority_shuffle(self):
        # add_rule re-sorts by priority, so a later higher-priority add can
        # move the earlier rule. The returned rule_index must point at the
        # rule JUST added (resolved by name), and get_rule(name=...) is the
        # durable handle regardless of the shuffle.
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_add_rule, tool_get_rule

        engine = RuleEngine()
        low = tool_add_rule(engine, pattern="(low ?x)", skeleton=":x",
                            name="low", priority=1)
        high = tool_add_rule(engine, pattern="(high ?x)", skeleton=":x",
                             name="high", priority=100)
        # Each returned index points at its OWN rule at the time of the call.
        assert tool_get_rule(engine, rule_index=high["rule_index"])["name"] == "high"
        # The durable handle is the name, even after the priority re-sort.
        assert tool_get_rule(engine, name="low")["name"] == "low"
        assert tool_get_rule(engine, name="high")["name"] == "high"

    def test_add_rule_bad_example_raises_validation_error(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_add_rule
        from rerum.mcp.errors import MCPToolError

        engine = RuleEngine()
        with pytest.raises(MCPToolError) as exc_info:
            tool_add_rule(
                engine, pattern="(+ ?x 0)", skeleton=":x", name="bad",
                examples=[{"in": "(+ y 0)", "out": "wrong"}],
            )
        assert exc_info.value.code == "validation_error"

    def test_list_rules_filter_by_category(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_list_rules

        engine = RuleEngine.from_dsl("""
            @r1 {category=identity}: (a ?x) => :x
            @r2 {category=distributivity}: (b ?x) => :x
        """)
        result = tool_list_rules(engine, category="identity")
        assert result["count"] == 1
        assert result["rules"][0]["name"] == "r1"

    def test_get_rule_unknown_raises(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_get_rule
        from rerum.mcp.errors import MCPToolError

        engine = RuleEngine()
        with pytest.raises(MCPToolError) as exc_info:
            tool_get_rule(engine, name="nonexistent")
        assert exc_info.value.code == "unknown_rule"

    def test_validate_examples_returns_failures_as_data(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_validate_examples

        engine = RuleEngine()
        engine.add_rule(
            pattern=["a", ["?", "x"]], skeleton=[":", "x"], name="bad",
            examples=[{"in": "(a 1)", "out": "wrong"}],
            validate_examples=False,
        )
        result = tool_validate_examples(engine)
        assert result["ok"] is False
        assert result["errors"][0]["rule_name"] == "bad"


class TestAuthoringJsonSafety:
    """Every authoring response must survive json.dumps, even when a rule
    or example carries an exact rational (a Fraction atom).

    Fraction is not JSON-native; expression fields must render to s-expr
    strings via format_sexpr and structured fields must pass through
    _json_safe. These tests are the proof.
    """

    def _rational_engine(self):
        from rerum import RuleEngine
        from fractions import Fraction
        engine = RuleEngine()
        # A Fraction atom in both pattern and skeleton: this is the value
        # that breaks raw json.dumps and must be rendered via format_sexpr.
        # The example uses s-expr STRINGS (the validator parses them), and
        # rewriting (half 1/2) -> 1/1 holds, so it validates clean.
        # (Rational literals: 1/2 parses to the Fraction atom, the exact
        # round-trip form.)
        engine.add_rule(
            pattern=["half", Fraction(1, 2)],
            skeleton=Fraction(1, 1),
            name="frac",
            category="arith",
            examples=[{"in": "(half 1/2)", "out": "(half 1/2)"}],
            validate_examples=False,
        )
        return engine

    def test_get_rule_json_dumps_clean_with_rational(self):
        import json
        from rerum.mcp.tools import tool_get_rule

        engine = self._rational_engine()
        result = tool_get_rule(engine, name="frac")
        # Must not raise. The expr fields are strings; examples are sanitized.
        json.dumps(result)
        assert result["pattern"] == "(half 1/2)"
        assert result["skeleton"] == "1/1"

    def test_list_rules_json_dumps_clean_with_rational(self):
        import json
        from rerum.mcp.tools import tool_list_rules

        engine = self._rational_engine()
        result = tool_list_rules(engine)
        json.dumps(result)
        assert any(r["name"] == "frac" for r in result["rules"])

    def test_validate_examples_json_dumps_clean_with_rational(self):
        import json
        from fractions import Fraction
        from rerum.mcp.tools import tool_validate_examples

        engine = self._rational_engine()
        # A second rule that carries a Fraction atom AND a wrong example, so
        # the failure-as-data path reports an example referencing a rational.
        # (half 1/2) rewrites to 1/1, not 1/3, so this fails.
        engine.add_rule(
            pattern=["half", Fraction(1, 2)],
            skeleton=Fraction(1, 1),
            name="frac-bad",
            examples=[{"in": "(half 1/2)", "out": "1/3"}],
            validate_examples=False,
        )
        result = tool_validate_examples(engine)
        # The response must json.dumps regardless of pass/fail, and the bad
        # rule must surface as data (not a raised traceback).
        json.dumps(result)
        assert result["ok"] is False
        assert any(e["rule_name"] == "frac-bad" for e in result["errors"])


class TestApplyingTools:
    def test_simplify_returns_situated_trace_and_prose(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_simplify

        engine = RuleEngine.from_dsl(
            '@add-zero {category=identity}: (+ ?x 0) => :x'
        )
        result = tool_simplify(engine, expr="(+ y 0)")
        assert result["result"] == "y"
        assert result["converged"] is True
        step = result["trace"]["steps"][0]
        assert step["rule_id"] == "add-zero"
        assert step["kind"] == "rule"
        assert "before_root" in step
        assert isinstance(result["prose"], str)
        assert "prose" not in result["trace"]  # prose is top-level now

    def test_apply_once_returns_single_step(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_apply_once

        engine = RuleEngine.from_dsl("""
            @r1: (a ?x) => (b :x)
            @r2: (b ?x) => (c :x)
        """)
        result = tool_apply_once(engine, expr="(a y)")
        # apply_once does one rewrite, returning matched rule metadata.
        assert result["result"] == "(b y)"
        assert result["trace"]["total_steps"] == 1

    def test_equivalents_basic(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_equivalents

        engine = RuleEngine.from_dsl('@commute: (+ ?x ?y) <=> (+ :y :x)')
        result = tool_equivalents(engine, expr="(+ a b)", max_depth=3)
        assert "(+ a b)" in result["forms"]
        assert "(+ b a)" in result["forms"]

    def test_prove_equal_proven_carries_prose(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_prove_equal

        engine = RuleEngine.from_dsl('@commute: (+ ?x ?y) <=> (+ :y :x)')
        result = tool_prove_equal(
            engine, expr_a="(+ a b)", expr_b="(+ b a)", max_depth=3
        )
        assert result["proven"] is True
        assert "prose" in result

    def test_prove_equal_unprovable_budget_reports_not_found(self):
        # An unprovable query under a tiny budget must return found=False,
        # never a partial or a hang. The budget is the caller's; honor it.
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_prove_equal

        engine = RuleEngine.from_dsl('@commute: (+ ?x ?y) <=> (+ :y :x)')
        result = tool_prove_equal(
            engine, expr_a="(+ a b)", expr_b="(* a b)",
            max_depth=2, max_expressions=10,
        )
        assert result["proven"] is False
        assert isinstance(result["prose"], str)

    def test_minimize_basic(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_minimize

        engine = RuleEngine.from_dsl(
            '@add-zero {category=identity}: (+ ?x 0) => :x'
        )
        result = tool_minimize(engine, expr="(+ y 0)", metric="size")
        assert result["best"] == "y"
        assert "prose" in result


class TestApplyingJsonSafety:
    """Every applying response must survive json.dumps on a derivation that
    PRODUCES or BINDS an exact rational (a Fraction atom).

    The applying tools emit the richest data -- full traces, equivalent
    forms, proof paths, minimize derivations -- all dense with computed
    VALUES that can be Fraction. A Fraction is not JSON-native, so every
    result/equivalent/proof-path expression must render via format_sexpr
    and every structured field (bindings, guard) must route through
    _json_safe. These tests are the proof, per applying tool.
    """

    def _compute_engine(self):
        # (third ?x) computes (! / 1 3) -> Fraction(1, 3) in the rewrite, so
        # both the simplify result AND the captured step's after are a
        # Fraction. One-way (=>) so the fixpoint is the rational.
        from rerum import RuleEngine
        from rerum.rewriter import ARITHMETIC_PRELUDE
        return RuleEngine.from_dsl(
            '@third: (third ?x) => (! / 1 3)', fold_funcs=ARITHMETIC_PRELUDE)

    def _dual_compute_engine(self):
        # Two compute edges that meet at the SAME Fraction atom: both
        # (third) and (one-over-three) rewrite to (! / 1 3) -> Fraction(1, 3).
        # prove_equal's bidirectional BFS then intersects at the Fraction
        # atom (the proof's common form), so the proof path carries a step
        # whose after IS a Fraction -- the rational-bearing derivation we
        # need to prove json-safe. (Fraction atoms are now spellable as
        # rational literals -- "1/3" round-trips exactly -- but reaching the
        # rational via a compute step also exercises the fold path.)
        from rerum import RuleEngine
        from rerum.rewriter import ARITHMETIC_PRELUDE
        return RuleEngine.from_dsl(
            "@third: (third) => (! / 1 3)\n"
            "@oot: (one-over-three) => (! / 1 3)",
            fold_funcs=ARITHMETIC_PRELUDE)

    def test_simplify_json_dumps_clean_with_rational(self):
        import json
        from rerum.mcp.tools import tool_simplify

        engine = self._compute_engine()
        result = tool_simplify(engine, expr="(third a)")
        # The result and the step's computed value are Fraction(1, 3);
        # both must have rendered to the rational literal "1/3".
        assert result["result"] == "1/3"
        json.dumps(result)

    def test_equivalents_json_dumps_clean_with_rational(self):
        import json
        from rerum.mcp.tools import tool_equivalents

        engine = self._compute_engine()
        result = tool_equivalents(
            engine, expr="(third a)", max_depth=3,
            include_unidirectional=True)
        assert "1/3" in result["forms"]
        json.dumps(result)

    def test_prove_equal_json_dumps_clean_with_rational(self):
        import json
        from rerum.mcp.tools import tool_prove_equal

        engine = self._dual_compute_engine()
        result = tool_prove_equal(
            engine, expr_a="(third)", expr_b="(one-over-three)", max_depth=3,
            include_unidirectional=True)
        assert result["proven"] is True
        # The common form and both path steps' after are Fraction(1, 3),
        # rendered to "1/3".
        assert result["common_form"] == "1/3"
        json.dumps(result)

    def test_minimize_json_dumps_clean_with_rational(self):
        import json
        from rerum.mcp.tools import tool_minimize

        engine = self._compute_engine()
        # Make (third) expensive so minimize selects the computed (/ 1 3);
        # the derivation then carries the Fraction-bearing step.
        result = tool_minimize(
            engine, expr="(third a)", op_costs={"third": 100},
            include_unidirectional=True)
        assert result["best"] == "1/3"
        json.dumps(result)

    def test_apply_once_json_dumps_clean_with_rational(self):
        # apply_once was the one applying tool with no JSON-safety test even
        # though its single-step trace can carry a Fraction (the review's
        # most-suspected gap). Pin it.
        import json
        from rerum.mcp.tools import tool_apply_once
        engine = self._compute_engine()
        result = tool_apply_once(engine, expr="(third a)")
        assert result["result"] == "1/3"
        assert result["changed"] is True
        json.dumps(result)  # must not raise


class TestPathProse:
    """The prose answer line for prove_equal / minimize must reflect the
    result, not 'None'. Regression for the Group 3 review finding: _path_prose
    never set trace.final, so every path-based prose closed with 'Answer:
    None.' and prove_equal carried synthetic no-op filler lines."""

    def test_minimize_prose_answer_line_is_result(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_minimize
        engine = RuleEngine.from_dsl("@az {category=identity}: (+ ?x 0) => :x")
        result = tool_minimize(engine, expr="(+ (+ y 0) 0)")
        last = result["prose"].splitlines()[-1]
        assert last == "Answer: y.", last

    def test_prove_equal_prose_answer_line_and_no_filler(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_prove_equal
        engine = RuleEngine.from_dsl("@comm: (+ ?x ?y) <=> (+ :y :x)")
        result = tool_prove_equal(engine, expr_a="(+ a b)", expr_b="(+ b a)")
        assert result["proven"] is True
        lines = result["prose"].splitlines()
        assert lines[-1] == "Answer: (+ b a).", lines[-1]
        # No synthetic "Applying (anonymous rule): X becomes X." filler.
        assert not any("anonymous rule" in line for line in lines)


class TestErrorMappingRedesign:
    """0.9.0 error-model behaviors: context wiring, HookError unwrapping,
    numeric-failure codes, and JSON-safe details."""

    def test_context_lands_in_details(self):
        from rerum.mcp.errors import map_exception
        err = map_exception(ValueError("boom"), context={"tool": "simplify"})
        assert err["error"]["details"]["context"] == {"tool": "simplify"}

    def test_hook_error_unwraps_mcp_tool_error(self):
        # An MCPToolError raised inside a hook (e.g. the sampling bridge)
        # keeps its own code instead of degrading to internal_error.
        from rerum.mcp.errors import MCPToolError, map_exception
        from rerum.hooks import HookError
        inner = MCPToolError("sampling_unsupported", "no sampling channel")
        try:
            try:
                raise inner
            except MCPToolError as cause:
                raise HookError(lambda: None, "no_match", cause) from cause
        except HookError as wrapped:
            err = map_exception(wrapped, context={"tool": "solve_assisted"})
        assert err["error"]["code"] == "sampling_unsupported"

    def test_hook_error_unwraps_plain_cause(self):
        from rerum.mcp.errors import map_exception
        from rerum.hooks import HookError
        try:
            try:
                raise RuntimeError("kaboom")
            except RuntimeError as cause:
                raise HookError(lambda: None, "no_match", cause) from cause
        except HookError as wrapped:
            err = map_exception(wrapped, context={"tool": "solve_assisted"})
        assert err["error"]["code"] == "internal_error"
        assert "kaboom" in err["error"]["message"]
        assert "via_hook" in err["error"]["details"]["context"]

    def test_numeval_domain_error_gets_domain_code(self):
        from rerum.mcp.errors import map_exception
        from rerum.numeval import NumevalDomainError, NumevalError
        assert map_exception(NumevalDomainError("log of a negative"))[
            "error"]["code"] == "domain_error"
        assert map_exception(NumevalError("unbound symbol: 'q'"))[
            "error"]["code"] == "eval_error"

    def test_to_dict_sanitizes_fraction_details(self):
        import json
        from fractions import Fraction
        from rerum.mcp.errors import MCPToolError
        err = MCPToolError("validation_error", "bad example",
                           details={"example": {"out": Fraction(1, 3)}})
        json.dumps(err.to_dict())  # must not raise
        assert err.to_dict()["error"]["details"]["example"]["out"] == "1/3"


class TestStrictInputs:
    """0.9.0: garbage input is rejected at the boundary, not poured into the
    engine as a None atom (the 'Answer: None.' bug)."""

    def test_empty_expr_is_parse_error(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_simplify
        engine = RuleEngine.from_dsl("@az: (+ ?x 0) => :x")
        for bad in ("", "   ", "("):
            with pytest.raises(MCPToolError) as exc_info:
                tool_simplify(engine, expr=bad)
            assert exc_info.value.code == "parse_error"

    def test_empty_expr_on_prove_equal(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_prove_equal
        engine = RuleEngine.from_dsl("@c: (+ ?x ?y) <=> (+ :y :x)")
        with pytest.raises(MCPToolError) as exc_info:
            tool_prove_equal(engine, expr_a="", expr_b="(+ a b)")
        assert exc_info.value.code == "parse_error"

    def test_empty_pattern_on_add_rule(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_add_rule
        with pytest.raises(MCPToolError) as exc_info:
            tool_add_rule(RuleEngine(), pattern="", skeleton=":x")
        assert exc_info.value.code == "parse_error"


class TestTruthfulConverged:
    """0.9.0: converged reflects the engine's fixpoint event, never a
    hard-coded True (a budget-exhausted simplify no longer lies)."""

    CHAIN = "@s1: a => b\n@s2: b => c\n@s3: c => d"

    def test_budget_exhausted_is_not_converged(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_simplify
        engine = RuleEngine.from_dsl(self.CHAIN)
        result = tool_simplify(engine, expr="a", max_steps=1)
        assert result["converged"] is False
        assert result["result"] != "d"  # genuinely did not finish

    def test_natural_fixpoint_is_converged(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_simplify
        engine = RuleEngine.from_dsl(self.CHAIN)
        result = tool_simplify(engine, expr="a")
        assert result["converged"] is True
        assert result["result"] == "d"

    def test_once_strategy_converged_is_none(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_simplify
        engine = RuleEngine.from_dsl(self.CHAIN)
        result = tool_simplify(engine, expr="a", strategy="once")
        assert result["converged"] is None


class TestApplyOnceMatchSurfacing:
    def test_no_rule_matched(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_apply_once
        result = tool_apply_once(RuleEngine(), expr="(a y)")
        assert result["matched"] is False
        assert result["rule"] is None
        assert result["changed"] is False

    def test_noop_match_is_not_applied(self):
        # apply_once now returns (expr, None) for no-op bindings (result == expr).
        # A matched-but-unchanged rule is indistinguishable from no-match at the
        # apply_once boundary: matched=False, rule=None, changed=False.
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_apply_once
        engine = RuleEngine.from_dsl("@noop: (a ?x) => (a :x)")
        result = tool_apply_once(engine, expr="(a y)")
        assert result["matched"] is False
        assert result["rule"] is None
        assert result["changed"] is False


class TestProveEqualTwoSidedProse:
    def test_prose_narrates_both_sides(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_prove_equal
        engine = RuleEngine.from_dsl("@comm: (+ ?x ?y) <=> (+ :y :x)")
        result = tool_prove_equal(engine, expr_a="(+ a b)", expr_b="(+ b a)")
        assert result["proven"] is True
        prose = result["prose"]
        assert prose.startswith("Both sides reach ")
        assert prose.count("From (") == 2
        assert "anonymous rule" not in prose


class TestAtomicLoadViaTool:
    def test_mid_batch_failure_leaves_engine_unchanged(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_load_rules
        engine = RuleEngine.from_dsl("@good1: (g1 ?x) => :x")
        batch = (
            '{"rules": ['
            '{"name": "good2", "pattern": ["g2", ["?", "x"]],'
            ' "skeleton": [":", "x"]},'
            '{"name": "bad", "pattern": ["b", ["?", "x"]],'
            ' "skeleton": [":", "x"],'
            ' "examples": [{"in": "(b 1)", "out": "wrong"}]}'
            ']}'
        )
        with pytest.raises(MCPToolError) as exc_info:
            tool_load_rules(engine, text=batch, format="json")
        assert exc_info.value.code == "validation_error"
        assert len(engine) == 1  # nothing committed, not even good2
        assert engine.simplify(["g2", "y"]) == ["g2", "y"]


class TestMinimizeProseTruthful:
    def test_minimize_prose_answer_equals_best(self):
        # The derivation path is not oriented original->best, so the prose
        # answer line must come from opt.expr, not the last step's after.
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_minimize
        engine = RuleEngine.from_dsl(
            "@az: (+ ?x 0) <=> :x\n@comm: (+ ?x ?y) <=> (+ :y :x)")
        result = tool_minimize(engine, expr="(+ 0 a)")
        assert result["best"] == "a"
        assert result["prose"].splitlines()[-1] == "Answer: a."


class TestMinimizeProseNoPhantomSteps:
    """After the inverse() fix, the MCP minimize prose narrates real moves:
    no phantom no-op step (before == after) and the answer is the best form.
    Pins the 0.9.0 review's minimize-prose finding once inverse() lands."""

    def test_prose_has_no_no_op_steps(self):
        import re
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_minimize
        engine = RuleEngine.from_dsl(
            "@az: (+ ?x 0) <=> :x\n@comm: (+ ?x ?y) <=> (+ :y :x)")
        result = tool_minimize(engine, expr="(+ 0 a)")
        assert result["best"] == "a"
        assert result["prose"].splitlines()[-1] == "Answer: a."
        step_line = re.compile(
            r"^(?:Applying|Simplifying with|Computing with) .*?: "
            r"(.+) becomes (.+)\.$")
        for line in result["prose"].splitlines():
            m = step_line.match(line)
            if m:
                assert m.group(1) != m.group(2), (
                    f"phantom no-op step in prose: {line!r}")


class TestStatusDiscoverability:
    def test_status_lists_bundles_and_fold_ops(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_get_status, tool_reset_engine
        from rerum.rewriter import PRELUDE_BUNDLES

        engine = RuleEngine()
        status = tool_get_status(engine)
        assert status["available_preludes"] == sorted(PRELUDE_BUNDLES)
        assert status["fold_ops"] == []
        # 'minimal' (previously CLI-only) is now resolvable over MCP.
        tool_reset_engine(engine, prelude="minimal")
        assert tool_get_status(engine)["fold_ops"]


class TestCheckNumericEquiv:
    """The general verification surface: expression-vs-expression numeric
    equivalence over caller-supplied ranges (DATA in, bool out -- same
    security posture as goal kinds). An agent that simplified or solved
    can now confirm its answer over the wire."""

    def test_equivalent_forms_verify(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_check_numeric_equiv
        engine = RuleEngine()
        out = tool_check_numeric_equiv(
            engine, expr_a="(* (+ x 1) (+ x 1))",
            expr_b="(+ (* x x) (+ (* 2 x) 1))",
            ranges={"x": [-2.0, 2.0]})
        assert out["equivalent"] is True

    def test_inequivalent_forms_refute(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_check_numeric_equiv
        out = tool_check_numeric_equiv(
            RuleEngine(), expr_a="(* x 2)", expr_b="(* x 3)",
            ranges={"x": [0.5, 2.0]})
        assert out["equivalent"] is False

    def test_domain_errors_skip_not_refute(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_check_numeric_equiv
        # sqrt undefined on the negative part of the range: those points
        # skip; the defined points agree.
        out = tool_check_numeric_equiv(
            RuleEngine(), expr_a="(sqrt (* x x))", expr_b="(abs x)",
            ranges={"x": [-1.0, 1.0]}, prelude="math")
        assert out["equivalent"] is True

    def test_unbound_symbol_is_an_error_not_false(self):
        from rerum import RuleEngine
        from rerum.mcp.errors import MCPToolError
        from rerum.mcp.tools import tool_check_numeric_equiv
        with pytest.raises(MCPToolError):
            tool_check_numeric_equiv(
                RuleEngine(), expr_a="(+ x y)", expr_b="(+ x 1)",
                ranges={"x": [0.0, 1.0]})  # y never sampled

    def test_bad_ranges_shape_is_parse_error(self):
        from rerum import RuleEngine
        from rerum.mcp.errors import MCPToolError
        from rerum.mcp.tools import tool_check_numeric_equiv
        with pytest.raises(MCPToolError) as exc:
            tool_check_numeric_equiv(
                RuleEngine(), expr_a="x", expr_b="x",
                ranges={"x": [1.0]})  # not a [lo, hi] pair
        assert exc.value.code == "parse_error"


class TestLoadRulesErrorTaxonomy:
    """A structurally malformed rule JSON is the caller's contract error
    (validation_error), not a server fault (internal_error); and garbage
    DSL that parses to zero rules surfaces a note instead of a silent
    ok:True/rules_added:0."""

    def test_malformed_json_rule_is_validation_error(self):
        from rerum import RuleEngine
        from rerum.mcp.errors import MCPToolError
        from rerum.mcp.tools import tool_load_rules
        with pytest.raises(MCPToolError) as exc:
            tool_load_rules(RuleEngine(),
                            text='{"rules": [{"pattern": "(+ ?x 0)"}]}',
                            format="json")  # no skeleton
        assert exc.value.code == "validation_error"

    def test_garbage_dsl_surfaces_a_note(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_load_rules
        result = tool_load_rules(RuleEngine(),
                                 text="this is not a rule\nblah blah",
                                 format="dsl")
        assert result["rules_added"] == 0
        assert "note" in result

    def test_comment_only_input_gets_no_spurious_note(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_load_rules
        result = tool_load_rules(RuleEngine(),
                                 text="# just a comment\n\n", format="dsl")
        assert result["rules_added"] == 0
        assert "note" not in result

    def test_valid_load_has_no_note(self):
        from rerum import RuleEngine
        from rerum.mcp.tools import tool_load_rules
        result = tool_load_rules(RuleEngine(), text="@r: (f ?x) => :x",
                                 format="dsl")
        assert result == {"ok": True, "rules_added": 1}
