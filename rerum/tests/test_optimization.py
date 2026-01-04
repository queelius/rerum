"""Tests for cost optimization (minimize) and random sampling."""

import random
import pytest
from rerum.engine import (
    RuleEngine, OptimizationResult, format_sexpr, _expr_to_tuple,
    expr_size, expr_depth, expr_ops, expr_atoms, make_op_cost_fn,
    COST_METRICS,
)


class TestCostFunctions:
    """Tests for built-in cost functions."""

    def test_expr_size_atom(self):
        """Atoms have size 1."""
        assert expr_size("x") == 1
        assert expr_size(42) == 1
        assert expr_size(3.14) == 1

    def test_expr_size_simple(self):
        """Simple expression size."""
        assert expr_size(["+", "x", 1]) == 3

    def test_expr_size_nested(self):
        """Nested expression size."""
        assert expr_size(["+", ["*", "a", "b"], "c"]) == 5

    def test_expr_depth_atom(self):
        """Atoms have depth 0."""
        assert expr_depth("x") == 0
        assert expr_depth(42) == 0

    def test_expr_depth_simple(self):
        """Simple expression depth."""
        assert expr_depth(["+", "x", 1]) == 1

    def test_expr_depth_nested(self):
        """Nested expression depth."""
        assert expr_depth(["+", ["*", "a", "b"], "c"]) == 2
        assert expr_depth(["+", ["*", ["-", "a", "b"], "c"], "d"]) == 3

    def test_expr_ops_atom(self):
        """Atoms have 0 ops."""
        assert expr_ops("x") == 0
        assert expr_ops(42) == 0

    def test_expr_ops_simple(self):
        """Simple expression has 1 op."""
        assert expr_ops(["+", "x", 1]) == 1

    def test_expr_ops_nested(self):
        """Nested expression ops count."""
        assert expr_ops(["+", ["*", "a", "b"], "c"]) == 2

    def test_expr_atoms_single(self):
        """Single atom count."""
        assert expr_atoms("x") == 1
        assert expr_atoms(42) == 1

    def test_expr_atoms_simple(self):
        """Simple expression atoms."""
        assert expr_atoms(["+", "x", 1]) == 2

    def test_expr_atoms_nested(self):
        """Nested expression atoms."""
        assert expr_atoms(["+", ["*", "a", "b"], "c"]) == 3

    def test_cost_metrics_dict(self):
        """COST_METRICS contains all built-in metrics."""
        assert "size" in COST_METRICS
        assert "depth" in COST_METRICS
        assert "ops" in COST_METRICS
        assert "atoms" in COST_METRICS


class TestMakeOpCostFn:
    """Tests for make_op_cost_fn."""

    def test_basic_op_costs(self):
        """Basic operator cost calculation."""
        cost_fn = make_op_cost_fn({"+": 1, "*": 2})
        assert cost_fn(["+", "a", "b"]) == 1
        assert cost_fn(["*", "a", "b"]) == 2

    def test_nested_op_costs(self):
        """Nested operator costs sum up."""
        cost_fn = make_op_cost_fn({"+": 1, "*": 2})
        # (+ (* a b) c) = 1 + 2 = 3
        assert cost_fn(["+", ["*", "a", "b"], "c"]) == 3

    def test_default_cost(self):
        """Default cost for unknown operators."""
        cost_fn = make_op_cost_fn({"+": 1}, default=10)
        assert cost_fn(["+", "a", "b"]) == 1
        assert cost_fn(["unknown", "a", "b"]) == 10

    def test_atom_zero_cost(self):
        """Atoms have zero operator cost."""
        cost_fn = make_op_cost_fn({"+": 1})
        assert cost_fn("x") == 0
        assert cost_fn(42) == 0


class TestOptimizationResultClass:
    """Tests for OptimizationResult class."""

    def test_basic_attributes(self):
        """OptimizationResult stores attributes correctly."""
        result = OptimizationResult(
            expr=["+", "a", "b"],
            cost=3,
            original=["+", ["+", "a", "b"], 0],
            original_cost=5,
            expressions_checked=10
        )
        assert result.expr == ["+", "a", "b"]
        assert result.cost == 3
        assert result.original_cost == 5
        assert result.expressions_checked == 10

    def test_improvement(self):
        """improvement property calculates cost reduction."""
        result = OptimizationResult(
            expr="a", cost=1, original="b", original_cost=5
        )
        assert result.improvement == 4

    def test_improvement_ratio(self):
        """improvement_ratio calculates ratio."""
        result = OptimizationResult(
            expr="a", cost=2, original="b", original_cost=4
        )
        assert result.improvement_ratio == 0.5

    def test_improvement_ratio_zero_original(self):
        """improvement_ratio handles zero original cost."""
        result = OptimizationResult(
            expr="a", cost=0, original="b", original_cost=0
        )
        assert result.improvement_ratio == 1.0

    def test_bool_improved(self):
        """Bool is True when improved."""
        result = OptimizationResult(
            expr="a", cost=1, original="b", original_cost=5
        )
        assert bool(result) is True

    def test_bool_not_improved(self):
        """Bool is False when not improved."""
        result = OptimizationResult(
            expr="a", cost=5, original="b", original_cost=5
        )
        assert bool(result) is False

    def test_repr(self):
        """Repr shows key info."""
        result = OptimizationResult(
            expr=["+", "a", "b"], cost=3, original="x", original_cost=5
        )
        r = repr(result)
        assert "(+ a b)" in r
        assert "cost=3" in r

    def test_to_dict(self):
        """to_dict produces serializable dictionary."""
        result = OptimizationResult(
            expr="a", cost=1, original="b", original_cost=5,
            expressions_checked=10
        )
        d = result.to_dict()
        assert d["expr"] == "a"
        assert d["cost"] == 1
        assert d["original_cost"] == 5
        assert d["improvement"] == 4
        assert d["expressions_checked"] == 10


class TestMinimizeBasic:
    """Basic tests for minimize() method."""

    def test_minimize_returns_result(self):
        """minimize returns OptimizationResult."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)
        result = engine.minimize(["+", "a", "b"])
        assert isinstance(result, OptimizationResult)

    def test_minimize_with_metric(self):
        """minimize with built-in metric."""
        engine = RuleEngine.from_dsl("""
            @add-zero: (+ ?x 0) <=> :x
        """)
        # (+ a 0) has size 3, simplified "a" has size 1
        result = engine.minimize(["+", "a", 0], metric="size")
        assert result.cost <= result.original_cost

    def test_minimize_with_custom_cost(self):
        """minimize with custom cost function."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)

        def my_cost(expr):
            # Prefer "a" to be first
            if isinstance(expr, list) and len(expr) > 1:
                if expr[1] == "a":
                    return 0
                return 1
            return 0

        result = engine.minimize(["+", "b", "a"], cost=my_cost)
        # Should find (+ a b) which has cost 0
        assert result.expr == ["+", "a", "b"]

    def test_minimize_with_op_costs(self):
        """minimize with operator costs."""
        engine = RuleEngine.from_dsl("""
            @dist: (* ?x (+ ?y ?z)) <=> (+ (* :x :y) (* :x :z))
        """)

        # x * (y + z) = 1 mul + 1 add = 3
        # (x*y) + (x*z) = 2 mul + 1 add = 5
        result = engine.minimize(
            ["*", "x", ["+", "y", "z"]],
            op_costs={"+": 1, "*": 2}
        )
        # Original form should be cheaper
        assert result.expr == ["*", "x", ["+", "y", "z"]]

    def test_minimize_invalid_metric(self):
        """Invalid metric raises ValueError."""
        engine = RuleEngine()
        with pytest.raises(ValueError, match="Unknown metric"):
            engine.minimize(["+", "a", "b"], metric="invalid")

    def test_minimize_no_improvement(self):
        """minimize when no improvement possible."""
        engine = RuleEngine()  # No rules
        result = engine.minimize(["+", "a", "b"], metric="size")
        assert result.expr == ["+", "a", "b"]
        assert result.cost == result.original_cost


class TestMinimizeComplex:
    """Complex optimization tests."""

    def test_minimize_finds_simplest(self):
        """Finds simplest equivalent form."""
        engine = RuleEngine.from_dsl("""
            @add-zero: (+ ?x 0) <=> :x
            @mul-one: (* ?x 1) <=> :x
        """)

        # (* (+ a 0) 1) should simplify to just a
        result = engine.minimize(
            ["*", ["+", "a", 0], 1],
            metric="size",
            max_depth=5
        )
        assert result.expr == "a"
        assert result.cost == 1

    def test_minimize_depth_reduction(self):
        """Minimize can reduce depth."""
        engine = RuleEngine.from_dsl("""
            @assoc: (+ (+ ?x ?y) ?z) <=> (+ :x (+ :y :z))
        """)

        # Both have same depth, so either form is fine
        result = engine.minimize(
            ["+", ["+", "a", "b"], "c"],
            metric="depth",
            max_depth=3
        )
        assert result.cost <= 2  # Depth is at most 2

    def test_minimize_with_groups(self):
        """minimize respects group filtering."""
        engine = RuleEngine.from_dsl("""
            [algebra]
            @add-zero: (+ ?x 0) <=> :x

            [other]
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)

        result = engine.minimize(
            ["+", "a", 0],
            metric="size",
            groups=["algebra"]
        )
        # Should find simplified form via add-zero
        assert result.expr == "a"


class TestRandomEquivalent:
    """Tests for random_equivalent() method."""

    def test_returns_expression(self):
        """random_equivalent returns an expression."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)
        result = engine.random_equivalent(["+", "a", "b"])
        assert result is not None

    def test_reproducible_with_rng(self):
        """Results are reproducible with same RNG seed."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
            @assoc: (+ (+ ?x ?y) ?z) <=> (+ :x (+ :y :z))
        """)

        rng1 = random.Random(42)
        rng2 = random.Random(42)

        result1 = engine.random_equivalent(
            ["+", ["+", "a", "b"], "c"], steps=5, rng=rng1
        )
        result2 = engine.random_equivalent(
            ["+", ["+", "a", "b"], "c"], steps=5, rng=rng2
        )
        assert result1 == result2

    def test_no_rules_returns_original(self):
        """With no applicable rules, returns original."""
        engine = RuleEngine()
        result = engine.random_equivalent(["+", "a", "b"], steps=10)
        assert result == ["+", "a", "b"]

    def test_steps_zero_returns_original(self):
        """With steps=0, returns original."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)
        result = engine.random_equivalent(["+", "a", "b"], steps=0)
        assert result == ["+", "a", "b"]


class TestSampleEquivalents:
    """Tests for sample_equivalents() method."""

    def test_returns_list(self):
        """sample_equivalents returns a list."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)
        samples = engine.sample_equivalents(["+", "a", "b"], n=3)
        assert isinstance(samples, list)

    def test_correct_count(self):
        """Returns requested number of samples (if possible)."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
            @assoc: (+ (+ ?x ?y) ?z) <=> (+ :x (+ :y :z))
        """)
        samples = engine.sample_equivalents(
            ["+", ["+", "a", "b"], "c"], n=5, steps=5
        )
        # May get fewer if not enough unique expressions
        assert len(samples) <= 5

    def test_unique_default(self):
        """By default, returns unique samples."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)
        samples = engine.sample_equivalents(
            ["+", "a", "b"], n=10, steps=5, max_attempts=50
        )
        keys = [_expr_to_tuple(s) for s in samples]
        assert len(keys) == len(set(keys))  # All unique

    def test_non_unique(self):
        """With unique=False, may have duplicates."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)
        rng = random.Random(42)
        samples = engine.sample_equivalents(
            ["+", "a", "b"], n=10, steps=5, unique=False, rng=rng
        )
        assert len(samples) == 10

    def test_reproducible(self):
        """Results reproducible with RNG."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)
        rng1 = random.Random(123)
        rng2 = random.Random(123)

        samples1 = engine.sample_equivalents(
            ["+", "a", "b"], n=3, steps=5, rng=rng1
        )
        samples2 = engine.sample_equivalents(
            ["+", "a", "b"], n=3, steps=5, rng=rng2
        )
        assert samples1 == samples2


class TestRandomWalk:
    """Tests for random_walk() method."""

    def test_yields_expressions(self):
        """random_walk yields expressions."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)
        walk = list(engine.random_walk(["+", "a", "b"], max_steps=3))
        assert len(walk) > 0

    def test_starts_with_original(self):
        """First yielded expression is the original."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)
        walk = list(engine.random_walk(["+", "a", "b"], max_steps=3))
        assert walk[0] == ["+", "a", "b"]

    def test_max_steps_limit(self):
        """Respects max_steps limit."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)
        walk = list(engine.random_walk(["+", "a", "b"], max_steps=5))
        # At most max_steps + 1 (including original)
        assert len(walk) <= 6

    def test_stops_when_stuck(self):
        """Stops when no more rewrites possible."""
        engine = RuleEngine()  # No rules
        walk = list(engine.random_walk(["+", "a", "b"], max_steps=100))
        assert len(walk) == 1  # Just the original

    def test_reproducible(self):
        """Walk is reproducible with RNG."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)
        rng1 = random.Random(42)
        rng2 = random.Random(42)

        walk1 = list(engine.random_walk(
            ["+", "a", "b"], max_steps=5, rng=rng1
        ))
        walk2 = list(engine.random_walk(
            ["+", "a", "b"], max_steps=5, rng=rng2
        ))
        assert walk1 == walk2

    def test_is_lazy(self):
        """random_walk is a lazy generator."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
        """)
        gen = engine.random_walk(["+", "a", "b"], max_steps=1000)

        # Just get first 3 without consuming all
        first = next(gen)
        second = next(gen)
        third = next(gen)

        assert first == ["+", "a", "b"]


class TestOptimizationPractical:
    """Practical optimization usage tests."""

    def test_simplify_nested_zeros(self):
        """Simplify expression with multiple zeros."""
        engine = RuleEngine.from_dsl("""
            @add-zero: (+ ?x 0) <=> :x
        """)

        result = engine.minimize(
            ["+", ["+", ["+", "a", 0], 0], 0],
            metric="size",
            max_depth=5
        )
        assert result.expr == "a"

    def test_minimize_can_increase_first(self):
        """Optimization may increase size before decreasing."""
        engine = RuleEngine.from_dsl("""
            @expand: (square ?x) <=> (* :x :x)
            @fold-mul: (* ?a:const ?b:const) => (! * :a :b)
        """)
        # Note: fold-mul is unidirectional, so won't be used by default

        # square 3 -> (* 3 3) -> actually we can't fold without include_unidirectional
        result = engine.minimize(
            ["square", 3],
            metric="size",
            max_depth=3
        )
        # Without fold, (* 3 3) is larger than (square 3)
        # So original should be kept
        assert result.expr == ["square", 3]

    def test_sample_diversity(self):
        """Sampling produces diverse equivalents."""
        engine = RuleEngine.from_dsl("""
            @commute: (+ ?x ?y) <=> (+ :y :x)
            @assoc: (+ (+ ?x ?y) ?z) <=> (+ :x (+ :y :z))
        """)

        samples = engine.sample_equivalents(
            ["+", ["+", "a", "b"], "c"],
            n=5, steps=10,
            max_attempts=100
        )

        # Should have at least 2 different forms
        unique_keys = set(_expr_to_tuple(s) for s in samples)
        assert len(unique_keys) >= 2
