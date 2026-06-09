"""Tests for theory-driven normalization (rerum/normalize.py).

Every test constructs its own small Theory. The engine ships NO built-in
theory naming +/*; arithmetic is just one data instance, boolean is another.
"""

import pytest

from rerum.normalize import (
    Theory, flatten, ORDER_KEY, canonical_sort, collect_like_terms, normalize,
)

# Arithmetic theory built IN-TEST from data (not from the engine).
ARITH = Theory.from_dict({
    "+": {"ac": True, "identity": 0, "repeat": {"op": "*", "via": "count"}},
    "*": {"ac": True, "identity": 1, "annihilator": 0,
          "repeat": {"op": "^", "via": "exp"}},
})

# Boolean theory: a DIFFERENT data instance, same machinery.
BOOL = Theory.from_dict({
    "and": {"ac": True, "identity": True, "annihilator": False},
    "or": {"ac": True, "identity": False, "annihilator": True},
})

EMPTY = Theory.from_dict({})


class TestTheory:
    def test_is_ac_reads_data(self):
        assert ARITH.is_ac("+") is True
        assert ARITH.is_ac("*") is True
        assert ARITH.is_ac("-") is False
        assert ARITH.is_ac("dd") is False

    def test_is_ac_for_boolean(self):
        assert BOOL.is_ac("and") is True
        assert BOOL.is_ac("or") is True
        assert BOOL.is_ac("+") is False  # arithmetic ops unknown to a boolean theory

    def test_empty_theory_has_no_ac_ops(self):
        assert EMPTY.is_ac("+") is False
        assert EMPTY.is_ac("*") is False
        assert EMPTY.is_ac("and") is False

    def test_identity(self):
        assert ARITH.identity("+") == 0
        assert ARITH.identity("*") == 1
        assert ARITH.identity("-") is None
        assert BOOL.identity("and") is True

    def test_annihilator(self):
        assert ARITH.annihilator("*") == 0
        assert ARITH.annihilator("+") is None
        assert BOOL.annihilator("or") is True

    def test_repeat(self):
        assert ARITH.repeat("+") == {"op": "*", "via": "count"}
        assert ARITH.repeat("*") == {"op": "^", "via": "exp"}
        # boolean ops declare no repeat (idempotent): None.
        assert BOOL.repeat("and") is None
        assert ARITH.repeat("-") is None

    def test_from_json(self):
        import json
        t = Theory.from_json(json.dumps({"+": {"ac": True, "identity": 0}}))
        assert t.is_ac("+") is True
        assert t.identity("+") == 0
        assert t.annihilator("+") is None


class TestFlatten:
    def test_flatten_nested_plus(self):
        assert flatten(["+", ["+", "a", "b"], "c"], ARITH) == ["+", "a", "b", "c"]

    def test_flatten_nested_times(self):
        assert flatten(["*", ["*", "a", "b"], "c"], ARITH) == ["*", "a", "b", "c"]

    def test_flatten_right_nested(self):
        assert flatten(["+", "a", ["+", "b", "c"]], ARITH) == ["+", "a", "b", "c"]

    def test_flatten_deep(self):
        expr = ["+", ["+", ["+", "a", "b"], "c"], "d"]
        assert flatten(expr, ARITH) == ["+", "a", "b", "c", "d"]

    def test_flatten_does_not_merge_mixed_ops(self):
        assert flatten(["+", ["*", "a", "b"], "c"], ARITH) == \
            ["+", ["*", "a", "b"], "c"]

    def test_flatten_recurses_into_non_ac_ops(self):
        assert flatten(["-", ["+", ["+", "a", "b"], "c"], "d"], ARITH) == \
            ["-", ["+", "a", "b", "c"], "d"]

    def test_flatten_atom_unchanged(self):
        assert flatten("x", ARITH) == "x"
        assert flatten(5, ARITH) == 5

    def test_flatten_idempotent(self):
        once = flatten(["+", ["+", "a", "b"], "c"], ARITH)
        assert flatten(once, ARITH) == once

    def test_flatten_empty_theory_no_change(self):
        # Empty theory: no operator is AC, so no flattening happens.
        expr = ["+", ["+", "a", "b"], "c"]
        assert flatten(expr, EMPTY) == expr


class TestOrderKey:
    def test_numbers_before_symbols(self):
        assert ORDER_KEY(2) < ORDER_KEY("x")
        assert ORDER_KEY(0) < ORDER_KEY("a")

    def test_symbols_before_compounds(self):
        assert ORDER_KEY("z") < ORDER_KEY(["+", "a", "b"])
        assert ORDER_KEY("x") < ORDER_KEY(["*", 2, "y"])

    def test_numbers_before_compounds(self):
        assert ORDER_KEY(100) < ORDER_KEY(["+", "a", "b"])

    def test_numbers_sorted_by_value(self):
        assert ORDER_KEY(1) < ORDER_KEY(2) < ORDER_KEY(10)
        assert ORDER_KEY(-5) < ORDER_KEY(0)

    def test_symbols_lexicographic(self):
        assert ORDER_KEY("a") < ORDER_KEY("b") < ORDER_KEY("z")
        assert ORDER_KEY("x") < ORDER_KEY("xy")

    def test_compounds_by_head_then_args(self):
        assert ORDER_KEY(["+", "a", "b"]) < ORDER_KEY(["+", "b", "b"])
        assert ORDER_KEY(["*", "a"]) < ORDER_KEY(["^", "a"]) or \
            ORDER_KEY(["^", "a"]) < ORDER_KEY(["*", "a"])

    def test_total_order_no_typeerror(self):
        items = [["+", "a", "b"], "x", 3, 1, "a", ["*", 2, "y"], -1]
        ordered = sorted(items, key=ORDER_KEY)
        assert ordered[:3] == [-1, 1, 3]
        assert ordered[3:5] == ["a", "x"]
        assert all(isinstance(e, list) for e in ordered[5:])

    def test_no_theory_argument(self):
        # ORDER_KEY is domain-free: it takes only an expression.
        import inspect
        params = list(inspect.signature(ORDER_KEY).parameters)
        assert params == ["expr"]

    def test_key_is_strict(self):
        exprs = [1, 2, "a", "b", ["+", "a", "b"], ["*", "a", "b"]]
        keys = [ORDER_KEY(e) for e in exprs]
        assert len(set(keys)) == len(exprs)

    def test_order_key_handles_bool_without_crash(self):
        # Bools are valid atoms (boolean-theory identities/annihilators).
        # ORDER_KEY must not crash on them.
        kt = ORDER_KEY(True)
        kf = ORDER_KEY(False)
        assert kf < kt  # False sorts before True
        # comparable with numbers and symbols without TypeError
        keys = [ORDER_KEY(x) for x in [True, False, 1, 0, "x", ["+", "a"]]]
        assert sorted(keys) == sorted(keys)  # no TypeError raised by sorting

    def test_order_key_bool_in_compound(self):
        # A compound containing bool atoms is orderable.
        k = ORDER_KEY(["and", True, "x"])
        assert isinstance(k, tuple)


class TestCanonicalSort:
    def test_sort_numbers_first_then_vars(self):
        # The contract's worked example: (+ x 2 y) -> (+ 2 x y).
        assert canonical_sort(["+", "x", 2, "y"], ARITH) == ["+", 2, "x", "y"]

    def test_sort_times(self):
        assert canonical_sort(["*", "y", "x", 3], ARITH) == ["*", 3, "x", "y"]

    def test_sort_is_stable_on_already_sorted(self):
        assert canonical_sort(["+", 2, "x", "y"], ARITH) == ["+", 2, "x", "y"]

    def test_sort_recurses(self):
        expr = ["+", ["*", "b", "a"], 1]
        assert canonical_sort(expr, ARITH) == ["+", 1, ["*", "a", "b"]]

    def test_sort_preserves_non_commutative_order(self):
        # subtraction is not AC: operands not reordered; children still sorted.
        assert canonical_sort(["-", ["*", "b", "a"], "c"], ARITH) == \
            ["-", ["*", "a", "b"], "c"]

    def test_sort_atom(self):
        assert canonical_sort("x", ARITH) == "x"
        assert canonical_sort(7, ARITH) == 7

    def test_sort_empty_theory_no_change(self):
        assert canonical_sort(["+", "x", 2, "y"], EMPTY) == ["+", "x", 2, "y"]

    def test_sort_confluent_over_permutations(self):
        import itertools
        base = ["+", "c", "a", "b", 2, 1]
        ref = canonical_sort(base, ARITH)
        for perm in itertools.permutations(["a", "b", "c", 1, 2]):
            assert canonical_sort(["+", *perm], ARITH) == ref


class TestCollectLikeTerms:
    def test_collect_x_plus_x(self):
        # x + x -> (* 2 x) via repeat {op:*, via:count}
        assert collect_like_terms(["+", "x", "x"], ARITH) == ["*", 2, "x"]

    def test_collect_coeff_terms(self):
        # (* 2 x) + (* 3 x) -> (* 5 x)
        assert collect_like_terms(["+", ["*", 2, "x"], ["*", 3, "x"]], ARITH) == \
            ["*", 5, "x"]

    def test_collect_mixed_coeff_and_bare(self):
        # x + (* 2 x) -> (* 3 x)
        assert collect_like_terms(["+", "x", ["*", 2, "x"]], ARITH) == \
            ["*", 3, "x"]

    def test_collect_keeps_distinct_terms(self):
        assert collect_like_terms(["+", "x", "y"], ARITH) == ["+", "x", "y"]

    def test_collect_x_times_x(self):
        # x * x -> (^ x 2) via repeat {op:^, via:exp}
        assert collect_like_terms(["*", "x", "x"], ARITH) == ["^", "x", 2]

    def test_collect_power_factors(self):
        # (^ x 2) * (^ x 3) -> (^ x 5)
        assert collect_like_terms(["*", ["^", "x", 2], ["^", "x", 3]], ARITH) == \
            ["^", "x", 5]

    def test_collect_mixed_power_and_bare(self):
        # x * (^ x 2) -> (^ x 3)
        assert collect_like_terms(["*", "x", ["^", "x", 2]], ARITH) == \
            ["^", "x", 3]

    def test_collect_distinct_factors(self):
        assert collect_like_terms(["*", "x", "y"], ARITH) == ["*", "x", "y"]

    def test_collect_recurses(self):
        assert collect_like_terms(["-", ["+", "x", "x"], "y"], ARITH) == \
            ["-", ["*", 2, "x"], "y"]

    def test_collect_single_operand_unwraps(self):
        assert collect_like_terms(["+", "x"], ARITH) == "x"
        assert collect_like_terms(["*", "x"], ARITH) == "x"

    def test_collect_idempotent_op_collapses(self):
        # No repeat declared (boolean and is idempotent): (and a a) -> a.
        assert collect_like_terms(["and", "a", "a"], BOOL) == "a"
        assert collect_like_terms(["or", "x", "x", "y"], BOOL) == ["or", "x", "y"]

    def test_collect_empty_theory_no_change(self):
        # No AC op: nothing collected.
        assert collect_like_terms(["+", "x", "x"], EMPTY) == ["+", "x", "x"]
