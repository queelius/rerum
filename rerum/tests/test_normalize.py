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
