"""Tests for goal-directed best-first search (solve)."""

import pytest

from rerum.engine import RuleEngine
from rerum.solve import contains_op
from rerum.optimize import expr_size


class TestContainsOp:
    def test_atom_has_no_op(self):
        assert contains_op("x", {"foo"}) is False
        assert contains_op(42, {"foo"}) is False

    def test_top_level_op(self):
        assert contains_op(["foo", "x"], {"foo"}) is True

    def test_nested_op(self):
        assert contains_op(["+", "x", ["foo", "y"]], {"foo"}) is True

    def test_absent_op(self):
        assert contains_op(["+", "x", ["bar", "y"]], {"foo"}) is False

    def test_multiple_ops(self):
        assert contains_op(["aaa", "x", "x"], {"aaa", "bbb"}) is True
        assert contains_op(["bbb", "x", "x", 0], {"aaa", "bbb"}) is True
