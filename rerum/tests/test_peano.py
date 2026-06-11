"""Tests for Peano arithmetic as pure example content (examples/peano.rules).

THE demonstration that computation lives in rules, not the engine: this set
has NO prelude, NO fold functions, NO theory -- arithmetic emerges from
structural rewriting alone. The salient certification is the property test:
for all m, n in 0..6, rewriting (add/mul of the encodings) equals the
encoding of m+n / m*n.
"""

from pathlib import Path

import pytest

from rerum.engine import RuleEngine, parse_sexpr

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
RULES = EXAMPLES / "peano.rules"
SIDECAR = EXAMPLES / "peano.metadata.json"


def _engine():
    eng = RuleEngine()  # deliberately NO prelude: pure structural rewriting
    eng.load_file(RULES, validate_examples=False)
    eng.load_metadata_json(SIDECAR.read_text(), validate_examples=True)
    return eng


def int_to_peano(n):
    expr = "z"
    for _ in range(n):
        expr = ["s", expr]
    return expr


def peano_to_int(expr):
    n = 0
    while isinstance(expr, list) and expr[0] == "s":
        n += 1
        expr = expr[1]
    assert expr == "z", f"not a Peano normal form: {expr!r}"
    return n


class TestLoadAndValidate:
    def test_loads_with_no_prelude(self):
        eng = _engine()
        assert len(eng.list_rules()) > 0
        assert eng.has_fold_funcs() is False

    def test_every_rule_has_an_example(self):
        eng = _engine()
        for _i, _r, meta in eng.iter_rules():
            assert meta.examples, f"rule {meta.name!r} has no examples"


class TestArithmeticEmerges:
    """The killer property: arithmetic from five rules and an empty prelude."""

    def test_addition_is_correct_for_all_small_pairs(self):
        eng = _engine()
        for m in range(7):
            for n in range(7):
                out = eng(["add", int_to_peano(m), int_to_peano(n)])
                assert peano_to_int(out) == m + n, f"{m} + {n}"

    def test_multiplication_is_correct_for_all_small_pairs(self):
        eng = _engine()
        for m in range(6):
            for n in range(6):
                out = eng(["mul", int_to_peano(m), int_to_peano(n)])
                assert peano_to_int(out) == m * n, f"{m} * {n}"

    def test_monus_subtraction_truncates_at_zero(self):
        eng = _engine()
        for m in range(6):
            for n in range(6):
                out = eng(["sub", int_to_peano(m), int_to_peano(n)])
                assert peano_to_int(out) == max(0, m - n), f"{m} - {n}"

    def test_parity(self):
        eng = _engine()
        for n in range(8):
            out = eng(["even?", int_to_peano(n)])
            assert out == ("true" if n % 2 == 0 else "false"), f"even? {n}"

    def test_comparison(self):
        eng = _engine()
        for m in range(6):
            for n in range(6):
                out = eng(["le", int_to_peano(m), int_to_peano(n)])
                assert out == ("true" if m <= n else "false"), f"{m} <= {n}"

    def test_compound_expression(self):
        eng = _engine()
        # (2 + 3) * 2 = 10, nested under one fixpoint run.
        expr = ["mul", ["add", int_to_peano(2), int_to_peano(3)],
                int_to_peano(2)]
        assert peano_to_int(eng(expr)) == 10

    def test_five_times_five_terminates_within_default_budget(self):
        eng = _engine()
        assert peano_to_int(eng(["mul", int_to_peano(5),
                                 int_to_peano(5)])) == 25


class TestEveryRuleFires:
    def test_each_rule_appears_in_a_trace(self):
        eng = _engine()
        drivers = {
            "add-zero": "(add z (s z))",
            "add-succ": "(add (s z) (s z))",
            "mul-zero": "(mul z (s z))",
            "mul-succ": "(mul (s z) (s z))",
            "pred-zero": "(pred z)",
            "pred-succ": "(pred (s z))",
            "sub-zero": "(sub (s z) z)",
            "sub-succ": "(sub (s z) (s z))",
            "even-zero": "(even? z)",
            "even-succ": "(even? (s z))",
            "odd-zero": "(odd? z)",
            "odd-succ": "(odd? (s z))",
            "le-zero": "(le z z)",
            "le-succ-zero": "(le (s z) z)",
            "le-succ-succ": "(le (s z) (s z))",
        }
        for name, src in drivers.items():
            _result, trace = eng.simplify(parse_sexpr(src), trace=True)
            fired = [s.metadata.name for s in trace.steps]
            assert name in fired, f"{name} did not fire on {src}"


class TestDerivationQuality:
    def test_trace_names_rules_and_chains(self):
        eng = _engine()
        result, trace = eng.simplify(
            parse_sexpr("(add (s z) (s z))"), trace=True)
        assert result == ["s", ["s", "z"]]
        names = [s.metadata.name for s in trace.steps]
        assert "add-succ" in names and "add-zero" in names
        seq = trace.to_global_sequence()
        assert seq[0]["before_root"] == ["add", ["s", "z"], ["s", "z"]]
        assert seq[-1]["after_root"] == result
        for k in range(len(seq) - 1):
            assert seq[k]["after_root"] == seq[k + 1]["before_root"]
