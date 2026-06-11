"""Tests for SKI combinators as pure example content (examples/ski.rules).

Three rules form a Turing-complete rewrite system; the salient behaviors
are (a) classic reductions work, (b) Church-style booleans compute through
the rules, and (c) the engine is HONEST about non-termination: a divergent
term returns partially reduced under a budget instead of hanging.
"""

from pathlib import Path

import pytest

from rerum.engine import RuleEngine, parse_sexpr
from rerum.solve import contains_op

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
RULES = EXAMPLES / "ski.rules"
SIDECAR = EXAMPLES / "ski.metadata.json"


def _engine():
    eng = RuleEngine()  # NO prelude
    eng.load_file(RULES, validate_examples=False)
    eng.load_metadata_json(SIDECAR.read_text(), validate_examples=True)
    return eng


def app(*terms):
    """Left-nested application: app(f, x, y) == ((f x) y)."""
    out = terms[0]
    for t in terms[1:]:
        out = ["app", out, t]
    return out


class TestLoadAndValidate:
    def test_loads_with_no_prelude(self):
        eng = _engine()
        assert len(eng.list_rules()) == 3
        assert eng.has_fold_funcs() is False

    def test_every_rule_has_an_example(self):
        eng = _engine()
        for _i, _r, meta in eng.iter_rules():
            assert meta.examples, f"rule {meta.name!r} has no examples"


class TestClassicReductions:
    def test_identity(self):
        eng = _engine()
        assert eng(app("I", "a")) == "a"

    def test_k_discards(self):
        eng = _engine()
        assert eng(app("K", "a", "b")) == "a"

    def test_s_distributes(self):
        eng = _engine()
        # S f g a -> ((f a) (g a))
        assert eng(app("S", "f", "g", "a")) == [
            "app", ["app", "f", "a"], ["app", "g", "a"]]

    def test_skk_behaves_as_identity(self):
        eng = _engine()
        # S K K x -> (K x)(K x) -> x : the classic extensional identity.
        assert eng(app("S", "K", "K", "a")) == "a"

    def test_sii_duplicates(self):
        eng = _engine()
        # S I I x -> (I x)(I x) -> (x x)
        assert eng(app("S", "I", "I", "a")) == ["app", "a", "a"]

    def test_every_rule_fires_in_traces(self):
        # One driver per rule (a single combined term would let K discard
        # the I-redex before it fires -- correct reduction, wrong probe).
        eng = _engine()
        drivers = {
            "ski-i": app("I", "a"),
            "ski-k": app("K", "a", "b"),
            "ski-s": app("S", "f", "g", "a"),
        }
        for name, term in drivers.items():
            _result, trace = eng.simplify(term, trace=True)
            fired = {s.metadata.name for s in trace.steps}
            assert name in fired, f"{name} did not fire"


class TestChurchBooleans:
    """TRUE = K, FALSE = K I: selection computes through the rules."""

    def test_true_selects_first(self):
        eng = _engine()
        assert eng(app("K", "a", "b")) == "a"

    def test_false_selects_second(self):
        eng = _engine()
        false = ["app", "K", "I"]
        assert eng(app(false, "a", "b")) == "b"


class TestHonestNonTermination:
    def test_omega_returns_under_budget(self):
        eng = _engine()
        sii = app("S", "I", "I")
        omega = ["app", sii, sii]
        # omega reduces to itself forever; the engine must RETURN (budget
        # or cycle guard), not hang, and the result is honestly not a
        # normal form (an app head survives).
        out = eng.simplify(omega, max_steps=50)
        assert contains_op(out, {"app"})

    def test_terminating_term_unaffected_by_budget(self):
        eng = _engine()
        assert eng.simplify(app("S", "K", "K", "a"), max_steps=50) == "a"


class TestStrategyConfluence:
    def test_strategies_agree_on_terminating_term(self):
        eng = _engine()
        term = app("S", "K", "I", ["app", "I", "a"])
        exhaustive = eng.simplify(term)
        bottomup = eng.simplify(term, strategy="bottomup")
        assert exhaustive == bottomup == "a"
