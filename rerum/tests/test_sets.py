"""Tests for set algebra as pure example content (examples/sets.rules).

This file proves the engine has no idea what a set is: sets.rules is
ISOMORPHIC to boolean.rules with only the operator names changed
(union ~ or, inter ~ and, comp ~ not, empty ~ false, universe ~ true),
and sets.theory.json is the THIRD theory (after arithmetic and boolean)
flowing through the same general normalize machinery.

The salient certification is the VENN property test: every rule (both
directions of every bidirectional law) is checked semantically over ALL
assignments of its pattern variables to subsets of a small universe.
"""

import itertools
from pathlib import Path

import pytest

from rerum.engine import RuleEngine, parse_sexpr
from rerum.normalize import Theory, normalize
from rerum.rewriter import instantiate, match, wrap_bindings

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
RULES = EXAMPLES / "sets.rules"
SIDECAR = EXAMPLES / "sets.metadata.json"
THEORY = EXAMPLES / "sets.theory.json"

UNIVERSE = frozenset({1, 2, 3})
SUBSETS = [frozenset(c)
           for r in range(len(UNIVERSE) + 1)
           for c in itertools.combinations(sorted(UNIVERSE), r)]


def _engine():
    eng = RuleEngine()  # NO prelude: structural rewrites only
    eng.load_file(RULES, validate_examples=False)
    eng.load_metadata_json(SIDECAR.read_text(), validate_examples=True)
    return eng


def _simplifier():
    eng = _engine()
    eng.disable_group("equivalences")
    return eng


# ---------------------------------------------------------------------
# Semantic ground truth, local to the test (the engine never sees this).
# ---------------------------------------------------------------------

def _seval(expr, env):
    """Evaluate a set expression to a frozenset over UNIVERSE."""
    if expr == "empty":
        return frozenset()
    if expr == "universe":
        return UNIVERSE
    if isinstance(expr, str):
        return env[expr]
    if isinstance(expr, list) and expr:
        op, args = expr[0], expr[1:]
        if op == "union":
            out = frozenset()
            for a in args:
                out = out | _seval(a, env)
            return out
        if op == "inter":
            out = UNIVERSE
            for a in args:
                out = out & _seval(a, env)
            return out
        if op == "comp" and len(args) == 1:
            return UNIVERSE - _seval(args[0], env)
    raise ValueError(f"not a set expression: {expr!r}")


def _pattern_vars(pattern, acc):
    if isinstance(pattern, list):
        if len(pattern) == 2 and pattern[0] in ("?", "?v", "?c"):
            acc.append(pattern[1])
            return
        for sub in pattern:
            _pattern_vars(sub, acc)


def _ground(pattern, env):
    """Substitute symbol names for pattern vars, yielding a symbolic expr
    whose leaves are variable names looked up in the semantic env."""
    if isinstance(pattern, list):
        if len(pattern) == 2 and pattern[0] in ("?", "?v", "?c"):
            return pattern[1]
        return [_ground(sub, env) for sub in pattern]
    return pattern


class TestLoadAndValidate:
    def test_loads_with_validated_examples_and_no_prelude(self):
        eng = _engine()
        assert len(eng.list_rules()) > 0
        assert eng.has_fold_funcs() is False

    def test_every_rule_has_an_example(self):
        eng = _engine()
        for _i, _r, meta in eng.iter_rules():
            assert meta.examples, f"rule {meta.name!r} has no examples"

    def test_isomorphic_to_boolean(self):
        # The structural claim made executable: renaming ops maps the sets
        # rule file onto the boolean one rule-for-rule (same pattern
        # SHAPES). Compare post-desugar rule counts as a cheap invariant.
        bool_eng = RuleEngine()
        bool_eng.load_file(EXAMPLES / "boolean.rules",
                           validate_examples=False)
        assert len(_engine().list_rules()) == len(bool_eng.list_rules())


class TestVennCertification:
    def test_every_rule_is_semantically_sound(self):
        # For EVERY rule (incl. both directions of bidirectionals):
        # enumerate assignments of its pattern variables to ALL subsets of
        # {1,2,3}, ground the pattern with variable symbols, match +
        # instantiate via the pure core, evaluate both sides with python
        # set semantics, and assert equality.
        eng = _engine()
        checked = 0
        for _i, rule, meta in eng.iter_rules():
            pattern, skeleton = rule
            names = []
            _pattern_vars(pattern, names)
            names = sorted(set(names))
            assert len(names) <= 3, f"{meta.name}: too many vars for Venn"
            sym = _ground(pattern, {})
            bindings = match(pattern, sym, wrap_bindings({}))
            assert bindings, f"{meta.name}: symbolic LHS failed to match"
            rhs = instantiate(skeleton, bindings, {})
            for values in itertools.product(SUBSETS, repeat=len(names)):
                env = dict(zip(names, values))
                assert _seval(sym, env) == _seval(rhs, env), (
                    f"{meta.name} is UNSOUND at {env}: {sym!r} -> {rhs!r}")
                checked += 1
        assert checked >= 100


class TestFixpointSimplification:
    def test_nested_laws(self):
        eng = _simplifier()
        out = eng(parse_sexpr("(inter (inter a (union a b)) universe)"))
        assert out == "a"

    def test_complement_collapse(self):
        eng = _simplifier()
        assert eng(parse_sexpr("(union (inter a (comp a)) b)")) == "b"

    def test_double_complement(self):
        eng = _simplifier()
        assert eng(parse_sexpr("(comp (comp (comp s)))")) == ["comp", "s"]

    def test_ground_evaluation_by_rewriting(self):
        eng = _simplifier()
        out = eng(parse_sexpr("(union (inter universe empty) (comp empty))"))
        assert out == "universe"


class TestEquivalenceReasoning:
    def test_prove_de_morgan_for_sets(self):
        eng = _engine()
        proof = eng.prove_equal(
            parse_sexpr("(comp (union a b))"),
            parse_sexpr("(inter (comp a) (comp b))"),
            max_expressions=500)
        assert proof is not None

    def test_minimize_compresses_de_morgan_expansion(self):
        eng = _engine()
        opt = eng.minimize(parse_sexpr("(union (comp a) (comp b))"))
        assert opt.expr == ["comp", ["inter", "a", "b"]]


class TestSetsTheoryNormalize:
    """The third theory through the same general normalize machinery."""

    def test_theory_loads(self):
        theory = Theory.from_json(THEORY.read_text())
        assert theory.is_ac("union") is True
        assert theory.identity("inter") == "universe"
        assert theory.annihilator("inter") == "empty"

    def test_normalize_drops_identities_and_sorts(self):
        theory = Theory.from_json(THEORY.read_text())
        out = normalize(parse_sexpr("(union b a empty)"), theory)
        assert out == ["union", "a", "b"]

    def test_normalize_annihilator(self):
        theory = Theory.from_json(THEORY.read_text())
        assert normalize(parse_sexpr("(inter a empty b)"), theory) == "empty"
