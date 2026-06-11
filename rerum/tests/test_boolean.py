"""Tests for boolean algebra as pure example content (examples/boolean.rules).

Drives the GENERAL engine through example rule data: the engine has no idea
what `and` means. The set needs NO prelude (every rewrite is structural),
constants are the SYMBOLS true / false, and the boolean theory JSON flows
through the same normalize machinery that serves arithmetic.

The salient certification is the TRUTH-TABLE property test: every rule in
the file (both directions of every bidirectional law) is checked
semantically over all assignments of its pattern variables -- the boolean
analog of the calculus checker's is_derivative.
"""

import itertools
import json
from pathlib import Path

import pytest

from rerum.engine import RuleEngine, parse_sexpr
from rerum.normalize import Theory, normalize
from rerum.rewriter import instantiate, match, wrap_bindings

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
RULES = EXAMPLES / "boolean.rules"
SIDECAR = EXAMPLES / "boolean.metadata.json"
THEORY = EXAMPLES / "boolean.theory.json"


def _engine():
    # NO prelude: the boolean set computes nothing; every rewrite is
    # structural. (Loading with examples validated proves that too -- a
    # (! op ...) compute would raise without fold_funcs.)
    eng = RuleEngine()
    eng.load_file(RULES, validate_examples=False)
    eng.load_metadata_json(SIDECAR.read_text(), validate_examples=True)
    return eng


def _simplifier():
    """The directed fixpoint core: equivalences disabled (their reverse
    directions re-grow what the core shrinks)."""
    eng = _engine()
    eng.disable_group("equivalences")
    return eng


# ---------------------------------------------------------------------
# Semantic ground truth, local to the test (the engine never sees this).
# ---------------------------------------------------------------------

def _beval(expr):
    """Evaluate a ground boolean expression over symbol constants."""
    if expr == "true":
        return True
    if expr == "false":
        return False
    if isinstance(expr, list) and expr:
        op, args = expr[0], expr[1:]
        if op == "and":
            return all(_beval(a) for a in args)
        if op == "or":
            return any(_beval(a) for a in args)
        if op == "not" and len(args) == 1:
            return not _beval(args[0])
    raise ValueError(f"not a ground boolean expression: {expr!r}")


def _pattern_vars(pattern, acc):
    """Collect pattern-variable names (?x / ?x:var / ?x:const nodes)."""
    if isinstance(pattern, list):
        if len(pattern) == 2 and pattern[0] in ("?", "?v", "?c"):
            acc.append(pattern[1])
            return
        for sub in pattern:
            _pattern_vars(sub, acc)


def _ground(pattern, env):
    """Substitute an assignment into a pattern, yielding a ground expr."""
    if isinstance(pattern, list):
        if len(pattern) == 2 and pattern[0] in ("?", "?v", "?c"):
            return env[pattern[1]]
        return [_ground(sub, env) for sub in pattern]
    return pattern


class TestLoadAndValidate:
    def test_loads_with_validated_examples_and_no_prelude(self):
        eng = _engine()
        assert len(eng.list_rules()) > 0
        assert eng.has_fold_funcs() is False  # structural rules only

    def test_every_rule_has_an_example(self):
        eng = _engine()
        for _i, _r, meta in eng.iter_rules():
            assert meta.examples, f"rule {meta.name!r} has no examples"


class TestTruthTableCertification:
    def test_every_rule_is_semantically_sound(self):
        # For EVERY rule (including both directions of each bidirectional
        # law): enumerate all true/false assignments of its pattern
        # variables, ground the pattern, match + instantiate (the pure
        # core), and check LHS and RHS agree semantically. This certifies
        # the whole file the way is_derivative certifies calculus rules.
        eng = _engine()
        checked = 0
        for _i, rule, meta in eng.iter_rules():
            pattern, skeleton = rule
            names = []
            _pattern_vars(pattern, names)
            names = sorted(set(names))
            for values in itertools.product(["true", "false"],
                                            repeat=len(names)):
                env = dict(zip(names, values))
                lhs = _ground(pattern, env)
                bindings = match(pattern, lhs, wrap_bindings({}))
                assert bindings, (
                    f"{meta.name}: grounded LHS failed to match its own "
                    f"pattern: {lhs!r}")
                rhs = instantiate(skeleton, bindings, {})
                assert _beval(lhs) == _beval(rhs), (
                    f"{meta.name} is UNSOUND at {env}: "
                    f"{lhs!r} -> {rhs!r}")
                checked += 1
        assert checked >= 50  # every rule, every assignment


class TestFixpointSimplification:
    def test_nested_simplification(self):
        eng = _simplifier()
        # (and a (or a b) true) -> absorption + identity -> a
        out = eng(parse_sexpr("(and (and a (or a b)) true)"))
        assert out == "a"

    def test_complement_collapses(self):
        eng = _simplifier()
        assert eng(parse_sexpr("(or (and a (not a)) b)")) == "b"

    def test_double_negation(self):
        eng = _simplifier()
        assert eng(parse_sexpr("(not (not (not c)))")) == ["not", "c"]

    def test_ground_expression_evaluates_by_rewriting(self):
        eng = _simplifier()
        # No prelude, yet ground expressions evaluate: pure rewriting.
        assert eng(parse_sexpr("(or (and true false) (not false))")) == "true"

    def test_every_directed_rule_fires_somewhere(self):
        # Each directed rule is exercised end-to-end (not just by its
        # sidecar example): drive an input crafted for it and check the
        # rule appears in the trace.
        eng = _simplifier()
        drivers = {
            "not-true": "(not true)",
            "not-false": "(not false)",
            "and-true": "(and a true)",
            "and-true-left": "(and true a)",
            "or-false": "(or a false)",
            "or-false-left": "(or false a)",
            "and-false": "(and a false)",
            "and-false-left": "(and false a)",
            "or-true": "(or a true)",
            "or-true-left": "(or true a)",
            "and-same": "(and a a)",
            "or-same": "(or a a)",
            "and-comp": "(and a (not a))",
            "and-comp-left": "(and (not a) a)",
            "or-comp": "(or a (not a))",
            "or-comp-left": "(or (not a) a)",
            "double-neg": "(not (not a))",
            "and-absorb": "(and a (or a b))",
            "and-absorb-left": "(and (or a b) a)",
            "or-absorb": "(or a (and a b))",
            "or-absorb-left": "(or (and a b) a)",
        }
        for name, src in drivers.items():
            _result, trace = eng.simplify(parse_sexpr(src), trace=True)
            fired = [s.metadata.name for s in trace.steps]
            assert name in fired, f"{name} did not fire on {src}"


class TestEquivalenceReasoning:
    def test_prove_de_morgan_consequence(self):
        eng = _engine()  # full set, equivalences enabled
        proof = eng.prove_equal(
            parse_sexpr("(not (and a b))"),
            parse_sexpr("(or (not a) (not b))"),
            max_expressions=500)
        assert proof is not None

    def test_minimize_finds_smaller_de_morgan_form(self):
        eng = _engine()
        # (or (not a) (not b)) [5 nodes] <=> (not (and a b)) [4 nodes].
        opt = eng.minimize(parse_sexpr("(or (not a) (not b))"))
        assert opt.expr == ["not", ["and", "a", "b"]]
        assert opt.cost < opt.original_cost

    def test_minimize_derivation_chains(self):
        eng = _engine()
        opt = eng.minimize(parse_sexpr("(or (not a) (not b))"))
        if opt.derivation is not None:
            seq = opt.derivation.to_global_sequence()
            for k in range(len(seq) - 1):
                assert seq[k]["after_root"] == seq[k + 1]["before_root"]


class TestBooleanTheoryNormalize:
    """The executable swap test: the same general normalize machinery that
    canonicalizes arithmetic, fed boolean DATA."""

    def test_theory_loads(self):
        theory = Theory.from_json(THEORY.read_text())
        assert theory.is_ac("and") is True
        assert theory.identity("and") == "true"
        assert theory.annihilator("or") == "true"

    def test_normalize_drops_identities_and_sorts(self):
        theory = Theory.from_json(THEORY.read_text())
        out = normalize(parse_sexpr("(or b a false)"), theory)
        assert out == ["or", "a", "b"]

    def test_normalize_is_permutation_invariant(self):
        theory = Theory.from_json(THEORY.read_text())
        forms = ["(and c true a b)", "(and b a c)", "(and a true b c)"]
        outs = [normalize(parse_sexpr(f), theory) for f in forms]
        assert outs[0] == outs[1] == outs[2]

    def test_normalize_annihilator(self):
        theory = Theory.from_json(THEORY.read_text())
        assert normalize(parse_sexpr("(and a false b)"), theory) == "false"
