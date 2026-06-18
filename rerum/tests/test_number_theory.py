"""Soundness tests for the examples/number_theory.rules fixture.

The headline check is the guarded ``@mod-same`` identity: ``x mod x = 0`` is
only valid for nonzero x, since ``0 mod 0`` is undefined (division by zero).
The rule carries ``when (! != :x 0)`` so it must NOT fire on ``(mod 0 0)``
while still firing on ``(mod 5 5)`` and on a symbolic ``(mod y y)``.

number_theory.rules needs a prelude that supplies the number-theory fold ops
(gcd/lcm/mod/factorial/...) as well as the guard primitives (!=, and, const?).
The stock FULL_PRELUDE carries the guard primitives but NOT gcd/lcm/mod, so the
canonical prelude for this fixture is examples/custom_prelude.py's PRELUDE.
"""

import importlib.util
from pathlib import Path

import pytest

from rerum import RuleEngine, E, FULL_PRELUDE


EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _custom_prelude():
    """Load the PRELUDE dict from examples/custom_prelude.py."""
    spec = importlib.util.spec_from_file_location(
        "_nt_custom_prelude", EXAMPLES / "custom_prelude.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PRELUDE


@pytest.fixture
def engine():
    """number_theory.rules loaded with the number-theory prelude."""
    return (
        RuleEngine()
        .with_prelude(_custom_prelude())
        .load_file(str(EXAMPLES / "number_theory.rules"))
    )


class TestModSameSoundness:
    """The guarded @mod-same identity must not fire at x = 0."""

    def test_mod_same_nonzero_fires(self, engine):
        """(mod 5 5) reduces to 0 -- the guard passes for nonzero x."""
        assert engine(E("(mod 5 5)")) == 0

    def test_mod_zero_zero_does_not_reduce_to_zero(self, engine):
        """(mod 0 0) is undefined, so the guard blocks @mod-same.

        Neither the guarded identity nor the eval-mod fold may produce a value:
        the fold raises ZeroDivisionError (caught as a non-match), so the term
        stays unreduced rather than silently becoming a (wrong) 0.
        """
        result = engine(E("(mod 0 0)"))
        assert result == ["mod", 0, 0]
        assert result != 0

    def test_mod_same_symbolic_still_fires(self, engine):
        """(mod y y) reduces to 0 -- the symbolic identity is preserved.

        The guard compares the bound symbol "y" against 0; a symbol is never
        equal to the integer 0, so the rule fires as the standard identity.
        """
        assert engine(E("(mod y y)")) == 0

    def test_mod_general_via_eval(self, engine):
        """A genuine remainder still computes through eval-mod."""
        assert engine(E("(mod 7 3)")) == 1

    def test_guard_is_present_on_mod_same(self, engine):
        """The @mod-same rule carries a when-clause condition."""
        meta = next(m for m in engine._metadata if m.name == "mod-same")
        assert meta.condition is not None
        assert meta.condition == ["!", "!=", [":", "x"], 0]


class TestEvalRulesCompute:
    """A couple of eval-* rules fold constant operands correctly."""

    def test_eval_gcd(self, engine):
        assert engine(E("(gcd 12 8)")) == 4

    def test_eval_lcm(self, engine):
        assert engine(E("(lcm 4 6)")) == 12

    def test_eval_factorial(self, engine):
        assert engine(E("(factorial 5)")) == 120

    def test_eval_floor(self, engine):
        assert engine(E("(floor 3.7)")) == 3


class TestPreludeProvenance:
    """Document why FULL_PRELUDE alone is insufficient for this fixture."""

    def test_full_prelude_lacks_number_theory_ops(self):
        """FULL_PRELUDE carries guard primitives but not gcd/lcm/mod.

        This is why the fixture is loaded with custom_prelude.py instead: the
        guard primitive (!=) is present in FULL_PRELUDE, but the fold ops the
        eval-* rules invoke are domain extensions, not core arithmetic.
        """
        assert "!=" in FULL_PRELUDE
        assert "and" in FULL_PRELUDE
        assert "const?" in FULL_PRELUDE
        assert "mod" not in FULL_PRELUDE
        assert "gcd" not in FULL_PRELUDE
