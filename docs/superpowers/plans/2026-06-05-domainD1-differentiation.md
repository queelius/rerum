# Domain D1: Differentiation Example Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Demonstrate that a whole symbolic domain (differentiation) is nothing but rules plus data by shipping a COMPLETE single- and multi-variable differentiation rule set over the `["dd", f, v]` operator entirely under `examples/`: `examples/differentiation.rules` (every rule family, with load-validated `{in, out}` examples carried in a `examples/differentiation.metadata.json` sidecar), `examples/arithmetic.theory.json` (declares `+`/`*` AC with identities so the general `normalize` cleans derivative output), and `examples/calculus_checker.py` (a domain validator `is_derivative` built ON TOP of the general `rerum.numeval`/`numeric_equiv`, never imported by the engine). Loaded under `combine_preludes(MATH_PRELUDE, PREDICATE_PRELUDE)` with the arithmetic theory set, the existing `simplify` driver produces CLEAN output: `d/dx(x*x) -> (* 2 x)`, `d/dx(x^3) -> (* 3 (^ x 2))`.

**Architecture:** This phase adds NO engine code. Every artifact lands under `examples/` (rule file, theory JSON, metadata sidecar, domain checker) or `rerum/tests/` (the example-exercising tests). It is a demonstration that a domain is just rules + data: differentiation is CONFLUENT, so it runs on the EXISTING `simplify` fixpoint driver with no search and no domain-named engine module. The pieces are:
1. `examples/differentiation.rules`: the canonical, human-readable DSL rule file over `["dd", f, v]` (rest-patterns for n-ary linearity; priorities so the constant-exponent power rule beats the general `f^g` log-diff rule; the `free-of?` predicate guard for partials). The DSL has NO `examples` annotation (only `{category=...}`), so the `{in, out}` examples live in a sidecar `examples/differentiation.metadata.json` merged via `engine.load_metadata_json(text, validate_examples=True)`. Rules stay readable as DSL while every rule still carries examples validated at load (v0.7 metadata layer).
2. `examples/arithmetic.theory.json`: declares `+` and `*` as associative-commutative with their identities/annihilator. This is the Phase 2 theory DATA the general `normalize` machinery consumes via `engine.with_theory(Theory.from_json(...))`; the SAME machinery serves a hypothetical `boolean.theory.json`. Created here if Phase 2 has not already created it; referenced if it has.
3. `examples/calculus_checker.py`: `is_derivative(expr, var, result, *, samples=8, tol=1e-6) -> bool` built ON TOP of the GENERAL `rerum.numeval.numeval`/`numeric_equiv` (Phase 3). It finite-difference-evaluates `d(expr)/d(var)` numerically and compares to evaluating `result`, using `numeval` over a numeric prelude. This is example content; the engine never imports it. There is NO core `rerum/verify.py`.

The swap test holds: replace "differentiation" with "boolean algebra" and nothing in `rerum/` changes, because nothing in `rerum/` changes at all in this phase.

**Tech Stack:** Python 3.9+, pytest with plain asserts (config in `pyproject.toml`: `testpaths = ["rerum/tests"]`, `addopts = "-v"`). Tests in `rerum/tests/test_differentiation.py`. Each differentiation task adds a rule family (DSL rule + sidecar example) AND a test that differentiates an example end-to-end through `simplify` + `normalize` AND confirms it numerically with `is_derivative`. The combined-load helper `make_diff_engine()` lives in the test module so the motivating pipeline is exercised end to end through the example content.

**Dependency note (read before starting):** This phase consumes engine capabilities delivered earlier and adds none of its own:
- Phase 0: the `free-of?` predicate in `PREDICATE_PRELUDE`, the `?x:free(v)` binding-order fix, the guard-on-undefined-op raise, and `combine_preludes(*preludes)` exported from `rerum/__init__.py`.
- Phase 2: `rerum.normalize.normalize(expr, theory)` and `rerum.normalize.Theory` (`Theory.from_json(text)`), plus `RuleEngine.with_theory(theory)`.
- Phase 3: `rerum.numeval.numeval(expr, env, prelude)` and `rerum.numeval.numeric_equiv(...)`.

If `combine_preludes`, `normalize`/`Theory`/`with_theory`, or `numeval` are not yet importable, the corresponding imports fail: complete Phases 0, 2, and 3 first. This is a domain-demonstration phase (D1) and lands after those engine phases per the contract.

---

## File Structure

```
examples/
  differentiation.rules              (NEW - canonical DSL rule file over ["dd", f, v])
  differentiation.metadata.json      (NEW - sidecar {in,out} examples + category, validated at load)
  arithmetic.theory.json             (NEW here, or reuse Phase 2 - +/* AC + identities)
  calculus_checker.py                (NEW - is_derivative on rerum.numeval; example content only)
  algebra.rules                      (read-only: existing algebraic-simplification rules)
rerum/
  engine.py                          (read-only: with_prelude, with_theory, load_file, load_metadata_json, simplify/__call__)
  normalize.py                       (read-only: normalize, Theory - Phase 2)
  numeval.py                         (read-only: numeval, numeric_equiv - Phase 3)
  rewriter.py                        (read-only: MATH_PRELUDE, PREDICATE_PRELUDE, free-of? - Phase 0)
  __init__.py                        (read-only: combine_preludes, MATH_PRELUDE, PREDICATE_PRELUDE, Theory)
  tests/
    test_differentiation.py          (NEW - this phase; loads the example files, asserts each family)
```

NO `rerum/` core files are created or edited in this phase. The only non-`examples/` artifact is the test file, which exercises the ENGINE through the example content (it would be deleted or swapped if the example changed, and the engine would not).

`examples/calculus_checker.py` public surface (contract-verbatim for `is_derivative`):

```
is_derivative(expr, var, result, *, samples=8, tol=1e-6) -> bool
```

`examples/differentiation.rules` operator and conventions:
- Operator `["dd", f, v]` = d f / d v.
- Priorities (higher fires first): the partial-derivative `free-of?` guard at `[110]`, the constant/variable base cases at `[100]`, the constant-exponent power rule and constant-base `a^x` rule at `[60]`, and the general `f^g` log-diff rule at `[50]` (so `(dd (^ x 3) x)` takes the constant-power rule, `(dd (^ 2 x) x)` takes `a^x`, and `(dd (^ x x) x)` takes log-diff).
- Linearity uses rest-patterns: `(dd (+ ?f ?rest...) ?v) => (+ (dd :f :v) (dd (+ :rest...) :v))`.

**Single-step example semantics (load-bearing):** `load_metadata_json(..., validate_examples=True)` validates ONE match + ONE `instantiate` of the rule, NOT a full simplification. So each sidecar example's `out` is the rule's direct single-step rewrite output (with `(! op ...)` folds in the skeleton evaluated), with inner `(dd ...)` left UNREDUCED. Example: for `dd-sin`, `(dd (sin x) x)` validates to `(* (cos x) (dd x x))`, not `(* (cos x) 1)`.

**Loading pattern (the motivating pipeline):**

```python
engine = (
    RuleEngine()
    .with_prelude(combine_preludes(MATH_PRELUDE, PREDICATE_PRELUDE))
    .with_theory(Theory.from_json(ARITH_THEORY.read_text()))
    .load_file(ALGEBRA_RULES)
    .load_file(DIFF_RULES)
)
engine.load_metadata_json(DIFF_SIDECAR.read_text(), validate_examples=True)
```

Then `engine(parse_sexpr(src))` (i.e. `simplify`) with the theory set normalizes between steps and yields clean output. No `solve`, no domain engine code.

---

### Task 1: `examples/calculus_checker.py` - `is_derivative` on the general `numeval`

This is the verification artifact RELOCATED out of the engine: there is NO `rerum/verify.py`. The checker is example content built ON TOP of the general `rerum.numeval`/`numeric_equiv` primitives. Build it first so every later differentiation family leans on it. Its tests run with no rule files, so they have no Phase 2 dependency.

**Files:**
- Create: `examples/calculus_checker.py`
- Test: `rerum/tests/test_differentiation.py` (create with the checker tests; later tasks extend it)

- [ ] **Step 1: Write the failing test for `is_derivative` on elementary and transcendental derivatives.**
  Create `rerum/tests/test_differentiation.py`:

  ```python
  """Tests for examples/differentiation.rules + examples/calculus_checker.py.

  These tests exercise the GENERAL engine through example content: they load
  the example rule files, theory, and metadata sidecar, differentiate concrete
  expressions on the existing ``simplify`` driver with a ``normalize`` finishing
  pass, and confirm each answer numerically with the example checker
  ``is_derivative`` (built on the general ``rerum.numeval``). They would be
  deleted or swapped if the example changed; the engine would not.
  """

  import importlib.util
  from pathlib import Path

  import pytest

  from rerum import RuleEngine, MATH_PRELUDE, PREDICATE_PRELUDE, combine_preludes
  from rerum.engine import parse_sexpr, format_sexpr
  from rerum.normalize import normalize, Theory

  EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
  DIFF_RULES = EXAMPLES_DIR / "differentiation.rules"
  DIFF_SIDECAR = EXAMPLES_DIR / "differentiation.metadata.json"
  ALGEBRA_RULES = EXAMPLES_DIR / "algebra.rules"
  ARITH_THEORY = EXAMPLES_DIR / "arithmetic.theory.json"
  CHECKER_PATH = EXAMPLES_DIR / "calculus_checker.py"


  def _load_checker():
      """Import examples/calculus_checker.py by path (it is example content,
      never importable as a package module)."""
      spec = importlib.util.spec_from_file_location("calculus_checker",
                                                    CHECKER_PATH)
      module = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(module)
      return module


  class TestCheckerElementary:
      """is_derivative on hand-computed elementary derivatives."""

      def test_constant_derivative_is_zero(self):
          checker = _load_checker()
          assert checker.is_derivative("5", "x", "0") is True

      def test_identity_derivative_is_one(self):
          checker = _load_checker()
          assert checker.is_derivative("x", "x", "1") is True

      def test_power_rule(self):
          checker = _load_checker()
          assert checker.is_derivative("(^ x 3)", "x", "(* 3 (^ x 2))") is True

      def test_product_x_squared(self):
          checker = _load_checker()
          assert checker.is_derivative("(* x x)", "x", "(* 2 x)") is True

      def test_sum_derivative(self):
          checker = _load_checker()
          assert checker.is_derivative("(+ (^ x 2) x)", "x",
                                       "(+ (* 2 x) 1)") is True

      def test_wrong_derivative_rejected(self):
          checker = _load_checker()
          assert checker.is_derivative("(^ x 3)", "x",
                                       "(* 2 (^ x 2))") is False


  class TestCheckerTranscendental:
      """is_derivative across the transcendental families and domain skipping."""

      def test_sin(self):
          checker = _load_checker()
          assert checker.is_derivative("(sin x)", "x", "(cos x)") is True

      def test_cos(self):
          checker = _load_checker()
          assert checker.is_derivative("(cos x)", "x", "(- (sin x))") is True

      def test_tan(self):
          checker = _load_checker()
          assert checker.is_derivative("(tan x)", "x", "(^ (sec x) 2)") is True

      def test_exp(self):
          checker = _load_checker()
          assert checker.is_derivative("(exp x)", "x", "(exp x)") is True

      def test_ln(self):
          checker = _load_checker()
          assert checker.is_derivative("(ln x)", "x", "(/ 1 x)") is True

      def test_sqrt(self):
          checker = _load_checker()
          assert checker.is_derivative("(sqrt x)", "x",
                                       "(/ 1 (* 2 (sqrt x)))") is True

      def test_asin(self):
          checker = _load_checker()
          assert checker.is_derivative("(asin x)", "x",
                                       "(/ 1 (sqrt (- 1 (^ x 2))))") is True

      def test_atan(self):
          checker = _load_checker()
          assert checker.is_derivative("(atan x)", "x",
                                       "(/ 1 (+ 1 (^ x 2)))") is True

      def test_sinh(self):
          checker = _load_checker()
          assert checker.is_derivative("(sinh x)", "x", "(cosh x)") is True

      def test_tanh(self):
          checker = _load_checker()
          assert checker.is_derivative("(tanh x)", "x",
                                       "(- 1 (^ (tanh x) 2))") is True

      def test_log_base(self):
          checker = _load_checker()
          assert checker.is_derivative("(log 2 x)", "x",
                                       "(/ 1 (* x (ln 2)))") is True

      def test_wrong_transcendental_rejected(self):
          checker = _load_checker()
          assert checker.is_derivative("(sin x)", "x", "(sin x)") is False
  ```

- [ ] **Step 2: Run the test, expect FAIL (checker file missing).**
  ```bash
  pytest rerum/tests/test_differentiation.py::TestCheckerElementary -v
  ```
  Expected: every test errors because `examples/calculus_checker.py` does not exist (the `spec_from_file_location` loader fails / `FileNotFoundError`).

- [ ] **Step 3: Implement `examples/calculus_checker.py` on the general `numeval`.**
  Create `examples/calculus_checker.py`:

  ```python
  """Domain validators for the calculus example, built on the GENERAL numeric
  primitives ``rerum.numeval`` / ``rerum.numeric_equiv``.

  This file is EXAMPLE CONTENT. The engine never imports it. It encodes the
  domain semantics that a derivative result must match the finite-difference of
  the input, which is calculus knowledge that has no place in the engine. It is
  passed to ``rerum.training.generate_corpus`` as the ``checker`` for the
  differentiation demonstration.

  ``is_derivative(expr, var, result)`` evaluates the central finite difference
  of ``expr`` w.r.t. ``var`` and the claimed ``result`` at random sample points
  via ``numeval`` over a numeric prelude, and accepts when they agree at every
  usable point. Points where a function is undefined (``ln`` of a non-positive
  number, ``sqrt`` of a negative, division by zero, ``asin``/``acos`` outside
  [-1, 1], etc.) are skipped via ``_DomainError`` so a correct derivative is not
  rejected for an unlucky draw.
  """

  import math
  import random
  from fractions import Fraction

  from rerum.expr import parse_sexpr
  from rerum.numeval import numeval


  # ------------------------------------------------------------------
  # Numeric prelude for the checker (the fold functions numeval interprets).
  # GENERAL primitives; the calculus knowledge is only in is_derivative below.
  # ------------------------------------------------------------------

  class _DomainError(Exception):
      """An op evaluated outside its real domain at a sample point. Caught by
      the sampler so the offending point is skipped, never to reject a correct
      derivative for an unlucky draw."""


  def _ln(x):
      if x <= 0.0:
          raise _DomainError("ln of non-positive")
      return math.log(x)


  def _sqrt(x):
      if x < 0.0:
          raise _DomainError("sqrt of negative")
      return math.sqrt(x)


  def _asin(x):
      if x < -1.0 or x > 1.0:
          raise _DomainError("asin out of domain")
      return math.asin(x)


  def _acos(x):
      if x < -1.0 or x > 1.0:
          raise _DomainError("acos out of domain")
      return math.acos(x)


  def _sec(x):
      c = math.cos(x)
      if c == 0.0:
          raise _DomainError("sec undefined")
      return 1.0 / c


  def _csc(x):
      s = math.sin(x)
      if s == 0.0:
          raise _DomainError("csc undefined")
      return 1.0 / s


  def _cot(x):
      t = math.tan(x)
      if t == 0.0:
          raise _DomainError("cot undefined")
      return 1.0 / t


  def _div(a, b):
      if b == 0.0:
          raise _DomainError("division by zero")
      return a / b


  def _pow(base, exp):
      if base < 0.0 and not float(exp).is_integer():
          raise _DomainError("negative base, fractional exponent")
      if base == 0.0 and exp < 0.0:
          raise _DomainError("zero base, negative exponent")
      try:
          return math.pow(base, exp)
      except (ValueError, OverflowError):
          raise _DomainError("pow out of domain")


  def _minus(*args):
      if len(args) == 1:
          return -args[0]
      acc = args[0]
      for a in args[1:]:
          acc -= a
      return acc


  def _plus(*args):
      total = 0.0
      for a in args:
          total += a
      return total


  def _times(*args):
      prod = 1.0
      for a in args:
          prod *= a
      return prod


  def _log(base, x):
      # Two-arg log: log base ``base`` of ``x``.
      if base <= 0.0 or base == 1.0 or x <= 0.0:
          raise _DomainError("log base/argument out of domain")
      return math.log(x) / math.log(base)


  # The numeric prelude numeval interprets. Named by what it computes.
  NUMERIC_PRELUDE = {
      "+": _plus,
      "*": _times,
      "-": _minus,
      "/": _div,
      "^": _pow,
      "sin": math.sin,
      "cos": math.cos,
      "tan": math.tan,
      "asin": _asin,
      "acos": _acos,
      "atan": math.atan,
      "sinh": math.sinh,
      "cosh": math.cosh,
      "tanh": math.tanh,
      "exp": math.exp,
      "ln": _ln,
      "sqrt": _sqrt,
      "sec": _sec,
      "csc": _csc,
      "cot": _cot,
      "log": _log,
  }


  # ------------------------------------------------------------------
  # The domain validator. THIS is the calculus knowledge.
  # ------------------------------------------------------------------

  def _free_symbols(expr, acc):
      """Collect variable symbols (non-numeric atoms that are not op heads)."""
      if isinstance(expr, str):
          acc.add(expr)
          return
      if isinstance(expr, list) and expr:
          for a in expr[1:]:
              _free_symbols(a, acc)


  def _numeval(expr, env):
      """Evaluate a ground term with the checker's numeric prelude, surfacing
      the _DomainError raised by an out-of-domain op."""
      return numeval(expr, env, NUMERIC_PRELUDE)


  def _central_difference(expr, env, var, h):
      """Central finite difference d expr / d var at the point in ``env``."""
      env_plus = dict(env)
      env_minus = dict(env)
      env_plus[var] = env[var] + h
      env_minus[var] = env[var] - h
      return (_numeval(expr, env_plus) - _numeval(expr, env_minus)) / (2.0 * h)


  def is_derivative(expr, var, result, *, samples=8, tol=1e-6) -> bool:
      """Numerically check that ``result`` is d(``expr``)/d(``var``).

      ``expr`` and ``result`` are s-expression strings (or already-parsed nested
      lists). For ``samples`` random points (a fixed-seed RNG for determinism),
      evaluate the central finite difference of ``expr`` w.r.t. ``var`` via the
      general ``numeval`` and compare to ``result`` evaluated at the same point;
      the derivative is accepted when the relative-or-absolute error is within
      ``tol`` at every usable point. Points where either side is undefined (a
      ``_DomainError``) are skipped. Returns ``True`` only if at least one point
      was usable and all usable points agreed.
      """
      if isinstance(expr, str):
          expr = parse_sexpr(expr)
      if isinstance(result, str):
          result = parse_sexpr(result)

      syms = set()
      _free_symbols(expr, syms)
      _free_symbols(result, syms)
      syms.add(var)
      syms = sorted(syms)

      rng = random.Random(0xD1FF)  # deterministic
      h = 1e-5                     # finite-difference step
      checked = 0

      for _ in range(samples * 4):  # extra draws to absorb skipped points
          if checked >= samples:
              break
          env = {s: rng.uniform(0.25, 1.75) for s in syms}
          try:
              fd = _central_difference(expr, env, var, h)
              claimed = _numeval(result, env)
          except _DomainError:
              continue
          except (KeyError, ValueError):
              # Structural problems (an unreduced (dd ...) head, an unbound var)
              # are real failures, not domain skips.
              return False
          checked += 1
          denom = max(1.0, abs(fd), abs(claimed))
          if abs(fd - claimed) > tol * denom:
              return False

      return checked > 0
  ```

  Note: `numeval` is the GENERAL Phase 3 primitive that interprets a ground term over a prelude. If `numeval` raises its own out-of-domain signal for a given prelude function (rather than letting `_DomainError` propagate), wrap the `_numeval` call to re-raise as `_DomainError`; do not weaken the checker. The prelude functions here raise `_DomainError` directly, which `numeval` propagates to the sampler.

- [ ] **Step 4: Run the test, expect PASS.**
  ```bash
  pytest rerum/tests/test_differentiation.py::TestCheckerElementary rerum/tests/test_differentiation.py::TestCheckerTranscendental -v
  ```
  Expected: all pass. If a transcendental fails, the bug is a missing prelude entry or an out-of-domain sample (the (0.25, 1.75) range avoids the nearest `sec`/`csc`/`cot`/`tan` poles; `_DomainError` skips any unlucky point). Fix the prelude/range, do NOT weaken the test.

- [ ] **Step 5: Commit.**
  ```bash
  git add examples/calculus_checker.py rerum/tests/test_differentiation.py
  git commit -m "feat(examples): calculus_checker.is_derivative on general numeval

  is_derivative is example content (not engine code): it checks a claimed
  derivative by central finite difference at random sample points, evaluating
  both sides via the general rerum.numeval over a numeric prelude and skipping
  out-of-domain draws via _DomainError. The engine never imports it. There is
  no core rerum/verify.py; verification lives entirely under examples/.

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task 2: `arithmetic.theory.json` and the engine-loading fixture

Create the arithmetic theory DATA the general `normalize` consumes, and the test fixture that loads differentiation + algebra under the combined prelude with that theory set. This is the pipeline the rest of the phase exercises.

**Files:**
- Create: `examples/arithmetic.theory.json` (if Phase 2 has not already created it)
- Test: `rerum/tests/test_differentiation.py` (extend with the loader fixture and a load test)

- [ ] **Step 1: Write the failing test for the loader and the theory.**
  Append to `rerum/tests/test_differentiation.py`:

  ```python
  def make_diff_engine():
      """Load differentiation + algebra under combine_preludes(MATH_PRELUDE,
      PREDICATE_PRELUDE) with the arithmetic theory set, examples validated via
      the metadata sidecar. This is the motivating pipeline, exercised end to
      end through the example content."""
      theory = Theory.from_json(ARITH_THEORY.read_text())
      engine = (
          RuleEngine()
          .with_prelude(combine_preludes(MATH_PRELUDE, PREDICATE_PRELUDE))
          .with_theory(theory)
          .load_file(ALGEBRA_RULES)
          .load_file(DIFF_RULES)
      )
      engine.load_metadata_json(DIFF_SIDECAR.read_text(), validate_examples=True)
      return engine


  def differentiate(engine, src):
      """Run the existing simplify driver (engine call) then a normalize
      finishing pass under the arithmetic theory."""
      theory = Theory.from_json(ARITH_THEORY.read_text())
      simplified = engine(parse_sexpr(src))
      return normalize(simplified, theory)


  class TestTheoryAndLoad:
      def test_theory_declares_plus_times_ac(self):
          theory = Theory.from_json(ARITH_THEORY.read_text())
          assert theory.is_ac("+") is True
          assert theory.is_ac("*") is True
          assert theory.identity("+") == 0
          assert theory.identity("*") == 1
          assert theory.annihilator("*") == 0

      def test_rules_load_and_examples_validate(self):
          # Loading the sidecar with validate_examples=True must not raise:
          # every rule's example is a correct single-step rewrite.
          engine = make_diff_engine()
          assert len(engine) > 0
  ```

- [ ] **Step 2: Run the test, expect FAIL (theory file / rule files missing).**
  ```bash
  pytest rerum/tests/test_differentiation.py::TestTheoryAndLoad -v
  ```
  Expected: `FileNotFoundError` for `examples/arithmetic.theory.json` (and, on the second test, for `examples/differentiation.rules` / the sidecar).

- [ ] **Step 3: Create `examples/arithmetic.theory.json`.**
  Create `examples/arithmetic.theory.json` (skip if Phase 2 already created an identical file; otherwise this is the canonical content):

  ```json
  {
    "+": {"ac": true, "identity": 0},
    "*": {"ac": true, "identity": 1, "annihilator": 0}
  }
  ```

- [ ] **Step 4: Run the theory test alone, expect PASS.**
  ```bash
  pytest rerum/tests/test_differentiation.py::TestTheoryAndLoad::test_theory_declares_plus_times_ac -v
  ```
  Expected: pass. The second load test (`test_rules_load_and_examples_validate`) still fails until the rule file and sidecar exist (Task 3). That is expected; the rule files arrive next.

- [ ] **Step 5: Commit.**
  ```bash
  git add examples/arithmetic.theory.json rerum/tests/test_differentiation.py
  git commit -m "feat(examples): arithmetic.theory.json + diff-engine loading fixture

  Declares + and * as associative-commutative with identities (0, 1) and *'s
  annihilator (0). This is the Phase 2 theory DATA the general normalize
  machinery consumes via engine.with_theory(Theory.from_json(...)); the same
  machinery serves a boolean.theory.json. make_diff_engine() loads
  differentiation + algebra under combine_preludes(MATH_PRELUDE,
  PREDICATE_PRELUDE) with the theory set.

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task 3: `differentiation.rules` basics + linearity (constants, variables, sum, difference, constant multiple)

Start the rule file with the base cases and linearity, using rest-patterns for variadic sums, plus the sidecar examples. Each rule's example `out` is its SINGLE-STEP rewrite (inner `dd` left unreduced).

**Files:**
- Create: `examples/differentiation.rules`
- Create: `examples/differentiation.metadata.json`
- Test: `rerum/tests/test_differentiation.py` (extend)

- [ ] **Step 1: Write the failing test for basics + linearity.**
  Append to `rerum/tests/test_differentiation.py`:

  ```python
  class TestBasicsAndLinearity:
      def test_constant(self):
          engine = make_diff_engine()
          assert differentiate(engine, "(dd 5 x)") == 0

      def test_variable_same(self):
          engine = make_diff_engine()
          assert differentiate(engine, "(dd x x)") == 1

      def test_variable_other(self):
          engine = make_diff_engine()
          assert differentiate(engine, "(dd y x)") == 0

      def test_sum(self):
          engine = make_diff_engine()
          checker = _load_checker()
          out = differentiate(engine, "(dd (+ x x) x)")
          assert out == 2
          assert checker.is_derivative("(+ x x)", "x", format_sexpr(out)) is True

      def test_difference(self):
          engine = make_diff_engine()
          checker = _load_checker()
          out = differentiate(engine, "(dd (- x 5) x)")
          assert out == 1
          assert checker.is_derivative("(- x 5)", "x", format_sexpr(out)) is True

      def test_constant_multiple(self):
          engine = make_diff_engine()
          checker = _load_checker()
          out = differentiate(engine, "(dd (* 3 x) x)")
          assert out == 3
          assert checker.is_derivative("(* 3 x)", "x", format_sexpr(out)) is True

      def test_nary_sum(self):
          engine = make_diff_engine()
          # d/dx(x + y + x) = 2 (y free of x -> 0); rest-pattern linearity.
          out = differentiate(engine, "(dd (+ x y x) x)")
          assert out == 2
  ```

- [ ] **Step 2: Run the test, expect FAIL (rule file / sidecar missing).**
  ```bash
  pytest rerum/tests/test_differentiation.py::TestBasicsAndLinearity -v
  ```
  Expected: `FileNotFoundError` for `examples/differentiation.rules` (or the sidecar).

- [ ] **Step 3: Create `examples/differentiation.rules` with the basics + linearity.**
  Create `examples/differentiation.rules`:

  ```
  # Differentiation rules over the dd operator: ["dd", f, v] = d f / d v.
  # Load with combine_preludes(MATH_PRELUDE, PREDICATE_PRELUDE) and the
  # arithmetic theory (engine.with_theory). Compose with algebra.rules; the
  # general normalize gives clean output. Examples live in
  # differentiation.metadata.json (validated at load via
  # engine.load_metadata_json), because the DSL supports only {category=...}.
  # Priorities: partial-free guard [110] > base cases [100] > constant-power
  # and constant-base [60] > general power/log-diff [50].

  [partials]
  # A subexpression free of the differentiation variable differentiates to 0.
  # Highest priority so a v-free product/quotient/compound short-circuits to 0
  # before the structural rules below decompose it.
  @dd-free[110] "Free-of-v subexpression": (dd ?f ?v:var) => 0 when (! free-of? :f :v)

  [basics]
  # Constants and variables.
  @dd-const[100] "d/dv(c) = 0": (dd ?c:const ?v:var) => 0
  @dd-var-same[100] "d/dx(x) = 1": (dd ?x:var ?x) => 1
  @dd-var-diff[90] "d/dx(y) = 0 when y != x": (dd ?y:var ?x:var) => 0

  [linearity]
  # Sum, n-ary via rest-pattern.
  @dd-sum "Sum rule (n-ary)": (dd (+ ?f ?rest...) ?v:var) => (+ (dd :f :v) (dd (+ :rest...) :v))
  @dd-diff "Difference rule": (dd (- ?f ?g) ?v:var) => (- (dd :f :v) (dd :g :v))
  # Constant multiple (constant on either side).
  @dd-const-mult "Constant multiple (left)": (dd (* ?c:const ?f) ?v:var) => (* :c (dd :f :v))
  @dd-mult-const "Constant multiple (right)": (dd (* ?f ?c:const) ?v:var) => (* :c (dd :f :v))
  ```

- [ ] **Step 4: Create `examples/differentiation.metadata.json` with the sidecar examples.**
  Each `out` is the SINGLE-STEP rewrite (inner `dd` unreduced). Create `examples/differentiation.metadata.json`:

  ```json
  {
    "dd-free": {
      "category": "partial",
      "examples": [{"in": "(dd (sin y) x)", "out": "0"}]
    },
    "dd-const": {
      "category": "basic",
      "examples": [{"in": "(dd 5 x)", "out": "0"}]
    },
    "dd-var-same": {
      "category": "basic",
      "examples": [{"in": "(dd x x)", "out": "1"}]
    },
    "dd-var-diff": {
      "category": "basic",
      "examples": [{"in": "(dd y x)", "out": "0"}]
    },
    "dd-sum": {
      "category": "linearity",
      "examples": [{"in": "(dd (+ x y) x)", "out": "(+ (dd x x) (dd (+ y) x))"}]
    },
    "dd-diff": {
      "category": "linearity",
      "examples": [{"in": "(dd (- x 5) x)", "out": "(- (dd x x) (dd 5 x))"}]
    },
    "dd-const-mult": {
      "category": "linearity",
      "examples": [{"in": "(dd (* 3 x) x)", "out": "(* 3 (dd x x))"}]
    },
    "dd-mult-const": {
      "category": "linearity",
      "examples": [{"in": "(dd (* x 3) x)", "out": "(* 3 (dd x x))"}]
    }
  }
  ```

  Note on `dd-sum`: the rest-pattern `?rest...` binds the tail `[y]`, so the single-step `out` is `(+ (dd x x) (dd (+ y) x))` (the `(+ y)` is the spliced rest re-wrapped). This is exactly what `instantiate` produces; the engine reduces `(+ y)` later.

- [ ] **Step 5: Run the test, expect PASS.**
  ```bash
  pytest rerum/tests/test_differentiation.py::TestTheoryAndLoad rerum/tests/test_differentiation.py::TestBasicsAndLinearity -v
  ```
  Expected: all pass. The sidecar validates every example as a correct single-step rewrite; the end-to-end tests confirm clean simplified output under the theory-driven normalize.

- [ ] **Step 6: Commit.**
  ```bash
  git add examples/differentiation.rules examples/differentiation.metadata.json rerum/tests/test_differentiation.py
  git commit -m "feat(examples): differentiation.rules basics + linearity, with examples

  Constant/variable base cases, n-ary sum (rest-pattern), difference, and
  constant-multiple rules over the dd operator. A free-of? guard at top
  priority sends any v-free subexpression to 0 (partial derivatives). Examples
  live in a metadata sidecar validated at load via load_metadata_json, since
  the DSL supports only {category=...}.

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task 4: Product, quotient, and constant-exponent power rules

Add the multiplicative rules and the constant-exponent power rule (which must out-rank the general `f^g` rule added in Task 8). These produce the two motivating clean-output cases under the theory-driven `normalize`.

**Files:**
- Edit: `examples/differentiation.rules` (append `[products]` group)
- Edit: `examples/differentiation.metadata.json`
- Test: `rerum/tests/test_differentiation.py` (extend)

- [ ] **Step 1: Write the failing test for products/quotient/power.**
  Append to `rerum/tests/test_differentiation.py`:

  ```python
  class TestProductsQuotientPower:
      def test_product_x_times_x(self):
          engine = make_diff_engine()
          checker = _load_checker()
          # THE motivating example: d/dx(x*x) = 2x
          out = differentiate(engine, "(dd (* x x) x)")
          assert out == ["*", 2, "x"]
          assert checker.is_derivative("(* x x)", "x", format_sexpr(out)) is True

      def test_product_general(self):
          engine = make_diff_engine()
          checker = _load_checker()
          # d/dx(x * sin x) = sin x + x cos x ; verify numerically (shape may vary).
          out = differentiate(engine, "(dd (* x (sin x)) x)")
          assert checker.is_derivative("(* x (sin x))", "x",
                                       format_sexpr(out)) is True

      def test_quotient(self):
          engine = make_diff_engine()
          checker = _load_checker()
          out = differentiate(engine, "(dd (/ x (sin x)) x)")
          assert checker.is_derivative("(/ x (sin x))", "x",
                                       format_sexpr(out)) is True

      def test_power_constant_exponent(self):
          engine = make_diff_engine()
          checker = _load_checker()
          # THE motivating example: d/dx(x^3) = 3 x^2
          out = differentiate(engine, "(dd (^ x 3) x)")
          assert out == ["*", 3, ["^", "x", 2]]
          assert checker.is_derivative("(^ x 3)", "x", format_sexpr(out)) is True

      def test_power_quadratic(self):
          engine = make_diff_engine()
          checker = _load_checker()
          out = differentiate(engine, "(dd (^ x 2) x)")
          assert out == ["*", 2, "x"]
          assert checker.is_derivative("(^ x 2)", "x", format_sexpr(out)) is True
  ```

- [ ] **Step 2: Run the test, expect FAIL.**
  ```bash
  pytest rerum/tests/test_differentiation.py::TestProductsQuotientPower -v
  ```
  Expected: failures because the product/quotient/power rules are not yet in the file (e.g. `(dd (* x x) x)` does not reduce, leaving a `dd` in the result).

- [ ] **Step 3: Append the `[products]` group to `examples/differentiation.rules`.**
  Add after the `[linearity]` group:

  ```
  [products]
  # Product rule: d/dx(f g) = f' g + f g'.
  @dd-product "Product rule": (dd (* ?f ?g) ?v:var) => (+ (* (dd :f :v) :g) (* :f (dd :g :v)))
  # Quotient rule: d/dx(f/g) = (f' g - f g') / g^2.
  @dd-quotient "Quotient rule": (dd (/ ?f ?g) ?v:var) => (/ (- (* (dd :f :v) :g) (* :f (dd :g :v))) (^ :g 2))
  # Power rule, constant exponent: d/dx(f^n) = n f^(n-1) f'. Priority 60 so it
  # beats the general f^g log-diff rule (priority 50) when the exponent is a
  # number. The (! - :n 1) fold computes the decremented exponent at rewrite time.
  @dd-power[60] "Power rule (constant exponent)": (dd (^ ?f ?n:const) ?v:var) => (* :n (* (^ :f (! - :n 1)) (dd :f :v)))
  ```

- [ ] **Step 4: Append the examples to `examples/differentiation.metadata.json`.**
  Add these entries (single-step `out`, folds evaluated, inner `dd` unreduced); insert inside the top-level JSON object after the `dd-mult-const` entry, keeping valid JSON:

  ```json
    "dd-product": {
      "category": "product",
      "examples": [{"in": "(dd (* x x) x)", "out": "(+ (* (dd x x) x) (* x (dd x x)))"}]
    },
    "dd-quotient": {
      "category": "quotient",
      "examples": [{"in": "(dd (/ x y) x)", "out": "(/ (- (* (dd x x) y) (* x (dd y x))) (^ y 2))"}]
    },
    "dd-power": {
      "category": "power",
      "examples": [{"in": "(dd (^ x 3) x)", "out": "(* 3 (* (^ x 2) (dd x x)))"}]
    }
  ```

- [ ] **Step 5: Run the test, expect PASS.**
  ```bash
  pytest rerum/tests/test_differentiation.py::TestProductsQuotientPower -v
  ```
  Expected: all pass. `(dd (* x x) x)` simplifies to `(* 2 x)` and `(dd (^ x 3) x)` to `(* 3 (^ x 2))` (motivating examples), and every shape is numerically confirmed by `is_derivative`.

- [ ] **Step 6: Commit.**
  ```bash
  git add examples/differentiation.rules examples/differentiation.metadata.json rerum/tests/test_differentiation.py
  git commit -m "feat(examples): product, quotient, constant-exponent power rules

  Product and quotient rules plus the constant-exponent power rule (n f^(n-1)
  f', exponent decremented by a (! - :n 1) fold) at priority 60 so it beats the
  general f^g log-diff rule. d/dx(x*x)=(* 2 x) and d/dx(x^3)=(* 3 (^ x 2)) after
  the theory-driven normalize finishing pass.

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task 5: Exponential, logarithm (natural and base-b), sqrt, and constant-base `a^x`

**Files:**
- Edit: `examples/differentiation.rules` (append `[exp-log]` group)
- Edit: `examples/differentiation.metadata.json`
- Test: `rerum/tests/test_differentiation.py` (extend)

- [ ] **Step 1: Write the failing test for exp/log/sqrt/a^x.**
  Append to `rerum/tests/test_differentiation.py`:

  ```python
  class TestExpLogSqrt:
      def test_exp(self):
          engine = make_diff_engine()
          checker = _load_checker()
          out = differentiate(engine, "(dd (exp x) x)")
          assert checker.is_derivative("(exp x)", "x", format_sexpr(out)) is True

      def test_ln(self):
          engine = make_diff_engine()
          checker = _load_checker()
          out = differentiate(engine, "(dd (ln x) x)")
          assert checker.is_derivative("(ln x)", "x", format_sexpr(out)) is True

      def test_sqrt(self):
          engine = make_diff_engine()
          checker = _load_checker()
          out = differentiate(engine, "(dd (sqrt x) x)")
          assert checker.is_derivative("(sqrt x)", "x", format_sexpr(out)) is True

      def test_log_base(self):
          engine = make_diff_engine()
          checker = _load_checker()
          out = differentiate(engine, "(dd (log 2 x) x)")
          assert checker.is_derivative("(log 2 x)", "x",
                                       format_sexpr(out)) is True

      def test_a_to_the_x(self):
          engine = make_diff_engine()
          checker = _load_checker()
          # d/dx(2^x) = 2^x ln 2 (constant base, variable exponent)
          out = differentiate(engine, "(dd (^ 2 x) x)")
          assert checker.is_derivative("(^ 2 x)", "x", format_sexpr(out)) is True
  ```

- [ ] **Step 2: Run the test, expect FAIL.**
  ```bash
  pytest rerum/tests/test_differentiation.py::TestExpLogSqrt -v
  ```
  Expected: failures because exp/ln/sqrt/log/a^x rules are not yet present.

- [ ] **Step 3: Append the `[exp-log]` group to `examples/differentiation.rules`.**
  Add after `[products]`:

  ```
  [exp-log]
  # Exponential: d/dx(e^f) = e^f f'.
  @dd-exp "Exponential rule": (dd (exp ?f) ?v:var) => (* (exp :f) (dd :f :v))
  # Natural log: d/dx(ln f) = f'/f.
  @dd-ln "Natural log rule": (dd (ln ?f) ?v:var) => (/ (dd :f :v) :f)
  # Log base b (constant base): d/dx(log_b f) = f'/(f ln b).
  @dd-logb "Log base-b rule": (dd (log ?b:const ?f) ?v:var) => (/ (dd :f :v) (* :f (ln :b)))
  # sqrt: d/dx(sqrt f) = f'/(2 sqrt f).
  @dd-sqrt "Square-root rule": (dd (sqrt ?f) ?v:var) => (/ (dd :f :v) (* 2 (sqrt :f)))
  # a^x, constant base a: d/dx(a^f) = a^f (ln a) f'. Priority 60 so it beats the
  # general f^g log-diff rule; const base + general exponent.
  @dd-aexp[60] "Constant-base exponential": (dd (^ ?a:const ?f) ?v:var) => (* (* (^ :a :f) (ln :a)) (dd :f :v))
  ```

- [ ] **Step 4: Append the examples to `examples/differentiation.metadata.json`.**
  Single-step `out` (inner `dd` unreduced):

  ```json
    "dd-exp": {
      "category": "exp-log",
      "examples": [{"in": "(dd (exp x) x)", "out": "(* (exp x) (dd x x))"}]
    },
    "dd-ln": {
      "category": "exp-log",
      "examples": [{"in": "(dd (ln x) x)", "out": "(/ (dd x x) x)"}]
    },
    "dd-logb": {
      "category": "exp-log",
      "examples": [{"in": "(dd (log 2 x) x)", "out": "(/ (dd x x) (* x (ln 2)))"}]
    },
    "dd-sqrt": {
      "category": "exp-log",
      "examples": [{"in": "(dd (sqrt x) x)", "out": "(/ (dd x x) (* 2 (sqrt x)))"}]
    },
    "dd-aexp": {
      "category": "exp-log",
      "examples": [{"in": "(dd (^ 2 x) x)", "out": "(* (* (^ 2 x) (ln 2)) (dd x x))"}]
    }
  ```

- [ ] **Step 5: Run the test, expect PASS.**
  ```bash
  pytest rerum/tests/test_differentiation.py::TestExpLogSqrt -v
  ```
  Expected: all pass and numerically confirm. `dd-aexp` and `dd-logb` both require the `?a:const`/`?b:const` tag so they do not shadow the natural-base or general cases.

- [ ] **Step 6: Commit.**
  ```bash
  git add examples/differentiation.rules examples/differentiation.metadata.json rerum/tests/test_differentiation.py
  git commit -m "feat(examples): exp, natural log, log base-b, sqrt, and a^x rules

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task 6: Trigonometric rules (sin, cos, tan, sec, csc, cot)

Add the six circular-trig derivatives. `tan`'s derivative emits `sec`, so the `sec` rule must exist to differentiate that further; the `sec`/`csc`/`cot` rules consume the `sec` that `tan`'s derivative emits.

**Files:**
- Edit: `examples/differentiation.rules` (append `[trig]` group)
- Edit: `examples/differentiation.metadata.json`
- Test: `rerum/tests/test_differentiation.py` (extend)

- [ ] **Step 1: Write the failing test for the trig family.**
  Append to `rerum/tests/test_differentiation.py`:

  ```python
  class TestTrig:
      def test_sin(self):
          engine = make_diff_engine()
          checker = _load_checker()
          out = differentiate(engine, "(dd (sin x) x)")
          assert checker.is_derivative("(sin x)", "x", format_sexpr(out)) is True

      def test_cos(self):
          engine = make_diff_engine()
          checker = _load_checker()
          out = differentiate(engine, "(dd (cos x) x)")
          assert checker.is_derivative("(cos x)", "x", format_sexpr(out)) is True

      def test_tan(self):
          engine = make_diff_engine()
          checker = _load_checker()
          out = differentiate(engine, "(dd (tan x) x)")
          assert checker.is_derivative("(tan x)", "x", format_sexpr(out)) is True

      def test_sec(self):
          engine = make_diff_engine()
          checker = _load_checker()
          out = differentiate(engine, "(dd (sec x) x)")
          assert checker.is_derivative("(sec x)", "x", format_sexpr(out)) is True

      def test_csc(self):
          engine = make_diff_engine()
          checker = _load_checker()
          out = differentiate(engine, "(dd (csc x) x)")
          assert checker.is_derivative("(csc x)", "x", format_sexpr(out)) is True

      def test_cot(self):
          engine = make_diff_engine()
          checker = _load_checker()
          out = differentiate(engine, "(dd (cot x) x)")
          assert checker.is_derivative("(cot x)", "x", format_sexpr(out)) is True

      def test_chain_sin_of_square(self):
          engine = make_diff_engine()
          checker = _load_checker()
          # d/dx(sin(x^2)) = cos(x^2) * 2x ; verify numerically.
          out = differentiate(engine, "(dd (sin (^ x 2)) x)")
          assert checker.is_derivative("(sin (^ x 2))", "x",
                                       format_sexpr(out)) is True
  ```

- [ ] **Step 2: Run the test, expect FAIL.**
  ```bash
  pytest rerum/tests/test_differentiation.py::TestTrig -v
  ```
  Expected: failures because the trig rules are not yet present (leftover `dd`).

- [ ] **Step 3: Append the `[trig]` group to `examples/differentiation.rules`.**
  Add after `[exp-log]`:

  ```
  [trig]
  # d/dx(sin f) = cos f f'.
  @dd-sin "Sine rule": (dd (sin ?f) ?v:var) => (* (cos :f) (dd :f :v))
  # d/dx(cos f) = -sin f f'.
  @dd-cos "Cosine rule": (dd (cos ?f) ?v:var) => (* (- (sin :f)) (dd :f :v))
  # d/dx(tan f) = sec^2 f f' (emits sec, consumed by @dd-sec).
  @dd-tan "Tangent rule": (dd (tan ?f) ?v:var) => (* (^ (sec :f) 2) (dd :f :v))
  # d/dx(sec f) = sec f tan f f'.
  @dd-sec "Secant rule": (dd (sec ?f) ?v:var) => (* (* (sec :f) (tan :f)) (dd :f :v))
  # d/dx(csc f) = -csc f cot f f'.
  @dd-csc "Cosecant rule": (dd (csc ?f) ?v:var) => (* (- (* (csc :f) (cot :f))) (dd :f :v))
  # d/dx(cot f) = -csc^2 f f'.
  @dd-cot "Cotangent rule": (dd (cot ?f) ?v:var) => (* (- (^ (csc :f) 2)) (dd :f :v))
  ```

- [ ] **Step 4: Append the examples to `examples/differentiation.metadata.json`.**
  Single-step `out` (inner `dd` unreduced):

  ```json
    "dd-sin": {
      "category": "trig",
      "examples": [{"in": "(dd (sin x) x)", "out": "(* (cos x) (dd x x))"}]
    },
    "dd-cos": {
      "category": "trig",
      "examples": [{"in": "(dd (cos x) x)", "out": "(* (- (sin x)) (dd x x))"}]
    },
    "dd-tan": {
      "category": "trig",
      "examples": [{"in": "(dd (tan x) x)", "out": "(* (^ (sec x) 2) (dd x x))"}]
    },
    "dd-sec": {
      "category": "trig",
      "examples": [{"in": "(dd (sec x) x)", "out": "(* (* (sec x) (tan x)) (dd x x))"}]
    },
    "dd-csc": {
      "category": "trig",
      "examples": [{"in": "(dd (csc x) x)", "out": "(* (- (* (csc x) (cot x))) (dd x x))"}]
    },
    "dd-cot": {
      "category": "trig",
      "examples": [{"in": "(dd (cot x) x)", "out": "(* (- (^ (csc x) 2)) (dd x x))"}]
    }
  ```

- [ ] **Step 5: Run the test, expect PASS.**
  ```bash
  pytest rerum/tests/test_differentiation.py::TestTrig -v
  ```
  Expected: all pass and numerically confirm, including the chain-rule case `d/dx(sin(x^2))`.

- [ ] **Step 6: Commit.**
  ```bash
  git add examples/differentiation.rules examples/differentiation.metadata.json rerum/tests/test_differentiation.py
  git commit -m "feat(examples): sin, cos, tan, sec, csc, cot derivative rules

  sec/csc/cot rules consume the sec that tan's derivative emits.

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task 7: Inverse-trig (asin, acos, atan) and hyperbolic (sinh, cosh, tanh)

**Files:**
- Edit: `examples/differentiation.rules` (append `[inverse-trig]` and `[hyperbolic]` groups)
- Edit: `examples/differentiation.metadata.json`
- Test: `rerum/tests/test_differentiation.py` (extend)

- [ ] **Step 1: Write the failing test for inverse-trig and hyperbolic.**
  Append to `rerum/tests/test_differentiation.py`:

  ```python
  class TestInverseTrig:
      def test_asin(self):
          engine = make_diff_engine()
          checker = _load_checker()
          out = differentiate(engine, "(dd (asin x) x)")
          assert checker.is_derivative("(asin x)", "x", format_sexpr(out)) is True

      def test_acos(self):
          engine = make_diff_engine()
          checker = _load_checker()
          out = differentiate(engine, "(dd (acos x) x)")
          assert checker.is_derivative("(acos x)", "x", format_sexpr(out)) is True

      def test_atan(self):
          engine = make_diff_engine()
          checker = _load_checker()
          out = differentiate(engine, "(dd (atan x) x)")
          assert checker.is_derivative("(atan x)", "x", format_sexpr(out)) is True


  class TestHyperbolic:
      def test_sinh(self):
          engine = make_diff_engine()
          checker = _load_checker()
          out = differentiate(engine, "(dd (sinh x) x)")
          assert checker.is_derivative("(sinh x)", "x", format_sexpr(out)) is True

      def test_cosh(self):
          engine = make_diff_engine()
          checker = _load_checker()
          out = differentiate(engine, "(dd (cosh x) x)")
          assert checker.is_derivative("(cosh x)", "x", format_sexpr(out)) is True

      def test_tanh(self):
          engine = make_diff_engine()
          checker = _load_checker()
          out = differentiate(engine, "(dd (tanh x) x)")
          assert checker.is_derivative("(tanh x)", "x", format_sexpr(out)) is True
  ```

- [ ] **Step 2: Run the test, expect FAIL.**
  ```bash
  pytest rerum/tests/test_differentiation.py::TestInverseTrig rerum/tests/test_differentiation.py::TestHyperbolic -v
  ```
  Expected: failures (leftover `dd` since the rules are absent).

- [ ] **Step 3: Append the `[inverse-trig]` and `[hyperbolic]` groups to `examples/differentiation.rules`.**
  Add after `[trig]`:

  ```
  [inverse-trig]
  # d/dx(asin f) = f' / sqrt(1 - f^2).
  @dd-asin "Arcsine rule": (dd (asin ?f) ?v:var) => (* (/ 1 (sqrt (- 1 (^ :f 2)))) (dd :f :v))
  # d/dx(acos f) = -f' / sqrt(1 - f^2).
  @dd-acos "Arccosine rule": (dd (acos ?f) ?v:var) => (* (- (/ 1 (sqrt (- 1 (^ :f 2))))) (dd :f :v))
  # d/dx(atan f) = f' / (1 + f^2).
  @dd-atan "Arctangent rule": (dd (atan ?f) ?v:var) => (* (/ 1 (+ 1 (^ :f 2))) (dd :f :v))

  [hyperbolic]
  # d/dx(sinh f) = cosh f f'.
  @dd-sinh "Hyperbolic sine rule": (dd (sinh ?f) ?v:var) => (* (cosh :f) (dd :f :v))
  # d/dx(cosh f) = sinh f f'.
  @dd-cosh "Hyperbolic cosine rule": (dd (cosh ?f) ?v:var) => (* (sinh :f) (dd :f :v))
  # d/dx(tanh f) = (1 - tanh^2 f) f' (avoids introducing a sech function).
  @dd-tanh "Hyperbolic tangent rule": (dd (tanh ?f) ?v:var) => (* (- 1 (^ (tanh :f) 2)) (dd :f :v))
  ```

- [ ] **Step 4: Append the examples to `examples/differentiation.metadata.json`.**
  Single-step `out` (inner `dd` unreduced):

  ```json
    "dd-asin": {
      "category": "inverse-trig",
      "examples": [{"in": "(dd (asin x) x)", "out": "(* (/ 1 (sqrt (- 1 (^ x 2)))) (dd x x))"}]
    },
    "dd-acos": {
      "category": "inverse-trig",
      "examples": [{"in": "(dd (acos x) x)", "out": "(* (- (/ 1 (sqrt (- 1 (^ x 2))))) (dd x x))"}]
    },
    "dd-atan": {
      "category": "inverse-trig",
      "examples": [{"in": "(dd (atan x) x)", "out": "(* (/ 1 (+ 1 (^ x 2))) (dd x x))"}]
    },
    "dd-sinh": {
      "category": "hyperbolic",
      "examples": [{"in": "(dd (sinh x) x)", "out": "(* (cosh x) (dd x x))"}]
    },
    "dd-cosh": {
      "category": "hyperbolic",
      "examples": [{"in": "(dd (cosh x) x)", "out": "(* (sinh x) (dd x x))"}]
    },
    "dd-tanh": {
      "category": "hyperbolic",
      "examples": [{"in": "(dd (tanh x) x)", "out": "(* (- 1 (^ (tanh x) 2)) (dd x x))"}]
    }
  ```

- [ ] **Step 5: Run the test, expect PASS.**
  ```bash
  pytest rerum/tests/test_differentiation.py::TestInverseTrig rerum/tests/test_differentiation.py::TestHyperbolic -v
  ```
  Expected: all pass and numerically confirm.

- [ ] **Step 6: Commit.**
  ```bash
  git add examples/differentiation.rules examples/differentiation.metadata.json rerum/tests/test_differentiation.py
  git commit -m "feat(examples): inverse-trig (asin/acos/atan) and hyperbolic (sinh/cosh/tanh)

  tanh' expressed as (1 - tanh^2) to avoid introducing a sech function the
  checker would also have to handle.

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task 8: General power `f^g` via logarithmic differentiation

The general power rule covers `f^g` when the exponent CONTAINS the variable (e.g. `x^x`, `x^(sin x)`, `(sin x)^x`). Logarithmic differentiation gives:

> d/dx f^g = f^g * (g' * ln f + g * f'/f)

This is the form in spec 6.1 and the contract. It must sit BELOW the constant-exponent rule (`dd-power[60]`) and the constant-base rule (`dd-aexp[60]`); set this one to priority 50 so a numeric exponent or numeric base is handled by the cleaner specialized rule first, and only a genuine variable-in-both case reaches log-diff.

**Ambiguity resolved (general-power form):** `(dd (^ ?f ?g) ?v)` with no type tag on `?f` or `?g` would also match `(^ x 3)` and `(^ 2 x)`. Priority ordering (60 for the specialized rules, 50 here) makes the specialized rules win, so log-diff only fires when neither `?n:const` nor `?a:const` matched, i.e. both base and exponent are non-constant. The skeleton is exactly the spec form `(* (^ :f :g) (+ (* (dd :g :v) (ln :f)) (* :g (/ (dd :f :v) :f))))`.

**Files:**
- Edit: `examples/differentiation.rules` (append `[general-power]` group)
- Edit: `examples/differentiation.metadata.json`
- Test: `rerum/tests/test_differentiation.py` (extend)

- [ ] **Step 1: Write the failing test for general power.**
  Append to `rerum/tests/test_differentiation.py`:

  ```python
  class TestGeneralPower:
      def test_x_to_the_x(self):
          engine = make_diff_engine()
          checker = _load_checker()
          # d/dx(x^x) = x^x (ln x + 1) ; verify numerically (domain x>0 by sampling).
          out = differentiate(engine, "(dd (^ x x) x)")
          assert checker.is_derivative("(^ x x)", "x", format_sexpr(out)) is True

      def test_x_to_the_sin_x(self):
          engine = make_diff_engine()
          checker = _load_checker()
          # d/dx(x^(sin x)) via log-diff ; verify numerically.
          out = differentiate(engine, "(dd (^ x (sin x)) x)")
          assert checker.is_derivative("(^ x (sin x))", "x",
                                       format_sexpr(out)) is True

      def test_constant_exponent_still_uses_power_rule(self):
          engine = make_diff_engine()
          # Regression: a numeric exponent must STILL take the clean power rule,
          # not log-diff. d/dx(x^3) = 3 x^2 exactly.
          out = differentiate(engine, "(dd (^ x 3) x)")
          assert out == ["*", 3, ["^", "x", 2]]

      def test_constant_base_still_uses_aexp(self):
          engine = make_diff_engine()
          checker = _load_checker()
          # Regression: a numeric base must STILL take the a^x rule.
          out = differentiate(engine, "(dd (^ 2 x) x)")
          assert checker.is_derivative("(^ 2 x)", "x", format_sexpr(out)) is True
  ```

- [ ] **Step 2: Run the test, expect FAIL.**
  ```bash
  pytest rerum/tests/test_differentiation.py::TestGeneralPower -v
  ```
  Expected: `test_x_to_the_x` and `test_x_to_the_sin_x` FAIL (no general-power rule, so a leftover `dd` survives and `is_derivative` sees a `dd` head -> `ValueError` -> returns False). The two regression tests should already PASS (specialized rules handle them).

- [ ] **Step 3: Append the `[general-power]` group to `examples/differentiation.rules`.**
  Add after `[hyperbolic]`:

  ```
  [general-power]
  # General power f^g (variable in the exponent) via logarithmic
  # differentiation: d/dx f^g = f^g (g' ln f + g f'/f). Priority 50 so the
  # constant-exponent power rule (60) and constant-base a^x rule (60) win for
  # their specialized shapes; this fires only when both base and exponent are
  # non-constant.
  @dd-genpow[50] "General power (log-diff)": (dd (^ ?f ?g) ?v:var) => (* (^ :f :g) (+ (* (dd :g :v) (ln :f)) (* :g (/ (dd :f :v) :f))))
  ```

- [ ] **Step 4: Append the example to `examples/differentiation.metadata.json`.**
  Single-step `out` (inner `dd` unreduced):

  ```json
    "dd-genpow": {
      "category": "general-power",
      "examples": [{"in": "(dd (^ x x) x)", "out": "(* (^ x x) (+ (* (dd x x) (ln x)) (* x (/ (dd x x) x))))"}]
    }
  ```

- [ ] **Step 5: Run the test, expect PASS.**
  ```bash
  pytest rerum/tests/test_differentiation.py::TestGeneralPower -v
  ```
  Expected: all pass. The two regression tests confirm the priority ordering keeps `x^3` and `2^x` on their specialized rules; the log-diff cases verify numerically (the checker samples in (0.25, 1.75) so `ln x`/`x^x` stay in domain).

- [ ] **Step 6: Commit.**
  ```bash
  git add examples/differentiation.rules examples/differentiation.metadata.json rerum/tests/test_differentiation.py
  git commit -m "feat(examples): general power f^g via logarithmic differentiation

  d/dx f^g = f^g (g' ln f + g f'/f) at priority 50, below the constant-exponent
  power rule and constant-base a^x rule (both 60), so log-diff fires only when
  base and exponent are both non-constant.

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task 9: Partial derivatives, full-pipeline integration, and full suite

Tie it together: confirm partial derivatives via the `free-of?` guard, the motivating clean-output cases, and that every rule carries a validated example; then run the whole suite. No engine code; nothing to export (the checker is example content imported by path, not from the package).

**Files:**
- Test: `rerum/tests/test_differentiation.py` (extend)

- [ ] **Step 1: Write the failing test for partials, the motivating pipeline, and example coverage.**
  Append to `rerum/tests/test_differentiation.py`:

  ```python
  class TestPartialDerivatives:
      def test_partial_treats_other_var_as_constant(self):
          engine = make_diff_engine()
          checker = _load_checker()
          # d/dx(x * y) = y (y free of x; product rule + free-of? -> y*1 + x*0)
          out = differentiate(engine, "(dd (* x y) x)")
          assert out == "y"
          assert checker.is_derivative("(* x y)", "x", format_sexpr(out)) is True

      def test_partial_wrt_y(self):
          engine = make_diff_engine()
          checker = _load_checker()
          out = differentiate(engine, "(dd (* x y) y)")
          assert out == "x"
          assert checker.is_derivative("(* x y)", "y", format_sexpr(out)) is True

      def test_partial_sum_of_two_vars(self):
          engine = make_diff_engine()
          checker = _load_checker()
          # d/dx(x^2 + y^2) = 2x
          out = differentiate(engine, "(dd (+ (^ x 2) (^ y 2)) x)")
          assert out == ["*", 2, "x"]
          assert checker.is_derivative("(+ (^ x 2) (^ y 2))", "x",
                                       format_sexpr(out)) is True

      def test_free_subexpression_is_zero(self):
          engine = make_diff_engine()
          # d/dx(sin y) = 0 directly (free-of? guard at top priority).
          assert differentiate(engine, "(dd (sin y) x)") == 0


  class TestMotivatingCleanOutput:
      def test_motivating_examples_are_clean(self):
          engine = make_diff_engine()
          # The two spec/contract motivating cases, exact forms.
          assert differentiate(engine, "(dd (* x x) x)") == ["*", 2, "x"]
          assert differentiate(engine, "(dd (^ x 3) x)") == ["*", 3, ["^", "x", 2]]


  class TestEveryRuleHasExample:
      def test_every_named_rule_carries_an_example(self):
          engine = make_diff_engine()
          # After the sidecar merge, every dd-* rule carries at least one
          # validated example.
          for meta in engine._metadata:
              if meta.name and meta.name.startswith("dd-"):
                  assert meta.examples, f"{meta.name} has no example"
  ```

- [ ] **Step 2: Run the tests, expect PASS (all rules are in place from Tasks 3 to 8).**
  ```bash
  pytest rerum/tests/test_differentiation.py::TestPartialDerivatives rerum/tests/test_differentiation.py::TestMotivatingCleanOutput rerum/tests/test_differentiation.py::TestEveryRuleHasExample -v
  ```
  Expected: all pass. Partial derivatives fall out of the `free-of?` guard plus the product/sum rules and the theory-driven `normalize` (`d/dx(x*y)=y`, `d/dy(x*y)=x`); both motivating cases are exact; every named rule carries a validated example. If a partial test fails, the suspect is the `free-of?` guard priority or the Phase 0 `?free`/guard fixes; fix there, do not weaken the test.

- [ ] **Step 3: Run both new files and the full suite.**
  ```bash
  pytest rerum/tests/test_differentiation.py -v
  pytest
  ```
  Expected: the differentiation file fully green; the full suite green with no regressions (the new example files are additive; existing `examples/calculus.rules` and `examples/algebra.rules` are untouched and still load).

- [ ] **Step 4: Smoke-test the example files load via the includes path and the motivating one-liner.**
  ```bash
  pytest rerum/tests/test_includes.py rerum/tests/test_cli.py -v
  python -c "
  from pathlib import Path
  from rerum import RuleEngine, MATH_PRELUDE, PREDICATE_PRELUDE, combine_preludes
  from rerum.engine import parse_sexpr, format_sexpr
  from rerum.normalize import normalize, Theory
  ex = Path('examples')
  theory = Theory.from_json((ex/'arithmetic.theory.json').read_text())
  e = (RuleEngine()
       .with_prelude(combine_preludes(MATH_PRELUDE, PREDICATE_PRELUDE))
       .with_theory(theory)
       .load_file(ex/'algebra.rules')
       .load_file(ex/'differentiation.rules'))
  e.load_metadata_json((ex/'differentiation.metadata.json').read_text(), validate_examples=True)
  print(format_sexpr(normalize(e(parse_sexpr('(dd (* x x) x)')), theory)))
  print(format_sexpr(normalize(e(parse_sexpr('(dd (^ x 3) x)')), theory)))
  "
  ```
  Expected: includes/CLI suites pass; the one-liner prints `(* 2 x)` then `(* 3 (^ x 2))`.

- [ ] **Step 5: Commit.**
  ```bash
  git add rerum/tests/test_differentiation.py
  git commit -m "test(examples): partial derivatives, clean-output pipeline, example coverage

  Partial derivatives fall out of the free-of? guard plus the product/sum rules
  and the theory-driven normalize: d/dx(x*y)=y, d/dy(x*y)=x, d/dx(sin y)=0. The
  two motivating cases simplify to (* 2 x) and (* 3 (^ x 2)). Every dd-* rule
  carries a validated example. This phase added no engine code: a whole domain
  is just rules + theory + a checker, all under examples/.

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

## Done When

- [ ] This phase added NO `rerum/` core code. Every new artifact is under `examples/` (`differentiation.rules`, `differentiation.metadata.json`, `arithmetic.theory.json`, `calculus_checker.py`) or is the example-exercising test `rerum/tests/test_differentiation.py`. The swap test holds trivially (nothing in `rerum/` changed).
- [ ] `examples/differentiation.rules` is a COMPLETE differentiation rule set over `["dd", f, v]`: constant/variable base cases; n-ary sum (rest-pattern), difference, constant multiple; product and quotient; constant-exponent power; exp, natural log, log base-b, sqrt, and constant-base `a^x`; sin/cos/tan/sec/csc/cot (sec/csc/cot consume the sec that tan emits); asin/acos/atan; sinh/cosh/tanh; general `f^g` via logarithmic differentiation; and partial derivatives via the `free-of?` guard.
- [ ] Every rule carries an `{in, out}` example in `examples/differentiation.metadata.json`, all validated at load via `engine.load_metadata_json(text, validate_examples=True)` (each `out` is the rule's correct single-step rewrite).
- [ ] `examples/arithmetic.theory.json` declares `+` and `*` as AC with identities (0, 1) and `*`'s annihilator (0), and is consumed by the general `normalize` via `engine.with_theory(Theory.from_json(...))`.
- [ ] Priority ordering is correct: the partial `free-of?` guard `[110]` outranks the base cases `[100]`; the constant-exponent power rule and constant-base `a^x` rule `[60]` outrank the general `f^g` log-diff rule `[50]`, so `(dd (^ x 3) x)` and `(dd (^ 2 x) x)` use the specialized rules and `(dd (^ x x) x)` uses log-diff.
- [ ] Differentiation runs on the EXISTING `simplify` driver (it is confluent) with the theory set and `combine_preludes(MATH_PRELUDE, PREDICATE_PRELUDE)`; NO `solve`. The rule set composes with `examples/algebra.rules` plus the general `normalize` finishing pass to produce CLEAN output on the motivating examples: `d/dx(x*x) -> (* 2 x)` and `d/dx(x^3) -> (* 3 (^ x 2))`, exactly.
- [ ] Verification lives entirely under `examples/`: `examples/calculus_checker.py` exposes `is_derivative(expr, var, result, *, samples=8, tol=1e-6) -> bool` built ON TOP of the general `rerum.numeval`/`numeric_equiv` (finite-difference compare over a numeric prelude), skipping out-of-domain sample points via `_DomainError`. There is NO core `rerum/verify.py`; the engine never imports the checker.
- [ ] Every derivative family is exercised end to end in `rerum/tests/test_differentiation.py`: differentiate a concrete example through `simplify` + theory-driven `normalize` AND confirm the simplified answer numerically with `is_derivative`.
- [ ] Partial derivatives work: `d/dx(x*y) -> y`, `d/dy(x*y) -> x`, `d/dx(sin y) -> 0`.
- [ ] `pytest` (full suite) is green with no regressions; the example files load via `load_file` and the includes/CLI smoke tests pass.
