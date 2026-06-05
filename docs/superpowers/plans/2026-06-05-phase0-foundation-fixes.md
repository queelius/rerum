# Phase 0: Foundation Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Fix the two pre-existing correctness bugs (the `?x:free(v)` binding-order bug and the guard-on-undefined-op corruption footgun), add a general `free-of?` predicate fold op, and ship a general `combine_preludes(*preludes)` helper, so any rule set can compose the preludes it needs. Per the general-engine principle (spec Section 0), the engine ships NO domain-named bundle: there is no `CALCULUS_PRELUDE`. Calculus is just one example consumer that documents it needs `combine_preludes(MATH_PRELUDE, PREDICATE_PRELUDE)`.

**Architecture:** All changes live in the pure core (`rewriter.py`: match, predicates, prelude, the `combine_preludes` helper) and the OO API (`engine.py`: guard evaluation), plus exports in `rerum/__init__.py`. No new modules. Existing public signatures of `match`, `_check_condition`, and the prelude constants are preserved; behavior is corrected and additively extended. Nothing added here names a domain: `free-of?` is a structural predicate, and `combine_preludes` is a generic dict-merge over fold dicts.

**Tech Stack:** Python 3.9+, pytest with plain asserts, one test file per feature area under `rerum/tests/`. Config in `pyproject.toml` (`testpaths = ["rerum/tests"]`, `addopts = "-v"`).

---

## File Structure

| File | Status | Responsibility |
|------|--------|----------------|
| `rerum/rewriter.py` | Modify | Split `match` into a public wrapper plus recursive core; add `_check_free_constraints` post-pass that re-checks every `?free` against the FINAL bindings; add `free-of?` to `PREDICATE_PRELUDE`; add the general `combine_preludes(*preludes)` helper that merges fold dicts left-to-right and returns a fresh dict. |
| `rerum/engine.py` | Modify | In `_check_condition`, raise `ValueError("guard references undefined op ...")` when the instantiated guard still contains an unfolded compute form whose head op is not in the active prelude, instead of letting it evaluate truthy. |
| `rerum/__init__.py` | Modify | Import and export `combine_preludes` (both the `from .rewriter import (...)` block and `__all__`). |
| `rerum/tests/test_free_of.py` | Create | New file: the `?x:free(v)` binding-order fix and the `free-of?` predicate. |
| `rerum/tests/test_preludes.py` | Create | New file: `combine_preludes` merge semantics, later-wins, fresh-dict (no input mutation), and the `combine_preludes(MATH_PRELUDE, PREDICATE_PRELUDE)` contents. |
| `rerum/tests/test_guards.py` | Modify | Add a `TestGuardUndefinedOp` class: undefined-op guard raises `ValueError`; valid guards still pass. |

Ground-truth line references (read before editing; they may have shifted):
- `rewriter.py`: `free_in` at line 558; `match` at line 648; `match_compound` at line 720 (recurses into `match` at line 757); `arbitrary_free` at line 516; `arbitrary_rest` at line 521; `skeleton_compute` at line 550; `binary_only` at line 257; `FoldFuncsType` at line 213; `PREDICATE_PRELUDE` at lines 339-359; `MATH_PRELUDE` at lines 304-325; `FULL_PRELUDE` at lines 362-365; `NO_PRELUDE` at line 368; the rewriter factory calls `match` at line 1037.
- `engine.py`: `_check_condition` at line 1907; `_condition_truthy` at line 338; `_validate_example` at line 355; `instantiate` is imported and `self._fold_funcs` holds the active prelude; `self._undefined_op_resolver` at line 2349 (None when no resolver registered).
- `__init__.py`: `from .rewriter import (...)` block at lines 43-71 (fold builders at lines 58-63, preludes at lines 64-70); `__all__` fold builders at lines 130-134, preludes at lines 136-141.

---

### Task 1: Add the `free-of?` predicate fold op to `PREDICATE_PRELUDE`

This is the smallest, dependency-free change and it provides the principled, general replacement the spec wants for the fragile `?free` tag, so do it first. `free-of?` is a structural predicate (no domain knowledge); it tests whether a symbol occurs in a term.

**Files:**
- Modify: `rerum/rewriter.py` (`PREDICATE_PRELUDE`, lines 339-359; reuse `free_in` at line 558)
- Test: `rerum/tests/test_free_of.py` (create)

- [ ] **Step 1: Write failing test for `free-of?` predicate.**
  Create `rerum/tests/test_free_of.py` with this content:

  ```python
  """Tests for the free-of? predicate and the ?free binding-order fix."""

  import pytest
  from rerum import RuleEngine, E, PREDICATE_PRELUDE


  class TestFreeOfPredicate:
      """The free-of? fold op: (! free-of? f v) is true iff symbol v does not occur in f."""

      def test_free_of_in_prelude(self):
          """PREDICATE_PRELUDE exposes the free-of? operator."""
          assert "free-of?" in PREDICATE_PRELUDE

      def test_free_of_true_when_absent(self):
          """free-of? returns True when v does not occur in f."""
          engine = (RuleEngine()
              .with_prelude(PREDICATE_PRELUDE)
              .load_dsl("@check: (q ?f ?v) => yes when (! free-of? :f :v)"))
          assert engine(E("(q (sin y) x)")) == "yes"

      def test_free_of_false_when_present(self):
          """free-of? returns False when v occurs in f, so a guarded rule does not fire."""
          engine = (RuleEngine()
              .with_prelude(PREDICATE_PRELUDE)
              .load_dsl("@check: (q ?f ?v) => yes when (! free-of? :f :v)"))
          # x occurs in (sin x); guard fails, expression is left unchanged.
          assert engine(E("(q (sin x) x)")) == ["q", ["sin", "x"], "x"]

      def test_free_of_atom_self(self):
          """A symbol is not free of itself."""
          engine = (RuleEngine()
              .with_prelude(PREDICATE_PRELUDE)
              .load_dsl("@check: (q ?f ?v) => yes when (! free-of? :f :v)"))
          assert engine(E("(q x x)")) == ["q", "x", "x"]

      def test_free_of_constant_always_free(self):
          """A constant contains no variable, so free-of? is always True for it."""
          engine = (RuleEngine()
              .with_prelude(PREDICATE_PRELUDE)
              .load_dsl("@check: (q ?f ?v) => yes when (! free-of? :f :v)"))
          assert engine(E("(q 7 x)")) == "yes"
  ```

- [ ] **Step 2: Run the test, expect FAIL.**
  Command:
  ```
  cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_free_of.py::TestFreeOfPredicate -v
  ```
  Expected: `test_free_of_in_prelude` FAILS with `assert 'free-of?' in PREDICATE_PRELUDE` (KeyError-style AssertionError); the guarded tests FAIL or error because the op is undefined.

- [ ] **Step 3: Implement `free-of?` in `PREDICATE_PRELUDE`.**
  In `rerum/rewriter.py`, inside the `PREDICATE_PRELUDE` dict (between the `"negative?"` entry and the `# Logical operators` comment, around line 354), add a `# Structural predicates` group. Note `free_in(var, expr)` (line 558) takes `(var, expr)` order, while `free-of?` is called as `(! free-of? f v)`, so the lambda must reorder arguments. Use `binary_only` so it receives positional `(f, v)`:

  ```python
      "negative?": unary_only(lambda x: isinstance(x, (int, float)) and x < 0),
      # Structural predicates
      "free-of?": binary_only(lambda f, v: isinstance(v, str) and not free_in(v, f)),
  ```

  Place this directly after the `"negative?"` line. `binary_only` is already defined in this module (line 257) and used elsewhere in `ARITHMETIC_PRELUDE`. `free_in` is defined later in the file (line 558) but is referenced at call time inside the lambda, so the forward reference resolves at evaluation, not at module load.

- [ ] **Step 4: Run the test, expect PASS.**
  Command:
  ```
  cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_free_of.py::TestFreeOfPredicate -v
  ```
  Expected: all 5 tests PASS.

- [ ] **Step 5: Commit.**
  ```
  cd /home/spinoza/github/repos/rerum && git add rerum/rewriter.py rerum/tests/test_free_of.py
  git commit -m "feat(prelude): add free-of? predicate fold op

  (! free-of? f v) is true iff symbol v does not occur in f, reusing the
  existing free_in helper. Added to PREDICATE_PRELUDE as the principled,
  domain-agnostic replacement for the fragile ?free pattern tag in guard
  contexts.

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task 2: Fix the `?x:free(v)` binding-order bug in `match`

Today the `?free` constraint is checked the instant the `?free` pattern is visited. When the excluded var `v` is bound to the RIGHT of the `?free` pattern (as in `(dd ?f:free(v) ?v:var)`), `v` is still unbound at check time, `bindings.lookup("v")` returns the symbol `"v"` itself, `free_in("v", f)` is vacuously False, and the constraint passes wrongly. Documented failing case: `(dd ?f:free(v) ?v:var)` matching `(dd (sin x) x)` MUST NOT match (today it wrongly binds `f=(sin x), v=x`). The fix per the contract: evaluate the free check against the FINAL resolved bindings.

This is a general matcher-correctness fix. The pattern uses `dd` only as an arbitrary operator symbol in test data; the matcher knows nothing about `dd`. The same bug would bite any rule set whose `?free` pattern precedes the binding of its excluded variable (substitution-style rules in any domain).

Mechanism: keep the optimistic bind inside the recursion, then run a single post-pass at the public `match` entry point that re-checks every `?free` node against the final bindings. The recursion uses an internal `_match_recursive` so intermediate frames never validate (and never prematurely fail before `v` is bound).

**Files:**
- Modify: `rerum/rewriter.py` (`match` at line 648, `match_compound` recursion at line 757; `arbitrary_free` at line 516; `arbitrary_rest` at line 521; `free_in` at line 558)
- Test: `rerum/tests/test_free_of.py` (extend)

- [ ] **Step 1: Write failing test for the binding-order fix.**
  Append this class to `rerum/tests/test_free_of.py`:

  ```python
  class TestFreeBindingOrder:
      """The ?x:free(v) tag must be checked against the FINAL resolved bindings."""

      def test_free_left_of_var_does_not_match(self):
          """The documented failing case: (dd ?f:free(v) ?v:var) must NOT match (dd (sin x) x)."""
          from rerum.rewriter import match
          from rerum.engine import parse_sexpr
          pat = parse_sexpr("(dd ?f:free(v) ?v:var)")
          exp = parse_sexpr("(dd (sin x) x)")
          assert match(pat, exp) is None

      def test_free_left_of_var_matches_when_truly_free(self):
          """The same pattern still matches when f is genuinely free of the bound v."""
          from rerum.rewriter import match
          from rerum.engine import parse_sexpr
          pat = parse_sexpr("(dd ?f:free(v) ?v:var)")
          exp = parse_sexpr("(dd (sin y) x)")
          b = match(pat, exp)
          assert b is not None
          assert b.to_dict() == {"f": ["sin", "y"], "v": "x"}

      def test_free_right_of_var_still_works(self):
          """When v is bound to the LEFT of ?free, the legacy ordering still rejects/accepts correctly."""
          from rerum.rewriter import match
          from rerum.engine import parse_sexpr
          # v is bound first (?v:var), then ?free(v) is checked.
          pat = parse_sexpr("(g ?v:var ?f:free(v))")
          # f contains v -> no match
          assert match(pat, parse_sexpr("(g x (sin x))")) is None
          # f free of v -> match
          b = match(pat, parse_sexpr("(g x (sin y))"))
          assert b is not None
          assert b.to_dict() == {"v": "x", "f": ["sin", "y"]}

      def test_free_of_compound_excluded_var(self):
          """If the excluded var resolves to a non-symbol, free-of is treated structurally."""
          from rerum.rewriter import match
          from rerum.engine import parse_sexpr
          # ?v binds a variable symbol; baseline sanity that a free f still matches.
          pat = parse_sexpr("(dd ?f:free(v) ?v:var)")
          b = match(pat, parse_sexpr("(dd 5 x)"))
          assert b is not None
          assert b.to_dict() == {"f": 5, "v": "x"}
  ```

- [ ] **Step 2: Run the test, expect FAIL.**
  Command:
  ```
  cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_free_of.py::TestFreeBindingOrder -v
  ```
  Expected: `test_free_left_of_var_does_not_match` FAILS with `assert Bindings({'f': ['sin', 'x'], 'v': 'x'}) is None` (the match wrongly succeeds today). The other three may pass already; the load-bearing failure is the first one.

- [ ] **Step 3: Rename the recursive core and add the public wrapper.**
  In `rerum/rewriter.py`, rename the current `def match(...)` (line 648) to `def _match_recursive(...)`, keeping its body identical. Then add a new public `match` directly above it:

  ```python
  def match(pat: ExprType, exp: ExprType,
            bindings: Optional["Bindings"] = None) -> Optional["Bindings"]:
      """Structural pattern match (public entry point).

      Delegates to the recursive core, then validates every ``?free``
      constraint against the FINAL resolved bindings. This fixes the
      binding-order bug where a ``?free`` pattern appearing to the LEFT of
      the binding of its excluded variable passed vacuously (the excluded
      var was still unbound when the free check ran). See ``?x:free(v)``.

      Returns ``Bindings`` on success, ``None`` on failure.
      """
      result = _match_recursive(pat, exp, bindings)
      if result is None:
          return None
      if not _check_free_constraints(pat, exp, result):
          return None
      return result
  ```

  Update the renamed `_match_recursive` docstring's first line to note it is the recursive core that does NOT run the deferred `?free` validation.

- [ ] **Step 4: Point internal recursion at the core.**
  In `_match_recursive` (the renamed function) there is no self-recursive call (it ends by delegating to `match_compound`). In `match_compound` (line 720), the call at line 757 is `submatch = match(current_pat, car(exp), bindings)`. Change it to call the core so intermediate frames do not run the deferred validation:

  ```python
      submatch = _match_recursive(current_pat, car(exp), bindings)
  ```

- [ ] **Step 5: Add the `_check_free_constraints` post-pass.**
  In `rerum/rewriter.py`, directly after `free_in` (line 575) or after the new public `match`, add a helper that walks the pattern and expression in parallel, finding every `?free` node and re-checking it against the final bindings. Rest patterns (`?...`) consume the remaining expression tail, so account for them:

  ```python
  def _check_free_constraints(pat: ExprType, exp: ExprType,
                              bindings: "Bindings") -> bool:
      """Re-validate every ``?free`` constraint in ``pat`` against ``exp``
      using the FINAL ``bindings``.

      Returns True if all ``?free`` constraints hold, False otherwise. A
      ``?free`` node ``["?free", name, var]`` requires that the resolved
      value of ``var`` does not occur in the subexpression that ``name``
      matched (which, structurally, is the aligned ``exp`` position).
      """
      if arbitrary_free(pat):
          excluded = bindings.lookup(pat[2])
          if isinstance(excluded, str) and free_in(excluded, exp):
              return False
          return True
      # Other leaf/binding patterns impose no free constraint.
      if not isinstance(pat, list) or not pat:
          return True
      head = pat[0]
      if head in ("?", "?c", "?v"):
          return True
      if not isinstance(exp, list):
          # Pattern is compound but expression is an atom; no aligned
          # children to recurse into (the recursive matcher already
          # decided match/non-match).
          return True
      # Walk children pairwise; a rest pattern (?...) consumes the tail.
      i = 0
      for sub_pat in pat:
          if arbitrary_rest(sub_pat):
              # Rest patterns bind a list and impose no ?free constraint.
              break
          if i >= len(exp):
              break
          if not _check_free_constraints(sub_pat, exp[i], bindings):
              return False
          i += 1
      return True
  ```

  `arbitrary_free` (line 516) and `arbitrary_rest` (line 521) are already defined in this module.

- [ ] **Step 6: Run the test, expect PASS.**
  Command:
  ```
  cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_free_of.py::TestFreeBindingOrder -v
  ```
  Expected: all 4 tests PASS, including `test_free_left_of_var_does_not_match` now returning `None`.

- [ ] **Step 7: Run the existing matcher + rewriter suites, expect PASS (no regressions).**
  Command:
  ```
  cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_rewriter.py rerum/tests/test_strategies.py rerum/tests/test_bidirectional.py -v
  ```
  Expected: all pass; the public `match` rename and the recursion redirect do not change any matching outcome except the corrected `?free` cases.

- [ ] **Step 8: Commit.**
  ```
  cd /home/spinoza/github/repos/rerum && git add rerum/rewriter.py rerum/tests/test_free_of.py
  git commit -m "fix(match): evaluate ?x:free(v) against final bindings

  The ?free constraint was checked the instant its pattern node was
  visited, so when the excluded variable v was bound to the right of the
  ?free pattern (e.g. (dd ?f:free(v) ?v:var)) it was still unbound and the
  check passed vacuously. Split match into a recursive core plus a public
  wrapper that re-validates every ?free node against the final resolved
  bindings. (dd ?f:free(v) ?v:var) no longer matches (dd (sin x) x). This
  is a general matcher-correctness fix; dd is just an arbitrary operator
  symbol in the test data.

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task 3: Fix the guard-on-undefined-op footgun in `_check_condition`

A guard is always a `(! op ...)` compute form. When `op` is absent from the active prelude (and no resolver handles it), `instantiate` leaves the form as a bare compound list `[op, ...args]` (rewriter.py line 849). `_condition_truthy` (engine.py line 338) then sees a non-empty list and returns `True`, so the guard passes silently and the rule fires with a corrupt result. Guards must be decidable: an undefined-op guard must raise `ValueError("guard references undefined op ...")` instead of evaluating truthy. Valid guards (all ops in the prelude) must keep working. This is a general guard-correctness fix; it references no domain.

Detection is structural, not heuristic: a guard condition top-level form is `["!", op, ...]`; if `op` is not in `self._fold_funcs` and there is no resolver that supplied it, the guard is undecidable. To also catch undefined ops nested inside a decidable head (e.g. `(! and (! undefined? x) ...)`), scan the condition's compute nodes before evaluating.

**Files:**
- Modify: `rerum/engine.py` (`_check_condition` at line 1907; `self._fold_funcs` is the active prelude; `skeleton_compute` is importable from rewriter)
- Test: `rerum/tests/test_guards.py` (add `TestGuardUndefinedOp`)

- [ ] **Step 1: Write failing test for undefined-op guard raising.**
  Append this class to `rerum/tests/test_guards.py`:

  ```python
  class TestGuardUndefinedOp:
      """A guard referencing an op absent from the active prelude must raise."""

      def test_undefined_op_guard_raises(self):
          """An undefined op in a guard raises ValueError rather than passing truthy."""
          engine = (RuleEngine()
              .with_prelude(PREDICATE_PRELUDE)
              .load_dsl("@bad: (f ?x) => (g :x) when (! no-such-op? :x)"))
          with pytest.raises(ValueError, match="undefined op"):
              engine(E("(f 5)"))

      def test_nested_undefined_op_guard_raises(self):
          """An undefined op nested inside a decidable head still raises."""
          engine = (RuleEngine()
              .with_prelude(PREDICATE_PRELUDE)
              .load_dsl("@bad: (f ?x) => (g :x) when (! and (! const? :x) (! mystery? :x))"))
          with pytest.raises(ValueError, match="undefined op"):
              engine(E("(f 5)"))

      def test_defined_op_guard_does_not_raise(self):
          """A guard whose ops are all in the prelude evaluates normally."""
          engine = (RuleEngine()
              .with_prelude(PREDICATE_PRELUDE)
              .load_dsl("@ok: (f ?x) => (g :x) when (! const? :x)"))
          assert engine(E("(f 5)")) == ["g", 5]
          # Guard false (x is a var, not const): rule does not fire, no raise.
          assert engine(E("(f y)")) == ["f", "y"]
  ```

  Confirm `pytest`, `RuleEngine`, `E`, and `PREDICATE_PRELUDE` are imported at the top of `test_guards.py`; add any missing import.

- [ ] **Step 2: Run the test, expect FAIL.**
  Command:
  ```
  cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_guards.py::TestGuardUndefinedOp -v
  ```
  Expected: `test_undefined_op_guard_raises` and `test_nested_undefined_op_guard_raises` FAIL with `DID NOT RAISE <class 'ValueError'>` (today the guard passes truthy and `(f 5)` rewrites to `["g", 5]`). `test_defined_op_guard_does_not_raise` should already PASS.

- [ ] **Step 3: Add the undefined-op scan and raise in `_check_condition`.**
  First, extend the rewriter import block in `rerum/engine.py` (the `from .rewriter import (...)` block beginning at line 57) to also import `skeleton_compute`:

  ```python
  from .rewriter import (
      rewriter, match as _match_internal, instantiate, ExprType,
      skeleton_compute,
  ```

  (Add `skeleton_compute` to the existing import list; keep the rest of that import block unchanged.)

  Then modify `_check_condition` (line 1907). Before the `instantiate` call (currently line 1923), scan every compute node in the condition for an op the engine cannot decide, and raise. Replace the method body from the `if condition is None:` guard onward with:

  ```python
          if condition is None:
              return True

          # Guards must be decidable: every (! op ...) compute node in the
          # condition must reference an op the engine can fold. An op absent
          # from the active prelude (and not supplied by a resolver) would
          # otherwise leave an unfolded compound that _condition_truthy reads
          # as truthy, silently passing a bogus guard. Raise instead.
          undefined = self._undefined_guard_ops(condition)
          if undefined:
              raise ValueError(
                  f"guard references undefined op {sorted(undefined)!r}; "
                  f"guards must be decidable (add the op to the prelude)"
              )

          # Instantiate the condition with bindings, then apply the shared
          # truthiness rule (bool as-is; 0/""/[] falsy; everything else truthy).
          result = instantiate(condition, bindings, self._fold_funcs,
                               undefined_op_resolver=self._undefined_op_resolver,
                               fold_error_resolver=self._fold_error_resolver)
          return _condition_truthy(result)
  ```

- [ ] **Step 4: Add the `_undefined_guard_ops` scanner helper.**
  In `rerum/engine.py`, directly after `_check_condition`, add:

  ```python
      def _undefined_guard_ops(self, condition: ExprType) -> set:
          """Return the set of compute-op names in ``condition`` that the
          engine cannot decide.

          Walks every ``(! op ...)`` node. An op is undecidable when it is
          absent from the active prelude and no ``undefined_op_resolver``
          can supply it. The resolver, when present, is treated as able to
          supply any op (it is consulted lazily during instantiation), so a
          configured resolver suppresses the raise and defers to runtime.
          """
          undefined = set()

          def walk(node):
              if not isinstance(node, list) or not node:
                  return
              if skeleton_compute(node):
                  op = node[1]
                  if (op not in self._fold_funcs
                          and self._undefined_op_resolver is None):
                      undefined.add(op)
                  for arg in node[2:]:
                      walk(arg)
                  return
              for child in node:
                  walk(child)

          walk(condition)
          return undefined
  ```

  Note: `skeleton_compute(node)` (rewriter.py line 550) returns True for `["!", op, ...]`. `self._fold_funcs` holds the active prelude dict; `self._undefined_op_resolver` is the hook bridge (None when no resolver is registered). When a resolver IS registered, the scan defers to runtime, preserving the LLM-rule-inference path documented in CLAUDE.md.

- [ ] **Step 5: Run the new guard test, expect PASS.**
  Command:
  ```
  cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_guards.py::TestGuardUndefinedOp -v
  ```
  Expected: all 3 tests PASS.

- [ ] **Step 6: Run the full existing guard suite, expect PASS (no regressions).**
  Command:
  ```
  cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_guards.py -v
  ```
  Expected: every pre-existing guard test still passes; all guards in those tests use prelude ops (`>`, `<`, `=`, `const?`, `var?`, `and`, etc.), so none trip the new raise.

- [ ] **Step 7: Run the hook and examples-validation suites, expect PASS.**
  The examples validator (`_validate_example`, engine.py line 355) instantiates conditions directly and does NOT call `_check_condition`, so it is unaffected; the hook path uses a resolver, which the scanner defers to. Confirm:
  ```
  cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_hooks.py rerum/tests/test_hooks_integration.py rerum/tests/test_examples_validation.py -v
  ```
  Expected: all pass.

- [ ] **Step 8: Commit.**
  ```
  cd /home/spinoza/github/repos/rerum && git add rerum/engine.py rerum/tests/test_guards.py
  git commit -m "fix(guards): raise on undefined-op guard instead of passing truthy

  A guard (! op ...) whose op is absent from the active prelude (and not
  supplied by a resolver) left an unfolded compound that _condition_truthy
  read as truthy, silently firing the rule with a corrupt result. Scan the
  condition for undecidable compute ops and raise ValueError. A registered
  undefined_op_resolver defers the decision to runtime, preserving the
  LLM rule-inference path.

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task 4: Add and export the general `combine_preludes` helper

Per the general-engine principle (spec Section 0 and Section 5.7), the engine ships NO domain-named bundle. Instead it provides one general way to compose preludes: `combine_preludes(*preludes: dict) -> dict` merges fold dicts left-to-right (later wins on key conflict) and returns a fresh dict (it must not mutate its inputs). A rule set that needs both math functions and predicates simply documents it requires `combine_preludes(MATH_PRELUDE, PREDICATE_PRELUDE)`; calculus is just one such consumer, and that combination is data the caller assembles, not engine-resident domain knowledge. Export the helper from `rerum/__init__.py`.

**Files:**
- Modify: `rerum/rewriter.py` (after `NO_PRELUDE`, line 368; uses `FoldFuncsType` at line 213, `MATH_PRELUDE` at line 304, `PREDICATE_PRELUDE` at line 339)
- Modify: `rerum/__init__.py` (import block lines 43-71; `__all__` lines 130-141)
- Test: `rerum/tests/test_preludes.py` (create)

- [ ] **Step 1: Write failing test for `combine_preludes`.**
  Create `rerum/tests/test_preludes.py` with this content:

  ```python
  """Tests for the general combine_preludes helper (no domain bundle)."""

  import pytest


  class TestCombinePreludes:
      """combine_preludes merges fold dicts left-to-right, later-wins, fresh dict."""

      def test_combine_preludes_importable(self):
          """combine_preludes is importable from the package root."""
          from rerum import combine_preludes
          assert callable(combine_preludes)

      def test_combine_two_dicts_merges_keys(self):
          """The result contains every key from each input prelude."""
          from rerum import combine_preludes
          a = {"f": 1, "g": 2}
          b = {"h": 3}
          merged = combine_preludes(a, b)
          assert merged == {"f": 1, "g": 2, "h": 3}

      def test_combine_later_wins_on_conflict(self):
          """When a key appears in more than one prelude, the later prelude wins."""
          from rerum import combine_preludes
          a = {"f": 1, "g": 2}
          b = {"g": 99}
          merged = combine_preludes(a, b)
          assert merged["g"] == 99

      def test_combine_returns_fresh_dict_no_mutation(self):
          """combine_preludes returns a new dict and does not mutate its inputs."""
          from rerum import combine_preludes
          a = {"f": 1}
          b = {"g": 2}
          merged = combine_preludes(a, b)
          assert merged is not a
          assert merged is not b
          # Inputs unchanged.
          assert a == {"f": 1}
          assert b == {"g": 2}
          # Mutating the result does not touch the inputs.
          merged["z"] = 0
          assert "z" not in a and "z" not in b

      def test_combine_empty_returns_empty_dict(self):
          """combine_preludes() with no args returns a fresh empty dict."""
          from rerum import combine_preludes
          assert combine_preludes() == {}

      def test_combine_math_and_predicates(self):
          """combine_preludes(MATH_PRELUDE, PREDICATE_PRELUDE) has sin, const?, and free-of?."""
          from rerum import combine_preludes, MATH_PRELUDE, PREDICATE_PRELUDE
          merged = combine_preludes(MATH_PRELUDE, PREDICATE_PRELUDE)
          assert "sin" in merged
          assert "const?" in merged
          assert "free-of?" in merged

      def test_combine_math_and_predicates_usable_in_engine(self):
          """A guarded rule loads and fires under the combined prelude."""
          from rerum import combine_preludes, MATH_PRELUDE, PREDICATE_PRELUDE
          from rerum import RuleEngine, E
          prelude = combine_preludes(MATH_PRELUDE, PREDICATE_PRELUDE)
          engine = (RuleEngine()
              .with_prelude(prelude)
              .load_dsl("@free: (dd ?f ?v) => 0 when (! free-of? :f :v)"))
          assert engine(E("(dd (sin y) x)")) == 0
  ```

- [ ] **Step 2: Run the test, expect FAIL.**
  Command:
  ```
  cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_preludes.py -v
  ```
  Expected: `test_combine_preludes_importable` FAILS with `ImportError: cannot import name 'combine_preludes' from 'rerum'`; the rest error on the same import.

- [ ] **Step 3: Define `combine_preludes` in `rewriter.py`.**
  In `rerum/rewriter.py`, after the `NO_PRELUDE` definition (line 368), add the general helper:

  ```python
  def combine_preludes(*preludes: FoldFuncsType) -> FoldFuncsType:
      """Merge fold-function dicts left-to-right into a fresh dict.

      This is the general way a rule set composes the preludes it needs:
      ``combine_preludes(MATH_PRELUDE, PREDICATE_PRELUDE)`` yields a prelude
      with both math functions and predicates. Later preludes win on key
      conflict. The result is a new dict; inputs are not mutated.

      The engine ships no domain-named bundle. A rule set documents the
      combination it requires and assembles it via this helper as data.
      """
      merged: FoldFuncsType = {}
      for prelude in preludes:
          merged.update(prelude)
      return merged
  ```

- [ ] **Step 4: Export `combine_preludes` from `rerum/__init__.py`.**
  In the `from .rewriter import (...)` block, add `combine_preludes` to the `# Fold operation builders` group (after `safe_div`, before the `# Standard preludes` comment):

  ```python
      # Fold operation builders
      nary_fold,
      unary_only,
      binary_only,
      special_minus,
      safe_div,
      combine_preludes,
  ```

  And in `__all__`, add it to the matching group (after `"safe_div"`):

  ```python
      "nary_fold",
      "unary_only",
      "binary_only",
      "special_minus",
      "safe_div",
      "combine_preludes",
  ```

- [ ] **Step 5: Run the test, expect PASS.**
  Command:
  ```
  cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_preludes.py -v
  ```
  Expected: all 7 tests PASS.

- [ ] **Step 6: Commit.**
  ```
  cd /home/spinoza/github/repos/rerum && git add rerum/rewriter.py rerum/__init__.py rerum/tests/test_preludes.py
  git commit -m "feat(prelude): add general combine_preludes helper

  combine_preludes(*preludes) merges fold dicts left-to-right (later wins)
  into a fresh dict, without mutating inputs. This is the general,
  domain-agnostic way a rule set composes the preludes it needs, e.g.
  combine_preludes(MATH_PRELUDE, PREDICATE_PRELUDE). The engine ships no
  domain-named bundle (no CALCULUS_PRELUDE); calculus is just one example
  consumer that documents this combination. Exported from rerum.

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task 5: Full-suite verification

- [ ] **Step 1: Run the entire test suite, expect PASS.**
  Command:
  ```
  cd /home/spinoza/github/repos/rerum && python -m pytest
  ```
  Expected: all tests green, including the new `test_free_of.py` (2 classes: `TestFreeOfPredicate` 5 tests, `TestFreeBindingOrder` 4 tests), the new `test_preludes.py` (`TestCombinePreludes`, 7 tests), and the added `TestGuardUndefinedOp` (3 tests) in `test_guards.py`. No regressions in `test_rewriter`, `test_strategies`, `test_bidirectional`, `test_guards`, `test_hooks*`, or `test_examples_validation`.

- [ ] **Step 2: Run with branch coverage to confirm the new paths are exercised.**
  Command:
  ```
  cd /home/spinoza/github/repos/rerum && python -m pytest --cov=rerum --cov-report=term-missing
  ```
  Expected: `_check_free_constraints`, the `free-of?` lambda, `_undefined_guard_ops`, and `combine_preludes` all show as covered (no new uncovered lines introduced by this phase).

- [ ] **Step 3: Re-confirm the documented failing case at the REPL.**
  Command:
  ```
  cd /home/spinoza/github/repos/rerum && python -c "from rerum.rewriter import match; from rerum.engine import parse_sexpr; print(match(parse_sexpr('(dd ?f:free(v) ?v:var)'), parse_sexpr('(dd (sin x) x)')))"
  ```
  Expected output: `None` (the buggy match no longer succeeds).

- [ ] **Step 4: Confirm `combine_preludes` composes a usable prelude at the REPL.**
  Command:
  ```
  cd /home/spinoza/github/repos/rerum && python -c "from rerum import combine_preludes, MATH_PRELUDE, PREDICATE_PRELUDE; p = combine_preludes(MATH_PRELUDE, PREDICATE_PRELUDE); print('sin' in p, 'const?' in p, 'free-of?' in p)"
  ```
  Expected output: `True True True`.

- [ ] **Step 5: Run the example rule files as a smoke test.**
  Command:
  ```
  cd /home/spinoza/github/repos/rerum && python -m pytest rerum/tests/test_includes.py rerum/tests/test_cli.py -v
  ```
  Expected: pass; the existing example `.rules` files still load (they do not rely on the corrected `?free` semantics in a way that changes behavior).

- [ ] **Step 6: Commit any incidental fixes (only if a regression surfaced and was fixed).**
  If steps 1-5 revealed a regression that required a follow-up edit, commit it:
  ```
  cd /home/spinoza/github/repos/rerum && git add -A
  git commit -m "test(phase0): verify foundation fixes across full suite

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```
  If no follow-up was needed, skip this step.

---

## Phase 0 Done When

- [ ] `free-of?` is in `PREDICATE_PRELUDE`; `(! free-of? f v)` is True iff symbol `v` does not occur in `f`. It is a structural predicate, not domain code.
- [ ] The documented failing case now correctly fails to match: `match((dd ?f:free(v) ?v:var), (dd (sin x) x))` returns `None`.
- [ ] `(dd ?f:free(v) ?v:var)` still matches `(dd (sin y) x)` (genuinely free) and `(dd 5 x)`.
- [ ] A guard referencing an op absent from the active prelude raises `ValueError("guard references undefined op ...")`, both at the top level and nested; valid guards still evaluate (`test_guards.py` fully green).
- [ ] `combine_preludes(*preludes)` exists in `rewriter.py`, merges fold dicts left-to-right (later wins), returns a fresh dict that does not mutate its inputs, and is importable as `from rerum import combine_preludes`.
- [ ] `combine_preludes(MATH_PRELUDE, PREDICATE_PRELUDE)` contains `sin`, `const?`, and `free-of?`.
- [ ] There is NO `CALCULUS_PRELUDE` (or any other domain-named bundle) anywhere in `rerum/`; the engine ships only computation-named preludes plus the general `combine_preludes` helper. A rule set documents the combination it needs.
- [ ] All new tests pass: `rerum/tests/test_free_of.py` (2 classes), `rerum/tests/test_preludes.py` (`TestCombinePreludes`), and `TestGuardUndefinedOp` in `rerum/tests/test_guards.py`.
- [ ] The full suite is still green: `cd /home/spinoza/github/repos/rerum && python -m pytest` passes with no regressions.
