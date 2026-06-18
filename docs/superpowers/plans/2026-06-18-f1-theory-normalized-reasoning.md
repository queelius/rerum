# F1: Theory-Normalized Reasoning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make RERUM's five equational-reasoning methods reason MODULO an equational theory by routing expression identity through a single `_canonicalize` seam, so `prove_equal(x+y, y+x)` holds with no search and equivalence classes collapse AC-variants.

**Architecture:** Add one private helper `RuleEngine._canonicalize(expr)` that returns `expr` when no theory is set and `normalize(expr, theory)` otherwise. Thread it into `equivalents` and `prove_equal` at the start-node, neighbor, and quick-check sites; `enumerate_equivalents` and `are_equal` inherit for free; `minimize` needs its raw baseline seeded through the same seam. No new algorithm; the already-built `rerum/normalize.py` does the work.

**Tech Stack:** Python 3.9+, pytest. Pure-list `ExprType` expressions. `rerum.normalize.Theory` + `rerum.normalize.normalize`.

**Spec:** `docs/superpowers/specs/2026-06-18-f1-theory-normalized-core-design.md` (read it first; this plan implements it exactly, including the documented Soundness boundary).

**Constraints (from CLAUDE.md and the spec):**
- ASCII only in all files (a commit hook rejects non-ASCII such as em-dashes).
- General-engine principle: no domain operator is hardcoded in `rerum/`; theories are DATA. F1 touches only `rerum/engine.py` and adds one test file.
- Backward-compat is the prime directive: with no theory set, behavior is byte-for-byte unchanged. The existing suite must show no new failures.
- Do NOT commit `.mcp.json` (untracked local config). Stage only the files named in each task.

---

## File Structure

- **Modify** `rerum/engine.py`:
  - Add `RuleEngine._canonicalize` (new method, near the other reasoning helpers).
  - Edit `equivalents` (currently ~line 3663): start node, neighbor, frontier.
  - Edit `prove_equal` (currently ~line 3797): start keys/seed, both expansion loops.
  - Edit `minimize` (currently ~line 4081): baseline seed + identity check.
  - `enumerate_equivalents` (~3772) and `are_equal` (~4050): NO change (inherit).
- **Create** `rerum/tests/test_theory_reasoning.py`: the one-file-per-feature test suite for F1.

> Line numbers drift as edits land. Each task anchors edits on a unique CURRENT code snippet to find, not just a line number.

---

### Task 1: The `_canonicalize` seam

**Files:**
- Modify: `rerum/engine.py` (add a method to `RuleEngine`)
- Test: `rerum/tests/test_theory_reasoning.py` (create)

- [ ] **Step 1: Create the test file with the canonicalize unit tests**

Create `rerum/tests/test_theory_reasoning.py`:

```python
"""F1: theory-normalized equational reasoning.

Verifies the five reasoning methods (equivalents, enumerate_equivalents,
prove_equal, are_equal, minimize) reason MODULO an equational theory, that
the no-theory path is unchanged, and that the documented soundness boundary
(position-pinning rules under an AC theory) holds as intended behavior.

GENERAL ENGINE: theories are DATA. The boolean fixture proves the same engine
code reasons over a non-arithmetic AC theory with no code change.
"""

from rerum.engine import RuleEngine
from rerum.normalize import Theory


# --- Theory fixtures (DATA; no operator is special-cased in rerum/) ---------

AC_PLUS = Theory.from_dict({"+": {"ac": True, "identity": 0}})
AC_TIMES = Theory.from_dict({"*": {"ac": True, "annihilator": 0}})
AC_BOOL = Theory.from_dict({"and": {"ac": True}, "or": {"ac": True}})


class TestCanonicalizeSeam:
    def test_no_theory_is_identity(self):
        eng = RuleEngine()
        expr = ["+", "b", "a"]
        # Identity function: returns the argument unchanged (same value).
        assert eng._canonicalize(expr) == ["+", "b", "a"]

    def test_theory_returns_normal_form(self):
        eng = RuleEngine().with_theory(AC_PLUS)
        # AC + sorts operands: (+ b a) canonicalizes to (+ a b).
        assert eng._canonicalize(["+", "b", "a"]) == ["+", "a", "b"]

    def test_theory_collapses_identity_unit(self):
        eng = RuleEngine().with_theory(AC_PLUS)
        # identity 0 is dropped, single operand unwraps: (+ x 0) -> x.
        assert eng._canonicalize(["+", "x", 0]) == "x"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest rerum/tests/test_theory_reasoning.py::TestCanonicalizeSeam -v`
Expected: FAIL with `AttributeError: 'RuleEngine' object has no attribute '_canonicalize'`.

- [ ] **Step 3: Add the `_canonicalize` method to `RuleEngine`**

In `rerum/engine.py`, find the `with_theory` method (its body ends with `return self` at the `self._simplifier = None` block). Immediately AFTER the `with_theory` method, insert:

```python
    def _canonicalize(self, expr: ExprType) -> ExprType:
        """Canonical form of ``expr`` under the engine's theory, else unchanged.

        Identity function when no theory is set (``self._theory is None``) --
        this is the backward-compat path that keeps the no-theory reasoning
        behavior byte-for-byte unchanged. When a theory is set, returns
        ``normalize(expr, theory)``, which is idempotent and confluent, so the
        result is the single, well-defined IDENTITY of the expression for the
        reasoning layer (equivalents / prove_equal / minimize key on it).

        The theory is DATA; no operator is special-cased here.
        """
        if self._theory is None:
            return expr
        from .normalize import normalize as _normalize
        return _normalize(expr, self._theory)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest rerum/tests/test_theory_reasoning.py::TestCanonicalizeSeam -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add rerum/engine.py rerum/tests/test_theory_reasoning.py
git commit -m "feat(f1): add _canonicalize seam for theory-normalized reasoning

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `equivalents` modulo theory

**Files:**
- Modify: `rerum/engine.py` (`equivalents`)
- Test: `rerum/tests/test_theory_reasoning.py`

- [ ] **Step 1: Write the failing tests**

Append to `rerum/tests/test_theory_reasoning.py`:

```python
def _comm_plus_engine():
    """Engine with a single commutativity rule for +."""
    return RuleEngine.from_dsl("@comm: (+ ?x ?y) <=> (+ :y :x)")


class TestEquivalentsModuloTheory:
    def test_ac_class_dedups_with_theory(self):
        eng = _comm_plus_engine().with_theory(AC_PLUS)
        members = eng.enumerate_equivalents(["+", "a", "b"], max_depth=3)
        # (+ a b) and (+ b a) share a canonical key -> one class member.
        assert len(members) == 1
        # The yielded form is canonical (sorted).
        assert members[0] == ["+", "a", "b"]

    def test_same_class_without_theory_has_both_arrangements(self):
        eng = _comm_plus_engine()  # no theory
        members = eng.enumerate_equivalents(["+", "a", "b"], max_depth=3)
        # Without the theory, the commute rule yields both arrangements.
        assert len(members) == 2
        assert ["+", "a", "b"] in members
        assert ["+", "b", "a"] in members

    def test_every_yielded_form_is_canonical_and_unique(self):
        eng = _comm_plus_engine().with_theory(AC_PLUS)
        members = eng.enumerate_equivalents(["+", "c", "a", "b"], max_depth=5)
        # Each yielded form equals its own normal form (a dedup guarantee).
        for m in members:
            assert eng._canonicalize(m) == m
        # No duplicate canonical keys among the yielded members.
        keys = [tuple(eng._canonicalize(m)) if isinstance(m, list) else m
                for m in members]
        assert len(set(keys)) == len(members)

    def test_no_theory_output_unchanged_value_and_order(self):
        # Backward-compat at the VALUE+ORDER level: identical generator output.
        eng = _comm_plus_engine()
        out = list(eng.equivalents(["+", "a", "b"], max_depth=3))
        assert out == [["+", "a", "b"], ["+", "b", "a"]]
```

- [ ] **Step 2: Run the tests to verify the theory cases fail**

Run: `pytest rerum/tests/test_theory_reasoning.py::TestEquivalentsModuloTheory -v`
Expected: `test_ac_class_dedups_with_theory` and `test_every_yielded_form_is_canonical_and_unique` FAIL (without the seam, the theory engine still yields both arrangements, so the dedup count is 2 not 1, and `(+ b a)` is non-canonical). The two no-theory tests PASS already.

- [ ] **Step 3: Thread `_canonicalize` into `equivalents`**

In `rerum/engine.py`, inside `equivalents`, find this CURRENT code:

```python
        # Track visited expressions
        visited: Set[tuple] = set()
        start_key = _expr_to_tuple(expr)
        visited.add(start_key)

        # Count of yielded expressions
        count = 0

        # Yield the starting expression
        yield expr
```

Replace it with (canonicalize the start node):

```python
        # Track visited expressions. Under a theory, identity is the NORMAL
        # FORM: the class is the set of canonical representatives. With no
        # theory _canonicalize is the identity, so this is unchanged.
        cexpr = self._canonicalize(expr)
        visited: Set[tuple] = set()
        start_key = _expr_to_tuple(cexpr)
        visited.add(start_key)

        # Count of yielded expressions
        count = 0

        # Yield the starting (canonical) expression
        yield cexpr
```

Next, find the CURRENT frontier seeding:

```python
        # Initialize queue/stack with (expression, depth)
        if strategy == "bfs":
            frontier: deque = deque([(expr, 0)])
        else:  # dfs
            frontier: List = [(expr, 0)]
```

Replace with (seed the frontier with the canonical start):

```python
        # Initialize queue/stack with (canonical expression, depth)
        if strategy == "bfs":
            frontier: deque = deque([(cexpr, 0)])
        else:  # dfs
            frontier: List = [(cexpr, 0)]
```

Finally, find the CURRENT neighbor loop:

```python
            for new_expr in rewrites:
                key = _expr_to_tuple(new_expr)
                if key not in visited:
                    visited.add(key)
                    yield new_expr
                    count += 1
```

Replace with (canonicalize each neighbor before keying/yielding):

```python
            for new_expr in rewrites:
                cnew = self._canonicalize(new_expr)
                key = _expr_to_tuple(cnew)
                if key not in visited:
                    visited.add(key)
                    yield cnew
                    count += 1
```

And find the CURRENT frontier append (a few lines below, inside that same `if`):

```python
                    frontier.append((new_expr, depth + 1))
```

Replace with (explore from the canonical form):

```python
                    frontier.append((cnew, depth + 1))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest rerum/tests/test_theory_reasoning.py::TestEquivalentsModuloTheory -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add rerum/engine.py rerum/tests/test_theory_reasoning.py
git commit -m "feat(f1): equivalents reasons modulo theory (canonical-rep class)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `prove_equal` modulo theory

**Files:**
- Modify: `rerum/engine.py` (`prove_equal`)
- Test: `rerum/tests/test_theory_reasoning.py`

- [ ] **Step 1: Write the failing tests**

Append to `rerum/tests/test_theory_reasoning.py`:

```python
class TestProveEqualModuloTheory:
    def test_commute_holds_instantly_with_theory(self):
        # No commute rule loaded: equality holds ONLY via the theory.
        eng = RuleEngine().with_theory(AC_PLUS)
        proof = eng.prove_equal(["+", "x", "y"], ["+", "y", "x"])
        assert proof is not None
        # Zero-step: the canonical keys match, so the quick check fires.
        assert proof.depth_a == 0 and proof.depth_b == 0

    def test_commute_not_provable_without_theory(self):
        eng = RuleEngine()  # no theory, no commute rule
        proof = eng.prove_equal(["+", "x", "y"], ["+", "y", "x"])
        assert proof is None

    def test_associativity_and_commutativity_modulo_theory(self):
        eng = RuleEngine().with_theory(AC_PLUS)
        proof = eng.prove_equal(["+", ["+", "a", "b"], "c"],
                                ["+", "a", ["+", "c", "b"]])
        assert proof is not None

    def test_proof_path_states_are_canonical_no_normalize_steps(self):
        # A real proof under a theory: every step.after is the canonical state,
        # and no step is a kind="normalize" micro-step. Distinct operands a, b
        # avoid the idempotent-collapse of (+ a a) (AC_PLUS has no "repeat"
        # rule, so it is a join-semilattice on repeated operands).
        eng = RuleEngine.from_dsl("@f: (f ?x ?y) => (+ :x :y)")
        eng.with_theory(AC_PLUS)
        proof = eng.prove_equal(["f", "b", "a"], ["+", "a", "b"],
                                include_unidirectional=True, trace=True)
        assert proof is not None
        for step in (proof.path_a or []):
            assert step.kind != "normalize"
            if isinstance(step.after, list):
                assert eng._canonicalize(step.after) == step.after
```

- [ ] **Step 2: Run the tests to verify the theory cases fail**

Run: `pytest rerum/tests/test_theory_reasoning.py::TestProveEqualModuloTheory -v`
Expected: `test_commute_holds_instantly_with_theory`, `test_associativity_and_commutativity_modulo_theory`, and `test_proof_path_states_are_canonical_no_normalize_steps` FAIL (without the seam the keys differ, so no quick-check match and no path within budget). `test_commute_not_provable_without_theory` PASSES already.

- [ ] **Step 3: Thread `_canonicalize` into `prove_equal`**

In `rerum/engine.py`, inside `prove_equal`, find the CURRENT key setup:

```python
        # Convert to hashable for set operations
        key_a = _expr_to_tuple(expr_a)
        key_b = _expr_to_tuple(expr_b)
```

Replace with (canonical keys; AC-variants now collide at the quick check):

```python
        # Convert to hashable for set operations. Under a theory the key is
        # the NORMAL FORM, so AC-variant inputs collide here and the quick
        # check below returns a zero-step proof. With no theory _canonicalize
        # is the identity (unchanged behavior).
        ca = self._canonicalize(expr_a)
        cb = self._canonicalize(expr_b)
        key_a = _expr_to_tuple(ca)
        key_b = _expr_to_tuple(cb)
```

(The quick-check branch immediately below uses `expr_a` for `common`/`before`/`after`; leave it as-is. The zero-step proof reports the user's own input as the common form, which is correct and avoids touching the existing zero-step proof contract.)

Next, find the CURRENT visited/frontier seeding:

```python
        visited_a: Dict[tuple, Tuple[ExprType, int, Optional[tuple], Optional[dict]]] = {
            key_a: (expr_a, 0, None, None)
        }
        visited_b: Dict[tuple, Tuple[ExprType, int, Optional[tuple], Optional[dict]]] = {
            key_b: (expr_b, 0, None, None)
        }
```

Replace with (store the canonical start so reconstructed `.after` values are canonical):

```python
        visited_a: Dict[tuple, Tuple[ExprType, int, Optional[tuple], Optional[dict]]] = {
            key_a: (ca, 0, None, None)
        }
        visited_b: Dict[tuple, Tuple[ExprType, int, Optional[tuple], Optional[dict]]] = {
            key_b: (cb, 0, None, None)
        }
```

Find the CURRENT frontier init:

```python
        # BFS frontiers: (expression, depth)
        frontier_a: deque = deque([(expr_a, 0)])
        frontier_b: deque = deque([(expr_b, 0)])
```

Replace with:

```python
        # BFS frontiers carry CANONICAL states (expression, depth).
        frontier_a: deque = deque([(ca, 0)])
        frontier_b: deque = deque([(cb, 0)])
```

Now the two expansion loops. Find the CURRENT "Expand from A" neighbor handling:

```python
                    for new_expr, label in rewrites:
                        new_key = _expr_to_tuple(new_expr)
                        if new_key not in visited_a:
                            visited_a[new_key] = (new_expr, depth + 1, current_key, label)
                            frontier_a.append((new_expr, depth + 1))

                            # Check for intersection
                            if new_key in visited_b:
                                _, depth_b, _, _ = visited_b[new_key]
```

Replace with (canonicalize neighbor; store + queue + report the canonical form):

```python
                    for new_expr, label in rewrites:
                        cnew = self._canonicalize(new_expr)
                        new_key = _expr_to_tuple(cnew)
                        if new_key not in visited_a:
                            visited_a[new_key] = (cnew, depth + 1, current_key, label)
                            frontier_a.append((cnew, depth + 1))

                            # Check for intersection
                            if new_key in visited_b:
                                _, depth_b, _, _ = visited_b[new_key]
```

In that same A-branch, find the CURRENT `EqualityProof` construction and change its `common`:

```python
                                return EqualityProof(
                                    expr_a=expr_a,
                                    expr_b=expr_b,
                                    common=new_expr,
                                    depth_a=depth + 1,
                                    depth_b=depth_b,
                                    path_a=path_a,
                                    path_b=path_b
                                )
```

Replace `common=new_expr` with `common=cnew` (leave the rest):

```python
                                return EqualityProof(
                                    expr_a=expr_a,
                                    expr_b=expr_b,
                                    common=cnew,
                                    depth_a=depth + 1,
                                    depth_b=depth_b,
                                    path_a=path_a,
                                    path_b=path_b
                                )
```

Now the "Expand from B" branch. Find the CURRENT:

```python
                    for new_expr, label in rewrites:
                        new_key = _expr_to_tuple(new_expr)
                        if new_key not in visited_b:
                            visited_b[new_key] = (new_expr, depth + 1, current_key, label)
                            frontier_b.append((new_expr, depth + 1))

                            # Check for intersection
                            if new_key in visited_a:
                                _, depth_a_val, _, _ = visited_a[new_key]
```

Replace with:

```python
                    for new_expr, label in rewrites:
                        cnew = self._canonicalize(new_expr)
                        new_key = _expr_to_tuple(cnew)
                        if new_key not in visited_b:
                            visited_b[new_key] = (cnew, depth + 1, current_key, label)
                            frontier_b.append((cnew, depth + 1))

                            # Check for intersection
                            if new_key in visited_a:
                                _, depth_a_val, _, _ = visited_a[new_key]
```

And in that B-branch, find the CURRENT `EqualityProof` and change `common=new_expr` to `common=cnew`:

```python
                                return EqualityProof(
                                    expr_a=expr_a,
                                    expr_b=expr_b,
                                    common=cnew,
                                    depth_a=depth_a_val,
                                    depth_b=depth + 1,
                                    path_a=path_a,
                                    path_b=path_b
                                )
```

(Note: `current_key = _expr_to_tuple(current)` in each branch is unchanged. `current` is popped from a frontier that now holds canonical forms, so `current_key` is already the canonical key, consistent with the visited dicts.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest rerum/tests/test_theory_reasoning.py::TestProveEqualModuloTheory -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Run the existing prove_equal suite as a regression guard**

Run: `pytest rerum/tests/test_prove_equal.py -q`
Expected: PASS (no theory is set in those tests, so behavior is unchanged).

- [ ] **Step 6: Commit**

```bash
git add rerum/engine.py rerum/tests/test_theory_reasoning.py
git commit -m "feat(f1): prove_equal reasons modulo theory (canonical keys + states)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: `are_equal`, `enumerate_equivalents`, units, and domain-freeness (inheritance, no engine code)

**Files:**
- Test: `rerum/tests/test_theory_reasoning.py` (these pass once Tasks 2-3 land; they lock in inherited behavior)

- [ ] **Step 1: Write the inheritance + units + domain-free tests**

Append to `rerum/tests/test_theory_reasoning.py`:

```python
class TestInheritedAndUnits:
    def test_are_equal_true_for_ac_variants(self):
        eng = RuleEngine().with_theory(AC_PLUS)
        assert eng.are_equal(["+", "x", "y"], ["+", "y", "x"]) is True

    def test_are_equal_false_without_theory(self):
        eng = RuleEngine()  # no theory, no commute rule
        assert eng.are_equal(["+", "x", "y"], ["+", "y", "x"]) is False

    def test_identity_unit_collapses_into_class(self):
        eng = RuleEngine().with_theory(AC_PLUS)
        # (+ x 0) and x are the same modulo the identity unit.
        assert eng.are_equal(["+", "x", 0], "x") is True

    def test_annihilator_unit_collapses_into_class(self):
        eng = RuleEngine().with_theory(AC_TIMES)
        # (* x 0) and 0 are the same modulo the annihilator unit.
        assert eng.are_equal(["*", "x", 0], 0) is True

    def test_domain_free_boolean_theory(self):
        # Same engine code, a non-arithmetic AC theory: (and p q) == (and q p).
        eng = RuleEngine().with_theory(AC_BOOL)
        assert eng.are_equal(["and", "p", "q"], ["and", "q", "p"]) is True
        assert eng.are_equal(["or", "p", "q"], ["or", "q", "p"]) is True
```

- [ ] **Step 2: Run the tests to verify they pass (inherited from Tasks 2-3)**

Run: `pytest rerum/tests/test_theory_reasoning.py::TestInheritedAndUnits -v`
Expected: PASS (6 passed). No engine change is needed; `are_equal` wraps `prove_equal` and the units fall out of `normalize`. If any FAIL, stop: it means Task 3's seam or the theory fixtures are wrong.

- [ ] **Step 3: Commit**

```bash
git add rerum/tests/test_theory_reasoning.py
git commit -m "test(f1): are_equal/units/domain-free inherit theory reasoning

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: `minimize` modulo theory

**Files:**
- Modify: `rerum/engine.py` (`minimize`)
- Test: `rerum/tests/test_theory_reasoning.py`

- [ ] **Step 1: Write the failing tests**

Append to `rerum/tests/test_theory_reasoning.py`:

```python
from rerum.engine import expr_size  # noqa: E402  (cost metric helper)


class TestMinimizeModuloTheory:
    def test_returns_canonical_representative(self):
        eng = RuleEngine.from_dsl("@comm: (+ ?x ?y) <=> (+ :y :x)")
        eng.with_theory(AC_PLUS)
        result = eng.minimize(["+", "b", "a"], metric="size")
        # The result is canonical (sorted), not the raw (+ b a).
        assert eng._canonicalize(result.expr) == result.expr

    def test_no_spurious_improvement_on_normalization_only(self):
        # Input (+ a 0) has raw size 3 but canonical size 1 (a). Without the
        # canonical seed, minimize would measure improvement against the RAW
        # baseline and report ~0.667 -- counting normalization as a win. With
        # the seed, the baseline is the canonical form, so the ratio is 0.0.
        eng = RuleEngine().with_theory(AC_PLUS)
        result = eng.minimize(["+", "a", 0], metric="size")
        assert result.improvement_ratio == 0.0
        assert result.expr == "a"

    def test_mixed_directional_rules_under_theory(self):
        # A => simplification rule that genuinely fires under the theory:
        # @unwrap reduces (id a) to a, then the AC sort canonicalizes the sum.
        # The result is canonical and the derivation is well-formed or None.
        eng = RuleEngine.from_dsl("@unwrap: (id ?x) => :x")
        eng.with_theory(AC_PLUS)
        result = eng.minimize(["+", ["id", "a"], "b"], metric="size",
                              include_unidirectional=True)
        assert eng._canonicalize(result.expr) == result.expr
        assert result.cost <= expr_size(["+", ["id", "a"], "b"])
        # Derivation contract: present-and-wellformed or None, never malformed.
        assert result.derivation is None or result.derivation.final == result.expr
```

- [ ] **Step 2: Run the tests to verify the relevant cases fail**

Run: `pytest rerum/tests/test_theory_reasoning.py::TestMinimizeModuloTheory -v`
Expected: `test_no_spurious_improvement_on_normalization_only` FAILS (without the seed, the raw baseline `(+ a 0)` has size 3 while the found `a` has size 1, so `improvement_ratio` is ~0.667, not 0.0) and `test_returns_canonical_representative` FAILS (without the seed, `best_expr` stays the raw non-canonical `(+ b a)` because the canonical `(+ a b)` is not STRICTLY smaller). `test_mixed_directional_rules_under_theory` already passes (the `=>` reduction is a strict size win independent of the seed); it is pinned as a regression guard.

- [ ] **Step 3: Thread `_canonicalize` into `minimize`'s baseline**

In `rerum/engine.py`, inside `minimize`, find the CURRENT baseline seed:

```python
        # Track best found
        best_expr = expr
        best_cost = cost_fn(expr)
        original_cost = best_cost
        count = 0
```

Replace with (seed from the canonical form so the baseline shares the
enumerated reps' normalization state):

```python
        # Track best found. Seed from the CANONICAL form so the baseline lives
        # in the same normalization state as the reps equivalents() yields;
        # otherwise normalization alone would read as a spurious improvement.
        # With no theory _canonicalize is the identity (unchanged behavior).
        best_expr = self._canonicalize(expr)
        best_cost = cost_fn(best_expr)
        original_cost = best_cost
        count = 0
```

Next, find the CURRENT improvement / identity check:

```python
        derivation = None
        if _expr_to_tuple(best_expr) != _expr_to_tuple(expr):
```

Replace with (compare against the canonical input):

```python
        derivation = None
        if _expr_to_tuple(best_expr) != _expr_to_tuple(self._canonicalize(expr)):
```

(The `self.equivalents(...)` enumeration and the `self.prove_equal(...)`
derivation call below already route through the seam, so they need no change.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest rerum/tests/test_theory_reasoning.py::TestMinimizeModuloTheory -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Run the existing optimization suite as a regression guard**

Run: `pytest rerum/tests/test_optimization.py -q`
Expected: PASS (no theory set in those tests).

- [ ] **Step 6: Commit**

```bash
git add rerum/engine.py rerum/tests/test_theory_reasoning.py
git commit -m "feat(f1): minimize seeds canonical baseline under a theory

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Soundness boundary, value-level backward-compat, and full-suite gate

**Files:**
- Test: `rerum/tests/test_theory_reasoning.py`
- Verify: whole repository test suite

- [ ] **Step 1: Write the soundness-boundary and backward-compat tests**

Append to `rerum/tests/test_theory_reasoning.py`:

```python
class TestSoundnessBoundary:
    def test_position_pinning_rule_is_not_reached_under_ac_theory(self):
        # DOCUMENTED INCOMPLETENESS (spec "Soundness boundary"): a distribute
        # rule pins the (+ ...) factor as the FIRST operand of *. Under an AC *
        # theory, canonical_sort moves it, so the syntactic matcher never sees
        # the arrangement and the distributed form is NOT reached. This is
        # intended behavior until F3 (AC-matching). Pinned here so a future
        # reader does not mistake it for completeness.
        eng = RuleEngine.from_dsl(
            "@distrib: (* (+ ?a ?b) ?c) => (+ (* :a :c) (* :b :c))"
        )
        eng.with_theory(Theory.from_dict({"*": {"ac": True}}))
        target = ["+", ["*", "a", "c"], ["*", "b", "c"]]
        proof = eng.prove_equal(["*", ["+", "a", "b"], "c"], target,
                                include_unidirectional=True)
        assert proof is None  # boundary: F3 is expected to make this succeed

    def test_position_pinning_rule_DOES_fire_without_theory(self):
        # Control: the same rule reaches the distributed form with no theory.
        eng = RuleEngine.from_dsl(
            "@distrib: (* (+ ?a ?b) ?c) => (+ (* :a :c) (* :b :c))"
        )
        target = ["+", ["*", "a", "c"], ["*", "b", "c"]]
        proof = eng.prove_equal(["*", ["+", "a", "b"], "c"], target,
                                include_unidirectional=True)
        assert proof is not None


class TestBackwardCompatValueLevel:
    def test_equivalents_identical_with_no_theory(self):
        eng = RuleEngine.from_dsl("@comm: (+ ?x ?y) <=> (+ :y :x)")
        out = list(eng.equivalents(["+", "a", "b"], max_depth=3))
        # Exact value AND order, byte-for-byte with pre-F1 behavior.
        assert out == [["+", "a", "b"], ["+", "b", "a"]]

    def test_prove_equal_path_identical_with_no_theory(self):
        eng = RuleEngine.from_dsl("@comm: (+ ?x ?y) <=> (+ :y :x)")
        proof = eng.prove_equal(["+", "a", "b"], ["+", "b", "a"],
                                include_unidirectional=True, trace=True)
        assert proof is not None
        assert proof.common == ["+", "b", "a"]
```

- [ ] **Step 2: Run the new tests to verify they pass**

Run: `pytest rerum/tests/test_theory_reasoning.py::TestSoundnessBoundary rerum/tests/test_theory_reasoning.py::TestBackwardCompatValueLevel -v`
Expected: PASS (4 passed). If `test_prove_equal_path_identical_with_no_theory` fails on `proof.common`, inspect whether the no-theory `common` changed (it must not).

- [ ] **Step 3: Run the FULL test suite as the backward-compat gate**

Run: `pytest -q`
Expected: PASS with NO NEW failures versus baseline (the pre-F1 suite count). Theory-less tests are unaffected by the seam. If any previously-passing test now fails, STOP and fix before continuing -- the no-theory path must be byte-for-byte unchanged.

- [ ] **Step 4: Run the no-domain guard explicitly**

Run: `pytest rerum/tests/test_mcp_no_domain.py -q`
Expected: PASS. F1 added no domain operator literal to `rerum/` source code (the operator strings live only in tests, which the guard ignores).

- [ ] **Step 5: ASCII check on the new test file (commit hook guard)**

Run: `LC_ALL=C grep -n '[^[:print:][:space:]]' rerum/tests/test_theory_reasoning.py && echo "FOUND non-ASCII" || echo "clean"`
Expected: `clean`.

- [ ] **Step 6: Commit**

```bash
git add rerum/tests/test_theory_reasoning.py
git commit -m "test(f1): pin soundness boundary + value-level backward-compat

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Post-Implementation

After all six tasks pass, F1 is complete: the five reasoning methods reason modulo a loaded theory; the no-theory path is unchanged; the soundness boundary is documented and test-pinned. Follow-ups (tracked in the spec, NOT built here): F3 AC-matching (closes the boundary), normalized rewriting in `simplify`, theory-awareness for the stochastic methods, and `normalize` memoization.

Use the superpowers:finishing-a-development-branch skill to complete the work (verify tests, present merge/PR options). Note: this work is on `main` directly; if there is no feature branch, the merge-locally option does not apply -- present the push decision to the user instead (per the standing constraint, pushing requires explicit user authorization).
