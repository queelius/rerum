# F3: AC-Matching Proper -- Design

> TRS-frontier roadmap feature F3. Matching modulo associativity and
> commutativity (multiset partition matching). Closes the F1 soundness boundary
> pinned in `rerum/tests/test_theory_reasoning.py`
> (`test_position_pinning_rule_is_not_reached_under_ac_theory`).

**Status:** design approved (brainstorming). Next: implementation plan.

**Date:** 2026-06-21

---

## Goal

Let a rule whose left-hand side is an AC pattern fire on any AC arrangement of
the subject without the author pre-sorting. Today a rule like

```
@distrib: (* (+ ?a ?b) ?c) => (+ (* :a :c) (* :b :c))
```

does NOT fire under an AC `*` theory: F1 canonicalizes the subject, the `(+ ...)`
factor moves out of the first operand position, and the SYNTACTIC matcher never
sees the arrangement the LHS pins. F3 makes the matcher itself work modulo AC so
the rule fires regardless of operand order, and a single rule such as
`(+ ?x (- ?x) ?rest...)` cancels a term wherever it sits in a sum.

## Scope (decided)

| Decision | Choice |
|----------|--------|
| Matching vs unification | **Matching only.** The pattern has variables; the subject is a concrete (ground-for-matching) term. AC-UNIFICATION (both sides have variables) is NOT in scope. |
| Consumers | **Full rewriting surface:** `simplify`/`apply_once` AND the equational methods (`equivalents`, `prove_equal`, `minimize`). |
| Blow-up bound | **Work budget + fail-safe.** A cap on assignment-tree steps; on hit, stop enumerating and set a `truncated` signal. Matches already yielded stay valid. |
| `?rest...` binding | **List + existing splice.** `?rest...` binds the leftover as a LIST (unchanged from today's `?...`); spliced with `(+ :rest...)`. The only new thing under AC: the matcher CHOOSES which elements are leftover (multiset selection) instead of taking the positional tail. |
| Structure | **New pure module `rerum/acmatch.py`** + one engine seam. `match()` is untouched. |

### Why matching-only is enough for this feature

AC-matching makes `simplify` and the equational methods AC-aware, because in
those operations the SUBJECT is a concrete term. It does NOT make F2
(critical-pair computation) or F5 (Knuth-Bendix completion) AC-aware: those
overlap two PATTERNS and need AC-UNIFICATION (Stickel's algorithm, NP-hard),
which is a separate, larger feature. Completion's inability to orient
commutativity therefore remains a documented gap after F3, to be addressed by a
future AC-unification increment. This boundary is intentional and keeps F3
bounded and shippable.

## Architecture

A new pure module `rerum/acmatch.py`, parallel to `confluence.py`,
`termination.py`, and `completion.py`. It exports one primary function:

```python
def ac_match(pat, exp, theory, bindings=None, budget=None) -> Iterator[Bindings]:
    """Yield every Bindings extension under which `pat` matches `exp` modulo
    the AC operators declared in `theory`. Multi-valued and lazy."""
```

Properties:

- **Pure.** No engine state. Takes the `Theory` carrier (which already exposes
  `is_ac(op)`, `identity(op)`), reuses `rewriter.py`'s element predicates
  (`arbitrary_constant`, `arbitrary_variable`, `arbitrary_free`,
  `arbitrary_rest`, `variable_name`, `constant`, `variable`, `free_in`) and
  `normalize.py`'s `flatten` and `ORDER_KEY`.
- **`match()` is untouched.** The single-valued syntactic matcher in
  `rewriter.py` and the cached no-theory `rewriter()` fast path do not change.
  The common no-theory case is a strict no-op: zero behavioral or performance
  change.
- **General-engine principle.** `acmatch.py` names NO domain operator. It keys
  entirely on `theory.is_ac(op)` -- AC-ness arrives as DATA in the `Theory`.
  Enforced by `test_mcp_no_domain.py`-style discipline (no operator literal as
  code).
- **Budget holder.** A small mutable carrier
  `MatchBudget(steps: int, truncated: bool = False)` threaded through.
  `ac_match` decrements `steps` per assignment attempt; at zero it sets
  `truncated = True` and stops yielding. Soundness is unaffected -- truncation
  only bounds completeness.

## The matcher algorithm

`ac_match(pat, exp, theory, bindings, budget)` dispatches on the pattern node
and yields zero or more consistent `Bindings`:

1. **Atom / literal pattern** -> yield `bindings` once iff `exp` is an atom
   equal to `pat`. (0 or 1 result.)

2. **Single-variable pattern** (`?`, `?c`, `?v`, `?free`) -> reuse the existing
   element logic: bind the variable to the whole `exp` when the type constraint
   holds and is consistent with any prior binding. Non-linearity: a re-bound
   variable must equal its prior value MODULO the theory's canonical form (so
   `?x` bound to `(+ a b)` still matches `(+ b a)`). `?free` is validated inline
   at yield time, when the bindings for that branch are fully known. (0 or 1.)

3. **Compound `[op, p1..pm]` with AC `op` and `exp` headed by `op`** -- the
   multiset case:
   1. `flatten` both sides under `op`, giving the expression multiset `E` and
      the explicit sub-patterns `p1..pm` plus an optional trailing `?rest`.
   2. Backtracking assignment: for `p1`, try each element `e` in `E` (iterated
      in `ORDER_KEY` canonical order); recursively `ac_match(p1, e, ...)`
      threading the bindings forward; then recurse on `p2..pm` over `E \ e`.
      Each candidate attempt decrements the budget.
   3. Leftover: with `?rest` present, bind it to the LIST of unchosen elements
      (canonical order); without `?rest`, require the chosen elements to exhaust
      `E` (`m == len(E)`).
   4. Yield each complete consistent assignment.

4. **Compound with non-AC `op`, or head mismatch, or `exp` not compound** ->
   positional matching in lockstep (the existing `match_compound` shape) but
   multi-valued, because a positional child may itself be an AC node. `?rest` in
   a non-AC context keeps its current positional-tail-as-list meaning.

### Soundness and determinism

- **Soundness.** Every yielded `Bindings` is a real match: bindings only extend
  through consistent element matches. The budget bounds enumeration only; it
  never fabricates a match.
- **Determinism.** `E` is enumerated in `ORDER_KEY` canonical order, so
  `simplify`'s first-yield choice is reproducible across runs.
- **Pruning.** Bindings thread THROUGH the backtracking, so once `p1` binds
  `?x=a`, a later `p2 = (- ?x)` can only match the element `(- a)` -- the branch
  collapses immediately. Non-linear patterns (the cancellation idiom) are cheap,
  not combinatorial.

## Integration seam

The engine funnels all rule-LHS matching through two internal points, so F3 adds
ONE helper that both consume:

```python
def _match_lhs(self, pattern, subterm) -> Iterator[Bindings]:
    # no theory, or no AC op in theory -> yield match(pattern, subterm) (0 or 1)
    # AC theory present                -> yield from ac_match(pattern, subterm,
    #                                                 self._theory, budget=...)
```

Two consumers, unchanged in spirit:

- **`_simplify_once` (engine.py ~2499)** -- the recursive single-pass rewriter
  behind `simplify`/`apply_once`. It takes the FIRST binding `_match_lhs`
  yields (rule fires once, leftmost-outermost, as today). `simplify`'s fast-path
  gate (engine.py ~2490) gains one condition: an AC theory present joins "hooks
  registered" as a reason to bypass the cached `rewriter()` and use the
  `_simplify_once` slow path. No-theory `simplify` stays byte-identical.

- **`_all_single_rewrites` (engine.py ~3593)** -- the per-step enumerator behind
  every equational method (`equivalents`, `prove_equal`, `minimize`, and their
  labeled variants). It ITERATES all of `_match_lhs`'s yields, so one AC rule at
  one subterm now produces several distinct rewrites. This is what closes the
  pinned boundary: `prove_equal` reaches the distributed form by finding the AC
  arrangement.

No new call sites. The pinned test
`test_position_pinning_rule_is_not_reached_under_ac_theory` flips from
`assert proof is None` to a real proof.

## Edge cases and error handling

- **Empty / singleton leftover.** `(+ ?x ?rest...)` on `(+ a)` gives
  `?rest = []`, and `(+ :rest...)` instantiates to `(+)`. AC rewrite results are
  run through `flatten` + identity cleanup (reusing `normalize.py`), so `(+)`
  collapses to the theory identity (e.g. `0`) and `(+ x)` collapses to `x`. No
  new logic.
- **Literals / typed sub-patterns in AC nodes.** `(+ 2 ?x ?rest...)` matches a
  `2` element by equality; `?c`/`?v`/`?free` element-patterns reuse the existing
  predicates.
- **Nested / non-AC children.** `(+ (* ?x ?y) ?rest)` recurses (`*` AC ->
  multiset; `(f ?x)` non-AC -> structural). Handled by `ac_match` recursion.
- **Budget exhaustion.** The generator stops and sets `truncated = True`. For
  `simplify`, if the cap is hit before any match is found, the rule simply does
  not fire (sound, conservative). For `prove_equal`/`equivalents`, the
  result/report carries the `truncated` signal so the caller knows enumeration
  was bounded. The default budget is configurable on the methods that already
  take work budgets.
- **`RecursionError`** on pathological depth -> caught at the engine seam and
  treated as no-match (conservative), consistent with `confluence` and
  `completion`.

## Non-goals (documented boundaries)

- **AC-unification.** F2 critical-pair computation and F5 completion stay
  syntactic. Completion still cannot orient commutativity; that gap awaits a
  future AC-unification feature.
- **ACU (matching modulo identity).** Pure AC only -- `(+ ?x ?y)` will NOT match
  a bare `a` by treating it as `(+ a 0)`. The common unit cases are already
  covered incidentally, because F1's `normalize` strips units from the subject
  before the equational methods match; full ACU is a future increment.
- **Changing `match()` or the no-theory fast path.** Out of scope by design.
- **Performance work beyond the budget.** Correctness first; the work budget is
  the only blow-up control. An optional `experiments/` probe may validate it.

## Invariants preserved

- General-engine principle: `acmatch.py` names no operator; AC-ness is data.
- No-theory path is a strict no-op (matcher and fast path unchanged).
- Pure-core / mutable-engine boundary: `acmatch.py` is pure; the engine holds
  the seam and the budget default.
- Soundness: every yielded match is real; the budget bounds only completeness.

## Testing strategy

- **New `rerum/tests/test_acmatch.py`** (pure matcher, the bulk):
  - Enumeration correctness: `(+ ?x ?y)` on `(+ a b)` yields exactly 2;
    `(+ ?x ?y ?z)` on `(+ a b c)` yields 6; the exhaust rule (no `?rest`
    requires `m == len(E)`).
  - Non-linearity / cancellation: `(+ ?x (- ?x) ?rest...)` on `(+ a (- a) b)`
    binds `?x=a, ?rest=[b]`; yields nothing when no cancelling pair exists.
  - `?rest` capture: leftover is a canonical-order list; empty and singleton
    cases.
  - Literals, `?c`/`?v`/`?free` in AC nodes, nested AC, non-AC children.
  - Budget truncation: a wide sum hits the cap, `truncated = True`, yet every
    yielded match is valid.
- **Property tests:** every yielded binding, re-applied to the pattern,
  reproduces a term AC-equal to the subject (soundness); for small inputs, the
  yield count matches a brute-force permutation oracle (completeness within
  budget).
- **Flip the pinned boundary** in `test_theory_reasoning.py`
  (`assert proof is None` -> a real proof); keep the no-theory control green.
- **Engine integration:** `simplify` fires an AC cancellation rule wherever the
  term sits; `prove_equal` proves AC-distributivity; `equivalents`/`minimize`
  under AC.
- **A light `examples/` demo:** a small AC rule snippet showing one rule firing
  across arrangements, kept minimal (likely riding the existing boolean/sets
  theories rather than a new domain).
- **Guards:** full suite stays green (no-theory strict no-op -> all current
  tests hold), `test_mcp_no_domain.py` passes (`acmatch.py` names no operator),
  ASCII clean. Optional `experiments/` probe: the budget caps a wide-sum
  blow-up.

## Dependencies

- F1 (theory-normalized reasoning): `Theory` carrier, `flatten`, `ORDER_KEY`,
  the `_canonicalize` seam and the equational methods that consume AC matches.
- Existing matcher element predicates in `rewriter.py` (reused, not changed).

## Out-of-family note

F3 is squarely CORE term rewriting: it strengthens the `match` primitive itself.
It is not search, numeric evaluation, an agent interface, or data generation, so
it belongs in the core, alongside F1/F2/F4/F5.
