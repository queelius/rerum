# AC-Unification -- Design

> The keystone primitive: unification modulo associativity and commutativity.
> Generalizes F3 AC-matching (one-sided, ground subject) to the two-sided case
> (both terms have variables). Unlocks, as FUTURE follow-ons, AC-narrowing (F6
> modulo AC) and AC-completion (F5 modulo AC, closing the commutativity gap).

**Status:** design approved (brainstorming). Next: implementation plan.

**Date:** 2026-06-24

---

## Goal

Add `ac_unify(t1, t2, theory)`: enumerate the unifiers of two terms modulo the
associative-commutative operators declared in a `Theory`. Where F2's `unify` is
syntactic and single-valued (one most-general unifier or none), AC-unification
is MULTI-VALUED: there is no single mgu, so the result is a COMPLETE SET of
unifiers. This is the one primitive that, once built, makes AC-matching (F3),
critical-pair computation (F5), and narrowing (F6) all able to work modulo AC.

## Scope (decided)

| Decision | Choice |
|----------|--------|
| AC vs ACU | **Pure AC.** Variables take >= 1 occurrence; the Theory's declared identity/unit is NOT used by the unifier (the engine's `normalize` strips units upstream). ACU (unit absorption) is a clean follow-on. |
| Generality | **Stickel-general.** Nested terms, free function symbols, and multiple AC operators are all handled (the last two by recursion). NOT elementary-only (variables + constants). |
| Output | **Complete (not necessarily minimal)** set of unifiers, as a LAZY generator, bounded by a budget. Subsumption-minimal sets are a follow-on. |
| Consumers | **Primitive only.** Ship `ac_unify` standalone + a verification demo. Wiring into F5 critical pairs (AC-completion) and F6 narrowing (AC-narrowing) are SEPARATE follow-on features -- each carries its own truncation-honesty/soundness work. |
| Placement | **New core module `rerum/acunify.py`**, re-exported. F2's `unify` is untouched. |

### Why a complete set, not one mgu

Syntactic unification has the most-general-unifier property: one answer subsumes
all others. AC-unification does NOT. The canonical example `x+y =? u+v` (four
distinct variables) has a SEVEN-element minimal complete set, none subsuming the
rest. Every consumer (critical pairs, narrowing) needs the SET: a single
AC-unifier would silently drop critical pairs and make `check_confluence` lie.
The generator shape, the budget, and the "truncated means incomplete" contract
all follow from "no mgu exists".

## Architecture

A new core module `rerum/acunify.py`, mirroring `acmatch.py` / `narrowing.py`.
Public surface:

```python
@dataclass
class UnifyBudget:
    """Fail-safe budget for the basis + subset enumeration. On exhaustion,
    ``truncated`` is set and enumeration stops; the yielded set is then SOUND but
    INCOMPLETE (every yield is a real unifier; some may be missing)."""
    steps: Optional[int] = None
    truncated: bool = False

def ac_unify(t1, t2, theory, *, bindings=None,
             budget=None) -> Iterator[Subst]:
    """Yield each AC-unifier of ``t1`` and ``t2`` modulo the AC operators in
    ``theory``. Multi-valued and lazy. Pure AC (units NOT used). Returns a
    COMPLETE set within the budget, not necessarily minimal."""
```

Properties:

- **Reuses F2 wholesale**: `Subst` (a `{name: term}` dict), `apply_subst`,
  `_occurs`, `_compose_bind`, `_is_var`, `UnsupportedPattern` from
  `rerum/confluence.py`; plus `flatten` from `rerum/normalize.py` and `gensym`
  from `rerum/rewriter.py`. F2's `unify` is NOT changed.
- **Operates on pattern terms**: variables are `["?", name]`, exactly the form
  F2's substitution machinery consumes.
- **General-engine principle**: `acunify.py` names NO domain operator. It keys on
  `theory.is_ac(op)`; fresh variables are `gensym`'d. Domains arrive as the
  `Theory` data and the rules' terms.
- **First-order only**: a `?c`/`?v`/`?free`/`?...`/skeleton node raises
  `UnsupportedPattern` (reuse F2's refuse-first guard), consistent with F2/F4/
  F5/F6.

### Dispatch

`ac_unify` dispatches on the pair of terms (analogous to `ac_match`):

- **Either side a variable** -> bind it to the other side (occurs-check via
  `_occurs`); a single result.
- **Both atoms** -> yield the running bindings iff they are equal.
- **Compound, head NOT AC (or heads differ / arity differs)** -> positional
  unification, MULTI-VALUED (a child may be an AC node): a backtracking product
  over the argument pairs (the two-sided analog of `_match_positional`).
- **Compound, both headed by the same AC `op`** -> the Stickel core (below).

## The Stickel AC-node algorithm

For `f(s...) =? f(t...)` with `f` AC, after applying the running bindings:

1. **Flatten + cancel.** Flatten both sides under `f` into argument multisets
   `S`, `T`. Remove their multiset intersection (syntactically-equal common args
   cancel -- sound, they unify trivially).

2. **Abstract into atoms.** Each remaining argument is a VARIABLE (`["?", n]`)
   or a NON-VARIABLE term (constant or compound). Collect LHS atoms `u_1..u_M`
   with multiplicities `a_i` and RHS atoms `v_1..v_N` with multiplicities `b_j`.
   Each distinct non-variable term is abstracted as a constant atom (reconciled
   in step 4).

3. **Solve the linear Diophantine equation.** Compute the Hilbert basis (the
   finite set of minimal non-negative integer solutions) of
   `a_1*m_1 + ... + a_M*m_M = b_1*n_1 + ... + b_N*n_N`, all `m_i, n_j >= 0`.
   Each basis vector becomes a fresh variable `z_b = gensym(...)`.

4. **Enumerate admissible covering subsets.** For each subset of basis vectors
   that COVERS every atom (each atom's summed multiplicity >= 1):
   - VARIABLE atom `u_i`: bind it to the AC-combination (under `f`) of the
     `z_b`'s in which it participates, with their multiplicities -- a single
     term, or `(f z_b ...)`.
   - CONSTANT / COMPOUND atom: the subset forces its fresh variable to BE that
     term; two non-variable atoms (one LHS, one RHS) coupled by a shared basis
     vector are RECURSIVELY `ac_unify`'d (this is what handles nested terms /
     free symbols -- Stickel-general). A coupling whose terms do not unify
     discards that subset.
   - Compose the resulting bindings (occurs-check via `_occurs`); if consistent,
     YIELD it (composed with the incoming bindings).

5. **Budget.** The basis size and the `2^|basis|` subset space are the blow-up.
   `UnifyBudget` caps both; on cap, set `truncated = True` and stop -- the
   yielded set is then sound but incomplete.

The Hilbert-basis solver is the one genuinely new sub-component: a bounded
enumeration of minimal non-negative solutions of a single linear Diophantine
equation (Fortenbacher-style), budget-capped. Everything else is term plumbing
reused from F2. References: Stickel 1981; Baader and Nipkow, "Term Rewriting and
All That", Ch. 10.

## Soundness, budget, and edge cases

- **Soundness.** Every yielded `sigma` satisfies `sigma(t1) =_AC sigma(t2)`,
  independently checkable as
  `normalize(apply_subst(sigma, t1), theory) == normalize(apply_subst(sigma, t2), theory)`.
  The budget bounds COMPLETENESS only; it never fabricates a wrong unifier.
- **Truncation contract.** `truncated = True` means the set is incomplete. A
  consumer must treat a truncated result as UNKNOWN, never "complete". The flag
  is the contract; honoring it is the future consumer's responsibility.
- **Edge cases:**
  - Both multisets empty after cancellation -> yield the current bindings.
  - One side empty, other non-empty (pure AC has no unit) -> NO unifier.
  - A variable on both sides (`x+a =? x+b`) -> `x` cancels by multiset
    intersection -> `a =? b` -> fail. Correct.
  - Single remaining arg per side -> unify the two args directly (no Diophantine).
  - Nested non-variable args that are AC nodes -> handled by the step-4 recursion.
  - Occurs-check (`sigma(x)` containing `x`) -> reject via `_occurs`.
  - Fresh-variable hygiene: the gensym avoid set is seeded from the current
    bindings and the free variables of both terms, so a recursive AC node (an AC
    operator inside a coupled atom) never recaptures the outer node's fresh `z`
    names.
  - Non-variable atom multiplicity: a repeated non-variable atom (e.g. `a + a`)
    is kept as separate weight-1 columns in the Diophantine encoding, not merged
    into one column of multiplicity 2 (merging would be unsound).
  - No AC op, or node not AC -> never reaches Stickel (dispatch handles it).

## Testing and verification

- **New `rerum/tests/test_acunify.py`:**
  - Canonical complete-set counts (pure AC): `x+y =? u+v` -> 7; `x+y =? a+b` ->
    2; `x =? a+b` -> 1; `a+b =? a+b` -> 1; `a =? b` -> 0.
  - Soundness property (keystone): for every yield on a battery of AC problems,
    `normalize(sigma(t1)) == normalize(sigma(t2))` under the theory.
  - Completeness on small cases: the yielded set (normalized, deduped) equals a
    brute-force oracle for tiny problems.
  - The F3 special case: when `t2` is ground, `ac_unify(t1, t2)` restricted to
    `t1`'s variables COVERS `ac_match(t1, t2, theory)`.
  - Nested / free symbols: `(+ (f ?x) ?y) =? (+ a (f b))` -> `{x: b, y: a}`.
  - Budget truncation: a wide all-variable problem hits the cap ->
    `truncated = True`, every yield still sound.
  - Determinism: stable yield order.
- **A light `examples/` demo** (pure data) solving a small AC-unification problem
  through the general engine / primitive.
- **Re-exports** `ac_unify`, `UnifyBudget`.
- **Guards:** full suite green, `test_mcp_no_domain.py` passes (`acunify.py`
  names no operator), ASCII clean.

## Non-goals (documented boundaries)

- **ACU (modulo unit).** Pure AC only; unit absorption is a follow-on.
- **Minimal complete set.** v1 returns a complete set that may include redundant
  / subsumed unifiers; subsumption pruning is a follow-on.
- **Consumer wiring.** F5 AC critical pairs (AC-completion) and F6 AC narrowing
  are SEPARATE follow-on features; this feature ships only the primitive.
- **Combining multiple equational theories** (Baader-Schulz). One AC theory plus
  free function symbols (handled by recursion) only.
- **Non-first-order patterns** (`?c`/`?v`/`?free`/`?...`). Refused, like F2.

## Invariants preserved

- General-engine principle: `acunify.py` names no operator; the theory is data;
  fresh variables are `gensym`'d.
- F2's `unify` is untouched (no risk to F2/F4/F5/F6).
- Soundness: every yielded unifier is real (re-verifiable via `normalize`).
- Budget honesty: `truncated` distinguishes an incomplete set from a complete one.

## Dependencies

- `rerum/confluence.py` (F2): `Subst`, `apply_subst`, `_occurs`,
  `_compose_bind`, `_is_var`, `UnsupportedPattern`.
- `rerum/normalize.py` (F1): `flatten`, and `normalize` for the soundness tests.
- `rerum/rewriter.py`: `gensym`, term predicates.
- `rerum/acmatch.py` (F3): the `ac_match` cross-check (test only).

## Out-of-family note

AC-unification is core term rewriting: it strengthens the `unify` primitive that
critical-pair computation and narrowing are built on. It is the two-sided
counterpart of F3's AC-matching and the prerequisite for AC-completion and
AC-narrowing. Search, numeric evaluation, agents, and data generation are NOT
involved; it belongs in the core, alongside F1-F6.
