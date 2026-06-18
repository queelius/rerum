# F1: Theory-normalized core rewriting (reasoning layer) -- design

**Date:** 2026-06-18
**Roadmap:** first sub-project of the TRS-frontier roadmap
(`docs/superpowers/specs/2026-06-18-trs-frontier-roadmap.md`), feature F1, the
keystone.
**Status:** design approved; adversarially verified (3 critics:
code-grounding, TRS-soundness, internal-consistency); findings folded in.
Ready for the implementation plan.

## Goal

Make RERUM's equational-reasoning layer reason MODULO an equational theory.
With an associative-commutative (AC) theory loaded, `prove_equal(x+y, y+x)`
holds with no search, equivalence classes collapse AC-variants, and
`minimize` optimizes over the AC-quotiented class. No new algorithm: the
already-built `normalize.py` (flatten -> sort -> collect -> fold modulo a
`Theory`) is threaded into the reasoning methods as the function that decides
expression IDENTITY.

This is the keystone the solver-layer prune exposed: `normalize.normalize()`
is currently called from EXACTLY ONE place, `solve.py`'s `maybe_normalize`
(the demoted, non-core search layer). The core reasoning methods
(`equivalents`, `prove_equal`, ...) never normalize, so the richest piece of
genuine TRS machinery in the library is reachable only through the layer that
does not belong in the core. F1 lifts it into the core reasoning surface.

## Scope

IN scope -- the five equational-reasoning methods on `RuleEngine`:

- `equivalents` (lazy generator)
- `enumerate_equivalents` (collects `equivalents`)
- `prove_equal` (bidirectional BFS)
- `are_equal` (boolean wrapper over `prove_equal`)
- `minimize` (cost search over the class)

OUT of scope -- stated so the plan does not drift:

- `simplify` and the strategy drivers (`_simplify_exhaustive`,
  `_simplify_bottomup`, `_simplify_topdown`, the cached `rewriter()`
  fast-path). They stay theory-blind in F1. Normalized rewriting is a
  separate, trickier change whose payoff is only completed by AC-matching.
- The STOCHASTIC equivalence methods `random_equivalent`, `sample_equivalents`,
  and `random_walk`. They walk the same syntactic rewrite graph and would be
  silently theory-blind after F1; making them theory-aware is deferred so the
  "which methods reason modulo theory" boundary is TOTAL and explicit, not
  accidental. Tracked as a follow-up.
- AC-MATCHING of patterns (roadmap F3). F1 changes the equivalence RELATION
  the reasoning search quotients by; it does NOT make a syntactic pattern like
  `(+ ?x (- ?x))` unify against an AC arrangement. See "Soundness boundary".
- Per-call memoization of `normalize` (a performance note, see Performance).

## Semantics: the equivalence class is the set of canonical representatives

Decision (approved): when a theory is set, the class members the reasoning
methods expose are the NORMALIZED forms. One coherent story -- "modulo a
theory, we reason over canonical forms":

- `equivalents` yields one canonical representative per AC-class, starting from
  `normalize(expr, theory)`.
- `prove_equal`'s reported `common` is the canonical meeting form.
- `minimize` costs and returns the min-cost canonical representative.

When NO theory is set (`self._theory is None`, the default), behavior is
byte-for-byte unchanged. This is the backward-compatibility contract and the
primary regression guard (see Testing, items 8-9).

## Soundness boundary (what F1 delivers completely vs. what needs F3)

This section exists because the design canonicalizes every frontier neighbor
while RERUM's matcher (`rewriter.match`) is purely SYNTACTIC and position-
sensitive (it walks operands left-to-right by index, with no AC awareness).
The adversarial TRS-soundness review confirmed empirically that this creates a
real boundary. Be precise about it.

COMPLETE under F1 alone (KEYING-ONLY behaviors -- decided by
`_expr_to_tuple(_canonicalize(e))` identity, no pattern matching involved, so
they are independent of F3):

- Deciding equality/membership of AC-variants: `prove_equal(x+y, y+x)` is an
  instant zero-step proof via the existing `key_a == key_b` quick check;
  `are_equal` likewise.
- Class DEDUP: `equivalents`/`enumerate_equivalents` collapse AC-variants to
  one canonical representative per class.
- Associativity collapse and unit/annihilator collapse (e.g. `(+ x 0) == x`,
  `(* x 0) == 0`) -- whatever `normalize` already canonicalizes.

INCOMPLETE under F1 (MATCHING-DEPENDENT behaviors -- a derivation that must
FIRE a rule whose LHS pins an AC operator's argument by POSITION):

- A non-AC rule such as distribute `(* (+ ?a ?b) ?c) => (+ (* ?a ?c) (* ?b ?c))`
  requires the `(+ ...)` factor to be the first operand of `*`. Under an AC
  `*` theory, `canonical_sort` reorders `(* (+ a b) c)` to `(* c (+ a b))`
  (compounds sort after symbols), so the syntactic matcher never sees the
  arrangement the rule needs. Crucially, an explicit `*`-commute `<=>` rule
  does NOT rescue this: F1 re-canonicalizes the rebuilt neighbor straight back
  before it can become a frontier state, so the redex is never explored. The
  no-theory search (with explicit commute + distribute rules) DOES reach the
  distributed form; F1 with a theory can lose it.

Therefore: F1 is sound and complete for AC-variant EQUALITY, class DEDUP, and
unit collapse (the keystone). For DERIVATIONS that must fire a position-pinning
rule over an AC operator, F1 is INCOMPLETE; that completeness requires
AC-matching (roadmap F3). The boundary is narrow -- it bites only when a user
(1) loads a theory marking an operator AC, (2) keeps non-AC rules whose LHS
fixes that operator's argument position, AND (3) relies on those rules firing
inside `equivalents`/`prove_equal`/`minimize`. The no-theory path is
untouched. F1 must NOT present canonicalize-the-frontier as lossless; this
boundary is documented behavior and is pinned by an expected-incompleteness
test (Testing, item 10).

## Architecture: a single canonicalization seam (approach A)

The entire feature is "what is the identity of an expression?" Today identity
is its syntactic tuple (`_expr_to_tuple`). F1 makes identity the tuple of its
NORMAL FORM, but only at the seam where the reasoning layer decides "have I
seen this / are these the same / which class is this." Rewriting STEPS are
untouched; we change the equivalence relation the search quotients by.

Add one private helper to `RuleEngine`:

```python
def _canonicalize(self, expr: ExprType) -> ExprType:
    """Canonical form under the engine's theory, or expr unchanged.

    Identity function when no theory is set (the backward-compat path).
    When a theory is set, returns normalize(expr, theory) -- idempotent,
    confluent, and the single definition of expression identity for the
    reasoning layer.
    """
    if self._theory is None:
        return expr
    from .normalize import normalize as _normalize
    return _normalize(expr, self._theory)
```

The existing `_expr_to_tuple` stays the hashing primitive; reasoning-layer
keys become `_expr_to_tuple(self._canonicalize(e))`. Three things route
through the seam: the START node(s), each generated NEIGHBOR before it is
keyed/stored/queued/yielded, and the quick-equality CHECK in `prove_equal`.

Approach A is chosen over (B) normalizing inside `_all_single_rewrites` -- the
neighbor generator has OTHER callers outside F1's scope (`random_equivalent`,
`random_walk`), so B would change their behavior and still leave start-nodes
and quick-checks needing separate handling: wider blast radius for no gain --
and over (C) a normalizing wrapper engine, which is over-engineered and unlike
the codebase.

## Per-method changes

### `equivalents`

- Seed `visited` with `_expr_to_tuple(self._canonicalize(expr))` and yield
  `self._canonicalize(expr)` as the first element (was: raw `expr`). The start
  canonical form is yielded EXACTLY ONCE: it is pre-seeded into `visited`, so a
  later neighbor that canonicalizes back to the start key is never re-yielded.
- For each neighbor from `_all_single_rewrites`, canonicalize it; use the
  canonical form for the `visited`-membership test, for `yield`, AND as the
  value appended to the frontier (so neighbor generation explores from
  canonical forms).
- Net effect: one canonical representative per AC-class, deduped by canonical
  key. With no theory, `_canonicalize` is identity and the method is
  unchanged.

### `enumerate_equivalents`

No direct change. It is `list(self.equivalents(...))` and inherits the new
semantics.

### `prove_equal`

- Canonicalize `expr_a`/`expr_b` for `key_a`/`key_b`. AC-equal inputs then
  satisfy the EXISTING `key_a == key_b` quick check and return a zero-step
  proof immediately (this is how `prove_equal(x+y, y+x)` becomes instant).
- Seed `visited_a`/`visited_b` and `frontier_a`/`frontier_b` with the
  canonical forms. For each neighbor: canonicalize it, and STORE THE CANONICAL
  form (not the raw neighbor) in `visited_*[new_key]` and on the frontier. This
  is required so `reconstruct_path`, which builds each step's `.after` from the
  STORED visited value, emits canonical `.after` values and the
  "step.after == state" path contract continues to hold.
- The two frontiers meet on a shared canonical key. F1 emits NO separate
  `kind="normalize"` micro-steps inside the proof path; normalization is the
  keying, not a path step. The initial canonicalization of the inputs is
  reflected by the start node being the canonical form.

### `are_equal`

No direct change. It is `self.prove_equal(...) is not None` and inherits.

### `minimize`

Mostly inherits via `self.equivalents(...)`, but it seeds its baseline from
the RAW input, so it needs the seam in two spots to keep the baseline in the
same normalization state as the enumerated reps:

- Seed `best_expr = self._canonicalize(expr)`, then
  `best_cost = cost_fn(best_expr)` and `original_cost = best_cost` (preserving
  the current single-evaluation structure where `original_cost` aliases
  `best_cost` rather than calling `cost_fn` twice).
- The "did we improve?" identity check (currently
  `_expr_to_tuple(best_expr) != _expr_to_tuple(expr)`) compares canonical
  keys: `_expr_to_tuple(best_expr) != _expr_to_tuple(self._canonicalize(expr))`.
- `include_unidirectional` asymmetry (a known footgun): `minimize` defaults
  `include_unidirectional=True` while `equivalents`/`prove_equal` default
  `False`. `minimize` already THREADS its flag through to BOTH its
  `equivalents` enumeration AND its `prove_equal` derivation call, so the
  derivation traverses the same `=>` rules the enumeration used; the derivation
  remains best-effort (it is already allowed to be `None`, handled at the
  existing guard). Under a theory the meeting test in that `prove_equal` call
  is modulo-AC like any other. A test covers `minimize` over a mixed
  `=>`/`<=>` rule set under a theory (Testing, item 11).
- Reported improvement: with the canonical seed, `original_cost` (and hence
  `improvement_ratio`/`cost_ratio`, 0.5.0 semantics) is measured against the
  CANONICAL form of the input, NOT the raw input. The cost reduction from
  normalization itself is intentionally not counted as "improvement".

## Units, theory preconditions, and edge cases

- Identity/annihilator folding fires ONLY for operators marked `ac: True` in
  the theory (`_fold_constants` gates units inside `if theory.is_ac(head)`).
  An operator carrying `identity`/`annihilator` but `ac: False` (or omitting
  `ac`) is left untouched. The collapse examples below assume `ac: True`.
- Units: a `+` theory with `ac: True, identity: 0` puts `(+ x 0)` and `x` in
  the same class; a `*` theory with `ac: True, annihilator: 0` puts `(* x 0)`
  and `0` together. This falls out of `normalize`; F1 adds nothing.
- Empty theory `Theory({})`: `normalize` is the identity, so it is
  BEHAVIORALLY indistinguishable from no theory -- but `_canonicalize` branches
  on `self._theory is None`, not on emptiness, so a set-but-empty theory still
  takes the theory-loaded code path and pays `normalize`'s cost. Prefer setting
  NO theory (not an empty one) to retain the no-theory fast path.
- `normalize` idempotence/confluence is a PRECONDITION F1 relies on for the
  canonical key to be well-defined; F1 does NOT validate the theory. A
  non-confluent or self-contradictory `Theory` (e.g. one whose declarations
  make `normalize` non-idempotent) is out of scope: behavior is undefined and
  is `normalize.py`'s responsibility, not F1's.
- Cancellation: `equivalents`/`prove_equal` already honor
  `self._cancel_requested`; canonicalization adds no new cancellation points.

## Performance

A theory recomputes `normalize` for every neighbor and every quick check.
`normalize` is roughly O(size log size) (a sort dominates). Theory use is
OPT-IN, so the cost is paid only when a theory is loaded (including a
set-but-empty theory, per the edge case above). v1 does NOT memoize; a
per-call `{raw_key -> canonical}` cache is a clean future optimization but is
YAGNI now and is noted, not built. No change to the no-theory path's
performance.

## Design boundaries preserved

- GENERAL ENGINE: the theory is DATA (a `Theory` built from a dict/JSON). F1
  hardcodes no operator; the same seam canonicalizes an arithmetic `+`
  theory, a boolean `and`/`or` theory, or any AC theory. The no-domain swap
  test (`test_mcp_no_domain.py`) is unaffected (F1 touches `engine.py`, not
  `rerum/mcp/`).
- RULES ARE DATA, PRELUDES ARE CODE: unchanged. F1 touches neither rule
  loading nor the prelude boundary.
- PURE CORE, MUTABLE ENGINE: `normalize.py` stays pure; the seam reads
  `self._theory` (engine state) and calls the pure `normalize`. No engine
  state is allocated in `normalize.py`.

## Testing plan

New file `rerum/tests/test_theory_reasoning.py` (one-file-per-feature
convention). Fixtures:

- An AC `+` theory `Theory({"+": {"ac": True, "identity": 0}})` and a `*`
  theory with `annihilator: 0` for the unit tests.
- A FIXED, fully specified rule set for the determinism-sensitive tests:
  for the `+` theory fixture, NO commutativity rule is loaded, so the
  no-theory `prove_equal(x+y, y+x)` returns `None` within the default budget
  (its equality holds ONLY via the theory). State this in the test so item 1
  is deterministic.
- A second, non-arithmetic boolean `and`/`or` AC theory fixture to prove the
  machinery is domain-free.

Tests:

1. `prove_equal(["+","x","y"], ["+","y","x"])` returns a proof WITH the `+`
   theory (instant zero-step via the quick check) and returns `None` WITHOUT
   it on the no-commute fixture (determinate, not "None/longer").
2. Associativity + commutativity: `(+ (+ a b) c)` and `(+ a (+ c b))` prove
   equal under the theory.
3. `equivalents`/`enumerate_equivalents` over an AC sum yield strictly FEWER
   members with the theory than the syntactic class (no AC duplicates); every
   yielded form is canonical (`normalize(e) == e`); and the yielded sequence
   has no duplicate canonical keys (`len(set(keys)) == count`). (This is a
   DEDUP guarantee, not a reachability guarantee -- see item 10.)
4. `are_equal` returns True for AC-variants under the theory, False without
   (no-commute fixture).
5. Units: with `ac:True, identity:0`, `(+ x 0)` and `x` are `are_equal` and
   share a class; with `*` `ac:True, annihilator:0`, `(* x 0)` and `0` are
   `are_equal`.
6. `minimize` returns a canonical representative; with a theory whose only
   effect on the input is normalization, `minimize` does NOT spuriously report
   "improved", and `improvement_ratio` is measured against the CANONICAL
   baseline (assert the concrete ratio, e.g. 0.0 when the canonical input is
   already minimal).
7. Domain-free (BEHAVIORAL): the boolean `and`/`or` theory fixture proves
   `and`/`or` AC-equalities through the SAME `_canonicalize`/reasoning code
   path. (Static no-domain-in-`engine.py` is already guarded by
   `test_mcp_no_domain.py`'s pattern for the MCP layer; this test asserts
   behavior, not a whole-file grep.)
8. Backward-compat, VALUE + ORDER level: with `self._theory is None`,
   `list(engine.equivalents(e))` equals the pre-F1 list element-for-element
   and in the same order for a fixed seeded rule set; same for the
   `prove_equal` proof path and the `minimize` result. (With no theory
   `_canonicalize` returns its argument unchanged, so frontier order is
   provably identical.)
9. Full suite stays green: the existing test suite shows NO NEW failures vs.
   baseline (the theory-less tests are unaffected). (Informational anchor: the
   baseline is the current passing count; the assertion is "no new failures",
   not an exact integer, to avoid count drift.)
10. EXPECTED-INCOMPLETENESS (pins the soundness boundary as intended
    behavior): with the AC `*` theory and a distribute-shaped rule whose LHS
    pins position `(* (+ ?a ?b) ?c) => ...`, assert that the F1 reasoning
    method does NOT reach the distributed form, documenting the boundary.
    Cross-reference the future F3 spec, where the same case is expected to
    succeed.
11. `minimize` over a MIXED `=>`/`<=>` rule set under a theory: the result is
    a canonical rep and the (best-effort) derivation either reconstructs
    end-to-end or is cleanly `None` (never a malformed trace).
12. Proof-path STRUCTURE under a theory: for a non-trivial `prove_equal` proof,
    every step's `.after` equals the corresponding canonical state, no step has
    `kind == "normalize"`, and the path reconstructs end-to-end (guards the
    "no micro-steps / contract preserved" claim).

## Out-of-scope follow-ups (tracked, not built here)

- F3 AC-matching makes `simplify` rules AND position-pinning reasoning rules
  fire on AC arrangements (closes the soundness boundary above).
- Normalized rewriting in `simplify` (canonical output + fast-path bypass +
  trace-visible normalize steps) -- revisit after F3.
- Theory-awareness for the stochastic methods (`random_equivalent`,
  `sample_equivalents`, `random_walk`).
- `normalize` memoization for hot reasoning loops.
