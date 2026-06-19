# F2: Confluence + critical-pair diagnostics -- design

**Date:** 2026-06-18
**Roadmap:** second sub-project of the TRS-frontier roadmap
(`docs/superpowers/specs/2026-06-18-trs-frontier-roadmap.md`), feature F2.
**Status:** design approved; adversarially verified (3 critics: TRS-algorithm
correctness, code-grounding, internal-consistency); findings folded in
(including a BLOCKING normal-form-oracle fix and the directed-rule accessor).
Ready for the implementation plan.

## Goal

Give a rule-set author a read-only diagnostic: "are these rules confluent, and
if not, exactly which overlap breaks it?" F2 computes the CRITICAL PAIRS of a
rule set (the overlaps between rule left-hand sides), checks each for
JOINABILITY (do both sides reduce to a common form?), and reports a local
confluence verdict plus the offending non-joinable pairs. It changes no rewrite
behavior; it is pure analysis OVER the rewrite relation.

This is the diagnostic that tells a user the single most useful thing about a
rule set, and it gates the eventual Knuth-Bendix completion capstone (F5):
completion is "repeatedly add rules to join the non-joinable critical pairs",
so F2 builds the machine that finds them.

## What F2 needs that does not yet exist

RERUM's `match` is ONE-SIDED: a pattern (variables on the left) against a
mostly-ground expression. A critical pair requires UNIFICATION of two rule
LHSs -- patterns with variables on BOTH sides, renamed apart, with an
occurs-check and a most-general unifier (mgu). No unification exists in the
codebase. F2's foundational new primitive is first-order syntactic unification
over RERUM's structured pattern terms.

The good news: patterns are already STRUCTURED terms. A plain `?x` is the node
`["?", "x"]` (the DSL parser in `expr.py` produces this, never a raw `"?x"`
string); a compound is `[op, arg, ...]`; typed/sequence forms are
`["?c", name]` / `["?v", name]` / `["?free", name, v]` / `["?...", name]`. So
first-order unification is a clean structural recursion that treats `["?", n]`
as a unification variable.

## Scope

Decision (approved): FIRST-ORDER unification only; conservatively REFUSE the
rest.

IN scope:
- Unification of plain variables `["?", name]` and compound terms
  `[op, args...]` (operator-literal head, same arity), with occurs-check.
- Critical-pair computation over the engine's directed reduction rules.
- Joinability checking via the engine's reduction (`simplify`), modulo the
  loaded theory (reusing F1's `_canonicalize`).
- A local-confluence verdict + a structured report.

OUT of scope (the unification REFUSES these by raising `UnsupportedPattern`, so
the affected rule is reported NOT ANALYZED -- never silently mis-analyzed):
- Typed variables `["?c", name]` (const) and `["?v", name]` (var).
- Constrained `["?free", name, v]`.
- Sequence/rest variables `["?...", name]` (these need AC/sequence
  unification -- effectively F3).
- Any skeleton-only form (`["!", ...]` compute, `["fresh", ...]`) appearing in
  a pattern position.
- CONDITIONAL rules (guards): a conditional rule's overlap is only a real
  critical pair when the guards are jointly satisfiable, which F2 does not
  reason about. Conditional rules are reported not-analyzed.
- Global-confluence CERTIFICATION (requires termination; see Verdict rigor).
- Knuth-Bendix completion itself (F5); termination ordering (F4); AC-matching
  (F3).

The first-order fragment covers the genuinely confluent example domains
(differentiation, boolean, sets, peano -- largely plain `?x` rules). The
soundness guarantee is the only one that matters: F2 NEVER reports a rule set
"locally confluent" on the basis of an overlap it could not actually analyze
or a joinability it could not actually decide; unanalyzable rules are surfaced
in `not_analyzed`, undecided pairs in `unknown`. All the verification-found
bugs were FALSE-NEGATIVE risks (mislabeling a good system) except one false-
positive risk (the `apply_once` normal-form oracle), which is fixed below.

## Architecture: a pure `confluence.py` module + thin engine methods

New module `rerum/confluence.py`, pure functions over the nested-list
`ExprType` (mirrors `normalize.py`: pure, imports from `rewriter`, never
allocates engine state). The engine exposes thin wrappers, matching the
pure-core / mutable-engine boundary.

### How F2 reuses F1 (the reason F2 follows F1 in Wave 1)

A critical pair `(s, t)` is JOINABLE iff `s` and `t` reduce to a common form.
F1 gave the engine modulo-theory equality through `_canonicalize`. So
joinability is decided by reducing both sides with `simplify` and comparing
through `_canonicalize` -- confluence-checking MODULO the loaded theory, with no
extra machinery. F2 consumes F1's output.

## Positions and term surgery

A POSITION `p` is a path from the root: a tuple of child indices, with `()` the
root. Two pure helpers in `confluence.py`:
- `subterm_at(term, p)` -- the subterm reached by following `p`.
- `replace_at(term, p, new)` -- a copy of `term` with the subterm at `p`
  replaced by `new`.
- `positions(term)` -- enumerate positions. NON-VARIABLE positions are those
  whose subterm is a compound `[op, ...]` or a constant atom -- i.e. NOT a
  `["?", ..]`/`["?c", ..]`/`["?v", ..]`/`["?free", ..]`/`["?...", ..]` node.
  Overlaps are computed only at non-variable positions.

## The primitives

### `unify(t1, t2) -> Subst | None` (raises `UnsupportedPattern`)

First-order syntactic unification over structured pattern terms.

- `Subst` is a mapping `{var_name: term}` (a small dict wrapper or plain dict),
  maintained as a FULLY-APPLIED (idempotent) substitution: when a new binding
  `x -> u` is added, `u` is first instantiated under the current `Subst` and
  the binding is composed so no bound variable appears in any range term.
  `apply_subst(subst, term)` is therefore a SINGLE structural pass.
- REFUSAL FIRST: before any other branch, if EITHER argument's head is an
  unsupported pattern marker (`?c`, `?v`, `?free`, `?...`, or a skeleton-only
  marker `!`/`fresh`), raise `UnsupportedPattern`. This ordering matters:
  without it, `unify(["?","x"], ["?c","y"])` would bind `x` to the typed node
  as an opaque term instead of refusing. (Defense-in-depth: `critical_pairs`
  also pre-scans whole rule patterns -- see below -- so `unify` normally never
  sees these, but the raise-first guard makes `unify` correct as a standalone
  primitive.)
- A VARIABLE is `["?", name]`. `unify(["?", x], term)`: if `term` is the same
  variable node, succeed with no new binding; else occurs-check (`x` must not
  occur in the fully-applied `term`), then add `x -> term`. Symmetric when the
  second argument is a variable.
- An ATOM (a string symbol that is not a pattern node, or a number) unifies
  with an equal atom; otherwise fails (`None`).
- A COMPOUND `[h, a1..an]` (operator-literal head) unifies with `[h', b1..bm]`
  iff `h == h'`, `n == m`, and the argument lists unify left-to-right with the
  substitution threaded and applied as it grows. A compound never unifies with
  an atom.
- Returns the mgu `Subst` on success; `None` on a normal failure (clash /
  occurs-check / arity / head mismatch).

`apply_subst(subst, term)` walks `term` replacing each `["?", name]` whose
`name` is bound; because `subst` is fully-applied, one pass suffices and it
recurses into compound arguments. `rename_apart(rule, avoid)` walks a rule's
pattern and skeleton, gensym-renaming each DISTINCT `["?", name]` variable to a
fresh name not in `avoid` (it enumerates `["?", name]` nodes specifically;
`free_symbols` is used only to seed `avoid`, since it returns a conservative
superset that also includes operator heads and the `"?"` marker token).

### `critical_pairs(rules) -> list[CriticalPair]` (+ a `not_analyzed` set)

`rules` is a list of DIRECTED-rule records, each a small struct
`(name, pattern, skeleton, condition)`. (`check_confluence` builds these from
`engine.rule_set()`; see below.)

For each ORDERED pair `(Ri, Rj)`:

1. PRE-SCAN: if `Ri` or `Rj` is conditional (`condition is not None`), or its
   `pattern` contains ANY unsupported pattern node anywhere (a full recursive
   descent), record its post-desugar `name` in `not_analyzed` and skip the
   pair. (Each directed form -- a `<=>` rule's `-fwd` and `-rev` -- is judged
   independently; names are unique post-desugar, so `not_analyzed` dedups by
   name.)
2. RENAME APART: produce `Rj'` by renaming all variables of `Rj` to fresh names
   avoiding `free_symbols(Ri.pattern) | free_symbols(Ri.skeleton)`. This keeps a
   shared `?x` in two rules (or the self-overlap, `i == j`) from being
   conflated.
3. For each NON-VARIABLE position `p` in `Ri.pattern`:
   - EXCLUDE the trivial overlap when `i == j AND p == ()` (a rule's root always
     overlaps a renamed copy of itself, yielding a trivially joinable pair).
     This exclusion is on `(i==j, p==root)` STRUCTURALLY -- not on the unifier
     being the identity (after renaming, the root self-overlap mgu is a
     RENAMING, never the identity).
   - `sigma = unify(subterm_at(Ri.pattern, p), Rj'.pattern)`.
     - `UnsupportedPattern`: record not-analyzed, continue. (Belt-and-braces;
       the pre-scan should have already excluded such rules.)
     - `None`: no overlap here, continue.
     - an mgu `sigma`: emit the critical pair (construction below).

CRITICAL-PAIR CONSTRUCTION (the standard superposition, verified against
Baader & Nipkow Def 6.2.1). Both legs descend from the OVERLAPPED TERM
`u = apply_subst(sigma, Ri.pattern)`:
- `left = apply_subst(sigma, Ri.skeleton)` -- rewrite `u` at the root with Ri.
- `right = apply_subst(sigma, replace_at(Ri.pattern, p, Rj'.skeleton))` --
  rewrite `u` at position `p` with Rj'. (Substitution commutes with
  replacement at a fixed position, so this equals `u` with the subterm at `p`
  replaced by `apply_subst(sigma, Rj'.skeleton)`.)
The pair is unordered for joinability purposes.

`CriticalPair` is a frozen dataclass:
`(left: ExprType, right: ExprType, rule_left: str, rule_right: str,
position: tuple, joinable: Optional[bool])` -- `joinable` is `None` until
`check_confluence` fills it.

### `check_confluence(engine, *, max_steps=1000) -> ConfluenceReport`

1. Collect the engine's ENABLED directed reduction rules via
   `engine.rule_set()`: its `RuleSet.__iter__` yields `(idx, [pattern,
   skeleton], metadata)` triples, POST-DESUGAR (a `<=>` rule appears as two
   directed entries `-fwd`/`-rev`, which is what `simplify` reduces with) and
   ALREADY filtered by `excluding_disabled` (so disabled groups contribute no
   overlaps). Build a `(name=metadata.name, pattern=rule[0], skeleton=rule[1],
   condition=metadata.condition)` record per triple.
2. `pairs, not_analyzed = critical_pairs(records)`.
3. For each pair `(s, t)`, decide JOINABILITY. Let
   `s2 = engine.simplify(s, max_steps=max_steps)`,
   `t2 = engine.simplify(t, max_steps=max_steps)`,
   `cs = engine._canonicalize(s2)`, `ct = engine._canonicalize(t2)`.
   - If `cs == ct`: `joinable = True` -- they reach a common form (modulo
     theory), regardless of normal-form status. (Equality is checked FIRST;
     this is sound and avoids a false negative when both converge to a shared
     non-normal form within budget.)
   - Else if BOTH `s2` and `t2` are normal forms: `joinable = False` -- the
     engine's reduction drives them to DISTINCT normal forms.
   - Else: `joinable = None` (UNKNOWN -- reduction hit the step budget or a
     cycle without normalizing; not decided, never a false verdict).

   NORMAL-FORM ORACLE (the BLOCKING fix): a term `u` is a normal form iff NO
   rule rewrites it ANYWHERE -- a RECURSIVE test. Do NOT use `engine.apply_once`
   (it is ROOT-ONLY and returns a `(result, metadata)` tuple, so it falsely
   declares a term normal when its only redex is in a subexpression). Use a
   recursive single-step idempotence test: `engine._simplify_once(u) == u`
   (the engine's recursive one-rule-anywhere pass), or equivalently the public
   `engine.simplify(u, max_steps=max_steps) == u`. The plan picks one; both are
   recursive and sound.

`ConfluenceReport` is a frozen dataclass:
- `locally_confluent: bool` -- True iff NO critical pair is `joinable is False`
  AND NO critical pair is `joinable is None`; equivalently `all(cp.joinable is
  True for cp in critical_pairs)`. An empty critical-pair set is vacuously True
  (this holds UNCONDITIONALLY -- the Critical Pair Lemma needs no termination
  assumption). An `unknown` pair therefore blocks a True verdict (we never
  claim local confluence with an undecided overlap).
- `critical_pairs: list[CriticalPair]` (all, with `joinable` filled).
- `non_joinable: list[CriticalPair]` (`joinable is False`).
- `unknown: list[CriticalPair]` (`joinable is None`).
- `not_analyzed: list[str]` (post-desugar rule names skipped: conditional or
  unsupported pattern), deduplicated by name.
- `analyzed_pair_count: int` (number of critical pairs with a definite True/
  False decision) -- an INFORMATIONAL field so a caller can distinguish
  "proven locally confluent over real overlaps" from "vacuously true, nothing
  analyzable". It does NOT enter the `locally_confluent` definition.

### Verdict rigor (the honest framing, like F1's soundness boundary)

The verdict is LOCAL confluence (all critical pairs joinable), NEVER bare
"confluent". Newman's Lemma: local confluence implies global confluence ONLY
for terminating systems, and F2 does not check termination (that is F4). The
`ConfluenceReport` docstring states this. Two further honesty notes carried in
the report docstring:
- A `joinable is False` is STRATEGY-RELATIVE: it means "not joinable by the
  engine's own reduction (`simplify`)", which is the right notion for a
  confluence DEFECT report but is not a claim of absolute non-joinability under
  every conceivable strategy.
- `unknown` pairs are reported separately and never counted as joinable.

## API

- `engine.critical_pairs() -> list[CriticalPair]` -- thin wrapper building
  records from `self.rule_set()` and calling `confluence.critical_pairs`.
- `engine.check_confluence(*, max_steps=1000) -> ConfluenceReport` -- thin
  wrapper over `confluence.check_confluence(self, ...)`.
- `rerum.confluence` importable directly: `unify`, `apply_subst`,
  `rename_apart`, `subterm_at`, `replace_at`, `positions`, `Subst`,
  `critical_pairs`, `check_confluence`, `CriticalPair`, `ConfluenceReport`,
  `UnsupportedPattern`.
- Re-export the public surface from the `rerum` core (confluence analysis is in
  the term-rewriting family, like `normalize`): add `check_confluence`,
  `critical_pairs`, `CriticalPair`, `ConfluenceReport`, `unify`,
  `UnsupportedPattern` to `rerum.__all__`.

## Design boundaries preserved

- GENERAL ENGINE: `confluence.py` hardcodes NO operator. It analyzes whatever
  rules the engine holds; operators are data. The no-domain swap test is
  unaffected (F2 adds a module + engine methods, touches no `rerum/mcp/`
  domain literal). Tests prove it with both an arithmetic and a boolean rule
  set.
- PURE MODULE: `confluence.py` is pure functions over `ExprType`; it imports
  from `rewriter` (predicates, `gensym`, `free_symbols`) and never allocates
  engine state. `check_confluence` READS the engine (`rule_set`, `simplify`,
  `_simplify_once`, `_canonicalize`, `_theory`) but mutates nothing.
- READ-ONLY: F2 changes no reduction behavior. `simplify`/`match`/the reasoning
  layer are untouched.

## Testing plan

New file `rerum/tests/test_confluence.py`:

1. `unify` units: two distinct variables unify (`["?","x"]`, `["?","y"]`);
   variable with a compound binds it; occurs-check fails
   (`unify(["?","x"], ["f", ["?","x"]])` is `None`); head clash and arity
   mismatch fail; equal atoms unify, distinct atoms fail; a compound does not
   unify with an atom.
2. `unify` REFUSAL: each of `["?c","x"]`, `["?v","x"]`, `["?free","x","y"]`,
   `["?...","r"]` raises `UnsupportedPattern`; AND the MIXED case
   `unify(["?","x"], ["?c","y"])` raises (not binds) -- pins the raise-first
   ordering.
3. `apply_subst` units: substitutes bound variables, leaves free ones,
   RECURSES into compound arguments, and is single-pass-correct on a
   fully-applied substitution (a variable bound to a term containing another
   bound variable resolves fully).
4. `rename_apart` / apartness: two rules that both use `?x` with overlapping
   LHSs produce a critical pair whose substitution treats the two `?x` as
   DISTINCT (no capture).
5. Self-overlap: a single rule whose LHS unifies with a NON-root subterm of its
   own LHS emits a (non-root) critical pair, while the trivial root overlap
   (`i==j, p==()`) is EXCLUDED.
6. Known-CONFLUENT small set: every critical pair joinable; `locally_confluent
   is True`; `analyzed_pair_count >= 1`.
7. Deliberately NON-confluent set: two rules overlap and reduce to DIFFERENT
   normal forms; `non_joinable` names the offending rules and the EXACT overlap
   position (assert the `position` tuple value); `locally_confluent is False`.
8. Vacuous case: a rule set with no overlaps -> no critical pairs ->
   `locally_confluent is True`, `analyzed_pair_count == 0`, `not_analyzed`
   empty.
9. NOT-ANALYZED: a rule whose LHS uses `?...` (and, separately, a conditional
   rule) is reported in `not_analyzed`; it does not make the verdict falsely
   confluent.
10. UNKNOWN-pair verdict: a set with exactly one critical pair that does not
    normalize within `max_steps` yields that pair in `unknown`, `joinable is
    None`, and `locally_confluent is False` (an undecided pair blocks True);
    the call RETURNS (does not hang).
11. Joinability MODULO theory: an overlap whose two results are equal only
    after AC-normalization is `joinable is True` WITH the theory and a single,
    DETERMINISTIC `joinable is False` WITHOUT it (choose an example whose two
    sides reduce to distinct normal forms without the theory).
12. GENERAL: the same `check_confluence` analyzes a boolean rule set and an
    arithmetic rule set; assert no operator literal is hardcoded in
    `confluence.py` (a source grep, mirroring the no-domain guard's spirit).
13a. `engine.critical_pairs()` / `engine.check_confluence()` delegate: equal to
    calling `confluence.*` on records built from the engine's rules.
13b. Disabled groups: a rule placed in a disabled group contributes NO critical
    pair / no overlap.
14. Full suite stays green (F2 adds a module + methods + re-exports; changes no
    existing behavior).

## Non-goals (tracked, not built here)

- F3 AC-matching / sequence unification (the `?...` and AC overlaps F2
  refuses).
- F4 termination ordering (needed to upgrade "locally confluent" to
  "confluent").
- F5 Knuth-Bendix completion (uses F2's critical pairs + F4's order).
- Typed/`?free` unification (the other refused fragments).
