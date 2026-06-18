# RERUM TRS-frontier roadmap

**Status:** active roadmap. Supersedes the A/B/C "future-features roadmap"
(2026-06-18), which the solver-layer prune (commits b9e9901/7f3a0bf) reframed:
its themes -- A (MCP tooling), B (corpus/synthdata), C (search frontier) -- are
now in the OPTIONAL, non-core layers, not the rewriting core. This roadmap
re-aims at genuine term-rewriting-system frontier: the features that make
RERUM a stronger *rewrite library*, not a stronger agent/search platform.
**Date:** 2026-06-18
**Nature:** a DECOMPOSITION + SEQUENCING index, not a single implementation
spec. Each feature gets its own brainstorm -> spec -> plan -> implement cycle.
This doc orders the work and records dependencies.

## What "the TRS family" means here

A term rewriting system is: terms, rewrite rules, matching, strategies that
apply rules toward a normal form, and reasoning ABOUT the resulting relation
(confluence, termination, equational consequence). Reduction is the core; the
theory of WHEN reduction is well-behaved is the frontier. Best-first SEARCH
over the rewrite graph (the demoted `rerum.solve`) and numeric MODEL
INTERPRETATION (the demoted `rerum.numeval`) are things you build ON a
rewriter; they are not rewriting. This roadmap stays inside the family.

## Grounding: what the core already has, and the one thing it is missing

RERUM's core already covers a lot of the family:

- Terms (nested-list `ExprType`), rules (DSL + JSON), syntactic matching with
  binding patterns (`?x`, `?x:const`, `?x:var`, `?x:free(v)`, `?x...`),
  skeleton instantiation with compute (`(! op ...)`).
- Conditional rewriting (guards / `when`), strategies (exhaustive/fixpoint,
  bottom-up, top-down, once), groups, priorities, bidirectional `<=>` rules.
- Equational REASONING -- the Knuth-Bendix-adjacent surface WITHOUT completion:
  `equivalents` (lazy), `enumerate_equivalents`, `prove_equal` (bidirectional
  BFS), `are_equal`, cost-directed `minimize`.
- Derivations/traces, `RewriteTrace.inverse()`.
- Normalization modulo an equational `Theory` (`normalize.py`): flatten ->
  sort -> collect -> fold for AC operators, units, and repeat rules, all
  carried as DATA. Idempotent and confluent.

The frontier check is blunt: there is NO confluence checking, NO termination
ordering, NO critical-pair analysis, NO completion, and NO narrowing anywhere
in `rerum/`. And one structural gap dominates the rest:

> **The AC-normalization machinery is stranded in the search layer.**
> `normalize.normalize()` is called from EXACTLY ONE place -- `solve.py`'s
> `maybe_normalize`. `match()` takes no theory argument (matching is purely
> syntactic), and `simplify`/`equivalents`/`prove_equal`/`minimize` never
> normalize. So in the core, `prove_equal(["+","x","y"], ["+","y","x"])` does
> NOT hold by associativity-commutativity; it only would inside the now-demoted
> `solve`. The single richest piece of genuine TRS machinery we have is
> reachable only through the layer we just pushed out of the core.

That stranding is the keystone: fixing it (F1) is already-built code that just
needs threading, and it unblocks everything downstream.

## The features

### F1. Theory-normalized core rewriting (KEYSTONE)

- Scope: thread the engine's `_theory` through the CORE rewrite/reasoning loop
  so reduction and equational reasoning work MODULO the theory. Concretely:
  `simplify` normalizes between (or after) steps; `equivalents`/`prove_equal`/
  `are_equal`/`minimize` compare canonical forms under the theory; the
  normalized state shows up faithfully in traces. No new algorithm -- it lifts
  `normalize.py` out of `solve` and into the core.
- Output: `prove_equal(x+y, y+x)` holds by AC in the core with no search;
  equivalence classes shrink to true AC classes; `minimize` prices canonical
  forms. The library finally rewrites modulo its own theories.
- Dependencies: `normalize.py` (done), `_theory` slot + `with_theory` (done).
- Effort: low-moderate. Threading + canonical-comparison wiring + tests; the
  risk is performance (normalize on every comparison) and trace fidelity.
- Carries: whether AC-matching (F3) is even needed for the common cases, or
  whether normalize-then-syntactic-match covers most of them.

### F2. Confluence + critical-pair diagnostics (read-only)

- Scope: given a rule set, compute critical pairs (overlaps between rule LHSs)
  and report local confluence (are the pairs joinable by the engine?). A pure
  ANALYSIS pass -- it changes no rewrite behavior, it tells the user whether
  their rules are confluent and, if not, exactly which overlap breaks it.
- Output: a diagnostic ("rules R3 and R7 form a non-joinable critical pair on
  `(* (+ ...) ...)`") -- the single most useful thing you can tell someone
  authoring a rule set. Gates completion (F5).
- Dependencies: matching (done); benefits from F1's modulo-theory comparison.
- Effort: moderate. Overlap unification + joinability check + reporting + tests.

### F3. AC-matching proper

- Scope: matching MODULO AC, not just canonicalize-then-syntactic-match. A
  pattern `(+ ?x ?y ?rest...)` against `(+ a b c)` should enumerate the
  multiset partitions (with `?rest...` capturing the remainder). This is the
  combinatorial heart of rewriting modulo AC and is what F1's normalize pass
  cannot give you (canonical order != pattern unification).
- Output: rules whose LHS is an AC pattern fire on any AC arrangement of the
  subject without the author pre-sorting -- e.g. a single `(+ ?x (- ?x))` rule
  cancels a term wherever it sits in a sum.
- Dependencies: F1 (so the rest of the loop is already modulo-theory); match
  internals.
- Effort: high. Backtracking multiset matcher + `?rest...` interaction + a
  bound on the combinatorial blow-up + tests.

### F4. Termination via a reduction order

- Scope: a reduction order the engine can check -- start with a weight/
  polynomial order, optionally a lexicographic path order (LPO). Lets the
  engine ORIENT an equation (which side is "simpler") and certify that a rule
  set cannot loop.
- Output: `simplify` can warn on a non-terminating rule set instead of relying
  on the cycle-detection backstop; and completion (F5) gets the orientation
  oracle it requires.
- Dependencies: terms (done). Independent of F1/F2/F3 but pairs with F5.
- Effort: moderate-high. The order + precedence/weights config (as DATA, no
  hardcoded operators) + tests.

### F5. Knuth-Bendix completion (CAPSTONE)

- Scope: the algorithm RERUM's equational-reasoning surface gestures at but
  does not implement: given equations + a reduction order (F4), orient them,
  compute critical pairs (F2), add joining rules, iterate to a confluent +
  terminating system (or report failure/divergence). Turns "apply these rules"
  into "DERIVE a decision procedure for this equational theory."
- Output: feed RERUM a set of `<=>` equations and get back a confluent
  terminating rule set whose normal forms decide equality -- the single most
  powerful thing a rewrite library can do.
- Dependencies: F2 (critical pairs) + F4 (orientation). Strongly informed by
  F1.
- Effort: high. The completion loop + divergence handling + tests on classic
  systems (group axioms -> the standard 10-rule completion).

### F6. Narrowing (the in-family replacement for `solve`)

- Scope: run rewriting "backwards" with unification -- narrowing -- to solve
  goals `find sigma such that sigma(t)` reduces to a target. This is the
  PRINCIPLED, in-the-TRS-family way to do the goal solving that the demoted
  best-first `solve` did ad hoc. Narrowing is sound and complete for confluent
  terminating systems; best-first search is neither.
- Output: equation solving / goal solving expressed as rewriting, not as an
  external search layer -- a candidate to eventually retire `rerum.solve`.
- Dependencies: unification (new), F1 (modulo-theory), benefits from F4.
- Effort: high. Unification + the narrowing strategy + termination control +
  tests.

## Dependency map

```
F1 (theory-normalized core) --underlies--> F2, F3, F5, F6
F2 (critical pairs)        --+
                             +--> F5 (Knuth-Bendix completion)
F4 (reduction order)       --+
F3 (AC-matching)            -- strengthens F2/F5; independent of F4
F6 (narrowing)             -- needs F1 + unification; in-family replacement
                              for the demoted solve search
```

Keystone: **F1 is cheap, fixes a real correctness gap the prune exposed, and
underlies everything else.** It is the obvious first move.

## Sequencing: three waves

### Wave 1 -- make the core rewrite modulo its theories

- **F1. Theory-normalized core rewriting** (keystone; low-moderate).
- **F2. Confluence + critical-pair diagnostics** (read-only; moderate). High
  user value on its own and gates the F5 capstone.

### Wave 2 -- matching modulo AC + an orientation oracle

- **F3. AC-matching proper** (high). Informed by F1: build it only for the
  cases F1's normalize pass cannot cover.
- **F4. Termination via a reduction order** (moderate-high). Independent;
  schedule alongside F3.

### Wave 3 -- the capstones

- **F5. Knuth-Bendix completion** (high; needs F2 + F4). The headline
  capability.
- **F6. Narrowing** (high; needs F1 + unification). The principled, in-family
  replacement for the demoted `solve` search.

## Non-goals (explicitly out, post-prune)

- Re-promoting `rerum.solve` / `rerum.numeval` to the core. They stay optional.
- Anything domain-specific in `rerum/` (calculus, boolean, sets remain
  example rule sets; the no-domain swap test still guards this).
- MCP-surface, corpus-generation, and synthdata work -- valuable, but they
  live in the optional layers and are tracked separately, not here.

## Execution model

- Each feature gets its own `brainstorm -> spec -> plan -> implement` cycle.
  This doc is the index; we work it wave by wave.
- First sub-project: **F1** -- cheapest, fixes a real gap, underlies the rest.
- The general-engine principle holds throughout: theories, precedences, and
  weights are DATA; `rerum/` hardcodes no operator.
