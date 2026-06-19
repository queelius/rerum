# F5: Knuth-Bendix completion (basic) -- design

**Date:** 2026-06-19
**Roadmap:** fifth sub-project of the TRS-frontier roadmap
(`docs/superpowers/specs/2026-06-18-trs-frontier-roadmap.md`), feature F5 -- the
capstone.
**Status:** design approved; ready for adversarial verification, then the
implementation plan.

## Goal

Turn a set of EQUATIONS into a CONFLUENT + TERMINATING rewrite system -- a
decision procedure for the equational theory. F5 runs the basic Knuth-Bendix
completion loop: orient each equation into a rule (F4), compute critical pairs
(F2), normalize both sides with the current rules, and add any un-joined pair
as a new oriented rule, until every critical pair joins. It is pure
ORCHESTRATION of the pieces already built: it writes almost no new math.

When it succeeds, the output is self-validating: every critical pair joined (so
the system is locally confluent) and every rule was admitted only via `orient`
(so `l >_lpo r`, hence terminating); by Newman's Lemma it is confluent. F2+F4's
own `check_confluence(engine, precedence).confluent` confirms it -- the capstone
is verified by the features beneath it.

## Scope

IN scope: BASIC completion -- the orient-and-add fixpoint loop with critical-pair
normalization. Each derived equation is normalized to normal form before
orienting (a light compose effect), keeping added rules tidy.

OUT of scope (honest follow-ups):
- INTER-REDUCTION (the COLLAPSE rule: when a new rule reduces an existing rule's
  LHS, demote that rule back to an equation; full COMPOSE of every rule's RHS).
  This is what lets completion converge on the group axioms and produce the
  minimal canonical system. Tracked as F5b. Basic completion is SOUND (any
  "complete" output is genuinely confluent + terminating) but may DIVERGE or
  bloat on inputs that need inter-reduction.
- Unfailing / ordered completion (handles un-orientable equations like
  commutativity via ordered rewriting) -- needs F3 (AC) territory.
- Automatic precedence search (the user supplies the precedence).
- Non-first-order equations (typed/sequence patterns, conditional rules):
  refused, reusing F2's `is_analyzable` at the `engine.complete` boundary; the
  pure `complete` assumes first-order plain-variable term inputs.

## The completion loop

`complete(equations, precedence, *, max_iterations=100, max_steps=1000) ->
CompletionResult`. `equations` is a `List[Tuple[ExprType, ExprType]]` of
`(l, r)` TERM pairs (both in `["?", name]` variable representation -- the same
representation `critical_pairs` produces and `orient` consumes). `max_iterations`
bounds the outer fixpoint passes; `max_steps` is the per-normalization reduction
budget threaded into every `eng.simplify` call. Both are keyword-only with the
stated defaults.

Internal representation: a RULE is a `(l, r)` term pair with `l >_lpo r`. All
duplicate and membership tests below use STRUCTURAL equality (`==`/`in`) over
the nested-list term representation (terms are unhashable lists; `_dedup` is
O(n^2) list-membership by design, n is small). Two small helpers bridge to the
engine, which speaks skeletons:
- `_term_to_skeleton(term)`: map each `["?", name]` to `[":", name]` (the
  forward of `instantiate_skeleton(.., {})`), recursing compounds, leaving
  atoms. A rule `(l, r)` becomes the engine pair `[l, _term_to_skeleton(r)]`
  and the `DirectedRule(pattern=l, skeleton=_term_to_skeleton(r))` that
  `critical_pairs` consumes. (`critical_pairs` gensym-renames rule variables
  apart, so a `CriticalPair`'s `.left`/`.right` are `["?", name]` terms with
  FRESH names -- harmless, since both legs share the renamed name and `orient`/
  `from_rules` are name-agnostic.)
- `_dedup(rules)`: drop structurally-duplicate `(l, r)` pairs (order-preserving).

The join test is SYNTACTIC (`s == t` on the two `simplify` results) and is
sound precisely because the internal `from_rules` engines carry NO theory
(`_canonicalize` is the identity), so `s == t` coincides with F2's
`_canonicalize`-based join test. A future modulo-theory (AC) extension MUST
switch to `_canonicalize`-based comparison, as F2 already does, to stay sound.

Algorithm:

1. ORIENT THE INPUT. `rules = []`. FIRST filter out every input `(l, r)` with
   `l == r` (a trivial equation -- `orient` returns `None` on structurally-equal
   terms, so this filter must precede `orient` to avoid a spurious "failed").
   Then for each remaining `(l, r)`: `d = orient(l, r, precedence)`; if
   `d is None` return `CompletionResult(status="failed", failed_equation=(l, r),
   rules=list(rules), iterations=0)` (`rules` is the partial set accumulated so
   far, empty at this point). Else append `(l, r)` if `d == "lr"` else `(r, l)`.
   Then `rules = _dedup(rules)`.

2. FIXPOINT. `passes = 0`. For `iteration` in `range(max_iterations)`:
   `passes = iteration + 1`.
   a. Build `eng = RuleEngine.from_rules([[l, _term_to_skeleton(r)] for (l, r) in rules])`.
   b. `records = [DirectedRule(name=str(i), pattern=l, skeleton=_term_to_skeleton(r), condition=None) for i, (l, r) in enumerate(rules)]`;
      `pairs, _na = critical_pairs(records)`.
   c. `new_rules = []`. For each `cp` in `pairs`:
      - `s = eng.simplify(cp.left, max_steps=max_steps)`,
        `t = eng.simplify(cp.right, max_steps=max_steps)`
        (catch `RecursionError`: on it, set `s, t = cp.left, cp.right` so the
        pair is treated as NOT joining -- conservative, see below).
      - If `s == t`: the pair joins, continue.
      - `d = orient(s, t, precedence)`; if `d is None` return
        `status="failed", failed_equation=(s, t), rules=list(rules),
        iterations=passes`.
      - `new = (s, t) if d == "lr" else (t, s)`. If `new` is not already in
        `rules` and not in `new_rules`, append it to `new_rules`.
   d. If `new_rules` is empty: return `status="complete"`, `rules=list(rules)`,
      `iterations=passes`. (Every critical pair joined.)
   e. `rules = _dedup(rules + new_rules)`.

3. If the loop exhausts `max_iterations` without converging: return
   `status="max_iterations"`, `rules=list(rules)` (the partial set),
   `iterations=max_iterations`.

`iterations` is thus the COUNT of fixpoint passes performed (1 on a first-pass
convergence, `max_iterations` when the budget is exhausted). The RecursionError/
budget-exhaustion handling is conservative: a normalization that does not finish
makes the pair NOT join (a new equation), so the worst case is divergence
reported as `max_iterations`, never a false `complete`.

## CompletionResult

A frozen dataclass. `status` and `rules` are always set; `failed_equation` and
`iterations` carry defaults so every return path can construct it without
under-specifying:
- `status: str` -- one of `"complete"`, `"failed"`, `"max_iterations"`.
- `rules: List[Tuple[ExprType, ExprType]]` -- the oriented `(l, r)` rules (the
  completed system when `status == "complete"`; the partial set accumulated so
  far for `"failed"`/`"max_iterations"`).
- `failed_equation: Optional[Tuple[ExprType, ExprType]] = None` -- the
  un-orientable pair (set only when `status == "failed"`).
- `iterations: int = 0` -- the count of fixpoint passes performed.
- `to_engine() -> RuleEngine` -- a method building a fresh `RuleEngine` from
  `rules` via `RuleEngine.from_rules([[l, _term_to_skeleton(r)] for ...])`. The
  ergonomic way to USE the completed system.

## Soundness and self-validation

- SOUNDNESS: a `status == "complete"` result is genuinely confluent +
  terminating. Local confluence: the loop only returns "complete" when a full
  pass over the critical pairs adds no new rule, i.e. every critical pair's two
  normal forms are equal -- the pairs all join. Termination: every rule was
  admitted only via `orient` returning a direction, i.e. `l >_lpo r` under the
  LPO, which (F4) proves termination. By Newman's Lemma the system is
  confluent.
- SELF-VALIDATION (an invariant, checked by the acceptance tests, NOT asserted
  inside `complete`): for a `"complete"` result,
  `check_confluence(result.to_engine(), precedence=precedence, max_steps=...).
  confluent is True`. The self-check MUST pass the SAME `max_steps` the
  completion used, so its joinability oracle is identical -- otherwise a CP that
  `complete` joined within budget could read `unknown` under a smaller budget
  and the invariant would be only budget-relative. (Empirically confirmed: the
  pinned test-2 example completes to 3 rules and `check_confluence` reports
  `confluent is True`, `terminating is True`.) This ties F5 back to F2+F4.
- HONESTY: `complete` NEVER returns a false `"complete"`. The only non-terminal
  risk (basic completion diverging) surfaces as `"max_iterations"`, and an
  un-orientable equation as `"failed"` -- never a wrong verdict.

## API

- `complete(equations, precedence, *, max_iterations=100, max_steps=1000) -> CompletionResult`
  in `rerum/completion.py`.
- `engine.complete(precedence, *, max_iterations=100, max_steps=1000) -> CompletionResult`
  -- extracts equations from `self`'s ANALYZABLE rules (each yields
  `(pattern, instantiate_skeleton(skeleton, {}))`; non-analyzable rules are
  skipped) and calls `complete`.
- Re-export `complete`, `CompletionResult` from the `rerum` core.

## Reuse and dependency direction

- REUSE: `orient` (F4 `termination`); `critical_pairs`, `DirectedRule`,
  `is_analyzable`, `instantiate_skeleton` (F2 `confluence`);
  `RuleEngine.from_rules` / `simplify` (engine). No reduction, matching,
  ordering, or critical-pair logic is reimplemented.
- IMPORTS: `completion.py` TOP-imports `RuleEngine` from `.engine`,
  `critical_pairs`/`DirectedRule`/`is_analyzable`/`instantiate_skeleton` from
  `.confluence`, and `orient` from `.termination`. None of `engine`,
  `confluence`, or `termination` imports `completion` at module level, so there
  is no load-time cycle. (`engine.complete` lazy-imports `complete` from
  `.completion` inside the method, matching the established idiom.)
- The confluence self-check used by tests imports `check_confluence` from
  `.confluence`; that is test-only, not a completion-runtime dependency.

## Design boundaries preserved

- GENERAL ENGINE: the precedence and equations are DATA. `completion.py`
  hardcodes no operator. The same loop completes a boolean or an arithmetic
  equation set. Tests prove both.
- READ-ONLY w.r.t. the caller's engine: `engine.complete` reads `self`'s rules
  but mutates nothing; it builds FRESH engines internally for normalization.
- FIRST-ORDER only (reuses F2's refusal at the `engine.complete` boundary).

## Testing plan

New file `rerum/tests/test_completion.py`. Equations are passed as `(l, r)`
TERM pairs in `["?", name]` representation (NOT raw DSL strings): tests build
them directly as nested lists (e.g. `["*", ["*", ["?","x"], ["?","y"]],
["?","z"]]`), so they line up with what `orient`/`critical_pairs` consume.

1. ORIENT-ONLY convergence: associativity `(* (* ?x ?y) ?z) = (* ?x (* ?y ?z))`
   completes (`status == "complete"`) to the single right-associativity rule
   (already confluent + terminating once oriented; its critical pairs join).
   Assert `len(result.rules) == 1`, `result.iterations == 1`, and the rule is
   oriented to the right (`result.rules[0] == (l, r)` for the right-nested `r`).
2. ADD-A-RULE convergence (PINNED, empirically verified): equations
   `[(["f",["g",["?","x"]]], "a"), (["g",["g",["?","x"]]], ["?","x"])]` with
   `precedence = ["f", "g", "a"]` complete in 2 passes to 3 rules, ADDING the
   derived rule `(["f", ["?", v]], "a")` (the un-joined critical pair
   `(a, (f x))` orients to `(f x) -> a`). Assert `status == "complete"`,
   `len(result.rules) == 3`, `result.iterations == 2`.
3. SELF-VALIDATION (PINNED): for the test-2 result,
   `check_confluence(result.to_engine(), precedence=["f","g","a"],
   max_steps=<same as complete>).confluent is True` and `.terminating is True`
   (empirically confirmed; the same max_steps must be used).
4. FAILED on un-orientable INPUT: a commutativity equation
   `(["+", ["?","x"], ["?","y"]], ["+", ["?","y"], ["?","x"]])` with
   `precedence = ["+"]` returns `status == "failed"` with `failed_equation` set
   and `result.rules == []` (no reduction order orients commutativity).
5. (Folded into test 4.) A standalone un-orientable DERIVED-pair example is hard
   to construct deterministically under basic completion without inter-reduction;
   the un-orientable-equation path is covered by test 4 (`orient is None` is the
   same code path whether the equation is an input or a derived pair).
6. MAX_ITERATIONS (PINNED, deterministic): the test-2 equation set with
   `max_iterations=1` returns `status == "max_iterations"` (it needs 2 passes;
   capping the budget below the requirement forces the path deterministically)
   and the call RETURNS, not hangs. `result.iterations == 1`.
7. `to_engine()`: the test-2 completed rules load into a `RuleEngine` that
   reduces a known instance to its normal form (e.g. `simplify((f (g a)))` and
   `simplify((g (g a)))` reduce to `a`).
8. GENERAL: the same `complete` handles a boolean equation set and an arithmetic
   one (build both as term pairs under their own precedences); behavioral
   generality (no operator literal hardcoded in `completion.py`).
9. `engine.complete(precedence)`: an engine whose rules are the test-2 equations
   (loaded as `<=>` or `=>` DSL rules) completes to a result whose `rules` match
   calling `complete` on the extracted equations directly; a non-analyzable
   rule in the engine (`?...`) is skipped from the extracted equations.
10. `_term_to_skeleton` round-trip: `_term_to_skeleton(["?","x"]) == [":","x"]`;
    on a compound term it is the inverse of `instantiate_skeleton(.., {})`
    (`instantiate_skeleton(_term_to_skeleton(t), {}) == t` for a `["?",name]`
    term -- pin the bridge the loop depends on).
11. `_dedup`: drops structural duplicates preserving first-occurrence order,
    e.g. `_dedup([(["a"],"b"), (["c"],"d"), (["a"],"b")]) == [(["a"],"b"),
    (["c"],"d")]` (including two equal-by-value but distinct-by-identity pairs).
12. WIRING: re-exports `complete`/`CompletionResult` in `rerum.__all__`; an
    import-smoke check (`import rerum.completion` succeeds; importing
    `rerum.completion` and `rerum.engine` in either order has no cycle). Full
    suite green.

## Non-goals (tracked, not built here)

- F5b inter-reduction (COMPOSE/COLLAPSE) -- convergence on group axioms and
  minimal canonical systems.
- Unfailing/ordered completion and AC-completion (need F3).
- Precedence synthesis.
