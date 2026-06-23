# F6: Narrowing -- Design

> TRS-frontier roadmap feature F6 (final item). Unification-driven backward
> rewriting to solve goals: find a substitution sigma such that sigma(start)
> reduces to a target. The principled, in-family equation solver.

**Status:** design approved (brainstorming). Next: implementation plan.

**Date:** 2026-06-22

---

## Goal

Add narrowing: rewriting run "backwards" with unification instead of matching.
Where `simplify` evaluates a term toward a normal form, narrowing SOLVES for the
inputs -- given `start` (a term with variables) and `target`, it finds a
substitution `sigma` such that `sigma(start)` reduces to (a term unifying)
`sigma(target)`. Its signature application is E-UNIFICATION: solve an equation
`s =? t` modulo the rewrite rules, e.g. solve `append(?xs, [c]) = [a, b, c]` for
`?xs = [a, b]`. Narrowing is sound and complete for confluent terminating
systems; the demoted best-first `solve` layer is neither.

## Scope (decided)

| Decision | Choice |
|----------|--------|
| Placement | **Core** module `rerum/narrowing.py`, re-exported. Narrowing is an in-family TRS calculus (sound + complete for confluent terminating systems), unlike `solve`'s heuristic search. |
| Relationship to `solve.py` | **Untouched.** Narrowing is the principled equation solver; `solve` stays the optional cost-directed escalation. Retiring `solve` is a possible FUTURE step, not this feature. |
| Completeness/strategy | **Unrestricted narrowing + budget.** Enumerate every narrowing step (all non-variable positions x analyzable rules x F2 mgu), breadth-first, bounded by `max_nodes`/`max_depth`. Sound, complete-in-the-limit, honest budget. Mirrors F5's basic-completion v1 scope. |
| Goal form | **Reachability primitive + E-unification wrapper.** `narrow(start, target)` is the primitive; `solve_equation(s, t)` is a thin wrapper returning answer substitutions. |
| Unification | **Reuse F2's `unify`** (first-order, occurs-check, refuse-first on non-first-order). No new unification code. |
| Theory | **Syntactic only.** A loaded AC theory is NOT used (AC-narrowing needs AC-unification, which F3 did not provide -- F3 was matching-only). |

## Architecture

A new core module `rerum/narrowing.py`, mirroring `confluence.py` /
`termination.py` / `completion.py`. Public surface:

```python
def narrow_step(term, rules) -> Iterator[NarrowStep]:
    """Yield every one-step narrowing successor of `term` under `rules`."""

def narrow(engine, start, target, *, max_nodes=1000, max_depth=20) -> NarrowResult:
    """Budget-bounded BFS: find sigma such that sigma(start) narrows to a term
    unifying sigma(target). Returns the FIRST solution."""

def solve_equation(engine, s, t, *, max_nodes=1000, max_depth=20) -> NarrowResult:
    """E-unification: solve s =? t modulo the engine's rules. Thin wrapper over
    narrow via an auto-added reflexivity rule (domain-free, gensym'd)."""

@dataclass(frozen=True)
class NarrowResult:
    found: bool
    substitution: Optional[dict]   # answer {name: term}, restricted to original vars
    derivation: list               # the labeled narrowing steps (the witness)
    nodes_expanded: int
    exhausted: bool                # True = budget hit; False = tree exhausted
```

Properties:

- **Reuses F2 wholesale.** `unify`, `apply_subst`, `rename_apart`,
  `is_analyzable`, plus `instantiate_skeleton` and `gensym`. No new unification.
- **Operates on pattern terms.** Variables in `start` / `target` are
  `["?", name]` (rerum's `?x`), the form `unify` consumes. Rule LHSs are already
  in this form.
- **General-engine principle.** `narrowing.py` names NO domain operator. The
  `eq` / `true` symbols used by `solve_equation` are `gensym`'d fresh, not
  hardcoded. Domains (Peano, lists, ...) arrive as DATA in the engine's rules.
- **Scope guard.** Narrows over ANALYZABLE first-order rules only. Non-analyzable
  rules (`?c` / `?v` / `?free` / `?...` / skeleton-compute) are skipped and
  counted, never errored, consistent with F2/F4/F5.

## The narrowing step

`narrow_step(term, rules)` yields every one-step successor. For each
NON-VARIABLE position `p` in `term` (the subterm `term|p` is not a `["?", _]`
node) and each analyzable rule `(l, skel)`:

```
l_r, skel_r = rename_apart(l, skel, variables(term))   # fresh rule vars, no capture
r_term      = instantiate_skeleton(skel_r, {})         # RHS skeleton -> term ([":",n]->["?",n])
sigma       = unify(term_at(p), l_r)                   # F2 mgu; None -> skip; UnsupportedPattern -> skip rule
successor   = apply_subst(sigma, replace(term, p, r_term))
```

Each yielded `NarrowStep` carries `(successor, sigma, position, rule_id)`.
Positions are integer paths (the `splice_at`/path convention from F1's trace
machinery). Rule variables are renamed apart from `term` BEFORE unification, so a
rule reused down a derivation never captures.

Narrowing is rewriting with the matcher swapped for a unifier: a rewrite step
asks "does `l` MATCH `term|p`?" (only `l`'s variables bind); a narrowing step
asks "can `l` and `term|p` be UNIFIED?" (both sides' variables bind). Everything
else (positions, RHS substitution, search) is structurally the same as
rewriting, so `narrow_step` reads almost exactly like `_all_single_rewrites` with
`unify` in place of `match`.

## The reachability driver

`narrow(engine, start, target, *, max_nodes=1000, max_depth=20)` runs
budget-bounded BFS. Each frontier node carries `(term, theta, depth, derivation)`
where `term` is the narrowed form (with `theta` applied along the path), `theta`
is the composed substitution so far, and `derivation` is the step sequence.

```
rules    = analyzable (l, skel) pairs from engine.rule_set()   # non-analyzable skipped
frontier = deque([(start, {}, 0, [])]);  seen = {key(start, {})}
nodes_expanded = 0
while frontier and nodes_expanded < max_nodes:
    term, theta, depth, deriv = frontier.popleft()
    tau = unify(term, apply_subst(theta, target))              # GOAL CHECK
    if tau is not None:
        sigma = restrict(compose(tau, theta), variables(start) | variables(target))
        return NarrowResult(found=True, substitution=sigma, derivation=deriv,
                            nodes_expanded=nodes_expanded, exhausted=False)
    nodes_expanded += 1
    if depth < max_depth:
        for step in narrow_step(term, rules):                  # successors
            theta2 = compose(step.sigma, theta)
            k = key(step.successor, theta2)
            if k not in seen:
                seen.add(k)
                frontier.append((step.successor, theta2, depth + 1, deriv + [step]))
return NarrowResult(found=False, substitution=None, derivation=[],
                    nodes_expanded=nodes_expanded,
                    exhausted=(nodes_expanded >= max_nodes))
```

- **Rewriting is subsumed.** When a step's mgu binds only rule variables, the
  narrowing step IS a rewrite, so `narrow` interleaves reduction and
  instantiation automatically -- no separate `simplify` pass.
- **`compose(s2, s1)`** = apply `s2` through `s1`'s values, then add `s2`'s
  bindings not already in `s1`. A small helper.
- **Determinism.** Positions in pre-order, rules in `rule_set` order, FIFO
  frontier -> the first solution is reproducible. `seen` keys on `(term, theta)`
  to drop exact duplicates without collapsing distinct-substitution paths.

## The E-unification wrapper

`solve_equation(engine, s, t, *, ...)` reduces equation solving to reachability
via reflexivity, kept domain-free with `gensym`:

```
avoid     = symbols(s) | symbols(t) | symbols(rules)
eq, true_ = gensym("eq", avoid), gensym("true", avoid)   # fresh: NO domain literal
refl      = ([eq, ["?", "x"], ["?", "x"]], true_)        # (eq ?x ?x) -> true_  (RHS already a term)
result    = _narrow_with_rules(rules + [refl], [eq, s, t], true_, ...)
return result with substitution restricted to variables(s) | variables(t)
```

Unrestricted narrowing explores positions INSIDE `s` and `t`, and the reflexivity
rule fires at the root exactly when the two narrowed sides unify -- so
`solve_equation` returns a `sigma` with `sigma(s)` and `sigma(t)` joinable. F2's
`unify` handles the non-linear `(eq ?x ?x)` LHS naturally (unify `s'` with `?x`,
then `t'` with `?x`, which succeeds iff `s'` unifies `t'`). This is textbook
E-unification by narrowing, symmetric and complete-in-the-limit.

A small refactor supports both entry points: a private
`_narrow_with_rules(rules, start, target, *, max_nodes, max_depth)` holds the
BFS; `narrow(engine, ...)` extracts analyzable rules and calls it;
`solve_equation` appends the reflexivity rule and calls it.

## Edge cases and error handling

- **Non-analyzable rules.** An `is_analyzable` pre-scan drops rules with
  `?c` / `?v` / `?free` / `?...` / skeleton-compute nodes (counted, not errored);
  a stray `UnsupportedPattern` from `unify` is also caught per attempt.
- **Occurs-check and capture.** F2's `unify` does the occurs-check;
  `rename_apart` runs PER STEP so a reused rule always gets fresh variables.
  Both are load-bearing for soundness.
- **Infinite / large trees.** `max_nodes` / `max_depth` bound the search;
  `exhausted=True` means the budget was hit, `exhausted=False` with `found=False`
  means a genuinely finite tree was exhausted with no solution.
- **Soundness of the answer.** Every returned `sigma` is witnessed by the
  derivation, and is independently re-verifiable: `sigma(start)` and
  `sigma(target)` are joinable under `engine.simplify`.
- **Engine theory ignored.** Narrowing is syntactic; a loaded AC theory is not
  used. Stated explicitly.

## Testing strategy

- **New `rerum/tests/test_narrowing.py`:**
  - `narrow_step`: one-step successors on a small term -- each
    `(successor, sigma, position)` correct; a variable position yields nothing.
  - Reachability (Peano): `add(0,?y)->?y`, `add(s(?x),?y)->s(add(?x,?y))`;
    `narrow(add(?x, s(0)), s(s(0)))` binds `?x = s(0)`.
  - E-unification (headline): cons-list `append`;
    `solve_equation(append(?xs, [c]), [a,b,c])` binds `?xs = [a, b]`.
  - Budget / exhaustion: a non-terminating rule set hits `max_nodes` ->
    `exhausted=True, found=False`.
  - Determinism: first solution stable across runs.
- **Soundness property (key test):** for each found `sigma`, independently
  verify `sigma(start)` and `sigma(target)` are joinable via `engine.simplify` --
  the answer must re-derive, not just be asserted.
- **A light `examples/` demo** reusing an existing domain (Peano or SKI as data)
  showing equation solving through the GENERAL engine.
- **Guards:** full suite green, `test_mcp_no_domain.py` passes (`narrowing.py`
  names no operator; `eq` / `true` are gensym'd; domains are data), ASCII clean.

## Non-goals (documented boundaries)

- **AC / modulo-theory narrowing.** Needs AC-unification, which we do not have
  (F3 was matching-only). Narrowing is syntactic.
- **Complete set of E-unifiers.** v1 returns the FIRST solution. A lazy
  multi-solution generator (yield successive answers up to budget, like
  `equivalents` vs `prove_equal`) is a natural future increment.
- **basic-narrowing / needed-narrowing optimizations.** v1 is unrestricted; the
  completeness-preserving prunes and optimal constructor-system strategies are
  future.
- **Replacing `solve.py`.** Out of scope by design. `solve` remains the optional
  heuristic escalation; retiring it is a separate future decision.
- **Non-first-order rules.** Rules with `?c` / `?v` / `?free` / `?...` are
  skipped (F2 `unify` refuses them), consistent with F2/F4/F5.

## Invariants preserved

- General-engine principle: `narrowing.py` names no operator; `eq` / `true` are
  gensym'd; domains are data.
- Reuses F2: no new unification primitive.
- Budget honesty: the `exhausted` flag distinguishes budget-hit from
  tree-exhausted.
- Soundness: every answer is witnessed by a derivation and re-verifiable via
  `simplify`.
- Core / optional boundary: narrowing is core and reuses core F2; the optional
  non-core `solve` is untouched (no inverted dependency).

## Dependencies

- F2 (`rerum/confluence.py`): `unify`, `apply_subst`, `rename_apart`,
  `is_analyzable`, `instantiate_skeleton`.
- `rerum/rewriter.py`: `gensym`, term predicates, the path/`splice_at` convention.
- The engine's `rule_set()` (analyzable-rule extraction).

## Out-of-family note

Narrowing is a genuine TRS calculus (Baader and Nipkow, Ch. 7): its steps are
mgu-unification against rule LHSs and its completeness for E-unification over
confluent terminating systems is a theorem. That is what distinguishes it from
the demoted best-first `solve` (heuristic, no completeness) and justifies its
placement in the core, even though it is search-flavored.
