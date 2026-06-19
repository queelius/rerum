# F4: Termination via a lexicographic path order -- design

**Date:** 2026-06-19
**Roadmap:** fourth sub-project of the TRS-frontier roadmap
(`docs/superpowers/specs/2026-06-18-trs-frontier-roadmap.md`), feature F4.
**Status:** design approved; ready for adversarial verification, then the
implementation plan.

## Goal

Let RERUM certify that a rule set TERMINATES, and ORIENT an equation toward a
terminating direction. F4 implements a reduction order -- the lexicographic
path order (LPO) -- from a precedence on operators supplied as DATA. Two
payoffs: (1) `check_termination` proves a rule set cannot loop; (2) `orient`
is the orientation oracle Knuth-Bendix completion (F5) needs. And it closes
the loop with F2: a locally-confluent + terminating system is GLOBALLY
confluent (Newman's Lemma), so `check_confluence` can finally report a true
`confluent` verdict.

## What a reduction order must be

A reduction order `>` is a strict order on terms that is (1) WELL-FOUNDED (no
infinite descending chain -- this is what proves termination), (2) closed under
CONTEXTS (`s > t` implies `C[s] > C[t]`), and (3) closed under SUBSTITUTIONS
(`s > t` implies `sigma(s) > sigma(t)`). If every rule `l -> r` satisfies
`l > r`, then every rewrite step strictly decreases the term in `>`, so no
infinite rewrite sequence exists: the system terminates. LPO is such an order,
derived from a precedence on function symbols alone (no weights).

## Scope

IN scope:
- `lpo_greater(s, t, precedence)`: the LPO on the structured first-order terms
  (plain `["?", name]` variables, constant atoms treated as 0-ary symbols, and
  compounds).
- `orient(l, r, precedence)`: pick the terminating direction (or neither).
- `check_termination(engine, precedence)`: certify every rule is oriented
  `l >_lpo r`; report the unoriented / not-analyzed rules.
- The Newman integration into F2's `check_confluence` (additive).

OUT of scope (reuse F2's refusal -- non-first-order rules are not analyzed):
- Typed/sequence/constrained patterns (`?c`/`?v`/`?free`/`?...`), non-trivial
  skeletons (`:.../!/fresh`), conditional rules. Handled by reusing
  `confluence.is_analyzable`.
- KBO and weight orders (LPO is the chosen order).
- AUTOMATIC precedence synthesis (the user supplies the precedence; F4 does not
  search for one -- that is a completion-era refinement).
- F5 completion itself (consumes `orient` + F2's critical pairs).

## The lexicographic path order

A TERM is a variable (`["?", name]`), a constant ATOM (a bare string/number
that is not a pattern node, e.g. `"0"`, `"true"`, `5`), or a compound
application `[f, arg, ...]`. LPO treats a constant atom `c` as the 0-ARY
application of the function symbol `c` (head `c`, no arguments) -- the standard
treatment of constants as nullary function symbols. A single helper unifies
this: `_head_args(t)` returns `(t, [])` for a non-variable atom and
`(t[0], t[1:])` for a compound.

`precedence` is a LIST of FUNCTION SYMBOLS (operators AND constants) in
DECREASING precedence: the head of the list is the greatest. `["*", "+", "0"]`
means `* > + > 0`. Two helpers:
- `_prec_gt(f, g, precedence)`: True iff both `f` and `g` are in the list and
  `index(f) < index(g)`. Symbols not in the list (or only one of them in it)
  are INCOMPARABLE -- `_prec_gt` returns False both ways. A symbol is never
  greater than itself. The precedence list MUST have no duplicate entries
  (deduplicate / reject duplicates so `index()` is unambiguous and the order is
  a genuine strict partial order on the listed symbols).
- A VARIABLE is `["?", name]`. `_is_var` (from `confluence`).

`lpo_greater(s, t, precedence) -> bool` is True iff `s` strictly dominates `t`:

- If `s == t`: False (strict).
- If `s` is a variable: False. (A variable dominates nothing -- it has no
  subterms and no precedence rank. This anchors well-foundedness.)
- If `t` is a variable (and `s` is not, by the previous case): True iff `t`
  occurs in `s` (necessarily as a PROPER subterm, since `s != t`). I.e.
  `s >_lpo x` iff `x` is buried inside `s`.
- Otherwise both are NON-VARIABLE terms. Let `(f, s1..sn) = _head_args(s)` and
  `(g, t1..tm) = _head_args(t)` (a constant `c` has head `c` and zero args).
  True iff ANY of:
  1. SUBTERM: some argument `si` satisfies `si == t` OR
     `lpo_greater(si, t, precedence)` -- a subterm of `s` already dominates `t`.
     (A constant `s` has no arguments, so case 1 never fires for it.)
  2. PRECEDENCE: `_prec_gt(f, g, precedence)` AND
     `lpo_greater(s, tj, precedence)` for EVERY `tj` -- `s`'s head outranks
     `t`'s head and `s` beats all of `t`'s arguments. (For a constant `t`
     there are no `tj`, so the conjunction over `tj` is vacuously True: a
     compound or constant `s` whose head outranks constant `t` dominates it.)
  3. LEXICOGRAPHIC: `f == g` AND `n == m` (same head AND same arity) AND
     `lpo_greater(s, tj, precedence)` for every `tj` AND the argument tuples
     compare `(s1..sn) >lex (t1..tm)`: there is an index `k` with `si == ti`
     for all `i < k` and `lpo_greater(sk, tk, precedence)`. (Two equal-head
     constants have `n == m == 0` and no differing index, so case 3 yields
     False; constant-vs-equal-constant is already handled by `s == t` / case 2.)

VARIADIC OPERATORS: RERUM operators can be variadic (`+` may be 2-ary or
3-ary). Soundness under variadicity follows from treating each `(symbol, arity)`
as a DISTINCT symbol of an extended signature: case 3's `n == m` guard plus
`_prec_gt`'s symbol-only comparison make different arities of the same operator
PRECEDENCE-INCOMPARABLE (e.g. `+/2` and `+/3` share a list index, so
`_prec_gt(+, +)` is False both ways). LPO over a well-founded partial precedence
is a genuine reduction order, so well-foundedness is preserved and there is NO
infinite descent. A same-head / different-arity pair (`(+ a b c)` vs `(+ a b)`)
is therefore reachable only via case 1 (subterm) -- a COMPLETENESS loss (some
orientations missed), never a soundness loss (never a mis-orientation). The
implementation must compare arities consistently so case 3 never fires across
arities. A normalizing AC theory (F1) is the principled way to relate
different-arity sums.

This is the standard LPO (Baader & Nipkow Def 5.4.12 / the "lpo" of the
literature). Notes:
- Case 1 uses `>=_lpo` (`si == t` OR `si >_lpo t`); cases 2/3's inner checks use
  strict `>_lpo`.
- The recursion terminates: every recursive call has a strictly smaller pair of
  terms by total size, so there is no infinite descent in the algorithm itself.
- Var(r) subset of Var(l) falls out: if `r` contains a variable `x` not in `l`,
  then `lpo_greater(l, r, ...)` must at some point require `lpo_greater(?, x)`
  with `x` not occurring there, which fails -- so the rule is unorientable,
  exactly the classic termination prerequisite.

## orient

```
orient(l, r, precedence):
    if lpo_greater(l, r, precedence): return "lr"
    if lpo_greater(r, l, precedence): return "rl"
    return None
```
For F5 completion, an equation is oriented into the terminating direction;
`None` means this LPO/precedence cannot orient it (try a different precedence,
or it is genuinely problematic, e.g. a commutativity axiom `x+y = y+x`, which
NO reduction order can orient -- that is why completion needs AC, i.e. F3).

## check_termination

`check_termination(engine, precedence) -> TerminationReport`:
1. Build `DirectedRule` records from `engine.rule_set()` (reusing F2's record
   shape and `rule_set()` accessor: post-desugar, disabled groups excluded).
2. For each record: if not `is_analyzable(pattern, skeleton, condition)`
   (reused from `confluence`), record its name in `not_analyzed`. Otherwise
   convert the RHS to a term via `r_term = instantiate_skeleton(skeleton, {})`
   (the empty-substitution call maps `[":", x] -> ["?", x]`, leaving the
   pattern variables free), and test `lpo_greater(pattern, r_term, precedence)`:
   - True -> the rule is oriented; record `(name, "lr")` in `oriented`.
   - False -> record `name` in `unoriented` (this LPO does not prove this rule
     decreasing -- the rule may be reversed, incomparable, or genuinely
     non-terminating).
3. `terminating = (not unoriented) and (not not_analyzed)` -- PROVABLY
   terminating only if every rule is analyzable AND oriented `l >_lpo r`.
   ("Not terminating" here means "not proven by THIS LPO", not "proven
   non-terminating" -- like F2's `unknown`, it is honest about the limit.)

`TerminationReport` (frozen dataclass):
`terminating: bool`, `oriented: List[Tuple[str, str]]` (name, direction),
`unoriented: List[str]`, `not_analyzed: List[str]`.

## The Newman integration into check_confluence (additive)

`check_confluence(engine, *, max_steps=1000, precedence=None)` -- the new
`precedence` keyword is the ONLY signature change; `precedence=None` preserves
F2's behavior byte-for-byte. `ConfluenceReport` gains two fields:
`terminating: Optional[bool]` and `confluent: Optional[bool]` (both default
`None`).

When `precedence is not None`, after computing the critical-pair report:
- `terminating = check_termination(engine, precedence).terminating`.
- `confluent` by the precise logic below. The KEY SUBTLETY (verification-found):
  F2's `locally_confluent is False` has TWO causes, and they must not be
  conflated: a `non_joinable` pair (both legs reduce to DISTINCT IRREDUCIBLE
  normal forms -- a SOUND non-confluence witness, since a confluent system gives
  a critical peak a unique normal form) versus an `unknown` pair (reduction hit
  the budget/cycle -- UNDECIDED, NOT a witness). So `confluent=False` keys on the
  presence of a real witness (`report.non_joinable`), never on bare
  `not locally_confluent`:
  - if `report.non_joinable` is non-empty: `confluent = False` (a distinct-
    irreducible-normal-forms critical pair refutes confluence outright; this is
    strategy-independent because irreducibility is a property of the term, not
    the reduction path).
  - elif `report.unknown` is non-empty: `confluent = None` (undecided critical
    pairs -- the system may or may not be confluent; we cannot say).
  - elif `terminating`: `confluent = True` (here `locally_confluent` is True --
    no `non_joinable`, no `unknown` -- and Newman's Lemma: local confluence +
    termination implies global confluence; the `True` polarity is sound because
    every critical pair was JOINED, and exhibiting a join proves a join exists).
  - else: `confluent = None` (locally confluent but termination not proven --
    Newman does not apply, so the global question is undetermined).

Precondition (F2 guarantee): `ConfluenceReport.locally_confluent` is ALWAYS a
concrete `bool` (never `None`) -- it is `not non_joinable and not unknown` -- so
the branching above is well-defined. Note that `locally_confluent is True`
exactly when `non_joinable` and `unknown` are both empty, so the last two
branches are the `locally_confluent is True` cases.

When `precedence is None`: `terminating = None`, `confluent = None` (F2's
verdict is `locally_confluent` only, exactly as before).

Dependency direction (no load-time cycle): `termination.py` TOP-imports
`DirectedRule`, `is_analyzable`, `instantiate_skeleton` (and `_is_var`) from
`confluence.py`; `confluence.py` LAZILY imports `check_termination` from
`termination.py` INSIDE `check_confluence`, only when `precedence is not None`.

## API

- `engine.check_termination(precedence) -> TerminationReport` -- thin wrapper.
- `engine.check_confluence(*, max_steps=1000, precedence=None)` -- extended.
- PUBLIC surface of `rerum.termination`: `lpo_greater`, `orient`,
  `check_termination`, `TerminationReport`. The helpers `_prec_gt`,
  `_head_args`, and any `_lex_gt` stay PRIVATE (leading underscore, not
  re-exported).
- Re-export from the `rerum` core (TRS family): `lpo_greater`, `orient`,
  `check_termination`, `TerminationReport` (and nothing else from this module).
- `TerminationReport.terminating` is unconditionally a `bool`;
  `ConfluenceReport.terminating` is `Optional[bool]` (`None` exactly when
  `precedence is None`). The type difference is deliberate.

## Design boundaries preserved

- GENERAL ENGINE: the precedence is DATA (a list of function symbols --
  operators and constants). `termination.py` hardcodes no operator. The same
  LPO orders an arithmetic or a boolean rule set. Tests prove both.
- PURE MODULE: `termination.py` is pure functions over `ExprType` and a
  precedence list; `check_termination` READS the engine (`rule_set`) but
  mutates nothing. Read-only.
- REUSE, NOT DUPLICATE: F4 reuses `confluence.DirectedRule`/`is_analyzable`/
  `instantiate_skeleton`/`_is_var` rather than re-deriving the analyzable-rule
  notion or the skeleton->term conversion.

## Testing plan

New file `rerum/tests/test_termination.py`:

1. `_prec_gt` / precedence: `_prec_gt("*", "+", ["*","+"])` is True and
   `_prec_gt("+", "*", ["*","+"])` is False; an unlisted symbol is incomparable
   both ways (`_prec_gt("z", "+", ["*","+"])` and the reverse are False); a
   symbol is not greater than itself (`_prec_gt("+", "+", ["+"])` is False).
2. LPO variable cases: `lpo_greater(["?","x"], ["f","a"], [])` is False (a
   variable dominates nothing); `lpo_greater(["f",["?","x"]], ["?","x"], [])`
   is True (`x` is a proper subterm); `lpo_greater(["f",["?","x"]], ["?","y"],
   [])` is False (`y` not in `s`).
3. CONSTANT cases (the atom-as-0-ary handling): `lpo_greater("a", "b",
   ["a","b"])` is True and `lpo_greater("b", "a", ["a","b"])` is False (constant
   precedence); `lpo_greater(["f","a"], "a", ["f","a"])` is True (compound head
   `f > a`, and `a` has no args -- case 2 vacuous); `lpo_greater("a", ["f","a"],
   ["f","a"])` is False.
4. LPO case 1 (subterm): `lpo_greater(["f", ["g","a"]], ["g","a"], [])` is True
   (the argument `["g","a"]` equals `t`).
5. LPO case 2 (precedence): with `["*","+"]`, `lpo_greater(["*","a","b"],
   ["+","a","b"], ["*","+","a","b"])` is True (`* > +`, and the `*`-term beats
   each `+`-arg `a`, `b` via constant-subterm/precedence); and is False under
   the reversed precedence `["+","*","a","b"]`.
6. LPO case 3 (lexicographic, same head + arity): `lpo_greater(["f","b","a"],
   ["f","a","a"], ["b","a"])` is True (`b > a`, the first differing argument
   decreases, and `s` beats every `tj`).
7. VARIADIC (same head, different arity): `lpo_greater(["+","a","b","c"],
   ["+","a","b"], ["+","a","b","c"])` -- case 3 is SKIPPED (arity mismatch);
   only case 1 (subterm) can fire, so the result is whatever the subterm rule
   gives (assert it does not raise and matches the conservative expectation);
   and the reverse `lpo_greater(["+","a","b"], ["+","a","b","c"], ...)` is
   handled without case 3 firing.
8. `orient`: associativity `(+ (+ ?x ?y) ?z) -> (+ ?x (+ ?y ?z))` orients to
   "lr" -- this holds via case 3 (lexicographic on the shared `+` head) and is
   PRECEDENCE-INDEPENDENT, so assert it with `precedence=[]`; commutativity
   `(+ ?x ?y) = (+ ?y ?x)` orients to `None` (no reduction order can orient a
   permutation of equal multisets); with `precedence=["f","g"]`,
   `orient(["f",["g",["?","x"]]], ["g",["?","x"]], ["f","g"])` is "lr" (subterm)
   and the reversed equation is "rl".
9. `check_termination` POSITIVE: a terminating set under a precedence is
   `terminating is True` with each rule in `oriented` -- e.g. an engine with
   `@assoc: (+ (+ ?x ?y) ?z) => (+ ?x (+ :y :z))` ... use the correct `:` RHS
   syntax; build a small set whose every rule satisfies `l >_lpo r` and assert
   `terminating is True`, `unoriented == []`.
10. `check_termination` NEGATIVE: a commutativity rule (`(+ ?x ?y) => (+ :y :x)`)
    is in `unoriented` and `terminating is False`; a rule whose RHS introduces a
    fresh variable likewise.
11. NOT-ANALYZED: a `?...` rule (and a conditional rule) is in `not_analyzed`;
    `terminating is False` (cannot certify with an unanalyzable rule present).
12. GENERAL: the same `check_termination` certifies a boolean rule set and an
    arithmetic rule set under their own precedences; behavioral generality (no
    operator literal hardcoded in `termination.py`).
13. `instantiate_skeleton({})` reuse contract (pin the relied-upon conversion):
    `instantiate_skeleton([":","x"], {}) == ["?","x"]` and a compound skeleton
    converts to a term whose variables are `["?", name]` (so it is comparable
    to the pattern LHS).
14. NEWMAN payoff via `check_confluence(precedence=...)`:
    - a locally-confluent AND terminating set reports `confluent is True` and
      `terminating is True`;
    - a set with a genuine NON-JOINABLE critical pair (distinct normal forms,
      e.g. F2's `@l: (f (f ?x)) => a` / `@r: (f ?x) => b`) reports
      `confluent is False` (regardless of termination);
    - a locally-confluent, overlap-free, but LPO-UNORIENTABLE set reports
      `confluent is None` and `terminating is False` -- use a single rule with a
      fresh RHS variable, `@fr: (f ?x) => (g ?y)` (no critical pairs, so
      `locally_confluent is True`; unorientable because `?y` is not in the LHS),
      with `precedence=["f","g"]`. (Verify the engine reports
      `non_joinable == []` and `unknown == []` for this set before asserting
      `confluent is None`.)
15. BACKWARD-COMPAT: `check_confluence(engine)` with NO `precedence` returns
    `terminating is None`, `confluent is None`, and `locally_confluent`
    identical to pre-F4; the existing confluence tests are unaffected.
16. WIRING: `engine.check_termination(precedence)` delegates to
    `termination.check_termination`; re-exports `lpo_greater`/`orient`/
    `check_termination`/`TerminationReport` present in `rerum.__all__` (and
    `_prec_gt` is NOT exported); an import-smoke check (`import rerum.termination`
    and `import rerum.confluence` in either order succeeds -- pins the no-cycle
    lazy-import boundary). Full suite green.

## Non-goals (tracked, not built here)

- KBO / weight orders (LPO chosen).
- Automatic precedence search (the user supplies it).
- F5 Knuth-Bendix completion (uses `orient` + F2's critical pairs + this F4
  termination check to drive the orient-and-add loop).
- F3 AC-matching (orients commutativity-style axioms that no plain reduction
  order can -- completion modulo AC).
