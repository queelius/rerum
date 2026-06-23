# TRS-Frontier (F1-F6) Code Review

> Multi-agent review (8 reviewers + per-finding adversarial verification), 2026-06-23.
> Scope: the six core TRS-frontier modules (normalize/confluence/termination/
> completion/acmatch/narrowing), the engine seams, and cross-module consistency.
> Every non-NIT finding below was independently REPRODUCED by a verifier; none refuted.

**Totals:** 29 findings, 20 confirmed (verifier-reproduced), 0 refuted.

## Confirmed findings

| # | adj-sev | feature | id | title |
|---|---------|---------|----|-------|
| 1 | **MAJOR** | F1-normalize | `frac-key-collision-1` | Distinct Fractions with colliding float() projections are merged as like terms, corrupting the value |
| 2 | **MAJOR** | F1-normalize | `theory-malformed-entry-2` | Theory.from_json accepts non-dict signature entries that later crash with an unmapped AttributeError |
| 3 | **MAJOR** | F2-confluence | `budget-1` | check_confluence max_steps budget is ignored on the common no-theory/unconditional fast path |
| 4 | **MAJOR** | F3-acmatch | `typed-rest-1` | Typed rest constraints (?rest:const... / ?rest:var...) are silently dropped, yielding constraint-violating matches |
| 5 | **MAJOR** | F3-acmatch | `empty-pat-crash-1` | Empty-list pattern crashes ac_match with uncaught ValueError; engine does not catch it |
| 6 | **MAJOR** | cross-cutting | `completion-1` | complete reports complete while ignoring non-analyzable rules |
| 7 | **MAJOR** | cross-cutting | `confluence-1` | confluence feeds dangling-RHS variable nodes to engine.simplify |
| 8 | **MAJOR** | engine-integration | `ac-strategy-gap-1` | bottomup/topdown strategies silently ignore the AC theory |
| 9 | **MINOR** | F1-normalize | `nan-non-total-order-3` | NaN operand makes ORDER_KEY non-total, so canonical_sort/normalize of an AC sum is non-confluent |
| 10 | **MINOR** | F1-normalize | `repeat-op-not-ac-4` | collect synthesizing a repeat.op compound that is not itself a declared AC op yields a non-simplified, surprising form |
| 11 | **MINOR** | F2-confluence | `unify-refuse-1` | unify silently binds an unsupported (?c/?v/?free/?...) node nested inside a compound, violating its documented refuse-first contract |
| 12 | **MINOR** | F2-confluence | `notanalyzed-none-1` | Anonymous rules (name=None) collapse to a single [None] entry in not_analyzed, undercounting skipped rules |
| 13 | **MINOR** | F3-acmatch | `budget-non-ac-1` | Work budget only guards the AC multiset enumeration, not positional/nested recursion |
| 14 | **MINOR** | F5-completion | `max-steps-decorative-1` | max_steps parameter is effectively a no-op for the engines completion builds |
| 15 | **MINOR** | F6-narrowing | `narrowing-1` | exhausted=True over-reported when cap-depth node's successors are all already-seen (false inconclusive) |
| 16 | **MINOR** | F6-narrowing | `narrowing-2` | narrow() reachability claim ('reduces to a term unifying sigma(target)') only holds as joinability when target has variables |
| 17 | **MINOR** | cross-cutting | `confluence-2` | not-analyzed dedups on rule.name |
| 18 | **MINOR** | engine-integration | `apply-once-ac-completeness-2` | apply_once / `once` strategy can skip a productive AC binding (and reports applied on no-change) |
| 19 | **MINOR** | engine-integration | `truncated-flag-not-reset-3` | ac_match_truncated is only reset by non-trace simplify, so it is stale after other top-level calls |
| 20 | **MINOR** | engine-integration | `recursionerror-silent-truncation-4` | RecursionError in ac_match is swallowed without setting the truncation flag |

---

### 1. [MAJOR] Distinct Fractions with colliding float() projections are merged as like terms, corrupting the value
- **id / feature:** `frac-key-collision-1` (F1-normalize) -- original severity MAJOR
- **location:** rerum/normalize.py:138 (ORDER_KEY) and rerum/normalize.py:206-228 (_collect_ac)
- **description:** ORDER_KEY keys numbers as (RANK, (float(expr), type_name)). For exact rationals whose magnitude exceeds float's 52-bit mantissa, float() is lossy, so two DISTINCT Fractions get the SAME key. collect_like_terms groups operands by ORDER_KEY, so the two distinct numbers are combined into one group and one is silently discarded. Repro: with the arithmetic theory, a=Fraction(10**18,3), b=Fraction(10**18+1,3) satisfy a!=b but ORDER_KEY(a)==ORDER_KEY(b); normalize(['*', a, b, 'x'], ARITH) returns ['*', 'x', ['^', Fraction(10**18,3), 2]] -- i.e. a*b*x was rewritten to a^2*x, a wrong value. Fraction is a first-class supported numeric type (NUMERIC_TYPES, and the CLAUDE.md repeatedly flags Fraction handling), so this is reachable, not purely theoretical. This violates the idempotent/confluent-AND-meaning-preserving intent of the canonical form.
- **suggested fix:** Make the numeric key exact and lossless. Key numbers on an exact, totally-ordered representation rather than float(): e.g. convert every numeric atom to fractions.Fraction and key on (Fraction.numerator, Fraction.denominator) (or (sign, Fraction)) so equal values share a key and unequal values never collide. Drop the type-name tie-breaker for value comparison (or keep it only as a final, post-value tiebreak) so equal-valued int/Fraction are treated consistently with _same_atom/==. Add a regression test in test_normalize.py with large-denominator Fractions.
- **verification:** The finding is real and reproduces exactly as described. ORDER_KEY (normalize.py:138) keys numeric atoms as (RANK, (float(expr), type_name)). The float() projection is lossy for exact values whose magnitude exceeds float's 53-bit mantissa, so two DISTINCT numbers collapse to the SAME key. _collect_ac (normalize.py:206-228) groups operands purely by ORDER_KEY(base) into a dict, with no value-equality fallback (no _canon_eq / == check on collision), so two distinct numbers landing on the same key are merged into one group; one is silently discarded and the surviving one is given a multiplicity of 2.

Verified concretely: with a=Fraction(10**18,3) and b=Fraction(10**18+1,3), a!=b but float(a)== [...]
- **repro:** Probe (run from /home/spinoza/github/repos/rerum):

python3 -c "
from fractions import Fraction
from rerum.normalize import ORDER_KEY, Theory, normalize
a = Fraction(10**18, 3); b = Fraction(10**18 + 1, 3)
print('a != b           :', a != b)                       # True
print('float(a)==float(b):', float(a) == float(b))        # True
print('ORDER_KEY(a)     :', ORDER_KEY(a))                 # (0,(3.33e17,'Fraction'))
print('ORDER_KEY(b)     :', ORDER_KEY(b))                 # same
print('keys equal       :', ORDER_KEY(a)==ORDER_KEY(b))   # True
ARITH = Theory.from_dict({
  '+': {'ac':True,'identity':0,'repeat':{'op':'*','via':'count'}},
  '*': {'ac':True,'identity':1,'annihilator':0,'repeat' [...]

### 2. [MAJOR] Theory.from_json accepts non-dict signature entries that later crash with an unmapped AttributeError
- **id / feature:** `theory-malformed-entry-2` (F1-normalize) -- original severity MAJOR
- **location:** rerum/normalize.py:66 (has_ac), :62 (is_ac), :43-58 (__init__/from_json); consumed at rerum/mcp/persistence.py:130
- **description:** Theory.from_json only validates that the TOP LEVEL is an object; it does not validate that each value is a dict. has_ac() iterates self._sig.values() and calls entry.get('ac', False) unconditionally, and is_ac() calls entry.get(...) after a bool(entry) guard that is True for a non-empty string. So a theory like {'+': 'garbage'} (or list/int/None entry) is accepted by from_json and then crashes with AttributeError: 'str' object has no attribute 'get' the first time any reasoning runs. Repro: Theory.from_json('{"+": "garbage"}').has_ac() raises AttributeError; normalize(['+','a','b'], that_theory) raises AttributeError. This matters because mcp/persistence.py load_theory deliberately catches only ValueError from from_json to map malformed theories to a clean 'parse_error'; a malformed-entry theory slips past, then later blows up deep inside solve_goal/normalize with an unmapped AttributeError (i.e. a 500-class failure rather than the intended parse_error).
- **suggested fix:** Validate entry shape at construction. In Theory.__init__ (or from_dict/from_json), reject any signature value that is not a dict with a clear ValueError naming the offending operator, e.g. for op, entry in (sig or {}).items(): if not isinstance(entry, dict): raise ValueError(f'theory entry for {op!r} must be an object, got {type(entry).__name__}'). This keeps the MCP parse_error mapping correct and makes accessors safe. Add a test in TestTheoryFromJsonValidation.
- **verification:** The finding is accurate on every point I could check.

1. Code reading confirms the shape: `Theory.from_json` (normalize.py:51-58) validates ONLY that the top-level parse is a dict; it does NOT validate that each value is a dict, then passes `parsed` straight to `Theory(parsed)`. `Theory.__init__` (43-44) just does `dict(sig or {})` with no per-entry check.

2. The accessors crash as described. `has_ac` (66) calls `entry.get('ac', False)` unconditionally over `self._sig.values()` -- so ANY non-dict entry (str/list/int/None) raises AttributeError. `is_ac` (60-62) guards with `bool(entry)` first, so a None entry is safe there, but a non-empty string/list/int passes the truthiness guard and then [...]
- **repro:** Direct accessor repro:
$ python3 -c "from rerum.normalize import Theory, normalize; t=Theory.from_json('{\"+\": \"garbage\"}'); print(repr(t)); t.has_ac()"
Theory(['+'])
AttributeError: 'str' object has no attribute 'get'

Coverage of accessors and entry types:
  is_ac('+')                  -> AttributeError: 'str' object has no attribute 'get'
  normalize(['+','a','b'], t) -> AttributeError: 'str' object has no attribute 'get'
  entry [1,2,3]: has_ac -> AttributeError: 'list' object has no attribute 'get'
  entry 5:       has_ac -> AttributeError: 'int' object has no attribute 'get'
  entry null:    has_ac -> AttributeError: 'NoneType' object has no attribute 'get'

End-to-end through the r [...]

### 3. [MAJOR] check_confluence max_steps budget is ignored on the common no-theory/unconditional fast path
- **id / feature:** `budget-1` (F2-confluence) -- original severity MAJOR
- **location:** rerum/confluence.py:_decide_joinable (lines 412-414) and check_confluence signature (line 428); root cause rerum/rewriter.py:1184-1196 (rewriter().simplify hardcodes max_iterations=1000)
- **description:** _decide_joinable reduces each critical-pair leg with engine.simplify(cp.left, max_steps=max_steps). engine.simplify only honors max_steps on the slow path (_simplify_exhaustive uses `for _ in range(max_steps)`). For a plain rule set with no theory, no conditions, no groups, and no hooks -- the dominant confluence-analysis case -- simplify takes the fast path (self._simplifier = rewriter(...)), whose internal loop is hardcoded to max_iterations=1000 and never receives max_steps. So the documented work budget is a silent no-op there. Repro: `RuleEngine.from_dsl('@grow: (s ?x) => (s (s :x))').simplify(['s','c'], max_steps=3)` and `... max_steps=50` BOTH ignore the budget and run to the internal 1000 cap, raising RecursionError after ~0.37s; the same rule on a group-forced slow path returns size-9 (max_steps=3) and size-103 (max_steps=50), i.e. honors it. End-to-end: `check_confluence(eng_with_grow, max_steps=2)` spends ~0.8s (RecursionError caught -> unknown) instead of the tiny budget requested. Soundness is preserved (an over-budget leg is non-normal -> unknown, conservative; Recursio [...]
- **suggested fix:** Make the budget real on the fast path too: thread max_steps into rewriter()/its simplify (replace the hardcoded 1000 with the caller's value), or in _decide_joinable bypass the fast path by always calling the budget-honoring strategy (e.g. engine.simplify(cp.left, max_steps=max_steps, strategy='exhaustive') routed so it cannot fall into the unbounded rewriter()). At minimum, document that max_steps is only enforced when a theory/condition/group/hook is active, and add a confluence test that asserts a growth-rule leg under a small max_steps is bounded (size scales with max_steps) on a plain rul [...]
- **verification:** Verified by code reading + empirical reproduction. (1) engine.simplify (engine.py:2506-2528) takes a fast path when there are no conditions, no groups, no AC theory, and no hooks; that path calls self._simplifier(expr) (line 2528), the cached rewriter() closure, WITHOUT passing max_steps. The closure's loop is hardcoded to max_iterations=1000 (rewriter.py:1193). So the caller's budget is discarded on the most common path. (2) confluence.py:_decide_joinable (lines 413-414) reduces each critical-pair leg via engine.simplify(cp.left, max_steps=max_steps); its docstring and check_confluence's docstring (line 428) advertise max_steps as a work budget, so the advertised guarantee is a silent no-op [...]
- **repro:** Fast path ignores budget (unit):
  eng = RuleEngine.from_dsl('@grow: (s ?x) => (s (s :x))')
  eng.simplify(['s','c'], max_steps=3)  -> RecursionError after 0.366s
  eng.simplify(['s','c'], max_steps=50) -> RecursionError after 0.378s

Slow path (group-forced) honors budget on same bare grow rule:
  dsl = '\n[g]\n@grow: (s ?x) => (s (s :x))\n'
  eng.simplify(['s','c'], max_steps=3,  groups=['g']) -> size=9   in 0.000s
  eng.simplify(['s','c'], max_steps=50, groups=['g']) -> size=103 in 0.003s

End-to-end check_confluence on a 2-rule overlap that yields a growth leg:
  dsl = '@a: (f (s ?x)) => (g :x)\n@grow: (s ?x) => (s (s :x))'
  eng = RuleEngine.from_dsl(dsl)   # has_conditions=False, disab [...]

### 4. [MAJOR] Typed rest constraints (?rest:const... / ?rest:var...) are silently dropped, yielding constraint-violating matches
- **id / feature:** `typed-rest-1` (F3-acmatch) -- original severity MAJOR
- **location:** rerum/acmatch.py:184-189 (bare rest), :236 (_match_ac rest), :265-272 (_match_positional rest)
- **description:** match_compound() in rewriter.py enforces the rest-type constraint (rest_type_constraint -> each remaining item must be constant/variable). None of the three rest-binding sites in acmatch.py consult rest_type_constraint, so the constraint is ignored. This breaks both the central 'every yield is a REAL match' claim (the yielded binding does not satisfy the pattern's stated constraint) and the 'no-theory path byte-identical to match()' claim. Repro (no-theory): `ac_match(['f', ['?...','rest','const']], ['f',1,'a',3], NO_AC)` yields `{'rest':[1,'a',3]}` while `match(...)` returns None. End-to-end under AC: rule `(+ ?x ?rest:const...) => (matched :x)` fires on `(+ 1 a b)` binding rest=[a,b] (a,b are symbols, not constants). Reachable from the DSL: `?rest:const...` parses to `['?...','rest','const']` (engine.py:545).
- **suggested fix:** Before binding any rest variable, apply the type constraint: read `rest_type_constraint(rest_pat)` (already imported-able from rewriter) and, if non-None, reject (do not yield) any candidate leftover/tail list containing an item that fails `constant`/`variable`. Apply at all three sites: the bare-rest branch (184-189), the `_match_ac` leftover binding (236), and the `_match_positional` rest branch (269). Add tests with const/var rests under both NO_AC and an AC theory.
- **verification:** CONFIRMED as a real soundness bug. The finding claims that typed rest constraints (?rest:const... / ?rest:var...) are silently dropped at all three rest-binding sites in rerum/acmatch.py, producing constraint-violating matches that diverge from match() in rewriter.py.

Code inspection: rewriter.match_compound (rewriter.py:931-938) reads rest_type_constraint(current_pat) and rejects the match (returns None) if any remaining item fails constant()/variable(). None of the three rest-binding sites in acmatch.py reference rest_type_constraint at all: the bare-rest branch (acmatch.py:184-189) just _bind()s the whole list; the _match_ac leftover branch (236) _bind()s leftover unconditionally; and th [...]
- **repro:** All probes run from /home/spinoza/github/repos/rerum:

# DSL reachability
$ python3 -c "from rerum.engine import parse_sexpr; print(parse_sexpr('(+ ?x ?rest:const...)'))"
['+', ['?', 'x'], ['?...', 'rest', 'const']]

# Probe 1 (finding's primary repro; no-theory, routes via _match_positional:269)
from rerum.acmatch import ac_match; from rerum.normalize import Theory; from rerum.rewriter import match
NO_AC = Theory({})
list(ac_match(['f',['?...','rest','const']], ['f',1,'a',3], NO_AC)) -> [{'rest': [1, 'a', 3]}]
match(['f',['?...','rest','const']], ['f',1,'a',3]) -> None

# Probe 2 (genuine bare-rest branch 184-189)
from rerum.acmatch import _ac_match_core; from rerum.rewriter import Bindings [...]

### 5. [MAJOR] Empty-list pattern crashes ac_match with uncaught ValueError; engine does not catch it
- **id / feature:** `empty-pat-crash-1` (F3-acmatch) -- original severity BLOCKING
- **location:** rerum/acmatch.py:176 (_ac_match_core dispatch) via rewriter.car at :475
- **description:** `_ac_match_core` immediately calls `arbitrary_constant(pat)` etc., which call `car(pat)`; on `pat == []` that raises `ValueError: car: argument is an empty list`. The rewriter's `match` handles this via `null(pat)` (returns Bindings({}) for [] vs [], None otherwise). Empty-list patterns arise as nested sub-patterns, e.g. `['f', []]` (an empty application `(f ())`). Confirmed end-to-end: `RuleEngine.from_dsl('@r: (f ()) => done')` with an AC theory loaded raises `ValueError` on `eng.simplify(['f', []])`, because `_match_lhs` routes through `ac_match` whenever `_theory_has_ac()` and the surrounding try/except in engine.py:1893-1897 catches only `RecursionError`. The same input simplifies fine without an AC theory. This is a crash and a divergence from match().
- **suggested fix:** At the top of `_ac_match_core`, mirror the rewriter: `if pat == []: \n    if exp == []: yield bindings\n    return` (before any predicate that calls car). Equivalently guard with `not pat`. Add regression tests for `[]` vs `[]`, `[]` vs `['a']`, and nested `['f', []]` under both NO_AC and an AC theory.
- **verification:** CONFIRMED. The dispatch in `_ac_match_core` (acmatch.py:176-177) calls `arbitrary_constant(pat)`/`arbitrary_variable(pat)`/`arbitrary_expression(pat)`/`arbitrary_free(pat)` with no empty-list guard. Each of those predicates (rewriter.py:584-601) is `compound(pat) and car(pat) == "..."`; since `compound([])` is `isinstance([], list) == True`, the short-circuit does NOT save it and `car([])` runs, raising `ValueError: car: argument is an empty list` (rewriter.py:475). By contrast, `rewriter.match`/`match_compound` guard empty lists structurally via `null(pat)`/`null(exp)` (rewriter.py:916-942), returning `Bindings({})` for `[] vs []` and `None` for `[] vs ['a']`. So the AC path genuinely diver [...]
- **repro:** End-to-end probe (the finding's exact scenario):\n\n$ python3 -c "\nfrom rerum.engine import RuleEngine\nfrom rerum.normalize import Theory\ntheory = Theory({'+': {'ac': True}})\neng = RuleEngine.from_dsl('@r: (f ()) => done')\nprint('NO theory:', eng.simplify(['f', []]))\neng.with_theory(theory)\ntry:\n    print('AC theory:', eng.simplify(['f', []]))\nexcept Exception as e:\n    print('AC theory RAISED:', type(e).__name__, e)\n"\nis_ac(+): True has_ac: True\nNO theory  simplify([f, []]) -> done\nAC theory  simplify([f, []]) RAISED: ValueError car: argument is an empty list\n\nTraceback (filtered) showing the dispatch-to-car chain:\n  engine.py:1894 _match_lhs\n  acmatch.py:166 ac_match\n  a [...]

### 6. [MAJOR] complete reports complete while ignoring non-analyzable rules
- **id / feature:** `completion-1` (cross-cutting) -- original severity MAJOR
- **location:** completion.py:118-121
- **description:** discards critical_pairs not-analyzed; reports complete with non-analyzable rule overlaps untested. Repro confirmed.
- **suggested fix:** assert not-analyzed empty or pre-validate with is_analyzable
- **verification:** The finding is real. completion.py:121 binds `pairs, _na = critical_pairs(records)` and discards `_na` (the not_analyzed rule names), and the comment at lines 118-120 asserts "not_analyzed is always empty here: every rule the loop builds is a first-order [pattern, [:,x]-skeleton] with no condition, so is_analyzable accepts it." That invariant is FALSE: the loop never validates the INPUT equations with is_analyzable. The first iteration's `rules` come straight from the oriented input equations (lines 96-105), and `orient` accepts non-first-order terms (verified: orient(['f','a',['?...','rest']],'b',...) returns 'lr'). If an input equation LHS contains a rest pattern (?...) or other non-first- [...]
- **repro:** Probe 1 (single non-analyzable equation):
$ python3 -c "import rerum.completion as cmp; from rerum.confluence import critical_pairs, DirectedRule, is_analyzable; from rerum.completion import _term_to_skeleton; l=['f',['?...','rest']]; r=['g',['?...','rest']]; sk=_term_to_skeleton(r); print('is_analyzable=',is_analyzable(l,sk,None)); rec=DirectedRule(name='0',pattern=l,skeleton=sk,condition=None); pairs,na=critical_pairs([rec]); print('pairs=',pairs,'not_analyzed=',na); res=cmp.complete([(l,r)],['f','g']); print('status=',res.status)"
=> is_analyzable= False
   pairs= []  not_analyzed= ['0']
   status= complete   (rule was never analyzed, yet 'complete')

Probe 2 (concrete non-confluent syste [...]

### 7. [MAJOR] confluence feeds dangling-RHS variable nodes to engine.simplify
- **id / feature:** `confluence-1` (cross-cutting) -- original severity MAJOR
- **location:** confluence.py:393-425
- **description:** dangling-RHS free-var node reaches engine.simplify as rewritable pattern; real reduction grounds it to a symbol. narrowing skips, confluence does not. Repro confirmed.
- **suggested fix:** apply narrowing variable-containment check in check_confluence
- **verification:** The finding is real and the code path exists exactly as described.

Mechanism: A rule with a dangling RHS variable (a `[":", name]` skeleton reference with no matching `["?", name]` binder in the LHS) is judged ANALYZABLE by `confluence.is_analyzable` (it checks only the condition and the marker bad-sets `_PATTERN_BAD`/`_SKELETON_BAD`, which do NOT include a plain `:x`; verified: `is_analyzable` returns True). When `critical_pairs` builds the pair, `instantiate_skeleton` (confluence.py:193-194) models the unbound `:b` as the free-variable pattern node `["?", "b"]`. But the REAL engine reduces the same dangling `:b` via `Bindings.lookup` (rewriter.py:161-162), which returns the bare ground sy [...]
- **repro:** Repro (python3 -c), run from /home/spinoza/github/repos/rerum:

  from rerum.engine import RuleEngine
  eng = RuleEngine()
  eng.load_dsl('''
  (f (a)) => (g :b)      # dangling RHS variable :b (no ?b binder)
  (a) => c
  (g b) => (f c)         # joining rule, keyed on the GROUND symbol b
  ''')
  rep = eng.check_confluence()
  print('locally_confluent:', rep.locally_confluent)
  print('non_joinable:', [(cp.left, cp.right) for cp in rep.non_joinable])
  # ground truth: both real reducts of (f (a)) join
  print('simplify (g b) ->', eng.simplify(['g','b']))    # the real left reduct
  print('simplify (f c) ->', eng.simplify(['f','c']))    # the right reduct

Output:
  locally_confluent: False
 [...]

### 8. [MAJOR] bottomup/topdown strategies silently ignore the AC theory
- **id / feature:** `ac-strategy-gap-1` (engine-integration) -- original severity MAJOR
- **location:** rerum/engine.py:3117 (_bottomup_pass) and 3223 (_topdown_pass); gate at simplify() lines 2506-2534
- **description:** Only the exhaustive driver (_simplify_exhaustive), apply_once, and _all_single_rewrites were rewired to _match_lhs. _bottomup_pass (line 3117) and _topdown_pass (line 3223) still call _match_internal(pattern, current) directly, so they never invoke ac_match and never canonicalize. The fast-path gate's has_ac check only governs the exhaustive branch. Result: the SAME engine + AC theory + rule gives divergent results across strategies with no error or warning. Repro: th=Theory({'+':{'ac':True,'identity':0}}); eng.with_theory(th); eng.load_dsl('(+ a b) => MATCHED'); subj=['+','b','a']; simplify(subj,strategy='exhaustive') -> 'MATCHED' and strategy='once' -> 'MATCHED', but strategy='bottomup' -> ['+','b','a'] and strategy='topdown' -> ['+','b','a']. The no-theory behavior of all strategies is unchanged (verified), so this is purely a silent AC-omission, but it is a real cross-strategy honesty/completeness divergence a user can hit by simply choosing bottomup with a theory loaded.
- **suggested fix:** Either (a) route _bottomup_pass/_topdown_pass through self._match_lhs and canonicalize like _simplify_exhaustive, or (b) if leaving them non-AC is intentional, make simplify() raise (or warn) when strategy in {'bottomup','topdown'} and self._theory_has_ac(), and document the limitation in the docstring and CLAUDE.md alongside the other strategy notes.
- **verification:** Confirmed by direct reproduction and code inspection. (1) Code path exists exactly as described: _simplify_exhaustive uses the AC-aware self._match_lhs (engine.py:2965) and self._canonicalize (2948), whereas _bottomup_pass (engine.py:3117) and _topdown_pass (engine.py:3223) call the bare structural matcher _match_internal directly, with no _match_lhs, no _canonicalize, no _theory_has_ac, and no warning/guard anywhere in the 3049-3290 driver region. _match_lhs (1878-1900) is the only place that routes into acmatch.ac_match under an AC theory; bottomup/topdown never reach it. (2) The simplify() gate (2504-2507) checks has_ac only to choose _simplify_exhaustive over the fast path within the str [...]
- **repro:** Probe:
python3 -c "
from rerum.engine import RuleEngine
from rerum.normalize import Theory
th = Theory({'+': {'ac': True, 'identity': 0}})
eng = RuleEngine().with_theory(th)
eng.load_dsl('(+ a b) => MATCHED')
subj = ['+', 'b', 'a']
for strat in ('exhaustive','once','bottomup','topdown'):
    print(strat, eng.simplify(subj, strategy=strat))
"
Output:
 exhaustive: 'MATCHED'
       once: 'MATCHED'
   bottomup: ['+', 'b', 'a']
    topdown: ['+', 'b', 'a']

Silence/control checks:
- warnings emitted: []   ; ac_match_truncated: False
- NO-THEORY (+ b a): all four strategies -> ['+','b','a'] (agree); NO-THEORY (+ a b): all four -> 'MATCHED' (agree). Divergence is AC-only.
- NESTED redex ['f', ['+', [...]

### 9. [MINOR] NaN operand makes ORDER_KEY non-total, so canonical_sort/normalize of an AC sum is non-confluent
- **id / feature:** `nan-non-total-order-3` (F1-normalize) -- original severity MINOR
- **location:** rerum/normalize.py:135-138 (ORDER_KEY) and :148-163 (canonical_sort)
- **description:** ORDER_KEY(nan) = (0, (nan, 'float')). Because nan compares False against everything, sorted() with a NaN key is not a total order and is permutation-dependent. Repro: canonical_sort(['+', nan, 1.0, 2.0], ARITH) yields 4 different operand orders across the 3! input permutations (e.g. one output is ['+', 1.0, 2.0, nan], another ['+', nan, 1.0, 2.0]), so two AC-equal inputs do not converge -- a direct counterexample to the unconditional confluence claim in the module docstring (lines 11-12). NaN is an exotic atom, hence MINOR, but the claim is stated without qualification.
- **suggested fix:** Either (a) sort non-finite floats into a deterministic bucket: detect math.isnan/isinf in the numeric branch of ORDER_KEY and assign a fixed sentinel sub-key (e.g. NaN -> a max sentinel, +/-inf -> their signed extremes) so the order is total; or (b) explicitly document that operands must be finite and reject non-finite numeric atoms at normalize entry. Add a confluence test over a sum containing a non-finite operand.
- **verification:** The finding is real and precisely described. (1) Code path confirmed: ORDER_KEY (normalize.py:135-138) returns (0, (float(expr), type-name)) for numeric atoms, so ORDER_KEY(nan) = (0, (nan, 'float')). canonical_sort (148-163) calls sorted(sorted_args, key=ORDER_KEY) for AC heads. Since NaN compares False against everything, the sort key is not a total order and Python's Timsort result is permutation-dependent. (2) Reproduced: over the 3! permutations of the AC-equal multiset {nan, 1.0, 2.0} under an AC '+' theory, canonical_sort yields 4 distinct operand orders, and the top-level normalize() yields 4 distinct normal forms. (3) Direct counterexample to the docstring's unqualified confluence c [...]
- **repro:** Probe 1 (canonical_sort + ORDER_KEY):
$ python3 -c "import itertools; from rerum.normalize import canonical_sort, Theory, ORDER_KEY; nan=float('nan'); ARITH=Theory.from_dict({'+':{'ac':True,'identity':0}}); print('ORDER_KEY(nan)=',ORDER_KEY(nan)); outs=set(); [outs.add(tuple('nan' if (isinstance(x,float) and x!=x) else x for x in canonical_sort(['+']+list(p),ARITH)[1:])) for p in itertools.permutations([nan,1.0,2.0])]; print('distinct outputs:',len(outs))"
=> ORDER_KEY(nan)= (0, (nan, 'float'))
=> distinct outputs: 4  [('nan',1.0,2.0), (1.0,'nan',2.0), (1.0,2.0,'nan'), (2.0,'nan',1.0)]

Probe 2 (normalize entry point, direct confluence counterexample):
$ python3 -c "from rerum.normalize impo [...]

### 10. [MINOR] collect synthesizing a repeat.op compound that is not itself a declared AC op yields a non-simplified, surprising form
- **id / feature:** `repeat-op-not-ac-4` (F1-normalize) -- original severity MINOR
- **location:** rerum/normalize.py:183-203 (_emit_group) and :363-378 (_normalize_once pipeline order)
- **description:** _emit_group builds [repeat.op, total, base] (or [repeat.op, base, total]). If the theory declares the parent op AC with a repeat whose 'op' is NOT also declared as a (unit-bearing) AC operator in the same theory, the synthesized compound is never simplified by fold. Repro: with T = Theory.from_dict({'+': {'ac': True, 'identity': 0, 'repeat': {'op': '*', 'via': 'count'}}}) (note: '*' is NOT declared), normalize(['+', 0, 0], T) returns ['*', 2, 0] instead of 0, because fold only collapses annihilators/identities for AC ops and '*' is not AC here. The output is still idempotent and confluent (deterministic), so it does NOT break the central claim, but '(+ 0 0) -> (* 2 0)' is a genuinely surprising result and a quiet correctness trap for a theory author who declares a repeat op without fully specifying it.
- **suggested fix:** Either validate the theory (at construction) that every repeat.op referenced is itself a declared, AC operator with consistent units, raising a clear ValueError otherwise; or document explicitly that repeat.op must be a fully-specified AC operator in the same theory. A defensive alternative is to have collect avoid emitting a coefficient/power form when the base is the parent op's identity/annihilator, but theory-level validation is the principled fix.
- **verification:** The finding is accurate in every checkable particular.

CODE PATH (verified by reading normalize.py):
- `Theory.__init__` (lines 43-44) just stores `dict(sig)`. There is NO construction-time validation that a declared `repeat.op` is itself a declared AC operator. A theory referencing a totally undeclared `repeat.op` (e.g. "NONEXISTENT") is accepted silently.
- `_normalize_once` (lines 363-378) runs the pipeline in the order flatten -> sort -> collect -> fold. `collect_like_terms` (via `_collect_ac` -> `_emit_group`, lines 183-203) synthesizes `[repeat.op, total, base]` (here `['*', 2, 0]`) for `via="count"`. Crucially, `_emit_group` emits this BEFORE the fold step.
- `_fold_constants` (lines [...]
- **repro:** Probe 1 (exact finding repro):
$ python3 -c "from rerum.normalize import Theory, normalize; T = Theory.from_dict({'+': {'ac': True, 'identity': 0, 'repeat': {'op': '*', 'via': 'count'}}}); print(normalize(['+', 0, 0], T))"
['*', 2, 0]        # surprising: expected 0
And (+ 0 0 0) -> ['*', 3, 0]; (+ x x) -> ['*', 2, 'x'].

Probe 2 (pipeline trace shows collect-before-fold is the cause):
flatten: ['+', 0, 0]
sort:    ['+', 0, 0]
collect: ['*', 2, 0]     # _emit_group synthesizes coefficient form
fold:    ['*', 2, 0]     # '*' not AC -> _fold_constants leaves it; annihilator never applied
normalize twice == normalize once == ['*', 2, 0]  -> idempotent (central claim intact)

Probe 3 (root cause [...]

### 11. [MINOR] unify silently binds an unsupported (?c/?v/?free/?...) node nested inside a compound, violating its documented refuse-first contract
- **id / feature:** `unify-refuse-1` (F2-confluence) -- original severity MINOR
- **location:** rerum/confluence.py:unify (lines 133-150, _unsupported at 95-96) and _unify_var (171-178)
- **description:** _unsupported only inspects the ROOT of t1/t2, and the top-of-function guard runs once per call before descent. When a variable unifies against a compound that CONTAINS a nested unsupported node, _unify_var binds the whole compound (occurs-check passes, no re-scan) and unify returns a substitution instead of raising. Repro: `unify(['?','x'], ['f', ['?c','y']])` returns `{'x': ['f', ['?c','y']]}` with no UnsupportedPattern, capturing ?c as opaque data. The unify docstring explicitly claims it 'Raises UnsupportedPattern on any ?c/?v/?free/?... or skeleton-only node, checked BEFORE the variable/compound branches so a typed node is never bound as opaque.' This is a public re-export (rerum.unify), so a direct caller can be misled into treating an unsupported pattern as literal structure and conclude a spurious overlap. It does NOT affect critical_pairs/check_confluence: is_analyzable recursively (_has_marker) refuses any rule whose pattern contains such a node before unify is ever called, so the soundness of the confluence verdict is intact (confirmed: is_analyzable(['f',['?c','x']],...) i [...]
- **suggested fix:** Recurse the unsupported-node check, or scan binding values: in _unify_var (and/or when binding in _compose_bind) reject `other` if it contains any unsupported node (a recursive variant of _unsupported / reuse _has_marker over the union of _PATTERN_BAD and skeleton markers). Alternatively, soften the docstring to state the guard is root-only and that callers must pre-screen with is_analyzable. Add a test mirroring test_nested_unsupported_raises but with the unsupported node reached via a variable binding (`unify(['?','x'], ['f', ['?c','y']])`).
- **verification:** The finding is real and reproduces exactly. The root guard in unify (confluence.py:149) `_unsupported(t1) or _unsupported(t2)` only inspects the ROOTS of the two arguments (via _unsupported at 95-96, which checks `t[0] in _UNSUPPORTED_HEADS`). When a variable `['?','x']` unifies against a compound `['f', ['?c','y']]`, the root heads are `?` and `f`, neither unsupported, so no raise. unify then takes the var branch (line 153-154 -> _unify_var at 171), occurs-check passes (x does not occur), and _compose_bind binds `x -> ['f', ['?c','y']]` with NO re-scan of the bound value. The nested `?c` node is captured as opaque data. This directly violates the docstring's unconditional claim (lines 138-1 [...]
- **repro:** Probe (python3 -c, importing the public re-export):

  from rerum import unify, UnsupportedPattern
  # Var unifies against a compound containing a nested ?c:
  unify(['?','x'], ['f', ['?c','y']])
    -> NO EXCEPTION; returned {'x': ['f', ['?c', 'y']]}   # ?c captured opaquely

  # Contrast: structural-descent path (the existing test) DOES raise:
  unify(['f', ['?c','x']], ['f', 'a'])
    -> UnsupportedPattern: cannot unify pattern form: ['?c', 'x'] ~ 'a'

  # Symmetric direction also leaks:
  unify(['f', ['?c','y']], ['?','x'])  -> {'x': ['f', ['?c', 'y']]}  (no raise)
  # Other markers also leak through the var binding:
  unify(['?','x'], ['g', ['?...','rest']]) -> {'x': ['g', ['?...', 'res [...]

### 12. [MINOR] Anonymous rules (name=None) collapse to a single [None] entry in not_analyzed, undercounting skipped rules
- **id / feature:** `notanalyzed-none-1` (F2-confluence) -- original severity MINOR
- **location:** rerum/confluence.py:critical_pairs skip()/seen_skips (lines 313-319)
- **description:** skip() dedups on rule.name; for unnamed rules name is None, so multiple distinct non-analyzable anonymous rules all map to the one key None and only a single None is appended. RuleMetadata.name defaults to None (engine.py:101,114), and DSL rules written without an `@name:` prefix are anonymous, so this is reachable from the real engine path, not just hand-built DirectedRules. Repro via engine: loading two unnamed rest-pattern rules `(f ?x...) => (g :x...)` and `(h ?x...) => (k :x...)` yields report.not_analyzed == [None] (length 1) for two skipped rules. The verdict stays sound (skipping is conservative), but the report understates how many rules went unanalyzed and leaks a bare None into a user-/JSON-facing list, which is awkward for callers that sort or serialize it.
- **suggested fix:** Dedup on rule identity rather than name (e.g. id(rule) or the rule's index), and surface a stable label for anonymous rules in not_analyzed (e.g. '<anonymous#k>' or the pattern's s-expr) instead of None, mirroring RuleMetadata's '<anonymous>' rendering. Add a test with two distinct unnamed non-analyzable rules asserting two entries appear.
- **verification:** All claims verified against the actual code and reproduced through the real engine path.

CODE: rerum/confluence.py:315-319 -- `skip(rule)` sets `key = rule.name` and dedups via `seen_skips`, appending `key` to `not_analyzed`. `DirectedRule.name` is `Optional[str]` (line 279). For anonymous rules `name is None`, so multiple distinct non-analyzable anonymous rules all map to the single key `None` and only one `None` is appended.

REACHABILITY: `RuleMetadata.name` defaults to `None` (engine.py:101 param default, :114 assignment). DSL rules without an `@name:` prefix are anonymous (confirmed: `rule_set()` shows `(None, ...)` for both). `engine.check_confluence`/`engine.critical_pairs` build `Dir [...]
- **repro:** Probe (engine DSL path):

  from rerum.engine import RuleEngine
  eng = RuleEngine()
  eng.load_dsl('(f ?x...) => (g :x...)')   # anonymous, non-analyzable (rest pattern)
  eng.load_dsl('(h ?x...) => (k :x...)')   # anonymous, distinct, non-analyzable
  rep = eng.check_confluence()
  print(rep.not_analyzed, len(rep.not_analyzed))

Output:
  report.not_analyzed: [None]
  len(report.not_analyzed): 1

Control with named rules (@n1:, @n2:) -> not_analyzed: ['n1','n2'] len 2.
Mixed (@n1: + one anonymous) -> not_analyzed: ['n1', None] len 2, and:
  sorted(['n1', None]) -> TypeError: '<' not supported between instances of 'NoneType' and 'str'
  json.dumps([None]) -> '[null]'

Soundness check (anon  [...]

### 13. [MINOR] Work budget only guards the AC multiset enumeration, not positional/nested recursion
- **id / feature:** `budget-non-ac-1` (F3-acmatch) -- original severity MINOR
- **location:** rerum/acmatch.py:247-250 (only _match_ac spends budget); _match_positional / non-AC recursion unbudgeted
- **description:** MatchBudget.spend() is called only in `_match_ac`'s element loop. The non-AC positional path (`_match_positional`) and the general recursion never spend budget. The docstring frames the budget as a 'fail-safe work budget for AC enumeration', so this is arguably in-scope, but a pattern whose blowup is in nested non-AC structure (matched under an AC theory, since the engine routes everything through ac_match) is not bounded by the budget and relies solely on Python recursion limits. Not a soundness issue (no spurious yields), and `_match_lhs` catches RecursionError, so it degrades rather than crashes. Flagging as an honesty/coverage gap in what 'bounds completeness' actually covers.
- **suggested fix:** Either document explicitly that the budget bounds only the multiset-assignment fan-out (the combinatorial source) and not depth, or thread a `budget.spend()` check into `_match_positional`/`_ac_match_core` entry so deep non-AC structure under an AC theory is also bounded. Document choice in the MatchBudget docstring.
- **verification:** The finding is accurate on every point. (1) `grep` confirms `budget.spend()` is invoked at exactly ONE site: acmatch.py:247, inside `_match_ac`'s element loop (the AC multiset assignment fan-out). `_match_positional` (lines 255-276) and the positional branch of `_ac_match_core` (line 212) both thread `budget` as a parameter but never call `spend()`, so the non-AC positional/nested recursion path is entirely unbudgeted. (2) Because engine `_match_lhs` (engine.py:1886-1900) routes EVERYTHING through `ac_match` whenever any AC operator is declared, a pattern whose blowup is in nested non-AC structure runs unbounded by the budget and is limited only by Python's recursion limit. (3) The finding c [...]
- **repro:** Probe 1 (deep non-AC nesting under AC theory -- budget never engages):
```
from rerum.acmatch import ac_match, MatchBudget
from rerum.normalize import Theory
theory = Theory({'+': {'ac': True}})
def nest(depth, leaf):
    e = leaf
    for _ in range(depth): e = ['f', e]
    return e
D = 5000
pat  = ['+', nest(D, '?x'), 'b']
expr = ['+', nest(D, 'a'),   'b']
budget = MatchBudget(steps=1000000); start = budget.steps
results = []; rec_err = False
try:
    for r in ac_match(pat, expr, theory, budget=budget): results.append(r)
except RecursionError: rec_err = True
```
OUTPUT:
  nesting depth D = 5000
  RecursionError raised: True
  budget spent: 0
  budget.truncated: False
  num results: 0
=> The [...]

### 14. [MINOR] max_steps parameter is effectively a no-op for the engines completion builds
- **id / feature:** `max-steps-decorative-1` (F5-completion) -- original severity MINOR
- **location:** rerum/completion.py:79-93,129-130 (complete signature and the eng.simplify calls)
- **description:** complete() advertises a `max_steps` reduction budget and threads it into `eng.simplify(cp.left, max_steps=max_steps)`. But the engines complete() builds via RuleEngine.from_rules carry no theory, no conditions, and no hooks, so simplify() takes the cached fast path (the `rewriter()` closure), which IGNORES max_steps entirely and hardcodes its own 1000-iteration cap (rewriter.py:1193). I confirmed this: `eng.simplify(['f','a'], max_steps=1)` still fully reduces. So the knob does nothing in the actual code path -- a caller passing max_steps=50 to bound work, or max_steps=100000 to allow a long join, gets neither. The docstring is honest about this (lines 88-93 explain the 1000 cap is the real bound), and soundness is unaffected because complete and check_confluence both route through the same fast path consistently. But shipping a parameter that silently has no effect is an API-honesty wart: a >1000-step terminating reduction would be truncated to a non-normal form regardless of any max_steps the caller supplies, and the only signal is buried in the docstring.
- **suggested fix:** Either (a) drop max_steps from the public signature for the completion path (it cannot be honored), or (b) make it honored by passing it through a path that respects it (e.g. force the slow path, or have the fast path accept a cap). At minimum, the docstring should state outright that max_steps is currently inert for completion and exists only for forward-compatibility with a future theory-carrying engine.
- **verification:** The finding is accurate on every load-bearing claim.

1. complete() (completion.py:79-80) advertises a `max_steps: int = 1000` reduction budget and threads it into the join test at completion.py:129-130 via `eng.simplify(cp.left, max_steps=max_steps)` / `eng.simplify(cp.right, max_steps=max_steps)`.

2. The engines complete() builds come from RuleEngine.from_rules (engine.py:3604-3606 -> `cls(...).load_rules(...)`), which produces a vanilla engine: no theory, no conditions (rules are `[pattern, [":",x]-skeleton]`, condition=None), no disabled groups, no hooks. So in simplify() (engine.py:2502-2528) all of has_conditions/has_groups/has_ac/hooks_active are False, and the FAST PATH at line 2526 [...]
- **repro:** Probe A (max_steps ignored on the exact from_rules fast path completion uses):
  rules = [[['f', ['?', 'x']], [':', 'x']]]; eng = RuleEngine.from_rules(rules)
  expr = nested (f (f (f (f (f a)))))  # depth 5
  eng.simplify(expr, max_steps=1)  ->  'a'   # fully reduced; max_steps=1 had no effect

Probe B (1000-iter cap is the real bound; max_steps cannot raise it; terminating reduction truncated to non-normal form):
  from rerum.rewriter import ARITHMETIC_PRELUDE
  rules = [ [['c', 0], 'done'],
            [['c', ['?', 'n']], ['c', ['!', '-', [':', 'n'], 1]]] ]
  eng = RuleEngine.from_rules(rules, fold_funcs=ARITHMETIC_PRELUDE)
  eng.simplify(['c', 500],  max_steps=10)     -> 'done'        #  [...]

### 15. [MINOR] exhausted=True over-reported when cap-depth node's successors are all already-seen (false inconclusive)
- **id / feature:** `narrowing-1` (F6-narrowing) -- original severity MINOR
- **location:** rerum/narrowing.py:183-190 (_narrow_with_rules, the depth-cap branch)
- **description:** At the depth cap, the code sets depth_capped=True whenever narrow_step(term) yields ANY successor, without checking whether those successors are already in `seen` (i.e. whether they would have been pruned anyway). When the entire reachable set has already been explored at shallower depth, a cap-depth node can still have successors that loop back to seen states, falsely marking the search as truncated/inconclusive. Repro (confirmed): engine `@ab: a=>b / @ac: a=>c / @bd: b=>d / @cd: c=>d / @da: d=>a`; `narrow(eng, 'a', 'never', max_nodes=100000, max_depth=2)` returns found=False, exhausted=True, nodes=4 -- but {a,b,c,d} is the COMPLETE reachable set and was fully visited (a@0, b@1, c@1, d@2), so the False result is genuinely conclusive and exhausted should be False. The error is in the safe direction (over-cautious): it never reports exhausted=False on a genuinely truncated tree (verified with a growing-term rule across depths 1/3/5), so it cannot cause a false 'no solution exists'. It only makes a conclusive negative look inconclusive.
- **suggested fix:** At the depth cap, only set depth_capped=True if at least one successor's key is NOT already in `seen` (a successor that the search has not and will not otherwise reach). E.g. replace the bare `for _ in narrow_step(...): depth_capped=True; break` with: `for step in narrow_step(term, rules): if _key(step.successor, _compose(step.sigma, theta)) not in seen: depth_capped=True; break`. This mirrors the membership test the in-budget branch already does and makes exhausted honest for cyclic finite trees.
- **verification:** The finding is real and the description is precise. The depth-cap branch in _narrow_with_rules (rerum/narrowing.py:183-190) sets depth_capped=True whenever narrow_step(term) yields ANY successor, with no check against `seen`. The in-budget branch (lines 176-182) only enqueues a successor when its key is NOT already in `seen`, but the cap branch omits this membership test.

I reproduced the exact claimed repro: with rules a=>b, a=>c, b=>d, c=>d, d=>a and narrow(eng,'a','never',max_nodes=100000,max_depth=2), the current code returns found=False, exhausted=True, nodes=4. I then manually traced the BFS and confirmed the analysis: the complete reachable set is exactly {a,b,c,d}, fully visited at  [...]
- **repro:** Repro (current code) -- confirms finding:
$ python3 -c "from rerum.engine import RuleEngine; from rerum.narrowing import narrow; eng=RuleEngine(); eng.load_dsl('@ab: a => b\n@ac: a => c\n@bd: b => d\n@cd: c => d\n@da: d => a'); r=narrow(eng,'a','never',max_nodes=100000,max_depth=2); print(r.found, r.exhausted, r.nodes_expanded)"
-> False True 4

BFS trace (manual) -- proves the negative is conclusive:
visited order: [('a',0),('b',1),('c',1),('d',2)]; reachable set size = 4
CAP node d depth 2 successors (succ, already_seen): [('a', True)]
=> the single cap-successor 'a' is already in `seen`, so nothing was truncated.

Direction check (current code) on genuinely-truncated growing tree (f ?x)=>(f [...]

### 16. [MINOR] narrow() reachability claim ('reduces to a term unifying sigma(target)') only holds as joinability when target has variables
- **id / feature:** `narrowing-2` (F6-narrowing) -- original severity MINOR
- **location:** rerum/narrowing.py:154-203 (_narrow_with_rules / narrow docstrings)
- **description:** The docstrings for `_narrow_with_rules` and `narrow` state the guarantee as 'find sigma such that sigma(start) reduces to a term unifying sigma(target)'. When `target` contains variables, the join step `unify(term, apply_subst(theta, target))` can bind those target variables to NON-normal-form subterms of the narrowed start, so the literal reachability invariant fails. Repro (confirmed): Peano add rules, start=add(s(z),s(z)), target=s(?w); answer sigma={w: add(z,s(z))}. Then simplify(sigma(start))=s(s(z)) but sigma(target)=s(add(z,s(z))), and unify(s(s(z)), s(add(z,s(z)))) is None -- the reduct does NOT unify sigma(target). The answer is still SOUND under joinability: simplify(sigma(start))==simplify(sigma(target))==s(s(z)). So the actual guarantee is joinability (both sides reduce to a common form), which is the meaningful E-unification property, but it is strictly weaker than the stated reachability claim. For ground targets (the common case and all current tests) the reachability claim holds exactly. This is a precision gap in the documented invariant, not an unsoundness.
- **suggested fix:** Tighten the docstrings to state the actual guarantee: either restrict the reachability phrasing to ground/variable-free targets, or state the general guarantee as JOINABILITY -- 'returns sigma such that sigma(start) and sigma(target) reduce to a common form under the engine's reduction' -- matching what `solve_equation`'s sound-answer test already asserts. Optionally add a test with a variable-containing target asserting joinability (not unify-of-reduct) to pin the real contract.
- **verification:** Docstring claims reachability but real guarantee is joinability for variable targets. Sound, ground targets fine.
- **repro:** add(s(z),s(z)) vs s(?w): sigma w=add(z,s(z)); reduct s(s(z)) does not unify s(add(z,s(z))); joinability holds.

### 17. [MINOR] not-analyzed dedups on rule.name
- **id / feature:** `confluence-2` (cross-cutting) -- original severity MINOR
- **location:** confluence.py:315-326
- **description:** distinct anonymous name-None rules collapse to one entry; termination reports each. Repro confirmed.
- **suggested fix:** key dedup on id(rule)
- **verification:** Confirmed by direct reproduction. In confluence.py the inner skip() function (lines 315-319) dedups on key = rule.name and tracks it in seen_skips. Anonymous rules carry name=None (RuleEngine creates RuleMetadata(name=None) for unnamed DSL/JSON rules; engine.py:3949,4021). Conditional rules are not analyzable (is_analyzable returns False when condition is not None, confluence.py:263), so each gets routed to skip(). Two DISTINCT anonymous conditional rules both have key=None, so the second one hits the `if key not in seen_skips` guard and is suppressed: not_analyzed collapses to a single [None] entry instead of two.

The title's second half ("termination reports each") is also accurate and is [...]
- **repro:** Probe (python3 -c):

from rerum.engine import RuleEngine
from rerum.confluence import check_confluence
from rerum.termination import check_termination
eng = RuleEngine()
eng.load_dsl('''
(f ?x) => (g ?x) when (gt ?x 0)
(h ?y) => (k ?y) when (gt ?y 0)
''')
prec = ['f','g','h','k']
conf = check_confluence(eng, precedence=prec)
term = check_termination(eng, prec)
print('CONFLUENCE not_analyzed :', conf.not_analyzed, 'count =', len(conf.not_analyzed))
print('TERMINATION not_analyzed:', term.not_analyzed, 'count =', len(term.not_analyzed))
print('distinct not-analyzable rules:', sum(1 for _,_,m in eng.rule_set() if m.condition is not None))

Output:
CONFLUENCE not_analyzed : [None]    count = 1
T [...]

### 18. [MINOR] apply_once / `once` strategy can skip a productive AC binding (and reports applied on no-change)
- **id / feature:** `apply-once-ac-completeness-2` (engine-integration) -- original severity MINOR
- **location:** rerum/engine.py:2405-2421 (apply_once)
- **description:** apply_once iterates `for bindings in self._match_lhs(...)` and unconditionally `return result, metadata` on the FIRST binding that passes condition + should_fire, even when result == expr (no change). Under an AC theory _match_lhs yields several bindings; if an earlier binding is non-productive (result == expr) it still returns, never trying a later productive binding. In the no-AC path there is only one binding so this is byte-identical to the prior code, but under AC it (a) can miss a valid rewrite and (b) returns (unchanged_expr, metadata) claiming a rule applied. Repro: th=Theory({'+':{'ac':True,'identity':0}}); eng.with_theory(th); eng.load_dsl('(+ ?x ?y) => (+ :x :y)'); apply_once(['+','a','b']) -> (['+','a','b'], <meta>): metadata non-None but result unchanged; a later binding {x:b,y:a} producing (+ b a) is never reached. Note _simplify_exhaustive does NOT have this gap (line 2973 only breaks when new_expr != current), so the default exhaustive driver is correct; the impact is confined to apply_once and the `once` strategy (and to confluence's _is_normal_form, which uses _simp [...]
- **suggested fix:** In apply_once, only return when result != expr (mirroring _simplify_exhaustive's productive-binding break); continue the binding loop on a no-change binding so a later productive AC binding is tried, and do not report metadata for a no-op application.
- **verification:** The finding is accurate on all stated points. Reading engine.py:2405-2421, apply_once iterates `for bindings in self._match_lhs(pattern, expr)` and unconditionally `return result, metadata` on the first binding that passes condition + should_fire, regardless of whether `result == expr`. The `if result != expr:` guard at line 2414 only gates trace/hook emission, NOT the return -- so a non-productive first binding still returns with non-None metadata, and a later productive binding is never reached.

Under an AC theory, _match_lhs (engine.py:1878) delegates to acmatch.ac_match and yields multiple bindings. I verified that for pattern `(+ ?x ?y)` against `(+ a b)` it yields binding[0] `{x:a,y:b} [...]
- **repro:** Primary repro (apply_once false-applied + missed productive binding):

  python3 -c "
  from rerum.engine import RuleEngine
  from rerum.normalize import Theory
  from rerum.rewriter import instantiate
  eng = RuleEngine(); eng.with_theory(Theory({'+': {'ac': True, 'identity': 0}}))
  eng.load_dsl('(+ ?x ?y) => (+ :x :y)')
  expr = ['+', 'a', 'b']
  res, meta = eng.apply_once(expr)
  print('result == expr (no change):', res == expr)        # True
  print('metadata claims applied:', meta is not None)        # True
  pattern, skeleton = eng._rules[0]
  for b in eng._match_lhs(pattern, expr):
      r = instantiate(skeleton, b, eng._fold_funcs)
      print(b.to_dict(), '->', r, 'productive=', r  [...]

### 19. [MINOR] ac_match_truncated is only reset by non-trace simplify, so it is stale after other top-level calls
- **id / feature:** `truncated-flag-not-reset-3` (engine-integration) -- original severity MINOR
- **location:** rerum/engine.py:2500 (only reset site) vs property at 1906-1909; _simplify_with_trace at 3298-3318; apply_once/equivalents/prove_equal/minimize
- **description:** The property docstring says 'True if an AC match hit its budget since the last top-level call', but `self._ac_match_truncated = False` appears exactly once outside __init__, at line 2500 inside the non-trace branch of simplify(). simplify(trace=True), apply_once, equivalents, prove_equal, and minimize all use _match_lhs under AC and can SET the flag, but none RESET it first. So after a truncating call, a subsequent clean reasoning call leaves the flag stuck True. Repro: set_ac_match_budget(1); a truncating simplify -> ac_match_truncated True; then set_ac_match_budget(10**6) and run a clean prove_equal(['+','a','b'],['+','b','a']) -> ac_match_truncated is still True (stale). Same for simplify(trace=True). The truncation completeness signal is therefore unreliable for every entry point except non-trace simplify.
- **suggested fix:** Reset self._ac_match_truncated = False at the top of every top-level entry point that can run AC matching (apply_once when _top_level, equivalents, prove_equal, minimize, and _simplify_with_trace), or move the reset into a shared _begin_top_level() helper that all these methods call.
- **verification:** stale flag
- **repro:** reproduced

### 20. [MINOR] RecursionError in ac_match is swallowed without setting the truncation flag
- **id / feature:** `recursionerror-silent-truncation-4` (engine-integration) -- original severity MINOR
- **location:** rerum/engine.py:1893-1900 (_match_lhs AC branch)
- **description:** The AC branch of _match_lhs catches RecursionError and silently returns (lines 1896-1897). Only budget.truncated (set when the MatchBudget runs out) flips self._ac_match_truncated in the finally block. A deep/pathological pattern that blows the recursion limit therefore loses completeness (some matches never yielded) with NO signal in ac_match_truncated, which is the documented 'completeness was bounded' indicator. This is inconsistent with the budget path, which does record truncation.
- **suggested fix:** Set self._ac_match_truncated = True in the `except RecursionError` handler (or in the finally when a RecursionError was caught) so that any incomplete AC enumeration -- budget OR recursion -- is reflected by the public flag.
- **verification:** The code path exists exactly as described at rerum/engine.py:1893-1900. _match_lhs's AC branch wraps ac_match in try/except RecursionError: ... return, and the finally clause flips self._ac_match_truncated only when budget.truncated is set. RecursionError is a distinct way to lose AC-enumeration completeness and is NOT recorded.

I verified all three legs of the claim with concrete probes:
1. ac_match itself raises RecursionError on a deeply nested pattern/expression, and it does so BEFORE the MatchBudget is exhausted (budget.steps=10000 but recursionlimit=1000 hits first), yielding 0 results with budget.truncated == False. So the budget mechanism does not subsume the recursion case.
2. Driv [...]
- **repro:** Probe 1 (ac_match raises before budget exhaustion):
$ python3 -c "...theory=Theory.from_dict({'+':{'ac':True,'identity':0}}); pat=exp=nest(5000,'z'); budget=MatchBudget(steps=10000); try: list(ac_match(pat,exp,theory,budget=budget)) except RecursionError: ... ; print(budget.truncated)"
Output:
  recursionlimit = 1000
  RecursionError raised by ac_match directly? True
  results yielded: 0
  budget.truncated: False

Probe 2 (the seam under review swallows it and leaves the flag False):
$ python3 -c "eng=RuleEngine().with_theory(Theory.from_dict({'+':{'ac':True,'identity':0}})); pat=exp=nest(5000,'z'); eng._ac_match_truncated=False; r=list(eng._match_lhs(pat,exp)); print(len(r), eng.ac_match_tr [...]

---

## NITs (not independently verified -- cosmetic / low-stakes)

- **F1-normalize / `silent-fixpoint-cap-5`** -- Fixpoint loop silently returns after 1000 iterations without signaling non-convergence (rerum/normalize.py:390-395 (normalize))
- **F2-confluence / `deadcode-1`** -- Documented-dead defense-in-depth branch can double-record a rule as both analyzed and not_analyzed (rerum/confluence.py:critical_pairs (lines 344-352, the UnsupportedPattern except branch))
- **F4-termination / `orient-docstring-1`** -- Sentence fragment in orient() docstring (rerum/termination.py:98-99 (orient docstring))
- **F4-termination / `precedence-dup-precondition-2`** -- Duplicate-free precedence is a documented but unenforced precondition (rerum/termination.py:26-35 (Precedence type alias and _prec_gt))
- **F5-completion / `redundant-dedup-1`** -- Double deduplication of new rules (rerum/completion.py:147 and 152)
- **F5-completion / `alpha-variant-convergence-1`** -- Structural dedup does not catch alpha-variant (variable-renamed) rules (rerum/completion.py:37-44 (_dedup) and 147-152)
- **F3-acmatch / `dead-fallthrough-1`** -- Unreachable fall-through return in _match_one (rerum/acmatch.py:147 (_match_one final `return None`))
- **engine-integration / `mcp-theory-bypasses-invalidation-5`** -- MCP load_theory sets the _theory slot directly, bypassing with_theory's cache invalidation (rerum/mcp/persistence.py:137 (engine._theory = theory); contrast engine.with_theory at rerum/engine.py:1842-1856)
- **cross-cutting / `init-1`** -- DirectedRule not re-exported though critical_pairs is (__init__.py)

## Per-target summaries

### F1-normalize -- F1 theory-normalized reasoning
The central claim largely holds: for well-formed numeric/boolean theories over integer, symbolic, and bool operands, normalize(flatten->sort->collect->fold) is idempotent and confluent, the engine names no operator (every operator, unit, and repeat rule is read from the Theory DATA -- a token-stripped scan of normalize.py finds zero operator literals used as code), and the _canonicalize seam is the strict identity (returns `is expr`) when no theory is set. I confirmed idempotence/confluence across permutations, reassociation, coefficient/power collection, nested collection, boolean theories, and self-referential repeat rules; I also confirmed normalize does not mutate its input. However, the "always comparable, idempotent, confluent" promise is unconditional in the docstring and breaks on two reachable inputs: (a) distinct exact Fractions whose float() projections collide get treated as like terms and one is silently DISCARDED -- a true soundness/value-corruption bug (a*b*x became a^2*x); and (b) a NaN operand makes ORDER_KEY non-total, so canonical_sort of an AC sum yields different orders per permutation (non-confluent). Separately, Theory.has_ac()/is_ac() crash with an unmapped AttributeError on a malformed (non-dict) signature entry that Theory.from_json accepts without validation, which defeats the MCP load_theory parse_error mapping. Two further low-severity items: a repe [...]

### F2-confluence -- F2 confluence/critical-pairs
The central correctness claim holds: in the production path, check_confluence never reports a false "locally confluent". unify is a sound first-order syntactic mgu with occurs-check; critical_pairs computes correct superpositions with proper rename-apart (verified on overlap, self-overlap, non-left-linear, collapse-variable, and constant-in-LHS cases), excludes the trivial root self-overlap, and is_analyzable recursively refuses every non-first-order pattern/skeleton/conditional rule before it can reach unify -- so the documented unify refuse-first gap and other issues below cannot turn into a false confluence verdict. Joinability uses the engine's real reduction relation and canonical-equality-first comparison, both directions sound; over-budget/cyclic/RecursionError legs map to "unknown" (never counted joinable), and the Newman/precedence path's confluent=True is gated by terminating, which (sharing the same is_analyzable) is False whenever any rule was skipped, keeping global confluent=True reachable only with zero not-analyzed rules. The real defects are robustness/honesty, not soundness: (1) check_confluence's documented max_steps work budget is silently ignored on the common no-theory/unconditional fast path (the engine's rewriter() uses a hardcoded 1000-iteration cap), so a tight caller budget does not bound work and a growth rule can do far more reduction (even hitting R [...]

### F4-termination -- F4 termination via LPO
The central correctness claim HOLDS. I read rerum/termination.py in full, its paired test file, and the reused primitives in confluence.py and rewriter.py. I then verified the reduction-order properties empirically: over 50-60 random terms (with both total and partial precedences, including a precedence listing only one symbol so most operators are mutually incomparable) lpo_greater was irreflexive, asymmetric, transitive, substitution-stable (s>t ? sigmas>sigmat) and context-closed (s>t ? C[s]>C[t]) with zero violations. Variable handling correctly enforces Var(t) subset Var(s) at every recursion depth (a buried fresh RHS variable is rejected), so no rule with a fresh RHS variable is ever oriented. The variadic-operator hazard is correctly neutralized: the lexicographic case (case 3) is guarded by len(sargs)==len(targs), so LPO never applies lex status across differing arities -- the one place naive LPO becomes unsound for rerum's n-ary operators. orient returns "lr"/"rl" only when the corresponding direction strictly decreases under the (proven) reduction order, and a bidirectional/commutativity axiom is correctly refused (the -rev leg stays unoriented, forcing terminating=False). check_termination's verdict terminating=(not unoriented)and(not not_analyzed) is sound given the reduction-order guarantee. The Newman wiring in confluence.check_confluence sets confluent=False on a genuine non_joi [...]

### F5-completion -- F5 Knuth-Bendix completion
The central correctness claim holds. `complete` never returns status=="complete" for a system that is not actually confluent+terminating. I verified this empirically: every "complete" result I could construct was independently confirmed confluent+terminating by F2's `check_confluence` using the same budget. The soundness chain is sound on every limb: (1) the syntactic `s == t` join test coincides exactly with `check_confluence`'s join test because both no-theory engines route through the identical fast path where `_canonicalize` is the identity; (2) LPO orientation (`orient`) provably refuses any rule whose RHS contains a variable not dominated by the LHS (confirmed: `orient(f(x), g(y))` returns None -> "failed"), so completion can never introduce an extra-variable or variable-LHS rule, and a single fixed precedence makes the union of LPO-oriented rules terminating (Newman applies); (3) when a critical pair fails to syntactically join, completion ALWAYS orients-and-adds (or fails) and never silently treats a non-join as joined, so it is conservative -- it can only over-add (yielding max_iterations) or fail, never a false complete; (4) the RecursionError handler is genuinely reached for deeply-nested terms (it fires inside eng.simplify's `_expr_key` hashing) and maps to "not joined", which is conservative; (5) max_iterations=0 and trivial/empty inputs all return honest verdicts ( [...]

### F3-acmatch -- F3 AC-matching
The AC-matcher's core soundness machinery is largely correct: every yield reconstructs an AC-equal subject, the `?free` post-pass validates against the COMPLETE binding (more correctly than the rewriter's structural pass, since it checks the actual bound value rather than the unreordered exp position), budget truncation bounds completeness only (verified sound at every budget level 0..7), and binding dedup neither over- nor under-merges. HOWEVER, two real defects break the stated claim. (1) SOUNDNESS BUG: typed rest patterns (`["?...","name","const"]` / `"var"`) silently drop their type constraint everywhere in acmatch.py (the bare-rest path, `_match_positional`, and the AC-node rest in `_match_ac`), so `ac_match` yields bindings that violate the pattern's own constraint -- e.g. `(+ ?x ?rest:const...)` fires on `(+ 1 a b)` binding `rest=[a,b]` though a,b are non-constant. `match()` correctly rejects this, so the "byte-identical to match()" sub-claim also fails. (2) CRASH BUG: an empty-list pattern `[]` (reachable as a nested sub-pattern like `(f ())` ) makes `_ac_match_core` call `arbitrary_constant([])` -> `car([])`, raising an uncaught `ValueError`; under an AC theory the engine routes all matching through `ac_match` and catches only `RecursionError`, so the whole `simplify` call crashes where the no-theory path succeeds. Both are confirmed end-to-end through the engine and ar [...]

### F6-narrowing -- F6 narrowing
The central correctness claim holds: every answer returned by `narrow` and `solve_equation` re-derives/joins under `engine.simplify`. I confirmed this across 500+ randomized stress trials (append-prefix E-unification and Peano addend-solving) plus targeted adversarial probes, all sound. The four soundness mechanisms named in the brief are all correct and verified empirically: (a) the `Var(r)  subset  Var(l)` filter in `_extract_rules` correctly drops dangling-`:x` rules whose `instantiate_skeleton` model (free var) diverges from the engine's real reduction (ground symbol); (b) `solve_equation`'s `eq`/`true` are gensym'd against all rule symbols and remain fresh even when rules literally use operators named `eq`/`true`; (c) per-step `rename_apart` in `narrow_step` prevents capture, including when the user equation and the injected reflexivity rule share variable `?x`; (d) `_compose(s2, s1) = s2?s1` is associative and applies `s1` first. No domain operator is hardcoded as code -- the general-engine principle is respected. I found two honesty/precision gaps, both in the SAFE direction (neither can yield a false "no solution" nor an unsound answer): a depth-cap `exhausted` over-report when the finite reachable set is actually fully explored, and an imprecise reachability claim in `narrow`'s docstring that only holds as joinability when the target contains variables.

### engine-integration -- engine seams for F1-F6
The central correctness claim holds. The no-theory / no-AC paths are byte-identical to pre-feature behavior: _match_lhs uses `is not None` (correct, since match() returns Bindings|None and Bindings is always truthy) and yields exactly match()'s single result; the three rewired drivers (_simplify_exhaustive, apply_once, _all_single_rewrites) preserve original control flow on a single yield; the fast-path gate (has_conditions or has_groups or has_ac, plus hooks) routes correctly and is robust even against the MCP layer's direct `_theory` slot assignment that bypasses cache invalidation (the gate re-checks has_ac independently, verified empirically). normalize is idempotent (fuzzed 2000 terms), so the AC canonicalize-at-top in _simplify_exhaustive is sound and terminates via the canonical visited set. Lazy imports correctly break a real cycle (completion.py imports engine at top level); `import rerum`, complete(), check_confluence(), check_termination() all load and run cleanly. The findings are NOT soundness breaks of the no-theory path; they are AC-only completeness/honesty gaps: (1) apply_once / the `once` strategy can skip a productive AC binding when an earlier non-productive binding of the same rule short-circuits (the more-important exhaustive driver does NOT have this gap, as it only breaks on a productive binding); (2) bottomup/topdown strategies silently ignore an AC the [...]

### cross-cutting -- cross-module consistency
s

---

## Remediation (2026-06-23)

All 20 confirmed MAJOR+MINOR findings were fixed under
`docs/superpowers/plans/2026-06-23-trs-frontier-review-remediation.md` (15 tasks,
TDD, subagent-driven). Full suite 1742 passed; no-domain guard 12; ASCII clean.
An Opus holistic review of the remediation returned APPROVE-WITH-NITS; its one
MINOR (an ORDER_KEY value-ordering regression) and two doc nits were then fixed.

Key fix commits: 23e3923 (ORDER_KEY exact key), 846764e (ORDER_KEY value-order
+ doc nits), 04e48ce/82bdea4 (Theory validation), 1346a5f (max_steps fast path),
6b87eb2/b434836 (acmatch empty-pat/typed-rest), ed9655b (confluence dangling-RHS),
91130a1 (completion invariant), c14812f/d986cd9 (unify/not_analyzed), 823876a
(strategy AC refusal), e2052b4 (apply_once productive-only), b8620f5 (truncation
flag), aeaa9a1 (narrowing exhausted). The 9 NITs remain deferred.
