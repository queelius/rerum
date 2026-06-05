# RERUM as a Traceable General-Purpose Rewriting Engine

**Date:** 2026-06-04 (revised 2026-06-05)
**Status:** Design approved, revised for the general-engine principle
**Scope:** One unified vision spec for the general engine. Implementation is
phased (plans per phase). Calculus appears throughout ONLY as a worked example
that exercises the general machinery; it is never engine code.

## 0. The General-Engine Principle (the hard constraint)

RERUM is a general term-rewriting engine. It rewrites s-expression terms by
matching patterns and applying rules. That is the whole of what the engine
knows.

No particular domain is ever hardcoded in the engine. Differentiation,
integration, limits, boolean algebra, lambda calculus, logic, type checking,
peephole compiler optimization: every one of these is expressed entirely as
DATA that the engine consumes:

- **Rules** (`=>` and `<=>`) carrying metadata, examples, categories, reasoning.
- **Theories**: declarations of which operators are associative/commutative and
  their identity/annihilator elements (data, not code; see Section 5.2).
- **Preludes**: bundles of fold functions (the existing extension point for
  computable operations like `+`, `sin`, `const?`). Preludes are general; they
  are named by what they compute (arithmetic, math, predicates), never by a
  domain.
- **Domain configs**: when a workflow needs a driver (which search goal) or a
  checker (how to numerically validate a result), those are supplied by the
  caller as data/callables, not baked into the engine.

The test for every line of `rerum/` core code: if you mentally swap "calculus"
for "boolean algebra," does the engine code change? If yes, the design is wrong.
The engine must not contain the words `dd`, `int`, `lim`, `and`, `or`, or any
other domain operator as a special case. Those are operator symbols that appear
only in rule files under `examples/`.

Consequences enforced by this spec:

- Every calculus artifact (rules, theory declaration, domain checker) lives
  under `examples/`, shipped as a demonstration, exactly like the existing
  `examples/algebra.rules`. None of it is importable engine capability.
- Engine modules expose GENERAL machinery parameterized by data: normalization
  takes a theory; search takes a goal predicate; the numeric evaluator takes a
  prelude; the corpus generator takes a driver and a checker.
- Where the previous draft named things by domain (a "CALCULUS_PRELUDE", a
  "verify_derivative" in core), this revision removes the domain name from the
  engine and pushes the domain content out to `examples/`.

## 1. Motivation

RERUM rewrites terms deterministically against a rule library. Three general
limitations block it from being a first-class reasoning tool and a source of
high-quality "show your work" training data:

1. **Traces lose the reasoning.** A `RewriteStep` keeps only
   `(rule_index, metadata, before, after)`, where `before`/`after` are the
   local subtree, not the whole expression. The matching `bindings` (the
   substitution that justifies a step) and the redex `path` are computed and
   discarded. `prove_equal`/`equivalents`/`minimize` return bare expression
   chains with no rule labels, so the rule sequence that proves an equality is
   unrecoverable.

2. **Simplification is weak.** Matching is strictly positional (verified:
   `(+ a ?y)` does not match `(+ b a)`), and there is no normal form, so any
   rule set over commutative operators must spell out every ordering by hand.

3. **No goal-directed search and no agent surface.** Directed rewriting
   (`simplify`) cannot back out of a wrong move, so non-confluent rule sets
   cannot be solved; and there is no structured, NL-explainable interface for an
   LLM agent to drive the engine.

All three are GENERAL deficiencies of the engine. Fixing them makes every rule
set better, calculus included. Calculus is simply the worked example this spec
uses to demonstrate and test the fixes, because it produces rich, verifiable,
interpretable derivations.

## 2. Goals and Non-Goals

**Goals (all general engine capabilities)**

- A trace model where each step is self-contained (rule identity, direction,
  bindings, redex path, guard result, rationale) and the whole-expression
  derivation is reconstructible; labeled paths for `prove_equal`/`equivalents`/
  `minimize`.
- Theory-driven canonical normalization (associative/commutative flattening,
  ordering, like-term collection) parameterized by a declared theory.
- Goal-directed search (`solve`) as an escalation above directed rewriting,
  with the goal predicate supplied by the caller.
- General supporting primitives: fresh-variable generation, exact rationals, a
  numeric evaluator over a prelude, a `free-of?` predicate, two engine bug
  fixes.
- A trace-to-text and trace-to-record layer (NL prose and structured JSONL)
  that operates on any derivation, domain-agnostic.
- An MCP server that exposes the general engine to LLM agents (authoring,
  applying, proving, explaining, goal-solving).

**Non-Goals**

- Any domain capability hardcoded in `rerum/` core. Calculus, boolean algebra,
  logic, etc. are example rule sets, never engine code (Section 0).
- A general CAS (factorization over fields, Groebner bases).
- Performance tuning beyond budgeted termination.
- A GUI. Files (rules, JSONL, prose) and the Python API and the MCP are the
  surfaces; humans use the existing CLI.

## 3. Architecture: engine vs content

Two strictly separated parts.

```
============================  GENERAL ENGINE (rerum/)  ============================
  trace.py        situated steps, labeled paths, global reconstruction, to_prose
  normalize.py    theory-driven AC normalization machinery (takes a Theory)
  solve.py        goal-directed search (takes a goal predicate)
  rewriter.py     match/instantiate (+ fresh vars, rationals, free-of?, bug fixes)
  numeval.py      numeric evaluation of a ground term under a prelude
  training.py     trace -> JSONL record and trace -> prose (domain-agnostic)
  engine.py       RuleEngine: load/apply/prove/minimize/solve, theory wiring
  mcp/            agent-facing server exposing the above (general)
=================================  DOMAIN CONTENT (examples/)  ====================
  algebra.rules            (exists)        boolean.rules (illustrative)
  differentiation.rules    + theory + checker, as a worked example
  integration.rules        limits.rules
  *.theory.json            operator signatures (which ops are AC, identities)
  *_checker.py             numeric domain validators built on numeval (data/config)
==================================================================================
```

Everything above the line is domain-agnostic and is the engineering work.
Everything below the line is data and demonstration. A new domain (say, boolean
algebra) is added by writing files below the line and changing nothing above it.

The single-step rewrite remains the first-class unit. `_all_single_rewrites`
already generates whole-expression neighbors carrying rule, bindings, and redex
position; keeping those labels is what powers both rich tracing and search.

## 4. Directed-first, search-as-escalation

A rule set is solved by directed rewriting (`simplify`, the existing fixpoint
driver) when it is confluent: applying matching rules in any order reaches the
same result, so a greedy driver never needs to reconsider. Search is needed only
for non-confluent rule sets, where solving requires trying a move and backing
out of dead ends.

The engine therefore offers two general drivers:

- `simplify(expr, ...)`: greedy fixpoint (exists). The default. Handles every
  confluent rule set.
- `solve(expr, goal_predicate, ...)`: best-first search over the rewrite graph,
  stopping when the goal predicate holds, budgeted, with backtracking. The
  escalation for non-confluent rule sets.

Neither driver knows any domain. `solve`'s goal is a caller-supplied predicate
(for the calculus example: "no `int`/`lim` operator remains"; for a boolean
example it might be "is a literal" or "is in CNF").

This is also exactly how the MCP agentic loop behaves: try `simplify`, and when
the engine is stuck, escalate (to `solve`, or to asking the agent for a rule).

## 5. General engine components

### 5.1 Trace foundation (`rerum/trace.py`)

`RewriteStep` gains keyword fields (additive; existing `before`/`after` retained
and aliasing the redex-local edit): `rule_id` (stable: name, else content hash
of pattern+skeleton), `direction` ('fwd'/'rev'/None), `bindings`
(`Bindings.to_dict()` form), `path` (child-index path to the redex in the root),
`kind` ('rule' | 'normalize' | 'fold'), `guard` (the instantiated condition and
its result, or None), `rationale` (`metadata.reasoning`/`category`).

`RewriteStep.to_dict()` emits all fields. trace.py adds pure helpers
`rule_identity(metadata, pattern, skeleton)` and `splice_at(root, path,
subtree)`, plus `RewriteTrace.to_global_sequence()` (replay from `initial`,
splicing each step's `after` at `path`, yielding whole-expression
`before_root`/`after_root` per step) and `to_dict(global_sequence=False)`.

Strategy drivers thread an accumulating `path`; `HookContext.expr_path` (today
always `[]`) is populated from it. `_all_single_rewrites` returns labeled edges;
`prove_equal`/`equivalents`/`minimize` carry the label on parent pointers;
`EqualityProof.path_a`/`path_b` become `List[RewriteStep]`; `OptimizationResult`
gains `.derivation`.

All of this is domain-agnostic: a step records what rule fired where with what
bindings, for any rule.

### 5.2 Theory-driven normalization (`rerum/normalize.py`)

This is the subtle place the general principle bites. Flattening `(+ (+ a b) c)`
to `(+ a b c)` and sorting commutative operands requires knowing that `+` is
associative and commutative with identity `0`. That knowledge must be DATA, not
hardcoded.

Introduce a `Theory` (operator signature), declared as data:

```
Theory({
  "+": {"ac": true, "identity": 0},
  "*": {"ac": true, "identity": 1, "annihilator": 0},
})
```

The engine ships NO built-in theory naming `+`/`*`. A theory is loaded from a
`*.theory.json` (data, under `examples/` for the calculus/algebra demo) or
constructed by the caller. `normalize.py` is parameterized by it:

```
flatten(expr, theory) -> ExprType
ORDER_KEY(expr) -> tuple                      # total order, structural, domain-free
canonical_sort(expr, theory) -> ExprType      # sorts operands of ac operators
collect_like_terms(expr, theory) -> ExprType  # uses identities from the theory
normalize(expr, theory, *, listener=None) -> ExprType   # to fixpoint; emits kind="normalize" steps
```

`ORDER_KEY` is a structural total order (numbers, then symbols, then compounds
by head and recursively by args); it embeds no domain knowledge. Like-term
collection (`x + x -> (* 2 x)`) is expressed in terms of the theory's identities
and the two ac operators it names, so it works for any AC pair, not just
arithmetic. An empty theory makes `normalize` the identity function.

`RuleEngine` optionally holds a theory (`with_theory(theory)`); `simplify` and
`solve` normalize between steps when a theory is set, and not at all when it is
not. No theory is the default.

### 5.3 Goal-directed search (`rerum/solve.py`)

```
class SolveResult: solution; derivation: RewriteTrace; explored: int; found: bool
solve(engine, expr, goal_predicate, *, cost_fn=expr_size, max_nodes=10000,
      fresh_vars=True, normalize_between=True) -> SolveResult
contains_op(expr, ops: set) -> bool           # a convenience predicate builder, domain-free
```

Best-first over labeled single-step rewrites (Section 5.1), stop when
`goal_predicate(node)` holds, budget `max_nodes`, fire `max_depth` on
exhaustion and return `found=False` (never a partial result). Engine wrapper
`RuleEngine.solve(expr, goal_predicate, **kw)`. `contains_op` is a generic
helper for building "no operator X remains" goals; it is not tied to any
operator.

### 5.4 Fresh variables (`rerum/rewriter.py`)

A skeleton form `["fresh", base]` resolves during `instantiate` to the smallest
of `base, base+"1", ...` not free in the whole expression being built
(deterministic). Helpers `gensym(base, avoid)` and `free_symbols(expr)`. General:
any rule set whose rewrites must introduce a new symbol (substitution-style
rules in any domain) can use it.

### 5.5 Exact rationals (`rerum/rewriter.py`, `rerum/expr.py`)

A central `coerce_number(x)` normalizes int/float/`fractions.Fraction`
(Fraction with denominator 1 collapses to int; a Fraction is never silently
floated). `safe_div`/`nary_fold` return exact `Fraction` for non-integral exact
integer results; `format_sexpr(Fraction(p,q))` renders `["/", p, q]`. General
numeric capability; no domain knowledge.

### 5.6 Numeric evaluator (`rerum/numeval.py`)

```
numeval(expr, env: dict, prelude) -> number        # evaluate a ground term
numeric_equiv(a, b, sampler, prelude, *, samples=8, tol=1e-6) -> bool
```

`numeval` interprets a variable-free (after `env` substitution) term using the
fold functions in `prelude`. `numeric_equiv` samples variable assignments and
checks two expressions evaluate equal. Both are GENERAL: they validate that a
rewrite or a claimed equality is numerically sound for any expressions over a
prelude. Domain-specific validators (for example, "is this the derivative of
that?") are NOT here; they are domain content (Section 6.3) that calls these
primitives.

### 5.7 Predicates, prelude bundles, bug fixes (`rerum/rewriter.py`, `engine.py`)

- `free-of?` fold predicate: `(! free-of? f v)` true iff symbol `v` does not
  occur in `f`. Added to `PREDICATE_PRELUDE`. General.
- Fix the `?x:free(v)` binding-order bug: evaluate the free-of check against the
  final resolved bindings. General matcher correctness.
- Fix the guard-on-undefined-op footgun: a guard that does not fully fold (its
  head is an undefined op) raises, rather than evaluating truthy. General guard
  correctness.
- Prelude bundles are named by computation, never by domain. The existing
  `MATH_PRELUDE` and `PREDICATE_PRELUDE` already follow this. A rule set that
  needs both simply documents that it requires `{**MATH_PRELUDE,
  **PREDICATE_PRELUDE}`; the engine provides a helper to combine preludes
  (`combine_preludes(*ps)`), but ships no domain-named bundle. (The previous
  draft's `CALCULUS_PRELUDE` is removed.)

### 5.8 Trace-to-text and trace-to-record (`rerum/training.py`)

```
to_training_record(trace, *, problem, operator, answer, verified=None) -> dict
to_prose(trace) -> str
generate_corpus(engine, problems, *, driver, checker=None) -> Iterator[dict]
```

`to_training_record` and `to_prose` operate on a `RewriteTrace` and know nothing
about any domain: a step renders from its `kind`, `rule_id`, `rationale`, and
the global before/after. `to_prose` is a deterministic projection of the
structured trace (per-`kind` templates plus `rationale`), so prose and record
cannot drift.

`generate_corpus` is parameterized: `driver` is a callable
`(engine, problem) -> (answer, trace)` (for the calculus demo, a small adapter
that runs `simplify` for `dd` and `solve` for `int`/`lim`), and `checker` is an
optional callable `(problem, answer) -> bool` (the domain validator). The engine
supplies the corpus MACHINERY; the domain supplies the driver and checker as
data. No operator names appear in `training.py`.

### 5.9 MCP server (`rerum/mcp/`)

The agent-facing surface, general. It exposes authoring (load/add/list/get
rules, with metadata and example validation), applying (`simplify`, `apply_once`,
`equivalents`, `prove_equal`, `minimize`), goal-solving (`solve` with a
caller-described goal), explaining (the situated trace plus a `prose` rendering
via `to_prose`), and an optional agentic loop (when the engine is stuck, request
a rule from the connected LLM via MCP sampling, validate, install, retry). It
loads rule sets and theories as data; it contains no domain logic. Detailed tool
surface and lifecycle live in the companion MCP design doc
(`docs/superpowers/specs/2026-05-04-mcp-design.md`), reconciled to this spec's
trace shape, the `simplify`-vs-`solve` naming, and the addition of a prose
rendering (the earlier MCP non-goal on NL explanation is reversed: agents want
`to_prose` output to relay to users).

## 6. Worked example: calculus as pure content

This section demonstrates the general engine. Nothing here is engine code; every
artifact ships under `examples/`.

### 6.1 It is rule sets plus a theory plus a prelude requirement

- `examples/differentiation.rules`: the `dd` operator's rules. Differentiation is
  confluent, so it runs on the existing `simplify` driver with no search. This
  is the proof that a whole domain can be "just a rule set."
- `examples/integration.rules`, `examples/limits.rules`: the `int`/`lim`
  operators' rules. The easy cases (linearity, power rule, table forms, direct
  substitution, L'Hopital, known limits) are directed and also run on `simplify`.
  Only the genuinely non-confluent cases (u-substitution, integration by parts,
  non-obvious algebraic limit manipulation) escalate to `solve` with the goal
  "no `int`/`lim` remains".
- `examples/arithmetic.theory.json`: declares `+` and `*` as AC with their
  identities, so `normalize` cleans up derivative output. This is the data that
  Section 5.2's machinery consumes; the same machinery serves a
  `boolean.theory.json` declaring `and`/`or` as AC.
- The rule files document that they require `{**MATH_PRELUDE,
  **PREDICATE_PRELUDE}` (combined via `combine_preludes`), plus `free-of?`.

Each rule carries `examples` metadata. Because the DSL annotation grammar only
supports `{category=...}`, examples are carried in a `*.metadata.json` sidecar
merged via `load_metadata_json` (the v0.7 layer), validated at load.

### 6.2 Differentiation needs zero engine code

The differentiation rule set covers constants/variables, linearity, product,
quotient, power (constant exponent), general power via logarithmic
differentiation, exp/log, trig, inverse trig, hyperbolic, and partials (via the
general `free-of?` predicate). Loaded alongside `examples/algebra.rules` under a
combined prelude with the arithmetic theory set, `simplify` produces clean
results (for example `d/dx(x*x)` reduces to `(* 2 x)`). No `solve`, no
domain engine code.

### 6.3 Domain validators are content, not core

`examples/calculus_checker.py` provides `is_derivative(expr, var, result)`,
`is_integral(integrand, var, result)`, `is_limit(...)` built ON TOP of the
general `numeval`/`numeric_equiv` primitives (Section 5.6). These encode the
domain semantics (a derivative result must match the finite-difference of the
input). They are passed to `generate_corpus` as the `checker`. They are example
files; the engine never imports them.

### 6.4 What the example demonstrates about the engine

That the trace foundation, theory-driven normalization, goal-directed search,
fresh variables, rationals, and the corpus/prose layer are sufficient, with no
domain code, to take a hard symbolic problem to a clean, verified, fully
explained derivation. Swapping in `boolean.rules` + `boolean.theory.json` would
exercise the same machinery to put expressions in CNF, with no engine change.

## 7. Module and file plan

General engine (new/extended under `rerum/`):

- `rerum/normalize.py` (new): theory-driven normalization.
- `rerum/solve.py` (new): goal-directed search.
- `rerum/numeval.py` (new): numeric evaluation and equivalence.
- `rerum/training.py` (new): trace -> record and trace -> prose, parameterized.
- `rerum/mcp/` (new): agent server (see companion MCP doc).
- `rerum/trace.py` (extend): situated steps, global reconstruction.
- `rerum/engine.py` (extend): path threading, labeled edges, theory wiring,
  `solve` wrapper, prelude combination, bug fixes.
- `rerum/rewriter.py` (extend): `free-of?`, `fresh`, rationals, `?x:free(v)` fix.
- `rerum/expr.py` (extend): Fraction parse/format.

Domain content (new under `examples/`, demonstration only):

- `examples/differentiation.rules`, `examples/integration.rules`,
  `examples/limits.rules` and their `*.metadata.json` sidecars.
- `examples/arithmetic.theory.json` (and, illustratively, the shape a
  `boolean.theory.json` would take).
- `examples/calculus_checker.py` (domain validators built on `numeval`).

Tests (one file per area): `test_trace_situated.py`, `test_normalize.py`,
`test_solve.py`, `test_numeval.py`, `test_rationals.py`, `test_free_of.py`,
`test_training.py`, plus example-exercising tests `test_differentiation.py`,
`test_integration.py`, `test_limits.py` that load the example files and assert
behavior (these test the ENGINE through the example content, not engine-resident
domain logic).

## 8. Testing strategy

- General machinery is unit-tested without any domain: normalization idempotence
  and confluence over a toy theory; search termination and budget over a toy
  rule set; fresh-var determinism; rational exactness; `numeval`/`numeric_equiv`;
  trace global-sequence round-trip; labeled proof paths.
- The calculus example files are loaded by `test_differentiation.py` etc. to
  show the engine handles a real domain end to end, including numeric checking
  via the example checker. These tests would be deleted or swapped if the
  example changed, and the engine would not.
- Property checks: for the example, differentiate-then-numeric-check and
  integrate-then-differentiate-back, all via the general `numeval`.

## 9. Risks and mitigations

- **Leaking a domain into core.** The standing risk this revision exists to
  prevent. Mitigation: the Section 0 swap test applied in review of every engine
  change; no operator symbol literals in `rerum/` except the existing
  arithmetic fold builders (which are prelude content, themselves general).
- **Theory expressiveness.** AC plus identity/annihilator covers arithmetic and
  boolean algebra; richer theories (distributivity as a normalizer) are out of
  scope and remain ordinary rules. Mitigation: keep the theory minimal; anything
  it cannot express stays a rule.
- **Search blowup.** Strict node budgets, cost-guided ordering, honest failure.
- **n-ary representation ripple.** Flattening changes shapes binary rules
  assumed; the example rule sets use rest-patterns. Normalization confluence is
  tested before the example rule sets are written.
- **Scope.** One vision; strictly phased; each phase independently shippable.

## 10. Implementation phasing

Engine phases (general; the actual engineering):

0. **Foundation fixes**: `?x:free(v)` bug, guard-on-undefined-op, `free-of?`
   predicate, `combine_preludes` helper. (No domain bundle.)
1. **Trace foundation**: situated steps, path threading, labeled search paths,
   global reconstruction.
2. **Theory-driven normalization**: `normalize.py` parameterized by a `Theory`.
3. **Search, fresh vars, rationals, numeval**: `solve.py`, `["fresh", base]`,
   `Fraction`, `numeval.py`.
4. **Trace-to-text/record**: `training.py` (prose + JSONL + parameterized
   corpus). [Was Phase 7; promoted because it is general and unblocks the agent
   surface.]
5. **MCP server**: agent surface over the above, on today's plus Phases 0 to 4
   capabilities. (Sequenced early per the agent-tool priority.)

Domain demonstration phases (content under `examples/`; exercise the engine):

D1. **Differentiation example**: `differentiation.rules` + `arithmetic.theory`,
    runs on `simplify`. Proves "a domain is just a rule set."
D2. **Integration and limits example**: `integration.rules`, `limits.rules`,
    escalating to `solve`; `calculus_checker.py` on `numeval`.

The domain phases can land any time after the general capabilities they use
(D1 after Phases 1 to 2; D2 after Phase 3). They add no engine code.

## 11. Success criteria

- No `rerum/` core module references any domain operator (`dd`/`int`/`lim`/
  `and`/`or`) as a special case; the Section 0 swap test passes by inspection.
- A confluent rule set (differentiation) is solved end to end by `simplify`
  with a reconstructible, labeled, prose-explainable derivation, using only
  example content plus the general engine.
- A non-confluent case (an integral needing substitution) is solved by `solve`
  with the same trace quality.
- `prove_equal` returns the rule sequence, not just an expression chain.
- `normalize` cleans output for the arithmetic theory and, unchanged, for a
  boolean theory.
- The MCP server lets an agent author rules, apply them, prove an equality,
  solve to a goal, and receive an NL explanation, with no domain logic in the
  server.
- The corpus generator emits verified, prose-paired records for the example
  domain, using a caller-supplied driver and checker.
