# RERUM future-features roadmap

**Status:** PARTIALLY SUPERSEDED 2026-06-18 by the solver-layer prune (commits b9e9901/7f3a0bf). After pruning rerum.solve/rerum.numeval to OPTIONAL NON-CORE layers, most features here are non-core: A=MCP (integration), B=corpus/synthdata (training/numeric), C=search (rerum.solve; C1 done+deleted). A TRS-focused roadmap would instead emphasize confluence/termination checking, Knuth-Bendix completion, AC-matching, and narrowing. Kept as historical record.
**Date:** 2026-06-18
**Nature:** a DECOMPOSITION + SEQUENCING index, not a single implementation
spec. Each feature below is its own sub-project that gets the full
brainstorm -> spec -> plan -> implement cycle. This doc orders the work and
records dependencies and decisions.

## Foundations (already shipped this session)

Every feature builds on capabilities that already exist, so none start from
zero: rule-set manifests (`from_manifest`, fail-loud op audit), the
`PRELUDE_BUNDLES` registry, `solve_goal` goal-kinds + `op_costs`,
`RewriteTrace.inverse()`, `check_numeric_equiv`, training-record provenance
(`category` + `rules_used`), and the example domain library
(differentiation, integration, limits, boolean, sets, peano, ski).

## Dependency map

```
C1 (by-parts op_costs experiment) --gates--> C3 (AND/OR / subgoal search)
A1 (load-manifest MCP tool) --+
                              +--compounds--> the full agent loop
B1 (certified-corpus demo)  --+
C2 (subst skeleton marker) --enables--> general u-substitution
A2 (custom fold-op registry) --unblocks--> limits-domain-over-MCP
A3 (structural verify tool) -- independent, small
B2 (synthdata-v2) --needs--> B1's corpus pattern + inverse()  [cross-repo]
```

Keystone: **C1 is cheap and resolves the most expensive downstream
decision** (whether C3 is worth building). A1 + B1 are independent but
compound into the whole training-data flywheel (an agent loads a domain by
name, generates a certified corpus, inspects provenance -- all over MCP).

## Sequencing: three waves (critical-path hybrid)

### Wave 1 -- cheap experiments + the compounding loop

All low/moderate effort, all on shipped foundations, mutually independent.

**C1. General by-parts via `op_costs` (experiment).**
- Scope: activate the commented general integration-by-parts schema in
  `examples/integration.rules`; drive `solve` with an `op_costs` table that
  prices `int` high (the lever `solve_goal`/`solve` now expose); measure
  whether best-first closes by-parts cases that the concrete `x*e^x` rule
  does not, and characterize where it diverges (the boomerang family
  `int(e^x sin x)`).
- Output: a DECISION on C3 (is a cost-steered search enough, or is
  subgoal/AND-OR machinery genuinely required?) plus, if the lever works,
  newly-active example content + tests.
- Dependencies: `op_costs` (done), the commented schema (exists).
- Effort: low. Mostly experiment + content + tests; no engine code unless
  the experiment reveals a needed primitive.
- Key decision it carries: whether C3 happens at all.

**A1. Load-manifest MCP tool (path-restricted).**
- Scope: an MCP tool that assembles a domain from a manifest FILE by name,
  restricted to a configured directory (default `examples/`), rejecting path
  traversal. Wraps `RuleEngine.from_manifest`; surfaces the assembled
  status + the fail-loud audit errors as structured responses.
- Output: agents can load whole domains (`differentiation`, `boolean`, ...)
  in one call instead of pasting rule text.
- Dependencies: manifest (done); the registry-driven tool surface (done).
- Effort: low-moderate. One `tool_*` handler + path-restriction + tests.
- Key decision: the allowed-directory configuration model (server arg vs
  env vs fixed); resolved during A1's own brainstorm.

**B1. Certified-corpus demo.**
- Scope: a runnable demonstration (script + tests, NOT engine code) that
  generates a CERTIFIED training corpus over a domain: N differentiation
  and/or integration problems, each solved, each numerically verified by
  `calculus_checker`, each emitted as a JSONL record with prose
  chain-of-thought + `rules_used` provenance, via `generate_corpus`.
- Output: the tangible pattern-#3 artifact; exercises manifests +
  provenance + checkers end-to-end.
- Dependencies: `generate_corpus` (done), checkers (done), provenance
  (done), and (compounds with) A1.
- Effort: low-moderate. Driver wiring + a worked corpus + sanity tests.
- Key decision: which domain(s) and problem generators to seed it with;
  resolved during B1's brainstorm.

### Wave 2 -- MCP surface completion + the substitution primitive

**A2. Custom fold-op registry (server-configured named preludes).**
- Scope: extend the named-prelude resolution so the MCP server operator can
  register additional NAMED preludes (e.g. the limits fold ops) that an
  agent selects by name. Agents pick by name; they never inject code. This
  preserves the code-not-data security boundary while reaching the
  custom-op domains (limits/integration) over MCP.
- Dependencies: `PRELUDE_BUNDLES` registry (done); A1's tool pattern.
- Effort: moderate. The registry extension + reset/assembly wiring + the
  general-engine no-domain-leak guard + tests.
- Key decision: registration API (constructor arg / a register call) and
  whether `from_manifest`-over-MCP can reach these named preludes.

**A3. Structural-verify tool.**
- Scope: an MCP tool that verifies an expression against a STRUCTURAL
  predicate supplied as data (reusing the `matches`/`op_free` goal-kind
  compiler) -- the symbolic complement to `check_numeric_equiv`.
- Dependencies: goal-kind compiler (done).
- Effort: low. Thin handler reusing `_compile_goal`.

**C2. `subst` skeleton marker.**
- Scope: general u-substitution fails because repeated `(fresh u)` markers
  resolve to distinct names; a capture-avoiding skeleton-level substitution
  form fixes co-reference. Step 1 (cheap): promote the existing domain-free
  `_subst` walk (currently example code in `limits_fold_ops`) to a general
  structural fold op. Step 2: the skeleton marker interacting with
  fresh-vars.
- Dependencies: fresh-vars (done), the `_subst` walk (exists).
- Effort: medium. Engine work (the marker + fresh-var interaction) + tests +
  example content proving general u-sub.
- Key decision: marker syntax and resolution order vs fresh-var resolution.

### Wave 3 -- heavy / conditional / cross-repo (informed by Wave 1)

**C3. AND/OR search / search-introduces-subgoals. CONDITIONAL on C1.**
- Scope: only if C1 shows the `op_costs` lever is insufficient AND the value
  justifies it. Required for the fully general by-parts schema and provably
  for the boomerang family (`int(e^x sin x)`, solved via `I = A - I`
  algebraically -- beyond pure rewriting).
- Dependencies: C1's measurement (gates go/no-go).
- Effort: high. New search machinery in the engine.

**B2. synthdata-v2 reconciliation. Cross-repo.**
- Scope: reconcile the existing `fuj-gultepe/synthdata` reverse-process
  generator with rerum's current capabilities -- collapse its dual rule
  files into bidirectional `<=>` + `RewriteTrace.inverse()`, adopt situated
  traces, `to_prose`, and the provenance.
- Dependencies: `inverse()` (done), situated traces (done), and B1's corpus
  pattern.
- Effort: high; different repo; collaboration-shaped.

## Execution model

- Each feature gets its own `brainstorm -> spec -> plan -> implement` cycle.
  This doc is the index; we work it wave by wave.
- First sub-project: **C1** (recommended -- cheapest, de-risks the most by
  resolving the C3 decision). B1 is the alternative if the visible payoff is
  wanted first.
- Cross-feature ordering within a wave is flexible; the wave boundary is the
  meaningful checkpoint.

## Open decisions deferred to each feature's own brainstorm

- A1: allowed-directory configuration model.
- A2: prelude registration API; manifest-over-MCP interaction.
- B1: seed domains + problem generators.
- C2: marker syntax + resolution order.
- C3: go/no-go gated by C1; not designed until then.
