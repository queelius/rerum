# C1: general integration-by-parts -- measured experiment + cheap refinement

**Status:** approved design (user sign-off 2026-06-18)
**Date:** 2026-06-18
**Roadmap:** Wave 1, first sub-project. See
`docs/superpowers/specs/2026-06-18-future-features-roadmap.md`.

## Purpose

Decide -- with measurement, not assertion -- whether general integration by
parts can be made ROBUST with cheap, content-level mechanisms (the
`op_costs` lever plus pattern-restriction and theory-normalization), or
whether it genuinely needs the heavy AND/OR / subgoal search (C3). This is
the keystone experiment: its result is the C3 go/no-go.

## Background measurement (the probe that motivated this)

The commented general by-parts schema
`(int (* ?u ?dv) ?v) => (- (* :u (int :dv :v)) (int (* (int :dv :v) (dd :u :v)) :v))`,
co-loaded with the differentiation rules and driven by `solve` with
`op_costs({"int": 50})`, gives a MIXED, fragile result:

| Case | Result |
|------|--------|
| `int(x*e^x)` | closes in 2 nodes -- via the CONCRETE rule, not the general schema |
| `int(x*cos x)` | closes in 7 nodes (general schema works) |
| `int(x*sin x)` | EXPLODES: 3000 nodes, ~27s, no solution |
| `int(x^2*e^x)` | EXPLODES: 3000 nodes, ~40s (needs parts twice) |
| `int(x*ln x)` | EXPLODES: 3000 nodes, ~36s (needs the u/dv choice) |

So raw `op_costs` is fragile -- it closes `int(x*cos x)` but explodes on the
structurally-identical `int(x*sin x)` (cost-landscape luck, not robust
solving). The schema fires on EVERY product and spawns more integrals, so
the branching is unbounded. The refinements below aim to BOUND the
branching, not merely reorder it.

## Mechanisms under test

1. **Pattern-restriction (primary).** Replace the unguarded `(* ?u ?dv)`
   with patterns requiring the polynomial factor explicitly:
   `(int (* ?v ?g) ?v)` (the n=1 case) and
   `(int (* (^ ?v ?n:const) ?g) ?v)` (general power). By-parts then fires
   only on `polynomial^k * g`, not every product -- the LIATE "pick u = the
   algebraic factor" heuristic encoded structurally. No engine code.
2. **Theory-normalized factor order.** Thread the arithmetic theory so `*`
   canonicalizes (the probe ran un-normalized). For `symbol * compound`,
   `ORDER_KEY` already orders the simpler factor first, giving a consistent
   u/dv split for free. Measure its standalone contribution.
3. **Cost-shaping.** Beyond pricing `int` high, a size- or degree-aware cost
   function. Fragile alone (per the probe); tested as a complement to 1/2.
4. **Fallback: guard-predicate.** Keep the general pattern, gate with a NEW
   GENERAL structural predicate `poly-in?(u, v)` (true when `u` is built from
   `v`, numbers, `+`, `*`, and `^` with constant exponents). More general
   than pattern-restriction but needs engine code -- pursued only if 1 is
   too narrow. It must name no domain (a structural predicate, like
   `free-of?`).

## Case battery

Classic polynomial x transcendental (the textbook by-parts set):
`int(x*sin x)`, `int(x*cos x)`, `int(x^2*e^x)` (parts twice),
`int(x*ln x)` (u/dv choice), plus `int(x*e^x)` (concrete-rule control), plus
the BOOMERANG `int(e^x*sin x)` as the known-hard frontier case (parts twice
reproduces the original; closing it needs the algebraic `I = A - I` step,
beyond pure rewriting).

## Success bar

A mechanism (or combination) WORKS iff it closes the classic
polynomial-times-transcendental battery within a small node budget
(target <= 500 nodes each) AND regresses no existing integration test. The
boomerang is EXPECTED to remain out of reach under every cheap mechanism --
that is the result that pins C3's necessity to a specific, named family.

## Deliverables

- ALWAYS: `experiments/byparts_search.py` -- a runnable matrix over
  (case x mechanism x budget) reporting found/explored/time per cell, plus a
  findings summary printed at the end. Follows the `experiments/` convention
  (a script, not pytest). Its module docstring records the C3 DECISION:
  which cases cheap refinement closes, which remain frontier, and therefore
  whether C3 is needed and for exactly which family.
- CONDITIONAL (iff a cheap mechanism makes the classic battery robust):
  ship the tamed general by-parts rule(s) as ACTIVE content in
  `examples/integration.rules` (replacing/supplementing the concrete
  `int-byparts-x-exp`), with sidecar examples in
  `examples/integration.metadata.json` and end-to-end + numeric-verification
  tests in `rerum/tests/test_integration.py` (driven by the existing
  `integrate()` helper and confirmed by `is_integral`).
- NO engine code unless mechanism 4 is required; if it is, a single general
  structural predicate added to `PREDICATE_PRELUDE` (no domain), with its
  own unit tests, and the `test_mcp_no_domain` swap guard must still pass.

## Out of scope

- C3 itself (AND/OR / subgoal search) -- this experiment DECIDES whether to
  build it; it does not build it.
- The boomerang family's algebraic `I = A - I` resolution (that is C3's
  burden if pursued).
- General u-substitution (C2).

## Risks

- A cheap mechanism might close the battery but introduce search slowness on
  UNRELATED integration cases (the new by-parts rule competing on every
  product). Mitigation: the success bar includes "regresses no existing
  integration test", and the experiment measures the non-by-parts cases too.
- Pattern-restriction might be too narrow (misses `int((x+1)*e^x)` and other
  non-monomial polynomials). The experiment notes such gaps; mechanism 4
  (poly-in? predicate) is the documented next step if the gap matters.
