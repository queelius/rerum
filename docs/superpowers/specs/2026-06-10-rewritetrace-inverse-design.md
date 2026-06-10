# RewriteTrace.inverse(): the reverse-trace primitive

**Status:** approved design, ready for implementation planning
**Date:** 2026-06-10
**Scope:** the pure trace-level primitive (`RewriteStep.inverse`,
`RewriteTrace.inverse`) plus its first real consumer (orienting the engine's
`minimize` derivation). No `prove_equal` change; synthdata is unblocked but
out of scope.

## Motivation

A `RewriteTrace` records a forward rewriting derivation: `initial` to `final`
via a list of `RewriteStep`s. There is no way to turn a trace around -- to
produce the derivation that goes from `final` back to `initial`. Three things
want exactly that operation, and it is the same operation:

1. **The Phase 1 minimize-derivation limitation.** `RuleEngine.minimize`
   builds its `OptimizationResult.derivation` from a `prove_equal` proof:
   `path_a` (expr -> common) forward, then `path_b` (best -> common)
   reversed. But reversing the LIST only reverses step ORDER; each reversed
   `path_b` step still carries its forward `before`/`after`/`direction`, so
   the combined derivation is endpoint-correct (initial=expr, final=best) but
   NOT chain-correct under `to_global_sequence` (a step's `after_root` does
   not equal the next step's `before_root`). The fix is to INVERT each
   reversed step, not merely reorder it.

2. **The minimize prose defect (0.9.0 review finding).** Because the
   derivation is not chain-correct, the MCP `minimize` prose (rendered from
   that derivation) narrates phantom no-op steps and, before the fast-follow,
   closed with a false `Answer:` line. Orienting the derivation correctly
   fixes the prose step lines too; the MCP layer consumes
   `OptimizationResult.derivation` and gets the fix for free.

3. **Reverse-process synthetic data (synthdata).** Run an easy confluent
   forward process (differentiate, fold, normalize), then invert the trace to
   manufacture a harder problem whose guaranteed-correct solution is the
   original. The trace inversion is the missing primitive; synthdata proper
   lives in another repo, so it is out of scope here, but this primitive
   unblocks it.

## What a RewriteStep records

(Grounding for the inversion semantics.) `RewriteStep` fields:

- `before`, `after`: the REDEX-LOCAL subtree before/after the edit (not the
  whole expression).
- `path`: child-index path locating the redex relative to the running root at
  the moment the step fired (`[]` = root). `RewriteTrace.to_global_sequence`
  replays the whole-expression states by splicing each step's `after` at its
  `path` into the running root.
- `direction`: `"fwd"`, `"rev"`, or `None` (a bidirectional rule desugars to a
  fwd and a rev variant; `direction` records which fired).
- `kind`: `"rule"`, `"normalize"`, `"fold"`, or `"initial"` (a synthetic
  anchor with `before == after`).
- `bindings`, `guard`: artifacts of the FORWARD match (the pattern bindings;
  the evaluated guard condition+result).
- `rule_id`, `metadata`, `rule_index`, `rationale`: rule identity/labels.

## The primitive

### `RewriteStep.inverse() -> RewriteStep`

A new pure method returning a new step:

- `before, after = self.after, self.before` (swap the redex-local edit).
- `direction`: flip `"fwd" <-> "rev"`; `None` stays `None`.
- `path`: UNCHANGED. `splice_at` replaces the subtree at a path without
  changing list lengths along that path, so the same index path that located
  `before` in the pre-step root locates `after` in the post-step root. The
  inverse step therefore edits the same structural position.
- `kind`: UNCHANGED. A `normalize`/`fold` step inverts to its SPECIFIC reverse
  edit (e.g. `5 -> (+ 2 3)`), which is a valid trace step even though
  normalization/folding are not injective as functions. An `initial` step
  (before == after) inverts to itself.
- `rule_id`, `metadata`, `rule_index`, `rationale`: UNCHANGED. The step refers
  to the same logical rule; `direction` is the source of truth for
  orientation. (We do NOT string-munge `rule_id` suffixes like `-fwd`/`-rev`;
  that is fragile and `direction` already carries it.)
- `bindings = None`, `guard = None`. DECISION (option 1 of the design): these
  describe the FORWARD match. A pure trace transform cannot know the reverse
  application's bindings without re-running the matcher, and claiming the
  forward values on a reverse step would be misleading -- the one field
  synthdata would actually read. Nulling is honest and keeps `inverse()`
  engine-free. The cost: `inverse()` is a STRUCTURAL involution, not a
  field-identical one (see properties).

### `RewriteTrace.inverse() -> RewriteTrace`

A new pure method returning a new trace:

- `initial = self.final`, `final = self.initial`.
- `steps = [s.inverse() for s in reversed(self.steps)]`.

## Properties (the test contract)

1. **Replay correctness (the load-bearing property).**
   `t.inverse().to_global_sequence()` replays `t.final -> t.initial`:
   - `steps[0]["before_root"] == t.final`
   - `steps[-1]["after_root"] == t.initial`
   - adjacent join: `steps[k]["after_root"] == steps[k+1]["before_root"]`.
   Tested on a NESTED-redex trace (a step with `path != []`), since
   path-preservation is the non-obvious part.

2. **Structural involution.** `t.inverse().inverse()` equals `t` on
   `initial`, `final`, and every step's `before`, `after`, `direction`,
   `path`, `kind`. `bindings`/`guard` are cleared by inversion and not
   restored (documented; this is the deliberate cost of option 1).

3. **Endpoints swap** and the step count is preserved.

4. **All kinds invert.** `rule`, `normalize`, `fold`, and `initial` steps each
   invert per the rules above; a `kind="initial"` step (before == after)
   inverts to an equivalent no-op step.

## The minimize fix (first consumer)

In `RuleEngine.minimize`'s derivation construction (engine.py, the
`prove_equal`-based block), change the reversed `path_b` loop from

```python
for step in reversed(proof.path_b[1:]):  # common -> best
    trace(step)
```

to

```python
for step in reversed(proof.path_b[1:]):  # common -> best
    trace(step.inverse())
```

The combined derivation then reads `expr -(path_a, forward)-> common
-(path_b, inverted)-> best`, which IS chain-correct under
`to_global_sequence` (verified by the path-preservation argument: the first
inverted `path_b` step edits `common` at its path, replacing the
common-side subtree with the best-side subtree, walking back toward `best`).

Effects:
- `OptimizationResult.derivation.to_global_sequence()` chains correctly
  (closes the Phase 1 limitation for all consumers).
- The MCP `minimize` tool's `derivation` and `prose` (which render from this
  derivation) narrate the real original->best moves with no phantom no-ops.

Note: `bindings` on the inverted `path_b` steps are now `None` while the
forward `path_a` steps keep theirs. This asymmetry is expected and honest --
the reverse-direction bindings are genuinely not computed.

## Scope boundary

In scope:
- `rerum/trace.py`: `RewriteStep.inverse`, `RewriteTrace.inverse` (pure).
- `rerum/engine.py`: one-line change in `minimize`'s derivation construction.
- Tests: `rerum/tests/test_trace.py` (primitive: replay correctness on a
  nested trace, structural involution, endpoint swap, all kinds); the minimize
  derivation chain-correctness test (engine-level); the MCP minimize prose
  step-line correctness (already partially pinned by the 0.9.0 fast-follow,
  strengthen to assert the global-sequence chain).

Out of scope:
- `prove_equal`: its MCP two-sided prose narrates `path_a`/`path_b`
  separately, each in its own forward orientation, and is already correct.
  No change.
- synthdata reverse-process generation: unblocked by this primitive but a
  separate project in another repo.
- Recomputing reverse-match bindings (option 3): rejected; it would couple the
  trace primitive to the engine matcher.

## Risks

- The path-preservation argument is the crux. The nested-redex replay test
  (property 1) is the guard: if `path` needed transformation under inversion,
  that test fails loudly. Confidence is high (splice_at preserves indices),
  but the test is non-negotiable.
- Changing `OptimizationResult.derivation` orientation alters existing
  behavior for any reader of that field. It is a correctness fix (the prior
  derivation was not chain-correct), not a feature change, and the only known
  consumer is the MCP `minimize` tool, but a CHANGELOG note is warranted.
