# Experiments

Empirical probes of RERUM's equivalence, proof, minimize, and random-walk
features. These are not unit tests. They measure what works, how well, and
where the cliffs are for the features added in v0.4.

## Running

```bash
python experiments/features_benchmark.py
python experiments/scaling.py
```

Each script is self-contained and prints timings alongside correctness checks.

## What each experiment measures

### `features_benchmark.py`

| # | Probe | What it validates |
|---|-------|-------------------|
| 1 | `prove_equal` on De Morgan, triple-neg, commute-mix | Correctness and sub-ms latency on small bidirectional rule sets |
| 2 | `enumerate_equivalents` on `a+b`, `(a+b)+c`, `(a+b)+(c+d)` | Class sizes match `n! × Catalan(n-1)` exactly (2, 12, 120) |
| 3 | `prove_equal` on reordered 4-term sum | Bidirectional BFS finds the meet-in-the-middle proof at depth 3 |
| 4 | `minimize` with and without `include_unidirectional` | Demonstrates the default-flag footgun |
| 5 | `sample_equivalents` diversity vs. sample size | Birthday-paradox collision curve in a 120-state class |
| 6 | Cycle stress: rule where `lhs == rhs` | Dedup terminates cleanly |
| 7 | `prove_equal` vs enumerate-then-check | Bidirectional BFS is roughly 23x cheaper on 4-term identity |

### `scaling.py`

Enumerates the equivalence class of an `n`-term sum `a+b+c+...` under
`assoc ∧ commute` for `n = 2..5`. Validates that the discovered class size
matches the theoretical `n! × Catalan(n-1)`.

| n | theory | forms | time |
|---|--------|-------|------|
| 2 | 2      | 2     | ~0.2 ms |
| 3 | 12     | 12    | ~2 ms |
| 4 | 120    | 120   | ~30 ms |
| 5 | 1680   | 1680  | ~600 ms |

Enumeration becomes impractical around `n=6` (~30k forms). For larger problems
use `prove_equal` with a sensible `max_expressions` budget instead of full
enumeration.

## Observations that drove follow-up fixes

1. **`minimize()` default was a footgun.** Prior to v0.4, `include_unidirectional`
   defaulted to `False`, so minimize silently ignored every `=>` rule. Flipped
   the default to `True` because simplification rules are overwhelmingly
   unidirectional in practice.
2. **`prove_equal` needed a work budget.** Un-provable queries exhaust the full
   depth-bounded reachable set on both sides, which can take tens of seconds
   even at depth 6 on rich bidirectional rule sets. Added `max_expressions` as
   a total-work cap.
