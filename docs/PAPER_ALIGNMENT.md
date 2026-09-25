# Paper alignment

## Authority and scope

This is a newly written, standalone NumPy reference for Section 4 (Method)
and Algorithm 1 of **When Greedy Turns Risky, Can Curvature Coordinate Growth
Across Capabilities?**, ICLR 2027 review copy, checked on 2026-09-25.
It is not the original experiment trainer. No historical local trainer is used
as evidence for the paper's algorithm or experimental identity.

The reviewed 20-page PDF has SHA256:

```text
326883348f1a7332356d1b54f4d0d1ab19aa9d6b549300bde430c6966dabcad9
```

The Method source has SHA256:

```text
2fa5a805a9e336e190cfe57416674d7101e211a0ec4d0cd7feb8ff024202c81a
```

Only these content fingerprints are distributed, not the local paper source,
PDF, build logs, source directories, or author metadata.

## Equation-to-code map

| Paper interface | Reference function | Contract |
| --- | --- | --- |
| Eq. (4), domain cosine | `cosine_similarity` | Domain prototypes against the target prototype |
| Eq. (5), feedback cost | caller input | Supply `psi_cost(domain_reward_ema)` explicitly |
| Eq. (6), clipped z-score | `normalize` | Population standard deviation; zero signal below epsilon |
| Eq. (7), domain cost | `domain_costs` | Negative normalized cosine plus normalized feedback cost |
| Eq. (8), KL allocator | `allocate_domains` | Positive reference; exponential tilt; no added minimum share |
| Eq. (9), integer quotas | `largest_remainder`, `allocate_quotas` | Exact budget; supply cap and proportional redistribution |
| Eq. (10), residual sketch | `make_projection`, `residual_sketches` | Fixed Gaussian projection; subtract the domain prototype |
| Eq. (11), Gram state | `update_gram`, `sketch_risk` | EMA of excluded-anchor outer products; raw quadratic risk |
| Eq. (12), case coordinates | `cosine_similarity`, caller input | Explicit `psi_learn(case_reward_ema)` |
| Eq. (13), fixed case score | `candidate_scores` | Relevance plus learnability minus normalized risk |
| Eq. (14), finite-step utility | `finite_step_scores` | Exact candidate-dependent coordinate, not global optimization |
| Eq. (15), sampling | `softmax`, `TaylorMixSelector.select` | Conditional softmax without replacement |
| Eq. (16), feedback | `ema`, caller integration | Model/rollout/cache refresh remains external |
| Algorithm 1, pool policy | `TaylorMixSelector` | Persistent pools, exclusion, reset, removal, final shuffle |

## Critical finite-step formula

For a prefix of size `k` and within-domain sketch mean `mean_z`:

```text
raw_risk_i = z_i^T G z_i
Gamma_i = S_i - k/(k+1) * z_i^T G mean_z - raw_risk_i/(2*(k+1))
P(next=i) = softmax(Gamma / sample_temperature)_i
```

At `k=0`, the score is `S_i - raw_risk_i/2`. The conditional score uses
**unnormalized** risk, whereas Eq. (13) uses window-normalized risk.
The score, sketches, Gram matrix, and window normalizations remain fixed
throughout a domain quota. Prefix means restart at zero for each domain.
There is no extra age, anchor-alignment, or minimum-domain-share term.

For a nonempty prefix, define `V = mean(S) - mean_z^T G mean_z/2`.
The tests check that pairwise differences in actual next-batch utility gains
equal pairwise differences in `Gamma/(k+1)`. They separately check the
first-draw utility and the sampled conditional probabilities.

## Pool and rounding conventions

- Exclude shadow-anchor IDs before calculating eligible capacities.
- Initialize each ordered pool by a seeded shuffle and retain it across steps.
- Round shares by largest remainder. Exact fractional ties follow domain order.
- Cap quotas at full eligible domain supply, then redistribute the deficit
  by the original shares among domains with spare capacity, repeating if needed.
- When remaining supply is smaller than `min(quota, eligible_capacity)`, reset
  to a shuffled full eligible pool, even if the remaining pool is not empty.
- A configured prefix window is enlarged to at least the quota. `None` means
  the whole remaining pool. If the window is exactly the quota, take all its
  IDs directly, as specified in Algorithm 1.
- Remove selected IDs from the temporary candidate set and persistent pool.
  Preserve the relative order of unselected IDs; never swap a selected batch
  member out. Shuffle the final concatenated ID list.

## Explicit limits and unspecified integration choices

The PDF names feedback mappings, EMA/cache maintenance, and missing-history
handling, but does not give a complete executable specification for these
trainer interfaces. This release does not invent experimental settings for
them. Callers must supply finite mapped signals, prototypes, representations,
anchor sketches, reference weights, temperatures, and score weights.

The demo uses synthetic inputs and explicitly illustrative hyperparameters.
Population variance, deterministic quota tie-breaking, float64 computation,
and the supplied iterative redistribution routine are reference conventions,
not verified claims about an unavailable training implementation.

Full-support domain shares are a real-arithmetic property. If float64
underflows a share to zero, this reference raises instead of silently imposing
a floor that changes Eq. (8). Candidate softmax may underflow negligible tails.
Zero-vector cosine inputs, missing candidate history, nonfinite values, invalid
Gram matrices, and globally insufficient supply raise explicit errors.
Gram validation allows only small floating-point symmetry/PSD tolerance.

The quadratic state is a sketch-space Gram matrix, not a training Hessian.
No unit test validates proxy calibration, training convergence, benchmark
scores, timing, or empirical coverage. In particular, the paper's **56/56**
integer coverage observation is not a general guarantee of positive rounded
quotas and is not reproduced by this synthetic test suite.
