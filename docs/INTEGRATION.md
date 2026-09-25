# Trainer integration boundary

The original experiment trainer is not included. The reference selector
accepts cached numerical signals and returns integer case IDs; it performs
no model loading, dataset reading, network calls, filesystem writes, rollout,
reward evaluation, or optimizer updates.

## Before rollout

1. Assign globally unique, nonnegative integer case IDs and group them by domain.
   Keep domain order fixed for array inputs and rounding ties.
2. Configure shadow anchors as `excluded_ids`. This reference uses a fixed
   exclusion set for a selector instance. Supply only those excluded anchors
   to `update_gram`; the caller owns consistency between anchor IDs and sketches.
3. Extract cached representations and domain/target prototypes externally.
   Construct the Gaussian projection once and persist it or its seed.
4. Calculate residual sketches against each case's domain prototype. Build the
   Gram EMA using `update_gram`, with an explicitly selected decay.
5. Supply raw domain alignment and mapped reward costs to `domain_costs`.
   Supply case relevance and mapped learnability for every exposed candidate.
   Missing-history fallback must be resolved by the caller; it is never
   silently interpreted as zero.
6. Call `selector.select(costs, reference, relevance=..., learnability=...,
   sketches=..., gram=...)`. `costs` and `reference` follow the constructor's
   domain order. Candidate containers are indexed by integer case ID and may
   be arrays or dictionaries.
7. Feed `result["indices"]` to the existing GRPO rollout and update unchanged.

## After rollout and checkpointing

Refresh reward histories, prototypes, cached representations, anchor sketches,
and the Gram state for the next selector call, not during quota construction.
The generic `ema` helper uses `decay * previous + (1-decay) * observation`;
the caller chooses initialization, decay, refresh cadence, reward aggregation,
and `psi_cost` / `psi_learn` mappings.

Persist `selector.state_dict()` with the model and the external caches.
`load_state_dict()` requires matching domains, eligible IDs, and configuration;
it restores pool order and the RNG so future draws can be replayed. Treat real
state snapshots as private data and do not include them in public exports.
Failed selection calls leave pool and RNG state unchanged.

## Running the synthetic example

```bash
python -m pip install -r requirements.txt
python -B examples/selector_demo.py
python -B -m pytest -p no:cacheprovider tests
```

The example needs only CPU NumPy and synthetic arrays. Its values and seed
are illustrative, not paper hyperparameters or a claimed training result.
No model path, training path, account, access token, or cluster is required.
