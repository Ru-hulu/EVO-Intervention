# Validated runtime optimization of the SOC disruption evaluation

This package overlays the immutable `soc-disrupted-eval-20261115/code`
snapshot. Set `PYTHONPATH` to the optimization snapshot followed by the original
snapshot. Do not replace the original input manifest or scientific code.

The user authorized pausing the queue, selecting faster parameters, discarding
the previous temporary results, and starting the nine model/suite evaluations
again without an additional approval. The deletion receipt is kept outside Git.

## Changes

- ProcVLM and Robometer reuse a bounded 512 MiB CPU cache of the native image
  processor output. Task/token templates are cached by exact image grid layout.
  Real-image pixel tensors, grids and token sequences are checked against the
  original processor. Each sample retains its own original history window.
- Consecutive predictions can be batched. ProcVLM retains greedy progress-only
  generation, and Robometer retains the native final progress-token head.
- FoREST advances the native vision and CLIP/max-depth memory stream one step
  at a time. Independent language forwards can be batched using detached copies
  of each step's bounded history. Batch one calls the frozen backend directly.
- An exact-input result cache can eliminate repeated singleton forwards for
  ProcVLM and Robometer across perturbation versions. Its key contains the
  checkpoint, task text and every ordered image path under the hash-frozen
  manifest. A real-GPU reference/repeated-input gate precedes formal evaluation.
  FoREST is excluded from this cache because its recurrent state can differ.
- Writes are coalesced to eight finished predictions or two seconds and saved
  atomically, with a final flush on cooperative pause and episode completion.
  Resume rebuilds only FoREST's vision memory and never overwrites saved values.
- A single controller schedules the longest estimated remaining tasks first.
  One entire model/suite job occupies each GPU; all original resource guards,
  admission checks, numerical tolerance and checkpoint audits remain in force.

## Release gates

`tune_queue` tests all nine actual suite checkpoints on isolated real static and
rewind trajectories. The native checkpoint/resume canary is also required.
Candidate batches are 1/2/4/8/16/32 (FoREST through 16), with conservative memory
limits and early stopping before another doubled allocation would risk headroom.
Once the State checkpoint rejects the batched family, later suite tests may
prune that family and independently validate native singleton plus caching;
each receipt records the actual candidates and the pruning evidence.
No candidate is accepted if it changes invalid-answer positions, input identities,
or differs from native singleton predictions by more than 0.002. A second test
changes batch boundaries after a reconstructed prefix. Rejected parameters never
write formal predictions.

`runtime-manifest.json` hashes every released Python file.
`execution-configs.json` binds each selected configuration to its GPU test receipt
and runtime manifest. Both are checked at worker startup. The weight initialization
audit remains the immutable native-reference audit; `execution_segments` explicitly
records the actual batching, cache settings and release hashes used in each curve.
The optimized report checks execution provenance before calling the unchanged
scientific metric and source/input/checkpoint verification code.

No data, predictions, weights, tuning artifacts or generated receipts belong in Git.
