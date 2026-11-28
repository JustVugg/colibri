# CASCADE-SAMPLE-001 — frozen qualifying plan

Integration base: `ab98f29f` (`upstream/dev`), independent branch
`feat/colibri-optimisation-cascade`. No Dory commits are merged.

Single shared-code integration owner: parent (Dory coordinating session);
Cascade submits isolated proposals for parent integration and maintainer review.
Dory retains ownership of its evidence and experiments. Cascade owns only
`docs/experiments/cascade/` and `c/tools/cascade/` in this proposal.
No shared production edits, GPU jobs, model downloads or installed changes.
Any eventual production patch requires separate review by a Colibri maintainer.

Prior: Dory TRANSFER_REGISTRY M03/E06 at `f84e14b1`, with earlier
`1af3c5d5` mechanism anchor. Source claims are not recipient measurements.
Ranked first because Qwen3.6 actually includes the shared header, requires no
GPU admission, has a pre-existing independent full-sort oracle, and preserves
the arithmetic/RNG contract. Packed kernels require inventory and GPU admission;
batching requires model state/cancellation tests; arena reuse is already present
in Qwen's tier. None is justified by another model's headline gain.

Prediction recorded before execution: sort-heavy full-distribution CPU sampling
should improve; greedy should be unchanged; nucleus may lose when few entries
are sorted. Numeric effects are unknown. No prediction of model throughput.

Minimum decisive test: extract only the guarded stable-sort helper from the
immutable Dory Git object; replace only the two sorting calls in a generated
copy of this integration base's header. Compare original executable and
candidate executable (selector off/on). Preserve upstream logit-dump behavior.
Use identical compiler flags, synthetic logits, seed, temperature, vocabulary,
single-thread/process conditions, warmed code, no GPU and rotating paired order.
Record exact picked tokens plus subsequent RNG state, five paired repetitions
per case and sampler wall-clock distributions. Run existing sampler oracle on
the generated header with selector both off/on and on the original header.

Cases: Dory-size 131072 and Qwen3.6-size 248320; tied, peaked and broad logits;
temperature .7 at top_p .9/1 plus greedy control. No sweep beyond this screen.
Timing excludes construction/reseeding and includes allocations and sorting.
Stop promotion on any token/RNG or regression-oracle failure. Component benefit
does not authorize promotion: full-model correctness, serving stability,
prefill/decode, load time, total/p95 latency, throughput, RAM/VRAM and cold state
remain unmeasured. Memory is explicitly not inferred from allocation sizes.

Acceptance: accept only a test-only mechanism qualification if exactness passes;
retain all losing cases. Do not generalize average gains to all sample regimes.
Follow-on: Qwen3.6 matched serving only after model availability/resource
admission; no duplicate broad sweep. Rollback is removal/revert of isolated
Cascade files; production rollback is unnecessary because production is unchanged.
