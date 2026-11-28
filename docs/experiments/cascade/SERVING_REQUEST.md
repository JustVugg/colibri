# Parent admission request: CASCADE-Q36-SERVE-001

**REQUESTED, NOT ADMITTED. No model inference or GPU job has started.**

Parent integration/GPU owner: please approve or reject the bounded scope below.
An idle queue is not permission. Approval must identify this experiment and
reserve the serial queue; it must not be inferred from this published proposal.

## Verified local target

User-provided model root: `C:/Users/giaco/.lmstudio/models`.
Six Qwen GGUFs were parsed read-only; tensor bounds are present for all six.
See `local-model-candidates.json`. They are source-compatible with `qwen36`,
not runtime-qualified merely because the GGUF headers parse.

First target: **Qwen3.5-9B Q6_K**, existing registered `qwen35-9b-q6` model.
It is not the literal Qwen3.6-35B; that separate registered download is partial.
The actual family registry routes `qwen3_5_text` to `qwen36`, which calls the
shared sampler. Vocab 248320 also matches the bounded synthetic prior.

- File: `lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q6_K.gguf`
- Size: **7,359,259,040 bytes**
- SHA-256: `b2ccc477f0a2449de299bef560b63e498cae94a2f3a3bae13df01168d77773e6`
- **427 tensors: 250 Q6_K, 177 F32**; all complete byte ranges verified.
- 32 layers (24 linear attention, 8 full attention); original depth retained.
- Read-only inventory/full-file hashing cost: 5.68 s; no model copy/conversion.
- GPU query: RTX 5080, 16303 MiB, driver 621.40. Residency is **not** verified.

Why this target first: complete file, registered tokenizer/config, matched
sampler vocabulary and smaller bounded loading cost than the 18.56GB Qwen3
MoE. Other complete attention-only Qwen3 models offer held-out architecture
coverage if a useful first-model effect is actually established.

## Bounded requested work

The machine-readable plan is `serving-admission.json`; frozen public prompt
bytes/hashes are in `serving-prompts.json`. Both histories use unchanged model,
quantisation, context cap2048, concurrency1, seed42, prefix-off and speculation-off.
Exact native prompt token counts must be frozen in preflight (no Python
`tokenizers` package is installed); they are not claimed from text byte counts.
Reject overflow; do not truncate prompts or change depth/context to fit.

1. Parent-reviewed private same-build Q36 engine with generated, default-off
   guarded radix sorter. Retain original engine; no installed files changed.
   Pin compiler, source, flags, engine/backend/runtime hashes and reached
   selectors before loading. Re-run matching sampler/edge oracles.
2. **Phase A, at most two starts:** one baseline diagnostic process per
   short/long history. Warm up identically, collect actual sampler fraction,
   correctness, load time, prefill/decode and memory. Instrumentation is
   test-only and must be reviewed/implemented before build.
3. **Phase B, conditional, at most eight starts:** two balanced off/on pairs
   per history (AB then BA), same executable, diagnostic profiling off.
   Each process gets identical greedy warmup, a measured greedy control,
   then full-sort sampled request; maximum32 output tokens per measured turn.
   Checkpoint complete pairs atomically.
4. Stop on correctness/load/state mismatch, insufficient memory/chain decline,
   or a credible Amdahl upper bound below3% total-request gain. No silent
   fallback, precision change or expanded sweep. No nucleus sweep.

Record exact token/text hashes, counts/stops/RNG calls, TTFT, inter-DATA
intervals (not assumed token boundaries), request/throughput distributions,
prefill/decode, sampled peak VRAM/process-tree RAM, cold setup and total cost.
Profiling/observer cost and coarse sampled-memory/tail limitations are explicit.
Never sum overlapping CPU/GPU intervals or pool histories as replicates.

Hard bounds: **10 total model starts**, **240 s/start**, **1800 s total**,
one GPU job family at once; no overlapping compilers/heavy CPU work.
Whichever timeout is reached first ends the campaign.

This requests G0/G1 recipient preflight and conditional G2 only.
G3–G5, alternative models, installation and default promotion are **not**
admitted. Keep NO_CHANGE until useful model evidence and later gates pass.

## Current blocker / decision receipt

No approval receipt exists. GPU/build admission and shared integration remain
parent-owned; this request cannot authorize itself.
Please return an explicit approval/rejection for **CASCADE-Q36-SERVE-001**
and a queue reservation. Results will extend the existing Dory transfer registry.
No Dory working-tree edits are required to review or admit this isolated trial.
