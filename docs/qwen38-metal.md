# Qwen3.8 on Metal (status)

Status: the Metal kernels of the Qwen3.8 (`qwen38`) engine are written and tested, and a
`METAL=1` build runs the model's dense layers on them (`c/qwen38_metal_chain.h`, on by
default, `COLI_METAL=0` the CPU engine): every layer, the final mixer and lm_head on the
GPU. With a GGUF whose routed experts are Q8_0, IQ4_NL or IQ4_XS, the routed experts run on
the GPU too (M3, `c/qwen38_metal_experts.h`): the GPU routes (softmax over all experts,
top-k with ties to the lowest id, weights over the top-k's sum: the CPU's rule), a GPU expert
cache for the whole model (`QWEN38_EXPERT_CACHE_BYTES`; by default min(all experts,
max(6 GiB, RAM − dense − 12 GiB))) holds the experts the GGUF's rows read into it, groups of
top-k at a time on a loader thread while the GPU runs the shared expert, and
`q36_expert_sum` adds them in the router's order, then the shared expert, as the CPU does.
The CPU's own expert cache drops to one slot a layer; if the chain turns off mid-run (a
step the device cannot take memory for, for example), the GPU experts release their cache
(one `GPU experts released` line) and the CPU's expert cache gets its slots back before the
CPU runs the rest. `QWEN38_METAL_EXPERTS=0`, a
safetensors checkpoint or another expert type keep the routed experts on the CPU (M2), with
one `[qwen38-metal] GPU experts are off: <reason>` line. The chain's report at exit adds
`experts on GPU: N, loads: L, hits: H`. Expert read-ahead (`QWEN38_METAL_AHEAD`, on by
default, `0` off) in decode: each layer also runs the next layer's router on its MLP row and announces that
layer's six likeliest experts the cache does not hold to the OS (`F_RDADVISE`), once the
next trunk is submitted, so the SSD reads them while the GPU computes; the report then adds
`[metal] qwen38 read-ahead: guesses: G, right: R`. Advice only, the tokens and logits are
the same; a read-ahead that cannot run (its buffers refused at the start, or its router
declining mid-run) turns itself off with one `[qwen38-metal] read-ahead off: <reason>` line
and the steps and the GPU experts go on. Measured on the real model (24 GB M4 Pro): decode 4.53 → 5.18 tok/s (medians of
3), 80% of the guesses right, prefill unchanged
([docs/experiments/qwen38-metal-ahead-2026-11-19](experiments/qwen38-metal-ahead-2026-11-19/manifest.json)). Prompt-lookup drafts are off beside the chain
unless `COLI_LOOKUP=1`: the chain hands a verify to the CPU, the recurrent state down and up
again. Debug-only switches: `Q38M_CHECK_WEIGHTS`, `Q38M_LAYER_CHECK`, `Q38M_DECLINE_EVERY`,
`Q38M_FORCE_ROUTES`, `Q38M_FAIL_SLOT_ALLOC`, `Q38M_REFUSE_AT` (refuses the k-th step's device
memory: the off path's test), `Q38M_FAIL_AT_STEP` (the k-th step fails as a kernel would: the
failure path's test), `Q38M_AHEAD_EARLY`, `Q38M_AHEAD_FAIL` (read-ahead's degrade
path) ([ENVIRONMENT.md](ENVIRONMENT.md)). Each is read once and has no effect unset.

The device's K/V/IK caches grow in steps (256, 512, 1024, then 1024 more at a time). A
growth drops the device's rows and uploads them again from the host's, which are always
current, and pools the index keys again; the report's `[metal] qwen38 kv:` line counts the
growths and the rows uploaded again.

## Measured (real model, Plan 2)

Heretic-2 IQ4_XS GGUF (125 GB, experts streamed from the SSD), 24 GB M4 Pro, `bench/q38_bench.sh`
(64 tokens on a 61-token chat prompt for decode, the 12,689-token `long12k` prompt for
prefill, medians of 3, one warm-up), fast kernels:

| build | decode tok/s | prefill 12.7K s | peak footprint, GiB (GB) |
|---|---|---|---|
| CPU engine, MTP off (the baseline) | 3.28 | 1076 | 12.6 (13.5) |
| M2: GPU trunk, experts on the CPU (`QWEN38_METAL_EXPERTS=0`) | 3.20 | 1239 | 11.8 (12.7) |
| M3: GPU experts, read-ahead off (`QWEN38_METAL_AHEAD=0`) | 4.53 | 751 | not measured |
| **M3, the default** (GPU experts + read-ahead) | **5.07** | **758** | 14.7 (15.8), in prefill |

GiB = 2^30 bytes, GB = 10^9 bytes. A same-day CPU recheck ran 3.50 tok/s and 1052 s. So M3
decodes 1.54× faster than the baseline (5.07 / 3.28) and 1.45× faster than the same-day CPU
run (5.07 / 3.50), and prefills 1.42× and 1.39× faster (1076 / 758, 1052 / 758). How busy
the GPU is (its residency) was not measured; `sudo powermetrics --samplers gpu_power -i 2000
-n 3` shows it during a run. Read-ahead alone is +14% on decode (interleaved single runs: 4.40–4.52 off, 5.09–5.17 on), 80% of
its guesses right. The SSD is still the main cost: the report's `expert reads waited` is
about 12.5 s of a 22 s decode run (the 61-token prompt, then 64 tokens).

Agreement with the CPU engine in reference precision (`GQ_EXACT=1 Q38_TRUNK_CPU_INT8=0`),
teacher-forced on the CPU's tokens (`bench/q38_agree.sh`): 128/128 on the two short
prompts. Long prompts: with routes and QSA selections forced to the reference run
(`Q38M_FORCE_ROUTES=2`), every layer of a 3K-token prompt is within 5.3e-5 of the CPU;
unforced, long12k agrees 12 of 14 scored steps, the misses after router near-ties
(top-k margin about 2e-6) that a one-ulp change flips in the CPU itself.

## Limits of the chain

- MTP is off beside the chain (`Q38_MTP=1` turns the chain off with its line); GPU MTP is
  planned (M5).
- Prompt-lookup drafts are off beside the chain unless `COLI_LOOKUP=1` (their verifies run
  on the CPU).
- `KV_SLOTS > 1` (`mux_rows`), the vision tower, and dense weight types other than Q8_0,
  Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS, BF16 and F32 keep the whole model on the CPU, with one
  `[qwen38-metal] the Metal chain is off: <reason>` line.
- In `coli serve`, prompt segments are 2048 rows for the Metal build (the serve prefill does
  not know the chain is on); prefill also re-reads the routed experts for every 256-row
  chunk, since the cache (6.8 GiB here) is far smaller than one chunk's experts. Streaming
  prefill is planned (M4).
- A GPU command batch that fails, or a kernel that declines, once the recurrent state has
  moved on the GPU (any step after the first, or a later chunk of the first) leaves no state
  the CPU could take over. In serve mode the turn ends with `ERROR <id> GPU_FAILED ...`. The
  client sees the error: a streamed reply (the gateway has already sent its 200 once the engine
  accepts the prompt, also when the failure comes while the prompt is read) keeps the text sent
  so far and ends with a final `data: {"error": {"code": "gpu_failed", "type": "server_error",
  "message": "The GPU path failed and this reply was stopped. ..."}}` chunk and no `[DONE]` (the
  OpenAI SDKs raise it; the chat page shows it as the reply's red error notice); a reply that is
  not streamed is an HTTP 503 with the same error object. The chain turns off with its line and
  frees its device memory, the CPU's expert cache gets its slots back, and every state that came
  from the GPU goes (the live state, the prompt-end snapshot, a parked conversation, the prefix
  store's prompts in RAM and, with `Q38_PREFIX_DISK=1`, this model's checkpoint files on disk,
  those of earlier runs too). The engine keeps serving: the next request reads its whole prompt on the
  CPU, which runs the model until the server is restarted; the `METAL` frame says `chain=0`
  and why. Outside a serve the run stops as before (`the Metal chain failed: ...; stopping`,
  exit 1). An SSD read of an expert that fails stays fatal everywhere
  (`GPU experts: reading ... failed: <error>`, then `... stopping`, exit 3): the server
  reports `engine exited` and must be restarted (`COLI_METAL=0` runs the CPU engine). A step
  whose device memory is refused, or a step the chain declines at its start, is not fatal:
  the CPU runs it.
- GPU residency was not measured (see above).
- macOS only (Metal).

The kernels share the device, the command batch and the exact/fast switch of the Qwen3.6
Metal path ([qwen36-metal.md](qwen36-metal.md)). `q38_gpu_init()` initializes qwen36's
library first, then compiles `c/metal/qwen38.metal` on the same device.

## Kernels

All in `c/qwen38_gpu.h` (host wrappers, `c/qwen38_metal.m`; source `c/metal/qwen38.metal`):

| Wrapper | What it does |
|---|---|
| `q38_copy` | plumbing test only |
| `q38_dot_bf16` | `y = x . w^T` with `w` raw BF16 (the indexer projections) |
| `q38_hc_low`, `q38_hc_inj`, `q38_hc_mix`, `q38_hc_apply` | hyper-connection element-wise ops |
| `q38_hc_norm` | per-stream rms0 with the per-stream weight rows |
| `q38_ple_gate`, `q38_ple_conv` | per-layer n-gram embedding gate and dilated conv with ring and snapshots |
| `q38_qsa_pool`, `q38_qsa_score`, `q38_qsa_select` | QSA block keys, indexer scores, top-block selection |
| `q38_attn_sel` | sparse attention over the selection (f32 or f16 KV) |

Qwen3.6 kernels extended for Qwen3.8: `q36_router_select` takes up to 512 experts and
top-k 16 (a separate `q36_router_select_wide` kernel; the old path is unchanged at
256 experts and top-k 8), and `q36_expert_sum`, `q36_expert_gate_up_gq` and
`q36_expert_down_gq` accept top-k up to 16.

## Running the tests

```
make -C c test-qwen38-metal-kernels     # every kernel against a double-precision reference
make -C c test-qwen38-metal-counter     # builds with one injected bug each, n = 1..5; exits 0 only if all five fail
QWEN38_BENCH=1 ./c/tests/test_qwen38_metal_kernels   # also prints a report-only microbench
make -C c test-qwen36-kernels           # the Qwen3.6 kernels, unchanged
make -C c test-qwen36-graph METAL=1
```

The tests run each kernel at the real model's shapes (H 2560, C 4, 512 experts, top-k 10,
24 heads, QSA budget 2048, positions to about 9000) and at the tiny fixture's, in exact and
fast mode. `exact twice same bits` runs every op twice in exact mode and requires
byte-identical outputs. They skip on a machine without a Metal device and on GitHub's Apple
Paravirtual device.

The counter-tests build the test with `-DQ38_COUNTER_TEST=<n>`, which reaches the Metal
sources as a preprocessor macro (`MTLCompileOptions.preprocessorMacros`, set only in those
builds): 1 hc_mix divisor, 2 PLE ring shift, 3 QSA tie order, 4 router tie order (in
`q36_router_select_wide`), 5 bf16 expansion. The test must fail for each.

## Exact and fast

`q36_gpu_set_exact(1)` selects the exact kernels (fixed summation order, no `simd_sum`);
the default is fast. `QWEN38_METAL_EXACT=1` maps to it ([ENVIRONMENT.md](ENVIRONMENT.md)). Kernels without a reduction that differs
(`hc_norm`, `ple_gate`, `qsa_*`, `attn_sel`) are the same kernel in both modes.

## Tolerances

`|got - ref| <= tol * (|ref| + 1e-3)`:

- exact mode: 1e-5 for element-wise and norm ops, 2e-5 for dots and attention;
- fast mode: 3e-4;
- selected ids and blocks, the rings and snapshots: identical in both modes.

For dots and reductions that can cancel (dots, matmuls, `hc_mix`, expert sums, attention) the
scale is the sum of the absolute terms instead of `|ref| + 1e-3` (the default is only
meaningful for element-wise and norm ops). The tolerance values themselves are unchanged.

## Known limits

- End-to-end QSA selections and router ids can differ from the CPU near ties: the dot order,
  float against double `rms0` and Metal's trig differ, and the router selects on a
  normalized probability whose Metal `exp` differs from libm's by a few ulps. The tests use
  logits separated by more than float rounding; exact mode is not bit-identical to the CPU
  on the real model, and an end-to-end gate has to measure the divergence.
- `mux_rows` (a per-row position or index key) is unsupported: the GPU assumes row `s` is at
  `pos_base + s`. The PLE ring would also need one ring per sequence.
- Concurrent mode needs `q36_gpu_barrier` between dependent dispatches (for example
  `q38_ple_gate` then `q38_ple_conv`); the wrappers do not insert it.
- `q38_attn_sel` does not range-check the selection against the cache capacity.
- The RoPE angle correction is tested up to position 8996.
- Wrappers return 0, enqueueing nothing, for shapes outside their limits (see the comments in
  `c/qwen38_gpu.h`).
