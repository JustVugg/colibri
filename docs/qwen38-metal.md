# Qwen3.8 on Metal: kernels (status)

Status: the Metal kernels of the Qwen3.8 (`qwen38`) engine are written and tested, and a
`METAL=1` build runs the model's dense layers on them (`c/qwen38_metal_chain.h`, on by
default, `COLI_METAL=0` the CPU engine): every layer, the final mixer and lm_head on the
GPU, the routed experts on the CPU (M2). Prompt-lookup drafts are off beside the chain
unless `COLI_LOOKUP=1`: the chain hands a verify to the CPU, the recurrent state down and up
again. Debug-only switches: `Q38M_CHECK_WEIGHTS`, `Q38M_LAYER_CHECK`, `Q38M_DECLINE_EVERY`
([ENVIRONMENT.md](ENVIRONMENT.md)).

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

- Not wired into the engine: nothing in `qwen38` calls these kernels yet.
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
