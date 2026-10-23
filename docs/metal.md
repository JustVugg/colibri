# Metal backend (Apple Silicon, experimental)

On Apple Silicon the decode profile is matmul-bound, and unified memory removes
the PCIe copy tax that keeps CUDA's streaming experts on the CPU — so colibrì
has an opt-in Metal backend that runs the **routed-expert SwiGLU (batched,
zero-copy from the RAM slabs)**, the **fused decode attention** (full MLA layer
in one command buffer, S≤4), and **prefill's large GEMMs** on the GPU.
Decode is token-exact vs the CPU path. Prefill's large GEMMs run on the GPU in a
different accumulation order, so on **near-tie logits** they can occasionally pick a
different top token than the CPU — a floating-point ordering difference, not a kernel
bug (`make metal-test` passes the GEMM at ~3e-6 against a 1e-4 tolerance; see
[#622](https://github.com/JustVugg/colibri/issues/622)). It is invisible in normal use
but can surface in teacher-forced oracle comparisons on pathological 4-bit toy
containers. Set `COLI_METAL_GEMM_MIN=100000` to keep every GEMM on the CPU for
bit-exact prefill (`DEBUG_LOGITS=1` on a `TF=1` run dumps the top-5 logits and the
top1–top2 margin at each mismatch, so you can see how close the tie was).

`COLI_METAL_PREFILL=1` extends the fused attention to **prefill** (S>4): the whole
attention — projections, scores, softmax, value, output — runs on the GPU in one
command buffer instead of the CPU. On a 544-token prompt this cuts prefill attention
~4x (35.9 s → 9.0 s). It is **off by default**: like the prefill GEMM above, the GPU
accumulates in a different order and can pick a different top token on near-tie logits
(same [#622](https://github.com/JustVugg/colibri/issues/622) family), so a greedy stream
is not guaranteed bit-identical to the CPU — on natural prompts it stays consistent, on
pathological repetitive prompts an early token can flip. Turn it on when prefill latency
matters more than exact CPU parity; prompts past the single-dispatch thread cap fall
back to the CPU automatically.

```bash
cd c
make colibri METAL=1          # macOS only; no Xcode needed (shader compiles at runtime)
                              # any macOS SDK builds; the COLI_METAL_RESSET residency-set
                              # path needs the macOS 15 SDK and is compiled out below it
make metal-test           # standalone kernel/attention correctness vs CPU reference
COLI_METAL=1 COLI_MODEL=/path/glm52_i4 ./coli chat --ram 96
```

The release's `macos-arm64.tar.gz` has `colibri`, `inkling` and `kimi_k3` built
with `METAL=1`: `COLI_METAL=1` (`K3_METAL=1` for Kimi K3) turns the backend on
with nothing to build, and without it they run on the CPU as before.

Measured on an M4 Max (128 GB, warm cache, MTP on): CPU 0.30 → Metal
**0.42 tok/s (~1.4×)** (best config adds `DIRECT=1`; ~3× vs this machine's
first cold run). An M5 Max with a 46.9 GB learned pin reached **2.06 tok/s**
([#103](https://github.com/JustVugg/colibri/issues/103); see also the
[M5 Max performance report](METAL-M5MAX-PERF-REPORT.md)).

## Metal Performance Shaders (MPS) GEMM

The multi-row matmuls -- prefill's large GEMMs (`coli_metal_gemm`) and the resident-tensor
matmuls GLM-5.3 and Kimi K3 use (`coli_metal_matmul`) -- run on Apple's
**MPSMatrixMultiplication** whenever a call has at least `COLI_METAL_MPS_MIN` rows
(default 8). The built-in `mm_gemv` kernel computes one output element per simdgroup,
so at S rows it reads and decodes every weight row S times; the MPS path decodes the
quantized weight **once** into an f32 scratch on the GPU (`mm_dequant`, all formats
0/1/2/3/4/8) and lets MPS's tiled GEMM reuse each weight tile across all S rows. The
decode and the GEMM share one command buffer. Single-row decode calls are untouched.

| variable | default | effect |
|---|---|---|
| `COLI_METAL_MPS` | on | `0` keeps every matmul on the built-in kernels (pre-MPS behavior) |
| `COLI_METAL_MPS_MIN` | `8` | minimum rows S for the MPS path |
| `COLI_METAL_MPS_SCRATCH_MB` | `256` | f32 decode scratch; bigger weights run in output-column chunks |

It is built with `METAL=1` (MetalPerformanceShaders ships with macOS; nothing to
install). `make ... METAL=1 METAL_MPS=0` compiles it out. Any MPS failure -- device not
supported, allocation, command-buffer error -- falls back to the built-in kernel for
that call, so MPS never turns a GPU call into a CPU fallback. At exit the engine prints
`METAL-MPS: GEMM N | fallback M | gpu Xs`; `fallback` should be 0.

Numerics: f32 throughout (f32 weights after decode, f32 activations, f32 accumulate), so
the error budget is the same as `mm_gemv`'s; only the summation order differs, which is
the same near-tie class described above (#622). `COLI_METAL_GEMM_MIN=100000` still keeps
every prefill GEMM on the CPU for bit-exact runs. `make metal-test` checks every format
through MPS **and** the built-in kernel against the CPU reference, including multi-chunk
runs, and fails if the MPS counter does not advance (a silent fallback cannot pass).

Key design points: Metal's ~5 ms submit latency makes per-matmul dispatch a
loss — everything is batched into few command buffers per layer, and the
resident experts' GPU work is submitted *before* the missed experts' disk reads
so I/O and compute overlap. `COLI_METAL_GEMM_MIN` tunes the prefill GEMM row
threshold (default 16). Streaming, cache, MTP, DSA and the persistence formats
are unchanged; every GPU path falls back to the CPU per-block on any fault.
Numerics are dequant→f32-MAC (same as the CUDA tier); greedy outputs are
byte-identical to the CPU engine.
