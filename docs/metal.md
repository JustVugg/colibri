# Metal backend (Apple Silicon, experimental)

On Apple Silicon the decode profile is matmul-bound, and unified memory removes
the PCIe copy tax that keeps CUDA's streaming experts on the CPU — so colibrì
has an opt-in Metal backend that runs the **routed-expert SwiGLU (batched,
zero-copy from the RAM slabs)**, the **fused decode attention** (full MLA layer
in one command buffer, S≤4), and **prefill's large GEMMs** on the GPU.
Token-exact vs the CPU path.

```bash
cd c
make glm METAL=1          # macOS only; no Xcode needed (shader compiles at runtime)
make metal-test           # standalone kernel/attention correctness vs CPU reference
COLI_METAL=1 COLI_MODEL=/path/glm52_i4 ./coli chat --ram 96
```

Measured on an M4 Max (128 GB, warm cache, MTP on): CPU 0.30 → Metal
**0.42 tok/s (~1.4×)** (best config adds `DIRECT=1`; ~3× vs this machine's
first cold run). An M5 Max with a 46.9 GB learned pin reached **2.06 tok/s**
([#103](https://github.com/JustVugg/colibri/issues/103); see also the
[M5 Max performance report](METAL-M5MAX-PERF-REPORT.md)).

Key design points: Metal's ~5 ms submit latency makes per-matmul dispatch a
loss — everything is batched into few command buffers per layer, and the
resident experts' GPU work is submitted *before* the missed experts' disk reads
so I/O and compute overlap. `COLI_METAL_GEMM_MIN` tunes the prefill GEMM row
threshold (default 16). Streaming, cache, MTP, DSA and the persistence formats
are unchanged; every GPU path falls back to the CPU per-block on any fault.
Numerics are dequant→f32-MAC (same as the CUDA tier); greedy outputs are
byte-identical to the CPU engine.
