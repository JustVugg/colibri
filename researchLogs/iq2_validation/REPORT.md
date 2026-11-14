# IQ2_XXS native Metal GEMV — real-model validation (Qwen3.6-27B-UD-IQ2_XXS)

Date: 2026-11-05 · Host: M4 (16 GB), macOS · Engine: `c/qwen36` built METAL=1
Model: `/Users/creemac/.lmstudio/models/unsloth/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-UD-IQ2_XXS.gguf` (local, untouched)

## 1. GGUF metadata (tools/gguf_inspect.c → gguf_inspect.txt)
- File: 8.910 GiB, 1 file, 866 tensors. arch `qwen35` (Qwen3.6 hybrid: DeltaNet + Attention,
  `full_attention_interval=4`), dense (0 experts): 64 layers, hidden 5120, MLP 17408, vocab 248320.
- Type histogram: F32 x456 (0.098), Q8_0 x1 (0.052), Q2_K x14 (0.330), Q3_K x2 (0.511),
  Q4_K x96 (2.109), Q5_K x1 (0.814: output.weight), **IQ2_XXS x222 (4.106 GiB)**, IQ3_XXS x64 (0.568),
  IQ3_S x5 (0.178), IQ2_S x5 (0.133).
- All IQ2_XXS blocks are the native 66-bytes-per-256-element form the new Metal GEMV consumes.

## 2. Native dispatch reaches the Metal GEMV — verified, not inferred
- `make iq2-xxs-metal-test METAL=1` (tests/test_iq2_xxs_metal.c): Metal results match the CPU IQ2_XXS
  reference (max_norm ≤ 1e-5 vs the CPU int8 kernel; ~4.4e-3 vs dequantize-to-F32 oracle) for small and
  large shapes; real GPU execution measured via GPUEndTime (warm GPU 0.32–3.28 ms for 5120×17408×{exc}). 
- Engine level (process sample, logs/iq2_validation/sample_metal_long.txt): decode stacks show
  `qwen_shared_experts_cpu → matmul_d → coli_metal_iq2_xxs_gemv_profile →
  -[_MTLCommandBuffer waitUntilCompleted]` with real AGX command-buffer frames underneath.
  Non-IQ2 tensors (Q4_K/Q2_K/Q3_K/Q5_K) stay on `gq_matmul` (CPU/OMP) in the same traces — no
  silent skipping or mis-routing of unsupported types.

## 3. Loading with COLI_METAL_IQ2_XXS=1
- Loaded successfully. Pipeline/table init did not print the "native IQ2_XXS GEMV unavailable"
  warning (dispatch is usable; threadExecutionWidth=32, grid/signs tables built).
- GGUF_CHECK=1 dry pass: OK, 866 tensors, 9.56 GB of weights, tokenizer ok (exit 0).

## 4–5. Short inference + CPU/Metal comparison (32-token greedy, prompt "The origins of algebra…")
- Generated text identical across runs (byte-identical stdout); a 64-token Metal run's first
  32 tokens are also byte-identical to the CPU run → no argmax drift over 96 forwards total.
- Last-token logits (248320 f32): max_abs 0.271, max_norm 1.25e-2, rms_norm 9.74e-3,
  argmax identical (token 310). Top-5 identical; ranks 6/7 swap (318/1330) — within the
  different accumulation order of GPU GEMV (dequant-multiply-add reorder); ordering swaps
  far outside the greedy decision margin.

## 6. Metrics (N_NEW=32 unless noted)
| run                        | load   | RSS after load | TTFT   | decode      | PEAK RSS |
|----------------------------|--------|----------------|--------|-------------|----------|
| CPU (IQ2=0)                | 3.1 s  | 3.75 GB        | 4.25 s | 0.93 tok/s  | 8.39 GB  |
| Metal (IQ2=1)              | 3.2 s  | 5.28 GB        | 3.81 s | **1.48 tok/s** | 8.95 GB |
| CPU (IQ2=0, repeat=metaloff)| 2.9 s | 4.80 GB        | 4.51 s | 0.86 tok/s  | 8.94 GB  |
| Metal 64-token             | —      | —              | 4.16 s | 1.53 tok/s  | 8.24 GB  |

- CPU vs Metal = **1.6×** end-to-end at this decode shape; per-GEMV (unit test): CPU 2.01 ms vs
  Metal 3.71 ms wall (GPU exec 3.28 ms, non-GPU dispatch/sync ~0.43 ms/call, ~7 GB/s effective
  weight-read rate) — the win comes from not serializing every GEMV into a 4-core CPU loop.
- CPU fallback: COLI_METAL_IQ2_XXS=0 run is bit-identical to the baseline (248320/248320 logits
  equal, max_abs=0); the dispatch contract (`return 0` → gq_matmul) is exercised for every
  non-IQ2 tensor in the same processes. The init-failure fallback ("native IQ2_XXS GEMV
  unavailable: table buffers or SIMD width unsupported") was not reachable on this device,
  since the pipeline creation succeeded — its CPU-fallback branch exists and is honored by
  the `return 0` contract but is not exercised on this Mac.
- Load time measured warm (the file had just been read; cold-cache not measured).

## Known limitations / honest notes
- 16 GB host: decode peaks ~8.4–8.95 GB RSS; load pages the GGUF in place (pread, nothing mapped),
  so system page cache adds beyond the engine's RSS numbers. Runs completed without memory errors.
- Metal path keeps IQ2_XXS weights in 16 KB-aligned registered buffers (RSS-after-load 3.75 → 5.28 GB)
  since those pages must stay GPU-visible.
- The 0.43 ms/call dispatch overhead (visible in the unit-test profile as "inferred non-GPU
  dispatch/sync") is the next optimisation target; per the run order given, tuning was not attempted.
- Pre-optimisation, post-verification: Metal dispatch + CPU fallback verified end to end on native
  IQ2_XXS GGUF blocks.
