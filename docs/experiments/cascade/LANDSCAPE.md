# Transfer landscape (source inspection, not benchmark evidence)

Colibri base `ab98f29f`; donor `f84e14b1`; independently fetched llama.cpp
HEAD `16f090f92c0e2ac6dfddcaf93589a9b158c598c7`. No llama.cpp checkout is
embedded in Colibri: GGUF interoperability is not shared execution code.
The donor's old llama pin `288115...` returned HTTP 404 for the requested
sampling source; it is **not** credited as independently verified here.

| Pathway | Actual Colibri locus | Actual llama.cpp locus at fetched pin | Transfer boundary / existing solution |
|---|---|---|---|
| Loading | `c/family_registry.py`, `c/qwen36.c` GGUF loader, `c/gguf.h` | `src/llama-model-loader.cpp` | Validate tensor geometry/format; no loader change proposed |
| GPU kernels | `c/backend_cuda.cu`, `c/cuda_bd.cuh`, `c/qwen36_cuda_chain.h`; Metal `c/metal/qwen36.metal` | `ggml/src/ggml-cuda/ggml-cuda.cu:ggml_cuda_mul_mat` | Hardware/kernel specific; exact-float dispatch cannot be assumed for Q8 activation paths |
| Quantisation | `c/qwen36_tier.c`, GGUF row formats | model loader and ggml CUDA dispatch | Same Q5/Q6 name does not prove arithmetic/layout equivalence |
| KV/cache | `c/qwen36.c` ring, DeltaNet state, `COLI_KV_PREFIX` | `src/llama-memory.cpp`, `src/llama-context.cpp` | `Q36_KV_RING`, prefix reuse already exist; Dory recurrence aliases differ |
| Batching | `c/family_registry.py` bounded Q38 rows; Q36 chain/mux | `src/llama-context.cpp` `n_batch/n_ubatch` | Existing physical batch control; larger chunks trade cancellation latency against throughput |
| Memory/offload | `c/qwen36_tier.c` upload-at arena, `GPU_BACKENDS.md` | `common/arg.cpp` GPU layers; model loader mmap | Qwen already has arenas; Dory byte saving is not incremental recipient saving |
| Prefill/decode | Q36 chain, `serve_sample`, `serve_mux` | context decode and server context batching | Sampler screen excludes forward computation; no token/s inference claim |
| Context handling | Q36 attention ceiling 262144/default 8192 and ring rollback | context KV-unified/sequence controls | Match context/state, do not shrink workload to obtain gain |
| Concurrency | Q36 `KV_SLOTS`/`serve_mux` | `tools/server/server-context.cpp` slots | Research parallelism is not concurrent GPU timing |
| Scheduler | `c/openai_server.py:GenerationScheduler` | server context queued prompt/decode processing | Async enqueue timing is not completed-work latency |
| Warmup/unload | `c/openai_server.py` worker lifecycle/load failures | server context `handle_sleeping_state/load_model`, `--sleep-idle-seconds` | Cold/warm must be measured separately; no lifecycle patch proposed |
| Recovery | Q36 prefix/ring restoration; gateway `parse_load_fail` | server context destroy/reload after sleep | Reload and cancellation tests required before serving promotion |
| Sampling | `c/qwen36_sample.h`, private Inkling/Q38 routines | `src/llama-sampler.cpp` top-p/dist | llama uses `std::discrete_distribution`, not libc rand/CDF coupling; no drop-in sort/RNG equivalence |

Fetched source anchors: llama context lines 324–377 (flash attention, microbatch,
KV unified); server context 1138 (async timing), 1172–1187 (sleep/reload);
common args 1660 (microbatch), 2786 (GPU layers), 3633 (prompt cache);
llama sampler 245 (`std::discrete_distribution`), 1726 (top-p constructor);
CUDA 2215 (matmul). These source reads establish applicability only.

## Ranked frontier and cascade graph

1. M03 stable host radix: runtime mechanism, Q36 shared contract; low test cost,
   highest immediate confidence/coverage per effort. Test-only qualification first.
2. M01/M02 packed extraction: kernel/backend mechanism; conditional on actual
   formats 113/114 and exact-float route. Inventory first, GPU admission later.
3. Batch policy: runtime + architecture-family semantics. Existing knobs before
   patches; state/cancel gates before throughput.
4. Arena: backend + lifetime mechanism. Qwen already implements upload-at;
   close duplicate implementation, not the entire memory research line.
5. Q-norm-scale: model/arithmetic-specific; do not port Dory reduction trees to
   generic RMSNorm. Deferred absent target equivalence.

Dependency edges: provenance → route inventory → mechanism oracle → paired screen
→ model correctness → short/long serving → regression → reviewed PR → promotion.
Packed extraction → larger sampler time fraction → reassess M03, **not additive
speedups**. Arena capacity → possible larger batch/KV → cancellation/memory
requalification. Batch → kernel geometry → recheck packed resource limits.
Sampler → increased transient scratch → measure peak RAM before promotion.

No production improvement is accepted in this screen, so bottleneck reevaluation
remains a follow-on measurement, not an invented cascade of gains.
