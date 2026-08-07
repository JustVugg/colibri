# GLM-5.2 decode failure ledger

Date: 2026-07-31

This is the stop-doing list for GLM-5.2 744B decode work. It records ideas
that looked plausible but failed under controlled measurements, why they
failed, and the evidence required before anybody spends time on them again.

The default reference is the 6 x RTX 5090 host with dual-socket DDR5 RAM,
Colibri INT4 weights, greedy decode (`COLI_TEMP=0 DRAFT=0`), and the model
fully resident across VRAM and RAM unless stated otherwise.

## Rules

- Do not reopen a rejected direction because a microbenchmark is faster.
- Do not compare separate processes without an interleaved ABBA control.
- A compute experiment must report residency, disk reads, CPU expert
  bandwidth, output equality or quality, and the exact generated token count.
- “Conditional” means the feature may help a narrow workload; it is not a
  safe default.
- A new name, paper, or implementation does not invalidate the measurements
  below. The stated reopen gate must change.

## Rejected or quarantined directions

| Direction | Controlled evidence | Why it failed | Status and reopen gate |
|---|---|---|---|
| Whole-expert NUMA ownership | Interleaved controls: 8.84/8.92 tok/s and 64.74/64.89 GB/s CPU expert bandwidth. Node-local ownership: 6.16 tok/s and 30.97 GB/s, at 100% hit and zero disk. | With roughly 80% of routes on GPU, only 1-2 RAM experts remain per layer. Assigning each whole expert to one socket idles the other memory controller; interleaving lets one expert consume both. | **Rejected on this topology.** Reopen only when many CPU experts execute concurrently per layer, or when the memory topology changes. |
| One CPU task per expert / expert-parallel OpenMP | Earlier trials regressed about 23%; each expert obtained only about 2.5 GB/s. Row batching and persistent-team variants were neutral or negative. | There are too few CPU experts per layer. Splitting workers across experts sacrifices row-parallel bandwidth and NUMA/cache locality. | **Rejected.** Reopen only if CPU expert concurrency becomes materially larger. |
| Impact/cost-aware placement without a hard residency constraint | Fixed replay fell from a 7.46 tok/s frequency-fill median to 4.01 tok/s (-46.2%). It caused about 490-491 disk loads, 9.28 GB fetched, and 4.29-4.46 s of wait. A capacity-preserving variant measured 6.88 versus 7.26 tok/s; a held-out learned layout also produced a zero-disk regression of about 38%. | Optimizing an estimated CPU/GPU compute balance evicted useful resident weights and did not transfer across prompt classes. When full capacity is enforced, the policy largely collapses back to frequency fill. | **Rejected as implemented.** Reopen only with hard full-residency constraints, actual tier-latency costs, and category-held-out validation. |
| Cross-layer expert prefetch (`PREFETCH`, `PILOT`, learned coupling) | `PREFETCH=1` reduced median replay from 4.01 to 3.91 tok/s (-2.5%), while service/wait changed by less than 1%. Learned coupling produced only six nonresident hints and worsened misses and wait. | The predictable experts were already resident. Staging contended with PCIe/attention, so routing recall did not translate into avoided cold misses. | **Rejected for resident decode.** Reopen only on a genuine disk-miss workload and optimize measured cold-miss recall, not routing recall. |
| WarpDecode low-row CUDA kernel | Bit-exact isolated GLM-sized eight-expert group: block kernel 0.248 ms, WarpDecode 0.326 ms (about 31% slower). Earlier end-to-end wins tracked CPU bandwidth and machine-state drift. | The specialized mapping did not offset its scheduling and reduction costs on RTX 5090. | **Rejected on sm_120.** Reopen on a materially different GPU only after an isolated kernel win plus interleaved end-to-end controls with stable CPU bandwidth. |
| Fused ANS decompress + expert matvec | Same-process ABBA: materialize path 6.505 tok/s, fused path 5.285 tok/s (-18.8%), with identical predictions. The earlier cross-process +7.1% result was drift. | Per-row decode/control overhead cost more than the eliminated scratch materialization. | **Rejected.** Reopen only with a decode-native, vector-aligned encoding whose isolated fused kernel first beats materialization. ANS residency itself remains valid. |
| GPU shared-expert microkernels and GPU routed scatter-add in isolation | Shared-expert end-to-end run was 110.08 s versus a 94.05 s control; routed scatter-add was 135.27 s versus 94.05 s, despite faster isolated arithmetic. | Launches, synchronization, transfers, and reductions dominated the saved matrix time. | **Rejected in the current split layer.** Reopen only as a genuinely persistent/fused whole-layer path that removes the round trips. |
| GPU-resident residual ownership across a PCIe star | Parallel device dispatch helped (6.82 to 7.44 tok/s, +9.1%), but extending ownership across the layer did not. | Moving the roughly 95 MB layer state among peers serializes on the no-NVLink PCIe topology; host traffic was not the dominant cost. | **Reject residual/P2P expansion; keep parallel dispatch.** Reopen with NVLink or whole-layer fusion that eliminates transfers. |
| GPU zero-copy reads of RAM-resident experts | The preserved dual-lane prototype measured 4.24 tok/s for CPU-direct, but only 1.71 tok/s when a GPU kernel demand-read mapped host weights. Effective PCIe zero-copy bandwidth was about 2.8 GB/s, versus 56 GB/s for bulk pinned DMA. Later double-buffered DMA reached only parity (8.16 versus 8.25 tok/s); registration consumed roughly 0.57 extra RSS bytes per registered byte, while `cudaHostAlloc` broke NUMA interleave and reduced the CPU leg to 38 GB/s. | Fine-grained PCIe reads are latency-bound. Bulk DMA needs a large pinned store, but this 251 GiB host cannot register enough of the 191 GiB warm tier without losing memory capacity or NUMA placement; the offload fraction is too small to beat CPU contention. | **Rejected on this host.** Reopen only with 384-512 GiB RAM headroom, a driver/IOMMU configuration that removes registration inflation, or a coherent high-bandwidth CPU-GPU interconnect. |
| Static expert pruning / `EXPERT_BUDGET` as a lossless speed path | Every tested budget was either no faster or incoherent; the control is quarantined behind `EXPERT_BUDGET_EXPERIMENTAL`. | It reduces model computation by changing the model, not by removing redundant engine work; quality and routing behavior fail before a dependable speed win appears. | **Rejected as a lossless/default feature.** Reopen only as an explicitly lossy model-compression study with full quality evaluation and measured byte reduction. |
| Direct cross-expert common-basis extraction | Real weights from layers 3/10/30/50/77 and all 256 experts showed only 0.38%-0.69% shared mean energy, near the random baseline of 1/256. Pairwise cosine means were near zero; 90% energy needed roughly 192-224 of 256 bases. A 35% storage cut caused about 18%-30% sampled reconstruction error. | Independently trained experts do not contain a large aligned linear component that can be losslessly factored after training. | **Rejected for engine-only transformation.** Reopen only with training/distillation; that produces a derivative model. |
| Permutation-based expert neuron alignment | Across layers 3/30/77, nearest matched cosine was 0.246-0.253 versus a random nearest baseline near 0.245. | Apparent matches are the expected nearest-neighbour statistic, not reusable neurons. | **Rejected.** Reopen only if a new method beats a matched random control by a substantial margin and reconstructs held-out activations. |
| SERE-style functional expert substitution | Five diverse prompts, real hidden states, layers 3/30/77, all 256 experts, and 489,600 comparisons: mean nearest similarity was 0.08457/0.02133/0.07281; maximum pair similarity was 0.30321/0.06062/0.28433; no pair reached 0.5. | GLM-5.2 did not exhibit the strong functional expert redundancy required by SERE. Substitution here is lossy skipping with no redundancy signal. | **Rejected for runtime integration.** Reopen only if a much broader activation corpus yields stable cross-prompt clusters above 0.5. |
| MTP/speculative draft as a global default | Latest full-resident retest: `DRAFT=0` 8.90 tok/s; `DRAFT=1` 7.51 (-15.6%, 60% acceptance, +30.4% CPU rows); `DRAFT=3` 3.95 (-55.6%, 23% acceptance, +145% CPU rows). A predictable prompt previously gave D1 +9.2% at 91% acceptance. | Rejected drafts widen the expert union. Acceptance alone does not guarantee that saved forwards exceed added expert work. | **Conditional, never default.** Keep opt-in/guarded. Reopen for a stronger trained draft only when acceptance is near 90% and expert-union growth is bounded across prompt classes. |
| Adding GPUs without fixing placement/execution | vLLM-Moet TP4 measured 2.6 tok/s; TP2 x PP3 using all six GPUs measured 1.78 tok/s. | Extra aggregate VRAM did not compensate for per-rank residency, communication, replay, and serial pipeline latency. | **Rejected as a standalone remedy.** Reopen only with a materially different placement and parallel execution path under matched quality. |
| Repin/promotion policy sweeps | `REPIN=8`, `REPIN=32`, 64 swaps/round, and 8 swaps/round all regressed. A second prefill correction pass measured 5.86 tok/s; lazy demotion 4.77; GPU-to-RAM recovery 6.15. | More frequent movement pays synchronization overhead; less frequent movement leaves stale expert sets. Recovery costs approximately as much as page-cache rereads. | **Rejected around the tested operating point.** Current local optimum is 16 tokens and 16 swaps/round. Reopen only after the movement mechanism or workload changes. |
| CPU/GPU expert overlap through extra pthreads | 6.09-6.30 tok/s with no stable gain. | Extra workers compete with the 24 CPU workers while the GPU expert group is too small to hide enough wall time. | **Rejected implementation.** Do not confuse this with the existing async device dispatch; reopen only with an execution graph that removes worker contention. |
| Generic CPU/JIT tuning as the main breakthrough | Global NUMA interleave, huge pages, 12-core restriction, smaller VRAM reserve, normalized profiles, VNNI int4 x int8, and OpenMP restructuring were neutral or negative. AVX-512 showed 5.12 to 5.89 tok/s in a greedy A/B but changed accumulation order and eventually diverged; row blocking collapsed from a short benchmark win to 6.84 versus 6.83 tok/s over sustained decode. | The remaining path is memory-bound and sensitive to cache/turbo state; arithmetic microbench wins do not reliably preserve output or end-to-end throughput. | **Rejected as a blanket direction.** Reopen a specific kernel only with sustained ABBA, numeric-error/quality tests, and end-to-end gain. |
| Existing continuous batching as a throughput multiplier | Full-resident mux ABBA medians for 1/2/4/8 active sessions were 4.84/6.33/8.14/8.30 aggregate tok/s. A profiled eight-session run reached 9.05 tok/s, but CPU expert bandwidth was only about 33-35 GB/s versus 66.88 GB/s in the ordinary 8.90 tok/s single-stream baseline; the CPU side was 5.6-11.1x the GPU critical path. | The current multi-row INT4 CPU expert path does not convert shared expert weights into enough reuse and loses roughly half the effective RAM bandwidth. Batching raises throughput relative to its own slow mux S=1 path, then saturates at the existing single-stream ceiling. | **Current implementation rejected as a speed claim; serving mechanism retained.** Reopen after a dedicated multi-row expert kernel restores 60+ GB/s and pushes full-resident 4/8-session aggregate throughput beyond 8.9 tok/s. |

## Keep these results

These are not failures and must not be discarded while avoiding the directions
above:

- Keep the model fully resident whenever possible; disk avoidance dominates.
- Keep frequency/hotness-based expert placement as the current robust default.
- Keep selective NUMA interleaving for the small number of RAM experts.
- Keep parallel per-device dispatch and the existing CPU/GPU asynchronous path.
- Keep the ANS compressed resident tier; only fused consumption was rejected.
- Keep the continuous-batching scheduler and paged/ragged KV machinery as
  functional serving infrastructure, but do not claim a throughput gain until
  its multi-row CPU expert path passes the gate above. Keep prefix reuse and
  prefill acceleration for TTFT.
- Keep MTP available as an opt-in workload-specific feature, not an automatic
  speed claim.

## Remote evidence index

- `/data/test/glm52-cross-expert-structure-20260731.json`
- `/data/test/glm52-neuron-alignment-20260731.json`
- `/data/test/sere-calib-output/aggregate-5-summary.json`
- `/data/test/sere-calib-output/layer-003-aggregate-5.csv`
- `/data/test/sere-calib-output/layer-030-aggregate-5.csv`
- `/data/test/sere-calib-output/layer-077-aggregate-5.csv`
- `/data/test/mtp-retest-20260731/`

The temporary SERE calibration code lives at
`/data/test/colibri-sere-calib`. The latest-dev MTP retest source lives at
`/data/test/colibri-mtp-retest`. Neither is a production implementation.

## Remaining honest directions

1. A persistent or whole-layer expert kernel that removes launches, host
   round-trips, and reductions together, validated on the target GPU.
2. A stronger trained draft model whose acceptance and expert-union cost are
   jointly controlled.
3. A retrained/distilled derivative MoE designed for shared structure or
   locality; this is model research, not a post-training lossless rewrite.
4. Hardware that raises resident capacity and memory bandwidth without adding
   a worse communication topology.

The detailed experimental context remains in
`inference-paper-test-matrix-2026-07-28.md` and
`glm52-6x5090-2026-07-12.md`.

## Final software-ceiling conclusion

For the original GLM-5.2 744B architecture, lossless greedy single-stream
decode, and this 6 x RTX 5090 + dual Xeon Silver 4510 + 251 GiB DDR5 host, the
best current full-resident result is 8.90 tok/s:

- 9,335 experts in VRAM and 10,121 in RAM;
- 100% expert hit rate and zero disk traffic;
- CPU expert bandwidth 66.88 GB/s;
- approximately 44% expert matmul, 34% attention, and 22% remaining work.

The controlled failures in this ledger remove every identified path to a
general 1.5-2x engine-only improvement without changing the model, quality
policy, workload, or hardware. The remaining realistic engineering headroom is
incremental: roughly 10-25% in aggregate, with a practical stable target around
9.5-10.5 tok/s and an exceptional upper target around 11-12 tok/s. These are
planning bounds, not measured benchmark claims.

Material gains beyond that require changing a boundary condition: more VRAM,
substantially higher memory bandwidth/coherent memory, a trained route-aware
draft, or a retrained locality-aware MoE derivative. Future engine work should
therefore prioritize stability, automatic hardware planning, and cross-machine
performance portability unless a proposal demonstrates that it reduces bytes
per token or removes a measured whole-layer critical-path cost.
