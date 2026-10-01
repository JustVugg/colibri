# Independent MTP GPU head and resident CPU head scheduling

Follow-up to the CPU offload study: five-GPU MTP spent 20.011 of 24.148 measured seconds drafting, including 5.044 seconds in the head region. Target layers were fully GPU-resident. The previous `V4_MTP_GPU_HEAD=1` setting still depended on `V4_MTP_GPU_DENSE=1` through `g_v4ds_gpu_engine`, so turning off GPU dense execution also disabled the GPU head.

This change lets the explicit GPU head flag use an enabled GPU engine independently of drafter dense placement. The existing GPU head implementation and its failure propagation remain unchanged. GPU head execution also avoids allocating the unused host logits buffer. CPU-only engines still use the CPU path.

For a resident CPU head, the projection now schedules all vocabulary rows in one OpenMP parallel loop rather than starting a loop for each 64-row tile. Nonresident weights retain the bounded 64-row buffered-read path. Each output row retains its original accumulation order.

## Validation

- Full default CPU engine build passed.
- CUDA-enabled generation translation unit passed host C syntax checking. This is not a CUDA-linked build or hardware acceptance.
- 22 existing DSpark/build/oracle Python tests passed.
- Extracted before/after projection loops passed 24 bit-exact comparisons: resident and buffered paths, one and four OpenMP threads, and vocabulary sizes 1, 63, 64, 65, 129, and 8193. This checks projection logits, not the complete MTP model or GPU path.
- `git diff --check` passed.

Local validation artifacts are in `/home/Kei/colibri/result/dsv4-offload-head-20261001`.

No new full-model throughput or token-parity claim is made. The next hardware comparison should reuse the five-GPU fully resident target configuration, with CPU drafter stages and independent GPU head enabled, and include a head-disabled control. Retain the previous RAM-locking and timing protocol. GPU/CPU head arithmetic can change draft candidates, so acceptance and exact target-versus-MTP outputs must be checked again.

This does not address the larger CPU target-layer or CPU drafter-stage costs. Eliminating the entire previously measured head region would still leave only about 16.75 output tokens/s in the five-GPU MTP case, versus 43.69 with MTP disabled; that estimate is not a measured speedup.
