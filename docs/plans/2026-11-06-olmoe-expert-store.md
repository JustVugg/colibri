# OLMoE as a second ExpertStore consumer

Continuation of https://github.com/JustVugg/colibri/discussions/1057, based on
upstream dev `5cfb55bd`. Serving transport work is tracked separately in #2654.

## Implemented boundary

The CPU decode and expert-grouped prefill paths acquire `ColiExpertView` through
`ColiExpertStore` and release each lease after the expert computation. The math
reads tensor views instead of the cache's `Slot` structure. The adapter borrows
OLMoE's existing cache: `Model` still owns its buffers, indexing, PILOT worker,
prefetch policy and memory-pressure handling. Existing disk-load failures remain
fail-fast; invalid keys return a cleared view without acquiring a lease.

Lookup/release retain the cache lock and busy count. Multiple lookups of one
expert hold distinct leases; release decrements exactly once and clears the view.
The existing prefetch eviction test now acquires its live slot through the shared
interface, checks that PILOT cannot replace it, then releases and permits eviction.

Views describe row-int8 data with separate float scales, native GGML blocks, and
CPU R16 interleaved blocks. `block_columns` carries the actual GGML held type,
including `GQ_*_R16`. The tensor enum and V4 APIs are unchanged.

This is a borrowed facade, not a new cache owner or configurable factory. Its
optional prefetch/stats/destroy callbacks are absent; callers must not invoke
absent operations. The model must outlive the facade and all views. No new
resource-owning destructor or Engine/Session lifecycle is claimed here. The
Vulkan expert tier and its upload/cache policy retain their existing paths.

## Reproduce focused acceptance

Build and run from the repository root:

```sh
make -C c olmoe tests/test_olmoe_expert_store tests/test_olmoe_segment_lifecycle tests/test_olmoe_moe_by_expert tests/test_olmoe_cache_index tests/test_expert_store_ops tests/test_olmoe_serve_framing -j4
OMP_NUM_THREADS=2 ./c/tests/test_olmoe_expert_store
OMP_NUM_THREADS=2 ./c/tests/test_olmoe_segment_lifecycle
OMP_NUM_THREADS=2 ./c/tests/test_olmoe_moe_by_expert
./c/tests/test_olmoe_cache_index
./c/tests/test_expert_store_ops
./c/tests/test_olmoe_serve_framing
```

The MoE test optionally appends raw output to `OLMOE_TEST_OUTPUT`. Use distinct,
new files on the baseline commit and adapter commit, with the same compiler,
hardware and thread count, then compare using `cmp`. It covers norm_topk off/on,
multiple prefill block sizes and available default/IDOT/FUSED3 execution modes.
The new store test separately checks row-int8, GGML Q8_0 and Q8_0_R16 metadata,
output against direct kernels, simultaneous leases, repeated release and invalid
keys. It does not establish correctness for every GGML quantization type.

`OLMOE_STORE_BENCH=1` on the store test reports resident lookup/release time for
the original slot API and shared view API (best of five, 200000 iterations).
This measures adapter overhead only, not model throughput or disk performance.

## Remaining shared-runtime work

- Keep the remaining GLM header/control codec work tracked in #2654.
- Establish an owning store lifecycle, including teardown with active leases,
  before changing common construction or introducing a backend factory.
- Reuse the existing `segment_runtime.h/.c` engine/session boundary and real
  family adapters. Its live-session close guard and snapshot capabilities already
  exist; ordinary CLI/serve integration and capability-contract gaps still need
  work. The ten-family conformance fixtures are synthetic state-topology tests,
  not proof of real-model acceptance.
- Keep native GGML and R16 layout identity explicit in any new backend adapter.
- Require separate hardware acceptance before migrating the GPU expert tier.
- Retain family-specific KV/recurrent state and unsupported snapshot/prefix
  operations explicitly when extending the existing session adapters.

## Local evidence for this change

On the same local CPU build, the before/after MoE artifacts were 1711104 bytes
and compared equal (`sha256: 24b71f7310ec9e1efb8059bb75aaa699337ce11503387e5f6c44fd18fabdeafb`).
The focused store test also passed ASan/UBSan with leak detection enabled.
One isolated resident microbenchmark measured 23.8 ns for slot lookup/release
and 31.9 ns for the view API. These host-specific measurements do not substitute
for end-to-end model throughput or GPU acceptance.

The production Segment teardown also releases the model-wide `ehit` matrix.
Before the fix, a 32-cycle synthetic resident-cache fixture leaked 640 bytes in
96 allocations, with LeakSanitizer stacks pointing to `ehit_mark`. The lifecycle
test calls the production session/store/engine callbacks, checks independent KV
buffers, and covers partial layer ranges and engines that never acquire an
expert. These are resident synthetic fixtures, not checkpoint-based inference.

## Checkpoint-based public adapter gate

A focused target reuses `test_segment_adapters_real.c` with only the OLMoE
registrar and object linked. The default all-family test remains unchanged.
Generate a new fixture directory (the generator refuses existing paths):

```sh
python3 c/tests/make_olmoe_segment_fixture.py --output /tmp/olmoe-segment-fixture
make -C c tests/test_olmoe_segment_real -j4
OMP_NUM_THREADS=2 c/tests/test_olmoe_segment_real olmoe /tmp/olmoe-segment-fixture 0 2 4 8
OMP_NUM_THREADS=2 c/tests/test_olmoe_segment_real olmoe /tmp/olmoe-segment-fixture 1 2 4 8
```

This standard-library-only generator writes deterministic synthetic merged-int8
weights and config, consumed by the production loader. The public runtime opens
full and partial layer ranges, creates sessions, performs prefill/continuation,
compares chained results with a full range, and restores snapshots. Corrupt and
truncated snapshots must fail without changing live state. The checkpoint is
synthetic: this is actual loader/adapter execution, not a trained-model accuracy
oracle or GPU acceptance.

Unlike the resident-cache lifecycle fixture, this path exercises config parsing.
LeakSanitizer exposed the missing `json_free` in `load_cfg`; releasing the input
buffer and string arena alone did not release the JSON tree. The loader now frees
the tree after copying its numeric configuration.
