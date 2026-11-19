# The SYCL backend: colibri's CUDA backend on Intel GPUs

The units in this directory are the CUDA backend (`backend_cuda.cu`, `cuda_chain.cu`, `cu_xb.cu`)
migrated by dpct and compiled by oneAPI's icpx for Intel GPUs: Data Center GPU Max 1100 / 1550
(Ponte Vecchio), Arc Pro B60 / B70 and Arc B580 / B570 (Battlemage, Xe2), the Xe2 / Xe3 iGPUs
(Lunar Lake, Panther Lake) and whatever comes after them. They export the C ABI the engines link
against with `-DCOLI_CUDA` (`backend_cuda.h`'s `coli_cuda_*`, `cuda_chain.h`'s `ColiCudaChainOps`,
`cu_tier_ops.h`), so `qwen36.c`, `colibri.c` and the chains are not edited: `make qwen36 SYCL=1` is
the CUDA=1 / HIP=1 build with another compiler behind it. `MIGRATION.md` is the migration's
record (every dpct site and its disposition, the hand-edit list); this file is how to build and run
it, and how the device is chosen.

Nothing about the GPU is decided at compile time. The profile (`coli_sycl_profile.h`) is read from
the device when it is opened: its architecture, sub-group sizes, EUs, local memory, and what its
matrix engines (XMX) take; the warp-32-style kernels exist at sub-group 16 and 32 and the profile
picks one; the matrix-engine GEMMs run only where the device lists the shape. An architecture this
code has never heard of gets the generic profile from the queried values and runs.

## Model qualification on PVC

These checkpoint-specific results are from one Data Center GPU Max 1100.
The detailed gates, numerical references, settings and logs are in
[`MIGRATION.md`](MIGRATION.md); other device families still need their own ladder.

| Model | Current qualification | Record |
| --- | --- | --- |
| Qwen36 | Native, API, quality and colibench accepted; automatic GPU placement | Sections 14.4–20 |
| Qwen38 Flash Next | Native greedy NLL gate still fails; matched CPU/GPU stage diagnostics completed | Sections 21 and 24 |
| GLM5.3 | Native/full host pin, API and complete Light benchmark pass | Sections 22.5–22.7 |
| GLM5.2 | Native/full host pin, API and complete Light benchmark pass; test service closed | Sections 23–23.3 |

GLM results cover a hybrid CPU/GPU engine, not a full device chain. GLM5.2 uses
`MTP=0 DRAFT=0`, which excludes its additional MTP head. The qualified full-pin
configuration keeps 376 experts in GPU memory and 18,824 in host RAM, with a 450 GB
host ceiling; it is distinct from the limited-cache smoke. Neither GLM result
qualifies sparse DSA, speculative decoding or long-context numerical accuracy.

MIGRATION.md sections 25–29 also record a separate SYCL port of the DeepSeek V4 engine, made
against an earlier version of that engine; it is not part of this backend, and DeepSeek V4 builds
and runs here as `Makefile.deepseek-v4` has it.

## Building

The toolchain is oneAPI 2025.3 (icpx 2025.3.3, clang 21) in `intel/oneapi-basekit:2025.3.2-0-devel-ubuntu24.04`;
no icpx is needed on the host. From the repo root:

```
docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp -v $PWD:/src -w /src/c --entrypoint bash \
    intel/oneapi-basekit:2025.3.2-0-devel-ubuntu24.04 \
    -lc 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; make qwen36 colibri sycl-test SYCL=1 ARCH=x86-64-v4'
```

| target | what |
| --- | --- |
| `make qwen36 SYCL=1`, `make qwen38 SYCL=1`, `make colibri SYCL=1` | the engines, as CUDA=1 builds them: gcc compiles the C, icpx the SYCL units, the gcc link takes the SYCL objects |
| `make sycl-compile SYCL=1` | every SYCL object and the device link, nothing linked into an engine (CI without a GPU; `make sycl/coli_sycl_profile.o SYCL=1` for one unit) |
| `make sycl-host-sync-test` | host-only synchronization tests: ready/delayed word, cancellation, and state lifetime; plain C++, no device |
| `make sycl-profile-test` | the host-only test of the profile's decisions (`tests/test_sycl_profile.cpp`): plain C++, no SYCL runtime, runs anywhere; no SYCL=1 needed |
| `make sycl-test SYCL=1` | that, plus the GPU tests built (not run): `cuda_chain_test`, `backend_cuda_test` (the migrated ones, under their cuda-test names in `c/`) and every other `sycl/tests/*.cpp` as `sycl/tests/<name>` (the dispatch test, the XMX test) |
| `make sycl/tests/test_sycl_fp8_experts SYCL=1` | FP8 dense/grouped expert CPU oracles at small, tail and Qwen38 dimensions; async and row invariance. Run with `--device N`, test both SG widths and `COLI_CUDA_F8_WARP=1/0` under the host guard. |
| `make tests/export_qwen38_fp8_stages SYCL=1 ARCH=x86-64-v4` | CPU-only diagnostic fixtures from Qwen38’s actual vector/table math. Run the exporter without GPU devices; the original native quality gates still apply. See section 24.1. |
| `make tests/export_qwen38_fp8_native SYCL=1 ARCH=x86-64-v4` | CPU-only stage capture from selected native weights and activations, using the actual loader scale conversion. Requires a validated input packet; see section 24.2. |
| `make sycl/tests/test_qwen38_fp8_stages SYCL=1` | Matched CPU/GPU stage diagnostic reader. Host parsing and guarded GPU checks pass; native accuracy remains held. See sections 24.3–24.4. |
| `make sycl-chain-check SYCL=1` | runs `cuda_chain_test` (needs a card: the runtime container, below) |

Variables, all optional:

| variable | default | meaning |
| --- | --- | --- |
| `ARCH` | `native` | the C engine's `-march`: `x86-64-v4` (AVX-512) for the Xeon hosts of a Max or a B60, `x86-64-v3` for a host without AVX-512 |
| `SYCL_DEVICES` | `pvc,bmg` | ocloc's `-device` list: the GPUs that carry native code in the binary (ahead of time). Any other Intel GPU still runs: the binary also carries SPIR-V and the driver compiles it on first use. `SYCL_DEVICES=pvc` for one card, `pvc,bmg,lnl,ptl` for more |
| `SYCL_TARGETS` | `spir64_gen` | `spir64` builds SPIR-V only (no ocloc, every device JITs): a faster build for a check |
| `SYCL_IEEE_DIV_SQRT` | `0` | AOT build option: `1` requests correctly rounded fp32 division and square root from IGC. Requires `SYCL_TARGETS=spir64_gen`; changing it rebuilds the objects. Experimental, qualification in MIGRATION20; default remains the accepted OpenCL precision |
| `SYCL_DNNL` | `0` | `1` builds `sycl/coli_sycl_dnnl.cpp` (a prompt chunk's attention through oneDNN's fused attention, taken with `COLI_SYCL_DNNL_ATTN=1`) and links the toolkit's `libdnnl` (setvars' `DNNLROOT`; the runtime image's redist carries `libdnnl.so.3`). Changing it rebuilds the objects |
| `ICPX` | `icpx` | the compiler |
| `ICPXFLAGS` | see the Makefile | `-O3 -std=c++17 -fsycl <targets> -fp-model=precise -fPIE ... -DCOLI_GPU_PORTABLE_ONLY=1`. `-fp-model=precise` matters: icpx defaults to fast math and the backend's fmt=8 guard refuses it by design |

How the link works (the one mechanical difference from CUDA=1): `icpx -fsycl -c` makes fat objects
(host code and device IR in one `.o`), and one `-fsycl-link` step over all of them AOT-compiles
every kernel for `SYCL_DEVICES` and writes `sycl/coli_sycl_dev.o`, a plain object holding the
device images and their registration with the runtime. `CUDA_OBJ` is the fat objects plus that
one, and the engines' gcc link lines take them unchanged, with `-lsycl -lstdc++` and icpx's own
runtime (`-lintlc -limf -lsvml -lirng`: the optimizer emits `_intel_fast_memcpy` into the host
code, and the math libraries are what an icpx-driven link would add; the runtime image has all
four under `/opt/intel/oneapi/redist/lib`). The gcc-linked
binary carries the pvc and bmg images (`readelf -S` shows the `.tgtimg` section; the
`__CLANG_OFFLOAD_BUNDLE__sycl-spir64_gen` sections live in the fat objects and are consumed by
`-fsycl-link`; `readelf -d` shows `NEEDED libsycl.so.8` and `libintlc.so.5`); the first run on a
card is what confirms it registers them. If it ever did not,
the fallback is an icpx-driven final link (`icpx -fsycl <targets> qwen36.o ... -lgomp`), which
was also checked to link.

`SYCL=1` refuses `CUDA=1` / `HIP=1` (each builds the one ABI). `VK=1` stays independent, as it
is of CUDA=1.

### The image

`Dockerfile.sycl` builds the engines in the basekit and copies them, the launcher (`c/coli` and
its files, as `docker/Dockerfile.slim` does) and the tests into `intel/oneapi-runtime:2025.3.1-0-devel-ubuntu24.04`,
which carries libsycl.so.8 and the Level Zero adapter under `/opt/intel/oneapi/redist/lib`
(compute runtime 25.18, the driver stack of the host this was written on; the host itself has only
runtime 2026.1 with libsycl.so.9, so the binaries run in the container, not on the host). One thing
the runtime image lacks is copied from the basekit stage: `libumf.so.1` (and its `libhwloc.so.15`),
which both UR adapters need; without it `ldd libur_adapter_level_zero.so.0` says `not found`, no
adapter loads and the engine sees no GPU on any card (`COLI_SYCL_LIST=1` prints `0 GPU devices`
with the card present, the same words as a missing `/dev/dri`).

```
docker build -t colibri-sycl -f c/sycl/Dockerfile.sycl .
docker run --rm --device /dev/dri --group-add $(getent group render | cut -d: -f3) \
    -v /nvme/qwen36_i4:/model colibri-sycl info
```

The container's user is not root: `--device /dev/dri` gives it the card nodes, `--group-add` the
host's render group. One card only: `--device /dev/dri/card1 --device /dev/dri/renderD129`
(the pair of one PCI device; `ls -l /dev/dri/by-path` maps them).

## Defaults measured on a Data Center GPU Max

A switch the environment leaves unset takes the value measured best where there is such a measurement (a Max 1100,
2026-11-18/19); an explicit value always wins, and `COLI_PLATFORM_DEFAULTS=0` keeps every code default (all three
groups below). The backend's rows apply only where every device the process opens is a Max (with devices of different
architectures, the code's defaults); they leave the output bits as they are (texts identical to the switch off), but
for oneDNN's attention, whose texts were identical in every measured class but whose bits are not (its row says how
its quality was checked).

| where | switch | unset on Max cards (code default elsewhere) | measured |
| --- | --- | --- | --- |
| backend profile (`coli_sycl_tuned_default`) | `COLI_CUDA_ATTN_GROUPED_PLAIN` | `32` (`0`) | the five rows together: Qwen3.6 first token of a 32K prompt 119.9 → 26 s, eight conversations 2.63 × one |
| | `COLI_CUDA_ATTN_GROUPED_VEC` | `2` (`0`) | |
| | `COLI_CUDA_DNREC_PROMPT_SG` | `16` (the context's) | |
| | `COLI_CUDA_GEMV_XPAD` | `2` (`1`) | |
| | `COLI_CUDA_MUX_BATCH` | `15` (`0`) | |
| | `COLI_TRUNK_PROBE` | `0`: no start-up race against the CPU (`1`) | a start beside a busy card fell back to the CPU, with another text, in 4 of 8 pairs of starts |
| | `COLI_SYCL_DNNL_ATTN` | `1` (`0`), in a `SYCL_DNNL=1` build only, head size 256 only: a prompt chunk's attention through oneDNN's fused attention, its first compile off the request's path. **The attention's bits change** (float16 Q, P and O) | Qwen3.6 first token of a 32K prompt 25.99 → 11.51 s, 16K 9.07 → 5.89 s, 3K, 21 tokens and decode unchanged; texts identical to the switch off in every class; the prompt-path loss (8 448 prompt tokens, 512 scored) 2.759443 against the grouped kernel's 2.761657 (limit +0.5 %) |
| `qwen38`, SYCL build, where its FP8 expert tier will start (`COLI_CUDA=1`, the native FP8 checkpoint, no GGUF, no int4/low-bit sidecar) and every card is a Max (row `Q38_SYCL_HYBRID`); not under an explicit `COLI_CUDA_CHAIN=1` or `Q38_DN_GPU=1` | `COLI_SYCL_F8_CPU_ORDER=4`, `COLI_SYCL_F8_HOST_SILU=1`, `COLI_SYCL_SILU_CPU=1`, `Q38_TIER_RANK_ORDER=1`, `Q38_TRUNK_GPU=0`, `Q38_KV=f16`, `Q38_MTP=0`, `COLI_LOOKUP=0` | the hybrid the fork qualified (greedy NLL 0.3030, human 2.6323) with the CPU-order kernels `4` and no drafting. **These change the bits** against the code's default (towards the CPU's: the default, the trunk on the card, failed that gate at 0.3288); on other cards the CPU-order kernels may not run (they need FP64), so nowhere else | warm decode 16.8 → 19.1 tok/s against the switches set by hand in the qualified environment, the same bits as those |
| | `COLI_CUDA_EXPERT_ARENA=2`, `Q38_PREFILL_BATCH_ROWS=128`, `Q38_PREFILL_WORKSPACE_MIB=2048`, `Q38_PREFILL_GPU=1`, `Q38_PREFILL_GPU_BALANCE=1` | the tier settings those were measured with, and a prompt chunk's resident experts on the card at the measured split | first token of a 3 080-token prompt −9 %; four conversations +23 % with the kernels above |
| backend profile, a start's load (the qwen36 engine on the card, whatever model it runs; measured with Qwen3.6) | `COLI_CUDA_EXPERT_ARENA` | `2` (`0`): the native int4 experts in one device arena a card, staged through page-locked buffers | the three rows with the pool below: expert warm start 9.5–11.5 s → 1.25 s on a restart (4.0 s on the first start after the page cache is empty), trunk phase 4.8 → 1.1 s with the file; texts and both losses identical |
| | `COLI_EXPERT_PAGECACHE` | `1` (`0`): the experts left in the page cache once read (on the measured host 16.7 GB of reclaimable cache) | |
| | `COLI_CUDA_DENSE_CACHE` | `1` (`0`): the int8 trunk read from `canonical.qc` beside the model or named by `QWEN_DENSE_CACHE` (made once by `qwen36-canonical`; without one the trunk is converted as before) | |
| every engine `coli` starts for the measured Qwen3.6 MoE on Max cards (`serve`, `chat`, `run`, ...) | `UR_L0_USM_ALLOCATOR`, `SYCL_PI_LEVEL_ZERO_USM_ALLOCATOR` | `1;32M;device:4M,4,256M` (the runtime's own), both names, unless either is set: the runtime's pool for small device allocations (the gateway sets it: the runtime reads it when it starts) | warm start 11.5 → 3.9 s alone |
| `coli serve` (and `plan`, `doctor`, `tune`) | `--kv-slots` | `8` for the Qwen3.6 MoE measured (`qwen3_5_moe`, hidden 2048, 256 experts) on the card with a SYCL engine where every Intel GPU is a Max (`COLI_KV_SLOTS`, else `1`) | 207 tok/s at eight requests against 79 for one, every text the request's alone. With more than one slot the engine drafts nothing, keeps no prompt-end snapshot, parks nothing and reuses no attention-only prefix (one request alone lost 0–4 % on the prompts tried), and the automatic context is planned for eight conversations |

The engine prints each value it takes (`[SYCL] <switch> unset: ...`, `[qwen38] SYCL defaults ...`, `coli serve:
--kv-slots 8 ...`). The documented multi-CCS mode (`ZEX_NUMBER_OF_CCS=0:2`, a user-space variable) is not a default: two
gateways on one card in that mode served sixteen requests at 285 tok/s against one gateway's eight at 208, with every
text the request's alone, but one engine on half a card has a prompt pass about 1.8 × longer; it is a deployment choice
for more than eight users a card.

## Run-time knobs

The engine's own switches are unchanged (`COLI_CUDA=1`, `COLI_GPU` / `COLI_GPUS`, the tier's,
`docs/cuda.md`, `docs/qwen36-cuda-tier.md`): a device ordinal there is a position in the GPU list
this backend enumerates, Level Zero's GPUs in PCI order (the OpenCL view of the same cards is not
listed twice). These are the backend's:

| variable | values | meaning |
| --- | --- | --- |
| `COLI_TRUNK_PROBE_PIPE` | `0`, `1` | Qwen3.6 placement probe: default `1` in a SYCL build with the chain enabled, timing ten resident GEMVs per completion with distinct layers' weights. `0` restores synchronous per-matrix timing. `COLI_TRUNK_PROBE=0` still skips probing. `coli serve --gpu 0` supplies the GPU selection; VRAM allowance, placement and chain execution retain their automatic defaults |
| `COLI_SYCL_LIST` | `1` | print every GPU's profile (device, profile, sub-groups, EUs, SLM, memory, matrix shapes, platform, driver) at the first call into the backend and exit 0. No kernel runs. The first thing to run on a new box |
| `COLI_SYCL_SG` | `16`, `32` | the sub-group the warp-32-style kernels run at, where the device offers it (otherwise logged and ignored). Default 32 on pvc, 16 everywhere else |
| `Q38_TIER_RANK_ORDER` | `0` (default; `1` in qwen38's Max-card defaults, above), `1` | Qwen38 host-side diagnostic for native FP8 row decode: copy GPU outputs into owned rank slots and merge CPU/GPU experts in router rank order. Shared expert remains last. Only the native FP8 tier uses it; Vulkan, int4/GGUF and default collection are unchanged. Host/sanitizer/tiny GPU gates pass, but with both arithmetic knobs1 native greedy NLL0.3122 still exceeds0.305221. Not promoted alone (MIGRATION24.9–24.10); with the trunk on the CPU and the host SiLU — the hybrid of qwen38's Max-card defaults — the gate passes (0.3030 / 2.6323). |
| `Q38_BOUNDARY_TRACE`, `Q38_BOUNDARY_ROW`, `Q38_BOUNDARY_LAYERS` | unset (default), directory / zero-based row / `lo-hi` | Qwen38 host-layer diagnostic: save one selected row at attention, residual and MoE boundaries. The output directory must already exist; files are created exclusively and an existing name stops the run. Rows count calls per layer, including prefill; compare against `Q38_ACT_DUMP` to establish alignment. This tracing may affect scheduling and does not qualify a model. |
| `Q38_BOUNDARY_STATE` | unset (default), `1` | With the boundary trace enabled, additionally save the selected DeltaNet row's host recurrence, convolution ring and intermediate projections. Supported only for a single-row, host-owned, non-multiplexed DeltaNet call; unsupported calls report a skip. See MIGRATION30. |
| `COLI_SYCL_F8_CPU_ORDER` | `0` (default; `4` in qwen38's Max-card defaults, above), `1`, `2`, `3`, `4` | FP8 accumulation matching Qwen38's eight CPU FMA lanes, hsum tree and ascending double block sum. Requires FP64 and the warp family; dense input widths and both grouped widths must be multiples of 128. Other shapes retain the original path. `COLI_CUDA_F8_WARP=0` also restores the original family. SiLU and routing are unchanged; native quality is qualified only in the hybrid of qwen38's Max-card defaults (the trunk on the CPU, the host SiLU, the rank order: 0.3030 / 2.6323). `2`: the expert groups' kernels do the same operations in the same order with every lane at work (a group of eight lanes an output row, its weights decoded once for the rows of a pass), so `1`'s bits faster; the dense path keeps `1`'s kernel; the host SiLU and the other switches treat `2` as `1`. `3`: the same operations in the same order, an output row's 128-blocks SG/8 at a time (four at SG 32, two at SG 16; eight lanes a block, lane 0 adding them in ascending order), `1`'s grid: a decode step's critical path a quarter of `1`'s at SG 32. `4`: `3`'s gate/up kernel with `2`'s down kernel. |
| `COLI_SYCL_SILU_CPU` | `0` (default; `1` in qwen38's Max-card defaults, above), `1` | Experimental stable CPU-form sigmoid/SiLU using double exp rounded to float and rounded division. Applies to FP8 MLP/grouped experts with both widths divisible128, and explicitly to the generic public `pipe_silu_mul` operation used by stage diagnostics. Requires FP64 and the warp family; WARP0 restores the original expression. Other fused format epilogues and routing are unchanged. Separate from CPU-order dots; SG16/32 operator and stage gates pass, alone, native Qwen38 quality remains rejected (MIGRATION24.7–24.8); in the hybrid of qwen38's Max-card defaults the gate passes. |
| `COLI_SYCL_F8_HOST_SILU` | `0` (default; `1` in qwen38's Max-card defaults, above), `1` | Experimental grouped-FP8 correctness path: with `COLI_SYCL_F8_CPU_ORDER=1`, FP64, warp mode and both widths divisible by 128, run GPU gate/up, copy to host for the process's `expf`-based SiLU, then copy hidden values back for GPU down. This synchronizes transfers and may be slow. Other formats and unsupported shapes retain their usual path. Qwen38 CPU-trunk/GPU-expert fixed greedy/human scores and isolated CPU-matched API checks passed with rank-order merging; full GPU chain requires separate validation. See MIGRATION31. |
| `COLI_SYCL_MM_ROWS` | `auto` (default), positive row cap, `0` | Bound legacy dense matmul launches to `INT_MAX` global work-items by rebasing whole rows. Positive caps can force smaller batches for tests; `0` restores the unsplit diagnostic path and is unsuitable for oversized shapes. Per-row arithmetic is unchanged. Applies to host matmul, expert MLP, async fallback and pipeline GEMM; qualification in MIGRATION22 |
| `COLI_SYCL_XMX` | `0`, `1` | the matrix engines off, or on where the device lists the kernels' tile, f16 8x16x16 (`1` on a device that does not is logged and stays off). Default: on when that shape is listed and sub-group 16 is offered; `0` brings back the migrated warp and block kernels on every prompt-row site |
| `COLI_SYCL_PROFILE` | `pvc`, `bmg`, `lnl`, `ptl`, `arl`, `mtl`, `dg2`, `wcl`, `xelp`, `gen9`, `generic`, each optionally `-sg16` / `-sg32` | take that architecture's decisions (and its arch / gen tags, which the kernels key on) on this device; the measured values stay the device's. An unknown name is logged and ignored; `COLI_SYCL_SG` wins over the suffix |
| `COLI_SYCL_PROFILING` | `1` | create the queues with profiling (event timestamps): what `cu_xb`'s elapsed times need; off by default, since Level Zero timestamps every command of such a queue |
| `PROF` | `1` | Native `colibri` SCORE also reports GPU residency, expert groups, resident expert requests and GPU critical time on stderr after scoring, without changing the SCORE stdout protocol. Large batches can use individual MLPs and have no grouped calls; the execution-time report covers that path. In native `colibri` multiplexed API serving, the per-turn profile starts after prefill: a one-token response has an empty decode profile, so its zeros do not measure prefill GPU work or I/O. These are software counters and timers, not hardware metric collection. |
| `COLI_SYCL_FRAME_PROF` | `1`, `2` | the frame profile (`MIGRATION.md` section 14): profiling queues as above, and every command the chain and the backend submit on the compute queue is timed from its event once the frame has been waited for, booked per op kind (a matmul by its shape and format) and per frame (device span, busy time, the idle between consecutive commands, the host's submit and wait); frames are bucketed by their command count (a decode token's frames share one) and the modal bucket's table is printed at the chain's shutdown as `[SYCL] frame profile`. `2` also prints one frame command by command (kind, start, duration, the idle before it). Off by default: the hooks cost one flag test. Reading ~900 events a frame costs host time after the wait, so the wall a token under it is not the production wall; the device times are |
| `COLI_SYCL_XMX_TM` | `8` (default), `16` | the M of the matrix engines' tile in the XMX GEMM family: 16 takes the compound f16 16x16x16 shape where the device lists it (pvc, bmg); read once by `coli_sycl_xmx.cpp` |
| `COLI_SYCL_XMX_ATTN` | `0` (default), `1` opt-in | plain causal prompt attention with f16 XMX Q/K/V/probabilities and f32 online softmax; hd64/128/256, SG16, both cache formats. Declines windows/sinks/rings/selection lists; tiny/full and chain/backend gates pass, but K32/K64 are slower; keep off. Model quality unqualified (MIGRATION18) |
| `COLI_SYCL_XMX_ATTN_K` | `32` (default), `64` opt-in | key block for XMX attention; 64 needs about71KiB SLM at hd256. K32 was slower than fallback; K64 passes correctness but is slower too; no adoption (MIGRATION18). Attention remains off by default |
| `COLI_SYCL_DNNL_ATTN` | `0` (code default; unset on Max cards in a `SYCL_DNNL=1` build: `1` for head size 256, the defaults above), `1`, `2` | a `SYCL_DNNL=1` build's plain causal prompt chunks (from `COLI_CUDA_CHAIN_FLASH` rows, a float16 cache starting 16-byte aligned, hd 64/128/256, a device with the matrix engines; no window, sinks, ring or selection list) through oneDNN's fused attention (oneDNN 3.9.1: its micro-kernel `sdpa`), asked before `COLI_SYCL_XMX_ATTN`'s kernel; float16 Q, P and O and a float16 scale (exact for hd 64 and 256), so not bit-identical to the grouped kernel. A cache starting 6 bytes into its buffer gave wrong results without an error in oneDNN 3.9.1, hence the alignment rule; a head layout oneDNN refuses once is not asked again. Kernel level on a Max 1100: 0.75 ms against 6.88 at 2 048 rows; one partition per exact chunk shape (about 0.55 s the first in a process, about 0.5 ms each later one, 64 kept a device). `1`: that first compile runs on a thread of its own and the chunks that come meanwhile keep the grouped kernel (a short prompt in a fresh process never waits); `2`: compiled on the caller, every eligible chunk from the first (the tests and the quality gates). A build without `SYCL_DNNL=1` says once that it is not there |
| `COLI_SYCL_XMX_MOE_TM` | `8`, `16`; unset inherits `COLI_SYCL_XMX_TM` | grouped-MoE-only f16 tile override; 16 requires the listed compound shape. Dense GEMMs and decode GEMV stay unchanged. First-use log reports the actual tile; all-TM16 regressed gate/up; default unchanged (MIGRATION17) |
| `COLI_SYCL_XMX_MOE_DN_TM` | `8`, `16`; unset inherits MoE/global tile | down-projection-only override, applied after the all-MoE choice; 16 requires the listed compound shape. Separates the down kernel’s measured preference from gate/up. Down16 improved microbench12–16%, but engine TTFT ranges overlap; default unchanged (MIGRATION17) |
| `COLI_SYCL_XMX_S8` | `0` (default), `1` | experimental rowwise int8 activations for int8 trunk prompt GEMMs (more than 8 rows, K multiple64 and <=131071, listed s8 XMX shape). Changes prompt numerics; Qwen3.6 failed PPL (0.227469 > 0.2245), so keep off. Decode/verify and grouped experts unchanged. `0` restores f16 activations; evidence in MIGRATION16 |
| `COLI_CUDA_GEMV_SG` | `16`, `32` | int8 GEMV width, if offered. Default on PVC: 16 for 1024–2048 outputs, the profile width otherwise (32 normally); other architectures keep the profile width. Chosen by matrix shape, never row count, so verify rows and one-row calls share a reduction tree |
| `COLI_CUDA_GEMV_XS` | `0` (default), `1` | int8 GEMV at one row (rows and W4 kernels): `0` reads x from global memory without staging/barrier; `1` restores SLM staging. Multi-row calls keep staging, with identical values and lane order |
| `COLI_CUDA_GEMV_W4` | `0`, `1` | default on for every architecture after PLAN6 verify-row fallback. `1` uses four outputs per sub-group and its shape-based split-K; `0` uses one output per sub-group. The choice applies to every row count, preserving verify/decode bit equality |
| `COLI_CUDA_MOE_SG` | `16`, `32` | expert gate/up and down GEMV width, if offered: default 16 on PVC, profile width elsewhere. The lane rule gives 8 gate/up lanes and 2 down lanes for Qwen3.6 int4-g64. Sum, sort and XMX prompt kernels are unchanged |
| `COLI_CUDA_DNREC_SG` | `16`, `32` | width of the decode/verify rows kernel below `DNREC_MIN`, if offered: default 16 on PVC, profile width elsewhere. Prompt kernels continue using the profile width (`COLI_SYCL_SG`) |
| `COLI_CUDA_DNREC_DEC` | `1` (default), `0` | use `cc_dnrec_rows_k` below `DNREC_MIN` when `DNREC_ROWS` is enabled; `0` restores `cc_dnrec_k`. Prompt dispatch is independent |
| `COLI_CUDA_ATTN_TRACE` | `0` (default), `1` | Optional host-only once-per-path-and-row-class success message for contiguous or ragged MLA GPU projection when `COLI_CUDA_ATTN=1`; no dispatch or arithmetic change. Tiny GLM checks passed; native quality and speed remain unqualified. See MIGRATION32. |
| `COLI_CUDA_ATTN_SG` | `16`, `32` | both stages of block attention, including GQA and window/sinks, at the same width: default 16 on PVC if offered, profile width elsewhere |
| `COLI_CUDA_DNREC_MIN` | `8` | the rows from which the DeltaNet recurrence takes the prompt dispatch (split prep/recurrence/norm, or rows with `COLI_CUDA_DNREC_SPLIT=0`). Below this threshold, `DNREC_DEC` selects the decode row kernel. `1` also sends decode through prompt dispatch |
| `COLI_CUDA_GEMV_XPAD` | `1` (default; unset on pvc: `2`), `0`, `2` | the int8 GEMV kernels that take several activation rows a block (verify rows, the rows of several conversations): `0` keeps x unpadded in local memory for every such block; `2` only for a block of four rows of an unsplit matrix whose padded rows pass 32 KiB (four rows of 2048 inputs: 36.1 KiB padded, 32.0 unpadded). Same values and sums; a one-row call is untouched; a matrix whose padded row does not fit keeps its kernel |
| `COLI_CUDA_MUX_BATCH` | `0` (default; unset on pvc: `15`), mask `1`, `2` | a decode step of several conversations (`KV_SLOTS` > 1): the conversations' DeltaNet convolutions (`1`) and recurrences (`2`) as one launch a layer (`cc_dnconv_mx`, `cc_dnrec_mx`: work-group (.., j) runs the one-row kernel's body on conversation j's state). Same bits a conversation; `4` a full layer's attention over float16 mirrors as one call (`cc_attn_h_mx`), `8` the step's K and V rows into the mirrors as one launch a cache (`cc_copy_regions_h_mx`); `15` all four |
| `COLI_CUDA_MUX_BATCH_MIN` | `3` | the number of conversations in a step from which `COLI_CUDA_MUX_BATCH` gives them as one launch; below it a launch a conversation, as without the switch (two conversations measured 1.6 % slower as one launch on a Max 1100, four 7.4 % faster) |
| `COLI_CUDA_DNREC_CFG` | `162` on PVC, `84` elsewhere | split prompt recurrence: P threads over C columns, encoded as P*10+C. `84` restores the previous default. At the unchanged SG32, `162` reduced time per row by 29% at both 256 and 2048 rows (MIGRATION 14.4) |
| `COLI_CUDA_ATTN_DEC_MIN` | `1` (default) | plain decode/verify rows split from the first position; `128` restores the old threshold. Selection-list rows always keep 128 positions and 64-position blocks. The window/sinks threshold also stays 128 (section 14.4) |
| `COLI_CUDA_ATTN_DEC_L` | `64` | the first block length of that split (16, 32, 64, 128 or 256; doubled while the blocks would exceed 1024): shorter blocks give a short context more work-groups on a wide device (section 14.3) |
| `ZES_ENABLE_SYSMAN` | `1` | Level Zero's sysman, which the device's free-memory query needs: the backend sets it before its first runtime call and the image's `ENV` has it; the open line says `free memory from the device` when the query is there, else `estimated from the backend's books` (the total less the resident tensors, the chain's buffers, the pools and the arena) |

Every override is reported on stderr with a `[SYCL]` prefix when it cannot be honoured; the
device line at open is `[SYCL] device N: <name>, <GB>, profile <name>, <EUs>, SLM, xmx on|off`.

## Detection

`coli_sycl_profile.cpp` reads the device: `ext::oneapi::experimental::info::device::architecture`
(every `intel_gpu_*` value of oneAPI 2025.3's `device_architecture.def` has a row below; a part the
runtime's table does not know answers `architecture::unknown`, and that, or any value with no row on
a device whose vendor id is Intel's, is `generic`), `ext::intel::info::device::device_id`,
`pci_address`, `gpu_eu_count`, `gpu_hw_threads_per_eu`, `info::device::sub_group_sizes`,
`local_mem_size`, `max_work_group_size`, `global_mem_size`, `host_unified_memory` (integrated),
`max_compute_units`, and `matrix_combinations` for the matrix engines' shapes. Each query is its
own try/catch: one the backend declines (OpenCL has no PCI address) leaves a zero and the rules
fill a stated default. `coli_sycl_profile_rules.cpp` then decides, with no SYCL in it:

| architecture (`device_architecture.def`) | profile | gen | sub-group default | matrix engines |
| --- | --- | --- | --- | --- |
| `intel_gpu_pvc`, `pvc_vg` (Data Center GPU Max 1100 / 1550) | `pvc` | pvc | 32 | on: f16 / bf16 8x16x16 and 16x16x16, s8 8x16x32, tf32 8x16x8 (from the query) |
| `intel_gpu_bmg_g21`, `bmg_g31` (Arc B580 / B570, Arc Pro B60 / B70) | `bmg` | xe2 | 16 | on, the same shapes |
| `intel_gpu_lnl_m` (Lunar Lake) | `lnl` | xe2 | 16 | on, integrated |
| `intel_gpu_ptl_h`, `ptl_u` (Panther Lake) | `ptl` | xe3 | 16 | from the query; treated as xe2 until measured |
| `intel_gpu_wcl` (Wildcat Lake) | `wcl` | xe3 | 16 | from the query |
| `intel_gpu_arl_h` (Arrow Lake H) | `arl` | xe | 16 | from the query |
| `intel_gpu_mtl_u`, `mtl_h` (Meteor Lake; Arrow Lake U / S alias `mtl_u`) | `mtl` | xe | 16 | from the query |
| `intel_gpu_acm_g10 / g11 / g12` (Alchemist, "dg2") | `dg2` | xe | 16 | off: dg2's XMX takes 8x8 tiles at sub-group 8, not the kernels' shape (the query lists no 8x16 shape, so nothing is set) |
| `intel_gpu_tgllp`, `rkl`, `adl_s / p / n`, `dg1` (Xe-LP) | `xelp` | legacy | 16 | off (none) |
| `intel_gpu_bdw` .. `cml`, `icllp`, `ehl` (gen 8, 9, 11) | `gen9` | legacy | 16 | off (none) |
| anything else on an Intel device (the runtime's `unknown`: a part newer than oneAPI 2025.3's table, Nova Lake say) | `generic` | unknown | 16 if offered, else 32, else 16 (the kernels' only instances, said in the log) | from the query alone, which is keyed by the architecture's name inside the runtime and lists no shape for one it cannot name: off in practice until a runtime that knows the part |

The rules, in one place (`coli_sycl_profile_derive`): the sub-group is 32 on pvc and 16 everywhere
else, each only where offered (16 is Xe2's native width and what joint_matrix runs at; 32 is what
the kernels were written at and PVC's); the matrix engines are on when the kernels' tile, f16
8x16x16, is listed *and* sub-group 16 is offered, never by the device's name (so `xmx on` means
the engines run; the runtime lists each shape with M as a range up to 8 and N, K fixed, and the
rule reads a row dimension by dimension); the profile's name is
`<arch>-sg<N>` (`pvc-sg32`, `bmg-sg16`, `generic-sg16`). An unknown architecture never fails the
open. The test `tests/test_sycl_profile.cpp` pins all of it over fake property sets.

PCI IDs the log line shows (`device_id`): 0x0bda the Max 1100, 0x0bd5 the Max 1550, 0xe20b the
B580, 0xe20c the B570, 0xe211 the Arc Pro B60 (all Battlemage G21), 0x64a0 Lunar Lake's 140V.

## The per-device state (`coli_sycl.h`, section 2)

`coli_sycl_open(devices, n)` opens the engine's ordinals: a context and two in-order queues per
device (compute, and the chain's copy queue), the profile, logged once. The same list again keeps
everything; another list while devices are open returns 0, as `coli_cuda_init` documents.
`coli_sycl_queue(d)` / `coli_sycl_copy_queue(d)` / `coli_sycl_context(d)` / `coli_sycl_device(d)`
are the accessors (a fault on a device that is not open; `coli_sycl_is_open` first where the
ordinal comes from outside), `coli_sycl_queue_device(q)` maps a queue back to its ordinal.

`coli_sycl_scratch(d, bytes, slot)` is the arena that replaces `cudaMallocAsync` / `cudaFreeAsync`
on the decode path: one device allocation a slot, 256-byte aligned, grown to the largest request
(with headroom, half again) and never shrunk, under a mutex (the worker and the uploader may ask
at once). A grow never waits on the queues: inside a chain frame a wait could sit behind a
`cc_wait_host` kernel whose host side comes after the frame (the device-MoE forward is one frame),
so the old block is retired and freed by `coli_sycl_scratch_reap`, which the chain calls where it
has seen the queue pass the frames (`cc_submit` with a wait, `cc_finish`), and by `coli_sycl_close`.
`coli_sycl_retire(q, p, bytes)` puts a block used on the device's own queues on the same list
(`bd_xq_scratch`'s old copies; another queue's owner waits on it and frees). A grow after a
slot's first allocation is
logged (the GEMV partials are sized for every row count their path takes at the first use).
Slots: 0 the GEMV split-K partials, 1 the dnrec split's rows, 2 spare, 3 the XMX family's staging
copy of the activations.

`coli_sycl_lost(d)` is 1 once a queue's async handler took an error the runtime delivers that way
instead of throwing; the chain's `live` / `cc_try` and the backend's `fault_injected` read it as
they read a caught exception. `coli_sycl_mem_info(d, &free, &total, &queried)` is `cudaMemGetInfo`:
free from the device where it has the query (above), else the total less the backend's books.

## First contact (one card)

Everything up to here was done without touching a GPU. The order on the first box, one card
(`--device /dev/dri/cardN --device /dev/dri/renderDM`), each step before the next:

1. `COLI_SYCL_LIST=1 /app/tests/backend_cuda_test` (any entry point of the backend answers it: no
   kernel runs; the engine's form is `COLI_CUDA=1 COLI_SYCL_LIST=1 SNAP=/model /app/qwen36 256 4 p.txt`,
   answered after the model loads, since qwen36 only enters the backend with `COLI_CUDA=1`).
   Check the device name, the profile (`pvc-sg32` on a Max, `bmg-sg16` on a B-series), the
   sub-group sizes (16 32), EUs (448 the Max 1100, 160 the B580 / B60), SLM (the device's word:
   128 KiB on a Max and a B-series card), the matrix shapes (f16 / bf16 8x16x16, s8 8x16x32), the
   platform (Level Zero) and driver version, and on the open line whether free memory comes
   `from the device` (else `ZES_ENABLE_SYSMAN` did not reach the driver). If the
   list is empty: the runtime does not see the card (driver, `/dev/dri` permissions, the render
   group); `sycl-ls` in the container says the same in the vendor's words.
2. `/app/tests/test_coli_sycl_launch`: one kernel body at sub-group 16 and 32 through the
   dispatch, a warp-per-row sum checked against the host. Both sizes must pass on pvc and bmg.
   This is the first kernel that runs: a hang here is a driver or AOT problem, not the backend's.
3. `/app/tests/test_sycl_xmx` where it exists: the joint_matrix GEMM against the host.
4. `/app/tests/cuda_chain_test` (the migrated `test_cuda_chain`: the dense chain's ops against
   their CPU references), then with `COLI_SYCL_SG=16` and `COLI_SYCL_SG=32` on a card that offers
   both: the two instantiations must agree with the host, which is the gate before any engine run.
   `COLI_SYCL_XMX=0` isolates the matrix-engine path when a prompt-row op disagrees.
5. `/app/tests/backend_cuda_test --device 0` (the migrated `test_backend_cuda`: resident tensors
   and their accounting, the expert tier's groups, sync, async and pinned, every format's matmul
   against a host reference, `tensor_update`, the fault-injection hook, the deliberate out-of-memory
   upload that must not poison the next call): one line, `cuda backend: ... correctness ok`, and
   exit 0; a check that fails prints `FAIL test_backend_cuda.dp.cpp:<line>`.
6. The engine on the tiny fixture (`make qwen36-tiny-check` style, `docs/qwen36-cuda-tier.md`),
   `COLI_CUDA=1 COLI_GPU=0`; then the real container.

PVC's default profile remains `pvc-sg32`; the performance pass chooses widths per kernel
(MIGRATION.md 14.4). A per-kernel override wins over that kernel's architecture choice; it is
ignored with a log line when invalid or unsupported. `COLI_SYCL_SG=16` still changes the profile
width used by the other kernels and by prompt recurrence.

| kernel | PVC width by default | restore override |
| --- | --- | --- |
| int8 GEMV, 1024–2048 outputs | 16 | `COLI_CUDA_GEMV_SG=32` |
| other int8 GEMVs, route | profile width (32) | `COLI_SYCL_SG` / `COLI_CUDA_GEMV_SG` |
| expert gate/up and down GEMVs | 16 | `COLI_CUDA_MOE_SG=32` |
| recurrence below 8 rows | rows kernel at 16 | `COLI_CUDA_DNREC_DEC=0` for the old kernel |
| recurrence at 8+ rows | profile width (32), existing prompt dispatch | `COLI_SYCL_SG` |
| block attention, both stages | 16 | `COLI_CUDA_ATTN_SG=32` |

What to expect on the first runs: the hybrid steps' flags (`cc_signal` / `cc_wait_host`) are
host tasks on the chain's queue, never a kernel touching host memory with an atomic (an access
violation that bans the context on these cards) or spinning under i915 / xe's hang check
(`MIGRATION.md` section 13: what the first card taught); the wait's host side naps 20 us between
polls, so a hybrid step's resume costs tens of microseconds, to be measured against a decode
token before experts are split between the host and the card; launch overhead on this stack
was measured on the first card (MIGRATION.md section 14.2: 2.2 us of host a queued launch, a
0.64 us gap between back-to-back kernels, 2.9 us for a one-work-group kernel; ~850 launches a
decode token) and is not what bounds decode, the kernels' own durations are; prompt rows run on the warp / block kernels until the joint_matrix GEMM
family (`coli_sycl_xmx.cpp`) is in, and `COLI_SYCL_XMX=0` is the way back to them at any time.
fp32 division and square root in the kernels are the device compiler's OpenCL-precision ones
(2.5 / 3 ulp: `-fp-model=precise` asks for no contraction beyond a statement, not for correctly
rounded division; section 13), which is why every "the same bits" pair of kernels computes its
shared values through one shared subroutine (`cc_rstd`).

## Benches in the chain test

`COLI_CUDA_CHAIN_BENCH=<mode> /app/tests/cuda_chain_test --device 0` runs one of the test's benches instead of
the op checks (perf m-kernels, MIGRATION.md section 14.3): `adec` (a decode row's attention over 10 layers' caches,
`COLI_CUDA_BENCH_ADEC="H KVH hd pos"`), `route`, `gemv` (the chain's int8 GEMV at one row over
`COLI_CUDA_BENCH_GEMV="OxI[xS],.."`, Qwen3.6-35B's shapes by default), `moe` (a decode step's routed experts,
`COLI_CUDA_BENCH_MOE="D I K layers"`) and `dnrec` (the recurrence, `COLI_CUDA_BENCH_DNREC="VH KH KD VD layers S"`).
Each runs `COLI_CUDA_BENCH_N` (200) calls in one frame as a decode token does, rotating over copies of the operands
that pass the device's L2 (`COLI_CUDA_BENCH_POOL_MB`, 768), prints the pipelined time a call (the host clock around
the frame), the submit share and the bytes a second, and checks the last answer against the CPU (`[FAIL ...]` on the
line otherwise). The backend's knobs apply (`COLI_SYCL_SG`, `COLI_CUDA_GEMV_W4` / `_SK`, `COLI_CUDA_MOE_GV`,
`COLI_CUDA_DNREC_*`, `COLI_CUDA_ATTN_DEC_*`).

## Files

| file | what |
| --- | --- |
| `coli_sycl.h` | the shared interface: the profile (section 1), the devices and the arena (2), the SG16 / SG32 dispatch (3), the XMX GEMM family (4) |
| `coli_sycl_profile.h` | the profile's types, the derivation API and the names: plain C++ |
| `coli_sycl_profile_rules.cpp` | the derivation, the overrides, the names, the printed form: plain C++, no SYCL, testable anywhere |
| `coli_sycl_profile.cpp` | reading a device, the GPU list, open / close, queues, the arena, `COLI_SYCL_LIST` |
| `coli_sycl_xmx.cpp` | the joint_matrix GEMMs (section 4) |
| `coli_sycl_stubs.cpp` | weak "unsupported" definitions of section 4, so a unit links before its implementation exists |
| `backend_cuda.dp.cpp`, `cuda_chain.dp.cpp`, `cu_xb.dp.cpp`, `cuda_bd.dp.hpp`, `backend_gpu_compat.h` | the migrated units and their headers (`MIGRATION.md`) |
| `include/dpct/` | dpct's helper headers, as generated |
| `tests/test_sycl_profile.cpp` | the host-only profile test; `tests/test_coli_sycl_launch.cpp` the dispatch on a device; `tests/*.dp.cpp` the migrated GPU tests |
| `tools/dpct-migrate.sh`, `tools/dpct-prep.py` | re-running the migration on upstream changes |
| `Dockerfile.sycl` | the build and runtime image |

## Host synchronization on a lost context

The hybrid chain uses CPU host tasks for its UP/DOWN words. Tasks capture shared
cancellation state; a delivered asynchronous error or fatal chain failure releases
blocked waits and prevents later host flag accesses. The host `ready`/`lost` queries
collect asynchronous errors too, so engine route waits can stop. No device atomics
on host memory and no SYCL queue calls inside host tasks are used.

`sycl/tests/test_sycl_host_cancel --device 0` exercises a software failed launch;
`COLI_TEST_CANCEL_STATE=1` selects shared-state cancellation and status propagation.
Both have a CPU rescue writer and require it to remain unused. These are software
cancellation tests, not driver-fault recovery tests. Normal GPU guard/timeout rules
still apply. See MIGRATION section20 for qualification results.
