# MIGRATION.md — the dpct migration of the CUDA backend (prepare step, 2026-11-07)

What is in `c/sycl/`, how it was produced, what the tool reported, what one compile attempt per
unit says, and the hand-edit work list for the Edit phase. No GPU was touched: dpct 2025.3 (clang
21) and icpx 2025.3.3 ran in `intel/oneapi-basekit:2025.3.2-0-devel-ubuntu24.04` against the
CUDA 12.8 headers extracted under `<ops>/work/cuda-hdr`.

Paths written `<ops>/…` are the work tree, logs and scripts of the machine the port was made and
qualified on, and `<models>` its model directory; none of them is in this repository.

## 1. What was done

1. **Portable-only gating** (`c/backend_cuda.cu`, `c/cuda_chain.cu`, `c/backend_gpu_compat.h`):
   `COLI_GPU_PORTABLE_ONLY` compiles out everything the tool cannot carry, so dpct never sees
   WMMA, `mma.sync`, `cp.async`, `ldmatrix`, `griddepcontrol`, the stream-ordered allocator's
   pool, `cuda_fp8.h` or `cudaMemcpyBatchAsync`. 21 gates in backend_cuda.cu (lines 16, 706,
   740, 779, 1342, 1350, 3086, 3222, 3594, 4074, 4098, 4106, 4129, 4207, 4295, 4321, 4584, 4691,
   4906, 5251, 5269), 11 in cuda_chain.cu (523, 670, 884, 912, 1878, 2577, 2597, 2659, 2970, 3021,
   3459), plus backend_gpu_compat.h (the WMMA flags, `COLI_PDL_WAIT`, the PDL launcher, and
   `coli_launch` as a plain macro under the guard). Where the HIP arm of a gate is the portable
   stub (`return 0`, `smem_optin = 48 KiB`, `major = minor = 0`), the guard joins that arm
   (`|| defined(COLI_GPU_PORTABLE_ONLY)`); where the gate selects NVIDIA code it is
   `&& !defined(COLI_GPU_PORTABLE_ONLY)`. Two more lines had to move for the tool's parse:
   `backend_cuda.cu:405` (the fmt=8 fast-math guard tests nvcc's `__USE_FAST_MATH__`, which dpct
   predefines and no flag clears, only outside the guard; `__FAST_MATH__` stays, see §6) and the
   portable arm of backend_gpu_compat.h includes `<cuda_fp16.h>` (mma.h used to bring `__half`
   in; cuda_chain.cu's f16 KV mirrors need it). The CUDA and HIP arms are unchanged when the macro
   is undefined: every edit is an added conjunct/disjunct on an existing `#if`, or a new `#elif`
   arm. The CPU build (`make -C c qwen36 ARCH=native`, gcc 13.3) passes before and after; no `.c`
   file includes backend_gpu_compat.h.
   Left ungated on purpose (portable CUDA C the tool handles): the width-32 shuffle macros
   (`f8_shfl_down`, `mg_shfl_xor`, `cc_shfl_xor`, `bd_shfl_xor`), the f16 KV mirror code
   (`cc_kvf(const __half *)`, `cc_copy_grid_h_k`, `kvh`), the dnrec split (`cuda_chain.cu:2703`,
   cudaMallocAsync at 2714: a scratch-arena site, §4), `__dp4a` / `__byte_perm` (dpct maps them),
   `cudaMemcpyPeer*` (GLM pipe), `cudaHostRegister` (removed by the tool, §4).
2. **`coli_launch`**: the tiny probe (`<ops>/work/dpct-tiny/tiny.cu`) showed dpct 2025.3 migrates
   neither form. The kernel-pointer template becomes `dpct::kernel_launcher::launch(k, g, b, sm,
   st, a, a, a)` (the parameter pack expanded wrongly) plus a `*_wrapper` reading static state; a
   function-like macro gets its *definition* rewritten for the first expansion site (DPCT1038)
   and every other site is left as CUDA. A direct `k<2, float><<<g, b, sm, st>>>(args)` migrates
   cleanly. So under the guard `coli_launch` is the plain macro the task asked for, and
   `tools/dpct-prep.py` expands it textually at every call site of the migration's input (139
   sites: backend_cuda.cu 60, cuda_chain.cu 71, cuda_bd.cuh 8; the 14 inside the `HCM_MOE` macro
   have no g/b/sm/st of their own and sit in an NVIDIA-only arm, left as they are). The splitter
   tracks template brackets, so `coli_launch(cc_attn_decg_k<2, KT>, ...)` and
   `cc_f2_smem_bytes<64, 4, 64, true>()` as the dynamic-memory argument split correctly; the staged
   files keep the originals' line numbers (checked: 6245 / 3501 / 1314 / 704), so the tool's
   diagnostics below name the lines of `c/*.cu`.
3. **The migration** (`tools/dpct-migrate.sh`, re-runnable): stage `c/` into `<ops>/work/dpct-in`
   with the launches expanded, then
   ```
   dpct --cuda-include-path=/cuda-inc --in-root=/work/dpct-in --out-root=/work/dpct-out --gen-helper-function \
        --use-experimental-features=matrix,masked-sub-group-operation,local-memory-kernel-scope-allocation \
        --extra-arg=-I/work/dpct-in --extra-arg=-DCOLI_HIP_NO_WMMA=1 --extra-arg=-DCOLI_GPU_PORTABLE_ONLY=1 \
        backend_cuda.cu cuda_chain.cu cu_xb.cu tests/test_cuda_chain.cu tests/test_backend_cuda.cu
   ```
   (`-DCOLI_HIP_NO_WMMA=1` is inert for a CUDA parse, kept for fidelity to the plan; no
   `-DCOLI_CUDA` is needed: the Makefile's GPUFLAGS do not pass it to nvcc either.) Three runs: the
   first two had parse errors (`__half` undeclared without mma.h; the fast-math `#error`), the
   third parsed with 0 errors. Output copied into the tree:

   | file | lines | from |
   |---|---|---|
   | `c/sycl/backend_cuda.dp.cpp` | 11147 | backend_cuda.cu (6245) |
   | `c/sycl/cuda_chain.dp.cpp` | 6074 | cuda_chain.cu (3501) |
   | `c/sycl/cu_xb.dp.cpp` | 1125 | cu_xb.cu (704) |
   | `c/sycl/cuda_bd.dp.hpp` | 1980 | cuda_bd.cuh (1314), included by the first and third |
   | `c/sycl/backend_gpu_compat.h` | — | the compat header with `<sycl/sycl.hpp>` + `<dpct/dpct.hpp>` in place of `<cuda_runtime.h>` |
   | `c/sycl/tests/test_cuda_chain.dp.cpp` | 1842 | tests/test_cuda_chain.cu |
   | `c/sycl/tests/test_backend_cuda.dp.cpp` | 575 | tests/test_backend_cuda.cu |
   | `c/sycl/include/dpct/` | — | the helper headers (`--gen-helper-function`) |

   **Not kept**: the tool also rewrote `cuda_chain.h`, `cu_tier_ops.h`, `vk_alloc.h` and
   `backend_vulkan.h` only to name their anonymous structs (`typedef struct dpct_type_NNN`). The
   units include the originals from `c/` through `-I`; the engine's contract stays byte-identical.
4. **The interface header** `c/sycl/coli_sycl.h` (profile, per-device state, scratch arena,
   sub-group dispatch, the XMX GEMM family) and `c/sycl/coli_sycl_stubs.cpp` (weak "unsupported"
   definitions so each unit links alone). Both compile with icpx (`-Wall -Wextra`, 0 warnings),
   JIT and AOT. `c/sycl/tests/test_coli_sycl_launch.cpp` exercises the dispatch at sub-group 16 and
   32 in a real kernel: it builds and AOT-links for `-device pvc,bmg` (ocloc: "Build succeeded
   for : pvc." and "bmg."), so **the comma list works** and one object can carry both families.
   Running it needs a device.

## 2. DPCT diagnostics (run 3, `<ops>/work/dpct-migrate.log`)

Per file (the tool reports against the input `.cu` lines):

| file | sites | auto-ok | needs-hand-edit | delete-dead-code |
|---|---|---|---|---|
| backend_cuda.cu | 236 | 94 | 142 | 0 |
| cuda_chain.cu | 197 | 78 | 119 | 0 |
| cuda_bd.cuh | 46 | 39 | 6 | 1 |
| cu_xb.cu | 42 | 17 | 25 | 0 |
| tests/test_cuda_chain.cu | 2 | 2 | 0 | 0 |
| tests/test_backend_cuda.cu | 0 | | | |
| **total** | **523** | **230** | **292** | **1** |

Per code (what it means here, and the disposition rule the appendix applies):

| code | n | what | disposition |
|---|---|---|---|
| DPCT1065 | 110 | `barrier()` could take `local_space` | auto-ok (later tuning) |
| DPCT1118 | 109 | a group op (barrier, shuffle) in possibly divergent flow | needs-hand-edit: check each kernel's early `return`s before its barriers (the CUDA code has the same rule; backend_cuda 26, cuda_chain 79, cuda_bd 3, cu_xb 1: cuda_chain's attention, dnrec, qsa, route and enc kernels hold most) |
| DPCT1010 | 77 | `cudaGetLastError()` -> 0 | needs-hand-edit: the `cuda_ok` / `cc_ok` protocol becomes try/catch of `sycl::exception` around the submit (one wrapper, then the sites) |
| DPCT1110 | 56 | >128 bytes of locals in a kernel | auto-ok (register note; relevant to the SG16 choice) |
| DPCT1124 | 46 | `cudaMemcpyAsync` from pageable host memory is synchronous in CUDA, asynchronous here | needs-hand-edit: `wait()` where the source is a temporary or a stack buffer (uploads, `coli_cuda_matmul`'s round trip) |
| DPCT1098 | 28 | `__ldg` -> plain load | auto-ok |
| DPCT1083 | 13 | `local_accessor` size from the launch's bytes | needs-hand-edit: check the element type of each `extern __shared__` (float4 / uint4 arrays) |
| DPCT1093 | 10 | `dpct::select_device(ctx->device)` | needs-hand-edit: the ordinal is the engine's; use `coli_sycl_queue(device)` (GPU list, PCI order), not dpct's device manager (it lists CPUs too) |
| DPCT1024 | 10 | error code consumed by logic -> 0 | needs-hand-edit (as DPCT1010) |
| DPCT1048 | 9 | `cudaHostAlloc` flags dropped | auto-ok (`malloc_host` is mapped and portable) |
| DPCT1013 | 8 | rounding mode of a conversion | auto-ok (float->half is RN in `sycl::half`; the KV f16 parity test is the check) |
| DPCT1106 | 5 | Intel device-info extension | auto-ok (icpx has it; `coli_sycl_profile` carries the numbers) |
| DPCT1025 | 5 | queue flags dropped | auto-ok |
| DPCT1108 | 4 | masked sub-group helper (experimental) | auto-ok |
| DPCT1038 | 4 | a launch inside a local macro | needs-hand-edit: all four are `HCG_BT_LAUNCH` (backend_cuda.cu 4274, 4280, 5334, 5335), whose definition the tool rewrote for its first expansion: delete with the hcg dispatch (§3). cuda_chain's `CC_AN`, `CC_ATTN_G`, `CC_DS` were rewritten in place without a warning; `CC_AN` gained a trailing `;` (§3) |
| DPCT1026 | 4 | removed call | 3 `cudaGetLastError` auto-ok; 1 `cudaFuncSetAttribute` (bd_smem_raise) delete-dead-code |
| DPCT1007 | 4 | `cudaMallocAsync` / `cudaFreeAsync` unsupported | needs-hand-edit: `coli_sycl_scratch()` at backend_cuda.cu 4824/4830 (gemv_i8_w4, decode) and cuda_chain.cu 2714/2727 (dnrec split, prompt rows) |
| DPCT1064 | 3 | a shuffle inside a macro definition | needs-hand-edit: the width is the kernel's SG (`f8_shfl_down`, `mg_shfl_xor`, `cc_shfl_xor`; `bd_shfl_xor` likewise) |
| DPCT1006 | 3 | integrated vs discrete | needs-hand-edit: `coli_sycl_profile.is_integrated` |
| DPCT1078 | 2 | `__threadfence_system` memory order | needs-hand-edit: seq_cst, system scope (`cc_signal_k` / `cc_wait_k`) |
| DPCT1027 | 2 | `cudaHostRegister` / `Unregister` removed | needs-hand-edit: `coli_cuda_host_register` returns 0 (host pool off) |
| DPCT1009 | 2 | `cudaGetErrorString` | needs-hand-edit (exception text) |
| DPCT1005 | 2 | compute-capability compare | needs-hand-edit: the profile decides (`coli_sycl_xmx_gemm_ok`) |
| DPCT1001/1000/1002 | 4 | error if-statements | needs-hand-edit (as DPCT1010) |
| DPCT1051 | 1 | `canMapHostMemory` -> false | needs-hand-edit: true (USM host) |
| DPCT1049 | 1 | work-group size may exceed the limit | auto-ok (128 / 256 <= 1024 everywhere) |
| DPCT1019 | 1 | `sharedMemPerBlockOptin` -> `local_mem_size` | auto-ok |

Other facts from the output: dpct pinned `[[sycl::reqd_sub_group_size(32)]]` at 55 launches
(backend_cuda 18, cuda_chain 25, cu_xb 2, cuda_bd 10) — the warp-32 contract is a compiler
requirement now, which is the SG32 instance of §5; `__CUDA_ARCH__` became `DPCT_COMPATIBILITY_TEMP`
(`(900)` in dpct.hpp: every arch gate would be *true* without the portable-only guard, which is why
it was needed); `cudaStreamCreateWithFlags` -> `dev.create_queue()` (in order); events ->
`dpct::sync_barrier` / `ext_oneapi_submit_barrier` / `wait_and_throw`; `cudaHostAlloc(Mapped)` ->
`sycl::malloc_host` with the device pointer the same pointer; `__constant__` -> `dpct::constant_memory`;
`__trap` -> `assert(0)`; `__launch_bounds__` dropped; `cudaEventElapsedTime` (cu_xb) ->
profiling-info reads, which need `DPCT_PROFILING_ENABLED` (the migrated compat header defines it,
so every queue carries `enable_profiling`: review).

## 3. One compile attempt per unit

Inside the container, A = the literal command
`icpx -fsycl -std=c++17 -O2 -fsycl-targets=spir64 -I/work/source-sycl/c -I/work/source-sycl/c/sycl/include -c <TU>`,
B = A plus `-DCOLI_GPU_PORTABLE_ONLY=1 -fp-model=precise -ferror-limit=0 -I/work/source-sycl/c/sycl`
(the migrated compat header keeps the guard, so without the define it includes `mma.h`; the tests'
`"../x.h"` includes resolve to `c/` through the extra `-I`). Logs: `<ops>/work/tu-compile/`.

| unit | A errors | B errors | B distinct sites | cause |
|---|---|---|---|---|
| backend_cuda.dp.cpp | 1 (fatal: mma.h) | 4358 | 27 | 25 lines of the `HCG_BT_LAUNCH` macro the tool rewrote for its first expansion (238 expansions each: `auto __VA_ARGS___ct0 = __VA_ARGS__;` ...) and the two `*_wrapper` functions of cuda_bd.dp.hpp (a local `nr` clashing with the kernel's `nr` parameter: the tool's kernel-pointer wrappers for `bd_smem_raise`) |
| cuda_chain.dp.cpp | 1 (fatal: mma.h) | 5 | 5 | the rewritten `CC_AN` ends in `;`, so `if (..) CC_AN(8); else` is `;;` before an `else` (3 "expected expression"), `cudaMallocAsync` / `cudaFreeAsync` at the dnrec split (2) |
| cu_xb.dp.cpp | 1 (fatal: mma.h) | 2 (+3 warnings) | 2 | the same two cuda_bd.dp.hpp wrappers; warnings: `if (!ok) 0;` leftovers of DPCT1010 |
| tests/test_cuda_chain.dp.cpp | 2 (include path, mma.h) | **0** | | compiles |
| tests/test_backend_cuda.dp.cpp | 1 (include path) | **0** | | compiles |
| coli_sycl_stubs.cpp | 0 | 0 | | compiles (also AOT pvc,bmg) |

So the errors have six causes, not thousands: the three rewritten launch macros (delete
`HCG_BT_LAUNCH` with the hcg host dispatch it serves — `hcg_ok` is 0 and the XMX family replaces
it — and expand `CC_AN` by hand), the two cuda_bd.dp.hpp wrappers (delete with `bd_smem_raise`:
no opt-in exists), the four scratch sites, and the `0;` statements. Error-free is not
correct, though: the needs-hand-edit list below is what makes the units *right*.

## 4. The hand-edit work list (the Edit phase), by unit

Common first (touches every unit):
- The error protocol: `cuda_ok(expr, what)` / `cc_ok(c, expr, what)` take a `dpct::err0` that is
  always 0 now (DPCT1010/1024/1001/1000/1002/1009: 95 sites). Replace with one helper that runs
  the submit inside try/catch of `sycl::exception`, logs `what`, marks the device lost
  (`cc_lost`, the chain's device-lost protocol: `tests/test_cuda_chain` checks a failed launch
  marks the device lost through `COLI_GPU_FAIL_AFTER`), and returns 0.
- Devices and queues: `dpct::select_device(ctx->device)` / `dpct::get_device(device)` (DPCT1093,
  10 sites) and `dpct::get_in_order_queue()` (the `stream = 0` launches of `coli_cuda_matmul`)
  -> `coli_sycl_open` / `coli_sycl_queue(device)` / `coli_sycl_copy_queue(device)`; the migrated
  `coli_cuda_init` device loop enumerates through dpct's manager, which counts every SYCL device
  (CPU included) and numbers them its own way.
- Sub-groups: every kernel the tool pinned at 32 becomes a template over SG launched through
  `coli_sycl_launch_sg(q, profile.sg_default, ...)` (or `COLI_SYCL_SG_SWITCH` at a migrated
  `parallel_for`); `& 31` -> `& (SG - 1)`, `>> 5` -> `/ SG`, "8 warps" -> `block / SG`, the
  shuffle macros' width -> SG (DPCT1064), `dpct_local` sizes unchanged. The list of kernels: §5.
- Dynamic local memory (DPCT1083, 13): check the `local_accessor<uint8_t>` byte counts against
  the launch's `sm` (they are the same expression; the element-type casts inside the kernels are
  what to read).
- Delete the dead NVIDIA arms left in CUDA syntax (§7) rather than carrying them.
- Build flags: `-fp-model=precise` (icpx defaults to fast math and `backend_cuda.dp.cpp:444`
  refuses it, correctly: FTZ breaks CPU parity); `-DCOLI_GPU_PORTABLE_ONLY=1` until §7 is done.

`backend_cuda.dp.cpp`:
- `HCG_BT_LAUNCH` (dp.cpp 6047; the 4 DPCT1038 sites 4274/4280/5334/5335 of the .cu) and the hcg
  host dispatch (`hcg_gemm`, `hcg_part`, `hcg_moe`, `hcg_bt_wf`, `coli_cuda_planar_rows` -> 0):
  delete; `coli_sycl_xmx_gemm` / `coli_sycl_xmx_moe` take their place behind
  `coli_sycl_xmx_gemm_ok` / `coli_sycl_xmx_moe_ok`.
- `cudaMallocAsync` / `cudaFreeAsync` at gemv_i8_w4 (.cu 4824/4830, dp.cpp 8028/8096, decode):
  `coli_sycl_scratch(t->device, kz * S * O * 4, 0)`, no free; the `kz == 1 ||` fallback stays.
- `coli_cuda_host_register` / `unregister` (.cu 4286-4291, DPCT1027): return 0 / no-op.
- `coli_cuda_load_module` and `cc_preload`'s use of it: return 1 (no lazy module; AOT).
- `coli_cuda_init`: `prop.major/minor` -> `compute_major = compute_minor = 0` (keeps every
  `>= 7/8` gate off), `sms` -> `profile.eu_count / 8` or the Xe-core count (the grid-sizing
  heuristics of `f32_gemm_launch`, `gemv_i8_sk_splits`, `hcg_part` were tuned to SMs),
  `smem_optin` -> `profile.slm_bytes`, `is_integrated` (DPCT1006, 3) -> the profile; the device
  line prints the profile's name and arch instead of `sm_%d%d`.
- `cudaMemcpyAsync` from host temporaries (DPCT1124, 34 here): the sync uploads in
  `coli_cuda_tensor_upload*`, `coli_cuda_matmul`'s H2D/D2H round trip and the group `host_x/host_y`
  staging are the ones to `wait()` (pinned `malloc_host` staging buffers are fine).
- `__constant__` tables (`c_e8_grid`, `c_e4m3`, `cu_e2m1`): `dpct::constant_memory` + `init` on
  the right queue (fmt 6/8 only: low priority).
- `coli_cuda_pipe_peer_copy` / the GLM pipe's `cudaMemcpyPeerAsync` (DPCT1124 peer, 2): enable
  peer access first (`ext_oneapi_enable_peer_access`) or return 0 (not qwen36).
- `cudaFuncSetAttribute` in `bd_smem_raise` and the two `*_wrapper` functions in cuda_bd.dp.hpp:
  delete (SLM is 128 KiB / 64 KiB with no opt-in; sizes above `profile.slm_bytes` must refuse).
- DPCT1118 (26 sites: `quant_matmul`, the `grouped_*` family, `attention_absorb_*`, `gemv_i8_w4`,
  `f32_gemm_k`, `f32_few_k`, `moe_bd_*`, the `pipe_*` kernels): verify no `return` precedes a
  barrier inside a divergent branch.

`cuda_chain.dp.cpp`:
- `CC_AN` (dp.cpp 3263, used at 3280): drop the trailing `;` of the rewritten macro, or expand
  the four `cc_addnorm_k<NV>` launches by hand; `CC_ATTN_G` (4153) and `CC_DS` (4325) were
  rewritten in place and compile: read them once.
- `cudaMallocAsync` / `cudaFreeAsync` at the dnrec split (.cu 2714/2727, dp.cpp 4303/4371):
  `coli_sycl_scratch(c->device, (nq + nbd) * 4, 1)`; `goto rows` stays the fallback.
- `cc_init`: `major = minor = 0` arm is already the active one; `cc_preload` -> 1.
- `cc_buf` CC_UP / CC_DOWN: `malloc_host` on the device's context, the device pointer is the host
  pointer (DPCT1048 x3 auto); `cudaMemcpyBatchAsync` arm is dead (§7).
- `cc_signal_k` / `cc_wait_k` (DPCT1078): `sycl::atomic_fence(memory_order::seq_cst,
  memory_scope::system)`; a kernel spinning on host memory stays a caveat under i915's hang check
  (keep every expert resident on this hardware).
- DPCT1118 (79 sites) in `cc_norm_k`, `cc_addnorm_k`, `cc_attn_k`, `cc_attn_gk`, `cc_dnrec_k`,
  `cc_dnrec_rows_k`, `cc_dnrec_split_k`, `cc_dnnorm_k`, `cc_qsa_*`, `cc_ple_k`, `cc_route_k`,
  `cc_attn_dec_k`, `cc_attn_decg_k`, `cc_copy_grid_h_k`, `cc_enc_*`: same check.
- `kvh()` returns 1 (the f16 KV mirrors are migrated: `sycl::half`), or 0 until the parity test
  of `rope_f16` / `attn_h` passes on a device.

`cu_xb.dp.cpp` (not on qwen36's path; `coli_cuda_tier_ops()` may return NULL for the first
build, `cu_tier.c` installs its stub table):
- `if (!ok) 0;` / `else 0;` (dp.cpp 440, 495, 1061): the DPCT1010 leftovers of `cudaPeekAtLastError`.
- `cudaEventElapsedTime` -> profiling reads: either profiling queues (`DPCT_PROFILING_ENABLED`)
  or host clocks around `wait()` (the tier's stats only).
- `cudaHostAlloc(Portable)` host pool and `cudaHostGetDevicePointer`: `malloc_host`, same pointer.
- `vk_alloc.h` sub-allocator over `cudaMalloc` blocks: `malloc_device` on the device's context.

`tests/test_cuda_chain.dp.cpp`, `tests/test_backend_cuda.dp.cpp`: compile as migrated; they call
the ABI only (the chain test's two sites are `cudaHostAlloc(Mapped)` auto-ok). Link each with
its backend unit(s) + `coli_sycl_stubs.cpp` and `-lsycl`; running needs a device
(`make cuda-chain-check` is the gate, in the runtime image).

Makefile (not this step): a `SYCL=1` arm beside `CUDA=1`/`HIP=1` with new object rules over
`c/sycl/*.dp.cpp` (the existing rules hard-code the `.cu` sources), `GPUCC=icpx`,
`GPUFLAGS=-fsycl -std=c++17 -O3 -fp-model=precise -fsycl-targets=spir64_gen
-Xsycl-target-backend=spir64_gen "-device pvc,bmg" -I c/sycl/include`, `LDFLAGS += -lsycl -lstdc++`
(also added to `NOCUDA_LDFLAGS` at Makefile:1985), `SYCL=$(SYCL)` in `BUILD_CONFIG` (901), the
`qwen36` switch at 1472, `coli serve`'s `cuda_binary()` check (`coli:800-812`) extended to
`libsycl`. The engine links with `-fsycl-link`-wrapped objects or an icpx-driven final link.

## 5. Warp-32-sensitive kernels (the SG16/SG32 instantiation list)

From a per-kernel scan of the sources (shfl = width-32 shuffle macros or `__shfl_*`, &31 =
lane/warp arithmetic `& 31`, `>> 5`, `% 32`, `* 32`; smem = `__shared__` tree reductions with
`__syncthreads`; dyn = `extern __shared__`; lb = `__launch_bounds__`). Decode-path kernels in
bold (§6). Kernels the tool pinned at 32 are those with shfl or &31 below.

| kernel | file:line | shfl | &31 | other | path |
|---|---|---|---|---|---|
| **gemv_i8_rows<NR,PAD>** | backend_cuda.cu:4353 | 1 | 3 | smem, dyn, lb256 | D (int8 trunk GEMV) |
| **gemv_i8_w4<NR>** | backend_cuda.cu:4408 | 1 | 3 | smem, dyn, lb256, scratch | D (lm_head, dnproj; split-K) |
| gemv_i8_sk_sum | backend_cuda.cu:4465 | | | | D (with the split) |
| **f32_gemv_k** / f32_gemm_k<BM> / f32_few_k | backend_cuda.cu:4552 / 4509 / 4774 | | | smem (block reduce), lb256 | D / P |
| **moe_gv_gu_k<FMT>**, **moe_gv_dn_k<FMT>** | backend_cuda.cu:5405, 5449 | 2, 1 (`mg_shfl_xor`) | 3, 3 | smem, dyn, lb256 | D (default MoE) |
| **moe_gu_k<FMT>**, **moe_dn_k<FMT>** | backend_cuda.cu:4977, 5005 | 2, 1 (`moe_warp_sum`) | 2, 2 | smem, dyn, lb256 | D (`COLI_CUDA_MOE_GV=0`) |
| **moe_sum_k** | backend_cuda.cu:5031 | | | | D |
| quant_matmul, quant_matmul_f8w | backend_cuda.cu:518, 489 | 0, `f8_shfl_down` | 1, 4 | smem | T (tier, chain off) / N |
| grouped_hidden_g4_dual, grouped_down_g4 (+ w4, e8, f8 twins) | backend_cuda.cu:923, 944 (876-1040) | f8w: `f8_shfl_down` | f8w: 3 | smem block reduce | T |
| dn_head_kernel (`dn_block_sum`) | backend_cuda.cu:2002 | 3 | | smem, lb256 | T, O (`Q36_DN_GPU`) |
| moe_bd_gu_k<T>, moe_bd_dn_k<T>, bd_stage_k | backend_cuda.cu:5573, 5615, 5513 | 2, 1 | 4, 4 | smem, dyn, lb256 | G (GGUF) |
| cc_attn_k<KT>, cc_attn_gk<G,KT> | cuda_chain.cu:289, 377 | | 0, 1 | smem, lb128 | P (prompt), D fallback |
| **cc_attn_dec_k<KT>**, **cc_attn_decg_k<G,KT>** | cuda_chain.cu:2254, 2370 | 4, 4 (`cc_shfl_xor`) | 2, 4 | smem, dyn (decg), lb256 | D (decode attention) |
| **cc_dnrec_k<KD>** | cuda_chain.cu:1027 | | | smem, lb128 | D |
| cc_dnrec_rows_k, cc_dnprep_k, cc_dnrec_split_k | cuda_chain.cu:1114, 1193, 1229 | 2, 1, 2 | 2, 2, 0 | smem, lb128 | P (rows), verify |
| **cc_dnnorm_k** | cuda_chain.cu:1332 | 1 | 2 | lb128 | D |
| **cc_route_k** | cuda_chain.cu:1667 | `cc_wmax/wsum/wmin` (width 32, 1664-1666) | 2 | smem, lb256, `nsg = 8` warps | D |
| **cc_pick_k** | cuda_chain.cu:1775 | | 2 | smem, lb256 | D (greedy, MTP) |
| cc_matmul_rows_i8_k, cc_matmul_rows_k | cuda_chain.cu:1815, 1843 | | 2, 2 | smem, dyn, lb256 | O (MTP) |
| **cc_norm_k**, **cc_addnorm_k<NV>** | cuda_chain.cu:103, 157 | | | smem tree reduce (256) | D, P |
| cc_qsa_score_t, cc_qsa_k, cc_ple_k | cuda_chain.cu:1395, 1445, 1551 | | 3, 0, 0 | smem | N (Qwen3.8) |
| cu_gemv_k<FMT,NR> | cu_xb.cu:395 | 1 (`bd_shfl_xor`) | 2 | lb256 | N (shared tier) |
| bd_gemv_i8_k, bd_gemv_rows_i8_k, bd_gemv_f_k, bd_gemv_bf16_k, bdx_quant_k | cuda_bd.cuh:926, 989, 1021, 1093, 135 | 2, 1, 1, 1, 0 | 4, 4, 3, 2, 1 | smem, dyn, lb256 | G |

Not sensitive (no lane arithmetic, no shuffle; a barrier at most): `cc_rope_k`, `cc_rope_f16_k`,
`cc_dnconv_k`, `cc_dnconv_rows_k`, `cc_dnconv_ring_k`, `cc_ew_k`, `cc_copy_k`, `cc_regions_k`,
`cc_copy_grid_k`, `cc_copy_grid_h_k`, `cc_signal_k`, `cc_wait_k`, `silu_mul`, `situ_mul`,
`offset_to_signed_s4`, `moe_sort_*`, `moe_rows_sum_k`, `e8_rot_rows_kernel`, the `pipe_*`,
`cc_enc_*`, `cu_act_*`. The NVIDIA-only kernels (`w4a16_*`, `grouped_s4_wmma`, `hcg*`, `hcm*`,
`tcg*`, `f32t_gemm_k`, `cc_attn_fk`, `cc_attn_f2`) are empty under the guard and not in the list.

## 6. The decode path (cuda-contract.md §2, the first target: int4-g64 container, chain on, every expert resident)

Kernels a decode token runs, in order of a layer: `cc_norm_k` / `cc_addnorm_k` (cuda_chain.cu
103, 157) -> `gemv_i8_rows` (backend_cuda.cu 4353; `gemv_i8_w4` + `gemv_i8_sk_sum` 4408/4465 for
lm_head and dnproj) -> `cc_rope_k` (239; `cc_rope_f16_k` 253 with the f16 mirrors) -> attention
`cc_attn_dec_k` / `cc_attn_decg_k` (2254, 2370; `cc_attn_k` 289 as the `COLI_CUDA_ATTN_DEC=0`
fallback) or DeltaNet `cc_dnconv_k` (951) + `cc_dnrec_k` (1027) + `cc_dnnorm_k` (1332) ->
`cc_ew_k` (1611, 13 ops) -> `f32_gemv_k` (4552: router 2048x257, DeltaNet b/a rows, shared gate)
-> `cc_route_k` (1667) -> `moe_gv_gu_k` + `moe_gv_dn_k` (5405, 5449; `moe_gu_k` / `moe_dn_k`
4977/5005 with `COLI_CUDA_MOE_GV=0`) -> `moe_sum_k` (5031) -> copies `cc_copy_k` / `cc_regions_k` /
`cc_copy_grid_k` (2019, 2031, 2075; `cc_copy_grid_h_k` 3127) -> `cc_pick_k` (1775). Hybrid layers
(some experts on the CPU) add `cc_signal_k` / `cc_wait_k` (1770-1771). Upload-time:
`offset_to_signed_s4` (395). Tier decode with the chain off adds `quant_matmul` (518) and the
`grouped_*_g4` pair (923, 944). The two decode allocation sites are `gemv_i8_w4`'s partials (4824)
and, for a verify's rows, the dnrec split (cuda_chain.cu 2714). The chain issues ~800 launches a
token (docs/cuda.md): launch cost on this stack (11.5 us a queued OpenCL launch measured) is the
first thing to measure once a device may be used.

## 7. Dead NVIDIA arms left in CUDA syntax (delete in the Edit phase)

The tool migrates the active preprocessor arm only; the inactive NVIDIA arms are still CUDA text
in the units, to delete rather than carry (they cannot compile and have no Intel form):
backend_cuda.dp.cpp 22 (cuda_fp8.h), 860 / 894 / 957 (w4a16_matmul, w4a16_gate_up,
grouped_s4_wmma bodies), 2153 (the memory pool), 5718 (TCG_BODY), 5879 (HCG_BODY), 6573
(HCM_BODY), 7095-7405 (hcg_part, hcg_part_sum, hcm_launch, hcm_gemm, hcmix, hcg_ok, tcg_ok),
7733 (F32T_BODY), 7841 (f32t_gemm), 8433 (load_module), 8963 / 8983 (hcm_moe_able, hcm_moe);
cuda_chain.dp.cpp 806 (CC_FA_BODY), 955 (CC_F2_BODY), 1169 (cc_f2_256), 1197 (cc_attn_f2_go),
2748 (major/minor), 3972 / 3992 / 4143 / 5157 (the f2/fk dispatch), 5070 (cudaMemcpyBatchAsync),
6034 (cc_preload); backend_gpu_compat.h 143 (mma.h), 169-174 (PDL), 211 (the launch macro: the
units have no `coli_launch` left). The `COLI_ANS` block (dietgpu) is unmigrated too (never built
here). After the deletion the units no longer need `-DCOLI_GPU_PORTABLE_ONLY`.
**cuda_chain.dp.cpp's arms are deleted (§9); the unit needs neither the macro nor backend_gpu_compat.h.**

## 8. Notes

- **Fast math**: icpx defaults to `-fp-model=fast`, which defines `__FAST_MATH__`;
  `backend_cuda.dp.cpp:444` then stops the build ("fmt=8 kernels forbid fast-math builds (FTZ
  breaks CPU parity)"). That is the intended protection: build every SYCL unit with
  `-fp-model=precise`. The nvcc macro `__USE_FAST_MATH__`, which dpct predefines, is tested only
  outside the guard.
- **Sub-group attribute spelling**: `[[sycl::reqd_sub_group_size(N)]]` (what dpct emits and what
  coli_sycl.h uses); `[[intel::reqd_sub_group_size(N)]]` still compiles with a deprecation note.
- **AOT**: `-fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device pvc,bmg"` verified
  through ocloc on a real kernel (both device images in one binary).
- **dpct's shuffle helpers** take the logical width as a trailing argument that the migration
  dropped (default 32): with the SG16 instance it must be passed (`COLI_SYCL_SHFL_WIDTH(SG)`).
- **Profiling**: the migrated compat header defines `DPCT_PROFILING_ENABLED` (for cu_xb's
  `cudaEventElapsedTime`); it puts `enable_profiling` on dpct-created queues. Decide once.
- The migration is re-runnable (`tools/dpct-migrate.sh`); hand edits should be kept as a patch
  set over a re-run, or the fork drifts from upstream's `.cu` files.
- **The XMX GEMM family** (`coli_sycl_xmx.cpp`, `tests/test_sycl_xmx.cpp`, the edit-xmx step):
  hcg_tile's 64 x 64 x 64 tile on `joint_matrix` at sub-group 16 (f16 8x16x16, the compound
  16x16x16 with `COLI_SYCL_XMX_TM=16`; s8 8x16x32 for `a_s8` on fmt 1 and 2), the weights decoded
  to f16 into VNNI-packed local tiles (two, one barrier a step), the activations rounded once to
  f16 (or quantized per row to s8) into a scratch copy a small kernel writes before the GEMM
  (the arena's slot 3 on an open device's queue, a private copy for a foreign queue: §12, fix
  15), the epilogue in the accumulators (`ext::intel` `joint_matrix_apply`
  with coordinates: row scales, GLU, scatter to the rows' places). Semantics adjusted in
  `coli_sycl.h` section 4, all in the comments: a device without the f16 shape (or
  `COLI_SYCL_XMX=0`) runs the same staging through a plain f32 tile, so `coli_sycl_xmx_gemm_ok`
  answers for shape and format alone (the API never fails for a shape it takes; the backend's
  call sites ask `coli_sycl_xmx_supported` first and keep the migrated kernels without the
  shape: §12, fix 6); the MoE's act is
  `CcMoe.act 0` = silu(g) * u (cuda_chain.h's wording, not "act 1"); `coli_sycl_moe_sort.TR` must
  be 64; `CcExpert.host` entries are read where they are (USM host memory) rather than refused
  (a kernel cannot refuse per entry); `a_s8` is taken for fmt 1 and 2 only. The profile unit owns
  the matrix_combinations query and `coli_sycl_xmx_supported`. Build and run (one card, no
  engine): `tests/test_sycl_xmx.cpp`'s header; `--list`, `--small` first, exit 1 on a mismatch.

## 9. The Edit phase: `cuda_chain.dp.cpp` and the two migrated tests (edit-chain, 2026-11-07)

The dense chain is rewritten from the migrated text, by hand, with nothing of dpct left in it (no
`dpct.hpp`, no `backend_gpu_compat.h`, no `-DCOLI_GPU_PORTABLE_ONLY` needed by this unit): 3422
lines from 6076. `icpx -fsycl -std=c++17 -O2 -fp-model=precise -Wall -Wextra` compiles it with 0
errors and 0 warnings (JIT); the AOT fat object and the device link for `-device pvc,bmg` are in
§9.6. Every kernel's arithmetic is the `.cu`'s (the dpct text, its DPCT comments and
`COLI_PDL_WAIT()` lines stripped, `item_ct1` renamed `it`); the host code follows `cuda_chain.cu`
function for function. Nothing was run on a device.

### 9.1 Sub-groups: the warp-32-style kernels at 16 and 32

Every kernel the tool had pinned at `reqd_sub_group_size(32)` is a template over `SG`, launched
through `coli_sycl.h`'s `coli_sycl_launch_sg` at the context's size (`CcCtx.sg`, the profile's
`sg_default`: 32 on Ponte Vecchio, 16 on Xe2, `COLI_SYCL_SG` overrides), so each exists at 16 and
at 32 in every binary and the profile picks. Lane arithmetic is SG's throughout: `lane = id &
(SG - 1)`, `warp = id / SG`, strides `+= SG`, shuffle trees from `SG / 2`, a block's warps `W =
block / SG` (the per-warp partials `red[]` sized `256 / SG`; host-side grids that counted "4
warps a block of 128" or "8 a block of 256" use `128 / c->sg` and `256 / c->sg`). The shuffle is
`sycl::permute_group_by_xor` over the real sub-group (`cc_shfl_xor`), never dpct's width-32
helper (its default `logical_sub_group_size` of 32 indexes lanes 16..31 of a 16-wide sub-group:
undefined). The kernels and what changed in each:

| kernel | SG-relative edits |
|---|---|
| `cc_route_k<SG>` | `nsg = 256 / SG`; `cc_wmax<SG>` / `cc_wsum<SG>` / `cc_wmin<SG>` (templates, head of the file) |
| `cc_pick_k<SG>` | `rv[W]`, `ri[W]`, the warp loops to `W` |
| `cc_attn_dec_k<KT, SG>` | `red[W]`, a warp a position (`j += W`), the dot `d += SG`, the trees from `SG / 2`, the warp merges to `W` (both modes) |
| `cc_attn_decg_k<G, KT, SG>` | `red[G * W]` indexed `g * W + wp`, `64 / SG` float4s of a K row a lane (tpp 0), the trees, `l` over `W` |
| `cc_dnrec_rows_k<KD, R, SG>` | `W = 128 / SG` warps a block: vectors `w += W`, rows `r += W`, `k += SG`, `c += SG`, the trees |
| `cc_dnprep_k<SG>` | a warp a vector: `g = block * (128 / SG) + id / SG` (the host's `nvb` counts `128 / c->sg` a block) |
| `cc_dnnorm_k<SG>` | the same warp indexing; the host grid `(S * VH + W - 1) / W` |
| `cc_matmul_rows_i8_k<PAD, SG>`, `cc_matmul_rows_k<SG>` | a warp a row: `j = block * (256 / SG) + warp` (the host grid `grid1(n, 256 / c->sg)`), `i += SG` |
| `cc_dnrec_split_k<KD, R, P, C>` | no SG inside (its shuffles join P lanes, P <= 16, a power of two: within either sub-group); launched pinned so the compiler cannot pick 8; a `COLI_CUDA_DNREC_CFG` with P > `c->sg` falls back to 84 |

The block-tree kernels (`cc_norm_k`, `cc_addnorm_k`, `cc_attn_k`, `cc_attn_gk`, `cc_dnrec_k`, the
qsa, ple, enc and copy kernels) have no lane arithmetic and take the compiler's sub-group size
through a plain `parallel_for` (`CC_K`); every barrier dpct emitted stays as it emitted it (the
`local_space` ones where the kernel touched no global memory between them, the full ones
elsewhere: the DPCT1065 tuning is not taken). The early-return-before-barrier check (DPCT1118,
79 sites): every return before a barrier is block-uniform (`g >= p.nseg`, `s >= p.S`, `cnt <= 0`,
`take >= blocks` -- all from `get_group` and the parameters), so no work-item leaves a barrier
behind; the lane-divergent branches (`if (lane == 0)`, `if (i < n)`) enclose no group operation.

### 9.2 The host side, construct by construct

| CUDA | here |
|---|---|
| `cudaSetDevice` in `live()` / `cc_init` | none: the queues carry their device; `cc_init` takes `coli_sycl_queue(device)` and `coli_sycl_copy_queue(device)` of a device `coli_cuda_init` opened (`coli_sycl_is_open`), the profile's `sg_default` and `slm_bytes` |
| the stream, the copy stream | `c->stream`, `c->cs`: the device's two in-order queues (one stream = one in-order queue; a frame's launches stay in order) |
| `cudaEventRecord(ev, s)` | `ev = s->ext_oneapi_submit_barrier()` (complete once everything before it on the in-order queue is); `cudaStreamWaitEvent(s, ev)` is `s->ext_oneapi_submit_barrier({ev})`; `cudaEventQuery` is `get_info<command_execution_status>`; `cudaEventSynchronize` is `wait_and_throw` |
| events in `CcCtx` | a `CcSync` (the frame's, staging's, the 1024 host_up marks, the staging slots') made with `new` at `cc_init`: no SYCL object in the static table, nothing destroyed at exit after the runtime |
| the error protocol (`cc_ok` on `cudaGetLastError`) | `cc_try(c, what, f)`: the submit or wait inside a try, a `sycl::exception` logs `what` and marks the context lost; `launched` returns `!c->lost` as before. An asynchronous error (one the runtime hands to the queue's handler at a `wait_and_throw` instead of throwing) sets the device's flag, `coli_sycl_lost`, which `live` and `cc_try` read (`cc_async_lost`) and treat the same (§12, fix 4). `tests/test_cuda_chain`'s lost check (`COLI_GPU_FAIL_AFTER`) goes through `cc_matmul`'s backend call as it did |
| `cudaMalloc` / `cudaHostAlloc(Mapped)` | `sycl::malloc_device` / `sycl::malloc_host` on the device's queue (`cc_alloc`: NULL when there is no room, the context kept, as cudaMalloc's failure was the caller's to size down); the UP/DOWN device pointer is the host pointer |
| `cudaMemcpyAsync` from pageable memory in `cc_write` | staged: the bytes go to one of `CC_STAGE_N` (32) host-USM slots first (a slot reused once the copy that last read it is done), so the caller may reuse its memory on return as the contract says; `cc_fetch` / `cc_host_up` keep their asynchronous copies (their sources stay until the ticket lands) |
| `cudaMallocAsync` at the dnrec split (`.cu:2714`) | `coli_sycl_scratch(c->device, (nq + nbd) * 4, 1)`: grown with headroom (half again) by a larger chunk, the old block retired and freed by `coli_sycl_scratch_reap` at the frame's end (`cc_submit` with a wait, `cc_finish`), never a queue wait inside the frame (§12, fix 2); `cc_dnrec_rows_k` when the arena has no room |
| `cudaMalloc` for `cc_attn_dec_k`'s scratch | the chain's own persistent block (`cc_adec_fit`): grown rarely, the old block retired to `cc_adec_reap` after a frame wait rather than freed under the frames, never a queue wait inside a frame (a `cc_wait_host` kernel whose host side comes after could be there: the deadlock the `.cu` names). `coli_sycl_scratch`'s grow retires the same way now (§12, fix 2), and its reap runs next to `cc_adec_reap`; the chain's block predates it and stays |
| `cudaFuncSetAttribute` (the 48 KiB check of `cc_matmul_rows`) | `smem <= c->slm` (the profile's local memory: 128 KiB on pvc, 64 KiB on Xe2) |
| `__threadfence_system` + a volatile word (`cc_signal_k`, `cc_wait_k`) | *(the Edit phase's form, superseded on the first card: section 13)* `atomic_fence(seq_cst, system)` and a system-scope `atomic_ref` store / spin load on the host-USM word -- an AtomicAccessViolation on Ponte Vecchio; now host tasks on the in-order queue (`cc_signal` / `cc_wait_host`), no kernel |
| `cudaMemcpyBatchAsync` (`cc_host_up_n`) | the copies one after the other on the copy queue, the last one's event the mark |
| `cc_preload` (module loading) | 1: AOT, nothing to load |
| `CcRegK` (128 regions a launch) | 64 a launch: the list is a kernel argument, kept under a kilobyte |
| `cudaDeviceGetAttribute(fp64)` for the encoders' double sums | `cc_enc_norm` returns 0 on a device without the fp64 aspect (`CcCtx.fp64`) |

### 9.3 The tensor-core attention: the matrix engines' hook

`cc_attn_fk`, `cc_attn_f2`, `cc_f2_256`, `cc_attn_f2_go`, the `major / minor` fields and the
`#if` arms around their dispatch are deleted (WMMA and `mma.sync` have no SYCL form). In their
place `coli_sycl.h` gained (section 4 (c)) `coli_sycl_attn` / `coli_sycl_xmx_attn` /
`coli_sycl_xmx_attn_ok`: a prompt chunk's causal attention on the matrix engines, with CcAttn's
offsets, the f32 or f16 cache, the gate, a window, sinks, a ring and a selection list (everything
the two deleted kernels took), 1 submitted or 0 not taken; weak stubs return 0
(`coli_sycl_stubs.cpp`), so until coli_sycl_xmx.cpp implements it every prompt row runs
`cc_attn_k` / `cc_attn_gk` (f32, the bits of the CPU reference), as on a device without tensor
cores. `cc_attn_t` and `cc_attn_w2_t` ask `cc_attn_xmx` at the same three places the `.cu` asked
`cc_attn_f2_go` (plain rows, rows with a selection list at head dim 256, a window with sinks).

The decode split (`cc_attn_dec_k`, "positions in blocks") was tied to the tensor cores in the
`.cu` (`fk`): without them a decode row went through `cc_attn_k`, one block a (head, row), so that
a resumed prefix answers bit for bit as the prompt recomputed does. Here `cc_attn_dec_on`
decides: `COLI_CUDA_ATTN_DEC=0` never, `=1` the CUDA rule (only where a prompt chunk takes the
engines), unset the split always -- the default, because a block walking every position leaves
448 EUs idle and the engines' attention is not there on every part. The test's attention checks
exercise the split at every row count (it is what a decode row runs).

### 9.4 The tests (`tests/test_cuda_chain.dp.cpp`, `tests/test_backend_cuda.dp.cpp`)

The ops and references are the CUDA tests' (every op against the CPU reference, the same
tolerances). What changed: `sycl_test_args.h` (new, shared) gives both `--list` (every GPU's
profile through `coli_sycl_print_profiles`, then exit 0: the first guarded contact, no kernel),
`--device N[,M..]` and `COLI_GPUS=N[,M..]`, else device 0; the chain test's context and every
tensor are on that device (`g_dev`: 16 upload sites and the `coli_cuda_matmul` call rewritten),
its planar experts' host memory is host USM on the device's queue (`moe_planar`, the one SYCL
construct the test has), dpct's `try / catch` wrapper and `DPCT_CHECK_ERROR` are gone, the
includes are `../../backend_cuda.h` etc. (no reliance on an `-I` whose parent is `c/`), and no
test includes `backend_gpu_compat.h`. Both compile with 0 errors and 0 warnings. Build:
`make sycl-test SYCL=1` (the Makefile's `cuda_chain_test` and `backend_cuda_test` rules take
`sycl/tests/*.dp.cpp` with every fat object); run: `./cuda_chain_test --list`, then
`./cuda_chain_test --device 1` (or `COLI_GPUS=1`), in the runtime container.

### 9.5 What is deliberately not the `.cu`'s

- `kvh()` returns 1 (the f16 KV mirrors are migrated: `sycl::half`, `cc_kv4` reads a `half4`);
  the parity test of `rope_f16` / `attn_h` on a device is the gate before an engine relies on it.
- The decode split's default (§9.3).
- `cc_write`'s staging ring (§9.2): one more host memcpy a write, for the contract's sake.
- `CC_REG_K` 64 (was 128).
- `cc_enc_norm` refused without fp64 (the CUDA kernel assumed it).

### 9.6 Build evidence (container, no GPU)

- `icpx -fsycl -std=c++17 -O2 -fp-model=precise -fsycl-targets=spir64 -Wall -Wextra -I c -I c/sycl -c
  c/sycl/cuda_chain.dp.cpp`: exit 0, 0 errors, 0 warnings.
- the two tests with the Makefile's `ICPXFLAGS` (`-O2`, spir64): exit 0, 0 errors, 0 warnings each.
- the AOT fat object with the Makefile's flags (`-O3 -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen
  "-device pvc,bmg" -fp-model=precise -fPIE`) and the device link (`-fsycl-link`, ocloc for pvc and
  bmg): see the step's summary (`<ops>/work/edit-chain/aot.log`).
- a link of `cuda_chain_test` and `backend_cuda_test` against `cuda_chain.dp.o`,
  `coli_sycl_profile.o`, `coli_sycl_profile_rules.o`, `coli_sycl_xmx.o`, `coli_sycl_stubs.o` and
  a scratch stub of the 28 `coli_cuda_*` entry points of `backend_cuda.dp.cpp` (which was
  mid-edit by its owner at the time): both link (`<ops>/work/edit-chain/link.log`); the real link
  is `make sycl-test SYCL=1` once that unit compiles.

Open, for the first run on a card: the shuffle-order sums at SG 16 against the references
(`cc_dnrec_rows_k`, `cc_dnprep_k`, `cc_dnnorm_k`, `cc_attn_dec_k`, `cc_attn_decg_k`, `cc_route_k`,
`cc_pick_k`, `cc_matmul_rows_*`: the test at `COLI_SYCL_SG=16` and `=32` on a part that offers
both); `cc_wait_k` under the hang check (moot: a host task since section 13); the f16 mirrors' parity; the `host_up` ring's events.

## 10. The Edit phase: `backend_cuda.dp.cpp`, `cuda_bd.dp.hpp`, `cu_xb.dp.cpp` (edit-expert, 2026-11-07)

The resident tensors, the VRAM expert tier, the grouped expert GEMV / GEMM, the uploads, the
stats and the multi-device placement: the `coli_cuda_*` ABI of `backend_cuda.h` (every entry point
of the CUDA build, by name and signature) and `coli_cuda_tier_ops()` (cu_tier_ops.h). The migrated
text is the base: the dead NVIDIA arms are deleted rather than carried (§7's list, plus the
`COLI_ANS` block and the HIP arms, resolved by a preprocessor pass over the known macros), the
DPCT diagnostic comments stripped (every site's disposition is the appendix's `-> done`), and
the hand edits below made. 6393 lines from 11147, 1500 from 1980, 844 from 1125; two new small
headers: `backend_sycl.h` (the error protocol, the shuffles, the SG launch; shared by the three
units) and a rewritten `sycl/backend_gpu_compat.h` (SYCL only: no WMMA, no PDL, no guard needed:
`-DCOLI_GPU_PORTABLE_ONLY` is harmless, defined or not). Nothing was run on a device.

### 10.1 What is gone, and what took its place

| CUDA | here |
|---|---|
| `w4a16_matmul`, `w4a16_gate_up`, `grouped_s4_wmma`, `quantize_s4_rows` (WMMA, the `COLI_CUDA_TC_INT4` / `_TC_W4A16` branches of `expert_group_impl`) | `coli_sycl_xmx_gemm_glu` + `coli_sycl_xmx_gemm` per expert with enough rows (the `COLI_CUDA_TC_W4A16` branch, on by default now, `=0` keeps the block kernels), behind `coli_sycl_xmx_gemm_ok`; `coli_cuda_shared_mlp_w4a16` the same pair for the shared expert |
| `tcg_*` (TF32), `hcg_*` / `hcg2_*` (f16 WMMA), `hcm_*` (f16 mma.sync over GGUF blocks), `f32t_*`, `HCG_BT_LAUNCH`, `hcg_part` / `hcg_part_sum`, `hcg_ok` / `tcg_ok`, `hcm_moe`, `moe_rows_launch`, `moe_rows_bd_launch`, `moe_tile_rows`, the `tcg_moe_*` / `hcg_moe_*` / `hcm_moe_*` kernels | the dense prompt GEMM of `coli_cuda_tensor_gemm_async` calls `coli_sycl_xmx_gemm` (int8 / int4 / int4-g64 rows, K a multiple of 64) where `coli_sycl_xmx_gemm_ok` says so, else `quant_matmul` (the block kernel, as on a device without tensor cores); the routed experts of a prompt chunk (`coli_cuda_moe_async`, more than `CC_MOE_FRAME_ROWS` rows) sort the slots by expert (`moe_sort_*`, kept) and call `coli_sycl_xmx_moe` for gate / up and down, `moe_rows_sum_k` after (`moe_xmx_launch`), where `coli_sycl_xmx_moe_ok` says so, else the warp kernels (`moe_gv_*`, `moe_gu_k` / `moe_dn_k`) at any row count; a GGUF's prompt rows stay on the GEMV kernels (`moe_bd_launch`, `bd_gemv_launch`: the family takes the tier's rows alone); `coli_cuda_tensor_gemm_hcmix_async` and `coli_cuda_moe_cols_async` return -1 (not taken), `coli_cuda_tensor_glu_async` 0 past `BD_GEMV_MAX_S` rows; a one-line `[SYCL] <site>: XMX joint_matrix GEMM` / `portable block kernels` note at the first use of each site (`xmx_note`) |
| `coli_cuda_planar_rows` (`hcg_ok` and widths of 64) | `coli_sycl_xmx_gemm_ok(p, 4, 64, D, I) && (.., I, D)` with the widths: planar entries are read by the matrix engines' grouped GEMM alone, so a table with them is refused unless that path is taken |
| `cudaMallocAsync` / `cudaFreeAsync` (gemv_i8_w4's split-K partials) | `coli_sycl_scratch(t->device, kz * S * O * 4, 0)`, never freed; the arena refusing keeps the unsplit kernel, as a failed cudaMallocAsync did |
| `cudaHostRegister` / `Unregister` | 0 with a one-line note (the tier's host pool is off: SYCL registers no existing host memory), a no-op |
| `cudaMemcpyPeerAsync` (the GLM pipe's resident groups, `coli_cuda_pipe_peer_copy`) | `peer_copy`: a memcpy on the queue where both devices are in one SYCL context, else refused with a message (an error the caller reports); every device opened by `coli_sycl_open` has a context of its own, so a cross-device copy is a refusal until the pipe asks for a shared context (not qwen36's path) |
| `cudaFuncSetAttribute` (max dynamic shared memory, `bd_smem_raise`, the hcm tiles) | none: the device's SLM is the profile's `slm_bytes` (`ctx->smem_optin`: 128 KiB pvc, 64 KiB xe2), every launcher that sizes local memory checks against it and refuses or shrinks (`bd_gemv_*_launch`'s `nr`, the gemv_i8 rows' 48 KiB cap within it, `moe_launch`, `moe_bd_*_launch`) |
| `__constant__ c_e8_grid`, `c_e4m3` (dpct: a single `constant_memory` for the process) | a device array per context (`DeviceContext.e8_grid`, `.e4m3`: zeroed at init, filled by `coli_cuda_e8_set_grid` / `coli_cuda_fp8_set_lut` for every open device); the kernels take plain pointers; cu_xb's `cu_e2m1` likewise (`C.e2m1`) |
| `cudaMemcpy2D` (`bd_rows_copy`, `coli_cuda_pipe_copy2d`) | `queue.ext_oneapi_memcpy2d(...).wait()` |
| `cudaDeviceSynchronize` (`coli_cuda_pipe_sync`) | both of the device's queues `wait_and_throw` |
| `coli_cuda_load_module` | 1: AOT images, nothing lazy to ask for |
| the `COLI_ANS` (dietgpu) block, `coli_cuda_tensor_upload_compressed` | deleted (never built here): the entry is defined by no build without `COLI_ANS`, as `backend_cuda.cu` had it under `#ifdef COLI_ANS` |

### 10.2 The host side, construct by construct

| CUDA | here |
|---|---|
| `cudaSetDevice` (`select_ctx`, thread-local cache) | `select_ctx` records the thread's ordinal (a device `coli_sycl_open` opened, `coli_sycl_is_open`); `dq()` is that device's in-order compute queue (`coli_sycl_queue`), where the legacy default stream's work goes: the allocations (`sycl::malloc_device` / `malloc_host` on it), the synchronous uploads (`memcpy(...).wait()`), `coli_cuda_matmul`'s round trips; `cur_ctx()` the context (the tables for a launch on the chain's queue) |
| `ctx->stream` (the non-blocking stream of the expert groups) | the same queue (`&coli_sycl_queue(device)`): one in-order queue a device orders everything the CUDA code ordered with events and syncs, at the price of the overlap the two streams had between a group in flight and a resident matmul (the chain's queue is this one too: `cc_init` takes `coli_sycl_queue(device)`) |
| `cudaStreamCreate` / `Destroy` | none: `coli_cuda_init` opens the devices (`coli_sycl_open`: a context, two queues, the profile each), `coli_cuda_shutdown` closes them (`coli_sycl_close`, after the chain's shutdown: atexit's order is the engine's convention); a re-init with another set is refused as before |
| `cudaGetDeviceCount`, `cudaGetDeviceProperties` | `coli_sycl_available_device_count()` (the GPUs in PCI order) and `coli_sycl_profile_for(device)`: `compute_major = compute_minor = 0` (every `>= 7 / 8` gate off), `sms` = Xe-cores (`eu_count / 8`: what the grid heuristics tuned to SMs count), `smem_optin` = `slm_bytes`, `sg` = `sg_default` (32 on pvc, 16 elsewhere; `COLI_SYCL_SG`), `wpb = 256 / sg` (sub-groups a block of 256: CUDA's "8 warps"), `has_fp64` = the fp64 aspect; the init prints the CUDA line the tools look for (`[CUDA] device N: <name>, <GB> GB VRAM, sm_00`) and `[SYCL] device N: <name> arch=<pvc|bmg|..> sg=<default>/<sizes> eu=<n> slm=<KB> xmx=<yes:f16,int8,..|no> mem=<GB> profile=<name>` |
| `cudaMemGetInfo` | `dev_mem_info` / `cu_mem_info` call `coli_sycl_mem_info`: `global_mem_size`, and `ext::intel::info::device::free_memory` where the device has the aspect (Level Zero with `ZES_ENABLE_SYSMAN=1`, which the GPU list sets before the runtime's first call and the image's `ENV` carries), else the total less the backend's books (the arena, the resident tensors by `coli_cuda_stats`, the chain's `dev_bytes`, cu_xb's pool totals: a ledger each unit registers at init), since the engine budgets on this figure (`qwen36_tier`'s allowance, `q36cc_rows_max`'s prompt chunk, the staging guard) and free = total on a filled card would plan it twice (§12, fix 3). `coli_cuda_alloc_footprint`'s probe measures nothing without the query (its blocks are not booked) and keeps the logical size, the pre-existing behaviour |
| `cudaDeviceProp.integrated` | the profile's `is_integrated` (`coli_cuda_device_integrated`, `cu_open`) |
| the error protocol (`cuda_ok` on an error code) | `COLI_SYCL_TRY(expr)` runs a call in a try and gives 0 / 1 (`backend_sycl.h`), every entry point is dpct's function-try-block whose handler `sycl_caught` logs and returns the ABI's failure (0, NULL, nothing; `coli_cuda_alloc_footprint` its logical size); either sets `g_sycl_lost`, which `fault_injected` reads, and `fault_injected` also polls `coli_sycl_lost` for an asynchronous error a queue's handler took (§12, fix 4): every compute entry then refuses, as under `COLI_GPU_FAIL_AFTER`, and the engine keeps its CPU paths. `cuda_ok(0, "... launch")` after a submit stays as the check that always passes (a submit that fails throws into the handler) |
| `cudaMemcpyAsync` from pageable memory (DPCT1124, 34 sites) | `.wait()` where the source dies before the copy would read it: the group descriptors (a stack array, three sites), the ragged attention's page tables and lengths (freed in the function), the resident group's weights (the caller's, and the issue returns at once); the pinned staging buffers (`host_x`, `host_y`, `host_kv`, `moe_host`) and a caller's buffer the function syncs before returning stay asynchronous (the appendix names each) |
| `cudaEventRecord` / `cudaEventElapsedTime` (`COLI_CUDA_PROFILE`, cu_xb's batch time) | `sync_barrier(ev, q)`: `ev = q->ext_oneapi_submit_barrier()`; the profiling read is in a try of its own: the backend's queues are made without profiling unless `COLI_SYCL_PROFILING=1` (the times stay 0), cu_xb makes its two queues with it (`enable_profiling`), so `xb_join`'s elapsed time is real. `DPCT_PROFILING_ENABLED` is no longer defined anywhere: nothing creates a dpct queue |
| cu_xb's device, streams, pools | the device is `coli_cuda_device_at(0)` (opened by `coli_cuda_init`), its profile the name, memory, SLM and sub-group; `C.xs` / `C.cs` two in-order queues of the unit's own on the device's context; the pools' blocks `sycl::malloc_device` on that queue (the host pool `malloc_host`, but off: `C.mapped = 0`, no mapped host memory the device reads in place, as `coli_cuda_host_register` says); `host_dev_ptr` is the identity (USM host memory has one address on both sides) |
| `dpct::get_current_device_id()` in `bd_xq_scratch` | the buffers are keyed by queue alone and live on the queue's context |

### 10.3 The kernels: SG, local memory, divergence

Every kernel the tool had pinned at `reqd_sub_group_size(32)` is a template over `SG` launched
through `COLI_SG_LAUNCH(q, sg, grid, 256, slm_bytes, kernel<..., SG>(args..., dpct_local))`
(`backend_sycl.h` over `coli_sycl_launch_sg`): both instances are in the binary and the device's
`sg` picks, so a kernel's lane arithmetic and shuffle tree are compiled for its width. Lane
arithmetic is SG's throughout: `lane = id & (SG - 1)`, `warp = id / SG`, strides `+= SG * 16`
(the 16-byte-a-lane loops), trees from `SG / 2`, a block's warps `256 / SG` (the host's
`ctx->wpb`: the grids that counted "8 warps a block" use it, `GEMV_I8_ROWS` is `wpb`,
`MOE_ROWS(wpb)`, the f8 and GGUF GEMVs' `bx`); `moe_gv_lanes` and `bd_lpr` cap the lanes a row
at `sg` (a row over `lpr <= SG` lanes, `rps = SG / lpr` rows a sub-group). The shuffles are
`sycl::shift_group_left` / `sycl::permute_group_by_xor` over the real sub-group (`shfl_down`,
`shfl_xor`), never dpct's width-32 helper.

| kernel | SG-relative edits |
|---|---|
| `f8w_block<HW, DUAL, SG>` | `CPL = 32 / SG` 4-byte chunks a lane of the 128-column block (one at 32: the CUDA warp's arithmetic exactly; two at 16, a lane's two dots added before the tree) |
| `quant_matmul_f8w<SG>`, `grouped_hidden_f8w_dual<HW, SG>`, `grouped_down_f8w<HW, SG>` | `warp = id / SG`, `nw = 256 / SG`, `o = block * (256 / SG) + warp`; the host grids `(I + wpb - 1) / wpb` |
| `dn_block_sum<SG>`, `dn_head_kernel<SG>` | the tree from `SG / 2`, `nw` sub-groups a block; `threads` rounded to `sg` (<= 256: `red[32]` holds the per-sub-group sums at 16 too) |
| `gemv_i8_rows<NR, PAD, SG>`, `gemv_i8_w4<NR, SG>` | `o = block * (256 / SG) + warp` (`o0 = (...) * GEMV_I8_OPW`), `i0 += SG * 16`, the tree; the host's `bx` / `bw` over `wpb` |
| `moe_lane_dot<FMT, SG>`, `moe_warp_sum<SG>`, `moe_gu_k<FMT, SG>`, `moe_dn_k<FMT, SG>` | `i0 += SG * 16`; FMT 8: a lane's 4-byte chunks of the 128-column block `i0 += SG * 4` (one at 32); `o = block * MOE_ROWS(256 / SG) + r * (256 / SG) + warp` |
| `moe_gv_gu_k<FMT, SG>`, `moe_gv_dn_k<FMT, SG>`, `moe_bd_gu_k<TG, SG>`, `moe_bd_dn_k<TD, SG>` | `rps = SG / lpr`, the unit / row strides `(256 / SG) * rps`; the host's `upw = wpb * (sg / l0)` (= 256 / l0 as before) |
| `bd_gemv_i8_k<T, NR, GQ, XS, GLU, SG>`, `bd_gemv_rows_i8_k<T, SG>` | `rps = SG / lpr`, strides `(256 / SG) * rps` |
| `bd_gemv_f_k<NR, SG>`, `bd_gemv_bf16_k<NR, SG>` | `o = block * (256 / SG) + warp`, `u += SG` / `c += SG`, the tree; the host's `bx` over `wpb` |
| `cu_gemv_k<FMT, NR, SG>` | `o = block * (256 / SG) + id / SG`, `wi += SG` / `g += SG`, the tree; `bx` over `wpb` |

Numerics: at SG 32 every kernel does the CUDA warp's operations in its order (the same bits); at
SG 16 a lane sums twice the chunks before a tree of 4 steps instead of 5, so a row's float sum
is the same products in another association, the "another order of the float additions" the
CUDA kernels already allow against the CPU. The block-tree kernels (`quant_matmul`, the
`grouped_*` family but the f8w pair, the attention absorbs, `f32_*`, the `pipe_*` kernels) have
no lane arithmetic and take the compiler's sub-group size through a plain `parallel_for`.

Local memory (DPCT1083, 10 sites): every `local_accessor<uint8_t>` is sized with the expression
the CUDA launch passed as dynamic shared memory, and the kernel casts `dpct_local` as it cast
`extern __shared__` (`float4` in the gemv rows, `float` in the moe pairs and the absorbs,
`uint4` in the GGUF GEMVs); the static `group_local_memory` arrays are the `__shared__`
arrays (the largest `f32_gemm_k<64>`'s 34 KiB). Every dynamic size is checked against the
profile's SLM by its launcher (§10.1), the absorbs' 38 KiB at the limits of their arguments
within 64 KiB.

Divergence (DPCT1118, 30 sites): every `return` before a barrier is block-uniform (`s >= d.rows`,
`e < 0`, `!ex.g`, `j >= p.S * p.K`: from `get_group` and the parameters), and every shuffle tree
is reached by whole sub-groups (`o >= O` / `o >= I` is a sub-group's own row; `nrow` and `live`
guard the stores, not the shuffles; the `lpr`-lane joins run on every lane). The barriers stay
as dpct emitted them (the `local_space` tuning of DPCT1065 is not taken).

Double sums: `quant_matmul_f8w` and the f8w pair (`f8_warp_mode`), `dn_head_kernel` and the
pipe's `rmsnorm` accumulate in double as the CUDA kernels did; a device without the fp64 aspect
(`ctx->has_fp64`) takes the block kernel for fmt 8 and returns 0 (the CPU path) from
`coli_cuda_dn_step` and `coli_cuda_pipe_rmsnorm*`.

### 10.4 The matrix engines' hooks

`xmx_dense_ok(ctx, fmt, gs, K, N)` is `coli_sycl_xmx_gemm_ok` over the device's profile; the
three dense sites (`coli_cuda_tensor_gemm_async` past `GEMV_I8_MAX_S` rows, `expert_group_impl`'s
per-expert branch, `coli_cuda_shared_mlp_w4a16`) build a `coli_sycl_gemm` (C = y, A = the f32
activations, W and scales from the tensor, fmt, gs, M, K, N) and take 1 from the family or fall
back; `coli_cuda_moe_async`'s `rows` is `coli_sycl_xmx_moe_ok(prof, *p)` and the widths (a pure
function of the call, so a later call with parts 2 alone finds the sorted layout where it is).
With the weak stubs every guard is 0: prompt rows run the block and warp kernels, planar tables
are refused, which is the first build's intended state.

### 10.5 Build evidence (container, no GPU)

- `icpx -fsycl -std=c++17 -O2 -fsycl-targets=spir64 -fp-model=precise -Wall -Wextra -I c -I c/sycl/include -c`
  of `backend_cuda.dp.cpp` and `cu_xb.dp.cpp`: exit 0, 0 errors each (the warnings are the
  migrated text's `-Wsign-compare` / unused `item_ct1` and dpct's headers).
- AOT: `-fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device pvc,bmg"` of both units:
  exit 0, 0 errors, 0 warnings (`<ops>/work/ee-backend-aot.log`, `ee-xb-aot.log`).
- a link of `tests/test_backend_cuda.dp.cpp` with both AOT objects, `coli_sycl_profile.cpp`,
  `coli_sycl_profile_rules.cpp` and `coli_sycl_stubs.cpp`: exit 0 (`<ops>/work/ee-link-test.log`),
  a 39 MB binary carrying the pvc and bmg images.
- the Makefile's build (`make sycl-compile SYCL=1`, then `make qwen36 colibri backend_cuda_test
  SYCL=1 ARCH=x86-64-v3`) in a copy of the tree: `<ops>/work/ee-make.log`.

Open, for the first run on a card: the SG16 sums against the CPU references (`test_backend_cuda`
at `COLI_SYCL_SG=16` and `=32` on a part that offers both; the moe and gemv kernels of the decode
path first); `coli_cuda_alloc_footprint` without sysman (a probe that measures nothing);
`dpct::dp4a` in the GGUF GEMVs (four multiplies: whether IGC folds them into the dp4a instruction
is a look at the assembly); the one-queue ordering's cost against the two CUDA streams' overlap.

## 11. Build (integrate, 2026-11-07)

The four Edit-phase commits (c53a748c xmx, 4cd0d664 + ebc5e4c1 profile / Makefile / image, eb20602e
chain, 8d12a91d backend) were integrated as they stood: no duplicate definition and no header
disagreement was left between the units (`coli_sycl.h` is the one interface; every `coli_sycl_*`
of sections 1, 2 and 4 has exactly one strong definition, in `coli_sycl_profile.cpp`,
`coli_sycl_profile_rules.cpp` or `coli_sycl_xmx.cpp`, and `coli_sycl_stubs.cpp` keeps weak
definitions only for the family the XMX unit implements plus the attention pair
`coli_sycl_xmx_attn` / `_ok`, which no unit implements yet: the chain keeps `cc_attn_k` /
`cc_attn_gk` for every prompt row until it does). The integrate edits: `c/coli`'s `cuda_binary()`
counts `libsycl` as a GPU build (the launcher refused `--gpu` on a SYCL engine), the LIST path
prints an explicit "no SYCL GPU device" line on an empty list, two independent `if`s of the
migrated ragged-KV cleanup dedented (a `-Wmisleading-indentation` that read as a nesting),
`Dockerfile.sycl` (see below) and the README's first-contact line (qwen36 enters the backend
only with `COLI_CUDA=1`, after the model loads: the test binary is the immediate form).
Nothing touched a GPU: container compiles, host-only tests, a CPU run of the engine.

### 11.1 The worktree build (basekit container, no `--device`)

    docker run --rm -u 1000:1000 -e HOME=/tmp -v <ops>/work:/work \
        -w /work/source-sycl/c --entrypoint bash intel/oneapi-basekit:2025.3.2-0-devel-ubuntu24.04 -lc \
        'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1;
         make sycl-compile SYCL=1 ARCH=x86-64-v4 -j8;
         make qwen36 colibri sycl-test SYCL=1 ARCH=x86-64-v4 -j8'

from a tree with every SYCL product removed first (`sycl/*.o`, the test binaries, the engines).
Result (`SYCL_COMPILE_EXIT=0`, `FULL_EXIT=0`; icpx 2025.3.3, gcc 13.3):

- the seven units compiled with the Makefile's `ICPXFLAGS` (`-O3 -std=c++17 -fsycl
  -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device pvc,bmg" -fp-model=precise
  -fPIE -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -I. -Isycl -Isycl/include
  -DCOLI_GPU_PORTABLE_ONLY=1`): `backend_cuda.dp.cpp`, `coli_sycl_profile.cpp`,
  `coli_sycl_profile_rules.cpp`, `coli_sycl_stubs.cpp`, `coli_sycl_xmx.cpp`, `cuda_chain.dp.cpp`,
  `cu_xb.dp.cpp`: **0 errors**;
- the device link `icpx <AOT flags> -fPIE -fsycl-link <7 objects> -o sycl/coli_sycl_dev.o`
  (46 MB): ocloc `Build succeeded for : pvc.` and `bmg.` for every module (117 of each over the
  whole build, the four test links included);
- the engine links, unchanged gcc lines: `gcc -O3 -march=x86-64-v4 ... -o qwen36 -lm -fopenmp
  -pthread -lsycl -lstdc++ -lintlc -limf -lsvml -lirng` and the same for `colibri`: 54 MB each,
  79 `T coli_cuda_*` symbols (`coli_cuda_init`, `_planar_rows`, `_moe_async`,
  `_tensor_gemm_async`, `_tier_ops`, `_chain_ops`, `_host_register`, `_shutdown` among them),
  the `.tgtimg` device-image section, `NEEDED libsycl.so.8 libintlc.so.5 libgomp.so.1`, no TEXTREL;
- `sycl-test`: the host-only `sycl/tests/test_sycl_profile` built with g++ and run
  (`PASSED: 95 checks, 0 failed`), then the four device tests linked by icpx (each its own AOT
  device link): `cuda_chain_test` (52.7 MB), `backend_cuda_test` (52.4 MB),
  `sycl/tests/test_coli_sycl_launch` (52.4 MB), `sycl/tests/test_sycl_xmx` (52.4 MB). Not run
  (they need a card).

Warnings, triaged (85 compiler warnings, 0 after the integrate edit were new to this phase):

| count | kind | where | disposition |
| --- | --- | --- | --- |
| 58 | `-Wsign-compare` | `backend_cuda.dp.cpp` (the migrated text: `item_ct1.get_local_id(2) < s`, size_t against int) and dpct's `util.hpp` | the .cu's own comparisons, unsigned ids from the SYCL API: harmless, left |
| 12 | `-Wbitwise-op-parentheses` | `sycl/include/dpct/util.hpp` | dpct's header, not ours |
| 10 | `-Wmisleading-indentation` | `backend_cuda.dp.cpp` 1252 / 1290 / 1324 / 1352 (the .cu's one-line `for (...) a += ...; qa[k] = a;` style, the body is the one statement in both) and 3850 (two independent `if`s dpct indented as nested: dedented here) | 4 cosmetic (the .cu's own layout, untouched), 1 fixed |
| 4 | `-Wreorder-ctor` | `sycl/include/dpct/bindless_images.hpp` | dpct's header |
| 1 | `-Wunused-variable` | dpct's `memory.hpp` | dpct's header |

and from IGC during the AOT (not compiler warnings): 45 `[RetryManager] Start recompilation of the
kernel` notes and the register-spill notes of the same kernels, all in `cuda_chain.dp.cpp`'s
`cc_dnrec` lambdas (the dnrec rows / prep / norm kernels and the pinned split at SG32: spills of
115..900 bytes, SIMD32, 128 or 256 registers), `cc_attn_t<half>` (6), `cc_addnorm` (16..35),
`cc_decg_launch<half>` at SG16 (35), and `bd_gemv_i8_go<10,4,...>` at SG32 (4); and the XMX unit's
plain-tile fallback `XmxGemmK<2,2,8,half,false>` on bmg (3). Performance notes for the first
profile on a card, not correctness: the heavy spillers are the dnrec rows path at sub-group 32
(the SG16 instances spill less or not at all), i.e. the PVC default.

### 11.2 Host-only tests

- `./sycl/tests/test_sycl_profile` in the basekit container (no device, no SYCL runtime linked):
  `PASSED: 95 checks, 0 failed`; `make sycl-profile-test` (no `SYCL=1`): the same.
- the same binary inside the runtime image (`/app/tests/test_sycl_profile`): PASSED.

### 11.3 The image

    docker build -f c/sycl/Dockerfile.sycl -t colibri-sycl:dev <ops>/work/source-sycl

(from the repo root, no GPU). Three things the first build taught, each now in the Dockerfile:

1. `setvars.sh` returns 3 in the basekit image (its ENV already ran it, `SETVARS_COMPLETED=1`), so
   `source setvars.sh && make` ran nothing and failed with make's exit code 3 and no output; the
   status is no longer chained (icpx is on the image's PATH either way).
2. the runtime image already has a uid-1000 user (`ubuntu`) and already ships python3 3.12 and
   libgomp1: the `apt-get` was redundant and `useradd -m -u 1000 colibri` failed ("UID 1000 is
   not unique"). The stock user is renamed to `colibri` (uid 1000, the bind-mounted model's usual
   owner), created only where no uid 1000 exists.
3. the runtime image 2025.3.1 does not ship `libumf.so.1`, which both UR adapters
   (`libur_adapter_level_zero.so.0`, `libur_adapter_opencl.so.0`) link against: `ldd` of the
   adapter said `not found`, so libsycl could have loaded no adapter and the engine would have
   seen no GPU on any card, with the same `0 GPU devices` a missing `/dev/dri` prints. The
   basekit's umf 1.0 (`libumf.so.1`, the file `libumf.so.1.0.2`) and the `libhwloc.so.15` it needs are copied next to
   libsycl. Found by step 6's `ldd`, before any card was involved.

The image's build stage ran `make -C c -j8 qwen36 colibri sycl-test SYCL=1 ARCH=x86-64-v4
SYCL_DEVICES=pvc,bmg` from a clean copy of `c/` (`.dockerignore` drops every `.o`): 0 errors,
117 + 117 ocloc successes, the host profile test PASSED inside the build, 245 s for the make.
Final image: `colibri-sycl:dev`, id ``3d87e817fa5b` (`sha256:3d87e817fa5b14738c219f2a521502a3047701ccdad95f9aa5e75336832b2ee9`)`, 6.1 GB (the runtime image is 5.7 GB of it);
`/app`: `qwen36`, `colibri`, `coli` and its files, `tools/`; `/app/tests`: `backend_cuda_test`,
`cuda_chain_test`, `test_coli_sycl_launch`, `test_sycl_xmx`, `test_sycl_profile`.

### 11.4 CPU smoke in the runtime image (`COLI_CUDA` unset, no `--device`)

    docker run --rm -u 1000:1000 -v <models>:/models:ro -v <dir>:/prompt:ro \
        -e SNAP=/models/qwen36-35b-a3b -e TOK=/models/qwen36-35b-a3b/tokenizer.json \
        -e N_NEW=8 -e OMP_NUM_THREADS=8 --entrypoint /app/qwen36 colibri-sycl:dev 256 4 /prompt/hello.txt

(the prompt as a mounted file: a `docker run` without `-i` has `/dev/null` on stdin, and the
engine `fseek`s its prompt file). The engine loaded with libsycl.so.8 mapped and generated on the
CPU: `[enc] prompt tokens: 1 | generating 8 new tokens`, `resident weights loaded in 5.9s | RSS
after load: 2.95 GB`, `TTFT: 0.686 s`, `Speed: 2.60 tok/s (3.1s for 8 tokens)`, `PEAK RSS:
4.31 GB`, the text `, I have a problem with the`, exit 0 (the run on the image before the libumf
copy: the same, 0.585 s / 2.96 tok/s). No `[SYCL]` or `[CUDA]` line: with `COLI_CUDA` unset the tier never enters the
backend (qwen36_tier.c's `qt_init` returns before `coli_cuda_init`), so nothing enumerated a device.

### 11.5 `COLI_SYCL_LIST=1` without a device

    docker run --rm -e COLI_SYCL_LIST=1 --entrypoint /app/tests/backend_cuda_test colibri-sycl:dev

`coli_cuda_init`'s first call is `coli_sycl_available_device_count()`, which answers the variable
before any other query. Output, exit 0, no crash, no hang (the same from `cuda_chain_test --list`):

    0 GPU devices
      no SYCL GPU device: the runtime sees no Level Zero / OpenCL GPU (is /dev/dri passed to the
      container and the user in the render group? sycl-ls says the same in the vendor's words)

With the UR loader's own log on (`UR_LOG_LOADER=level:info;output:stderr`, the same container,
no device) the run shows what the words above cannot: `loaded adapter ... libur_adapter_level_zero.so.0`,
`libur_adapter_level_zero_v2.so.0` and `libur_adapter_opencl.so.0` from `/opt/intel/oneapi/redist/lib`
(the cuda / hip / native_cpu / offload adapters are absent, as expected), then `0 GPU devices`:
the adapters load, and the emptiness is the missing `/dev/dri`. On the image before the libumf
copy the same two lines would have come from an adapter that could not load at all.

### 11.6 `ldd` in the runtime image

`ldd /app/qwen36`: `libsycl.so.8 => /opt/intel/oneapi/redist/lib/libsycl.so.8`, `libintlc.so.5`,
`libimf.so`, `libsvml.so`, `libirng.so` and `libur_loader.so.0` from the same directory,
`libgomp.so.1` and `libstdc++.so.6` from the system; nothing `not found`. The adapters are
`dlopen`ed by `libur_loader`, so they are checked on their own:
`ldd /opt/intel/oneapi/redist/lib/libur_adapter_level_zero.so.0` resolves every entry (`libumf.so.1`, `libhwloc.so.15`, `libsvml.so`, `libirng.so`, `libimf.so`,
`libintlc.so.5` from `redist/lib`, the rest from the system; before the copy: `libumf.so.1 => not
found`), and `libur_adapter_opencl.so.0` has no `not found` line either. The launcher's
`cuda_binary('/app/qwen36')` (the `ldd` check `coli` uses for `--gpu`) is True in the image.

### 11.7 Not green, or not checked here

- nothing ran on a GPU: the first contact is README's procedure (`COLI_SYCL_LIST=1` with
  `--device /dev/dri`, `test_coli_sycl_launch`, `test_sycl_xmx --small`, `cuda_chain_test` at
  `COLI_SYCL_SG=16` and `=32`, then the engine);
- `coli_sycl_xmx_attn` has no implementation (weak stub): prompt attention runs the f32 kernels;
- the register spills of the `cc_dnrec` kernels at SG32 (11.1) are the first thing to profile on
  a Max; the GGUF prompt path has no XMX twin (section 10);
- `sycl-test` builds the device tests but cannot run them; `test_sycl_profile` is the only test
  that ran, in the basekit container, on the host and inside the runtime image.

## 12. Review fixes (fix, 2026-11-07)

The review of commit 588f1caf (four lenses: sub-group semantics, memory ordering / ABI, profile,
detection and build, the XMX kernels) returned sixteen findings; each was verified against the
tree before anything was changed, and every one held. The dispositions, numbered in the review's
order (the two combo_takes findings, 6 and 13, are one defect seen through two lenses):

1. **`cc_addnorm_k`'s epilogues (`cuda_chain.dp.cpp:258`, `:277`; `cuda_chain.cu:193`, `:212`)**
   — confirmed: dpct turned `__fmul_rn(__fmul_rn(__fmul_rn(v, 1.f + w), r), 1.f)` into
   `v * 1.f + w * r * 1.f` (v + w·r), a wrong value in every residual add + norm of every chain
   engine, invisible to the host-only tests. Fixed to `((v * (1.f + w)) * r) * 1.f` (the only
   two such sites: every other `__*_rn` / `__fdividef` of the four sources has a plain second
   operand); Appendix A notes both sites for a re-migration. `-fp-model=precise` keeps the order.
2. **The arena's grow waited for the queues inside a frame** — confirmed (the device-MoE forward
   is one frame; a hybrid layer's `cc_wait_host` is served after the next layer is recorded, so
   a wait while recording it never returns). `coli_sycl_scratch` never waits now: the old block
   goes to the device's retired list (`Dev::retired`, the shape of `CcCtx.adec_old`), freed by
   the new `coli_sycl_scratch_reap(device)` where the chain knows the queue passed the frames
   (`cc_submit(1)`, `cc_finish`, `cc_shutdown`, next to `cc_adec_reap`) and by `coli_sycl_close`;
   allocations carry headroom (`bytes + bytes/2`); only nine grows without a reap make the ninth
   wait, logged. Slot 0 is asked for `GEMV_I8_MAX_S` rows at its first use (a verify after a
   decode never regrows it). `bd_xq_scratch` and the XMX unit's copy retire the same way
   (`coli_sycl_retire(q, p, bytes)` for any block of an open device's context; the XMX copy is
   the arena's slot 3 outright, fix 15). *Not applied:* reserving slot 1 at `cc_init`: the chain
   knows no dnrec shape at init (the shapes arrive with the ops), the split path runs for
   `CC_DNREC_ROWS` (8) rows and more, and with the retire its first grow is one logged
   allocation, never a wait, so a guessed reservation would buy nothing. Section 9.2 corrected.
3. **`cudaMemGetInfo` answered free = total without the sysman aspect** — confirmed (nothing
   set `ZES_ENABLE_SYSMAN`; `qwen36_tier`'s allowance, `q36cc_rows_max` and the staging guard
   budget on the figure). `gpu_list` sets `ZES_ENABLE_SYSMAN=1` (`setenv` without overwrite)
   before `get_devices`, the process's first runtime call; the open line says `free memory from
   the device` or `estimated from the backend's books`; the image's `ENV` has it; the README's
   table has the row. Without the aspect `coli_sycl_mem_info` returns the total less the books:
   the arena's slots and retired blocks, plus a ledger each unit registers once at init
   (`coli_sycl_ledger_add`: the backend's resident tensors by `coli_cuda_stats(device)`, the
   chain's `g_st.dev_bytes`, which is one count for every device, cu_xb's device pool totals).
   `dev_mem_info` and `cu_mem_info` are now that one call. Section 10.2 corrected.
4. **Asynchronous errors only printed** — confirmed. The handler is a lambda holding its `Dev`
   and sets `Dev::lost` (atomic); `coli_sycl_lost(device)` (coli_sycl.h section 2) returns it,
   after polling both queues with `throw_asynchronous` so an error collected since the last wait
   counts; the chain reads it in `live()` and after every `cc_try` (`cc_async_lost`: the same
   `cc_fail`), the backend in `fault_injected` (every open device). The rethrow alternative was
   not taken: `coli_cuda_shutdown`'s waits have no try around them.
5. **Shutdown's frees before any wait; ordinals past the table** — confirmed both. (a)
   `coli_cuda_shutdown` drains both queues of a context before its first `sycl::free`;
   `coli_cuda_tensor_free` says it relies on the engine's `issue_open` / `chain_hold` ordering.
   (b) `coli_sycl_open` refuses an ordinal at or above `COLI_CUDA_MAX_DEVICES` with a message.
6. / 13. **`combo_takes` read the runtime's rows wrong** — confirmed, and the encoding verified
   here by scanning the `combination` initializer tables out of the basekit's `libsycl.so.8.0.0`
   (64-byte rows: six `size_t`, four `matrix_type`): Intel's rows are `(8,0,0,0,16,32)` s8,
   `(8,0,0,0,16,16)` f16 and bf16, `(8,0,0,0,16,8)` tf32, the fixed `(0,0,0,16,16,16)` f16, and
   dg2's `(8,0,0,0,8,16)` / `(8,0,0,0,8,32)`; M ranged, N and K fixed, per dimension. The rule
   is `dim_takes` per dimension; the test's fixture is rewritten in the runtime's encoding and
   keeps the all-fixed rows as an extra case, plus rows that must not take the tile (M up to 4,
   s8 with K 16, pvc's 32x64x16, an all-zero row); the header's comment says what a row is.
   Two decisions on top: `has_xmx` now requires the f16 8x16x16 flag (the tile every kernel of
   the family runs on), so `xmx on` means the engines run, and `COLI_SYCL_XMX=1` cannot force it
   without that shape; and the backend's guards (`xmx_dense_ok`, `coli_cuda_planar_rows`, the
   routed experts' `rows`) ask `coli_sycl_xmx_supported(p, F16)` before `coli_sycl_xmx_gemm_ok`,
   so a device without the shape (or `COLI_SYCL_XMX=0`) keeps the migrated warp and block
   kernels, which is what the README promised and the arithmetic of a CUDA device without
   tensor cores. `coli_sycl_xmx_gemm_ok` itself still answers for shape and format alone (the
   family's contract; `test_sycl_xmx --no-xmx` reaches the plain tile through it). No
   `xmx=fallback` banner: with the guard the plain tile never runs in the engine; the test
   warns when it runs there (fix 14).
7. **An unknown Intel part was `none`** — confirmed: `architecture::unknown` is
   `0x9900000000000000` in 2025.3's `.def`, the `>> 56 == 0` arm was dead (the switch names all
   31 values). `arch_of` maps `unknown` to OTHER; a value with no row on a device with Intel's
   vendor id (0x8086) is OTHER too; the README's table and detection text say so, and that the
   matrix query is keyed by the architecture's name inside the runtime, so such a part lists no
   shape and runs with the engines off. *Not applied:* moving the value → row table into the
   rules unit for the host test: the mapping is over the SYCL enum, type-checked against the
   `.def`; a copy of its constants in a unit without SYCL would drift from it silently.
8. **`default_sg` could return 8** — confirmed (theoretical for Intel GPUs). It returns 16 with
   a `[SYCL]` line when neither 16 nor 32 is offered; the test's `eight` case expects 16 and the
   log line.
9. **A header edit rebuilt nothing** — confirmed (`backend_sycl.h` was not in `SYCL_HDR`, no
   `-MMD`). `backend_sycl.h` is in `SYCL_HDR`, `sycl_test_args.h` on the three test rules, and
   the units compile with `-MMD -MP`, their `.d` files read back under `SYCL=1` (checked first
   that icpx `-fsycl` writes a sane one: the host compile's includes); `.gitignore`'s `sycl/*.d`
   now names something that exists.
10. **Dockerfile's first-contact example** — confirmed; it runs `/app/tests/backend_cuda_test`
    as the README does, with the engine's form spelled out.
11. **README's `readelf -S`** — confirmed; it says `.tgtimg`, where the bundle sections live,
    and the `readelf -d` NEEDED entries.
12. **Documentation consistency** — confirmed: `coli_sycl_profile.h:45` (the M is a range up to
    8, the compound shape a fixed row) and `:72` (SLM is the device's word: 128 KiB on pvc and
    the Arc B-series, 64 KiB on the Xe2 iGPUs) rewritten; `COLI_SYCL_XMX_TM` and
    `ZES_ENABLE_SYSMAN` in the README's table; §11.3 names the file `libumf.so.1.0.2` (the
    basekit's; the `1.0.3` was in the integrate step's result, not in this document).
14. **`test_sycl_xmx`'s tolerance over the whole output** — confirmed (row 0's sentinel sets
    a tolerance above every other row's values on the small and GLU cases). The dense and GLU
    cases are judged a row at a time (`row_max[i]`, row 0 then tests the saturation alone); the
    moe cases carry no sentinel and stay. The test prints a WARNING when the plain tile runs in
    place of the engines (no f16 shape, or `COLI_SYCL_XMX=0` without `--no-xmx`).
15. **The XMX scratch's static `sycl::device` / `sycl::context` copies** — confirmed as a
    hazard (not reproduced: no GPU). On an open device's compute queue the copy is the arena's
    slot 3 (freed by `coli_sycl_close`; a close / open cycle keys the new context's slot); a
    foreign queue (the standalone test's) keeps a private block in a heap-held list that is
    never destroyed, the rule the per-device state follows.

Files: `coli_sycl_profile.cpp`, `coli_sycl.h`, `coli_sycl_profile_rules.cpp`,
`coli_sycl_profile.h`, `cuda_chain.dp.cpp`, `backend_cuda.dp.cpp`, `cuda_bd.dp.hpp`,
`cu_xb.dp.cpp`, `coli_sycl_xmx.cpp`, `backend_sycl.h`, `tests/test_sycl_profile.cpp`,
`tests/test_sycl_xmx.cpp`, `Makefile`, `Dockerfile.sycl`, `README.md`, this file. The engine
files are untouched; the C ABI is unchanged (the new `coli_sycl_*` entry points are the SYCL
units' own, in `coli_sycl.h`).

### 12.1 Build evidence (container, no GPU)

All in `intel/oneapi-basekit:2025.3.2-0-devel-ubuntu24.04`, `docker run` without `--device`,
nothing ran on a card. (1) `make sycl-compile SYCL=1 ARCH=x86-64-v4 -j8`: the seven units and the
device link rebuilt (every unit, since `coli_sycl.h` changed), `SYCL_COMPILE_EXIT=0`, `grep -c
' error:'` 0, `Build succeeded for : pvc.` and `bmg.` for every module. (2) `make qwen36 colibri
sycl-test SYCL=1 ARCH=x86-64-v4 -j8`: `FULL_EXIT=0`, 0 errors, `test_sycl_profile` `PASSED: 104
checks, 0 failed` (95 before the review), `qwen36` 54.6 MB and `colibri` 54.2 MB linked by gcc,
`cuda_chain_test`, `backend_cuda_test`, `sycl/tests/test_coli_sycl_launch`,
`sycl/tests/test_sycl_xmx` built; the units now leave `sycl/<unit>.d` beside their objects. A
confirming pass over the final tree (the `coli_sycl_retire` narrowing landed while the first ran):
`CONFIRM_EXIT=0`, 0 errors, `nm` 79 `T coli_cuda_*` symbols in `qwen36`, `readelf -S` one
`.tgtimg`, `readelf -d` `NEEDED libsycl.so.8`, `libintlc.so.5`. Compiler warnings: the set §11.1
triaged, none from the edited lines; the IGC spill notes are the same kernels as before. The
host test also passes with the host's g++ 13 (`PASSED: 104 checks`). (3) `docker build -f
c/sycl/Dockerfile.sycl -t colibri-sycl:dev .`: `DOCKER_BUILD_EXIT=0`, 0 errors, 117 + 117 ocloc
successes, the profile test PASSED inside the build stage; image id
`sha256:7fdc776419f60ad28616894e48ccd99f1cb6438584ef0e1d25ad93ec3f07f5a0`. (4) In the runtime
image, no `--device`: `/app/tests/test_sycl_profile` `PASSED: 104 checks`; `ZES_ENABLE_SYSMAN=1`
is in the environment, the user is `colibri`; `COLI_SYCL_LIST=1 /app/tests/backend_cuda_test`
prints `0 GPU devices` and the no-device line, exit 0 (the Dockerfile's first-contact command);
`/app/tests/test_sycl_xmx --list` says `no GPU device found`, exit 2; `ldd /app/qwen36` has 0
`not found`, `libsycl.so.8`, `libintlc.so.5` and `libur_loader.so.0` from `/opt/intel/oneapi/redist/lib`.
Not checked here (needs a card): that the fixed `cc_addnorm` epilogue restores
`test_cuda_chain`'s `D ... x and its norm bit for bit` cases, that the flags read `f16 ... s8` on
a Max or a B580 and `xmx on` with the engines running (`test_sycl_xmx` without the WARNING),
that `free memory from the device` appears on the open line, and that a grow inside a frame
no longer stalls the token loop.

## 13. First card (fix, 2026-11-07)

The first run of the migrated chain test on a card (rung 4 of `<ops>/PLAN.md`: `/app/tests/cuda_chain_test
--device 0` on card1, a Data Center GPU Max 1100, `pvc-sg32`, xmx on; log
`<ops>/logs/gpu-rung4-chain-test-card1-20261107T185602Z.log`) passed about 200 checks and failed in three
places: five `addnorm` cases ("x and its norm bit for bit": D 64 normalised, D 2048 and D 2560 in both
branches; D 64 as it is passed), three `moe` cases ("a chunk's two lanes' sums over one parts 1 of its
rows: the chunk's sum's bits", int8, int4 and int4 per 64; fp8 passed), and then `signal, wait_host`,
where the runtime printed `Segmentation fault from GPU at 0xfff718802d2000 ... type: 2
(AtomicAccessViolation) ... access: 2 (Atomic), banned: 1, aborting` and aborted the process (rc 134,
3 s; dmesg: `i915 0000:18:00.0: Atomic access violation @ 0xfff718802d2000`, `Catastrophic memory
error in context cuda_chain_test (ccs0)`). The card was healthy afterwards (enumeration, error
counters 0, SEL unchanged at entry 34c). Every cause below was found and the fix checked without a
GPU (the basekit container, `ocloc` for pvc with IGC's shader dumps, the CUDA sources beside the
migrated ones); the card then confirmed them within the run budget (13.4).

### 13.1 `cc_signal` / `cc_wait_host`: device atomics on host memory

`cuda_chain.cu:1770-1771` ran a one-thread kernel for each flag: `cc_signal_k` a `__threadfence_system`
and a volatile store into a word of a DOWN buffer (mapped pinned host memory), `cc_wait_k` a volatile
spin on a word of an UP buffer until the host had written `v`. The Edit phase (§9.2) made both a
`sycl::atomic_ref<unsigned, relaxed, memory_scope::system>` store / load on the host-USM word, with
system-scope fences. On Ponte Vecchio a device atomic on host memory over PCIe is not a slow
operation but a fault: the GPU's page walk marks the access `Atomic` on a host PTE, i915 reports
`Atomic access violation`, the context is banned (`banned: 1`) and the UMD aborts the process
(`drm_neo.cpp:288`). Battlemage sits on the same bus with the same rule, so there is no device
form of this at all.

Both are now host tasks on the chain's in-order queue (`cc_flag_store`, `cc_flag_wait`;
`cuda_chain.dp.cpp`, the comment above them says all of this): `signal` submits
`h.host_task([=]{ __atomic_store_n(f, v, seq_cst); })`, which the runtime runs on its own thread once
every earlier command of the queue is complete (the step's rows the kernels wrote to DOWN memory are
there by then: their events completed) -- the word holds `v` exactly when the `.cu`'s kernel would
have stored it, and the host polls it as before; `wait_host` submits a host task that polls the UP
word with acquire loads until it is at least `v` (no timeout: the contract's "until the host has
written v", `cuda_chain.h:247-255`), spinning with `pause` for the first 1024 polls and napping 20 us
(`nanosleep`) between later ones, so the host's experts keep their cores during the milliseconds
they take; every later command of the in-order queue is held until it returns. The engine-facing
semantics and return values are unchanged (`launched(c, ...)` after a `cc_try` around the submit: a
submit that throws loses the context as a launch did). The words are host USM (`sycl::malloc_host`,
`cc_buf`): a host store is an ordinary cacheable write that the device's later reads (the copy or
kernel after the wait) see over PCIe, exactly as the `.cu`'s mapped pinned memory; nothing is copied
and no memory kind makes the store invisible. The host task holds no lock and calls nothing of the
chain; the runtime does not hold its scheduler lock while a host task's body runs, so the engine's
writer thread may submit on either queue (host_up, write, the next frame) meanwhile, and the
deadlock note at `cuda_chain.cu:64` / `dp.cpp:82` (a queue wait inside a frame could sit behind a
wait whose host side comes after the frame) applies to the host task exactly as it did to the
kernel. The hazard §9's open list named -- a kernel spinning on host memory under i915's hang check
with heartbeat 0 -- is gone with the kernel. One thread of the runtime's host-task pool is taken per
chain while a wait is pending (one engine per card: one). No compile-time option keeps the kernel
form: there is no card this code targets where it is legal.
*Open (review, 19:55Z):* a wait whose host side never comes keeps the host task spinning with no
way out -- `q36cc_dm_serve` returns 0 without writing the word after 60 s or when `cc_lost()`, and
where a lost CUDA context ended the spinning kernel, a lost SYCL context does not end the host
task: the next queue drain (`cc_submit(1)`, `cc_finish`, the shutdown's drain) blocks behind it,
and the runtime's pool thread stays taken. `cc_flag_wait` should also return once the context's
lost flag is set (an atomic word it may read without calling the chain), and the engine should
mark the chain lost when it gives up; not changed here.

### 13.2 `addnorm`: not contraction, the compiler's two lowerings of `1 / sqrt`

What the test compares (`tests/test_cuda_chain.dp.cpp`, "addnorm"): on the device, `norm` of the branch
(when `pre`), `ew` ADD and `norm` again, against `addnorm` in one pass, both results bit for bit --
the reference is the chain's own `cc_norm_k` and `cc_ew_k`, not a host function (the CPU reference
of `norm` is a separate check with a tolerance). The hypotheses in the task's order, each settled
offline by compiling the two kernels' exact bodies through the Makefile's pipeline (`icpx -fsycl
-fsycl-targets=spir64_gen ... -fp-model=precise`, `-fsycl-link`, `ocloc -device pvc`) with
`IGC_ShaderDumpEnable=1` and reading the ISA:

(a) *FMA contraction.* `-fp-model=precise` gives icpx `-ffp-contract=on` (the `cc1` line): clang
fuses only within one expression, emitting `llvm.fmuladd`, which the SPIR-V translator writes as
the OpenCL `mad` and IGC lowers to the hardware's fused `mad`. The SPIR-V carries no
`NoContraction` decoration under any flag (`#pragma clang fp contract(off)` and `-ffp-contract=off`
both produce the same SPIR-V and the same IGC module hash as each other, but only remove clang's
`fmuladd`), and IGC does not fuse undecorated `OpFMul` / `OpFAdd` pairs on its own (checked on a
probe kernel: a cross-statement `y + (x * w) * r` stays `mul`, `add`). So the only fused operations
are the syntactic `acc += v * v` in both kernels -- `cc_norm_k` 9 `mad`s (the 8 unrolled terms and
the rolled loop's), `cc_addnorm_k<8>` 16 (both sums), `<16>` 32 -- the same places nvcc fuses them
in the `.cu` (`--fmad=true` with `__fmul_rn` / `__fadd_rn` guarding only the two epilogues), and the
dpct-era epilogue fix of §12.1 (`((v * (1 + w)) * r) * 1`, then `xv + v` as a separate statement)
is not contracted. Not the cause; no flag or pragma changed.

(b) *The reduction.* Both kernels take a thread's elements `tid + 256 k`, the same `red[256]` tree,
the same masks; the sums' ISA is the same `mad` chain and the same tree. Not the cause.

(c) *`sqrt` and division.* The cause. Nothing in the build asks IGC for correctly rounded fp32
division or square root (`correctly-rounded-divide-sqrt-fp-math="false"` in IGC's IR;
`-fp-model=precise` does not imply `-foffload-fp32-prec-div` / `-prec-sqrt` for SPIR-V, and the
driver passes ocloc no `-options`), so IGC lowers them at OpenCL precision (2.5 / 3 ulp) and keeps
its pattern freedom. In `cc_addnorm_k` it matched `1.f / sycl::sqrt(x)` and emitted one
`math.rsqt`; in `cc_norm_k`, where the two branches of `flags & 4` (`1 / sqrt(sum + eps)` and
`1 / sqrt(sum / D + eps)`) meet before the division, it emitted `math.sqt` then `math.inv` -- two
roundings, a different `r` in some rows, and every element of that row's norm off by an ulp (the
ISA counts: norm `4 inv 2 sqt`, addnorm `2 inv 2 rsqt`). It explains the one passing case too: D 64
as it is has five `r` values (five rows), D 64 normalised ten, and the two lowerings agree on most
inputs. nvcc gave both kernels the same IEEE `1.f / sqrtf`, so the `.cu` never met this.

The fix (`cc_rstd`, `cuda_chain.dp.cpp`): one `__attribute__((noinline))` device function computes
`rsqrt(sum / n + eps)` (n 0 for the L2 branch) for both kernels. IGC honours `noinline` (a
`call ... cc_rstd` subroutine in every kernel's ISA, cloned per kernel but compiled from the same
body with no caller context around it: exactly one `math.inv` and one `math.rsqt` in each), and the
`rsqrt` is explicit -- what the matched kernel already computed on the card, within the hardware's
ulp as the `sqt` / `inv` pair was, and what the Vulkan shaders' `inversesqrt` computes. The CPU
reference of `norm` is unaffected (tolerance 1e-5; the difference is an ulp). Exactness turned out
possible without touching the test or any reference. *Not taken, recorded:* making the whole build
IEEE with `-Xsycl-target-backend=spir64_gen "-device pvc,bmg -options
-cl-fp32-correctly-rounded-divide-sqrt"` (ocloc accepts it; IGC then emits `math.invm` /
`math.rsqtm` macro sequences, which is what the `.cu`'s nvcc defaults `-prec-div` / `-prec-sqrt`
mean) -- it would carry the whole chain's and backend's divisions to IEEE at a cost not measured
today, it does not reach a JIT-compiled device, and in the dump it still left the two kernels
asymmetric (norm's `sum / D` stayed an `inv`, under a `!fpmath` that IGC's own IR carries on that division, addnorm's became `invm`),
so it is not a fix for this pair on its own. A candidate for the performance pass, with the PPL
reference in hand.

(d) *The `i < D` masks for NV > 1.* Identical in both kernels; the D 2048 / 2560 failures were
(c)'s with more rows' `r` to disagree on.

The test now prints, under a failing "bit for bit" line, how many words differ, the largest ulp
distance, and the first differing (row, col) with both values (`bits_differ`); it stays, for the
next card.

### 13.3 `moe`: a parts-2 lane refused by the matrix engines' `_ok`

`coli_sycl_xmx_moe_ok` (`coli_sycl_xmx.cpp`, `moe_shape_ok`) refused any call whose rows were not the
layout's (`lay_S != S || lay_s0`), with the comment "the sort's layout is this call's rows". True of a
parts 1 (`cc_moe` enforces it); false of a parts 2 alone over a slice of the layout's rows -- the
prompt chunk's lanes the test runs (rows 0..15 and 16..39 over the 40-row layout, each with its own
offsets, `CcMoe.lay_S` / `lay_s0`), whose sums the backend computes with `moe_rows_sum_k` over the
sort at `lay_s0 + s` (identical to the `.cu`'s). With `_ok` false the backend's `rows` was 0 for the
lanes and it took `moe_sum_k`, which reads `hid` in the unsorted layout: wrong sums, not an ulp. fp8
passed because it is not a matrix-engine format and takes `moe_sum_k` for the whole and the lanes
alike. `moe_shape_ok` now checks the rows are inside the layout (what `cc_moe` checks) and nothing
more; `coli_sycl_xmx_moe` itself (the GEMMs, a parts 1) refuses a lane as before. CUDA's `tcg_ok` /
`hcg_ok` look only at the device, which is why the `.cu` never had this.

### 13.4 Runs on card1 (all under `<ops>/guard.sh`, card1 only, the rebuilt test from the worktree mount)

| run | binary | result | log |
|---|---|---|---|
| 1 | the worktree's `c/cuda_chain_test` (built 19:25Z with 13.1-13.3) | 3 s, rc 1, no kernel lines, SEL 34c. `addnorm` 6 of 6 ok; `moe` lanes 4 of 4 ok; `signal, wait_host` ok ("eight rounds: each answer read after its word, before the next"); every section the rung-4 run never reached ran (pick, pick_rows, matmul_rows, the encoders' ops, hyper-connection mix, lost). 3 FAIL, all one cause (13.4a) | `<ops>/logs/gpu-rung4-fix-chain-test-card1-run1-20261107T192548Z.log` |
| 2 | the same, rebuilt 19:30Z with the test's hcmix skip (13.4a) | **2 s, rc 0: `OK test_cuda_chain: every op of the CUDA chain matches its reference`**; 228 ok, 0 FAIL, 2 skip (hcmix not taken), no kernel lines, SEL 34c | `<ops>/logs/gpu-rung4-fix-chain-test-card1-run2-20261107T193114Z.log` |
| 3 | `/app/tests/cuda_chain_test` of the rebuilt image `colibri-sycl:dev` (`ea77a30425d4`, 19:36Z, from this tree: 0 errors, 234 ocloc successes, the profile test PASSED in the build stage) | **4 s, rc 0, the same `OK` line**; 228 ok, 0 FAIL, 2 skip, no kernel lines, SEL 34c | `<ops>/logs/gpu-rung4-fix-chain-test-card1-run3-image-20261107T193624Z.log` |

Three of the eight runs the step allowed; every one in the one form `<ops>/PLAN.md` prescribes (card1 and
renderD128 alone, `ZE_AFFINITY_MASK=0`, unprivileged, `--init --stop-signal SIGKILL` with `timeout -s
KILL` inside and `GUARD_CONTAINER` set), `--device 0`, the default profile (`pvc-sg32`, xmx on); the
`COLI_SYCL_SG=16` pass of the README's step 4, `test_sycl_xmx`, `backend_cuda_test` and the engine are
the next rungs, not run here.

(a) The three FAILs of run 1 were the test's, not the chain's: "300 / 100 rows: the up projection and
the mix in one pass, those of the two ops to the bit" require `matmul_hcmix` to return 1, and "the
counters saw the frames, ops and matmuls" counts its two passes in `matmuls == 50`. The fused kernel
was `hcm_mix_k`, f16 `mma.sync` over GGUF blocks, deleted in the Edit phase (§10): the SYCL
`coli_cuda_tensor_gemm_hcmix_async` answers -1, the contract's "not taken" (`cuda_chain.h`: the
engine then runs the matmul and `CC_EW_HC_MIX` apart, which is what the test's own first frame does).
`tests/test_cuda_chain.cu` has the same lines (`r == 1`, `== 50`): a CUDA build with
`COLI_GPU_PORTABLE_ONLY` or HIP fails them the same way. The migrated test now says
`skip <what>: matmul_hcmix not taken on this backend (-1)` for a -1 and expects `48 + taken` matmuls;
a 0 (a failure) or differing bits of a taken pass fail as before. Nothing else was loosened.


### 13.5 Files
`cuda_chain.dp.cpp` (`cc_rstd`, `cc_flag_store` / `cc_flag_wait`, `cc_signal`, `cc_wait_host`, the
header comment), `coli_sycl_xmx.cpp` (`moe_shape_ok`, `coli_sycl_xmx_moe`),
`tests/test_cuda_chain.dp.cpp` (`bits_differ`; the hcmix skip and `hcmix_taken`), `README.md` (what to expect on the first runs),
this file (§9.2's flag row, §9.6's open list). No engine file, no ABI change, no Makefile change.

### 13.6 `backend_cuda_test`: the deliberate out-of-memory upload (backend-test, 2026-11-07; the card run of the fix pending)

Two guarded runs on card1 from the image `ea77a30425d4` (the one 13.4's run 3 used): `--list`, rc 0
(one device, `pvc-sg32`, sub-groups 16 and 32, xmx on; `<ops>/logs/gpu-rung4-backend-test-card1-run1-list-20261107T193935Z.log`),
then `/app/tests/backend_cuda_test --device 0`: rc 1 in 1 s, no kernel lines, no GPU fault, SEL 34c
(`<ops>/logs/gpu-rung4-backend-test-card1-run2-image-20261107T194016Z.log`). Every check up to the
test's deliberate ~16 TB `coli_cuda_tensor_upload` passed; there the log shows `[CUDA] call: SYCL
exception: NULL pointer argument in memory copy operation.`, `[CUDA] tensor upload: failed`, and on
stdout `Exception caught at File: backend_cuda.dp.cpp | Function: operator() | Line: 1900 | Column:
41` -- `sycl::queue::memcpy`'s own `report(CodeLoc)` before it rethrows the same exception, one throw
and no exit -- and `main` later returned 1 in silence (the CUDA test's `return 1`s).

Cause: `COLI_SYCL_TRY(p = sycl::malloc_device(...))` took the NULL the Level Zero adapter answers a
request it cannot meet with (the specification's failure form, no exception) as success, where
`cudaMalloc` returned `cudaErrorMemoryAllocation`; the copy into the NULL threw, `sycl_caught` marked
the backend lost (`g_sycl_lost`), and every later entry was refused -- the test's own "must not
poison later healthy launches" check, which is what failed.

Fix in the worktree (uncommitted at the time of writing, with this section): `COLI_SYCL_NOMEM`,
`sycl_nomem` and `COLI_SYCL_ALLOC(lhs, expr, n)` in `backend_sycl.h` (a NULL for a non-zero ask, or
an exception with `errc::memory_allocation`, is NOMEM and nothing is lost; any other exception is
`sycl_caught`'s as before; a zero-byte ask may answer NULL and is not a failure, as `cudaMalloc(0)`);
`cuda_ok` prints `out of memory` for NOMEM (`cudaGetErrorString`'s words, the line the test's
"expecting one out-of-memory line next" announces); 35 allocation sites rewritten (22 in
`backend_cuda.dp.cpp`, 1 in `cuda_bd.dp.hpp`, 12 in `cu_xb.dp.cpp`; no `COLI_SYCL_TRY(... malloc_...)`
remains; the chain's `cc_alloc`, the XMX scratch and the arena already handled NULL);
`tests/test_backend_cuda.dp.cpp` prints `FAIL test_backend_cuda.dp.cpp:<line>` (`fail_line`) at each
of the formerly silent `return 1`s of `main`; `README.md` step 5. `tests/test_backend_cuda.cu` is not
edited.

Build: `make sycl-test SYCL=1 ARCH=x86-64-v4 -j8` (basekit container, no GPU) completed on these
sources -- `c/backend_cuda_test` (19:49:57Z) and `c/cuda_chain_test` (19:50:01Z) relinked, and a
re-run of the same make is a no-op with exit 0 (review, 19:55Z). Not yet done: the guarded run of the
worktree's `c/backend_cuda_test --device 0` on card1 (expected: one `[CUDA] tensor allocation: out of
memory` line and `cuda backend: q8/q4/q2/f32/e8 correctness ok on 1 device(s)`, exit 0; everything
after the 16 TB case -- the fmt 7 smoke upload, the fault-injection hook, `tensor_update`, the
matmuls, `expert_mlp` / `expert_group` in every variant, `group_stats`, `stats`, fmt 6 and fmt 8 --
has not run on a card yet), the image rebuild (`ea77a30425d4` predates this fix), one confirming run
from the image, and the commit.

## Appendix A. Every DPCT site (run 3), file:line of `c/*.cu`, disposition

Dispositions over 523 sites: auto-ok 230, delete-dead-code 1, needs-hand-edit 292


#### backend_cuda.cu (236 sites: auto-ok 94, needs-hand-edit 142; every site resolved, "-> done" says how: §10)

- `backend_cuda.cu:170` DPCT1009 **needs-hand-edit** — cudaGetErrorString -> exception text -> done: cuda_ok prints "<what>: failed"; the exception's text was printed where it was caught
- `backend_cuda.cu:171` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: nothing sticky to consume
- `backend_cuda.cu:190` DPCT1093 **needs-hand-edit** — dpct::select_device(ordinal): the ordinal is the engine's, use coli_sycl_queue(device) (GPU list, PCI order) -> done: select_ctx records the ordinal (coli_sycl_is_open checked)
- `backend_cuda.cu:414` DPCT1108 **auto-ok** — masked sub-group helper (experimental): keep, or drop the full mask for the plain dpct helper -> done: as DPCT1064:60
- `backend_cuda.cu:414` DPCT1064 **needs-hand-edit** — the shuffle macro's width is the kernel's SG, not 32 (COLI_SYCL_SHFL_WIDTH) -> done: f8_shfl_down is shfl_down<float / double>(v, off) over the SG sub-group (f8w_block<HW, DUAL, SG>, dn_block_sum<SG>)
- `backend_cuda.cu:518` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg) -> done: a note; the SG16 instance exists beside the SG32 one (COLI_SG_LAUNCH)
- `backend_cuda.cu:563` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers -> done: checked: the return before the barrier is block-uniform (from get_group and the parameters), the lane-divergent branches hold no group operation (§10.3)
- `backend_cuda.cu:566` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers -> done: checked: the return before the barrier is block-uniform (from get_group and the parameters), the lane-divergent branches hold no group operation (§10.3)
- `backend_cuda.cu:621` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers -> done: checked: the return before the barrier is block-uniform (from get_group and the parameters), the lane-divergent branches hold no group operation (§10.3)
- `backend_cuda.cu:644` DPCT1083 **needs-hand-edit** — local_accessor sized from the launch's dynamic bytes: check the element type of the extern __shared__ -> done: the byte count is the launch's (the same expression the CUDA launch passed); the kernel casts dpct_local to its float4 / float / uint4 as before (§10.3)
- `backend_cuda.cu:653` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:659` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:666` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers -> done: checked: the return before the barrier is block-uniform (from get_group and the parameters), the lane-divergent branches hold no group operation (§10.3)
- `backend_cuda.cu:666` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:680` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:768` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers -> done: checked: the return before the barrier is block-uniform (from get_group and the parameters), the lane-divergent branches hold no group operation (§10.3)
- `backend_cuda.cu:806` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg) -> done: a note; the SG16 instance exists beside the SG32 one (COLI_SG_LAUNCH)
- `backend_cuda.cu:813` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:814` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers -> done: checked: the return before the barrier is block-uniform (from get_group and the parameters), the lane-divergent branches hold no group operation (§10.3)
- `backend_cuda.cu:814` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:823` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:824` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers -> done: checked: the return before the barrier is block-uniform (from get_group and the parameters), the lane-divergent branches hold no group operation (§10.3)
- `backend_cuda.cu:824` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:830` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg) -> done: a note; the SG16 instance exists beside the SG32 one (COLI_SG_LAUNCH)
- `backend_cuda.cu:848` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:849` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers -> done: checked: the return before the barrier is block-uniform (from get_group and the parameters), the lane-divergent branches hold no group operation (§10.3)
- `backend_cuda.cu:849` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:853` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg) -> done: a note; the SG16 instance exists beside the SG32 one (COLI_SG_LAUNCH)
- `backend_cuda.cu:865` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:866` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers -> done: checked: the return before the barrier is block-uniform (from get_group and the parameters), the lane-divergent branches hold no group operation (§10.3)
- `backend_cuda.cu:866` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:876` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg) -> done: a note; the SG16 instance exists beside the SG32 one (COLI_SG_LAUNCH)
- `backend_cuda.cu:883` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:884` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers -> done: checked: the return before the barrier is block-uniform (from get_group and the parameters), the lane-divergent branches hold no group operation (§10.3)
- `backend_cuda.cu:884` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:888` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg) -> done: a note; the SG16 instance exists beside the SG32 one (COLI_SG_LAUNCH)
- `backend_cuda.cu:896` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:897` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers -> done: checked: the return before the barrier is block-uniform (from get_group and the parameters), the lane-divergent branches hold no group operation (§10.3)
- `backend_cuda.cu:897` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:907` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg) -> done: a note; the SG16 instance exists beside the SG32 one (COLI_SG_LAUNCH)
- `backend_cuda.cu:913` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:914` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers -> done: checked: the return before the barrier is block-uniform (from get_group and the parameters), the lane-divergent branches hold no group operation (§10.3)
- `backend_cuda.cu:914` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:923` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg) -> done: a note; the SG16 instance exists beside the SG32 one (COLI_SG_LAUNCH)
- `backend_cuda.cu:936` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:937` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers -> done: checked: the return before the barrier is block-uniform (from get_group and the parameters), the lane-divergent branches hold no group operation (§10.3)
- `backend_cuda.cu:937` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:944` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg) -> done: a note; the SG16 instance exists beside the SG32 one (COLI_SG_LAUNCH)
- `backend_cuda.cu:953` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:954` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers -> done: checked: the return before the barrier is block-uniform (from get_group and the parameters), the lane-divergent branches hold no group operation (§10.3)
- `backend_cuda.cu:954` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:964` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg) -> done: a note; the SG16 instance exists beside the SG32 one (COLI_SG_LAUNCH)
- `backend_cuda.cu:975` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:976` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers -> done: checked: the return before the barrier is block-uniform (from get_group and the parameters), the lane-divergent branches hold no group operation (§10.3)
- `backend_cuda.cu:976` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:983` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg) -> done: a note; the SG16 instance exists beside the SG32 one (COLI_SG_LAUNCH)
- `backend_cuda.cu:990` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:991` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers -> done: checked: the return before the barrier is block-uniform (from get_group and the parameters), the lane-divergent branches hold no group operation (§10.3)
- `backend_cuda.cu:991` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:1005` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg) -> done: a note; the SG16 instance exists beside the SG32 one (COLI_SG_LAUNCH)
- `backend_cuda.cu:1009` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:1040` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg) -> done: a note; the SG16 instance exists beside the SG32 one (COLI_SG_LAUNCH)
- `backend_cuda.cu:1043` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:1067` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg) -> done: a note; the SG16 instance exists beside the SG32 one (COLI_SG_LAUNCH)
- `backend_cuda.cu:1088` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg) -> done: a note; the SG16 instance exists beside the SG32 one (COLI_SG_LAUNCH)
- `backend_cuda.cu:1106` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers -> done: checked: the return before the barrier is block-uniform (from get_group and the parameters), the lane-divergent branches hold no group operation (§10.3)
- `backend_cuda.cu:1110` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers -> done: checked: the return before the barrier is block-uniform (from get_group and the parameters), the lane-divergent branches hold no group operation (§10.3)
- `backend_cuda.cu:1123` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg) -> done: a note; the SG16 instance exists beside the SG32 one (COLI_SG_LAUNCH)
- `backend_cuda.cu:1134` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:1139` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:1141` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:1142` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers -> done: checked: the return before the barrier is block-uniform (from get_group and the parameters), the lane-divergent branches hold no group operation (§10.3)
- `backend_cuda.cu:1142` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:1144` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:1145` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:1146` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers -> done: checked: the return before the barrier is block-uniform (from get_group and the parameters), the lane-divergent branches hold no group operation (§10.3)
- `backend_cuda.cu:1146` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:1148` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:1153` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:1340` DPCT1005 **needs-hand-edit** — compute capability compare: the profile decides (hcg_ok/tcg_ok twins are coli_sycl_xmx_gemm_ok) -> done: compute_major = compute_minor = 0 at init (§10.2)
- `backend_cuda.cu:1347` DPCT1025 **auto-ok** — stream flags dropped: an in-order queue per device -> done: ctx->stream = &coli_sycl_queue(device), the device's in-order compute queue
- `backend_cuda.cu:1370` DPCT1005 **needs-hand-edit** — compute capability compare: the profile decides (hcg_ok/tcg_ok twins are coli_sycl_xmx_gemm_ok) -> done: the device line prints sm_00 and the [SYCL] line the profile (§10.2)
- `backend_cuda.cu:1466` DPCT1106 **auto-ok** — ext::intel device info (icpx has it); coli_sycl_profile carries the same numbers -> done: dev_mem_info(device, ...)
- `backend_cuda.cu:1476` DPCT1006 **needs-hand-edit** — integrated vs discrete: coli_sycl_profile.is_integrated -> done: coli_sycl_profile_for(device).is_integrated
- `backend_cuda.cu:1580` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:1662` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:1702` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:1881` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:1951` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:1988` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:1990` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:2002` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg) -> done: a note; the SG16 instance exists beside the SG32 one (COLI_SG_LAUNCH)
- `backend_cuda.cu:2018` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:2114` DPCT1049 **auto-ok** — work-group size <= 1024 on every target (blocks of 128/256) -> done: threads rounded to the device's sub-group (<= 256)
- `backend_cuda.cu:2119` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:2157` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:2172` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:2202` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:2243` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:2261` DPCT1124 **needs-hand-edit** — memcpy from pageable host memory is asynchronous in SYCL: wait() where the source is a temporary -> done: the input goes through the pinned host_x (no wait needed); the function is rewritten over coli_sycl_xmx_gemm_glu / _gemm (§10.4)
- `backend_cuda.cu:2269` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:2270` DPCT1124 **needs-hand-edit** — memcpy from pageable host memory is asynchronous in SYCL: wait() where the source is a temporary -> done: the download is followed by the queue wait, as before
- `backend_cuda.cu:2352` DPCT1124 **needs-hand-edit** — memcpy from pageable host memory is asynchronous in SYCL: wait() where the source is a temporary -> done: moe_host is pinned (malloc_host): asynchronous as before
- `backend_cuda.cu:2416` DPCT1124 **needs-hand-edit** — memcpy from pageable host memory is asynchronous in SYCL: wait() where the source is a temporary -> done: the descriptors are a stack array: .wait() added on the async branch
- `backend_cuda.cu:2426` DPCT1124 **needs-hand-edit** — memcpy from pageable host memory is asynchronous in SYCL: wait() where the source is a temporary -> done: host_x is pinned: asynchronous
- `backend_cuda.cu:2542` DPCT1124 **needs-hand-edit** — memcpy from pageable host memory is asynchronous in SYCL: wait() where the source is a temporary -> done: the download into pinned host_y, then the queue wait, as before
- `backend_cuda.cu:2544` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:2637` DPCT1124 **needs-hand-edit** — memcpy from pageable host memory is asynchronous in SYCL: wait() where the source is a temporary -> done: the descriptors are a stack array: .wait() added
- `backend_cuda.cu:2640` DPCT1124 **needs-hand-edit** — memcpy from pageable host memory is asynchronous in SYCL: wait() where the source is a temporary -> done: host_x is pinned: asynchronous (the issue returns, take waits)
- `backend_cuda.cu:2704` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:2705` DPCT1124 **needs-hand-edit** — memcpy from pageable host memory is asynchronous in SYCL: wait() where the source is a temporary -> done: the download into pinned host_y: asynchronous, take waits
- `backend_cuda.cu:2756` DPCT1124 **needs-hand-edit** — memcpy from pageable host memory is asynchronous in SYCL: wait() where the source is a temporary -> done: the caller's q outlives the call (the queue wait at the end): asynchronous
- `backend_cuda.cu:2757` DPCT1124 **needs-hand-edit** — memcpy from pageable host memory is asynchronous in SYCL: wait() where the source is a temporary -> done: as 300 (latent)
- `backend_cuda.cu:2758` DPCT1124 **needs-hand-edit** — memcpy from pageable host memory is asynchronous in SYCL: wait() where the source is a temporary -> done: as 300 (rope)
- `backend_cuda.cu:2759` DPCT1083 **needs-hand-edit** — local_accessor sized from the launch's dynamic bytes: check the element type of the extern __shared__ -> done: the byte count is the launch's (the same expression the CUDA launch passed); the kernel casts dpct_local to its float4 / float / uint4 as before (§10.3)
- `backend_cuda.cu:2762` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:2763` DPCT1124 **needs-hand-edit** — memcpy from pageable host memory is asynchronous in SYCL: wait() where the source is a temporary -> done: the download, then the queue wait, as before
- `backend_cuda.cu:2779` DPCT1124 **needs-hand-edit** — memcpy from pageable host memory is asynchronous in SYCL: wait() where the source is a temporary -> done: as 300
- `backend_cuda.cu:2780` DPCT1124 **needs-hand-edit** — memcpy from pageable host memory is asynchronous in SYCL: wait() where the source is a temporary -> done: as 300
- `backend_cuda.cu:2781` DPCT1124 **needs-hand-edit** — memcpy from pageable host memory is asynchronous in SYCL: wait() where the source is a temporary -> done: as 300
- `backend_cuda.cu:2782` DPCT1083 **needs-hand-edit** — local_accessor sized from the launch's dynamic bytes: check the element type of the extern __shared__ -> done: the byte count is the launch's (the same expression the CUDA launch passed); the kernel casts dpct_local to its float4 / float / uint4 as before (§10.3)
- `backend_cuda.cu:2785` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:2791` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:2793` DPCT1124 **needs-hand-edit** — memcpy from pageable host memory is asynchronous in SYCL: wait() where the source is a temporary -> done: the download, then the queue wait, as before
- `backend_cuda.cu:2888` DPCT1124 **needs-hand-edit** — memcpy from pageable host memory is asynchronous in SYCL: wait() where the source is a temporary -> done: host_kv is pinned: asynchronous
- `backend_cuda.cu:2890` DPCT1124 **needs-hand-edit** — memcpy from pageable host memory is asynchronous in SYCL: wait() where the source is a temporary -> done: the caller's q outlives the call: asynchronous
- `backend_cuda.cu:2891` DPCT1124 **needs-hand-edit** — memcpy from pageable host memory is asynchronous in SYCL: wait() where the source is a temporary -> done: the page table dl is freed before the sync: .wait() added
- `backend_cuda.cu:2892` DPCT1124 **needs-hand-edit** — memcpy from pageable host memory is asynchronous in SYCL: wait() where the source is a temporary -> done: dr likewise: .wait() added
- `backend_cuda.cu:2893` DPCT1124 **needs-hand-edit** — memcpy from pageable host memory is asynchronous in SYCL: wait() where the source is a temporary -> done: lengths is the caller's, but copied with .wait() beside the others
- `backend_cuda.cu:2894` DPCT1124 **needs-hand-edit** — memcpy from pageable host memory is asynchronous in SYCL: wait() where the source is a temporary -> done: old is freed before the sync: .wait() added
- `backend_cuda.cu:2895` DPCT1124 **needs-hand-edit** — memcpy from pageable host memory is asynchronous in SYCL: wait() where the source is a temporary -> done: add likewise: .wait() added
- `backend_cuda.cu:2896` DPCT1124 **needs-hand-edit** — memcpy from pageable host memory is asynchronous in SYCL: wait() where the source is a temporary -> done: off likewise: .wait() added
- `backend_cuda.cu:2902` DPCT1083 **needs-hand-edit** — local_accessor sized from the launch's dynamic bytes: check the element type of the extern __shared__ -> done: the byte count is the launch's (the same expression the CUDA launch passed); the kernel casts dpct_local to its float4 / float / uint4 as before (§10.3)
- `backend_cuda.cu:2907` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:2908` DPCT1124 **needs-hand-edit** — memcpy from pageable host memory is asynchronous in SYCL: wait() where the source is a temporary -> done: the download, then the queue wait, as before
- `backend_cuda.cu:3021` DPCT1106 **auto-ok** — ext::intel device info (icpx has it); coli_sycl_profile carries the same numbers -> done: dev_mem_info(cur_device(), ...) before and after the probe; the probe runs only with a device open
- `backend_cuda.cu:3026` DPCT1106 **auto-ok** — ext::intel device info (icpx has it); coli_sycl_profile carries the same numbers -> done: as 321
- `backend_cuda.cu:3093` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg) -> done: a note; the SG16 instance exists beside the SG32 one (COLI_SG_LAUNCH)
- `backend_cuda.cu:4274` DPCT1038 **needs-hand-edit** — a launch inside a local macro (CC_AN, CC_ATTN_G, CC_DS): expand the macro by hand -> done: bd_tensor_gemm is deleted: a GGUF's prompt rows stay on the GEMV kernels (§10.4)
- `backend_cuda.cu:4276` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:4280` DPCT1038 **needs-hand-edit** — a launch inside a local macro (CC_AN, CC_ATTN_G, CC_DS): expand the macro by hand -> done: as 67
- `backend_cuda.cu:4282` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:4286` DPCT1027 **needs-hand-edit** — cudaHostRegister/Unregister -> 0: coli_cuda_host_register returns 0 (host pool off) -> done: returns 0 with the note
- `backend_cuda.cu:4286` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: gone with the registration
- `backend_cuda.cu:4291` DPCT1027 **needs-hand-edit** — cudaHostRegister/Unregister -> 0: coli_cuda_host_register returns 0 (host pool off) -> done: a no-op
- `backend_cuda.cu:4291` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: gone with the registration
- `backend_cuda.cu:4353` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg) -> done: a note; the SG16 instance exists beside the SG32 one (COLI_SG_LAUNCH)
- `backend_cuda.cu:4356` DPCT1083 **needs-hand-edit** — local_accessor sized from the launch's dynamic bytes: check the element type of the extern __shared__ -> done: the byte count is the launch's (the same expression the CUDA launch passed); the kernel casts dpct_local to its float4 / float / uint4 as before (§10.3)
- `backend_cuda.cu:4408` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg) -> done: a note; the SG16 instance exists beside the SG32 one (COLI_SG_LAUNCH)
- `backend_cuda.cu:4509` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg) -> done: a note; the SG16 instance exists beside the SG32 one (COLI_SG_LAUNCH)
- `backend_cuda.cu:4527` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers -> done: checked: the return before the barrier is block-uniform (from get_group and the parameters), the lane-divergent branches hold no group operation (§10.3)
- `backend_cuda.cu:4538` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers -> done: checked: the return before the barrier is block-uniform (from get_group and the parameters), the lane-divergent branches hold no group operation (§10.3)
- `backend_cuda.cu:4571` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers -> done: checked: the return before the barrier is block-uniform (from get_group and the parameters), the lane-divergent branches hold no group operation (§10.3)
- `backend_cuda.cu:4723` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:4745` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:4767` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:4794` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers -> done: checked: the return before the barrier is block-uniform (from get_group and the parameters), the lane-divergent branches hold no group operation (§10.3)
- `backend_cuda.cu:4824` DPCT1007 **needs-hand-edit** — gemv_i8_w4's split-K partials (DECODE): coli_sycl_scratch(dev, bytes, 0) -> done: coli_sycl_scratch(t->device, kz * S * O * sizeof(float), 0) (§10.2)
- `backend_cuda.cu:4830` DPCT1007 **needs-hand-edit** — gemv_i8_w4's split-K partials (DECODE): coli_sycl_scratch(dev, bytes, 0) -> done: nothing freed: the arena keeps its block
- `backend_cuda.cu:4832` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:4834` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:4846` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:4860` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:4870` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:4876` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:4880` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:4888` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:4989` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:5005` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg) -> done: a note; the SG16 instance exists beside the SG32 one (COLI_SG_LAUNCH)
- `backend_cuda.cu:5017` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:5031` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg) -> done: a note; the SG16 instance exists beside the SG32 one (COLI_SG_LAUNCH)
- `backend_cuda.cu:5122` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:5133` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:5159` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:5170` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:5184` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:5196` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:5209` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg) -> done: a note; the SG16 instance exists beside the SG32 one (COLI_SG_LAUNCH)
- `backend_cuda.cu:5215` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:5222` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg) -> done: a note; the SG16 instance exists beside the SG32 one (COLI_SG_LAUNCH)
- `backend_cuda.cu:5228` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:5334` DPCT1038 **needs-hand-edit** — a launch inside a local macro (CC_AN, CC_ATTN_G, CC_DS): expand the macro by hand -> done: moe_rows_bd_launch is deleted: a GGUF's prompt rows take moe_bd_launch (§10.4)
- `backend_cuda.cu:5335` DPCT1038 **needs-hand-edit** — a launch inside a local macro (CC_AN, CC_ATTN_G, CC_DS): expand the macro by hand -> done: as 80
- `backend_cuda.cu:5336` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:5354` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:5396` DPCT1108 **auto-ok** — masked sub-group helper (experimental): keep, or drop the full mask for the plain dpct helper -> done: mg_shfl_xor is shfl_xor(v, off) (mg_lanes_sum; lpr <= SG)
- `backend_cuda.cu:5405` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg) -> done: a note; the SG16 instance exists beside the SG32 one (COLI_SG_LAUNCH)
- `backend_cuda.cu:5421` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:5449` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg) -> done: a note; the SG16 instance exists beside the SG32 one (COLI_SG_LAUNCH)
- `backend_cuda.cu:5463` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:5494` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:5513` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg) -> done: a note; the SG16 instance exists beside the SG32 one (COLI_SG_LAUNCH)
- `backend_cuda.cu:5536` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:5542` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:5554` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:5573` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg) -> done: a note; the SG16 instance exists beside the SG32 one (COLI_SG_LAUNCH)
- `backend_cuda.cu:5598` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:5615` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg) -> done: a note; the SG16 instance exists beside the SG32 one (COLI_SG_LAUNCH)
- `backend_cuda.cu:5635` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:5711` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:5722` DPCT1083 **needs-hand-edit** — local_accessor sized from the launch's dynamic bytes: check the element type of the extern __shared__ -> done: the byte count is the launch's (the same expression the CUDA launch passed); the kernel casts dpct_local to its float4 / float / uint4 as before (§10.3)
- `backend_cuda.cu:5726` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:5760` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:5796` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:5836` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers -> done: checked: the return before the barrier is block-uniform (from get_group and the parameters), the lane-divergent branches hold no group operation (§10.3)
- `backend_cuda.cu:5850` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `backend_cuda.cu:5906` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:5915` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:5924` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:5932` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:5956` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers -> done: checked: the return before the barrier is block-uniform (from get_group and the parameters), the lane-divergent branches hold no group operation (§10.3)
- `backend_cuda.cu:6002` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:6079` DPCT1124 **needs-hand-edit** — memcpy from pageable host memory is asynchronous in SYCL: wait() where the source is a temporary -> done: the descriptors are a stack array: .wait() added
- `backend_cuda.cu:6081` DPCT1124 **needs-hand-edit** — memcpy from pageable host memory is asynchronous in SYCL: wait() where the source is a temporary -> done: the weights are the caller's and the issue returns at once: .wait() added
- `backend_cuda.cu:6086` DPCT1124 **needs-hand-edit** — cudaMemcpyPeerAsync (GLM pipe, not qwen36): dpct::async_dpct_memcpy, peer access enabled first -> done: peer_copy: a memcpy on the queue where the two devices share a SYCL context, else refused (an error) (§10.2)
- `backend_cuda.cu:6095` DPCT1124 **needs-hand-edit** — cudaMemcpyPeerAsync (GLM pipe, not qwen36): dpct::async_dpct_memcpy, peer access enabled first -> done: peer_copy, as 374
- `backend_cuda.cu:6098` DPCT1024 **needs-hand-edit** — error code consumed by the program logic, replaced with 0: same protocol as DPCT1010 -> done: sync_barrier records the event (ext_oneapi_submit_barrier) inside COLI_SYCL_TRY
- `backend_cuda.cu:6099` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:6111` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:6132` DPCT1083 **needs-hand-edit** — local_accessor sized from the launch's dynamic bytes: check the element type of the extern __shared__ -> done: the byte count is the launch's (the same expression the CUDA launch passed); the kernel casts dpct_local to its float4 / float / uint4 as before (§10.3)
- `backend_cuda.cu:6135` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:6140` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:6141` DPCT1124 **needs-hand-edit** — memcpy from pageable host memory is asynchronous in SYCL: wait() where the source is a temporary -> done: the download, then the queue wait, as before
- `backend_cuda.cu:6150` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:6156` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:6163` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:6176` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:6197` DPCT1083 **needs-hand-edit** — local_accessor sized from the launch's dynamic bytes: check the element type of the extern __shared__ -> done: the byte count is the launch's (the same expression the CUDA launch passed); the kernel casts dpct_local to its float4 / float / uint4 as before (§10.3)
- `backend_cuda.cu:6200` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:6203` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:6216` DPCT1083 **needs-hand-edit** — local_accessor sized from the launch's dynamic bytes: check the element type of the extern __shared__ -> done: the byte count is the launch's (the same expression the CUDA launch passed); the kernel casts dpct_local to its float4 / float / uint4 as before (§10.3)
- `backend_cuda.cu:6219` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:6233` DPCT1124 **needs-hand-edit** — memcpy from pageable host memory is asynchronous in SYCL: wait() where the source is a temporary -> done: the caller's q outlives the call: asynchronous
- `backend_cuda.cu:6234` DPCT1083 **needs-hand-edit** — local_accessor sized from the launch's dynamic bytes: check the element type of the extern __shared__ -> done: the byte count is the launch's (the same expression the CUDA launch passed); the kernel casts dpct_local to its float4 / float / uint4 as before (§10.3)
- `backend_cuda.cu:6237` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the submit sits in the entry point's function-try-block (sycl_caught, backend_sycl.h); cuda_ok(0, "... launch") stays as the check that always passes (§10.2)
- `backend_cuda.cu:6238` DPCT1124 **needs-hand-edit** — memcpy from pageable host memory is asynchronous in SYCL: wait() where the source is a temporary -> done: the download, then the queue wait, as before

#### cuda_chain.cu (197 sites: auto-ok 78, needs-hand-edit 119)

- `cuda_chain.cu:86` DPCT1093 **needs-hand-edit** — dpct::select_device(ordinal): the ordinal is the engine's, use coli_sycl_queue(device) (GPU list, PCI order)
- `cuda_chain.cu:86` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit
- `cuda_chain.cu:91` DPCT1009 **needs-hand-edit** — cudaGetErrorString -> exception text
- `cuda_chain.cu:92` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit
- `cuda_chain.cu:96` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit
- `cuda_chain.cu:103` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg)
- `cuda_chain.cu:136` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:157` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg)
- `cuda_chain.cu:183` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:183` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:186` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:186` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:189` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:189` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:193` DPCT1013 **auto-ok** — rounding mode: float -> half is round-to-nearest in sycl::half too; keep an eye on the f16 KV mirror parity test. **Review fix 1 (§12):** dpct dropped the parentheses of `__fmul_rn`'s second operand here (`__fmul_rn(__fmul_rn(__fmul_rn(v, 1.f + wp), r), 1.f)` came out as `v * 1.f + wp * r * 1.f`, i.e. v + w * r); now `((v * (1.f + wp)) * r) * 1.f`. Re-check this site after any re-migration
- `cuda_chain.cu:200` DPCT1013 **auto-ok** — rounding mode: float -> half is round-to-nearest in sycl::half too; keep an eye on the f16 KV mirror parity test
- `cuda_chain.cu:203` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:206` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:206` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:212` DPCT1013 **auto-ok** — rounding mode: float -> half is round-to-nearest in sycl::half too; keep an eye on the f16 KV mirror parity test. **Review fix 1 (§12):** the same lost parentheses as `:193` (`wn`), fixed the same way; the only two sites of the four migrated sources where a `__*_rn` had a compound second operand
- `cuda_chain.cu:226` DPCT1013 **auto-ok** — rounding mode: float -> half is round-to-nearest in sycl::half too; keep an eye on the f16 KV mirror parity test
- `cuda_chain.cu:236` DPCT1013 **auto-ok** — rounding mode: float -> half is round-to-nearest in sycl::half too; keep an eye on the f16 KV mirror parity test
- `cuda_chain.cu:237` DPCT1013 **auto-ok** — rounding mode: float -> half is round-to-nearest in sycl::half too; keep an eye on the f16 KV mirror parity test
- `cuda_chain.cu:289` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg)
- `cuda_chain.cu:326` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:329` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:332` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:336` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:339` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:353` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:377` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg)
- `cuda_chain.cu:415` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:443` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:452` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:460` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:466` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:471` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:474` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:480` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:497` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:1027` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg)
- `cuda_chain.cu:1053` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:1053` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:1056` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:1056` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:1060` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:1060` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:1062` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:1062` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:1079` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:1079` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:1082` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:1082` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:1094` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:1094` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:1104` DPCT1108 **auto-ok** — masked sub-group helper (experimental): keep, or drop the full mask for the plain dpct helper
- `cuda_chain.cu:1104` DPCT1064 **needs-hand-edit** — the shuffle macro's width is the kernel's SG, not 32 (COLI_SYCL_SHFL_WIDTH)
- `cuda_chain.cu:1114` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg)
- `cuda_chain.cu:1152` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:1152` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:1169` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:1169` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:1183` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:1183` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:1229` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg)
- `cuda_chain.cu:1282` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:1324` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:1324` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:1358` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:1361` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:1361` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:1363` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:1363` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:1367` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:1395` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg)
- `cuda_chain.cu:1414` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:1425` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:1445` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg)
- `cuda_chain.cu:1461` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:1464` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:1467` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:1469` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:1508` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:1508` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:1513` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:1513` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:1519` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:1519` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:1521` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:1521` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:1524` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:1530` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:1532` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:1535` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:1535` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:1537` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:1537` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:1545` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:1548` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:1548` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:1551` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg)
- `cuda_chain.cu:1565` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:1567` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:1567` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:1572` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:1577` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:1577` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:1585` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:1667` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg)
- `cuda_chain.cu:1683` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:1686` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:1691` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:1694` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:1716` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:1724` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:1726` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:1742` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:1770` DPCT1078 **needs-hand-edit** — __threadfence_system: seq_cst, system scope (cc_signal_k/cc_wait_k spin on host memory)
- `cuda_chain.cu:1771` DPCT1078 **needs-hand-edit** — __threadfence_system: seq_cst, system scope (cc_signal_k/cc_wait_k spin on host memory)
- `cuda_chain.cu:1815` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg)
- `cuda_chain.cu:1818` DPCT1083 **needs-hand-edit** — local_accessor sized from the launch's dynamic bytes: check the element type of the extern __shared__
- `cuda_chain.cu:1874` DPCT1093 **needs-hand-edit** — dpct::select_device(ordinal): the ordinal is the engine's, use coli_sycl_queue(device) (GPU list, PCI order)
- `cuda_chain.cu:1874` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit
- `cuda_chain.cu:1875` DPCT1025 **auto-ok** — stream flags dropped: an in-order queue per device
- `cuda_chain.cu:1896` DPCT1093 **needs-hand-edit** — dpct::select_device(ordinal): the ordinal is the engine's, use coli_sycl_queue(device) (GPU list, PCI order)
- `cuda_chain.cu:1910` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit
- `cuda_chain.cu:1935` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit
- `cuda_chain.cu:1942` DPCT1048 **auto-ok** — cudaHostAlloc flags dropped: malloc_host is mapped and portable by nature
- `cuda_chain.cu:1942` DPCT1048 **auto-ok** — cudaHostAlloc flags dropped: malloc_host is mapped and portable by nature
- `cuda_chain.cu:1945` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit
- `cuda_chain.cu:1958` DPCT1093 **needs-hand-edit** — dpct::select_device(ordinal): the ordinal is the engine's, use coli_sycl_queue(device) (GPU list, PCI order)
- `cuda_chain.cu:1964` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit
- `cuda_chain.cu:1990` DPCT1024 **needs-hand-edit** — error code consumed by the program logic, replaced with 0: same protocol as DPCT1010
- `cuda_chain.cu:2058` DPCT1124 **needs-hand-edit** — memcpy from pageable host memory is asynchronous in SYCL: wait() where the source is a temporary
- `cuda_chain.cu:2149` DPCT1124 **needs-hand-edit** — memcpy from pageable host memory is asynchronous in SYCL: wait() where the source is a temporary
- `cuda_chain.cu:2167` DPCT1124 **needs-hand-edit** — memcpy from pageable host memory is asynchronous in SYCL: wait() where the source is a temporary
- `cuda_chain.cu:2254` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg)
- `cuda_chain.cu:2281` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:2281` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:2284` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:2284` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:2292` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:2292` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:2297` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:2297` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:2300` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:2300` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:2305` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:2305` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:2331` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:2337` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:2337` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:2339` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:2339` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:2352` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:2370` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg)
- `cuda_chain.cu:2393` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:2429` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:2429` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:2470` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:2473` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:2485` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:2519` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:2519` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:2556` DPCT1083 **needs-hand-edit** — local_accessor sized from the launch's dynamic bytes: check the element type of the extern __shared__
- `cuda_chain.cu:2642` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit
- `cuda_chain.cu:2714` DPCT1007 **needs-hand-edit** — the dnrec split's q/k rows (prompt rows): coli_sycl_scratch(dev, bytes, 1)
- `cuda_chain.cu:2714` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit
- `cuda_chain.cu:2727` DPCT1007 **needs-hand-edit** — the dnrec split's q/k rows (prompt rows): coli_sycl_scratch(dev, bytes, 1)
- `cuda_chain.cu:2885` DPCT1025 **auto-ok** — stream flags dropped: an in-order queue per device
- `cuda_chain.cu:2904` DPCT1124 **needs-hand-edit** — memcpy from pageable host memory is asynchronous in SYCL: wait() where the source is a temporary
- `cuda_chain.cu:2908` DPCT1024 **needs-hand-edit** — error code consumed by the program logic, replaced with 0: same protocol as DPCT1010
- `cuda_chain.cu:2919` DPCT1024 **needs-hand-edit** — error code consumed by the program logic, replaced with 0: same protocol as DPCT1010
- `cuda_chain.cu:2942` DPCT1024 **needs-hand-edit** — error code consumed by the program logic, replaced with 0: same protocol as DPCT1010
- `cuda_chain.cu:2944` DPCT1124 **needs-hand-edit** — memcpy from pageable host memory is asynchronous in SYCL: wait() where the source is a temporary
- `cuda_chain.cu:2945` DPCT1024 **needs-hand-edit** — error code consumed by the program logic, replaced with 0: same protocol as DPCT1010
- `cuda_chain.cu:2967` DPCT1024 **needs-hand-edit** — error code consumed by the program logic, replaced with 0: same protocol as DPCT1010
- `cuda_chain.cu:2987` DPCT1124 **needs-hand-edit** — memcpy from pageable host memory is asynchronous in SYCL: wait() where the source is a temporary
- `cuda_chain.cu:2988` DPCT1024 **needs-hand-edit** — error code consumed by the program logic, replaced with 0: same protocol as DPCT1010
- `cuda_chain.cu:2998` DPCT1001 **needs-hand-edit** — error statement kept: same protocol as DPCT1010
- `cuda_chain.cu:2998` DPCT1002 **needs-hand-edit** — error if-statement: same protocol as DPCT1010
- `cuda_chain.cu:2998` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit
- `cuda_chain.cu:3060` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit
- `cuda_chain.cu:3113` DPCT1083 **needs-hand-edit** — local_accessor sized from the launch's dynamic bytes: check the element type of the extern __shared__
- `cuda_chain.cu:3181` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:3184` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:3184` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:3187` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning
- `cuda_chain.cu:3191` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg)
- `cuda_chain.cu:3273` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg)
- `cuda_chain.cu:3312` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:3328` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:3345` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:3350` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:3361` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:3392` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers
- `cuda_chain.cu:3397` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers

#### cu_xb.cu (42 sites: auto-ok 17, needs-hand-edit 25; every site resolved, "-> done" says how: §10)

- `cu_xb.cu:100` DPCT1093 **needs-hand-edit** — dpct::select_device(ordinal): the ordinal is the engine's, use coli_sycl_queue(device) (GPU list, PCI order) -> done: cu_set is C.ready
- `cu_xb.cu:107` DPCT1093 **needs-hand-edit** — dpct::select_device(ordinal): the ordinal is the engine's, use coli_sycl_queue(device) (GPU list, PCI order) -> done: cu_open reads coli_sycl_profile_for(C.dev), the device coli_cuda_init opened
- `cu_xb.cu:109` DPCT1006 **needs-hand-edit** — integrated vs discrete: coli_sycl_profile.is_integrated -> done: the profile's is_integrated
- `cu_xb.cu:110` DPCT1051 **needs-hand-edit** — canMapHostMemory -> false: USM host memory is always mapped, make it true -> done: C.mapped 0 by decision: the host pool is off (§10.2)
- `cu_xb.cu:110` DPCT1006 **needs-hand-edit** — integrated vs discrete: coli_sycl_profile.is_integrated -> done: as 487
- `cu_xb.cu:112` DPCT1019 **auto-ok** — sharedMemPerBlockOptin -> local_mem_size (128 KiB pvc, 64 KiB xe2) -> done: the profile's slm_bytes
- `cu_xb.cu:113` DPCT1025 **auto-ok** — stream flags dropped: an in-order queue per device -> done: C.xs: an in-order profiling queue of this unit's own on the device's context
- `cu_xb.cu:114` DPCT1025 **auto-ok** — stream flags dropped: an in-order queue per device -> done: C.cs likewise
- `cu_xb.cu:134` DPCT1106 **auto-ok** — ext::intel device info (icpx has it); coli_sycl_profile carries the same numbers -> done: cu_mem_info
- `cu_xb.cu:144` DPCT1106 **auto-ok** — ext::intel device info (icpx has it); coli_sycl_profile carries the same numbers -> done: cu_mem_info
- `cu_xb.cu:166` DPCT1048 **auto-ok** — cudaHostAlloc flags dropped: malloc_host is mapped and portable by nature -> done: sycl::malloc_host on the device's queue (USM host memory: mapped and portable by nature)
- `cu_xb.cu:166` DPCT1048 **auto-ok** — cudaHostAlloc flags dropped: malloc_host is mapped and portable by nature -> done: sycl::malloc_host on the device's queue (USM host memory: mapped and portable by nature)
- `cu_xb.cu:168` DPCT1026 **auto-ok** — cudaGetLastError removed (redundant) -> done: nothing to consume
- `cu_xb.cu:303` DPCT1124 **needs-hand-edit** — memcpy from pageable host memory is asynchronous in SYCL: wait() where the source is a temporary -> done: the image stays until the copy queue's wait below: asynchronous
- `cu_xb.cu:304` DPCT1124 **needs-hand-edit** — memcpy from pageable host memory is asynchronous in SYCL: wait() where the source is a temporary -> done: as 496
- `cu_xb.cu:308` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the "if (!ok) 0;" leftover removed
- `cu_xb.cu:336` DPCT1124 **needs-hand-edit** — memcpy from pageable host memory is asynchronous in SYCL: wait() where the source is a temporary -> done: device to device on the copy queue, waited below
- `cu_xb.cu:337` DPCT1124 **needs-hand-edit** — memcpy from pageable host memory is asynchronous in SYCL: wait() where the source is a temporary -> done: as 499
- `cu_xb.cu:340` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: as 498
- `cu_xb.cu:348` DPCT1093 **needs-hand-edit** — dpct::select_device(ordinal): the ordinal is the engine's, use coli_sycl_queue(device) (GPU list, PCI order) -> done: no selection (tensor_release frees on the queue's context)
- `cu_xb.cu:395` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg) -> done: a note; the SG16 instance exists beside the SG32 one (COLI_SG_LAUNCH)
- `cu_xb.cu:422` DPCT1098 **auto-ok** — __ldg -> plain load -> done: the plain load
- `cu_xb.cu:467` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: "return 1" (a failed submit throws)
- `cu_xb.cu:527` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `cu_xb.cu:530` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers -> done: checked: the return before the barrier is block-uniform (from get_group and the parameters), the lane-divergent branches hold no group operation (§10.3)
- `cu_xb.cu:530` DPCT1065 **auto-ok** — barrier(): local_space fence where no global memory is touched is a later tuning -> done: the barrier kept as emitted (the local_space tuning not taken)
- `cu_xb.cu:545` DPCT1093 **needs-hand-edit** — dpct::select_device(ordinal): the ordinal is the engine's, use coli_sycl_queue(device) (GPU list, PCI order) -> done: no selection
- `cu_xb.cu:590` DPCT1048 **auto-ok** — cudaHostAlloc flags dropped: malloc_host is mapped and portable by nature -> done: sycl::malloc_host on the device's queue (USM host memory: mapped and portable by nature)
- `cu_xb.cu:591` DPCT1048 **auto-ok** — cudaHostAlloc flags dropped: malloc_host is mapped and portable by nature -> done: sycl::malloc_host on the device's queue (USM host memory: mapped and portable by nature)
- `cu_xb.cu:592` DPCT1048 **auto-ok** — cudaHostAlloc flags dropped: malloc_host is mapped and portable by nature -> done: sycl::malloc_host on the device's queue (USM host memory: mapped and portable by nature)
- `cu_xb.cu:593` DPCT1026 **auto-ok** — cudaGetLastError removed (redundant) -> done: nothing to consume
- `cu_xb.cu:611` DPCT1024 **needs-hand-edit** — error code consumed by the program logic, replaced with 0: same protocol as DPCT1010 -> done: sync_barrier(C.e0, st) records the batch's start
- `cu_xb.cu:612` DPCT1124 **needs-hand-edit** — memcpy from pageable host memory is asynchronous in SYCL: wait() where the source is a temporary -> done: hX is pinned: asynchronous
- `cu_xb.cu:613` DPCT1124 **needs-hand-edit** — memcpy from pageable host memory is asynchronous in SYCL: wait() where the source is a temporary -> done: hW is pinned: asynchronous
- `cu_xb.cu:624` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: as 504
- `cu_xb.cu:626` DPCT1124 **needs-hand-edit** — memcpy from pageable host memory is asynchronous in SYCL: wait() where the source is a temporary -> done: the download into pinned hY: asynchronous, join waits
- `cu_xb.cu:627` DPCT1024 **needs-hand-edit** — error code consumed by the program logic, replaced with 0: same protocol as DPCT1010 -> done: sync_barrier(C.e1, st) records its end
- `cu_xb.cu:630` DPCT1026 **auto-ok** — cudaGetLastError removed (redundant) -> done: nothing to consume
- `cu_xb.cu:648` DPCT1093 **needs-hand-edit** — dpct::select_device(ordinal): the ordinal is the engine's, use coli_sycl_queue(device) (GPU list, PCI order) -> done: no selection
- `cu_xb.cu:652` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: the "else 0;" leftover removed
- `cu_xb.cu:659` DPCT1001 **needs-hand-edit** — error statement kept: same protocol as DPCT1010 -> done: the profiling read is in a try of its own (0 ms without profiling)
- `cu_xb.cu:659` DPCT1000 **needs-hand-edit** — error if-statement: same protocol as DPCT1010 -> done: as 517

#### cuda_bd.cuh (46 sites: auto-ok 39, delete-dead-code 1, needs-hand-edit 6; every site resolved, "-> done" says how: §10)

- `cuda_bd.cuh:30` DPCT1098 **auto-ok** — __ldg -> plain load -> done: the plain load
- `cuda_bd.cuh:31` DPCT1098 **auto-ok** — __ldg -> plain load -> done: the plain load
- `cuda_bd.cuh:71` DPCT1098 **auto-ok** — __ldg -> plain load -> done: the plain load
- `cuda_bd.cuh:72` DPCT1098 **auto-ok** — __ldg -> plain load -> done: the plain load
- `cuda_bd.cuh:73` DPCT1098 **auto-ok** — __ldg -> plain load -> done: the plain load
- `cuda_bd.cuh:92` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg) -> done: a note; the SG16 instance exists beside the SG32 one (COLI_SG_LAUNCH)
- `cuda_bd.cuh:104` DPCT1013 **auto-ok** — rounding mode: float -> half is round-to-nearest in sycl::half too; keep an eye on the f16 KV mirror parity test -> done: sycl's float -> half conversion rounds to nearest even as __float2half_rn did; the GGUF parity test on a device is the check
- `cuda_bd.cuh:112` DPCT1013 **auto-ok** — rounding mode: float -> half is round-to-nearest in sycl::half too; keep an eye on the f16 KV mirror parity test -> done: sycl's float -> half conversion rounds to nearest even as __float2half_rn did; the GGUF parity test on a device is the check
- `cuda_bd.cuh:179` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg) -> done: a note; the SG16 instance exists beside the SG32 one (COLI_SG_LAUNCH)
- `cuda_bd.cuh:337` DPCT1098 **auto-ok** — __ldg -> plain load -> done: the plain load
- `cuda_bd.cuh:451` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg) -> done: a note; the SG16 instance exists beside the SG32 one (COLI_SG_LAUNCH)
- `cuda_bd.cuh:455` DPCT1098 **auto-ok** — __ldg -> plain load -> done: the plain load
- `cuda_bd.cuh:464` DPCT1098 **auto-ok** — __ldg -> plain load -> done: the plain load
- `cuda_bd.cuh:490` DPCT1098 **auto-ok** — __ldg -> plain load -> done: the plain load
- `cuda_bd.cuh:511` DPCT1098 **auto-ok** — __ldg -> plain load -> done: the plain load
- `cuda_bd.cuh:565` DPCT1098 **auto-ok** — __ldg -> plain load -> done: the plain load
- `cuda_bd.cuh:578` DPCT1098 **auto-ok** — __ldg -> plain load -> done: the plain load
- `cuda_bd.cuh:590` DPCT1098 **auto-ok** — __ldg -> plain load -> done: the plain load
- `cuda_bd.cuh:591` DPCT1098 **auto-ok** — __ldg -> plain load -> done: the plain load
- `cuda_bd.cuh:605` DPCT1098 **auto-ok** — __ldg -> plain load -> done: the plain load
- `cuda_bd.cuh:644` DPCT1098 **auto-ok** — __ldg -> plain load -> done: the plain load
- `cuda_bd.cuh:699` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg) -> done: a note; the SG16 instance exists beside the SG32 one (COLI_SG_LAUNCH)
- `cuda_bd.cuh:748` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg) -> done: a note; the SG16 instance exists beside the SG32 one (COLI_SG_LAUNCH)
- `cuda_bd.cuh:809` DPCT1098 **auto-ok** — __ldg -> plain load -> done: the plain load
- `cuda_bd.cuh:820` DPCT1098 **auto-ok** — __ldg -> plain load -> done: the plain load
- `cuda_bd.cuh:821` DPCT1098 **auto-ok** — __ldg -> plain load -> done: the plain load
- `cuda_bd.cuh:830` DPCT1098 **auto-ok** — __ldg -> plain load -> done: the plain load
- `cuda_bd.cuh:831` DPCT1098 **auto-ok** — __ldg -> plain load -> done: the plain load
- `cuda_bd.cuh:848` DPCT1098 **auto-ok** — __ldg -> plain load -> done: the plain load
- `cuda_bd.cuh:849` DPCT1098 **auto-ok** — __ldg -> plain load -> done: the plain load
- `cuda_bd.cuh:852` DPCT1098 **auto-ok** — __ldg -> plain load -> done: the plain load
- `cuda_bd.cuh:868` DPCT1098 **auto-ok** — __ldg -> plain load -> done: the plain load
- `cuda_bd.cuh:904` DPCT1108 **auto-ok** — masked sub-group helper (experimental): keep, or drop the full mask for the plain dpct helper -> done: as DPCT1064:20
- `cuda_bd.cuh:904` DPCT1064 **needs-hand-edit** — the shuffle macro's width is the kernel's SG, not 32 (COLI_SYCL_SHFL_WIDTH) -> done: bd_shfl_xor is shfl_xor(v, off) over the SG sub-group (every GEMV kernel a template over SG)
- `cuda_bd.cuh:926` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg) -> done: a note; the SG16 instance exists beside the SG32 one (COLI_SG_LAUNCH)
- `cuda_bd.cuh:945` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers -> done: checked: the return before the barrier is block-uniform (from get_group and the parameters), the lane-divergent branches hold no group operation (§10.3)
- `cuda_bd.cuh:951` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers -> done: checked: the return before the barrier is block-uniform (from get_group and the parameters), the lane-divergent branches hold no group operation (§10.3)
- `cuda_bd.cuh:957` DPCT1118 **needs-hand-edit** — group op in possibly divergent flow: check the kernel's early returns before its barriers -> done: checked: the return before the barrier is block-uniform (from get_group and the parameters), the lane-divergent branches hold no group operation (§10.3)
- `cuda_bd.cuh:989` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg) -> done: a note; the SG16 instance exists beside the SG32 one (COLI_SG_LAUNCH)
- `cuda_bd.cuh:1021` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg) -> done: a note; the SG16 instance exists beside the SG32 one (COLI_SG_LAUNCH)
- `cuda_bd.cuh:1093` DPCT1110 **auto-ok** — register pressure note; the SG16 instance halves it (coli_sycl_launch_sg) -> done: a note; the SG16 instance exists beside the SG32 one (COLI_SG_LAUNCH)
- `cuda_bd.cuh:1103` DPCT1098 **auto-ok** — __ldg -> plain load -> done: the plain load
- `cuda_bd.cuh:1110` DPCT1098 **auto-ok** — __ldg -> plain load -> done: the plain load
- `cuda_bd.cuh:1141` DPCT1026 **delete-dead-code** — cudaFuncSetAttribute removed: no opt-in needed (bd_smem_raise), delete the raise bookkeeping -> done: bd_smem_raise and the kernel-pointer wrappers are deleted; the launchers refuse a local memory past the profile's slm_bytes (§10.3)
- `cuda_bd.cuh:1152` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: bd_xq_scratch keys its buffers by queue (no device id)
- `cuda_bd.cuh:1161` DPCT1010 **needs-hand-edit** — cudaGetLastError -> 0: the cuda_ok/cc_ok protocol becomes try/catch of sycl::exception at the submit -> done: as 228

#### tests/test_cuda_chain.cu (2 sites: auto-ok 2)

- `tests/test_cuda_chain.cu:404` DPCT1048 **auto-ok** — cudaHostAlloc flags dropped: malloc_host is mapped and portable by nature
- `tests/test_cuda_chain.cu:405` DPCT1048 **auto-ok** — cudaHostAlloc flags dropped: malloc_host is mapped and portable by nature


## 14. Performance pass

### 14.1 The frame profile (perf m-frame, 2026-11-07): one decode token at the op level

Where a decode token's 14 ms go was measured from inside the engine, with no external profiler
(<ops>/CRASH-2026-11-07.md: no VTune, no Level Zero metrics): `COLI_SYCL_FRAME_PROF=1` (coli_sycl.h
section 5, the sink in coli_sycl_profile.cpp) makes the queues with `enable_profiling` and books
every command the chain and the backend submit on the compute queue from its event's
`command_start` / `command_end`, read after the frame has been waited for (nothing waits inside a
frame). The hooks are the chain's `CC_K` / `CC_KSG` and its memcpy / memset / barrier submissions,
the backend's `COLI_SG_LAUNCH` and the plain launches on the chain's decode path (`gemv_i8_sk_sum`,
`f32_gemv_k`, `f32_few_k`, `f32_gemm_k`, `quant_matmul`, `moe_sum_k`, `moe_rows_sum_k`), all through
`coli_sycl_fprof_ev`, one flag test when off; `launched()` names the commands just submitted by the
op's kind, `cc_matmul` by the matrix's shape and format, `cc_moe` by its parts. A frame runs from the
first `cc_begin` after a wait to the wait that covers it; frames are bucketed by their command
count (a decode token's frames all have the same), and the modal bucket's table is printed at
`cc_shutdown` as `[SYCL] frame profile`; `=2` dumps one frame command by command. The prompt's
matrix-engine GEMMs (coli_sycl_xmx.cpp) are not hooked, so a prompt frame's books show their time
as idle; a decode frame's are complete. The profiling queues cost nothing measurable on the wall
(run 3 below); reading ~850 events a frame costs the host ~2.5 ms after each wait, so the step time
under the env is not the production step time, the device times are.

Runs (all under <ops>/guard.sh, card1, the mandated form, the engine of HANDOVER 20:15Z with the chain
and the routed experts in the frame, prompt `decode-prompt.txt` = 21 tokens; the GPU gateway `gpu-gw`
was up and idle beside them, holding 21 GB and submitting nothing; m-submit's micro-benchmarks ran on
the same card at times, noted per run):

| run | env | step ms/token | tok/s | chain: waiting ms/token | log |
|---|---|---|---|---|---|
| baseline (rung 5 run 2, no gateway) | - | 14.3 | 58.3 | 9.46 | gpu-rung5-qwen36-card1-n64-forced-20261107T200706Z |
| 1 (hooks built in, off) | - | 14.6 | 57.7 | 8.49 | perf-m-frame-run1-base-20261107T203458Z (m-submit run2 overlapped the model load only) |
| 2 | `COLI_SYCL_FRAME_PROF=2` | 16.9 | 51.4 | 13.67 (incl. the event reads) | perf-m-frame-run2-fprof-20261107T203626Z (the card to itself) |
| 3 | `COLI_SYCL_PROFILING=1` (profiling queues, hooks off) | 14.5 | 57.7 | 8.32 | perf-m-frame-run3-profq-20261107T203947Z (the card to itself) |
| 4 | `COLI_SYCL_FRAME_PROF=1`, N_NEW=256 | 107.5 | 9.9 | - | DISCARDED: m-submit run4-events shared the card for the whole run (72-79 ms of gaps a frame) |
| 5 | the same, after waiting for a free card | n/a | n/a | - | n/a (not finished when the section was written) |

Run 2, the 63 decode frames (847 commands each; the per-kind table is in
<ops>/work/perf/m-frame/REPORT.md): host 13.99 ms a frame (submitting 3.04, waiting 10.94); **device
span 13.87 ms, busy 13.12 ms (95 % of the span), idle between consecutive commands 0.75 ms over
846 gaps (mean 0.9 us: 843 under 5 us, 1 of 5-20, 2 of 20-100 (before two `rope` launches), none over
100)**. The one frame dumped (frame 11) agrees: span 13.63 ms, busy 12.84, gaps 0.79; the largest
gap is 68 us before the frame's first kernel, after the `write` memcpy of x; a DeltaNet layer is
19 commands in 0.29-0.32 ms, an attention layer 27 in 0.34-0.37 ms. Kernel durations: 197 under
5 us, 392 of 5-10, 32 of 10-20, 195 of 20-50, 30 of 50-100, 1 over 100 (lm_head, 766 us).

What that says: the decode token is not host-launch-bound and not queue-gap-bound; the device is the
critical path and it is busy for 95 % of the frame with 847 mostly tiny commands. 566 of them run
under 10 us each and take 3.76 ms (29 % of the busy time) together: the floor of a command on this
stack is about 4 us of device time (the smallest kernels: `ew swiglu` 3.8, `matmul 1x2048 f32` 4.0,
`rope` 4.0, `copy` 4.7), so the 847 commands carry ~3.4 ms of pure per-command cost. The large
kernels run far under the card's bandwidth because a single decode row gives them small grids:
`dnrec` 53.6 us for 4 MiB of state (78 GB/s; 32 work-groups), the experts 61 us a layer for ~13 MiB
(~220 GB/s), `matmul 2048x4096` 29 us for 8 MiB (285 GB/s, two kernels with split-K), `512x2048`
9.1 us for 1 MiB (115 GB/s), the router 8.5 us for 0.5 MiB; only `in_proj 12288x2048` (497 GB/s) and
lm_head (664 GB/s) approach it. A token reads ~2.5 GB of weights and state: 3.2 ms at 774 GB/s, so
the frame is 4.3x its bandwidth floor. Per kind a token: matmuls 5.70 ms (371 launches), the experts
2.44 (120), DeltaNet conv + recurrence 1.90 (60), attention with its norms, RoPE and cache writes
~1.25 (70), route 0.80 (40 launches of 19.9 us, one work-group each), norms 0.68 (101), elementwise
0.35 (80).

The five candidates (REPORT.md section 4 has the arithmetic), launches saved a token: (1) the shared
expert's five launches a layer (gate, up, SWIGLU, down, the f32 gate row: 200 a token, 1.32 ms): an
int8 GLU GEMV as `bd_glu_launch` already does for GGUF, and the gate row folded into the router's
GEMV (`rgf`, declined today because the router is int8): -120 launches, ~0.5 ms; (2) the router GEMV
and `cc_route_k` as one kernel: -40 launches and most of route's 0.8 ms; (3) an attention layer's
prologue and epilogue (q norm, RoPE q, k norm, RoPE+f16 k|v, K and V into the f16 cache, K and V into
the DOWN buffer: 8 launches, 80 a token) as one kernel a layer: -70 launches, ~0.3 ms, and the decode
attention split (`cc_attn_dec_k`) from position 0 rather than 128 (`cc_attn_k` walks every position
in 16 work-groups: 98 us a layer at 85 positions); (4) DeltaNet: `cc_dnconv_k` and the b|a GEMV into
`cc_dnrec_k`'s prologue (-60 launches) and the recurrence over more than 32 work-groups (the
`cc_dnrec_split_k` path exists from `CC_DNREC_ROWS` = 8 rows; at S = 1 it is the 32-block kernel):
1.6 ms today; (5) `ew combine` + the next layer's input norm as one `cc_addnorm`-like kernel
(-40 launches, ~0.2 ms), the input norm into the consumer GEMV's prologue, one kernel for the split-K
GEMVs (-40 launches), and x written by a kernel from the host-USM staging rather than a memcpy
(the 68 us gap at the frame's head). Together: 847 -> ~480 commands and ~2.5-3.5 ms of the 13.9;
the recurrence's and attention's occupancy (m-kernels' pass) are the other 1.5 ms. SYCL graphs or
eventless submission alone cannot move the token below the device span, since the host's 3 ms of
submission already overlaps it; they matter once the span shrinks, or if the L0 in-order list
options lower the ~4 us per-command floor (m-submit's micro-benchmarks).

Files: `coli_sycl.h` (section 5), `coli_sycl_profile.cpp` (the sink, the env in `coli_sycl_open`),
`cuda_chain.dp.cpp` (the hooks and names, `cc_fprof_begin` / `cc_fprof_end`, the report at
`cc_shutdown`), `backend_sycl.h` (`COLI_SG_LAUNCH`), `backend_cuda.dp.cpp` (seven plain launches),
`README.md` (the knob). No kernel, no ABI and no default behaviour changed; the tests are untouched.
Measurement files: <ops>/work/perf/m-frame/ (REPORT.md, run_engine.sh, parse_log.py).

### 14.2 Submission cost (perf m-submit, 2026-11-07): what a launch costs on card1, and what does not help

Measured with `<ops>/work/perf/m-submit/submit_bench.cpp` (REPORT.md there has every table, command and log;
nine guarded runs, no kernel lines, SEL unchanged) on the chain's queue shape: in-order, no profiling, UR 0.12's
Level Zero **v1** adapter with **immediate command lists**, which is the default on this card (the v2 adapter is
in the image and used only with `SYCL_UR_USE_LEVEL_ZERO_V2=1`; `UR_L0_USE_IMMEDIATE_COMMANDLISTS=1` reproduces
the default). Device times from `submit_profiling_tag` pairs and, on a profiling queue, per-kernel event
timestamps; host times from `steady_clock`; 800 launches a measurement (a decode token's worth), median of five.

| form | tiny 1-WG kernel (4 KB): host us / launch, device us / kernel, 800 in ms | int8 GEMV 16 MiB: the same | tiny + GEMV alternating: 800 in ms |
|---|---|---|---|
| `q.parallel_for`, event dropped (`CC_K`) | 2.2 / 3.53 / 2.92 | 2.5 / 31.3 / 25.1 | 11.0-11.5 |
| `CC_KSG` / `coli_sycl_launch_sg` (submit + 1 B local_accessor + SG32) | 2.3 / 3.54 / 2.93 | 2.5 / 31.3 / 25.1 | 11.1-11.5 |
| events kept until the wait | 2.2-3.0 / 3.54 / 2.93 | 2.5 / 31.2 / 25.1 | 11.0 |
| `discard_events` queue (deprecated) | 2.2 / - / 2.89 | 2.5 / - / 25.1 | 11.0-11.4 |
| eventless `ext::oneapi::experimental::nd_launch` | 2.4 / **2.77** / **2.35** | 2.6 / 33.8 / 27.1 (+8 %) | 10.5-11.0 (-2..-4 %) |
| graph, 200 nodes x 4 replays | 2.5 (host blocks on the previous replay) / 3.35 / 2.69 | 25.7 / 34.4 / 27.5 | 10.7-11.4 |
| graph, 800 nodes x 1 | **0.03** (28 us a token) / 3.20 / 2.57 | 0.03 / 34.3 / 27.5 | 11.1 |
| one `host_task` every 40 kernels | +65 us a host task | +40 us | +20 us |
| one `ext_oneapi_submit_barrier` every 40 | within noise | within noise | within noise |
| launch + wait each (the placement probe's form) | 7.2 us a round trip | 33.7 us (kernel 29.1 = 576 GB/s) | 17.4 us |

Per-kernel, from the events: a one-work-group kernel is 2.88 us on the device, the gap to the next kernel
0.64 us (mean 0.59, p90 0.80), the same gap behind a 29 us GEMV. So the floor is 3.5 us a small kernel
streamed, 2.2 us of host a launch, and the chain's exact launch form costs nothing over a bare
`parallel_for`. The frame profile of 14.1 agrees (847 commands, gaps 0.75 ms of 13.9 ms): **submission is
not what bounds decode; the kernels' durations are.** Eventless launches are the one code-level change with a
device-side gain (-0.76 us a small kernel), but their effect on GEMV-class kernels depends on the kernel's
size (-3 us at 2 MB, +7 us at 4 MB, +2.6 us at 16 MiB): an engine A/B behind a switch (`COLI_SYCL_EVENTLESS`,
proposed, not written) is the way to decide, with the step time (an eventless frame has no events for the
frame profile).

Graphs (`sycl_ext_oneapi_graph`): record and replay work (aspect `ext_oneapi_limited_graph`); **graph update
does not** on this driver ("Device does not support Command Graph update" at `finalize(updatable)`: neither
whole-graph update nor `dynamic_parameter`, although both are in the 2025.3 headers), so a recorded frame
could only be replayed with device-side tables for the per-token arguments; replaying an executable graph
blocks the host until the previous replay is done. On v1 a graph's GEMVs pay the eventless-like penalty (not
under v2). The host side of a token would drop from ~3 ms to tens of us, which is already overlapped with the
device today: worth it only when the host becomes critical (batched decode, the hybrid path).

Knobs of the Level Zero adapter, all verified present in the image's `libur_adapter_level_zero.so` and each
run once at least on a quiet card: `UR_L0_USE_IMMEDIATE_COMMANDLISTS=0` (regular lists; with and without
`UR_L0_BATCH_SIZE=64`) halves the host cost (1.4-2.4 us) but adds 5-12 % device time and doubles the
launch+wait round trip (14 us); `UR_L0_IN_ORDER_BARRIER_BY_SIGNAL=0`, `UR_L0_DEVICE_SCOPE_EVENTS=1|2`,
`UR_L0_REUSE_DISCARDED_EVENTS=0`, `UR_L0_USE_COPY_ENGINE=0`, `UR_L0_ENABLE_SYSMAN_ENV_DEFAULT=0` (with
`ZES_ENABLE_SYSMAN` unset), `UR_L0_IMMEDIATE_COMMANDLISTS_BATCH_EVENT_COMPLETIONS=1`: within noise;
`UR_L0_USE_DRIVER_INORDER_LISTS=1`: -4 % on plain tiny launches, cancels the eventless gain (a first reading
of 30x slower was contention with an engine run on the same card, not the knob); **`UR_L0_CMD_BUFFER_USE_IMMEDIATE_APPEND_PATH=1`
hangs the host** on a graph replay (100 % CPU in the adapter, device idle in xpu-smi; killed after 10 min;
card unaffected): never set it. `SYCL_UR_USE_LEVEL_ZERO_V2=1`: host 1.8-2.1 us a launch, the mix -2 to -4 %
(-8 % with eventless), a host task 12 us instead of 40-65, graphs without the GEMV penalty: a candidate for an
engine A/B once the two op tests pass under it. Nothing else moves.

A side result that belongs to the kernels (14.1's fusion candidates, the GEMV work): a 2-4 MB int8 GEMV
streamed back-to-back from HBM is bimodal, median 10.2 us but mean 17.8 and p90 37.9 us at 4 MB (2 MB: 7.2 /
14.0 / 35.4), while the same kernel launched one at a time takes 14.5 us round trip and on an L2-resident
working set 6.3 us with no tail; 16 MiB kernels show no tail (576 GB/s). Not an allocation effect (one
allocation against 64: identical); it begins where the working set leaves the L2 (64 MB of matrices: no
tail; 128 MB: tail). Achieved bandwidth in that regime is 147-228 GB/s, which is where the engine's decode
sits (138 GB/s read, 15.5 us a command): the per-op duration distribution of the frame profile (median
against mean), launched alone against streamed, is the next measurement for m-kernels.

Files: `<ops>/work/perf/m-submit/` (`submit_bench.cpp`, `run_env.sh`, `guarded.sh`, `REPORT.md`); no engine
code changed in this pass; `README.md`'s first-contact paragraph now states the measured launch cost.

### 14.3 Kernel efficiency (perf m-kernels, 2026-11-07): the decode-critical kernels against the bandwidth

Where the device's 95 % busy time goes kernel by kernel, measured on card1 with two instruments and no external
profiler (the full tables, the variants and the reasoning in `<ops>/work/perf/m-kernels/REPORT.md`):

- the chain test's new bench modes (`COLI_CUDA_CHAIN_BENCH=gemv|moe|dnrec`, beside the migrated `adec` and
  `route`; `tests/test_cuda_chain.dp.cpp`): the engine's kernels through the chain's table, 200 calls a frame as
  a decode token submits them, rotating over copies of the operands larger than the 192 MB L2, the best of five
  frames, the last answer checked against the CPU (README, "Benches in the chain test");
- `<ops>/work/perf/m-kernels/kbench.cpp`: copies of `gemv_i8_rows`, `gemv_i8_w4` + `gemv_i8_sk_sum`, `moe_gv_gu_k`
  / `moe_gv_dn_k` beside parametrised variants, every launch event-timed on a profiling in-order queue.

Runs: `perf-m-kernels-run1..9` under <ops>/guard.sh (card1, the mandated form; rc 0, no kernel lines, SEL 34c).
The read ceiling of a plain SYCL read kernel on this stack is 674-702 GB/s (the task's 774 is the OpenCL stream
probe's figure); an empty kernel costs 1.9 us of device time and ~5 us a launch pipelined, and the host submits
a launch in 4.7 us whether events are kept, discarded or `nd_launch` is used.

| kernel (a decode token's share, m-frame's frame profile) | today | GB/s (% of 774) | measured fix | after |
|---|---|---|---|---|
| in_proj [12288 x 2048] int8, `gemv_i8_w4` (1.52 ms) | 50.6 us | 498 (64 %) | one output a sub-group, x read from global memory (L1) instead of the SLM staging | 41.0 us (617) |
| q+gate [8192 x 2048], `gemv_i8_w4` (0.37 ms) | 37.1 | 453 (59 %) | the same | 29.3 (576) |
| o_proj / out_proj [2048 x 4096], `gemv_i8_w4` split 2 + sum (1.17 ms) | 29.3 | 286 (37 %) | `gemv_i8_rows` at SG16 (`COLI_SYCL_SG=16`: 20.8 in the chain bench), x from global | 18.4 (458) |
| k, v, shared gate / up [512 x 2048], shared down, router (1.5 ms over 180 launches) | 6.8-9.1 | 62-155 (8-20 %) | at the ~5 us kernel floor: no kernel change helps (best variant 7.4 us); fusion (m-frame 14.1) | - |
| lm_head [248320 x 2048], `gemv_i8_w4` (0.77 ms) | 765.7 | 665 (86 %) | `gemv_i8_rows` at SG32 | 710.6 (719) |
| 8 routed experts int4-g64, `moe_gv_gu_k` + `moe_gv_dn_k` + `moe_sum_k` (2.44 ms) | 53.6 + 7.3 | 232 (30 %) | the expert kernels at SG16 (the gate/up kernel 42.5 -> 24.3 us at the same geometry: register pressure at SIMD32); at SG32 16 lanes a row (66 -> 53) | 46.2 us in kbench (306), 56.1 in the chain bench |
| DeltaNet recurrence `cc_dnrec_k` (1.61 ms) | 53.6 (41.7 in the bench) | 78-104 (10-13 %) | `cc_dnrec_rows_k` at SG16 for a decode row (`COLI_CUDA_DNREC_MIN=1 COLI_CUDA_DNREC_SPLIT=0 COLI_SYCL_SG=16`); the prompt's split kernels are 1.6-2.3x slower at one row | 30.2 us (142) |
| decode attention below 128 positions, `cc_attn_k` (0.98 ms at a 21-token prompt) | 107.8 us at 100 positions | ~2 | the split from position 1 (`COLI_CUDA_ATTN_DEC_MIN=1`) | 31.7 (28.9 with blocks of 16) |
| decode attention at 16K positions f16, `cc_attn_decg_k` | 128.3 | 262 (34 %) | SG16 (118); the f16 path loads 8 bytes a lane and converts (the f32 cache streams 428 GB/s): 16-byte `half8` loads are the fix, not measured | - |
| `cc_route` (0.80 ms) | 19.9 (one work-group; the RTX PRO 4000: 4.9) | - | fusion with the router GEMV (14.1) | - |

Against the CUDA docs (RTX PRO 4000 Blackwell SFF, 432 GB/s): the CUDA card runs the same kernels at 75-95 % of
its bandwidth (lm_head 96 %, the experts 78 %, the 18K attention 79 %) and a decode token at 58 %; PVC today
15-64 % and 23 %, and the variants found here reach 60-80 % on the wide GEMVs. Why PVC is below: one output a
sub-group over 3584 thread slots leaves a 2048-output projection at half a wave and the W4 path at an eighth;
the SLM staging of x costs a barrier and caps the occupancy of long rows (I = 8192: three work-groups a Xe-core);
the int4 expert kernel and the f16 attention read path are instruction-bound (4 ALU ops a nibble and 8 SLM reads
a 16-byte step; `half4` loads), and SIMD32 doubles their register footprint (SG16 is 1.75x faster on the gate/up
kernel with the same geometry); the recurrence is 32 work-groups with a 128-float column a thread (the SIMD16
build spills). The 1 MB matrices and the route are launch-floor-bound and need fusion, not kernels.

Knobs added for these measurements (`cuda_chain.dp.cpp`, defaults unchanged; README): `COLI_CUDA_DNREC_MIN` (the
rows from which the prompt's recurrence kernels are taken, 8), `COLI_CUDA_ATTN_DEC_MIN` (the positions from which a
decode row's attention splits over blocks, 128), `COLI_CUDA_ATTN_DEC_L` (the first block length, 64; the attention
kernels take it in the high half of their `hc` argument so host and device agree; a selection list's rows keep 64,
since their block count assumes it: `=16` failed their op check, run 6). The op test (`cuda_chain_test --device 0`)
is green at SG32 and, for the first time, at SG16 (228 / 228 each), and with `COLI_CUDA_ATTN_DEC_MIN=1
COLI_CUDA_DNREC_MIN=1 COLI_CUDA_DNREC_SPLIT=0` at both (run 5 / 6).

End to end (runs 7 and 8, the engine with N_NEW 256 on the colibench prompt): single runs vary by +-1.3 ms a
token in the engine's own "waiting for the device", so run 8 is an A / B / A / B in one container under the frame
profile (device timestamps): A = today's defaults, B = `COLI_SYCL_SG=16 COLI_CUDA_ATTN_DEC_MIN=1
COLI_CUDA_DNREC_MIN=1 COLI_CUDA_DNREC_SPLIT=0`. The decode frame's device span: A 14.32 / 14.34 ms below 128
positions and 13.73 / 13.71 above; B 13.16 / 13.15 ms everywhere (-8 % / -4 %), with the recurrence 53.1 -> 35.8 us
a layer (-0.52 ms a token), the experts 26.9 -> 25.0, the 2048x4096 GEMVs 14.8 -> 13.3, and the global SG16
costing the one-work-group `cc_route_k` +6.4 us a call (+0.25 ms) and the 1 MB GEMVs +2-3 us (+0.4 ms): the
sub-group width is a per-kernel choice on pvc (SG16 for the experts, `cc_dnrec_rows_k`, the 2048-output GEMVs and
the decode attention; SG32 for `cc_route_k`, the small GEMVs, `addnorm`, lm_head), worth another ~0.6 ms a token
by the same table before any kernel is rewritten.

What to implement (REPORT.md section 6, in order): the GEMV dispatch on pvc (no W4, x from global memory, SG16
for O <= 2048, rows at SG32 for lm_head); the expert kernels at SG16 (and 16 lanes a row at SG32); the attention
split from position 1 and the recurrence through `cc_dnrec_rows_k` at SG16 for a decode row, then a decode
recurrence kernel with a work-group per (head, 32 columns); 16-byte loads in the f16 cache read; then the fusion
candidates of 14.1 and, for the experts and the trunk at one row, int8 activations on the matrix engines' s8 tiles.

### 14.4 Performance pass: the implementation (Codex, 2026-11-07)

**Kept: image68f478d4d795 on gpu-gw:8001; colibench234715 passes the keep rule and decode targets.**
The raw-engine step/wait/frame targets were not all met; their measured misses remain below.
Continuation of `<ops>/work/perf/PLAN.md` from b16d4ee2. Code lives only in the SYCL worktree;
the production CPU source/service was not changed. Each item passed both op suites before
the next edit. All GPU work used card1, the mandated guard, named containers, timeouts,
pre/post snapshots, safe telemetry and the persistent watcher. Through run14: SEL34c,
memory healthOK, no kernel lines. Budget14/15. Full commands/results are in RUNLOG/logs;
patches, tested binary hashes, parsed JSON and reports are in `<ops>/work/perf/codex/`.

Final implementation and reversal:

| item / commit | resulting behavior | restore knob (`COLI_CUDA_` prefix) |
| --- | --- | --- |
| 2.1 / 1c977a2a | plain attention splits from position1; selection lists retain128/64-position blocks, window threshold128 | `ATTN_DEC_MIN=128` |
| 2.2 / 8d190428 | decode/verify recurrence uses rows kernel at SG16 on PVC; KD256 keeps R8, other KDs R16; prompt dispatch independent | `DNREC_DEC=0`, `DNREC_SG=32` |
| 2.3 / 906b9596 | global-read NR1 int8 GEMV; shape-only SG16 for1024–2048 outputs on PVC, profile width elsewhere; NR rows preserve decode bits | `GEMV_XS=1`, `GEMV_SG=32`; original W4-off default superseded below |
| 2.4 / 9d35a9ed | expert gate/up/down and both block-attention stages SG16 on PVC; existing lane/grid rules | `MOE_SG=32`, `ATTN_SG=32` |
| 2.5 / 1e56f13e | README knobs and this section, updated as qualification proceeds | documentation |
| PLAN4 / 958e0fa1 | prompt split cfg162 on PVC,84 elsewhere, at unchanged profile subgroup width | `DNREC_CFG=84` |
| PLAN6 / 57910f9c | W4 restored for every S; W4 NR1 also reads x globally, same split/reduction; S>1 staged | `GEMV_XS=1`; `GEMV_W4=0` retains the rejected experimental family |

The W4-off experiment made four-row lm_head3424.29us versus2240.11us with W4=1
(**1.5286x, over1.5**). PLAN6 allows W4-off only with zero lookup hits. That exception
was not established: baseline colibench and the new engine both accept drafts. The conservative
fallback is W4 for every S, tested first by env in run11, then made the default. The final
W4 XG variant skips NR1 staging/barrier but preserves every value, split, accumulation and
shuffle order. SLM-cap eligibility stays unchanged so row counts cannot change kernel family.
The final tiny tests cover I64 and2048 at O1023/1024/2048/2049, CPU accuracy and exact
verify/decode bits. There are244 checks (228 originally); eight were added with item2.3,
eight with the W4 fallback. SG overrides accept offered16/32 only, otherwise warn and fall back.

Complete old-behavior restoration on this source:

```sh
COLI_CUDA_GEMV_W4=1 COLI_CUDA_GEMV_XS=1 COLI_CUDA_GEMV_SG=32 COLI_CUDA_MOE_SG=32 \
COLI_CUDA_DNREC_DEC=0 COLI_CUDA_DNREC_SG=32 COLI_CUDA_DNREC_CFG=84 \
COLI_CUDA_ATTN_SG=32 COLI_CUDA_ATTN_DEC_MIN=128
```

Builds used HANDOVER7.2, without GPU nodes: `make qwen36 colibri sycl-test SYCL=1
ARCH=x86-64-v4 -j8`. All completed with MAKE_EXIT0 and host profile104/104. Existing migrated
warnings and IGC spill notes remain. Item2.3 initially exposed an unused wpb; removed before
its corrected `build-2.3-final.log` and GPU gate. Final build is `build-w4-xg.log`.

| item / guarded run | chain | backend | log under `<ops>/logs/` | safety |
| --- | --- | --- | --- | --- |
| 2.1 /1 | 228 | green | `codex-perf1-gate21-20261107T224604Z.log` | rc0, SEL34c, no kernel lines |
| 2.2 /2 | 228 | green | `codex-perf1-gate22-20261107T225029Z.log` | rc0, SEL34c, no kernel lines |
| 2.3 /3 | 236 | green | `codex-perf1-gate23-20261107T225949Z.log` | rc0, SEL34c, no kernel lines |
| 2.4 /4 | 236 | green | `codex-perf1-gate24-20261107T230436Z.log` | rc0, SEL34c, no kernel lines |
| 2.5 /5 | 236, also SG16 | green | `codex-perf1-gate25-20261107T230631Z.log` | rc0, SEL34c, no kernel lines |
| prefill /7 | 236, also SG16 | green | `codex-perf1-prefill-gate-20261107T231528Z.log` | rc0, SEL34c, no kernel lines |
| W4 env fallback /11 | 236, also SG16 | green | `codex-perf1-w4-fallback-20261107T232206Z.log` | rc0, SEL34c, no kernel lines |
| final W4 XG /12 | 244, also SG16 | green | `codex-perf1-w4-xg-20261107T232955Z.log` | rc0, SEL34c, no kernel lines |

Measurement history (all fast-path assertions checked before taking numbers):

| variant / logs | profiled B device span | unprofiled step / wait ms/token | raw tok/s | PPL nats/token |
| --- | --- | --- | --- | --- |
| before, measured earlier | 14.32 below128 /13.73 above128 | 14.3–14.6 /8.3–9.5 | see14.3 | 0.223001 (CPU0.222445) |
| W4-off, runs8/9/10 | 11.86 /11.85 (817commands) | 12.5 /9.38 | 78.01 | 0.222407 |
| W4 staged env fallback, run11 | 12.59 /12.77 (857commands) | 13.1 /9.92 | 74.27 | 0.222407 |
| final W4 XG, run12 | 12.44 /12.61 (857commands) | 13.2 /8.82 | 72.91 | 0.222407 |

Final step13.2 and wait8.82 **miss the13.0/7.5 targets**. Final frame span also misses the
original <=12.1 goal after the required W4 fallback. These are recorded as misses, not passes.
All variants had VRAM100%,331matmuls/forward (final84405/255), no host-routed layers.
PPL is320tokens with f16 KV; final0.222407 passes0.2245. Host submission varied between
about3 and6ms across profiled arms; wall timing includes host work and is never inferred
from profiled tok/s. Colibench's keep rule remains the deployment decision.

Run8 log `codex-perf1-engine-abab-20261107T231622Z.log`, run9
`codex-perf1-engine-timed-20261107T231853Z.log`, run10
`codex-perf1-ppl-20261107T232008Z.log`. Run10 engine returned0 but wrapper returned1:
it expected a decode-only resident summary which PPL skips on its early return. Residency
was independently proved by warmstart10240VRAM,40GPU layers,0hybrid and XMX; wrapper fixed,
original log preserved, no GPU rerun. Full evidence `codex/ppl.json`. Runs11/12 PPL wrappers
passed with the corrected check. Experimental reports: `engine-abab.md/json`,
`engine-timed.json`, `w4-staged.json`. Final report follows.

| arm | commands | frames | device span ms | busy ms | submit / wait ms |
| --- | --- | --- | --- | --- | --- |
| A1 | 847 | 105 | 14.12 | 13.38 | 2.95 / 11.28 |
| A1 | 857 | 146 | 13.49 | 12.72 | 2.93 / 10.66 |
| B1 | 857 | 248 | 12.44 | 11.76 | 2.84 / 9.7 |
| A2 | 847 | 105 | 14.1 | 13.36 | 2.86 / 11.34 |
| A2 | 857 | 146 | 13.48 | 12.71 | 2.89 / 10.7 |
| B2 | 857 | 248 | 12.61 | 11.88 | 5.65 / 7.17 |

Per-kind means, modal frames. A is above128 positions; B spans all positions. Recurrence names are combined.

| kind | A launches | B launches | A device ms | B device ms | delta ms |
| --- | --- | --- | --- | --- | --- |
| moe gate+up, down | 80 | 80 | 2.1420 | 1.9930 | -0.1490 |
| dnrec | 30 | 30 | 1.6110 | 1.1160 | -0.4950 |
| matmul 12288x2048 fmt1 | 30 | 30 | 1.5155 | 1.4600 | -0.0555 |
| matmul 2048x4096 fmt1 | 80 | 80 | 1.1740 | 1.0765 | -0.0975 |
| matmul 512x2048 fmt1 | 100 | 100 | 0.9205 | 0.8085 | -0.1120 |
| route | 40 | 40 | 0.7970 | 0.8000 | +0.0030 |
| matmul 248320x2048 fmt1 | 1 | 1 | 0.7620 | 0.7620 | +0.0000 |
| attn (positions in blocks) | 20 | 20 | 0.5580 | 0.5010 | -0.0570 |
| norm | 61 | 61 | 0.3855 | 0.3945 | +0.0090 |
| matmul 8192x2048 fmt1 | 10 | 10 | 0.3705 | 0.3600 | -0.0105 |
| matmul 256x2048 fmt1 | 40 | 40 | 0.3420 | 0.2880 | -0.0540 |
| addnorm | 40 | 40 | 0.2970 | 0.3355 | +0.0385 |
| dnconv | 30 | 30 | 0.2980 | 0.3500 | +0.0520 |
| moe sum | 40 | 40 | 0.2915 | 0.2970 | +0.0055 |
| matmul 2048x512 fmt1 | 40 | 40 | 0.2725 | 0.2440 | -0.0285 |
| ew combine | 40 | 40 | 0.1955 | 0.1985 | +0.0030 |
| matmul 64x2048 fmt0 | 30 | 30 | 0.1760 | 0.1810 | +0.0050 |
| matmul 1x2048 fmt0 | 40 | 40 | 0.1640 | 0.1960 | +0.0320 |
| ew swiglu | 40 | 40 | 0.1540 | 0.1560 | +0.0020 |
| copy (kernel) | 23 | 23 | 0.1050 | 0.1115 | +0.0065 |
| copy regions (float16) | 20 | 20 | 0.0990 | 0.1015 | +0.0025 |
| rope_f16 | 10 | 10 | 0.0450 | 0.0460 | +0.0010 |
| rope | 10 | 10 | 0.0405 | 0.0410 | +0.0005 |
| barrier | 1 | 1 | 0.0020 | 0.0020 | +0.0000 |
| write (memcpy) | 1 | 1 | 0.0000 | 0.0005 | +0.0005 |

Unprofiled: step13.2ms/token and wait8.82ms both miss13.0/7.5 targets; raw72.91tok/s, VRAM100%,84405/255=331matmuls/forward. These target misses remain explicit; colibench run234715 passes the deployment keep rule.

W4 staged/global-read microbench, XS1/0/1/0 (us/launch):

| shape O x I | staged A1 | global B1 | staged A2 | global B2 |
| --- | --- | --- | --- | --- |
| 8192 x 2048 | 39.68 | 34.55 | 40.04 | 34.73 |
| 2048 x 8192 | 40.96 | 39.36 | 40.59 | 39.20 |
| 12288 x 2048 | 50.48 | 47.28 | 51.07 | 47.16 |
| 2048 x 4096 | 25.51 | 26.04 | 25.76 | 26.89 |
| 512 x 2048 | 8.76 | 7.53 | 8.79 | 7.55 |
| 248320 x 2048 | 753.83 | 748.85 | 754.35 | 750.34 |

Prefill measurement (run6 `codex-perf1-benches-20261107T230730Z.log`, CPU-reference checks green):

| configuration | 256rows us/row | 2048rows us/row | decision |
| --- | --- | --- | --- |
| SG32 cfg84 | 8.1679 | 8.0338 | old default |
| SG32 cfg162 | 5.7701 | 5.6344 | -29.36%/-29.87%; adopted on PVC |
| SG16 cfg162 | 2.9327 | 2.8390 | not adopted: global SG16 regresses decode |

Other256-row configurations and all chain benches are in `bench-summary.txt` /
`prefill-measurements.json`. Decode recurrence41.35->30.98us/call, MoE55.37us/call.
Global subgroup width remains32 on PVC. Colibench confirms prefill_long TTFT2.563->2.349s (-8.36%); prefill_short0.1388->0.1429s is slightly slower.

| end-to-end qualification | baseline201354 | candidate |
| --- | --- | --- |
| image / rollback tag | 18eebc617946, preserved as pre-perf1 | 68f478d4d795, code57910f9c, also tagged perf1 |
| outputs dev-base / cpu-spr | dev-base3/3 identical; CPU late numerical drift | dev-base3/3 identical (369/541/661chars); CPU first differences107/identical541/91chars; sane1.0 |
| decode_short tok/s | mean70.559, max70.679777 | mean78.4476, min78.2778>70.6798, max78.6403; target passed |
| decode_long tok/s | mean73.438, max74.752423 | mean78.5379, min78.3244>74.7524, max78.6865; target passed |
| mixed_long tok/s | mean73.476 | mean77.9081, min77.5743; target passed |
| prefill_long TTFT s | 2.563 | mean2.3490, max2.3517; target passed |
| expert_disk_s | all-resident | 0 on all18 measured requests; target passed |

The new colibench run uses `codex/colibench_card1.py` to restrict samplers to card1
(the old harness enumerated all four cards), with the correct gateway ENGINE_PID and
engine log. This telemetry/config-capture difference is recorded in TUNING.
Deferred as PLAN5: shared-expert GLU, router/route and attention fusion, conv/recurrence
fusion, norm boundaries, eventless/v2 submission, s8 XMX experts and wider f16 KV loads.
HANDOVER9 automatic placement is next only after this pass's gateway qualification.

Image build log `codex/build-image.log`, manifest `image-manifest.txt`; the source revision
is an OCI image label. Run13 `codex-perf1-gateway-20261107T234530Z.log` repeated the image’s
chain244/backend gates, then started the gateway and completed64tokens. Its proof log records
40GPUlayers,10240VRAM experts,XMX and0hybrid. The old guard’s rc137 was the explicitly journaled,
planned SIGKILL stop; it completed its post-snapshot before the new guard started. The new
service is detached with init/SIGKILL; bounded startup and every agent inference client are
guarded, so an idle healthy service has no delayed lifetime timeout. Persistent watcher and
docker log follower remain active. CPU8000 unchanged.

Run14 `codex-perf1-clients-20261107T234653Z.log`: outputs checks and colibench234715,
label `gpu-sycl-card1-perf1`, Light,2warmups/3repetitions,18/18 measured requests green.
Keep margins: short min78.277827>baseline max70.679777; long min78.324432>74.752423.
Mean gains11.18%/6.94%; all decode repetitions>=78. Lookup accepted110drafts on measured
turns, generating turns1.00–1.09tokens/forward, confirming W4’s verify path is relevant.
Outputs and divergence contexts: `codex/outputs-divergences.json` and
`colibench/data/experiments/gpu-sycl-card1-perf1-outputs.md/jsonl`.
Verdict: `codex/colibench-verdict.json`; every colibench run has a TUNING row.

Metadata caveat: actual gateway301047/engine301744 and all CUDA environment values are in
snapshot stable.processes. The harness’s dedicated engine_knobs regex omits some CUDA/SYCL
names, and source_dir/container defaults still name the production checkout/colibri-vulkan.
`codex/colibench-provenance.json` records actual code/image, captured GPU env and original
artifact SHA256s. Raw artifacts were not rewritten. The wrapper now supplies the worktree
source/gpu-gw and expands the knob regex; CPU-only metadata verification passed. These
metadata changes do not alter requests or timing. Telemetry was restricted to card1.
Both runtime PIDs were correctly captured; the old201354 PID gap is resolved.

Performance pass ends at14/15 guarded runs with no hardware safety event. The remaining
roadmap proceeds with automatic placement first; keep this accepted image available while
new source changes are qualified.

## 15. Automatic placement (qualified and deployed, 2026-11-08)

The old Qwen3.6 probe times ten host-to-device-to-host GEMVs with a completion
after each. The engine's resident chain pays one completion for many operations.
`SYCL=1` now also defines `COLI_SYCL` for the C engine and selects a resident probe
by default when the chain is enabled. Other backends retain the synchronous
default. `COLI_TRUNK_PROBE_PIPE=0` restores that probe; `COLI_TRUNK_PROBE=0`
continues to trust placement without measuring. Explicit placement is unchanged.

The resident probe uploads up to ten distinct layers' projections, matching the
CPU's working set, writes the activation once and keeps output on the device.
Three rounds each warm three matrices and time ten with one completion per batch.
It frees its temporary tensors and buffers and keeps the context usable. An
operation failure stops startup instead of launching a second probe. Engine
reporting takes a counter baseline so probe operations do not inflate the
matrix count per forward. No new device kernel or backend ABI is introduced.

Host fake-chain tests cover batched completions, temporary buffers/handle usage,
the override, a disabled or missing chain and a failed submission. Final-source run passed, including the SYCL default; CPU-only syntax also passed.
Builds passed. Run1 `codex-placement-gates-20261108T000714Z.log`: chain244 at default
and globalSG16, backend green, rc0/SEL34c/no kernel lines/memory healthOK. Engine
qualification passed as below.
Launcher host tests:31 passed. Artifacts and bounded eight-run verification plan:
`<ops>/work/placement/codex/`. The accepted perf1 gateway stayed idle throughout
qualification and was replaced only after the candidate passed.

Image audit also reproduced a packaging bug without any GPU device exposed:
`COLI_ENGINE=/app/colibri` overrides the family resolver and selects GLM for Qwen.
With that override removed, the existing launcher selects `/app/qwen36` and
recognizes its SYCL link. The candidate Dockerfile removes that default. GPU
selection, automatic allowance, placement and chain policy already exist in the
launcher/engine; a model-specific40GB default is unnecessary. Candidate-image
routing and real `coli serve --gpu 0` qualification passed in run4.

Card1 evidence, with the accepted perf1 gateway idle throughout:

| measurement | legacy automatic probe | resident automatic probe |
| --- | ---: | ---: |
| one8192x2048 GEMV, GPU | 0.161 ms (synchronous) | 0.035 ms (both profiled runs) |
| CPU, ten layers' projections | 0.124 ms | 0.123 / 0.127 ms |
| placement | trunk and experts withdrawn | 40 layers, head,10240 experts in VRAM |
| matmuls per forward | chain off | 331, startup probe excluded |

Run2 `codex-placement-engine-20261108T000811Z.log`, N_NEW256 with frame profiling:

| arm, in execution order after the legacy check | device span,857 commands | profiled raw tok/s |
| --- | ---: | ---: |
| manual1 | 12.59 ms | 55.85 |
| automatic1 | 12.62 ms | 56.35 |
| manual2 | 12.55 ms | 56.90 |
| automatic2 | 12.44 ms | 61.88 |

Device costs agree; host profiling overhead varies. The legacy fallback ran at
14.50tok/s in this portable SYCL build; that is not the production CPU control
build's rate. Every GPU arm proved40layers/10240VRAM/XMX/0hybrid/331matmuls.
Run3 `codex-placement-quality-20261108T001108Z.log`, without event profiling:
73.28tok/s, step13.1ms, decode wait8.90ms, VRAM100%. Perf1 was72.91tok/s,
step13.2ms/wait8.82ms. The original13.0ms/7.5ms goals remain unmet.
PPL is unchanged:0.222407nats/token over320tokens, below0.2245.
Both guards ended0, SEL34c, no kernel lines, memory healthOK.
`placement/codex/engine-abab.json` and `quality.json` retain the parsed proofs.
Image15deeb14b462 (codea33fcca9) passed image chain244/backend tests, actual
`coli serve --gpu 0` startup and64tokens on8002 in run4
`codex-placement-gateway-20261108T001819Z.log`. No explicit budget, placement,
probe or chain setting is present. Run5 `codex-placement-clients-20261108T001943Z.log`
passed outputs: dev-base3/3 identical; cpu-spr first differences107/identical541/91,
sane1.0, matching perf1. Its colibench002002 is excluded from speed comparison:
the random nonce changed all prompt lengths by3tokens, independently of seed42.

Run6 `codex-placement-fixedprompt-20261108T002434Z.log`, colibench20261108-002449,
labelgpu-sycl-card1-auto1-fixedprompt, reused the exact20 prompts and caps from
accepted234715 through a local wrapper. It passed18/18 requests with matched
prompt/completion counts, expert_disk_s0 and110accepted drafts.

| workload | perf1 mean [min,max] | automatic mean [min,max] | decision |
| --- | --- | --- | --- |
| decode_short tok/s | 78.448 [78.278,78.640] | 79.481 [79.337,79.658] | min exceeds old max |
| decode_long tok/s | 78.538 [78.324,78.686] | 79.434 [79.130,79.655] | min exceeds old max |
| mixed_long tok/s | 77.908 [77.574,78.326] | 78.955 [78.650,79.419] | retained |
| prefill_long TTFT s | 2.349 [2.346,2.352] | 2.342 [2.328,2.360] | retained |

This establishes throughput retention; the small mean change is not attributed
to a startup-only probe. Original201354 keep gates and>=78decode means also pass.
TUNING has both runs, including the exclusion. `placement/codex/colibench-verdict.json`
checks exact prompts/counts/source/runtime metadata and preserves artifact hashes.

Run7 `codex-placement-promote-20261108T002734Z.log` intentionally stopped both
old/candidate services, then returned125 before any new GPU work: Docker's --rm
cleanup had not yet released the gpu-gw name. After confirming removal, run8
`codex-placement-recover-20261108T002841Z.log` passed image gates/startup/64tokens
on8001. The promotion script now waits for container deletion. No hardware abort
occurred; all runs ended with SEL34c/no kernel lines/memory healthOK. CPU8000 was
untouched. Eight of eight phase runs consumed, including the Docker recovery.

Primary gpu-gw/d80e962a0a78: Python325511, engine326238, image15deeb14b462.
Tags dev/auto1 now point there; perf1 retains68f478d4d795 and pre-perf1 retains18eebc617946.
Primary log `gpu-gateway-auto1-primary-20261108T002841Z.log`; complete process env,
device isolation and startup proof in `placement/codex/primary-provenance.json`.
Next: HANDOVER9.2 prefill upgrades; no prefill candidate is enabled by this change.

## 16. Int8 trunk prompt activations (rejected; default off, 2026-11-08)

COLI_SYCL_XMX_S8=1 requests the existing int8 XMX kernel for fmt1 dense prompt
rows only; default0 preserves f16 activation staging. The profile must list s8,
K must be a multiple64 and<=131071 (conservative signed32 accumulation bound).
Decode/verify GEMV, fmt2 and grouped experts retain their arithmetic. A first-use
log proves the s8 path. No new device kernel is introduced.

Independent CPU row-quantization reference covers the backend prompt dispatch;
original f16 tolerance and decode bit-equality tests remain. The chain benchmark
now compares against its selected activation precision rather than an f32 dot
which also mismatched the established f16 prompt path. Direct XMX tests already
cover s8 tails, zero padding, large outliers and the integer staging policy.

Usual PPL does exercise21promptrows; a second fixed text reference uses256prefix
rows and400scored tokens to test longer quantized state. Bounded8-run plan,
artifacts and checkpoint: <ops>/work/prefill/codex/. Qualification below rejected
adoption; acceptedauto1 gateway/image unchanged.

Build passed (build-prefill-s8.log, host profile104checks). Guarded run1
`codex-prefill-s8-gates-20261108T003752Z.log` passed tiny/full XMX, chain244
at default and S8/globalSG16, and backend at default/S8. The s8 dispatch was
logged and tested against the independent quantized reference. Decode verify
bit-equality remains green. rc0, SEL34c, no kernel lines, memory healthOK.
Performance and quality results follow; default remains0.

Run2 `codex-prefill-s8-bench-20261108T003906Z.log`,64launches/frame, A/B/A/B.
Three projection shapes at256/2048rows; allocated weight pool768MiB. Large
matrices cycle48copies beyondL2. The512x2048 case touches only64MiB of the
allocated pool and is explicitly a cache-resident comparison.

| O x I, rows | f16 A1 / A2 us | s8 B1 / B2 us | mean reduction |
| --- | ---: | ---: | ---: |
| 8192x2048,256 | 411.48 / 402.38 | 229.94 / 224.93 | 44.11% |
| 2048x8192,256 | 460.94 / 462.58 | 323.75 / 324.20 | 29.84% |
| 512x2048,256 | 81.33 / 81.82 | 71.24 / 70.82 | 12.93% |
| 8192x2048,2048 | 2091.68 / 2090.06 | 1178.16 / 1177.31 | 43.67% |
| 2048x8192,2048 | 2517.97 / 2511.97 | 1614.68 / 1614.53 | 35.80% |
| 512x2048,2048 | 238.37 / 238.92 | 218.13 / 218.88 | 8.44% |

All selected-arithmetic CPU checks passed, s8 first-use log proved the path.
Run3 `codex-prefill-s8-quality-20261108T004001Z.log` rejected the candidate:
original reference NLL0.227469/320 exceeds0.2245 (A0.222407, +2.28%). Baseline
longer-prefix text NLL2.793515/400. The script stopped at the first candidate
failure; candidate longer-prefix and remaining ABAB arms were NOT run.
No raw throughput/gateway/outputs/colibench promotion followed this rejection.

Run4 `codex-prefill-s8-restore-20261108T004202Z.log` explicitly sets
COLI_SYCL_XMX_S8=0: NLL returns exactly0.222407/320. The option stays off by
default in the commit, as PLAN6 allows rejected optimizations behind their knob.
Acceptedauto1 image/gateway was never changed. The synthetic speed gain does
not justify the model-quality loss. Further S8 refinement is deferred; proceed
to grouped-MoE measurements. Four of eight phase runs used; hardware health
remained clean (SEL34c, no kernel lines, memoryOK); run3 rc42 was a numerical
quality rejection, not a timeout/device fault. Raw tables/verdicts/checkpoint
are under <ops>/work/prefill/codex/.

## 17. Grouped MoE tile (default unchanged, 2026-11-08)

COLI_SYCL_XMX_MOE_TM=8|16 isolates the experts' existing f16 tile from the dense
trunk. Unset inherits COLI_SYCL_XMX_TM (default8);16 requires the listed compound
shape. No new kernel, sort layout or activation precision is introduced. The
first-use log reports actual selection. No default change before evidence.

The direct XMX test adds --moe-bench at Qwen3.6 geometry:256distinct experts,
D2048,I512,top8,S256/2048, planar int4-g64, uniform synthetic routes with5%invalid.
It checks64evenly-spaced placedslots against independent CPU gate/up/down
references, including first/last. Ordinary tiny/full tests still check all
placedslots. Timings separate gate/up and down. This synthetic route distribution
is not a substitute for engine measurement. Bounded8-run plan/checkpoint/artifacts:
<ops>/work/prefill-moe/codex/. S8 trunk remainsOFF; acceptedauto1 unchanged.

Build passed (build-prefill-moe.log/profile104). Run1
`codex-prefill-moe-gates-20261108T004758Z.log` passed9suites: tiny/fullXMX atTM8
and16, chain244/backend each8/16, andTM16chain atglobalSG16. Actual tile logs
match selection; no arithmetic mismatches. rc0/SEL34c/no kernel lines/memoryOK.

Run2 `codex-prefill-moe-bench-20261108T004932Z.log` completed during a conversation
interruption; resumed00:55Z and checked rc0/SEL34c/no kernel lines/memoryOK.
No run was retried. All independent sampled references passed.

| rows, phase | TM8 A1 / A2 ms | TM16 B1 / B2 ms | TM16 time reduction |
| --- | ---: | ---: | ---: |
| 256 gate/up | 2.710 / 2.710 | 3.279 / 3.282 | -21.05% |
| 256 down | 1.185 / 1.181 | 0.992 / 0.992 | 16.15% |
| 2048 gate/up | 3.686 / 3.694 | 4.429 / 4.433 | -20.08% |
| 2048 down | 1.655 / 1.659 | 1.453 / 1.454 | 12.28% |

All-TM16 is rejected. The phases are timed separately with repeatedweights;
small downweights may fitL2. Summedtimes are not an engine prediction. Next
candidate keepsgate/upTM8 andsetsdownTM16 via independent override, samef16
arithmetic. Newgates/ABABbench/enginequality required; no default change yet.

COLI_SYCL_XMX_MOE_DN_TM=8|16 now overrides only phase1 after the all-MoE
choice; unset preserves it. Phase-specific first-use logs prove gate/upTM8
anddownTM16 independently. Both are existing kernels; no math or layout change.
Phase budget amended8->12 before the next run to qualify this isolated candidate.

Down-only build passed (build-prefill-moe-down.log/profile104). Run3
`codex-prefill-moe-down-gates-20261108T010026Z.log`:9suites passed atGU8/DN8
andGU8/DN16, tiny/fullXMX, chain244/backend, mixedglobalSG16chain. The script
asserts both phase logs for each XMX/chain suite. rc0/SEL34c/no kernel lines/
memoryOK. Next mixedtile benchmark; defaults unchanged.

Run4 `codex-prefill-moe-down-bench-20261108T010154Z.log`, GUfixedTM8,
DN8/16 A/B/A/B, samegeometry/routes/reference sampling:

| rows, phase | GU8/DN8 A1 / A2 ms | GU8/DN16 B1 / B2 ms | mean time reduction |
| --- | ---: | ---: | ---: |
| S256 gate/up | 2.713 / 2.715 | 2.704 / 2.696 | 0.52% |
| S256 down | 1.189 / 1.182 | 0.993 / 0.993 | 16.24% |
| S2048 gate/up | 3.684 / 3.683 | 3.684 / 3.694 | -0.15% |
| S2048 down | 1.658 / 1.662 | 1.450 / 1.453 | 12.56% |

Gate/up remainsunchanged withinnoise; down gains12.56–16.24%. All sampled
references andphaseproofs pass. rc0/SEL34c/no kernel lines/memoryOK. Accepted
primaryloghash unchanged sincequalification. Summedseparatephase times fall
5.30% at256rows/3.80% at2048; cache/route caveats stillapply. Run5 realengine
fixedlongpromptABAB andPPLpair pending. No default/image adoption yet.


Run5 `codex-prefill-moe-down-engine-20261108T010445Z.log`: fixed raw prompt,
3052 encoded tokens, N_NEW256, timers enabled, full residency/331matmuls per
forward, gate/up8 and selected down tile proved in every arm. No S8 path.

| arm | down tile | TTFT s | decode step ms/token |
| --- | ---: | ---: | ---: |
| A1 | 8 | 2.399 | 18.6 |
| B1 | 16 | 2.319 | 18.5 |
| A2 | 8 | 2.333 | 18.5 |
| B2 | 16 | 2.387 | 18.6 |

Mean TTFT reduction0.55%, overlapping ranges: no established end-to-end gain.
The long prompt explains why these decode times are not comparable to the
short-prompt perf1 target. PPL is identical0.222407/320 at both down tiles,
below0.2245. rc0/SEL34c/no kernel lines/memoryOK. Analyzer initially mistook
[timers] prefill tokens for input length: qwen36.c counts all S>1 calls there,
including verification. Encoded prompt tokens are3052 in every arm; timer
counts3155/3163/3163/3155 are preserved separately in engine.json. No GPU retry.

Decision: retain both overrides as opt-in, default8 unchanged. No candidate
image, gateway replacement or colibench run: microbench gain alone is insufficient.
Phase closed after5/12 guarded runs. Accepted auto1 and CPU services untouched.
Large S2048 useful synthetic throughput is about17.7TF/s gate/up and19.7/22.5TF/s
down8/16; historical4–8TF/s used smaller geometry, not these model dimensions.
Continue attention separately; 16K/32K qualification remains outstanding.


## 18. Opt-in XMX prompt attention (slower; default off)

COLI_SYCL_XMX_ATTN=1 replaces the weak attention stub for plain causal chunks.
Default0 preserves accepted auto1. WG128/SG16,8query rows by32keys, profile-listed
f16 8x16x16 QK/PV matrices, float online softmax/accumulators. Q/K/V clamp tohalf,
probabilities round tohalf; this changes arithmetic and needs modelquality gates.
Both f32/f16 KV inputs stage the same values, preserving existing rounded-cache
bit equality expectations. hd64/128/256 only; windows/sinks/rings/selections and
invalid dimensions/strides/offsets decline before submission. No scratch or host
atomics; finite uniform key loop and workgroup barriers. Device exceptions reach
cc_try rather than silently submitting fallback after an error.

New test_sycl_attn --small checks each head dimension first at1row, then9/17row
and key-block tails, offsets/strides, GQA, gate, zero queries, output sentinels,
unsupporteddispatches, independent CPU globalsoftmax and cache-format equality.
Full mode adds256/2048rows (four sampledCPUrows, alloutput bounds/finite checks).
--bench uses actualchain dispatch and8operations/completion after CPU checking.
Existing chain/backend tests and thresholds untouched. Phase PLAN/REVIEW/scripts/
checkpoint: <ops>/work/prefill-attn/codex/. No GPU result claimed before qualification.


Initial AOT build failed in IGC AccessChain for loop-carried matrix arrays;
unrolling alone did not fix it. Named accumulators preserve the same math and
pass isolated pvc+bmg AOT, then full build-prefill-attn-named.log MAKE0/profile104.
No GPU contact occurred during those compile failures. Run1
`codex-prefill-attn-tiny-20261108T012050Z.log`: all18 tiny cache cases passed,
126unsupported forms declined without touching output, cache forms bit-equal.
Maxerror7.16e-5 against halfQ CPUreference,7.90e-5 against originalfloatQ.
Guardrc0/SEL34c/no kernel lines/memoryOK. Full tests/gates remain next.


Run2 `codex-prefill-attn-gates-20261108T012138Z.log` passed6suites: full direct
(including2048rows after6144), chain/backend withoption0 and1, option1chain at
COLI_SYCL_SG=16. Existing chain longchunk error4.00e-6 (limit1e-4), original
rounded f32/f16 cache bit equality passes; no tolerance/test bypass changes.
Guardrc0/SEL34c/no kernel lines/memoryOK. Implementation committed withdefault0;
interleaved benchmark and engine/PPL remain required before adoption.


Run3 `codex-prefill-attn-bench-20261108T012241Z.log`: actual chain, H16/KV2/hd256,
f16cache, gate, causalpos0,5batches of8ops/completion, same synthetic inputs.

| rows | fallback A1 / A2 ms | XMX K32 B1 / B2 ms | decision |
| --- | ---: | ---: | --- |
| 256 | 0.667495 / 0.667739 | 1.064793 / 1.040455 | slower, reject |
| 2048 | 47.812643 / 47.760306 | 48.545186 / 48.499107 | slower, reject |

EveryCPUcheck passed, actualpath logged, guardrc0/SEL34c/kernelclean/memoryOK.
No engine/PPL/image/colibench adoption from this result. Defaultattention0.
Next isolate keyblock64, which halves loopbarriers and uses4/8QKsubgroups, but
increases SLM and may lower occupancy. Phasebudget amended12->15 before newGPU
work; PLAN records tiny4/gates5/bench6 then conditionalmodel/gatewayqualification.


K64 variant: COLI_SYCL_XMX_ATTN_K=64 (default32) selects a64keyblock, same8query
rows/WG128/SG16 and precision. FourQKsubgroups, halfloopbarriers;71KiB SLM athd256
is checked against profile. Generalized per-lane softmax to4keys and addedtiny
9rows after59 for the new boundary. IsolatedAOT and fullbuild-prefill-attn-k64
passpvc+bmg/profile104. Run4 `codex-prefill-attn-k64-tiny-20261108T012852Z.log`
passes24cachecases/168declines withbit-equalcacheforms, rc0/SEL34c/kernelclean/
memoryOK. Fullgates andspeed remainpending; defaultattention0 unchanged.


Run5 `codex-prefill-attn-k64-gates-20261108T012942Z.log` passed7suites:
expandedtinyK32, fullK64 direct, chain/backend off/onK64, K64chain globalSG16.
Original cachebitchecks and numerical limits unchanged; longchunk4.23e-6<=1e-4.
Guardrc0/SEL34c/no kernel lines/memoryOK. Primarylog hashunchanged, no concurrent
inference. K64 correctnessqualified, interleaved speed comparison next.


Run6 `codex-prefill-attn-k64-bench-20261108T013047Z.log`, sameactualchain benchmark:

| rows | fallback A1 / A2 ms | XMX K64 B1 / B2 ms | decision |
| --- | ---: | ---: | --- |
| 256 | 0.663423 / 0.668240 | 1.650789 / 1.652492 | slower, reject |
| 2048 | 47.767313 / 47.766390 | 93.247990 / 93.188744 | slower, reject |

K64 worsens performance despite fewer barriers; increased SLM/occupancy cost is
a hypothesis, not a measured attribution. No hardware counters were collected.
EveryCPUreference and actualpath check passed. rc0/SEL34c/no kernel lines/memoryOK.
Neither K32 nor K64 improves either benchmark shape. Closephase after6/15runs;
keep attentionOFF/default0 and K32 as the inactive default tile. Both variants
remain available explicitly for future tuning; they are not production winners.

ModelPPL, rawengineABAB, outputs_check, imagebuild and colibench promotion were
NOT RUN for these slower candidates. Modelquality of the new attention math is
therefore unqualified. Acceptedauto1 image/gateway and CPU stayed untouched.
Next16K/32Kfirstlong qualification uses acceptedauto1, not experimental kernels.
All logs, sourcepatches, AOTfailure/fix records, binaryhashes, gate JSON, benchmark
JSON and checkpoint are in <ops>/work/prefill-attn/codex/.

## 19. Accepted auto1 long-context qualification (2026-11-08)


Primary card1 gateway idle but its weights remain resident; temporary engine begins with29GBfree, all10240 experts fit. No identical-prompt B60 comparison claimed.

Image15deeb14b462 / sourcea33fcca9, OMP16, automatic CLI placement, temporary card1 container per case. CPU and primary gateway unchanged. Experimental prefill kernels remain disabled.

| case | prompt tokens | first-long TTFT s | after-decode chunks/s | chunk rows / free GB |
| --- | ---: | ---: | --- | --- |
| short (short-r3) | 16103 | 33.750 | 78.992 / 79.514 | 4096 / 10.2 |
| 16000 (16000) | 16079 | 33.627 | 78.160 / 79.428 | 4096 / 10.2 |
| 32000 (32000) | 32146 | 120.599 | 78.674 / 80.079 | 4096 / 10.2 |

| warmed long_context case | prompt tokens | output tokens | TTFT s | stream chunks/s |
| --- | ---: | ---: | ---: | ---: |
| 32000 | 32163 | 11 | 120.374 | 55.825 |
| short | 16073 | 11 | 33.547 | 59.055 |

Every completed request has explicit new chain stats, zero hybrid layer steps, zero host expert milliseconds, plus startup40GPUlayers/10240VRAM/denseXMX proof. Raw counters are cumulative and logical forwards can contain multiple chunks;331MM/forward is not a long-context invariant.

Historical B60 TTFT32.6s at16K and81.1s at32K are reference points, not matched-prompt paired results. Report stream chunks/s separately from usage completion tokens.

Incomplete attempts retained:
- short: not enough values to unpack (expected 3, got 2). Raw results/server log retained; see parser-recovery.json and chunk-proof-recovery.json. They are not counted as complete cases.
- short-r2: per-request full GPU proof failed. Raw results/server log retained; see parser-recovery.json and chunk-proof-recovery.json. They are not counted as complete cases.

Prompt source/seeds/hashes: prompt-provenance.json. Exact requests, raw SSE, resultJSON, fsynced event journal and server logs: results/<case>/. Guard logs and health snapshots remain in <ops>/logs and <ops>/snapshots.

All three complete-case guards returned0, SEL34c unchanged, no GPU kernel lines,
and memory healthOK. Latest log codex-long-context-32000-20261108T014639Z.log
END01:51:34 (266s). The two incomplete attempts were CPU wrapper assertions,
not GPU failures; preserve raw logs and recovery companions. Budget5/5 closed.
No image or kernel defaults changed. Gateway8001/image15deeb14b462 and CPU8000
unchanged; primary log SHA f1905dda39b2d8b59155e2ad08074ea259038a73ac114badb990ca8d4c87167f.
32K TTFT exceeds historical B60; this qualifies completion/full-device execution,
not a performance win or a numerical PPL comparison for long prompts.

## 20. Hybrid host-wait cancellation and precision review (2026-11-08)

Robustness qualification is complete in source; accepted auto1 image/gateway remains unchanged.
`coli_sycl_host_sync.h` supplies a shared host-only cancellation state. Async
handlers and signal/wait host tasks capture it by value, independent of Dev or
CcCtx lifetime. A short mutex serializes cancellation with flag accesses; after
cancellation returns, even a task starting later cannot touch a retired flag.
Neither mutex scope nor host task invokes a SYCL queue API. CPU atomics only;
no device polling or atomics on host memory. Normal waits retain >= comparison
and the spin/20us backoff. Cancellation does not fabricate completion values.

All fatal chain paths cancel this state. Main-thread ready/lost queries collect
asynchronous errors, including when hybrid engine code polls for route readiness.
Already submitted results after loss remain invalid. This handles observed loss,
not an undelivered driver error or a wedged device; existing guard abort policy
still applies. ZES_ENABLE_SYSMAN is retained for the free-memory query.

Host-only actual-helper tests pass: ready/equal, insufficient word, delayed signal,
blocked cancellation, unchanged flag, null flag after cancellation, shared lifetime.
The tiny integration test has a2s CPU rescue writer and is run with a30s timeout;
it must exit without rescue after either software launch failure or direct shared
state cancellation. It does not inject real driver faults. Tiny qualification passed in `codex-robust-host-tiny-20261108T020139Z.log`:
both modes exited with watchdog0 and word0; ready/lost propagation passed.
Guard rc0, SEL34c, no GPU kernel lines, memoryOK and348 error counters unchanged.
Full gates passed in `codex-robust-host-gates-20261108T020218Z.log`:
chain default and globalSG16, backend q8/q4/q2/f32/e8. Original arithmetic
thresholds and verify/decode bit checks unchanged. Guard rc0/6s, SEL34c,
no GPU kernel lines, memoryOK and348 counters unchanged. No image promotion yet.

Artifacts, review, build/probe logs and durable run budget/checkpoint:
<ops>/work/robustness/codex/. IEEE divide/sqrt cost measurement follows cancellation
qualification; AOT option probe compiled for pvc+bmg with no GPU devices attached.

### 20.1 IEEE division/square root candidate (default remains0)

`SYCL_IEEE_DIV_SQRT=1` adds `-options -cl-fp32-correctly-rounded-divide-sqrt`
to the AOT backend options for both compile/link paths. Build-config tracks it;
invalid/empty/multiple values and IEEE+JIT are rejected before build. Local icpx
and ocloc help plus a tiny no-device probe establish option forwarding; full
candidate build succeeded for pvc+bmg with profile104 and host helper tests.

Immutable base and IEEE engine/test binaries, source patch and SHA manifests are
under <ops>/work/robustness/codex/bin/. Base source2a6b663e includes the qualified
host-wait fix. The IEEE build changes device precision; C engine flags are the same.
Run3 `codex-robust-ieee-gates-20261108T020809Z.log` passed both tiny cancellation
modes, chain244/default and244/SG16, backend q8/q4/q2/f32/e8, and XMX small.
Original tolerances and bit checks retained. Guard rc0/6s, SEL34c/no kernel lines,
memoryOK and348 error counters unchanged. All6 pre/post counter captures across
runs1–3 are identical; later runs pin that reference and also check SEL before launch.

Paired PPL run4 and performance runs5–6 are complete (results below). The
cost experiment uses identical fixed decode and3052-token prefill prompts,
N_NEW256 and COLI_LOOKUP=0 in BOTH variants to remove adaptive draft-count changes.
It is a controlled raw-engine comparison, not a production colibench claim.
Prompt hashes, parser tests on actual prior logs/negative fixtures, and commands
are saved in robustness/codex/. Quality limit remains0.2245 over320 scored tokens.
No default or image promotion yet; accepted auto1 gateway remains unchanged.

### 20.2 Precision measurements and decision


Decision: keep SYCL_IEEE_DIV_SQRT=0. The host-wait fix2a6b663e is qualified; opt-in IEEE knobdc71dba2 remains available. Accepted auto1 image and both gateways unchanged. Default rebuild completed with MAKE0; qwen36, colibri and both op-test binaries are SHA256-identical to the qualified base (restored-artifacts.json). No source default changed.

Host synchronization: actual CPU helper tests passed; tiny software launch-failure/shared-state cancellation both released without rescue, word unchanged. Default and IEEE builds each passed chain244/default+244/SG16 and backend q8/q4/q2/f32/e8; IEEE also XMXsmall13. All original thresholds/bit checks retained.

| raw controlled metric | base A1 / A2 | IEEE B1 / B2 | interpretation |
| --- | --- | --- | --- |
| decode step_ms | 12.8 / 12.9 | 12.8 / 12.9 | ms/token; tied at timer resolution |
| decode decode_wait_ms | 7.25 / 7.33 | 7.64 / 7.48 | ms/token; generated trajectories differ |
| prefill ttft_s | 2.383 / 2.405 | 2.335 / 2.336 | s; same3052-token input, IEEE2.44% lower in this pair |
| prefill step_ms | 13.1 / 12.9 | 12.9 / 12.9 | ms/token; ranges overlap |

All8 arms have256 forwards/256 generated tokens, input21 or3052 tokens, full GPU and100%VRAM. COLI_LOOKUP=0 in both arms removes adaptive verification counts. Generated text differs consistently by precision (raw normalized first differing characters335 decode/89 prefill, zero-based; repeats match within each variant). This is not token-ID or API outputs_check evidence. Model device-wait differences cannot be attributed solely to arithmetic cost when routing differs.

| fixed-input op | base A1 / A2 us | IEEE B1 / B2 us | mean IEEE cost | ranges overlap |
| --- | --- | --- | ---: | --- |
| dnrec | 30.84 / 30.6 | 33.81 / 33.85 | +10.12% | False |
| moe | 55.43 / 56.42 | 57.32 / 55.92 | +1.24% | True |
| adec | 115.5 / 117.4 | 116.9 / 115.2 | -0.34% | True |

DNrec/MoE include independent CPU references and whole-frame submission/wait time. Adec is the existing wait-only18Kcache measurement. DNrec shows a clear10.12%cost; MoE ranges overlap. These are op measurements, not summed predictions for engine throughput.

PPL/ref.json320 scored tokens: base0.222407, IEEE0.222541; both<=0.2245, CPUreference0.222445. No NLL improvement from IEEE on this reference. Decode step timings tied and output traces changed, so the modest single-prompt prefill gain does not justify changing the accepted precision default.

All6 guarded runs rc0, SEL34c, no GPU kernel lines, memoryOK and348 error counters unchanged, including comparison to the fixed run1 baseline. Sourcearchives/hashes, base+IEEE binaries, input hashes, raw logs, parsed timing/quality/op JSON, text differences and post-run audit JSON remain under this directory. Kernel-watch and primary gateway log unchanged. No forbidden profiling or driver fault injection.

Build logs: <ops>/work/perf/codex/build-robustness-{host-wait,host-status,ieee}.log. The first build was followed by a source-review correction to ready/lost propagation before any GPU use. Compiler help/setvars wrapper mistakes were CPU-only and documented in REVIEW.md.

Default0 restored; all four artifacts exactly match the already gated/scored baseline, so no redundant GPU rerun. Phase closed6/8runs. Next HANDOVER9.4 qwen38. No image/outputs_check/colibench promotion for IEEE; acceptedauto1 stays on8001.


## 21. Qwen38 native FP8 qualification (in progress, 2026-11-08)

The existing Qwen38 engine builds with `make qwen38 SYCL=1 ARCH=x86-64-v4`
against the qualified default-precision backend at db6f041e, without an engine
source change. The accepted Qwen36 auto1 image and both production gateways remain
unchanged. Artifacts, run plan, hashes and checkpoint: `<ops>/work/models/qwen38/codex/`.

The upstream tiny FP8 fixture uses the unmodified generator, Transformers5.16.1,
tokenizers0.23.1 and torch2.13.0 running on the CPU. Its original logit limit and
cosine threshold are unchanged. The CPU reference uses the same SYCL-linked engine
in a container with no GPU devices and `COLI_CUDA=0 COLI_CUDA_CHAIN=0`.

| Tiny oracle | Greedy tokens | Logit cosine | Maximum absolute error (limit0.01) |
| --- | --- | --- | --- |
| CPU, cache1 and4 | 8/8 each | 0.99996059 | 0.00104511 |
| SYCL, cache1, 4layers on card1 | 8/8 | 0.99998261 | 0.000735082 |

Guard `codex-qwen38-gates-tiny-20261108T022949Z.log` passed listing, both subgroup
launches, XMXsmall/full, chain244 at default andSG16, and backend all formats before
the tiny engine. Actual chain work:8forwards,73frames,1513ops,552matmuls.
END02:30:29Z rc0/15s; SEL34c, empty kernel window, memoryOK, all348 pinned card1
error counters unchanged. One of12 guarded runs used; native checkpoint pending.

Two original roadmap assumptions do not apply to nativeFP8: the pinned host pool
and in-frame MoE are restricted to int4/GGUF. Native experts use the streaming tier
and host step, so this is not evidence for `cc_wait_host`. Also, `coli --gpu0`
sets CUDA/GPUS/DENSE but the native chain still needs `COLI_CUDA_CHAIN=1`.
The native tier does not reserve the chain's memory as the GGUF path does; initial
full-model qualification will use an explicit bounded allowance and context.
Qwen36's absolute PPL threshold cannot qualify a different model; Qwen38 gets its
own fixed reference and matched CPU NLL before gateway/benchmark qualification.


### 21.1 Native checkpoint smoke

`codex-qwen38-native-smoke-20261108T023142Z.log` completed rc0/37s with48GPU
layers (12QSA),553/553 offered trunk matrices, and XMX prompt GEMMs. The allowance
was18GiB with29GiB free beside the idle Qwen36 service;14.03GiB remained for experts.
The RAM cache was128/layer (maximum28.125GiB), OMP32, f16KV, context4096,
MTP/lookup off. No kernel/default/image change was made.

| Metric, initial21prompt/64token raw run | Result |
| --- | --- |
| Load / TTFT | 8.7s / 5.95s |
| Raw total rate (includes prefill) | 2.59tok/s,24.7s |
| Resident experts / VRAM hit rate | 2384/24576 / 52.9% |
| RAM cache hit rate / peak RSS | 55.0% / 35.46GB |
| Decode device wait | 14.32ms/step |
| Routed-expert host step (includes reads/tier work) | 262.86ms/step |
| Decode expert disk service | 173.189ms/step |

These are initial cache/tier timings, not steady-state colibench throughput.
Health remained SEL34c, no kernel lines, memoryOK and348unchanged counters.
The C tokenizer's21prompt IDs match the independent tokenizer library exactly.
Own CPU320-token greedy and human-text quality references are complete: NLL0.3030
and2.6323, respectively, bothrc0. GPUlimits were fixed at CPU*1.007+.0001
(0.305221/2.6508261), allowing for the engine's four-decimal print.
an initial CPU-only input lacked Qwen38's mandatory `schema_version` and was
refused before generation. Its artifacts remain in `quality/attempt-schema-missing`;
correct schema1 and the same-line text parser were fixed without changing the engine.

### 21.2 Native quality rejection and isolation

| Path | Greedy320 NLL | Original limit | Result |
|---|---:|---:|---|
| CPU default | 0.3030 | — | Reference |
| CPU scalar FP8 fallback (no GPU) | 0.3134 | 0.305221 | Diagnostic, fails limit |
| GPU full chain | 0.3288 | 0.305221 | Reject |
| GPU full chain, DNREC_DEC=0 | 0.3288 | 0.305221 | Reject |
| CPU dense + GPU FP8 expert tier | 0.3163 | 0.305221 | Diagnostic, fails limit |

Logs: `codex-qwen38-quality-20261108T023855Z.log`,
`codex-qwen38-quality-dnrec0-20261108T024230Z.log`,
`codex-qwen38-experts-only-20261108T024547Z.log`. First two exit42 at the quality
gate; human reference not run. The DNREC fallback also passes default/SG16 chain
and backend op gates. All three hardware audits pass348 unchanged counters,
SEL34c, clean kernel windows and memoryOK. Native GEMV is already SG32 for all
text matrices (none has1024..2048 output rows); no redundant SG32 run was made.
No cache tuning, candidate image, gateway, outputs_check or colibench promotion.
The expert-only diagnostic returns0 on completion but is not full-path acceptance.

CPU-only `Q38_FP8_KERNEL=scalar` changes this greedy NLL by+3.4323% without any
GPU work (`quality/cpu-fp8-scalar.log/json`). The existing scalar/vector CPU FP8
kernels differ in within-block summation order. This demonstrates sensitivity
of the CPU-generated reference, not acceptance of the GPU path. The original
quality limits and failed results remain; human-text comparison and stronger
FP8 expert op coverage are the next diagnostics.

### 21.3 FP8 expert ABI coverage and human-text diagnostic

`test_sycl_fp8_experts` adds independent FP8 decoding/block-sum CPU references
for dense and grouped expert calls at32x8,201x300 and native2560x640 dimensions.
It requires finite values, relative RMS<=1e-4 (the existing CUDA oracle bound),
async/sync byte equality, row invariance and no tensor accounting leak. It uses
the public ABI and adds no kernels. All shapes pass at both SG widths and both
F8_WARP1/0 paths: maximum relative RMS3.77200533e-7.

Guard `codex-qwen38-fp8-human-20261108T025502Z.log`, rc0/88s: these four suites,
chain244/default+244/SG16, backend gate and original human-text PPL all pass.
Human320 NLL is2.5775 versusCPU2.6323 (limit2.6508261). Full48layer/320forward
chain and positive expert VRAM work confirmed. Hardware audit:348 unchanged
pinned counters, SEL34c, clean kernel window and memoryOK. This result does not
erase the failed greedy gate or authorize candidate promotion; full qualification
remains incomplete. No image, gateway, outputs_check or colibench change.

### 21.4 Stronger expert fixture; native promotion held

The upstream generator's original geometry with `--expert-gain 3` passes CPU
at cache1/4:8/8IDs, cosine0.99995777/maxabs0.00118846. GPUcache1/4 also pass:
8/8IDs, cosine0.99995496/maxabs0.00114489 (original limit0.01), actual4layers/
8forwards/positive FP8expert VRAMhits. Log `codex-qwen38-gain3-20261108T030034Z.log`,
rc0/1s; pinned348counters unchanged, SEL34c, clean kernel window, memoryOK.

A separate `--geometry gguf-256` FP8 fixture failed on CPU before any GPU attempt:
int8trunk8/8IDs butcos0.99942767/maxabs0.019388>0.01; BF16trunk3/8IDs with first
divergence at generatedindex2. Its final logits use different greedy contexts.
The generator fake-quantizes with f32 scales but writes BF16 scales; record this
oracle caveat without modifying its thresholds or claiming the backend caused it.
That fixture remains excluded from GPU qualification.

Seven of12 guarded runs used. Native greedy PPL remains failed; independent human
PPL and op oracles pass, but no complete model acceptance or image/gateway/outputs/
colibench claim is made. Cache tuning is unrun. The remaining five runs are reserved
for a diagnosed numerical fix; GLM-5.3 preparation proceeds independently in roadmap
order, with the unresolved Qwen38 gate retained in the handover. All services unchanged.


## 22. GLM-5.3 / GLM-5.2 container qualification (in progress, 2026-11-08)

Both local checkpoints are glm_moe_dsa int4-gs64 containers for `colibri`.
The separate Glm5Next engine is not applicable. `glm_chain.h` is Vulkan-only;
SYCL uses the CUDA ABI for dense/attention operations and pinned expert groups,
with host orchestration. Do not infer a full-device chain from these results.

Unmodified upstream `make_glm_oracle.py --fmt4`, pinned torch2.13.0 and
transformers5.16.1, generated the tiny five-layer/16-routed-expert fixture offline.
CPU strict TF32/32 and greedy20/20 pass. Host-only basekit build is up to date,
colibri SHA4e3ca883913f7fc65d9730b663c7975d8103da12d977b1e984250edbbdc9702a.

Guard `codex-glm-gates-tiny-20261108T030707Z.log`, rc0/6s: chain244/default and
244/SG16, backend tests, GPU strict TF32/32 and greedy20/20 all pass. Explicit
synthetic PIN puts all16tiny experts on the GPU;94resident tensors, TF2groups/
128rows and greedy40groups/89experts/124rows prove actual work. Prompt grouped
MoE XMX is active. No upload fallback. Fixed348health counters unchanged,
SEL34c, clean kernel window, memoryOK. Tiny rates are not model benchmarks.

Native headers:75MoE layers,256experts each,21233664bytes/expert:407.69GB total.
Cap16/128 caches25.48/203.84GB before dense/scratch/KV. CAP_RAISE=0 and bounded
RAM are required for initial trials. Primary gateway leaves about29GB VRAM;
initial native expert budget8GB(decimal) plus <=11.6GB nonexpert payload keeps
headroom. GLM's native expert tier needs explicit PIN/sufficient history;
CUDA_EXPERT_GB alone does not prove expert residency. USAGE_SAVE=0 suppresses
STATS too, so native CPU route tracing will supply an explicit placement file.
Native CPU16 smoke PASS rc0/256s, peakRSS38.3GB. Measured route trace produces
PIN158f9a4284906d3ed3195dc02020a06f2ed73df50adafc771b3887247d52ec26.
No native PPL, gateway, outputs_check or colibench result yet.

Native GPU guard codex-glm53-native-gpu-smoke-20261108T031450Z.log, END03:16:39 rc0/84s.
376resident experts/7.98GB,1753resident tensors/18.15GB total,999groups/2566experts/5528rows;
actual expert VRAM calls2193. Zero upload fallback.22prompt/16generated,
prefill27.76s/decode24.37s,72%decode expert I/O. Fresh process only; uncontrolled
OS cache after CPU run, no speed comparison. Raw greedy output first differs
at character58 (zero based); human SCORE remains the quality gate.
Health348fixed counters unchanged, memoryOK, SEL34c, clean kernel window.
Next CPU human SCORE, then paired GPU SCORE; no image/gateway/colibench yet.

Run3 codex-glm53-score-gpu-20261108T032248Z.log stopped deliberately withSIGTERM
(rc143/49s, END03:24:11), not a timeout. At S338 dense matmul raised caught host
SYCL exception: range/offset does not fit in int. First tensor[28672,512],
then dense CPU fallbacks. GPU SCORE INVALID; no completed NLL or quality pass.
All fixed348health counters unchanged before/after, SEL34c, memoryOK, kernelclean.
Original CPU NLL2.769693496875 and GPU limit2.789181351353125 retained.
No unchanged command retry; read-only launch geometry diagnosis before any edit
and new op tests before any full model retry. Cache/image/gateway/CB held.

### 22.1 Native launch-range correction and SCORE execution proof (qualified)

The installed oneAPI2025.3 id_queries_fit_in_int.hpp checks the product of global
nd_range dimensions against INT_MAX. Legacy native matmul used O*S*256 work-items;
S338/O28672 exceeds that bound. This is a caught host nd_range exception, with
unchanged hardware counters, not evidence of insufficient VRAM.

COLI_SYCL_MM_ROWS now plans complete row batches: auto bounds the product, a
positive value caps it further,0 keeps the legacy diagnostic path (tiny only).
The failed shape becomes292+46 rows with rebased x/y pointers and unchanged
per-output reductions. Native matmul/expert MLP and async fallback/pipeline GEMM
share this planning; pipeline FP8 retains its original non-warp reduction.
No new kernel or compiler-limit relaxation. Host boundary and GPU row-boundary
regressions are added; no qualification claim until their guarded results land.
Rollback binaries and exact diagnosis are in models/glm/codex/pre-rowchunk/ and
rowchunk-review.md. Original CPU SCORE reference and limit remain unchanged.

Run4 codex-glm-rowchunk-gates-20261108T033725Z.log, END03:38:03 rc0/8s:
host planner tests and rebuilt CPUtinyTF32/32/greedy20/20PASS. GPU tiny row tests
atSG32/16, fullchain244/default+244/SG16, backenddefault+forcedMM_ROWS1,
strict GLMtinyTF32/32/greedy20/20, native338x512->28672 fmt0/4 allPASS.
Host/async/pipe results match independent two-column oracle exactly and each
other byte-for-byte; forced1/3-row boundaries match legacy0 on tiny cases.
The oversized native shape is tested only with automatic safe batches.
No device kernel arithmetic or compiler-limit change. Fixed348health unchanged,
SEL34c, clean kernel, memoryOK. Colibri883682a34815a4d532c994d9a88ce96fc3b5b27f6b8bbe50c0a2baa524962fdc.
Full-model quality still pending; original CPU reference and limit unchanged.

Run5 codex-glm53-score-rowchunk-20261108T033944Z.log, END03:44:03 guardrc1/226s.
Engine completed rc0/223.884s,320targets/logprob-884.672294 (raw NLL2.76460091875),
but validator rejected missing resident/group counters: run_score never calls
cuda_stats_print. No fallback, fixed348health unchanged/SEL34c/kernelclean/memoryOK.
This attempt remains UNQUALIFIED; raw numerical pass is not promotion. Add
PROF-gated stderr counters, rebuild/gate before repeating with new artifact names.
Original CPU reference/limit and run5 artifacts/driver retained. No cache/gateway.

Run6 codex-glm53-report-gates-score-20261108T034632Z.log, END03:49:29 guardrc1/150s.
All standard/tiny gates pass; CPU/GPU tiny SCORE PROF0/1 identical. Native engine
rc0/143.513s, raw320 NLL2.76460091875 identical to run5; resident1753/18.15GB.
Still UNQUALIFIED: group counter is zero because group_enabled requires S<=64
or CUDA_PIPE; native S338/PIPE0 takes individual MLP. Source confirms GPU critical
time increments only after successful individual MLP. Extend stderr report with
existing expert request count and GPU critical time; test both32/338-row tiny
SCORE before full run. No numerical/proof waiver. Health348same/SEL34c/clean/OK.

Run7 codex-glm53-execution-gates-score-20261108T035136Z.log, END03:54:12 rc0/122s:
all standard/tiny gates pass. Tiny SCORE32 (grouped) and338 (individual MLP) use
the same fixture/IDs, PROF0/1 exact on both CPU/GPU; stats stderr only. Native
320-target NLL2.76460091875 passes original CPU2.769693496875 and limit2.789181351353125
(-0.1839%). Same exact logprob as unqualified runs5/6; reporting did not alter math.
ActualGPU proof:1753resident tensors/18.15GB,314resident expert requests and
1.613462s successful individual GPU MLP critical time;197173CPU expert rows.
Zero grouped calls is expected at S338/PIPE0. Not a full-device-chain claim.
Engine115.421s/peakRSS38.927GB; elapsed times across runs are NOT a speed comparison
(OS cache uncontrolled). Fixed348health unchanged, SEL34c, clean kernel, memoryOK.
Source colibriSHA4ecefed6b9d55c45fb9dd4eb7b41ff46d05f71f07cc768b61d1042d67e304fcf.
Native quality now QUALIFIED; previous aborted/incomplete attempts remain recorded.
Next bounded cache A/B/A/B, then model-specific gateway/outputs/colibench. No deployment yet.

| Fixed human SCORE, 18 context + 320 targets | NLL (nats/token) | Status |
|---|---:|---|
| Original CPU, binary4e3ca883 | 2.769693496875 | Frozen reference |
| GPU run3, before launch fix | — | Stopped on host range exception/fallback |
| GPU run5, row batches | 2.764600918750 | Unqualified: reporting absent |
| GPU run6, resident/group report | 2.764600918750 | Unqualified: validator assumed grouped path |
| GPU run7, complete execution report | 2.764600918750 | Qualified, original limit2.789181351353125 |

The reporting change is gated by `PROF=1`; it reads existing counters/timers at
successful SCORE completion. It adds no work to the kernels or SCORE stdout.
At S338 with CUDA_PIPE off, the host selects individual expert MLP calls; their
GPU critical time advances after successful calls. Grouped-call counts are still
required for the tiny32-row and native raw22-row paths that actually group them.
Raw logs, validator failures, pinned health audits and hashes are under
`<ops>/work/models/glm/codex/`; run7-summary.json is the qualification record.

2026-11-08 04:05Z documentation correction from colibri.c14612-14625 and saved
logs: CPU native SCORE uses default float16 latent KV; CUDA/SYCL native attention
forces float32 KV. Both arms left KV16 unset. Earlier notes saying both use f16
were inaccurate; no runtime setting changed. The frozen CPU reference and original
NLL limit remain unchanged; the qualified test compares each backend's supported
default. Tiny REF/TF oracle mode forces CPU f32 too, as the source documents.
All GPU cache A/B/A/B cases use the same f32 path, so that comparison is unchanged.
Full-pin memory estimates must charge GPU f32 KV, including the DSA index reserve.

### Native cache A/B/A/B (run8, no cache128 performance win)

Guard codex-glm53-cache-abab-20261108T035635Z.log, END04:07:27 rc0/617s.
Fixed raw22-token prompt/64 generated tokens, same binary4ecefed6/PIN/8GBGPU,
RAM280GB/CTX1024/OMP32/PIPE0, all GPU cases use effective f32 KV.
Every case has identical output, GPU residency/group counts/VRAM calls and healthy
fixed348 counters; outer SEL34c/kernelclean/memoryOK. Source and input hashes saved.

| Case | Cap | Prefill s | Decode s / 64 tokens | Decode tok/s | Hit % | I/O wait s | Peak RSS GB |
|---|---:|---:|---:|---:|---:|---:|---:|
| A1 | 16 | 21.71 | 108.58 | 0.5894 | 49.3 | 78.9 | 38.482 |
| B1 | 128 | 37.09 | 197.25 | 0.3245 | 82.2 | 162.9 | 212.726 |
| A2 | 16 | 48.41 | 57.16 | 1.1197 | 49.3 | 32.8 | 38.475 |
| B2 | 128 | 14.32 | 66.23 | 0.9663 | 82.2 | 38.4 | 212.744 |

No keep: candidate decode range overlaps baseline and includes slower samples;
increased hit rate did not establish a latency win. Retain cap16 for the LRU
configuration. Each process starts a fresh LRU; OS/page cache is uncontrolled,
and the RAID check remains active. These are not steady-service/colibench numbers.
Next separately qualify a bounded all-host-pin configuration for zero expert I/O,
with unchanged8GBGPU placement, original quality limits and actual memory proof.
No image/gateway/colibench change has been made. See cache/verdict.json and
pin-memory-review.md/fullpin-memory.json; the latter is still an unrun proposal.

### 22.2 All-expert host pin, qualified zero-expert-I/O configuration

Run9 codex-glm53-fullpin-qual-20261108T041044Z.log, END04:21:49 rc0/630s.
CPU tiny all16RAM and GPU mixed4VRAM+12RAM strict TF32/32 and greedy20/20 pass;
chain244 at default/SG16 and backend gates pass before the native runs.
PIN_GB=all/PIN_FILL=1 retains the frozen GPU prefix376 and pins18824experts in
host RAM. Same8GB GPU allowance, cap16, RAM450GB, CTX1024, OMP32, PIPE0,
no history/speculation. Codeca5d707e, colibri4ecefed6; no kernel/default change.

| Check | Result |
|---|---|
| Native human SCORE | NLL2.76460091875, exact qualified GPU logprob-884.672294; original CPU2.769693496875/limit2.789181351353125 retained |
| GPU SCORE execution | 1753tensors/18.15GB,314resident requests,1.604789s successful individual MLP,197173CPUrows |
| Raw22+64 output | Byte-identical to cacheA1, same3314GPUgroups/9762rows/6427decodeVRAMcalls |
| Raw prefill / decode | 7.79s /25.45s for64tokens (2.514735tok/s) |
| Raw expert I/O | 100%hit,37800pin/0LRU/0loads,0.000GB fetched,0.0s service/wait |
| Peak process RSS | SCORE411.861GB; raw411.716GB, below450GB |
| Host pin startup only | SCORE164s; raw279s,399.7GB host experts |
| Safety | Fixed348counters unchanged, memoryOK, SEL34c, clean kernel; sampled engineVmSwap0, minimum sampled hostavailable87.762GB |

This qualifies a zero-expert-I/O service configuration. One raw sample is not
an interleaved causal speed win over the cache experiment; startup and OS/RAID
I/O remain separate. Host pin does not use mlock. Global swap was already nearly
full; no VM/swap/NUMA/RAID settings changed. Full artifacts and offline audit:
models/glm/codex/fullpin/verdict.json and fullpin-qualified.json.
Next is a separately tagged image, model-specific CPU outputs reference and
GPU API/colibench qualification. No GLM service or colibench result yet.

### 22.3 GLM5.3 image and API qualification

Separate image `colibri-sycl:glm53`, id1dcca7bb6b4f, source17457d9a and
codeca5d707e, built without GPU nodes. Its colibri SHA256
4ecefed6b9d55c45fb9dd4eb7b41ff46d05f71f07cc768b61d1042d67e304fcf is byte-identical
to the native full-pin qualification. Native SCORE was therefore not repeated;
the original CPU reference and NLL limit remain unchanged. The image itself
passed chain245/default and245/SG16, backend correctness, and the strict tiny
teacher-forcing32/32 and greedy20/20 tests before native startup.

Gate-count correction: run9 had244 intact `ok` lines because concurrent SYCL
stderr split one prefix into `o` and `k`. Removing complete diagnostic lines
reconstructs245 identical successful check names at both widths in run9 and the
image logs, plus two unsupported hcmix skips. No test or threshold changed, and
no GPU test was repeated for this correction. Evidence:
`models/glm/codex/chain-count-normalization.json`.

The actual CLI serves `--gpu 0` on temporary port8002, with the qualified
376GPU+18824RAM expert placement, GPUexpert8GB, RAM450GB/cap16, CTX8192,
KV_SLOTS1, OMP32, PIPE0, no history or speculation. The conservative reviewed
host bound is446.600GB. The CPU outputs reference used the same image and
three stock prompts with `--gpu none`, all19200 experts in RAM, cap1 and the
same context/thread settings; its container stopped before GPU startup.
CPU f16 and GPU f32 KV remain their supported defaults.

| Check | CPU reference | GPU candidate |
|---|---|---|
| CLI startup, including pin load | 322.848s | 312.128s |
| Peak sampled RSS | 419.615GB | 411.787GB |
| Sampled engine swap | 0 | 0 |
| Smoke request | separate CPU reference below | 17prompt+64completion,33.502s; decode GPUcritical0.207s |
| Sky: completion tokens / HTTP seconds | 88 /28.595 | 92 /33.423 |
| Fibonacci: completion tokens / HTTP seconds | 160 /47.062 | 160 /60.827 |
| Capitals: completion tokens / HTTP seconds | 160 /45.034 | 160 /60.915 |

These are output checks, not a matched performance experiment. The GPU sky
request reused its smoke prompt's KV prefix; other prompts shared three prefix
tokens. Startup and request timings must not be interpreted as CPU speedup.

Stock outputs_check versus `reference-glm53-cpu-fullpin.json`:

| Prompt | First differing character, including five-character separator | GPU / CPU characters | Sane heuristic |
|---|---:|---:|---:|
| Sky | 5 (first content character) | 471 /474 | 1.0 |
| Fibonacci | identical638 | 638 /638 | 1.0 |
| Capitals | 153 (content148) | 627 /661 | 1.0 |

All three are readable and on topic; this does not certify factual accuracy.
The fixed160-token cap truncates the Fibonacci docstring and capitals answers;
they are comparison artifacts, not complete solutions. Preserve the early sky
divergence. Native held-out SCORE, not string identity or this heuristic,
remains the numerical quality gate. The CPU reference SHA256 is
0cb353d3c66cd60feb219b817c846856daff07a36a3d142784fd09ead0cc4897.

Run10 `codex-glm53-image-gateway-20261108T043915Z.log` ended04:45:42 rc0/347s.
Run11 `codex-glm53-api-outputs-20261108T044616Z.log` ended04:49:21 rc0/156s.
Both offline audits pass:348 fixed health counters unchanged, memoryOK,
SEL34c, no kernel lines, RSS below450GB and sampled engine swap0. Every decode
profile shows100%hits/zero expert reads and positive GPU critical time.
Artifacts: `gpu-api-{startup,outputs}-verdict.json`, `service-{cpu,gpu}/`,
`outputs/`, image manifest/build log and exact per-request bodies/responses.
Production CPU8000/GPU8001 and accepted image15deeb14b462 remain unchanged.

Two scope findings must accompany subsequent benchmarks:

- All141 GLM5.3 safetensors headers contain zero indexer weights. The existing
  engine detection disables DSA; all original CPU/GPU logs agree. This is full
  attention, including beyond2048 positions, not sparse DSA qualification.
  The metadata-only scan read14.226MB/116915 tensor entries, not all weight bytes.
- Native colibri multiplexed API serving calls `prof_base` after prefill
  (`colibri.c11723/11751`). Its profile wall, phase, GPU and I/O counters cover
  decode only. A one-token cap has zero forwards and an empty profile; those
  zeros cannot measure prefill GPU work or I/O. HTTP wall and TTFT include
  prefill. The local benchmark collector was corrected before any colibench
  result to record this limitation and require positive GPU time for decode.

Colibench remains pending, with the final12th guarded slot reserved. Planned
Light workloads, nonce186e76efdedad1a9 and token counts were frozen before any
benchmark request. This will establish a first GLM5.3 baseline, not a causal
speed win over CPU or the uncontrolled OS-cache experiments.

### 22.4 GLM5.3 Light aborted by the safety policy

Run `20261108-045055`, label `gpu-sycl-card1-glm53`, is **ABORTED and is not
an accepted baseline**. Its twelfth and final workload guard started04:50:36
and ended06:22:53 UTC on2026-11-08, rc1/5503seconds. Thirteen requests completed
and passed their individual checks (2warmups,11measured); request14, the second
long prefill, was interrupted. Requests15–20 never started. The frozen20-task
workload was not shortened, completed by inference, or retried. Partial
`results.json`, `runs.csv` and `summary.csv` exist; their presence does not mean
the benchmark completed. `complete.json`, `validated.json` and the successful
`verdict.json` are absent. `colibench/abort-verdict.json` explicitly records failure.

At06:22:24 the watcher recorded four systemd/containerd warnings. The guard
subsequently emitted `KERNEL LINES ...: 4`, which requires an immediate stop.
The warnings say that leftover processes usually indicate an unclean previous
termination or service implementation deficiencies. Their case-insensitive
`CAT` match is the substring in `indiCATes`; the journal identifies systemd
as their source and kernel as their transport. The captured journal also
records systemd reexecution requested by `apt-daily-upgrade.service`, followed
by containerd stopping and starting. Dpkg records the systemd update from
255.4-1ubuntu8.17 to255.4-1ubuntu8.18. Host maintenance occurred during the run;
this agent did not initiate it. Neither the monitor expression nor its alerts
were changed to permit continuation.

The controller requested termination of `gpu-glm53-gw`; Docker reported a
10-second kill delay during the containerd restart, then used direct SIGKILL
at06:22:36. Candidate gateway525990/engine526825 and the container were gone
in the post-abort observation. At06:24:15, AFTER the guard had ended, the kernel
also warned that `drm_fb_helper_damage_work` hogged CPU for more than13333us
64times. This is retained as a separate later DRM workqueue warning; its cause
has not been assigned to the benchmark, maintenance, or hardware.

SEL remained34c. Read-only sysfs observations at06:23:50 and06:26:08 found
all348 fixed card1 counters unchanged and memory healthOK, including after the
later DRM warning. These observations do not replace a successful guard audit
or waive the stop rule. No new hardware fault is established by these messages.
The watcher remains active and its alert file is preserved. No GPU probe,
model restart, new benchmark or further roadmap execution followed the abort.

Peak sampled candidate memory across startup/API/benchmark was413.950GB<450GB,
engine swap0; the last sample was06:22:21.198452 UTC. Post-abort available host
memory was497.453GB. Production CPU8000 (gateway151265/engine151277) and accepted
Qwen GPU8001 (gateway325511/engine326238, image15deeb14b462) remained running.
The accepted GPU log SHA remainedf1905dda39b2d8b59155e2ad08074ea259038a73ac114badb990ca8d4c87167f,
supporting no additional logged inference on that service. No inference was
issued to test the primary after the safety stop.

The native SCORE and image/API results in22.1–22.3 remain separately recorded;
this aborted benchmark does not complete GLM deployment qualification.
All completed decode windows had positive GPU expert time and zero expert I/O.
Native mux counters exclude prefill; one-token requests have empty decode
profiles, not measured prefill GPU-time/I/O. Full attention/no indexer weights,
GPUf32/CPUf16 supported KV defaults, and short-SCORE quality scope are unchanged.

The following are individual saved measurements from the aborted attempt,
not final means, a matched comparison, or a performance acceptance result.

| Request | Workload | Repetition | Prompt / completion tokens | HTTP wall s | HTTP TTFT s | Decode tok/s |
|---:|---|---:|---:|---:|---:|---:|
| 1 | warmup | 1 | 25 / 3 | 7.736 | 6.907 | n/a |
| 2 | warmup | 2 | 25 / 4 | 3.983 | 2.748 | 7.459 |
| 3 | decode_long | 1 | 42 / 1024 | 402.622 | 8.137 | 2.593 |
| 4 | prefill_long | 1 | 3050 / 1 | 1168.666 | 1168.636 | n/a |
| 5 | decode_short | 1 | 42 / 128 | 52.361 | 8.111 | 2.872 |
| 6 | mixed_short | 1 | 31 / 256 | 100.776 | 4.586 | 2.652 |
| 7 | prefill_short | 1 | 26 / 1 | 3.082 | 3.051 | n/a |
| 8 | mixed_long | 1 | 3056 / 512 | 1450.744 | 1181.307 | 1.897 |
| 9 | decode_long | 2 | 42 / 1024 | 395.818 | 7.373 | 2.634 |
| 10 | prefill_short | 2 | 26 / 1 | 3.553 | 3.523 | n/a |
| 11 | decode_short | 2 | 42 / 128 | 55.628 | 8.038 | 2.670 |
| 12 | mixed_short | 2 | 31 / 256 | 100.757 | 4.561 | 2.651 |
| 13 | mixed_long | 2 | 3056 / 512 | 1452.414 | 1180.977 | 1.883 |

Evidence root: `<ops>/work/models/glm/codex/abort-20261108T062224Z/`, including
journal/dpkg extracts, preserved watcher lines, before/after observations,
snapshot paths and an SHA256 evidence manifest. Guard log:
`<ops>/logs/codex-glm53-colibench-light-20261108T045036Z.log`.
Guard kernel snapshot: `kernel-codex-glm53-colibench-light-20261108T062246Z.txt`;
post snapshot: `post-codex-glm53-colibench-light-20261108T062238Z.txt`.
Exact requests and checks remain in the phase `colibench/` directory.

Budget is12/12 workload guards. The planned lifecycle guard was cancelled
because the safety controller already stopped the temporary service; no extra
guard ran. The successful audit/report/stop/closeout sequence must not be used
for this aborted run. GLM52 is prepared but unrun0/8; DeepSeek has read-only
preparation only; Qwen38 remains held on its earlier numerical gate. Further
GPU work is stopped pending review of maintenance, monitor matching and the
later DRM warning, followed by an explicitly revised continuation plan.
Do not clear alerts or retry the same command automatically.

### 22.5 GLM5.3 fresh native/API rerun after warning-policy review

The user requested a fresh rerun and explicitly permitted warnings. Previous
Light20261108-045055 remains aborted, with its raw evidence and verdict unchanged
(22.4). New artifacts are in `<ops>/work/models/glm/rerun1/`. No engine arithmetic
changed: immutable image1dcca7bb6b4f, colibri4ecefed6, codeca5d707e.

Fresh chain gates passed245/default and245/SG16; backend correctness passed.
Fresh tiny CPU and GPU both passed strict32/32 teacher-forced positions and
20/20 greedy tokens. The GPU tiny uses4VRAM+12RAM experts, exercising mixed
placement before the native model. Native SCORE was executed again, not waived
on binary identity. Its original CPU reference/PIN/input bytes were reused
unchanged, with provenance.json recording their origin and hashes.

| Native quality | nats/token | Result |
|---|---:|---|
| Original CPU reference | 2.769693496875 | unchanged reference |
| Fresh GPU SCORE,320 tokens | 2.764600918750 | PASS, exact prior GPU SCORE |
| Original acceptance limit | 2.789181351353 | unchanged |

Fresh native work recorded314 successful VRAM expert
requests and1.602401s GPU critical time. It placed
376VRAM+18824RAM experts, with1753 resident tensors and
18.15GB reported resident VRAM. API startup uses actual
CLI--gpu0, RAM450GB/cap16/CTX8192/slot1, OMP32/PIPE0/DRAFT0/lookup0/fullpin.
CPU8000 and accepted QwenGPU8001 remain running; the candidate is isolated8002.

Fresh API smoke64 and the three stock outputs_check prompts completed.
Comparison uses the original CPU API reference, not a newly generated CPU run.
Character offsets include the5-character reasoning/content separator.

| Stock prompt | Full-text first divergence | GPU prompt/output tokens |
|---|---|---|
| Sky | 5 | 17/92 |
| Fibonacci | identical (638 chars) | 25/160 |
| Capitals | 153 | 17/160 |

Outputs are preserved verbatim with actual caps and usage. Readability and
numerical gates do not establish factual correctness; capped answers may be
incomplete. API native profiles cover decode only: all four requests show
positive GPU expert time and zero expert I/O in that window. They do not
measure prefill GPU time/I/O. Full attention (no indexer weights), supported
GPUf32/CPUf16 KV defaults; no sparse-DSA or full-device-chain claim.

Peak sampled RSS411.914GB,
sampled engine swap0. Both guard audits passed fixed348 counters/memoryOK/SEL34c;
new-window warning classifications are retained in their audit JSON. The
watcher retains5 historical lines, reviewed under the user's
warning instruction. Fatal/unknown messages, device loss/OOR, changed counters,
new SEL and timeouts still stop. No watcher alerts or old verdicts were cleared.
Enabled apt timers were temporarily held only after transactions finished,
with original state saved and automatic8h restoration scheduled. No reboot or
package transaction interruption. Guard logs: `<ops>/logs/codex-glm53-r1-image-gateway-20261108T080048Z.log`
and `<ops>/logs/codex-glm53-r1-api-outputs-20261108T081457Z.log`. Light benchmarking is still pending.

### 22.6 GLM5.3 completed fresh Light rerun

Fresh run20261108-081932, label `gpu-sycl-card1-glm53-rerun1`, completed all20 requests
(2warmups and18 measured, three repetitions of six stock workloads).
Offline guard/request/provenance audit passed. The original aborted run stays rejected. This is the first complete model baseline,
not an interleaved speed comparison against CPU, Qwen, or the earlier LRU trial.

| Workload | Prompt tokens | Completion tokens, mean [min,max] | HTTP TTFT s | Decode tok/s |
|---|---:|---:|---|---|
| prefill_short | 26 | 1.0 [1, 1] | 3.276 ± 0.272 [2.963, 3.454] | n/a (one-token cap) |
| prefill_long | 3050 | 1.0 [1, 1] | 1184.196 ± 1.226 [1183.377, 1185.606] | n/a (one-token cap) |
| decode_short | 42 | 128.0 [128, 128] | 7.924 ± 0.427 [7.438, 8.237] | 2.825 ± 0.032 [2.793, 2.858] |
| decode_long | 42 | 1024.0 [1024, 1024] | 8.135 ± 0.047 [8.093, 8.186] | 2.603 ± 0.030 [2.568, 2.624] |
| mixed_short | 31 | 256.0 [256, 256] | 4.759 ± 0.607 [4.393, 5.460] | 2.767 ± 0.069 [2.719, 2.846] |
| mixed_long | 3056 | 512.0 [512, 512] | 1188.241 ± 8.476 [1178.584, 1194.448] | 1.869 ± 0.009 [1.859, 1.877] |

Entries are mean ± sample standard deviation [minimum,maximum]. TTFT and
decode throughput are the stock colibench HTTP measurements. Exact20 task
bytes, nonce186e76efdedad1a9, caps and prompt counts were frozen before results;
actual counts match. The maximum planned prompt plus cap was3568<CTX8192.

Image1dcca7bb6b4f/source17457d9a, codeca5d707e, colibri4ecefed6; metadata source
daa2c68b includes documentation only. Actual CLI--gpu0,
376VRAM+18824RAM experts, GPUexpert8GB/denseGPU/hostrelease, RAM450GB/cap16,
CTX8192/slot1/OMP32/PIPE0/DRAFT0/lookup0, all-pin and no history changes.
Accepted Qwen GPU8001 remained idle; CPU8000 was unchanged. No build or second
model inference overlapped. Read-only host observations and maintenance timer
hold/restoration records are retained with the new attempt.

Native mux profile counters start AFTER prefill. All decode windows recorded
positive successful GPU expert time and zero expert I/O; the six one-token
prefill requests have empty decode profiles. Their GPU time and I/O are not
measured by these counters. Completed prefill was checked on the same qualified
resident image with no fallback. The HTTP TTFT includes prefill; native profile
wall/phase totals exclude it. All-pin startup remains separate.

Peak sampled engine RSS413.951GB<450; sampled engine
swap0, minimum sampled host available85.104GB.
Fixed348 counters unchanged, memoryOK, SEL34c. Window warnings are classified
in the outer audit; raw historical alerts remain. The user explicitly permitted
nonfatal warnings; fatal/unknown/device/timeout/counter/SEL stops remain. These memory
extrema cover startup/API/benchmark samples. Card1-only sysman telemetry;
no hardware sampling profiler or other GPU was used.

Guard telemetry retained9625 timestamped samples,
plus13342 TEELIB and13342
MEI device-busy diagnostic lines. The same diagnostics occurred in earlier
healthy benchmark guards. They are preserved in raw output and excluded from
numeric samples. Concurrent safe sysman clients are a possible cause, not a
demonstrated one. Zero-frequency readings also occurred in earlier healthy
runs; they alone do not establish a stalled engine. Counter, kernel-watch,
SEL and request audits determine this run's recorded health outcome.

Fresh native short human SCORE passed and exactly matched the prior GPU result;
its original CPU2.769693496875 reference and limit2.789181351353125 were unchanged. Output differences and capped answers are retained
in22.5. Full attention (no indexer weights), GPUf32/CPUf16 supported KV defaults;
this benchmark is not long-context quality or sparse DSA qualification.

Raw run artifacts: `colibench/data/runs/20261108-081932/`. Exact per-request records,
hashes, provenance and offline verdict: `<ops>/work/models/glm/rerun1/colibench/`.
Guard `<ops>/logs/codex-glm53-r1-colibench-light-20261108T081918Z.log`. Three workload guards have completed in
this new4-planned/5-maximum phase; shutdown is the fourth planned guard. Temporary-service shutdown and memory
reclamation must precede GLM5.2. No GLM deployment replaced the accepted primary.

### 22.7 GLM5.3 rerun lifecycle closeout

The fresh qualification phase closed successfully at 2026-11-08T11:03:20Z.
Planned guard 4, `codex-glm53-r1-service-stop`, ended at 11:03:06Z with rc=0.
Its saved shutdown result records the temporary container gone and 500.410 GB
of host memory available. The 348 fixed counters remained unchanged, memory
health was OK, SEL stayed at 34c, and there were no new kernel-window lines.
The original apt timers were restored before phase_closed was published.
Restoration evidence is hashed in the actual
`<ops>/work/models/glm/rerun1/closeout.json`. Four guards were used: three workload
runs and one lifecycle shutdown. The fifth reserve was unused. Both primary
services remained running, and the accepted GPU log hash was reverified
unchanged after shutdown. No GLM service replaced the accepted Qwen gateway.

The benchmark report is committed as b002f8ef. Complete native, API and Light
evidence remains in sections 22.5–22.6 and the rerun1 directory. The former
aborted attempt remains failed, with its closeout absent; no partial results
were joined. GLM5.2 is the next independent model qualification and requires
its own CPU reference.

## 23. GLM5.2 native qualification, 2026-11-08

The independent GLM5.2 ladder passes tiny/op, native execution, numerical
comparison and full host-pin qualification on card1. API and Light remain
separate later gates. Image1dcca7bb6b4f and colibri4ecefed6 are reused unchanged
from the qualified generic GLM backend (codeca5d707e); no new kernel, arithmetic
change or model-quality waiver. The accepted CPU8000 and Qwen GPU8001 stay up.

Local checkpoint mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp, revision6bbb01ed:
142 shard headers rechecked before full pin,116915 main tensors/419.282GB.
Header verification covers names/dtypes/shapes/offset bounds, not a fresh full
weight hash. The9.959GB MTP layer78 is excluded with **MTP0 AND DRAFT0**.
The native logs confirm78layers/256experts and MTPabsent. No indexer weights:
full attention, no sparse DSA or speculative-decode qualification.

Fresh CPU and GPU tiny tests retain the upstream TF32/32 and greedy20/20
oracle. GPU placement is4VRAM+12RAM; the fullpin guard repeats the tiny gate
before native loads. Image chain tests pass245 checks at default SG and245
at SG16; backend q8/q4/q2/f32/e8 correctness passes. No fallback accepted.

Own CPU raw22+16 completes and supplies a new ROUTE_TRACE-only PIN:
2775 rows/22200 selections,
7749 unique experts; SHA8424f4f882517f01ec1a0c696a78c58fabe230e0a8c48283ba778bb381c71cef.
The inherited57600-selection .coli_usage was discovered after that CPU smoke
started, then hashed/reviewed and explicitly frozen for later runs. USAGE_SAVE0
prevents writes, not reads. AUTOPIN0 disables history placement; explicit PIN
comes only from the fresh trace and REPIN0 stays fixed. History is unchanged,
mounted read-only. The source review found no history-fed router arithmetic
in this configuration; this is not a history-on/off performance experiment.
See usage-history-review.json; the late discovery is retained as provenance.

| Own fixed human SCORE,18 context +320 targets | NLL nats/token | Result |
|---|---:|---|
| CPU, supported default f16 KV | 2.846801415625 | Immutable independent reference |
| GPU, limited cache/default f32 KV | 2.827951843750 | Pass |
| GPU, all experts pinned/default f32 KV | 2.827951843750 | Exact same GPU logprob-904.944590 |
| Original ceiling CPU*1.007+0.0001 | 2.866829025534 | Unchanged |

The input IDs/tokenizer/corpus are frozen independently of model outputs.
No GLM53 score, PIN or output reference is reused. Both GPU SCORE runs prove
1753 resident tensors/18.15GB and positive successful
GPU expert execution. Limited cache:314 VRAM requests/
1.535256s GPUcritical; fullpin:
314 requests/1.497600s.
At338 rows with PIPE0, individual MLP calls are expected; grouped-call counts
are required for raw decode, not this SCORE path. This short comparison does
not qualify long-context quality or imply improved answer accuracy.

RAM80 to450 enables the full host suffix while cap16/CTX1024/OMP32,
CUDA_EXPERT_GB8/CUDA_RESERVE_GB2/CUDA_RELEASE_HOST1, PIPE0, MTP0/DRAFT0,
REPIN0 and the GPU prefix stay fixed. PIN_GB0 places376VRAM+1RAM because the
existing prefix estimate includes one-device slack; it is not literally
zero RAM pins. PIN_GBall/FILL1 places376VRAM+18824RAM. Fullpin raw output
is byte-identical to the bounded GPU output, with the same
3363 GPU groups/8996 grouped rows/
5994 decode VRAM calls.

| GPU raw22 + 64 | Limited cache | All experts pinned |
|---|---:|---:|
| Prefill seconds | 12.08 | 7.53 |
| Decode seconds | 158.13 | 25.14 |
| Decode tokens/s | 0.4047 | 2.5457 |
| Expert hit percent | 48.3 | 100.0 |
| Decode expert loads | 19558 | 0 |
| Expert I/O service seconds | 263.0 | 0.0 |
| Expert I/O wait seconds | 122.2 | 0.0 |
| Peak sampled RSS GB | 38.502 | 411.764 |

These single samples qualify zero expert I/O, not a matched speedup. OS/page
cache and RAID activity were not controlled. Pin startup is separate:
SCORE188s, raw281s, as rounded in the engine log. Fullpin SCORE
peak sampled RSS411.927GB; both native
fullpin loads remain below450GB and sample zero engine swap. Conservative
prelaunch estimate444.187GB, with at least480GB available before each load;
MTP payload excluded and GPUf32 KV/index/workspace conservatively charged.
No VM, swap, NUMA, overcommit or RAID settings changed.

All four guard post-snapshot/offline audits pass: fixed348 error counters
unchanged, memoryOK, SEL34c, no new kernel-window lines. Historical reviewed
warnings remain preserved; timeouts/device failures/new health faults still
stop. The temporary maintenance hold is recorded and automatically expires;
original timers must be restored at phase closeout.

- Guard1: `codex-glm52-gates-20261108T110722Z.log`.
- Guard2: `codex-glm52-raw-20261108T111452Z.log`.
- Guard3: `codex-glm52-score-20261108T112303Z.log`.
- Guard4: `codex-glm52-fullpin-20261108T112747Z.log`.

Evidence: `<ops>/work/models/glm52/codex/` contains pre-run1–4.json, CPU launch
manifests, exact commands/helper/input hashes, raw logs, memory samples,
gates-qualified/raw-qualified/native-qualified/fullpin-qualified.json,
memory-review/fullpin-launch-review and history/placement reviews. Full phase
budget is8 guards (7 workload +1 lifecycle),4 used at this native milestone.
Next: independent CPU API reference, GPU API/outputs/Light and guarded shutdown.

### 23.1 GLM5.2 independent API qualification

The actual CLI served GLM5.2 on temporary loopback port 8002, first with
`--gpu none` for its own CPU reference, then `--gpu 0` on card1. The CPU
container stopped and at least 480 GB of host memory was reclaimed before
GPU startup. Accepted primary services on 8000/8001 remain unchanged.

Image 1dcca7bb6b4f and binary 4ecefed6 are unchanged from the independent
native qualification in section 23. Fresh GPU image gates pass 245 chain
checks at default SG and 245 at SG16, backend correctness, and strict tiny
TF 32/32 and greedy 20/20. The image native check verifies byte identity
against the already qualified GLM5.2 full-pin score; it does not rescore.

Both services use CTX8192, one KV slot, OMP32, RAM450 GB, MTP0/DRAFT0,
PIPE0, fixed own PIN, PIN_GBall/FILL1, REPIN0 and USAGE_SAVE0. CPU uses cap1,
all 19,200 host experts and supported f16 KV. GPU uses cap16, 376 VRAM plus
18,824 host experts, the fixed 8 GB expert allowance, resident dense tensors
and supported f32 KV. This is a hybrid engine, not a full device chain.
The read-only history input and its provenance remain as recorded in section 23.

GPU startup includes a 64-token API smoke: 17
prompt tokens, 64 generated, positive successful
decode GPU time 0.305s and zero decode expert I/O.
All three stock outputs_check prompts retain max160/temp0, identical request
bodies, saved responses and their own CPU reference SHA 514d49d778738ecee3ebcd432b8ebaa49b4c9a2fc60a8438ce0c653cf932b5f5.

| Prompt | CPU prompt/output tokens | GPU prompt/output tokens | First full-text difference | First content difference | GPU finish reason |
|---|---:|---:|---|---|---|
| Sky | 17/77 | 17/74 | 206 | 201 | stop |
| Fibonacci | 25/160 | 25/160 | 404 | 399 | length |
| Capitals | 17/160 | 17/160 | identical | identical | length |

Differences are zero-based character positions, including an earlier end of
text. Full text includes the reasoning/content separator used by outputs_check;
content differences compare the response's content field directly. Exact
text and finish reasons are retained; a cap-limited answer is not assumed
complete. Review notes: All six saved responses were read. Both sky answers are readable three-sentence explanations with different wording. Both Fibonacci responses hit max160 inside their docstrings, before a function body; neither was executed or accepted as a complete coding answer. The capitals content is identical and contains five readable entries, but both finish_reason values are length. The geographical and historical claims were not independently fact-checked. This review covers readable transport/output behavior and truncation, not task accuracy.
The character sanity heuristic is not factual or task-accuracy validation.
The unchanged native numerical gate remains GPU 2.827951843750
versus CPU 2.846801415625, ceiling 2.866829025534.

Every CPU API decode profile has zero GPU time; every GPU API decode profile
has positive successful GPU time. All have 100% expert hits and zero expert
disk loads/wait. These native mux counters begin after prefill; HTTP request
time includes prefill. No measured prefill GPU time or I/O is inferred from them.
Peak sampled CPU/GPU API RSS: 419.615/
411.785 GB, below 450 GB, with sampled engine swap 0.
Startup and these few requests do not establish a CPU/GPU performance comparison.

Both GPU guard audits pass with fixed 348 counters unchanged, memory OK,
SEL34c and reviewed kernel warnings. Expected container bind/auth warnings
are preserved; the host port is published on 127.0.0.1 only. Exact commands,
hashes, warning reviews, raw logs and first-divergence evidence are under
`<ops>/work/models/glm52/codex/`. Guards:
`codex-glm52-image-gateway-20261108T115128Z.log` and
`codex-glm52-api-outputs-20261108T120305Z.log`. Six of eight planned guards are now
complete; Light and lifecycle shutdown remain. No GLM deployment replaced
the accepted primary services.

### 23.2 GLM5.2 first Light benchmark

Run20261108-120756, label `gpu-sycl-card1-glm52`, completed all20 requests
(2warmups and18 measured, three repetitions of six stock workloads).
Offline guard/request/provenance audit passed. This is a first model baseline,
not an interleaved speed comparison against CPU, Qwen, or GLM5.3.

| Workload | Prompt tokens | Completion tokens, mean [min,max] | HTTP TTFT s | Decode tok/s |
|---|---:|---:|---|---|
| prefill_short | 26 | 1.0 [1, 1] | 3.543 ± 0.306 [3.256, 3.865] | n/a (one-token cap) |
| prefill_long | 3050 | 1.0 [1, 1] | 1167.282 ± 4.690 [1163.442, 1172.509] | n/a (one-token cap) |
| decode_short | 42 | 128.0 [128, 128] | 8.203 ± 0.036 [8.161, 8.228] | 2.769 ± 0.068 [2.691, 2.811] |
| decode_long | 42 | 1024.0 [1024, 1024] | 7.812 ± 0.527 [7.331, 8.376] | 2.596 ± 0.017 [2.578, 2.612] |
| mixed_short | 31 | 256.0 [256, 256] | 4.742 ± 0.297 [4.470, 5.059] | 2.793 ± 0.006 [2.787, 2.798] |
| mixed_long | 3056 | 512.0 [512, 512] | 1170.021 ± 6.475 [1162.934, 1175.627] | 1.855 ± 0.008 [1.847, 1.863] |

Entries are mean ± sample standard deviation [minimum,maximum]. TTFT and
decode throughput are the stock colibench HTTP measurements. Exact20 task
bytes, nonce186e76efdedad1a9, caps and prompt counts were frozen before results;
actual counts match. The maximum planned prompt plus cap was3568<CTX8192.

Image1dcca7bb6b4f/source17457d9a, codeca5d707e, colibri4ecefed6; metadata source
bc77f7eb includes documentation only. Actual CLI--gpu0,
376VRAM+18824RAM experts, GPUexpert8GB/denseGPU/hostrelease, RAM450GB/cap16,
CTX8192/slot1/OMP32/PIPE0/MTP0/DRAFT0/lookup0, all-pin and no history changes.
Accepted Qwen GPU8001 remained idle; CPU8000 was unchanged. No build or second
model inference overlapped. The pre-run /proc/mdstat contents are preserved
in the phase's colibench/start.json.

Native mux profile counters start AFTER prefill. All decode windows recorded
positive successful GPU expert time and zero expert I/O; the six one-token
prefill requests have empty decode profiles. Their GPU time and I/O are not
measured by these counters. Completed prefill was checked on the same qualified
resident image with no fallback. The HTTP TTFT includes prefill; native profile
wall/phase totals exclude it. All-pin startup remains separate.

Peak sampled engine RSS413.953GB<450; sampled engine
swap0, minimum sampled host available84.693GB.
Fixed348 counters unchanged, memoryOK, SEL34c; kernel warnings reviewed under
the explicit user authorization, raw alerts preserved (outer audit JSON). These memory
extrema cover startup/API/benchmark samples. Card1-only sysman telemetry;
no hardware sampling profiler or other GPU was used.

Guard telemetry retained9530 timestamped samples,
plus18123 TEELIB and18123
MEI device-busy diagnostic lines. The same diagnostics occurred in earlier
healthy benchmark guards. They are preserved in raw output and excluded from
numeric samples. Concurrent safe sysman clients are a possible cause, not a
demonstrated one. Zero-frequency readings also occurred in earlier healthy
runs; they alone do not establish a stalled engine. Counter, kernel-watch,
SEL and request audits determine this run's recorded health outcome.

Native short human SCORE remains2.827951843750 versus its own original
CPU2.846801415625 and limit2.866829025534. Output differences and capped answers are retained
in23.1. Full attention (no indexer weights), GPUf32/CPUf16 supported KV defaults;
this benchmark is not long-context quality or sparse DSA qualification.

Raw run artifacts: `colibench/data/runs/20261108-120756/`. Exact per-request records,
hashes, provenance and offline verdict: `<ops>/work/models/glm52/codex/colibench/`.
Guard `codex-glm52-colibench-light-20261108T120747Z.log`. Seven workload guards are planned; the eighth
guard is reserved for lifecycle shutdown, not a performance retry. Temporary-service shutdown and memory
reclamation must precede the Qwen38 revisit and later DeepSeek work.
No GLM deployment replaced the accepted primary.

### 23.3 GLM5.2 phase closeout

The temporary `gpu-glm52-gw` service stopped after the successful Light report
commit `84ad63b5`. Lifecycle guard
`codex-glm52-service-stop-20261108T145000Z.log` ended at14:50:46 UTC with rc0
(12s); the actual driver session60046 also returned0. The container was gone
and host available memory was500.146GB.
Its post-snapshot audit passed: all348 fixed counters unchanged, memoryOK,
SEL34c, and no new kernel-window lines. This eighth guard performed shutdown
only, with no inference or performance retry. The phase used seven workload
guards and one lifecycle guard, exactly its budget of eight.

The closeout helper restored the original enabled apt timers successfully
before publishing `phase_closed` at2026-11-08T14:51:07.314598+00:00. Subsequent read-only checks
confirmed both timers active/enabled, both apt services inactive, and the
transient restoration timer inactive. No package transaction was interrupted.
The accepted Qwen GPU8001 image15deeb14b462 and CPU8000 remain unchanged.
No GLM image was promoted to the accepted primary service.

Native quality, full host pin, independent CPU/GPU API and complete Light
run20261108-120756 are qualified within the scopes recorded in23–23.2.
Long-context numerical accuracy, sparse DSA and speculation remain unqualified.
Qwen38 still has its original greedy NLL failure; a diagnosed revisit follows,
then the separate DeepSeek REAP/full backend work. No two-GPU run has occurred.

Recovery evidence: `<ops>/work/models/glm52/codex/pre-run8.json`,
`pre-closeout.json`, `service-stop.json`, the lifecycle audit,
`maintenance-restored.json` and `closeout.json`. The closeout hashes all final
phase verdicts and the successful maintenance restoration. Its SHA256 is
`486bca19258ee0d37acfd890a87914533f9d657d2182d4848dde58caf11826ed`.

## 24. Qwen38 numerical revisit (in progress)

GLM5.2 closed successfully before this revisit. Qwen38's original greedy
reference remains unchanged: CPU NLL 0.3030, ceiling 0.305221, GPU full chain
0.3288, GPU experts with CPU trunk 0.3163, and CPU scalar FP8 0.3134. The last
three fail that gate. Section 24.2 reproduces the original CPU result; no new GPU result is claimed here.

### 24.1 Actual CPU expert-stage fixtures

`tests/export_qwen38_fp8_stages.c` includes `qwen38.c` with `QWEN38_NO_MAIN`,
as the existing CPU test does. It calls the actual AVX/FMA vector function,
the existing table function, and Qwen's SiLU expression. It exports gate, up,
hidden and down results for later comparison through the public GPU ABI.
The engine and backend sources, defaults and original CPU executable remain
unchanged; this adds a separate diagnostic target only.

The no-device oneAPI build uses GCC 13.3.0, glibc 2.39, `-O3`,
`-march=x86-64-v4` and the existing NOCUDA flags. The executable links only
libm, libgomp and libc. The first wrapper attempt exited before compilation
because `setvars.sh` returned 3 for an already initialized environment. That
warning and the failed attempt are preserved; the corrected wrapper checks
`SETVARS_COMPLETED` and verifies both compiler paths. Build attempt 2 passed.

The CPU fixture run and offline packet audit passed:

| Check | Result |
| --- | --- |
| Main thread and 32 OpenMP workers | Initial MXCSR 0x1f80; nearest rounding, FTZ/DAZ off; the harness does not change these modes |
| FP8 decoding on each worker | All 256 encodings pass, including signed zero and both NaN classifications |
| Fixture coverage | 24 cases; rows 1/2/5/8/9, full blocks, divisible-by-8 and odd tails, zero, cancellation, native 2560×640 geometry |
| Packet size and payload | 28,150,012 bytes; exact shapes/lengths, finite scales/inputs/results, no trailing data |
| Random cases: largest vector/table relative RMS across stages | 5.168363897187e-7 |
| Zero-input cases | Both paths return zero throughout |
| Deliberate cancellation cases | Large order sensitivity; largest stage relative RMS 71.0361, not representative native-model evidence |
| First row versus S=1 | Vector down differs at 201×300 for S=2/5/8/9; other recorded comparisons, including native dimensions, match bytewise |

The odd-tail row difference is diagnostic; the original native widths 2560
and 640 have no such residual. Synthetic differences do not establish the
cause of the native NLL failure or justify changing a model threshold.
The earlier scalar-reference GPU ABI tests and their 1e-4 bound remain intact.

Fixture SHA256: `61bc2a3684edb28722752820ef768bd3321e0952e3855f6831e53734dda64053`.
Exporter SHA256: `416e5cf174c43bdb1d2062b919183d44c981cf25f6e8fb69ae317bdf0f2ed29c`.
Original Qwen38 executable SHA256 remains
`a92c24e05dc284734a38261d28f5694d7e9b2fb23b387c3f38d7acfac8aed49e`.
The writer bounds its output, refuses replacement, syncs the complete file
and publishes it atomically. The offline audit rechecks source/binary/input
hashes and the original model references before accepting the fixture.

Evidence and the prepared next steps are under
`<ops>/work/models/qwen38/revisit1/`: pre-build manifests, both build logs,
setup-diagnostic.log, pre-cpu-fixture.json, cpu-fixture.log and
cpu-fixture-verdict.json. No new GPU guard has been used; seven of the original
twelve remain spent. Next, capture matched native inputs with the original
CPU executable/reference and confirm its NLL, then isolate dot, activation
and full-MLP results with existing GPU kernels under one reviewed guard.

### 24.2 Matched native CPU capture and expert stages

The unchanged original CPU executable reproduced NLL **0.3030 over 320 targets**
while recording layers 44–47 on the original greedy continuation. Each layer
contains 340 records: 21 prompt tokens plus 319 subsequent target inputs.
The four files total 14,035,264 bytes. IDs, routing weights and activations
passed the offline audit. Maximum sampled VmHWM was 38.122930 GB, sampled swap
was zero, and the CPU-only capture container exited successfully. This was a
diagnostic capture, not a performance comparison.

Before observing stage errors, the selection was fixed to layers 44–47,
records 20/180/339, ranks 0/9, plus ranks 1–8 of layer 47 record 339. These are
32 distinct experts. The original CPU executable and synthetic exporter were
copied, hashed and preserved under `reference-binaries/` before further builds.

A metadata-only extraction attempt rejected an incorrect F32-only assumption:
all 96 selected scale tensors are BF16. That failed attempt is preserved.
The revised packet retains raw weights and the actual scale dtype; the new
`tests/export_qwen38_fp8_native.c` calls the engine's existing
`q38_decode_scale_tensor` and vector/table math. The `_scale_inv` values are
used directly, as the loader does. The extractor freezes shard identities,
header hashes and byte ranges before reading the selected payloads.

| Check | Result |
| --- | --- |
| Native input packet | 157,634,320 bytes; 32 cases, each D=2560, I=640, S=1 |
| CPU stage packet | 158,800,016 bytes; all shapes and lengths verified |
| Raw weights and activations | Byte-identical between input and output packets |
| BF16 scales | All 9,600 conversions match the exact binary32 representation |
| CPU environment | 32 workers; rounding/FTZ/DAZ and all-256-code FP8 checks pass |
| Stage values | All finite |
| gate: maximum vector/table relative RMS | 1.961350518427e-07 |
| up: maximum vector/table relative RMS | 2.451144688433e-07 |
| hidden: maximum vector/table relative RMS | 3.468334798157e-07 |
| down: maximum vector/table relative RMS | 5.299291701638e-07 |

The native stage exporter build and execution both returned zero. The offline
audit completed at 2026-11-08T15:14:59.579410 UTC. The original engine and
synthetic exporter executables remain byte-identical. No engine/backend math,
default, native reference or numerical gate changed. These CPU comparisons
provide matched inputs for GPU diagnosis; they do not identify the cause of
the native NLL failure or qualify a new GPU result.

Native input SHA256:
`660c4ef5da19ff4b62ffc8b4eeeb14802c7d0c7facfac4329f5dd0cd4bcc1cbc`.
Native stage SHA256:
`a94c96474c16ece197a9204ce4b11b5ed07de5c4ccdd111247e8b4512e1018f2`.
Native exporter SHA256:
`b8413b7f6b4fecf13e698e5fca577c6bf05de93880346343b3504a44aee124e4`.

Evidence under `<ops>/work/models/qwen38/revisit1/`: capture preflight/result/audit,
`native-selection.json`, both header inventories, `pre-native-extract.json`,
`native-extract-verdict.json`, `pre-build-native.json`, `build-native.log`,
`pre-native-cpu-stages.json`, `native-cpu-stages.log` and
`native-cpu-stages-verdict.json`. Seven of the original twelve guarded GPU runs
remain spent; this CPU work used no new GPU guard. Next is a bounded public-ABI
reader using existing GPU kernels, with whole-file validation before device
initialization and tiny synthetic cases before native cases.

### 24.3 Public-ABI diagnostic reader: host checks pass, GPU run stopped in preflight

`sycl/tests/test_qwen38_fp8_stages.cpp` reads the saved CPU stage format and
uses existing backend functions only. It validates every input packet before
initializing a device, runs the smaller shapes first, and records gate/up dots,
SiLU with saved CPU inputs, down with saved CPU hidden values, full MLP,
grouped sync, and grouped issue/take. Borrowed async results are copied before
any subsequent backend call; outputs must be finite and sync/async results
must match bytewise. Numerical differences are diagnostic output, not altered
native or scalar-reference thresholds. Complete GPU results would be saved in
a bounded binary file published only after successful completion.

The reader built successfully against the existing backend objects; actual
build session71240 returned0. Original objects, engine and frozen inputs remain
unchanged. Compiler spill/recompilation warnings in existing kernels are
retained in build-reader.log. The no-device parser validation also returned0
(actual session76362): both accepted packets, totaling56 cases, passed; ten
malformed packets were rejected with device_open=0. Cases cover truncation,
trailing bytes, invalid version/count/ID/dimension and NaNs in weights, scales,
inputs and reference outputs. This qualifies the reader's host parsing only.

Reader SHA256:
`87b2b2d0d1746c76dc783c2d98208a1bdbe8c35c66c60ad8c10525b4799a045f`.
Evidence: pre-build-reader.json, build-reader.log,
pre-reader-host-validation.json, reader-host-validation-verdict.json and its
per-case logs. Build target: `make sycl/tests/test_qwen38_fp8_stages SYCL=1`.
For host parsing without a device, pass `--fixture FILE --validate-only`.
Any future GPU execution requires the host guard and fresh safety checks.

**Safety stop before GPU launch.** During preparation for original slot8,
`sudo -n ipmitool sel elist` exceeded the new preflight's15-second bound.
The maintenance helper exited1 (actual session16243); it had not armed the
restoration timer or changed either apt timer. No GPU guard, model inference
or device comparison started, and no pre-run8 launch manifest was published.
The command was not retried. Pending preflight/maintenance/GPU wrappers now
refuse to run while safety-stop.json is present; the original helper bytes
are preserved in the stop snapshot.

The host-only snapshot at15:28:13 UTC shows the same boot, live watcher,
unchanged348 fixed counters, memoryOK, no new kernel entries in the captured
window, no lingering ipmitool process, both primary Qwen36 services running,
and only the original gpu-gw container. Apt timers remain active/enabled,
package services inactive, and the proposed restoration timer does not exist.
These observations do not verify the current BMC event log or establish a new
hardware fault. Last successfully verified SEL remains34c; the fresh read is
unverified. Per the crash policy, GPU work stops pending review.

Recovery: <ops>/work/models/qwen38/revisit1/safety-stop.json and
stop-sel-timeout-20261108T152813Z/snapshot.json, plus checkpoint.json and RUNLOG.
Original Qwen38 guards remain7/12 spent, zero new in this revisit. The original
CPU0.3030 / limit0.305221 versus full GPU0.3288 failure remains unresolved.
No engine/backend kernel/default or accepted gateway changed. DeepSeek and
two-GPU work remain unstarted.

### 24.4 Authorized recovery and matched GPU stage results

The user explicitly requested continued diagnosis despite errors. The earlier
full extended SEL query's15s timeout remains preserved. The local ipmitool
manual documents that extended listing can be slow and supports latest-record
queries. A bounded `sel list last 1` succeeded in5.894787s and returned34c
(actual session90762/exit0). This recovers the health prerequisite without
proving the earlier timeout's cause. A dated authorization/result/hash record
allowed the prepared wrappers to resume; the old stop file was retained.

Fresh checks verified the original boot/watcher, fixed348 counters, memoryOK,
SEL34c, idle package services and unchanged primary services. A temporary idle
timer hold armed automatic1h restoration before changing timers. Guard
`codex-q38-r1-stage-diagnostics-20261108T164317Z.log` completed at16:44:03 UTC,
rc0/10s workload time; actual driver37012 returned0. Its offline audit1125
returned0. All39 frozen inputs remained unchanged. Post-snapshot/counter/SEL
and kernel-window audits passed; no hardware fault was observed.

Gates:245 chain checks at default subgroup width and245 at SG16; backend suite;
original scalar-reference FP8 expert suite at WARP1/0 and SG32/16, unchanged
1e-4 bound. All passed. Each of four stage runs completed56 cases,672 comparisons
and one3,209,936-byte output packet. Every output is finite, sync/async results
match bytewise, and all recorded first-row comparisons across synthetic row
counts are byte-identical. No new kernel or engine math was involved.

Maximum relative RMS against the actual CPU vector stages, across32 matched
native experts:

| Stage | WARP1 SG32 | WARP1 SG16 | WARP0 SG32 | WARP0 SG16 |
| --- | ---: | ---: | ---: | ---: |
| gate | 1.015421598e-07 | 1.058791958e-07 | 1.162687161e-07 | 1.162687161e-07 |
| up | 1.371504144e-07 | 1.376272374e-07 | 1.515526430e-07 | 1.515526430e-07 |
| silu_cpu | 8.954688123e-08 | 8.954688123e-08 | 8.954688123e-08 | 8.954688123e-08 |
| down_cpu | 2.110131772e-07 | 2.121402478e-07 | 2.170869776e-07 | 2.170869776e-07 |
| mlp | 3.023010467e-07 | 3.027575418e-07 | 3.262174084e-07 | 3.262174084e-07 |
| group | 3.023010467e-07 | 3.027575418e-07 | 3.262174084e-07 | 3.262174084e-07 |
| async | 3.023010467e-07 | 3.027575418e-07 | 3.262174084e-07 | 3.262174084e-07 |

`silu_cpu` receives saved CPU gate/up values; `down_cpu` receives saved CPU
hidden values, isolating those operations. Full MLP/group/async compare complete
expert results. The synthetic cancellation cases remain separate diagnostics;
their large relative differences are not used to change a correctness threshold.
All raw arrays, per-element comparison counts and maxima are retained in slot8/
and gpu-stage-verdict.json, with hashes and the original CPU fixture provenance.

These are small arithmetic differences, not an identified cause of the native
NLL failure. The original CPU scalar FP8 substitution also failed that model
gate, and the inspected CPU uses eight FMA lanes plus a fixed hsum tree while
the GPU uses a different block reduction. An opt-in CPU-order experiment can
isolate that variable next; keep SiLU, routing order, dense activation math,
references and defaults unchanged. No native model retry occurred in slot8.

Original apt timers were restored at16:45:11 UTC after the audit, before further
CPU-side work. Both timers are active, both package services inactive, and the
transient restoration timer is inactive. The temporary diagnostic container
is gone; original services remain unchanged. Total Qwen38 guard use is8/12
(seven original plus this one); remaining four slots require concrete plans.
No new throughput benchmark or accepted model result follows from this run.

Recovery evidence is under <ops>/work/models/qwen38/revisit1/: resume-20261108T1640/,
resume-after-sel-timeout.json, pre-run8.json, pre-slot8-audit.json,
gpu-stage-verdict.json, pre-slot8-restoration.json and maintenance-restored.json.

### 24.5 Opt-in CPU-order FP8 dots: operator and matched-stage gates pass

`COLI_SYCL_F8_CPU_ORDER=1` is a default-off diagnostic for the FP8 warp
family on FP64-capable devices. For full128-column blocks it reproduces the
CPU's eight explicit float FMA lanes, hsum tree and ascending double scaled
block sum. Dense input widths must be divisible128; grouped D and I must
both be divisible128. The existing partial-width paths, SiLU, routing, dense
activation quantization and other formats remain unchanged. WARP0 remains
the original-family escape hatch. Supported subgroups remain16/32; no SG8,
new device capability, device atomics or peer operation was introduced.

No-device build65240 exited0. Guard original-slot9,
`codex-q38-order1-gates-stages-20261108T170153Z.log`, START17:01:53,
END17:02:42 rc0/14s workload; actual driver76221 exited0. Offline auditor82861
exited0 at17:02:58. All65 frozen pre-run inputs were unchanged. Four chain
configurations (off/on, default/SG16) each passed245 checks; both backend
suites passed. Five FP8 scalar-suite configurations passed all five shapes,
including new tiny128x128 and odd-output128x129 before native2560x640, with
the original1e-4 bound. Four stage configurations each produced56 finite
cases/672 comparisons and3,209,936 output bytes; sync/async and all synthetic
first-row invariance checks were byte-identical.

All **276 eligible CPU-vector dot comparisons were bit-exact**, including
all32 native experts' gate/up/CPU-hidden-down at both SG32 andSG16. Default
and WARP0 escape packets remained byte-identical to slot8. Maximum relative
RMS against saved native CPU stages:

| Stage | Default SG32 | CPU-order SG32 | CPU-order SG16 |
| --- | ---: | ---: | ---: |
| gate | 1.015421598e-07 | 0 | 0 |
| up | 1.371504144e-07 | 0 | 0 |
| silu_cpu | 8.954688123e-08 | 8.954688123e-08 | 8.954688123e-08 |
| down_cpu | 2.110131772e-07 | 0 | 0 |
| mlp | 3.023010467e-07 | 2.244288918e-07 | 2.244288918e-07 |
| group | 3.023010467e-07 | 2.244288918e-07 | 2.244288918e-07 |
| async | 3.023010467e-07 | 2.244288918e-07 | 2.244288918e-07 |

The complete MLP still includes the unchanged GPU SiLU; its nonzero difference
is retained. Bit-exact isolated dots do not establish native quality or a
throughput gain. Original CPU0.3030/limit0.305221 and prior failed scores remain
unchanged. Next is the separately frozen CPU-trunk/GPU-experts native rung.

Fixed348 counters/memoryOK/SEL34c and the complete kernel window passed.
Original CPU8000 and accepted GPU8001 services remain unchanged. Total9/12
original guarded attempts spent. The phase maintenance hold is still active,
with automatic restoration armed at17:01:08 for1h; restore explicitly at phase
end. Raw inputs, logs, outputs and audit are in
<ops>/work/models/qwen38/fp8-order1/{pre-build,build-verdict,pre-run9,
pre-slot9-audit,slot9-verdict}.json and slot9/. No new model benchmark yet.

### 24.6 CPU-order native isolation improves but still fails the original gate

The next rung enabled only CPU-order FP8 (SG32/WARP1) with the native dense
trunk on CPU, matching the original expert-only isolation settings: cap128,
18GiB expert allowance, OMP32, f16KV, context4096, MTP/lookup off. The immutable
21-context/320-target CPU-greedy reference and its threshold were retained.
Original tiny fixture first passed8/8 and its numeric oracle, with positive
GPU expert hits and CPU dense execution.

| Native greedy320 configuration | NLL nats/token | Original ceiling | Result |
| --- | ---: | ---: | --- |
| Original CPU vector reference | 0.3030 | 0.305221 | Reference |
| Original CPU trunk/GPU experts | 0.3163 | 0.305221 | Rejected |
| CPU trunk/GPU experts, CPU-order FP8 | 0.3129 | 0.305221 | Rejected |

This single experiment reduces the earlier excess but does not qualify the
model or prove the remaining cause. The native log confirms553 CPU int8 trunk
matrices, no dense chain, the FP8 expert tier and the new2560x640 SG32 grouped
arithmetic. It reports61.3% expert VRAM hits,3060/24576 resident experts,
4879 uploads/1819 swaps, and93820 GPU expert executions. The CPU-only dot
matching result does not include GPU SiLU or host CPU/GPU rank-merge order;
those remain separate diagnosed candidates, not established fixes.

Guard `codex-q38-order1-experts-ppl-20261108T170508Z.log` START17:05:08,
END17:07:09 rc42/88s workload. Actual driver96080 returned42 deliberately for
the numerical rejection; the native engine itself returned0 in85.339s.
The offline audit returned0 directly (no yielded session). All62 frozen hashes
and145 checkpoint file size/mtime/inode records were unchanged. Fixed348
counters/memoryOK/SEL34c and the kernel window passed; no hardware abort.
42 native memory samples recorded maximumRSS38,386,872,320 bytes, zero swap,
and minimum available461,517,910,016 bytes. These timings are not a throughput
benchmark or an interleaved performance comparison.

The failed predecessor blocks the planned full-chain rung; it was not run.
Default CPU-order remains0 and the accepted primary image and production CPU
are unchanged. Original apt timers were restored after the audit; no candidate
container/maintenance hold remains. Total original guards10/12, including all
prior attempts. A fresh plan is required before spending the remaining slots
on a diagnosed arithmetic change. Raw evidence:
<ops>/work/models/qwen38/fp8-order1/{pre-run10,pre-slot10-audit,slot10-verdict,
pre-restoration,maintenance-restored}.json and slot10/.

### 24.7 Stable SiLU candidate: matched stages improve, native score pending

After the CPU-order native rejection, host-only saved-array analysis exactly
reproduced52,116 original CPU hidden values (20,480 native,31,636 synthetic)
using q38_silu's stable sigmoid expression and host expf. On native inputs,
the original GPU expression using even the exact CPU expf differs in8,073
values; saved GPU SiLU differs in8,225. The stable CPU expression with double
exp rounded to float differs in9, max relative RMS6.179226396e-9. This identifies
an expression-order variable; it does not establish the native failure cause.
Host script, library/input hashes and results are retained in fp8-silu1/.

`COLI_SYCL_SILU_CPU=1` is an independent default-off diagnostic. It uses the
stable branch, double exp rounded to float, float1+z, double division rounded
to float, then two float products. FP8 MLP/grouped dispatch requires FP64,
warp family and both widths divisible128. The explicitly enabled public
pipe_silu operation uses the same expression for CPU-input stage diagnostics;
other fused format epilogues remain unchanged. WARP0 restores original math.
No queue/allocation/lifetime/routing changes or new device capabilities.
The isolated experiment keeps CPU-order1 and toggles only the activation.

No-device build actual1240 exited0, verifier30937 exited0, original binaries
preserved. Actual builder image79a5e333ff6f recorded with no device nodes,
networknone and no added privileges. Source patch/inputs remained frozen.
Original unused slots11/12 were explicitly reassigned to this diagnosis after
slot10 blocked the full-chain rung; all ten prior attempts remain spent.

Guard slot11 `codex-q38-silu1-gates-stages-20261108T172121Z.log` START17:21:21,
END17:22:13 rc0/17s workload, actual46894 exit0. All74 frozen inputs remained
unchanged. Twenty tiny-first commands completed: four245-check chain runs,
two backend suites, nine FP8 scalar configurations/five shapes each at the
original1e-4 bound, five56-case/672-comparison stage runs. Independent SiLU-
only/dots-off kernels were covered atSG32/16 too. Three control packets
(default, prior CPU-order-only, WARP0 escape) are byte-identical to phase1.
All outputs finite, sync/async and synthetic first-row invariance byte-identical.

The original offline audit9651 exited1 on an incorrect logging assertion:
the scalar suite exercises dense/grouped ABI calls, not standalone MLP. A
separately frozen v2 audit requires its group marker; the stage logs still
must contain all three pipe/MLP/group markers. Original audit/log preserved
under audit-attempt1/. No GPU rerun or numerical limit changed. V2 audit87602
exited0, with414 eligible CPU-vector dot comparisons bit-exact (baseline
CPU-order plus both new subgroup configurations).

Predeclared native activation limits were max relative RMS2e-8 and<=411
bit differences, plus full-expert max RMS1e-7. Both subgroup widths pass:

| Native stage/control | CPU-order only SG32 | New SiLU SG32 | New SiLU SG16 |
| --- | ---: | ---: | ---: |
| CPU-input SiLU max relative RMS | 8.954688123e-8 | 6.179226396e-9 | 6.179226396e-9 |
| CPU-input SiLU differing bits by element | 8225/20480 | 9/20480 | 9/20480 |
| MLP/group/async max relative RMS | 2.244288918e-7 | 2.136127867e-8 | 2.136127867e-8 |
| Gate/up/CPU-hidden-down | exact | exact | exact |

Fixed348 counters/memoryOK/SEL34c and kernel-window audit passed. Original
services unchanged. Apt hold remains active with1h automatic restoration
armed17:20:46. Total11/12 original guards; next slot12 is the exact prior
expert-only native rung, adding only SILU_CPU1. Original0.3030 reference and
0.305221 ceiling remain unchanged. No native qualification/throughput result
is implied by stage accuracy. Evidence: <ops>/work/models/qwen38/fp8-silu1/
PLAN.md, VALIDATION-ADDENDUM.md, pre-run11.json, pre-slot11-audit-v2.json,
slot11-verdict.json, slot11/ and slot11-audit-v2.log.

### 24.8 Native SiLU experiment remains rejected; launcher correction preserved

Slot12 actual87350 returned1 before any engine invocation: the copied native
launcher still read pre-run10.json, while the new manifest was pre-run12.json.
The copied offline auditor had the same stale name. Its output directory stayed
empty; guard `codex-q38-silu1-experts-ppl-20261108T172409Z.log` END17:24:38
rc1/0s. A separate failure audit verified unchanged inputs/health and no device
runtime initialization. This attempt remains spent in the original12/12.

Under the user's explicit continued-error instruction, CORRECTION-PLAN.md
allocated exactly one additional guard to complete the unchanged native rung.
New scripts/manifest/output/tag preserved the broken originals. No-device
startup check90001 exited0: the old missing filename was reproduced, then the
corrected manifest/flags/tiny command reached an intercepted engine boundary.
No engine or GPU ran in that host test, and no synthetic model score was made.

Corrected guard `codex-q38-silu1-experts-ppl-fixed-20261108T172929Z.log`
START17:29:29, END17:31:19 rc42/80s workload; actual88467 returned42 for the
original numerical gate. Tiny8/8/numeric oracle passed; native engine returned0
in77.251s with both2560x640 SG32 CPU-order and FP8-group CPU-form SiLU markers.
553 CPU int8 trunk matrices/no dense chain and positive GPU expert hits were
verified. The native greedy320 result remains outside the unchanged ceiling:

| Configuration | NLL nats/token | Original ceiling | Result |
| --- | ---: | ---: | --- |
| Original CPU vector | 0.3030 | 0.305221 | Reference |
| Original GPU experts/CPU trunk | 0.3163 | 0.305221 | Rejected |
| CPU-order dots only | 0.3129 | 0.305221 | Rejected |
| CPU-order dots + stable SiLU | 0.3133 | 0.305221 | Rejected |

Closer isolated stages did not resolve the model-level gate. These single
native trials are not an interleaved performance comparison; asynchronous
residency and cache/loading timing may differ. The latest run reports3060/24576
resident experts,4872 uploads/1812 swaps,93880 GPU expert hits and61.3% VRAM hit
rate. No throughput optimization claim follows from77.251 versus85.339s.
The separate CPU-misses-then-GPU-hits weighted sum order remains a hypothesis;
no rank-merge change or full-chain/API/Light run occurred in this phase.

The offline native audit returned0 directly, with all71 hashes and145 checkpoint
file size/mtime/inode records unchanged.38 native memory samples: maximumRSS
38,387,408,896 bytes, zero swap, minimum available461,425,238,016 bytes. Fixed348
counters/memoryOK/SEL34c and complete kernel-window audits passed. No hardware
abort occurred. All raw failed and completed attempts are retained.

Original apt timers were restored after audit; transient restoration timer
inactive, no experimental container/build remains. closeout.json verifies the
unchanged primary image/engine PIDs/log and only gpu-gw running. Both arithmetic
knobs remain default0; operator gates pass, native model qualification remains
held. Cumulative guarded count13 = original12 + one diagnosed-launcher
correction. This budget is exhausted; further work requires a new concrete
plan. Evidence: <ops>/work/models/qwen38/fp8-silu1/{slot12-failure-verdict,
pre-startup-check,startup-check-exec,pre-run13,pre-slot13-audit,slot13-verdict,
pre-restoration,maintenance-restored,closeout}.json and slot13/.

### 24.9 Opt-in native FP8 rank merge: host and tiny GPU gates

New bounded plan `<ops>/work/models/qwen38/rank-order1/PLAN.md` allows three
guards after the prior13 attempts, preserving every prior rejection. The
independent variable is Q38_TIER_RANK_ORDER (default0). Native FP8 row decode
buffers unweighted CPU outputs; qt_take_rows drains every issued device,
validates ranks/mask, copies raw results into owned rank slots, and only then
releases issue_open. The engine adds all experts in router rank order with
the CPU weighted-add loop, then the shared expert. Default collection,
Vulkan/int4/GGUF, device kernels and placement policy remain unchanged.

The host arithmetic diagnostic uses all ten previously selected CPU-vector
outputs/gates at layer47 record339.1013 of1024 hypothetical residency masks
change bits when CPU misses precede GPU hits; max relativeRMS8.22216e-8 and
absolute2.98023e-8. All-CPU/all-GPU controls are exact. Actual48780 exited0;
input/script/libm hashes were frozen before and verified after. The saved
dump has no residency mask, so this does not establish native causality.

No-device build/test33074 exited0. Six tests passed: raw collection,
ASan/UBSan collection, legacy take-error, shutdown, and tiny fake-engine
option off/on. Raw tests cover both device failures, invalid/duplicate/missing
ranks, bad destination/stride/overflow, mask0 and bit31, untouched CPU rows/
padding and a recycler waking after issue release. The fake engine refuses
issues, so its result is an all-CPU integration check. Five old binaries are
preserved in baseline-binaries. New qwen38 SHA
`c9acb6e6024101565b22fa68f60861c75cfba472940c62197f43fecebe3fbfe9`.

No-device startup38005 caught an interleaved-log count assertion (raw244,
245 after the existing SYCL diagnostic normalization). It ran no engine;
original scripts/logs/manifests remain in startup-attempt1. Corrected startup
24594 passed all three intercepted launch boundaries plus saved-log positive
and negative controls. No guarded attempt was consumed by these host checks.

Guard1 `codex-q38-rank1-gates-20261108T175226Z.log` START17:52:26,
END17:53:14 rc0/16s; actual72508 exited0. Chain default and SG16 each245 PASS,
backend PASS, original tiny rank0/rank1 each8/8 and numeric-oracle PASS with
positive expert hits; rank1 dispatch logged. Independent audit returned0
directly:75 hashes/145 checkpoint file stats unchanged, fixed348/memoryOK/
SEL34c and kernel window clean. New1/3 spent, cumulative14. Native greedy/
human and full-chain quality remain pending at unchanged references/limits.
Candidate math knobs explicitly stay1 as in the previous0.3133 trial;
all three knobs remain0 by default. No image/gateway/throughput promotion.

Evidence: rank-order1/{HOST-PLAN,PLAN,REPORT}.md, host-rank-result.json,
pre-host-build.json, build-verdict.json, startup/, pre-gates.json,
pre-gates-audit.json, gates-verdict.json, gates/ and gates-audit.log.
The temporary apt-timer hold has automatic3h restoration armed17:51:58;
restore it immediately when this phase closes.

### 24.10 Native rank merge rejected; phase closed, primary unchanged

Optional rank-merge code committedc6263616 after gates. Guard2
`codex-q38-rank1-experts-20261108T175420Z.log`, START17:54:20,
END17:56:11 rc42/80s workload; actual26848 exited42 for the numerical gate.
The engine itself exited0. Native logs prove553 CPU int8 trunk matrices,
no dense chain, CPU-order FP8 dots/CPU-form SiLU and rank merge for all-CPU,
mixed and all-GPU routes. No human or full-chain rung ran after rejection.

| Configuration, CPU trunk/GPU experts | Greedy320 NLL | Ceiling | Result |
| --- | ---: | ---: | --- |
| Original implementation | 0.3163 | 0.305221 | Rejected |
| CPU-order dots | 0.3129 | 0.305221 | Rejected |
| CPU-order dots + CPU-form SiLU | 0.3133 | 0.305221 | Rejected |
| Above + global router-rank merge | 0.3122 | 0.305221 | Rejected |
| Original CPU vector reference | 0.3030 | 0.305221 | Reference |

These are separate diagnostic native trials, not an interleaved throughput
comparison; runtime residency/cache timing can differ. No speedup or native
qualification follows from the score changes. Latest3060/24576 residents,
4880 uploads/1820 swaps,93716 GPU expert hits,61.2% VRAM hit rate.

Independent audit exited0 directly:76 hashes/145 checkpoint file size/mtime/
inode records unchanged;39 native memory samples, maximumRSS38,388,473,856
bytes, zero swap, minimum available461,140,873,216 bytes. Fixed348 counters,
memoryOK, SEL34c and kernel-window audit passed. No hardware abort.

Original apt timers restored and transient restore timer inactive before
closeout; onlygpu-gw, primary engine PIDs/image/log unchanged. All three knobs
remain default0. New phase used2/3 guards, cumulative15; the conditional third
guard is unused and cannot bypass the rejected prerequisite. No image/gateway,
API/Light or XeLink promotion. Qwen38 remains held; remaining DeepSeek models
require their separate backend port and tiny-first plan. Evidence under
rank-order1/: pre-experts.json, experts/, experts-verdict.json,
pre-restoration.json, maintenance-restored.json, closeout.json and REPORT.md.

## 25. DeepSeek V4: isolated single-card operation backend

> Not in this tree: the port of sections 25–29 (`sycl/dsv4/`, its Makefile targets and launcher
> hooks) was made against an earlier DeepSeek V4 engine and is kept outside this backend. These
> sections stay as the record.

2026-11-08, `<ops>/work/models/deepseek/sycl-port1/`. The remaining DeepSeek
models need the separate DSV4 ABI; the qualified generic `coli_cuda_*` backend
does not provide it. Starting sourcec174cd6a, original CUDA backend SHA
`f6c74d5cded17161af914020e917de5f33846dac6d2f5e4020c422ebdbf89960`.
Native REAP and full Flash remain untested. No engine/image/gateway promotion,
PPL/API/colibench or XeLink result is claimed in this section.

### 25.1 Source migration and ownership

`prepare-input.py` stages the original source without modifying it; removes
cooperative fused MoE, graphs/profilers, peer operations and EP2/TP2; restricts
initialization to exactly device0; and excludes cuBLASLt/DeepGEMM/NCCL. dpct
ran in the oneAPI container with no device mounts/network. DriverEND18:03:24
recordedrc0. Its initial tool response was truncated at context compaction:
the tool-session ID/exit is not claimed or invented. Completion is corroborated
by the driver END, complete translation log and absent container; no rerun.
Twelve frozen inputs and75 existing generic SYCL sources were unchanged.
All49 raw output/helper files and diagnostics are hashed and preserved.

Only new nested `sycl/dsv4/` source is added. The parent's generic wildcard
does not include it. No shared DPCT helpers, qualified kernels or production
checkout were overwritten. Raw translation is retained separately from the
curated source; NVIDIA-only/preprocessor branches and implicit DPCT runtime
ownership are removed. The compiled binary exports exactly the original86
C ABI functions, including explicit refusals for unsupported operations.

The initial runtime has an independently owned context/two in-order queues,
created only after complete argument validation and actual capability checks.
It requires one visible Intel Level Zero GPU, SG32, WG256, device USM and FP64.
An allocation ledger records every device block. All H2D/D2H transfers finish
before returning, protecting stack descriptors and released/reused host-cache
leases. Local D2D copies preserve queue order. Every free drains both queues;
buffer growth allocates before retiring old storage. Shutdown includes persistent
KV and projection caches and clears their pointers/dimensions for re-init.
It retains storage/context after a failed drain rather than freeing in-flight
memory; actual driver-fault recovery has not been tested. Error state outlives
queues/context, and no driver object is constructed or destroyed globally.

Host review additionally fixed cache dimension/contiguous-prefix checks,
integer growth, one-past device-copy bounds and projection pointer lifetime
when callers omit the optional release. Native engine close joins the MoE
prefetch worker before releasing mirrors; callers must retain that ownership
contract. Independent-context native FP4 operation does not qualify combined
GGUF/generic-backend lifecycle or any cross-context pointer sharing.

E8 scale storage belongs to the context and preserves original table values.
The unsupported CUDA E4M3 conversion uses the CPU binary32 ties-to-even algorithm,
checked against an independent finite codebook search at representations,
rounding boundaries, special values and1,000,000 deterministic random inputs.
All1,001,022 host cases passed. Compilation uses precise math with contraction
off and explicit FMA retained. The compiler's override warning remains in logs;
saved effective cc1 flags verify contractoff and absence of fast/finite-only,
NaN-discarding or reassociation flags. Dynamic local-memory launches check the
actual limit, reserving4KiB for static locals. Card1 reports128KiB local memory.

### 25.2 Host builds and tiny-first GPU gates

All builds ran without devices, with read-only source and isolated outputs.
Build1 actual26290 exited1 after two obsolete generated host-timestamp member
references; the FP8 host gate and runtime compilation had passed. Full failed
inputs/logs are preserved. Build2 actual42745 and build3 actual46215 exited0;
each was preserved before the additional lifetime review fixes. Final build4
actual90220 exited0:17 frozen inputs and75 generic sources unchanged, host FP8
gate, invalid-init checks and14 malformed CLI requests all passed. The original
test's dangerous `argc>1` six-device selection is replaced by strict complete
argument parsing before initialization.

Final standalone test SHA
`af3e2a195337e90846eb67235cbe584ab46c7350073f8e4c7ca4efa766995a1e`.
Exact compile/link command is `build4.sh`/`pre-build4.json`, logs `build4.log`;
effective flags and CLI evidence in `build4/`. No-device startup actual76858
exited0 after intercepting all three first tracked-command boundaries before
subprocess launch. Its manifests match the actual guarded commands. Each
guard freezes34 source/helper/binary hashes and separately checks all build
runtime/generic hashes. No models were mounted or loaded.

| Guard / actual session | Result | Guard workload time | Evidence log under <ops>/logs |
| --- | --- | ---: | --- |
| Allocation-only /60375 | Two complete context lifecycles;128 borrowed host/local-copy iterations, ledger/cache checks PASS | 3s | codex-dsv4-port1-alloc-20261108T182139Z.log |
| Tiny + generic gates /28162 | Six asymmetric/block-tail FP8/FP4 cases, source-reuse refills, projection shutdown/reinit;245 default+245 SG16 chain and backend PASS | 10s | codex-dsv4-port1-tiny-20261108T182242Z.log |
| Original DSV4 ops /72688 | Finite maximum absolute error2.98023e-08 against unchanged1e-3 limit | 3s | codex-dsv4-port1-full-20261108T182349Z.log |

Small matrix geometries are O/I=7/32,33/160,129/160, each FP4 and FP8. The
original op suite retains its references/tolerance with explicit finite checks.
Some composed operations use GPU intermediates, as before; these checks are
not an independent full-model quality oracle. Optional attention overlap,
optimized sparse paths, native shapes and combined generic/DSV4 operation
remain unqualified. The standalone DSV4 subgroup remains32; the SG16 checks
above exercise the existing generic backend.

Every actual driver exited0. Each independent post-run audit returned0 directly:
frozen hashes unchanged, memoryOK/fixed348 counters unchanged, no new kernel
window line, finalSEL34c. Three of three guards used, no retry. The unique2h
apt-timer restoration was armed before holding idle timers at18:21:23; original
active/enabled timers were restored at phase close, transient timer inactive.
Closeout verifies all four primary PIDs, accepted image15de, primary GPU log,
and onlygpu-gw running. No maintenance hold remains.

Evidence: HOST-PLAN.md, GPU-PLAN.md, translation-verdict.json, pre-build*.json,
build*-verdict.json and preserved build1/2/3-inputs, startup/, pre-{alloc,tiny,full}.json,
the three output directories and verdicts/audits, pre-restoration.json,
maintenance-restored.json and closeout.json. Next: host engine integration,
expanded operation/native-shape gates, official independent target-only tiny
oracle, then own CPU/native REAP quality/API/Light ladder; full Flash follows.


## 26. Native DeepSeek V4 engine integration, tiny gates (2026-11-08)

The Linux native engine now links the separate SYCL operation backend with
`make deepseek-v4 SYCL=1`. Four independent tiny-model cases and repeated
engine lifecycle gates pass on card1. Native REAP/full Flash checkpoints,
quality scores, API/colibench and performance are still untested. The user
prioritizes making every model correct and operational before optimization.
No gateway/image promotion or new colibench run belongs to this milestone.

### 26.1 Integration and activation contracts

The recursive build keeps DSV4 objects/device link separate from the generic
SYCL context, uses the qualified precise/contract-off flags and tracks compiler
flags/dependencies. CUDA/HIP/Vulkan/DLL/combined-backend requests are refused.
The SYCL engine defaults to CPU unless DSV4_CUDA=1 (or the launcher's explicit
GPU selection) requests the device. A failed requested initialization is an
error. Malformed/multiple/nonzero ordinals and GGUF fail before initialization.
`--backend-info` reports a unique compile-gated marker before runtime/model
work; launcher detection checks that marker and resolved libsycl using ldd,
without executing an unknown binary. Existing CUDA behavior remains supported.
Native VRAM cache budgeting is not supplied by the generic --vram option.

Engine teardown joins prefetch work, releases mirrors and engine-owned dense,
MoE, projection and mHC scratch before destroying the independent context.
Two real engine opens with six fresh sessions verify repeatability. Optional
scratch paths not reached by this tiny geometry remain unqualified.

The oracle helpers previously passed NULL as their engine and silently left
dense matrices on the CPU. They now use the opened engine, like ordinary
generation and --record-oracle. Tests require positive resident uploads and
actual FP8 matvec/matmul and FP4 expert execution markers before accepting any
model result. This change exposed two missing engine/primitive contracts:

| Contract | Prior behavior | Corrected SYCL engine behavior |
| --- | --- | --- |
| Dense FP8 host input | Raw floats passed to the primitive; CPU first QDQs activations per128 | DSV4_SYCL_DENSE_QDQ=1(default) uses the actual CPU E4M3/UE8M0 quantizer;0 preserves raw diagnostic input |
| Host routed/shared expert input | Raw floats passed to moe/expert_group; CPU FP4/FP8 entries first QDQ | DSV4_SYCL_EXPERT_QDQ=1(default) applies the same CPU quantizer;0 preserves raw input |
| Hybrid expert scratch | A CPU expert can overwrite thread-local quantizer storage | Acquire GPU input only after the CPU subset, immediately before blocking upload |

Raw low-level operation ABI, GPU kernels, original operation references and
thresholds were not changed. Activation-buffer MoE already quantizes input and
uses its existing path. Host tests call the actual CPU quantizer: dense9/9
cases for default/on/off, expert8/8 cases for each setting plus invalid-input
refusals. The engine3 before/after dense proof retains the old bridge's0/9
failure and corrected9/9 result. These host tests validate input contracts;
the independent model gates below establish the resulting GPU correctness.

### 26.2 Evidence and preserved failures

The deterministic official fixture uses Transformers5.14.1 and torch2.13.0+xpu
exclusively on CPU in a pinned, no-device/no-network container. Its original
three cases are unchanged. Engine4 independently reconstructed byte-identical
exported weights and original oracle before producing a fourth custom prompt
reference. The custom prompt is the original lifecycle failure case, not a
replacement. HF predicts83,84,85,21 with final top1 margin0.03155476; the CPU
engine agrees, margin0.005667448. GPU engine3 incorrectly produced88 last.

| Phase | Preserved result | Consequence |
| --- | --- | --- |
| engine1 | Tiny tokens passed but required resident dense execution was absent | Correct oracle helpers to pass the actual engine |
| engine2 | Actual resident path, short TF15/16; first pos0 expected55/got53, greedy8/8 | Correct dense input QDQ |
| engine3 | Official TF116/116 and greedy16/16; internal reopen repeatability passed but CPU equality failed custom8 token4 CPU21/GPU88 | Keep complete phase failed; investigate expert input |
| engine4 (CPU only) | Independent HF and CPU agree all12 custom greedy tokens; original fixture exact | Correct host expert QDQ; planned GPU diagnostic unused |
| engine5 | All original and custom references plus CPU-identical lifecycle pass | Accept tiny integration milestone; native models still pending |

All failed attempts remain on disk. Host-only setup/compile failures include
engine1 test-helper/stamp mistakes, engine2 final detection fixture path,
engine3 omitted compile defines and a POSIX MAX_INPUT macro collision, and
engine4's missing read-only watcher mount (refused before child launch).
Their corrected attempts have distinct logs/results; none is a GPU retry
without a changed, documented plan. RUNLOG and per-phase reports preserve the
actual tool sessions and return codes.

Final immutable build engine5 actual60407 exited0. Independent audit verified
1394 frozen inputs,2780 source copies, unchanged qualified worktree binaries,
49 host unit checks, actual launcher detection and14 pre-init CLI refusals.
CPU/SYCL-off model ladder actual31822 exited0: each128/128 teacher-forced
predictions and20/20 generated tokens,12 fresh lifecycle sessions across both
builds, explicit/unset CPU behavior and5 pre-runtime refusals. No-device
startup actual82518 intercepted both tracked command boundaries.

Final CPU engine SHA `1b702ec1b2e8db51187c8728487aae73eedb61da759a8c232bb16804e6a53540`.
Final SYCL engine SHA `838911dc2036517b1e06d55bf45a30a25bb3c5a7c8e9534dd8737785ba473467`.
Standalone operation binary remains identical to the section25 artifact af3e2a19.

| Guard / actual session | Result | Guard workload time | Log under <ops>/logs |
| --- | --- | ---: | --- |
| engine5 gates /32468 | Original tiny/full ops, worst_abs2.98023e-08 within1e-3;245 default+245 SG16 chain and backend PASS |13s|codex-dsv4-engine5-gates-20261108T193141Z.log|
| engine5 tiny /89568 | Original3+custom8:128/128 TF,20/20 greedy; two engines/six sessions exactly equal CPU |12s|codex-dsv4-engine5-tiny-20261108T193248Z.log|

Each stage freezes56 file hashes and checks all1394 build inputs. Independent
post-run audits verify actual markers, child exits/log hashes, no swap/RSS
violation, unchanged348 counters/memoryOK, clean kernel window and SEL34c.
Two of two guards consumed; no untracked run. Actual16238 closeout exits0:
original apt timers restored active/enabled, temporary restorer inactive,
all four primary PIDs/image/log unchanged, only gpu-gw running. No hold remains.

Evidence root `<ops>/work/models/deepseek/engine5/`: PLAN/HOST-PLAN/GPU-PLAN,
contract-verdict, build1-verdict, host-tiny-verdict, startup-audit, pre-* manifests,
gates/tiny directories and verdicts, maintenance-restored and closeout.json.
Earlier engine1–4 reports and failed outputs remain intact. Next: bounded native
operation shapes and sequential teacher-forced NLL validation (prefill alone
does not exercise GPU decode experts), then REAP and full Flash CPU/GPU/API
qualification, followed by the remaining Qwen38 accuracy hold.

## 27. DeepSeek sequential scoring validation (2026-11-08)

Prefill alone can leave experts on the CPU, so a prefill-only NLL result does
not qualify the native GPU decode path. `sycl/dsv4/test_scores.c` adds bounded
teacher forcing through actual `target_token(engine)` calls. Its additional
batch mode calls `target_batch(engine)`. Both score exactly `full_ids.size-1`
next-token positions, with continuation starting at the last prompt predictor.
They report separate all-token and continuation NLL. No engine/kernel arithmetic
or default changed from commit3cbf2ec4.

The driver rejects malformed CLI, nonzero device ordinals, unknown model
geometry, noninteger/out-of-range IDs, invalid prefix alignment, over1024 IDs
and invalid1..256GiB limits before GPU initialization. It exclusively creates
the output file, fsyncs each score row, rejects nonfinite logits, and computes
double logsumexp with a maximum shift. CPU builds refuse device mode. The
validation forces MTP, DSpark, prefix reuse and optional batch/overlap paths off.

CPU and SYCL tools reused59 accepted engine objects. Build10297 compiled and
linked both, then exited1 on a Python checker typo (`else64`). The original
inputs/log remain intact. Distinct checker2 corrected that syntax and passed
2 analytic math self-tests plus26 malformed-input checks, without rebuilding
or exposing devices.1395 frozen inputs and all reused objects were unchanged.
CPU scorer SHA256 `da9d3aba8af79742ac1ce1d93174b34ba8fa2d482b7a6fa190f40a4946cfc92b`;
SYCL scorer `f24bd14ac8a326094bf7e5a85bfc422b4aa7bbfeff61ded645e3a7789fec9dd7`.

Host actual69666/0 ran16 cases: CPU and SYCL-disabled, decode and batch, on the
four independent fixtures. All496 rows passed exact oracle argmax, input/target
alignment, finite float32 dump checks and independent Python NLL within1e-11.
The two builds and two modes produced byte-identical logits. Independent HF
teacher logits also confirmed custom8/long top1; their quantized C logits are
not claimed bit-identical to HF. Limits were frozen before GPU execution:
exact tiny-oracle top1 and continuation NLL<=CPU*1.007+0.0001.

| Case | Scored positions per mode | CPU continuation NLL | GPU decode NLL | GPU batch NLL |
| --- | ---: | ---: | ---: | ---: |
| short |15|3.745977118596|3.745977118596|3.745977118596|
| compressed |23|3.711990419934|3.711990419934|3.711990419934|
| long |75|3.785563084105|3.785563084105|3.785563084105|
| independent custom8 |11|3.993565872308|3.993565872308|3.993565872308|

These are deliberately synthetic tiny-model NLL values, not native model
perplexity measurements. GPU decode proves dense/expert QDQ and FP8/FP4 execution
markers; batch proves resident dense GPU execution. All three resident layers
upload positively. The8 GPU cases cover248 next-token positions.

| Guard / actual session | Result | Workload time | Log under <ops>/logs |
| --- | --- | ---: | --- |
| engine6 gates /12111 | Original tiny/full ops worst_abs2.98023e-08;245 default+245 SG16 chain and backend PASS |14s|codex-dsv4-engine6-gates-20261108T194709Z.log|
| engine6 tiny /21717 |8 score cases,248 exact independent predictions, continuation NLL equal CPU |18s|codex-dsv4-engine6-tiny-20261108T194812Z.log|

Both guards independently audited39 phase hashes plus1395 build inputs,
child exits/log hashes, memory samples, unchanged348 counters/memoryOK,
SEL34c and clean kernel windows. Closeout actual62823/0 restored original
active/enabled apt timers and stopped the transient restorer; four primary
PIDs, image and log remained unchanged.2/2 guards consumed, no hold remains.

Evidence is `<ops>/work/models/deepseek/engine6/`: PLAN/HOST-PLAN/GPU-PLAN,
build1 failure and compile audit, cli2, host-scores/verdict, acceptance.json,
pre-* manifests, gates/tiny logs and independent verdicts, closeout and REPORT.
Native checkpoint qualification, API output comparison and performance are
still pending. Next: bounded native operation shapes, then REAP/full Flash
CPU/GPU/API checks and the outstanding Qwen38 accuracy issue. No image changed,
no colibench run occurred, and optimization/XeLink remain deferred.

## 28. DeepSeek V4 REAP hybrid correctness candidate (2026-11-08)

The native REAP checkpoint's default GPU dense/expert configuration failed
the original CPU-greedy NLL gate. Diagnostics preserved under
`<ops>/work/models/deepseek/reap1/` through `reap13/` identified GPU dense
accumulation differences, and independently compared default and existing
CPU-order GPU reference kernels. The CPU-order path is opt-in through
`DSV4_SYCL_DENSE_CPU_ORDER=1`; it handles supported packed FP8 direct, grouped
and batch dense projections using the existing reference kernel, transferring
weights each call. Unsupported layouts retain the normal dispatch.
`DSV4_SYCL_DECODE_DENSE_MIRRORS=0` is a separate diagnostic; default mirrors
remain enabled. The tested correctness candidate uses CPU-order GPU dense,
`DSV4_CUDA_EXPERT_MIRRORS=1` (CPU routed experts), explicit
`V4_PREFILL_CHUNK=5`, MTP/draft/DSpark and optional overlap/fused expert
paths off. It is hybrid, not a GPU expert or performance qualification.

| Native REAP fixed continuation | CPU NLL | Candidate GPU NLL | Original maximum | Evidence |
| --- | ---: | ---: | ---: | --- |
| Greedy, 64 targets | 0.19272071082975584 | 0.19272071082975584 | 0.1941697558055641 | REAP12, all 68 logit rows identical CPU, raw64 generation and self-check 64/64 |
| Human text, 320 targets | 3.0220531375853694 | 3.0220531375853694 | 3.043307509548467 | REAP13, all 336 logit rows identical CPU sequential |

REAP12 guard actual12818/0 and REAP13 actual24712/0 have independent audits,
source/model manifests, resource and health evidence. REAP11 host adapter
contracts passed 12/12, and its tiny/native batch reference comparisons
matched CPU; guard76466/0. The isolated API candidate image
`sha256:748cc6088db9e12b0449d499fff2fade5b22de8121670ab0f653dca8cc46ad98`
used that exact engine/config: tiny guard17947/0 and native guard54601/0
passed, and raw completion plus chat returned HTTP200 with exact CPU text and
token usage, with no first character divergence. The token-capped CPU chat
reference contains an inaccurate qualifier; equality is functional, not a
factual-quality claim. Gateway parent shutdown was clean; the earlier CPU
tiny group-shutdown warning is preserved in `reap-api1/`. Postflights recorded
348 unchanged GPU counters, healthy memory and SEL34c, with primary services
unchanged and maintenance timers restored. No persistent service was promoted.

The full Flash checkpoint is distinct: E256 rather than E132 can take the GPU
router, which needs its own H4096/E256 CPU comparison or validated CPU
fallback. Its independent CPU human NLL is 2.855230780322369 (frozen maximum
2.8753173957846254), greedy NLL 0.23987959788719587 (maximum
0.2416587550724062); see `<ops>/work/models/deepseek/full-flash1/`. Full
Flash GPU and API are not yet qualified. Qwen38's original accuracy hold also
remains. No new performance or multi-GPU claim follows from these results.

Post-adapter regression guard `reap-gates1` actual46387/0 passed the DSV4
tiny/full operation suites, `cuda_chain_test --device 0` (245/245) with both
default subgroup and `COLI_SYCL_SG=16`, and `backend_cuda_test --device 0`.
The backend's deliberately impossible allocation is an expected negative
case. Independent audit chunk141279/0 checked saved child results, logs,
29 frozen inputs and 1403 source hashes; all but these documentation files
matched the tested REAP11 build. Postflight reported unchanged 348 GPU
counters, memory OK and SEL34c; primary services remained unchanged and
maintenance timers were restored. No image or production service was promoted.

## 29. Full Flash E256 CPU-router hybrid candidate (2026-11-08)

The full Flash checkpoint has 256 routed experts, so unlike REAP it can enter
the E256/top6 GPU token router. Its H4096 GPU router was not independently
validated against CPU. `DSV4_SYCL_CPU_ROUTER=1` is a default-off SYCL-only
correctness fallback: the token GPU router entry returns before a device job,
and its existing caller executes the BF16 CPU router. The separate optional
GPU MoE batch router is not controlled by this knob; the tested candidate
explicitly keeps `COLI_CUDA_MOE_BATCH=0`. Its one-time selector marker can
appear in a CPU-only SYCL-linked service because the bypass decision precedes
mirror lookup; it proves the choice, not device initialization. GPU evidence
requires the device, positive dense uploads and dense execution markers too.

The no-device host spy contract tests unset, `0`, `1`, `true`, `01` and `2`
at H128/H4096: 12/12 passed, exact `1` bypassing all mocked GPU calls. The
staged oneAPI candidate build passed no-device backend-info and scorer
self-test. A synthetic H128/E256/top6 fixture repeated weights from the E4
tiny fixture and generated its own CPU self-oracle. `full-flash-router-gates1`
guard16404/0 and audit chunkc06ce5/0 passed DSV4 tiny/full operation tests,
generic chain default and SG16 (245/245 each), backend, and the E256 tiny
engine teacher16/16, greedy8/8 and 15 score rows with byte-identical CPU
logits. This synthetic fixture is not external model-quality evidence.

The full checkpoint used the exact CPU-router/token fallback, CPU-order GPU
dense (`DSV4_SYCL_DENSE_CPU_ORDER=1`), CPU routed experts
(`DSV4_CUDA_EXPERT_MIRRORS=1`), chunk5 and optional GPU MoE batch off;
MTP/draft/DSpark/prefix/async/fused/TC/DG were excluded. 43 GPU dense layers
uploaded. Original CPU limits were frozen before any full GPU attempt.

| Full Flash continuation | CPU NLL | Hybrid GPU NLL | Original maximum | Evidence |
| --- | ---: | ---: | ---: | --- |
| Greedy, 64 targets | 0.23987959788719587 | 0.23987959788719587 | 0.2416587550724062 | guard47516/0, 68 logit rows byte-identical CPU; raw64 matches and self-check64/64 |
| Human text, 320 targets | 2.855230780322369 | 2.855230780322369 | 2.8753173957846254 | guard22876/0, 336 batch logit rows byte-identical CPU sequential |

Independent audits63704/0 and66378/0 and `full-flash-native1/closeout.json`
record the model metadata and all 48 shard headers, sizes and mtimes; shard
payload hashes are not claimed. A separate local image
`sha256:836fa4f55262a149172855b18d569ec170d58076d75ddec178a170199566a922`
used the same candidate CLI. CPU tiny/native and GPU tiny/native isolated
API lifecycles passed, with native GPU guard28726/0 audit chunk6edddf/0
returning raw completion and chat HTTP200, exact CPU text/usage and no first
divergence; the chat response was `Rome`. Native GPU API peak RSS was
155,251,990,528 bytes with no swap. Postflights recorded all348 monitored GPU
counters and memory OK, SEL34c, unchanged primary services, and restored apt
timers. All artifacts are under `<ops>/work/models/deepseek/full-flash1/`,
`full-flash-router1/`, `full-flash-router-gates1/`, `full-flash-native1/` and
`full-flash-api1/`. No persistent service was promoted, and this hybrid does
not qualify default GPU routing/experts, other devices, speed or XeLink.

## 30. Qwen38 fixed-row precision investigation (2026-11-08)

Qwen38 remains held: the original CPU greedy reference prints 0.3030 nats/token
and its unchanged maximum is 0.305221; the CPU-trunk/GPU-expert candidate with
CPU-order FP8 dot, CPU-form SiLU and rank-order merging prints 0.3122. This is
not a qualified GPU model, throughput result or image promotion.

Default-off `Q38_BOUNDARY_TRACE`, `Q38_BOUNDARY_ROW` and
`Q38_BOUNDARY_LAYERS` save a selected host-layer row at attention, residual
and expert boundaries; optional `Q38_BOUNDARY_STATE=1` adds host DeltaNet
state and intermediate values for a single non-multiplexed row. Trace files
are exclusively created and capture can perturb scheduling. In no-device tiny
CPU tests, the old engine, new engine with trace off and new engine with trace
on all passed 8/8 and had identical dumped logits; recorded expert inputs
matched the existing activation dump. Full CPU trace runs printed 0.3030 and
reproduced the previous layer25/26 activation files and boundary records
byte-for-byte. The state-trace tiny test produced all 60 expected files.

Card1-only guarded `boundary-gpu1` exec18769/0 and `recurrent-gpu1`
exec62208/0 each passed the tiny GPU numeric oracle before the fixed 320-token
diagnostic. Both independently audited, with 348 monitored counters and memory
health unchanged, SEL34c, primary services unchanged and apt timers restored.
At fixed zero-based row22, layer26 entered recurrent attention with all 786432
state floats and all 30720 convolution-ring floats identical to CPU. Its
gated norm differed by relative RMS 4.54e-7, then its CPU int8 output
projection differed by 0.002282. The actual `dense_act_i8` on the saved norm
vectors changed one code at index769, from -26 to -27.

The device-free `projection-replay1` extracted only the checked native BF16
layer26 output-projection tensor and reproduced both saved projection outputs
bit-for-bit with the existing integer-dot kernels. Replacing just that one
activation code with the CPU value, while retaining the GPU-run scale, reduced
the output relative RMS from 0.002281847 to 8.415e-8. This proves the local
rounding-boundary effect for that row; it does not establish a whole-sequence
fix. Full commands, source/model identities, traces and audits are retained in
`<ops>/work/models/qwen38/{boundary-gpu1,recurrent-trace1,recurrent-gpu1,recurrent-quant1,projection-replay1}/`.

The existing `Q38_TRUNK_SKIP=dnout` setting keeps all 36 DeltaNet output
projections in BF16 as a broad precision experiment. Tiny CPU scoring passed
8/8, but fixed native CPU greedy NLL was 0.3062, above the original 0.305221
maximum; human NLL 2.6363 was below its 2.6508261 maximum. The candidate is
rejected on greedy quality and was not run on the GPU or deployed. See
`<ops>/work/models/qwen38/dnout-bf16-1/closeout.json`.

## 31. Qwen38 opt-in host-exact FP8 expert SiLU (2026-11-09)

`COLI_SYCL_F8_HOST_SILU=1` is a default-off grouped-FP8 correctness path.
For CPU-order full-block warp groups on a device with FP64, GPU gate/up rows
are copied to pinned host buffers, transformed using the process's `expf`
and the CPU engine's stable sigmoid/SiLU expression, then copied back for GPU
down projection. Explicit waits protect the staging lifetime; there are no
device atomics on host memory or cross-device copies. Unsupported shapes,
other formats and the default path keep their previous behavior. These extra
synchronizations favor correctness over speed, with no performance claim.

No-device host scalar and bulk tests matched all 20480 saved CPU hidden values
and 100010 additional finite inputs. The initial guarded operator phase
`fp8-host-gates1` exec62719/0 passed generic/tiny and knob-off byte controls,
but its independent strict audit failed 301/81920 native and 47/51200 targeted
output values. That failure is retained. Host probes found its icpx-linked
runner defines Intel `expf`, unlike the Qwen engine's `expf@GLIBC_2.27`; its
math differed from saved CPU hidden values in precisely the four native case
IDs and one targeted case ID that failed. Explicit `-lm` alone did not change
the icpx binding. The same unmodified stage test linked by GCC with the Qwen
engine's library order and SYCL device registration resolves glibc `expf`;
the reproducible target is `sycl/tests/test_qwen38_fp8_stages_hostlink`.

Separately guarded `fp8-host-gates2` exec6891/0 and independent audit passed
four chain tests of 245 operations each, two backend gates and tiny FP8 tests;
knob-off bytes matched the earlier baseline. With host SiLU on, all 32 saved
native experts matched CPU vector output bit-for-bit at SG32 and SG16, and
all 20 paired layer25 experts matched at SG32; synchronous and asynchronous
groups matched. Native guard `fp8-host-native1` exec84340/0 passed tiny GPU
8/8, then the original fixed 320-token references with CPU trunk, GPU FP8
experts and `Q38_TIER_RANK_ORDER=1`:

| Reference | CPU NLL | GPU-expert NLL | Original maximum |
| --- | ---: | ---: | ---: |
| Greedy | 0.3030 | 0.3030 | 0.305221 |
| Human text | 2.6323 | 2.6323 | 2.6508261 |

The scores match at the engine's printed precision; no unrecorded exact-logit
claim is implied. GPU hits, native dimensions and host-SiLU/rank-order markers
were verified. Both guards passed independent audits and postflight with all
348 monitored counters unchanged, memory OK, SEL34c, primary services unchanged
and apt timers restored. The complete build, linkage probes, both operator
audits, frozen references, process/memory logs and scores are in
`<ops>/work/models/qwen38/fp8-host-{silu1,linkfix1,gates1,gates2,native1}/`.
Isolated serving was subsequently tested with the exact staged engine
`b19b09fa3b1658970044146211b93dadb7a8205691ed02b056d428d11b302f79`
mounted read-only into the accepted runtime, without changing either primary
gateway. `fp8-host-api2` tiny GPU guard exec91501/0 and independent audit
passed two 32-token completion requests, both matching CPU text and token
usage; nonzero mixed and all-GPU rank masks proved expert execution. Full
native GPU guard `fp8-host-api3` exec17293/0 and independent audit passed raw
completion, reasoning chat and no-thinking chat; all three matched CPU text
and token usage exactly, with the final chat answer `Rome`. The GPU log proved
host SiLU, CPU-order and nonzero GPU expert masks, 0/553 trunk matrices
resident on the GPU and 553 CPU int8 trunk matrices. Peak sampled process
RSS was 38,410,625,024 bytes, with no swap. Both guards passed postflight
with SEL34c, 348 unchanged monitored counters, memory OK, apt timers restored
and primary services unchanged.

Preserved failed harness attempts: `fp8-host-api1` tiny guard rc1 returned
the CPU-matching answer but a four-token request did not prove GPU expert
execution; `fp8-host-api2` native guard rc1 nevertheless returned CPU-matching
raw/chat output, but required a conditional dense-chain-off message that
serving did not emit. The corrected test asserts actual zero GPU trunk
placement and absence of the full GPU chain. The first CPU audit also exposed
an omitted Q38 environment prefix in its metadata; api2/api3 record it. The
`fp8-host-api3/pre-gpu-native.json` inherited a stale `gpu_budget: 2` field;
its explicit phase plan and checkpoint authorized one guard, and only one was
used. Details and original evidence are retained in
`<ops>/work/models/qwen38/fp8-host-api{1,2,3}/`.

These are isolated single-card hybrid functionality and quality checks, not a
performance measurement. The full GPU dense chain, persistent deployment,
other devices and multi-GPU operation remain unqualified; no gateway or image
was promoted.


## 32. GLM MLA attention: tiny-first opt-in investigation (2026-11-09)

The accepted GLM5.3 Light run's **decode-only** software phase counters show
mean attention shares 50.2% for short decode, 52.6% for long decode, and 65.0%
for mixed-long decode; these include projections and RoPE. Expert matmul shares
were 33.4%, 32.0% and 24.0%; expert disk time was zero in those decode windows.
This does not measure prefill or hardware utilization. Frozen source results
and calculation are in `<ops>/work/models/glm/opt-attn1/baseline-profile.json`.

`COLI_CUDA_ATTN=1` is an existing opt-in GPU MLA path that the accepted GLM
baseline did not set. This investigation adds only optional host-side
`COLI_CUDA_ATTN_TRACE=1` messages after successful contiguous and ragged GPU
project calls. No default, dispatch or arithmetic changes. A new independent
small CPU-reference test exercises causal contiguous projection and ragged
cache growth, shrink and host-pointer replacement. The device-free oneAPI build
and host reference passed; CPU tiny off/on teacher-forced 32/32 and greedy
20/20 passed. Separately guarded card1 run `codex-glm53-opt-attn1-gates`
exec74071/0 passed the new GPU test, chain/backend gates and GPU tiny off/on
32/32 and 20/20 oracles. Both opt-in tiny cases logged successful contiguous
GPU projection on prefill. Independent audit passed with SEL34c, 348 fixed
counters unchanged, memory OK, apt timers restored and both primary services
unchanged. Source, staged binaries, fixed inputs, logs and audit are retained in
`<ops>/work/models/glm/opt-attn1/`.

The once-per-path marker in this first phase proves prefill execution; it does
not establish decode execution. Native-shape quality, performance, ragged
serving, long-context behavior and persistent deployment remain unqualified;
no performance claim or default change is made.

The trace was then refined to report one-row and multirow execution
independently for each path, still default off and without changing math or
dispatch. Device-free rebuild and CPU controls passed. Guarded tiny run
`codex-glm53-opt-attn2-gates` exec9721/0 independently passed the same op,
chain/backend and strict tiny oracles; GPU trace proved contiguous prefill
S=12 and single-row S=1 during tiny greedy generation. Audit and postflight
passed (SEL34c, 348 unchanged counters, memory OK, apt restored and primary
services unchanged). See `<ops>/work/models/glm/opt-attn2/`. Native-shape
quality, serving and performance remain unqualified.

A separate native-width operation ramp `opt-attn3` retained tiny-first ordering
and tested synthetic **float32** weights at H64/Q192/R64/V256/K512 with a small
D4 output: contiguous S1/S4/S16 and ragged S2 all passed an independent CPU
reference; relative RMS was 1.26e-6, 2.24e-6, 1.91e-6 and 2.23e-6,
respectively. Chain/backend passed; guard exec86181/0, independent audit and
postflight passed with SEL34c, 348 unchanged counters, memory OK, timers
restored and primary services unchanged. The GLM checkpoint's actual attention
weights are grouped int4 and its output dimension is 6144; neither is covered
by this D4 float test. The original SCORE path uses ~338 rows and bypasses
`COLI_PREFILL_CHUNK`, so native quality must wait for appropriate full-format
and output/row-size coverage. See `<ops>/work/models/glm/opt-attn3/` and
`opt-native1/PLAN.md`; no native model or performance test was run.

Subsequent `opt-attn4` ramp tested synthetic native geometry with grouped
int4/group64 in both MLA and full D6144 output projection. Contiguous S1, S4,
S16, S64 and S338, plus ragged S2, passed; all outputs were finite and six
spaced output coordinates at first/middle/last rows (all rows for S<=4)
matched an independent CPU reference with worst sampled relative RMS 4.72e-7.
This is sampled operation coverage using synthetic weights, not a full model
quality or performance result. Existing tiny, chain and backend tests also
passed. Guard exec72059/0, independent audit and postflight passed with SEL34c,
348 unchanged counters, memory OK, timers restored and primary unchanged.
See `<ops>/work/models/glm/opt-attn4/`.

The first native full-pin SCORE attempt `opt-native1` stopped under its
predeclared zero-swap rule: sampled candidate VmSwap19,734,528 bytes at
RSS365.53GB, before a score. Guard exec99366/1, SEL34c, 348 counters unchanged,
memory OK, snapshot and restoration details are preserved in
`<ops>/work/models/glm/opt-native1/`. Do not repeat its full-pin A/B timing plan
without a separately reviewed safe approach.

A distinct lower-memory original SCORE `opt-native2` used the earlier qualified
RAM_GB80/PIN_GB0/PIN_FILL0 setting, original score and pin, and opt-in GPU MLA.
Guard exec90632/0 and independent audit passed: 320-token NLL **2.7646007625**
<= original CPU-based limit **2.789181351353125**, versus CPU 2.769693496875
and previous GPU 2.76460091875. The real model logged contiguous GPU MLA
S338/H64/K512 and 314 GPU expert requests. Candidate process swap stayed zero;
169 read-only primary swap samples remained at their respective starting
values. SEL34c, 348 counters/memory OK, both primary services unchanged and
apt timers restored. Evidence and memory records are under
`<ops>/work/models/glm/opt-native2/`. This qualifies that quality input under
this hybrid configuration only; it does not measure throughput, qualify the
full-pin setting, or authorize a default change.

A separate low-memory A/B/A/B original SCORE comparison (`opt-prefill1`)
used the same staged engine, input and configuration, toggling only
`COLI_CUDA_ATTN`; every case passed the original NLL gate and independent
hardware/memory audit. For the 338-token input, elapsed time **to the start of
layer 78/78** was A1 99.40s, B1 87.11s, A2 101.59s, B2 87.60s. The off/on
means were 100.495s/87.355s, a 13.08% reduction in this marker; both on
runs were faster than both off runs. The timestamp is printed before layer 78,
so this excludes the last layer, scoring and full prefill completion. It is
not a decode, serving, TTFT, colibench or full-pin benchmark. Both B runs
proved GPU MLA S338; all cases passed NLL (off 2.76460091875, on
2.7646007625), GPU expert work, candidate zero swap, unchanged primary swap,
SEL34c/348 counters/memory OK, and timers restored. Sampled process physical
reads were zero, a sampled lower bound not a per-expert I/O proof. A prior
isolated attention-on quality run recorded 304.57s at the same marker; its
cause is unknown, so do not generalize the later warm sequence. Enabling the
knob also changes the attention algorithm for S338. Frozen commands,
`comparison.json`, audits and I/O/memory logs are in
`<ops>/work/models/glm/opt-prefill1{,-a1,-b1,-a2,-b2}/`. Default remains off.

A distinct single-card low-memory raw-decode A/B/A/B (`opt-decode1`) used the
same staged engine, original 22-token lighthouse prompt, 16 generated greedy
tokens, RAM80/PIN0/cap16, and toggled only GPU attention. All four guard and
independent audits passed; actual GPU experts ran, and B cases proved real
single-row GPU MLA. Decode times A1/B1/A2/B2 were 13.87/9.79/9.75/9.57s,
corresponding to 1.154/1.634/1.641/1.672 tok/s. Ranges overlap, so the test
does **not** establish a decode speedup. The engine reported expert waits of
7.5/4.1/4.0/3.9s and 92.834GB fetched per case; sampled process physical
reads were 8.80GB/0/0/0 (sampled lower bounds, not expert-specific). This
illustrates OS cache and expert-I/O confounding. Software attention times were
2.994/2.785/2.895/2.784s; these are host phase counters, not hardware kernel
profiling. The four GPU outputs matched each other exactly. All differ from
the old CPU smoke text at character 58: CPU continues `tended the`, GPU
continues `been t`; the separate original numeric quality gate passed. Every
case had candidate swap0 and unchanged primary swap, SEL34c, 348 counters and
memory OK, apt timers restored. Original commands, full output, first
text divergence, memory/I/O samples and audits are in
`<ops>/work/models/glm/opt-decode1{,-a1,-b1,-a2,-b2}/`. This is one short,
fresh-process hybrid test, not a fullpin, long-context, serving or colibench
result, and the knob remains default off.

### Qwen38 first hybrid Light baseline (2026-11-09)

After ctx4096 small and raw 574/1691/3049-prompt CPU/GPU correctness gates,
`baseline-light1` completed 20/20 frozen Light tasks (2 warmups, 18 measured)
on card1. It used stock colibench `measure`/`summarize` functions in a bounded
loopback wrapper, not the stock CLI telemetry pipeline. The three-repetition
means ± sample standard deviation [min,max] were decode_short **5.096 ± 0.228
[4.900,5.346] tok/s** and decode_long **5.701 ± 0.310 [5.345,5.901] tok/s**;
mixed_long decode 4.965 ± 0.130 [4.835,5.096] tok/s; long-prefill TTFT mean
286.416s. Record actual prompt/output counts and request logs when comparing.

This is the qualified opt-in hybrid candidate (CPU trunk, GPU FP8 experts,
CPU-order and host SiLU), ctx4096/RAM128/cap128, not an all-GPU chain or a
persistent deployment. Positive GPU masks and host-SiLU markers were verified.
Guard exec23598/0 and independent audit passed, candidate peak sampled RSS
39,162,056,704 bytes with zero swap; primary swap growth CPU7,069,696 and
GPU11,571,200 bytes within user-authorized 512MiB each; SEL34c/348 counters
unchanged/memory OK, apt timers restored. Frozen tasks, raw requests/results,
monitoring and audit live under `<ops>/work/models/qwen38/baseline-light1/`;
the TUNING row is labelled `gpu-sycl-card1-qwen38-light1`.

### DeepSeek V4 Flash REAP first hybrid Light baseline (2026-11-09)

After independent ctx4096 small-API and raw 573/1690/3046-token CPU/GPU
correctness comparisons, `reap-light1-observe1` completed 20/20 frozen Light
requests (two warmups, 18 measured) on card1. It called the stock colibench
`measure`/`summarize` functions in an isolated loopback wrapper; this is not
stock CLI telemetry. All measured responses reached their requested caps.
Three-repetition means ± sample standard deviation [min,max] were:

| Workload | Metric | Result |
|---|---|---|
| decode_short | decode proxy | 0.745 ± 0.005 [0.742,0.750] tok/s |
| decode_long | decode proxy | 0.721 ± 0.001 [0.720,0.722] tok/s |
| mixed_long | decode proxy | 0.636 ± 0.054 [0.573,0.669] tok/s |
| prefill_long | time to first token | 986.934 ± 2.965 [984.749,990.309] s |
| prefill_short | time to first token | 7.564 ± 0.061 [7.506,7.628] s |

This is an isolated CPU-expert/router, GPU-dense hybrid using the accepted
image and CPU-order dense/chunk5/expert1 settings, ctx4096/RAM128/cap1. It is
not an all-GPU path or persistent deployment. The earlier GPU-routed expert
candidate failed its original greedy NLL gate; do not enable it solely to
improve these numbers. The user removed the benchmark's per-gateway swap-growth
cutoff; no global VM settings were changed. Primary CPU/GPU engine VmSwap
maxima were 10,543,747,072/13,195,423,744 bytes from starting values
7,880,478,720/9,852,649,472. Peak sampled candidate RSS was 84,530,880,512
bytes and sampled candidate swap was zero. Guard rc0 and independent audit
passed; SEL34c and 348 fixed counters/memory health were unchanged, and apt
timers restored. The aggregate other-card observer began after the benchmark
started; its valid device-number samples saw no other card2/card4 clients,
but gaps and CPU/RAM/I/O interference prevent an exclusivity claim.

Allowed xpu-smi sysman telemetry retained 14,312 samples with a 15-second
maximum internal gap and samples in all 18 measured request windows. It is
card1-wide, includes the existing gpu-gw and is neither per-process nor EU/OA
efficiency profiling; PCM was not collected. Raw requests, results, audit,
telemetry and shared-host caveats are in
`<ops>/work/models/deepseek/reap-light1-observe1/`; the TUNING row is labelled
`gpu-sycl-card1-reap-light1`.

## 24. DeepSeek V4 Flash: card2 hybrid Light baseline (2026-11-11)

A fresh, independent 20-request Light run completed on card2 (`0000:36:00.0`,
`renderD129`); two warmups and all 18 measured requests succeeded. The run used
CPU expert/router execution with GPU dense execution, ctx4096, RAM224/cap1 and
prefill chunk5. It is a hybrid baseline, not an all-GPU deployment or a
combined two-GPU inference result. Qwen36 `gpu-gw` was concurrently running on
card4 (`0000:72:00.0`), without cross-card copies. Shared CPU, memory and I/O
can influence comparisons with earlier isolated runs.

| Workload | Metric | Mean ± sample SD [min, max] |
|---|---|---|
| decode_short | decode proxy | 0.775 ± 0.008 [0.767, 0.783] tok/s |
| decode_long | decode proxy | 0.753 ± 0.001 [0.752, 0.754] tok/s |
| mixed_long | decode proxy | 0.696 ± 0.001 [0.695, 0.698] tok/s |
| prefill_long | time to first token | 959.410 ± 2.539 [956.697, 961.729] s |
| prefill_short | time to first token | 7.267 ± 0.044 [7.241, 7.318] s |

Each row has three measured samples; none stopped early. The independent audit
verified the 20 frozen requests and outputs, backend execution markers, all 43
GPU-uploaded dense layers, guard rc0, SEL34c and unchanged 348 target counters
and memory health. A first *read-only* audit attempt briefly encountered package
maintenance during timer restoration; it passed after the jobs became inactive,
without another GPU run. Both apt timers were restored.

The host procfs profile contained 6,572 samples, maximum internal gap 2.14 s,
and minimum MemAvailable 353,634,582,528 bytes; the earlier sudden host-memory
decline did not recur. The GPU gateway's swap was observed without a cutoff.
Allowed device-wide xpu-smi sysman telemetry supplied 13,673 samples, with a
30-second maximum internal gap. Sample-weighted GPU utilization was 98.9% in
long-decode windows and 85.9% in long-prefill windows. These figures are not
per-process or EU/OA efficiency measures and do not independently establish the
bottleneck. No PCM was collected.

The user authorized resumption despite the prior fatal BERT event and requested
corrected PCIe reports be recorded without stopping; raw incident evidence is
preserved. No new kernel-line event was recorded for this guard. Full raw
requests, results, audit, profiles and shared-host caveats are under
`<ops>/work/models/deepseek/full-light-card2-corrected1/`; the TUNING label is
`gpu-sycl-card2-full-light-corrected1`.
