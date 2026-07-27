# GPU backends: CUDA and HIP/ROCm

colibrì's GPU expert backend is **one source file** (`c/backend_cuda.cu`) compiled
for either vendor through `c/backend_gpu_compat.h` — the same one-shim-header
pattern `compat.h` uses for the Windows port. Compiled by nvcc the shim is a
pass-through to `cuda_runtime.h` (the NVIDIA path is byte-identical to the
pre-HIP tree); compiled by hipcc it maps the 14-symbol CUDA runtime surface the
backend uses onto HIP 1:1. The kernels use only shared syntax
(`__global__`, `__shared__`, `__syncthreads__`, `<<<>>>`), no vendor intrinsics.

**Rule for contributors:** vendor differences go in `backend_gpu_compat.h`
only — never `#ifdef __HIP__` (or CUDA-specific code) in `backend_cuda.cu`.

## Supported environments

| backend | platform | toolchain | build |
|---|---|---|---|
| CUDA (`CUDA=1`) | Linux x86-64 | CUDA toolkit (nvcc), `CUDA_HOME=/usr/local/cuda` default | `make -C c glm CUDA=1 [CUDA_ARCH=native\|sm_XX]` |
| HIP (`HIP=1`) | Linux x86-64 | ROCm (hipcc), `ROCM_HOME=/opt/rocm` default; tested on ROCm 7.2 | `make -C c glm HIP=1 [HIP_ARCH=native\|gfxXXXX]` |

`CUDA=1` and `HIP=1` are mutually exclusive and both opt-in: the default build
remains pure, dependency-free CPU. Both are refused on non-Linux with an early
`$(error)`. `*_ARCH=native` targets the local GPU; pass an explicit arch when
distributing or on machines with an unsupported iGPU visible to the runtime
(and mask iGPUs at runtime with `HIP_VISIBLE_DEVICES=<ordinal>` on ROCm).

## Runtime configuration (identical for both vendors)

- `COLI_CUDA=1` + `COLI_GPU=N` (or `COLI_GPUS=0,1,...`) — enable, select devices
- `CUDA_EXPERT_GB=G` — VRAM budget for the expert tier (clamped to free VRAM
  minus projected dense set and 2 GB headroom per device)
- `CUDA_RELEASE_HOST=1` — GPU-tier experts drop their host backing after
  upload (default on multi-GPU); combined with `PIN=auto`/`PIN_FILL`, VRAM
  becomes additional pinned capacity at zero RAM cost. The engine
  rematerializes an expert from disk (`expert_host_ensure`) whenever the CPU
  path needs one whose host copy was released — validated under total GPU
  failure below.
- `CUDA_DENSE=1` — experimental resident-dense path (unchanged)
- `COLI_CUDA_TC_W4A16=1` — opt-in W4A16 tensor-core path. **NVIDIA-only**:
  the WMMA kernels are compile-gated (`COLI_GPU_HAS_WMMA` in the compat
  header) because gfx GPUs report `compute_major >= 7` and a runtime check
  alone would select empty kernel bodies under HIP. On AMD, all compute uses
  the portable kernels; rocWMMA matrix-core support is a possible follow-up.
- `COLI_CUDA_WARP_DECODE=1` — opt-in low-row INT4 decode kernel on NVIDIA.
  Maps one output neuron to one 32-lane warp and eight adjacent outputs to
  each block, while preserving the original 256-thread accumulation order.
  This removes block-wide barriers without changing results bit-for-bit.

## Validation

### Unit tests (run on GPU hardware)

```sh
make -C c cuda-test [CUDA_ARCH=...]    # NVIDIA
make -C c hip-test  [HIP_ARCH=...]     # AMD (same test source)
```

Covers q8/q4/q2/f32 matmul correctness, multi-device placement/stats, and
`tensor_update` — the standard upstream suite, unchanged, compiled by hipcc.
(A companion PR adds failure-path tests for the backend; they are
vendor-neutral and run under `hip-test` identically.)

### CI (no GPU required)

The `engine-hip-syntax` job in `.github/workflows/ci.yml` compiles the
backend and its test binary with hipcc (`rocm/dev` container pinned to
`6.2`, `gfx1100`) on every PR, mirroring `engine-cuda-syntax`. Kernel
*execution* is not possible on hosted runners; that is what `hip-test`
on real hardware is for (matrix below).

### Hardware test matrix (documented results)

| environment | result |
|---|---|
| AMD RX 9070 XT (gfx1201), ROCm 7.2.4, Linux 7.0 | `hip-test` **pass** (all cases above); GLM-5.2 end-to-end runs (0.32 tok/s @ 61% expert hit with CUDA_RELEASE_HOST=1); benchmark series in PR #112 |
| NVIDIA | compile-verified in CI (`sm_80`); nvcc path is a pass-through include — **runtime run of `make cuda-test` on NVIDIA hardware welcomed**, the test source is vendor-neutral |

## Known behavior notes

- GPU float matmuls round differently than the CPU int8-dot (IDOT) kernels:
  greedy output is **not token-identical** across backends (consistent with
  the shape-dependence documented in #100), and MTP draft acceptance measures
  lower on GPU-heavy configs (~40% → ~31% on the PR #112 machine). A
  numerics-matched integer GPU kernel is the planned follow-up.
- An earlier revision of this branch carried `CUDA_EXTEND=1` (VRAM tier
  holding experts beyond the RAM pin). It was superseded by upstream's
  `PIN=auto` + `PIN_FILL` + `CUDA_RELEASE_HOST`, which achieve the same
  capacity extension with deeper engine integration; this branch's safety
  and validation work now targets that mechanism.
