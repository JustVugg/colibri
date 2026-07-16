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

## Validation

### Unit tests (run on GPU hardware)

```sh
make -C c cuda-test [CUDA_ARCH=...]    # NVIDIA
make -C c hip-test  [HIP_ARCH=...]     # AMD (same test source)
```

Covers: q8/q4/q2/f32 matmul correctness; multi-device placement and stats
accounting; cached tensors remaining callable **without live host pointers**,
including 64× sustained reuse and upload-from-a-freed-temporary (the VRAM-only
slot lifecycle); graceful upload failures (invalid device/format, missing
scales, missing host data, ~16 TB allocation) with stats integrity; and the
`COLI_GPU_FAIL_AFTER` fault-injection hook.

These failure-path tests caught (and the branch fixes) a latent bug: a failed
allocation left a sticky runtime error that poisoned the next healthy launch's
`cudaGetLastError()` check.

### CI (no GPU required)

`.github/workflows/gpu-build.yml` compile-verifies the shared source under
**both** toolchains on every push/PR — nvcc (`nvidia/cuda` container,
`sm_80`) and hipcc (`rocm/dev` container, `gfx1100`) — building the backend,
the test binary, and the full engine link, plus the standard CPU `make check`.
Kernel *execution* is not possible on hosted runners; that's what the unit
tests above and the hardware matrix below are for.

### Engine-level fault injection (host-released repair path)

`COLI_GPU_FAIL_AFTER=N` makes every GPU *compute* entry point (matmul, fused
expert MLP, grouped experts, shared MLP, the attention pipeline ops) report
failure after N successful calls; uploads and queries are not gated. With
`N=0` (every GPU call fails) and `CUDA_RELEASE_HOST=1`, the validated
properties on GLM-5.2 (RX 9070 XT, greedy, `DRAFT=0`) are:

- **No crash, no corruption**: every host-released expert is rematerialized
  from disk (`expert_host_ensure`) before the CPU fallback touches it; the
  run completes with coherent output.
- **Deterministic**: two total-failure runs from the same `.coli_usage`
  snapshot are byte-identical. (Across un-frozen runs, outputs vary because
  the learning cache evolves pin membership between runs — by design.)
- **Speculation disengages losslessly** under total GPU failure (0 MTP
  proposals, plain decode) — a graceful degradation, but note the perf
  cliff: a failing GPU also costs the MTP speedup.
- **Not byte-identical to a pure-CPU run**: the GPU-enabled engine's CPU
  fallbacks select different kernel shapes/paths (grouped experts, attention
  pipeline fallbacks) than the CPU-only flow, and per #100 the kernels are
  shape-dependent. This is a reproducible numerics identity limit of kernel
  selection, not a correctness defect; both outputs are stable and coherent.

```sh
# repeatability: snapshot .coli_usage, then run twice and cmp
cp <model>/.coli_usage /tmp/u; \
COLI_CUDA=1 COLI_GPU=0 CUDA_EXPERT_GB=12 CUDA_RELEASE_HOST=1 \
COLI_GPU_FAIL_AFTER=0 DRAFT=0 ./coli run "<prompt>" --temp 0   # x2, restoring /tmp/u between runs
```

### Hardware test matrix (documented results)

| environment | result |
|---|---|
| AMD RX 9070 XT (gfx1201), ROCm 7.2.4, Linux 7.0 | `hip-test` **pass** (all cases above); engine-level fault-injection validation **pass** (byte-identical repair, mid-run survival); 10-run GLM-5.2 benchmark series in PR #112 |
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
