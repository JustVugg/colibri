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
| HIP DLL (`HIP_DLL=1`) — build **and** runtime; validated on one configuration | Windows x86-64 | HIP SDK (hipcc) + a compatible MSVC x64 host toolchain; `HIP_SDK_ROOT` from `HIP_PATH` | `make -C c hip-dll HIP_DLL=1 HIP_SDK_ROOT=<sdk-root> HIP_ARCH=gfxNNNN` → `c/coli_hip.dll` |

`CUDA=1` and `HIP=1` are mutually exclusive and both opt-in: the default build
remains pure, dependency-free CPU. Both are **directly linked** paths and remain
Linux-only, refused elsewhere with an early `$(error)`. The Windows HIP path is
selected separately as `HIP_DLL=1` and does not go through `HIP=1`; on Windows
`CUDA_DLL=1` and `HIP_DLL=1` are likewise mutually exclusive. `*_ARCH=native`
targets the local GPU; pass an explicit arch when distributing or on machines
with an unsupported iGPU visible to the runtime (and mask iGPUs at runtime with
`HIP_VISIBLE_DEVICES=<ordinal>` on ROCm). On Windows `HIP_ARCH` **must** be an
explicit `gfxNNNN` — see below.

### Windows HIP DLL

Mirrors the Windows CUDA split: MinGW gcc cannot compile `.cu`, and Windows
hipcc targets the MSVC ABI, so the backend is built into a standalone
`coli_hip.dll` instead of being linked into the host. The same
`c/backend_cuda.cu` and the same `coli_cuda_*` ABI the Linux HIP path already
reuses are used unchanged.

The host loads `coli_hip.dll` at runtime through the same loader seam the
Windows CUDA split uses, and binds the HIP runtime **explicitly**: the directory
holding `amdhip64_7.dll` is named by `COLI_HIP_RUNTIME_DIR`, and the loader
fails closed if the mapped module is not that exact file or if a second module
of the same basename is already present. It never silently accepts a copy from
`System32`, from `PATH`, or from an unrelated ROCm install.

Read [what this was tested on](#windows-hip-limitations) before relying on it;
`docs/windows.md` has the full setup walkthrough.

The two halves are built separately.

*Host build mode* — prepares the host for the DLL split. Needs **no HIP SDK**
and **no `HIP_ARCH`**, because it only compiles `c/backend_loader.c` and links
`colibri.exe`; `amdhip64` is never linked into the host:

```sh
make -C c colibri.exe HIP_DLL=1
```

*Backend DLL* — requires a selected SDK and an explicit architecture:

```sh
make -C c hip-dll \
    HIP_DLL=1 \
    HIP_SDK_ROOT=<sdk-root> \
    HIP_ARCH=gfxNNNN
```

It produces `c/coli_hip.dll`; the linker may also emit `c/coli_hip.lib` (and
`.exp`/`.pdb` on toolchains that generate them). All are ignored by git and
removed by `make -C c clean`.

#### SDK selection variables

No install location is hardcoded. `HIP_SDK_ROOT` defaults from the `HIP_PATH`
environment variable the Windows HIP SDK installer sets; pass it explicitly for
a relocated or source-built SDK (for example from `rocm-sdk path --root`). Every
component can be overridden independently, because packaged layouts do not all
keep runtime, development and device files under one root:

| variable | default | selects |
|---|---|---|
| `HIP_SDK_ROOT` | `$(HIP_PATH)` | SDK root (`--hip-path`) |
| `HIP_BIN_DIR` | `$(HIP_SDK_ROOT)/bin` | directory holding `hipcc` |
| `HIP_INCLUDE_DIR` | `$(HIP_SDK_ROOT)/include` | headers (`-I`) |
| `HIP_LIB_DIR` | `$(HIP_SDK_ROOT)/lib` | `amdhip64` import library (`-L`) |
| `HIP_DEVICE_LIB_PATH` | `$(HIP_SDK_ROOT)/lib/llvm/amdgcn/bitcode` | device bitcode (`--rocm-device-lib-path`) |
| `HIPCC` | `$(HIP_BIN_DIR)/hipcc.exe` | compiler driver |
| `HIP_ARCH` | *(none — must be explicit)* | `--offload-arch=` |

`--hip-path` pins the SDK deliberately: a machine can carry both a
driver-installed HIP SDK and a source-built one, and clang otherwise injects the
former on its own, which silently mixes headers from one tree with the import
library and device bitcode from the other.

`HIP_ARCH=native` is rejected on Windows: `rocm_agent_enumerator` is not
available there, so there is nothing to resolve `native` against.

<a id="windows-hip-limitations"></a>
#### What this was tested on, and what it does not claim

Validated on **one** configuration: AMD Radeon(TM) 8060S Graphics reporting
`gfx1151`, TheRock HIP 7.14.60850, VS2022 MSVC 14.44.35207, Windows SDK
10.0.26100.0. Nothing here is a statement about other GPUs, drivers or SDKs.

- **Hybrid placement.** In the validated workload `CUDA_DENSE=1` put the
  eligible dense tensors on HIP while **routed experts stayed on CPU**. Routed
  experts reach the GPU only through the existing expert-placement controls
  (`CUDA_EXPERT_GB` plus a pin/usage source), which is untested here. This is
  **not** full-GPU MoE inference.
- **Device discovery is not proof of GPU work.** A run can print
  `[CUDA] device 0: ...` and still execute every tensor on CPU. The number that
  settles it is `[CUDA] resident set: N tensors` — `N = 0` means no model tensor
  was resident on the GPU. Check that, not the device line.
- **CPU fallback keeps the command successful.** A per-tensor upload failure
  falls back to CPU and the process still exits 0. Read the first fallback
  diagnostics and the final fallback count together with the resident set.
- **Lifecycle.** Normal one-shot model exits rely on Windows process teardown
  rather than an explicit `coli_cuda_shutdown` call. This matches the existing
  host path and is not presented as explicit backend shutdown.
- There is **no hosted CI coverage** for the Windows HIP build or runtime — no
  hosted runner provides a Windows HIP toolchain or an AMD GPU.
  `engine-hip-syntax` covers the Linux HIP compile only; the Windows job added
  alongside this work runs the loader contract tests with **synthetic DLL
  fixtures and no GPU**.
- Physical validation therefore comes from a local Windows AMD/HIP environment
  and is reported as external evidence, not as CI coverage.

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
