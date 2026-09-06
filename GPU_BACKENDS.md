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

## Optional XDNA2 (Ryzen AI NPU) lane

XDNA2 is an *optional compute lane*, not an engine and not a GPU. Colibri keeps
model semantics, routing, expert identity, weight ownership, scheduling and
fallback; the lane executes one already-selected, already-qualified operation
family and nothing else. Nothing about it is required to build or run Colibri.

It is **Windows-only**, **off by default**, and **explicitly requested**. The
qualified operations run in BF16, which is not the arithmetic the normal path
uses, so output may differ.

### What is optional, and how

| concern | answer |
|---|---|
| does ordinary Colibri need XRT? | **no** — no header, no import library, no DLL import |
| does the default build need an XDNA SDK? | **no** |
| does a machine need an NPU? | **no** — absence is a normal machine, not an error |
| does the default path use the lane? | **no** — it is never enabled by discovery |
| where does XRT live? | only inside an optional native helper, `coli_xdna.dll` |
| how is the helper reached? | resolved at runtime by absolute path beside the executable |

`c/backend_xdna.h` and `c/backend_xdna.c` are the host-side owners, and they
never include or link XRT. `XDNA=1` compiles them into the host and defines
`COLI_XDNA`; the Windows release build uses it (`.github/workflows/release.yml`),
because an `XDNA=1` host has the same import table as a default one and starts
normally on a machine with no NPU.

### Turning it on

```
coli run   --xdna --model <glm-model> "..."
coli chat  --xdna --model <glm-model>
coli serve --xdna --model <glm-model>
```

`--xdna` sets `COLI_XDNA=1` in the engine's environment, and the engine enables
the lane when `COLI_XDNA` parses as a non-zero integer. Both halves are worth
stating exactly, because a harness that drives the engine directly uses the
second one:

```
--xdna absent, COLI_XDNA unset       disabled, silent
--xdna present                       coli sets COLI_XDNA=1
COLI_XDNA inherited from the shell   passed through unchanged
COLI_XDNA = 0 / empty / non-numeric  disabled
```

`--xdna` therefore wins when it is present, because it assigns `1` over whatever
was inherited; when it is absent the inherited value stands. `--xdna` lives on
the shared parent parser so it appears on every subcommand, but it reaches an
engine only for GLM models, and only the `colibri` engine is built with the lane.

On success — and only after the package has been resolved and every qualified
artifact verified — the engine prints, to **stderr**:

```
[XDNA] experimental: qualified GLM shared expert operations will use the
[XDNA] native NPU with a reduced-precision BF16 compute path. Model output
[XDNA] may differ from the normal path, and generated text has been observed
[XDNA] to diverge. Operations this lane does not support continue to use the
[XDNA] normal path.
```

The ordering is deliberate. An earlier revision announced the lane as soon as the
request was parsed, and then dispatched zero times because no artifact package
existed — a promise the run did not keep. Anything short of a fully provisioned
lane now gets its own diagnostic and the normal path.

Every `[XDNA]` line goes to stderr, never stdout. SCORE mode writes
machine-readable results to stdout and its parser consumes any line starting with
a digit or a minus sign, so a diagnostic there would be read as a score.

A build without `-DCOLI_XDNA` still answers an explicit request rather than
ignoring it: *"requested, but this build has no XDNA support"*.

### Qualified scope

These are hard gates, not guidance.

```
platform      Windows, AMD XDNA2 (Ryzen AI) NPU
family        MoE shared-expert gate / up  (sh_gate, sh_up)
K             6144
N             2048
stored format fmt=4 grouped int4, group size 64, PAIR nibble layout
logical M     1..64    -> the M64 artifact,  rows zero-padded to 64
              65..256  -> the M256 artifact, rows zero-padded to 256
              >256     -> declines to the current path
execution     blocking, one operation at a time
```

Eligibility is evaluated in this order, cheapest and most semantic first. Every
gate returns; none merely records, so no later gate can excuse an earlier
refusal:

```
1   semantic family — passed in by the call site, never inferred
2   logical M within the range some compiled bucket serves
3   K / N positive
4   stored format is fmt=4
5   group size is the qualified 64
5b  in-memory layout is the pair layout, not K1 planar
6   a qualified artifact exists for exactly this family, bucket, shape and
    dtypes, and its bytes match the SHA256 compiled into the engine
```

There is no bucket between 64 and 256, so an `M=65` operation runs on the M256
artifact with 191 padded rows. Padding is legitimate because `C = A x B` is
row-independent: output row *i* depends only on input row *i* and on B. Only the
logical rows are copied out, and padded rows never reach anything downstream.

The layout constraint is not decoration. `fmt=4` has two in-memory layouts: the
classic **pair** layout (elements `2j` and `2j+1` in byte `j`) and the K1
**planar** layout (elements `k` and `k+32` in byte `k` of each 64-element block),
which `qt_planarize()` writes in place when the grouped planar IDOT path is
opted into with `IDOT_GS=1`. The prepared-weight converter was qualified against
the pair layout only, and planar bytes would decode to plausible-looking nonsense
rather than fail — so a planar tensor is refused (`LAYOUT_UNSUPPORTED`) and runs
the current path. Supporting the planar layout is a separate question with its
own qualification.

`sh_down` is deliberately **not** accelerated: its orientation is `I=2048,
O=6144`, which is not what the qualified artifact computes. Nor is the generic
`matmul_qt` intercepted. The semantic family is passed explicitly by the call
site and never inferred from a shape, so an unrelated operation that happens to
be 6144x2048 cannot inherit this one's qualification.

### Artifact registry

Colibri decides which artifact answers which operation, and whether that
artifact may be trusted. The registry is a table compiled into
`c/backend_xdna.c` carrying, per artifact, the logical filenames for the
`.xclbin` and its instruction stream and the SHA256 of each.

**Integrity is byte identity.** Both files must exist and both SHA256 values must
match, or the operation declines. There is no tolerance and no "close enough".
SHA256 is implemented in `backend_xdna.c` rather than taken from a system library
so that the check has no external dependency.

`STATIC_ARTIFACT_QUALIFIED` means Colibri knows this operation, holds a matching
qualified artifact definition, and the bytes on disk are the bytes that were
qualified. It does **not** mean a device exists, a context can be created, a
weight has been prepared, a pointer is aligned, or that running on the NPU would
be preferable. Helper availability, artifact qualification, device readiness,
prepared-weight validity and economic preference are five independent concepts
and are not collapsed into one flag.

Logical filenames resolve under the package root beside the executable. There is
no PATH search, no current-directory fallback, and a name that tries to escape
its root is rejected during registry validation, before it can reach the
filesystem.

### Prepared host state

The prepared BF16 image is **derived, disposable host state**. The stored fmt=4
tensor stays authoritative and is never replaced, mutated or freed by anything in
the prepared-state path. A prepared image is published only when the whole
conversion succeeded; a failure leaves the previous state untouched rather than
half-written, so partially converted bytes can never reach the device.

The image is aligned for the helper's userptr wrapping, and that alignment is a
hard gate rather than an assumption.

### Failure and fallback

The invariant the lane is built around: **a failure anywhere is the current
path, never a partial result.** The candidate returns "not handled" and
`matmul_qt` computes the answer, exactly as it would have without the lane.

Each provisioning failure has its own diagnostic, because they have different
fixes:

```
PACKAGE_MISSING       the package was not found
PACKAGE_INCOMPLETE    missing <named artifact>
INTEGRITY_FAILED      <named artifact> does not match its expected hash
HELPER_UNAVAILABLE    coli_xdna.dll is not usable beside the executable
REGISTRY_INVALID      this build's artifact registry is invalid
```

The package gate runs **before** the helper is loaded, so unknown or corrupt
bytes never reach execution. Restoring the correct bytes restores the lane with
no residue: no cached verdict, no disabled flag, no retry counter.

### The optional package

The helper and the qualified artifacts are **not** in the core archive. They ship
as an optional sidecar built on a machine that has the XRT SDK and the AIE
toolchain, which the release runners do not:

```
coli_xdna.dll
xdna/wa_F3_M64_K6144_N2048.xclbin
xdna/wa_F3_M64_K6144_N2048_insts.bin
xdna/wa_F3_M256_K6144_N2048.xclbin
xdna/wa_F3_M256_K6144_N2048_insts.bin
```

Unpacked beside the executable. `c/tools/build_xdna_package.py` owns producing
and verifying it, and `docs/xdna.md` carries the user- and maintainer-facing
procedure. XRT and the MSVC redistributable are the user's to install and are
never bundled.

### Building the optional helper

The helper is the only component that links XRT, and it is opt-in: an ordinary
build compiles without XRT headers, links without XRT, imports no XRT DLL, and
does not require `coli_xdna.dll` to exist.

Tested provenance:

```
compiler     MSVC 14.44.35207 (Visual Studio 2022), Windows SDK 10.0.26100
XRT SDK      2.21.75 -- headers + xrt_coreutil.lib
XRT runtime  2.21.0  -- installed xrt_coreutil.dll
device       AMD XDNA2 (Ryzen AI), driver 32.0.20102.3930
```

```
cl /nologo /LD /EHsc /std:c++17 /Zc:__cplusplus /O2 /MD /W3 ^
   /I <XRT_SDK>\include c\backend_xdna_helper.cpp ^
   /Fe:coli_xdna.dll /link <XRT_SDK>\lib\xrt_coreutil.lib
```

`/Zc:__cplusplus` is required: without it MSVC reports `__cplusplus` as
`199711L`, `xrt/detail/any.h` takes its Boost branch, and the build fails on a
header Colibri does not use.

The host looks for `coli_xdna.dll` **beside the executable, by absolute path
only** — no PATH search, no current directory, no application-directory
fallback. A helper that is not there is simply absent, which is a normal state.

The helper ABI is generation **2**, and binding is all-or-nothing: a helper is
usable only when it loads, reports the expected ABI generation, and exports every
entry point this host requires. Generation 1 is refused outright rather than
partially bound — it exports two of the seven entry points generation 2 requires,
so binding it would produce a helper that reports availability and then fails.
The verdict is sticky, so a machine without a helper pays one lookup rather than
one per operation.

### What is not claimed

No full-model acceleration, no token-throughput figure, no general XDNA backend,
no routed-expert support, no concurrency with the GPU, and no scheduler. One
family, one shape, two buckets, one dispatch at a time.

Nothing here chooses between the NPU and the normal path on speed or cost, and
no speed claim is made.

Device execution is qualified against a BF16 oracle. That is **not** a claim that
the lane is numerically interchangeable with `matmul_qt`: the current path
accumulates f32 activations against dequantised int4, the lane is BF16
throughout, and the two legitimately differ. Whether substituting BF16 activation
semantics is acceptable *for a model* is a separate question, and it is not
answered by inventing an elementwise tolerance.

### Validation

Without an NPU, XRT or any hardware — this is what CI runs:

```
python -m unittest test_backend_loader.XdnaLoaderOwnerTest \
                   test_backend_loader.XdnaOptionalBindingTest \
                   test_backend_loader.XdnaDefaultBuildIndependenceTest
python -m unittest tests.test_xdna_package
make tests/test_xdna_registry       && ./tests/test_xdna_registry
make tests/test_xdna_prepared_state && ./tests/test_xdna_prepared_state
make tests/test_xdna_qt_state       && ./tests/test_xdna_qt_state
make tests/test_xdna_execution      && ./tests/test_xdna_execution
make tests/test_xdna_failure        && ./tests/test_xdna_failure
```

The last two are **Windows gates**, and only there: they bind a synthetic helper
DLL through the loader, which is `LoadLibraryExA`. Off Windows the loader answers
ABSENT by design, so there is nothing for them to bind to. The registry, prepared
state and QT state owners have no loader in them and run on every platform.

The prepared-state owner has three cases that ask the allocator for a buffer it
cannot satisfy, to prove the refusal is reported as an allocation failure rather
than an arithmetic one. AddressSanitizer replaces the allocator with one that
treats an oversized request as fatal instead of refusing it, so those three are
skipped under ASan and say so; the arithmetic-overflow cases, which never reach
the allocator, still run there.

The binding contract is qualified against *synthetic* helper DLLs built by the
tests with the same MinGW gcc the loader tests already require. Those fixtures
contain no XRT and do no accelerator work; the contract under test is the
loader's verdict, which is reached long before any real runtime would be. One of
those tests inspects the ordinary `colibri` binary and asserts it imports neither
XRT nor the helper.

With the hardware, the helper and the qualified artifacts, one more owner runs
the production registry, integrity check, loader, weight preparation, eligibility
gates and candidate function against the real device:

```
make tests/xdna_physical_probe
tests/xdna_physical_probe <artifact-root> <path-to-coli_xdna.dll> [M-list]
```

`M-list` defaults to `1,32,64`; pass `65,130,256` to qualify the M256 bucket
through the same binary, or a value above 256 to see the decline.
