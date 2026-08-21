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

Status: **binding foundation only.** There is no NPU compute path yet.

XDNA2 is an *optional compute lane*, not an engine and not a GPU. Colibri keeps
model semantics, routing, expert identity, weight ownership, scheduling and
fallback; the lane would only ever execute one already-selected, already
qualified operation. Nothing about it is required to build or run Colibri.

### What is optional, and how

| concern | answer |
|---|---|
| does ordinary Colibri need XRT? | **no** — no header, no import library, no DLL import |
| does the default build need an XDNA SDK? | **no** |
| does a machine need an NPU? | **no** — absence is a normal machine, not an error |
| where does XRT live? | only inside an optional native helper, `coli_xdna.dll` |
| how is the helper reached? | resolved at runtime, exactly like `coli_cuda.dll` / `coli_hip.dll` in `backend_loader.c` |

`c/backend_xdna.h` and `c/backend_xdna.c` are the host-side owners. They never
include or link XRT. `colibri` does not link them yet either: there is no caller
until an operation seam lands, so the default binary is byte-for-byte what it
was. `make xdna-obj` compiles the host side on its own.

### What this slice implements

- a versioned C ABI boundary (`COLI_XDNA_ABI_VERSION`) to an optional helper
- runtime discovery and binding of `coli_xdna.dll`, looked up by absolute path
  beside the executable — no PATH search, no current-directory search
- **all-or-nothing binding**: a helper is usable only when it loads, reports the
  expected ABI generation, and exports every entry point this host requires. Any
  failure leaves nothing callable behind
- a sticky verdict, so a machine without a helper pays one lookup, not one per
  operation
- safe shutdown: before probing, after a failed probe, after a good bind, and
  repeatedly

A successful bind means `HELPER_ABI_AVAILABLE` and nothing more.

### What this slice does NOT implement

Device discovery, XRT initialization, artifact registry, `.xclbin` loading,
weight preparation, buffer wrapping, dispatch, matmul interception, scheduling
policy, and any user-facing switch. There is deliberately no `--xdna` flag and
no `COLI_XDNA*` environment variable: the lane has no operation to offer yet, so
advertising a control would be premature.

**No real XRT-linked helper is built by this slice.** The XRT-owning helper
source is not compiled here, because binding is fully qualified without it (see
below) and dead XRT-linked code would contaminate the build for no gain.

### Validation (no NPU, no XRT, no hardware)

The whole binding contract is qualified against *synthetic* helper DLLs built by
the tests with the same MinGW gcc the loader tests already require. Those
fixtures contain no XRT and do no accelerator work; the contract under test is
the loader's verdict, which is reached long before any real runtime would be.

```
python -m unittest test_backend_loader.XdnaLoaderOwnerTest \
                   test_backend_loader.XdnaOptionalBindingTest \
                   test_backend_loader.XdnaDefaultBuildIndependenceTest
```

Covered: helper absent, good helper, wrong ABI generation, missing required
entry point, present-but-unloadable helper (a deleted dependency), repeated
probes, the three shutdown orders, and a dependency inspection asserting the
ordinary `colibri` binary imports neither XRT nor the helper.

### Artifact registry (engine-owned)

Colibri decides which artifact answers which operation, and whether that artifact
can be trusted. The helper decides neither, and never sees a choice it did not
receive.

A registry row records what the research programme established about one
artifact: its semantic family, the exact M bucket it was compiled for, K and N,
the activation/prepared-weight/output dtypes, the target device family, logical
filenames for the `.xclbin` and its instruction stream, the SHA256 of each, and
four independent qualification facts.

**Lookup requires an explicit semantic family.** Two operations with identical
M/K/N are still different operations, and one may never inherit the other's
qualification. Shape alone is not eligibility.

**Buckets are exact.** A row describes the M the program was compiled and
qualified for. Nothing interpolates between qualified buckets.

**Integrity is byte identity.** Both files must exist and both SHA256 values must
match before an artifact is usable. Presence and integrity are separate verdicts:
a missing file means this build does not ship that artifact, a hash mismatch
means the bytes are not the bytes that were qualified.

**Compiling is not qualifying.** Four research facts are required — runtime
weight, correctness, userptr and structural — and they are checked *before* the
filesystem, so a row that was never correctness-qualified declines whether or not
its bytes are present and intact. This matters: research measured a design that
compiled, loaded, dispatched to completion, returned finite numbers and was
numerically wrong. A successful dispatch is not evidence of a correct one.

SHA256 is implemented in `backend_xdna.c` rather than taken from a system
library, so the registry stays portable and the host acquires no new link
dependency. It is covered by the standard NIST vectors.

### What STATIC_ARTIFACT_QUALIFIED means

It means Colibri knows this operation, holds a matching qualified artifact
definition, and the bytes on disk are the bytes that were qualified.

It does **not** mean a device exists, a context can be created, a weight has been
prepared, a pointer is aligned, memory is available, or that running on the NPU
would be preferable. Those gates do not exist yet. Helper availability, artifact
qualification, device readiness, prepared-weight validity and economic preference
are five independent concepts and are not collapsed into one flag.

### Artifacts are not shipped

The registry names the qualified artifacts for the first family; the build does
not contain them. Where they should live is a packaging decision that has not
been made. An absent artifact yields `ARTIFACT_UNAVAILABLE` and the operation
continues on its current path — that is the intended state, not an error, and no
startup path can fail because of it.

Logical filenames resolve under a caller-supplied root. There is no PATH search,
no current-directory fallback, and a name that tries to escape its root is
rejected during registry validation, before it can reach the filesystem.

### Prepared host state (engine-owned)

The prepared BF16 image is **derived, disposable host state**. The stored fmt=4
tensor stays authoritative and is never replaced, mutated or freed by anything in
the prepared-state path.

Three properties vary independently and are never collapsed into one flag:

| property | question |
|---|---|
| allocation | is a buffer held? |
| validity | are its contents usable? |
| consumption | how many host bytes does it cost? |

An invalid buffer still costs memory, and no amount of successful allocation
makes contents valid.

```
UNPREPARED ──begin──> PREPARING ──publish success──> PREPARED_VALID
                          │                                │
                          └──publish failure──> PREPARED_INVALID <──invalidate──┘
```

`PREPARED_VALID` is reachable **only** through an explicit success publication
after `PREPARING`. An `INVALID` image can never shortcut back to valid: it must
go through a complete new cycle. A writable destination exists only while
`PREPARING`, so a published image cannot be rewritten behind its own back.

Allocation is 4096-byte aligned, reusing the repository's existing
`posix_memalign` / `compat_aligned_free` pair from `c/compat.h`. The **payload
size need not be a page multiple** and is never rounded up behind the caller: a
30-byte payload reports 30 bytes and still gets an aligned pointer. Sizes are
computed with checked arithmetic, so a product that would exceed `SIZE_MAX`
is refused rather than wrapping into a small, plausible and far-too-small
allocation.

A defensive alignment validator exists alongside the allocator guarantee,
because a buffer that arrives from a pool or at an offset does not carry that
guarantee — and a misaligned pointer fails at the XRT boundary with a message
about video memory, which points at entirely the wrong subsystem.

Bytes are **host memory**. They are not VRAM, not NPU memory and not an XRT
device allocation, and they are not reported to any GPU accounting.

### What PREPARED_VALID does not mean

It means the derived host image was published successfully by its producer. It
does **not** mean a device exists, a pointer has been wrapped, an artifact is
loaded, or an operation can run. No XRT call, no helper call and no device open
is involved anywhere in this path — the whole contract is exercised with the
helper absent.

Prepared state attaches to a tensor as a single opaque pointer that starts
`NULL`. Loading a model allocates none of it; a registry query allocates none of
it; only an explicit preparation request does. When an expert slot is reused for
a different expert the derived image is dropped, exactly as the GPU tiers drop
theirs — a stale prepared image would be a silently wrong weight rather than a
missing one.

There is still **no fmt4 → BF16 conversion**: filling the destination is the next
slice's work.

### fmt4 to BF16 preparation

The converter turns the authoritative fmt=4 grouped-int4 weight into the prepared
BF16 image, writing straight into the aligned destination. The source is read and
never modified.

Semantics are taken from the production kernel (`quant.h`, `matmul_i4_grouped`),
not from a description of it:

```
rb    = (I+1)/2                       bytes per output row
ng    = (I+gs-1)/gs                   groups per output row
row   = q4    + o*rb                  weights are stored [O][I]
scl   = scale + o*ng
byte  = row[i>>1]
nib   = (i&1) ? byte>>4 : byte&0x0F   even i low nibble, odd i high nibble
value = (nib - 8) * scl[i/gs]
```

The destination is BF16 `B[K,N]` with `K = I` and `N = O`, so each value is
written to `dst[i*O + o]`. The `[O,I]` to `[I,O]` transform is inherent in that
addressing and needs no intermediate matrix: values pass through a single float
scalar, so **no full-sized FP32 image is ever allocated**. The only allocation a
conversion makes is the BF16 destination itself.

Float to BF16 uses round-to-nearest-even, not truncation. The two differ on
exact ties, and the qualified image rounds.

### Publication and failure

```
validate source -> begin -> PREPARING -> convert -> publish success -> PREPARED_VALID
                                              \
                                               -> publish failure -> PREPARED_INVALID
```

Everything that can be rejected is rejected *before* the cycle opens, so a bad
source never strands an object in `PREPARING`, and no path returns while still
`PREPARING`.

A mid-conversion failure leaves genuinely partial bytes — some elements
converted, the rest whatever they held — and forces `PREPARED_INVALID`. Those
bytes carry no authority: state decides validity, not contents, and the partial
image cannot be published. The authoritative fmt=4 weight is untouched by any of
this.

A failed buffer keeps its capacity and can be reused, but only through a
**complete** re-preparation: `INVALID -> PREPARING -> VALID`, never a shortcut.
A re-prepared image is bit-identical to one prepared from scratch.

The conversion is currently serial. The frozen contract permits parallelisation
but does not require it, and correctness and publication semantics come first; a
6144x2048 weight converts in well under a tenth of a second.

Conversion needs no helper, no artifact, no XRT and no device — it is engine
representation logic, and the helper knows nothing about nibble packing or group
scales. 
### Native execution: the GLM shared gate/up lane

Colibri can now execute one real operation on the NPU: the **MoE shared-expert
gate and up projections**, and nothing else. This is the whole supported surface,
and the constraints below are hard gates rather than guidance.

```
family        MoE shared-expert gate / up  (sh_gate, sh_up)
K             6144
N             2048
stored format fmt=4 grouped int4, group size 64, PAIR nibble layout
logical M     1..64, zero-padded to the artifact's M=64
execution     blocking, one operation at a time
```

The layout constraint is not decoration. `fmt=4` has two in-memory layouts: the
classic **pair** layout (elements `2j` and `2j+1` in byte `j`) and the K1
**planar** layout (elements `k` and `k+32` in byte `k` of each 64-element block),
which `qt_planarize()` writes in place when the grouped planar IDOT path is
opted into with `IDOT_GS=1`. The prepared-weight converter was qualified against
the pair layout only, and planar bytes would decode to plausible-looking nonsense
rather than fail — so a planar tensor is refused (`LAYOUT_UNSUPPORTED`) and falls
back to the current path. Supporting the planar layout is a separate question
with its own qualification.

`sh_down` is deliberately **not** accelerated: its orientation is `I=2048,
O=6144`, which is not what the qualified artifact computes. Nor is the generic
`matmul_qt` intercepted. The semantic family is passed explicitly by the call
site and never inferred from a shape, so an unrelated operation that happens to
be 6144x2048 cannot inherit this one's qualification.

Row padding is legitimate because `C = A x B` is row-independent: output row *i*
depends only on input row *i* and on B. Only the logical rows are copied out;
padded rows never reach anything downstream. Logical M above 64 declines to the
current path — row tiling is a qualified strategy but is not implemented here.

**There is no automatic selection.** No `--xdna` flag, no `COLI_XDNA`
environment variable and no economic policy exists. An ordinary build with a
helper, a device and valid artifacts present still runs exactly the path it runs
today; the lane is reachable only from an internal test control. Deciding *when*
XDNA is preferable is a separate concern with a separate owner.

Hard eligibility is evaluated in this order, cheapest and most semantic first,
and no later gate can excuse an earlier refusal:

```
family -> logical M -> K/N -> fmt -> group size -> byte layout -> registry row
       -> artifact qualification -> artifact present -> artifact SHA256
       -> helper ABI -> prepared weight VALID -> 4096-byte alignment
       -> device/runtime -> artifact runtime object -> userptr wrap -> execute
```

Any refusal, and any helper failure at any stage, returns "not handled" and the
caller runs its current `matmul_qt` path. The candidate never calls `matmul_qt`
itself, so there is no recursion and no double dispatch, and the caller's output
buffer is written only after the helper reports successful completion — a
failure cannot leave a half-written result behind.

### Building the optional helper

The helper is the only component that links XRT, and it is **opt-in**: an
ordinary build compiles without XRT headers, links without XRT, imports no XRT
DLL, and does not require `coli_xdna.dll` to exist.

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

The helper ABI is generation **2**. Generation 1 is refused outright rather than
partially bound: it exports two of the seven entry points generation 2 requires
and cannot execute anything, so binding it would produce a helper that reports
availability and then fails. Binding is all-or-nothing.

### Artifacts are not shipped

The qualified `.xclbin` and instruction-stream bytes are **not** distributed with
Colibri, and how they eventually should be is still an open question. The
registry carries their SHA256, and a missing, unreadable or hash-mismatched
artifact fails closed before the device is opened or any byte reaches the
helper. With no artifact root configured — the default, and the only production
value — every request declines.

### What is not claimed

No full-model acceleration, no token-throughput figure, no general XDNA backend,
no routed-expert support, no concurrency with the GPU, and no scheduler. One
family, one shape, one bucket, one dispatch at a time.

### Failure and fallback

The invariant the lane is built around:

> **Optional XDNA work may fail. Current Colibri operation semantics may not.**

Every decline and every runtime failure ends the same way: the candidate returns
"not handled" and the caller runs the exact `matmul_qt` call that stood at that
site before the lane existed. There is no special CPU fallback, no alternative
dequantisation path, no partial-result salvage, and no helper-side recovery. The
candidate never calls `matmul_qt` itself, so the call graph stays acyclic and no
operation can be computed twice.

Failure stages are classified rather than collapsed, because they call for
different actions:

```
DECLINED                      not eligible -- never an error
HELPER_UNAVAILABLE            absent, or would not load
HELPER_ABI_INCOMPATIBLE       loaded, wrong generation or incomplete
ARTIFACT_UNAVAILABLE          this build does not ship those bytes
ARTIFACT_INTEGRITY_FAILED     the bytes are not the bytes that were qualified
ARTIFACT_UNQUALIFIED          a known artifact that was never correctness-qualified
LAYOUT_UNSUPPORTED            fmt=4 bytes are in the K1 planar layout, not pairs
REGISTRY_INVALID              the registry itself is malformed
WEIGHT_PREPARE_FAILED         fmt4 to BF16 conversion failed
PREPARED_INVALID              no usable prepared image
ALIGNMENT_INVALID             prepared pointer not 4096-aligned
DEVICE_INIT_FAILED            the device would not initialise
ARTIFACT_OPEN_FAILED          verified bytes, but the runtime would not open them
WEIGHT_WRAP_FAILED            the userptr wrap was refused
EXECUTE_FAILED                dispatch refused or threw
COMPLETION_FAILED             dispatched, did not complete cleanly
```

An absent artifact and a *tampered* one are deliberately different verdicts. So
are a missing helper and an incompatible one.

### Output validity is a state, not a measurement

XDNA output is valid **only** after the helper reports successful completion.
A failure at any stage — including one that occurs after the helper has already
written a full, finite, entirely plausible result — leaves the output invalid,
and the caller's buffer untouched. Nothing about the bytes themselves can raise
that verdict: there is no NaN scan, no finiteness check and no plausibility
heuristic anywhere in the lane, because none of them would be evidence.

Structurally, the caller's output buffer is never passed to the helper at all.
The helper writes into a lane-owned staging buffer, and only a successful
completion causes the logical rows to be copied out. A late failure therefore
cannot leave a half-written result behind even in principle.

### Lane health

The narrowest model the observed failures justify:

| failure | scope |
|---|---|
| helper absent / unloadable / ABI mismatch | sticky loader verdict, never retried |
| **device init** | **process-scoped** — the lane is marked unavailable |
| artifact runtime open | one shape; the lane stays healthy |
| weight wrap, dispatch, completion | one operation; the lane stays healthy |

There is no retry, backoff or quarantine policy beyond this. A full
`coli_xdna_execution_shutdown()` is the only thing that clears an unavailable
lane, because a fresh attempt after a complete teardown is meaningful and a
fresh attempt on every operation is not.

### Prepared state survives runtime failure

Whether the prepared BF16 image is **correct** and whether the runtime
**succeeded** are separate facts, and a runtime failure does not invalidate a
correct image. A wrap, dispatch or completion failure leaves the prepared weight
`PREPARED_VALID` and reusable; only a conversion failure produces
`PREPARED_INVALID`. This matters for any future cache policy, which would
otherwise throw away good work on unrelated news.

### Userptr wrapper lifetime

The helper-owned wrapper is **persistent runtime state**: it is created on
demand, reused across operations, and released when it is replaced, when the
engine frees or invalidates the memory it borrows, or at shutdown.

It is keyed on **(pointer, publication generation)**, not on the pointer alone.
Retained prepared capacity is reused in place, so a different weight can occupy
the same address — and because the helper snapshots at wrap time via
`sync(BO_TO_DEVICE)`, keying on the address alone left the device computing
against a view that no longer matched the engine's image. That was measured on
real XDNA2 hardware. Every publication now bumps a generation, and the engine
tells the lane to release a wrapper before freeing or invalidating the memory it
borrows, so a wrapper can never outlive or alias released engine memory.

### Two internal modes, neither of them public

```
AUTO-LIKE   coli_xdna_try_matmul()    decline or failure -> current path
EXPLICIT    coli_xdna_test_attempt()  decline or failure -> classified failure,
                                      no fallback, no output claimed
```

Both share one implementation so their gate order and classification cannot
drift apart. Explicit mode is a separate entry point rather than a mode flag, so
no global state can leave the production seam in a no-fallback configuration.

**Default behaviour is unchanged and remains so.** With a helper present, a
device available and valid artifacts staged, an ordinary build still runs the
current path and dispatches zero XDNA operations. Automatic selection needs an
economic policy that does not exist, and a semantic qualification that has not
been done — see below.

### What successful XDNA execution does and does not mean

```
SUCCESSFUL_XDNA_CORRECTNESS_OWNER  = the qualified BF16 oracle
FAILED_XDNA_FALLBACK_CORRECTNESS_OWNER = the current matmul_qt path
```

These are different contracts and must not be conflated. Device execution is
qualified against a BF16 oracle under a criterion frozen by the research
programme. It is **not** a claim that the lane is numerically interchangeable
with `matmul_qt`: the current path accumulates f32 activations against
dequantised int4, the lane is BF16 throughout, and the two legitimately differ.

```
MODEL_LEVEL_BF16_REPLACEMENT_ACCEPTABILITY = NOT YET QUALIFIED
```

Whether substituting BF16 activation semantics is acceptable *for a model* is an
open question, and answering it is a prerequisite for any automatic selection.
It is not answered by inventing an elementwise tolerance.

### A note on the registry

The compiled-in production registry is now covered by a regression that reads it
with no test rows installed. An earlier revision initialised its row count to
zero, leaving the table present but empty; every caller at the time was a test
that installed its own registry, so nothing noticed until the first production
consumer arrived.
