# Windows 11 native install — a complete walkthrough (no WSL)

A start-to-finish, reproducible path from a fresh Windows 11 machine to GLM-5.2 generating tokens, with the GPU tier. Every step and every failure mode below was hit and verified on real hardware: Core Ultra 9 285K (AVX-VNNI) / RTX 5080 (sm_120) / 128 GB RAM / Windows 11 24H2 (issue #306). Steps are ordered so the long downloads run while you build.

---
> **2026-08-11: Additional validation with detailed steps for laptop setup** \
Lenovo Thinkpad P16v (Intel Core i7 ultra (155H 1.4GHz), 64GB RAM, 2Tb NVME drive, Nvidia RTX 2000 Ada generation 8GB (AD107, 2023)).
Windows 11 pro english.\
**Software installs:**\
`- nvidia drivers` from https://www.nvidia.com/en-us/drivers/\
`- nvidia cuda toolkit` from https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local\
`- msys2` from https://github.com/msys2/msys2-installer/releases/download/2026-06-11/msys2-x86_64-20260611.exe\
`- Microsoft Visual Studio 2022 built tools` installer from https://aka.ms/vs/17/release/vs_buildtools.exe then install Desktop Development with C++\
`- winget install git.git python.python.3.14`\
\
`msys2` is installed in `c:\msys64` with its main launchers. (we will use `C:\msys64\mingw64.exe`)\
Its executables tools `sh`, `bash`, `make`, `sed` etc go in `c:\msys64\usr\bin`. \
From within `mingw64` we need to install compiler and make. \
\
`C:\msys64\mingw64.exe`\
`pacman -S --needed mingw-w64-x86_64-gcc make`\
\
To see which AI oriented instructions your processor has, run CPU-Z from `https://www.cpuid.com/softwares/cpu-z.html` \
or use the following command from msys2: `cat /proc/cpuinfo | grep ^flags | head -n 1`\
To see which code optimizations are active in gcc compiler with the native flag check:\
`gcc -march=native -dM -Q --help=target | grep -E "m(avx|ssw|aes|fma|sha|mmx)"`\
For guidance about gcc optimizations, a good source cab be: `https://wiki.gentoo.org/wiki/GCC_optimization`\
\
**colibri.exe and coli_cuda.*** build from MS Visual Studio and Nvidia CUDA Toolkit:\
Microsoft Visual Studio tools includes a `vcvars64.bat` batch file that appropriately sets all the paths. \
It is in `"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"`\
\
CUDA compilation is to be performed from a CMD shell, where we run `vcvars64` and then the path \
extension for mingw64 build tools, as required by the make process (mingw64 `make` is used). \
Open a command prompt shell (`cmd`, not powershell), cd to subfolder `c\` of the cloned repo and run:\
\
`%comspec% /k "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"`
`set PATH=%PATH%;C:\msys64\usr\bin`\
`cd C:\Users\YOUR_USER\COLIBRI_REPO_FOLDER\c`\
`make cuda-dll CUDA-ARCH=sm_89`\
`make colibri.exe CUDA_DLL=1 ARCH=native`\
`make iobench.exe`\
\
Replace `sm_89` with your Nvidia GPU architecture according to this table

| Architecture | Example GPUs / Products | Compute Capability | `nvcc` Flag (`-arch=sm_XX`) |
| :--- | :--- | :--- | :--- |
| **Blackwell** | B100, B200, RTX 50x0 | 10.0, 12.0 | `sm_100`, `sm_120` |
| **Hopper** | H100, H200, GH200 | 9.0 | `sm_90` / `sm_90a` |
| **Ada Lovelace** | RTX 40x0, RTX 2000, L4, L40 | 8.9 | `sm_89` |
| **Ampere** | A100, RTX 30x0, A10, Orin | 8.0, 8.6, 8.7 | `sm_80`, `sm_86`, `sm_87` |
| **Turing** | RTX 2080, GTX 1660 Ti, T4 | 7.5 | `sm_75` |
| **Volta** | V100, Titan V, Xavier | 7.0, 7.2 | `sm_70`, `sm_72` |
| **Pascal** | P100, GTX 1080 Ti, P40 | 6.0, 6.1, 6.2 | `sm_60`, `sm_61`, `sm_62` |
| **Maxwell** | M60, GTX 980, GTX TITAN X | 5.0, 5.2, 5.3 | `sm_50`, `sm_52`, `sm_53` |

> The last make links `colibri.exe` with the newly built `coli_cuda.dll`\
to just build `colibri.exe` with no nvidia CUDA support omit `CUDA_DLL=1`\
`ARCH=native` ensures that colibri is build with optimizations specific for your CPU.\
\
**caveat**: to be sure the correct `make` is used, run `where.exe make` after path extension.\
after compilation you should have `colibri.exe coli_cuda.lib, coli_cuda.exp, coli_cuda.dll` files.
---




## 0. What you need

| Piece | Why | Get it |
|---|---|---|
| git, Python 3 | clone + `coli` launcher | winget / python.org |
| MinGW-w64 gcc + make | builds the engine (MSVC can't) | `scoop install mingw-winlibs`, MSYS2, or portable **w64devkit** (no admin, unzip and go) |

> **scoop MinGW caveat (#478):** `scoop install mingw-winlibs` ships `gcc` + `make` but **no `sh.exe`** — the Makefile's recipes use POSIX shell idioms (`command -v`, `{ ...; }`, redirect to `/dev/null`) that GNU make runs through `/bin/sh`. Without sh.exe on PATH, make falls back to `cmd.exe` and the build fails with `'printf' is not recognized` / `The system cannot find the path specified`. Two fixes: install **MSYS2** (recommended — it's what the recipes target), or `set PATH=%PATH%;C:\msys64\usr\bin` in any shell you build from. The portable **w64devkit** bundle includes sh.exe and works as-is.
| CUDA Toolkit ≥ 12.8 | GPU tier; ≥12.8 required for Blackwell/sm_120 | `winget install Nvidia.CUDA` |
| MSVC Build Tools (C++ workload) | nvcc's host compiler for the CUDA DLL | `winget install Microsoft.VisualStudio.2022.BuildTools` + "Desktop development with C++" |
| ~400 GB free on a local NVMe | the int4 model (~370–384 GB) | NTFS is fine; **never** a network mount |

RAM: 16 GB minimum, more = bigger expert cache = faster. The build itself needs none of the CUDA/MSVC pieces — do the CPU build first, add the GPU tier later.

## 1. Start the model download first (it's the long pole)

```powershell
python -m pip install -U "huggingface_hub[hf_transfer]"
$env:HF_HUB_ENABLE_HF_TRANSFER = "1"
hf download <model-repo> --local-dir D:\glm52_i4
```

Use the container recommended in the README (with **int8 MTP heads** — int4 heads silently give 0% draft acceptance). The download is resumable: if it stops, rerun the same command. Expect hours; everything below fits inside them.

## 2. Build the engine (CPU)

From a normal PowerShell, in the repo's `c\` directory:

```powershell
make colibri.exe ARCH=native      # ARCH=native unlocks AVX-VNNI on Alder Lake+/Arrow Lake
make iobench.exe              # disk benchmark, useful before committing to the download
```

Warnings about `#pragma comment` and unused variables are normal (MSVC-isms gcc ignores). The engine banner should print `idot: avx-vnni` on VNNI-capable CPUs — if it says avx2, you built without `ARCH=native`.

### ⚠️ Smart App Control will block your fresh binary

On Windows 11 machines with **Smart App Control** enforced (`VerifiedAndReputablePolicyState = 1`), running your self-compiled `colibri.exe` fails with:

```
Program 'colibri.exe' failed to run: An Application Control policy has blocked this file
```

This is not Defender and not Mark-of-the-Web — SAC blocks *all* unsigned, unknown binaries, which includes anything you compile yourself. **Fix:** Windows Security → App & browser control → Smart App Control settings → **Off**, then **reboot** (the policy only reloads on restart). Note SAC is one-way: re-enabling later requires resetting Windows. If the settings page is missing, the registry equivalent is setting `HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy\VerifiedAndReputablePolicyState` to `0` (admin PowerShell), then rebooting. Check your current state before touching anything:

```powershell
(Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy").VerifiedAndReputablePolicyState
# 0 = off, 1 = enforced, 2 = evaluation
```

## 3. Build the CUDA DLL (GPU tier)

nvcc needs MSVC as host compiler, so this one step must run from a shell with the MSVC environment: open **"x64 Native Tools Command Prompt for VS 2022"** from the Start menu (plain PowerShell will fail the `cl` check). Then:

> **The VS prompt has no `sh.exe` (#478):** that prompt is a `cmd.exe` shell, and the `cuda-dll` recipe uses POSIX idioms (`command -v`, `{ ...; }`) that need `/bin/sh`. Run this once in the VS prompt before building:
> ```cmd
> set PATH=%PATH%;C:\msys64\usr\bin
> ```
> (adjust the path if you installed MSYS2 elsewhere). If you skipped MSYS2 in favor of w64devkit or scoop MinGW, point this at wherever `sh.exe` lives.

```cmd
make cuda-dll CUDA_ARCH=sm_120        # match your GPU: sm_120 Blackwell, sm_89 Ada, ...
make colibri.exe CUDA_DLL=1 ARCH=native   # relink host with the runtime loader
```

Two pitfalls, both fixed on current `dev` (#314) but worth knowing on older checkouts:

- **Spaces in `CUDA_HOME`** (`C:\Program Files\...`) used to break the recipe → fixed; nvcc now comes from PATH and `"$(NVCC)"` is quoted.
- **`make colibri.exe CUDA_DLL=1` after a CPU-only build** used to report `up to date` and silently keep the CPU-only binary (GPU tier never engages, no error). Current `dev` has a build-config stamp that forces the relink. On older trees: delete the binary (`colibri.exe`; `glm.exe` pre-rename) first.

Sanity check: first GPU run should print `[CUDA] device 0: <your GPU>, ... sm_XX` and `[CUDA] mode: routed experts + resident dense tensors`.

## 4. First run

```powershell
cd <repo>\c
$env:OMP_NUM_THREADS = "<physical cores>"
python coli run "Explain what a mixture-of-experts model is." --model D:\glm52_i4 --ngen 48
```

The first run is cold — expect the profile to be dominated by `expert-disk` while the cache warms; hit rate climbs run over run. GPU tier on top:

```powershell
$env:COLI_CUDA="1"; $env:COLI_GPU="0"; $env:CUDA_DENSE="1"; $env:CUDA_EXPERT_GB="4"
python coli run "..." --model D:\glm52_i4 --ngen 64
```

Size `CUDA_EXPERT_GB` so dense (~10 GB) + experts + working set stays under your VRAM. Note MTP speculation is off by default under CUDA (#293, float-accumulation divergence between draft and verify) — `COLI_CUDA_MTP=1` opts back in.

## 5. Reference numbers from this walkthrough's hardware

285K / RTX 5080 / 128 GB / NVMe at 5.85 GB/s random-read (19 MB blocks, `iobench`): 0.26 tok/s cold CPU → 0.30 warm CPU (MTP 2.2–2.3 tok/forward) → 0.42 tok/s GPU tier + auto-pin, expert hit 66%, ~65% of wall time in expert-disk. Disk-bound is the expected shape at ~25% expert residency — a faster disk and more RAM move the floor, the GPU moves the compute.

## Quick failure index

| Symptom | Cause | Fix |
|---|---|---|
| `'printf' is not recognized` / `The system cannot find the path specified` during `make colibri.exe` | scoop MinGW has no `sh.exe`; make fell back to cmd.exe (#478) | §0 — use MSYS2/w64devkit, or `set PATH=%PATH%;C:\msys64\usr\bin` |
| `An Application Control policy has blocked this file` | Smart App Control | §2 — turn SAC off + **reboot** |
| `cuda-dll ... Error 1` immediately | old tree: spaced CUDA_HOME / MSVC rejects `-Wextra` | update to current `dev` (#314) |
| `colibri.exe is up to date` but GPU never engages | old tree: stale CPU-only binary | update to `dev`, or delete the binary and rebuild |
| `cl.exe (MSVC) not in PATH` | built from plain PowerShell | use the x64 Native Tools prompt |
| `nvcc fatal: unsupported gpu architecture 'sm_120'` | CUDA < 12.8 | install CUDA 12.8+ |
| MTP `0% (0/0)` on CPU path | int4 MTP heads in the container | use the int8-MTP container |
| MTP `draft=0` under CUDA | intended default since #293 | `COLI_CUDA_MTP=1` to opt in |

---

## Reference: build flags & warmup

```sh
# AVX-VNNI: Intel Alder Lake+ (and Meteor Lake+) CPUs have a 128-bit int8
# dot-product instruction (VPDPBUSD) the engine can use for ~1.3x faster
# quantized matmul. The x86-64-v3 default (portable AVX2) compiles it out;
# build for THIS machine to enable it:
make colibri.exe ARCH=native                       # banner prints "idot: avx-vnni"

# Verify (tiny model, 2.4 MB):
pip install torch transformers safetensors huggingface_hub
python tools/make_glm_oracle.py                # generate tiny oracle
SNAP=./glm_tiny TF=1 ./colibri.exe 64 16 16        # expect "~30-32/32 positions"

# Run with real model:
SNAP=D:\glm52_i4 ./colibri.exe 64 4 16            # batch inference
python coli chat --model D:\glm52_i4            # interactive chat
python coli serve --model D:\glm52_i4            # OpenAI-compatible API
```

> Windows Store's `python` alias stub is the single most common native-Windows
> trap: install real Python (python.org or `winget install Python.Python.3.12`)
> or disable the alias under *Settings → Apps → App execution aliases*.

## Warmup (overnight cache priming)

The engine's expert cache learns from your workload. The included `warmup.ps1`
script runs `coli run` in a loop with diverse prompts to build the
`.coli_usage` histogram unattended, so the next real session starts with a
large, accurate hot-expert pin. Each run saves usage atomically on clean
completion.

```powershell
.\warmup.ps1 -Rounds 1 -Ngen 32               # ~60-90 min, durable progress
```

## NVIDIA GPU (optional, via runtime DLL)

On Windows the engine is built with MinGW gcc but CUDA kernels require MSVC +
nvcc. The split is clean: build the CUDA backend into a standalone
`coli_cuda.dll` (nvcc + MSVC), then the host `colibri.exe` loads it at runtime via
`LoadLibrary` (`c/backend_loader.c`). The host never links cudart directly; if
the DLL is absent the engine falls back to CPU without error.

```powershell
# Prerequisites: CUDA Toolkit + MSVC Build Tools (cl.exe) + nvcc on PATH.
# Build the DLL from a shell with the MSVC environment set (vcvars64.bat or
# "x64 Native Tools Command Prompt for VS"):
make cuda-dll CUDA_HOME="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8" CUDA_ARCH=sm_120

# Build the host with the runtime loader (CUDA_DLL=1 adds -DCOLI_CUDA and
# links backend_loader.o instead of cudart):
make colibri.exe CUDA_DLL=1 ARCH=native

# Run with the GPU expert tier (8 GB VRAM budget here; scale to your free VRAM):
$env:COLI_CUDA="1"; $env:COLI_GPU="0"; $env:CUDA_EXPERT_GB="8"
python coli chat --model D:\glm52_i4 --topp 0.7
```

The DLL exports the full `extern "C"` surface (including the #111 pipeline ABI);
`backend_loader.c` resolves symbols via `GetProcAddress` on first use.
`ColiCudaTensor*` is opaque to the host (stored, never dereferenced), so the
MSVC-allocated struct is safe across the ABI boundary. `CUDA_ARCH` must match
your GPU's compute capability (e.g. `sm_120` for Blackwell / RTX 50-series,
`sm_89` for Ada / RTX 40-series). A one-shot `build_cuda.bat` wrapper is also
available.

**Measured on a single RTX 5070 Ti + Core Ultra 9 (32 GB RAM):** CPU-only 0.63
→ CUDA attention+dense 0.72 → **1.07 tok/s** with the GPU-resident pipeline at
decode ([#273](https://github.com/JustVugg/colibri/issues/273), merged in #274).

## AMD GPU

The AMD sibling of the CUDA split above, and the same reasoning: MinGW gcc
cannot compile `.cu`, and Windows hipcc targets the MSVC ABI, so the backend is
built into a standalone `coli_hip.dll` rather than linked into the host. The
same `c/backend_cuda.cu` and the same `coli_cuda_*` ABI the Linux HIP path
already reuses are used unchanged — see [GPU_BACKENDS.md](../GPU_BACKENDS.md).

> **Validated end to end on one configuration.** The host resolves and loads
> `coli_hip.dll`, enforces which HIP runtime that DLL binds to, and has been
> exercised on real hardware: an AMD Radeon(TM) 8060S Graphics reporting
> `gfx1151`. Teacher forcing and a fixed-length free decode both produced the
> same token IDs on CPU and on the hybrid HIP path.
>
> That is one GPU, one SDK and one toolchain — see
> [Tested and untested](#amd-tested-and-untested) for the exact boundary before
> you rely on it. The CPU path is unaffected and always available.

### What you need

| Piece | Why |
|---|---|
| Windows x86-64 | the target platform |
| A compatible **MSVC x64 host toolchain** | hipcc brings its own clang front end but still needs the MSVC linker and Windows SDK; build from a shell with the MSVC environment set (`vcvars64.bat`, or an "x64 Native Tools Command Prompt") |
| A **Windows HIP SDK** providing `hipcc`, the HIP headers, the `amdhip64` import library and the device bitcode | compiles and links the backend |
| Your GPU's **architecture name** (`gfxNNNN`) | must be stated explicitly — see below |

MinGW-w64 `make` and `gcc` from §0 are still what build the host.

### Build

Two halves, built separately. The host build needs **no HIP SDK at all** — it
only compiles `c/backend_loader.c` and links `colibri.exe`, and never links
`amdhip64`:

```powershell
# 1. Host build mode (no SDK, no HIP_ARCH needed):
make -C c colibri.exe HIP_DLL=1

# 2. The backend DLL, from a shell with the MSVC environment set:
make -C c hip-dll HIP_DLL=1 HIP_SDK_ROOT=<sdk-root> HIP_ARCH=gfxNNNN
```

`HIP_SDK_ROOT` defaults from the `HIP_PATH` environment variable the HIP SDK
installer sets, so it can be omitted when that is already correct. Pass it
explicitly for a relocated or source-built SDK — that is preferable to editing
your machine-wide `HIP_PATH`. Each component (`HIP_BIN_DIR`, `HIP_INCLUDE_DIR`,
`HIP_LIB_DIR`, `HIP_DEVICE_LIB_PATH`, `HIPCC`) can be overridden on its own for
split layouts; the full list is in
[GPU_BACKENDS.md](../GPU_BACKENDS.md#sdk-selection-variables).

Generated artifacts: `c/coli_hip.dll`, plus `c/coli_hip.lib` if the linker emits
an import library (and `.exp`/`.pdb` on toolchains that produce them). All are
git-ignored and removed by `make -C c clean`.

> **Validated on:** AMD Radeon(TM) 8060S Graphics (`gfx1151`), TheRock HIP
> 7.14.60850, Visual Studio 2022 MSVC **14.44.35207**, Windows SDK
> **10.0.26100.0**. Other SDK versions and MSVC toolchains are expected to work
> but have not been exercised — there is **no hosted CI coverage** for this
> path, because no hosted runner ships a Windows HIP toolchain or an AMD GPU.

#### Pin the MSVC toolset

hipcc's clang picks the **newest** installed MSVC toolset. On a machine that
also carries Visual Studio 2026 that is MSVC **14.51.36231**, whose `<cmath>`
declares comparison builtins that collide with HIP's `__device__` overloads, and
the backend fails to compile. Pin the toolset for the build shell:

```powershell
$vs2022 = "C:\Program Files\Microsoft Visual Studio\2022\Community"
$env:VCToolsVersion     = "14.44.35207"
$env:VCToolsInstallDir  = "$vs2022\VC\Tools\MSVC\14.44.35207\"
```

VS2026 may stay installed; it simply must not be the toolset hipcc selects.

#### Isolate a stale machine-wide `HIP_PATH`

The HIP SDK installer sets a machine-wide `HIP_PATH`, and on a machine with more
than one ROCm/HIP install it can point somewhere you did not intend. Override it
**inside the build shell only** — do not edit the machine-wide value:

```powershell
$env:HIP_PATH = $sdk          # the SDK you actually want, for this shell only
# ROCM_PATH / ROCM_HOME / HIP_DEVICE_LIB_PATH are read by the ROCm toolchain, not
# by Colibri; clear them so a second install cannot redirect headers or bitcode:
Remove-Item env:ROCM_PATH, env:ROCM_HOME, env:HIP_DEVICE_LIB_PATH -EA SilentlyContinue
```

The build shell also needs `C:\Windows\System32` on `PATH` (native executables
invoked by the toolchain fail without it), and a writable `TMP`/`TEMP`/`TMPDIR`
— clang writes temporary files there and reports "unable to make temporary file"
when they are unset, which MSYS2 shells commonly leave empty.

### Runtime setup

Running a HIP host needs two things in place. Both are checked before any GPU
work starts, and anything that cannot be proven disables the GPU tier rather
than guessing — the engine continues on the CPU path.

**1. `coli_hip.dll` next to `colibri.exe`.** The HIP host loads *only* that
exact absolute path. There is deliberately no bare-name fallback and no
System32 search: a fallback could let some other `coli_hip.dll` satisfy the
load and quietly undo the runtime check below.

**2. `COLI_HIP_RUNTIME_DIR` naming the HIP runtime.** An absolute path to the
directory containing `amdhip64_7.dll`:

```powershell
$env:COLI_HIP_RUNTIME_DIR = "C:\path\to\hip\bin"
$env:COLI_CUDA = "1"
.\colibri.exe
```

Relative, drive-relative (`C:runtime`) and rooted-without-drive (`\runtime`)
values are rejected, because resolving them against the current directory is
exactly the ambiguity the setting exists to remove. Spaces, non-ASCII
characters and a trailing separator are all fine.

Why it is mandatory: Windows resolves a DLL import by **base name** against
whatever is already loaded, and a machine can easily carry more than one
`amdhip64_7.dll` (a system-wide ROCm install plus whatever you unpacked). Left
to chance, the backend binds to whichever one happened to load first. So the
loader refuses to start the GPU tier when it finds:

- a **different** `amdhip64_7.dll` already loaded than the one configured
- **more than one** `amdhip64_7.dll` loaded, even if one of them is the right
  file — which module an import binds to is decided by load order
- a loaded-module list it could not read, or a file whose identity it could
  not establish

Files are compared by **physical identity**, not by path text, so reaching the
configured runtime through a hard link or a differently-spelled path is
accepted.

One limitation stated plainly: the runtime is verified *after* it is mapped.
`LoadLibraryExW` takes a path rather than an open handle, so a file swapped
between validation and loading is **detected, not prevented**. Long paths
beyond `MAX_PATH` no longer hit a fixed buffer in the loader, but whether they
work still depends on your process manifest and Windows policy, and that has
not been validated on hardware here.

**CUDA is unaffected.** A `CUDA_DLL=1` host ignores `COLI_HIP_RUNTIME_DIR`
entirely and keeps its existing `coli_cuda.dll` search behaviour unchanged.

### Putting model tensors on the GPU — and checking that it happened

`COLI_CUDA=1` initialises the backend. It does **not**, on its own, put a single
model tensor on the GPU. Add `CUDA_DENSE=1` to make the dense tensors eligible:

```powershell
$env:COLI_CUDA  = "1"
$env:COLI_GPU   = "0"
$env:CUDA_DENSE = "1"
$env:COLI_HIP_RUNTIME_DIR = "$sdk\bin"
.\colibri.exe 64 16 16
```

Routed experts stay on **CPU** in this configuration. They become GPU-eligible
only through the existing expert-placement controls (`CUDA_EXPERT_GB` together
with a pin or usage source); that combination has not been validated on Windows
HIP.

**Verify residency — the device line is not proof.** A run can print

```
[CUDA] device 0: AMD Radeon(TM) 8060S Graphics, 84.0 GB VRAM, sm_115
```

and still execute every tensor on the CPU. The line that settles it is printed
at the end of the run:

```
[CUDA] resident set: 46 tensors, 0.00 GB VRAM
```

`N = 0` means **no model tensor was resident on the GPU**, even though the
device was discovered — the usual cause is `CUDA_DENSE` not being set. Treat
`resident set: N tensors` with `N > 0` as the evidence, and check it every time.
(The `GB` figure rounds at 1e9, so a small model legitimately shows `0.00 GB`
alongside a non-zero tensor count.)

**CPU fallback keeps the command successful.** If a tensor fails to upload, the
engine logs it, moves that tensor to the CPU and carries on; the process still
exits 0 and the numbers stay correct. You get a line per tensor for the first
few, an escalation notice once several have failed, and a final count:

```
[CUDA] N tensors ran on CPU after failed uploads: this run did NOT use the GPU for them.
```

Read those together with the resident set — a run can be *partly* on the GPU.

**Lifecycle.** Normal one-shot model exits rely on **Windows process teardown**
to release the backend and the HIP runtime; `coli_cuda_shutdown` is not called
on the ordinary teacher-forcing or decode exit paths. This matches the existing
Windows CUDA host behaviour and is safe for one-shot CLI use — it is process
teardown, not an explicit backend shutdown.

<a id="amd-tested-and-untested"></a>
### Tested and untested

**Tested**, on the configuration named above: the native Windows HIP build; the
host loading `coli_hip.dll` from its own directory; exact runtime binding with
fail-closed identity and duplicate-runtime checks; device discovery reporting
`gfx1151`; a real GPU kernel returning results identical to the CPU reference;
teacher forcing against the bundled synthetic oracle; and a fixed-length free
decode producing the same token IDs on CPU and on the hybrid HIP path, with 46
dense tensors resident.

**Not tested, and not claimed:** routed experts on the GPU or full-GPU MoE
inference; production-scale models; performance, throughput or memory-scaling
conclusions; long-run or server-mode stability; any GPU, driver, SDK or MSVC
toolchain other than the one listed; and behaviour against upstream `dev`
commits made after this work was validated.

### Generating the synthetic GLM oracle model

To exercise the engine end to end you need *a* model, and the smallest one is
generated locally rather than downloaded. `c/tools/make_glm_oracle.py` builds
**`glm_tiny`** — a ~0.6 MB GLM-5.2 (`glm_moe_dsa`) with the real architecture
(MLA + DSA indexer + sigmoid router + shared expert, 5 layers, 8 routed experts,
top-2) but random weights, plus `ref_glm.json`, a teacher-forcing oracle of
expected token IDs.

> **It is test tooling, not a language model.** Random weights produce
> meaningless text. Its only purpose is that the C engine must reproduce the
> reference token IDs exactly.

The packages below are **development tooling only** — the engine itself is a C
binary and needs no Python once the model files exist.

**1. Create an isolated environment.** Nothing is installed system-wide:

```powershell
$tooling = "$env:USERPROFILE\colibri-oracle-tooling"   # anywhere outside the repo
uv venv --python "$env:LOCALAPPDATA\Programs\Python\Python314\python.exe" "$tooling\venv"

# CPU-only Torch from the official PyTorch CPU index; the rest from PyPI
uv pip install --python "$tooling\venv\Scripts\python.exe" `
    --index-url https://download.pytorch.org/whl/cpu `
    --extra-index-url https://pypi.org/simple `
    --index-strategy unsafe-best-match `
    torch "transformers==5.11.0" "safetensors>=0.4"
```

Validated on **CPython 3.14.6 (Windows x64)** with **torch 2.13.0+cpu**,
**transformers 5.11.0** and **safetensors 0.8.0**. `transformers>=5.11` is a
hard floor — the generator exits below it, because older releases apply
split-half Llama RoPE and silently emit an oracle the engine cannot match
(issue #281). Pin 5.11.0 exactly to avoid oracle drift.

**2. Verify the environment** (imports only — builds nothing, downloads nothing):

```powershell
& "$tooling\venv\Scripts\python.exe" -c "import torch, transformers; from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM; print(torch.__version__, torch.version.cuda, transformers.__version__)"
```

Expect a `+cpu` Torch build and `None` for the CUDA version.

**3. Generate the model — from an external working directory.** The generator
takes no `--outdir`: it writes `glm_tiny/` **and** `ref_glm.json` relative to the
current directory. `c/ref_glm.json` is a tracked file, so generating inside `c/`
would overwrite it. Run it from outside the repository instead:

```powershell
$repo = "<your colibri checkout>"       # e.g. "$env:USERPROFILE\src\colibri"
$out  = "$tooling\model"
New-Item -ItemType Directory -Force -Path $out | Out-Null
Push-Location $out
& "$tooling\venv\Scripts\python.exe" "$repo\c\tools\make_glm_oracle.py"
Pop-Location
# -> $out\glm_tiny\{config.json,model.safetensors} and $out\ref_glm.json
```

Generation is fully offline once the packages are installed: the model is built
from a literal config with a fixed seed, and nothing is fetched from the network.

**4. Run the engine against it.** `SNAP` and `REF` both accept external paths,
so the generated files never need to enter the repository:

```powershell
$env:SNAP = "$out\glm_tiny"
$env:REF  = "$out\ref_glm.json"
$env:TF   = "1"
.\colibri.exe 64 16 16
```

`TF=1` is the teacher-forcing self-test: it prints
`PREFILL (teacher-forcing) C vs oracle: N/32 positions` and one
`[ORACLE] mismatch pos=… expected=… got=…` line per disagreement. Add
`DEBUG_LOGITS=1` to dump the top-5 logits at a mismatch. Floating-point
near-ties are toolchain-dependent, so a small shortfall is expected; compare a
GPU run against the CPU run on the same machine rather than against a fixed
number.

To exercise the HIP backend instead, add the runtime settings from
[Runtime setup](#runtime-setup-experimental-developer-facing) to the same
command.

**Cleanup** removes the tooling and the generated model in one step; the
repository is untouched:

```powershell
Remove-Item -Recurse -Force $tooling
```

### AMD failure index

| Symptom | Cause | Fix |
|---|---|---|
| `choose CUDA_DLL=1 or HIP_DLL=1, not both` | both runtime-DLL modes selected; the host resolves one DLL name | pick one per build |
| `Windows HIP build: no HIP SDK selected` | `hip-dll` requested with neither `HIP_PATH` nor `HIP_SDK_ROOT` set (`HIP_PATH` is not visible in every shell — an MSYS2 login shell drops it) | pass `HIP_SDK_ROOT=<sdk-root>` |
| `Windows HIP build: HIP_ARCH is empty` | no architecture given | pass `HIP_ARCH=gfxNNNN` |
| `Windows HIP build: set an explicit HIP_ARCH=gfxNNNN` | `HIP_ARCH=native` on Windows; `rocm_agent_enumerator` does not exist there, so there is nothing to resolve `native` against | name the arch, e.g. `HIP_ARCH=gfx1151` |
| `hipcc not found at "..."` / `HIP include dir not found` / `amdhip64.lib missing under "..."` / `device bitcode dir not found` | the selected SDK root does not have the expected layout | fix `HIP_SDK_ROOT`, or override just the component named in the message (`HIP_INCLUDE_DIR`, `HIP_LIB_DIR`, `HIP_DEVICE_LIB_PATH`, `HIPCC`) |
| `rocwmma/rocwmma.hpp missing under "..."` | the SDK lacks the rocWMMA component | install it, or point `HIP_INCLUDE_DIR` at a tree that has it |
| A no-SDK error from a **host-only** build | none — the host build is SDK-independent by design | `make -C c colibri.exe HIP_DLL=1` needs no SDK and no `HIP_ARCH`; if you see an SDK error here, you asked for the `hip-dll` target |
| Built the DLL but the GPU never engages | expected today | runtime loading of `coli_hip.dll` is not implemented yet; the engine stays on the CPU path |

SDK paths containing spaces are supported — every SDK-derived path the recipe
passes to the compiler is quoted. If you supply one yourself, quote it:
`HIP_SDK_ROOT="<path with spaces>"`.
