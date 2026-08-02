# Quick Start — from zero to a running model

A step-by-step guide for first-time users on **Linux**, **Windows**, and **macOS**.
No prior experience with C, CUDA, or model conversion is assumed. If you get
stuck, `./coli doctor` (below) tells you exactly what's missing.

> **What you're setting up:** colibrì runs a very large Mixture-of-Experts model
> (e.g. GLM-5.2, 744B parameters) on a normal machine by streaming the model's
> experts from disk instead of needing them all in RAM. The engine is a single
> C program; Python is only used once, to prepare the model files.

---

## 0. What you need first (prerequisites)

| | Minimum | Recommended |
|---|---|---|
| **RAM** | ~16 GB | 24 GB+ |
| **Free disk** | ~380 GB for the int4 model | a fast NVMe SSD (streaming speed = your token speed) |
| **OS** | Linux, Windows 10/11, or macOS | any |
| **Tools** | a C compiler + `make` + `git` + `python3` | — |

You do **not** need a GPU. A GPU only helps if you have one; the engine runs
CPU-only by default.

---

## 1. Install the build tools

> **Shortcut — skip sections 1 and 2 entirely.** Prebuilt archives are published
> for **Linux, macOS and Windows** on the
> [Releases page](https://github.com/JustVugg/colibri/releases): unpack one,
> install [Python 3](https://www.python.org/downloads/), and jump straight to
> [step 3](#3-get-the-model). The engine ships ready to run and the `coli`
> launcher finds it next to itself — no compiler, no renaming, no configuration.
>
> ```bash
> mkdir colibri && tar xzf colibri-v1.1.0-linux-x86_64.tar.gz -C colibri && cd colibri
> python3 coli info        # engine ready ✓
> ```
>
> Build from source instead if you want the fastest binary for *your* CPU
> (`ARCH=native` unlocks the vector instructions your chip actually has), or if
> you plan to hack on the engine.
>
> **On ARM64 Linux (AWS Graviton, Ampere, Raspberry Pi, aarch64 VMs) there is no
> shortcut**: the published Linux archive is x86_64 only. Build from source —
> sections 1 and 2 work unchanged, and the engine needs no ARM-specific flags.

### Linux (Ubuntu / Debian)

```bash
sudo apt update
sudo apt install -y build-essential git python3
```

`build-essential` gives you `gcc`, `make`, and OpenMP (libgomp) — everything the
engine needs. The same line works on aarch64: the engine is portable C with
OpenMP and no x86-only intrinsics, so `./setup.sh` builds it on ARM64 without
source changes or extra flags (verified on AWS Graviton4, Ubuntu 24.04, gcc 13).

> **Moving a build to another machine?** The engine links `libgomp.so.1` at run
> time. A host that has never had a compiler installed — a minimal cloud image, a
> fresh container, a restored volume attached to a new instance — may not carry
> it, and the engine then exits at startup before printing anything. Install the
> runtime package alone (`sudo apt install -y libgomp1`); `coli doctor` names the
> missing library.

### Windows

You have two options.

**Option A — download a prebuilt binary (no compiler needed).**
Grab `colibri-<version>-windows-x86_64.zip` from the
[Releases page](https://github.com/JustVugg/colibri/releases) and unzip it.
Inside you'll find:

| File | What it is |
|---|---|
| `colibri.exe` | **the engine** — the C program that actually runs the model |
| `coli` | the command-line launcher (`chat`, `serve`, `convert`, `doctor`, …) |
| `openai_server.py`, `resource_plan.py`, `doctor.py`, `autotune.py` | Python support for the API server, placement planner, diagnostics, and measured tuning |

One setup step: **install Python 3** from
[python.org](https://www.python.org/downloads/) — the `coli` launcher and the
API gateway are Python scripts (the engine itself is pure C and needs nothing).
No renaming, no configuration: the launcher finds `colibri.exe` next to itself.

For better understanding, from powershell prompt, a complete invocation line 
(relying on py launcher, to be launched from the folder where colibri.exe is) is:

PS1> $env:COLI_MODEL="drive:/path/to/1st_copy/model/folder/"; $env:COLI_MODEL_MIRROR="/2nd_copy/model/folder/"; & py ./coli chat


Then continue to [step 3](#3-get-the-model). Prefer to skip the launcher? You can
run the engine directly — `.\colibri.exe` reads the model path from the `SNAP`
environment variable (see [docs/windows.md](windows.md)) — but `coli chat` is the
easy path.

**Option B — build from source with MSYS2.**
Install [MSYS2](https://www.msys2.org/), open the **UCRT64** shell, and run:

```bash
pacman -S --needed mingw-w64-ucrt-x86_64-gcc make git python
```

### macOS

```bash
xcode-select --install          # C compiler (clang)
brew install libomp git python  # OpenMP for multithreading
```

---

## 2. Get the code and build the engine

```bash
git clone https://github.com/JustVugg/colibri.git
cd colibri/c
./setup.sh
```

`setup.sh` checks your compiler and OpenMP, builds the engine, and runs a tiny
self-test. When it prints:

```
engine self-test: 32/32  (expected ~30-32/32; FP near-ties are toolchain-dependent)
```

the engine is working correctly. Some toolchains report 30/32 or 31/32 because
two tiny-oracle positions are floating-point near-ties; this is still a valid
self-test result. (On Windows Option A you already have the binary — you can
skip this step.)

---

## 3. Get the model

You have two paths.

### Easiest — download a ready-made int4 container

A pre-converted **GLM-5.2 int4** model is on Hugging Face. Use the
**group-scaled (gs64)** container with the **int8 MTP head**:

**https://huggingface.co/mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp**

Group scales matter: the older per-row int4 containers
(`mateogrgic/…-int4-with-int8-mtp`, `jlnsrk/…`) measure ~9pp worse on quality
benchmarks and are the root cause of the think-mode loops and never-terminating
generations in [#455](https://github.com/JustVugg/colibri/issues/455) — the
gs64 container cured every failing case in that report. (The int8 MTP head is
also required: plain int4 heads disable speculative decoding, see
[#8](https://github.com/JustVugg/colibri/issues/8).)

Download it into a folder on a fast disk, e.g. `/nvme/glm52_i4` (Linux/macOS) or
`D:\glm52_i4` (Windows). It is about **372 GB**, so make sure you have the space.

### Or convert it yourself from the FP8 source

One resumable command downloads and converts the model shard by shard, so it
never needs the full ~756 GB on disk at once:

```bash
./coli convert --model /nvme/glm52_i4
```

This produces a group-scaled (gs64) container — the same quality-validated
format as the recommended download. This step uses Python and runs only once.
Safe to interrupt and re-run — it resumes where it left off.

---

## 4. Run it

Point `COLI_MODEL` at the folder from step 3 and start chatting:

```bash
# Linux / macOS
COLI_MODEL=/nvme/glm52_i4 ./coli chat

# Windows (UCRT64 shell)
COLI_MODEL=/d/glm52_i4 ./coli chat
```

Useful first commands:

```bash
COLI_MODEL=/nvme/glm52_i4 ./coli doctor   # read-only check: is everything ready?
COLI_MODEL=/nvme/glm52_i4 ./coli plan     # shows where the model will live (RAM/disk/GPU)
COLI_MODEL=/nvme/glm52_i4 ./coli chat --topp 0.85   # faster: reads less from disk, same quality
```

> **Tip:** `--topp 0.85` is worth adding on a disk-bound machine — it reads
> fewer expert bytes per token with no quality loss, which directly means more
> tokens per second.

---

## 5. What to expect

- **First launch loads the resident weights** (~10 GB) — this takes a moment.
- **Speed depends on your disk.** The experts stream from storage, so a fast
  NVMe SSD is the single biggest factor in tokens/second. On a slow or shared
  disk, generation can be well under 1 token/second — that's expected, and it's
  the honest cost of running a 744B model on a small machine.
- **It's still the full model.** Placement only changes speed, never the model's
  answers or precision.

If something doesn't work, run `./coli doctor` — it reports exactly what's
missing (compiler, model files, permissions) and how to fix it.

---

## Where to go next

| Topic | Doc |
|---|---|
| Windows native build (and CUDA DLL) | [docs/windows.md](windows.md) |
| Tuning: cache, prefetch, speculation | [docs/tuning.md](tuning.md) |
| OpenAI-compatible API + web dashboard | [docs/api.md](api.md) |
| Every environment variable | [docs/ENVIRONMENT.md](ENVIRONMENT.md) |
