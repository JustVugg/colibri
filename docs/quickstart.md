# Quick Start — from zero to a running model

A step-by-step guide for first-time users on **Linux**, **Windows**, and **macOS**.
No prior experience with C, CUDA, or model conversion is assumed. If you get
stuck, `./coli doctor` (below) tells you exactly what's missing.

> **What you're setting up:** colibrì runs a very large Mixture-of-Experts model
> (e.g. GLM-5.2, 744B parameters) on a normal machine by streaming the model's
> experts from disk instead of needing them all in RAM. The engine is a single
> C program; Python is only used once, to prepare the model files.

Use an AI coding assistant? Ask it to set up colibri following [docs/AI_SETUP.md](AI_SETUP.md).

---

## The one-step way

**Windows:** download the repository (on GitHub: **Code**, then **Download ZIP**,
and unzip it; or `git clone`), then double-click **`START-HERE.bat`** in the
top folder.

**Linux and macOS:**

```bash
git clone https://github.com/JustVugg/colibri.git
cd colibri
./start-here.sh
```

Both run `coli setup`, which does the rest:

1. **Finds your hardware**: RAM, free disk where the model goes, CPU features,
   and GPUs (any Vulkan GPU with its memory, NVIDIA cards through nvidia-smi).
2. **Recommends a model that fits**, in a short numbered menu with download
   sizes; Enter takes the recommendation. "Fits" means the part of the model
   that always stays in RAM (the dense part) plus a minimum expert cache fit
   in your RAM, and the download fits on your disk.
3. **Gets the engine**: it builds it for your machine when a compiler is
   there, with Vulkan or CUDA when your GPU can use it, or downloads the
   prebuilt one when there is no compiler (CPU, and on Linux and Windows
   Vulkan too). If a package is missing for
   your GPU it prints the exact command to install it and carries on with the
   CPU; nothing is installed system-wide without you. Run the setup again once
   you have installed it, and it rebuilds the engine for the GPU. CUDA is used
   only when the installed CUDA toolkit can build for your card: CUDA 13, for
   example, no longer builds for Maxwell, Pascal or Volta cards (a V100), so
   there the setup says so and uses Vulkan. If a GPU build fails anyway, it
   moves to the next one (CUDA, then Vulkan, then the CPU) and tells you where
   the build log is.
4. **Downloads the model** with progress and resume. Interrupt it whenever you
   like: running it again continues where it stopped.
5. **Starts colibri** and opens the dashboard in your browser, and prints the
   addresses other apps can use.

What you see on Linux (on Windows the same, in the window `START-HERE.bat` opens):

```
$ ./start-here.sh
colibri setup

Your machine
  CPU     13th Gen Intel(R) Core(TM) i7-1355U, 6 cores (12 threads), AVX2, AVX_VNNI
  RAM     27.3 GB (24.7 GB free now)
  Disk    787 GB free in /home/me/colibri-models
  GPU     Intel(R) Iris(R) Xe Graphics via Vulkan 1.3 (integrated, shares RAM)
  System  Ubuntu 24.04 LTS

Models that fit this machine
  (fits = the dense part, which always stays in RAM, plus a minimum expert cache fit in RAM, and the download fits on the disk)
   1) Qwen3.6-35B-A3B                   23 GB   runs from RAM         [recommended]
      general chat with thinking and tools; int4-gs64 container; the whole model fits in RAM (20 GB)
   2) Qwen3-Coder-30B-A3B               19 GB   runs from RAM
      coding model with tool calls, no thinking; int4-gs64 container; the whole model fits in RAM (18 GB)
   3) DeepSeek V4 Flash REAP 150B       85 GB   streams from the SSD
   ...
  (2 more fit too: `--all` lists them, `--model ID` picks one)
  (4 more need more RAM or disk: `coli setup --list` shows why)
Choose a model [Enter = 1]:

Engine: qwen36 with VULKAN (Intel(R) Iris(R) Xe Graphics, integrated GPU)
  building: make qwen36 ARCH=native VK=1
Download: Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64 (23.1 GB); safe to interrupt, rerun to continue
  [##########..............]  41.0%  9.5 GB/23.1 GB  38.2 MB/s  6 min left
...
Starting colibri
  Browser:             http://127.0.0.1:8000/
  OpenAI base URL:     http://127.0.0.1:8000/v1
  Anthropic base URL:  http://127.0.0.1:8000
  stop: press Ctrl+C here (or close this window)
```

**Later:** run `START-HERE.bat` or `./start-here.sh` again and colibri starts
straight away, with no second download or build. To stop it, press Ctrl+C in
its window, or run `c/coli stop` from another terminal (`c\coli.cmd stop` on
Windows). `c/coli status` shows what is installed, whether it runs, its
addresses and the speed of the last answer. To pick another model:
`./start-here.sh --reconfigure`.

**On WSL** keep the model on the Linux disk (the default, `~/colibri-models`),
never under `/mnt/c`. If WSL's own network is much slower than Windows' (it can
be: 63 KB/s against 3.3 MB/s was measured on one machine), the setup offers to
download through Windows' `curl.exe` into the same folder.

Useful options (`c/coli setup --help` lists them all): `--yes` takes every
default without asking, `--model ID` picks a model (`--list` shows the ids
against your machine), `--dir DIR` puts the models elsewhere, `--no-gpu` keeps
everything on the CPU, `--backend vulkan` (or `cuda`) picks the GPU path
yourself, `--model-dir DIR` uses a model you already have.

The rest of this page is the same thing done by hand, step by step.

---

## 0. What you need first (prerequisites)

| | Minimum | Recommended |
|---|---|---|
| **RAM** | ~16 GB | 24 GB+ |
| **Free disk** | ~430 GB for the GLM-5.2 int4 model (19 GB for the smallest model the one-step setup offers) | a fast NVMe SSD (streaming speed = your token speed) |
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
| `qwen36.exe`, `kimi_k3.exe`, … | the engines of the other model families, chosen from the model's `config.json` |
| `shaders\` | the Vulkan shaders: the engines run on a Vulkan GPU too, with nothing to build |
| `coli` | the command-line launcher (`chat`, `serve`, `convert`, `doctor`, …) |
| `openai_server.py`, `resource_plan.py`, `doctor.py`, `autotune.py` | Python support for the API server, placement planner, diagnostics, and measured tuning |

One setup step: **install Python 3** from
[python.org](https://www.python.org/downloads/) — the `coli` launcher and the
API gateway are Python scripts (the engine itself is pure C and needs nothing).
No renaming, no configuration: the launcher finds `colibri.exe` next to itself.
With an NVIDIA RTX 30, 40 or 50 series card, also unpack
`colibri-<version>-windows-x86_64-cuda.zip` into the same folder for CUDA
([windows.md](windows.md#if-you-downloaded-a-release-archive-start-here)).

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
# MacPorts works too: sudo port install libomp   (the Makefile finds either)
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

**GLM-5.3** is the same family and loads with the same engine. Its own
group-scaled (gs64) container is about **419 GB** and ships **without** the MTP
head, so speculative decoding stays off:

**https://huggingface.co/Justvugg/GLM-5.3-colibri-int4-g64**

Group scales matter: the older per-row int4 containers
(`mateogrgic/…-int4-with-int8-mtp`, `jlnsrk/…`) measure ~9pp worse on quality
benchmarks and are the root cause of the think-mode loops and never-terminating
generations in [#455](https://github.com/JustVugg/colibri/issues/455) — the
gs64 container cured every failing case in that report. (The int8 MTP head is
also required: plain int4 heads disable speculative decoding, see
[#8](https://github.com/JustVugg/colibri/issues/8).)

Download it into a folder on a fast disk, e.g. `/nvme/glm52_i4` (Linux/macOS) or
`D:\glm52_i4` (Windows). It is about **429 GB**, so make sure you have the space.

The MTP head must be **int8, not int4** (int4 gives 0% draft acceptance,
[#8](https://github.com/JustVugg/colibri/issues/8)). Check it with
`ls -l <model>/out-mtp-*`: int8 (correct) is `3527131672 / 5366238584 / 1065950496`
as three files, or a single `out-mtp-00000.safetensors` of `9959321520` bytes
(the current upload of the recommended container ships it as one file: same
int8 tensors, 777 of them at one byte per element).

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
