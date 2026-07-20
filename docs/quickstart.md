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

### Linux (Ubuntu / Debian)

```bash
sudo apt update
sudo apt install -y build-essential git python3
```

`build-essential` gives you `gcc`, `make`, and OpenMP (libgomp) — everything the
engine needs.

### Windows

You have two options.

**Option A — download a prebuilt binary (no compiler needed).**
Grab the latest `colibri-<version>-windows-x86_64.zip` from the
[Releases page](https://github.com/JustVugg/colibri/releases), unzip it, and
skip to [step 3](#3-get-the-model). Python 3 (from
[python.org](https://www.python.org/downloads/)) is still needed if you want to
convert a model yourself.

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
engine self-test: 32/32  (expected 32/32)
```

the engine is working correctly. (On Windows Option A you already have the
binary — you can skip this step.)

---

## 3. Get the model

You have two paths.

### Easiest — download a ready-made int4 container

A pre-converted **GLM-5.2 int4** model is on Hugging Face. **Use the version
with the int8 MTP heads** (the plain int4 heads disable speculative decoding —
see [#8](https://github.com/JustVugg/colibri/issues/8)):

**https://huggingface.co/mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp**

Download it into a folder on a fast disk, e.g. `/nvme/glm52_i4` (Linux/macOS) or
`D:\glm52_i4` (Windows). It is about **372 GB**, so make sure you have the space.

### Or convert it yourself from the FP8 source

One resumable command downloads and converts the model shard by shard, so it
never needs the full ~756 GB on disk at once:

```bash
./coli convert --model /nvme/glm52_i4
```

This step uses Python and runs only once. Safe to interrupt and re-run — it
resumes where it left off.

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
