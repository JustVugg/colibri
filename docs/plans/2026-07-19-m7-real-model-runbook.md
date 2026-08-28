# Runbook: real GLM-5.2 on this Mac — storage, download, and the road through M7–M9

Machine: M4 Max, 64 GB unified memory, 926 GiB internal NVMe (~365 GiB free as of
2026-07-19). Training design target per [AGENTS.md](../../AGENTS.md): trainer RSS
≤ 52 GB, hard ceiling 56 GB, zero steady-state swap.

---

## 1. How much disk space to free

**Target: ≥ 420 GiB free before starting the download. 450 GiB is comfortable.**

| Item | Size |
|---|---|
| GLM-5.2 colibri int4 snapshot (int8 MTP heads), measured via HF API | **383.8 GB = 357.4 GiB** (150 files) |
| Download slack (partial files during resume, HF cache metadata) | ~10 GiB |
| Training artifacts: tokenized SFT data, adapters, optimizer checkpoints | < 5 GiB (adapters are MBs) |
| APFS + macOS breathing room (swap must stay usable, purgeable churn) | ~50 GiB |

So from today's ~365 GiB free: **free up at least another ~60–90 GiB**.

Notes:
- The alternative route (`./coli convert` — streams FP8 shard by shard, never needs
  the full 756 GB) wants ~400 GB free *plus* Python/torch during conversion and
  hours of CPU time. **The pre-converted download is strictly better here** unless
  the mirror disappears.
- After the model is in place you'll have ~35–60 GiB free. That's workable but
  tight; don't let anything else large land on the disk during training runs.

**Before downloading, exclude the model dir from Time Machine and Spotlight**
(otherwise TM tries to back up 358 GiB and Spotlight burns CPU indexing it):

```bash
mkdir -p ~/Work/models/glm52_i4
tmutil addexclusion ~/Work/models/glm52_i4
mdutil -i off ~/Work/models 2>/dev/null || true   # optional, needs sudo on some setups
```

---

## 2. Downloading the right weights

**Use the pre-converted mirror with int8 MTP heads** (this is the one upstream
recommends; the older `jlnsrk` mirror has int4 MTP heads → 0% speculative
acceptance):

> https://huggingface.co/mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp

Download with the project venv's HF client (resumable — safe to interrupt and
re-run; do it overnight, ~358 GiB even at 40 MB/s is ~2.5 h on 10 GbE, a day on
slower links):

```bash
cd ~/Work/calibri_qlora
.venv/bin/pip install -q -U huggingface_hub
.venv/bin/hf download mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp \
    --local-dir ~/Work/models/glm52_i4
```

**Verify before first use:**

```bash
# 1) MTP heads must be int8 — exactly these sizes (int4 heads = broken MTP):
ls -l ~/Work/models/glm52_i4/out-mtp-*
#   3527131672 / 5366238584 / 1065950496   <- correct (int8)
#   1765523544 / 2686077736 / 536747200    <- WRONG (int4) — re-fetch those 3 files

# 2) read-only readiness check + storage plan (no tensors loaded):
cd ~/Work/calibri_qlora/colibri/c
COLI_MODEL=~/Work/models/glm52_i4 ./coli doctor
COLI_MODEL=~/Work/models/glm52_i4 ./coli plan

# 3) inference smoke — proves the snapshot end-to-end before any training:
COLI_MODEL=~/Work/models/glm52_i4 ./coli chat
```

Storage location: `~/Work/models/glm52_i4` on the internal NVMe (APFS is fine for
the download path; the ext4 note in upstream docs applies to the conversion
route only). The internal SSD on an M4 Max does ~5–7 GB/s reads — better than
most external TB enclosures, so running from the internal drive is actually the
fast option, just space-tight.

---

## 3. Detailed plan for the next milestones

### 3a. Pre-M7 engineering (all validated on the tiny oracle — needs NO download,
can proceed while the disk fills)

These close the gap between "trainer proven correct at tiny scale" (M0–M6, done)
and "trainer can touch the real snapshot":

1. **fmt=4 (grouped int4) support in the backward.**
   The real snapshot stores some tensors as grouped int4 (per-16..256-column
   scales, `detect_group_size`). `train_qt_bwd_dx` handles fmt 0/1/2 today and
   exits on 4. Add the grouped path (CPU first, then the `t_tmul` Metal variant)
   + unit tests against a dequantized reference, tiny-fixture gated like
   everything else.
2. **Memory budget manager (AGENTS.md §11).**
   One struct that owns the `--ram` budget: reserves dense-resident bytes,
   adapter+grad+optimizer state, training scratch, checkpoint reserve, OS
   margin; the *remainder* sizes the expert cache in **bytes** (today's slot
   count is a tiny-model stand-in). Refuses to start if the plan doesn't fit.
   Periodic log line: RSS, peak RSS, per-category estimates, macOS memory
   pressure + swap activity (sysctl / task_info). A run that swaps = failed run.
3. **Trainer CLI (`coli_train`).**
   `train/train_main.c` binary: `--model --data --adapter-out --ram 52
   --seq-len 128 --micro-batch 1 --grad-accum 16 --rank 8 --alpha 16
   --targets attention --lr 1e-4 --steps 100 --resume`. Wires together the
   already-proven pieces (model_init with real fmts, lora.h, train_model.h
   checkpointed mode, streamed expert cache, AdamW). Per-step metrics line per
   §19: loss, lr, tok/s (training, not forward-only), step time split
   fwd/bwd/opt/recompute, NVMe bytes+time, unique experts, cache hit rate, RSS.
4. **SFT dataset pipeline (§14).**
   `tools/prepare_sft.py`: JSONL chat/completion in → GLM-5.2 chat template via
   the HF tokenizer (tokenizer files are small, downloadable independently of
   weights) → `train.bin/train.idx/valid.bin/valid.idx/metadata.json`
   (pre-tokenized, prompt-mask flags). C-side reader `train/dataset.c`:
   deterministic seeded shuffling, seq-len windows, micro-batch iterator. No
   packing in v1.
5. **Checkpoint/resume (§15).**
   Checkpoint = adapter + AdamW state + step + data cursor + RNG + config +
   base fingerprint, atomic tmp→fsync→rename. Acceptance test on the tiny
   model: 20 uninterrupted steps ≡ 10 steps → save → restart → 10 steps
   (loss + adapter tensors match within tolerance). Wired into
   `test_train_tiny` or a sibling test.
6. **Scale-readiness sweep of `train_model.h`** (paper audit + asserts, cheap):
   int64 indexing everywhere at T=128/vocab≈150k/D≈6144, probs stash
   `[H,T,T]` sizing at real head count, logits buffer ~T·V·4 ≈ 78 MB
   (documented as retained), lm_head/embed at real vocab.

### 3b. Milestone 7 — real GLM-5.2 64 GB smoke test

Preconditions: §1+§2 done (model verified by `coli doctor` + a chat smoke),
pre-M7 items 1–3 done (4–5 can lag; a synthetic token file suffices for smoke).

Run configuration (per the brief — start at the floor, not the target):

```bash
./coli_train \
  --model ~/Work/models/glm52_i4 \
  --data  data/smoke_tokens \
  --adapter-out adapters/m7-smoke \
  --ram 52 --seq-len 128 --micro-batch 1 --grad-accum 4 \
  --rank 4 --alpha 8 --targets attention --lr 1e-4 --steps 10
```

Progression: 1 forward → 1 forward+backward → 1 full optimizer step → 10 steps.
At each stage watch the metrics line; abort criteria are hard:

- RSS > 52 GB sustained or *any* steady-state swap → stop, fix the budget, rerun;
- non-finite loss → stop, bisect layer-by-layer against expectations;
- a layer pass whose unique-expert count approaches E → expected for training
  (fan-out risk, §21) — this is a *throughput* problem, not an error; record
  bytes/step honestly.

Success = the §16-M7/§27 checklist: no OOM, no swap, RSS under ceiling, finite
loss, nonzero adapter update, checkpoint loads back into `coli chat
ADAPTER=...`, zero-adapter run still bit-identical, `make check` still green.
Expect single-digit steps/hour on the first attempt — the brief explicitly says
feasibility and correctness first, throughput later (M9).

### 3c. Milestone 8 — overfit proof

Small repeated SFT set (~20–50 short samples, built with `prepare_sft.py`), a
few hundred steps at seq-len 128, rank 8:

- training loss must drop *materially* (not noise);
- adapter tensor norms move;
- greedy generation on a training prompt moves toward the target completion;
- `ADAPTER` unset → base behavior restored exactly.

This is the first end-to-end proof that the whole pipeline *learns something
real* on the 744B base. Also the point where resume (§15) gets exercised on the
real model (kill a run mid-training, resume, confirm trajectory).

### 3d. Milestone 9 — practical training (only after M7+M8 are honest passes)

In priority order (brief §20): longer seq-len (256 → 512 as memory accounting
allows) · expert scheduling and cache policy tuned from measured hit rates ·
NVMe prefetch overlapped with compute (reuse inference's pilot/prefetch
machinery) · Metal dispatch of the training hotspots with *measured* CPU/GPU
thresholds (kernels exist since M6; wiring waits for real profiles) ·
bf16 adapter/optimizer state option · gradient-accumulation performance ·
dataset packing · selective layer targeting (train fewer layers, save state).

### Open risks to keep in view (§21)

- **Expert fan-out**: a 128-token batch can route to a large fraction of the 256
  experts per layer; backward reloads double the traffic. Bytes/step is THE
  number to watch — dedup + cache-across-steps (same batch reuses experts) are
  the levers.
- **Disk headroom**: post-download free space (~35–60 GiB) leaves no room for
  careless artifacts; checkpoints are small but keep an eye on `df`.
- **Attention-only LoRA capacity**: if M8 overfit is sluggish, that's signal
  about adapter scope (§22 sparse-expert LoRA is the future lever, not v1).
