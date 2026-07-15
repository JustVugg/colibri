# GLM-5.2 colibrì int4 — Evaluation Report (RTX 4090 Laptop)

Log-likelihood benchmark of the int4 (int8-MTP) GLM-5.2 build served by the
colibrì C engine, run via `coli bench` (`tools/eval_glm.py`) on a single
RTX 4090 Laptop GPU. Dated 2026-07-15.

## Setup

| Component | Value |
|---|---|
| Model | `mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp` (744B MoE, ~384 GB, int4 weights + int8 MTP heads) |
| Engine | `glm.c` built with `CUDA=1 CUDA_ARCH=sm_89` (no-root conda-forge `cuda-nvcc` 12.6) |
| GPU | RTX 4090 Laptop, ~13 GB usable VRAM → expert tier `CUDA_EXPERT_GB=12`, pinned via `CUDA_RELEASE_HOST=1 PIN=.coli_usage` |
| Host | 32 GB RAM (24 GB to WSL2 via `.wslconfig`); model on an **ext4 external SSD** (Samsung 990 PRO 2 TB) |
| Sandbox | runs wrapped in `unshare --user --map-root-user --net` (no outgoing network; GPU via `/dev/dxg` still works) |
| Harness | `tools/eval_glm.py` — EleutherAI-style **log-likelihood** MC scoring, 0-shot, `acc` / `acc_norm` |
| Tasks | HellaSwag, ARC-Challenge, MMLU (datasets via `tools/fetch_benchmarks.py`, 200 Q/task cached) |

## Results

### Completed run — `--limit 10` (n=10/task, 120 forwards)

| task | n | acc | acc_norm |
|---|---|---|---|
| hellaswag | 10 | 30.0% | 50.0% |
| arc_challenge | 10 | 60.0% | 60.0% |
| mmlu | 10 | 60.0% | 60.0% |
| **mean** | | | **56.7%** |

- **Cost:** 13,708 s engine time (~3.8 h) for 120 forwards → **~114 s/forward**, cache hit **~4 %**. Fully disk-bound.

### Partial run — `--limit 40` (n=40/task, 480 forwards)

Started to tighten the confidence interval to the reference sample size; **stopped
by request at ~17 %** (~2.6 h of compute). Not completed. See "Operational notes".

## Interpretation

The only apples-to-apples anchor is the community full-run of the **same harness**
(project issue #108, n=40): **62.5 % mean `acc_norm`**. Our **56.7 %** (n=10) sits
within the n=10 sampling noise of that figure (per-task CI ≈ ±30 pp), so the int4
build **tracks the reference harness** — consistent, and **not** evidence of
quantization loss. `eval_glm.py`'s `REFERENCE` field is still unfilled upstream;
62.5 % is the community anchor, not an official model-card number.

## This is NOT comparable to "actual GLM-5.2" performance

GLM-5.2 (Z.ai, released 2026-06-16; 744B MoE, ~40B active, 384 experts) publishes
**no** MMLU / HellaSwag / ARC-Challenge numbers on its model card — those classic
sets are saturated (frontier ≈ 90 % MMLU CoT, ≈ 95 % HellaSwag). Its headline
figures are reasoning/agentic/coding suites instead:

- Terminal-Bench 2.1 **81.0**, SWE-bench Pro **62.1** (beats GPT-5.5; ~1 pt behind Opus 4.8 on FrontierSWE)
- ARC-AGI-2 **22.8 %**

Two reasons our numbers can't be read against those:

1. **Different benchmarks.** We test saturated MC sets; the card tests hard
   reasoning/coding suites.
2. **Different method.** We score 0-shot **log-likelihood** (model never reasons);
   the card uses **CoT / generative** scoring, which is where a reasoning model
   earns its results.

> ⚠️ Do not confuse our **ARC-Challenge** (grade-school science QA) with the
> card's **ARC-AGI-2** (abstraction puzzles). Unrelated benchmarks that share the
> "ARC" prefix — 60 % vs 22.8 % is meaningless as a comparison.

The eval is therefore a **relative sanity check** (int4 colibrì vs. the community's
full run of the same harness), not a measure of GLM-5.2's frontier reputation.

## Operational notes

- **Disk-bound, not compute-bound.** ~114 s/forward with ~4 % expert-cache hit;
  the 4090 sits at single-digit % util in disk-wait (load avg ~31). The VRAM
  expert tier absorbs traffic but cold experts still stream from the SSD.
- **Host sleep freezes the run.** WSL2's VM clock pauses while the laptop is
  suspended, so an overnight run makes **no progress during sleep** — it survives
  the suspend and resumes, but only banks *awake* compute. A full n=40 needs
  ~15 h of **awake, plugged-in** time; n=100 ≈ 1.5 days.
- **Reproduce:** re-mount the ext4 drive (`wsl --mount \\.\PHYSICALDRIVE1 --partition 1`,
  UAC), then `coli bench --limit N` with `datasets`/`tokenizers` on `PYTHONPATH`.

Sources for the GLM-5.2 model-card figures: kie.ai GLM-5.2 benchmark deep dive;
digitalapplied GLM-5 744B release analysis; layerlens ARC-AGI-2 verified score.
