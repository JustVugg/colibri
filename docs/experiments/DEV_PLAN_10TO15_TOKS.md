# Colibrì performance plan — 10–15+ tok/s decode on Apple M4 (16 GB)

**Date:** 2026-11-07 · **Branch:** `feature-IQ2_XXS` · **Author:** investigation agent (plan only; no source changes)
**Predecessor reports:** `logs/TQ1_0_COMPATIBILITY_REPORT.md`, `logs/QWEN3_CODER_30B_MAXIMUM_LOAD_TEST.md`, `logs/QWEN3_CODER_50K_AUTO_EXECUTION_TEST.md`, `logs/TQ1_0_IMPORT_TEST.md`

---

## 0.4 EXECUTION ADDENDUM — 2026-11-07, import + A/B of Qwen3.6-27B-UD-IQ2_XXS (corrects §2)

Executed §2.2 and §10.1–10.2 evidence steps. Findings that **correct the plan**:

### 0.4.1 The 27B-MTP model is DENSE, not MoE

- Import OK (link mode): `/Users/creemac/.lmstudio/models/unsloth/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-UD-IQ2_XXS.colibri`
- Engine verify: `qwen35 (dense): 64 layers (16 attention), hidden 5120, MLP 17408, vocab 248320, DeltaNet v/k heads 48/16` — **0 experts** (`GGUF CHECK OK`, 866 tensors, 9.56 GB).
- The "all-IQ2_XXS gate+up+down" tensors are **dense-FFN**, not expert tensors. Consequence: there is no expert cache, cap is meaningless, and every weight is read every token. §2's "64/65 layers all-IQ2_XXS experts — the exact tensors the Metal kernel handles" reasoning is re-interpreted: the Metal kernel still handles them (it did — see 0.4.2), but the workload is dense.

### 0.4.2 Measured decode on the 27B dense (mode 13, 64 tokens, greedy; `plan/doctor` misclassify nothing here — planner honestly reports dense/memory-bound)

| Config | decode | TTFT (T(1)) | notes |
|---|---|---|---|
| CPU, OMP=4 | **0.99 tok/s** | 1057 ms | 13,764 decode GEMVs on CPU |
| CPU, OMP=10 | 1.25 tok/s | 827 ms | |
| Metal, OMP=4 | **1.51 tok/s** | 692 ms | `0:13764/0` = all decode GEMVs → Metal, 0 fallbacks |
| Metal, OMP=10 | 1.25 tok/s | 890 ms | thread contention |

**Verdict: the 27B dense is NOT the primary model for 10–15 tok/s** on this host — it starts an order of
magnitude below the MoE Coder baseline (5.2 tok/s). MTP (§4) would at most double ~1.5. The MoE
Qwen3-Coder-30B stays primary; the 27B-dense becomes the *paradoxical best target for the Metal
dense-FFN kernel work* (§6.4) because 100 % of its FFN is IQ2_XXS and Metal already runs every
GEMV: its 1.5 tok/s at ~14 GB/s effective proves kernel headroom, not a hardware ceiling
(theoretical ~9.56 GB/token at ~90 GB/s ⇒ ~9 tok/s).

### 0.4.3 MTP head confirmed OFF even with `Q36_MTP=1`

`[mtp] Q36_MTP=1: the head stays off` despite the GGUF carrying `blk.64.nextn.*` (eh_proj, enorm,
hnorm, shared_head_norm). Either the metadata key `nextn_predict_layers` is absent from this export
or the loader only accepts a sidecar. §4.1.1 (GGUF-native `nextn.*` branch in `qwen36_mtp.h`) is
therefore REQUIRED for any MTP milestone and is now the top dev item whose outcome is directly
measurable (rerun mode 13 and look for `MTP proposed/accepted` lines).

### 0.4.4 Corrected milestone model mapping

| Milestone | Primary artifact |
|---|---|
| M1 tune | Coder-30B (unchanged) |
| M2 MTP ≥10 tok/s | **Coder-30B quality MoE + MTP=27B-dense-head is NOT viable** — §4 must be re-scoped: MTP on the dense 27B (needs a fast dense trunk first, M3/M4) or a MoE GGUF that ships `nextn.*` (inventory task) |
| M3 batched CPU experts | Coder-30B (unchanged; its 1,152 GEMV/token problem stands) |
| M4 Metal | **Coder-30B experts + dense trunk**, with the 27B dense as the cleanest bench because every GEMV is already Metal-dispatched |

Revised realistic ladder on this host: M1 ⇒ ~6.5 tok/s (Coder), M3 ⇒ ~10–13 tok/s (Coder),
M4 ⇒ dense-27B from 1.5 toward 4–6 tok/s while MoE gains accumulate via §6.1–6.3.

---

## 0.5 EXECUTION LOG — 2026-11-07, GGUF-native MTP head implemented + 35B-A3B-MTP baseline (executes §4.1.1, partial §2)

### 0.5.1 §4.1.1 DONE — `q36_mtp_load()` now reads GGUF-native `blk.<L>.nextn.*` heads

**Code** (branch `feature-IQ2_XXS`, working tree only, no commits):
- `c/qwen36.c`: 4 identity entries in `gg_name`'s `lay` table (`nextn.{eh_proj,enorm,hnorm,shared_head_norm}.weight`); GGUF-native branch in `q36_mtp_lm_rows()` (dequant-row + dot on `lm_head.gq`, with `.scale` factor).
- `c/qwen36_mtp.h`: new `q36_mtp_load_gguf()` dispatched first from `q36_mtp_load()` when `g_gguf`; dense-FFN variant `q36_mtp_moe_dense()` and routed-expert variant `q36_mtp_moe_gguf()` (same routing math as the sidecar path; experts served by `gq_matmul` straight from the file's blocks); dispatch by `p->dmoe` (0 sidecar int8 / 1 GGUF routed / 2 GGUF dense); `Q36_MTP_VOCAB_IDS` builder and draft-vocab default top made GGUF-aware.

**Design decisions worth recording**:
- Loader names are **HF-style** (`model.layers.64.nextn.*`, `self_attn.q_proj`, ...) so they go through `gg_name` — mandatory because `q36_rope_heads` only recognizes HF suffixes; GGUF-style names would silently skip the RoPE un-permute of q/k rows. (First attempt with literal `blk.*` names failed with `GGUF: missing blk.64.nextn.eh_proj.weight`; second with HF names + literal-gate failed silently since `gguf_find` only knows literal names — the probe must be literal `blk.<L>.nextn.enorm.weight`, the loads must be HF-style.)
- The GGUF-native head block is a **complete layer**: in the 27B file `blk.64` carries full attention (gated q_proj 24×512) + dense FFN 17408 + `nextn.*` extras; in the 35B file `blk.40` carries full attention + `ffn_{gate,up,down}_exps` (256 experts × 1.09 MB) + `ffn_*_shexp` + `ffn_gate_inp` router + `nextn.*` 4096→2048 `eh_proj` (Q8_0). Both head shapes supported.
- Routed experts: whole `ffn_*_exps` tensors read at load into a resident slab (278 MB for the 35B head) with per-expert `.scale` factors via the trunk's `gg_scale`; per-(row,expert) `gq_matmul` calls, shared-expert QWs via `load_tq` (device path untouched).
- Draft-vocab default: GGUF heads default to `DRAFT_VOCAB_DEFAULT_TOP` (16k rows) like the CUDA chain — the whole-vocab lm_head on the 27B is Q5_K ~874 MB, per-draft full scans are not viable.

### 0.5.2 Verification — the head loads and drafts on BOTH model shapes

Downloads/verification:
- `Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf` downloaded (11,819,399,456 B, exactly the HF size), sha256 `627e2b05…441203` = HF LFS sha. Imported link mode: `…/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-IQ2_XXS.colibri` → `qwen35moe, 40 layers`.
- Build clean (`make -C c qwen36`), no new warnings.

Qwen3.6-27B (dense, `./c/qwen36 13 4`, `Q36_MTP=1`): `[mtp] GGUF-native head (blk.64.nextn): dense FFN 17408`; **1.38 tok/forward, acceptance 42.9% (3/7)**; head row 50.6 ms / fed row 1.9 ms; 0.79 vs 0.69 tok/s MTP-off (small +15% at cap-13 sampling config).

Qwen3.6-35B-A3B (MoE, `./c/qwen36 0 4`, `RAM_GB=11`, `Q36_MTP=1`): `[mtp] GGUF-native head (blk.40.nextn): 256 experts top-8, 278 MB resident`; **1.72 tok/forward, acceptance 71.6% (53/74)**; head row 21.8–22.2 ms / fed row 0.2 ms.
**Greedy-output invariant verified**: 128-token fixed-prompt runs differ in ids by `diff` in NO token, MTP on/off at both cap 13 and auto cap (§8 invariant holds).

### 0.5.3 Measured 35B-A3B baseline (128 tokens, fixed prompt, OMP=4) — MTP correct but currently nets NEGATIVE on the streamed MoE

| config | tok/forward | Speed | expert hit | PEAK RSS |
|---|---|---|---|---|
| cap 13, MTP off | 1.03 | **6.82** | 39.4% | 2.31 GB |
| auto (RAM_GB=8 → 50 slots/L), MTP off | 1.03 | **6.74** | 67.8% | 3.60 GB |
| auto (RAM_GB=11 → 73/L), MTP off | — | (see below) | — | — |
| cap 13, MTP on | 1.72 (71.6%) | 6.16 | 23.5% | 2.62 GB |
| auto RAM_GB=8, MTP on | 1.72 | 5.66 | 64.7% | 3.45 GB |
| auto RAM_GB=11, MTP on | 1.72 | 5.30 | 73.4% | 4.55 GB |
| auto RAM_GB=11, MTP on, DRAFTS=2 | 2.02 (71.1%) | 5.23 | 72.9% | — |

### 0.5.3b Addendum 2026-11-08 — the C3 (zero-copy slots) regime, same protocol (128 tok fixed, greedy-ids identical every run)

| config | tok/forward | Speed | expert hit | PEAK RSS |
|---|---|---|---|---|
| RAM_GB=8, ZC off (copy path) | 1.72 | 4.40 / 4.62 / 4.72 | 66–68% | 3.3–3.6 GB |
| RAM_GB=8, **ZC on** | 1.72 | **5.46 / 5.56 / 5.61 / 5.64 / 5.75** | 65–69% | 5.6–6.6 GB |
| auto cap (no RAM_GB), ZC on | 1.72 | 5.50 / 5.71 / 5.73 / 5.82 | 52–69% | 5.7–5.9 GB |
| RAM_GB=16 (cap 113/L), ZC on | 2.02 | 5.05 | 82.2% | 6.32 GB |
| ZC on, DRAFTS=2 / 3 / 4 / auto | 2.02 / 2.08 / 2.12 / 1.63 | 5.39–5.71 / 5.01 / 4.70 / 5.00 | ~ | ~5.8 GB |
| ZC on, DRAFTS=1 (default) | 1.72 | **5.36–5.82** (today's best config) | ~ | ~5.8 GB |

**C9 addendum, same day (2026-11-08, MTP on, same protocol)**:

| config | head row | Speed | stream |
|---|---|---|---|
| COLI_DENSE_IDOT=0 (gq Q4_K head, reference) ×3 | 20.38 / 22.13 / 23.15 ms | 7.57 / 7.02 / 6.72 tok/s | ref |
| **C9 int8 head** ×3 | **10.96 / 10.99 / 11.06 ms (−51%)** | **8.20 / 7.98 / 7.84 tok/s** | IDENTICAL ×3 |
| C9, DRAFTS=2 / 3 / 4 / auto | 10.90 / 10.94 / 10.89 / 10.80 | 7.93 / 7.42 / 7.44 / 7.60 | IDENTICAL |

C9 = the GGUF lm_head (Q4_K, 286 MB) is converted at load to the int8-per-row copy qw_quantize makes (dequant row-by-row out of the gq blocks — no f32 staging; 508 MB int8 in, gq given back at the first allocation site that stops needing it, peak +~500 MB, steady +222 MB). Every lm_head walk — head rows, drafts, logits — now runs the measured `COLI_DENSE_IDOT` kernels. Peak RSS 7.93 → 8.48 GB. Draft phase 1.52 s → 0.85 s (48.6 → 85.9 proposed/s). Depth sweep still lands on D1: even at 11 ms/head-row the marginal draft costs the ~115 ms trunk verify row, and D3's tf 2.05 does not outrun it. C9 re-baselined explicitly (requantization, +1%-ppl class): 128-token greedy stream IDENTICAL vs the IDOT-off reference in every run, including all depths and MTP-on.

**C10a addendum, same day (2026-11-08, MTP on, same protocol)** — the trunk's GGUF dense (attn_qkv/attn_gate Q5_K, ssm_out Q6_K, ffn_*_shexp Q5/Q6) converted to the int8 idot regime at load (C9's mechanism, `Q36_TRUNK_I8` default on):

| config | T(1) verify forward | Speed | stream |
|---|---|---|---|
| pre-C10 (gq kernels) ×3 | 106.8 ms | 8.00 / 7.74 / 7.58 | ref |
| **C10a int8 trunk** ×3 | **76.4 ms (−28%)** | **10.58 / 9.73 / 10.02** | IDENTICAL ×3 |

Peak RSS 8.48 → 8.15 GB (Q5_K is 1.31 B/w, int8 1.0 — the conversion shrinks the trunk's dense). D2 9.79 / D3 9.18 / D1 10.40 — D1 stays. **Session best to date: 14.43–14.59 tok/s plain decode (14.55 mid), 12.99–13.59 with the MTP head on — through the upper half of the 10–15 tok/s objective band** (C11, the fused ARM IQ2/IQ3 grid dots; ledger §11 row C11).

**What changed in the picture**:

1. Observation 1's "expert-stream-bound" ceiling was an artifact of the pread copy path: C3's profile shows `expert_fetch→pread` went from 1382/5178 samples (~26%) to **0 samples**. Misses now cost an async `F_RDADVISE` kick + page faults the matmul already paid for in the copy path; hit reads are pure compute on mapped bytes. The wall moved: it is **compute, not IO** — trunk verifies run at a flat ~8.8–8.8 rows/s (~113 ms/row; ~40 GFLOP/s effective), and thread count 4→10 changes nothing.
2. Head rows cost 20.8–22.4 ms each and ~91% of that is the **lm_head GEMV** (`output.weight` = Q4_K, 286 MB, 0.563 B/weight exactly). `Q36_MTP_DRAFTS=1` (default) is the right default today: each extra draft costs a head row + a verify row (~135 ms) against 0.92/0.50/0.50 accepted tokens by position — break-even at best, negative deeper. **C9 landed 2026-11-08 (rows below): the head row halves to ~11 ms, but depth 2–4 still does not pay** — the marginal draft's cost is the ~115 ms trunk verify row, not the head row. Cf. the C9 addendum table.
3. RAM: the mapped file's pages count in process RSS (+2–3 GB vs the copy path — partly page-cache accounting moving into RSS), and `RAM_GB`'s copy budget is now obsolete (cap 30/layer ≥ cap 113/layer on speed; the page cache is the cache). MTP head slab (C6) unchanged at 278 MB — **C6 landed later the same day as an opt-in (`Q36_MTP_ZC=1`), copy default: see its §11 row; the mapped slab reads slightly slower than the private copy but starts without it and is evictable.**

**Observations that set direction (pre-C3)**:

1. The 35B is **expert-stream-bound, not compute-bound**: 40 layers × ~9 routed experts/token ≈ 2.7 GB/token streamed from the mmap at ~2 GB/s → hard ceiling ≈ 6.8 tok/s at this quant mix, *whatever the kernels do*, unless hit rate rises much higher or experts stay resident. Bigger expert cache helps hit rate (39→73%) but NOT wall time — misses are cheap sequential file reads, hits pay an LRU/staging overhead instead. `RAM_GB` on the raw runner also engages with cap 0 (argv), not cap −1 (rejected) — 0 = auto-sized.
2. MTP on MoE currently loses 8–22% wall time **because the head is the same weak CPU kernel**: 22 ms/head-row ≈ 6 effective GMAC/s IQ2_XXS dequant loop, the same kernel pathology as the Coder trunk (~32 GMAC/s claimed available machinery, 2% of machine bandwidth is irrelevant — it's the scalar dequant+dot loop, §1's diagnosis). At 5 drafts × 22 ms the head work eats the forwards it saves.
3. MTP **does** pay on computebound-decode models: dense 27B +15% at the sampling config; and every structural milestone (acceptance 71.6% at depth, verify-only, greedy invariant) is now green. The blocker for M2's ≥10 tok/s is §5 (fast CPU expert kernel for BOTH trunk and head rows — one code path, `gq_matmul`, serves both), not the MTP plumbing.
4. **New best baseline for ≥14B on this host: 6.82 tok/s (35B-A3B-MTP, cap 13, streamed)** — beats the Coder-30B's 5.52 with +2 GB more params, no tuning, and it also runs the MTP head. Still under the 10 tok/s floor of the ladder; gap is §5.
5. Head-row economics for M2: net-positive MTP requires head row ≲ 8–10 ms at these acceptance rates (53 accepted × Δhead vs 49 trunk forwards saved). §5's batched/quantized-activation kernels (plan §5.1) target both trunk forwards and head rows; measure "23.5 ms per extra draft" (gate's own printout) as the §5 acceptance metric for the head path.

### 0.5.4 Revised next steps
1. §5.1 CPU expert kernels first (trunk `gq_matmul` + head rows share it): int8-activation IQ2_XXS/ IQ3_XXS idot GEMV (the `q36_rows_i8` path exists for served slots — check why the streamed path misses it), thread-scale the per-row top-8 loop (today 1 row × 8 experts is serial over OMP=4 without batching). Target stat: trunk forward ≤ 60 ms at cap-13 stream mix → ≥ 10 tok/s without MTP; head row ≤ 8 ms → MTP net ≥ +25% on top.
2. Re-run the §0.5.3 table after §5.1 (same commands) — the tok/forward and acceptance numbers must stay (they are kernel-independent); only wall time moves.
3. §3.1 (`TUNE decode:` emission) unchanged; `coli tune` remains structurally unable to measure qwen36 until it lands.
4. Metal (§6) stays deprioritized for sparse dispatch (0.4.2 + Coder A/B evidence); re-evaluate only for the *dense-FFN head* variant, which has the dense pattern the 27B showed wins on.

## 0. Objective, constraints, and success criteria

### 0.1 Objective
Reach **≥10 tok/s, ideally 12–15+ tok/s** decode throughput on the local M4 (10-core CPU, 10-core
GPU, 16 GB unified memory) with a Colibrì-supported MoE model of **≥14 B total parameters**,
leveraging everything the project already ships (`plan`, `doctor`, `tune`, `bench`, `mirror`,
`serve`, `chat --attach`, `Q36_PERF_STATS`) and developing against the gaps only.

### 0.2 Constraints and doctrine
- Open, user-visible decisions recorded in logs; no overwrite of existing profiles/configs/logs.
- Model selection is free; the *engine* stays Colibrì (`c/qwen36` / `c/qwen38` / `c/colibri`…).
- Every change is validated against the CPU reference (greedy parity, GGUF_CHECK, byte-exact
  int paths) before it becomes an accepted profile — this is the project's own doctrine
  (`q36_perf` gates, `OutputDrift` disqualification in `autotune.py`).
- Mixed-IQ GGUFs are run in place; no conversion may be required for the primary model.

### 0.3 Success ladder (acceptance criteria, all measured with §6 protocol)
| Milestone | Throughput bar on M4/16 GB | Mechanism |
|---|---|---|
| M0 — baseline honesty | 5.2–5.5 tok/s (documented) | current CPU path |
| M1 — tuned launch | ≥6.5 tok/s (≥1.2×) | OMP thread sweep + residency-only config via `coli tune` |
| M2 — MTP engages | ≥10 tok/s effective (≥1.8×) | Qwen3.6-27B-MTP model + `Q36_MTP=1`, acceptance ≥0.5 |
| M3 — batched expert kernels | ≥13 tok/s | CPU batched-expert GEMV (8 experts × 3 mats in one call) |
| M4 — GPU decode path | ≥15–20 tok/s extended | Metal S>1 + batched experts + dense trunk |

M1–M2 are the realistic "10 tok/s" milestone for a 16 GB box; M3/M4 raise the ceiling and are
reusable for every sibling family.

---

## 1. Why 10 tok/s is reachable: the measured headroom

Current state (all measured this investigation, see predecessor reports):

| Metric | Value | Note |
|---|---|---|
| Decode, CPU, fully resident | 5.2–5.55 tok/s | `--cap 128` config |
| Decode, Metal IQ2 GEMV | 4.27 tok/s | correct, 12,896 dispatches, 0 fallbacks, but slower |
| Prefill | ~25 tok/s | 1,300-token curl test |
| TTFT (warm) | ~155 ms | `[qwen36 spec gate] T(1)` |
| Expert-cache hit (resident) | 57.6% → pinning ≈100% | `Expert cache hit rate` line |
| Memory bytes touched per token | ≈ 350 MB ⇒ ~2 GB/s | vs ~120 GB/s machine bandwidth |

**Diagnosis:** decode is CPU-compute/dequant-loop bound (~32 GMAC/s effective), not
memory-bound and not disk-bound. Compute budget for 10 tok/s ≈ 60–70 GFLOP/s-equivalent —
achievable with batched kernels and speculative decode; unreachable by configuration alone.
This is why the plan is: tooling repair → speculative decode → kernel work → GPU offload,
each phase independently shippable.

---

## 2. Model track — what to run first (zero new code)

### 2.1 Candidate inventory (family engine, size fit on 16 GB, MTP/quant story)

| Model | Engine family | Fit on 16 GB | Differentiator |
|---|---|---|---|
| **Qwen3.6-27B-UD-IQ2_XXS** (`Qwen3.6-27B-MTP-GGUF/`) | `qwen36` | 9.56 GB; experts 4.65 GB; dense ~2 GB; KV f16 1.5 GiB @16k | **Ships the MTP head** (`blk.64.nextn.*`, verified in file); 64/65 layers are all-IQ2_XXS gate+up+down — the exact tensors the branch's Metal kernel handles |
| Qwen3-Coder-30B-A3B-UD-TQ1 (current, imported) | `qwen36` | 8 GB; experts 7.08 GB | No MTP head; 98×IQ1_S experts are the CPU-kernel-special tax |
| Qwen3-Coder-30B UD-Q4_K_XL / UD-Q3_K_XL (HF, to import) | `qwen36` | similar size, simpler dequant | CPU-friendlier block types; fallback if 27B-MTP acceptance disappoints |
| GLM-5.3-Flash (321 B) | glm53 | 18 GB resident — **over budget** | disk-streaming decode ≈ single-digit tok/s only |
| Kimi K3, Qwen3.8-Flash-Next, DeepSeek V4 | kimi_k3 / qwen38 / dsv4 | ≫16 GB | out for this host; engine perf still earns from shared phases below |

**Decision: Qwen3.6-27B-UD-IQ2_XXS is the primary model.** It is the only ≥14 B candidate here
that exercises MTP (biggest existing lever), together with the branch's all-IQ2_XXS expert
profile. Qwen3-Coder-30B stays as the parity/second opinion and as the target for the pure-CPU
kernel phase.

### 2.2 Actions (uses existing `coli import`, `coli doctor`, `coli plan`, `GGUF_CHECK`)
```bash
cd ~/colibri/colibri
# 1. Import (link mode; nothing is copied/converted)
./c/coli import "/Users/creemac/.lmstudio/models/unsloth/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-UD-IQ2_XXS.gguf"
# 2. Verify the file, then the engine's view of it + MTP head visibility
GGUF_CHECK=1 ./c/qwen36 8 4 /dev/null 2>&1 | head -5   # engine verify-and-exit mode
Q36_MTP=1 SNAP=<imported-dir> ./c/qwen36 13 4 /dev/null
# 3. Diagnostics + plan, then a bounded served smoke
./c/coli doctor --model <dir> --json ; ./c/coli plan --model <dir> --ram 10
```
Expected diagnostics to record: engine banner `mtp=…`/`nextn_predict_layers=1`, expert bytes ≈
4.65 GB (fits at cap 128 ≈ 5.9 GB … planner decides), per-layer expert dtype report.

### 2.3 Acceptance for the model track
- Engine loads the file with zero CPU-fallback refusals (`GGUF CHECK OK`).
- `coli plan` reports ≥1 expert slot/layer at `--ram 10–12`, warm tier ≥ 4 GB.
- Baseline served runs (3×, greedy, 256 tokens) recorded in the harness log with the §6 format.

---

## 3. Tuning track — make the existing tools drive Qwen (dev, small)

These are the integration repairs identified by the investigation (`§9` of the capability
report). Each is small, independently testable, and unlocks a shipped tool on a shipped family.

| # | Task | Where | Acceptance |
|---|---|---|---|
| 3.1 | Emit `TUNE decode: N tokens in T.TTTs` per generation turn (copy `inkling.c:3267-3273` pattern) | `qwen36.c` (next to the `[qwen36 phase2] turn` printout; stdout, engine process) | `./c/coli tune` completes a full sweep on the imported model and writes `~/.config/colibri/tuning/<fp>.json` |
| 3.2 | Add `COLI_METAL_IQ2_XXS` to `TUNABLE_KEYS` in `autotune.py`, so tune A/Bs Metal on/off per hardware instead of assuming | `c/autotune.py` (constant block lines ~31-40) | profile shows both candidates measured, winner + gates recorded |
| 3.3 | Plug real Metal detection into the doctor GPU check (extend `_gk_backward`-style probe set with the Mach-O Metal-link test used in `logs/QWEN3_CODER_30B_MAXIMUM_LOAD_TEST.md §1`) | `c/doctor.py` `accelerator.gpu` check (~line 712) | doctor reports Metal-capable engine as GPU-capable (darwin) |
| 3.4 | Replace hardcoded `plan_has_metal=False` with the true value from `_discover_metal_gpus()`; keep the NPU path untouched | `c/resource_plan.py:3677` | on darwin, `_auto_tune` emits its Metal OMP-spin knob in the plan JSON |
| 3.5 | `c/coli` (launcher) gains `--metal-experiments` (or honours the env) so `serve`/`chat` set `COLI_METAL_IQ2_XXS` consistently; document in `docs/ENVIRONMENT.md` | `c/coli` `env_for_engine()` | `coli chat --attach` shows the two modes diverge in measured tok/s |
| 3.6 | Count the CPU fallback when Metal is *disabled* (today `q36_perf_iq2_fallback` only fires after a failed attempt) | `qwen36.c` matmul paths | `=0` runs report non-zero fallback counts on an IQ2-heavy model |
| 3.7 | Extend `bench` to sibling families: greedy, fixed-prompt, exact/agreement scoring against `c/ref_qwen36.json`-style reference sets (per-family ref files next to the harness tasks) | `c/coli bench` + new `c/tools/bench_qwen.py` | `./c/coli bench qwen36 --limit 20` produces a score line that tracks GLM-family harness stability |

Notes:
- 3.1–3.2 unblock **`coli tune`** — the project's measured-profile doctrine — for the highest-value
  family; without them, the only "tuning" available is manual.
- 3.3–3.4 make the planner able to *opinion* rather than shrug on Apple GPUs.
- 3.7 gives the kernel work in §5 a quality gate per milestone: fastest path must score within
  tolerance of the CPU reference (the same doctrine the GLM engine already enforces).

---

## 4. Engine track A — make MTP pay (dev, targeted)

Current machinery (verified in source): `qwen36_mtp.h` (head loader + expert sidecar
loader), `Q36_MTP=1` enablement (`qwen36.c:794`), `nextn_predict_layers` consumption
(`qwen36.c:3724`), verify-forwards batching in the turn loop, and the `spec gate` / `lookup`
stats lines. What is unproven is the *wiring against this GGUF* and its measured acceptance.

### 4.1 Work items
1. **Map `blk.64.nextn.*` tensors into the engine's head loader.** The GGUF carries
   `nextn.eh_proj / enorm / hnorm / shared_head_norm` (verified). If `qwen36_mtp.h` expects a
   sidecar format (its "expert vocabulary / float shared lm_head" loader), add a GGUF-native
   branch reading `nextn.*` in place — consistent with the engine's in-place doctrine.
   File refs: `qwen36_mtp.h:79-136`, `qwen36.c:3724-3726`.
2. **Verify batching**: confirm `verify forwards` accept 1–3 rows (S≤4 in the MTP verify loop)
   and that draft hidden states reuse layer caches (no re-prefill). `qwen36.c` verify path +
   `Q36_PERF_STATS=1` evidence.
3. **Tune the gates**: the spec gate currently declines (`gate on, N declined, M probes`).
   Tune `rho`/acceptance thresholds conservatively first; any quality-affecting threshold is
   excluded from `coli tune` candidates by design — expose as env, measure, then ship a default
   after parity (MTP acceptance never changes greedy outputs: verify-and-accept only).
4. **Profile** three prompt shapes: code-generation (high local token reuse), prose, and
   chat-with-context (the agent use-case from the `cafe` test). Acceptance target ≥0.5
   accepted/verify on two shapes, measured effective tok/s ≥1.8× baseline.
5. **Persist**: fold the winning gate + MTP on/off into a saved profile once 3.1/3.2 land, so
   `serve`/`chat` pick it up via the existing `load_profile`/`apply_measured_profile` path
   (already wired for non-GLM engines).

### 4.2 Expected effect
- 2-row verify at ~60% acceptance ⇒ ~1.6 tokens/forward ⇒ **8.5–9.5 tok/s** from the 5.5 base
  *without any kernel work*; combined with §5 (batched experts ⇒ forward cost ÷1.5) ⇒ 10–14.

### 4.3 Risks
- GGUF `nextn` tensor dtype mismatch (defaults to Q8/large dtypes in some exports) — mitigation:
  same `gq_*` table, plus reference parity vs CPU baseline.
- Draft acceptance lower on pure Coder-style outputs — mitigation: gate remains cheap
  (T(1)-overhead measured ~230 ms amortized over ≥50 tokens).

---

## 5. Engine track B — the CPU expert path (dev, medium)

The IQ2_XXS Metal GEMV (`backend_metal.mm:163,2065-2150`) proved the shape of the win; the CPU
side has the same opportunity far cheaper. Per generated token today: **1,152 separate GEMV
invocations** (48 layers × 8 experts × gate+up+down), each `falloc`s, dequant-walks, and
dot-products one 1.57 MB matrix independently.

### 5.1 Work items
1. **Batched-expert entry point**: one call per layer that runs all selected experts consecutively,
   sharing the activation vector and thread fan-out:
   - `moe_gq_cpu()` group call (mq/paired path near `qwen36.c:5739-5803`) becomes one
     `gq_matmul_multi(y[K][D], x, exps[K], types[K], I, O)` invocation.
   - Verify/lookup paths (`provide verify rows`) reuse the same kernel with S>1.
2. **Kernel specialization per dtype** (the files already carry per-type decode tables in
   `ggml_quants.h`): hoist per-block scale groups out of the dot loop; NEON `sdot/udot` for
   the int lanes; 4-row accumulators (the codebase already has `dot4_f32_lanes`).
3. **Memory access order**: iterate experts outer/weights inner so the resident slab stays
   sequential (hardware prefetch friendly); overlap of slot-hold/`slot_release` with compute
   so the LRU never blocks the current layer.
4. **Parity gate**: every batching refactor must reproduce byte-identical greedy output on the
   128-token sea-canary prompt and on `ref_qwen36.json` equivalents before merging.
5. **Instrument**: `Q36_PERF_STATS` gains `experts batched calls / rows` counters so regressions
   are visible in the same turn lines the tuner parses.

### 5.2 Expected effect
- Kernel-level 1.5–2× on expert generation work ⇒ decode ~8–11 tok/s CPU-only (measured wall).
- Combined with §4: **12–16 tok/s** — the headline goal, reached on 16 GB.

### 5.3 Risks
- Register pressure regressions on the P/E asymmetric core count — mitigate with the §3.1
  OMP sweep landing in the tuned profile rather than hardcoding thread counts.

---

## 6. Engine track C — Metal decode path (dev, large, run after B)

The branch already owns the hardest part (correctness of one native kernel + zero-copy
residency via `coli_metal_register`). Extending it is incremental.

1. **Remove the S==1 gate** on `mm_iq2_xxs_gemv` (decode-only today): enable the 2–4 rows the
   MTP verify path already produces (`qwen36.c:2246-2257`, `2257-2271`).
2. **Batched-expert Metal call**: one command encoder per layer processing all 8 selected
   experts (`coli_metal_iq2_xxs_gemv` becomes per-K-expert arrays) — dispatch amortization is
   what made Metal lose on this host (4.27 tok/s), and this is its fix.
3. **Pull in IQ1_S / IQ3_S expert types**: same template; the Coder-30B model's 98×IQ1_S then
   stops being "the CPU tax", and Metal coverage jumps from 26/114 expert tensors to ~88%.
4. **Dense trunk on Metal (last)**: attention/QKV/output matmuls via the same pipeline
   factory; gate behind `COLI_METAL_DENSE` until parity-gated (§3.7 harness as the gate).
5. **Planner integration**: once the engine reports per-phase Metal dispatch (3.2/3.6
   telemetry), let `coli tune` pick the best executor combination per host and persist it.

Acceptance: on M4 ≥ M3 bar with the Coder model — and the 27B model should show the clearest
gain (all-IQ2 layer Mix, 64 of 65 layers).

---

## 7. Serving track — `coli serve` / `chat` end-state configuration (no dev)

After M1, the recommended served config for the migrated model (16 GB host):

```bash
PILOT=2 WIDE=2 Q36_MTP=1 Q36_MAXT=16384 Q36_PERF_STATS=1 \
./c/coli serve --model <27B-UD-IQ2_XXS-dir> --ram 12 --port 8000
```

Plus, once 3.1–3.2 land, the same launch after `coli tune` reads its profile automatically
(`[TUNE] applied measured profile` in the stderr banner) — that is the moment configuration
stops being hand-tuned and becomes a persisted engineering artefact per machine/model pair.

For the agent use-case (the `cafe` experiment), prefill remains CPU-bound (~25 tok/s at this
scale): big contexts should use streaming (`stream:true`) + trimmed system prompts. If
prefill becomes the dominant product constraint, its fix is a §6.4-style Metal dense trunk,
which mainly accelerates prefill-shaped workloads — a separate decision to be made with data.

---

## 8. Measurement protocol (every claim in this plan is re-verifiable)

Harness (per milestone, same format as predecessor reports):
1. Three consecutive runs of a fixed 40-word prompt, `--ngen 256`, `temperature 0`.
2. Record: wall, TTFT (`spec gate T(1)`), decode tok/s (`lookup tokens/forward`, server line
   `[serve] … tokens per turn`), expert hit (`Expert cache hit rate`), IQ2 dispatch counts
   (`phase2 by phase`), RSS (`PEAK RSS` / Activity Monitor real mem), swap (`sysctl vm.swapusage`).
3. parity: greedy output equal to M0 baseline byte-for-byte (or diff reported explicitly).
4. Store results per milestone in `logs/PERF_M*_REPORT.md`; keep raw server logs in `logs/`.

Sweep tool: after 3.1, `./c/coli tune --model <dir> --tokens 32 --repeats 2` replaces the
manual loop; its profile JSON is the acceptance artefact for M1.

---

## 9. Milestone order and dependency graph

```
M0 baseline (done, measured)
 │
 ├─ §3.1–3.2 TUNE line + Metal knob      → M1 tuned launch (configs only, no kernels)
 ├─ §2 model import + plan check         → primary model chosen (zero code)
 │
 ├─ §4 MTP (needs §2 primary model)      → M2  ≥10 tok/s        [biggest single lever]
 ├─ §5 CPU batched experts               → M3  ≥13 tok/s        [biggest engineering value,
 │                                                                 family-agnostic]
 └─ §6 Metal S>1 + batch + dense trunk   → M4  ≥15–20 tok/s     [extendable, late]
        (needs §5's parity gates to exist for reuse)
```

Everything above M1 is additive, parity-gated, and reversible per feature flag
(`Q36_MTP`, `COLI_METAL_*`, `--cap`-levels) — so the plan can stop at any milestone while
retaining the previous one as the shipped default.

## 11. CHANGE LEDGER — every code/config change, tracked (status: PLANNED → WIP → DONE / REJECTED / RISK)

One row per change. Measured effects fill in as work completes; §0.5's lettered findings (C1/C2) continue here as ledger items. Protocol for every speed claim: §8.

| id | status | files | what | why / measured |
|---|---|---|---|---|
| C1 | DONE | `c/qwen36_mtp.h`, `c/qwen36.c` | GGUF-native MTP head loader + dense/`gq_matmul` expert MoE draft path (§0.5.1–0.5.2) | 1.72 tok/forward, 71.6% acceptance on 35B; greedy ids identical |
| C2 | REJECTED | config (no code) | `PILOT=3 WIDE=4` prefetch on the streamed 35B | 3.29 tok/s vs 5.3–6.8 baseline (−40%): prefetcher from this era thrashes on a 256-expert/streamed workload. Do not use for the 35B; PILOT stays for the Coder serve doc. |
| C3 | DONE | `c/qwen36.c` | **Zero-copy GGUF expert slots** (`Q36_GQ_ZC`, default on): one RO mmap per shard at open (`compat_map_readonly`, `fstat` size); slots point at `g_zc_base[tensor->file] + tensor->off + eid*eb` (`Slot.zcg/zcu/zcd`), `load_expert_merged` returns without the three preads; `slot_ensure_allocated` skips the 3 MB malloc for ZC layers; every CPU expert reader (`moe_vk_gq_cpu`, `moe_gq_decode`, `moe_gq_run`, `gq_slot_i8`, `gq_slot_file_rows`) answers through `gq_slot_W(slot, x, k)`. Eviction = bookkeeping + `posix_fadvise(DONTNEED)` hint; publish = `posix_fadvise(WILLNEED)` kick (macOS `F_RDADVISE`, async) + `MADV_RANDOM` on the map. | §8 A/B (128 tok, RAM_GB=8): copy 4.40/4.62/4.72 tok/s → ZC 5.46–5.75 tok/s (**+18–22%**, three back-to-back pairs). Profile: `expert_fetch→pread` 1382/5178 → 0 samples. Greedy ids IDENTICAL (off/on and reruns). iter notes: (1) madvise `MADV_WILLNEED` on the mapping pages the range in synchronously (~15% of samples) — fadvise instead; (2) `RAM_GB` copy budget is obsolete: cap 30/layer auto = 5.50 tok/s, cap 113/layer (RAM_GB=16) slower (5.05) — the page cache is the cache now; (3) mapped file pages count in process RSS (~+2–3 GB vs copy, some is the same page cache moved into RSS accounting). Kill-switch `Q36_GQ_ZC=0` restores the byte-copy path (verified identical ints). |
| C3b | DONE | `c/qwen36.c` (slot_ensure_allocated/load_expert_merged/tier tmp sites) | ZC plumbing side-work: eager `gguf_zc_init()` on the first slot allocation (no wasted gqe malloc), `layer` threaded through `slot_ensure_allocated`'s signature, tier tmp-slot sites moved into scope, `slot_gqe_free` also clears zcg/zcu/zcd | Build-clean; all greedy-identical runs above include it |
| C4 | REJECTED (as designed) | `c/qwen36.c` `moe_vk_gq_cpu` | Hoist the per-row activation quantization (`gq_act_set`) out of the per-expert loop | Measured reality: `moe_vk_gq_cpu` is the Vulkan-tier CPU fallback, not on the M4 macOS path (the hot decode path is `moe_gq_decode`, which already quantizes once via `pax`). Phase-1 (2026-11-08) profile shows `expert_fetch` at 1.5% and MoE matmul at ~19% on the main thread share; the re-quantization site is not on the critical path. |
| C5 | REJECTED (as designed) | `c/qwen36.c` `moe_gq_decode` | Parallelize the (row, expert) pair loop with OMP `schedule(dynamic,1)` | Already parallel (`#pragma omp parallel for schedule(static)` over the row-group loop at ~5817/5848). Thread sweep on the ZC build: OMP 2→3.85, 4→5.73, 6→5.63, 8→6.03, 10→5.80 tok/s — the trunk is a flat ~8.8 rows/s (~113 ms/row, M4 4P+6E); more threads give nothing. Parallelism is not the wall. |
| C6 | DONE (2026-11-08, opt-in) | `c/qwen36_mtp.h` | MTP head expert slab (278 MB duplicated RAM copy) → read the head block's `ffn_*_exps` through the same C3 mmap views (zero-copy), keeping the per-expert `.scale` factors | Implemented in `q36_mtp_load_gguf`: with the C3 maps up, the three slab pointers land at `g_zc_base[file] + t->off` and `q36_mtp_moe_gguf` reads the same bytes through `gq_matmul` (types are the file's already); one async `F_RDADVISE` per tensor, no `DONTNEED` republication (the head stays hot). **Default stays the copy path**: 6 alternating pairs measured the private copy equal-or-better (8.27 vs 8.11 tok/s avg; 5/6 pairs, the +0.52 outlier carried most of the gap) — the slab's private pages are never evicted by the trunk's expert traffic, while mapped pages are reclaimable cache and a draft eats the fault. `Q36_MTP_ZC=1` opts the slab into the zero-copy regime (`Q36_GQ_ZC=0` still forces the copy everywhere): faster startup (no 278 MB read), evictable instead of molten, ~130 MB less measured peak RSS (faulted pages count in RSS, only partially). Outputs IDENTICAL in every pair vs both arms and vs the pre-C6 reference including all C9 configs. |
| C6b | DONE (measured, no code) | config | `Q36_MTP_VOCAB`/`VOCAB_IDS` adaptive draft vocabulary | Measured on the 35B: head rows 24.2 ms (vocab 30000) and 29.6 ms (65536) vs 22.4 unset — SLOWER (indirect `vid` row gather beats the caching). Do not enable. |
| C7 | DONE (protocol runs throughout) | — | Re-measure after each change with §8 (128-token fixed prompt, greedy ids diffed) | All speed claims today: greedy-identical outputs vs the ZC-off reference (`diff` on `^C engine` = IDENTICAL every run). Ledger rows carry their numbers. |
| C9 | DONE (2026-11-08) | `c/qwen36.c` `load_tq` (GGUF branch) | **Convert the GGUF lm_head to the int8 dense copy at load.** `output.weight` = Q4_K sat on the gq path while the engine already ships a measured int8 idot regime (non-GGUF load: lm_head 12.6→10.2 ms, +1.0% ppl, default on). Behind `dense_idot_on()` && `dense_bits()==8`, tag `lmhead`, non-IQ2: dequant each row out of the gq blocks (exact-to-the-bit `gq_dequant_row`, r16-aware), then qw_quantize's contract (one per-row scale amax/127, clamped ±127, the file's `gs` folded into the row scale), int8 buffer via `q36_walloc`; the gq blocks are freed and `q36_gq_held` already ran before. Row-by-row: no f32 staging matrix. The routed experts are untouched (zero-copy slots, 2–3 bit types). COLI_DENSE_IDOT=0 skips it entirely = the true reference path. | Measured (128 tok, Q36_MTP=1, OMP=4, alternating pairs ×3): head rows 20.38–23.15 → **10.96–11.06 ms (−51%, dead flat)**; tok/s 7.57/7.02/6.72 → 8.20/7.98/7.84; draft phase 1.52 s → 0.85 s (proposed/s 48.6→85.9); spec gate T(1) 170.8→134.6→104.3 ms across the three configs, verify target-rows/s 10.76→11.39. Peak RSS 7.93→8.48 GB (both copies during conversion; steady +222 MB). Depth sweep on the new build D2 7.93 / D3 7.42 / D4 7.44 / auto 7.60 vs D1 8.01 — **D1 stays the default**: the 11 ms head row no longer dominates a draft, but the ~115 ms trunk verify row does. Re-baseline: requantization (+1%-ppl class), yet 128-token greedy streams IDENTICAL vs IDOT=0 in every pair, every depth, MTP on and off. |
| C10 | DONE (2026-11-08) — C10a shipped default-on; C10b trialed and REJECTED, superseded by C11 | `c/qwen36.c` `load_tq` GGUF branch | **C10a: the trunk's GGUF dense joins the int8 idot regime at load** (C9's one gate extended: tag `lmhead` OR the trunk default; Q5_K/Q6_K/Q8_0 — the routed experts and the IQ2/IQ3 types never touch this branch). The gq kernels served `attn_qkv` (Q5_K), `attn_gate` (Q5_K), `ssm_out` (Q6_K), `ffn_*_shexp` (Q5_K/Q6_K) per layer ~25.7 MB; the conversion is also a net RAM shrink there (Q5_K 1.31 B/w → int8 1.0). **Default on**, `Q36_TRUNK_I8=0` keeps the file's blocks (e.g. a device tier reading the file's formats). | Measured (128 tok, Q36_MTP=1, OMP=4, alternating pairs ×3): **8.00/7.74/7.58 → 10.58/9.73/10.02 tok/s (+30%)**; spec gate T(1) 106.8 → 76.4 ms (−28%); head rows 11.31 → 10.76 ms (the head's own dense joined too); peak RSS 8.48 → 8.15 GB. Depth re-sweep on the new row cost: D1 still best (10.40 vs D2 9.79 / D3 9.18). Streams: all pairs IDENTICAL off vs on — the third numeric regime (gq Q5_K/Q6_K → int8 per-row) also flipped zero argmax in 128 tokens; treated as re-baselined, not bit-identity. **C10b open:** the row is now ~76 ms of compute (from 113) — next levers: ARM idot kernel time on the fused qk/v projections, the Metal dense question (C8), a profile pass to see what the 76 ms is made of. |
| C10b | TRIALED 2026-11-08 — REJECTED as built, code kept behind `Q36_EXP_I8` | `c/qwen36.c` (`q36_slot_i8`, `q36_row_i8_from_gq`, both `moe_gq_decode` and the S>1 `moe_gq_run` got i8 branches) | **Routed experts in the idot regime too**: a slot's three IQ2_XXS/IQ3_XXS matrices also kept as per-row int8 + scale (qw_quantize's contract), read by the same `dot_i8i8` gates/up/down loops the dense matmuls use; activation quantization once per matmul input as the dense path does; ".scale" factors stay outside. Applied **only** under `Q36_EXP_I8=1`. | Mechanically sound (clean build, greedy stream IDENTICAL vs default in every run) but **the conversion is the wall**: on-the-fly misses (hit rate ~65–70% × 8 routed experts) pay a serial dequant + quant of 3M weights ≈ 3 ms/expert; at ~96 expert-misses per verify row that is +~290 ms of row the i8 kernels never earn back — 128-tok wall 12.4 s → 55 s once (scalar quant, Q36_EXP_I8=1 → **1.56 tok/s**), and with a NEON amax + transform+clamp+cast quantizer (the vector pass landed) → **2.32 tok/s**: the scalar block-dequant (IQ2/IQ3 per-block code) is the remaining 80%. Verdict: revisit only after the IQ2/IQ3 kernel-side dequant goes vector (est. ~0.5 ms/expert ⇒ viable, then saves maybe ~20 ms of the 44 ms MoE block on a well-hit cache). Metal IQ2 GEMV, measured adjacent (3 alternating pairs to C10a's default): 10.49/9.86/10.09 vs CPU 10.79/10.28/9.72 — **no win** (C8's sparse-dispatch evidence again); `COLI_METAL_IQ2_XXS` stays default-off. |
| C10b' archived phase data (COLI_TIMERS=1, Q36_MTP=1, the T(1) design row) | — | — | What the 76.4 ms verify row was made of after C10a: **MoE 42.7 ms (56%)** (the routed experts' IQ2/IQ3 slot GEMVs), DeltaNet 18.1 (24%, its `proj` 11.4), lm_head 7.0 (9%), attention 4.6 (6%) — the shared expert and router visible as subsets (2.7/1.4). | After C10a the wall is the routed-expert MoE; C10b's rejected trial (above) indicates what matter it takes down the MoE block; other machine opportunities are the DeltaNet proj and the idot kernels' ARM efficiency on the fused QKV (which shows inside DeltaNet `proj`). |
| C11 | DONE (2026-11-08) — default on (no switch; the old path is the fallback for any ARM node without dotprod at build time) | `c/ggml_quants.h` (`gq_arm_sb_iq2_xxs`, `gq_arm_sb_iq3_xxs`, `gq_arm_dot2`, `GQ_DOT_GRID_ARM` fused kernels, dispatch in `gq_row_all`'s non-AVX tail) | **The routed experts' IQ2_XXS/IQ3_XXS GEMV goes fused on ARM**, mirroring the AVX2 `GQ_DOT_SB256` family: the sub-block decoders the NEON unpack already runs (same gathers, GQ_SIGN8 rows, same f16 scale expressions) now produce their 32 int8 in registers and join `vdotq` at once — the old ARM tail unpacked every weight row into a thread buffer (`gq_unpack_row`) and dotted it back out (`gq_dot_i8`), a store + reload + type switch per weight row. Dispatcher: `I % 256 == 0` else the old path; `GQ_F8_ON`/`GQ_R16_ON` are x86-only so on the M4 every MoE row came through `gq_row_all`'s tail. Constructed bit-exact with the unpack path (same integer product lanes, same scale products, same accumulator order), and it measures so. | Measured (128 tok, Q36_MTP=1, OMP=4): **MoE block 42.7 → 28.4 ms (−33%)**, verify row 76.4 → 58.1 ms; **tok/s 10.31–10.45 → 12.99–13.59** (runs 13.39/13.59/13.46/12.99 + timers run 13.39); head rows 10.11–10.40 ms. Greedy stream **IDENTICAL** vs the C10a reference in every run (five) — the exactness construction shows. **Plain decode (Q36_MTP unset — the engine's default) with the row now cheap: 14.43/14.48/14.59 tok/s, stream IDENTICAL**; MTP turned from a win into a slight loss (D1 draft rows cost ~10 ms each against a 58 ms row, the acceptance no longer pays): the server-style configs read **Q36_MTP=0/unset** now; the MTP head code stays shipped and opt-in. Depth re-sweep on the new row cost: D1 still best (12.92 D0 / 12.40 D2 / 11.00 D3). This replaces C10b′ as the MoE lever: with the fused dot the routed MoE is ~24 ms, and C10b's i8 slots (conversion ~1 ms/expert × ~120 fresh slots/row) would have to beat a read-only idot estimate it no longer beats — **C10b stays rejected, do not revisit without a new profile**. Remaining row: DeltaNet 18.2 (proj 11.5, ~88 GMAC/s — near the idot kernel's ceiling), lm_head 7.0 (~72 GMAC/s), attention 4.6; the routed MoE is ~24 ms and now bandwidth-shaped (2.4-bit expert bytes streamed at ~12 GB/s). |
| C8 | RISK (not started) | `c/backend_metal.mm`, `COLI_METAL_IQ2_XXS` | Re-evaluate Metal IQ2 GEMV for the 35B's *dense-FFN head* variant only | Dense-27B precedent is +50%; sparse MoE precedent is a loss. Only if MTP head rows stay >8 ms after C3–C5. |
| C12 | DONE (2026-11-09, agent track lever 1) | `c/qwen36.c` (new `q36_pcache_*` section before `ensure_kv`; hooks in `q36_serve_start`, `generate`, and `main`'s SNAP guard) | **Persistent prompt cache — the conversation state's KV rows, DeltaNet recurrence + conv windows and MTP pending row photographed to disk, keyed by the token prefix.** A snapshot (`[SNAP]/.promptcache/pfx_<fnv1a-h1><fnv1a-h2>.bin`, ~72 B header + `magic "Q36PC01"`, 12×i32 geometry, model key = fnv over family tag + every shape the layout depends on, payload hash fnv-chained over hdr+key then the payload) stores ids[0..np), per attention layer K then V re-laid t-major `[t][head][dim]` through `q36_kv_at` (restore assumes nothing about a future `kv_cap`), one rec + one conv row per DeltaNet layer (via a fresh `q36_pin_state_save` — ~63 MB), the vocab logits row, and the MTP pending row when valid. Restore probes the directory (header + ids memcmp, deepest strict prefix wins), then replays the winner's bytes hashing along the write order and only installs on hash equality (`q36_pin_state_put` installs the DeltaNet/MTP state incl. the device mirrors' bookkeeping; rollback mirrors like `q36_unpark`; `kv_len = len`; `kv_prefix_record` ids[0..len)); any failure leaves only rows what a cold turn rewrites and the caller's `!reuse` branch rewrites them. CLI `generate()` and the single-slot serve path both restore-and-prefill-only-the-tail; store fires after every successful prefill ≥ min. Knobs: `COLI_PROMPT_CACHE` (default on when `SERVE=1`, `=0` kill switch, `1` forces CLI), `COLI_PROMPT_CACHE_MIN` (1024, floor 64), `COLI_PROMPT_CACHE_MAX` (8 files, 0–64, prune newest-kept / deepest-first among the rest, `keep_path` survives), `COLI_PROMPT_CACHE_DIR`. Debug: `COLI_PCACHE_LOG=1`. Scope v1: single-slot serve, generation requests (logprobs requests pay full prefill, the rule reuse itself follows), no vision payloads (`vis_map`), no sliding-window models. | Measured on the 35B-A3B (M4, OMP=4, no hybrid knob): **(a) CLI A/B** on the doubled 1,839-token prompt (`/tmp/p2k2.txt`, 3,678 tokens; prefix identity verified via TOK_ONLY — 3,678 = 2×1839 ids exact): cold TTFT 163.5 s (pristine HEAD binary) → **80.1 s restored** (2.04× faster first token; the restored run saved its own deeper 3,678-token snapshot for the next turn). (b) **Greedy identity: the restored-tail run's 16-token continuation is byte-identical to the pristine cold run** (OPENAI chunk stream compared token-for-token: "Case 0025: The compiler analyzes..."). (c) **Serve end-to-end** (coli serve, single slot): turn 1 cold 67.3 s → turn 2 (same transcript + new user/assistant messages, `prompt_tokens` 1853→1882) **`restored 1853 of 1882` → 2.0 s total HTTP latency (≈34×)**. (d) Roundtrip across process restarts and store idempotence (same prefix skips), eviction (`MAX=1` keeps the just-written, drops the rest) verified. (e) Known boundary: appending text inside the SAME user message can merge the last tokens (`ids diverge at 1844`) and the snapshot correctly refuses; new messages (the agent shape, template tokens between) restore — messages must be sent as separate chat messages. (f) Repair history: store staging blob missed the `kv_heads` factor (heap overflow, SIGSEGV post-store — pre-existing crash excluded via pristine-binary control + `COLI_PROMPT_CACHE=0`), probe forgot the 8-byte hash slot (ids read from offset 64), apply loop lost its K/V `fread` in a rewrite — all fixed and re-verified. (g) Tokenizer note: `restored N of np` prints make the column visible. |
| C13 | RISK (crash reproduction unconfirmed) | `c/qwen36.c` (`load_expert_merged`) | **Hybrid expert slot mapping lifecycle.** A corrected run of the reported 3,678-token command (`SNAP=... Q36_HYBRID_GB=6 OMP_NUM_THREADS=4`, cache off) completed both before and after the code change; the earlier crash therefore is not currently reproducible and cannot be attributed to this fix. While investigating, found and corrected a real pointer-lifetime defect: `load_expert_merged` published zero-copy pointers before `slot_gqe_free`, which clears those pointers. Cleanup now precedes publishing views, with an explicit incomplete-map check. | The reported SIGSEGV is **not confirmed fixed**: proper-command post-change hybrid run completed, but a proper-command pre-change run also completed. Keep this risk open until the original failure is reproduced. |
| C14 | REJECTED (measured slower) | `c/qwen36.c` `pilot_prefetch` | Lift the `S <= 8` gate and overlap predictive expert prefetch with long prefill. Tested sampled prediction (up to 8 prompt rows) and exact one-expert-ahead scheduling; removed the exact scheduler and restored the `S <= 8` call gate. | Identical eight-token output, but 1.8k-token TTFT regressed **68.997→77.941 s** with sampled prediction and **69.711→80.578 s** with one-ahead scheduling. Do not enable for long prefill without a new measured design. |
| C15 | REJECTED (greedy identity failure) | `c/qwen36.c` `attention` | Tile four adjacent prefill queries to improve K/V cache reuse. Removed after A/B showed a different greedy stream on the 1,839-token prompt: tiled output diverged from pristine at the first generated token; the untiled implementation matched it byte-for-byte. The tile also remained unproven for speed. | No attention tile ships. Plain greedy parity verified with `COLI_LOOKUP=0 Q36_MTP=0` on the 3,678-token prompt (3 generated tokens, identical stdout); untiled default path also matched the pristine 1,839-token stream (12 generated tokens). |

Constraint reminders written into each implementation: engine stays Colibrì; no branch switches/commits; measure before/after with the same §8 command; keep greedy-output invariant (diff `C engine` ids off vs on for every change).

## Quick run: Qwen3.6 OpenAI-compatible server (tested 2026-11-09)

From the repository root, generate a fresh API key and start the tested Qwen3.6-35B-A3B model on port 8001:

```bash
export COLI_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
printf 'API key: %s\n' "$COLI_API_KEY"
python3 -m colibri.cli serve \
  --model "/Users/creemac/.lmstudio/models/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-IQ2_XXS.colibri" \
  --host 0.0.0.0 --port 8001 --api-key "$COLI_API_KEY"
```

Copy the printed key into a second terminal, then confirm the model is listed and send a short completion:

```bash
export COLI_API_KEY='paste-the-printed-key-here'
curl http://127.0.0.1:8001/v1/models \
  -H "Authorization: Bearer $COLI_API_KEY"

curl http://127.0.0.1:8001/v1/chat/completions \
  -H "Authorization: Bearer $COLI_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-colibri","messages":[{"role":"user","content":"Reply with exactly: Server test passed."}],"temperature":0,"max_tokens":24}'
```

Verified response: `Server test passed.` (`prompt_tokens: 20`, `completion_tokens: 4`). The API key is generated at launch; keep it private. `0.0.0.0` accepts connections from other devices on reachable networks, so use only on a trusted network and apply appropriate firewall rules.

## 10. Immediate next commits (in order, smallest first)

1. §3.1 `TUNE decode` emission in `qwen36.c` — unlocks the whole tuning doctrine on Qwen.
2. §2 import + `doctor/plan` JSON snapshot of the 27B-MTP file (read-only evidence).
3. §4.1.1 GGUF-native MTP head loading — the 2× lever.
4. §5.1 batched-expert CPU entry point + parity gates.
5. §3.3–3.5 launcher/planner Metal honesty (enables the `coli tune` decision between
   CPU and Metal without user archaeology).
6. §6 extensions, ordered by §6 list, each behind its own env flag.
