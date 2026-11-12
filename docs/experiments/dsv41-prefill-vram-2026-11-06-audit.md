# Track B — colibri prefill VRAM audit (DeepSeek-V4.1-Flash, Vulkan chain, 16 GiB RX 6900 XT)

Status: measurements + code-cited allocation map. Read-only on tracked files; the new
`vram-*.csv/log/out`, `vram-proc-sampler.py`, `vram-analyze.py`, `vram-run-b2.sh` and this file are
untracked scratch in `/root/Projects/colibri`.
Companion document: `/root/Refs/trackA-deepinfer-vs-colibri.md` (recommendations R1–R6). Line numbers below
are from this checkout; all sizes are GiB (2^30 B) unless marked MiB/KiB.

---

## 0. Executive summary

On this 16 GiB card the engine loads to **9.64 GiB** (trunk 7.62 + warm expert tier 1.87; device total 12.15
with the 2.51 GiB non-engine floor) and every mid-prompt arm peaks at **15.91–15.92 GiB of the 15.98 GiB device** (13.40–13.41 GiB of engine
VRAM) — i.e. the peak leaves 68–79 MiB of the card, and **it is the same peak for ctx 16384 and ctx 65536**
(15.907 GiB at chunk 1280, 15.918 GiB at chunk 256, byte-identical step for step in both pairs) and **the same
peak for chunk 1280 and chunk 256** (15.907 vs 15.918 GiB, engine 13.40 vs 13.41). Nothing that matters is sized by the configured context: the indexer
scratch is `rows × E` with `E` the *forward's end position* (the prompt length), the KV mirrors' capacity
doubles to the conversation length, and the window ring is `40 × (128 + rows)`. The two real variables are
**chunk rows** and **prompt length `E`**, and they set the *prefill plateau*, not the peak: the engine prints
**1293 KiB/row at E≈11 and 1317 KiB/row at E=3107** (the 24 KiB difference is exactly `8·E`, the `isc`+`mask`
pair) and a per-forward chain footprint of 75.4 MiB at 11 rows, **1765.7 MiB at 1280 rows**, 739.9 MiB at 256
rows, and 2133.7 MiB once `E` is fully populated; cutting the chunk 1280 → 256 lowers the prefill plateau by
0.6–0.9 GiB but the peak, which arrives at the *turn boundary* (last chunk → decode → next turn's first chunk)
when the tier promotes itself to its 3.19 GiB budget, does not move at all. That single fact reorders the
levers: the chunk is a **throughput** knob (chunk 256 costs 2.33× prefill wall and 4.1× the CPU-fallback
expert-rows for the same prompt), not a peak-VRAM one. The "~2.2 GiB of device growth per 16K turn" quoted in
the Track A document is an artifact of comparing the pre-load floor with the post-first-forward state on the
short-prompt arm: 12.15 GiB is the post-load floor, not a pre-load number, and the first forward — not the
turn — is what allocates (+1.7 GiB of chunk buffers at 1280 rows, ~1.1 GiB of tier bookkeeping: streaming
slots, the extra-layers pool, promotions that continue to +1.32 GiB over the first turn). Verdicts: R1's "no new memory" is **false as stated** (tiling
`isc`/`mask` bounds the pair at a tile instead of up to 256 MiB, while the chunk it enables costs 978 KiB per
row of fp32 scratch), its **direction and speed factor are confirmed** (2.33× measured in the inverse
direction), and its **peak effect is nil**; R2's saving is **0.02 GiB at a 3107-token prompt** (mirrors are
`E`-sized, not ctx-sized) and only becomes a lever at ≥64K-*token* prompts; R3–R6 have zero VRAM effect. The
levers the comparison missed are in the tier: the `16 rows or more` device-streaming threshold (which the
engine's own log says was never measured) is what costs the 2.33× at small chunks, the streaming-slot pool and
the 232 MiB extra-layers pool are invisible to the engine's VRAM accounting and to the docs, and the "5 GiB
mapped window" is a **probe**, not a resident allocation.

---

## 1. Protocol and what the numbers mean

Harness: `prefill-vram.py` (unchanged), one `./c/coli serve` per arm, `COLI_VULKAN=1 V41_STATS=1`, model
`/root/Models/DeepSeek-V4.1-Flash-1017`, `--ram 28`, 2 turns (warm + measured), `max_tokens=4`, temperature 0,
streaming. Two prompts: the fixed "short" one (7 tokens) and `--words 2600` (18720 chars → **3107 prompt
tokens**, per the harness' `DONE` line). Sampling at 50 Hz of
`/sys/class/drm/card2/device/mem_info_vram_used` (whole device, compositor included).

Two measurement caveats, both checked:

- **The `proc_vram` column in the harness' CSVs is always 0.** It parses `drm-total-vram0` in
  `/proc/<pid>/fdinfo/*`; this amdgpu/radv exports `drm-memory-vram`, `drm-resident-vram`, `drm-memory-gtt`
  (KiB) instead. I left the harness untouched for protocol comparability and added `vram-proc-sampler.py`,
  which samples the same tree at 25 Hz with the right field names (`<tag>.proc.csv`). That field is used here
  **only to time allocations** — it reads 1.9 GiB high for part of a run (§7) — while every level in this
  document is `mem_info_vram_used` minus the 2.509 GiB non-engine floor.
- **Phase marks exist only for the arms run in this session.** The previous attempt's stdout (`<tag>.out`) was
  kept only for a3. For a1/a2/b1 the turn boundaries are reconstructed from the harness' fixed structure
  (`req` at t≈2.0 s, sleep 1.0 s between turns) and the serve log's `prefill … wall N s` line; the decode
  window's length is bounded from the a3 arm's `ttft_s`/`wall_s`. Phase numbers for a1/a2/b1 are therefore
  ±10 s, which matters only for the decode windows (they are labelled as such).

Arms (`vram-<arm>-*.csv`, `.log`, `.out`):

| arm | ctx | prompt | prompt tokens `E` | `COLI_VK_CHAIN_ROWS` | chunk rows used | turns | prefill wall/turn |
|---|---|---|---|---|---|---|---|
| a1-16k-short | 16384 | short | 11 | — (auto) | 1536 cap, rows=11 | 2 | 13.92 s / 13.21 s |
| a2-16k-mid | 16384 | 2600 w | 3107 | — (auto) | 1280, 1280, 547 | 2 | 302.01 s / 289.43 s |
| b1-64k-mid | 65536 | 2600 w | 3107 | — (auto) | 1280, 1280, 547 | 2 | 314.79 s / 313.96 s |
| a3-16k-ch256 | 16384 | 2600 w | 3107 | 256 | 256 ×13 | 2 | 704.38 s / 714.07 s |
| b2-64k-ch256 (partial) | 65536 | 2600 w | 3107 | 256 | 256 x13 | 1 | 800.11 s |

---

## 2. Peaks and phases

Device VRAM (`mem_info_vram_used`), GiB. "post-load" = the flat state before the first request (trunk +
warm tier + compositor); "first forward" = the state after the first allocation staircase; "turn-N
prefill" = the peak inside turn N's prefill window (`req + wall`); "transition" = the peak between the end of
turn 1's prefill and the end of turn 2's first chunk, which includes turn 1's decode.

| arm | post-load | first forward | turn-1 prefill | turn-1 → turn-2 transition | turn-2 prefill | **peak** | peak at | run length |
|---|---|---|---|---|---|---|---|---|
| a1-16k-short | 12.145 | 13.380 (14.8 s) | 14.356 (18.0 s) | 14.356 | 14.356 | **14.356** | 18.0 s | 41.8 s |
| a2-16k-mid | 12.145 | 14.965 (3.3 s) | 15.532 (304 s) | 15.907 (316 s) | 15.907 | **15.907** | 316.1 s | 617.2 s |
| b1-64k-mid | 12.145 | 14.969 (3.4 s) | 15.532 (317 s) | 15.907 (329 s) | 15.907 | **15.907** | 329.3 s | 655.7 s |
| a3-16k-ch256 | 12.145 | 14.038 (2.6 s) | 14.415 (613 s) | 15.918 (727 s) | 15.918 | **15.918** | 727.5 s | 1 465.2 s |
| b2-64k-ch256 (partial) | 12.145 | n/a (sampler started at 2.5 min: 14.102) | 14.415 (606 s) | 15.918 (732 s) | — (stopped after turn 1) | **15.918** | 732.5 s | 750 s shown |

Reading of the table:

1. **ctx 16384 → 65536 changes nothing** — twice, at two chunk sizes. At chunk 1280: a2 vs b1, same 12.145
   post-load, same 14.965 first forward, same 15.907 peak, identical steps. At chunk 256: b2 vs a3, the same
   sampled trace to the MiB (14.102 → 14.415 → 15.648 → 15.690 → 15.918 GiB at the same five points), the same
   739.9 MiB chain footprint, the same 1222 frames, the same 65.4% device share of routed experts and the same
   74.7 GiB of disk reads (1 160 845 vs 1 165 752 MB). The one difference is time: b2's turn-1 prefill wall is
   800.11 s against a3's 704.38 s (+13.6%), all of it in the engine's matmul counter (293.72 vs 205.73 s, the
   disk counter is +1.5%), so the configured context buys no memory but may cost compute. The reason memory is
   indifferent is in §4: every "context-sized" buffer is sized by `E` (the forward's end position), and `E` is
   the same 3107 in both arms. The 5 GiB "window" and the 32M/E cap are not allocations (see §4.8 and §4.3).
2. **Short prompt vs mid prompt** differs by the chunk buffers only: a1's chain reports **75.4 MiB on the
   device** vs a2's **1765.7 MiB**, and the measured peaks differ by 1.551 GiB — i.e. the ~1.7 GiB of chain
   buffer that a 1280-row chunk needs, minus the 0.08 GiB a1's 11-row chunk also took, plus rounding.
3. **The peak is not reached during prefill.** In a2/b1 the highest sample is 316–329 s, i.e. within a second of
   turn 2's request (turn 1's prefill ends at 304/317 s), and turn 2's remaining 289–314 s of prefill add
   **0.000 GiB**. A single-turn benchmark would understate the peak by 0.375 GiB. This is the most transferable
   measurement in the table: this engine's VRAM high-water mark needs a *second* forward to appear.
4. **The device is nearly full at peak**: 15.907–15.918 GiB of 15.984 GiB (17.16e9 B), i.e. 68 MiB of headroom
   at the chunk-256 peak and 79 MiB at the chunk-1280 one, with the engine's own `memory_budget` reserve at
   1.00 GiB already spent. Any lever that adds VRAM must first remove
   the same amount from another pool.
5. **Chunk 256 does not lower the peak — it lowers the prefill plateau.** a3's engine sits at **14.415 GiB sys**
   through the last 100 s of its 714 s prefill (engine 11.91), 0.87 GiB *below* a2's prefill level (15.28, engine
   12.77), tracking the 1.00 GiB by which its chain footprint is smaller (739.9 vs 1765.7 MiB). Its
   first-forward staircase is correspondingly smaller (+1.893 vs +2.820 GiB). Then, at the last chunk and the
   decode, it allocates **+1.46 GiB** (a2: +0.63 GiB) — and lands at **15.918 GiB, 11 MiB *above* a2's 15.907**.
   Total growth over the run is conserved to 11 MiB: **3.762 GiB (a2) vs 3.773 GiB (a3)**. The chunk decides
   *when* the growth happens, not how much; §3d and §5 R1 turn on this.
6. **The peak arrives in the decode, not in prefill.** a3's marks: prefill's TTFT at 715.76 s, last prefill chunk
   at 705.9 s (+1.233 GiB), decode at 727.5 s (+0.227 GiB) → the peak sample at 727.5 s, which then holds for
   all 714 s of turn 2. Same shape in a1 (peak in turn 1's decode at 18.0 s).

---

## 3. Where the growth actually goes

The Track A document's caveat reads "device usage moving 12.15 → 14.36 GiB, i.e. ~2.2 GiB of growth per turn at
16k … bundles the KV split's uploads, the ring, and the per-chunk scratch". That attribution is wrong in three
ways, and this section replaces it.

**(a) 12.145 GiB is not the pre-load state, it is the post-load state.** It is identical to the byte in the
four arms that recorded it from t=0 (13 040 701 440 B; the fifth, b2, reports the same base), it equals compositor (≈2.5 GiB, measured after the engine exits: 2.509 GiB in
a3's tail) + trunk (`device memory held 8187281408 B` = 7.625 GiB) + the warm tier (`resident 91 … 1.87 GiB`).
The engine pages in before it answers `/v1/models`, so the harness' first sample is already post-load.

**(b) The 2.21 GiB on the short arm is not per turn, and mostly not the chain.** a1 grows in five steps:
1.235 GiB at 14.8 s (the end of turn 1's prefill), 0.250 GiB at 15.6 s, then 0.226 + 0.250 + 0.250 GiB across
turn 1's decode (17.5–18.0 s), and then **nothing for the rest of the run**. Its own chain accounting for the
same forward is **75.4 MiB** ("chain: 3 forwards … 75.4 MiB on the device", `vram-a1-16k-short.log`).
2.21 − 0.07 = 2.14 GiB is allocated by something the chain does not count — the tier's own bookkeeping (its
3.46 GiB budget, the 64-slot streaming pool of 21.1 MiB slots, the extra-layers pool of 232.0 MiB) and the
64 MiB-granular pool blocks. The sample stream shows the granularity directly: the last three growth steps are
+0.250 GiB each, and 0.250 GiB = 4 × `VKC_BLOCK` (`c/vk_chain.c:215`, 64 MiB), i.e. these are pool block
allocations, not buffer-sized ones. I cannot attribute the 2.14 GiB further from sysfs alone; that is a
finding in its own right (see §6, lever 3).

**(c) What the first forward allocates, term by term.** a2, turn 1:

| term | size | source |
|---|---|---|
| chain buffers, 1280 rows at `E`=3107 | +1.724 GiB | `chain: 3 forwards … 1765.7 MiB on the device`, a2 log |
| KV mirrors, `E`=3107 | +0.026 GiB | §4.4 |
| **measured first-forward growth** | **+2.820 GiB** | 12.145 → 14.965 (3.3 s) |
| **non-chain part of it** | **≈1.07 GiB** | by subtraction |
| candidates for that 1.07 GiB (upper bounds, code) | tier growth ≤ 1.32 GiB over the turn; 64 streaming slots ≤ 1.35 GiB; extra-layers pool 0.23 GiB; pool blocks in 64 MiB quanta | §4.2, §4.8 |

So the chain buffers are 61% of the first forward's growth and the tier's own bookkeeping is the rest — but sysfs
cannot split the tier term into slots / promotions / blocks, and the bounds above do not all bind at once. That
is the residue §6.3 is about. The remaining 0.94 GiB from the first forward to the peak is observed as:

| increment (a2) | measured | explained by |
|---|---|---|
| turn-1 prefill, 39.5 → 230 s | +0.315 GiB | tier promotions inside its 3.46 GiB budget (0.06–0.13 GiB steps) |
| end of turn 1 → turn 2's first chunk (304 → 316 s) | +0.627 GiB | the tier finishes promoting to 3.19 GiB + pool blocks; the engine's own 6-forward print shows the chain footprint growing 1765.7 → **2133.7 MiB** (+0.359 GiB), and the extra layers go `0 of 45` → `21 of 90` routed experts on the device |
| rest of turn 2's prefill (289 s) | +0.000 GiB | nothing left to allocate |

**(d) The chunk-256 arm (a3) is the clean experiment, and it splits the answer in two.** Everything below is
the sampled device trace, so no instrument ambiguity:

|  | a2 (chunk 1280) | a3 (chunk 256) |
|---|---|---|
| post-load | 12.145 | 12.145 |
| first-forward staircase | **+2.820** (by 3.3 s) | **+1.893** (by 2.6 s) |
| prefill plateau | 15.03 → 15.28 (4 × ≈0.06 steps) | 14.10 → 14.415 (6 × ≈0.06 steps) |
| last prefill chunk | +0.252 | **+1.233** |
| decode | +0.022, +0.227, +0.126 | +0.042, +0.227 |
| **peak** | **15.907** | **15.918** |
| peak time / run length | 316 s of 617 s | 727 s of 1 465 s |
| **total growth over the run** | **3.762** | **3.773** |
| chain buffers (`N MiB on the device`) | 1765.7 MiB | 739.9 MiB |

b2 (ctx 65536, chunk 256) reproduced a3's column exactly — 14.102 → 14.415 → 15.648 → 15.690 → 15.918 GiB at
the same five sampled points — so the chunk result is not a ctx-16K artifact.

So the chunk 1280 → 256 delta is **−0.93 GiB at the first forward** (against a measured −1.00 GiB of chain
footprint) and **−0.87 GiB at the prefill plateau**, but **+0.011 GiB at the peak**: the freed GiB comes back
as `+1.233 GiB` in the last prefill chunk and `+0.23 GiB` in the decode, the phase where the tier promotes
itself from its warm 1.87 GiB to its 3.19 GiB budget (`resident 90 … 3.19 GiB of 3.46 GiB, 13 blocks, frag
0.63`) and the pools/staging grow. The two runs' *total* growth is conserved to 11 MiB — the chunk decides
when the allocation happens, not how much of the card the engine ends up holding.

That is the single most consequential measurement in this audit for ranking the levers: on this engine the
chunk size does **not** buy peak headroom. Either the tier/pool allocation is opportunistic (it grows into
whatever the chunk left free), or the small-chunk path needs more staging because it runs 2.3× as many steps
(407 frames per forward vs 176, §4.7). Distinguishing those two needs the ledger of §6.3.

Nothing in any arm armed the KV split: `v41c_plan` splits only when the whole mirror would not fit, no arm
printed a split decision, and the mirrors' capacity stops at 4096 rows (§4.4). The ring's share is 115 MB at
1280 rows and 31 MB at 256. So the comparison's "bundles the KV split's uploads, the ring, and the per-chunk
scratch" describes three items of which the first does not exist here and the second is 3% of the total.

---

## 4. Allocation map on the device during prefill

Everything below is what `v41c_scratch`/`v41c_forward_seg`/`vk_chain.c` reserve for a forward, with the size at
`E` = 3107, chunk = 1280 unless noted. Measured = read from the sampled CSVs or printed by the engine.
Derived = arithmetic on the cited reservation.

### 4.1 Trunk — 7.625 GiB (measured, fixed)

`40 of 40 layers placed: 7888437644 B of matrices on the device … device memory held 8187281408 B`
(`vram-a2-16k-mid.log`): 7.346 GiB of dense matrices + 56.2 MiB of chain buffers (`chain buffers 58929856 B`).
Identical in all arms. This is 48% of the card and the largest single item; it is also the only item the
comparison treats correctly.

### 4.2 Expert tier — 1.87 GiB warm, 3.19 GiB at peak, plus up to 1.35 GiB of streaming slots (measured + code)

`budget 3.46 GiB = 91 experts of 21.1 MiB … resident 91 (budget 91, 3.19 GiB of 3.46 GiB, 13 blocks, frag
0.15)` (a2; a3 ends at `resident 90 … 3.19 GiB … 13 blocks, frag 0.63`). Allocated lazily: 1.87 GiB before the
first request, 3.19 GiB by the end of turn 1. Plus the **64 streaming slots**, allocated in one go at the first
step that streams (`c/vk_tier.c:1048-1078`: `st_slots` × (gu+dn+scales), 64 × 21.1 MiB = **≤ 1.35 GiB**; no arm
printed the `n of 64` warning, so all 64 were made), and the extra layers' experts (`(3, fmt 7): 11 of 21.1
MiB in a pool of 232.0 MiB`; turn 1: 0 of 45 routed on the device, turn 2: 21 of 90). Neither the slots nor the
extra pool is counted by the chain's `dev_bytes`, and neither is printed at turn end — they are the tier-side
candidates for the ~1.07 GiB non-chain part of the first forward (§3c), and the reason a1's short-prompt arm
reaches 2.21 GiB with only 75 MiB of chain buffers.

### 4.3 Indexer score + candidate mask — `2 × rows × E × 4 B` (derived, code)

`v41c_scratch` sets `ch->wcap = ctx` where `ctx` is the *forward's end position* — `v41c_forward_seg` passes
`E = start + n` (`c/deepseek_v41_chain.h:786`, `:1082` call site `v41c_scratch(ch, m, rows, E)`), and reserves
`isc` and `mask` at `rows × wcap` each (`:794`). At 1280 rows, `E`=3107: **15.9 MB + 15.9 MB = 31.8 MB**. This
is the term the printed row cost exposes: the log prints **1293 KiB/row at E=11** (a1) and **1317 KiB/row at
E=3107** (a2/b1); the 24 KiB difference is `8·3107 B/row` exactly. The `32 MiB / E` row cap (`:991-992`)
therefore bites at **E ≈ 32 MiB/row_cost ≈ 26 000 prompt tokens**, not at 26 000 *configured* context — which is
why a2 (16K) and b1 (64K) are indistinguishable. At `E` = 131072 the pair is 268 MB at 256 rows and 1.54 GB at
1470 rows; tiling it (R1) caps it at ~64 MiB, which is the true size of R1's VRAM win.

### 4.4 KV mirrors (4 kv-source layers) — 26 MB at E=3107 (derived, code)

`v41c_cache` doubles capacity from 64 until it covers `want = E/r` (`:731-748`); `ckv` is `cap × head_dim ×
4 B` = 8.4 MB for layer 20 (ratio 1, cap 4096) and 4.2 MB each for layers 2/8/14 (ratio 2, cap 2048) = 21 MB;
`ikey` is `cap × index_head_dim × 4 B` = 2.1 MB + 3 × 1.05 MB = 5.2 MB. **The configured context does not enter
this**: `max_positions/r` is only the ceiling. So the comparison's "105 MB at 16k / 419 MB at 64k" table
describes mirrors for a 16K/64K-*token conversation*, not for the 16K/64K `--ctx` flag; at a 3107-token prompt
R2's fp8 saving is ~20 MB.

### 4.5 Window ring — `40 × (128 + rows) × 512 × 4 B` (derived, code)

`v41c_grow_win` (`:769-782`): `Wd = W + rows`, one buffer per chain layer, freed and re-mirrored when `Wd`
grows (`win_valid = 0`). 115 MB at 1280 rows, 31 MB at 256, 188 MB at 2414, 681 MB at 8192 (the last two match
R1/R6's derived numbers). This is the only term that grows with the chunk *and* is re-uploaded on growth.

### 4.6 Per-row activation scratch — 698 KiB DEV + 200 KiB DOWN (derived from `:783-800`)

`v41c_scratch` reserves 30 buffers per row; summing the ones whose shapes are in `config.json` (D=5120, H=4 →
HD=20480, h=24, hd=512, nh=64, q_lora=1280, o_groups×o_lora=8192, IH·ID=4096, moe_inter=2304, T=5) gives 698 KiB
of `VKC_DEV` buffers (`xs`, `xn`, `q`, `heads`, `ekv`, `gs/us/hs`, `ds`, `grp`, …) and 200 KiB of `VKC_DOWN`
(`h2d`, `mh`, `xd`, `hpd`). The full printed row cost is 1293 KiB, so the remaining ≈395 KiB/row is: ring 80 KiB,
expert-path staging 260 KiB (`2·n_activated+1` × dim, `:752-768`), CPU-path gather 58 KiB. The last two are
`VKC_UP`: a host-visible allocation, but still device-local on this card (§4.8), so the printed 1293 KiB is the
VRAM cost and **978 KiB of it is scratch + ring**. Cross-check: the measured prefill-plateau slope between chunk
1280 and 256 is 0.6–0.9 MiB/row, i.e. inside the 978–1293 KiB/row band; I use **1.0 MiB/row** as the planning
figure with ±25%. **This is the term that scales with the chunk**: 1280 → 256 rows frees 1024 × 978 KiB =
**1.00 GiB** of scratch, and the net measured effect is 0.6–0.9 GiB because the fixed per-forward term grows at
small chunks (§4.7).

### 4.7 Non-scratch per-forward buffers — 120 MiB at 1280 rows, 410 MiB at 256 rows (measured, unexplained)

From the same walk: `pull` (`pn` rows × W windows + per-layer compressed rows, `:1075-1077`), `cmd` / `ld`
only for a partial chain, and the KV-split `cnew`/`cdn` (off here). Reconciles a2: 1280 × 1317 KiB = 1.647 GiB
of scratch + 0.118 GiB = **1.765 GiB**, exactly the engine's printed `1765.7 MiB`. The same subtraction on a3
gives 739.9 MiB − 256 × 1317 KiB (0.322 GiB) = **410 MiB**, i.e. the fixed per-forward term is *not* constant:
it triples at the smaller chunk. The mechanisms that could do that are the frame ring's 16 live per-frame
staging buffers (`f->stage`, `c/vk_chain.c:1248-1267`, never freed until the frame is reclaimed) and the fact
that chunk 256 needs 407 frames per forward against 176 at chunk 1280 (the engine prints the frame counts:
1222 frames for 3 forwards vs 528). I cannot separate them from sysfs; the *net* is what the table below uses.

### 4.8 Pools, staging and the "5 GiB window"

- `vkc_fit_pools` (`c/vk_chain.c:3426-3430`) reserves `256 MiB + 3 × VKC_BLOCK (64 MiB) + 16 frames × 1 MiB`
  = **464 MiB**; the fit line prints it (`the pools' 486539264 B` = 464 MiB) plus `the engine's 471598080 B`
  = 450 MiB. Allocations come out of a 64-MiB-block pool per kind (`VKC_BLOCK`, `:215`), which is why the
  sample stream quantizes to 256 MiB. The fit's reserve is a *floor*: `pool_grow` (`:216-236`) is unbounded.
- Frame staging: `VKC_STAGE0 = 16 MiB / 16 frames = 1 MiB` per frame (`:55-56`), doubling on demand
  (`:1248-1267`), and **it lands in device memory here**: `o->memtype_host = g_up[0].on ? g_up[0].mt_stage :
  G.memtype` (`c/backend_vulkan.c:6429`) and `g_up[0].on` is 0 because the mapped-weights path was taken
  (`gave the 5 GiB asked`). So `VKC_UP` staging is VRAM on this card, not GTT — which is why §3's residual
  closes against a pool that grows in 64 MiB blocks.
- The `5 GiB` in the logs is a **probe**: `vk_window_decide` allocates 1 GiB blocks until the driver refuses,
  frees them all, prints `gave the 5 GiB asked`, and then the tier allocates its own memory inside that heap
  (`c/backend_vulkan.c:748-786`). There is no 5 GiB resident "mapped window" to reclaim; the resident tier is
  §4.2. The comparison's §1 row ("trunk 7.62 GiB, a 5 GiB mapped window, a 3.31 GiB expert tier") double-counts
  by ~5 GiB if read as an allocation map.

### 4.9 The row-cost table the chunk actually pays (derived)

| chunk rows | chain scratch (dev+down+ring) | `isc`+`mask` at `E`=3107 | `isc`+`mask` at `E`=131072 | engine's "N MiB on the device" |
|---|---|---|---|---|
| 256 | 0.239 GiB | 6 MB | 268 MB | 407 MiB |
| 512 | 0.479 GiB | 12 MB | 537 MB | 737 MiB |
| 1024 | 0.957 GiB | 24 MB | — | 1395 MiB |
| 1280 | 1.196 GiB | 31 MB | — | 1765.7 MiB (measured) |
| 1470 | 1.374 GiB | 35 MB | 1.54 GB | 1969 MiB |
| 2414 | 2.256 GiB | 58 MB | — | 3184 MiB |
| 8192 | 7.66 GiB | 198 MB | — | 10617 MiB (does not fit) |

("engine's N MiB" = `rows × 1317 KiB` of scratch plus the per-forward fixed term of §4.7, ≈120 MiB at 1280 rows;
that matches the measured 1765.7 MiB, and overstates the 256-row arm (739.9 MiB measured against 407 MiB
predicted) because the fixed term is itself chunk-dependent.)

---

## 5. Verdicts on R1–R6

### R1 — "decouple the prefill chunk from the indexer's quadratic scratch; VRAM: the same buffers, re-tiled — no new memory" → **refuted as stated, direction confirmed, magnitude wrong in both directions**

- **The cap is real but triggered by prompt length, not context.** `lim = 32 MiB / E` (`:991-992`) with
  `wcap = E` (§4.3). At ctx 65536 with a 3107-token prompt the cap is 10,788 rows — 8.4× away from binding;
  measured proof: b1 (64K) is byte-identical to a2 (16K) and its chunk stayed 1280/1280/547. The cap only
  binds once the *prompt* passes ~26K tokens. The comparison's table ("cap 512 at 64k, 256 at 128k") is
  numerically right only if "64k/128k" means prompt tokens.
- **What tiling frees**: at most **256 MiB** — `rows × E ≤ 32 M` *floats* per buffer, i.e. `isc` and `mask` can
each reach 128 MiB (`:991-992`), and a tiled version holds a tile (~64 MiB) instead. Not "the chunk-sized isc
reservation", and not 5-10 GiB.
- **What a bigger chunk costs**: 978 KiB/row of VRAM-resident scratch (§4.6) + 80 KiB/row of ring. Raising the
  chunk from 256 to 1470 rows at a 128K-token prompt needs `1214 × 1058 KiB` = **+1.22 GiB**; the `isc`/`mask`
  pair it replaces there is 268 MB at 256 rows → ~64 MiB tiled, i.e. **0.2 GiB freed**. Net **+1.0 GiB** that
  this card does not have (68 MiB of headroom at peak, §2.4).
  R1 is thus a **VRAM-for-throughput trade**: ~2.9× fewer expert passes at 64K-*token* prompts for +1.0 GiB.
  It is only affordable after R2 or bf16 activations free ≥1 GiB — and the comparison already lists bf16 acts
  as "a second, independent 2×", which now becomes a *precondition*, not an option.
- **Throughput side, measured in both directions.** chunk 1280 → 256 (what the `32M/E` cap does to a long
  prompt) costs **2.33× prefill wall** on the same prompt: 97.2 ms/row at 1280 rows (a2: 302.01 s / 3107 rows)
  vs 226.7 ms/row at 256 rows (a3: 704.38 s / 3107 rows). The CPU share of the tier rises **4.34×** (40.8 →
  176.8 ms/row) and expert-rows kept on the CPU rise **4.10×** (57 925 → 237 528), while rows actually streamed
  to the device fall 44% (674 878 → 376 458). So R1's speed claim is corroborated with a measured factor (2.3×
  for 5× the rows, in the inverse direction), and its "no new memory" claim is not.
- **But the peak does not move** (§3d): 13.40 GiB engine at chunk 1280, 13.41 GiB at chunk 256. Raising the
  chunk through R1 would raise the *prefill* level by ~1 GiB with no measured effect on the turn-boundary
  peak — which is the number that decides whether a 1M-token ceiling survives. That makes R1 cheaper than
  "+1.2 GiB at peak" and still not free: it is +1 GiB where every other phase already has 1.5 GiB of slack.

### R2 — "store the device KV mirror in the checkpoint's quantized format" → **quantified, but two orders of magnitude smaller than the table implies for realistic prompts**

Mirrors are `E`-sized (§4.4): **26 MB at E=3107** (fp32) → ~6.5 MB at fp8. The table's "105 MB at 16k, 419 MB at
64k" presumes a conversation that reaches 16K/64K tokens. Correct form: the saving is `≈ 4.7 KiB per token of
conversation`, i.e. 0.02 GiB at 3107 tokens, 0.31 GiB at 64K tokens, 0.62 GiB at 128K tokens. It only pays for
R1's ring growth at the top of the range, exactly as the comparison says for the *bottom* — the timing is
right, the sizes need the `E` correction. Effort/risk verdict unchanged (L, breaks bit-exactness, needs an
explicit approximate mode).

### R3 — fuse the KV row transform (norm → rope → ring store → compress) → **supported on dispatch count, zero VRAM; cannot be confirmed or refuted for VRAM by this measurement**

Measured: prefill wall is dominated by expert bytes (a2: `prefill 412655.2 MB, disk 151.30s, matmul 49.07s, wall
302.01s` — 50% disk, 16% matmul, the rest host expert time); GPU idle-wait is `66% of device time hidden` and
`65775.1 ms waiting for the device` per turn. The ~160–360 dispatches R3 removes are not visible in any VRAM
number (device VRAM is flat during prefill, §3c) and are second-order against a 151 s disk bill. No new
evidence either way; keep as hygiene, do it after R1.

### R4 — remove the per-layer full-device drain → **not observable in VRAM; latency claim untested here**

`vkc_finish` drains all 16 frames (`c/vk_chain.c:802-807`) once per layer; the logs show the consequence
(`waited 3281.5 ms`, `56% of device time hidden` on b1 vs `90%` on the short arm). VRAM: no term anywhere in §4
changes. Agree it is nearly free and second-order on a disk-bound prefill.

### R5 — one dispatch for the indexer score → **not exercised by these arms**

`vkc_dsv4_score` is recorded once per row when the chunk's key owners differ (`:906-916`), but the counters do
not separate those dispatches (a2 reports 1945 matmuls / 1170 tiled GEMM per turn, a3 5477 / 4608), so I can
neither confirm nor refute the mixed-owner worst case from these runs. The only VRAM statement that holds is
that the per-row path allocates nothing (§4.6): zero either way.

### R6 — plan-time, fixed buffer set (no runtime reallocation, no re-mirror) → **refuted as a memory saving; real as a peak-shape fix**

- The ring re-mirror does happen, but it costs 115 MB at 1280 rows (measured against §4.5's formula) and it is
  charged *inside* the chunk-row budget, not on top of it.
- The measured peak is **not** set by re-mirroring but by the *turn boundary* — the last chunk plus the decode,
  +0.63 GiB in a2 and +1.46 GiB in a3 (§3d) — and by *tier/pool block growth* in 64 MiB quanta, which a
  plan-time buffer set would not remove.
- What a plan-time set *would* fix is the 376 MiB of "per-layer `malloc/free` + `pull`/`cmd` reallocation"
  visible as the chain footprint going 1765.7 → 2133.7 MiB between turns: the second turn's buffers are
  allocator-noise-invisible but real. Still small. Keep R6 as hygiene; do not sell it as VRAM.

---

## 6. Levers the comparison missed

1. **The tier's `16 rows or more` streaming threshold, which its own log admits is unmeasured.**
   `stream: a step of 256 rows; cold experts with 16 rows or more go to the device (the GEMM's 16 rows: the
   upload and the CPU's rows not measured yet)` (`vram-a3-16k-ch256.log`, every step). At chunk 256, layer 0
   streams **18–20** experts and keeps **100–134 on the CPU (593–702 rows)** per step; at chunk 1280 it streams
   **128** and keeps **0** (except the last 547-row chunk: 65 kept, 484 rows). Aggregate: 7.8% of expert-rows
   fall to the CPU at 1280 rows (`10883 cold experts (57925 rows) kept on the CPU`) versus **34.5%** at 256
   (`49725 cold experts (237528 rows)` of 688 317 routed expert-rows), and the device share of routed experts
   drops from 92.1% to 65.4%. The engine prints both sides of the threshold (`an upload of 21.1 MiB at 10.0
   GB/s takes 2.212 ms, the CPU 2.5160 ms a row`), so the crossover is ≈1 row *if* a padded 16-row device GEMM
   costs under ~0.3 ms — the engine itself says that term was never measured (`the GEMM's 16 rows: the upload
   and the CPU's rows not measured yet`), so this lever needs one micro-benchmark before it is a change. Lowering the threshold to the GEMM's real minimum (4
   rows) would reclaim most of the 2.33× penalty measured at chunk 256 — the single largest throughput lever
   this audit found, and it is most of R1's benefit without R1's VRAM cost. The churn confirms the mechanism:
   per prompt, device uploads rise 29% (4.89 → 6.30 GiB), evictions 44% (195 → 281) and the CPU's row share
   4.34×, because each 256-row step re-decides placement for the whole layer from a thinner routing signal.
2. **The 64-slot streaming pool is a reservation of up to 1.35 GiB made at the first streaming step**
   (`c/vk_tier.c:1048-1078`, `st_slots` × 21.1 MiB, all 64 made in every arm — no `n of 64` warning) and the
   extra-layers pool is another 232 MiB. Neither appears in the engine's chain accounting (`dev_bytes`) nor in
   the docs' allocation map, and both are allocated **after** the chunk is sized — so a chunk budget computed
   from `free` memory (`half of 3.84 GiB free`) can be spent by the tier afterwards. They are also the reason a
   *bigger* chunk does not raise the peak by its full 1.7 GiB: at 1280 rows the plateau is 12.52–12.77 GiB
   engine, which is post-load 9.64 + slots 1.35 + chain 1.72 — the budget is already committed elsewhere. Sizing
   the slots from the measured concurrent streams (`prefetched 7096 (5303 used)` per turn ⇒ ~44 slots ever
   useful) would free ~0.4 GiB of peak headroom, more than the 68 MiB that exists today.
3. **Nothing prints the VRAM ledger.** The engine prints matrices, chain buffers, tier residents — but not the
   pool blocks, the streaming slots, the extra-layers pool or the total. That is why a1's 2.14 GiB cannot be
   attributed further (§3b) and why the peak is only visible as a 64-MiB-quantized sysfs staircase. One line at
   turn end (`dev_bytes`, pools, tier bytes, mirrors, `mem_info_vram_used`) would turn every future VRAM
   question into a lookup, and is cheaper than any of R1–R6.
4. **The peak needs the end of a turn to appear** (§2.3, §3d): in a2/b1 the last +0.63 GiB arrives at the
   turn-1 → turn-2 boundary, in a3 +1.46 GiB lands in the last prefill chunk and the decode. Any budget or
   regression test that measures a single prefill forward — or that stops at the last token of it — understates
   the peak by 0.4–1.5 GiB depending on the chunk, and can pass while the real turn OOMs.
5. **`isc`/`mask` are not the long-context constraint below ~26K prompt tokens; the 978 KiB/row scratch is.**
   The comparison's framing ("above ~26k context a second, hard-coded cap takes over") is right about where the
   cap starts, but at that point the *scratch* (1.4 GiB at 1470 rows) is already the binding VRAM term, so the
   cap and the scratch fight over the same budget. A cheaper move than R1, and bit-exact: **`mask` is read only
   as a boolean** (`deepseek_v41_chain.h:1225`: `mk[...] != 0`) yet stored as fp32 at `rows × E` (`:794`).
   Packing it to one bit cuts the capped pair from 256 MiB to 132 MiB and buys 2× on the cap without touching a
   single score. bf16 on `isc` would be a further 2× but is not bit-exact for the selection step.

---

## 7. Limitations / open items

- **a1's 2.14 GiB is unattributed** (§3b). sysfs cannot separate the tier's pools from the chain's buffers; the
  engine's own counters would (§6.3). Until then, treat any "VRAM freed by X" claim below ~2 GiB as unproven.
- **The compositor floor is assumed constant at 2.509 GiB.** Every absolute number here is
  `mem_info_vram_used − 2.509`, measured twice: in the killed a3 run's tail and (live) as `sys − Σ fdinfo`, and
  the post-load ledger only closes (9.64 = trunk 7.625 + tier 1.87 + 0.14) with that floor. The per-client
  `drm-memory-vram` field is **not** usable for levels: it reads 1.9 GiB above the sysfs-derived engine value
  for part of a run (11.53 vs 9.64 at post-load, exact agreement at the peak), most likely because the trunk's
  imported BOs are counted per import. It is used here only to *time* allocations (`vram-*.proc.csv`), never for
  absolute sizes.
- **Phase windows for a1/a2/b1 are reconstructed** (§1), ±10 s; a3 has harness marks and a 25 Hz process trace.
- **b2 was stopped after turn 1** (at ~750 s of its own clock, in turn 2's prefill) once its turn-1 peak matched
  a3's to the MiB: a second turn could only reproduce a constant. Its evidence is
  `vram-b2-64k-ch256.{sys.csv,proc.csv,log}` (no harness CSV — the harness writes that at the end of turn 2),
  sampled by `vram-sys-sampler.py` at 2 Hz and `vram-proc-sampler.py` at 25 Hz.
- The `VKC_DOWN`/`VKC_UP` residency claim (§4.6, §4.8) rests on `backend_vulkan.c:6429` plus the fact that the
  totals close; `memtype_cached` is a `pick_memtype(COLI_VK_MEM_CPU_READS)` pick (`:1620`) whose heap I did not
  enumerate.
- The 3107-token prompt is 2.4% of the configured 1M ceiling. Nothing in this audit measures a prompt of
  ≥26K tokens, where the `32M/E` cap actually binds; R1's long-prompt factors (2.9×/5.7×) remain derived from
  code plus the measured 1280↔256 slope (2.33×).
- The 2133.7 MiB chain footprint on the second turn (+368 MiB over the first) is measured but not decomposed
  (§3c); it is the last unexplained >100 MiB item in the ledger.

---

## 8. Reproduction

```
cd /root/Projects/colibri
# arms on disk: a1-16k-short, a2-16k-mid, b1-64k-mid, a3-16k-ch256 (.csv/.log/.out)
python3 prefill-vram.py --ctx 16384 --words 2600 --tag vram-a3-16k-ch256 --port 8023 \
        --env COLI_VK_CHAIN_ROWS=256                  # 2 turns x 714 s = 24 min at chunk 256
python3 vram-proc-sampler.py vram-a3-16k-ch256 1800 $(pgrep -x deepseek_v41)   # 25 Hz engine trace
python3 vram-analyze.py vram-a1-16k-short vram-a2-16k-mid vram-b1-64k-mid
```
The context 65536 / chunk 256 arm (`vram-b2-64k-ch256`) is chained behind a3 by `vram-run-b2.sh` and costs the
same 24 min; the 2.33x prefill penalty at chunk 256 is why the two chunk-256 arms take ~2.5x a default-chunk arm.
Isolation: kill leftovers with `pkill -f '[c]oli serve'; pkill -x deepseek_v41`.
