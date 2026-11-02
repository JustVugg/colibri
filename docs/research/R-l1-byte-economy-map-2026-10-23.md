# L1 — the expert byte economy: implementer's map

Trees: **fork** = `/root/Projects/colibri-ds4-vulkan` @ `d11fba5b`; **port** = `/root/bench-worktrees/port-l3`
(`port/l3-hotset` @ `0c6ddae2`). All line cites are from those two checkouts.

Headline: **two premises in the task are wrong, and they matter more than any code detail.**
(i) The measured arm's chunk loop is *outer* and the layer loop *inner* (`deepseek_v4_chain.h:1224` vs
`:1296`), so "the same layer's next chunk" is **43 layer-steps later**, not step N+1 — a per-layer
retention window (dossier L1 shape) has a reuse distance of ~2.0–3.5 GiB and cannot pay on the chain path.
(ii) **The fork's 68 GiB and the port's 192 GiB are different counters.** In the *same* fork arm, the
store view says 68.4 GB (`expert_requests=16254 hits=11141 misses=5113`) while `v4_direct
payload_bytes=202,849,124,352` (189 GiB) and the host-RAM ring reports `prefetched=11008` experts
(`bench/up2276/fork-noexperts-r1.log:3333,3338,3345`). The port's comparable payload is 250.95 GB
(`l3-a0-hot0.log`, `v4_direct reads=19944 payload_bytes=250953596928`). **Apples to apples the disk gap
is 1.24x (251 vs 203 GB ≈ 5 s at 8.6 GB/s), not 2.8x**, and the port is already at ~1.01x "once per
(layer,expert) per prompt". The 90 s TTFT gap is *not* in these bytes (§2, §3).

---

## Files Retrieved

1. `port/c/deepseek_v4_chain.h` (1224–1300, 1376, 169–181, 824–837) — the prompt forward: chunk loop
   **outer**, layer loop **inner**; where the tier's big step and prefetch are called.
2. `port/c/vk_tier.c` (1091–1134, 1136–1170, 1191–1296, 1375–1378, 1424–1441, 1504–1511, 1544–1573,
   1672, 1691–1740, 2643–2700, 2806–2856, 2540–2600) — streaming: decision, slot lifecycle, prefetch,
   budget carve, host-pool sizing.
3. `port/c/deepseek_v4.c` (13782–13800, 13862–13881) — `v4_vkt_load` = `coli_expert_lookup` (the *lease*
   path, i.e. cache-filling); `VktConfig` has **no** `.load_batch` and **no** `.stage`.
4. `port/c/deepseek_v4.c` (8874–8990, 9215–9230) — the store: per-layer LRU, `slots_per_layer =
   cache_bytes/(layers*record_bytes)` = 48; the disk read happens inline under `state->mutex`.
5. `port/c/deepseek_v4.c` (14850–14880) — the **CPU** path's chunk loop sits *inside* the layer loop
   (the fork's order) — so the port has both orders, and only the chain one defeats dedup.
6. `fork/c/deepseek_v4.c` (19284–19296, 19411–19460, 19495–19550) — bank validity bitmap, missing-set
   dedup, cache-bypassing staging refill.
7. `fork/c/deepseek_v4.c` (12251–12259, 23524, 23580–23587) — "the chunk loop sits INSIDE the layer
   loop, so one bank upload per layer serves every chunk of that layer"; the actual nesting.
8. `fork/c/deepseek_v4.c` (13726–13775) — `coli_v4_expert_store_read_bytes` (no LRU/lease/slot state,
   no mutex). **The port does not have this function.**
9. `bench/up2276/l3-a0-hot0.log`, `bench/up2276/base-18g-r1.log`, `bench/up2276/fork-noexperts-r{1,2}.log`,
   `bench/up2276/fork-full-r3.log` — the measured arms.
10. `docs/research/P-optionB-dossier-2026-10-22.md` §2, §2.3–2.6, §5, §8 — the contested attribution and
    the traps (read before sizing effort).

---

## 1. Where the port's 190 GiB comes from

**The measured configuration** (`bench/up2276/l3-a0-hot0.log`): RDNA2 profile sets
`COLI_VK_TIER_STREAM_SLOTS=256 COLI_VK_TIER_GEMM_ROWS=4 COLI_VK_CHAIN_GEMM_ROWS=-1`; the chain picks
prompt chunks of **2048 rows** from the budget (`deepseek_v4_chain.h:824-837` → `vkc_chunk_rows`), so
2,515 tokens = **2 chunks (2048 + 467) × 43 layers = 86 stream steps** ✓ (log: "stream: 86 steps").

**The call chain, per (chunk, layer):**
`v4c_forward_seg` chunk loop (`deepseek_v4_chain.h:1224`) → layer loop (`:1296`) → `vkt_issue_w` →
`issue_big` (`vk_tier.c:1373`).

**The decision, per cold expert** (`vk_tier.c:1375-1378`):
```c
cls[e] = !cnt[e] ? BC_NONE : v->state == VS_RESIDENT && v->ex ? (v->dev ? BC_DEV2 : BC_DEV)
       : T.st_ok && cnt[e] >= R ? BC_STREAM : BC_CPU;
```
`R` = `stream_min_rows()` (`vk_tier.c:1120-1134`): `max(gemm_rows, exp_bytes/bw_up / cpu_row_ms)`, with
`bw_up` a *measured EWMA of the upload speed* and `cpu_row_ms` an EWMA of CPU cost per (row,expert). With
GEMM_ROWS=4 and a 2048-row chunk, ~221 of ~250 cold experts clear R for layer 0 (log line 1) → almost
every cold expert streams. **Note the coupling:** any arm that changes the load path changes `bw_up` →
changes `R` → changes which experts stream. Log `R` in every arm or the arms are not comparable.

**Why a streamed expert is not there next time — the slot lifecycle is entirely step-local:**
- `st_alloc` (`:1136`) allocates `T.st_n` = 239 slots out of the budget: `slots` (256) is clamped to
  `fit_raw/2` and then `fit = fit_raw - slots` (`:2806-2820`). In a0: budget 7.04 GiB = **240 residents
  + 239 staging slots** (log). 239 × 15.0 MiB ≈ **3.5 GiB of VRAM that retains nothing.**
- fill: `st_fill_group` (`:1213-1292`), up to 64 experts per group, each loaded through
  `T.c.load` = `v4_vkt_load` = **`coli_expert_lookup`** (`deepseek_v4.c:13786-13796`) — the lease path,
  which *fills* the store's 25.7 GiB LRU slab cache (`deepseek_v4.c:8874-8990`).
- free: `vk_tier.c:1435` (a prefetched slot whose expert this step does not stream), `:1672` (**every
  BUSY slot freed at `join_big`**), `:1697` (prefetch frees all non-BUSY slots at entry), `:1304`
  (`pending` → `SS_FREE`). Retention after a step = **0**.
- the *one* reuse that exists: `vk_tier.c:1504-1511` reuses a `SS_FILLED` slot whose `layer`/`eid` match
  the current step. That only fires if the same layer is the *next* big step — true in the CPU path
  (`deepseek_v4.c:14857`, chunk-inside-layer) and **never true on the chain path**
  (`deepseek_v4_chain.h:1224/1296`), which is the path the 190 GiB arm runs. It is effectively dead code
  in the measured configuration.
- the only cross-step transfer is `vkt_stream_prefetch` (`:1691-1740`): while layer *i*'s attention runs,
  fill free slots with layer *i+1*'s predicted set (`T.pred[i]`, `:1206-1210`). Log: prefetched 12,644,
  used 10,663 → **~16% of prefetched fills are wasted**; they are counted in `st_bytes` but not in
  `st_experts`, which is why `st_bytes` reads 192.01 GiB (13,109 fills) while `st_experts` = 11,127.

**Which cache should have caught it, and why none does (quantified):**
- *rows16 packed cache* — `deepseek_v4.c:9737-9860`, `packed_slots=953` (≈14 GiB of the 25.7 GiB). It is a
  **hot-pin** cache: 16 pins/layer (`pin_slots_per_layer=16`, `repin_interval=6`, autopin history
  625,094 selections). 16/256 per layer = **6% of a layer's set**, and only experts in the pin set can be
  packed (`:9760` gates it). Ceiling on prefill hits from pins ≈ 6%.
- *tier host pool* — 11 experts / 165 MiB, sized from `VktConfig.ram_spare` = **179.6 MiB** (`:2540-2600`,
  log line). 11 of 11,008 places. Arithmetic zero.
- *the store's LRU* — 48 slots/layer (`:9215`), LRU per layer (`oldest_unreferenced_slot`). A layer's set
  is 250 experts, swept expert-major; the reuse distance between the two chunk visits of a layer is
  43 layers × ~250 ≈ 10,750 touches against 48 slots/layer → **structural thrash**, exactly the fork's
  "4-6x" comment (`fork/c/deepseek_v4.c:19288-19290`).
- Measured consequence: `expert_requests=30305 hits=10612 misses=19693` → **35.0% hit rate overall**
  (a0) and 21.5% in the 18 GiB arm. Roughly: the hits are decode (the 48 slots/layer *do* hold the decode
  hot set) + pins; the **prefill hit rate is ≈ pin-only ≈ 6%**.

**Re-reads vs unique — the arithmetic the task asks for:**
- `exp_bytes` = **15.0 MiB**/expert (log: "budget 7.04 GiB = 240 experts of 15.0 MiB"; 7.04 GiB / 479
  places = 15.05 MiB).
- Streamed instances = 11,127. Upper bound on unique (layer,expert) pairs = 43 × 256 = **11,008**.
  → **11,127 / 11,008 = 1.01.** Chunk 1 of a layer routes essentially the whole layer (2048 rows × topk
  ≈ 16k routings over 256 experts), so chunk 2 mostly *re-routes the same experts*, and the tier streams
  them **again** — but the *first* chunk already had to read them. The duplicate is only chunk 2's set,
  which is small in rows (467) and the log shows it collapsing: layer 0 streams 221 + 168, while the
  run average is 129.4/step ⇒ chunk-2 fills average ≈ 37/layer, not 168.
  → **re-read fraction ≈ 37×43 / 11,127 ≈ 14%** of instances, i.e. **~27 GiB of the 192 GiB**, not 2.8x.
  MEASURED-adjacent (derived from the log's own step counts + the two per-step lines); the exact
  distinct-vs-total needs the instrumentation arm in §4 (D0).
- Where the 2.8x actually comes from is the **counter mismatch**: 192.01 GiB is *slot fills* (PCIe, at
  8.59 GB/s = **22.4 s**, of which the run reports only `waited 3884.8 ms`), while 68.5 GiB is the
  fork's *store-view miss* bytes. The fork's own disk payload in that arm is 202.8 GB.

---

## 2. The fork's mechanism, precisely

Three layers, in the order they matter:

**(a) Per-layer validity bitmap → "at most once per layer" (the dedup).**
`fork/c/deepseek_v4.c:18206` `static unsigned char v4_moe_bank_valid[256];`,
cleared on layer change at `:19291-19294` (`if (bank_layer != weights->plan.layer) memset(...)`).
Refill builds the missing set from it at `:19415-19433`:
```c
for (int r = 0; r < batch * topk; r++) {
    int expert = ids[r];
    if (!bank_valid[expert]) { bank_valid[expert] = 2; /* pending */ missing[missing_count++] = expert; }
}
```
and sets `bank_valid[missing[i]] = 1` after upload (`:19548`). The bank itself is
`dsv4_cuda_expert_bank_create(256, 4096, 2048, …)` (`:19273`) = **256 experts = one whole layer,
~3.3 GiB VRAM**, addressable as a weight-stationary buffer; the fork sheds the VRAM tier to make room
(`:19239`).

**(b) Cache-bypassing reads → the retention survives the prefill.** `:19429-19438`:
"read each missing expert with `coli_v4_expert_store_read_bytes` into a BOUNDED pool of aligned buffers
… **never the store-lease path** … `read_bytes` touches no cache/lease/slot state ('no mutex is
needed')". Implementation `:13726-13775` (temp `V4ExpertSlot` with `aligned_slab = 1`, O_DIRECT window
read, `+8192` slack, no LRU entry, no `state->stats`). Pool: 8 × 16 MiB static, process lifetime
(`:19440-19456`).

**(c) The host-RAM ring, layer-ahead.** `v4_ram_take` / `v4_ram_prefetch` at `:19297-19302`
(`deepseek_v4_ram_prefetch.h`); arm: `moe-ram=on ring=3.19 GiB qd=16 take=par`
(`fork-full-r3.log:23`) and `layers=43 prefetched=11008 consumed=11008 takes=43 take_join_ms=0
take_upload_ms=332 streams=43 stream_ms=1476` (`:3335`). So the fork reads *every* expert once, into a
one-layer ring, 43 layers ahead with queue depth 16 — the bytes are the same as the port's, the
*stall* is not.

**(d) The precondition that makes (a) work: loop order.** `:12251-12259` — "The bank is a single
streaming buffer: target_batch's chunk loop sits INSIDE the layer loop, so one bank upload per layer
serves every … chunk of that layer". Verified: layer loop `:23524`, chunk loop `:23586`.
**Reuse distance = 0.** This is the single structural difference from the port's chain.

Cross-layer retention (decode) is a *different* mechanism: the fork's v5/MFU cache,
`v5-cache: slots=1982 slab=12.8MiB arena=24.69GiB gpu_cap=954` (`fork-full-r3.log:18`), frozen
MFU-ranked from history (`mfu_table.c`, `expert_cache.c`, `gpu_mfu_cache.c`). Note the capacity:
**24.69 GiB vs the port's 25.70 GiB target_cache** — same money, 68.5% vs 35%.

---

## 3. The design, ranked by expected-GiB-saved per LOC

Byte→time constant: 1 GiB at 8.59 GB/s = **0.125 s**; 10 GiB ⇒ 1.25 s. The measured run reports
`waited 3884.8 ms` against 11,127 streams ⇒ the fill path is ~83% hidden behind device work, so wall
savings are `≈ 0.125 × GiB × (fraction not hidden)`.

**S1 — `load_batch`: the engine's parallel read hook (~40–60 LOC, highest confidence/cost).**
`vk_tier.c:1244-1246` already asks: `T.c.load_batch(ctx, layer, eids+base, left, srcs, hs)`, but
`VktConfig` (`deepseek_v4.c:13862-13881`) never sets `.load_batch`, so every group of up to 64 experts
falls back to `T.c.load` **one expert at a time** (`:1256-1268`), each a synchronous lease that takes
`state->mutex` and does the disk read inline (`deepseek_v4.c:8874-8930`). 410 sub-batches × ~64 serial
reads on the *engine thread*. GiB saved: **0**. Estimated wall: **5–15 s** (removes engine-thread
serialization that the double buffer cannot hide). Upstreamable as a generic tier/engine hook; the fork
has the same idea under a different name.

**S2 — cache-bypassing stream reads: `read_bytes` in the port (~120–200 LOC, the fork's mechanism (b)).**
The port lacks the function entirely. Add `coli_v4_expert_store_read_bytes`-equivalent over the port's
`coli_st_read_at_streaming_rep` + a bounded aligned stage pool (8 × 16 MiB), then have
`v4_vkt_load`/`load_batch` use it for `BC_STREAM` (never for resident/promote/CPU-union asks). Effects:
(i) the 192 GiB stops flooding the 25.7 GiB LRU ⇒ the decode hot set and the rows16 pins survive the
prefill (**the 68.5%-vs-35% hit-rate gap lives here as much as in the policy**); (ii) `bw_up` rises ⇒
`R` rises ⇒ the policy shifts (log it). GiB saved: 0 by count — **hits converted from miss to hit**,
which is exactly what the fork's 68.5% counter reads. Estimated wall: **3–10 s** on TTFT,
and it protects decode (device share 14% today). One caveat: `note_routed` must still be called for
every routing — it is (`v4_vkt_routed`, `deepseek_v4.c:13708`), independent of the lease.

**S3 — step-local dedup.** **Already implemented**: `order[]` holds each expert once (`vk_tier.c:1503-
1515`), and an expert with more rows than a sub-batch is cut into parts that share one slot
(`:1571-1590`, `pending`). Expected gain **0**; do not write it.

**S4 — cross-chunk retention of streamed experts (the dossier's L1 shape, ~150–250 LOC).**
Replace free-on-non-route (`:1435`, `:1672`, `:1697`) with a per-layer keep + LRU inside the 239 slots.
Value is bounded by the reuse distance: on the chain path that is 43 layers ≈ 2.0–3.5 GiB of working
set, versus 3.5 GiB of slots ⇒ it *cannot* hold it (`O-day2` §11's 0-hit ring is the same arithmetic).
Expected: **~1–4% of the stream (~2–8 GiB) at chunk count 2**; the dossier's 40% figure belongs to a
4-chunk (768-row) config. **This is the shape to drop unless S5 is done.**

**S5 — loop order: make the layer's chunks adjacent, as the fork does (~600–1200 LOC, L).**
Two ways: (a) re-plumb `v4c_forward_seg` to layer-outer/chunk-inner for the prompt (needs window rings of
`W + n` rows and per-layer multi-chunk submissions; the CPU path `deepseek_v4.c:14857` is the model);
(b) grow the chunk to the whole prompt (C=1) by trading the 239 stream slots' VRAM for chunk rows —
3.5 GiB at 1054 KiB/row ≈ 3,400 rows, enough for 2,515. **But note the ceiling: at C=2 the port is
already at 1.01x unique, so (b) saves ~4%; only a C=4 configuration (768-row chunks — the *default*
profile, `base-18g-r1.log`) has 40% to recover.** So S5's value is contingent on the profile keeping
small chunks. Verify with D0 before writing it.

**S6 — plan rebalance: fund the pool / the store by shrinking the plan (~50 LOC of knobs, zero code).**
`ram_spare` = 179.6 MiB because `target_cache` = 25.70 GiB with `target_slots=48`
(`l3-a0-hot0.log`; `ressource_plan.py` writes the plan). `COLI_VK_TIER_HOST_GB` overrides the pool and
already warns "over the RAM plan" (`vk_tier.c:2568-2580`). Expect **0 GiB saved on TTFT** (the pool is
read in place over PCIe — strictly worse than a slot) and a decode-side effect at best. The interesting
half of this shape is not capacity but *policy*: 48 LRU slots/layer (35% hits) vs the fork's frozen
1,982-slot MFU arena (68.5%) at the same 25 GiB. That is what the L3 hot overlay tested — and it measured
**neutral**, so treat "freeze the store" as contested (§5 OPEN-2).

**Ranking (expected wall saved per LOC):** S1 > S2 > (S6 as a knob to try) > S4 ≫ S5. S3 is a no-op.
Nothing in this list reaches the 90 s; see §2 OPEN-0.

---

## 4. Falsification plan (the smallest arm set)

Method = the campaign's: interleaved configs, ×3 reps, same prompt (2,515 tok / 64 gen), RDNA2 profile,
`run-arm2276.sh` + `chain-arms.sh` in `bench/up2276/`; a rep set is ~5.5 min/run (TTFT ~200 s + ~95–115 s
decode) plus the warm start. Today's band (`l3rep-ctl-{1,2,3}.log` vs `l3rep-ly50-{1..3}`) means a single
rep cannot resolve <5%.

- **D0 (diagnostic, 1 run, ~100 LOC of counters — do this first, it decides everything):** per-layer
  counters in `st_fill_group`/`issue_big`: distinct (layer,expert) streamed vs total fills; fills per
  chunk index; store misses split prefill vs decode; wasted prefetch fills. Settles §1's 14% and §2's
  1.24x vs 2.8x. Gate: if distinct/total > 0.85 **and** the payload delta to the fork is <30 GB, then
  L1's framing is falsified and the effort belongs on the host-MoE wall (134.2 s "routed experts on the
  host" vs the fork's device MoE), not on bytes.
- **A (S2):** a0 + cache-bypassing stream reads; 3 reps. Read: `hits/misses` shift, decode device share,
  TTFT, `packed_slots` (must stop collapsing: 953 of 25.7 GiB today).
- **B (S1):** a0 + `.load_batch` (qd 8); 3 reps. Read: `stream GB/s` (`bw_up`), `R` (must be logged),
  `waited`, TTFT.
- **C (S2+S1 together):** 3 reps — they compose (the pool is what makes the parallel reads bounded).
- **Control:** a0 × 3, same day, interleaved.
- **Also needed to keep the arms comparable:** pin `COLI_VK_TIER_STREAM_ROWS` (freezes `R`, killing the
  `bw_up` feedback) in at least one B/C rep, or the policy moves under the measurement.

Success gate for L1 as a lever: A or C must beat the control by more than the rep band on TTFT **without**
costing decode (the slots 256 ↔ decode 0.528/0.654 frontier, dossier §6 cross-cutting note).

---

## 5. Risks

- **RAM budget.** 28 GiB plan (`target_cache` 25.70 + dense 0.25 + head) + host pool + slot staging is the
  documented OOM shape (`HANDOVER` §8 trap 2: "28 GiB plan + 8 GiB pool + 192 slots OOMs a 34 GiB
  cgroup"). Slot staging is *host* RAM (~15 MiB/expert); a stage pool of 8 × 16 MiB is already in the
  fork's budget but new to the port. Any S4 retention multiplies it by N.
- **`.coli_usage` snapshot contract.** The store writes the selection history at exit
  (`[STATS] 1300944 selections across 9097 distinct experts -> …/.coli_usage`) and reads it at start
  (`v4_autopin history=… selections=625094`). S2 changes the *read* path, not the *note* path
  (`v4_vkt_routed` → `coli_v4_expert_store_note_routed`, `deepseek_v4.c:13708`) — but a refactor that
  routes streaming through a new function must keep the note, or the history rots silently and the next
  run's autopin/pins regress. Also `packed_slots` (rows16) must not be starved by a new stage pool.
- **Numerics.** Bytes are identical (same expert bytes, same device form-0 conversion, same tiled GEMM in
  the slot) so a *retained* slot is bit-identical to a *fresh* one — **provided** the arm does not move an
  expert between the CPU path and the device path: `stream_min_rows` (`:1120`) already moves experts when
  `bw_up` changes, and the RDNA2 profile runs `COLI_VK_CHAIN_GEMM_ROWS=-1`, which the chain's own comment
  (`deepseek_v4_chain.h:1189-1198`) says makes *a prompt row's bits depend on how the prompt is
  chunked*. So: S5 (chunk count) **is** a numerics change; S1/S2 must not be. Gate with the identical-
  output check the campaign already uses (all arm logs carry the same generated text) plus
  `deepseek-v4-tiny-check` and the `.golden` leg (≈4.1 h/run — budget it only for S5).
- **Determinism.** `COLI_VK_TIER_SYNC=1`'s deterministic drain must survive any new async stage pool.
- **Coupling.** `bw_up`/`cpu_row_ms` EWMAs feed the admission rule — arms must log `R`.
- **Attribution risk (the big one).** Three competing explanations of the fork's edge are on record
  (dossier §2.3); the 68.4 GB counter is *store-view only* in an arm that also ran the ring. Do not size
  effort on 2.8x before D0.

## 6. Size estimate

| item | lines | files | risk |
|---|---:|---|---|
| D0 instrumentation | 80–120 | `vk_tier.c`, log line | none |
| S1 `.load_batch` | 40–60 | `deepseek_v4.c` (V4 engine), tier ready | low |
| S2 cache-bypassing reads + stage pool | 120–200 | `deepseek_v4.c` (store + `v4_vkt_load`), `vk_tier.h` doc | med (numerics, `.coli_usage`) |
| S3 step-local dedup | 0 (exists) | — | — |
| S4 cross-chunk retention | 150–250 | `vk_tier.c` only | med, low value on chain path |
| S5 loop order / C=1 chunk | 600–1200 | `deepseek_v4_chain.h`, `vk_chain.c` | high (numerics gate, merge cost: `vk_chain.c` is 26% of the last 400 upstream commits) |
| test | 100–200 | `c/tests` (streaming retention / bypass) | — |

Total for the defensible slice (D0 + S1 + S2 + tests): **~400–600 lines, no shaders**, upstreamable as
"the tier's streaming reads do not enter the expert cache, and it reads its groups in parallel".

---

## OPENs

1. **OPEN-0 (blocking the premise):** is the byte economy the #1 TTFT lever? The same-arm numbers say the
   fork's disk payload is 203 GB vs the port's 251 GB (1.24x ≈ 5 s) while TTFT is 104 s vs 202 s, and the
   port's own report attributes 134.2 s to "routed experts on the host" and only 3.9 s to waiting. D0
   plus one host-MoE arm settles it.
2. **OPEN-1:** does the port's streamed load actually cost the *store* anything in the metric that
   matters — i.e. how much of the 19,693 misses / 251 GB would S2 convert to hits? Needs D0 + arm A.
3. **OPEN-2:** the frozen-cache policy (fork 1,982 MFU slots → 68.5%) vs the port's L3 hot overlay
   (measured **neutral**). Same capacity, opposite results — either the overlay's pin set is wrong (8 pins
   vs 1,982 slots) or the freeze is not the mechanism. Not resolvable from code.
4. **OPEN-3:** the exact re-read fraction (14% derived here vs the dossier's 40–45% at 768-row chunks).
   Depends on the profile's chunk count; D0 gives it directly.
5. **OPEN-4:** whether `T.c.stage` (`vk_dmoe.h`'s host staging carve, `vk_tier.c:2806-2820`) is dead for
   V4 — `VktConfig` never sets `.stage`, so the 3.84 GiB host-staging reservation never happens; confirm
   that is intended and not a missed knob for the host pool.
