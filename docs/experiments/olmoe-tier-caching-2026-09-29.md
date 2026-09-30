# OLMoE tier-caching & PILOT evaluation — 2026-09-29

Engine: c/olmoe.c (OLMoE-1B-7B int8 experts, 16 layers x 64 experts, top-8), cap=8 slots/layer
(~0.8 GB expert cache of a 6.4 GB model). Workload: 200-token greedy decode from
c/bench_tier/ref200.json (91-token prompt), serve-mode driver c/bench_tier/bench_ab.py.

## Root cause
Decode is read-bandwidth-bound, not latency-bound: with a warm OS page cache the
"disk" tier is served from RAM, so speculative prefetch adds bytes without removing
demand bytes. At cap=8 the cache holds exactly one token's top-8 per layer, so the
hit rate is governed by consecutive-token routing overlap (40.3% ceiling; 27.7%
realized under LRU). Offline replay (c/bench_tier/policy_replay.py) reproduces the
engine's LRU hit counts exactly (10249/26743) and shows frequency-primary LFRU wins
+5.0pp at cap=8 but loses 3.2-4.1pp at cap>=16 (Belady oracle at cap=24 is 84.1%;
measured LRU hit 68.4% at cap=24 — see Results).

## Changes (all env-gated; defaults preserve prior behavior)
- Phase 1 (c/tier.h + c/olmoe.c): TierRecent temporal ring (16 routed sets/layer) and
  two-generation rolling Bloom over recent routing; guards demand/prefetch victim
  selection. RECENT_RING / RECENT_BLOOM / BLOOM_WINDOW / RECENT_BONUS.
- Phase 2 (c/tier.h + c/olmoe.c): TierMarkov saturating transition counts
  ((n_layers-1)*E*E, 120 KB for OLMoE) with MARKOV / MARKOV_TOP / MARKOV_DECAY /
  MARKOV_STATS; PILOT stages L+1/L+2 successors right after layer L routing.
- Phase 3 (c/olmoe.c): DEMAND_POLICY=lru|lfru|auto (auto: lfru when cap<=12);
  GROUP_EVICT co-routing affinity eviction (128 KB table, decode-only);
  EXPERT_DIRECT F_NOCACHE/O_DIRECT reads via aligned bounce buffers;
  PILOT_WORKERS 1..8 SPMC loader threads; ALIGN_SLOTS 2 MiB slot alignment.
- c/Makefile: dependency lines for tier.h on olmoe/test targets.

## Results (cap=24, reps=3, interleaved; external load present — flagged)

Measured 2026-09-29 21:37-22:06 (`ab_final_cap24_quiet`). An unrelated Ollama
llama-server job ran throughout (~1.5-2 cores; 1-min load 7.3-17.6, median 12.3;
see `ab_final_cap24_quiet_load.log`), so absolute tok/s are depressed and P99 is
noisy. Hit rates are deterministic and identical across reps; every serve run was
token-exact with one SHA.

| config | decode tok/s (med) | P50 ms | P90 ms | P99 ms | hit% | disk ms/step | SHA-256(16) | standalone exact |
|---|---|---|---|---|---|---|---|---|
| baseline (LRU, no prefetch) | 1.529 | 530.9 | 1447.7 | 2869.3 | 68.4 | 271.8 | 88e4d132afc16539 | 200/200 OK |
| best_demand_group (LFRU + GROUP_EVICT + EXPERT_DIRECT) | 4.506 | 183.1 | 309.3 | 1117.1 | 64.9 | 105.3 | 88e4d132afc16539 | 200/200 OK |
| p3_full (PILOT=2 + PILOT_WORKERS=4) | 5.761 | 149.9 | 225.7 | 545.8 | 70.5 | 103.8 | 88e4d132afc16539 | 200/200 OK |
| pilot2_markov2 (PILOT=2 + MARKOV_TOP=2) | 3.236 | 150.3 | 731.3 | 1245.7 | 70.4 | 145.4 | 88e4d132afc16539 | 200/200 OK |

### Clean-machine confirmation

Run A (cap=24, reps=3, interleaved) per-rep decode tok/s: baseline 0.952 / 1.540 /
1.529; best_demand_group 4.470 / 4.506 / 5.066; p3_full 3.882 / 6.782 / 5.761;
pilot2_markov2 3.236 / 2.002 / 5.585. Medians in the table above. All 12 serve runs
share one SHA-256[:16] (88e4d132afc16539) and all four standalone golds are 200/200.
Load log (63 samples): 1-min load min/med/max 7.25/12.29/17.63; the Ollama
llama-server held 5-205% CPU (>50% in 57/63 samples). The machine never went quiet
within the 20-minute wait, so Run A is flagged loaded per protocol.

Run B (cap=8, reps=3) was skipped: it is gated on the machine staying quiet, and the
external Ollama job ran throughout the wait and Run A, so a clean-absolute cap=8 pair
could not be obtained. Even under this load the policy split is stark: the buffered
baseline collapses (1.529 tok/s median, 271.8 ms/step disk) while LFRU + GROUP_EVICT
+ EXPERT_DIRECT stays fast (4.506 tok/s, 105.3 ms/step) despite a *lower* hit rate
(64.9% vs 68.4%) — buffered reads under memory pressure cost far more than the misses
they save.

### Earlier under-load cap=8 phase sweep (reps=5; superseded, retained for history)
| Optimization Phase | Decode tok/s (median) | Cache hit % | Disk misses | P99 ms |
|---|---|---|---|---|
| Baseline (LRU, no prefetch) | 3.474 | 27.8 | 26743 | 819.4 |
| + Phase 1 (Temporal + Bloom) | 3.437 | 27.8 | 26743 | 896.9 |
| + Phase 2 (Markov + prefetch) | 3.392 | 27.8 | 26743 | 744.4 |
| + Phase 3 (LFRU + Groups + Direct I/O + workers) | 4.845 | 33.3 | 24668 | 344.1 |
| Best config (DEMAND_POLICY=lfru + GROUP_EVICT + EXPERT_DIRECT) | 5.062 | 33.3 | 24668 | 378.9 |

Deterministic deltas: lfru cuts misses 7.7% (26743 -> 24677 at PILOT=0; 24668 with
GROUP_EVICT) and raises hit rate +5.5pp. On a quiet machine (earlier phase-2 run:
baseline 11.186 tok/s, disk 59.4 ms/step of 90.3 ms) the hit-rate delta alone projects
~+4-5% tok/s; the larger deltas above were measured while an unrelated LLM job evicted
the model from page cache and inflated disk stalls, which the LFRU policy then avoided.

## Correctness
- All configs token-exact: standalone 200/200 on ref200.json for every measured config
  (and 12/12 on ref_olmoe_real.json); serve payload SHA-256 88e4d132afc16539 identical
  across all 37 successful serve runs.
- Baseline hit/miss byte-identical to pre-change history (10249/26743); policy changes
  never touch routing/logits/sampling/KV.

## Guardrails
- Deterministic fallback: every new path degrades to the legacy behavior on absent
  arrays/bad env values; in-flight slots are never stolen.
- Memory: ring+Bloom 18 KB, Markov 120 KB, coc 128 KB — all allocated only when the
  matching env flag is set and capped (Markov/coc refuse > 8 MB or E > 512).
- Latency: victim keys are O(cap) (cap=8 -> 64 ops), no allocation in hot paths.

## Rejected on this host
- PILOT prefetch itself: +3.3pp hit but -4.1% tok/s at cap=8 (speculative bytes).
- MARKOV chains: -1.7 to -2.9pp hit, -1.6 to -2.3% tok/s vs PILOT=2 (displaces residents).
- ALIGN_SLOTS: malloc is already 2 MiB-aligned on macOS (no-op).
- EXPERT_DIRECT: neutral-to-positive under page-cache pressure; kept opt-in.

## True cold-disk regime (page cache defeated by working-set size)
EXPERT_DIRECT=F_NOCACHE does not force device reads of resident pages on macOS
(UBC still serves them), and the machine's effective file cache (~1.3 GB under
memory pressure) cannot hold the 6.3 GB expert working set, so misses are real
NVMe reads at ~1.9 ms each (measured with c/iobench; warm buffered 0.7-1.0 ms,
cold 1.9-4.0 ms). Under this true-disk regime the PILOT verdict flips:
- Phase A (1 rep each, EXPERT_DIRECT=1): demand 4.437 tok/s / P99 702.8 ms;
  pilot2 4.480 / 455.0; pilot3 2.207 / 4347.1; pilot2_workers4 3.116 / 1460.9;
  pilot2_markov2 5.124 / 320.3 (hit 31.1%, disk 182.4 ms/step).
- Phase B (2 reps, interleaved): demand 3.800 med / P99 1099.6 vs
  pilot2_markov2 4.276 med / P99 499.6 => +12.5% tok/s, P90 -13%, P99 -55%.
- PILOT=2 alone ~+1% (noise); PILOT=3 and PILOT_WORKERS=4 regress (I/O and CPU
  contention); Markov chains are the decisive win once misses cost device latency.
All 11 cold runs token-exact (one SHA; 12/12 standalone golds 200/200).

### Kitchen-sink combination (lfru + groups + direct I/O + Markov)

The full combination — `DEMAND_POLICY=lfru + GROUP_EVICT=1 + EXPERT_DIRECT=1 +
PILOT=2 + MARKOV=1 MARKOV_TOP=2` — had never been tested together; this run
tests it against `demand` and the previous cold winner `pilot2_markov2`, plus
the optional `kitchen_sink_w2` (`PILOT_WORKERS=2`).

Method (`ab_cold_kitchen_sink`, 2026-09-30 12:13–12:39; ablation
`ab_cold_kitchen_sink_ablate`, 12:39–12:44): the machine carried pre-existing
pressure (active Ollama llama-server PID 55014, 6.1→8.5 GB RSS, 49–194% CPU
throughout; swap ~20.4/21.5 GB used at start), so a controlled
anonymous-memory ballast was added to pin the regime instead of waiting for
natural pressure: python mmap, 8 KiB urandom + 8 KiB zeros per 16 KiB page,
target 4.0 GiB in 512 MiB chunks, swap-free floor 400 MB. It stopped at
1.00 GiB (swap free fell 702→353 MB) and stayed alive for every run
(`load_cold_kitchen_sink.log`; ballast PIDs 59845/59881, never GONE). Regime
gate with `c/iobench` (6.3 MB × 64 reads × 8 threads, buffered): fresh
first-run reads on untouched shards 4.0 / 2.8 / 3.3 / 3.2 / 3.2 ms/block
(median 3.2 ≥ 1.5 ms; warm repeats of the same offsets 0.8–1.3 ms). The
ballast measurably evicted previously-warm pages: the shard-0 offsets first
measured at 1.2 ms (warm) read 3.2 ms after ballast. File-backed cache across
the run window: 0.74–2.55 GB (median 2.02 GB) vs the 6.3 GB expert working
set; 1-min load 3.7–16.4 (median 7.8). Post-run re-check (fresh offsets,
6.5 MB, shard 3): 1.6 ms — regime held.

| config | decode tok/s (med) | P50 ms | P90 ms | P99 ms | hit% | disk ms/step | per-rep tok/s |
|---|---|---|---|---|---|---|---|
| demand (LRU, no prefetch) | 3.962 | 217.5 | 348.6 | 885.6 | 27.8 | 194.2 | 3.130 / 5.135 / 3.962 |
| pilot2_markov2 (LRU + PILOT=2 + MARKOV_TOP=2) | 4.380 | 223.6 | 279.7 | 411.9 | 28.2 | 235.1 | 3.410 / 4.380 / 4.731 |
| **kitchen_sink** (lfru + GROUP_EVICT + PILOT=2 + MARKOV_TOP=2 + EXPERT_DIRECT) | **4.817** | 181.4 | 286.8 | 594.0 | 33.3 | 149.5 | 3.989 / 4.817 / 5.187 |
| kitchen_sink_w2 (+ PILOT_WORKERS=2) | 4.218 | 202.3 | 297.0 | 710.0 | 33.3 | 171.2 | 4.218 / 3.640 / 4.767 |

Deltas: kitchen_sink vs demand +21.6% (median; per-rep +27.4 / −6.2 / +30.9);
kitchen_sink vs pilot2_markov2 +10.0% (median; +17.0 / +10.0 / +9.6 — all
three reps, so not within noise); pilot2_markov2 vs demand +10.6% (median;
+8.9 / −14.7 / +19.4). The demand rep-2 outlier (5.135 tok/s, P99 291.9 ms) is
a fast window — within-rep interleaving is the trustworthy signal.
`PILOT_WORKERS=2` regresses (median −12.4% vs kitchen_sink; 1/3 reps wins),
consistent with the workers=4 result — keep the single loader.

Deterministic miss counts (standalone golds, all 200/200): demand 10249/26743
(27.7%), pilot2_markov2 10425/26567 (28.2%), kitchen_sink and kitchen_sink_w2
12324/24668 (33.3%). All 12 serve runs share SHA-256[:16] 88e4d132afc16539.

**Interaction finding — the combination helps, but LFRU neutralizes the
prefetch's own hits.** Ablation `lfru_group_only` (lfru+GROUP_EVICT,
PILOT=0/MARKOV=0) interleaved against kitchen_sink, reps=2: 4.985 / 4.927 vs
4.876 / 2.986 tok/s (clean pair −2.2%; the rep-2 kitchen_sink outlier, P99
2745 ms, coincided with swap free dipping to 354 MB at 12:45 while swap total
ballooned to 33.8 GB — machine event, not config). Both ablation golds are
12324/24668. Hit-count fingerprint: PILOT+MARKOV adds +176 hits under LRU
(10425/26567 vs 10249/26743, with ~±13 run jitter) and **+0 under lfru+group**
(12324/24668 with prefetch, identical to the warm `best_demand_group` run
without it and to `p3_full`; five lfru+group golds, all exactly 12324/24668).
Mechanism: the LFRU victim key is `(heat<<8)|recent`, so a freshly staged
heat-0 slot (max score 255) is the preferred victim over any heat≥1 resident
(≥256) on the next demand miss (~92 misses/token) — staged slots do not
survive to be used; `PILOT_EVICT_GUARD` guards pilot-vs-resident only, not
demand-vs-staged. `MARKOV_STATS=1` diagnostics agree with the changed staging
pattern: staged candidates 9173 (lfru+group) vs 10323 (LRU) — fewer staged
because more predictions are already resident and dropped as duplicates —
48.5% vs 42.7% later routed (routing precision unchanged; the stat counts
routing, not slot survival). Net: the deterministic +5.1pp demand-hit gain of
lfru+group (−2075 misses) dominates the lost +176 prefetch hits, so the
kitchen sink is the best of the four configs in the main cold run — but it is
effectively lfru+group demand-only plus overlapped-but-hitless prefetch I/O,
and its median P99 (594.0 ms) is mixed vs pilot2_markov2 (411.9). Artifacts:
`c/bench_tier/ab_cold_kitchen_sink{,_ablate}.jsonl` / `_summary.md`,
`c/bench_tier/load_cold_kitchen_sink.log`.

## Backport to core engines (commit de1682fd)
Shared demand-eviction key promoted to c/tier.h (`tier_demand_victim_key`,
`tier_demand_policy_env`, auto cap 12) and wired into c/colibri.c (GLM) and
c/qwen36.c (Qwen3.6; gained last_access). Default DEMAND_POLICY=lru is
bit-identical to prior victim choice in every engine. qwen36 required a new
Slot.hold guard: the planar-int4 path keeps slot pointers across further
expert_get calls, and frequency-primary LFRU can otherwise evict a
freshly-loaded low-frequency slot. Verified: model-free cache-index tests pass
under lru/lfru/auto; OLMoE ref200 keeps 10249/26743 (lru) and 12315/24677 (lfru),
200/200; qwen36 emits identical token ids under lru/lfru/auto while hit rates
differ (policy engaged, output invariant).
