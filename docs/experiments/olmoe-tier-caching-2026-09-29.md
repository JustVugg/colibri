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
+5.0pp at cap=8 but loses 3.2-4.1pp at cap>=16 (Belady oracle at cap=24 is 84.1%,
LRU 68.4%).

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

## Results (cap=8, reps=5, interleaved; machine under external load — relative deltas are the signal)
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
