# OLMoE tier A/B benchmark: `cold_nvme_a`

- date: 2026-09-29T20:12:03
- engine: `c/olmoe` (argv `8 8`), SNAP: `/Users/khalid/models/olmoe_merged`
- serve env: `SERVE=1 CTX=2048`; config env passed through as-is
- reps: 1; warmup: 1 discarded serve run(s) of `demand`; per-run timeout: 900s
- primary metric: decode-only tok/s = 1000 / mean(DATA frame delta), the 199 deltas after the first DATA
- interleaving: rep-major (`for rep in 1..1: for config in configs`), so page-cache/thermal drift cancels
- JSONL: `c/bench_tier/ab_cold_nvme_a.jsonl`

| config | decode tok/s (med) | P50 ms | P90 ms | P99 ms | hit% | disk ms/step | SHA-256(16) | standalone exact |
|---|---|---|---|---|---|---|---|---|
| demand | 4.437 | 190.0 | 323.1 | 702.8 | 27.8 | 179.3 | 88e4d132afc16539 | 200/200 OK |
| pilot2 | 4.480 | 212.4 | 267.7 | 455.0 | 31.1 | 194.6 | 88e4d132afc16539 | 200/200 OK |
| pilot3 | 2.207 | 232.9 | 631.0 | 4347.1 | 31.1 | 294.3 | 88e4d132afc16539 | 200/200 OK |
| pilot2_workers4 | 3.116 | 266.9 | 431.6 | 1460.9 | 31.2 | 244.6 | 88e4d132afc16539 | 200/200 OK |
| pilot2_markov2 | 5.124 | 191.7 | 223.6 | 320.3 | 31.1 | 182.4 | 88e4d132afc16539 | 200/200 OK |

## Standalone golden check (`./olmoe 8 8 bench_tier/ref200.json`)

| config | match | hit% | tok/s | wall s |
|---|---|---|---|---|
| demand | 200/200 | 27.7 | 3.56 | 56.7 |
| pilot2 | 200/200 | 31.0 | 3.72 | 54.6 |
| pilot3 | 200/200 | 31.0 | 2.94 | 68.8 |
| pilot2_workers4 | 200/200 | 31.2 | 3.17 | 63.8 |
| pilot2_markov2 | 200/200 | 31.1 | 3.77 | 53.5 |

## Exact commands

Harness invocation:

```
python3 c/bench_tier/bench_ab.py --label cold_nvme_a --reps 1 --cap 8 --timeout 900 --configs 'demand:PILOT=0|HOT=0|DEMAND_POLICY=lru|EXPERT_DIRECT=1|RECENT_RING=0|RECENT_BLOOM=0|MARKOV=0|GROUP_EVICT=0,pilot2:PILOT=2|HOT=0|DEMAND_POLICY=lru|EXPERT_DIRECT=1|RECENT_RING=0|RECENT_BLOOM=0|MARKOV=0|GROUP_EVICT=0,pilot3:PILOT=3|HOT=0|DEMAND_POLICY=lru|EXPERT_DIRECT=1|RECENT_RING=0|RECENT_BLOOM=0|MARKOV=0|GROUP_EVICT=0,pilot2_workers4:PILOT=2|HOT=0|DEMAND_POLICY=lru|EXPERT_DIRECT=1|PILOT_WORKERS=4|RECENT_RING=0|RECENT_BLOOM=0|MARKOV=0|GROUP_EVICT=0,pilot2_markov2:PILOT=2|HOT=0|DEMAND_POLICY=lru|EXPERT_DIRECT=1|MARKOV=1|MARKOV_TOP=2|RECENT_RING=0|RECENT_BLOOM=0|MARKOV=0|GROUP_EVICT=0'
```

Before every run (serve and standalone): `rm -f /Users/khalid/models/olmoe_merged/hot_pinned.bin`

Serve run (once per config per rep; cwd `c`):

```
# demand
SNAP=/Users/khalid/models/olmoe_merged CTX=2048 SERVE=1 DEMAND_POLICY=lru EXPERT_DIRECT=1 GROUP_EVICT=0 HOT=0 MARKOV=0 PILOT=0 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 8 8
stdin: SUBMIT bench 0 430 200 0.0 1.0 + payload c/bench_tier/prompt.txt (430 bytes) + LF
# pilot2
SNAP=/Users/khalid/models/olmoe_merged CTX=2048 SERVE=1 DEMAND_POLICY=lru EXPERT_DIRECT=1 GROUP_EVICT=0 HOT=0 MARKOV=0 PILOT=2 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 8 8
stdin: SUBMIT bench 0 430 200 0.0 1.0 + payload c/bench_tier/prompt.txt (430 bytes) + LF
# pilot3
SNAP=/Users/khalid/models/olmoe_merged CTX=2048 SERVE=1 DEMAND_POLICY=lru EXPERT_DIRECT=1 GROUP_EVICT=0 HOT=0 MARKOV=0 PILOT=3 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 8 8
stdin: SUBMIT bench 0 430 200 0.0 1.0 + payload c/bench_tier/prompt.txt (430 bytes) + LF
# pilot2_workers4
SNAP=/Users/khalid/models/olmoe_merged CTX=2048 SERVE=1 DEMAND_POLICY=lru EXPERT_DIRECT=1 GROUP_EVICT=0 HOT=0 MARKOV=0 PILOT=2 PILOT_WORKERS=4 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 8 8
stdin: SUBMIT bench 0 430 200 0.0 1.0 + payload c/bench_tier/prompt.txt (430 bytes) + LF
# pilot2_markov2
SNAP=/Users/khalid/models/olmoe_merged CTX=2048 SERVE=1 DEMAND_POLICY=lru EXPERT_DIRECT=1 GROUP_EVICT=0 HOT=0 MARKOV=0 MARKOV_TOP=2 PILOT=2 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 8 8
stdin: SUBMIT bench 0 430 200 0.0 1.0 + payload c/bench_tier/prompt.txt (430 bytes) + LF
```

Standalone golden check (once per config, after the serve matrix):

```
# demand
SNAP=/Users/khalid/models/olmoe_merged DEMAND_POLICY=lru EXPERT_DIRECT=1 GROUP_EVICT=0 HOT=0 MARKOV=0 PILOT=0 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 8 8 bench_tier/ref200.json
# pilot2
SNAP=/Users/khalid/models/olmoe_merged DEMAND_POLICY=lru EXPERT_DIRECT=1 GROUP_EVICT=0 HOT=0 MARKOV=0 PILOT=2 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 8 8 bench_tier/ref200.json
# pilot3
SNAP=/Users/khalid/models/olmoe_merged DEMAND_POLICY=lru EXPERT_DIRECT=1 GROUP_EVICT=0 HOT=0 MARKOV=0 PILOT=3 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 8 8 bench_tier/ref200.json
# pilot2_workers4
SNAP=/Users/khalid/models/olmoe_merged DEMAND_POLICY=lru EXPERT_DIRECT=1 GROUP_EVICT=0 HOT=0 MARKOV=0 PILOT=2 PILOT_WORKERS=4 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 8 8 bench_tier/ref200.json
# pilot2_markov2
SNAP=/Users/khalid/models/olmoe_merged DEMAND_POLICY=lru EXPERT_DIRECT=1 GROUP_EVICT=0 HOT=0 MARKOV=0 MARKOV_TOP=2 PILOT=2 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 8 8 bench_tier/ref200.json
```

## Per-run detail

| run | config | rep | decode tok/s | n DATA | SHA-256(16) | status |
|---|---|---|---|---|---|---|
| serve | demand | 1 | 4.437 | 200 | 88e4d132afc16539 | ok |
| serve | pilot2 | 1 | 4.480 | 200 | 88e4d132afc16539 | ok |
| serve | pilot3 | 1 | 2.207 | 200 | 88e4d132afc16539 | ok |
| serve | pilot2_workers4 | 1 | 3.116 | 200 | 88e4d132afc16539 | ok |
| serve | pilot2_markov2 | 1 | 5.124 | 200 | 88e4d132afc16539 | ok |
| gold | demand | - | 3.560 | - | - | exact |
| gold | pilot2 | - | 3.720 | - | - | exact |
| gold | pilot3 | - | 2.940 | - | - | exact |
| gold | pilot2_workers4 | - | 3.170 | - | - | exact |
| gold | pilot2_markov2 | - | 3.770 | - | - | exact |
