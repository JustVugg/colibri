# OLMoE tier A/B benchmark: `cold_nvme_b`

- date: 2026-09-29T20:21:34
- engine: `c/olmoe` (argv `8 8`), SNAP: `/Users/khalid/models/olmoe_merged`
- serve env: `SERVE=1 CTX=2048`; config env passed through as-is
- reps: 2; warmup: 1 discarded serve run(s) of `demand`; per-run timeout: 900s
- primary metric: decode-only tok/s = 1000 / mean(DATA frame delta), the 199 deltas after the first DATA
- interleaving: rep-major (`for rep in 1..2: for config in configs`), so page-cache/thermal drift cancels
- JSONL: `c/bench_tier/ab_cold_nvme_b.jsonl`

| config | decode tok/s (med) | P50 ms | P90 ms | P99 ms | hit% | disk ms/step | SHA-256(16) | standalone exact |
|---|---|---|---|---|---|---|---|---|
| demand | 3.800 | 217.0 | 318.2 | 1099.6 | 27.8 | 193.7 | 88e4d132afc16539 | 200/200 OK |
| pilot2_markov2 | 4.276 | 216.4 | 276.4 | 499.6 | 28.4 | 226.5 | 88e4d132afc16539 | 200/200 OK |

## Standalone golden check (`./olmoe 8 8 bench_tier/ref200.json`)

| config | match | hit% | tok/s | wall s |
|---|---|---|---|---|
| demand | 200/200 | 27.7 | 3.01 | 67.1 |
| pilot2_markov2 | 200/200 | 28.2 | 3.04 | 66.4 |

## Exact commands

Harness invocation:

```
python3 c/bench_tier/bench_ab.py --label cold_nvme_b --reps 2 --cap 8 --timeout 900 --configs 'demand:PILOT=0|HOT=0|DEMAND_POLICY=lru|EXPERT_DIRECT=1|RECENT_RING=0|RECENT_BLOOM=0|MARKOV=0|GROUP_EVICT=0,pilot2_markov2:PILOT=2|HOT=0|DEMAND_POLICY=lru|EXPERT_DIRECT=1|MARKOV=1|MARKOV_TOP=2|RECENT_RING=0|RECENT_BLOOM=0|GROUP_EVICT=0'
```

Before every run (serve and standalone): `rm -f /Users/khalid/models/olmoe_merged/hot_pinned.bin`

Serve run (once per config per rep; cwd `c`):

```
# demand
SNAP=/Users/khalid/models/olmoe_merged CTX=2048 SERVE=1 DEMAND_POLICY=lru EXPERT_DIRECT=1 GROUP_EVICT=0 HOT=0 MARKOV=0 PILOT=0 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 8 8
stdin: SUBMIT bench 0 430 200 0.0 1.0 + payload c/bench_tier/prompt.txt (430 bytes) + LF
# pilot2_markov2
SNAP=/Users/khalid/models/olmoe_merged CTX=2048 SERVE=1 DEMAND_POLICY=lru EXPERT_DIRECT=1 GROUP_EVICT=0 HOT=0 MARKOV=1 MARKOV_TOP=2 PILOT=2 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 8 8
stdin: SUBMIT bench 0 430 200 0.0 1.0 + payload c/bench_tier/prompt.txt (430 bytes) + LF
```

Standalone golden check (once per config, after the serve matrix):

```
# demand
SNAP=/Users/khalid/models/olmoe_merged DEMAND_POLICY=lru EXPERT_DIRECT=1 GROUP_EVICT=0 HOT=0 MARKOV=0 PILOT=0 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 8 8 bench_tier/ref200.json
# pilot2_markov2
SNAP=/Users/khalid/models/olmoe_merged DEMAND_POLICY=lru EXPERT_DIRECT=1 GROUP_EVICT=0 HOT=0 MARKOV=1 MARKOV_TOP=2 PILOT=2 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 8 8 bench_tier/ref200.json
```

## Per-run detail

| run | config | rep | decode tok/s | n DATA | SHA-256(16) | status |
|---|---|---|---|---|---|---|
| serve | demand | 1 | 3.629 | 200 | 88e4d132afc16539 | ok |
| serve | pilot2_markov2 | 1 | 4.375 | 200 | 88e4d132afc16539 | ok |
| serve | demand | 2 | 3.970 | 200 | 88e4d132afc16539 | ok |
| serve | pilot2_markov2 | 2 | 4.177 | 200 | 88e4d132afc16539 | ok |
| gold | demand | - | 3.010 | - | - | exact |
| gold | pilot2_markov2 | - | 3.040 | - | - | exact |
