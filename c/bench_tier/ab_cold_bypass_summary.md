# OLMoE tier A/B benchmark: `cold_bypass`

- date: 2026-09-29T19:30:10
- engine: `c/olmoe` (argv `8 8`), SNAP: `/Users/khalid/models/olmoe_merged`
- serve env: `SERVE=1 CTX=2048`; config env passed through as-is
- reps: 1; warmup: 1 discarded serve run(s) of `demand_cached`; per-run timeout: 900s
- primary metric: decode-only tok/s = 1000 / mean(DATA frame delta), the 199 deltas after the first DATA
- interleaving: rep-major (`for rep in 1..1: for config in configs`), so page-cache/thermal drift cancels
- JSONL: `c/bench_tier/ab_cold_bypass.jsonl`

| config | decode tok/s (med) | P50 ms | P90 ms | P99 ms | hit% | disk ms/step | SHA-256(16) | standalone exact |
|---|---|---|---|---|---|---|---|---|
| demand_cached | 3.977 | 240.3 | 314.1 | 521.1 | 27.8 | 177.2 | 88e4d132afc16539 | 200/200 OK |
| demand_direct | 4.275 | 222.3 | 335.3 | 658.9 | 27.8 | 175.9 | 88e4d132afc16539 | 200/200 OK |

## Standalone golden check (`./olmoe 8 8 bench_tier/ref200.json`)

| config | match | hit% | tok/s | wall s |
|---|---|---|---|---|
| demand_cached | 200/200 | 27.7 | 2.39 | 84.1 |
| demand_direct | 200/200 | 27.7 | 2.86 | 70.6 |

## Exact commands

Harness invocation:

```
python3 c/bench_tier/bench_ab.py --label cold_bypass --reps 1 --cap 8 --warmup 1 --timeout 900 --configs 'demand_cached:PILOT=0|HOT=0|DEMAND_POLICY=lru|EXPERT_DIRECT=0|RECENT_RING=0|RECENT_BLOOM=0|MARKOV=0|GROUP_EVICT=0,demand_direct:PILOT=0|HOT=0|DEMAND_POLICY=lru|EXPERT_DIRECT=1|RECENT_RING=0|RECENT_BLOOM=0|MARKOV=0|GROUP_EVICT=0'
```

Before every run (serve and standalone): `rm -f /Users/khalid/models/olmoe_merged/hot_pinned.bin`

Serve run (once per config per rep; cwd `c`):

```
# demand_cached
SNAP=/Users/khalid/models/olmoe_merged CTX=2048 SERVE=1 DEMAND_POLICY=lru EXPERT_DIRECT=0 GROUP_EVICT=0 HOT=0 MARKOV=0 PILOT=0 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 8 8
stdin: SUBMIT bench 0 430 200 0.0 1.0 + payload c/bench_tier/prompt.txt (430 bytes) + LF
# demand_direct
SNAP=/Users/khalid/models/olmoe_merged CTX=2048 SERVE=1 DEMAND_POLICY=lru EXPERT_DIRECT=1 GROUP_EVICT=0 HOT=0 MARKOV=0 PILOT=0 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 8 8
stdin: SUBMIT bench 0 430 200 0.0 1.0 + payload c/bench_tier/prompt.txt (430 bytes) + LF
```

Standalone golden check (once per config, after the serve matrix):

```
# demand_cached
SNAP=/Users/khalid/models/olmoe_merged DEMAND_POLICY=lru EXPERT_DIRECT=0 GROUP_EVICT=0 HOT=0 MARKOV=0 PILOT=0 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 8 8 bench_tier/ref200.json
# demand_direct
SNAP=/Users/khalid/models/olmoe_merged DEMAND_POLICY=lru EXPERT_DIRECT=1 GROUP_EVICT=0 HOT=0 MARKOV=0 PILOT=0 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 8 8 bench_tier/ref200.json
```

## Per-run detail

| run | config | rep | decode tok/s | n DATA | SHA-256(16) | status |
|---|---|---|---|---|---|---|
| serve | demand_cached | 1 | 3.977 | 200 | 88e4d132afc16539 | ok |
| serve | demand_direct | 1 | 4.275 | 200 | 88e4d132afc16539 | ok |
| gold | demand_cached | - | 2.390 | - | - | exact |
| gold | demand_direct | - | 2.860 | - | - | exact |
