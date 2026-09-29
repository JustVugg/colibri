# OLMoE tier A/B benchmark: `final_cap8`

- date: 2026-09-29T15:24:54
- engine: `c/olmoe` (argv `8 8`), SNAP: `/Users/khalid/models/olmoe_merged`
- serve env: `SERVE=1 CTX=2048`; config env passed through as-is
- reps: 5; warmup: 1 discarded serve run(s) of `baseline`; per-run timeout: 600s
- primary metric: decode-only tok/s = 1000 / mean(DATA frame delta), the 199 deltas after the first DATA
- interleaving: rep-major (`for rep in 1..5: for config in configs`), so page-cache/thermal drift cancels
- JSONL: `c/bench_tier/ab_final_cap8.jsonl`

| config | decode tok/s (med) | P50 ms | P90 ms | P99 ms | hit% | disk ms/step | SHA-256(16) | standalone exact |
|---|---|---|---|---|---|---|---|---|
| baseline | 3.474 | 258.4 | 440.3 | 819.4 | 27.8 | 203.7 | 88e4d132afc16539 | 200/200 OK |
| p1_temporal | 3.437 | 247.4 | 467.8 | 896.9 | 27.8 | 194.3 | 88e4d132afc16539 | 200/200 OK |
| p2_markov | 3.392 | 262.3 | 427.9 | 744.4 | 27.8 | 207.7 | 88e4d132afc16539 | 200/200 OK |
| p3_full | 4.845 | 196.9 | 253.4 | 344.1 | 33.3 | 156.9 | 88e4d132afc16539 | 200/200 OK |
| best_demand | 4.406 | 211.4 | 298.6 | 465.0 | 33.3 | 149.0 | 88e4d132afc16539 | 200/200 OK |
| best_demand_group | 5.062 | 190.3 | 244.5 | 378.9 | 33.3 | 152.5 | 88e4d132afc16539 | 200/200 OK |

## Standalone golden check (`./olmoe 8 8 bench_tier/ref200.json`)

| config | match | hit% | tok/s | wall s |
|---|---|---|---|---|
| baseline | 200/200 | 27.7 | 2.53 | 79.6 |
| p1_temporal | 200/200 | 27.7 | 2.88 | 70.0 |
| p2_markov | 200/200 | 27.7 | 3.01 | 67.2 |
| p3_full | 200/200 | 33.3 | 3.75 | 54.0 |
| best_demand | 200/200 | 33.3 | 2.94 | 68.4 |
| best_demand_group | 200/200 | 33.3 | 2.48 | 81.2 |

## Exact commands

Harness invocation:

```
python3 c/bench_tier/bench_ab.py --label final_cap8 --reps 5 --cap 8 --configs 'baseline:PILOT=0|HOT=0|DEMAND_POLICY=lru|RECENT_RING=0|RECENT_BLOOM=0|MARKOV=0|GROUP_EVICT=0|EXPERT_DIRECT=0,p1_temporal:PILOT=0|HOT=0|DEMAND_POLICY=lru|RECENT_RING=1|RECENT_BLOOM=1|MARKOV=0|GROUP_EVICT=0|EXPERT_DIRECT=0,p2_markov:PILOT=0|HOT=0|DEMAND_POLICY=lru|RECENT_RING=1|RECENT_BLOOM=1|MARKOV=1|MARKOV_TOP=1|GROUP_EVICT=0|EXPERT_DIRECT=0,p3_full:PILOT=2|HOT=0|DEMAND_POLICY=lfru|RECENT_RING=1|RECENT_BLOOM=1|MARKOV=1|MARKOV_TOP=1|GROUP_EVICT=1|EXPERT_DIRECT=1|PILOT_WORKERS=4,best_demand:PILOT=0|HOT=0|DEMAND_POLICY=lfru|RECENT_RING=0|RECENT_BLOOM=0|MARKOV=0|GROUP_EVICT=0|EXPERT_DIRECT=0,best_demand_group:PILOT=0|HOT=0|DEMAND_POLICY=lfru|RECENT_RING=0|RECENT_BLOOM=0|MARKOV=0|GROUP_EVICT=1|EXPERT_DIRECT=1'
```

Before every run (serve and standalone): `rm -f /Users/khalid/models/olmoe_merged/hot_pinned.bin`

Serve run (once per config per rep; cwd `c`):

```
# baseline
SNAP=/Users/khalid/models/olmoe_merged CTX=2048 SERVE=1 DEMAND_POLICY=lru EXPERT_DIRECT=0 GROUP_EVICT=0 HOT=0 MARKOV=0 PILOT=0 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 8 8
stdin: SUBMIT bench 0 430 200 0.0 1.0 + payload c/bench_tier/prompt.txt (430 bytes) + LF
# p1_temporal
SNAP=/Users/khalid/models/olmoe_merged CTX=2048 SERVE=1 DEMAND_POLICY=lru EXPERT_DIRECT=0 GROUP_EVICT=0 HOT=0 MARKOV=0 PILOT=0 RECENT_BLOOM=1 RECENT_RING=1 ./olmoe 8 8
stdin: SUBMIT bench 0 430 200 0.0 1.0 + payload c/bench_tier/prompt.txt (430 bytes) + LF
# p2_markov
SNAP=/Users/khalid/models/olmoe_merged CTX=2048 SERVE=1 DEMAND_POLICY=lru EXPERT_DIRECT=0 GROUP_EVICT=0 HOT=0 MARKOV=1 MARKOV_TOP=1 PILOT=0 RECENT_BLOOM=1 RECENT_RING=1 ./olmoe 8 8
stdin: SUBMIT bench 0 430 200 0.0 1.0 + payload c/bench_tier/prompt.txt (430 bytes) + LF
# p3_full
SNAP=/Users/khalid/models/olmoe_merged CTX=2048 SERVE=1 DEMAND_POLICY=lfru EXPERT_DIRECT=1 GROUP_EVICT=1 HOT=0 MARKOV=1 MARKOV_TOP=1 PILOT=2 PILOT_WORKERS=4 RECENT_BLOOM=1 RECENT_RING=1 ./olmoe 8 8
stdin: SUBMIT bench 0 430 200 0.0 1.0 + payload c/bench_tier/prompt.txt (430 bytes) + LF
# best_demand
SNAP=/Users/khalid/models/olmoe_merged CTX=2048 SERVE=1 DEMAND_POLICY=lfru EXPERT_DIRECT=0 GROUP_EVICT=0 HOT=0 MARKOV=0 PILOT=0 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 8 8
stdin: SUBMIT bench 0 430 200 0.0 1.0 + payload c/bench_tier/prompt.txt (430 bytes) + LF
# best_demand_group
SNAP=/Users/khalid/models/olmoe_merged CTX=2048 SERVE=1 DEMAND_POLICY=lfru EXPERT_DIRECT=1 GROUP_EVICT=1 HOT=0 MARKOV=0 PILOT=0 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 8 8
stdin: SUBMIT bench 0 430 200 0.0 1.0 + payload c/bench_tier/prompt.txt (430 bytes) + LF
```

Standalone golden check (once per config, after the serve matrix):

```
# baseline
SNAP=/Users/khalid/models/olmoe_merged DEMAND_POLICY=lru EXPERT_DIRECT=0 GROUP_EVICT=0 HOT=0 MARKOV=0 PILOT=0 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 8 8 bench_tier/ref200.json
# p1_temporal
SNAP=/Users/khalid/models/olmoe_merged DEMAND_POLICY=lru EXPERT_DIRECT=0 GROUP_EVICT=0 HOT=0 MARKOV=0 PILOT=0 RECENT_BLOOM=1 RECENT_RING=1 ./olmoe 8 8 bench_tier/ref200.json
# p2_markov
SNAP=/Users/khalid/models/olmoe_merged DEMAND_POLICY=lru EXPERT_DIRECT=0 GROUP_EVICT=0 HOT=0 MARKOV=1 MARKOV_TOP=1 PILOT=0 RECENT_BLOOM=1 RECENT_RING=1 ./olmoe 8 8 bench_tier/ref200.json
# p3_full
SNAP=/Users/khalid/models/olmoe_merged DEMAND_POLICY=lfru EXPERT_DIRECT=1 GROUP_EVICT=1 HOT=0 MARKOV=1 MARKOV_TOP=1 PILOT=2 PILOT_WORKERS=4 RECENT_BLOOM=1 RECENT_RING=1 ./olmoe 8 8 bench_tier/ref200.json
# best_demand
SNAP=/Users/khalid/models/olmoe_merged DEMAND_POLICY=lfru EXPERT_DIRECT=0 GROUP_EVICT=0 HOT=0 MARKOV=0 PILOT=0 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 8 8 bench_tier/ref200.json
# best_demand_group
SNAP=/Users/khalid/models/olmoe_merged DEMAND_POLICY=lfru EXPERT_DIRECT=1 GROUP_EVICT=1 HOT=0 MARKOV=0 PILOT=0 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 8 8 bench_tier/ref200.json
```

## Per-run detail

| run | config | rep | decode tok/s | n DATA | SHA-256(16) | status |
|---|---|---|---|---|---|---|
| serve | baseline | 1 | 1.710 | 200 | 88e4d132afc16539 | ok |
| serve | p1_temporal | 1 | 3.471 | 200 | 88e4d132afc16539 | ok |
| serve | p2_markov | 1 | 4.645 | 200 | 88e4d132afc16539 | ok |
| serve | p3_full | 1 | 4.845 | 200 | 88e4d132afc16539 | ok |
| serve | best_demand | 1 | 4.774 | 200 | 88e4d132afc16539 | ok |
| serve | best_demand_group | 1 | 4.964 | 200 | 88e4d132afc16539 | ok |
| serve | baseline | 2 | 3.474 | 200 | 88e4d132afc16539 | ok |
| serve | p1_temporal | 2 | 3.755 | 200 | 88e4d132afc16539 | ok |
| serve | p2_markov | 2 | 3.345 | 200 | 88e4d132afc16539 | ok |
| serve | p3_full | 2 | 4.731 | 200 | 88e4d132afc16539 | ok |
| serve | best_demand | 2 | 4.406 | 200 | 88e4d132afc16539 | ok |
| serve | best_demand_group | 2 | 5.285 | 200 | 88e4d132afc16539 | ok |
| serve | baseline | 3 | 3.792 | 200 | 88e4d132afc16539 | ok |
| serve | p1_temporal | 3 | 3.113 | 200 | 88e4d132afc16539 | ok |
| serve | p2_markov | 3 | 3.392 | 200 | 88e4d132afc16539 | ok |
| serve | p3_full | 3 | 4.875 | 200 | 88e4d132afc16539 | ok |
| serve | best_demand | 3 | 4.389 | 200 | 88e4d132afc16539 | ok |
| serve | best_demand_group | 3 | 5.396 | 200 | 88e4d132afc16539 | ok |
| serve | baseline | 4 | 2.356 | 200 | 88e4d132afc16539 | ok |
| serve | p1_temporal | 4 | 3.114 | 200 | 88e4d132afc16539 | ok |
| serve | p2_markov | 4 | 4.462 | 200 | 88e4d132afc16539 | ok |
| serve | p3_full | 4 | 4.514 | 200 | 88e4d132afc16539 | ok |
| serve | best_demand | 4 | 2.356 | 200 | 88e4d132afc16539 | ok |
| serve | best_demand_group | 4 | 4.710 | 200 | 88e4d132afc16539 | ok |
| serve | baseline | 5 | 3.651 | 200 | 88e4d132afc16539 | ok |
| serve | p1_temporal | 5 | 3.437 | 200 | 88e4d132afc16539 | ok |
| serve | p2_markov | 5 | 2.839 | 200 | 88e4d132afc16539 | ok |
| serve | p3_full | 5 | 4.935 | 200 | 88e4d132afc16539 | ok |
| serve | best_demand | 5 | 5.102 | 200 | 88e4d132afc16539 | ok |
| serve | best_demand_group | 5 | 5.062 | 200 | 88e4d132afc16539 | ok |
| gold | baseline | - | 2.530 | - | - | exact |
| gold | p1_temporal | - | 2.880 | - | - | exact |
| gold | p2_markov | - | 3.010 | - | - | exact |
| gold | p3_full | - | 3.750 | - | - | exact |
| gold | best_demand | - | 2.940 | - | - | exact |
| gold | best_demand_group | - | 2.480 | - | - | exact |
