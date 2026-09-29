# OLMoE tier A/B benchmark: `final_cap24_quiet`

- date: 2026-09-29T22:05:39
- engine: `c/olmoe` (argv `24 8`), SNAP: `/Users/khalid/models/olmoe_merged`
- serve env: `SERVE=1 CTX=2048`; config env passed through as-is
- reps: 3; warmup: 1 discarded serve run(s) of `baseline`; per-run timeout: 600s
- primary metric: decode-only tok/s = 1000 / mean(DATA frame delta), the 199 deltas after the first DATA
- interleaving: rep-major (`for rep in 1..3: for config in configs`), so page-cache/thermal drift cancels
- JSONL: `c/bench_tier/ab_final_cap24_quiet.jsonl`

| config | decode tok/s (med) | P50 ms | P90 ms | P99 ms | hit% | disk ms/step | SHA-256(16) | standalone exact |
|---|---|---|---|---|---|---|---|---|
| baseline | 1.529 | 530.9 | 1447.7 | 2869.3 | 68.4 | 271.8 | 88e4d132afc16539 | 200/200 OK |
| best_demand_group | 4.506 | 183.1 | 309.3 | 1117.1 | 64.9 | 105.3 | 88e4d132afc16539 | 200/200 OK |
| p3_full | 5.761 | 149.9 | 225.7 | 545.8 | 70.5 | 103.8 | 88e4d132afc16539 | 200/200 OK |
| pilot2_markov2 | 3.236 | 150.3 | 731.3 | 1245.7 | 70.4 | 145.4 | 88e4d132afc16539 | 200/200 OK |

## Standalone golden check (`./olmoe 24 8 bench_tier/ref200.json`)

| config | match | hit% | tok/s | wall s |
|---|---|---|---|---|
| baseline | 200/200 | 68.4 | 1.41 | 142.3 |
| best_demand_group | 200/200 | 64.8 | 4.36 | 46.5 |
| p3_full | 200/200 | 70.5 | 1.81 | 110.9 |
| pilot2_markov2 | 200/200 | 70.3 | 4.37 | 46.6 |

## Exact commands

Harness invocation:

```
python3 c/bench_tier/bench_ab.py --label final_cap24_quiet --reps 3 --cap 24 --configs 'baseline:PILOT=0|HOT=0|DEMAND_POLICY=lru,best_demand_group:PILOT=0|HOT=0|DEMAND_POLICY=lfru|GROUP_EVICT=1|EXPERT_DIRECT=1,p3_full:PILOT=2|HOT=0|DEMAND_POLICY=auto|GROUP_EVICT=1|EXPERT_DIRECT=1|PILOT_WORKERS=4,pilot2_markov2:PILOT=2|HOT=0|DEMAND_POLICY=auto|MARKOV=1|MARKOV_TOP=2|GROUP_EVICT=1|EXPERT_DIRECT=1'
```

Before every run (serve and standalone): `rm -f /Users/khalid/models/olmoe_merged/hot_pinned.bin`

Serve run (once per config per rep; cwd `c`):

```
# baseline
SNAP=/Users/khalid/models/olmoe_merged CTX=2048 SERVE=1 DEMAND_POLICY=lru HOT=0 PILOT=0 ./olmoe 24 8
stdin: SUBMIT bench 0 430 200 0.0 1.0 + payload c/bench_tier/prompt.txt (430 bytes) + LF
# best_demand_group
SNAP=/Users/khalid/models/olmoe_merged CTX=2048 SERVE=1 DEMAND_POLICY=lfru EXPERT_DIRECT=1 GROUP_EVICT=1 HOT=0 PILOT=0 ./olmoe 24 8
stdin: SUBMIT bench 0 430 200 0.0 1.0 + payload c/bench_tier/prompt.txt (430 bytes) + LF
# p3_full
SNAP=/Users/khalid/models/olmoe_merged CTX=2048 SERVE=1 DEMAND_POLICY=auto EXPERT_DIRECT=1 GROUP_EVICT=1 HOT=0 PILOT=2 PILOT_WORKERS=4 ./olmoe 24 8
stdin: SUBMIT bench 0 430 200 0.0 1.0 + payload c/bench_tier/prompt.txt (430 bytes) + LF
# pilot2_markov2
SNAP=/Users/khalid/models/olmoe_merged CTX=2048 SERVE=1 DEMAND_POLICY=auto EXPERT_DIRECT=1 GROUP_EVICT=1 HOT=0 MARKOV=1 MARKOV_TOP=2 PILOT=2 ./olmoe 24 8
stdin: SUBMIT bench 0 430 200 0.0 1.0 + payload c/bench_tier/prompt.txt (430 bytes) + LF
```

Standalone golden check (once per config, after the serve matrix):

```
# baseline
SNAP=/Users/khalid/models/olmoe_merged DEMAND_POLICY=lru HOT=0 PILOT=0 ./olmoe 24 8 bench_tier/ref200.json
# best_demand_group
SNAP=/Users/khalid/models/olmoe_merged DEMAND_POLICY=lfru EXPERT_DIRECT=1 GROUP_EVICT=1 HOT=0 PILOT=0 ./olmoe 24 8 bench_tier/ref200.json
# p3_full
SNAP=/Users/khalid/models/olmoe_merged DEMAND_POLICY=auto EXPERT_DIRECT=1 GROUP_EVICT=1 HOT=0 PILOT=2 PILOT_WORKERS=4 ./olmoe 24 8 bench_tier/ref200.json
# pilot2_markov2
SNAP=/Users/khalid/models/olmoe_merged DEMAND_POLICY=auto EXPERT_DIRECT=1 GROUP_EVICT=1 HOT=0 MARKOV=1 MARKOV_TOP=2 PILOT=2 ./olmoe 24 8 bench_tier/ref200.json
```

## Per-run detail

| run | config | rep | decode tok/s | n DATA | SHA-256(16) | status |
|---|---|---|---|---|---|---|
| serve | baseline | 1 | 0.952 | 200 | 88e4d132afc16539 | ok |
| serve | best_demand_group | 1 | 4.470 | 200 | 88e4d132afc16539 | ok |
| serve | p3_full | 1 | 3.882 | 200 | 88e4d132afc16539 | ok |
| serve | pilot2_markov2 | 1 | 3.236 | 200 | 88e4d132afc16539 | ok |
| serve | baseline | 2 | 1.540 | 200 | 88e4d132afc16539 | ok |
| serve | best_demand_group | 2 | 4.506 | 200 | 88e4d132afc16539 | ok |
| serve | p3_full | 2 | 6.782 | 200 | 88e4d132afc16539 | ok |
| serve | pilot2_markov2 | 2 | 2.002 | 200 | 88e4d132afc16539 | ok |
| serve | baseline | 3 | 1.529 | 200 | 88e4d132afc16539 | ok |
| serve | best_demand_group | 3 | 5.066 | 200 | 88e4d132afc16539 | ok |
| serve | p3_full | 3 | 5.761 | 200 | 88e4d132afc16539 | ok |
| serve | pilot2_markov2 | 3 | 5.585 | 200 | 88e4d132afc16539 | ok |
| gold | baseline | - | 1.410 | - | - | exact |
| gold | best_demand_group | - | 4.360 | - | - | exact |
| gold | p3_full | - | 1.810 | - | - | exact |
| gold | pilot2_markov2 | - | 4.370 | - | - | exact |
