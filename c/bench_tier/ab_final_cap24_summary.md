# OLMoE tier A/B benchmark: `final_cap24`

- date: 2026-09-29T16:11:35
- engine: `c/olmoe` (argv `24 8`), SNAP: `/Users/khalid/models/olmoe_merged`
- serve env: `SERVE=1 CTX=2048`; config env passed through as-is
- reps: 2; warmup: 1 discarded serve run(s) of `baseline`; per-run timeout: 600s
- primary metric: decode-only tok/s = 1000 / mean(DATA frame delta), the 199 deltas after the first DATA
- interleaving: rep-major (`for rep in 1..2: for config in configs`), so page-cache/thermal drift cancels
- JSONL: `c/bench_tier/ab_final_cap24.jsonl`

| config | decode tok/s (med) | P50 ms | P90 ms | P99 ms | hit% | disk ms/step | SHA-256(16) | standalone exact |
|---|---|---|---|---|---|---|---|---|
| baseline | 2.754 | 260.0 | 1585.4 | 3220.0 | 68.4 | 178.0 | 88e4d132afc16539 | 200/200 OK |
| auto24 | 2.882 | 617.8 | 1279.5 | 2037.1 | 68.4 | 239.5 | 88e4d132afc16539 | 200/200 OK |
| best_demand_group | 4.950 | 177.8 | 303.6 | 530.9 | 64.9 | 103.7 | 88e4d132afc16539 | 200/200 OK |
| p3_full | 1.515 | 205.0 | 1534.6 | 5917.9 | 70.5 | 253.3 | 88e4d132afc16539 | 200/200 OK |

- `best_demand_group`: 1/2 serve runs failed (timeout(frames): no data before deadline)

## Standalone golden check (`./olmoe 24 8 bench_tier/ref200.json`)

| config | match | hit% | tok/s | wall s |
|---|---|---|---|---|
| baseline | 200/200 | 68.4 | 1.86 | 108.4 |
| auto24 | 200/200 | 68.4 | 1.73 | 116.2 |
| best_demand_group | 200/200 | 64.8 | 1.49 | 134.9 |
| p3_full | 200/200 | 70.5 | 4.51 | 44.8 |

## Exact commands

Harness invocation:

```
python3 c/bench_tier/bench_ab.py --label final_cap24 --reps 2 --cap 24 --configs 'baseline:PILOT=0|HOT=0|DEMAND_POLICY=lru,auto24:PILOT=0|HOT=0|DEMAND_POLICY=auto,best_demand_group:PILOT=0|HOT=0|DEMAND_POLICY=lfru|GROUP_EVICT=1|EXPERT_DIRECT=1,p3_full:PILOT=2|HOT=0|DEMAND_POLICY=auto|GROUP_EVICT=1|EXPERT_DIRECT=1|PILOT_WORKERS=4'
```

Before every run (serve and standalone): `rm -f /Users/khalid/models/olmoe_merged/hot_pinned.bin`

Serve run (once per config per rep; cwd `c`):

```
# baseline
SNAP=/Users/khalid/models/olmoe_merged CTX=2048 SERVE=1 DEMAND_POLICY=lru HOT=0 PILOT=0 ./olmoe 24 8
stdin: SUBMIT bench 0 430 200 0.0 1.0 + payload c/bench_tier/prompt.txt (430 bytes) + LF
# auto24
SNAP=/Users/khalid/models/olmoe_merged CTX=2048 SERVE=1 DEMAND_POLICY=auto HOT=0 PILOT=0 ./olmoe 24 8
stdin: SUBMIT bench 0 430 200 0.0 1.0 + payload c/bench_tier/prompt.txt (430 bytes) + LF
# best_demand_group
SNAP=/Users/khalid/models/olmoe_merged CTX=2048 SERVE=1 DEMAND_POLICY=lfru EXPERT_DIRECT=1 GROUP_EVICT=1 HOT=0 PILOT=0 ./olmoe 24 8
stdin: SUBMIT bench 0 430 200 0.0 1.0 + payload c/bench_tier/prompt.txt (430 bytes) + LF
# p3_full
SNAP=/Users/khalid/models/olmoe_merged CTX=2048 SERVE=1 DEMAND_POLICY=auto EXPERT_DIRECT=1 GROUP_EVICT=1 HOT=0 PILOT=2 PILOT_WORKERS=4 ./olmoe 24 8
stdin: SUBMIT bench 0 430 200 0.0 1.0 + payload c/bench_tier/prompt.txt (430 bytes) + LF
```

Standalone golden check (once per config, after the serve matrix):

```
# baseline
SNAP=/Users/khalid/models/olmoe_merged DEMAND_POLICY=lru HOT=0 PILOT=0 ./olmoe 24 8 bench_tier/ref200.json
# auto24
SNAP=/Users/khalid/models/olmoe_merged DEMAND_POLICY=auto HOT=0 PILOT=0 ./olmoe 24 8 bench_tier/ref200.json
# best_demand_group
SNAP=/Users/khalid/models/olmoe_merged DEMAND_POLICY=lfru EXPERT_DIRECT=1 GROUP_EVICT=1 HOT=0 PILOT=0 ./olmoe 24 8 bench_tier/ref200.json
# p3_full
SNAP=/Users/khalid/models/olmoe_merged DEMAND_POLICY=auto EXPERT_DIRECT=1 GROUP_EVICT=1 HOT=0 PILOT=2 PILOT_WORKERS=4 ./olmoe 24 8 bench_tier/ref200.json
```

## Per-run detail

| run | config | rep | decode tok/s | n DATA | SHA-256(16) | status |
|---|---|---|---|---|---|---|
| serve | baseline | 1 | 4.258 | 200 | 88e4d132afc16539 | ok |
| serve | auto24 | 1 | 4.927 | 200 | 88e4d132afc16539 | ok |
| serve | best_demand_group | 1 | 4.950 | 200 | 88e4d132afc16539 | ok |
| serve | p3_full | 1 | 1.520 | 200 | 88e4d132afc16539 | ok |
| serve | baseline | 2 | 1.250 | 200 | 88e4d132afc16539 | ok |
| serve | auto24 | 2 | 0.837 | 200 | 88e4d132afc16539 | ok |
| serve | best_demand_group | 2 | 3.009 | 94 | d745102047dd4ed2 | timeout(frames): no data before deadline |
| serve | p3_full | 2 | 1.510 | 200 | 88e4d132afc16539 | ok |
| gold | baseline | - | 1.860 | - | - | exact |
| gold | auto24 | - | 1.730 | - | - | exact |
| gold | best_demand_group | - | 1.490 | - | - | exact |
| gold | p3_full | - | 4.510 | - | - | exact |
