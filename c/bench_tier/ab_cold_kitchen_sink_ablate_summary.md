# OLMoE tier A/B benchmark: `cold_kitchen_sink_ablate`

- date: 2026-09-30T12:46:04
- engine: `c/olmoe` (argv `8 8`), SNAP: `/Users/khalid/models/olmoe_merged`
- serve env: `SERVE=1 CTX=2048`; config env passed through as-is
- reps: 2; warmup: 1 discarded serve run(s) of `lfru_group_only`; per-run timeout: 900s
- primary metric: decode-only tok/s = 1000 / mean(DATA frame delta), the 199 deltas after the first DATA
- interleaving: rep-major (`for rep in 1..2: for config in configs`), so page-cache/thermal drift cancels
- JSONL: `c/bench_tier/ab_cold_kitchen_sink_ablate.jsonl`

| config | decode tok/s (med) | P50 ms | P90 ms | P99 ms | hit% | disk ms/step | SHA-256(16) | standalone exact |
|---|---|---|---|---|---|---|---|---|
| lfru_group_only | 4.956 | 192.3 | 244.8 | 400.8 | 33.3 | 158.1 | 88e4d132afc16539 | 200/200 OK |
| kitchen_sink | 3.931 | 218.1 | 353.7 | 1646.5 | 33.3 | 177.6 | 88e4d132afc16539 | 200/200 OK |

## Standalone golden check (`./olmoe 8 8 bench_tier/ref200.json`)

| config | match | hit% | tok/s | wall s |
|---|---|---|---|---|
| lfru_group_only | 200/200 | 33.3 | 3.42 | 59.3 |
| kitchen_sink | 200/200 | 33.3 | 3.52 | 57.5 |

## Exact commands

Harness invocation:

```
python3 c/bench_tier/bench_ab.py --label cold_kitchen_sink_ablate --reps 2 --cap 8 --timeout 900 --configs 'lfru_group_only:PILOT=0|HOT=0|DEMAND_POLICY=lfru|EXPERT_DIRECT=1|MARKOV=0|GROUP_EVICT=1|RECENT_RING=0|RECENT_BLOOM=0,kitchen_sink:PILOT=2|HOT=0|DEMAND_POLICY=lfru|EXPERT_DIRECT=1|MARKOV=1|MARKOV_TOP=2|GROUP_EVICT=1|RECENT_RING=0|RECENT_BLOOM=0'
```

Before every run (serve and standalone): `rm -f /Users/khalid/models/olmoe_merged/hot_pinned.bin`

Serve run (once per config per rep; cwd `c`):

```
# lfru_group_only
SNAP=/Users/khalid/models/olmoe_merged CTX=2048 SERVE=1 DEMAND_POLICY=lfru EXPERT_DIRECT=1 GROUP_EVICT=1 HOT=0 MARKOV=0 PILOT=0 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 8 8
stdin: SUBMIT bench 0 430 200 0.0 1.0 + payload c/bench_tier/prompt.txt (430 bytes) + LF
# kitchen_sink
SNAP=/Users/khalid/models/olmoe_merged CTX=2048 SERVE=1 DEMAND_POLICY=lfru EXPERT_DIRECT=1 GROUP_EVICT=1 HOT=0 MARKOV=1 MARKOV_TOP=2 PILOT=2 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 8 8
stdin: SUBMIT bench 0 430 200 0.0 1.0 + payload c/bench_tier/prompt.txt (430 bytes) + LF
```

Standalone golden check (once per config, after the serve matrix):

```
# lfru_group_only
SNAP=/Users/khalid/models/olmoe_merged DEMAND_POLICY=lfru EXPERT_DIRECT=1 GROUP_EVICT=1 HOT=0 MARKOV=0 PILOT=0 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 8 8 bench_tier/ref200.json
# kitchen_sink
SNAP=/Users/khalid/models/olmoe_merged DEMAND_POLICY=lfru EXPERT_DIRECT=1 GROUP_EVICT=1 HOT=0 MARKOV=1 MARKOV_TOP=2 PILOT=2 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 8 8 bench_tier/ref200.json
```

## Per-run detail

| run | config | rep | decode tok/s | n DATA | SHA-256(16) | status |
|---|---|---|---|---|---|---|
| serve | lfru_group_only | 1 | 4.985 | 200 | 88e4d132afc16539 | ok |
| serve | kitchen_sink | 1 | 4.876 | 200 | 88e4d132afc16539 | ok |
| serve | lfru_group_only | 2 | 4.927 | 200 | 88e4d132afc16539 | ok |
| serve | kitchen_sink | 2 | 2.986 | 200 | 88e4d132afc16539 | ok |
| gold | lfru_group_only | - | 3.420 | - | - | exact |
| gold | kitchen_sink | - | 3.520 | - | - | exact |
