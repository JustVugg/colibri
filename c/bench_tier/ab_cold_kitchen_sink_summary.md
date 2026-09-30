# OLMoE tier A/B benchmark: `cold_kitchen_sink`

- date: 2026-09-30T12:32:30
- engine: `c/olmoe` (argv `8 8`), SNAP: `/Users/khalid/models/olmoe_merged`
- serve env: `SERVE=1 CTX=2048`; config env passed through as-is
- reps: 3; warmup: 1 discarded serve run(s) of `demand`; per-run timeout: 900s
- primary metric: decode-only tok/s = 1000 / mean(DATA frame delta), the 199 deltas after the first DATA
- interleaving: rep-major (`for rep in 1..3: for config in configs`), so page-cache/thermal drift cancels
- JSONL: `c/bench_tier/ab_cold_kitchen_sink.jsonl`

| config | decode tok/s (med) | P50 ms | P90 ms | P99 ms | hit% | disk ms/step | SHA-256(16) | standalone exact |
|---|---|---|---|---|---|---|---|---|
| demand | 3.962 | 217.5 | 348.6 | 885.6 | 27.8 | 194.2 | 88e4d132afc16539 | 200/200 OK |
| pilot2_markov2 | 4.380 | 223.6 | 279.7 | 411.9 | 28.2 | 235.1 | 88e4d132afc16539 | 200/200 OK |
| kitchen_sink | 4.817 | 181.4 | 286.8 | 594.0 | 33.3 | 149.5 | 88e4d132afc16539 | 200/200 OK |
| kitchen_sink_w2 | 4.218 | 202.3 | 297.0 | 710.0 | 33.3 | 171.2 | 88e4d132afc16539 | 200/200 OK |

## Standalone golden check (`./olmoe 8 8 bench_tier/ref200.json`)

| config | match | hit% | tok/s | wall s |
|---|---|---|---|---|
| demand | 200/200 | 27.7 | 3.47 | 58.3 |
| pilot2_markov2 | 200/200 | 28.2 | 3.20 | 63.1 |
| kitchen_sink | 200/200 | 33.3 | 4.05 | 50.0 |
| kitchen_sink_w2 | 200/200 | 33.3 | 3.46 | 58.4 |

## Exact commands

Harness invocation:

```
python3 c/bench_tier/bench_ab.py --label cold_kitchen_sink --reps 3 --cap 8 --timeout 900 --configs 'demand:PILOT=0|HOT=0|DEMAND_POLICY=lru|EXPERT_DIRECT=1|RECENT_RING=0|RECENT_BLOOM=0|MARKOV=0|GROUP_EVICT=0,pilot2_markov2:PILOT=2|HOT=0|DEMAND_POLICY=lru|EXPERT_DIRECT=1|MARKOV=1|MARKOV_TOP=2|RECENT_RING=0|RECENT_BLOOM=0|GROUP_EVICT=0,kitchen_sink:PILOT=2|HOT=0|DEMAND_POLICY=lfru|EXPERT_DIRECT=1|MARKOV=1|MARKOV_TOP=2|GROUP_EVICT=1|RECENT_RING=0|RECENT_BLOOM=0,kitchen_sink_w2:PILOT=2|PILOT_WORKERS=2|HOT=0|DEMAND_POLICY=lfru|EXPERT_DIRECT=1|MARKOV=1|MARKOV_TOP=2|GROUP_EVICT=1|RECENT_RING=0|RECENT_BLOOM=0'
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
# kitchen_sink
SNAP=/Users/khalid/models/olmoe_merged CTX=2048 SERVE=1 DEMAND_POLICY=lfru EXPERT_DIRECT=1 GROUP_EVICT=1 HOT=0 MARKOV=1 MARKOV_TOP=2 PILOT=2 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 8 8
stdin: SUBMIT bench 0 430 200 0.0 1.0 + payload c/bench_tier/prompt.txt (430 bytes) + LF
# kitchen_sink_w2
SNAP=/Users/khalid/models/olmoe_merged CTX=2048 SERVE=1 DEMAND_POLICY=lfru EXPERT_DIRECT=1 GROUP_EVICT=1 HOT=0 MARKOV=1 MARKOV_TOP=2 PILOT=2 PILOT_WORKERS=2 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 8 8
stdin: SUBMIT bench 0 430 200 0.0 1.0 + payload c/bench_tier/prompt.txt (430 bytes) + LF
```

Standalone golden check (once per config, after the serve matrix):

```
# demand
SNAP=/Users/khalid/models/olmoe_merged DEMAND_POLICY=lru EXPERT_DIRECT=1 GROUP_EVICT=0 HOT=0 MARKOV=0 PILOT=0 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 8 8 bench_tier/ref200.json
# pilot2_markov2
SNAP=/Users/khalid/models/olmoe_merged DEMAND_POLICY=lru EXPERT_DIRECT=1 GROUP_EVICT=0 HOT=0 MARKOV=1 MARKOV_TOP=2 PILOT=2 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 8 8 bench_tier/ref200.json
# kitchen_sink
SNAP=/Users/khalid/models/olmoe_merged DEMAND_POLICY=lfru EXPERT_DIRECT=1 GROUP_EVICT=1 HOT=0 MARKOV=1 MARKOV_TOP=2 PILOT=2 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 8 8 bench_tier/ref200.json
# kitchen_sink_w2
SNAP=/Users/khalid/models/olmoe_merged DEMAND_POLICY=lfru EXPERT_DIRECT=1 GROUP_EVICT=1 HOT=0 MARKOV=1 MARKOV_TOP=2 PILOT=2 PILOT_WORKERS=2 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 8 8 bench_tier/ref200.json
```

## Per-run detail

| run | config | rep | decode tok/s | n DATA | SHA-256(16) | status |
|---|---|---|---|---|---|---|
| serve | demand | 1 | 3.130 | 200 | 88e4d132afc16539 | ok |
| serve | pilot2_markov2 | 1 | 3.410 | 200 | 88e4d132afc16539 | ok |
| serve | kitchen_sink | 1 | 3.989 | 200 | 88e4d132afc16539 | ok |
| serve | kitchen_sink_w2 | 1 | 4.218 | 200 | 88e4d132afc16539 | ok |
| serve | demand | 2 | 5.135 | 200 | 88e4d132afc16539 | ok |
| serve | pilot2_markov2 | 2 | 4.380 | 200 | 88e4d132afc16539 | ok |
| serve | kitchen_sink | 2 | 4.817 | 200 | 88e4d132afc16539 | ok |
| serve | kitchen_sink_w2 | 2 | 3.640 | 200 | 88e4d132afc16539 | ok |
| serve | demand | 3 | 3.962 | 200 | 88e4d132afc16539 | ok |
| serve | pilot2_markov2 | 3 | 4.731 | 200 | 88e4d132afc16539 | ok |
| serve | kitchen_sink | 3 | 5.187 | 200 | 88e4d132afc16539 | ok |
| serve | kitchen_sink_w2 | 3 | 4.767 | 200 | 88e4d132afc16539 | ok |
| gold | demand | - | 3.470 | - | - | exact |
| gold | pilot2_markov2 | - | 3.200 | - | - | exact |
| gold | kitchen_sink | - | 4.050 | - | - | exact |
| gold | kitchen_sink_w2 | - | 3.460 | - | - | exact |
