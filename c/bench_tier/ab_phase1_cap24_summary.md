# OLMoE tier A/B benchmark: `phase1_cap24`

- date: 2026-09-29T12:16:52
- engine: `c/olmoe` (argv `24 8`), SNAP: `/Users/khalid/models/olmoe_merged`
- serve env: `SERVE=1 CTX=2048`; config env passed through as-is
- reps: 2; warmup: 1 discarded serve run(s) of `baseline`; per-run timeout: 600s
- primary metric: decode-only tok/s = 1000 / mean(DATA frame delta), the 199 deltas after the first DATA
- interleaving: rep-major (`for rep in 1..2: for config in configs`), so page-cache/thermal drift cancels
- JSONL: `c/bench_tier/ab_phase1_cap24.jsonl`

| config | decode tok/s (med) | P50 ms | P90 ms | P99 ms | hit% | disk ms/step | SHA-256(16) | standalone exact |
|---|---|---|---|---|---|---|---|---|
| baseline | 13.070 | 73.4 | 96.2 | 121.3 | 68.4 | 32.3 | 88e4d132afc16539 | 200/200 OK |
| pilot2 | 12.532 | 76.6 | 100.2 | 128.0 | 70.9 | 34.0 | 88e4d132afc16539 | 200/200 OK |
| phase1 | 10.119 | 83.4 | 137.6 | 362.5 | 68.4 | 45.5 | 88e4d132afc16539 | 200/200 OK |
| pilot2_phase1 | 12.013 | 78.9 | 101.2 | 145.2 | 70.8 | 35.1 | 88e4d132afc16539 | 200/200 OK |

## Standalone golden check (`./olmoe 24 8 bench_tier/ref200.json`)

| config | match | hit% | tok/s | wall s |
|---|---|---|---|---|
| baseline | 200/200 | 68.4 | 9.85 | 20.7 |
| pilot2 | 200/200 | 70.8 | 10.53 | 19.3 |
| phase1 | 200/200 | 68.4 | 10.51 | 19.3 |
| pilot2_phase1 | 200/200 | 70.8 | 10.23 | 19.9 |

## Exact commands

Harness invocation:

```
python3 c/bench_tier/bench_ab.py --label phase1_cap24 --reps 2 --cap 24 --configs 'baseline:PILOT=0|HOT=0|RECENT_RING=0|RECENT_BLOOM=0,pilot2:PILOT=2|HOT=0|RECENT_RING=0|RECENT_BLOOM=0,phase1:PILOT=0|HOT=0|RECENT_RING=1|RECENT_BLOOM=1,pilot2_phase1:PILOT=2|HOT=0|RECENT_RING=1|RECENT_BLOOM=1'
```

Before every run (serve and standalone): `rm -f /Users/khalid/models/olmoe_merged/hot_pinned.bin`

Serve run (once per config per rep; cwd `c`):

```
# baseline
SNAP=/Users/khalid/models/olmoe_merged CTX=2048 SERVE=1 HOT=0 PILOT=0 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 24 8
stdin: SUBMIT bench 0 430 200 0.0 1.0 + payload c/bench_tier/prompt.txt (430 bytes) + LF
# pilot2
SNAP=/Users/khalid/models/olmoe_merged CTX=2048 SERVE=1 HOT=0 PILOT=2 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 24 8
stdin: SUBMIT bench 0 430 200 0.0 1.0 + payload c/bench_tier/prompt.txt (430 bytes) + LF
# phase1
SNAP=/Users/khalid/models/olmoe_merged CTX=2048 SERVE=1 HOT=0 PILOT=0 RECENT_BLOOM=1 RECENT_RING=1 ./olmoe 24 8
stdin: SUBMIT bench 0 430 200 0.0 1.0 + payload c/bench_tier/prompt.txt (430 bytes) + LF
# pilot2_phase1
SNAP=/Users/khalid/models/olmoe_merged CTX=2048 SERVE=1 HOT=0 PILOT=2 RECENT_BLOOM=1 RECENT_RING=1 ./olmoe 24 8
stdin: SUBMIT bench 0 430 200 0.0 1.0 + payload c/bench_tier/prompt.txt (430 bytes) + LF
```

Standalone golden check (once per config, after the serve matrix):

```
# baseline
SNAP=/Users/khalid/models/olmoe_merged HOT=0 PILOT=0 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 24 8 bench_tier/ref200.json
# pilot2
SNAP=/Users/khalid/models/olmoe_merged HOT=0 PILOT=2 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 24 8 bench_tier/ref200.json
# phase1
SNAP=/Users/khalid/models/olmoe_merged HOT=0 PILOT=0 RECENT_BLOOM=1 RECENT_RING=1 ./olmoe 24 8 bench_tier/ref200.json
# pilot2_phase1
SNAP=/Users/khalid/models/olmoe_merged HOT=0 PILOT=2 RECENT_BLOOM=1 RECENT_RING=1 ./olmoe 24 8 bench_tier/ref200.json
```

## Per-run detail

| run | config | rep | decode tok/s | n DATA | SHA-256(16) | status |
|---|---|---|---|---|---|---|
| serve | baseline | 1 | 14.095 | 200 | 88e4d132afc16539 | ok |
| serve | pilot2 | 1 | 13.527 | 200 | 88e4d132afc16539 | ok |
| serve | phase1 | 1 | 8.908 | 200 | 88e4d132afc16539 | ok |
| serve | pilot2_phase1 | 1 | 11.859 | 200 | 88e4d132afc16539 | ok |
| serve | baseline | 2 | 12.045 | 200 | 88e4d132afc16539 | ok |
| serve | pilot2 | 2 | 11.538 | 200 | 88e4d132afc16539 | ok |
| serve | phase1 | 2 | 11.330 | 200 | 88e4d132afc16539 | ok |
| serve | pilot2_phase1 | 2 | 12.167 | 200 | 88e4d132afc16539 | ok |
| gold | baseline | - | 9.850 | - | - | exact |
| gold | pilot2 | - | 10.530 | - | - | exact |
| gold | phase1 | - | 10.510 | - | - | exact |
| gold | pilot2_phase1 | - | 10.230 | - | - | exact |
