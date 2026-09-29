# OLMoE tier A/B benchmark: `phase2_cap8`

- date: 2026-09-29T12:51:59
- engine: `c/olmoe` (argv `8 8`), SNAP: `/Users/khalid/models/olmoe_merged`
- serve env: `SERVE=1 CTX=2048`; config env passed through as-is
- reps: 3; warmup: 1 discarded serve run(s) of `baseline`; per-run timeout: 600s
- primary metric: decode-only tok/s = 1000 / mean(DATA frame delta), the 199 deltas after the first DATA
- interleaving: rep-major (`for rep in 1..3: for config in configs`), so page-cache/thermal drift cancels
- JSONL: `c/bench_tier/ab_phase2_cap8.jsonl`

| config | decode tok/s (med) | P50 ms | P90 ms | P99 ms | hit% | disk ms/step | SHA-256(16) | standalone exact |
|---|---|---|---|---|---|---|---|---|
| baseline | 11.186 | 90.3 | 98.0 | 103.9 | 27.8 | 59.4 | 88e4d132afc16539 | 200/200 OK |
| pilot2 | 10.499 | 94.8 | 103.5 | 112.2 | 31.1 | 62.5 | 88e4d132afc16539 | 200/200 OK |
| pilot2_markov2 | 10.375 | 95.9 | 104.5 | 113.6 | 28.2 | 69.5 | 88e4d132afc16539 | 200/200 OK |
| pilot2_markov1 | 10.334 | 96.2 | 107.7 | 118.0 | 29.4 | 67.6 | 88e4d132afc16539 | 200/200 OK |

## Standalone golden check (`./olmoe 8 8 bench_tier/ref200.json`)

| config | match | hit% | tok/s | wall s |
|---|---|---|---|---|
| baseline | 200/200 | 27.7 | 7.94 | 25.5 |
| pilot2 | 200/200 | 31.0 | 8.12 | 24.9 |
| pilot2_markov2 | 200/200 | 28.3 | 7.95 | 25.4 |
| pilot2_markov1 | 200/200 | 29.3 | 8.15 | 24.8 |

## Exact commands

Harness invocation:

```
python3 c/bench_tier/bench_ab.py --label phase2_cap8 --reps 3 --cap 8 --configs 'baseline:PILOT=0|HOT=0|RECENT_RING=0|RECENT_BLOOM=0|MARKOV=0,pilot2:PILOT=2|HOT=0|RECENT_RING=0|RECENT_BLOOM=0|MARKOV=0,pilot2_markov2:PILOT=2|HOT=0|RECENT_RING=0|RECENT_BLOOM=0|MARKOV=1|MARKOV_TOP=2,pilot2_markov1:PILOT=2|HOT=0|RECENT_RING=0|RECENT_BLOOM=0|MARKOV=1|MARKOV_TOP=1'
```

Before every run (serve and standalone): `rm -f /Users/khalid/models/olmoe_merged/hot_pinned.bin`

Serve run (once per config per rep; cwd `c`):

```
# baseline
SNAP=/Users/khalid/models/olmoe_merged CTX=2048 SERVE=1 HOT=0 MARKOV=0 PILOT=0 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 8 8
stdin: SUBMIT bench 0 430 200 0.0 1.0 + payload c/bench_tier/prompt.txt (430 bytes) + LF
# pilot2
SNAP=/Users/khalid/models/olmoe_merged CTX=2048 SERVE=1 HOT=0 MARKOV=0 PILOT=2 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 8 8
stdin: SUBMIT bench 0 430 200 0.0 1.0 + payload c/bench_tier/prompt.txt (430 bytes) + LF
# pilot2_markov2
SNAP=/Users/khalid/models/olmoe_merged CTX=2048 SERVE=1 HOT=0 MARKOV=1 MARKOV_TOP=2 PILOT=2 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 8 8
stdin: SUBMIT bench 0 430 200 0.0 1.0 + payload c/bench_tier/prompt.txt (430 bytes) + LF
# pilot2_markov1
SNAP=/Users/khalid/models/olmoe_merged CTX=2048 SERVE=1 HOT=0 MARKOV=1 MARKOV_TOP=1 PILOT=2 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 8 8
stdin: SUBMIT bench 0 430 200 0.0 1.0 + payload c/bench_tier/prompt.txt (430 bytes) + LF
```

Standalone golden check (once per config, after the serve matrix):

```
# baseline
SNAP=/Users/khalid/models/olmoe_merged HOT=0 MARKOV=0 PILOT=0 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 8 8 bench_tier/ref200.json
# pilot2
SNAP=/Users/khalid/models/olmoe_merged HOT=0 MARKOV=0 PILOT=2 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 8 8 bench_tier/ref200.json
# pilot2_markov2
SNAP=/Users/khalid/models/olmoe_merged HOT=0 MARKOV=1 MARKOV_TOP=2 PILOT=2 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 8 8 bench_tier/ref200.json
# pilot2_markov1
SNAP=/Users/khalid/models/olmoe_merged HOT=0 MARKOV=1 MARKOV_TOP=1 PILOT=2 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 8 8 bench_tier/ref200.json
```

## Per-run detail

| run | config | rep | decode tok/s | n DATA | SHA-256(16) | status |
|---|---|---|---|---|---|---|
| serve | baseline | 1 | 11.248 | 200 | 88e4d132afc16539 | ok |
| serve | pilot2 | 1 | 10.499 | 200 | 88e4d132afc16539 | ok |
| serve | pilot2_markov2 | 1 | 10.379 | 200 | 88e4d132afc16539 | ok |
| serve | pilot2_markov1 | 1 | 10.334 | 200 | 88e4d132afc16539 | ok |
| serve | baseline | 2 | 11.186 | 200 | 88e4d132afc16539 | ok |
| serve | pilot2 | 2 | 10.744 | 200 | 88e4d132afc16539 | ok |
| serve | pilot2_markov2 | 2 | 10.375 | 200 | 88e4d132afc16539 | ok |
| serve | pilot2_markov1 | 2 | 10.626 | 200 | 88e4d132afc16539 | ok |
| serve | baseline | 3 | 10.614 | 200 | 88e4d132afc16539 | ok |
| serve | pilot2 | 3 | 10.425 | 200 | 88e4d132afc16539 | ok |
| serve | pilot2_markov2 | 3 | 10.183 | 200 | 88e4d132afc16539 | ok |
| serve | pilot2_markov1 | 3 | 10.120 | 200 | 88e4d132afc16539 | ok |
| gold | baseline | - | 7.940 | - | - | exact |
| gold | pilot2 | - | 8.120 | - | - | exact |
| gold | pilot2_markov2 | - | 7.950 | - | - | exact |
| gold | pilot2_markov1 | - | 8.150 | - | - | exact |
