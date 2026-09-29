# OLMoE tier A/B benchmark: `phase1_cap8`

- date: 2026-09-29T12:08:01
- engine: `c/olmoe` (argv `8 8`), SNAP: `/Users/khalid/models/olmoe_merged`
- serve env: `SERVE=1 CTX=2048`; config env passed through as-is
- reps: 3; warmup: 1 discarded serve run(s) of `baseline`; per-run timeout: 600s
- primary metric: decode-only tok/s = 1000 / mean(DATA frame delta), the 199 deltas after the first DATA
- interleaving: rep-major (`for rep in 1..3: for config in configs`), so page-cache/thermal drift cancels
- JSONL: `c/bench_tier/ab_phase1_cap8.jsonl`

| config | decode tok/s (med) | P50 ms | P90 ms | P99 ms | hit% | disk ms/step | SHA-256(16) | standalone exact |
|---|---|---|---|---|---|---|---|---|
| baseline | 9.572 | 101.4 | 123.9 | 172.6 | 27.8 | 68.1 | 88e4d132afc16539 | 200/200 OK |
| pilot2 | 9.232 | 104.9 | 124.7 | 162.7 | 31.1 | 72.9 | 88e4d132afc16539 | 200/200 OK |
| phase1 | 9.242 | 105.8 | 126.3 | 152.6 | 27.8 | 70.8 | 88e4d132afc16539 | 200/200 OK |
| pilot2_phase1 | 7.952 | 114.7 | 146.5 | 311.8 | 31.1 | 81.9 | 88e4d132afc16539 | 200/200 OK |

## Standalone golden check (`./olmoe 8 8 bench_tier/ref200.json`)

| config | match | hit% | tok/s | wall s |
|---|---|---|---|---|
| baseline | 200/200 | 27.7 | 7.97 | 25.4 |
| pilot2 | 200/200 | 31.0 | 7.42 | 27.3 |
| phase1 | 200/200 | 27.7 | 7.41 | 27.3 |
| pilot2_phase1 | 200/200 | 31.1 | 6.16 | 32.8 |

## Exact commands

Harness invocation:

```
python3 c/bench_tier/bench_ab.py --label phase1_cap8 --reps 3 --cap 8 --configs 'baseline:PILOT=0|HOT=0|RECENT_RING=0|RECENT_BLOOM=0,pilot2:PILOT=2|HOT=0|RECENT_RING=0|RECENT_BLOOM=0,phase1:PILOT=0|HOT=0|RECENT_RING=1|RECENT_BLOOM=1,pilot2_phase1:PILOT=2|HOT=0|RECENT_RING=1|RECENT_BLOOM=1'
```

Before every run (serve and standalone): `rm -f /Users/khalid/models/olmoe_merged/hot_pinned.bin`

Serve run (once per config per rep; cwd `c`):

```
# baseline
SNAP=/Users/khalid/models/olmoe_merged CTX=2048 SERVE=1 HOT=0 PILOT=0 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 8 8
stdin: SUBMIT bench 0 430 200 0.0 1.0 + payload c/bench_tier/prompt.txt (430 bytes) + LF
# pilot2
SNAP=/Users/khalid/models/olmoe_merged CTX=2048 SERVE=1 HOT=0 PILOT=2 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 8 8
stdin: SUBMIT bench 0 430 200 0.0 1.0 + payload c/bench_tier/prompt.txt (430 bytes) + LF
# phase1
SNAP=/Users/khalid/models/olmoe_merged CTX=2048 SERVE=1 HOT=0 PILOT=0 RECENT_BLOOM=1 RECENT_RING=1 ./olmoe 8 8
stdin: SUBMIT bench 0 430 200 0.0 1.0 + payload c/bench_tier/prompt.txt (430 bytes) + LF
# pilot2_phase1
SNAP=/Users/khalid/models/olmoe_merged CTX=2048 SERVE=1 HOT=0 PILOT=2 RECENT_BLOOM=1 RECENT_RING=1 ./olmoe 8 8
stdin: SUBMIT bench 0 430 200 0.0 1.0 + payload c/bench_tier/prompt.txt (430 bytes) + LF
```

Standalone golden check (once per config, after the serve matrix):

```
# baseline
SNAP=/Users/khalid/models/olmoe_merged HOT=0 PILOT=0 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 8 8 bench_tier/ref200.json
# pilot2
SNAP=/Users/khalid/models/olmoe_merged HOT=0 PILOT=2 RECENT_BLOOM=0 RECENT_RING=0 ./olmoe 8 8 bench_tier/ref200.json
# phase1
SNAP=/Users/khalid/models/olmoe_merged HOT=0 PILOT=0 RECENT_BLOOM=1 RECENT_RING=1 ./olmoe 8 8 bench_tier/ref200.json
# pilot2_phase1
SNAP=/Users/khalid/models/olmoe_merged HOT=0 PILOT=2 RECENT_BLOOM=1 RECENT_RING=1 ./olmoe 8 8 bench_tier/ref200.json
```

## Per-run detail

| run | config | rep | decode tok/s | n DATA | SHA-256(16) | status |
|---|---|---|---|---|---|---|
| serve | baseline | 1 | 9.572 | 200 | 88e4d132afc16539 | ok |
| serve | pilot2 | 1 | 9.761 | 200 | 88e4d132afc16539 | ok |
| serve | phase1 | 1 | 9.719 | 200 | 88e4d132afc16539 | ok |
| serve | pilot2_phase1 | 1 | 9.336 | 200 | 88e4d132afc16539 | ok |
| serve | baseline | 2 | 9.926 | 200 | 88e4d132afc16539 | ok |
| serve | pilot2 | 2 | 9.232 | 200 | 88e4d132afc16539 | ok |
| serve | phase1 | 2 | 9.242 | 200 | 88e4d132afc16539 | ok |
| serve | pilot2_phase1 | 2 | 7.952 | 200 | 88e4d132afc16539 | ok |
| serve | baseline | 3 | 4.687 | 200 | 88e4d132afc16539 | ok |
| serve | pilot2 | 3 | 5.649 | 200 | 88e4d132afc16539 | ok |
| serve | phase1 | 3 | 1.188 | 200 | 88e4d132afc16539 | ok |
| serve | pilot2_phase1 | 3 | 2.788 | 200 | 88e4d132afc16539 | ok |
| gold | baseline | - | 7.970 | - | - | exact |
| gold | pilot2 | - | 7.420 | - | - | exact |
| gold | phase1 | - | 7.410 | - | - | exact |
| gold | pilot2_phase1 | - | 6.160 | - | - | exact |
