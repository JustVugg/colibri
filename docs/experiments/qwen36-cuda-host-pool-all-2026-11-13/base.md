# base.md: the host pool holds the residents' RAM copies too, on a Tesla T4

Two measures on one Colab T4 high-RAM VM (Intel Xeon @ 2.00 GHz, 4 cores / 8 threads, 51 GiB; Tesla T4 15360 MiB,
driver 580.82.07, CUDA 13.0), 13 November 2026, Qwen3.6-35B-A3B int4-g64 (6605 of the 10240 experts on the card),
prompts of 64, 512 and 64 tokens in one process, 32 words after each, teacher-forced, a warm-up and 4 + 4 runs in ABBA
order. The bench's summaries follow as printed (their headings one level down). The labels in them are:

- Measure A, this change alone on dev: v0 "dev" is dev fd7ed68a; v1 "p33" is this change's code as local commit
  93d20516 on dev fd7ed68a.
- Measure B, this change on the routed-copy commit it sits on: v0 "p28" is dev plus that commit as local commit
  4306f437; v1 "p2833" is the same plus this change as local commit 5ab88509.

The manifest.json of this folder is measure B, the 64-token prompt after the 512 (the third row of each run's CSV).
The costs (ready time, peak memory, the page-locked RAM and the engine's report lines) are in correctness.txt.

## Measure A: this change alone on dev

### bench: misura-p33

- bench 8965f2c3fdef; engine qwen36; Linux (Intel(R) Xeon(R) CPU @ 2.00GHz, Tesla T4); 2026-11-13 21:10
- model q36; text promessi_sposi.txt (sha256 f53e0d80cb2d4492); lengths 64,512,64; gen 32; teacher-forced
- runs: a warm-up of v0 (does not count), then 4 per version, the order turning at every round (ABBA); before every run: wait until cool (at most 900 s), the card under 70 °C, then 60 s of rest
- v0 dev: qwen36-bench (commit fd7ed68afa1f88706eae08ef172a5bc02c42bd77; harness qwen36_bench.c ca1bd9cc3d24; CUDA); env: COLI_CUDA=1 COLI_GPUS=0 CUDA_EXPERT_GB=auto OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close OMP_DISPLAY_AFFINITY=TRUE OMP_AFFINITY_FORMAT=omp-thread:%n:cpus:%A
- v1 p33: qwen36-bench (commit 93d205168571ddd345b712dde602384a97beefce; harness qwen36_bench.c ca1bd9cc3d24; CUDA); env: COLI_CUDA=1 COLI_GPUS=0 CUDA_EXPERT_GB=auto OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close OMP_DISPLAY_AFFINITY=TRUE OMP_AFFINITY_FORMAT=omp-thread:%n:cpus:%A

#### v0 dev: 4 runs (median, min–max)

| tokens | new | prompt tok/s | first step ms | decode tok/s | experts read from the file | reads' s | MB to the card | nll | top1 | fingerprint |
|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 64 | 88.16 (88.00–88.82) | 60 (60–61) | 43.117 (42.885–43.217) | 0 | 0.00 | 6351.9 | 2.6927 | 0.469 | 4e0d1505a6060a16 |
| 512 | 512 | 511.77 (510.17–515.19) | 24 (23–24) | 49.725 (49.618–49.949) | 0 | 0.00 | 6361.8 | 2.9621 | 0.469 | e82105bcd0bd1efa |
| 64 after 512 | 64 | 103.58 (103.24–104.96) | 20 (19–20) | 51.792 (51.559–52.014) | 0 | 0.00 | 646.0 | 2.6907 | 0.469 | 9d8c5feaaca51a44 |

peak memory 20040 MiB, model ready in 19.59 s (medians)

decode by steps (step 1 gives the second new word; medians of 4 runs):

| tokens | steps | experts read from the file | decode tok/s | reads' s | MB to the card |
|---|---|---|---|---|---|
| 64 | 2 to 32 | 0 | 43.117 | 0.000 | 209.5 |
| 512 | 2 to 32 | 0 | 49.725 | 0.000 | 209.5 |
| 64 after 512 | 2 to 32 | 0 | 51.792 | 0.000 | 209.5 |

#### v1 p33: 4 runs (median, min–max)

| tokens | new | prompt tok/s | first step ms | decode tok/s | experts read from the file | reads' s | MB to the card | nll | top1 | fingerprint |
|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 64 | 88.29 (87.50–88.71) | 60 (60–64) | 43.404 (43.175–43.634) | 0 | 0.00 | 6351.9 | 2.6927 | 0.469 | 4e0d1505a6060a16 |
| 512 | 512 | 555.26 (543.58–558.40) | 23 (23–23) | 49.672 (49.526–49.850) | 0 | 0.00 | 6361.8 | 2.9621 | 0.469 | e82105bcd0bd1efa |
| 64 after 512 | 64 | 89.94 (89.48–90.40) | 20 (20–20) | 51.809 (50.789–52.077) | 0 | 0.00 | 6358.1 | 2.6959 | 0.469 | 54bb171c568c376e |

peak memory 20091 MiB, model ready in 21.68 s (medians)

decode by steps (step 1 gives the second new word; medians of 4 runs):

| tokens | steps | experts read from the file | decode tok/s | reads' s | MB to the card |
|---|---|---|---|---|---|
| 64 | 2 to 32 | 0 | 43.405 | 0.000 | 209.5 |
| 512 | 2 to 32 | 0 | 49.672 | 0.000 | 209.5 |
| 64 after 512 | 2 to 32 | 0 | 51.809 | 0.000 | 209.5 |

#### v1 p33 against v0 dev (medians of 4 and 4 runs; positive = faster)

| tokens | prompt | first step | decode | results |
|---|---|---|---|---|
| 64 | +0.2%, ranges overlap | +0.2%, ranges overlap | +0.7%, ranges overlap | same results; nll 2.6927 → 2.6927 (+0.00%), top1 0.469 → 0.469 |
| 512 | +8.5%, every run faster | +3.2%, every run faster | -0.1%, ranges overlap | same results; nll 2.9621 → 2.9621 (+0.00%), top1 0.469 → 0.469 |
| 64 after 512 | -13.2%, every run slower | -2.3%, every run slower | +0.0%, ranges overlap | logits differ; nll 2.6907 → 2.6959 (+0.19%), top1 0.469 → 0.469 |

| tokens | steps | experts read from the file | decode | reads' s | MB to the card |
|---|---|---|---|---|---|
| 64 | 2 to 32 | 0 → 0 | +0.7%, ranges overlap | 0.000 → 0.000 s | 209.5 → 209.5 |
| 512 | 2 to 32 | 0 → 0 | -0.1%, ranges overlap | 0.000 → 0.000 s | 209.5 → 209.5 |
| 64 after 512 | 2 to 32 | 0 → 0 | +0.0%, ranges overlap | 0.000 → 0.000 s | 209.5 → 209.5 |

warm-up (does not count): 64 tokens: prompt 87.96 tok/s, decode 43.185 tok/s; 512 tokens: prompt 512.44 tok/s, decode 49.469 tok/s; 64 tokens: prompt 102.78 tok/s, decode 51.937 tok/s
backend: v0 dev CUDA; v1 p33 CUDA
runs not cool: none
the card while running (samples with a kernel running; median, min–max of the runs): v0 dev SM 1590 (1560–1590) MHz, under the power cap 29 (29–43) % of the samples, up to 56 °C, 7 (7–7) samples; v1 p33 SM 1590 (1590–1590) MHz, under the power cap 22 (11–22) % of the samples, up to 56 °C, 9 (9–9) samples
threads: v0 dev 4 on 2 cores (CPUs 0,4,1,5); v1 p33 4 on 2 cores (CPUs 0,4,1,5)
pages written to swap during the measurement: 0, 0 of them in the runs that count

## Measure B: on top of the routed-copy commit

### bench: misura-p2833

- bench 8965f2c3fdef; engine qwen36; Linux (Intel(R) Xeon(R) CPU @ 2.00GHz, Tesla T4); 2026-11-13 21:58
- model q36; text promessi_sposi.txt (sha256 f53e0d80cb2d4492); lengths 64,512,64; gen 32; teacher-forced
- runs: a warm-up of v0 (does not count), then 4 per version, the order turning at every round (ABBA); before every run: wait until cool (at most 900 s), the card under 70 °C, then 60 s of rest
- v0 p28: qwen36-bench (commit 4306f437ec97afd0fcd45166620c6a0175e0a5ed; harness qwen36_bench.c ca1bd9cc3d24; CUDA); env: COLI_CUDA=1 COLI_GPUS=0 CUDA_EXPERT_GB=auto OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close OMP_DISPLAY_AFFINITY=TRUE OMP_AFFINITY_FORMAT=omp-thread:%n:cpus:%A
- v1 p2833: qwen36-bench (commit 5ab885097c02ccd1fbae47253521b7e6942d95cf; harness qwen36_bench.c ca1bd9cc3d24; CUDA); env: COLI_CUDA=1 COLI_GPUS=0 CUDA_EXPERT_GB=auto OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close OMP_DISPLAY_AFFINITY=TRUE OMP_AFFINITY_FORMAT=omp-thread:%n:cpus:%A

#### v0 p28: 4 runs (median, min–max)

| tokens | new | prompt tok/s | first step ms | decode tok/s | experts read from the file | reads' s | MB to the card | nll | top1 | fingerprint |
|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 64 | 181.85 (181.63–182.02) | 61 (60–62) | 43.020 (42.553–43.129) | 0 | 0.00 | 2192.3 | 2.6927 | 0.469 | 4e0d1505a6060a16 |
| 512 | 512 | 517.89 (511.64–520.15) | 24 (24–24) | 49.206 (49.151–49.295) | 0 | 0.00 | 6361.8 | 2.9621 | 0.469 | e82105bcd0bd1efa |
| 64 after 512 | 64 | 104.79 (104.40–106.76) | 20 (20–20) | 51.713 (51.557–51.762) | 0 | 0.00 | 391.1 | 2.6907 | 0.469 | 9d8c5feaaca51a44 |

peak memory 20045 MiB, model ready in 19.73 s (medians)

decode by steps (step 1 gives the second new word; medians of 4 runs):

| tokens | steps | experts read from the file | decode tok/s | reads' s | MB to the card |
|---|---|---|---|---|---|
| 64 | 2 to 32 | 0 | 43.021 | 0.000 | 209.5 |
| 512 | 2 to 32 | 0 | 49.205 | 0.000 | 209.5 |
| 64 after 512 | 2 to 32 | 0 | 51.713 | 0.000 | 209.5 |

#### v1 p2833: 4 runs (median, min–max)

| tokens | new | prompt tok/s | first step ms | decode tok/s | experts read from the file | reads' s | MB to the card | nll | top1 | fingerprint |
|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 64 | 182.00 (181.74–182.24) | 61 (60–62) | 42.284 (41.752–42.716) | 0 | 0.00 | 2192.3 | 2.6927 | 0.469 | 4e0d1505a6060a16 |
| 512 | 512 | 556.81 (555.92–566.65) | 24 (23–24) | 48.891 (48.505–49.319) | 0 | 0.00 | 6361.8 | 2.9621 | 0.469 | e82105bcd0bd1efa |
| 64 after 512 | 64 | 222.12 (221.13–222.95) | 20 (20–20) | 51.320 (51.182–51.546) | 0 | 0.00 | 1737.8 | 2.6959 | 0.469 | 54bb171c568c376e |

peak memory 20091 MiB, model ready in 21.96 s (medians)

decode by steps (step 1 gives the second new word; medians of 4 runs):

| tokens | steps | experts read from the file | decode tok/s | reads' s | MB to the card |
|---|---|---|---|---|---|
| 64 | 2 to 32 | 0 | 42.284 | 0.000 | 209.5 |
| 512 | 2 to 32 | 0 | 48.891 | 0.000 | 209.5 |
| 64 after 512 | 2 to 32 | 0 | 51.319 | 0.000 | 209.5 |

#### v1 p2833 against v0 p28 (medians of 4 and 4 runs; positive = faster)

| tokens | prompt | first step | decode | results |
|---|---|---|---|---|
| 64 | +0.1%, ranges overlap | +0.5%, ranges overlap | -1.7%, ranges overlap | same results; nll 2.6927 → 2.6927 (+0.00%), top1 0.469 → 0.469 |
| 512 | +7.5%, every run faster | +0.0%, ranges overlap | -0.6%, ranges overlap | same results; nll 2.9621 → 2.9621 (+0.00%), top1 0.469 → 0.469 |
| 64 after 512 | +112.0%, every run faster | -3.7%, every run slower | -0.8%, every run slower | logits differ; nll 2.6907 → 2.6959 (+0.19%), top1 0.469 → 0.469 |

| tokens | steps | experts read from the file | decode | reads' s | MB to the card |
|---|---|---|---|---|---|
| 64 | 2 to 32 | 0 → 0 | -1.7%, ranges overlap | 0.000 → 0.000 s | 209.5 → 209.5 |
| 512 | 2 to 32 | 0 → 0 | -0.6%, ranges overlap | 0.000 → 0.000 s | 209.5 → 209.5 |
| 64 after 512 | 2 to 32 | 0 → 0 | -0.8%, every run slower | 0.000 → 0.000 s | 209.5 → 209.5 |

warm-up (does not count): 64 tokens: prompt 181.08 tok/s, decode 40.816 tok/s; 512 tokens: prompt 514.28 tok/s, decode 48.059 tok/s; 64 tokens: prompt 101.30 tok/s, decode 50.743 tok/s
backend: v0 p28 CUDA; v1 p2833 CUDA
runs not cool: none
the card while running (samples with a kernel running; median, min–max of the runs): v0 p28 SM 1590 (1575–1590) MHz, under the power cap 29 (14–43) % of the samples, up to 55 °C, 7 (7–7) samples; v1 p2833 SM 1575 (1568–1575) MHz, under the power cap 38 (38–38) % of the samples, up to 55 °C, 8 (8–8) samples
threads: v0 p28 4 on 2 cores (CPUs 0,4,1,5); v1 p2833 4 on 2 cores (CPUs 0,4,1,5)
pages written to swap during the measurement: 0, 0 of them in the runs that count
