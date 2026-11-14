# bench: misura-p24

- bench 8965f2c3fdef; engine qwen36; Linux (Intel(R) Xeon(R) CPU @ 2.00GHz, Tesla T4); 2026-11-13 20:44
- model q36; text promessi_sposi.txt (sha256 f53e0d80cb2d4492); lengths 64,512,64; gen 32; teacher-forced
- runs: a warm-up of v0 (does not count), then 4 per version, the order turning at every round (ABBA); before every run: wait until cool (at most 900 s), the card under 70 °C, then 60 s of rest
- v0 dev: qwen36-bench (commit fd7ed68afa1f88706eae08ef172a5bc02c42bd77; harness qwen36_bench.c ca1bd9cc3d24; CUDA); env: COLI_CUDA=1 COLI_GPUS=0 CUDA_EXPERT_GB=auto OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close OMP_DISPLAY_AFFINITY=TRUE OMP_AFFINITY_FORMAT=omp-thread:%n:cpus:%A
- v1 p24: qwen36-bench (commit 0a6fcca3bf78dad106cfa786f9566f6e1a830489; harness qwen36_bench.c ca1bd9cc3d24; CUDA); env: COLI_CUDA=1 COLI_GPUS=0 CUDA_EXPERT_GB=auto OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close OMP_DISPLAY_AFFINITY=TRUE OMP_AFFINITY_FORMAT=omp-thread:%n:cpus:%A

## v0 dev: 4 runs (median, min–max)

| tokens | new | prompt tok/s | first step ms | decode tok/s | experts read from the file | reads' s | MB to the card | nll | top1 | fingerprint |
|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 64 | 88.12 (87.97–88.16) | 62 (60–62) | 42.477 (42.270–42.934) | 0 | 0.00 | 6351.9 | 2.6927 | 0.469 | 4e0d1505a6060a16 |
| 512 | 512 | 515.89 (507.21–516.33) | 24 (24–24) | 49.226 (48.894–49.478) | 0 | 0.00 | 6361.8 | 2.9621 | 0.469 | e82105bcd0bd1efa |
| 64 after 512 | 64 | 101.96 (100.28–102.76) | 20 (20–20) | 51.566 (50.379–51.708) | 0 | 0.00 | 646.0 | 2.6907 | 0.469 | 9d8c5feaaca51a44 |

peak memory 20040 MiB, model ready in 19.66 s (medians)

decode by steps (step 1 gives the second new word; medians of 4 runs):

| tokens | steps | experts read from the file | decode tok/s | reads' s | MB to the card |
|---|---|---|---|---|---|
| 64 | 2 to 32 | 0 | 42.476 | 0.000 | 209.5 |
| 512 | 2 to 32 | 0 | 49.226 | 0.000 | 209.5 |
| 64 after 512 | 2 to 32 | 0 | 51.566 | 0.000 | 209.5 |

## v1 p24: 4 runs (median, min–max)

| tokens | new | prompt tok/s | first step ms | decode tok/s | experts read from the file | reads' s | MB to the card | nll | top1 | fingerprint |
|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 64 | 98.56 (98.21–98.69) | 74 (70–79) | 42.371 (41.893–43.464) | 0 | 0.00 | 6351.9 | 2.6927 | 0.469 | 4e0d1505a6060a16 |
| 512 | 512 | 640.77 (635.02–652.49) | 24 (23–24) | 49.341 (49.081–49.826) | 0 | 0.00 | 6361.8 | 2.9621 | 0.469 | e82105bcd0bd1efa |
| 64 after 512 | 64 | 101.58 (100.04–102.73) | 20 (20–20) | 51.764 (51.526–52.178) | 0 | 0.00 | 646.0 | 2.6907 | 0.469 | 9d8c5feaaca51a44 |

peak memory 20040 MiB, model ready in 19.78 s (medians)

decode by steps (step 1 gives the second new word; medians of 4 runs):

| tokens | steps | experts read from the file | decode tok/s | reads' s | MB to the card |
|---|---|---|---|---|---|
| 64 | 2 to 32 | 0 | 42.371 | 0.000 | 209.5 |
| 512 | 2 to 32 | 0 | 49.341 | 0.000 | 209.5 |
| 64 after 512 | 2 to 32 | 0 | 51.765 | 0.000 | 209.5 |

## v1 p24 against v0 dev (medians of 4 and 4 runs; positive = faster)

| tokens | prompt | first step | decode | results |
|---|---|---|---|---|
| 64 | +11.8%, every run faster | -16.8%, every run slower | -0.2%, ranges overlap | same results; nll 2.6927 → 2.6927 (+0.00%), top1 0.469 → 0.469 |
| 512 | +24.2%, every run faster | +0.8%, ranges overlap | +0.2%, ranges overlap | same results; nll 2.9621 → 2.9621 (+0.00%), top1 0.469 → 0.469 |
| 64 after 512 | -0.4%, ranges overlap | +1.0%, ranges overlap | +0.4%, ranges overlap | same results; nll 2.6907 → 2.6907 (+0.00%), top1 0.469 → 0.469 |

| tokens | steps | experts read from the file | decode | reads' s | MB to the card |
|---|---|---|---|---|---|
| 64 | 2 to 32 | 0 → 0 | -0.2%, ranges overlap | 0.000 → 0.000 s | 209.5 → 209.5 |
| 512 | 2 to 32 | 0 → 0 | +0.2%, ranges overlap | 0.000 → 0.000 s | 209.5 → 209.5 |
| 64 after 512 | 2 to 32 | 0 → 0 | +0.4%, ranges overlap | 0.000 → 0.000 s | 209.5 → 209.5 |

warm-up (does not count): 64 tokens: prompt 88.14 tok/s, decode 43.457 tok/s; 512 tokens: prompt 515.79 tok/s, decode 48.780 tok/s; 64 tokens: prompt 102.88 tok/s, decode 51.760 tok/s
backend: v0 dev CUDA; v1 p24 CUDA
runs not cool: none
the card while running (samples with a kernel running; median, min–max of the runs): v0 dev SM 1582 (1560–1590) MHz, under the power cap 39 (25–57) % of the samples, up to 57 °C, 8 (7–8) samples; v1 p24 SM 1590 (1575–1590) MHz, under the power cap 29 (14–43) % of the samples, up to 55 °C, 7 (7–7) samples
threads: v0 dev 4 on 2 cores (CPUs 0,4,1,5); v1 p24 4 on 2 cores (CPUs 0,4,1,5)
pages written to swap during the measurement: 0, 0 of them in the runs that count
