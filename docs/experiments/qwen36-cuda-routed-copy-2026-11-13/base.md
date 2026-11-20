# bench: misura-p28

- bench 8965f2c3fdef; engine qwen36; Linux (Intel(R) Xeon(R) CPU @ 2.00GHz, Tesla T4); 2026-11-13 20:57
- model q36; text promessi_sposi.txt (sha256 f53e0d80cb2d4492); lengths 64,512,64; gen 32; teacher-forced
- runs: a warm-up of v0 (does not count), then 4 per version, the order turning at every round (ABBA); before every run: wait until cool (at most 900 s), the card under 70 °C, then 60 s of rest
- v0 dev: qwen36-bench (commit fd7ed68afa1f88706eae08ef172a5bc02c42bd77; harness qwen36_bench.c ca1bd9cc3d24; CUDA); env: COLI_CUDA=1 COLI_GPUS=0 CUDA_EXPERT_GB=auto OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close OMP_DISPLAY_AFFINITY=TRUE OMP_AFFINITY_FORMAT=omp-thread:%n:cpus:%A
- v1 p28: qwen36-bench (commit 4306f437ec97afd0fcd45166620c6a0175e0a5ed; harness qwen36_bench.c ca1bd9cc3d24; CUDA); env: COLI_CUDA=1 COLI_GPUS=0 CUDA_EXPERT_GB=auto OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close OMP_DISPLAY_AFFINITY=TRUE OMP_AFFINITY_FORMAT=omp-thread:%n:cpus:%A

## v0 dev: 4 runs (median, min–max)

| tokens | new | prompt tok/s | first step ms | decode tok/s | experts read from the file | reads' s | MB to the card | nll | top1 | fingerprint |
|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 64 | 88.03 (88.02–88.09) | 60 (60–62) | 42.723 (42.088–43.040) | 0 | 0.00 | 6351.9 | 2.6927 | 0.469 | 4e0d1505a6060a16 |
| 512 | 512 | 513.59 (513.19–516.76) | 24 (23–25) | 48.930 (48.626–49.756) | 0 | 0.00 | 6361.8 | 2.9621 | 0.469 | e82105bcd0bd1efa |
| 64 after 512 | 64 | 102.39 (100.94–102.76) | 19 (19–20) | 51.723 (51.482–51.864) | 0 | 0.00 | 646.0 | 2.6907 | 0.469 | 9d8c5feaaca51a44 |

peak memory 20035 MiB, model ready in 19.66 s (medians)

decode by steps (step 1 gives the second new word; medians of 4 runs):

| tokens | steps | experts read from the file | decode tok/s | reads' s | MB to the card |
|---|---|---|---|---|---|
| 64 | 2 to 32 | 0 | 42.723 | 0.000 | 209.5 |
| 512 | 2 to 32 | 0 | 48.930 | 0.000 | 209.5 |
| 64 after 512 | 2 to 32 | 0 | 51.723 | 0.000 | 209.5 |

## v1 p28: 4 runs (median, min–max)

| tokens | new | prompt tok/s | first step ms | decode tok/s | experts read from the file | reads' s | MB to the card | nll | top1 | fingerprint |
|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 64 | 180.75 (180.51–181.73) | 61 (60–62) | 42.809 (42.685–42.991) | 0 | 0.00 | 2192.3 | 2.6927 | 0.469 | 4e0d1505a6060a16 |
| 512 | 512 | 514.82 (507.80–518.98) | 23 (23–24) | 49.511 (49.111–49.712) | 0 | 0.00 | 6361.8 | 2.9621 | 0.469 | e82105bcd0bd1efa |
| 64 after 512 | 64 | 106.98 (105.35–107.52) | 20 (19–20) | 51.794 (51.621–51.884) | 0 | 0.00 | 391.1 | 2.6907 | 0.469 | 9d8c5feaaca51a44 |

peak memory 20035 MiB, model ready in 19.66 s (medians)

decode by steps (step 1 gives the second new word; medians of 4 runs):

| tokens | steps | experts read from the file | decode tok/s | reads' s | MB to the card |
|---|---|---|---|---|---|
| 64 | 2 to 32 | 0 | 42.809 | 0.000 | 209.5 |
| 512 | 2 to 32 | 0 | 49.510 | 0.000 | 209.5 |
| 64 after 512 | 2 to 32 | 0 | 51.794 | 0.000 | 209.5 |

## v1 p28 against v0 dev (medians of 4 and 4 runs; positive = faster)

| tokens | prompt | first step | decode | results |
|---|---|---|---|---|
| 64 | +105.3%, every run faster | -0.9%, ranges overlap | +0.2%, ranges overlap | same results; nll 2.6927 → 2.6927 (+0.00%), top1 0.469 → 0.469 |
| 512 | +0.2%, ranges overlap | +1.7%, ranges overlap | +1.2%, ranges overlap | same results; nll 2.9621 → 2.9621 (+0.00%), top1 0.469 → 0.469 |
| 64 after 512 | +4.5%, every run faster | -0.8%, ranges overlap | +0.1%, ranges overlap | same results; nll 2.6907 → 2.6907 (+0.00%), top1 0.469 → 0.469 |

| tokens | steps | experts read from the file | decode | reads' s | MB to the card |
|---|---|---|---|---|---|
| 64 | 2 to 32 | 0 → 0 | +0.2%, ranges overlap | 0.000 → 0.000 s | 209.5 → 209.5 |
| 512 | 2 to 32 | 0 → 0 | +1.2%, ranges overlap | 0.000 → 0.000 s | 209.5 → 209.5 |
| 64 after 512 | 2 to 32 | 0 → 0 | +0.1%, ranges overlap | 0.000 → 0.000 s | 209.5 → 209.5 |

warm-up (does not count): 64 tokens: prompt 87.99 tok/s, decode 42.937 tok/s; 512 tokens: prompt 513.91 tok/s, decode 49.740 tok/s; 64 tokens: prompt 102.37 tok/s, decode 51.694 tok/s
backend: v0 dev CUDA; v1 p28 CUDA
runs not cool: none
the card while running (samples with a kernel running; median, min–max of the runs): v0 dev SM 1582 (1575–1590) MHz, under the power cap 36 (29–43) % of the samples, up to 55 °C, 7 (7–7) samples; v1 p28 SM 1590 (1575–1590) MHz, under the power cap 29 (29–43) % of the samples, up to 54 °C, 7 (7–7) samples
threads: v0 dev 4 on 2 cores (CPUs 0,4,1,5); v1 p28 4 on 2 cores (CPUs 0,4,1,5)
pages written to swap during the measurement: 0, 0 of them in the runs that count
