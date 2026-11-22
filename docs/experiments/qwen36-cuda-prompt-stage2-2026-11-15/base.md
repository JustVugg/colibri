# bench: misura-p24

- bench 8965f2c3fdef; engine qwen36; Linux (Intel(R) Xeon(R) CPU @ 2.00GHz, Tesla T4); 2026-11-15 20:39
- model q36; text promessi_sposi.txt (sha256 f53e0d80cb2d4492); lengths 64,512,64; gen 32; teacher-forced
- runs: a warm-up of v0 (does not count), then 4 per version, the order turning at every round (ABBA); before every run: wait until cool (at most 900 s), the card under 70 °C, then 60 s of rest
- v0 base: qwen36-bench (commit d5bc67005fb6a11a2212354e92def3ab8f7d9ded; harness qwen36_bench.c ca1bd9cc3d24; CUDA); env: COLI_CUDA=1 COLI_GPUS=0 CUDA_EXPERT_GB=auto OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close OMP_DISPLAY_AFFINITY=TRUE OMP_AFFINITY_FORMAT=omp-thread:%n:cpus:%A
- v1 p24: qwen36-bench (commit a143ab341bd280439711009f737a4d5ff733b69e; harness qwen36_bench.c ca1bd9cc3d24; CUDA); env: COLI_CUDA=1 COLI_GPUS=0 CUDA_EXPERT_GB=auto OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close OMP_DISPLAY_AFFINITY=TRUE OMP_AFFINITY_FORMAT=omp-thread:%n:cpus:%A

## v0 base: 4 runs (median, min–max)

| tokens | new | prompt tok/s | first step ms | decode tok/s | experts read from the file | reads' s | MB to the card | nll | top1 | fingerprint |
|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 64 | 182.96 (182.05–183.90) | 62 (61–63) | 42.087 (41.939–42.727) | 0 | 0.00 | 2192.3 | 2.6927 | 0.469 | 4e0d1505a6060a16 |
| 512 | 512 | 561.89 (555.16–565.32) | 24 (24–25) | 48.468 (48.248–49.317) | 0 | 0.00 | 6361.8 | 2.9621 | 0.469 | e82105bcd0bd1efa |
| 64 after 512 | 64 | 223.49 (220.90–224.58) | 20 (20–21) | 50.988 (50.687–51.573) | 0 | 0.00 | 1737.8 | 2.6959 | 0.469 | 54bb171c568c376e |

peak memory 20081 MiB, model ready in 22.11 s (medians)

decode by steps (step 1 gives the second new word; medians of 4 runs):

| tokens | steps | experts read from the file | decode tok/s | reads' s | MB to the card |
|---|---|---|---|---|---|
| 64 | 2 to 32 | 0 | 42.087 | 0.000 | 209.5 |
| 512 | 2 to 32 | 0 | 48.468 | 0.000 | 209.5 |
| 64 after 512 | 2 to 32 | 0 | 50.988 | 0.000 | 209.5 |

## v1 p24: 4 runs (median, min–max)

| tokens | new | prompt tok/s | first step ms | decode tok/s | experts read from the file | reads' s | MB to the card | nll | top1 | fingerprint |
|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 64 | 182.81 (182.44–183.04) | 63 (61–63) | 42.047 (41.493–42.399) | 0 | 0.00 | 2192.3 | 2.6927 | 0.469 | 4e0d1505a6060a16 |
| 512 | 512 | 738.05 (737.01–741.12) | 25 (24–25) | 48.181 (47.987–48.503) | 0 | 0.00 | 6361.8 | 2.9621 | 0.469 | e82105bcd0bd1efa |
| 64 after 512 | 64 | 220.70 (219.78–222.11) | 21 (20–21) | 50.740 (50.519–50.775) | 0 | 0.00 | 1737.8 | 2.6959 | 0.469 | 54bb171c568c376e |

peak memory 20091 MiB, model ready in 22.18 s (medians)

decode by steps (step 1 gives the second new word; medians of 4 runs):

| tokens | steps | experts read from the file | decode tok/s | reads' s | MB to the card |
|---|---|---|---|---|---|
| 64 | 2 to 32 | 0 | 42.047 | 0.000 | 209.5 |
| 512 | 2 to 32 | 0 | 48.180 | 0.000 | 209.5 |
| 64 after 512 | 2 to 32 | 0 | 50.741 | 0.000 | 209.5 |

## v1 p24 against v0 base (medians of 4 and 4 runs; positive = faster)

| tokens | prompt | first step | decode | results |
|---|---|---|---|---|
| 64 | -0.1%, ranges overlap | -1.5%, ranges overlap | -0.1%, ranges overlap | same results; nll 2.6927 → 2.6927 (+0.00%), top1 0.469 → 0.469 |
| 512 | +31.4%, every run faster | -1.0%, ranges overlap | -0.6%, ranges overlap | same results; nll 2.9621 → 2.9621 (+0.00%), top1 0.469 → 0.469 |
| 64 after 512 | -1.2%, ranges overlap | -0.5%, ranges overlap | -0.5%, ranges overlap | same results; nll 2.6959 → 2.6959 (+0.00%), top1 0.469 → 0.469 |

| tokens | steps | experts read from the file | decode | reads' s | MB to the card |
|---|---|---|---|---|---|
| 64 | 2 to 32 | 0 → 0 | -0.1%, ranges overlap | 0.000 → 0.000 s | 209.5 → 209.5 |
| 512 | 2 to 32 | 0 → 0 | -0.6%, ranges overlap | 0.000 → 0.000 s | 209.5 → 209.5 |
| 64 after 512 | 2 to 32 | 0 → 0 | -0.5%, ranges overlap | 0.000 → 0.000 s | 209.5 → 209.5 |

warm-up (does not count): 64 tokens: prompt 183.89 tok/s, decode 41.680 tok/s; 512 tokens: prompt 561.36 tok/s, decode 48.654 tok/s; 64 tokens: prompt 225.38 tok/s, decode 50.746 tok/s
backend: v0 base CUDA; v1 p24 CUDA
runs not cool: none
the card while running (samples with a kernel running; median, min–max of the runs): v0 base SM 1560 (1552–1560) MHz, under the power cap 38 (38–38) % of the samples, up to 68 °C, 8 (8–8) samples; v1 p24 SM 1541 (1522–1560) MHz, under the power cap 38 (38–38) % of the samples, up to 67 °C, 8 (8–8) samples
threads: v0 base 4 on 2 cores (CPUs 0,4,1,5); v1 p24 4 on 2 cores (CPUs 0,4,1,5)
pages written to swap during the measurement: 0, 0 of them in the runs that count
