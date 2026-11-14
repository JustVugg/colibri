Note: `tre` below is dev fd7ed68a with three changes merged, this one (the two banks), the short chunk's routed copy and the host pool with the residents' RAM copies: a local commit, a9863af1, made only for this measure. With two banks the routed copy goes out with its layer's bank (`cc_fetch_at`).

# bench: misura-tre

- bench 8965f2c3fdef; engine qwen36; Linux (Intel(R) Xeon(R) CPU @ 2.00GHz, Tesla T4); 2026-11-13 22:38
- model q36; text promessi_sposi.txt (sha256 f53e0d80cb2d4492); lengths 64,512,64; gen 32; teacher-forced
- runs: a warm-up of v0 (does not count), then 4 per version, the order turning at every round (ABBA); before every run: wait until cool (at most 900 s), the card under 70 °C, then 60 s of rest
- v0 dev: qwen36-bench (commit fd7ed68afa1f88706eae08ef172a5bc02c42bd77; harness qwen36_bench.c ca1bd9cc3d24; CUDA); env: COLI_CUDA=1 COLI_GPUS=0 CUDA_EXPERT_GB=auto OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close OMP_DISPLAY_AFFINITY=TRUE OMP_AFFINITY_FORMAT=omp-thread:%n:cpus:%A
- v1 tre: qwen36-bench (commit a9863af1f9221f265e14f7e7abe17aa3a4340e4f; harness qwen36_bench.c ca1bd9cc3d24; CUDA); env: COLI_CUDA=1 COLI_GPUS=0 CUDA_EXPERT_GB=auto OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close OMP_DISPLAY_AFFINITY=TRUE OMP_AFFINITY_FORMAT=omp-thread:%n:cpus:%A

## v0 dev: 4 runs (median, min–max)

| tokens | new | prompt tok/s | first step ms | decode tok/s | experts read from the file | reads' s | MB to the card | nll | top1 | fingerprint |
|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 64 | 88.86 (88.70–88.94) | 61 (60–62) | 43.024 (42.233–43.128) | 0 | 0.00 | 6351.9 | 2.6927 | 0.469 | 4e0d1505a6060a16 |
| 512 | 512 | 518.25 (514.13–522.15) | 24 (24–24) | 49.386 (49.144–49.490) | 0 | 0.00 | 6361.8 | 2.9621 | 0.469 | e82105bcd0bd1efa |
| 64 after 512 | 64 | 101.97 (98.24–103.58) | 20 (20–20) | 51.582 (51.210–51.853) | 0 | 0.00 | 646.0 | 2.6907 | 0.469 | 9d8c5feaaca51a44 |

peak memory 20045 MiB, model ready in 19.79 s (medians)

decode by steps (step 1 gives the second new word; medians of 4 runs):

| tokens | steps | experts read from the file | decode tok/s | reads' s | MB to the card |
|---|---|---|---|---|---|
| 64 | 2 to 32 | 0 | 43.024 | 0.000 | 209.5 |
| 512 | 2 to 32 | 0 | 49.386 | 0.000 | 209.5 |
| 64 after 512 | 2 to 32 | 0 | 51.582 | 0.000 | 209.5 |

## v1 tre: 4 runs (median, min–max)

| tokens | new | prompt tok/s | first step ms | decode tok/s | experts read from the file | reads' s | MB to the card | nll | top1 | fingerprint |
|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 64 | 181.68 (181.52–182.15) | 61 (61–61) | 43.224 (42.796–43.485) | 0 | 0.00 | 2192.3 | 2.6927 | 0.469 | 4e0d1505a6060a16 |
| 512 | 512 | 730.44 (730.36–730.91) | 24 (23–24) | 49.128 (48.849–49.331) | 0 | 0.00 | 6361.8 | 2.9621 | 0.469 | e82105bcd0bd1efa |
| 64 after 512 | 64 | 220.20 (219.76–221.01) | 20 (20–21) | 51.670 (51.396–51.980) | 0 | 0.00 | 1737.8 | 2.6959 | 0.469 | 54bb171c568c376e |

peak memory 20091 MiB, model ready in 22.01 s (medians)

decode by steps (step 1 gives the second new word; medians of 4 runs):

| tokens | steps | experts read from the file | decode tok/s | reads' s | MB to the card |
|---|---|---|---|---|---|
| 64 | 2 to 32 | 0 | 43.224 | 0.000 | 209.5 |
| 512 | 2 to 32 | 0 | 49.128 | 0.000 | 209.5 |
| 64 after 512 | 2 to 32 | 0 | 51.670 | 0.000 | 209.5 |

## v1 tre against v0 dev (medians of 4 and 4 runs; positive = faster)

| tokens | prompt | first step | decode | results |
|---|---|---|---|---|
| 64 | +104.4%, every run faster | -0.7%, ranges overlap | +0.5%, ranges overlap | same results; nll 2.6927 → 2.6927 (+0.00%), top1 0.469 → 0.469 |
| 512 | +40.9%, every run faster | +0.2%, ranges overlap | -0.5%, ranges overlap | same results; nll 2.9621 → 2.9621 (+0.00%), top1 0.469 → 0.469 |
| 64 after 512 | +116.0%, every run faster | -4.2%, every run slower | +0.2%, ranges overlap | logits differ; nll 2.6907 → 2.6959 (+0.19%), top1 0.469 → 0.469 |

| tokens | steps | experts read from the file | decode | reads' s | MB to the card |
|---|---|---|---|---|---|
| 64 | 2 to 32 | 0 → 0 | +0.5%, ranges overlap | 0.000 → 0.000 s | 209.5 → 209.5 |
| 512 | 2 to 32 | 0 → 0 | -0.5%, ranges overlap | 0.000 → 0.000 s | 209.5 → 209.5 |
| 64 after 512 | 2 to 32 | 0 → 0 | +0.2%, ranges overlap | 0.000 → 0.000 s | 209.5 → 209.5 |

warm-up (does not count): 64 tokens: prompt 88.69 tok/s, decode 42.910 tok/s; 512 tokens: prompt 513.25 tok/s, decode 49.271 tok/s; 64 tokens: prompt 103.79 tok/s, decode 51.578 tok/s
backend: v0 dev CUDA; v1 tre CUDA
runs not cool: none
the card while running (samples with a kernel running; median, min–max of the runs): v0 dev SM 1575 (1575–1590) MHz, under the power cap 43 (25–57) % of the samples, up to 56 °C, 7 (7–8) samples; v1 tre SM 1560 (1560–1575) MHz, under the power cap 38 (38–38) % of the samples, up to 55 °C, 8 (8–8) samples
threads: v0 dev 4 on 2 cores (CPUs 0,4,1,5); v1 tre 4 on 2 cores (CPUs 0,4,1,5)
pages written to swap during the measurement: 0, 0 of them in the runs that count
