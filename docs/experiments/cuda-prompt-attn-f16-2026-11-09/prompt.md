# bench: misura2

- bench b7107cb35aee; engine qwen36; Linux (Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz, Tesla V100-SXM2-32GB); 2026-11-09 16:11
- model q36; text promessi_sposi.txt (sha256 f53e0d80cb2d4492); lengths 2048,4096,8192; gen 1; teacher-forced
- runs: a warm-up of v0 (does not count), then 4 per version, the order turning at every round (ABBA); before every run: wait until cool (at most 900 s), the card under 70 °C, then 60 s of rest
- v0 a: qwen36-bench (commit bf0b0eae719208e7603c282db280149564f7d6a8; harness qwen36_bench.c ca1bd9cc3d24; CUDA); env: COLI_CUDA=1 COLI_GPUS=0 CUDA_EXPERT_GB=auto OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close OMP_DISPLAY_AFFINITY=TRUE OMP_AFFINITY_FORMAT=omp-thread:%n:cpus:%A OMP_NUM_THREADS=14 COLI_TRUNK_PROBE=0
- v1 b: qwen36-bench (commit a37779ed052d57a1413a4e914075e2f5f420ca72; harness qwen36_bench.c ca1bd9cc3d24; CUDA); env: COLI_CUDA=1 COLI_GPUS=0 CUDA_EXPERT_GB=auto OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close OMP_DISPLAY_AFFINITY=TRUE OMP_AFFINITY_FORMAT=omp-thread:%n:cpus:%A OMP_NUM_THREADS=14 COLI_TRUNK_PROBE=0

## v0 a: 4 runs (median, min–max)

| tokens | new | prompt tok/s | first step ms | decode tok/s | experts read from the file | reads' s | MB to the card | nll | top1 | fingerprint |
|---|---|---|---|---|---|---|---|---|---|---|
| 2048 | 2048 | 2442.37 (2427.05–2449.32) | 65 (61–70) | - | 0 | 0.00 | 17.1 | 4.9818 | 0.000 | 9dafbfbea1ae2c97 |
| 4096 | 4096 | 2341.59 (2334.46–2359.81) | 12 (11–14) | - | 0 | 0.00 | 33.0 | 0.6188 | 1.000 | f1011990c3f971a6 |
| 8192 | 8192 | 2218.47 (2216.97–2228.19) | 12 (11–12) | - | 0 | 0.00 | 66.0 | 3.5177 | 0.000 | dc624137db26fe6c |

peak memory 20830 MiB, model ready in 25.64 s (medians)

decode by steps (step 1 gives the second new word; medians of 4 runs):

| tokens | steps | experts read from the file | decode tok/s | reads' s | MB to the card |
|---|---|---|---|---|---|

## v1 b: 4 runs (median, min–max)

| tokens | new | prompt tok/s | first step ms | decode tok/s | experts read from the file | reads' s | MB to the card | nll | top1 | fingerprint |
|---|---|---|---|---|---|---|---|---|---|---|
| 2048 | 2048 | 2595.14 (2574.26–2621.56) | 67 (63–72) | - | 0 | 0.00 | 17.1 | 4.9818 | 0.000 | 9dafbfbea1ae2c97 |
| 4096 | 4096 | 2643.43 (2641.98–2652.72) | 11 (11–12) | - | 0 | 0.00 | 33.0 | 0.6188 | 1.000 | f1011990c3f971a6 |
| 8192 | 8192 | 3014.55 (3005.61–3016.08) | 12 (11–13) | - | 0 | 0.00 | 66.0 | 3.5177 | 0.000 | dc624137db26fe6c |

peak memory 20860 MiB, model ready in 25.66 s (medians)

decode by steps (step 1 gives the second new word; medians of 4 runs):

| tokens | steps | experts read from the file | decode tok/s | reads' s | MB to the card |
|---|---|---|---|---|---|

## v1 b against v0 a (medians of 4 and 4 runs; positive = faster)

| tokens | prompt | first step | decode | results |
|---|---|---|---|---|
| 2048 | +6.3%, every run faster | -3.0%, ranges overlap | - | same results; nll 4.9818 → 4.9818 (+0.00%), top1 0.000 → 0.000 |
| 4096 | +12.9%, every run faster | +8.1%, ranges overlap | - | same results; nll 0.6188 → 0.6188 (+0.00%), top1 1.000 → 1.000 |
| 8192 | +35.9%, every run faster | +5.2%, ranges overlap | - | same results; nll 3.5177 → 3.5177 (+0.00%), top1 0.000 → 0.000 |

warm-up (does not count): 2048 tokens: prompt 2426.31 tok/s, decode 0.000 tok/s; 4096 tokens: prompt 2347.61 tok/s, decode 0.000 tok/s; 8192 tokens: prompt 2208.83 tok/s, decode 0.000 tok/s
backend: v0 a CUDA; v1 b CUDA
runs not cool: none
the card while running (samples with a kernel running; median, min–max of the runs): v0 a SM 1395 (1380–1425) MHz, under the power cap 22 (0–22) % of the samples, up to 55 °C, 9 (9–9) samples; v1 b SM 1395 (1395–1395) MHz, under the power cap 6 (0–25) % of the samples, up to 55 °C, 8 (8–8) samples
threads: v0 a 14 on 7 cores (CPUs 0,32,1,33,2,34,3,35,4,36,5,37,6,38); v1 b 14 on 7 cores (CPUs 0,32,1,33,2,34,3,35,4,36,5,37,6,38)
pages written to swap during the measurement: 0, 0 of them in the runs that count
