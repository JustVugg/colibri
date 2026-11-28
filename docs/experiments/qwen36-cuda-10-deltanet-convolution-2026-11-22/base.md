# L4 incremental measurement · Tile DeltaNet convolution across input rows

These measurements compare consecutive cumulative builds on the same physical L4. Each build was also checked against the original dev reference.

Baseline: pr09 (source 55850c919b35dcfead37f535e8be195a296b1f0e, compiled c/ tree 80d01f7ebc0edd60b77583d19e89494a23acebd8).
Trial: pr10 (source dc8a4379a9189d01515056f0c09812274235b4ad, compiled c/ tree a288561cdf5ec701085e7d27291fc057197b8d89).
Campaign dev reference: `e677aaee4424d27b8eb0a7e7d7edbb32a309c267`. Model: `Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64@c619aa594ad1e70af82168fb6b4878427896e21c`.
The measurements predate the final packaging of the pull requests.
This series used the earlier benchmark. Its chat first-step timings exclude the server prompt-state save. Fixed-length decode used the request handling of that benchmark.

One reference warm-up per group on dev before the series. Cumulative builds ran three measured repetitions per group. Percentage change between the two build medians. Each build contributes three retained runs per workload.
Groups preserve process history: 64 → 512 → 64; cold 512; 2,048 → 4,096 → 8,192 → 16,384; then a separate eight-turn chat replay.
The first decode step is timed separately. Decode throughput covers the remaining steps. A dagger marks a reported spread above 10% after the repeat.

| Workload | Baseline prefill tok/s | Trial prefill tok/s | Change | Baseline decode tok/s | Trial decode tok/s | Change |
|---|---:|---:|---:|---:|---:|---:|
| Cold 64 | 884.60 | 885.42 | +0.09% | 71.39 | 71.00 | -0.55% |
| 512 in sequence | 2,607.68 | 2,605.08 | -0.10% | 71.11 | 70.98 | -0.18% |
| Final 64 | 1,047.34 | 1,036.77 | -1.01% | 71.43 | 71.34 | -0.12% |
| Cold 512 | 2,483.85 | 2,491.78 | +0.32% | 70.92 | 71.07 | +0.21% |
| 2,048 | 3,273.59 | 3,287.61 | +0.43% | 70.34 | 70.56 | +0.31% |
| 4,096 | 3,324.65 | 3,335.82 | +0.34% | 69.34 | 69.04 | -0.43% |
| 8,192 | 3,274.11 | 3,280.87 | +0.21% | 67.15 | 67.31 | +0.23% |
| 16,384 | 3,140.45 | 3,148.18 | +0.25% | 64.19 | 64.36 | +0.26% |
| Chat 1 | 1,629.54 | 1,634.04 | +0.28% | 80.18 | 79.55 | -0.78% |
| Chat 2 | 2,931.06 | 2,942.34 | +0.38% | 92.17 | 94.09 | +2.08% |
| Chat 3 | 3,106.38 | 3,097.43 | -0.29% | 78.05 | 78.47 | +0.53% |
| Chat 4 | 2,816.79 | 2,847.13 | +1.08% | 91.85 | 92.59 | +0.81% |
| Chat 5 | 2,623.66 | 2,644.73 | +0.80% | 84.64 | 85.24 | +0.71% |
| Chat 6 | 2,282.24 | 2,299.33 | +0.75% | 79.12 | 79.31 | +0.23% |
| Chat 7 | 1,594.28 | 1,600.20 | +0.37% | 74.76 | 75.21 | +0.61% |
| Chat 8 | 402.65 | 404.09 | +0.36% | 71.71 | 69.95 | -2.45% |

## Benchmark invocations

The commands show the recorded benchmark arguments with portable file paths. Set MODEL_DIR to the checkpoint directory and use the binary for each build. The timing runs also recorded the FP32 comparisons reported in correctness.txt.

```sh
COLI_CUDA=1 COLI_GPUS=0 CUDA_EXPERT_GB=auto OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close OMP_NUM_THREADS=6 OMP_DISPLAY_AFFINITY=TRUE OMP_AFFINITY_FORMAT=omp-thread:%n:cpus:%A /usr/bin/time -v ./qwen36-bench --model "$MODEL_DIR" --steps steps.tsv --text promessi_sposi.txt --lengths 64,512,64 --gen 128 --teacher
COLI_CUDA=1 COLI_GPUS=0 CUDA_EXPERT_GB=auto OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close OMP_NUM_THREADS=6 OMP_DISPLAY_AFFINITY=TRUE OMP_AFFINITY_FORMAT=omp-thread:%n:cpus:%A /usr/bin/time -v ./qwen36-bench --model "$MODEL_DIR" --steps steps.tsv --text promessi_sposi.txt --lengths 512 --gen 128 --teacher
COLI_CUDA=1 COLI_GPUS=0 CUDA_EXPERT_GB=auto OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close OMP_NUM_THREADS=6 OMP_DISPLAY_AFFINITY=TRUE OMP_AFFINITY_FORMAT=omp-thread:%n:cpus:%A /usr/bin/time -v ./qwen36-bench --model "$MODEL_DIR" --steps steps.tsv --text promessi_sposi.txt --lengths 2048,4096,8192,16384 --gen 128 --teacher
COLI_CUDA=1 COLI_GPUS=0 CUDA_EXPERT_GB=auto OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close OMP_NUM_THREADS=6 OMP_DISPLAY_AFFINITY=TRUE OMP_AFFINITY_FORMAT=omp-thread:%n:cpus:%A /usr/bin/time -v ./qwen36-bench --model "$MODEL_DIR" --steps steps.tsv --session agente.sess
```

Prose SHA-256: `f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f`. Chat session SHA-256: `8441e1a125a1c7647315c0c10104088b7d5298fd46f5c56036ee39bad74cf5fd`.

## Retained samples

Rows follow the table order above. Times and throughputs use the precision recorded by the benchmark.

```csv
arm,workload_id,round,context_tokens,new_prompt_tokens,generated_tokens,prefill_s,prefill_tok_s,first_decode_ms,steady_decode_tok_s
baseline,1,1,64,64,128,0.072,892.97,15.3,71.392
baseline,1,2,64,64,128,0.072,884.60,15.0,71.403
baseline,1,3,64,64,128,0.073,875.07,15.2,70.921
trial,1,1,64,64,128,0.073,880.53,15.3,71.000
trial,1,2,64,64,128,0.072,885.42,15.4,70.986
trial,1,3,64,64,128,0.072,888.95,15.4,71.319
baseline,2,1,512,512,128,0.196,2618.02,15.1,71.169
baseline,2,2,512,512,128,0.196,2607.68,15.5,71.105
baseline,2,3,512,512,128,0.202,2536.41,15.7,70.817
trial,2,1,512,512,128,0.193,2649.51,15.2,70.978
trial,2,2,512,512,128,0.197,2605.08,15.3,70.762
trial,2,3,512,512,128,0.197,2596.04,15.2,71.181
baseline,3,1,64,64,128,0.061,1047.34,15.3,71.452
baseline,3,2,64,64,128,0.061,1042.90,15.1,71.432
baseline,3,3,64,64,128,0.058,1110.64,14.9,71.064
trial,3,1,64,64,128,0.061,1048.87,15.3,71.344
trial,3,2,64,64,128,0.062,1036.77,15.4,71.203
trial,3,3,64,64,128,0.062,1028.28,15.1,71.366
baseline,4,1,512,512,128,0.205,2497.35,15.2,70.921
baseline,4,2,512,512,128,0.206,2483.85,15.8,71.008
baseline,4,3,512,512,128,0.208,2455.71,15.3,70.706
trial,4,1,512,512,128,0.202,2536.85,15.6,71.171
trial,4,2,512,512,128,0.205,2491.78,15.5,70.988
trial,4,3,512,512,128,0.206,2482.16,15.4,71.069
baseline,5,1,2048,2048,128,0.626,3273.59,15.1,70.314
baseline,5,2,2048,2048,128,0.625,3276.82,15.2,70.530
baseline,5,3,2048,2048,128,0.627,3267.37,15.1,70.343
trial,5,1,2048,2048,128,0.621,3295.79,15.0,70.565
trial,5,2,2048,2048,128,0.623,3287.61,15.2,70.370
trial,5,3,2048,2048,128,0.630,3253.14,15.1,70.564
baseline,6,1,4096,4096,128,1.212,3378.56,15.2,69.082
baseline,6,2,4096,4096,128,1.232,3324.65,15.2,69.561
baseline,6,3,4096,4096,128,1.235,3315.75,15.0,69.339
trial,6,1,4096,4096,128,1.217,3366.75,15.1,69.041
trial,6,2,4096,4096,128,1.232,3323.81,15.1,68.949
trial,6,3,4096,4096,128,1.228,3335.82,15.1,69.643
baseline,7,1,8192,8192,128,2.465,3322.78,15.4,67.154
baseline,7,2,8192,8192,128,2.502,3274.11,15.4,67.285
baseline,7,3,8192,8192,128,2.509,3265.18,15.4,67.138
trial,7,1,8192,8192,128,2.458,3332.52,15.4,67.375
trial,7,2,8192,8192,128,2.497,3280.87,15.4,66.969
trial,7,3,8192,8192,128,2.500,3276.58,15.5,67.310
baseline,8,1,16384,16384,128,5.118,3201.51,16.2,64.192
baseline,8,2,16384,16384,128,5.217,3140.45,16.2,64.393
baseline,8,3,16384,16384,128,5.221,3138.31,16.2,64.062
trial,8,1,16384,16384,128,5.109,3206.65,16.1,64.358
trial,8,2,16384,16384,128,5.204,3148.18,16.2,63.992
trial,8,3,16384,16384,128,5.213,3142.90,16.1,64.388
baseline,9,1,663,663,57,0.405,1635.02,15.3,80.926
baseline,9,2,663,663,57,0.408,1626.44,15.5,78.790
baseline,9,3,663,663,57,0.407,1629.54,15.3,80.176
trial,9,1,663,663,57,0.402,1650.28,15.2,81.180
trial,9,2,663,663,57,0.406,1634.04,15.5,78.986
trial,9,3,663,663,57,0.407,1629.67,15.5,79.547
baseline,10,1,1954,1234,32,0.414,2980.87,15.2,92.173
baseline,10,2,1954,1234,32,0.421,2930.02,15.4,94.446
baseline,10,3,1954,1234,32,0.421,2931.06,15.3,91.777
trial,10,1,1954,1234,32,0.411,2999.27,15.4,92.491
trial,10,2,1954,1234,32,0.420,2940.02,15.5,94.091
trial,10,3,1954,1234,32,0.419,2942.34,15.3,95.336
baseline,11,1,4248,2262,87,0.727,3111.67,15.7,78.257
baseline,11,2,4248,2262,87,0.731,3094.22,15.5,78.032
baseline,11,3,4248,2262,87,0.728,3106.38,15.4,78.054
trial,11,1,4248,2262,87,0.725,3121.90,15.2,78.469
trial,11,2,4248,2262,87,0.731,3093.89,15.6,78.026
trial,11,3,4248,2262,87,0.730,3097.43,15.6,78.613
baseline,12,1,5741,1406,44,0.493,2850.70,15.6,95.758
baseline,12,2,5741,1406,44,0.499,2816.79,15.9,91.793
baseline,12,3,5741,1406,44,0.500,2810.08,15.6,91.852
trial,12,1,5741,1406,44,0.490,2866.48,15.5,92.594
trial,12,2,5741,1406,44,0.494,2847.13,15.6,95.113
trial,12,3,5741,1406,44,0.494,2846.31,15.6,92.526
baseline,13,1,7083,1298,54,0.491,2643.28,15.9,84.420
baseline,13,2,7083,1298,54,0.495,2623.66,15.9,84.638
baseline,13,3,7083,1298,54,0.496,2619.09,15.9,85.158
trial,13,1,7083,1298,54,0.488,2658.11,15.7,85.401
trial,13,2,7083,1298,54,0.493,2631.60,16.0,84.146
trial,13,3,7083,1298,54,0.491,2644.73,16.0,85.239
baseline,14,1,8041,904,24,0.395,2288.61,16.4,79.370
baseline,14,2,8041,904,24,0.396,2282.24,16.3,79.125
baseline,14,3,8041,904,24,0.400,2260.71,15.9,78.769
trial,14,1,8041,904,24,0.393,2299.33,16.4,76.094
trial,14,2,8041,904,24,0.393,2302.80,16.3,79.306
trial,14,3,8041,904,24,0.394,2292.27,16.2,79.819
baseline,15,1,8413,348,467,0.217,1602.96,16.2,74.758
baseline,15,2,8413,348,467,0.219,1591.88,16.0,74.671
baseline,15,3,8413,348,467,0.218,1594.28,16.2,74.853
trial,15,1,8413,348,467,0.216,1612.82,15.8,75.212
trial,15,2,8413,348,467,0.218,1594.45,15.9,75.067
trial,15,3,8413,348,467,0.217,1600.20,16.0,75.235
baseline,16,1,8915,35,186,0.087,403.25,15.9,71.968
baseline,16,2,8915,35,186,0.087,402.65,15.7,69.204
baseline,16,3,8915,35,186,0.088,398.19,15.8,71.712
trial,16,1,8915,35,186,0.086,407.08,15.6,69.747
trial,16,2,8915,35,186,0.087,404.09,15.8,71.651
trial,16,3,8915,35,186,0.087,403.81,15.7,69.953
```
