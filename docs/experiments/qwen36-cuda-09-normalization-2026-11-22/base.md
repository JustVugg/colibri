# L4 incremental measurement · Fuse residual normalization and attention RoPE

These measurements compare consecutive cumulative builds on the same physical L4. Each build was also checked against the original dev reference.

Baseline: pr08 (source e9f481106cd4f82436b757978c514ffdd8b497a1, compiled c/ tree 2d003f9464b9b1ce112e7c6539b12e692887788b).
Trial: pr09 (source 55850c919b35dcfead37f535e8be195a296b1f0e, compiled c/ tree 80d01f7ebc0edd60b77583d19e89494a23acebd8).
Campaign dev reference: `e677aaee4424d27b8eb0a7e7d7edbb32a309c267`. Model: `Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64@c619aa594ad1e70af82168fb6b4878427896e21c`.
The measurements predate the final packaging of the pull requests.
This series used the earlier benchmark. Its chat first-step timings exclude the server prompt-state save. Fixed-length decode used the request handling of that benchmark.

One reference warm-up per group on dev before the series. Cumulative builds ran three measured repetitions per group. Percentage change between the two build medians. Each build contributes three retained runs per workload.
Groups preserve process history: 64 → 512 → 64; cold 512; 2,048 → 4,096 → 8,192 → 16,384; then a separate eight-turn chat replay.
The first decode step is timed separately. Decode throughput covers the remaining steps. A dagger marks a reported spread above 10% after the repeat.

| Workload | Baseline prefill tok/s | Trial prefill tok/s | Change | Baseline decode tok/s | Trial decode tok/s | Change |
|---|---:|---:|---:|---:|---:|---:|
| Cold 64 | 889.59 | 884.60 | -0.56% | 71.99 | 71.39 | -0.83% |
| 512 in sequence | 2,583.84 | 2,607.68 | +0.92% | 71.79 | 71.11 | -0.96% |
| Final 64 | 1,027.20 | 1,047.34 | +1.96% | 72.11 | 71.43 | -0.94% |
| Cold 512 | 2,460.64 | 2,483.85 | +0.94% | 71.88 | 70.92 | -1.33% |
| 2,048 | 3,233.43 | 3,273.59 | +1.24% | 71.26 | 70.34 | -1.28% |
| 4,096 | 3,294.09 | 3,324.65 | +0.93% | 70.27 | 69.34 | -1.32% |
| 8,192 | 3,236.20 | 3,274.11 | +1.17% | 67.92 | 67.15 | -1.13% |
| 16,384 | 3,116.20 | 3,140.45 | +0.78% | 65.01 | 64.19 | -1.26% |
| Chat 1 | 1,621.58 | 1,629.54 | +0.49% | 80.42 | 80.18 | -0.31% |
| Chat 2 | 2,903.47 | 2,931.06 | +0.95% | 95.94 | 92.17 | -3.93% |
| Chat 3 | 3,076.23 | 3,106.38 | +0.98% | 79.40 | 78.05 | -1.70% |
| Chat 4 | 2,803.71 | 2,816.79 | +0.47% | 95.31 | 91.85 | -3.63% |
| Chat 5 | 2,611.86 | 2,623.66 | +0.45% | 86.19 | 84.64 | -1.80% |
| Chat 6 | 2,265.29 | 2,282.24 | +0.75% | 79.70 | 79.12 | -0.72% |
| Chat 7 | 1,590.08 | 1,594.28 | +0.26% | 75.96 | 74.76 | -1.59% |
| Chat 8 | 407.48 | 402.65 | -1.19% | 72.61 | 71.71 | -1.24% |

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
baseline,1,1,64,64,128,0.072,883.40,15.0,71.993
baseline,1,2,64,64,128,0.072,890.53,15.0,72.044
baseline,1,3,64,64,128,0.072,889.59,15.0,71.991
trial,1,1,64,64,128,0.072,892.97,15.3,71.392
trial,1,2,64,64,128,0.072,884.60,15.0,71.403
trial,1,3,64,64,128,0.073,875.07,15.2,70.921
baseline,2,1,512,512,128,0.195,2621.69,14.8,71.791
baseline,2,2,512,512,128,0.198,2583.84,15.1,71.835
baseline,2,3,512,512,128,0.200,2557.92,15.2,71.728
trial,2,1,512,512,128,0.196,2618.02,15.1,71.169
trial,2,2,512,512,128,0.196,2607.68,15.5,71.105
trial,2,3,512,512,128,0.202,2536.41,15.7,70.817
baseline,3,1,64,64,128,0.060,1073.11,14.6,72.110
baseline,3,2,64,64,128,0.063,1020.89,15.2,72.168
baseline,3,3,64,64,128,0.062,1027.20,15.2,72.037
trial,3,1,64,64,128,0.061,1047.34,15.3,71.452
trial,3,2,64,64,128,0.061,1042.90,15.1,71.432
trial,3,3,64,64,128,0.058,1110.64,14.9,71.064
baseline,4,1,512,512,128,0.205,2492.31,14.9,71.841
baseline,4,2,512,512,128,0.208,2460.64,15.5,71.926
baseline,4,3,512,512,128,0.210,2438.58,15.0,71.875
trial,4,1,512,512,128,0.205,2497.35,15.2,70.921
trial,4,2,512,512,128,0.206,2483.85,15.8,71.008
trial,4,3,512,512,128,0.208,2455.71,15.3,70.706
baseline,5,1,2048,2048,128,0.619,3310.35,15.0,71.316
baseline,5,2,2048,2048,128,0.633,3233.43,15.0,71.257
baseline,5,3,2048,2048,128,0.635,3225.70,15.4,71.249
trial,5,1,2048,2048,128,0.626,3273.59,15.1,70.314
trial,5,2,2048,2048,128,0.625,3276.82,15.2,70.530
trial,5,3,2048,2048,128,0.627,3267.37,15.1,70.343
baseline,6,1,4096,4096,128,1.220,3358.61,14.9,70.286
baseline,6,2,4096,4096,128,1.243,3294.09,14.9,70.147
baseline,6,3,4096,4096,128,1.245,3290.03,15.1,70.268
trial,6,1,4096,4096,128,1.212,3378.56,15.2,69.082
trial,6,2,4096,4096,128,1.232,3324.65,15.2,69.561
trial,6,3,4096,4096,128,1.235,3315.75,15.0,69.339
baseline,7,1,8192,8192,128,2.481,3302.15,15.2,67.977
baseline,7,2,8192,8192,128,2.531,3236.20,15.3,67.852
baseline,7,3,8192,8192,128,2.534,3232.72,15.4,67.924
trial,7,1,8192,8192,128,2.465,3322.78,15.4,67.154
trial,7,2,8192,8192,128,2.502,3274.11,15.4,67.285
trial,7,3,8192,8192,128,2.509,3265.18,15.4,67.138
baseline,8,1,16384,16384,128,5.170,3169.23,16.0,65.083
baseline,8,2,16384,16384,128,5.258,3116.20,16.1,64.878
baseline,8,3,16384,16384,128,5.286,3099.58,16.0,65.012
trial,8,1,16384,16384,128,5.118,3201.51,16.2,64.192
trial,8,2,16384,16384,128,5.217,3140.45,16.2,64.393
trial,8,3,16384,16384,128,5.221,3138.31,16.2,64.062
baseline,9,1,663,663,57,0.405,1636.38,15.2,81.452
baseline,9,2,663,663,57,0.409,1621.58,15.1,80.052
baseline,9,3,663,663,57,0.410,1617.58,15.0,80.424
trial,9,1,663,663,57,0.405,1635.02,15.3,80.926
trial,9,2,663,663,57,0.408,1626.44,15.5,78.790
trial,9,3,663,663,57,0.407,1629.54,15.3,80.176
baseline,10,1,1954,1234,32,0.413,2985.20,15.2,92.809
baseline,10,2,1954,1234,32,0.425,2903.47,15.2,95.942
baseline,10,3,1954,1234,32,0.427,2891.42,15.0,96.429
trial,10,1,1954,1234,32,0.414,2980.87,15.2,92.173
trial,10,2,1954,1234,32,0.421,2930.02,15.4,94.446
trial,10,3,1954,1234,32,0.421,2931.06,15.3,91.777
baseline,11,1,4248,2262,87,0.733,3086.66,15.1,79.258
baseline,11,2,4248,2262,87,0.738,3065.89,15.5,79.404
baseline,11,3,4248,2262,87,0.735,3076.23,15.5,79.426
trial,11,1,4248,2262,87,0.727,3111.67,15.7,78.257
trial,11,2,4248,2262,87,0.731,3094.22,15.5,78.032
trial,11,3,4248,2262,87,0.728,3106.38,15.4,78.054
baseline,12,1,5741,1406,44,0.494,2848.61,15.3,93.606
baseline,12,2,5741,1406,44,0.502,2799.82,15.6,95.898
baseline,12,3,5741,1406,44,0.501,2803.71,16.0,95.313
trial,12,1,5741,1406,44,0.493,2850.70,15.6,95.758
trial,12,2,5741,1406,44,0.499,2816.79,15.9,91.793
trial,12,3,5741,1406,44,0.500,2810.08,15.6,91.852
baseline,13,1,7083,1298,54,0.494,2628.65,15.7,86.188
baseline,13,2,7083,1298,54,0.497,2609.41,16.0,86.192
baseline,13,3,7083,1298,54,0.497,2611.86,16.2,86.338
trial,13,1,7083,1298,54,0.491,2643.28,15.9,84.420
trial,13,2,7083,1298,54,0.495,2623.66,15.9,84.638
trial,13,3,7083,1298,54,0.496,2619.09,15.9,85.158
baseline,14,1,8041,904,24,0.396,2280.11,16.0,79.701
baseline,14,2,8041,904,24,0.399,2265.29,16.2,80.397
baseline,14,3,8041,904,24,0.400,2259.53,15.7,79.457
trial,14,1,8041,904,24,0.395,2288.61,16.4,79.370
trial,14,2,8041,904,24,0.396,2282.24,16.3,79.125
trial,14,3,8041,904,24,0.400,2260.71,15.9,78.769
baseline,15,1,8413,348,467,0.216,1611.96,15.8,75.964
baseline,15,2,8413,348,467,0.219,1587.87,16.1,75.836
baseline,15,3,8413,348,467,0.219,1590.08,16.1,76.070
trial,15,1,8413,348,467,0.217,1602.96,16.2,74.758
trial,15,2,8413,348,467,0.219,1591.88,16.0,74.671
trial,15,3,8413,348,467,0.218,1594.28,16.2,74.853
baseline,16,1,8915,35,186,0.086,407.48,15.7,72.615
baseline,16,2,8915,35,186,0.083,423.20,15.2,72.841
baseline,16,3,8915,35,186,0.088,398.80,15.6,72.343
trial,16,1,8915,35,186,0.087,403.25,15.9,71.968
trial,16,2,8915,35,186,0.087,402.65,15.7,69.204
trial,16,3,8915,35,186,0.088,398.19,15.8,71.712
```
