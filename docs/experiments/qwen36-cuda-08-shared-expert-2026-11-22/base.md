# L4 incremental measurement · Fuse SwiGLU with the shared expert projections

These measurements compare consecutive cumulative builds on the same physical L4. Each build was also checked against the original dev reference.

Baseline: pr07 (source 9df9dc63437a9c8971ef8f1c7f06b89418ea86e7, compiled c/ tree d23e208c13c49ac143d024e2e94d6c94bcdb59f4).
Trial: pr08 (source e9f481106cd4f82436b757978c514ffdd8b497a1, compiled c/ tree 2d003f9464b9b1ce112e7c6539b12e692887788b).
Campaign dev reference: `e677aaee4424d27b8eb0a7e7d7edbb32a309c267`. Model: `Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64@c619aa594ad1e70af82168fb6b4878427896e21c`.
The measurements predate the final packaging of the pull requests.
This series used the earlier benchmark. Its chat first-step timings exclude the server prompt-state save. Fixed-length decode used the request handling of that benchmark.

One reference warm-up per group on dev before the series. Cumulative builds ran three measured repetitions per group. Percentage change between the two build medians. Each build contributes three retained runs per workload.
Groups preserve process history: 64 → 512 → 64; cold 512; 2,048 → 4,096 → 8,192 → 16,384; then a separate eight-turn chat replay.
The first decode step is timed separately. Decode throughput covers the remaining steps. A dagger marks a reported spread above 10% after the repeat.

| Workload | Baseline prefill tok/s | Trial prefill tok/s | Change | Baseline decode tok/s | Trial decode tok/s | Change |
|---|---:|---:|---:|---:|---:|---:|
| Cold 64 | 888.90 | 889.59 | +0.08% | 71.04 | 71.99 | +1.34% |
| 512 in sequence | 2,578.22 | 2,583.84 | +0.22% | 70.77 | 71.79 | +1.44% |
| Final 64 | 1,023.96 | 1,027.20 | +0.32% | 71.17 | 72.11 | +1.33% |
| Cold 512 | 2,452.83 | 2,460.64 | +0.32% | 70.81 | 71.88 | +1.51% |
| 2,048 | 3,217.70 | 3,233.43 | +0.49% | 70.39 | 71.26 | +1.23% |
| 4,096 | 3,294.88 | 3,294.09 | -0.02% | 69.47 | 70.27 | +1.15% |
| 8,192 | 3,238.68 | 3,236.20 | -0.08% | 67.08 | 67.92 | +1.26% |
| 16,384 | 3,117.64 | 3,116.20 | -0.05% | 64.28 | 65.01 | +1.13% |
| Chat 1 | 1,626.24 | 1,621.58 | -0.29% | 79.28 | 80.42 | +1.44% |
| Chat 2 | 2,909.68 | 2,903.47 | -0.21% | 94.31 | 95.94 | +1.73% |
| Chat 3 | 3,061.70 | 3,076.23 | +0.47% | 78.14 | 79.40 | +1.61% |
| Chat 4 | 2,803.34 | 2,803.71 | +0.01% | 95.23 | 95.31 | +0.08% |
| Chat 5 | 2,625.23 | 2,611.86 | -0.51% | 84.97 | 86.19 | +1.43% |
| Chat 6 | 2,281.39 | 2,265.29 | -0.71% | 79.36 | 79.70 | +0.42% |
| Chat 7 | 1,591.77 | 1,590.08 | -0.11% | 74.89 | 75.96 | +1.43% |
| Chat 8 | 405.15 | 407.48 | +0.58% | 71.94 | 72.61 | +0.94% |

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
baseline,1,1,64,64,128,0.071,899.98,15.5,71.040
baseline,1,2,64,64,128,0.073,880.91,15.2,71.138
baseline,1,3,64,64,128,0.072,888.90,15.3,71.007
trial,1,1,64,64,128,0.072,883.40,15.0,71.993
trial,1,2,64,64,128,0.072,890.53,15.0,72.044
trial,1,3,64,64,128,0.072,889.59,15.0,71.991
baseline,2,1,512,512,128,0.196,2613.56,15.5,70.760
baseline,2,2,512,512,128,0.201,2548.89,15.2,70.967
baseline,2,3,512,512,128,0.199,2578.22,15.8,70.773
trial,2,1,512,512,128,0.195,2621.69,14.8,71.791
trial,2,2,512,512,128,0.198,2583.84,15.1,71.835
trial,2,3,512,512,128,0.200,2557.92,15.2,71.728
baseline,3,1,64,64,128,0.061,1052.52,15.3,71.066
baseline,3,2,64,64,128,0.063,1023.36,15.3,71.204
baseline,3,3,64,64,128,0.063,1023.96,15.1,71.166
trial,3,1,64,64,128,0.060,1073.11,14.6,72.110
trial,3,2,64,64,128,0.063,1020.89,15.2,72.168
trial,3,3,64,64,128,0.062,1027.20,15.2,72.037
baseline,4,1,512,512,128,0.205,2494.97,15.1,70.835
baseline,4,2,512,512,128,0.209,2452.83,15.4,70.780
baseline,4,3,512,512,128,0.210,2440.56,15.2,70.809
trial,4,1,512,512,128,0.205,2492.31,14.9,71.841
trial,4,2,512,512,128,0.208,2460.64,15.5,71.926
trial,4,3,512,512,128,0.210,2438.58,15.0,71.875
baseline,5,1,2048,2048,128,0.620,3301.63,15.1,70.418
baseline,5,2,2048,2048,128,0.636,3217.70,15.3,70.251
baseline,5,3,2048,2048,128,0.638,3210.34,15.6,70.393
trial,5,1,2048,2048,128,0.619,3310.35,15.0,71.316
trial,5,2,2048,2048,128,0.633,3233.43,15.0,71.257
trial,5,3,2048,2048,128,0.635,3225.70,15.4,71.249
baseline,6,1,4096,4096,128,1.224,3346.61,15.0,69.479
baseline,6,2,4096,4096,128,1.243,3294.88,15.1,69.315
baseline,6,3,4096,4096,128,1.248,3283.14,15.4,69.472
trial,6,1,4096,4096,128,1.220,3358.61,14.9,70.286
trial,6,2,4096,4096,128,1.243,3294.09,14.9,70.147
trial,6,3,4096,4096,128,1.245,3290.03,15.1,70.268
baseline,7,1,8192,8192,128,2.486,3295.24,15.4,67.078
baseline,7,2,8192,8192,128,2.530,3237.48,15.7,67.220
baseline,7,3,8192,8192,128,2.529,3238.68,15.5,66.975
trial,7,1,8192,8192,128,2.481,3302.15,15.2,67.977
trial,7,2,8192,8192,128,2.531,3236.20,15.3,67.852
trial,7,3,8192,8192,128,2.534,3232.72,15.4,67.924
baseline,8,1,16384,16384,128,5.170,3168.97,16.2,64.285
baseline,8,2,16384,16384,128,5.255,3117.64,16.4,64.153
baseline,8,3,16384,16384,128,5.287,3098.68,16.2,64.298
trial,8,1,16384,16384,128,5.170,3169.23,16.0,65.083
trial,8,2,16384,16384,128,5.258,3116.20,16.1,64.878
trial,8,3,16384,16384,128,5.286,3099.58,16.0,65.012
baseline,9,1,663,663,57,0.406,1634.81,15.5,80.046
baseline,9,2,663,663,57,0.409,1620.04,15.5,79.284
baseline,9,3,663,663,57,0.408,1626.24,15.2,79.138
trial,9,1,663,663,57,0.405,1636.38,15.2,81.452
trial,9,2,663,663,57,0.409,1621.58,15.1,80.052
trial,9,3,663,663,57,0.410,1617.58,15.0,80.424
baseline,10,1,1954,1234,32,0.417,2957.62,15.5,91.345
baseline,10,2,1954,1234,32,0.424,2909.68,16.0,94.707
baseline,10,3,1954,1234,32,0.427,2892.46,15.4,94.312
trial,10,1,1954,1234,32,0.413,2985.20,15.2,92.809
trial,10,2,1954,1234,32,0.425,2903.47,15.2,95.942
trial,10,3,1954,1234,32,0.427,2891.42,15.0,96.429
baseline,11,1,4248,2262,87,0.731,3096.08,15.8,77.962
baseline,11,2,4248,2262,87,0.742,3049.80,15.7,78.325
baseline,11,3,4248,2262,87,0.739,3061.70,15.5,78.143
trial,11,1,4248,2262,87,0.733,3086.66,15.1,79.258
trial,11,2,4248,2262,87,0.738,3065.89,15.5,79.404
trial,11,3,4248,2262,87,0.735,3076.23,15.5,79.426
baseline,12,1,5741,1406,44,0.496,2832.52,15.6,95.235
baseline,12,2,5741,1406,44,0.503,2795.73,15.9,95.389
baseline,12,3,5741,1406,44,0.502,2803.34,16.3,94.331
trial,12,1,5741,1406,44,0.494,2848.61,15.3,93.606
trial,12,2,5741,1406,44,0.502,2799.82,15.6,95.898
trial,12,3,5741,1406,44,0.501,2803.71,16.0,95.313
baseline,13,1,7083,1298,54,0.494,2629.21,16.1,84.327
baseline,13,2,7083,1298,54,0.494,2625.23,16.4,84.975
baseline,13,3,7083,1298,54,0.497,2612.12,16.5,85.093
trial,13,1,7083,1298,54,0.494,2628.65,15.7,86.188
trial,13,2,7083,1298,54,0.497,2609.41,16.0,86.192
trial,13,3,7083,1298,54,0.497,2611.86,16.2,86.338
baseline,14,1,8041,904,24,0.396,2281.39,16.3,79.191
baseline,14,2,8041,904,24,0.400,2260.45,16.3,79.364
baseline,14,3,8041,904,24,0.388,2330.69,16.3,79.598
trial,14,1,8041,904,24,0.396,2280.11,16.0,79.701
trial,14,2,8041,904,24,0.399,2265.29,16.2,80.397
trial,14,3,8041,904,24,0.400,2259.53,15.7,79.457
baseline,15,1,8413,348,467,0.218,1596.67,15.9,74.494
baseline,15,2,8413,348,467,0.220,1579.56,16.2,74.890
baseline,15,3,8413,348,467,0.219,1591.77,15.9,75.072
trial,15,1,8413,348,467,0.216,1611.96,15.8,75.964
trial,15,2,8413,348,467,0.219,1587.87,16.1,75.836
trial,15,3,8413,348,467,0.219,1590.08,16.1,76.070
baseline,16,1,8915,35,186,0.086,405.23,15.6,71.939
baseline,16,2,8915,35,186,0.086,405.15,15.7,71.863
baseline,16,3,8915,35,186,0.087,404.31,15.6,72.022
trial,16,1,8915,35,186,0.086,407.48,15.7,72.615
trial,16,2,8915,35,186,0.083,423.20,15.2,72.841
trial,16,3,8915,35,186,0.088,398.80,15.6,72.343
```
