# L4 incremental measurement · Reuse packed activations within a CUDA frame

These measurements compare consecutive cumulative builds on the same physical L4. Each build was also checked against the original dev reference.

Baseline: pr06 (source cc5490429557cc3f3181194cf4457024e701dda3, compiled c/ tree eb3f58dfd302e09a9b667a6fe7cc5442fa2a110b).
Trial: pr07 (source 9df9dc63437a9c8971ef8f1c7f06b89418ea86e7, compiled c/ tree d23e208c13c49ac143d024e2e94d6c94bcdb59f4).
Campaign dev reference: `e677aaee4424d27b8eb0a7e7d7edbb32a309c267`. Model: `Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64@c619aa594ad1e70af82168fb6b4878427896e21c`.
The measurements predate the final packaging of the pull requests.
This series used the earlier benchmark. Its chat first-step timings exclude the server prompt-state save. Fixed-length decode used the request handling of that benchmark.

One reference warm-up per group on dev before the series. Cumulative builds ran three measured repetitions per group. Percentage change between the two build medians. Each build contributes three retained runs per workload.
Groups preserve process history: 64 → 512 → 64; cold 512; 2,048 → 4,096 → 8,192 → 16,384; then a separate eight-turn chat replay.
The first decode step is timed separately. Decode throughput covers the remaining steps. A dagger marks a reported spread above 10% after the repeat.

| Workload | Baseline prefill tok/s | Trial prefill tok/s | Change | Baseline decode tok/s | Trial decode tok/s | Change |
|---|---:|---:|---:|---:|---:|---:|
| Cold 64 | 874.79 | 888.90 | +1.61% | 71.09 | 71.04 | -0.06% |
| 512 in sequence | 2,630.81 | 2,578.22 | -2.00% | 70.92 | 70.77 | -0.21% |
| Final 64 | 1,006.71 | 1,023.96 | +1.71% | 71.20 | 71.17 | -0.05% |
| Cold 512 | 2,488.38 | 2,452.83 | -1.43% | 70.88 | 70.81 | -0.10% |
| 2,048 | 3,362.27 | 3,217.70 | -4.30% | 70.22 | 70.39 | +0.24% |
| 4,096 | 3,467.61 | 3,294.88 | -4.98% | 69.23 | 69.47 | +0.35% |
| 8,192 | 3,420.83 | 3,238.68 | -5.32% | 67.01 | 67.08 | +0.10% |
| 16,384 | 3,273.87 | 3,117.64 | -4.77% | 64.07 | 64.28 | +0.34% |
| Chat 1 | 1,658.93 | 1,626.24 | -1.97% | 80.44 | 79.28 | -1.44% |
| Chat 2 | 3,030.97 | 2,909.68 | -4.00% | 92.24 | 94.31 | +2.24% |
| Chat 3 | 3,197.58 | 3,061.70 | -4.25% | 78.16 | 78.14 | -0.02% |
| Chat 4 | 2,902.01 | 2,803.34 | -3.40% | 92.89 | 95.23 | +2.53% |
| Chat 5 | 2,711.01 | 2,625.23 | -3.16% | 85.26 | 84.97 | -0.33% |
| Chat 6 | 2,316.16 | 2,281.39 | -1.50% | 78.94 | 79.36 | +0.54% |
| Chat 7 | 1,611.99 | 1,591.77 | -1.25% | 74.85 | 74.89 | +0.05% |
| Chat 8 | 399.03 | 405.15 | +1.53% | 70.95 | 71.94 | +1.39% |

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
baseline,1,1,64,64,128,0.073,874.79,15.2,71.098
baseline,1,2,64,64,128,0.072,883.79,15.3,71.086
baseline,1,3,64,64,128,0.074,864.51,15.3,70.952
trial,1,1,64,64,128,0.071,899.98,15.5,71.040
trial,1,2,64,64,128,0.073,880.91,15.2,71.138
trial,1,3,64,64,128,0.072,888.90,15.3,71.007
baseline,2,1,512,512,128,0.189,2709.71,15.4,70.924
baseline,2,2,512,512,128,0.195,2630.81,15.6,70.966
baseline,2,3,512,512,128,0.198,2590.42,15.4,70.747
trial,2,1,512,512,128,0.196,2613.56,15.5,70.760
trial,2,2,512,512,128,0.201,2548.89,15.2,70.967
trial,2,3,512,512,128,0.199,2578.22,15.8,70.773
baseline,3,1,64,64,128,0.063,1015.12,15.2,71.204
baseline,3,2,64,64,128,0.064,1003.41,15.3,71.263
baseline,3,3,64,64,128,0.064,1006.71,15.1,71.133
trial,3,1,64,64,128,0.061,1052.52,15.3,71.066
trial,3,2,64,64,128,0.063,1023.36,15.3,71.204
trial,3,3,64,64,128,0.063,1023.96,15.1,71.166
baseline,4,1,512,512,128,0.201,2544.65,15.2,70.844
baseline,4,2,512,512,128,0.206,2488.38,15.4,70.879
baseline,4,3,512,512,128,0.206,2482.14,15.5,70.897
trial,4,1,512,512,128,0.205,2494.97,15.1,70.835
trial,4,2,512,512,128,0.209,2452.83,15.4,70.780
trial,4,3,512,512,128,0.210,2440.56,15.2,70.809
baseline,5,1,2048,2048,128,0.606,3381.87,15.6,70.301
baseline,5,2,2048,2048,128,0.609,3362.27,15.6,70.145
baseline,5,3,2048,2048,128,0.617,3320.73,15.6,70.225
trial,5,1,2048,2048,128,0.620,3301.63,15.1,70.418
trial,5,2,2048,2048,128,0.636,3217.70,15.3,70.251
trial,5,3,2048,2048,128,0.638,3210.34,15.6,70.393
baseline,6,1,4096,4096,128,1.155,3545.08,15.1,69.319
baseline,6,2,4096,4096,128,1.181,3467.61,15.1,69.230
baseline,6,3,4096,4096,128,1.187,3450.85,15.1,69.210
trial,6,1,4096,4096,128,1.224,3346.61,15.0,69.479
trial,6,2,4096,4096,128,1.243,3294.88,15.1,69.315
trial,6,3,4096,4096,128,1.248,3283.14,15.4,69.472
baseline,7,1,8192,8192,128,2.364,3464.79,15.6,66.975
baseline,7,2,8192,8192,128,2.395,3420.83,15.6,67.062
baseline,7,3,8192,8192,128,2.412,3396.07,15.6,67.011
trial,7,1,8192,8192,128,2.486,3295.24,15.4,67.078
trial,7,2,8192,8192,128,2.530,3237.48,15.7,67.220
trial,7,3,8192,8192,128,2.529,3238.68,15.5,66.975
baseline,8,1,16384,16384,128,4.914,3334.35,16.2,64.106
baseline,8,2,16384,16384,128,5.004,3273.87,16.2,64.020
baseline,8,3,16384,16384,128,5.014,3267.68,16.4,64.070
trial,8,1,16384,16384,128,5.170,3168.97,16.2,64.285
trial,8,2,16384,16384,128,5.255,3117.64,16.4,64.153
trial,8,3,16384,16384,128,5.287,3098.68,16.2,64.298
baseline,9,1,663,663,57,0.398,1666.77,15.5,80.850
baseline,9,2,663,663,57,0.400,1658.93,15.4,78.954
baseline,9,3,663,663,57,0.402,1651.22,15.6,80.441
trial,9,1,663,663,57,0.406,1634.81,15.5,80.046
trial,9,2,663,663,57,0.409,1620.04,15.5,79.284
trial,9,3,663,663,57,0.408,1626.24,15.2,79.138
baseline,10,1,1954,1234,32,0.405,3046.83,15.2,92.242
baseline,10,2,1954,1234,32,0.407,3030.97,15.3,94.650
baseline,10,3,1954,1234,32,0.408,3024.36,15.2,91.971
trial,10,1,1954,1234,32,0.417,2957.62,15.5,91.345
trial,10,2,1954,1234,32,0.424,2909.68,16.0,94.707
trial,10,3,1954,1234,32,0.427,2892.46,15.4,94.312
baseline,11,1,4248,2262,87,0.697,3244.09,15.3,78.331
baseline,11,2,4248,2262,87,0.707,3197.58,15.3,77.947
baseline,11,3,4248,2262,87,0.709,3189.13,15.7,78.161
trial,11,1,4248,2262,87,0.731,3096.08,15.8,77.962
trial,11,2,4248,2262,87,0.742,3049.80,15.7,78.325
trial,11,3,4248,2262,87,0.739,3061.70,15.5,78.143
baseline,12,1,5741,1406,44,0.476,2950.70,15.6,92.886
baseline,12,2,5741,1406,44,0.484,2902.01,15.7,92.052
baseline,12,3,5741,1406,44,0.486,2894.75,16.3,95.908
trial,12,1,5741,1406,44,0.496,2832.52,15.6,95.235
trial,12,2,5741,1406,44,0.503,2795.73,15.9,95.389
trial,12,3,5741,1406,44,0.502,2803.34,16.3,94.331
baseline,13,1,7083,1298,54,0.478,2716.96,16.2,85.463
baseline,13,2,7083,1298,54,0.479,2711.01,16.2,85.258
baseline,13,3,7083,1298,54,0.483,2689.47,16.4,84.351
trial,13,1,7083,1298,54,0.494,2629.21,16.1,84.327
trial,13,2,7083,1298,54,0.494,2625.23,16.4,84.975
trial,13,3,7083,1298,54,0.497,2612.12,16.5,85.093
baseline,14,1,8041,904,24,0.385,2348.57,16.0,78.937
baseline,14,2,8041,904,24,0.390,2316.16,15.9,79.610
baseline,14,3,8041,904,24,0.393,2297.93,16.0,78.407
trial,14,1,8041,904,24,0.396,2281.39,16.3,79.191
trial,14,2,8041,904,24,0.400,2260.45,16.3,79.364
trial,14,3,8041,904,24,0.388,2330.69,16.3,79.598
baseline,15,1,8413,348,467,0.216,1611.99,16.3,75.138
baseline,15,2,8413,348,467,0.216,1612.56,15.9,74.853
baseline,15,3,8413,348,467,0.217,1600.15,16.4,74.605
trial,15,1,8413,348,467,0.218,1596.67,15.9,74.494
trial,15,2,8413,348,467,0.220,1579.56,16.2,74.890
trial,15,3,8413,348,467,0.219,1591.77,15.9,75.072
baseline,16,1,8915,35,186,0.086,406.72,15.6,70.106
baseline,16,2,8915,35,186,0.088,399.03,15.9,71.790
baseline,16,3,8915,35,186,0.089,391.65,15.9,70.951
trial,16,1,8915,35,186,0.086,405.23,15.6,71.939
trial,16,2,8915,35,186,0.086,405.15,15.7,71.863
trial,16,3,8915,35,186,0.087,404.31,15.6,72.022
```
