# L4 incremental measurement · Tile native INT8 and INT4 prompt projections

These measurements compare consecutive cumulative builds on the same physical L4. Each build was also checked against the original dev reference.

Baseline: pr05 (source 1740e7d290e0362f85c6b53efdc3282e6034855b, compiled c/ tree 65c39e33b354fb02c8ee864cdd248f1a036248c1).
Trial: pr06 (source cc5490429557cc3f3181194cf4457024e701dda3, compiled c/ tree eb3f58dfd302e09a9b667a6fe7cc5442fa2a110b).
Campaign dev reference: `e677aaee4424d27b8eb0a7e7d7edbb32a309c267`. Model: `Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64@c619aa594ad1e70af82168fb6b4878427896e21c`.
The measurements predate the final packaging of the pull requests.
This series used the earlier benchmark. Its chat first-step timings exclude the server prompt-state save. Fixed-length decode used the request handling of that benchmark.

One reference warm-up per group on dev before the series. Cumulative builds ran three measured repetitions per group. Percentage change between the two build medians. Each build contributes three retained runs per workload.
Groups preserve process history: 64 → 512 → 64; cold 512; 2,048 → 4,096 → 8,192 → 16,384; then a separate eight-turn chat replay.
The first decode step is timed separately. Decode throughput covers the remaining steps. A dagger marks a reported spread above 10% after the repeat.

| Workload | Baseline prefill tok/s | Trial prefill tok/s | Change | Baseline decode tok/s | Trial decode tok/s | Change |
|---|---:|---:|---:|---:|---:|---:|
| Cold 64 | 851.77 | 874.79 | +2.70% | 70.26 | 71.09 | +1.17% |
| 512 in sequence | 2,357.44 | 2,630.81 | +11.60% | 70.04 | 70.92 | +1.26% |
| Final 64 | 982.49 | 1,006.71 | +2.47% | 70.39 | 71.20 | +1.15% |
| Cold 512 | 2,279.99 | 2,488.38 | +9.14% | 70.37 | 70.88 | +0.72% |
| 2,048 | 2,924.34 | 3,362.27 | +14.98% | 70.23 | 70.22 | -0.00% |
| 4,096 | 3,023.23 | 3,467.61 | +14.70% | 69.43 | 69.23 | -0.28% |
| 8,192 | 2,966.27 | 3,420.83 | +15.32% | 67.27 | 67.01 | -0.39% |
| 16,384 | 2,850.35 | 3,273.87 | +14.86% | 64.33 | 64.07 | -0.41% |
| Chat 1 | 1,541.46 | 1,658.93 | +7.62% | 80.92 | 80.44 | -0.59% |
| Chat 2 | 2,665.37 | 3,030.97 | +13.72% | 91.97 | 92.24 | +0.29% |
| Chat 3 | 2,788.65 | 3,197.58 | +14.66% | 78.48 | 78.16 | -0.40% |
| Chat 4 | 2,566.72 | 2,902.01 | +13.06% | 95.74 | 92.89 | -2.98% |
| Chat 5 | 2,404.87 | 2,711.01 | +12.73% | 84.71 | 85.26 | +0.65% |
| Chat 6 | 2,102.00 | 2,316.16 | +10.19% | 79.02 | 78.94 | -0.10% |
| Chat 7 | 1,505.63 | 1,611.99 | +7.06% | 74.86 | 74.85 | -0.00% |
| Chat 8 | 403.01 | 399.03 | -0.99% | 71.39 | 70.95 | -0.61% |

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
baseline,1,1,64,64,128,0.076,846.63,15.5,70.263
baseline,1,2,64,64,128,0.075,851.77,15.6,70.262
baseline,1,3,64,64,128,0.075,852.45,15.5,70.077
trial,1,1,64,64,128,0.073,874.79,15.2,71.098
trial,1,2,64,64,128,0.072,883.79,15.3,71.086
trial,1,3,64,64,128,0.074,864.51,15.3,70.952
baseline,2,1,512,512,128,0.217,2362.73,15.7,70.044
baseline,2,2,512,512,128,0.217,2357.44,15.3,70.184
baseline,2,3,512,512,128,0.218,2351.67,15.4,70.038
trial,2,1,512,512,128,0.189,2709.71,15.4,70.924
trial,2,2,512,512,128,0.195,2630.81,15.6,70.966
trial,2,3,512,512,128,0.198,2590.42,15.4,70.747
baseline,3,1,64,64,128,0.065,977.71,15.5,70.394
baseline,3,2,64,64,128,0.065,982.49,15.5,70.479
baseline,3,3,64,64,128,0.065,989.42,15.4,70.370
trial,3,1,64,64,128,0.063,1015.12,15.2,71.204
trial,3,2,64,64,128,0.064,1003.41,15.3,71.263
trial,3,3,64,64,128,0.064,1006.71,15.1,71.133
baseline,4,1,512,512,128,0.222,2307.80,15.3,70.959
baseline,4,2,512,512,128,0.225,2279.99,15.7,70.374
baseline,4,3,512,512,128,0.226,2269.64,15.4,70.214
trial,4,1,512,512,128,0.201,2544.65,15.2,70.844
trial,4,2,512,512,128,0.206,2488.38,15.4,70.879
trial,4,3,512,512,128,0.206,2482.14,15.5,70.897
baseline,5,1,2048,2048,128,0.688,2976.39,15.6,70.430
baseline,5,2,2048,2048,128,0.700,2924.34,15.5,70.228
baseline,5,3,2048,2048,128,0.701,2920.36,15.2,70.046
trial,5,1,2048,2048,128,0.606,3381.87,15.6,70.301
trial,5,2,2048,2048,128,0.609,3362.27,15.6,70.145
trial,5,3,2048,2048,128,0.617,3320.73,15.6,70.225
baseline,6,1,4096,4096,128,1.332,3074.17,15.1,69.530
baseline,6,2,4096,4096,128,1.355,3023.23,15.1,69.427
baseline,6,3,4096,4096,128,1.361,3009.38,15.3,69.181
trial,6,1,4096,4096,128,1.155,3545.08,15.1,69.319
trial,6,2,4096,4096,128,1.181,3467.61,15.1,69.230
trial,6,3,4096,4096,128,1.187,3450.85,15.1,69.210
baseline,7,1,8192,8192,128,2.721,3010.76,15.5,67.307
baseline,7,2,8192,8192,128,2.764,2963.32,15.8,66.984
baseline,7,3,8192,8192,128,2.762,2966.27,15.7,67.274
trial,7,1,8192,8192,128,2.364,3464.79,15.6,66.975
trial,7,2,8192,8192,128,2.395,3420.83,15.6,67.062
trial,7,3,8192,8192,128,2.412,3396.07,15.6,67.011
baseline,8,1,16384,16384,128,5.637,2906.46,16.2,64.480
baseline,8,2,16384,16384,128,5.748,2850.35,16.4,64.274
baseline,8,3,16384,16384,128,5.760,2844.33,16.4,64.332
trial,8,1,16384,16384,128,4.914,3334.35,16.2,64.106
trial,8,2,16384,16384,128,5.004,3273.87,16.2,64.020
trial,8,3,16384,16384,128,5.014,3267.68,16.4,64.070
baseline,9,1,663,663,57,0.427,1551.54,15.5,80.932
baseline,9,2,663,663,57,0.430,1540.89,15.5,80.920
baseline,9,3,663,663,57,0.430,1541.46,15.5,78.689
trial,9,1,663,663,57,0.398,1666.77,15.5,80.850
trial,9,2,663,663,57,0.400,1658.93,15.4,78.954
trial,9,3,663,663,57,0.402,1651.22,15.6,80.441
baseline,10,1,1954,1234,32,0.460,2681.74,15.7,91.974
baseline,10,2,1954,1234,32,0.463,2665.37,15.9,91.782
baseline,10,3,1954,1234,32,0.464,2657.07,15.9,93.931
trial,10,1,1954,1234,32,0.405,3046.83,15.2,92.242
trial,10,2,1954,1234,32,0.407,3030.97,15.3,94.650
trial,10,3,1954,1234,32,0.408,3024.36,15.2,91.971
baseline,11,1,4248,2262,87,0.809,2796.05,15.6,78.476
baseline,11,2,4248,2262,87,0.813,2783.99,15.8,78.569
baseline,11,3,4248,2262,87,0.811,2788.65,15.4,77.938
trial,11,1,4248,2262,87,0.697,3244.09,15.3,78.331
trial,11,2,4248,2262,87,0.707,3197.58,15.3,77.947
trial,11,3,4248,2262,87,0.709,3189.13,15.7,78.161
baseline,12,1,5741,1406,44,0.533,2639.36,15.7,95.898
baseline,12,2,5741,1406,44,0.549,2559.54,15.7,95.738
baseline,12,3,5741,1406,44,0.548,2566.72,15.8,91.605
trial,12,1,5741,1406,44,0.476,2950.70,15.6,92.886
trial,12,2,5741,1406,44,0.484,2902.01,15.7,92.052
trial,12,3,5741,1406,44,0.486,2894.75,16.3,95.908
baseline,13,1,7083,1298,54,0.535,2427.22,16.2,85.183
baseline,13,2,7083,1298,54,0.541,2399.42,15.8,84.706
baseline,13,3,7083,1298,54,0.540,2404.87,15.9,84.596
trial,13,1,7083,1298,54,0.478,2716.96,16.2,85.463
trial,13,2,7083,1298,54,0.479,2711.01,16.2,85.258
trial,13,3,7083,1298,54,0.483,2689.47,16.4,84.351
baseline,14,1,8041,904,24,0.426,2122.39,16.0,79.956
baseline,14,2,8041,904,24,0.430,2100.37,16.6,78.868
baseline,14,3,8041,904,24,0.430,2102.00,16.7,79.016
trial,14,1,8041,904,24,0.385,2348.57,16.0,78.937
trial,14,2,8041,904,24,0.390,2316.16,15.9,79.610
trial,14,3,8041,904,24,0.393,2297.93,16.0,78.407
baseline,15,1,8413,348,467,0.231,1505.63,16.3,74.856
baseline,15,2,8413,348,467,0.230,1513.21,16.3,75.049
baseline,15,3,8413,348,467,0.232,1500.78,16.4,74.305
trial,15,1,8413,348,467,0.216,1611.99,16.3,75.138
trial,15,2,8413,348,467,0.216,1612.56,15.9,74.853
trial,15,3,8413,348,467,0.217,1600.15,16.4,74.605
baseline,16,1,8915,35,186,0.087,402.11,15.6,70.098
baseline,16,2,8915,35,186,0.087,403.92,15.6,72.033
baseline,16,3,8915,35,186,0.087,403.01,15.8,71.389
trial,16,1,8915,35,186,0.086,406.72,15.6,70.106
trial,16,2,8915,35,186,0.088,399.03,15.9,71.790
trial,16,3,8915,35,186,0.089,391.65,15.9,70.951
```
