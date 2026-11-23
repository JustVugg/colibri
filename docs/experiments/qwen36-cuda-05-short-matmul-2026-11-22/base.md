# L4 incremental measurement · Specialize short INT8 and FP32 projections

These measurements compare consecutive cumulative builds on the same physical L4. Each build was also checked against the original dev reference.

Baseline: pr04 (source 1f5bba1efcac99dab7227f68ee098452d60af900, compiled c/ tree 24a49c8e227780e760424a27e6b169a6815bbfb4).
Trial: pr05 (source 1740e7d290e0362f85c6b53efdc3282e6034855b, compiled c/ tree 65c39e33b354fb02c8ee864cdd248f1a036248c1).
Campaign dev reference: `e677aaee4424d27b8eb0a7e7d7edbb32a309c267`. Model: `Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64@c619aa594ad1e70af82168fb6b4878427896e21c`.
The measurements predate the final packaging of the pull requests.
This series used the earlier benchmark. Its chat first-step timings exclude the server prompt-state save. Fixed-length decode used the request handling of that benchmark.

One reference warm-up per group on dev before the series. Cumulative builds ran three measured repetitions per group. Percentage change between the two build medians. Each build contributes three retained runs per workload.
Groups preserve process history: 64 → 512 → 64; cold 512; 2,048 → 4,096 → 8,192 → 16,384; then a separate eight-turn chat replay.
The first decode step is timed separately. Decode throughput covers the remaining steps. A dagger marks a reported spread above 10% after the repeat.

| Workload | Baseline prefill tok/s | Trial prefill tok/s | Change | Baseline decode tok/s | Trial decode tok/s | Change |
|---|---:|---:|---:|---:|---:|---:|
| Cold 64 | 850.22 | 851.77 | +0.18% | 70.07 | 70.26 | +0.28% |
| 512 in sequence | 2,367.15 | 2,357.44 | -0.41% | 70.54 | 70.04 | -0.71% |
| Final 64 | 991.01 | 982.49 | -0.86% | 71.02 | 70.39 | -0.87% |
| Cold 512 | 2,273.09 | 2,279.99 | +0.30% | 70.85 | 70.37 | -0.67% |
| 2,048 | 2,926.48 | 2,924.34 | -0.07% | 70.20 | 70.23 | +0.05% |
| 4,096 | 3,001.23 | 3,023.23 | +0.73% | 69.34 | 69.43 | +0.13% |
| 8,192 | 2,971.84 | 2,966.27 | -0.19% | 67.21 | 67.27 | +0.09% |
| 16,384 | 2,853.44 | 2,850.35 | -0.11% | 64.13 | 64.33 | +0.32% |
| Chat 1 | 1,541.02 | 1,541.46 | +0.03% | 79.44 | 80.92 | +1.87% |
| Chat 2 | 2,657.22 | 2,665.37 | +0.31% | 92.17 | 91.97 | -0.21% |
| Chat 3 | 2,750.68 | 2,788.65 | +1.38% | 77.17 | 78.48 | +1.69% |
| Chat 4 | 2,588.42 | 2,566.72 | -0.84% | 94.16 | 95.74 | +1.67% |
| Chat 5 | 2,407.02 | 2,404.87 | -0.09% | 84.67 | 84.71 | +0.04% |
| Chat 6 | 2,098.72 | 2,102.00 | +0.16% | 79.39 | 79.02 | -0.47% |
| Chat 7 | 1,499.42 | 1,505.63 | +0.41% | 74.97 | 74.86 | -0.15% |
| Chat 8 | 403.13 | 403.01 | -0.03% | 71.79 | 71.39 | -0.56% |

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
baseline,1,1,64,64,128,0.076,847.60,15.5,69.971
baseline,1,2,64,64,128,0.075,850.22,15.3,70.067
baseline,1,3,64,64,128,0.074,868.97,15.4,71.005
trial,1,1,64,64,128,0.076,846.63,15.5,70.263
trial,1,2,64,64,128,0.075,851.77,15.6,70.262
trial,1,3,64,64,128,0.075,852.45,15.5,70.077
baseline,2,1,512,512,128,0.213,2400.48,15.4,70.543
baseline,2,2,512,512,128,0.216,2367.15,15.7,69.740
baseline,2,3,512,512,128,0.227,2256.93,15.5,70.752
trial,2,1,512,512,128,0.217,2362.73,15.7,70.044
trial,2,2,512,512,128,0.217,2357.44,15.3,70.184
trial,2,3,512,512,128,0.218,2351.67,15.4,70.038
baseline,3,1,64,64,128,0.064,1002.42,15.3,71.202
baseline,3,2,64,64,128,0.065,991.01,15.6,70.874
baseline,3,3,64,64,128,0.066,975.71,15.4,71.015
trial,3,1,64,64,128,0.065,977.71,15.5,70.394
trial,3,2,64,64,128,0.065,982.49,15.5,70.479
trial,3,3,64,64,128,0.065,989.42,15.4,70.370
baseline,4,1,512,512,128,0.222,2304.39,15.7,70.880
baseline,4,2,512,512,128,0.225,2273.09,15.7,70.697
baseline,4,3,512,512,128,0.227,2258.01,15.7,70.850
trial,4,1,512,512,128,0.222,2307.80,15.3,70.959
trial,4,2,512,512,128,0.225,2279.99,15.7,70.374
trial,4,3,512,512,128,0.226,2269.64,15.4,70.214
baseline,5,1,2048,2048,128,0.692,2959.07,15.2,70.196
baseline,5,2,2048,2048,128,0.704,2910.50,15.2,69.886
baseline,5,3,2048,2048,128,0.700,2926.48,15.6,70.300
trial,5,1,2048,2048,128,0.688,2976.39,15.6,70.430
trial,5,2,2048,2048,128,0.700,2924.34,15.5,70.228
trial,5,3,2048,2048,128,0.701,2920.36,15.2,70.046
baseline,6,1,4096,4096,128,1.344,3047.56,15.3,69.337
baseline,6,2,4096,4096,128,1.365,3001.23,15.1,69.481
baseline,6,3,4096,4096,128,1.366,2998.63,15.2,68.626
trial,6,1,4096,4096,128,1.332,3074.17,15.1,69.530
trial,6,2,4096,4096,128,1.355,3023.23,15.1,69.427
trial,6,3,4096,4096,128,1.361,3009.38,15.3,69.181
baseline,7,1,8192,8192,128,2.724,3007.89,15.6,67.211
baseline,7,2,8192,8192,128,2.762,2965.48,15.6,67.255
baseline,7,3,8192,8192,128,2.757,2971.84,15.6,65.963
trial,7,1,8192,8192,128,2.721,3010.76,15.5,67.307
trial,7,2,8192,8192,128,2.764,2963.32,15.8,66.984
trial,7,3,8192,8192,128,2.762,2966.27,15.7,67.274
baseline,8,1,16384,16384,128,5.652,2899.02,16.3,64.127
baseline,8,2,16384,16384,128,5.742,2853.44,16.4,64.174
baseline,8,3,16384,16384,128,5.759,2845.17,16.4,64.031
trial,8,1,16384,16384,128,5.637,2906.46,16.2,64.480
trial,8,2,16384,16384,128,5.748,2850.35,16.4,64.274
trial,8,3,16384,16384,128,5.760,2844.33,16.4,64.332
baseline,9,1,663,663,57,0.430,1542.03,15.4,79.435
baseline,9,2,663,663,57,0.430,1541.02,15.5,80.869
baseline,9,3,663,663,57,0.435,1525.53,15.4,77.756
trial,9,1,663,663,57,0.427,1551.54,15.5,80.932
trial,9,2,663,663,57,0.430,1540.89,15.5,80.920
trial,9,3,663,663,57,0.430,1541.46,15.5,78.689
baseline,10,1,1954,1234,32,0.460,2683.15,15.4,90.276
baseline,10,2,1954,1234,32,0.464,2657.22,15.2,92.171
baseline,10,3,1954,1234,32,0.467,2641.19,15.5,92.617
trial,10,1,1954,1234,32,0.460,2681.74,15.7,91.974
trial,10,2,1954,1234,32,0.463,2665.37,15.9,91.782
trial,10,3,1954,1234,32,0.464,2657.07,15.9,93.931
baseline,11,1,4248,2262,87,0.813,2783.28,15.8,77.175
baseline,11,2,4248,2262,87,0.822,2750.68,15.9,78.327
baseline,11,3,4248,2262,87,0.823,2748.06,15.9,77.163
trial,11,1,4248,2262,87,0.809,2796.05,15.6,78.476
trial,11,2,4248,2262,87,0.813,2783.99,15.8,78.569
trial,11,3,4248,2262,87,0.811,2788.65,15.4,77.938
baseline,12,1,5741,1406,44,0.543,2590.45,15.9,90.624
baseline,12,2,5741,1406,44,0.543,2588.42,15.7,94.162
baseline,12,3,5741,1406,44,0.551,2549.49,16.3,95.182
trial,12,1,5741,1406,44,0.533,2639.36,15.7,95.898
trial,12,2,5741,1406,44,0.549,2559.54,15.7,95.738
trial,12,3,5741,1406,44,0.548,2566.72,15.8,91.605
baseline,13,1,7083,1298,54,0.544,2387.71,16.1,83.913
baseline,13,2,7083,1298,54,0.539,2407.50,15.7,84.974
baseline,13,3,7083,1298,54,0.539,2407.02,16.5,84.674
trial,13,1,7083,1298,54,0.535,2427.22,16.2,85.183
trial,13,2,7083,1298,54,0.541,2399.42,15.8,84.706
trial,13,3,7083,1298,54,0.540,2404.87,15.9,84.596
baseline,14,1,8041,904,24,0.430,2103.76,16.2,79.386
baseline,14,2,8041,904,24,0.431,2098.72,16.5,79.620
baseline,14,3,8041,904,24,0.434,2084.29,16.3,79.299
trial,14,1,8041,904,24,0.426,2122.39,16.0,79.956
trial,14,2,8041,904,24,0.430,2100.37,16.6,78.868
trial,14,3,8041,904,24,0.430,2102.00,16.7,79.016
baseline,15,1,8413,348,467,0.232,1499.42,16.1,75.299
baseline,15,2,8413,348,467,0.230,1512.86,16.2,74.659
baseline,15,3,8413,348,467,0.232,1498.63,16.4,74.971
trial,15,1,8413,348,467,0.231,1505.63,16.3,74.856
trial,15,2,8413,348,467,0.230,1513.21,16.3,75.049
trial,15,3,8413,348,467,0.232,1500.78,16.4,74.305
baseline,16,1,8915,35,186,0.087,403.13,15.7,71.892
baseline,16,2,8915,35,186,0.087,403.20,15.8,71.684
baseline,16,3,8915,35,186,0.088,397.38,16.0,71.792
trial,16,1,8915,35,186,0.087,402.11,15.6,70.098
trial,16,2,8915,35,186,0.087,403.92,15.6,72.033
trial,16,3,8915,35,186,0.087,403.01,15.8,71.389
```
