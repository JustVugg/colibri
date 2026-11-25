# L4 incremental measurement · Prepare DeltaNet recurrence inputs in parallel

These measurements compare consecutive cumulative builds on the same physical L4. Each build was also checked against the original dev reference.

Baseline: pr10 (source dc8a4379a9189d01515056f0c09812274235b4ad, compiled c/ tree a288561cdf5ec701085e7d27291fc057197b8d89).
Trial: pr11 (source ccbe8d8d27287f7273c8ed7c4fd68b349febb383, compiled c/ tree bc99e0f193cbf1b57e3e97e2208b76a537f44755).
Campaign dev reference: `e677aaee4424d27b8eb0a7e7d7edbb32a309c267`. Model: `Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64@c619aa594ad1e70af82168fb6b4878427896e21c`.
The measurements predate the final packaging of the pull requests.
This series used the earlier benchmark. Its chat first-step timings exclude the server prompt-state save. Fixed-length decode used the request handling of that benchmark.

One reference warm-up per group on dev before the series. Cumulative builds ran three measured repetitions per group. Percentage change between the two build medians. Each build contributes three retained runs per workload.
Groups preserve process history: 64 → 512 → 64; cold 512; 2,048 → 4,096 → 8,192 → 16,384; then a separate eight-turn chat replay.
The first decode step is timed separately. Decode throughput covers the remaining steps. A dagger marks a reported spread above 10% after the repeat.

| Workload | Baseline prefill tok/s | Trial prefill tok/s | Change | Baseline decode tok/s | Trial decode tok/s | Change |
|---|---:|---:|---:|---:|---:|---:|
| Cold 64 | 885.42 | 887.83 | +0.27% | 71.00 | 73.25 | +3.17% |
| 512 in sequence | 2,605.08 | 2,641.61 | +1.40% | 70.98 | 72.95 | +2.77% |
| Final 64 | 1,036.77 | 1,033.67 | -0.30% | 71.34 | 73.21 | +2.62% |
| Cold 512 | 2,491.78 | 2,487.57 | -0.17% | 71.07 | 72.78 | +2.41% |
| 2,048 | 3,287.61 | 3,286.08 | -0.05% | 70.56 | 72.48 | +2.72% |
| 4,096 | 3,335.82 | 3,342.15 | +0.19% | 69.04 | 71.42 | +3.44% |
| 8,192 | 3,280.87 | 3,300.64 | +0.60% | 67.31 | 69.01 | +2.53% |
| 16,384 | 3,148.18 | 3,161.03 | +0.41% | 64.36 | 65.98 | +2.52% |
| Chat 1 | 1,634.04 | 1,645.71 | +0.71% | 79.55 | 83.00 | +4.34% |
| Chat 2 | 2,942.34 | 2,972.05 | +1.01% | 94.09 | 93.77 | -0.34% |
| Chat 3 | 3,097.43 | 3,126.99 | +0.95% | 78.47 | 80.07 | +2.04% |
| Chat 4 | 2,847.13 | 2,866.85 | +0.69% | 92.59 | 95.23 | +2.85% |
| Chat 5 | 2,644.73 | 2,658.71 | +0.53% | 85.24 | 86.48 | +1.46% |
| Chat 6 | 2,299.33 | 2,295.94 | -0.15% | 79.31 | 82.53 | +4.07% |
| Chat 7 | 1,600.20 | 1,614.97 | +0.92% | 75.21 | 77.59 | +3.17% |
| Chat 8 | 404.09 | 407.74 | +0.90% | 69.95 | 74.18 | +6.04% |

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
baseline,1,1,64,64,128,0.073,880.53,15.3,71.000
baseline,1,2,64,64,128,0.072,885.42,15.4,70.986
baseline,1,3,64,64,128,0.072,888.95,15.4,71.319
trial,1,1,64,64,128,0.072,887.83,14.6,72.956
trial,1,2,64,64,128,0.072,890.84,14.9,73.383
trial,1,3,64,64,128,0.072,887.83,14.8,73.250
baseline,2,1,512,512,128,0.193,2649.51,15.2,70.978
baseline,2,2,512,512,128,0.197,2605.08,15.3,70.762
baseline,2,3,512,512,128,0.197,2596.04,15.2,71.181
trial,2,1,512,512,128,0.192,2664.06,15.0,72.857
trial,2,2,512,512,128,0.194,2641.61,15.1,72.947
trial,2,3,512,512,128,0.194,2634.42,14.7,73.220
baseline,3,1,64,64,128,0.061,1048.87,15.3,71.344
baseline,3,2,64,64,128,0.062,1036.77,15.4,71.203
baseline,3,3,64,64,128,0.062,1028.28,15.1,71.366
trial,3,1,64,64,128,0.060,1065.81,14.6,73.212
trial,3,2,64,64,128,0.062,1033.67,14.8,73.402
trial,3,3,64,64,128,0.062,1028.71,14.9,72.918
baseline,4,1,512,512,128,0.202,2536.85,15.6,71.171
baseline,4,2,512,512,128,0.205,2491.78,15.5,70.988
baseline,4,3,512,512,128,0.206,2482.16,15.4,71.069
trial,4,1,512,512,128,0.204,2513.44,15.0,72.785
trial,4,2,512,512,128,0.206,2486.13,15.1,72.690
trial,4,3,512,512,128,0.206,2487.57,14.9,73.130
baseline,5,1,2048,2048,128,0.621,3295.79,15.0,70.565
baseline,5,2,2048,2048,128,0.623,3287.61,15.2,70.370
baseline,5,3,2048,2048,128,0.630,3253.14,15.1,70.564
trial,5,1,2048,2048,128,0.615,3330.52,14.7,72.471
trial,5,2,2048,2048,128,0.630,3248.80,14.6,72.518
trial,5,3,2048,2048,128,0.623,3286.08,14.8,72.481
baseline,6,1,4096,4096,128,1.217,3366.75,15.1,69.041
baseline,6,2,4096,4096,128,1.232,3323.81,15.1,68.949
baseline,6,3,4096,4096,128,1.228,3335.82,15.1,69.643
trial,6,1,4096,4096,128,1.205,3398.62,14.8,71.423
trial,6,2,4096,4096,128,1.226,3342.15,14.8,71.417
trial,6,3,4096,4096,128,1.229,3333.90,14.6,71.273
baseline,7,1,8192,8192,128,2.458,3332.52,15.4,67.375
baseline,7,2,8192,8192,128,2.497,3280.87,15.4,66.969
baseline,7,3,8192,8192,128,2.500,3276.58,15.5,67.310
trial,7,1,8192,8192,128,2.450,3343.07,15.1,69.042
trial,7,2,8192,8192,128,2.482,3300.64,15.2,68.924
trial,7,3,8192,8192,128,2.495,3283.98,15.3,69.014
baseline,8,1,16384,16384,128,5.109,3206.65,16.1,64.358
baseline,8,2,16384,16384,128,5.204,3148.18,16.2,63.992
baseline,8,3,16384,16384,128,5.213,3142.90,16.1,64.388
trial,8,1,16384,16384,128,5.095,3215.60,15.9,65.716
trial,8,2,16384,16384,128,5.183,3161.03,15.9,66.027
trial,8,3,16384,16384,128,5.208,3145.84,15.8,65.977
baseline,9,1,663,663,57,0.402,1650.28,15.2,81.180
baseline,9,2,663,663,57,0.406,1634.04,15.5,78.986
baseline,9,3,663,663,57,0.407,1629.67,15.5,79.547
trial,9,1,663,663,57,0.401,1651.51,14.7,83.003
trial,9,2,663,663,57,0.403,1645.71,15.1,83.079
trial,9,3,663,663,57,0.407,1628.92,14.7,82.957
baseline,10,1,1954,1234,32,0.411,2999.27,15.4,92.491
baseline,10,2,1954,1234,32,0.420,2940.02,15.5,94.091
baseline,10,3,1954,1234,32,0.419,2942.34,15.3,95.336
trial,10,1,1954,1234,32,0.415,2972.05,14.7,93.418
trial,10,2,1954,1234,32,0.413,2988.77,14.9,93.878
trial,10,3,1954,1234,32,0.420,2936.33,14.9,93.769
baseline,11,1,4248,2262,87,0.725,3121.90,15.2,78.469
baseline,11,2,4248,2262,87,0.731,3093.89,15.6,78.026
baseline,11,3,4248,2262,87,0.730,3097.43,15.6,78.613
trial,11,1,4248,2262,87,0.721,3137.20,14.8,80.132
trial,11,2,4248,2262,87,0.723,3126.99,15.0,80.070
trial,11,3,4248,2262,87,0.725,3122.05,15.0,80.044
baseline,12,1,5741,1406,44,0.490,2866.48,15.5,92.594
baseline,12,2,5741,1406,44,0.494,2847.13,15.6,95.113
baseline,12,3,5741,1406,44,0.494,2846.31,15.6,92.526
trial,12,1,5741,1406,44,0.489,2875.59,15.6,95.229
trial,12,2,5741,1406,44,0.490,2866.85,15.3,96.381
trial,12,3,5741,1406,44,0.495,2843.14,15.2,93.197
baseline,13,1,7083,1298,54,0.488,2658.11,15.7,85.401
baseline,13,2,7083,1298,54,0.493,2631.60,16.0,84.146
baseline,13,3,7083,1298,54,0.491,2644.73,16.0,85.239
trial,13,1,7083,1298,54,0.487,2667.81,15.3,86.583
trial,13,2,7083,1298,54,0.488,2658.71,15.3,85.736
trial,13,3,7083,1298,54,0.495,2623.68,15.6,86.481
baseline,14,1,8041,904,24,0.393,2299.33,16.4,76.094
baseline,14,2,8041,904,24,0.393,2302.80,16.3,79.306
baseline,14,3,8041,904,24,0.394,2292.27,16.2,79.819
trial,14,1,8041,904,24,0.390,2318.60,15.7,82.953
trial,14,2,8041,904,24,0.394,2295.94,15.7,82.530
trial,14,3,8041,904,24,0.396,2283.03,16.0,82.372
baseline,15,1,8413,348,467,0.216,1612.82,15.8,75.212
baseline,15,2,8413,348,467,0.218,1594.45,15.9,75.067
baseline,15,3,8413,348,467,0.217,1600.20,16.0,75.235
trial,15,1,8413,348,467,0.215,1617.71,15.4,77.650
trial,15,2,8413,348,467,0.215,1614.97,15.5,77.595
trial,15,3,8413,348,467,0.217,1605.30,15.5,76.631
baseline,16,1,8915,35,186,0.086,407.08,15.6,69.747
baseline,16,2,8915,35,186,0.087,404.09,15.8,71.651
baseline,16,3,8915,35,186,0.087,403.81,15.7,69.953
trial,16,1,8915,35,186,0.086,407.74,15.4,74.254
trial,16,2,8915,35,186,0.087,403.22,15.2,74.179
trial,16,3,8915,35,186,0.086,408.15,15.2,74.075
```
