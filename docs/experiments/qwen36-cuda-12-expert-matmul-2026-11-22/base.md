# L4 incremental measurement · Tile routed-expert matmul by expert

These measurements compare consecutive cumulative builds on the same physical L4. Each build was also checked against the original dev reference.

Baseline: pr11 (source ccbe8d8d27287f7273c8ed7c4fd68b349febb383, compiled c/ tree bc99e0f193cbf1b57e3e97e2208b76a537f44755).
Trial: pr12 (source 7efe604d95380ded913baf25434d57801c6f0f36, compiled c/ tree 911b1f7ecf33762a81ac08f8250d444503921724).
Campaign dev reference: `e677aaee4424d27b8eb0a7e7d7edbb32a309c267`. Model: `Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64@c619aa594ad1e70af82168fb6b4878427896e21c`.
The measurements predate the final packaging of the pull requests.
This series used the earlier benchmark. Its chat first-step timings exclude the server prompt-state save. Fixed-length decode used the request handling of that benchmark.

One reference warm-up per group on dev before the series. Cumulative builds ran three measured repetitions per group. Percentage change between the two build medians. Each build contributes three retained runs per workload.
Groups preserve process history: 64 → 512 → 64; cold 512; 2,048 → 4,096 → 8,192 → 16,384; then a separate eight-turn chat replay.
The first decode step is timed separately. Decode throughput covers the remaining steps. A dagger marks a reported spread above 10% after the repeat.

| Workload | Baseline prefill tok/s | Trial prefill tok/s | Change | Baseline decode tok/s | Trial decode tok/s | Change |
|---|---:|---:|---:|---:|---:|---:|
| Cold 64 | 887.83 | 1,066.57 | +20.13% | 73.25 | 73.23 | -0.02% |
| 512 in sequence | 2,641.61 | 3,302.07 | +25.00% | 72.95 | 73.01 | +0.08% |
| Final 64 | 1,033.67 | 1,315.82 | +27.30% | 73.21 | 73.54 | +0.45% |
| Cold 512 | 2,487.57 | 3,115.75 | +25.25% | 72.78 | 73.31 | +0.73% |
| 2,048 | 3,286.08 | 3,791.20 | +15.37% | 72.48 | 72.56 | +0.10% |
| 4,096 | 3,342.15 | 3,801.99 | +13.76% | 71.42 | 71.59 | +0.24% |
| 8,192 | 3,300.64 | 3,744.39 | +13.44% | 69.01 | 69.04 | +0.04% |
| 16,384 | 3,161.03 | 3,583.92 | +13.38% | 65.98 | 66.20 | +0.35% |
| Chat 1 | 1,645.71 | 1,989.60 | +20.90% | 83.00 | 83.36 | +0.42% |
| Chat 2 | 2,972.05 | 3,532.06 | +18.84% | 93.77 | 93.81 | +0.04% |
| Chat 3 | 3,126.99 | 3,599.05 | +15.10% | 80.07 | 80.32 | +0.32% |
| Chat 4 | 2,866.85 | 3,361.95 | +17.27% | 95.23 | 93.89 | -1.41% |
| Chat 5 | 2,658.71 | 3,109.11 | +16.94% | 86.48 | 86.46 | -0.03% |
| Chat 6 | 2,295.94 | 2,733.36 | +19.05% | 82.53 | 83.22 | +0.83% |
| Chat 7 | 1,614.97 | 1,964.52 | +21.64% | 77.59 | 77.99 | +0.51% |
| Chat 8 | 407.74 | 476.43 | +16.85% | 74.18 | 75.33 | +1.56% |

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
baseline,1,1,64,64,128,0.072,887.83,14.6,72.956
baseline,1,2,64,64,128,0.072,890.84,14.9,73.383
baseline,1,3,64,64,128,0.072,887.83,14.8,73.250
trial,1,1,64,64,128,0.061,1048.54,14.4,73.235
trial,1,2,64,64,128,0.060,1066.57,14.4,73.252
trial,1,3,64,64,128,0.060,1068.21,14.4,73.157
baseline,2,1,512,512,128,0.192,2664.06,15.0,72.857
baseline,2,2,512,512,128,0.194,2641.61,15.1,72.947
baseline,2,3,512,512,128,0.194,2634.42,14.7,73.220
trial,2,1,512,512,128,0.153,3337.88,14.6,73.007
trial,2,2,512,512,128,0.155,3302.07,14.9,73.075
trial,2,3,512,512,128,0.155,3299.29,14.7,72.945
baseline,3,1,64,64,128,0.060,1065.81,14.6,73.212
baseline,3,2,64,64,128,0.062,1033.67,14.8,73.402
baseline,3,3,64,64,128,0.062,1028.71,14.9,72.918
trial,3,1,64,64,128,0.049,1315.82,14.3,73.392
trial,3,2,64,64,128,0.049,1314.66,14.1,73.541
trial,3,3,64,64,128,0.049,1316.73,14.4,73.568
baseline,4,1,512,512,128,0.204,2513.44,15.0,72.785
baseline,4,2,512,512,128,0.206,2486.13,15.1,72.690
baseline,4,3,512,512,128,0.206,2487.57,14.9,73.130
trial,4,1,512,512,128,0.165,3106.23,14.8,73.215
trial,4,2,512,512,128,0.164,3115.75,14.7,73.348
trial,4,3,512,512,128,0.164,3116.29,14.8,73.313
baseline,5,1,2048,2048,128,0.615,3330.52,14.7,72.471
baseline,5,2,2048,2048,128,0.630,3248.80,14.6,72.518
baseline,5,3,2048,2048,128,0.623,3286.08,14.8,72.481
trial,5,1,2048,2048,128,0.538,3807.74,15.0,72.556
trial,5,2,2048,2048,128,0.540,3791.20,14.5,72.486
trial,5,3,2048,2048,128,0.542,3782.08,14.9,72.664
baseline,6,1,4096,4096,128,1.205,3398.62,14.8,71.423
baseline,6,2,4096,4096,128,1.226,3342.15,14.8,71.417
baseline,6,3,4096,4096,128,1.229,3333.90,14.6,71.273
trial,6,1,4096,4096,128,1.068,3836.86,14.6,71.381
trial,6,2,4096,4096,128,1.077,3801.99,14.7,71.586
trial,6,3,4096,4096,128,1.081,3787.62,14.6,71.620
baseline,7,1,8192,8192,128,2.450,3343.07,15.1,69.042
baseline,7,2,8192,8192,128,2.482,3300.64,15.2,68.924
baseline,7,3,8192,8192,128,2.495,3283.98,15.3,69.014
trial,7,1,8192,8192,128,2.176,3764.62,14.9,69.040
trial,7,2,8192,8192,128,2.188,3744.39,15.0,68.972
trial,7,3,8192,8192,128,2.188,3743.71,15.1,69.220
baseline,8,1,16384,16384,128,5.095,3215.60,15.9,65.716
baseline,8,2,16384,16384,128,5.183,3161.03,15.9,66.027
baseline,8,3,16384,16384,128,5.208,3145.84,15.8,65.977
trial,8,1,16384,16384,128,4.544,3605.91,15.7,65.956
trial,8,2,16384,16384,128,4.574,3582.36,15.8,66.205
trial,8,3,16384,16384,128,4.572,3583.92,15.7,66.206
baseline,9,1,663,663,57,0.401,1651.51,14.7,83.003
baseline,9,2,663,663,57,0.403,1645.71,15.1,83.079
baseline,9,3,663,663,57,0.407,1628.92,14.7,82.957
trial,9,1,663,663,57,0.335,1976.69,14.4,81.759
trial,9,2,663,663,57,0.333,1993.79,14.3,83.358
trial,9,3,663,663,57,0.333,1989.60,14.3,83.355
baseline,10,1,1954,1234,32,0.415,2972.05,14.7,93.418
baseline,10,2,1954,1234,32,0.413,2988.77,14.9,93.878
baseline,10,3,1954,1234,32,0.420,2936.33,14.9,93.769
trial,10,1,1954,1234,32,0.348,3542.39,14.5,93.807
trial,10,2,1954,1234,32,0.349,3532.06,14.7,93.946
trial,10,3,1954,1234,32,0.349,3531.46,15.0,93.559
baseline,11,1,4248,2262,87,0.721,3137.20,14.8,80.132
baseline,11,2,4248,2262,87,0.723,3126.99,15.0,80.070
baseline,11,3,4248,2262,87,0.725,3122.05,15.0,80.044
trial,11,1,4248,2262,87,0.624,3627.73,14.6,80.180
trial,11,2,4248,2262,87,0.629,3596.40,14.6,80.323
trial,11,3,4248,2262,87,0.628,3599.05,14.7,80.445
baseline,12,1,5741,1406,44,0.489,2875.59,15.6,95.229
baseline,12,2,5741,1406,44,0.490,2866.85,15.3,96.381
baseline,12,3,5741,1406,44,0.495,2843.14,15.2,93.197
trial,12,1,5741,1406,44,0.418,3361.95,14.9,93.527
trial,12,2,5741,1406,44,0.419,3359.41,14.9,93.953
trial,12,3,5741,1406,44,0.416,3380.13,15.3,93.889
baseline,13,1,7083,1298,54,0.487,2667.81,15.3,86.583
baseline,13,2,7083,1298,54,0.488,2658.71,15.3,85.736
baseline,13,3,7083,1298,54,0.495,2623.68,15.6,86.481
trial,13,1,7083,1298,54,0.419,3101.28,15.5,86.636
trial,13,2,7083,1298,54,0.416,3116.48,15.1,86.456
trial,13,3,7083,1298,54,0.417,3109.11,15.1,86.423
baseline,14,1,8041,904,24,0.390,2318.60,15.7,82.953
baseline,14,2,8041,904,24,0.394,2295.94,15.7,82.530
baseline,14,3,8041,904,24,0.396,2283.03,16.0,82.372
trial,14,1,8041,904,24,0.332,2719.55,15.5,82.965
trial,14,2,8041,904,24,0.330,2742.94,15.2,83.244
trial,14,3,8041,904,24,0.331,2733.36,15.6,83.217
baseline,15,1,8413,348,467,0.215,1617.71,15.4,77.650
baseline,15,2,8413,348,467,0.215,1614.97,15.5,77.595
baseline,15,3,8413,348,467,0.217,1605.30,15.5,76.631
trial,15,1,8413,348,467,0.179,1943.43,15.4,77.987
trial,15,2,8413,348,467,0.177,1964.52,15.6,77.827
trial,15,3,8413,348,467,0.177,1966.77,15.4,78.122
baseline,16,1,8915,35,186,0.086,407.74,15.4,74.254
baseline,16,2,8915,35,186,0.087,403.22,15.2,74.179
baseline,16,3,8915,35,186,0.086,408.15,15.2,74.075
trial,16,1,8915,35,186,0.073,476.43,14.8,75.460
trial,16,2,8915,35,186,0.073,476.67,14.8,75.333
trial,16,3,8915,35,186,0.074,472.23,14.6,75.152
```
