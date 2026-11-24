# L4 incremental measurement · Split attention over long prefixes

These measurements compare consecutive cumulative builds on the same physical L4. Each build was also checked against the original dev reference.

Baseline: pr12 (source 7efe604d95380ded913baf25434d57801c6f0f36, compiled c/ tree 911b1f7ecf33762a81ac08f8250d444503921724).
Trial: pr13 (source b536a26f4162e7e2b24c5d5bcf1a9f9d6ecf66c4, compiled c/ tree f9e6a14305c8a2682c229dd341fb4d3ead32ee13).
Campaign dev reference: `e677aaee4424d27b8eb0a7e7d7edbb32a309c267`. Model: `Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64@c619aa594ad1e70af82168fb6b4878427896e21c`.
The measurements predate the final packaging of the pull requests.
This series used the earlier benchmark. Its chat first-step timings exclude the server prompt-state save. Fixed-length decode used the request handling of that benchmark.

One reference warm-up per group on dev before the series. Cumulative builds ran three measured repetitions per group. Percentage change between the two build medians. Each build contributes three retained runs per workload.
Groups preserve process history: 64 → 512 → 64; cold 512; 2,048 → 4,096 → 8,192 → 16,384; then a separate eight-turn chat replay.
The first decode step is timed separately. Decode throughput covers the remaining steps. A dagger marks a reported spread above 10% after the repeat.

| Workload | Baseline prefill tok/s | Trial prefill tok/s | Change | Baseline decode tok/s | Trial decode tok/s | Change |
|---|---:|---:|---:|---:|---:|---:|
| Cold 64 | 1,066.57 | 1,058.74 | -0.73% | 73.23 | 73.44 | +0.28% |
| 512 in sequence | 3,302.07 | 3,299.05 | -0.09% | 73.01 | 73.29 | +0.38% |
| Final 64 | 1,315.82 | 1,320.21 | +0.33% | 73.54 | 73.69 | +0.20% |
| Cold 512 | 3,115.75 | 3,089.13 | -0.85% | 73.31 | 73.29 | -0.03% |
| 2,048 | 3,791.20 | 3,797.38 | +0.16% | 72.56 | 72.71 | +0.21% |
| 4,096 | 3,801.99 | 3,809.05 | +0.19% | 71.59 | 71.68 | +0.13% |
| 8,192 | 3,744.39 | 3,750.05 | +0.15% | 69.04 | 69.33 | +0.43% |
| 16,384 | 3,583.92 | 3,581.69 | -0.06% | 66.20 | 66.24 | +0.05% |
| Chat 1 | 1,989.60 | 1,980.81 | -0.44% | 83.36 | 83.27 | -0.11% |
| Chat 2 | 3,532.06 | 3,550.40 | +0.52% | 93.81 | 94.04 | +0.24% |
| Chat 3 | 3,599.05 | 3,621.85 | +0.63% | 80.32 | 80.51 | +0.23% |
| Chat 4 | 3,361.95 | 3,389.82 | +0.83% | 93.89 | 97.43 | +3.77% |
| Chat 5 | 3,109.11 | 3,091.23 | -0.58% | 86.46 | 86.63 | +0.20% |
| Chat 6 | 2,733.36 | 2,736.43 | +0.11% | 83.22 | 83.12 | -0.12% |
| Chat 7 | 1,964.52 | 1,950.45 | -0.72% | 77.99 | 77.95 | -0.05% |
| Chat 8 | 476.43 | 440.31 | -7.58% | 75.33 | 75.28 | -0.08% |

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
baseline,1,1,64,64,128,0.061,1048.54,14.4,73.235
baseline,1,2,64,64,128,0.060,1066.57,14.4,73.252
baseline,1,3,64,64,128,0.060,1068.21,14.4,73.157
trial,1,1,64,64,128,0.060,1058.74,14.0,73.441
trial,1,2,64,64,128,0.060,1062.58,14.3,73.667
trial,1,3,64,64,128,0.061,1055.95,14.3,73.395
baseline,2,1,512,512,128,0.153,3337.88,14.6,73.007
baseline,2,2,512,512,128,0.155,3302.07,14.9,73.075
baseline,2,3,512,512,128,0.155,3299.29,14.7,72.945
trial,2,1,512,512,128,0.153,3339.35,14.6,73.288
trial,2,2,512,512,128,0.156,3283.28,14.8,73.397
trial,2,3,512,512,128,0.155,3299.05,15.0,73.155
baseline,3,1,64,64,128,0.049,1315.82,14.3,73.392
baseline,3,2,64,64,128,0.049,1314.66,14.1,73.541
baseline,3,3,64,64,128,0.049,1316.73,14.4,73.568
trial,3,1,64,64,128,0.048,1322.20,14.0,73.670
trial,3,2,64,64,128,0.048,1320.21,14.2,73.825
trial,3,3,64,64,128,0.049,1313.46,14.2,73.691
baseline,4,1,512,512,128,0.165,3106.23,14.8,73.215
baseline,4,2,512,512,128,0.164,3115.75,14.7,73.348
baseline,4,3,512,512,128,0.164,3116.29,14.8,73.313
trial,4,1,512,512,128,0.164,3123.33,14.7,73.273
trial,4,2,512,512,128,0.166,3089.13,15.0,73.468
trial,4,3,512,512,128,0.166,3083.60,15.1,73.293
baseline,5,1,2048,2048,128,0.538,3807.74,15.0,72.556
baseline,5,2,2048,2048,128,0.540,3791.20,14.5,72.486
baseline,5,3,2048,2048,128,0.542,3782.08,14.9,72.664
trial,5,1,2048,2048,128,0.536,3821.17,14.9,72.797
trial,5,2,2048,2048,128,0.539,3797.38,15.0,72.711
trial,5,3,2048,2048,128,0.541,3783.17,14.5,72.405
baseline,6,1,4096,4096,128,1.068,3836.86,14.6,71.381
baseline,6,2,4096,4096,128,1.077,3801.99,14.7,71.586
baseline,6,3,4096,4096,128,1.081,3787.62,14.6,71.620
trial,6,1,4096,4096,128,1.069,3830.14,14.6,71.822
trial,6,2,4096,4096,128,1.075,3809.05,14.6,71.681
trial,6,3,4096,4096,128,1.079,3794.52,14.7,71.468
baseline,7,1,8192,8192,128,2.176,3764.62,14.9,69.040
baseline,7,2,8192,8192,128,2.188,3744.39,15.0,68.972
baseline,7,3,8192,8192,128,2.188,3743.71,15.1,69.220
trial,7,1,8192,8192,128,2.171,3772.66,15.0,69.335
trial,7,2,8192,8192,128,2.185,3750.05,15.0,69.344
trial,7,3,8192,8192,128,2.193,3735.26,15.2,69.330
baseline,8,1,16384,16384,128,4.544,3605.91,15.7,65.956
baseline,8,2,16384,16384,128,4.574,3582.36,15.8,66.205
baseline,8,3,16384,16384,128,4.572,3583.92,15.7,66.206
trial,8,1,16384,16384,128,4.542,3607.27,15.7,66.328
trial,8,2,16384,16384,128,4.574,3581.69,15.7,66.202
trial,8,3,16384,16384,128,4.580,3577.40,15.8,66.239
baseline,9,1,663,663,57,0.335,1976.69,14.4,81.759
baseline,9,2,663,663,57,0.333,1993.79,14.3,83.358
baseline,9,3,663,663,57,0.333,1989.60,14.3,83.355
trial,9,1,663,663,57,0.335,1978.54,14.5,83.223
trial,9,2,663,663,57,0.332,1996.39,14.5,83.265
trial,9,3,663,663,57,0.335,1980.81,14.5,83.339
baseline,10,1,1954,1234,32,0.348,3542.39,14.5,93.807
baseline,10,2,1954,1234,32,0.349,3532.06,14.7,93.946
baseline,10,3,1954,1234,32,0.349,3531.46,15.0,93.559
trial,10,1,1954,1234,32,0.347,3555.21,14.7,93.517
trial,10,2,1954,1234,32,0.348,3550.40,14.5,94.254
trial,10,3,1954,1234,32,0.350,3527.41,14.6,94.036
baseline,11,1,4248,2262,87,0.624,3627.73,14.6,80.180
baseline,11,2,4248,2262,87,0.629,3596.40,14.6,80.323
baseline,11,3,4248,2262,87,0.628,3599.05,14.7,80.445
trial,11,1,4248,2262,87,0.625,3621.57,15.5,80.047
trial,11,2,4248,2262,87,0.625,3621.85,14.6,80.511
trial,11,3,4248,2262,87,0.624,3626.36,14.6,80.584
baseline,12,1,5741,1406,44,0.418,3361.95,14.9,93.527
baseline,12,2,5741,1406,44,0.419,3359.41,14.9,93.953
baseline,12,3,5741,1406,44,0.416,3380.13,15.3,93.889
trial,12,1,5741,1406,44,0.415,3389.82,15.0,93.338
trial,12,2,5741,1406,44,0.415,3390.15,15.4,97.600
trial,12,3,5741,1406,44,0.417,3369.38,15.3,97.427
baseline,13,1,7083,1298,54,0.419,3101.28,15.5,86.636
baseline,13,2,7083,1298,54,0.416,3116.48,15.1,86.456
baseline,13,3,7083,1298,54,0.417,3109.11,15.1,86.423
trial,13,1,7083,1298,54,0.420,3090.17,15.8,86.631
trial,13,2,7083,1298,54,0.415,3126.47,15.4,87.009
trial,13,3,7083,1298,54,0.420,3091.23,15.2,86.418
baseline,14,1,8041,904,24,0.332,2719.55,15.5,82.965
baseline,14,2,8041,904,24,0.330,2742.94,15.2,83.244
baseline,14,3,8041,904,24,0.331,2733.36,15.6,83.217
trial,14,1,8041,904,24,0.330,2739.71,15.6,83.116
trial,14,2,8041,904,24,0.330,2736.43,15.8,83.086
trial,14,3,8041,904,24,0.331,2732.96,15.3,83.183
baseline,15,1,8413,348,467,0.179,1943.43,15.4,77.987
baseline,15,2,8413,348,467,0.177,1964.52,15.6,77.827
baseline,15,3,8413,348,467,0.177,1966.77,15.4,78.122
trial,15,1,8413,348,467,0.179,1941.97,15.2,77.886
trial,15,2,8413,348,467,0.178,1950.45,15.4,78.188
trial,15,3,8413,348,467,0.178,1951.39,15.5,77.951
baseline,16,1,8915,35,186,0.073,476.43,14.8,75.460
baseline,16,2,8915,35,186,0.073,476.67,14.8,75.333
baseline,16,3,8915,35,186,0.074,472.23,14.6,75.152
trial,16,1,8915,35,186,0.080,438.42,14.8,75.276
trial,16,2,8915,35,186,0.079,440.63,14.7,75.599
trial,16,3,8915,35,186,0.079,440.31,14.5,73.658
```
