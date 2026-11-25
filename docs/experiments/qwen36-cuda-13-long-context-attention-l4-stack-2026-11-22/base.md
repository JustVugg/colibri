# NVIDIA L4 vs dev, full CUDA stack

These measurements compare the historical full stack with dev on the same physical GPU.

Baseline: dev (source 05245ef45048b837fee5ed7b5b1ab2e59161fb80, compiled c/ tree 34355d8f798ffb5fdf6c1330d14f592cb5909205).
Trial: full stack (source da49a1611a76bfdf840b07cd326c31c4733c0e18, compiled c/ tree e7cb67d79e678045d4f9aaeb9fa9f745e1d5443b).
Campaign dev reference: `05245ef45048b837fee5ed7b5b1ab2e59161fb80`. Model: `Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64@c619aa594ad1e70af82168fb6b4878427896e21c`.
The measurements predate the final packaging of the pull requests.
The full-stack benchmark kept the CUDA request open through generation and saved the chat prompt state at the first token.

One excluded warm-up per build and workload group. Median of three paired percentage changes. Each build contributes three retained runs per workload.
Groups preserve process history: 64 → 512 → 64; cold 512; 2,048 → 4,096 → 8,192 → 16,384; then a separate eight-turn chat replay.
The first decode step is timed separately. Decode throughput covers the remaining steps. A dagger marks a reported spread above 10% after the repeat.

| Workload | Baseline prefill tok/s | Trial prefill tok/s | Change | Baseline decode tok/s | Trial decode tok/s | Change |
|---|---:|---:|---:|---:|---:|---:|
| Cold 64 | 615.35 | 1,027.30 | +66.54% | 68.77 | 72.83 | +5.81% |
| 512 in sequence | 1,964.99 | 3,357.47 | +71.78% | 68.53 | 72.65 | +6.01% |
| Final 64 | 873.29 | 1,321.63 | +51.37% | 68.80 | 73.10 | +6.06% |
| Cold 512 | 1,798.52 | 3,139.89 | +73.42% | 69.22 | 73.34 | +5.94% |
| 2,048 | 2,307.49 | 3,768.56 | +63.34% | 68.64 | 72.99 | +6.47% |
| 4,096 | 2,335.56 | 3,748.41 | +60.49% | 67.48 | 71.68 | +6.58% |
| 8,192 | 2,541.98 | 3,683.80 | +44.51% | 64.92 | 69.04 | +6.54% |
| 16,384 | 2,342.42 | 3,514.72 | +50.03% | 61.55 | 66.28 | +7.66% |
| Chat 1 | 1,165.76 | 1,975.45 | +70.27% | 76.87 | 83.83 | +8.99% |
| Chat 2 | 1,896.27 | 3,497.79 | +84.65% | 74.78 | 95.23 | +26.61% |
| Chat 3 | 1,949.17 | 3,567.48 | +83.10% | 75.48 | 81.10 | +7.28% |
| Chat 4 | 1,956.10 | 3,323.66 | +69.91% | 91.62 | 95.65 | +5.71% |
| Chat 5 | 1,810.70 | 3,070.08 | +69.55% | 82.12 | 87.49 | +6.74% |
| Chat 6 | 1,865.54 | 2,709.34 | +46.39% | 73.19 | 83.97 | +14.51% |
| Chat 7 | 1,386.87 | 1,942.53 | +40.22% | 72.03 | 78.61 | +9.21% |
| Chat 8 | 309.11 | 439.57 | +41.40% | 70.07 | 73.44 | +5.58% |

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
baseline,1,1,64,64,128,0.104,615.35,57.3,68.769
baseline,1,2,64,64,128,0.105,609.45,51.6,68.613
baseline,1,3,64,64,128,0.103,621.31,50.7,68.826
trial,1,1,64,64,128,0.062,1024.81,14.3,72.714
trial,1,2,64,64,128,0.061,1047.94,14.2,72.902
trial,1,3,64,64,128,0.062,1027.30,14.3,72.828
baseline,2,1,512,512,128,0.260,1971.57,15.4,68.525
baseline,2,2,512,512,128,0.261,1964.99,15.3,68.367
baseline,2,3,512,512,128,0.264,1939.88,15.0,68.534
trial,2,1,512,512,128,0.152,3357.47,14.6,72.462
trial,2,2,512,512,128,0.152,3375.41,14.7,72.799
trial,2,3,512,512,128,0.153,3357.37,14.8,72.650
baseline,3,1,64,64,128,0.073,873.29,15.6,68.801
baseline,3,2,64,64,128,0.074,861.55,15.5,68.792
baseline,3,3,64,64,128,0.073,876.31,15.8,68.958
trial,3,1,64,64,128,0.048,1321.87,14.1,72.967
trial,3,2,64,64,128,0.048,1321.63,14.0,73.346
trial,3,3,64,64,128,0.049,1319.52,14.3,73.101
baseline,4,1,512,512,128,0.285,1797.53,49.7,68.402
baseline,4,2,512,512,128,0.285,1798.52,49.6,69.225
baseline,4,3,512,512,128,0.282,1813.12,49.1,69.395
trial,4,1,512,512,128,0.164,3117.25,14.7,72.862
trial,4,2,512,512,128,0.157,3256.68,14.8,73.339
trial,4,3,512,512,128,0.163,3139.89,14.6,73.362
baseline,5,1,2048,2048,128,0.891,2298.98,47.5,68.657
baseline,5,2,2048,2048,128,0.888,2307.49,47.1,68.639
baseline,5,3,2048,2048,128,0.884,2315.86,47.7,68.368
trial,5,1,2048,2048,128,0.543,3768.56,14.6,72.807
trial,5,2,2048,2048,128,0.543,3769.03,14.6,73.080
trial,5,3,2048,2048,128,0.551,3719.30,15.0,72.986
baseline,6,1,4096,4096,128,1.758,2330.17,14.8,67.612
baseline,6,2,4096,4096,128,1.742,2351.19,14.9,66.849
baseline,6,3,4096,4096,128,1.754,2335.56,14.9,67.476
trial,6,1,4096,4096,128,1.088,3763.85,14.6,71.679
trial,6,2,4096,4096,128,1.094,3744.49,14.7,71.528
trial,6,3,4096,4096,128,1.093,3748.41,14.5,71.919
baseline,7,1,8192,8192,128,3.223,2541.98,15.2,64.924
baseline,7,2,8192,8192,128,3.204,2556.68,15.2,64.804
baseline,7,3,8192,8192,128,3.229,2536.90,15.2,65.211
trial,7,1,8192,8192,128,2.224,3683.80,14.8,69.034
trial,7,2,8192,8192,128,2.217,3694.64,14.9,69.043
trial,7,3,8192,8192,128,2.236,3663.60,14.8,69.515
baseline,8,1,16384,16384,128,6.999,2340.92,19.4,61.550
baseline,8,2,16384,16384,128,6.994,2342.68,16.2,61.233
baseline,8,3,16384,16384,128,6.994,2342.42,16.1,61.674
trial,8,1,16384,16384,128,4.651,3522.70,15.7,66.386
trial,8,2,16384,16384,128,4.662,3514.72,15.6,65.925
trial,8,3,16384,16384,128,4.685,3497.07,15.7,66.279
baseline,9,1,663,663,57,0.571,1160.21,109.6,76.504
baseline,9,2,663,663,57,0.569,1165.76,104.9,77.000
baseline,9,3,663,663,57,0.566,1171.25,107.4,76.867
trial,9,1,663,663,57,0.336,1975.45,44.3,83.832
trial,9,2,663,663,57,0.332,1997.25,44.6,83.895
trial,9,3,663,663,57,0.336,1975.07,44.5,83.777
baseline,10,1,1954,1234,32,0.654,1888.07,83.1,74.778
baseline,10,2,1954,1234,32,0.650,1898.96,80.6,75.714
baseline,10,3,1954,1234,32,0.651,1896.27,81.6,73.947
trial,10,1,1954,1234,32,0.354,3486.28,44.5,94.673
trial,10,2,1954,1234,32,0.351,3512.23,45.9,95.653
trial,10,3,1954,1234,32,0.353,3497.79,45.7,95.227
baseline,11,1,4248,2262,87,1.161,1948.38,46.2,75.433
baseline,11,2,4248,2262,87,1.160,1949.17,45.5,76.306
baseline,11,3,4248,2262,87,1.150,1967.66,45.7,75.480
trial,11,1,4248,2262,87,0.634,3567.48,44.7,80.922
trial,11,2,4248,2262,87,0.629,3593.35,45.4,81.104
trial,11,3,4248,2262,87,0.635,3563.31,44.9,81.165
baseline,12,1,5741,1406,44,0.719,1956.10,46.4,90.478
baseline,12,2,5741,1406,44,0.719,1956.27,45.7,92.618
baseline,12,3,5741,1406,44,0.730,1926.84,46.1,91.623
trial,12,1,5741,1406,44,0.423,3323.66,45.4,95.646
trial,12,2,5741,1406,44,0.425,3312.05,45.8,94.786
trial,12,3,5741,1406,44,0.418,3367.08,44.8,97.106
baseline,13,1,7083,1298,54,0.722,1798.26,81.4,81.683
baseline,13,2,7083,1298,54,0.716,1812.06,46.4,82.135
baseline,13,3,7083,1298,54,0.717,1810.70,47.0,82.121
trial,13,1,7083,1298,54,0.427,3037.21,79.3,87.190
trial,13,2,7083,1298,54,0.422,3078.44,46.0,87.487
trial,13,3,7083,1298,54,0.423,3070.08,76.5,87.828
baseline,14,1,8041,904,24,0.491,1839.93,45.4,73.191
baseline,14,2,8041,904,24,0.485,1865.54,45.5,73.508
baseline,14,3,8041,904,24,0.483,1873.48,46.5,70.423
trial,14,1,8041,904,24,0.336,2693.51,45.4,83.511
trial,14,2,8041,904,24,0.327,2763.88,45.4,84.176
trial,14,3,8041,904,24,0.334,2709.34,45.7,83.965
baseline,15,1,8413,348,467,0.247,1411.66,78.2,71.635
baseline,15,2,8413,348,467,0.251,1386.87,78.7,72.035
baseline,15,3,8413,348,467,0.252,1380.39,79.2,72.108
trial,15,1,8413,348,467,0.181,1921.19,75.0,78.600
trial,15,2,8413,348,467,0.179,1944.69,75.9,78.670
trial,15,3,8413,348,467,0.179,1942.53,73.7,78.607
baseline,16,1,8915,35,186,0.113,310.86,46.0,70.129
baseline,16,2,8915,35,186,0.113,309.11,48.5,69.557
baseline,16,3,8915,35,186,0.114,307.17,78.0,70.067
trial,16,1,8915,35,186,0.080,439.57,46.0,73.068
trial,16,2,8915,35,186,0.081,433.56,47.0,73.438
trial,16,3,8915,35,186,0.080,440.21,45.1,75.304
```
