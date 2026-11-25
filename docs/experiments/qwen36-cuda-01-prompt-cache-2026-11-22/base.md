# L4 incremental measurement · Grow the Qwen3.6 prompt cache on NVIDIA L4

These measurements compare consecutive cumulative builds on the same physical L4. Each build was also checked against the original dev reference.

Baseline: dev (source e677aaee4424d27b8eb0a7e7d7edbb32a309c267, compiled c/ tree c46a68bb61e65d111ba436bef2751c8fb03df6cd).
Trial: pr01 (source 617ae5eb22dd08852c25e0d8cb97772395d03d2c, compiled c/ tree 903b01d2b085e08923985ebb7cfeaaa648d9cd17).
Campaign dev reference: `e677aaee4424d27b8eb0a7e7d7edbb32a309c267`. Model: `Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64@c619aa594ad1e70af82168fb6b4878427896e21c`.
The measurements predate the final packaging of the pull requests.
This series used the earlier benchmark. Its chat first-step timings exclude the server prompt-state save. Fixed-length decode used the request handling of that benchmark.

One reference warm-up per group on dev before the series. Cumulative builds ran three measured repetitions per group. Percentage change between the two build medians. Each build contributes three retained runs per workload.
Groups preserve process history: 64 → 512 → 64; cold 512; 2,048 → 4,096 → 8,192 → 16,384; then a separate eight-turn chat replay.
The first decode step is timed separately. Decode throughput covers the remaining steps. A dagger marks a reported spread above 10% after the repeat.

| Workload | Baseline prefill tok/s | Trial prefill tok/s | Change | Baseline decode tok/s | Trial decode tok/s | Change |
|---|---:|---:|---:|---:|---:|---:|
| Cold 64 | 625.23 | 851.67 | +36.22% | 69.59 | 71.34 | +2.52% |
| 512 in sequence | 1,891.68 | 2,284.21 | +20.75% | 69.27 | 71.17 | +2.74% |
| Final 64 | 841.37 | 966.93 | +14.92% | 69.63 | 71.36 | +2.48% |
| Cold 512 | 1,767.04 | 2,179.94 | +23.37% | 69.48 | 70.97 | +2.13% |
| 2,048 | 2,293.78 | 2,744.15 | +19.63% | 68.97 | 70.54 | +2.28% |
| 4,096 | 2,307.13 | 2,768.38 | +19.99% | 67.69 | 69.55 | +2.74% |
| 8,192 | 2,520.96 | 2,603.68 | +3.28% | 65.26 | 67.29 | +3.11% |
| 16,384 | 2,311.93 | 2,357.98 | +1.99% | 61.89 | 64.41 | +4.07% |
| Chat 1 | 1,150.84 | 1,496.06 | +30.00% | 77.14 | 79.13 | +2.57% |
| Chat 2 | 1,869.18 | 2,503.96 | +33.96% | 75.63 | 94.26 | +24.64% |
| Chat 3 | 1,921.80 | 2,491.82 | +29.66% | 75.72 | 78.80 | +4.06% |
| Chat 4 | 1,910.53 | 2,238.78 | +17.18% | 92.57 | 93.12 | +0.60% |
| Chat 5 | 1,827.99 | 1,935.38 | +5.87% | 82.38 | 85.36 | +3.61% |
| Chat 6 | 1,878.27 | 1,771.37 | -5.69% | 73.38 | 80.00 | +9.03% |
| Chat 7 | 1,385.71 | 1,242.85 | -10.31% | 72.60 | 75.09 | +3.43% |
| Chat 8 | 309.45 | 337.26 | +8.99% | 69.98 | 71.86 | +2.68% |

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
baseline,1,1,64,64,128,0.102,627.36,49.0,69.873
baseline,1,2,64,64,128,0.103,620.44,48.8,69.549
baseline,1,3,64,64,128,0.102,625.23,48.6,69.589
trial,1,1,64,64,128,0.075,851.67,49.2,71.264
trial,1,2,64,64,128,0.074,860.22,47.3,71.388
trial,1,3,64,64,128,0.075,850.81,49.4,71.341
baseline,2,1,512,512,128,0.271,1891.68,15.6,69.535
baseline,2,2,512,512,128,0.273,1878.27,15.2,69.267
baseline,2,3,512,512,128,0.268,1907.80,15.8,69.267
trial,2,1,512,512,128,0.224,2284.21,15.3,71.042
trial,2,2,512,512,128,0.226,2265.81,15.2,71.167
trial,2,3,512,512,128,0.222,2310.03,14.8,71.206
baseline,3,1,64,64,128,0.072,884.93,15.4,70.042
baseline,3,2,64,64,128,0.076,841.37,16.3,69.438
baseline,3,3,64,64,128,0.076,839.77,15.6,69.632
trial,3,1,64,64,128,0.065,981.08,15.4,71.134
trial,3,2,64,64,128,0.066,966.93,15.5,71.393
trial,3,3,64,64,128,0.067,953.89,15.5,71.362
baseline,4,1,512,512,128,0.287,1784.08,47.3,69.484
baseline,4,2,512,512,128,0.293,1745.65,49.2,69.675
baseline,4,3,512,512,128,0.290,1767.04,48.6,69.316
trial,4,1,512,512,128,0.233,2194.64,48.3,70.965
trial,4,2,512,512,128,0.236,2173.18,47.0,71.107
trial,4,3,512,512,128,0.235,2179.94,47.9,70.592
baseline,5,1,2048,2048,128,0.891,2299.09,47.0,68.967
baseline,5,2,2048,2048,128,0.901,2273.26,48.7,68.966
baseline,5,3,2048,2048,128,0.893,2293.78,47.2,68.901
trial,5,1,2048,2048,128,0.745,2748.90,47.2,70.540
trial,5,2,2048,2048,128,0.746,2744.15,46.4,70.595
trial,5,3,2048,2048,128,0.751,2725.99,47.4,70.090
baseline,6,1,4096,4096,128,1.764,2321.59,14.8,67.688
baseline,6,2,4096,4096,128,1.775,2307.13,14.8,67.940
baseline,6,3,4096,4096,128,1.781,2299.51,14.8,67.695
trial,6,1,4096,4096,128,1.469,2787.76,14.5,69.628
trial,6,2,4096,4096,128,1.490,2749.51,14.5,69.547
trial,6,3,4096,4096,128,1.480,2768.38,14.5,68.977
baseline,7,1,8192,8192,128,3.235,2532.55,15.2,65.258
baseline,7,2,8192,8192,128,3.250,2520.96,15.3,65.528
baseline,7,3,8192,8192,128,3.262,2511.54,15.4,65.114
trial,7,1,8192,8192,128,3.119,2626.52,14.9,67.396
trial,7,2,8192,8192,128,3.146,2603.68,14.8,67.286
trial,7,3,8192,8192,128,3.147,2602.76,14.8,66.580
baseline,8,1,16384,16384,128,7.043,2326.12,16.1,61.889
baseline,8,2,16384,16384,128,7.087,2311.93,16.5,61.931
baseline,8,3,16384,16384,128,7.104,2306.25,16.2,61.202
trial,8,1,16384,16384,128,6.869,2385.14,15.6,64.407
trial,8,2,16384,16384,128,6.954,2356.02,16.0,64.411
trial,8,3,16384,16384,128,6.948,2357.98,15.5,63.837
baseline,9,1,663,663,57,0.575,1153.87,50.6,77.144
baseline,9,2,663,663,57,0.576,1150.84,50.2,77.148
baseline,9,3,663,663,57,0.576,1150.11,50.8,75.644
trial,9,1,663,663,57,0.443,1496.06,48.9,79.126
trial,9,2,663,663,57,0.444,1493.01,49.0,79.461
trial,9,3,663,663,57,0.443,1497.61,47.7,78.492
baseline,10,1,1954,1234,32,0.662,1864.45,14.9,75.626
baseline,10,2,1954,1234,32,0.658,1875.93,14.9,74.522
baseline,10,3,1954,1234,32,0.660,1869.18,14.9,75.679
trial,10,1,1954,1234,32,0.491,2511.24,14.9,94.904
trial,10,2,1954,1234,32,0.495,2491.86,14.8,94.257
trial,10,3,1954,1234,32,0.493,2503.96,14.9,94.066
baseline,11,1,4248,2262,87,1.178,1920.23,14.9,75.722
baseline,11,2,4248,2262,87,1.177,1921.80,15.0,75.876
baseline,11,3,4248,2262,87,1.171,1931.66,14.9,74.627
trial,11,1,4248,2262,87,0.906,2496.48,14.6,78.799
trial,11,2,4248,2262,87,0.912,2480.70,14.7,78.810
trial,11,3,4248,2262,87,0.908,2491.82,14.5,78.059
baseline,12,1,5741,1406,44,0.732,1920.44,15.3,93.189
baseline,12,2,5741,1406,44,0.736,1910.53,15.2,91.459
baseline,12,3,5741,1406,44,0.736,1910.14,15.3,92.573
trial,12,1,5741,1406,44,0.628,2238.78,15.0,95.200
trial,12,2,5741,1406,44,0.630,2233.41,14.9,93.125
trial,12,3,5741,1406,44,0.627,2242.84,14.9,92.233
baseline,13,1,7083,1298,54,0.723,1795.13,15.5,82.382
baseline,13,2,7083,1298,54,0.710,1827.99,15.5,82.675
baseline,13,3,7083,1298,54,0.710,1828.29,15.6,81.395
trial,13,1,7083,1298,54,0.664,1955.68,15.2,85.430
trial,13,2,7083,1298,54,0.676,1920.75,15.1,85.355
trial,13,3,7083,1298,54,0.671,1935.38,15.1,84.237
baseline,14,1,8041,904,24,0.483,1872.99,15.7,73.471
baseline,14,2,8041,904,24,0.481,1881.27,15.6,73.379
baseline,14,3,8041,904,24,0.481,1878.27,15.7,72.502
trial,14,1,8041,904,24,0.509,1775.55,15.3,80.002
trial,14,2,8041,904,24,0.510,1771.37,15.4,80.298
trial,14,3,8041,904,24,0.511,1770.57,15.4,78.291
baseline,15,1,8413,348,467,0.251,1385.71,16.1,72.635
baseline,15,2,8413,348,467,0.249,1399.24,16.3,72.602
baseline,15,3,8413,348,467,0.253,1376.98,16.1,71.631
trial,15,1,8413,348,467,0.280,1242.85,15.6,75.092
trial,15,2,8413,348,467,0.276,1258.84,15.6,76.210
trial,15,3,8413,348,467,0.282,1232.17,15.6,74.654
baseline,16,1,8915,35,186,0.114,308.00,16.0,70.006
baseline,16,2,8915,35,186,0.113,309.45,15.9,69.983
baseline,16,3,8915,35,186,0.113,310.41,16.2,67.727
trial,16,1,8915,35,186,0.104,338.06,15.5,71.857
trial,16,2,8915,35,186,0.104,336.67,15.4,72.266
trial,16,3,8915,35,186,0.104,337.26,15.5,71.273
```
