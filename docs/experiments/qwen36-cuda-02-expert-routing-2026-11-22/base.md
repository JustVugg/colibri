# L4 incremental measurement · Specialize CUDA routing for 256 experts

These measurements compare consecutive cumulative builds on the same physical L4. Each build was also checked against the original dev reference.

Baseline: pr01 (source 617ae5eb22dd08852c25e0d8cb97772395d03d2c, compiled c/ tree 903b01d2b085e08923985ebb7cfeaaa648d9cd17).
Trial: pr02 (source 71013f621b4585548b6fc98dcdae08b8f0a60af6, compiled c/ tree fa4073cc7b863f6dfba557db8c38108275a565b7).
Campaign dev reference: `e677aaee4424d27b8eb0a7e7d7edbb32a309c267`. Model: `Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64@c619aa594ad1e70af82168fb6b4878427896e21c`.
The measurements predate the final packaging of the pull requests.
This series used the earlier benchmark. Its chat first-step timings exclude the server prompt-state save. Fixed-length decode used the request handling of that benchmark.

One reference warm-up per group on dev before the series. Cumulative builds ran three measured repetitions per group. Percentage change between the two build medians. Each build contributes three retained runs per workload.
Groups preserve process history: 64 → 512 → 64; cold 512; 2,048 → 4,096 → 8,192 → 16,384; then a separate eight-turn chat replay.
The first decode step is timed separately. Decode throughput covers the remaining steps. A dagger marks a reported spread above 10% after the repeat.

| Workload | Baseline prefill tok/s | Trial prefill tok/s | Change | Baseline decode tok/s | Trial decode tok/s | Change |
|---|---:|---:|---:|---:|---:|---:|
| Cold 64 | 851.67 | 864.04 | +1.45% | 71.34 | 71.60 | +0.36% |
| 512 in sequence | 2,284.21 | 2,264.17 | -0.88% | 71.17 | 71.27 | +0.15% |
| Final 64 | 966.93 | 965.45 | -0.15% | 71.36 | 71.55 | +0.26% |
| Cold 512 | 2,179.94 | 2,208.96 | +1.33% | 70.97 | 71.44 | +0.67% |
| 2,048 | 2,744.15 | 2,764.08 | +0.73% | 70.54 | 70.84 | +0.43% |
| 4,096 | 2,768.38 | 2,791.97 | +0.85% | 69.55 | 69.85 | +0.44% |
| 8,192 | 2,603.68 | 2,624.70 | +0.81% | 67.29 | 67.80 | +0.77% |
| 16,384 | 2,357.98 | 2,382.33 | +1.03% | 64.41 | 64.64 | +0.37% |
| Chat 1 | 1,496.06 | 1,503.72 | +0.51% | 79.13 | 79.64 | +0.65% |
| Chat 2 | 2,503.96 | 2,519.02 | +0.60% | 94.26 | 95.29 | +1.09% |
| Chat 3 | 2,491.82 | 2,499.83 | +0.32% | 78.80 | 78.71 | -0.11% |
| Chat 4 | 2,238.78 | 2,247.06 | +0.37% | 93.12 | 94.85 | +1.85% |
| Chat 5 | 1,935.38 | 1,971.07 | +1.84% | 85.36 | 85.53 | +0.21% |
| Chat 6 | 1,771.37 | 1,779.73 | +0.47% | 80.00 | 80.07 | +0.09% |
| Chat 7 | 1,242.85 | 1,254.24 | +0.92% | 75.09 | 75.35 | +0.34% |
| Chat 8 | 337.26 | 340.20 | +0.87% | 71.86 | 70.23 | -2.26% |

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
baseline,1,1,64,64,128,0.075,851.67,49.2,71.264
baseline,1,2,64,64,128,0.074,860.22,47.3,71.388
baseline,1,3,64,64,128,0.075,850.81,49.4,71.341
trial,1,1,64,64,128,0.074,865.02,47.7,71.410
trial,1,2,64,64,128,0.074,864.04,47.4,71.601
trial,1,3,64,64,128,0.075,848.26,47.9,71.621
baseline,2,1,512,512,128,0.224,2284.21,15.3,71.042
baseline,2,2,512,512,128,0.226,2265.81,15.2,71.167
baseline,2,3,512,512,128,0.222,2310.03,14.8,71.206
trial,2,1,512,512,128,0.224,2286.74,14.8,71.254
trial,2,2,512,512,128,0.230,2221.91,15.1,71.457
trial,2,3,512,512,128,0.226,2264.17,15.1,71.271
baseline,3,1,64,64,128,0.065,981.08,15.4,71.134
baseline,3,2,64,64,128,0.066,966.93,15.5,71.393
baseline,3,3,64,64,128,0.067,953.89,15.5,71.362
trial,3,1,64,64,128,0.065,982.92,15.4,71.577
trial,3,2,64,64,128,0.066,965.45,15.4,71.549
trial,3,3,64,64,128,0.066,963.08,15.5,71.502
baseline,4,1,512,512,128,0.233,2194.64,48.3,70.965
baseline,4,2,512,512,128,0.236,2173.18,47.0,71.107
baseline,4,3,512,512,128,0.235,2179.94,47.9,70.592
trial,4,1,512,512,128,0.232,2208.96,47.5,71.439
trial,4,2,512,512,128,0.231,2217.38,47.0,71.478
trial,4,3,512,512,128,0.232,2206.89,47.0,71.285
baseline,5,1,2048,2048,128,0.745,2748.90,47.2,70.540
baseline,5,2,2048,2048,128,0.746,2744.15,46.4,70.595
baseline,5,3,2048,2048,128,0.751,2725.99,47.4,70.090
trial,5,1,2048,2048,128,0.737,2778.05,47.1,70.856
trial,5,2,2048,2048,128,0.741,2762.62,47.1,70.831
trial,5,3,2048,2048,128,0.741,2764.08,47.0,70.841
baseline,6,1,4096,4096,128,1.469,2787.76,14.5,69.628
baseline,6,2,4096,4096,128,1.490,2749.51,14.5,69.547
baseline,6,3,4096,4096,128,1.480,2768.38,14.5,68.977
trial,6,1,4096,4096,128,1.459,2808.02,14.4,69.750
trial,6,2,4096,4096,128,1.472,2781.89,14.4,69.852
trial,6,3,4096,4096,128,1.467,2791.97,14.4,69.964
baseline,7,1,8192,8192,128,3.119,2626.52,14.9,67.396
baseline,7,2,8192,8192,128,3.146,2603.68,14.8,67.286
baseline,7,3,8192,8192,128,3.147,2602.76,14.8,66.580
trial,7,1,8192,8192,128,3.085,2655.07,14.8,67.888
trial,7,2,8192,8192,128,3.121,2624.70,14.8,67.801
trial,7,3,8192,8192,128,3.121,2624.53,14.8,67.783
baseline,8,1,16384,16384,128,6.869,2385.14,15.6,64.407
baseline,8,2,16384,16384,128,6.954,2356.02,16.0,64.411
baseline,8,3,16384,16384,128,6.948,2357.98,15.5,63.837
trial,8,1,16384,16384,128,6.826,2400.19,15.5,64.706
trial,8,2,16384,16384,128,6.877,2382.33,15.5,64.644
trial,8,3,16384,16384,128,6.890,2377.89,15.9,64.594
baseline,9,1,663,663,57,0.443,1496.06,48.9,79.126
baseline,9,2,663,663,57,0.444,1493.01,49.0,79.461
baseline,9,3,663,663,57,0.443,1497.61,47.7,78.492
trial,9,1,663,663,57,0.441,1503.72,48.7,79.370
trial,9,2,663,663,57,0.441,1504.18,48.2,79.700
trial,9,3,663,663,57,0.442,1498.40,47.8,79.640
baseline,10,1,1954,1234,32,0.491,2511.24,14.9,94.904
baseline,10,2,1954,1234,32,0.495,2491.86,14.8,94.257
baseline,10,3,1954,1234,32,0.493,2503.96,14.9,94.066
trial,10,1,1954,1234,32,0.488,2530.99,14.7,95.455
trial,10,2,1954,1234,32,0.494,2499.09,14.7,95.079
trial,10,3,1954,1234,32,0.490,2519.02,14.7,95.288
baseline,11,1,4248,2262,87,0.906,2496.48,14.6,78.799
baseline,11,2,4248,2262,87,0.912,2480.70,14.7,78.810
baseline,11,3,4248,2262,87,0.908,2491.82,14.5,78.059
trial,11,1,4248,2262,87,0.916,2469.18,14.6,78.711
trial,11,2,4248,2262,87,0.905,2499.83,14.5,78.752
trial,11,3,4248,2262,87,0.904,2502.44,14.5,78.646
baseline,12,1,5741,1406,44,0.628,2238.78,15.0,95.200
baseline,12,2,5741,1406,44,0.630,2233.41,14.9,93.125
baseline,12,3,5741,1406,44,0.627,2242.84,14.9,92.233
trial,12,1,5741,1406,44,0.625,2250.15,14.8,92.768
trial,12,2,5741,1406,44,0.626,2247.06,14.8,94.847
trial,12,3,5741,1406,44,0.626,2245.81,14.8,94.930
baseline,13,1,7083,1298,54,0.664,1955.68,15.2,85.430
baseline,13,2,7083,1298,54,0.676,1920.75,15.1,85.355
baseline,13,3,7083,1298,54,0.671,1935.38,15.1,84.237
trial,13,1,7083,1298,54,0.669,1941.50,15.0,85.688
trial,13,2,7083,1298,54,0.659,1971.07,15.0,85.533
trial,13,3,7083,1298,54,0.655,1982.88,15.1,85.349
baseline,14,1,8041,904,24,0.509,1775.55,15.3,80.002
baseline,14,2,8041,904,24,0.510,1771.37,15.4,80.298
baseline,14,3,8041,904,24,0.511,1770.57,15.4,78.291
trial,14,1,8041,904,24,0.504,1795.29,15.2,80.387
trial,14,2,8041,904,24,0.508,1779.73,15.2,79.745
trial,14,3,8041,904,24,0.509,1777.35,15.3,80.071
baseline,15,1,8413,348,467,0.280,1242.85,15.6,75.092
baseline,15,2,8413,348,467,0.276,1258.84,15.6,76.210
baseline,15,3,8413,348,467,0.282,1232.17,15.6,74.654
trial,15,1,8413,348,467,0.282,1235.89,15.6,75.437
trial,15,2,8413,348,467,0.275,1266.28,15.5,74.668
trial,15,3,8413,348,467,0.277,1254.24,15.5,75.348
baseline,16,1,8915,35,186,0.104,338.06,15.5,71.857
baseline,16,2,8915,35,186,0.104,336.67,15.4,72.266
baseline,16,3,8915,35,186,0.104,337.26,15.5,71.273
trial,16,1,8915,35,186,0.103,340.68,15.4,70.231
trial,16,2,8915,35,186,0.103,339.11,15.4,71.265
trial,16,3,8915,35,186,0.103,340.20,15.3,69.940
```
