# L4 incremental measurement · Tile prompt attention with Tensor Cores

These measurements compare consecutive cumulative builds on the same physical L4. Each build was also checked against the original dev reference.

Baseline: pr02 (source 71013f621b4585548b6fc98dcdae08b8f0a60af6, compiled c/ tree fa4073cc7b863f6dfba557db8c38108275a565b7).
Trial: pr03 (source 5040b7e784ac013b3af894aab75ef3eaea4dda8d, compiled c/ tree 66dcbe713f33530310f6388c01eebcb3983133e9).
Campaign dev reference: `e677aaee4424d27b8eb0a7e7d7edbb32a309c267`. Model: `Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64@c619aa594ad1e70af82168fb6b4878427896e21c`.
The measurements predate the final packaging of the pull requests.
This series used the earlier benchmark. Its chat first-step timings exclude the server prompt-state save. Fixed-length decode used the request handling of that benchmark.

One reference warm-up per group on dev before the series. Cumulative builds ran three measured repetitions per group. Percentage change between the two build medians. Each build contributes three retained runs per workload.
Groups preserve process history: 64 → 512 → 64; cold 512; 2,048 → 4,096 → 8,192 → 16,384; then a separate eight-turn chat replay.
The first decode step is timed separately. Decode throughput covers the remaining steps. A dagger marks a reported spread above 10% after the repeat.

| Workload | Baseline prefill tok/s | Trial prefill tok/s | Change | Baseline decode tok/s | Trial decode tok/s | Change |
|---|---:|---:|---:|---:|---:|---:|
| Cold 64 | 864.04 | 855.46 | -0.99% | 71.60 | 71.49 | -0.15% |
| 512 in sequence | 2,264.17 | 2,275.76 | +0.51% | 71.27 | 71.26 | -0.02% |
| Final 64 | 965.45 | 978.20 | +1.32% | 71.55 | 71.50 | -0.07% |
| Cold 512 | 2,208.96 | 2,206.03 | -0.13% | 71.44 | 71.19 | -0.34% |
| 2,048 | 2,764.08 | 2,797.17 | +1.20% | 70.84 | 70.67 | -0.24% |
| 4,096 | 2,791.97 | 2,866.81 | +2.68% | 69.85 | 69.72 | -0.19% |
| 8,192 | 2,624.70 | 2,769.76 | +5.53% | 67.80 | 67.57 | -0.34% |
| 16,384 | 2,382.33 | 2,651.39 | +11.29% | 64.64 | 64.53 | -0.17% |
| Chat 1 | 1,503.72 | 1,509.68 | +0.40% | 79.64 | 79.34 | -0.38% |
| Chat 2 | 2,519.02 | 2,543.25 | +0.96% | 95.29 | 95.21 | -0.09% |
| Chat 3 | 2,499.83 | 2,552.14 | +2.09% | 78.71 | 78.58 | -0.16% |
| Chat 4 | 2,247.06 | 2,417.64 | +7.59% | 94.85 | 96.12 | +1.34% |
| Chat 5 | 1,971.07 | 2,121.29 | +7.62% | 85.53 | 85.01 | -0.61% |
| Chat 6 | 1,779.73 | 1,924.97 | +8.16% | 80.07 | 80.18 | +0.14% |
| Chat 7 | 1,254.24 | 1,330.11 | +6.05% | 75.35 | 75.07 | -0.36% |
| Chat 8 | 340.20 | 400.67 | +17.77% | 70.23 | 70.11 | -0.18% |

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
baseline,1,1,64,64,128,0.074,865.02,47.7,71.410
baseline,1,2,64,64,128,0.074,864.04,47.4,71.601
baseline,1,3,64,64,128,0.075,848.26,47.9,71.621
trial,1,1,64,64,128,0.075,855.46,48.1,71.492
trial,1,2,64,64,128,0.074,860.10,47.2,71.484
trial,1,3,64,64,128,0.077,832.93,48.3,71.576
baseline,2,1,512,512,128,0.224,2286.74,14.8,71.254
baseline,2,2,512,512,128,0.230,2221.91,15.1,71.457
baseline,2,3,512,512,128,0.226,2264.17,15.1,71.271
trial,2,1,512,512,128,0.222,2307.17,14.7,71.257
trial,2,2,512,512,128,0.225,2275.76,15.2,71.218
trial,2,3,512,512,128,0.226,2262.68,14.7,71.294
baseline,3,1,64,64,128,0.065,982.92,15.4,71.577
baseline,3,2,64,64,128,0.066,965.45,15.4,71.549
baseline,3,3,64,64,128,0.066,963.08,15.5,71.502
trial,3,1,64,64,128,0.065,985.53,15.3,71.533
trial,3,2,64,64,128,0.065,978.20,15.2,71.383
trial,3,3,64,64,128,0.066,967.88,15.4,71.499
baseline,4,1,512,512,128,0.232,2208.96,47.5,71.439
baseline,4,2,512,512,128,0.231,2217.38,47.0,71.478
baseline,4,3,512,512,128,0.232,2206.89,47.0,71.285
trial,4,1,512,512,128,0.231,2217.07,48.2,71.193
trial,4,2,512,512,128,0.232,2206.03,46.8,71.339
trial,4,3,512,512,128,0.234,2184.95,47.8,71.142
baseline,5,1,2048,2048,128,0.737,2778.05,47.1,70.856
baseline,5,2,2048,2048,128,0.741,2762.62,47.1,70.831
baseline,5,3,2048,2048,128,0.741,2764.08,47.0,70.841
trial,5,1,2048,2048,128,0.727,2816.96,47.2,70.668
trial,5,2,2048,2048,128,0.732,2797.17,46.6,70.735
trial,5,3,2048,2048,128,0.735,2785.93,48.5,69.844
baseline,6,1,4096,4096,128,1.459,2808.02,14.4,69.750
baseline,6,2,4096,4096,128,1.472,2781.89,14.4,69.852
baseline,6,3,4096,4096,128,1.467,2791.97,14.4,69.964
trial,6,1,4096,4096,128,1.401,2922.76,14.5,69.696
trial,6,2,4096,4096,128,1.430,2864.74,14.5,69.722
trial,6,3,4096,4096,128,1.429,2866.81,14.4,69.813
baseline,7,1,8192,8192,128,3.085,2655.07,14.8,67.888
baseline,7,2,8192,8192,128,3.121,2624.70,14.8,67.801
baseline,7,3,8192,8192,128,3.121,2624.53,14.8,67.783
trial,7,1,8192,8192,128,2.901,2823.74,14.8,67.570
trial,7,2,8192,8192,128,2.960,2767.26,14.9,67.559
trial,7,3,8192,8192,128,2.958,2769.76,14.8,67.603
baseline,8,1,16384,16384,128,6.826,2400.19,15.5,64.706
baseline,8,2,16384,16384,128,6.877,2382.33,15.5,64.644
baseline,8,3,16384,16384,128,6.890,2377.89,15.9,64.594
trial,8,1,16384,16384,128,6.095,2688.22,15.5,64.570
trial,8,2,16384,16384,128,6.179,2651.39,15.6,64.375
trial,8,3,16384,16384,128,6.197,2644.07,15.5,64.532
baseline,9,1,663,663,57,0.441,1503.72,48.7,79.370
baseline,9,2,663,663,57,0.441,1504.18,48.2,79.700
baseline,9,3,663,663,57,0.442,1498.40,47.8,79.640
trial,9,1,663,663,57,0.435,1522.60,49.0,79.340
trial,9,2,663,663,57,0.442,1498.62,47.7,79.093
trial,9,3,663,663,57,0.439,1509.68,47.4,79.440
baseline,10,1,1954,1234,32,0.488,2530.99,14.7,95.455
baseline,10,2,1954,1234,32,0.494,2499.09,14.7,95.079
baseline,10,3,1954,1234,32,0.490,2519.02,14.7,95.288
trial,10,1,1954,1234,32,0.481,2567.11,14.7,95.554
trial,10,2,1954,1234,32,0.485,2543.25,14.7,94.676
trial,10,3,1954,1234,32,0.486,2541.35,14.7,95.207
baseline,11,1,4248,2262,87,0.916,2469.18,14.6,78.711
baseline,11,2,4248,2262,87,0.905,2499.83,14.5,78.752
baseline,11,3,4248,2262,87,0.904,2502.44,14.5,78.646
trial,11,1,4248,2262,87,0.878,2575.23,14.5,78.582
trial,11,2,4248,2262,87,0.886,2552.14,14.5,78.238
trial,11,3,4248,2262,87,0.899,2515.09,14.5,78.774
baseline,12,1,5741,1406,44,0.625,2250.15,14.8,92.768
baseline,12,2,5741,1406,44,0.626,2247.06,14.8,94.847
baseline,12,3,5741,1406,44,0.626,2245.81,14.8,94.930
trial,12,1,5741,1406,44,0.577,2438.41,14.9,96.748
trial,12,2,5741,1406,44,0.583,2411.12,14.9,92.589
trial,12,3,5741,1406,44,0.582,2417.64,14.9,96.121
baseline,13,1,7083,1298,54,0.669,1941.50,15.0,85.688
baseline,13,2,7083,1298,54,0.659,1971.07,15.0,85.533
baseline,13,3,7083,1298,54,0.655,1982.88,15.1,85.349
trial,13,1,7083,1298,54,0.609,2130.77,15.1,85.270
trial,13,2,7083,1298,54,0.622,2085.71,15.0,85.011
trial,13,3,7083,1298,54,0.612,2121.29,15.1,84.774
baseline,14,1,8041,904,24,0.504,1795.29,15.2,80.387
baseline,14,2,8041,904,24,0.508,1779.73,15.2,79.745
baseline,14,3,8041,904,24,0.509,1777.35,15.3,80.071
trial,14,1,8041,904,24,0.460,1966.55,15.5,80.181
trial,14,2,8041,904,24,0.470,1924.97,15.4,79.805
trial,14,3,8041,904,24,0.472,1917.22,15.4,80.356
baseline,15,1,8413,348,467,0.282,1235.89,15.6,75.437
baseline,15,2,8413,348,467,0.275,1266.28,15.5,74.668
baseline,15,3,8413,348,467,0.277,1254.24,15.5,75.348
trial,15,1,8413,348,467,0.262,1325.99,15.8,75.350
trial,15,2,8413,348,467,0.262,1330.11,15.8,75.074
trial,15,3,8413,348,467,0.261,1332.74,15.8,74.512
baseline,16,1,8915,35,186,0.103,340.68,15.4,70.231
baseline,16,2,8915,35,186,0.103,339.11,15.4,71.265
baseline,16,3,8915,35,186,0.103,340.20,15.3,69.940
trial,16,1,8915,35,186,0.087,400.67,15.7,70.002
trial,16,2,8915,35,186,0.085,410.28,15.6,72.049
trial,16,3,8915,35,186,0.088,395.86,15.7,70.106
```
