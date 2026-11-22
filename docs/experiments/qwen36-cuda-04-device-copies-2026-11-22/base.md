# L4 incremental measurement · Overlap device-to-host copies with computation

These measurements compare consecutive cumulative builds on the same physical L4. Each build was also checked against the original dev reference.

Baseline: pr03 (source 5040b7e784ac013b3af894aab75ef3eaea4dda8d, compiled c/ tree 66dcbe713f33530310f6388c01eebcb3983133e9).
Trial: pr04 (source 1f5bba1efcac99dab7227f68ee098452d60af900, compiled c/ tree 24a49c8e227780e760424a27e6b169a6815bbfb4).
Campaign dev reference: `e677aaee4424d27b8eb0a7e7d7edbb32a309c267`. Model: `Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64@c619aa594ad1e70af82168fb6b4878427896e21c`.
The measurements predate the final packaging of the pull requests.
This series used the earlier benchmark. Its chat first-step timings exclude the server prompt-state save. Fixed-length decode used the request handling of that benchmark.

One reference warm-up per group on dev before the series. Cumulative builds ran three measured repetitions per group. Percentage change between the two build medians. Each build contributes three retained runs per workload.
Groups preserve process history: 64 → 512 → 64; cold 512; 2,048 → 4,096 → 8,192 → 16,384; then a separate eight-turn chat replay.
The first decode step is timed separately. Decode throughput covers the remaining steps. A dagger marks a reported spread above 10% after the repeat.

| Workload | Baseline prefill tok/s | Trial prefill tok/s | Change | Baseline decode tok/s | Trial decode tok/s | Change |
|---|---:|---:|---:|---:|---:|---:|
| Cold 64 | 855.46 | 850.22 | -0.61% | 71.49 | 70.07 | -1.99% |
| 512 in sequence | 2,275.76 | 2,367.15 | +4.02% | 71.26 | 70.54 | -1.00% |
| Final 64 | 978.20 | 991.01 | +1.31% | 71.50 | 71.02 | -0.68% |
| Cold 512 | 2,206.03 | 2,273.09 | +3.04% | 71.19 | 70.85 | -0.48% |
| 2,048 | 2,797.17 | 2,926.48 | +4.62% | 70.67 | 70.20 | -0.67% |
| 4,096 | 2,866.81 | 3,001.23 | +4.69% | 69.72 | 69.34 | -0.55% |
| 8,192 | 2,769.76 | 2,971.84 | +7.30% | 67.57 | 67.21 | -0.53% |
| 16,384 | 2,651.39 | 2,853.44 | +7.62% | 64.53 | 64.13 | -0.63% |
| Chat 1 | 1,509.68 | 1,541.02 | +2.08% | 79.34 | 79.44 | +0.12% |
| Chat 2 | 2,543.25 | 2,657.22 | +4.48% | 95.21 | 92.17 | -3.19% |
| Chat 3 | 2,552.14 | 2,750.68 | +7.78% | 78.58 | 77.17 | -1.79% |
| Chat 4 | 2,417.64 | 2,588.42 | +7.06% | 96.12 | 94.16 | -2.04% |
| Chat 5 | 2,121.29 | 2,407.02 | +13.47% | 85.01 | 84.67 | -0.40% |
| Chat 6 | 1,924.97 | 2,098.72 | +9.03% | 80.18 | 79.39 | -0.99% |
| Chat 7 | 1,330.11 | 1,499.42 | +12.73% | 75.07 | 74.97 | -0.14% |
| Chat 8 | 400.67 | 403.13 | +0.61% | 70.11 | 71.79 | +2.40% |

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
baseline,1,1,64,64,128,0.075,855.46,48.1,71.492
baseline,1,2,64,64,128,0.074,860.10,47.2,71.484
baseline,1,3,64,64,128,0.077,832.93,48.3,71.576
trial,1,1,64,64,128,0.076,847.60,15.5,69.971
trial,1,2,64,64,128,0.075,850.22,15.3,70.067
trial,1,3,64,64,128,0.074,868.97,15.4,71.005
baseline,2,1,512,512,128,0.222,2307.17,14.7,71.257
baseline,2,2,512,512,128,0.225,2275.76,15.2,71.218
baseline,2,3,512,512,128,0.226,2262.68,14.7,71.294
trial,2,1,512,512,128,0.213,2400.48,15.4,70.543
trial,2,2,512,512,128,0.216,2367.15,15.7,69.740
trial,2,3,512,512,128,0.227,2256.93,15.5,70.752
baseline,3,1,64,64,128,0.065,985.53,15.3,71.533
baseline,3,2,64,64,128,0.065,978.20,15.2,71.383
baseline,3,3,64,64,128,0.066,967.88,15.4,71.499
trial,3,1,64,64,128,0.064,1002.42,15.3,71.202
trial,3,2,64,64,128,0.065,991.01,15.6,70.874
trial,3,3,64,64,128,0.066,975.71,15.4,71.015
baseline,4,1,512,512,128,0.231,2217.07,48.2,71.193
baseline,4,2,512,512,128,0.232,2206.03,46.8,71.339
baseline,4,3,512,512,128,0.234,2184.95,47.8,71.142
trial,4,1,512,512,128,0.222,2304.39,15.7,70.880
trial,4,2,512,512,128,0.225,2273.09,15.7,70.697
trial,4,3,512,512,128,0.227,2258.01,15.7,70.850
baseline,5,1,2048,2048,128,0.727,2816.96,47.2,70.668
baseline,5,2,2048,2048,128,0.732,2797.17,46.6,70.735
baseline,5,3,2048,2048,128,0.735,2785.93,48.5,69.844
trial,5,1,2048,2048,128,0.692,2959.07,15.2,70.196
trial,5,2,2048,2048,128,0.704,2910.50,15.2,69.886
trial,5,3,2048,2048,128,0.700,2926.48,15.6,70.300
baseline,6,1,4096,4096,128,1.401,2922.76,14.5,69.696
baseline,6,2,4096,4096,128,1.430,2864.74,14.5,69.722
baseline,6,3,4096,4096,128,1.429,2866.81,14.4,69.813
trial,6,1,4096,4096,128,1.344,3047.56,15.3,69.337
trial,6,2,4096,4096,128,1.365,3001.23,15.1,69.481
trial,6,3,4096,4096,128,1.366,2998.63,15.2,68.626
baseline,7,1,8192,8192,128,2.901,2823.74,14.8,67.570
baseline,7,2,8192,8192,128,2.960,2767.26,14.9,67.559
baseline,7,3,8192,8192,128,2.958,2769.76,14.8,67.603
trial,7,1,8192,8192,128,2.724,3007.89,15.6,67.211
trial,7,2,8192,8192,128,2.762,2965.48,15.6,67.255
trial,7,3,8192,8192,128,2.757,2971.84,15.6,65.963
baseline,8,1,16384,16384,128,6.095,2688.22,15.5,64.570
baseline,8,2,16384,16384,128,6.179,2651.39,15.6,64.375
baseline,8,3,16384,16384,128,6.197,2644.07,15.5,64.532
trial,8,1,16384,16384,128,5.652,2899.02,16.3,64.127
trial,8,2,16384,16384,128,5.742,2853.44,16.4,64.174
trial,8,3,16384,16384,128,5.759,2845.17,16.4,64.031
baseline,9,1,663,663,57,0.435,1522.60,49.0,79.340
baseline,9,2,663,663,57,0.442,1498.62,47.7,79.093
baseline,9,3,663,663,57,0.439,1509.68,47.4,79.440
trial,9,1,663,663,57,0.430,1542.03,15.4,79.435
trial,9,2,663,663,57,0.430,1541.02,15.5,80.869
trial,9,3,663,663,57,0.435,1525.53,15.4,77.756
baseline,10,1,1954,1234,32,0.481,2567.11,14.7,95.554
baseline,10,2,1954,1234,32,0.485,2543.25,14.7,94.676
baseline,10,3,1954,1234,32,0.486,2541.35,14.7,95.207
trial,10,1,1954,1234,32,0.460,2683.15,15.4,90.276
trial,10,2,1954,1234,32,0.464,2657.22,15.2,92.171
trial,10,3,1954,1234,32,0.467,2641.19,15.5,92.617
baseline,11,1,4248,2262,87,0.878,2575.23,14.5,78.582
baseline,11,2,4248,2262,87,0.886,2552.14,14.5,78.238
baseline,11,3,4248,2262,87,0.899,2515.09,14.5,78.774
trial,11,1,4248,2262,87,0.813,2783.28,15.8,77.175
trial,11,2,4248,2262,87,0.822,2750.68,15.9,78.327
trial,11,3,4248,2262,87,0.823,2748.06,15.9,77.163
baseline,12,1,5741,1406,44,0.577,2438.41,14.9,96.748
baseline,12,2,5741,1406,44,0.583,2411.12,14.9,92.589
baseline,12,3,5741,1406,44,0.582,2417.64,14.9,96.121
trial,12,1,5741,1406,44,0.543,2590.45,15.9,90.624
trial,12,2,5741,1406,44,0.543,2588.42,15.7,94.162
trial,12,3,5741,1406,44,0.551,2549.49,16.3,95.182
baseline,13,1,7083,1298,54,0.609,2130.77,15.1,85.270
baseline,13,2,7083,1298,54,0.622,2085.71,15.0,85.011
baseline,13,3,7083,1298,54,0.612,2121.29,15.1,84.774
trial,13,1,7083,1298,54,0.544,2387.71,16.1,83.913
trial,13,2,7083,1298,54,0.539,2407.50,15.7,84.974
trial,13,3,7083,1298,54,0.539,2407.02,16.5,84.674
baseline,14,1,8041,904,24,0.460,1966.55,15.5,80.181
baseline,14,2,8041,904,24,0.470,1924.97,15.4,79.805
baseline,14,3,8041,904,24,0.472,1917.22,15.4,80.356
trial,14,1,8041,904,24,0.430,2103.76,16.2,79.386
trial,14,2,8041,904,24,0.431,2098.72,16.5,79.620
trial,14,3,8041,904,24,0.434,2084.29,16.3,79.299
baseline,15,1,8413,348,467,0.262,1325.99,15.8,75.350
baseline,15,2,8413,348,467,0.262,1330.11,15.8,75.074
baseline,15,3,8413,348,467,0.261,1332.74,15.8,74.512
trial,15,1,8413,348,467,0.232,1499.42,16.1,75.299
trial,15,2,8413,348,467,0.230,1512.86,16.2,74.659
trial,15,3,8413,348,467,0.232,1498.63,16.4,74.971
baseline,16,1,8915,35,186,0.087,400.67,15.7,70.002
baseline,16,2,8915,35,186,0.085,410.28,15.6,72.049
baseline,16,3,8915,35,186,0.088,395.86,15.7,70.106
trial,16,1,8915,35,186,0.087,403.13,15.7,71.892
trial,16,2,8915,35,186,0.087,403.20,15.8,71.684
trial,16,3,8915,35,186,0.088,397.38,16.0,71.792
```
