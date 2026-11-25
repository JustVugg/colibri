# NVIDIA A100-SXM4-40GB vs dev, full CUDA stack

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
| Cold 64 | 927.78 | 1,654.25 | +77.55% | 107.51 | 143.01 | +33.33% |
| 512 in sequence | 3,553.84 | 6,261.12 | +76.18% | 106.85 | 142.07 | +33.27% |
| Final 64 | 1,582.78 | 2,405.50 | +51.97% | 107.79 | 144.43 | +34.47% |
| Cold 512 | 3,042.85 | 5,496.06 | +81.42% | 107.01 | 141.72 | +33.80% |
| 2,048 | 4,244.00 | 7,700.72 | +81.40% | 106.10 | 143.25 | +34.75% |
| 4,096 | 4,453.74 | 8,345.62 | +87.35% | 105.61 | 141.94 | +34.30% |
| 8,192 | 5,484.55 | 8,422.67 | +53.64% | 103.48 | 137.23 | +32.58% |
| 16,384 | 5,074.96 | 8,199.19 | +61.55% | 100.57 | 132.43 | +31.57% |
| Chat 1 | 1,876.69 | 3,289.53 | +75.28% | 114.64 | 155.22 | +34.88% |
| Chat 2 | 3,326.32 | 7,084.88 | +112.87% | 117.10 | 176.59 | +50.00%† |
| Chat 3 | 3,292.05 | 7,381.25 | +124.53% | 117.32 | 155.89 | +32.16%† |
| Chat 4 | 3,638.80 | 6,784.86 | +86.46% | 140.93 | 187.06 | +31.32% |
| Chat 5 | 3,265.14 | 5,861.67 | +79.54% | 129.20 | 167.03 | +28.76% |
| Chat 6 | 3,406.73 | 4,709.26 | +38.27% | 111.14 | 154.46 | +39.65% |
| Chat 7 | 2,429.34 | 3,039.80 | +25.13% | 113.83 | 156.26 | +35.59% |
| Chat 8 | 401.17 | 669.46 | +66.84% | 110.68 | 150.54 | +35.90% |

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
baseline,1,1,64,64,128,0.069,927.78,43.4,107.260
baseline,1,2,64,64,128,0.069,925.43,44.1,107.655
baseline,1,3,64,64,128,0.068,935.25,43.6,107.511
trial,1,1,64,64,128,0.039,1647.31,7.1,143.011
trial,1,2,64,64,128,0.038,1677.90,7.1,144.331
trial,1,3,64,64,128,0.039,1654.25,7.1,142.915
baseline,2,1,512,512,128,0.144,3553.84,9.3,106.597
baseline,2,2,512,512,128,0.144,3543.78,9.3,106.956
baseline,2,3,512,512,128,0.144,3555.69,9.3,106.846
trial,2,1,512,512,128,0.082,6261.12,7.0,142.066
trial,2,2,512,512,128,0.082,6265.98,6.9,143.583
trial,2,3,512,512,128,0.082,6247.50,7.0,141.563
baseline,3,1,64,64,128,0.040,1582.91,9.3,107.409
baseline,3,2,64,64,128,0.040,1582.78,9.3,107.821
baseline,3,3,64,64,128,0.040,1581.40,9.3,107.794
trial,3,1,64,64,128,0.027,2405.50,7.0,144.428
trial,3,2,64,64,128,0.026,2416.37,6.9,145.222
trial,3,3,64,64,128,0.027,2397.71,7.0,143.570
baseline,4,1,512,512,128,0.171,2991.25,43.0,107.137
baseline,4,2,512,512,128,0.168,3042.85,43.2,107.006
baseline,4,3,512,512,128,0.166,3086.11,42.6,105.800
trial,4,1,512,512,128,0.093,5496.06,7.2,141.725
trial,4,2,512,512,128,0.093,5520.47,7.1,143.909
trial,4,3,512,512,128,0.094,5468.92,7.2,141.564
baseline,5,1,2048,2048,128,0.481,4254.41,43.1,106.101
baseline,5,2,2048,2048,128,0.483,4242.38,43.2,105.498
baseline,5,3,2048,2048,128,0.483,4244.00,42.5,106.589
trial,5,1,2048,2048,128,0.266,7700.72,7.6,141.093
trial,5,2,2048,2048,128,0.266,7695.63,7.5,143.248
trial,5,3,2048,2048,128,0.266,7711.34,7.5,143.625
baseline,6,1,4096,4096,128,0.919,4458.78,9.5,105.615
baseline,6,2,4096,4096,128,0.921,4446.00,9.5,105.377
baseline,6,3,4096,4096,128,0.920,4453.74,9.4,106.075
trial,6,1,4096,4096,128,0.491,8349.07,7.2,140.544
trial,6,2,4096,4096,128,0.492,8329.56,7.1,141.944
trial,6,3,4096,4096,128,0.491,8345.62,7.1,142.462
baseline,7,1,8192,8192,128,1.498,5470.17,9.7,103.478
baseline,7,2,8192,8192,128,1.491,5492.90,9.6,102.721
baseline,7,3,8192,8192,128,1.494,5484.55,9.6,103.907
trial,7,1,8192,8192,128,0.973,8422.67,7.3,134.571
trial,7,2,8192,8192,128,0.973,8417.12,7.3,137.233
trial,7,3,8192,8192,128,0.972,8426.47,7.2,137.758
baseline,8,1,16384,16384,128,3.225,5080.16,9.9,100.566
baseline,8,2,16384,16384,128,3.228,5074.96,9.9,100.359
baseline,8,3,16384,16384,128,3.229,5074.73,9.9,100.946
trial,8,1,16384,16384,128,1.998,8199.19,7.7,131.054
trial,8,2,16384,16384,128,1.998,8198.77,7.6,132.431
trial,8,3,16384,16384,128,1.997,8204.66,7.6,132.817
baseline,9,1,663,663,57,0.353,1878.49,100.4,114.635
baseline,9,2,663,663,57,0.356,1863.44,100.7,113.517
baseline,9,3,663,663,57,0.353,1876.69,99.8,116.213
trial,9,1,663,663,57,0.201,3291.97,38.1,154.625
trial,9,2,663,663,57,0.202,3279.77,37.9,155.335
trial,9,3,663,663,57,0.202,3289.53,38.2,155.223
baseline,10,1,1954,1234,32,0.371,3326.32,71.8,117.101
baseline,10,2,1954,1234,32,0.371,3325.23,71.7,118.180
baseline,10,3,1954,1234,32,0.369,3342.41,72.4,106.264
trial,10,1,1954,1234,32,0.174,7080.82,40.0,175.565
trial,10,2,1954,1234,32,0.174,7095.05,39.8,177.274
trial,10,3,1954,1234,32,0.174,7084.88,40.7,176.594
baseline,11,1,4248,2262,87,0.687,3292.05,39.9,117.324
baseline,11,2,4248,2262,87,0.698,3241.96,40.0,118.878
baseline,11,3,4248,2262,87,0.675,3350.77,40.2,106.751
trial,11,1,4248,2262,87,0.306,7391.65,37.5,155.060
trial,11,2,4248,2262,87,0.306,7381.25,38.8,156.246
trial,11,3,4248,2262,87,0.307,7375.84,38.7,155.893
baseline,12,1,5741,1406,44,0.386,3638.80,40.0,140.434
baseline,12,2,5741,1406,44,0.387,3635.88,40.0,140.929
baseline,12,3,5741,1406,44,0.385,3650.41,40.2,142.854
trial,12,1,5741,1406,44,0.207,6784.86,69.1,184.413
trial,12,2,5741,1406,44,0.207,6787.67,68.9,187.056
trial,12,3,5741,1406,44,0.208,6752.23,69.2,187.427
baseline,13,1,7083,1298,54,0.397,3265.88,70.9,129.201
baseline,13,2,7083,1298,54,0.398,3265.14,71.2,129.916
baseline,13,3,7083,1298,54,0.401,3235.38,70.9,128.065
trial,13,1,7083,1298,54,0.221,5861.67,38.1,165.766
trial,13,2,7083,1298,54,0.221,5862.21,38.4,167.275
trial,13,3,7083,1298,54,0.222,5853.56,37.8,167.028
baseline,14,1,8041,904,24,0.266,3402.98,40.4,110.394
baseline,14,2,8041,904,24,0.265,3406.73,40.1,111.345
baseline,14,3,8041,904,24,0.263,3442.09,41.1,111.139
trial,14,1,8041,904,24,0.192,4709.26,38.5,154.164
trial,14,2,8041,904,24,0.192,4710.62,38.0,155.850
trial,14,3,8041,904,24,0.192,4701.22,38.7,154.462
baseline,15,1,8413,348,467,0.143,2429.34,40.4,113.830
baseline,15,2,8413,348,467,0.143,2432.23,40.1,115.244
baseline,15,3,8413,348,467,0.144,2424.03,41.0,113.764
trial,15,1,8413,348,467,0.114,3039.80,38.0,153.876
trial,15,2,8413,348,467,0.114,3050.99,38.3,156.262
trial,15,3,8413,348,467,0.115,3027.82,38.3,156.642
baseline,16,1,8915,35,186,0.087,401.17,39.9,109.824
baseline,16,2,8915,35,186,0.087,400.90,40.2,110.772
baseline,16,3,8915,35,186,0.087,401.26,40.7,110.680
trial,16,1,8915,35,186,0.053,664.36,38.4,148.792
trial,16,2,8915,35,186,0.052,671.52,38.1,150.538
trial,16,3,8915,35,186,0.052,669.46,38.0,151.972
```
