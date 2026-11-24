# NVIDIA L40S vs dev, full CUDA stack

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
| Cold 64 | 913.56 | 1,949.51 | +113.26% | 146.46 | 155.27 | +6.37% |
| 512 in sequence | 3,571.35 | 9,043.74 | +153.91% | 146.17 | 155.30 | +6.24% |
| Final 64 | 1,757.23 | 2,849.19 | +61.95% | 148.57 | 157.03 | +5.78% |
| Cold 512 | 3,032.25 | 7,760.67 | +155.68% | 145.78 | 160.14 | +9.88% |
| 2,048 | 4,248.03 | 10,639.94 | +148.12% | 145.97 | 154.67 | +5.93% |
| 4,096 | 4,265.31 | 10,773.89 | +152.59% | 144.12 | 153.80 | +6.95% |
| 8,192 | 5,882.91 | 10,406.15 | +76.93% | 141.34 | 146.63 | +3.45% |
| 16,384 | 5,551.93 | 10,889.04 | +96.94% | 135.39 | 148.29 | +9.51% |
| Chat 1 | 1,897.49 | 4,332.07 | +130.09% | 150.79 | 168.33 | +11.63% |
| Chat 2 | 3,184.09 | 9,788.40 | +207.42% | 145.40 | 189.98 | +25.57%† |
| Chat 3 | 3,151.79 | 9,551.75 | +203.06% | 145.42 | 173.94 | +19.80% |
| Chat 4 | 3,870.87 | 8,648.99 | +123.44% | 188.97 | 196.49 | +3.61% |
| Chat 5 | 3,653.46 | 5,605.87 | +53.37% | 181.79 | 193.61 | +7.47% |
| Chat 6 | 4,277.97 | 6,925.06 | +62.15% | 172.19 | 181.20 | +5.31% |
| Chat 7 | 2,838.57 | 4,329.11 | +52.51% | 165.12 | 177.47 | +7.18% |
| Chat 8 | 517.69 | 848.77 | +64.21% | 159.46 | 163.34 | +2.36% |

## Benchmark invocations

The commands show the recorded benchmark arguments with portable file paths. Set MODEL_DIR to the checkpoint directory and use the binary for each build. The timing runs also recorded the FP32 comparisons reported in correctness.txt.

```sh
COLI_CUDA=1 COLI_GPUS=0 CUDA_EXPERT_GB=auto OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close OMP_NUM_THREADS=23 OMP_DISPLAY_AFFINITY=TRUE OMP_AFFINITY_FORMAT=omp-thread:%n:cpus:%A /usr/bin/time -v ./qwen36-bench --model "$MODEL_DIR" --steps steps.tsv --text promessi_sposi.txt --lengths 64,512,64 --gen 128 --teacher
COLI_CUDA=1 COLI_GPUS=0 CUDA_EXPERT_GB=auto OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close OMP_NUM_THREADS=23 OMP_DISPLAY_AFFINITY=TRUE OMP_AFFINITY_FORMAT=omp-thread:%n:cpus:%A /usr/bin/time -v ./qwen36-bench --model "$MODEL_DIR" --steps steps.tsv --text promessi_sposi.txt --lengths 512 --gen 128 --teacher
COLI_CUDA=1 COLI_GPUS=0 CUDA_EXPERT_GB=auto OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close OMP_NUM_THREADS=23 OMP_DISPLAY_AFFINITY=TRUE OMP_AFFINITY_FORMAT=omp-thread:%n:cpus:%A /usr/bin/time -v ./qwen36-bench --model "$MODEL_DIR" --steps steps.tsv --text promessi_sposi.txt --lengths 2048,4096,8192,16384 --gen 128 --teacher
COLI_CUDA=1 COLI_GPUS=0 CUDA_EXPERT_GB=auto OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close OMP_NUM_THREADS=23 OMP_DISPLAY_AFFINITY=TRUE OMP_AFFINITY_FORMAT=omp-thread:%n:cpus:%A /usr/bin/time -v ./qwen36-bench --model "$MODEL_DIR" --steps steps.tsv --session agente.sess
```

Prose SHA-256: `f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f`. Chat session SHA-256: `8441e1a125a1c7647315c0c10104088b7d5298fd46f5c56036ee39bad74cf5fd`.

## Retained samples

Rows follow the table order above. Times and throughputs use the precision recorded by the benchmark.

```csv
arm,workload_id,round,context_tokens,new_prompt_tokens,generated_tokens,prefill_s,prefill_tok_s,first_decode_ms,steady_decode_tok_s
baseline,1,1,64,64,128,0.073,874.03,49.2,146.539
baseline,1,2,64,64,128,0.070,914.13,42.4,146.457
baseline,1,3,64,64,128,0.070,913.56,45.5,145.258
trial,1,1,64,64,128,0.032,1988.88,6.6,153.519
trial,1,2,64,64,128,0.033,1949.51,6.5,155.783
trial,1,3,64,64,128,0.033,1941.92,6.5,155.271
baseline,2,1,512,512,128,0.147,3478.10,6.9,146.533
baseline,2,2,512,512,128,0.143,3571.35,6.8,146.172
baseline,2,3,512,512,128,0.142,3593.85,6.8,144.085
trial,2,1,512,512,128,0.057,9043.74,6.4,155.434
trial,2,2,512,512,128,0.056,9068.07,6.5,155.297
trial,2,3,512,512,128,0.059,8727.22,6.5,154.896
baseline,3,1,64,64,128,0.037,1749.12,6.7,148.175
baseline,3,2,64,64,128,0.036,1761.73,6.7,148.584
baseline,3,3,64,64,128,0.036,1757.23,6.7,148.567
trial,3,1,64,64,128,0.022,2884.40,6.5,157.026
trial,3,2,64,64,128,0.022,2849.19,6.3,157.173
trial,3,3,64,64,128,0.022,2845.90,6.3,156.037
baseline,4,1,512,512,128,0.170,3016.62,43.8,143.799
baseline,4,2,512,512,128,0.169,3032.25,44.9,146.698
baseline,4,3,512,512,128,0.169,3035.29,44.4,145.780
trial,4,1,512,512,128,0.066,7764.18,6.7,160.140
trial,4,2,512,512,128,0.066,7727.15,6.5,159.957
trial,4,3,512,512,128,0.066,7760.67,6.6,160.177
baseline,5,1,2048,2048,128,0.477,4297.82,45.3,146.001
baseline,5,2,2048,2048,128,0.482,4248.03,43.5,145.335
baseline,5,3,2048,2048,128,0.487,4205.01,46.8,145.965
trial,5,1,2048,2048,128,0.192,10663.74,7.4,154.666
trial,5,2,2048,2048,128,0.192,10639.94,25.4,154.916
trial,5,3,2048,2048,128,0.197,10409.32,7.5,153.298
baseline,6,1,4096,4096,128,0.959,4272.60,7.0,144.118
baseline,6,2,4096,4096,128,0.960,4265.31,6.9,143.549
baseline,6,3,4096,4096,128,0.961,4261.64,7.1,145.976
trial,6,1,4096,4096,128,0.379,10818.55,6.7,154.173
trial,6,2,4096,4096,128,0.380,10773.89,6.7,153.519
trial,6,3,4096,4096,128,0.385,10641.24,6.8,153.802
baseline,7,1,8192,8192,128,1.393,5882.91,7.1,141.335
baseline,7,2,8192,8192,128,1.393,5881.41,7.1,141.739
baseline,7,3,8192,8192,128,1.391,5890.43,7.2,140.506
trial,7,1,8192,8192,128,0.781,10487.33,6.9,143.536
trial,7,2,8192,8192,128,0.787,10406.15,7.1,146.631
trial,7,3,8192,8192,128,0.790,10367.74,7.1,147.382
baseline,8,1,16384,16384,128,2.951,5551.93,7.7,135.388
baseline,8,2,16384,16384,128,2.939,5574.84,7.7,135.972
baseline,8,3,16384,16384,128,2.975,5507.35,7.6,135.196
trial,8,1,16384,16384,128,1.498,10937.42,7.3,148.260
trial,8,2,16384,16384,128,1.505,10889.04,7.3,148.290
trial,8,3,16384,16384,128,1.511,10846.40,7.4,148.359
baseline,9,1,663,663,57,0.349,1897.49,89.3,144.799
baseline,9,2,663,663,57,0.352,1882.75,88.3,150.786
baseline,9,3,663,663,57,0.347,1908.12,112.1,153.724
trial,9,1,663,663,57,0.151,4385.04,15.5,170.057
trial,9,2,663,663,57,0.153,4332.07,15.9,168.327
trial,9,3,663,663,57,0.153,4320.53,15.6,167.022
baseline,10,1,1954,1234,32,0.388,3181.88,49.3,145.398
baseline,10,2,1954,1234,32,0.388,3184.09,50.3,145.215
baseline,10,3,1954,1234,32,0.386,3195.54,49.3,192.052
trial,10,1,1954,1234,32,0.125,9867.56,50.8,182.579
trial,10,2,1954,1234,32,0.126,9788.40,51.9,191.108
trial,10,3,1954,1234,32,0.126,9780.14,51.6,189.981
baseline,11,1,4248,2262,87,0.718,3151.79,49.8,145.424
baseline,11,2,4248,2262,87,0.720,3139.75,48.2,145.190
baseline,11,3,4248,2262,87,0.717,3155.37,16.6,146.040
trial,11,1,4248,2262,87,0.237,9551.75,16.1,174.395
trial,11,2,4248,2262,87,0.241,9367.82,29.6,173.939
trial,11,3,4248,2262,87,0.236,9604.08,16.3,172.556
baseline,12,1,5741,1406,44,0.363,3878.13,16.3,189.649
baseline,12,2,5741,1406,44,0.371,3786.33,16.3,188.975
baseline,12,3,5741,1406,44,0.363,3870.87,17.2,187.215
trial,12,1,5741,1406,44,0.162,8657.81,16.1,196.488
trial,12,2,5741,1406,44,0.163,8632.47,22.2,189.477
trial,12,3,5741,1406,44,0.163,8648.99,16.1,198.422
baseline,13,1,7083,1298,54,0.356,3641.94,16.5,173.976
baseline,13,2,7083,1298,54,0.355,3653.46,47.0,182.537
baseline,13,3,7083,1298,54,0.351,3695.51,16.3,181.788
trial,13,1,7083,1298,54,0.232,5585.71,16.6,189.954
trial,13,2,7083,1298,54,0.232,5605.87,48.1,196.164
trial,13,3,7083,1298,54,0.231,5614.89,16.4,193.610
baseline,14,1,8041,904,24,0.211,4277.97,16.6,164.185
baseline,14,2,8041,904,24,0.212,4270.68,33.3,174.874
baseline,14,3,8041,904,24,0.211,4293.90,48.5,172.189
trial,14,1,8041,904,24,0.130,6952.19,20.6,174.937
trial,14,2,8041,904,24,0.131,6925.06,17.9,181.201
trial,14,3,8041,904,24,0.131,6920.51,19.9,181.337
baseline,15,1,8413,348,467,0.123,2838.57,16.4,163.157
baseline,15,2,8413,348,467,0.123,2830.18,33.4,165.641
baseline,15,3,8413,348,467,0.122,2853.53,69.2,165.118
trial,15,1,8413,348,467,0.080,4329.11,45.6,172.825
trial,15,2,8413,348,467,0.080,4337.30,16.3,177.526
trial,15,3,8413,348,467,0.081,4313.96,47.6,177.467
baseline,16,1,8915,35,186,0.068,517.69,18.6,158.316
baseline,16,2,8915,35,186,0.068,516.87,16.9,159.464
baseline,16,3,8915,35,186,0.067,520.19,33.4,159.583
trial,16,1,8915,35,186,0.041,851.67,17.6,159.275
trial,16,2,8915,35,186,0.041,848.77,16.4,165.032
trial,16,3,8915,35,186,0.041,847.46,64.0,163.343
```
