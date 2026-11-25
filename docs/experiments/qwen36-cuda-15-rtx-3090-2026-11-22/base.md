# NVIDIA GeForce RTX 3090 vs dev, full CUDA stack

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
| Cold 64 | 760.28 | 1,585.91 | +108.69% | 149.14 | 156.91 | +5.33% |
| 512 in sequence | 2,525.19 | 4,693.45 | +85.75% | 148.71 | 155.97 | +4.74% |
| Final 64 | 1,355.51 | 2,101.59 | +54.81% | 151.27 | 157.37 | +4.27% |
| Cold 512 | 2,175.42 | 4,357.65 | +100.47% | 149.42 | 157.33 | +5.29% |
| 2,048 | 2,920.61 | 6,263.53 | +114.13% | 148.34 | 156.18 | +5.42% |
| 4,096 | 2,979.35 | 6,655.37 | +122.87% | 149.05 | 156.74 | +5.22% |
| 8,192 | 3,742.98 | 6,592.44 | +76.43% | 144.76 | 151.87 | +5.04% |
| 16,384 | 3,250.62 | 6,208.80 | +92.29% | 140.19 | 148.33 | +5.63% |
| Chat 1 | 1,457.54 | 2,521.76 | +73.41%† | 149.73 | 169.55 | +13.70% |
| Chat 2 | 2,229.52 | 5,443.13 | +143.77% | 187.95 | 196.28 | +4.43%† |
| Chat 3 | 2,260.68 | 5,804.62 | +155.09% | 163.60 | 170.81 | +4.11% |
| Chat 4 | 2,738.04 | 5,228.82 | +90.97% | 194.64 | 208.34 | +6.74% |
| Chat 5 | 2,490.72 | 3,441.32 | +38.17%† | 174.93 | 185.23 | +6.49% |
| Chat 6 | 2,705.50 | 4,383.14 | +62.15% | 164.26 | 169.81 | +3.27% |
| Chat 7 | 1,851.95 | 2,958.77 | +59.79% | 164.67 | 171.64 | +3.88% |
| Chat 8 | 412.88 | 627.89 | +52.08% | 160.76 | 164.31 | +2.21% |

## Benchmark invocations

The commands show the recorded benchmark arguments with portable file paths. Set MODEL_DIR to the checkpoint directory and use the binary for each build. The timing runs also recorded the FP32 comparisons reported in correctness.txt.

```sh
COLI_CUDA=1 COLI_GPUS=0 CUDA_EXPERT_GB=auto OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close OMP_NUM_THREADS=8 OMP_DISPLAY_AFFINITY=TRUE OMP_AFFINITY_FORMAT=omp-thread:%n:cpus:%A /usr/bin/time -v ./qwen36-bench --model "$MODEL_DIR" --steps steps.tsv --text promessi_sposi.txt --lengths 64,512,64 --gen 128 --teacher
COLI_CUDA=1 COLI_GPUS=0 CUDA_EXPERT_GB=auto OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close OMP_NUM_THREADS=8 OMP_DISPLAY_AFFINITY=TRUE OMP_AFFINITY_FORMAT=omp-thread:%n:cpus:%A /usr/bin/time -v ./qwen36-bench --model "$MODEL_DIR" --steps steps.tsv --text promessi_sposi.txt --lengths 512 --gen 128 --teacher
COLI_CUDA=1 COLI_GPUS=0 CUDA_EXPERT_GB=auto OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close OMP_NUM_THREADS=8 OMP_DISPLAY_AFFINITY=TRUE OMP_AFFINITY_FORMAT=omp-thread:%n:cpus:%A /usr/bin/time -v ./qwen36-bench --model "$MODEL_DIR" --steps steps.tsv --text promessi_sposi.txt --lengths 2048,4096,8192,16384 --gen 128 --teacher
COLI_CUDA=1 COLI_GPUS=0 CUDA_EXPERT_GB=auto OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close OMP_NUM_THREADS=8 OMP_DISPLAY_AFFINITY=TRUE OMP_AFFINITY_FORMAT=omp-thread:%n:cpus:%A /usr/bin/time -v ./qwen36-bench --model "$MODEL_DIR" --steps steps.tsv --session agente.sess
```

Prose SHA-256: `f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f`. Chat session SHA-256: `8441e1a125a1c7647315c0c10104088b7d5298fd46f5c56036ee39bad74cf5fd`.

## Retained samples

Rows follow the table order above. Times and throughputs use the precision recorded by the benchmark.

```csv
arm,workload_id,round,context_tokens,new_prompt_tokens,generated_tokens,prefill_s,prefill_tok_s,first_decode_ms,steady_decode_tok_s
baseline,1,1,64,64,128,0.082,775.78,50.9,149.978
baseline,1,2,64,64,128,0.084,760.28,47.4,149.137
baseline,1,3,64,64,128,0.087,738.83,48.0,148.865
trial,1,1,64,64,128,0.040,1619.01,6.3,156.007
trial,1,2,64,64,128,0.040,1585.91,6.5,157.085
trial,1,3,64,64,128,0.040,1585.38,6.5,156.907
baseline,2,1,512,512,128,0.201,2547.93,6.8,148.621
baseline,2,2,512,512,128,0.203,2525.19,6.7,148.712
baseline,2,3,512,512,128,0.205,2497.90,7.1,149.457
trial,2,1,512,512,128,0.109,4694.37,6.7,155.973
trial,2,2,512,512,128,0.109,4690.65,6.6,155.767
trial,2,3,512,512,128,0.109,4693.45,6.6,156.051
baseline,3,1,64,64,128,0.047,1362.30,6.7,150.931
baseline,3,2,64,64,128,0.047,1355.51,6.8,151.269
baseline,3,3,64,64,128,0.048,1346.49,6.8,151.778
trial,3,1,64,64,128,0.030,2108.95,6.3,157.372
trial,3,2,64,64,128,0.030,2101.59,6.4,158.706
trial,3,3,64,64,128,0.031,2078.12,6.4,155.931
baseline,4,1,512,512,128,0.235,2175.42,46.4,149.423
baseline,4,2,512,512,128,0.236,2168.63,48.8,148.765
baseline,4,3,512,512,128,0.226,2265.25,47.0,149.532
trial,4,1,512,512,128,0.117,4362.83,6.9,157.326
trial,4,2,512,512,128,0.118,4347.49,6.7,156.267
trial,4,3,512,512,128,0.117,4357.65,6.7,157.462
baseline,5,1,2048,2048,128,0.688,2974.85,51.7,148.339
baseline,5,2,2048,2048,128,0.701,2920.61,50.1,147.834
baseline,5,3,2048,2048,128,0.703,2914.64,47.0,148.730
trial,5,1,2048,2048,128,0.325,6295.24,7.2,156.386
trial,5,2,2048,2048,128,0.327,6263.53,7.2,156.184
trial,5,3,2048,2048,128,0.328,6241.23,7.5,156.154
baseline,6,1,4096,4096,128,1.369,2991.04,6.8,149.172
baseline,6,2,4096,4096,128,1.375,2979.35,6.8,148.562
baseline,6,3,4096,4096,128,1.379,2971.26,6.8,149.054
trial,6,1,4096,4096,128,0.614,6666.06,6.8,156.965
trial,6,2,4096,4096,128,0.615,6655.37,6.7,156.739
trial,6,3,4096,4096,128,0.623,6579.78,6.7,155.467
baseline,7,1,8192,8192,128,2.189,3742.98,6.9,144.765
baseline,7,2,8192,8192,128,2.178,3761.05,6.9,144.522
baseline,7,3,8192,8192,128,2.201,3721.76,6.9,144.973
trial,7,1,8192,8192,128,1.241,6603.73,6.7,152.063
trial,7,2,8192,8192,128,1.243,6592.44,6.7,151.866
trial,7,3,8192,8192,128,1.247,6568.69,6.8,150.907
baseline,8,1,16384,16384,128,4.981,3289.17,7.1,140.185
baseline,8,2,16384,16384,128,5.040,3250.62,7.2,140.596
baseline,8,3,16384,16384,128,5.138,3188.60,7.3,140.063
trial,8,1,16384,16384,128,2.662,6153.86,7.0,148.326
trial,8,2,16384,16384,128,2.621,6250.74,7.0,148.506
trial,8,3,16384,16384,128,2.639,6208.80,7.7,147.842
baseline,9,1,663,663,57,0.454,1461.83,104.7,149.127
baseline,9,2,663,663,57,0.469,1413.74,105.4,152.460
baseline,9,3,663,663,57,0.455,1457.54,103.7,149.733
trial,9,1,663,663,57,0.262,2534.91,28.2,169.551
trial,9,2,663,663,57,0.263,2521.76,33.3,167.270
trial,9,3,663,663,57,0.286,2319.19,27.6,171.153
baseline,10,1,1954,1234,32,0.554,2228.49,67.1,167.614
baseline,10,2,1954,1234,32,0.538,2295.59,73.0,190.546
baseline,10,3,1954,1234,32,0.553,2229.52,68.8,187.948
trial,10,1,1954,1234,32,0.227,5432.47,26.6,196.759
trial,10,2,1954,1234,32,0.224,5508.09,30.8,196.207
trial,10,3,1954,1234,32,0.227,5443.13,28.6,196.278
baseline,11,1,4248,2262,87,1.001,2260.68,27.8,165.021
baseline,11,2,4248,2262,87,0.988,2288.63,28.9,150.616
baseline,11,3,4248,2262,87,1.015,2228.46,29.0,163.596
trial,11,1,4248,2262,87,0.392,5766.66,29.8,171.800
trial,11,2,4248,2262,87,0.390,5804.62,27.9,170.812
trial,11,3,4248,2262,87,0.387,5848.78,28.7,170.042
baseline,12,1,5741,1406,44,0.514,2738.04,28.1,195.511
baseline,12,2,5741,1406,44,0.511,2753.36,30.5,193.073
baseline,12,3,5741,1406,44,0.518,2716.55,66.9,194.638
trial,12,1,5741,1406,44,0.269,5228.82,77.3,208.685
trial,12,2,5741,1406,44,0.270,5215.52,62.5,208.336
trial,12,3,5741,1406,44,0.269,5229.33,66.1,206.998
baseline,13,1,7083,1298,54,0.515,2518.49,28.9,174.925
baseline,13,2,7083,1298,54,0.527,2461.15,33.4,178.191
baseline,13,3,7083,1298,54,0.521,2490.72,33.7,163.347
trial,13,1,7083,1298,54,0.426,3045.44,27.8,186.274
trial,13,2,7083,1298,54,0.372,3485.44,30.3,185.226
trial,13,3,7083,1298,54,0.377,3441.32,31.0,184.114
baseline,14,1,8041,904,24,0.336,2688.80,28.7,164.445
baseline,14,2,8041,904,24,0.332,2726.73,28.9,164.265
baseline,14,3,8041,904,24,0.334,2705.50,32.0,163.624
trial,14,1,8041,904,24,0.206,4383.14,29.3,169.815
trial,14,2,8041,904,24,0.206,4382.49,26.7,169.554
trial,14,3,8041,904,24,0.206,4386.98,29.5,170.622
baseline,15,1,8413,348,467,0.188,1851.95,33.8,164.670
baseline,15,2,8413,348,467,0.187,1865.79,67.6,165.751
baseline,15,3,8413,348,467,0.188,1850.09,27.3,163.958
trial,15,1,8413,348,467,0.118,2959.21,27.3,171.636
trial,15,2,8413,348,467,0.118,2940.99,29.0,172.182
trial,15,3,8413,348,467,0.118,2958.77,25.3,170.096
baseline,16,1,8915,35,186,0.085,412.88,26.5,161.076
baseline,16,2,8915,35,186,0.084,416.45,29.8,160.757
baseline,16,3,8915,35,186,0.087,401.62,35.8,157.231
trial,16,1,8915,35,186,0.056,627.89,28.3,164.370
trial,16,2,8915,35,186,0.056,628.07,31.0,164.309
trial,16,3,8915,35,186,0.056,620.00,25.2,163.367
```
