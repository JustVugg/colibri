# NVIDIA GeForce RTX 5090 vs dev, full CUDA stack

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
| Cold 64 | 1,514.17 | 3,079.10 | +103.35% | 255.47 | 267.10 | +4.56% |
| 512 in sequence | 5,983.22 | 13,575.48 | +126.72% | 255.24 | 268.54 | +5.35% |
| Final 64 | 2,778.61 | 4,535.41 | +63.61% | 257.56 | 269.01 | +4.60% |
| Cold 512 | 5,087.62 | 11,726.00 | +130.48% | 255.81 | 269.26 | +5.27% |
| 2,048 | 6,965.76 | 16,855.86 | +141.98% | 254.58 | 267.75 | +5.08% |
| 4,096 | 7,022.75 | 18,031.52 | +156.81% | 254.53 | 266.96 | +5.00% |
| 8,192 | 9,377.94 | 17,665.26 | +88.37% | 247.27 | 259.53 | +5.01% |
| 16,384 | 8,364.30 | 16,518.42 | +97.57% | 243.65 | 253.42 | +4.09% |
| Chat 1 | 3,117.64 | 7,014.13 | +125.46% | 266.94 | 297.19 | +11.47% |
| Chat 2 | 5,247.61 | 15,835.79 | +201.04%† | 288.56 | 315.71 | +9.41% |
| Chat 3 | 5,226.53 | 16,235.29 | +210.37% | 284.76 | 298.37 | +4.84% |
| Chat 4 | 6,270.82 | 14,376.05 | +128.73% | 350.73 | 365.66 | +4.20% |
| Chat 5 | 6,076.41 | 9,222.98 | +51.79% | 316.27 | 328.31 | +4.06% |
| Chat 6 | 6,299.39 | 11,928.41 | +89.59% | 297.05 | 313.04 | +5.43% |
| Chat 7 | 4,172.34 | 7,674.94 | +83.84% | 289.86 | 303.46 | +4.69% |
| Chat 8 | 654.75 | 1,193.28 | +82.25% | 279.04 | 292.14 | +4.69% |

## Benchmark invocations

The commands show the recorded benchmark arguments with portable file paths. Set MODEL_DIR to the checkpoint directory and use the binary for each build. The timing runs also recorded the FP32 comparisons reported in correctness.txt.

```sh
COLI_CUDA=1 COLI_GPUS=0 CUDA_EXPERT_GB=auto OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close OMP_NUM_THREADS=17 OMP_DISPLAY_AFFINITY=TRUE OMP_AFFINITY_FORMAT=omp-thread:%n:cpus:%A /usr/bin/time -v ./qwen36-bench --model "$MODEL_DIR" --steps steps.tsv --text promessi_sposi.txt --lengths 64,512,64 --gen 128 --teacher
COLI_CUDA=1 COLI_GPUS=0 CUDA_EXPERT_GB=auto OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close OMP_NUM_THREADS=17 OMP_DISPLAY_AFFINITY=TRUE OMP_AFFINITY_FORMAT=omp-thread:%n:cpus:%A /usr/bin/time -v ./qwen36-bench --model "$MODEL_DIR" --steps steps.tsv --text promessi_sposi.txt --lengths 512 --gen 128 --teacher
COLI_CUDA=1 COLI_GPUS=0 CUDA_EXPERT_GB=auto OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close OMP_NUM_THREADS=17 OMP_DISPLAY_AFFINITY=TRUE OMP_AFFINITY_FORMAT=omp-thread:%n:cpus:%A /usr/bin/time -v ./qwen36-bench --model "$MODEL_DIR" --steps steps.tsv --text promessi_sposi.txt --lengths 2048,4096,8192,16384 --gen 128 --teacher
COLI_CUDA=1 COLI_GPUS=0 CUDA_EXPERT_GB=auto OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close OMP_NUM_THREADS=17 OMP_DISPLAY_AFFINITY=TRUE OMP_AFFINITY_FORMAT=omp-thread:%n:cpus:%A /usr/bin/time -v ./qwen36-bench --model "$MODEL_DIR" --steps steps.tsv --session agente.sess
```

Prose SHA-256: `f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f`. Chat session SHA-256: `8441e1a125a1c7647315c0c10104088b7d5298fd46f5c56036ee39bad74cf5fd`.

## Retained samples

Rows follow the table order above. Times and throughputs use the precision recorded by the benchmark.

```csv
arm,workload_id,round,context_tokens,new_prompt_tokens,generated_tokens,prefill_s,prefill_tok_s,first_decode_ms,steady_decode_tok_s
baseline,1,1,64,64,128,0.043,1504.82,27.2,254.495
baseline,1,2,64,64,128,0.042,1514.17,27.0,255.465
baseline,1,3,64,64,128,0.041,1543.34,26.8,255.582
trial,1,1,64,64,128,0.021,3107.58,3.8,267.214
trial,1,2,64,64,128,0.021,3079.10,3.8,267.103
trial,1,3,64,64,128,0.021,2991.09,3.8,265.878
baseline,2,1,512,512,128,0.086,5945.78,3.9,253.324
baseline,2,2,512,512,128,0.086,5983.22,3.9,255.408
baseline,2,3,512,512,128,0.086,5987.87,3.9,255.238
trial,2,1,512,512,128,0.038,13589.05,3.7,268.542
trial,2,2,512,512,128,0.038,13510.59,3.7,267.802
trial,2,3,512,512,128,0.038,13575.48,3.7,268.885
baseline,3,1,64,64,128,0.023,2772.15,3.9,256.731
baseline,3,2,64,64,128,0.023,2778.61,3.9,257.562
baseline,3,3,64,64,128,0.023,2781.43,3.9,257.587
trial,3,1,64,64,128,0.014,4535.41,3.7,268.851
trial,3,2,64,64,128,0.014,4574.12,3.7,269.009
trial,3,3,64,64,128,0.014,4485.93,3.7,269.448
baseline,4,1,512,512,128,0.102,5027.78,26.8,255.787
baseline,4,2,512,512,128,0.100,5094.59,26.1,255.845
baseline,4,3,512,512,128,0.101,5087.62,25.7,255.807
trial,4,1,512,512,128,0.044,11547.50,3.8,269.265
trial,4,2,512,512,128,0.043,11790.79,3.8,269.104
trial,4,3,512,512,128,0.044,11726.00,3.7,269.346
baseline,5,1,2048,2048,128,0.294,6965.76,24.7,255.363
baseline,5,2,2048,2048,128,0.294,6958.13,24.5,254.572
baseline,5,3,2048,2048,128,0.293,6980.41,24.5,254.578
trial,5,1,2048,2048,128,0.121,16885.46,3.9,267.745
trial,5,2,2048,2048,128,0.122,16837.35,3.9,267.499
trial,5,3,2048,2048,128,0.122,16855.86,4.0,268.013
baseline,6,1,4096,4096,128,0.583,7022.75,4.0,254.529
baseline,6,2,4096,4096,128,0.583,7031.74,4.0,254.849
baseline,6,3,4096,4096,128,0.583,7020.70,4.0,251.485
trial,6,1,4096,4096,128,0.227,18031.52,3.8,267.250
trial,6,2,4096,4096,128,0.227,18058.36,3.8,266.964
trial,6,3,4096,4096,128,0.227,18031.21,3.8,266.887
baseline,7,1,8192,8192,128,0.874,9377.94,4.8,247.273
baseline,7,2,8192,8192,128,0.868,9439.84,4.0,246.879
baseline,7,3,8192,8192,128,0.876,9349.99,4.0,249.965
trial,7,1,8192,8192,128,0.464,17665.26,3.9,259.668
trial,7,2,8192,8192,128,0.463,17684.99,3.9,259.434
trial,7,3,8192,8192,128,0.464,17639.00,3.9,259.533
baseline,8,1,16384,16384,128,1.965,8336.93,4.3,239.557
baseline,8,2,16384,16384,128,1.959,8364.30,4.2,243.654
baseline,8,3,16384,16384,128,1.935,8467.12,4.2,243.967
trial,8,1,16384,16384,128,0.992,16518.42,4.2,253.424
trial,8,2,16384,16384,128,0.991,16525.43,4.0,253.617
trial,8,3,16384,16384,128,0.993,16505.88,4.0,253.065
baseline,9,1,663,663,57,0.214,3103.21,57.2,266.229
baseline,9,2,663,663,57,0.213,3117.64,55.6,266.938
baseline,9,3,663,663,57,0.212,3122.28,56.7,267.693
trial,9,1,663,663,57,0.095,7014.13,12.1,296.755
trial,9,2,663,663,57,0.096,6924.26,12.5,297.613
trial,9,3,663,663,57,0.094,7039.55,12.1,297.185
baseline,10,1,1954,1234,32,0.262,4717.40,34.5,288.555
baseline,10,2,1954,1234,32,0.235,5247.61,35.0,288.591
baseline,10,3,1954,1234,32,0.234,5273.71,34.4,288.514
trial,10,1,1954,1234,32,0.078,15843.40,32.3,315.715
trial,10,2,1954,1234,32,0.078,15797.17,32.5,315.652
trial,10,3,1954,1234,32,0.078,15835.79,32.5,316.367
baseline,11,1,4248,2262,87,0.433,5226.53,34.2,284.261
baseline,11,2,4248,2262,87,0.432,5237.03,33.7,284.757
baseline,11,3,4248,2262,87,0.441,5133.24,33.3,284.766
trial,11,1,4248,2262,87,0.140,16198.24,17.2,298.365
trial,11,2,4248,2262,87,0.139,16254.13,14.2,298.541
trial,11,3,4248,2262,87,0.139,16235.29,19.4,297.798
baseline,12,1,5741,1406,44,0.222,6329.48,12.2,350.437
baseline,12,2,5741,1406,44,0.225,6259.00,12.0,350.725
baseline,12,3,5741,1406,44,0.224,6270.82,12.0,351.407
trial,12,1,5741,1406,44,0.098,14383.37,33.7,365.155
trial,12,2,5741,1406,44,0.098,14376.05,33.8,366.144
trial,12,3,5741,1406,44,0.098,14343.31,33.7,365.658
baseline,13,1,7083,1298,54,0.213,6082.96,11.9,315.481
baseline,13,2,7083,1298,54,0.214,6076.41,11.8,316.272
baseline,13,3,7083,1298,54,0.214,6076.12,11.8,316.279
trial,13,1,7083,1298,54,0.141,9193.61,12.5,328.305
trial,13,2,7083,1298,54,0.141,9224.27,12.6,329.276
trial,13,3,7083,1298,54,0.141,9222.98,12.5,328.246
baseline,14,1,8041,904,24,0.144,6291.64,12.0,296.356
baseline,14,2,8041,904,24,0.144,6299.39,11.9,297.184
baseline,14,3,8041,904,24,0.143,6304.21,12.0,297.049
trial,14,1,8041,904,24,0.076,11928.20,12.5,312.504
trial,14,2,8041,904,24,0.076,11944.03,12.5,313.042
trial,14,3,8041,904,24,0.076,11928.41,12.2,313.173
baseline,15,1,8413,348,467,0.083,4172.34,12.2,289.863
baseline,15,2,8413,348,467,0.083,4170.44,11.9,288.777
baseline,15,3,8413,348,467,0.083,4176.65,12.0,290.307
trial,15,1,8413,348,467,0.046,7633.01,12.5,303.456
trial,15,2,8413,348,467,0.045,7674.94,12.3,304.054
trial,15,3,8413,348,467,0.045,7678.18,12.3,302.906
baseline,16,1,8915,35,186,0.053,654.75,11.9,279.043
baseline,16,2,8915,35,186,0.054,654.16,11.8,278.455
baseline,16,3,8915,35,186,0.053,657.05,11.9,279.214
trial,16,1,8915,35,186,0.029,1193.28,11.9,292.143
trial,16,2,8915,35,186,0.029,1192.92,11.9,290.658
trial,16,3,8915,35,186,0.029,1195.91,11.7,300.022
```
