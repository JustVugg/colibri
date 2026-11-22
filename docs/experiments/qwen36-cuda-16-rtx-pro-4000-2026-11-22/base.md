# NVIDIA RTX PRO 4000 Blackwell vs dev, full CUDA stack

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
| Cold 64 | 811.56 | 1,799.47 | +122.08% | 144.94 | 147.40 | +1.79% |
| 512 in sequence | 3,080.67 | 6,588.92 | +113.94% | 144.95 | 147.43 | +1.68% |
| Final 64 | 1,607.62 | 2,588.32 | +61.00% | 146.83 | 148.80 | +1.34% |
| Cold 512 | 2,687.14 | 5,813.31 | +117.94% | 144.89 | 147.25 | +1.63% |
| 2,048 | 3,542.50 | 7,323.85 | +106.74% | 144.06 | 146.87 | +1.95% |
| 4,096 | 3,563.48 | 7,432.58 | +108.50% | 140.89 | 145.83 | +3.51% |
| 8,192 | 4,621.25 | 7,280.51 | +57.52% | 139.41 | 140.12 | +0.50% |
| 16,384 | 4,156.02 | 6,886.68 | +65.73% | 133.65 | 136.01 | +1.90% |
| Chat 1 | 1,663.85 | 3,492.43 | +110.83% | 149.23 | 161.83 | +8.34% |
| Chat 2 | 2,710.89 | 6,920.96 | +155.30% | 164.27 | 168.99 | +2.87%† |
| Chat 3 | 2,680.80 | 6,823.92 | +154.55% | 160.29 | 162.84 | +1.59% |
| Chat 4 | 3,344.81 | 6,451.11 | +92.87% | 185.25 | 195.75 | +5.58% |
| Chat 5 | 2,967.76 | 5,480.17 | +84.84% | 172.29 | 175.19 | +1.69% |
| Chat 6 | 3,292.48 | 4,776.05 | +45.06% | 164.41 | 165.63 | +0.85% |
| Chat 7 | 2,259.59 | 3,299.18 | +45.80% | 161.88 | 161.97 | +0.10% |
| Chat 8 | 460.62 | 777.17 | +68.58% | 158.83 | 158.25 | -0.37% |

## Benchmark invocations

The commands show the recorded benchmark arguments with portable file paths. Set MODEL_DIR to the checkpoint directory and use the binary for each build. The timing runs also recorded the FP32 comparisons reported in correctness.txt.

```sh
COLI_CUDA=1 COLI_GPUS=0 CUDA_EXPERT_GB=auto OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close OMP_NUM_THREADS=10 OMP_DISPLAY_AFFINITY=TRUE OMP_AFFINITY_FORMAT=omp-thread:%n:cpus:%A /usr/bin/time -v ./qwen36-bench --model "$MODEL_DIR" --steps steps.tsv --text promessi_sposi.txt --lengths 64,512,64 --gen 128 --teacher
COLI_CUDA=1 COLI_GPUS=0 CUDA_EXPERT_GB=auto OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close OMP_NUM_THREADS=10 OMP_DISPLAY_AFFINITY=TRUE OMP_AFFINITY_FORMAT=omp-thread:%n:cpus:%A /usr/bin/time -v ./qwen36-bench --model "$MODEL_DIR" --steps steps.tsv --text promessi_sposi.txt --lengths 512 --gen 128 --teacher
COLI_CUDA=1 COLI_GPUS=0 CUDA_EXPERT_GB=auto OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close OMP_NUM_THREADS=10 OMP_DISPLAY_AFFINITY=TRUE OMP_AFFINITY_FORMAT=omp-thread:%n:cpus:%A /usr/bin/time -v ./qwen36-bench --model "$MODEL_DIR" --steps steps.tsv --text promessi_sposi.txt --lengths 2048,4096,8192,16384 --gen 128 --teacher
COLI_CUDA=1 COLI_GPUS=0 CUDA_EXPERT_GB=auto OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close OMP_NUM_THREADS=10 OMP_DISPLAY_AFFINITY=TRUE OMP_AFFINITY_FORMAT=omp-thread:%n:cpus:%A /usr/bin/time -v ./qwen36-bench --model "$MODEL_DIR" --steps steps.tsv --session agente.sess
```

Prose SHA-256: `f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f`. Chat session SHA-256: `8441e1a125a1c7647315c0c10104088b7d5298fd46f5c56036ee39bad74cf5fd`.

## Retained samples

Rows follow the table order above. Times and throughputs use the precision recorded by the benchmark.

```csv
arm,workload_id,round,context_tokens,new_prompt_tokens,generated_tokens,prefill_s,prefill_tok_s,first_decode_ms,steady_decode_tok_s
baseline,1,1,64,64,128,0.079,811.56,29.4,143.980
baseline,1,2,64,64,128,0.074,864.05,29.7,144.945
baseline,1,3,64,64,128,0.080,798.04,29.9,144.960
trial,1,1,64,64,128,0.036,1802.34,6.8,147.377
trial,1,2,64,64,128,0.036,1799.47,6.8,147.537
trial,1,3,64,64,128,0.036,1789.54,6.8,147.401
baseline,2,1,512,512,128,0.171,2994.57,6.9,144.124
baseline,2,2,512,512,128,0.166,3092.00,6.9,144.949
baseline,2,3,512,512,128,0.166,3080.67,6.9,144.998
trial,2,1,512,512,128,0.078,6588.92,6.8,142.215
trial,2,2,512,512,128,0.078,6587.79,6.8,147.643
trial,2,3,512,512,128,0.078,6590.88,6.8,147.433
baseline,3,1,64,64,128,0.040,1596.77,6.8,146.830
baseline,3,2,64,64,128,0.040,1607.62,6.8,146.861
baseline,3,3,64,64,128,0.040,1607.85,6.8,146.808
trial,3,1,64,64,128,0.025,2586.83,6.7,148.804
trial,3,2,64,64,128,0.025,2588.32,6.7,148.964
trial,3,3,64,64,128,0.025,2588.67,6.7,148.782
baseline,4,1,512,512,128,0.191,2687.14,28.7,145.109
baseline,4,2,512,512,128,0.194,2638.78,29.1,143.404
baseline,4,3,512,512,128,0.190,2696.29,28.7,144.889
trial,4,1,512,512,128,0.087,5856.37,6.9,147.479
trial,4,2,512,512,128,0.088,5813.31,7.0,147.192
trial,4,3,512,512,128,0.090,5706.82,6.9,147.251
baseline,5,1,2048,2048,128,0.575,3562.67,29.0,144.251
baseline,5,2,2048,2048,128,0.578,3542.50,29.2,144.056
baseline,5,3,2048,2048,128,0.581,3522.41,29.1,143.951
trial,5,1,2048,2048,128,0.280,7305.88,7.3,146.808
trial,5,2,2048,2048,128,0.280,7323.85,7.3,146.869
trial,5,3,2048,2048,128,0.279,7331.47,7.3,146.948
baseline,6,1,4096,4096,128,1.149,3564.74,8.7,143.583
baseline,6,2,4096,4096,128,1.149,3563.48,7.0,140.887
baseline,6,3,4096,4096,128,1.153,3551.92,7.0,140.705
trial,6,1,4096,4096,128,0.551,7432.58,7.0,145.857
trial,6,2,4096,4096,128,0.551,7429.07,6.9,145.826
trial,6,3,4096,4096,128,0.551,7433.17,6.9,145.807
baseline,7,1,8192,8192,128,1.755,4668.39,7.2,140.245
baseline,7,2,8192,8192,128,1.774,4616.56,7.2,138.869
baseline,7,3,8192,8192,128,1.773,4621.25,7.2,139.405
trial,7,1,8192,8192,128,1.125,7280.51,7.2,140.188
trial,7,2,8192,8192,128,1.125,7280.65,7.1,140.124
trial,7,3,8192,8192,128,1.125,7279.45,7.1,140.103
baseline,8,1,16384,16384,128,3.979,4117.65,7.5,133.653
baseline,8,2,16384,16384,128,3.942,4156.02,7.5,133.782
baseline,8,3,16384,16384,128,3.937,4161.24,7.5,133.138
trial,8,1,16384,16384,128,2.379,6886.68,7.4,136.193
trial,8,2,16384,16384,128,2.379,6887.94,7.4,136.013
trial,8,3,16384,16384,128,2.508,6531.74,7.4,135.973
baseline,9,1,663,663,57,0.398,1663.85,80.7,149.370
baseline,9,2,663,663,57,0.398,1664.65,81.6,148.735
baseline,9,3,663,663,57,0.401,1652.00,80.6,149.232
trial,9,1,663,663,57,0.189,3507.89,21.6,161.830
trial,9,2,663,663,57,0.190,3492.43,21.2,162.277
trial,9,3,663,663,57,0.190,3485.93,21.2,161.381
baseline,10,1,1954,1234,32,0.456,2704.77,64.8,144.547
baseline,10,2,1954,1234,32,0.455,2710.89,63.4,164.272
baseline,10,3,1954,1234,32,0.454,2719.15,65.6,164.361
trial,10,1,1954,1234,32,0.178,6921.48,21.3,169.751
trial,10,2,1954,1234,32,0.178,6920.96,21.0,168.988
trial,10,3,1954,1234,32,0.179,6901.80,21.5,168.757
baseline,11,1,4248,2262,87,0.844,2680.80,21.3,159.631
baseline,11,2,4248,2262,87,0.838,2700.73,21.8,160.291
baseline,11,3,4248,2262,87,0.860,2629.84,21.9,160.462
trial,11,1,4248,2262,87,0.331,6823.89,34.7,162.574
trial,11,2,4248,2262,87,0.331,6823.92,34.5,162.841
trial,11,3,4248,2262,87,0.331,6827.95,31.6,162.937
baseline,12,1,5741,1406,44,0.420,3349.40,73.3,184.985
baseline,12,2,5741,1406,44,0.420,3344.81,75.0,185.402
baseline,12,3,5741,1406,44,0.424,3318.41,73.8,185.251
trial,12,1,5741,1406,44,0.218,6443.75,74.6,195.136
trial,12,2,5741,1406,44,0.218,6451.11,69.3,195.749
trial,12,3,5741,1406,44,0.218,6456.30,71.5,196.224
baseline,13,1,7083,1298,54,0.439,2959.10,21.9,172.289
baseline,13,2,7083,1298,54,0.437,2967.76,21.3,173.759
baseline,13,3,7083,1298,54,0.437,2969.29,22.2,170.430
trial,13,1,7083,1298,54,0.237,5480.17,21.1,175.193
trial,13,2,7083,1298,54,0.237,5467.80,21.4,175.253
trial,13,3,7083,1298,54,0.236,5488.51,22.0,174.419
baseline,14,1,8041,904,24,0.275,3292.48,21.8,164.412
baseline,14,2,8041,904,24,0.274,3295.11,21.6,164.241
baseline,14,3,8041,904,24,0.275,3283.29,22.0,164.470
trial,14,1,8041,904,24,0.189,4776.05,21.5,165.410
trial,14,2,8041,904,24,0.189,4771.08,21.1,165.630
trial,14,3,8041,904,24,0.189,4795.20,21.4,165.879
baseline,15,1,8413,348,467,0.153,2278.07,22.2,162.004
baseline,15,2,8413,348,467,0.154,2257.03,22.2,161.876
baseline,15,3,8413,348,467,0.154,2259.59,21.4,160.520
trial,15,1,8413,348,467,0.105,3301.49,22.0,161.635
trial,15,2,8413,348,467,0.106,3290.78,21.2,162.041
trial,15,3,8413,348,467,0.105,3299.18,21.9,161.965
baseline,16,1,8915,35,186,0.076,460.62,21.9,158.858
baseline,16,2,8915,35,186,0.077,456.69,21.6,158.829
baseline,16,3,8915,35,186,0.076,461.01,22.0,157.718
trial,16,1,8915,35,186,0.046,766.88,21.7,155.033
trial,16,2,8915,35,186,0.045,781.28,20.9,158.249
trial,16,3,8915,35,186,0.045,777.17,21.4,158.420
```
