# NVIDIA RTX PRO 6000 Blackwell Server Edition vs dev, full CUDA stack

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
| Cold 64 | 1,820.06 | 2,295.10 | +26.10% | 238.76 | 246.37 | +3.18% |
| 512 in sequence | 8,152.14 | 12,850.58 | +57.38% | 239.44 | 247.54 | +3.38% |
| Final 64 | 2,860.77 | 4,062.59 | +41.82% | 239.88 | 247.43 | +3.18% |
| Cold 512 | 7,039.96 | 9,870.30 | +40.44% | 240.03 | 247.69 | +3.13% |
| 2,048 | 10,980.47 | 18,496.82 | +68.45% | 239.50 | 247.49 | +3.34% |
| 4,096 | 11,559.86 | 21,300.76 | +84.26% | 237.71 | 246.78 | +3.80% |
| 8,192 | 13,309.75 | 21,277.52 | +59.86% | 237.05 | 242.46 | +2.32% |
| 16,384 | 11,747.75 | 20,490.22 | +74.38% | 230.20 | 240.86 | +3.48% |
| Chat 1 | 4,580.82 | 7,125.02 | +56.69% | 268.37 | 268.97 | +0.26% |
| Chat 2 | 8,437.08 | 17,185.40 | +103.69% | 239.15 | 292.88 | +22.41% |
| Chat 3 | 8,587.28 | 18,630.54 | +116.96% | 239.67 | 275.47 | +14.89% |
| Chat 4 | 8,748.70 | 16,506.34 | +88.72% | 313.03 | 336.38 | +7.96% |
| Chat 5 | 7,794.50 | 14,802.67 | +90.13% | 290.02 | 300.96 | +3.70% |
| Chat 6 | 7,417.95 | 9,029.30 | +21.93% | 276.54 | 284.82 | +3.00% |
| Chat 7 | 4,457.92 | 7,283.28 | +63.03% | 269.97 | 278.42 | +3.51% |
| Chat 8 | 630.09 | 1,232.66 | +95.63% | 262.38 | 273.64 | +4.43% |

## Benchmark invocations

The commands show the recorded benchmark arguments with portable file paths. Set MODEL_DIR to the checkpoint directory and use the binary for each build. The timing runs also recorded the FP32 comparisons reported in correctness.txt.

```sh
COLI_CUDA=1 COLI_GPUS=0 CUDA_EXPERT_GB=auto OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close OMP_NUM_THREADS=24 OMP_DISPLAY_AFFINITY=TRUE OMP_AFFINITY_FORMAT=omp-thread:%n:cpus:%A /usr/bin/time -v ./qwen36-bench --model "$MODEL_DIR" --steps steps.tsv --text promessi_sposi.txt --lengths 64,512,64 --gen 128 --teacher
COLI_CUDA=1 COLI_GPUS=0 CUDA_EXPERT_GB=auto OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close OMP_NUM_THREADS=24 OMP_DISPLAY_AFFINITY=TRUE OMP_AFFINITY_FORMAT=omp-thread:%n:cpus:%A /usr/bin/time -v ./qwen36-bench --model "$MODEL_DIR" --steps steps.tsv --text promessi_sposi.txt --lengths 512 --gen 128 --teacher
COLI_CUDA=1 COLI_GPUS=0 CUDA_EXPERT_GB=auto OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close OMP_NUM_THREADS=24 OMP_DISPLAY_AFFINITY=TRUE OMP_AFFINITY_FORMAT=omp-thread:%n:cpus:%A /usr/bin/time -v ./qwen36-bench --model "$MODEL_DIR" --steps steps.tsv --text promessi_sposi.txt --lengths 2048,4096,8192,16384 --gen 128 --teacher
COLI_CUDA=1 COLI_GPUS=0 CUDA_EXPERT_GB=auto OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close OMP_NUM_THREADS=24 OMP_DISPLAY_AFFINITY=TRUE OMP_AFFINITY_FORMAT=omp-thread:%n:cpus:%A /usr/bin/time -v ./qwen36-bench --model "$MODEL_DIR" --steps steps.tsv --session agente.sess
```

Prose SHA-256: `f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f`. Chat session SHA-256: `8441e1a125a1c7647315c0c10104088b7d5298fd46f5c56036ee39bad74cf5fd`.

## Retained samples

Rows follow the table order above. Times and throughputs use the precision recorded by the benchmark.

```csv
arm,workload_id,round,context_tokens,new_prompt_tokens,generated_tokens,prefill_s,prefill_tok_s,first_decode_ms,steady_decode_tok_s
baseline,1,1,64,64,128,0.035,1815.78,13.0,238.710
baseline,1,2,64,64,128,0.035,1824.21,13.1,238.780
baseline,1,3,64,64,128,0.035,1820.06,13.0,238.762
trial,1,1,64,64,128,0.028,2307.69,13.2,246.150
trial,1,2,64,64,128,0.028,2286.27,12.9,246.372
trial,1,3,64,64,128,0.028,2295.10,12.7,246.606
baseline,2,1,512,512,128,0.063,8152.14,4.2,238.933
baseline,2,2,512,512,128,0.062,8231.13,4.1,239.443
baseline,2,3,512,512,128,0.063,8149.28,4.2,239.524
trial,2,1,512,512,128,0.040,12829.74,4.0,246.956
trial,2,2,512,512,128,0.040,12858.83,4.0,247.539
trial,2,3,512,512,128,0.040,12850.58,4.0,247.916
baseline,3,1,64,64,128,0.022,2860.77,4.2,239.685
baseline,3,2,64,64,128,0.022,2856.35,4.2,239.876
baseline,3,3,64,64,128,0.022,2864.77,4.2,239.896
trial,3,1,64,64,128,0.016,4057.23,4.0,247.426
trial,3,2,64,64,128,0.016,4071.14,4.0,247.268
trial,3,3,64,64,128,0.016,4062.59,4.0,247.527
baseline,4,1,512,512,128,0.073,7043.02,12.8,240.165
baseline,4,2,512,512,128,0.073,7039.96,12.8,240.035
baseline,4,3,512,512,128,0.073,6994.59,12.8,240.014
trial,4,1,512,512,128,0.052,9891.31,12.8,247.693
trial,4,2,512,512,128,0.052,9870.30,12.6,247.782
trial,4,3,512,512,128,0.052,9843.80,12.8,247.530
baseline,5,1,2048,2048,128,0.187,10980.47,12.6,239.265
baseline,5,2,2048,2048,128,0.187,10957.32,12.5,239.502
baseline,5,3,2048,2048,128,0.186,10981.86,12.5,240.163
trial,5,1,2048,2048,128,0.111,18496.82,12.5,246.505
trial,5,2,2048,2048,128,0.111,18435.10,12.7,247.494
trial,5,3,2048,2048,128,0.110,18553.11,12.7,248.185
baseline,6,1,4096,4096,128,0.354,11559.86,4.1,237.706
baseline,6,2,4096,4096,128,0.353,11590.94,4.1,235.620
baseline,6,3,4096,4096,128,0.354,11554.95,4.1,240.778
trial,6,1,4096,4096,128,0.192,21300.76,4.0,246.742
trial,6,2,4096,4096,128,0.192,21285.44,4.0,246.781
trial,6,3,4096,4096,128,0.192,21302.03,4.0,246.824
baseline,7,1,8192,8192,128,0.615,13325.33,4.2,236.102
baseline,7,2,8192,8192,128,0.615,13309.75,4.2,237.054
baseline,7,3,8192,8192,128,0.615,13309.70,4.2,237.252
trial,7,1,8192,8192,128,0.385,21303.66,4.1,241.578
trial,7,2,8192,8192,128,0.385,21277.52,4.1,242.456
trial,7,3,8192,8192,128,0.386,21246.83,4.1,243.580
baseline,8,1,16384,16384,128,1.394,11755.22,4.3,227.860
baseline,8,2,16384,16384,128,1.396,11737.02,4.4,233.164
baseline,8,3,16384,16384,128,1.395,11747.75,4.4,230.197
trial,8,1,16384,16384,128,0.800,20490.22,4.2,235.792
trial,8,2,16384,16384,128,0.799,20498.01,4.3,240.863
trial,8,3,16384,16384,128,0.800,20485.52,4.2,242.099
baseline,9,1,663,663,57,0.145,4580.82,29.0,268.280
baseline,9,2,663,663,57,0.144,4594.61,29.3,268.366
baseline,9,3,663,663,57,0.147,4519.31,29.2,268.384
trial,9,1,663,663,57,0.092,7177.66,19.3,268.974
trial,9,2,663,663,57,0.093,7112.68,19.4,268.963
trial,9,3,663,663,57,0.093,7125.02,19.1,269.101
baseline,10,1,1954,1234,32,0.146,8445.07,23.8,239.256
baseline,10,2,1954,1234,32,0.146,8437.08,23.5,239.148
baseline,10,3,1954,1234,32,0.147,8418.91,24.7,239.072
trial,10,1,1954,1234,32,0.072,17126.24,16.4,292.883
trial,10,2,1954,1234,32,0.072,17185.40,16.7,292.704
trial,10,3,1954,1234,32,0.071,17274.59,15.6,293.405
baseline,11,1,4248,2262,87,0.263,8587.28,10.6,240.436
baseline,11,2,4248,2262,87,0.263,8589.89,10.6,239.667
baseline,11,3,4248,2262,87,0.264,8569.21,10.6,239.221
trial,11,1,4248,2262,87,0.121,18630.54,10.6,275.490
trial,11,2,4248,2262,87,0.121,18652.65,10.6,275.356
trial,11,3,4248,2262,87,0.122,18584.85,10.6,275.474
baseline,12,1,5741,1406,44,0.160,8778.60,10.6,313.028
baseline,12,2,5741,1406,44,0.161,8727.00,10.7,313.552
baseline,12,3,5741,1406,44,0.161,8748.70,10.6,311.332
trial,12,1,5741,1406,44,0.085,16589.05,23.3,337.955
trial,12,2,5741,1406,44,0.085,16469.22,23.6,336.380
trial,12,3,5741,1406,44,0.085,16506.34,23.0,336.180
baseline,13,1,7083,1298,54,0.167,7794.50,23.2,290.025
baseline,13,2,7083,1298,54,0.166,7835.75,23.1,290.567
baseline,13,3,7083,1298,54,0.167,7771.64,23.4,289.247
trial,13,1,7083,1298,54,0.088,14819.80,10.7,300.748
trial,13,2,7083,1298,54,0.088,14783.00,10.6,301.317
trial,13,3,7083,1298,54,0.088,14802.67,10.6,300.965
baseline,14,1,8041,904,24,0.122,7417.95,10.7,276.537
baseline,14,2,8041,904,24,0.121,7463.25,10.6,277.513
baseline,14,3,8041,904,24,0.123,7345.86,10.5,274.856
trial,14,1,8041,904,24,0.100,9044.91,10.8,284.820
trial,14,2,8041,904,24,0.101,8922.58,10.7,283.468
trial,14,3,8041,904,24,0.100,9029.30,10.7,287.168
baseline,15,1,8413,348,467,0.078,4441.37,10.5,269.973
baseline,15,2,8413,348,467,0.078,4457.92,10.6,270.063
baseline,15,3,8413,348,467,0.078,4467.42,10.5,268.962
trial,15,1,8413,348,467,0.047,7410.58,10.5,280.605
trial,15,2,8413,348,467,0.048,7254.34,10.5,277.913
trial,15,3,8413,348,467,0.048,7283.28,10.5,278.416
baseline,16,1,8915,35,186,0.056,630.09,10.5,262.384
baseline,16,2,8915,35,186,0.055,631.55,10.6,262.652
baseline,16,3,8915,35,186,0.056,629.89,10.6,261.316
trial,16,1,8915,35,186,0.028,1232.66,11.4,266.877
trial,16,2,8915,35,186,0.028,1238.34,11.3,274.290
trial,16,3,8915,35,186,0.028,1230.85,11.2,273.637
```
