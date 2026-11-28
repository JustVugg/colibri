# NVIDIA GeForce RTX 4090 vs dev, full CUDA stack

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
| Cold 64 | 1,098.29 | 2,487.29 | +125.90% | 174.59 | 187.23 | +7.12% |
| 512 in sequence | 4,114.44 | 10,185.07 | +147.54% | 174.86 | 188.28 | +7.63% |
| Final 64 | 1,887.06 | 3,448.03 | +82.72% | 176.07 | 188.60 | +7.06% |
| Cold 512 | 3,571.98 | 8,944.08 | +150.31% | 174.40 | 187.40 | +7.52% |
| 2,048 | 5,055.32 | 12,338.08 | +144.00% | 173.85 | 186.61 | +7.34% |
| 4,096 | 5,027.29 | 12,824.28 | +155.12% | 172.47 | 185.07 | +7.28% |
| 8,192 | 6,739.61 | 12,825.73 | +90.30% | 169.40 | 181.02 | +6.88% |
| 16,384 | 6,294.94 | 12,310.86 | +95.57% | 162.51 | 180.89 | +11.31% |
| Chat 1 | 2,286.32 | 5,195.74 | +127.52% | 187.17 | 199.64 | +7.21% |
| Chat 2 | 3,777.76 | 11,230.25 | +196.93% | 195.38 | 246.12 | +26.02% |
| Chat 3 | 3,768.44 | 11,404.80 | +202.95% | 193.60 | 202.03 | +4.35% |
| Chat 4 | 4,775.12 | 10,404.26 | +117.88% | 247.03 | 266.27 | +7.73% |
| Chat 5 | 4,375.66 | 9,211.54 | +110.23% | 220.02 | 234.81 | +6.63% |
| Chat 6 | 4,840.98 | 5,882.13 | +21.51% | 213.47 | 225.33 | +5.63% |
| Chat 7 | 3,134.49 | 5,704.83 | +82.05% | 202.10 | 218.53 | +8.13% |
| Chat 8 | 554.32 | 936.64 | +68.95% | 196.91 | 207.86 | +5.56% |

## Benchmark invocations

The commands show the recorded benchmark arguments with portable file paths. Set MODEL_DIR to the checkpoint directory and use the binary for each build. The timing runs also recorded the FP32 comparisons reported in correctness.txt.

```sh
COLI_CUDA=1 COLI_GPUS=0 CUDA_EXPERT_GB=auto OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close OMP_NUM_THREADS=15 OMP_DISPLAY_AFFINITY=TRUE OMP_AFFINITY_FORMAT=omp-thread:%n:cpus:%A /usr/bin/time -v ./qwen36-bench --model "$MODEL_DIR" --steps steps.tsv --text promessi_sposi.txt --lengths 64,512,64 --gen 128 --teacher
COLI_CUDA=1 COLI_GPUS=0 CUDA_EXPERT_GB=auto OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close OMP_NUM_THREADS=15 OMP_DISPLAY_AFFINITY=TRUE OMP_AFFINITY_FORMAT=omp-thread:%n:cpus:%A /usr/bin/time -v ./qwen36-bench --model "$MODEL_DIR" --steps steps.tsv --text promessi_sposi.txt --lengths 512 --gen 128 --teacher
COLI_CUDA=1 COLI_GPUS=0 CUDA_EXPERT_GB=auto OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close OMP_NUM_THREADS=15 OMP_DISPLAY_AFFINITY=TRUE OMP_AFFINITY_FORMAT=omp-thread:%n:cpus:%A /usr/bin/time -v ./qwen36-bench --model "$MODEL_DIR" --steps steps.tsv --text promessi_sposi.txt --lengths 2048,4096,8192,16384 --gen 128 --teacher
COLI_CUDA=1 COLI_GPUS=0 CUDA_EXPERT_GB=auto OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close OMP_NUM_THREADS=15 OMP_DISPLAY_AFFINITY=TRUE OMP_AFFINITY_FORMAT=omp-thread:%n:cpus:%A /usr/bin/time -v ./qwen36-bench --model "$MODEL_DIR" --steps steps.tsv --session agente.sess
```

Prose SHA-256: `f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f`. Chat session SHA-256: `8441e1a125a1c7647315c0c10104088b7d5298fd46f5c56036ee39bad74cf5fd`.

## Retained samples

Rows follow the table order above. Times and throughputs use the precision recorded by the benchmark.

```csv
arm,workload_id,round,context_tokens,new_prompt_tokens,generated_tokens,prefill_s,prefill_tok_s,first_decode_ms,steady_decode_tok_s
baseline,1,1,64,64,128,0.059,1090.39,30.2,174.872
baseline,1,2,64,64,128,0.058,1098.29,30.5,174.586
baseline,1,3,64,64,128,0.058,1107.15,30.4,174.529
trial,1,1,64,64,128,0.026,2463.18,28.9,187.288
trial,1,2,64,64,128,0.026,2487.29,28.4,187.228
trial,1,3,64,64,128,0.026,2492.27,28.7,186.954
baseline,2,1,512,512,128,0.124,4114.44,5.8,174.953
baseline,2,2,512,512,128,0.124,4120.30,5.8,174.859
baseline,2,3,512,512,128,0.125,4088.22,5.8,174.768
trial,2,1,512,512,128,0.050,10185.07,5.4,188.307
trial,2,2,512,512,128,0.050,10196.13,5.3,188.275
trial,2,3,512,512,128,0.050,10166.38,5.4,187.890
baseline,3,1,64,64,128,0.034,1887.56,5.7,176.163
baseline,3,2,64,64,128,0.034,1885.22,5.7,176.073
baseline,3,3,64,64,128,0.034,1887.06,5.7,175.943
trial,3,1,64,64,128,0.019,3446.76,5.3,188.603
trial,3,2,64,64,128,0.019,3452.66,5.3,188.603
trial,3,3,64,64,128,0.019,3448.03,5.3,187.996
baseline,4,1,512,512,128,0.143,3571.98,30.1,174.506
baseline,4,2,512,512,128,0.142,3596.05,30.1,174.399
baseline,4,3,512,512,128,0.144,3563.25,30.1,174.195
trial,4,1,512,512,128,0.057,8941.20,29.2,187.398
trial,4,2,512,512,128,0.057,8944.08,28.5,187.616
trial,4,3,512,512,128,0.057,8957.11,29.0,187.287
baseline,5,1,2048,2048,128,0.408,5022.30,30.2,173.704
baseline,5,2,2048,2048,128,0.405,5055.32,30.2,173.850
baseline,5,3,2048,2048,128,0.403,5085.33,29.6,174.158
trial,5,1,2048,2048,128,0.166,12338.08,29.2,186.591
trial,5,2,2048,2048,128,0.166,12335.16,29.1,186.606
trial,5,3,2048,2048,128,0.166,12343.44,29.5,186.716
baseline,6,1,4096,4096,128,0.813,5039.40,7.7,171.502
baseline,6,2,4096,4096,128,0.815,5027.29,6.5,173.169
baseline,6,3,4096,4096,128,0.817,5015.59,5.9,172.475
trial,6,1,4096,4096,128,0.320,12816.60,5.5,185.096
trial,6,2,4096,4096,128,0.319,12825.66,5.5,185.068
trial,6,3,4096,4096,128,0.319,12824.28,5.5,185.038
baseline,7,1,8192,8192,128,1.215,6739.61,6.0,169.680
baseline,7,2,8192,8192,128,1.220,6713.41,6.0,168.226
baseline,7,3,8192,8192,128,1.214,6748.86,6.0,169.403
trial,7,1,8192,8192,128,0.639,12825.73,5.6,180.944
trial,7,2,8192,8192,128,0.639,12823.37,5.6,181.020
trial,7,3,8192,8192,128,0.638,12836.39,5.7,181.062
baseline,8,1,16384,16384,128,2.603,6294.94,6.2,162.399
baseline,8,2,16384,16384,128,2.640,6206.07,6.3,162.510
baseline,8,3,16384,16384,128,2.601,6298.98,6.2,162.619
trial,8,1,16384,16384,128,1.331,12310.86,5.8,180.927
trial,8,2,16384,16384,128,1.332,12299.14,5.8,180.892
trial,8,3,16384,16384,128,1.331,12311.63,5.9,180.792
baseline,9,1,663,663,57,0.290,2283.65,65.1,187.167
baseline,9,2,663,663,57,0.289,2291.23,64.6,187.188
baseline,9,3,663,663,57,0.290,2286.32,66.7,185.714
trial,9,1,663,663,57,0.128,5195.74,44.2,200.657
trial,9,2,663,663,57,0.128,5191.64,43.8,199.140
trial,9,3,663,663,57,0.127,5208.84,44.1,199.638
baseline,10,1,1954,1234,32,0.327,3777.76,46.3,194.995
baseline,10,2,1954,1234,32,0.328,3765.30,46.1,195.494
baseline,10,3,1954,1234,32,0.326,3784.70,46.7,195.375
trial,10,1,1954,1234,32,0.110,11217.18,27.4,246.121
trial,10,2,1954,1234,32,0.110,11230.25,21.8,245.675
trial,10,3,1954,1234,32,0.110,11234.92,25.8,246.215
baseline,11,1,4248,2262,87,0.601,3763.82,45.8,196.471
baseline,11,2,4248,2262,87,0.599,3779.29,44.9,193.604
baseline,11,3,4248,2262,87,0.600,3768.44,45.4,190.595
trial,11,1,4248,2262,87,0.198,11402.55,19.5,201.671
trial,11,2,4248,2262,87,0.198,11404.80,19.7,202.029
trial,11,3,4248,2262,87,0.198,11424.39,19.5,202.127
baseline,12,1,5741,1406,44,0.294,4775.12,20.1,246.483
baseline,12,2,5741,1406,44,0.295,4764.65,20.1,247.032
baseline,12,3,5741,1406,44,0.294,4776.70,20.2,249.450
trial,12,1,5741,1406,44,0.135,10404.26,45.5,266.511
trial,12,2,5741,1406,44,0.135,10393.19,43.8,266.119
trial,12,3,5741,1406,44,0.135,10405.52,44.5,266.274
baseline,13,1,7083,1298,54,0.297,4375.66,19.9,220.019
baseline,13,2,7083,1298,54,0.296,4381.55,20.1,219.902
baseline,13,3,7083,1298,54,0.297,4371.78,20.2,220.206
trial,13,1,7083,1298,54,0.141,9216.60,19.5,234.327
trial,13,2,7083,1298,54,0.141,9211.54,20.0,235.018
trial,13,3,7083,1298,54,0.142,9171.14,19.6,234.806
baseline,14,1,8041,904,24,0.187,4840.98,20.1,212.744
baseline,14,2,8041,904,24,0.186,4849.05,20.0,213.525
baseline,14,3,8041,904,24,0.187,4832.80,20.0,213.465
trial,14,1,8041,904,24,0.154,5882.13,20.1,225.333
trial,14,2,8041,904,24,0.154,5874.02,20.3,225.553
trial,14,3,8041,904,24,0.154,5885.47,20.2,216.683
baseline,15,1,8413,348,467,0.112,3117.39,43.7,202.376
baseline,15,2,8413,348,467,0.111,3134.49,20.0,202.099
baseline,15,3,8413,348,467,0.111,3139.65,20.0,200.642
trial,15,1,8413,348,467,0.061,5703.79,20.1,218.436
trial,15,2,8413,348,467,0.061,5704.83,20.1,218.528
trial,15,3,8413,348,467,0.061,5715.62,20.0,218.569
baseline,16,1,8915,35,186,0.063,553.44,19.8,194.698
baseline,16,2,8915,35,186,0.063,554.32,20.0,196.914
baseline,16,3,8915,35,186,0.063,554.61,20.0,196.980
trial,16,1,8915,35,186,0.037,936.64,19.4,207.938
trial,16,2,8915,35,186,0.037,935.25,19.9,207.858
trial,16,3,8915,35,186,0.037,936.99,19.6,207.787
```
