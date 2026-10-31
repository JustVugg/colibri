# Kimi K3 on the RTX 4090 machine, 2026-10-31: the first measurements

moonshotai/Kimi-K3 at f831ab66, laid out by `tools/download_kimi_k3_4090.ps1` (every MoE
layer's experts 550 on `C:`, 346 on `D:`), on the Ryzen 9 7950X3D, 192 GB DDR5-5400, RTX
4090, `C:` 10.4 GB/s and `D:` 6.6 GB/s for 17.5 MB reads. Each run is `k3run.sh NAME 48`: a
fresh `kimi_k3.exe` (CUDA_DLL=1 build) on the chat prompt "What is the capital of France?
Answer in one short sentence." (35 tokens, thinking on), greedy, 48 tokens at most, with
the 4090 profile (`K3_CUDA_DENSE=1` 20 GiB with host copies released, `RAM_GB=174`,
`K3_CHUNK=512`, 16 threads, `AUTOPIN=0`); `monitor.ps1` samples both drives, the CPU, the
GPU and the engine's memory every 2 s (`raw/*.csv`). Every run gave the same text:

> [think] The user is asking a simple factual question: the capital of France. The answer
> is Paris. They want one short sentence.
>
> [response] The capital of France is Paris.

| run | engine | load | prompt (35 tokens) | decode (45 tokens) | read a decode token |
|---|---|---:|---:|---:|---:|
| run1 | 71c243f5 | 161.4 s | 57.0 s | 0.56 tok/s (80.7 s) | 16.8 GB |
| run2 | + quantization at load over every worker, the RAM budget without the released GPU host copies (83 -> 96 experts a layer), OpenMP regions on omp_spin.c's spinning team | 53.3 s | 57.0 s | 0.62 tok/s (73.1 s) | 15.7 GB |
| run3 | + a MoE step's experts in waves (k3_moe_waves) | 53.3 s | 41.3 s | **0.67 tok/s** (67.6 s) | 15.2 GB |
| run4_nopf | run3 with `K3_PREFETCH=0` | 53.3 s | 41.3 s | 0.63 tok/s (71.7 s) | 9.9 GB |

Where a decode token's time goes (run3): both drives about 73% busy, `C:` 5.9 GB/s and `D:`
3.9 GB/s on average (58% of what they read together), the CPU half busy. The next layer's
prediction covers 63% of the routed experts and pays its way (run4: 0.67 against 0.63 tok/s),
though about half its reads go unused: 5.3 GB more a token read for 0.1 s less. Without it the
RAM cache alone holds about 48% of the experts a token asks for (10.7% of all experts fit:
consecutive tokens reuse many). The rest of the time is the layers' sequence: a layer's
missing experts can be read only once its routing is known, and are computed after they arrive.

The prompt in run2 read one drive at a time (`raw/run2.csv`): experts_apply_union read a
step's missing experts 64 at a time in the order of their ids as text, and a cut layer keeps
the first ids on `C:`. k3_moe_waves gives every missing expert a slot at once, its read on its
drive, the two drives' reads interleaved, and computes whatever has arrived: 57.0 -> 41.3 s.

## Runs 5-13: the reads before the latent work, the prediction, BF16 routers

The same command and prompt; every run gave the same text again. Runs 5-11 are the
engine before its defaults changed (prediction from this layer's input, the routing's 16
experts a row), with the settings named; from run 12 the defaults are those of run 10.

| run | engine / settings | prompt | decode | read a decode token | waiting for reads | predicted |
|---|---|---:|---:|---:|---:|---:|
| run5 | + a step's reads issued before its latent projection and shared experts | 37.4 s | 0.72 tok/s (62.9 s) | 14.4 GB | 371 ms | 63.2% |
| run6_mix | run5, `K3_PREFETCH_MIX=1`: the prediction from the next layer's own mix | 37.3 s | 0.74 tok/s (61.2 s) | 13.8 GB | 335 ms | 66.9% |
| run7_top8 | run5, `K3_PREFETCH_TOP=8` | 37.2 s | 0.74 tok/s (61.2 s) | 11.2 GB | 362 ms | 40.9% |
| run8_io3 | run5, `K3_IO_PER_DRIVE=3` | 38.6 s | 0.69 tok/s (65.5 s) | 15.4 GB | 424 ms | 63.2% |
| run9_idle_mix | + the shared experts while no read has arrived; `MIX=1` | 37.1 s | 0.74 tok/s (60.9 s) | 13.8 GB | 323 ms | 66.9% |
| run10_idle_mix_t12 | run9, `TOP=12` | 37.1 s | 0.75 tok/s (59.9 s) | 12.2 GB | 314 ms | 57.1% |
| run11_idle_mix_t8 | run9, `TOP=8` | 37.1 s | 0.75 tok/s (60.3 s) | 10.8 GB | 333 ms | 42.7% |
| run12_bf16r | + the routers in BF16; defaults `MIX=1`, `TOP=12` | 37.1 s | **0.78 tok/s** (57.4 s) | 12.1 GB | 299 ms | 57.1% |
| run13_bf16r_gpu215 | run12, `K3_CUDA_DENSE_GB=21.5 K3_CUDA_RESERVE_GB=1` | 36.5 s | **0.81 tok/s** (55.9 s) | 12.1 GB | 286 ms | 57.0% |

Every run's last lines split a decode token's time (`raw/*.err`, "a decode token"); run12
against run10, the BF16 routers: the routing and the next layer's prediction, which read
every router twice a token, 94 -> 58 ms. Run13 put 35 more matrices on the GPU (404, 21.50
GiB): attention 178 -> 153 ms; `nvidia-smi` then showed 23.9 of the card's 24 GB in use.
The profile keeps 20 GiB (0.6 GiB free is little beside a browser: past the free memory
Windows pages GPU memory to RAM without an error); `-DenseGB 21.5 -ReserveGB 1` is the
faster choice when nothing else uses the card.

Three readers a drive (run8) wait longer than two: the drives were already as busy as
reads of 17.5 MB keep them, and the third reader's reads were mostly the prediction's.
Fewer predicted experts a row read less and wait about as long: 12 is where the reads
saved stop paying for the misses they leave. Of a decode token's 1.24 s in run13 the
routed experts' compute is 497 ms (25.8 GB of weights read from RAM, about 52 GB/s, with
the drives' 12.1 GB written into it; `membw.c` here read 66 GB/s with 16 threads) and the
wait for reads 286 ms; the start's 53-62 s are the BF16 weights read from `C:` at about 2
GB/s by the loader (`raw/run13_bf16r_gpu215.csv` is the whole run, every 2 s).
