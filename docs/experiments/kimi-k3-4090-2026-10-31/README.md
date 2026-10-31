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
