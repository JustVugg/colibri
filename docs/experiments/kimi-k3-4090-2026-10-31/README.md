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

## Runs 14-18: the GPU's one-row kernel, reads in parts, where the waits are

| run | engine / settings | prompt | decode | waiting for reads | note |
|---|---|---:|---:|---:|---|
| run14_qmr | + `quant_matvec_warp` for one row on the GPU | 38.5 s | 0.78 tok/s (57.8 s) | 353 ms | attention 178 -> 136 ms; the GPU's 16,605 calls 4.7 -> 2.8 s |
| run15_parts4 | run14, every unbuffered read cut in 4 parts | 43.7 s | 0.70 tok/s (64.1 s) | 498 ms | slower reads; not kept |
| run16_parts2 | the same, 2 parts | 38.8 s | 0.78 tok/s (57.8 s) | 360 ms | no gain; not kept |
| run17_wait | run14, `K3_WAIT_LAYERS=1` | 38.4 s | 0.78 tok/s (57.7 s) | 349 ms | the waits and demand reads by layer |
| run18_long | run17 and `K3_REUSE` on another prompt, 320 tokens | 47.7 s (52 tokens) | 0.79 tok/s (403.9 s) | 309 ms | LRU hit curves by layer |

The one-row kernel ([docs/kimi_k3-4090.md](../../kimi_k3-4090.md)) gives the GPU's bytes of
before and the same text, and takes 42 ms off a token's attention, but not off the token:
the wait for the drives grew by as much (299 -> 353 ms). A decode token here is as fast as
the experts that are neither in RAM nor predicted arrive.

Cutting each 17.5 MB read in parts so that a drive's two readers share an expert and a
demand waits for a part of a prefetch, not a whole one (runs 15, 16), made the reads
slower: the 35-token prompt, which only reads, took 43.7 instead of 38.5 s with 4.4 MB
parts. These drives want large requests; the parts were taken out again.

`K3_WAIT_LAYERS=1` (run17, and run18 below) splits the wait by layer: about 1.3 ms of it
for every expert a layer reads on demand (one the cache lacks and the prediction did not
name). Layers 1-8 read 4-7 that way a token and wait 6-12 ms each, layers 49-59 about one
and under a millisecond; layer 1, whose prediction has only layer 0 to hide behind and
whose cache holds 26% of what it is asked, alone waits 12 ms a token.

`K3_REUSE` (run18) records every routed expert's LRU stack distance. The decode's hit
rate over all layers that an LRU cache of each size would have had, every layer the same:

| slots a layer | 48 | 64 | 80 | **97** (here) | 120 | 160 | 200 |
|---|---:|---:|---:|---:|---:|---:|---:|
| LRU hit rate, decode | 53.7% | 58.4% | 61.8% | **64.7%** | 68.0% | 72.9% | 76.6% |

Given to the layers where they hold most, the same 8,924 slots hit 64.9% (0.6% fewer
misses): the layers' curves have the same shape, and a split of the cache by layer is
not worth its code. The prediction holds least at the start of an AttnRes block (layer 12:
10% of its experts, 24: 28%, 36: 30%; 58% on average): there the MLP's input is mostly
the layer's own attention, which the layer before cannot know.

## Run 19: a 2,637-token prompt

`raw/run19_long512.*`: the first 9,000 characters of docs/kimi_k3.md to summarize, in
512-token steps: **1,401 s (1.88 tokens a second)**, 265 s a step, of which attention
388 s and the MoE 993 s, only 29.6 s of it waiting for reads: a long prompt here is bound
by the CPU, not the drives (steps of 2,048 would read less and compute as much; the run
was stopped). Where the CPU goes, found after it:
- the gated MLA's scores: `s2+=qabs[i]*Lt[i]` is one sequential chain of 576 adds a
  position (GCC -O3 multiplies 16 at a time, then adds them in order), about 86 ms a token
  at 1,300 positions over 24 layers, and growing with the context in decode too; the same
  bytes 8 positions at a time (rounded products, added in i's order) is the next step;
- `matmul_i4_grouped` (the shared experts and latent projections on the CPU): 155 GMAC/s on
  16 threads at 512 rows, about 55 s a step; it decodes the weights again for every row;
- the shared experts' SiTU: one thread for 3.1M elements a layer, about 17 s a step
  (now over every worker, element by element, so the same bytes);
- `matmul_mxfp4_i8_pre` now decodes an output's weights once for 4 rows (the same bytes,
  `tests/test_mxfp4_i8_rows.c`): 1.2-1.6x a thread for an expert's 4-37 rows.

