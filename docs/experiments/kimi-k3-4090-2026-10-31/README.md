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

## Runs 21-25: a long prompt's CPU work, the same bytes

The CPU kernels that run19 found slow, rewritten so that every row and position keeps its
arithmetic: the gated MLA's scores 16 positions at a time (`k3_mla_scores`), kv_b's rows
8 at a time (`k3_rowdot8_i8`, from run 25), `matmul_i4_grouped` over 8 rows and a tile of
16, `matmul_mxfp4_i8_pre` over 4 rows, the shared experts' SiTU over every worker.

| run | what | result |
|---|---|---|
| run21_short | the usual 35-token prompt and 45 tokens | the same text; prompt 36.2 s; decode 0.78 tok/s (1,288 ms: shared experts 175 -> 159 ms, the reads' wait 353 -> 366) |
| run22_logits_new / run23_logits_head | `K3_LOGITS` of that prompt (`--ngen 0`), this engine and the commit before it | **the same 22,937,600 bytes** |
| run24_long512 | run19's 2,637-token prompt, 512-token steps | **844.5 s** (1,401.3): attention 200.4 s (388.3), MoE 621.9 s (993.0; routed experts 321.6, shared 97.2, latent 39.1, router 25.0, waiting for reads 119.1); 5,784 GB read |
| run25_long2048 | run24's engine, 2,048-token steps | **697.0 s**: attention 200.9 s, MoE 474.4 s (routed experts 268.8, waiting for reads 17.7); 2,455 GB read; the working set 160.3 GB at most |
| run26_logits / run27_short | + kv_b's rows 8 at a time, `res_mix`'s chains side by side, the step's rows over every worker | the logits the same bytes as run23's; the same text; decode 0.79 tok/s (1,268 ms, attention 129) |
| run28_long2048 | the same engine | **653.1 s**: attention 189.7 s, MoE 460.3 s, the rest 3.1 s (21.7) |
| run29-31 | + the MLA's scores six heads at a time | the logits the same bytes; the same text; the long prompt **639 s** (4.13 tok/s): attention 174.7 s (KDA 90.3, MLA 84.4), MoE 461.1 s |

The launcher now takes 2,048-token prompt steps (`-Chunk`, at most 4,096): the RAM budget
reserves 1.4 MB a row past 512 (2.1 GB at 2,048, one expert slot a layer less), and a
long prompt reads its experts a quarter as often.
## Runs 32-40: KDA by step, int8 and f32 rows, the GPU's tree

| run | what | result |
|---|---|---|
| run32-34 | KDA's conv channel by channel and heads head by head over a step's tokens | the logits the same bytes; the same text; decode 0.79 tok/s; the long prompt 626.0 s (KDA 90.3 -> 86.0) |
| run35-37 | + `matmul_q`'s rows 16 at a time; coli_cuda.dll's tiled kernel ending its tree in warp shuffles | the same bytes and text; 616.5 s (the gated MLA 83.6 -> 72.1, the GPU's calls 40.4 -> 36.4 s) |
| run38-40 | + matmul_f32.h's `matmul` over rows the same way | the same bytes and text; **613 s** (4.30 tok/s): attention 151.9 (KDA 79.8: on the CPU f32 projections 3.8, conv 11.2, heads 17.4; MLA 72.1), MoE 458.0, the GPU's calls 36.3 s |

`matmul_f32.h` and `quant.h` serve other engines too: MiMo's tiny model (`tools/make_mimo_tiny.py`,
`MIMO_LOGITS`, steps of 1 and 40 rows) gives the same logits with these headers as with the
commit before, and `tests/mimo_tiny_harness.py` passes.
## Runs 41-45: the routers' rows, a step's buffers kept

| run | what | result |
|---|---|---|
| run41-42 | + the routers' BF16 rows over a step's rows; the prompt line counts page faults | the logits the same bytes; 603.4 s (the routing 31.3 -> 23.1 s); **93.7 million page faults** in the prompt |
| run43-45 | + a step's big buffers kept from one layer to the next (`k3_balloc`), layer 0's SiTU over every worker | the same bytes and text; decode 0.79 tok/s; **581.8 s** (4.53 tok/s), 40.9 million page faults (the expert cache's 158 GB filled for the first time: 38.4 million in the 35-token prompt too) |
## Runs 46-51: the routed experts 8 rows at a time, 8 rows' sums together

| run | what | result |
|---|---|---|
| run46-48 | `matmul_mxfp4_i8_pre`: an output's weights decoded once for 8 rows, their group scales one vector multiply (1.47x a thread at 16-37 rows) | the logits the same bytes; the same text; 546.7 s (the routed experts 255.3 -> 215.5) |
| run49-51 | + `matmul_i4_grouped`: a group's 8 rows' hsum256 and fmaf at once (`hsum256_x8`, the same pairs; 1.13-1.3x in `bench_i4`) | the same bytes and text; **549 s** (4.81 tok/s), the same as run48 within the runs' noise (the shared experts 88.8 -> 88.0 s) |

## The upstream merge and a prompt on the GPU (`raw/s2_*`, 2026-10-31 to 11-01)

Upstream (JustVugg/colibri) had 1,409 commits since this fork's base (9e47a82a merges them).
Checked before and after on this machine: the tiny model's logits at steps of 1, 32, 300 and
2,048 and a 40-token decode the bytes of the build before (`k3tiny.sh`), its tiny, checkpoint,
dashboard and upstream's new park checks pass; the real model's 35-token prompt the same logits
(`s2_mrg_logits`), the same expert cache (97 slots a layer) and prompt time; MiMo's tiny model
the same logits at steps of 1 and 40 (`mimotiny.sh`; the real MiMo shards are no longer on the
drives). Every C test both trees build passed in both (191; 47 of upstream's 50 new ones, the
other 3 take arguments from test-python: `run_ctests.py`); test-python 2,520 -> 3,200 tests, the
only new failures 3 Qwen3.8 converter tests whose temporary files Windows still holds open
(their older siblings fail the same way run alone). Fixed on the way: AVX-512 builds on Windows
(a `grp2` that windows.h defines), upstream's new `tmpfile()` calls, inkling's strings, K3's
Vulkan build, omp_spin.o only without Vulkan.

| run | what | the 2,637-token prompt |
|---|---|---:|
| `s2_gpu_logits` | the routed experts on the GPU (coli_cuda_k3.dll, the CPU's bytes), 35 tokens | the logits the same bytes |
| `s2_gpu_long2048` | the same, steps of 2,048 | 548.7 -> 407.6 s (130,528 experts on the GPU, 17,068 on the CPU) |
| `s2_gpu2_long2048` | + the dense matrices the GPU does not hold (`k3cu_matmul`) | 352.3 s (shared 115 -> 20 s, latent 35 -> 6) |
| `s2_gpu2_long4096` | one step of 4,096 | 268.5 s (1,390 GB read instead of 2,455) |
| `s2_gpu3_long4096` | + the gated MLA's attention core (`k3cu_mla`) | 213.7 s (MLA 67 -> 11 s) |
| `s2_gpu4_long4096` | + 64 GiB of the cache's slots pinned | 212.2 s |
| `s2_gpu5_long4096` | + the routers (`k3cu_router`), KDA's gates and conv through exp_f8 | 184.7 s (conv 11.1 -> 1.9, heads 17.6 -> 4.6) |
| `s2_gpu6_long4096`, `_short` | + the shared experts fused (`k3cu_shared`), the routing over every worker | 170.0 s; decode 0.79 -> 0.84 tok/s, the same text |
| `s2_ra_long4096` | + a 10 GB read-ahead pool | 165.0 s |
| `s2_ra2_long4096` | the pool refilled as each expert is done: **a bug** (a refilled slot given back at the step's end, its read still on its way: non-finite router logits at layer 6, another first token) | discarded |
| `s2_long_ref`, `s2_long_ra3` | fixed; the logits of all 2,637 positions (1.73 GB) with the pool and without it | the same bytes |
| `s2_ra20_long`, `s2_ra20b_long` | a 20 GB pool (the default) | **130.3, 136.8 s**; waiting for reads 54 -> 0.9 s |
| `s2_long_ra20` | the 20 GB pool's logits at all 2,637 positions | the same bytes |
| `s2_ra20_short`, `s2_ra0_short` | decode with the pool's RAM taken from the cache (82 slots a layer) and without (94) | 0.81 and 0.83 tok/s, the same text |
| `s2_d32_long`, `bench_shared.c` | the exact dense kernel over 32 rows a block | 341 against 328 ms a 4,096-row shared expert step: dropped |
| `s2_a2b_long`, `s2_ab_*` (`series_b2b.sh`) | the experts' uploads back to back (no wait a copy), A/B twice | the same bytes; 141.8, 145.8 s against 145.1, 143.6 s (the shared experts lose what the routed experts gain on PCIe): dropped |

The GPU's timing on this machine varies by up to a quarter for minutes at a time
(`s2_d32_long`: coli_cuda.dll's unchanged dense calls 35.3 -> 44.6 s), so a change is kept
on an alternating series or a bench of its own, never on one run.

## The card's own matrices 2x sooner (`raw/s3_*`, `ab3_*`, 2026-11-01)

The 369 matrices coli_cuda.dll holds (KDA's projections, the gated MLA's o_proj and gate,
the head) took 32.6 s of the 2,637-token prompt in `quant_matmul_tiled`, 8 outputs by 8 rows a
block. `quant_matmul_blocked` keeps quant_matmul's 256 sums an output and its tree, so the same
bytes, with 16 outputs by 8 rows a thread, the rows in blocks of 8 (two 16-byte loads a step),
the next step's inputs and weights in flight during this one, and a nibble's decode one mask
and one fused add (`(b & m) ^ c` reads as 2^23 + 16^p (n + 8); the old decode's shift, constant
and mask cost more than the product). Found on the way with `bench_qmt.cu` and `fp32peak.cu`
(the GPU's counters need an administrator's setting, so by probes): the 4090 issues about 35 T
fused adds a second, 19 T int4 multiply-adds with the product rounded first, 10.7 T with the
old decode at 8 rows a weight; with its inputs or weights served from L1 the int4 kernel kept
its time (bound by its instructions and its loads' latency at 2 warps a scheduler, not by L2's
3.6 TB/s), while int8 halved (its byte loads). Weights through shared memory, two named
register sets, two threads a chain: no faster.

| run | what | the 2,637-token prompt |
|---|---|---:|
| `bench_qmt.cu` | 12288x7168 int4 (KDA q/k/v/g), 7168x12288 int4 (KDA o), 7168x12288 and 12288x7168 int8 (MLA), 2,637 rows | 72 -> 24, 69 -> 23, 109 -> 27, 120 -> 30 ms a matmul; the same bytes (zeros of both signs, subnormal and large inputs) |
| `s3_qmb_logits`, `s3_qmb2_logits` | the logits of all 2,637 positions, the first build and the final one | the same bytes as `s2_long_ref` |
| `s3_qmb_time` | the first build: a third device buffer of 130 MB | 125.6 s, the MoE 10 s slower beside it: x now lands in the output's buffer |
| `ab3_qmb_*`, `ab3_tiled_*` (`series_qmb.sh`) | A/B twice, `COLI_CUDA_QMB=0` the old kernel | **113.0, 112.4 s** against 128.8, 128.2 s: attention 48.7 -> 32.5 s, the resident matmuls 32.6 -> 15.6 s |

The prompt now waits for C:. During the A/B's prompts C: read 91-92% of the time (8.2 GiB/s
when busy, against 83% with the old kernel) and D: 81-84% (5.6 GiB/s): every MoE layer reads
nearly all of its experts once, 1,446 GB a cold prompt, 61.4% of them from C:. A copy to the
card from pinned memory runs at 26.7 GB/s alone, 26.75 beside the drives' reads and 23.3 beside
37 GB/s of CPU sweeps too (`h2d_load.cu`), so the routed experts' copies were not what held it.
What is left of the 113 s: attention 32.5 s (KDA 25: its resident matmuls 15.6, its CPU work
9.6), the MoE 76.5 s (routed experts 44, shared 17.3, latent 5.6).

## Steps past 4,096 rows (`raw/big_*`, `decc_*`, `dec_*`)

A step of 4,096 rows or more reads nearly every expert of every MoE layer once, so a prompt
of n tokens read 1.4 TB n/4,096 times. `K3_CHUNK` now goes to 16,384. A first try (a
5,862-token prompt, `long6k_prompt.txt`) found that a whole 4,096-row step's device buffers
did not fit beside the 20 GiB of resident matrices (kimi_k3 22.8 GiB dedicated, dwm 1.6):
Windows moved the card's memory to RAM without an error, the GPU at 100% and the drives
idle for minutes (the 2,637-token prompt had never filled a step). The GPU's calls now take
a step's rows 2,048 at a time (`K3_GPU_ROWS`: coli_cuda.dll's resident matmuls, the MLA
core, the routers; the dense matrices and shared experts were 1,024 already): every
kernel's row is its own, so the pieces give the step's bytes (`k3tiny_big.sh`: 6,000 ids at
steps of 2,048, 4,096 and 8,192 the same logits, every GPU path forced).

| run | what | result |
|---|---|---:|
| `big_4096_logits`, `big_8192_logits` | 5,862 tokens, the logits of every position | the same bytes; 2,891 -> 1,446 GB read |
| `big_4096_time`, `big_8192_time` | the same, timed | **253.5 -> 199.6 s** |
| `decc_4096_*`, `decc_8192_*` (`series_decchunk.sh`) | decode with the activations' RAM of each (81 -> 78 slots a layer) | 0.81 -> 0.80 tok/s (55.7 -> 56.0 s for 45 tokens, twice), the same text |
| `dec_cpu_*`, `dec_gpu_*` (`series_dec.sh`) | decode's routed experts on the GPU too (`K3_GPU_EXPERT_ROWS=1`, not kept) | 0.70 against 0.81-0.88 tok/s, the same text: the copies take the RAM the CPU's experts read (shared 157 -> 210 ms, waiting for reads 328 -> 430) |

The launcher's default step is now 8,192.

## The exact dense kernel's lanes in one thread (`raw/kb*`, `decr*`, 2026-11-01)

coli_cuda_k3.dll's dense kernel for the matrices the card does not hold and the shared
experts (`k3_dense`) gave an output's 8 lanes to 8 threads, added them with shuffles every
group of 64 columns and read its inputs one float a load: 3.1-3.6 TMAC/s. `k3_dense_b` keeps
the CPU's arithmetic (`matmul_i4_grouped`: 8 lanes of 8 fused adds a group, hsum256, fmaf a
group) with a thread's 8 lanes in its own registers: 4 rows by 6 outputs a thread, the lanes
run 0, 4, 2, 6, 1, 5, 3, 7 so each finished one joins hsum256's tree at once (three kept an
output), nibble l of the group's word m decoded with one mask and one fused add, x staged by
column 128 columns a stage (`bench_k3d.cu`).

| run | what | result |
|---|---|---:|
| `bench_k3d.cu` | shared gate/up, shared down, latent down, latent up, a KDA projection; 1,024 rows | 14.3 -> 3.5, 12.5 -> 3.2, 7.3 -> 1.8, 7.3 -> 1.9, 25.0 -> 6.1 ms (13-15 TMAC/s); the same bytes; from 48 rows faster, at 32 even |
| `tests/test_k3_cuda_experts.c` | k3cu_matmul and k3cu_shared against matmul_i4_grouped and k3_shared on the CPU, 300 and 1,030 rows, K3's shapes | 0 of 9.6 M outputs differ |
| `kb_2637_logits`, `kb_5862_logits` | the logits of every position: 2,637 tokens against `long_ref`, 5,862 in a step of 8,192 against two of 4,096 with k3_dense | the same bytes |
| `kb5862_kb_*`, `kb5862_old_*` (`series_kb.sh`) | A/B twice, `K3_GPU_DENSE_B=0` the old kernel, 5,862 tokens | **173.8, 172.7 s** against 195.4, 198.6 s: shared 35 -> 20 s, latent 12 -> 6 |
| `kb2637_kb_*`, `kb2637_old_*` | the same at 2,637 tokens | 118.9, 118.3 against 170.7, 170.7 s: the card's memory short (below) |
| `kb2637_res3`, `kb2637_res3b`, `kb5862_res3` | 3 GB left free beside the resident matrices (359 of them instead of 369) | 106.4, 106.5 s; 172.4 s; the dense calls 20 -> 5 s |
| `decr2_*`, `decr3_*` (`series_res.sh`) | decode with 2 GB and 3 GB left free | 55.8, 55.9 against 56.2, 56.0 s for 45 tokens, the same text |
| `kb4_2637_logits`, `kb4_2637_*`, `kb4_5862` (`series_kb4.sh`) | the GPU's calls 1,024 rows at a time, the MLA core's scores 128 MB, 2 GB left free (all 369 resident) | the same bytes as `long_ref`; **106.5, 106.5 s**; **173.1 s** |

The 2,637-token A/B ran after the desktop compositor had grown from 1.6 to 1.7 GB of the
card: coli_cuda_k3.dll's dense calls took 20 s (k3_dense_b) and 67 s (k3_dense) where they
had taken 5 and 11, its buffers in RAM. Leaving 3 GB free fixed it but took 10 matrices off
the card, and their arithmetic with them (the CPU's instead of quant_matmul's: not the
reference's bytes); the GPU's calls in pieces of 1,024 rows and the MLA core's scores within
128 MB free about 0.4 GB instead and keep every matrix where it was.

Where the 106.5 s go now: attention 30.6 s (KDA 23.8: its resident matmuls 15.5, its CPU
work 9.6; MLA 6.9), the MoE 72.1 s (routed experts 41.5, shared 10.5, latent 3.1, waiting for
reads 8.8). C: is the floor: its share of a cold prompt's 1.4 TB takes about 100-108 s.

## The KDA layer on the card (`raw/kda_*`, 2026-11-02)

A prompt step's 69 KDA layers ran their five projections on the card (coli_cuda.dll: each
call uploading the layer's input and downloading its 12,288 outputs a row) and the rest on
the CPU: the f32 projections (fa, fb, bp), the short convolutions and the heads (norms,
decay, recurrence, gated RMSNorm), 9.6 s of the 2,637-token prompt and 20 s of the
5,862-token one. `k3cu_kda` (coli_cuda_k3.dll) now takes the whole layer in pieces of 512
rows: the input goes up once, the four projections the card holds run there on
coli_cuda.dll's own copies (`coli_cuda_tensor_dev_ptrs`) in the kernel w_matmul got for them
(`quant_matmul_blocked`, now `qmb_kernel.cuh`, compiled into both DLLs), then the conv, the
f32 projections (matmul_f32.h's 32 lanes, the router's kernel with f32 weights), the norms
(the squares rounded and added in order: GCC's in-order reduction of k3_kda_head's loop, read
in its .s), the decay through k3_exact.h's exp_f, the recurrence (a thread a column of the
head's state, in registers; the two sweeps' fused adds in k3_kda_head's order), the per-head
RMSNorm (its sum in double, in order) and the gate, and o_proj; the output comes back, and
the state and conv windows at the end of the layer. A projection the card does not hold is
computed as before and handed in. The KDA pieces and the MLA core share one device buffer
(they never run at once), so the card holds no more than before.

| run | what | result |
|---|---|---:|
| `tests/test_k3_cuda_experts.c --kda` | kda_forward through k3cu_kda against its by-step path on the CPU: the tiny model's shapes, K3's with every projection on the card (1,100 rows: 3 pieces), K3's with k and o on the card | 0 of 12.2 M outputs, 0 of the state and windows differ |
| `k3tiny.sh`, `k3tiny_big.sh` | the tiny model at steps of 1 to 8,192 (`K3_GPU_MIN_ROWS=1`) | the same logits as the build before; 40-token decode the same |
| `kda_2637_logits`, `kda_5862_logits` (`series_kda.sh`) | the logits of every position: 2,637 tokens against `long_ref`, 5,862 against `big6k_4096` | the same bytes |
| `kda_2637_new*`, `kda_2637_old*` | A/B twice, the build before in `pre_kda/` | **105.5, 106.0 s** against 106.6, 106.6 s: KDA 24 -> 11.8 s, waiting for reads 7 -> 20-24 s |
| `kda_5862_new`, `kda_5862_old` | the same at 5,862 tokens | **149.2 s** against 173.7 s: KDA 50.4 -> 25.2 s |

The 2,637-token prompt waits for C: as before (its 1.4 TB of reads take about 100-108 s):
the 12 s the layers saved went into waiting for reads. At 5,862 tokens the layers take 23.3
s on the card, 0.34 s a layer, nearly all of it the five quant_matmul_blocked projections
(53 ms each at 5,862 rows); the MoE (97.7 s: routed experts 51.4 with 65.6 s of copying
their bytes, shared experts 20.5) is now the prompt's larger part. The card's memory: peaks
of 23.5-23.8 GB in both builds, the desktop's share varying by 0.3 GB.

## The MoE step's copies and tail (`raw/moe_*`, `dn_*`, `pin_trace`, 2026-11-02)

With KDA on the card the MoE was two thirds of the 5,862-token prompt (97.7 of 149.2 s), and
the card copied its routed experts at 19.5 GB/s (65.6 s for 1.28 TB) where `pin_probe.cu`
copies the same 17.5 MB from the same kind of memory at 26.2 GB/s one at a time and waited
for, 26.8 back to back, 20.9 from ordinary memory. The experts came from pinned slots
(`K3_PIN_TRACE`, a temporary trace: 20 chunks of 1 GiB in the first 3.6 s, the read-ahead
pool's 1,139 slots, nothing refused; the cache's own slots are only used at decode). What
slowed them was `k3_down`: it writes the experts' outputs straight into pinned host memory
over PCIe, a warp 8 rows of 4 floats, 16-byte pieces (111 GB of them in the prompt).
`zc_probe.cu`: beside such writes at 1.9 GB/s the copies ran at 20.0 GB/s, beside the same
rate in whole 128-byte rows at 26.5. `k3_down` now gathers its block's 8 rows by 32 outputs
in shared memory and a warp writes one row's 128 bytes: the same values.

The main thread then held three serial passes over a step's rows: the experts' inputs
quantized (`mxfp4_qx`, 4.8 s), their outputs' weighted sum (7.7 s: 1.2 GB a layer read on one
thread, every token's row read and written once per expert) and the latent norm with the
shared experts' add. The quantization goes 64 rows a task over every worker; the sum a token
a worker, the token's pairs in the union's order -- j, then q: the union's heads are permuted
with the ids after pfirst is assigned, so listing a token's pairs by their index e changed the
tiny model's logits at every step; by j they are its bytes again.

| run | what | result |
|---|---|---:|
| `pin_probe.cu` | cudaHostAlloc 1 GiB chunks; 17.5 MB copies | 64 GiB pinned in 5.6 s; 26.2 GB/s one at a time, 26.8 back to back, 20.9 from ordinary memory |
| `zc_probe.cu` | the same copies beside a kernel writing to mapped pinned memory | none 25.8; k3_down's 16-byte pieces at 1.9 / 3.3 GB/s: 20.0 / 17.6; 128-byte rows at 1.9 / 3.7: 26.5 / 26.5 GB/s |
| `dn_5862_a` | k3_down's rows whole | copies 65.6 -> 56.7 s (22.9 GB/s); 149.2 -> 140.8 s |
| `dn_5862_t` | + timers | the weighted sums 7.7 s, the quantization 4.8 s |
| `dn_5862_b` | + quantization over every worker, the sum by column slices | 132.2 s: 0.8 s and 4.8 s (each token's row still read and written once per expert) |
| `dn_5862_c` | + the sum by token | 130.6 s: the sums 2.8 s |
| `moe_2637_logits`, `moe_5862_logits` (`series_moe.sh`) | the logits of every position, against `long_ref` and `big6k_4096` | the same bytes |
| `moe_5862_new*`, `moe_5862_old*` | A/B twice against fdcce1bc (`pre_moe/`) | **129.2, 130.4 s** against 147.4, 147.4 s: copies 65.2 -> 56.5 s, MoE 97 -> 78 s |
| `moe_2637_new*`, `moe_2637_old*` | the same at 2,637 tokens | 106.1, 106.4 s against 105.9, 106.1 s: waiting for reads 25 -> 35 s |

The 2,637-token prompt waits for C: as before. At 5,862 tokens a layer now takes about 1.35 s
of compute against about 1.05 s of the drives' reads (C: busy 73% of the prompt): the routed
experts' copies are 90% of the MoE's expert phase at 23 GB/s (the drives' DMA and the CPU share
the RAM they are read from), and the rest is attention (KDA 25 s: its five quant_matmul_blocked
projections; the MLA core 11.5 s at about 4 TMAC/s, where the card's FP32 allows several times
that).

## The MLA core in tiles, exp_f in float pairs (`raw/mla_*`, `mlb_*`, 2026-11-02)

The gated MLA's attention core (`k3cu_mla`) took 11.3 s of the 5,862-token prompt, 0.43 s a
layer. `mla_bench.cu` timed its five kernels at that prompt's calls (1,024 rows a call, the
scores 128 MB a chunk): the absorbed query 29.3 ms a layer, the scores 128.2 ms (8.7 TMAC/s,
64-by-64 tiles of 4 by 4 a thread), the softmax 62.9, the values 93.0 (10.6 TMAC/s) and
kv_b's rows 114.3 -- a block a row, a thread one of kv_b's rows, read a byte at a time. The
softmax was the card's doubles: k3_exact.h's exp_f costs about 20 double FMAs a value, and
the 4090 runs doubles at 1/64 of its FP32 rate (`expf_fast_check.cu`: 27.6 G values/s).

- **exp_f in float pairs** (`k3d_expf`): e^x = 2^m 2^(j/64) e^r with r = x - k ln2/64 kept as
  a float pair (ln2/64 in three parts; k times the first is exact, and so is x minus it),
  2^(j/64) a pair from a table, e^r - 1 to r^5/120: e^x within about 2^-46 as f + d. Where
  every number within 2^-42 of f + d rounds to f, f is exp_f's value; elsewhere -- 2,681 of
  the 2.24 billion floats in [-87, 88] (`expf_fallback_count.cu`), and the floats past them
  -- the double path decides (`k3d_expf_d`). 458 G values/s, exp_f's bits on every one of
  the 2^32 floats (`tests/test_k3_cuda_experts.c --all-floats`, the double path alone too).
  Every kernel of the DLL takes it: the softmax and the gates, KDA's conv, decay and beta,
  SiTU's sigmoid.
- **Four products in 128-by-128 tiles**, 8 by 8 outputs a thread, 8 k a step through two
  shared buffers: the absorbed query and kv_b's rows (a GEMM a head over the chunk's rows,
  kv_b's int8 widened as it loads), the scores (products rounded, added in order) and the
  values (fused). A block of scores whose positions all lie past its rows' nt does not run.
- **The softmax between them**: its max from the score blocks' own (the largest of the row's,
  order-free; s[0] if that is NaN, as softmax_), the exps and their sum in t's order in a
  block of 32 rows -- a thread the exps of one position in 16 rows, read and written whole,
  into shared memory, then warp 0 adds the step's 128 a row while the others load the next --
  and the division as the values load them.
- **The step whole, in chunks**: one call a layer (the caller handed it 1,024 rows at a
  time), chunks of up to 256 rows whose scores fit in 128 MB, each with the positions up to
  its own last row's; a chunk's q and gate rows go up on a copy stream into one of two
  buffers while the chunk before computes, its outputs down while the next one does. The
  card no longer holds a call's q, gate and outputs whole (175 MB at 1,024 rows).

| run | what | result |
|---|---|---:|
| `mla_bench.cu`, `mla_bench2.cu` | one layer of the 5,862-token prompt's calls, random data, kernels only | 427.6 -> 145-160 ms: the query 29.3 -> 5.2, scores 128.2 -> 69-83 (13.8 TMAC/s at best), softmax 62.9 -> 14.4, values 93.0 -> 50 (17 TMAC/s), kv_b's rows 114.3 -> 9 |
| `mla_call_bench.cu` | the whole call with its copies from ordinary memory: 1,024-row calls against one | a layer at 5,862 tokens 165-180 -> 162-166 ms, at 2,637 41-60 -> 36-48 ms; the same outputs |
| `expf_fast_check.cu`, `expf_fallback_count.cu` | k3d_expf against k3d_expf_d | every float the same; 458 against 27.6 G values/s; 2,681 floats of [-87, 88] to the doubles |
| `tests/test_k3_cuda_experts.c` | k3cu_mla against mla_forward's loop: K3's shapes at 40, 300 and 200 rows (two chunks, chunks cut by their scores), the tiny model's at 37 and 600 rows (three chunks), 5 heads of 24+8 (partial tiles every way) | 0 differ; exp_f on all 2^32 floats the same |
| `k3tiny.sh`, `k3tiny_big2.sh` | the tiny model at steps of 1 to 8,192 | the same logits as 04623a27 at every position; 40-token decode the same |
| `mla_*_logits` (`series_mla.sh`) | the kernels alone (1,024-row calls): the logits of every position, against `long_ref` and `big6k_4096` | the same bytes |
| `mla_5862_*`, `mla_2637_*` | A/B twice against 04623a27 (`pre_mla/`) | **126.5, 124.7 s** against 131.8, 131.3 s (the core 11.6 -> 5.4 s); 2,637 tokens 106.3, 106.3 against 107.0, 106.3 s |
| `mlb_*_logits` (`series_mlb.sh`) | + one call a step, the copies under the kernels, exp_f in pairs everywhere: the logits of every position | the same bytes |
| `mlb_5862_*`, `mlb_2637_*` | A/B against 04623a27, twice at 5,862 tokens, once at 2,637 | **122.9, 122.5 s** against 131.6, 131.7 s: the core 11.6 -> 4.0 s, KDA's layers 23.6 -> 22.5 s, attention 44.3 -> 35.7 s; 2,637 tokens 106.2 against 106.3 s (waiting for reads 34.8 -> 38.8 s) |

Tried and dropped: a lane a row for the softmax's chain (125 ms a layer: 177 warps for the
whole card), 16 k a barrier (no faster), the chunk's rows a multiple of the values' wave of
blocks (47-49 against 51 ms, within the noise). The card ran at 2.865 GHz: 23.5 TMAC/s
unfused and 47 fused at most, so the scores reach about 59% of their peak and the values
about 36%; neither the values' data (their A or B from a 16 KB corner) nor the divisions in
their loads (multiplies instead) moved them.

The 2,637-token prompt waits for C: as before: the 3 s attention saved went into waiting for
reads. At 5,862 tokens attention is now 35.7 s (KDA 24.5, the MLA layers 11.2 with their core
4.0) and the MoE 79 s; a layer's compute is still above the drives' 1.05 s of reads.

## The MLA layers whole on the card (`raw/mll_*`, 2026-11-03)

With the core at 4.0 s the gated MLA layers still took 11.2 s of the 5,862-token prompt:
around the core every matrix went call by call -- qa, qb and kva through `k3cu_matmul` (the
host's weights up every call, `k3_dense`), the gate and o_proj on coli_cuda.dll's resident
copies -- each call's input up and output back, about 2.4 GB a layer, and the norms and the
cache rows on the CPU. `k3cu_mla_layer` takes the layer in the core's chunks: x up, qa, its
RMSNorm (`k3_rms_rows`: the squares summed in double in order on one thread, rmsnorm_'s
arithmetic), qb, kva into the cache rows (the latent's RMSNorm, the rope part as it is), the
gate, the core, o_proj, then the output and the cache rows down; the held matrices on
coli_cuda.dll's copies in quant_matmul_blocked, the others from their host weights, uploaded
once a step. A layer's traffic is now x and its output once, about 0.35 GB at 5,862 rows.

| run | what | result |
|---|---|---:|
| `tests/test_k3_cuda_experts.c --mla` | mla_forward through k3cu_mla_layer against its call-by-call path: K3's shapes with g and o on the card (int8), every matrix the host's (g, o int4), qa, kva, g and o on the card at rows 3,000.. (chunks cut by their scores), qb on the card (refused: call by call), the tiny model's shapes in 3 chunks | 0 of the outputs and cache values differ |
| `k3tiny.sh`, `k3tiny_big2.sh` | the tiny model (every matrix held, none in quant_matmul_blocked's shapes: call by call) | the same logits as 04623a27 |
| `k3tiny_big3.sh`, `K3_CUDA_DENSE=0` | every matrix the host's: the layer path at steps of 2,048, 4,096 and 8,192 (12, 8 and 4 layer steps) | the same logits as 04623a27 at every position |
| `mll_*_logits` (`series_mll.sh`) | both long prompts' logits at every position | the same bytes |
| `mll_5862_*`, `mll_2637_*` | A/B against the same build with `K3_GPU_MLA=0` | 5,862 tokens **120.5, 121.4 s** against 122.4, 121.9 s: the MLA layers 11.1-11.2 -> 9.0-9.1 s, the host-weight matmuls 284 -> 213 calls; 2,637 tokens 106.3 against 106.3 s (MLA 4.5 -> 3.2 s, waiting for reads 39.3 -> 41.1 s) |

Of the layers' 9.0 s at 5,862 tokens the core is 4.0, the held gate and o_proj about 2.5
(quant_matmul_blocked, about 10 TMAC/s) and qa, qb and kva about 1.8 (k3_dense's int8 lanes,
about 3.3 TMAC/s).

## The card's matrices in the CPU's arithmetic (`raw/hl_*`, `h2_*`, 2026-11-03)

The 369 matrices `K3_CUDA_DENSE` puts on the card (the head, the dense layer's FFN, the gated
MLA's gate and o_proj, 317 of KDA's 345 projections: 20 GiB) ran in coli_cuda.dll's kernels,
the CPU's products summed in quant_matmul's order (256 chains an output, then a fixed tree), so
the GPU's logits were not the CPU build's. coli_cuda_k3.dll now computes them on coli_cuda.dll's
copies in the CPU's arithmetic: `k3_dense_b` (int4 groups of 64; the copy's nibbles, flipped by
its upload, flipped back as the words come in) and the new `k3_dense_b8` (int8 rows: all 8 of
matmul_q's lanes in a thread for 4 rows by 4 outputs, a weight a byte permute and an add) from
64 rows, `k3_dense` below, and for a decode step `k3_mv4` and `k3_mv8`. `w_matmul` reaches them
through `k3cu_matmul_dev`; the KDA and MLA layers take the copies in the same kernels (no blocked
copy of their inputs any more). Only the int8 rows and int4 groups those kernels take go up.

| run | what | result |
|---|---|---:|
| `tests/test_k3_cuda_experts.c --dev` | `k3cu_matmul_dev` against matmul_i4_grouped and matmul_q: 1, 2, 3, 4, 5, 37, 64, 300 and 1,030 rows at 7,168 columns, KDA's and MLA's o_proj at 1 and 260-300, qb, groups of 32, widths the fast kernels leave to `k3_dense`, 1,100 rows by 60,000 outputs (pieces cut by `K3_MMBYTES`) | 0 of 66.5M outputs differ |
| `--kda`, `--mla`, the whole test | the KDA and MLA layers with the card's matrices, on the card and by step or call by call, against the CPU with every matrix the host's | 0 differ |
| `k3tiny_held.sh` | the tiny model, 6,000 ids at steps of 2,048, 4,096 and 8,192: every matrix on the card (now also in the layer paths), none there, the CPU alone; 40 tokens of decode | the same bytes |
| `hl_*_logits`, `hl_*_cpu`, `h2_2637_logits` | Kimi K3, both long prompts at the launcher's step: the matrices on the card against the CPU alone (664.8 s and 1,666.9 s with no GPU path) | the same bytes at every position |
| `h2_short_held1`, `h2_short_cpu` | the 35-token chat prompt one token a step (every position through the decode kernels) against the CPU alone, then 48 tokens | the same bytes at every position; the same text |
| `hl_5862_*`, `h2_5862_*` | A/B against 9372715b | **111.1, 111.1, 115.0 s** against 118.9, 119.2, 123.0 s: the KDA layers 21.7-21.9 -> 15.2-15.3 s, the MLA layers 8.8 -> 8.1-8.2 s, the dense layer's FFN and the head 1.4 -> 0.5 s |
| `hl_2637_*` | the same at 2,637 tokens | 105.5 against 106.0 s (C:'s reads) |
| `h2_dec_*` | 48 tokens of decode | 0.87, 0.87 against 0.86, 0.87 tok/s; the card's matrices 1.7 against 1.9 s a run (about 37 against 41 ms a token); every run the CPU's text |

`bench_b8.cu` (the kernels alone, T multiply-adds a second, 256 to 1,024 rows): `k3_dense_b8`
10.9-11.2 at MLA's gate (quant_matmul_blocked 7.4-7.9), 12.2-13.9 at its o_proj (8.9), 12.6-14.6 at
qb; `k3_dense_b` on the card's copies 14.6-14.9 at KDA's q (9.0), 11.8-13.9 at its o_proj (9.5-10.2).
`bench_k3_mv.c` (a whole call with its copies): a decode row 0.05 ms at KDA's q (0.05-0.07 in
coli_cuda.dll's `quant_matvec_warp`), 0.10-0.11 at MLA's gate and o_proj, 1.2 at the head; a
token's held matrices 24 ms against 24-29. The first decode kernels gave an output's 8 lanes to 8
threads that loaded the same 16 bytes: 4 loads in flight, 3.4 s a run against 1.9 (`hl_dec_*`:
0.79 against 0.81 tok/s); now `k3_mv4` takes 64-column groups a thread, four outputs a task (its
x loads were L1's limit with one), the sums and scales in shared memory, then each output's chain
in order, and `k3_mv8` brings 128 bytes of each of a warp's 4 rows into shared memory a thread 16
distinct bytes at a time.

The new logits against the old references (`logit_diff.py`): no position the same bytes; the top
token differs at 149 of 2,637 and 320 of 5,862 positions, KL mean 0.014, max 0.71 and 1.16 --
rounding differences of about 1e-7 that routing flips and the recurrent state carry along a
long prompt. The prompts' own next tokens (`ppl.py`, teacher-forced): perplexity 13.094 against
13.014 at 2,637 tokens and 10.557 against 10.534 at 5,862, the old order 0.006 and 0.002 nats
lower, 1.6 and 0.9 standard errors of the positions' differences, which split 1,327 to 1,309 and
2,968 to 2,893: no difference the measurement resolves. (For an int8 row of 7,168 columns
quant_matmul's 256 chains of 28 and its tree round less than matmul_q's 8 chains of 896:
typically about 5e-7 against 2e-6 of an output; the int8 weights themselves are about 1e-3 off.)

One run did not give the CPU's bytes: `hl_2637_none` (every matrix the host's, so an expert cache
of 65 slots a layer instead of 78) left them from position 1,226 on (top token at 76 positions,
KL max 0.50). The same configuration again (`hl_2637_none2`) and 9372715b in it (`hl_2637_none_old`)
gave the CPU's bytes. That run had put 10,189 experts on the CPU (7,858 and 8,132 in the other
two) and taken 6 more from the read-ahead pool: a timing-dependent fault in the MoE step's reads,
slots or waves, outside the code changed here; not reproduced, open (the next section).

## The rare fault, hunted (`raw/ck_*`, `mx_*`, `hc_*`, `pd_*`, `pn_*`, 2026-11-03)

`hl_2637_none`'s divergence did not come back in 32 runs of its configuration. Hunting it found a
different fault in the same reads -- a prompt that hangs when a drive falls behind -- now fixed.

- **A debug mode**, `K3_CHECK` (`docs/ENVIRONMENT.md`): a sampled hash of each expert's bytes (a
  word of every 4 KiB page) as its read lands, checked again when the card or a CPU wave takes the
  expert and when it is done with it; a step's output rows filled with a canary first, checked
  before the sums; the waves' x87 and SSE control words. `K3_CHECK_DUMP`: per layer every row's
  hashes at seven points (attention, the MoE's input, latent, routed sum, shared experts, output,
  the layer's output) and every pair's output with its expert, row and who computed it.
  `ckdiff.py` gives the first layer, record and rows where two runs' dumps part, and for the pairs
  whether the CPU or the card computed them in each.
- **The configuration again** (2,637 tokens at the launcher's 8,192 step, every matrix the host's;
  `series_ck.sh`, `series_mix.sh`, `series_pair.sh`): every run the CPU's bytes, the dumps the same
  record for record, no finding; with the same numbers a 3 GB read-ahead pool (most experts misses,
  the cache's 75 slots reused within every step), CPU waves of one expert, one reader a drive, and
  the default profile's run just before, as in the parent's series (its logits file still being
  written). 10,189 experts on the CPU was no outlier: 5,632 to 12,852 here, every run exact.
- **What the bad run shows**: its logits part at 1,226 already by 0.12 rms (KL 0.003), most at
  1,227-1,235 (0.32, KL up to 0.32), then about 0.2 to the end: one perturbation, carried along by
  the recurrent state. Its 6 experts more from the pool came after it (the routing past 1,226).
  1,226 is no piece's edge (the MLA core's chunks start at 0, 256, .., 1,024, 1,280, 1,504; KDA's
  pieces are 512 rows, the dense matrices' 1,024). It and its exact rerun ran one build (05:34);
  the edits after them touched only decode kernels, which no prompt call reaches. Its one
  outlier: waiting for reads 37.0 s, against 27.8-31.0 in every other run of the configuration.
- **The machine**: the card's memory runs at 12,101 MHz (its stock maximum 10,501), the core at
  2,925 MHz; the RAM is 4 x 48 GB DDR5 (JEDEC 4,800) at 5,400 MT/s, two DIMMs a channel; no ECC.
  `vramtest.cu` (18 GiB for 300 s: 2,342 passes of write, read-and-invert, read, 135 TB read, and
  pinned copies both ways), `ramtest.c` (150 GiB for 600 s, 49 passes) and `k3stress.c` (the real
  `k3_wave` and `k3cu_moe_*` over K3's shapes, repeated on the CPU and the card at once for 300 s:
  1,542 and 14,194 repeats against the first result, bit for bit): no error.
- **Read for races, none found**: the claims, `avail` and the slots' publication between the
  feeder and the waves, `busy` and the pool's drop and refill, the reader queues,
  `k3cu_moe_expert`'s copy before its kernels and its device slots, `k3_down`'s rows,
  `k3_dense_b`/`k3_dense_b8`'s staging, the KDA recurrence and conv, the MLA core's and layer's two
  streams and two buffers, `mg_run`, `k3_mla_soft`, `omp_spin.c`'s fork, join and dynamic loops,
  `exp_f`'s x87 stack. The canaries also show the card's writes into pinned memory reaching the
  CPU's caches.

**The hang.** A run with its reads slowed (`K3_IO_DELAY_US=1500`, about 15 ms on Windows' timer)
stopped at layer 65 with nothing on stderr, and so did d5e9925c's build (`raw/mx_5_iod.bt`,
`raw/hc_prod.bt`): the feeder thread inside `exit`, in `k3_io_stop` waiting for `g_io.mx` --
which it held itself, having called `exit` from `k3_io_submit`'s "reader queue overflow" -- and
the readers and the main thread waiting on it. C:'s prefetch queue held 4,096 entries (tail less
head) while 508 of its demands waited; D:'s held 6. A read ahead that a step asks for moves to the
demands (`k3_io_promote`) and leaves a hole where it was; a reader passes a hole only when it
gets there, and it does not get there while its drive has demands waiting. The holes counted
against `K3_IOQ` though the queue never held more than the pool's 1,139 reads: a drive that falls
behind for a few dozen layers fills it with them, and the next read ahead exits -- into a
deadlock, since `exit` runs `k3_io_stop` (atexit) before it flushes stderr. Now a full queue is
compacted first (its reads moved up over the holes, their slots' positions with them), the exit
lets go of the mutex, a promotion stays where it is when the demands are full, and `k3_io_stop`
does not join the thread that calls it. `hc_fix`: the same slowed reads end with the CPU's bytes
(448 s, waiting 345 s: C: at about a quarter of its rate).

| run | what | result |
|---|---|---:|
| `ck_1`..`ck_12` | the configuration, `K3_CHECK` and its dump | the CPU's bytes, 153.2-155.1 s; no finding (`ck_1`, `ck_2`: the FPU check counted MXCSR's sticky flags, masked since) |
| `ck_13`..`ck_22` | the same with the fix (the committed build) | the CPU's bytes, 153.0-154.6 s; no finding |
| `mx_1_ra3` | + `K3_READAHEAD_GB=3` | the CPU's bytes; 171.0 s, waiting 45.0 s, 9,198 experts on the CPU |
| `mx_2_as`, `mx_4_as` | as `ck_*` (`mx_4` beside `k3stress`) | the CPU's bytes (`mx_4`: 188.3 s, 12,852 on the CPU) |
| `mx_3_wave1` | + `K3_GPU_CPU_WAVE=1` | the CPU's bytes; 162.0 s |
| `mx_5_iod`, `hc_prod` | + `K3_IO_DELAY_US=1500`: this build, d5e9925c's | hung at layer 65 (`raw/*.bt`) |
| `hc_fix` | the same, with the fix | the CPU's bytes; 448.0 s, waiting 344.6 s |
| `pd_1`..`pd_5` | the default profile (the card's matrices), with the fix | the CPU's bytes, 109.2-109.3 s |
| `pn_1`..`pn_5` | at once after each `pd`: the configuration with the fix (`pn_1`, `pn_3`, `pn_5`: `K3_IO_PER_DRIVE=1`) | the CPU's bytes; 153.3-153.4 s, one reader a drive 172.1-172.4 s (waiting 56 s); no finding |
| `vramtest`, `ramtest`, `k3stress` | the card's memory, the RAM, the expert kernels at their clocks | no error (`raw/vramtest_18g_300s.txt`, `ramtest_150g_600s.txt`, `k3stress_300s.txt`) |

Open: the divergence. It has come up once in 35 runs of this configuration (the parent's three
and these 32), as a perturbation at one row (1,226), and the bad run's only measurable difference
was slower reads. Runs that check logits with `K3_CHECK=1
K3_CHECK_DUMP=<file>` locate it if it returns: `ckdiff.py` against a good run's dump names the
layer, the rows and the experts (and who computed them), and the slot checks say whether an
expert's bytes changed between its read and its use.

## The tile kernels' weights decoded once a block (`raw/tl_*`, `kr_*`, 2026-11-03)

`k3_dense_b` (int4 groups of 64) and `k3_dense_b8` (int8 rows) -- every prompt step's dense
matrices: KDA's five projections, the shared experts, the latent projections, the gated MLA's gate,
o_proj, qa, qb and kva, the head -- ran 13-15 and 11-14 T multiply-adds a second: each thread decoded
its own weights from global memory for its 4 rows, two instructions a weight beside 4 fused adds.
Their successors decode a stage's 64 columns of weights once a block into shared memory as floats and
bring the block's rows of x there by column, the next stage's going up while this one's products run
(two buffers, a barrier a stage):

- `k3_dense_s4` (int4): a thread all 8 lanes of 4 rows by 8 outputs, lane by lane in hsum256's order
  with three partial sums kept, as before; a lane's step one 16-byte load of x (4 rows) and two of
  weights (8 outputs) for 32 fused adds; 64 rows by 128 outputs a block.
- `k3_dense_o8` (int8): 8 threads a tile of 8 rows by 16 outputs, thread l lane l over the whole row,
  hsum256's tree by xor shuffles 4, 2, 1 at the end; a step two loads of x and four of weights for
  128 fused adds; 32 rows by 128 outputs a block.

Row quad q of column c sits at q ^ (c & 7), and o8's output quads likewise, so a step's loads and a
staging store's spread over the banks. What bounds them, by probes (`mio_probe.cu`, `lds_probe.cu`;
the card's counters need an administrator's setting): a warp's shared-memory load takes 2 cycles of
the SM's pipe at any width, 4 bytes a thread or 16, when it reads at most 256 distinct bytes (4 cycles
at 512), and a shuffle takes 2 as well, against 4 warp instructions of FFMA a cycle. The int4 kernel's
3 loads a 32-FMA step keep that pipe 75% busy before any staging, and its tile is as large as 255
registers allow with five sets (the lane, three partial sums, the total); handing the lanes to more
threads pays the tree's exchange on the same pipe: a quad of threads with two lanes each, an 8-by-8
tile and the tree by shuffles (`bench_q.cu`) ran 15-17 T. `bench_s.cu`'s timing variants (wrong
outputs): the int4 kernel 21.5 T, 25 without the staging, 22.6 without the barrier; int8 24, 28 and
25; the int4 staging spread between the lanes spilled registers (8.5 T).

`bench_tiles.cu`, the kernels alone at Kimi K3's shapes (the best of 3 runs of 20 launches, the old
kernels copied beside them), T multiply-adds a second, old -> new; every output the old kernels' bytes:

| matrix | 64 rows | 230 | 256 | 512 | 1,024 |
|---|---:|---:|---:|---:|---:|
| KDA q/k/v/g 12288 x 7168 int4, the card's copy | 12.7 -> 15.1 | 12.4 -> 20.3 | 14.3 -> 17.1 | 14.3 -> 18.6 | 14.3 -> 21.1 |
| KDA o_proj 7168 x 12288 int4, the card's copy | 8.4 -> 10.6 | 10.4 -> 14.9 | 11.4 -> 17.2 | 13.6 -> 15.6 | 13.6 -> 20.9 |
| shared gate/up 12288 x 7168 int4 | 15.0 -> 18.6 | 13.5 -> 15.8 | 14.4 -> 16.8 | 14.6 -> 19.7 | 14.6 -> 21.4 |
| shared down 7168 x 12288 int4 | 8.7 -> 10.8 | 10.7 -> 15.3 | 11.8 -> 17.1 | 13.9 -> 16.0 | 14.1 -> 20.5 |
| latent down 3584 x 7168 int4 | 4.5 -> 5.5 | 8.0 -> 14.3 | 8.9 -> 15.8 | 11.9 -> 17.1 | 13.9 -> 16.4 |
| latent up 7168 x 3584 int4 | 8.7 -> 10.7 | 10.7 -> 15.7 | 11.9 -> 17.3 | 13.8 -> 15.6 | 13.9 -> 17.9 |
| MLA gate 12288 x 7168 int8 | 10.5 -> 16.1 | 10.0 -> 16.6 | 10.9 -> 18.6 | 10.9 -> 21.3 | 11.0 -> 22.1 |
| MLA o_proj 7168 x 12288 int8 | 11.9 -> 17.2 | 10.6 -> 15.1 | 11.7 -> 16.8 | 13.5 -> 19.7 | 13.5 -> 22.4 |
| MLA qb 18432 x 1536 int8 | 10.4 -> 14.8 | 13.2 -> 20.7 | 14.1 -> 23.4 | 14.8 -> 20.7 | 14.7 -> 18.8 |
| MLA qa 1536 x 7168 int8 | 3.0 -> 5.2 | 10.7 -> 18.5 | 11.9 -> 20.5 | 10.4 -> 16.8 | 15.0 -> 24.9 |
| MLA kva 576 x 7168 int8 | 1.1 -> 2.0 | 4.1 -> 7.0 | 4.5 -> 7.8 | 9.0 -> 15.6 | 7.9 -> 12.4 |

A grid fills the card's 128 SMs (one block each) in whole waves only at some row counts: KDA's o_proj
at its 512-row pieces is 448 blocks, three and a half waves.

| run | what | result |
|---|---|---:|
| `tests/test_k3_cuda_experts.c`, the whole test | every dense case on the host's matrices and the card's copies (new cases: one stage, 25 stages, outputs 128 + 4 and not a multiple of 4), the KDA and MLA layers, the shared experts | 0 differ |
| `k3tiny_held.sh`, `k3tiny.sh` | the tiny model: every matrix on the card, none there and the CPU alone at steps of 2,048 to 8,192, 40 tokens of decode; against HEAD's logits | the same bytes |
| `tl_2637_logits`, `tl_5862_logits` | both long prompts' logits at every position against the CPU alone (`long_cpu`, `big6k_cpu`) | the same bytes |
| `tl_5862_new*`, `tl_5862_old*` (`series_tiles.sh`) | A/B twice against 254338ab (`pre_tiles/`) | **110.6, 108.8 s** against 112.5, 111.2 s: the KDA layers 15.1 -> 13.7 s, the MLA layers 8.2 -> 6.7 s, the shared experts 18.5-18.7 -> 18.0 s, the dense matrices the GPU does not hold (the latent projections) 7.0-8.2 -> 6.6 s |
| `tl_dec_new1`, `tl_dec_old1` | 48 tokens of decode (its kernels unchanged) | 0.81 against 0.81 tok/s, both the CPU's text |
| `kr_5862_1024_*`, `kr_5862_512_*` (`series_kdarows.sh`) | KDA's pieces of 1,024 rows (`K3_GPU_KDA_ROWS=1024`) against the default 512, alternated | 109.3, 108.6 s against 109.0, 109.3 s (the KDA layers 13.8, 13.4 against 14.0, 13.7 s): no difference to keep; the default stays 512 |
| `kr_5862_logits` | the same with 1,024-row pieces, logits at every position | the CPU alone's bytes |
| `bench_shared.c` | `k3cu_shared` alone, 5,862 rows (one layer's call with its copies), this DLL against 254338ab's | 125.5-128.0 against 138.2-139.9 ms; in the prompt the call takes about 196 ms a layer with either |

The prompt gained less than the kernels. At 5,862 tokens the attention's 3 s went partly into
waiting for reads (0.8-0.9 -> 3.1-5.3 s): this prompt reads the 1,446 GB the 2,637-token one does,
and its drives are now near their floor too. The shared experts' call, 18 s of it, moves its weights
(148 MB a layer) and its rows up and down from pageable memory on its kernels' own stream, beside the
routed experts' copies: alone at a layer's 5,862 rows it is 10% sooner with the new kernels, in the
prompt it is not measurably so. The KDA layers' pieces go up and come back on their kernels' stream
as well.

## The drives' shares: a long prompt against decode (`raw/tl_*`, `h2_dec_*`, 2026-11-03)

Each MoE shard is cut 61.4% to C: and 38.6% to D: (`k3_download.py`'s weights 10.4 and 6.55, the
drives' rates read alone). The monitor's samples (2 s each) during the runs above give the rates
while the drives work beside the rest of the engine:

| phase | C: busy | C: while busy | D: busy | D: while busy | the share that would balance them |
|---|---:|---:|---:|---:|---:|
| a long prompt (`tl_2637_logits`, `tl_5862_*`, `kr_5862_512_1`) | 90% or more in 39 of 40-41 samples | 7.9-8.2 GB/s | the same in 28-30 | 5.3-5.6 GB/s | 59.7% on C: |
| decode, 45-48 tokens (`tl_dec_*`, `h2_dec_*`) | 74-79% | 8.0-8.4 GB/s | 74-80% | 4.9-5.6 GB/s | about 61% on C: |

A cut at 59.7% would take about 3 s off a long cold prompt (C: reads 865 instead of 888 GB) and
put 4.4% more of every decode token's reads on D:, which decode already keeps as busy as C:, while
decode waits for reads about 330 ms of its 1.24 s a token: at a few hundred tokens a reply that can
cost more than the prompt saves. The shares stay as they are; a third drive or a faster one is
what would lower the long prompt's floor.
