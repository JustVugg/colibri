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
