# Kimi K3 on one RTX 4090 (192 GB RAM, two NVMe drives)

The machine of this fork's MiMo work: a Ryzen 9 7950X3D (16 cores, AVX-512), 4 x 48 GB
DDR5 at 5400 MT/s, an RTX 4090 (24 GB), and two NVMe volumes: `C:` (one drive, 1.70 TB
free) and `D:` (three NVMe drives in one RAID 0 volume, 3.21 TB free).
[moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) is 1.561 TB in 96
shards: 2.8T parameters, 104B active, 93 layers (69 KDA + 24 gated MLA), 896 routed
experts of which 16 a token, native MXFP4 experts (quantization-aware trained) and BF16
everything else. The engine is [`c/kimi_k3.c`](kimi_k3.md).

## What a token costs here

| | |
|---|---:|
| one routed expert (w1, w3: 3072 x 3584 e2m1 + a scale per 32; w2 the transpose) | 17.55 MB |
| a decode token's experts, none cached: 16 x 92 MoE layers | 25.8 GB |
| the rest of the model in RAM at the engine's default bits (int4 KDA/latent/shared, int8 MLA and head) | about 35 GiB |
| 17.5 MB unbuffered reads, 3 readers: `C:` alone / `D:` alone / both at once | 10.36 / 6.56 / 10.33 + 6.57 GB/s |

Both drives at once read 16.9 GB/s, so a token whose experts all come from disk needs
at least 1.53 s of reads: about 0.65 tokens a second before any expert is found in RAM.
The RAM left beside the resident weights holds roughly 100-150 GB of experts, 7-10% of
the 1.45 TB, and K3's router was trained for flat expert use (Quantile Balancing), so
the cache's hit rate stays near that share -- it turned out higher: consecutive tokens
share many experts, and the cache holds 61.5% of those a decode token asks for (below).

## Other engines that run Kimi K3, and what was taken from them

Surveyed on 2026-10-30. None holds K3 in this machine's memory: every one is bound by
reading experts from disk here, and the ones with published single-machine numbers
read fewer bytes a token by re-encoding the experts below the checkpoint's 4.25 bits.

| Engine | What it does with K3 | Reported | Taken here |
|---|---|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp/pull/26185) (K3 merged 2026-08-15) with [Unsloth's GGUFs](https://unsloth.ai/docs/models/kimi-k3) | the whole graph, CPU/CUDA; experts in RAM or mmapped | quants from UD-IQ1_S (594 GB) to Q8 (1.56 TB, lossless); every one larger than 192 GB, so mmapped experts page from disk; no single-GPU decode numbers published | -- |
| [Pulsar](https://dev.classmethod.jp/en/articles/dgx-spark-pulsar-ssd-expert-streaming/) (Rust + CUDA) | GGUF experts streamed from NVMe, host and VRAM caches | 0.70 tok/s, UD-IQ1_S (about 1.8 bits a weight), DGX Spark; a larger cache was slower there | the cache sizing caution (the RAM the experts take is the drives' write target too) |
| [WASTE / warp](https://github.com/sqliteai/warp) (C) | experts re-encoded to 3-bit residual VQ (982 GB), router lookahead prefetch, one aligned read an expert | 0.45-0.62 tok/s, 64 GB M5 Pro; top-16/12/8/4 experts: 0.59/0.70/0.89/1.06 tok/s | **the next layer's router applied early to read its experts ahead** (also the MiMo engine's, 77% of its experts predicted) |
| [deltafin](https://github.com/argonautlabsai/deltafin) (Rust) | experts streamed from four SSD replicas, every expert read split across them; a draft model | 1.00 tok/s, M5 Max with four SSDs (0.55 with one); "the slowest of each layer's 16 reads sets the pace" | **every layer's reads on every drive at once** -- here by cutting each layer's shard over the drives instead of full replicas, so no extra space |
| [KTransformers](https://github.com/kvcache-ai/ktransformers) | no K3 support yet; its MXFP4 MoE operator (DeepSeek-V4-Flash) computes experts held in RAM | -- | -- (1.45 TB of experts do not fit in RAM) |
| vLLM, SGLang, TokenSpeed ([recipes](https://recipes.vllm.ai/moonshotai/Kimi-K3)) | the whole model in VRAM | 8 x H200 and up | -- |

Not taken: re-encoding the experts (WASTE's 3-bit VQ, the 1-2 bit GGUFs). K3's experts
are quantization-aware trained at MXFP4, and every published re-encode is lossy against
them; `K3_TOPP` already offers the one byte-cutting lever that keeps the experts' bits,
dropping the routing's low-weight tail, as an explicit quality choice.

From this fork's MiMo engine on the same machine: per-drive reader pools with their own
unbuffered handles, demands before prefetches, unused prefetches cancelled, the experts
computed as they arrive in the order the sum always had.

## Download and placement

```powershell
.\c\tools\download_kimi_k3_4090.ps1 -DryRun   # the small files, the plan, the space check
.\c\tools\download_kimi_k3_4090.ps1           # 1.56 TB; rerun to continue after an interruption
.\c\tools\download_kimi_k3_4090.ps1 -Verify   # every shard's SHA-256 against the published one
```

The profile writes `C:\AI\Kimi-K3` (the model folder: config, tokenizer, layer 0, the
embeddings and the head, and 61.4% of every MoE layer's experts, 1,002 GB) and
`D:\AI\Models\Kimi-K3` (the other 38.6%, 559 GB): the drives' weights are their
measured read speeds, 10.4 and 6.55. Each MoE layer's shard is cut between experts
([tools/k3_download.py](../c/tools/k3_download.py)): `C:` keeps the shard's own file,
its header rewritten in place to name only the tensors it still holds, and `D:` gets
`model-000NN-of-000096.p2.safetensors` with the rest. The engine reads them as one
model with `K3_DIRS=D:\AI\Models\Kimi-K3`, and a step's two drives read its experts
together. `-Join` puts the published shards back (for another engine), checked
against their SHA-256.

## Build and run

From `c`, with MinGW-w64 GCC, the CUDA toolkit and Visual Studio's x64 C++ tools (open the
"x64 Native Tools Command Prompt for VS 2022", or call its `vcvars64.bat` first):

```powershell
.\build_cuda.bat                                   # coli_cuda.dll (CUDA_ARCH=sm_89 for the 4090)
.\build_k3_cuda.bat                                # coli_cuda_k3.dll: a prompt step on the GPU
mingw32-make kimi_k3 CUDA_DLL=1 ARCH=native VK=0   # kimi_k3.exe with the DLLs' loaders
.\tools\start_kimi_k3_4090.ps1                      # the API on http://127.0.0.1:18080/v1, model kimi-k3
```

The profile ([tools/start_kimi_k3_4090.ps1](../c/tools/start_kimi_k3_4090.ps1)) reads the
model from both folders (`K3_DIRS`), two readers per drive with the next layer's
prefetch, puts up to 20 GiB of dense matrices on the GPU and releases their host copies
(`K3_CUDA_DENSE`, `K3_CUDA_RELEASE_HOST`), keeps a decode step's routed experts on the CPU
(`K3_CUDA=0`: uploading every expert a token uses would move 25.8 GB over PCIe) and puts a
prompt step's experts, its other dense matrices, its KDA layers, its MLA attention and its routers
on the GPU too, in the CPU's arithmetic to the bit (`K3_GPU_EXPERTS`, `coli_cuda_k3.dll`; a 20 GB
pool reads the next layer's experts ahead), gives the process 174 GB of RAM (the expert
cache takes what the rest leaves), a 131,072-token context (the MLA caches: 55 KB a token)
and 8,192-token prompt steps (each expert a step needs is read once for all of them; 1.4 MB
of RAM a token of the step past 512: decode 0.81 -> 0.80 tokens a second against steps of
4,096, a 5,862-token prompt 254 -> 200 s; `-Chunk` takes up to 16,384). `-CpuExperts` keeps a prompt on the CPU;
`-Cpu` keeps the dense matrices on the CPU;
`-DenseGB 21.5 -ReserveGB 1` puts 35 more on the GPU (0.78 -> 0.81 tokens a second
measured) and leaves about 0.6 GiB of VRAM to other programs: past what is free, Windows
moves GPU memory to RAM without an error and every call slows down.

**Agent turns.** The profile keeps four photos of the recurrent state (`COLI_K3_CKPT=4`,
about 434 MB each), taken at every prompt's and every reply's end. OpenAI-compatible
agents send the model's earlier replies back without their reasoning; the conversation
then differs from what the engine holds right after the previous prompt, and without a
photo the whole conversation would be computed again -- on K3 here, hours for a long
agent session. With them the next turn restores the previous prompt's photo and computes
only what follows. A prompt's rows are computed alike in any step (the GPU's kernels
compute each row alone), so the resumed turn is the bytes of the whole conversation
computed cold: tried through the API on the tiny model, the second turn restored the
first prompt's photo (150 of 288 tokens) and answered what a cold server answered.

## Measured on the real model

[docs/experiments/kimi-k3-4090-2026-10-31](experiments/kimi-k3-4090-2026-10-31/README.md):
each run a fresh start on a 35-token chat prompt, then 45 tokens greedy, with the profile's
settings; every run gave the same text.

| | the first engine (71c243f5) | 2026-10-31 (33351592) | now, the profile |
|---|---:|---:|---:|
| start: read, quantize, upload to the GPU | 161 s | 53 s | 44 s |
| the 35-token prompt | 57.0 s | 34.4 s | 32.9 s |
| decode | 0.56 tok/s | 0.79 tok/s | **0.80 tok/s** (0.81 with 4,096-token steps' RAM, 0.83 without the read-ahead pool) |
| read from the drives a decode token | 16.8 GB | 12.1 GB | 12.9 GB |
| a 2,637-token prompt (`raw/run19`, `run51`, `kda_2637_*`) | 1,401 s (f62cbd17, 512-token steps) | 549 s (the CPU, 2,048-token steps) | **106 s** (25 tok/s, one step on the GPU; C:'s reads its floor) |
| a 5,862-token prompt (`raw/big_4096_time`, `kda_5862_*`, `moe_5862_*`, `mlb_5862_*`, `mll_5862_*`, `hl_5862_*`, `tl_5862_*`) | -- | -- | **110 s** (53 tok/s, one 8,192-token step; 112 s with each thread of the tile kernels decoding its own weights, 119 s with the card's matrices in coli_cuda.dll's kernels, 122 s with the MLA layers call by call, 131-132 s with the MLA core in 64-by-64 tiles, 147-149 s before the MoE's tail went over every worker, 173 s with KDA's heads on the CPU, 254 s in two steps of 4,096) |

Every row of the last column gave the logits of the commit before (the 35-token prompt's,
and the 2,637-token prompt's at all 2,637 positions against a run with the read-ahead off)
and the same text, until the card's matrices went over to the CPU's arithmetic: since then
the logits are the CPU build's (both long prompts at every position against a run with no GPU
path at all).

Where a decode token's 1.24 s goes (the last column): attention 153 ms; the MoE 1,073 ms
-- the routing and issuing its reads 59, the latent projection 57, the shared experts 167,
the routed experts computed 497, waiting for reads 286 -- and the head 2. The routed
experts' compute reads their 25.8 GB from RAM at about 52 GB/s (16 threads read 66 GB/s at
most) while the drives write 12.1 GB into it: the cache (98 experts a layer, 11% of them)
holds 61.5% of what a token asks for, and the next layer's prediction names 57% of the
routed experts before their routing. The GPU's 404 dense matrices take 111 ms a token.

## What changed in the engine

| Change | Effect on the numbers |
|---|---|
| a pool of readers per drive (2 each), their own unbuffered handles on Windows, demands before prefetches; each expert computed as soon as its bytes are in, in the step's order | none: the logits of every configuration tested are the old loader's, byte for byte |
| the next MoE layer's experts predicted after a decode step's attention and read ahead, unused ones cancelled unread | none (a prediction only decides what is read early) |
| a step's experts summed in the order of their ids as text, the published shards' own order | none on the published layout; the cut layout gives the same bytes in any folder order |
| the largest dense matrices on the GPU (`K3_CUDA_DENSE`), host copies released | none since 2026-11-03: `coli_cuda_k3.dll` computes them in the CPU's arithmetic, so the GPU's logits are the CPU build's. Before, quant_matmul's sums in another order: on the tiny fixture's 70 positions the same argmax and max KL 4e-12, over 6,000 positions the top token at 2 of them differed (max KL 0.094) |
| the load's quantization over every worker; the RAM budget without the GPU's released host copies; OpenMP regions on omp_spin.c's spinning team | none |
| a step's experts in waves: every read issued at once on its drive, whatever arrived computed together, the sum in the old order (`K3_WAVES`) | none |
| the reads issued before the latent projection, the shared experts computed while none has arrived | none |
| the prediction from the next layer's own mix, its 12 best experts (`K3_PREFETCH_MIX`, `K3_PREFETCH_TOP`) | none (a prediction only decides what is read early) |
| the routers kept in BF16, widened in the f32 kernel's order | none: the f32 copy's bits from half the bytes |

The dense matrices on the 4090 against the CPU kernels (`tests/bench_k3_cuda_dense.c`,
random weights, the time of a whole call with its copies; "tiled" is
`coli_cuda_matmul_tiled`, which a step of two rows or more takes):

| matrix | GPU, 1 row | CPU, 1 row | GPU, 32 rows | tiled, 32 rows | CPU, 32 rows | GPU, 512 rows | tiled, 512 rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| KDA q/k/v/g, 12288 x 7168 int4-g64 | 0.06 ms (0.19) | 0.90 ms | 5.6 ms | 1.5 ms | 18.7 ms | 77.8 ms | 18.7 ms |
| MLA o, 7168 x 12288 int8 | 0.11 ms (0.17) | 0.98 ms | 4.8 ms | 1.7 ms | 12.5 ms | 74.8 ms | 24.4 ms |
| shared gate/up, 6144 x 7168 int4-g64 | 0.04 ms (0.09) | 0.72 ms | 2.5 ms | 0.7 ms | 9.2 ms | 38.8 ms | 9.5 ms |
| head, 163840 x 7168 int8 | 1.2 ms (2.2) | 18.4 ms | -- | -- | -- | -- | -- |

One row runs `quant_matvec_warp` (in brackets: `quant_matmul`, before it): a warp an
output, each lane taking 8 of quant_matmul's 256 threads -- their columns of a 256-column
chunk are one 32-bit word of int4, two float4 of the input -- with their own chains of
fused adds and quant_matmul's tree over them between lanes, so its bytes are
quant_matmul's (`COLI_CUDA_QMR=0` goes back to it). With 20 GiB of them on the card a
decode token's dense work there is about 25 ms in this bench (60 before) where the CPU
streamed the same bytes from RAM in about 360 ms. The plain kernel reads a
weight once per row; the tiled one decodes it once for 8 rows and serves 8 outputs from
each input, with quant_matmul's own sums and tree, so its bytes are the plain kernel's
(`tests/test_cuda_matmul_tiled.c`: 91 shapes and row counts, and one row of 18 shapes,
K3's among them, against quant_matmul's arithmetic done on the CPU; the tiny model's GPU
logits are the same bytes in prompt steps of 1, 7 and 32 tokens). On the real model the
faster kernel took attention from 178 to 136 ms a token and the text stayed the same, but
not the token's time: the drives' reads then took the 42 ms back (run14 in the experiment
record), the decode being as fast as the experts that are not in RAM arrive. With
`coli_cuda_k3.dll` a decode row runs `k3_mv4` and `k3_mv8` instead, in the CPU's arithmetic:
0.05 ms at KDA's projections, 0.10-0.11 at MLA's gate and o_proj, 1.2 at the head, a token's
held matrices about 24 ms (`bench_k3_mv.c`; 24-29 in the kernels above).

## Not done yet

Done since: the card's own matrices in `quant_matmul_blocked` (16 outputs by 8 rows a
thread, quant_matmul's bytes: 72 -> 24 ms a 2,637-row KDA projection), the exact dense
kernel's lanes in one thread (`k3_dense_b`, the CPU's bytes: 3.3 -> 13-15 TMAC/s), steps
up to 16,384 rows (the launcher's 8,192), the GPU's calls 1,024 rows at a time, the KDA
layers whole on the card (`k3cu_kda`: a 5,862-token prompt 173.7 -> 149.2 s, the same bytes)
the MoE step's copies and tail (`k3_down` writing whole 128-byte rows into the pinned
outputs, the inputs' quantization and the weighted sums over every worker: 147.4 -> 130 s)
the MLA core in 128-by-128 tiles with exp_f in float pairs, a step in one call with its
copies under the kernels (the core 11.6 -> 4.0 s: 131.7 -> 122.7 s), the MLA layers whole
on the card (`k3cu_mla_layer`: 11.2 -> 9.0 s, 122.2 -> 121.0 s), the card's matrices in the
CPU's arithmetic (`k3_dense_b` and the int8 `k3_dense_b8` on coli_cuda.dll's copies, `k3_mv4` and
`k3_mv8` for decode: the GPU's logits the CPU build's, 119.0 -> 111.1 s, decode unchanged), and
the tile kernels' weights decoded once a block into shared memory (`k3_dense_s4`, `k3_dense_o8`:
13-15 -> 16-21 and 11-14 -> 17-23 TMAC/s, the same bytes; at 5,862 tokens the KDA layers 15.1 ->
13.7 s and the MLA layers 8.2 -> 6.7 s, the prompt 111.9 -> 109.7 s on average, the rest gone
into waiting for reads).

- **The cold prompt waits for C:.** A step of 4,096 rows or more reads nearly every expert
  once, 1.4 TB, 61.4% from C: (8.2 GiB/s while busy): C: reads 92% of the 2,637-token
  prompt, so about 100-108 s is its floor. Today's speeds would put C:'s share at 59.4%
  (about 3 s).
- **The routed experts' copies** are 90% of the MoE's expert phase: 1.3 TB in 56.5 s at
  5,862 tokens, 23 GB/s against 26 alone (the drives' DMA and the CPU share the RAM they are
  read from). The shared experts hold the main thread meanwhile (18.0 s at 5,862 tokens): on
  a helper thread they would leave it to compute CPU waves, about 3,000 experts and 2 s of
  copies less. Their call itself moves its weights (148 MB a layer) and its rows up and down
  from pageable memory on its kernels' own stream: 126 ms alone at a layer's 5,862 rows (140
  with the old tile kernels), about 196 ms in the prompt beside the routed experts' copies.
- **The MLA core's products** (`k3cu_mla`, 4.0 s at 5,862 tokens, the square of the
  context) reach about 59% of the card's FP32 peak in the scores (two instructions a
  product: rounded, then added) and 36% in the values (fused); register-level tuning of the
  128-by-128 tiles might give a few seconds more at long contexts.
- **The tile kernels' rate.** `k3_dense_s4` and `k3_dense_o8` run 16-21 and 17-23 TMAC/s, about
  half of what the card's FP32 pipes issue: the int4 kernel's operands (3 shared-memory loads of
  2 of the pipe's cycles each a 32-FMA step) keep that pipe 75% busy, and its tile is as large as
  255 registers allow beside hsum256's partial sums. In the prompt they are no longer what holds
  it: at 5,862 tokens the attention's 3 s went partly into waiting for C:'s reads (0.8 -> 3.1-5.3
  s), and pieces of 1,024 rows did not shorten the KDA layers (`kr_*`), whose pieces also go up
  and down on the kernels' own stream.
- **A rare fault in the MoE step.** One run of 2,637 tokens with every matrix the host's
  (`raw/hl_2637_none`) left the CPU's bytes from position 1,226 on. 32 runs of that
  configuration since gave them, with `K3_CHECK` (each expert's bytes checked from its read to its
  use, the output rows canaried, every layer's rows and pairs hashed) and with timing variants, and
  tests of the card's memory, the RAM and the expert kernels at the machine's clocks (the card's
  memory +1,600 MHz, the RAM 4 x 48 GB at 5,400 MT/s) found no error: open; `K3_CHECK_DUMP` and
  `ckdiff.py` locate it if it returns. The hunt found and fixed a hang in the reads: a drive that
  fell behind filled its prefetch queue with the holes that promotions leave, and the overflow's
  exit deadlocked in `k3_io_stop` (the experiment record's "The rare fault, hunted").
- **The VRAM beside the resident matrices** is 2-2.5 GiB: a program that takes more of the
  card after the start (the desktop compositor held 1.6-1.7 GB) can still push the prompt's
  buffers into RAM; `-ReserveGB 3` keeps 10 matrices off the card (decode 0.81 -> 0.80 tok/s,
  the same bytes) and leaves 1 GB more.
- **Routed experts on the GPU at decode** were tried (`raw/dec_*`): 0.70 against 0.81 tok/s,
  the card's copies taking the RAM bandwidth the CPU's experts read.
