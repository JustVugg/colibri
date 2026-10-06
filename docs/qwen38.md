# Qwen3.8-Flash-Next on colibri

`c/qwen38.c` runs the language model in
[`Qwen/Qwen3.8-Flash-Next-FP8`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)
directly from the official safetensors shards. No conversion or second copy of
the weights is required. The engine supports text and images through the
checkpoint's vision encoder; see **Vision** below. The checkpoint's MTP layer
drafts tokens for speculative decoding, on by default (`Q38_MTP=0` turns it off); see
[Speculative decoding with the MTP head](#speculative-decoding-with-the-mtp-head).

The upstream language model has 125B ordinary parameters with 6B activated,
plus a 51B hashed n-gram embedding. It has 48 layers arranged as 12 repetitions
of three Gated DeltaNet layers and one Qwen Sparse Attention layer. Every layer
uses a four-branch gated residual and a 512-expert top-10 MoE plus one shared
expert.

## Download and run

Pin the checkpoint revision so a later upstream update cannot silently change
the local tensor contract:

```sh
hf download Qwen/Qwen3.8-Flash-Next-FP8 \
  --revision bcd9f01ddc9cff2316eb84281bebcd5b058bddce \
  --local-dir ~/Models/Qwen3.8-Flash-Next-FP8

make -C c qwen38
COLI_MODEL=~/Models/Qwen3.8-Flash-Next-FP8 ./c/coli chat
```

`coli serve` and `coli web` use the same gateway path. Qwen3.8 thinks
by default; `reasoning_effort` accepts `low`, `medium`, `high`, and `xhigh`, and
`enable_thinking: false` emits the model's official empty thinking prefix.
Audio and grammar constraints are rejected explicitly. Images are supported;
see **Vision** below.

Tool calling works. Qwen3.8 declares and emits calls in an XML-ish form of
its own rather than the JSON block GLM uses, so it has its own renderer and
its own parser:

```
<tool_call>
<function=NAME>
<parameter=KEY>
VALUE
</parameter>
</function>
</tool_call>
```

Both sides are transcribed from `chat_template.jinja` rather than
paraphrased, because the declaration is what teaches the model the syntax it
must emit: a preamble it has never seen is a different prompt. The whole
rendering is pinned byte for byte against the official template
(`tests/test_qwen38_chat_template.py`, 27 cases across the three reasoning
levels).

One asymmetry worth knowing: the template writes a string argument unquoted,
so a value's original type is not recoverable from the text alone. The parser
reads the declared schema and restores numbers and booleans from it, and
leaves anything the schema did not describe as a string rather than guessing.

The official FP8 repository is about 185.5 GB in decimal units (roughly 173
GiB). The default CPU engine keeps resident BF16 matrices in their native
two-byte representation and selected experts in native block-FP8. Activations
are FP32; BF16 matrices use FP32 accumulation, while native FP8 uses FP32 dot
products within each 128-column block and FP64 accumulation across the scaled
blocks. The bounded per-layer cache therefore spends about one quarter of the
previous memory per FP8 expert. Context state grows by about 54 KiB per token.

## What stays on disk

The 51B-parameter PLE table is never materialized in RAM. For each token the
engine hashes its bigram and trigram history, reads sixteen 160-byte FP8 rows,
and applies the checkpoint's scalar PLE scale. Routed experts are likewise read
on demand: the official per-expert E4M3 gate, up, and down bytes and their
128 x 128 floating-point `weight_scale_inv` blocks remain native in an LRU whose
capacity is the first engine positional argument. The canonical FP8 matmul
decodes values during accumulation rather than materializing three FP32 matrices
per slot. The launcher defaults to one expert slot per layer.

Prompt execution is expert-major in bounded chunks: it routes up to 32 rows,
groups their assignments by expert, and consumes cache-sized parallel load
groups when the complete demand set is larger than the configured LRU. Shared
expert and DeltaNet projections are batched over the same bounded window;
DeltaNet convolution and recurrent updates remain token-causal. The private
chunk workspace is capped at 64 MiB independently of prompt length.

The serve path owns one hybrid prefix slot. When the next prompt begins with
the exact cached token sequence, QSA K/V/index rows are reused in place and the
saved DeltaNet/PLE recurrent state is restored before evaluating only the
extension. An identical prompt also reuses its saved final logits. Mismatched
or shorter prompts reset every hybrid component rather than guessing at cache
identity. Decode and cancellation mutate only live state, not the published
prompt snapshot. The complete configured QSA context bank is allocated once
before the engine publishes `READY`; serve never grows it alongside a still-live
old bank, so the planner's single context-bank reservation is also the peak.

QSA caches the two K/V heads and the indexer's raw key. Complete four-token
blocks are pooled and scored, the best 512 blocks are retained, and a causal
tail of up to three tokens is appended. The main 24-head attention then operates
only on those original tokens. The native model limit is 262,144 tokens;
`Q38_MAXT` defaults to 8,192 and may raise the server limit up to that native
ceiling when the required RAM is available. `coli serve --ctx N` (and `coli
chat --ctx N`) reach the engine as `Q38_MAXT=N`, not as `CTX`: that is the
variable to look for in the engine process's environment. `max_tokens` is a
ceiling: a budget the prompt leaves no room for is clamped to the room left
(one `[serve] max_tokens ... clamped` line on stderr), and only a prompt that
does not fit is refused.

## Memory and speed

The 185.5 GB on disk is not a RAM requirement. What must be resident is the
dense set: the DeltaNet and QSA projections, norms, gated-residual mixers,
embedding, shared experts and LM head. In native BF16 that is 9.2 GiB (9.9 GB),
and the engine prints it at load. Everything else is sized by a knob or by the
prompt:

| | |
|---|---|
| resident weights (native BF16) | 9.2 GiB, fixed |
| routed-expert cache | 4.7 MiB per slot per layer over 48 layers: cap 16 is 3.5 GiB, cap 32 is 7.0 GiB, cap 64 is 14.1 GiB (2.6 MiB with the [int4-g64 sidecar](#routed-experts-as-int4-g64): cap 64 is 7.9 GiB) |
| FP8 scale bank | 28 MiB, fixed; every expert's block scales stay resident so a miss is one FP8 read |
| context state | 54 KiB per token, allocated for the whole `Q38_MAXT` ceiling before `READY`: 432 MiB at the 8,192 default |
| recurrent and PLE state, prefix snapshot, cached logits | 226 MiB, fixed |
| prompt and decode workspace | at most 1.1 GiB peak, independent of context length |
| PLE n-gram table (51B parameters) | 0; sixteen 160-byte row reads per token |
| routed experts on disk | 120.8 GB, streamed |

`coli plan --model <dir> --ram <GB> --ctx 8192 --gpu none` prints this
accounting for a budget and chooses the expert cap from it.

Measured on the official checkpoint on an Intel Core i9-14900K (24 physical
cores, OpenMP 24), 61 GiB RAM, Samsung 990 EVO on ext4, GCC 15.2, Linux 7.0,
default 8,192 context, native FP8 and BF16; the engine reports RSS in binary
units. Peak RSS for an eleven-token prompt plus one generated token was
12.9 GiB at cap 16, 16.5 GiB at cap 32 and 21.5 GiB at cap 64, with TTFT
within 0.6 s across the three. Cap 32 is the short-request knee on this disk;
it fits a 24 GB machine, and cap 64 wants 32 GB. A 16 GB machine is below the
floor once the cache, context bank and workspace are added.

A full `tools/datapoint.py` campaign at `8563799` at cap 32: one persistent
`SERVE=1` engine, page cache evicted before load, greedy decoding, 128
completion tokens per request; one cold request, one warm-identical repeat of
it, then four different prompts in fixed rotation as the primary workload.
Engine load 11.2 s; cold and buffered `iobench` 2.26 and 2.55 GB/s:

| phase | prompt tok | request s | TTFT s | decode tok/s | hit | RSS |
|---|---:|---:|---:|---:|---:|---:|
| cold | 31 | 129.5 | 20.6 | 1.17 | 58.8% | 16.6 GiB |
| warm-identical (exact prefix reuse; upper bound) | 31 | 107.7 | 0.01 | 1.18 | 63.8% | 16.6 GiB |
| rotating prompts, median of four (primary) | 35 to 40 | 140.2 | 23.8 | 1.09 | 53.8% | 16.6 GiB |

Of the 140 s median request, 96 s was synchronous expert disk service, 17 s
expert matmul, 14 s attention and 2.6 s LM head. One decode token routes ten
of 512 experts in each of 48 layers, 4.7 MiB each: 2.2 GiB of expert weights
when nothing is cached and roughly half that at the cap-32 hit rate, so at this
cache size the engine spends about two thirds of every request waiting on the
disk, and the planner labels cold expert reads as the expected bottleneck.

## Routed experts as int4-g64

Optional. Decode waits on routed experts: a miss is a disk read, and how many
experts fit in the cache decides how often that happens.
`tools/convert_qwen38_experts_int4.py` rewrites them as int4 with one f32 scale
per 64 inputs into `<model>/experts-int4g64/`, next to the FP8 shards, which it
only reads; qwen38 picks the directory up by itself.

```sh
python3 c/tools/convert_qwen38_experts_int4.py --model ~/Models/Qwen3.8-Flash-Next-FP8 --plan
python3 c/tools/convert_qwen38_experts_int4.py --model ~/Models/Qwen3.8-Flash-Next-FP8
```

| | FP8 (release) | int4-g64 sidecar |
|---|---|---|
| bytes per expert | 4,915,200 (scales in the 28 MiB resident bank) | 2,764,800: codes 2,457,600, scales 307,200 |
| experts per GiB of cache | 218 | 388 |
| cap per GiB of cache, 48 layers | 4.5 | 8.1 |
| all routed experts on disk | 120.8 GB | 68.0 GB, in addition to the FP8 shards |

The values are the ones the engine computes from the release (e4m3 byte times
its block's `weight_scale_inv`), quantized as `tools/convert_qwen36.py --ebits 4
--gs 64` does: per 64 inputs, scale = absmax / 7, code = round half to even of
w / scale, clamped to [-8, 7]. The codes are stored as v+8 in `expert_ffn.h`'s
planar layout and multiplied by its f32 kernel, activations staying f32 as on
the FP8 path. On synthetic Gaussian weights at the release's geometry the
relative L2 error is 10.9% per matrix; the converter prints the mean and the
worst per layer on the real weights, where it measured 11-12% per layer and
17.5% for the worst single matrix.

Measured on the release (Ryzen 7 PRO 8700GE, 16 threads, 61 GiB, NVMe; the
same binary with `Q38_EXPERT_INT4=0` and `=1`, `OMP_NUM_THREADS=8`). Perplexity,
teacher-forced over four 504-token chunks at cap 96:

| chunk | FP8 | int4-g64 |
|---|---|---|
| 0 | 4.23 | 4.20 |
| 1 | 11.64 | 11.74 |
| 2 | 10.94 | 11.04 |
| 3 | 16.04 | 16.97 |

On average +0.017 nats per token. Three chunks move by under 1%; chunk 3 moves
by 5.8%. Greedy answers stay on the same reasoning with a few words changed.

Decode, 100 tokens of one prompt, the model's files dropped from the page cache
before every run, load average under 2.5:

| cap | FP8 | int4-g64 |
|---|---|---|
| 32 | 1.91 tok/s, 14.4 GB RSS, 45.8% hits | 2.87 tok/s, 11.6 GB, 46.4% |
| 64 | 2.27 tok/s, 21.4 GB, 59.1% | 3.25 tok/s, 15.2 GB, 59.4% |
| 96 | 2.56 tok/s, 28.5 GB, 68.0% | 3.55 tok/s, 19.2 GB, 67.8% |
| 170 | | 3.99 tok/s, 28.0 GB, 79.2% |

At the same cap int4 is 1.4-1.5x faster, because each miss reads 56% of the
bytes and the routed-expert compute halves (55 to 31 ms per decode forward at
cap 96). At the same RAM, FP8 at cap 96 (28.5 GB) against int4 at cap 170
(28.0 GB), it is 1.56x faster: the RAM that held 96 experts per layer holds 170.

The converter streams: one expert per worker in flight, so peak RAM is a few
hundred MB for the writer (records queue there while the disk catches up; at
most one layer, 1.4 GB) plus 0.11 GB per worker, whatever the model size. On a 12-thread laptop
CPU (i7-1355U) and a synthetic snapshot of the release's expert geometry a
layer is 22 s of compute with 12 workers; the disk sets the rest. It is
resumable: a layer is written to a `.part` file and renamed when whole, and
`index.json` lists the layers done, so a second run continues where the first
stopped. A run against a snapshot whose `config.json` or expert shards changed
is refused.

Layout: one safetensors file per layer (`layer-NNN.safetensors`) and
`index.json`. Every expert is one record, so a miss is one read:

```
gate codes U8 [I, H/2] | up codes U8 [I, H/2] | down codes U8 [H, I/2] |
gate scales F32 [I, H/64] | up scales F32 [I, H/64] | down scales F32 [H, I/64]
```

named like the release's tensors (`...experts.E.gate_proj.weight`, and the same
name plus `.qs` for its scales). The data region of each file starts on a
4096-byte boundary; at the release's geometry a record is 675 pages, so every
record is page aligned. `index.json` carries the format name and version, the
geometry, the group size, the record size, the layers done and `complete`;
the engine refuses a sidecar whose index or files disagree with `config.json`
and skips (with a line on stderr) one still being written.

`Q38_EXPERT_INT4=0` keeps the snapshot's FP8 experts; `=1` refuses to start
without a complete sidecar. Every expert path reads the sidecar: the single LRU
load, the parallel batch reads (`Q38_EXPERT_PARALLEL_READS`), the prefetch, the
prefill batch and `COLI_MAP_EXPERTS=1`. The startup line names the expert
representation, the bytes of one expert, what the cache costs full at the
given cap, and how many experts per layer the RAM available at that moment
would hold. `coli plan` and `--auto-tier` size the cache cap with the
sidecar's records and plan no VRAM for them: the CUDA expert tier streams FP8
experts only, and with int4 experts it stays off, trunk included. The Segment and Edge adapters keep the snapshot's
experts.

`make -C c qwen38-tiny-int4-check` converts the FP8 tiny fixture (experts
scaled by 3, so that they move the logits beyond the oracle's tolerance) and
gates on the int4 matmul against a dequantized f32 reference, the converter's
round trip against the FP8 source, the engine reproducing the float32
reference of the dequantized sidecar through every expert path with the last
logits identical to the bit across them, and the CUDA tier declining the int4
experts on the fake backend.

## Speculative decoding with the MTP head

On by default from 1.13.0 when the checkpoint has it (`Q38_MTP=0` turns it off, and
`Q38_MTP=1` refuses a checkpoint without it). The release carries one more decoder layer under
`mtp.*` that reads the model's four hyper-connection streams at a position
together with the next token's embedding and predicts the token after it.
The engine loads it and decodes speculatively: after each token the head
drafts the next one, and one forward over both (S=2) returns the logits of the
first as a plain decode step would while it checks the draft. When the token
picked from those logits equals the draft, the second row's logits answer the
next step with no forward; when it does not, the second row is undone: the
Gated DeltaNet and PLE state go back to a copy taken after the first row (a
pointer swap, nothing is recomputed) and the attention, indexer and prefix
records go back one position.

Every row of the verify forward is computed as a decode step computes it, so
the output is the output of plain decoding, token for token and logit for logit,
greedy or sampled (a sampled token that happens to equal the draft is an
accepted draft). The CUDA expert tier is the exception it already is: its float
order follows which experts are resident when.

The head is built only from blocks the engine already runs for its own layers:
the decoder layer is a QSA attention layer with its indexer, gated residuals,
the MoE with its shared expert, the head's own hyper-connection mixer and the
shared `lm_head`. The tensors do not say how `pre_fc_norm_hidden` (4 x 2560
wide) groups the model's four streams before `fc_hidden` maps each of them, so
`Q38_MTP_WIRING` selects it (`c/qwen38_core.h`, "the MTP head"): `b`, the
default, normalizes each stream on its own, the way the hyper-connection norms
do; `a` normalizes the four streams as one vector and stays as a diagnostic.
The output does not depend on the choice. The acceptance rate does, and on the
release it picks `b`.

Every run prints its line, and a served request prints one per turn, on stderr:

```
[qwen38 MTP] run: 1.94 tokens/forward (51 forwards per 99 tokens) | acceptance 94.1% (48/51 drafts) | wiring b
```

Tokens are the decode tokens after the prompt and forwards the forwards that
produced them. The head's routed experts stay the snapshot's FP8 with or
without the int4-g64 sidecar (one layer read for at most two rows per draft is
a small share of the expert traffic); their cache holds `Q38_MTP_CAP` experts,
by default the cap every layer has, and the startup line prints what it costs.

### Measured

On the release with the int4-g64 sidecar (Ryzen 7 PRO 8700GE, 16 threads,
61 GiB, NVMe; model files dropped from the page cache before every run, load
average under 2, `OMP_NUM_THREADS=8`, 100 new tokens). Greedy output was
byte-identical to MTP off in every run.

| prompt, cap | wiring | acceptance | tokens/forward | tok/s |
|---|---|---|---|---|
| raw text prompt, cap 96 | MTP off | | | 3.57 |
| | a | 92.2% (47/51) | 1.90 | 3.97 |
| | **b** | **96.0% (48/50)** | **1.94** | **4.01 (+12%)** |
| | c | 78.2% (43/55) | 1.77 | 3.76 |
| chat-template coding prompt, cap 170 | MTP off | | | 4.15 |
| | a | 84.9% (45/53) | 1.83 | 4.58 |
| | **b** | **94.1% (48/51)** | **1.94** | **4.74 (+14%)** |
| | c | 63.3% (38/60) | 1.62 | 4.21 |

`c` was a third reading (the normalized streams averaged into one vector
before a single `fc_hidden`). It came last on both prompts and was removed.
The means of the head's norm weights are -0.76 (embedding), -0.33 (hidden) and
3.79 (mixer), consistent with norms that scale by 1 + w like every other norm
of the model.

Nearly two tokens per forward gives only 12-14% more tokens per second here
because the forward is not where the time goes. A verify forward feeds two
tokens, and each token routes its own 10 experts per layer. Consecutive tokens
share on average 3.2 of their 10 experts per layer (measured on a
`ROUTE_TRACE` of 199 tokens), so a pair needs almost twice the experts of a
single token. At cap 170 the cache missed 74 experts per token whether the
tokens went one at a time or in accepted pairs. The disk reads for experts are
therefore the same with MTP and without it. What MTP saves is the dense part
of the forward, read once per forward instead of once per token: the trunk
matrices, DeltaNet, attention and `lm_head`. Where the routed experts fit in
RAM, those disk reads go away and the dense part is a larger share of each
token, so the gain should be larger. That case has not been measured.

### Deeper verifies and prompt lookup

`Q38_MTP_DRAFTS` lets the head draft up to 3 tokens per verify: each draft past the
first reads the head's own streams from the row before, with the draft just proposed.
The verify copies the DeltaNet and PLE state after each of its rows but the last, so a
rejection after row `k` restores the state after row `k`. `Q38_MTP_DRAFTS=0` lets a gate
pick the depth per verify from the measured acceptance by position and the measured
cost of a verify by its rows. Prompt lookup (on by default) adds drafts (up to 5, from
the context's n-grams), and a verify carries whichever proposal is worth more. The
output stays that of plain decoding. Measured on the release with the int4-g64 sidecar: 4.94 tok/s at two drafts against 4.71 at one and 4.12 without MTP on the CPU (2.75 tokens per forward), 4.55 against 4.39 and 3.91 with the Vulkan tier and chain on a Radeon 780M, so two drafts is the default. The details, the settings and the
tests are in [speculative.md](speculative.md).

`make -C c qwen38-tiny-mtp-check` adds a head with random weights under the
release's names to the three tiny fixtures and gates on the head's draft
logits against a float32 reference of both wirings
(`c/tools/qwen38_mtp_ref.py`), and on byte-identical output with drafts and
without them across caps, prefill batching, both trunk formats, BF16, FP8 and
int4 experts, text and serve, with drafts all rejected, all accepted and
alternating.

## GPU: Vulkan expert tier

In a `VK=1` build, `COLI_VULKAN=1` puts the trunk on the Vulkan device (as before)
and the routed experts of the model's layers on the shared Vulkan expert tier
(`c/vk_tier.c`, [vulkan.md](vulkan.md#the-routed-expert-tier-vk_tierc)): a cache of
experts in device memory, filled at startup from `.coli_usage` (`coli setup` ships a
starting one) and adapted while
you chat, whose experts the device computes while the CPU computes the rest of the
step. It takes every expert form this engine reads: the int4-g64 sidecar's planar
records (as int4 groups of 64 on the device), the release's FP8 with the 128x128
block-scale bank (fmt 12, the block scale repeated over its rows), BF16. Decode and
prefill both use it; an MTP verify's two rows take the device's per-row route, so
they get a decode step's bits; the MTP head's own layer goes on the tier as an extra
layer on a discrete GPU (`COLI_VK_TIER_MTP`, [vulkan.md](vulkan.md#the-mtp-heads-layer-on-the-tier-coli_vk_tier_mtp)). Every
expert's output joins its row in rank order, device or not.

```bash
make -C c qwen38 VK=1
COLI_VULKAN=1 SNAP=<checkpoint> ./c/qwen38 96 8 prompt.txt    # experts; the trunk too on a discrete GPU
COLI_VULKAN=1 COLI_VK_DENSE=0 ...                              # experts only, trunk on the CPU
COLI_VULKAN=1 COLI_VK_DENSE=1 ...                              # trunk on the device on any GPU
```

On a GPU that shares the CPU's RAM (an integrated GPU, Lavapipe) the trunk stays on
the CPU by default while the tier is on; the startup line says where it went and why.

On an integrated Radeon 780M the tier with the trunk on the CPU decoded at
3.80 tok/s against the CPU's 3.51 and reached the first token of a 512-token prompt
in 38.3 s against 43.9 s; the trunk on the device cost more there than the tier
gained. The budget, the knobs (`COLI_VK_TIER*`), those measurements and what they
leave out are in [vulkan.md](vulkan.md#the-routed-expert-tier-vk_tierc). With
`COLI_CUDA=1` as well, the CUDA tier below wins.

## GPU: CUDA VRAM expert tier

With `CUDA=1` the engine links the same expert tier as Qwen3.6
(`c/qwen36_tier.c`, [qwen36-cuda-tier.md](qwen36-cuda-tier.md)) in its
**fp8 streaming mode**. Nothing about the model or the RAM cache changes:
experts still stream from disk into the per-layer LRU (`cap`), and the tier
keeps its own copies of the hot ones in VRAM as a third stage above it,
disk -> RAM LRU -> VRAM. Routed experts that are resident are computed on
the GPU (`grouped_hidden_f8w_dual` / `grouped_down_f8w` from #817: e4m3
bytes plus the checkpoint's `[ceil(O/128), ceil(I/128)]` block scales, no
conversion); the rest run on the CPU as before. Every expert the CPU
computes is reported to the tier, which copies the 4.7 MiB slab and the
three scale tables during that call -- the RAM slot may be recycled by the
next token -- and promotes it when it has room or when it is hotter than
the coldest resident expert on its device (budget-neutral swap). Prefill,
the DeltaNet recurrence, QSA and the PLE table stay on the CPU; the dense
trunk goes to the GPU as well, see below.

```bash
make -C c qwen38 CUDA=1 CUDA_ARCH=native      # NVCC=/usr/bin/nvcc on distro CUDA
COLI_CUDA=1 COLI_GPUS=0,1 CUDA_EXPERT_GB=auto COLI_TIMERS=1 \
OMP_NUM_THREADS=<physical cores - 1> OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close \
SNAP=<checkpoint> N_NEW=200 ./c/qwen38 128 8 prompt.txt
```

`coli chat --gpu <n>` does the same through the planner (`supports_accelerator`
in the `qwen38` descriptor). The tier reports next to the cache hit rate:
resident experts, uploads, VRAM hits and misses, LFRU swaps. Requirements:
native FP8 routed experts (`Q38_NATIVE_FP8=1`, the default) with every layer's
block-scale bank resident; a checkpoint that falls back to the per-matrix
scale loader stays on the CPU and says so.

**What a VRAM budget buys.** The routing of Qwen3.8-Flash-Next has no hot set
that carries over between prompts: the 1,400 experts (5.8 % of the model) that
were hottest in one 100-token run covered a different prompt's routes at
chance level (4.7 %; two cards, 2,800 experts: 9.3 % against 11.5 % chance;
rank correlation of expert frequency per layer between the two runs -0.2). A
heat file from an earlier run therefore does not help, and the engine does no
warmstart. Inside a run the locality is strong -- on the route trace of a
315-token prompt plus 100 generated tokens, an LRU of 32 slots per layer
serves 55 % of decode routes, 64 serve 79 %, 128 serve 90 % -- so the tier
earns its VRAM by promotion at touch time: with the RAM LRU at cap 128, one
8 GB card served 45 % of decode routes from VRAM on the real checkpoint, two
cards 59 %.

**VRAM per expert.** The tier charges what `cudaMalloc` takes, not the payload:
an allocation above 1 MiB rounds up to a multiple of 2 MiB, so each 1.56 MiB
expert matrix occupies 2 MiB and an expert costs 6.03 MiB of VRAM for 4.69 MiB
of bytes. A 6.5 GB budget holds about 1,080 experts. (Qwen3.6's int4 matrices
are exactly 512 KiB and are served exactly, so its accounting was already
right.) Pooling experts into one arena per device would recover the 22 %; it
is not done yet.

**What it is worth on this machine -- measured, not projected.** Threadripper
PRO 3945WX (12 cores, `OMP_PLACES=cores`), 94 GB, checkpoint in the page cache,
prompt of 315 tokens plus 100 generated, cap 128, `COLI_TIMERS=1` decode bank
(the `Speed` line the engine prints divides by the whole generation time
including the prompt; the numbers below are decode only):

| | CPU only | tier, one 8 GB card | tier, two cards |
|---|---:|---:|---:|
| VRAM share of routed experts | -- | 45 % | 59 % |
| routed-expert GEMV on the CPU, ms/token | 192 | 126 | 93 |
| decode, ms/token | 806 | 808 | 808 |
| decode tok/s (three runs each) | 1.14 - 1.24 | 1.20 - 1.24 | 1.21 - 1.24 |
| greedy tokens | identical | identical | identical |

A decode token costs about 0.8 s here: 260 ms dense kernels (BF16 trunk and
the DeltaNet projections), 244 ms expert reads (page-cache copies of the cap
misses), 192 ms routed-expert GEMVs on the CPU (0.5 ms for the three FP8
matrices of one expert), 42 ms LM head, 24 ms QSA, 19 ms shared expert. The
tier removes 66 - 99 ms of GEMV time and spends about as much on its own
per-layer issue/take round trips and on staging the experts it promotes, so
the wall time does not move. The parity holds, the plumbing is exercised on a
real checkpoint, and the honest summary is: on a machine of this class the
expert tier is not where Qwen3.8's decode time is. The dense trunk (a third of
the token) and the expert reads (another third, a RAM-cap question) are.

### The dense trunk in VRAM (stage 1)

The trunk is the largest fixed cost of a decode token and it is
bandwidth-bound: 8 GiB of BF16 matmul weights read on every token at the
memory bus's pace. With the tier on, the engine offers every dense matmul
matrix of at least 1 MiB to the tier's placer by name and layer (the
DeltaNet projections `dnqkv`/`dnz`/`dnout`, attention `attnq`/`attnk`/
`attnv`/`attno`/`qsaidx`, the hyper-connection mixers `hcad`/`hcau`/`hcmd`/
`hcmu`, the shared expert `shg`/`shu`/`shd`, the `router`, and `lmhead`);
whatever the placer accepts is quantized to **int8 per row** (scale = max|w|
/ 127, the qwen36 dnproj/lmhead format) when the tier starts -- about 2 s
for the 553 matrices, 3.96 GiB on one card -- and answers decode GEMVs from
there, one round trip per matmul (`x` up, `y` down; activations stay on the
CPU). Prefill rows (S > 1) and any backend failure take the CPU path, which
holds the same int8 rows (see [the trunk on the CPU](#the-trunk-on-the-cpu-int8-rows)). The placer takes the trunk before the experts (it is
read on every token), so on an 8 GB card about 2 GB remain for hot experts;
`coli plan` prices it the same way (`4.0 GB int8 trunk + ... hot tier`).

| variable | effect |
|---|---|
| `Q38_TRUNK_GPU=0` | keep the trunk on the CPU (experts-only tier, the behaviour before this) |
| `Q38_TRUNK_MIN_KB=<n>` | offer matrices of at least n KiB (default 1024; a round trip costs more than a tiny GEMV saves) |
| `Q38_TRUNK_SKIP=name,name` | leave the named components on the CPU (bisecting, or a component that does not pay) |
| `QT_UPLOAD_SYNC=1` | the tier's `qt_issue` waits for every in-flight upload first (tests and diagnostics: deterministic residency, no upload/compute overlap) |
| `Q38_TRUNK_CPU_INT8=0` | keep the trunk BF16 on the CPU with the f32 kernel (the numeric reference); the default is int8 rows with integer dot products, GPU or not |
| `Q38_TRUNK_SELFTEST=1` | at start, every placed matrix is checked once: GPU GEMV against the same int8 rows on the CPU (relative error printed per matrix) |

**Numerics.** GPU int8 against CPU int8 on the same rows: relative error
~1e-7 on all 553 matrices (float summation order), greedy tokens identical
over 30 tokens on the 315-token prompt and 20 on a short one. int8 against
BF16: the per-row quantization error is 1 - 3 % per matrix (largest on the
router and the hyper-connection down-mixers), and on the same prompts the
greedy text is identical to the BF16 run for the 30 tokens compared.
Perplexity on wikitext-2 (8 chunks of 512, 2048 scored tokens, llama-perplexity
protocol): 1.880 with the trunk and the expert tier on the GPU against 1.845 for
the BF16 CPU run, +1.9 %. The expert tier on its own is neutral (+0.0008 nats on
a 256-token probe), the lm_head contributes nothing, and keeping the 48 routers
in BF16 on the CPU (`Q38_TRUNK_SKIP=router`) only recovers 0.3 points (1.875):
the per-row int8 error is spread over the trunk, not concentrated in one class.
Group scales (gs 64/128) for the trunk are the lever left; that needs a kernel
format, not a placement change.

**Measured** (same machine and prompt as above, cap 224 so the RAM LRU
serves 84 % of expert reads, `COLI_TIMERS=1` decode bank, 100 tokens):

| | CPU only (BF16 trunk) | experts only, one card | trunk on one card, no experts | trunk + experts, one card | trunk + experts, two cards |
|---|---:|---:|---:|---:|---:|
| VRAM: trunk / experts | -- | 0 / 6.1 GB | 4.0 / 0.1 GB | 4.0 / 2.1 GB | 1.8 + 2.2 / 4.3 + 4.3 GB |
| VRAM share of routed experts | -- | 43 % | 2 % | 22 % | 52 % |
| resident-mm (dense trunk incl. DeltaNet proj.), ms/token | 284 | 285 | 52 | 63 | 89 |
| shared expert / lm-head, ms/token | 21 / 46 | 21 / 46 | 6 / 5 | 7 / 5 | 12 / 14 |
| routed-expert GEMV on the CPU, ms/token | 179 | 118 | 182 | 151 | 93 |
| expert-read (page-cache copies), ms/token | 168 | 164 | 162 | 160 | 145 |
| **decode, ms/token** | **744** | **754** | **557** | **523** | **466** |
| **decode tok/s** | **1.34** | **1.33** | **1.80** | **1.91** | **2.14** |
| greedy tokens (100) | reference | identical | identical | identical | identical |

Peak RSS 60.1 - 60.5 GB in every run (cap 224); TTFT 183 - 185 s in every run (the prompt's
forward stays on the CPU). On two cards the placer spreads the trunk by free room (1.8 GB on
the 3070, 2.2 GB with the lm_head on the Quadro) and the experts on both; the Quadro is the
slower card (lm-head 14 ms there against 5 ms on the 3070), which is the +26 ms of
resident-mm against one card -- and the 52 % expert share still wins by 57 ms. The trunk is
worth +34 % on its own (1.34 -> 1.80), the expert tier nothing on its own (1.33) and +6 %
on top of the trunk (1.91), because the GEMVs it saves were the small part all along; the
two-card run gets its +60 % from both. The first calibration (06.09.) gave 1.45 -> 1.76 on
the same prompt with the old binary; today's CPU baseline is 8 % lower, run-to-run scatter
of the page-cache expert reads (168 vs 149 ms), so the ratios are the comparable figures.

One thing found on the way: the backend's resident dense matvec shared its
`x`/`y` staging buffers with the asynchronous expert group. The shared
expert's three GEMVs run between `qt_issue` and `qt_take`, so with the trunk
in VRAM they overwrote the in-flight group's input and output -- no CUDA
error, only wrong tokens, worse the more experts were resident. The dense
path has its own buffers now ([qwen36-cuda-tier.md](qwen36-cuda-tier.md));
qwen36 never called the dense path inside that window.

### The DeltaNet layer on the card (`Q38_DN_GPU=1`)

With the trunk in VRAM a DeltaNet layer still crosses the bus four times per
decode token: the `dnqkv` and `dnz` results come down, the convolution, the
recurrence and the gated norm run on the CPU, and the normed rows go back up
for `dnout`. On qwen36 those round trips were 8 of 39 ms per token
([qwen36-cuda-tier.md](qwen36-cuda-tier.md#the-deltanet-layer-on-the-card-q36_dn_gpu1));
Qwen3.8 has 36 such layers of 48 value heads. `Q38_DN_GPU=1` keeps the layer on
the card: for every DeltaNet layer whose `dnqkv`, `dnz` and `dnout` the placer
put on one device, the conv ring, the recurrent state (48 x 128 x 128 f32, 3 MB
per layer) and the conv/norm weights go there too, and a decode token runs the
layer end to end in one device chain -- x up, the two in_proj GEMVs, causal
conv + SiLU, per value head the L2 norms, the decay, the delta rule, the gated
RMSNorm with `sigmoid(z)` (Qwen3.8's gate; Qwen3.6 uses `silu(z)`), the
out_proj GEMV, out down. The two gates (`a`, `b`) stay on the CPU and travel as
kernel parameters.

The host arrays remain the state's owner: a prompt (`S > 1`), an MTP verify
(`snap_after`: the CPU path takes the snapshot after its first row), a
`--pin` snapshot and the prompt cache pull the state down first; a reset, a pin
restore and a rejected draft's rollback invalidate the card's copy. A failing
GPU step turns the layer off and the CPU continues from what the card holds.
Opt-in; `tests/test_qwen38_dn_gpu.c` (`make qwen38-dn-gpu-check`) runs the tiny
fixture's oracle with the layers on the fake card.

### The trunk on the CPU: int8 rows

Without a GPU the same trunk is the decode's floor: 8 GiB of BF16 read on
every token, multiplied by a scalar loop. Since 1.12.1 the engine keeps the
trunk on the CPU as **int8 rows with one scale per row** (the format the
tier uploads, so GPU or not the rows are the same bytes), releases the BF16
copy once the rows exist, and multiplies with the integer kernels of
`idot.h`: the activation is quantized to int8 once per row and meets the
weights with maddubs / vpdpbusd (AVX2 / AVX-512 VNNI) or NEON dot products.
Decode and prefill take this path. Every matrix of at least
`Q38_TRUNK_MIN_KB` (default 1 MiB) is in; the small ones stay BF16 because
they cost nothing either way. `Q38_TRUNK_CPU_INT8=0` keeps the BF16 rows and
the f32 kernel, the numeric reference.

The routed experts stay e4m3 with their 128 x 128 block scales, but the
kernel decodes eight bytes at a time in registers and multiplies with FMA
(`q38_matmul_fp8_vec`); the block scale still applies once per block and the
blocks still add in double, so it differs from the table kernel only by the
float summation order inside a block. For a batch of prefill rows the block
is decoded once and every row runs through it. `Q38_FP8_KERNEL=scalar`
restores the table kernel.

**Measured** on the released Qwen3.8-Flash-Next-FP8, a 16-core server shared
with a training job, `OMP_NUM_THREADS=8`, RAM LRU cap 96 per layer, 100
generated tokens, `COLI_TIMERS=1` decode bank:

| | BF16 trunk, table FP8 (1.12.0) | BF16 trunk, vector FP8 | int8 trunk, vector FP8 (default) |
|---|---:|---:|---:|
| decode, tok/s | 0.61 | 0.77 | **1.42** |
| resident-mm (dense trunk), ms/token | 434 | 437 | 85 |
| lm-head, ms/token | 76 | 79 | 13 |
| routed-expert GEMV, ms/token | 388 | 154 | 140 |
| shared-expert, ms/token | 43 | 44 | 15 |
| expert-read (page cache), ms/token | 156 | 155 | 140 |
| peak RSS, GB | 32.2 | 32.2 | 28.5 |

The middle column reproduces the first's text byte for byte over the 100
tokens: the vector kernel is a faster way to compute the same thing. The
int8 trunk is a quantization, and its cost is measured as perplexity, not
assumed. Teacher-forced NLL on four 512-token chunks of the repository
docs (8 prompt tokens, 504 scored), same machine:

| chunk | BF16 trunk | int8 trunk | BF16 trunk, vector FP8 |
|---|---:|---:|---:|
| 0 | 1.4527 | 1.4416 | 1.4527 |
| 1 | 2.4594 | 2.4544 | |
| 2 | 2.3845 | 2.3929 | |
| 3 | 2.7463 | 2.7753 | |
| mean nats/token | 2.2607 | 2.2661 | |

+0.24% nats, about +0.5% perplexity, two chunks lower and two higher: the
per-row int8 error is noise at this size. The vector FP8 kernel alone
reproduces the BF16 run to four decimals.

Prefill takes the same paths, and it is where the scalar BF16 loop hurt
most. The 512-token chunk 0 as a prompt, one generated token:

| | BF16 trunk, table FP8 | int8 trunk, vector FP8 |
|---|---:|---:|
| wall, 512 prompt tokens | 495 s | 149 s |
| resident-mm | 205 s | 12 s |
| routed-expert GEMV | 172 s | 34 s |
| shared-expert | 15.5 s | 1.8 s |
| expert-read | 28 s | 26 s |

The trunk quantization takes 3.3 s at load for the 553 matrices and the
RSS after load grows by 0.6 GB while the int8 rows and the BF16 copy
coexist; once the BF16 is released the peak RSS of a run is 3.7 GB lower.

## Performance telemetry

Every served request reports its own routed-expert cache hit rate; persistent
engine counters are differenced at request boundaries rather than exposed as a
cumulative percentage. `DONE` decode throughput is based on completed
inter-token intervals: the first token comes from prefill, so a one-token reply
correctly reports zero decode tok/s instead of an artificial near-infinite rate.
The request wall time, prompt/completion counts, expert read/wait/matmul time,
sequence-mixer time, LM-head time and actual forward count are emitted through
the shared `PROF` frame consumed by `tools/datapoint.py`.

Set `COLI_TIMERS=1` for the finer Qwen-specific breakdown on stderr: expert
reads, FP8 expansion, routed and shared experts, resident matmuls, DeltaNet, QSA
indexing, QSA attention, PLE and the LM head. Architecture-phase times overlap
the resident-matmul counter by design; expert disk service also overlaps the
synchronous miss-wait value in `PROF`.

## Correctness gate

The tiny oracle is generated from the `Qwen4ExpForCausalLM` class in
`transformers==5.16.1`, the first release that exports the Qwen4-Exp text
class:

```sh
python -m pip install -r c/tools/requirements-qwen38-tiny.txt
make -C c qwen38-tiny-check
```

It exercises Gated DeltaNet, sparse QSA above its token budget, PLE, four gated
residual streams, routed and shared experts, prefill, cached decode, and LRU
eviction. The gate checks both greedy token IDs and the final upstream logit
vector, and runs both native-BF16 and expanded-FP32 resident modes at cache
capacities one and four. CI repeats the capacity-one path under ASan and UBSan
and verifies that a config/tensor shape disagreement is refused.
`make -C c qwen38-tiny-int4-check` gates the optional int4-g64 experts (see
[Routed experts as int4-g64](#routed-experts-as-int4-g64)), and
`make -C c qwen38-tiny-mtp-check` the MTP head (see
[Speculative decoding with the MTP head](#speculative-decoding-with-the-mtp-head)).

## Supported checkpoint layouts

The released multimodal checkpoint stores text tensors below
`model.language_model`; the standalone upstream text class stores them below
`model`. Both prefixes are accepted. Routed experts may be the official
per-expert block-FP8 matrices or the fused BF16 tensors emitted by the upstream
text class. `Q38_NATIVE_FP8=0` restores the former expanded-FP32 expert cache for
numerical/performance A/Bs; `Q38_NATIVE_BF16=0` does the same for resident and
routed BF16 matrices. Other model types, unsupported scale encodings, and
incompatible tensor shapes fail during load.

`Q38_PREFILL_BATCH=0` restores row-at-a-time prompt execution for controlled
A/B diagnosis. It does not change the single-token decode path.

The weights remain covered by the Qwen Community License 1.0 in the downloaded
checkpoint. They are not redistributed by Colibri.

## Vision

The released checkpoint is multimodal and the engine already reads its
`model.language_model` prefix, so the vision tensors are present and reachable.
The tower and gateway image path are implemented; their checks are described
below. Image preprocessing is checked with `tools/qwen38_image.py`, pinned
against the official `Qwen2VLImageProcessor` in `tests/test_qwen38_image.py`.

```
python3 tests/test_qwen38_image.py --config <model>/preprocessor_config.json
```

Neither needs the weights. The reference processor is built from
`preprocessor_config.json` alone, which is 390 bytes, so the preprocessing can be
developed and verified without the 185 GB.

Measured against it on eight shapes: **geometry and patch order identical**, and
pixels bit-identical wherever no resampling happens (0.0000 on 256x256 and
640x480), 0.0157 worst case where it does, which is Pillow's bicubic against
torchvision's. A wrong patch order would show as a discrepancy near 1, not 0.01.

Two things differ from GLM-5.3's tower and are the reason this is its own file:

**The resolution is dynamic.** GLM-5.3 fits everything onto a 448 canvas and
pads. Qwen keeps the aspect ratio and picks a canvas whose *area* falls inside
`[shortest_edge, longest_edge]`, so there is no padding but the token count
depends on the image. A 1080p photo becomes **2040 tokens**, which on a
disk-streaming engine is a prefill nobody will sit through -- the same reason
`GLM53_MAX_IMAGE_TOKENS` exists, and `preprocess(max_tokens=...)` is the same
lever here. It shrinks rather than crops: what is lost is detail, not pieces.

**Normalisation is 0.5/0.5**, not the CLIP constants.

### The tower

`qwen38_vision.h` implements it: 27 blocks, hidden 1152, 16 heads, patch 16,
spatial merge 2, projecting to 2560. Verified against the upstream
`Qwen4ExpVisionModel`, again without the checkpoint -- the fixture is a 240 kB
tower with random weights:

```
make -C c qwen38-vision-check
```

Two findings from building it are worth carrying, because both were invisible
until an oracle was there to see them.

**The merger uses a different GELU from the blocks.** The blocks take
`ACT2FN[hidden_act]`, which is `gelu_pytorch_tanh` here; the merger instantiates
`nn.GELU()`, the exact erf one. Using one for both matches to 4.6e-3 -- close
enough to look right and far enough to move the image tokens.

**A fixture can be too weak to test what it claims to.** The first version scaled
the random weights to 0.05, which makes the q.k products so small that softmax
comes out essentially uniform: attention degenerates into the mean of the values
and stops depending on the scores. A tower with **no RoPE at all** passed that
fixture. At 0.6 the scores have a real range, and the four negative controls
(rope off, wrong GELU, raster patch order, flat position interpolation) all fail
as they should.

The tolerance is measured, not chosen: the reference in float32 differs from
itself in float64 by 1.83e-4 on these activations, so a gap of that order is the
arithmetic rather than a defect, and the threshold sits between it and the
4.6e-3 the real bug produced.

### Wired up

Images work end to end. Send an OpenAI `image_url` part with a base64 data URI
(a local path is accepted only under `COLI_IMAGE_ROOT`, see
`docs/ENVIRONMENT.md`; `coli chat` reads a pasted path itself and sends the
data URI); the gateway preprocesses it, replaces the part with
`<|vision_start|>` + N x `<|image_pad|>` + `<|vision_end|>`, and hands the patches
to the engine in an `IMAGE` frame ahead of the `SUBMIT` they belong to. The engine
runs the tower once and substitutes its output for the embedding of each
placeholder.

N is not a constant. The resolution is dynamic, so the placeholder count comes
from the grid the preprocessor chose, and the engine **refuses** a request where
the prompt and the grid disagree rather than guessing which vectors go where.
`Q38_MAX_IMAGE_TOKENS` caps it; a 1080p photo is 2040 tokens without one.

```
make -C c qwen38-vision-serve-check
```

That gate does not check the model answers -- with random weights it would answer
regardless, and would answer identically while ignoring the picture entirely. It
checks that **two different images produce two different answers**, which is the
only question a random fixture can answer honestly, and the one that catches the
likeliest defect: patches loaded, tower run, result dropped somewhere between the
merger and the embeddings. It also checks the refusals, since accepting a wrong
image is worse than refusing it.

Remote URLs are refused rather than fetched, as elsewhere: a request should not
make the server open a connection of the sender's choosing.

One image per request for now -- the engine holds a single pending image, and a
second arriving before its `SUBMIT` drops the first and says so.
