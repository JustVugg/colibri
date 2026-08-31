# Qwen3.8-Flash-Next on colibri

`c/qwen38.c` runs the language model in
[`Qwen/Qwen3.8-Flash-Next-FP8`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)
directly from the official safetensors shards. No conversion or second copy of
the weights is required. This contribution is deliberately **text-only**:
Colibri does not load or advertise the checkpoint's vision encoder, and it does
not use the optional MTP layer.

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

`coli serve` and `coli web` use the same text-only gateway path. Qwen3.8 thinks
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
There is currently no Qwen3.8 GPU backend.

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
ceiling when the required RAM is available.

## Memory and speed

The 185.5 GB on disk is not a RAM requirement. What must be resident is the
dense set: the DeltaNet and QSA projections, norms, gated-residual mixers,
embedding, shared experts and LM head. In native BF16 that is 9.2 GiB (9.9 GB),
and the engine prints it at load. Everything else is sized by a knob or by the
prompt:

| | |
|---|---|
| resident weights (native BF16) | 9.2 GiB, fixed |
| routed-expert cache | 4.7 MiB per slot per layer over 48 layers: cap 16 is 3.5 GiB, cap 32 is 7.0 GiB, cap 64 is 14.1 GiB |
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
The tower itself is not implemented yet; images are still refused rather than
silently dropped.

What exists today is the half that decides whether vision is *correct* rather
than nearly correct: `tools/qwen38_image.py`, pinned against the official
`Qwen2VLImageProcessor` in `tests/test_qwen38_image.py`.

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

Images work end to end. Send an OpenAI `image_url` part with a base64 data URI or
a local path; the gateway preprocesses it, replaces the part with
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
