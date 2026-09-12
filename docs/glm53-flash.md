# GLM-5.3-Flash engine (`c/glm53.c`)

A sibling engine for [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)
(321B parameters, 45 layers + MTP, with a vision tower), following the
one-engine-per-family pattern (`colibri.c` = GLM-5.2, `kimi_k3.c`, `inkling.c`).
It shares `st.h`, `json.h`, `tok.h`, `quant.h` and `hyper_connections.h`, and
touches nothing in the other engines.

```
make glm53
./glm53 --model <dir> --prompt "Ciao, come stai?" --greedy 32
```

Or through the launcher, which is where images and chat templates live:

```
coli chat  --model <dir> --no-think
coli serve --model <dir>
coli web   --model <dir>
```

## Getting the model

The engine reads a converted container, not the HF snapshot. One pass does both
the download and the conversion, one shard at a time:

```
python3 tools/convert_glm53.py --outdir /path/glm53_i4 --min-free-gb 30
```

Peak disk is the output plus a single 5 GB source shard, never the repository's
328 GB. On the reference machine: 62 shards, **194.7 GB out, 25 hours**.

Routed experts become int4 group-scaled at 64 (`name` U8 nibbles + `name.qs`
F32 scales, the same container GLM-5.2 uses). Everything else stays BF16 — 9.7B
parameters of 321, about 18 GB — so the precision of the dense set is a load-time
choice and retuning it never means downloading the repository again. Every one of
the 92 tensor kinds is classified explicitly and an unrecognised name stops the
conversion, because a converter that skips what it does not know produces a
checkpoint that loads and is quietly missing a tensor.

## Architecture notes

Four pieces differ from everything else in this repository:

**KDA (Kimi Delta Attention)** on 34 of the 45 layers: `q,k,v = SiLU(ShortConv4(Wx))`,
q and k L2-normalised with the epsilon **inside** the square root (FLA's
convention, deliberately unlike `F.normalize`), then the gated delta rule. The
output gate is low-rank here where Kimi K3's is a single projection.

**DSA with k-pooling** on the other 11. Keys are grouped into pools of
`index_kpool`; the pooled key is a per-channel softmax mixture rather than an
average, so a pool is not forced to describe itself by its mean. A pool is
selectable only if it is complete and causally visible, the incomplete tail is
appended when asked for (hence a row `topk + pool - 1` wide, not `topk`), and
ties go to the lower index so selection is deterministic.

**mHC hyper-connections**, shared with DeepSeek V4 through
`hyper_connections.h` — measured bit-compatible, not assumed.

**Clamped SwiGLU in the text MLP**, not only in the vision tower: `gate` has a
ceiling, `up` is clamped both ways. Plain SiLU here gives a model that speaks
well and is wrong.

### MLA is absorbed

Caching expanded keys costs 1.39 MB per token across the DSA layers — 11.9 GB at
8192 positions, on an engine that exists to fit in small memory. `kv_b_proj` is
folded into the two ends instead, so the cache holds the 512-wide latent:

```
score_j = q · (W_k c_j) = (W_kᵀ q) · c_j
out     = Σ_j a_j (W_v c_j) = W_v (Σ_j a_j c_j)
```

An identity, not an approximation: only the order of the products changes.
**33 KB per token**, which is 1.1 GB at 32k positions and 4.4 GB at the full
128k. The weights do not grow either, because transposing W_k keeps its element
count.

## Memory and speed

### CPU streaming baseline

The following historical numbers were measured before the production
device-resident HIP backend, on the real checkpoint, 6 physical cores, 25 GB
RAM, and an ordinary disk:

| | |
|---|---|
| resident weights at `GLM53_BITS=4` | ~12 GB |
| KV state | 33 KB per token (1.1 GB at 32k) |
| prefill workspace | flat, ~60 MB at any context |
| decode | ~44 s/token cold, ~20 s/token with a warm expert cache |

**The disk is the wall, and it is worth knowing where it sits.** One token
touches 42 sparse layers × 8 experts × 14.2 MB = 4.8 GB. Measured with
`O_DIRECT`, the reference disk gives 72 MB/s at queue depth 1, 185 at QD4 and
207 at QD16 — saturating near 200 MB/s. That puts a **floor of 24 seconds per
token** on this hardware: what the disk takes to deliver the bytes, with any CPU
and any GPU. Faster silicon does not move it; fewer bytes would.

That is also why GPU paths are most useful on machines with enough memory to
hold a large expert working set.

### Device-resident HIP text backend

`GLM53_BACKEND=auto|gpu|cpu` selects one complete backend before a request:

- `auto` uses the GPU only after context, capability, model, smoke-session, and
  smoke-forward checks all pass, otherwise it selects CPU;
- `gpu` makes any startup or capability failure fatal;
- `cpu` never initializes HIP/CUDA.

The GPU text path keeps embedding, four mHC residual streams, RMSNorms, KDA
recurrent state/windows, paged MLA/indexer state, routing, shared/routed/dense
FFNs, stream collapse, final norm, and LM head on one device. The host retains
tokenization, sampling, expert file reads, and LRU metadata. Only selected
expert IDs and final logits cross the production boundary; hidden activations
do not.

```bash
make -C c glm53 HIP=1 HIP_ARCH=gfx942
HSA_OVERRIDE_GFX_VERSION=9.4.2 GLM53_BACKEND=gpu \
GLM53_EXPERT_GB=28 ./c/glm53 --model /path/to/glm53_i4 \
  --prompt "The key insight about mixture of experts is" --greedy 64
```

Validated on one AMD Instinct MI350P (gfx950 exposed as gfx942), 154.6 GB HBM
and 123 GB host RAM. The 64-token warm run measured 13.5 s/token, 56.3% expert
hit rate, 151.59 GB expert bytes, and 84.4% mean GPU busy.

This is not comparable to the earlier 1.4 s/token partial-HIP experiment:
that older path ran experts and some resident matmuls on the GPU while KDA,
MLA, routing, mHC, and activation flow remained CPU-side.

The present expert cache is double-banked and keeps host copies of GPU slots.
Automatic sizing attempted roughly 97 slots/layer and was OOM-killed on the
123 GB host. The completing run used `GLM53_EXPERT_GB=28`, or 47 slots on each
of 42 sparse layers. A larger host or removal of host mirrors is required for
a matched-residency benchmark.

## Vision

Reachable from every surface: a path pasted in `coli chat`, a file attached or
dropped in `coli web`, or an OpenAI `image_url` part with a base64 data URI or a
local path. Remote URLs are refused rather than fetched — a request should not
make the server open a network connection of the sender's choosing.

`tools/glm53_image.py` does the preprocessing and is pinned against the official
`Glm5NextImageProcessor`: identical geometry on every shape tried, bit-identical
pixels wherever no resampling happens, and 0.03 worst case where it does, which
is Pillow's bicubic against torchvision's. The image is scaled with its aspect
kept and padded, never stretched, and padded with zeros *before* normalisation.

Images travel in their own `IMAGE` frame, announced immediately before the
`SUBMIT` they belong to (see `docs/serve_protocol.md`). The engine holds one
pending image and drops an older one rather than answering about the previous
photo without saying so.

`GLM53_MAX_IMAGE_TOKENS` matters more than it looks. The checkpoint's own
ceiling is 8000 tokens per image, which is 2691 for an ordinary 1080p photo — on
an engine that streams experts from disk, a prefill nobody will sit through.
Each image token covers 28×28 pixels, so 256 keeps ordinary text legible and 64
keeps only shapes and colours. The image is shrunk, not cropped: what is lost is
detail rather than pieces.

## Reasoning and tools

The generation prompt opens `<think>` and the model closes it, which is what the
official template does; `--no-think` closes it immediately instead, and the
answer starts at the first word. That form is not in the template — it does not
contemplate switching reasoning off — but it is exactly what the template writes
in front of a past turn that had no reasoning, so the model has seen it.
`--effort` picks the level the template understands: low, high or max.

Both are time controls on this engine, not matters of taste.

Tool calling is complete. GLM-5.3 declares tools differently from GLM-5.2 (its
own preamble, its own JSON serialisation, its own spacing inside `<tools>`) but
emits calls identically, so the existing parser handles them unchanged. The
whole rendering is pinned byte for byte against `chat_template.jinja`
(`tests/test_glm53_chat_template.py`).

## Environment

See `docs/ENVIRONMENT.md` for the table. The ones that change the shape of a run:
`GLM53_BITS` (dense precision, default 4), `GLM53_EXPERT_GB` (expert cache;
measured from available memory when unset), `GLM53_MAX_IMAGE_TOKENS`,
`GLM53_PREFILL_CHUNK`, `GLM53_MAXT`.

## Tests

```
HSA_OVERRIDE_GFX_VERSION=9.4.2 make -C c hip-test HIP_ARCH=gfx942
taskset -c 0 make -C c test
make -C c glm53-quality HIP=1 HIP_ARCH=gfx942

python3 tools/make_glm53_multimodal_tiny.py --output ~/glm53_mm_tiny
python3 tools/make_glm53_streaming_pair.py --fixture ~/glm53_mm_tiny --output ~/glm53_stream
python3 tests/test_glm53_multimodal_tiny.py --binary ./glm53 --fixture ~/glm53_mm_tiny
python3 tests/test_glm53_streaming.py --binary ./glm53 \
        --quantized ~/glm53_stream-i4 --dequantized ~/glm53_stream-deq
python3 tests/test_glm53_serve.py        --binary ./glm53 --fixture ~/glm53_mm_tiny
python3 tests/test_glm53_vision_serve.py --binary ./glm53 --fixture ~/glm53_mm_tiny
python3 tests/test_glm53_chat_template.py --template <model>/chat_template.jinja
make VK=1 glm53 && python3 tests/test_glm53_vulkan.py --binary ./glm53 --fixture ~/glm53_mm_tiny
```

`hip-test` includes the generic backend suite plus GLM-5.3 mHC, KDA,
MLA/DSA, MoE, full-pipeline, and production request/startup integration.
The real-checkpoint quality tool compares GPU against the complete CPU engine.
On the MI350P run, mean NLL changed by `6.65e-6`; all numerical gates passed;
four greedy tokens matched CPU on three prompts; and repeated GPU hashes were
bitwise identical.

The first quality attempt is retained as useful negative evidence. It produced
GPU NLL around 73 because grouped `fmt=4` int4 was incorrectly decoded as a
signed nibble. Production `fmt=4` is offset binary (`nibble - 8`, packed zero
`0x88`); legacy `fmt=2` remains signed-nibble after its XOR conversion.

The generators want transformers 5.16.1, pinned because an oracle written by a
different version is a different oracle. Each test skips with the command that
builds what it is missing rather than throwing.

Two of the generators refuse to write a fixture that cannot fail: one rejects a
degenerate model that answers the same token everywhere, the other a fixture
whose answer does not change when the image is inverted. The vision test does
not check that the model answers — it would answer anyway, ignoring the pixels —
but that **two different images give two different answers**.

`tools/check_glm53_container.c` compares a real converted expert against a numpy
dequantisation of the same bytes. It needs a converted checkpoint, so it does not
run in CI, and it is the first thing to reach for when the real model answers
strangely.
