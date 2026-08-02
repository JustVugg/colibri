# Kimi K3 engine (`c/kimi_k3.c`)

A sibling engine for [Kimi K3](https://huggingface.co/moonshotai/Kimi-K3)
(2.8T parameters, 104B active, 93 layers), following the one-engine-per-family
pattern (`colibri.c` = GLM-5.2, `olmoe.c`, `inkling.c`). It shares `st.h`,
`json.h`, `tok.h`, `quant.h` and touches nothing in the GLM engine.

```
make kimi_k3
./kimi_k3 <model_dir> "prompt" --ngen 64
```

`<model_dir>` is either the plain HF snapshot (config.json +
`model-*-of-000096.safetensors`) or a repacked container (below). Text-only:
the vision tower (shards 95–96) is never read.

## Architecture notes

K3 is not a DeepSeek-shaped model; four pieces are new relative to everything
else in this repo:

- **Hybrid attention, NoPE.** 69 KDA (Kimi Delta Attention) layers + 24 gated
  MLA layers (every 4th, plus the final layer). There is no positional
  encoding anywhere — position lives in KDA's convolution and decay. The MLA
  block has DeepSeek dims (q_lora 1536, kv_lora 512, nope 128 + 64 uncached
  extra dims, v 128, 96 heads) and both attention types multiply their output
  by a full-rank sigmoid gate `σ(W_g x)` before `o_proj`.
- **KDA** per head (dim 128, 96 heads): `q,k,v = SiLU(causal-conv4(W x))`,
  q/k L2-normalized (ε=1e-6 inside the sqrt), q scaled by 128^-1/2; per-channel
  decay `gk = -5·σ(exp(A_log[h])·(W_fb W_fa x + dt_bias))`; the delta-rule
  state update `S = (I − βkkᵀ)·Diag(e^gk)·S + βkvᵀ`; output
  `W_o[σ(W_g x) ⊙ RMSNorm_head(Sᵀq)]`. The checkpoint stores `A_log` as
  `[128]`: that is the per-head `[96]` parameter zero-padded (verified —
  entries 96..127 are exactly 0). Decode state: 96×128×128 f32 per layer.
- **AttnRes replaces the residual stream.** Each layer keeps a running
  `prefix_sum`; at layers 0, 12, 24, …, 84 it is snapshotted into a block
  list. Twice per layer (before attention, before the MLP) and once at the
  end, the hidden state is REPLACED by a softmax mix over
  `[snapshots…, prefix_sum]`, scored by `(v · (res_norm.w ⊙ res_proj.w)) /
  rms(v)` and mixing the raw (un-normalized) entries in fp32.
- **Stable LatentMoE.** Sigmoid router `[896, 7168]` + score-correction bias,
  top-16 selected on biased scores, weights are the raw scores renormalized.
  The token is projected 7168→3584, the 16 experts run in that latent space
  (GLU, inter 3072), the aggregate is RMSNorm-ed and projected back
  3584→7168; two fused full-width shared experts (inter 6144) are added.
  Activation is **SiTU-GLU**: `4·tanh(g/4)·σ(g) · 25·tanh(u/25)`.

### Weights

Only the routed experts (`experts.N.w{1,2,3}`) are quantized in the
checkpoint — **MXFP4** (compressed-tensors `mxfp4-pack-quantized`,
quantization-aware trained): e2m1 nibbles packed two per byte (low nibble =
even column, bit 3 = sign) with a ue8m0 power-of-two scale per 32 columns,
`w = v · 2^(scale-127)`. 17.55 MB per expert, 82,432 experts ≈ 1.45 TB.
`quant.h` gained `matmul_mxfp4` (scalar + AVX2) that computes on this layout
directly — expert bytes are **never converted**; QAT weights are exactly the
trained values and any re-encode only adds error.

Everything else is BF16 and is quantized into RAM **at load time** (int8
per-row / int4-g64), or read pre-quantized from a repacked container.

## Streaming design

Experts are streamed straight from the safetensors shards through a per-layer
LRU (`K3_EXPERT_GB`). Measured layout facts the engine exploits: the six
tensors of an expert are stored back-to-back (one expert = one `pread`), and
experts are *not* id-ordered inside a shard, so loads are issued in disk-offset
order. A token's misses are read **in parallel** (OMP over working-set slots)
and by default with **O_DIRECT** (`K3_DIRECT=0` for buffered): the resident
weights leave little page-cache headroom, and flat routing means cached reads
mostly cannot be reused anyway — measured on the 93-layer model this took
expert reads from ~1.8 to ~6.3 GB/s (drive ceiling 7.1) and decode from ~21
to ~9.4 s/token. The LRU slot floor is 1 (experts are consumed one at a time),
so `K3_EXPERT_GB` is honored even at tiny budgets. On top of that, `K3_PIPE`
(default on) runs the reads on loader threads so expert j's matmuls overlap
expert j+1's pread, `K3_IDOT` (default on) computes the expert matmuls with
per-32-group int8-quantized activations via integer dots (the e2m1 doubled
values are exact int8 — same trick as `dot_i4i8`), and `K3_TOPP` optionally
drops the low-weight tail of the top-16 (renormalized; quality-gate any
setting with a `K3_LOGITS` A/B first). Note that K3's router was trained with Quantile Balancing (deliberately
flat expert usage), so LRU hit rates are structurally lower than on models
with skewed routing — expect the expert tier to be bandwidth-bound.

## Repacked container (`tools/k3_repack.py`)

```
python3 tools/k3_repack.py <hf_src> <dst> --bits 8 [--mla-bits 8] [--head-bits 8]
```

One streaming read of the source (so a disk-to-disk copy can *be* the
conversion): experts pass through byte-identical (id-ordered, single-pread
layout), the big BF16 matrices are quantized with exactly the engine's
load-time algorithm and stored as `U8` + `<name>.qs` f32 scales, the small /
sensitive tensors (norms, router, conv taps, `dt_bias`, `A_log`, `f_a/f_b`,
`b_proj`, embeddings) pass through, vision is dropped. Output is spec-valid
safetensors (`model-XXXXX-of-000094.safetensors`) plus a regenerated index;
interrupted runs resume per shard. `--verify-full` re-reads every expert byte
and compares against the source.

The engine auto-detects container tensors (dtype U8 + `.qs` sidecar) and skips
load-time quantization. Setting `K3_BITS=4` **explicitly** on an int8 container
downcasts the int8 matrices to int4-g64 at load (~35 vs ~57 GiB resident on
the 93-layer model — fits next to a desktop session; the int8 grid is 16x
finer than int4, so the double-quant noise ~ direct int4). Unset `K3_BITS`
keeps the container's own bits. Startup: measured on two layers, init drops **30.4 s → 0.6 s**;
on the full model this removes a 10–15 minute quantization pass. Quantized
values are bit-identical to the load-time path (verified by hidden-state
trace), so the two formats are numerically interchangeable.

Sizes: source 1.56 TB → ≈1.50 TB (`--bits 8`) / ≈1.48 TB (`--bits 4`). The
experts (93 % of bytes) are already at 4.25 bits/weight and cannot shrink
losslessly; sub-4-bit expert re-encodes (fmt=5/6) would be double quantization
of QAT weights and are deliberately not offered here.

## Tokenizer

The HF repo ships only a raw tiktoken vocab (`tiktoken.model`).
`tools/k3_tokenizer.py` writes a `tokenizer.json` for it, and `tok.h` gained:

- a **kimi pre-tokenizer family** (sniffed via `\p{Han}` in the Split regex):
  the o200k case-aware rules plus a leading Han-run rule, Han excluded from
  the letter classes, and no `/` tail in the punctuation rule;
- a **rank-BPE mode**, used when `model.merges` is empty: merge the adjacent
  pair whose concatenation has the lowest vocab id — which is tiktoken's own
  algorithm, so encoding is exact by construction. (Merge-list recovery from
  ranks is provably lossy: the standard reconstruction mis-merges e.g.
  "newlines"; that is why the generator emits no merges.)

`tools/k3_tokenizer.py --ctest tests/test_tok_kimi` cross-checks the C
tokenizer against tiktoken; it matches on all 18 corpus cases (Chinese,
kanji-vs-kana, Korean, CRLF, contractions, emoji, mixed-script boundaries).

## Validation

`tools/k3_ref.py` is an independent numpy implementation of the full layer
stack (KDA recurrence, gated MLA, AttnRes, LatentMoE with MXFP4 dequant). With
injected inputs (`K3_X0`, bypassing the embedding) and f32 weights
(`K3_BITS=32`), the C engine matches it on real checkpoint weights across all
four layer types to **rel-L2 ≤ 2.2e-6** (float32 noise).

One property worth knowing: the AttnRes mix logits sit close together
(margins ~0.02), so the mix leverages *weight* noise strongly even though it
does not amplify *input* perturbations (a 1e-4 input perturbation stays 1e-4
through the stack). Load-time int8 produces a hidden-state drift growing to
~14 % over four layers on synthetic inputs — per-tensor quantization SNR is a
uniform ~1 %, so this is architectural sensitivity, not an outlier problem.
Judge quantization choices on real-text logits, not synthetic-vector norms.

## Environment variables

| var | default | meaning |
|---|---|---|
| `K3_BITS` | 4 | load-time bits for KDA/latent/shared/dense mats (4, 8, 32=f32) |
| `K3_MLA_BITS` | 8 | load-time bits for MLA projections |
| `K3_HEAD_BITS` | 8 | load-time bits for lm_head |
| `K3_EXPERT_GB` | 8 | routed-expert LRU budget |
| `K3_VK` | 1 | Vulkan tier when built with `make VK=1 kimi_k3` (0 = pure CPU) |
| `K3_VK_GB` | driver budget | VRAM cap for the Vulkan tier |
| `K3_VK_UP` | 8 | routed-expert uploads per step (fill-once tier) |
| `K3_DIRECT` | 1 | O_DIRECT expert reads (0 = buffered + WILLNEED) |
| `K3_IDOT` | 1 | int8-activation expert matmuls (0 = exact-float kernel) |
| `K3_PIPE` | 1 | overlap expert loads with compute (loader threads) |
| `K3_LOAD_THREADS` | 4 | loader threads for `K3_PIPE` |
| `K3_TOPP` | 0 | keep routed experts to cumulative weight p (0 = off) |
| `K3_CHUNK` | 32 | prefill chunk size (1 = token-at-a-time; forced 1 under `K3_TRACE`) |
| `K3_THINK` | 1 | chat mode: open the structural think channel (0 = response-only) |
| `K3_DIRS` | — | extra shard directories (multi-drive split, no duplication) |
| `K3_MAXT` | prompt+ngen | context capacity |
| `K3_LAYERS` | all | truncate the stack (validation) |
| `K3_TRACE` | — | dump f32 hidden state after every layer (validation) |
| `K3_X0` | — | inject input rows, bypass embedding (validation) |
| `COLI_TEMP` | 0 | 0 = greedy, else softmax temperature |

## Chunked prefill

Prefill processes the prompt in chunks of `K3_CHUNK` tokens, layer-major:
every dense matmul batches over the chunk (each weight matrix streams from
RAM once per chunk instead of once per token), the MoE loads each unique
expert of the chunk once (measured at C=32 on real text: 2.7x dedup —
neighbouring tokens share experts far more than QB-flat routing suggests —
9.6 instead of 25.8 GB/token), and the lm_head runs only on the chunk's last
token. Sequential state (KDA recurrence, MLA cache, AttnRes bookkeeping)
advances per token inside each layer in the original order, so chunked
results are **bit-identical** to token-at-a-time (verified: 125-position
teacher-forced logit streams at C=32 vs C=1 match exactly). Measured prefill:
~5.3 -> **2.0 s/token** at C=32. The KDA state-update sweeps are AVX2
(scalar fallback for head dims not divisible by 8).

## Chat, API, and Web

```sh
# standalone one-turn diagnostic
./kimi_k3 <model_dir> --chat "your question" [--system "..."] --ngen 300

# first-class multi-turn interfaces (config.json auto-detects Kimi K3)
COLI_MODEL=<model_dir> ./coli chat
COLI_MODEL=<model_dir> ./coli serve
COLI_MODEL=<model_dir> ./coli web
```

K3's chat format ("XTML", from the checkpoint's own `encoding_k3.py`) uses
only four special tokens — `<|open|>`, `<|close|>`, `<|sep|>`,
`<|end_of_msg|>` — around ordinary-text tag names and attributes:

```
<|open|>message role="user"<|sep|>TEXT<|close|>message<|sep|><|end_of_msg|>
<|open|>message role="assistant"<|sep|><|open|>think<|sep|>        <- generation prompt
```

The assistant's thinking is a *structural* channel and is preserved when an
OpenAI-compatible client sends prior `reasoning_content`; `enable_thinking=false`
opens `<response>` directly. The gateway does not flatten XTML into a string:
it sends length-framed messages to the C engine, which builds every structural
and ordinary-text segment at the tokenizer boundary required by K3's rank-BPE.
We checked our text handling against Moonshot's own: their `encoding_k3.py`
turns a conversation into the numbers the model actually reads, and ours
produced **identical numbers on all 77 test conversations** — multi-turn
histories with system, user and assistant messages, including non-ASCII text.
So the engine is not subtly mangling anything before the model sees it.

`coli chat` starts a private local server for Kimi and keeps the 2.8T model
loaded for the whole terminal session. `coli serve` exposes streaming and
non-streaming `/v1/chat/completions`; `coli web` uses that same API. Reasoning
is returned as `reasoning_content`, response text as `content`, and
`<|end_of_msg|>` remains the model-owned stop token. `STOP` and `CANCEL` are
honoured between generated tokens.

## Vulkan tier (`make VK=1 kimi_k3`)

The shared Vulkan backend (`backend_vulkan.c`) gained an **fmt=7 MXFP4**
decode path for K3's expert format — e2m1 nibbles with the ue8m0 exponents
expanded to f32 per-32-group scales at upload, so the QAT bytes are uploaded
exactly as stored and never re-encoded (kernel vs `matmul_mxfp4`: rel_l2
2.2e-07 on an RX 9070/RADV, 2.6e-07 on llvmpipe;
`tests/test_vk_mxfp4.c`). The engine keeps two residency classes on the
card, both with transparent CPU fallback and identical output:

- **shared experts**, uploaded once at init (int4/int8, the existing
  fmt-1/4 shaders): they run every token and are the largest always-on
  dense slice that fits VRAM (7.5 GB for all 92 MoE layers at int4);
- a **fill-once routed-expert tier** in fmt=7: experts enter from
  freshly-read RAM slots (`K3_VK_UP` per step) until the VRAM budget
  (`K3_VK_GB`) is reached. At decode, tier-resident experts skip **both**
  the 17.5 MB disk read and the CPU matmuls (one paired w1/w3 submit,
  SiTU-GLU on CPU, w2 down). Chunked prefill stays on the CPU-batched path
  and still warms the tier.

K3's Quantile-Balancing-flat routing caps what any cache tier can do — the
tier's value scales with how long the server lives (fill-once) and with the
measured short-term reuse (temporal locality), not with marginal expert
heat. `K3_VK=0` disables the tier at runtime.

## Current limitations

- Decode is single-token (no speculative decoding — K3 has no MTP head).
- Tool declarations/calls and image content are not exposed through the shared
  gateway yet; unsupported requests fail explicitly.
- CPU + optional Vulkan tier (no CUDA/Metal).
- The protocol, tokenizer, gateway, TUI, and Web client paths are locally
  testable without the 1.5 TB checkpoint. A release claim still requires one
  full-model multi-turn TUI/Web run on a host that owns the complete snapshot.
