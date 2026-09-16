# Qwen3 (dense) on colibri

`c/qwen3.c` runs the **dense** Qwen3 checkpoints — `Qwen3ForCausalLM`,
`model_type: qwen3` — of which [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
is the reference. Not to be confused with `c/qwen36.c` (Qwen3.6-35B-A3B, MoE +
DeltaNet) or `c/qwen38.c` (Qwen3.8-Flash-Next): those are separate
architectures with their own engines, and the family registry matches
`model_type` exactly so none of the three can claim another's checkpoint.

Architecture: GQA (32 query heads / 8 KV heads, head_dim 128), per-head
RMSNorm on q and k, SwiGLU MLP, full RoPE on every head dimension, untied
`lm_head`. No experts, no sliding window, no attention output gate. Every
weight is resident — there is nothing to stream and nothing to cache, which is
why `coli plan` declines this family rather than inventing an expert budget.

One detail worth naming because it is silently wrong if you assume otherwise:
Qwen3's RMSNorm applies the weight **plain** (`x * rsqrt(mean(x²)+eps) * w`),
where Qwen3.6 applies `(1.0 + weight)`. Reusing the 3.6 kernel here produces
fluent, confident, incorrect output.

## Quickstart

Convert a checkpoint into a colibri container, then run it:

```sh
# int4-gs64 for the big matrices (default), ~4.5 GB from an 8B bf16 source
python3 c/tools/convert_qwen3_dense.py --model ~/Models/src-qwen3-8b \
        --out ~/Models/qwen3_8b_i4 --bits 4

make -C c qwen3
COLI_MODEL=~/Models/qwen3_8b_i4 ./c/coli chat
```

`--bits 0` instead produces an f16 passthrough container (~16 GB): same tensor
names, same layout as the source, no quantization anywhere. Use it when you
need to compare the engine against the HF reference token for token; use
`--bits 4` for anything else.

Direct invocation without the gateway:

```sh
SNAP=~/Models/qwen3_8b_i4 TOK=~/Models/qwen3_8b_i4/tokenizer.json \
N_NEW=200 ./c/qwen3 0 4 prompt.txt
```

The two positional arguments (`cap`, `bits`) are accepted for CLI-shape
compatibility with the MoE engines and **ignored**: a dense model has no expert
cache to cap, and the container's `qwen3_meta.json` states its own bit width.
The engine prints that it ignores them rather than letting you believe a
number took effect.

## Container format

A directory of safetensors shards plus `config.json`, `tokenizer.json` and
`qwen3_meta.json`. Tensor names are kept **exactly** as HF writes them; the
container differs from the source only in the dtype/layout of the large
matrices and in the extra meta file.

```
model-globals.safetensors     embed_tokens, lm_head, model.norm  (f16)
model-00000.safetensors ...   one shard per layer
config.json                   copied verbatim from the source
tokenizer.json                copied when present
qwen3_meta.json               dimensions, rope_theta, bits, gs
```

At `--bits 4`, seven matrices per layer — `self_attn.{q,k,v,o}_proj.weight`,
`mlp.{gate,up,down}_proj.weight` — become:

| tensor | dtype | shape | meaning |
|---|---|---|---|
| `<name>` | `U8` | `[O, ceil(I/2)]` | nibble-packed, **low nibble = even column**, stored value `q + 8` (0..15) |
| `<name>.qs` | `F32` | `[O, ceil(I/64)]` | one scale per 64 **input** elements |

Dequantization is `(nibble - 8) * scale`, matching `matmul_i4_grouped` in
`c/quant.h` (fmt=4, `int4-grouped`, `docs/FORMATS.md`).

Everything else — embeddings, `lm_head`, all norms including `q_norm`/`k_norm`
— stays f16 regardless of `--bits`. Embeddings and the output head enter every
single token, and int4 there measurably breaks coherence; this is the same
policy `convert_inkling_dense_int4.py` applies for the same reason.

Expect ~10–11% per-weight L2 error on the int4 matrices. That is normal for
gs64 and is why only the f16 container is held to token-exactness.

## `rope_theta` is read from two places, and defaulted from neither

transformers <5 writes `rope_theta` at the top level of `config.json`;
transformers >=5 nests it under `rope_parameters`. Both the converter and the
engine check both locations, and **refuse** when neither has it:

```
[cfg] rope_theta missing from both config.json and qwen3_meta.json -- refusing
```

The refusal is deliberate. During development a plausible default (1e6) was
substituted for a fixture's actual 1e4, and nothing failed — the engine loaded,
ran at full speed, and produced wrong tokens from generated position 8 onward.
A RoPE base is not something to guess at.

## Context

`Q3_MAXT` is how `coli --ctx` reaches this engine. It only ever **lowers** the
ceiling that `max_position_embeddings` sets (40960 on Qwen3-8B); raising it
would push RoPE positions outside the trained range, so the container keeps the
last word. The hard build-time ceiling is 262144.

The KV cache is f32 and sized `n_layers * ctx * kv_heads * head_dim * 2 * 4`
bytes — on Qwen3-8B that is 288 KB per token, so 8k of context costs ~2.3 GB on
top of the weights.

## Validation

```sh
make -C c qwen3-tiny-check
```

builds a tiny model with the same dense layout at toy dimensions
(hidden 64, 4 layers, 4 Q / 2 KV heads, head_dim 16, inter 128, vocab 320),
generates a greedy reference from HF, converts it both ways, and requires:

- `--bits 0`: **token-exact** against the HF reference, at two prefill caps;
- `--bits 4`: loads and generates (no exactness contract — see the L2 error
  above).

`python3 c/tools/convert_qwen3_dense.py --selftest` checks the int4 pack/unpack
round-trip on its own, including the odd-input-width tail. CI runs both, plus
the same containers under ASan + UBSan and a container with its RoPE base
removed, which must be refused.
