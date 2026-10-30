# Mistral-Large-4 (`c/colibri.c`)

[Mistral-Large-4-1T-A52B](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B): 1T parameters,
52B active, 72 layers, hidden 6144, 64 heads, 128 routed experts (4 per token) and one shared,
the first layer dense, vocabulary 204800. Its text model is GLM-5.2's DSA attention with six
changes, so it runs on the GLM-5.2 engine (`colibri.c`, the `glm` family in `coli`).

```
./coli import ~/models/Mistral-Large-4-1T-A52B-UD-IQ1_S-00001-of-00006.gguf   # first part of a split model
./coli chat --model ~/models/Mistral-Large-4-1T-A52B-UD-IQ1_S.colibri
```

## What differs from GLM-5.2

- **No query LoRA.** The query comes straight from the layer's input (`q_proj`, the GGUF's `attn_q`).
- **NoPE layers.** Every fourth layer (`no_rope_layers` 0, the GGUF's `rope_pattern` false) has no
  rotary part: the query head's 128 dims are all unrotated and the latent has no rotary key.
- **Shared KV.** Layers 39, 43, ..., 71 read layer 35's latent (`kv_source_layers`, the GGUF's
  `kv_source_map`) and have no `kv_a` of their own; their `kv_b` is their own. The engine copies the
  source's rows into the reader's cache as it writes them.
- **SSMax.** Every head's query is multiplied by `s * log(position + 1) + b`, its own `s` and `b`, in
  the attention and in the indexer.
- **The indexer.** `q = wq_b(LayerNorm(x))` with its SSMax, `k = LayerNorm(wk(x))`, both rotated on
  their first 64 dims on RoPE layers; the scores `relu(q . k / sqrt(128))` weighted by
  `weights_proj(x) / sqrt(32)`, the top 2048 keys kept. A "shared" layer reads the selection of the
  last full layer before it.
- **RoPE and the rest.** Interleaved pairs; YaRN's frequencies (factor 2048 over 512, attention
  factor 1) divided by a learned factor per pair (the GGUF's `rope_freqs` is `exp(-alpha)` of the
  checkpoint's `lerope.alpha`); a softmax router renormalised over its top 4; the logits capped at
  30 (`30 * tanh(z / 30)`).

The tokenizer is Tekken (`tok.h`): the o200k pattern without its English contractions and digits
in groups of three from the right. On the release's `tokenizer.json` it splits 3017 test strings
as transformers does (`tests/test_tok_tekken.c` holds a tiny fixture of it).

The gateway writes the chat as the release's `chat_template.jinja` does, byte for byte
(`tests/test_mistral_large4_chat.py`, `COLI_ML4_TEMPLATE` pointing at the template runs that check):
`[MODEL_SETTINGS]` with the reasoning effort, `[INST]` turns, tools and their results. The model's
reasoning comes back as `[THINK]...[/THINK]`, which the gateway gives as `reasoning_content`, and
its calls as `[TOOL_CALLS]name[ARGS]{json}`, which it gives as `tool_calls`.

## From a GGUF

`coli import` (`tools/gguf_import.py`) takes any of unsloth's eleven files
(`unsloth/Mistral-Large-4-1T-A52B-GGUF`), read in place. Their types, from their headers:

| file | attention q, output, indexer q, head | routed experts | the rest |
|---|---|---|---|
| BF16 | BF16 | BF16 | BF16 |
| Q8_0 | Q8_0 | Q8_0 | Q8_0 |
| UD-Q6_K_XL | Q8_0 | Q6_K, Q8_0 | Q8_0 |
| UD-Q5_K_XL | Q8_0 | Q5_K, Q6_K | Q8_0 |
| UD-Q4_K_XL | Q8_0 | Q4_K, Q5_K | Q8_0 |
| UD-Q3_K_XL | Q8_0 | Q3_K, Q4_K, IQ4_XS | Q8_0 (embeddings Q6_K) |
| UD-Q2_K_XL | Q8_0 | IQ2_S, IQ3_XXS, IQ3_S | Q8_0 (embeddings Q6_K) |
| UD-IQ3_XXS | Q8_0 | IQ3_XXS, IQ3_S, IQ4_XS | Q8_0 (embeddings Q6_K) |
| UD-IQ2_XXS | Q6_K | IQ2_XXS, IQ2_S, IQ2_XS, IQ3_XXS | Q8_0 (embeddings Q6_K) |
| UD-IQ1_M | Q6_K | IQ1_S, IQ2_XXS, IQ2_XS, IQ3_XXS | Q8_0 (embeddings Q6_K) |
| UD-IQ1_S | Q6_K | IQ1_S, IQ2_XXS | Q8_0 (embeddings Q6_K) |

Norms, SSMax, the router, the indexer's `weights_proj` and `rope_freqs` are F32 in all of them.

`tools/make_mistral_large4_tiny.py --quant NAME` writes a tiny model in one file's types, layer by
layer as it has them, and the reference reads what those blocks decode to. All eleven, measured:

- the float path (`GQ_EXACT=1`): 16/16 tokens greedy and 28/28 positions of teacher forcing;
- the CPU's integer kernels against a reference that rounds their inputs to int8 per 32 as they do
  (`--act-q8`): the logits equal on the first positions with every type, then a code rounding the
  other way from a sum in another order moves later ones (the tiny model amplifies that);
- Vulkan's expert tier (the experts' blocks on the device, the dense part on the CPU) against the
  reference that rounds only the dense inputs: the same.

CI runs six of the eleven, which hold all fourteen types.

The real files: `tools/gguf_header.py` fetched the eleven headers, the importer wrote their configs
and the engine maps every tensor of them. UD-IQ1_S (233 GB) runs: on an RTX PRO 4000 (24 GB) with
62 GB of RAM and two NVMe drives, the 17.9 GB dense part resident, the experts from the disk,

| | decode, 48 tokens | prompt, 240 tokens |
|---|---|---|
| the CPU | 0.52 tok/s | |
| Vulkan, the experts' tier on the card, the attention on the CPU | 0.76-0.88 tok/s | 79.8 s |
| Vulkan, the dense chain too (all 72 layers, 16.9 GiB) | 0.81 tok/s | 52.7 s |

Decode stays where the disk puts it: 0.73-0.81 tok/s whether the chain holds 0, 24, 48 or 72
layers (`COLI_VK_CHAIN_LAYERS`; the experts' tier gets what the chain leaves, 22.3 GiB to 5.5).

The CPU half of the expert work was slower than it had to be on this box's i5-13500: the engine's
OpenMP team of six P-core threads sat on CPUs 0-5, three cores, until the re-exec set
`OMP_PLACES=cores` (see [tuning](tuning.md)). With the chain, 24 tokens of decode and a 240-token
prompt, before and after alternated:

| | before | after |
|---|---|---|
| decode, the default team of six | 0.59, 0.61 tok/s | 0.76, 0.80 tok/s |
| decode, `OMP_NUM_THREADS=14` (every physical core) | | 0.89, 0.91 tok/s |
| prompt, 240 tokens | 47.6, 48.3 s | 35.6, 36.0 s |

and "The capital of France is Paris. The capital of Italy is" goes on with " Rome. The capital of
Germany is Berlin. The capital of Spain is Madrid. The capital of Portugal is Lisbon." on both.
Through the gateway the chat template, the `[THINK]` reasoning (as `reasoning_content`) and the
answer come back as the release writes them.

On an NVIDIA card a GGUF of this model runs on Vulkan, as GLM-5.2's does (a build with both
backends switches to it; the CUDA backend has no ggml block types).

## Pictures

The vision tower is the GGUF repository's `mmproj-BF16.gguf` (or `mmproj-F16.gguf`): Pixtral, 30
layers of width 2048, 16 heads, patches of 14 with an axial RoPE, a SiLU-gated MLP; the projector
puts each 2 x 2 block of patches side by side (8192 to 2048), then 2048 to 6144, GELU, 6144 to
6144 (`mistral_vision.h`, transcribed from transformers' `modeling_pixtral.py` and
`modeling_mistral3.py`). `coli import` links one it finds next to the model (`--mmproj` names one,
`--no-mmproj` leaves it out) and writes its geometry to `preprocessor_config.json`.

```
coli import Mistral-Large-4-1T-A52B-UD-IQ1_S-00001-of-00006.gguf --mmproj mmproj-BF16.gguf
coli serve --model Mistral-Large-4-1T-A52B-UD-IQ1_S.colibri
```

The gateway cuts a picture as the release's processor does (`tools/mistral_image.py`: scaled down
under 1540 pixels a side, then to whole tokens of 28 x 28 pixels, bicubic, the CLIP mean and std;
Pillow and numpy needed) and writes a row of `[IMG]` a token, each row closed by `[IMG_BREAK]`, the
last by `[IMG_END]`. The engine's rows replace those tokens' embeddings; the text before the
picture keeps its cached prefix, the picture's rows are never reused. The tower is 3.4 GB beside the
model, counted in the RAM plan; it is loaded in serve mode only (`ML4_VISION=0` skips it).

An mmproj stores the q and k rows of each head in interleaved rotary order (its converter's
permutation): the engine rotates pairs (2j, 2j + 1). Rotating halves, as the release's code does on
its own weights, kept a test picture's colours and lost its shapes. Checked:

- `tools/mistral_vision_ref.py`, a numpy transcription that undoes the permutation and rotates
  halves: the engine's rows within 1.0e-5 of their RMS on the real mmproj (`ML4_VIS_DUMP` writes
  them from a served picture);
- `make mistral-vision-check` (CI): a tiny tower in the published layout, within 1e-4, and failing
  with the halves;
- UD-IQ1_S, a red disc, a blue square and a green triangle: "The picture contains a red circle, a
  blue square, and a green triangle."

The tower runs on the CPU: 8.0 s for a picture of 256 tokens on an i5-13500 (6 P + 8 E cores), 96 s for 1024 (its
attention is quadratic), so the gateway holds a picture to 256 tokens (`ML4_MAX_IMAGE_TOKENS`).

## From the FP8 release

`tools/convert_fp8_to_int4.py` reads Mistral's own release: `consolidated-*.safetensors` in
Mistral's names (`layers.N.attention.wq`, `experts.E.w1/w2/w3`, `shared_experts`, `gate`,
`ssmax_scaling`, `lerope.alpha`), FP8 with a `qscale_weight` per 128 x 128 block.

```
python3 tools/convert_fp8_to_int4.py --repo mistralai/Mistral-Large-4-1T-A52B --outdir ~/models/mistral-large4-colibri
```

It downloads a shard, converts it and deletes it, so the release (1060 GB) never sits on the disk
whole. The default is int4 with a scale per 64 for every matrix, int8 for the embeddings and the
head (`--ebits`, `--xbits`, `--io-bits` change them). Checked on the release's own bytes against
the Q8_0 GGUF: `qscale_weight` multiplies (dividing is off by 1e7), `w1` is the gate and `w3` the up.
The converter writes the engine's flat `config.json` (the release's kept as `config.hf.json`).

A container runs on CUDA too (the experts and the dense matrices on the card, `CUDA_DENSE=1`). The
tiny model's int4 container gives the same tokens on the CPU, CUDA and Vulkan, the logits within
about 1% of each other.

## Not done yet

- On Vulkan the dense chain takes the attention (the query without a latent, NoPE and shared-KV
  layers, SSMax, the indexer) but not yet the KV split past the card's memory, a second device or
  several conversations' rows at once: those run on the CPU. On CUDA (our container) the attention
  runs on the CPU.
- The vision tower runs on the CPU only.
- The DSpark draft head (`dspark-*.gguf`) is not read.
