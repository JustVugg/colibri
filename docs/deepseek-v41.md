# DeepSeek V4.1 Flash

552B parameters, 510 GB on disk, and the shape of it is why this engine exists.

| part | on disk | how it is read |
|---|---|---|
| **engram** (2 tables, layers 1 and 14) | **203 GB** | `[384,006,168 x 256]` fp8 plus ue8m0 scales: **264 bytes per row**, looked up by n-gram hash |
| **routed experts** | **289 GB** | 15,360 experts of 18.8 MB, already fp4; 6 of 384 per layer |
| dense, embeddings, vision | ~18 GB | fp8 with 32x32 block scales, bf16 elsewhere; resident |

Two numbers decide whether this model is servable on a machine you own:

- **4.5 GB of experts per token** (6 routed x 40 layers x 18.8 MB). GLM-5.2 reads
  12.7 GB per token, so V4.1 Flash is lighter per token than a model less than half
  its size.
- **~13 KB of engram per token**. The n-gram memory is 40% of the checkpoint and
  costs a few dozen random reads of 264 bytes: at most `(max_ngram_size - 1) x
  n_heads` rows per table per token, 48 rows for the released layout. Streaming it
  is not a compromise, it is the only sane way to hold it -- 203 GB does not fit in
  anyone's VRAM.

## Running it

No conversion. The released checkpoint is already fp8 (dense) and fp4 (experts), and
colibri reads both formats natively -- the fp4 expert layout is byte-identical to the
mxfp4 it already reads for Kimi K3.

```sh
hf download deepseek-ai/DeepSeek-V4.1-Flash --local-dir ~/Models/DeepSeek-V4.1-Flash
python3 c/tools/prepare_dsv41.py --model ~/Models/DeepSeek-V4.1-Flash   # ~1 MB sidecar, once
make -C c deepseek_v41
coli chat --model ~/Models/DeepSeek-V4.1-Flash
```

`prepare_dsv41.py` writes the three tables the engine cannot derive at load time and
that never change for a checkpoint: the compressed-token map (the tokenizer's
normalizer collapses ids that hash alike), the bucket primes (one prime-sized range
per n-gram size and head), and the hash multipliers (numpy's PCG64, seeded per layer).
They ship as `dsv41_engram.json` beside `config.json`. The multipliers are stored as
decimal **strings**: they run past a double's 53-bit mantissa, and a rounded
multiplier would silently hash every n-gram into a different -- and perfectly
valid-looking -- row.

## What the engine implements

Each mechanism mirrors a named piece of the vendor's `inference/model.py`:

- **hyper-connections**: the residual stream is `hc_mult` parallel copies, mixed
  through a doubly stochastic matrix (Sinkhorn). Shared with DeepSeek V4 and
  GLM-5.3-Flash through `hyper_connections.h`.
- **sliding window**: every layer attends a ring of `window_size` raw KV, MQA --
  one KV vector per position for all 64 heads, which is why the KV cache is small.
- **compressed KV**: `kv_source_layers` pool `compress_ratio` tokens into one latent
  with a softmax gate and publish it; the layers between them read that cache.
- **DSA indexer**: `index_source_layers` score compressed positions and keep
  `index_topk` of them, two-level when a candidate source is configured.
- **engram**: n-gram lookups gated into the residual stream by how well the looked-up
  key agrees with it.
- **MoE**: `sqrtsoftplus` scores with the noaux_tc bias, which picks experts but does
  not scale them, plus one shared expert every token pays for.
- **vision**: a 32-layer ViT with 2D split-half RoPE and a 3x3 aligner, resident.

## The index-key slot, and why the default is the odd one

`model.py` keeps the published index-key cache in one module-level slot and
republishes it only when a layer **completes** a compression group:

```python
if self.owns_k and latent is not None:
    ...
    shared_attn.index_k = self.k_cache
```

On a decode step where a ratio-2 layer's group is still filling, that layer therefore
scores its queries against whichever cache was published last -- in the released
config, layer 20's, whose rows are ratio-1 latents and mean something else. In the
released layout this happens to layers 2, 8 and 14 on every other decode step.

This engine reproduces that, because it is what DeepSeek ships and what every
published number for this model was measured with. `V41_INDEX_OWNER=1` selects the
reading the architecture implies instead -- each layer against its own owner's keys.
That is a different model, not a bug fix, and the CI asserts the two disagree so the
default cannot be "tidied" by accident.

## Environment

| variable | default | what it does |
|---|---|---|
| `V41_ENGRAM_ROWS` | 65536 | rows of engram cache per table. The traffic is Zipfian: common 2-grams repeat constantly, so a small cache absorbs most of it. 65536 rows is 64 MB per table on the released head_dim. |
| `V41_INDEX_OWNER` | unset | each layer scores against its own owner's index keys (see above). Changes the model's behaviour. |
| `V41_MAX_IMAGE_TOKENS` | the checkpoint's `max_image_tokens` | a ceiling on what one image costs in prompt tokens. |
| `V41_TRACE` | unset | print a checksum of the tensors the reference prints too, for locating a divergence by diffing two columns. |

## How it is tested

`tools/dsv41_ref.py` is a torch reimplementation of the vendor's kernels on the CPU:
the vendor's own forward runs through tilelang GPU kernels and cannot be an oracle
here. `tools/make_dsv41_tiny.py` builds a container with the released structure at toy
dimensions and emits a reference; CI runs the engine against it at three cache
capacities and holds the vision tower to its own reference rows.

Writing both sides found three defects that a single implementation would have kept:
a hyper-connection mix summed over the wrong axis, hash multipliers rounded by a JSON
reader that stores numbers as doubles, and an index list that a non-source layer
inherited from uninitialized memory instead of from its source.
