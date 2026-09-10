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

## DSpark: drafting, and why accepting a draft is safe

The checkpoint carries an MTP draft head under `mtp.*`: three stages of the same block
the backbone uses, reading the attention input of layers 37, 38 and 39, and proposing
five tokens at a time. Their attention never compresses -- it is window-only, and the
window holds the **main** stream's keys, not the drafts' -- and they route over 128
experts of their own instead of 384, three per token, so a round of drafting costs
about a fifth of what one main token costs and produces up to five candidates.

Nothing is emitted on the draft head's word. The five drafts and the token that
produced them go back through the main model as **one forward**, which prices five
positions for one pass over the dense trunk and one pass over the union of their
experts. Each draft is kept while it agrees with what that forward says (greedy), or
with probability `p(draft)` under the sampler's own distribution when the temperature
is above zero, which for a deterministic drafter is exactly the speculative-sampling
rule; a rejected draft is banned from the resample, so the residual distribution is
the right one. What survives is what sequential decoding would have produced, position
by position.

Being exactly equal to sequential decoding is the whole load-bearing claim, and it is
not free. Three pieces of per-position state are addressed modulo something and so can
be clobbered by a draft several positions ahead:

- the **window ring**, where a draft at `p + 2` overwrites the key of `p - 6`, which a
  committed position still needs;
- the **compressor's group slots**, where a draft can overwrite the earlier half of a
  group that has not closed yet;
- the **published index-key slot**, which is per-step state in the vendor and has to
  read, for each row, what was published as of that row's own position.

Each row saves what it displaced, and a rejected row puts it back. `V41_SPEC_FORCE` in
the oracle drafts the reference's own next tokens (`1`), or corrupts the last of them
so a round is rejected part way (`2`), or keeps the head's own (`3`); all three have to
reproduce the reference token for token, and CI runs them.

Drafting is on whenever the checkpoint carries the head. It is not always worth it: a
round that is rejected has still read its stages' experts, so acceptance is measured
over a window and the drafts pause for 64 tokens when it falls under
`V41_DSPARK_MINACC`. `V41_DSPARK=0` turns the head off entirely, and it is then not
even loaded.

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
| `V41_TRACE` | unset | print a checksum of the tensors the reference prints too, for locating a divergence by diffing two columns. `2` follows the first row of a speculative step instead of the last, which is the row a sequential decode is comparable to. |
| `V41_DSPARK` | on when the checkpoint carries the head | `0` disables the draft head and does not load it. |
| `V41_DSPARK_MAX` | the checkpoint's `dspark_block_size` | how many of the drafted tokens are put in front of the main model. Fewer means a cheaper rejected round and a lower ceiling on the win. |
| `V41_DSPARK_MINACC` | 30 | percent of drafts that must be accepted over a window of 24 for drafting to continue; below it, drafts pause for 64 tokens. |
| `V41_SPEC_FORCE` | unset | oracle mode only: draft the reference's tokens (`1`), corrupt the last one (`2`), or use the head's own (`3`), to exercise the verification path on a fixture whose draft head is random. |

## How it is tested

`tools/dsv41_ref.py` is a torch reimplementation of the vendor's kernels on the CPU:
the vendor's own forward runs through tilelang GPU kernels and cannot be an oracle
here. `tools/make_dsv41_tiny.py` builds a container with the released structure at toy
dimensions and emits a reference; CI runs the engine against it at three cache
capacities and holds the vision tower to its own reference rows.

Writing both sides found three defects that a single implementation would have kept:
a hyper-connection mix summed over the wrong axis, hash multipliers rounded by a JSON
reader that stores numbers as doubles, and an index list that a non-source layer
inherited from uninitialized memory instead of from its source. Holding a speculative
step to the same reference found two more, both in what a rejected draft leaves
behind: a window slot and a compressor group slot that a committed position still
needed.

The gateway is tested against the checkpoint's own encoding fixtures
(`tests/test_openai_tools_v41_e2e.py`), and an image is followed all the way from an
OpenAI request to a different answer in `tests/test_dsv41_image_serve.py`.

## Where this engine is not the vendor

The vendor quantizes some activations on the fly, in kernels that only exist for the
GPU: the KV cache is rounded to fp8 before it is stored (`act_quant(..., inplace=True)`
in `Attention.forward`), and the indexer's queries and keys, and the compressor's
latents, are rounded to fp4. This engine keeps all of them in fp32 -- more precise, and
therefore not bit-identical to a GPU run of the same weights. Nothing downstream is an
argmax over a near-tie by construction, but it is a difference, and it is the reason a
divergence against a GPU reference would not necessarily be a bug here.
