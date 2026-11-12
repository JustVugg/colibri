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

Preparing the engram sidecar needs `transformers`, `tokenizers`, `numpy`, and
`sympy` in the Python environment used for `prepare_dsv41.py`. For example, on
Linux or macOS, create a virtual environment and install them once:

```sh
python3 -m venv ~/.venvs/dsv41
~/.venvs/dsv41/bin/python -m pip install transformers tokenizers numpy sympy
```

Use `~/.venvs/dsv41/bin/python` in place of `python3` in the preparation commands
below. The model engine itself does not require those Python packages.

```sh
hf download deepseek-ai/DeepSeek-V4.1-Flash --local-dir ~/Models/DeepSeek-V4.1-Flash
python3 c/tools/prepare_dsv41.py --model ~/Models/DeepSeek-V4.1-Flash   # ~1 MB sidecar, once
make -C c deepseek_v41
coli chat --model ~/Models/DeepSeek-V4.1-Flash
```

**DeepSeek-V4.1-Flash-1017** (October 17) is the same model with new weights: the same
`config.json`, the same 96,085 tensors in the same formats, the same tokenizer and prompt
encoding, byte for byte. It runs the same way:

```sh
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --local-dir ~/Models/DeepSeek-V4.1-Flash-1017
python3 c/tools/prepare_dsv41.py --model ~/Models/DeepSeek-V4.1-Flash-1017
coli chat --model ~/Models/DeepSeek-V4.1-Flash-1017
```

It ships no `chat_template.jinja`, and nothing here needs one: the gateway writes V4.1's
prompts itself (`render_chat_dsv41` in `openai_server.py`).

`prepare_dsv41.py` writes the three tables the engine cannot derive at load time and
that never change for a checkpoint: the compressed-token map (the tokenizer's
normalizer collapses ids that hash alike), the bucket primes (one prime-sized range
per n-gram size and head), and the hash multipliers (numpy's PCG64, seeded per layer).
They ship as `dsv41_engram.json` beside `config.json`. The multipliers are stored as
decimal **strings**: they run past a double's 53-bit mantissa, and a rounded
multiplier would silently hash every n-gram into a different -- and perfectly
valid-looking -- row.

When `--cap` is omitted, `coli chat`, `coli serve`, and `coli web` size the expert
cache from the resource plan. `--ram 120` (or `RAM_GB=120`) supplies its RAM budget;
without either, the planner uses available memory. Dense weights, context state,
and runtime reserves are subtracted before choosing slots per layer. `--cap N`
still selects an explicit slot count, and an existing `--auto-tier` plan or measured
profile retains precedence over this fallback. The startup line reports the chosen
cap. The budget is a ceiling, not a promise to fill RSS: cache slots fill on demand
and engram tables continue streaming from disk.

The standalone `deepseek_v41` binary still takes its cap as its first argument;
`RAM_GB` planning happens in the Python gateway.

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
so a round is rejected part way (`2`), or keeps the head's own (`3`), or corrupts the
first (`4`: every drafted row is rolled back, the common case at low acceptance) or a
different one each round (`5`); all of them have to reproduce the reference token for
token, and CI runs them.

Drafting is on, and the default is a measurement. Three runs of the same
24-token turn on a 16-thread CPU server holding 68% of the experts, each from a
dropped page cache:

| | forwards | wall | tok/s |
|---|---|---|---|
| drafts off | 24 | 116.6 s | 0.206 |
| drafts on, guard at 60% | 16 | **99.3 s** | **0.242** |
| drafts on, guard disabled | 9 | 111.5 s | 0.215 |

Two things in that table are worth more than the 17% it shows.

The first is that **stopping pays as much as starting**. Nine forwards is fewer
than sixteen and slower: the early rounds push six positions through a cold
cache and win, the later ones pay six times the per-row attention for a cache
that is now warm and would have served single tokens cheaply. The acceptance
guard fires at 53% and turns 111 s into 99. It is not really measuring
acceptance, it is measuring the moment batching stops being free, and
acceptance is the signal that moves with it.

The second is that the first comparison I ran said the opposite, that drafting
cost 32%. It was wrong, because the two runs had not started from the same
cache state. On an engine that reads its weights from disk, a warm page cache
is worth more than any optimization in this file. Cold, or it is not a number.

`V41_DSPARK=0` turns the head off entirely and it is then not even loaded;
`V41_DSPARK_MINACC` moves the guard, whose default of 60% is where these runs
put it.

How many of a round's drafts the verify carries comes from the head itself. Each stage
has a confidence projection, and the sigmoid of its score is the acceptance the head
predicts for that draft. `spec_draft.h`'s gate (`spec_gate_pick_p`) maps each prediction
through the acceptance it measured for drafts given that prediction, and picks the depth
that yields the most tokens per unit of what a verify of that many rows costs here (the
gate of [speculative.md](speculative.md#the-gate)); a round whose drafts it does not
believe in takes a plain step. This is DeepSeek's confidence-scheduled verification. The
acceptance guard counts every draft the stages wrote, verified or not: the stages read
their experts for each one, and a draft the gate left unverified bought nothing.
`V41_DSPARK_GATE=0` verifies every draft, as before.

On the RTX PRO 4000 above with UD-IQ1_S and the Q8_0 head (named next to a GGUF, not
linked by default: [gguf.md](gguf.md)), the Vulkan chain, 128 tokens, two rounds of
three prompts:

| | decode tok/s |
|---|---|
| no head (`V41_DSPARK=0`) | 4.68-4.99 |
| every draft verified (`V41_DSPARK_GATE=0`) | 3.64-4.28 |
| the gate | 4.10-4.41 |

On a 1-bit model the head agrees with the backbone too rarely to pay for its stages
(1 of 20 drafts accepted in a turn), and the gate takes back most of what verifying them
costs. Before the guard counted the drafts the gate left unverified, the same gate
measured 2.81-3.64: the drafts it trusted were accepted, so the guard never paused, and
every round paid the stages for drafts it then declined.

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

## Streaming the experts

Six of 384 experts are routed per layer per position, and each is 18.8 MB of fp4
weights plus ue8m0 scales. A turn of 26 prompt tokens and 24 generated ones moves
88.5 GB through the expert cache on the released checkpoint, so how fast those bytes
arrive is not a detail of the engine, it is most of the engine.

The reads go out together. A step's routed experts are all resolved to cache slots
first -- hits stamped, misses given a reserved victim -- and only then are the missing
tensors read, `V41_READ_DEPTH` of them at a time. Reserving before reading is what
makes the batch possible: a reserved victim carries the newest clock and a cleared id,
so a later reservation in the same step neither evicts it again nor mistakes it for a
hit. The arithmetic is untouched -- the experts are still accumulated in rank order --
and a cache too small to hold one step's experts (`cap < topk`) keeps the old
one-at-a-time path, because there the slots genuinely cannot all be live at once.

Measured on the released checkpoint, cold (page cache dropped before every run), same
prompt, same seed, on a striped pair of Samsung PM9A1:

| read depth | expert I/O | rate | wall | tok/s |
|---|---|---|---|---|
| 1 | 35.8 s | 2.47 GB/s | 67.0 s | 0.358 |
| 4 | 15.0 s | 5.91 GB/s | 45.0 s | 0.533 |
| 8 (default) | 12.9 / 13.0 s | 6.84 / 6.81 GB/s | 43.1 / 42.4 s | 0.557 / 0.566 |
| 12 | 13.1 s | 6.77 GB/s | 43.4 s | 0.554 |
| 16 | 13.5 s | 6.57 GB/s | 47.5 s | 0.506 |

Every row reads the same 88.5 GB in the same 4709 misses and emits the same 24 tokens;
only the depth changes. Past eight the device stops answering faster and the reader
threads start taking cores away from the matmuls, which is why the wall turns back up
while the rate barely moves. The depth-8 row is two independent cold runs, to show how
much of the wall is run-to-run noise: the I/O figures repeat to within half a percent,
the compute figures do not.

Prefill is where most of it goes: 52.3 GB of the 87.4, because each of the 26 prompt
positions routes independently and a 49-slot cache cannot hold what 26 positions ask
of one layer.

### Where the bytes come from, and whether they touch the page cache

Two things move expert bytes that the V4 and GLM engines already had and this one did
not, and both are about the same fact: an expert is read once and never wanted again.

**A second drive.** `COLI_MODEL_MIRROR=<dir>[;<dir>...]` registers read-only copies of
the checkpoint on other drives; experts are split across the primary and every replica,
each drive answering for a deterministic subset. A copy is accepted only when its size
and safetensors header are byte-identical to the primary's, so the data offsets match
by construction and any expert can be served by any replica -- and a copy that is
*not* identical is refused rather than trusted, which is the whole reason the check is
a header compare and not a filename. Partial mirrors are legal: a smaller SSD holding
only the routed-expert shards is the case this exists for, and a shard a replica does
not carry simply stays on the primary.

The split follows `COLI_DISK_WEIGHTS`, or is measured at startup with this engine's own
access pattern. Splitting is the point and also the risk: two independent drives
answer in parallel, but one drive serving two copies of itself answers no faster than
one drive, so the split is a measurement rather than an assumption.

The hash is over the flat expert index. `deepseek_v4.c` records why that matters: an
XOR of layer and expert spread the whole 43x256 grid evenly and still clustered the
dozen experts a decode step actually touches onto one replica. Multiplying the flat
index by the golden-ratio constant and taking bits 16..23 inherits the uniformity of
that index, so a hot subset splits like a cold one.

**Not through the page cache.** The reader already tells the kernel each expert is not
wanted again (`POSIX_FADV_DONTNEED` after the read), which is the honest admission that
a buffered read of it pays for the bytes twice -- once into the cache, once out -- in
order to leave the cache as cold as it was. `V41_DIRECT=1`, the default, reads them
with `O_DIRECT` instead.

There is a second, cheaper way to put the container on more than one drive, and for a
510 GB checkpoint it is the one most people can actually build.
`COLI_MODEL_DIRS=<dir>` names directories holding **distinct** shards -- no
duplication, combined capacity, each shard read from the one drive that has it -- so
the same container can live on two 256 GB drives instead of one 512 GB drive. It is
the same split the GLM, V4 and Kimi K3 engines already accept. It composes with the
mirror: a mirror directory may copy any subset of the split's shards.

The two mechanisms answer different questions. `COLI_MODEL_DIRS` asks whether the
bytes can be *stored* across drives; `COLI_MODEL_MIRROR` asks whether they can be
*read* from more than one place at once. With the CPU side in order -- the OpenMP team
sized, which `c/coli` already does at launch -- the second one is worth a fifth of the
expert disk phase and the engine reaches the device's own ceiling; the measurements,
and the regime in which neither is worth anything, are in
[the experiment record](experiments/dsv41-expert-io-2026-09-14.md).

### The cheapest resident tier there is

The expert cache has a floor that no cache size can move: a turn's *first* touch of
each expert it routes. On the released checkpoint a 16-token turn touches 3,371
distinct experts, and a cache of about 96 slots per layer already sees no capacity
misses at all -- every miss left is a first read, and only having the bytes resident
*before the turn starts* removes it.

The engine cannot ask for that, but a filesystem can. Point a partial mirror at RAM:

```sh
# the 46 non-engram shards carry every routed expert and the dense weights
mkdir -p /dev/shm/dsv41_ram
cd <container> && ls model-*.safetensors | grep -v -E -- '-0004[78]-' \
    | xargs cp -t /dev/shm/dsv41_ram/

COLI_MODEL_MIRROR=/dev/shm/dsv41_ram COLI_DISK_WEIGHTS=1,1000 \
    SNAP=<container> ./c/deepseek_v41 8 ref.json
```

Measured on the released checkpoint, 16-token turn, cold cache, `cap=8`: expert disk
16.05 s -> **4.15 s** (6.19 -> 23.89 GB/s) and the turn 37.3 s -> **25.5 s**, -31.7%,
token-exact. That is more than a cache four times its size buys -- `cap=384`, which
holds all 289 GB of experts, still reads 63 GB and spends 11.3 s on disk -- because
caching during a turn cannot remove a first read and residency before it can. It costs
287 GiB of RAM for the duration, which is the whole trade; 1,000 is not magic, any
weight that sends nearly every expert to the RAM copy will do.

That needs the destination and the file offset to be block-aligned, and a safetensors
range starts wherever the writer put it. The obvious fix is to bounce each transfer
through an aligned scratch buffer, and it is the wrong one here: an extra copy of every
expert byte, over a turn that moves 88.5 GB, is most of what the change was for. Each
slot instead keeps its six buffers in page-aligned windows with `ST_DIRECT_ALIGN` of
scratch in front, and `st.h`'s `st_read_range_rep` starts the read at the enclosing
block boundary, carries the block-aligned bulk with one `O_DIRECT` pread and the
sub-block tail with one short buffered pread, and lands the payload exactly where the
slot wants it. Nothing is copied afterwards.

The alignment is verified, not assumed. A destination that was not set up that way --
or a platform without `O_DIRECT`, or a replica whose twin failed to open -- takes the
buffered path, which reads exactly the same bytes. `V41_DIRECT=0` forces it, both as
the other arm of the A/B and as the escape hatch on a device where direct I/O turns
out to be slower.

## Blocks of positions, not one position at a time

Everything above the experts used to run one position at a time, which meant a block
of positions made one full pass over every matrix per position. The matrices are the
things that do not fit in cache: wq_b is 42 MB on the released checkpoint, so is wo_b,
the shared expert is 35 MB of them and the engram projection is 157 MB. Prefill of a
26-token prompt was reading each of those twenty-six times per layer.

So the engine now drives a block of positions through one pass instead. `mv8_rows`
does it for the fp8 projections, holding the decoded weight tile across the block;
the output projection waits until every position has its heads, because it reads the
heads and writes the output and touches no cached state; the MoE routes the whole
block first, reduces the six-per-position draws to a list of distinct experts, and
applies each expert to every position that asked for it, which matmul_mxfp4 was
already able to do for a block of inputs.

None of it moves a value. Each output folds the same tiles in the same order, and
each expert's contribution is kept apart and summed into the output in rank order
afterwards, exactly as the position-major loop accumulated it. Blocks are capped at
32 positions so the scratch stays a few megabytes however long the prompt is.

Cold, same turn, same prompt, same seed, each step measured on the one before it:

| | wall | disk | expert matmul | attention | tok/s |
|---|---|---|---|---|---|
| serial reads, scalar kernels | 78.7 s | 36.9 s | 15.0 s | 19.1 s | 0.305 |
| batched expert reads | 42.4 s | 13.0 s | 8.4 s | 16.1 s | 0.566 |
| vector attention kernel | 41.1 s | 12.9 s | 7.8 s | 15.9 s | 0.584 |
| blocked attention matrices | 29.5 s | 12.7 s | 9.1 s | 3.4 s | 0.812 |
| expert-major MoE | 25.1 s | 10.4 s | 7.6 s | 3.4 s | 0.957 |

Of that last 25.1 s turn, prefill is 10.7 s (5.9 disk, 2.3 matmul) and decode is
13.8 s (4.5 disk, 4.7 matmul).

## The prompt's tail: the layers after layer 20 on its last 2414 rows

Layer 20 is the first with `compress_ratio` 1, and it is where the long context lives
for every layer after it: its compressed KV and index keys are the ones layers 21 to
39 attend (`kv_source`, `index_owner`), and its candidate blocks are theirs
(`candidate_source`). What those 19 layers keep of their own is a window of 128 rows.
So once layer 20 has run on every row of a prompt, a row's output in the layers after
it depends on the rows before it only through those windows, 127 rows back a layer:
the last `19 x 127 + 1 = 2414` rows of a prompt give the last row's logits, and every
window row the next decode step reads, exactly as running all the rows does. The
engine runs layers 21 to 39 on that tail only (`v41_prefill_tail`, the Vulkan chain's
chunks included: a chunk wholly before the tail stops after layer 20). A prompt of
3000 rows saves 19 of 40 layers on 586 of them; one of 8000, on 5586. DeepSeek's
deployment notes call this a bounded replay of the decoder's sliding windows.

The tiny fixture has the same layout at toy size (layers 4 and 5 after layer 3, a
window of 8, so a tail of 15): its logits are the same bytes with the tail and without
it, on the CPU and on the chain, the prompt's forward and every decode step after it,
and a tail of 14 rows already differs, so 15 is the shortest exact tail
(`tests/test_dsv41_prefill_tail.py`).

Measured on an RTX PRO 4000 (24 GB) with 62 GB of RAM, UD-IQ1_S, the Vulkan chain and
tier, DSpark off, one request per server, every row (`V41_PREFILL_TAIL=0`) against the
tail:

| prompt | every row | the tail | decode, every row / the tail |
|---|---|---|---|
| 3403 tokens | 29.3 tok/s | 31.8 tok/s (+8.5%) | 3.91 / 3.92 tok/s |
| 6777 tokens | 21.5 tok/s | 26.6 tok/s (+24%) | 3.84 / 3.88 tok/s |

With the tier's placement fixed (the same starting history, no promotions, no balance,
no streaming: `COLI_VK_TIER_GB=4 COLI_VK_TIER_RATE=0 COLI_VK_TIER_BALANCE=0
COLI_VK_TIER_STREAM=0 COLI_VK_TIER_PROMOTE=0`), four prompts of 3860 to 9145 tokens gave
the same logits bit for bit with the tail and without it, the prompt's forward and 15
decode steps each, and the same text; the four took 4589 s against 5172. With the
adaptive tier the 6777-token prompt's answer was not the same text: which experts sit on
the device follows what the prompt routed, and an expert on the device sums in a different
order than on the CPU, so two runs of the same setting with different histories differ
the same way ([speculative.md](speculative.md#how-a-verify-works)).

DeepSeek serves with a shorter tail, 140 rows (a window and the head's backtracking),
which is below the exact one and so changes the answer. `V41_PREFILL_TAIL=140` runs it
for comparison. On three prompts of 3860 to 3990 tokens it took the prompts from 520 to
352 s (1.48x), and the first token's distribution moved by a KL of 0.004, 0.015 and
0.047 nats from the exact one, single logits by up to 3.5; the first token was the same
in all three. It is not the default.

## Two ways of hiding the reads that did not work

Both are written down because they are the obvious next ideas and both were built,
measured cold, and removed.

A `posix_fadvise(WILLNEED)` hint for the next chunk of experts, issued while the
current one is being multiplied, blocks against a device that the demand reads have
already saturated. It cost four seconds of wall to save one of disk. The V4 engine
reached the same conclusion by a different route.

A pool of reader threads, so each expert is multiplied as soon as its own six tensors
land, does overlap: the main thread's wait on disk falls from 10.4 s to 1.0 s. But the
expert matmul rises from 7.6 s to 18.1 s and the turn is 7% slower with eight readers,
25% slower with four and 68% slower with two. A 5.9 MB pread is not free CPU -- it is a kernel-side copy
competing for the memory bandwidth the matmuls are already bound by -- so moving the
work off the disk wait and onto the cores that were doing the arithmetic buys nothing
and costs the contention. Anything that tries again has to start from that.

## Environment

| variable | default | what it does |
|---|---|---|
| `COLI_KV_PREFIX` | **off on this engine** | `1` reuses the previous turn's state when the new prompt begins with what that state was built from, instead of re-reading the transcript. Off by default here and on nowhere else: see "Reusing a turn" below for what it costs. `0` turns the prefix store off too. |
| `COLI_PREFIX_STORE` | on | every prompt read is kept by content (see "The prefix store" below): a new conversation on the same system prompt, or a conversation's next turn, reads only what follows the deepest place an earlier prompt was kept, with the logits a cold read gives. `0` off; `COLI_PREFIX_STORE_MB` its budget, `COLI_PREFIX_STORE_BLOCK` its block ([ENVIRONMENT.md](ENVIRONMENT.md)). |
| `V41_STATS` | off | per-turn accounting on stderr: the n-gram cache, the expert bytes and their rate, and how many drafts were accepted. Off by default because `coli chat` shows the engine's stderr next to the answer. |
| `V41_READ_DEPTH` | 8 | expert tensors read at once when a step misses the cache (see above). `1` restores the serial read the engine used to do, for an A/B. |
| `COLI_MODEL_MIRROR` | unset | `;`/`,`-separated read-only copies of the checkpoint on other drives; expert reads split across the primary and every replica (see above). `SNAP_MIRROR` is the legacy alias. |
| `COLI_MODEL_DIRS` | unset | `;`/`,`-separated directories holding **distinct** shards of the same container, so it can be split across drives with no second copy (see above). |
| `COLI_DISK_WEIGHTS` | unset (startup probe) | the split ratio, one positive weight per drive (`1,1` for an even pair). Unset measures each drive with the engine's own access pattern. |
| `V41_DIRECT` | 1 | read experts with `O_DIRECT` instead of through the page cache. `0` is the buffered arm and the escape hatch; the two paths return identical bytes. |
| `COLI_VULKAN` | off | `VK=1` build: the trunk on the Vulkan device and the backbone's routed experts on the shared expert tier (see "Vulkan" below; `COLI_VK_DENSE`, `COLI_VK_TIER*` in [ENVIRONMENT.md](ENVIRONMENT.md#vulkan-any-gpu-with-a-vulkan-12-driver)). With the tier on, the expert history `COLI_USAGE` is kept. |
| `V41_ENGRAM_ROWS` | 65536 | rows of engram cache per table. The traffic is Zipfian: common 2-grams repeat constantly, so a small cache absorbs most of it. 65536 rows is 64 MB per table on the released head_dim. |
| `V41_INDEX_OWNER` | unset | each layer scores against its own owner's index keys (see above). Changes the model's behaviour. |
| `V41_MAX_IMAGE_TOKENS` | the checkpoint's `max_image_tokens` | a ceiling on what one image costs in prompt tokens. |
| `V41_TRACE` | unset | print a checksum of the tensors the reference prints too, for locating a divergence by diffing two columns. `2` follows the first row of a speculative step instead of the last, which is the row a sequential decode is comparable to. |
| `V41_DSPARK` | on when the checkpoint carries the head | `0` disables the draft head and does not load it. Measured at +17% on the real checkpoint, cold; see above. |
| `V41_DSPARK_MAX` | the checkpoint's `dspark_block_size` | how many of the drafted tokens are put in front of the main model. Fewer means a cheaper rejected round and a lower ceiling on the win. |
| `V41_DSPARK_MINACC` | 60 | percent of the drafts written that must be accepted over a window of ten for drafting to continue (a draft the gate left unverified counts as not accepted); below it, drafts pause for 64 tokens. 60 is the measured break-even, not a guess. |
| `V41_DSPARK_GATE` | on | how many of a round's drafts are verified, from the head's confidence in each (see above). `0` verifies every draft. |
| `V41_PREFILL_TAIL` | the exact tail (2414 rows on V4.1) | how many of a prompt's last rows the layers after layer 20 run on (see "The prompt's tail"). Unset: the exact tail, the same bits as every row (on the Vulkan chain chunk by chunk; layer by layer, `COLI_VK_V41_LAYERWISE`, the routed experts' step rounds apart in the last bits, the text the same). `0` runs every row; a smaller number (DeepSeek serves with 140) is faster and approximate. |
| `V41_SPEC_FORCE` | unset | oracle mode only: draft the reference's tokens (`1`), corrupt the last one (`2`), use the head's own (`3`), corrupt the first one (`4`) or a different one each round (`5`), to exercise the verification path on a fixture whose draft head is random. |

## Vulkan (`VK=1`)

`make deepseek_v41 VK=1` links the shared Vulkan backend and its routed-expert tier;
`COLI_VULKAN=1` opens the device after the weights load (no usable device: one line,
and the run stays on the CPU). The trunk goes up in the checkpoint's own formats, fp8
in 32x32 ue8m0 tiles and bf16, vision tower included. The backbone's routed experts go
to the shared tier ([vulkan.md](vulkan.md#the-routed-expert-tier-vk_tierc)) as the
container stores them, fp4 with a ue8m0 scale per 32 (MXFP4, fmt 7), with the clamped
SwiGLU: a cache on the device that fills from the history and adapts while you chat,
computed while the CPU computes the experts it does not hold. `moe_run_at` hands each
block of positions to the device first, computes the other draws expert-major as
before and the shared expert beside them, then adds every rank in rank order, the
device's rows and the CPU's alike, and the shared expert last: the CPU path's order,
whatever was resident. The DSpark stages keep their own experts and caches on the
CPU; a verify's rows take the device like any others.

On a device that shares the CPU's RAM (an integrated GPU, Lavapipe) the trunk stays on
the CPU while the tier is on (`COLI_VK_DENSE=1` puts it on the device anyway). The
engine keeps an expert history only while the tier is on (`COLI_USAGE`, default
`<snap>/.coli_usage`, saved at every run and serve turn end): the next run's warm
start reads its hottest experts from the container in parallel. EMAP shows a
device-resident expert as tier 2, and each run and serve turn ends with a
`[VK] tier deepseek_v41` line. On Lavapipe every tiny oracle, the 40-token prompt,
DSpark at every forced acceptance and a budget that evicts give the CPU's tokens
(`tests/vulkan_engines.sh deepseek`; `deepseek-sanitize` under ASan and UBSan), and so
do they on an Intel Iris Xe through Mesa's Dozen, the one GPU it has run on (for
correctness; no speed has been measured).

`COLI_VK_CHAIN=1` runs every layer as the dense chain instead
([vulkan.md](vulkan.md#deepseek-v41-flash-and-deepseek-v4-on-the-chain)): the streams,
the attention with its window ring, compressor and indexer, the engram's projection and
gate, the mHC sites and the shared expert on the device, one host round trip per layer
for the router and the routed experts; the host's state stays canonical (written back
after every forward), so rollbacks, prefix reuse and a lost device need nothing extra.
Off by default on an integrated GPU (not measured on a V4.1 checkpoint); on the tiny
fixtures every configuration gives the CPU's tokens (`tests/vulkan_engines.sh
deepseek-chain`, `deepseek-chain-sanitize`).

### Trunk-first placement (a discrete card with Resizable BAR)

On a discrete card whose host-visible device-local heap covers its memory (Resizable
BAR), the engine places the **trunk first** and sizes everything else around it:

1. the device opens, and the chain fit runs before anything is allocated — it reads the
   device's real free memory, so every layer that fits goes up (the trunk of the Flash
   checkpoint, 40 layers of fp8/bf16 matrices plus the chain's fixed buffers, held about
   7.6 GiB of the heap once placed);
2. the trunk is placed as the layers are read, its host copies given back
   (`COLI_VK_DENSE_HOST`'s auto rule), and only then is the mapped-weights window asked:
   the heap less what the trunk took and `COLI_VK_FREE_RESERVE_GB` (1 GiB), not 9/10 of
   the heap. A driver that refuses what is left beside the trunk takes the staged path
   for what follows, as a card without a window does; `COLI_VK_STAGED=1` still forces it;
3. the expert tier sizes itself from what the device has left less
   `COLI_VK_TIER_RESERVE_GB` (1 GiB) and the dense weights still to come — inside the
   window, beside the trunk — and its **host pool stays off**: a mirror of the experts in
   system RAM the GTT imports would double what the engine's own RAM cache holds.
   `COLI_VK_TIER_HOST=1` or `COLI_VK_TIER_HOST_GB` turns it back on, and `COLI_VK_TIER_GB`
   still caps the tier.

The order matters because the window's probe allocates 1 GiB blocks and frees them, and
a driver that reports the budget only once they are back shows the fit a few hundred
MiB — which left the whole trunk on the CPU while the tier filled the window and an
823-expert mirror of it sat in system RAM. The `[VK] memory: mapped weights` line says
what was asked and why; other engines keep the order they had.

What that leaves the tier is smaller than the window itself. On the 16 GiB card this
was measured on, the driver's budget exposes about 90% of the heap (13.3 of 14.4 GiB),
the trunk holds 7.62 GiB once placed, and the window asks the remaining 5 GiB; the
tier's own sizing then sets aside its 1 GiB reserve, the ~1.4 GiB of bf16 dense tail
still to be uploaded lazily, and the streaming slots (48, ~1 GiB: measured, each one's
saving goes back to the residents), leaving a ~3.4 GiB budget — 107 experts of 21.1 MiB —
beside a trunk the whole prompt pays for on every token. `COLI_VK_TIER_RESERVE_GB=0` and
`COLI_VK_TIER_HOST` trade that back. The ledger below shows where it ends up, and the
2.1 GiB the trunk still uploads after that fit (the lm head, the 40 routers, the DSpark
stages: the token embedding is never on the device, see below).

### Quantized KV mirrors (`COLI_VK_KV_FP32`)

The only VRAM item that grows with the conversation is the chain's KV mirrors, and the
checkpoint's own cache already stores that state quantized. The packed cache keeps the
mirrors in those formats — the same layouts DeepInfer's paged cache defines
(`c/dsv41_kvq.h`) — and does so BY DEFAULT: `COLI_VK_KV_FP32` unset (or `0`) runs
packed, and `1` is the exact fp32 opt-out. There is no half-packed setting: the host's
arrays are packed exactly when the mirrors are (`v41_host_kvq` reads the same flag), and
the old middle level that kept the host fp32 while the mirrors packed measured
byte-identical to the packed host on the 12-prompt battery (at 16K only the packed host
was ever needed), so it cost the dequant→re-pack round trip the ckv subnormal-scale fix
had to repair and bought nothing measurable.

| mirror | format | bytes a row | was (fp32) |
|---|---|---|---|
| window ring (`win`) | e4m3, a ue8m0 scale a group of 32, `[data \| scales]` a token | **528 B** | 2048 B |
| compressed rows (`ckv`) | e2m1 nibbles, an e4m3 scale a group of 16 (the scale as stored scales the group) | **288 B** | 2048 B |
| index keys (`ikey`) | e2m1, a ue8m0 scale a group of 32, block-wide sections (a block of 32 tokens: 2048 B of data, then 128 B of scales — a token's 4 scale bytes one packed int32) | **68 B** | 512 B |

The compressor's `ring` stays fp32 (it is the pooling accumulator, not a mirror), as do
`isc`, the masks and the engram buffers. A row is quantized once, after its last fp32
reader (`chain_dsv4.comp` mode 13), and dequantized where it is read — the attention and
the indexer's scores decode inline (modes 0, 3, 9, 10) — and on the write-back that ends
a forward (mode 14), so the host's state stays the canonical copy — under
`COLI_VK_KV_FP32=1` fp32 rows holding the dequantized values, under the packed default
the codes themselves (below) — and rollback, prefix reuse and a lost device still
rebuild nothing.

The arithmetic at the released geometry, per token across the layers: about **6,400 B
of mirror a position under `COLI_VK_KV_FP32=1` against about 890 B packed** (the
compressed rows and index keys alone: 2.5 rows a position over the layers' compression
ratios × (288 + 68) B; the window rings and the rollback rows are the planner's fixed
term, priced below, not a per-position cost) — the figures of the prefill-VRAM analysis
([dsv41-prefill-vram-2026-11-06](experiments/dsv41-prefill-vram-2026-11-06.md), whose
§10 parked this as the lever that scales with context: 0.31 GiB of mirrors at 64K
tokens, 0.62 at 128K, 4.7 at the 1M ceiling). Quantized, the mirrors' ~6,400 B a position
become ~890: **0.78 → ~0.11 GiB at 128K (~690 MiB saved) and 6.25 → ~0.87 GiB at the 1M
ceiling (~5.4 GiB saved)** — §10's 0.62/4.7 GiB figures were the audit's savings mislabeled
as totals; the corrected basis is 6,400/890 B a position — the same math
the kv-quant test family pins on the ledger line, which reports `(quantized)` when the
mirrors are (a second device's mirrors are not in that line — the pre-existing gap).

It is an **approximate mode** — and the default — while `COLI_VK_KV_FP32=1` is the exact
fp32 opt-out, byte for byte what the engine ran before the default flipped (the flag
branches only at the mirrors' widths and stores; a run's mirrors are summed against the
flag's widths at the first forward, so a missed branch fails loudly). The FP4 index keys
can move a top-k pick, so the logits drift a little — the `kv-quant-deepseek` family gates
a documented tolerance instead of bit-identity, the way `V41_PREFILL_TAIL`'s tests accept
the tail's own approximation — and the pull's write-back moves the code rows as bytes;
under the KV split's give-way, where the host's codes dequantize up into the fp32 mirrors,
the write-back packs again (a quantize→dequant hop the pack's idempotence bounds:
re-packing a dequantized row reproduces its values, which the unit test pins). The flag
needs `head_dim` 512 and
`index_head_dim` 128: at any other geometry the flag declines — the default exactly like
a set value, one line on stderr — and the mirrors stay fp32, so models at other KV
geometries are untouched by the default. The planner prices the state below, and the
fixed term with it: the KV split decides exactly as before, and when it arms the mirrors
go back to fp32 whole (`v41c_kvq_give_way`), re-mirrored from the host.

The fixed term is not the per-position cost, and it is the engine's own
`v41_host_kv_bytes`: **(layers + stages) window rings plus `(dspark_block_size + 1)`
rollback rows a layer** — the ring rows a speculative verify parks and rolls back to —
which at the released geometry is 43 rings (40 layers + 3 DSpark stages) and 240 rows
(6 a layer, `dspark_block_size` 5), **3,032,832 B** packed against 11,763,712 B fp32.
With 2.5 compressed rows a position on top, the planner prints the engine's
**17,614,592 B at 16K** (116,621,312 B under `COLI_VK_KV_FP32=1`) — the `host kv`
16.8 MiB the ledger prints. The figure assumes DSpark loaded — the default, and a
single KV slot. `V41_DSPARK=0` (or `KV_SLOTS` > 1) leaves the three stage rings
unallocated, so the planner reserves 202,752 B packed (786,432 B fp32) more than the
engine holds; the rollback rows stay, because a layer allocates them whether or not
anything drafts. The stage count is read the way the checkpoint spells it,
`num_nextn_predict_layers` (the released `config.json` carries no `n_mtp_layers`), which
is also why the old planner priced 40 rings for a checkpoint whose engine keeps 43.

### The host's state in the same formats (the default)

`COLI_VK_KV_FP32` unset — the default, what an unset environment runs — holds the
HOST's own KV arrays -- the window rings, the compressed rows, the index keys -- in the
same cache formats the mirrors use, where `1` keeps dequantized fp32 rows: the host
state drops from 6,400 B to 890 B a position at the released geometry (86.1% less). Writers pack at their stores (rms and
the rope tail stage in a scratch row, never in place inside a code row), the read paths
dequantize on read -- the kv_all build and the indexer decode into per-forward scratch,
`V41_STATS` breaks that cost out as its own `kv read` line -- and the boundaries (the
pull, park, reset, rollback, the swap between conversations) move code rows as bytes,
with the index keys' block-wide sections never touched by a linear copy but a
whole-buffer or whole-block one. The planner re-prices the state whenever the flag
engages -- the default included (the
host layout never gives way to the KV split -- only the device mirrors do, and the
split's paths read the codes), and the turn-end ledger gains a `host kv` term so the
savings are visible at all.

What it buys, honestly measured (the memory ladder, --ram 28, kv_slots 1; every
projection tagged by the audit's regime):

| ctx | freed host RAM | +expert slots | effective --ram | prefill | decode |
|---|---|---|---|---|---|
| 16K | 86 MiB | 0 | +0.09 | ~0 | ~0 |
| 64K | 344 MiB | 0 | +0.34 | guard margin (not throughput) | ≤+2% |
| 128K | 689 MiB | +1 | +0.67 | guard margin, the split stays unarmed ~4x longer | ≤+2% (+3-8% only with a `COLI_VK_TIER_GB` raise) |
| 256K | 1,377 MiB | +2 | +1.34 | ≤2% | ≤+4% |
| 1M | 5.4 GiB | +7 | +5.51 | unmeasured | up to +50% capacity, unmeasured |

The 64K row is the plan's ±1 resolved to 0, and the reason is the floor: the 344 MiB
freed there is below one expert of every layer (771.14 MiB), and the slot a freed
budget buys is the whole cache's `cache_bytes // per_cap`, not the freed bytes' own
quotient.

The honest ceiling, from the ladder's completed arm: in the measured 16K regime
nothing moves -- prefill is disk-bound and decode is CPU/disk-bound -- and the past-32K
walls (the MemAvailable guard, an unattributed end-of-prefill spike) are not KV-sized,
so the packed host does not rescue them. The defensible wins are the guard margin at ≥64K, the
`--ram` equivalence growing linearly with context, the KV split staying unarmed about
4x longer (the quantized mirror crosses the ~1 GiB reserve near 640K of ckv/ikey
against fp32's 160K), and -- only through an explicit `COLI_VK_TIER_GB` raise, since
the tier's budget is fixed before the mirrors exist -- more device experts (at 128K:
+35 of 167). The 34 touchpoints — every writer, read path, boundary and planner price that moved —
are itemized in #2701's follow-up.

### Where the VRAM goes at a turn's end

`V41_STATS=1` (or `COLI_VK_LEDGER=1`) ends every serve turn with one ledger line
(`v41c_ledger`, `c/deepseek_v41_chain.h`), because nothing else printed the pieces
that decide whether a prompt fits:

```
[VK] deepseek_v41 vram: weights 9710.5 MiB in 34 blocks (9608.2 MiB of rows in 521 tensors) |
  chain 2060.0 MiB (dev 900.0 in 11 blocks, down 208.0 in 3, up 1024.0 in 10) | tier resident
  2257.0 MiB + 48 slots 1012.5 MiB = 3269.5 of 3310.9 in 13 blocks | extra layers 232.0 in 1
  blocks | kv mirror 115.0 MiB | process 15.31 GiB of 15.02 GiB in pools (290.5 MiB outside) |
  device 15.91 GiB used of 15.98 (card2)
```

`weights` is the weight pool's blocks (the trunk's matrices *and* the ones the first
forward still uploads: see below), `chain` the dense chain's live buffers and the three
pools it allocates from, `tier` the expert pool — residents and streaming slots together,
which is what its budget buys — `process` this engine's device bytes from
`backend_vulkan.c`'s books, and `device` the card's live `mem_info_vram_used`.

What it showed on the 16 GiB card, a 2-turn run with a 2600-word prompt:

- **The audit's unattributed 2.14 GiB on a short-prompt run is not the tier.** On that
  arm the tier's pools do not move after the warm start: the growth from the post-load
  line (`device memory held 8187281408 B`, 7.63 GiB) to the turn's end is 1262.5 MiB of
  the **lm head** (129280 x 5120 bf16) + 150 MiB of the 40 routers + 673 MiB of the
  DSpark stages' matrices, all uploaded by the *first forward* and counted by neither the
  chain's `dev_bytes` nor the tier's budget; 52 MiB is the KV mirrors. The weight pool
  holds 9608.2 MiB of rows at the turn's end against the 7523.0 MiB the setup places.
  The commit that added this line called the 1262.5 MiB the "token embedding"; that was
  wrong and the correction is worth keeping, because it is a lever that does not exist.
  The embedding has always lived on the **host**: `tie_word_embeddings` is false on this
  checkpoint, the embedding and the head are two different `129280 x 5120` bf16 tensors,
  and the embedding's only reader is `wb_row_f32` (`c/deepseek_v41.c`), which copies the
  rows a step needs into the host's rows -- decode reads one row a token, a chunk reads a
  row a position. A run instrumented at its first forward (every device upload printed)
  shows `head.weight` going up as the 1262.5 MiB item and the embedding never touching
  the device at all: the head is what the logits GEMV runs on the device, 662 million
  multiply-adds a token, so it is hot and it stays. The routers' 150 MiB are the same
  kind of item (a per-layer GEMV every forward). What is left of that 2.1 GiB is the
  DSpark stages' 673 MiB, used whenever the draft head runs.
- **The streaming slots cost one 256 MiB block, not 1.35 GiB.** The tier splits its
  budget between residents and slots (`fit = fit_raw - slots`), so shrinking the slot
  set does not by itself free anything: 64 slots and 91 residents fill the same 13 pool
  blocks as 48 and 107. The slots are now 48 by default (`COLI_VK_TIER_STREAM_SLOTS`;
  the measured runs used 44 of 64 at most), and the chunk's sizing reserves them before
  it reads the free memory (`vkt_slot_reserve`), so a 1280-row chunk becomes 1024 at
  default settings. Together that took the engine's own hold at the turn boundary from
  15.53 to 15.26 GiB (chain 2133.7 -> 2060.0 MiB) with the prefill wall +2%.
- **The card's sysfs peak does not measure the engine here.** It is pinned at the card's
  ceiling because the desktop yields: with the tier forced to 2.0 GiB the engine held
  1.46 GiB less and the sysfs peak fell 0.19 GiB, the compositor taking 1.27 GiB of what
  was freed. Read `process` for the engine, not the card. (On a prompt small enough not
  to press on the ceiling — the 11-token one — the two agree: peak = `process` + 2.49
  GiB of desktop.)
- **The candidate mask is one bit a column now, and the chunk it allows is twice as
  tall.** The indexer's scores and its candidate mask are quadratic in the prompt (a row
  of a chunk scores every compressed column below the prompt's end), and the mask was an
  fp32 word a column that only ever meant "kept" or "not" -- read as `!= 0` by the score
  kernel, by the host, and by the CPU's layers after a partial chain. `vkc_dsv4_cand`
  writes it as a bit now (`c/shaders/chain_dsv4.comp`, modes 3/4/9/10), so a row costs
  `4E` bytes of scores plus `E/8` of mask instead of `8E`, and the chunk's quadratic cap
  (`v41c_forward_seg`) keeps the same 256 MiB pair budget for 1.94x the rows: at a
  20000-position prompt a chunk was 1677 rows and is 3253, and the row cost the chunk
  sizing prints fell from 181 to 105 KiB on the tiny fixture. On the released checkpoint,
  where the mask was 4 bytes of every one of a row's E columns: 1303 -> 1298 KiB a row at
  E=1300 (measured) and 1317 -> 1305 at E=3107 (the 1317 the audit prints; the mask's
  12.1 KiB of it becomes 0.4). Nothing selects differently: the picked blocks are
  still kept whole, and the same bits come out (`tests/test_vk_chain.c`'s indexer case
  compares the mask column for column, through the single-row and the grid kernels).
- **The per-layer drain is not what the chain waits for.** The audit's R4 asked for the
  `vkc_finish()` once a layer (`c/deepseek_v41_chain.h`, after the attention frame) to
  wait only the frames the host reads. Measured with both halves timed separately, a
  1280-row chunk spends 28095 ms waiting for the frame it reads and **0.5 ms** waiting
  for the rest: two frames are in flight at a time, and a single queue runs submissions
  in order (every frame starts with a barrier on everything submitted before it), so the
  frame holding the previous layer's shared expert is always through before this
  layer's attention starts. The change was implemented, measured at 0.5 ms a chunk and
  dropped: the wait that remains is the attention itself, which the host needs (the
  routed experts' norm and, at the candidate layer, the mask). R4's own estimate
  (2-8 ms a chunk) is right; nothing 60-second-sized hides in the drain -- the device is
  idle during `moe_run_at` instead, and that is the pipelining the chain does not do.
- **Streaming a cold expert pays from two rows, not sixteen.** `COLI_VK_TIER_STREAM_MIN`
  is 2 now, from a measurement of one 21.1 MiB expert (`COLI_VK_CHAIN_BENCH=xbrows`): a
  device step costs 0.12 ms at 1 row, 0.40 at 4 and 0.76 at 16 (the tiled GEMM from
  there) against a 2.2 ms upload and the CPU's 2.05-2.52 ms a row. On the chunk-256
  workload the CPU's share of expert-rows fell from 34.5% to 1.6% and the prefill from
  226.7 to 175.9 ms a row; the device reads 2 TB of experts to do it, so it pays in disk
  bandwidth, not in time.

## Reusing a turn, and why it is asked for rather than assumed

A chat client resends the whole conversation every turn. Every other engine in
the tree skips the part it already holds, because for them a reused prefix is
the same computation as a cold prefill: the reuse changes the time and nothing
else. Here it is not, and the reason belongs to the model rather than to us.

A query reads one set of index keys when its position is **prefilled** -- its
layer's own index owner, masked to what the query can reach -- and another when
the position is **decoded**, namely whatever was published last. Both are the
vendor's, and neither can be moved to match the other:

| | tiny oracle |
|---|---|
| as shipped | 8/8 |
| the prefill given the decode schedule | 7/8 |
| the decode given the prefill owner | 1/8 |

So a position generated in an earlier turn does not attend the way the same text
attends when it is prefilled cold. A resumed conversation and the same
conversation re-read from scratch can answer differently. The resumed state is
the sequential one, so it is not the wrong answer -- it is a different one, and
that is a trade the person running the engine makes, not one the engine makes
for them. `COLI_KV_PREFIX=1` asks for it.

What is exact, and gated in CI, is a resumed **prefill**: a prompt that grows
without generated text in between -- an agent resending a document, a tool loop
-- reuses losslessly. Measured on the tiny fixture, a prefill that resumes 4,
12, 24 or 48 tokens from the end reproduces the cold prefill's tokens exactly.
That was not free: the index-key schedule used to be selected by `start_pos > 0`,
which meant "the speculative verify batch" only because a prefill always started
at position 0. A resumed prefill matched it too and took the decode schedule,
reading another layer's keys and choosing a different index top-k. It is now
selected by the flag that actually distinguishes the two.

Three more things the attempt turned up, all of them unreachable before reuse
existed and all fixed here: the per-layer undo buffers (`ring_save`,
`cstate_save_*`) are sized for one draft block and were written on any multi-row
forward past position 0, which a resumed prefill overflows; `spec_step` decided
"seed the stages, do not draft" by `start_pos == 0` and seeded them at position
0, so a resumed prefill fell into the drafting path with no draft buffer; and the
record follows the speculative rollback, so it never claims rows the caches gave
back.

## The prefix store: a new conversation, and the next turn, without reading again

Resuming the live state is opt-in here (above), and nothing can wind back the window rings
or the compressor's filling group: so every request read its whole prompt. An agent
starts each task as a new conversation on the same system prompt and tools, and a chat
client resends the conversation every turn, so most of what a request read had been read
before. `prefix_store.h` keeps it (qwen36 does the same, [qwen36.md](qwen36.md#the-prefix-store)):

- **What is kept.** The compressed rows and index keys in blocks of 256 tokens named by
  their tokens and the block before them (a row for `ratio` positions: 64 a block at
  ratio 4, 2 at ratio 128), shared by every prompt that starts with them; and
  checkpoints of the rest, every layer's window ring and its positions, the
  compressor's filling group and the layer whose index keys were published last. The
  engram history is the prompt's ids again, so it is rebuilt rather than kept.
- **Where.** A prompt is read in pieces where its messages start (`<｜User｜>`,
  `<｜Assistant｜>`, `<｜System｜>`): at the first one past what the slot held (after a new
  conversation's system prompt, where the next conversation parts from it) and at the
  last one, and a checkpoint is kept after each piece and where the prompt ends.
- **Only a prompt read's.** A request that started from nothing or from the store
  keeps what it reads; one that resumed the live state (`COLI_KV_PREFIX=1`: a decoded
  reply in it) or a photo keeps nothing. A request takes the deepest checkpoint its
  prompt starts with when it reaches further than its slot, two positions short of the
  prompt's end at most: a single position read alone would score its index keys against
  the keys published last, as a decoded one does. So taking a checkpoint is a resumed
  prefill, which is exact: the logits are the ones the same prompt read cold gives.
- **What it serves.** A new conversation on the same system prompt reads only its own
  message; a conversation's next turn reads from where its last prompt ended (the
  reply and the new message; with a thinking-mode reply, whose opening `<think>` the
  next prompt writes as `</think>`, from where its last message started); a
  conversation sent back to the other slot the same.
- **Not kept or taken** for a picture's prompt (its placeholders are the same for every
  picture) or a read-out (`logprobs`: every position's frame).

A prompt's top-k lists and candidate masks are not in a checkpoint: the rows read past
it publish their own before any layer reads them (a decoded position would read the last
forward's, and nothing here resumes one). Leaving out any of the window rings, the
filling group, the rows or the engram history makes `tests/test_dsv41_prefix_store.py`
fail; leaving out those lists does not.

`tests/test_dsv41_prefix_store.py` holds every request that took the store to a fresh
engine reading the same prompt with the store off, each step's top five logits to the
last digit: a new conversation on the same system prompt, a conversation's next turn
after another, two slots with the conversation on the other one, the speculative
decoder on (the same bytes), a budget of nothing, the store off, a request that resumed
the live state keeping nothing, a read-out reading everything. `DSV41_PS_BACKEND=vulkan`
and `cuda` run both engines on that chain, the same top five within 1e-4: Lavapipe in
`tests/vulkan_engines.sh` (the safetensors fixture and the GGUF one), the CUDA chain on
an RTX PRO 4000 (a GGUF's).

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

The quantized mirrors have their own family: `tests/test_vk_kv_quant.c` holds the
device's pack to `c/dsv41_kvq.h`'s reference bit for bit (the edge rows of the scale
math included), the unpack, a re-upload's lanes and the pack's idempotence, and the
index scores over quantized keys to the decoded rows' bits with the original rows'
top-k overlap reported; `tests/vulkan_engines.sh kv-quant-deepseek` runs the tiny
fixture at the released KV geometry under the flag against a documented logits
tolerance (the approximate mode cannot be bit-exact; the flag-off run is the exactness
proof) and checks the ledger's mirror math.

## Where this engine is not the vendor

The vendor quantizes some activations on the fly, in kernels that only exist for the
GPU: the KV cache is rounded to fp8 before it is stored (`act_quant(..., inplace=True)`
in `Attention.forward`), and the indexer's queries and keys, and the compressor's
latents, are rounded to fp4. This engine keeps all of them in fp32 -- more precise, and
therefore not bit-identical to a GPU run of the same weights. Nothing downstream is an
argmax over a near-tie by construction, but it is a difference, and it is the reason a
divergence against a GPU reference would not necessarily be a bug here.
