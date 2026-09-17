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
| `COLI_KV_PREFIX` | **off on this engine** | `1` reuses the previous turn's state when the new prompt begins with what that state was built from, instead of re-reading the transcript. Off by default here and on nowhere else: see "Reusing a turn" below for what it costs. |
| `V41_STATS` | off | per-turn accounting on stderr: the n-gram cache, the expert bytes and their rate, and how many drafts were accepted. Off by default because `coli chat` shows the engine's stderr next to the answer. |
| `V41_READ_DEPTH` | 8 | expert tensors read at once when a step misses the cache (see above). `1` restores the serial read the engine used to do, for an A/B. |
| `COLI_MODEL_MIRROR` | unset | `;`/`,`-separated read-only copies of the checkpoint on other drives; expert reads split across the primary and every replica (see above). `SNAP_MIRROR` is the legacy alias. |
| `COLI_MODEL_DIRS` | unset | `;`/`,`-separated directories holding **distinct** shards of the same container, so it can be split across drives with no second copy (see above). |
| `COLI_DISK_WEIGHTS` | unset (startup probe) | the split ratio, one positive weight per drive (`1,1` for an even pair). Unset measures each drive with the engine's own access pattern. |
| `V41_DIRECT` | 1 | read experts with `O_DIRECT` instead of through the page cache. `0` is the buffered arm and the escape hatch; the two paths return identical bytes. |
| `V41_ENGRAM_ROWS` | 65536 | rows of engram cache per table. The traffic is Zipfian: common 2-grams repeat constantly, so a small cache absorbs most of it. 65536 rows is 64 MB per table on the released head_dim. |
| `V41_INDEX_OWNER` | unset | each layer scores against its own owner's index keys (see above). Changes the model's behaviour. |
| `V41_MAX_IMAGE_TOKENS` | the checkpoint's `max_image_tokens` | a ceiling on what one image costs in prompt tokens. |
| `V41_TRACE` | unset | print a checksum of the tensors the reference prints too, for locating a divergence by diffing two columns. `2` follows the first row of a speculative step instead of the last, which is the row a sequential decode is comparable to. |
| `V41_DSPARK` | on when the checkpoint carries the head | `0` disables the draft head and does not load it. Measured at +17% on the real checkpoint, cold; see above. |
| `V41_DSPARK_MAX` | the checkpoint's `dspark_block_size` | how many of the drafted tokens are put in front of the main model. Fewer means a cheaper rejected round and a lower ceiling on the win. |
| `V41_DSPARK_MINACC` | 60 | percent of drafts that must be accepted over a window of ten for drafting to continue; below it, drafts pause for 64 tokens. 60 is the measured break-even, not a guess. |
| `V41_SPEC_FORCE` | unset | oracle mode only: draft the reference's tokens (`1`), corrupt the last one (`2`), or use the head's own (`3`), to exercise the verification path on a fixture whose draft head is random. |

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
