# Qwen3.6 on Metal

[Qwen3.6 engine](qwen36.md)

On Apple Silicon the Qwen3.6 engine runs the forward pass on the GPU (in a
fast prompt batch the CPU's matrix unit takes part of the products). The
routed experts are still read from the model on disk through a bounded cache,
so they do not have to fit in RAM. The timings on this page were measured on
an 8 GB M1 MacBook Air while the graph was developed; the measured comparison
with the CPU engine is the
[experiment record](experiments/qwen36-metal-graph-2026-10-05/summary.txt).

## Build and run

```sh
make -C c qwen36 METAL=1
SNAP=/path/to/qwen36 ./c/qwen36 8 4 prompt.txt
```

The build needs a clang with C23's `#embed` (the kernels are built into the
executable) and the macOS 15 SDK (Accelerate's threading interface, called
only on macOS 15 and later). `make -C c test-qwen36-graph METAL=1` checks the
graph against the CPU engine on a small model the test writes.

Unlike the other engines' Metal paths, which `COLI_METAL=1` turns on
(`K3_METAL=1` for Kimi K3), a `METAL=1` build of qwen36 runs the graph by
default; the release archives build qwen36 without it, so their qwen36 is the
CPU engine. `COLI_METAL=0` keeps the CPU engine. It also runs instead, with one
line that says why, where the graph cannot: with a qpack container
(`QWEN36_QPACK`), a GGUF with a matrix of a type the graph does not read (see
[GGUF](#gguf)), `COLI_DENSE_I8=0`, `COLI_DENSE_BITS=4` or `16` or
`COLI_KEEP_F32` (the graph reads the INT8 dense weights), `COLI_VULKAN=1` in a
`VK=1` build (the Vulkan dense chain), `CACHE_ROUTE=1` (cache-aware routing is
the CPU engine's), routed experts that are not planar INT4 with groups of 64 or
`QWEN_EXPERT_KERNEL=0`, a model without routed experts or without a shared
expert, a model with sigmoid routing, attention sinks, a sliding window or dense
layers (Flint, Muse Spark: the CPU engine runs them), a geometry outside its
kernels (top-k up to 8, up to 256 experts, heads of up to 256, DeltaNet keys of
up to 128, widths in multiples of 64), a Clef decision head ([clef.md](clef.md):
DECIDE reads the hidden state of every position, and the graph returns only the
last one), no Metal device or one that cannot build the kernels, and the Apple
Paravirtual device of virtual machines and CI runners. A serve of several
conversations at once (`KV_SLOTS` above 1) keeps the CPU engine too: the graph
holds one conversation's state on the GPU.

The graph takes no images: the vision tower is not loaded (a line at start says
so), so a chat turn with an image is refused, and through the gateway the
refusal reads as for a container without the tower, though it is there;
`COLI_METAL=0` serves images. The graph takes no prompt-lookup drafts
(`COLI_LOOKUP`; its drafts come from the checkpoint's MTP head, see [Speculative
decoding](#speculative-decoding), and `Q36_MTP=1`, the other engines' head, is
not loaded, with one line) and no read-out of the prompt's logits: a logprobs
request is refused, so `/v1/systemone` answers an error (the server log says
why) and its scoring needs `COLI_METAL=0`. It keeps no pins (`pin=1` is ignored,
with one line). The CPU engine's per-layer expert cache (the size argument
above, `PILOT`, `HOT`, `COLIBRI_RESIDENT`) is not used: the graph has one cache,
`QWEN_EXPERT_CACHE_BYTES`. In a serve the dashboard's expert map shows that
cache; its profile has the wall time, the counts and the disk time (summed over
the readers, which read in parallel), and its phase columns stay 0 unless
`COLI_TIMERS=1`.

## GGUF

A GGUF of a Qwen3.5 or Qwen3.6 MoE model (`qwen35moe`, [gguf.md](gguf.md)) runs
on the graph too, read in place: its dense matrices and routed experts are
computed as the file's ggml blocks, with one decoder per type and no second
quantization. The model's own format (`canonical.qc`, `experts.qc`) and its
kernels are unchanged beside it.

- The graph reads Q8_0, Q4_K, Q5_K, Q6_K, IQ4_NL and IQ4_XS, in the dense
  matrices and in the experts, each matrix and each layer of its own type, as
  published files mix them, and a dense matrix of floats (F32, F16 or BF16; the
  published files keep the router in F32) as the CPU engine holds it, in F32.
  Of unsloth's Qwen3.6-35B-A3B files, UD-Q4_K_S, UD-Q4_K_M, UD-Q4_K_XL,
  UD-Q5_K_M, UD-Q6_K and Q8_0 hold only these. UD-IQ4_XS and UD-IQ4_NL keep
  their experts' gate and up in IQ3_S: a file with a matrix of another type
  runs on the CPU engine, with a line that names the matrix and its type.
- The dense matrices are seen in the file's mapping, each in a Metal buffer of
  its own pages, since the file keeps the experts between them and the GPU holds
  a buffer resident whole: nothing is copied. The embedding rows are read from
  the mapping too. A matrix's `.scale` companion multiplies its products, per
  matrix or per expert.
- The routed experts go into the expert cache's slots as the file holds them,
  gate, up and down one after the other, in three reads (one with `coli import
  --mode copy`). A slot is as large as the file's largest expert: 2.0 MB on the
  35B model's UD-Q4_K_M, against 1.8 MB in the model's own format, so the same
  `QWEN_EXPERT_CACHE_BYTES` holds fewer of them. At an 8 GB Mac's default (a
  sixteenth of the RAM, 512 MiB) the 35B model's files leave fewer than the 256
  retained slots the layer-by-layer prompt needs (UD-Q4_K_M: 263 slots, 255
  retained; the larger types fewer): a prompt then loads its experts eight at a
  time, with a line, unless `QWEN_EXPERT_CACHE_BYTES` holds top-k + 256 slots
  (538509312 bytes for UD-Q4_K_M). On an 8 GB M1 that one slot more made
  UD-Q4_K_M's 512-token prompts 15 to 31% faster, with the same logits; in the
  decode after them the medians were 0.2 to 4.8% lower and the runs
  overlapped, but on code three of the four default runs were faster than
  every run with the larger cache
  ([the experiment's record](experiments/qwen36-gguf-metal-2026-10-19/README.md)).
- The file's three differences from the checkpoint are taken as it stores
  them, as the CPU engine takes them: full-gamma norms, `A = -exp(A_log)`, and
  DeltaNet value heads in llama.cpp's tiled order (value head h reads key head
  h modulo the key heads).
- Exact mode (`QWEN36_METAL_EXACT=1`) sums each product as the CPU engine does
  under `GQ_EXACT=1` (each product rounded, added in input order), and the F32
  router as the CPU engine does in both modes: the test finds the CPU engine's
  layers and logits within 1e-4 of their largest value (about 7e-6 measured).
  In a `METAL=1` build that CPU path is `COLI_METAL=0 GQ_EXACT=1`. Fast mode
  spreads a row's blocks over the lanes of a SIMD group in a decode step, as
  llama.cpp's matrix-vector kernels do, and applies the block scales to each
  block's sums of the quants times x; a prompt runs on the matrix units, each
  lane decoding its own weights. The routed experts take the same kernels, and
  in a prompt the int4 experts' tiles. Fast mode is float throughout but
  groups its arithmetic differently (the block scales applied to each block's
  sums, fused multiply-adds), so its results differ from exact mode's in the
  last bits (the test's fast and exact layers within about 3e-5).
- On an 8 GB M1 the model's own format is faster. There, in fast mode, the
  35B model's UD-Q4_K_M decodes 21 to 26% slower than the model's own format
  (7.41 against 9.44 tok/s on prose after a 512-token prompt) and, at the
  default budget (its prompt's experts eight at a time, none of the CPU's
  share), reads 512-token prompts at about half the speed (66 against 132
  tok/s), with 107 to 109 MiB more peak memory; with
  `QWEN36_FAST_EXPERTS=5 QWEN36_FAST_GUESSED=4` it decodes 15 to 22% slower,
  in exact mode 39 to 42%
  ([the experiment's record](experiments/qwen36-gguf-metal-2026-10-19/README.md)).
  Most of a decode step's extra time is the wait for the experts read from
  disk (see [Where the time goes](#where-the-time-goes)).
- Not with a GGUF: the CPU's share of a fast prompt (`QWEN36_CPU_MATMUL`: its
  products take INT8 weights), the MTP head, the graph's only drafter (no
  drafts; a line says so when the file has one, or with `QWEN36_MTP=1` or
  `COLI_LOOKUP=1`), `experts.qc` and `QWEN36_PREFETCH` (both
  ignored, with a line). At exit a line counts the products with the file's
  matrices and experts that ran on the GPU.
- `GGUF_CHECK=1` (a header alone, [gguf.md](gguf.md)) builds no graph: the CPU
  engine's loader checks the model, and a line says whether the graph would
  run it.

## Memory

The dense weights (everything except the routed experts, lm_head included)
are read by every token. Converted once to row-scaled INT8 and kept beside the
model as `canonical.qc`, they are mapped at start and read in place, by the
GPU too:

```sh
make -C c qwen36-canonical
./c/qwen36-canonical "$SNAP"        # writes $SNAP/canonical.qc
```

Without the file the engine converts the same matrices at every start and
copies them to GPU memory, then frees its own copies: the same numbers, but a
longer start, and 1.9 GB on the 35B model that the system cannot drop and read
back from the file, as it does with the pages of a mapped one. `QWEN_DENSE_CACHE=<path>` points to a
file kept elsewhere. The CPU engine (`COLI_METAL=0`) uses the file too. The
embedding rows are read in place from the checkpoint as well, instead of a
float32 table of 2 GB (`COLI_EMBED_MMAP=0` keeps the table).

A prompt's expert reads fill the system's file cache, and by its end the
pages of the file it did not read lately are out of memory: the head's, and
in a long prompt the earlier layers' too. Near the end of a prompt the engine
asks the system to read them back (`madvise` from another thread), so the
first head after the prompt does not wait about 300 ms for them and the first
decode step after an 8192-token prompt takes about 360 ms instead of 510.

At start the file is checked against the model as the CPU engine checks it
([qwen36.md](qwen36.md#canonicalqc-the-dense-trunk-converted-once)): a file
beside the model that does not match is set aside with the reason and the
matrices are converted, while one named by `QWEN_DENSE_CACHE` stops the start.

The routed-expert cache defaults to a sixteenth of the RAM (512 MiB on an
8 GB Mac); `QWEN_EXPERT_CACHE_BYTES=<bytes>` changes it. A larger cache means
fewer disk reads, as long as the rest of the working set still fits. With
`QWEN36_PREFETCH=1` eight more expert slots sit beside the cache (14.2 MB on
the 35B model, see below).

The batch tensors hold the rows of the current forward: up to 2048 for a
prompt while the system reports no memory pressure, 1024
otherwise (about 340 KB a row: 350 MB more for the larger batch, which runs an
8192-token prompt 4% faster because each expert gets twice the rows;
`QWEN36_PROMPT_ROWS=1024` or `2048` sets it), one in decode (eight with the
MTP head, for a verify; up to eight after a short prompt, whose tensors decode
keeps). Past 1024 rows
the KV cache grows in steps of 1024 (42 MB each on the 35B model, 21 MB in
fast mode) instead of doubling. Fast mode keeps attention keys and values in
half precision (168 MB less at 8192 tokens of context) and only the high plane
of the DeltaNet state: the low plane of the compensated sums is the exact
kernels', and the fast ones always left it at zero (63 MB less, 126 MB with
the serve-mode copy). Memory a forward does not use goes back to the system,
whose file cache then keeps more experts.

In decode the Metal graph keeps the experts whose next use is expected
soonest: for each expert it tracks how often it is used and how often it is
used again right after, as averages that halve every 16 decode steps
(`c/qwen36_expected_policy.h`). The prompt does not train these estimates.
`QWEN36_CACHE_EXPECTED=0` keeps the global cache's frequency policy
(`c/qwen36_cache_policy.h`), which prompts use. Both use the same
expert-payload budget.

In fast mode `QWEN36_FAST_EXPERTS=J` reads fewer experts from the disk in
decode (every one-token forward after the first position), at a cost in
quality: a token uses its J likeliest routed experts, and the rest of its top-8
only when the expert cache already holds them
(`c/qwen36_fast_experts.h`). A left-out choice costs no read and no cache
slot: it points at the heaviest expert with weight 0, whose FFN runs once more
for it. The kept weights stay as the router computed them, divided by the
probability mass of the whole top-8, so a left-out expert's share is simply
missing; renormalizing over the kept experts would raise their gain above what
the model was trained with (arXiv 2609.04575 studies this on Qwen3.6-35B-A3B).
Unlike `CACHE_ROUTE` of the CPU engine ([CACHE_ROUTE.md](CACHE_ROUTE.md)),
nothing is substituted. Which experts are left out depends on what the cache
holds (the RAM, `QWEN_EXPERT_CACHE_BYTES`, the requests before), and so does
the output. Unset or 0, the default, and in exact mode every chosen expert is
used; a value that is not a whole number from 1 to top-k − 1 is refused with a
line, and a line says when the option is on. The speed and the quality on an
8 GB M1 are in
[the experiment's record](experiments/qwen36-fast-experts-2026-10-06/README.md).
The quality there is the likelihood of three texts, which the same paper finds
can rank such settings wrongly on this model: with 6 of 8 experts and the
weights divided by the top-8 mass, as here, its 5-shot MMLU lost 1.10 points
(p = 0.021). No downstream task was measured.

`QWEN36_FAST_GUESSED=R` narrows fast experts in the middle ranks, so fewer
experts are read, at a further cost in quality: a token uses its R − 1
likeliest routed experts, its next ones up to the J likeliest only when the
expert cache holds them or the layer before guessed them for its layer, and
the rest of its top-8 as above. A guessed expert's read was announced while
the layer before computed, so keeping it costs little; a choice neither
guessed nor in the cache would wait for a whole SSD read. The first layer of
a step, which nothing guessed, and a verify's draft rows keep the rule of fast
experts. Unset or 0, the default, changes nothing; in exact mode, without
`QWEN36_FAST_EXPERTS` of 2 or more, or with a value that is not a whole number
from 2 to J, it is refused with a line. A line says when it is on, and one at
exit counts the rows the rule ran for (a token at a layer) and the choices
that stayed because they were guessed. The speed and the quality on an 8 GB
M1 are in
[the experiment's record](experiments/qwen36-fast-guessed-2026-10-14/README.md);
the quality is again the likelihood of three texts. A token keeps from R − 1
to 8 experts (3 to 8 with R = 4, the setting measured), and on this model the
paper above found that 4 of 8, divided by the top-8 mass, lost 3.10 points of
5-shot MMLU.

An optional disk cache, `experts.qc` beside the model, stores the experts in
the layout of those slots, each expert's weights and scales together, so a
slot is filled by one read:

```sh
make -C c qwen36-canonical
./c/qwen36-canonical --experts "$SNAP"   # writes $SNAP/experts.qc
```

It repacks the same INT4 values, without requantization. The scales that
Colibri's converter (`c/tools/convert_qwen36.py`) writes from a BF16
checkpoint are max|w| / 7 of a bfloat16 max|w| (1e-12 / 7 for an all-zero
group): when every scale is one, the file stores that bfloat16 in 2 bytes and
the reader divides it by 7 again into the same F32 bits (layout 2, checked
over every scale before writing); otherwise it keeps the F32 scales (layout
1). On the 40-layer, 256-expert model that is 17.1 GB of disk beside the 23 GB
of the checkpoint (18.1 GB in layout 1), and 5.6% fewer bytes read per expert.
It is a file beside the checkpoint rather than another container: the CPU
engine and the adapters keep reading the checkpoint, and the file can be
deleted or rebuilt without touching it. The conversion of the 35B model reads
about 20 GB (the scales twice) and writes 17.1 GB; on an 8 GB M1 it took 23 s
and 10 MB of memory, measured before the converter synced the file
(`F_FULLFSYNC` on macOS) and handled signals. A rebuild needs 17.1 GB free
while the old file is there.
A conversion stopped by a signal removes its temporary; one killed with
SIGKILL, or by a machine that stops, leaves `experts.qc.tmp.XXXXXX`, which can
be deleted. The file gets the mode of a plain file under the umask.

The RAM budget and the cache policy stay the same. `QWEN36_EXPERTS_QC=0`
reads the checkpoint, to measure; `QWEN36_EXPERTS_QC=<path>` names a file kept
elsewhere. At start the file's header and size are checked against the model
and against a stamp of the checkpoint: the source tensor descriptors and, for
each shard, its inode, size, modification and change times and two samples of
its first expert tensor. The stamp leaves out the volume's device number,
which macOS can change at a restart; any change to a shard's metadata (a
`chmod` or an extended attribute too) makes the file stale. A file beside the
model that is stale (for example after copying the checkpoint to another
disk), or whose header or size does not match, is reported and ignored: the
experts come from the checkpoint until it is rebuilt. One that
`QWEN36_EXPERTS_QC` names stops the start instead. The records carry no
checksum: damage inside a file of the right size is not detected, and a read
error during a run stops the engine, as one in the checkpoint does. The
converter writes a temporary file, syncs it to the disk (`F_FULLFSYNC` on
macOS) and publishes it by rename.

With `experts.qc`, `QWEN36_PREFETCH=1` copies the experts guessed for the
next layer (those announced to the disk) while this layer computes, on two
queues of their own, into eight slots beside the expert cache, one expert
each: 14.2 MB on the 35B model, on top of the expert cache's budget
(`QWEN_EXPERT_CACHE_BYTES`). A layer that misses one of them takes the copy
instead of reading the record while the step waits, and waits for a copy
still reading: the cache's slot and the copy's trade their storage. A copy
done when the layer starts is computed early, with the experts the cache
holds, while the layer's other misses read. The
cache makes the same choices as without the copies and its slots end up with
the same bytes, so the output does not change, in any mode. Only decode steps
of one token copy; a verify of the MTP head's drafts copies nothing. A guess
is not copied when the cache holds it, when its layer has read its experts
already, or when the system's file cache does not hold every page of its
record (asked with `mincore`): a cold one would hold a queue for a disk read,
and is only announced, as before. Unset or 0, the default, keeps it off;
another value and the variable without `experts.qc` are refused with a line.
A line at start gives the slots' bytes, and one at exit counts the copies,
those their layer took or computed early, and why the others were not made.
The speed on an 8 GB M1 is in
[the experiment's record](experiments/qwen36-prefetch-2026-10-18/README.md).

In serve mode (`SERVE=1`, what `coli chat` and the OpenAI server drive) chat
clients resend the whole conversation at every turn. The forward keeps a copy
of the DeltaNet state where the prompt's last turn opens, and a next request
that begins the same way reads only what it adds: at 8800 tokens a second turn
starts in 4 s instead of 100. The copy takes 129 MB (66 MB in fast mode), from
the first request on. `COLI_KV_PREFIX=0` turns it off with the rest of the
prompt reuse.

## Speculative decoding

Qwen3.6's checkpoint carries a multi-token-prediction (MTP) head: one more
decoder layer that reads the model's last hidden state at a position with the
token after it, and predicts the token after that. `tools/convert_qwen36.py`
carries it into the container as `model-mtp.safetensors` (f16 dense weights and
norms, the routed experts in the model's own int4 layout), and `--mtp-only
--base-container` adds it to a container converted before, in a new directory of
links ([qwen36.md](qwen36.md#the-mtp-head-q36_mtp)). The graph reads the same
tensors as the other engines' head (`Q36_MTP=1`), runs its own on the GPU, and
does not load theirs.

With the head in the container, the graph decodes speculatively; `QWEN36_MTP=0`
turns it off (a value other than 0 and 1 is ignored, with a line). The engine
reads every `*.safetensors` of the container, with the head off too: a damaged
`model-mtp.safetensors` stops it as a damaged shard of the model would.

After each token the head drafts the next one, and when its confidence (the
draft's probability under a softmax of the head's logits for the token ids it
drafts from, below) is at least 0.6 (`QWEN36_MTP_MIN_P`, a number from 0 to 1;
0 verifies every draft), one forward verifies the picked token and the draft
together: when the next pick equals the draft, the second row's logits answer
it with no forward of their own. The picks always come from the model's logits
([speculative.md](speculative.md)). In exact mode the tokens and logits are
those of plain decoding, byte for byte: a verify computes each row with a
decode step's kernels (the DeltaNet recurrence row by row), and a rejected
draft swaps back the DeltaNet state copied after the first row. Fast mode runs
the two rows through the batch kernels. The gate of the other draft sources
([speculative.md](speculative.md#the-gate)) stays off for these drafts: a
draft goes to a verify on its confidence alone. The gate weighs a verify
against the plain steps of the same run, while a verify also makes the steps
after it faster (the expert cache then holds two tokens' experts).

The head runs on the GPU as one more layer. Its attention keys and values have
their own cache, and its routed experts go through the expert cache as the
layer after the model's last (the 41st on the 35B model). It reads only its two
heaviest routed experts, so that its reads take few of the cache's slots from
the model's layers: its other choices (six on the 35B model) point at the
heaviest expert with weight 0, whose FFN runs once more for each (no read, but
its compute). It drafts among the first 32768 token ids: a draft reads 67 MB of
the model's 0.5 GB `lm_head`. Those ids hold most tokens of English, other
Latin-script text and code, but no Chinese character (their ids start at 95726)
and few Arabic or Korean ones, and its confidence counts only them. Where half
or more of the last 16 tokens lie past them, the head does not run: it could
draft only the other tokens there, and it costs about 5% of a step on the 35B
model; its attention starts again when it runs again. Of a prompt it reads only
the last 32 positions, which it would otherwise go through whole, and keeps
only their final residuals; its attention starts at the first of them, and
starts again where a position does not follow its last one (a turn resumed from
the copy above). Where 8 or more of a prompt's last 15 tokens lie past the ids
it drafts from, the prompt skips the head: it will not run at the next step,
whatever the token, and its pass would go unused. A verify reads the experts of
both rows in one lease, and the expected-reuse cache policy counts them as one
visit of the layer. So between steps the cache keeps top-k fewer experts than
without the head (287 of the 303 slots of the default size on an 8 GB Mac,
instead of 295), and a prompt runs its layers in windows (below) only with 272
slots or more instead of 264. A budget under two top-k groups (16 slots on the
35B model) stops the start with a line; `QWEN36_MTP=0` runs with it. The hit
rate a serve turn reports also counts the head's experts and those of each
verify.

With `QWEN36_FAST_EXPERTS` above 3, a verify's draft row keeps two sure
experts fewer (3 with J = 5) except in the last quarter of the layers; with any
J it counts the experts the first row chose as held by the cache, since the
verify reads them anyway.

The head's memory: its dense weights, the copy of the DeltaNet state a verify
keeps (63 MiB on the 35B model in fast mode, 123 MiB in exact mode) and the
verify's buffers; decode keeps the tensors of an eight-row batch
(`Q36_SPEC_ROWS`), whose attention scores grow with the context.

## How it is built

| File | Role |
| --- | --- |
| `c/metal/qwen36.metal` | the kernels, built into the executable |
| `c/qwen36_metal.m` | Metal objects: device, buffers, the command batch, one wrapper per kernel |
| `c/qwen36_gpu.h` | the C interface to the above |
| `c/qwen36_graph.h` | the forward pass: one struct of GPU tensors, one function per layer kind, one forward for prefill and decode |
| `c/qwen36_global_cache.h` | the expert cache; on Metal its slots are GPU buffers |
| `c/qwen36_prefill_stream.h` | fast prompt layers: trunk windows, then the experts on the whole batch |
| `c/qwen36_layer_loader.h` | a prompt layer's experts, loaded by four reader threads |
| `c/qwen36_expected_policy.h` | decode expert-reuse estimates for the cache |
| `c/qwen36_advice_schedule.h` | which guessed experts a decode step announces (the six likeliest), after the next trunk is submitted |
| `c/qwen36_fast_experts.h` | fast experts: which of a token's routed choices a decode step keeps |
| `c/qwen36_dense_file.h` | the `canonical.qc` format: header, table, INT8 matrices |
| `c/qwen36_expert_file.h` | the `experts.qc` format and checked record reads |
| `c/qwen36_expert_source.h` | the checkpoint stamp `experts.qc` is checked against |
| `c/qwen36_expert_convert.h` | the writer of `experts.qc` |

Per layer, the trunk and the router run as one GPU batch. The CPU then reads
which experts were chosen and sorts the tokens by expert (a row the router
could not fill, from a NaN in its input, degrades as on the CPU engine: the
slot's own expert at weight 0, with one line). A worker thread
fetches the missing experts from disk, eight at a time and up to four at once;
the first eight come while the GPU computes the shared expert. In generation the GPU also runs the
next layer's router on this layer's residual: about four times in five it
picks the experts the next layer will choose, and of its six likeliest picks
(its J likeliest with `QWEN36_FAST_EXPERTS=J` below six),
those not in the cache are announced to the SSD as soon as the next layer's
trunk is submitted, so the system calls run while the GPU computes. Each expert is read once per batch
of tokens and computes every token that chose it. A last kernel adds up the
experts of each token in the router's order, with the shared expert and the
residual, so the result does not depend on how the experts were grouped, and
writes the next layer's input. In generation the GPU is waited for once per
layer for the choices, and once more for each further group of eight experts,
before their cache slots can be reused; a fast prompt waits once per window. Otherwise the tensors stay on the GPU for the
whole pass.

A prompt batch in fast mode runs its layers differently: the trunk and the
router run in windows of up to 64 rows (128 in a batch of 1024 rows or more,
256 in one of 1536), and after each window the CPU adds the experts those rows
chose to the layer's set (at most 256). Four reader threads load the missing ones straight into
cache slots while the next windows run. The shared expert and the routed
experts then process the whole batch, eight experts at a time, with no wait
between groups (the groups already loaded run concurrently, see below): the
layer's experts stay borrowed until its GPU work is done. Every miss is
admitted to the cache, so a prompt can leave a different set of experts for
decode. A prompt longer than one batch runs layer by layer: all its batches
through a layer, with that layer's experts kept, then all through the next. A
batch chooses nearly every expert of a layer, so batch by batch each batch
read them all again (about 18 GB); layer by layer the prompt reads each once.
The later batches of a layer find its experts loaded and run the trunk in one
window: one wait for the GPU instead of 8 or 16, and matrix products of the
whole batch instead of 128. The residual of every token waits between layers
in one buffer, 4 bytes per hidden unit and token (67 MB at 8192 tokens), freed
at the end of the prompt; the results are the same bits.
`QWEN36_PREFILL_STREAM=0` keeps the schedule above, to measure; exact mode,
batches of at most eight rows and a cache with fewer than 256 retained slots
(reported at startup) use it too.

Of a prompt's last layer only the last token's output goes on, to the final
norm and the head. Where that layer is an attention layer, as on the 35B
model, the tokens before the last only write their keys and values to the
cache (the input norm, the k and v projections, the k norm and RoPE, as the
whole layer computes them), and the last token runs the whole layer alone, at
its position: the queries, attention, output projection, router and experts
of the last layer are skipped for every token but one. Exact mode keeps every
bit; in fast mode the last token takes the generation kernels.

With the MTP head, the rows whose final residuals the head reads run the last
layer whole too: its window, the last 33 rows (all rows of a shorter prompt),
or every row of the prompt when the window starts in an earlier batch (a last
batch of fewer than 33 rows); a prompt that skips the head runs its last row
alone, as without it. In fast mode they take batch kernels, so the
first token's logits can differ in the last bits from a run with
`QWEN36_MTP=0`; exact mode keeps every bit.

## Exact and fast kernels

By default a prompt batch (more than 8 rows) runs its dense projections and
shared expert (`q36_dot_i8_mm`), its attention (64 rows at a time) and its
routed experts (`q36_expert_gate_up_mm`, `q36_expert_down_mm`) on the GPU's
matrix operations. A threadgroup computes 32 rows by 128 columns of a
projection, or up to 64 choices of an expert by 64 columns, and every lane
reads its weights straight from memory into the matrix operand, so only the
rows of inputs go through threadgroup memory; an expert's last tile computes
only the blocks of 8 rows it has, and an expert of up to 64 choices decodes
its int4 weights once. Where a KV head serves eight query heads, the attention
takes the 8 query heads of 4 rows as the 32 rows of the matrix and reads keys
and values the same way (`q36_attn_qk8`, `q36_attn_pv8`): the scores' kernel
also keeps each row's maximum over every 32 positions, and the values' kernel
takes the exponentials while it stages them and divides by their sum at the
end, so the softmax has no pass of its own over the scores. Other shapes use
`q36_attn_qk`, softmax and `q36_attn_pv` in 16-by-32 tiles. The expert groups
whose weights are loaded run in one concurrent region: every gate/up, a
barrier, every down. The batch's DeltaNet recurrence runs in chunks of up to
32 rows as float matrix products (`q36_dn_chunk_*`), where the exact kernel
takes one token after the other with compensated sums: what does not depend on
the state (the decay products, the chunk's key and query products and the
inverse of its triangular matrix) is prepared for eight chunks at once, then
one dispatch per chunk reads the state, writes the chunk's outputs and the new
state. Its scratch holds the eight prepared chunks, 10 MiB on the 35B model.

Generation has fast kernels too: the dense projections, the router and the
shared expert read four inputs per SIMD lane (`q36_dot_i8_vec4`), the DeltaNet
runs in plain float with SIMD sums (`q36_dn_scan`), and from 128 tokens of
context the eight query heads of a KV head are the rows of a matrix for the
attention (`q36_attn_gqa_*`). RMS norms sum with SIMD reductions in both; a
norm of a few long rows takes a threadgroup of 256 threads per row
(`q36_rmsnorm_wide`, one SIMD group took 50 us for a row of 2048), and the
router's softmax adds its exponentials in parallel (`q36_router_softmax_fast`).

On Apple silicon a fast prompt batch of 32 rows or more also uses the CPU's
matrix unit, through Accelerate's BLAS (`qwen36_cpu_mm.h`). The CPU computes
the products the GPU's next steps do not read: DeltaNet's z projection while
the GPU runs qkv and the recurrence (only the gate after them reads z), the
gate half of the attention's q projection while the GPU runs the query half,
k, v and the attention (the matrix kernel takes the first half of each head's
weight rows and writes the queries in place), and the shared expert while the
GPU runs the routed experts. It also takes routed experts: on a real text the
router is skewed, and in a batch of 2048 rows a few experts often get hundreds
of choices, sometimes every row. The CPU decodes their int4 weights to floats
(about 0.3 ms an expert) and computes those with the most choices, as many as
by measured estimates it finishes before the GPU's share (`qg_cpu_experts`).
In the later batches of a long prompt it also computes the attention of the
batch's last rows, from the queries and the cache the GPU wrote, while the GPU
computes the others (`qg_cpu_attn_rows`: 320 to 384 of 2048 rows at 2048 to
8192 positions), and there, in batches of 1024 rows or more, the last 640 of
every 2048 columns of each layer's output projection (`qg_out`). Every split
between the two follows the data alone, never their timing: they round
differently, and a split that followed which side was ready first would
change the results from run to run. On the M1 that unit does about 60% of
what the GPU's matrix kernels do, and running both at once did not slow the
GPU: an 8192-token prompt of the 35B model ran 23% faster. That is for a
prompt on a cool machine: a fanless MacBook Air cannot shed the heat of both
for long, and after about a minute of continuous work (the third 8192-token
prompt in a row) it slows down to the speed of the GPU alone, never below.
The converted float weights and the scratch take about 110 MB while a prompt
runs.
`QWEN36_CPU_MATMUL=0` leaves every product on the GPU.

Where a KV head serves eight query heads, as on the 35B model, fast mode keeps
attention keys and values in half precision: they are rounded when written to
the cache, and the prompt and generation kernels read halves and compute in
float (`q36_attn_*_half`). Rounding moves a value by up to 2^-11 of its size,
more than any fast sum, and halves the bytes attention reads from the cache. At
8192 tokens of the test text the largest key or value is about 20, far below
the half-precision limit of 65504.

Their sums run in another order, so the results differ from the exact ones in
the last bits (BLAS too, with each output's scale folded into its weights):
about 2e-7 of the magnitude of the summed terms for a projection, up to 2e-6 of the largest output for the experts and the DeltaNet,
a few 1e-5 of the output for attention, which has no running maximum or
compensated sums there and takes the GPU's fast exponentials in the prompt.
The router can then swap two nearly tied experts. Shapes these kernels do not
support keep the exact ones.

Two exact kernels are shaped for speed without changing a bit: in generation
the exact DeltaNet reads the state twice (`q36_dn_delta_row`) instead of
keeping a 128-float column per thread, and the four-tap convolution of a prompt
keeps its history in registers.

`QWEN36_METAL_EXACT=1` keeps the exact kernels everywhere. Their sums run in
one fixed order, so the output does not depend on how a prompt is split into
batches or chat turns, and a change that should not alter the results is
checked bit for bit in this mode. Neither mode gives the CPU engine's bits: the
graph keeps the activations in float, where the CPU engine quantizes them to
int8 for its integer dot products (`COLI_DENSE_IDOT`, `QWEN_EXPERT_ACT`). Over
1024 tokens of a known text after a 64-token prompt, each read with the true
tokens before it, the mean negative log-likelihood is 2.945 in fast mode, 2.948
in exact mode and 2.948 on the CPU engine; the model ranks the true token first
46.6%, 46.3% and 46.3% of the time.

## Where the time goes

`COLI_TIMERS=1` prints, with the CPU engine's labels, the GPU time of
DeltaNet, attention, the MoE (router and shared expert included) and lm_head.
The GPU reports the time of each batch itself: nothing waits for it, though
each phase ends a GPU batch of its own (the GGUF record's timed runs decoded
1 to 7% slower than its speed runs, which also differed in other ways). The
MoE time also holds the wait for the experts read from disk, which
`expert kernel: fetch | compute` shows apart from the GPU work on the experts
computed after it. In decode the GPU runs the shared expert during that wait,
so its time is counted twice, and the experts the cache holds, untimed; in a
fast prompt it works during part of the wait. So the phases can add up to
more than the elapsed time, and they leave out some of the GPU's work.
`outside the phases` is the rest: work on the CPU, and the GPU idle while it
waits for the CPU.

On a GGUF most of a decode step's extra time is that wait for the disk: on
the 35B model's UD-Q4_K_M after a 512-token prompt on an 8 GB M1, 70.0
against 46.5 ms a step on prose, of 135.2 against 103.4 ms for all the phases
(steps 2 to 32 of one timed run per file, less a first step timed in another
run). On the three texts the decode read 1.6 to 6.1% more experts than on the
model's own format, each 14% larger, and its wait grew 40 to 50%; a GGUF
expert is also three reads instead of one (its gate, up and down are three
tensors of the file; one read with `coli import --mode copy`), which was not
measured apart
([the experiment's record](experiments/qwen36-gguf-metal-2026-10-19/README.md)).

## Tests

```sh
make -C c test-qwen36-kernels          # every kernel against a double-precision reference
make -C c test-qwen36-graph METAL=1    # the graph against the CPU engine, on a model it writes
make -C c test                         # includes the qwen36 component tests
```

`tests/test_qwen36_graph.py` writes a random model of the graph's geometry (8
layers, hidden 256, 64 experts of which 4 per token, eight query heads on one KV
head of 256, DeltaNet heads of 128) and checks: the exact graph against the CPU
engine in the same arithmetic (`COLI_DENSE_IDOT=0 QWEN_EXPERT_ACT=f32`) with an
expert cache too small for the model, a 1600-token prompt in two batches, the
fast graph against the exact one, the CPU's share of a fast prompt against
`QWEN36_CPU_MATMUL=0`, a chat turn that resumes from the serve copy against a
fresh engine, a NaN before the router, `Q36_MTP=1` (the other engines' head, not
loaded), and the steps aside to the CPU engine, a Clef decision head among them.
With an MTP head it writes too, one that drafts the token just picked again, it
checks the drafts: in exact mode the tokens and logits of plain decoding with
every draft verified, some right and some not, the same in a serve turn resumed
from the copy before the head's window and in the turn after one that ended
inside a verify, the head's logits (`Q36_MTP_DUMP`) the same bits with its
window in one batch or in two, the steps it does not run where most tokens lie
past the ids it drafts from (`Q36_MTP_VOCAB`) and the prompts that skip it,
and fast experts in a verify.
Without the head, prompt lookup's forced drafts (`COLI_LOOKUP_FORCE`) check
verifies of up to six rows against plain decoding. On the Apple Paravirtual
device it checks only the steps aside.

It also writes the same kind of model as a GGUF, in the form `coli import`
gives a model (`gguf.json` beside the file), as llama.cpp's converter writes
`qwen35moe`: full-gamma norms, A instead of A_log, two DeltaNet key heads for
four value heads in the file's tiled order, experts 256 wide, the dense
matrices in Q8_0 and the experts' gate and up in Q4_K and down in Q5_K, as a
Q4_K_M file has them, with every other type the graph reads in some matrices
and layers, and `.scale` companions on a few. It checks the exact graph against
the CPU engine on the same file (`GQ_EXACT=1`), the fast graph against the
exact one, fast experts, a serve turn resumed from the copy, verifies of
drafts forced in ref.json mode (`COLI_LOOKUP_FORCE`: nothing drafts on a GGUF)
with plain decoding's bits, on a copy of the file whose router bias puts four
experts of a layer first, so a verify reads that layer's experts in one lease
(the 35B model's files have no bias), the line and the CPU engine with a dense
matrix of Q4_0 and with experts of IQ3_S, the layout of
`coli import --mode copy` (the same bits as the file it was copied from), the
header check (`GGUF_CHECK=1`), and that the model's own format's options
change nothing there.

`DUMP_LAYERS=<file>` writes the projection, residual and MoE output of every
layer for the last token, and `DUMP=<file>` the last logits. Two builds can be
compared layer by layer this way.

## Limits

- The routed experts must be in the planar INT4 format with groups of 64, or
  a GGUF's blocks of the types in [GGUF](#gguf).
- Exact mode keeps attention keys and values and the DeltaNet state in
  float32; fast mode keeps keys and values in half precision where a KV head
  serves eight query heads. The KV cache grows on demand.
- A `STOP` or `CANCEL` sent during a prompt takes effect once the prompt is
  read: the graph does not poll controls between layers as the CPU engine does
  ([serve_protocol.md](serve_protocol.md)).
