# Speculative decoding: MTP drafts and prompt lookup

Speculative decoding guesses the next few tokens cheaply, then checks them all in one
forward of the model. Tokens are selected from the main model's verified logits, never
accepted solely on the draft's prediction. With the same arithmetic and execution
placement, the output matches plain decoding; the number of forwards changes. Dynamic
CPU/GPU placement can change rounding and therefore occasionally a greedy token, as
described below.

What runs where:

| Engine | Draft sources | Verify |
|---|---|---|
| `qwen38` (Qwen3.8 Flash Next) | the checkpoint's MTP head, up to 3 drafts (on by default when the checkpoint has it; `Q38_MTP=0` off, `Q38_MTP_DRAFTS`); prompt lookup, up to 5 (on by default; `COLI_LOOKUP=0` off) | up to 6 rows, on the CPU or in the Vulkan dense chain |
| `qwen36` (Qwen3.6, Qwen3-Coder, Qwen3.8-27B) | prompt lookup, up to 5 (on by default; `COLI_LOOKUP=0` off) | up to 6 rows, on the CPU or in the Vulkan dense chain |
| `colibri` (GLM-5.2) | its MTP head (`DRAFT`), carried by the gate below (`COLI_SPEC_GATE=0`: the acceptance guard of `COLI_MTP_GUARD_PCT` instead); without a head, the n-gram drafts come from the prompt lookup below and its gate by default (`COLI_LOOKUP=0`: its own n-gram drafts as before) | as before |

## How a verify works

The caller picks a token and asks for the logits after it. A draft source proposes `k`
tokens after that one, and one forward over the token and its `k` drafts (the verify,
`k + 1` rows) returns the first row's logits, the ones a plain step would give, and
keeps the others. The caller's next picks settle the drafts one at a time:

- a pick that equals the next draft is answered from that draft's row, with no forward;
- the first pick that differs ends the verify: the rows after the ones that stood are
  undone, and the next verify starts from the token the caller picked.

The picks come from the exact logits, greedy or sampled. A draft never reaches the
sampler: a sampled token that happens to equal a draft is an accepted draft.

**Undoing rows on a model with recurrent state.** Qwen3.6 and Qwen3.8 are hybrids: three
of every four layers are Gated DeltaNet, which keeps a recurrent state rather than rows
indexed by position, and Qwen3.8 adds the PLE convolution's ring. Such a state cannot be
rewound. So a verify copies the DeltaNet states, the convolution rings and the PLE ring
(with its n-gram history) after each of its rows but the last, one copy slot per row. A
rejection after row `r` swaps slot `r` in (pointers, nothing recomputed). The attention
and indexer rows past the standing rows stay where they are as a stale tail that the
next forward overwrites, as every rewind in these engines treats them. `kv_len` and the
`kv_prefix` record come down to the standing rows. In the Vulkan chain, the KV mirror's
watermark and the pooled-key watermark come down too.

The copies cost memory, allocated the first time a verify that deep runs:

| Model | One copy slot | Slots for 3 MTP drafts | Slots for 5 lookup drafts |
|---|---|---|---|
| Qwen3.8 Flash Next (36 DeltaNet layers, 48 value heads of 128 x 128) | 113 MiB | 3 (338 MiB) | 5 (563 MiB) |
| Qwen3.6-35B-A3B (30 DeltaNet layers, 32 value heads of 128 x 128) | 63 MiB | | 5 (314 MiB) |

The chain keeps its own copies on the device, the same size.

**Every verify row gets a decode step's bits.** The CPU kernels compute each row the
same way whatever the row count. A device matrix runs a verify's rows one at a time
(`q38_weight_matmul`, qwen36's `vk_dense_matmul`), and the chain takes the per-row GEMV
for every matrix of a verify. The Vulkan expert tier sends at most 15 rows of an expert
through its per-row route, so the 6 rows of a verify always take it. `lm_head` reads
each row on its own. In the chain, one copy is written by the recurrent shaders as they
pass its row. More copies split the convolution and the recurrence into one dispatch
per copy, each ending on its row. The state crosses between them through memory in f32,
so the rows get the bits of the single dispatch.

Drafting stays off where a row's result depends on what is resident: qwen36 under a
CUDA expert tier holding part of the experts, `CACHE_ROUTE` or a qpack container (with
`COLI_LOOKUP=1` asked for, a `[qwen36] COLI_LOOKUP=1: no drafts under ...` line says so;
the default stays quiet). A tier holding every routed expert, run by the CUDA chain, drafts:
nothing there depends on residency, and the chain's verify rows get a decode step's bits.
On an RTX PRO 4000 (Qwen3.6-35B-A3B int4-g64) a code edit of 400 tokens took 3.25 s
against 4.40 s without lookup, 2.91 tokens per forward, the same text; on a chat answer
the gate declined almost every proposal (98.2-98.8 against 97.8-98.3 tok/s). One case stays inside the gate but changes the last bits: with the
Vulkan expert tier on, a verify's rows, the rejected drafts' included, feed the tier's
history, so which experts are on the device follows what was drafted. An expert on the
device sums in a different order than on the CPU, so the logits of a run with drafts and
one without can differ in their last bits, as two runs with different histories do.
When two logits are close, this can also change a greedy token. The tiny tests compare
tokens there, and drafts against no drafts byte for byte with the tier off; they do not
establish universal token identity with dynamic tier placement on real checkpoints.
A real FP8 Qwen3.8 check produced different text with the adaptive tier, but identical
text with zero, two and three MTP drafts when the starting history, explicit tier budget
and placement were fixed (`COLI_VK_TIER_GB=4`, `COLI_VK_TIER_RATE=0`,
`COLI_VK_TIER_BALANCE=0`, `COLI_VK_TIER_STREAM=0`). This is a diagnostic comparison,
not a recommendation to freeze the tier for normal use.

## The MTP head's drafts (qwen38)

The engine loads the head by default when the checkpoint has one and the whole model is
loaded, and says so when it cannot; `Q38_MTP=0` leaves it unread, and `Q38_MTP=1` refuses a
checkpoint without one (see [qwen38.md](qwen38.md#speculative-decoding-with-the-mtp-head)).
With the Vulkan dense chain running every layer and the lm_head on the device, the head
stays off unless `Q38_MTP=1` asks for it. With it, the head's layer runs on the device
too (`q38c_mtp_rows`): its input (the embedding's norm and `fc_embedding` added to each
stream's norm and `fc_hidden`), its attention over its own K/V mirror (the KV split's last
attention ordinal, its own RoPE base), its routed experts in the frame once the tier holds
all of them (the tier keeps an MTP head's experts whole there, `VktConfig.extra_whole`),
its mixer and the lm_head. `Q38_MTP_CHAIN=0` runs the head's layer on the CPU instead,
once per draft and once per accepted row.

The head still does not pay for itself beside the full chain: a verify of three rows costs
about two and a half plain steps there (each row routes its own experts), and the head
reads every prompt row. An RTX PRO 4000 with Qwen3.8-Flash-Next IQ2_XS, a 507-token
prompt, two drafts, 2.89 tokens a forward (95% of the drafts accepted), the three settings
started in turn twice (prefill / decode, tok/s):

| | first | second |
|---|---|---|
| the head off (the default) | 88.8 / 33.8 to 35.3 | 94.2 / 37.1 to 38.0 |
| the head's layer on the CPU (`Q38_MTP_CHAIN=0`) | 76.8 to 80.2 / 26.7 to 30.5 | 72.0 to 76.4 / 24.8 to 27.9 |
| the head's layer on the device | 71.7 / 29.0 to 32.4 | 77.7 / 35.6 to 40.1 |
Its first draft reads the model's four streams at the last fed position with the picked
token. Each next draft reads the head's own streams from the row before, in place of the
model's (only the verify computes those), with the draft just proposed. Those deeper
rows write the head's K/V and indexer rows past its settled ones, a tail that the
settled pairs overwrite once the verify has run. After the verify the head takes the
standing rows' streams, each with the token that followed it.

`Q38_MTP_DRAFTS` sets the drafts per verify: `1`, `2` or `3` drafts every time, or `0` /
`auto` to let the gate pick 0 to 3 per verify. The default is 2, the depth measured
fastest (below), and 1 on an integrated GPU with 32 or more compute units, the expert tier
and the dense chain on ([vulkan.md](vulkan.md#the-chain-on-a-radeon-780m)). `Q38_MTP_DRAFTS=1` is the
one-draft verify that came before.

### The draft vocabulary

For a draft the head scores a list of tokens instead of the 248,320 of the vocabulary. The list
has the `Q38_MTP_VOCAB` tokens with the highest counts (`16000` by default), every token with a
count of at least 3/4 of the count of the last of those, and every token of the current
conversation. Every token read as part of a prompt (past the part restored from the prefix cache)
and every token picked adds 1 to the count of that token. All counts halve every 1024 counted
tokens. The list is rebuilt when a request starts and at every halving. A picked token missing from
it joins it at once.

The counts are written to a file in the model directory when a run ends or serve receives SIGTERM,
and the next run starts from them. `COLI_VOCAB_USAGE` sets another file and `USAGE_SAVE=0` skips the
write. A file that cannot be read as counts, has counts for another vocabulary size or ends on a
damaged record is not replaced. The list is off when `COLI_VOCAB_USAGE` and `COLI_USAGE` are the
same file.

The output does not change: the verify step checks every token, and a token missing from the list
can only turn an accepted draft into a rejected one. The CPU scores the listed rows from the int8
rows of the `lm_head`, so `COLI_VK_DENSE_HOST=0` does not release those rows from host RAM. The
head scores the whole vocabulary with `Q38_MTP_VOCAB=0`, with `Q38_MTP_DUMP`, with
`Q38_DRAFT_HEAD=int4`, without int8 rows on the host (for example with `Q38_TRUNK_CPU_INT8=0`), and
while the dense chain is on, where the device scores it. Each run and served turn prints the
average size of the list, the share of picks inside it, and the ids appended and rebuilds.

Qwen3.6's head uses the same list with `Q36_MTP_VOCAB`, on by default where it drafts on the CUDA
chain ([qwen36.md](qwen36.md#the-mtp-head-q36_mtp)); its engine writes the counts at exit, not on
SIGTERM.

## The MTP head's drafts (qwen36)

`Q36_MTP=1` reads the release's `mtp.*` block from the converted container
([conversion and settings](qwen36.md#the-mtp-head-q36_mtp)). Its input pairs
the backbone's output **after its final norm** with the next token's embedding.
The head has its own causal KV cache; every completed pair contributes its K/V,
and only a draft row runs the full attention/MoE/output computation.

`Q36_MTP_DRAFTS=1..7` bounds the depth, `auto` lets the measured gate choose,
`Q36_MTP_PMIN` stops uncertain deeper drafts, and `Q36_MTP_VOCAB` (the most
used tokens, on by default on the CUDA chain) or `Q36_MTP_VOCAB_IDS` limits
only the draft's output scoring. When lookup also proposes tokens the gate
chooses a source by its measured value. Both sources share verification and
rollback; `[qwen36 mtp]` reports the head's acceptance separately. A rejected
row or an unconsumed tail never commits to the backbone state. Pins preserve
the pending residual as well as the backbone recurrence.

`make -C c qwen36-tiny-mtp-check` checks reference logits, exact output and final
logits at every depth, forced rejection/acceptance/mixed rows, and multi-turn
serve including prefix reuse, pin restore and sampling. The reference uses
Qwen's Transformers decoder with the converted int8 or int4-g64 weights.

## Prompt lookup

Prompt lookup (on by default; `COLI_LOOKUP=0` turns it off) looks for the longest n-gram of
4 down to 2 tokens that ends the token
history (prompt and output so far) and also occurs earlier in it, at its most recent
earlier occurrence. The up to 5 tokens that followed it there are the proposal
(`COLI_LOOKUP_DRAFTS` caps it, 1 to 5). Code edits, quotes and repeated structure give
long accepted runs; free text gives few proposals, and those few are short. On qwen38
with the MTP head, a verify carries whichever proposal has the higher expected tokens
per unit of time, the longer one on a tie.

## The gate

`spec_draft.h` decides how many drafts a verify carries. Per source it keeps the running
probability `a_j` that draft `j` is accepted when the drafts before it were (an average
over about 16 verifies, from priors of 0.8, 0.6 and 0.5 for the MTP head and 0.5 for
lookup), the measured seconds `T(n)` of a forward of `n` rows, and the measured seconds
`d` a source spends per draft past its first (the MTP head's deeper drafts). The depth
it picks maximizes

```
value(k) = E(k) / cost(k)
E(k)     = 1 + a_1 + a_1 a_2 + ... + a_1 ... a_k        tokens a verify of k drafts yields
cost(k)  = (T(k+1) + (k-1) d) / T(1)                    in plain decode steps
```

and it drafts only when that beats a plain step (`value(k) > 1`). A row count not
measured yet costs `T(1) (1 + (n-1) rho)`, with `rho` fitted from the multi-row forwards
that ran once `T(1)` is measured (0.5 until then). Every 16 picks it tries one draft
deeper than its best, so an estimate it stopped feeding can recover. After 64 forwards
without a plain step it takes one, so `T(1)` stays measured. A fixed depth of 2 or more
whose drafts are trimmed (below) takes one too, because the trim prices rows with `T(1)`
and `rho`. Prompt lookup is always gated. The MTP head is
gated under `Q38_MTP_DRAFTS=0`. colibri's MTP head (GLM-5.2) is always gated: its head
runs a layer of its own for each draft and absorbs the drafts that stood, so a verify's
`T(n)` counts the forward, the head's drafts and that absorb. `COLI_SPEC_GATE=0` drafts
every proposal in full (the tests use it), and on colibri brings back the acceptance
guard that pauses the head for 256 tokens under 70% (`COLI_MTP_GUARD_PCT`,
`COLI_MTP_GUARD_WINDOW`).

**Each step's own drafts.** The averages above give every step the same depth whatever
its drafts look like. A source that can say how likely each of this step's drafts is
passes that estimate `p_j` to `spec_gate_pick_p`, and `a_j` in `E(k)` becomes the
acceptance the gate measured for drafts given that estimate (ten bins of `p`, each
starting at the bin's own `p` and worth two verifies, then fed by every verify). Those
`k` drafts are already written, so `spec_gate_pick_p` prices only their verify
(`T(k+1) / T(1)`), not the time spent writing them. Before the head writes draft `n + 1`,
`spec_gate_deeper_p` lets it write that draft only when no shorter verify is already
worth more than `n` drafts and `n + 1` drafts, the new one as likely as the gate's best
bin, would beat both a plain step and `n`. The time of that draft is not counted there:
it stops only drafts that could not pay for their verify row. A step whose drafts are
sure verifies deeper, a doubtful one stops early. This is DeepSeek's
confidence-scheduled verification, and the estimate is:

- for the MTP heads, the head's probability of its own token (the softmax over the scores
  it computed, those within 20 of the best): Qwen3.6, Qwen3.8 and colibri's GLM-5.2;
- for DeepSeek V4.1's DSpark, the sigmoid of its confidence head
  ([deepseek-v41.md](deepseek-v41.md#dspark-drafting-and-why-accepting-a-draft-is-safe)).

Qwen3.6 and Qwen3.8 keep the first draft (the head pass that wrote it is one a plain step
makes too), trim the deeper ones within a fixed depth (`Q36_MTP_DRAFTS`, `Q38_MTP_DRAFTS`
from 2), and stop writing drafts when no next one could be worth its verify row
(`spec_gate_deeper_p`). Under `auto` the averages pick the
depth alone: fed by the verifies the estimates trimmed, which kept the sure drafts, they
learned a higher acceptance than the head has and picked deeper. GLM-5.2's head runs a
layer of its own per draft, so there the probability can also turn the verify into a
plain step: on a model read from the disk the verify's second row brings experts of its
own.

It is on by default for Qwen3.6 (`Q36_MTP_PSTEP=0` turns it off) and opt-in for Qwen3.8
(`Q38_MTP_PSTEP=1`) and GLM-5.2 (`MTP_PSTEP=1`). Measured on an RTX PRO 4000 (24 GB),
`coli serve`, 128 tokens, two rounds of three prompts, each setting in turn (decode
tok/s; every Qwen3.6 run printed the text of the run without the head, prompt by prompt):

| Qwen3.6-35B-A3B int4-g64, its head | CUDA chain | Vulkan chain |
|---|---|---|
| no head | 99.7-104.5 | 83.2-83.8 |
| 1 draft (the default; nothing to trim) | 115.8-120.1 | 99.9-109.3 |
| 2 drafts, `Q36_MTP_PSTEP=0` | 94.6-108.6 | 100.4-103.7 |
| 2 drafts, per step | 104.2-111.9 | 101.5-107.2 |
| 3 drafts, `Q36_MTP_PSTEP=0` | 88.6-95.2 | |
| 3 drafts, per step | 104.9-110.2 | |
| `auto` | 114.7-118.8 | |
| `auto` with the per-step depth (not what ships) | 106.0-120.6 | |

So one draft stays the fastest depth on that card, and the per-step depth takes most of
the loss out of a deeper one. Those runs predate the plain step every 64 forwards at a
fixed depth and the check before each draft: on an L4 with every expert in VRAM, that
change made two drafts 9-15% faster
([qwen36-mtp-fixed-depth-2026-10-25](experiments/qwen36-mtp-fixed-depth-2026-10-25/)).
On Qwen3.8 Flash Next IQ2_XS with its Q8_0 head, Vulkan chain, two drafts, it measured
1 to 5% slower (33.9-38.0 against 34.1-39.0 tok/s; the head itself is off by default
beside that chain, 35.8-49.6 without it), hence opt-in there (measured before that
change, not since). GLM-5.2 has not been measured on a real checkpoint with it.

A step that does not run colibri's head (a plain step, or another source's verify)
leaves the head's KV row for that position unwritten. The head's window then restarts
at its next draft, or at the next rows it absorbs, instead of attending a row that
holds nothing of this text.

When the gate decided anything, a run prints what it measured on a `[qwen38 spec gate]`
or `[qwen36 spec gate]` line: `T(1)`, the cost of each row count it saw in units of
`T(1)`, `rho`, the MTP head's cost per extra draft and the acceptance by position.

## Report lines

Every run, and every served turn, prints its counts on stderr (never on the wire):

```
[qwen38 MTP] run: 2.75 tokens/forward (36 forwards per 99 tokens) | acceptance 90.0% (63/70 drafts) | wiring b | fixed depth, by position: 1 97.1% (34/35) 2 82.9% (29/35) 3 0.0% (0/0)
[qwen36 lookup] run: 1.56 tokens/forward (102 forwards per 159 tokens) | acceptance 85.1% (57/67 drafts in 64 verifies) | gate on, 32 declined, 5 probes
```

Tokens are the decode tokens fed after the prompt, and forwards are the forwards that
fed them. The acceptance by position counts a draft when its verify carried it. `ended
mid-verify` marks a turn that stopped with drafts unconsumed: they were undone, and the
state is the one plain decoding leaves.

## Settings

| Variable | Default | Effect |
|---|---|---|
| `Q38_MTP_CHAIN` | on | qwen38 with the MTP head on beside a dense chain that holds every layer: the head's layer runs on the device (`q38c_mtp_rows`); `0`: on the CPU. |
| `Q38_MTP` | on when the checkpoint has the head, off beside a dense chain that holds every layer | `0`: plain decoding, the head unread; `1`: the head, and a checkpoint without one is refused. The RAM plan (`coli plan`) prices the head when it is on: its dense tensors and its own expert cache (`Q38_MTP_CAP`, the layers' cap by default and every expert of the head where the engine sets that, at the head's FP8 record). |
| `Q38_MTP_DRAFTS` | `2`; `1` on an integrated GPU with 32 or more compute units, the expert tier and the dense chain on | qwen38 with the MTP head on: the MTP head's drafts per verify, `1` to `3`, or `0` / `auto` for the gate's pick. |
| `Q38_MTP_VOCAB` | `16000` | qwen38 with the MTP head on: how many of the most used tokens the head scores for a draft. `0`: the whole vocabulary. |
| `COLI_LOOKUP` | `1` | Prompt-lookup drafts (qwen38, qwen36), and colibri's n-gram drafts from the same lookup and gate when it has no MTP head; always gated. `0` turns them off. |
| `COLI_LOOKUP_DRAFTS` | `5` | Lookup drafts per verify at most, `1` to `5`. |
| `COLI_SPEC_GATE` | `1` | `0`: every proposal drafted in full, the gate's estimates ignored (tests). |
| `Q36_MTP_PSTEP` | `1` | qwen36: within a fixed `Q36_MTP_DRAFTS` of 2 or more, the verify's depth from each step's drafts' probabilities under the head (see the gate). `0`: the fixed depth. |
| `Q38_MTP_PSTEP`, `MTP_PSTEP` | `0` | qwen38 (within a fixed `Q38_MTP_DRAFTS` of 2 or more) and colibri (GLM-5.2's head): `1` takes the verify's depth from each step's drafts' probabilities under the head too. |
| `Q38_MTP_FORCE`, `COLI_LOOKUP_FORCE` | unset | Tests: in `ref.json` mode, `accept` drafts the reference's tokens, `mixed` makes every other verify's first draft wrong, `cycle` moves the wrong draft over the rows verify by verify, `rowN` makes draft `N` wrong; `Q38_MTP_FORCE=reject` also refuses every draft. |

## Measured

On `ds`: Ryzen 7 PRO 8700GE (8 cores, 16 threads), 61 GiB, NVMe, Radeon 780M (RADV),
`OMP_NUM_THREADS=8`. Every run started with the 1-minute load under 2 and no other
engine running. Qwen3.8 runs started after the model files were dropped from the page
cache. Qwen3.6's container (22 GB) stayed in the page cache, and its numbers are two
separate rounds. The same binary ran every arm of a table. With drafts or without,
every run printed the same text (the same md5 within each prompt). These are single
runs. Two runs repeated later in the same conditions (Qwen3.8, CPU: no MTP 4.12 tok/s,
one draft 4.69) put the run-to-run spread at about 0.5%.

**Qwen3.8 Flash Next, int4-g64 sidecar, chat prompt with thinking, 100 tokens.**
The CPU runs use cap 170. The tier-and-chain runs (`COLI_VULKAN=1 COLI_VK_CHAIN=1`) use
cap 96, each from a copy of the same expert history (one unrelated 40-token run).

| MTP | CPU tok/s | tier + chain tok/s | tokens/forward | acceptance by draft position |
|---|---|---|---|---|
| off | 4.12 | 3.91 | 1.00 | |
| 1 draft | 4.71 (+14%) | 4.39 (+12%) | 1.94 | 94.1% |
| **2 drafts** | **4.94 (+20%)** | **4.55 (+16%)** | 2.75 | 97.1%, 82.9% |
| 3 drafts | 4.83 (+17%) | 4.42 (+13%) | 3.30 | 93.1%, 79.3%, 65.5% |
| the gate's depth | 4.76 (+16%) | 4.41 (+13%) | 2.02 | 95.9%, 75.0% (4 verifies of 2) |

The head's acceptance falls with depth: each draft past the first reads the head's own
streams instead of the model's. Tokens per forward still rise to 3.30 at three drafts,
but the speed peaks at two. A verify of `k` drafts reads the trunk once, but every row
routes its own 10 experts per layer (the cache hit rate falls from 81% without MTP to
76% at two drafts), and a rejected row's expert reads are spent all the same. At three
drafts the third row stands 65% of the time and costs more than it returns. The
default is therefore 2 (`Q38_MTP_DRAFTS=2`) on the CPU and in the chain alike, measured on a Radeon 780M.

**Prompt lookup on a code edit.** The prompt asks to rename a variable in 23 lines of C
and print the code again (`enable_thinking: false`). 160 tokens; Qwen3.8 at cap 170 on
the CPU, Qwen3.6-35B-A3B (int4 gs64 container) at cap 256 on the CPU.

| | Qwen3.8 tok/s | tokens/forward | Qwen3.6 tok/s (two rounds) | tokens/forward |
|---|---|---|---|---|
| no drafts | 2.91 | 1.00 | 7.40, 7.44 | 1.00 |
| lookup (`COLI_LOOKUP=1`) | 3.11 (+7%) | 1.56, 83.8% of 68 drafts | 7.89, 7.90 (+6%) | 1.56, 85.1% of 67 drafts |
| MTP, 1 draft | 3.20 (+10%) | 1.99, 100% | | |
| MTP, 1 draft + lookup | 3.19 | 1.99 (lookup never chosen) | | |
| MTP at the gate's depth + lookup | 3.30 (+13%) | 2.74, 100% | | |

**Prompt lookup on a chat prompt** (the thinking chat above for Qwen3.8; a chat prompt
without thinking for Qwen3.6). Free text gave a few short proposals, and the gate
declined every one (11 on Qwen3.8, 12 on Qwen3.6), so no verify ran. Speed was the
plain run's within the spread: Qwen3.8 4.16 against 4.12 tok/s, Qwen3.6 10.38 and 10.43
against 10.37 and 10.47.

So lookup pays where the output repeats the context (+6 to 7% on the edit) and costs
nothing where it does not. With the MTP head attached, the head's drafts were worth more
than lookup's on this edit (100% acceptance). Since it costs nothing where it
does not pay, lookup is on by default from 1.13.0, like the MTP head; `COLI_LOOKUP=0` turns
it off. A turn's `PROF` frame counts the forwards it took, which drafts change.

**Vulkan follow-up, existing int4-g64 sidecar.** On the same 780M, with the code-edit
prompt above (241 prompt tokens), 128 generated tokens, cap 170, eight threads and
the same starting expert history, all eight runs below produced byte-identical text.
Each configuration ran once, after advisory eviction of checkpoint file pages, with
no other inference process running. These are the engine's `Speed` values: generated
tokens divided by **prefill plus generation time**, excluding model load. They are
not steady decode throughput.

| Draft source | Expert tier, CPU dense work | Expert tier and Vulkan dense chain |
|---|---:|---:|
| off | 2.90 tok/s | 2.68 tok/s |
| lookup, up to 5 drafts | 3.24 tok/s | 2.85 tok/s |
| MTP, 3 drafts | 3.49 tok/s | 2.97 tok/s |
| MTP 3 and lookup | 3.47 tok/s | 3.00 tok/s |

MTP accepted all 95 judged drafts in each of its runs: 32 forwards for 127 decode
tokens, or 3.97 tokens per forward. Lookup alone accepted 58/66 drafts on the tier
and 77/102 in the chain; the adaptive gate chose different depths. With both sources
enabled, every verify used MTP and lookup supplied no drafts. Their gains therefore
do not add. The small differences between MTP alone and the combined mode are not
evidence of a lookup benefit.

A subsequent three-run comparison on commit
`2f33f211ea6e28186d62cf9c00d511df826590a0` isolated `Q38_MTP=0` versus `1`, with
`Q38_MTP_DRAFTS=3` fixed and the expert-tier configuration above. The samples were
2.93, 2.93 and 2.90 tok/s without MTP, and 3.49, 3.50 and 3.47 with it: median
2.93 to 3.49 tok/s (+19.1%, including prefill). All six outputs were byte-identical.
The [validated experiment manifest](experiments/vulkan-qwen38-mtp-2026-10-05/manifest.json)
links the raw logs and records the command, hardware, cache policy, prompt and starting
history. It measures MTP on this workload, not the combined effect of every Vulkan
change.

The tier-only run took 44.2 seconds without drafts and 36.7 seconds with MTP 3;
time to first token was 16.99 and 17.44 seconds respectively. Subtracting those rounded
times gives approximately 4.67 and 6.59 tokens/s after the first token. The dense
chain took 47.7 and 43.0 seconds, with time to first token of 13.35 and 14.08 seconds.
It released 4.07 GiB of dense host copies without MTP, 4.16 GiB with MTP, and reloaded
none for CPU fallback. Lower memory use and faster prefill need not imply faster
decode on a shared-memory GPU.

The MTP layer still follows the host's control and attention path. In the device-only
dense configuration its dense matrices dispatch on the GPU, while its routed experts
remain in their native FP8 format on the CPU; the model's existing int4-g64 sidecar
does not include that layer. This is not a fully device-resident drafter.

Not measured: a long context with speculation, a model whose routed experts all fit
in RAM, and any discrete GPU (none was available).

**GLM-5.2's MTP head under the gate.** GLM-5.2 UD-IQ1_S (202 GB) on an RTX PRO 4000
(24 GB) with 62 GB of RAM, the Vulkan chain and tier, most experts from the disk, a
96-token answer, servers alternated:

| | decode tok/s | |
|---|---|---|
| the gate | 0.681, 0.704 | drafting all along: 75-81% accepted, a verify of 2 rows 1.50-1.65 plain steps |
| `MTP=0` | 0.678, 0.712 | |
| the acceptance guard (before) | 0.638, 0.646 | 67% over the first 24 proposals, then paused |

On this machine a verify's second row brings experts of its own from the disk, so a draft
returns about what it costs: the head comes out even with plain decoding, where the
guard's first window lost 4%.

## Another engine with a verify path

`spec_draft.h` holds no engine state. An engine that verifies drafts calls:

- `spec_lookup(history, len, 2, 4, k, out)` for a proposal from the context;
- `spec_gate_pick(&gate, source, proposed, &value)` for how many of them to carry;
- `spec_gate_forward(&gate, rows, seconds)` after every forward (plain steps too);
- `spec_gate_result(&gate, source, judged, accepted)` when a verify is settled, where
  `judged` is the accepted count plus one after a rejection;
- `spec_gate_draft_cost(&gate, source, seconds)` per extra draft, for a source whose
  deeper drafts cost time.

colibri's n-gram source does exactly this by default (`COLI_LOOKUP=0`: its own drafts; `spec_decode` in
`c/colibri.c`). GLM-5.2 is attention only, so its verify needs no copies: a rejection
rewinds `kv` and the next forward overwrites the rows. DeepSeek V4.1's DSpark drafter
(`c/deepseek_v41.c`) has a verify path of its own, with the window rings' undo rows; its
depth could come from `spec_gate_pick` with `SPEC_SRC_MTP` the same way, timing its
verifies with `spec_gate_forward`. That is not wired: DSpark's three stages run at a
fixed depth today, and choosing it per verify needs measurements on V4.1 that this work
did not make.

## Tests

- `make -C c spec-drafts-check`: `tests/test_spec_draft.c` (the lookup and the gate on
  their own), `tests/test_qwen38_spec_alloc.c` (every allocation failure while adding
  deeper rollback slots releases the unfinished slot, preserves completed state and
  permits a retry), then `tests/spec_drafts_harness.py` on qwen38's BF16, FP8 and int4-g64
  fixtures with an MTP head and on qwen36 fixtures with an ASCII vocabulary (so a serve
  session can resend a reply as text). Each run with drafts against the same run
  without, tokens and last logits byte for byte: MTP depths 1, 2 and 3 with every draft
  rejected, every one right, alternating, the rejection cycling over the rows and at
  each row, with the forwards, tokens and accepted drafts the loop must count; the
  gate's depth; lookup forced the same ways at 5 and 2 drafts, and natural lookup on a
  repeating prompt; MTP and lookup together; caps 1 and 8, prefill batching off, the
  int8 trunk, int4-g64 experts.
- Serve sessions in the harness: a repeating prompt, its continuation by the reply
  (prefix reuse of the state a speculating turn left, where turns end with a verify's
  drafts unconsumed), a pinned prompt and a turn restoring the pin, a sampled turn and a
  logprobs turn, every frame equal to the session without drafts.
- `bash c/tests/vulkan_engines.sh qwen-spec` (CI leg `qwen-spec`, Lavapipe): the same
  harness on the CPU and in the dense chain (tier off, byte for byte), the chain in
  chunks of 3 rows, the per-matrix path and prompts only (`COLI_VK_CHAIN=2`); deep
  verifies against the CPU with the tier on; the device lost while verifies of up to 4
  rows run; colibri's lookup against its oracle on the CPU and in the chain.
- `qwen-spec-sanitize`: the harness's subset under ASan and UBSan, on the CPU and in the
  chain.
- The existing gates of the S = 2 verify (`make -C c qwen38-tiny-mtp-check`, the
  `qwen-chain` family) run unchanged.
