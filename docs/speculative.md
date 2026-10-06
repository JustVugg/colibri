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
| `colibri` (GLM-5.2) | its MTP head as before; without a head, the n-gram drafts come from the prompt lookup below and its gate by default (`COLI_LOOKUP=0`: its own n-gram drafts as before) | as before |

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

Drafting stays off where a row's result depends on what is resident: qwen36 under the
CUDA expert tier, `CACHE_ROUTE` or a qpack container (with `COLI_LOOKUP=1` asked for, a
`[qwen36] COLI_LOOKUP=1: no drafts under ...` line says so; the default stays quiet). One case stays inside the gate but changes the last bits: with the
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
Its first draft reads the model's four streams at the last fed position with the picked
token. Each next draft reads the head's own streams from the row before, in place of the
model's (only the verify computes those), with the draft just proposed. Those deeper
rows write the head's K/V and indexer rows past its settled ones, a tail that the
settled pairs overwrite once the verify has run. After the verify the head takes the
standing rows' streams, each with the token that followed it.

`Q38_MTP_DRAFTS` sets the drafts per verify: `1`, `2` or `3` drafts every time, or `0` /
`auto` to let the gate pick 0 to 3 per verify. The default is 2, the depth measured
fastest (below). `Q38_MTP_DRAFTS=1` is the one-draft verify that came before.

## The MTP head's drafts (qwen36)

`Q36_MTP=1` reads the release's `mtp.*` block from the converted container
([conversion and settings](qwen36.md#the-mtp-head-q36_mtp)). Its input pairs
the backbone's output **after its final norm** with the next token's embedding.
The head has its own causal KV cache; every completed pair contributes its K/V,
and only a draft row runs the full attention/MoE/output computation.

`Q36_MTP_DRAFTS=1..7` bounds the depth, `auto` lets the measured gate choose,
`Q36_MTP_PMIN` stops uncertain deeper drafts, and `Q36_MTP_VOCAB_IDS` limits
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
that ran (0.5 until one has). Every 16 picks it tries one draft deeper than its best,
so an estimate it stopped feeding can recover. After 64 forwards without a plain step it
takes one, so `T(1)` stays measured. Prompt lookup is always gated. The MTP head is
gated under `Q38_MTP_DRAFTS=0`. `COLI_SPEC_GATE=0` drafts every proposal in full
(the tests use it).

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
| `Q38_MTP` | on when the checkpoint has the head | `0`: plain decoding, the head unread; `1`: the head, and a checkpoint without one is refused. The RAM plan (`coli plan`) prices the head when it is on: its dense tensors and its own expert cache (`Q38_MTP_CAP`, the layers' cap by default, at the head's FP8 record). |
| `Q38_MTP_DRAFTS` | `2` | qwen38 with the MTP head on: the MTP head's drafts per verify, `1` to `3`, or `0` / `auto` for the gate's pick. |
| `COLI_LOOKUP` | `1` | Prompt-lookup drafts (qwen38, qwen36), and colibri's n-gram drafts from the same lookup and gate when it has no MTP head; always gated. `0` turns them off. |
| `COLI_LOOKUP_DRAFTS` | `5` | Lookup drafts per verify at most, `1` to `5`. |
| `COLI_SPEC_GATE` | `1` | `0`: every proposal drafted in full, the gate's estimates ignored (tests). |
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
default is therefore 2 (`Q38_MTP_DRAFTS=2`) on the CPU and in the chain alike.

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
