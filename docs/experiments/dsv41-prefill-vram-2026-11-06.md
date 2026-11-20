# DeepSeek V4.1 prefill on a 16 GiB card: the levers the audit ranked, measured

Two read-only documents were written before this code: Track A, a comparison against
DeepSeek's DeepInfer, and Track B, a VRAM audit of this engine's Vulkan chain. They
ranked six levers (R1-R6) and three they found themselves (the streaming threshold, the
streaming-slot pool, the missing ledger). This record is what the wave that followed
actually measured. It ties four commits to their numbers, records the two corrections
the measurements forced, and adds the one change that never reached a GPU until now: the
RAM plan at 128K and 256K context.

The two documents are copied in as they were written, because their predictions are what
was tested — including the ones that were wrong:

- `dsv41-prefill-vram-2026-11-06-comparison.md` — Track A (`trackA-deepinfer-vs-colibri.md`)
- `dsv41-prefill-vram-2026-11-06-audit.md` — Track B (`vram-audit-trackB.md`)
- `dsv41-prefill-vram-2026-11-06-raw.txt` — every line this record quotes, plus the
  reproduction of the two runs that had never been made

Commits, in order:

| commit | what it is |
|---|---|
| `c615cca4` | the turn-end VRAM ledger, 48 streaming slots reserved before the chunk, `COLI_VK_TIER_STREAM_MIN=2` |
| `90959d61` | a tier pool's limit rounds up to a page, so a three-expert tier's extra layers upload |
| `955a0f6b` | the DSpark stages' default check keys on the device rather than on Lavapipe |
| `12b2908a` | the candidate mask is one bit a column, so the chunk's quadratic cap buys twice the rows |
| `a6486c71` | the correction to the row cost `12b2908a` quoted |
| `95083279` | merge of `fix/ram-plan-128k` (`008a01bc`): the RAM workspace is priced at one prompt chunk |
| `05d9b979` | the long-context table `make -C c test` wanted after that merge (§9) |

Machine, every run: Intel Core i9-12900K, 18 logical CPUs visible to the engine, 38 GiB
RAM, one NVMe (f2fs) the checkpoint streams from, RADV on an AMD Radeon RX 6900 XT
16 GiB with ReBAR, Arch Linux kernel 7.1.11. Model: deepseek-ai/DeepSeek-V4.1-Flash-1017,
the released checkpoint (48 shards, 510.3 GB, fp8 dense + fp4 experts, no conversion).
`COLI_VULKAN=1 V41_STATS=1 ./c/coli serve --model <checkpoint> --ram 28 --ctx <ctx>`,
two turns of the 2600-word prompt (3107 tokens) or one of the 7-token short prompt, through
`prefill-vram.py`. The desktop compositor held 2.5-3.5 GiB of the card throughout, which
matters in §8.

---

## 1. Where the audit left it

The peak is the number everything is judged by, and it is not the prefill's: on the
2600-word prompt the card reached **15.907 GiB of 15.984** (chunk 1280) and **15.918**
(chunk 256), 68-79 MiB of headroom, at the *turn boundary* — the last chunk plus the
decode — in both cases. Cutting the chunk 1280 → 256 moved the first forward by −0.93 GiB
and the prefill plateau by −0.87 GiB and the peak by **+0.011 GiB**; total growth over a
run was conserved to 11 MiB. That is what made the chunk a throughput knob and not a
peak-VRAM one, and it is the reason this wave went after the *tier* and its bookkeeping
instead of the chunk.

Four things the audit left open:

1. 2.14 GiB of a short-prompt run's growth could not be attributed (its §3b, its §7).
2. The tier's "16 rows or more" streaming threshold was, in the engine's own words, never
   measured, and it was the largest throughput lever the audit found (its §6.1).
3. Nothing printed the pools, the streaming slots or the extra layers' pool (§6.3).
4. The candidate mask was stored as an fp32 word a column and read only as a boolean.

## 2. The ledger (`c615cca4`) — the 2.14 GiB, item by item

One line at a turn's end (`V41_STATS`, or `COLI_VK_LEDGER=1` alone): the weight pool's
blocks and the bytes of rows in them, the chain's `dev_bytes` and its three pools'
blocks, the tier's residents and streaming slots with its pool, the extra layers' pool,
the KV mirrors, this process's device bytes against the pools, and the card's live
`mem_info_vram_used`.

A short-prompt 2-turn run (`vram-r2-a1e`), wrapped here, one line in the log:

```
[VK] deepseek_v41 vram: weights 9710.5 MiB in 34 blocks (9608.2 MiB of rows in 521 tensors) |
     chain 75.4 MiB (dev 64.0 in 1 blocks, down 64.0 in 1, up 64.0 in 1) |
     tier resident 2257.0 MiB + 0 slots 0.0 MiB = 2257.0 of 2304.0 in 9 blocks |
     extra layers 232.0 in 1 blocks | kv mirror 50.6 MiB |
     process 12.09 GiB of 12.15 GiB in pools (0.0 MiB outside) | device 14.58 GiB used of 15.98 (card2)
```

The answer: the 2.14 GiB was **not** the tier — its pools do not move after the warm
start. It is 1262.5 MiB of lm head + 150 MiB of the 40 routers + 673 MiB of the DSpark
stages' matrices, all uploaded by the *first forward* and counted by neither the chain's
`dev_bytes` nor the tier's budget, plus 52 MiB of KV mirrors. The weight pool holds
9608.2 MiB of rows at the turn's end against the 7523.0 MiB the setup places.

Two things followed from having the line at all: the head-not-embedding correction (§8a)
and the fact that the audit's 2.14 GiB question is now a one-line lookup rather than a
sysfs reconstruction. The ledger's own first drafts printed nonsense (`vram-r2-a1c.log`,
"tier resident 2257.0 MiB + 1 slots 0.0 MiB = 0.0 of 2257.0 in 0 blocks") and were fixed
before the commit.

## 3. The slots and the chunk reserve (`c615cca4`) — 0.27 GiB off the engine, 0.001 off the card

The audit asked for the 64 streaming slots to be sized from the measured concurrent
streams (~44 ever useful) and guessed ~0.4 GiB of peak headroom. Measured:

| | 64 slots, no reserve (`vram-r1-base-a2`) | 48 slots, reserved before the chunk (`vram-r2-new-a2`) |
|---|---|---|
| chunk rows | 1280 | 1024 |
| tier | 3.46 GiB = 91 residents + 64 slots, 13 blocks | 3.46 GiB = 107 residents + 48 slots, same 13 blocks |
| chain footprint, turn boundary | 2133.7 MiB | 2060.0 MiB |
| prefill wall, 3107 rows | 289.95 s | 295.60 s (+2.0%) |

So the slot set alone frees nothing — the tier splits its budget between residents and
slots, and 64+91 and 48+107 fill the same 13 blocks of 256 MiB. What the reservation plus
the smaller set do is take the engine's own hold at the turn boundary **0.27 GiB** down
(the commit's aggregate 15.53 → 15.26 GiB; the chain term in the two arms' own lines is
the 2133.7 → 2060.0 MiB above), and the card's sysfs peak does not move at all
(15.907 → 15.906 GiB) because the desktop yields (§8c). The audit's "~0.4 GiB of peak
headroom" is right about the engine's hold and wrong about the card: on this box a VRAM
win is only a win if you read `process`.

## 4. The streaming threshold (`c615cca4`) — the audit's largest lever, kept

The engine streamed a cold expert from 16 rows and its own line admitted the number was
never measured. The bench (`COLI_VK_CHAIN_BENCH=xbrows` in `tests/test_vk_chain.c`), one
21.1 MiB DeepSeek V4.1 expert: a device step costs **0.12 ms at 1 row, 0.21 at 2, 0.40 at
4, 0.76 at 16** (the tiled GEMM from there) against a **2.2 ms** upload and the CPU's
**2.05-2.52 ms a row**. Two rows already pay for the upload. `COLI_VK_TIER_STREAM_MIN=2`
is the default now; `COLI_VK_TIER_GEMM_ROWS` stays 16, because below 8 rows the GEMM route
is slower than the GEMV.

On the chunk-256 workload (`COLI_VK_CHAIN_ROWS=256`, the 2600-word prompt), the same
prompt both sides:

| | min 16 (`vram-a3-16k-ch256`) | min 2 (`vram-r3-ch256`) |
|---|---|---|
| routed experts on the device | 65.4% | **98.4%** |
| expert rows kept on the CPU | 237 528 of 688 317 (34.5%) | **10 165 (1.5%)** |
| prefill wall a row | 226.7 ms | **178.0 ms** (−21%) |
| expert reads a turn | 74.7 GB | **1.15 TB** |

It pays in disk bandwidth (10.5 GB/s of uploads), not in time. This is most of R1's
throughput benefit without R1's ~1 GiB of scratch: the audit's §6.1 called it "most of
R1's benefit without R1's VRAM cost" and the measurement agrees.

## 5. The tiny-budget pool (`90959d61`) — a page, and the MTP stages on the device

`tier_pool_limit` sized a block from the budget rounded up to a page but left the limit at
the budget itself, which the pool cannot grow a block past; `pool_suballoc` then fell back
to the limit floored to a page and counted a refusal. deepseek_v41's three-expert fixture
(`COLI_VK_TIER_GB=0.00005`) is the case that exposes it: one extra-layer expert is
15360 B, the page floor of the block capacity 12288 B, and the first MTP stage's upload
was refused — "failed 1 refused by the pool", `T.xmax` 0, the extra layers left on the
CPU. The limit rounds up to a page too now: "extra layers (2) 4 of 48 routed experts,
resident 1 (budget 1), uploads 7, evictions 4, failed 0", and the run's output is the
CPU's. The default-budget serve is unchanged, and §9's two runs reproduce it exactly.
`cu_tier.c` renames the two accessors the ledger added, without which `make -C c test`
does not compile the CUDA build of `vk_tier.c`.

`955a0f6b` is the test that had to move with it: deepseek_v41's DSpark loop asserted the
MTP stages never go on the tier *by default*, which is only true where the device's memory
is the host's (Lavapipe). On a discrete card `COLI_VK_TIER_MTP`'s default puts them on the
tier, so the assertion fired on every DSpark case before any of them could be judged. It
now keys on the engine's own tier "on" line (any "shared RAM" wording vs "device
memory").

## 6. The packed candidate mask (`12b2908a`) — 1.94× the rows for the same 256 MiB

The indexer's candidate mask is only ever read as a boolean: the score kernel tests it,
the host's copy feeds a "kept or not" byte, nothing multiplies by it. It was an fp32 word
a column beside fp32 scores, so a chunk row cost `4E` bytes of scores + `4E` of mask with
`E` the forward's end position. One bit a column now (`vkc_dsv4_cand` writes it,
`vkc_dsv4_score` reads it, `mask_row` and `mask_off` in words of 32 columns), so a row
costs `4E + E/8` and the chunk's quadratic cap keeps the same 256 MiB pair for 1.94× the
rows: `256 MiB / (4E + 4·ceil(E/32))` instead of `32M / E`. The arithmetic reproduces
both quoted cases exactly:

| prompt positions `E` | cap before | cap now | whole-row cost (the pair's share) |
|---|---|---|---|
| 20 000 (tiny fixture) | 1 677 rows | **3 253 rows** | 181 → 105 KiB (156.25 → 80.6 KiB) |
| 131 072 | 256 rows | 496 rows | — |

Measured: the tiny fixture at 20 000 positions prints "a prompt of 20000 positions caps a
chunk at 3253 rows (80.6 KiB a row of scores and packed candidate mask, 256 MiB of 256)";
on the released checkpoint the row cost goes **1317 → 1305 KiB** at `E`=3107 (the mask is
12.1 KiB of that, one bit a column 0.4), and a later run at `E`=1300 measured 1303 → 1298
(41 words a row).

No row of arithmetic changes. The same blocks are picked and kept whole, and
`tests/test_vk_chain.c`'s indexer case compares the mask column for column through the
single-workgroup, wide-query and grid kernels, with the families giving the CPU's tokens.
Nothing printed the cap before either; it prints one line now.

**The correction (`a6486c71`).** `12b2908a`'s message quoted the saving as "1293 → 1281
KiB a row at E=3107". 1293 is the row cost at `E`=11 (the short-prompt arm); at `E`=3107
the same 8 bytes a column make it 1317, which is also the number the audit prints. The
pair is 1317 → 1305. The saving was never in doubt — 8 bytes a column became 4 + 1/8 —
only the base it was subtracted from. §9's runs print 1293 KiB a row at `E`=11, which is
that base.

## 7. Two refutations (the wave-2b list, measured rather than argued)

**M1 — the 1262.5 MiB is the lm head, not the embedding.** `tie_word_embeddings` is false
on this checkpoint; the embedding and the head are two different 129280 × 5120 bf16
tensors, and the embedding's only reader is `wb_row_f32` (`c/deepseek_v41.c`), which
copies the rows a step needs into the host's rows — one a token when decoding, one a
position for a chunk. A run with every device upload printed shows `head.weight` as the
1262.5 MiB item and the embedding never uploaded at all. The head is a GEMV of 662M
multiply-adds a token, so it stays. This matters because "the embedding is on the device"
was a lever that does not exist.

**M2 — the per-layer drain waits on nothing the host does not read.** The audit's R4
asked for the `vkc_finish()` after each layer's attention frame to wait only the frames the
host reads; it estimated 2-8 ms a chunk. Measured with both halves timed separately, a
1280-row chunk spends **28095 ms** waiting for the frame it reads and **0.5 ms** for
everything else: two frames are in flight and one queue runs submissions in order, every
frame starting with a barrier on everything submitted before it, so the previous layer's
shared expert is always through. The change was implemented, measured at 0.5 ms a chunk and
dropped. The audit's estimate was right; what it would buy is not there. What remains is
the attention wait itself, which the host needs, and the device is idle during
`moe_run_at` instead.

## 8. The three corrections to the two documents

**(a) head, not embedding** — §7/M1. Track A and Track B both say the token embedding is
uploaded; it never is. The corrected text is in `docs/deepseek-v41.md` so it does not come
back.

**(b) the drain is a no-op** — §7/M2. R4 stays listed as "nearly free" in the audit; it is
now measured: 0.5 ms a chunk, dropped.

**(c) sysfs is not the engine's hold — the compositor is elastic.** Every absolute number
in the audit is `mem_info_vram_used` − a 2.509 GiB compositor floor. That floor is a
constant only while nothing presses it. With the tier forced to 2.0 GiB on the same
prompt (`vram-r4-tiergb2`), the engine held **1.46 GiB less** (`process` 15.31 → 13.85 GiB)
and the card read **0.19 GiB less** (`device` 15.91 → 15.72): the desktop took **1.27 GiB**
of the freed VRAM for itself. Hence §3's 48-slot change: 0.27 GiB off the engine's hold,
0.001 GiB off the card's peak. On a prompt small enough not to press the ceiling (the
7-token one) the two agree: peak = `process` + 2.49 GiB of desktop. The rule for the next
measurement: read `process` for the engine, `device` only for the card.

## 9. The RAM plan at long context, and its first GPU run (`95083279` of `008a01bc`)

A sibling branch, `fix/ram-plan-128k`, fixed the one thing in this area that had never run
on a GPU. `family_registry._dsv41_geometry` priced deepseek_v41's forward workspace at
`batch == context`: hc_mult residual copies plus the sublayer scratch, **200.0 KiB a row of
context** — 3.125 GiB at 16384, 25.0 GiB at 131072, 50.0 GiB at 262144 against a `--ram 28`
budget. At 131072 and 262144 the plan therefore held a zero-byte expert cache and
`cap_for_arch()` raised "DeepSeek V4.1 Flash RAM budget cannot hold one expert slot per
layer" before the model was read. deepseek_v41's prefill reads a prompt in bounded chunks
(default 1280 rows; `COLI_VK_CHAIN_ROWS` / `COLI_PROMPT_SEGMENT` set it), so the workspace
is priced at `min(ctx, chunk)` — **250.0 MiB** — and stays bounded as the context grows.
The regression test (`c/tests/test_resource_plan.py`) builds a two-layer 128-expert
checkpoint and asserts a positive expert cache per tier at both contexts, and that the long
context's workspace stays under 64 MiB.

Before and after, the released checkpoint, `COLI_VULKAN=1 RAM_GB=28`, python only
(`cap_for_arch`, the function `coli serve` asks):

| `--ctx` | 008a01bc's parent (`05e2c6d7`) | merged |
|---|---|---|
| 8192 | 20 slots/layer | 22 |
| 16384 | 18 | 22 |
| 32768 | 14 | 22 |
| 65536 | 5 | 21 |
| 131072 | **ValueError** | **21** |
| 262144 | **ValueError** | **20** |

The residual 22 → 20 is the resident KV/state, which does grow with the context. What the
old code did — 20, 18, 14, 5, fail, fail — is the whole bug: a plan that starves the expert
cache at every context and then refuses to start.

One thing the merge surfaced: `05e77483` had pinned the contexts `chat_context()`
recommends for V4.1 to the whole-context workspace (16384 on 16 GB, 65536 on 64 GB,
131072 only from 512 GB, and the refusal at 262144), in
`c/tests/test_context_compaction.py`. With the workspace priced at a chunk the same call
gives 65536 on 8 GB, 131072 on 16 GB and the family's 262144 ceiling from 64 GB — the
workspace is still in the sum the reason prints (0.3 GB at every step). The merge left
that table asserting the old answers, so `make -C c test` failed 5 times on it; the
expectations and the comment are updated in the commit after the merge, which is the only
source change this wave made outside the four commits and the merge itself.

The GPU run (never made before): a 1-turn short-prompt serve at each ceiling,
`prefill-vram.py --turns 1 --ctx <ctx>`.

| | `--ctx 131072` | `--ctx 262144` |
|---|---|---|
| RAM plan | 21 slots/layer | 20 slots/layer |
| trunk placement | 40 of 40 layers on the device, 7.41 GiB | same |
| mapped window | "gave the 4 GiB asked" | same |
| tier | 2.59 GiB = 68 experts, 48 slots, extra layers 168.8 MiB | same |
| chunk | 768 rows (1293 KiB a row) | same |
| turn | TTFT 13.04 s, wall 17.93 s, 4 tokens, DSpark 2 of 2 drafts accepted | TTFT 13.17 s, wall 18.12 s |
| ledger | `process` 11.28 of 11.34 GiB in pools, `device` 14.67 of 15.98 | identical to the digit |
| chain | 3 forwards, 326 frames, 75.4 MiB on the device | same |
| expert I/O | 1946 misses, 36.6 GB in 13.3 s (2.76 GB/s) | 1956 misses, 36.8 GB in 13.5 s (2.74 GB/s) |

No OOM, no `COLI_VK_WINDOW_ASK_MB` retry (the ask was satisfied at 4 GiB with the desktop's
3.46 GiB already on the card), and the ledger's default-budget figures reproduce `90959d61`'s
exactly — a second sighting of the same line.

**What this validates, and what it does not.** It validates the plan → load → first-forward
path at both ceilings, which is what the planner could not do at all before. It does not
measure a 128K- or 256K-*token* conversation: with a short prompt `E`≈11, so the KV mirrors
(the audit's 0.62 GiB at 128K tokens) and the indexer's quadratic pair are not exercised,
and the chunk cap does not bind below ~85 000 prompt positions. A long conversation is
still the unmeasured case, exactly as the audit's §7 said.

## 10. Parked, with the reason

- **R2 — the device KV mirror in the checkpoint's quantized format.** The only VRAM lever
  that scales with context (0.31 GiB at 64K tokens, 0.62 at 128K, 4.7 at the 1M ceiling)
  and it is what would pay for R1's ring growth. It is also the only recommendation that
  cannot be bit-exact, so it belongs behind an explicit approximate mode with its KL
  measured, the way `V41_PREFILL_TAIL` is. Not started.
- **R1 proper, with bf16 activations.** The packed mask landed the cheap 1.94× of the
  quadratic cap (§6) without the scratch. Tiling the scores further and running the
  activation scratch in bf16 is a VRAM-for-throughput trade the audit priced at +1.0 GiB
  net at 128K-token prompts against 68 MiB of headroom — i.e. bf16 acts are a precondition,
  not an option, and the audit's measured "peak does not move with the chunk" says the
  prefill plateau is where the GiB would go. Parked until bf16 has a bit-exactness story.
- **`V41_HEAD_HOST` — the lm head back on the host.** The head is the largest single item
  the ledger attributes to the first forward (1262.5 MiB) and it would be a device-VRAM
  win… at the cost of a GEMV of 662M multiply-adds a token moving to the CPU, i.e. the
  decode rate. Nothing has measured that trade, so there is no knob: parked as a candidate,
  with M1 (§7) as the reason it is not the embedding that first looks like it.
- **Layer pipelining** (the host MoE of chunk *k* hidden behind something else). Prompt
  rows are causally ordered — chunk *k*+1's attention reads chunk *k*'s KV — and within a
  chunk the router round trip blocks cross-layer overlap. Track A closed this as "close
  it", and M2 (§7) shows where the idle time actually is: the device idles during
  `moe_run_at`. Out of scope without a router placement change.

## 11. Gates

On the merged tree, `PYTHON=/root/.venv-oracle/bin/python3`,
`VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json`, one after another with
orphans killed in between:

| gate | result |
|---|---|
| `bash c/tests/vulkan_engines.sh deepseek` | 29 OK, 1 FAIL (environmental) |
| `bash c/tests/vulkan_engines.sh deepseek-chain` | 36 OK, 1 FAIL (environmental) |
| `bash c/tests/vulkan_engines.sh prefill-deepseek` | 2 OK, 1 FAIL (environmental) |
| `make -C c vk-chain-check VK=1` | PASS |
| `make -C c test` | test-c ALL PASS (0 failures); test-python 3227 tests, 294 skipped, 2 FAIL (both environmental) |

The five failures are the known environmental set — identical before this wave's commits,
reproduced twice on this box: "deepseek_v4 session: no matmul ran on the device" and "chain
deepseek_v4 4 experts: the chain never ran" (the deepseek_v4 engine's dense placement
declines with the desktop holding VRAM; its tokens still equal the CPU's), "pf
deepseek_v41 staged: logits" (max |logit diff| 3.22e-02 at both commits), and the two
`test_harness_builds_without_warnings` cases (a GCC ABI note in `mistral_vision.h`).

The first run of `make -C c test` on the merge reported 3227 tests, 294 skipped and 7
failures: the five `test_context_compaction` ones §9 describes, plus the two environmental
`test_harness_builds_without_warnings` cases. The table above is the last run, on the tree
this record is committed with.

That target's own totals move between runs on this box: 3227 tests / 294 skipped, then
3019 / 358, then 3227 / 294 again. The two 3227 runs are on either side of `05d9b979`,
whose only change is one test method's expected values, so the movement is in the suite's
own discovery and skips (what the gate's earlier families leave on disk), not in the
change — worth knowing before a count is read as a regression.

## 12. Reproduce

```
cd /root/Projects/colibri
pkill -f '[c]oli serve'; pkill -x deepseek_v41     # between runs
# the ledger, the slots and the threshold (§2-§4)
python3 prefill-vram.py --ctx 16384 --words 2600 --tag vram-mid --port 8021
# the long contexts (§9): one turn each, short prompt
python3 prefill-vram.py --ctx 131072 --turns 1 --tag vram-c1-128k-short --port 8031
python3 prefill-vram.py --ctx 262144 --turns 1 --tag vram-c2-256k-short --port 8032
# the planner alone, no GPU (§9)
COLI_VULKAN=1 python3 -c 'import sys; sys.path.insert(0,"c"); from openai_server import cap_for_arch; \
  print(cap_for_arch("deepseek_v41", None, {"RAM_GB": "28", "CTX": "131072"}, "/root/Models/DeepSeek-V4.1-Flash-1017"))'
```

The `.csv` files `prefill-vram.py` writes are the 50 Hz sysfs series (up to ~2 MB); they
are not committed, only named, in `dsv41-prefill-vram-2026-11-06-raw.txt` §6.
