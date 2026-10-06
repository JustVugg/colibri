# qwen36: CUDA VRAM expert tier

Applies colibri's placement concept ("route -> place -> overlap -> learn") to
Qwen3.6-35B-A3B one level up from the GLM disk tier: all 10,240 experts live
in RAM, the **hot** ones are promoted into DEVICE_LOCAL VRAM across one or
more GPUs and computed there through the existing shared CUDA backend
(`backend_cuda.cu` expert-group API — no new backend).

> **Two engines.** The tier also serves Qwen3.8-Flash-Next (`c/qwen38.c`,
> [qwen38.md](qwen38.md#gpu-cuda-vram-expert-tier)) in its *fp8 streaming
> mode* (`qt_init_fp8`): experts do not all live in RAM there, `cap` may be
> smaller than the expert count, the tier copies each expert's e4m3 slab and
> block scales when the engine reports it and keeps no pointer into the
> engine's slot, and promotion happens at report time instead of a warmstart.
> Everything below describes the Qwen3.6 modes unless it says otherwise.


## How it works

- **Home device:** expert `eid` lives on GPU `eid % n_gpus`; no duplicates.
- **Placement:** routing heat decides who earns VRAM (LFRU semantics from
  `tier.h`, 25%+4 hysteresis). Runtime heat is halved every 1024 decode ticks,
  so a long-lived process can replace experts from an old workload instead of
  permanently freezing its initial hot set. A parallel **warmstart** fills the per-device
  budget before the first token — ordered by a persisted heat table
  (`HEAT_FILE`) when present, so a second run starts fully placed.
- **Decode:** per (token, layer) the resident experts are issued as async
  groups on all devices (`coli_cuda_expert_group_issue/take`); VRAM misses
  fall back to the CPU int8 path and overlap with the in-flight groups, as
  does the shared expert. Placement never changes routing or precision.
- **Memory:** on an **int4 container** the warmstart frees the RAM int8 copies
  of VRAM-resident experts (rematerialized from the packed int4 copy on LFRU
  eviction; no container access). Peak RSS for the 35B int4 container: ~29 GB
  with two 8 GB GPUs. The RSS saving is a property of packed containers only:
  on an **int8 container** there is no second copy to rematerialize from, so
  since #1341 nothing is freed and every resident keeps its full RAM weights —
  such a run gets the tier's VRAM speed at full residency cost, and the 29 GB
  figure below does not apply to it.

## Usage

```bash
make -C c qwen36 CUDA=1 CUDA_ARCH=native   # NVCC=/usr/bin/nvcc on distro CUDA
COLI_CUDA=1 COLI_GPUS=0,1 HEAT_FILE=heat.bin CUDA_EXPERT_GB=auto \
OMP_NUM_THREADS=<physical cores> OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close \
SNAP=<container> N_NEW=200 ./c/qwen36 256 4 prompt.txt
```

### Windows (CUDA_DLL=1)

MinGW cannot link CUDA directly, so the backend is built into `coli_cuda.dll`
with nvcc + MSVC and `qwen36.exe` reaches it through `backend_loader.c`.
`CUDA=1` is rejected on Windows by design. From an *x64 Native Tools* prompt
with MSYS2's `mingw64\bin` and `usr\bin` on `PATH`:

```cmd
cd c
make cuda-dll CUDA_ARCH=sm_89
make qwen36.exe CUDA_DLL=1 ARCH=native
set COLI_CUDA=1
set COLI_GPUS=0
set CUDA_EXPERT_GB=auto
qwen36.exe <same arguments as the CPU build>
```

Keep `coli_cuda.dll` next to `qwen36.exe`, built from the same checkout, and
the CUDA toolkit's `bin` directory on `PATH` for `cudart`. A startup line
`[gpu] MoE experts -> CUDA VRAM tier` confirms the tier is active; without it
the run is CPU-only.

`cap` (argv[1]) must equal `n_experts` (full RAM residency). int4 containers
only (the int8 container keeps the CPU path). `COLI_TIMERS=1` prints
per-phase timings and tier telemetry.

## Placement: where the dense trunk goes (`COLI_PLACE`)

The tier moves the routed experts. On this hybrid model that is the *small*
part of a token: 40 layers × top-8 experts, ~190 MB of int8 per token at an
81 % hit rate, against a dense trunk -- attention, DeltaNet projections,
shared expert, lm_head -- of **1.8 GB of int8 read on every token**. Leaving
the trunk on the CPU is why a 6 GB card sees the hit rate stop mattering
(#1040): the GPU is doing the cheap job.

By default the engine now places the trunk itself. Before the tier decides its
budget, the engine offers each trunk component with its size, and the tier
prices them against the experts they would displace, in **bytes saved on the
memory bus per token, per byte of VRAM**. The components, with their int8 size
on Qwen3.6-35B-A3B (hidden 2048, 30 DeltaNet and 10 attention layers):

| component | what | per layer | total |
|---|---|---|---|
| `lmhead` | the output head, once | | 508 MB |
| `dnproj` | DeltaNet in_proj qkv ++ z, fused | 25.2 MB | 755 MB |
| `dnout` | DeltaNet out_proj | 8.4 MB | 252 MB |
| `attnproj` | attention q, k, v, o (one item, four matrices) | 27.3 MB | 273 MB |
| `shexp` | the shared expert's gate, up, down (offered only with `Q36_OFFER_SHEXP=1`, see below) | 3.1 MB | 126 MB |

Offer order is the placement priority once the budget runs short: `lmhead`,
then `dnproj`, `dnout`, `attnproj`, `shexp`, each in layer order, so a partial
placement is whole layers. `dnout`, `attnproj` and `shexp` were added after
the M10 datapoint in #1652 showed the un-offloaded dense path as the ceiling
on a CPU without AVX2; measured on a 16-core CPU, of the 37.5 ms a decoded
token spends in the DeltaNet stack 23.4 are the input projections, 8.3 the
out_proj and norm, 3.3 the convolution and 2.4 the recurrence -- the matmuls,
not the recurrence, are what the trunk costs. They are served from VRAM on
decode (one GEMV each); a prompt batch keeps the batched CPU matmul.

Placed DeltaNet input projections (`dnproj`, qkv ++ z) also run as CUDA
batches during prefill: at most 256 rows per call, with input/output staging
bounded to 32 MiB (or one row if that alone is larger). The convolution and
recurrent state still advance one token at a time. An unavailable or failed
projection uses the existing per-token CPU path. This changes dispatch count,
not the recurrent update order; it is not a full GPU DeltaNet implementation.

**Measured, not assumed.** The pricing rule below presumes the GPU answers a
GEMV faster than the CPU does. Four Tesla M10 (sm_50, no tensor cores, four
GPUs on one PCIe board) said otherwise in #1652: with every expert
VRAM-resident, placing the trunk made every component slower (lm_head 68.8 ms
against 41.7 on the CPU, the 30 DeltaNet projections 106 against 66) and
decode fell from 3.56 to 2.68 tok/s. So the engine measures before it uploads
a byte of trunk: one DeltaNet input projection is timed both ways on the
device that would host it (ten GEMVs, best of three rounds, after a warm-up),
one `[place] probe:` line reports both times, and if the GPU loses the whole
automatic placement is withdrawn (`[place] trunk stays on the CPU`) and its
bytes go back to the expert budget before the warmstart. Only the automatic
placement is questioned: a hand-written `COLI_PLACE` stands, and
`COLI_TRUNK_PROBE=0` skips the probe. The pricing rule:

- a dense component is read every token: 1.0 per byte;
- a routed expert is read with the probability a token routes to it -- its heat
  share when `HEAT_FILE` exists, `topk / n_experts` otherwise -- and the CPU
  fallback reads the int8 slot, twice the VRAM bytes of an int4 expert: 2·p per
  byte.

A component goes to the device with the most room if its value beats that of
the coldest experts it pushes out. Without heat that tail is worth 0.06 per byte
on the 35B and the trunk always wins; with heat, a card whose marginal expert is
routed on more than every second token keeps its experts. Placed bytes come out
of that device's expert budget, and each decision prints as a `[place]` line.

| `COLI_PLACE` | behaviour |
|---|---|
| unset or `auto` | automatic, as above |
| `off` | nothing placed: experts only (the behaviour before this) |
| `lmhead=0,dnproj=0:20+1:20,experts=0` | hand-written list (the measurement tool); obeyed as written, trunk bytes still charged to the budget |
| `dnout=0,attnproj=0,shexp=0` | the same list form for the newer components; any of the five names, per layer or split with `+` |

First calibration, one Quadro RTX 4000 (8 GB), per-row int4 container, 200-token
decode, same prompt, output bit-identical in all four runs:

| | `off` | `auto` |
|---|---|---|
| trunk in VRAM | -- | lm_head 0.47 GB + 30 dnproj 0.70 GB |
| experts resident | 4,391 | 3,595 |
| cold: hit rate / tok/s | 44 % / 8.64 | 36 % / **9.62** |
| warm: hit rate / tok/s | 95 % / 9.63 | 90.6 % / **12.92** |
| same card, budget capped at 5 GB (a 6 GB card's share), warm | 88.9 % / 9.50 | 81.2 % / **13.15** |
| RTX 3070 (8 GB) alone, warm | 93.6 % / 11.09 | 88.4 % / **16.55** |
| both cards, experts on both, warm | 100 % / 10.79 | 100 % / **14.80** |
| reference: Ollama 0.32.5, same model Q4_K_M, same prompt, both cards (57 % CPU / 43 % GPU, 11.8 GB VRAM) | 20.3 warm (21.3 cold) | |

The warm row is the one that matters: at a 95 % hit rate the marginal expert
is as valuable as it gets on this card, and the trunk still wins by a third.
The hit rate drops only 4.4 points for 796 fewer residents because the
displaced experts are the coldest of the heat order -- exactly the ones the
placer priced as cheap. The relative win grows with the trunk's share of the
token: +34 % on the Quadro, +38 % at a 5 GB budget, +49 % on the 3070. (An
earlier version of this table had the two card names swapped: CUDA orders
devices fastest-first, `nvidia-smi` by bus, and I had read the wrong one.)
All sixteen runs of this calibration produced bit-identical text. Against
Ollama on the same box the gap closes from 1.5× (14.97 vs 22.4 in August, two
cards, hand-placed) to **1.23× on a single 8 GB card** (16.55 vs 20.3) --
with Ollama holding its dense weights at ~0.56 bytes per weight (Q4_K_M)
against this engine's 1.0 (int8), and using both cards.

**Two cards are the open case.** With experts on both cards the second card
paces every layer (the slower `take()` gates the chain), so `off` on two cards
is barely ahead of the 3070 alone (10.79 vs 11.09), and `auto` -- which in
this version spreads the trunk by free room and leaves the experts on both --
reaches 14.80 where the hand-written R4 split (`experts=0,lmhead=0,
dnproj=0:20+1:20`: experts on ONE card, trunk across both) reaches 17.11. On
two unequal cards the list still wins; the next version of the placer has to
learn that lesson (experts on one card, the trunk on the other) rather than
have it written for it.

Peak RSS is ~2 GB higher under `auto`: the host-side int8 copies stay as the
CPU fallback. Known, not yet addressed.

**Any dense matrix, by name.** The offer table is not limited to `lmhead` and
`dnproj`: an engine offers whatever it wants placed with `qt_trunk_offer(name,
layer, bytes)` before `qt_init`, asks `qt_place_of(name, layer)` afterwards,
and hands the placed matrices over as int8 rows with
`qt_dense_init(q, scales, I, O, device)`, which returns a handle;
`qt_dense_matmul(handle, y, x, I, O)` answers one GEMV from VRAM and returns 0
(CPU from here on) if the backend fails. The qwen36 calls remain thin
wrappers over the same mechanism. Qwen3.8 uses it for its whole trunk -- 553
matrices, 4.0 GiB int8 on one card ([qwen38.md](qwen38.md), "GPU") -- and
that is also where the backend's resident dense matvec got its own staging
buffers: `coli_cuda_matmul` used to share the `x`/`y` device buffers with the
expert group, which runs asynchronously on its own stream between `qt_issue`
and `qt_take`. A dense GEMV issued in that window (Qwen3.8's shared expert)
overwrote the group's input and output mid-flight -- no CUDA error, only
wrong numbers. qwen36 never called the dense path inside that window, so its
outputs were unaffected.

## The DeltaNet layer on the card (`Q36_DN_GPU=1`)

With the trunk in VRAM the CPU still spent about 8 of 39 ms per token on the
thirty DeltaNet layers' convolution, L2 norms, recurrence and gated norm
(`dn-sub: proj 7.9 | conv 3.0 | l2n+rec 2.4 | norm+out 2.8`). None of that is
arithmetic -- a token's convolution is 33k multiply-adds, the recurrence 32
outer products of 128 x 128 -- it is the round trip: the in_proj's result
came down to the host, the recurrence ran there, its output went back up for
the out_proj, four copies and their synchronisations per layer, thirty layers
per token.

`Q36_DN_GPU=1` keeps the layer on the card. For every DeltaNet layer whose
in_proj (`dnproj`) and out_proj (`dnout`) the placer put on the same device,
the conv ring, the recurrent state (32 x 128 x 128 f32, 2 MB per layer, 63 MB
for the 35B) and the conv/norm weights go there too, and a decode token runs
the layer end to end in one device chain (`coli_cuda_dn_step`): x up, in_proj
GEMV, causal conv + SiLU, per value head the L2 norms (double, eps inside the
square root as on the CPU), the decay, the delta against the decayed state,
the write-back, `out = q^T S` over the updated state, the gated RMSNorm with
`silu(z)`, out_proj GEMV, out down. The two gates (`a`, `b` projections, 32
outputs each) stay on the CPU and travel as kernel parameters. One upload and
one download of 8 KB per layer, the state never crosses the bus.

The host arrays remain the state's owner. The card's copy is a cache that is
either fresh (holds what the host holds) or ahead of it; a CPU step -- a
prompt (`S > 1` keeps the batched CPU path), a layer whose projections are
not on one card, a step after a failed GPU step -- pulls the state down
first, a snapshot for `--pin` pulls all layers, `reset_recurrent` and a pin
restore invalidate the card's copy. So `--pin`, prefix reuse and the
`QT_UPLOAD_SYNC`-style diagnostics keep working unchanged. The automatic
placer now puts a layer's `dnout` on the device its `dnproj` went to when that
device has the room, so under `auto` every DeltaNet layer qualifies; a
hand-written list must keep the two on one card itself (note that a split
such as `dnproj=0:20+1:20` counts *model* layers 0..39, not DeltaNet layers).

Measured on the 3070 alone (per-row int4 35B, 49-token prompt, 200 decoded
tokens, trunk in VRAM, `COLI_TIMERS=1`):

| | `Q36_DN_GPU=0` | `Q36_DN_GPU=1` |
|---|---|---|
| DeltaNet ms/token | 16.2 | **10.8** |
| token ms | 39.4 / 39.3 | **33.9 / 33.3** |
| decode tok/s | 25.4 | **30.0** |
| tok/s as the engine prints it (TTFT included) | 21.2 / 21.3 | **24.1 / 24.4** |
| VRAM hit rate | 82 % | 82 % |
| layers on the card | 0 | 30 |

Fourteen per cent per token; the 10.8 ms that remain are the two GEMVs (16
and 8 MB per layer) and the two copies. The output is not byte-identical to
the CPU path (the GEMVs sum in a different order; two GPU runs are identical
to each other), same as every other placed component. On two cards the layer
gains the same 6-7 ms, but this box's second card (Quadro RTX 4000) runs a
dense GEMV about 65 % slower than the 3070 (lm_head 7.3 against 4.4 ms), and
at 99-100 % expert residency the MoE phase only drops from 14.0 to 11.2-13.6
ms -- the expert groups are bound by GPU time and by the slower card's
`take()`, not by misses -- so the best two-card form (trunk on the 3070,
`experts=all`) lands at 32.9 ms, level with the 3070 alone.

`COLI_PLACE` gained `experts=all`: the routed experts stay on every
`COLI_GPUS` card even when a hand-written list reserves some of them for the
trunk (the default hands a reserved card's experts back to the others, #1361).

### The shared expert stays on the CPU by default

`shexp` on the card is 120 synchronous small GEMVs per token that sit between
`qt_issue` and `qt_take`, where on the CPU the shared expert hides behind the
expert group. Measured on the 35B with the DeltaNet layer on the card: shared
3.7-4.0 ms/token on the CPU against 6.3 on the same 3070, and 11-12.7 ms when
a slower second card holds some of the layers; the token went 33.3 -> 31.8 ms
on one card and 38.6 -> 33.6 ms as a two-card pipeline once `shexp` left the
card. The engine therefore offers `shexp` to the placer only with
`Q36_OFFER_SHEXP=1`; a hand-written `COLI_PLACE` naming it is obeyed when the
offer is made.

### Two cards as a pipeline (`QT_HOME=layer`)

By default expert `eid` is homed on device `eid % n_gpus` in *every* layer,
so each layer's group is issued to both cards and `qt_take` waits for the
slower one -- measured on the 3070 + Quadro RTX 4000 pair, `take` 5.2 ms per
token against 0.2 on one card. `QT_HOME=layer` homes every expert of a layer
on one device, chosen per layer range by each card's allowance divided by
the time it needs for a dense GEMV (probed at startup with a 64 MB int8
matrix; `QT_LAYER_SPLIT=<n>` layers on the first card overrides), puts the layer's trunk
components on the same device and lm_head on the last layer's, and issues a
group to exactly one device: the token runs the first layers on one card and
the rest on the other, like llama.cpp's layer split. Budgets, warmstart, LFRU
and the swap path follow the homes; a hand-written list's reservation does
not remove a card in this mode, its layers' experts live there.

Measured (same box, `Q36_DN_GPU=1`, shared expert on the CPU, 200 tokens):
the join is gone (`take` 5.2 -> 0.17 ms) and the MoE phase falls from 12.4-13.4
to 9.4 ms at 99.7 % residency, but the pipeline ends level with the 3070 alone
-- 33.6 ms/token (29.8 tok/s) at a 28/12 split against 31.8-33.1 ms (30.2-31.4
tok/s) -- because lm_head (+2.8 ms) and twelve DeltaNet layers (+1.6 ms) now run
on the slower card. The startup probe measures that card 154 % slower per dense
byte and chooses 28/12 by itself; under `auto` that is 36.0 ms (27.8 tok/s). Two equal
cards would keep the MoE gain without that price. Cold, two cards win either
way (45.3 against 52.3 ms), since more experts are resident at once. For the
record, Ollama 0.34.4 on the same pair shows the same shape: 38.1 tok/s on the
3070 alone, 36.6-37.5 on both cards, 33.0 on the Quadro alone.
## The residents follow the prompt (`QT_PREFILL_REPLAN=1`)

The warmstart fills VRAM from the heat file, i.e. from what earlier prompts
routed to, and the LFRU tick then corrects one expert per sixteen tokens and
device. For a prompt on a new topic that is thousands of tokens of catching up:
measured on the 35B with a heat file accumulated over six other prompts, a
fresh prompt's decode hits VRAM 56-63 % of the time at 37 % residency, a cold
start 38 %, and a heat file from the *same* prompt 89-91 % -- the heat
warmstart is an oracle for a repeated prompt and a stranger's guess for a new
one.

The prefill knows better. Its routing is this prompt's routing: offline, a
static per-layer set of B experts chosen from the prompt's own prefill counts
covers its decode routing far better than the heat file at the same budget
(held-out prompts, four topics):

| B per layer (of 256) | heat from other prompts | own prefill counts | oracle (own decode) |
|---|---|---|---|
| 32 | 0.217 | **0.421** | 0.566 |
| 64 | 0.423 | **0.613** | 0.777 |
| 96 (~the 8 GB card's share) | 0.593 | **0.741** | 0.890 |
| 128 | 0.725 | **0.825** | 0.952 |

`QT_PREFILL_REPLAN=1` spends that: after each prefill layer's routing the
engine hands the tier the layer's counts over the prompt rows, and the tier
swaps residents the prompt never routed to for the prompt's most-routed
non-residents of that layer -- budget-neutral, through the same victim-first
swap the LFRU tick uses, in strict count order and only while the newcomer's
count beats the victim's. The swaps of layer L upload while layers L+1.. still
compute; whatever the upload queue does not take at once is drained on the
next layers and, a few per token, on the decode ticks (each swap is a
`cudaFree` + `cudaMalloc`, which synchronise the device under the async
groups, so the decode drains slowly on purpose). `QT_PREFILL_REPLAN_MAX`
caps the swaps per layer (default 256). Placement never changes routing:
the tokens are the ones the CPU path produces.

Two costs come with an eviction and are handled here rather than paid in the
miss path: the tier reports evicted experts (`qt_evicted_take`) and the engine
rebuilds their RAM int8 copies at the next step, in parallel, before the
layers run -- on the int4 container the warmstart had freed them, and left to
the miss path 2,000 victims cost ~7 ms/token over the following 300 tokens.
The `[qtier]` footer also reports the hit rate counted from the first decode
token (`decode VRAM hit rate`), which is the number that moves.

Measured on the 3070 alone (per-row int4 35B, 200-250-token prompts, 300
decoded tokens, four held-out prompts, heat file from six other prompts,
`QT_PREFILL_REPLAN_MAX=24` = 960 swaps per prompt, 12 cores `OMP_PLACES=cores`).
`off -> on`, decode-only figures from `COLI_TIMERS=1`:

| prompt | trunk | decode VRAM hit rate | MoE ms/token | total ms/token | TTFT |
|---|---|---|---|---|---|
| chat | CPU (`COLI_PLACE=off`) | 54.8 -> **72.8 %** | 18.8 -> 15.2 | 67.7 -> 68.5 | 3.9 -> 5.0 s |
| code | CPU | 59.9 -> **75.1 %** | 18.0 -> 14.5 | 69.5 -> 66.8 | 5.1 -> 5.4 s |
| Chinese | CPU | 62.7 -> **74.8 %** | 16.8 -> 14.6 | 68.8 -> 65.8 | 4.0 -> 5.0 s |
| reasoning | CPU | 63.1 -> **75.3 %** | 17.4 -> 15.9 | 70.4 -> 71.6 | 4.4 -> 5.5 s |
| chat | VRAM (all five components) | 41.8 -> **61.7 %** | 22.9 -> 18.7 | 49.2 -> **45.2** | 6.4 -> 6.6 s |
| code | VRAM | 45.1 -> **64.8 %** | 22.3 -> 17.5 | 49.0 -> **44.1** | 8.1 -> 8.2 s |
| Chinese | VRAM | 49.6 -> **60.3 %** | 20.4 -> 20.4 | 46.3 -> 47.8 | 6.2 -> 7.0 s |
| reasoning | VRAM | 44.5 -> **66.7 %** | 22.4 -> 17.6 | 48.7 -> **44.4** | 7.0 -> 7.5 s |

The hit rate moves as the offline table predicted (+12 to +22 points), the
MoE phase loses 2 to 5 ms/token, and with the trunk in VRAM that is 8-10 % of
the token on three prompts of four (the fourth is inside the run-to-run noise
of about ±2 ms). With the trunk on the CPU the same MoE saving disappears in
the 50 ms the dense trunk costs there. The price is at the front: the swaps
upload during the prefill, and each one is six synchronous host-to-device
copies (three matrices, three scale vectors), about 1 ms on this box, so 960
swaps add 0.1-1.1 s to the TTFT of a 200-token prompt; the decode saving pays
that back after 100-250 tokens. Fewer swaps buy most of the gain -- offline at
B = 96, a cap of 12 per layer (480 swaps) reaches 0.709 of the 0.741 the
uncapped plan reaches, 24 per layer reaches 0.741 -- which is why the default
cap is 24. The follow-up that would take the TTFT price away is a pinned,
asynchronous staging path (or one contiguous device block per expert) so the
copies stop serialising with the prefill's expert groups; the swap itself
already writes into the victim's buffers (`coli_cuda_tensor_overwrite`, no
`cudaFree`/`cudaMalloc`; a backend without the entry point falls back to
free-then-upload).

## Measured (Threadripper 3945WX 12C, RTX 3070 8 GB + Quadro RTX 4000 8 GB, Qwen3.6-35B-A3B int4, 200-token decode)

| | 1 GPU (8 GB) | 2 GPUs (16 GB) |
|---|---|---|
| decode tok/s (cold / warm heat) | 9.2 / 9.9 | 10.6 / **11.3** |
| VRAM-resident experts | 4,391 (43 %) | 8,532 (83 %) |
| VRAM hit rate (cold / warm) | 44 % / 95 % | 85 % / 100 % |
| peak RSS | 40 GB | **29 GB** |
| reference: Ollama q4_K_M, same box | 7.5 | 10.5 |

All figures above are int4-container measurements; the peak-RSS row in
particular has no int8 analogue (see **Memory**).

CPU-only baseline of this engine before the tier: 0.35 tok/s.
Numerics: logits cosine vs the f32 CPU reference 0.9992 (dense int8 on),
bit-identical GPU-vs-CPU on the same container (cosine 1.0000001).

## Resident projection batching microbenchmark

Build and run a bounded, model-free comparison of `S` resident-int8 one-row GPU
calls with one `S`-row GPU call:

```sh
make -C c tests/bench_cuda_resident_batch CUDA=1 CUDA_ARCH=sm_89
c/tests/bench_cuda_resident_batch > resident-batch.jsonl
```

Set `CUDA_HOME` if the toolkit is not on PATH and select the architecture for
your device. The harness uses CUDA device 0, uploads each matrix once, warms
both paths twice, then alternates their order over nine paired measurements.
The host wall time includes synchronous input/output copies and compute, but
excludes upload and validation. All output elements are checked between arms
after every pair; sampled elements also have an independent CPU reference.
JSONL retains all nine timings per arm and the median of the paired ratios.
An execution/validation failure exits nonzero; absent CUDA initialization exits
77. The target is explicitly run and is not part of ordinary CI.

A single run on 2026-09-22 (RTX 4070 12 GB, driver 591.86, CUDA 12.9, `sm_89`,
Core Ultra 9 285K, WSL2 Linux 6.6.114.1, `-O3 -ftz=false`) produced:

| Input × output | Rows | Serial median ms | Batch median ms | Median paired ratio |
|---|---:|---:|---:|---:|
| 512 × 1024 | 32 | 2.975 | 0.294 | 10.02 |
| 2048 × 2048 | 32 | 3.588 | 0.813 | 4.31 |
| 2048 × 2048 | 128 | 12.716 | 2.913 | 4.37 |

[Raw paired timings](experiments/qwen36-resident-batch-2026-09-22.jsonl)
include zero sampled CPU-reference error for these deterministic synthetic
inputs. The device was not isolated: telemetry before/after showed 4% GPU
utilization, approximately 3 GB occupied VRAM and 2550 MHz SM clock. Timings show
visible variation; one nine-pair run does not establish reproducibility across
sessions, devices or real activation distributions.

The serial GPU baseline reflects the projection-call pattern replaced by the
DeltaNet input batching change. These are synthetic shapes, not complete
DeltaNet layers: convolution, recurrent updates, normalization, other projections,
model loading and HTTP queueing are excluded. This is **not** an end-to-end model
speedup, nor an attention-prefill speedup: the old attention prefill path used
CPU batch matmul, which is not an arm in this benchmark.
