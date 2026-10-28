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
  (`HEAT_FILE`) when present, so a second run starts fully placed. Without one
  it takes expert 0 of every layer, then expert 1, and so on, so each layer
  holds its share: in index order the card held whole layers (on a Colab T4,
  layers 0-24 and part of 25), and a prompt chunk copied the other layers'
  experts over the bus one layer after another, after 25 layers with the bus
  idle (a Colab T4's 512-token prompt, after a 64-token one and its decode,
  reads 14 % faster: `docs/experiments/qwen36-tier-spread-2026-10-27`).
- **Decode:** per (token, layer) the resident experts are issued as async
  groups on all devices (`coli_cuda_expert_group_issue/take`); VRAM misses
  fall back to the CPU and overlap with the in-flight groups, as does the
  shared expert. Placement never changes routing. With planar int4 in RAM it
  decides which kernel computes an expert, and so its last digits (see "The
  residents follow the conversation").
- **Memory:** on an **int4 gs64 container** the experts stay planar int4 in
  RAM (`expert_ffn.h`, the CPU's kernel for the misses), one copy at the size
  on disk; the tier converts the planar bytes to its pairs when it stages an
  upload. Before #2090 the CUDA build unpacked every expert to int8 and kept
  the packed int4 beside it for the uploads, three times the bytes on disk,
  while the planner (`--auto-tier`) priced the bytes on disk.
  `QWEN_EXPERT_KERNEL=0` keeps that historical layout (int8 for the CPU, the
  int8 copy of VRAM residents dropped at the warmstart). On an **int8
  container** every expert keeps its int8 weights in RAM (#1341).
- **Any RAM size:** with every expert of a layer in RAM (`cap` >= `n_experts`)
  the tier keeps pointers into the slots and swaps residents on its own
  (LFRU ticks, `QT_PREFILL_REPLAN`). With fewer, it copies an expert while the
  engine holds it for the CPU and decides then whether it earns a place in
  VRAM (the coldest resident of its device makes way when the newcomer is
  hotter); VRAM residents are never read into RAM, so the two caches add up.
  The warmstart then loads only the experts it places in VRAM, and the RAM
  cache fills as the CPU's experts are routed. Before #2090 a `cap` below
  `n_experts` switched the tier off, trunk included.
- **Heat:** every routed expert counts once per token, a hit in `qt_issue`,
  a miss when the engine hands its bytes over (`qt_note`). Hits used to count
  only through that note, which in copy mode (and on Qwen3.8) never comes for
  a resident: residents stayed at the heat they arrived with, and any
  newcomer warmer than that evicted them.
- **No experts, still the trunk:** the mixed layout (int4 gate/up, int8
  down), int8 experts with grouped scales and `COLI_PLACE` `experts=cpu` keep
  the experts on the CPU, and the devices take the dense trunk below.

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

`cap` (argv[1]) is the RAM cache per layer as for the CPU build, any size
(see "Any RAM size" above; before #2090 any `cap` but `n_experts` switched the
tier off, a larger one included); `--auto-tier` sizes it from the RAM. `COLI_TIMERS=1`
prints per-phase timings and tier telemetry.

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
bytes go back to the expert budget before the warmstart. On the CPU the ten
GEMVs read the projections of ten layers, as decode reads each once a step:
one 16 MiB projection read ten times stays in a big L3, and beside an RTX PRO
6000 an EPYC 9B45 (three 32 MiB L3s) came within about 10% of the card that way
and tied it under nsys. The experts follow
(`[qtier] the experts stay on the CPU too`) when their allowance is automatic:
the tier computes a resident expert with the same int8 GEMV and a round trip
over PCIe, and a layer waits for its share. On a GTX 1650 Max-Q (#2211; 0.900
ms on the card against 0.306 on the CPU) the tier with a tenth of the experts
resident lost decode (9.2 against 10.9 tok/s) and prefill (14 against 46).
`CUDA_EXPERT_GB=<GB>` keeps them on the device. Only the automatic
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
## The dense chain on the card (`COLI_CUDA_CHAIN`, on by default)

With the trunk in VRAM and `Q36_DN_GPU=1`, a decode token still crossed the bus about
seventy times: every attention projection (`qt_dense_matmul`: the input up, the kernel,
the output down, a synchronization), every DeltaNet step, lm_head, and between them the
CPU ran the attention core, the norms, the router and the shared expert. The Vulkan
backend solved the same problem with its dense chain (`vk_chain.h`, docs/vulkan.md "The
dense chain"): a whole layer recorded as one submission, the residual stream on the
device from layer to layer, only the routed experts' rows crossing. `COLI_CUDA_CHAIN=1`
is the CUDA twin, and it is shaped like the Vulkan one on purpose, so the engine wires
both the same way:

- `c/cuda_chain.h` / `c/cuda_chain.cu` (the backend): buffers (`CC_DEV` on the device,
  `CC_UP` and `CC_DOWN` page-locked host memory the device reads or writes in place),
  frames (`cc_begin` / `cc_submit(wait)` on one non-blocking stream per device, an event
  per frame), and the ops `cc_matmul` (the backend's own GEMV / GEMM kernel over the
  tier's resident tensors, on the chain's stream), `cc_norm`, `cc_rope`, `cc_attn`,
  `cc_dnconv`, `cc_dnrec`, `cc_ew`, each the arithmetic of the shader of the same name
  in `shaders/chain_*.comp`, ported line for line, with the Vkc parameter structs field
  for field. The backend exports them as one table (`coli_cuda_chain_ops`), one symbol
  for the Windows loader, versioned by its size: a DLL that predates the chain has none
  and the engine stays on the per-matrix path.
- `c/qwen36_cuda_chain.h` (the engine): `qwen36_chain.h`'s frames and state contract.
  Per layer, frame A1: the previous layer's MoE output joins the residual (routed in
  rank order, then the gated shared expert, then the add: the CPU's order), the input
  norm, the gated attention (q/k/v, q/k norms, RoPE from the CPU's own cos/sin table,
  the new K/V rows into the device cache and down to the host's, the attention with the
  output gate, o_proj) or the Gated DeltaNet (the tier's fused qkv ++ z projection, b|a,
  the convolution with its ring, the recurrence with its state, the gated norm,
  out_proj), the residual add, the post-attention norm, the router logits; waited for.
  The host routes, runs the routed experts through the tier (`moe_ex` routed only) and
  hands the sum up. Frame A2, not waited for: the shared expert and its gate, on the
  chain's stream while the tier's groups run on the backend's. The last frame runs the
  final norm and lm_head on the last row, or on every row of a speculative verify; the
  logits are written in device memory and one copy brings them to the host's buffer.
  (Written into it directly, one float per warp over the bus, the head took 7.4 ms of a
  13.5 ms step on an RTX PRO 4000. On an L4 the step does not get faster and the extra
  copy costs 0.1 ms.)

**What it needs.** Every projection of every layer, the shared expert's three matrices
and lm_head on one card: `COLI_CUDA_CHAIN=1` offers the shared expert to the placer
(what `Q36_OFFER_SHEXP=1` does) and `COLI_PLACE=auto` takes the whole 35B trunk on an
8 GB card. The chain uploads itself the router's matrix, the DeltaNet b|a rows and the
shared expert's gate row, and the parameter arena. It declines, naming the first matrix
on the CPU, when the placement left anything behind, and under a qpack container or PILOT
prefetch (which reads the residual on the host). With `KV_SLOTS` > 1 it runs each
conversation's prompts, the held conversation's state handed over through the host when
the conversation changes, and the decode steps run every conversation's row on the tier
and the host. `Q36_DN_GPU`'s per-layer step is not set up beside it: the chain owns the
DeltaNet state.

**State, and who owns it.** The residual stream lives on the device for a forward. The
host's KV cache stays canonical: every step copies its new rows back, the device holds
a mirror per attention layer with a watermark (`kv_valid`), a step from `pos_base`
uploads the rows between the watermark and `pos_base` first, a CPU step lowers the
watermark, a cache that grows is mirrored again. The DeltaNet state and conv rings stay
on the device while the chain runs: the host copy is brought back before anything reads
it there (a pinned snapshot, a CPU step: `q36cc_sync_host`) and pushed up after anything
writes it there (a reset: zeros filled on the device; a restored snapshot: an upload). A
speculative verify copies the state after each of its rows but the last into a device
slot, the convolution and the recurrence split into one dispatch per copy (the same bits),
and a rejected draft swaps slot r's buffers in (`q36cc_rollback`) as the CPU swaps its own.
A CUDA error inside a frame loses the device to the chain: the engine zeroes the
recurrent state, rebuilds it on the CPU from the prefix record (a prefill's worth of CPU
work; the KV rows are the host's already) and runs on the CPU from there, as the Vulkan
chain does. `COLI_CUDA_CHAIN_FAULT=n` fakes the loss at the n-th frame.

**Tests.** `tests/test_qwen36_cuda_chain.c` drives an in-memory two-layer model (a
DeltaNet layer, a gated-attention layer, four routed experts and a gated shared expert
per layer, lm_head) through `step_ex` on the fake CUDA tier, the chain's ops answered by
the fake's host-side table (`tests/qwen36_fake_cuda.h`, the same arithmetic as the
kernels): a prompt and decode tokens give the CPU path's logits (max diff 2e-5 of 20),
the host state is untouched while the chain runs and reads back as the CPU path's, the
K/V rows reach the host cache, a reset and a CPU step in between continue from the right
state, a verify's three rows and the rollback of a rejected draft, the head's logits
computed on the card and copied down (a prompt and a verify read in chunks of three and
two rows), a frame that fails (the state rebuilt, the step and the rest on the CPU, the
same logits), and a backend without the table. `tests/test_cuda_chain.cu`
(`make cuda-chain-check CUDA=1`, needs a device; `gpu-compile` builds it) checks every op
on a real card against the same CPU references: buffers and frames, int8 and f32 matmuls
at offsets, the four norms, RoPE, causal GQA with the gate and a selection list, the
convolution in both orders with its ring and snapshot, the recurrence at KD 8 and 128 with
both gates, every element-wise op, and a failed launch marking the device lost.

**Measured** (RTX 3070 alone, Qwen3.6-35B-A3B int4, the trunk in VRAM by `COLI_PLACE=auto`,
expert tier at 81-82 % hits, the usual 49-token prompt, 200 new tokens, two runs each):

| | TTFT | tok/s, the run | ms per decode token | decode tok/s | device wait per token |
|---|---|---|---|---|---|
| `Q36_DN_GPU=1` (the per-layer step) | 1.22-1.26 s | 26.2-27.6 | 30-32 | 32-33 | |
| `COLI_CUDA_CHAIN=1` | 0.98 s | 42.7-42.8 | 18.7 | 53 | 5.7 ms |
| Ollama 0.34.4 on the same card | | | 26.2 | 38.1 | |

The chain alone, with the backend's block GEMV, gave 27.6 ms a token: the CPU's parts
went (the attention core 3.9 ms, the shared expert 4.1 ms, about seventy bus round trips)
and the token was the device's time, 14.7 ms, which was the trunk's 1.9 GB of int8 rows
read at 120 GB/s. `gemv_i8_rows` (backend_cuda.cu: a warp per output row, 16 bytes a
lane, the activation in shared memory) reads them at 330-418 GB/s, and the device's part
of a token is 5.7 ms. What is left is the routed experts on the host (9.9 ms: the tier's
groups and the CPU's share) and the routing, launches and forty waits (about 3 ms).

### Decode after a long prompt

A decode step reads every position's K and V once a layer, and after a long prompt that is
most of the step's attention. On an RTX PRO 4000 it was the widest gap to the reference
GGUF engine (decode tok/s after a prompt of 1K / 18K tokens):

| | before | a KV head's heads a block | and float16 mirrors | reference |
|---|---|---|---|---|
| Qwen3.6 UD-Q4_K_M | 101.3 / 74.5 | 101.4 / 84.8 | 101.4 / 89.8 | 103.8 / 94.7 |
| Flint UD-Q3_K_XL | 100.8 / 65.4 | 102.5 / 79.2 | 102.7 / 86.2 | 100.6 / 90.0 |

- `cc_attn_decg_k` (`COLI_CUDA_ATTN_DEC_GQA`): the decode's blocks of positions take the
  query heads of a KV head at once, each K and V row read once for them (eight times
  before, at Qwen3.6's and Flint's grouping), four floats a load.
  Its scores take two threads a position (`COLI_CUDA_ATTN_DEC_TPP`), adjacent lanes reading
  a K row's float4s in turn against the q rows in shared memory, where a warp's lanes over
  one row summed each head with five shuffles a position; the join takes each block's weight
  once into shared memory instead of once a value. One layer after 18K positions in float16
  (`COLI_CUDA_CHAIN_BENCH=adec` in `tests/test_cuda_chain.cu`): Qwen3.6's 119.5 -> 108.6 us,
  Flint's 144.5 -> 122.1; after 60K, Qwen3.6's 391 -> 379.
- Float16 mirrors (`COLI_CUDA_KV16`): the host's cache is float16, and the mirrors held its
  values as float32. They now hold them as float16, and every attention kernel reads a cache
  of either type. The numbers are the same, so the greedy answers are the same, in half the
  VRAM: a window's or sinks' model's tier keeps half the room for them out of its budget.

### The routed experts in the frame (`COLI_CUDA_CHAIN_MOE`, on by default)

The chain above still ended its frame at every MoE layer: the router's logits down, the
host's routing, the tier's groups and the CPU's share, the sum back up. For a step of up
to 8 rows (decode, a verify) the frame now routes and computes the experts itself, as
the Vulkan chain does (docs/vulkan.md, "The routed experts in the frame"):

- `cc_route` (`cuda_chain.cu`, the port of `chain_route.comp`): softmax, group-limited
  top-k, renormalization, the lower expert on a tie, one block per row.
- `cc_moe` (`backend_cuda.cu`: `moe_gu_k`, `moe_dn_k`, `moe_sum_k`): every (row, slot)
  pair whose expert the table holds, gate and up, then down, a warp per output row, in
  the tier's formats (1 int8 rows, 2 int4 rows, 4 int4 with a scale per 64, 8 fp8 with
  its 128 x 128 block scales); then each row's sum in the tier's order (the host's
  misses, then the device's slots in rank order).
- The table: one `CcExpert` per expert of every layer, the device addresses of a
  resident's weights and scales (`qt_chain_resident`, the table's `tensor_dev`), built
  again when the tier's residency changed (`qt_chain_epoch`). The tier is held for the
  forward (`qt_chain_hold`): an eviction clears the flag at once and moves the epoch, but
  no swap writes into a resident's buffers until the forward is done.
- A layer whose experts are all in the table has no host step, and its route ids and
  weights stay on the device during the frame: the ids of every layer come down in one
  copy at its end, for the books below. A layer with some
  elsewhere (VRAM was short) signals the host once its routes and rows are down
  (`cc_signal`: a one-thread kernel writes a word of host memory after a system fence).
  The host computes the slots the table lacks as the tier's misses always were
  (`qt_cpu_misses`: the RAM cache, the disk) while the device computes its own, writes
  its rows into an UP buffer and its word; the device waits for the word (`cc_wait_host`)
  before the sum. The host serves a layer after recording the next one, so the device
  has that layer queued when the answer lands. A round trip costs about 8 us a layer.
- The books the host's step kept (the tier's heat, hits and LFRU tick: `qt_chain_count`;
  the resident collection; the hot-pin counts) follow from the ids after the frame, with
  the K/V rows into the host's cache.

A prompt chunk (more than 8 rows) runs its routed experts in the frame too, layer by
layer, where the table holds all of that layer's experts: `cc_moe` sorts the chunk's
slots by expert (a count, the offsets and 64-row tiles, each slot's place) and runs each
expert's rows through the tensor cores (`tcg_moe_k`: gate and up into silu(g) * u, then
down), then each row's sum in rank order. A layer with experts elsewhere, or a prompt
that re-plans the residents (`QT_PREFILL_REPLAN=1`), keeps the host's step in that chunk.

The dense matrices of a prompt chunk go through the same tensor-core GEMM
(`tcg_dense_k`, from `coli_cuda_tensor_gemm_async` for more than 8 rows), where the block
kernel spent 256 threads on each output: TF32 fragments on sm_80 and later, the inputs
rounded to TF32 (a 10-bit mantissa, the fp32 range), int8 and int4 codes staged exact
with the row's scale after the sum, int4 per 64 times its group's scale as staged, sums
in fp32. Against an fp64 reference the outputs are within 2e-4 to 4e-4 (relative to the
largest); the CPU's own experts quantize their inputs to int8. `COLI_CUDA_TC_GEMM=0`
keeps the block kernel.

A prompt chunk's attention (16 rows or more, `COLI_CUDA_CHAIN_FLASH`; no selection list)
runs on the tensor cores too (`cc_attn_fk`, sm_70 and later, head dim 64, 128 or 256): a
block takes 16 query rows, the query heads of one kv head times 16 / GQ tokens, so each
block of 64 keys is staged once for all of them as f16; two passes (each row's max and
sum, then exp(s - m) * V into the fragments, divided by the sum at the end). The kernel
it replaces gave each (head, row) a block that read every K and V row again. On the
35B, a 1024-token prompt: 226 ms of attention -> 23 ms; a 467-token prompt 819-837 ->
890-903 tok/s; the loss above 3.2223 -> 3.2271.

On sm_80 and later `cc_attn_f2` takes it instead (one pass, FlashAttention-2's running max
and sum): a block of four warps takes 64 query rows (the query heads of a kv head dividing
64), so a block of keys is staged once for four times the rows, and its scores never leave
the registers. The m16n8k16 accumulator is the A operand's layout, so exp(s - m) goes from
a score tile straight into P V, and each row's output fragments are rescaled as its max
moves. Windows, rings and sinks as `cc_attn_fk` takes them; head dim 256 reads Q from
shared memory each block (the registers hold O). Flint at 16K tokens (RTX PRO 4000): the
attention 9.6 s -> 1.09 s, the prompt 671 -> 1032 tok/s before the host pool's GGUF
staging; Qwen3.6 UD-Q4_K_M 796 -> 1956 tok/s at 16K with both.
`COLI_CUDA_CHAIN_FLASH2=0` keeps `cc_attn_fk`.

On sm_70 and later the same GEMMs run in f16 (`hcg_tile`: `hcg_dense_k`, `hcg_moe_gu_k`,
`hcg_moe_dn_k`), at twice the TF32 fragments' rate and with a tile that keeps more of the
card busy: 256 threads (eight warps) for 64 rows x 64 outputs, K in steps of 64, the next
step's words and inputs fetched into registers while the tensor cores work. int8 and int4
codes are exact in f16 (the row's scale after the sum), int4 per group goes in times its
scale, x is rounded to f16 and held to +-65504 (a NaN stays one), sums in fp32. On the 35B,
a 1024-token prompt: the experts' GEMMs 513 -> 236 ms, the dense ones 234 -> 97 ms; a
467-token prompt 889-910 -> 1561-1609 tok/s; the loss above 3.2271 (TF32) -> 3.2262.
`COLI_CUDA_TC_F16=0` keeps TF32 where it runs.

It is off with cache routing or the agreement meter (their state crosses tokens on the
host), a gate bias, more groups than the kernel keeps, or a tier on more than one card.
`COLI_CUDA_CHAIN_MOE=0` keeps the host's step. A decode step's text is the same either
way; a prompt's rows sum in another order and with TF32 inputs, and the greedy text may
part where two tokens are that close. The teacher-forced loss after a 600-token prompt
(125 tokens scored, `PPL=1`) is 3.2223 nats a token with the experts in the frame,
3.2326 with the host's step, 3.2181 with no tensor cores at all and 3.2340 on the CPU.

**Measured** (RTX PRO 4000 Blackwell 24 GB, i5-13500, Qwen3.6-35B-A3B int4-g64, two runs
of `bench_stream.py` each, two rounds):

| | decode tok/s | prefill tok/s |
|---|---|---|
| per matrix (`COLI_CUDA_CHAIN=0`) | 33.6-34.2 | 60 |
| the chain, the host's step, the block kernel (`COLI_CUDA_CHAIN_MOE=0 COLI_CUDA_TC_GEMM=0`) | 58.9-59.5 | 71 |
| a decode step's experts in the frame, a prompt's on the host, the block kernel | 85.3-86.0 | 71 |
| the same with the tensor cores' dense GEMM | 86.0 | 120 |
| every step's experts in the frame, the tensor cores (the default) | 85.9-86.1 | 809-834 |
| the default at `CUDA_EXPERT_GB=8` (a third of the experts in VRAM) | 49.1-50.2 | 114-116 |
| the host's step at `CUDA_EXPERT_GB=8`, the block kernel | 45.1-45.7 | 63 |

With every expert in VRAM a decode token is the device's 10 ms, and a 467-token prompt
0.57 s instead of 6.6. With a third of them (every layer then has some on the host, so a
prompt keeps the host's step for its experts),
the CPU computes two thirds of the slots (about 10 ms a token) and the layer after waits
for them: the hybrid step overlaps the device's experts with the host's, not the layers.
The host pool below takes a prompt's share off the CPU.

### The residents follow the conversation (`COLI_CUDA_CHAIN_PROMOTE`, on by default)

The warm start fills the card from the heat file: the history's routes. A conversation on
another topic misses it -- on the RTX 3070 (2748 of Qwen3.6-35B's 10240 experts resident)
a story prompt found 44.5 % of its decode routes in a set trained on a question about MoE
routing, against 82 % for that question itself -- and every miss is a row the CPU computes
while the device waits (`qt_cpu_misses`, about 0.1 ms each).

After a decode-sized step the chain asks the tier for promotions (`qt_chain_promote`): up to
`COLI_CUDA_CHAIN_PROMOTE` (4) of the step's misses routed at least `COLI_CUDA_CHAIN_PROMOTE_MIN`
(2) times in this conversation, the worth the most first. An expert's worth is its heat --
the history's routes, decayed, plus this conversation's (`qt_chain_count` for the residents,
`qt_note` for the misses) -- plus this conversation's routes once more, so that a conversation
away from the history overtakes within tens of tokens. Each takes the device buffers of the
coldest of 64 residents drawn at random on its device, where it clears the tier's admission
margin (`tier_should_promote`: a quarter and four over the victim; the same rule as the LFRU
tick, which keeps running once in 16 tokens). The chain stages the newcomer's RAM bytes
into a page-locked slot (`qt_chain_promote_stage`: as the RAM holds them for int8 and
pair-ordered int4 rows, or converted from planar int4 blocks into the device's signed pairs)
and from there into the victim's buffers on its copy stream (`cc_host_up`, six copies an
expert), while the next steps run. The tier's uploader thread does not take part:
its synchronous copies take the driver's lock from another thread while the chain's frame
waits for the host (`cc_wait_host`), and that deadlocked inside the driver on the 3070.
Once the copies landed (`cc_host_up_done`, checked after the next steps) the tier hands the
buffers to the newcomer (`qt_chain_promote_done`) and it is resident; the victim computes on
the CPU from the step after its eviction on (its int8 copy rebuilt by `tier_rebuild_evicted`
before that forward, in parallel, where `QWEN_EXPERT_KERNEL=0` needs one; planar int4 keeps
its RAM blocks). The chain's table takes the changes entry by entry
(`qt_chain_changes`: the tier records every change of residency; the whole table is read
again only where the tier lost count), and the staged table of a prompt chunk follows when
a chunk needs it.

The brake: where the last 64 decode steps found `COLI_CUDA_CHAIN_PROMOTE_HITS` percent (75)
or more of their routes on the card, the step promotes nothing and its misses are dropped. A
conversation the history fits pays for every swap -- a promoted expert displaces a resident
the history still routes to -- and gains little; one away from it, at 40-65 % hits, does not
reach the brake for hundreds of tokens. `0` brakes never.

The brake holds only where the warm start's set came from a history: a heat file with this
model's shape, its layers and experts (`qt_chain_history`). Without one the set is the experts
in index order, no conversation's, and the brake starts off (the chain's startup line says so).
On a Google Colab T4 (6605 of 10240 experts in VRAM, OMP 4 threads, no heat file) the 41-token
story at 75 held at 78-79 % hits, 24.4 / 23.9 ms a token over 200 / 600 tokens; without the
brake it reached 86 / 94 %, 22.6 / 19.7 ms. Through the server's path on another such T4 a chat
of 5 turns (1458 answer tokens, no heat file) took 29.4 s instead of 36.7. On a rented RTX 3070
(EPYC 7B13, 2654 experts in VRAM, OMP 12 threads) the same runs differ by 0.6 and 2.0 % (17.5 /
14.8 ms at 75 against 17.6 / 14.5), and the chat's session by less than 1 %.
`COLI_CUDA_CHAIN_PROMOTE_HITS`, when set, wins either way.

**Measured** (RTX 3070 alone, Qwen3.6-35B-A3B int4 per row, `COLI_PLACE=auto
COLI_TRUNK_PROBE=0 CUDA_EXPERT_GB=auto`, OMP 12 threads, the heat file frozen; decode ms a
token = (N / speed - TTFT) / (N - 1)):

| prompt, heat file | tokens | `COLI_CUDA_CHAIN_PROMOTE=0` | the default (4 a step, the brake at 75 %) | the brake at 80 % |
|---|---|---|---|---|
| a story; the heat file from the MoE question | 200 | 44.5 % hits, 24.3 / 24.8 ms | 64.6 % hits, 20.8 ms (548 promoted, no step braked) | 64.3 %, 23.1 / 23.2 / 22.6 ms (543 promoted) |
| the same | 600 | 38.2 % hits, 26.4 ms | 63.0 % hits, 22.9 ms (1062 promoted, 70 steps braked) | 63.4-67.0 %, 20.8 / 21.6 ms (1008-1075 promoted, 0-21 braked) |
| the MoE question; its own heat file | 200 | 81.5-82.4 % hits, 14.9 / 15.0 ms | 82.2-82.4 % hits, 15.3 / 14.9 ms (12 / 4 promoted, 196 / 198 of 200 steps braked) | 79.6-81.7 %, 15.5-17.9 ms (79-158 promoted, 56-155 braked); without the brake 16.7 ms (166 promoted) |

Away from the history a token gets 10-15 % cheaper over 200 tokens and 13 % over 600, where
the set keeps adapting until the brake takes over near the end; on the very prompt the
history was trained on (80-82 % hits here) the brake holds from the first steps and nothing is
lost, where at 80 % it was marginal and let 79-158 swaps through. Eight a step instead of four
moved the same experts in 200 tokens (the margin limits, not the rate) and measured the same
within noise; `0` keeps the warm start's set. With rows in RAM (an int8 container, or
`QWEN_EXPERT_KERNEL=0`) the greedy text with and without is the same. A promotion costs the
engine thread about 0.15 ms (the staging memcpy) and the victim's int8 rebuild about 0.85 ms
of one CPU thread before the next forward.

With planar int4 in RAM (`QWEN_EXPERT_KERNEL=1`, an int4 container's default) the staging
converts the newcomer's blocks into the card's signed pairs (`planar_to_pairs_signed`, an AVX2
path or a scalar loop). A promoted expert then runs `cc_moe` on the card instead of
`expert_ffn.h` on the CPU, which rounds its input to int8 (`QWEN_EXPERT_ACT`): its result moves
in the last digits, and the greedy text can part where two tokens are close. Measured on a
Google Colab L4 (Xeon, 6 cores / 12 threads, 52 GiB) with the int4-g64 container at
`CUDA_EXPERT_GB=5` (1823 of 10240 experts in VRAM after the warm start), OMP 6 threads, 200
tokens and the heat file of one origin/dev run at `auto` on a 652-token question about MoE
offloading, frozen (every run starts from a copy): origin/dev 561966ec (planar int4 refused)
against the promotion, after one warm-up run, in A B B A order on one VM; the last two rows on
another, both engines built without AVX2 (`ARCH=x86-64-v2`). tok/s is the engine's `Speed`
line, ms a token the `COLI_TIMERS` step() total
([raw](experiments/qwen36-cuda-promote-planar-2026-10-25-raw.txt)):

| prompt, engine | decode VRAM hits | promoted | ms a token | tok/s |
|---|---|---|---|---|
| 41 tokens, origin/dev (4 runs) | 20.8 % | 0 | 33.0-33.1 | 24.99-25.05 |
| 41 tokens, the promotion (4 runs) | 52.6 % | 591 | 26.3-26.4 | 29.96-30.05 |
| 664 tokens, origin/dev (2 runs) | 19.6 % | 0 | 33.2-33.3 | 24.42-24.46 |
| 664 tokens, the promotion (2 runs) | 53.9 % | 715 | 26.0 | 29.62-29.65 |
| 41 tokens, no AVX2, origin/dev (4 runs) | 19.7 % | 0 | 305.1-305.6 | 3.21-3.22 |
| 41 tokens, no AVX2, the promotion (4 runs) | 50.9 % | 598 | 202.5-203.0 | 4.78-4.79 |

Choosing and staging take the engine thread 0.46-0.58 ms a promotion with AVX2 (332-342 ms for
591, 332-333 ms for 715), about what rows take on the same L4 (0.50-0.51 ms: 354-359 ms for
709, origin/dev with `QWEN_EXPERT_KERNEL=0`), and 0.59-0.60 ms without AVX2 (355-359 ms for
598: GCC vectorizes the loop with SSE2). Without AVX2 the CPU's kernel for the misses is far
slower, so both engines are. RSS is 19.2 GB on both sides (`QWEN_EXPERT_KERNEL=0` keeps int8
copies: 44.0 GB, 46.1 GB with its promotions). With every expert in VRAM (`auto`, the L4's
default) nothing is promoted, the text is origin/dev's and the speed the same within 0.5 %; the
warm start converts with the same AVX2 block (`planar_to_pairs`): 14.4 to 8.8 s there, 7.6-7.7
to 6.4-6.5 s at `CUDA_EXPERT_GB=5` (warm runs; origin/dev's first runs on those VMs took 15.7
and 10.9 s).

The greedy text parted from origin/dev's on both story prompts (at character 204 of 864 and 350
of 863); with `COLI_CUDA_CHAIN_PROMOTE=0` it is origin/dev's (checked on the 41-token one). The
teacher-forced loss does not get worse (`PPL=1`, `CUDA_EXPERT_GB=5`, the heat file of one
32-token run on the story prompt, frozen): 3.0719 nats a token with promotions (508) against
3.0774 without, 3.0725 with every expert in VRAM (`auto`) and 3.0728 with
`QWEN_EXPERT_KERNEL=0`, on 250 tokens of prose after 600; 0.2984 (462) against 0.2998, 0.2960
and 0.2953 on 200 tokens of the story after its 41-token prompt.

### A prompt's experts past VRAM: the host pool (`COLI_CUDA_HOST_POOL`, on by default)

A prompt chunk routes nearly every expert of a layer (467 rows, 8 slots each, over 256
experts: 15 rows an expert), and the CPU's step costs per row where a copy over the bus
costs per expert: 1.65 MB of int4-g64 per expert of the 35B is 0.13 ms at the 12.8 GB/s of
a PCIe 4.0 x8 link, the CPU's 15 rows about 0.7 ms. So a chunk of more than 32 rows copies
a layer's non-resident experts into VRAM and runs them with the residents:

- The source is the engine's RAM cache as it is, page-locked at the warm start
  (`qt_note_host`, `coli_cuda_host_register`): expert_ffn.h's planar int4 blocks, which the
  f16 GEMMs read directly (`CcExpert.planar`: in each 64 weights' 32 bytes, byte k holds
  weight k low and k + 32 high, both v + 8). No second copy in RAM, no conversion. On when
  the RAM holds every expert (no slot is ever recycled), one card, int4-g64 and the f16
  GEMMs (`coli_cuda_planar_rows`).
- The pool follows the residents (`qt_chain_host_follow`). The chain's promotions and the
  tier's LFRU swaps move experts between the card and the RAM after the warm start; before a
  prompt chunk of 128 rows or more (`Q36CC_FOLLOW_ROWS`), the experts that left the card are
  page-locked into the pool and those that came in are released, so the pool is the
  non-residents and nothing else. Without it an expert that left the card was in neither, and
  a chunk's layer holding one went to the host's step whole. It costs about 0.3 ms for each
  expert moved on the T4 below, and after a long answer that is more than the copies give
  back to a chunk of a few dozen rows: a smaller chunk leaves the pool behind for the next
  large one, its layers holding an expert that left on the host's step as before. An expert
  whose page-lock is refused stays out until the next follow.
- The staging (`cc_fetch`, `cc_fetch_wait`, `cc_fetch_done`): one layer's worth of VRAM,
  the most non-resident experts any layer has, filled on a copy stream of its own. The copy
  of the next such layer starts once the layer before has run its experts' GEMMs, so it
  overlaps that layer's sum and the next layer's attention; one buffer, because the copy
  is the longer of the two and a second one would buy little.
- The prompt chunk's table points each staged expert at its place in the staging; a
  decode step keeps the table of the residents and the CPU's share (a row at a time,
  the CPU's kernel is faster than a copy).

The chunk's rows matter as much: every chunk copies the layer's experts again, so the chain
now sizes a chunk from the card's free memory (`COLI_CUDA_CHAIN_ROWS`, unset: the most of
256, 512, ... 4096 rows whose scratch takes half the free memory less the staging; the
line `rows per prompt chunk` says which). With every expert in VRAM larger chunks help
too: fewer, fuller tiles per expert.

**Measured** (the same box, two runs each, `bench_stream.py`):

| | 467-token prompt | 1727-token prompt | decode after each |
|---|---|---|---|
| `CUDA_EXPERT_GB=8`, the host's step, 256 rows (before) | 127-128 tok/s | 122-125 | 52 / 43 |
| `CUDA_EXPERT_GB=8`, the host pool, 4096 rows | 365-385 | 997-1037 | 52-54 / 45-46 |
| every expert in VRAM, 256 rows (before) | 1626-1686 | 1705-1732 | 86 / 65 |
| every expert in VRAM, 4096 rows | 1922-2137 | 2321-2516 | 86 / 65 |

At `CUDA_EXPERT_GB=8` the 1727-token prompt copies 11 GB once (6714 experts staged in 27
layers) in about 0.9 s; the host pool at 256 rows copied them seven times (215-225 tok/s). The teacher-forced loss
after a 2000-token prompt (200 tokens scored): 1.7567 nats every expert in VRAM, 1.7590 the
host pool, 1.7555 the host's step. The staging takes about 450 MB of VRAM here;
without room for it the chain says so once and the chunk keeps the host's step.
`COLI_CUDA_HOST_POOL=0` keeps the host's step for a prompt's non-resident experts.

**After a conversation** (the pool following the residents): a Google Colab T4 high-RAM VM
(15 GiB, 6605 of 10240 experts on the card under `auto`, 3635 in the host pool), the
record's harness (`bench.c`), a warm-up and 4 runs each in A B B A order, min-max of the runs
(record: `docs/experiments/qwen36-cuda-host-pool-follow-2026-10-26/`). A 64-token prompt and
its 32 teacher-forced steps, then a 512-token prompt in the same process:

| | 512-token prompt, after | 64-token prompt | decode after 64 / 512 |
|---|---|---|---|
| the pool as the warm start left it (before) | 179.19-180.94 tok/s | 73.78-79.01 | 38.85-40.09 / 46.81-47.73 |
| the pool following the residents | 436.44-441.96 | 76.09-77.90 | 39.99-40.50 / 47.28-47.56 |

Two conversations replayed through the server's path (`bench.c --session`, each answer
teacher-forced), the time to the first token of every turn summed, and of the turn after the
longest answer:

| | agent, 8 turns: all / turn 8 (35 tokens) | chat, 5 turns: all / turn 3 (36 tokens) |
|---|---|---|
| before | 46.7-47.1 s / 0.92-0.94 s | 3.4-3.5 s / 0.76-0.77 s |
| the pool following the residents | 18.7-18.8 s / 0.90-0.92 s | 3.4-3.5 s / 0.76-0.77 s |

The 64-token run, its 32 steps with promotions included, gives the same logits in every run of
both; after 512 the teacher-forced loss over the next 32 tokens is 2.9578 before and 2.9328
with the pool following, the layers that held an expert that left the card now in the chain's
frame on its f16 GEMMs instead of the host's path. No chunk of the chat reads 128 tokens: it
runs as before, with the same logits. In one run each with a diagnostic line (`diag.txt` in
the record) the follow took 58 ms of the 512 prompt's 1.14 s (98 experts joining the pool, 98
leaving it), and 0.08 to 147 ms before the agent's prompts, 365 ms in all. `threshold.txt` has
the runs behind the 128 rows.

**A GGUF's experts** take the same road. When the tier takes them (`qt_init_gguf`), the
engine holds the file's rows in RAM instead of the rows interleaved by 16, so the tier and
the host pool read them in place and the CPU computes its share eight rows at a time
(`gq_rows_f8`). An interleaved slot had unpacked each expert into a second buffer of its
own: Flint UD-Q3_K_XL (an 18.5 GB file) took 30.6 GB of RSS, 19.1 GB now. The staging holds
a layer's experts at the largest layer's three matrices, 256-byte aligned: one copy an
expert where its gate and up end on 256 bytes, three otherwise. A layer whose rows are not
whole words keeps the host's step. The tier keeps the staging out of its expert budget
when not every routed expert fits (`qt_reserve_short`). Taken from the mirrors' room
instead, a 16K prompt found no memory for the routed experts' buffers and the chain took
the host's step from then on. Where the mirrors or a step's buffers still lack memory, the
staging gives its VRAM back. The room it leaves free for the mirrors to grow into is the
room the tier left them: a window's or sinks' model's (Flint, Muse Spark), whose chain needs
its mirrors whole. Another model's mirrors take the staging's memory back when they grow.
Before, the staging left the mirrors' room at the full context free for every model:
Qwen3.6 UD-Q4_K_M with 7 GiB free kept 1.3 GB that no one had set aside, found none to stage
in, and read every prompt at 97 tok/s; 570 / 656 now at 1K / 4K tokens. At 16K its float32
mirrors take the staging's memory and the prompt keeps the host's step (96 tok/s).

Before, a GGUF's prompt computed its non-resident experts on the CPU while the device
waited: 1.8 s a 4096-row chunk for the last two layers of Flint on a whole RTX PRO 4000,
and 58 tok/s with 7 GiB free. Measured on the RTX PRO 4000 (tok/s; the reference GGUF
engine on the same card in the last column):

| Flint UD-Q3_K_XL | before | the host pool and `cc_attn_f2` | reference |
|---|---|---|---|
| 7 GiB free, 1K tokens | 60 | 512 | 317 |
| 7 GiB free, 4K tokens | 58 | 1696 | 378 |
| whole card, 1K tokens | 758 | 1577 | 1852 |
| whole card, 4K tokens | 1011 | 2344 | 2060 |
| whole card, 16K tokens | 715 | 2276 | 1988 |

Decode is the same: 111.5 against 111.3 tok/s on the whole card, and 50.9 against 53.3
with 7 GiB free.

### A GGUF's experts at their own size (`COLI_CUDA_EXACT_SLOTS`, on by default)

A GGUF's layers mix block types (Flint UD-Q4_K_XL: IQ3_S, Q4_K, Q5_K and Q6_K among its
experts), and the arena gave every expert the room of the largest layer's three matrices.
On an RTX PRO 4000 that slot was 2.45 MB for an average expert of 1.91 MB: the card held 8140
of the model's 12032 experts. With qwen36's GGUFs on one card each expert now takes a slot of
its own layer's matrices, carved from one allocation of the budget as the warm start fills it;
a freed slot waits for the next expert of its size, and a swap pairs slots of one size. The
budget counts each expert at its slot. The qwen38 chain reads the arena as slots of one size
(`qt_chain_arena`) and does not ask for this. Measured on the RTX PRO 4000 (tok/s; prompts of
1K, 4K and 16K tokens):

| | on the card, 32K context | prompt | on the card, 8K context | decode |
|---|---|---|---|---|
| Flint UD-Q4_K_XL, the largest layer's slot | 7591 | 782 / 1925 / 1871 | 8140 | 66.5-73.7 |
| the layer's own slot | 9861 | 1078 / 2096 / 2066 | 10700 | 89.9 |
| Flint UD-Q3_K_XL, the largest layer's slot | 11520 | 1567 / 2246 / 2232 | | 105.4 |
| the layer's own slot | 12032 | 1637 / 2295 / 2239 | | 105.2 |

With 7 GiB free, Flint UD-Q3_K_XL: prompt 528 / 1762 -> 540 / 1725 tok/s at 1K / 4K, decode
53.3 -> 54.5. Qwen3.6 UD-Q4_K_M (2 sizes, 1.82 to 1.95 MB): 1791 -> 1921 experts on the card,
decode 46.6 -> 45.6.

`COLI_CUDA_EXACT_SLOTS=0` keeps the largest layer's slot for every expert.

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
