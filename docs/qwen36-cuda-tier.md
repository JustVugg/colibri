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
| `shexp` | the shared expert's gate, up, down | 3.1 MB | 126 MB |

Offer order is the placement priority once the budget runs short: `lmhead`,
then `dnproj`, `dnout`, `attnproj`, `shexp`, each in layer order, so a partial
placement is whole layers. `dnout`, `attnproj` and `shexp` were added after
the M10 datapoint in #1652 showed the un-offloaded dense path as the ceiling
on a CPU without AVX2; measured on a 16-core CPU, of the 37.5 ms a decoded
token spends in the DeltaNet stack 23.4 are the input projections, 8.3 the
out_proj and norm, 3.3 the convolution and 2.4 the recurrence -- the matmuls,
not the recurrence, are what the trunk costs. They are served from VRAM on
decode (one GEMV each); a prompt batch keeps the batched CPU matmul.

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
