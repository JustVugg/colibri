# qwen36: CUDA VRAM expert tier

Applies colibri's placement concept ("route -> place -> overlap -> learn") to
Qwen3.6-35B-A3B one level up from the GLM disk tier: all 10,240 experts live
in RAM, the **hot** ones are promoted into DEVICE_LOCAL VRAM across one or
more GPUs and computed there through the existing shared CUDA backend
(`backend_cuda.cu` expert-group API — no new backend).

> **A second mode.** Besides the Qwen3.6 modes below, the tier has an *fp8
> streaming mode* (`qt_init_fp8`) for an engine whose experts do not all live
> in RAM: `cap` may be smaller than the expert count, the tier copies each
> expert's e4m3 slab and block scales when the engine reports it and keeps no
> pointer into the engine's slot, and promotion happens at report time instead
> of a warmstart. It is exercised by `tests/test_qwen36_tier_fp8.c`; the
> Qwen3.8 engine that uses it comes in its own PR. Everything below describes
> the Qwen3.6 modes unless it says otherwise.

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
- **Memory:** the warmstart frees the RAM int8 copies of VRAM-resident
  experts (rematerialized from the packed int4 copy on LFRU eviction; no
  container access). Peak RSS for the 35B int4 container: ~29 GB with two
  8 GB GPUs.

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
budget, the engine offers each trunk component with its size (lm_head once, the
fused DeltaNet projection of every DeltaNet layer), and the tier prices them
against the experts they would displace, in **bytes saved on the memory bus per
token, per byte of VRAM**:

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

## Measured (Threadripper 3945WX 12C, RTX 3070 8 GB + Quadro RTX 4000 8 GB, Qwen3.6-35B-A3B int4, 200-token decode)

| | 1 GPU (8 GB) | 2 GPUs (16 GB) |
|---|---|---|
| decode tok/s (cold / warm heat) | 9.2 / 9.9 | 10.6 / **11.3** |
| VRAM-resident experts | 4,391 (43 %) | 8,532 (83 %) |
| VRAM hit rate (cold / warm) | 44 % / 95 % | 85 % / 100 % |
| peak RSS | 40 GB | **29 GB** |
| reference: Ollama q4_K_M, same box | 7.5 | 10.5 |

CPU-only baseline of this engine before the tier: 0.35 tok/s.
Numerics: logits cosine vs the f32 CPU reference 0.9992 (dense int8 on),
bit-identical GPU-vs-CPU on the same container (cosine 1.0000001).
