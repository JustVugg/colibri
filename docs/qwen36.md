# Qwen3.6-35B-A3B on colibri

`c/qwen36.c` runs [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
(35B total / ~3B active, Apache 2.0) — a hybrid architecture: 25% Gated
Attention layers, 75% Gated DeltaNet (linear attention) layers, each followed
by a streamed MoE block (256 experts, top-8 + 1 shared). Dense weights stay
resident; routed experts stream from the container through an LRU + pinned
cache. Development notes live in `docs/qwen36-phase01.md` /
`qwen36-phase02.md`.

Architecture-identical checkpoints (same config geometry, e.g.
KAT-Coder-V2.5-Dev) run on this engine unchanged.

## Quickstart

Pre-converted containers (int4 experts, self-contained, ~20 GB):

```sh
# group-scaled int4 (gs64) — recommended, see "Which container" below
hf download Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64 --local-dir ~/Models/qwen36_i4_gs64

# per-row int4
hf download Kreuzzelg/qwen36-35b-a3b-colibri-i4 --local-dir ~/Models/qwen36_i4
```

or convert the original bf16 checkpoint yourself (~70 GB download):

```sh
python3 c/tools/convert_qwen36.py --repo Qwen/Qwen3.6-35B-A3B --out ~/Models/qwen36_i4_gs64 --gs 64
```

Build and chat:

```sh
make -C c qwen36
COLI_MODEL=~/Models/qwen36_i4_gs64 ./c/coli chat
```

`coli` reads the model's `config.json` and matches its `model_type` against the
family registry (`qwen3_5_moe` / `qwen3_5_moe_text` — an exact match, so other
Qwen architectures are not claimed by this engine), picks it, and drives it over the serve protocol — `coli web` and
`coli serve` (OpenAI-compatible API) work the same way.

Direct invocation without the gateway:

```sh
SNAP=~/Models/qwen36_i4_gs64 TOK=~/Models/qwen36_i4_gs64/tokenizer.json \
N_NEW=200 ./c/qwen36 256 4 prompt.txt
```

## CPU prefill batching

On AVX2/FMA CPUs, dense int8 projections process two prompt rows per weight
decode.  The same kernel is used by attention, the router, DeltaNet projections,
and the CPU shared expert.  `S=1` decode keeps the original four-register GEMV,
and non-AVX2 builds retain the established per-row implementation.  The shared
expert additionally batches gate/up/down calls across prompt rows with at most
32 MiB of temporary activations; the CUDA expert tier keeps its per-token shared
work so it can continue to overlap the in-flight GPU groups.

Both optimizations are bit-exact and on by default.  For controlled A/Bs,
`QWEN_DENSE_BATCH=0` restores per-row dense-int8 GEMVs and
`QWEN_SHARED_BATCH=0` restores per-row shared-expert calls.  A positive
`QWEN_SHARED_BATCH=N` limits each shared chunk to `N` rows.

Requirements: ~30 GB RAM for comfortable expert caching and NVMe storage for
the container. The default build is CPU-only; `make -C c qwen36 CUDA=1` adds
the optional CUDA VRAM expert tier documented in
[`qwen36-cuda-tier.md`](qwen36-cuda-tier.md).

## Which container?

The gs64 container carries one scale per 64-weight group instead of one per
row. On GLM, per-row int4 was the root cause of think-mode loops and
never-terminating generations (#455), and group scales fixed them in
controlled A/Bs — with `moe_intermediate_size=512`, Qwen's rows are short, so
per-row quantization error concentrates the same way. The gs64 container costs
~1.7 GB more on disk and a few percent on cold-start; warm decode speed is the
same or slightly better.

## Which checkpoints, and what the banner calls them

Two Qwen checkpoints declare `model_type: qwen3_5_moe_text` and resolve to
this engine:

| checkpoint | layers | experts | hidden | banner |
|---|---|---|---|---|
| Qwen/Qwen3.6-35B-A3B | 40 (10 attention) | 256, top-8 | 2048 | `Qwen3.6-35B-A3B · 35B MoE` |
| Qwen/Qwen3.8-2.4T-A95B | 92 (23 attention) | 512, top-10 | 8192 | `Qwen3.8-2.4T-A95B · 2.4T MoE` |

The registry names a checkpoint by its geometry (`display_variants` on the
`qwen36` descriptor), so the banner says what is on disk. A config that
matches neither, a tiny fixture for instance, is named by its own
`model_type` and measured geometry rather than by a sibling's parameter
count (#1045).

The 2.4T checkpoint is **architecture-identical** to the 35B: same layer
pattern, every engine guard holds, and the registry's planner puts its KV
cache at 1.44 GiB for 8k context and 46 GiB at the 256k maximum, with a
context-free DeltaNet state of 0.55 GiB. What this engine cannot do for it
is hold the experts: the warmstart keeps every expert in RAM by design (see
`--ram` below), which is ~1.4 TB of int4 for 2.4T. Serving it needs the
disk-streaming design, not this one. The conversion and the geometry checks
are in place so that work starts from a verified shape, not from a guess.

### The converter's tensor contract

Both checkpoints ship experts **fused** per layer (`mlp.experts.gate_up_proj`,
`mlp.experts.down_proj`), a one-layer multi-token-prediction head (`mtp.*`,
`mtp_num_hidden_layers: 1`), and the 35B additionally a vision tower
(`visual.*`). `tools/qwen36_tensor_kinds.py` classifies every tensor name
before the first shard is read: layer tensors are converted, `mtp.*` and
`visual.*` are skipped **on purpose** and reported with a count, and a name
the contract does not know stops the conversion. A converter that silently
drops what it does not recognise produces a container that loads and is
quietly missing a tensor; this one refuses instead (the GLM-5.3 precedent).
`tests/test_qwen36_tensor_kinds.py` pins the contract to both real indexes.

### Validating the 2.4T shape without a single weight

`tools/make_qwen36_tiny.py --geometry qwen38-2p4t` builds a fixture with the
2.4T's structural numbers at toy widths -- 92 layers, interval 4, 512 experts
top-10, 16:1 attention heads, 8:1 DeltaNet heads -- and rewrites the shard
into the real layout: fused experts plus an `mtp.*` head. The converter must
split the one and skip the other, and the engine must match the transformers
reference token for token. CI runs it at cache capacities 1, 2 and 512, and
under ASan/UBSan. Locally:

```sh
cd c && make qwen36
python3 tools/make_qwen36_tiny.py --geometry qwen38-2p4t --seed 3 \
        --out q24 --ref-mode full --emit-ref q24/ref_full.json
python3 tools/convert_qwen36.py --model q24 --out q24_c --ebits 8
COLI_DENSE_I8=0 SNAP=q24_c ./qwen36 512 8 q24/ref_full.json
```

(`--seed 3`: the default seed collapses this geometry's reference to one
repeated token, which a shape error could still reproduce; seed 3 yields
twelve distinct tokens over sixteen.)

## `--ram` is not honoured by this engine

`coli --ram` sizes the RAM budget for engines that stream experts from disk on
demand. qwen36 does not: the CUDA expert tier it is built for (#713) requires
full RAM residency of the expert set, so the budget is decided by the container,
not by a flag. The engine reads no `RAM_GB`, and passing `--ram` changes
nothing. Said here rather than silently ignored, because a flag that appears to
work and does not is worse than one that is documented as unsupported.
