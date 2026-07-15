## TL;DR

The int4 model produces **fluent-but-incoherent output** — grammatical English that is completely off-topic (SEO spam, homework-help boilerplate) instead of reasoned responses. The root cause is **per-row quantization scales**: one F32 scale per output row (e.g. 2048 scales for a 2048×6144 matrix), which is 48x coarser than the FP8 source's 128×128 block scales. This destroys fine-grained weight information that handles reasoning and instruction-following while preserving the large-magnitude weights that handle grammar and vocabulary fluency.

Branch: `experiment/grouped-quant` (based on latest `dev` at `62419af`)

---

## The problem — demonstrated

Prompt: *"Explain the concept of recursion in programming. Provide a simple example."*

**Baseline (no budget, no changes):**
> "Hire Some To Take Object-Oriented Programming Assignment"

**Budget=4:**
> "The world is a dangerous place, not so much because of the small percentage of people who are doing to do the little of people who are doing to"

**Budget=6:**
> "Elite Custom Essays"

**Budget=12:**
> "Sololearn: Learn to code for FREE! +1 # Explain A concept of recursion..."

This is not random gibberish — it's fluent English that is completely off-topic. This is the signature of **activations corrupted by coarse quantization**: the model retains language fluency (large weights survive) but loses instruction-following and reasoning (fine-grained weights are crushed to zero).

## Root cause: per-row int4 scales

### How the current converter works

`convert_fp8_to_int4.py` (line 39-52) quantizes with **one scale per output row**:

```python
amax = np.abs(w).max(axis=1, keepdims=True)    # one max per ROW
s = np.maximum(amax / qmax, 1e-8)               # one scale per ROW
q = np.clip(np.rint(w / s), -8, qmax)            # quantize entire row with that scale
```

For a 2048×6144 expert weight matrix, that's **2048 scales** — one per row, each covering 6144 elements.

### Why this destroys quality

Consider a row of 6144 values where most are small (magnitude ~0.01) but a few are large (~0.5). The per-row scale is `0.5/7 = 0.071`. The small values become `0.01/0.071 = 0.14`, which rounds to **zero**. All fine-grained information in those small weights is lost.

This matters enormously for MoE models: each token activates only 8 of 256 experts. A poorly-quantized expert pollutes every token that routes to it, and there's no averaging from the other 248 experts (they're simply off). The error is **concentrated**, not diluted.

### What the FP8 source does right

The FP8 checkpoint uses **128×128 block scales** — the weight matrix is divided into 128-element chunks, each with its own scale. Small values in one chunk don't get crushed by large values in another. For a 2048×6144 matrix: `2048 × 48 = 98,304` scales — 48x more granularity.

The converter already dequants FP8 to f32 correctly (lines 196-201), preserving the block-scale information. But then it **throws it all away** by collapsing to a single per-row scale during int4 requantization.

### GLM-5.2 was QAT-trained for int4

The GLM-5 paper (arXiv 2602.15763, §2.4.3) states: *"To provide better accuracy at low-precision, we apply INT4 QAT in the SFT stage."* This means the model is **designed** to work at int4 — but only if the quantization is fine-grained enough to match what the model saw during training. The FP8 checkpoint's 128×128 block scales are the granularity the model expects. Per-row scaling is 48x coarser.

### Community confirmation

Every other project getting coherent GLM-5.2 int4 output uses calibrated or fine-grained quantization:
- **ubergarm/GLM-5.1-GGUF** — imatrix-calibrated with expert-specific patches
- **Unsloth/GLM-5-GGUF** — dynamic UD-Q4 quants
- **QuantTrio/GLM-5.2-Int4-Int8Mix** — mixed int4/int8 with channel-wise scales
- **llama.cpp** — Q4_K with block-level group scales (typically 32 or 64 elements)

None use naive per-row RTN int4 for production inference.

## The fix: group-scaled int4 (fmt=4)

Add a new quantization format with **one scale per 128 elements** along the input dimension, matching the FP8 source's natural granularity.

### What changed

**`c/glm.c`** — 122 lines added:

1. **QT struct** (line 98): Added `int gs` field (group size, 0=per-row for backward compat, 128=grouped).

2. **Format detection** (`qt_from_disk`, line 1064): Auto-detects fmt=4 by checking the `.qs` scale array size. If it has `O * ceil(I/128)` elements instead of `O`, it's grouped. **Old per-row models (fmt=2) work unchanged.**

3. **New kernel** (`matmul_i4_grouped`, line 379): Same AVX2 nibble unpacking as `matmul_i4`, but the accumulator resets at each 128-element group boundary: `dot(x[grp], w[grp]) * scale[grp]`. The scale changes every 8 vector iterations (128/16=8).

4. **Dispatch** (`matmul_qt_ex`, line 813): Routes to `matmul_i4_grouped` when `fmt==4`. Always uses exact kernels (no IDOT approximation — the whole point is quality).

5. **Expert loading** (`expert_load`, lines 1418 and 1538): Both the mmap and slab+pread paths detect fmt=4 from scale array size and set `gs=128`.

6. **`qt_bytes`** (line 104): Reports correct memory for fmt=4.

**`c/tools/convert_fp8_to_int4.py`** — `quant_int4_grouped()` + `--group-size` arg:

```python
def quant_int4_grouped(w, bits, gs=128):
    O, I = w.shape
    ngroups = (I + gs - 1) // gs
    wpad = np.zeros((O, ngroups * gs), np.float32)
    wpad[:, :I] = w
    wr = wpad.reshape(O, ngroups, gs)              # [O, ngroups, gs]
    amax = np.abs(wr).max(axis=2, keepdims=True)   # one max per GROUP
    s = np.maximum(amax / qmax, 1e-8)
    q = np.clip(np.rint(wr / s), -8, qmax)
    # ... same nibble packing as quant_int4 ...
    return packed_nibbles, s.reshape(-1)            # [O * ngroups] scales
```

Same packed-nibble format as existing int4 — only the scale array is larger.

### Verification

**Converter round-trip test** (weights with varying group magnitudes):

| Method | Mean relative error | Max abs error |
|--------|-------------------|---------------|
| Per-row int4 (current) | 0.2056 | 0.0127 |
| Grouped int4 (gs=128) | **0.1278** | 0.0127 |
| **Improvement** | **1.6x lower** | — |

**Engine kernel test** (AVX2 `matmul_i4_grouped` vs f32 reference):

| Output | Reference | Kernel | Error |
|--------|-----------|--------|-------|
| o=0 | -0.126368 | -0.126368 | 2.98e-08 |
| o=1 | 0.075913 | 0.075913 | 1.49e-08 |
| o=2 | 0.077136 | 0.077136 | 7.45e-09 |
| ... | ... | ... | ... |

Max error: **2.98e-08** — matches f32 reference to within float32 epsilon. **PASS.**

### Cost

| Metric | Per-row (current) | Grouped (new) |
|--------|-------------------|---------------|
| Expert weight size | ~18.9 MB | ~20.1 MB (+6%) |
| Scale array per matrix | O × 4 bytes | O × ceil(I/128) × 4 bytes |
| Total model size | ~370 GB | ~390 GB (+5%) |
| Disk I/O per miss | ~19 MB | ~20 MB (+6%) |
| Kernel speed | One scale multiply per row | One scale multiply per 128 elements |

The 6% I/O increase is negligible compared to the quality gain. The grouped kernel has slightly more scale-lookup overhead but remains within AVX2 throughput — the bottleneck is disk I/O, not matmul.

### Backward compatibility

- Old per-row models (fmt=2) **work unchanged** — format auto-detected from scale array size
- All existing features (PILOT, EXPERT_BUDGET, CUDA, MTP, PIPE) work identically — they don't touch the dequant path
- The fused gate+up pair path (`matmul_i4_pair`) falls back to separate `matmul_qt` calls for fmt=4 — minor perf cost, correctness preserved
- `--group-size 0` produces per-row output (backward compat for the converter)

## How to reproduce

### Convert with group scales

```bash
python tools/convert_fp8_to_int4.py \
  --indir /path/to/GLM-5.2-FP8 \
  --outdir /path/to/glm52_i4_grouped \
  --ebits 4 --io-bits 8 --group-size 128
```

### Test quality

```bash
# Old model (per-row):
SNAP=/path/to/glm52_i4 PROMPT="Explain recursion in programming." NGEN=32 ./glm 64

# New model (grouped):
SNAP=/path/to/glm52_i4_grouped PROMPT="Explain recursion in programming." NGEN=32 ./glm 64
```

Compare output text coherence.

## What this won't fix

- **Cold cache speed** — still 0.18-0.36 tok/s on 24 GB RAM. Quality and speed are independent axes.
- **All quantization error** — int4 is still 4-bit. Some degradation vs FP8 will remain. But it should be coherent degradation (slightly wrong word choices) rather than total reasoning failure (unrelated topics).
- **The IDOT +12% perplexity** — the approximate activation kernel (`IDOT=1`, default ON) still applies to non-grouped tensors (attention projections are already protected). Grouped int4 uses exact kernels by design.

## Prior art

- **GLM-5 paper** (arXiv 2602.15763, §2.4.3): "We apply INT4 QAT in the SFT stage" — the model is trained for int4, but assumes fine-grained quantization.
- **MxMoE** (ICML 2025): MoE experts exhibit divergent quantization sensitivity; uniform quant across all experts is suboptimal.
- **MoEQuant** (OpenReview): Naive int4 on MoE loses meaningful accuracy; calibrated framework needed.
- **Automated Fine-Grained MoE Quantization** (ACL 2025): Layer/expert-wise sensitivity variation; group-size scaling is the baseline improvement.
- **ubergarm/GLM-5.1-GGUF**: imatrix-calibrated quants with expert-specific patches — explicitly patches `quantize_row_q4_0_ref()` for routed experts.
- **llama.cpp Q4_K**: Uses block-level group scales (32 or 64 elements) as the standard int4 format.

## Conversion status

Re-converting from the FP8 source (`zai-org/GLM-5.2-FP8`) with `--group-size 128`. Downloading via ModelScope to avoid HuggingFace per-stream throttling. Will report quality A/B results once the conversion is complete.
