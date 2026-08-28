# MiniMax-M3 (rides `c/colibri.c`)

Support for [MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3)
(426B parameters, 60 layers, 128 experts / top-4 + 1 shared) inside the GLM
engine — **not** a sibling binary. `colibri.c` reads `config.json` at startup
and switches to `ARCH_M3`; the control plane says so once, in the
`minimax_m3` `FamilyDescriptor` (`c/family_registry.py`), which declares
`engine_artifact="colibri"` and `engine_group="colibri-core"` — the same
group as GLM, and the reason `coli` hands M3 the GLM binary's env contract,
CAP channel and byte-protocol REPL rather than treating it as a sister
engine. Everything downstream — serve, web, chat, tools, the Vulkan expert
tier — is the GLM plumbing.

```
python3 c/tools/convert_fp8_to_int4.py --arch m3 --repo MiniMaxAI/MiniMax-M3 \
        --outdir m3_i4 --ebits 4 --io-bits 8        # --arch auto also detects it
./c/coli chat  --model m3_i4
./c/coli serve --model m3_i4                        # OpenAI API, tool calling included
```

## Architecture notes

Relative to GLM-5.2 (MLA + DSA), M3 swaps in four pieces:

- **GQA attention.** Plain q/k/v projections, 64 query heads over 4 KV heads,
  explicit `head_dim` 128 (≠ hidden/heads). Per-head **Gemma RMSNorm**
  (`x/rms·(1+w)`) on Q and K *before* a partial split-half NEOX RoPE over the
  first `rotary_dim` dims. KV cache is the full K and V rows per token — no
  latent compression.
- **MSA (Lightning Indexer).** A small scoring branch (4 index heads,
  dim 128) max-pools per-key scores into 128-token blocks and keeps the
  top-16 blocks per query (+1 forced local block). Layers 0–2 are dense/full;
  3–59 are sparse. `COLI_MSA=0` disables selection for A/B runs (full causal
  attention everywhere).
- **swigluoai activation.** `gate = min(g, limit)`, `up = clamp(u, ±limit)`,
  `out = (up+1)·gate·σ(alpha·gate)` with `alpha`/`limit` from the config —
  one dispatch point, `act_glu()`, shared with GLM's silu path.
- **Sigmoid router.** Raw sigmoid weights renormalized over the top-4 chosen
  on bias-corrected scores, times `routed_scaling_factor`, plus the shared
  expert.

## Context and MSA exactness

Two honest caveats worth knowing before raising `CTX`:

- The engine default is `CTX=4096` — a *conservative* default for a model
  whose selling point is long context. Raise it explicitly
  (`CTX=32768 ./c/coli serve …`); `coli plan` sizes the KV reservation from
  the real GQA + indexer-cache formulas, via the descriptor's
  `_minimax_geometry` adapter.
- MSA selection is **exact** for attention windows up to
  `sparse_topk_blocks × sparse_block_size` = 16 × 128 = **2048 tokens** (the
  indexer selects every causal block then). Beyond that it is the model's own
  trained approximation — that is how MiniMax runs it, but teacher-forcing
  comparisons against a full-attention reference will diverge past 2048 by
  construction.

## Tool calling

`coli serve` renders and parses the official `chat_template.jinja` dialect
byte-exactly: `]<]minimax[>[`-prefixed `<tool_call>`/`<invoke>` XML blocks,
nested values via `<item>`/child tags, tool results as `<response>` messages.
Streaming holds output at the first tool-call marker instead of leaking the
raw block. EOS is `[e~[`, resolved by the shared `tok_eos_resolve()` for
`run`, `chat`, `serve` and batched serve alike.

## Validation

`make -C c m3-tiny-check` builds a tiny random M3 checkpoint, converts it
with `--arch m3`, runs the numpy oracle (`c/tools/oracle_m3.py`) and requires
the engine to match every teacher-forced prefill and decode position
token-exactly (CI runs this on every PR — it is the only gate that notices
when a GLM-side refactor of the shared hot paths breaks M3). `IDOT=0` for the
compare: int8 activation quantization can legitimately flip a borderline
argmax.
