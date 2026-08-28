# Tools

These scripts support model preparation and offline engineering work. They are
not runtime dependencies of the C engine.

- `convert_fp8_to_int4.py`, `download_glm52.py`: model preparation
- `convert_fmt4_to_fmt2.py`: fmt=4 (grouped int4) -> fmt=2 (per-row int4)
  re-quant of a GLM-5.2 container, for Metal-backend compatibility
  (see `docs/METAL-M1ULTRA-FMT2-REPORT.md`)
- `repack_fp8_passthrough.py`: fmt=8 repack (byte-preserved FP8) minting a
  standalone-loadable model directory: routed experts plus the full resident
  family (attention projections including `kv_b_proj`, shared-expert and
  dense-MLP weights), norms/router/embed/lm_head copied byte-identically in
  their original BF16/F32, and the non-tensor files the loader opens at
  runtime (`config.json`, `tokenizer.json`; `generation_config.json`
  best-effort). `--mtp` repacks the multi-token-prediction head as a separate
  pass into the same outdir. Caveat: the minted `kv_b_proj` loads today but
  must not serve batched-path decode until the engine's fmt=8 absorb support
  lands (branch `f8/absorb-fmt8`); the failure is a loud crash, not silent
  corruption -- see the module docstring and `docs/FORMATS.md`.
  Synthetic-fixture-tested only, no real-shard runs yet.
- `make_glm_oracle.py`, `make_glm_bench_model.py`: deterministic fixtures
- `benchmark_cuda_fixture.py`, `eval_glm.py`, `fetch_benchmarks.py`: benchmarks
- `gen_unicode.py`: tokenizer table generation

Run them from `c/`, for example:

```sh
python3 tools/convert_fp8_to_int4.py --selftest
python3 tools/make_glm_bench_model.py --output /tmp/colibri-bench
```

`make_glm_oracle.py` also produces the quantized routed-expert fixtures for the
fmt=6 (E8/IQ3, rotation-bearing) and fmt=4 (grouped int4, no-rotation control)
parity gate (#3/#7). Only the routed experts are quantized; shared/dense/attn
stay f32, and the reference (`ref_glm.json` inside each fixture dir) is computed
from the dequantized weights so the engine reproduces it token-exactly:

```sh
python3 tools/make_glm_oracle.py --fmt6   # -> glm_tiny_fmt6/
python3 tools/make_glm_oracle.py --fmt4   # -> glm_tiny_fmt4/
# verify the engine loads the formats directly (32/32 expected):
SNAP=./glm_tiny_fmt6 REF=./glm_tiny_fmt6/ref_glm.json TF=1 COLI_TEMP=0 ./colibri 64 16 16
SNAP=./glm_tiny_fmt4 REF=./glm_tiny_fmt4/ref_glm.json TF=1 COLI_TEMP=0 ./colibri 64 16 16
```
