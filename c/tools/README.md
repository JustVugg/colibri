# Tools

These scripts support model preparation and offline engineering work. They are
not runtime dependencies of the C engine.

- `convert_fp8_to_int4.py`, `download_glm52.py`: model preparation
- `qpack_install_policy.py`: source-bound resume decisions and manifest-last
  publication primitives for future qpack download/repack frontends. Hold its
  install session for the full transfer. Qpack v1 has sizes but no content
  hashes, so frontends must also enforce immutable revisions and any available
  transport checksum or ETag. It covers manifest-declared container artifacts,
  not required runtime auxiliaries such as `config.json`. Directory-fsync crash
  durability is POSIX-only.
- `qpack_http_install.py`: dependency-free Hugging Face frontend for that
  policy. It resolves a branch or tag to an immutable commit before opening the
  install session, streams manifest-declared files directly into `.part`
  files, resumes only exact HTTP ranges, and publishes `manifest.json` last.
  `config.json` is required; manifest-declared tokenizer/configuration files
  use the same resumable artifact path, while undeclared optional files are
  copied as verified sidecars when present. Strong ETags, documented server
  SHA-256 values, and a locally computed full-file SHA-256 are persisted in
  `.qpack-http.json` and checked on resume. `HF_TOKEN` is sent only to the Hub
  origin and is stripped on cross-origin HTTPS redirects.
- `qpack_mirror_install.py`: static HTTPS/R2 frontend over the same transfer
  engine. A strict `hashes.json` is required and fetched before the manifest;
  it must cover `manifest.json`, `config.json`, and every manifest artifact.
  The legacy Swiftlet `{"files":{"path":"sha256"}}` schema and the sized
  `qpack.hashes.v1` schema are accepted. The normalized mirror URL plus the
  digest of the exact raw index bytes bind resume state; indexed runtime files receive
  authoritative full-object SHA-256 values, while other valid indexed
  sidecars are ignored. The exact raw index is persisted beside the installed
  container. Mirror installs never inherit
  `HF_TOKEN`; an explicit bearer token can be read from a named environment
  variable and is stripped on cross-origin redirects.
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
python3 tools/qpack_http_install.py OWNER/REPO \
  --revision main --output /path/to/model.qpack
python3 tools/qpack_mirror_install.py \
  https://models.example/qwen/model.qpack \
  --output /path/to/model.qpack \
  --hashes-sha256 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
```

An interrupted qpack install can be restarted with the same command. Keep the
source repository and revision unchanged. The installer rejects HTTP range or
validator inconsistencies instead of appending uncertain bytes. It requires
`Content-Length` for large artifacts and never stages the original BF16/FP8
checkpoint, but it does not convert a raw checkpoint into qpack.

Static mirrors must use HTTPS and publish objects first, `manifest.json` next,
and `hashes.json` last as the remote completion marker. Without
`--hashes-sha256`, the first valid TLS response is trust on first use: the
URL-and-index identity prevents a later snapshot from resuming into those
partials, but it cannot authenticate a mirror that was already compromised on
the first request. Prefer a versioned mirror prefix and an out-of-band index
digest for production distribution. The installer uses exact byte ranges and
`If-Match` when a strong ETag is available; final SHA-256 verification remains
the authority.

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
