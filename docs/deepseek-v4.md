# DeepSeek V4 target engine (colibri CPU)

[简体中文](deepseek-v4.zh-CN.md)

This is the target-only DeepSeek V4 Flash engine for the first PR of the V4
split. DSpark speculative decoding is intentionally excluded and belongs in a
separate stacked follow-up.

## Scope

- Production code is in `c/deepseek_v4.c`; the experimental public engine and
  session API is in `c/deepseek_v4.h`.
- Official sharded safetensors checkpoints load through shared `st.h`.
- Standard MXFP4 matrix multiplication uses shared `quant.h`.
- Unified `c/coli` routes `run`, `chat`, `serve`, and `web` to V4. Serving keeps
  the engine and caches warm across requests.
- `--no-dspark` is a compatibility no-op. This PR has no DSpark model, memory
  tier, or speculative loop.
- Build targets are x86-64/aarch64 Linux and Windows/MSYS2.

Destroy every session before destroying its engine.

## Shared migration status

| Checkpoint path | Current implementation | Follow-up |
|---|---|---|
| Safetensors index/range reads | shared `st.h` | done |
| fmt7 standard MXFP4 matmul | shared `quant.h` | done |
| fmt7 resident rows16 expert cache | temporary V4-private layout | **TODO:** migrate after upstream exposes a resident rows16 API |
| fmt8 E4M3 + UE8M0 128x128 scales | shared `st_read_scale_f32` + `quant.h` `matmul_fp8` | done |

Only the rows16 resident-cache layout remains V4-private. Its
`TODO(upstream-fmt7-rows16)` marker names the shared API still needed before
that specialized cache layout can be removed.

## Memory policy

A typical checkpoint has 43 transformer layers, hidden size 4096, and 256
routed experts per sparse layer with top-k 6. Dense weights occupy about
6.27 GiB and a resident BF16 output head about 1.06 GiB. Routed-expert weights
are streamed and cached according to the RAM budget.

The planner reserves workspace and a minimum expert working set, then enables
dense/head residency and grows the expert cache when memory permits. Dense
residency is independent of DSpark and works in this target-only build,
including with the legacy `--no-dspark` option.

`--ram GiB` is a planner budget, not an OS-enforced limit. Without it, the
budget is derived from currently available OS memory.

## Download

```bash
hf download deepseek-ai/DeepSeek-V4-Flash-0731 \
  --local-dir /path/to/DeepSeek-V4-Flash
```

A download can finish with a truncated shard even when the client reports
success. If `st.h` rejects a shard as out of bounds, compare every local shard
size with the Hugging Face repository before treating it as an engine failure.

## Build and use

```bash
cd c
make deepseek-v4
python ./coli run --model /path/to/DeepSeek-V4-Flash --ram 32 \
  "What is the capital of France?"
python ./coli chat --model /path/to/DeepSeek-V4-Flash --ram 32
python ./coli serve --model /path/to/DeepSeek-V4-Flash --ram 32
python ./coli web --model /path/to/DeepSeek-V4-Flash --ram 32
```

Generation length: `--ngen` is a ceiling, not a target — answers end at EOS.
If the ceiling exceeds what the context window can hold, the engine clamps it
and says so on stderr; raise `CTX` for genuinely longer answers.

V4 chat uses native model markers. Native serving currently supports greedy
generation and one active KV slot; tools and grammar are rejected. Requests
re-prefill their context, while the process, weights, dense tensors, head, and
expert cache stay warm.

## Validation

The tiny safetensors fixture is generated locally, ignored, and not committed:

```bash
python -m pip install -r tools/requirements-deepseek-v4-tiny.txt
make deepseek-v4-tiny-check
```

This covers loading, teacher forcing, greedy decode, long/repeated sessions,
`--no-dspark` compatibility, and two requests through the persistent
`SUBMIT`/`DATA`/`DONE` protocol.

For a real checkpoint:

```bash
make deepseek-v4-oracle MODEL=/path/to/DeepSeek-V4-Flash \
  MEMORY_GB=32 ORACLE_TEACHER_FORCING=32 ORACLE_GREEDY=20
```

The oracle is target-only. DSpark on/off speed, acceptance, and token identity
evidence belong to the stacked DSpark PR.

## Follow-ups

- Add non-greedy sampling and more serving slots.
- Add shared replacements for the two temporary private quant paths above.
- In the stacked PR, restore DSpark without changing target tokens and report
  DSpark on/off performance and acceptance data.
