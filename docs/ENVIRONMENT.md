# Environment Variables

Reference for the environment variables read by the colibrì engine.

**Generated from `upstream/dev @ 6d3ed7e`** by scanning every `getenv()` site in `c/glm.c`. Defaults and behavior are taken from the source; see [MAINTAINING-DOCS.md](MAINTAINING-DOCS.md) to regenerate this after the code changes.

## Which program reads these?

The C engine binary (`c/glm`, built from `c/glm.c`) reads **all** of these. You rarely export them by hand — the `coli` CLI and `openai_server.py` translate most of their flags into these variables before launching `glm` (e.g. `--temp` → `TEMP`, `--ctx` → `CTX`). See [SETTINGS.md](SETTINGS.md) for the flag → variable mapping. Export a variable directly only to reach a knob the CLI doesn't surface, or to override what the CLI would set.

Format: `VAR` — default — effect.

---

## Common — everyday use

| Variable | Default | Effect |
|---|---|---|
| `RAM_GB` | `0` (auto ≈ 88% of free RAM) | RAM budget in GB for the resident/streamed expert working set. Higher → more experts stay hot → higher cache hit rate. |
| `CTX` | `4096` | Maximum context length (tokens) the KV cache is sized for. |
| `NGEN` | `256` (engine) | Max tokens to generate before stopping (stop tokens can end sooner). `coli --ngen` defaults to `1024`. |
| `TEMP` | `-1` (auto: `1.0` for chat/text, greedy elsewhere) | Sampling temperature. **`TEMP=0` = greedy/argmax = deterministic.** |
| `NUCLEUS` | `0.90` | Nucleus (top-p) mass kept when sampling. Slightly tighter than the official 0.95 because the int4 tail is noisy. |
| `TOPK` | `0` (off) | Top-k filter on the sampling distribution (`0` = no limit). |
| `TOPP` | `0` (off) | Top-p filter (`0` = use `NUCLEUS`). |
| `SEED` | unset → seeded from clock + PID | RNG seed for sampling. **Unset = different every run.** Set a fixed value for reproducible sampling. |
| `KVSAVE` | `1` (on) | Persist the KV cache to `<model>/.coli_kv` so a conversation reopens warm. `KVSAVE=0` disables save+load (lossless round-trip; does not change output). |
| `KV_SLOTS` | `1` | Number of independent KV conversation slots (1–16), used in serve mode. |
| `THINK` | `0` (off) | Emit a `<think>` reasoning block. `THINK=1` turns on visible reasoning. |
| `MTP` | on | Multi-Token Prediction (speculative draft head). `MTP=0` disables it. |

---

## Performance / tuning

| Variable | Default | Effect |
|---|---|---|
| `COLI_METAL` | off | Enable the Apple-Silicon Metal GPU backend. Requires a `make METAL=1` build. |
| `COLI_METAL_GEMM_MIN` | `16` | Minimum matmul rows to dispatch a GEMM to the GPU (below this, stays on CPU). |
| `COLI_METAL_SPIN` | off | Keep a GPU keep-alive spinner running (reduces dispatch latency; costs power). |
| `PIPE` | `0` (off) | Overlap expert disk-load with matmul via I/O worker threads. Byte-identical output; reorders I/O. `PIPE=1` opts in. |
| `PIPE_WORKERS` | `8` | Number of I/O worker threads when `PIPE=1`. Tune to your SSD (fewer avoids over-subscribing cores). |
| `DIRECT` | `0` (off) | Use `O_DIRECT`/unbuffered reads for expert slabs. Helps sustained NVMe; keeps the zero-copy GPU path. |
| `COLI_NO_OMP_TUNE` | off | **Kill-switch** for the OpenMP hot-thread tuning (`OMP_WAIT_POLICY=active` spin + proc-bind). Set `=1` when the CPU is mostly waiting on the GPU (Metal) so spin doesn't steal the shared power budget. |
| `MLOCK` | `-1` (auto: on for macOS) | Wire the streamed expert cache into physical RAM (`mlock`) to dodge the memory compressor. `0` off, `1` force. |
| `CAP_RAISE` | `1` (on) | Let the engine raise the expert-cache cap above `topk` when RAM allows (bigger batches). `0` fixes the cap. |
| `PREFETCH` | `0` | Prefetch depth for streamed experts. |
| `COLI_MMAP` | `0` | `mmap` the weights instead of read()-ing into slabs. |
| `PIN` | unset | Path to a `.coli_usage` file; pins the hottest experts into a resident "hot store" at startup. |
| `PIN_GB` | `10.0` | Size budget (GB) for the pinned hot store when `PIN` is set. |
| `AUTOPIN` | `1` (on) | Auto-pin the hot store from usage history once ≥5000 selections are recorded. |
| `REPIN` | `0` (off) | Live re-pin the hot store every N emitted tokens (RFC). |
| `PILOT` | `0` (off) | Router-piloted cross-layer expert prefetch. |
| `PILOT_REAL` | `0` (off) | Value-preserving real cross-layer prefetch loads (`PILOT_REAL=1` opts in). |
| `PILOT_K` | `6` if `PILOT_REAL` else `8` | Number of experts the pilot prefetches per step. |
| `ABSORB` | `-1` (auto: absorbed for S≤4) | MLA attention absorption mode. |
| `IDOT` | `1` | Integer dot-product kernel. `IDOT=0` uses exact f32 kernels (for A/B numerical checks). |
| `COLI_POLICY` | `quality` | Resource policy: `quality`, `balanced`, or `experimental-fast`. |

---

## CUDA (NVIDIA)

| Variable | Default | Effect |
|---|---|---|
| `COLI_CUDA` | off | Enable the CUDA backend. Requires a CUDA build. |
| `COLI_GPU` / `COLI_GPUS` | unset | Device selection (`auto`, `none`, or a list like `0,1`). Requires `COLI_CUDA=1`. |
| `CUDA_DENSE` | `0` | Place dense (non-expert) matmuls on the GPU. |
| `CUDA_EXPERT_GB` | `0` | VRAM budget (GB) for caching experts on the GPU. |
| `CUDA_RELEASE_HOST` | auto (`1` if >1 device) | Release host-side copies after upload. |
| `COLI_CUDA_ATTN` | off | Run S≤4 attention on the GPU. |
| `COLI_CUDA_PROFILE` | off | Emit CUDA timing. |

---

## Advanced / experimental / debug

These are for testing, benchmarking, or internal use — not part of the everyday surface, and some may change without notice.

| Variable | Default | Effect |
|---|---|---|
| `SPEC` | `1` | Speculative decoding on/off. |
| `DRAFT` | `-1` (auto: 3 with MTP, else 0) | Number of speculative draft tokens per step. |
| `GRAMMAR` | unset | Path to a GBNF grammar file to constrain generation. |
| `GRAMMAR_DRAFT` | unset | Max grammar-forced draft span length. |
| `DSA` | on | Dynamic Sparse Attention indexer. `DSA=0` disables. |
| `DSA_FORCE` | `0` | Force the DSA path on. |
| `DSA_TOPK` | model value | Override the DSA index top-k (testing). |
| `LOOKA` | `0` | Measure router predictability (instrumentation). |
| `I4_ACC512` / `I4_ACC512_TEST` | off | int4 512-wide accumulator kernel toggle / self-test. |
| `NOPACK` | off | Disable weight packing. |
| `DROP` | off | Drop-related debug toggle. |
| `PIN_FILL` | `0` | Fill the pinned store even without usage data. |
| `MTP_DEBUG` / `MTP_PRENORM` / `MTP_SWAP` | off | MTP head debugging / ablations. |
| `STATS` | unset | Write an expert-usage histogram to `STATS=<file>` at end of run. |
| `SCORE` | unset | Scoring/eval mode over `SCORE=<file>`. |
| `REF` / `REF_FORCE` | `ref_glm.json` | Reference-output comparison mode. |
| `REPLAY` | unset | Replay mode. |
| `TF` | unset | Teacher-forcing mode. |
| `CHAT_TEMPLATE` | `1` | Apply the GLM chat template (`0` = raw prompt). |

---

## Server / CLI (`openai_server.py`, `coli`)

These are read by the Python programs (not the `glm` engine), so they don't appear in `glm.c`. They cover the OpenAI-compatible server, tool calling, and the debug view.

| Variable | Default | Effect |
|---|---|---|
| `COLI_DEBUG` | `0` (off) | Tee the engine transaction to stderr, by level. **`1`** = decoded model output stream only (byte-by-byte, on both the tool-call and plain paths). **`2`** = both sides — the fully-rendered prompt the engine received *and* the output, bracketed and correlated by request id, so stderr reads as the whole conversation. Invaluable for seeing what the model received vs. emitted during an OpenCode session. |
| `COLI_TOOL_SALVAGE` | `0` (off) | Opt-in de-mangler: reconstruct a malformed int4 tool call by mapping its lone payload onto the tool's primary parameter. Never rewrites well-formed output; recommended for int4 deployments. |
| `COLI_THINK` | `0` (off) | Make thinking the default when the client sends *neither* `reasoning_effort` nor `enable_thinking`. Any explicit client value still wins. |
| `COLI_MODEL` | unset | Default model directory (fallback for `--model`). |
| `COLI_MODEL_ID` | `glm-5.2-colibri` | Model id reported by the API. |
| `COLI_API_KEY` | unset | Required bearer token for the server. |
| `COLI_MAX_QUEUE` | `8` | Max queued requests. |
| `COLI_QUEUE_TIMEOUT` | `300` | Seconds a request may wait in the queue. |
| `COLI_KV_SLOTS` | `1` | Independent KV conversation slots (→ engine `KV_SLOTS`). |
| `COLI_POLICY` | `quality` | Resource policy (shared with the engine): `quality` \| `balanced` \| `experimental-fast`. |
| `COLI_COLOR` | auto (TTY) | `COLI_COLOR=1` forces colored `coli` output when not a TTY. |
| `COLI_RAW` | `0` | `coli` raw output mode. |

> **Debugging an OpenCode session:** `COLI_DEBUG=1` watches the model's output stream; `COLI_DEBUG=2` shows both sides (prompt + output) as a transcript. Add `COLI_TOOL_SALVAGE=1` on int4 to catch mangled tool calls.

## Set by the CLI (don't usually set by hand)

`coli` / `openai_server.py` set these internally to select a run mode or pass through a flag:

- `SNAP` — model snapshot directory (required by `glm`; set from `--model`).
- `SERVE`, `SERVE_BATCH` — select serve / batched-serve mode.
- `PROMPT` — one-shot text mode.
- `COLI_OMP_TUNED` — internal sentinel guarding the OMP re-exec (see `COLI_NO_OMP_TUNE`); not user-facing.

---

## Worked example — the fast, reproducible Apple-Silicon config

```bash
# fast (sampling, non-deterministic by design):
COLI_METAL=1 DIRECT=1 COLI_NO_OMP_TUNE=1 PIPE=1 PIPE_WORKERS=6 MTP=0 \
  ./coli run --model /path/to/model --ram 113 "your prompt"

# same, but reproducible (greedy):
TEMP=0 COLI_METAL=1 DIRECT=1 COLI_NO_OMP_TUNE=1 PIPE=1 PIPE_WORKERS=6 MTP=0 \
  ./coli run --model /path/to/model --ram 113 "your prompt"
```
