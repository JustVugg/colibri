# Colibrì support plan — GLM-4.7-Flash GGUF (`deepseek2`)

**Date:** 2026-11-09  
**Status:** importer/registry/engine mapping implemented; first exact reference comparison passed; broader parity and tuning in progress  
**Target:** Docker Desktop model `glm-4.7-flash:31B-UD-Q4_K_XL`  
**Source GGUF:** `~/.docker/models/blobs/sha256/5cecd3746c323936eedcb4867fc64dcc548a9f6a5fd239dea07d678ec51a91d9`  
**Source metadata:** `~/.docker/models/blobs/sha256/eb4dbd985289fed645b0132d79af5f69e75abe13ee32793f87cf4e72eb61c4b9`

## 0. Goal and guardrails

Add correct Colibrì support for this exact GLM-4.7-Flash GGUF, beginning with lossless link-mode import, then reference-validated CPU inference, resource planning, and tuning. Keep it distinct from GLM-5.2/5.3 and DeepSeek V4 even though it shares their MLA/MoE machinery. Reuse the GLM engine only behind architecture-specific GGUF mapping and verified numerical parity.

- Preserve Docker Desktop’s source blob; import with `--mode link` so the 17.52 GB weights are not duplicated.
- Do not change defaults for the existing GLM-5.2/5.3, DeepSeek, or other GGUF variants.
- Require deterministic token/logit checks against llama.cpp before tuning or declaring this model supported.
- Tuning must wait for parity on multiple prompts and enough decode tokens to exercise the expert cache.

## 1. Evidence collected

### 1.1 Downloaded artifact

Docker Desktop lists `glm-4.7-flash:31B-UD-Q4_K_XL`. Its local OCI manifest names the GGUF `GLM-4.7-Flash-UD-Q4_K_XL.gguf`, size 17,520,169,280 bytes (about 16.31 GiB), with a LICENSE layer. The GGUF layer is present locally as a content-addressed blob.

Header-only inspection found 844 tensors:

| GGML type | Count |
|---|---:|
| Q4_K | 306 |
| Q5_K | 23 |
| Q6_K | 49 |
| Q8_0 | 180 |
| F32 | 281 |
| F16 | 5 |

### 1.2 Architecture metadata

The GGUF declares `general.architecture=deepseek2`; its Docker model metadata identifies GLM-4.7-Flash.

| Property | Value |
|---|---:|
| Layers | 47 total; one leading dense FFN layer |
| Hidden width / vocabulary | 2,048 / 154,880 |
| Attention heads / KV heads | 20 / 1 |
| Query LoRA / KV LoRA rank | 768 / 512 |
| MLA key/value dimensions | 256 / 256 |
| RoPE dimension | 64 |
| Routed experts / selected per token | 64 / 4 |
| Expert intermediate width / shared experts | 1,536 / 1 |
| Router metadata | sigmoid (`expert_gating_func=2`), normalized weights, scale 1.8 |
| Declared context | 202,752 tokens |

The tensor inventory shows DeepSeek2-style MLA (`attn_q_a`, `attn_q_b`, `attn_kv_a_mqa`, `attn_k_b`, `attn_v_b`, `attn_output`) and shared/routed FFN tensors. No DSA indexer tensors occur in the file. The current importer maps `k_b`/`v_b` into the GLM loader’s split-KV representation; DSA is explicitly disabled for this model.

### 1.3 Support boundary

The GLM C engine already implements query-LoRA MLA, split `k_b`/`v_b`, dense-prefix and shared/routed MoE layers, and GGUF block matmuls. New work adds a `deepseek2` GGUF name map and a `glm47_moe` family identity; it does not claim the MLA/MoE equations are correct solely from tensor-name similarity.

## 2. Implemented changes

- `c/tools/gguf_import.py`: dedicated `glm47_config` for `deepseek2`; validates all layers’ MLA/FFN tensors and supported router/group settings, emits `glm47_moe`, tokenizer IDs, no-indexer config, and one reserved engine cache/KV row (without enabling a draft head).
- `c/st_gguf.h`: distinct `deepseek2` tensor map to the existing GLM MLA/MoE loader names.
- `c/family_registry.py`: GLM-4.7 display/model ID variant, planner support, context ceiling enforcement, and GLM-47-specific MLA planner geometry without GLM-5.2 DSA indexer overhead.
- Tests: importer config fixture, family/planner/display/model ID, expert inventory, and doctor engine-dispatch assertions.
- Imported directory: `~/colibri-models/glm-4.7-flash-31b-ud-q4_k_xl.colibri` (link mode; its `gguf.json` points to the Docker model blob).

## 3. Current measurements and checks

### 3.1 Import, planning, build

- Real GGUF header generated the expected importer config: 47 layers, one dense prefix, 64 experts, top-4, no indexer, no MTP head.
- Link import completed without copying the 17.52 GB file.
- `make -B colibri` succeeded.
- All 64 `test_family_registry` tests and all 3 `test_doctor_engine_arch` tests passed; the focused GLM-47 tests are included.
- Importer tests pass under `/Users/creemac/mlx-env/bin/python` (NumPy 2.5.3); the system Python 3.14 lacks NumPy.
- A header-only loader check indexed 9,538 tensor views (706 names translated; 138 fused expert tensors split into expert views). The standalone oracle-mode CLI then fell through to its default missing `ref_glm.json`; this was not interpreted as a GGUF load failure.

### 3.2 Planner result on this M4

At context 4096 the current `coli plan` reports approximately 15.9 GB of cold experts, 1.6 GB dense resident, about 5.5 GB usable RAM budget, zero projected warm-expert slots, and disk expert misses as the expected bottleneck. It detects Apple Metal but does not assign a Metal placement budget on this platform. The initial 16 GB host plan/run is therefore CPU + disk-backed experts, not GPU-offloaded inference.

### 3.3 Smoke run (execution only, not parity)

A 16-token CPU generation completed at about 3.43 tok/s. One short output was coherent; an earlier short output was unrelated to the prompt. This variability is why exact token comparison is required.

### 3.4 Reference A/B — first case PASSED

Reference: `/Users/creemac/.unsloth/llama.cpp/build/bin/llama-server` (llama.cpp 0.8.0-dev, build 11985), which has a `deepseek2` graph and GLM-4.7 chat-template support. Comparison used raw, already-templated input to avoid differences in server-side chat rendering:

```text
[gMASK]<sop><|user|>Reply with one word: blue.<|assistant|><think>
```

Settings: CPU, four threads, context 512, temperature 0, top-k 1, top-p 1, eight generated tokens. llama.cpp IDs:

```text
16 13 220 3070 2082 55481 279 6145
```

Colibrì IDs:

```text
16 13 220 3070 2082 55481 279 6145
```

**Result: exact 8/8 greedy-token parity for this prompt and short continuation.** This is a useful first signal, not a broad correctness certification. The llama.cpp server reported 2.15 prompt tok/s and about 2.55 decode tok/s; Colibrì reported about 4.11 decode tok/s. Do not compare these short-run rates as stable benchmark numbers.

## 4. Next validation sequence

### M3 — broaden reference parity (in progress)

**Parity progress:** exact greedy token matches on a 16-token English prompt and a 32-token English continuation. A Chinese prompt matched through token 14 and diverged at generated token 15 (`llama.cpp=17114`, `Colibrì=1467`). This is not broad parity. Verify prompt-token IDs and inspect top logits at the first differing position; tuning remains blocked until the Chinese divergence is explained and more cases pass.

1. Repeat identical raw-prompt greedy comparisons with at least one English and one Chinese prompt; capture prompt token IDs from both engines.
2. Run a 32-token continuation on at least one prompt so the same sparse expert cache is exercised across multiple decode steps.
3. Compare the model’s GLM-4.7 chat template separately after raw-prompt token parity, including thinking mode, end-of-turn/EOS handling, and one tool-call-shaped prompt without requiring execution of a tool.
4. If logits are available via both server interfaces, compare first-token logits/top candidates. Otherwise preserve exact token IDs, stop reason, and output text.
5. Save request/response JSON and concise logs under `logs/glm47_validation/`; do not overwrite the earlier plan or unrelated logs.
6. If a mismatch appears, reduce to one token and investigate template/tokenization first, then attention projection/RoPE and router/expert weighting. Do not tune around a correctness discrepancy.

### M4 — tune only after parity

When M3 passes, run a small `coli tune` sweep on a safely bounded context/output and record the baseline plus each candidate. The current M4 plan predicts disk-miss pressure; tuning should evaluate measured cache capacity and `PIPE`/prefetch behavior, not assume more RAM or Metal helps. Stop and report any failure or disk/memory constraint rather than treating a partial sweep as a profile.

## 5. Repeatable setup commands

```bash
GGUF="$HOME/.docker/models/blobs/sha256/5cecd3746c323936eedcb4867fc64dcc548a9f6a5fd239dea07d678ec51a91d9"
MODEL="$HOME/colibri-models/glm-4.7-flash-31b-ud-q4_k_xl.colibri"
LLAMA_SERVER="$HOME/.unsloth/llama.cpp/build/bin/llama-server"

# Existing imported link-mode directory; do not copy the weights.
python3.14 ./c/coli plan --model "$MODEL" --ctx 4096

# Reference-server shape used in the first comparison (example port 18081).
"$LLAMA_SERVER" --model "$GGUF" --host 127.0.0.1 --port 18081 \
  --ctx-size 512 --threads 4 --n-gpu-layers 0 --no-webui --parallel 1 --jinja

# Colibrì's text mode uses [gMASK]<sop> prefixing by default; supply a raw prompt with
# CHAT_TEMPLATE=0 when matching the exact already-templated llama.cpp completion request.
SNAP="$MODEL" COLI_PROMPT='[gMASK]<sop><|user|>Reply with one word: blue.<|assistant|><think>' \
  CHAT_TEMPLATE=0 NGEN=8 COLI_TEMP=0 DRAFT=0 TOKENS=1 OMP_NUM_THREADS=4 \
  COLI_NO_OMP_TUNE=1 ./c/colibri 0 4 4
```

These commands support a reproducible local check; they are not a promise that tuning is complete.

## 6. Non-goals for the first implementation

- No GGUF-to-Colibrì packed-copy conversion or requantization.
- No speculative/MTP support: the downloaded model has no verified MTP head.
- No long-context claims based only on GGUF metadata.
- No Metal dispatch claims; the planner currently has no qualified Apple-GPU placement budget for this GLM engine.
- No changes to GLM-5.2/5.3 or DeepSeek V4 semantics as a side effect of this architecture.
