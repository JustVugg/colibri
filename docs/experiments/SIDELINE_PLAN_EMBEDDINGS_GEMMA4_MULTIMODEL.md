# Colibri Sideline Plan: Embeddings, Gemma 4 E2B and Multi-Model Serving

**Status:** Sideline plan — embeddings, Unicode normalization, and the first
static router baseline implemented; Gemma 4 E2B compatibility investigation
active.
**Prepared:** 2026-11-18
**Related:** `docs/experiments/DEV_PHASED_PLAN_TTFT_AND_THROUGHPUT.md` (main performance roadmap — kept separate; do not merge)
**Priority order when resumed:**
1. Embedding support in `coli.serve` (first priority, must-have capability)
2. Embedding model compatibility and serving requirements
3. Gemma 4 E2B compatibility (active)
4. Unified multi-model serving architecture (router baseline started)

> Sections 1–5 record the original objectives and investigation. Sections 6–7
> record verified implementation and oracle results; remaining items are marked
> explicitly as open.
>
> **Constraint on resume:** keep this work independent of the main Colibri
> performance roadmap until explicitly directed otherwise.

---

## 1. Embedding support in `coli.serve` (First Priority)

This is a **must-have** capability and should be explored first when this sideline
work resumes.

### Objectives

- Establish whether Colibri's engine already supports embedding inference.
- Inspect existing `coli.serve` CLI options, model-loading paths, inference
  capabilities and API endpoints **before proposing anything new**. Avoid
  reinventing functionality that already exists.
- Identify compatible embedding model architectures and formats.
- Establish whether the required embedding/pooling path is implemented and
  produces correct vector representations.
- Determine how an embedding API can be exposed through `coli.serve`, if it is
  not already available.
- Investigate whether embedding workloads can coexist with the IQ2 model serving
  generation requests, including resource and performance implications.

### Explicit non-assumptions

- **Do not assume** embeddings require a `--supporting-model` argument. The first
  step on resume is to check whether Colibri already has a dedicated embedding
  model-loading or serving mechanism. If it does not, document what would be
  required instead.

### Existing reference implementation (Ollama)

The current reference setup used for embeddings is Ollama:

- Endpoint: `http://localhost:11434/v1/embeddings`
- Model: `nomic-embed-text:latest`
- Round-4 fact: that tag resolves to **`nomic-embed-text-v1.5`**
  (nomic-bert, 12×768, mean pooling, L2-normalised output, 768 dimensions,
  context 2048) — see the full metadata table in Section 5.8.

Use this as the reference when eventually evaluating Colibri's embedding output
(vector format, dimensionality, correctness, latency).

### Suggested resume checks (investigation only, no changes)

- Does the engine expose an embedding/pooling path, or is the forward pass
  generation-only?
- Which model architectures/formats would an embedding GGUF (e.g. a BERT-family
  model such as `nomic-embed-text`) require? Does the loader accept them?
- What endpoint shape (`/v1/embeddings` or otherwise) would fit the existing
  `coli.serve` API surface?
- Memory/CPU footprint of a concurrent small embedding model alongside the IQ2
  generation model.

---

## 2. Gemma 4 E2B compatibility

Establish whether Colibri can load and serve Gemma 4 E2B.

- **Model location:**
  `/Users/creemac/.lmstudio/models/lmstudio-community/gemma-4-E2B-it-GGUF`
- **Intended purpose:** Gemma will act as a *supporting* generation model.
  Qwen 3.6 35B remains the **primary reasoning model**.
- Both should eventually be available through the Colibri serving architecture.

### Open questions on resume

- Does Colibri's model-loading and inference path accept the Gemma 4 E2B GGUF
  (architecture, tokenizer, chat template, any novel tensor shapes)?
- What context/configuration does E2B need to run usefully as a supporting model?
- How would a supporting generation model be exposed alongside the primary model
  (see Section 3)?

---

## 3. Multi-model serving architecture

Explore serving multiple independent models under one `coli.serve` umbrella.

### Conceptual `GET /v1/models` target (illustrative only)

```json
{
  "data": [
    {"id": "qwen3.6-colibri", "type": "generation"},
    {"id": "gemma-4-e2b", "type": "generation"},
    {"id": "nomic-embed-text", "type": "embedding"}
  ]
}
```

This is an **illustrative target, not a claim about existing functionality** and
not an immediate implementation requirement.

### Design principle (named 2026-11-18, round 3)

> **Colibri engines remain independent. Colibri serving becomes unified.**

Multi-model capability must not turn the inference engine into a
multi-model manager. Each engine stays a single-family, single-model process
with its own weights, configuration and context; unification lives in a
serving layer above it (model listing, routing, lifecycle). This is the same
separation `c/cluster.py` already draws for expert workers: the coordinator
never runs inference, it only knows who is running it.

### Architectural requirements

1. **Independent model instances.** Each model has its own weights,
   configuration and context.
2. **Capability-aware routing.** Generation requests reach generative models;
   embedding requests reach the embedding model.
3. **Resource management.** Full RAM residency is the initial design assumption.
   Memory requirements, model lifecycle (load/unload) and concurrent workload
   behaviour must eventually be measured.

### Open questions on resume

- How does the current `coli.serve` process model ownership? One model per
  process, or a shared registry?
- What would capability-aware routing imply for request parsing, model registry
  and the `/v1/models` response shape?
- Co-residency budget: primary Qwen 3.6 35B (IQ2) + Gemma 4 E2B + an embedding
  model — what is the combined footprint under the full-RAM-residency assumption?

---

## Resume checklist

When this sideline work is resumed, in order:

1. **Investigate existing embedding support in `coli.serve`.** Avoid reinventing
   functionality that already exists.
2. **Establish embedding model compatibility and serving requirements**
   (architectures, formats, pooling path, endpoint exposure).
3. **Investigate Gemma 4 E2B compatibility** (loading, serving, supporting-model
   role).
4. **Explore how both capabilities fit into a unified multi-model serving
   architecture** (independent instances, capability-aware routing, resource
   management).

Keep this work independent of the main Colibri performance roadmap
(`docs/experiments/DEV_PHASED_PLAN_TTFT_AND_THROUGHPUT.md`) until explicitly
directed otherwise.

---

## 5. Round 1 findings (2026-11-18, read-only code inspection)

First investigation pass against the codebase. Facts below were verified by
reading the named files; anything marked "inferred" is a conclusion to re-check
on the next pass.

### 5.1 Embedding support in `coli.serve`

- **No embedding endpoint exists.** `grep` for `/v1/embeddings` across `c/`,
  `colibri/`, `ui/` and `docs/` returns nothing outside this document. The
  served endpoints are `/v1/chat/completions`, `/v1/completions`,
  `/v1/messages`, `/v1/systemone` and the image-generation path
  (`c/openai_server.py`, `TEXT_ENDPOINTS` at ~line 7844).
- **No embedding family in the registry.** `c/family_registry.py` `FAMILIES`
  (line 1316) defines 13 families: `glm53`, `glm`, `inkling`, `kimi`, `olmoe`,
  `qwen36`, `qwen38`, `deepseek_v4`, `deepseek_v41`, `mimo`, `qwen_image`,
  `laya`, `gliner_decide`. None is an embedding/pooling model.
- **The non-assumption held:** there is no `--supporting-model` argument
  anywhere in the `c/coli` CLI. `serve`'s options (`c/coli` ~line 4208) are
  single-`--model`, host/port, auth, queueing, KV slots, CORS, allowed hosts.
- **Encoder forward passes already exist.** `c/laya.c` implements a complete
  ModernBERT encoder (28 pre-norm layers, RoPE theta 160000/10000, GeGLU MLP,
  no biases, final LayerNorm; `docs/laya.md` "What the engine computes").
  `c/gliner_decide.c` implements DeBERTa-v3 (24 layers x 1024; registry
  comment at line 1866). Both run inside `coli serve` today, exposed on
  `POST /v1/systemone`.
  - Inference: the pooling head (CLS or mean pooling + L2 normalisation) and
    an embeddings endpoint are missing; the encoder machinery is not.
- **Format mismatch to resolve.** Decision engines load HF safetensors
  directly and unconverted (`c/st.h`; `docs/laya.md`: "no conversion"). LLM
  families load converted GGUF (`tools/gguf_import.py`). The reference model
  `nomic-embed-text` exists as GGUF (Ollama) and as HF safetensors
  (nomic-bert architecture); neither existing loader accepts the GGUF form
  today. A safetensors path via the encoder machinery is the closer fit.
- **Inferred (re-check on next pass):** `nomic-bert` is a BERT-family encoder
  with rotary embeddings and no pooler weights — closer to the ModernBERT
  implementation in `laya.c` than to DeBERTa. Whether `laya.c`'s encoder
  layer can be generalised or needs a sibling artifact is an open question.

### 5.2 Gemma 4 E2B

- No `gemma` entry exists in `family_registry.py`; the registry is the first
  gate a checkpoint hits (root `config.json` `model_type` lookup).
- Serving a new decoder family has historically meant a new engine artifact
  + chain (`c/qwen38.c` + `c/qwen38_chain.h` pattern, `FamilyDescriptor`
  fields `engine_artifact` / `build_target` / `process_names`). E2B must
  follow that pattern; it cannot safely inherit an existing family.

### 5.3 Multi-model serving

- **One model per process, structurally.** `APIServer.__init__`
  (`c/openai_server.py` line 7912) holds exactly one `engine` and one
  `model_id`; `ARCH` is a module global set once in `main()` (line 1133).
  `c/coli serve` takes a single `--model`.
- **Capability-aware routing already exists in single-model form.**
  `APIServer.capabilities()` (line 7979) returns `["chat", "systemone"]`,
  `["systemone", "decision"]` or `["image_generation"]`; `/v1/models` carries
  it (`model_entry()`, line 7950), and requests for a capability the model
  lacks are refused with a 400 that names the right endpoint (e.g. line 8615:
  decision model → "it answers typed…"). The proposed `type` field
  (`generation` / `embedding`) is an extension of this existing mechanism.
- **Existing multi-process precedent:** `coli cluster` (coordinator + expert
  workers, `docs/cluster.md`) already coordinates several engine processes.
  If multi-model serving goes the multi-process route — one engine process
  per model behind one API server — the coordinator/worker pattern is the
  closest analogue in-tree. Alternative (single process, multiple models)
  would require removing the `ARCH` global and per-engine globals — a large
  refactor. Decision deferred; not an implementation plan yet.
- **Resource planning today is per-model.** `c/resource_plan.py` and
  `c/serve_budget.h` plan RAM/VRAM for one engine; co-residency budgeting
  (Qwen3.6 35B IQ2 + Gemma 4 E2B + embedding model under full-RAM residency)
  has no existing mechanism. Measurement needed before any design commits.

### 5.4 Round 1 conclusion

No embedding functionality exists to reuse, so Section 1's "avoid reinventing"
answer is: **the encoder runtime is reusable, the pooling/endpoint/serving
surface is not.** The next pass should (a) read `c/laya.c`'s encoder in
detail against a nomic-bert checkpoint layout, (b) confirm how
`coli serve` selects the engine artifact for a family, and (c) measure the
RAM profile of a small encoder under `c/st.h`'s on-demand paging (it claims
non-resident pages by design — relevant to the co-residency budget).

### 5.5 Round 2 (2026-11-18): how the encoder engines are actually reached

Motivated by the practical question: *"I run Qwen3.6 35B with
`coli serve --model <qwen checkpoint>` — would that also load the BERT?"*

**Answer: no.** `c/coli`'s dispatch is strictly one model per process:

1. `resolve_model(model)` reads the checkpoint directory's config files and
   picks a family from `family_registry.py` (for a Qwen3.6 checkpoint: family
   `qwen36`).
2. `engine_for(model)` (line ~433) maps the family to **its own engine
   binary** via `family.engine_artifact` (`qwen36` → `./c/qwen36`,
   `glm` → the GLM binary, etc.), unless `COLI_ENGINE` overrides the path.
3. `coli serve` launches that one binary with that one model. Nothing else is
   loaded. ModernBERT/DeBERTa code never executes for a Qwen checkpoint.

The encoder engines are therefore reached like this — each is a standalone
artifact, not a shared library loaded beside a generation model:

| Family | Engine binary | Build target | Checkpoint signature it accepts |
|---|---|---|---|
| `laya` | `./c/laya` | `make -C c laya` | `rl_agent_config.json` + `encoder/config.json` + `tokenizer/` (no root `config.json`) |
| `gliner_decide` | `./c/gliner_decide` | `make -C c gliner_decide` | `config.json` + `encoder_config/config.json` (GLiNER2 layout) |

To run one today (from `docs/laya.md`, unchanged):

```sh
hf download convaiinnovations/laya model.safetensors rl_agent_config.json \
   "encoder/*" "tokenizer/*" --local-dir ~/Models/laya
make -C c laya
./c/coli serve --model ~/Models/laya --port 8002    # separate process/port from the IQ2 server
```

What it serves: `POST /v1/systemone`, calibrated **probabilities** for typed
questions. Not embeddings. And the loaders are checkpoint-specific: `laya.c`
looks for Laya's decision-head weights, `gliner_decide.c` for GLiNER2's span
head — pointing either at `nomic-embed-text` would be rejected at resolve
time. So the true statement is narrower than "we have a BERT engine":

> **`laya.c` proves `coli serve` can run a full modern encoder forward pass
> (batched rows, RoPE, f32 weights, ~1.7 GB resident for 421M params, no KV
> cache). It is not a generic "load any BERT" path. An embeddings capability
> needs a new family + engine artifact in that same shape, with a pooling
> output instead of a decision head.**

### 5.6 Architectural insight from round 2

The user's actual setup — Qwen3.6 IQ2 on `:8001`, Ollama embeddings on
`:11434` — is already, in miniature, the Section 3 target:

- **Independent model instances:** separate processes, separate weights,
  separate ports. ✔ (exists today)
- **Capability-aware routing:** absent — the client (the agent) is the
  router; nothing inspects `capabilities` and dispatches by request type
  across processes.
- **Resource management:** manual — the operator chose ports and models; no
  memory budget spans the two servers.

So the realistic shape of multi-model serving may be "one `coli serve`
process per model + a thin router" rather than one process hosting several
models — which matches `c/openai_server.py`'s single-`ARCH` structure and
`c/cluster.py`'s coordinator/worker precedent. Both routes stay open; next
pass should size them against each other (what breaks if `ARCH` becomes
per-engine state vs. what a router must do).

**Correction recorded (2026-11-18, round 2):** "one model per `coli serve`
process" does *not* mean multi-model serving is impossible. It means the
umbrella is a coordinating layer, not a shared process. The in-tree proof is
`c/cluster.py`: `ClusterServer` subclasses `openai_server.APIServer` and fronts
a registry of independent worker engine processes (`ClusterRegistry`,
heartbeats, `expert_endpoints()`), reusing `APIHandler`. One HTTP surface,
several engines behind it — the same shape as the user's current Qwen :8001 +
Ollama :11434 pair. Against Section 3's three requirements:

| Requirement | Native `coli serve` support |
|---|---|
| Independent model instances | Yes — one engine process per model is the native mode |
| Capability-aware routing | Precedent: `ClusterServer` routing + `APIServer.capabilities()` |
| Resource management | Not yet — no cross-process memory budget exists |

The route that is effectively ruled out without a major refactor is
single-process multi-model (de-globalising `ARCH` and per-engine state in
`openai_server.py`). The process-per-model route satisfies all three
requirements with new code *around* `coli serve`, not inside it.

### 5.7 Round 3 (2026-11-18): `cluster.py` anatomy and the minimum router

Read-only inspection of `c/cluster.py` (186 lines, the whole file) and the
request-handling mechanics of `c/openai_server.py`.

**What `cluster.py` actually is.** A registration-and-discovery control
plane, *not* a proxy. Workers `POST /v1/cluster/register` (node_id, host,
port, role), heartbeat to stay fresh, and clients fetch
`GET /v1/cluster/topology` to find them — then talk to workers directly.
The coordinator never forwards inference traffic. So the multi-model router
is genuinely new code; what `cluster.py` proves is the reuse pattern.

**The reuse pattern (`ClusterServer`, lines 112–131).** It subclasses
`openai_server.APIServer` with *no engine and no model*: it skips
`APIServer.__init__` and initialises the connection-cap bookkeeping
(`_conn_lock`, `_conn_live`, `_conn_by_ip`, `_conn_owner`) directly, so the
inherited `process_request` / `_release` machinery works. Its `_Handler`
subclasses `APIHandler` and keeps the shared plumbing for free: Host guard
(`_check_host`), body cap and JSON validation (`read_json`), plus auth/CORS
state on the server object. That is the structural template for a router
handler.

**Two mechanics a router must get right, both with precedents:**

1. **Model-field dispatch.** `APIHandler.check_model(body)`
   (`openai_server.py` line 8420) already does per-request model matching —
   it compares `body["model"]` against the single `server.model_id` and
   raises OpenAI's `404 model_not_found`. A router's dispatch is exactly
   this check generalised: look `body["model"]` up in a registry of
   backends (by model id and capability) instead of comparing against one
   constant. Wire shape unchanged, error shape unchanged.
2. **Streaming passthrough.** SSE responses are sent as
   `Content-Type: text/event-stream`, *no* Content-Length and *no* chunked
   framing — end-of-message framing, flushed per event
   (`openai_server.py` ~9187, ~9632 and the comment at 9635). A router must
   therefore byte-passthrough the upstream's SSE body (forward the status
   line, the event-stream header, then pump bytes until the upstream
   closes), not re-serialise events. Connection: close semantics on the
   downstream side match the upstream's. Re-serialising would re-time
   tokens and drop engine timing metadata — the passthrough is also the
   cheap option.

**Minimum router inventory (the v1 implementation now covers the static
configuration slice; dynamic registration and lifecycle remain open):**

- **Backend registry.** Static config at first (list of {base URL, model
  id(s), capability}); dynamic registration can come later via the existing
  `ClusterRegistry` pattern. `/v1/models` merges entries and maps each
  backend's `capabilities()` list to the `type` field of the Section 3
  target (`generation` / `embedding` / `decision`).
- **Dispatch.** By endpoint + body model: `/v1/chat/completions`,
  `/v1/completions`, `/v1/messages` → a generation backend;
  `/v1/embeddings` (when it exists) → the embedding backend;
  `/v1/systemone` → a decision backend if one is served. Unknown model →
  the same `404 model_not_found` a single server emits today.
- **Passthrough forwarding.** Buffer-and-forward for JSON responses;
  byte-stream passthrough for SSE. Timeouts long enough for queue waits
  (the server's own default is 300 s, `--queue-timeout`).
- **Auth and guards.** Inherited from `APIHandler`. Open design question:
  does the router hold the public API key and call backends keyless on
  loopback, or forward the key? Loopback-keyless backends behind an
  authenticating front is the simpler and more defensible default.
- **Health.** Aggregate backend `/health` probes into one router response,
  or report which backends are down. A dead backend should read as
  "unavailable" for its capability, not as a router crash.
- **Lifecycle (open).** `c/coli` already has per-port pidfile/stopfile
  conventions (`serve_pidfile`/`serve_stopfile`, line 3367) for spawned
  servers. v1 router can assume operators start backends (as they do
  today); lifecycle management is a later concern, not a prerequisite.

The v1 router lives in `c/router.py` as a sibling HTTP service. It assumes
operators start the backends separately. Dynamic cluster registration,
backend lifecycle management, CORS parity, and richer telemetry aggregation
remain open.

### 5.8 Round 4 (2026-11-18): `laya.c`'s encoder vs. the actual reference model

The decisive compatibility question from the resume checklist. Both sides were
read directly: `c/laya.c` (1620 lines, the encoder forward and loader) and the
GGUF the Ollama reference actually serves (parsed from the local Ollama blob
`~/.ollama/models/blobs/sha256-970aa74c…`, the `nomic-embed-text:latest`
manifest's model layer).

**What the reference model actually is** (from the GGUF metadata):

| Field | Value |
|---|---|
| architecture / name | `nomic-bert` / `nomic-embed-text-v1.5` |
| layers × hidden × heads | 12 × 768 × 12 |
| FFN | 3072, **gated** (`ffn_gate` + `ffn_up` + `ffn_down`) |
| context | 2048 |
| positions | **rotary only** (theta 1000, full sequence; no learned position table in the file) |
| attention | bidirectional (`causal = false`), fused `attn_qkv` [768×2304], **no qkv/output bias** |
| norms | BERT-style, **with biases** (`token_embd_norm`, per-block `attn_output_norm` + `layer_output_norm`, `final` implied) |
| token types | `token_types.weight` [768×2] present |
| pooling | `pooling_type = 1` (llama.cpp convention: **mean pooling**); **no pooler/head weights in the file** |
| tokenizer | BERT **WordPiece**, vocab 30522, [CLS]=101 [SEP]=102 =103 |
| weights | F16 (file 274 MB) |

**Reusable from `laya.c` nearly as-is (verified in code):**

- `layernorm()` already handles the optional bias (`b ? v + b[i] : v`, line
  666) — nomic's biased norms fit; ModernBERT's biasless norms were the
  special case, not the general one.
- `attention()` with `window < 0` is full bidirectional attention over
  batched sequences (`row_seq`/`seq_off`), exactly what embeddings need —
  one input string per sequence, several per batch. No causal assumption to
  undo.
- `apply_rope()` + `rope_table()`: nomic wants rotary on **every** layer at
  theta 1000; laya's global/local per-layer policy is a config difference,
  not a missing mechanism.
- The fused QKV GEMM shape (`wqkv` = `[d × 3d]`, optional bias) matches
  nomic's fused `attn_qkv` with no bias.
- The gated MLP loop (line 816: `gelu(u[j]) * u[I + j]`) is the same
  gate×up shape as nomic's `ffn_gate`/`ffn_up`; activation form is already
  config-selectable (`gelu_tanh`). *(Round-4 correction, 2026-11-18: the
  oracle sweep settled it — nomic v1.5 is SiLU-gated, not GELU. See 7.1.)*
- `st.h` safetensors reading, f32-weight loading, one-shot forward with no
  KV cache, and the Vulkan/CUDA hooks (`decide_vk.h` / `decide_cuda.h`)
  carry over unchanged.

**Genuine gaps (must be built):**

1. **Layer wiring: post-norm vs pre-norm.** Laya's loop (lines 801–820) is
   ModernBERT pre-norm — norms *before* the QKV and MLP GEMMs, residual
   added after. Nomic's GGUF tensor names (`attn_output_norm`,
   `layer_output_norm`) are BERT post-norm — norm *after* each residual
   add. The primitives are identical; the composition order differs. A new
   forward loop of the same primitives, not new kernels.
2. **Tokenizer: WordPiece vs BPE.** `load_tokenizer` hard-refuses anything
   but byte-level BPE (line 263: "only a byte-level BPE tokenizer is
   supported"). Nomic is WordPiece (lowercase, `##` continuations,
   greedy-longest-match). This is the largest missing piece — a separate
   tokenizer path (~100–200 lines, well-specified), not a parameter.
3. **Token-type embeddings.** `token_types.weight` must be added at the
   embedding stage (type 0 for every single-string input; BERT segment B is
   unused by an embeddings API). Laya's `type_emb` is the *decision head's*
   type embedding — different thing, not reusable here.
4. **The embedding head itself.** Mean pooling over non-pad tokens (what
   `pooling_type = 1` declares) + L2 normalisation (Ollama's default
   `normalize=true`), then the vector is the output. About twenty lines,
   and the file needs no pooler weights — verified above.
5. **Family, endpoint, loading surface.** A `FamilyDescriptor` keyed on the
   nomic checkpoint, an OpenAI-shaped `/v1/embeddings` ({"input": [...],
   "model": ...} → {"data":[{"embedding":[…768…], "index":0}]}), and a
   loader decision (see below).

**Open decision — loader route, two viable options:**

- **(a) HF safetensors via `st.h`** (the laya precedent: official
  checkpoint, no conversion). Tensor names differ
  (`embeddings.word_embeddings.weight`, `encoder.layer.N.…`); a name-mapping
  table per `st_gguf.h`'s precedent would bridge it.
- **(b) Read the llama.cpp GGUF directly.** `c/gguf.h` already exists for the
  LLM side. Advantage: the exact artifact Ollama serves — the file parsed
  above — becomes the reference input, so output parity can be tested
  bit-for-bit against the user's running Ollama instance. The embedded
  WordPiece vocab (tokens, scores, types) also rides along in the file,
  closing part of the tokenizer gap.

**(b) is the cheaper path to a first correct vector; (a) is the more
colibri-native one. Both stay on the table; the decision belongs to the
implementation phase, not this investigation.**

**Resource note for the co-residency question:** F16 file 274 MB, weights
held f32 after load (laya convention) ≈ 550 MB, activations bounded by
T × wide (worst case 2048 × ~4600 f32 ≈ 37 MB). An order of magnitude below
the IQ2 server's footprint — coexistence on one machine is not a memory
problem; contention for compute during generation is the thing to measure
later.

---

## 6. Proposed order of work (2026-11-18 — proposal only, nothing started)

Answering "where would implementation start, and which model is served?"

**The model question, settled:** the embedding model to serve is
`nomic-embed-text-v1.5` — the exact weights the Ollama reference runs —
because the goal is to replace Ollama, and output parity is only meaningful
against the same weights. ModernBERT is not a candidate model: Laya is a
decision model with no embedding head, and its encoder contributes
*machinery*, not weights. An embedding engine is therefore a **new artifact
running nomic weights on laya.c's primitives** (attention, rotary, layernorm,
gated MLP, batched one-shot forward), not a repointing of `laya.c`.

**Starting sequence, in dependency order:**

1. **Loader route: read the llama.cpp GGUF directly.** `c/gguf.h` already
   exists, and the GGUF embeds the WordPiece vocab (tokens, scores, types),
   so one decision solves the loader and the tokenizer's data problem
   together. The local Ollama blob becomes the canonical input.
2. **WordPiece tokenizer** — the largest Section 5.8 gap; everything
   downstream consumes its ids.
3. **Post-norm forward loop** — the same primitives composed in BERT order
   (norm after each residual add), plus the token-type embedding add.
4. **Mean pooling + L2 normalisation** — after this, the first real vector
   exists.
5. **Oracle check before any serving.** Compare Colibri's vectors against the
   live Ollama on `:11434` for a fixed input set (the
   `tools/compare_laya.py` oracle pattern). Correctness gate first; the
   serving surface comes after the numbers agree.
6. **Only then:** family registration, `POST /v1/embeddings`, `coli serve`
   wiring, and the Section 5.7 router.

**Parity detail to resolve at step 5:** nomic v1.5 is trained around task
prefixes (`search_document: `, `search_query: `). Whether Ollama prepends any
prefix automatically is an open question the oracle check must answer
empirically — otherwise Colibri's vectors will be subtly wrong against the
reference for reasons that look like engine bugs.

**Gemma 4 E2B and the router stay parked** until the embedding capability
exists end-to-end, per the resume priorities (1–2 first, 3–4 later).

---

## 7. Implementation record: `c/nomic_check.c` and `POST /v1/embeddings` (2026-11-18, steps 1–6 in progress)

Steps 1–4 of Section 6 were built as one standalone checkpoint tool
(`make -C c nomic-check`) and **step 5's oracle check passed**: Colibri's
vectors match the live Ollama reference on every input tried.

### 7.1 The two facts the oracle resolved

1. **The MLP gate is SiLU, not GELU.** Section 5.8 inferred "gelu-gated" from
   the shape; the sweep (24 compositions scored by cosine against Ollama)
   says `cos = 1.000000` only with the SiLU gate (`silu(gate) ⊙ up`).
   nomic-bert v1.5 is SwiGLU. GELU-erf scored 0.9586, GELU-tanh 0.9585.
2. **The GGUF vocab is converted from HF WordPiece form.** Word-initial
   pieces carry a leading `▁` (U+2581: HF `the` → GGUF `▁the`), and `##`
   continuations are stored plain (HF `##the` → GGUF `the`). The tokenizer
   must look up `▁`+piece at word start and the plain piece only for
   continuations — a plain-piece fallback at word start silently mis-splits
   (it caught "Ollama": 0.61 cosine → 1.000000 after the fix).

Also confirmed empirically: Ollama applies **no task prefix** (raw input is
the closest match), answers bidirectional full attention with half-split
rotary at theta 1000, and post-norm block order (the pre-norm variant scores
0.18 — decisively wrong).

### 7.2 Oracle results

`NOMIC_NORM=post NOMIC_ACT=silu NOMIC_ROPE=1` (the resolved composition):

| Input | cosine vs Ollama | max per-dim diff |
|---|---|---|
| "Ollama" | 1.0000003 | 5.0e-05 |
| "would replace Ollama entirely" | 1.0000000 | 4.2e-05 |
| "Embedding support in coli.serve would replace Ollama entirely." | 0.9999994 | 9.2e-05 |
| "The quick brown fox!" | 1.0000002 | 4.5e-05 |
| "Colibri is a hierarchical inference engine" | 0.9999992 | 9.2e-05 |
| "hello world" | 0.9999995 | 1.0e-04 |

The ~1e-4 residual is F16 weight storage rounding (both sides read the same
F16 file; the accumulation order differs). Token ids for
"The quick brown fox!" are the canonical bert-base-uncased sequence
`[101, 1996, 4248, 2829, 4419, 999, 102]`.

### 7.3 What the tool is

`c/nomic_check.c` (~600 lines, standalone, built via `make -C c nomic-check`):

- **Loader:** `gguf.h` in-place reads of the Ollama blob; all 112 tensors
  converted to f32 and held in RAM (~550 MB, the laya.c convention).
- **Tokenizer:** WordPiece from the embedded vocab (30522 entries,
  insertion-sort + binary search), UTF-8 aware Unicode lowercase plus
  dependency-free full-scalar-range canonical decomposition (NFD) and
  combining-mark removal. Non-Latin Unicode is preserved for lookup in the
  embedded vocabulary.
- **Forward:** post-norm BERT loop using laya.c's primitive shapes: fused
  QKV (no bias), per-sequence full attention, half-split rotary on every
  layer, biased layernorms, SiLU-gated MLP, no final norm.
- **Head:** mean pool over all tokens ([CLS]…[SEP] included — verified
  correct) + L2 normalisation.
- **Sweep knobs** (env, for future re-verification): `NOMIC_NORM`
  (post/pre), `NOMIC_ACT` (erf/tanh/silu), `NOMIC_ROPE` (off/half/interleaved),
  `NOMIC_POOL` (all/core), `NOMIC_TT` (token-type add on/off).
- Modes: `--dump` (metadata), `--tok "text"` (ids), `--embed "text"`
  (768-vector as JSON on stdout).

### 7.4 Serving surface

The standalone engine now also runs the mux protocol:

- `SERVE=1 ./c/nomic --model <GGUF>` announces `CAPS embed=1 chat=0`.
- `EMBED <id> <slot> <bytes>` carries a JSON array of input strings; the
  response is one JSON payload containing one 768-vector per input.
- `openai_server.py` dispatches that frame and exposes OpenAI-compatible
  `POST /v1/embeddings`, including batch inputs, usage token counts, model
  capabilities, and the normal `/v1/models` advertisement.
- `family_registry.py` registers `nomic_embed`; `coli serve --model <GGUF>`
  selects the `nomic` engine. Extensionless Ollama blob paths are accepted
  because Ollama stores canonical blobs under SHA-256 filenames.

#### Working local command

From the repository root, build the engine and start the API on port 8004:

```bash
make -C c nomic
python3 c/coli serve \
  --model "$HOME/.ollama/models/blobs/sha256-970aa74c0a90ef7482477cf803618e776e173c007bf957f635f1015bfcfef0e6" \
  --port 8004
```

Then request a vector:

```bash
curl http://127.0.0.1:8004/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"nomic-embed-text-v1.5","input":"The quick brown fox!"}'
```

End-to-end verification through `coli serve` passed: single and batch requests
returned 768-dimensional vectors, and the single-input vector matched Ollama
at cosine `0.99999994`. A real local run on port 8004 returned HTTP 200,
`capabilities=["embeddings"]`, a 768-dimensional vector, and
`prompt_tokens=7`. Clean engine shutdown and repeated mux requests were also
verified.

### 7.5 Remaining for the embedding capability

- The normalization gap is closed for the current Unicode database used to
  generate `c/nomic_unicode.h`: canonical decomposition, combining-mark
  removal, and lowercase mappings cover every Unicode scalar value. The
  generated table is maintained by `c/tools/gen_nomic_unicode.py`; the runtime
  has no ICU/libunistring dependency.
- BasicTokenizer punctuation splitting remains deliberately conservative
  (ASCII punctuation); expanding Unicode punctuation boundaries is separate
  tokenizer behavior, not normalization.
- Permanent API contract tests now cover single input, batch input, and invalid
  empty batches. A model-backed tokenizer/oracle fixture remains optional
  because the Ollama blob is not part of the repository.
- With normalization complete, the router is now the active work item.

### 7.6 Unicode normalization follow-up (2026-11-18)

The earlier Latin-only normalizer was replaced with generated Unicode data:

- `c/tools/gen_nomic_unicode.py` uses Python's Unicode database at maintenance
  time to emit canonical decomposition, `Mn` combining-mark ranges, and
  lowercase mappings.
- `c/nomic_unicode.h` is checked-in generated data; `nomic_check.c` performs
  binary-search lookups and recursive canonical decomposition at runtime.
- No ICU or other runtime Unicode library is required.
- Local model-backed checks produced identical IDs for NFC/NFD pairs:
  `Café`/`Cafe\u0301`, `Ångström`/`A\u030Angstro\u0308m`, and
  `İstanbul`/`I\u0307stanbul`.

The remaining tokenizer limitation is Unicode punctuation boundary handling;
the current split rule still recognizes ASCII punctuation only.

### 7.7 Static router baseline (2026-11-18)

Implemented and tested the first router slice:

- `c/router.py` provides a static JSON backend registry. Each backend declares
  its base URL, model IDs, capabilities, and optional backend-only API key.
- `coli router --config <file> --port <port>` starts the router. Public
  authentication is optional; configured backend keys are not exposed or
  forwarded from the client.
- `/v1/embeddings`, `/v1/chat/completions`, `/v1/completions`,
  `/v1/messages`, and `/v1/systemone` dispatch by endpoint capability and
  request `model`.
- `/v1/models` merges configured models; `/health` probes every backend and
  reports per-backend availability.
- JSON responses are buffered and SSE responses are streamed byte-for-byte.
  Unknown models return an OpenAI-shaped `404 model_not_found`.
- `c/tests/test_router.py` covers embedding dispatch, model/health reporting,
  authentication setup, and unknown-model refusal.

Still open for the router: dynamic cluster registration, backend lifecycle
management, CORS parity, and richer aggregate telemetry. Gemma 4 E2B remains
the next major model investigation.

### 7.8 Gemma 4 E2B compatibility investigation (2026-11-18)

The requested local path is not present in this checkout:
`/Users/creemac/.lmstudio/models/lmstudio-community/gemma-4-E2B-it-GGUF`.
The architecture was therefore checked against the published Gemma 4 E2B
configuration and llama.cpp's Gemma 4 loader/graph rather than by opening a
local GGUF.

Reference artifacts:
`https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/raw/main/config.json`
and
`https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/models/gemma4.cpp`.

**Verified model shape (`gemma-4-E2B-it`):**

- Text backbone: `gemma4_text`, 35 dense decoder layers, hidden size 1536,
  vocabulary 262,144, context 131,072, and 8 query heads / 1 KV head.
- Hybrid attention: 30 sliding-window layers with a 512-token window and
  five full-attention layers. Full-attention layers use proportional RoPE
  (theta 1,000,000); sliding layers use ordinary RoPE (theta 10,000).
- KV sharing: only the first 15 layers own KV; later layers reuse KV from an
  earlier compatible layer. This is distinct from ordinary GQA and requires
  explicit cache-source wiring.
- Dense MLP: GELU-tanh, 6144 intermediate width, double-wide MLP handling,
  RMS norms before and after attention/FFN sub-blocks, and final-logit
  soft-capping at 30.
- Per-layer embeddings (PLE): each of the 35 layers receives a dedicated
  256-dimensional token embedding, projected from the main 1536-dimensional
  stream. This is a major memory and forward-graph feature, not a metadata
  alias.
- The published model is multimodal (text, image, audio), although a first
  Colibri milestone can deliberately target text-only GGUF serving.

**Compatibility verdict:** **not loadable by an existing Colibri family.**
Current engines do not provide the combined PLE + hybrid sliding/full
attention + shared-KV graph, and `family_registry.py` has no `gemma4` entry.
Adding a descriptor alone would create a false-positive model selection and
must not be done.

**Required implementation order:**

1. Obtain a representative text-only E2B GGUF and record its complete GGUF
   metadata/tensor inventory, tokenizer, and chat template.
2. Add a standalone `gemma4` loader/CPU reference forward path first, covering
   PLE, Q/K RMS norms, layer-specific RoPE, sliding masks, shared KV sources,
   GELU-tanh double-wide MLP, and logit soft-capping.
3. Compare a short greedy generation against llama.cpp before adding Colibri
   mux serving.
4. Register the family and add `coli serve` only after parity passes.
5. Add image/audio support separately; text-only support is the first useful
   supporting-model milestone.

This investigation is now concrete, but implementation is blocked on the
representative GGUF artifact and remains separate from the main performance
roadmap.
