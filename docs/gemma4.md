# Gemma 4 integration

Gemma 4 support is being added as a distinct model graph. It deliberately does
not add Gemma branches to the GLM execution path: Colibri's placement and
streaming policies are reusable, while attention, routing, normalization,
tokenization, and output semantics are model-specific.

## Current milestone: native Gemma 4 vision tower

The first native integration milestone is implemented in:

- `c/model.h`: the placement-independent model/expert backend contract;
- `c/gemma4_backend.c`: validated loading of g4lab manifests and packed
  expert-major records, Q4_0 expert execution, learned per-expert scaling,
  weighted selected-expert aggregation, an optional exact-record LRU cache, a
  protected live-learning tier, restart-persistent usage profiles, and optional
  asynchronous cold-record staging driven through Colibri's placement-
  independent prepare/run/release contract;
- `c/gemma4_gguf.c`: a native GGUF v2/v3 metadata and tensor index for the
  tensor types used by the target model;
- `c/gemma4_model.c`: Gemma's router RMS normalization, complete-expert
  softmax, stable top-k selection, selected-probability renormalization, and
  routed-branch RMSNorm, plus reusable resident Q4_0 and Q6_K matrices and the
  Q/K/V and output projection boundaries, default/proportional RoPE,
  local/global KV storage, causal grouped-query attention, non-causal image-
  embedding chunks, the resident
  gated-GELU MLP, complete decoder-layer residual/norm ordering, scaled token
  embeddings, persistent 30-layer execution, final normalization, the tied
  vocabulary head, and final logit soft-capping;
- `c/gemma4_tokenizer.c`: the GGUF-native Gemma 4 tokenizer, including
  space-to-`▁` normalization, raw UTF-8 BPE over non-newline spans, ranked
  merges, newline handling, byte-token fallback, automatic BOS insertion,
  control-token partitioning, canonical initial/continuation user frames, and
  exact image-open/image-close frame partitioning;
- `c/gemma4_sampling.c`: deterministic greedy and seeded temperature/top-k/
  top-p selection over the native vocabulary logits;
- `c/gemma4_vision.c`: validated loading of the separate Gemma 4 vision GGUF,
  BF16 tensor indexing, reference-compatible dynamic image sizing, bilinear RGB
  resize, normalization, patch/output-token geometry, scaled 16x16 RGB patch
  convolution, learned independent X/Y position lookup, all 27 streamed BF16
  transformer blocks, per-head 2D NeoX RoPE, full non-causal attention,
  gated quick-GELU feed-forward layers, 3x3 average pooling, learned output
  calibration, and the final 1,152-to-2,816 projection;
- `c/gemma4.c`: a small command-line probe using that backend.

It consumes a directory created by the companion lab:

```powershell
& $g4lab pack $model --out ".\packed-gemma4" --layer 0
```

A one-layer pack is intentional for bring-up. Its manifest describes all routed
layers, while only `layer-00.g4ex` is present. The `info` command labels the
remaining entries as `direct GGUF fallback`: those experts are read from their
original fused gate/up and down tensor slices. The original GGUF must remain at
the manifest's recorded source path for fallback records and the small learned
scale vector.

Build it with Colibri's existing Makefile:

```powershell
cd c
make gemma4
```

Inspect the packed model boundary:

```powershell
.\gemma4.exe info C:\path\to\packed-gemma4
```

Inspect and validate the full text-model GGUF directly:

```powershell
.\gemma4.exe model-info C:\path\to\gemma4.gguf
```

This reads the GGUF directory without loading all weights and reports the real
layer width, vocabulary, MoE configuration, and local/global attention
schedule. On the 26B-A4B file, global layers are 5, 11, 17, 23, and 29; the
other layers use the 1,024-token sliding window.

Tokenize ordinary prompt text directly from the GGUF vocabulary and merge
tables:

```powershell
.\gemma4.exe tokenize C:\path\to\gemma4.gguf `
  "Hello, world!"
```

For exact UTF-8 input on Windows (including emoji or non-Latin scripts), use a
file so command-line code-page conversion cannot alter the bytes:

```powershell
.\gemma4.exe tokenize-file C:\path\to\gemma4.gguf `
  C:\path\to\prompt-utf8.txt
```

Both commands add the model's BOS token by default. Pass `--no-bos` when a raw
tokenization boundary is needed.

Build a canonical text-only user turn and inspect the exact tokens that will
feed generation:

```powershell
.\gemma4.exe chat-user C:\path\to\gemma4.gguf `
  "Explain why the sky is blue."
```

The UTF-8-safe form is:

```powershell
.\gemma4.exe chat-user-file C:\path\to\gemma4.gguf `
  C:\path\to\user-message-utf8.txt
```

This implements the text-only path from the target GGUF's embedded Google Gemma
4 canonical chat template: optional system turn, user turn, closed boundaries,
`<|turn>model`, and the empty thought-channel generation prefix. User and system
content are trimmed exactly as the template specifies. Interactive tool calls
and responses are supported.

Inspect the canonical image-first frame and the precise point where projected
vision vectors enter the decoder:

```powershell
.\gemma4.exe image-chat-user C:\path\to\gemma4.gguf `
  "What is shown in this image?"
```

The command prints prefix IDs ending in `<|image>`, an image-embedding slot, and
suffix IDs beginning with `<image|>`. The model API accepts `N x 2816` float32
projector outputs at that slot. It evaluates the complete image chunk with
non-causal self-attention in every decoder layer, stores its keys and values,
then resumes normal causal text evaluation. This is the real Gemma 4 decoder
boundary: image vectors are not token embeddings and must not be represented by
repeated placeholder IDs. The separate `gemma-4-26B-it-mmproj.gguf` vision tower
now produces these vectors natively; connecting decoded image files and those
vectors to interactive generation is the next integration boundary.

Inspect that projector and calculate the exact resize/token geometry for an
image without loading its 1.19 GB of tensor payloads:

```powershell
.\gemma4.exe vision-info `
  C:\path\to\gemma-4-26B-it-mmproj.gguf 1920 1080
```

The native vision API accepts interleaved RGB8 pixels. It aligns both dimensions
to `patch_size * pooling_size` (48 here), preserves aspect ratio, enforces the
40–280 projected-token range, bilinearly resizes with the same endpoint mapping
as llama.cpp, and returns normalized interleaved float32 pixels plus patch and
pooled-token geometry. Binary PPM remains dependency-free. Windows uses WIC for
PNG/JPEG; non-Windows Makefile builds enable the same codecs when their
`pkg-config` metadata is available.

Run image-conditioned generation directly from an image file. Binary PPM
(`P6`, 8-bit RGB) is portable and dependency-free. Windows builds decode PNG
and JPEG through WIC, and non-Windows builds use detected libpng/libjpeg:

```powershell
.\gemma4.exe generate C:\path\to\gemma4.gguf `
  C:\path\to\packed-gemma4 "Describe this image." --chat `
  --image C:\path\to\image.ppm `
  --mmproj C:\path\to\gemma-4-26B-it-mmproj.gguf --max-new 64
```

The same options work with `next-token` for deterministic logit inspection.
Both options must be supplied together and require `--chat`. Colibri builds the
canonical prefix through `<|image>`, evaluates the projector output as one
non-causal decoder chunk, then evaluates the suffix beginning with `<image|>`
before sampling. Image positions therefore occupy real KV positions without
inventing placeholder token IDs. Inspect decoding and prepared geometry without
running the transformer using `vision-image-info MMPROJ.gguf IMAGE`.

Run the first computed vision-graph boundary on a deterministic RGB image and
optionally save all patch vectors:

```powershell
.\gemma4.exe vision-patch-probe `
  C:\path\to\gemma-4-26B-it-mmproj.gguf 48 48 `
  --output-f32 C:\path\to\patch-embeddings.f32
```

The probe performs preprocessing, scales pixels to `2*x-1`, applies the real
F32 `[16,16,3,1152]` convolution kernel, and adds rows from the separate learned
X and Y position tables. The GGUF reader supports bounded tensor-slice reads, so
only the used rows of the 94 MB position tensor are read.

Run the complete native vision tower on that deterministic image and save the
decoder-ready image vectors:

```powershell
.\gemma4.exe vision-encode-probe `
  C:\path\to\gemma-4-26B-it-mmproj.gguf 48 48 `
  --output-f32 C:\path\to\projected-image.f32
```

For the 48x48 source, smart resize produces 21x21 patches, the 3x3 pool produces
49 image tokens, and the output file contains `49 * 2816` consecutive float32
values. Add `--layers N` to stop after N transformer blocks; this diagnostic
form writes the 441 width-1,152 patch vectors before pooling and projection.

Build and run the independent llama.cpp comparison with:

```powershell
powershell -ExecutionPolicy Bypass `
  -File tests\build_llama_gemma4_vision_oracle.ps1 `
  -LlamaRoot C:\path\to\llama.cpp

python tests\validate_gemma4_vision.py `
  .\gemma4.exe `
  .\build-gemma4\llama-gemma4-vision-oracle.exe `
  C:\path\to\gemma4.gguf C:\path\to\packed-gemma4 `
  C:\path\to\gemma-4-26B-it-mmproj.gguf `
  --llama-bin C:\path\to\llama.cpp\build\bin\Release
```

Add a system instruction to the tokenizer probe with `--system`, or use
`--system-file` to preserve arbitrary UTF-8 exactly on Windows:

```powershell
.\gemma4.exe chat-user C:\path\to\gemma4.gguf `
  "Hello, Gemma!" --system "Answer briefly."
```

Declare OpenAI-compatible function tools with `--tools-file`. The file must be
a JSON array of `{"type":"function","function":...}` declarations; the
repository includes a complete small example at
`tests/fixtures/gemma4_weather_tools.json`. Colibri validates the schema and
renders Gemma 4's compact declaration syntax, including sorted properties,
descriptions, string enums, arrays, nested objects, required fields, nullable
fields, and optional response schemas:

```powershell
.\gemma4.exe chat-user C:\path\to\gemma4.gguf `
  "Weather in Rome?" --system "Be exact." `
  --tools-file tests\fixtures\gemma4_weather_tools.json
```

Run a manual tool-call probe with the full model using:

```powershell
.\gemma4.exe generate C:\path\to\gemma4.gguf `
  C:\path\to\packed-gemma4 "What is the weather in Rome? Use the tool." `
  --chat --tools-file tests\fixtures\gemma4_weather_tools.json `
  --show-special --max-new 32
```

To complete the tool round trip, use the persistent `chat` command instead:

```powershell
.\gemma4.exe chat C:\path\to\gemma4.gguf `
  C:\path\to\packed-gemma4 `
  --tools-file tests\fixtures\gemma4_weather_tools.json `
  --show-special --max-context 512 --max-new 64
```

When Gemma requests a function, the session prints its name and compact Gemma
arguments, then prompts with `result JSON>`. Enter one complete JSON value, for
example `{"city":"Rome","temp_c":31,"conditions":"sunny"}`. Colibri
validates it, renders the canonical `<|tool_response>` block, evaluates that
block into the existing KV cache, and resumes the same model turn. Multiple
calls in one handoff are prompted and injected in order. Enter `/exit` at a
result prompt to end the session without injecting a response.

Run the complete text model and inspect its deterministic next-token ranking:

```powershell
.\gemma4.exe next-token C:\path\to\gemma4.gguf `
  C:\path\to\packed-gemma4 "Hello" --top 10
```

Add `--chat` to apply the canonical single-user chat frame before evaluation.
When `--chat` is active, `--system TEXT` or `--system-file FILE` prepends the
canonical system turn. Supplying a system prompt without `--chat` is rejected.
`--tools-file FILE` likewise requires `--chat`; tool declarations force the
canonical system turn even when no system instruction is present.
The command keeps all 30 attention/dense layers resident, maintains one causal
KV cache per layer, asks the Colibri expert backend for each routed top-8, then
applies the final RMS norm, tied Q6_K vocabulary head, and logit soft-cap. Use
`--residual-f32 FILE` or `--logits-f32 FILE` to save exact validation artifacts.
For UTF-8 text on Windows, use `next-token-file` with the same arguments and a
path to the user-message file in place of the command-line text.

Generate text with the same persistent model and KV state:

```powershell
.\gemma4.exe generate C:\path\to\gemma4.gguf `
  C:\path\to\packed-gemma4 "Say hello in one short sentence." `
  --chat --system "Answer in a friendly, concise style." --max-new 64
```

`generate-file` is the UTF-8-safe equivalent. Generation remains greedy by
default. A positive temperature enables seeded sampling, using the GGUF's
declared top-k/top-p defaults unless explicitly overridden:

```powershell
.\gemma4.exe generate C:\path\to\gemma4.gguf `
  C:\path\to\packed-gemma4 "Write a cheerful greeting." --chat `
  --temperature 0.8 --top-k 40 --top-p 0.9 --seed 123 --max-new 64
```

Each step decodes Gemma's SentencePiece-style token bytes, feeds the selected
token back through the 30 persistent layers, and stops on EOS or `<turn|>`.
Control/channel tokens are hidden by default; pass `--show-special` to inspect
the raw chat protocol while debugging. With tools enabled, this exposes
`<|tool_call>...<tool_call|><|tool_response>` so a manual caller can inspect the
requested function and arguments. The non-interactive `generate` command stops
at the tool-response handoff token. Interactive `chat` instead requests JSON,
injects the canonical response into the active cache, and resumes generation.
The same seed and parameters produce the same token sequence.

Start an interactive multi-turn session without reloading model weights or KV
state between turns:

```powershell
.\gemma4.exe chat C:\path\to\gemma4.gguf `
  C:\path\to\packed-gemma4 --max-context 512 --max-new 64 `
  --system-file C:\path\to\system-prompt-utf8.txt `
  --temperature 0.8 --seed 123
```

Enter `/exit` to quit. The optional system turn is evaluated once, immediately
before the first user turn. After every response, the selected `<turn|>` is
evaluated into all layer caches; subsequent messages append the canonical
continuation user frame at the next position. If a response reaches `--max-new`,
the runtime inserts and evaluates `<turn|>` before accepting another user turn
so session state never contains an unterminated model message.

Expert records stream directly from the packed layer or source GGUF by default.
For repeated inference in the same process, `generate`, `next-token`, and `chat`
accept `--expert-cache N`, where `N` is a global number of encoded expert-record
slots. The records remain quantized and are decoded only when selected, so this
does not change routing or floating-point results. For this 26B model each slot
is 3,346,432 bytes: 256 slots can grow to about 817 MiB, in addition to resident
model state. Cache allocation is lazy, and a hit/miss/eviction summary is
printed when the command exits. For example:

```powershell
.\gemma4.exe chat C:\path\to\gemma4.gguf `
  C:\path\to\packed-gemma4 --expert-cache 256
```

On the local four-step raw-generation probe, cached and uncached greedy output
was identical. A 256-slot run recorded 240 hits and 720 misses, loaded 2,297.1
MiB of records, and took 6.09 seconds versus 6.52 seconds without the userspace
cache. Windows' filesystem cache already absorbs much of this small warm-run I/O;
the explicit cache primarily establishes the Colibri placement boundary and
exposes deterministic reuse telemetry for the next learned-pinning/prefetch
phase.

Longer sessions can reserve part of that cache as a learned hot tier with
`--expert-pins P`, where `P` must be smaller than `N`. Gemma reuses Colibri's
LFRU scoring rule: frequency is primary, recency breaks close scores, and a
streamed expert must exceed the coldest protected record by 25% plus four
accesses before promotion. The swap exchanges resident encoded records and does
not reread either expert. Live heat resets when the process exits so an old
workload cannot prevent adaptation. Cumulative selection counts can persist
separately with `--expert-usage FILE`. The file uses Colibri's compatible
`layer expert count` format, is replaced atomically, and seeds protected slots
with the globally hottest records before inference. Interactive chat saves it
after every completed turn as well as at clean exit.

```powershell
.\gemma4.exe chat C:\path\to\gemma4.gguf `
  C:\path\to\packed-gemma4 --expert-cache 256 --expert-pins 64 `
  --expert-usage C:\path\to\packed-gemma4\.gemma4_usage
```

Protected slots are intended for longer, repetitive sessions. Leave
`--expert-pins` at its default of zero for short one-shot prompts, where there
is not enough history to overcome the admission hysteresis. A real-model smoke
test with 256 total and 64 protected slots produced the same greedy text as the
uncached and plain-LRU runs; the synthetic suite deterministically exercises an
actual hot-record promotion and verifies its numerical output.

The real restart probe evaluated the same two-token prompt twice. Its first run
saved 480 routed selections; the second loaded those 480, saved 960 cumulative
selections, and produced the same greedy token. Warm seeding increased cache
hits from 14 to 128 and reduced demand misses from 466 to 352. The 64 deliberate
startup record loads are reported separately as `preloads`, not as misses.

`--expert-prefetch` changes `prepare_layer` from a blocking load into a
background staging request. The decoder launches it immediately after routing,
computes Gemma's independent resident dense MLP on the caller thread, and joins
the worker before routed-expert math. Cache hits stay synchronous and do not
launch a worker. The option requires `--expert-cache`; it changes scheduling
only, not selected experts, weights, or arithmetic.

```powershell
.\gemma4.exe generate C:\path\to\gemma4.gguf `
  C:\path\to\packed-gemma4 "Hi" --max-new 3 `
  --expert-cache 256 --expert-prefetch
```

In the paired real-model four-step probe, synchronous staging took 7.55 seconds
and asynchronous staging took 6.29 seconds, a 16.6% reduction. Both runs
generated identical text and performed the same 720 demand loads; telemetry
reported `prefetch=120/720` (worker launches/records) for the asynchronous run.

`--expert-lookahead` extends that worker across layer boundaries. After layer
N finishes, the next router runs on the current residual as a prediction and
stages its missing records while layer N+1 computes attention. The authoritative
router still runs after attention and alone decides expert IDs and weights, so
lookahead changes placement only. It requires both `--expert-cache` and
`--expert-prefetch`:

```powershell
.\gemma4.exe next-token MODEL.gguf PACKED_DIR "Hello" `
  --expert-cache 256 --expert-prefetch --expert-lookahead
```

On the local two-position real-model probe, prediction covered 363/464 selected
experts (78.2%) and preserved identical top-three IDs and logits. Two paired
orders averaged 4.80 seconds without lookahead and 4.18 seconds with it (12.9%
lower wall time); individual pair wins ranged from 3.1% to 20.3%, showing why
the policy remains opt-in and measurable.

The source tree includes a validated persistent-buffer CUDA Q4_0 expert kernel,
but the focused Makefile integration in this PR builds the dependency-free CPU
backend. Wiring the optional CUDA source into Colibri's existing platform CUDA
build conventions is follow-up work. During development, the focused benchmark
kept one encoded record resident in the host cache and reused CUDA allocations.

On the local 4 GB Quadro P1000, CUDA averaged 1.45 ms/expert versus 1.60 ms on
CPU (9.4% faster). Agreement was strong: maximum absolute error `4.66e-10`, RMS
`1.25e-10`. First-call process startup and device initialization remain slower,
so the GPU path is useful for persistent inference rather than one-shot probes.

This first integration deliberately exposes Gemma through the standalone
`gemma4` executable. Wiring it into Colibri's evolving OpenAI-compatible
gateway and WebUI is follow-up work so that the engine implementation can be
reviewed independently from gateway scheduling and streaming-tool changes.

The input/output boundaries can also be probed independently:

```powershell
.\gemma4.exe embed C:\path\to\gemma4.gguf 2 `
  --output-f32 bos-embedding.f32
.\gemma4.exe lm-head C:\path\to\gemma4.gguf `
  --input-f32 final-residual.f32 --top 10
```

Run only the exact router boundary for a saved residual state:

```powershell
.\gemma4.exe route C:\path\to\gemma4.gguf `
  --layer 0 --input-f32 C:\path\to\layer-residual.f32 `
  --probabilities-f32 C:\path\to\router-probabilities.f32
```

`route` applies the router's scale-only RMS normalization, its learned input
scale, the `1/sqrt(hidden_size)` factor, softmax over all 128 experts, stable
top-8 selection, and top-8 renormalization. It prints both normalized weights
and effective weights after the learned expert scale.

Replay a real hidden state through one Colibri-native expert:

```powershell
.\gemma4.exe expert C:\path\to\packed-gemma4 `
  --layer 0 --expert 0 `
  --input-f32 C:\path\to\hidden-state.f32 `
  --output-f32 C:\path\to\colibri-expert-output.f32
```

The output can be compared byte-for-byte with g4lab using the same input. This
proves that the native Colibri backend preserves the packed record layout,
Q4_0 decoding, gated GELU MLP, learned expert scale, and selected-expert
aggregation.

Connect the router to the streamed expert backend:

```powershell
.\gemma4.exe routed-mlp C:\path\to\gemma4.gguf `
  C:\path\to\packed-gemma4 --layer 0 `
  --input-f32 C:\path\to\layer-residual.f32 `
  --output-f32 C:\path\to\routed-branch-output.f32
```

This command routes from the residual input, separately applies
`pre_ffw_norm_2`, executes the selected packed or direct-GGUF experts, applies
each learned expert scale exactly once, and writes the weighted routed-branch
aggregate. Passing the GGUF explicitly also makes older manifests with
working-directory-relative source paths portable.

Inspect the normalized attention projections before RoPE:

```powershell
.\gemma4.exe attention-proj C:\path\to\gemma4.gguf `
  --layer 0 --position 123 --input-f32 C:\path\to\layer-residual.f32 `
  --query-f32 C:\path\to\query.f32 `
  --key-f32 C:\path\to\key.f32 `
  --value-f32 C:\path\to\value.f32
```

The layer handle keeps the encoded Q4_0 projection matrices resident and
reusable. It applies `attn_norm` to the residual, runs Q/K/V projections,
applies Q/K RMSNorm independently to each head, and rotates Q/K for the given
position. Value heads use scale-free RMSNorm. Sliding layers use 16 query heads
and 8 KV heads at dimension 256 with default base-10,000 RoPE. Global layers
use 16 query heads and 2 KV heads at dimension 512, load proportional frequency
factors from `rope_freqs.weight`, and derive values from the unnormalized key
projection, matching the model's missing global `attn_v.weight` tensors.

The model API also provides KV storage with distinct policies: sliding layers
use a bounded ring capped by the model's 1,024-token window, while global layers
retain append-only positions up to the caller's context allocation. Cache slots
carry their absolute position, so a wrapped sliding slot cannot be mistaken for
the older token it replaced.

Run a sequence through the complete attention branch:

```powershell
.\gemma4.exe attention-seq C:\path\to\gemma4.gguf `
  --layer 0 --tokens 2 --input-f32 C:\path\to\two-residuals.f32 `
  --output-f32 C:\path\to\attention-output.f32
```

The input and output contain `tokens * model_width` consecutive float32
values. For each token, this projects and normalizes Q/K/V, applies the layer's
RoPE variant, stores K/V under the correct cache policy, performs causal
grouped-query softmax over retained positions, and applies the resident Q4_0
output projection. Gemma 4's normalized Q/K attention uses a score scale of
`1.0`; no additional `1/sqrt(head_dim)` factor is applied.

The focused real-model NumPy oracle can be rerun from `c`:

```powershell
python tests\validate_gemma4_attention.py `
  .\gemma4.exe C:\path\to\gemma4.gguf
```

It uses the native projection probe for the already-validated rotated Q/K/V
boundary, then independently evaluates grouped-query causal softmax, decodes
the Q4_0 output projection, and compares the final sequence output.

Probe the resident dense MLP by itself:

```powershell
.\gemma4.exe dense-mlp C:\path\to\gemma4.gguf `
  --layer 0 --input-f32 C:\path\to\normalized-residual.f32 `
  --output-f32 C:\path\to\dense-branch-output.f32
```

This keeps all three Q4_0 matrices resident and evaluates the model's
2816-to-2112-to-2816 gate/up/down path with tanh-approximate GELU gating.

Run a complete decoder layer, including both feed-forward branches:

```powershell
.\gemma4.exe layer-seq C:\path\to\gemma4.gguf `
  C:\path\to\packed-gemma4 --layer 0 --tokens 2 `
  --input-f32 C:\path\to\two-residuals.f32 `
  --output-f32 C:\path\to\layer-output.f32
```

After attention, this applies `post_attention_norm` and the first residual add.
The dense branch consumes `ffn_norm`; the routed branch routes from that same
residual while experts consume `pre_ffw_norm_2`. Their outputs receive
`post_ffw_norm_1` and `post_ffw_norm_2`, respectively, before addition. The
combined branch receives `post_ffw_norm`, is added to the residual, and is
multiplied by `layer_output_scale`. Expert access still goes through the
placement-independent backend, so packed records and direct-GGUF fallback have
identical graph semantics.

The complete-layer boundary oracle can be rerun from `c`:

```powershell
python tests\validate_gemma4_layer.py `
  .\gemma4.exe C:\path\to\gemma4.gguf `
  C:\path\to\packed-gemma4
```

An optional whole-stack comparison against a current llama.cpp build is also
available. Build llama.cpp with Visual Studio first, then build the small tensor
capture helper:

```powershell
powershell -ExecutionPolicy Bypass -File tests\build_llama_gemma4_oracle.ps1 `
  -LlamaRoot C:\path\to\llama.cpp
```

Run the layer-0 comparison with the llama.cpp DLL directory on its search path:

```powershell
python tests\validate_gemma4_llama_layer0.py `
  .\gemma4.exe `
  .\build-gemma4\llama-gemma4-oracle.exe `
  C:\path\to\gemma4.gguf C:\path\to\packed-gemma4 `
  --llama-bin C:\path\to\llama.cpp\build\bin\Release
```

The helper uses llama.cpp's public evaluation callback and does not modify the
llama.cpp checkout. The comparison covers rotated Q/K, scale-free normalized V,
post-normalized attention, the dense-branch input, all router probabilities,
and the complete first decoder-layer output.

Run the full raw-prompt comparison, or add `--chat-file` for a canonical UTF-8
user turn:

```powershell
python tests\validate_gemma4_full.py `
  .\gemma4.exe `
  .\build-gemma4\llama-gemma4-oracle.exe `
  C:\path\to\gemma4.gguf C:\path\to\packed-gemma4 `
  --llama-bin C:\path\to\llama.cpp\build\bin\Release `
  --chat-file C:\path\to\user-message-utf8.txt
```

## Real-model validation

The native index opens the 14,439,363,584-byte QAT Q4_0 model and validates 658
tensors, 30 layers, width 2,816, vocabulary 262,144, 128 experts, and top-8
routing. On the saved layer-0 input from the companion lab:

- the C router and an independent NumPy implementation selected the same eight
  experts in the same order;
- the maximum probability difference was `4.28e-8`;
- the routed aggregate was compared with eight independent `g4lab expert-cpu`
  replays, with maximum absolute error `1.79e-6` and RMS error `3.41e-7`.
- sliding layer 0 and global layer 5 attention projections were checked against
  independent NumPy Q4_0 dequantization. The worst Q/K/V maximum absolute error
  was `3.81e-6`; every projected value was finite.
- default and proportional RoPE at position 123 were checked on those same
  layers. The worst rotated-Q error was `2.86e-6` and the worst rotated-K error
  was `2.98e-7`; the synthetic suite also covers sliding-cache wraparound.
- two-token causal attention plus output projection was checked independently
  on sliding layer 0 and global layer 5. Maximum absolute errors were
  `3.81e-6` and `4.29e-6`, respectively, with RMS error below `3.7e-7`.
- the resident gated-GELU MLP was independently decoded from Q4_0 on layers 0
  and 5, with maximum absolute errors `2.38e-6` and `4.29e-6`. A focused
  two-token oracle then recomposed every decoder norm, branch add, residual,
  routed expert call, and layer scalar; its outputs matched `layer-seq`
  exactly on both layers.
- a current llama.cpp graph independently evaluated the same two-token layer-0
  input. Colibri selected the same router top-8, router probabilities differed
  by at most `1.14e-6`, and the complete layer output reached cosine similarity
  above `0.99994` with RMS error below `0.015`.
- the native tokenizer matched current llama.cpp token-for-token on ASCII,
  repeated spaces/newlines, emoji, accented Latin text, an em dash, CJK text,
  and multi-digit input; the synthetic suite also exercises byte fallback.
- the canonical single-user chat frame, including its control-token boundaries
  and UTF-8 content, matched llama.cpp's special-token parse exactly (18/18
  tokens on the saved real-model probe).
- the canonical system-plus-user frame was derived from the target GGUF's
  embedded 2026-07-09 Google template and matched llama.cpp token-for-token
  (25/25 IDs), including whitespace trimming and the empty thought channel.
- the canonical image-first frame was partitioned exactly around `<|image>` and
  `<image|>`. Recombining its five prefix and seventeen suffix IDs matched
  current llama.cpp's special-token parse exactly (22/22 IDs) on the target
  GGUF.
- a synthetic two-token image chunk independently recomputed Q/K/V, RoPE,
  two-key softmax, and output projection. The first image position attended the
  later image position as required, matching the native non-causal path within
  `1e-5`; the same decoder then returns to causal evaluation.
- the native projector index opened the real 1,194,828,160-byte GGUF v3 and
  validated 356 tensors, 27 blocks, width 1,152, feed-forward width 4,304,
  16 heads, patch size 16, and decoder projection width 2,816. Its core F32/BF16
  tensor shapes match the Gemma 4 vision graph.
- reference-compatible smart resize maps a 1920x1080 source to 1056x576 and
  264 projected tokens (22x12). Synthetic tests cover ordinary/max-area sizing,
  bilinear RGB endpoint values, normalization, and 16-pixel patch/3x3 pool
  geometry.
- the real patch/position probe mapped a deterministic 48x48 source to 441
  width-1,152 vectors in about 91 ms. An independent NumPy convolution and X/Y
  lookup reached cosine similarity `0.999999642` and RMS error `2.79e-4`; five
  scalar-order spot checks differed by at most `1.57e-4`.
- the complete native vision tower evaluated those 441 patches through all 27
  blocks, pooled them to 49 tokens, calibrated them, and projected each token to
  width 2,816 in about 46 seconds on the local Release build. Against current
  llama.cpp, layer 0 reached cosine similarity `0.999999807` with RMS error
  `0.00137`; the final 137,984 projected values reached cosine `0.999975707`
  with RMS error `0.00800` and maximum absolute error `0.1301`.
- the restored matching llama.cpp runtime also evaluated the complete
  image-conditioned decoder. End-to-end vocabulary logits reached cosine
  `0.999673543`, RMS `0.18775`, and maximum error `0.92984`; both paths produced
  the same top-10 candidate set, although projector drift inverted the nearly
  tied top two (`27.9531` versus `27.9500` natively). Feeding llama.cpp's saved
  projected vectors into the native decoder reproduced llama.cpp's exact
  top-10 order, isolating that inversion to accumulated vision-projector
  arithmetic rather than image insertion or decoder/KV semantics.
- the oracle now captures transformer-depth checkpoints in the same run. The
  native-vs-llama.cpp RMS/cosine curve is layer 0 `0.00137/0.999999807`, layer
  8 `0.00484/0.999999974`, layer 17 `0.05046/0.999999499`, and layer 26
  `0.69951/0.999957335`. The late, nonlinear growth localizes the remaining
  ordering discrepancy to accumulated tower arithmetic (especially reduction
  order in repeated BF16 matmuls and attention), while projection compresses
  the final hidden-state difference back to RMS `0.00800`.
- matching ggml's BF16 activation conversion before projector matrix products
  restored the previously inverted top-two order: both runtimes now select
  token `236776` first. The oracle enforces top-1 equality in addition to the
  existing top-10 set and decoder-ranking checks. Conversion is hoisted once
  per activation tensor, keeping the full validation at about 410 seconds
  instead of the naive inner-product implementation's 676 seconds.
- a live 26-token system-plus-user prompt completed the full 30-layer path and
  greedily produced `Hi`, confirming the frame is accepted by generation as
  well as the standalone tokenizer.
- a real weather-function declaration produced a 93-token prompt that matched
  the GGUF Jinja template plus llama.cpp tokenizer exactly (93/93 IDs). The
  native synthetic suite independently covers schema validation, property
  sorting, enums, and empty tool lists.
- a live 97-token tool prompt made the Q4 model emit
  `<|tool_call>call:get_weather{city:<|"|>Rome<|"|>}<tool_call|>` and stop on
  `<|tool_response>`, confirming the declaration-to-call handoff end to end.
- a live interactive tool round trip accepted
  `{"city":"Rome","temp_c":31,"conditions":"sunny"}`, evaluated the
  canonical response block in the same KV cache, resumed the model turn, and
  produced `The weather in Rome is currently sunny with a temperature of
  31°C.` before `<turn|>`.
- native Q6_K decoding reproduced llama.cpp's scaled BOS embedding bit-exactly.
  Given llama.cpp's saved final residual, the final norm differed by at most
  `9.54e-6`, vocabulary logits reached cosine similarity `0.999962`, and both
  implementations selected token `3324` as top-1.
- the complete native 30-layer pass over `BOS + Hello` selected that same token
  `3324`. Its final residual reached cosine similarity `0.999837` and its logits
  reached `0.999280`; the first nine ranked candidate IDs were identical and in
  the same order. The local Release run completed in about five seconds.
- an 18-token canonical UTF-8 chat prompt was then evaluated end to end by both
  runtimes. The final residual and logits reached cosine similarities `0.999904`
  and `0.999905`, respectively, and all top-10 candidate IDs matched in exactly
  the same order, led by token `236776`.
- a live greedy chat smoke test for “Say hello in one short sentence.” produced
  `Hello there!` and terminated correctly on `<turn|>` after four generated
  tokens.
- paired temperature `0.8`, top-k `20`, top-p `0.9` runs with seed `123` were
  byte-for-byte reproducible and produced `Hello! How can I help you today` in
  the bounded eight-token probe.
- a persistent two-turn test first established “my name is Alice” and received
  `OK<turn|>`, then asked “What is my name?” in the same process and received
  `Your name is Alice.<turn|>`, confirming retained context and turn-boundary KV
  alignment.

- cached and uncached real-model generation produced identical greedy text; a
  256-record cache reported 240 hits over the four evaluated token positions,
  while the synthetic suite verifies exact output through both cache hits and
  forced single-slot LRU evictions.
- the protected-tier suite raises a streamed expert above Colibri's LFRU
  hysteresis, swaps it into the hot tier without a disk reload, records later
  protected hits, and preserves the exact weighted expert output.
- a saved usage profile atomically round-trips cumulative counts and seeds the
  hottest protected record in a fresh backend; the real two-process smoke test
  doubled its cumulative selection count and improved warm-start cache hits
  while preserving generated output.
- asynchronous staging overlapped those same real-model record reads with the
  resident dense branch, reduced paired wall time by 16.6%, and produced the
  same greedy text; the synthetic suite verifies cold-worker and all-hit paths.

## Deliberate boundary

This milestone is a usable persistent CPU text-chat generator with canonical
system instructions, complete interactive tool declaration/call/response
continuation, the correct multimodal decoder insertion/attention boundary, and
native projector inspection/image-to-patch preprocessing and patch/position
embedding, the complete native vision transformer and decoder-width projection,
plus bounded LRU,
live/persistent learned placement, and lossless I/O overlap.
The follow-up integration stages are now implemented: the matching llama.cpp
runtime and image-conditioned final-logit oracle, portable libpng/libjpeg codec
paths, persistent per-slot Gemma KV reuse, constrained grammar sampling, and
OpenAI/Anthropic tool-call serving. The streaming parser recognizes Gemma's
native call sentinels across arbitrary engine-chunk boundaries, emits each
complete structured call, supports multiple calls, and never exposes protocol
markers as assistant text. Cooperative prefill/decode cancellation and opt-in
parallel execution through isolated Gemma workers and aggregate per-worker
expert/profile telemetry are also implemented and validated. The explicit
MSVC/AVX2 compatibility path is now bit-identical to llama.cpp through the
projected image vectors; final top-1 and top-10 behavior also matches.

The saved oracle was also used to test individual compatibility changes before
the `0.29.0` release. None was safe as a standalone production change. The
operation trace subsequently identified the compatible combination and it is
available as the explicit `llama-avx2` isolation path. This path combines
ggml-compatible patch vec-dot, RMS accumulation, RoPE frequency iteration,
BF16 tinyBLAS/FMA, tiled attention, and FP16 quick-GELU semantics while leaving
default portable arithmetic unchanged.

The probe now supports exact boundary replay and per-operation traces. This
example evaluates only block 17 from a saved llama.cpp layer-16 boundary and
writes each native intermediate for `compare_gemma4_vision_trace.py`:

```powershell
$env:COLI_GEMMA4_VISION_COMPAT = "llama-avx2"
.\gemma4.exe vision-encode-probe MMPROJ.gguf 48 48 `
  --layers 18 --start-layer 17 --input-f32 llama-vision-layer17.f32 `
  --trace-layer 17 --trace-dir trace --output-f32 native-layer18.f32
python tests\compare_gemma4_vision_trace.py trace --layer 17
```

Pass `--prepared-f32 PATH` to also write the planar, scaled image tensor at the
convolution boundary. The validator requires this tensor to be bit-identical
to llama.cpp. Matching llama.cpp's ratio-first bilinear coordinate arithmetic
removed 361 one-byte resize discrepancies on the canonical 48x48 input.

The full validator runs this exact-input trace automatically. With default
arithmetic, block-17 RMS is `0.00592096`. The completed `llama-avx2` path is
bit-identical at every captured operation. Two final association fixes close
the previous residuals: flash attention adds each double tile sum to the
current float accumulator before rounding, and patch convolution uses ggml's
four-register vec-dot plus two `_mm_hadd_ps` reductions. Layers 0, 8, 17, and
26 and all `137,984` projected values now have RMS and maximum error `0`.
The full tower takes about `59.3 s`, within 15% of the portable path.
The final canonical run measured layer-17 RMS `0.05016`, layer-26 RMS `0.71726`,
projection RMS `0.008595`, and vocabulary-logit RMS `0.21298`; top-1 and the
top-10 candidate set matched llama.cpp.

With portable production arithmetic, the exact resize correction measures layer-17
RMS `0.050049`, layer-26 RMS `0.715061`, projection RMS `0.008437`, and logit
RMS `0.215213`; top-1 and the top-10 candidate set still match. With
`llama-avx2`, image-conditioned logit RMS is `0.214735`, exactly equal to the
native decoder's result when fed llama.cpp image vectors; this isolates the
remaining logit difference to decoder arithmetic rather than vision.

Keeping those stages separate makes graph errors distinguishable from storage,
quantization, and cache-placement errors.
