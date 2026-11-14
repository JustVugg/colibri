# Inkling (Thinking Machines 975B MoE) on colibri

`c/inkling.c` runs [Thinking Machines Inkling](https://huggingface.co/thinkingmachines/Inkling)
(975B total / 41B active, Apache 2.0) with colibri's expert-streaming approach:
dense weights resident (RAM or VRAM), routed experts streamed from disk with an
LRU + pinned cache. Audio input is supported when the audio tensors are present
(see [Audio input](#audio-input-dmel) below); the vision encoder and the MTP
head are not loaded. The same engine runs
[Inkling-2-Small](#inkling-2-small) (594B total / 17B active), from its GGUFs or
from the official checkpoint converted to int4 g64.

## Quickstart

Pre-converted weights (int4 experts + bf16 residents, 514 GB as Hugging Face lists the repository):

```sh
hf download nbeerbower/Inkling-colibri-int4 --local-dir ~/Models/inkling_i4
```

or convert the original bf16 checkpoint yourself (shard-resumable; `--watch`
converts while the download is still running):

```sh
python3 c/tools/convert_inkling_int4.py --indir <bf16-checkpoint> --outdir ~/Models/inkling_i4
```

Build and run:

```sh
make -C c inkling                # pure CPU (dependency-free, like glm)
make -C c inkling CUDA=1         # + bf16 residents in VRAM (needs ~37 GB free)

SNAP=~/Models/inkling_i4 ./c/inkling -p "The capital of France is" -n 64
```

Requirements: ~120 GB RAM (CPU build keeps ~86 GB of bf16 residents in RAM;
the CUDA build moves them to VRAM and uses the freed RAM for a larger expert
cache), NVMe storage for the snapshot.

## If you have less RAM than that

The pre-converted container has **int4 routed experts but bf16 dense weights**, and
the dense set is resident: every token needs all of it. Measured on the real
snapshot it is **49.4 GB**, and `load_w` expands bf16 to f32 while loading, so the
peak is ~99 GB. Below roughly 64 GB of RAM the process dies before it generates
anything.

Quantizing just the dense set to **int4-gs64** brings it to **15.3 GB**, which fits a
25 GB box. One pass over the shards, originals untouched, ~14 minutes:

```sh
python3 c/tools/convert_inkling_dense_int4.py --dir ~/Models/inkling_i4 --plan   # estimate first
python3 c/tools/convert_inkling_dense_int4.py --dir ~/Models/inkling_i4

mkdir -p ~/Models/inkling_i4/dense-int4g64
mv ~/Models/inkling_i4/dense-int4g64.safetensors \
   ~/Models/inkling_i4/dense-int4g64/dense.safetensors
```

The engine picks the container up automatically (it prints `[dense] container
int4-gs64: …`); `INK_DENSE_Q4=0` ignores it and goes back to bf16. **With enough RAM
you do not need any of this** — leave the container out and nothing changes.

Quantization error, measured against the real bf16 weights:

| Tensors | Format | Rel. L2 error |
|---|---|---|
| attention, shared experts, dense MLP | int4-gs64 | ~11% |
| `embed_tokens`, `lm_head` | int8 per-row | ~0.9% |
| norms, biases, router, conv1d | untouched | 0 |

11% is what 4 bits costs on this weight distribution (quantization noise ≈ 0.135σ for
16 levels over ±amax in groups of 64) — not a defect of the conversion. It is enough
for coherent output: the 975B answers correctly on a 25 GB host. `embed`/`lm_head` are
kept at int8 because they enter *every* token. `ATTN_BITS=8` moves attention to int8
too (error 11% → 1.1%, +4 GB) if you would rather spend the RAM.

**Speed, honestly:** with only ~8 GB left for the expert cache after the dense set,
residency is ~1.7% of the 464 GB expert bank, so decode is disk-bound — tens of
seconds per token on a single NVMe, not interactive. This path is what makes the model
*runnable and testable* on a small host, not fast. Pinning (`PIN`) and a larger cache
help in proportion to the RAM you can give them.

## Inkling-2-Small

[thinkingmachines/Inkling-2-Small](https://huggingface.co/thinkingmachines/Inkling-2-Small)
(594B total / 17B active, Apache 2.0) is a new architecture on the same engine: 48 layers of
384 routed experts (top 8, no shared expert, no dense layer), 40 sliding-window layers and 8
global ones, attention residuals over blocks of six layers, short convolutions after attention
and after the MLP. The engine reads it under the official tensor names, its vision tower
included ([pictures](#inkling-2-pictures) below); the MTP drafter is not loaded
([why](#inkling-2-the-mtp-drafter)).

Two ways to run it:

- **A GGUF.** unsloth's files ([unsloth/Inkling-2-Small-GGUF](https://huggingface.co/unsloth/Inkling-2-Small-GGUF))
  import like any other ([gguf.md](gguf.md)); point `coli import` at the first part:

  ```sh
  ./coli import ~/models/Inkling-2-Small-UD-Q2_K_XL-00001-of-00006.gguf   # every part next to it
  ```

- **The official checkpoint as int4 g64.** Thinking Machines publish the experts in NVFP4
  (layers 0-39) and MXFP8 (layers 40-47), 393.5 GB. `c/tools/convert_inkling2_int4.py` turns
  them into colibri's int4 with one f32 scale per 64 weights (the planar layout of
  `expert_ffn.h`) and copies every other tensor as it is, one `layer-NN.safetensors` per layer:

  ```sh
  hf download thinkingmachines/Inkling-2-Small --local-dir ~/Models/ink2-nvfp4
  python3 c/tools/convert_inkling2_int4.py --indir ~/Models/ink2-nvfp4 --outdir ~/Models/ink2-int4g64
  ```

  The container is 347.6 GB (the MTP drafter's 3.4 GB included). Without room for both, run
  the converter next to the download with `--watch --delete-src`: each layer is converted as soon as its shards are complete, and
  a shard is removed once every output that reads it is written. `--min-free-gb N` (default
  20) waits before a write would leave less than N GB free (without `--watch` it stops
  instead). A second run converts only what is missing. The MTP drafter goes to `mtp/`
  untouched.

  Relative L2 error against the source weights, measured on every layer of the real
  checkpoint: 9.5-9.7%. The scale of each group of 64 is the one of nine candidates around
  absmax/7 and absmax/8 with the least squared error; absmax/7 alone gives 11.3%.

Build and run as for Inkling:

```sh
make -C c inkling                # CPU
make -C c inkling CUDA=1         # + the expert tier and the dense matrices on a CUDA card
make -C c inkling VK=1           # + the same on Vulkan (run with COLI_VULKAN=1)

SNAP=~/Models/ink2-int4g64 ./c/inkling --chat -p "What is the capital of France?" -n 64
```

`--chat` and the gateway (`coli chat`, `coli serve`, `coli web`, model id
`inkling-2-small-colibri`) use Inkling-2's own template: the effort level goes in a system
message before the first message, and with thinking off the model's turn is opened for it as
text. Tools as the template declares them: the specs in a `tool_declare` system message before
everything else, a call a model message of its own under the tool's name
(`<|content_invoke_tool_json|>` and `{"name": ..., "args": {...}}`), a result a tool message
under the name of the call it answers; streamed or not, OpenAI's and Anthropic's APIs, and
`tool_choice` (`required` or a name adds the instruction the other families get). With tools
declared the model opens its turn itself even with thinking off: a turn opened as text never
called a tool, and it reasons a few words first (they come back as `reasoning_content`). The
engine's repetition penalty is off for Inkling-2 (`REP_PEN`): a call repeats the declaration's
tokens, and with the penalty it came out broken.

On a GPU, the CUDA build (`CUDA=1`) or the Vulkan one (`VK=1`, `COLI_VULKAN=1`):

- the routed experts go to the expert tier, CUDA's or Vulkan's
  ([vulkan.md](vulkan.md#the-routed-expert-tier-vk_tierc)): the history's hottest at start, then
  the ones the CPU computes as you chat, while the CPU computes the rest from its RAM cache.
  The int4 g64 container's experts go up as they are; a GGUF's as the file has them. A GGUF's
  NVFP4 layers keep each expert's global scale outside its blocks (one number for gate and up,
  as the checkpoint's w13 has one amax an expert); the engine folds it into the row the device
  reads and into the pair's weight.
- the bf16 dense matrices (the attention projections, its gate and the unembedding) go to the
  device. With Vulkan the layers run as a dense chain there (`COLI_VK_CHAIN`, on by default on
  a discrete GPU, [vulkan.md](vulkan.md#inkling-2-on-the-chain)): the attention, the residual
  mixes and the convolutions on the device, one frame a layer, the routed experts on the host;
  with CUDA, or `COLI_VK_CHAIN=0`, the matrices go a call at a time and the rest runs on the CPU.
- the RAM cache's pinned experts that the tier holds go back to its LRU after the warm start,
  so that RAM and VRAM hold different experts. The tier's host pool is off for Inkling-2
  (`COLI_VK_TIER_HOST=1` turns it on): the RAM it takes from the expert cache cost more than
  the device gained (numbers below).
- a build with both backends runs Inkling-2 on Vulkan, as it runs Inkling (`COLI_VULKAN=0`
  keeps CUDA): measured ahead below. Metal leaves Inkling-2 on the CPU.

Measured on an RTX PRO 4000 Blackwell (24 GB), an i5-13500 (the engine's six P-cores) and
62 GB of RAM, the container on two NVMe in RAID0: `--chat`, a prompt of 26-27 tokens and 64
generated, three prompts, the expert history of earlier runs:

| Build | Prompt | Decode | Decode, reads first (`INK_FILL_ASYNC=0`) |
|---|---|---|---|
| CPU | 16.1-16.6 s | 1.15-1.19 tok/s | 1.11-1.12 tok/s |
| CUDA (`CUDA=1`) | 14.8-15.6 s | 1.64-1.68 tok/s | 1.50-1.56 tok/s |
| Vulkan (`VK=1`, `COLI_VULKAN=1`) | 14.6-16.3 s | 1.89-1.92 tok/s | 1.43-1.81 tok/s |

A decode step reads the experts it misses on a thread of its own while the CPU computes the
ones it found in RAM, then computes the read ones (`INK_FILL_ASYNC`, on by default); before,
it read them all first. The two columns are the same session, each prompt run both ways one
after the other (and again in the other order); the fill the step waits for went from 29-38 s
of a run to 19-24 s. The prompt that gains most (1.43-1.44 -> 1.91-1.92 on Vulkan, both orders)
is the one whose experts come most from the disk. Each expert's output has a row of its own and
the CPU's sum keeps its order, so the bits do not change: the oracles match either way.

At this size the disk sets the pace: of the routed experts the card's 587 serve 24%, the RAM
cache 48%, and the other 28% come from the disk (17.9 MB an expert, 2.7-3.1 ms each with
`INK_FILL_THREADS` readers), about half of a decode step. unsloth's UD-Q2_K_XL GGUF (191 GB,
its experts about half the bytes) ran the CPU build at 1.40 tok/s here on 8 tokens, before the
GPU tiers and the reads above; it was not measured since. The host pool on
(`COLI_VK_TIER_HOST=1`) gave CUDA 1.27-1.29 tok/s against 1.60-1.61 off, and Vulkan 1.33-1.41
against 1.26-1.63, before the pieced reads.

The CUDA build's CPU computes its share of the experts more slowly than the Vulkan build's (an
`expert-mm` of 18.5-19.8 s against 13.9-14.4 s over the same experts, whichever CUDA sync policy
the backend takes); that is where its decode trails, and it is not found yet.

### Inkling-2: pictures

The container carries the vision tower (`model.visual.*`, 655 MB, the converter copies it), and
the engine loads it with the rest: a ViT of 23 layers, 1024 wide, 20-pixel patches merged 2 x 2,
so that each 40 x 40 tile of a picture is one token of the prompt (`c/inkling2_vision.h`). The
gateway takes pictures in user and tool messages, as the template does (`image_url` parts,
data URIs), and prepares them as Inkling-2's processor does (`c/tools/inkling2_image.py`, Pillow
and numpy): the picture keeps its size, padded to whole tiles, transparency on the processor's
checkerboard. A picture of more than 256 tiles (`INK2_MAX_IMAGE_TOKENS`) is scaled down until it
fits, since the tower runs on the CPU and the prompt pays for every tile. One picture a request.

A picture of 320 x 200 (5 x 8 tiles) went through the tower in 0.51 s on the machine below, and
the model described what it showed (two shapes, their colors and places). An engine started
from a container without the tower says so at its handshake, and the gateway refuses pictures by
name.

### Inkling-2: the MTP drafter

The checkpoint ships a drafter (`mtp/mtp.safetensors`, 3.4 GB: five decoder layers fed by eight
of the model's layers, blocks of 8 tokens, a recurrent head over the vocabulary). Drafting pays
when checking k drafted tokens in one forward costs much less than k forwards. Here it does not:
a forward over n tokens at the decode position, the same prompt and text fed n at a time, the
Vulkan build on the machine below, cost 0.51-0.69 s for one token, 1.01 s for two, 2.58 s for four
and 4.56 s for eight. Each token brings experts of its own, and most of the step is reading them
(a decode step reads its misses from the disk and computes the rest from RAM). A block of 8
would lose even with every draft accepted, so the drafter is not wired. On a machine that holds
every expert in memory the reads go away and the answer may change; it has not been measured
there.

Validation: `c/tools/make_tiny_inkling2.py` builds a random-weight model with transformers'
`Inkling2ForCausalLM` (sliding and global layers, two blocks of the attention residuals, every
parameter drawn at random) and its oracle, and the engine reproduces it token for token; `--g64`
and `--int4-ref` do the same through the converter, from a fabricated NVFP4/MXFP8 checkpoint to
the container. A GGUF as unsloth's lay it out (`tests/test_gguf_import.py`: folded convolutions,
queries by layer, the NVFP4 layers' experts without their global scale) reads as the checkpoint
it was written from, tokens and logits. CI runs these (`inkling2-oracle`), and the expert tier
against the CPU on Lavapipe for the f32 fixture, the container and a Q8_0 GGUF
(`tests/vulkan_engines.sh inkling2`). `tests/test_inkling2_kernels.c` holds the bf16 and int4 g64
kernels to their references. The gateway's renderer matches the official chat template's output.
`--vision` adds a tiny random tower (`Inkling2ForConditionalGeneration`) and one 2 x 3-tile
picture among the prompt's tokens; the engine's generation matches transformers' token for token
(CI runs it), and the last prompt position's logits agree to 3e-6. `tests/test_inkling2_chat.py`
holds the picture preparation to the processor's rules.

## Modes

| Invocation | What it does |
|---|---|
| `-p "text" [-n N]` | streaming greedy generation (stops at eos or N tokens) |
| `--chat -p "text"` | wraps the prompt in Inkling's chat template (role tokens + `<\|content_text\|>`, `<\|message_model\|>` as the generation prompt). Instruct models fed raw text are out of distribution and answer badly. `THINK=<0..1>` raises the reasoning effort (default 0) |
| `-f prompts.txt [-n N]` | one prompt per line (`#` comments skipped), single model load, state reset between prompts — the cache-warming workflow below |
| `--audio file.dmel [-p "text"]` | spoken input: raw u8 DMel frames `[n_frames, 80]`, one `<\|audio\|>` position per frame (implies `--chat`) |
| `[cap] [bits] [ref.json]` | token-exact oracle harness against a `tools/make_tiny_inkling.py` fixture (CI-style validation; `tools/make_tiny_inkling_audio.py` for the audio path) |

`coli chat` / `coli serve` / `coli web` render the same template through the
gateway, so there is nothing to pass there.

## Audio input (DMel)

Inkling hears through discretized log-mel frames ("DMel"): 80 mel bands per
50 ms of audio, each quantized to 16 levels. Its audio "tower" is one
embedding table plus one norm — a frame's embedding is the RMSNorm of a sum
of 80 table rows, swapped in at that frame's `<|audio|>` position. No
Whisper-style encoder, no extra runtime: with the two audio tensors present
the engine takes spoken input at full precision, and without them it is the
same text-only engine as before.

Two ways to get the tensors:

- **Converting yourself:** add `--keep-audio` to `convert_inkling_int4.py`.
  It passes `model.audio.*` through (~10 MB bf16 on Inkling-Small).
- **Already have a text-only container?** The two tensors can be pulled out
  of the original checkpoint's shards with HTTP range requests — ~10 MB
  moved instead of re-downloading the full bf16 checkpoint — and written as
  an `audio.safetensors` sidecar next to the snapshot. The engine indexes
  every `*.safetensors` in the directory, so it just gets picked up:

```python
import json, struct, urllib.request

REPO = "https://huggingface.co/thinkingmachines/Inkling-Small/resolve/main"
NAMES = ["model.audio.encoder.weight", "model.audio.final_norm.weight"]

def rng(url, a, b):
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
    return urllib.request.urlopen(req).read()

index = json.loads(urllib.request.urlopen(f"{REPO}/model.safetensors.index.json").read())
header, blobs, off = {}, [], 0
for name in NAMES:
    url = f"{REPO}/{index['weight_map'][name]}"
    n = struct.unpack("<Q", rng(url, 0, 7))[0]
    info = json.loads(rng(url, 8, 7 + n))[name]
    a, b = info["data_offsets"]
    raw = rng(url, 8 + n + a, 8 + n + b - 1)
    header[name] = {"dtype": info["dtype"], "shape": info["shape"],
                    "data_offsets": [off, off + len(raw)]}
    blobs.append(raw)
    off += len(raw)
hdr = json.dumps(header).encode()
with open("audio.safetensors", "wb") as f:
    f.write(struct.pack("<Q", len(hdr)) + hdr + b"".join(blobs))
```

Input format: mono 16 kHz. Through the gateway, send a WAV (PCM16 or
float32) as an OpenAI `input_audio` content part — the gateway computes the
DMel frames itself (numpy needed for that path only). Through the CLI, pass
pre-encoded frames as raw bytes with `--audio`. Note that audio inflates the
prompt: ~5 s of speech is ~100 positions, so prefill speed counts double for
spoken turns.

```sh
curl localhost:8000/v1/chat/completions -d '{
  "model": "inkling-colibri",
  "messages": [{"role": "user", "content": [
    {"type": "input_audio", "input_audio": {"data": "<base64 wav>", "format": "wav"}}
  ]}]
}'
```

## Every prompt kept: the prefix store

The sliding layers' rings and every layer's short convolutions cannot be wound back, so a slot
resumes only from the end of what it holds (or a pin, whose global rows must still be in the slot):
an agent's new conversation on the same system prompt read its whole prompt again, and so did a
conversation back after another took the slot. `prefix_store.h` keeps every prompt read, as qwen36
does ([qwen36.md](qwen36.md#the-prefix-store)): the global layers' K/V rows in blocks of 256 tokens
named by their tokens, shared by every prompt that starts with them, and checkpoints of the rest
(the sliding layers' rings, every layer's four convolution banks) where a prompt's messages start
(`<|message_system|>`, `<|message_user|>`, `<|message_model|>`, `<|message_tool|>`) and where a turn
ends. A request takes the deepest checkpoint its prompt starts with when it reaches further than
the slot. `COLI_PREFIX_STORE=0` or `COLI_KV_PREFIX=0` turns it off; not for audio, a pin or a
read-out. `tests/test_inkling_prefix_store.py` holds every request that took the store to a fresh
engine reading the same prompt with the store off, each step's top five logits to the last digit;
`INK_PS_BACKEND=vulkan` the same on the dense chain. The 975B checkpoint has not run it here.

## Expert cache size

`cap` (first positional argument, or `--cap` through `coli`) is how many experts each
layer keeps in RAM. Each slot is ~28 MB on the 975B, so `cap × n_layers × 28 MB` has to
fit next to the resident dense set — with the int4 dense container on a 25 GB host that
means `cap` around 2.

Small values are **correct** but slow: the engine processes the routed experts in rounds
of `cap` and accumulates, so an expert evicted mid-token is re-read rather than silently
replaced by the wrong one. (It used to be silently replaced: acquiring all `S×topk`
slots up front meant a full cache evicted slots that were already handed out for the
current token, and the model computed with whatever landed in them. A prefill of 18
tokens at `topk=6` needs up to 108 distinct experts per layer, so anything below that
produced incoherent output with no error message.)

## Cache warming (same idea as glm's `.coli_usage`)

Expert selections are counted per `(layer, expert)` and written to
`SNAP/.coli_usage` after each generation run. On startup the top `PIN_N`
experts per layer are pinned (non-evictable, loaded in one parallel burst).
Counts **accumulate across runs**, so the ranking converges toward your real
workload — pins trained on a single prompt overfit badly (see the benchmark
table), which is why a diverse warmup matters:

```sh
SNAP=~/Models/inkling_i4 ./c/inkling -f warmup_prompts.txt -n 32
```

| Env | Effect |
|---|---|
| `PIN=off` | disable warming entirely: no seeding, no pins, no history rewrite |
| `PIN=<path>` | alternate history file |
| `PIN_N=<n>` | pins per layer (default `cap/2`; 0 = seed ranking only) |
| `USAGE_SAVE=0` | don't rewrite the history (benchmark runs) |
| `NOGPU=1` / `GPU_DEV=<n>` | disable CUDA / select device |
| `IDOT=0` | byte-exact scalar int kernels (debugging) |
| `INK_SHARED_BATCH=0` | disable shared-expert prefill batching for a controlled A/B; positive values cap rows per chunk. The default uses up to 64 MiB of bounded scratch and never changes decode (`S=1`) |
| `TOPP=<p>` | adaptive routing: keep routed experts up to cumulative weight `p`, drop the tail. **Trims the routing** — fewer experts read per token (the lever that matters on a disk-bound host), but a different computation from the declared top-k. Off by default; the run reports `[topp] … N/M routed used (X% trimmed)` so the trade is measurable. Same semantics as `TOPP` in `colibri.c` and `K3_TOPP` in `kimi_k3.c` |
| First positional arg | expert-cache cap per layer (`0` = auto-size from free RAM) |

## Vulkan: the routed experts on the GPU

`make -C c inkling VK=1`, then `COLI_VULKAN=1`: the routed experts go to the shared
Vulkan expert tier ([vulkan.md](vulkan.md#inkling-and-olmoe)), a cache of experts on
the device that fills at startup from the history the cache warming above keeps
(`SNAP/.coli_usage`, or `PIN=<path>`; the `ref.json` oracle reads none) and adapts
while you chat. Every MoE layer sends the resident experts' (token, rank) pairs to the
device as one batch while the CPU computes the rest and the shared experts, then adds
every rank in routing order, so the sum's order does not depend on what was resident.

- The experts go over as the cache holds them, nothing requantized: the int4
  container (packed nibbles, one scale per row), the int8 container, the runtime
  quantization (`bits` 2 to 8) and f32 (`bits=0`). The fused `gate_up` tensor is
  read as two matrices, gate from row 0 and up from row I.
- The dense matrices (attention, dense MLP, shared experts, lm_head) go to the
  device in their RAM form, the `dense-int4g64/` container's int8 and int4-g64
  included, except on a GPU that shares the CPU's RAM while the tier is on: there
  they stay on the CPU unless `COLI_VK_DENSE=1`.
- With the CUDA backend (`CUDA=1` and a device) or Metal on, the Vulkan tier stays
  off: the other backend wins.
- One line per run and serve turn, `[VK] tier inkling run: device N of M routed
  experts ...`; the dashboard's expert map shows a device-resident expert as tier 2.
  The knobs are `COLI_VK_TIER*` in [ENVIRONMENT.md](ENVIRONMENT.md#vulkan-any-gpu-with-a-vulkan-12-driver).

Status: correct, not yet measured for speed. CI checks the tiny fixture on Lavapipe in
every expert format above, under eviction, with a warm start and through serve, and
again under ASan and UBSan; the 975B has not run on it.

## Performance (975B, Ryzen 9 7900 / 24t, 187 GB DDR5, RTX A6000, NVMe)

24-token greedy decode, 5-token prompt, commit-tagged runs, single run each.
"Trained" = usage history built from the same prompt; "novel" = never-seen prompt.

| Configuration | Prefill | Decode | Cache hit |
|---|---|---|---|
| Stage A (plain LRU, serial I/O, CPU) | 150.2 s | 0.06 tok/s | ~0% |
| + packed-int4 cache, parallel fills, pins (CPU) | 21.1 s | 0.25 tok/s | 81.5% |
| + CUDA resident tier (A6000) | 18.4 s | 0.32 tok/s | 83.6% |
| + deep pins, trained prompt (`PIN_N=64`) | 1.9 s | 2.51 tok/s | 100.0% |
| deep pins, novel prompt (overfit pins) | 33.8 s | 0.17 tok/s | 79.8% |
| **steady state: 11-prompt diverse history, default pins, novel prompt** ¹ | **35.4 s** | **0.25 tok/s** | **82.2%** |

¹ 48-token generation, 13-token prompt. With only a small warmup corpus the
ranking is barely ahead of plain LRU; hit rate (and therefore decode speed)
grows toward the trained-prompt number as real-use history accumulates.

Phase profile at high hit rates: ~90% CPU expert matmul — the next lever is
expert compute on the GPU, not more I/O work.

## Validation

Every mode is token-exact against HF transformers on a tiny random-init oracle
(`c/tools/make_tiny_inkling.py`): f32, int4-container (VNNI and `IDOT=0`
scalar), bf16 residents on CPU, and bf16 residents through the CUDA kernel.
The tokenizer (o200k family, auto-detected by `tok.h`) encodes 357/357 test
strings identically to HF `tokenizers`. The converter round-trips a fabricated
TML-layout checkpoint back through the engine exactly (`--selftest-e2e`).
