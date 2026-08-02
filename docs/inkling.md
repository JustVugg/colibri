# Inkling (Thinking Machines 975B MoE) on colibri

`c/inkling.c` runs [Thinking Machines Inkling](https://huggingface.co/thinkingmachines/Inkling)
(975B total / 41B active, Apache 2.0) with colibri's expert-streaming approach:
dense weights resident (RAM or VRAM), routed experts streamed from disk with an
LRU + pinned cache. Audio input is supported when the audio tensors are present
(see [Audio input](#audio-input-dmel) below); the vision encoder and the MTP
head are not loaded.

## Quickstart

Pre-converted weights (int4 experts + bf16 residents, ~469 GiB):

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

## Modes

| Invocation | What it does |
|---|---|
| `-p "text" [-n N]` | streaming greedy generation (stops at eos or N tokens) |
| `--chat -p "text"` | wraps the prompt in Inkling's chat template (role tokens + `<|content_text|>`, `<|message_model|>` as the generation prompt). Instruct models fed raw text are out of distribution and answer badly. `THINK=<0..1>` raises the reasoning effort (default 0) |
| `-f prompts.txt [-n N]` | one prompt per line (`#` comments skipped), single model load, state reset between prompts — the cache-warming workflow below |
| `--audio file.dmel [-p "text"]` | spoken input: raw u8 DMel frames `[n_frames, 80]`, one `<|audio|>` position per frame (implies `--chat`) |
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
| `TOPP=<p>` | adaptive routing: keep routed experts up to cumulative weight `p`, drop the tail. **Trims the routing** — fewer experts read per token (the lever that matters on a disk-bound host), but a different computation from the declared top-k. Off by default; the run reports `[topp] … N/M routed used (X% trimmed)` so the trade is measurable. Same semantics as `TOPP` in `colibri.c` and `K3_TOPP` in `kimi_k3.c` |
| First positional arg | expert-cache cap per layer (`0` = auto-size from free RAM) |

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
