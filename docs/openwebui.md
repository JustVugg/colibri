# Open WebUI and other Ollama clients: `coli gateway`

`coli gateway` is the Ollama API in front of `coli serve`. It listens on Ollama's port
(11434), answers `/api/tags`, `/api/chat`, `/api/generate`, `/api/show`, `/api/ps`,
`/api/pull`, `/api/delete` and `/api/version` in Ollama's forms, and translates each
request into the OpenAI request `coli serve` already takes. Open WebUI, the Ollama CLI,
the Ollama Python and JavaScript libraries, Continue, Zed and the other clients that
speak to Ollama run on colibri without a change on their side: point them at the gateway
where they pointed at Ollama.

Inference is the same C engine as always: the gateway starts a `coli serve` for the model
a request names, keeps it, and starts another when a request names another model. It is a
Python standard-library program (`c/ollama_gateway.py`), no dependency to install.

## Start it

```bash
cd c
./coli gateway                       # 127.0.0.1:11434, the models under ~/colibri-models
./coli gateway --models-dir /nvme/models --default qwen36-35b
./coli gateway --model qwen3.6:35b=/nvme/qwen36_i4 --model qwen3.8=/nvme/qwen38_flash_fp8
./coli stop --port 11434             # ends the gateway and the engine it started
```

Ollama must not be running on the same port: `sudo systemctl stop ollama` (and `disable`,
if the gateway is to take its place at boot; the systemd unit below declares the conflict).

Every directory under `--models-dir` that holds a model (a converted colibri container, a
`coli import`ed GGUF, or a checkpoint the engine reads as it is) is a model; `--model
NAME=PATH` adds one from elsewhere. A request may use the name with or without `:latest`.
`GET /api/tags` lists them with their size, family, format (`colibri`, `gguf`,
`safetensors`) and quantization (the container's `int4`, a GGUF's `Q4_K_M`, Qwen3.8's
`FP8+int4-g64` with its expert sidecar).

### gateway.json: aliases, context, per-model environment

Ollama clients know the models by Ollama's names (`qwen3.6:35b`); the engines want their
flags and environment (the Vulkan or CUDA chain, OpenMP threads, Qwen3.8's `--cap`). Both
go in `~/.local/share/colibri/gateway.json` (`--config` for another path):

```json
{
  "default": "qwen36-35b",
  "models": {
    "qwen36-35b": {
      "path": "/nvme/qwen36_i4",
      "aliases": ["qwen3.6:35b", "qwen3.6", "qwen36"],
      "params": "35B",
      "ctx": 16384,
      "env": {"COLI_VULKAN": "1", "COLI_VK_DEV": "0", "COLI_VK_KEEPALIVE": "1",
              "OMP_NUM_THREADS": "12", "OMP_PLACES": "cores", "OMP_PROC_BIND": "close"}
    },
    "qwen38-flash-next": {
      "path": "/nvme/qwen38_flash_fp8",
      "aliases": ["qwen3.8", "qwen38"],
      "params": "125B",
      "ctx": 16384,
      "kv_slots": 1,
      "think": "low",
      "args": ["--cap", "8"],
      "env": {"COLI_VULKAN": "1", "Q38_EXPERT_INT4": "1"}
    }
  }
}
```

`path` is the model directory, `aliases` the other names a request may use, `ctx` the
context the serve starts with (`--ctx`; unset, the serve takes what the RAM allows, which
on a 94 GB box was 65536 tokens and 7 GB of state taken from Qwen3.8's host pool),
`kv_slots` the conversations served at once (`--kv-slots`, the gateway's `--kv-slots` when
unset), `keep_alive` how long the model stays loaded after its last request (`"20m"`; the gateway's `--keep-alive` when unset, `-1` = until another model is asked for), `args` extra `coli serve` arguments, `env` the environment of that engine (added
to the gateway's own). `think` is the model's thinking when a request says nothing about it (Open WebUI sends no
`think` unless its reasoning toggle is used): `"low"`, `"medium"`, `"high"`, `true` or
`false`; a request's own `think` wins. Without it Qwen3.8 follows its template and thinks
at the highest effort on every question (one answer in a test ran to 5930 tokens);
`"think": "low"` keeps the reasoning short. Qwen3.8 wants `"kv_slots": 1`: with two slots the engine keeps its
Vulkan dense chain off (one conversation's state on the device), and decode fell from 15
to 6 tok/s on the RTX 3070. Qwen3.6's chain serves two slots, but on an 8 GB card the second
conversation's KV mirror does not fit beside the first once a prompt is long (the log says so),
and its attention goes through the host: a 12.7k-token context decoded at 10.5 tok/s with two
slots and at 36 with one, a 20.6k one at 7.8 and 34, and the tier's memory the mirror took did
not come back after the conversation; `"kv_slots": 1` there too. `default` is the model loaded at
start (`--no-preload` skips that) and the one used when a request names none. Models
found under `--models-dir` need no entry; an entry adds names and settings to one.

## Open WebUI

Open WebUI talks to Ollama at `OLLAMA_BASE_URL`; the gateway on the same machine is
`http://127.0.0.1:11434`, which is Open WebUI's default. With the gateway running (and
Ollama stopped), Open WebUI's model list shows colibri's models under the names of
`/api/tags`; chat, the thinking block (`think`), tool calls, JSON output (`format`), images
for models with a vision tower, the title and tag generation after a reply, and the
"download a model" dialog (`/api/pull`, below) go through the gateway. Model switching
in the chat starts the other engine: the first reply after a switch waits for the load
(40 s for Qwen3.6-35B, longer for a model with a 60 GB host pool).

Open WebUI's local embeddings (document upload, web search) do not go to Ollama by
default, and the gateway answers `/api/embed` with 501: keep the embedding setting on
its default (sentence-transformers inside Open WebUI) or point it at another engine.

Clients on other machines need the gateway on the LAN interface, as Ollama with
`OLLAMA_HOST=0.0.0.0`: `--host 0.0.0.0` wants `--api-key` (Open WebUI sends it as the
Ollama connection's key, OpenAI clients as the bearer token) or `COLI_ALLOW_INSECURE_BIND=1`
for a network that is trusted as a whole.

A docker Open WebUI needs the gateway reachable from the container: `--network host`
with `OLLAMA_BASE_URL=http://127.0.0.1:11434`, or a bridge network with
`OLLAMA_BASE_URL=http://host.docker.internal:11434` and the gateway bound to that
interface (`--host 0.0.0.0` requires `--api-key`, which Open WebUI sends as the Ollama
connection's key, or `COLI_ALLOW_INSECURE_BIND=1`).

## What is translated

| Ollama | colibri |
|---|---|
| `messages[].content`, `images` (base64) | OpenAI content parts with `data:` URLs |
| `messages[].tool_calls`, `tools`, `role: tool` | OpenAI tool calls (`arguments` object <-> string) |
| `think` (true/false/"low"/"medium"/"high") | `enable_thinking`, `reasoning_effort` |
| `message.thinking` in the reply | `reasoning_content` (or `<think>` in content with `--think-in-content`, for clients before Ollama 0.9) |
| `format: "json"` / a JSON schema | `response_format` json_object / json_schema |
| `options.num_predict`, `temperature`, `top_p`, `stop`, `presence_penalty`, `frequency_penalty` | the OpenAI fields |
| `options.seed`, `top_k`, `num_ctx`, `repeat_penalty`, ... | dropped (listed once in the gateway's log) |
| `keep_alive: 0` with an empty prompt | unloads the engine; a non-empty prompt loads it |
| `keep_alive` on a request (`"20m"`, `"1h"`, seconds, `-1`) | as Ollama's: the model unloads once it has sat idle that long after the request (never under one); unset, the model's `keep_alive` in gateway.json, else the gateway's `--keep-alive` (COLI_KEEP_ALIVE; default `-1`, loaded until another model is asked for). `/api/ps` shows `expires_at` |
| `prompt_eval_count`, `eval_count`, `eval_duration`, `load_duration` | from the usage block `coli serve` streams last |
| `/api/pull` | `coli setup --model ID` as a child, its download progress as Ollama's status lines (the ids are the setup catalog's) |
| `/api/delete` | removes a model directory under `--models-dir` (nothing elsewhere) |
| `/api/embed`, `/api/embeddings`, `/api/create`, `/api/copy`, `/api/push` | 501 with a line saying so |
| `/v1/*`, `/health`, `/` | passed through to the engine unchanged |

OpenAI clients get the same: `/v1/models` lists every model of the gateway, and a
`/v1/chat/completions` (`/v1/completions`, `/v1/messages`, `/v1/responses`) names its
model by any of the gateway's names, which loads it before the request goes to the
engine. A client with the engine's address and model id works as before.

`GET /api/gateway` is the gateway's own state (model loaded, loading, failed, pulling).
`/api/version` says `0.12.0-colibri`: Open WebUI reads the leading numbers for its
version check.

## systemd

```ini
[Unit]
Description=Colibri gateway (Ollama API on 11434)
After=network-online.target
Conflicts=ollama.service

[Service]
User=me
WorkingDirectory=/home/me/colibri/c
Environment=COLI_VULKAN=1
ExecStart=/usr/bin/python3 coli gateway --host 127.0.0.1 --port 11434 --models-dir /nvme/models
ExecStop=/usr/bin/python3 coli stop --port 11434
Restart=on-failure
TimeoutStopSec=90

[Install]
WantedBy=multi-user.target
```

`sudo systemctl disable --now ollama && sudo systemctl enable --now colibri-gateway` makes
the change; the reverse undoes it. The engines' logs are under
`~/.local/share/colibri/logs/gateway-<model>.log`.

## Measured

Qwen3.6-35B-A3B (int4 container, Vulkan chain on an RTX 3070, 12 OpenMP threads): a 212
token answer over `/api/chat` streamed in 5.1 s end to end, 47 tok/s of decode; the
gateway adds under a millisecond per token (one JSON line out per SSE line in). The load
after a start or a model switch is the engine's: 38-41 s for this model.
