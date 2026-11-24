# colibri's chat UI

One page (`index.html`, no build step) and a small server (`server.py`, Python's standard
library only) in front of a colibri gateway.

```
coli web --model ~/colibri-models/qwen36-35b       # the gateway on 127.0.0.1:8000 and the UI on :8790, the browser opened on it
```

`coli web` (and `coli start`, after `coli setup`) starts the UI beside the gateway, on the
gateway's port + 790 (`--ui-port` picks another), and stops it with the gateway. `coli ui`
starts the UI alone, next to a gateway that already runs (`coli serve`), here or elsewhere
with `--backend http://HOST:PORT`. The server proxies
`/v1/`, `/health`, `/profile` and `/experts` to the gateway, so the page and the API share
one origin (the gateway sends no CORS headers); a chat stream passes through chunk by chunk,
and when the page stops a request the upstream connection closes, which stops the engine.

While the UI listens on this machine only (the default), those paths answer only a request whose
`Host` names this machine (`127.0.0.1`, `localhost`, `[::1]`), as the gateway itself does, so a web
page on another site cannot reach the gateway through a name it re-points here. If the UI is
reached through a reverse proxy or a tunnel name, add that name with `--allowed-host NAME`
(repeatable, or `COLI_ALLOWED_HOSTS=a,b`). A UI bound to `0.0.0.0` or a LAN address is unchanged.

The page-only routes (`/agents`, `/models/`, `/web/`, `/doc/extract`, which download and switch
models, open agents and fetch pages) answer only a caller on this machine: their `Host`, `Origin` and
`X-Colibri-UI` checks are headers the caller writes, so they also check the caller's address. On a UI
bound to `0.0.0.0` or a LAN address a request from another machine gets 403. Docker's port mapping and a
reverse proxy show a peer that is not loopback; there, and only if you want the Models page and the rest
reachable that way, start the UI with `--allow-remote-ui` (or `COLI_UI_ALLOW_REMOTE=1`, which `coli serve`
passes on). That gives anyone who can reach the port those routes, with no key.

## What it does

- **Chat**: streaming, the reasoning in its own block (open or closed while it streams, the
  choice kept), Markdown, copy, regenerate and continue, conversations kept in the browser.
  A turn stopped while it was still reasoning continues inside its open block (Qwen3.6 and
  Qwen3.8). HTML code blocks open as a page in the right panel.
- **Pictures**: "attach an image" appears only when the model card lists image input, which
  the gateway does only when the engine loaded its vision tower; pictures are resized to
  1280 px in the browser.
- **Documents**: "attach a document" (or a file dropped on the chat) reads a PDF, a Word
  (`.docx`), OpenDocument (`.odt`) or PowerPoint (`.pptx`) file, a web page or any text file
  (`docs.py`, `POST /doc/extract`, this page only) and puts its text in the message, with the
  file's name; the message shows it as a chip. A PDF's pages without text (scans) come as
  pictures, at most 6, for a model with a vision tower; a PDF needs PyMuPDF or pypdf, and
  pypdf reads text only. The old binary formats (`.doc`, `.xls`, `.ppt`) are refused with the
  way forward. At most 80 MB a file.
- **Expert map**: one card per transformer block, its experts in a grid, colored by where
  they live (RAM or VRAM) and by what this answer used. Engines that send per-step frames
  light the experts each step routes (`/experts?since=N`).
- **Monitor**: the turn in flight (reading the prompt or writing, tokens, tok/s over the
  last 32 tokens) and the box (RAM, the engine's RSS and CPU, load), from `/health`.
- **Profiling**: where each turn's time went, from `/profile`.
- **System One**: a decision with a probability, `POST /v1/systemone`.
- **Coding agents** (Settings, "Agents"): Pi, Codex, Claude Code, opencode, OpenClaw and
  Hermes, found on this machine and opened in a new terminal window already pointed at
  colibri, through their own options and environment (Windows and WSL, with the UI in WSL or
  on Windows itself; from Windows, a WSL agent needs WSL to reach the server on 127.0.0.1,
  `networkingMode=mirrored`). Codex speaks
  only the Responses API: `responses_shim.py` translates it to the gateway's chat API.
- **Web search**: with it on, the model can search and read pages before it answers
  (`web.py`), and decides itself when and what to search; a small model does so more often
  with reasoning on. Results from a SearXNG instance when `--searx URL` or `COLIBRI_SEARX`
  names one, else DuckDuckGo's plain HTML results (no key), its lite page where the first
  answers with its bot check; the pages' readable text only, at public addresses only.
- **Mascot**: the hummingbird follows the turn (reading, reasoning, writing, done, stopped).

- **Models**: what fits the machine the gateway runs on and how fast it would go, each model
  three ways (all in RAM, with the GPU, the experts streamed from the SSD); the models on its
  disk, our catalog and a search over the GGUFs on Hugging Face that colibri reads, with each
  variant's own estimate (`models.py`).

## The Models page

Everything on it is read on the machine that runs it: the CPU, the RAM, the GPUs and the free
disk from colibri's own probe (the one `coli setup` uses), the RAM bandwidth (numpy, when it is
there) and the SSD's random direct reads (Linux) measured there; "Measure again" repeats both
while the engine is idle. A speed is either **measured**, the median of the gateway's own
finished turns with that model, on the CPU or with the GPU, or an **estimate**: the bytes a
token reads (the dense part plus k of E experts, from the tensor headers, read over HTTP without
downloading anything) over the measured bandwidth. A dedicated GPU gets no estimate until it is
measured; an integrated one only for the engines `coli setup` runs on it, with the ratio
measured on a Radeon 780M on that GPU alone.

It runs inside `coli ui` when the gateway is on the same machine (`--backend` on 127.0.0.1, the
default). Models are looked for in `--models-dir` (repeatable), else in the folder `coli setup`
uses (`~/colibri-models`), and in the folder of the model the gateway serves; downloads go to the
first. With the gateway on another machine, run the service there and point `coli ui` at it:

```
python3 ui/models.py --backend http://127.0.0.1:8000     # on the engine's machine, 127.0.0.1:8791
coli ui --backend http://BOX:8000 --models-backend http://127.0.0.1:8791   # through an ssh tunnel
```

"Use" restarts the running `coli serve` with its own command, folder and environment, only
`--model` changed ("Use with the GPU" and "Use the CPU only" turn the GPU on or off); when the new
model does not start, the previous one is started again. The process is found through /proc,
so switching works on Linux (and WSL); elsewhere the page says how to restart it by hand.

## Safety

The server listens on 127.0.0.1. `/agents` answers only this page (its own Host and Origin,
and a header of its own on a POST), and launches only an agent from its list, with a model
the gateway lists, a folder that exists and an API key of plain characters. `/web/` and
`/models/` answer only this page too; `/web/` opens public addresses only, and `/models/`
switches only to a folder under its models folders and stops only a `coli serve` process. API keys stay in the browser and in the
agents' own environment. What leaves the machine: the chat to the gateway, and with web search
on, the query to the search engine and the pages the model chooses to read.
