# colibrì web

React/Vite interface for an OpenAI-compatible colibrì server.

```sh
npm install
npm run dev
```

The default endpoint is `http://127.0.0.1:8000/v1`. Start the API server from
PR #21 (or any compatible backend), then use **Probe server** to load its models.

Local validation:

```sh
npm test
npm run build
```

Besides Chat and Brain, the **Profiling** tab charts where the engine spent
each turn's wall time (I/O wait, expert matmul, attention, LM head) from the
server's `/profile` endpoint — a rolling window of per-turn `PROF` snapshots
emitted by the engine.

When `/v1/models` marks the selected model with `"capabilities":
["image_generation"]` (Qwen-Image-2.1 under `coli serve`), the chat view
generates pictures instead of text. The composer then offers size presets
(every side a multiple of 32, from 256 to 2048) or a custom size, the number of
denoising steps and a seed; an empty seed draws a new random one for every
picture. The request goes to `POST /v1/images/generations` with `"stream":
true`: the turn shows the stage (encoding the prompt, denoising step k of N,
decoding), the elapsed time and, when the engine sends them, blurred previews.
If the stream never starts, the same request is made once without it. A
finished picture carries its size, steps, seed and time, and can be downloaded
as PNG, have its seed copied, or be made again with a new seed. Image turns
stay on screen when you switch back to a text model but are never sent to it.

## FlappyBri

The **FlappyBri** tab is colibri's own small game: the hummingbird of the logo
flies between pipes, and touching a pipe, the ground or the top edge ends the
round. It is there to show how fast a decision model answers when it runs
locally in colibri.

```sh
./coli web --model <model-dir>      # API + dashboard; or coli serve and open its address
```

Open **FlappyBri** from the navigation dock and pick who plays.

- **You play**: Space, a click or a tap flaps; P pauses, R restarts. Needs no
  server. The best score is kept per browser, apart from the model's.
- **The model plays**: the page asks `/v1/models` which model the server runs,
  then at every step describes the screen in a few sentences plus compact JSON
  (where the next gap is, how fast the hummingbird is falling) and sends it to
  `POST /v1/systemone` with one question: by default a `noul`, "Should the
  hummingbird flap its wings now?", or a `choice` between `flap` and `glide`
  with what each does. It reads `answers.flap.noul` (or
  `answers.move.probabilities.flap`) and flaps when that is above the
  threshold, 0.5 unless you move it. The endpoint and API key are the
  dashboard's own, from Settings.

There is one request in flight at a time and the game does not wait for it:
the answer is about the screen the model read and is applied to the step on
which it arrives, and the panel says how many steps late that was. A slower
model can still play: the game speed goes from 0.01x to 1x, or follows the
model's pace so that about two steps pass while it decides; the speed it is
actually running at is measured and shown. If the server cannot be reached,
has no model loaded, or runs an image model, the page says so and hands the
controls back to you.

The panel shows the model's name, the last decision with its probability
against the threshold, the latency of each decision (last, average and p95
over the last 100, with a sparkline, and the engine's own time when the
browser can read the `x-colibri-elapsed-ms` header), decisions per second,
steps per decision, score and best. *What the model reads* shows the exact
last request.

The model is not trained to play: it reads the described state and decides
each step. What the page shows is the speed and the calibration of a decision
model running locally, not a game-playing agent. The world is deterministic
for a seed (shown under the field), the physics runs at a fixed 60 steps per
second of game time whatever the display's frame rate, and decorative motion
stops under `prefers-reduced-motion`. The game core, the request builder, the
decision logic and the latency statistics live in `src/lib/flappybri/`, with
their tests.

The test suite stays browser-light: API requests use a mocked `fetch`, while
runtime capability and storage behavior are covered through pure helpers. It
checks that `/health` and `/profile` are resolved next to (not below) the OpenAI `/v1` prefix,
supports both boolean and numeric `scheduler.active` responses, and sends the
colibrì-specific `cache_slot` field only when KV-slot support was advertised.

The endpoint and selected model are persisted locally. API keys are intentionally
memory-only; startup/persistence also removes the legacy `colibri.apiKey` value.
