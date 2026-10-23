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

The test suite stays browser-light: API requests use a mocked `fetch`, while
runtime capability and storage behavior are covered through pure helpers. It
checks that `/health` and `/profile` are resolved next to (not below) the OpenAI `/v1` prefix,
supports both boolean and numeric `scheduler.active` responses, and sends the
colibrì-specific `cache_slot` field only when KV-slot support was advertised.

The endpoint and selected model are persisted locally. API keys are intentionally
memory-only; startup/persistence also removes the legacy `colibri.apiKey` value.
