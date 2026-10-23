import { afterEach, describe, expect, it, vi } from "vitest"

import {
  askSystemOne, extractSSE, extractSSEEvents, generateImage, generatesImages, getHealth, getProfile,
  listModelInfo, serverEndpoint, streamChat, type GenerateImageOptions, type ImageProgress,
} from "./api"

afterEach(() => vi.unstubAllGlobals())

describe("extractSSE", () => {
  it("keeps an incomplete frame for the next network chunk", () => {
    const parsed = extractSSE('data: {"choices":[]}\n\ndata: {"cho')
    expect(parsed.data).toEqual(['{"choices":[]}'])
    expect(parsed.rest).toBe('data: {"cho')
  })

  it("supports CRLF and multiple data frames", () => {
    const parsed = extractSSE("data: one\r\n\r\ndata: two\r\n\r\n")
    expect(parsed.data).toEqual(["one", "two"])
    expect(parsed.rest).toBe("")
  })
})

describe("runtime API", () => {
  it.each([
    ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8000/health"],
    ["https://example.test/api/v1/", "https://example.test/api/health"],
    ["https://example.test/api", "https://example.test/api/health"],
  ])("resolves the health endpoint outside the OpenAI v1 prefix", (baseUrl, expected) => {
    expect(serverEndpoint(baseUrl, "health")).toBe(expected)
  })

  it("requests health with the configured bearer credential", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ status: "ok", scheduler: { active: true } })))
    vi.stubGlobal("fetch", fetchMock)

    await expect(getHealth("http://localhost:8000/v1/", "secret")).resolves.toMatchObject({ status: "ok" })
    expect(fetchMock).toHaveBeenCalledWith("http://localhost:8000/health", expect.objectContaining({
      headers: expect.objectContaining({ Authorization: "Bearer secret" }),
    }))
  })

  it("requests the profiling history next to the OpenAI v1 prefix", async () => {
    const turn = {
      wall_s: 2.5, prompt_tokens: 7, completion_tokens: 12,
      expert_disk_s: 0.4, expert_wait_s: 0.1, expert_matmul_s: 0.9,
      attention_s: 0.6, lm_head_s: 0.2, forwards: 15,
    }
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ seq: 1, turns: [turn] })))
    vi.stubGlobal("fetch", fetchMock)

    await expect(getProfile("http://localhost:8000/v1/")).resolves.toEqual({ seq: 1, turns: [turn] })
    expect(fetchMock).toHaveBeenCalledWith("http://localhost:8000/profile", expect.anything())
  })
})

describe("chat request extensions", () => {
  const completedStream = () => new Response("data: [DONE]\n\n", {
    headers: { "content-type": "text/event-stream" },
  })

  async function requestBody(cacheSlot?: number) {
    const fetchMock = vi.fn().mockResolvedValue(completedStream())
    vi.stubGlobal("fetch", fetchMock)
    await streamChat({
      baseUrl: "http://localhost:8000/v1",
      apiKey: "",
      model: "test-model",
      messages: [],
      temperature: 0,
      maxTokens: 8,
      enableThinking: false,
      cacheSlot,
      signal: new AbortController().signal,
      onDelta: () => undefined,
    })
    return JSON.parse(fetchMock.mock.calls[0][1].body as string) as Record<string, unknown>
  }

  it("omits cache_slot for a generic OpenAI-compatible backend", async () => {
    expect(await requestBody()).not.toHaveProperty("cache_slot")
  })

  it("sends cache_slot zero when colibrì advertises KV slots", async () => {
    expect(await requestBody(0)).toMatchObject({ cache_slot: 0 })
  })
})

describe("reasoning stream", () => {
  /* The server puts thinking on delta.reasoning_content (openai_server.py,
     22 call sites). #1148 / #1153 / #1156 taught the dashboard to read that
     field. The workspace redesign dropped the callback, so a turn with
     enable_thinking on streamed thinking tokens that never appeared. */
  const reasoningStream = () => new Response(
    'data: {"choices":[{"delta":{"reasoning_content":"think"}}]}\n\n'
    + 'data: {"choices":[{"delta":{"content":"answer"}}]}\n\n'
    + "data: [DONE]\n\n",
    { headers: { "content-type": "text/event-stream" } },
  )

  it("forwards reasoning_content to onReasoning and keeps it out of content", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(reasoningStream()))
    const content: string[] = []
    const reasoning: string[] = []
    await streamChat({
      baseUrl: "http://localhost:8000/v1",
      apiKey: "",
      model: "test-model",
      messages: [],
      temperature: 0,
      maxTokens: 8,
      enableThinking: true,
      signal: new AbortController().signal,
      onDelta: (text) => content.push(text),
      onReasoning: (text) => reasoning.push(text),
    })
    expect(reasoning).toEqual(["think"])
    expect(content).toEqual(["answer"])
  })
})

describe("askSystemOne", () => {
  /* The options must travel as the labels of a `choice`, never folded into the
     state: the gateway reads each one as the answer's continuation, and a
     refactor that "helpfully" appended them to the text would be invisible in
     the answer. */
  it("asks one choice question on /v1/systemone and draws its probabilities", async () => {
    const seen: { url?: string; body?: { state: string; questions: { q: { type: string; criteria: object } } } } = {}
    vi.stubGlobal("fetch", vi.fn(async (url: string, init: RequestInit) => {
      seen.url = url
      seen.body = JSON.parse(String(init.body))
      return new Response(JSON.stringify({
        id: "req_1", model: "laya", provider: "colibri",
        answers: { q: { type: "choice", choice: "b", probabilities: { a: 0.25, b: 0.75 }, confidence: 0.5 } },
        usage: { input_tokens: 40, output_tokens: 3, cost: 0 },
      }), { status: 200, headers: { "Content-Type": "application/json" } })
    }))
    const out = await askSystemOne("http://x/v1", "", "laya", "state", "q?", ["a", "b"])
    expect(seen.url).toBe("http://x/v1/systemone")
    expect(seen.body?.state).toBe("state")
    expect(seen.body?.questions.q).toEqual({ type: "choice", instructions: "q?", criteria: { a: null, b: null } })
    expect(out.answer).toBe("b")
    expect(out.choices.map((c) => c.option)).toEqual(["b", "a"])
    expect(out.entropy).toBeCloseTo(-(0.25 * Math.log(0.25) + 0.75 * Math.log(0.75)) / Math.log(2), 6)
    expect(out.usage).toEqual({ input_tokens: 40, output_tokens: 3 })
  })

  it("surfaces the server's own message instead of a bare status", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(
      JSON.stringify({ error: { message: "`questions.q.criteria` must be a non-empty object" } }),
      { status: 422, headers: { "Content-Type": "application/json" } })))
    await expect(askSystemOne("http://x/v1", "", "m", "s", "q", []))
      .rejects.toThrow("must be a non-empty object")
  })
})

describe("extractSSEEvents", () => {
  it("keeps the event name of each frame and defaults to message", () => {
    const parsed = extractSSEEvents(
      'event: image_generation.progress\ndata: {"stage":"encode"}\n\n'
      + "data: [DONE]\n\n"
      + "event: image_generation.completed\ndata: {\"cre")
    expect(parsed.events).toEqual([
      { event: "image_generation.progress", data: '{"stage":"encode"}' },
      { event: "message", data: "[DONE]" },
    ])
    expect(parsed.rest).toBe('event: image_generation.completed\ndata: {"cre')
  })

  it("skips comments and empty frames, joins multi-line data, accepts CRLF", () => {
    const parsed = extractSSEEvents(": keepalive\r\n\r\nevent: x\r\ndata: a\r\ndata: b\r\n\r\nevent: lonely\r\n\r\n")
    expect(parsed.events).toEqual([{ event: "x", data: "a\nb" }])
    expect(parsed.rest).toBe("")
  })
})

describe("model capabilities", () => {
  it("reads capabilities from /v1/models and recognises an image model", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ data: [
      { id: "qwen-image-2.1-colibri", object: "model", capabilities: ["image_generation", 7] },
      { id: "glm-5.2-colibri", object: "model" },
    ] }))))
    const models = await listModelInfo("http://x/v1", "")
    expect(models).toEqual([
      { id: "qwen-image-2.1-colibri", capabilities: ["image_generation"] },
      { id: "glm-5.2-colibri" },
    ])
    expect(models.map(generatesImages)).toEqual([true, false])
  })
})

describe("generateImage", () => {
  const encoder = new TextEncoder()
  const sse = (event: string, data: unknown) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`
  const answer = {
    created: 1790000000,
    data: [{ b64_json: "iVBORw0KGgo=", revised_prompt: null }],
    colibri: { width: 768, height: 512, seed: 1234, steps: 8, timings: { encode: 1.2, denoise: 80, decode: 9.1 } },
  }
  const events = [
    sse("image_generation.progress", { stage: "encode", step: 0, steps: 8, elapsed: 0.1 }),
    sse("image_generation.progress", { stage: "denoise", step: 3, steps: 8, elapsed: 12.3 }),
    sse("image_generation.partial_image", { b64_json: "UFJFVklFVw==", partial_image_index: 0 }),
    sse("image_generation.progress", { stage: "decode", step: 8, steps: 8, elapsed: 80.5 }),
    sse("image_generation.completed", answer),
    "data: [DONE]\n\n",
  ].join("")

  /* A body cut at arbitrary byte offsets, the way the network delivers it,
     optionally left open after the last chunk like a kept-alive connection. */
  const streamed = (text: string, { cut = 37, close = true } = {}) => new Response(new ReadableStream<Uint8Array>({
    start(controller) {
      for (let at = 0; at < text.length; at += cut) controller.enqueue(encoder.encode(text.slice(at, at + cut)))
      if (close) controller.close()
    },
  }), { headers: { "content-type": "text/event-stream" } })

  const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), {
    status, headers: { "content-type": "application/json" },
  })

  const options = (extra: Partial<GenerateImageOptions> = {}): GenerateImageOptions => ({
    baseUrl: "http://localhost:8000/v1/",
    apiKey: "secret",
    model: "qwen-image-2.1-colibri",
    prompt: "a hummingbird",
    width: 768,
    height: 512,
    steps: 8,
    seed: 1234,
    signal: new AbortController().signal,
    ...extra,
  })

  const bodyOf = (call: unknown[]) => JSON.parse(String((call[1] as RequestInit).body)) as Record<string, unknown>

  it("streams progress, previews and the result, and posts the contract's body", async () => {
    const fetchMock = vi.fn().mockResolvedValue(streamed(events))
    vi.stubGlobal("fetch", fetchMock)
    const progress: ImageProgress[] = []
    const previews: Array<[string, number]> = []
    const image = await generateImage(options({
      onProgress: (item) => progress.push(item),
      onPartial: (url, index) => previews.push([url, index]),
    }))

    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(fetchMock.mock.calls[0][0]).toBe("http://localhost:8000/v1/images/generations")
    expect((fetchMock.mock.calls[0][1] as RequestInit).headers).toMatchObject({ Authorization: "Bearer secret" })
    expect(bodyOf(fetchMock.mock.calls[0])).toEqual({
      model: "qwen-image-2.1-colibri", prompt: "a hummingbird", size: "768x512", n: 1,
      response_format: "b64_json", steps: 8, seed: 1234, stream: true,
    })
    expect(progress.map((item) => [item.stage, item.step, item.steps])).toEqual([
      ["encode", 0, 8], ["denoise", 3, 8], ["decode", 8, 8],
    ])
    expect(previews).toEqual([["data:image/png;base64,UFJFVklFVw==", 0]])
    expect(image).toEqual({
      prompt: "a hummingbird", width: 768, height: 512, steps: 8, seed: 1234,
      url: "data:image/png;base64,iVBORw0KGgo=", timings: { encode: 1.2, denoise: 80, decode: 9.1 },
    })
  })

  it("stops reading at [DONE] even when the connection stays open", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(streamed(events, { close: false })))
    await expect(generateImage(options())).resolves.toMatchObject({ seed: 1234 })
  })

  it("reports the seed the engine used over the one that was asked for", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(streamed(
      sse("image_generation.completed", { ...answer, colibri: { ...answer.colibri, seed: 99 } }) + "data: [DONE]\n\n")))
    await expect(generateImage(options())).resolves.toMatchObject({ seed: 99 })
  })

  it("shows the server's own message for an HTTP 400 and does not retry", async () => {
    const fetchMock = vi.fn().mockResolvedValue(json({ error: {
      message: "Width and height must be multiples of 32 between 256 and 2048.", type: "invalid_request_error", param: "size",
    } }, 400))
    vi.stubGlobal("fetch", fetchMock)
    await expect(generateImage(options())).rejects.toThrow("Width and height must be multiples of 32 between 256 and 2048.")
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it("surfaces a busy engine (503) as its message", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({ error: { message: "The image engine is busy." } }, 503)))
    await expect(generateImage(options())).rejects.toThrow("The image engine is busy.")
  })

  it("fails with the event's message when the stream carries an error", async () => {
    const fetchMock = vi.fn().mockResolvedValue(streamed(
      sse("image_generation.progress", { stage: "encode", step: 0, steps: 8, elapsed: 0 })
      + sse("error", { error: { message: "engine crashed" } })))
    vi.stubGlobal("fetch", fetchMock)
    await expect(generateImage(options())).rejects.toThrow("engine crashed")
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it("does not start over when the stream breaks after it began", async () => {
    const fetchMock = vi.fn().mockResolvedValue(streamed(
      sse("image_generation.progress", { stage: "denoise", step: 2, steps: 8, elapsed: 20 })))
    vi.stubGlobal("fetch", fetchMock)
    await expect(generateImage(options())).rejects.toThrow("before the image was finished")
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it("falls back to one plain request when the stream closes before any event", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(streamed(""))
      .mockResolvedValueOnce(json(answer))
    vi.stubGlobal("fetch", fetchMock)
    const plain = vi.fn()
    await expect(generateImage(options({ onPlainRequest: plain }))).resolves.toMatchObject({ url: "data:image/png;base64,iVBORw0KGgo=" })
    expect(plain).toHaveBeenCalledTimes(1)
    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(bodyOf(fetchMock.mock.calls[1])).toMatchObject({ stream: false, seed: 1234 })
  })

  it("falls back when the server refuses the stream parameter", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(json({ error: { message: "stream is not supported", param: "stream" } }, 400))
      .mockResolvedValueOnce(json(answer))
    vi.stubGlobal("fetch", fetchMock)
    await expect(generateImage(options())).resolves.toMatchObject({ seed: 1234 })
    expect(bodyOf(fetchMock.mock.calls[1])).toMatchObject({ stream: false })
  })

  it("takes a plain JSON answer to a stream request as the result", async () => {
    const fetchMock = vi.fn().mockResolvedValue(json(answer))
    vi.stubGlobal("fetch", fetchMock)
    await expect(generateImage(options())).resolves.toMatchObject({ width: 768, height: 512 })
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it("rejects with AbortError when cancelled mid-stream, without a fallback request", async () => {
    const fetchMock = vi.fn((_url: string, init: RequestInit) => Promise.resolve(new Response(new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode(sse("image_generation.progress", { stage: "denoise", step: 1, steps: 8, elapsed: 3 })))
        init.signal?.addEventListener("abort", () => controller.error(new DOMException("The operation was aborted.", "AbortError")))
      },
    }), { headers: { "content-type": "text/event-stream" } })))
    vi.stubGlobal("fetch", fetchMock)
    const controller = new AbortController()
    const run = generateImage(options({ signal: controller.signal, onProgress: () => controller.abort() }))
    await expect(run).rejects.toMatchObject({ name: "AbortError" })
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it("rejects with AbortError when cancelled before the answer arrives", async () => {
    const fetchMock = vi.fn((_url: string, init: RequestInit) => new Promise<Response>((_, reject) => {
      init.signal?.addEventListener("abort", () => reject(new DOMException("The operation was aborted.", "AbortError")))
    }))
    vi.stubGlobal("fetch", fetchMock)
    const controller = new AbortController()
    const run = generateImage(options({ signal: controller.signal }))
    controller.abort()
    await expect(run).rejects.toMatchObject({ name: "AbortError" })
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })
})
