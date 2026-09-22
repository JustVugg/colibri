import { afterEach, describe, expect, it, vi } from "vitest"

import { askBrio, extractSSE, getHealth, getProfile, serverEndpoint, streamChat } from "./api"

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

describe("askBrio", () => {
  /* The options must travel as a field, never folded into the prompt: keeping
     them out of the text is half of what the mode saves, and a refactor that
     "helpfully" appended them would be invisible in the answer. */
  it("sends the options as data and posts to the brio endpoint", async () => {
    const seen: { url?: string; body?: unknown } = {}
    vi.stubGlobal("fetch", vi.fn(async (url: string, init: RequestInit) => {
      seen.url = url
      seen.body = JSON.parse(String(init.body))
      return new Response(JSON.stringify({
        answer: "b", entropy: 0.5, normalize: "mean",
        choices: [{ option: "b", p: 0.7, logprob: -1, mean_logprob: -1, tokens: 1 }],
        usage: { prompt_tokens: 9, completion_tokens: 0, read_tokens: 2, total_tokens: 11 },
      }), { status: 200, headers: { "Content-Type": "application/json" } })
    }))
    const out = await askBrio("http://x/v1", "", "m", "state", "q?", ["a", "b"])
    expect(seen.url).toBe("http://x/v1/brio")
    expect(seen.body).toMatchObject({ model: "m", state: "state", question: "q?", options: ["a", "b"] })
    expect(out.answer).toBe("b")
    expect(out.usage.completion_tokens).toBe(0)
  })

  it("surfaces the server's own message instead of a bare status", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(
      JSON.stringify({ error: { message: "options must be a non-empty array" } }),
      { status: 400, headers: { "Content-Type": "application/json" } })))
    await expect(askBrio("http://x/v1", "", "m", "s", "q", ["a"]))
      .rejects.toThrow("options must be a non-empty array")
  })
})
