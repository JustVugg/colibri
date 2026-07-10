import { afterEach, describe, expect, it, vi } from "vitest"

import { extractSSE, getHealth, serverEndpoint, streamChat } from "./api"

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
