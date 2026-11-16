import { afterEach, describe, expect, it, vi } from "vitest"

import { streamChat, type ChatMessage } from "./api"
import { appendDelta, continuable, continuation, setFinish, uncachedPromptTokens } from "./transcript"

afterEach(() => vi.unstubAllGlobals())

const stream = (...deltas: string[]) => new Response(
  deltas.map((content) => `data: ${JSON.stringify({ choices: [{ delta: { content } }] })}\n\n`).join("") + "data: [DONE]\n\n",
  { headers: { "content-type": "text/event-stream" } },
)

describe("continuing a trailing assistant turn", () => {
  const transcript: ChatMessage[] = [
    { id: "u1", role: "user", content: "Capital of France?" },
    { id: "a1", role: "assistant", content: "The capital of France is Par \n", finish: "length" },
  ]

  it("resends the transcript ending on that turn and appends to its bubble", async () => {
    const fetchMock = vi.fn().mockResolvedValue(stream("ís", "."))
    vi.stubGlobal("fetch", fetchMock)

    const request = continuation(transcript)
    expect(request).not.toBeNull()
    let messages = request!.history
    await streamChat({
      baseUrl: "http://localhost:8000/v1", apiKey: "", model: "m", messages,
      temperature: 0, maxTokens: 8, enableThinking: false, signal: new AbortController().signal,
      onDelta: (delta) => { messages = appendDelta(messages, request!.targetId, "content", delta) },
    })

    const sent = JSON.parse(fetchMock.mock.calls[0][1].body).messages
    expect(sent.at(-1)).toEqual({ role: "assistant", content: "The capital of France is Par" })
    expect(messages).toHaveLength(2)
    expect(messages[1]).toMatchObject({ id: "a1", role: "assistant", content: "The capital of France is París." })
  })

  it("has nothing to continue unless an assistant turn with content is last", () => {
    expect(continuation([])).toBeNull()
    expect(continuation(transcript.slice(0, 1))).toBeNull()
    expect(continuation([...transcript.slice(0, 1), { id: "a1", role: "assistant", content: " \n" }])).toBeNull()
  })

  it("offers to continue a turn cut off by max_tokens, a stop or an error, not a completed one", () => {
    const reply = transcript[1]
    for (const finish of ["length", "aborted", "error", "incomplete"]) expect(continuable({ ...reply, finish })).toBe(true)
    expect(continuable({ ...reply, finish: "stop" })).toBe(false)
    expect(continuable({ ...reply, finish: undefined })).toBe(false)
    expect(continuable({ ...reply, content: " ", finish: "length" })).toBe(false)
  })

  it("keeps Continue on a reply whose stream closed before its finish_reason", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(
      `data: ${JSON.stringify({ choices: [{ delta: { content: "ís" } }] })}\n\n`,
      { headers: { "content-type": "text/event-stream" } })))
    const result = await streamChat({
      baseUrl: "http://localhost:8000/v1", apiKey: "", model: "m", messages: transcript,
      temperature: 0, maxTokens: 8, enableThinking: false, signal: new AbortController().signal,
      onDelta: () => {},
    })
    const marked = setFinish(transcript, "a1", result.finishReason)
    expect(marked[1].finish).toBe("incomplete")
    expect(continuable(marked[1])).toBe(true)
  })

  it("records the finish on the turn it streamed into, not on others", () => {
    const marked = setFinish(transcript, "a1", "stop")
    expect(marked[1].finish).toBe("stop")
    expect(marked[0].finish).toBeUndefined()
  })
})

describe("uncachedPromptTokens", () => {
  it("returns full prompt_tokens on the first turn without prior assistant usage", () => {
    const history: ChatMessage[] = [{ id: "u1", role: "user", content: "hi" }]
    expect(uncachedPromptTokens(history, { prompt_tokens: 120, completion_tokens: 30, total_tokens: 150 })).toBe(120)
  })

  it("subtracts previous prompt + completion tokens on follow-up turns", () => {
    const history: ChatMessage[] = [
      { id: "u1", role: "user", content: "hi" },
      { id: "a1", role: "assistant", content: "hello", usage: { prompt_tokens: 100, completion_tokens: 50, total_tokens: 150 } },
      { id: "u2", role: "user", content: "follow up" },
    ]
    expect(uncachedPromptTokens(history, { prompt_tokens: 175, completion_tokens: 20, total_tokens: 195 })).toBe(25)
  })

  it("falls back to subtracting previous prompt_tokens when past reasoning was stripped", () => {
    const history: ChatMessage[] = [
      { id: "u1", role: "user", content: "hi" },
      { id: "a1", role: "assistant", content: "ok", reasoning: "long thought", usage: { prompt_tokens: 100, completion_tokens: 80, total_tokens: 180 } },
      { id: "u2", role: "user", content: "follow up" },
    ]
    expect(uncachedPromptTokens(history, { prompt_tokens: 130, completion_tokens: 10, total_tokens: 140 })).toBe(30)
  })

  it("prefers server-reported prompt_tokens_details.cached_tokens", () => {
    const history: ChatMessage[] = [
      { id: "u1", role: "user", content: "hi" },
      { id: "a1", role: "assistant", content: "hello", usage: { prompt_tokens: 100, completion_tokens: 50, total_tokens: 150 } },
      { id: "u2", role: "user", content: "follow up" },
    ]
    expect(uncachedPromptTokens(history, {
      prompt_tokens: 175,
      completion_tokens: 20,
      total_tokens: 195,
      prompt_tokens_details: { cached_tokens: 160 },
    })).toBe(15)
  })

  it("returns full prompt_tokens when the turn compacted the history", () => {
    const history: ChatMessage[] = [
      { id: "u1", role: "user", content: "hi" },
      { id: "a1", role: "assistant", content: "hello", usage: { prompt_tokens: 100, completion_tokens: 50, total_tokens: 150 } },
      { id: "u2", role: "user", content: "follow up" },
    ]
    expect(uncachedPromptTokens(history, { prompt_tokens: 175, completion_tokens: 20, total_tokens: 195 }, true)).toBe(175)
  })

  it("subtracts cached prefix up to prevTotal on continuation turns", () => {
    const history: ChatMessage[] = [
      { id: "u1", role: "user", content: "hi" },
      { id: "a1", role: "assistant", content: "partial", finish: "length", usage: { prompt_tokens: 100, completion_tokens: 50, total_tokens: 150 } },
    ]
    expect(uncachedPromptTokens(history, { prompt_tokens: 148, completion_tokens: 20, total_tokens: 168 })).toBe(0)
    expect(uncachedPromptTokens(history, { prompt_tokens: 165, completion_tokens: 20, total_tokens: 185 })).toBe(15)
  })
})
