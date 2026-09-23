import { afterEach, describe, expect, it, vi } from "vitest"

import { streamChat, type ChatMessage } from "./api"
import { appendDelta, continuation } from "./transcript"

afterEach(() => vi.unstubAllGlobals())

const stream = (...deltas: string[]) => new Response(
  deltas.map((content) => `data: ${JSON.stringify({ choices: [{ delta: { content } }] })}\n\n`).join("") + "data: [DONE]\n\n",
  { headers: { "content-type": "text/event-stream" } },
)

describe("continuing a trailing assistant turn", () => {
  const transcript: ChatMessage[] = [
    { id: "u1", role: "user", content: "Capital of France?" },
    { id: "a1", role: "assistant", content: "The capital of France is Par \n" },
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
})
