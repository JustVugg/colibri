import { afterEach, describe, expect, it, vi } from "vitest"

import { streamChat, type ChatMessage } from "./api"
import { appendDelta, continuable, continuation, setFinish } from "./transcript"

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
