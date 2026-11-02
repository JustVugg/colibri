import { afterEach, describe, expect, it, vi } from "vitest"

import { streamChat, type ChatMessage } from "./api"
import { chatMarkdown, exportName, resendFrom } from "./chat"

afterEach(() => vi.unstubAllGlobals())

const user = (content: string, images?: string[]): ChatMessage =>
  images?.length
    ? { id: "u", role: "user", content, images }
    : { id: "u", role: "user", content }

const assistant = (content: string): ChatMessage => ({ id: "a", role: "assistant", content })

describe("resendFrom", () => {
  it("keeps the pictures that arrived with the last user turn", () => {
    const photo = "data:image/png;base64,aaa"
    const messages = [user("what is this?", [photo]), assistant("a cat")]
    expect(resendFrom(messages, 1)).toEqual({
      text: "what is this?",
      previous: [],
      pictures: [photo],
    })
  })

  it("lets an image-only turn be regenerated", () => {
    const photo = "data:image/jpeg;base64,bbb"
    const messages = [user("", [photo]), assistant("a diagram")]
    const retry = resendFrom(messages, 1)
    expect(retry).not.toBeNull()
    expect(retry!.text).toBe("")
    expect(retry!.pictures).toEqual([photo])
  })

  it("uses the nearest user turn and leaves earlier history alone", () => {
    const first: ChatMessage = { id: "u0", role: "user", content: "hi" }
    const reply: ChatMessage = { id: "a0", role: "assistant", content: "hello" }
    const photo = "data:image/png;base64,ccc"
    const second: ChatMessage = { id: "u1", role: "user", content: "and this?", images: [photo] }
    const last: ChatMessage = { id: "a1", role: "assistant", content: "a bird" }
    const messages = [first, reply, second, last]
    expect(resendFrom(messages, 3)).toEqual({
      text: "and this?",
      previous: [first, reply],
      pictures: [photo],
    })
  })

  it("returns null when there is no user turn to resend", () => {
    expect(resendFrom([assistant("orphan")], 0)).toBeNull()
  })

  it("leaves a text-only turn without a pictures field on the wire", () => {
    expect(resendFrom([user("hello"), assistant("hi")], 1)).toEqual({
      text: "hello",
      previous: [],
      pictures: [],
    })
  })
})

describe("regenerate request body", () => {
  it("puts those pictures on the wire as image_url parts", async () => {
    const photo = "data:image/png;base64,ddd"
    const retry = resendFrom([user("describe this", [photo]), assistant("skipped")], 1)
    const fetchMock = vi.fn().mockResolvedValue(new Response("data: [DONE]\n\n", {
      headers: { "content-type": "text/event-stream" },
    }))
    vi.stubGlobal("fetch", fetchMock)
    await streamChat({
      baseUrl: "http://localhost:8000/v1",
      apiKey: "",
      model: "test-model",
      messages: [{ id: "u", role: "user", content: retry!.text, images: retry!.pictures }],
      temperature: 0,
      maxTokens: 8,
      enableThinking: false,
      signal: new AbortController().signal,
      onDelta: () => undefined,
    })
    const body = JSON.parse(fetchMock.mock.calls[0][1].body as string) as {
      messages: Array<{ content: unknown }>
    }
    expect(body.messages[0].content).toEqual([
      { type: "text", text: "describe this" },
      { type: "image_url", image_url: { url: photo } },
    ])
  })

  it("sends an image-only regenerate as image parts with no empty text", async () => {
    const photo = "data:image/png;base64,eee"
    const retry = resendFrom([user("", [photo]), assistant("skipped")], 1)
    const fetchMock = vi.fn().mockResolvedValue(new Response("data: [DONE]\n\n", {
      headers: { "content-type": "text/event-stream" },
    }))
    vi.stubGlobal("fetch", fetchMock)
    await streamChat({
      baseUrl: "http://localhost:8000/v1",
      apiKey: "",
      model: "test-model",
      messages: [{ id: "u", role: "user", content: retry!.text, images: retry!.pictures }],
      temperature: 0,
      maxTokens: 8,
      enableThinking: false,
      signal: new AbortController().signal,
      onDelta: () => undefined,
    })
    const body = JSON.parse(fetchMock.mock.calls[0][1].body as string) as {
      messages: Array<{ content: unknown }>
    }
    expect(body.messages[0].content).toEqual([
      { type: "image_url", image_url: { url: photo } },
    ])
  })
})

describe("chatMarkdown", () => {
  const labels = { user: "You", assistant: "colibrì", reasoning: "Reasoning", image: "image" }
  it("writes a heading a turn, folds the reasoning and leaves pictures as placeholders", () => {
    const messages: ChatMessage[] = [
      { id: "s", role: "system", content: "be brief" },
      { id: "u", role: "user", content: "what is this?", images: ["data:image/png;base64,aaa"] },
      { id: "a", role: "assistant", content: "A cat.", reasoning: "It has whiskers." },
    ]
    const text = chatMarkdown(messages, "qwen3.6", new Date(2026, 9, 11, 9, 30), labels)
    expect(text.startsWith("# colibrì: qwen3.6\n\n")).toBe(true)
    expect(text).not.toContain("be brief")
    expect(text).not.toContain("base64")
    expect(text).toContain("## You\n\n[image 1]\n\nwhat is this?")
    expect(text).toContain("## colibrì\n\n<details><summary>Reasoning</summary>\n\nIt has whiskers.\n\n</details>\n\nA cat.")
  })
})

describe("exportName", () => {
  it("carries the date and the time", () => {
    expect(exportName(new Date(2026, 9, 11, 9, 5), "md")).toBe("colibri-chat-2026-10-11-0905.md")
  })
})
