export type ChatRole = "system" | "user" | "assistant"

export interface ChatMessage {
  id: string
  role: ChatRole
  content: string
}

interface OpenAIError {
  error?: { message?: string }
}

function endpoint(baseUrl: string, path: string) {
  return `${baseUrl.replace(/\/+$/, "")}/${path.replace(/^\/+/, "")}`
}

function headers(apiKey: string) {
  return {
    "Content-Type": "application/json",
    ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}),
  }
}

async function responseError(response: Response) {
  const fallback = `${response.status} ${response.statusText}`
  try {
    const body = (await response.json()) as OpenAIError
    return body.error?.message || fallback
  } catch {
    return fallback
  }
}

export async function listModels(baseUrl: string, apiKey: string, signal?: AbortSignal) {
  const response = await fetch(endpoint(baseUrl, "models"), { headers: headers(apiKey), signal })
  if (!response.ok) throw new Error(await responseError(response))
  const body = (await response.json()) as { data?: Array<{ id: string }> }
  return (body.data || []).map((model) => model.id)
}

export function extractSSE(buffer: string) {
  const frames = buffer.split(/\r?\n\r?\n/)
  const rest = frames.pop() || ""
  const data = frames.flatMap((frame) =>
    frame
      .split(/\r?\n/)
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trimStart()),
  )
  return { data, rest }
}

interface StreamChatOptions {
  baseUrl: string
  apiKey: string
  model: string
  messages: ChatMessage[]
  temperature: number
  maxTokens: number
  enableThinking: boolean
  signal: AbortSignal
  onDelta: (text: string) => void
}

export async function streamChat(options: StreamChatOptions) {
  const response = await fetch(endpoint(options.baseUrl, "chat/completions"), {
    method: "POST",
    headers: headers(options.apiKey),
    signal: options.signal,
    body: JSON.stringify({
      model: options.model,
      messages: options.messages.map(({ role, content }) => ({ role, content })),
      temperature: options.temperature,
      max_completion_tokens: options.maxTokens,
      enable_thinking: options.enableThinking,
      stream: true,
      stream_options: { include_usage: true },
    }),
  })
  if (!response.ok) throw new Error(await responseError(response))
  if (!response.body) throw new Error("The server returned an empty stream.")

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""

  const consume = (data: string) => {
    if (data === "[DONE]") return
    const event = JSON.parse(data) as {
      choices?: Array<{ delta?: { content?: string } }>
    }
    const text = event.choices?.[0]?.delta?.content
    if (text) options.onDelta(text)
  }

  while (true) {
    const { value, done } = await reader.read()
    buffer += decoder.decode(value, { stream: !done })
    const parsed = extractSSE(buffer)
    buffer = parsed.rest
    parsed.data.forEach(consume)
    if (done) break
  }
}
