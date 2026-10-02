export type ChatRole = "system" | "user" | "assistant"

export interface ChatMessage {
  id: string
  role: ChatRole
  content: string
  /* Reasoning models stream their thinking on a separate delta field before
     the answer. Kept apart from `content` so it can be rendered as its own
     block and excluded from what is sent back as conversation history. */
  reasoning?: string
  /* Data URIs of the pictures attached to this turn. Kept on the message rather
     than on the draft, because the transcript is resent on every later turn and
     the model has to keep seeing what it was shown. */
  images?: string[]
  /* How an assistant turn's last generation ended: the server's finish_reason;
     "aborted" / "error" when the client stopped or lost the stream; or
     "incomplete" when the stream closed without a finish_reason, which colibri
     always sends last, so its absence means the reply was cut off. Kept on
     the message so it travels with the transcript through slot switches and
     archives; never sent to the server. */
  finish?: string
  /* Both turns of a text-to-image exchange, the prompt and its picture, carry
     mode "image". They stay in the conversation on screen but are never sent to
     a text model, which would read a question followed by an empty answer. */
  mode?: "image"
  /* The picture of an image turn: the request while it runs, then the result.
     `url` appears only once the image is finished. */
  generated?: GeneratedImage
}

interface OpenAIError {
  error?: { message?: string; param?: string | null }
}

export interface SchedulerHealth {
  active: boolean | number
  capacity?: number
  queued: number
  max_queue: number
  queue_timeout_seconds: number
  admitted: number
  completed: number
  rejected: number
  timed_out: number
  cancelled: number
}

export interface TiersHealth {
  vram: number
  ram: number
  disk: number
  vram_gb: number
  ram_gb: number
}

export interface HwinfoHealth {
  cores: number
  ram_total_gb: number
  ram_avail_gb: number
  gpus: number
  vram_total_gb: number
  cpu: string
  gpu: string
}

export interface HealthResponse {
  status: string
  scheduler?: SchedulerHealth
  kv_slots?: number
  tiers?: TiersHealth
  hwinfo?: HwinfoHealth
  /* Whether a message list ending on an assistant turn is continued rather than
     answered fresh (COLI_CONTINUE_ASSISTANT). Absent on older servers. */
  continue_assistant?: boolean
}

export interface ProfileTurn {
  wall_s: number
  prompt_tokens: number
  completion_tokens: number
  expert_disk_s: number
  expert_wait_s: number
  expert_matmul_s: number
  attention_s: number
  lm_head_s: number
  forwards: number
}

export interface ProfileResponse {
  seq: number
  turns: ProfileTurn[]
}

export interface TokenUsage {
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
}

export interface StreamChatResult {
  finishReason: string | null
  usage: TokenUsage | null
  requestId: string | null
  queueWaitMs: number | null
}

export function endpoint(baseUrl: string, path: string) {
  return `${baseUrl.replace(/\/+$/, "")}/${path.replace(/^\/+/, "")}`
}

export function serverEndpoint(baseUrl: string, path: string) {
  return endpoint(baseUrl.replace(/\/v1\/?$/, ""), path)
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

export interface ModelInfo {
  id: string
  /* What a model does beyond chat. colibri marks an image model with
     "image_generation": it answers /v1/images/generations and refuses chat. */
  capabilities?: string[]
}

export async function listModelInfo(baseUrl: string, apiKey: string, signal?: AbortSignal): Promise<ModelInfo[]> {
  const response = await fetch(endpoint(baseUrl, "models"), { headers: headers(apiKey), signal })
  if (!response.ok) throw new Error(await responseError(response))
  const body = (await response.json()) as { data?: Array<{ id: string; capabilities?: unknown }> }
  return (body.data || []).map(({ id, capabilities }) => Array.isArray(capabilities)
    ? { id, capabilities: capabilities.filter((item): item is string => typeof item === "string") }
    : { id })
}

export async function listModels(baseUrl: string, apiKey: string, signal?: AbortSignal) {
  return (await listModelInfo(baseUrl, apiKey, signal)).map((model) => model.id)
}

export function generatesImages(model: ModelInfo | undefined) {
  return !!model?.capabilities?.includes("image_generation")
}

export async function getHealth(baseUrl: string, apiKey = "", signal?: AbortSignal): Promise<HealthResponse> {
  const response = await fetch(serverEndpoint(baseUrl, "health"), { headers: headers(apiKey), signal })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as HealthResponse
}

export async function getProfile(baseUrl: string, apiKey = "", signal?: AbortSignal): Promise<ProfileResponse> {
  const response = await fetch(serverEndpoint(baseUrl, "profile"), { headers: headers(apiKey), signal })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as ProfileResponse
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

export interface SSEEvent {
  event: string
  data: string
}

/* extractSSE with the event names kept. The image endpoint tells progress,
   previews and the result apart by the `event:` line, not by the payload, so the
   name has to survive parsing. A frame without an event line is a "message" (the
   SSE default), which is how `data: [DONE]` arrives; comment lines such as
   ": keepalive" and frames without data are dropped, as the spec says. */
export function extractSSEEvents(buffer: string): { events: SSEEvent[]; rest: string } {
  const frames = buffer.split(/\r?\n\r?\n/)
  const rest = frames.pop() || ""
  const events: SSEEvent[] = []
  for (const frame of frames) {
    let event = "message"
    const data: string[] = []
    for (const line of frame.split(/\r?\n/)) {
      if (line.startsWith("event:")) event = line.slice(6).trim() || "message"
      else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""))
    }
    if (data.length) events.push({ event, data: data.join("\n") })
  }
  return { events, rest }
}

export interface StreamChatOptions {
  baseUrl: string
  apiKey: string
  model: string
  messages: ChatMessage[]
  temperature: number
  maxTokens: number
  enableThinking: boolean
  cacheSlot?: number
  signal: AbortSignal
  onDelta: (text: string) => void
  onReasoning?: (text: string) => void
}

export async function streamChat(options: StreamChatOptions): Promise<StreamChatResult> {
  const response = await fetch(endpoint(options.baseUrl, "chat/completions"), {
    method: "POST",
    headers: headers(options.apiKey),
    signal: options.signal,
    body: JSON.stringify({
      model: options.model,
      /* A turn with pictures goes out in the content-array form the API takes;
         a plain turn stays a string, so a text-only server sees exactly what it
         saw before this existed. */
      messages: options.messages.map(({ role, content, images }) => images?.length
        ? { role, content: [
            ...(content ? [{ type: "text", text: content }] : []),
            ...images.map(url => ({ type: "image_url", image_url: { url } })),
          ] }
        : { role, content }),
      temperature: options.temperature,
      max_completion_tokens: options.maxTokens,
      enable_thinking: options.enableThinking,
      ...(options.cacheSlot === undefined ? {} : { cache_slot: options.cacheSlot }),
      stream: true,
      stream_options: { include_usage: true },
    }),
  })
  if (!response.ok) throw new Error(await responseError(response))
  if (!response.body) throw new Error("The server returned an empty stream.")

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""
  let finishReason: string | null = null
  let usage: TokenUsage | null = null

  const consume = (data: string) => {
    if (data === "[DONE]") return
    const event = JSON.parse(data) as {
      choices?: Array<{ delta?: { content?: string; reasoning_content?: string }; finish_reason?: string | null }>
      usage?: TokenUsage | null
    }
    const choice = event.choices?.[0]
    const text = choice?.delta?.content
    if (text) options.onDelta(text)
    const reasoning = choice?.delta?.reasoning_content
    if (reasoning) options.onReasoning?.(reasoning)
    if (choice?.finish_reason) finishReason = choice.finish_reason
    if (event.usage) usage = event.usage
  }

  while (true) {
    const { value, done } = await reader.read()
    buffer += decoder.decode(value, { stream: !done })
    const parsed = extractSSE(buffer)
    buffer = parsed.rest
    parsed.data.forEach(consume)
    if (done) break
  }

  const queueWaitHeader = response.headers.get("x-colibri-queue-wait-ms")
  const parsedQueueWait = queueWaitHeader === null ? null : Number(queueWaitHeader)
  return {
    finishReason,
    usage,
    requestId: response.headers.get("x-request-id"),
    queueWaitMs: parsedQueueWait !== null && Number.isFinite(parsedQueueWait) ? parsedQueueWait : null,
  }
}

/* Modalita brio: il modello non genera, assegna una probabilita a ogni opzione
 * ammessa. Il ciclo (fotografia del prefisso condiviso, una lettura per
 * opzione, normalizzazione per lunghezza) sta nel gateway: qui si manda una
 * richiesta e si riceve una distribuzione. */
export interface BrioChoice {
  option: string
  p: number
  logprob: number
  mean_logprob: number
  tokens: number
}

export interface BrioResponse {
  answer: string
  entropy: number
  normalize: "mean" | "sum"
  choices: BrioChoice[]
  usage: { prompt_tokens: number; completion_tokens: number; read_tokens: number; total_tokens: number }
}

export async function askBrio(
  baseUrl: string,
  apiKey: string,
  model: string,
  state: string,
  question: string,
  options: string[],
  signal?: AbortSignal,
): Promise<BrioResponse> {
  const response = await fetch(endpoint(baseUrl, "brio"), {
    method: "POST",
    headers: headers(apiKey),
    body: JSON.stringify({ model, state, question, options }),
    signal,
  })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as BrioResponse
}

/* ---- decisions: POST /v1/systemone -------------------------------------------
 * The Jev-compatible decision route (docs/brio.md). `state` is what the
 * questions are about; every question is typed: `noul` answers with the
 * probability of yes, `choice` with a probability per label, `score` with the
 * expected level. The route accepts any `model` name and replies with the one
 * the server runs. Fields this client does not read (an `id`, a `provider`, a
 * `usage.cost`) may appear in the reply and are left alone. */
export type SystemOneQuestion =
  | { type: "noul"; instructions?: string; criteria?: { true?: string; false?: string } }
  | { type: "choice"; instructions?: string; criteria: Record<string, string> }
  | { type: "score"; instructions?: string; criteria: string[] }

export interface SystemOneRequest {
  model: string
  state: string
  questions: Record<string, SystemOneQuestion>
}

export type SystemOneAnswer =
  | { type: "noul"; noul: number }
  | { type: "choice"; choice: string; probabilities: Record<string, number>; confidence: number }
  | { type: "score"; score: number; legend: Record<string, string>; probabilities: Record<string, number>; confidence: number }

export interface SystemOneResponse {
  model: string
  answers: Record<string, SystemOneAnswer>
  usage?: { input_tokens: number; output_tokens: number }
}

export interface SystemOneReply {
  response: SystemOneResponse
  /* The engine's own time and the scheduler queue, from the gateway's
     x-colibri-elapsed-ms and x-colibri-queue-wait-ms headers. A browser only
     sees a header cross-origin when the server exposes it, so either can be
     null even on a server that sends it. */
  engineMs: number | null
  queueMs: number | null
}

/* An HTTP answer that is not a 2xx, with its status kept: a caller tells
   "the route does not exist" (404) from "the request was refused" by it. */
export class HttpError extends Error {
  readonly status: number
  constructor(message: string, status: number) {
    super(message)
    this.name = "HttpError"
    this.status = status
  }
}

const headerNumber = (response: Response, name: string) => {
  const raw = response.headers.get(name)
  const value = raw === null ? NaN : Number(raw)
  return Number.isFinite(value) ? value : null
}

export async function askSystemOne(
  baseUrl: string,
  apiKey: string,
  request: SystemOneRequest,
  signal?: AbortSignal,
): Promise<SystemOneReply> {
  const response = await fetch(endpoint(baseUrl, "systemone"), {
    method: "POST",
    headers: headers(apiKey),
    body: JSON.stringify(request),
    signal,
  })
  if (!response.ok) throw new HttpError(await responseError(response), response.status)
  return {
    response: (await response.json()) as SystemOneResponse,
    engineMs: headerNumber(response, "x-colibri-elapsed-ms"),
    queueMs: headerNumber(response, "x-colibri-queue-wait-ms"),
  }
}

/* ---- text-to-image ----------------------------------------------------------
 * POST /v1/images/generations, in the OpenAI shape plus colibri's `steps`,
 * `seed` and the `colibri` block of the answer. With `stream: true` the gateway
 * sends named SSE events: progress while the engine works, partial_image when
 * it has a preview, completed with the same object the plain request returns. */

export type ImageStage = "encode" | "denoise" | "decode"

export interface ImageProgress {
  stage: ImageStage
  step: number
  steps: number
  elapsed: number
}

export interface ImageTimings {
  encode?: number
  denoise?: number
  decode?: number
}

export interface ImageGenerationResponse {
  created: number
  data: Array<{ b64_json?: string; url?: string; revised_prompt?: string | null }>
  colibri?: { width: number; height: number; seed: number; steps: number; timings?: ImageTimings }
}

/* One picture of the conversation. The request fields are known from the
   start; url, seconds and timings arrive with the result, and seed becomes the
   one the engine reports having used. */
export interface GeneratedImage {
  prompt: string
  width: number
  height: number
  steps: number
  seed: number
  url?: string
  seconds?: number
  timings?: ImageTimings
}

export interface GenerateImageOptions {
  baseUrl: string
  apiKey: string
  model: string
  prompt: string
  width: number
  height: number
  steps: number
  seed: number
  signal: AbortSignal
  onProgress?: (progress: ImageProgress) => void
  /* A preview as a data URI, small and noisy: the latent of the step so far. */
  onPartial?: (url: string, index: number) => void
  /* Streaming was not available and the picture is being made by one plain
     request: nothing more will arrive until it is done. */
  onPlainRequest?: () => void
}

/* The stream did not work as a stream (no body, or a close before the first
   event), so the same request is worth making once without it. Not raised for
   an HTTP error, an abort or a stream that broke halfway: those would only
   start the same generation over, or hide the server's own message. */
class StreamUnavailable extends Error {}

const IMAGE_EVENT = {
  progress: "image_generation.progress",
  partial: "image_generation.partial_image",
  completed: "image_generation.completed",
}

function imageRequest(options: GenerateImageOptions, stream: boolean) {
  return {
    method: "POST",
    headers: headers(options.apiKey),
    signal: options.signal,
    body: JSON.stringify({
      model: options.model,
      prompt: options.prompt,
      size: `${options.width}x${options.height}`,
      n: 1,
      response_format: "b64_json",
      steps: options.steps,
      seed: options.seed,
      stream,
    }),
  }
}

async function imageError(response: Response) {
  const fallback = `${response.status} ${response.statusText}`
  try {
    const body = (await response.json()) as OpenAIError
    if (response.status === 400 && body.error?.param === "stream") return new StreamUnavailable(body.error.message || fallback)
    return new Error(body.error?.message || fallback)
  } catch {
    return new Error(fallback)
  }
}

function streamError(data: string) {
  try {
    const body = JSON.parse(data) as OpenAIError & { message?: string }
    return new Error(body.error?.message || body.message || data)
  } catch {
    return new Error(data)
  }
}

async function streamImage(options: GenerateImageOptions): Promise<ImageGenerationResponse> {
  const response = await fetch(endpoint(options.baseUrl, "images/generations"), imageRequest(options, true))
  if (!response.ok) throw await imageError(response)
  /* A server that ignores `stream` answers with the plain object: that is the
     result, not a failure, and asking again would generate a second picture. */
  if ((response.headers.get("content-type") || "").includes("application/json")) {
    return (await response.json()) as ImageGenerationResponse
  }
  if (!response.body) throw new StreamUnavailable("The server returned an empty stream.")

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""
  let seen = false
  let finished = false
  let result: ImageGenerationResponse | null = null

  const consume = ({ event, data }: SSEEvent) => {
    seen = true
    if (data === "[DONE]") { finished = true; return }
    if (event === "error") throw streamError(data)
    const payload = JSON.parse(data) as Record<string, unknown>
    if (payload.error) throw streamError(data)
    if (event === IMAGE_EVENT.progress) options.onProgress?.(payload as unknown as ImageProgress)
    else if (event === IMAGE_EVENT.partial && typeof payload.b64_json === "string") {
      options.onPartial?.(`data:image/png;base64,${payload.b64_json}`, Number(payload.partial_image_index) || 0)
    } else if (event === IMAGE_EVENT.completed) result = payload as unknown as ImageGenerationResponse
  }

  /* [DONE] ends the read even if the connection stays open; an error event
     cancels the body so the gateway sees the client leave. */
  try {
    while (!finished) {
      const { value, done } = await reader.read()
      const before = buffer.length
      buffer += decoder.decode(value, { stream: !done })
      /* A frame ends at a blank line. A chunk that brings none cannot complete
         one, so the pending frame (the finished image is megabytes of base64)
         is not split again: re-splitting it on every chunk was quadratic. The
         three characters before the chunk catch a boundary cut in half. */
      if (done || /\r?\n\r?\n/.test(buffer.slice(Math.max(0, before - 3)))) {
        const parsed = extractSSEEvents(buffer)
        buffer = parsed.rest
        parsed.events.forEach(consume)
      }
      if (done) break
    }
  } catch (cause) {
    reader.cancel().catch(() => undefined)
    throw cause
  }
  if (finished) reader.cancel().catch(() => undefined)
  if (result) return result
  if (!seen) throw new StreamUnavailable("The server closed the stream without sending anything.")
  throw new Error("The server closed the stream before the image was finished.")
}

async function requestImage(options: GenerateImageOptions): Promise<ImageGenerationResponse> {
  const response = await fetch(endpoint(options.baseUrl, "images/generations"), imageRequest(options, false))
  if (!response.ok) throw await imageError(response)
  return (await response.json()) as ImageGenerationResponse
}

/* Streams when it can, so the page can show the stage and the previews; falls
   back to one plain request when streaming is not available. */
export async function generateImage(options: GenerateImageOptions): Promise<GeneratedImage> {
  let answer: ImageGenerationResponse
  try {
    answer = await streamImage(options)
  } catch (cause) {
    if (!(cause instanceof StreamUnavailable) || options.signal.aborted) throw cause
    options.onPlainRequest?.()
    answer = await requestImage(options)
  }
  const picture = answer.data?.[0]
  const url = picture?.b64_json ? `data:image/png;base64,${picture.b64_json}` : picture?.url
  if (!url) throw new Error("The server answered without an image.")
  const info = answer.colibri
  return {
    prompt: options.prompt,
    width: info?.width ?? options.width,
    height: info?.height ?? options.height,
    steps: info?.steps ?? options.steps,
    seed: info?.seed ?? options.seed,
    url,
    ...(info?.timings ? { timings: info.timings } : {}),
  }
}
