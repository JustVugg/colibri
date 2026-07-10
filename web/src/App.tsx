import { useEffect, useMemo, useRef, useState } from "react"
import {
  Activity,
  ArrowUp,
  BrainCircuit,
  CircleStop,
  Cpu,
  Feather,
  KeyRound,
  Link2,
  LoaderCircle,
  MessageSquareText,
  RefreshCw,
  SlidersHorizontal,
  Trash2,
} from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { getHealth, listModels, streamChat, type ChatMessage, type HealthResponse, type StreamChatResult } from "@/lib/api"
import { activeRequests, supportsCacheSlots } from "@/lib/runtime"
import { persistPublicSettings, stored } from "@/lib/storage"
import { cn } from "@/lib/utils"

const message = (role: ChatMessage["role"], content: string): ChatMessage => ({ id: crypto.randomUUID(), role, content })

export default function App() {
  const [baseUrl, setBaseUrl] = useState(() => stored(localStorage, "colibri.baseUrl", "http://127.0.0.1:8000/v1"))
  const [apiKey, setApiKey] = useState("")
  const [models, setModels] = useState<string[]>([])
  const [model, setModel] = useState(() => stored(localStorage, "colibri.model", "glm-5.2-colibri"))
  const [temperature, setTemperature] = useState(0.7)
  const [maxTokens, setMaxTokens] = useState(512)
  const [thinking, setThinking] = useState(false)
  const [cacheSlot, setCacheSlot] = useState(0)
  const [conversations, setConversations] = useState<Record<number, ChatMessage[]>>({ 0: [] })
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [healthError, setHealthError] = useState("")
  const [lastRun, setLastRun] = useState<StreamChatResult | null>(null)
  const [draft, setDraft] = useState("")
  const [loading, setLoading] = useState(false)
  const [connecting, setConnecting] = useState(false)
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState("")
  const abortRef = useRef<AbortController | null>(null)
  const probeRef = useRef<AbortController | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const messages = conversations[cacheSlot] || []
  const kvSlots = Math.max(1, health?.kv_slots || 1)
  const active = activeRequests(health)
  const capacity = health?.scheduler?.capacity || kvSlots
  const failures = health?.scheduler ? health.scheduler.rejected + health.scheduler.timed_out + health.scheduler.cancelled : 0

  const updateMessages = (next: ChatMessage[] | ((current: ChatMessage[]) => ChatMessage[])) =>
    setConversations((current) => ({
      ...current,
      [cacheSlot]: typeof next === "function" ? next(current[cacheSlot] || []) : next,
    }))

  useEffect(() => {
    persistPublicSettings(localStorage, baseUrl, model)
  }, [baseUrl, model])

  useEffect(() => {
    setConnected(false)
    setHealth(null)
    setHealthError("")
  }, [baseUrl, apiKey])

  useEffect(() => () => {
    probeRef.current?.abort()
    abortRef.current?.abort()
  }, [])

  useEffect(() => {
    if (!connected) return
    let disposed = false
    const poll = async () => {
      if (document.visibilityState === "hidden") return
      try {
        const result = await getHealth(baseUrl, apiKey)
        if (!disposed) { setHealth(result); setHealthError("") }
      } catch (cause) {
        if (!disposed) setHealthError(cause instanceof Error ? cause.message : "Runtime metrics unavailable")
      }
    }
    const timer = window.setInterval(() => void poll(), 5000)
    return () => { disposed = true; window.clearInterval(timer) }
  }, [apiKey, baseUrl, connected])

  useEffect(() => {
    if (cacheSlot >= kvSlots) setCacheSlot(0)
  }, [cacheSlot, kvSlots])

  useEffect(() => setLastRun(null), [cacheSlot])

  useEffect(() => bottomRef.current?.scrollIntoView({ behavior: "smooth" }), [messages])

  const connect = async () => {
    probeRef.current?.abort()
    const controller = new AbortController()
    probeRef.current = controller
    setConnecting(true)
    setError("")
    try {
      const found = await listModels(baseUrl, apiKey, controller.signal)
      setModels(found)
      if (found.length && !found.includes(model)) setModel(found[0])
      setConnected(true)
      try {
        setHealth(await getHealth(baseUrl, apiKey, controller.signal))
        setHealthError("")
      } catch (cause) {
        if (!controller.signal.aborted) {
          setHealth(null)
          setHealthError(cause instanceof Error ? cause.message : "Runtime metrics unavailable")
        }
      }
    } catch (cause) {
      if (controller.signal.aborted) return
      setConnected(false)
      setError(cause instanceof Error ? cause.message : "Could not reach the server.")
    } finally {
      if (probeRef.current === controller) { probeRef.current = null; setConnecting(false) }
    }
  }

  const canSend = useMemo(() => draft.trim() && model && !loading, [draft, loading, model])

  const send = async () => {
    const content = draft.trim()
    if (!content || loading) return
    const user = message("user", content)
    const assistant = message("assistant", "")
    const history = [...messages, user]
    setDraft("")
    setError("")
    updateMessages([...history, assistant])
    setLoading(true)
    const controller = new AbortController()
    abortRef.current = controller
    try {
      const result = await streamChat({
        baseUrl,
        apiKey,
        model,
        messages: history,
        temperature,
        maxTokens,
        enableThinking: thinking,
        cacheSlot: supportsCacheSlots(health) ? cacheSlot : undefined,
        signal: controller.signal,
        onDelta: (delta) =>
          updateMessages((current) => current.map((item) =>
            item.id === assistant.id ? { ...item, content: item.content + delta } : item,
          )),
      })
      setLastRun(result)
      setConnected(true)
    } catch (cause) {
      if (controller.signal.aborted) {
        updateMessages((current) => current.filter((item) => item.id !== assistant.id || item.content))
      } else {
        setError(cause instanceof Error ? cause.message : "Generation failed.")
        updateMessages((current) => current.filter((item) => item.id !== assistant.id || item.content))
      }
    } finally {
      abortRef.current = null
      setLoading(false)
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-row">
          <div className="brand-mark"><Feather className="size-5" /></div>
          <div><h1>colibrì</h1><p>local giant, tiny footprint</p></div>
        </div>

        <section className="side-section">
          <div className="section-title"><Link2 className="size-3.5" /> Connection</div>
          <label>API endpoint<Input value={baseUrl} onChange={(event) => setBaseUrl(event.target.value)} /></label>
          <label>API key<div className="relative"><KeyRound className="field-icon" /><Input className="pl-9" type="password" value={apiKey} placeholder="optional" onChange={(event) => setApiKey(event.target.value)} /></div><span className="field-help">Kept in memory only · sent to this endpoint</span></label>
          <Button type="button" variant="secondary" onClick={connect} disabled={connecting}>
            {connecting ? <LoaderCircle className="size-4 animate-spin" /> : <RefreshCw className="size-4" />}
            Probe server
          </Button>
          <div className={cn("connection-state", connected && "connected")} aria-live="polite"><span />{connected ? "Engine reachable" : "Not connected"}</div>
        </section>

        <section className="side-section runtime-section" aria-live="polite">
          <div className="section-title"><Activity className="size-3.5" /> Runtime</div>
          {health?.scheduler ? <>
            <div className="runtime-grid">
              <div><span>Active</span><strong>{active}<small> / {capacity}</small></strong></div>
              <div><span>Queued</span><strong>{health.scheduler.queued}<small> / {health.scheduler.max_queue}</small></strong></div>
              <div><span>Completed</span><strong>{health.scheduler.completed}</strong></div>
              <div><span>Failures</span><strong>{failures}</strong></div>
            </div>
            <div className="runtime-foot"><span className="runtime-dot" /> Scheduler online <code>{kvSlots} KV</code></div>
          </> : <p className="runtime-unavailable">{connected ? (healthError || "Runtime metrics unavailable") : "Probe the server to inspect runtime state."}</p>}
        </section>

        <section className="side-section">
          <div className="section-title"><SlidersHorizontal className="size-3.5" /> Inference</div>
          <label>Model<select value={model} onChange={(event) => setModel(event.target.value)}>{models.length ? models.map((id) => <option key={id}>{id}</option>) : <option>{model}</option>}</select></label>
          {health?.kv_slots && health.kv_slots > 1 ? <label>KV session<select value={cacheSlot} onChange={(event) => setCacheSlot(Number(event.target.value))} disabled={loading}>
            {Array.from({ length: kvSlots }, (_, slot) => <option key={slot} value={slot}>Session {slot + 1}</option>)}
          </select><span className="field-help">Isolated context · conversation follows the selected slot</span></label> : null}
          <label><span className="label-line"><span>Temperature</span><code>{temperature.toFixed(1)}</code></span><input className="range" type="range" min="0" max="2" step="0.1" value={temperature} onChange={(event) => setTemperature(Number(event.target.value))} /></label>
          <label>Max output tokens<Input type="number" min={1} max={4096} value={maxTokens} onChange={(event) => { const value = Number(event.target.value); if (Number.isFinite(value)) setMaxTokens(Math.min(4096, Math.max(1, Math.round(value)))) }} /></label>
          <button type="button" className={cn("toggle-row", thinking && "active")} aria-pressed={thinking} onClick={() => setThinking((value) => !value)}>
            <span><BrainCircuit className="size-4" /> Reasoning</span><i><b /></i>
          </button>
        </section>

        <div className="sidebar-foot"><Cpu className="size-3.5" /><span>OpenAI-compatible transport</span></div>
      </aside>

      <main className="chat-panel">
        <header className="topbar">
          <div><span className="eyebrow">ACTIVE MODEL</span><strong>{model}</strong></div>
          <div className="top-actions">{lastRun?.queueWaitMs != null ? <Badge>queue {Math.round(lastRun.queueWaitMs)}ms</Badge> : null}<Badge><Activity className="size-3" /> slot {cacheSlot + 1}</Badge><Button variant="ghost" size="sm" onClick={() => updateMessages([])} disabled={!messages.length || loading}><Trash2 className="size-3.5" /> Clear</Button></div>
        </header>

        <div className="conversation">
          {!messages.length ? (
            <div className="empty-state">
              <div className="orb"><Feather /></div>
              <span className="eyebrow">COLIBRÌ ENGINE</span>
              <h2>Ask the giant.<br /><em>Keep the machine yours.</em></h2>
              <p>Connect to a local colibrì server and stream responses directly from your hardware. Nothing leaves the endpoint you choose.</p>
              <div className="suggestions">
                {["Explain how expert routing works", "Write a small C benchmark", "Compare RAM and VRAM caching"].map((item) => <button key={item} onClick={() => setDraft(item)}>{item}<ArrowUp className="size-3.5 rotate-45" /></button>)}
              </div>
            </div>
          ) : (
            <div className="message-list">
              {messages.map((item) => (
                <article key={item.id} className={cn("message", item.role)}>
                  <div className="avatar">{item.role === "user" ? "Y" : <Feather className="size-4" />}</div>
                  <div><div className="message-meta">{item.role === "user" ? "You" : "colibrì"}</div><div className="message-body">{item.content || <span className="typing" aria-label="Generating"><i /><i /><i /></span>}</div></div>
                </article>
              ))}
              <div ref={bottomRef} />
            </div>
          )}
        </div>

        <div className="composer-wrap">
          {error && <div className="error-banner" role="alert">{error}</div>}
          <div className="composer">
            <Textarea value={draft} onChange={(event) => setDraft(event.target.value)} placeholder="Message colibrì…" onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); void send() } }} />
            <div className="composer-foot"><span><MessageSquareText className="size-3.5" /> Enter to send · Shift+Enter for newline</span>{loading ? <Button variant="destructive" size="icon" aria-label="Stop generation" onClick={() => abortRef.current?.abort()}><CircleStop className="size-4" /></Button> : <Button size="icon" aria-label="Send message" disabled={!canSend} onClick={() => void send()}><ArrowUp className="size-4" /></Button>}</div>
          </div>
        </div>
      </main>
    </div>
  )
}
