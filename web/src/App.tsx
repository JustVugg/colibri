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
import { listModels, streamChat, type ChatMessage } from "@/lib/api"
import { cn } from "@/lib/utils"

const stored = (key: string, fallback: string) => localStorage.getItem(key) || fallback
const message = (role: ChatMessage["role"], content: string): ChatMessage => ({ id: crypto.randomUUID(), role, content })

export default function App() {
  const [baseUrl, setBaseUrl] = useState(() => stored("colibri.baseUrl", "http://127.0.0.1:8000/v1"))
  const [apiKey, setApiKey] = useState(() => stored("colibri.apiKey", ""))
  const [models, setModels] = useState<string[]>([])
  const [model, setModel] = useState(() => stored("colibri.model", "glm-5.2-colibri"))
  const [temperature, setTemperature] = useState(0.7)
  const [maxTokens, setMaxTokens] = useState(512)
  const [thinking, setThinking] = useState(false)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [draft, setDraft] = useState("")
  const [loading, setLoading] = useState(false)
  const [connecting, setConnecting] = useState(false)
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState("")
  const abortRef = useRef<AbortController | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    localStorage.setItem("colibri.baseUrl", baseUrl)
    localStorage.setItem("colibri.apiKey", apiKey)
    localStorage.setItem("colibri.model", model)
  }, [apiKey, baseUrl, model])

  useEffect(() => bottomRef.current?.scrollIntoView({ behavior: "smooth" }), [messages])

  const connect = async () => {
    setConnecting(true)
    setError("")
    try {
      const found = await listModels(baseUrl, apiKey)
      setModels(found)
      if (found.length && !found.includes(model)) setModel(found[0])
      setConnected(true)
    } catch (cause) {
      setConnected(false)
      setError(cause instanceof Error ? cause.message : "Could not reach the server.")
    } finally {
      setConnecting(false)
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
    setMessages([...history, assistant])
    setLoading(true)
    const controller = new AbortController()
    abortRef.current = controller
    try {
      await streamChat({
        baseUrl,
        apiKey,
        model,
        messages: history,
        temperature,
        maxTokens,
        enableThinking: thinking,
        signal: controller.signal,
        onDelta: (delta) =>
          setMessages((current) => current.map((item) =>
            item.id === assistant.id ? { ...item, content: item.content + delta } : item,
          )),
      })
      setConnected(true)
    } catch (cause) {
      if (controller.signal.aborted) {
        setMessages((current) => current.filter((item) => item.id !== assistant.id || item.content))
      } else {
        setError(cause instanceof Error ? cause.message : "Generation failed.")
        setMessages((current) => current.filter((item) => item.id !== assistant.id || item.content))
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
          <label>API key<div className="relative"><KeyRound className="field-icon" /><Input className="pl-9" type="password" value={apiKey} placeholder="optional" onChange={(event) => setApiKey(event.target.value)} /></div></label>
          <Button variant="secondary" onClick={connect} disabled={connecting}>
            {connecting ? <LoaderCircle className="size-4 animate-spin" /> : <RefreshCw className="size-4" />}
            Probe server
          </Button>
          <div className={cn("connection-state", connected && "connected")}><span />{connected ? "Engine reachable" : "Not connected"}</div>
        </section>

        <section className="side-section">
          <div className="section-title"><SlidersHorizontal className="size-3.5" /> Inference</div>
          <label>Model<select value={model} onChange={(event) => setModel(event.target.value)}>{models.length ? models.map((id) => <option key={id}>{id}</option>) : <option>{model}</option>}</select></label>
          <label><span className="label-line"><span>Temperature</span><code>{temperature.toFixed(1)}</code></span><input className="range" type="range" min="0" max="2" step="0.1" value={temperature} onChange={(event) => setTemperature(Number(event.target.value))} /></label>
          <label>Max output tokens<Input type="number" min={1} max={4096} value={maxTokens} onChange={(event) => setMaxTokens(Number(event.target.value))} /></label>
          <button className={cn("toggle-row", thinking && "active")} onClick={() => setThinking((value) => !value)}>
            <span><BrainCircuit className="size-4" /> Reasoning</span><i><b /></i>
          </button>
        </section>

        <div className="sidebar-foot"><Cpu className="size-3.5" /><span>OpenAI-compatible transport</span></div>
      </aside>

      <main className="chat-panel">
        <header className="topbar">
          <div><span className="eyebrow">ACTIVE MODEL</span><strong>{model}</strong></div>
          <div className="top-actions"><Badge><Activity className="size-3" /> stream</Badge><Button variant="ghost" size="sm" onClick={() => setMessages([])} disabled={!messages.length || loading}><Trash2 className="size-3.5" /> Clear</Button></div>
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
                  <div><div className="message-meta">{item.role === "user" ? "You" : "colibrì"}</div><div className="message-body">{item.content || <span className="typing"><i /><i /><i /></span>}</div></div>
                </article>
              ))}
              <div ref={bottomRef} />
            </div>
          )}
        </div>

        <div className="composer-wrap">
          {error && <div className="error-banner">{error}</div>}
          <div className="composer">
            <Textarea value={draft} onChange={(event) => setDraft(event.target.value)} placeholder="Message colibrì…" onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void send() } }} />
            <div className="composer-foot"><span><MessageSquareText className="size-3.5" /> Enter to send · Shift+Enter for newline</span>{loading ? <Button variant="destructive" size="icon" aria-label="Stop generation" onClick={() => abortRef.current?.abort()}><CircleStop className="size-4" /></Button> : <Button size="icon" aria-label="Send message" disabled={!canSend} onClick={() => void send()}><ArrowUp className="size-4" /></Button>}</div>
          </div>
        </div>
      </main>
    </div>
  )
}
