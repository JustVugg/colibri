import { useEffect, useMemo, useRef, useState } from "react"
import {
  Activity,
  ArrowUp,
  BrainCircuit,
  CircleStop,
  Clock,
  Cpu,
  Database,
  Feather,
  Gauge,
  Globe,
  HardDrive,
  KeyRound,
  Layers,
  Link2,
  LoaderCircle,
  MemoryStick,
  MessageSquareText,
  MonitorDot,
  RefreshCw,
  SlidersHorizontal,
  Timer,
  Trash2,
  Zap,
} from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { getHealth, listModels, streamChat, type ChatMessage, type HealthResponse, type StreamChatResult } from "@/lib/api"
import { activeRequests, supportsCacheSlots } from "@/lib/runtime"
import { Brain } from "./Brain"
import { Profiling } from "./Profiling"
import { persistPublicSettings, stored } from "@/lib/storage"
import { cn } from "@/lib/utils"
import { useLocale } from "./i18n"

const message = (role: ChatMessage["role"], content: string): ChatMessage => {
  let id: string
  try { id = crypto.randomUUID() } catch { id = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => { const r = Math.random() * 16 | 0; return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16) }) }
  return { id, role, content }
}

export default function App() {
  const { t, locale, setLocale, locales } = useLocale()

  const servedByEngine = typeof window !== "undefined" && window.location.port !== "5173" && window.location.protocol.startsWith("http")
  const defaultBase = servedByEngine ? `${window.location.origin}/v1` : "http://127.0.0.1:8000/v1"
  const [baseUrl, setBaseUrl] = useState(() => {
    const saved = stored(localStorage, "colibri.baseUrl", defaultBase)
    if (servedByEngine && saved === "http://127.0.0.1:8000/v1" && defaultBase !== saved) return defaultBase
    return saved
  })
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
  const [streamStart, setStreamStart] = useState<number | null>(null)
  const [tokenCount, setTokenCount] = useState(0)
  const [tokPerSec, setTokPerSec] = useState<number | null>(null)
  const [ttft, setTtft] = useState<number | null>(null)
  const [totalTokens, setTotalTokens] = useState({ prompt: 0, completion: 0 })
  const [connecting, setConnecting] = useState(false)
  const [connected, setConnected] = useState(false)
  const [view, setView] = useState<"chat" | "brain" | "profiling">("chat")
  const [error, setError] = useState("")
  const autoConnected = useRef(false)
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

  // EFFECT #1
  useEffect(() => {
    persistPublicSettings(localStorage, baseUrl, model)
  }, [baseUrl, model])

  // EFFECT #2
  useEffect(() => {
    setConnected(false)
    setHealth(null)
    setHealthError("")
  }, [baseUrl, apiKey])

  // EFFECT #3
  useEffect(() => () => {
    probeRef.current?.abort()
    abortRef.current?.abort()
  }, [])

  // EFFECT #4
  useEffect(() => {
    if (!connected) return
    let disposed = false
    const poll = async () => {
      if (document.visibilityState === "hidden") return
      try {
        const result = await getHealth(baseUrl, apiKey)
        if (!disposed) { setHealth(result); setHealthError("") }
      } catch (cause) {
        if (!disposed) setHealthError(cause instanceof Error ? cause.message : "status.runtimeUnavailable")
      }
    }
    const timer = window.setInterval(() => void poll(), 5000)
    return () => { disposed = true; window.clearInterval(timer) }
  }, [apiKey, baseUrl, connected])

  // EFFECT #5
  useEffect(() => {
    if (cacheSlot >= kvSlots) setCacheSlot(0)
  }, [cacheSlot, kvSlots])

  // EFFECT #6
  useEffect(() => { setLastRun(null) }, [cacheSlot])

  // EFFECT #7
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }) }, [messages])

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
          setHealthError(cause instanceof Error ? cause.message : "status.runtimeUnavailable")
        }
      }
    } catch (cause) {
      if (controller.signal.aborted) return
      setConnected(false)
      setError(cause instanceof Error ? cause.message : "status.serverError")
    } finally {
      if (probeRef.current === controller) { probeRef.current = null; setConnecting(false) }
    }
  }

  if (servedByEngine && !autoConnected.current && !connected) {
    autoConnected.current = true
    setTimeout(() => connect(), 0)
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
    setStreamStart(null)
    setTokenCount(0)
    setTokPerSec(null)
    setTtft(null)
    const t0 = performance.now()
    let firstToken = true
    let count = 0
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
        onDelta: (delta) => {
          if (firstToken) { setTtft(performance.now() - t0); setStreamStart(performance.now()); firstToken = false }
          count++
          setTokenCount(count)
          const elapsed = (performance.now() - (firstToken ? t0 : t0)) / 1000
          if (elapsed > 0.3) setTokPerSec(count / ((performance.now() - t0) / 1000))
          updateMessages((current) => current.map((item) =>
            item.id === assistant.id ? { ...item, content: item.content + delta } : item,
          ))
        },
      })
      const finalElapsed = (performance.now() - t0) / 1000
      if (count > 0 && finalElapsed > 0) setTokPerSec(count / finalElapsed)
      if (result.usage) setTotalTokens(prev => ({
        prompt: prev.prompt + (result.usage?.prompt_tokens || 0),
        completion: prev.completion + (result.usage?.completion_tokens || 0),
      }))
      setLastRun(result)
      setConnected(true)
    } catch (cause) {
      if (controller.signal.aborted) {
        updateMessages((current) => current.filter((item) => item.id !== assistant.id || item.content))
      } else {
        setError(cause instanceof Error ? cause.message : "status.generationFailed")
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
          <div><h1>colibrì</h1><p>{t("brand.tagline")}</p></div>
        </div>

        <section className="side-section">
          <div className="section-title"><Link2 className="size-3.5" /> {t("sidebar.connection")}</div>
          <label>{t("sidebar.endpoint")}<Input value={baseUrl} onChange={(event) => setBaseUrl(event.target.value)} /></label>
          <label>{t("sidebar.apiKey")}<div className="relative"><KeyRound className="field-icon" /><Input className="pl-9" type="password" value={apiKey} placeholder={t("sidebar.apiKeyPlaceholder")} onChange={(event) => setApiKey(event.target.value)} /></div><span className="field-help">{t("sidebar.apiKeyHelp")}</span></label>
          <Button type="button" variant="secondary" onClick={connect} disabled={connecting}>
            {connecting ? <LoaderCircle className="size-4 animate-spin" /> : <RefreshCw className="size-4" />}
            {t("sidebar.probe")}
          </Button>
          <div className={cn("connection-state", connected && "connected")} aria-live="polite"><span />{connected ? t("status.connected") : t("status.notConnected")}</div>
        </section>

        <section className="side-section runtime-section" aria-live="polite">
          <div className="section-title"><Activity className="size-3.5" /> {t("sidebar.runtime")}</div>
          {health?.hwinfo ? <div className="hw-panel">
            {health.hwinfo.cpu ? <div className="hw-row"><Cpu className="size-3.5" /><span>{health.hwinfo.cpu}</span></div> : null}
            {health.hwinfo.gpus > 0 ? <div className="hw-row"><MonitorDot className="size-3.5" /><span>{health.hwinfo.gpus}× GPU<small>{health.hwinfo.vram_total_gb.toFixed(0)} GB VRAM</small></span></div> : null}
            <div className="hw-row"><MemoryStick className="size-3.5" /><span>{health.hwinfo.ram_total_gb.toFixed(0)} GB RAM<small>{health.hwinfo.ram_avail_gb.toFixed(0)} GB free</small></span></div>
            <div className="hw-row"><HardDrive className="size-3.5" /><span>{health.hwinfo.cores} cores</span></div>
          </div> : null}
          {health?.scheduler ? <>
            <div className="runtime-grid">
              <div><span>{t("dashboard.active")}</span><strong>{active}<small> / {capacity}</small></strong></div>
              <div><span>{t("dashboard.queued")}</span><strong>{health.scheduler.queued}<small> / {health.scheduler.max_queue}</small></strong></div>
              <div><span>{t("dashboard.completed")}</span><strong>{health.scheduler.completed}</strong></div>
              <div><span>{t("dashboard.failures")}</span><strong>{failures}</strong></div>
            </div>
            {health.tiers ? (() => {
              const ti = health.tiers
              const total = Math.max(ti.vram + ti.ram + ti.disk, 1)
              return <div className="tier-panel">
                <div className="tier-bar" role="img" aria-label={t("tier.ariaLabel", { vram: ti.vram, ram: ti.ram, disk: ti.disk })}>
                  <span className="tier-vram" style={{ width: `${(100 * ti.vram) / total}%` }} />
                  <span className="tier-ram" style={{ width: `${(100 * ti.ram) / total}%` }} />
                  <span className="tier-disk" style={{ width: `${(100 * ti.disk) / total}%` }} />
                </div>
                <div className="tier-legend">
                  <span><i className="tier-vram" />{t("tier.vram")} <strong>{ti.vram.toLocaleString()}</strong><small>{ti.vram_gb.toFixed(1)} GB</small></span>
                  <span><i className="tier-ram" />{t("tier.ram")} <strong>{ti.ram.toLocaleString()}</strong><small>{ti.ram_gb.toFixed(1)} GB</small></span>
                  <span><i className="tier-disk" />{t("tier.disk")} <strong>{ti.disk.toLocaleString()}</strong></span>
                </div>
              </div>
            })() : null}
            {totalTokens.prompt + totalTokens.completion > 0 ? <div className="session-stats">
              <span><Database className="size-3" /> {t("dashboard.session")} <strong>{totalTokens.prompt.toLocaleString()}</strong> {t("dashboard.prompt")} + <strong>{totalTokens.completion.toLocaleString()}</strong> {t("dashboard.completion")}</span>
            </div> : null}
            <div className="runtime-foot"><span className="runtime-dot" /> {t("sidebar.schedulerOnline")} <code>{kvSlots} KV</code></div>
          </> : <p className="runtime-unavailable">{connected ? (healthError ? t(healthError) : t("status.runtimeUnavailable")) : t("sidebar.runtimeProbe")}</p>}
        </section>

        <section className="side-section">
          <div className="section-title"><SlidersHorizontal className="size-3.5" /> {t("sidebar.inference")}</div>
          <label>{t("sidebar.model")}<select value={model} onChange={(event) => setModel(event.target.value)}>{models.length ? models.map((id) => <option key={id}>{id}</option>) : <option>{model}</option>}</select></label>
          {health?.kv_slots && health.kv_slots > 1 ? <label>{t("sidebar.kvSession")}<select value={cacheSlot} onChange={(event) => setCacheSlot(Number(event.target.value))} disabled={loading}>
            {Array.from({ length: kvSlots }, (_, slot) => <option key={slot} value={slot}>{t("sidebar.sessionLabel", { slot: slot + 1 })}</option>)}
          </select><span className="field-help">{t("sidebar.kvSessionHelp")}</span></label> : null}
          <label><span className="label-line"><span>{t("sidebar.temperature")}</span><code>{temperature.toFixed(1)}</code></span><input className="range" type="range" min="0" max="2" step="0.1" value={temperature} onChange={(event) => setTemperature(Number(event.target.value))} /></label>
          <label>{t("sidebar.maxTokens")}<Input type="number" min={1} max={4096} value={maxTokens} onChange={(event) => { const value = Number(event.target.value); if (Number.isFinite(value)) setMaxTokens(Math.min(4096, Math.max(1, Math.round(value)))) }} /></label>
          <button type="button" className={cn("toggle-row", thinking && "active")} aria-pressed={thinking} onClick={() => setThinking((value) => !value)}>
            <span><BrainCircuit className="size-4" /> {t("sidebar.reasoning")}</span><i><b /></i>
          </button>
        </section>

        <div className="sidebar-foot">
          <div><Cpu className="size-3.5" /><span>{t("sidebar.transport")}</span></div>
          <div className="locale-switcher">
            <Globe className="size-3.5" />
            <select value={locale} onChange={(e) => setLocale(e.target.value)}>
              {locales.map((l) => <option key={l.code} value={l.code}>{l.label}</option>)}
            </select>
          </div>
        </div>
      </aside>

      <main className="chat-panel">
        <header className="topbar">
          <div><span className="eyebrow">{t("topbar.activeModel")}</span><strong>{model}</strong></div>
          <div className="view-tabs">
            <button className={view === "chat" ? "active" : ""} onClick={() => setView("chat")}><MessageSquareText className="size-3.5" /> {t("nav.chat")}</button>
            <button className={view === "brain" ? "active" : ""} onClick={() => setView("brain")}><BrainCircuit className="size-3.5" /> {t("nav.brain")}</button>
            <button className={view === "profiling" ? "active" : ""} onClick={() => setView("profiling")}><Gauge className="size-3.5" /> {t("nav.profiling")}</button>
          </div>
          <div className="top-actions">
              {loading && tokenCount > 0 ? <Badge className="badge-live"><Zap className="size-3 flash" /> {t("topbar.tokens", { n: tokenCount })}</Badge> : null}
              {!loading && tokPerSec != null ? <Badge className="badge-speed"><Gauge className="size-3" /> {t("topbar.tokPerSec", { n: tokPerSec.toFixed(1) })}</Badge> : null}
              {!loading && ttft != null ? <Badge><Timer className="size-3" /> TTFT {(ttft/1000).toFixed(1)}s</Badge> : null}
              {!loading && lastRun?.usage ? <Badge><Layers className="size-3" /> {lastRun.usage.prompt_tokens}→{lastRun.usage.completion_tokens}</Badge> : null}
              {lastRun?.queueWaitMs != null ? <Badge><Clock className="size-3" /> queue {Math.round(lastRun.queueWaitMs)}ms</Badge> : null}
              <Badge><MonitorDot className="size-3" /> {t("topbar.slot", { n: cacheSlot + 1 })}</Badge>
              <Button variant="ghost" size="sm" onClick={() => { updateMessages([]); setTokPerSec(null); setTtft(null); setTokenCount(0); setTotalTokens({prompt:0,completion:0}) }} disabled={!messages.length || loading}><Trash2 className="size-3.5" /> {t("topbar.clear")}</Button>
            </div>
        </header>

        {view === "brain" ? <Brain baseUrl={baseUrl} apiKey={apiKey} connected={connected} />
          : view === "profiling" ? <Profiling baseUrl={baseUrl} apiKey={apiKey} connected={connected} /> : <>

        <div className="conversation">
          {!messages.length ? (
            <div className="empty-state">
              <div className="orb"><Feather /></div>
              <span className="eyebrow">{t("hero.title")}</span>
              <h2>{t("hero.subtitle")}<br /><em>{t("hero.tagline")}</em></h2>
              <p>{t("hero.description")}</p>
              <div className="suggestions">
                {[t("prompts.routing"), t("prompts.benchmark"), t("prompts.caching")].map((item) => <button key={item} onClick={() => setDraft(item)}>{item}<ArrowUp className="size-3.5 rotate-45" /></button>)}
              </div>
            </div>
          ) : (
            <div className="message-list">
              {messages.map((item) => (
                <article key={item.id} className={cn("message", item.role)}>
                  <div className="avatar">{item.role === "user" ? "Y" : <Feather className="size-4" />}</div>
                  <div><div className="message-meta">{item.role === "user" ? t("chat.you") : t("chat.colibri")}</div><div className="message-body">{item.content || <span className="typing" aria-label="Generating"><i /><i /><i /></span>}</div></div>
                </article>
              ))}
              <div ref={bottomRef} />
            </div>
          )}
        </div>

        <div className="composer-wrap">
          {error && <div className="error-banner" role="alert">{t(error)}</div>}
          <div className="composer">
            <Textarea value={draft} onChange={(event) => setDraft(event.target.value)} placeholder={t("chat.placeholder")} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); void send() } }} />
            <div className="composer-foot"><span><MessageSquareText className="size-3.5" /> {t("chat.inputHint")}</span>{loading ? <Button variant="destructive" size="icon" aria-label={t("chat.stop")} onClick={() => abortRef.current?.abort()}><CircleStop className="size-4" /></Button> : <Button size="icon" aria-label={t("chat.send")} disabled={!canSend} onClick={() => void send()}><ArrowUp className="size-4" /></Button>}</div>
          </div>
        </div>
        </>}
      </main>
    </div>
  )
}
