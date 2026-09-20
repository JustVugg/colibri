import { useEffect, useMemo, useRef, useState } from "react"
import { ArrowLeft, Focus, Minus, Pause, Play, Plus, Search, SlidersHorizontal, X } from "lucide-react"
import { Brain } from "./Brain"
import { useLocale } from "./i18n"
import { brainTopics } from "./lib/brain-topics"
import { createCortex, initialCamera, type AtlasEntry, type CortexNode, type ProjectedNode } from "./lib/cortex"
import "./brain-design.css"

export function BrainWorkspace(props: { baseUrl: string; apiKey: string; connected: boolean }) {
  const { t } = useLocale()
  const [mode, setMode] = useState("explore")
  return <section className="brain-space">
    <nav className="brain-mode-tabs" aria-label={t("nav.brain")}>
      <button aria-current={mode === "explore" ? "page" : undefined} onClick={() => setMode("explore")}>{t("ui.explore")}</button>
      <button aria-current={mode === "live" ? "page" : undefined} onClick={() => setMode("live")}>{t("ui.liveRouting")}</button>
    </nav>
    {mode === "explore" ? <BrainExplorer /> : <div className="brain-live"><Brain {...props} /></div>}
  </section>
}

function BrainExplorer() {
  const { t } = useLocale()
  const canvas = useRef<HTMLCanvasElement>(null)
  const panelRef = useRef<HTMLElement>(null)
  const panelToggle = useRef<HTMLButtonElement>(null)
  const engine = useRef<ReturnType<typeof createCortex> | null>(null)
  const [atlas, setAtlas] = useState<Record<string, AtlasEntry> | null>(null)
  const [failed, setFailed] = useState(false)
  const [retry, setRetry] = useState(0)
  const [topic, setTopic] = useState<string | null>(null)
  const [selected, setSelected] = useState<number | null>(null)
  const [level, setLevel] = useState(0)
  const [panel, setPanel] = useState(false)
  const [query, setQuery] = useState("")
  const [labels, setLabels] = useState<ProjectedNode[]>([])
  const [paused, setPaused] = useState(() => matchMedia("(prefers-reduced-motion: reduce)").matches)
  const [tour, setTour] = useState(false)
  const drag = useRef<{ x: number; y: number; startX: number; startY: number } | null>(null)
  useEffect(() => {
    const controller = new AbortController(); setFailed(false)
    fetch(`${import.meta.env.BASE_URL}experts.json`, { signal: controller.signal }).then(r => { if (!r.ok) throw new Error(String(r.status)); return r.json() }).then(data => { if (!data?.experts || typeof data.experts !== "object") throw new Error("Atlas unavailable"); if (!controller.signal.aborted) setAtlas(data.experts) }).catch(() => { if (!controller.signal.aborted) setFailed(true) })
    return () => controller.abort()
  }, [retry])
  const { topics, nodes } = useMemo(() => {
    const entries = Object.entries(atlas || {}).filter(([id, value]) => /^\d+:\d+$/.test(id) && value && typeof value.affinity === "object" && value.affinity && typeof value.label === "string")
    const nodes: CortexNode[] = []
    const topics = brainTopics.map(topic => {
      const group = entries.filter(([, entry]) => entry.top === topic.key).sort((a, b) => (b[1].affinity[topic.key] || 0) - (a[1].affinity[topic.key] || 0))
      const count = Math.min(36, group.length)
      for (let j = 0; j < count; j++) {
        const [id, entry] = group[Math.floor(j * group.length / count)]
        const [layer, expert] = id.split(":").map(Number), u = (j + .5) / count, latitude = Math.acos(1 - 2 * u), longitude = j * 2.39996323, radius = .22 * Math.cbrt(.18 + .82 * ((j * 13) % 36) / 35)
        nodes.push({ ...entry, id, layer, expert, index: nodes.length, topic: topic.key, x: topic.center[0] + radius * Math.sin(latitude) * Math.cos(longitude), y: topic.center[1] + radius * Math.cos(latitude), z: topic.center[2] + radius * Math.sin(latitude) * Math.sin(longitude) })
      }
      return { ...topic, count: group.length }
    })
    return { topics, nodes }
  }, [atlas])
  useEffect(() => {
    if (!canvas.current || !nodes.length) return
    const renderer = createCortex(canvas.current, topics, nodes, setLabels); engine.current = renderer
    return () => { renderer.dispose(); engine.current = null }
  }, [topics, nodes])
  useEffect(() => { engine.current?.pause(paused) }, [paused, nodes])
  useEffect(() => {
    const preference = matchMedia("(prefers-reduced-motion: reduce)")
    const change = () => setPaused(preference.matches)
    preference.addEventListener("change", change)
    return () => preference.removeEventListener("change", change)
  }, [])
  useEffect(() => {
    const target = topics.find(item => item.key === topic)
    const node = selected === null ? null : nodes[selected]
    engine.current?.move(level === 0 || !target ? initialCamera : level === 2 && node ? { ...initialCamera, x: node.x, y: node.y, z: node.z, distance: .43, pitch: -.1 } : { ...initialCamera, x: target.center[0], y: target.center[1], z: target.center[2], distance: .86, pitch: -.1 }, level, topic, selected)
  }, [level, topic, selected, topics, nodes])
  const closePanel = () => { if (panelRef.current?.contains(document.activeElement)) panelToggle.current?.focus(); setPanel(false) }
  useEffect(() => {
    const escape = (e: KeyboardEvent) => { if (e.key === "Escape") { if (panelRef.current?.contains(document.activeElement)) panelToggle.current?.focus(); setPanel(false) } }
    document.addEventListener("keydown", escape); return () => document.removeEventListener("keydown", escape)
  }, [])
  const enterTopic = (key: string, guided = false) => {
    if (!guided) setTour(false)
    setTopic(key); setLevel(1); setLabels([])
    const group = nodes.filter(node => node.topic === key); setSelected(group[Math.floor(group.length * .72)]?.index ?? null)
    if (!guided && innerWidth > 760) setPanel(true)
  }
  const enterExpert = (index: number) => { setTour(false); setSelected(index); setTopic(nodes[index].topic); setLevel(2); setPanel(true) }
  const reset = () => { setTour(false); setTopic(null); setSelected(null); setLevel(0); setLabels([]); closePanel() }
  useEffect(() => {
    if (!tour) return
    let index = 0
    const next = () => { if (document.hidden) { setTour(false); return }; if (index >= topics.length) { setTour(false); return }; enterTopic(topics[index++].key, true) }
    next(); const timer = setInterval(next, 4400)
    const stop = () => { if (document.hidden) setTour(false) }
    document.addEventListener("visibilitychange", stop)
    return () => { clearInterval(timer); document.removeEventListener("visibilitychange", stop) }
  }, [tour, topics, nodes])
  useEffect(() => {
    const el = canvas.current
    if (!el) return
    const wheel = (e: WheelEvent) => { e.preventDefault(); setTour(false); engine.current?.zoom(Math.sign(e.deltaY) * .08) }
    el.addEventListener("wheel", wheel, { passive: false }); return () => el.removeEventListener("wheel", wheel)
  }, [])
  const current = selected === null ? null : nodes[selected]
  return <div className="cortex-explorer">
    <header className="cortex-heading"><span>BRAIN / ATLAS GLM-5.2</span><h1>{t("ui.enterModel")}</h1></header>
    <div className="cortex-actions">
      <button aria-label={t(tour ? "ui.stopTour" : "ui.tour")} title={t(tour ? "ui.stopTour" : "ui.tour")} aria-pressed={tour} disabled={!nodes.length} onClick={() => setTour(value => !value)}>{tour ? <Pause /> : <Play />}<span>{t(tour ? "ui.stopTour" : "ui.tour")}</span></button>
      <button ref={panelToggle} aria-label={t("ui.topicsDetails")} aria-controls="brain-details" aria-expanded={panel} onClick={() => setPanel(value => !value)}><SlidersHorizontal /><span>{t("ui.topicsDetails")}</span></button>
    </div>
    <nav className="cortex-breadcrumb" aria-label={t("ui.position")}><button onClick={reset}>{t("ui.overview")}</button>{topic && <><span>/</span><button onClick={() => enterTopic(topic)}>{t(`topic.${topic}`)}</button></>}{level === 2 && current && <span>/ L{current.layer} · E{current.expert}</span>}</nav>
    <canvas ref={canvas} tabIndex={0} role="img" aria-label={t("ui.cortexControls")}
      onPointerDown={e => { if (e.button !== 0) return; setTour(false); closePanel(); drag.current = { x: e.clientX, y: e.clientY, startX: e.clientX, startY: e.clientY }; e.currentTarget.setPointerCapture(e.pointerId) }}
      onPointerMove={e => { const d = drag.current; if (!d) return; engine.current?.rotate((e.clientX - d.x) * .006, (e.clientY - d.y) * .006); d.x = e.clientX; d.y = e.clientY }}
      onPointerUp={e => { const d = drag.current; if (d && Math.hypot(e.clientX - d.startX, e.clientY - d.startY) < 5 && level) { const box = e.currentTarget.getBoundingClientRect(), node = engine.current?.closest(e.clientX - box.left, e.clientY - box.top); if (node) enterExpert(node.index) }; drag.current = null; if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId) }}
      onPointerCancel={() => { drag.current = null }}
      onKeyDown={e => { if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "+", "-", "=", "Enter", "Escape"].includes(e.key)) { e.preventDefault(); setTour(false); if (e.key === "ArrowLeft") engine.current?.rotate(-.1, 0); if (e.key === "ArrowRight") engine.current?.rotate(.1, 0); if (e.key === "ArrowUp") engine.current?.rotate(0, -.1); if (e.key === "ArrowDown") engine.current?.rotate(0, .1); if (e.key === "+" || e.key === "=") engine.current?.zoom(-.15); if (e.key === "-") engine.current?.zoom(.15); if (e.key === "Escape") reset(); if (e.key === "Enter") { if (!topic) enterTopic(topics[0].key); else if (current) enterExpert(current.index) } } }} />
    {!atlas && <div className="atlas-state" role="status">{t(failed ? "ui.atlasError" : "ui.atlasLoading")}{failed && <button onClick={() => setRetry(value => value + 1)}>{t("error.retry")}</button>}</div>}
    {atlas && level === 0 && <div className="cortex-regions">{topics.map((item, i) => <button key={item.key} disabled={!item.count} className={i < 5 ? "left" : "right"} style={{ top: `${18 + (i % 5) * 10.5}%` }} onClick={() => enterTopic(item.key)}>{t(`topic.${item.key}`)}<small>{item.count.toLocaleString()} expert</small></button>)}</div>}
    {level > 0 && labels.map(node => <button key={node.id} className="cortex-node" aria-label={`Layer ${node.layer}, expert ${node.expert}`} style={{ left: node.x, top: node.y - 18 }} onClick={() => enterExpert(node.index)}>L{node.layer} · E{node.expert}</button>)}
    <div className="cortex-caption"><span>{topic ? t("ui.inside") : `${topics.length} ${t("ui.topics")}`}</span><strong>{level === 2 && current ? `Expert ${current.expert} · layer ${current.layer}` : topic ? t(`topic.${topic}`) : t("ui.chooseRegion")}</strong></div>
    <div className="cortex-tools">
      <button disabled={!level} aria-label={t("ui.back")} title={t("ui.back")} onClick={() => { if (level === 2 && topic) enterTopic(topic); else reset() }}><ArrowLeft /></button>
      <button aria-label={t("ui.overview")} title={t("ui.overview")} onClick={reset}><Focus /></button>
      <button aria-label={t("ui.zoomOut")} title={t("ui.zoomOut")} onClick={() => { setTour(false); engine.current?.zoom(.2) }}><Minus /></button>
      <button aria-label={t("ui.zoomIn")} title={t("ui.zoomIn")} onClick={() => { setTour(false); engine.current?.zoom(-.2) }}><Plus /></button>
      <button aria-label={t(paused ? "ui.resume" : "ui.pause")} title={t(paused ? "ui.resume" : "ui.pause")} aria-pressed={paused} onClick={() => setPaused(value => !value)}>{paused ? <Play /> : <Pause />}</button>
    </div>
    <aside ref={panelRef} id="brain-details" className="cortex-details" hidden={!panel} aria-label={t("ui.topicsDetails")}>
      <header><h2>{t("ui.topicsDetails")}</h2><button className="icon-action" aria-label={t("ui.close")} onClick={closePanel}><X /></button></header>
      {topic && current && <section className="cortex-expert-details"><span>{t("ui.atlasAffinities")}</span><h3>{t(`topic.${topic}`)}</h3><p>{t(`topic.${topic}.description`)}</p><label>{t("ui.regionExpert")}<select aria-label={t("ui.regionExpert")} value={current.index} onChange={e => enterExpert(Number(e.target.value))}>{nodes.filter(node => node.topic === topic).map(node => <option key={node.id} value={node.index}>Layer {node.layer} · expert {node.expert}</option>)}</select></label><p>{current.label.startsWith("specialist") ? t("ui.specialist") : t("brain.generalist")} · entropy {current.entropy}</p>
        <div className="cortex-affinities">{Object.entries(current.affinity).sort((a, b) => b[1] - a[1]).slice(0, 4).map(([key, value]) => <div key={key}><span>{t(`topic.${key}`)}<b>{(value * 100).toFixed(1)}%</b></span><i><b style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }} /></i></div>)}</div>
      </section>}
      <label className="cortex-search"><Search /><input type="search" aria-label={t("ui.searchTopic")} placeholder={t("ui.searchTopic")} value={query} onChange={e => setQuery(e.target.value)} /></label>
      <div className="cortex-topic-list">{topics.filter(item => `${t(`topic.${item.key}`)} ${t(`topic.${item.key}.description`)}`.toLocaleLowerCase().includes(query.toLocaleLowerCase())).map(item => <button key={item.key} disabled={!item.count} aria-pressed={topic === item.key} onClick={() => enterTopic(item.key)}>{t(`topic.${item.key}`)}<small>{item.count.toLocaleString()} expert</small></button>)}</div>
      <p className="cortex-source">{Object.keys(atlas || {}).length.toLocaleString()} expert · {nodes.length} {t("ui.sample")}<br />{t("ui.atlasNote")}</p>
    </aside>
  </div>
}
