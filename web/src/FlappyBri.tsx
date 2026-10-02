import { useCallback, useEffect, useRef, useState } from "react"
import { Bird, Cpu, LoaderCircle, Pause, Play, RotateCcw, X } from "lucide-react"

import {
  STEPS_PER_SECOND, WORLD, accumulate, createWorld, nextBest, readBest, step, writeBest,
  type Phase, type World,
} from "@/lib/flappybri/game"
import { Pilot, QUESTION_ID, buildQuestions, findModel, type Decision, type Form, type PilotFailure } from "@/lib/flappybri/pilot"
import { draw, makeSprite, readPalette, type Palette, type TrailPoint } from "@/lib/flappybri/render"
import type { LatencySummary } from "@/lib/flappybri/stats"
import { useLocale } from "./i18n"
import "./flappybri.css"

/* FlappyBri: colibri's hummingbird between the pipes, flown by you or by the
 * model loaded in colibri. The model mode is the point of the page: every step
 * the game describes the screen to POST /v1/systemone and flaps when the
 * model says so, live, and the HUD shows how fast and how sure it is. The game
 * itself lives in lib/flappybri (pure, tested); this file is the loop, the
 * canvas and the panel around it. */

type Mode = "human" | "model"

/* Game speed stops, from a hundredth of real time to real time. A model that
   needs a second per answer still plays at 0.01x: the pipes wait for it. */
export const SPEEDS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1] as const
/* Matching the model's pace: the speed at which about this many steps pass
   while the model decides. Two keeps a plain rule alive; more and it falls. */
const STEPS_PER_DECISION = 2
const TRAIL = 120
const SPARK = 100
const RESTART_MS = 1600

const SETTINGS_KEY = "colibri.flappybri"
const bestKey = (mode: Mode) => `colibri.flappybri.best.${mode}`

interface Settings {
  mode: Mode
  form: Form
  threshold: number
  speed: number         // index into SPEEDS
  match: boolean        // follow the model's pace instead of the slider
  autoRestart: boolean
}

const DEFAULTS: Settings = { mode: "human", form: "noul", threshold: 0.5, speed: SPEEDS.length - 1, match: true, autoRestart: true }

function loadSettings(): Settings {
  try {
    const raw = JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}") as Partial<Settings>
    return {
      mode: raw.mode === "model" ? "model" : "human",
      form: raw.form === "choice" ? "choice" : "noul",
      threshold: typeof raw.threshold === "number" && raw.threshold > 0 && raw.threshold < 1 ? raw.threshold : DEFAULTS.threshold,
      speed: Number.isInteger(raw.speed) && raw.speed! >= 0 && raw.speed! < SPEEDS.length ? raw.speed! : DEFAULTS.speed,
      match: typeof raw.match === "boolean" ? raw.match : DEFAULTS.match,
      autoRestart: typeof raw.autoRestart === "boolean" ? raw.autoRestart : DEFAULTS.autoRestart,
    }
  } catch {
    return DEFAULTS
  }
}

const storage = () => {
  try { return window.localStorage } catch { return undefined }
}

const randomSeed = () => {
  try { return crypto.getRandomValues(new Uint32Array(1))[0] } catch { return Math.floor(Math.random() * 2 ** 32) }
}

const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value))

interface Hud {
  phase: Phase
  score: number
  best: number
  newBest: boolean
  seed: number
  decision: Decision | null
  summary: LatencySummary | null
  rate: number | null
  spark: number[]
  busy: boolean
  speed: number               // the speed the game is set to run at now
  effective: number | null    // what it is actually running at, measured
  lastState: string
}

/* The mutable side of the game, out of React: the loop reads and writes it
   sixty times a second, and React only sees a snapshot ten times a second. */
interface Engine {
  world: World
  mode: Mode
  settings: Settings
  carry: number
  last: number | null
  queued: boolean
  pilot: Pilot | null
  ema: number | null
  matchSpeed: number
  trail: TrailPoint[]
  crashAt: number | null
  steps: number
  clock: Array<[number, number]>
  best: number
  newBest: boolean
  flushedAt: number
  dirty: boolean
  reduced: boolean
  palette: Palette | null
  restartTimer: number | null
}

const fmtMs = (ms: number) => (ms < 10 ? ms.toFixed(1) : String(Math.round(ms)))
const fmtSpeed = (speed: number) => (speed >= 0.1 ? speed.toFixed(2).replace(/0$/, "") : speed.toFixed(2))

export default function FlappyBri({ baseUrl, apiKey, model, connected }: {
  baseUrl: string; apiKey: string; model: string; connected: boolean
}) {
  const { t } = useLocale()
  const [settings, setSettings] = useState<Settings>(loadSettings)
  const [preparing, setPreparing] = useState(false)
  const [modelName, setModelName] = useState<string | null>(null)
  const [failure, setFailure] = useState<PilotFailure | null>(null)
  const stage = useRef<HTMLDivElement>(null)
  const canvas = useRef<HTMLCanvasElement>(null)
  const probe = useRef<AbortController | null>(null)

  const engine = useRef<Engine>(null as unknown as Engine)
  if (!engine.current) {
    const mode = settings.mode
    engine.current = {
      world: createWorld(randomSeed()), mode, settings, carry: 0, last: null, queued: false,
      pilot: null, ema: null, matchSpeed: SPEEDS[0], trail: [], crashAt: null,
      steps: 0, clock: [], best: readBest(storage(), bestKey(mode)), newBest: false,
      flushedAt: 0, dirty: true, reduced: false, palette: null, restartTimer: null,
    }
  }

  const speedOf = (e: Engine) =>
    e.mode === "model" && e.settings.match ? e.matchSpeed : SPEEDS[e.settings.speed]

  const snapshot = (e: Engine): Hud => {
    const now = performance.now()
    const recent = e.clock.filter(([at]) => now - at <= 1000)
    let effective: number | null = null
    if (e.world.phase === "playing" && recent.length > 1) {
      const [t0, s0] = recent[0]
      const [t1, s1] = recent[recent.length - 1]
      if (t1 - t0 > 250) effective = ((s1 - s0) * 1000) / (t1 - t0) / STEPS_PER_SECOND
    }
    const pilot = e.pilot
    return {
      phase: e.world.phase, score: e.world.score, best: e.best, newBest: e.newBest, seed: e.world.seed,
      decision: pilot?.last ?? null, summary: pilot?.latency.summary() ?? null, rate: pilot?.latency.rate() ?? null,
      spark: pilot ? pilot.latency.samples().slice(-SPARK) : [], busy: !!pilot?.busy,
      speed: speedOf(e), effective, lastState: pilot?.lastState ?? "",
    }
  }

  const [hud, setHud] = useState<Hud>(() => snapshot(engine.current))
  const flush = useCallback(() => {
    const e = engine.current
    e.flushedAt = performance.now()
    e.dirty = false
    setHud(snapshot(e))
  }, [])

  /* Settings are remembered per browser; the engine sees them at once. */
  useEffect(() => {
    const e = engine.current
    e.settings = settings
    if (e.pilot) { e.pilot.form = settings.form; e.pilot.threshold = settings.threshold }
    try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings)) } catch { /* restricted storage */ }
    e.dirty = true
  }, [settings])
  const patch = (change: Partial<Settings>) => setSettings((current) => ({ ...current, ...change }))

  const focusStage = () => stage.current?.focus({ preventScroll: true })

  const clearRestart = () => {
    const e = engine.current
    if (e.restartTimer !== null) { window.clearTimeout(e.restartTimer); e.restartTimer = null }
  }

  /* A fresh round on a new seed, waiting or already flying. */
  const newRound = (play: boolean) => {
    const e = engine.current
    clearRestart()
    e.pilot?.stop()
    e.world = createWorld(randomSeed())
    e.carry = 0
    e.last = null
    e.queued = false
    e.trail = []
    e.crashAt = null
    e.newBest = false
    e.clock = []
    if (play) e.world.phase = "playing"
    flush()
  }

  /* The model failed or is missing: say why and hand the controls back. */
  const fallBack = (why: PilotFailure) => {
    const e = engine.current
    clearRestart()
    e.pilot?.stop()
    setFailure(why)
    setPreparing(false)
    if (e.world.phase === "playing") e.world.phase = "paused"
    e.mode = "human"
    e.best = readBest(storage(), bestKey("human"))
    patch({ mode: "human" })
    flush()
  }
  const fallBackRef = useRef(fallBack)
  fallBackRef.current = fallBack

  /* The pace follows answers as they arrive: waiting for the next step to
     apply one would, at the slowest speed, wait more than a second. */
  const onAnswer = (decision: Decision) => {
    const e = engine.current
    e.ema = e.ema === null ? decision.latencyMs : e.ema * 0.75 + decision.latencyMs * 0.25
    e.matchSpeed = clamp((STEPS_PER_DECISION * 1000) / (e.ema * STEPS_PER_SECOND), SPEEDS[0], 1)
    e.dirty = true
  }

  const onApply = (decision: Decision) => {
    const e = engine.current
    e.trail.push({ distance: e.world.distance, y: e.world.y, flap: decision.flap })
    if (e.trail.length > TRAIL) e.trail.shift()
    e.dirty = true
  }

  /* Check the server has a model that decides, then fly. */
  const startModel = async () => {
    const e = engine.current
    probe.current?.abort()
    const controller = new AbortController()
    probe.current = controller
    setPreparing(true)
    setFailure(null)
    const found = await findModel(baseUrl, apiKey, model, controller.signal)
    if (controller.signal.aborted || e.mode !== "model") return
    probe.current = null
    if ("failure" in found) { fallBack(found.failure); return }
    setPreparing(false)
    setModelName(found.model)
    if (!e.pilot || e.pilot.model !== found.model || e.pilot.baseUrl !== baseUrl || e.pilot.apiKey !== apiKey) {
      e.pilot?.stop()
      e.pilot = new Pilot({
        baseUrl, apiKey, model: found.model, form: e.settings.form, threshold: e.settings.threshold, window: SPARK,
        onAnswer, onApply, onFailure: (why) => fallBackRef.current(why),
      })
      e.ema = null
      e.matchSpeed = SPEEDS[0]
    }
    e.pilot.reset()
    newRound(true)
    focusStage()
  }
  const startModelRef = useRef(startModel)
  startModelRef.current = startModel

  const pause = () => {
    const e = engine.current
    if (e.world.phase !== "playing") return
    e.world.phase = "paused"
    e.pilot?.stop()
    e.queued = false
    flush()
  }
  const resume = () => {
    const e = engine.current
    if (e.world.phase !== "paused") return
    if (e.mode === "model" && !e.pilot) { void startModelRef.current(); return }
    e.world.phase = "playing"
    e.last = null
    e.clock = []
    flush()
    focusStage()
  }
  const restart = () => {
    const e = engine.current
    if (e.mode === "model") void startModelRef.current()
    else { newRound(false); focusStage() }
  }

  /* Space, a click or a tap: what it means depends on who plays. */
  const act = () => {
    const e = engine.current
    const phase = e.world.phase
    if (phase === "paused") { resume(); return }
    if (e.mode === "model") {
      if ((phase === "ready" || phase === "over") && !preparingRef.current) void startModelRef.current()
      return
    }
    if (phase === "over") {
      /* a tap that was meant for the last flap should not start the next round */
      if (e.crashAt !== null && performance.now() - e.crashAt < 450) return
      newRound(true)
      e.queued = true
      return
    }
    if (phase === "ready") { e.world.phase = "playing"; e.last = null; flush() }
    e.queued = true
  }
  const actRef = useRef(act)
  actRef.current = act
  const preparingRef = useRef(preparing)
  preparingRef.current = preparing
  const pauseRef = useRef(pause)
  pauseRef.current = pause
  const restartRef = useRef(restart)
  restartRef.current = restart

  const chooseMode = (mode: Mode) => {
    const e = engine.current
    if (mode === e.mode) return
    probe.current?.abort()
    setPreparing(false)
    setFailure(null)
    e.mode = mode
    e.best = readBest(storage(), bestKey(mode))
    patch({ mode })
    newRound(false)
    if (mode === "model") probeModel()
  }

  /* Show who would play before anyone presses Start: the model's name, or
     why there is none (and the controls back to you), right away. */
  const probeModel = () => {
    probe.current?.abort()
    const controller = new AbortController()
    probe.current = controller
    setPreparing(true)
    void findModel(baseUrl, apiKey, model, controller.signal).then((found) => {
      if (controller.signal.aborted || engine.current.mode !== "model") return
      probe.current = null
      setPreparing(false)
      if ("failure" in found) fallBackRef.current(found.failure)
      else setModelName(found.model)
    })
  }
  const probeModelRef = useRef(probeModel)
  probeModelRef.current = probeModel

  /* On opening the page in model mode, and on a new endpoint or key: the pilot
     built for the old server is dropped and the new one asked what it runs. */
  useEffect(() => {
    const e = engine.current
    if (e.pilot && (e.pilot.baseUrl !== baseUrl || e.pilot.apiKey !== apiKey)) {
      e.pilot.stop()
      e.pilot = null
      if (e.world.phase === "playing" && e.mode === "model") { e.world.phase = "paused"; flush() }
    }
    if (e.mode === "model" && !e.pilot) { setModelName(null); probeModelRef.current() }
  }, [baseUrl, apiKey, flush])

  /* ---- the loop ---------------------------------------------------------------- */
  useEffect(() => {
    const e = engine.current
    const node = canvas.current
    const ctx = node?.getContext("2d")
    if (!node || !ctx) return
    const sprites = { up: makeSprite(false), beat: makeSprite(true) }
    let font = "700 52px ui-sans-serif, system-ui, sans-serif"
    let disposed = false
    try {
      void document.fonts?.load("52px Bytesized").then((faces) => {
        if (!disposed && faces.length) font = "52px Bytesized, ui-sans-serif, system-ui, sans-serif"
      }).catch(() => undefined)
    } catch { /* no font loading API */ }

    const motion = window.matchMedia?.("(prefers-reduced-motion: reduce)")
    const onMotion = () => { e.reduced = !!motion?.matches }
    onMotion()
    motion?.addEventListener?.("change", onMotion)

    /* The canvas follows the theme: tokens are read again when it changes. */
    const repaint = () => { e.palette = readPalette(document.documentElement) }
    repaint()
    const themes = new MutationObserver(repaint)
    themes.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme", "class", "style"] })
    const scheme = window.matchMedia?.("(prefers-color-scheme: dark)")
    scheme?.addEventListener?.("change", repaint)

    /* Crisp at any size: the backing store is the CSS size times the pixel
       ratio, and the world is scaled into it. */
    let scale = 1
    const resize = () => {
      const box = node.getBoundingClientRect()
      const ratio = window.devicePixelRatio || 1
      const width = Math.max(1, Math.round(box.width * ratio))
      const height = Math.max(1, Math.round(box.height * ratio))
      if (node.width !== width || node.height !== height) { node.width = width; node.height = height }
      scale = width / WORLD.width
    }
    resize()
    const sizer = new ResizeObserver(resize)
    sizer.observe(node)

    let frame = 0
    const tick = (now: number) => {
      const dt = e.last === null ? 0 : now - e.last
      e.last = now
      const world = e.world
      if (world.phase === "playing") {
        const out = accumulate(e.carry, dt, speedOf(e))
        e.carry = out.carry
        for (let i = 0; i < out.steps && world.phase === "playing"; i++) {
          let flap = false
          if (e.mode === "model") flap = e.pilot?.take(world)?.flap ?? false
          else if (e.queued) { flap = true; e.queued = false }
          const events = step(world, flap)
          e.steps += 1
          if (events.scored) {
            e.dirty = true
            if (world.score > e.best) {
              e.best = nextBest(e.best, world.score)
              e.newBest = true
              writeBest(storage(), bestKey(e.mode), e.best)
            }
          }
          if (events.died) {
            e.crashAt = now
            e.dirty = true
            e.pilot?.stop()
            if (e.mode === "model" && e.settings.autoRestart) {
              clearRestart()
              e.restartTimer = window.setTimeout(() => {
                e.restartTimer = null
                if (e.mode === "model" && e.world.phase === "over") { e.pilot?.reset(); newRound(true) }
              }, RESTART_MS)
            }
          }
        }
        e.clock.push([now, e.steps])
        while (e.clock.length && now - e.clock[0][0] > 1200) e.clock.shift()
        if (e.mode === "model" && world.phase === "playing") e.pilot?.poll(world)
      }

      ctx.setTransform(scale, 0, 0, scale, 0, 0)
      ctx.imageSmoothingEnabled = false
      draw(ctx, {
        world, alpha: e.carry, palette: e.palette || readPalette(null), reduced: e.reduced, time: now,
        sprites, trail: e.mode === "model" ? e.trail : [], font,
        crashAge: e.crashAt === null ? null : now - e.crashAt,
      })

      /* The panel follows at ten frames a second while a round runs, and
         at once on anything that changes it (a point, a crash, a setting). */
      if (e.dirty || (world.phase === "playing" && now - e.flushedAt > 100)) flush()
      frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)

    /* A hidden tab pauses the round: nobody is watching, and a model mode
       would keep asking for nothing. */
    const hidden = () => { if (document.visibilityState === "hidden") pauseRef.current() }
    document.addEventListener("visibilitychange", hidden)

    return () => {
      disposed = true
      cancelAnimationFrame(frame)
      sizer.disconnect()
      themes.disconnect()
      motion?.removeEventListener?.("change", onMotion)
      scheme?.removeEventListener?.("change", repaint)
      document.removeEventListener("visibilitychange", hidden)
      clearRestart()
      e.pilot?.stop()
      probe.current?.abort()
    }
  }, [])

  /* Keys: Space flaps (or starts, or resumes), P pauses, R restarts. Ignored
     while typing in a field, and Space is left to a focused button. */
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey) return
      const target = event.target instanceof Element ? event.target : null
      if (target?.closest("input, textarea, select, [contenteditable=''], [contenteditable=true], dialog")) return
      if (event.code === "Space" || event.key === " ") {
        const control = target && target !== stage.current && target.closest("button, a, summary, [role=radio]")
        if (control) return
        event.preventDefault()
        if (!event.repeat) actRef.current()
      } else if (event.key === "p" || event.key === "P") {
        if (engine.current.world.phase === "playing") pauseRef.current()
        else if (engine.current.world.phase === "paused") actRef.current()
      } else if (event.key === "r" || event.key === "R") {
        restartRef.current()
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [])

  /* ---- the page ------------------------------------------------------------------ */
  const isModel = settings.mode === "model"
  const phase = hud.phase
  const primary = phase === "playing"
    ? { label: t("flappy.pause"), Icon: Pause, run: pause }
    : phase === "paused"
      ? { label: t("flappy.resume"), Icon: Play, run: resume }
      : { label: t("flappy.start"), Icon: Play, run: () => { if (isModel) void startModel(); else { act(); focusStage() } } }

  const decision = hud.decision
  const summary = hud.summary
  /* Measured when a round is running: the steps the game actually ran per
     decision it actually got. An answer waits for the next step to be applied
     and the next question for the step after it, so this is the latency in
     steps plus up to one: the number that decides whether the model keeps up. */
  const stepsPerDecision = hud.rate && hud.effective
    ? (hud.effective * STEPS_PER_SECOND) / hud.rate
    : summary ? (summary.avg / 1000) * hud.speed * STEPS_PER_SECOND : null
  const failureText = failure ? t(`flappy.err.${failure.kind}`, {
    url: baseUrl, model: failure.model || "", message: failure.message,
  }) : ""

  const overlay = phase === "playing" ? null : phase === "paused" ? (
    <><strong>{t("flappy.paused")}</strong><span>{t("flappy.pausedHint")}</span></>
  ) : phase === "over" ? (
    <>
      <strong>{t("flappy.over")}</strong>
      <span className="fb-final">{t("flappy.finalScore", { score: hud.score })}{hud.newBest ? <b>{t("flappy.newBest")}</b> : null}</span>
      <span>{isModel ? (settings.autoRestart ? t("flappy.nextRound") : t("flappy.pressStart")) : t("flappy.again")}</span>
    </>
  ) : (
    <>
      <strong className="fb-title">FlappyBri</strong>
      <span>{isModel ? (preparing ? t("flappy.checking") : t("flappy.readyModel")) : t("flappy.readyHuman")}</span>
    </>
  )

  return (
    <div className="fb-page">
      <div className="fb-bar">
        <div className="fb-toolbar">
          <div className="fb-modes" role="radiogroup" aria-label={t("flappy.mode")}>
            {(["human", "model"] as const).map((mode) => (
              <button key={mode} type="button" role="radio" aria-checked={settings.mode === mode}
                      onClick={() => chooseMode(mode)}>
                {mode === "human" ? <Bird /> : <Cpu />}
                {t(mode === "human" ? "flappy.youPlay" : "flappy.modelPlays")}
              </button>
            ))}
          </div>
          <div className="fb-actions">
            <button type="button" className="fb-primary" disabled={isModel && preparing && phase !== "playing"}
                    onClick={() => { primary.run(); if (phase !== "playing") focusStage() }}>
              {isModel && preparing && phase !== "playing" ? <LoaderCircle className="fb-spin" /> : <primary.Icon />}
              {primary.label}
            </button>
            <button type="button" className="fb-secondary" title={t("flappy.restart")} aria-label={t("flappy.restart")}
                    onClick={restart}><RotateCcw /></button>
          </div>
        </div>

        {failure ? (
          <div className="fb-failure" role="alert">
            <p><strong>{failureText}</strong> {t("flappy.fallback")}</p>
            <button type="button" aria-label={t("flappy.dismiss")} title={t("flappy.dismiss")} onClick={() => setFailure(null)}><X /></button>
          </div>
        ) : null}
      </div>

      <div className="fb-main">
        <div ref={stage} className="fb-stage" tabIndex={0} role="group" aria-roledescription={t("flappy.game")}
             aria-label={t("flappy.stageLabel")} aria-describedby="fb-keys"
             onPointerDown={(event) => {
               if (event.pointerType === "mouse" && event.button !== 0) return
               event.preventDefault()
               focusStage()
               act()
             }}>
          <canvas ref={canvas} aria-hidden="true" />
          {overlay ? <div className="fb-overlay" data-phase={phase}>{overlay}</div> : null}
          <span className="fb-live" aria-live="polite">
            {phase === "over" ? t("flappy.liveOver", { score: hud.score, best: hud.best }) : ""}
          </span>
        </div>
        <p className="fb-keys" id="fb-keys">{t("flappy.keys")} <span>{t("flappy.seed", { seed: hud.seed })}</span></p>
      </div>

      <aside className="fb-hud" aria-label={t("flappy.hud")}>
        {isModel ? (
          <section className="fb-card fb-model">
            <div className="fb-model-head">
              <span className={`fb-dot${hud.busy ? " busy" : ""}`} aria-hidden="true" />
              <span className="fb-label">{t("flappy.model")}</span>
              <strong title={modelName || ""}>{decision?.model || modelName || (preparing ? t("flappy.checkingShort") : t("flappy.unknownModel"))}</strong>
            </div>
            <div className="fb-decision">
              <span className="fb-label">{t("flappy.lastDecision")}</span>
              <div className="fb-decision-row">
                <strong data-flap={decision ? String(decision.flap) : undefined}>
                  {decision ? t(decision.flap ? "flappy.flap" : "flappy.glide") : "-"}
                </strong>
                <span>{decision ? t("flappy.pFlap", { p: decision.p.toFixed(3) }) : t("flappy.noDecision")}</span>
              </div>
              <div className="fb-meter" role="img"
                   aria-label={decision ? t("flappy.meterLabel", { p: decision.p.toFixed(3), threshold: settings.threshold.toFixed(2) }) : t("flappy.noDecision")}>
                <span className="fb-meter-fill" data-flap={decision ? String(decision.flap) : undefined}
                      style={{ width: `${(decision?.p ?? 0) * 100}%` }} />
                <span className="fb-meter-mark" style={{ left: `${settings.threshold * 100}%` }} />
              </div>
              {decision ? (
                <p className="fb-sub">
                  {t(decision.stepsLate === 0 ? "flappy.lateNone" : decision.stepsLate === 1 ? "flappy.lateOne" : "flappy.late", { n: decision.stepsLate })}
                  {decision.engineMs !== null ? <> · {t("flappy.engineMs", { ms: fmtMs(decision.engineMs) })}</> : null}
                </p>
              ) : null}
            </div>
          </section>
        ) : null}

        <div className="fb-tiles">
          <div className="fb-tile"><span className="fb-label">{t("flappy.score")}</span><strong>{hud.score}</strong></div>
          <div className="fb-tile"><span className="fb-label">{t(isModel ? "flappy.bestModel" : "flappy.best")}</span><strong>{hud.best}</strong></div>
          {isModel ? <>
            <div className="fb-tile">
              <span className="fb-label">{t("flappy.latency")}</span>
              <strong>{summary ? <>{fmtMs(summary.last)}<small> ms</small></> : "-"}</strong>
              <span className="fb-sub">{summary ? t("flappy.avgP95", { avg: fmtMs(summary.avg), p95: fmtMs(summary.p95) }) : t("flappy.noDecision")}</span>
            </div>
            <div className="fb-tile">
              <span className="fb-label">{t("flappy.perSecond")}</span>
              <strong>{hud.rate !== null ? hud.rate.toFixed(1) : "-"}</strong>
              <span className="fb-sub">{stepsPerDecision !== null ? t("flappy.stepsPerDecision", { n: stepsPerDecision.toFixed(1) }) : t("flappy.oneInFlight")}</span>
            </div>
          </> : null}
        </div>

        {isModel ? <Sparkline samples={hud.spark} summary={summary} t={t} /> : null}

        <section className="fb-card fb-settings">
          {isModel ? <>
            <div className="fb-field">
              <span className="fb-label" id="fb-form">{t("flappy.question")}</span>
              <div className="fb-segment" role="radiogroup" aria-labelledby="fb-form">
                {(["noul", "choice"] as const).map((form) => (
                  <button key={form} type="button" role="radio" aria-checked={settings.form === form}
                          onClick={() => patch({ form })}>{t(form === "noul" ? "flappy.formNoul" : "flappy.formChoice")}</button>
                ))}
              </div>
              <p className="fb-help">{t(settings.form === "noul" ? "flappy.formNoulHelp" : "flappy.formChoiceHelp")}</p>
            </div>
            <label className="fb-field">
              <span className="fb-line"><span className="fb-label">{t("flappy.threshold")}</span><code>{settings.threshold.toFixed(2)}</code></span>
              <input type="range" min={0.05} max={0.95} step={0.05} value={settings.threshold}
                     onChange={(event) => patch({ threshold: Number(event.target.value) })} />
              <span className="fb-help">{t("flappy.thresholdHelp")}</span>
            </label>
          </> : null}
          <label className="fb-field">
            <span className="fb-line">
              <span className="fb-label">{t("flappy.speed")}</span>
              <code>{fmtSpeed(hud.speed)}x</code>
            </span>
            <input type="range" min={0} max={SPEEDS.length - 1} step={1}
                   value={isModel && settings.match ? closestSpeed(hud.speed) : settings.speed}
                   disabled={isModel && settings.match}
                   aria-valuetext={`${fmtSpeed(hud.speed)}x`}
                   onChange={(event) => patch({ speed: Number(event.target.value) })} />
            <span className="fb-help">
              {hud.effective !== null
                ? t("flappy.effective", { x: fmtSpeed(hud.effective), steps: Math.round(hud.effective * STEPS_PER_SECOND) })
                : t("flappy.speedHelp")}
            </span>
          </label>
          {isModel ? <>
            <Toggle on={settings.match} onChange={(match) => patch({ match })} label={t("flappy.match")}
                    help={t("flappy.matchHelp", { n: STEPS_PER_DECISION })} />
            <Toggle on={settings.autoRestart} onChange={(autoRestart) => patch({ autoRestart })} label={t("flappy.autoRestart")} />
          </> : null}
        </section>

        <p className="fb-note">{t("flappy.note")}</p>

        {isModel ? (
          <details className="fb-card fb-reads">
            <summary>{t("flappy.reads")}</summary>
            {hud.lastState ? <>
              <p className="fb-help">{t("flappy.readsHelp")}</p>
              <pre>{`POST /v1/systemone\n${JSON.stringify({ model: decision?.model || modelName || model, state: hud.lastState, questions: buildQuestions(settings.form) }, null, 2)}`}</pre>
              <p className="fb-help">{t("flappy.readsAnswer", { field: settings.form === "noul" ? `answers.${QUESTION_ID.noul}.noul` : `answers.${QUESTION_ID.choice}.probabilities.flap` })}</p>
            </> : <p className="fb-help">{t("flappy.readsEmpty")}</p>}
          </details>
        ) : null}

        {!connected && isModel ? <p className="fb-help">{t("flappy.notConnected")}</p> : null}
      </aside>
    </div>
  )
}

function closestSpeed(speed: number) {
  let best = 0
  SPEEDS.forEach((value, index) => { if (Math.abs(value - speed) < Math.abs(SPEEDS[best] - speed)) best = index })
  return best
}

function Toggle({ on, onChange, label, help }: { on: boolean; onChange: (on: boolean) => void; label: string; help?: string }) {
  return (
    <div className="fb-field">
      <button type="button" className="fb-toggle" role="switch" aria-checked={on} onClick={() => onChange(!on)}>
        <span>{label}</span><i><b /></i>
      </button>
      {help ? <span className="fb-help">{help}</span> : null}
    </div>
  )
}

/* Latency per decision over the rolling window: one series, so no legend; a
   2 px line, the latest point marked, and the p95 as a hairline. */
function Sparkline({ samples, summary, t }: {
  samples: number[]; summary: LatencySummary | null; t: (key: string, vars?: Record<string, string | number>) => string
}) {
  const W = 300
  const H = 64
  const top = summary ? Math.max(...samples, 1) * 1.15 : 1
  const x = (index: number) => (samples.length < 2 ? W : (index / (SPARK - 1)) * W)
  const offset = SPARK - samples.length
  const y = (value: number) => H - (value / top) * (H - 4) - 2
  const points = samples.map((value, index) => `${x(index + offset).toFixed(1)},${y(value).toFixed(1)}`).join(" ")
  const lastIndex = samples.length - 1
  const min = samples.length ? Math.min(...samples) : 0
  const max = samples.length ? Math.max(...samples) : 0
  return (
    <section className="fb-card fb-spark">
      <div className="fb-line">
        <span className="fb-label">{t("flappy.sparkTitle", { n: SPARK })}</span>
        {summary ? <span className="fb-help">{t("flappy.sparkRange", { min: fmtMs(min), max: fmtMs(max) })}</span> : null}
      </div>
      {samples.length ? (
        <div className="fb-spark-plot" role="img"
             aria-label={t("flappy.sparkLabel", { n: samples.length, min: fmtMs(min), max: fmtMs(max), p95: fmtMs(summary?.p95 ?? 0) })}>
          <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden="true">
            <line className="fb-spark-base" x1="0" x2={W} y1={H - 1} y2={H - 1} vectorEffect="non-scaling-stroke" />
            {summary ? <line className="fb-spark-p95" x1="0" x2={W} y1={y(summary.p95)} y2={y(summary.p95)} vectorEffect="non-scaling-stroke" /> : null}
            {samples.length > 1 ? <polyline points={points} vectorEffect="non-scaling-stroke" /> : null}
          </svg>
          {summary ? <span className="fb-spark-tag" style={{ top: `${(y(summary.p95) / H) * 100}%` }}>p95</span> : null}
          <span className="fb-spark-dot" style={{ left: `${(x(lastIndex + offset) / W) * 100}%`, top: `${(y(samples[lastIndex]) / H) * 100}%` }} />
        </div>
      ) : <p className="fb-help fb-spark-empty">{t("flappy.noDecision")}</p>}
    </section>
  )
}
