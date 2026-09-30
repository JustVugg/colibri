import { useEffect, useState, type ReactNode } from "react"
import { ChevronDown, Copy, Dices, Download, Footprints, Proportions, RefreshCw, RotateCcw, X } from "lucide-react"

import type { GeneratedImage, ImageProgress } from "@/lib/api"
import {
  MAX_STEPS, MIN_STEPS, SIZE_PRESETS, clampSteps, duration, elapsedClock, presetOf, progressFraction,
  sizeKey, snapSide, validSide, type ImageSize,
} from "@/lib/images"
import { cn } from "@/lib/utils"
import { useLocale } from "../i18n"

/* The state of the one picture being generated. Kept out of the transcript:
   it changes on every step, and only the finished picture belongs there. */
export interface ImageRun {
  targetId: string
  started: number
  progress: ImageProgress | null
  preview: string | null
  /* No stream: one plain request, so no stage will ever be reported. */
  plain?: boolean
}

/* The room a picture takes, sized from the request before a single pixel
   exists, so the placeholder, the previews and the result sit in the same
   place and the conversation does not jump when the image arrives. Never wider
   than the column or the picture itself, never taller than most of the screen. */
const turnWidth = (width: number, height: number) => ({
  width: `min(100%, ${width}px, calc(62dvh * ${(width / height).toFixed(4)}))`,
})

function Frame({ width, height, className, children }: { width: number; height: number; className?: string; children: ReactNode }) {
  return <div className={cn("image-frame", className)} style={{ aspectRatio: `${width} / ${height}` }}>{children}</div>
}

function Elapsed({ since }: { since: number }) {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 250)
    return () => window.clearInterval(timer)
  }, [])
  return <time className="image-clock">{elapsedClock((now - since) / 1000)}</time>
}

export function ImageProgressCard({ request, run, onCancel }: { request: GeneratedImage; run: ImageRun; onCancel: () => void }) {
  const { t } = useLocale()
  const progress = run.progress
  const fraction = progressFraction(progress)
  const stage = !progress ? t(run.plain ? "image.stage.working" : "image.stage.queued")
    : progress.stage === "denoise" ? t("image.stage.denoise", { step: Math.min(progress.steps, progress.step + 1), steps: progress.steps })
    : progress.stage === "encode" || progress.stage === "decode" ? t(`image.stage.${progress.stage}`)
    : t("image.stage.working")
  return <div className="image-turn" style={turnWidth(request.width, request.height)}>
    <Frame width={request.width} height={request.height} className="pending">
      {run.preview ? <img className="image-preview" src={run.preview} alt={t("image.previewAlt")} /> : <span className="image-shimmer" />}
    </Frame>
    <div className="image-status">
      <span className="image-stage" aria-live="polite">{stage}</span>
      <Elapsed since={run.started} />
      <button type="button" className="image-cancel" onClick={onCancel}><X />{t("image.cancel")}</button>
    </div>
    <div className={cn("image-bar", !progress && "waiting")} role="progressbar" aria-label={t("image.progress")}
      aria-valuemin={0} aria-valuemax={100} aria-valuenow={progress ? Math.round(fraction * 100) : undefined}>
      <span style={{ width: `${(fraction * 100).toFixed(1)}%` }} />
    </div>
  </div>
}

export function ImageResultCard({ image, copied, busy, onDownload, onCopySeed, onAgain }: {
  image: GeneratedImage
  copied: boolean
  busy: boolean
  onDownload: () => void
  onCopySeed: () => void
  onAgain: () => void
}) {
  const { t, locale } = useLocale()
  const timings = image.timings
  const breakdown = timings && [timings.encode, timings.denoise, timings.decode].every((value) => typeof value === "number")
    ? t("image.timings", {
      encode: duration(timings.encode!, locale), denoise: duration(timings.denoise!, locale), decode: duration(timings.decode!, locale),
    })
    : undefined
  return <div className="image-turn" style={turnWidth(image.width, image.height)}>
    <Frame width={image.width} height={image.height}>
      <img src={image.url} alt={t("image.alt", { prompt: image.prompt })} />
    </Frame>
    <p className="image-meta">
      <span>{image.width} × {image.height}</span>
      <span>{t("image.meta.steps", { n: image.steps })}</span>
      <span>{t("image.meta.seed", { seed: image.seed })}</span>
      {image.seconds != null && <span title={breakdown}>{duration(image.seconds, locale)}</span>}
    </p>
    <div className="image-actions">
      {/* aria-label as well as the text: on a phone the first two show only their icon */}
      <button type="button" className="image-action" onClick={onDownload} aria-label={t("image.download")} title={t("image.download")}><Download /><span>{t("image.download")}</span></button>
      <button type="button" className="image-action" onClick={onCopySeed} aria-label={t("image.copySeed")} title={t("image.copySeed")}><Copy /><span>{t("image.copySeed")}</span></button>
      <button type="button" className="image-action primary" onClick={onAgain} disabled={busy} aria-label={t("image.again")} title={t("image.again")}><RefreshCw /><span>{t("image.againShort")}</span></button>
      {copied && <span role="status">{t("image.seedCopied")}</span>}
    </div>
  </div>
}

export function ImageStoppedCard({ request, finish, busy, onRetry }: { request: GeneratedImage; finish?: string; busy: boolean; onRetry: () => void }) {
  const { t } = useLocale()
  return <div className="image-turn">
    <div className="image-stopped">
      <span>{finish === "aborted" ? t("image.cancelled") : t("image.failed")}</span>
      <small>{request.width} × {request.height} · {t("image.meta.steps", { n: request.steps })}</small>
      <button type="button" className="image-action" onClick={onRetry} disabled={busy}><RotateCcw /><span>{t("image.retry")}</span></button>
    </div>
  </div>
}

/* The presets grouped by shape: the closed select shows only the pixels,
   the open list says which ratio each one is. */
const RATIOS = [...new Set(SIZE_PRESETS.map((preset) => preset.ratio))]

/* Size, steps and seed, in the composer where the chat's reasoning and slot
   chips sit for a text model. A custom side is typed freely and snapped to a
   multiple of 32 when the field is left; a valid value counts right away, so
   sending straight from the field uses what is on screen. */
export function ImageControls({ size, custom, steps, seed, seedValid, onSize, onCustom, onSteps, onSeed }: {
  size: ImageSize
  custom: boolean
  steps: number
  seed: string
  seedValid: boolean
  onSize: (size: ImageSize) => void
  onCustom: (custom: boolean) => void
  onSteps: (steps: number) => void
  onSeed: (seed: string) => void
}) {
  const { t } = useLocale()
  const [width, setWidth] = useState(String(size.width))
  const [height, setHeight] = useState(String(size.height))
  const [stepsText, setStepsText] = useState(String(steps))
  useEffect(() => { setWidth(String(size.width)); setHeight(String(size.height)) }, [size.width, size.height])
  useEffect(() => { setStepsText(String(steps)) }, [steps])

  const side = (axis: "width" | "height", text: string, commit: boolean) => {
    const value = Number(text)
    const next = commit ? snapSide(value) : value
    if (commit) (axis === "width" ? setWidth : setHeight)(String(next))
    if (commit || validSide(value)) onSize({ ...size, [axis]: next })
  }
  const selected = custom || !presetOf(size) ? "custom" : sizeKey(size)
  return <div className="image-controls">
    <label className="image-chip" title={t("image.size")}>
      <Proportions aria-hidden="true" />
      <select aria-label={t("image.size")} value={selected} onChange={(event) => {
        if (event.target.value === "custom") { onCustom(true); return }
        const preset = SIZE_PRESETS.find((item) => sizeKey(item) === event.target.value)
        if (preset) { onCustom(false); onSize({ width: preset.width, height: preset.height }) }
      }}>
        {RATIOS.map((ratio) => <optgroup key={ratio} label={ratio}>
          {SIZE_PRESETS.filter((preset) => preset.ratio === ratio).map((preset) =>
            <option key={sizeKey(preset)} value={sizeKey(preset)}>{preset.width}×{preset.height}</option>)}
        </optgroup>)}
        <option value="custom">{t("image.sizeCustom")}</option>
      </select>
      <ChevronDown className="image-caret" aria-hidden="true" />
    </label>
    {selected === "custom" && <span className="image-chip image-custom" title={t("image.sizeHelp")}>
      <input aria-label={t("image.width")} inputMode="numeric" value={width}
        className={cn(!validSide(Number(width)) && "invalid")}
        onChange={(event) => { setWidth(event.target.value); side("width", event.target.value, false) }}
        onBlur={(event) => side("width", event.target.value, true)}
        onKeyDown={(event) => { if (event.key === "Enter") side("width", event.currentTarget.value, true) }} />
      <span aria-hidden="true">×</span>
      <input aria-label={t("image.height")} inputMode="numeric" value={height}
        className={cn(!validSide(Number(height)) && "invalid")}
        onChange={(event) => { setHeight(event.target.value); side("height", event.target.value, false) }}
        onBlur={(event) => side("height", event.target.value, true)}
        onKeyDown={(event) => { if (event.key === "Enter") side("height", event.currentTarget.value, true) }} />
    </span>}
    <label className="image-chip" title={t("image.steps")}>
      <Footprints aria-hidden="true" />
      <input aria-label={t("image.steps")} inputMode="numeric" className="image-steps" value={stepsText}
        onChange={(event) => {
          setStepsText(event.target.value)
          const value = Number(event.target.value)
          if (event.target.value && Number.isInteger(value) && value >= MIN_STEPS && value <= MAX_STEPS) onSteps(value)
        }}
        onBlur={(event) => { const value = clampSteps(Number(event.target.value || steps)); setStepsText(String(value)); onSteps(value) }}
        onKeyDown={(event) => {
          /* Enter submits the form without a blur first: commit what is on screen */
          if (event.key !== "Enter") return
          const value = clampSteps(Number(event.currentTarget.value || steps)); setStepsText(String(value)); onSteps(value)
        }} />
      <span className="image-unit">{t("image.stepsUnit")}</span>
    </label>
    <label className={cn("image-chip", !seedValid && "invalid")} title={t("image.seedHelp")}>
      <Dices aria-hidden="true" />
      <input aria-label={t("image.seed")} aria-invalid={!seedValid} inputMode="numeric" className="image-seed" value={seed}
        placeholder={t("image.seedRandom")} onChange={(event) => onSeed(event.target.value)} />
      {seed && <button type="button" className="image-clear" aria-label={t("image.seedClear")} title={t("image.seedClear")} onClick={() => onSeed("")}><X /></button>}
    </label>
  </div>
}
