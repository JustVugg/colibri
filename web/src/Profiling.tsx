import { useEffect, useState } from "react"
import { Activity, Gauge, HardDrive, Timer } from "lucide-react"

import { getProfile, type ProfileTurn } from "@/lib/api"
import { useLocale } from "./i18n"

const PHASES = [
  { key: "expert_wait_s", i18n: "profile.ioWait", color: "#3987e5" },
  { key: "expert_matmul_s", i18n: "profile.expertMatmul", color: "#199e70" },
  { key: "attention_s", i18n: "profile.attention", color: "#c98500" },
  { key: "lm_head_s", i18n: "profile.lmHead", color: "#008300" },
  { key: "other_s", i18n: "profile.other", color: "#9085e9" },
] as const

interface Turn extends ProfileTurn { other_s: number; toks: number }

const derive = (turn: ProfileTurn): Turn => ({
  ...turn,
  other_s: Math.max(0, turn.wall_s - turn.expert_wait_s - turn.expert_matmul_s - turn.attention_s - turn.lm_head_s),
  toks: turn.wall_s > 0 ? turn.completion_tokens / turn.wall_s : 0,
})

const seconds = (value: number) => (value >= 10 ? value.toFixed(1) : value.toFixed(2)) + "s"

function ShareBar({ label, turns }: { label: string; turns: Turn[] }) {
  const { t } = useLocale()
  const total = turns.reduce((sum, turn) => sum + turn.wall_s, 0)
  const parts = PHASES.map((phase) => ({ ...phase, name: t(phase.i18n), value: turns.reduce((sum, turn) => sum + turn[phase.key], 0) }))
  return (
    <div className="prof-share">
      <div className="prof-share-head"><span>{label}</span><code>{seconds(total)}</code></div>
      <div className="prof-share-bar" role="img" aria-label={parts.map((part) => `${part.name} ${seconds(part.value)}`).join(", ")}>
        {parts.map((part) => {
          const share = total > 0 ? part.value / total : 0
          return share > 0.001 ? (
            <span key={part.key} style={{ width: `${100 * share}%`, background: part.color }} title={`${part.name} — ${seconds(part.value)} (${(100 * share).toFixed(1)}%)`}>
              {share >= 0.09 ? `${Math.round(100 * share)}%` : ""}
            </span>
          ) : null
        })}
      </div>
    </div>
  )
}

function TurnColumns({ turns, stacked, height, format, footLabel, footLabelOne }: { turns: Turn[]; stacked: boolean; height: number; format: (turn: Turn) => string; footLabel: string; footLabelOne: string }) {
  const [hover, setHover] = useState<number | null>(null)
  const peak = Math.max(...turns.map((turn) => (stacked ? turn.wall_s : turn.toks)), 1e-9)
  const gap = 2
  const width = Math.max(1, (100 - gap * (turns.length - 1)) / turns.length)
  return (
    <div className="prof-plot" onMouseLeave={() => setHover(null)}>
      <svg viewBox={`0 0 100 ${height}`} preserveAspectRatio="none" aria-hidden="true">
        {[0.25, 0.5, 0.75].map((line) => <line key={line} x1="0" x2="100" y1={height * line} y2={height * line} className="prof-grid" />)}
        {turns.map((turn, index) => {
          const x = index * (width + gap)
          if (!stacked) {
            const h = (height * turn.toks) / peak
            return <rect key={index} x={x} y={height - h} width={width} height={h} rx="1" fill="var(--primary)" opacity={hover === null || hover === index ? 1 : 0.45} />
          }
          let y = height
          return PHASES.map((phase) => {
            const h = (height * turn[phase.key]) / peak
            y -= h
            return h > 0.1 ? <rect key={`${index}-${phase.key}`} x={x} y={y + 0.35} width={width} height={Math.max(h - 0.7, 0.35)} fill={phase.color} opacity={hover === null || hover === index ? 1 : 0.45} /> : null
          })
        })}
        {turns.map((_, index) => <rect key={index} x={index * (width + gap) - gap / 2} y="0" width={width + gap} height={height} fill="transparent" onMouseEnter={() => setHover(index)} />)}
      </svg>
      <div className="prof-plot-foot">
        <span>{turns.length > 1 ? footLabel : footLabelOne}</span>
        <code>{hover !== null && turns[hover] ? format(turns[hover]) : `peak ${stacked ? seconds(peak) : peak.toFixed(1) + " tok/s"}`}</code>
      </div>
    </div>
  )
}

export function Profiling({ baseUrl, apiKey, connected }: { baseUrl: string; apiKey: string; connected: boolean }) {
  const { t } = useLocale()
  const [turns, setTurns] = useState<Turn[]>([])

  useEffect(() => {
    if (!connected) return
    let disposed = false
    const poll = async () => {
      if (document.visibilityState === "hidden") return
      try {
        const result = await getProfile(baseUrl, apiKey)
        if (!disposed) setTurns(result.turns.map(derive))
      } catch { /* engine busy or restarting — keep the last snapshot */ }
    }
    void poll()
    const timer = window.setInterval(() => void poll(), 2000)
    return () => { disposed = true; window.clearInterval(timer) }
  }, [baseUrl, apiKey, connected])

  const latest = turns[turns.length - 1]
  const recent = turns.slice(-40)
  const diskService = turns.reduce((sum, turn) => sum + turn.expert_disk_s, 0)

  return (
    <div className="prof-page">
      <div className="prof-head">
        <div className="section-title"><Gauge className="size-4" /> {t("profile.title")}</div>
        <div className="prof-legend">
          {PHASES.map((phase) => <span key={phase.key}><i style={{ background: phase.color }} />{t(phase.i18n)}</span>)}
        </div>
      </div>

      {!latest ? (
        <p className="runtime-unavailable">{connected ? t("profile.empty") : t("profile.connectHint")}</p>
      ) : (
        <>
          <div className="prof-tiles">
            <div><span><Gauge className="size-3" /> {t("profile.lastTurn")}</span><strong>{latest.toks.toFixed(1)}</strong><small>tok/s</small></div>
            <div><span><Timer className="size-3" /> {t("profile.wallTime")}</span><strong>{seconds(latest.wall_s)}</strong><small>{latest.prompt_tokens} → {latest.completion_tokens} tokens</small></div>
            <div><span><Activity className="size-3" /> {t("profile.batching")}</span><strong>{latest.forwards > 0 ? (latest.completion_tokens / latest.forwards).toFixed(2) : "—"}</strong><small>{t("profile.tokensPerForward")}</small></div>
            <div><span><HardDrive className="size-3" /> {t("profile.diskService")}</span><strong>{seconds(latest.expert_disk_s)}</strong><small>{t("profile.overlapped")}</small></div>
          </div>

          <div className="prof-shares">
            <ShareBar label={t("profile.lastTurn")} turns={[latest]} />
            {turns.length > 1 ? <ShareBar label={t("profile.window", { n: turns.length })} turns={turns} /> : null}
          </div>

          <div className="prof-charts">
            <div className="prof-chart">
              <div className="prof-chart-title">{t("profile.throughputTitle")}</div>
              <TurnColumns turns={recent} stacked={false} height={36} footLabel={t("profile.turnsLabel", { n: recent.length })} footLabelOne={t("profile.oneTurn")} format={(turn) => `${turn.toks.toFixed(1)} tok/s · ${turn.completion_tokens} tokens`} />
            </div>
            <div className="prof-chart">
              <div className="prof-chart-title">{t("profile.phaseTitle")}</div>
              <TurnColumns turns={recent} stacked height={36} footLabel={t("profile.turnsLabel", { n: recent.length })} footLabelOne={t("profile.oneTurn")} format={(turn) => `${seconds(turn.wall_s)} · ${PHASES.map((phase) => `${t(phase.i18n)} ${seconds(turn[phase.key])}`).join(" · ")}`} />
            </div>
          </div>

          <div className="prof-table-wrap">
            <table className="prof-table">
              <thead><tr><th>{t("profile.turnCol")}</th><th>{t("profile.tokensCol")}</th><th>tok/s</th><th>{t("profile.wallCol")}</th>{PHASES.map((phase) => <th key={phase.key}><i style={{ background: phase.color }} />{t(phase.i18n)}</th>)}<th>{t("profile.diskService")}</th></tr></thead>
              <tbody>
                {recent.slice().reverse().map((turn, index) => (
                  <tr key={turns.length - index}>
                    <td>{turns.length - index}</td>
                    <td>{turn.prompt_tokens} → {turn.completion_tokens}</td>
                    <td>{turn.toks.toFixed(1)}</td>
                    <td>{seconds(turn.wall_s)}</td>
                    {PHASES.map((phase) => <td key={phase.key}>{seconds(turn[phase.key])}</td>)}
                    <td>{seconds(turn.expert_disk_s)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {diskService > 0 ? <p className="prof-note">{t("profile.diskNote")}</p> : null}
          </div>
        </>
      )}
    </div>
  )
}
