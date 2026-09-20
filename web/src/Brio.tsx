import { useRef, useState } from "react"
import { FileUp, ListChecks, LoaderCircle, Plus, X } from "lucide-react"

import { askBrio, type BrioResponse } from "@/lib/api"
import { useLocale } from "./i18n"

/* Modalita brio.
 *
 * Lo stesso modello con cui si chatta smette di scrivere: gli si da un insieme
 * chiuso di opzioni e lui dice quanto e probabile ciascuna.
 *
 * PERCHE' LE DOMANDE SONO PIU D'UNA. E' la forma in cui questa modalita rende
 * davvero: il documento si macina UNA volta e resta fotografato nel motore,
 * poi ogni domanda paga solo i propri token. Misurato su qwen36: la prima
 * domanda ~60 s, le successive pochi secondi. Una domanda sola non lo mostra,
 * e le domande si mandano in fila apposta -- cosi i tempi per domanda si
 * vedono uno per uno mentre arrivano.
 *
 * Le altre due cose in evidenza sono quelle che una risposta scritta non puo
 * dare: l'ENTROPIA (quanto il modello sa di sapere) e i TOKEN GENERATI, zero. */

const LEVEL = (h: number) => (h < 0.4 ? "sure" : h < 0.8 ? "unsure" : "unknown")
const lines = (text: string) => text.split("\n").map((o) => o.trim()).filter(Boolean)

interface Row {
  id: number
  text: string
  options: string          /* vuoto = usa le opzioni comuni */
  result?: BrioResponse
  seconds?: number
  error?: string
}

let nextId = 1
const blank = (): Row => ({ id: nextId++, text: "", options: "" })
const first = (): Row => ({ id: nextId++, text: "", options: "yes\nno" })

export default function Brio({ baseUrl, apiKey, model, connected }: {
  baseUrl: string; apiKey: string; model: string; connected: boolean
}) {
  const { t } = useLocale()
  const [state, setState] = useState("")
  const [source, setSource] = useState("")
  const [rows, setRows] = useState<Row[]>([first()])
  const [running, setRunning] = useState<number | null>(null)
  const file = useRef<HTMLInputElement>(null)

  /* Ogni domanda porta le SUE opzioni: "quanto e rischioso" vuole
   * basso/medio/alto, "si firma" vuole si/no, e un insieme unico per tutte
   * costringerebbe a piegare le domande alla lista invece del contrario. */
  const asked = rows.filter((row) => row.text.trim() && lines(row.options).length >= 2)
  const ready = connected && running === null && asked.length > 0

  const patch = (id: number, change: Partial<Row>) =>
    setRows((all) => all.map((row) => (row.id === id ? { ...row, ...change } : row)))

  const load = async (chosen: File | undefined) => {
    if (!chosen) return
    /* Solo testo: un PDF o un .docx qui arriverebbe come byte illeggibili e il
     * modello leggerebbe spazzatura senza che nessuno se ne accorga. Meglio
     * dirlo che accettarlo e sbagliare in silenzio. */
    const text = await chosen.text()
    if (text.includes("\u0000")) { setSource(t("brio.fileBinary")); return }
    setState(text)
    setSource(`${chosen.name} · ${(chosen.size / 1024).toFixed(1)} kB`)
  }

  const run = async () => {
    setRows((all) => all.map((row) => ({ ...row, result: undefined, error: undefined, seconds: undefined })))
    for (const row of asked) {
      setRunning(row.id)
      const started = performance.now()
      try {
        const result = await askBrio(baseUrl, apiKey, model, state, row.text, lines(row.options))
        patch(row.id, { result, seconds: (performance.now() - started) / 1000 })
      } catch (cause) {
        patch(row.id, { error: cause instanceof Error ? cause.message : String(cause) })
      }
    }
    setRunning(null)
  }

  return (
    <div className="brio-page">
      <section className="brio-card brio-doc">
        <div className="brio-doc-head">
          <h2>{t("brio.document")}</h2>
          <button type="button" onClick={() => file.current?.click()}>
            <FileUp />{t("brio.upload")}
          </button>
          <input ref={file} type="file" hidden accept=".txt,.md,.markdown,.json,.csv,.tsv,.log,.yml,.yaml,.xml,.html,text/*"
                 onChange={(e) => { void load(e.target.files?.[0]); e.target.value = "" }} />
        </div>
        <textarea rows={8} value={state} onChange={(e) => { setState(e.target.value); setSource("") }}
                  placeholder={t("brio.documentPlaceholder")} />
        <p className="brio-meta">
          {source ? <b>{source} · </b> : null}
          {state.length.toLocaleString()} {t("brio.chars")} · {t("brio.onceHint")}
        </p>
      </section>

      <div className="brio-questions">
        {rows.map((row, index) => {
          const own = lines(row.options)
          const level = row.result ? LEVEL(row.result.entropy) : "sure"
          return (
            <section className="brio-card brio-q" key={row.id}>
              <div className="brio-q-head">
                <span className="brio-num">{index + 1}</span>
                <input value={row.text} onChange={(e) => patch(row.id, { text: e.target.value })}
                       placeholder={t("brio.questionPlaceholder")} />
                {rows.length > 1 ? (
                  <button type="button" className="brio-drop" title={t("brio.removeQuestion")}
                          onClick={() => setRows((all) => all.filter((r) => r.id !== row.id))}><X /></button>
                ) : null}
              </div>
              <textarea className="brio-own" rows={own.length > 2 ? own.length : 2} value={row.options}
                        onChange={(e) => patch(row.id, { options: e.target.value })}
                        placeholder={t("brio.ownOptions")} />
              {own.length ? (
                <div className="brio-chips">
                  {own.map((option) => (
                    <button type="button" key={option} title={t("brio.removeOption")}
                            onClick={() => patch(row.id, { options: own.filter((o) => o !== option).join("\n") })}>
                      {option}<X />
                    </button>
                  ))}
                </div>
              ) : null}
              {own.length === 1 ? <p className="brio-warn">{t("brio.needTwo")}</p> : null}

              {running === row.id ? (
                <p className="brio-running"><LoaderCircle className="brio-spin" />{t("brio.scoring")}</p>
              ) : row.error ? (
                <p className="brio-error">{row.error}</p>
              ) : row.result ? (
                <div className="brio-answer">
                  <div className="brio-bars">
                    {row.result.choices.map((choice) => (
                      <div className={`brio-bar${choice.option === row.result!.answer ? " top" : ""}`}
                           key={choice.option}>
                        <span className="brio-name">{choice.option}</span>
                        <span className="brio-track">
                          <span className="brio-fill" style={{ width: `${Math.max(1.5, choice.p * 100)}%` }} />
                        </span>
                        <b className="brio-pct">{(choice.p * 100).toFixed(1)}<i>%</i></b>
                      </div>
                    ))}
                  </div>
                  <div className="brio-foot" data-level={level}>
                    <strong>{row.result.answer}</strong>
                    <span>{t("brio.entropy")} {row.result.entropy.toFixed(3)} · {t(`brio.${level}`)}</span>
                    <em>{row.seconds?.toFixed(1)}s · {row.result.usage.read_tokens} {t("brio.readTokens")} ·{" "}
                      <b>{row.result.usage.completion_tokens} {t("brio.generatedTokens")}</b></em>
                  </div>
                </div>
              ) : null}
            </section>
          )
        })}

        <div className="brio-actions">
          <button type="button" className="brio-add" onClick={() => setRows((all) => [...all, blank()])}>
            <Plus />{t("brio.addQuestion")}
          </button>
          <button type="button" className="brio-run" disabled={!ready} onClick={run}>
            {running !== null ? <LoaderCircle className="brio-spin" /> : <ListChecks />}
            {running !== null ? t("brio.scoring") : t("brio.scoreAll", { n: asked.length })}
          </button>
        </div>
        {!connected ? <p className="brio-note">{t("brio.offline")}</p> : null}
      </div>
    </div>
  )
}
