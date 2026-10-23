import { useEffect, useRef, useState } from "react"
import { BrainCircuit, Gauge, ListChecks, MessageSquareText } from "lucide-react"
import { useLocale } from "../i18n"

export type View = "chat" | "brio" | "brain" | "profiling" | "settings"

export function NavigationDock({ view, onNavigate, loading }: { view: View; onNavigate: (view: View) => void; loading: boolean }) {
  const { t } = useLocale()
  const [open, setOpen] = useState(false)
  const wrap = useRef<HTMLDivElement>(null)
  const handle = useRef<HTMLButtonElement>(null)
  const timer = useRef<ReturnType<typeof setTimeout>>()
  const gesture = useRef<number | null>(null)
  const swipe = useRef(false)
  const pointerKind = useRef("mouse")
  const close = (restore = false) => {
    clearTimeout(timer.current)
    if (restore || wrap.current?.contains(document.activeElement)) handle.current?.focus()
    setOpen(false)
  }
  useEffect(() => {
    const outside = (e: PointerEvent) => { if (!wrap.current?.contains(e.target as Node)) close() }
    const escape = (e: KeyboardEvent) => { if (e.key === "Escape") close() }
    document.addEventListener("pointerdown", outside)
    document.addEventListener("keydown", escape)
    return () => { clearTimeout(timer.current); document.removeEventListener("pointerdown", outside); document.removeEventListener("keydown", escape) }
  }, [])
  const reveal = () => { clearTimeout(timer.current); setOpen(true) }
  const hideLater = () => {
    clearTimeout(timer.current)
    timer.current = setTimeout(() => { if (!wrap.current?.contains(document.activeElement)) close() }, 850)
  }
  return <div ref={wrap} className={`navigation-dock ${open ? "open" : ""}`} onPointerEnter={e => { if (e.pointerType === "mouse") reveal() }} onPointerLeave={hideLater} onBlur={hideLater}>
    <nav id="workspace-navigation" aria-label={t("ui.navigation")} hidden={!open}>
      {([{ id: "chat", Icon: MessageSquareText }, { id: "brio", Icon: ListChecks }, { id: "brain", Icon: BrainCircuit }, { id: "profiling", Icon: Gauge }] as const).map(({ id, Icon }) =>
        <button key={id} aria-current={view === id ? "page" : undefined} onClick={() => { onNavigate(id); close(true) }}><Icon /><span>{t(`nav.${id}`)}</span>{id === "chat" && loading && <i className="dock-busy" />}</button>)}
    </nav>
    <button ref={handle} className="dock-handle" aria-label={t("ui.navigation")} aria-expanded={open} aria-controls="workspace-navigation"
      onClick={e => { if (swipe.current && e.detail !== 0) { swipe.current = false; return }; if (e.detail === 0) { reveal(); requestAnimationFrame(() => wrap.current?.querySelector<HTMLButtonElement>("nav button")?.focus()) } else if (pointerKind.current === "mouse") reveal(); else setOpen(value => !value) }}
      onPointerDown={e => { pointerKind.current = e.pointerType; gesture.current = e.clientY; swipe.current = false; e.currentTarget.setPointerCapture(e.pointerId) }}
      onPointerMove={e => { if (e.pointerType === "mouse" && gesture.current === null) reveal(); if (gesture.current !== null && Math.abs(e.clientY - gesture.current) > 15) { swipe.current = true; setOpen(e.clientY > gesture.current) } }}
      onPointerUp={e => { gesture.current = null; if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId) }}
      onPointerCancel={() => { gesture.current = null; swipe.current = false }}><span /></button>
  </div>
}
