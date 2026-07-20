import { createContext, useContext, useState, useCallback, useMemo, type ReactNode } from "react"
import { createElement } from "react"
import en from "./en"
import zhCN from "./zh-CN"
import zhTW from "./zh-TW"
import it from "./it"

const LOCALES = [
  { code: "en", label: "English" },
  { code: "zh-CN", label: "简体中文" },
  { code: "zh-TW", label: "繁體中文" },
  { code: "it", label: "Italiano" },
] as const

const DICTS: Record<string, Record<string, string>> = {
  "en": en,
  "zh-CN": zhCN,
  "zh-TW": zhTW,
  "it": it,
}

const STORAGE_KEY = "colibri-locale"

function detectLocale(): string {
  try {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved && DICTS[saved]) return saved
  } catch {}
  const nav = navigator.language || ""
  if (DICTS[nav]) return nav
  const prefix = nav.split("-")[0]
  if (prefix === "zh") return nav.includes("TW") || nav.includes("Hant") ? "zh-TW" : "zh-CN"
  for (const { code } of LOCALES) if (code.startsWith(prefix)) return code
  return "en"
}

function interpolate(template: string, vars?: Record<string, string | number>): string {
  if (!vars) return template
  return template.replace(/\{\{(\w+)\}\}/g, (_, key) => String(vars[key] ?? `{{${key}}}`))
}

interface LocaleContext {
  locale: string
  setLocale: (code: string) => void
  t: (key: string, vars?: Record<string, string | number>) => string
  locales: readonly { code: string; label: string }[]
}

const Ctx = createContext<LocaleContext>({
  locale: "en",
  setLocale: () => {},
  t: (key) => key,
  locales: LOCALES,
})

export function LocaleProvider({ children }: { children: ReactNode }) {
  const [locale, setLocaleState] = useState(detectLocale)

  const setLocale = useCallback((code: string) => {
    if (!DICTS[code]) return
    setLocaleState(code)
    try { localStorage.setItem(STORAGE_KEY, code) } catch {}
  }, [])

  const t = useCallback((key: string, vars?: Record<string, string | number>) => {
    const dict = DICTS[locale] || en
    const template = dict[key] ?? en[key] ?? key
    return interpolate(template, vars)
  }, [locale])

  const value = useMemo(() => ({ locale, setLocale, t, locales: LOCALES }), [locale, setLocale, t])

  return createElement(Ctx.Provider, { value }, children)
}

export function useLocale() {
  return useContext(Ctx)
}
