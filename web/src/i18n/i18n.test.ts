import { describe, expect, it } from "vitest"

import en from "./en"
import zhCN from "./zh-CN"
import zhTW from "./zh-TW"
import de from "./de"
import itDict from "./it"
import id from "./id"

/* The `t` interpolator only understands the double-brace form, `{{name}}`
 * (see interpolate() in index.ts). A single-brace `{name}` is left verbatim,
 * so it reaches the DOM as literal text — e.g. an image `alt` reading
 * "Attached image {n}". Keep every placeholder in the {{name}} form. */
const DICTS: Record<string, Record<string, string>> = {
  en, "zh-CN": zhCN, "zh-TW": zhTW, de, it: itDict, id,
}

describe("locale dictionaries", () => {
  for (const [code, dict] of Object.entries(DICTS)) {
    it(`${code} uses the {{name}} placeholder form only`, () => {
      const bad: string[] = []
      for (const [key, value] of Object.entries(dict)) {
        for (const run of value.match(/\{+\w+\}+/g) ?? []) {
          if (!run.startsWith("{{") || !run.endsWith("}}")) bad.push(`${key}: ${value}`)
        }
      }
      expect(bad).toEqual([])
    })
  }

  it("every non-English key also exists in English so the fallback resolves", () => {
    const missing: string[] = []
    for (const [code, dict] of Object.entries(DICTS)) {
      if (code === "en") continue
      for (const key of Object.keys(dict)) if (!(key in en)) missing.push(`${code}: ${key}`)
    }
    expect(missing).toEqual([])
  })

  /* FlappyBri is translated in full in every locale, and its name is not. */
  it("carries every FlappyBri string in every locale, under its own name", () => {
    const keys = Object.keys(en).filter((key) => key.startsWith("flappy.") || key === "nav.flappybri")
    expect(keys.length).toBeGreaterThan(50)
    for (const [code, dict] of Object.entries(DICTS)) {
      expect(keys.filter((key) => !(key in dict)), code).toEqual([])
      expect(dict["nav.flappybri"], code).toBe("FlappyBri")
    }
  })
})
