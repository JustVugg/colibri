import { describe, expect, it, vi } from "vitest"

import { persistPublicSettings, stored, type StringStorage } from "./storage"

function memoryStorage(initial: Record<string, string> = {}): StringStorage & { values: Map<string, string> } {
  const values = new Map(Object.entries(initial))
  return {
    values,
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => { values.set(key, value) },
    removeItem: (key) => { values.delete(key) },
  }
}

describe("browser settings persistence", () => {
  it("persists endpoint and model but removes legacy API credentials", () => {
    const storage = memoryStorage({ "colibri.apiKey": "legacy-secret" })
    persistPublicSettings(storage, "https://localhost/v1", "test-model")
    expect(Object.fromEntries(storage.values)).toEqual({
      "colibri.baseUrl": "https://localhost/v1",
      "colibri.model": "test-model",
    })
  })

  it("does not attempt to write an API key", () => {
    const storage = memoryStorage()
    const setItem = vi.spyOn(storage, "setItem")
    persistPublicSettings(storage, "http://localhost/v1", "test-model")
    expect(setItem).not.toHaveBeenCalledWith("colibri.apiKey", expect.anything())
  })

  it("uses a fallback when storage access is unavailable", () => {
    expect(stored({ getItem: () => { throw new Error("denied") } }, "key", "fallback")).toBe("fallback")
  })
})
