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

it.each(["colibri.baseUrl", "colibri.model"])("cleans a legacy credential even when writing %s exceeds quota", (failedKey) => {
  const storage = memoryStorage({ "colibri.apiKey": "synthetic-legacy-value", "colibri.theme": "dark" });
  const set = storage.setItem;
  storage.setItem = (key, value) => {
    if (key === failedKey) throw new DOMException("fixture quota reached", "QuotaExceededError");
    set(key, value);
  };
  persistPublicSettings(storage, "https://fixture.invalid/v1", "fixture-model");
  expect(storage.values.has("colibri.apiKey")).toBe(false);
  expect(storage.values.get("colibri.theme")).toBe("dark");
});

it("preserves public preferences when legacy cleanup is unavailable", () => {
  const storage = memoryStorage();
  storage.removeItem = () => { throw new Error("fixture removal blocked") };
  expect(() => persistPublicSettings(storage, "https://fixture.invalid/v1", "fixture-model")).not.toThrow();
  expect(storage.values.get("colibri.baseUrl")).toBe("https://fixture.invalid/v1");
  expect(storage.values.get("colibri.model")).toBe("fixture-model");
});
