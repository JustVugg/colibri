export interface StringStorage {
  getItem(key: string): string | null
  setItem(key: string, value: string): void
  removeItem(key: string): void
}

export function stored(storage: Pick<StringStorage, "getItem">, key: string, fallback: string) {
  try { return storage.getItem(key) || fallback } catch { return fallback }
}

export function persistPublicSettings(storage: StringStorage, baseUrl: string, model: string) {
  // Cleanup must not depend on preference writes: a full quota can reject
  // setItem while still allowing the legacy entry to be removed.
  try { storage.removeItem("colibri.apiKey") } catch { /* restricted storage mode */ }
  try {
    storage.setItem("colibri.baseUrl", baseUrl)
    storage.setItem("colibri.model", model)
  } catch { /* restricted storage mode */ }
}
