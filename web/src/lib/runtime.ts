import type { HealthResponse } from "./api"

export function activeRequests(health: HealthResponse | null): number {
  return Number(health?.scheduler?.active || 0)
}

export function supportsCacheSlots(health: HealthResponse | null): boolean {
  return typeof health?.kv_slots === "number" && health.kv_slots > 0
}
