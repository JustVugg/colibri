import type { HealthResponse } from "./api"

export function activeRequests(health: HealthResponse | null): number {
  return Number(health?.scheduler?.active || 0)
}

export function supportsCacheSlots(health: HealthResponse | null): boolean {
  return typeof health?.kv_slots === "number" && health.kv_slots > 0
}

/* A colibrì server that summarizes a conversation outgrowing the context when asked. */
export function supportsCompaction(health: HealthResponse | null): boolean {
  return health?.context_compaction === true
}

/* Only an explicit true: a server that predates the field answers a trailing
   assistant turn fresh, so a Continue button there would mislabel a re-answer. */
export function supportsContinuation(health: HealthResponse | null): boolean {
  return health?.continue_assistant === true
}
