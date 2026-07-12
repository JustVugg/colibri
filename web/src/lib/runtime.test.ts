import { describe, expect, it } from "vitest"

import type { HealthResponse } from "./api"
import { activeRequests, supportsCacheSlots } from "./runtime"

const healthWithActive = (active: boolean | number): HealthResponse => ({
  status: "ok",
  scheduler: {
    active,
    queued: 0,
    max_queue: 0,
    queue_timeout_seconds: 0,
    admitted: 0,
    completed: 0,
    rejected: 0,
    timed_out: 0,
    cancelled: 0,
  },
})

describe("runtime capability normalization", () => {
  it.each([
    [true, 1],
    [false, 0],
    [3, 3],
    [0, 0],
  ] as const)("normalizes scheduler.active %s to %d", (active, expected) => {
    expect(activeRequests(healthWithActive(active))).toBe(expected)
  })

  it("treats missing scheduler metrics as idle", () => {
    expect(activeRequests({ status: "ok" })).toBe(0)
    expect(activeRequests(null)).toBe(0)
  })

  it("only enables cache slots when the health response advertises them", () => {
    expect(supportsCacheSlots({ status: "ok", kv_slots: 4 })).toBe(true)
    expect(supportsCacheSlots({ status: "ok" })).toBe(false)
    expect(supportsCacheSlots(null)).toBe(false)
  })
})
