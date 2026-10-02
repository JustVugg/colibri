import { describe, expect, it } from "vitest"

import { LatencyWindow, percentile, perSecond, summarize } from "./stats"

describe("latency statistics", () => {
  it("takes the nearest-rank percentile, a value that was measured", () => {
    const hundred = Array.from({ length: 100 }, (_, i) => i + 1)
    expect(percentile(hundred, 0.95)).toBe(95)
    expect(percentile(hundred, 0.5)).toBe(50)
    expect(percentile([30, 10, 20], 0.95)).toBe(30)
    expect(percentile([7], 0.95)).toBe(7)
    expect(percentile([], 0.95)).toBeNaN()
    /* twenty samples: rank ceil(19) = 19, the second largest */
    expect(percentile(Array.from({ length: 20 }, (_, i) => (i + 1) * 10), 0.95)).toBe(190)
  })

  it("summarizes the last, the average and the p95", () => {
    expect(summarize([])).toBeNull()
    expect(summarize([40, 10, 30, 20])).toEqual({ last: 20, avg: 25, p95: 40, count: 4 })
  })

  it("counts decisions per second over the intervals between them", () => {
    expect(perSecond([])).toBeNull()
    expect(perSecond([100])).toBeNull()
    expect(perSecond([0, 50, 100, 150, 200])).toBe(20)
    expect(perSecond([5, 5])).toBeNull()
  })

  it("keeps a rolling window of the last N", () => {
    const window = new LatencyWindow(3)
    window.push(100, 0)
    window.push(200, 200)
    window.push(300, 500)
    window.push(400, 900)
    expect(window.samples()).toEqual([200, 300, 400])
    expect(window.summary()).toEqual({ last: 400, avg: 300, p95: 400, count: 3 })
    /* three completions at 200, 500 and 900 ms: two intervals in 700 ms */
    expect(window.rate()).toBeCloseTo(2000 / 700)
  })

  it("restarts the rate after a break but keeps the latencies", () => {
    const window = new LatencyWindow(10)
    window.push(50, 0)
    window.push(50, 50)
    window.breakRun()
    expect(window.rate()).toBeNull()
    expect(window.summary()?.count).toBe(2)
    window.push(60, 5000)
    window.push(60, 5100)
    expect(window.rate()).toBe(10)
    window.clear()
    expect(window.summary()).toBeNull()
  })
})
