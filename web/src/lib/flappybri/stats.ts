/* Latency of the decisions, as the HUD shows it: the last one, the average and
 * the 95th percentile over a rolling window, and how many decisions a second
 * the model is actually delivering. */

export interface LatencySummary {
  last: number
  avg: number
  p95: number
  count: number
}

/* Nearest-rank percentile: the smallest sample with at least q of the samples
   at or below it. Always a value that was measured, never an interpolation. */
export function percentile(values: readonly number[], q: number): number {
  if (!values.length) return NaN
  const sorted = [...values].sort((a, b) => a - b)
  const rank = Math.ceil(Math.min(1, Math.max(0, q)) * sorted.length)
  return sorted[Math.max(0, rank - 1)]
}

export function summarize(samples: readonly number[]): LatencySummary | null {
  if (!samples.length) return null
  const total = samples.reduce((sum, value) => sum + value, 0)
  return {
    last: samples[samples.length - 1],
    avg: total / samples.length,
    p95: percentile(samples, 0.95),
    count: samples.length,
  }
}

/* Decisions per second over a run of completion times (ms): the intervals
   between the first and the last, so a window of n answers holds n - 1 of them.
   Null until there are two. */
export function perSecond(times: readonly number[]): number | null {
  if (times.length < 2) return null
  const span = times[times.length - 1] - times[0]
  return span > 0 ? ((times.length - 1) * 1000) / span : null
}

export class LatencyWindow {
  readonly size: number
  private latencies: number[] = []
  private times: number[] = []

  constructor(size = 100) {
    this.size = Math.max(1, Math.floor(size))
  }

  push(latencyMs: number, at: number) {
    this.latencies.push(latencyMs)
    this.times.push(at)
    if (this.latencies.length > this.size) this.latencies.shift()
    if (this.times.length > this.size) this.times.shift()
  }

  /* The rate counts only continuous play: after a pause, a game over or a new
     round, the wait in between is not the model being slow. The latencies are
     kept, they belong to the model. */
  breakRun() {
    this.times = []
  }

  clear() {
    this.latencies = []
    this.times = []
  }

  samples(): readonly number[] {
    return this.latencies
  }

  summary(): LatencySummary | null {
    return summarize(this.latencies)
  }

  rate(): number | null {
    return perSecond(this.times)
  }
}
