import { afterEach, describe, expect, it, vi } from "vitest"

import { BIRD, PIPES, createWorld, step, type World } from "./game"
import {
  Pilot, QUESTION_ID, RULES, buildQuestions, buildRequest, classify, describeState, findModel,
  flapProbability, observe, shouldFlap, type PilotFailure,
} from "./pilot"
import { HttpError, type SystemOneResponse } from "../api"

afterEach(() => vi.unstubAllGlobals())

/* Replies in the exact shape c/openai_server.py's systemone() writes: `model`,
   `answers` keyed by question id, `usage` with input/output tokens. The extra
   fields a newer gateway adds (`id`, `provider`, `usage.cost`) are included on
   purpose: the client must read through them. */
const noulReply = (p: number): SystemOneResponse & Record<string, unknown> => ({
  id: "so_123", provider: "colibri", model: "qwen36",
  answers: { flap: { type: "noul", noul: p } },
  usage: { input_tokens: 182, output_tokens: 2, cost: 0 } as SystemOneResponse["usage"],
})
const choiceReply = (p: number): SystemOneResponse => ({
  model: "qwen36",
  answers: {
    move: {
      type: "choice", choice: p > 0.5 ? "flap" : "glide",
      probabilities: { flap: p, glide: 1 - p }, confidence: Math.abs(2 * p - 1),
    },
  },
  usage: { input_tokens: 230, output_tokens: 3 },
})

const json = (body: unknown, init: ResponseInit = {}) =>
  new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" }, ...init })

/* A fetch whose answer the test releases when it wants: the model "thinking". */
function deferredFetch() {
  const calls: Array<{ url: string; body: Record<string, unknown>; release: (r: Response) => void; fail: (e: unknown) => void }> = []
  const fetchMock = vi.fn((url: string, init: RequestInit) => new Promise<Response>((resolve, reject) => {
    calls.push({ url, body: JSON.parse(String(init.body)), release: resolve, fail: reject })
    init.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")))
  }))
  vi.stubGlobal("fetch", fetchMock)
  return { fetchMock, calls }
}

const playing = (seed = 42): World => {
  const world = createWorld(seed)
  world.phase = "playing"
  return world
}

const clock = () => {
  let t = 1000
  return { now: () => t, advance: (ms: number) => { t += ms } }
}

describe("the state the model reads", () => {
  it("places the gap, the edges and the ground relative to the hummingbird, up positive", () => {
    const world = playing()
    world.pipes = [{ id: 1, x: BIRD.x + BIRD.radius + 120, gapY: 200, scored: false }]
    world.y = 238          // 38 px lower on screen than the gap's middle
    world.vy = 4           // falling
    const o = observe(world)
    expect(o).toMatchObject({
      gapMiddle: 38, gapTop: 38 + PIPES.gap / 2, gapBottom: 38 - PIPES.gap / 2,
      ahead: 120, inGap: false, speed: -4, gapHeight: PIPES.gap, ground: 238 - 540, top: 238,
    })
    const text = describeState(o)
    expect(text.startsWith(RULES)).toBe(true)
    expect(text).toContain("the hummingbird is 38 px below the middle of the next gap and falling at 4.0 px per frame")
    expect(text).toContain("The next gap starts 120 px ahead and is 150 px tall")
    expect(text).toContain("its top edge is 113 px above the hummingbird and its bottom edge 37 px below it")
    expect(text).toContain("The ground is 302 px below and the top edge 238 px above.")
    const data = JSON.parse(text.split("\n").pop()!)
    expect(data).toEqual({
      gap_middle: 38, gap_top: 113, gap_bottom: -37, speed: -4, gap_ahead: 120,
      in_gap: false, gap_height: 150, ground: -302, top: 238,
    })
  })

  it("says when the hummingbird is inside the gap, rising, or level", () => {
    const world = playing()
    world.pipes = [{ id: 1, x: BIRD.x - 20, gapY: 300, scored: false }]
    world.y = 300
    world.vy = -2.04
    const o = observe(world)
    expect(o.inGap).toBe(true)
    expect(o.ahead).toBe(0)
    expect(o.exit).toBe(PIPES.width - 20 + BIRD.radius)
    const text = describeState(o)
    expect(text).toContain("level with the middle of the next gap and rising at 2.0 px per frame")
    expect(text).toContain(`It is inside the gap now, which ends ${o.exit} px ahead`)
    world.vy = 0
    expect(describeState(observe(world))).toContain("hanging still")
  })

  it("never writes a negative zero", () => {
    const world = playing()
    world.pipes = [{ id: 1, x: 300, gapY: 250.2, scored: false }]
    world.y = 250
    world.vy = 0.01
    const o = observe(world)
    expect(Object.is(o.gapMiddle, -0)).toBe(false)
    expect(Object.is(o.speed, -0)).toBe(false)
  })

  it("is a pure function of the world", () => {
    const world = playing()
    for (let i = 0; i < 30; i++) step(world, i % 9 === 0)
    const before = structuredClone(world)
    expect(describeState(observe(world))).toBe(describeState(observe(world)))
    expect(world).toEqual(before)
  })
})

describe("the question", () => {
  it("asks one noul by default and one choice with criteria on request", () => {
    expect(buildQuestions("noul")).toEqual({
      flap: { type: "noul", instructions: "Should the hummingbird flap its wings now?" },
    })
    const choice = buildQuestions("choice").move
    expect(choice.type).toBe("choice")
    expect(Object.keys((choice as { criteria: Record<string, string> }).criteria)).toEqual(["flap", "glide"])
    const request = buildRequest(playing(), "noul", "qwen36")
    expect(request.model).toBe("qwen36")
    expect(typeof request.state).toBe("string")
    expect(Object.keys(request.questions)).toEqual([QUESTION_ID.noul])
  })

  it("reads p(flap) from either reply and refuses a reply without it", () => {
    expect(flapProbability(noulReply(0.73), "noul")).toBe(0.73)
    expect(flapProbability(choiceReply(0.2), "choice")).toBe(0.2)
    expect(() => flapProbability(choiceReply(0.2), "noul")).toThrow(/flap/)
    expect(() => flapProbability({ model: "m", answers: { flap: { type: "noul", noul: Number.NaN } } }, "noul")).toThrow()
    expect(() => flapProbability({ model: "m", answers: {} } as SystemOneResponse, "choice")).toThrow(/move/)
  })

  it("flaps strictly above the threshold", () => {
    expect(shouldFlap(0.51, 0.5)).toBe(true)
    expect(shouldFlap(0.5, 0.5)).toBe(false)
    expect(shouldFlap(0.49, 0.5)).toBe(false)
    expect(shouldFlap(0.8, 0.85)).toBe(false)
  })
})

describe("the pilot", () => {
  it("posts the request to /v1/systemone with the key, and flaps above the threshold", async () => {
    const fetchMock = vi.fn(async () => json(noulReply(0.62), { headers: { "x-colibri-elapsed-ms": "37" } }))
    vi.stubGlobal("fetch", fetchMock)
    const world = playing()
    const pilot = new Pilot({ baseUrl: "http://127.0.0.1:8000/v1/", apiKey: "secret", model: "qwen36" })
    await pilot.poll(world)
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit]
    expect(url).toBe("http://127.0.0.1:8000/v1/systemone")
    expect(init.method).toBe("POST")
    expect(init.headers).toMatchObject({ Authorization: "Bearer secret", "Content-Type": "application/json" })
    const body = JSON.parse(String(init.body))
    expect(body).toEqual({ model: "qwen36", state: describeState(observe(world)), questions: buildQuestions("noul") })
    const decision = pilot.take(world)!
    expect(decision).toMatchObject({ flap: true, p: 0.62, threshold: 0.5, engineMs: 37, model: "qwen36" })

    /* the same answer under a higher threshold is a glide */
    pilot.threshold = 0.7
    step(world, false)
    await pilot.poll(world)
    expect(pilot.take(world)).toMatchObject({ flap: false, p: 0.62, threshold: 0.7 })
  })

  it("reads the choice form's probability of flap", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => json(choiceReply(0.81))))
    const world = playing()
    const pilot = new Pilot({ baseUrl: "http://x/v1", apiKey: "", model: "m", form: "choice", threshold: 0.6 })
    await pilot.poll(world)
    expect(pilot.take(world)).toMatchObject({ flap: true, p: 0.81 })
  })

  it("keeps one request in flight while the game runs on", async () => {
    const { fetchMock, calls } = deferredFetch()
    const world = playing()
    const pilot = new Pilot({ baseUrl: "http://x/v1", apiKey: "", model: "m" })
    const pending = pilot.poll(world)
    expect(pending).not.toBeNull()
    expect(pilot.busy).toBe(true)
    for (let i = 0; i < 8; i++) {
      step(world, pilot.take(world)?.flap ?? false)
      expect(pilot.poll(world)).toBeNull()
    }
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(world.tick).toBe(8)

    calls[0].release(json(noulReply(0.1)))
    await pending
    expect(pilot.busy).toBe(false)
    /* an answer waiting to be applied also holds the next question back */
    expect(pilot.poll(world)).toBeNull()
    step(world, pilot.take(world)!.flap)
    expect(pilot.poll(world)).not.toBeNull()
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it("does not ask twice about the same step", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => json(noulReply(0.3))))
    const world = playing()
    const pilot = new Pilot({ baseUrl: "http://x/v1", apiKey: "", model: "m" })
    await pilot.poll(world)
    pilot.take(world)
    expect(pilot.poll(world)).toBeNull()
    step(world, false)
    expect(pilot.poll(world)).not.toBeNull()
  })

  it("applies a late answer to the current step, and says how late it was", async () => {
    const { calls } = deferredFetch()
    const time = clock()
    const applied: number[] = []
    const world = playing()
    const pilot = new Pilot({ baseUrl: "http://x/v1", apiKey: "", model: "m", now: time.now, onApply: (d) => applied.push(d.stepsLate) })
    const asked = pilot.poll(world)!
    const seen = calls[0].body.state

    /* the model thinks for 5 steps; the world does not wait */
    for (let i = 0; i < 5; i++) step(world, pilot.take(world)?.flap ?? false)
    expect(world.flaps).toBe(0)
    time.advance(83)
    calls[0].release(json(noulReply(0.9)))
    await asked

    const decision = pilot.take(world)!
    expect(decision.flap).toBe(true)
    expect(decision.askedTick).toBe(0)
    expect(decision.stepsLate).toBe(5)
    expect(decision.latencyMs).toBe(83)
    step(world, decision.flap)
    /* the flap lands on step 6, the current one, not back on step 1 */
    expect(world.lastFlapTick).toBe(6)
    expect(applied).toEqual([5])
    expect(pilot.last).toBe(decision)
    expect(pilot.lastState).toBe(seen)
    /* applied once */
    expect(pilot.take(world)).toBeNull()
  })

  it("drops an answer that arrives after the round was stopped", async () => {
    const { calls } = deferredFetch()
    const world = playing()
    const pilot = new Pilot({ baseUrl: "http://x/v1", apiKey: "", model: "m" })
    const asked = pilot.poll(world)!
    pilot.stop()
    expect(pilot.busy).toBe(false)
    calls[0].release(json(noulReply(0.99)))
    await asked
    expect(pilot.take(world)).toBeNull()
    expect(pilot.failure).toBeNull()
  })

  it("measures latency and the decision rate", async () => {
    const time = clock()
    vi.stubGlobal("fetch", vi.fn(async () => { time.advance(40); return json(noulReply(0.4)) }))
    const world = playing()
    const pilot = new Pilot({ baseUrl: "http://x/v1", apiKey: "", model: "m", now: time.now })
    for (let i = 0; i < 5; i++) {
      await pilot.poll(world)
      step(world, pilot.take(world)!.flap)
      time.advance(10)
    }
    expect(pilot.latency.summary()).toEqual({ last: 40, avg: 40, p95: 40, count: 5 })
    expect(pilot.latency.rate()).toBe(20)    // one answer every 50 ms
  })

  it("falls back when the server cannot be reached, and stops asking", async () => {
    const fetchMock = vi.fn(async () => { throw new TypeError("Failed to fetch") })
    vi.stubGlobal("fetch", fetchMock)
    const failures: PilotFailure[] = []
    const world = playing()
    const pilot = new Pilot({ baseUrl: "http://x/v1", apiKey: "", model: "m", onFailure: (f) => failures.push(f) })
    await pilot.poll(world)
    expect(failures).toEqual([{ kind: "unreachable", message: "Failed to fetch" }])
    expect(pilot.failure?.kind).toBe("unreachable")
    step(world, false)
    expect(pilot.poll(world)).toBeNull()
    expect(fetchMock).toHaveBeenCalledTimes(1)
    pilot.reset()
    expect(pilot.poll(world)).not.toBeNull()
  })

  it("tells a missing route, a refusal and an unreadable answer apart", async () => {
    const cases: Array<[Response, PilotFailure["kind"]]> = [
      [json({ error: { message: "Not found." } }, { status: 404 }), "route"],
      [json({ error: { message: "`state` is required" } }, { status: 422 }), "http"],
      [json({ model: "m", answers: {} }), "answer"],
    ]
    for (const [response, kind] of cases) {
      vi.stubGlobal("fetch", vi.fn(async () => response))
      const pilot = new Pilot({ baseUrl: "http://x/v1", apiKey: "", model: "m" })
      await pilot.poll(playing())
      expect(pilot.failure?.kind).toBe(kind)
    }
    expect(classify(new HttpError("Invalid API key.", 401))).toEqual({ kind: "http", status: 401, message: "Invalid API key." })
  })
})

describe("finding the model", () => {
  it("keeps the dashboard's model when the server lists it, else takes the served one", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => json({ object: "list", data: [{ id: "qwen36", object: "model" }] })))
    expect(await findModel("http://x/v1", "", "glm-5.2-colibri")).toEqual({ model: "qwen36" })
    expect(await findModel("http://x/v1", "", "qwen36")).toEqual({ model: "qwen36" })
  })

  it("falls back when nothing answers, nothing is loaded, or the model only draws", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("Failed to fetch") }))
    expect(await findModel("http://x/v1", "", "m")).toMatchObject({ failure: { kind: "unreachable" } })
    vi.stubGlobal("fetch", vi.fn(async () => json({ object: "list", data: [] })))
    expect(await findModel("http://x/v1", "", "m")).toMatchObject({ failure: { kind: "noModel" } })
    vi.stubGlobal("fetch", vi.fn(async () => json({ data: [{ id: "qwen-image", capabilities: ["image_generation"] }] })))
    expect(await findModel("http://x/v1", "", "m")).toMatchObject({ failure: { kind: "imageModel", model: "qwen-image" } })
    vi.stubGlobal("fetch", vi.fn(async () => json({ error: { message: "Invalid API key." } }, { status: 401 })))
    expect(await findModel("http://x/v1", "", "m")).toMatchObject({ failure: { kind: "http", message: "Invalid API key." } })
  })
})
