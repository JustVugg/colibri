/* FlappyBri: the model at the controls.
 *
 * Every step the game can be described in a few sentences: where the next gap
 * is, how fast the hummingbird is falling. The pilot sends that description to
 * POST /v1/systemone with one typed question, reads the probability of
 * flapping, and flaps when it is above the threshold.
 *
 * REAL TIME, HONESTLY. One request at a time, and the game does not wait for
 * it: while the model reads, the pipes keep coming. The answer is about the
 * screen the model saw, and it is applied to the screen as it is when the
 * answer arrives, the way a slow reflex would be. How many steps late that is
 * gets measured and shown, not hidden. */

import {
  HttpError, askSystemOne, generatesImages, listModelInfo,
  type SystemOneQuestion, type SystemOneRequest, type SystemOneResponse,
} from "../api"
import { BIRD, PHYSICS, PIPES, WORLD, nextPipe, type World } from "./game"
import { LatencyWindow } from "./stats"

/* ---- what the model reads ---------------------------------------------------- */

/* Positions are relative to the hummingbird, with UP positive: a gap middle at
   +38 is 38 px above it. Speeds the same way: -4 is falling at 4 px a frame. */
export interface Observation {
  tick: number
  gapMiddle: number
  gapTop: number
  gapBottom: number
  gapHeight: number
  ahead: number       // px from the hummingbird's beak to the pipe; 0 once inside
  inGap: boolean      // between the pipe's two edges, horizontally
  exit: number        // px until its tail leaves the pipe, when inside
  speed: number
  ground: number      // negative: the ground is always below
  top: number         // the top edge, always above
}

const round = (value: number) => {
  const r = Math.round(value)
  return Object.is(r, -0) ? 0 : r
}
const round1 = (value: number) => {
  const r = Math.round(value * 10) / 10
  return Object.is(r, -0) ? 0 : r
}

export function observe(world: World): Observation {
  const pipe = nextPipe(world)
  const gapY = pipe ? pipe.gapY : WORLD.floor / 2
  const left = pipe ? pipe.x : WORLD.width
  const front = BIRD.x + BIRD.radius
  const back = BIRD.x - BIRD.radius
  const inGap = left <= front && left + PIPES.width >= back
  return {
    tick: world.tick,
    gapMiddle: round(world.y - gapY),
    gapTop: round(world.y - (gapY - PIPES.gap / 2)),
    gapBottom: round(world.y - (gapY + PIPES.gap / 2)),
    gapHeight: PIPES.gap,
    ahead: inGap ? 0 : round(Math.max(0, left - front)),
    inGap,
    exit: inGap ? round(left + PIPES.width - back) : 0,
    speed: round1(-world.vy),
    ground: round(world.y - WORLD.floor),
    top: round(world.y),
  }
}

/* What one flap does, from the same constants the physics uses: the height it
   gains before gravity wins again, and in how many frames. */
export const FLAP_RISE = Math.round(PHYSICS.flap ** 2 / (2 * PHYSICS.gravity))
export const FLAP_FRAMES = Math.round(PHYSICS.flap / PHYSICS.gravity)

export const RULES =
  "FlappyBri: a hummingbird flies right through the gaps between pipes. " +
  "Touching a pipe, the ground or the top edge ends the game. " +
  `A flap sends it up about ${FLAP_RISE} px over ${FLAP_FRAMES} frames; without one it falls faster every frame.`

const updown = (value: number) => (value >= 0 ? `${value} px above` : `${-value} px below`)

/* The state is a short paragraph a person could act on, then the same numbers
   as compact JSON. The paragraph comes first and the rules first of all: the
   part that never changes leads, so an engine that reuses a cached prefix
   reads only the tail again. */
export function describeState(o: Observation): string {
  const vertical = o.gapMiddle === 0
    ? "level with the middle of the next gap"
    : `${Math.abs(o.gapMiddle)} px ${o.gapMiddle > 0 ? "below" : "above"} the middle of the next gap`
  const motion = o.speed === 0
    ? "hanging still"
    : `${o.speed < 0 ? "falling" : "rising"} at ${Math.abs(o.speed).toFixed(1)} px per frame`
  const gap = o.inGap
    ? `It is inside the gap now, which ends ${o.exit} px ahead and is ${o.gapHeight} px tall`
    : `The next gap starts ${o.ahead} px ahead and is ${o.gapHeight} px tall`
  const json = JSON.stringify({
    gap_middle: o.gapMiddle, gap_top: o.gapTop, gap_bottom: o.gapBottom, speed: o.speed,
    gap_ahead: o.ahead, in_gap: o.inGap, gap_height: o.gapHeight, ground: o.ground, top: o.top,
  })
  return [
    RULES,
    `Now: the hummingbird is ${vertical} and ${motion}. ` +
      `${gap}: its top edge is ${updown(o.gapTop)} the hummingbird and its bottom edge ${updown(o.gapBottom)} it. ` +
      `The ground is ${-o.ground} px below and the top edge ${o.top} px above.`,
    "Heights in px relative to the hummingbird and speed in px per frame, up is positive:",
    json,
  ].join("\n")
}

/* ---- the question -------------------------------------------------------------- */

/* `noul` is one yes/no question, two one-token answers: the lightest request,
   the default. `choice` names the two moves and says what each does, which
   costs a few more tokens and gives the model the consequences to weigh. */
export type Form = "noul" | "choice"

export const QUESTION_ID: Record<Form, string> = { noul: "flap", choice: "move" }

const FLAP_MEANS = `beat the wings once: it rises about ${FLAP_RISE} px over the next ${FLAP_FRAMES} frames`
const GLIDE_MEANS = "do nothing: it keeps falling, a little faster every frame"

export function buildQuestions(form: Form): Record<string, SystemOneQuestion> {
  if (form === "choice") {
    return {
      [QUESTION_ID.choice]: {
        type: "choice",
        instructions: "What should the hummingbird do now?",
        criteria: { flap: FLAP_MEANS, glide: GLIDE_MEANS },
      },
    }
  }
  return { [QUESTION_ID.noul]: { type: "noul", instructions: "Should the hummingbird flap its wings now?" } }
}

export function buildRequest(world: World, form: Form, model: string): SystemOneRequest {
  return { model, state: describeState(observe(world)), questions: buildQuestions(form) }
}

export class AnswerError extends Error {}

/* p(flap) out of the reply: `noul` is the probability of yes, `choice` carries
   one probability per label. Anything else is a reply this game cannot play on. */
export function flapProbability(response: SystemOneResponse, form: Form): number {
  const answer = response?.answers?.[QUESTION_ID[form]]
  const p = answer?.type === "noul" ? answer.noul
    : answer?.type === "choice" ? answer.probabilities?.flap
    : undefined
  if (typeof p !== "number" || !Number.isFinite(p) || p < 0 || p > 1) {
    throw new AnswerError(`The reply has no probability for "${QUESTION_ID[form]}".`)
  }
  return p
}

/* Strictly above: at exactly the threshold the model is not saying flap. */
export const shouldFlap = (p: number, threshold: number) => p > threshold

/* ---- failures ------------------------------------------------------------------ */

export type FailureKind = "unreachable" | "noModel" | "imageModel" | "route" | "http" | "answer"

export interface PilotFailure {
  kind: FailureKind
  message: string
  status?: number
  model?: string
}

export function classify(cause: unknown): PilotFailure {
  if (cause instanceof HttpError) {
    return cause.status === 404
      ? { kind: "route", status: 404, message: cause.message }
      : { kind: "http", status: cause.status, message: cause.message }
  }
  if (cause instanceof AnswerError || cause instanceof SyntaxError) {
    return { kind: "answer", message: cause.message }
  }
  return { kind: "unreachable", message: cause instanceof Error ? cause.message : String(cause) }
}

/* Before the model plays: is there a server, and a model on it that can decide?
   /v1/models answers both. The dashboard's selected model is kept when the
   server lists it; otherwise the one the server runs. */
export async function findModel(baseUrl: string, apiKey: string, preferred: string, signal?: AbortSignal):
  Promise<{ model: string } | { failure: PilotFailure }> {
  let models
  try {
    models = await listModelInfo(baseUrl, apiKey, signal)
  } catch (cause) {
    if (cause instanceof TypeError) return { failure: { kind: "unreachable", message: cause.message } }
    return { failure: { kind: "http", message: cause instanceof Error ? cause.message : String(cause) } }
  }
  if (!models.length) return { failure: { kind: "noModel", message: "The server lists no model." } }
  const chosen = models.find((item) => item.id === preferred) || models[0]
  if (generatesImages(chosen)) {
    return { failure: { kind: "imageModel", model: chosen.id, message: `${chosen.id} generates images.` } }
  }
  return { model: chosen.id }
}

/* ---- the pilot ----------------------------------------------------------------- */

export interface Decision {
  flap: boolean
  p: number
  threshold: number
  latencyMs: number
  engineMs: number | null
  askedTick: number
  /* Steps the world moved on between the screen the model read and the step
     the answer was applied to. Known when it is applied. */
  stepsLate: number
  model: string
}

export interface PilotOptions {
  baseUrl: string
  apiKey: string
  model: string
  form?: Form
  threshold?: number
  window?: number
  now?: () => number
  onApply?: (decision: Decision) => void
  onFailure?: (failure: PilotFailure) => void
}

export class Pilot {
  baseUrl: string
  apiKey: string
  model: string
  form: Form
  threshold: number
  readonly latency: LatencyWindow
  failure: PilotFailure | null = null
  last: Decision | null = null
  lastState = ""
  private readonly now: () => number
  private readonly onApply?: (decision: Decision) => void
  private readonly onFailure?: (failure: PilotFailure) => void
  private controller: AbortController | null = null
  private arrived: Decision | null = null
  private askedTick = -1
  private generation = 0

  constructor(options: PilotOptions) {
    this.baseUrl = options.baseUrl
    this.apiKey = options.apiKey
    this.model = options.model
    this.form = options.form ?? "noul"
    this.threshold = options.threshold ?? 0.5
    this.latency = new LatencyWindow(options.window ?? 100)
    this.now = options.now ?? (() => performance.now())
    this.onApply = options.onApply
    this.onFailure = options.onFailure
  }

  /* A request is on the wire. */
  get busy() {
    return this.controller !== null
  }

  /* Ask about the world as it is, unless a question is already out, an answer
     is waiting to be applied, the world has not moved since the last question,
     or the pilot has failed. Returns the request, so a caller (a test) can wait
     for it; the game itself never does. */
  poll(world: World): Promise<void> | null {
    if (this.failure || this.controller || this.arrived) return null
    if (world.phase !== "playing" || world.tick === this.askedTick) return null
    return this.ask(world)
  }

  /* Called once per step, before the step: the answer that arrived since the
     last one, if any, now applied to the current world. */
  take(world: World): Decision | null {
    const decision = this.arrived
    if (!decision) return null
    this.arrived = null
    decision.stepsLate = world.tick - decision.askedTick
    this.last = decision
    this.onApply?.(decision)
    return decision
  }

  /* Drop the question in flight and any answer not yet applied: a pause, a new
     round. The latencies stay, they describe the model, not the round. */
  stop() {
    this.generation += 1
    this.controller?.abort()
    this.controller = null
    this.arrived = null
    this.askedTick = -1
    this.latency.breakRun()
  }

  /* Start over after a failure, for a new attempt. */
  reset() {
    this.stop()
    this.failure = null
  }

  private async ask(world: World) {
    const generation = this.generation
    const controller = new AbortController()
    this.controller = controller
    const form = this.form
    const request = buildRequest(world, form, this.model)
    const askedTick = world.tick
    this.askedTick = askedTick
    this.lastState = request.state
    const started = this.now()
    try {
      const reply = await askSystemOne(this.baseUrl, this.apiKey, request, controller.signal)
      if (generation !== this.generation) return
      const p = flapProbability(reply.response, form)
      const at = this.now()
      const latencyMs = at - started
      this.latency.push(latencyMs, at)
      this.arrived = {
        flap: shouldFlap(p, this.threshold), p, threshold: this.threshold, latencyMs,
        engineMs: reply.engineMs, askedTick, stepsLate: 0,
        model: typeof reply.response.model === "string" && reply.response.model ? reply.response.model : this.model,
      }
    } catch (cause) {
      if (generation !== this.generation || controller.signal.aborted) return
      this.failure = classify(cause)
      this.onFailure?.(this.failure)
    } finally {
      if (this.controller === controller) this.controller = null
    }
  }
}
