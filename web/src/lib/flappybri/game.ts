/* FlappyBri: the world and its physics.
 *
 * Everything here is plain data and pure arithmetic, with no clock, no canvas
 * and no Math.random: a world is a function of its seed and of the tick at
 * which each flap arrived. That is what makes a round reproducible, what lets
 * the tests run without a browser, and what keeps the speed slider honest: it
 * changes how many steps run per second of wall time, never what one step does.
 *
 * Units are logical pixels and steps. One step is 1/60 s of game time at full
 * speed; the playing field is WORLD.width x WORLD.height whatever the size of
 * the canvas that shows it. */

export const STEPS_PER_SECOND = 60

export const WORLD = { width: 400, height: 600, floor: 540 } as const

export const PHYSICS = {
  gravity: 0.36,      // px per step, added to the downward speed every step
  flap: 6.6,          // px per step: a flap sets the upward speed to this
  terminal: 9.5,      // px per step: the fastest the hummingbird can fall
} as const

export const BIRD = { x: 110, radius: 12 } as const

export const PIPES = {
  width: 64,
  gap: 150,           // px between the two halves of a pipe
  spacing: 220,       // px from one pipe to the next
  speed: 2.4,         // px per step the pipes travel towards the hummingbird
  first: 460,         // x of the first pipe when a round starts
  margin: 56,         // px a gap always keeps from the top and from the ground
  shift: 170,         // the most a gap's middle moves from one pipe to the next
} as const

export type Phase = "ready" | "playing" | "paused" | "over"
export type Cause = "pipe" | "ground" | "ceiling"

export interface Pipe {
  id: number
  x: number           // left edge
  gapY: number        // middle of the gap
  scored: boolean
}

export interface World {
  seed: number
  rng: number         // PRNG state, advanced only when a pipe is made
  tick: number
  distance: number    // px travelled since the start of the round
  y: number           // middle of the hummingbird
  vy: number          // px per step, positive is DOWN (screen coordinates)
  pipes: Pipe[]
  nextId: number
  score: number
  flaps: number
  lastFlapTick: number
  phase: Phase
  cause: Cause | null
}

export interface StepEvents {
  scored: number
  died: boolean
}

/* mulberry32: a 32-bit state, one multiply-xorshift per draw. Small, fast and
   good enough for gap heights; what matters is that it is ours and seeded. */
export function nextRandom(world: World): number {
  world.rng = (world.rng + 0x6d2b79f5) | 0
  let t = world.rng
  t = Math.imul(t ^ (t >>> 15), t | 1)
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296
}

export const GAP_MIN = PIPES.margin + PIPES.gap / 2
export const GAP_MAX = WORLD.floor - PIPES.margin - PIPES.gap / 2

const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value))

function addPipe(world: World, x: number) {
  const previous = world.pipes[world.pipes.length - 1]
  const r = nextRandom(world)
  /* The first gap anywhere; every next one within `shift` of the one before,
     so a gap is never out of reach of a hummingbird that just cleared the last. */
  const gapY = previous
    ? clamp(previous.gapY + (r * 2 - 1) * PIPES.shift, GAP_MIN, GAP_MAX)
    : GAP_MIN + r * (GAP_MAX - GAP_MIN)
  world.pipes.push({ id: world.nextId++, x, gapY, scored: false })
}

/* Pipes are made one spacing apart until one waits beyond the right edge. */
function fill(world: World) {
  if (!world.pipes.length) addPipe(world, PIPES.first)
  let last = world.pipes[world.pipes.length - 1]
  while (last.x < WORLD.width + PIPES.spacing) {
    addPipe(world, last.x + PIPES.spacing)
    last = world.pipes[world.pipes.length - 1]
  }
}

export function createWorld(seed: number): World {
  const world: World = {
    seed: seed >>> 0,
    rng: seed | 0,
    tick: 0,
    distance: 0,
    y: (WORLD.floor * 0.45),
    vy: 0,
    pipes: [],
    nextId: 1,
    score: 0,
    flaps: 0,
    lastFlapTick: -1,
    phase: "ready",
    cause: null,
  }
  fill(world)
  return world
}

export function circleHitsRect(cx: number, cy: number, r: number, x0: number, y0: number, x1: number, y1: number) {
  const dx = cx - clamp(cx, x0, x1)
  const dy = cy - clamp(cy, y0, y1)
  return dx * dx + dy * dy < r * r
}

/* What the hummingbird touches right now, if anything. */
export function collision(world: World): Cause | null {
  const { x, radius } = BIRD
  if (world.y + radius >= WORLD.floor) return "ground"
  if (world.y - radius <= 0) return "ceiling"
  for (const pipe of world.pipes) {
    if (pipe.x > x + radius || pipe.x + PIPES.width < x - radius) continue
    const top = pipe.gapY - PIPES.gap / 2
    const bottom = pipe.gapY + PIPES.gap / 2
    if (circleHitsRect(x, world.y, radius, pipe.x, -Infinity, pipe.x + PIPES.width, top)) return "pipe"
    if (circleHitsRect(x, world.y, radius, pipe.x, bottom, pipe.x + PIPES.width, WORLD.floor)) return "pipe"
  }
  return null
}

/* The first pipe the hummingbird has not yet left behind: the one that matters. */
export function nextPipe(world: World): Pipe | undefined {
  return world.pipes.find((pipe) => pipe.x + PIPES.width >= BIRD.x - BIRD.radius)
}

/* One step of game time. A flap is an input of THIS step: it replaces the
   speed before the move, so the same flaps at the same ticks give the same
   round. Does nothing unless the round is being played. */
export function step(world: World, flap: boolean): StepEvents {
  const events: StepEvents = { scored: 0, died: false }
  if (world.phase !== "playing") return events
  world.tick += 1
  if (flap) {
    world.vy = -PHYSICS.flap
    world.flaps += 1
    world.lastFlapTick = world.tick
  } else {
    world.vy = Math.min(world.vy + PHYSICS.gravity, PHYSICS.terminal)
  }
  world.y += world.vy
  world.distance += PIPES.speed
  for (const pipe of world.pipes) pipe.x -= PIPES.speed
  while (world.pipes.length && world.pipes[0].x + PIPES.width < -PIPES.width) world.pipes.shift()
  fill(world)

  const hit = collision(world)
  if (hit) {
    world.phase = "over"
    world.cause = hit
    if (hit === "ground") world.y = WORLD.floor - BIRD.radius
    events.died = true
    return events
  }
  for (const pipe of world.pipes) {
    if (!pipe.scored && pipe.x + PIPES.width < BIRD.x) {
      pipe.scored = true
      world.score += 1
      events.scored += 1
    }
  }
  return events
}

/* ---- fixed timestep --------------------------------------------------------
 * Wall time goes in, whole steps come out. `carry` is the fraction of a step
 * left over, kept for the next frame (and used to draw between two steps).
 * A frame longer than MAX_FRAME_MS (a tab coming back, a debugger pause) counts
 * as MAX_FRAME_MS: better a game that skips a moment than one that runs dozens
 * of steps at once and kills the hummingbird before anyone sees it. */
export const MAX_FRAME_MS = 250

export function accumulate(carry: number, dtMs: number, speed: number) {
  const total = carry + (clamp(dtMs, 0, MAX_FRAME_MS) * speed * STEPS_PER_SECOND) / 1000
  /* The epsilon absorbs float drift: 60 frames of 1000/60 ms must make 60 steps. */
  const steps = Math.floor(total + 1e-9)
  return { steps, carry: Math.max(0, total - steps) }
}

/* ---- best score ------------------------------------------------------------ */
export function nextBest(best: number, score: number) {
  return score > best ? score : best
}

export function readBest(storage: Pick<Storage, "getItem"> | undefined, key: string): number {
  try {
    const value = Number(storage?.getItem(key))
    return Number.isFinite(value) && value > 0 ? Math.floor(value) : 0
  } catch {
    return 0
  }
}

export function writeBest(storage: Pick<Storage, "setItem"> | undefined, key: string, best: number) {
  try { storage?.setItem(key, String(best)) } catch { /* private mode, blocked storage */ }
}
