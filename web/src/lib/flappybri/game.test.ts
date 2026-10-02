import { describe, expect, it } from "vitest"

import {
  BIRD, GAP_MAX, GAP_MIN, MAX_FRAME_MS, PHYSICS, PIPES, WORLD,
  accumulate, circleHitsRect, collision, createWorld, nextBest, nextPipe, readBest, step, writeBest,
  type World,
} from "./game"

/* A plain rule that plays well enough to cross many pipes: flap when falling
   below the middle of the next gap. Deterministic, so it is an input sequence. */
const autopilot = (world: World) => {
  const pipe = nextPipe(world)
  return !!pipe && world.vy > 0 && world.y > pipe.gapY + 14
}

const play = (seed: number, steps: number, rule = autopilot) => {
  const world = createWorld(seed)
  world.phase = "playing"
  for (let i = 0; i < steps && world.phase === "playing"; i++) step(world, rule(world))
  return world
}

/* The same tiny PRNG the game uses, for test inputs that are random but fixed. */
const lcg = (seed: number) => () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296)

describe("physics determinism", () => {
  it("replays the same round for the same seed and the same flaps", () => {
    const a = play(42, 3000)
    const b = play(42, 3000)
    expect(b).toEqual(a)
    expect(a.tick).toBeGreaterThan(600)
  })

  it("draws different gaps for a different seed", () => {
    const gaps = (seed: number) => createWorld(seed).pipes.map((pipe) => pipe.gapY)
    expect(gaps(1)).toEqual(gaps(1))
    expect(gaps(1)).not.toEqual(gaps(2))
  })

  it("keeps every gap inside the field and within reach of the one before", () => {
    const world = createWorld(7)
    world.phase = "playing"
    const seen = new Map<number, number>()
    for (let i = 0; i < 6000; i++) {
      world.y = nextPipe(world)!.gapY   // carry the bird through every gap: only the pipes matter here
      world.vy = 0
      step(world, false)
      for (const pipe of world.pipes) seen.set(pipe.id, pipe.gapY)
    }
    expect(world.phase).toBe("playing")
    const gaps = [...seen.entries()].sort((a, b) => a[0] - b[0]).map(([, gapY]) => gapY)
    expect(gaps.length).toBeGreaterThan(50)
    for (const [index, gapY] of gaps.entries()) {
      expect(gapY).toBeGreaterThanOrEqual(GAP_MIN)
      expect(gapY).toBeLessThanOrEqual(GAP_MAX)
      if (index) expect(Math.abs(gapY - gaps[index - 1])).toBeLessThanOrEqual(PIPES.shift + 1e-9)
    }
  })

  it("does not depend on how wall time is cut into frames, nor on the speed", () => {
    /* The reference: the rule applied step by step. */
    const reference = play(99, 1500)

    /* The same seed driven by a frame loop with jittery frame times and a
       speed slider moved halfway: the steps come out in different bundles,
       the round must not change. */
    const world = createWorld(99)
    world.phase = "playing"
    const random = lcg(5)
    let carry = 0
    let frames = 0
    while (world.tick < 1500 && world.phase === "playing") {
      const speed = frames++ < 200 ? 1 : 0.3
      const out = accumulate(carry, 4 + random() * 40, speed)
      carry = out.carry
      for (let i = 0; i < out.steps && world.tick < 1500 && world.phase === "playing"; i++) step(world, autopilot(world))
    }
    expect(world).toEqual(reference)
  })

  it("applies gravity, the flap impulse and the terminal velocity", () => {
    const world = createWorld(3)
    world.phase = "playing"
    const y0 = world.y
    step(world, false)
    expect(world.vy).toBeCloseTo(PHYSICS.gravity)
    expect(world.y).toBeCloseTo(y0 + PHYSICS.gravity)
    step(world, true)
    expect(world.vy).toBe(-PHYSICS.flap)
    expect(world.lastFlapTick).toBe(2)
    world.vy = PHYSICS.terminal
    world.y = 100
    step(world, false)
    expect(world.vy).toBe(PHYSICS.terminal)
  })

  it("does not move unless the round is being played", () => {
    const world = createWorld(3)
    const before = structuredClone(world)
    expect(step(world, true)).toEqual({ scored: 0, died: false })
    expect(world).toEqual(before)
    world.phase = "paused"
    step(world, false)
    expect(world.tick).toBe(0)
  })
})

describe("fixed timestep", () => {
  it("turns 60 frames of a 60 Hz display into 60 steps", () => {
    let carry = 0
    let steps = 0
    for (let i = 0; i < 60; i++) {
      const out = accumulate(carry, 1000 / 60, 1)
      carry = out.carry
      steps += out.steps
    }
    expect(steps).toBe(60)
  })

  it("runs fewer steps at a lower speed", () => {
    let carry = 0
    let steps = 0
    for (let i = 0; i < 120; i++) {
      const out = accumulate(carry, 1000 / 120, 0.25)
      carry = out.carry
      steps += out.steps
    }
    expect(steps).toBe(15)
  })

  it("caps a long frame instead of running every missed step", () => {
    expect(accumulate(0, 10_000, 1).steps).toBe(Math.floor((MAX_FRAME_MS * 60) / 1000))
    expect(accumulate(0, -5, 1)).toEqual({ steps: 0, carry: 0 })
  })
})

describe("collision", () => {
  const playing = () => {
    const world = createWorld(11)
    world.phase = "playing"
    world.pipes = [{ id: 1, x: BIRD.x - PIPES.width / 2, gapY: 300, scored: false }]
    return world
  }

  it("is safe in the middle of a gap", () => {
    const world = playing()
    world.y = 300
    expect(collision(world)).toBeNull()
  })

  it("hits the upper and the lower half of a pipe", () => {
    const world = playing()
    world.y = 300 - PIPES.gap / 2 + BIRD.radius - 1
    expect(collision(world)).toBe("pipe")
    world.y = 300 + PIPES.gap / 2 - BIRD.radius + 1
    expect(collision(world)).toBe("pipe")
    world.y = 300 + PIPES.gap / 2 - BIRD.radius - 1
    expect(collision(world)).toBeNull()
  })

  it("hits a pipe it reaches with its beak from the side", () => {
    const world = playing()
    world.pipes[0].x = BIRD.x + BIRD.radius - 1
    world.y = 100
    expect(collision(world)).toBe("pipe")
    world.pipes[0].x = BIRD.x + BIRD.radius + 1
    expect(collision(world)).toBeNull()
  })

  it("hits the ground and the top edge", () => {
    const world = playing()
    world.pipes = []
    world.y = WORLD.floor - BIRD.radius
    expect(collision(world)).toBe("ground")
    world.y = BIRD.radius
    expect(collision(world)).toBe("ceiling")
  })

  it("ends the round on the step that touches, with the cause", () => {
    const world = createWorld(5)
    world.phase = "playing"
    let died = false
    for (let i = 0; i < 400 && !died; i++) died = step(world, false).died
    expect(died).toBe(true)
    expect(world.phase).toBe("over")
    expect(["ground", "pipe"]).toContain(world.cause)
    const tick = world.tick
    step(world, true)
    expect(world.tick).toBe(tick)
  })

  it("measures a circle against a rectangle at its corner", () => {
    expect(circleHitsRect(0, 0, 10, 7, 7, 20, 20)).toBe(true)     // 7*sqrt(2) < 10
    expect(circleHitsRect(0, 0, 10, 8, 8, 20, 20)).toBe(false)    // 8*sqrt(2) > 10
  })
})

describe("scoring", () => {
  it("counts each pipe once, when the hummingbird is past it", () => {
    const world = createWorld(42)
    world.phase = "playing"
    let total = 0
    for (let i = 0; i < 4000 && world.phase === "playing"; i++) {
      total += step(world, autopilot(world)).scored
      const passed = world.pipes.filter((pipe) => pipe.x + PIPES.width < BIRD.x)
      expect(passed.every((pipe) => pipe.scored)).toBe(true)
    }
    expect(world.score).toBe(total)
    expect(world.score).toBeGreaterThanOrEqual(5)
  })

  it("scores nothing on the step that kills", () => {
    const world = createWorld(1)
    world.phase = "playing"
    world.pipes = [{ id: 1, x: BIRD.x - PIPES.width - 1, gapY: 300, scored: false }]
    world.y = WORLD.floor - BIRD.radius - 0.1
    world.vy = 5
    const events = step(world, false)
    expect(events).toEqual({ scored: 0, died: true })
    expect(world.score).toBe(0)
  })

  it("keeps the best score and survives a storage that throws", () => {
    expect(nextBest(4, 9)).toBe(9)
    expect(nextBest(9, 4)).toBe(9)
    const memory = new Map<string, string>()
    const storage = { getItem: (key: string) => memory.get(key) ?? null, setItem: (key: string, value: string) => void memory.set(key, value) }
    writeBest(storage, "best", 12)
    expect(readBest(storage, "best")).toBe(12)
    expect(readBest(storage, "other")).toBe(0)
    memory.set("junk", "lots")
    expect(readBest(storage, "junk")).toBe(0)
    const broken = {
      getItem: () => { throw new Error("SecurityError") },
      setItem: () => { throw new Error("QuotaExceededError") },
    }
    expect(readBest(broken, "best")).toBe(0)
    expect(() => writeBest(broken, "best", 3)).not.toThrow()
    expect(readBest(undefined, "best")).toBe(0)
  })
})
