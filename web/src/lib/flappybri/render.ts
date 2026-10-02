/* FlappyBri: drawing one frame.
 *
 * The hummingbird is colibri's logo, cell for cell: the same 11 x 9 grid and
 * the same five colors as assets/colibri-icon.svg and components/Brand.tsx,
 * mirrored so the beak points the way it flies. Everything else takes its
 * colors from the dashboard's tokens (--card, --primary, --foreground...), so
 * the game follows the dark and the light theme like the rest of the page.
 *
 * Decorative motion (the hills drifting behind, the wing beat, the tilt, the
 * bob while waiting, the flash on a crash) is switched off when the viewer
 * asks for reduced motion. What the game needs, the pipes and the hummingbird
 * moving, stays. */

import { BIRD, PIPES, PHYSICS, WORLD, type World } from "./game"

/* ---- the logo ---------------------------------------------------------------- */

/* Rows of the logo as drawn in the SVG, beak to the left: P pink, C light cyan,
   O orange, T teal, W white. */
export const LOGO_ROWS = [
  "....PPP....",
  "...PPPPP..C",
  "....PPPP.CC",
  "OOOOTTWTCC.",
  "....TTTTTCC",
  ".....TTTTCC",
  "......TTCC.",
  ".......TC..",
  "........C..",
] as const

export const LOGO_COLORS: Record<string, string> = {
  P: "#d75fd7", C: "#5fd7d7", O: "#ff8700", T: "#00afaf", W: "#ffffff",
}

export interface Cell { col: number; row: number; color: string }

/* The cells facing right. With `beat`, the raised wing (the light cells above
   the body on the right of the logo) is folded: the downstroke frame. */
export function logoCells(beat = false): Cell[] {
  const width = LOGO_ROWS[0].length
  const cells: Cell[] = []
  LOGO_ROWS.forEach((line, row) => {
    for (let col = 0; col < line.length; col++) {
      const key = line[col]
      if (key === ".") continue
      if (beat && key === "C" && row <= 2) continue
      cells.push({ col: width - 1 - col, row, color: LOGO_COLORS[key] })
    }
  })
  return cells
}

export const CELL = 3
export const SPRITE_W = LOGO_ROWS[0].length * CELL
export const SPRITE_H = LOGO_ROWS.length * CELL

/* ---- theme ------------------------------------------------------------------- */

export interface Palette {
  sky: string
  skyHigh: string
  far: string
  near: string
  pipe: string
  pipeLight: string
  pipeShade: string
  pipeEdge: string
  ground: string
  groundLine: string
  groundDot: string
  ink: string
  flap: string
  glide: string
}

type RGB = [number, number, number]

function parse(color: string): RGB | null {
  const hex = color.trim().replace(/^#/, "")
  if (/^[0-9a-f]{3}$/i.test(hex)) return [0, 1, 2].map((i) => parseInt(hex[i] + hex[i], 16)) as RGB
  if (/^[0-9a-f]{6}$/i.test(hex)) return [0, 2, 4].map((i) => parseInt(hex.slice(i, i + 2), 16)) as RGB
  return null
}

const css = ([r, g, b]: RGB) => `rgb(${Math.round(r)} ${Math.round(g)} ${Math.round(b)})`

/* `amount` of `a` over `b`, as color-mix(in srgb, a amount, b) would give. */
export function mix(a: string, b: string, amount: number): string {
  const x = parse(a)
  const y = parse(b)
  if (!x || !y) return amount >= 0.5 ? a : b
  return css([0, 1, 2].map((i) => x[i] * amount + y[i] * (1 - amount)) as RGB)
}

const FALLBACK = { background: "#212121", foreground: "#ececec", card: "#262626", primary: "#b2ccbb", secondary: "#303030", muted: "#a1a1a1" }

export function readPalette(root: Element | null): Palette {
  const style = root ? getComputedStyle(root) : null
  const token = (name: string, fallback: string) => {
    const value = style?.getPropertyValue(name).trim()
    return value && parse(value) ? value : fallback
  }
  const background = token("--background", FALLBACK.background)
  const foreground = token("--foreground", FALLBACK.foreground)
  const card = token("--card", FALLBACK.card)
  const primary = token("--primary", FALLBACK.primary)
  const secondary = token("--secondary", FALLBACK.secondary)
  const muted = token("--muted-foreground", FALLBACK.muted)
  return {
    sky: card,
    skyHigh: mix(background, card, 0.55),
    far: mix(foreground, card, 0.055),
    near: mix(foreground, card, 0.1),
    pipe: mix(primary, card, 0.42),
    pipeLight: mix(primary, card, 0.62),
    pipeShade: mix(primary, card, 0.28),
    pipeEdge: primary,
    ground: secondary,
    groundLine: mix(primary, secondary, 0.55),
    groundDot: mix(foreground, secondary, 0.08),
    ink: foreground,
    flap: primary,
    glide: muted,
  }
}

/* ---- scenery ----------------------------------------------------------------- */

/* Two bands of pixel hills, the same in every round (they are scenery, not
   part of the seed): sums of whole sine waves over one repeat, so the band
   loops without a seam, quantized to cells of 8 px. */
const HILL_CELL = 8
const HILL_SPAN = 64          // cells per repeat: 512 px
function hills(base: number, waves: Array<[number, number, number]>): number[] {
  return Array.from({ length: HILL_SPAN }, (_, i) => {
    const angle = (i / HILL_SPAN) * Math.PI * 2
    return Math.round(base + waves.reduce((sum, [amplitude, cycles, phase]) => sum + amplitude * Math.sin(angle * cycles + phase), 0))
  })
}
const FAR_HILLS = hills(10, [[3, 2, 0.4], [1.6, 5, 1.9]])
const NEAR_HILLS = hills(5, [[2.2, 3, 2.2], [1.1, 7, 0.3]])

const CLOUD = ["..XXX...", ".XXXXXX.", "XXXXXXXX"]
const CLOUDS = [{ x: 40, y: 92 }, { x: 250, y: 150 }, { x: 420, y: 60 }]

function drawHills(ctx: CanvasRenderingContext2D, heights: number[], offset: number, color: string) {
  ctx.fillStyle = color
  const span = HILL_SPAN * HILL_CELL
  const shift = ((offset % span) + span) % span
  for (let i = 0; i < HILL_SPAN * 2; i++) {
    const x = i * HILL_CELL - shift
    if (x > WORLD.width || x + HILL_CELL < 0) continue
    const h = heights[i % HILL_SPAN] * HILL_CELL
    ctx.fillRect(Math.round(x), WORLD.floor - h, HILL_CELL, h)
  }
}

function drawCloud(ctx: CanvasRenderingContext2D, x: number, y: number) {
  const size = 6
  CLOUD.forEach((line, row) => {
    for (let col = 0; col < line.length; col++) {
      if (line[col] === "X") ctx.fillRect(Math.round(x + col * size), y + row * size, size, size)
    }
  })
}

/* ---- the frame ----------------------------------------------------------------- */

export interface TrailPoint { distance: number; y: number; flap: boolean }

export interface Scene {
  world: World
  /* fraction of a step since the last one, to draw between two steps */
  alpha: number
  palette: Palette
  reduced: boolean
  /* ms clock for decorative motion */
  time: number
  sprites: { up: CanvasImageSource; beat: CanvasImageSource }
  trail: readonly TrailPoint[]
  /* ms since the crash, for the flash; null while alive */
  crashAge: number | null
  font: string
}

export function makeSprite(beat: boolean): HTMLCanvasElement {
  const canvas = document.createElement("canvas")
  canvas.width = SPRITE_W
  canvas.height = SPRITE_H
  const ctx = canvas.getContext("2d")
  if (ctx) for (const cell of logoCells(beat)) {
    ctx.fillStyle = cell.color
    ctx.fillRect(cell.col * CELL, cell.row * CELL, CELL, CELL)
  }
  return canvas
}

export function draw(ctx: CanvasRenderingContext2D, scene: Scene) {
  const { world, palette, reduced, time } = scene
  const moving = world.phase === "playing"
  const alpha = moving ? scene.alpha : 0
  const travelled = world.distance + PIPES.speed * alpha

  /* sky */
  const sky = ctx.createLinearGradient(0, 0, 0, WORLD.floor)
  sky.addColorStop(0, palette.skyHigh)
  sky.addColorStop(1, palette.sky)
  ctx.fillStyle = sky
  ctx.fillRect(0, 0, WORLD.width, WORLD.height)

  /* clouds and hills drift slower than the pipes: depth, and only decoration */
  const drift = reduced ? 0 : travelled
  ctx.fillStyle = palette.far
  for (const cloud of CLOUDS) {
    const span = WORLD.width + 120
    const x = ((((cloud.x - drift * 0.08) % span) + span) % span) - 60
    drawCloud(ctx, x, cloud.y)
  }
  drawHills(ctx, FAR_HILLS, drift * 0.18, palette.far)
  drawHills(ctx, NEAR_HILLS, drift * 0.4 + 200, palette.near)

  /* pipes */
  for (const pipe of world.pipes) {
    const x = Math.round(pipe.x - PIPES.speed * alpha)
    if (x > WORLD.width || x + PIPES.width < -8) continue
    const top = Math.round(pipe.gapY - PIPES.gap / 2)
    const bottom = Math.round(pipe.gapY + PIPES.gap / 2)
    drawPipe(ctx, palette, x, 0, top, true)
    drawPipe(ctx, palette, x, bottom, WORLD.floor, false)
  }

  /* ground: it moves with the world, like the pipes */
  ctx.fillStyle = palette.ground
  ctx.fillRect(0, WORLD.floor, WORLD.width, WORLD.height - WORLD.floor)
  ctx.fillStyle = palette.groundLine
  ctx.fillRect(0, WORLD.floor, WORLD.width, 4)
  ctx.fillStyle = palette.groundDot
  const pitch = 24
  const shift = travelled % pitch
  for (let x = -pitch; x < WORLD.width + pitch; x += pitch) {
    ctx.fillRect(Math.round(x - shift), WORLD.floor + 16, 8, 4)
    ctx.fillRect(Math.round(x - shift + pitch / 2), WORLD.floor + 32, 8, 4)
  }

  /* where the model decided: a dot per answer, left behind as the world moves */
  for (const point of scene.trail) {
    const x = BIRD.x - (travelled - point.distance)
    if (x < -6) continue
    ctx.fillStyle = point.flap ? palette.flap : palette.glide
    ctx.globalAlpha = point.flap ? 0.95 : 0.45
    const size = point.flap ? 6 : 4
    ctx.fillRect(Math.round(x - size / 2), Math.round(point.y - size / 2), size, size)
  }
  ctx.globalAlpha = 1

  /* the hummingbird */
  let y = world.y
  if (moving) y += Math.min(world.vy + PHYSICS.gravity, PHYSICS.terminal) * alpha
  if (world.phase === "ready" && !reduced) y += Math.sin(time / 320) * 5
  const sinceFlap = world.tick - world.lastFlapTick
  const beat = !reduced && (world.phase === "ready"
    ? Math.floor(time / 140) % 2 === 1
    : world.lastFlapTick >= 0 && sinceFlap < 6 && world.phase !== "over")
  const tilt = reduced || world.phase === "ready" ? 0 : Math.max(-0.42, Math.min(0.75, world.vy * 0.07))
  ctx.save()
  ctx.translate(BIRD.x, Math.round(y))
  ctx.rotate(tilt)
  ctx.drawImage(beat ? scene.sprites.beat : scene.sprites.up, -SPRITE_W / 2, -SPRITE_H / 2)
  ctx.restore()

  /* the score, in the wordmark's pixel face */
  if (world.phase !== "ready") {
    ctx.font = scene.font
    ctx.textAlign = "center"
    ctx.textBaseline = "top"
    ctx.fillStyle = palette.ink
    ctx.fillText(String(world.score), WORLD.width / 2, 34)
  }

  /* a short flash on the crash */
  if (!reduced && scene.crashAge !== null && scene.crashAge < 220) {
    ctx.fillStyle = palette.ink
    ctx.globalAlpha = 0.22 * (1 - scene.crashAge / 220)
    ctx.fillRect(0, 0, WORLD.width, WORLD.height)
    ctx.globalAlpha = 1
  }
}

const CAP_H = 20
const CAP_OVER = 4

function drawPipe(ctx: CanvasRenderingContext2D, palette: Palette, x: number, y0: number, y1: number, fromTop: boolean) {
  if (y1 <= y0) return
  const w = PIPES.width
  ctx.fillStyle = palette.pipe
  ctx.fillRect(x, y0, w, y1 - y0)
  ctx.fillStyle = palette.pipeLight
  ctx.fillRect(x + 6, y0, 8, y1 - y0)
  ctx.fillStyle = palette.pipeShade
  ctx.fillRect(x + w - 12, y0, 12, y1 - y0)
  /* the cap at the gap's edge */
  const capY = fromTop ? y1 - CAP_H : y0
  ctx.fillStyle = palette.pipeLight
  ctx.fillRect(x - CAP_OVER, capY, w + CAP_OVER * 2, CAP_H)
  ctx.fillStyle = palette.pipe
  ctx.fillRect(x - CAP_OVER + 10, capY, w + CAP_OVER * 2 - 24, CAP_H)
  ctx.fillStyle = palette.pipeEdge
  ctx.fillRect(x - CAP_OVER, fromTop ? y1 - 3 : y0, w + CAP_OVER * 2, 3)
}
