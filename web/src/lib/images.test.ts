import { describe, expect, it } from "vitest"

import type { ChatMessage } from "./api"
import {
  SIZE_PRESETS, chatTurns, clampSteps, dataUrlBlob, duration, elapsedClock, imageFileName,
  parseSeed, parseSize, progressFraction, randomSeed, snapSide, validSide,
} from "./images"

describe("image sizes", () => {
  it("offers only sizes the engine accepts", () => {
    for (const preset of SIZE_PRESETS) {
      expect(validSide(preset.width) && validSide(preset.height), `${preset.width}x${preset.height}`).toBe(true)
    }
  })

  it("snaps a typed side to the nearest multiple of 32 inside 256..2048", () => {
    expect([1000, 432, 100, 5000, 768].map(snapSide)).toEqual([992, 448, 256, 2048, 768])
    expect(validSide(432)).toBe(false)
  })

  it("parses a stored size and refuses one the engine would refuse", () => {
    expect(parseSize("1024x576")).toEqual({ width: 1024, height: 576 })
    expect(parseSize("768x432")).toBeNull()
    expect(parseSize("garbage")).toBeNull()
    expect(parseSize(null)).toBeNull()
  })
})

describe("steps and seed", () => {
  it("keeps steps a whole number from 2 to 200 (the gateway's range)", () => {
    expect([0, 8, 8.6, 500, Number.NaN].map(clampSteps)).toEqual([2, 8, 9, 200, 8])
  })

  it("reads an empty seed as random and refuses anything that is not a whole number", () => {
    expect(parseSeed("")).toBeNull()
    expect(parseSeed("  42 ")).toBe(42)
    expect(parseSeed("-1")).toBeUndefined()
    expect(parseSeed("1.5")).toBeUndefined()
    expect(parseSeed("99999999999999999999")).toBeUndefined()
  })

  it("draws seeds that fit a signed 32-bit integer", () => {
    expect(randomSeed(() => 0)).toBe(0)
    expect(randomSeed(() => 1 - 2 ** -53)).toBe(2 ** 31 - 1)
  })
})

describe("progressFraction", () => {
  it("grows through encode, every denoising step and decode", () => {
    const at = (stage: "encode" | "denoise" | "decode", step: number) => progressFraction({ stage, step, steps: 8, elapsed: 0 })
    const series = [at("encode", 0), at("denoise", 0), at("denoise", 4), at("denoise", 8), at("decode", 8)]
    expect(series[0]).toBeGreaterThan(0)
    for (let i = 1; i < series.length; i++) expect(series[i]).toBeGreaterThanOrEqual(series[i - 1])
    expect(series[4]).toBeLessThan(1)
    expect(progressFraction(null)).toBe(0)
  })

  it("never leaves 0..1 on odd numbers from the server", () => {
    expect(progressFraction({ stage: "denoise", step: 50, steps: 8, elapsed: 0 })).toBeLessThan(1)
    expect(progressFraction({ stage: "denoise", step: -3, steps: 0, elapsed: 0 })).toBeGreaterThan(0)
  })
})

describe("transcript and files", () => {
  it("keeps image exchanges out of what a text model is sent", () => {
    const turns: ChatMessage[] = [
      { id: "1", role: "user", content: "hello" },
      { id: "2", role: "assistant", content: "hi" },
      { id: "3", role: "user", content: "a fox", mode: "image" },
      { id: "4", role: "assistant", content: "", mode: "image",
        generated: { prompt: "a fox", width: 512, height: 512, steps: 8, seed: 1 } },
      { id: "5", role: "user", content: "thanks" },
    ]
    expect(chatTurns(turns).map((item) => item.id)).toEqual(["1", "2", "5"])
  })

  it("formats the clock and the duration", () => {
    expect(elapsedClock(92.7)).toBe("1:32")
    expect(elapsedClock(4)).toBe("0:04")
    expect(duration(7.44)).toBe("7.4 s")
    expect(duration(42.2)).toBe("42 s")
    expect(duration(92.4)).toBe("1 min 32 s")
  })

  it("names the download after the prompt and the seed", () => {
    expect(imageFileName("Un colibrì, sotto la pioggia!", 42)).toBe("colibri-un-colibri-sotto-la-pioggia-42.png")
    expect(imageFileName("!!!", 7)).toBe("colibri-image-7.png")
  })

  it("turns a data URI back into the PNG bytes", async () => {
    const blob = dataUrlBlob("data:image/png;base64,iVBORw0KGgo=")
    expect(blob.type).toBe("image/png")
    expect([...new Uint8Array(await blob.arrayBuffer())]).toEqual([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])
  })
})

describe("parseSeed range", () => {
  it("refuses a seed above what the gateway accepts", () => {
    expect(parseSeed("4294967295")).toBe(4294967295)
    expect(parseSeed("4294967296")).toBeUndefined()
    expect(parseSeed("5000000000")).toBeUndefined()
  })
})
