import { describe, expect, it } from "vitest"

import { REASONING_EFFORT, modelForcesReasoning, reasoningLevelsFor } from "./reasoning"

describe("modelForcesReasoning", () => {
  it("is true for GLM 5.3 ids and false otherwise", () => {
    expect(modelForcesReasoning("glm-5.3-flash-colibri")).toBe(true)
    expect(modelForcesReasoning("glm-5.2-colibri")).toBe(false)
    expect(modelForcesReasoning("qwen3.8-colibri")).toBe(false)
  })
})

describe("reasoningLevelsFor", () => {
  it("offers the full ladder (incl. off and medium) for non-forced models", () => {
    expect(reasoningLevelsFor("glm-5.2-colibri")).toEqual([
      "off",
      "low",
      "medium",
      "high",
      "max",
    ])
  })

  it("drops off and the redundant medium for GLM 5.3 (template is Low/High/Max)", () => {
    const levels = reasoningLevelsFor("glm-5.3-flash-colibri")
    expect(levels).toEqual(["low", "high", "max"])
    expect(levels).not.toContain("off")
    expect(levels).not.toContain("medium")
  })
})

describe("REASONING_EFFORT", () => {
  it("maps max onto the engine's xhigh effort", () => {
    expect(REASONING_EFFORT.max).toBe("xhigh")
    expect(REASONING_EFFORT.low).toBe("low")
  })
})
