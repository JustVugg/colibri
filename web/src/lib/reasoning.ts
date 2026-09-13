/* Reasoning depth for GLM. The engine renders enable_thinking + reasoning_effort;
   these levels map onto the words GLM understands (Low/Medium/High/Max), and "off"
   turns thinking off entirely. GLM 5.3 cannot disable reasoning, and its template
   exposes only Low/High/Max (it renders "medium" as High). So for 5.3 both "off" and
   the now-redundant "medium" are dropped -- mirroring the model's own constraint in
   the control instead of offering a level it silently collapses onto High. */
export type ReasoningLevel = "off" | "low" | "medium" | "high" | "max"

export const REASONING_EFFORT: Record<Exclude<ReasoningLevel, "off">, string> = {
  low: "low", medium: "medium", high: "high", max: "xhigh",
}

export const modelForcesReasoning = (model: string) => /5\.3/.test(model)

export const reasoningLevelsFor = (model: string): ReasoningLevel[] =>
  modelForcesReasoning(model)
    ? ["low", "high", "max"]
    : ["off", "low", "medium", "high", "max"]
