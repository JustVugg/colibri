import { describe, expect, it } from "vitest"

import { parseTableAt, rowAlignments, splitRow } from "./markdown-table"

describe("splitting a row into cells", () => {
  it("treats the outer pipes as delimiters, not as empty edge cells", () => {
    expect(splitRow("| a | b |")).toEqual(["a", "b"])
    expect(splitRow("a | b")).toEqual(["a", "b"])
  })

  it("keeps a genuinely empty cell", () => {
    expect(splitRow("| a |  | c |")).toEqual(["a", "", "c"])
  })

  it("reads a backslash-escaped pipe as content", () => {
    expect(splitRow("| a \\| b | c |")).toEqual(["a | b", "c"])
  })

  it("does not strip a trailing escaped pipe as a delimiter", () => {
    expect(splitRow("| a | b \\|")).toEqual(["a", "b |"])
  })
})

describe("recognising the delimiter row", () => {
  it("accepts dashes with optional colon anchors and reads the alignment", () => {
    expect(rowAlignments("|---|:---|---:|:---:|")).toEqual([null, "left", "right", "center"])
  })

  it("accepts a single dash per cell and no outer pipes", () => {
    expect(rowAlignments("- | -")).toEqual([null, null])
  })

  it("rejects a row that is not all dashes, so ordinary text with a pipe stays text", () => {
    expect(rowAlignments("| a | b |")).toBeNull()
    expect(rowAlignments("|--- | x |")).toBeNull()
    expect(rowAlignments("")).toBeNull()
    expect(rowAlignments(undefined as unknown as string)).toBeNull()
  })
})

describe("parsing a table out of a line array", () => {
  it("reads the header, alignments, body and the lines it consumed", () => {
    const lines = ["| Phase | ms |", "|---|---:|", "| moe | 126.8 |", "| attn | 12.2 |", "", "after"]
    const table = parseTableAt(lines, 0)
    expect(table).not.toBeNull()
    expect(table!.head).toEqual(["Phase", "ms"])
    expect(table!.align).toEqual([null, "right"])
    expect(table!.rows).toEqual([["moe", "126.8"], ["attn", "12.2"]])
    expect(table!.length).toBe(4)
  })

  it("pads a short row so the grid holds", () => {
    const lines = ["| a | b | c |", "|---|---|---|", "| 1 |"]
    expect(parseTableAt(lines, 0)!.rows).toEqual([["1", "", ""]])
  })

  it("keeps the extra cells of a long row rather than dropping content", () => {
    const lines = ["| a | b |", "|---|---|", "| 1 | 2 | 3 |"]
    expect(parseTableAt(lines, 0)!.rows).toEqual([["1", "2", "3"]])
  })

  it("stops at a blank line and at the first line without a pipe", () => {
    const lines = ["| a |", "|---|", "| 1 |", "prose follows", "| not | part |"]
    expect(parseTableAt(lines, 0)!.length).toBe(3)
  })

  it("is null when the second line is not a delimiter", () => {
    expect(parseTableAt(["a | b", "c | d"], 0)).toBeNull()
  })

  it("is null for a line with no pipe, and past the end", () => {
    expect(parseTableAt(["plain", "|---|"], 0)).toBeNull()
    expect(parseTableAt(["| a |", "|---|"], 9)).toBeNull()
  })

  it("accepts a header-only table with no body rows", () => {
    const table = parseTableAt(["| a | b |", "|---|---|"], 0)
    expect(table!.rows).toEqual([])
    expect(table!.length).toBe(2)
  })

  it("finds a table that starts partway down", () => {
    const lines = ["intro", "", "| a |", "|---|", "| 1 |"]
    expect(parseTableAt(lines, 2)!.head).toEqual(["a"])
  })
})
