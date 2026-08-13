import { describe, expect, it } from "vitest"
import { highlight, parseInline, parseMarkdown, type Block } from "./markdown"

const kinds = (blocks: Block[]) => blocks.map((b) => b.kind)

describe("parseInline", () => {
  it("hides the markers, not the text", () => {
    expect(parseInline("a **b** c")).toEqual([
      { kind: "text", text: "a " },
      { kind: "bold", text: "b" },
      { kind: "text", text: " c" },
    ])
    expect(parseInline("run `make check` now")).toEqual([
      { kind: "text", text: "run " },
      { kind: "code", text: "make check" },
      { kind: "text", text: " now" },
    ])
  })

  it("leaves a single asterisk alone", () => {
    // 3 * 4, a glob, a footnote. The TUI makes the same call for the same
    // reason: emphasis with one asterisk is rarer than multiplication.
    expect(parseInline("3 * 4 = 12")).toEqual([{ kind: "text", text: "3 * 4 = 12" }])
  })

  it("does not swallow the line on an unterminated marker", () => {
    // Mid-stream the model has typed the opening backtick and not the closing
    // one. Eating the rest of the line into a code span would make the answer
    // visibly lurch as the next chunk arrives.
    expect(parseInline("an unclosed ` backtick")).toEqual([
      { kind: "text", text: "an unclosed ` backtick" },
    ])
    expect(parseInline("an unclosed ** bold")).toEqual([
      { kind: "text", text: "an unclosed ** bold" },
    ])
  })
})

describe("parseMarkdown", () => {
  it("recognises the shapes the TUI recognises", () => {
    const md = [
      "# Title",
      "",
      "Some **prose**.",
      "",
      "- one",
      "- two",
      "1. first",
      "",
      "```python",
      "def f():",
      "    return 1",
      "```",
      "",
      "after",
    ].join("\n")
    expect(kinds(parseMarkdown(md))).toEqual(["h", "p", "li", "li", "li", "code", "p"])
  })

  it("keeps fenced text verbatim and records the language", () => {
    const blocks = parseMarkdown("```python\nx = '# not a heading'\n```")
    expect(blocks).toHaveLength(1)
    const block = blocks[0]
    if (block.kind !== "code") throw new Error("expected a code block")
    expect(block.lang).toBe("python")
    expect(block.text).toBe("x = '# not a heading'")
    expect(block.open).toBe(false)
  })

  it("treats an unclosed fence as open to the end", () => {
    // THE streaming case: the answer stops inside the code block. Leaving the
    // fence unrecognised would print ``` into the prose and render the code as
    // a paragraph, then reflow it when the closing fence arrives.
    const blocks = parseMarkdown("intro\n\n```c\nint main(void) {")
    expect(kinds(blocks)).toEqual(["p", "code"])
    const block = blocks[1]
    if (block.kind !== "code") throw new Error("expected a code block")
    expect(block.open).toBe(true)
    expect(block.text).toBe("int main(void) {")
  })

  it("does not interpret markdown inside code", () => {
    const blocks = parseMarkdown("```\n**not bold** and `not code`\n```")
    const block = blocks[0]
    if (block.kind !== "code") throw new Error("expected a code block")
    expect(block.text).toBe("**not bold** and `not code`")
  })

  it("survives an empty string and a bare fence", () => {
    expect(parseMarkdown("")).toEqual([])
    expect(kinds(parseMarkdown("```"))).toEqual(["code"])
  })
})

describe("highlight", () => {
  it("classifies keywords, strings, numbers and comments", () => {
    const toks = highlight('def f():\n    return "x"  # note\n', "python")
    const of = (k: string) => toks.filter((t) => t.kind === k).map((t) => t.text)
    expect(of("kw")).toContain("def")
    expect(of("kw")).toContain("return")
    expect(of("str")).toContain('"x"')
    expect(of("com")).toContain("# note")
  })

  it("uses the language's own comment marker", () => {
    // '#' opens a comment in python and is an operator nowhere in C; getting
    // this wrong paints half a C file grey.
    expect(highlight("int x = 1; // c", "c").some((t) => t.kind === "com" && t.text === "// c")).toBe(true)
    expect(highlight("#include <stdio.h>", "c").some((t) => t.kind === "com")).toBe(false)
    expect(highlight("# a python comment", "python").some((t) => t.kind === "com")).toBe(true)
  })

  it("stops an unterminated string at the newline", () => {
    // Mid-stream code is full of these. Painting the rest of the block as a
    // string would make every subsequent line change colour as it arrives.
    const toks = highlight('s = "open\nnext = 1\n', "python")
    const str = toks.find((t) => t.kind === "str")
    expect(str?.text).toBe('"open')
    expect(toks.some((t) => t.kind === "num" && t.text === "1")).toBe(true)
  })

  it("leaves an unknown language plain rather than guessing", () => {
    const toks = highlight("def f(): pass", "brainfuck")
    expect(toks.every((t) => t.kind !== "kw")).toBe(true)
  })

  it("is lossless: the concatenated tokens are the input", () => {
    // The property that matters most. A highlighter that drops or duplicates a
    // character corrupts the code the user is about to copy.
    for (const [src, lang] of [
      ['def f():\n  return "a" # c\n', "python"],
      ["int main(void) { /* x */ return 0; }", "c"],
      ["let x = `t${1}`; // y", "ts"],
      ["", "python"],
      ["no language here", ""],
    ] as const) {
      expect(highlight(src, lang).map((t) => t.text).join("")).toBe(src)
    }
  })
})
