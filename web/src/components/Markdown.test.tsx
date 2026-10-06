import { renderToStaticMarkup } from "react-dom/server"
import { describe, expect, it } from "vitest"

import { Markdown } from "./Markdown"

/* Static rendering only — no DOM needed, so this runs in the default vitest
   environment alongside the lib tests. */
const render = (text: string) => renderToStaticMarkup(<Markdown text={text} />)

describe("rendering a chat table", () => {
  const table = [
    "| Phase | ms/token |",
    "|---|---:|",
    "| moe_total | 126.8 |",
    "| deltanet | 34.8 |",
  ].join("\n")

  it("renders a real table rather than a run-on paragraph", () => {
    const html = render(table)
    expect(html).toContain("<table>")
    expect(html).toContain("<th")
    expect(html).toContain("moe_total")
    // The regression this fixes: rows joined into one paragraph.
    expect(html).not.toContain("| moe_total | 126.8 |")
  })

  it("carries the column alignment from the delimiter row", () => {
    expect(render(table)).toContain('style="text-align:right"')
  })

  it("renders inline marks inside cells", () => {
    const html = render("| a | b |\n|---|---|\n| **bold** | `code` |")
    expect(html).toContain("<strong>bold</strong>")
    expect(html).toContain("<code>code</code>")
  })

  it("keeps surrounding prose as its own blocks", () => {
    const html = render(`Before.\n\n${table}\n\nAfter.`)
    expect(html).toContain("<p>Before.</p>")
    expect(html).toContain("<p>After.</p>")
    expect(html).toContain("<table>")
  })

  it("leaves a pipe table inside a fence as literal code", () => {
    const html = render("```\n| a | b |\n|---|---|\n```")
    expect(html).not.toContain("<table>")
    expect(html).toContain("| a | b |")
  })

  it("does not read a delimiter row as a list item", () => {
    expect(render(table)).not.toContain("<li>")
  })
})

describe("the rest of the subset still renders", () => {
  it("keeps ordinary text with a single pipe as a paragraph", () => {
    const html = render("choose a | b at the prompt")
    expect(html).toContain("<p>")
    expect(html).not.toContain("<table>")
  })

  it("still renders lists, headings and fenced code", () => {
    const html = render("# One\n\n## Two\n\n- one\n- two\n\n```sh\nmake\n```")
    // Headings are demoted by one level so chat content sits under the page's own h1.
    expect(html).toContain("<h2>One</h2>")
    expect(html).toContain("<h3>Two</h3>")
    expect(html).toContain("<ul>")
    expect(html).toContain("<li>")
    expect(html).toContain("make")
  })

  it("escapes model-written markup instead of injecting it", () => {
    const html = render("| x |\n|---|\n| <img src=x onerror=alert(1)> |")
    expect(html).not.toContain("<img")
    expect(html).toContain("&lt;img")
  })
})
