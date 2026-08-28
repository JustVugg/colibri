/* Markdown for the chat pane, matching what `coli chat`'s TUI already renders.
 *
 * The TUI (c/coli, class MD) turns ``` into boxed blocks with a language label,
 * **x** into real bold, `x` into coloured inline code, # into headings and "- "
 * into real bullets, and never shows a marker. The web pane showed the raw
 * string: every asterisk, backtick and hash literal, and code as one grey
 * paragraph. This closes that gap.
 *
 * WHY THIS IS PARSED RATHER THAN PULLED IN
 *
 * react-markdown would do it, and brings the whole unified/remark tree with it —
 * roughly forty transitive packages to render bold text, in a project whose six
 * dependencies are all small utilities. It also renders through HTML, which for
 * model output is an XSS surface we would then have to sanitise. The renderer
 * that consumes this builds React elements instead, so untrusted output cannot
 * become markup by construction.
 *
 * STREAMING
 *
 * The TUI holds back partial markers because it writes to a terminal it cannot
 * take back. React re-renders from the accumulated string every chunk, so that
 * problem does not exist here — but an UNCLOSED fence does: mid-stream the text
 * ends inside a code block. `parseMarkdown` treats a fence with no closing pair
 * as open to the end, which is what the user is actually looking at, rather than
 * leaking ``` into the prose.
 */

export type Span =
  | { kind: "text"; text: string }
  | { kind: "bold"; text: string }
  | { kind: "code"; text: string }

export type Block =
  | { kind: "p"; spans: Span[] }
  | { kind: "h"; level: number; spans: Span[] }
  | { kind: "li"; spans: Span[]; ordered: boolean }
  | { kind: "code"; lang: string; text: string; open: boolean }

const FENCE = /^\s*```/

/** Inline: **bold** and `code`. Single * is left alone — it is multiplication,
 *  a glob, or a footnote far more often than it is emphasis, and the TUI makes
 *  the same call for the same reason. */
export function parseInline(line: string): Span[] {
  const spans: Span[] = []
  let text = ""
  let i = 0
  const flush = () => {
    if (text) { spans.push({ kind: "text", text }); text = "" }
  }
  while (i < line.length) {
    const ch = line[i]
    if (ch === "`") {
      const end = line.indexOf("`", i + 1)
      if (end > i) {
        flush()
        spans.push({ kind: "code", text: line.slice(i + 1, end) })
        i = end + 1
        continue
      }
      // unterminated: a backtick the model has not closed yet. Show it as text
      // rather than swallowing the rest of the line into a code span.
      text += ch; i++; continue
    }
    if (ch === "*" && line[i + 1] === "*") {
      const end = line.indexOf("**", i + 2)
      if (end > i + 1) {
        flush()
        spans.push({ kind: "bold", text: line.slice(i + 2, end) })
        i = end + 2
        continue
      }
      text += "**"; i += 2; continue
    }
    text += ch; i++
  }
  flush()
  return spans
}

export function parseMarkdown(source: string): Block[] {
  const lines = source.split("\n")
  const blocks: Block[] = []
  let code: { lang: string; body: string[] } | null = null
  let para: string[] = []

  const flushPara = () => {
    if (!para.length) return
    blocks.push({ kind: "p", spans: parseInline(para.join("\n")) })
    para = []
  }

  for (const line of lines) {
    if (FENCE.test(line)) {
      if (code) {
        blocks.push({ kind: "code", lang: code.lang, text: code.body.join("\n"), open: false })
        code = null
      } else {
        flushPara()
        code = { lang: line.trim().slice(3).trim().replace(/`/g, ""), body: [] }
      }
      continue
    }
    if (code) { code.body.push(line); continue }

    const trimmed = line.trim()
    if (!trimmed) { flushPara(); continue }

    const heading = /^(#{1,6})\s+(.*)$/.exec(trimmed)
    if (heading) {
      flushPara()
      blocks.push({ kind: "h", level: heading[1].length, spans: parseInline(heading[2]) })
      continue
    }
    const bullet = /^[-*+]\s+(.*)$/.exec(trimmed)
    if (bullet) {
      flushPara()
      blocks.push({ kind: "li", spans: parseInline(bullet[1]), ordered: false })
      continue
    }
    const numbered = /^\d+[.)]\s+(.*)$/.exec(trimmed)
    if (numbered) {
      flushPara()
      blocks.push({ kind: "li", spans: parseInline(numbered[1]), ordered: true })
      continue
    }
    para.push(line)
  }

  // Mid-stream the answer ends inside the fence it has not closed yet. Emit it
  // as an open code block: that is what the user is looking at, and it keeps the
  // ``` out of the prose.
  if (code) blocks.push({ kind: "code", lang: code.lang, text: code.body.join("\n"), open: true })
  flushPara()
  return blocks
}

/* ---- syntax highlighting -------------------------------------------------
 *
 * Deliberately approximate, and small. A real grammar per language is what
 * highlight.js is for, and it is 100 KB+ for the set people actually paste.
 * Comments, strings, numbers and keywords carry nearly all the readability, so
 * that is what this does; anything it cannot classify stays plain, which
 * degrades to "unhighlighted code" rather than to "wrong colours".
 */

export type Tok = { kind: "plain" | "kw" | "str" | "num" | "com"; text: string }

const KEYWORDS: Record<string, string[]> = {
  python: "def class return if elif else for while in not and or is None True False import from as with try except finally raise lambda yield global nonlocal pass break continue assert del await async".split(" "),
  c: "int char void float double long short unsigned signed const static struct union enum typedef return if else for while do switch case break continue sizeof goto extern inline volatile".split(" "),
  rust: "fn let mut const struct enum impl trait pub use mod match if else for while loop return self Self where async await move ref dyn type unsafe".split(" "),
  go: "func var const type struct interface package import return if else for range switch case defer go chan map select break continue fallthrough".split(" "),
  js: "function const let var return if else for while class extends new this import export from default async await try catch finally throw typeof instanceof null undefined true false".split(" "),
  sh: "if then else elif fi for while do done case esac function return export local readonly set unset echo cd exit".split(" "),
}
KEYWORDS.cpp = KEYWORDS.c
KEYWORDS.h = KEYWORDS.c
KEYWORDS.typescript = KEYWORDS.js
KEYWORDS.ts = KEYWORDS.js
KEYWORDS.tsx = KEYWORDS.js
KEYWORDS.javascript = KEYWORDS.js
KEYWORDS.jsx = KEYWORDS.js
KEYWORDS.py = KEYWORDS.python
KEYWORDS.bash = KEYWORDS.sh
KEYWORDS.shell = KEYWORDS.sh
KEYWORDS.zsh = KEYWORDS.sh

const LINE_COMMENT: Record<string, string> = {
  python: "#", py: "#", sh: "#", bash: "#", shell: "#", zsh: "#", yaml: "#", yml: "#", toml: "#",
}

export function highlight(text: string, lang: string): Tok[] {
  const key = (lang || "").toLowerCase()
  const words = KEYWORDS[key]
  const lineComment = LINE_COMMENT[key] ?? "//"
  const out: Tok[] = []
  let plain = ""
  const flush = () => { if (plain) { out.push({ kind: "plain", text: plain }); plain = "" } }

  let i = 0
  while (i < text.length) {
    const rest = text.slice(i)

    if (rest.startsWith(lineComment) || (lineComment === "//" && rest.startsWith("/*"))) {
      const end = rest.startsWith("/*")
        ? (text.indexOf("*/", i + 2) < 0 ? text.length : text.indexOf("*/", i + 2) + 2)
        : (text.indexOf("\n", i) < 0 ? text.length : text.indexOf("\n", i))
      flush(); out.push({ kind: "com", text: text.slice(i, end) }); i = end; continue
    }

    const q = text[i]
    if (q === '"' || q === "'" || q === "`") {
      let j = i + 1
      while (j < text.length && text[j] !== q) { if (text[j] === "\\") j++; j++ }
      // Unterminated string: stop at the newline rather than painting the rest
      // of the block. Mid-stream code is full of these.
      const nl = text.indexOf("\n", i)
      const end = j < text.length ? j + 1 : (nl < 0 ? text.length : nl)
      flush(); out.push({ kind: "str", text: text.slice(i, end) }); i = end; continue
    }

    if (/[0-9]/.test(q) && !/[A-Za-z_]/.test(text[i - 1] || "")) {
      let j = i
      while (j < text.length && /[0-9a-fA-FxXoObB._]/.test(text[j])) j++
      flush(); out.push({ kind: "num", text: text.slice(i, j) }); i = j; continue
    }

    if (words && /[A-Za-z_]/.test(q)) {
      let j = i
      while (j < text.length && /[A-Za-z0-9_]/.test(text[j])) j++
      const word = text.slice(i, j)
      if (words.includes(word)) { flush(); out.push({ kind: "kw", text: word }); i = j; continue }
      plain += word; i = j; continue
    }

    plain += q; i++
  }
  flush()
  return out
}
