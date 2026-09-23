import { useState } from "react"
import { Check, Copy } from "lucide-react"

/* A small Markdown subset, rendered as React nodes rather than HTML.
 *
 * The text comes from a model, so nothing here ever reaches innerHTML: every
 * piece of content ends up as a React text node, which escapes by construction.
 * That rules out the whole class of "the model wrote a script tag" problems
 * without a sanitiser to keep honest.
 *
 * Covered, because it is what a model actually emits in chat: fenced code with
 * its language and a copy button, inline code, bold, italic, links, headings,
 * ordered and unordered lists, blockquotes and horizontal rules. Not covered:
 * tables, footnotes, raw HTML. Those degrade to the literal text instead of
 * disappearing, which is the right failure for a reader.
 */

type Segment = { kind: "code"; lang: string; body: string } | { kind: "text"; body: string }

/** Split on fenced blocks first: everything inside a fence is literal. */
function segments(source: string): Segment[] {
  const out: Segment[] = []
  const fence = /```([^\n`]*)\n?([\s\S]*?)(?:```|$)/g
  let last = 0
  for (let match = fence.exec(source); match; match = fence.exec(source)) {
    if (match.index > last) out.push({ kind: "text", body: source.slice(last, match.index) })
    out.push({ kind: "code", lang: match[1].trim(), body: match[2].replace(/\n$/, "") })
    last = fence.lastIndex
  }
  if (last < source.length) out.push({ kind: "text", body: source.slice(last) })
  return out
}

/** Inline marks, innermost first so `**a *b* **` does not swallow its own stars. */
function inline(text: string, key = "i"): React.ReactNode[] {
  const pattern = /(`[^`\n]+`)|(\*\*[^*\n]+\*\*)|(\*[^*\n]+\*)|(\[[^\]\n]+\]\([^)\s]+\))/
  const nodes: React.ReactNode[] = []
  let rest = text
  let index = 0
  for (let match = pattern.exec(rest); match; match = pattern.exec(rest)) {
    if (match.index > 0) nodes.push(rest.slice(0, match.index))
    const token = match[0]
    const id = `${key}-${index++}`
    if (token.startsWith("`")) nodes.push(<code key={id}>{token.slice(1, -1)}</code>)
    else if (token.startsWith("**")) nodes.push(<strong key={id}>{token.slice(2, -2)}</strong>)
    else if (token.startsWith("*")) nodes.push(<em key={id}>{token.slice(1, -1)}</em>)
    else {
      const cut = token.indexOf("](")
      const label = token.slice(1, cut)
      const href = token.slice(cut + 2, -1)
      const safe = /^(https?:|mailto:)/i.test(href)
      nodes.push(safe
        ? <a key={id} href={href} target="_blank" rel="noreferrer noopener">{label}</a>
        : <span key={id}>{label}</span>)
    }
    rest = rest.slice(match.index + token.length)
  }
  if (rest) nodes.push(rest)
  return nodes
}

function CodeBlock({ lang, body }: { lang: string; body: string }) {
  const [copied, setCopied] = useState(false)
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(body)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1200)
    } catch { /* a denied clipboard is not worth an error state here */ }
  }
  return <figure className="md-code">
    <figcaption>
      <span>{lang || "text"}</span>
      <button type="button" onClick={() => void copy()} aria-label="Copy code">
        {copied ? <Check /> : <Copy />}
      </button>
    </figcaption>
    <pre><code>{body}</code></pre>
  </figure>
}

/** One text segment: block structure, then inline marks inside each block. */
function blocks(source: string, key: string): React.ReactNode[] {
  const out: React.ReactNode[] = []
  const lines = source.split("\n")
  let paragraph: string[] = []
  let list: { ordered: boolean; items: string[] } | null = null
  let quote: string[] = []

  const flushParagraph = () => {
    if (!paragraph.length) return
    out.push(<p key={`${key}-p${out.length}`}>{inline(paragraph.join(" "), `${key}-p${out.length}`)}</p>)
    paragraph = []
  }
  const flushList = () => {
    if (!list) return
    const items = list.items.map((item, at) => <li key={at}>{inline(item, `${key}-li${at}`)}</li>)
    out.push(list.ordered ? <ol key={`${key}-l${out.length}`}>{items}</ol>
                          : <ul key={`${key}-l${out.length}`}>{items}</ul>)
    list = null
  }
  const flushQuote = () => {
    if (!quote.length) return
    out.push(<blockquote key={`${key}-q${out.length}`}>{inline(quote.join(" "), `${key}-q${out.length}`)}</blockquote>)
    quote = []
  }
  const flushAll = () => { flushParagraph(); flushList(); flushQuote() }

  for (const line of lines) {
    const heading = /^(#{1,4})\s+(.*)$/.exec(line)
    const bullet = /^\s*[-*+]\s+(.*)$/.exec(line)
    const numbered = /^\s*\d+[.)]\s+(.*)$/.exec(line)
    const quoted = /^>\s?(.*)$/.exec(line)

    if (!line.trim()) { flushAll(); continue }
    if (/^\s*([-*_])\s*\1\s*\1[\s\-*_]*$/.test(line)) { flushAll(); out.push(<hr key={`${key}-hr${out.length}`} />); continue }
    if (heading) {
      flushAll()
      const level = heading[1].length
      const Tag = (["h2", "h3", "h4", "h5"][level - 1] || "h5") as "h2"
      out.push(<Tag key={`${key}-h${out.length}`}>{inline(heading[2], `${key}-h${out.length}`)}</Tag>)
      continue
    }
    if (quoted) { flushParagraph(); flushList(); quote.push(quoted[1]); continue }
    if (bullet || numbered) {
      flushParagraph(); flushQuote()
      const ordered = Boolean(numbered)
      const item = (numbered || bullet)![1]
      if (!list || list.ordered !== ordered) { flushList(); list = { ordered, items: [] } }
      list.items.push(item)
      continue
    }
    flushList(); flushQuote()
    paragraph.push(line.trim())
  }
  flushAll()
  return out
}

export function Markdown({ text }: { text: string }) {
  return <div className="md">
    {segments(text).map((segment, at) => segment.kind === "code"
      ? <CodeBlock key={at} lang={segment.lang} body={segment.body} />
      : <div key={at}>{blocks(segment.body, `s${at}`)}</div>)}
  </div>
}
