import { Check, Copy } from "lucide-react"
import { useCallback, useMemo, useState } from "react"
import { highlight, parseMarkdown, type Block, type Span } from "@/lib/markdown"

/* Renders an assistant turn the way `coli chat`'s TUI already does: fenced code
 * in a labelled box, real bold, coloured inline code, headings and bullets, and
 * no marker ever visible.
 *
 * Everything below builds React ELEMENTS. Nothing reaches innerHTML, so model
 * output cannot become markup no matter what it contains — the property that
 * makes this safe to point at an untrusted generation without a sanitiser.
 */

function Inline({ spans }: { spans: Span[] }) {
  return (
    <>
      {spans.map((span, index) => {
        if (span.kind === "bold") return <strong key={index}>{span.text}</strong>
        if (span.kind === "code") return <code key={index} className="md-inline">{span.text}</code>
        return <span key={index}>{span.text}</span>
      })}
    </>
  )
}

function CodeBlock({ lang, text, open }: { lang: string; text: string; open: boolean }) {
  const [copied, setCopied] = useState(false)
  const tokens = useMemo(() => highlight(text, lang), [text, lang])

  const copy = useCallback(() => {
    // Clipboard is unavailable over plain http on a non-localhost origin, which
    // is exactly how `coli web --host <lan-ip>` is used. Fail quietly to the
    // textarea path rather than throwing into the console.
    const done = () => { setCopied(true); window.setTimeout(() => setCopied(false), 1200) }
    if (navigator.clipboard?.writeText) {
      navigator.clipboard.writeText(text).then(done, () => undefined)
      return
    }
    const area = document.createElement("textarea")
    area.value = text
    area.setAttribute("readonly", "")
    area.style.position = "fixed"
    area.style.opacity = "0"
    document.body.appendChild(area)
    area.select()
    try { document.execCommand("copy"); done() } catch { /* nothing to do */ }
    document.body.removeChild(area)
  }, [text])

  return (
    <div className="md-code">
      <div className="md-code-head">
        <span>{lang || "code"}</span>
        <button type="button" onClick={copy} aria-label={copied ? "Copied" : "Copy code"}>
          {copied ? <Check className="size-3" /> : <Copy className="size-3" />}
        </button>
      </div>
      <pre>
        <code>
          {tokens.map((token, index) =>
            token.kind === "plain"
              ? <span key={index}>{token.text}</span>
              : <span key={index} className={`tok-${token.kind}`}>{token.text}</span>,
          )}
          {/* An unclosed fence means the answer is still arriving inside it.
              A caret says "more is coming" instead of leaving the box looking
              finished and truncated. */}
          {open ? <span className="md-caret" /> : null}
        </code>
      </pre>
    </div>
  )
}

export function Markdown({ source }: { source: string }) {
  const blocks = useMemo<Block[]>(() => parseMarkdown(source), [source])
  return (
    <div className="md">
      {blocks.map((block, index) => {
        if (block.kind === "code")
          return <CodeBlock key={index} lang={block.lang} text={block.text} open={block.open} />
        if (block.kind === "h") {
          const Tag = (`h${Math.min(block.level + 2, 6)}`) as "h3" | "h4" | "h5" | "h6"
          return <Tag key={index} className="md-h"><Inline spans={block.spans} /></Tag>
        }
        if (block.kind === "li")
          return (
            <div key={index} className="md-li">
              <span className="md-bullet">{block.ordered ? "›" : "•"}</span>
              <span><Inline spans={block.spans} /></span>
            </div>
          )
        return <p key={index} className="md-p"><Inline spans={block.spans} /></p>
      })}
    </div>
  )
}
