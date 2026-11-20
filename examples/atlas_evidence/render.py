"""Offline review page. Untrusted source and model text is always escaped.

Copyright 2026 Albert Tan Sin Hon. Licensed under Apache-2.0.
"""
from html import escape
from .core import assess


def render_html(capture, current_sources=None, min_quote_chars=1):
    report = assess(capture, current_sources, min_quote_chars)
    sources = {s["id"]: s for s in capture["sources"]}
    passed = sum(item["status"] == "span_verified" for item in report["claims"])
    cards = []
    for number, item in enumerate(report["claims"], 1):
        candidate = item["candidate"]
        if not isinstance(candidate, dict):
            candidate = {"text": str(candidate)}
        source = sources.get(candidate.get("source_id")) if isinstance(candidate.get("source_id"), str) else None
        text = escape(str(candidate.get("text", "Malformed candidate")))
        quote = escape(str(candidate.get("quote", "No quotation supplied")))
        good = item["status"] == "span_verified"
        badge = "Exact passage matched" if good else "Check failed"
        kind = "good" if good else "bad"
        reasons = "".join("<li>" + escape(reason) + "</li>" for reason in item["reasons"])
        source_label = escape(str(candidate.get("source_id", "Unknown source")))
        offsets = escape(f"{candidate.get('start', '?')}:{candidate.get('end', '?')}")
        context = ""
        if source:
            start, end = candidate.get("start"), candidate.get("end")
            if type(start) is int and type(end) is int and 0 <= start < end <= len(source["text"]):
                before = escape(source["text"][max(0, start - 200):start])
                selected = escape(source["text"][start:end])
                after = escape(source["text"][end:min(len(source["text"]), end + 200)])
                context = f'<details><summary>Inspect captured source context</summary><p class="context">{before}<mark>{selected}</mark>{after}</p></details>'
        cards.append(f'<article><div class="card-head"><span class="number">{number:02d}</span><span class="badge {kind}">{badge}</span></div><h2>{text}</h2><blockquote>{quote}</blockquote><p class="meta">Source: {source_label} · Unicode codepoints {offsets}, end exclusive</p><ul class="errors">{reasons}</ul>{context}<p class="human">Human review: does this passage actually support the proposed claim?</p></article>')
    source_rows = []
    for source in capture["sources"]:
        current_hash = report["current_source_hashes"].get(source["id"])
        state = "Same supplied text" if current_hash == source["sha256"] else ("Changed supplied text" if current_hash else "Missing current source")
        source_rows.append(f'<tr><td>{escape(source["id"])}</td><td>{escape(source["title"] or source["id"])}</td><td>{state}</td><td><code>{source["sha256"]}</code></td></tr>')
    errors = "".join("<li>" + escape(error) + "</li>" for error in report["errors"])
    overall = "Passage checks passed" if report["passed"] else "Review required"
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>Evidence Replay: {overall}</title><style>
:root{{color-scheme:light;--ink:#152b3d;--muted:#526575;--paper:#f4f3ef;--line:#d7deda;--green:#146347;--red:#9b342d}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.6 system-ui,sans-serif}}main{{max-width:1080px;margin:auto;padding:48px 24px}}header{{border-top:5px solid var(--ink);padding:24px 0}}.eyebrow{{font-size:12px;font-weight:700;letter-spacing:.18em;text-transform:uppercase}}h1{{font-size:clamp(32px,5vw,56px);line-height:1.1;margin:16px 0}}.intro{{max-width:760px;color:var(--muted)}}.summary{{display:flex;gap:20px;flex-wrap:wrap;border-block:1px solid var(--line);padding:20px 0;margin-bottom:28px}}.stat{{min-width:160px}}.stat strong{{display:block;font-size:27px}}.stat span{{font-size:13px;color:var(--muted)}}article{{background:white;border:1px solid var(--line);padding:28px;margin:20px 0;border-radius:5px}}.card-head{{display:flex;justify-content:space-between;align-items:center}}.number{{color:var(--muted);font-size:14px}}h2{{font-size:22px;line-height:1.4}}.badge{{font-size:12px;font-weight:700;border-radius:30px;padding:4px 12px}}.good{{background:#e4f1e8;color:var(--green)}}.bad{{background:#fae8e5;color:var(--red)}}blockquote{{margin:20px 0;padding:12px 18px;border-left:3px solid #9daea6;background:#f5f7f4;white-space:pre-wrap;overflow-wrap:anywhere}}.meta,.human{{font-size:13px;color:var(--muted)}}.human{{margin-top:22px;border-top:1px solid var(--line);padding-top:14px}}.errors{{color:var(--red)}}summary{{cursor:pointer;font-weight:600}}.context{{white-space:pre-wrap;overflow-wrap:anywhere}}mark{{background:#fff0ab;color:var(--ink)}}.table-wrap{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border-bottom:1px solid var(--line);padding:12px;text-align:left;vertical-align:top}}code{{font-size:11px;overflow-wrap:anywhere}}footer{{margin-top:40px;font-size:12px;color:var(--muted)}}@media print{{body{{background:white}}main{{padding:0}}article{{break-inside:avoid}}details{{display:block}}}}
</style></head><body><main><header><div class="eyebrow">Colibri × Atlas / Evidence Replay</div><h1>{overall}</h1><p class="intro">Inspect each proposed claim beside its recorded source passage. These checks establish an exact text match and source version. Meaning, accuracy, authenticity and completeness remain for human review.</p></header>
<section class="summary" aria-label="Review summary"><div class="stat"><strong>{passed} / {len(report['claims'])}</strong><span>Candidate passages matched</span></div><div class="stat"><strong>{len(capture['sources'])}</strong><span>Explicit source snapshots</span></div><div class="stat"><strong>Unreviewed</strong><span>Semantic support</span></div><div class="stat"><strong>{escape(capture['mode'].title())}</strong><span>Capture mode</span></div></section>
<ul class="errors">{errors}</ul>{''.join(cards)}
<section><h2>Source versions</h2><p class="meta">Current means the snapshots supplied for this check. This page does not monitor websites or read later file changes.</p><div class="table-wrap"><table><thead><tr><th>ID</th><th>Title</th><th>Current comparison</th><th>Captured SHA-256</th></tr></thead><tbody>{''.join(source_rows)}</tbody></table></div></section>
<footer><p>Capture: <code>{escape(capture['sha256'])}</code><br>Recorded: {escape(capture['created_at'])}<br>Minimum quotation length: {min_quote_chars} Unicode codepoints.</p><p>Offline review page. No scripts, external requests or automatic publication. Hashes check consistency, not authorship or protection against someone replacing both the data and its hashes.</p></footer></main></body></html>'''
