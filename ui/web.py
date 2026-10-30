"""Web search for the chat (the page's search step): results for a query, then the readable text
of the pages the model chooses to open. Standard library only.

The results come from a SearXNG instance when one is named (--searx URL or COLIBRI_SEARX),
else from DuckDuckGo's plain HTML page, which needs no key. What the model reads is short on
purpose: every character is prompt the engine has to read.

A page is opened only at a public address (no loopback, private, link-local or reserved one,
redirects checked the same way), so a page the model asks for cannot reach this machine's own
services or its network."""
import html
import ipaddress
import json
import re
import socket
import urllib.parse
import urllib.request
from html.parser import HTMLParser

UA = "Mozilla/5.0 (colibri local search)"
SEARX = ""   # set by server.py from --searx


def status():
    """{"search": bool, "engine": name}: SearXNG when it answers, else DuckDuckGo"""
    if SEARX:
        try:
            with urllib.request.urlopen(SEARX.rstrip("/") + "/healthz", timeout=3) as r:
                if r.status == 200:
                    return {"search": True, "engine": "searxng"}
        except Exception:   # noqa: BLE001 - a down SearXNG falls back below
            pass
    return {"search": True, "engine": "duckduckgo"}


def _clean(s, n):
    return re.sub(r"\s+", " ", html.unescape(s or "")).strip()[:n]


def _searx(q, n):
    url = SEARX.rstrip("/") + "/search?" + urllib.parse.urlencode({"q": q, "format": "json", "language": "all"})
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=15) as r:
        data = json.load(r)
    out, seen = [], set()
    for x in data.get("results") or []:
        u = x.get("url")
        if not u or u in seen:
            continue
        seen.add(u)
        out.append({"title": _clean(x.get("title"), 160), "url": u, "snippet": _clean(x.get("content"), 320),
                    "engine": x.get("engine"), "date": (x.get("publishedDate") or "")[:10] or None})
        if len(out) >= n:
            break
    answers = [a if isinstance(a, str) else a.get("answer", "") for a in (data.get("answers") or [])][:2]
    return out, answers


class _DDG(HTMLParser):
    """result__a links (title, href) and result__snippet texts of DuckDuckGo's HTML page"""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows, self.cur, self.mode = [], None, None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = a.get("class") or ""
        if tag == "a" and "result__a" in cls.split():
            self.cur = {"title": "", "url": a.get("href") or "", "snippet": ""}
            self.rows.append(self.cur)
            self.mode = "title"
        elif "result__snippet" in cls.split() and self.cur is not None:
            self.mode = "snippet"

    def handle_endtag(self, tag):
        if tag in ("a", "td", "div"):
            self.mode = None

    def handle_data(self, data):
        if self.cur is not None and self.mode:
            self.cur[self.mode] += data


def _ddg_url(u):
    """DuckDuckGo's redirect links (//duckduckgo.com/l/?uddg=...) to the address they name"""
    if u.startswith("//"):
        u = "https:" + u
    p = urllib.parse.urlsplit(u)
    if p.hostname and p.hostname.endswith("duckduckgo.com") and p.path.startswith("/l/"):
        return urllib.parse.parse_qs(p.query).get("uddg", [u])[0]
    return u


def _duckduckgo(q, n):
    body = urllib.parse.urlencode({"q": q}).encode()
    req = urllib.request.Request("https://html.duckduckgo.com/html/", data=body,
                                 headers={"User-Agent": UA, "Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=15) as r:
        page = r.read(2 << 20).decode("utf-8", errors="replace")
    p = _DDG()
    p.feed(page)
    out, seen = [], set()
    for row in p.rows:
        u = _ddg_url(row["url"])
        if not u.startswith(("http://", "https://")) or u in seen:
            continue
        host = urllib.parse.urlsplit(u).hostname or ""
        if host.endswith("duckduckgo.com"):   # its ads go through its own links
            continue
        seen.add(u)
        out.append({"title": _clean(row["title"], 160), "url": u, "snippet": _clean(row["snippet"], 320),
                    "engine": "duckduckgo", "date": None})
        if len(out) >= n:
            break
    return out, []


def search(q, n=6):
    q = (q or "").strip()[:400]
    if not q:
        raise ValueError("an empty query")
    n = max(1, min(int(n), 10))
    results, answers = _searx(q, n) if status()["engine"] == "searxng" else _duckduckgo(q, n)
    return {"q": q, "results": results, "answers": answers}


def _public_url(url):
    """http(s) to a public address only"""
    u = urllib.parse.urlsplit(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise ValueError("http and https links only")
    for info in socket.getaddrinfo(u.hostname, u.port or (443 if u.scheme == "https" else 80)):
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if not ip.is_global:
            raise ValueError("not a public address: %s" % u.hostname)
    return url


class _CheckedRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class _Text(HTMLParser):
    """Readable text of a page: whole subtrees that are navigation, menus, language lists,
    cookie banners or hidden are skipped (by tag, role, class or id), <article>/<main> kept apart."""
    SKIP = {"script", "style", "noscript", "svg", "template", "iframe", "nav", "footer", "form", "button",
            "select", "aside", "header"}
    VOID = {"br", "img", "input", "meta", "link", "hr", "area", "base", "col", "embed", "source", "track", "wbr", "param"}
    BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section", "article", "pre", "blockquote"}
    NAV = re.compile(r"(^|[\s_-])(nav|navbar|navbox|menu|sidebar|footer|cookie|cookies|banner|portlet|interlanguage|"
                     r"toc|share|breadcrumbs?|related|advert|ads|lang|languages|catlinks|printfooter)([\s_-]|$)", re.I)

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack, self.main, self.title_on = [], 0, False
        self.all, self.inmain, self.title = [], [], ""

    def _nl(self):
        self.all.append("\n")
        if self.main:
            self.inmain.append("\n")

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self.title_on = True
        if tag in self.VOID:
            if tag == "br" and not self.stack:
                self._nl()
            return
        if self.stack:
            self.stack.append(tag)
            return
        a = dict(attrs)
        mark = "%s %s" % (a.get("class") or "", a.get("id") or "")
        outer = tag in ("html", "body", "main", "article")      # their classes describe the page, not a menu
        if (tag in self.SKIP or (a.get("role") or "") in ("navigation", "banner", "contentinfo", "menu", "search")
                or (not outer and (self.NAV.search(mark) or a.get("aria-hidden") == "true" or "hidden" in a))):
            if tag != "header" or not self.main:      # a header inside the article is its title
                self.stack = [tag]
                return
        if tag in ("article", "main"):
            self.main += 1
        if tag in self.BLOCK:
            self._nl()

    def handle_endtag(self, tag):
        if tag == "title":
            self.title_on = False
        if self.stack:
            if tag in self.stack:
                while self.stack and self.stack.pop() != tag:
                    pass
            return
        if tag in ("article", "main") and self.main:
            self.main -= 1
        if tag in self.BLOCK:
            self._nl()

    def handle_data(self, data):
        if self.title_on:
            self.title += data
            return
        if self.stack:
            return
        self.all.append(data)
        if self.main:
            self.inmain.append(data)


def page(url, max_chars=5000):
    """The readable text of one page: the <article>/<main> part when it has one, else the
    whole body, scripts and navigation dropped, cut to max_chars."""
    _public_url(url)
    max_chars = max(200, min(int(max_chars), 20000))
    opener = urllib.request.build_opener(_CheckedRedirect)
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "text/html,text/plain;q=0.9"})
    with opener.open(req, timeout=10) as r:
        ctype = r.headers.get("Content-Type", "")
        if "html" not in ctype and "text/plain" not in ctype:
            raise ValueError("not a text page (%s)" % (ctype or "unknown type"))
        raw = r.read(3 << 20)
        final = r.geturl()
    charset = re.search(r"charset=([\w-]+)", ctype)
    text = raw.decode(charset.group(1) if charset else "utf-8", errors="replace")
    title = ""
    if "html" in ctype:
        p = _Text()
        p.feed(text)
        main = "".join(p.inmain)
        text = main if len(main.strip()) > 600 else "".join(p.all)
        title = p.title.strip()
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text).strip()
    return {"url": final, "title": title[:200], "text": text[:max_chars], "truncated": len(text) > max_chars,
            "chars": len(text)}
