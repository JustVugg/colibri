"""Documents for the chat (local): the text of a PDF, Word, OpenDocument, PowerPoint, HTML or
plain-text file, so the model reads it in the message. A PDF page with no text (a scan) comes
back as a picture, for a model with a vision tower; at most SCAN_PAGES of them."""
import base64
import html.parser
import io
import os
import re
import zipfile
import xml.etree.ElementTree as ET
import xml.parsers.expat

MAX_BYTES = 80 << 20
PART_MAX = 64 << 20       # an Office part's text once unpacked, and all of a file's parts together
ARCHIVE_MAX = 256 << 20
SCAN_PAGES = 6
SCAN_SIDE = 1280          # the longer side of a scanned page's picture, as the page shrinks a photo
TEXT_MIN = 20             # a PDF page with fewer characters than this is read as a scan

TEXT_EXT = {".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".jsonl", ".xml", ".log", ".yaml", ".yml",
            ".toml", ".ini", ".cfg", ".py", ".js", ".ts", ".tsx", ".jsx", ".c", ".h", ".cc", ".cpp", ".hpp",
            ".java", ".go", ".rs", ".sh", ".bat", ".ps1", ".sql", ".tex", ".rst", ".srt", ".vtt", ".css"}


def _pdf(data):
    try:
        import pymupdf as fitz   # PyMuPDF: text and a page's picture
    except ImportError:
        try:
            import fitz
        except ImportError:
            fitz = None
    if fitz is None:
        from pypdf import PdfReader   # text only
        reader = PdfReader(io.BytesIO(data))
        pages = [(p.extract_text() or "") for p in reader.pages]
        text = "\n\n".join("[Pagina %d]\n%s" % (i + 1, t.strip()) for i, t in enumerate(pages) if t.strip())
        empty = [i + 1 for i, t in enumerate(pages) if len(t.strip()) < TEXT_MIN]
        note = ("%d pagine senza testo (scansioni): per leggerle serve PyMuPDF" % len(empty)) if empty else ""
        return {"kind": "pdf", "pages": len(pages), "text": text, "images": [], "note": note}
    doc = fitz.open(stream=data, filetype="pdf")
    parts, images, scans = [], [], []
    for i, page in enumerate(doc):
        t = page.get_text("text").strip()
        if len(t) >= TEXT_MIN:
            parts.append("[Pagina %d]\n%s" % (i + 1, t))
            continue
        scans.append(i + 1)
        if len(images) < SCAN_PAGES:
            r = page.rect
            zoom = SCAN_SIDE / max(r.width, r.height, 1)
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            jpg = pix.tobytes("jpeg", jpg_quality=85)
            images.append("data:image/jpeg;base64," + base64.b64encode(jpg).decode())
            parts.append("[Pagina %d: una scansione, allegata come immagine n. %d]" % (i + 1, len(images)))
        else:
            parts.append("[Pagina %d: una scansione, non allegata (oltre le %d immagini)]" % (i + 1, SCAN_PAGES))
    n = doc.page_count
    doc.close()
    note = ""
    if scans:
        note = "%d %s senza testo (scansioni): %d %s come immagini" % (
            len(scans), "pagina" if len(scans) == 1 else "pagine", len(images), "allegata" if len(images) == 1 else "allegate")
    return {"kind": "pdf", "pages": n, "text": "\n\n".join(parts), "images": images, "note": note}


def _local(tag):
    return tag.rsplit("}", 1)[-1]


class _Parts:
    """an Office file's zip, its parts read whole up to PART_MAX each and ARCHIVE_MAX in all: the
    sizes the archive declares are what zipfile unpacks at most"""
    def __init__(self, data):
        self.z = zipfile.ZipFile(io.BytesIO(data))
        self.left = ARCHIVE_MAX

    def namelist(self):
        return self.z.namelist()

    def read(self, name):
        info = self.z.getinfo(name)
        if info.file_size > PART_MAX or info.file_size > self.left:
            raise ValueError("documento troppo grande una volta decompresso (%s: %d MB)" % (name, info.file_size >> 20))
        self.left -= info.file_size
        return self.z.read(info)


def _xml(data):
    """an Office XML part as ElementTree's: a document type (which Office XML never has) is
    refused first, so no entity of one is ever expanded"""
    def refuse(*_):
        raise ValueError("documento non valido (una DTD nel suo XML)")
    check = xml.parsers.expat.ParserCreate()
    check.StartDoctypeDeclHandler = refuse
    check.EntityDeclHandler = refuse
    check.Parse(data, True)
    return ET.fromstring(data)


def _xml_text(data, para_tags, text_tags, tab_tags=(), br_tags=(), cell_tags=(), row_tags=()):
    """the paragraphs of an Office XML part in document order, a table row on one line"""
    root = _xml(data)
    out = []

    def collect(node, sep):
        buf = []
        for el in node.iter():
            name = _local(el.tag)
            if name in text_tags:
                buf.append(el.text or "")
            elif name in tab_tags:
                buf.append("\t")
            elif name in br_tags:
                buf.append("\n")
            elif name in para_tags and el is not node and buf:
                buf.append(sep)
        return "".join(buf)

    def walk(node):
        for child in node:
            name = _local(child.tag)
            if name in row_tags:
                cells = [collect(c, " ").strip() for c in child if _local(c.tag) in cell_tags]
                out.append(" | ".join(cells))
            elif name in para_tags:
                out.append(collect(child, "\n"))
            else:
                walk(child)

    walk(root)
    return "\n".join(line.rstrip() for line in out if line.strip())


def _docx(data):
    z = _Parts(data)
    text = _xml_text(z.read("word/document.xml"), {"p"}, {"t"}, {"tab"}, {"br", "cr"}, {"tc"}, {"tr"})
    return {"kind": "docx", "pages": 0, "text": text, "images": [], "note": ""}


def _odt(data):
    z = _Parts(data)
    root = _xml(z.read("content.xml"))
    lines = []
    for node in root.iter():
        name = _local(node.tag)
        if name in ("p", "h"):
            lines.append("".join(node.itertext()).rstrip())
    return {"kind": "odt", "pages": 0, "text": "\n".join(l for l in lines if l.strip()), "images": [], "note": ""}


def _pptx(data):
    z = _Parts(data)
    names = sorted((n for n in z.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
                   key=lambda n: int(re.search(r"(\d+)", n.rsplit("/", 1)[-1]).group(1)))
    parts = []
    for i, n in enumerate(names):
        t = _xml_text(z.read(n), {"p"}, {"t"}, (), {"br"})
        parts.append("[Diapositiva %d]\n%s" % (i + 1, t))
    return {"kind": "pptx", "pages": len(names), "text": "\n\n".join(parts), "images": [], "note": ""}


class _HTMLText(html.parser.HTMLParser):
    SKIP = {"script", "style", "noscript", "template", "svg"}
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "pre", "table"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def _decode(data):
    if b"\x00" in data[:65536]:
        raise ValueError("non e un file di testo (PDF, Word, OpenDocument, PowerPoint o testo)")
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", "replace")


def extract(name, data):
    if len(data) > MAX_BYTES:
        raise ValueError("file troppo grande (%d MB, al massimo %d)" % (len(data) >> 20, MAX_BYTES >> 20))
    ext = os.path.splitext(name.lower())[1]
    if data[:5] == b"%PDF-" or ext == ".pdf":
        out = _pdf(data)
    elif ext == ".docx":
        out = _docx(data)
    elif ext == ".odt":
        out = _odt(data)
    elif ext == ".pptx":
        out = _pptx(data)
    elif ext in (".html", ".htm", ".xhtml"):
        p = _HTMLText()
        p.feed(_decode(data))
        text = re.sub(r"\n\s*\n+", "\n\n", "".join(p.out)).strip()
        out = {"kind": "html", "pages": 0, "text": text, "images": [], "note": ""}
    elif ext in (".doc", ".xls", ".ppt"):
        raise ValueError("il vecchio formato %s non si legge: salvalo come %sx o PDF" % (ext, ext))
    else:
        out = {"kind": "text", "pages": 0, "text": _decode(data).strip(), "images": [], "note": ""}
    out["text"] = out["text"].replace("\x00", "")
    out["name"] = name
    out["chars"] = len(out["text"])
    if not out["text"].strip() and not out["images"]:
        raise ValueError("nel file non c'e testo da leggere")
    return out


if __name__ == "__main__":
    # `python docs.py FILE`: the reader for colibri-tui (c/tui.c), one JSON object on stdout,
    # ASCII only (a Windows console's code page cannot garble it): extract()'s, or {"error": ...}
    import json
    import sys
    if len(sys.argv) != 2:
        sys.exit("usage: docs.py FILE")
    try:
        with open(sys.argv[1], "rb") as f:
            print(json.dumps(extract(os.path.basename(sys.argv[1]), f.read())))
    except Exception as error:   # the reader's own refusals and an unreadable file alike
        print(json.dumps({"error": str(error)}))
        sys.exit(1)
