#!/usr/bin/env python3
"""The chat's attachments: ui/docs.py gives the text of a PDF, Word, OpenDocument, PowerPoint, HTML
or plain-text file, and ui/server.py's /doc/extract hands it only to its own page. The files are
built here, in memory; a PDF needs pypdf or PyMuPDF and its test is skipped without them."""
import io
import json
import sys
import threading
import unittest
import unittest.mock
import urllib.error
import urllib.request
import zipfile
from http.server import ThreadingHTTPServer
from pathlib import Path

C = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(C.parent / "ui"))
import docs   # noqa: E402


def office(parts):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, text in parts.items():
            z.writestr(name, text)
    return buf.getvalue()


W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
DOCX = office({"word/document.xml": f'''<w:document {W}><w:body>
  <w:p><w:r><w:t>Primo paragrafo</w:t></w:r><w:r><w:tab/><w:t>dopo un tab</w:t></w:r></w:p>
  <w:tbl><w:tr><w:tc><w:p><w:r><w:t>cella a</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>cella b</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
  <w:p><w:r><w:t>Ultimo</w:t></w:r><w:r><w:br/><w:t>a capo</w:t></w:r></w:p>
</w:body></w:document>'''})
A = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
PPTX = office({f"ppt/slides/slide{i}.xml": f'<p:sld xmlns:p="x" {A}><a:p><a:r><a:t>diapositiva {i}</a:t></a:r></a:p></p:sld>'
               for i in (1, 2, 10)})
ODT = office({"content.xml": '''<office:document-content xmlns:office="o" xmlns:text="t"><office:body><office:text>
  <text:h>Titolo</text:h><text:p>Un paragrafo <text:span>con uno span</text:span></text:p>
</office:text></office:body></office:document-content>'''})


def tiny_pdf(text):
    """a one-page PDF with `text` in Helvetica, its xref offsets computed"""
    stream = b"BT /F1 12 Tf 72 720 Td (" + text.encode() + b") Tj ET"
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
            b"/Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offs = bytearray(b"%PDF-1.4\n"), []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    x = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offs)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, x)
    return bytes(out)


def has_pdf_reader():
    for mod in ("pymupdf", "fitz", "pypdf"):
        try:
            __import__(mod)
            return True
        except ImportError:
            pass
    return False


class ExtractTest(unittest.TestCase):
    def test_docx_keeps_order_tabs_tables_and_breaks(self):
        d = docs.extract("relazione.docx", DOCX)
        self.assertEqual(d["kind"], "docx")
        self.assertEqual(d["text"], "Primo paragrafo\tdopo un tab\ncella a | cella b\nUltimo\na capo")
        self.assertEqual(d["chars"], len(d["text"]))
        self.assertEqual(d["name"], "relazione.docx")

    def test_pptx_slides_in_their_number_order(self):
        d = docs.extract("x.pptx", PPTX)
        self.assertEqual(d["pages"], 3)
        self.assertEqual(d["text"], "[Diapositiva 1]\ndiapositiva 1\n\n[Diapositiva 2]\ndiapositiva 2\n\n"
                                    "[Diapositiva 3]\ndiapositiva 10")

    def test_odt_headings_and_paragraphs(self):
        self.assertEqual(docs.extract("x.odt", ODT)["text"], "Titolo\nUn paragrafo con uno span")

    def test_html_drops_scripts_and_styles(self):
        page = b"<html><head><style>p{}</style><script>alert(1)</script></head><body><h1>Titolo</h1><p>testo &amp; altro</p></body></html>"
        d = docs.extract("pagina.html", page)
        self.assertEqual(d["kind"], "html")
        self.assertEqual(d["text"], "Titolo\n\ntesto & altro")

    def test_text_in_utf8_with_bom_and_in_cp1252(self):
        self.assertEqual(docs.extract("a.txt", "﻿caffè".encode("utf-8"))["text"], "caffè")
        self.assertEqual(docs.extract("a.csv", "perché".encode("cp1252"))["text"], "perché")

    def test_refusals(self):
        with self.assertRaisesRegex(ValueError, "vecchio formato"):
            docs.extract("a.doc", b"\xd0\xcf\x11\xe0")
        with self.assertRaisesRegex(ValueError, "non e un file di testo"):
            docs.extract("a.bin", b"\x00\x01\x02")
        with self.assertRaisesRegex(ValueError, "non c'e testo"):
            docs.extract("a.txt", b"   \n ")
        with self.assertRaisesRegex(ValueError, "troppo grande"):
            docs.extract("a.txt", b"x" * (docs.MAX_BYTES + 1))

    def test_a_part_past_its_unpacked_size_is_refused(self):
        """a part is unpacked up to PART_MAX, and a file's parts up to ARCHIVE_MAX together:
        a few hundred KB of zip can hold far more"""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("word/document.xml", f"<w:document {W}><w:body>" + " " * (docs.PART_MAX + 1) + "</w:body></w:document>")
        self.assertLess(len(buf.getvalue()), 1 << 20)
        with self.assertRaisesRegex(ValueError, "troppo grande una volta decompresso"):
            docs.extract("a.docx", buf.getvalue())
        with unittest.mock.patch.object(docs, "ARCHIVE_MAX", 100):   # two slides past the whole file's
            with self.assertRaisesRegex(ValueError, "troppo grande una volta decompresso"):
                docs.extract("a.pptx", PPTX)

    def test_a_document_type_is_refused(self):
        """Office XML has none: a part with one (its entities could expand without bound) is
        refused before ElementTree reads it, in UTF-8 and in UTF-16 alike"""
        xml = ('<?xml version="1.0" encoding="%s"?><!DOCTYPE d [<!ENTITY a "aaaa"><!ENTITY b "&a;&a;&a;&a;">]>'
               f'<w:document {W}><w:body><w:p><w:r><w:t>&b;</w:t></w:r></w:p></w:body></w:document>')
        for enc in ("UTF-8", "UTF-16"):
            with self.subTest(encoding=enc):
                with self.assertRaisesRegex(ValueError, "DTD"):
                    docs.extract("a.docx", office({"word/document.xml": (xml % enc).encode(enc.lower())}))
        with self.assertRaisesRegex(ValueError, "DTD"):
            docs.extract("a.odt", office({"content.xml": '<!DOCTYPE x [<!ENTITY e "e">]><x>&e;</x>'}))

    @unittest.skipUnless(has_pdf_reader(), "neither pypdf nor PyMuPDF is installed")
    def test_pdf_text_by_page(self):
        d = docs.extract("a.pdf", tiny_pdf("Ciao colibri, questa e una pagina di prova"))
        self.assertEqual(d["kind"], "pdf")
        self.assertEqual(d["pages"], 1)
        self.assertIn("[Pagina 1]", d["text"])
        self.assertIn("Ciao colibri", d["text"])


class ServerRouteTest(unittest.TestCase):
    """/doc/extract answers its own page only: its Host and the header a page elsewhere cannot send"""

    @classmethod
    def setUpClass(cls):
        import server
        cls.quiet = unittest.mock.patch.object(server.Handler, "log_message", lambda *a: None)
        cls.quiet.start()
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        server.Handler.port = cls.srv.server_port
        cls.port = cls.srv.server_port
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()
        cls.quiet.stop()

    def post(self, body, host=None, ui=True, name="note.txt"):
        h = {"Host": host or "127.0.0.1:%d" % self.port, "X-File-Name": name, "Content-Type": "application/octet-stream"}
        if ui:
            h["X-Colibri-UI"] = "1"
        req = urllib.request.Request("http://127.0.0.1:%d/doc/extract" % self.port, data=body, headers=h)
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def test_the_page_gets_the_text(self):
        code, d = self.post("una nota".encode())
        self.assertEqual((code, d["text"], d["name"]), (200, "una nota", "note.txt"))

    def test_a_name_in_percent_encoding(self):
        code, d = self.post(DOCX, name="relazione%20finale.docx")
        self.assertEqual((code, d["name"], d["kind"]), (200, "relazione finale.docx", "docx"))

    def test_other_pages_and_hosts_are_refused(self):
        self.assertEqual(self.post(b"x", ui=False)[0], 403)
        self.assertEqual(self.post(b"x", host="evil.example:%d" % self.port)[0], 403)

    def test_a_file_it_cannot_read_is_a_400(self):
        code, d = self.post(b"\xd0\xcf\x11\xe0", name="vecchio.doc")
        self.assertEqual(code, 400)
        self.assertIn("vecchio formato", d["error"])


if __name__ == "__main__":
    unittest.main()
