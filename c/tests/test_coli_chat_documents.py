"""coli chat's documents (#2431): `/file PATH` in the chat and `coli chat --file PATH`.

The security contract, first:
- a path is never picked out of a message's text: only /file and --file read a file
  (pasted text would otherwise send a file of this machine to the server);
- only a regular file with a text suffix or .pdf, at most DOC_MAX_BYTES on disk and
  DOC_MAX_CHARS of text, no NUL byte, its control characters (but tab and newline) removed;
- a name printed to the terminal has its control characters shown as '?';
- a PDF is parsed in another process, from an empty directory, with a time limit and its
  output bounded; its path is absolute (a name starting with '-' is not an option);
- a server on another machine is named before the text goes there.
Then the reading itself: a text file's block, a PDF through pypdf (and pdftotext where
installed), and what to install with neither.
"""
import argparse
import importlib.util
import os
import shutil
import tempfile
import unittest
from importlib.machinery import SourceFileLoader
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent.parent
loader = SourceFileLoader("coli_chat_documents", str(HERE / "coli"))
spec = importlib.util.spec_from_loader(loader.name, loader)
coli = importlib.util.module_from_spec(spec)
loader.exec_module(coli)


def quiet(*_args, **_kwargs):
    pass


def hello_pdf_bytes():
    """a one-page PDF with "hello pdf" in its text layer, written by hand"""
    content = b"BT /F1 24 Tf 72 720 Td (hello pdf) Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = b"%PDF-1.4\n", []
    for i, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return out


class Documents(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)

    def tearDown(self):
        self.dir.cleanup()

    def write(self, name, data):
        path = self.root / name
        path.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
        return str(path)

    # ---- what is never read -------------------------------------------------------
    def test_a_path_in_a_message_is_never_read(self):
        path = self.write("secret.txt", "do not send\n")
        with mock.patch.object(coli, "print", quiet, create=True):
            payload = coli.message_with_images(f"what is in {path}?")
        self.assertEqual(payload, f"what is in {path}?")
        self.assertFalse(hasattr(coli, "documents_in_text"))

    def test_only_text_suffixes_and_pdf(self):
        for name in ("id_rsa", ".env", "key.pem", "notes.docx", "run.sh"):
            text, why = coli.document_text(self.write(name, "x\n"))
            self.assertIsNone(text, name)
            self.assertIn("not a text file or a PDF", why)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "no FIFOs here")
    def test_a_fifo_is_not_opened(self):
        path = str(self.root / "pipe.txt")
        os.mkfifo(path)
        text, why = coli.document_text(path)   # opening it would block
        self.assertIsNone(text)
        self.assertEqual(why, "not a regular file")

    def test_bounds(self):
        with mock.patch.object(coli, "DOC_MAX_BYTES", 10):
            text, why = coli.document_text(self.write("big.txt", "x" * 11))
        self.assertIsNone(text)
        self.assertIn("past the", why)
        with mock.patch.object(coli, "DOC_MAX_CHARS", 5):
            text, why = coli.document_text(self.write("long.txt", "abcdef"))
        self.assertIsNone(text)
        self.assertIn("past 5 characters", why)
        text, why = coli.document_text(self.write("blob.log", b"ab\0cd"))
        self.assertIsNone(text)
        self.assertIn("NUL", why)

    def test_control_characters_out_of_the_text_and_the_names(self):
        text, why = coli.document_text(self.write("ctl.txt", "a\x1b[2Jb\tc\nd\x07e\r\n"))
        self.assertIsNone(why)
        self.assertEqual(text, "a[2Jb\tc\nde\n")
        self.assertEqual(coli.printable_name("/tmp/x\x1b]0;evil\x07.txt"), "x?]0;evil?.txt")

    # ---- what is read ---------------------------------------------------------------
    def test_a_text_file_becomes_its_block(self):
        path = self.write("notes.txt", "line one\nline two\n")
        said = []
        block = coli.read_document(path, note=said.append)
        self.assertEqual(block, "[file notes.txt]\nline one\nline two\n[end of notes.txt]")
        self.assertTrue(any("attached" in s for s in said))

    def test_a_missing_file_says_why(self):
        said = []
        self.assertIsNone(coli.read_document(str(self.root / "nope.txt"), note=said.append))
        self.assertTrue(said and "nope.txt" in said[0])

    def test_a_remote_server_is_named(self):
        path = self.write("a.md", "# a\n")
        for base, remote in (("http://127.0.0.1:8000", False), ("http://localhost:8000", False),
                             ("http://[::1]:8000", False), ("http://192.168.1.20:8000", True),
                             ("https://example.org", True)):
            said = []
            self.assertIsNotNone(coli.read_document(path, base, note=said.append))
            self.assertEqual(any("another machine" in s for s in said), remote, base)

    def test_file_option_goes_with_the_first_message(self):
        a = self.write("a.txt", "alpha\n"); b = self.write("b.md", "beta\n")
        with mock.patch.object(coli, "print", quiet, create=True):
            pre = coli.chat_preload(argparse.Namespace(file=[a, b]))
        self.assertEqual(pre, "[file a.txt]\nalpha\n[end of a.txt]\n\n[file b.md]\nbeta\n[end of b.md]")
        self.assertEqual(coli.chat_preload(argparse.Namespace(file=None)), "")
        with mock.patch.object(coli, "print", quiet, create=True), self.assertRaises(SystemExit):
            coli.chat_preload(argparse.Namespace(file=[str(self.root / "missing.txt")]))

    def test_file_is_a_command(self):
        self.assertIn("file", coli.CHAT_COMMANDS)
        self.assertEqual(coli.chat_command("/file ~/notes.txt"), ("file", "~/notes.txt"))

    # ---- PDFs -------------------------------------------------------------------------
    def test_pdf_without_a_reader_says_what_to_install(self):
        path = self.write("paper.pdf", b"%PDF-1.4\n")
        with mock.patch.object(coli.shutil, "which", return_value=None), \
                mock.patch("importlib.util.find_spec", return_value=None):
            text, why = coli.document_text(path)
        self.assertIsNone(text)
        self.assertIn("pdftotext", why)
        self.assertIn("pypdf", why)

    @unittest.skipUnless(shutil.which("pdftotext"), "pdftotext is not installed")
    def test_pdf_through_pdftotext(self):
        text, why = coli.document_text(self.write("-hello.pdf", hello_pdf_bytes()))   # a name like an option
        self.assertIsNone(why)
        self.assertIn("hello pdf", text)

    @unittest.skipUnless(importlib.util.find_spec("pypdf"), "pypdf is not installed")
    def test_pdf_through_pypdf_in_another_process(self):
        real_which = shutil.which
        with mock.patch.object(coli.shutil, "which", lambda n: None if n == "pdftotext" else real_which(n)):
            text, why = coli.document_text(self.write("-hello.pdf", hello_pdf_bytes()))
            self.assertIsNone(why)
            self.assertIn("hello pdf", text)
            with mock.patch.object(coli, "DOC_MAX_CHARS", 3):   # the child's output is bounded
                text, why = coli.document_text(self.write("h2.pdf", hello_pdf_bytes()))
            self.assertIsNone(text)
            self.assertIn("past 3 characters", why)

    @unittest.skipUnless(os.name == "posix", "the timeout test uses sh")
    def test_a_reader_that_never_ends_is_stopped(self):
        with mock.patch.object(coli, "DOC_PDF_SECONDS", 1):
            raw, why = coli._run_bounded(["sh", "-c", "sleep 30"], 100)
        self.assertIsNone(raw)
        self.assertIn("longer than 1 s", why)


if __name__ == "__main__":
    unittest.main()
