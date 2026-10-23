"""`coli chat` reads a pasted image itself and sends a data: URI: the server no
longer opens local paths for a client (see ImageUrlPathGuard in
test_openai_server). A path the user pastes must therefore never reach the
request as a path."""
import base64
import importlib.machinery
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

CLI = Path(__file__).resolve().parent.parent / "coli"


def load_cli():
    loader = importlib.machinery.SourceFileLoader("coli_images_t", str(CLI))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    argv = sys.argv
    sys.argv = ["coli"]
    try:
        loader.exec_module(module)
    finally:
        sys.argv = argv
    return module


class ChatImagesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_cli()

    def test_pasted_path_becomes_a_data_uri(self):
        with tempfile.TemporaryDirectory() as d:
            img = Path(d) / "shot.png"
            img.write_bytes(b"\x89PNG\r\n\x1a\n")
            content = self.m.message_with_images(f"what is this {img} ?")
        self.assertIsInstance(content, list)
        parts = {p["type"]: p for p in content}
        # On Windows the MSYS2 Python spells the temp dir "D:/a/...": IMAGE_PATH
        # (untouched here) starts its match at the first slash and leaves the
        # drive letter in the text. What this test holds is that the file
        # travels as bytes, so the text is checked for the words, not the shape.
        self.assertIn("what is this", parts["text"]["text"])
        self.assertNotIn("shot.png", parts["text"]["text"])
        url = parts["image_url"]["image_url"]["url"]
        self.assertTrue(url.startswith("data:image/png;base64,"), url)
        self.assertEqual(base64.b64decode(url.split(",", 1)[1]), b"\x89PNG\r\n\x1a\n")
        self.assertNotIn(str(img), url)

    def test_jpeg_gets_its_mime(self):
        with tempfile.TemporaryDirectory() as d:
            img = Path(d) / "photo.jpg"
            img.write_bytes(b"\xff\xd8\xff")
            self.assertTrue(self.m.image_data_uri(str(img)).startswith("data:image/jpeg;base64,"))

    def test_plain_text_is_untouched(self):
        self.assertEqual(self.m.message_with_images("no images here"), "no images here")


if __name__ == "__main__":
    unittest.main()
