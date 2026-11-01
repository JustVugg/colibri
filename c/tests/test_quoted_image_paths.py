"""Explicitly quoted local PNG paths preserve filename punctuation."""
import ast
import base64
import contextlib
import importlib.util
import io
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

C_DIR = Path(__file__).resolve().parent.parent


def consumers():
    tree = ast.parse((C_DIR / "coli").read_text(encoding="utf-8"))
    names = {"_c", "C", "to_local_path", "message_with_images", "image_data_uri"}
    constants = {"IMAGE_SUFFIXES", "_ESTENSIONI", "_IMAGE_PATH_START", "IMAGE_PATH"}
    nodes = [node for node in tree.body
             if (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names)
             or (isinstance(node, ast.Assign) and any(
                 isinstance(target, ast.Name) and target.id in constants for target in node.targets))]
    namespace = dict(os=os, re=re, sys=sys)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(C_DIR / "coli"), "exec"), namespace)
    return namespace


class QuotedImagePathsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.namespace = consumers()
        spec = importlib.util.spec_from_file_location("actual_quoted_png", C_DIR / "image_engine.py")
        self.image = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.image)
        self.png = self.image.encode_png(2, 2, b"\xff\0\0\xff" * 4, 4)

    def check_path(self, name, quote, exact_prose=True):
        path = self.folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.png)
        content = self.namespace["message_with_images"](
            f"Guarda com'è questa foto {quote}{path}{quote} ?")
        self.assertIsInstance(content, list)
        prose = [part["text"] for part in content if part["type"] == "text"]
        if exact_prose:
            self.assertEqual(prose, ["Guarda com'è questa foto ?"])
        else:
            self.assertIn("Guarda com'è questa foto", prose[0])
        images = [part for part in content if part["type"] == "image_url"]
        self.assertEqual(len(images), 1)
        uri = images[0]["image_url"]["url"]
        self.assertTrue(uri.startswith("data:image/png;base64,"))
        self.assertEqual(base64.b64decode(uri.split(",", 1)[1]), self.png)

    def test_apostrophe_in_double_quoted_filename(self):
        self.check_path("John's cat.png", '"')

    def test_apostrophe_in_double_quoted_directory(self):
        self.check_path("John's photos/cat.png", '"')

    def test_full_quoted_filename_includes_earlier_image_extension(self):
        self.check_path("prior.png-edited.png", '"')

    def test_ordinary_quote_styles(self):
        for quote in ('"', "'"):
            with self.subTest(quote=quote):
                self.check_path("ordinary.png", quote, exact_prose=False)

    def test_unquoted_paths_and_plain_prose(self):
        path = self.folder / "ordinary.png"
        path.write_bytes(self.png)
        content = self.namespace["message_with_images"](f"look {path} ?")
        self.assertIsInstance(content, list)
        text = "Guarda com'è questa frase senza immagini"
        self.assertEqual(self.namespace["message_with_images"](text), text)

    def test_slash_inside_a_word_does_not_start_a_path(self):
        for text in ("Is WebP/AVIF smaller than a .png for icons",
                     "TCP/IP diagram.png please"):
            with self.subTest(text=text):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(self.namespace["message_with_images"](text), text)

    def test_multiple_images_still_rejected(self):
        paths = [self.folder / name for name in ("one.png", "two.png")]
        for path in paths:
            path.write_bytes(self.png)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(self.namespace["message_with_images"](
                f'"{paths[0]}" "{paths[1]}"'))


if __name__ == "__main__":
    unittest.main()
