"""Code spans are literal even when streamed across marker boundaries."""
import ast
import contextlib
import io
import re
import sys
import unittest
from pathlib import Path

CLI = Path(__file__).resolve().parent.parent / "coli"
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def renderer():
    # Execute the shipping renderer without starting a model or changing process limits.
    tree = ast.parse(CLI.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and node.name in {"_c", "C", "MDStream"}]
    namespace = {"sys": sys}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(CLI), "exec"), namespace)
    return namespace


class MarkdownCodeTest(unittest.TestCase):
    def rendered(self, text, split=None):
        namespace = renderer()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            stream = namespace["MDStream"]("")
            if split is None:
                stream.feed(text)
            else:
                stream.feed(text[:split])
                stream.feed(text[split:])
            stream.close()
        return output.getvalue(), namespace["C"]

    def test_inline_code_keeps_exponent_and_keyword_unpacking(self):
        for text, expected in (("`2 ** 3`\n", "2 ** 3\n"),
                               ("`**kwargs`\n", "**kwargs\n")):
            for split in (None, *range(1, len(text))):
                with self.subTest(text=text, split=split):
                    output, colours = self.rendered(text, split)
                    self.assertEqual(ANSI.sub("", output), expected)
                    self.assertIn(colours.org, output)
                    self.assertNotIn(colours.b, output)

    def test_bold_outside_code_still_works(self):
        text = "`2 ** 3` and **bold**\n"
        for split in (None, *range(1, len(text))):
            with self.subTest(split=split):
                output, colours = self.rendered(text, split)
                self.assertEqual(ANSI.sub("", output), "2 ** 3 and bold\n")
                self.assertEqual(output.count(colours.b), 1)

    def test_single_asterisk_inside_code_is_literal(self):
        text = "`x * y`\n"
        for split in (None, *range(1, len(text))):
            with self.subTest(split=split):
                output, _ = self.rendered(text, split)
                self.assertEqual(ANSI.sub("", output), "x * y\n")

    def test_fenced_code_preserves_asterisks(self):
        text = "```python\n2 ** 3\n```\n"
        for split in (None, *range(1, len(text))):
            with self.subTest(split=split):
                output, colours = self.rendered(text, split)
                self.assertIn("2 ** 3\n", ANSI.sub("", output))
                self.assertNotIn(colours.b, output)


if __name__ == "__main__":
    unittest.main()
