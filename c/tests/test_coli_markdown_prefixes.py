"""Formatting must not depend on how the engine partitions Markdown bytes."""
import contextlib
import io
import unittest
from tests.test_coli_markdown import renderer, ANSI


def render_chunks(chunks):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        stream = renderer()['MDStream']('')
        for chunk in chunks:
            stream.feed(chunk)
        stream.close()
    return ANSI.sub('', output.getvalue())


class MarkdownPrefixChunks(unittest.TestCase):
    def test_list_markers_split_from_their_space(self):
        for text in ('- item\n', '  - item\n', '  * item\n'):
            for split in range(1, len(text)):
                with self.subTest(text=text, split=split):
                    self.assertEqual(render_chunks([text[:split], text[split:]]), '• item\n')
            self.assertEqual(render_chunks(text), '• item\n')

    def test_leading_spaces_do_not_hide_a_streamed_heading(self):
        text = '  # Heading\n'
        for split in range(1, len(text)):
            with self.subTest(split=split):
                self.assertEqual(render_chunks([text[:split], text[split:]]), 'Heading\n')

    def test_literal_hyphens_and_plain_leading_spaces_stay_literal(self):
        for text in ('-word\n', '-\n', '  plain text\n'):
            self.assertEqual(render_chunks(text), render_chunks([text]))


if __name__ == '__main__':
    unittest.main()
