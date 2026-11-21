"""ui/index.html: the page's escaper must be safe inside a quoted HTML attribute.

`esc()` is used both for element text and inside attribute values
(`href="${esc(url)}"`, `title="${esc(title)}"`). A source title, a source URL or a link
in a model reply can hold a double or single quote, so an escaper that leaves quotes alone
lets that text end the attribute and add its own (`onmouseover=...`) to the tag.

The page has no JS test harness, so this replays the escaper's own `.replace()` chain, read
from index.html, and parses what comes out with the standard-library HTML parser.
"""
import re
import unittest
from html.parser import HTMLParser
from pathlib import Path

PAGE = Path(__file__).resolve().parents[2] / "ui" / "index.html"

HOSTILE = (
    '" onmouseover="alert(1)" x="',
    "' onfocus='alert(1)' autofocus='",
    'https://example.com/"onmouseover="location=\'javascript:alert%281%29\'"',
    '"><img src=x onerror=alert(1)>',
)


def escaper_chain():
    """the (pattern, replacement) pairs of `const esc=...` in index.html, in order"""
    for line in PAGE.read_text(encoding="utf-8").splitlines():
        if line.startswith("const esc="):
            return re.findall(r"\.replace\(/(.+?)/g,'(.+?)'\)", line)
    raise AssertionError("const esc= not found in ui/index.html")


def apply(chain, text):
    for pattern, replacement in chain:
        text = text.replace(pattern, replacement)
    return text


class Tags(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, [name for name, _ in attrs]))


def tags_of(html):
    parser = Tags()
    parser.feed(html)
    parser.close()
    return parser.tags


class EscaperTest(unittest.TestCase):
    def test_it_escapes_the_five_html_special_characters(self):
        out = apply(escaper_chain(), "&<>\"'")
        self.assertEqual(out, "&amp;&lt;&gt;&quot;&#39;")

    def test_text_cannot_leave_a_double_quoted_attribute(self):
        chain = escaper_chain()
        for text in HOSTILE:
            with self.subTest(text=text):
                tags = tags_of('<a class="cite" href="%s" title="%s">1</a>' % (apply(chain, text), apply(chain, text)))
                self.assertEqual(tags, [("a", ["class", "href", "title"])])

    def test_text_cannot_leave_a_single_quoted_attribute(self):
        chain = escaper_chain()
        for text in HOSTILE:
            with self.subTest(text=text):
                self.assertEqual(tags_of("<a title='%s'>1</a>" % apply(chain, text)), [("a", ["title"])])

    def test_the_check_would_catch_an_escaper_that_leaves_quotes_alone(self):
        # control: the escaper as it was (&, <, > only) lets the same text add an attribute
        old = [(p, r) for p, r in escaper_chain() if p in ("&", "<", ">")]
        broken = tags_of('<a href="%s">1</a>' % apply(old, HOSTILE[0]))
        self.assertIn("onmouseover", broken[0][1])


if __name__ == "__main__":
    unittest.main()
