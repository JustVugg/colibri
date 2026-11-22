#!/usr/bin/env python3
"""ui/web.py's DuckDuckGo search: from some addresses html.duckduckgo.com answers 202 with its bot
check instead of results, which the chat read as no results while the lite page answered. The
search then asks the lite page; with both checking, it says so instead of returning nothing.
No network: urlopen is a stand-in that plays the two pages."""
import sys
import unittest
from pathlib import Path
from unittest import mock

C = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(C.parent / "ui"))
import web   # noqa: E402

BOT = '<html><body><div class="anomaly-modal__title">Unfortunately, bots use DuckDuckGo too.</div></body></html>'
HTML = ('<div class="result"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FFred_Trump">'
        'Fred Trump - Wikipedia</a><a class="result__snippet">Frederick Christ Trump Sr. was an American real estate developer.</a></div>'
        '<div class="result"><a class="result__a" href="https://duckduckgo.com/y.js?ad=1">an ad</a></div>')
LITE = ('<table><tr><td><a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fa" class="result-link">Example A</a></td></tr>'
        '<tr><td class="result-snippet">The first result of the lite page.</td></tr>'
        '<tr><td><a rel="nofollow" href="https://example.org/b" class="result-link">Example B</a></td></tr>'
        '<tr><td class="result-snippet">The second.</td></tr></table>')


class Page:
    def __init__(self, status, body):
        self.status, self.body = status, body.encode()

    def read(self, n=-1):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def serve(pages):
    asked = []

    def urlopen(req, timeout=None):
        asked.append(req.full_url)
        return Page(*pages[req.full_url])
    return urlopen, asked


class DuckDuckGoTest(unittest.TestCase):
    def run_search(self, html, lite):
        urlopen, asked = serve({web._DDG_PAGES[0]: html, web._DDG_PAGES[1]: lite})
        with mock.patch.object(web, "SEARX", ""), mock.patch("urllib.request.urlopen", urlopen):
            return web._duckduckgo("padre di donald trump", 6), asked

    def test_html_page_results_its_redirects_unwrapped_and_its_ads_dropped(self):
        (rows, _), asked = self.run_search((200, HTML), (200, LITE))
        self.assertEqual(asked, [web._DDG_PAGES[0]])
        self.assertEqual([r["url"] for r in rows], ["https://en.wikipedia.org/wiki/Fred_Trump"])
        self.assertEqual(rows[0]["title"], "Fred Trump - Wikipedia")
        self.assertIn("real estate developer", rows[0]["snippet"])

    def test_bot_check_falls_back_to_the_lite_page(self):
        (rows, _), asked = self.run_search((202, BOT), (200, LITE))
        self.assertEqual(asked, list(web._DDG_PAGES))
        self.assertEqual([(r["title"], r["url"]) for r in rows],
                         [("Example A", "https://example.org/a"), ("Example B", "https://example.org/b")])
        self.assertEqual(rows[0]["snippet"], "The first result of the lite page.")

    def test_bot_check_with_status_200_is_still_a_bot_check(self):
        (rows, _), _asked = self.run_search((200, BOT), (200, LITE))
        self.assertEqual(len(rows), 2)

    def test_both_pages_checking_is_an_error_not_an_empty_list(self):
        with self.assertRaisesRegex(RuntimeError, "bot check"):
            self.run_search((202, BOT), (202, BOT))

    def test_no_results_is_an_empty_list(self):
        (rows, _), asked = self.run_search((200, "<html></html>"), (200, "<html></html>"))
        self.assertEqual(rows, [])
        self.assertEqual(asked, list(web._DDG_PAGES))


if __name__ == "__main__":
    unittest.main()
