"""The remote document warning must not trust a hostname's textual prefix."""
import unittest
from unittest import mock

from tests.test_coli_chat_documents import coli


class LocalServerAddress(unittest.TestCase):
    def test_only_literal_loopback_addresses_are_local(self):
        for host in ("127.0.0.1", "127.10.20.30", "localhost", "[::1]"):
            with self.subTest(host=host):
                self.assertTrue(coli.is_local_base(f"http://{host}:8000"))

    def test_loopback_looking_remote_names_are_not_local(self):
        for host in ("127.example.test", "127.0.0.1.example.test", "127.999.1.2",
                     "192.168.1.1", "[2001:db8::1]"):
            with self.subTest(host=host):
                self.assertFalse(coli.is_local_base(f"http://{host}:8000"))

    def test_explicit_remote_attachment_reports_destination(self):
        notes = []
        with mock.patch.object(coli, "document_text", return_value=("hello", None)):
            self.assertIsNotNone(coli.read_document(
                "notes.txt", "https://127.example.test", note=notes.append))
        self.assertTrue(any("another machine" in note for note in notes))


if __name__ == "__main__":
    unittest.main()
