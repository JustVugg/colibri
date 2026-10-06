"""An allowed image directory must not expose blocking special files to API clients."""
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from openai_server import APIError, _image_bytes_from_url


class LocalImageFileTest(unittest.TestCase):
    @unittest.skipUnless(hasattr(os, "mkfifo"), "host has no FIFO files")
    def test_fifo_is_refused_without_waiting_for_a_writer(self):
        with tempfile.TemporaryDirectory() as home, \
                patch.dict(os.environ, {"COLI_IMAGE_ROOT": home}):
            fifo = Path(home) / "picture.png"
            os.mkfifo(fifo)
            done = threading.Event()
            results = []

            def read():
                try:
                    results.append(_image_bytes_from_url(str(fifo)))
                except Exception as error:
                    results.append(error)
                finally:
                    done.set()

            reader = threading.Thread(target=read, daemon=True)
            reader.start()
            try:
                self.assertTrue(done.wait(0.5), "API image path blocks waiting for a FIFO writer")
                self.assertIsInstance(results[0], APIError)
                self.assertEqual(results[0].status, 400)
            finally:
                # Release the original blocking implementation after recording
                # the failure, without leaving a blocked test thread behind.
                if not done.is_set():
                    fd = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
                    os.close(fd)
                reader.join(timeout=2)

    def test_regular_file_still_reads_all_bytes(self):
        with tempfile.TemporaryDirectory() as home, \
                patch.dict(os.environ, {"COLI_IMAGE_ROOT": home}):
            image = Path(home) / "picture.png"
            image.write_bytes(b"image bytes")
            self.assertEqual(_image_bytes_from_url(str(image)), b"image bytes")


if __name__ == "__main__":
    unittest.main()
