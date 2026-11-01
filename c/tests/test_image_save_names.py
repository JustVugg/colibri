"""Automatic PNG saves preserve images when independent CLIs overlap."""
import importlib.util
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

SOURCE = Path(__file__).resolve().parent.parent / "image_engine.py"
SPEC = importlib.util.spec_from_file_location("image_save_names_module", SOURCE)
IMAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IMAGE)

CHILD = r'''
import importlib.util, json, os, sys
spec = importlib.util.spec_from_file_location("real_image_save", sys.argv[1])
image = importlib.util.module_from_spec(spec)
spec.loader.exec_module(image)
folder, colour = sys.argv[2], int(sys.argv[3])
original_exists = image.os.path.exists
waited = False
def exists(path):
    global waited
    result = original_exists(path)
    if str(path).endswith("-image-s42.png") and not waited:
        waited = True
        print("READY", flush=True)
        assert sys.stdin.readline().strip() == "go"
    return result
# Delay only the first real existence check; do not change syscall results.
image.os.path.exists = exists
png = image.encode_png(2, 2, bytes([colour, 0, 255-colour, 255])*4, 4)
prompt = "\u65e5\u672c\u306e\u732b" if colour else "\u4e2d\u6587\u732b"
path = image.save_png(png, prompt, 42, folder, now=0)
print(json.dumps({"path": path, "colour": colour}), flush=True)
'''


class ImageSaveNamesTest(unittest.TestCase):
    def test_independent_processes_keep_both_pngs(self):
        with tempfile.TemporaryDirectory() as directory:
            env = {"PATH": os.environ.get("PATH", ""), "HOME": directory,
                   "USERPROFILE": directory, "TMPDIR": directory, "TEMP": directory,
                   "TMP": directory, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"}
            for key in ("SystemRoot", "SYSTEMROOT", "WINDIR"):
                if key in os.environ:
                    env[key] = os.environ[key]
            processes, readers = [], []
            try:
                for colour in (0, 255):
                    process = subprocess.Popen(
                        [sys.executable, "-c", CHILD, str(SOURCE), directory, str(colour)],
                        env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, text=True)
                    processes.append(process)
                for process in processes:
                    ready = queue.Queue()
                    thread = threading.Thread(target=lambda stream=process.stdout, q=ready:
                                              q.put(stream.readline()), daemon=True)
                    thread.start()
                    readers.append(thread)
                    self.assertEqual(ready.get(timeout=5).strip(), "READY")
                for process in processes:
                    process.stdin.write("go\n")
                    process.stdin.flush()
                results = []
                for process in processes:
                    output, error = process.communicate(timeout=5)
                    self.assertEqual(process.returncode, 0, error)
                    results.append(json.loads(output))
                self.assertEqual(len({row["path"] for row in results}), 2)
                for row in results:
                    expected = IMAGE.encode_png(
                        2, 2, bytes([row["colour"], 0, 255-row["colour"], 255])*4, 4)
                    self.assertEqual(Path(row["path"]).read_bytes(), expected)
                self.assertEqual(len(list(Path(directory).glob("*.png"))), 2)
                self.assertFalse(list(Path(directory).glob("*.reserve")))
                self.assertFalse(list(Path(directory).glob("*.tmp")))
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                    process.communicate(timeout=5)
                for thread in readers:
                    thread.join(timeout=1)

    def test_existing_images_keep_sequential_suffixes(self):
        with tempfile.TemporaryDirectory() as directory:
            png = IMAGE.encode_png(1, 1, b"\xff\0\0\xff", 4)
            first = IMAGE.save_png(png, "A red fox", 42, directory, now=0)
            second = IMAGE.save_png(png, "A red fox", 42, directory, now=0)
            self.assertEqual(Path(first).read_bytes(), png)
            self.assertEqual(Path(second).read_bytes(), png)
            self.assertTrue(second.endswith("-2.png"))

    def test_abandoned_reservation_skips_only_its_name(self):
        with tempfile.TemporaryDirectory() as directory:
            png = IMAGE.encode_png(1, 1, b"\xff\0\0\xff", 4)
            first = IMAGE.save_png(png, "A red fox", 42, directory, now=0)
            Path(first).unlink()
            reservation = Path(first + ".reserve")
            reservation.write_bytes(b"")
            second = IMAGE.save_png(png, "A red fox", 42, directory, now=0)
            self.assertTrue(second.endswith("-2.png"))
            self.assertEqual(Path(second).read_bytes(), png)
            self.assertTrue(reservation.exists())

    def test_only_complete_png_is_published(self):
        with tempfile.TemporaryDirectory() as directory:
            png = IMAGE.encode_png(1, 1, b"\xff\0\0\xff", 4)
            replace = IMAGE.os.replace
            observed = []

            def publish(temporary, destination):
                self.assertFalse(Path(destination).exists())
                self.assertEqual(Path(temporary).read_bytes(), png)
                observed.append(destination)
                return replace(temporary, destination)

            with mock.patch.object(IMAGE.os, "replace", side_effect=publish):
                saved = IMAGE.save_png(png, "A red fox", 42, directory, now=0)
            self.assertEqual(observed, [saved])
            self.assertEqual(Path(saved).read_bytes(), png)

    def test_failed_publication_removes_owned_temporary_files(self):
        with tempfile.TemporaryDirectory() as directory:
            png = IMAGE.encode_png(1, 1, b"\xff\0\0\xff", 4)
            with mock.patch.object(IMAGE.os, "replace", side_effect=OSError("publication failed")):
                with self.assertRaisesRegex(OSError, "publication failed"):
                    IMAGE.save_png(png, "A red fox", 42, directory, now=0)
            self.assertEqual(list(Path(directory).iterdir()), [])
            saved = IMAGE.save_png(png, "A red fox", 42, directory, now=0)
            self.assertEqual(Path(saved).read_bytes(), png)
            self.assertFalse(saved.endswith("-2.png"))


if __name__ == "__main__":
    unittest.main()
