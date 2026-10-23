"""image_engine.py on its own: PNG, request rules, the protocol client, saving.

The PNG codec is checked against zlib-level ground truth (a hand-built file
with every filter type) as well as round trips, because a codec that only
ever decodes its own output can agree with itself about a wrong format.
"""
import json
import os
import struct
import sys
import tempfile
import threading
import time
import unittest
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qwen_image_fixture import STUB, make_fake_pipeline, stub_env  # noqa: E402

import image_engine as ie  # noqa: E402

INFO = dict(ie.DEFAULT_INFO)


def _chunk(kind, data):
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xffffffff)


def _png_with_filters(width, height, channels, pixels, filters):
    """A PNG whose rows use the given filter types, encoded straight from the
    PNG specification (independently of image_engine's encoder)."""
    stride = width * channels
    raw = bytearray()
    previous = bytes(stride)
    for y in range(height):
        row = pixels[y * stride:(y + 1) * stride]
        kind = filters[y % len(filters)]
        out = bytearray(stride)
        for i in range(stride):
            a = row[i - channels] if i >= channels else 0
            b = previous[i]
            c = previous[i - channels] if i >= channels else 0
            if kind == 0:
                pred = 0
            elif kind == 1:
                pred = a
            elif kind == 2:
                pred = b
            elif kind == 3:
                pred = (a + b) >> 1
            else:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
            out[i] = (row[i] - pred) & 255
        raw.append(kind)
        raw += out
        previous = row
    colour = {1: 0, 2: 4, 3: 2, 4: 6}[channels]
    return (b"\x89PNG\r\n\x1a\n" +
            _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, colour, 0, 0, 0)) +
            _chunk(b"IDAT", zlib.compress(bytes(raw))) + _chunk(b"IEND", b""))


def _noise(count, seed=1):
    state = seed
    out = bytearray(count)
    for i in range(count):
        state = (state * 1103515245 + 12345) & 0x7fffffff
        out[i] = state >> 16 & 255
    return bytes(out)


class PngTest(unittest.TestCase):
    def test_round_trip_every_channel_count(self):
        for channels in (1, 2, 3, 4):
            with self.subTest(channels=channels):
                width, height = 37, 23
                pixels = _noise(width * height * channels, channels)
                png = ie.encode_png(width, height, pixels, channels)
                self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
                self.assertEqual(ie.decode_png(png), (width, height, channels, pixels))

    def test_decodes_every_filter_type(self):
        width, height, channels = 19, 10, 4
        pixels = _noise(width * height * channels, 7)
        png = _png_with_filters(width, height, channels, pixels, [0, 1, 2, 3, 4])
        self.assertEqual(ie.decode_png(png)[3], pixels)

    def test_encoder_output_is_a_valid_png_by_the_spec(self):
        """Decode our PNG without our decoder: chunk CRCs, IHDR, and each row's
        Up filter undone by hand."""
        width, height = 5, 4
        pixels = _noise(width * height * 4, 3)
        png = ie.encode_png(width, height, pixels, 4)
        pos, idat, ihdr = 8, b"", None
        while pos < len(png):
            length, kind = struct.unpack(">I4s", png[pos:pos + 8])
            body = png[pos + 8:pos + 8 + length]
            crc = struct.unpack(">I", png[pos + 8 + length:pos + 12 + length])[0]
            self.assertEqual(zlib.crc32(kind + body) & 0xffffffff, crc, kind)
            if kind == b"IHDR":
                ihdr = struct.unpack(">IIBBBBB", body)
            elif kind == b"IDAT":
                idat += body
            pos += 12 + length
        self.assertEqual(ihdr, (width, height, 8, 6, 0, 0, 0))
        raw = zlib.decompress(idat)
        stride = width * 4
        previous = bytes(stride)
        for y in range(height):
            kind = raw[y * (stride + 1)]
            row = raw[y * (stride + 1) + 1:(y + 1) * (stride + 1)]
            self.assertIn(kind, (0, 2))
            if kind == 2:
                row = bytes((v + p) & 255 for v, p in zip(row, previous))
            self.assertEqual(row, pixels[y * stride:(y + 1) * stride])
            previous = row

    def test_corruption_and_unsupported_formats_are_refused(self):
        png = bytearray(ie.encode_png(4, 4, bytes(64), 4))
        png[-20] ^= 0xff
        with self.assertRaises(ValueError):
            ie.decode_png(bytes(png))
        with self.assertRaises(ValueError):
            ie.decode_png(b"GIF89a")
        sixteen = (b"\x89PNG\r\n\x1a\n" +
                   _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 16, 2, 0, 0, 0)) +
                   _chunk(b"IDAT", zlib.compress(bytes(7))) + _chunk(b"IEND", b""))
        with self.assertRaisesRegex(ValueError, "8-bit"):
            ie.decode_png(sixteen)
        with self.assertRaises(ValueError):
            ie.encode_png(4, 4, bytes(10), 4)


class RequestRulesTest(unittest.TestCase):
    def test_defaults_come_from_ready(self):
        request = ie.image_request({"prompt": "a cat", "seed": 3}, INFO)
        self.assertEqual((request["width"], request["height"], request["steps"],
                          request["seed"]), (768, 512, 8, 3))
        info = dict(INFO, default_width=1024, default_height=576, default_steps=4)
        request = ie.image_request({"prompt": "a cat"}, info)
        self.assertEqual((request["width"], request["height"], request["steps"]), (1024, 576, 4))
        self.assertFalse(request["seed_given"])
        self.assertTrue(0 <= request["seed"] <= ie.MAX_SEED)

    def test_the_contract_sizes(self):
        for size in ("768x512", "1024x576", "256x2048", "2048x2048", "512X288", " 512 x 512 "):
            with self.subTest(size=size):
                ie.image_request({"prompt": "x", "size": size}, INFO)
        for size in ("768x432", "250x256", "224x256", "2080x256", "768", "x", "0x0"):
            with self.subTest(size=size):
                with self.assertRaises(ie.ImageRequestError) as caught:
                    ie.image_request({"prompt": "x", "size": size}, INFO)
                self.assertEqual(caught.exception.param, "size")
        self.assertEqual(ie.nearest_size(768, 432, INFO), (768, 448))
        for preset in ie.SIZE_PRESETS:
            self.assertIsNone(ie.check_size(*preset, INFO), preset)

    def test_each_parameter_names_itself(self):
        cases = (({"prompt": ""}, "prompt"), ({"prompt": "x", "n": 2}, "n"),
                 ({"prompt": "x", "steps": 0}, "steps"), ({"prompt": "x", "seed": -5}, "seed"),
                 ({"prompt": "x", "seed": 2**40}, "seed"),
                 ({"prompt": "x", "width": 300}, "width"),
                 ({"prompt": "x", "height": 100}, "height"),
                 ({"prompt": "x", "response_format": "url"}, "response_format"),
                 ({"prompt": "x", "size": "512x512", "height": 512}, "size"))
        for body, param in cases:
            with self.subTest(body=body):
                with self.assertRaises(ie.ImageRequestError) as caught:
                    ie.image_request(body, INFO)
                self.assertEqual(caught.exception.param, param)


class ImageEngineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.model = make_fake_pipeline(Path(self.tmp.name) / "m")
        self.logs = []

    def engine(self, delay=0.0, **extra):
        engine = ie.ImageEngine(STUB, self.model, env=stub_env(delay, **extra),
                                on_log=self.logs.append, on_stderr=self.logs.append)
        self.addCleanup(engine.close, 5)
        return engine

    def test_ready_carries_the_contract_fields(self):
        info = self.engine().info
        for key, value in (("default_width", 768), ("default_height", 512), ("multiple", 32),
                           ("min_side", 256), ("max_side", 2048), ("default_steps", 8)):
            self.assertEqual(info[key], value)

    def test_generate_reports_progress_previews_and_the_image(self):
        engine = self.engine()
        progress, previews = [], []
        result = engine.generate("a cat", 256, 288, 3, 9, preview=True,
                                 on_progress=progress.append,
                                 on_preview=lambda meta, px: previews.append((meta, len(px))))
        self.assertEqual((result["width"], result["height"], result["seed"], result["steps"]),
                         (256, 288, 9, 3))
        self.assertEqual(len(result["rgba"]), 256 * 288 * 4)
        self.assertEqual([p["stage"] for p in progress][0], "encode")
        self.assertEqual(progress[-1]["stage"], "decode")
        self.assertEqual(len(previews), 3)
        meta, size = previews[0]
        self.assertEqual(size, meta["width"] * meta["height"] * 3)
        # stderr was drained from the first byte, into our callable
        self.assertTrue(any("stub" in line for line in self.logs))

    def test_engine_error_leaves_the_engine_usable(self):
        engine = self.engine()
        with self.assertRaisesRegex(ie.ImageEngineError, "stub failure"):
            engine.generate("__stub_error__", 256, 256, 2, 1)
        with self.assertRaisesRegex(ie.ImageEngineError, "multiple of 32"):
            engine.generate("x", 250, 256, 2, 1)      # the engine's own validation
        self.assertTrue(engine.alive)
        self.assertEqual(engine.generate("x", 256, 256, 2, 1)["width"], 256)

    def test_cancel_between_steps(self):
        engine = self.engine(delay=0.2)
        seen = []
        started = time.time()
        with self.assertRaises(ie.ImageCancelled):
            engine.generate("x", 256, 256, 30, 1, on_progress=seen.append,
                            cancelled=lambda: len(seen) >= 2)
        self.assertLess(time.time() - started, 4)       # 30 steps would take 6 s
        self.assertTrue(engine.alive)
        self.assertEqual(engine.generate("y", 256, 256, 2, 2)["seed"], 2)

    def test_a_crash_is_reported_and_the_engine_marked_dead(self):
        engine = self.engine()
        with self.assertRaisesRegex(ie.ImageEngineError, "exited"):
            engine.generate("__stub_crash__", 256, 256, 4, 1)
        self.assertFalse(engine.alive)
        with self.assertRaises(ie.ImageEngineError):
            engine.generate("x", 256, 256, 2, 1)

    def test_stray_output_is_ignored(self):
        engine = self.engine()
        result = engine.generate("__stub_garbage__", 256, 256, 2, 1)
        self.assertEqual(result["width"], 256)
        self.assertTrue(any("stray" in line for line in self.logs))

    def test_an_engine_that_dies_while_loading(self):
        with self.assertRaisesRegex(ie.ImageEngineError, "exited while loading"):
            ie.ImageEngine(STUB, str(Path(self.tmp.name) / "missing"), env=stub_env(),
                           on_stderr=lambda _l: None)

    def test_a_load_timeout(self):
        with self.assertRaisesRegex(ie.ImageEngineError, "did not become ready"):
            ie.ImageEngine(STUB, self.model, env=stub_env(QWENIMAGE_STUB_LOAD_DELAY=5),
                           load_timeout=0.5, on_stderr=lambda _l: None)

    def test_engine_command_runs_a_python_stub_under_this_interpreter(self):
        self.assertEqual(ie.engine_command("x/qwenimage", "m"), ["x/qwenimage", "--model", "m",
                                                                  "--serve"])
        self.assertEqual(ie.engine_command("t/stub.py", "m")[:2], [sys.executable, "t/stub.py"])


class SavingTest(unittest.TestCase):
    def test_names_describe_and_never_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            png = ie.encode_png(1, 1, b"\0\0\0\0", 4)
            first = ie.save_png(png, "A red fox, in the snow!", 42, tmp, now=0)
            second = ie.save_png(png, "A red fox, in the snow!", 42, tmp, now=0)
            self.assertNotEqual(first, second)
            self.assertTrue(os.path.basename(first).endswith("-a-red-fox-in-the-snow-s42.png"))
            self.assertTrue(second.endswith("-s42-2.png"))
            unicode_name = ie.save_png(png, "日本の猫", 1, tmp, now=0)
            self.assertIn("-image-s1", unicode_name)
            self.assertEqual(sorted(os.listdir(tmp)), sorted(
                os.path.basename(p) for p in (first, second, unicode_name)))

    def test_default_directory_and_override(self):
        self.assertTrue(ie.image_dir({}).endswith("colibri-images"))
        self.assertEqual(ie.image_dir({"COLI_IMAGE_DIR": "/x/y"}), "/x/y")


if __name__ == "__main__":
    unittest.main()
