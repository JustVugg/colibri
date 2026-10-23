"""term_image.py: protocol detection and the four renderers.

The renderers are checked structurally (what a terminal parses) and, where it
is cheap, by decoding: the sixel output is decoded back into pixels by an
independent parser here and compared with the source, and the kitty and
iTerm2 payloads are reassembled and must be byte-identical to the PNG."""
import base64
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import image_engine  # noqa: E402
import term_image as ti  # noqa: E402

UPPER = "▀"


class Tty:
    def __init__(self, tty=True):
        self.tty = tty

    def isatty(self):
        return self.tty

    def fileno(self):
        raise OSError("not a real terminal")


def gradient(width, height, alpha=255):
    out = bytearray()
    for y in range(height):
        for x in range(width):
            out += bytes((x * 255 // max(1, width - 1), y * 255 // max(1, height - 1),
                          (x + y) * 255 // max(1, width + height - 2), alpha))
    return bytes(out)


def solid(width, height, rgba):
    return bytes(rgba) * (width * height)


def decode_sixel(text):
    """An independent sixel reader: raster size, palette, bands, repeats."""
    start = text.index("\x1bPq") + 3
    body = text[start:text.index("\x1b\\", start)]
    match = re.match(r'"1;1;(\d+);(\d+)', body)
    width, height = int(match.group(1)), int(match.group(2))
    body = body[match.end():]
    palette, image = {}, [[None] * width for _ in range(height)]
    x = band = i = 0
    colour = None
    while i < len(body):
        ch = body[i]
        if ch == "#":
            m = re.match(r"#(\d+)(?:;2;(\d+);(\d+);(\d+))?", body[i:])
            i += m.end()
            if m.group(2) is not None:
                palette[int(m.group(1))] = tuple(round(int(m.group(k)) * 255 / 100)
                                                 for k in (2, 3, 4))
            else:
                colour = palette[int(m.group(1))]
            continue
        if ch == "$":
            x, i = 0, i + 1
            continue
        if ch == "-":
            x, band, i = 0, band + 6, i + 1
            continue
        count = 1
        if ch == "!":
            m = re.match(r"!(\d+)(.)", body[i:])
            count, ch = int(m.group(1)), m.group(2)
            i += m.end()
        else:
            i += 1
        value = ord(ch) - 63
        assert 0 <= value < 64, ch
        for _ in range(count):
            for bit in range(6):
                if value >> bit & 1 and band + bit < height:
                    image[band + bit][x] = colour
            x += 1
    return width, height, palette, image


class DetectTest(unittest.TestCase):
    def detect(self, env, tty=True, da1=None):
        return ti.detect(env, Tty(tty), query=False, da1=da1)[0]

    def test_override_wins_and_aliases(self):
        for value, expected in (("kitty", "kitty"), ("iterm2", "iterm"), ("SIXEL", "sixel"),
                                ("blocks", "blocks"), ("off", "none")):
            with self.subTest(value=value):
                self.assertEqual(self.detect({"COLI_IMAGE_PROTOCOL": value, "TERM": "dumb"},
                                             tty=False), expected)

    def test_not_a_terminal_draws_nothing(self):
        self.assertEqual(self.detect({"TERM": "xterm-kitty"}, tty=False), "none")
        self.assertEqual(self.detect({"TERM": "dumb"}), "none")

    def test_known_terminals(self):
        cases = (({"KITTY_WINDOW_ID": "1", "TERM": "xterm-kitty"}, "kitty"),
                 ({"TERM_PROGRAM": "ghostty", "TERM": "xterm-ghostty"}, "kitty"),
                 ({"TERM_PROGRAM": "iTerm.app"}, "iterm"),
                 ({"TERM_PROGRAM": "WezTerm"}, "iterm"),
                 ({"TERM": "xterm-256color"}, "blocks"),
                 ({"TERM_PROGRAM": "Apple_Terminal"}, "blocks"))
        for env, expected in cases:
            with self.subTest(env=env):
                self.assertEqual(self.detect(env), expected)

    def test_tmux_falls_back_to_blocks_even_inside_kitty(self):
        self.assertEqual(self.detect({"TMUX": "/tmp/x", "TERM": "tmux-256color",
                                      "KITTY_WINDOW_ID": "1"}), "blocks")

    def test_sixel_only_when_the_terminal_says_so(self):
        wt = {"WT_SESSION": "abc", "TERM": "xterm-256color"}
        self.assertEqual(self.detect(wt, da1={1, 4, 6, 22}), "sixel")
        self.assertEqual(self.detect(wt, da1={1, 6, 22}), "blocks")
        self.assertEqual(self.detect(wt, da1=None), "blocks")
        self.assertEqual(self.detect({"TERM": "foot"}, da1={4}), "sixel")
        self.assertEqual(self.detect({"TERM": "xterm"}, da1={62, 4}), "sixel")

    def test_vscode_uses_iterm_only_with_images_enabled(self):
        self.assertEqual(self.detect({"TERM_PROGRAM": "vscode"}, da1={62, 4}), "iterm")
        self.assertEqual(self.detect({"TERM_PROGRAM": "vscode"}, da1={62}), "blocks")

    def test_truecolor(self):
        self.assertTrue(ti.truecolor({"COLORTERM": "truecolor"}))
        self.assertFalse(ti.truecolor({"TERM_PROGRAM": "Apple_Terminal"}))


class PixelsTest(unittest.TestCase):
    def test_resize_keeps_solid_colours_and_averages(self):
        red = solid(40, 30, (200, 10, 20, 255))
        self.assertEqual(ti.resize(red, 40, 30, 4, 7, 5), solid(7, 5, (200, 10, 20, 255)))
        checker = bytearray()
        for y in range(8):
            for x in range(8):
                checker += bytes((255, 255, 255, 255) if (x + y) % 2 else (0, 0, 0, 255))
        grey = ti.resize(bytes(checker), 8, 8, 4, 2, 2)
        for value in grey[0::4]:
            self.assertTrue(120 <= value <= 135, value)
        self.assertEqual(ti.resize(red, 40, 30, 4, 40, 30), red)

    def test_resize_upscale_and_three_channels(self):
        tiny = bytes((10, 20, 30, 40, 50, 60))
        big = ti.resize(tiny, 2, 1, 3, 4, 3)
        self.assertEqual(len(big), 4 * 3 * 3)
        self.assertEqual(big[:3], bytes((10, 20, 30)))
        self.assertEqual(big[-3:], bytes((40, 50, 60)))

    def test_flatten_composites_over_a_checkerboard(self):
        opaque = solid(4, 4, (1, 2, 3, 255))
        self.assertEqual(ti.flatten(opaque, 4, 4, 4), bytes((1, 2, 3)) * 16)
        clear = solid(16, 16, (255, 0, 0, 0))
        flat = ti.flatten(clear, 16, 16, 4)
        self.assertEqual(set(flat[0::3]) | set(flat[1::3]), {153, 204})
        half = ti.flatten(solid(1, 1, (255, 255, 255, 128)), 1, 1, 4)
        self.assertTrue(200 < half[0] < 255)

    def test_fit_never_enlarges(self):
        self.assertEqual(ti.fit(768, 512, 100, 80), (100, 66))
        self.assertEqual(ti.fit(64, 32, 100, 80), (64, 32))
        self.assertEqual(ti.fit(512, 768, 100, 40), (26, 40))


class BlocksTest(unittest.TestCase):
    def test_dimensions_and_reset(self):
        text = ti.render_blocks(gradient(768, 512), 768, 512, 4, 96, 100, indent="  ")
        lines = text.rstrip("\n").split("\n")
        self.assertEqual(len(lines), 32)                 # 96 x 64 pixels, 2 per cell
        for line in lines:
            self.assertTrue(line.startswith("  "))
            self.assertTrue(line.endswith("\x1b[0m"))
            self.assertEqual(line.count(UPPER), 96)

    def test_rows_limit_and_odd_heights(self):
        text = ti.render_blocks(gradient(100, 101), 100, 101, 4, 200, 10)
        lines = text.rstrip("\n").split("\n")
        self.assertEqual(len(lines), 10)
        self.assertLessEqual(max(line.count(UPPER) for line in lines), 100)
        odd = ti.render_blocks(gradient(3, 3), 3, 3, 4, 80, 80).rstrip("\n").split("\n")
        self.assertEqual(len(odd), 2)

    def test_colours_are_the_pixels(self):
        top_red_bottom_blue = solid(2, 1, (255, 0, 0, 255)) + solid(2, 1, (0, 0, 255, 255))
        text = ti.render_blocks(top_red_bottom_blue, 2, 2, 4, 80, 80, indent="")
        self.assertIn("\x1b[38;2;255;0;0;48;2;0;0;255m", text)
        # one SGR for a run of identical cells, not one per cell
        self.assertEqual(text.count("\x1b[38;2;"), 1)
        eight = ti.render_blocks(top_red_bottom_blue, 2, 2, 4, 80, 80, indent="",
                                 colour24=False)
        self.assertIn("\x1b[38;5;196;48;5;21m", eight)


class KittyTest(unittest.TestCase):
    def test_chunking_and_payload(self):
        png = image_engine.encode_png(300, 200, gradient(300, 200), 4, level=0)
        text = ti.render_kitty(png, 40, env={})
        commands = re.findall(r"\x1b_G([^;]*);([^\x1b]*)\x1b\\", text)
        self.assertGreater(len(commands), 1)
        first_keys = dict(kv.split("=") for kv in commands[0][0].split(","))
        self.assertEqual((first_keys["a"], first_keys["f"], first_keys["q"], first_keys["c"]),
                         ("T", "100", "2", "40"))
        for i, (keys, payload) in enumerate(commands):
            self.assertLessEqual(len(payload), 4096)
            more = dict(kv.split("=") for kv in keys.split(","))["m"]
            self.assertEqual(more, "0" if i == len(commands) - 1 else "1")
            if i:
                self.assertEqual(keys, f"m={more}")    # later chunks carry only m
        self.assertEqual(base64.b64decode("".join(p for _k, p in commands)), png)

    def test_tmux_passthrough_when_forced(self):
        png = image_engine.encode_png(1, 1, b"\0\0\0\xff", 4)
        text = ti.render_kitty(png, 1, env={"TMUX": "x"})
        self.assertIn("\x1bPtmux;\x1b\x1b_G", text)


class ItermTest(unittest.TestCase):
    def test_header_and_payload(self):
        png = image_engine.encode_png(64, 32, gradient(64, 32), 4)
        text = ti.render_iterm(png, 30, env={})
        match = re.search(r"\x1b\]1337;File=([^:]*):([A-Za-z0-9+/=]*)\x07", text)
        self.assertIsNotNone(match)
        fields = dict(kv.split("=", 1) for kv in match.group(1).split(";"))
        self.assertEqual(fields["inline"], "1")
        self.assertEqual(fields["size"], str(len(png)))
        self.assertEqual(fields["width"], "30")
        self.assertEqual(fields["preserveAspectRatio"], "1")
        self.assertEqual(base64.b64decode(match.group(2)), png)


class SixelTest(unittest.TestCase):
    def test_structure(self):
        text = ti.render_sixel(gradient(120, 60), 120, 60, 4, 1000)
        self.assertTrue(text.lstrip().startswith("\x1bPq"))
        self.assertTrue(text.rstrip("\n").endswith("\x1b\\"))
        self.assertIn('"1;1;120;60', text)
        palette = re.findall(r"#(\d+);2;(\d+);(\d+);(\d+)", text)
        self.assertTrue(palette)
        self.assertLessEqual(len(palette), 256)
        for index, *rgb in palette:
            self.assertLess(int(index), 256)
            self.assertTrue(all(0 <= int(v) <= 100 for v in rgb))
        # six-row bands: 60 rows are exactly ten "-" separators
        body = text[text.index("\x1bPq"):text.index("\x1b\\")]
        self.assertEqual(body.count("-"), 10)
        self.assertRegex(body, r"![0-9]+[?-~]")        # run-length compression is used

    def test_decodes_back_to_the_image(self):
        width, height = 90, 44
        source = gradient(width, height)
        w, h, _palette, image = decode_sixel(ti.render_sixel(source, width, height, 4, 1000))
        self.assertEqual((w, h), (width, height))
        error = 0
        for y in range(height):
            for x in range(width):
                pixel = image[y][x]
                self.assertIsNotNone(pixel, (x, y))    # every pixel painted exactly
                p = (y * width + x) * 4
                error += sum(abs(pixel[c] - source[p + c]) for c in range(3))
        # a 6x7x6 palette with dither: within a quantization step on average
        self.assertLess(error / (width * height * 3), 26)

    def test_scaled_to_the_pixel_budget(self):
        text = ti.render_sixel(gradient(768, 512), 768, 512, 4, 600, 10**6)
        self.assertIn('"1;1;600;400', text)


class RenderDispatchTest(unittest.TestCase):
    def test_every_protocol(self):
        pixels = gradient(64, 32)
        for protocol, marker in (("kitty", "\x1b_G"), ("iterm", "\x1b]1337;File="),
                                 ("sixel", "\x1bPq"), ("blocks", UPPER)):
            with self.subTest(protocol=protocol):
                text = ti.render(pixels, 64, 32, 4, protocol, env={}, stream=Tty(),
                                 max_columns=80, max_rows=40)
                self.assertIn(marker, text)
        self.assertEqual(ti.render(pixels, 64, 32, 4, "none", env={}, stream=Tty()), "")
        with self.assertRaises(ValueError):
            ti.render(pixels, 64, 32, 4, "hologram", env={}, stream=Tty())



class UpscaleSmoothTest(unittest.TestCase):
    """The TUI enlarges the engine's one-pixel-per-token previews with this, so
    they look like a picture forming and not like a mosaic of squares."""

    def test_flat_colour_stays_exact(self):
        big = ti.upscale_smooth(bytes([10, 200, 30]) * 4, 2, 2, 3, 16, 12)
        self.assertEqual(big, bytes([10, 200, 30]) * (16 * 12))

    def test_corners_kept_and_middle_interpolated(self):
        # a black-to-white ramp, 2 pixels wide, enlarged 8x
        small = bytes([0, 0, 0, 255, 255, 255])
        big = ti.upscale_smooth(small, 2, 1, 3, 16, 1)
        row = [big[i * 3] for i in range(16)]
        self.assertEqual(row[0], 0)
        self.assertEqual(row[-1], 255)
        self.assertEqual(row, sorted(row))                 # monotone, no steps back
        self.assertGreater(len(set(row)), 8)               # a ramp, not two blocks

if __name__ == "__main__":
    unittest.main()
