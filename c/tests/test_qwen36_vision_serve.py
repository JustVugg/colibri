"""Images through qwen36's serve protocol (#1757): IMAGE frame, tower, generation.

The CLI oracle (tools/make_qwen36_vl_tiny.py, ./qwen36 cap 8 ref.json) holds the
engine token for token to transformers. This holds the SERVE path to that same
answer: the prompt goes in as text, the image as the IMAGE frame the gateway
sends, and the bytes that come back must be the reference's tokens decoded.
Then the refusals: patches that do not fit the grid, and placeholders that do
not match it. And a text-only turn still works after an image.

  QWEN36_VL_TINY=<converted container> QWEN36_VL_REF=<fixture>/ref.json \\
      python3 -m unittest tests.test_qwen36_vision_serve
"""
import json
import os
import struct
import subprocess
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ENGINE = HERE / "qwen36"
FIXTURE = os.environ.get("QWEN36_VL_TINY")
REF = os.environ.get("QWEN36_VL_REF")
IMAGE_PAD, START, END = b"<|image_pad|>", b"<|vision_start|>", b"<|vision_end|>"


def expected_bytes(tokenizer, ids):
    """The bytes the engine sends for `ids` with this fixture's tokenizer."""
    by_id = {v: k for k, v in tokenizer["model"]["vocab"].items()}
    specials = {t["id"]: t["content"] for t in tokenizer["added_tokens"]}
    direct = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
    unmap, spare = {}, 0
    for b in range(256):
        if b in direct:
            unmap[chr(b)] = b
        else:
            unmap[chr(256 + spare)] = b
            spare += 1
    out = b""
    for i in ids:
        if i in specials:
            out += specials[i].encode()
        else:
            out += bytes(unmap[c] for c in by_id[i])
    return out


class Qwen36VisionServe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (FIXTURE and REF and ENGINE.is_file()):
            raise unittest.SkipTest("QWEN36_VL_TINY / QWEN36_VL_REF not set, or qwen36 not built")
        cls.ref = json.loads(Path(REF).read_text())
        cls.tokenizer = json.loads((Path(FIXTURE) / "tokenizer.json").read_text())

    def start(self, snap):
        env = dict(os.environ, SNAP=str(snap), SERVE="1", COLI_DENSE_I8="0",
                   OMP_NUM_THREADS="2", COLI_NO_OMP_TUNE="1")
        p = subprocess.Popen([str(ENGINE), "1", "8"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, env=env)
        def stop():
            p.kill()
            p.wait()
            p.stdin.close()
            p.stdout.close()
        self.addCleanup(stop)
        while b"READY" not in p.stdout.readline():
            pass
        return p

    def turn(self, p, rid, prompt, patches=None, grid=(0, 0), max_new=16):
        if patches is not None:
            blob = struct.pack(f"<{len(patches)}f", *patches)
            p.stdin.write(f"IMAGE {rid} {len(blob)} {grid[0]} {grid[1]}\n".encode() + blob + b"\n")
        p.stdin.write(f"SUBMIT {rid} 0 {len(prompt)} {max_new} 0.0 1.0\n".encode() + prompt + b"\n")
        p.stdin.flush()
        got = b""
        while True:
            line = p.stdout.readline().decode("utf-8", "replace").strip("\r\n\x01")
            if not line:
                return "EOF", got
            if line.startswith(f"DATA {rid} "):
                n = int(line.split()[2])
                got += p.stdout.read(n + 1)[:n]
            elif line.startswith(f"DONE {rid}") or line.startswith(f"ERROR {rid}"):
                return line, got

    def prompt(self):
        ids = self.ref["prompt_ids"]
        tokens = ids.count(300)
        return bytes([1, 2, 3]) + START + IMAGE_PAD * tokens + END + bytes([4, 5, 6])

    def test_serve_gives_the_oracle_tokens(self):
        p = self.start(FIXTURE)
        image = self.ref["image"]
        verdict, got = self.turn(p, "a", self.prompt(), image["patches"], (image["grid_h"], image["grid_w"]))
        self.assertTrue(verdict.startswith("DONE"), verdict)
        want = expected_bytes(self.tokenizer, self.ref["full_ids"][len(self.ref["prompt_ids"]):])
        self.assertEqual(got, want)
        # a different picture must change the answer, or the tower's rows are lost
        verdict, other = self.turn(p, "b", self.prompt(), [-v for v in image["patches"]],
                                   (image["grid_h"], image["grid_w"]))
        self.assertTrue(verdict.startswith("DONE"), verdict)
        self.assertNotEqual(other, got)
        # and the turn after an image is text only again
        verdict, _ = self.turn(p, "c", bytes([3, 4, 5]))
        self.assertTrue(verdict.startswith("DONE"), verdict)

    def test_refusals(self):
        p = self.start(FIXTURE)
        image = self.ref["image"]
        grid = (image["grid_h"], image["grid_w"])
        verdict, _ = self.turn(p, "d", self.prompt(), image["patches"][:-8], grid)
        self.assertIn("BAD_IMAGE", verdict)
        short = bytes([1]) + START + IMAGE_PAD * 3 + END
        verdict, _ = self.turn(p, "e", short, image["patches"], grid)
        self.assertIn("BAD_IMAGE", verdict)
        verdict, _ = self.turn(p, "f", bytes([3, 4, 5]))
        self.assertTrue(verdict.startswith("DONE"), verdict)


if __name__ == "__main__":
    unittest.main()
