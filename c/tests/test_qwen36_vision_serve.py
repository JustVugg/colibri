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

    def start(self, snap, slots=1):
        env = dict(os.environ, SNAP=str(snap), SERVE="1", COLI_DENSE_I8="0",
                   OMP_NUM_THREADS="2", COLI_NO_OMP_TUNE="1", KV_SLOTS=str(slots))
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

    def test_several_conversations_with_an_image(self):
        """KV_SLOTS=3: an image turn decodes in the same steps as two text turns, and
        each gives what it gives alone: the image turn's decode rows sit at its own rope
        positions (pos + rope_delta), the others' at theirs."""
        texts = [bytes([3, 4, 5, 6]), bytes([7, 8, 9])]
        p = self.start(FIXTURE)
        alone = []
        for i, text in enumerate(texts):
            verdict, got = self.turn(p, f"t{i}", text)
            self.assertTrue(verdict.startswith("DONE"), verdict)
            alone.append(got)
        p = self.start(FIXTURE, slots=3)
        image, prompt = self.ref["image"], self.prompt()
        blob = struct.pack(f"<{len(image['patches'])}f", *image["patches"])
        w = f"SUBMIT m0 0 {len(texts[0])} 16 0.0 1.0\n".encode() + texts[0] + b"\n"
        w += f"IMAGE m1 {len(blob)} {image['grid_h']} {image['grid_w']}\n".encode() + blob + b"\n"
        w += f"SUBMIT m1 1 {len(prompt)} 16 0.0 1.0\n".encode() + prompt + b"\n"
        w += f"SUBMIT m2 2 {len(texts[1])} 16 0.0 1.0\n".encode() + texts[1] + b"\n"
        p.stdin.write(w)
        p.stdin.flush()
        got, ended = {"m0": b"", "m1": b"", "m2": b""}, {}
        while len(ended) < 3:
            line = p.stdout.readline().decode("utf-8", "replace").strip("\r\n\x01")
            self.assertTrue(line, "the engine closed its output")
            parts = line.split()
            if parts[0] == "DATA":
                n = int(parts[2])
                got[parts[1]] += p.stdout.read(n + 1)[:n]
            elif parts[0] in ("DONE", "ERROR"):
                ended[parts[1]] = line
        for rid in got:
            self.assertTrue(ended[rid].startswith("DONE"), ended[rid])
        want = expected_bytes(self.tokenizer, self.ref["full_ids"][len(self.ref["prompt_ids"]):])
        self.assertEqual(got["m1"], want)
        self.assertEqual(got["m0"], alone[0])
        self.assertEqual(got["m2"], alone[1])

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
