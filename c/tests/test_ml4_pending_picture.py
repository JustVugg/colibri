"""Mistral-Large-4 serve (colibri.c): a picture goes only to the request its IMAGE frame names.

The gateway sends `IMAGE <id> ...` and then `SUBMIT <id> ...`. A SUBMIT refused before the
picture reaches the tower (CONTEXT_EXCEEDED here; EMPTY_PROMPT, BAD_REQUEST, SLOT_BUSY and
DUPLICATE_ID return at the same point) must not leave it for the next request: that one,
text only, would be answered ERROR BAD_IMAGE. The picture's own request still takes it.

Needs a built `colibri`, COLI_ML4_FIXTURE set to a tiny Mistral-Large-4 container
(tools/make_mistral_large4_tiny.py, then tools/gguf_import.py) and its tower: ML4_MMPROJ set
to `tools/make_mistral_vision_tiny.py --out-hidden 64`'s mmproj.gguf, or one in the container.
Its vocabulary has no [IMG] token, so a picture that reaches the tower check is refused there.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prefix_serve_harness import ServeEngine  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
BINARY = next((path for path in (HERE / "colibri.exe", HERE / "colibri") if path.exists()), None)
FIXTURE = Path(os.environ.get("COLI_ML4_FIXTURE", ""))
CTX = 32
PATCH_FLOATS = 3 * 14 * 14     # the tiny tower: 3 channels, patch 14, merge 2
GRID = 4                       # 4 x 4 patches: 2 x 2 merged rows


@unittest.skipUnless(BINARY, "colibri is not built")
@unittest.skipUnless(FIXTURE.name and (FIXTURE / "config.json").is_file(),
                     "COLI_ML4_FIXTURE not set to a tiny Mistral-Large-4 container with a tower")
class Ml4PendingPictureTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ml4_picture_")
        self.addCleanup(self.tmp.cleanup)
        engine = ServeEngine(BINARY, ["8"],
                             {"SNAP": str(FIXTURE), "SERVE": "1", "SERVE_BATCH": "1", "KV_SLOTS": "1",
                              "CTX": str(CTX), "COLI_TEMP": "0", "KVSAVE": "0",
                              "COLI_USAGE": str(Path(self.tmp.name) / "usage"),
                              "OMP_NUM_THREADS": "2", "COLI_NO_OMP_TUNE": "1"},
                             log_prefix=False)
        self.addCleanup(engine.close)
        self.engine = engine
        self.assertEqual(self.line(), "CAPS vision=1", "the fixture's tower did not load")

    def line(self):
        raw = self.engine.p.stdout.readline()
        if not raw:
            raise RuntimeError("engine closed:\n" + "".join(self.engine._err)[-2000:])
        return raw.decode("latin-1").rstrip("\n")

    def image(self, rid):
        blob = bytes(GRID * GRID * PATCH_FLOATS * 4)
        self.engine.p.stdin.write(f"IMAGE {rid} {len(blob)} {GRID} {GRID}\n".encode() + blob + b"\n")

    def submit(self, rid, n_ids):
        """SUBMIT with n_ids token ids (ids=1); the frame that ends it: ERROR or DONE."""
        ids = " ".join(str(5 + i % 100) for i in range(n_ids)).encode()
        self.engine.p.stdin.write(f"SUBMIT {rid} 0 {len(ids)} 2 0 1 0 ids=1\n".encode() + ids + b"\n")
        self.engine.p.stdin.flush()
        while True:
            text = self.line()
            kind = text.split(" ", 1)[0]
            if kind == "DATA":
                self.engine.p.stdout.read(int(text.split()[2]) + 1)
            elif kind in ("DONE", "ERROR"):
                return text

    def test_a_refused_request_does_not_pass_its_picture_on(self):
        self.image(1)
        self.assertRegex(self.submit(1, CTX + 8), r"^ERROR 1 CONTEXT_EXCEEDED ")
        self.assertRegex(self.submit(2, 6), r"^DONE 2 ")

    def test_the_picture_still_goes_to_its_own_request(self):
        self.image(3)
        self.assertEqual(self.submit(3, 6), "ERROR 3 BAD_IMAGE")
        self.assertRegex(self.submit(4, 6), r"^DONE 4 ")


if __name__ == "__main__":
    unittest.main()
