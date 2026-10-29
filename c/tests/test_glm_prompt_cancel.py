"""GLM-5.2 (colibri.c): a CANCEL that arrives while a served prompt is read stops the reading.

The shared contract (prefix_serve_harness.PromptCancelContract), through the multiplexed
protocol the gateway speaks (SERVE=1 SERVE_BATCH=1, one KV slot), against the tiny GLM oracle
model (tools/make_glm_oracle.py). colibri.c reads the CANCEL between two layers with
mux_ctl_poll, the reader its speculative turn uses between tokens, and the slot keeps the
conversation it held before the cancelled prompt: attention alone, so the rows the stopped
layers wrote are past the slot's length.

Needs a built `colibri` and COLI_ORACLE_FIXTURE pointing at the oracle's directory. The
byte-level tokenizer.json is written into a scratch copy, as test_run_serve_context_exceeded
does: the oracle ships none.
"""
import json
import os
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prefix_serve_harness import PromptCancelContract, ServeEngine  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
BINARY = next((path for path in (HERE / "colibri.exe", HERE / "colibri") if path.exists()), None)
FIXTURE = Path(os.environ.get("COLI_ORACLE_FIXTURE", ""))
SLOT_PREFIX = re.compile(r"\[API\] KV slot 0 prefix (\d+)/(\d+) token")


def byte_tokenizer():
    """One token per byte; byte 0 is skipped (the C JSON parser rejects NUL keys)."""
    return {"version": "1.0",
            "model": {"type": "BPE",
                      "vocab": {chr(b): b for b in range(1, 256)},
                      "merges": []},
            "added_tokens": []}


@unittest.skipUnless(BINARY, "colibri is not built")
@unittest.skipUnless(FIXTURE.name and (FIXTURE / "model.safetensors").is_file(),
                     "COLI_ORACLE_FIXTURE not set to the tiny GLM oracle (tools/make_glm_oracle.py)")
class GlmPromptCancelTest(PromptCancelContract, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="glm_cancel_")
        cls.snap = Path(cls.tmp) / "snap"
        shutil.copytree(FIXTURE, cls.snap)
        (cls.snap / "tokenizer.json").write_text(json.dumps(byte_tokenizer()), encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def spawn(self, log_prefix=True, reuse=True):
        # every engine from nothing: colibri.c keeps a slot's KV in <SNAP>/.coli_kv and the next
        # process starts from the conversation the last one held
        for kept in self.snap.glob(".coli_kv*"):
            kept.unlink()
        return ServeEngine(BINARY, ["0"],
                           {"SNAP": str(self.snap), "SERVE": "1", "SERVE_BATCH": "1",
                            "KV_SLOTS": "1", "CTX": "256", "COLI_TEMP": "0",
                            "OMP_NUM_THREADS": "2", "COLI_NO_OMP_TUNE": "1"},
                           log_prefix=log_prefix, reuse=reuse, header_tail=" 0")

    def reused_after_cancel(self, tail):
        """This engine reports a slot's held prefix on every request, not a [PREFIX] line."""
        return any(int(m.group(1)) > 0 for m in SLOT_PREFIX.finditer(tail))


if __name__ == "__main__":
    unittest.main()
