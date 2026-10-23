"""KV prefix reuse on Qwen3.6: the shared contract, against the tiny fixture.

Qwen3.6 makes the contract stricter than a pure-attention engine would: three of
every four layers are DeltaNet, which carries a RECURRENT state rather than
position-indexed rows. Reuse keeps that state too, so a wrong reuse length
corrupts the recurrence and not just the attention. This engine also GROWS its
KV cache between turns, so tests/prefix_serve_harness.py's three-turn scenario
covers the per-head copy that growth performs.

Runs against the tiny random-init fixture the oracle job already builds
(tools/make_qwen36_tiny.py + tools/convert_qwen36.py): no checkpoint, no disk.
"""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prefix_serve_harness import (PrefixReuseContract, ServeEngine,  # noqa: E402
                                  ensure_byte_tokenizer)

HERE = Path(__file__).resolve().parent.parent
ENGINE = HERE / ("qwen36.exe" if sys.platform == "win32" else "qwen36")
FIXTURE = Path(os.environ.get("QWEN36_TINY", HERE / "qwen36_tiny_c"))


@unittest.skipUnless(ENGINE.exists(), "qwen36 is not built")
# The metadata of this container is TRACKED (config.json, qwen36_meta.json and
# the README), so its presence says nothing about whether the weights were ever
# converted. Gate on a shard the conversion writes, or this skips nothing and
# fails on a checkout that never ran the converter.
@unittest.skipUnless((FIXTURE / "model-globals.safetensors").exists(),
                     "tiny qwen36 container is absent (tools/convert_qwen36.py)")
class Qwen36PrefixServeTest(PrefixReuseContract, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The fixture ships no tokenizer: the oracle path feeds ids directly.
        ensure_byte_tokenizer(FIXTURE)

    def spawn(self, log_prefix=True, reuse=True):
        return ServeEngine(ENGINE, ["8", "8"],
                           {"SNAP": str(FIXTURE), "SERVE": "1", "COLI_DENSE_I8": "0"},
                           log_prefix=log_prefix, reuse=reuse)


if __name__ == "__main__":
    unittest.main()
