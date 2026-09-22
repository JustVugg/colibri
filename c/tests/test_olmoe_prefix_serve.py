"""KV prefix reuse on OLMoE: the shared contract, against the tiny fixture.

OLMoE is the simple end of the contract -- plain GQA attention, no recurrent
state, and a KV cache allocated once for the context and never reallocated, so
a recorded position stays valid for the life of the process. That makes it the
engine where a reuse bug would be quietest, which is why it is gated too.

Runs against the tiny fixture the oracle job already builds
(tools/make_olmoe_tiny.py + tools/convert_olmoe_merged.py +
tools/make_edge_tiny_tokenizer.py).
"""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prefix_serve_harness import PrefixReuseContract, ServeEngine  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
ENGINE = HERE / ("olmoe.exe" if sys.platform == "win32" else "olmoe")
FIXTURE = Path(os.environ.get("OLMOE_TINY", HERE / "olmoe_tiny_c"))


@unittest.skipUnless(ENGINE.exists(), "olmoe is not built")
@unittest.skipUnless((FIXTURE / "tokenizer.json").exists(),
                     "tiny olmoe container is absent (tools/convert_olmoe_merged.py)")
class OlmoePrefixServeTest(PrefixReuseContract, unittest.TestCase):
    def spawn(self, log_prefix=True, reuse=True):
        return ServeEngine(ENGINE, ["8", "8"],
                           {"SNAP": str(FIXTURE), "SERVE": "1", "CTX": "512"},
                           log_prefix=log_prefix, reuse=reuse)


if __name__ == "__main__":
    unittest.main()
