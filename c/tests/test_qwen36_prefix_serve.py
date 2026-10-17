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
import json
import os
import subprocess
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prefix_serve_harness import (PrefixReuseContract, ServeEngine,  # noqa: E402
                                  ensure_byte_tokenizer)

HERE = Path(__file__).resolve().parent.parent
ENGINE = HERE / ("qwen36.exe" if sys.platform == "win32" else "qwen36")
FIXTURE = Path(os.environ.get("QWEN36_TINY", HERE / "qwen36_tiny_c"))


def cpu_prefill_engine(cap, dense, slots):
    # This comparison owns the CPU polling path, even when Vulkan CI runs this
    # module. A single-slot Vulkan chain and multi-slot fallback are different
    # execution paths, not a bit-exact oracle for CPU cancellation.
    return ServeEngine(ENGINE, [str(cap), "8"],
                       {"SNAP": str(FIXTURE), "SERVE": "1",
                        "COLI_DENSE_I8": dense, "KV_SLOTS": str(slots),
                        "COLI_VULKAN": "0", "COLI_VK_CHAIN": "0",
                        "COLI_CUDA": "0", "COLI_CUDA_CHAIN": "0"})


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

    def watch(self, engine):
        # A lost control frame must fail, not block CI in stdout.readline().
        timer = threading.Timer(30, engine.p.kill)
        timer.daemon = True
        timer.start()
        self.addCleanup(timer.cancel)
        self.addCleanup(engine.close)
        return engine

    def test_queued_cancel_skips_prefill_and_emits_no_data(self):
        self.control_before_prefill("CANCEL")

    def test_queued_stop_is_successful_and_preserves_next_turn(self):
        self.control_before_prefill("STOP")

    def control_before_prefill(self, command):
        engine = ServeEngine(
            ENGINE,
            ["8", "8"],
            {"SNAP": str(FIXTURE), "SERVE": "1", "COLI_DENSE_I8": "0"},
            log_prefix=True,
        )
        self.watch(engine)
        prompt = b"x" * 400
        # Queue the control with the request, so this checks wire semantics
        # without racing completion on fast hosts. The native callback test
        # separately interrupts at EVERY real hybrid layer boundary.
        engine.p.stdin.write(
            f"SUBMIT cancel-prefill 0 {len(prompt)} 4 0 1\n".encode()
            + prompt + b"\n" + f"{command} cancel-prefill\n".encode()
        )
        engine.p.stdin.flush()
        accepted = False
        while True:
            line = engine.p.stdout.readline()
            self.assertTrue(line, "engine exited before acknowledging cancellation")
            text = line.decode("latin-1").strip()
            if text.startswith("ACCEPT cancel-prefill "):
                accepted = True
            if text.startswith("DATA cancel-prefill "):
                self.fail("engine generated data before honoring a prefill cancellation")
            if text.startswith(("ERROR cancel-prefill ", "DONE cancel-prefill ")):
                self.assertTrue(accepted, "terminal frame before ACCEPT")
                if command == "CANCEL":
                    self.assertEqual(text, "ERROR cancel-prefill CANCELLED")
                else:
                    self.assertTrue(text.startswith("DONE cancel-prefill STAT 0 "), text)
                    self.assertEqual(text.split()[-1], "0", "STOP is not length-limited")
                break

        # Cancellation must not poison the persistent process or its next turn.
        actual = engine.ask("after-cancel", prompt + b"y", max_tok=4)
        fresh = self.spawn(log_prefix=False)
        self.watch(fresh)
        self.assertEqual(actual, fresh.ask("fresh", prompt + b"y", max_tok=4))
        log = engine.close()
        self.assertNotIn("[PREFIX] reusing 400 of 401 prompt tokens", log,
                         f"{command} was acknowledged only after all 400 prompt tokens ran")

    def test_layer_polled_prefill_matches_unpolled_mux(self):
        opening = b"Repeated prefix. " * 20
        prompts = [opening, opening + b" More context. " * 12,
                   b"def process(items): return sorted(set(items))\n" * 7]
        for dense in ("0", "1"):
            for cap in (1, 8):
                with self.subTest(cap=cap, dense=dense):
                    polled = self.watch(cpu_prefill_engine(cap, dense, 1))
                    whole = self.watch(cpu_prefill_engine(cap, dense, 2))
                    try:
                        for index, prompt in enumerate(prompts):
                            # One token on the opening leaves exactly its prompt
                            # in the KV state; turn 2 must then reuse that prefix.
                            count = 1 if index == 0 else 8
                            self.assertEqual(polled.ask(f"polled-{index}", prompt, max_tok=count),
                                             whole.ask(f"whole-{index}", prompt, max_tok=count))
                        self.assertIn(f"[PREFIX] reusing {len(opening)} of", polled.close())
                    finally:
                        polled.close()
                        whole.close()

    def test_multiplexed_submissions_are_not_consumed_by_cancel_poll(self):
        wire = b""
        for slot, rid in enumerate(("mux-a", "mux-b")):
            prompt = b"x" * 200 + bytes([65 + slot])
            wire += f"SUBMIT {rid} {slot} {len(prompt)} 4 0 1\n".encode() + prompt + b"\n"
        result = subprocess.run([str(ENGINE), "8", "8"], input=wire, capture_output=True,
                                timeout=30, env=dict(os.environ, SNAP=str(FIXTURE),
                                SERVE="1", KV_SLOTS="2", COLI_DENSE_I8="0"))
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        for rid in ("mux-a", "mux-b"):
            self.assertIn(f"DONE {rid} STAT".encode(), result.stdout)
        self.assertNotIn(b"ERROR ", result.stdout)


class Qwen36PrefillEnvironmentTest(unittest.TestCase):
    """Probe child environments without requiring a model or a GPU."""

    ACCELERATORS = {"COLI_VULKAN": "1", "COLI_VK_CHAIN": "1",
                    "COLI_CUDA": "1", "COLI_CUDA_CHAIN": "1"}

    def capture(self, spawn):
        keys = [*self.ACCELERATORS, "COLI_DENSE_I8", "KV_SLOTS"]
        probe = (
            "import json, os, sys\n"
            f"print(json.dumps({{k: os.environ.get(k) for k in {keys!r}}}), "
            "file=sys.stderr, flush=True)\n"
            "sys.stdout.buffer.write(b'\\x01\\x01READY\\x01\\x01\\n')\n"
            "sys.stdout.buffer.flush()\n"
        )
        actual_popen = subprocess.Popen

        def child(_command, **kwargs):
            # Only replace the executable: exercise ServeEngine's real merging
            # of the parent environment and per-child overrides.
            return actual_popen([sys.executable, "-c", probe], **kwargs)

        with mock.patch.dict(os.environ, self.ACCELERATORS), \
                mock.patch("prefix_serve_harness.subprocess.Popen", side_effect=child):
            engine = spawn()
            try:
                captured = json.loads(engine.close())
                self.assertEqual(engine.p.returncode, 0)
                for key, value in self.ACCELERATORS.items():
                    self.assertEqual(os.environ[key], value)
                return captured
            finally:
                engine.close()
                engine.p.stdout.close()
                engine.p.stderr.close()

    def test_cpu_comparison_overrides_inherited_accelerators(self):
        for dense in ("0", "1"):
            for slots in (1, 2):
                with self.subTest(dense=dense, slots=slots):
                    env = self.capture(lambda: cpu_prefill_engine(1, dense, slots))
                    for key in self.ACCELERATORS:
                        self.assertEqual(env[key], "0", key)
                    self.assertEqual(env["COLI_DENSE_I8"], dense)
                    self.assertEqual(env["KV_SLOTS"], str(slots))

    def test_shared_prefix_contract_keeps_inherited_accelerators(self):
        case = Qwen36PrefixServeTest("test_layer_polled_prefill_matches_unpolled_mux")
        env = self.capture(case.spawn)
        for key, value in self.ACCELERATORS.items():
            self.assertEqual(env[key], value)


if __name__ == "__main__":
    unittest.main()
