"""`coli chat` surfaces the engine's CONTEXT_EXCEEDED refusal instead of an empty answer.

The legacy run_serve protocol has no ERROR frame: on an over-context turn the
engine emits an empty answer plus `STAT ... CONTEXT_EXCEEDED <cause>` (#2097).
The reason used to go only to the engine's stderr, which `coli chat` drains
into a temporary log and never shows, so the user saw empty answers until they
typed `:reset`, without being told to. The chat loop now reads the token on
the STAT line and prints the reason with the hint that actually fixes it.

The test runs the real cmd_chat in a child Python against a fake engine that
speaks the chat byte protocol and refuses every turn with a chosen cause
spelling, and asserts the user sees the reason plus the right hint.
"""
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

CLI = Path(__file__).resolve().parent.parent / "coli"

ENGINE = textwrap.dedent("""\
    import sys
    cause = sys.argv[1]
    err, out = sys.stderr.buffer, sys.stdout.buffer
    out.write(b"\\x01\\x01READY\\x01\\x01\\n"); out.flush()
    out.write(b"STAT 0 0.00 0.0 1.00\\n"); out.flush()
    out.write(b"TIERS 0 0 0 0 0\\n"); out.flush()
    for line in sys.stdin.buffer:
        if line.strip() == b"\\x02RESET":
            out.write(b"\\x01\\x01END\\x01\\x01\\n"); out.flush()
            out.write(b"STAT 0 0.00 0.0 1.00\\n"); out.flush()
            continue
        # An over-context turn: empty answer, STAT refusal, no stderr reason
        # (the real engine's reason lives in the temp log the chat never shows).
        stat = b"STAT 0 0.00 0.0 1.00 99 0 CONTEXT_EXCEEDED"
        if cause:
            stat += b" " + cause.encode()
        out.write(b"\\n\\x01\\x01END\\x01\\x01\\n"); out.flush()
        out.write(stat + b"\\n"); out.flush()
""")

DRIVER = textwrap.dedent("""\
    import importlib.util, io, json, os, subprocess, sys, tempfile
    from importlib.machinery import SourceFileLoader
    from pathlib import Path
    from unittest import mock

    cli, engine, cause = sys.argv[1], sys.argv[2], sys.argv[3]
    loader = SourceFileLoader("coli_chat_ctx", cli)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    coli = importlib.util.module_from_spec(spec)
    loader.exec_module(coli)

    model = tempfile.mkdtemp()
    Path(model, "config.json").write_text('{"model_type": "glm_moe_dsa"}')
    real_popen = subprocess.Popen

    class Spinner:
        def __init__(self, label, tick=None): pass
        def start(self): pass
        def stop(self): pass

    prompts = iter(["first question", "second question"])
    sent = []
    def fake_input(*_):
        try:
            sent.append(next(prompts)); return sent[-1]
        except StopIteration:
            raise EOFError

    args = coli.argparse.Namespace(model=model, cap=None, ngen=16, ram=0, topp=0, topk=0,
                                   no_attach=True, attach=None, api_key=None, stats="off",
                                   temp=None, think=None, effort=None, ctx=0)
    out = io.StringIO()
    with mock.patch.object(coli, "need_model"), \\
         mock.patch.object(coli, "banner"), \\
         mock.patch.object(coli, "env_for", return_value=dict(os.environ)), \\
         mock.patch.object(coli, "TTY", False), \\
         mock.patch.object(coli, "Spinner", Spinner), \\
         mock.patch.object(coli.subprocess, "Popen",
                           lambda cmd, **kw: real_popen([sys.executable, engine, cause], **kw)), \\
         mock.patch("builtins.input", fake_input), \\
         mock.patch.object(sys, "stdout", out):
        try:
            coli.cmd_chat(args)
        except SystemExit:
            pass
    print(json.dumps({"output": out.getvalue(), "turns": len(sent)}, ensure_ascii=True))
""")


def chat(cause):
    with tempfile.TemporaryDirectory() as scratch:
        engine = Path(scratch, "engine.py")
        engine.write_text(ENGINE, encoding="utf-8")
        driver = Path(scratch, "driver.py")
        driver.write_text(DRIVER, encoding="utf-8")
        try:
            result = subprocess.run([sys.executable, str(driver), str(CLI), str(engine), cause],
                                    capture_output=True, timeout=90)
        except subprocess.TimeoutExpired:
            raise AssertionError("`coli chat` hung talking to the refusing engine")
    if result.returncode:
        raise AssertionError(result.stderr.decode("utf-8", "replace")[-3000:])
    report = json.loads(result.stdout.decode("utf-8").strip().splitlines()[-1])
    return report


class ChatContextExceededTest(unittest.TestCase):
    def test_history_full_names_the_reset_hint(self):
        report = chat("HISTORY_FULL")
        self.assertEqual(report["turns"], 2)
        self.assertIn("the conversation history no longer fits the context", report["output"])
        self.assertIn("type :reset to clear it and continue", report["output"])

    def test_prompt_too_long_says_shorten_or_raise_ctx(self):
        report = chat("PROMPT_TOO_LONG")
        self.assertIn("this turn does not fit the context", report["output"])
        self.assertIn("shorten the prompt or raise CTX", report["output"])

    def test_render_clipped_says_paste_shorter_chunks(self):
        report = chat("RENDER_CLIPPED")
        self.assertIn("exceeded the 64KB render buffer", report["output"])
        self.assertIn("paste it in shorter chunks", report["output"])

    def test_older_engine_without_a_cause_token_still_explains(self):
        report = chat("")
        self.assertIn("[context is full:", report["output"])
        self.assertIn("shorten the prompt or raise CTX", report["output"])


if __name__ == "__main__":
    unittest.main()
