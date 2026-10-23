"""`coli chat` on GLM-5.2 must keep reading the engine's stderr whatever the locale.

cmd_chat drains the engine's stderr into a temporary log opened in the
locale's code page. The engine writes UTF-8: its per-layer progress line is
"[prefill] layer 1/78 · 12 token · +0.10s" and several status lines carry an
em dash. On a code page without those characters (cp932 has no U+00B7, cp949
and cp932 have no U+2014, and the C locale has neither) the first such line
raised UnicodeEncodeError in the drain thread, which caught it as a ValueError
and stopped. From then on:

  * the load status lines after it were never shown,
  * the engine's last words were missing when it died,
  * nobody read the stderr pipe any more, so once the engine's [prefill] lines
    filled the pipe buffer the engine blocked on its next write and the chat
    hung in the middle of a turn, for good.

The test runs the real cmd_chat in a child Python whose locale cannot encode
those characters (PYTHONUTF8=0 and the C locale, or a CJK Windows code page),
against a fake engine that speaks the chat byte protocol. It is skipped where
the child's locale encodes them anyway (macOS, cp1252), because there the
failure cannot happen.
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
LINES = ("—", "·")

CHILD_ENV = {**os.environ, "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0",
             "LC_ALL": "C", "LANG": "C"}
CHILD_ENV.pop("PYTHONIOENCODING", None)

READY = "b'\\x01\\x01READY\\x01\\x01\\nSTAT 0 0.00 0.0 1.00\\nTIERS 0 0 0 0 0\\n'"
END = "b'ok\\x01\\x01END\\x01\\x01\\nSTAT 1 1.00 0.0 1.00\\n'"

ENGINES = {
    # Dies while loading, with a reason that carries an em dash.
    "load_failure": textwrap.dedent("""
        import sys
        sys.stderr.buffer.write("model-00007.safetensors: short read at EOF \\u2014 truncated shard?\\n".encode())
        sys.stderr.buffer.flush()
        sys.exit(1)
    """),
    # Loads with an em-dash status line before a whitelisted one, then answers
    # every prompt after a long chunked prefill: enough [prefill] lines to fill
    # any OS pipe buffer (64 KiB on Linux, 4 KiB on Windows) in one turn.
    "chat": textwrap.dedent(f"""
        import sys
        err, out = sys.stderr.buffer, sys.stdout.buffer
        err.write("[MIRROR] /mnt/m equals the model dir \\u2014 ignored\\n".encode())
        err.write(b"[RAM_GB] budget 42.0 GB\\n"); err.flush()
        out.write({READY}); out.flush()
        for line in sys.stdin.buffer:
            for chunk in range(100):
                for layer in range(0, 78, 4):
                    err.write(f"[prefill] layer {{layer+1}}/78 \\u00b7 512 token \\u00b7 +0.10s\\n".encode())
            err.flush()
            out.write({END}); out.flush()
    """),
    # Loads, then dies in the middle of the first turn saying why.
    "dies_mid_turn": textwrap.dedent(f"""
        import sys
        err, out = sys.stderr.buffer, sys.stdout.buffer
        out.write({READY}); out.flush()
        sys.stdin.buffer.readline()
        err.write("model-00007.safetensors: short read at EOF \\u2014 truncated shard?\\n".encode())
        err.flush()
        sys.exit(1)
    """),
}

DRIVER = textwrap.dedent("""
    import importlib.util, io, json, locale, os, subprocess, sys, tempfile, time
    from importlib.machinery import SourceFileLoader
    from pathlib import Path
    from unittest import mock

    cli, engine, turns, engine_writes_progress = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    if all(ch.encode(locale.getpreferredencoding(False), "ignore") for ch in %r):
        print(json.dumps({"skip": locale.getpreferredencoding(False)})); sys.exit(0)
    loader = SourceFileLoader("coli_chat_stderr", cli)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    coli = importlib.util.module_from_spec(spec)
    loader.exec_module(coli)

    model = tempfile.mkdtemp()
    Path(model, "config.json").write_text('{"model_type": "glm_moe_dsa"}')
    real_popen = subprocess.Popen
    ticks, sent = [], []

    class Spinner:
        # The real one only ticks on a TTY; poll the progress reader directly.
        def __init__(self, label, tick=None):
            self.tick = tick
        def start(self):
            if not self.tick or engine_writes_progress != "1":
                return
            deadline = time.time() + 20
            while time.time() < deadline:
                value = self.tick()
                if value:
                    ticks.append(value); return
                time.sleep(0.05)
            ticks.append("")
        def stop(self):
            pass

    prompts = iter("question %%d" %% i for i in range(turns))
    def fake_input(*_):
        try:
            sent.append(next(prompts)); return sent[-1]
        except StopIteration:
            raise EOFError

    args = coli.argparse.Namespace(model=model, cap=None, ngen=16, ram=0, topp=0, topk=0,
                                   no_attach=True, attach=None, api_key=None, stats="off",
                                   temp=None, think=None, effort=None, ctx=0)
    out = io.StringIO()
    code = None
    with mock.patch.object(coli, "need_model"), \\
         mock.patch.object(coli, "banner"), \\
         mock.patch.object(coli, "env_for", return_value=dict(os.environ)), \\
         mock.patch.object(coli, "TTY", False), \\
         mock.patch.object(coli, "Spinner", Spinner), \\
         mock.patch.object(coli.subprocess, "Popen",
                           lambda cmd, **kw: real_popen([sys.executable, engine], **kw)), \\
         mock.patch("builtins.input", fake_input), \\
         mock.patch.object(sys, "stdout", out):
        try:
            coli.cmd_chat(args)
        except SystemExit as exit_:
            code = str(exit_)
    print(json.dumps({"output": out.getvalue(), "turns": len(sent), "ticks": ticks,
                      "exit": code}, ensure_ascii=True))
""") % (LINES,)


def chat(scenario, turns=2):
    with tempfile.TemporaryDirectory() as scratch:
        engine = Path(scratch, "engine.py")
        engine.write_text(ENGINES[scenario], encoding="utf-8")
        driver = Path(scratch, "driver.py")
        driver.write_text(DRIVER, encoding="utf-8")
        try:
            result = subprocess.run([sys.executable, str(driver), str(CLI), str(engine), str(turns),
                                     "1" if scenario == "chat" else "0"],
                                    env=CHILD_ENV, capture_output=True, timeout=90)
        except subprocess.TimeoutExpired:
            raise AssertionError(f"{scenario}: `coli chat` hung (engine blocked on a full stderr pipe)")
    if result.returncode:
        raise AssertionError(result.stderr.decode("utf-8", "replace")[-3000:])
    report = json.loads(result.stdout.decode("ascii").strip().splitlines()[-1])
    if "skip" in report:
        raise unittest.SkipTest(f"the child's locale ({report['skip']}) encodes every character "
                                f"the engine writes, so the drain cannot stop here")
    return report


class ChatEngineStderrTest(unittest.TestCase):
    def test_status_lines_progress_and_turns_survive_a_non_ascii_line(self):
        report = chat("chat", turns=2)
        self.assertEqual(report["turns"], 2)
        self.assertIn("[RAM_GB] budget 42.0 GB", report["output"])
        self.assertEqual(len(report["ticks"]), 2)
        for tick in report["ticks"]:
            self.assertRegex(tick, r"^prefill layer \d+/78 · 512 token · \+0\.10s$")

    def test_a_load_failure_shows_the_engine_reason(self):
        report = chat("load_failure")
        self.assertEqual(report["exit"], "the engine exited while loading")
        self.assertIn("short read at EOF — truncated shard?", report["output"])

    def test_an_engine_that_dies_mid_turn_shows_its_last_words(self):
        report = chat("dies_mid_turn")
        self.assertIn("[engine terminated: exit code 1]", report["output"])
        self.assertIn("short read at EOF — truncated shard?", report["output"])


if __name__ == "__main__":
    unittest.main()
