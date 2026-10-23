"""`coli run` on OLMoE must hand the engine the prompt as UTF-8 whatever the locale.

cmd_run feeds the one-shot prompt to olmoe's stdin chat loop, which reads raw
bytes and tokenizes them as UTF-8. The pipe was opened in text mode, so Python
encoded the prompt in the locale's code page instead:

  * where the code page cannot encode a character of the prompt (an accented
    letter or an emoji on cp949/cp932, anything non-ASCII in the C locale),
    `coli run` died with a UnicodeEncodeError traceback before the engine saw
    a byte;
  * where it can (é on cp1252, Hangul on cp949), the engine received the code
    page's bytes, which are not UTF-8, and answered a garbled prompt.

The test runs the real cmd_run in a child Python whose locale is not UTF-8
(PYTHONUTF8=0 and the C locale, or the Windows ANSI code page) against a fake
engine that records the bytes it reads on stdin.
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
PROMPT = "café — 世界 \U0001F600"

CHILD_ENV = {**os.environ, "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0",
             "LC_ALL": "C", "LANG": "C"}
CHILD_ENV.pop("PYTHONIOENCODING", None)

ENGINE = textwrap.dedent("""
    import os, sys
    data = sys.stdin.buffer.read()
    with open(os.environ["RECORD"], "wb") as record:
        record.write(data)
""")

DRIVER = textwrap.dedent("""
    import argparse, importlib.util, json, locale, subprocess, sys
    from importlib.machinery import SourceFileLoader
    from pathlib import Path
    from unittest import mock

    cli, engine, model, prompt = sys.argv[1], sys.argv[2], sys.argv[3], json.loads(sys.argv[4])
    loader = SourceFileLoader("coli_run_olmoe_prompt", cli)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    coli = importlib.util.module_from_spec(spec)
    loader.exec_module(coli)

    real_run = subprocess.run
    def run(command, **kwargs):
        return real_run([sys.executable, engine, *command[1:]], **kwargs)

    args = argparse.Namespace(model=model, prompt=[prompt], ngen=8, ram=0,
                              temp=0.0, ctx=0, cap=None)
    code = None
    with mock.patch.object(coli, "need_model"), \\
         mock.patch.object(coli, "banner"), \\
         mock.patch.object(coli, "engine_for", return_value="olmoe"), \\
         mock.patch.object(coli.subprocess, "run", run):
        try:
            coli.cmd_run(args)
        except SystemExit as exit_:
            code = exit_.code
    print(json.dumps({"exit": code, "locale": locale.getpreferredencoding(False)}))
""")


class ColiRunOlmoePromptTest(unittest.TestCase):
    def test_the_engine_reads_the_prompt_as_utf8(self):
        with tempfile.TemporaryDirectory() as scratch:
            model = Path(scratch, "model")
            model.mkdir()
            (model / "config.json").write_text(json.dumps({"model_type": "olmoe"}),
                                               encoding="utf-8")
            (model / "tokenizer.json").write_text("{}", encoding="utf-8")
            engine = Path(scratch, "engine.py")
            engine.write_text(ENGINE, encoding="utf-8")
            driver = Path(scratch, "driver.py")
            driver.write_text(DRIVER, encoding="utf-8")
            record = Path(scratch, "stdin.bin")
            # The prompt reaches the child as ASCII-escaped JSON, so the child's
            # own argv decoding (also locale-bound) is not what is under test.
            result = subprocess.run(
                [sys.executable, str(driver), str(CLI), str(engine), str(model),
                 json.dumps(PROMPT)],
                env={**CHILD_ENV, "RECORD": str(record)},
                capture_output=True, timeout=60)
            if result.returncode:
                raise AssertionError(result.stderr.decode("utf-8", "replace")[-3000:])
            report = json.loads(result.stdout.decode("ascii", "replace").strip().splitlines()[-1])
            self.assertEqual(report["exit"], 0, report)
            self.assertEqual(record.read_bytes(), (PROMPT + "\n").encode("utf-8"),
                             f"child locale {report['locale']}")


if __name__ == "__main__":
    unittest.main()
