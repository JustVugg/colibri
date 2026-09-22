"""`tools/datapoint.py --mode fresh-process` must speak UTF-8 to a legacy engine whatever the locale.

run_fresh_engine drives olmoe, inkling and colibri through their stdin chat
loop and captured the pipes in text mode, i.e. in the locale's code page. The
engines read and write UTF-8, so on a code page that is not UTF-8 (cp949,
cp932, or the C locale with UTF-8 mode off):

  * a prompt with a character the code page cannot encode died with
    UnicodeEncodeError before the engine started measuring, and one it can
    encode (é on cp1252) reached the engine as non-UTF-8 bytes;
  * an engine line the code page cannot decode (colibri's "[prefill] layer
    1/78 · 12 token" progress line) made the reader thread raise, proc.stdout
    came back None, and the datapoint died with a TypeError.

The test runs run_fresh_engine in a child Python whose locale is not UTF-8
against a fake legacy engine that records the bytes it reads.
"""
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent / "tools"
PROMPT = "café — 世界 \U0001F600"

CHILD_ENV = {**os.environ, "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0",
             "LC_ALL": "C", "LANG": "C"}
CHILD_ENV.pop("PYTHONIOENCODING", None)

ENGINE = textwrap.dedent("""
    import os, sys
    data = sys.stdin.buffer.read()
    with open(os.environ["RECORD"], "wb") as record:
        record.write(data)
    sys.stderr.buffer.write("[prefill] layer 1/4 \\u00b7 12 token \\u00b7 +0.00s\\n".encode("utf-8"))
    sys.stderr.buffer.write(b"resident weights loaded in 1.5s | RSS after load: 19.0 GB\\n")
    sys.stdout.buffer.write("answer \\u2014 done\\n".encode("utf-8"))
""")

DRIVER = textwrap.dedent("""
    import json, locale, subprocess, sys
    from unittest import mock
    tools, engine, prompt = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
    sys.path.insert(0, tools)
    import datapoint

    real_run = subprocess.run
    def run(command, **kwargs):
        return real_run([sys.executable, engine, *command[1:]], **kwargs)

    error = rows = None
    with mock.patch.object(datapoint.subprocess, "run", run):
        try:
            rows = datapoint.run_fresh_engine("/engines/olmoe", "/model", prompt,
                                              max_new=7, runs=1, cap=3, bits=8)
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
    print(json.dumps({"locale": locale.getpreferredencoding(False), "error": error,
                      "rows": rows}, ensure_ascii=True))
""")


class DatapointFreshLocaleTest(unittest.TestCase):
    def test_legacy_engine_gets_utf8_prompt_and_its_output_is_read(self):
        with tempfile.TemporaryDirectory() as scratch:
            engine = Path(scratch, "engine.py")
            engine.write_text(ENGINE, encoding="utf-8")
            driver = Path(scratch, "driver.py")
            driver.write_text(DRIVER, encoding="utf-8")
            record = Path(scratch, "stdin.bin")
            result = subprocess.run(
                [sys.executable, str(driver), str(TOOLS), str(engine), json.dumps(PROMPT)],
                env={**CHILD_ENV, "RECORD": str(record)},
                capture_output=True, timeout=120)
            if result.returncode:
                raise AssertionError(result.stderr.decode("utf-8", "replace")[-3000:])
            report = json.loads(result.stdout.decode("ascii").strip().splitlines()[-1])
            where = f"child locale {report['locale']}"
            self.assertIsNone(report["error"], where)
            # A text-mode pipe writes the platform's line ending ("\r\n" on
            # Windows, as before this fix); the engine strips it. The prompt
            # bytes themselves must be UTF-8.
            self.assertEqual(record.read_bytes().rstrip(b"\r\n"), PROMPT.encode("utf-8"), where)
            self.assertEqual(report["rows"][0]["rss"], 19.0, where)
            self.assertEqual(report["rows"][0]["load_s"], 1.5, where)


if __name__ == "__main__":
    unittest.main()
