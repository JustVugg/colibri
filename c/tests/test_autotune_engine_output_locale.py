"""`coli tune` on GLM must read the engine's output whatever the locale.

autotune._run captured the engine's stdout and stderr in text mode, i.e.
decoded in the locale's code page. The GLM engine writes UTF-8: with PROF=1,
which every calibration and replay run sets, its prefill progress line is
"[prefill] layer 1/78 · 12 token · +0.10s" and its profile verdict carries an
em dash. On a code page that cannot decode those bytes (cp949, cp932, or the C
locale with UTF-8 mode off) the reader thread raised UnicodeDecodeError,
proc.stdout came back None, and `coli tune` died in calibration with

    TypeError: unsupported operand type(s) for +: 'NoneType' and 'str'

before measuring anything. Where the code page happens to decode every byte
(cp1252) the output was silently mojibake instead.

The test drives autotune in a child Python whose locale is not UTF-8
(PYTHONUTF8=0 and the C locale, or the Windows ANSI code page) against a fake
engine that speaks the GLM calibration and replay lines.
"""
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

C_DIR = Path(__file__).resolve().parent.parent

CHILD_ENV = {**os.environ, "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0",
             "LC_ALL": "C", "LANG": "C"}
CHILD_ENV.pop("PYTHONIOENCODING", None)

ENGINE = textwrap.dedent("""
    import os, sys
    out, err = sys.stdout.buffer, sys.stderr.buffer
    err.write("[prefill] layer 1/4 \\u00b7 12 token \\u00b7 +0.00s\\n".encode("utf-8"))
    if os.environ.get("PROMPT"):
        out.write(b"[PROMPT_TOKENS] 3: 10 11 12\\n")
        out.write(b"[TOKENS] 4 generated: 20 21 22 23\\n")
    else:
        out.write(b"REPLAY decode: 4 tokens | 10.00 tok/s | expert hit 95.0%\\n")
        out.write(b"[PROF] decode forwards: 4 | latency p50 80.0 ms | p90 90.0 ms | p99 100.0 ms | max 100.0 ms\\n")
    out.write("[PROF] verdict: balanced \\u2014 no phase dominates\\n".encode("utf-8"))
""")

DRIVER = textwrap.dedent("""
    import json, locale, os, sys
    c_dir, engine, profile_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    sys.path.insert(0, c_dir)
    import autotune

    env = dict(os.environ, PROMPT="prompt", PROF="1")
    proc = autotune._run([engine, "8"], env, 60)
    plan = {"version": 2, "cpu": {"physical_cores": 1, "sockets": 1},
            "tiers": {"disk": {"cold_expert_bytes": 0}, "vram": {"devices": []}}}
    error = None
    try:
        profile, _ = autotune.run_tune(engine, 8, dict(os.environ), plan, profile_dir,
                                       "prompt", tokens=4, repeats=1, timeout=60,
                                       min_gain=0.03, profile_dir=profile_dir)
        baseline = profile["validation"]["baseline"]["tok_s"] if profile["validation"] else None
    except Exception as exc:
        error, baseline = f"{type(exc).__name__}: {exc}", None
    print(json.dumps({"locale": locale.getpreferredencoding(False),
                      "stdout": proc.stdout, "stderr": proc.stderr,
                      "error": error, "baseline": baseline}, ensure_ascii=True))
""")


class AutotuneEngineOutputLocaleTest(unittest.TestCase):
    def test_glm_tune_reads_utf8_engine_output_in_a_non_utf8_locale(self):
        with tempfile.TemporaryDirectory() as scratch:
            engine = Path(scratch, "glm-engine.py")
            engine.write_text(ENGINE, encoding="utf-8")
            driver = Path(scratch, "driver.py")
            driver.write_text(DRIVER, encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(driver), str(C_DIR), str(engine), scratch],
                env=CHILD_ENV, capture_output=True, timeout=120)
        if result.returncode:
            raise AssertionError(result.stderr.decode("utf-8", "replace")[-3000:])
        report = json.loads(result.stdout.decode("ascii").strip().splitlines()[-1])
        where = f"child locale {report['locale']}"
        self.assertIsNone(report["error"], where)
        self.assertIsNotNone(report["stdout"], where)
        self.assertIn("[PROF] verdict: balanced — no phase dominates", report["stdout"], where)
        self.assertIn("[prefill] layer 1/4 · 12 token", report["stderr"], where)


if __name__ == "__main__":
    unittest.main()
