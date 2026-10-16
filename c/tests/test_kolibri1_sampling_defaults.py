#!/usr/bin/env python3
"""Kolibri 1 sampling defaults at the gateway.

Precedence, per parameter (kolibri1 only; every other family keeps 0.7 / 0.9):
  temperature  client value > COLI_TEMP (coli --temp) > checkpoint generation_config.json > vendor 1.0
  top_p        client value > checkpoint generation_config.json > vendor 0.97
  top_k        operator K1_TOP_K > checkpoint generation_config.json > vendor 128   (engine env, at launch)
               request top_k (any value, 1 included) is IGNORED and not forwarded; one stderr warning line per request that sends it
  seed         engine env K1_SEED / SEED only (not on the wire); a request `seed` is ignored

Unit part: generation_options() with ARCH patched. Engine part (real kolibri1 binary on the tiny
fixture, K1_SERVE_LOG=1): the engine's own per-request log line shows the temperature / top_p it
received in the SUBMIT frame and the top_k it launched with. Needs numpy and ./kolibri1 for that part.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import openai_server as gw  # noqa: E402

try:
    import numpy  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

BINARY = Path(os.environ.get("KOLIBRI1_BIN", ROOT / "kolibri1")).resolve()
MSG = [{"role": "user", "content": "hi"}]


def opts(arch, body, model_dir=None, env=None):
    body = dict({"messages": MSG}, **body)
    base = {k: v for k, v in os.environ.items() if k != "COLI_TEMP"}
    base.update(env or {})
    with patch("openai_server.ARCH", arch), patch("openai_server.KOLIBRI1_MODEL_DIR", model_dir), \
            patch.dict(os.environ, base, clear=True):
        _m, t, p, _g, _s = gw.generation_options(body, 1000)
    return t, p


class Unit(unittest.TestCase):
    def cfg(self, **kw):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        if kw is not None:
            Path(d.name, "generation_config.json").write_text(json.dumps(kw))
        return d.name

    def test_omitted_uses_checkpoint_defaults(self):
        d = self.cfg(temperature=0.6, top_p=0.9, top_k=64)
        self.assertEqual(opts("kolibri1", {}, d), (0.6, 0.9))

    def test_vendor_values_from_file(self):
        d = self.cfg(temperature=1.0, top_p=0.97, top_k=128)
        self.assertEqual(opts("kolibri1", {}, d), (1.0, 0.97))

    def test_explicit_values_win_including_zero(self):
        d = self.cfg(temperature=1.0, top_p=0.97)
        self.assertEqual(opts("kolibri1", {"temperature": 0.2, "top_p": 0.5}, d), (0.2, 0.5))
        self.assertEqual(opts("kolibri1", {"temperature": 0}, d), (0.0, 0.97))
        self.assertEqual(opts("kolibri1", {"top_p": 1}, d), (1.0, 1.0))

    def test_each_parameter_independent(self):
        d = self.cfg(temperature=1.0, top_p=0.97)
        self.assertEqual(opts("kolibri1", {"temperature": 0.3}, d), (0.3, 0.97))
        self.assertEqual(opts("kolibri1", {"top_p": 0.4}, d), (1.0, 0.4))

    def test_coli_temp_beats_checkpoint_but_not_client(self):
        d = self.cfg(temperature=1.0, top_p=0.97)
        self.assertEqual(opts("kolibri1", {}, d, {"COLI_TEMP": "0.3"}), (0.3, 0.97))
        self.assertEqual(opts("kolibri1", {"temperature": 0.9}, d, {"COLI_TEMP": "0.3"}), (0.9, 0.97))

    def test_missing_or_bad_file_falls_back_to_vendor(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.assertEqual(opts("kolibri1", {}, d.name), (1.0, 0.97))
        bad = self.cfg(temperature=-5, top_p=7)
        self.assertEqual(opts("kolibri1", {}, bad), (1.0, 0.97))
        Path(bad, "generation_config.json").write_text("{not json")
        gw._K1_SAMPLING_CACHE.clear()
        self.assertEqual(opts("kolibri1", {}, bad), (1.0, 0.97))

    def test_top_k_defaults(self):
        self.assertEqual(gw.kolibri1_sampling_defaults(self.cfg(top_k=64))["top_k"], 64)
        self.assertEqual(gw.kolibri1_sampling_defaults(self.cfg())["top_k"], 128)

    def test_request_top_k_and_seed_are_ignored_with_one_warning(self):
        import io
        from contextlib import redirect_stderr
        d = self.cfg(temperature=1.0, top_p=0.97)
        for body in ({"top_k": 1}, {"top_k": 1, "temperature": 0}, {"top_k": 1, "top_p": 0.5}):
            want = (body.get("temperature", 1.0), body.get("top_p", 0.97))
            err = io.StringIO()
            with redirect_stderr(err):
                self.assertEqual(opts("kolibri1", body, d), (float(want[0]), want[1]))   # top_p never negative, nothing coded
            self.assertEqual(err.getvalue().count("\n"), 1, err.getvalue())
            self.assertIn("top_k ignored", err.getvalue())
        err = io.StringIO()
        with redirect_stderr(err):
            opts("kolibri1", {"top_k": 5, "seed": 3}, d)
        self.assertEqual(err.getvalue().count("\n"), 1)
        self.assertIn("top_k and seed ignored", err.getvalue())
        err = io.StringIO()
        with redirect_stderr(err):
            opts("kolibri1", {}, d)
        self.assertEqual(err.getvalue(), "")          # nothing sent, nothing logged
        err = io.StringIO()
        with patch("openai_server.ARCH", "glm"), redirect_stderr(err):
            opts("glm", {"top_k": 1}, d)
        self.assertEqual(err.getvalue(), "")          # other families are not touched by the kolibri1 warning
        self.assertEqual(opts("kolibri1", {"top_k": 40}, d), (1.0, 0.97))
        self.assertEqual(opts("kolibri1", {"top_k": True}, d), (1.0, 0.97))
        self.assertEqual(opts("kolibri1", {"seed": 5}, d), (1.0, 0.97))

    def test_other_families_unchanged(self):
        d = self.cfg(temperature=1.0, top_p=0.97)
        for arch in ("glm", "mimo", "deepseek_v41", "qwen36", "qwen38"):
            self.assertEqual(opts(arch, {}, d), (0.7, 0.9), arch)
            self.assertEqual(opts(arch, {}, d, {"COLI_TEMP": "0.3"}), (0.3, 0.9), arch)
            self.assertEqual(opts(arch, {"temperature": 1.5, "top_p": 0.5}, d), (1.5, 0.5), arch)
            self.assertEqual(opts(arch, {"top_k": 1}, d), (0.7, 0.9), arch)


@unittest.skipUnless(HAVE_NUMPY and BINARY.exists(), "needs numpy and the kolibri1 binary")
class RealEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import kolibri1_serve_fixture as FX
        import test_kolibri1_gateway_real_engine as R
        cls.R = R
        cls.tmp = tempfile.TemporaryDirectory(prefix="k1-p7d-")
        root = Path(cls.tmp.name)
        cls.vendor = FX.make_serve_fixture(root / "vendor")
        cls.custom = FX.make_serve_fixture(root / "custom", gen_config={
            "temperature": 0.6, "top_p": 0.8, "top_k": 17, "do_sample": True})
        cls.nofile = FX.make_serve_fixture(root / "nofile", gen_config=False)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_case(self, model, bodies, extra_env=None):
        R = self.R
        t = R.Kolibri1GatewayRealEngine("start")
        t.model, t.root = model, Path(self.tmp.name)
        base, err = t.start(extra_env=extra_env)
        try:
            for b in bodies:
                R.Kolibri1GatewayRealEngine.post(base, dict({"messages": R.USER, "max_tokens": 2}, **b))
            rows = t.engine_log(err, len(bodies))
            text = err.read_text(errors="replace")
        finally:
            t.doCleanups()
        return rows, text

    @staticmethod
    def tp(row):
        """(temperature, top_p) as the engine logged them: float32 on the wire, so compare at 6 places"""
        return (round(float(row["temperature"]), 6), round(float(row["top_p"]), 6))

    @staticmethod
    def launch_top_k(text):
        return int(text.split("top_k ", 1)[1].split(",", 1)[0])

    def test_omitted_reaches_engine_as_vendor_defaults(self):
        rows, text = self.run_case(self.vendor, [{}])
        self.assertEqual(self.tp(rows[0]), (1.0, 0.97))
        self.assertEqual(rows[0]["top_k"], "128")
        self.assertEqual(self.launch_top_k(text), 128)

    def test_checkpoint_values_followed_not_hardcoded(self):
        rows, text = self.run_case(self.custom, [{}])
        self.assertEqual(self.tp(rows[0]), (0.6, 0.8))
        self.assertEqual(rows[0]["top_k"], "17")

    def test_explicit_values_win_and_each_request_independent(self):
        rows, _ = self.run_case(self.vendor, [{"temperature": 0.25, "top_p": 0.5}, {"temperature": 0},
                                              {"top_p": 0.6}, {}, {"top_k": 1}])
        got = [self.tp(r) for r in rows]
        self.assertEqual(got, [(0.25, 0.5), (0.0, 0.97), (1.0, 0.6), (1.0, 0.97), (1.0, 0.97)])

    def test_operator_top_k_and_coli_temp_precedence(self):
        rows, text = self.run_case(self.vendor, [{}, {"temperature": 0.4}],
                                   {"K1_TOP_K": "5", "COLI_TEMP": "0.3"})
        self.assertEqual(self.launch_top_k(text), 5)
        self.assertEqual([self.tp(r) for r in rows],
                         [(0.3, 0.97), (0.4, 0.97)])

    def test_no_generation_config_uses_vendor_constants(self):
        rows, text = self.run_case(self.nofile, [{}])
        self.assertEqual(self.tp(rows[0]), (1.0, 0.97))
        self.assertEqual(self.launch_top_k(text), 128)


if __name__ == "__main__":
    unittest.main()
