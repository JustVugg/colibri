"""TypeSafe's own SDKs against `coli serve`: switching from Jev is a base-URL change.

The official clients, unmodified, pointed at colibri with nothing but
`base_url` and an API key changed:

  typesafe_sdk          Python, TypeSafeClient(api_key=..., base_url=...)
                        .system_one(...) and .models.list(); it validates the
                        reply strictly (pydantic, strict=True), so a missing
                        field, a wrong type or a score level not numbered from
                        zero fails here
  @typesafe-ai/sdk      TypeScript, new TypeSafeClient({apiKey, baseURL})
                        .systemOne(...) and .models.list(), run under node

Each one is driven against three servers, each a real `coli serve` process:
a language model (the tiny MiMo fixture, which scores options through the
logprob channel) and two decision engines (the tiny Laya and GLiNER2.5-Decide
fixtures, DECIDE). The same requests go to all three: every question type, several questions at once,
text, object and array states, objects and arrays as instructions and
criteria, and a request the server refuses (422), which each SDK must raise
as its own UnprocessableEntity error.

Skipped when an SDK, the engines or the fixtures are missing. CI installs the
pinned SDKs (tools/requirements-jev-sdk.txt, tools/jev-sdk/package.json) and
sets JEV_SDK_REQUIRED=1, which turns the skips into failures:

    make -C c jev-sdk-check
"""
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "tests"))
import mimo_serve_fixture  # noqa: E402

REQUIRED = os.environ.get("JEV_SDK_REQUIRED") == "1"
LAYA_FIXTURE = HERE / "laya_tiny"
LAYA_BINARY = HERE / ("laya.exe" if os.name == "nt" else "laya")
GLINER_FIXTURE = HERE / "gliner_decide_tiny"
GLINER_BINARY = HERE / ("gliner_decide.exe" if os.name == "nt" else "gliner_decide")
TS_SDK_DIR = Path(os.environ.get("JEV_TS_SDK_DIR", HERE / "tools" / "jev-sdk"))
TS_SCRIPT = HERE / "tests" / "jev_sdk_client.mjs"
API_KEY = "sk-colibri-test"

STATE = {"subject": "Duplicate charge on invoice #4411",
         "body": "Hi, we were billed twice for March. Please refund the duplicate today or we "
                 "will cancel our plan."}
# Raw dictionaries: what both SDKs put on the wire, and what a Jev client builds.
QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which department should handle this?",
                   "criteria": {"billing": "invoices, payments, refunds",
                                "technical": {"covers": ["bugs", "outages"]},
                                "other": None}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["not urgent", "soon", "blocking"]},
    "churn": {"type": "noul", "instructions": "Does the user threaten to cancel?",
              "criteria": {"true": "they say they will leave", "false": None}},
    "refund": {"type": "noul", "instructions": {"ask": "is a refund requested?"}},
}


def skip_or_fail(message):
    if REQUIRED:
        raise AssertionError(message)
    raise unittest.SkipTest(message)


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class ColiServe:
    """`coli serve` in its own process, on a free port, until stop()."""

    def __init__(self, model, env=None):
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.log = tempfile.TemporaryFile()
        child_env = dict(os.environ, COLI_API_KEY=API_KEY, OMP_NUM_THREADS="2",
                         COLI_NO_OMP_TUNE="1", **(env or {}))
        self.process = subprocess.Popen(
            [sys.executable, str(HERE / "coli"), "serve", "--model", str(model),
             "--host", "127.0.0.1", "--port", str(self.port)],
            env=child_env, stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 120
        while True:
            try:
                with urllib.request.urlopen(self.base + "/health", timeout=2):
                    break
            except OSError:
                if self.process.poll() is not None or time.monotonic() > deadline:
                    self.stop()
                    raise AssertionError(f"coli serve did not start:\n{self.output()}")
                time.sleep(0.2)

    def output(self):
        self.log.seek(0)
        return self.log.read().decode("utf-8", "replace")[-4000:]

    def stop(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.log.close()


_long_mimo = None


def long_mimo_fixture():
    """The served MiMo fixture with room for a Jev request. Its tokenizer reads one
    byte per token and the fixture has 256 positions, less than one request with
    a JSON state; the copy raises max_position_embeddings (RoPE has no limit of
    its own), and is this module's, so no other test sees it."""
    global _long_mimo
    if _long_mimo is None:
        root = Path(tempfile.mkdtemp(prefix="jev-sdk-mimo-"))
        import atexit
        atexit.register(shutil.rmtree, root, True)
        _long_mimo = root / "fixture"
        shutil.copytree(mimo_serve_fixture.served_fixture(), _long_mimo)
        config = json.loads((_long_mimo / "config.json").read_text(encoding="utf-8"))
        config["max_position_embeddings"] = 4096
        (_long_mimo / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return _long_mimo


def servers():
    """(name, model dir, env) of the three servers, or a skip naming what is missing."""
    found = []
    if mimo_serve_fixture.available():
        found.append(("mimo-tiny (language model)", long_mimo_fixture(), {"CTX": "4096"}))
    elif REQUIRED:
        raise AssertionError("mimo or mimo_tiny missing (make mimo; tools/make_mimo_tiny.py)")
    if LAYA_BINARY.exists() and (LAYA_FIXTURE / "model.safetensors").exists():
        found.append(("laya-tiny (decision engine)", LAYA_FIXTURE, {}))
    elif REQUIRED:
        raise AssertionError("laya or laya_tiny missing (make laya laya-tiny-generate)")
    if GLINER_BINARY.exists() and (GLINER_FIXTURE / "model.safetensors").exists():
        found.append(("gliner-decide-tiny (decision engine)", GLINER_FIXTURE, {}))
    elif REQUIRED:
        raise AssertionError("gliner_decide or gliner_decide_tiny missing "
                             "(make gliner_decide gliner-decide-tiny-generate)")
    if not found:
        raise unittest.SkipTest("no tiny fixture to serve (make mimo laya laya-tiny-generate "
                                "gliner_decide gliner-decide-tiny-generate)")
    return found


class _ServedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.served = []
        for name, model, env in servers():
            cls.served.append((name, ColiServe(model, env)))

    @classmethod
    def tearDownClass(cls):
        for _, server in cls.served:
            server.stop()

    def check_answers(self, where, answers):
        """The answer contract, read the way a Jev client reads it."""
        department = answers["department"]
        self.assertIn(department["choice"], {"billing", "technical", "other"}, where)
        self.assertEqual(set(department["probabilities"]), {"billing", "technical", "other"}, where)
        self.assertAlmostEqual(sum(department["probabilities"].values()), 1.0, places=4, msg=where)
        self.assertEqual(department["choice"],
                         max(department["probabilities"], key=department["probabilities"].get), where)
        self.assertTrue(0.0 <= department["confidence"] <= 1.0, where)
        urgency = answers["urgency"]
        self.assertEqual({int(k) for k in urgency["probabilities"]}, {0, 1, 2}, where)
        self.assertEqual({int(k) for k in urgency["legend"]}, {0, 1, 2}, where)
        self.assertAlmostEqual(sum(urgency["probabilities"].values()), 1.0, places=4, msg=where)
        expected = sum(int(k) * p for k, p in urgency["probabilities"].items())
        self.assertAlmostEqual(urgency["score"], expected, places=4, msg=where)
        for key in ("churn", "refund"):
            self.assertTrue(0.0 <= answers[key]["noul"] <= 1.0, where)


class PythonSdk(_ServedTest):
    @classmethod
    def setUpClass(cls):
        try:
            import typesafe_sdk  # noqa: F401
        except ImportError:
            skip_or_fail("typesafe_sdk is not installed (pip install -r tools/requirements-jev-sdk.txt)")
        super().setUpClass()

    def client(self, server):
        from typesafe_sdk import RetryPolicy, TypeSafeClient
        # The documented switch: base_url and the key. The longer timeout and no
        # retries are this test's, so a slow CI box fails once and visibly.
        return TypeSafeClient(api_key=API_KEY, base_url=server.base, timeout=120.0,
                              retry=RetryPolicy(max_retries=0))

    def test_system_one_with_dictionaries_and_question_objects(self):
        from typesafe_sdk import Choice, Noul, Score, SystemOneResponse
        for name, server in self.served:
            with self.subTest(server=name), self.client(server) as client:
                result = client.system_one(state=STATE, questions=QUESTIONS)
                self.assertIsInstance(result, SystemOneResponse)
                answers = {key: answer.model_dump() for key, answer in result.answers.items()}
                self.check_answers(name, answers)
                self.assertIn(result.choices["department"].choice, {"billing", "technical", "other"})
                self.assertEqual(sorted(result.scores["urgency"].legend), [0, 1, 2])
                self.assertGreater(result.usage.input_tokens, 0)
                self.assertTrue(result.request_id.startswith("req_"))
                self.assertEqual(result.model, client.models.list().models[0].name)
                # the SDK's own question objects, a text state and an array state
                typed = {"tone": Choice(instructions="What is the tone?",
                                        criteria={"calm": None, "angry": None}),
                         "angry": Noul(instructions="Is the sender angry?"),
                         "level": Score(instructions="How strong?", criteria=["weak", "strong"])}
                for state in ("I was charged twice. Please help.",
                              [{"role": "user", "content": "charged twice"},
                               {"role": "assistant", "content": "sorry about that"}]):
                    out = client.system_one(state=state, questions=typed)
                    self.assertIn(out.choices["tone"].choice, {"calm", "angry"})
                    self.assertTrue(0.0 <= out.nouls["angry"].noul <= 1.0)
                    self.assertEqual(sorted(out.scores["level"].probabilities), [0, 1])

    def test_a_refusal_is_the_sdks_unprocessable_entity(self):
        from typesafe_sdk import TypeSafeUnprocessableEntityError
        for name, server in self.served:
            with self.subTest(server=name), self.client(server) as client:
                with self.assertRaises(TypeSafeUnprocessableEntityError) as caught:
                    client.system_one(state="x", questions={"q": {"type": "guess"}})
                self.assertIn("type", str(caught.exception))
                self.assertTrue(caught.exception.request_id.startswith("req_"))


class TypeScriptSdk(_ServedTest):
    @classmethod
    def setUpClass(cls):
        if shutil.which("node") is None:
            skip_or_fail("node is not installed")
        if not (TS_SDK_DIR / "node_modules" / "@typesafe-ai" / "sdk").is_dir():
            skip_or_fail(f"@typesafe-ai/sdk is not installed (cd {TS_SDK_DIR} && npm ci)")
        super().setUpClass()

    def run_client(self, server, request):
        env = dict(os.environ, TYPESAFE_API_KEY=API_KEY, TYPESAFE_BASE_URL=server.base,
                   JEV_TS_SDK_DIR=str(TS_SDK_DIR))
        result = subprocess.run(["node", str(TS_SCRIPT), json.dumps(request)], env=env,
                                capture_output=True, text=True, timeout=300,
                                cwd=str(TS_SDK_DIR))
        self.assertEqual(result.returncode, 0, result.stderr[-3000:])
        return json.loads(result.stdout)

    def test_system_one_and_models(self):
        for name, server in self.served:
            with self.subTest(server=name):
                out = self.run_client(server, {"state": STATE, "questions": QUESTIONS})
                self.check_answers(name, out["result"]["answers"])
                self.assertTrue(out["requestId"].startswith("req_"))
                self.assertEqual(out["models"][0]["name"], out["result"]["model"])
                helpers = self.run_client(server, {"helpers": True})
                self.assertIn(helpers["result"]["answers"]["tone"]["choice"], {"calm", "angry"})

    def test_a_refusal_is_the_sdks_unprocessable_entity(self):
        for name, server in self.served:
            with self.subTest(server=name):
                out = self.run_client(server, {"state": "x", "questions": {"q": {"type": "guess"}}})
                self.assertEqual(out["error"]["name"], "UnprocessableEntityError")
                self.assertEqual(out["error"]["status"], 422)


if __name__ == "__main__":
    unittest.main()
