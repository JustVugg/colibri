"""Decision engines behind POST /v1/systemone: the DECIDE command, end to end.

A decision engine (Laya, GLiNER2.5-Decide) does not generate. It announces
`CAPS decide=1 chat=0`, and /v1/systemone hands it the request as one DECIDE
record instead of scoring options through the logprob channel. Three layers are
pinned here:

  DecideProtocol      the wire, against a fake engine process: the DECIDE frame
                      byte for byte, the DECISION + DONE reply, a refusal
                      (DECIDE_INVALID) as the caller's 422, the CAPS flags.
  DecisionGateway     the HTTP route against a fake decision engine: the reply
                      has exactly the shape the LLM path returns, the record
                      carries the state and the options as the reference reads
                      them, the generating endpoints answer 400 with a pointer,
                      validation stays the route's own.
  LayaTinyEndToEnd    the real laya engine on the tiny fixture behind the real
                      gateway: the probabilities that come back over HTTP are
                      the `laya` package's (laya_tiny/ref.json), and the
                      latency is split into gateway and engine time.
  GlinerDecideTinyEndToEnd
                      the same for the gliner_decide engine on its tiny
                      fixture, against the `gliner2` package's answers
                      (gliner_decide_tiny/ref.json).

Each end-to-end class is skipped without its fixture and binary (`make
laya-tiny-check` / `make gliner-decide-tiny-check` build them and turn the skip
into a failure).
"""
import json
import math
import os
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tests"))
import openai_server  # noqa: E402
from family_registry import family_by_id  # noqa: E402
from openai_server import APIError, APIServer, Engine, READY  # noqa: E402
from test_openai_server import BlockingStream, FakeProcess  # noqa: E402

FIXTURE = HERE / "laya_tiny"
BINARY = HERE / ("laya.exe" if os.name == "nt" else "laya")
REQUIRED = os.environ.get("LAYA_TINY_REQUIRED") == "1"
GLINER_FIXTURE = HERE / "gliner_decide_tiny"
GLINER_BINARY = HERE / ("gliner_decide.exe" if os.name == "nt" else "gliner_decide")
GLINER_REQUIRED = os.environ.get("GLINER_DECIDE_TINY_REQUIRED") == "1"

TICKET = {"from": "user@acme.com", "subject": "Duplicate charge on invoice #4411",
          "body": "Hi, we were billed twice for March. Please refund the duplicate today or we "
                  "will cancel our plan."}
QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which department should handle this request?",
                   "criteria": {"billing": "invoices, payments, refunds",
                                "technical": "bugs, outages, system errors",
                                "sales": "pricing, new contracts", "other": "everything else"}},
    "urgency": {"type": "score", "instructions": "How urgent is this request?",
                "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]},
    "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
    "refund_requested": {"type": "noul", "instructions": "Does the user explicitly request a refund?"},
}


def decision_process(on_decide):
    """A FakeProcess whose handshake announces a decision engine."""
    process = FakeProcess(on_decide)
    process.stdout = BlockingStream(READY + b"CAPS decide=1 chat=0 max_len=512\n"
                                    b"STAT 0 0.0 0.0 1.00 0 0\n")
    return process


class DecideProtocol(unittest.TestCase):
    def start(self, respond):
        process = decision_process(respond)
        with patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("laya", "model", family=family_by_id("laya"))
        self.addCleanup(engine.close)
        return engine, process

    def test_caps_say_decide_and_no_chat(self):
        engine, _ = self.start(lambda process, frame: None)
        self.assertTrue(engine.decides)
        self.assertFalse(engine.chats)
        self.assertTrue(openai_server.engine_decides(engine))
        self.assertFalse(openai_server.engine_chats(engine))

    def test_round_trip_is_byte_exact(self):
        record = {"state": "sé", "state_type": "string", "questions": [
            {"id": "q", "type": "noul", "instructions": "ok?",
             "options": [{"label": "false", "text": None}, {"label": "true", "text": None}]}]}
        payload = json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode()
        expected = f"DECIDE 1 0 {len(payload)}\n".encode() + payload + b"\n"
        reply = json.dumps({"answers": [{"id": "q", "logits": [0.0, 1.0], "probs": [0.25, 0.75],
                                         "temperature": 2.0, "tokens": 9}],
                            "input_tokens": 9, "engine_ms": 4.5}).encode()

        def respond(process, frame):
            self.assertEqual(frame, expected)
            process.stdout.feed(f"DECISION 1 {len(reply)}\n".encode() + reply +
                                b"\nDONE 1 STAT 0 2000.000 0.0 1.00 9 0\n")

        engine, process = self.start(respond)
        decision, stats = engine.decide(record)
        self.assertEqual(process.writes, [expected])
        self.assertEqual(decision["answers"][0]["probs"], [0.25, 0.75])
        self.assertEqual(stats["prompt_tokens"], 9)
        self.assertEqual(stats["completion_tokens"], 0)

    def test_a_refused_record_is_the_callers_422(self):
        def respond(process, frame):
            process.stdout.feed(b"ERROR 1 DECIDE_INVALID questions.bucket: only 38 of its 40 "
                                b"option markers fit in max_len=160\n")

        engine, _ = self.start(respond)
        with self.assertRaises(APIError) as caught:
            engine.decide({"state": "x", "state_type": "string", "questions": []})
        self.assertEqual(caught.exception.status, 422)
        self.assertEqual(caught.exception.param, "questions.bucket")
        self.assertIn("only 38 of its 40", caught.exception.message)

    def test_an_engine_failure_is_not_the_callers(self):
        def respond(process, frame):
            process.stdout.feed(b"ERROR 1 DECIDE_FAILED out of memory\n")

        engine, _ = self.start(respond)
        with self.assertRaises(RuntimeError):
            engine.decide({"state": "x", "state_type": "string", "questions": []})

    def test_a_nul_never_reaches_the_engine(self):
        engine, process = self.start(lambda process, frame: self.fail("written"))
        with self.assertRaises(APIError) as caught:
            engine.decide({"state": "a\0b", "state_type": "string", "questions": []})
        self.assertEqual(caught.exception.status, 422)
        self.assertEqual(process.writes, [])


class FakeDecisionEngine:
    """A decision engine that answers from a table: the probabilities of each
    question by id, in the record's option order."""

    decides = True
    chats = False
    kv_slots = 1

    def __init__(self, table, refuse=None):
        self.table = table
        self.refuse = refuse
        self.records = []

    def decide(self, record, cache_slot=0, cancelled=None):
        self.records.append(record)
        if self.refuse:
            raise APIError(422, self.refuse, "questions.q", "invalid_question")
        answers = [{"id": q["id"], "logits": [math.log(p) for p in self.table[q["id"]]],
                    "probs": self.table[q["id"]], "temperature": 1.0, "tokens": 10}
                   for q in record["questions"]]
        return {"answers": answers, "input_tokens": 10 * len(answers), "engine_ms": 12.5}, {}

    def generate(self, *args, **kwargs):
        raise AssertionError("a decision engine is never asked to generate")

    def close(self):
        pass


class DecisionGateway(unittest.TestCase):
    def serve(self, engine):
        self.engine = engine
        self.server = APIServer(("127.0.0.1", 0), engine, "laya")
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def post(self, path, body):
        request = Request(self.base + path, data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=10) as response:
            return json.loads(response.read()), dict(response.headers)

    def post_error(self, path, body):
        try:
            self.post(path, body)
        except HTTPError as error:
            return error.code, json.loads(error.read())
        self.fail("expected an error")

    def table(self):
        return {"department": [0.7, 0.1, 0.1, 0.1], "urgency": [0.2, 0.3, 0.5],
                "churn_risk": [0.25, 0.75], "refund_requested": [0.9, 0.1]}

    def test_reply_has_the_shape_the_llm_path_returns(self):
        self.serve(FakeDecisionEngine(self.table()))
        out, headers = self.post("/v1/systemone", {"model": "jev-latest", "state": TICKET,
                                                   "questions": QUESTIONS})
        self.assertEqual(out["model"], "laya")
        self.assertEqual(list(out["answers"]), list(QUESTIONS))
        self.assertEqual(out["answers"]["churn_risk"], {"type": "noul", "noul": 0.75})
        self.assertEqual(out["answers"]["refund_requested"], {"type": "noul", "noul": 0.1})
        department = out["answers"]["department"]
        self.assertEqual(set(department), {"type", "choice", "probabilities", "confidence"})
        self.assertEqual(department["choice"], "billing")
        self.assertEqual(list(department["probabilities"]), ["billing", "technical", "sales", "other"])
        self.assertAlmostEqual(department["confidence"], (4 * 0.7 - 1) / 3, places=6)
        urgency = out["answers"]["urgency"]
        self.assertEqual(set(urgency), {"type", "score", "legend", "probabilities", "confidence"})
        self.assertEqual(urgency["legend"], {"0": "not urgent", "1": "soon",
                                             "2": "critical deadline or blocking issue"})
        self.assertEqual(list(urgency["probabilities"]), ["0", "1", "2"])
        self.assertAlmostEqual(urgency["score"], 0 * 0.2 + 1 * 0.3 + 2 * 0.5, places=6)
        self.assertEqual(out["usage"], {"input_tokens": 40, "output_tokens": 0, "cost": 0})
        self.assertEqual(out["provider"], "colibri")
        self.assertEqual(headers["x-colibri-engine-ms"], "12.5")

    def test_the_record_carries_what_the_reference_reads(self):
        self.serve(FakeDecisionEngine({"q": [0.5, 0.5], "c": [0.6, 0.4], "s": [0.5, 0.5],
                                       "d": [0.5, 0.5]}))
        self.post("/v1/systemone", {"model": "jev-latest",
                                    "state": {"ticket": 4711, "text": "café", "ok": True},
                                    "questions": {
            "q": {"type": "noul", "instructions": {"ask": "money?"},
                  "criteria": {"true": "about a payment"}},
            "c": {"type": "choice", "instructions": "  Which?  ",
                  "criteria": {"a": None, "b": {"desc": "structured"}}},
            "s": {"type": "score", "criteria": ["low", {"n": 2}]},
            "d": {"type": "noul"}}})
        record = self.engine.records[0]
        # a JSON state as json.dumps(ensure_ascii=False) writes it, and its type
        self.assertEqual(record["state"], '{"ticket": 4711, "text": "café", "ok": true}')
        self.assertEqual(record["state_type"], "object")
        q, c, s, d = record["questions"]
        self.assertEqual(q["instructions"], '{"ask": "money?"}')
        self.assertEqual(q["options"], [{"label": "false", "text": None},
                                        {"label": "true", "text": "about a payment"}])
        self.assertEqual(c["instructions"], "  Which?  ")                  # exactly as sent
        self.assertEqual(c["options"], [{"label": "a", "text": None},
                                        {"label": "b", "text": '{"desc": "structured"}'}])
        self.assertEqual(s["options"], [{"label": "0", "text": "low"},
                                        {"label": "1", "text": '{"n": 2}'}])
        self.assertEqual(s["instructions"], "Rate this on the scale below.")
        self.assertEqual(d["instructions"], "Is this true?")

    def test_a_list_state_is_a_conversation(self):
        self.serve(FakeDecisionEngine({"q": [0.5, 0.5]}))
        turns = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        self.post("/v1/systemone", {"state": turns, "questions": {"q": {"type": "noul",
                                                                        "instructions": "?"}}})
        self.assertEqual(self.engine.records[0]["state_type"], "array")
        self.assertEqual(json.loads(self.engine.records[0]["state"]), turns)

    def test_generating_endpoints_point_to_systemone(self):
        self.serve(FakeDecisionEngine({}))
        for path, body in (("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}),
                           ("/v1/completions", {"prompt": "hi"}),
                           ("/v1/messages", {"max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]})):
            with self.subTest(path=path):
                status, error = self.post_error(path, dict(body, model="laya"))
                self.assertEqual(status, 400)
                if path != "/v1/messages":              # Anthropic's envelope has no code
                    self.assertEqual(error["error"]["code"], "unsupported_endpoint")
                self.assertIn("POST /v1/systemone", error["error"]["message"])

    def test_v1_brio_is_gone_here_too(self):
        self.serve(FakeDecisionEngine({}))
        status, error = self.post_error("/v1/brio", {"model": "laya", "state": "x", "options": ["a", "b"]})
        self.assertEqual((status, error["error"]["code"]), (404, "not_found"))

    def test_models_card_and_health_say_decision(self):
        self.serve(FakeDecisionEngine({}))
        with urlopen(self.base + "/v1/models", timeout=10) as response:
            card = json.loads(response.read())["data"][0]
        self.assertEqual(card["capabilities"], ["systemone", "decision"])
        with urlopen(self.base + "/health", timeout=10) as response:
            self.assertEqual(json.loads(response.read())["capabilities"], ["systemone", "decision"])

    def test_colibris_options_on_a_decision_engine(self):
        """normalize and pin_state act on a language model's logprobs and photos;
        a decision engine has neither and ignores them. A prefix is refused: it
        would be text the model never reads. cache_slot reaches the engine."""
        engine = FakeDecisionEngine({"q": [0.5, 0.5]})
        engine.kv_slots = 2
        engine.slots = []
        decide = engine.decide
        engine.decide = lambda record, cache_slot=0, cancelled=None: (
            engine.slots.append(cache_slot), decide(record, cache_slot, cancelled))[1]
        self.engine = engine
        self.server = APIServer(("127.0.0.1", 0), engine, "laya", kv_slots=2)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        question = {"q": {"type": "noul", "instructions": "?"}}
        out, _ = self.post("/v1/systemone", {"state": "x", "questions": question,
                                             "normalize": "mean", "pin_state": True, "cache_slot": 1})
        self.assertEqual(out["answers"]["q"]["noul"], 0.5)
        self.assertEqual(engine.slots, [1])
        status, error = self.post_error("/v1/systemone", {"state": "x", "questions": question,
                                                          "prefix": "rules"})
        self.assertEqual((status, error["error"]["param"]), (422, "prefix"))

    def test_validation_is_the_routes_own(self):
        self.serve(FakeDecisionEngine({}))
        for body, param in (({"questions": {"q": {"type": "noul"}}}, "state"),
                            ({"state": "x", "questions": {}}, "questions"),
                            ({"state": "x", "questions": {"q": {"type": "maybe"}}}, "questions.q.type"),
                            ({"state": "x", "questions": {"q": {"type": "score", "criteria": []}}},
                             "questions.q.criteria")):
            with self.subTest(param=param):
                status, error = self.post_error("/v1/systemone", body)
                self.assertEqual(status, 422)
                self.assertEqual(error["error"]["param"], param)
        self.assertEqual(self.engine.records, [])

    def test_an_engine_refusal_reaches_the_client_as_422(self):
        self.serve(FakeDecisionEngine({}, refuse="questions.q: only 3 of its 30 option markers fit"))
        status, error = self.post_error("/v1/systemone", {"state": "x", "questions": {
            "q": {"type": "choice", "criteria": {"a": "x", "b": "y"}}}})
        self.assertEqual(status, 422)
        self.assertIn("only 3 of its 30", error["error"]["message"])


class DecisionRegistryAndLauncher(unittest.TestCase):
    """A Laya checkpoint has no root config.json: the registry knows it by its
    rl_agent_config.json and its encoder's config. A GLiNER2 checkpoint's
    config.json says "extractor": the registry reads its architecture and its
    encoder_config. coli describes both and refuses to chat with them."""

    def checkpoint(self, encoder):
        import tempfile
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        (root / "encoder").mkdir()
        (root / "rl_agent_config.json").write_text(json.dumps({"head_layers": 2}), encoding="utf-8")
        (root / "encoder" / "config.json").write_text(json.dumps(encoder), encoding="utf-8")
        return root

    def test_the_registry_knows_a_laya_checkpoint(self):
        from family_registry import UnknownFamilyError, default_model_id, display_for, resolve_model
        english = resolve_model(self.checkpoint({"model_type": "modernbert", "hidden_size": 1024,
                                                 "num_hidden_layers": 28}))
        self.assertEqual((english.descriptor.id, english.descriptor.modality), ("laya", "decision"))
        self.assertTrue(english.descriptor.capabilities.decision)
        self.assertEqual(display_for(english), ("Laya", "421M"))
        multilingual = resolve_model(self.checkpoint({"model_type": "modernbert", "hidden_size": 768,
                                                      "num_hidden_layers": 22}))
        self.assertEqual(display_for(multilingual), ("Laya multilingual", "322M"))
        self.assertEqual(default_model_id(multilingual), "laya-multilingual")
        with self.assertRaises(UnknownFamilyError):
            resolve_model(self.checkpoint({"model_type": "deberta-v2"}))

    def gliner_checkpoint(self, config, encoder):
        import tempfile
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        (root / "encoder_config").mkdir()
        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (root / "encoder_config" / "config.json").write_text(json.dumps(encoder), encoding="utf-8")
        return root

    def test_the_registry_knows_a_gliner2_checkpoint(self):
        from family_registry import UnknownFamilyError, default_model_id, display_for, resolve_model
        large = {"model_type": "deberta-v2", "hidden_size": 1024, "num_hidden_layers": 24}
        decide = resolve_model(self.gliner_checkpoint(
            {"model_type": "extractor", "architecture": "span"}, large))
        self.assertEqual((decide.descriptor.id, decide.descriptor.modality),
                         ("gliner_decide", "decision"))
        self.assertEqual(decide.model_type, "gliner2_span_deberta-v2")
        self.assertTrue(decide.descriptor.capabilities.decision)
        self.assertEqual(display_for(decide), ("GLiNER2.5-Decide", "340M"))
        self.assertEqual(default_model_id(decide), "gliner2.5-decide")
        # a legacy config without an architecture is "span", as AutoExtractor reads it
        legacy = resolve_model(self.gliner_checkpoint({"model_type": "extractor"}, large))
        self.assertEqual(legacy.descriptor.id, "gliner_decide")
        # another geometry is named by its own model_type, not by the 340M's
        base = resolve_model(self.gliner_checkpoint(
            {"model_type": "extractor"}, {"model_type": "deberta-v2", "hidden_size": 768,
                                          "num_hidden_layers": 12}))
        self.assertEqual(display_for(base), ("gliner2_span_deberta-v2", ""))
        for config, encoder in (({"model_type": "extractor", "architecture": "boundary"}, large),
                                ({"model_type": "extractor"}, {"model_type": "modernbert"})):
            with self.subTest(config=config, encoder=encoder), self.assertRaises(UnknownFamilyError):
                resolve_model(self.gliner_checkpoint(config, encoder))

    def test_coli_describes_it_and_refuses_to_chat(self):
        import subprocess
        # each fixture with its weights generated and its engine built: coli plan reads the
        # weights' header, and coli chat says the engine is not built before anything else
        fixtures = [path for path, binary in ((FIXTURE, BINARY), (GLINER_FIXTURE, GLINER_BINARY))
                    if (path / "model.safetensors").exists() and binary.exists()]
        if not fixtures:
            self.skipTest("no tiny decision fixture with its engine built "
                          "(make laya-tiny-check / gliner-decide-tiny-check)")
        def coli(*args):
            return subprocess.run([sys.executable, str(HERE / "coli"), *args], capture_output=True,
                                  text=True, timeout=120, cwd=str(HERE))
        for fixture in fixtures:
            with self.subTest(fixture=fixture.name):
                info = coli("info", "--model", str(fixture))
                self.assertIn("decision model", info.stdout)
                plan = coli("plan", "--model", str(fixture), "--json")
                self.assertEqual(json.loads(plan.stdout)["modality"], "decision")
                doctor = json.loads(coli("doctor", "--model", str(fixture), "--json").stdout)
                checks = [{"id": c["id"], "status": c["status"]} for c in doctor["checks"]]
                self.assertIn({"id": "model.family", "status": "pass"}, checks)
                self.assertNotIn({"id": "model.files", "status": "fail"}, checks)
                for command in (("chat", "--model", str(fixture), "--no-attach"),
                                ("run", "--model", str(fixture), "hello")):
                    refused = coli(*command)
                    self.assertNotEqual(refused.returncode, 0)
                    self.assertIn("POST /v1/systemone", refused.stdout + refused.stderr)


def _laya_ready():
    return (FIXTURE / "model.safetensors").exists() and BINARY.exists()


class LayaTinyEndToEnd(unittest.TestCase):
    """The real engine behind the real gateway, on the tiny fixture."""

    @classmethod
    def setUpClass(cls):
        if not _laya_ready():
            message = "laya or laya_tiny missing: run make laya-tiny-check"
            if REQUIRED:
                raise AssertionError(message)
            raise unittest.SkipTest(message)
        cls.ref = {case["name"]: case for case in
                   json.loads((FIXTURE / "ref.json").read_text(encoding="utf-8"))["cases"]}

    def setUp(self):
        engine = Engine(BINARY, FIXTURE, env=dict(os.environ, OMP_NUM_THREADS="2"),
                        family=family_by_id("laya"))
        self.addCleanup(engine.process.stdout.close)
        self.addCleanup(engine.close)
        self.server = APIServer(("127.0.0.1", 0), engine, "laya-tiny")
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def post(self, path, body):
        request = Request(self.base + path, data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=120) as response:
            return json.loads(response.read()), dict(response.headers)

    def test_systemone_answers_with_the_reference_probabilities(self):
        # the cases the route accepts as Jev requests (it refuses a bare number as a
        # score level, which the engine-level oracle still covers)
        for name in ("readme_ticket", "plain_two_options", "conversation_truncated_left",
                     "json_instructions_and_state", "special_text", "unicode"):
            case = self.ref[name]
            with self.subTest(case=name):
                out, headers = self.post("/v1/systemone", {"model": "jev-latest",
                                                           "state": case["state"],
                                                           "questions": case["questions"]})
                for want in case["answers"]:
                    got = out["answers"][want["id"]]
                    probs = want["probs"]
                    if got["type"] == "noul":
                        self.assertAlmostEqual(got["noul"], probs[1], delta=2e-5)
                        continue
                    values = list(got["probabilities"].values())
                    self.assertEqual(len(values), len(probs))
                    for a, b in zip(values, probs):
                        self.assertAlmostEqual(a, b, delta=2e-5)
                    if got["type"] == "choice":
                        labels = list(case["questions"][want["id"]]["criteria"])
                        self.assertEqual(got["choice"], labels[probs.index(max(probs))])
                self.assertEqual(out["usage"]["input_tokens"],
                                 sum(len(a["ids"]) for a in case["answers"]))
                self.assertEqual(out["usage"]["output_tokens"], 0)
                self.assertIn("x-colibri-engine-ms", headers)

    def test_a_question_whose_options_do_not_fit_is_a_422(self):
        case = self.ref["too_many_options"]
        request = Request(self.base + "/v1/systemone",
                          data=json.dumps({"state": case["state"], "questions": case["questions"]}).encode(),
                          headers={"Content-Type": "application/json"})
        with self.assertRaises(HTTPError) as caught:
            urlopen(request, timeout=60)
        self.assertEqual(caught.exception.code, 422)
        error = json.loads(caught.exception.read())["error"]
        self.assertEqual(error["param"], "questions.bucket")
        self.assertIn("only 38 of its 40 option markers fit", error["message"])

    def test_chat_is_refused(self):
        request = Request(self.base + "/v1/chat/completions",
                          data=json.dumps({"model": "laya-tiny",
                                           "messages": [{"role": "user", "content": "hi"}]}).encode(),
                          headers={"Content-Type": "application/json"})
        with self.assertRaises(HTTPError) as caught:
            urlopen(request, timeout=10)
        self.assertEqual(caught.exception.code, 400)

    def test_latency_gateway_and_engine(self):
        """Not a gate: the numbers the report quotes, measured the same way every
        time. Wall time at the client, the engine's own compute time, and the rest
        (HTTP, JSON, the pipe) as the gateway's share."""
        case = self.ref["readme_ticket"]
        body = {"model": "jev-latest", "state": case["state"], "questions": case["questions"]}
        self.post("/v1/systemone", body)                       # warm
        walls, engines = [], []
        for _ in range(10):
            started = time.perf_counter()
            _, headers = self.post("/v1/systemone", body)
            walls.append((time.perf_counter() - started) * 1e3)
            engines.append(float(headers["x-colibri-engine-ms"]))
        walls.sort(), engines.sort()
        wall, engine = walls[len(walls) // 2], engines[len(engines) // 2]
        sys.stderr.write(f"\n[laya tiny e2e] /v1/systemone, 4 questions: median {wall:.1f} ms at the "
                         f"client, {engine:.1f} ms in the engine, {wall - engine:.1f} ms gateway\n")
        self.assertGreater(wall, engine)


def _gliner_ready():
    return (GLINER_FIXTURE / "model.safetensors").exists() and GLINER_BINARY.exists()


class GlinerDecideTinyEndToEnd(unittest.TestCase):
    """The gliner_decide engine behind the real gateway, on its tiny fixture."""

    @classmethod
    def setUpClass(cls):
        if not _gliner_ready():
            message = "gliner_decide or gliner_decide_tiny missing: run make gliner-decide-tiny-check"
            if GLINER_REQUIRED:
                raise AssertionError(message)
            raise unittest.SkipTest(message)
        ref = json.loads((GLINER_FIXTURE / "ref.json").read_text(encoding="utf-8"))
        cls.max_len = ref["max_len"]
        cls.ref = {case["name"]: case for case in ref["cases"]}

    def setUp(self):
        env = dict(os.environ, OMP_NUM_THREADS="2", COLI_GLINER_MAX_LEN=str(self.max_len))
        engine = Engine(GLINER_BINARY, GLINER_FIXTURE, env=env, family=family_by_id("gliner_decide"))
        self.addCleanup(engine.process.stdout.close)
        self.addCleanup(engine.close)
        self.server = APIServer(("127.0.0.1", 0), engine, "gliner-decide-tiny")
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def post(self, path, body):
        request = Request(self.base + path, data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=120) as response:
            return json.loads(response.read()), dict(response.headers)

    def test_systemone_answers_with_the_reference_probabilities(self):
        # the cases the route accepts as Jev requests (it refuses a bare number as a
        # criterion, which the engine-level oracle still covers)
        for name in ("ticket_three_types", "two_options_and_noul_criteria",
                     "twelve_labels_no_instructions", "long_state_cut", "conversation",
                     "json_state_and_instructions", "special_tokens_and_whitespace",
                     "urls_mail_handles", "unicode", "many_options", "empty_state"):
            case = self.ref[name]
            with self.subTest(case=name):
                out, headers = self.post("/v1/systemone", {"model": "jev-latest",
                                                           "state": case["state"],
                                                           "questions": case["questions"]})
                self.assertEqual(out["model"], "gliner-decide-tiny")
                for want in case["answers"]:
                    got = out["answers"][want["id"]]
                    probs = want["probs"]
                    if got["type"] == "noul":
                        self.assertAlmostEqual(got["noul"], probs[1], delta=2e-5)
                        continue
                    values = list(got["probabilities"].values())
                    self.assertEqual(len(values), len(probs))
                    for a, b in zip(values, probs):
                        self.assertAlmostEqual(a, b, delta=2e-5)
                    if got["type"] == "choice":
                        labels = list(case["questions"][want["id"]]["criteria"])
                        self.assertEqual(got["choice"], labels[probs.index(max(probs))])
                    else:
                        expected = sum(i * p for i, p in enumerate(probs))
                        self.assertAlmostEqual(got["score"], expected, delta=1e-4)
                # one sequence for the whole request
                self.assertEqual(out["usage"]["input_tokens"], len(case["ids"]))
                self.assertEqual(out["usage"]["output_tokens"], 0)
                self.assertIn("x-colibri-engine-ms", headers)

    def test_questions_that_do_not_fit_are_a_422(self):
        case = self.ref["questions_over_max_len"]
        request = Request(self.base + "/v1/systemone",
                          data=json.dumps({"state": case["state"], "questions": case["questions"]}).encode(),
                          headers={"Content-Type": "application/json"})
        with self.assertRaises(HTTPError) as caught:
            urlopen(request, timeout=60)
        self.assertEqual(caught.exception.code, 422)
        error = json.loads(caught.exception.read())["error"]
        self.assertEqual(error["param"], "questions")
        self.assertIn(case["refused"].split(": ", 1)[1], error["message"])

    def test_chat_is_refused(self):
        request = Request(self.base + "/v1/chat/completions",
                          data=json.dumps({"model": "gliner-decide-tiny",
                                           "messages": [{"role": "user", "content": "hi"}]}).encode(),
                          headers={"Content-Type": "application/json"})
        with self.assertRaises(HTTPError) as caught:
            urlopen(request, timeout=10)
        self.assertEqual(caught.exception.code, 400)

    def test_latency_gateway_and_engine(self):
        """Not a gate: wall time at the client, the engine's own compute time, and
        the rest (HTTP, JSON, the pipe) as the gateway's share."""
        case = self.ref["ticket_three_types"]
        body = {"model": "jev-latest", "state": case["state"], "questions": case["questions"]}
        self.post("/v1/systemone", body)                       # warm
        walls, engines = [], []
        for _ in range(10):
            started = time.perf_counter()
            _, headers = self.post("/v1/systemone", body)
            walls.append((time.perf_counter() - started) * 1e3)
            engines.append(float(headers["x-colibri-engine-ms"]))
        walls.sort(), engines.sort()
        wall, engine = walls[len(walls) // 2], engines[len(engines) // 2]
        sys.stderr.write(f"\n[gliner_decide tiny e2e] /v1/systemone, 4 questions: median {wall:.1f} ms "
                         f"at the client, {engine:.1f} ms in the engine, {wall - engine:.1f} ms gateway\n")
        self.assertGreater(wall, engine)


if __name__ == "__main__":
    unittest.main()
