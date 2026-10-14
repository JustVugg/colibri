"""POST /v1/systemone: the request and the reply of TypeSafe's Jev API, and
colibri's one decision API.

A client written for Jev sends `state`, `model` and a map of questions typed
noul / choice / score, and reads `answers` keyed by its own ids plus
`usage.input_tokens/output_tokens`. On a language model each question is
scored through the engine's logprob channel; this pins that path with a
deterministic scoring engine: what each primitive puts in the prompt, what
comes back, the confidence formula their docs give, that any model name is
accepted on this route, that nothing is generated, and which photographs the
state and the questions get. (Up to 1.12.1 the same channel had its own route,
POST /v1/brio; it is gone, and the test that it is gone is here.)

MimoSystemOneEndToEnd puts the real MiMo engine behind the same server, on the
tiny fixture (`make mimo mimo-tiny-generate`; skipped without them): every
option after the photo reads only its own tokens, and the probabilities equal
a cold engine's.
"""
import http.client
import json
import math
import os
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mimo_serve_fixture  # noqa: E402
import openai_server  # noqa: E402
from family_registry import family_by_id  # noqa: E402
from openai_server import APIServer, cors_origin_list  # noqa: E402


class ScoringEngine:
    """Scores like the real channel, deterministically.

    `table` maps an option string to the mean log-probability its tokens get;
    anything not in the table scores -5.0. A pinned call answers with ACCEPT
    (prompt_tokens) and one ECHO per position, like an engine with the channel."""

    def __init__(self, table):
        self.table = table
        self.calls = []
        self.kv_slots = 1

    def generate(self, prompt, max_tokens, temperature, top_p, on_text, cache_slot=0,
                 cancelled=None, grammar=None, stopped=None, on_accept=None, audio=None,
                 on_tool=None, image=None, logprobs=0, pin=False, on_echo=None):
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens, "pin": bool(pin),
                           "logprobs": logprobs})
        words = prompt.split()
        if on_accept:
            on_accept({"prompt_tokens": len(words)})
        if on_echo and logprobs:
            # The option is whatever follows the last "Answer:" or the last
            # opening quote of a skeleton cell; score its words from the table.
            option = None
            for key in self.table:
                if prompt.endswith(" " + key):
                    option = key
            per_token = self.table.get(option, -5.0)
            for pos, _word in enumerate(words):
                on_echo({"pos": pos, "logprob": per_token if option and pos >= len(words) - len(option.split()) else -1.0})

    def close(self):
        pass


STATE = ("Hi, I've been trying to connect my Stripe account for 3 days and the "
         "integration keeps failing. I'm losing sales. Please help ASAP.")


def confidence(ps):
    n = len(ps)
    return (n * max(ps) - 1.0) / (n - 1)


class SlotScoringEngine(ScoringEngine):
    """ScoringEngine that also writes down the KV slot of every call."""

    def __init__(self, table, kv_slots=1):
        super().__init__(table)
        self.kv_slots = kv_slots

    def generate(self, prompt, max_tokens, temperature, top_p, on_text, cache_slot=0, *args, **kwargs):
        super().generate(prompt, max_tokens, temperature, top_p, on_text, cache_slot, *args, **kwargs)
        self.calls[-1]["slot"] = cache_slot


class SystemOneApi(unittest.TestCase):
    def serve(self, table, kv_slots=1):
        self.engine = SlotScoringEngine(table, kv_slots)
        self.server = APIServer(("127.0.0.1", 0), self.engine, "test-model", kv_slots=kv_slots)
        self.addCleanup(self.server.shutdown)
        self.addCleanup(self.server.server_close)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def post(self, body):
        request = Request(self.base + "/v1/systemone", data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=10) as response:
            return json.loads(response.read())

    def post_error(self, body):
        try:
            self.post(body)
        except HTTPError as error:
            return error.code, json.loads(error.read())
        self.fail("expected an error")

    def pins(self):
        return [c["prompt"] for c in self.engine.calls if c["pin"]]

    def test_noul_is_the_probability_of_yes(self):
        self.serve({"yes": -0.1, "no": -2.0})
        out = self.post({"model": "jev-latest", "state": STATE, "questions": {
            "urgency": {"type": "noul", "instructions": "Does this message express urgency?"}}})
        answer = out["answers"]["urgency"]
        self.assertEqual(answer["type"], "noul")
        want = math.exp(-0.1) / (math.exp(-0.1) + math.exp(-2.0))
        self.assertAlmostEqual(answer["noul"], want, places=5)
        self.assertNotIn("confidence", answer)          # their docs: noul carries none
        # the question is asked as a yes/no question, the state as the context
        asked = [c["prompt"] for c in self.engine.calls if "Does this message" in c["prompt"]]
        self.assertTrue(asked and "Answer yes or no." in asked[0], asked[:1])

    def test_noul_criteria_go_into_the_question(self):
        self.serve({"yes": -0.1, "no": -2.0})
        self.post({"model": "jev-latest", "state": STATE, "questions": {
            "q": {"type": "noul", "instructions": "Is the customer at risk of churning?",
                  "criteria": {"true": "they threaten to leave or mention losses",
                               "false": "a routine question"}}}})
        pinned = self.pins()[-1]
        self.assertIn("yes: they threaten to leave", pinned)
        self.assertIn("no: a routine question", pinned)

    def test_choice_labels_probabilities_and_confidence(self):
        self.serve({"billing": -0.2, "technical": -3.0, "sales": -4.0})
        out = self.post({"model": "jev-latest", "state": STATE, "questions": {
            "department": {"type": "choice", "instructions": "Which team should handle this?",
                           "criteria": {"billing": "payments, invoices, Stripe",
                                        "technical": "bugs and outages",
                                        "sales": "pricing and plans"}}}})
        answer = out["answers"]["department"]
        self.assertEqual(answer["type"], "choice")
        self.assertEqual(answer["choice"], "billing")
        self.assertEqual(list(answer["probabilities"]), ["billing", "technical", "sales"])
        self.assertAlmostEqual(sum(answer["probabilities"].values()), 1.0, places=5)
        self.assertAlmostEqual(answer["confidence"],
                               confidence(list(answer["probabilities"].values())), places=5)
        # the descriptions are in the question, the labels are the options scored
        pinned = self.pins()[-1]
        self.assertIn("- billing: payments, invoices, Stripe", pinned)
        self.assertIn("- sales: pricing and plans", pinned)
        scored = [c["prompt"] for c in self.engine.calls if not c["pin"]]
        self.assertTrue(any(p.endswith(" technical") for p in scored), scored[-3:])

    def test_choice_with_null_descriptions_still_lists_the_labels(self):
        self.serve({"a": -0.1, "b": -1.0})
        out = self.post({"model": "jev-latest", "state": STATE, "questions": {
            "q": {"type": "choice", "criteria": {"a": None, "b": None}}}})
        self.assertEqual(out["answers"]["q"]["choice"], "a")
        self.assertIn("- a\n- b", self.pins()[-1])

    def test_score_expected_value_legend_and_confidence(self):
        # Jev numbers the levels from zero ("Each description's position determines
        # its score, starting at zero", its OpenAPI schema); the model reads 1..n.
        self.serve({"4": -0.1, "1": -5.0, "2": -5.0, "3": -5.0, "5": -5.0})
        levels = ["not urgent", "low", "moderate", "high", "critical"]
        out = self.post({"model": "jev-latest", "state": STATE, "questions": {
            "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": levels}}})
        answer = out["answers"]["urgency"]
        self.assertEqual(answer["type"], "score")
        self.assertEqual(answer["legend"], {str(i): d for i, d in enumerate(levels)})
        self.assertEqual(list(answer["probabilities"]), ["0", "1", "2", "3", "4"])
        ps = answer["probabilities"]
        self.assertAlmostEqual(sum(ps.values()), 1.0, places=5)
        self.assertAlmostEqual(answer["score"], sum(int(k) * v for k, v in ps.items()), places=5)
        self.assertGreater(answer["score"], 2.9)             # the mass sits on level 3, "high"
        self.assertAlmostEqual(answer["confidence"], confidence(list(ps.values())), places=5)
        self.assertIn("4: high", self.pins()[-1])

    def test_legend_gives_back_the_criteria_as_sent(self):
        self.serve({"1": -0.1, "2": -3.0})
        rubric = [{"level": "calm", "examples": ["thanks"]}, "angry"]
        out = self.post({"model": "jev-latest", "state": STATE, "questions": {
            "tone": {"type": "score", "criteria": rubric}}})
        self.assertEqual(out["answers"]["tone"]["legend"], {"0": rubric[0], "1": "angry"})

    def test_reply_carries_the_fields_the_jev_sdks_read(self):
        self.serve({"yes": -0.1, "no": -2.0})
        request = Request(self.base + "/v1/systemone",
                          data=json.dumps({"model": "jev-latest", "state": STATE, "questions": {
                              "q": {"type": "noul", "instructions": "Urgent?"}}}).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=10) as response:
            out = json.loads(response.read())
            request_id = response.headers["x-typesafe-request-id"]
        self.assertEqual(out["id"], request_id)
        self.assertTrue(request_id.startswith("req_"))
        self.assertEqual(out["provider"], "colibri")
        self.assertEqual(out["model"], "test-model")
        self.assertEqual(out["usage"]["cost"], 0)
        self.assertIsInstance(out["answers"]["q"]["noul"], float)

    def test_one_label_or_one_level_is_answered_without_the_engine(self):
        self.serve({"yes": -0.1, "no": -2.0})
        out = self.post({"model": "jev-latest", "state": STATE, "questions": {
            "only": {"type": "choice", "criteria": {"billing": "payments"}},
            "flat": {"type": "score", "criteria": ["the one level"]}}})
        self.assertEqual(out["answers"]["only"], {"type": "choice", "choice": "billing",
                                                  "probabilities": {"billing": 1.0}, "confidence": 1.0})
        self.assertEqual(out["answers"]["flat"]["probabilities"], {"0": 1.0})
        self.assertEqual(out["answers"]["flat"]["score"], 0.0)
        self.assertEqual(self.engine.calls, [])

    def test_an_empty_text_state_is_a_state(self):
        self.serve({"yes": -0.1, "no": -2.0})
        out = self.post({"model": "jev-latest", "state": "", "questions": {
            "q": {"type": "noul", "instructions": "Is there anything here?"}}})
        self.assertIn("noul", out["answers"]["q"])
        self.assertFalse(any(c["prompt"].startswith("Context:") for c in self.engine.calls))

    def test_structured_state_and_instructions_are_serialized(self):
        self.serve({"yes": -0.1, "no": -2.0})
        out = self.post({"model": "jev-latest",
                         "state": {"ticket": 4711, "text": "refund not received"},
                         "questions": {"q": {"type": "noul",
                                             "instructions": {"ask": "is this about money?"}}}})
        self.assertEqual(out["model"], "test-model")       # the served model answers
        question_pin = self.pins()[-1]                    # the state, then the question
        self.assertIn('"ticket": 4711', question_pin)
        self.assertIn('"ask": "is this about money?"', question_pin)

    def test_state_is_photographed_once_for_all_questions(self):
        self.serve({"yes": -0.1, "no": -2.0, "a": -0.1, "b": -1.0})
        out = self.post({"model": "jev-latest", "state": STATE, "questions": {
            "one": {"type": "noul", "instructions": "q1?"},
            "two": {"type": "choice", "criteria": {"a": "x", "b": "y"}},
            "three": {"type": "score", "criteria": ["bad", "good"]}}})
        self.assertEqual(list(out["answers"]), ["one", "two", "three"])
        pins = self.pins()
        self.assertEqual(len(pins), 4, pins)               # the state, then each question once
        self.assertEqual(pins[0], f"Context:\n{STATE}\n\n")
        # every question prefix extends the state's photo and ends where an
        # option is read as the answer's continuation; options are never pinned
        for pin in pins[1:]:
            self.assertTrue(pin.startswith(pins[0]), "a question prefix must extend the state")
            self.assertTrue(pin.endswith("\nAnswer:"), pin[-40:])
        # nothing is generated: every engine call reads the prompt and stops
        self.assertTrue(all(c["max_tokens"] == 0 and c["logprobs"] for c in self.engine.calls))
        self.assertEqual(set(out["usage"]), {"input_tokens", "output_tokens", "cost"})
        self.assertGreater(out["usage"]["input_tokens"], 0)
        self.assertEqual(out["usage"]["output_tokens"], 6)  # two one-token options x three

    def test_mean_with_unequal_option_token_counts_warns(self):
        import contextlib
        import io
        self.serve({"merge": -3.0, "request changes": -0.2, "yes": -0.1, "no": -2.0})
        unequal = {"q": {"type": "choice", "criteria": {"merge": None, "request changes": None}}}
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            self.post({"state": STATE, "questions": unequal, "normalize": "mean"})
        self.assertIn("unequal option token counts", stderr.getvalue())
        self.assertIn("merge=1", stderr.getvalue())
        # Equal token counts stay quiet: mean is valid there. (The [api] request
        # log also lands on stderr; only the bias warning matters.)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            self.post({"state": STATE, "questions": {"q": {"type": "noul", "instructions": "?"}},
                       "normalize": "mean"})
        self.assertNotIn("unequal option token counts", stderr.getvalue())

    def test_validation_errors_are_422(self):
        self.serve({})
        code, body = self.post_error({"model": "jev-latest", "questions": {"q": {"type": "noul"}}})
        self.assertEqual(code, 422)
        self.assertIn("state", body["error"]["message"])
        code, body = self.post_error({"model": "jev-latest", "state": STATE,
                                      "questions": {"q": {"type": "guess"}}})
        self.assertEqual(code, 422)
        self.assertIn("type", body["error"]["message"])
        code, body = self.post_error({"model": "jev-latest", "state": STATE,
                                      "questions": {"q": {"type": "choice", "criteria": ["a", "b"]}}})
        self.assertEqual(code, 422)
        code, body = self.post_error({"model": "jev-latest", "state": STATE,
                                      "questions": {"q": {"type": "score", "criteria": []}}})
        self.assertEqual(code, 422)
        self.assertIn("1 to 255", body["error"]["message"])
        code, body = self.post_error({"model": "jev-latest", "state": STATE, "questions": []})
        self.assertEqual(code, 422)

    # ---- colibri's options (none of which a Jev client sends) --------------
    def test_normalize_defaults_to_sum_and_mean_can_be_asked(self):
        # "a b" reads two tokens at -1.0 each, "c" one at -1.5: the joint
        # probability (sum, -2.0 vs -1.5) picks "c", the per-token mean "a b".
        question = {"q": {"type": "choice", "criteria": {"a b": None, "c": None}}}
        self.serve({"a b": -1.0, "c": -1.5})
        self.assertEqual(self.post({"state": STATE, "questions": question})["answers"]["q"]["choice"], "c")
        out = self.post({"state": STATE, "questions": question, "normalize": "mean"})
        self.assertEqual(out["answers"]["q"]["choice"], "a b")
        code, body = self.post_error({"state": STATE, "questions": question, "normalize": "max"})
        self.assertEqual((code, body["error"]["param"]), (422, "normalize"))

    def test_a_state_seen_once_is_not_photographed(self):
        self.serve({"yes": -0.1, "no": -2.0})
        question = {"q": {"type": "noul", "instructions": "Urgent?"}}
        state_pins = lambda: [p for p in self.pins() if p.endswith("\n\n") and "Question:" not in p]
        self.post({"state": "frame 1", "questions": question})
        self.post({"state": "frame 2", "questions": question})
        self.assertEqual(state_pins(), [])                  # a feed: never read twice
        self.post({"state": "frame 2", "questions": question})
        self.assertEqual(state_pins(), ["Context:\nframe 2\n\n"])   # it came back: worth it
        self.post({"state": "frame 3", "questions": dict(question, other={"type": "noul"})})
        self.assertEqual(state_pins()[-1], "Context:\nframe 3\n\n")  # two questions share it
        before = len(state_pins())
        self.post({"state": "frame 4", "questions": dict(question, other={"type": "noul"}),
                   "pin_state": False})
        self.assertEqual(len(state_pins()), before)
        self.post({"state": "frame 5", "questions": question, "pin_state": True})
        self.assertEqual(state_pins()[-1], "Context:\nframe 5\n\n")
        code, body = self.post_error({"state": "x", "questions": question, "pin_state": "yes"})
        self.assertEqual((code, body["error"]["param"]), (422, "pin_state"))

    def test_a_fixed_prefix_is_photographed_before_the_state(self):
        self.serve({"yes": -0.1, "no": -2.0})
        rules = "You play Flappy Bird. Flap when the pipe gap is above the bird."
        for frame in ("bird y=10, gap y=40", "bird y=12, gap y=40"):
            self.post({"state": frame, "prefix": rules,
                       "questions": {"flap": {"type": "noul", "instructions": "Flap now?"}}})
        pins = self.pins()
        self.assertEqual(pins.count(rules + "\n\n"), 2)
        question_pins = [p for p in pins if "Question:" in p]
        self.assertTrue(all(p.startswith(rules + "\n\nContext:\nbird y=") for p in question_pins))
        code, body = self.post_error({"state": "x", "prefix": "", "questions": {"q": {"type": "noul"}}})
        self.assertEqual((code, body["error"]["param"]), (422, "prefix"))

    def test_cache_slot_is_forwarded_and_the_default_is_unchanged(self):
        self.serve({"yes": -0.1, "no": -2.0}, kv_slots=2)
        question = {"q": {"type": "noul", "instructions": "Urgent?"}}
        self.post({"state": STATE, "questions": question, "cache_slot": 1})
        self.assertEqual({c["slot"] for c in self.engine.calls}, {1})
        self.engine.calls.clear()
        self.post({"state": STATE, "questions": question})
        want = openai_server.conversation_cache_slot([{"role": "system", "content": STATE}], 2)
        self.assertEqual({c["slot"] for c in self.engine.calls}, {want})
        code, body = self.post_error({"state": STATE, "questions": question, "cache_slot": 2})
        self.assertEqual((code, body["error"]["param"]), (422, "cache_slot"))

    def test_models_says_the_model_answers_systemone(self):
        self.serve({})
        with urlopen(self.base + "/v1/models", timeout=10) as response:
            card = json.loads(response.read())["data"][0]
        self.assertEqual(card["capabilities"], ["chat", "systemone"])

    def test_a_browser_can_read_the_timing_headers(self):
        self.serve({"yes": -0.1, "no": -2.0})
        request = Request(self.base + "/v1/systemone",
                          data=json.dumps({"state": STATE, "questions": {
                              "q": {"type": "noul", "instructions": "Urgent?"}}}).encode(),
                          headers={"Content-Type": "application/json",
                                   "Origin": "http://localhost:5173"})
        with urlopen(request, timeout=10) as response:
            exposed = {h.strip().lower() for h in
                       response.headers["Access-Control-Expose-Headers"].split(",")}
            self.assertIn("x-colibri-elapsed-ms", response.headers)
        self.assertTrue({"x-colibri-elapsed-ms", "x-colibri-queue-wait-ms",
                         "x-colibri-engine-ms", "x-typesafe-request-id"} <= exposed)

    def test_a_kept_alive_round_trip_does_not_wait_for_a_delayed_ack(self):
        """The header block and the body are two writes. With Nagle on, the body
        waits for the client's delayed ACK of the headers: about 40 ms per request
        on loopback (measured 44 ms median before TCP_NODELAY, 1 ms after)."""
        self.serve({"yes": -0.1, "no": -2.0})
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=10)
        self.addCleanup(connection.close)
        body = json.dumps({"state": STATE, "questions": {"q": {"type": "noul", "instructions": "?"}}})
        times = []
        for _ in range(15):
            started = time.perf_counter()
            connection.request("POST", "/v1/systemone", body, {"Content-Type": "application/json"})
            connection.getresponse().read()
            times.append((time.perf_counter() - started) * 1e3)
        median = sorted(times[3:])[len(times[3:]) // 2]
        self.assertLess(median, 20.0, f"keep-alive round trips: {[round(t, 1) for t in times]} ms")

    def test_validation_errors_carry_jevs_detail_list(self):
        self.serve({})
        code, body = self.post_error({"model": "jev-latest", "state": STATE, "questions": {
            "q": {"type": "score", "criteria": ["fine", 3]}}})
        self.assertEqual(code, 422)
        self.assertEqual(body["error"]["param"], "questions.q.criteria[1]")
        self.assertEqual(body["detail"], [{"loc": ["body", "questions", "q", "criteria", 1],
                                           "msg": body["error"]["message"], "type": "value_error"}])

    def test_models_lists_the_jev_card_beside_the_openai_one(self):
        self.serve({})
        with urlopen(self.base + "/v1/models", timeout=10) as response:
            listing = json.loads(response.read())
        self.assertEqual(listing["data"][0]["id"], "test-model")
        card = listing["models"][0]
        self.assertEqual(set(card), {"name", "description", "release_date"})
        self.assertEqual(card["name"], "test-model")
        self.assertRegex(card["release_date"], r"^\d{4}-\d{2}-\d{2}$")

    def test_v1_brio_is_gone(self):
        """/v1/systemone is the one decision API; the route that scored closed
        sets on its own up to 1.12.1 answers like any unknown route."""
        self.serve({"a": -0.1, "b": -1.0})
        request = Request(self.base + "/v1/brio",
                          data=json.dumps({"model": "test-model", "state": STATE,
                                           "options": ["a", "b"]}).encode(),
                          headers={"Content-Type": "application/json"})
        with self.assertRaises(HTTPError) as caught:
            urlopen(request, timeout=10)
        self.assertEqual(caught.exception.code, 404)
        self.assertEqual(json.loads(caught.exception.read())["error"]["code"], "not_found")
        self.assertEqual(self.engine.calls, [])


class DecideCommand(unittest.TestCase):
    """`/decide` in coli chat posts to /v1/systemone: the conversation so far as
    the state, the question as one choice whose labels are the options."""

    def test_the_tui_request_is_a_systemone_request(self):
        import importlib.machinery
        import importlib.util
        loader = importlib.machinery.SourceFileLoader(
            "coli_decide_test", str(Path(__file__).resolve().parent.parent / "coli"))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        cli = importlib.util.module_from_spec(spec)
        loader.exec_module(cli)
        msgs = [{"role": "user", "content": "The PR has no tests."},
                {"role": "assistant", "content": [{"type": "text", "text": "Noted."},
                                                  {"type": "image_url", "image_url": {"url": "x"}}]}]
        body = cli.decide_request("qwen36", msgs, "What now?", ["merge", "request changes"])
        self.assertEqual(body["state"], "user: The PR has no tests.\nassistant: Noted.")
        self.assertEqual(body["questions"], {"decide": {
            "type": "choice", "instructions": "What now?",
            "criteria": {"merge": None, "request changes": None}}})
        engine = ScoringEngine({"merge": -2.0, "request changes": -0.1})
        server = APIServer(("127.0.0.1", 0), engine, "qwen36")
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        request = Request(f"http://127.0.0.1:{server.server_port}/v1/systemone",
                          data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=10) as response:
            out = json.loads(response.read())
        self.assertEqual(out["answers"]["decide"]["choice"], "request changes")


class CorsOrigins(unittest.TestCase):
    def test_plus_adds_to_the_defaults_and_plain_values_replace_them(self):
        defaults = openai_server.DEFAULT_CORS_ORIGINS
        self.assertEqual(cors_origin_list(None), defaults)
        self.assertEqual(cors_origin_list(["+http://game.local:8080"]),
                         (*defaults, "http://game.local:8080"))
        self.assertEqual(cors_origin_list(["https://ui.example"]), ("https://ui.example",))
        self.assertEqual(cors_origin_list(["https://ui.example", "+http://game.local:8080"]),
                         ("https://ui.example", "http://game.local:8080"))


class _Recorder:
    """The real engine, with every call it is given written down: the prompt, the
    pin, the ACCEPT count and the ECHO positions -- what the saving is read from."""

    def __init__(self, engine):
        self.engine = engine
        self.kv_slots = engine.kv_slots
        self.calls = []

    def generate(self, prompt, *args, on_echo=None, on_accept=None, pin=False, **kwargs):
        call = {"prompt": prompt, "pin": bool(pin), "echo": [], "accept": None}
        self.calls.append(call)

        def echo(record):
            call["echo"].append(record["pos"])
            if on_echo:
                on_echo(record)

        def accept(value):
            call["accept"] = value.get("prompt_tokens")
            if on_accept:
                on_accept(value)

        return self.engine.generate(prompt, *args, on_echo=echo, on_accept=accept,
                                    pin=pin, **kwargs)

    def __getattr__(self, name):
        return getattr(self.engine, name)


def _cold_option_score(prefix, option):
    """What one option scores on a fresh engine that reads the whole prompt: the sum
    of the ECHO log-probabilities past the prefix, from the frames themselves."""
    env = dict(os.environ, **mimo_serve_fixture.engine_env())
    p = subprocess.Popen([str(mimo_serve_fixture.ENGINE), "8"], env=env,
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL)
    try:
        while b"READY" not in p.stdout.readline():
            pass
        n_prefix, values = None, []
        for rid, text in ((1, prefix), (2, prefix + " " + option)):
            data = text.encode()
            p.stdin.write(f"SUBMIT {rid} 0 {len(data)} 0 0 1 logprobs=1\n".encode()
                          + data + b"\n")
            p.stdin.flush()
            while True:
                fields = p.stdout.readline().split()
                if fields[0] == b"ECHO":
                    p.stdout.read(int(fields[2]) + 1)
                    if rid == 2 and int(fields[3]) >= n_prefix and fields[4] != b"nan":
                        values.append(float(fields[4]))
                elif fields[0] == b"ACCEPT" and rid == 1:
                    n_prefix = int(fields[2])
                elif fields[0] in (b"DONE", b"ERROR"):
                    break
        # sum(), as the endpoint adds them: on Python 3.12 it is compensated, and a
        # plain running total differs from it in the last bits
        return sum(values), len(values)
    finally:
        p.stdin.close()
        p.wait(timeout=30)
        p.stdout.close()


@unittest.skipUnless(mimo_serve_fixture.available(),
                     "mimo is not built or the tiny MiMo fixture is absent "
                     "(make mimo mimo-tiny-generate)")
class MimoSystemOneEndToEnd(unittest.TestCase):
    # Short: the fixture has 256 positions, its tokenizer reads one byte per
    # token, and a choice question lists its labels in the prompt.
    STATE = "The release is late and the tests are red."
    OPTIONS = ["ship it", "wait", "wait for green tests"]

    def setUp(self):
        engine = openai_server.Engine(
            mimo_serve_fixture.ENGINE, mimo_serve_fixture.served_fixture(), cap=8,
            env=dict(os.environ, **mimo_serve_fixture.engine_env()),
            family=family_by_id("mimo"))
        self.addCleanup(engine.process.stdout.close)
        self.addCleanup(engine.close)
        self.engine = _Recorder(engine)
        self.server = APIServer(("127.0.0.1", 0), self.engine, "mimo-tiny")
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def post(self, path, body):
        request = Request(self.base + path, data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=120) as response:
            return json.loads(response.read())

    def test_options_read_only_their_own_tokens_and_score_like_a_cold_engine(self):
        out = self.post("/v1/systemone", {
            "model": "mimo-tiny", "state": self.STATE, "pin_state": True,
            "questions": {"q": {"type": "choice", "instructions": "What should we do?",
                                "criteria": {option: None for option in self.OPTIONS}}}})
        answer = out["answers"]["q"]
        self.assertIn(answer["choice"], self.OPTIONS)
        self.assertAlmostEqual(sum(answer["probabilities"].values()), 1.0, places=5)

        calls = self.engine.calls
        state, prefix, options = calls[0], calls[1], calls[2:]
        self.assertTrue(state["pin"] and prefix["pin"])
        self.assertEqual(len(options), len(self.OPTIONS))
        # The question resumes from the state's photo, every option from the
        # question's: each read-out starts exactly where the photo ends and covers
        # nothing before it. A fallback to a full recompute would read from 0.
        self.assertEqual(min(prefix["echo"]), state["accept"])
        read = 0
        for call in options:
            self.assertEqual(sorted(call["echo"]),
                             list(range(prefix["accept"], call["accept"])),
                             f"{call['prompt']!r} did not resume from the photo")
            read += call["accept"] - prefix["accept"]
        self.assertEqual(out["usage"]["output_tokens"], read)
        # The probabilities are the softmax of what a cold engine reads for each
        # option after the same prompt (normalize sum), to the reply's 6 digits.
        cold = {option: _cold_option_score(prefix["prompt"], option)[0] for option in self.OPTIONS}
        top = max(cold.values())
        weights = {option: math.exp(value - top) for option, value in cold.items()}
        total = sum(weights.values())
        for option in self.OPTIONS:
            self.assertAlmostEqual(answer["probabilities"][option], weights[option] / total, places=6,
                                   msg=f"{option!r}: the photo scored differently from a cold engine")

    def test_chat_logprobs_reach_the_engine(self):
        with patch("openai_server.ARCH", "mimo"):
            out = self.post("/v1/chat/completions", {
                "model": "mimo-tiny", "max_tokens": 6, "logprobs": True,
                "top_logprobs": 2, "enable_thinking": False,
                "messages": [{"role": "user", "content": "Ship it?"}]})
        choice = out["choices"][0]
        entries = choice["logprobs"]["content"]
        self.assertTrue(entries, "no per-token logprobs came back")
        for entry in entries:
            self.assertLessEqual(entry["logprob"], 0.0)
            self.assertEqual(len(entry["top_logprobs"]), 2)


if __name__ == "__main__":
    unittest.main()
