"""Context compaction end to end: the gateway over HTTP, a mock engine with a small context.

The mock counts a token per four prompt bytes and refuses a prompt at its capacity the way
the engines do (ERROR <id> CONTEXT_EXCEEDED prompt_tokens=N requested=M capacity=C). It
answers the gateway's request to summarize with a fixed summary and every other prompt
with a short reply, and logs every prompt it receives.

A request without `context_compaction` keeps the standard 400. With it, the conversation
that does not fit runs on a summary of its first messages: the summary is asked for after
the history the engine holds, without the new message; the turn's prompt carries the
summary in the system message and the last messages verbatim; the response says what
was summarized; the next turn finds the same summary without asking for it again.
"""
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

SERVER = Path(__file__).resolve().parent.parent / "openai_server.py"
CAPACITY = 600            # tokens, a token per four bytes
SUMMARY = "SUMMARY: the user listed topics one to four."

MOCK_ENGINE = r'''#!/usr/bin/env python3
import json, os, sys
out, inp = sys.stdout.buffer, sys.stdin.buffer
cap = int(os.environ["MOCK_CTX"])
out.write(b"\x01\x01READY\x01\x01\n" + b"STAT 0 0 0 0 0\n"); out.flush()

def log(entry):
    with open(os.environ["MOCK_LOG"], "a") as f:
        f.write(json.dumps(entry) + "\n")

while True:
    line = inp.readline()
    if not line: break
    f = line.decode().strip().split()
    if not f or f[0] != "SUBMIT": continue
    rid, plen, max_tok = f[1], int(f[3]), int(f[4])
    prompt = inp.read(plen).decode("utf-8", "replace"); inp.read(1)
    tokens = (len(prompt.encode()) + 3) // 4
    if tokens >= cap:
        log({"prompt": prompt, "tokens": tokens, "refused": True})
        out.write(("ERROR %s CONTEXT_EXCEEDED prompt_tokens=%d requested=%d capacity=%d\n"
                   % (rid, tokens, max_tok, cap)).encode()); out.flush()
        continue
    log({"prompt": prompt, "tokens": tokens, "refused": False, "max_tokens": max_tok})
    out.write(("ACCEPT %s %d\n" % (rid, tokens)).encode()); out.flush()
    text = (SUMMARY if "Summarize our conversation so far" in prompt
            else "Noted, turn answered.")
    data = text.encode()
    out.write(("DATA %s %d\n" % (rid, len(data))).encode() + data + b"\n")
    out.write(("DONE %s STAT %d 1.0 50.0 1.0 %d 0\n" % (rid, len(text.split()), tokens)).encode())
    out.flush()
'''.replace("SUMMARY", repr(SUMMARY))


def filler(i):
    return f"Topic {i}: " + ("lorem ipsum dolor sit amet " * 10).strip()


def conversation(turns):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for i in range(turns):
        messages.append({"role": "user", "content": filler(i)})
        messages.append({"role": "assistant", "content": f"Reply {i}: " + "noted " * 40})
    messages.append({"role": "user", "content": "What was the last topic?"})
    return messages


@unittest.skipUnless(os.name == "posix",
                     "the mock engine is a shebang script the gateway execs directly")
class ContextCompactionE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        (Path(cls.tmp.name) / "config.json").write_text(
            json.dumps({"model_type": "glm_moe_dsa"}), encoding="utf-8")
        mock = Path(cls.tmp.name) / "mock_engine.py"
        mock.write_text(MOCK_ENGINE)
        mock.chmod(0o755)
        cls.log = Path(cls.tmp.name) / "prompts.jsonl"
        cls.log.touch()
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            cls.port = probe.getsockname()[1]
        env = dict(os.environ, MOCK_LOG=str(cls.log), MOCK_CTX=str(CAPACITY), CTX=str(CAPACITY))
        env.pop("COLI_API_KEY", None)
        env.pop("COLI_COMPACT", None)
        cls.server = subprocess.Popen(
            [sys.executable, str(SERVER), "--model", cls.tmp.name, "--engine", str(mock),
             "--port", str(cls.port), "--max-tokens", "64"],
            env=env, stderr=subprocess.DEVNULL)
        cls.base = f"http://127.0.0.1:{cls.port}/v1"
        for _ in range(100):
            try:
                with urllib.request.urlopen(cls.base + "/models", timeout=2) as resp:
                    cls.model = json.loads(resp.read())["data"][0]["id"]
                    return
            except OSError:
                if cls.server.poll() is not None:
                    raise RuntimeError("gateway exited during startup")
                time.sleep(0.1)
        raise RuntimeError("gateway did not come up")

    @classmethod
    def tearDownClass(cls):
        cls.server.terminate()
        cls.server.wait(timeout=5)
        cls.tmp.cleanup()

    def prompts(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def post(self, body, stream=False):
        body = dict(body, stream=stream, model=self.model)
        req = urllib.request.Request(self.base + "/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            headers = dict(resp.headers)
            if not stream:
                return json.loads(resp.read()), headers
            events = []
            for raw in resp:
                line = raw.decode().strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    events.append(json.loads(line[6:]))
            return events, headers

    def test_1_without_the_opt_in_the_400_stays(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post({"model": "m", "messages": conversation(6), "max_tokens": 16})
        self.assertEqual(caught.exception.code, 400)
        error = json.loads(caught.exception.read())["error"]
        self.assertEqual(error["code"], "context_length_exceeded")

    def test_2_compaction_summarizes_and_continues(self):
        start = len(self.prompts())
        messages = conversation(6)
        reply, headers = self.post({"model": "m", "messages": messages, "max_tokens": 16,
                                    "context_compaction": "auto"})
        self.assertEqual(reply["choices"][0]["message"]["content"], "Noted, turn answered.")
        compaction = reply["compaction"]
        self.assertEqual(compaction["summary"], SUMMARY)
        self.assertGreater(compaction["summarized_messages"], 0)
        self.assertEqual(headers.get("x-colibri-compacted-messages"),
                         str(compaction["summarized_messages"]))
        seen = self.prompts()[start:]
        asks = [p for p in seen if "Summarize our conversation so far" in p["prompt"]]
        self.assertTrue(asks, "the gateway never asked for a summary")
        for ask in asks:
            self.assertFalse(ask["refused"], "a summary request that did not fit was sent")
            # the summary covers the history before the new message, never the message itself
            self.assertNotIn("What was the last topic?", ask["prompt"])
        final = seen[-1]
        self.assertFalse(final["refused"])
        self.assertIn(SUMMARY, final["prompt"])
        self.assertIn("You are a helpful assistant.", final["prompt"])
        self.assertIn("What was the last topic?", final["prompt"])
        self.assertNotIn(filler(0), final["prompt"])           # summarized away
        self.assertLess(final["tokens"], CAPACITY)

        # the next turn: the same summary, found and not asked for again
        messages = messages + [{"role": "assistant", "content": "Noted, turn answered."},
                               {"role": "user", "content": "And the one before?"}]
        before = len(self.prompts())
        reply, _ = self.post({"model": "m", "messages": messages, "max_tokens": 16,
                              "context_compaction": "auto"})
        later = self.prompts()[before:]
        self.assertEqual(reply["compaction"]["summary"], SUMMARY)
        self.assertFalse(any("Summarize our conversation so far" in p["prompt"] for p in later),
                         "the summary was asked for again")
        self.assertEqual(len(later), 1)
        self.assertIn(SUMMARY, later[0]["prompt"])

    def test_3_streaming_reports_the_compaction_first(self):
        messages = conversation(7)
        messages[1] = {"role": "user", "content": filler(70)}  # another conversation
        events, headers = self.post({"model": "m", "messages": messages, "max_tokens": 16,
                                     "context_compaction": "auto"}, stream=True)
        marks = [e for e in events if "compaction" in e]
        self.assertEqual(len(marks), 1)
        first_text = next(i for i, e in enumerate(events)
                          if any(c.get("delta", {}).get("content") for c in e.get("choices", [])))
        self.assertLess(events.index(marks[0]), first_text)
        self.assertEqual(marks[0]["compaction"]["summary"], SUMMARY)
        text = "".join(c["delta"].get("content", "") for e in events for c in e.get("choices", []))
        self.assertEqual(text, "Noted, turn answered.")
        self.assertIn("x-colibri-compacted-messages", headers)

    def test_4_a_long_last_message_alone_still_says_why(self):
        messages = [{"role": "user", "content": "x " * (CAPACITY * 4)}]
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post({"model": "m", "messages": messages, "max_tokens": 16,
                       "context_compaction": "auto"})
        self.assertEqual(caught.exception.code, 400)
        self.assertEqual(json.loads(caught.exception.read())["error"]["code"],
                         "context_length_exceeded")

    def test_5_bad_value_is_a_400(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post({"model": "m", "messages": conversation(1), "context_compaction": "yes"})
        self.assertEqual(caught.exception.code, 400)

    def test_6_health_says_it_compacts_and_the_context(self):
        with urllib.request.urlopen(self.base[:-len("/v1")] + "/health", timeout=5) as resp:
            health = json.loads(resp.read())
        self.assertIs(health.get("context_compaction"), True)
        self.assertEqual(health.get("context_tokens"), CAPACITY)


if __name__ == "__main__":
    unittest.main()
