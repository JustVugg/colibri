#!/usr/bin/env python3
"""A model's start profile: the expert history and the MTP head's draft counts of a fixed set of
ordinary prompts, written beside the model as .coli_usage.start and .coli_vocab.start.

An engine that finds no history of its own starts from them (route_trace.h's rt_load, every
engine with an expert history; qwen38's q38_vocab_init for the draft counts), so the experts its
first answers route to are on the device from the start, not the ones nobody's history put there. The history the engine writes afterwards goes to
.coli_usage and .coli_vocab as before. The profile run itself starts from no history.

usage: tools/usage_start_profile.py --model DIR [--coli ./coli] [--port 18490] [--tokens 256]
       [--out DIR] [ENV=VALUE ...]   (ENV=VALUE: the server's environment, e.g. COLI_CUDA=1)
"""
import argparse
import http.client
import json
import os
import shutil
import signal
import subprocess
import sys
import time

# forty requests across languages, prose, code and reasoning, a quarter of them thinking
PROMPTS = [
    "Explain how a refrigerator works to a ten year old.",
    "Write a short poem about the sea at night.",
    "What are the main causes of inflation? Give a balanced answer.",
    "Write a Python function that checks whether a string is a palindrome, with tests.",
    "Riassumi la storia dell Impero Romano in dieci punti.",
    "Scrivi una lettera formale per chiedere un rimborso a una compagnia aerea.",
    "Explique la photosynthese de facon simple.",
    "Erklaere den Unterschied zwischen Wetter und Klima.",
    "Escribe una receta de tortilla espanola paso a paso.",
    "Write a C program that reads numbers from stdin and prints their average.",
    "Plan a three day trip to Kyoto with a budget.",
    "Solve: a train leaves at 9:15 and travels 240 km at 80 km/h. When does it arrive? Show the steps.",
    "Write a tense dialogue between two detectives who disagree about a suspect.",
    "Give me ten ideas for a birthday party for a seven year old.",
    "What is the difference between TCP and UDP? Use examples.",
    "Write a JavaScript function that debounces another function, and explain it.",
    "Describe the life cycle of a star.",
    "Come si prepara il risotto alla milanese?",
    "Write a product description for a waterproof hiking backpack.",
    "Explain recursion with an example in Python.",
    "Write a bedtime story about a dragon who is afraid of the dark.",
    "List the pros and cons of working from home.",
    "Translate into Italian: The meeting has been moved to Thursday afternoon because the manager is travelling.",
    "Write a SQL query that finds the five customers with the highest total orders, and explain it.",
    "What happened during the French Revolution? Keep it concise.",
    "Write a haiku sequence about autumn in a city.",
    "How do vaccines train the immune system?",
    "Write a Rust function that reverses a linked list.",
    "Give advice to someone starting to learn the guitar.",
    "Scrivi un racconto breve ambientato in un piccolo paese di montagna.",
    "Explain what a hash table is and how collisions are handled.",
    "Write an email to a colleague asking for feedback on a report.",
    "What are black holes made of?",
    "Write a limerick about a cat who loves coffee.",
    "Describe how to change a flat tire safely.",
    "Write a bash script that backs up a directory into a dated tar.gz file.",
    "Compare capitalism and socialism in a neutral way.",
    "Write the opening chapter of a science fiction novel set on Mars.",
    "Why is the sky blue?",
    "Write unit tests in Go for a function that adds two integers.",
]


def request(port, method, path, body=None, timeout=3600):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    c.request(method, path, json.dumps(body) if body is not None else None,
              {"Content-Type": "application/json"} if body is not None else {})
    r = c.getresponse()
    return r.status, r.read()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="the model's directory (the profile goes beside it)")
    ap.add_argument("--coli", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "coli"))
    ap.add_argument("--port", type=int, default=18490)
    ap.add_argument("--tokens", type=int, default=256, help="the most tokens an answer takes")
    ap.add_argument("--out", help="where the two files go (default: the model's directory)")
    ap.add_argument("env", nargs="*", help="ENV=VALUE for the server")
    a = ap.parse_args()
    out = a.out or a.model
    usage, vocab = os.path.join(out, ".coli_usage.start"), os.path.join(out, ".coli_vocab.start")
    tmp_u, tmp_v = usage + ".tmp", vocab + ".tmp"
    for p in (tmp_u, tmp_v):
        if os.path.exists(p):
            os.remove(p)
    env = dict(os.environ, COLI_USAGE=tmp_u, COLI_VOCAB_USAGE=tmp_v, USAGE_SAVE="1",
               COLI_USAGE_START=os.devnull, COLI_VOCAB_START=os.devnull)   # the run starts from no history
    for kv in a.env:
        k, _, v = kv.partition("=")
        env[k] = v
    srv = subprocess.Popen([sys.executable, a.coli, "serve", "--model", a.model, "--host", "127.0.0.1", "--port", str(a.port)],
                           env=env, start_new_session=True)
    try:
        for _ in range(600):
            try:
                if request(a.port, "GET", "/health", timeout=2)[0] == 200:
                    break
            except OSError:
                pass
            if srv.poll() is not None:
                sys.exit("the server ended before it was ready")
            time.sleep(1)
        model = json.loads(request(a.port, "GET", "/v1/models")[1])["data"][0]["id"]
        total = 0
        for i, p in enumerate(PROMPTS):
            body = {"model": model, "max_tokens": a.tokens, "temperature": 0.7, "seed": i, "enable_thinking": i % 4 == 0,
                    "messages": [{"role": "user", "content": p}]}
            status, raw = request(a.port, "POST", "/v1/chat/completions", body)
            if status != 200:
                sys.exit(f"request {i}: HTTP {status}")
            total += json.loads(raw).get("usage", {}).get("total_tokens", 0)
            print(f"{i + 1}/{len(PROMPTS)} requests, {total} tokens", flush=True)
    finally:
        if srv.poll() is None:
            os.killpg(srv.pid, signal.SIGTERM)   # the engine writes its history as it stops
            try:
                srv.wait(timeout=300)
            except subprocess.TimeoutExpired:
                os.killpg(srv.pid, signal.SIGKILL)
                srv.wait()
    done = []
    for tmp, final in ((tmp_u, usage), (tmp_v, vocab)):
        if os.path.exists(tmp) and os.path.getsize(tmp) > 0:
            shutil.move(tmp, final)
            done.append(final)
    if not done:
        sys.exit("the server wrote no history (USAGE_SAVE, or an engine without one)")
    print("start profile: " + ", ".join(done))


if __name__ == "__main__":
    main()
