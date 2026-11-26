#!/usr/bin/env python3
"""Decode speed of `coli serve` for one or more arms, one fresh server per run.

usage: bench.py MODEL_DIR OUT_PREFIX ARM [ARM ...]
  ARM = name:path/to/checkout/c[:KEY=VALUE[,KEY=VALUE...]]
  KEY NOWAIT=1 is the script's own: with Q36_PRELOAD=1 it sends the requests at once
  where the default waits for the "experts loaded" line.

Per run: start the server, one warm-up request, the three timed requests ("fresh"),
five other requests, the same three again ("warm"). Requests are streamed
/v1/chat/completions at temperature 0, 96 tokens. tok/s = (chunks - 1) / (last - first).
Writes OUT_PREFIX-<n>.log (the server's output) and OUT_PREFIX-results.json.
"""
import json, os, signal, subprocess, sys, time, urllib.request

MODEL, PREFIX = sys.argv[1], sys.argv[2]
PORT = 8011
MAX_TOKENS = 96
CONFIGS = []
for arm in sys.argv[3:]:
    name, path, *rest = arm.split(":", 2)
    env = dict(kv.split("=", 1) for kv in rest[0].split(",")) if rest else {}
    if env.pop("NOWAIT", None): env["_NOWAIT"] = "1"
    CONFIGS.append((name, dict(env, COLI_TIMERS="1", _DIR=path)))
WARM = ["Write a paragraph about mountains. /no_think", "Write a paragraph about oceans. /no_think",
        "Write a paragraph about deserts. /no_think", "Explain how a car engine works. /no_think",
        "Write a short story about a cat. /no_think"]
PROMPTS = [
    "Name three primary colors. /no_think",                       # warm-up
    "Explain in a short paragraph why the sky is blue. /no_think",
    "Write a short paragraph about the history of the bicycle. /no_think",
    "Describe how a refrigerator works in a short paragraph. /no_think",
]


def post_stream(prompt):
    body = json.dumps({"model": "qwen3.6-colibri", "stream": True, "temperature": 0,
                       "max_tokens": MAX_TOKENS,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.time(); stamps = []; text = []
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            ch = d.get("choices") or [{}]
            delta = ch[0].get("delta", {})
            if delta.get("content") or delta.get("reasoning_content"):
                stamps.append(time.time())
                text.append(delta.get("content") or delta.get("reasoning_content"))
    if len(stamps) < 2:
        return None
    ttft = stamps[0] - t0
    decode = (len(stamps) - 1) / (stamps[-1] - stamps[0])
    return {"chunks": len(stamps), "ttft_s": round(ttft, 2), "decode_tok_s": round(decode, 3), "text": "".join(text)}


def wait_ready(proc, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def run(name, extra):
    COLI = extra["_DIR"] + "/coli"
    prefix = extra.get("_PREFIX", "").split(); nowait = "_NOWAIT" in extra
    extra = {k: v for k, v in extra.items() if k not in ("_DIR", "_PREFIX", "_NOWAIT")}
    env = dict(os.environ, **extra)
    log = open(f"{PREFIX}-{len(results)}.log", "w")
    proc = subprocess.Popen(prefix + [COLI, "serve", "--model", MODEL, "--auto-tier", "--port", str(PORT)],
                            env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        if not wait_ready(proc):
            return {"name": name, "env": extra, "error": "server did not start"}
        if extra.get("Q36_PRELOAD") == "1" and not nowait:
            t0 = time.time()
            while time.time() - t0 < 900 and "experts loaded in" not in open(log.name, errors="replace").read():
                time.sleep(2)
            print("preload line:", [l for l in open(log.name, errors="replace").read().replace("\r", "\n").split("\n") if "preload:" in l][-1:], flush=True)
        post_stream(PROMPTS[0])  # warm-up
        fresh = [post_stream(p) for p in PROMPTS[1:]]
        for p in WARM: post_stream(p)
        runs = [post_stream(p) for p in PROMPTS[1:]]
        runs = [r for r in runs if r]
        avg = sum(r["decode_tok_s"] for r in runs) / len(runs) if runs else 0
        return {"name": name, "env": extra, "fresh": fresh, "runs": runs, "avg_decode_tok_s": round(avg, 3)}
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
        log.close()
        time.sleep(3)


results = []
for name, extra in CONFIGS:
    r = run(name, extra)
    results.append(r)
    print(json.dumps(r), flush=True)
json.dump(results, open(f"{PREFIX}-results.json", "w"), indent=2)
print("DONE", flush=True)
