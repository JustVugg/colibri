#!/usr/bin/env python3
"""C23 eval driver — runs the committed question set (c23_eval_set.py) against
one container and scores it mechanically.

Usage:
  python3 c23_eval.py <tag> [--smoke] [--base URL] [--key KEY] [--model ID]

  tag          output file suffix -> phaseC23_eval_<tag>.jsonl
  --smoke      1 code + 1 math + 1 format task only (harness validation)

Scoring (no human judging):
  code    extract the model's fenced python block, append the task's asserts,
          run `python3 -I -c` in a subprocess with a timeout; score 1 iff it
          prints ALL_OK.
  math    last 'Answer: <integer>' line, fallback last standalone integer;
          exact match against the committed answer.
  format  per-task mechanical checker (JSON schema/sort, bullet count+cap,
          exact tool-call object, computed hex, sorted string).
  recall  required-keyword presence (the WEAK scorer; rows carry weak=1).
  longctx the committed ledger doc + 4 questions over ONE appended
          conversation: each turn appends the engine's ACTUAL previous
          assistant text (C13 replay rule), so block reuse is the real
          engine path. reused_prompt_tokens is recorded; questions 2-4 are
          expected to reuse the ledger prefill.

Every row records the response sha12, full text, score, and the engine's
timings object, so a run doubles as a performance read.

Greedy sampling (temperature 0) throughout — same bytes in, deterministic
scores out; cross-container deltas are the model, not the harness.

Token budgets (iq2_val1 finding): the model's LaTeX step-by-step math runs
~600+ tokens and its longctx enumeration ~800+, so math and longctx get
1024 (400/256 truncated mid-derivation and mis-scored real reasoning as
failures); code stays 512 (all answers complete well inside), format 256,
recall 200.
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from c23_eval_set import CODE_TASKS, MATH_TASKS, FORMAT_TASKS, RECALL_TASKS, LONGCTX_QUESTIONS  # noqa: E402

DOC_PATH = HERE / "c23_eval_doc.txt"


def send(base, key, model, messages, max_tokens):
    body = {"model": model, "messages": messages,
            "max_tokens": max_tokens, "temperature": 0, "stream": False}
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=14400) as resp:
        payload = json.loads(resp.read())
    text = payload["choices"][0]["message"]["content"]
    t = payload.get("timings", {})
    return text, t


def fenced_code(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    return blocks[-1] if blocks else None


def run_code(task_id, block):
    """Run model block + asserts in a clean interpreter. 1 iff ALL_OK."""
    if not block:
        return 0, "no fenced python block"
    program = block + "\n" + dict((t[0], t[3]) for t in CODE_TASKS)[task_id]
    try:
        r = subprocess.run(["python3", "-I", "-c", program],
                           capture_output=True, text=True, timeout=20)
    except subprocess.TimeoutExpired:
        return 0, "timeout"
    if r.returncode == 0 and "ALL_OK" in r.stdout:
        return 1, "ALL_OK"
    err = (r.stderr.strip().splitlines() or ["?"])[-1][:200]
    return 0, err


def math_answer(text, want):
    m = re.findall(r"Answer:\s*(-?\d+)", text)
    got = m[-1] if m else None
    if got is None:
        nums = re.findall(r"(?<![\w.])(-?\d+)(?![\w.])", text)
        got = nums[-1] if nums else None
    return (1 if got is not None and int(got) == want else 0), f"got={got}"


def check_format(task_id, text):
    body = text.strip()
    if task_id == "json_cities":
        try:
            o = json.loads(body)
        except Exception:
            return 0, "not json"
        if set(o) != {"cities"} or not isinstance(o["cities"], list) or len(o["cities"]) != 4:
            return 0, f"shape: {json.dumps(o)[:120]}"
        want = [("Vesterby", 63450), ("Kaldnes", 56300), ("Holmstrand", 55123), ("Brannvik", 46320)]
        got = [(c.get("name"), c.get("population")) for c in o["cities"]]
        return (1 if got == want else 0), f"got={got}"
    if task_id == "bullets":
        lines = [ln for ln in body.splitlines() if ln.strip()]
        if len(lines) != 3 or not all(ln.startswith("- ") for ln in lines):
            return 0, f"lines={len(lines)}"
        wc = [len(ln[2:].split()) for ln in lines]
        return (1 if max(wc) <= 12 else 0), f"words={wc}"
    if task_id == "toolcall":
        try:
            o = json.loads(body)
        except Exception:
            return 0, "not json"
        ok = (set(o) == {"tool", "args"} and o["tool"] == "web_search"
              and isinstance(o["args"], dict) and set(o["args"]) == {"query", "limit"}
              and o["args"]["query"] == "ARM NEON dot product instruction"
              and o["args"]["limit"] == 5)
        return (1 if ok else 0), f"got={json.dumps(o)[:120]}"
    if task_id == "hexmix":
        got = re.search(r"#[0-9a-fA-F]{6}", body)
        ok = got is not None and got.group(0).lower() == "#7e3647"
        return (1 if ok else 0), f"got={got.group(0) if got else None}"
    if task_id == "sortlist":
        norm = re.sub(r"\s+", "", body).strip().lower().rstrip(".")
        ok = norm == "apple,kiwi,mango,pear,ugli"
        return (1 if ok else 0), f"got={norm[:60]}"
    return 0, "unknown format task"


def check_recall(text, keywords):
    low = text.lower()
    present = [k for k in keywords if k.lower() in low]
    missing = [k for k in keywords if k.lower() not in low]
    return (1 if not missing else 0), f"missing={missing}"


def longctx_answer(text, want, kind):
    if kind == "int":
        m = re.findall(r"Answer:\s*(-?\d+)", text)
        got = m[-1] if m else None
        if got is None:
            nums = re.findall(r"(?<![\w.])(\d{3,})(?![\w.])", text)
            got = nums[-1] if nums else None
        ok = got is not None and int(got) == want
    else:  # id
        m = re.findall(r"Answer:\s*(SHP-\d+)", text)
        got = m[-1] if m else (re.findall(r"SHP-\d+", text) or [None])[-1]
        ok = got == want
    return (1 if ok else 0), f"got={got}"


def row(cat, task, text, score, detail, t):
    return {"cat": cat, "task": task,
            "sha12": hashlib.sha256(text.encode()).hexdigest()[:12],
            "score": score, "detail": detail, "text": text,
            "prompt_tokens": t.get("prompt_tokens"),
            "reused_prompt_tokens": t.get("reused_prompt_tokens"),
            "prefill_s": t.get("prefill_s"),
            "engine_ttft_s": t.get("engine_ttft_s"),
            "decode_tokens_per_sec": t.get("decode_tokens_per_sec")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--base", default="http://localhost:8001/v1")
    ap.add_argument("--key", default="")
    ap.add_argument("--model", default="qwen3.6-colibri")
    args = ap.parse_args()

    out_path = HERE / f"phaseC23_eval_{args.tag}.jsonl"
    rows = []

    def emit(r):
        rows.append(r)
        with open(out_path, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(json.dumps({k: r[k] for k in
                          ("cat", "task", "score", "detail", "sha12",
                           "prompt_tokens", "reused_prompt_tokens",
                           "decode_tokens_per_sec")}), flush=True)

    code_tasks = CODE_TASKS[:1] if args.smoke else CODE_TASKS
    math_tasks = MATH_TASKS[:1] if args.smoke else MATH_TASKS
    fmt_tasks = FORMAT_TASKS[:1] if args.smoke else FORMAT_TASKS
    recall_tasks = [] if args.smoke else RECALL_TASKS

    for tid, mt, prompt, tests in code_tasks:
        text, t = send(args.base, args.key, args.model,
                       [{"role": "user", "content": prompt}], mt)
        score, detail = run_code(tid, fenced_code(text))
        emit(row("code", tid, text, score, detail, t))

    for tid, mt, prompt, want in math_tasks:
        text, t = send(args.base, args.key, args.model,
                       [{"role": "user", "content": prompt}], mt)
        score, detail = math_answer(text, want)
        emit(row("math", tid, text, score, detail, t))

    for tid, mt, prompt, kind in fmt_tasks:
        text, t = send(args.base, args.key, args.model,
                       [{"role": "user", "content": prompt}], mt)
        score, detail = check_format(tid, text)
        emit(row("format", tid, text, score, detail, t))

    for tid, mt, prompt, keywords in recall_tasks:
        text, t = send(args.base, args.key, args.model,
                       [{"role": "user", "content": prompt}], mt)
        score, detail = check_recall(text, keywords)
        emit(row("recall", tid, text, score, detail, t))

    if not args.smoke:
        doc = DOC_PATH.read_text()
        convo = []
        for qid, q, want, kind in LONGCTX_QUESTIONS:
            if not convo:
                user = (doc + "\n\n---\n\n" + q)
                convo = [{"role": "user", "content": user}]
            else:
                convo.append({"role": "assistant", "content": last_text})
                convo.append({"role": "user", "content": q})
            text, t = send(args.base, args.key, args.model, convo, 1024)
            last_text = text
            score, detail = longctx_answer(text, want, kind)
            r = row("longctx", qid, text, score, detail, t)
            emit(r)

    # summary
    n_scored = [r["score"] for r in rows]
    bycat = {}
    for r in rows:
        c = bycat.setdefault(r["cat"], [0, 0])
        c[0] += r["score"]
        c[1] += 1
    print("\n== C23 summary (%s) ==" % args.tag)
    for cat, (s, n) in bycat.items():
        print(f"  {cat:8s} {s}/{n}")
    print(f"  TOTAL    {sum(n_scored)}/{len(n_scored)}")
    print(f"rows -> {out_path.name}")


if __name__ == "__main__":
    main()
