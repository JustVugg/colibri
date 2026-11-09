"""Head-to-head benchmark: agent reference solutions vs Colibri 30B, plus an
Intel-NPU embedding retrieval test. Results -> results/bench_agent_vs_colibri.json.

Fairness: the agent ("me") supplies hand-written reference solutions that are
scored by the *same* execution harness (scripts/evaluate.py exec_score) used for
Colibri. The agent's own wall-clock latency is not measurable through this
interface, so only correctness is compared head-to-head; Colibri latency and
token throughput are reported separately as measured facts.

Run: .venv/Scripts/python.exe scripts/bench_agent_vs_colibri.py
Requires Colibri serve on :8181 and the NPU embedding server on :8182.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import extract_code, exec_score  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS, exist_ok=True)

COLIBRI = "http://127.0.0.1:8181/v1"
COLIBRI_MODEL = "qwen3-coder-30b-a3b-colibri"
EMBED = "http://127.0.0.1:8182/v1"
EMBED_MODEL = "bge-base-en-v1.5-int8-npu"

# Each task: instruction, target function, execution tests, and an agent-authored
# reference solution (the "me" side of the comparison). Difficulty tiers let the
# graphs show where the local model holds up vs where it struggles.
TASKS = [
    {
        "id": "sum_even", "difficulty": "easy", "func": "sum_even",
        "instruction": "Write a Python function `sum_even(lst)` that sums the even numbers in a list. Answer with code only.\n\n",
        "tests": "assert sum_even([1,2,3,4]) == 6\nassert sum_even([]) == 0\nassert sum_even([1,3,5]) == 0\nassert sum_even([-2,-4,7]) == -6\nassert sum_even([2]) == 2",
        "agent_ref": "def sum_even(lst):\n    return sum(x for x in lst if x % 2 == 0)\n",
    },
    {
        "id": "factorial", "difficulty": "easy", "func": "factorial",
        "instruction": "Write a Python function `factorial(n)` that returns n! for non-negative integers (factorial(0) == 1). Answer with code only.\n\n",
        "tests": "assert factorial(0) == 1\nassert factorial(1) == 1\nassert factorial(5) == 120\nassert factorial(10) == 3628800",
        "agent_ref": "def factorial(n):\n    r = 1\n    for i in range(2, n + 1):\n        r *= i\n    return r\n",
    },
    {
        "id": "reverse_words", "difficulty": "easy", "func": "reverse_words",
        "instruction": "Write a Python function `reverse_words(s)` that reverses the order of words in a string (words separated by single spaces). Answer with code only.\n\n",
        "tests": "assert reverse_words('hello world') == 'world hello'\nassert reverse_words('one') == 'one'\nassert reverse_words('a b c') == 'c b a'\nassert reverse_words('') == ''",
        "agent_ref": "def reverse_words(s):\n    return ' '.join(s.split()[::-1])\n",
    },
    {
        "id": "count_vowels", "difficulty": "easy", "func": "count_vowels",
        "instruction": "Write a Python function `count_vowels(s)` that counts vowels (a,e,i,o,u, case-insensitive) in a string. Answer with code only.\n\n",
        "tests": "assert count_vowels('hello') == 2\nassert count_vowels('AEIOU') == 5\nassert count_vowels('xyz') == 0\nassert count_vowels('') == 0",
        "agent_ref": "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')\n",
    },
    {
        "id": "fib", "difficulty": "medium", "func": "fib",
        "instruction": "Write a Python function `fib(n)` that returns a list of the first n Fibonacci numbers (starting 0, 1). Answer with code only.\n\n",
        "tests": "assert fib(0) == []\nassert fib(1) == [0]\nassert fib(2) == [0,1]\nassert fib(7) == [0,1,1,2,3,5,8]",
        "agent_ref": "def fib(n):\n    out = []\n    a, b = 0, 1\n    for _ in range(n):\n        out.append(a)\n        a, b = b, a + b\n    return out\n",
    },
    {
        "id": "flatten", "difficulty": "medium", "func": "flatten",
        "instruction": "Write a Python function `flatten(lst)` that flattens an arbitrarily nested list of integers into a flat list, preserving order. Answer with code only.\n\n",
        "tests": "assert flatten([1,[2,3],4]) == [1,2,3,4]\nassert flatten([1,[2,[3,[4]]]]) == [1,2,3,4]\nassert flatten([]) == []\nassert flatten([[],[[]]]) == []\nassert flatten([1,2]) == [1,2]",
        "agent_ref": "def flatten(lst):\n    out = []\n    for x in lst:\n        if isinstance(x, list):\n            out.extend(flatten(x))\n        else:\n            out.append(x)\n    return out\n",
    },
    {
        "id": "binary_search", "difficulty": "medium", "func": "binary_search",
        "instruction": "Write a Python function `binary_search(arr, target)` that returns the index of target in a sorted list, or -1 if absent. Answer with code only.\n\n",
        "tests": "assert binary_search([1,2,3,4,5], 3) == 2\nassert binary_search([1,2,3,4,5], 6) == -1\nassert binary_search([], 1) == -1\nassert binary_search([5], 5) == 0",
        "agent_ref": "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        if arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n",
    },
    {
        "id": "second_largest", "difficulty": "medium", "func": "second_largest",
        "instruction": "Write a Python function `second_largest(lst)` that returns the second largest distinct value in a list, or None if fewer than two distinct values exist. Answer with code only.\n\n",
        "tests": "assert second_largest([1,2,3]) == 2\nassert second_largest([5,5,4]) == 4\nassert second_largest([1]) is None\nassert second_largest([2,2,2]) is None",
        "agent_ref": "def second_largest(lst):\n    vals = sorted(set(lst), reverse=True)\n    return vals[1] if len(vals) >= 2 else None\n",
    },
    {
        "id": "merge_intervals", "difficulty": "hard", "func": "merge_intervals",
        "instruction": "Write a Python function `merge_intervals(intervals)` that merges all overlapping intervals (each a [start, end] list) and returns them sorted by start. Answer with code only.\n\n",
        "tests": "assert merge_intervals([[1,3],[2,6],[8,10]]) == [[1,6],[8,10]]\nassert merge_intervals([[1,4],[4,5]]) == [[1,5]]\nassert merge_intervals([]) == []\nassert merge_intervals([[1,4],[0,2]]) == [[0,4]]",
        "agent_ref": "def merge_intervals(intervals):\n    if not intervals:\n        return []\n    iv = sorted(intervals, key=lambda x: x[0])\n    out = [list(iv[0])]\n    for s, e in iv[1:]:\n        if s <= out[-1][1]:\n            out[-1][1] = max(out[-1][1], e)\n        else:\n            out.append([s, e])\n    return out\n",
    },
    {
        "id": "valid_parentheses", "difficulty": "hard", "func": "valid_parentheses",
        "instruction": "Write a Python function `valid_parentheses(s)` that returns True if every bracket ()[]{} in the string is correctly matched and nested. Answer with code only.\n\n",
        "tests": "assert valid_parentheses('()[]{}') is True\nassert valid_parentheses('(]') is False\nassert valid_parentheses('([)]') is False\nassert valid_parentheses('{[]}') is True\nassert valid_parentheses('') is True",
        "agent_ref": "def valid_parentheses(s):\n    pairs = {')': '(', ']': '[', '}': '{'}\n    stack = []\n    for c in s:\n        if c in pairs.values():\n            stack.append(c)\n        elif c in pairs:\n            if not stack or stack.pop() != pairs[c]:\n                return False\n    return not stack\n",
    },
]


def post(url, body, timeout=600):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def colibri_generate(instruction, max_tokens=400):
    t0 = time.time()
    try:
        d = post(COLIBRI + "/completions", {
            "model": COLIBRI_MODEL, "prompt": instruction,
            "max_tokens": max_tokens, "temperature": 0})
        dt = time.time() - t0
        text = d["choices"][0]["text"]
        usage = d.get("usage", {})
        ctok = usage.get("completion_tokens", 0)
        return text, dt, ctok, (ctok / dt if dt > 0 else 0.0), None
    except Exception as e:
        return "", time.time() - t0, 0, 0.0, str(e)


def run_coding_bench():
    rows = []
    for t in TASKS:
        # Agent reference (the "me" side), scored by the identical harness.
        agent_sc = exec_score(t["agent_ref"], t)
        # Colibri 30B.
        out, dt, ctok, tps, err = colibri_generate(t["instruction"])
        col_sc = exec_score(out, t)
        rows.append({
            "id": t["id"], "difficulty": t["difficulty"],
            "agent": {"exec_passed": agent_sc["exec_passed"],
                      "exec_error": agent_sc.get("exec_error"),
                      "code": t["agent_ref"]},
            "colibri": {"exec_passed": col_sc["exec_passed"],
                        "exec_error": col_sc.get("exec_error"),
                        "time_s": round(dt, 3), "completion_tokens": ctok,
                        "tokens_per_s": round(tps, 2), "output": out, "error": err},
        })
        print(f"[{t['difficulty']:6}] {t['id']:18} "
              f"agent={'PASS' if agent_sc['exec_passed'] else 'FAIL'} "
              f"colibri={'PASS' if col_sc['exec_passed'] else 'FAIL'} "
              f"{dt:.1f}s {tps:.1f}tok/s", flush=True)
    return rows


def summarize(rows, side):
    n = len(rows)
    passed = sum(1 for r in rows if r[side]["exec_passed"])
    by_diff = {}
    for d in ("easy", "medium", "hard"):
        sub = [r for r in rows if r["difficulty"] == d]
        by_diff[d] = {"total": len(sub),
                      "passed": sum(1 for r in sub if r[side]["exec_passed"])}
    return {"total": n, "passed": passed, "pass_at_1": round(passed / n, 4),
            "by_difficulty": by_diff}


# --- NPU embedding retrieval test -------------------------------------------
DOCS = [
    ("python_lang", "Python is a high-level programming language used for scripting and data science."),
    ("python_loop", "You can write a for loop in Python to iterate over the items of a list."),
    ("cooking", "To bake bread, mix flour, water, yeast and salt, then let the dough rise before baking."),
    ("sports", "The football team scored a winning goal in the final minute of the match."),
    ("space", "The rocket launched into orbit and deployed the satellite around Earth."),
    ("finance", "The stock market index fell sharply after the central bank raised interest rates."),
    ("medicine", "The doctor prescribed antibiotics to treat the bacterial infection."),
    ("music", "She played a beautiful melody on the piano during the evening concert."),
    ("travel", "We booked a flight to Paris and reserved a hotel near the Eiffel Tower."),
    ("gardening", "Plant tomatoes in well-drained soil and water them regularly during summer."),
    ("history", "The ancient Roman empire built roads and aqueducts across Europe."),
    ("weather", "A heavy thunderstorm with lightning and strong wind is expected tomorrow afternoon."),
]
QUERIES = [
    ("How do I loop over a list in Python?", "python_loop"),
    ("What language is good for data science?", "python_lang"),
    ("recipe for making bread at home", "cooking"),
    ("soccer match winning goal", "sports"),
    ("satellite orbit rocket launch", "space"),
    ("interest rates and stock prices", "finance"),
    ("treatment for a bacterial infection", "medicine"),
    ("piano concert melody", "music"),
]


def embed_one(text):
    d = post(EMBED + "/embeddings", {"model": EMBED_MODEL, "input": text})
    return np.asarray(d["data"][0]["embedding"], dtype=np.float32)


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def run_embedding_bench():
    t0 = time.time()
    doc_vecs = []
    for _, text in DOCS:
        doc_vecs.append(embed_one(text))
    doc_mat = np.stack(doc_vecs)
    doc_norm = doc_mat / np.linalg.norm(doc_mat, axis=1, keepdims=True)
    doc_time = time.time() - t0

    hits, rr = 0, 0.0
    detail = []
    qt0 = time.time()
    for q, target in QUERIES:
        qv = embed_one(q)
        sims = doc_norm @ (qv / np.linalg.norm(qv))
        order = np.argsort(-sims)
        ids = [DOCS[i][0] for i in order]
        rank = ids.index(target) + 1
        hits += 1 if rank == 1 else 0
        rr += 1.0 / rank
        detail.append({"query": q, "target": target, "rank": rank,
                       "top1": ids[0], "top1_score": round(float(sims[order[0]]), 4)})
    query_time = time.time() - qt0
    nq = len(QUERIES)
    total_emb = len(DOCS) + nq
    total_time = doc_time + query_time
    return {
        "n_docs": len(DOCS), "n_queries": nq, "dim": int(doc_mat.shape[1]),
        "accuracy_at_1": round(hits / nq, 4),
        "mrr": round(rr / nq, 4),
        "detail": detail,
        "doc_embed_s": round(doc_time, 3),
        "query_embed_s": round(query_time, 3),
        "throughput_emb_per_s": round(total_emb / total_time, 2) if total_time > 0 else 0,
        "avg_query_latency_s": round(query_time / nq, 3),
    }


def main():
    print("=== Coding: agent reference vs Colibri 30B ===", flush=True)
    rows = run_coding_bench()
    agent_sum = summarize(rows, "agent")
    col_sum = summarize(rows, "colibri")
    col_times = [r["colibri"]["time_s"] for r in rows]
    col_tps = [r["colibri"]["tokens_per_s"] for r in rows if r["colibri"]["tokens_per_s"] > 0]
    colibri_perf = {
        "total_time_s": round(sum(col_times), 3),
        "avg_latency_s": round(sum(col_times) / len(col_times), 3),
        "avg_tokens_per_s": round(sum(col_tps) / len(col_tps), 2) if col_tps else 0,
        "total_completion_tokens": sum(r["colibri"]["completion_tokens"] for r in rows),
    }

    print("\n=== NPU embedding retrieval ===", flush=True)
    emb = run_embedding_bench()
    print(f"accuracy@1={emb['accuracy_at_1']} MRR={emb['mrr']} "
          f"throughput={emb['throughput_emb_per_s']} emb/s "
          f"avg_query={emb['avg_query_latency_s']}s", flush=True)

    # Pull prior student-3B and NPU/CPU-cosine facts for the comparison charts.
    prior = {}
    for key, fn in (("student_3b", "eval_student_base.json"),
                    ("npu_cosine", "npu_embed_verify.json")):
        p = os.path.join(RESULTS, fn)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                prior[key] = json.load(f)

    report = {
        "coding": {"tasks": rows, "agent_summary": agent_sum,
                   "colibri_summary": col_sum, "colibri_perf": colibri_perf},
        "embedding_npu": emb,
        "prior": {
            "student_3b_summary": prior.get("student_3b", {}).get("summary"),
            "npu_min_cosine_vs_cpu": prior.get("npu_cosine", {}).get("npu", {}).get("min_cosine_vs_cpu"),
            "npu_semantics": prior.get("npu_cosine", {}).get("npu_semantics"),
        },
    }
    out = os.path.join(RESULTS, "bench_agent_vs_colibri.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("\nwrote", out, flush=True)
    print(json.dumps({"agent": agent_sum, "colibri": col_sum,
                      "colibri_perf": colibri_perf}, indent=2), flush=True)


if __name__ == "__main__":
    main()