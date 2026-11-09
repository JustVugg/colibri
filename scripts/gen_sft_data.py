"""Generate execution-verified coding SFT data from the Colibri teacher.

Each template has a reference implementation and randomized tests. The teacher
(Colibri OpenAI API) writes a solution; we keep it only if it passes every test
in a sandboxed subprocess. Output: results/sft_verified.jsonl (chat format).

The eval functions (sum_even/fib/palindrome/factorial/flatten) are deliberately
NOT templates here, so the fine-tune is not trained on its own benchmark.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "results" / "sft_verified.jsonl"

# name -> (signature, spec, reference source, test-source factory(rng)->str)
TEMPLATES = {
    "gcd": (
        "gcd(a, b)", "the greatest common divisor of two non-negative integers (gcd(0,0)=0)",
        "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n",
        lambda r: "\n".join(f"assert gcd({a},{b}) == {__import__('math').gcd(a,b)}"
                            for a, b in [(r.randint(0, 500), r.randint(0, 500)) for _ in range(6)]
                            + [(0, 0), (0, 7), (13, 13)]),
    ),
    "is_prime": (
        "is_prime(n)", "True if n is a prime number, False otherwise (n >= 0)",
        "def is_prime(n):\n    if n < 2:\n        return False\n    i = 2\n    while i * i <= n:\n        if n % i == 0:\n            return False\n        i += 1\n    return True\n",
        lambda r: "\n".join(f"assert is_prime({n}) == {str(n in (2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97))}"
                            for n in [0, 1, 2, 4, 17, 25, 97, 100] + [r.randint(3, 200) for _ in range(4)]),
    ),
    "reverse_words": (
        "reverse_words(s)", "the string with the order of whitespace-separated words reversed, joined by single spaces",
        "def reverse_words(s):\n    return ' '.join(s.split()[::-1])\n",
        lambda r: "\n".join(f"assert reverse_words({s!r}) == {' '.join(s.split()[::-1])!r}"
                            for s in ["hello world", "  a  b   c ", "", "one", "x y z w"]),
    ),
    "count_vowels": (
        "count_vowels(s)", "the number of vowels (aeiou, case-insensitive) in the string",
        "def count_vowels(s):\n    return sum(1 for ch in s.lower() if ch in 'aeiou')\n",
        lambda r: "\n".join(f"assert count_vowels({s!r}) == {sum(1 for c in s.lower() if c in 'aeiou')}"
                            for s in ["Hello World", "AEIOUaeiou", "xyz", "", "Programming"]),
    ),
    "second_largest": (
        "second_largest(lst)", "the second largest distinct value in the list; return None if there is none",
        "def second_largest(lst):\n    u = sorted(set(lst), reverse=True)\n    return u[1] if len(u) > 1 else None\n",
        lambda r: "\n".join(f"assert second_largest({l!r}) == {sorted(set(l), reverse=True)[1] if len(set(l)) > 1 else None!r}"
                            for l in [[3, 1, 2], [5, 5, 5], [], [1], [9, 9, 2, 7], [-1, -5, -3]]),
    ),
    "merge_dicts": (
        "merge_dicts(a, b)", "a new dict with keys of both; for shared keys the values are summed (values are ints)",
        "def merge_dicts(a, b):\n    out = dict(a)\n    for k, v in b.items():\n        out[k] = out.get(k, 0) + v\n    return out\n",
        lambda r: "\n".join(f"assert merge_dicts({a!r},{b!r}) == { {k: a.get(k,0)+b.get(k,0) for k in set(a)|set(b)}!r}"
                            for a, b in [({"x": 1}, {"y": 2}), ({"x": 1}, {"x": 4}), ({}, {}), ({"a": 3, "b": 1}, {"b": 2, "c": 5})]),
    ),
    "binary_search": (
        "binary_search(lst, x)", "the index of x in the sorted list, or -1 if absent",
        "def binary_search(lst, x):\n    lo, hi = 0, len(lst) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if lst[mid] == x:\n            return mid\n        if lst[mid] < x:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n",
        lambda r: "\n".join(f"assert binary_search({l!r},{x}) == {l.index(x) if x in l else -1}"
                            for l, x in [([1, 3, 5, 7], 5), ([1, 3, 5, 7], 4), ([], 1), ([2], 2), ([2], 3), (list(range(0, 100, 2)), 42)]),
    ),
    "caesar_cipher": (
        "caesar_cipher(s, k)", "the string with each letter shifted k positions forward in the alphabet (wrap around, preserve case, leave non-letters unchanged)",
        "def caesar_cipher(s, k):\n    out = []\n    for ch in s:\n        if ch.isalpha():\n            base = ord('A') if ch.isupper() else ord('a')\n            out.append(chr((ord(ch) - base + k) % 26 + base))\n        else:\n            out.append(ch)\n    return ''.join(out)\n",
        lambda r: "\n".join(f"assert caesar_cipher({s!r},{k}) == {''.join(chr((ord(c)-(65 if c.isupper() else 97)+k)%26+(65 if c.isupper() else 97)) if c.isalpha() else c for c in s)!r}"
                            for s, k in [("Hello, World!", 3), ("abcXYZ", 26), ("abc", -1), ("", 5), ("Zebra-9", 1)]),
    ),
    "running_average": (
        "running_average(lst)", "a list where element i is the average of lst[:i+1] (floats)",
        "def running_average(lst):\n    out, total = [], 0.0\n    for i, x in enumerate(lst):\n        total += x\n        out.append(total / (i + 1))\n    return out\n",
        lambda r: "\n".join(f"assert running_average({l!r}) == {[sum(l[:i+1])/(i+1) for i in range(len(l))]!r}"
                            for l in [[1, 2, 3, 4], [], [5], [1.5, 2.5], [10, -10, 10]]),
    ),
    "transpose": (
        "transpose(m)", "the transpose of a non-empty rectangular matrix (list of lists)",
        "def transpose(m):\n    return [list(row) for row in zip(*m)]\n",
        lambda r: "\n".join(f"assert transpose({m!r}) == {[list(row) for row in zip(*m)]!r}"
                            for m in [[[1, 2], [3, 4]], [[1, 2, 3]], [[1], [2], [3]], [[7]]]),
    ),
    "chunk_list": (
        "chunk_list(lst, n)", "the list split into consecutive chunks of size n (last chunk may be shorter); n >= 1",
        "def chunk_list(lst, n):\n    return [lst[i:i+n] for i in range(0, len(lst), n)]\n",
        lambda r: "\n".join(f"assert chunk_list({l!r},{n}) == {[l[i:i+n] for i in range(0,len(l),n)]!r}"
                            for l, n in [([1, 2, 3, 4, 5], 2), ([], 3), ([1], 1), ([1, 2, 3, 4], 4), ([1, 2, 3], 5)]),
    ),
    "digit_sum": (
        "digit_sum(n)", "the sum of the decimal digits of a non-negative integer",
        "def digit_sum(n):\n    return sum(int(d) for d in str(n))\n",
        lambda r: "\n".join(f"assert digit_sum({n}) == {sum(int(d) for d in str(n))}"
                            for n in [0, 7, 123, 99999] + [r.randint(0, 10**6) for _ in range(4)]),
    ),
    "common_elements": (
        "common_elements(a, b)", "a sorted list of the distinct values present in both lists",
        "def common_elements(a, b):\n    return sorted(set(a) & set(b))\n",
        lambda r: "\n".join(f"assert common_elements({a!r},{b!r}) == {sorted(set(a)&set(b))!r}"
                            for a, b in [([1, 2, 2, 3], [2, 3, 4]), ([], [1]), ([1, 1], [1]), ([5, 1], [1, 5])]),
    ),
    "is_anagram": (
        "is_anagram(a, b)", "True if the two strings are anagrams ignoring spaces and case",
        "def is_anagram(a, b):\n    return sorted(a.replace(' ', '').lower()) == sorted(b.replace(' ', '').lower())\n",
        lambda r: "\n".join(f"assert is_anagram({a!r},{b!r}) == {sorted(a.replace(' ','').lower()) == sorted(b.replace(' ','').lower())}"
                            for a, b in [("listen", "silent"), ("Hello", "olelh"), ("a b c", "cab"), ("", ""), ("aa", "a")]),
    ),
    "fibonacci_check": (
        "is_fibonacci(n)", "True if n (>= 0) is a Fibonacci number",
        "def is_fibonacci(n):\n    a, b = 0, 1\n    while a < n:\n        a, b = b, a + b\n    return a == n\n",
        lambda r: "\n".join(f"assert is_fibonacci({n}) == {str(n in (0,1,2,3,5,8,13,21,34,55,89,144,233,377,610,987))}"
                            for n in [0, 1, 4, 8, 14, 89, 100, 987, 988]),
    ),
}


def passes(code: str, tests: str, timeout: int = 10) -> bool:
    program = code + "\n" + tests + "\nprint('OK')\n"
    try:
        p = subprocess.run([sys.executable, "-I", "-c", program],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False
    return p.returncode == 0 and "OK" in p.stdout


def extract_code(resp: str, name: str) -> str | None:
    import re
    m = re.search(r"```(?:python)?\s*\n(.*?)```", resp, re.S)
    code = m.group(1) if m else resp
    return code if f"def {name}(" in code else None


def teacher(base: str, model_id: str, prompt: str, max_tokens: int) -> str:
    body = json.dumps({"model": model_id,
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0.7}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-base", default="http://127.0.0.1:8181/v1")
    ap.add_argument("--model-id", default="qwen3-coder-30b-a3b-colibri")
    ap.add_argument("--per-template", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    kept, tried = [], 0
    t0 = time.time()
    for name, (sig, spec, ref, make_tests) in TEMPLATES.items():
        tests = make_tests(rng)
        assert passes(ref, tests), f"reference for {name} fails its own tests"
        for v in range(args.per_template):
            tried += 1
            prompt = (f"Write a Python function `{sig}` that returns {spec}. "
                      "Answer with code only, no explanation, no tests.")
            try:
                resp = teacher(args.api_base, args.model_id, prompt, args.max_tokens)
            except Exception as e:
                print(f"{name}[{v}] teacher error: {e}", flush=True)
                continue
            code = extract_code(resp, name)
            ok = code is not None and passes(code, tests)
            print(f"{name}[{v}] {'PASS' if ok else 'fail'} ({time.time()-t0:.0f}s)", flush=True)
            if ok:
                kept.append({"messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "```python\n" + code.strip() + "\n```"},
                ], "task": name})
                with OUT.open("w", encoding="utf-8") as f:
                    for row in kept:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"kept {len(kept)}/{tried} -> {OUT} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()