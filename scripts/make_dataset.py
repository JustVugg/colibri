import json, os, random, time
from pathlib import Path

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)), "results"), "dataset.jsonl")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

examples = []

# Tiny starter set of coding instruction pairs for a codegen-style task.
# The user can expand this with their own corpus; this is a seed to verify the evaluate
# pipeline and a training script can read it.
seed = [
    {
        "instruction": "Write a Python function that returns the sum of all even numbers in a list.",
        "input": "",
        "output": "def sum_even(lst):\n    return sum(x for x in lst if x % 2 == 0)\n",
    },
    {
        "instruction": "Write a Python function that returns the Fibonacci sequence up to n terms.",
        "input": "",
        "output": "def fib(n):\n    a, b = 0, 1\n    seq = []\n    for _ in range(n):\n        seq.append(a)\n        a, b = b, a + b\n    return seq\n",
    },
    {
        "instruction": "Write a Python function that checks whether a string is a palindrome.",
        "input": "",
        "output": "def is_palindrome(s):\n    s = ''.join(c.lower() for c in s if c.isalnum())\n    return s == s[::-1]\n",
    },
    {
        "instruction": "Write a Python function that returns the factorial of n.",
        "input": "",
        "output": "def factorial(n):\n    if n < 0:\n        raise ValueError('negative factorial')\n    result = 1\n    for i in range(1, n + 1):\n        result *= i\n    return result\n",
    },
    {
        "instruction": "Write a Python function that returns the longest word in a sentence.",
        "input": "",
        "output": "def longest_word(sentence):\n    words = sentence.split()\n    return max(words, key=len) if words else ''\n",
    },
    {
        "instruction": "Write a Python function that flattens a nested list.",
        "input": "",
        "output": "def flatten(lst):\n    result = []\n    for x in lst:\n        if isinstance(x, list):\n            result.extend(flatten(x))\n        else:\n            result.append(x)\n    return result\n",
    },
    {
        "instruction": "Write a Python function that returns the intersection of two lists.",
        "input": "",
        "output": "def intersection(a, b):\n    set_b = set(b)\n    seen = set()\n    out = []\n    for x in a:\n        if x in set_b and x not in seen:\n            out.append(x)\n            seen.add(x)\n    return out\n",
    },
    {
        "instruction": "Write a Python function that returns the transpose of a matrix.",
        "input": "",
        "output": "def transpose(matrix):\n    return [list(row) for row in zip(*matrix)]\n",
    },
    {
        "instruction": "Write a Python function that returns the running median of a stream of numbers.",
        "input": "",
        "output": "import heapq\n\ndef running_median(stream):\n    low, high = [], []\n    medians = []\n    for x in stream:\n        heapq.heappush(low, -x)\n        heapq.heappush(high, -heapq.heappop(low))\n        if len(high) > len(low):\n            heapq.heappush(low, -heapq.heappop(high))\n        if len(low) > len(high):\n            median = -low[0]\n        else:\n            median = (-low[0] + high[0]) / 2.0\n        medians.append(median)\n    return medians\n",
    },
    {
        "instruction": "Write a Python function that removes duplicate elements from a list while preserving order.",
        "input": "",
        "output": "def dedupe(lst):\n    seen = set()\n    out = []\n    for x in lst:\n        if x not in seen:\n            out.append(x)\n            seen.add(x)\n    return out\n",
    },
]

random.seed(0)
random.shuffle(seed)
for e in seed:
    examples.append(e)

# A minimal local codegen target: repeat the seed with small perturbations for a tiny
# training corpus. In practice you would replace this with a larger curated corpus.
extra = []
for e in seed:
    dup = dict(e)
    dup["output"] = e["output"].rstrip("\n") + "\n"
    extra.append(dup)
examples.extend(extra)

random.shuffle(examples)
with open(OUT, "w", encoding="utf-8") as f:
    for e in examples:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")
print("wrote", OUT, "examples:", len(examples))


# Report size in tokens (approx) for the evaluate script to use as a reference.
tok = None
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Coder-30B-A3B-Instruct", trust_remote_code=True)
except Exception as ex:
    print("tokenizer not available for token count:", ex)

if tok is not None:
    n = 0
    for e in examples:
        text = e["instruction"] + "\n" + e.get("input", "") + "\n" + e["output"]
        n += len(tok.encode(text))
    print("approx dataset tokens:", n)
