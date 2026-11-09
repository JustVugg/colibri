"""Head-to-head, execution-verified benchmark for any Colibri OpenAI endpoint.

Scores generated code by actually running unit tests (scripts/evaluate.exec_score).
Supports optional best-of-N sampling with execution-based selection, which turns
extra test-time compute into a measurable quality gain.

Examples:
  .venv/Scripts/python.exe scripts/bench_best_model.py \
      --base http://127.0.0.1:8181/v1 --model qwen3-coder-30b-a3b-colibri --label coder30b

  .venv/Scripts/python.exe scripts/bench_best_model.py \
      --base http://127.0.0.1:8183/v1 --model qwen38-flash-next --label qwen38_nothink \
      --no-think --max-tokens 360

  ... --best-of 3 --temp 0.7 --top-p 0.8   # execution-selected best-of-3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import extract_code, exec_score  # noqa: E402
from bench_agent_vs_colibri import TASKS as BASE_TASKS  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS, exist_ok=True)

EXTRA_TASKS = [
    {
        "id": "lru_cache", "difficulty": "hard", "func": "__init__",
        "instruction": "Write a Python class `LRUCache` with `__init__(self, capacity)`, "
                       "`get(key)` returning the value or -1, and `put(key, value)`. "
                       "It must evict the least recently used entry when full. Answer with code only.\n\n",
        "tests": (
            "c = LRUCache(2)\n"
            "c.put(1, 1); c.put(2, 2)\n"
            "assert c.get(1) == 1\n"
            "c.put(3, 3)\n"
            "assert c.get(2) == -1\n"
            "assert c.get(3) == 3\n"
            "c.put(1, 10)\n"
            "assert c.get(1) == 10\n"
            "c2 = LRUCache(1)\n"
            "c2.put(5, 5); c2.put(6, 6)\n"
            "assert c2.get(5) == -1 and c2.get(6) == 6"
        ),
        "agent_ref": (
            "from collections import OrderedDict\n\n"
            "class LRUCache:\n"
            "    def __init__(self, capacity):\n"
            "        self.cap = capacity\n"
            "        self.d = OrderedDict()\n"
            "    def get(self, key):\n"
            "        if key not in self.d:\n"
            "            return -1\n"
            "        self.d.move_to_end(key)\n"
            "        return self.d[key]\n"
            "    def put(self, key, value):\n"
            "        if key in self.d:\n"
            "            self.d.move_to_end(key)\n"
            "        self.d[key] = value\n"
            "        if len(self.d) > self.cap:\n"
            "            self.d.popitem(last=False)\n"
        ),
    },
    {
        "id": "topo_sort", "difficulty": "hard", "func": "topo_sort",
        "instruction": "Write a Python function `topo_sort(n, edges)` that returns a topological "
                       "ordering of nodes 0..n-1 for the directed edges `(u, v)` (u before v). "
                       "Return `[]` if the graph has a cycle. Answer with code only.\n\n",
        "tests": (
            "from collections import deque\n"
            "def _valid(n, edges, order):\n"
            "    if len(order) != n or sorted(order) != list(range(n)):\n"
            "        return False\n"
            "    pos = {v: i for i, v in enumerate(order)}\n"
            "    return all(pos[u] < pos[v] for u, v in edges)\n"
            "e1 = [(0, 1), (1, 2), (2, 3)]\n"
            "assert _valid(4, e1, topo_sort(4, e1))\n"
            "e2 = [(3, 0), (1, 0), (2, 1)]\n"
            "assert _valid(4, e2, topo_sort(4, e2))\n"
            "assert topo_sort(3, [(0, 1), (1, 2), (2, 0)]) == []\n"
            "assert _valid(3, [], topo_sort(3, []))"
        ),
        "agent_ref": (
            "from collections import deque\n"
            "def topo_sort(n, edges):\n"
            "    adj = [[] for _ in range(n)]\n"
            "    indeg = [0] * n\n"
            "    for u, v in edges:\n"
            "        adj[u].append(v)\n"
            "        indeg[v] += 1\n"
            "    q = deque(i for i in range(n) if indeg[i] == 0)\n"
            "    out = []\n"
            "    while q:\n"
            "        u = q.popleft()\n"
            "        out.append(u)\n"
            "        for v in adj[u]:\n"
            "            indeg[v] -= 1\n"
            "            if indeg[v] == 0:\n"
            "                q.append(v)\n"
            "    return out if len(out) == n else []\n"
        ),
    },
]

DEFAULT_IDS = ["sum_even", "count_vowels", "flatten", "binary_search",
               "second_largest", "merge_intervals", "valid_parentheses",
               "fib", "lru_cache", "topo_sort"]

HARD_TASKS = [
    {
        "id": "expr_eval", "difficulty": "hard", "func": "expr_eval",
        "instruction": "Write a Python function `expr_eval(expr)` that evaluates a string arithmetic "
                       "expression containing non-negative integers, + - * /, parentheses and unary "
                       "minus, returning a float. Do not use eval(). Answer with code only.\n\n",
        "tests": (
            "assert abs(expr_eval('1+2*3') - 7) < 1e-9\n"
            "assert abs(expr_eval('(1+2)*3') - 9) < 1e-9\n"
            "assert abs(expr_eval('8/2/2') - 2) < 1e-9\n"
            "assert abs(expr_eval('-3+5') - 2) < 1e-9\n"
            "assert abs(expr_eval('2*(3+4)-5') - 9) < 1e-9\n"
            "assert abs(expr_eval('10-2-3') - 5) < 1e-9\n"
            "assert abs(expr_eval('-(2+3)*-(1+1)') - 10) < 1e-9"
        ),
        "agent_ref": (
            "def expr_eval(expr):\n"
            "    tokens = []\n"
            "    i = 0\n"
            "    while i < len(expr):\n"
            "        c = expr[i]\n"
            "        if c.isspace():\n"
            "            i += 1\n"
            "        elif c.isdigit() or c == '.':\n"
            "            j = i\n"
            "            while j < len(expr) and (expr[j].isdigit() or expr[j] == '.'):\n"
            "                j += 1\n"
            "            tokens.append(float(expr[i:j]))\n"
            "            i = j\n"
            "        else:\n"
            "            tokens.append(c)\n"
            "            i += 1\n"
            "    pos = [0]\n"
            "    def peek():\n"
            "        return tokens[pos[0]] if pos[0] < len(tokens) else None\n"
            "    def take():\n"
            "        t = tokens[pos[0]]; pos[0] += 1; return t\n"
            "    def primary():\n"
            "        t = peek()\n"
            "        if t == '(':\n"
            "            take(); v = add_sub()\n"
            "            take(); return v\n"
            "        if t == '-':\n"
            "            take(); return -primary()\n"
            "        if t == '+':\n"
            "            take(); return primary()\n"
            "        return take()\n"
            "    def mul_div():\n"
            "        v = primary()\n"
            "        while peek() in ('*', '/'):\n"
            "            op = take(); r = primary()\n"
            "            v = v * r if op == '*' else v / r\n"
            "        return v\n"
            "    def add_sub():\n"
            "        v = mul_div()\n"
            "        while peek() in ('+', '-'):\n"
            "            op = take(); r = mul_div()\n"
            "            v = v + r if op == '+' else v - r\n"
            "        return v\n"
            "    return add_sub()\n"
        ),
    },
    {
        "id": "parse_csv_line", "difficulty": "hard", "func": "parse_csv_line",
        "instruction": "Write a Python function `parse_csv_line(line)` that splits one CSV line into "
                       "fields. Double-quoted fields may contain commas and doubled quotes (\"\"). "
                       "Empty fields must be preserved. Answer with code only.\n\n",
        "tests": (
            "assert parse_csv_line('a,b,c') == ['a','b','c']\n"
            "assert parse_csv_line('\"a,b\",c') == ['a,b','c']\n"
            "assert parse_csv_line('\"he said \"\"hi\"\"\",2') == ['he said \"hi\"','2']\n"
            "assert parse_csv_line(',,') == ['','','']\n"
            "assert parse_csv_line('') == ['']\n"
            "assert parse_csv_line('a,\"b\"') == ['a','b']\n"
            "assert parse_csv_line('\"\"\"\"') == ['\"']"
        ),
        "agent_ref": (
            "def parse_csv_line(line):\n"
            "    fields, cur = [], []\n"
            "    i, n = 0, len(line)\n"
            "    inq = False\n"
            "    while i < n:\n"
            "        c = line[i]\n"
            "        if inq:\n"
            "            if c == '\"':\n"
            "                if i + 1 < n and line[i+1] == '\"':\n"
            "                    cur.append('\"'); i += 2; continue\n"
            "                inq = False; i += 1; continue\n"
            "            cur.append(c); i += 1\n"
            "        else:\n"
            "            if c == '\"':\n"
            "                inq = True; i += 1\n"
            "            elif c == ',':\n"
            "                fields.append(''.join(cur)); cur = []; i += 1\n"
            "            else:\n"
            "                cur.append(c); i += 1\n"
            "    fields.append(''.join(cur))\n"
            "    return fields\n"
        ),
    },
    {
        "id": "rotate_matrix", "difficulty": "medium", "func": "rotate_matrix",
        "instruction": "Write a Python function `rotate_matrix(m)` that rotates a square n x n matrix "
                       "90 degrees clockwise in place and returns it. Answer with code only.\n\n",
        "tests": (
            "assert rotate_matrix([[1,2],[3,4]]) == [[3,1],[4,2]]\n"
            "assert rotate_matrix([[1,2,3],[4,5,6],[7,8,9]]) == [[7,4,1],[8,5,2],[9,6,3]]\n"
            "assert rotate_matrix([[1]]) == [[1]]\n"
            "assert rotate_matrix([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]) == "
            "[[13,9,5,1],[14,10,6,2],[15,11,7,3],[16,12,8,4]]"
        ),
        "agent_ref": (
            "def rotate_matrix(m):\n"
            "    n = len(m)\n"
            "    for i in range(n):\n"
            "        for j in range(i + 1, n):\n"
            "            m[i][j], m[j][i] = m[j][i], m[i][j]\n"
            "    for row in m:\n"
            "        row.reverse()\n"
            "    return m\n"
        ),
    },
    {
        "id": "min_window_substring", "difficulty": "hard", "func": "min_window",
        "instruction": "Write a Python function `min_window(s, t)` returning the shortest substring of "
                       "s containing every character of t with the same multiplicity, or '' if none "
                       "exists. Answer with code only.\n\n",
        "tests": (
            "assert min_window('ADOBECODEBANC','ABC') == 'BANC'\n"
            "assert min_window('a','a') == 'a'\n"
            "assert min_window('a','aa') == ''\n"
            "assert min_window('aa','aa') == 'aa'\n"
            "assert min_window('ab','b') == 'b'\n"
            "assert min_window('bba','ab') == 'ba'"
        ),
        "agent_ref": (
            "from collections import Counter\n"
            "def min_window(s, t):\n"
            "    need = Counter(t)\n"
            "    missing = len(t)\n"
            "    best = None\n"
            "    left = 0\n"
            "    for right, c in enumerate(s):\n"
            "        if need[c] > 0:\n"
            "            missing -= 1\n"
            "        need[c] -= 1\n"
            "        while missing == 0:\n"
            "            if best is None or right - left + 1 < len(best):\n"
            "                best = s[left:right+1]\n"
            "            need[s[left]] += 1\n"
            "            if need[s[left]] > 0:\n"
            "                missing += 1\n"
            "            left += 1\n"
            "    return best or ''\n"
        ),
    },
    {
        "id": "can_word_break", "difficulty": "medium", "func": "can_word_break",
        "instruction": "Write a Python function `can_word_break(s, words)` returning True if s can be "
                       "segmented into one or more dictionary words. Answer with code only.\n\n",
        "tests": (
            "assert can_word_break('leetcode', ['leet','code']) is True\n"
            "assert can_word_break('applepenapple', ['apple','pen']) is True\n"
            "assert can_word_break('catsandog', ['cats','dog','sand','and','cat']) is False\n"
            "assert can_word_break('', ['a']) is True\n"
            "assert can_word_break('a', []) is False\n"
            "assert can_word_break('aaaaaaa', ['aaaa','aaa']) is True"
        ),
        "agent_ref": (
            "def can_word_break(s, words):\n"
            "    words = set(words)\n"
            "    n = len(s)\n"
            "    dp = [False] * (n + 1)\n"
            "    dp[0] = True\n"
            "    for i in range(1, n + 1):\n"
            "        for j in range(i):\n"
            "            if dp[j] and s[j:i] in words:\n"
            "                dp[i] = True\n"
            "                break\n"
            "    return dp[n]\n"
        ),
    },
    {
        "id": "kth_smallest_sorted_matrix", "difficulty": "medium", "func": "kth_smallest",
        "instruction": "Write a Python function `kth_smallest(matrix, k)` returning the k-th smallest "
                       "element (1-indexed) of a matrix whose rows and columns are sorted ascending. "
                       "Answer with code only.\n\n",
        "tests": (
            "assert kth_smallest([[1,5,9],[10,11,13],[12,13,15]], 8) == 13\n"
            "assert kth_smallest([[-5]], 1) == -5\n"
            "assert kth_smallest([[1,2],[3,4]], 2) == 2\n"
            "assert kth_smallest([[1,3,5],[6,7,12],[11,14,14]], 6) == 11"
        ),
        "agent_ref": (
            "import heapq\n"
            "def kth_smallest(matrix, k):\n"
            "    h = [(row[0], i, 0) for i, row in enumerate(matrix) if row]\n"
            "    heapq.heapify(h)\n"
            "    for _ in range(k):\n"
            "        v, r, c = heapq.heappop(h)\n"
            "        if c + 1 < len(matrix[r]):\n"
            "            heapq.heappush(h, (matrix[r][c+1], r, c+1))\n"
            "    return v\n"
        ),
    },
    {
        "id": "json_pointer", "difficulty": "hard", "func": "json_pointer",
        "instruction": "Write a Python function `json_pointer(obj, pointer)` resolving an RFC 6901 "
                       "JSON pointer such as '/a/0/b'. Unescape '~1' to '/' and '~0' to '~'. The empty "
                       "pointer returns obj. Raise KeyError for a missing object key and IndexError "
                       "for an out-of-range list index. Answer with code only.\n\n",
        "tests": (
            "obj = {'a': [{'b': 5}], 'c/d': 7, 'e~f': 8}\n"
            "assert json_pointer(obj, '') == obj\n"
            "assert json_pointer(obj, '/a/0/b') == 5\n"
            "assert json_pointer(obj, '/c~1d') == 7\n"
            "assert json_pointer(obj, '/e~0f') == 8\n"
            "try:\n"
            "    json_pointer(obj, '/missing'); assert False\n"
            "except KeyError: pass\n"
            "try:\n"
            "    json_pointer(obj, '/a/5'); assert False\n"
            "except (IndexError, KeyError): pass"
        ),
        "agent_ref": (
            "def json_pointer(obj, pointer):\n"
            "    if pointer == '':\n"
            "        return obj\n"
            "    parts = pointer.split('/')[1:]\n"
            "    cur = obj\n"
            "    for part in parts:\n"
            "        key = part.replace('~1', '/').replace('~0', '~')\n"
            "        if isinstance(cur, list):\n"
            "            cur = cur[int(key)]\n"
            "        else:\n"
            "            cur = cur[key]\n"
            "    return cur\n"
        ),
    },
    {
        "id": "rle_roundtrip", "difficulty": "medium", "func": "rle_encode",
        "instruction": "Write `rle_encode(s)` and `rle_decode(s)` for run-length encoding where every "
                       "run is encoded as its decimal count followed by the character (e.g. 'aaab' -> "
                       "'3a1b'). `rle_decode` must invert `rle_encode`. Answer with code only.\n\n",
        "tests": (
            "assert rle_encode('aaab') == '3a1b'\n"
            "assert rle_encode('') == ''\n"
            "assert rle_encode('abc') == '1a1b1c'\n"
            "assert rle_encode('aaaaaaaaaaaa') == '12a'\n"
            "assert rle_decode('3a1b') == 'aaab'\n"
            "assert rle_decode('12a') == 'aaaaaaaaaaaa'\n"
            "assert rle_decode('') == ''\n"
            "assert rle_decode(rle_encode('wwwwbbb')) == 'wwwwbbb'"
        ),
        "agent_ref": (
            "def rle_encode(s):\n"
            "    out = []\n"
            "    i = 0\n"
            "    while i < len(s):\n"
            "        j = i\n"
            "        while j < len(s) and s[j] == s[i]:\n"
            "            j += 1\n"
            "        out.append(str(j - i) + s[i])\n"
            "        i = j\n"
            "    return ''.join(out)\n"
            "def rle_decode(s):\n"
            "    out = []\n"
            "    i = 0\n"
            "    while i < len(s):\n"
            "        j = i\n"
            "        while j < len(s) and s[j].isdigit():\n"
            "            j += 1\n"
            "        out.append(s[j] * int(s[i:j]))\n"
            "        i = j + 1\n"
            "    return ''.join(out)\n"
        ),
    },
    {
        "id": "lexicographic_topo", "difficulty": "hard", "func": "topo_smallest",
        "instruction": "Write a Python function `topo_smallest(n, edges)` returning the "
                       "lexicographically smallest topological order of nodes 0..n-1 for directed "
                       "edges (u, v), or [] if the graph contains a cycle. Answer with code only.\n\n",
        "tests": (
            "assert topo_smallest(4, [(0,1),(0,2),(1,3),(2,3)]) == [0,1,2,3]\n"
            "assert topo_smallest(4, [(1,0),(2,0)]) == [1,2,0,3]\n"
            "assert topo_smallest(3, [(0,1),(1,2),(2,0)]) == []\n"
            "assert topo_smallest(3, []) == [0,1,2]\n"
            "assert topo_smallest(5, [(4,0),(3,1)]) == [2,3,1,4,0]"
        ),
        "agent_ref": (
            "import heapq\n"
            "def topo_smallest(n, edges):\n"
            "    adj = [[] for _ in range(n)]\n"
            "    indeg = [0] * n\n"
            "    for u, v in edges:\n"
            "        adj[u].append(v)\n"
            "        indeg[v] += 1\n"
            "    h = [i for i in range(n) if indeg[i] == 0]\n"
            "    heapq.heapify(h)\n"
            "    out = []\n"
            "    while h:\n"
            "        u = heapq.heappop(h)\n"
            "        out.append(u)\n"
            "        for v in adj[u]:\n"
            "            indeg[v] -= 1\n"
            "            if indeg[v] == 0:\n"
            "                heapq.heappush(h, v)\n"
            "    return out if len(out) == n else []\n"
        ),
    },
    {
        "id": "atoi", "difficulty": "medium", "func": "my_atoi",
        "instruction": "Write a Python function `my_atoi(s)` implementing string-to-integer conversion: "
                       "skip leading spaces, accept an optional sign, read digits until a non-digit, "
                       "return 0 if there are no digits, and clamp to the 32-bit signed range. "
                       "Answer with code only.\n\n",
        "tests": (
            "assert my_atoi('42') == 42\n"
            "assert my_atoi('   -42') == -42\n"
            "assert my_atoi('4193 with words') == 4193\n"
            "assert my_atoi('words and 987') == 0\n"
            "assert my_atoi('-91283472332') == -2147483648\n"
            "assert my_atoi('+1') == 1\n"
            "assert my_atoi('') == 0\n"
            "assert my_atoi('2147483648') == 2147483647"
        ),
        "agent_ref": (
            "def my_atoi(s):\n"
            "    i, n = 0, len(s)\n"
            "    while i < n and s[i] == ' ':\n"
            "        i += 1\n"
            "    sign = 1\n"
            "    if i < n and s[i] in '+-':\n"
            "        sign = -1 if s[i] == '-' else 1\n"
            "        i += 1\n"
            "    start = i\n"
            "    while i < n and s[i].isdigit():\n"
            "        i += 1\n"
            "    if i == start:\n"
            "        return 0\n"
            "    v = sign * int(s[start:i])\n"
            "    return max(-2**31, min(2**31 - 1, v))\n"
        ),
    },
    {
        "id": "simplify_path", "difficulty": "hard", "func": "simplify_path",
        "instruction": "Write a Python function `simplify_path(path)` returning the canonical absolute "
                       "Unix path: single slashes, no '.' segments, '..' pops one segment (ignored at "
                       "the root), no trailing slash except for the root. Answer with code only.\n\n",
        "tests": (
            "assert simplify_path('/home/') == '/home'\n"
            "assert simplify_path('/../') == '/'\n"
            "assert simplify_path('/home//foo/') == '/home/foo'\n"
            "assert simplify_path('/a/./b/../../c/') == '/c'\n"
            "assert simplify_path('/...') == '/...'\n"
            "assert simplify_path('/a//b////c/d//././/..') == '/a/b/c'"
        ),
        "agent_ref": (
            "def simplify_path(path):\n"
            "    stack = []\n"
            "    for part in path.split('/'):\n"
            "        if part in ('', '.'):\n"
            "            continue\n"
            "        if part == '..':\n"
            "            if stack:\n"
            "                stack.pop()\n"
            "        else:\n"
            "            stack.append(part)\n"
            "    return '/' + '/'.join(stack)\n"
        ),
    },
    {
        "id": "lis_length", "difficulty": "medium", "func": "lis_length",
        "instruction": "Write a Python function `lis_length(nums)` returning the length of the longest "
                       "strictly increasing subsequence. Answer with code only.\n\n",
        "tests": (
            "assert lis_length([10,9,2,5,3,7,101,18]) == 4\n"
            "assert lis_length([7,7,7]) == 1\n"
            "assert lis_length([]) == 0\n"
            "assert lis_length([0,1,0,3,2,3]) == 4\n"
            "assert lis_length([1,2,3,4,5]) == 5"
        ),
        "agent_ref": (
            "import bisect\n"
            "def lis_length(nums):\n"
            "    tails = []\n"
            "    for x in nums:\n"
            "        i = bisect.bisect_left(tails, x)\n"
            "        if i == len(tails):\n"
            "            tails.append(x)\n"
            "        else:\n"
            "            tails[i] = x\n"
            "    return len(tails)\n"
        ),
    },
]

HARD_IDS = [t["id"] for t in HARD_TASKS]

SYSTEM = ("You are an expert Python programmer. Solve the problem carefully, handle edge "
          "cases, and keep the complete solution compact (under 120 lines). Reply with "
          "exactly one ```python fenced code block and nothing else.")

THINKING_OFF_MODELS = ("qwen3-coder", "qwen3-coder-30b-a3b-colibri")


def tasks_for(ids):
    wanted = set(ids)
    pool = {t["id"]: t for t in BASE_TASKS + EXTRA_TASKS + HARD_TASKS}
    missing = wanted - set(pool)
    if missing:
        raise SystemExit(f"unknown task ids: {sorted(missing)}")
    return [pool[i] for i in ids]


def post(url, body, api_key, timeout=1800):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def generate(base, model, api_key, messages, args, temp, max_tokens=None):
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens or args.max_tokens,
        "temperature": temp,
        "top_p": args.top_p,
        "stream": False,
    }
    if args.no_think:
        body["enable_thinking"] = False
    elif args.effort:
        body["enable_thinking"] = True
        body["reasoning_effort"] = args.effort
    t0 = time.time()
    try:
        d = post(base.rstrip("/") + "/chat/completions", body, api_key)
        dt = time.time() - t0
        msg = d["choices"][0]["message"]
        text = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        usage = d.get("usage", {})
        ctok = usage.get("completion_tokens", 0)
        return {"text": text, "reasoning": reasoning, "time_s": round(dt, 3),
                "completion_tokens": ctok,
                "tokens_per_s": round(ctok / dt, 2) if dt > 0 and ctok else 0.0,
                "error": None}
    except Exception as e:  # noqa: BLE001 - benchmark harness reports failures
        return {"text": "", "reasoning": "", "time_s": round(time.time() - t0, 3),
                "completion_tokens": 0, "tokens_per_s": 0.0, "error": str(e)}


def run_attempt(base, model, api_key, task, args, temp, messages=None):
    msgs = messages or [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": task["instruction"]}]
    gen = generate(base, model, api_key, msgs, args, temp)
    code = extract_code(gen["text"], task["func"]) if gen["text"] else ""
    score = exec_score(code if code else "raise SystemExit('no code')", task)
    return {**gen, "code": code, "exec_passed": bool(score.get("exec_passed")),
            "exec_error": score.get("exec_error")}


def repair_messages(task, attempt, round_no):
    assistant = attempt["text"] or (f"```python\n{attempt['code']}\n```" if attempt["code"] else "")
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": task["instruction"]},
        {"role": "assistant", "content": assistant},
        {"role": "user", "content": (
            f"This solution fails the hidden unit tests (attempt {round_no}): "
            f"{attempt['exec_error']}. Find the bug (consider empty inputs, duplicates, "
            "escaping and edge cases), then return one complete corrected ```python "
            "code block and nothing else.")},
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--api-key", default=os.environ.get("COLI_API_KEY", "local"))
    ap.add_argument("--task-ids", default=None)
    ap.add_argument("--suite", default="hard", choices=["hard", "base", "extended"])
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--best-of", type=int, default=1, dest="best_of")
    ap.add_argument("--repair-rounds", type=int, default=0, dest="repair_rounds")
    ap.add_argument("--repair-temp", type=float, default=0.2, dest="repair_temp")
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh"])
    args = ap.parse_args()

    ids = ([x.strip() for x in args.task_ids.split(",") if x.strip()]
           if args.task_ids else
           {"hard": HARD_IDS, "base": DEFAULT_IDS,
            "extended": DEFAULT_IDS + HARD_IDS}[args.suite])
    tasks = tasks_for(ids)
    rows = []
    t_start = time.time()
    for t in tasks:
        # Reference solution must pass the identical harness; guards task validity.
        ref = exec_score(t["agent_ref"], t)
        if not ref["exec_passed"]:
            raise SystemExit(f"reference solution for {t['id']} failed: {ref['exec_error']}")
        attempts = []
        for i in range(args.best_of):
            temp = args.temp if i == 0 or args.temp > 0 else 0.0
            attempts.append({**run_attempt(args.base, args.model, args.api_key, t, args, temp),
                             "stage": "initial"})
        first_pass = attempts[0]["exec_passed"]
        repairs = 0
        if not any(a["exec_passed"] for a in attempts):
            base_attempt = attempts[0]
            for r in range(1, args.repair_rounds + 1):
                rep = run_attempt(args.base, args.model, args.api_key, t, args,
                                  args.repair_temp, repair_messages(t, base_attempt, r))
                rep["stage"] = f"repair{r}"
                attempts.append(rep)
                repairs += 1
                if rep["exec_passed"]:
                    break
                base_attempt = rep
        chosen = next((a for a in attempts if a["exec_passed"]), attempts[-1])
        rows.append({
            "id": t["id"], "difficulty": t["difficulty"],
            "passed": any(a["exec_passed"] for a in attempts),
            "initial_passed": first_pass,
            "repairs": repairs,
            "chosen_passed": chosen["exec_passed"],
            "chosen": {"code": chosen["code"], "time_s": chosen["time_s"],
                       "completion_tokens": chosen["completion_tokens"],
                       "tokens_per_s": chosen["tokens_per_s"],
                       "exec_error": chosen["exec_error"]},
            "attempts": [{"stage": a["stage"], "passed": a["exec_passed"],
                          "time_s": a["time_s"],
                          "completion_tokens": a["completion_tokens"],
                          "exec_error": a["exec_error"],
                          "content_head": a["text"][:160]} for a in attempts],
        })
        tag = "PASS" if rows[-1]["passed"] else "FAIL"
        best = rows[-1]["chosen"]
        print(f"[{t['difficulty']:6}] {t['id']:18} {tag} "
              f"{best['time_s']:7.1f}s {best['tokens_per_s']:6.1f}tok/s "
              f"tokens={best['completion_tokens']}", flush=True)
    total_s = time.time() - t_start
    n = len(rows)
    passed = sum(1 for r in rows if r["passed"])
    initial = sum(1 for r in rows if r["initial_passed"])
    repairs_used = sum(r["repairs"] for r in rows)
    tok = sum(r["chosen"]["completion_tokens"] for r in rows)
    gen_s = sum(a["time_s"] for r in rows for a in r["attempts"])
    by_diff = {}
    for d in ("easy", "medium", "hard"):
        sub = [r for r in rows if r["difficulty"] == d]
        by_diff[d] = {"total": len(sub), "passed": sum(1 for r in sub if r["passed"])}
    summary = {
        "total": n, "passed": passed, "pass_at_1": round(passed / n, 4),
        "initial_passed": initial, "initial_pass_rate": round(initial / n, 4),
        "repair_lift": passed - initial, "repairs_used": repairs_used,
        "by_difficulty": by_diff, "best_of": args.best_of,
        "repair_rounds": args.repair_rounds,
        "completion_tokens": tok, "generation_s": round(gen_s, 2),
        "tokens_per_s": round(tok / gen_s, 2) if gen_s else 0.0,
        "wall_s": round(total_s, 2),
        "model": args.model, "base": args.base,
        "max_tokens": args.max_tokens, "temp": args.temp, "top_p": args.top_p,
        "no_think": args.no_think, "effort": args.effort,
    }
    out = {"summary": summary, "rows": rows}
    path = os.path.join(RESULTS, f"bench_best_{args.label}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(summary, indent=1))
    print("saved", path)


if __name__ == "__main__":
    main()
