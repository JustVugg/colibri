"""Hunt for weaknesses in the served 30B Colibri model on harder, edge-case-heavy
coding tasks (execution-verified). Reuses evaluate.run_eval/exec_score. Writes
results/eval_weakness_hunt.json. Run while the Colibri serve is up on :8181.
"""
from __future__ import annotations

import json
import os
import urllib.request

import evaluate

HARD = [
    {"id": "roman_to_int", "func": "roman_to_int",
     "instruction": "Write a Python function `roman_to_int(s)` converting a Roman numeral string to an integer. Answer with code only.\n",
     "tests": "assert roman_to_int('III')==3\nassert roman_to_int('IV')==4\nassert roman_to_int('IX')==9\nassert roman_to_int('LVIII')==58\nassert roman_to_int('MCMXCIV')==1994"},
    {"id": "int_to_roman", "func": "int_to_roman",
     "instruction": "Write a Python function `int_to_roman(n)` converting an integer (1..3999) to a Roman numeral string. Answer with code only.\n",
     "tests": "assert int_to_roman(3)=='III'\nassert int_to_roman(4)=='IV'\nassert int_to_roman(9)=='IX'\nassert int_to_roman(58)=='LVIII'\nassert int_to_roman(1994)=='MCMXCIV'"},
    {"id": "edit_distance", "func": "edit_distance",
     "instruction": "Write a Python function `edit_distance(a,b)` returning the Levenshtein edit distance between two strings. Answer with code only.\n",
     "tests": "assert edit_distance('horse','ros')==3\nassert edit_distance('intention','execution')==5\nassert edit_distance('','')==0\nassert edit_distance('a','')==1\nassert edit_distance('','b')==1"},
    {"id": "sliding_window_max", "func": "sliding_window_max",
     "instruction": "Write a Python function `sliding_window_max(lst,k)` returning the list of maximums of each consecutive window of size k (empty list if lst empty). Answer with code only.\n",
     "tests": "assert sliding_window_max([1,3,-1,-3,5,3,6,7],3)==[3,3,5,5,6,7]\nassert sliding_window_max([1],1)==[1]\nassert sliding_window_max([],1)==[]\nassert sliding_window_max([7,2,4],2)==[7,4]"},
    {"id": "big_add", "func": "big_add",
     "instruction": "Write a Python function `big_add(a,b)` adding two non-negative integers given as decimal strings, returning the sum as a string (no int() on the whole input). Answer with code only.\n",
     "tests": "assert big_add('123','456')=='579'\nassert big_add('0','0')=='0'\nassert big_add('999','1')=='1000'\nassert big_add('9','9')=='18'"},
    {"id": "spiral_order", "func": "spiral_order",
     "instruction": "Write a Python function `spiral_order(m)` returning the elements of a rectangular matrix in clockwise spiral order (empty list for empty matrix). Answer with code only.\n",
     "tests": "assert spiral_order([[1,2,3],[4,5,6],[7,8,9]])==[1,2,3,6,9,8,7,4,5]\nassert spiral_order([[1,2],[3,4]])==[1,2,4,3]\nassert spiral_order([[1]])==[1]\nassert spiral_order([])==[]\nassert spiral_order([[1,2,3]])==[1,2,3]"},
    {"id": "merge_intervals", "func": "merge_intervals",
     "instruction": "Write a Python function `merge_intervals(iv)` merging overlapping intervals, returned sorted by start. Answer with code only.\n",
     "tests": "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]])==[[1,6],[8,10],[15,18]]\nassert merge_intervals([[1,4],[4,5]])==[[1,5]]\nassert merge_intervals([])==[]\nassert merge_intervals([[1,4],[0,4]])==[[0,4]]\nassert merge_intervals([[1,4],[2,3]])==[[1,4]]"},
    {"id": "product_except_self", "func": "product_except_self",
     "instruction": "Write a Python function `product_except_self(lst)` where output[i] is the product of all elements except lst[i], without using division. Answer with code only.\n",
     "tests": "assert product_except_self([1,2,3,4])==[24,12,8,6]\nassert product_except_self([0,1,2])==[2,0,0]\nassert product_except_self([1,0])==[0,1]\nassert product_except_self([2,3])==[3,2]\nassert product_except_self([0,0])==[0,0]"},
    {"id": "count_islands", "func": "count_islands",
     "instruction": "Write a Python function `count_islands(g)` counting connected components of 1s (4-directional) in a grid of 0/1. Answer with code only.\n",
     "tests": "assert count_islands([[1,1,0],[0,1,0],[1,0,1]])==3\nassert count_islands([[1,0],[0,1]])==2\nassert count_islands([[1,1],[1,1]])==1\nassert count_islands([[0,0],[0,0]])==0"},
    {"id": "topo_sort", "func": "topo_sort",
     "instruction": "Write a Python function `topo_sort(n,edges)` returning a topological order of nodes 0..n-1 for directed edges [u,v], or None if there is a cycle. Answer with code only.\n",
     "tests": "r=topo_sort(4,[[0,1],[0,2],[1,3],[2,3]])\nassert r is not None and sorted(r)==[0,1,2,3]\np={v:i for i,v in enumerate(r)}\nassert p[0]<p[1] and p[0]<p[2] and p[1]<p[3] and p[2]<p[3]\nassert topo_sort(2,[[0,1],[1,0]]) is None\nassert topo_sort(1,[])==[0]"},
]


for _t in HARD:
    _t.setdefault("expected_contains", ["return"])
    _t.setdefault("forbidden_contains", [])


class ApiModel:
    def __init__(self, base, model_id):
        self.base = base.rstrip("/")
        self.model_id = model_id

    def generate(self, prompt, max_new_tokens=512, **kw):
        body = json.dumps({"model": self.model_id, "prompt": prompt,
                           "max_tokens": max_new_tokens, "temperature": 0}).encode()
        req = urllib.request.Request(self.base + "/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())["choices"][0]["text"]


def main():
    model = ApiModel("http://127.0.0.1:8181/v1", "qwen3-coder-30b-a3b-colibri")
    report, path = evaluate.run_eval(model, tasks=HARD, max_new_tokens=512,
                                     out_path=os.path.join(evaluate.RESULTS, "eval_weakness_hunt.json"))
    fails = [r["id"] for r in report["results"] if not r["score"].get("exec_passed")]
    print("WEAKNESSES:", fails if fails else "none (all passed)")


if __name__ == "__main__":
    main()