import argparse, json, os, re, subprocess, sys, time
from pathlib import Path

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS, exist_ok=True)


# A stable, small eval set of coding tasks with a reference answer and a simple check.
EVAL = [
    {
        "id": "sum_even",
        "instruction": "You are a helpful coding assistant. Write a Python function `sum_even(lst)` that sums the even numbers in a list. Answer with code only.\n\n",
        "expected_contains": ["sum", "even", "return"],
        "forbidden_contains": [],
        "func": "sum_even",
        "tests": "assert sum_even([1,2,3,4]) == 6\nassert sum_even([]) == 0\nassert sum_even([1,3,5]) == 0\nassert sum_even([-2,-4,7]) == -6\nassert sum_even([2]) == 2",
    },
    {
        "id": "fib",
        "instruction": "You are a helpful coding assistant. Write a Python function `fib(n)` that returns a list of the first n Fibonacci numbers (starting 0, 1). Answer with code only.\n\n",
        "expected_contains": ["fib", "return"],
        "forbidden_contains": [],
        "func": "fib",
        "tests": "assert fib(0) == []\nassert fib(1) == [0]\nassert fib(2) == [0,1]\nassert fib(7) == [0,1,1,2,3,5,8]",
    },
    {
        "id": "palindrome",
        "instruction": "You are a helpful coding assistant. Write a Python function `is_palindrome(s)` that returns True if the string reads the same forwards and backwards (exact characters, case-sensitive). Answer with code only.\n\n",
        "expected_contains": ["palindrome", "return"],
        "forbidden_contains": [],
        "func": "is_palindrome",
        "tests": "assert is_palindrome('racecar') is True\nassert is_palindrome('hello') is False\nassert is_palindrome('') is True\nassert is_palindrome('abba') is True\nassert is_palindrome('Abba') is False",
    },
    {
        "id": "factorial",
        "instruction": "You are a helpful coding assistant. Write a Python function `factorial(n)` that returns n! for non-negative integers (factorial(0) == 1). Answer with code only.\n\n",
        "expected_contains": ["factorial", "return"],
        "forbidden_contains": [],
        "func": "factorial",
        "tests": "assert factorial(0) == 1\nassert factorial(1) == 1\nassert factorial(5) == 120\nassert factorial(10) == 3628800",
    },
    {
        "id": "flatten",
        "instruction": "You are a helpful coding assistant. Write a Python function `flatten(lst)` that flattens an arbitrarily nested list of integers into a flat list, preserving order. Answer with code only.\n\n",
        "expected_contains": ["flatten", "return"],
        "forbidden_contains": [],
        "func": "flatten",
        "tests": "assert flatten([1,[2,3],4]) == [1,2,3,4]\nassert flatten([1,[2,[3,[4]]]]) == [1,2,3,4]\nassert flatten([]) == []\nassert flatten([[],[[]]]) == []\nassert flatten([1,2]) == [1,2]",
    },
]


def extract_code(resp, func):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", resp, re.S)
    code = m.group(1) if m else resp
    if "def " + func not in code:
        return None
    return code


def exec_score(resp, task, timeout=15):
    code = extract_code(resp, task["func"])
    if code is None:
        return {"exec_passed": False, "exec_error": "no function definition found"}
    program = code + "\n\n" + task["tests"] + "\nprint('ALL_TESTS_PASSED')\n"
    try:
        p = subprocess.run([sys.executable, "-I", "-c", program],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"exec_passed": False, "exec_error": f"timeout after {timeout}s"}
    if p.returncode == 0 and "ALL_TESTS_PASSED" in p.stdout:
        return {"exec_passed": True, "exec_error": None}
    err = (p.stderr.strip().splitlines() or ["unknown failure"])[-1]
    return {"exec_passed": False, "exec_error": err[:300]}


def score_response(resp, task):
    resp_low = resp.lower()
    hits = sum(1 for k in task["expected_contains"] if k.lower() in resp_low)
    expected = len(task["expected_contains"])
    forbidden = sum(1 for k in task["forbidden_contains"] if k.lower() in resp_low)
    passed = hits >= expected and forbidden == 0 and len(resp.strip()) > 0
    return {
        "hits": hits,
        "expected": expected,
        "forbidden_mentions": forbidden,
        "length_chars": len(resp),
        "passed": passed,
    }


def run_eval(model, tasks=None, max_new_tokens=256, out_path=None):
    tasks = tasks or EVAL
    results = []
    for t in tasks:
        t0 = time.time()
        try:
            if hasattr(model, "generate"):
                out = model.generate(t["instruction"], max_new_tokens=max_new_tokens)
            elif hasattr(model, "chat") or hasattr(model, "generate_text"):
                out = model.chat(t["instruction"], max_new_tokens=max_new_tokens)
            else:
                out = ""
        except Exception as e:
            out = f"[generation error: {e}]"
        dt = time.time() - t0
        sc = score_response(out, t)
        if t.get("tests"):
            sc.update(exec_score(out, t))
        results.append(
            {"id": t["id"], "output": out, "score": sc, "time_s": round(dt, 3)}
        )
    summary = {
        "total": len(results),
        "passed": sum(1 for r in results if r["score"]["passed"]),
        "exec_passed": sum(1 for r in results if r["score"].get("exec_passed")),
        "total_time_s": round(sum(r["time_s"] for r in results), 3),
    }
    report = {"results": results, "summary": summary}
    if out_path is None:
        out_path = os.path.join(RESULTS, "eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report, out_path


def load_model_and_eval(args):
    model = None
    if getattr(args, "api_base", None):
        import urllib.request

        class ApiModel:
            def __init__(self, base, model_id):
                self.base = base.rstrip("/")
                self.model_id = model_id

            def generate(self, prompt, max_new_tokens=256, **kwargs):
                body = json.dumps({
                    "model": self.model_id,
                    "prompt": prompt,
                    "max_tokens": max_new_tokens,
                    "temperature": 0,
                }).encode("utf-8")
                req = urllib.request.Request(
                    self.base + "/completions",
                    data=body,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=600) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["text"]

        model = ApiModel(args.api_base, args.model_id or "local")
        print("evaluating via API:", args.api_base)
    elif getattr(args, "hf_model", None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("loading transformers model:", args.hf_model)
        tok = AutoTokenizer.from_pretrained(args.hf_model)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        m = AutoModelForCausalLM.from_pretrained(args.hf_model, dtype=torch.bfloat16,
                                                 device_map=dev)

        class HFModel:
            def generate(self, prompt, max_new_tokens=256, **kwargs):
                msgs = [{"role": "user", "content": prompt}]
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                ids = tok(text, return_tensors="pt").to(m.device)
                with torch.inference_mode():
                    out = m.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
                return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

        model = HFModel()
        print("transformers model loaded on", dev)
    elif args.model_path:
        # Try AirLLM path
        try:
            import torch
            from airllm import AutoModel
            from transformers import AutoTokenizer
            print("loading airllm model from", args.model_path)
            model_obj = AutoModel.from_pretrained(
                args.model_path, device="cpu", dtype=torch.bfloat16,
                layer_shards_saving_path=r"D:\.cache\huggingface\airllm-qwen3moe-grouped")
            tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

            class AIRLLMModel:
                def __init__(self, m, t):
                    self.m = m
                    self.t = t

                def generate(self, prompt, max_new_tokens=256, **kwargs):
                    ids = self.t(prompt, return_tensors="pt")
                    with torch.inference_mode():
                        out_ids = self.m.generate(
                            ids["input_ids"],
                            max_new_tokens=max_new_tokens,
                            eos_token_id=self.t.eos_token_id,
                            do_sample=False,
                            **kwargs,
                        )
                    return self.t.decode(out_ids[0], skip_special_tokens=True)

            model = AIRLLMModel(model_obj, tok)
            print("airllm model loaded")
        except Exception as e:
            print("airllm load failed:", e)
            return None, None, str(e)

    if model is None:
        print("no model provided; cannot evaluate")
        return None, None, "no model"

    report, path = run_eval(model, max_new_tokens=args.max_new_tokens, out_path=args.out)
    print("wrote", path)
    return model, report, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=None, help="HuggingFace model id or local path for AirLLM eval")
    ap.add_argument("--hf-model", default=None, help="local path or HF id for a transformers eval (GPU if available)")
    ap.add_argument("--api-base", default=None, help="OpenAI-compatible API base, e.g. http://127.0.0.1:8080/v1 (Colibri serve)")
    ap.add_argument("--model-id", default=None, help="model id for the API backend")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    _, report, err = load_model_and_eval(args)
    if err:
        print("ERROR:", err)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
