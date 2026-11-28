# Step 1 re-measure: the needle test as a timed, streaming run against any OpenAI-style server.
#   build: python3 -I nbench.py build <llama_url> <variant> <target>... -> prompts/v<variant>-<target>.txt (llama's tokenizer)
#   run:   python3 -I nbench.py run <base_url> <model> <prompt_file> <variant> <out.json> [max_tokens]
# Records TTFT (= prefill), prompt tok/s, decode tok/s from the stream, the server's own timings if it sends
# them, and the needle grade (same notes, questions and checks as ../needle.py).
import json, re, sys, time, urllib.request, os
sys.argv[0:1] = []
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "..", "needle.py"), encoding="utf-8").read()
ns = {"re": re}
exec(src[src.index("SETS = ["):src.index("S = SETS[VARIANT]")], ns)   # the needle sets only
SETS = ns["SETS"]
HAY = os.path.join(HERE, "..", "haystack.txt")
head = "Below is a long document. Somewhere inside it are a few lines that start with \"FIELD NOTE\". Read the whole document carefully, then answer the questions after it.\n\n<document>\n"

def post(url, body, timeout):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)

def build(url, variant, targets):
    S = SETS[variant]; hay = open(HAY, encoding="utf-8").read()
    qtext = "\n".join(f"Q{i+1}: {q}" for i, (q, _) in enumerate(S["questions"]))
    tail = ("\n</document>\n\nQuestions:\n" + qtext +
            "\n\nAnswer every question using only the document. Reply with exactly one line per question, formatted as \"A1: ...\", \"A2: ...\" and so on.")
    ntok = lambda t: len(json.load(post(url + "/tokenize", {"content": t}, 600))["tokens"])
    def make(nc):
        body = hay[:nc]
        for depth, note in sorted(S["notes"], reverse=True):
            pos = body.rfind("\n", 0, int(len(body) * depth)) + 1
            body = body[:pos] + note + "\n" + body[pos:]
        return head + body + tail
    cpt = 200000 / ntok(hay[:200000])
    os.makedirs(os.path.join(HERE, "prompts"), exist_ok=True)
    for T in targets:
        nc = int(T * cpt); p = make(nc); n = ntok(p); nc = int(nc * T / n); p = make(nc); n = ntok(p)
        f = os.path.join(HERE, "prompts", f"v{variant}-{T}.txt"); open(f, "w", encoding="utf-8").write(p)
        print(f"{f}: {n} tokens (llama tokenizer)", flush=True)

def run(url, model, pfile, variant, out, max_tokens):
    S = SETS[variant]; prompt = open(pfile, encoding="utf-8").read()
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "stream": True, "stream_options": {"include_usage": True}, "cache_prompt": False}
    body.update(json.loads(os.environ.get("NB_EXTRA") or "{}"))
    t0 = time.time(); t1 = None; reasoning = []; content = []; usage = {}; timings = {}; finish = None; nchunk = 0
    for line in post(url + "/v1/chat/completions", body, 6 * 3600):
        line = line.decode("utf-8", "replace").strip()
        if not line.startswith("data:") or line == "data: [DONE]": continue
        d = json.loads(line[5:])
        usage = d.get("usage") or usage; timings = d.get("timings") or timings
        for c in d.get("choices") or []:
            dl = c.get("delta") or {}; r = dl.get("reasoning_content") or ""; s = dl.get("content") or ""
            if (r or s) and t1 is None: t1 = time.time()
            if r or s: nchunk += 1
            reasoning.append(r); content.append(s); finish = c.get("finish_reason") or finish
    t2 = time.time(); t1 = t1 or t2
    text = "".join(content)
    answers = {}
    for m in re.finditer(r"A(\d)\s*[:.)-]\s*(.*)", text): answers.setdefault(int(m.group(1)), m.group(2).strip())
    grades = [bool(check(answers.get(i + 1, ""))) for i, (_, check) in enumerate(S["questions"])]
    pt = usage.get("prompt_tokens") or timings.get("prompt_n") or 0
    ct = usage.get("completion_tokens") or timings.get("predicted_n") or nchunk
    res = {"model": model, "prompt_file": os.path.basename(pfile), "prompt_tokens": pt, "variant": variant,
           "score": sum(grades), "of": len(grades), "grades": grades,
           "prefill_s": round(t1 - t0, 2), "prompt_tok_s": round(pt / (t1 - t0), 1) if pt else None,
           "completion_tokens": ct, "decode_s": round(t2 - t1, 2),
           "decode_tok_s": round((ct - 1) / (t2 - t1), 2) if ct > 1 and t2 > t1 else None,
           "server_pp_tok_s": round(timings.get("prompt_per_second", 0), 1) or None,
           "server_tg_tok_s": round(timings.get("predicted_per_second", 0), 2) or None,
           "finish": finish, "wall_s": round(t2 - t0, 1), "reasoning_chars": len("".join(reasoning))}
    json.dump({"result": res, "answers": answers, "content": text, "reasoning": "".join(reasoning)}, open(out, "w"), indent=1)
    print(json.dumps(res), flush=True)

if sys.argv[0] == "build": build(sys.argv[1], int(sys.argv[2]), [int(x) for x in sys.argv[3:]])
else: run(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5], int(sys.argv[6]) if len(sys.argv) > 6 else 6000)
