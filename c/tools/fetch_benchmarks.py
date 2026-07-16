"""
Scarica i benchmark LLM standard e li converte nel formato JSONL della harness
({"ctx","choices","gold"} per riga). Da eseguire UNA volta, quando hai rete.
Richiede `datasets`:  pip install --break-system-packages datasets   (o in una venv)

USO:
  python3 tools/fetch_benchmarks.py --out ./bench --tasks hellaswag,arc_challenge,arc_easy,mmlu,winogrande,piqa,openbookqa --limit 300
Poi:
  python3 tools/eval_glm.py --snap /home/vincenzo/glm52_i4 --data ./bench --tasks mmlu --limit 40 --ram 15
"""
import os, sys, json, time, argparse, random

def f_hellaswag(d):
    ctx = (d["activity_label"] + ": " + d["ctx_a"] + " " + d["ctx_b"].capitalize()).strip()
    return ctx, [" " + e.strip() for e in d["endings"]], int(d["label"])
def f_arc(d):
    letters, texts = d["choices"]["label"], d["choices"]["text"]
    return ("Question: " + d["question"].strip() + "\nAnswer:",
            [" " + t.strip() for t in texts], letters.index(d["answerKey"]))
def f_mmlu(d):
    ctx = d["question"].strip() + "\n" + "\n".join(f"{c}. {t}" for c, t in zip("ABCD", d["choices"])) + "\nAnswer:"
    return ctx, [f" {c}" for c in "ABCD"], int(d["answer"])
def f_winogrande(d):
    pre, post = d["sentence"].split("_")
    return pre.strip(), [(" " + o + post).rstrip() for o in (d["option1"], d["option2"])], int(d["answer"]) - 1
def f_piqa(d):
    return "Question: " + d["goal"].strip() + "\nAnswer:", [" " + d["sol1"], " " + d["sol2"]], int(d["label"])
def f_openbookqa(d):
    return d["question_stem"].strip(), [" " + t for t in d["choices"]["text"]], d["choices"]["label"].index(d["answerKey"])

TASKS = {  # task: (path, config, split, formatter)
    "hellaswag":     ("Rowan/hellaswag", None, "validation", f_hellaswag),
    "arc_easy":      ("allenai/ai2_arc", "ARC-Easy", "validation", f_arc),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "validation", f_arc),
    "mmlu":          ("cais/mmlu", "all", "test", f_mmlu),
    "winogrande":    ("allenai/winogrande", "winogrande_xl", "validation", f_winogrande),
    "piqa":          ("ybisk/piqa", None, "validation", f_piqa),
    "openbookqa":    ("allenai/openbookqa", "main", "validation", f_openbookqa),
}

def load_retry(path, cfg, split, tries=5):
    """L'hub restituisce 5xx/timeout transitori (#304): riprova con backoff invece di
    morire al primo HEAD fallito. hf_hub riprende i download parziali dalla cache,
    quindi il retry riparte da dove si era fermato, non da zero.
    EN: the hub throws transient 5xx/timeouts (#304): retry with backoff instead of
    dying on the first failed HEAD. hf_hub resumes partial downloads from its cache,
    so a retry continues where it stopped rather than starting over."""
    from datasets import load_dataset
    for k in range(tries):
        try:
            return load_dataset(path, cfg, split=split)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            if k == tries - 1: raise
            wait = 2 ** (k + 1)
            print(f"  {path}: {type(e).__name__}: {e} — retry {k+1}/{tries-1} in {wait}s")
            time.sleep(wait)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./bench")
    ap.add_argument("--tasks", default="hellaswag,arc_challenge,mmlu")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tries", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    failed = []
    for t in [x.strip() for x in a.tasks.split(",") if x.strip()]:
        if t not in TASKS: print("unknown task:", t); failed.append(t); continue
        path, cfg, split, fn = TASKS[t]
        try:
            ds = load_retry(path, cfg, split, tries=max(a.tries, 1))
        except Exception as e:
            # un task fallito non deve uccidere gli altri / one failed task must not kill the rest
            print(f"{t}: FAILED after {a.tries} tries ({type(e).__name__}: {e}) — skipping")
            failed.append(t); continue
        idx = list(range(len(ds))); random.Random(a.seed).shuffle(idx)
        rows, n = [], 0
        for i in idx:
            try:
                ctx, choices, gold = fn(ds[i])
                if ctx and choices and 0 <= gold < len(choices):
                    rows.append({"ctx": ctx, "choices": choices, "gold": gold}); n += 1
            except Exception: continue
            if n >= a.limit: break
        outp = os.path.join(a.out, t + ".jsonl")
        # scrittura atomica: coli controlla solo l'ESISTENZA del file, quindi un jsonl
        # troncato da un run interrotto bloccherebbe il re-download per sempre.
        # EN: atomic write: coli only checks the file EXISTS, so a truncated jsonl from
        # an interrupted run would block re-download forever.
        tmp = outp + ".part"
        with open(tmp, "w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
        os.replace(tmp, outp)
        print(f"{t}: {len(rows)} -> {outp}")
    if failed:
        print(f"incomplete: {', '.join(failed)} — rerun when the hub recovers (cached progress is kept)")
        sys.exit(1)

if __name__ == "__main__":
    main()
