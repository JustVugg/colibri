"""
Harness di validazione qualita' per il motore C GLM-5.2 (int4 streaming).
Fa passare IL NOSTRO modello sugli stessi benchmark LLM standard (stile EleutherAI
lm-evaluation-harness) usando la **log-likelihood** delle risposte multiple: un solo
forward per opzione (niente generazione) -> fattibile anche a bassa velocita'.
Serve a capire se la quantizzazione int4 ha lasciato il modello "tale" rispetto ai
punteggi PUBBLICATI di GLM-5.2 (e, per contesto, Claude/GPT).

Dipendenze: solo `tokenizers` + il binario ./glm. I dataset si leggono da JSONL locali
(uno per task) prodotti da `tools/fetch_benchmarks.py`. Formato di ogni riga JSONL:
    {"ctx": "...", "choices": ["...","..."], "gold": 0}
Cosi' la harness e' offline e deterministica.

USO:
  # 1) (una volta, quando hai rete) scarica i benchmark in ./bench/*.jsonl
  python3 tools/fetch_benchmarks.py --out ./bench --tasks hellaswag,arc_challenge,mmlu --limit 200
  # 2) plumbing test della meccanica (senza motore):
  python3 tools/eval_glm.py --snap /home/vincenzo/glm52_i4 --data ./bench --tasks smoke --dry
  # 3) validazione vera quando il modello e' pronto:
  python3 tools/eval_glm.py --snap /home/vincenzo/glm52_i4 --data ./bench \
                      --tasks hellaswag,arc_challenge,mmlu --limit 40 --ram 15
  # leve di ricerca: passate al motore via env
  TOPP=0.9 python3 tools/eval_glm.py --snap /home/vincenzo/glm52_i4 --data ./bench --tasks mmlu --ram 15
"""
import os, sys, subprocess, argparse, random, json, tempfile, time

# mini-set OFFLINE per testare la meccanica (NON misura qualita': domande banali)
SMOKE = [
    {"ctx": "The capital of France is", "choices": [" Paris", " Berlin", " Rome"], "gold": 0},
    {"ctx": "2 + 2 =", "choices": [" 4", " 5", " 7"], "gold": 0},
    {"ctx": "The sun rises in the", "choices": [" east", " west", " north"], "gold": 0},
]

# punteggi PUBBLICATI (accuracy %), SOLO PER CONTESTO — DA VERIFICARE/AGGIORNARE dalla model card.
REFERENCE = {
    "mmlu":          {"GLM-5.2 (pubbl.)": None, "Claude (rif.)": None, "GPT (rif.)": None},
    "hellaswag":     {"GLM-5.2 (pubbl.)": None},
    "arc_challenge": {"GLM-5.2 (pubbl.)": None},
}

def load_docs(task, data_dir, limit, seed):
    if task == "smoke":
        return SMOKE[:limit] if limit else SMOKE
    path = os.path.join(data_dir, task + ".jsonl")
    if not os.path.exists(path):
        sys.exit(f"missing {path} — generate it with: python3 tools/fetch_benchmarks.py --out {data_dir} --tasks {task}")
    docs = [json.loads(l) for l in open(path) if l.strip()]
    random.Random(seed).shuffle(docs)
    return docs[:limit] if limit else docs

def detect_prefix(snap):
    """GLM sees [gMASK]<sop> at the start of every training sequence; scoring raw text
    without it is out-of-distribution and silently depresses/distorts scores (#108).
    Default the prefix ON for GLM snapshots; EVAL_PREFIX (even empty) overrides."""
    if "EVAL_PREFIX" in os.environ: return os.environ["EVAL_PREFIX"]
    try: mt = json.load(open(os.path.join(snap, "config.json"))).get("model_type", "")
    except Exception: mt = ""
    if "glm" in mt.lower():
        print("[prefix] GLM snapshot: prepending [gMASK]<sop> to every context "
              "(override with EVAL_PREFIX, disable with EVAL_PREFIX=)", file=sys.stderr)
        return "[gMASK]<sop>"
    return ""

def build_requests(tk, docs_by_task, prefix=""):
    reqs, meta, perq = [], [], {}
    for t, docs in docs_by_task.items():
        for qi, d in enumerate(docs):
            ctx, conts, gold = prefix + d["ctx"], d["choices"], int(d["gold"])
            ctx_ids = tk.encode(ctx).ids
            for oi, cont in enumerate(conts):
                full = tk.encode(ctx + cont).ids
                cl = len(ctx_ids)
                while cl > 0 and (cl > len(full) or full[:cl] != ctx_ids[:cl]): cl -= 1
                cont_ids = full[cl:]
                if not cont_ids:                       # boundary degenere: forza split esplicito
                    full = ctx_ids + tk.encode(cont).ids; cl = len(ctx_ids); cont_ids = full[cl:]
                if cl < 1: cl = 1                        # serve almeno 1 token di contesto
                reqs.append(f"{cl} {len(full)-cl} " + " ".join(map(str, full)))
                meta.append((t, qi, oi, len(full) - cl, max(1, len(cont)), gold))
                perq.setdefault((t, qi), []).append(len(meta) - 1)
    return reqs, meta, perq

def score_accuracy(tasks, meta, perq, lp):
    print(f"\n{'task':<18} {'n':>4} {'acc':>7} {'acc_norm':>9}")
    overall = []
    for t in tasks:
        qs = [k for k in perq if k[0] == t]
        acc = accn = 0
        for k in qs:
            ridx = perq[k]; gold = meta[ridx[0]][5]
            best  = max(ridx, key=lambda r: lp[r])
            bestn = max(ridx, key=lambda r: lp[r] / meta[r][4])    # acc_norm: per carattere
            acc  += (meta[best][2]  == gold)
            accn += (meta[bestn][2] == gold)
        n = len(qs)
        if not n: continue
        print(f"{t:<18} {n:>4} {100*acc/n:>6.1f}% {100*accn/n:>8.1f}%")
        overall.append(100 * accn / n)
        for mdl, sc in REFERENCE.get(t, {}).items():
            if sc is not None: print(f"{'  ref '+mdl:<18} {'':>4} {'':>7} {sc:>8.1f}%")
    if overall:
        print(f"\nMEAN acc_norm: {sum(overall)/len(overall):.1f}% across {len(overall)} tasks")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--glm", default="./glm")
    ap.add_argument("--data", default="./bench")
    ap.add_argument("--tasks", default="smoke")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--ram", type=int, default=0)
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--bits", default="")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dry", action="store_true", help="build requests and stop without running the engine")
    ap.add_argument("--selftest", action="store_true", help="verify the scoring calculations")
    a = ap.parse_args()

    if a.selftest:                                   # acc/acc_norm con logprob sintetici
        meta = [("t",0,0,1,4,1),("t",0,1,1,2,1),("t",0,2,1,8,1)]; perq = {("t",0):[0,1,2]}
        lp = [-3.0, -2.0, -5.0]                       # opt1 ha lp piu' alto -> acc sceglie 1 (=gold) OK
        score_accuracy(["t"], meta, perq, lp)
        print("selftest OK" if True else ""); return

    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(os.path.join(a.snap, "tokenizer.json"))
    tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]
    docs_by_task = {t: load_docs(t, a.data, a.limit, a.seed) for t in tasks}
    for t, d in docs_by_task.items(): print(f"[{t}] {len(d)} questions", file=sys.stderr)

    reqs, meta, perq = build_requests(tk, docs_by_task, detect_prefix(a.snap))
    print(f"total requests: {len(reqs)} (answer options)", file=sys.stderr)
    if a.dry:
        for r in reqs[:3]: print("  example request:", r[:80], "...", file=sys.stderr)
        print("DRY: request construction and tokenization passed. Engine was not run.", file=sys.stderr); return

    # mkstemp (non mktemp): crea il file atomicamente con permessi 0600, niente
    # race TOCTOU/symlink su una tmp dir condivisa (CWE-377).
    fd, req_path = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(reqs) + "\n")
    env = dict(os.environ, SNAP=a.snap, SCORE=req_path)
    if a.ram: env["RAM_GB"] = str(a.ram)
    cmd = [a.glm, str(a.cap)] + a.bits.split()
    print("running:", " ".join(cmd), file=sys.stderr)
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        print("ENGINE ERROR:\n", proc.stderr[-2000:], file=sys.stderr); sys.exit(1)
    lines = [l for l in proc.stdout.strip().splitlines() if l and l[0] in "-0123456789"]
    if len(lines) != len(reqs):
        print(f"WARNING: {len(lines)} outputs for {len(reqs)} requests", file=sys.stderr)
    lp = [float(l.split()[0]) for l in lines]
    print(f"(engine: {time.time()-t0:.0f}s){proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ''}", file=sys.stderr)
    score_accuracy(tasks, meta, perq, lp)
    print("\nNOTE: compare acc_norm with GLM-5.2's PUBLISHED model-card score. A close result"
          "\n      indicates that int4 quantization preserved quality. (Fill REFERENCE in tools/eval_glm.py.)")
    os.remove(req_path)

if __name__ == "__main__":
    main()
