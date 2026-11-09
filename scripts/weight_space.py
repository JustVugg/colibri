import argparse, json, os, shutil, time
from collections import defaultdict

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)), "results"))

import torch
from safetensors.torch import safe_open, safe_torch_save


def shard_paths(repo):
    root = os.path.join(repo)
    files = sorted([f for f in os.listdir(root) if f.endswith(".safetensors")])
    return [os.path.join(root, f) for f in files]


def count_params(state_dicts):
    n = 0
    for sd in state_dicts:
        for name, t in sd.items():
            n += t.numel()
    return n


def analyze_shards(paths, limit=1):
    out = {
        "shards": [],
        "total_params": 0,
        "expert_tensors": [],
        "total_expert_params": 0,
        "summary": {},
    }
    total = 0
    experts = []
    for p in paths[:limit]:
        sd = {}
        with safe_open(p, framework="torch", device="cpu") as f:
            for name in f.keys():
                sd[name] = f.get_tensor(name)
        n = count_params([sd])
        total += n
        entry = {"path": p, "num_tensors": len(sd), "num_params": int(n)}
        out["shards"].append(entry)
        for name, t in sd.items():
            key = name.lower()
            if "expert" in key or "gate" in key or "moe" in key:
                experts.append(
                    {
                        "name": name,
                        "shape": list(t.shape),
                        "numel": int(t.numel()),
                        "dtype": str(t.dtype),
                        "device": str(t.device),
                    }
                )
    out["total_params"] = int(total)
    out["expert_tensors"] = experts
    out["total_expert_params"] = int(sum(e["numel"] for e in experts))
    out["summary"] = {
        "shards_scanned": min(limit, len(paths)),
        "total_shards": len(paths),
        "first_shard_path": paths[0] if paths else None,
    }
    return out


def build_patch_plan(analysis_path=None, scale=0.85, topk=200):
    import glob
    ws = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)), "models")
                       if False else os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # Prefer whatever was written into results/weight_analysis.json by a prior analyze run
    candidates = []
    if analysis_path and os.path.exists(analysis_path):
        candidates.append(analysis_path)
    for p in glob.glob(os.path.join(RESULTS, "weight_analysis.json")):
        candidates.append(p)
    if not candidates:
        raise FileNotFoundError("No results/weight_analysis.json found. Run analyze first.")
    analysis = json.load(open(candidates[0], encoding="utf-8"))
    shards = analysis.get("expert_tensors", []) or analysis.get("shards", [])
    plan = {
        "source_analysis": candidates[0],
        "scale": scale,
        "topk": topk,
        "edits": [],
    }
    idx = 0
    for tensor_meta in shards:
        if not isinstance(tensor_meta, dict):
            continue
        # tensor_meta may be expert tensor info with name+shape
        if "name" not in tensor_meta:
            continue
        name = tensor_meta["name"]
        plan["edits"].append(
            {
                "idx": idx,
                "name": name,
                "action": "scale",
                "scale": scale,
            }
        )
        idx += 1
        if idx >= topk:
            break
    plan["num_planned_edits"] = len(plan["edits"])
    plan["estimated_params"] = sum(
        (e.get("numel", 0) for e in shards[:topk]) if isinstance(shards[0], dict) else 0
    )
    return plan


def apply_patch(shard_path, plan, out_path=None, dry_run=False):
    if out_path is None:
        base = os.path.splitext(shard_path)[0]
        out_path = base + ".patched.safetensors"
    log_lines = []
    log_lines.append("PATCH PLAN FILE: %s" % plan.get("source_analysis", "<unknown>"))
    log_lines.append("SOURCE SHARD: %s" % shard_path)
    log_lines.append("OUTPUT SHARD: %s" % out_path)
    log_lines.append("dry_run=%s scale=%s topk=%s" % (dry_run, plan.get("scale", 0.85), plan.get("topk", 200)))
    log_lines.append("num_edits=%d" % len(plan.get("edits", [])))

    sd = {}
    with safe_open(shard_path, framework="torch", device="cpu") as f:
        for name in f.keys():
            sd[name] = f.get_tensor(name)

    edits_applied = 0
    for e in plan.get("edits", []):
        name = e["name"]
        if name in sd:
            t = sd[name]
            scale = float(e.get("scale", 1.0))
            if dry_run:
                log_lines.append("DRY-RUN edit: %s shape=%s scale=%.4f numel=%d" % (name, list(t.shape), scale, t.numel()))
            else:
                sd[name] = (t * scale).to(t.dtype)
                edits_applied += 1
                log_lines.append("APPLIED edit: %s shape=%s scale=%.4f numel=%d" % (name, list(t.shape), scale, t.numel()))
        else:
            log_lines.append("SKIP (missing): %s" % name)

    log_lines.append("edits_applied=%d / planned=%d" % (edits_applied, len(plan.get("edits", []))))

    if not dry_run:
        safe_torch_save(sd, out_path)
        log_lines.append("WROTE %s" % out_path)

    log_path = os.path.join(RESULTS, "patch_dryrun.log")
    os.makedirs(RESULTS, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as out:
        out.write("\n".join(log_lines) + "\n")
    print("\n".join(log_lines))
    return out_path


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="action")

    a = sub.add_parser("analyze", help="Analyze safetensors shards for weight-space edits")
    a.add_argument("--repo", required=True, help="Directory containing .safetensors shards")
    a.add_argument("--limit", type=int, default=1)
    a.add_argument("--out", default=None)

    p = sub.add_parser("plan", help="Build a patch plan from weight_analysis.json")
    p.add_argument("--analysis", default=None)
    p.add_argument("--scale", type=float, default=0.85)
    p.add_argument("--topk", type=int, default=200)
    p.add_argument("--out", default=None)

    pp = sub.add_parser("patch", help="Apply patch plan to a safetensors shard")
    pp.add_argument("--shard", required=True)
    pp.add_argument("--plan", required=True, help="results/patch_plan.json")
    pp.add_argument("--out", default=None)
    pp.add_argument("--dry-run", action="store_true")

    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    if args.action == "analyze":
        paths = shard_paths(args.repo)
        if not paths:
            print("No .safetensors shards in", args.repo)
            return
        print("shards:", paths)
        out = analyze_shards(paths, limit=args.limit)
        out_path = args.out or os.path.join(RESULTS, "weight_analysis.json")
        json.dump(out, open(out_path, "w", encoding="utf-8"), indent=2)
        print("wrote", out_path)
        print(json.dumps(out, indent=2))
        return

    if args.action == "plan":
        plan = build_patch_plan(analysis_path=args.analysis, scale=args.scale, topk=args.topk)
        out_path = args.out or os.path.join(RESULTS, "patch_plan.json")
        json.dump(plan, open(out_path, "w", encoding="utf-8"), indent=2)
        print("wrote", out_path)
        print(json.dumps(plan, indent=2))
        return

    if args.action == "patch":
        plan = json.load(open(args.plan, encoding="utf-8"))
        out = apply_patch(args.shard, plan, out_path=args.out, dry_run=args.dry_run)
        return


if __name__ == "__main__":
    main()
