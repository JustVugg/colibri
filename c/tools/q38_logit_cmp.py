#!/usr/bin/env python3
"""Compare two qwen38 teacher-forced logit dumps (Q38_LOGITS_DUMP) and give a tolerance verdict.

The dump: PPL=1 with a ref.json whose full_ids extend prompt_ids scores every position
past the prompt with the forced (reference) tokens fed back, so two runs see the same
inputs whatever either predicts (qwen38.c, tf_nll). Q38_LOGITS_DUMP=<file> writes one
line a scored position:

    <position> <NLL of the forced token> <log Z> <id>:<logit> ... (the top 32, best first)

where log Z = logsumexp over the whole vocabulary, so logit - log Z is an exact log
probability of that run.

Per position, A the baseline and B the candidate:
  top-1      the two runs' best ids agree;
  KL         KL(A || B) over the ids both top-32 lists hold, each run's probabilities
             renormalized on that shared support (the dump has nothing past the top 32);
  tail       A's probability mass outside the shared support (what that KL cannot see);
  max |dlogit| over the shared ids; |dNLL| of the forced token, an exact full-vocabulary
             quantity; |dlogZ|.

Criteria:
  strict   top-1 >= 99%, mean KL < 1e-3, max KL < 0.05   (a reduction order changed)
  loose    top-1 >= 99%, mean KL < 5e-3                  (a precision change: KV fp16/int8)
  longctx  top-1 >= 99%, mean KL <= 0.01                 (long-context stages)
--top1/--mean-kl/--max-kl override a criterion's numbers.

Usage:
  python3 tools/q38_logit_cmp.py BASE.txt CAND.txt [BASE2.txt CAND2.txt ...] [--criterion strict] [--json]
Several pairs (one per prompt) are pooled position by position. Exit 0: within tolerance,
1: outside it, 2: the dumps do not line up (other positions, a malformed line).
"""
import argparse
import json
import math
import sys

CRITERIA = {
    "strict": {"top1": 0.99, "mean_kl": 1e-3, "max_kl": 0.05, "mean_kl_inclusive": False},
    "loose": {"top1": 0.99, "mean_kl": 5e-3, "max_kl": None, "mean_kl_inclusive": False},
    "longctx": {"top1": 0.99, "mean_kl": 1e-2, "max_kl": None, "mean_kl_inclusive": True},
}


class DumpError(ValueError):
    pass


def parse_dump(path):
    """{position: (nll, logz, [(id, logit), ...])} in the file's order (best first)."""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            parts = line.split()
            if not parts:
                continue
            try:
                pos, nll, logz = int(parts[0]), float(parts[1]), float(parts[2])
                top = []
                for item in parts[3:]:
                    i, v = item.split(":", 1)
                    top.append((int(i), float(v)))
            except (ValueError, IndexError) as e:
                raise DumpError(f"{path}:{n}: not a Q38_LOGITS_DUMP line ({e})")
            if not top:
                raise DumpError(f"{path}:{n}: no top-k entries")
            if pos in out:
                raise DumpError(f"{path}:{n}: position {pos} twice")
            out[pos] = (nll, logz, top)
    if not out:
        raise DumpError(f"{path}: empty dump")
    return out


def _logsumexp(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def compare_position(a, b):
    """Metrics of one position: a, b = (nll, logz, top)."""
    (na, za, ta), (nb, zb, tb) = a, b
    da, db = dict(ta), dict(tb)
    shared = [i for i, _ in ta if i in db]
    m = {"top1": ta[0][0] == tb[0][0], "dnll": abs(na - nb), "dlogz": abs(za - zb), "shared": len(shared)}
    if not shared:
        m.update(kl=math.inf, tail=1.0, max_dlogit=math.inf)
        return m
    la = [da[i] for i in shared]; lb = [db[i] for i in shared]
    sa, sb = _logsumexp(la), _logsumexp(lb)
    kl = sum(math.exp(x - sa) * ((x - sa) - (y - sb)) for x, y in zip(la, lb))
    m["kl"] = max(0.0, kl)                         # rounding can leave -1e-17
    m["tail"] = max(0.0, 1.0 - math.exp(sa - za))  # A's mass the shared ids do not hold
    m["max_dlogit"] = max(abs(x - y) for x, y in zip(la, lb))
    return m


def compare(pairs):
    """pairs: [(base dump, candidate dump)] -> (rows, summary). Raises DumpError when a pair's
    positions differ."""
    rows = []
    for k, (a, b) in enumerate(pairs):
        if set(a) != set(b):
            only_a = sorted(set(a) - set(b))[:5]; only_b = sorted(set(b) - set(a))[:5]
            raise DumpError(f"pair {k}: positions differ (only in base {only_a}, only in candidate {only_b})")
        for pos in sorted(a):
            r = compare_position(a[pos], b[pos]); r.update(pair=k, pos=pos)
            rows.append(r)
    n = len(rows)
    kls = [r["kl"] for r in rows]
    summary = {
        "positions": n,
        "top1_agree": sum(r["top1"] for r in rows) / n,
        "top1_disagree_at": [(r["pair"], r["pos"]) for r in rows if not r["top1"]][:20],
        "mean_kl": sum(kls) / n,
        "max_kl": max(kls),
        "max_kl_at": max(rows, key=lambda r: r["kl"])["pos"],
        "max_tail": max(r["tail"] for r in rows),
        "max_dlogit": max(r["max_dlogit"] for r in rows),
        "mean_dnll": sum(r["dnll"] for r in rows) / n,
        "max_dnll": max(r["dnll"] for r in rows),
        "max_dlogz": max(r["dlogz"] for r in rows),
        "min_shared": min(r["shared"] for r in rows),
    }
    return rows, summary


def verdict(summary, crit):
    """[(check, ok, detail)] of a criterion dict"""
    out = [("top-1", summary["top1_agree"] >= crit["top1"], f"{summary['top1_agree']:.4f} >= {crit['top1']}")]
    if crit.get("mean_kl_inclusive"):
        out.append(("mean KL", summary["mean_kl"] <= crit["mean_kl"], f"{summary['mean_kl']:.3e} <= {crit['mean_kl']:g}"))
    else:
        out.append(("mean KL", summary["mean_kl"] < crit["mean_kl"], f"{summary['mean_kl']:.3e} < {crit['mean_kl']:g}"))
    if crit.get("max_kl") is not None:
        out.append(("max KL", summary["max_kl"] < crit["max_kl"], f"{summary['max_kl']:.3e} < {crit['max_kl']:g}"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dumps", nargs="+", help="BASE CAND [BASE2 CAND2 ...]")
    ap.add_argument("--criterion", choices=sorted(CRITERIA), default="strict")
    ap.add_argument("--top1", type=float, help="override: minimum top-1 agreement (0..1)")
    ap.add_argument("--mean-kl", type=float, help="override: mean KL bound")
    ap.add_argument("--max-kl", type=float, help="override: max KL bound (a negative value: no bound)")
    ap.add_argument("--json", action="store_true", help="the summary and verdict as one JSON object on stdout")
    ap.add_argument("--worst", type=int, default=5, help="print the N positions with the largest KL")
    args = ap.parse_args(argv)
    if len(args.dumps) % 2:
        ap.error("dumps come in pairs: BASE CAND [BASE2 CAND2 ...]")
    crit = dict(CRITERIA[args.criterion])
    if args.top1 is not None: crit["top1"] = args.top1
    if args.mean_kl is not None: crit["mean_kl"] = args.mean_kl
    if args.max_kl is not None: crit["max_kl"] = None if args.max_kl < 0 else args.max_kl
    try:
        pairs = [(parse_dump(args.dumps[i]), parse_dump(args.dumps[i + 1])) for i in range(0, len(args.dumps), 2)]
        rows, s = compare(pairs)
    except (DumpError, OSError) as e:
        print(f"q38_logit_cmp: {e}", file=sys.stderr)
        return 2
    checks = verdict(s, crit)
    ok = all(c[1] for c in checks)
    if args.json:
        print(json.dumps({"criterion": args.criterion, "thresholds": crit, "summary": s,
                          "checks": [{"check": c, "ok": o, "detail": d} for c, o, d in checks], "pass": ok}))
    else:
        print(f"{s['positions']} positions in {len(pairs)} pair(s), criterion {args.criterion}")
        print(f"  top-1 agreement {s['top1_agree'] * 100:.2f}%"
              + (f" (differs at {s['top1_disagree_at']})" if s["top1_disagree_at"] else ""))
        print(f"  KL(base||cand) on the shared top-32: mean {s['mean_kl']:.3e}, max {s['max_kl']:.3e} (position {s['max_kl_at']})")
        print(f"  base mass outside the shared ids: max {s['max_tail']:.3e}; fewest shared ids {s['min_shared']}")
        print(f"  max |dlogit| {s['max_dlogit']:.4g}; forced-token |dNLL| mean {s['mean_dnll']:.3e} max {s['max_dnll']:.3e}; "
              f"max |dlogZ| {s['max_dlogz']:.3e}")
        for r in sorted(rows, key=lambda r: -r["kl"])[:max(0, args.worst)]:
            print(f"    pair {r['pair']} pos {r['pos']}: KL {r['kl']:.3e} top-1 {'=' if r['top1'] else '!='} "
                  f"max|dlogit| {r['max_dlogit']:.4g} |dNLL| {r['dnll']:.3e}")
        for c, o, d in checks:
            print(f"  {'PASS' if o else 'FAIL'} {c}: {d}")
        print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
