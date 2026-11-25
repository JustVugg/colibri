#!/usr/bin/env python3
"""Drawn MTP drafts (spec_exact.h, COLI_SPEC_SOFT=1) on qwen38 and qwen36: the sampled
tokens follow the same distribution as plain decoding.

Byte-identity with plain decoding (tests/qwen38_mtp_harness.py, tests/qwen36_mtp_harness.py)
holds for argmax drafts only: a drawn draft is kept by min(1, p/q) and replaced by a draw
from the residual, so the tokens match plain decoding in distribution, not draw for draw.
This drives one serve session per mode -- plain and drawn drafts (COLI_SPEC_SOFT=1) -- over
N sampled turns of the same prompt, and compares the per-position frequency of the generated
text with a two-sample chi-square test (z > 4 fails). It also checks that drafts were drawn
and settled, so a run that silently fell back to argmax drafts fails.

    python tests/qwen38_spec_soft_harness.py --engine ./qwen38 --model ./qwen38_tiny_mtp [--turns 3000]
    python tests/qwen38_spec_soft_harness.py --family qwen36 --engine ./qwen36 --model build/q36soft/m --turns 20000

make qwen38-spec-soft-check and make qwen36-spec-soft-check build their fixtures and run it.
"""
import argparse
import collections
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

BASE_ENV = {"OMP_NUM_THREADS": "2", "COLI_NO_OMP_TUNE": "1", "COLI_CUDA": "0", "Q38_TRUNK_GPU": "0",
            "NOSTREAM": "1", "USAGE_SAVE": "0", "COLI_LOOKUP": "0", "Q38_MTP_DRAFTS": "2"}
SCRUB = ("Q38_MTP", "Q38_MTP_FORCE", "Q38_MTP_DRAFTS", "COLI_LOOKUP", "COLI_SPEC_SOFT", "Q38_MTP_DUMP",
         "DUMP", "Q36_MTP", "Q36_MTP_FORCE", "Q36_MTP_DRAFTS")
PROMPT = b"abcab"
TOKENS = 4

FAMILIES = {
    # argv, the env of every run, the env of plain / drawn runs, the per-turn stats line
    "qwen38": (["1", "8"], {}, {"Q38_MTP": "0"}, {"Q38_MTP": "1", "COLI_SPEC_SOFT": "1"},
               r"\[qwen38 MTP sampled\][^\n]*drawn drafts kept (\d+)/(\d+)"),
    # qwen36 on tests/basalt_fixture.py's converted pack (its nextn MTP head, CPU)
    "qwen36": (["2", "8"], {"Q36_MAXT": "64", "COLI_CUDA": "0", "COLI_VULKAN": "0"}, {"Q36_MTP": "0"},
               {"Q36_MTP": "1", "COLI_SPEC_SOFT": "1"},
               r"\[qwen36 mtp sampled\][^\n]*drawn drafts kept (\d+)/(\d+)"),
}


def session(engine, model, extra, turns, temp, top_p, family="qwen38"):
    argv, fam_env = FAMILIES[family][0], FAMILIES[family][1]
    env = {k: v for k, v in os.environ.items() if k not in SCRUB}
    env.update(BASE_ENV)
    env.update(fam_env)
    env.update(extra, SNAP=str(model), SERVE="1", TOK=str(model / "tokenizer.json"), Q38_MAXT="64")
    texts = []
    with tempfile.TemporaryFile() as err:
        p = subprocess.Popen([str(Path(engine).resolve())] + argv, env=env, stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=err)

        def frame():
            line = p.stdout.readline()
            if not line:
                raise EOFError("engine closed")
            fields = line.replace(b"\x01", b"").split()
            payload = b""
            if fields and fields[0] in (b"DATA", b"ECHO"):
                payload = p.stdout.read(int(fields[2]))
                p.stdout.read(1)
            return fields, payload

        while frame()[0][:1] != [b"READY"]:
            pass
        for t in range(turns):
            p.stdin.write(f"SUBMIT r{t} 0 {len(PROMPT)} {TOKENS} {temp} {top_p}\n".encode() + PROMPT + b"\n")
            p.stdin.flush()
            out = b""
            while True:
                fields, payload = frame()
                if not fields:
                    continue
                if fields[0] == b"DATA":
                    out += payload
                if fields[0] in (b"DONE", b"ERROR"):
                    if fields[0] == b"ERROR":
                        raise RuntimeError(b" ".join(fields).decode(errors="replace"))
                    break
            texts.append(out)
        p.stdin.close()
        rc = p.wait(timeout=120)
        err.seek(0)
        return rc, texts, err.read().decode(errors="replace")


def tv_by_position(a, b):
    """Per position: the TV of the two histograms, its same-law expectation, and the z of a
    two-sample chi-square (sum (a_k - b_k)^2 / (a_k + b_k) for equal sizes, df = cells - 1,
    Wilson-Hilferty to a normal). z > 4 is p < 3e-5 under the same law."""
    rows = []
    for pos in range(TOKENS):
        ca = collections.Counter(x[pos:pos + 1] for x in a)
        cb = collections.Counter(x[pos:pos + 1] for x in b)
        keys = set(ca) | set(cb)
        tv = 0.5 * sum(abs(ca[k] / len(a) - cb[k] / len(b)) for k in keys)
        pooled = {k: (ca[k] + cb[k]) / (len(a) + len(b)) for k in keys}
        null = 0.5 * sum((2 / 3.141592653589793) ** 0.5 *
                         (q * (1 - q) * (1 / len(a) + 1 / len(b))) ** 0.5 for q in pooled.values())
        chi = sum((ca[k] - cb[k]) ** 2 / (ca[k] + cb[k]) for k in keys)
        df = max(len(keys) - 1, 1)
        z = ((chi / df) ** (1 / 3) - (1 - 2 / (9 * df))) / (2 / (9 * df)) ** 0.5
        rows.append((tv, null, z))
        print(f"  position {pos}: TV {tv:.4f}, same-law expectation {null:.4f}, chi-square z {z:+.2f}, "
              f"{len(keys)} distinct outputs")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--turns", type=int, default=3000)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--family", choices=sorted(FAMILIES), default="qwen38")
    args = ap.parse_args()
    _, _, plain_env, drawn_env, stats_re = FAMILIES[args.family]
    rc0, plain, _ = session(args.engine, args.model, plain_env, args.turns, args.temp, args.top_p, args.family)
    rc1, drawn, log = session(args.engine, args.model, drawn_env, args.turns, args.temp, args.top_p, args.family)
    fails = []
    if rc0 or rc1:
        fails.append(f"engine exit codes {rc0}/{rc1}")
    stats = re.findall(stats_re, log)
    settled = max((int(n) for _, n in stats), default=0)
    kept = max((int(k) for k, _ in stats), default=0)
    print(f"drawn drafts settled {settled}, kept {kept}")
    least = args.turns if args.family == "qwen38" else args.turns // 10
    if settled < least:
        fails.append(f"only {settled} drawn drafts were settled over {args.turns} turns")
    rows = tv_by_position(plain, drawn)
    for pos, (tv, null, z) in enumerate(rows):
        if z > 4.0:
            fails.append(f"position {pos}: chi-square z {z:.2f} > 4 (TV {tv:.4f}, same-law {null:.4f})")
    if fails:
        print("FAIL: " + "; ".join(fails))
        sys.exit(1)
    print(f"{args.family} drawn drafts: sampled tokens follow plain decoding's distribution")


if __name__ == "__main__":
    main()
