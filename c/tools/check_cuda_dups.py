#!/usr/bin/env python3
"""Report device helpers copied across the CUDA sources (make check-cuda-dups).

Every __device__ function definition in the product .cu/.cuh files (and the
headers holding device code) is fingerprinted on its tokens, with comments,
whitespace and its own name left out. Two reports:

  same name   a helper defined in more than one file: identical copies, or
              DRIFTED when the copies differ (the case worth a look: one copy
              got a fix the others did not)
  same body   identical bodies under different names
  same shape  bodies equal up to the names of their variables and callees
              (alpha-renamed), under different names, not already identical
  families    every helper whose name says it is one of the usual
              primitives (warp/block sums, e4m3/fp8 and e2m1/fp4 decode,
              bf16 rounding), with its shape: copies with several shapes
              have drifted apart (or were never the same primitive)

A report, not a gate: exit 0 unless --strict and something drifted. The idea
and the tokenizer follow TensorFold's tools/zig/cuda_common_check.py
(Apache-2.0), which gates copies of shared CUDA primitives outside its common
headers against an allowlist.

stdlib only.
"""

import argparse
import hashlib
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parent
TOKEN = re.compile(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[A-Za-z_]\w*|\d+(?:\.\d*)?[fFuUlL]*|[^\s]')


def tokens(text):
    """C++ tokens with comments dropped; preprocessor lines dropped too (a
    #define inside a body would otherwise end the declaration scan)."""
    out = []
    for m in TOKEN.finditer(text):
        v = m.group()
        if v.startswith(("//", "/*")):
            continue
        out.append((v, m.start()))
    return out


KEEP = set("""
if else for while do return break continue switch case default goto sizeof const volatile static inline
unsigned signed int float double char short long bool void auto true false struct union enum template
typename uint8_t uint16_t uint32_t uint64_t int8_t int16_t int32_t int64_t size_t half half2 float2 float4
uint2 uint4 int2 int4 constexpr threadIdx blockIdx blockDim gridDim x y z
""".split())

FAMILIES = [
    ("warp/block sum", re.compile(r"(?i)(warp|block|blk)_?(sum|reduce|red)|wsum|bsum")),
    ("e4m3/fp8 decode", re.compile(r"(?i)e4m3|fp8")),
    ("e2m1/fp4 decode", re.compile(r"(?i)e2m1|fp4|mxfp")),
    ("bf16 round", re.compile(r"(?i)bf16")),
]


def shape(body):
    """Alpha-renamed tokens: keywords, types, CUDA builtins (__*) and literals
    stay, every other identifier becomes its first-appearance index."""
    names, out = {}, []
    for b in body:
        if re.fullmatch(r"[A-Za-z_]\w*", b) and b not in KEEP and not b.startswith("__"):
            out.append("v%d" % names.setdefault(b, len(names)))
        else:
            out.append(b)
    return hashlib.sha256(json.dumps(out, separators=(",", ":")).encode()).hexdigest()[:12]


def closing(ts, i, left, right):
    depth = 0
    for j in range(i, len(ts)):
        depth += (ts[j][0] == left) - (ts[j][0] == right)
        if depth == 0:
            return j
    return -1


def definitions(path):
    """(name, line, arity, fingerprint, size) of each __device__ definition."""
    text = path.read_text(errors="replace")
    text = re.sub(r"^\s*#[^\n]*(?:\\\n[^\n]*)*", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.M)
    ts = tokens(text)
    out = []
    i = 0
    while i < len(ts) - 1:
        v = ts[i][0]
        if not re.fullmatch(r"[A-Za-z_]\w*", v) or ts[i + 1][0] != "(":
            i += 1
            continue
        end = closing(ts, i + 1, "(", ")")
        if end < 0:
            break
        j = end + 1
        while j < len(ts) and ts[j][0] in ("const", "noexcept", "override"):
            j += 1
        if j >= len(ts) or ts[j][0] != "{":
            i += 1
            continue
        start = i
        while start > 0 and ts[start - 1][0] not in (";", "{", "}", ")"):
            start -= 1
        decl = [t[0] for t in ts[start:i]]
        # template<...> heads end with '>' after a ')' in default args only rarely; good enough
        if "__device__" not in decl or "__global__" in decl or "__launch_bounds__" in decl:
            i += 1
            continue
        last = closing(ts, j, "{", "}")
        if last < 0:
            break
        params = ts[i + 2:end]
        arity = 0 if not params or [p[0] for p in params] == ["void"] else 1 + sum(1 for p in params if p[0] == ",")
        body = [t[0] if k != i - start else "@" for k, t in enumerate(ts[start:last + 1])]
        body = [b for b in body if b not in ("static", "inline", "__forceinline__", "__inline__")]
        fp = hashlib.sha256(json.dumps(body, separators=(",", ":")).encode()).hexdigest()[:12]
        out.append((v, text.count("\n", 0, ts[i][1]) + 1, arity, fp, last - j, shape(body)))
        i = last + 1
    return out


def scan(files):
    defs = []
    for f in files:
        for name, line, arity, fp, size, shp in definitions(f):
            defs.append({"file": f.relative_to(SRC).as_posix(), "name": name, "line": line,
                         "arity": arity, "fp": fp, "size": size, "shape": shp})
    return defs


def product_files():
    files = sorted(SRC.glob("*.cu")) + sorted(SRC.glob("*.cuh"))
    files += [h for h in sorted(SRC.glob("*.h")) if "__device__" in h.read_text(errors="replace")]
    return [f for f in files if f.exists()]


def report(defs, min_tokens):
    by_name = {}
    for d in defs:
        by_name.setdefault((d["name"], d["arity"]), []).append(d)
    drifted, same = [], []
    for (name, arity), ds in sorted(by_name.items()):
        if len({d["file"] for d in ds}) < 2:
            continue
        fps = {d["fp"] for d in ds}
        (drifted if len(fps) > 1 else same).append((name, arity, ds))
    by_fp = {}
    for d in defs:
        if d["size"] >= min_tokens:
            by_fp.setdefault(d["fp"], []).append(d)
    renamed = [ds for ds in by_fp.values() if len({d["name"] for d in ds}) > 1]
    by_shape = {}
    for d in defs:
        if d["size"] >= min_tokens:
            by_shape.setdefault(d["shape"], []).append(d)
    shaped = [ds for ds in by_shape.values()
              if len({d["name"] for d in ds}) > 1 and len({d["fp"] for d in ds}) > 1]
    families = []
    for label, rx in FAMILIES:
        ms = [d for d in defs if rx.search(d["name"])]
        if len(ms) > 1:
            families.append((label, ms))
    return drifted, same, renamed, shaped, families


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="*", type=pathlib.Path, help="default: the product CUDA sources")
    ap.add_argument("--min-tokens", type=int, default=12, help="smallest body for the same-body report")
    ap.add_argument("--strict", action="store_true", help="exit 1 when a same-name helper drifted")
    a = ap.parse_args()
    files = [f.resolve() for f in a.files] or product_files()
    defs = scan(files)
    drifted, same, renamed, shaped, families = report(defs, a.min_tokens)
    for name, arity, ds in drifted:
        groups = {}
        for d in ds:
            groups.setdefault(d["fp"], []).append(f"{d['file']}:{d['line']}")
        print(f"DRIFTED {name}/{arity}: {len(groups)} variants")
        for fp, where in groups.items():
            print(f"    {fp}: {', '.join(where)}")
    for name, arity, ds in same:
        print(f"identical {name}/{arity}: {', '.join(d['file'] + ':' + str(d['line']) for d in ds)}")
    for ds in renamed:
        print("same body: " + ", ".join(f"{d['name']} {d['file']}:{d['line']}" for d in ds))
    for ds in shaped:
        print("same shape: " + ", ".join(f"{d['name']} {d['file']}:{d['line']}" for d in ds))
    for label, ms in families:
        print(f"family {label}: {len(ms)} helpers in {len({d['file'] for d in ms})} files, "
              f"{len({d['shape'] for d in ms})} shapes")
        for d in sorted(ms, key=lambda d: (d["shape"], d["file"], d["line"])):
            print(f"    {d['shape']} {d['name']}/{d['arity']} {d['file']}:{d['line']}")
    print(f"SUMMARY cuda dups: {len(files)} files, {len(defs)} device helpers, {len(drifted)} drifted, "
          f"{len(same)} identical copies, {len(renamed)} same bodies under other names, "
          f"{len(shaped)} same shapes under other names")
    return 1 if a.strict and drifted else 0


if __name__ == "__main__":
    sys.exit(main())
