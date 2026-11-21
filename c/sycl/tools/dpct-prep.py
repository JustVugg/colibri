#!/usr/bin/env python3
"""The SYCL migration's input: a staged copy of c/ with every coli_launch(k, g, b, sm, st, a...)
expanded to k<<<g, b, sm, st>>>(a...).

dpct (SYCLomatic 2025.3) migrates a launch whose kernel it sees at the site. It cannot migrate a
launch through a kernel pointer (backend_gpu_compat.h's coli_launch template: the tool emits
dpct::kernel_launcher::launch with the parameter pack expanded wrongly and a wrapper reading
static state) nor one inside a function-like macro (it rewrites the macro's definition for the
first expansion site, DPCT1038). So the macro COLI_GPU_PORTABLE_ONLY defines is expanded here,
textually, before the tool runs. A kernel's template arguments carry commas (cc_attn_decg_k<2, KT>);
the splitter tracks the angle brackets, so no call site needs parentheses.

    dpct-prep.py <c-dir> <staging-dir>

Copies <c-dir> (sources and tests, not build outputs) to <staging-dir> and rewrites the files
that launch kernels. Sites it cannot split (fewer than five arguments: the HCM_MOE macro's
coli_launch(K_<8, BM_>, __VA_ARGS__), NVIDIA-only and compiled out) are left as they are and
listed. Run with python3 -I.
"""
import os
import re
import shutil
import sys

LAUNCHERS = ("backend_cuda.cu", "cuda_chain.cu", "cuda_bd.cuh", "cu_xb.cu")
SKIP_DIRS = {"__pycache__", "clef_tiny", "deepseek_v4_tiny", "data"}
SKIP_EXT = {".o", ".d", ".so", ".a", ".exe"}
IDENT = re.compile(r"[A-Za-z0-9_]")


def split_args(text, start):
    """text[start] is the character after 'coli_launch('. Returns (args, end) where end is the
    index of the closing parenthesis and args the top-level comma-separated pieces."""
    depth = 0          # ( [ {
    tdepth = 0         # template < >
    args, cur, i = [], [], start
    n = len(text)
    while i < n:
        c = text[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            if depth == 0 and c == ")":
                args.append("".join(cur))
                return args, i
            depth -= 1
        elif c == "<":
            # a template bracket follows an identifier or a closing '>' at once (kvp<KT>,
            # cc_f2_smem_bytes<64, 4, 64, true>); a comparison has a space before it
            prev = text[i - 1] if i > 0 else ""
            if IDENT.match(prev) or prev == ">":
                tdepth += 1
        elif c == ">":
            prev = text[i - 1] if i > 0 else ""
            if tdepth > 0 and prev != "-":
                tdepth -= 1
        elif c == "," and depth == 0 and tdepth == 0:
            args.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(c)
        i += 1
    raise ValueError("unbalanced coli_launch")


def rewrite(text, name):
    out, pos, done, left = [], 0, 0, []
    pat = re.compile(r"(?<![A-Za-z0-9_])coli_launch\(")
    for m in pat.finditer(text):
        if m.start() < pos:
            continue
        args, end = split_args(text, m.end())
        if len(args) < 5:
            left.append((name, text.count("\n", 0, m.start()) + 1, text[m.start():end + 1][:80]))
            continue
        # spaces only: a newline inside an argument stays, so the staged file keeps the
        # original's line numbers (the tool's diagnostics then name the lines of c/*.cu)
        k, g, b, sm, st = (a.strip(" \t") for a in args[:5])
        rest = ",".join(args[5:]).lstrip(" \t")
        out.append(text[pos:m.start()])
        out.append("%s<<<%s, %s, %s, %s>>>(%s)" % (k, g, b, sm, st, rest))
        pos = end + 1
        done += 1
    out.append(text[pos:])
    return "".join(out), done, left


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    if os.path.exists(dst):
        shutil.rmtree(dst)
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        rel = os.path.relpath(root, src)
        os.makedirs(os.path.join(dst, rel), exist_ok=True)
        for f in files:
            if os.path.splitext(f)[1] in SKIP_EXT or f.startswith("."):
                continue
            shutil.copy2(os.path.join(root, f), os.path.join(dst, rel, f))
    total, left = 0, []
    for name in LAUNCHERS:
        p = os.path.join(dst, name)
        text = open(p).read()
        new, done, skipped = rewrite(text, name)
        open(p, "w").write(new)
        total += done
        left += skipped
        print("%s: %d launches expanded" % (name, done))
    for name, line, snippet in left:
        print("left as is: %s:%d %s" % (name, line, snippet))
    print("staged %s -> %s (%d launches)" % (src, dst, total))


if __name__ == "__main__":
    main()
