#!/usr/bin/env python3
"""compare_nll.py <base.nll> <variant.nll>...: a token-by-token comparison of the files bench.c writes with
TEACHER=1 BENCH_NLL=<file> (one line per teacher-forced token: its index, its negative log-likelihood, and 1 if the
model's first choice was the text's token). Both runs read the same tokens, so the comparison is paired: for each
variant, the mean change of the negative log-likelihood, in percent of the base's mean, with its standard error; the
change of the top-1 rate in points, with its standard error; and the change of each quarter of the text, in percent of
the base's mean over the whole text."""
import math, sys


def read(path):
    nll, top = [], []
    for line in open(path):
        _, n, t = line.split()
        nll.append(float(n))
        top.append(int(t))
    return nll, top


base_nll, base_top = read(sys.argv[1])
mb = sum(base_nll) / len(base_nll)
print(f"base {sys.argv[1]}: {len(base_nll)} tokens, mean nll {mb:.5f}, top-1 {sum(base_top) / len(base_top):.3f}")
for path in sys.argv[2:]:
    nll, top = read(path)
    n = min(len(nll), len(base_nll))
    d = [nll[i] - base_nll[i] for i in range(n)]
    md = sum(d) / n
    se = math.sqrt(sum((x - md) ** 2 for x in d) / (n - 1)) / math.sqrt(n)
    q = n // 4
    quarters = [sum(d[k * q:(k + 1) * q]) / q / mb * 100 for k in range(4)]
    dt = [top[i] - base_top[i] for i in range(n)]
    mt = sum(dt) / n
    set_ = math.sqrt(sum((x - mt) ** 2 for x in dt) / (n - 1)) / math.sqrt(n)
    print(f"{path}: mean nll {md / mb * 100:+.2f}% (standard error {se / mb * 100:.2f}%), top-1 {mt * 100:+.1f} points "
          f"(standard error {set_ * 100:.1f}), by quarter " + " ".join(f"{x:+.2f}%" for x in quarters))
