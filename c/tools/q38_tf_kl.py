#!/usr/bin/env python3
"""Paired KL / top-1 check of two qwen38 teacher-forced runs, from their Q38_TF_TOPK dumps.

  python -I q38_tf_kl.py A.topk.bin B.topk.bin [--worst N]   KL(A||B), top-1 agreement, paired NLL gap, worst rows
  python -I q38_tf_kl.py --dir OUT --ref A [--worst N]       every <text>.<arm>.nll.topk.bin against
                                                             <text>.<ref>.nll.topk.bin, per text and pooled
                                                             per arm; writes OUT/kl.json, checks OUT/results.json
  python -I q38_tf_kl.py --nll FILE                          the TF-NLL line the engine printed, rebuilt from the dump
  python -I q38_tf_kl.py --self-test

The engine side: ref.json + PPL=1 + Q38_TF_TOPK=<file> makes qwen38.c's tf_nll() write, little-endian,
  b'Q38TFTK1', int32 K, vocab, np, nfull, then per scored row (nfull - np rows):
  int32 pos, int32 target, float64 target log-prob, float64 log Z, int32 ids[K], float64 logp[K]
with K = min(32, vocab), logp descending and a tie keeping the lower id first (ids[0] is the greedy pick).
The target log-prob is the very term tf_nll() sums, so --nll reproduces the printed TF-NLL.

KL(A||B) of a row runs over U, the union of the two top-K sets. An id outside one side's top-K gets that
side's K-th log-prob (an upper bound on its true value); the mass outside U is one bucket per side (clamped at
1e-12); both vectors are renormalised, so a row's estimate is >= 0 and exactly 0 for identical rows. With
vocab <= K the estimate is the exact KL. Otherwise it is an approximation: the lumped bucket understates the
tail's share and the bound overstates a missing id, so rows with much mass outside the top-K are the least sure.
The NLL gap per row is B's NLL minus A's on the same target; its mean is the paired NLL gap, and its standard
error (sd / sqrt(n), paired by position) says how far that gap is from noise.
"""
import argparse
import json
import math
import os
import re
import struct
import sys

MAGIC = b'Q38TFTK1'
HEAD = struct.Struct('<4i')


def read(path):
    """-> {'k', 'vocab', 'np', 'nfull', 'rows': [(pos, target, target_lp, logz, ids, lps)]}."""
    with open(path, 'rb') as f:
        b = f.read()
    if b[:8] != MAGIC:
        raise ValueError(f'{path}: not a Q38_TF_TOPK dump')
    k, vocab, np_, nfull = HEAD.unpack_from(b, 8)
    row = struct.Struct(f'<2i2d{k}i{k}d')
    body = len(b) - 8 - HEAD.size
    if not 0 < k <= vocab or nfull <= np_ or body != (nfull - np_) * row.size:
        raise ValueError(f'{path}: {body} bytes after the header, expected {nfull - np_} rows of {row.size}')
    rows = []
    for off in range(8 + HEAD.size, len(b), row.size):
        v = row.unpack_from(b, off)
        rows.append((v[0], v[1], v[2], v[3], v[4:4 + k], v[4 + k:]))
    return {'k': k, 'vocab': vocab, 'np': np_, 'nfull': nfull, 'rows': rows}


def nll_line(d):
    """The engine's 'TF-NLL: x nats/token over n tokens', summed in its order (bit-identical mean)."""
    s = 0.0
    for r in d['rows']:
        s += -r[2]
    return f'TF-NLL: {s / len(d["rows"]):.4f} nats/token over {len(d["rows"])} tokens'


def row_kl(a, b, tail):
    """KL(A||B) of one row over the union of the two top-K sets (module doc); tail: K < vocab."""
    la, lb = dict(zip(a[4], a[5])), dict(zip(b[4], b[5]))
    u = sorted(la.keys() | lb.keys())
    xa = [la.get(i, a[5][-1]) for i in u]
    xb = [lb.get(i, b[5][-1]) for i in u]
    if tail:
        xa.append(math.log(max(1.0 - sum(map(math.exp, xa)), 1e-12)))
        xb.append(math.log(max(1.0 - sum(map(math.exp, xb)), 1e-12)))
    za, zb = _lse(xa), _lse(xb)   # renormalise (log domain: no underflow to log(0))
    return max(0.0, sum(math.exp(x - za) * ((x - za) - (y - zb)) for x, y in zip(xa, xb)))


def _lse(x):
    m = max(x)
    return m + math.log(sum(math.exp(v - m) for v in x))


def pair_rows(da, db, text=''):
    """Per-row records of A vs B; the two dumps must score the same ids at the same positions."""
    if (da['k'], da['vocab'], da['np'], da['nfull']) != (db['k'], db['vocab'], db['np'], db['nfull']):
        raise ValueError(f'{text}: the dumps differ in K/vocab/prompt/length')
    tail = da['k'] < da['vocab']
    out = []
    for a, b in zip(da['rows'], db['rows']):
        if a[:2] != b[:2]:
            raise ValueError(f'{text}: row pos/target {a[:2]} vs {b[:2]}: not the same text')
        out.append({'text': text, 'pos': a[0], 'target': a[1], 'kl': row_kl(a, b, tail), 'gap': a[2] - b[2],
                    'cover_a': sum(map(math.exp, a[5])),
                    'top1_a': [a[4][0], round(math.exp(a[5][0]), 4)], 'top1_b': [b[4][0], round(math.exp(b[5][0]), 4)]})
    return out


def summary(rows, worst=10):
    n = len(rows)
    kl = sorted(r['kl'] for r in rows)
    g = [r['gap'] for r in rows]
    gm = sum(g) / n
    se = math.sqrt(sum((x - gm) ** 2 for x in g) / (n - 1) / n) if n > 1 else float('nan')
    q = lambda p: kl[min(n - 1, int(p * n))]  # noqa: E731
    return {'rows': n, 'kl_mean': sum(kl) / n, 'kl_p50': q(0.5), 'kl_p90': q(0.9), 'kl_p99': q(0.99), 'kl_max': kl[-1],
            'top1_agree': sum(r['top1_a'][0] == r['top1_b'][0] for r in rows) / n, 'gap': gm, 'gap_se': se,
            'cover_a': sum(r['cover_a'] for r in rows) / n,
            'worst': sorted(rows, key=lambda r: -r['kl'])[:worst]}


def line(s):
    return (f'KL mean {s["kl_mean"]:.3e} (p50 {s["kl_p50"]:.3e} p90 {s["kl_p90"]:.3e} p99 {s["kl_p99"]:.3e} '
            f'max {s["kl_max"]:.3e}) nats over {s["rows"]} rows, top-1 agreement {100 * s["top1_agree"]:.2f} %, '
            f'NLL gap {s["gap"]:+.5f} +- {s["gap_se"]:.5f} (paired se), A top-K mass {s["cover_a"]:.4f}')


def worst_lines(s):
    return [f'    {r["text"]} pos {r["pos"]} target {r["target"]}: KL {r["kl"]:.3e}, gap {r["gap"]:+.4f}, '
            f'top-1 A {r["top1_a"][0]} ({r["top1_a"][1]}) B {r["top1_b"][0]} ({r["top1_b"][1]})' for r in s['worst']]


def run_dir(d, ref, worst):
    pat = re.compile(r'^(.+)\.(\w+)\.nll\.topk\.bin$')
    files = {m.groups(): os.path.join(d, f) for f in sorted(os.listdir(d)) for m in [pat.match(f)] if m}
    texts = sorted({t for t, a in files if a == ref})
    arms = sorted({a for t, a in files} - {ref})
    if not texts or not arms:
        sys.exit(f'{d}: no <text>.{ref}.nll.topk.bin with another arm beside it')
    res_path = os.path.join(d, 'results.json')
    res = json.load(open(res_path, encoding='utf-8')) if os.path.isfile(res_path) else {}
    dumps, bad = {}, []
    for (t, a), p in files.items():
        dumps[t, a] = read(p)
        printed = res.get(t, {}).get('arms', {}).get(a, {}).get('nll', {}).get('nll')
        mine = float(nll_line(dumps[t, a]).split()[1])
        if printed is not None and f'{printed:.4f}' != f'{mine:.4f}':
            bad.append(f'{t}.{a}: dump {mine:.4f} vs printed {printed:.4f}')
    if bad:
        sys.exit('MISMATCH, a dump is not its run: ' + '; '.join(bad))
    out = {'ref': ref, 'texts': {}, 'arms': {}}
    print(f'{"text":10} {"arm":5} {"rows":>4} {"KL mean":>9} {"KL p90":>9} {"KL max":>9} {"top1%":>7} {"gap":>9} {"+-se":>8}')
    for a in arms:
        pooled = []
        for t in texts:
            if (t, a) not in dumps:
                sys.exit(f'{t}: arm {a} has no dump')
            rows = pair_rows(dumps[t, ref], dumps[t, a], t)
            s = summary(rows, worst)
            pooled += rows
            out['texts'].setdefault(t, {})[a] = s
            print(f'{t:10} {a:5} {s["rows"]:4d} {s["kl_mean"]:9.3e} {s["kl_p90"]:9.3e} {s["kl_max"]:9.3e} '
                  f'{100 * s["top1_agree"]:7.2f} {s["gap"]:+9.5f} {s["gap_se"]:8.5f}')
        out['arms'][a] = s = summary(pooled, worst)
    for a, s in out['arms'].items():
        print(f'arm {a} vs {ref}, pooled: {line(s)}')
        print('\n'.join(worst_lines(s)))
    json.dump(out, open(os.path.join(d, 'kl.json'), 'w', encoding='utf-8'), indent=1)
    return 0


# ---- self-test: synthetic dumps written the way qwen38.c writes them
def _logp(lg):
    mx = max(lg)
    lz = math.log(sum(math.exp(x - mx) for x in lg))
    return [x - mx - lz for x in lg], mx + lz


def _dump(path, rows_logits, targets, np_, k=32):
    vocab = len(rows_logits[0])
    k = min(k, vocab)
    with open(path, 'wb') as f:
        f.write(MAGIC + HEAD.pack(k, vocab, np_, np_ + len(targets)))
        for i, (lg, t) in enumerate(zip(rows_logits, targets)):
            lp, lz = _logp(lg)
            ids = sorted(range(vocab), key=lambda v: (-lg[v], v))[:k]
            f.write(struct.pack(f'<2i2d{k}i{k}d', np_ + i, t, lp[t], lz, *ids, *[lp[v] for v in ids]))


def _exact(la, lb):
    pa, pb = _logp(la)[0], _logp(lb)[0]
    return sum(math.exp(x) * (x - y) for x, y in zip(pa, pb))


def self_test():
    import random
    import tempfile
    rnd = random.Random(7)
    with tempfile.TemporaryDirectory() as tmp:
        fa, fb = os.path.join(tmp, 'a.bin'), os.path.join(tmp, 'b.bin')
        for vocab, spike, noise, tol in ((20, 0, 0.3, 1e-12), (2000, 12.0, 0.05, None)):
            A = []
            for _ in range(40):
                lg = [rnd.gauss(0, 1.5) for _ in range(vocab)]
                for v in rnd.sample(range(vocab), 3):
                    lg[v] += spike
                A.append(lg)
            B = [[x + rnd.gauss(0, noise) for x in lg] for lg in A]
            tg = [rnd.randrange(vocab) for _ in A]
            _dump(fa, A, tg, 5)
            _dump(fb, B, tg, 5)
            da, db = read(fa), read(fb)
            rows = pair_rows(da, db)
            ex = [_exact(x, y) for x, y in zip(A, B)]
            if tol is not None:   # vocab <= K: the estimate is the exact KL
                assert all(abs(r['kl'] - e) <= tol for r, e in zip(rows, ex)), (vocab, rows[0]['kl'], ex[0])
            else:                 # vocab > K, top-32 holding ~99 % of the mass: the pooled mean within 5 % of exact
                est, tru = sum(r['kl'] for r in rows) / len(rows), sum(ex) / len(ex)
                assert abs(est - tru) <= 0.05 * tru and sum(r['cover_a'] for r in rows) > 0.98 * len(rows), (est, tru)
            top1 = sum(max(range(vocab), key=lambda v: (x[v], -v)) == max(range(vocab), key=lambda v: (y[v], -v))
                       for x, y in zip(A, B)) / len(A)
            s = summary(rows)
            assert abs(s['top1_agree'] - top1) < 1e-12 and s['worst'][0]['kl'] == s['kl_max']
            gap = sum(_logp(x)[0][t] - _logp(y)[0][t] for x, y, t in zip(A, B, tg)) / len(A)
            assert abs(s['gap'] - gap) < 1e-12, (s['gap'], gap)
            nll = -sum(_logp(x)[0][t] for x, t in zip(A, tg)) / len(A)
            assert nll_line(da) == f'TF-NLL: {nll:.4f} nats/token over {len(A)} tokens', nll_line(da)
            same = summary(pair_rows(da, read(fa)))   # A vs A: exactly 0 and 100 %
            assert same['kl_max'] == 0.0 and same['top1_agree'] == 1.0 and same['gap'] == 0.0
        _dump(fa, [[1.0] * 40], [3], 2)   # ties: the lower id first, like the engine's argmax
        assert read(fa)['rows'][0][4] == tuple(range(32))
        _dump(fb, [[1.0] * 40], [4], 2)   # another target: not the same text
        try:
            pair_rows(read(fa), read(fb))
            raise AssertionError('different targets accepted')
        except ValueError:
            pass
        with open(fa, 'r+b') as f:        # truncated: refused
            f.truncate(os.path.getsize(fa) - 8)
        try:
            read(fa)
            raise AssertionError('truncated dump accepted')
        except ValueError:
            pass
    print('q38_tf_kl self-test: OK')
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dumps', nargs='*', help='A.topk.bin B.topk.bin')
    ap.add_argument('--dir', help="output folder with <text>.<arm>.nll.topk.bin")
    ap.add_argument('--ref', default='A', help='--dir: the reference arm (default A)')
    ap.add_argument('--nll', metavar='FILE', help='print the TF-NLL line rebuilt from one dump')
    ap.add_argument('--worst', type=int, default=10, help='worst rows to list (default 10)')
    ap.add_argument('--json', help='two-dump mode: write the summary here')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.nll:
        print(nll_line(read(a.nll)))
        return 0
    if a.dir:
        return run_dir(a.dir, a.ref, a.worst)
    if len(a.dumps) != 2:
        ap.error('give two dumps, --dir, --nll or --self-test')
    s = summary(pair_rows(read(a.dumps[0]), read(a.dumps[1]), os.path.basename(a.dumps[1])), a.worst)
    print(line(s))
    print('\n'.join(worst_lines(s)))
    if a.json:
        json.dump(s, open(a.json, 'w', encoding='utf-8'), indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
