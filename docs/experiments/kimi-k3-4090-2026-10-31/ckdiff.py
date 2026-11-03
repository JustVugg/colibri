"""ckdiff.py A B [N] -- two K3_CHECK_DUMP files: the records (layer by layer: attention, the step's
pairs, the MoE's input, latent, routed sum, shared experts, output, the layer's output) in order, and
the first N that differ: which rows, and for the pairs which expert, row and who computed it."""
import sys
import numpy as np

NAMES = {1: 'attention', 2: 'MoE input', 3: 'latent z', 4: 'routed sum u', 5: 'shared experts', 6: 'MoE output',
         7: 'layer output', 10: 'pairs'}
PAIR = np.dtype([('uid', '<i4'), ('row', '<i4'), ('who', '<i4'), ('j', '<i4'), ('h', '<u8')])
WHO = {0: 'nobody', 1: 'CPU', 2: 'GPU'}


def records(path):
    with open(path, 'rb') as f:
        while True:
            hd = f.read(16)
            if len(hd) < 16:
                return
            tag, li, n, d = np.frombuffer(hd, '<i4')
            if tag == 10:
                body = np.frombuffer(f.read(n * PAIR.itemsize), PAIR)
            else:
                body = np.frombuffer(f.read(n * 8), '<u8')
            yield int(tag), int(li), int(n), body


def main():
    a, b = sys.argv[1], sys.argv[2]
    nshow = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    shown = 0; nrec = 0
    for ra, rb in zip(records(a), records(b)):
        nrec += 1
        ta, la, na, xa = ra; tb, lb, nb, xb = rb
        if (ta, la, na) != (tb, lb, nb):
            print(f"record {nrec}: A {NAMES.get(ta, ta)} layer {la} n {na}, B {NAMES.get(tb, tb)} layer {lb} n {nb}: the runs part here")
            return
        if ta == 10:
            d = np.nonzero((xa['h'] != xb['h']) | (xa['uid'] != xb['uid']) | (xa['row'] != xb['row']))[0]
            if len(d) == 0:
                continue
            rows = xa['row'][d]
            first = d[np.argmin(rows)]
            print(f"layer {la} pairs: {len(d)} of {na} differ, rows {rows.min()}..{rows.max()}; the lowest row's pair {first}: "
                  f"expert {xa['uid'][first]} (j {xa['j'][first]}) row {xa['row'][first]}, A {WHO.get(int(xa['who'][first]))} "
                  f"B {WHO.get(int(xb['who'][first]))}")
            same_who = int(np.sum(xa['who'][d] == xb['who'][d]))
            print(f"    who computed the differing pairs: A {np.bincount(xa['who'][d], minlength=3).tolist()} (nobody, CPU, GPU), "
                  f"B {np.bincount(xb['who'][d], minlength=3).tolist()}; the same in both: {same_who}")
            experts = sorted(set(int(u) for u in xa['uid'][d]))
            print(f"    experts: {experts[:20]}{' ...' if len(experts) > 20 else ''}")
        else:
            d = np.nonzero(xa != xb)[0]
            if len(d) == 0:
                continue
            print(f"layer {la} {NAMES.get(ta, ta)}: {len(d)} of {na} rows differ, the first {d[0]}, the last {d[-1]}")
        shown += 1
        if shown >= nshow:
            return
    print(f"{nrec} records compared; {'no difference' if not shown else ''}")


main()
