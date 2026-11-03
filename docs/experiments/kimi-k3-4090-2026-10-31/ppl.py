"""ppl.py IDS_ERR VOCAB LOGITS... -- the prompt's own next tokens under each K3_LOGITS file (position p predicts
id p+1): mean negative log-likelihood (nats), perplexity, and how often the top token is the next one; then the
same per position between the first file and each other one: the NLL differences' mean and spread."""
import re
import sys
import numpy as np

err, vocab, files = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
t = open(err, encoding='utf-8', errors='replace').read()
t = t[t.index('[K3] chat ids:') + len('[K3] chat ids:'):t.index('[K3] prompt:')]
t = re.sub(r'C:\S*: Permission denied\r?\n', '', t)   # perror's line, written into the middle of an id
ids = np.array([int(v) for v in t.split()], dtype=np.int64)
nll_all = []
for f in files:
    L = np.memmap(f, dtype=np.float32, mode='r')
    n = L.size // vocab
    assert n == len(ids), (f, n, len(ids))
    L = L.reshape(n, vocab)
    nll = np.empty(n - 1); top = 0
    for s in range(0, n - 1, 64):
        e = min(s + 64, n - 1)
        a = np.asarray(L[s:e], dtype=np.float64)
        m = a.max(axis=1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(a - m).sum(axis=1))
        nll[s:e] = lse - a[np.arange(e - s), ids[s + 1:e + 1]]
        top += int(np.sum(np.argmax(a, axis=1) == ids[s + 1:e + 1]))
    nll_all.append(nll)
    name = f.replace('\\', '/').split('/')[-1]
    print(f"{name}: {n - 1} next tokens, NLL {nll.mean():.5f} nats, perplexity {np.exp(nll.mean()):.4f}, "
          f"top token right at {top} ({100.0 * top / (n - 1):.2f}%)")
for i in range(1, len(files)):
    d = nll_all[i] - nll_all[0]
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"  file {i} - file 0: NLL {d.mean():+.5f} nats (standard error {se:.5f}); "
          f"{int((d < 0).sum())} positions lower, {int((d > 0).sum())} higher, {int((d == 0).sum())} equal")
