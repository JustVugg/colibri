"""logit_diff.py A B VOCAB -- two K3_LOGITS files (positions x VOCAB float32): positions whose argmax differs,
the largest |A-B| and |A-B| / max|A|, and KL(softmax A || softmax B) per position (in float64): its maximum and mean."""
import sys
import numpy as np

a_path, b_path, vocab = sys.argv[1], sys.argv[2], int(sys.argv[3])
A = np.memmap(a_path, dtype=np.float32, mode='r')
B = np.memmap(b_path, dtype=np.float32, mode='r')
assert A.size == B.size and A.size % vocab == 0, (A.size, B.size)
n = A.size // vocab
A = A.reshape(n, vocab); B = B.reshape(n, vocab)
arg_diff = 0; same_bytes = 0; max_abs = 0.0; max_rel = 0.0; kl_max = 0.0; kl_sum = 0.0; worst = -1
top5_diff = 0
for s in range(0, n, 64):
    a = np.asarray(A[s:s + 64], dtype=np.float64); b = np.asarray(B[s:s + 64], dtype=np.float64)
    same_bytes += int(np.sum(np.all(A[s:s + 64].view(np.uint32) == B[s:s + 64].view(np.uint32), axis=1)))
    arg_diff += int(np.sum(np.argmax(a, axis=1) != np.argmax(b, axis=1)))
    ta = np.argsort(-a, axis=1)[:, :5]; tb = np.argsort(-b, axis=1)[:, :5]
    top5_diff += int(np.sum(np.any(ta != tb, axis=1)))
    d = np.abs(a - b)
    max_abs = max(max_abs, float(d.max()))
    max_rel = max(max_rel, float((d.max(axis=1) / np.abs(a).max(axis=1)).max()))
    la = a - a.max(axis=1, keepdims=True); la -= np.log(np.exp(la).sum(axis=1, keepdims=True))
    lb = b - b.max(axis=1, keepdims=True); lb -= np.log(np.exp(lb).sum(axis=1, keepdims=True))
    kl = np.sum(np.exp(la) * (la - lb), axis=1)
    i = int(np.argmax(kl))
    if kl[i] > kl_max: kl_max = float(kl[i]); worst = s + i
    kl_sum += float(kl.sum())
print(f"{n} positions: {same_bytes} the same bytes; top token differs at {arg_diff}, top 5 (in order) at {top5_diff}; "
      f"max |A-B| {max_abs:.3g} ({max_rel:.3g} of the row's largest |A|); KL max {kl_max:.3g} (position {worst}), mean {kl_sum / n:.3g}")
