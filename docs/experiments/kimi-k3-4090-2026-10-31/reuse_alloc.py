"""Per-layer LRU hit curves from K3_REUSE's stack distances; the misses of a uniform cache
against the same slots given to the layers where they hold most (greedy on the marginal
hits, exact for concave curves)."""
import sys

rows = []
for line in open(sys.argv[1]):
    v = [int(x) for x in line.split()]
    rows.append((v[0], v[1], v[2], v[3:]))
cap = int(sys.argv[2]) if len(sys.argv) > 2 else 97
E = len(rows[0][3]) - 1
tot = {li: sum(h) for li, _, _, h in rows}
acc = sum(tot.values())


def hits(h, c):
    return sum(h[:min(c, E)])


uni = sum(hits(h, cap) for _, _, _, h in rows)
print(f"{len(rows)} layers, {acc} routed (decode), LRU at {cap} a layer: hit {uni / acc:.1%}")
for c in (48, 64, 80, 97, 120, 160, 200):
    print(f"  uniform {c:4d}: hit {sum(hits(h, c) for _, _, _, h in rows) / acc:.1%}")
# greedy: every layer starts at a floor, then each slot goes where its marginal hit count is largest
floor = int(sys.argv[3]) if len(sys.argv) > 3 else 16
budget = cap * len(rows)
alloc = {li: floor for li, _, _, _ in rows}
left = budget - floor * len(rows)
H = {li: h for li, _, _, h in rows}
import heapq
heap = [(-H[li][alloc[li]], li) for li in alloc]
heapq.heapify(heap)
while left > 0:
    g, li = heapq.heappop(heap)
    alloc[li] += 1
    left -= 1
    if alloc[li] < E:
        heapq.heappush(heap, (-H[li][alloc[li]], li))
opt = sum(hits(H[li], alloc[li]) for li in alloc)
print(f"best split of the same {budget} slots (floor {floor}): hit {opt / acc:.1%}, "
      f"misses {acc - uni} -> {acc - opt} ({(opt - uni) / max(1, acc - uni):.1%} fewer)")
print("slots by layer:", " ".join(f"L{li}:{alloc[li]}" for li in sorted(alloc)))
print("hit at 97 by layer:", " ".join(f"L{li}:{hits(H[li], cap) / max(1, tot[li]):.2f}" for li in sorted(alloc)))
print("predicted by layer:", " ".join(f"L{li}:{ph / max(1, p):.2f}" for li, p, ph, _ in rows))
