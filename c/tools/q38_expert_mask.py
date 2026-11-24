#!/usr/bin/env python3
"""Build a Q38_EXPERT_MASK file (qwen38's VRAM-resident-only routing) from expert usage.

The engine's routing history is the .coli_usage file qwen38 keeps next to the model
(<snap>/.coli_usage, or COLI_USAGE): text records "<layer> <expert> <count>", with the
header records "-1 <layers> <experts>" and "-2 <version> <engine id>"; the layer after
the last model layer is the MTP head's. Every given history is summed, then each model
layer keeps its N most routed experts (count descending, the lower id on a tie).

N comes from --per-layer, or from the VRAM the experts may take:
    budget = --free-gb - --reserve-gb - --trunk-gb - --kv-gb      (GiB)
    N      = floor(budget / (layers * --expert-bytes))
The defaults are a 32 GB card as the qwen38 logs show it with QT_ARENA=1: 30.3 GiB free
at start, 0.8 GiB reserve, 3.96 GiB int8 trunk, and 3.31 GiB taken by the CUDA chain's
workspace and the KV mirror at the serve context (the tier's budget went 25.49 -> 22.18
GiB). One MXFP4 expert costs 2,611,200 bytes in an arena slot plus its share of the
256 MiB slab's rounding (2,631,761 bytes); without QT_ARENA it is 3,317,760 bytes.

Output: the mask (text, one line "<layer> <ids...>" per layer, hottest first, comments
with the coverage) and, with --heat-out, a QTH1 heat table (HEAT_FILE) of the same counts
so the tier's warmstart loads the allowed experts hottest first.

    python tools/q38_expert_mask.py <snap>/.coli_usage \
        --out q38_mask.txt --heat-out q38_heat.bin
"""
import argparse
import struct
import sys

GIB = 1 << 30


def read_usage(path):
    """{(layer, expert): count} and the (layers, experts) header (None when absent).
    The header holds the model's layer count; the row after it is the MTP head's."""
    counts, dims = {}, None
    with open(path, "rb") as f:
        head = f.read(4)
        if head == b"IKU1":
            sys.exit(f"{path}: an inkling IKU1 history, not qwen38's")
    with open(path, "r", encoding="ascii", errors="strict") as f:
        for n, line in enumerate(f, 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 3:
                sys.exit(f"{path}:{n}: expected three numbers, got {line.strip()!r}")
            layer, expert, count = (int(p) for p in parts)
            if layer == -1:
                dims = (expert, count)
            elif layer >= 0:
                counts[(layer, expert)] = counts.get((layer, expert), 0) + count
    return counts, dims


def rank(total, layers, experts):
    """per layer: expert ids by count descending, the lower id on a tie; and the counts"""
    order, rows = [], []
    for layer in range(layers):
        row = [total.get((layer, e), 0) for e in range(experts)]
        rows.append(row)
        order.append(sorted(range(experts), key=lambda e: (-row[e], e)))
    return order, rows


def coverage(order, rows, n):
    """per layer: the share of the routed selections the top n experts took"""
    out = []
    for ids, row in zip(order, rows):
        s = sum(row)
        out.append(sum(row[e] for e in ids[:n]) / s if s else 0.0)
    return out


def needed(order, rows, share):
    """per layer: how many experts the top takes to reach `share` of the selections"""
    out = []
    for ids, row in zip(order, rows):
        s, acc = sum(row), 0
        for k, e in enumerate(ids, 1):
            acc += row[e]
            if s and acc >= share * s:
                out.append(k)
                break
        else:
            out.append(len(ids))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("usage", nargs="+", help=".coli_usage histories (summed)")
    ap.add_argument("--out", help="mask file to write (omitted: report only)")
    ap.add_argument("--heat-out", help="also write a QTH1 HEAT_FILE of the summed counts")
    ap.add_argument("--layers", type=int, default=0, help="model layers (default: the header's)")
    ap.add_argument("--experts", type=int, default=0, help="experts per layer (default: the header's)")
    ap.add_argument("--topk", type=int, default=10, help="num_experts_per_tok; N is never below it")
    ap.add_argument("--per-layer", type=int, default=0, help="N experts a layer (overrides the budget)")
    ap.add_argument("--free-gb", type=float, default=30.3, help="GiB free on the card at start")
    ap.add_argument("--reserve-gb", type=float, default=0.8, help="CUDA_RESERVE_GB")
    ap.add_argument("--trunk-gb", type=float, default=3.96, help="the int8 trunk the tier places")
    ap.add_argument("--kv-gb", type=float, default=3.31, help="CUDA chain workspace + KV mirror")
    ap.add_argument("--eval", nargs="*", default=[], help="histories to score the mask on (held out: not summed)")
    ap.add_argument("--expert-bytes", type=int, default=2631761, help="device bytes per expert (3317760 without QT_ARENA)")
    a = ap.parse_args(argv)

    total, dims = {}, None
    for path in a.usage:
        counts, d = read_usage(path)
        if d and dims and d != dims:
            sys.exit(f"{path}: dimensions {d} differ from {dims}")
        dims = dims or d
        for k, v in counts.items():
            total[k] = total.get(k, 0) + v
    layers = a.layers or (dims[0] if dims else 0)
    experts = a.experts or (dims[1] if dims else 0)
    if layers <= 0 or experts <= 0:
        sys.exit("no dimensions: pass --layers and --experts for a headerless history")
    if a.topk > experts:
        sys.exit(f"--topk {a.topk} exceeds the {experts} experts")

    budget = (a.free_gb - a.reserve_gb - a.trunk_gb - a.kv_gb) * GIB
    if a.per_layer:
        n = a.per_layer
    else:
        n = int(budget // (layers * a.expert_bytes))
        print(f"budget {budget / GIB:.2f} GiB = {a.free_gb} free - {a.reserve_gb} reserve - {a.trunk_gb} trunk - "
              f"{a.kv_gb} chain+KV; {a.expert_bytes} bytes an expert -> N = {n} a layer")
    if n < a.topk:
        sys.exit(f"N = {n} is below the top-{a.topk}: the mask would leave a layer without a full route")
    n = min(n, experts)

    order, rows = rank(total, layers, experts)
    cov = coverage(order, rows, n)
    sel = sum(sum(r) for r in rows)
    vram = n * layers * a.expert_bytes
    print(f"{len(a.usage)} histories, {sel} selections over {layers} layers "
          f"(~{sel / (layers * a.topk):.0f} routed rows a layer), {experts} experts a layer")
    print(f"N = {n}: {n * layers} experts, {vram / GIB:.2f} GiB; selections the mask keeps a layer: "
          f"min {min(cov):.3f} (layer {cov.index(min(cov))}), mean {sum(cov) / len(cov):.3f}, max {max(cov):.3f}")
    for share in (0.90, 0.95, 0.99):
        need = needed(order, rows, share)
        print(f"  experts a layer for {share:.0%} of its selections: min {min(need)}, "
              f"mean {sum(need) / len(need):.0f}, max {max(need)}")
    for path in a.eval:
        held, _ = read_usage(path)
        _, hrows = rank(held, layers, experts)
        hcov = coverage(order, hrows, n)
        print(f"  on {path} (held out): kept share a layer min {min(hcov):.3f} (layer {hcov.index(min(hcov))}), "
              f"mean {sum(hcov) / len(hcov):.3f}")
    zero = [sum(1 for e in ids[:n] if row[e] == 0) for ids, row in zip(order, rows)]
    if any(zero):
        print(f"  {sum(zero)} kept experts were never routed in the history (filled by id)")

    if a.out:
        with open(a.out, "w", encoding="ascii", newline="\n") as f:
            f.write(f"# Q38_EXPERT_MASK: {n} experts a layer, the most routed first, from {len(a.usage)} history file(s)\n")
            f.write(f"# {sel} selections; kept share a layer min {min(cov):.4f} mean {sum(cov) / len(cov):.4f}\n")
            for layer in range(layers):
                f.write(f"# layer {layer}: keeps {cov[layer]:.4f}\n")
                f.write(f"{layer} " + " ".join(str(e) for e in order[layer][:n]) + "\n")
        print(f"mask -> {a.out}")
    if a.heat_out:
        with open(a.heat_out, "wb") as f:
            f.write(struct.pack("<3I", 0x51544831, layers, experts))
            for row in rows:
                f.write(struct.pack(f"<{experts}I", *(min(c, 0xFFFFFFFF) for c in row)))
        print(f"heat -> {a.heat_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
