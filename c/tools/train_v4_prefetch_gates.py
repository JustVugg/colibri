#!/usr/bin/env python3
"""Train and export a V4PG v1 learned next-layer prefetch gate sidecar.

Fits ridge regression on gate logits per (target layer N, depth d) from a
V4_DUMP_MOE (v1) or V4_DUMP_MOE2 (v2, "V4D2") trace and folds the correction
and ffn_norm ratio into a single bf16 [experts, hidden] matrix + f32[experts]
intercept so runtime prediction remains one GEMV:

    z   = G'_{N,d} @ x_{N-d} + c_{N,d}
    sel = sqrt(softplus(z)) + bias_N

V4PG v1 binary format (little-endian):
    char[4]  "V4PG"
    uint32   version = 1
    uint32   layers
    uint32   experts
    uint32   hidden
    uint32   depths (1 or 2)
    For d in 1..depths, N in 0..layers-1:
        uint32   present (0 or 1; must be 0 when N < d)
        if present:
            uint16[experts * hidden]  bf16 G'_{N,d}, row-major (expert, hidden)
            float32[experts]          c_{N,d} intercept
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch


def load_config(model_dir: Path) -> dict:
    cfg_path = model_dir / "config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_router_tensors(model_dir: Path, layers: int, experts: int, device: str):
    """Load ffn.gate.weight, ffn.gate.bias, and ffn_norm.weight from safetensors."""
    from safetensors.torch import safe_open

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        with open(index_path, "r", encoding="utf-8") as f:
            weight_map = json.load(f).get("weight_map", {})
        shard_for = lambda name: model_dir / weight_map[name] if name in weight_map else None
    else:
        single = model_dir / "model.safetensors"
        shard_for = lambda name: single if single.is_file() else None

    needed = {}
    for l in range(layers):
        for suffix in ("ffn.gate.weight", "ffn.gate.bias", "ffn_norm.weight"):
            for prefix in (f"layers.{l}.", f"model.layers.{l}."):
                key = prefix + suffix
                p = shard_for(key)
                if p is not None:
                    needed.setdefault(p, []).append((l, suffix, key))
                    break

    gates = [None] * layers
    biases = [torch.zeros(experts, dtype=torch.float32) for _ in range(layers)]
    norms = [None] * layers
    for shard_path, items in needed.items():
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            keys = set(f.keys())
            for l, suffix, key in items:
                if key not in keys:
                    continue
                t = f.get_tensor(key).float()
                if suffix == "ffn.gate.weight":
                    gates[l] = t
                elif suffix == "ffn.gate.bias":
                    biases[l] = t
                elif suffix == "ffn_norm.weight":
                    norms[l] = t

    missing = [l for l in range(layers) if gates[l] is None or norms[l] is None]
    if missing:
        raise RuntimeError(
            f"Missing gate or ffn_norm tensors for layers {missing[:5]} in {model_dir}"
        )
    G = torch.stack(gates).to(device)
    B = torch.stack(biases).to(device)
    WN = torch.stack(norms).to(device)
    return G, B, WN


def load_dump(dump_path: Path, layers: int, hidden: int, topk: int, device: str):
    """Load V4_DUMP_MOE (v1) or V4_DUMP_MOE2 (v2) records as contiguous [T, L, ...]."""
    with open(dump_path, "rb") as f:
        magic = f.read(4)

    if magic == b"V4D2":
        with open(dump_path, "rb") as f:
            hdr = f.read(32)
        _, ver, d_layers, d_hidden, hc_mult, d_topk, rec_bytes, _ = struct.unpack("<4s7I", hdr)
        if ver != 2 or d_layers != layers or d_hidden != hidden or d_topk != topk:
            raise RuntimeError(
                f"V4D2 header mismatch: ver={ver} layers={d_layers} hidden={d_hidden} topk={d_topk}"
            )
        rec_dtype = np.dtype([
            ("layer", "<i4"),
            ("token", "<i4"),
            ("position", "<i4"),
            ("forward", "<u4"),
            ("flags", "<u4"),
            ("count", "<i4"),
            ("ids", "<i4", topk),
            ("w", "<f4", topk),
            ("score", "<f4", topk),
            ("ffn_pre", "<f4", hc_mult),
            ("ffn_post", "<f4", hc_mult),
            ("ffn_comb", "<f4", hc_mult * hc_mult),
            ("x_bf16", "<u2", hidden),
            ("streams_bf16", "<u2", (hc_mult, hidden)),
        ])
        assert rec_dtype.itemsize == rec_bytes, (rec_dtype.itemsize, rec_bytes)
        m = np.memmap(dump_path, dtype=rec_dtype, mode="r", offset=32)
        n = (m.shape[0] // layers) * layers
        T = n // layers
        m = m[:n]
        L = np.asarray(m["layer"]).reshape(T, layers)
        valid = np.all(L == np.arange(layers)[None, :], axis=1)
        m = m.reshape(T, layers)[valid]
        T = m.shape[0]
        x_u16 = np.ascontiguousarray(m["x_bf16"]).view(np.int16)
        X = torch.from_numpy(x_u16).view(torch.bfloat16).float().to(device)
        IDS = torch.from_numpy(np.ascontiguousarray(m["ids"]).astype(np.int64)).to(device)
        return X, IDS

    rec_dtype = np.dtype([
        ("L", "<i4"),
        ("t", "<i4"),
        ("c", "<i4"),
        ("ids", "<i4", topk),
        ("w", "<f4", topk),
        ("x", "<f4", hidden),
    ])
    m = np.memmap(dump_path, dtype=rec_dtype, mode="r")
    n = (m.shape[0] // layers) * layers
    T = n // layers
    m = m[:n]
    L = np.asarray(m["L"]).reshape(T, layers)
    if not np.all(L == np.arange(layers)[None, :]):
        valid = np.all(L == np.arange(layers)[None, :], axis=1)
        m = m.reshape(T, layers)[valid]
        T = m.shape[0]
    else:
        m = m.reshape(T, layers)
    X = torch.from_numpy(np.ascontiguousarray(m["x"])).to(device)
    IDS = torch.from_numpy(np.ascontiguousarray(m["ids"]).astype(np.int64)).to(device)
    return X, IDS


def fit_gate(X, G, WN, target_layer: int, depth: int, train_idx: torch.Tensor, lam_c: float):
    """Solve ridge regression on logit residuals and fold ffn_norm ratio into G'."""
    ratio = (WN[target_layer] / WN[target_layer - depth]).double()
    xs = X[:, target_layer - depth].double() * ratio
    Gd = G[target_layer].double()
    R = X[:, target_layer].double() @ Gd.T - xs @ Gd.T
    mx = xs[train_idx].mean(0)
    mr = R[train_idx].mean(0)
    Xc = xs[train_idx] - mx
    Rc = R[train_idx] - mr
    K = Xc @ Xc.T
    lam = lam_c * K.diagonal().mean()
    eye = torch.eye(len(train_idx), device=X.device, dtype=K.dtype)
    a = torch.linalg.solve(K + lam * eye, Rc)
    Delta = Xc.T @ a  # (hidden, experts)
    Gp = (Gd + Delta.T) * ratio
    c = mr - mx @ Delta
    return Gp, c


def sel_scores(logits: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.nn.functional.softplus(logits)) + bias


def hits_of(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    return (pred[:, :, None] == true[:, None, :]).any(1).sum(1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train and export V4PG v1 learned next-layer prefetch gates."
    )
    parser.add_argument("--model", required=True, help="Path to DeepSeek V4 model directory")
    parser.add_argument("--dump", required=True, help="Path to V4_DUMP_MOE or V4_DUMP_MOE2 file")
    parser.add_argument("--out", required=True, help="Output V4PG v1 binary path")
    parser.add_argument("--depths", type=int, default=1, choices=(1, 2), help="Max lookahead depth")
    parser.add_argument("--first-gate-layer", type=int, default=3, help="First non-hash MoE layer")
    parser.add_argument("--lambda-c", type=float, default=0.3, help="Ridge regularization scale")
    parser.add_argument("--eval", action="store_true", help="Print held-out fold recall@8")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (cuda or cpu)",
    )
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model_dir = Path(args.model)
    cfg = load_config(model_dir)
    layers = int(cfg["num_hidden_layers"])
    experts = int(cfg["n_routed_experts"])
    hidden = int(cfg["hidden_size"])
    topk = int(cfg["num_experts_per_tok"])

    X, IDS = load_dump(Path(args.dump), layers, hidden, topk, args.device)
    G, B, WN = load_router_tensors(model_dir, layers, experts, args.device)
    T = X.shape[0]
    if T < 5:
        raise RuntimeError(f"Need at least 5 complete forwards in dump, got {T}")

    if args.eval:
        split = (T * 4) // 5
        tr = torch.arange(0, split, device=args.device)
        te = torch.arange(split, T, device=args.device)
        for d in range(1, args.depths + 1):
            h64 = h16 = hb = n = 0
            for N in range(args.first_gate_layer, layers):
                if N - d < 0:
                    continue
                Gp, c = fit_gate(X, G, WN, N, d, tr, args.lambda_c)
                x = X[te, N - d]
                s64 = sel_scores((x.double() @ Gp.T + c).float(), B[N])
                G16 = Gp.float().bfloat16().float()
                s16 = sel_scores(x @ G16.T + c.float(), B[N])
                sb = sel_scores(x @ G[N].T, B[N])
                h64 += hits_of(s64.topk(8, 1).indices, IDS[te, N]).sum().item()
                h16 += hits_of(s16.topk(8, 1).indices, IDS[te, N]).sum().item()
                hb += hits_of(sb.topk(8, 1).indices, IDS[te, N]).sum().item()
                n += len(te) * topk
            if n:
                print(
                    f"d{d} held-out R@8: baseline {hb / n * 100:.1f}%  "
                    f"learned f64 {h64 / n * 100:.1f}%  learned bf16 {h16 / n * 100:.1f}%"
                )

    all_idx = torch.arange(T, device=args.device)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(b"V4PG" + struct.pack("<5I", 1, layers, experts, hidden, args.depths))
        for d in range(1, args.depths + 1):
            for N in range(layers):
                ok = N >= args.first_gate_layer and N - d >= 0
                f.write(struct.pack("<I", 1 if ok else 0))
                if not ok:
                    continue
                Gp, c = fit_gate(X, G, WN, N, d, all_idx, args.lambda_c)
                bf16_bytes = (
                    Gp.float().bfloat16().view(torch.int16).cpu().numpy().astype("<i2").tobytes()
                )
                c_bytes = c.float().cpu().numpy().astype("<f4").tobytes()
                f.write(bf16_bytes)
                f.write(c_bytes)
    print(f"wrote {out_path} ({os.path.getsize(out_path)} bytes, T={T}, depths={args.depths})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
