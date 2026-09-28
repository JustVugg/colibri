#!/usr/bin/env python3
"""Generate a deterministic one-layer Gemma 4 GGUF and reference outputs.

The fixture keeps Gemma 4's real tensor names, Q6_K token embeddings, Q4_0
resident projections, routed Q4_0 experts, tokenizer metadata, and packed
expert container.  A small independent NumPy implementation emits the logits
and greedy tokens used by CI; it does not invoke the Colibri executable.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np

W, FF, VOCAB, EXPERTS, TOPK = 256, 32, 8, 2, 1
EPS = np.float32(1e-6)
TOKENS = ["<bos>", "A", "B", "C", "D", "E", "F", "<eos>"]


def p32(v): return struct.pack("<I", v)
def p64(v): return struct.pack("<Q", v)
def pf(v): return struct.pack("<f", v)
def ps(v):
    b = v.encode("utf-8")
    return p64(len(b)) + b


def q4_matrix(rows: int, cols: int, shift: int = 0, zero: bool = False):
    """Q4_0 matrix plus its independently decoded float32 values."""
    raw = bytearray()
    decoded = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        for block in range(cols // 32):
            q = np.full(32, 8, dtype=np.uint8)
            if not zero:
                target = (r + shift) % cols
                if target // 32 == block:
                    q[target % 32] = 10 if (r + shift) % 3 else 9
            scale = np.float16(0.125 + 0.03125 * ((r + block) % 3))
            values = (q.astype(np.int16) - 8).astype(np.float32) * np.float32(scale)
            decoded[r, block * 32:(block + 1) * 32] = values
            packed = bytes(int(q[i]) | (int(q[i + 16]) << 4) for i in range(16))
            raw += struct.pack("<e", float(scale)) + packed
    return bytes(raw), decoded


def q6_rows(rows: int):
    raw = bytearray()
    decoded = np.zeros((rows, W), dtype=np.float32)
    rng = np.random.default_rng(1276)
    for row in range(rows):
        q = rng.integers(-7, 8, W, dtype=np.int16)
        q[(row * 29) % W] = 20
        q[(row * 29 + 7) % W] = -18
        scales = np.ones(16, dtype=np.int8)
        ql = np.zeros(128, dtype=np.uint8)
        qh = np.zeros(64, dtype=np.uint8)
        for half in range(2):
            for lane in range(32):
                for quarter in range(4):
                    idx = half * 128 + quarter * 32 + lane
                    enc = int(q[idx] + 32)
                    if quarter == 0:
                        ql[half * 64 + lane] |= enc & 15
                        qh[half * 32 + lane] |= ((enc >> 4) & 3) << 0
                    elif quarter == 1:
                        ql[half * 64 + lane + 32] |= enc & 15
                        qh[half * 32 + lane] |= ((enc >> 4) & 3) << 2
                    elif quarter == 2:
                        ql[half * 64 + lane] |= (enc & 15) << 4
                        qh[half * 32 + lane] |= ((enc >> 4) & 3) << 4
                    else:
                        ql[half * 64 + lane + 32] |= (enc & 15) << 4
                        qh[half * 32 + lane] |= ((enc >> 4) & 3) << 6
        super_scale = np.float16(0.03125)
        raw += ql.tobytes() + qh.tobytes() + scales.tobytes() + struct.pack("<e", float(super_scale))
        decoded[row] = q.astype(np.float32) * np.float32(super_scale)
    return bytes(raw), decoded


def rms(x, weight):
    x = np.asarray(x, dtype=np.float32)
    ss = np.sum(x * x, dtype=np.float32)
    inv = np.float32(1.0) / np.sqrt(np.float32(ss / np.float32(x.size) + EPS), dtype=np.float32)
    return (x * inv * weight).astype(np.float32)


def gelu(x):
    x = np.asarray(x, dtype=np.float32)
    inner = np.float32(0.7978845608028654) * (x + np.float32(0.044715) * x * x * x)
    return (np.float32(0.5) * x * (np.float32(1.0) + np.tanh(inner))).astype(np.float32)


def reference(weights, prompt, new_tokens):
    emb = weights["embedding"]
    norms = np.ones(W, dtype=np.float32)
    keys, values, generated, first_logits = [], [], [], None
    sequence = list(prompt)
    for step in range(len(prompt) + new_tokens):
        token = sequence[step]
        residual = (emb[token] * np.float32(math.sqrt(W))).astype(np.float32)
        normal = rms(residual, norms)
        q = rms(weights["q"] @ normal, norms)
        k = rms(weights["k"] @ normal, norms)
        v = rms(weights["v"] @ normal, np.ones(W, dtype=np.float32))
        pos = step
        half = W // 2
        idx = np.arange(half, dtype=np.float32)
        freq = np.power(np.float32(10000.0), -np.float32(2.0) * idx / np.float32(W)).astype(np.float32)
        angle = np.float32(pos) * freq
        cs, sn = np.cos(angle).astype(np.float32), np.sin(angle).astype(np.float32)
        for vec in (q, k):
            a, b = vec[:half].copy(), vec[half:].copy()
            vec[:half] = a * cs - b * sn
            vec[half:] = b * cs + a * sn
        keys.append(k); values.append(v)
        scores = np.array([np.sum(q * old, dtype=np.float32) for old in keys], dtype=np.float32)
        probs = np.exp(scores - np.max(scores)).astype(np.float32)
        probs = (probs / np.sum(probs, dtype=np.float32)).astype(np.float32)
        context = np.zeros(W, dtype=np.float32)
        for probability, old in zip(probs, values):
            context += probability * old
        attention = (weights["o"] @ context).astype(np.float32)
        after = (rms(attention, norms) + residual).astype(np.float32)
        dense_in = rms(after, norms)
        dense = (weights["dense_down"] @ (gelu(weights["dense_gate"] @ dense_in) *
                 (weights["dense_up"] @ dense_in))).astype(np.float32)
        router_in = rms(after, norms)
        route = weights["router"] @ router_in
        expert_id = int(np.argmax(route))
        expert_in = rms(after, norms)
        gate, up, down = weights["experts"][expert_id]
        expert = (down @ (gelu(gate @ expert_in) * (up @ expert_in))).astype(np.float32)
        combined = (rms(dense, norms) + rms(expert, norms)).astype(np.float32)
        final = (after + rms(combined, norms)).astype(np.float32)
        logits = (weights["output"] @ rms(final, norms)).astype(np.float32)
        if step == len(prompt) - 1:
            first_logits = logits.copy()
        if step >= len(prompt) - 1 and len(generated) < new_tokens:
            nxt = int(np.argmax(logits))
            generated.append(nxt)
            sequence.append(nxt)
            if nxt == 7:
                break
    return first_logits, generated


def write_fixture(out: Path, new_tokens: int):
    out.mkdir(parents=True, exist_ok=True)
    packed = out / "packed"
    packed.mkdir(exist_ok=True)
    embedding_raw, embedding = q6_rows(VOCAB)
    matrices = {}
    decoded = {"embedding": embedding}
    output_raw, output_values = q4_matrix(VOCAB, W, 17)
    decoded["output"] = output_values
    specs = [
        ("blk.0.attn_q.weight", W, W, 0, False, "q"),
        ("blk.0.attn_k.weight", W, W, 3, False, "k"),
        ("blk.0.attn_v.weight", W, W, 7, False, "v"),
        ("blk.0.attn_output.weight", W, W, 11, False, "o"),
        ("blk.0.ffn_gate.weight", FF, W, 2, False, "dense_gate"),
        ("blk.0.ffn_up.weight", FF, W, 5, False, "dense_up"),
        ("blk.0.ffn_down.weight", W, FF, 1, False, "dense_down"),
    ]
    for name, rows, cols, shift, zero, key in specs:
        raw, values = q4_matrix(rows, cols, shift, zero)
        matrices[name] = raw; decoded[key] = values
    router = np.zeros((EXPERTS, W), dtype=np.float32)
    router[0, 0] = 0.5; router[1, 1] = 0.5
    decoded["router"] = router
    expert_values, expert_payloads = [], []
    for expert in range(EXPERTS):
        parts, vals = [], []
        for rows, cols, shift in ((FF, W, expert), (FF, W, expert + 4), (W, FF, expert + 1)):
            raw, value = q4_matrix(rows, cols, shift)
            parts.append(raw); vals.append(value)
        expert_payloads.append(b"".join(parts)); expert_values.append(tuple(vals))
    decoded["experts"] = expert_values

    tensors = [("token_embd.weight", 14, [W, VOCAB], embedding_raw),
               ("output.weight", 2, [W, VOCAB], output_raw)]
    tensors += [(name, 2, [cols, rows], matrices[name]) for name, rows, cols, *_ in specs]
    ones = np.ones(W, dtype="<f4").tobytes()
    for name in ("blk.0.attn_norm.weight", "blk.0.post_attention_norm.weight",
                 "blk.0.ffn_norm.weight", "blk.0.post_ffw_norm_1.weight",
                 "blk.0.pre_ffw_norm_2.weight", "blk.0.post_ffw_norm_2.weight",
                 "blk.0.post_ffw_norm.weight", "output_norm.weight"):
        tensors.append((name, 0, [W], ones))
    tensors += [
        ("blk.0.attn_q_norm.weight", 0, [W], ones),
        ("blk.0.attn_k_norm.weight", 0, [W], ones),
        ("blk.0.ffn_gate_inp.scale", 0, [W], ones),
        ("blk.0.ffn_gate_inp.weight", 0, [W, EXPERTS], router.astype("<f4").tobytes()),
        ("blk.0.ffn_down_exps.scale", 0, [EXPERTS], np.ones(EXPERTS, dtype="<f4").tobytes()),
        ("blk.0.layer_output_scale.weight", 0, [1], pf(1.0)),
        ("rope_freqs.weight", 0, [W // 2], np.ones(W // 2, dtype="<f4").tobytes()),
    ]

    meta = []
    def ms(k, v): meta.append(ps(k) + p32(8) + ps(v))
    def mu(k, v): meta.append(ps(k) + p32(4) + p32(v))
    def mf(k, v): meta.append(ps(k) + p32(6) + pf(v))
    def ma(k, typ, vals, encoder):
        meta.append(ps(k) + p32(9) + p32(typ) + p64(len(vals)) + b"".join(encoder(x) for x in vals))
    ms("general.architecture", "gemma4"); mu("general.alignment", 32)
    mu("gemma4.block_count", 1); mu("gemma4.embedding_length", W)
    mu("gemma4.expert_count", EXPERTS); mu("gemma4.expert_used_count", TOPK)
    mu("gemma4.expert_feed_forward_length", FF); mu("gemma4.attention.sliding_window", 64)
    mu("gemma4.attention.head_count", 1); mu("gemma4.attention.key_length", W)
    mu("gemma4.attention.value_length", W); mf("gemma4.attention.layer_norm_rms_epsilon", float(EPS))
    mf("gemma4.rope.freq_base", 10000.0); mf("gemma4.final_logit_softcapping", 0.0)
    ma("gemma4.attention.head_count_kv", 5, [1], p32)
    ma("gemma4.attention.sliding_window_pattern", 7, [0], lambda x: bytes([x]))
    ms("tokenizer.ggml.model", "gemma4")
    ma("tokenizer.ggml.tokens", 8, TOKENS, ps)
    ma("tokenizer.ggml.merges", 8, ["A A"], ps)
    ma("tokenizer.ggml.token_type", 5, [3, 1, 1, 1, 1, 1, 1, 3], p32)
    mu("tokenizer.ggml.bos_token_id", 0); mu("tokenizer.ggml.eos_token_id", 7)
    mu("tokenizer.ggml.unknown_token_id", 0); mu("tokenizer.ggml.padding_token_id", 0)
    meta.append(ps("tokenizer.ggml.add_bos_token") + p32(7) + b"\x01")

    descriptors, offset = [], 0
    for name, typ, dims, data in tensors:
        desc = ps(name) + p32(len(dims)) + b"".join(p64(x) for x in dims) + p32(typ) + p64(offset)
        descriptors.append(desc); offset += len(data)
    header = b"GGUF" + p32(3) + p64(len(tensors)) + p64(len(meta)) + b"".join(meta) + b"".join(descriptors)
    data_offset = (len(header) + 31) & ~31
    model = out / "model.gguf"
    model.write_bytes(header + bytes(data_offset - len(header)) + b"".join(x[3] for x in tensors))

    gate_bytes = FF * (W // 32) * 18
    down_bytes = W * (FF // 32) * 18
    payload_bytes = gate_bytes * 2 + down_bytes
    stride = (payload_bytes + 4095) & ~4095
    h = bytearray(4096); h[:8] = b"G4EXPK01"
    struct.pack_into("<IIIII", h, 8, 1, 0, EXPERTS, 2, 4096)
    struct.pack_into("<QQQQQQ", h, 32, 4096, payload_bytes, stride, gate_bytes, gate_bytes, down_bytes)
    layer = bytearray(h)
    for payload in expert_payloads:
        layer += payload + bytes(stride - len(payload))
    (packed / "layer-00.g4ex").write_bytes(layer)
    component = lambda role, size, within: {
        "role": role, "tensor": "synthetic", "ggml_type": 2, "type_name": "Q4_0",
        "slice_bytes": size, "source_offset": 0,
        "source_expert_stride": payload_bytes, "source_within_expert_offset": within,
    }
    manifest = {
        "format": "g4lab-expert-manifest-v2", "architecture": "gemma4",
        "source": str(model.resolve()), "expert_count": EXPERTS,
        "expert_used_count": TOPK, "layers": [{
            "layer": 0, "expert_count": EXPERTS, "model_width": W,
            "expert_width": FF, "source_layout": "fused_gate_up",
            "packed_file": "layer-00.g4ex", "payload_bytes": payload_bytes,
            "record_stride": stride, "expert_scale": None,
            "components": [component("gate", gate_bytes, 0),
                           component("up", gate_bytes, gate_bytes),
                           component("down", down_bytes, gate_bytes * 2)],
        }],
    }
    (packed / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logits, generated = reference(decoded, [0, 1], new_tokens)
    ref = {"format": "colibri-gemma4-tiny-v1", "prompt": "A",
           "prompt_tokens": [0, 1], "logits": [float(x) for x in logits],
           "generated_tokens": generated,
           "generated_text": "".join(TOKENS[x] for x in generated)}
    (out / "ref.json").write_text(json.dumps(ref, indent=2), encoding="utf-8")
    print(f"wrote {model}, {packed}, and {out / 'ref.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("gemma4_tiny"))
    parser.add_argument("--max-new", type=int, default=6)
    args = parser.parse_args()
    write_fixture(args.out, args.max_new)
