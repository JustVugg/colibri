"""A tiny qwen35 pack in Basalt's `.basalt` form, for tests/test_convert_basalt_to_gguf.py.

The container is written from Basalt v0.4.0's format header (include/basalt/core/model_format.hpp, MIT): a 64 KiB
head (64-byte FileHeader: magic BASALTMF, u32 version 2, u32 n_sections, u64 file_bytes; then 128-byte SectionEntry
records: name[64], u64 offset, u64 bytes, sha256[32]), every section at a multiple of 2 MiB, zeros between. The
`gguf` section and `meta` follow what Basalt's tools/convert/checkpoint_qwen35.py writes for the QUASAR pack
(Qwen3.8-27B-QUASAR-NVFP4.basalt): llama.cpp qwen35 names, projections NVFP4_PLANES (type 45) with `<name>.scale`
F32 [1], the head F8_E4M3 (type 44) with `output.scale` F32 [rows], a / b projections and the embedding BF16, norms
F32 (offset folded), the MTP layer as mtp.* in Q8_0, the DFlash2 drafter as dflash.* with qwen35.dflash.* keys,
block_count the main layers and nextn_predict_layers 1. Tensors start at multiples of 4096 in `gguf`.

write_pack(path, ref=False): ref=True writes the same model with every NVFP4 / F8 matrix as F32 of its values (the
scale folded in), the oracle the engine's run on the converted file is compared with. Returns the values written.
"""
import hashlib
import json
import struct

import numpy as np

H, FF, VOCAB = 256, 256, 256
NH, NKV, HD = 2, 1, 128                       # attention: q rows 2*NH*HD (query and its gate)
KH, VH, KD = 2, 4, 64                         # DeltaNet key heads, value heads, head dim
CONV = 2 * KH * KD + VH * KD                  # 512
DF_L, DF_NH, DF_NKV, DF_HD, DF_I, DF_RANK = 1, 2, 1, 64, 128, 32
DF_TARGETS = [0, 1]

SECTION_ALIGN, HEADER_BYTES, TENSOR_ALIGN = 2 << 20, 64 << 10, 4096
TYPES = {"F32": 0, "Q8_0": 8, "BF16": 30, "F8_E4M3": 44, "NVFP4_PLANES": 45}

E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float64)


def e4m3(c):
    e, m = (c >> 3) & 15, c & 7
    v = np.where(e == 0, m / 8 * 2.0 ** -6, (1 + m / 8) * 2.0 ** (e - 7))
    return np.where(c & 0x80, -v, v)


E4M3 = e4m3(np.arange(256))


def scale_index(r, g, K):
    return ((r // 128) * (K // 64) + g // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + g % 4


def q8_0(x):
    b = x.reshape(-1, 32).astype(np.float32)
    d = np.abs(b).max(axis=1) / np.float32(127)
    inv = np.where(d > 0, np.float32(1) / np.where(d > 0, d, np.float32(1)), np.float32(0)).astype(np.float32)
    v = b * inv[:, None]
    q = (np.sign(v) * np.floor(np.abs(v) + np.float32(0.5))).astype(np.int8)
    out = np.empty((b.shape[0], 34), np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()


def q8_values(raw):
    b = np.frombuffer(raw, np.uint8).reshape(-1, 34)
    return np.ascontiguousarray(b[:, :2]).view("<f2").astype(np.float32) * b[:, 2:].view(np.int8).astype(np.float32)


def bf16_bits(x):
    return (np.ascontiguousarray(x, np.float32).view(np.uint32) >> 16).astype("<u2")


class Pack:
    def __init__(self, seed, ref):
        self.rng = np.random.default_rng(seed)
        self.ref = ref
        self.tensors = {}     # name -> (type, ne, bytes)
        self.planes = {}      # name -> (codes [rows, K/2], scales [rows, K/16], tensor scale)
        self.f8 = {}          # name -> (codes [rows, K], row scales)
        self.values = {}      # name -> float values [rows, K] (float64) of every matrix

    def put(self, name, ty, ne, data):
        assert name not in self.tensors
        self.tensors[name] = (ty, list(ne), data)

    def f32(self, name, x):
        x = np.asarray(x, np.float32)
        self.put(name, "F32", list(reversed(x.shape)), x.astype("<f4").tobytes())

    def norm(self, name, n):   # stored with the offset folded: 1 + w
        self.f32(name, 1 + self.rng.standard_normal(n) * 0.1)

    def bf16(self, name, rows, cols):
        x = (self.rng.standard_normal((rows, cols)) * 0.05).astype(np.float32)
        u = bf16_bits(x)
        self.values[name] = (u.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        self.put(name, "BF16", [cols, rows], u.tobytes())

    def q8(self, name, rows, cols):
        raw = q8_0((self.rng.standard_normal((rows, cols)) * 0.05).astype(np.float32))
        self.values[name] = q8_values(raw).reshape(rows, cols).astype(np.float64)
        self.put(name, "Q8_0", [cols, rows], raw)

    def nvfp4(self, name, rows, cols):
        """codes random, UE4M3 group scales 0.125 .. 0.5 (and a zero group), a tensor scale near 0.03"""
        codes = self.rng.integers(0, 256, (rows, cols // 2), dtype=np.uint8)
        sc = self.rng.integers(0x20, 0x31, (rows, cols // 16), dtype=np.uint8)
        sc[0, 0] = 0
        gs = np.float32(0.02 + 0.01 * self.rng.random())
        v = np.empty((rows, cols))
        v[:, 0::2], v[:, 1::2] = E2M1[codes & 15], E2M1[codes >> 4]
        x = v * np.repeat(E4M3[sc], 16, axis=1)            # e2m1 x ue4m3, exact
        self.planes[name] = (codes, sc, gs)
        self.values[name] = x * np.float64(gs)
        base = name[:-len(".weight")]
        if self.ref:
            self.f32(name, (x * np.float64(gs)).astype(np.float32))
            return
        r, g = np.meshgrid(np.arange(rows), np.arange(cols // 16), indexing="ij")
        tiles = np.zeros(rows * cols // 16, np.uint8)
        tiles[scale_index(r, g, cols)] = sc
        self.put(name, "NVFP4_PLANES", [cols, rows], codes.tobytes() + tiles.tobytes())
        self.f32(base + ".scale", np.array([gs], np.float32))

    def f8_rows(self, name, rows, cols):
        """the QUASAR head: E4M3 codes (no NaN), a scale per row"""
        codes = self.rng.integers(0, 0x7F, (rows, cols), dtype=np.uint8) | (self.rng.integers(0, 2, (rows, cols), dtype=np.uint8) << 7)
        rs = (np.float32(1e-4) * (1 + self.rng.random(rows))).astype(np.float32)
        self.f8[name] = (codes, rs)
        x = (E4M3[codes].astype(np.float32) * rs[:, None]).astype(np.float32)
        self.values[name] = x.astype(np.float64)
        if self.ref:
            self.f32(name, x)
            return
        self.put(name, "F8_E4M3", [cols, rows], codes.tobytes())
        self.f32(name[:-len(".weight")] + ".scale", rs)


def model(seed=11, ref=False):
    p = Pack(seed, ref)
    p.bf16("token_embd.weight", VOCAB, H)
    p.norm("output_norm.weight", H)
    p.f8_rows("output.weight", VOCAB, H)
    # layer 0: DeltaNet
    b = "blk.0."
    p.norm(b + "attn_norm.weight", H)
    p.norm(b + "post_attention_norm.weight", H)
    p.nvfp4(b + "attn_qkv.weight", CONV, H)
    p.nvfp4(b + "attn_gate.weight", VH * KD, H)
    p.nvfp4(b + "ssm_out.weight", H, VH * KD)
    p.bf16(b + "ssm_alpha.weight", VH, H)
    p.bf16(b + "ssm_beta.weight", VH, H)
    p.f32(b + "ssm_a", -np.exp(p.rng.standard_normal(VH) * 0.1))
    p.f32(b + "ssm_dt.bias", p.rng.standard_normal(VH) * 0.1)
    p.f32(b + "ssm_conv1d.weight", p.rng.standard_normal((CONV, 4)) * 0.2)
    p.f32(b + "ssm_norm.weight", 1 + p.rng.standard_normal(KD) * 0.1)
    for n, rows, cols in (("ffn_gate", FF, H), ("ffn_up", FF, H), ("ffn_down", H, FF)):
        p.nvfp4(b + n + ".weight", rows, cols)
    # layer 1: gated attention
    b = "blk.1."
    p.norm(b + "attn_norm.weight", H)
    p.norm(b + "post_attention_norm.weight", H)
    p.nvfp4(b + "attn_q.weight", 2 * NH * HD, H)
    p.nvfp4(b + "attn_k.weight", NKV * HD, H)
    p.nvfp4(b + "attn_v.weight", NKV * HD, H)
    p.nvfp4(b + "attn_output.weight", H, NH * HD)
    p.norm(b + "attn_q_norm.weight", HD)
    p.norm(b + "attn_k_norm.weight", HD)
    for n, rows, cols in (("ffn_gate", FF, H), ("ffn_up", FF, H), ("ffn_down", H, FF)):
        p.nvfp4(b + n + ".weight", rows, cols)
    # the MTP layer (BF16 in the checkpoint: Q8_0 in the pack)
    p.q8("mtp.fc.weight", H, 2 * H)
    for n in ("norm", "pre_fc_norm_embedding", "pre_fc_norm_hidden", "attn_norm", "post_attention_norm"):
        p.norm(f"mtp.{n}.weight", H)
    for n, rows, cols in (("attn_q", 2 * NH * HD, H), ("attn_k", NKV * HD, H), ("attn_v", NKV * HD, H),
                          ("attn_output", H, NH * HD)):
        p.q8(f"mtp.{n}.weight", rows, cols)
    p.norm("mtp.attn_q_norm.weight", HD)
    p.norm("mtp.attn_k_norm.weight", HD)
    for n, rows, cols in (("ffn_gate", FF, H), ("ffn_up", FF, H), ("ffn_down", H, FF)):
        p.q8(f"mtp.{n}.weight", rows, cols)
    # the DFlash2 drafter (Basalt's Drafter: projections Q8_0, norms and base kernels F32, selector BF16)
    for j in range(len(DF_TARGETS)):
        p.q8(f"dflash.fc.{j}.weight", H, H)
    p.f32("dflash.hidden_norm.weight", 1 + p.rng.standard_normal(H) * 0.1)
    p.f32("dflash.norm.weight", 1 + p.rng.standard_normal(H) * 0.1)
    p.bf16("dflash.selector.hidden.weight", DF_RANK, H)
    p.bf16("dflash.selector.pred", VOCAB, DF_RANK)
    p.bf16("dflash.selector.succ", VOCAB, DF_RANK)
    G = H // 16
    for l in range(DF_L):
        b = f"dflash.blk.{l}."
        p.f32(b + "attn_norm.weight", 1 + p.rng.standard_normal(H) * 0.1)
        p.f32(b + "post_attention_norm.weight", 1 + p.rng.standard_normal(H) * 0.1)
        for n, rows, cols in (("attn_q", DF_NH * DF_HD, H), ("attn_k", DF_NKV * DF_HD, H), ("attn_v", DF_NKV * DF_HD, H),
                              ("attn_output", H, DF_NH * DF_HD), ("ffn_gate", DF_I, H), ("ffn_up", DF_I, H),
                              ("ffn_down", H, DF_I), ("attn_conv.weight", 4 * G, H), ("ffn_conv.weight", 4 * G, H)):
            p.q8(b + n + ("" if n.endswith(".weight") else ".weight"), rows, cols)
        p.f32(b + "attn_q_norm.weight", 1 + p.rng.standard_normal(DF_HD) * 0.1)
        p.f32(b + "attn_k_norm.weight", 1 + p.rng.standard_normal(DF_HD) * 0.1)
        p.f32(b + "attn_conv.base", p.rng.standard_normal((H, 4)) * 0.1)
        p.f32(b + "ffn_conv.base", p.rng.standard_normal((H, 4)) * 0.1)
    return p


def gguf_kv():
    A = "qwen35."
    toks = [f"<0x{i:02X}>" for i in range(VOCAB - 3)] + ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    return {
        "general.architecture": "qwen35", "general.type": "model", "general.name": "tiny QUASAR-form pack",
        "general.sampling.top_k": 20, "general.sampling.top_p": 0.949999988079071,
        A + "block_count": 2, A + "context_length": 4096, A + "embedding_length": H, A + "feed_forward_length": FF,
        A + "attention.head_count": NH, A + "attention.head_count_kv": NKV, A + "attention.key_length": HD,
        A + "attention.value_length": HD, A + "attention.layer_norm_rms_epsilon": 9.999999974752427e-07,
        A + "rope.dimension_count": HD // 4, A + "rope.dimension_sections": [11, 11, 10, 0],
        A + "rope.freq_base": 10000000.0, A + "ssm.conv_kernel": 4, A + "ssm.state_size": KD,
        A + "ssm.group_count": KH, A + "ssm.time_step_rank": VH, A + "ssm.inner_size": VH * KD,
        A + "attention.recurrent_layers": [True, False], A + "full_attention_interval": 2,
        A + "nextn_predict_layers": 1, "general.file_type": 40, "general.quantization_version": 2,
        "tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "qwen35", "tokenizer.ggml.tokens": toks,
        "tokenizer.ggml.token_type": [1] * (VOCAB - 3) + [3, 3, 3], "tokenizer.ggml.merges": [],
        "tokenizer.ggml.eos_token_id": VOCAB - 1, "tokenizer.ggml.padding_token_id": VOCAB - 3,
        "tokenizer.ggml.add_bos_token": False, "tokenizer.chat_template": "{{ messages }}",
        A + "dflash.block_count": DF_L, A + "dflash.block_size": 8, A + "dflash.mask_token_id": VOCAB - 3,
        A + "dflash.target_layers": DF_TARGETS, A + "dflash.conv_kernel": 2, A + "dflash.conv_group": 16,
        A + "dflash.selector_rank": DF_RANK, A + "dflash.selector_top_k": 8, A + "dflash.sliding_window": 2048,
        A + "dflash.attention.head_count": DF_NH, A + "dflash.attention.head_count_kv": DF_NKV,
        A + "dflash.attention.key_length": DF_HD, A + "dflash.feed_forward_length": DF_I,
        A + "dflash.rope.freq_base": 10000000.0, A + "dflash.attention.layer_norm_rms_epsilon": 9.999999974752427e-07,
    }


def write_container(path, sections, version=2):
    """sections: [(name, bytes)] -> a .basalt file (model_format.hpp)."""
    with open(path, "wb") as f:
        f.write(b"\0" * HEADER_BYTES)
        entries = []
        for name, data in sections:
            off = (f.tell() + SECTION_ALIGN - 1) // SECTION_ALIGN * SECTION_ALIGN
            f.write(b"\0" * (off - f.tell()))
            f.write(data)
            entries.append((name, off, len(data), hashlib.sha256(data).digest()))
        end = f.tell()
        f.seek(0)
        f.write(struct.pack("<8sIIQ40s", b"BASALTMF", version, len(entries), end, b"\0" * 40))
        for name, off, n, digest in entries:
            f.write(struct.pack("<64sQQ32s16s", name.encode(), off, n, digest, b"\0" * 16))


def write_pack(path, ref=False, seed=11):
    p = model(seed, ref)
    gg, tens = bytearray(), {}
    for name, (ty, ne, data) in p.tensors.items():
        gg += b"\0" * ((-len(gg)) % TENSOR_ALIGN)
        tens[name] = {"type": ty, "type_id": TYPES[ty], "shape": ne, "bytes": len(data), "offset": len(gg)}
        gg += data
    meta = {"format": "basalt", "version": 2, "name": "tiny", "arch": "qwen35", "gguf_kv": gguf_kv(), "tensors": tens}
    write_container(path, [("tokenizer/tokenizer.json", b"{}"), ("gguf", bytes(gg)), ("vision/mmproj", b"GGUF"),
                           ("meta", json.dumps(meta).encode())])
    return p
