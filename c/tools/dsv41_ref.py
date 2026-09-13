#!/usr/bin/env python3
"""Torch-only reference for DeepSeek-V4.1-Flash, and the quantizers its container uses.

The vendor's `inference/model.py` is the authority on this architecture, but it cannot
be the oracle: its attention, quantization and Sinkhorn all run through tilelang JIT
kernels that need a GPU. This file re-implements the same math in plain torch on the
CPU so `make_dsv41_tiny.py` can emit a token-exact reference the C engine is held to.

Everything here mirrors a named piece of the vendor file; where the two could drift the
vendor's line is quoted. Deviations that are deliberate:

- fp32 everywhere. The vendor keeps activations in bf16 between sublayers; colibri's
  engines carry fp32. On the tiny fixture the two agree token-for-token, and fp32 is
  what the C side has to reproduce, so the oracle is written in it.
- weights are dequantized once, up front, exactly as the engine dequantizes them
  (`fp8_dequant` / `fp4_dequant` below). The oracle therefore validates the engine's
  arithmetic, not its ability to guess a rounding mode.
- no distributed anything: world_size == 1, so the parallel/sharded classes collapse.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------- quantization

E4M3_MAX = 448.0
# e2m1: 3 magnitude bits index this table, bit 3 is the sign (quant.h, matmul_mxfp4)
E2M1_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_MAX = 6.0


def _pow2(exponent: torch.Tensor) -> torch.Tensor:
    """2**e for an integer tensor, as fp32. torch.ldexp keeps this exact."""
    return torch.ldexp(torch.ones_like(exponent, dtype=torch.float32), exponent)


def ue8m0_exponent(amax: torch.Tensor, dtype_max: float) -> torch.Tensor:
    """The ue8m0 byte for a block whose largest magnitude is `amax`.

    ue8m0 stores a power of two as a bare exponent byte with bias 127, so the scale can
    only be 2**k: pick the smallest k that brings the block inside the target format's
    range. amax == 0 gives k = 0 rather than -inf, which keeps an all-zero block's byte
    at 127 instead of 0 (0 would decode as a denormal and the SIMD path in quant.h
    flushes it to +0 -- harmless here, but a needless difference between the two paths).
    """
    safe = torch.where(amax > 0, amax, torch.ones_like(amax))
    exponent = torch.ceil(torch.log2(safe / dtype_max))
    exponent = torch.where(amax > 0, exponent, torch.zeros_like(exponent))
    return exponent.to(torch.int32).clamp(-127, 128)


def fp8_quant_blocked(w: torch.Tensor, block: int = 32):
    """Quantize [O, I] to e4m3 bytes with one ue8m0 scale per `block` x `block` tile.

    Returns (bytes [O, I] uint8-viewed e4m3, scale bytes [O/block, I/block] uint8).
    The released checkpoint stores exactly this: `attn.wkv.weight F8_E4M3 [512, 5120]`
    beside `attn.wkv.scale F8_E8M0 [16, 160]`, i.e. 512/16 == 5120/160 == 32.
    """
    rows, cols = w.shape
    assert rows % block == 0 and cols % block == 0, (rows, cols, block)
    tiles = w.reshape(rows // block, block, cols // block, block).permute(0, 2, 1, 3)
    exponent = ue8m0_exponent(tiles.abs().amax(dim=(-2, -1)), E4M3_MAX)
    scaled = tiles / _pow2(exponent)[..., None, None]
    q = scaled.to(torch.float8_e4m3fn)
    packed = q.permute(0, 2, 1, 3).reshape(rows, cols)
    return packed, (exponent + 127).to(torch.uint8)


def fp8_dequant_blocked(q: torch.Tensor, scale: torch.Tensor, block: int = 32) -> torch.Tensor:
    """Inverse of fp8_quant_blocked; this is the read path the C engine implements."""
    rows, cols = q.shape
    values = q.to(torch.float32).reshape(rows // block, block, cols // block, block).permute(0, 2, 1, 3)
    exponent = scale.to(torch.int32) - 127
    out = values * _pow2(exponent)[..., None, None]
    return out.permute(0, 2, 1, 3).reshape(rows, cols)


def fp4_quant_rowgroups(w: torch.Tensor, group: int = 32):
    """Quantize [O, I] to packed e2m1 nibbles with one ue8m0 scale per row group of 32.

    Returns (packed [O, I/2] uint8, scale [O, I/group] uint8). This is the released
    expert layout: `ffn.experts.0.w1.weight I8 [2304, 2560]` for a [2304, 5120] matrix
    beside `w1.scale F8_E8M0 [2304, 160]`, and it is also colibri's existing mxfp4
    layout, so `matmul_mxfp4` in quant.h reads it unchanged.

    LOW nibble = even column, exactly as quant.h documents and as torch's
    float4_e2m1fn_x2 packs.
    """
    rows, cols = w.shape
    assert cols % group == 0 and group % 2 == 0, (cols, group)
    grouped = w.reshape(rows, cols // group, group)
    exponent = ue8m0_exponent(grouped.abs().amax(dim=-1), E2M1_MAX)
    scaled = grouped / _pow2(exponent)[..., None]
    levels = E2M1_LEVELS.to(scaled.dtype)
    # nearest level, ties away from zero -- torch.bucketize on midpoints is exact and
    # avoids depending on any float8/float4 cast rounding mode
    mids = (levels[1:] + levels[:-1]) / 2
    idx = torch.bucketize(scaled.abs().contiguous(), mids.contiguous())
    nibble = idx.to(torch.uint8) | (torch.signbit(scaled).to(torch.uint8) << 3)
    nibble = nibble.reshape(rows, cols)
    packed = nibble[:, 0::2] | (nibble[:, 1::2] << 4)
    return packed.contiguous(), (exponent + 127).to(torch.uint8)


def fp4_dequant_rowgroups(packed: torch.Tensor, scale: torch.Tensor, cols: int, group: int = 32) -> torch.Tensor:
    rows = packed.shape[0]
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    nibble = torch.stack([low, high], dim=-1).reshape(rows, cols).to(torch.int64)
    values = E2M1_LEVELS.to(torch.float32)[nibble & 0x7] * torch.where(
        (nibble & 0x8) != 0, -1.0, 1.0
    )
    exponent = scale.to(torch.int32) - 127
    return (values.reshape(rows, cols // group, group) * _pow2(exponent)[..., None]).reshape(rows, cols)


def fp8_quant_rowgroups(w: torch.Tensor, group: int = 32):
    """e4m3 values with one ue8m0 scale per row group -- the engram table's layout
    (`embed.weight F8_E4M3 [rows, 256]` beside `embed.scale F8_E8M0 [rows, 8]`)."""
    rows, cols = w.shape
    assert cols % group == 0
    grouped = w.reshape(rows, cols // group, group)
    exponent = ue8m0_exponent(grouped.abs().amax(dim=-1), E4M3_MAX)
    q = (grouped / _pow2(exponent)[..., None]).to(torch.float8_e4m3fn)
    return q.reshape(rows, cols), (exponent + 127).to(torch.uint8)


def fp8_dequant_rowgroups(q: torch.Tensor, scale: torch.Tensor, group: int = 32) -> torch.Tensor:
    rows, cols = q.shape
    exponent = scale.to(torch.int32) - 127
    values = q.to(torch.float32).reshape(rows, cols // group, group)
    return (values * _pow2(exponent)[..., None]).reshape(rows, cols)


# ------------------------------------------------------------------- primitives


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """model.py RMSNorm: the mean of squares, in fp32, weight applied after."""
    var = x.float().square().mean(-1, keepdim=True)
    return weight.float() * x.float() * torch.rsqrt(var + eps)


def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    """model.py precompute_freqs_cis, YaRN included. Returns [seqlen, dim/2] complex."""

    def find_correction_dim(num_rotations, d, b, max_seq_len):
        return d * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(b))

    def find_correction_range(low_rot, high_rot, d, b, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, d, b, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, d, b, max_seq_len))
        return max(low, 0), min(high, d - 1)

    def linear_ramp_factor(minimum, maximum, d):
        if minimum == maximum:
            maximum += 0.001
        ramp = (torch.arange(d, dtype=torch.float32) - minimum) / (maximum - minimum)
        return torch.clamp(ramp, 0, 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0 and seqlen > original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    t = torch.arange(seqlen, dtype=torch.float32)
    return torch.polar(torch.ones(seqlen, dim // 2), torch.outer(t, freqs))


def apply_rotary(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """model.py apply_rotary_emb. `x` is the RoPE tail only, shaped [..., rope_dim];
    freqs_cis broadcasts over the leading dims. `inverse` rotates the other way, which
    the output projection uses to undo the query rotation."""
    shape = x.shape
    pairs = torch.view_as_complex(x.float().reshape(*shape[:-1], -1, 2))
    while freqs_cis.dim() < pairs.dim():
        freqs_cis = freqs_cis.unsqueeze(-2)
    rotated = pairs * (freqs_cis.conj() if inverse else freqs_cis)
    return torch.view_as_real(rotated).reshape(shape)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, iters, eps):
    """kernel.py hc_split_sinkhorn, in torch. mixes: [n, (2+hc)*hc]."""
    pre = torch.sigmoid(mixes[..., :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult])
    comb = mixes[..., 2 * hc_mult :] * hc_scale[2] + hc_base[2 * hc_mult :]
    comb = comb.reshape(*comb.shape[:-1], hc_mult, hc_mult)
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """kernel.py sparse_attn, in torch.

    q: [s, h, d]; kv: [n, d] (one KV per position -- this model is MQA, num_key_value
    heads == 1); topk_idxs: [s, topk] with -1 for "nothing here". The sink adds
    exp(sink_h - max) to the denominator only, so a row with no valid index comes out
    all zero instead of NaN -- the kernel's -1e30 floor does the same.
    """
    s, h, d = q.shape
    out = torch.zeros_like(q)
    for i in range(s):
        idx = topk_idxs[i]
        valid = idx >= 0
        scores = torch.full((h, idx.numel()), -float("inf"))
        if valid.any():
            keys = kv[idx[valid]]                      # [k, d]
            scores[:, valid] = (q[i] @ keys.T) * softmax_scale
        row_max = torch.maximum(scores.max(dim=-1).values, torch.full((h,), -1e30))
        weights = torch.exp(scores - row_max[:, None])
        weights = torch.where(torch.isfinite(scores), weights, torch.zeros_like(weights))
        denom = weights.sum(dim=-1) + torch.exp(attn_sink.float() - row_max)
        if valid.any():
            out[i] = (weights[:, valid] @ kv[idx[valid]]) / denom[:, None]
        else:
            out[i] = 0
    return out


# ------------------------------------------------------------------- n-gram hash


class NgramHash:
    """engram.py NgramHashState, driven by tables computed once at conversion time.

    The vendor builds the compressed-token map from the tokenizer, the bucket primes with
    sympy and the per-(layer, lookback) multipliers from `np.random.default_rng(10007 *
    layer_id)`. None of that belongs in a C engine at load time, so `prepare_dsv41.py`
    computes it and ships it in the container; this class and the engine both read those
    arrays. The hash itself is the part that must agree byte for byte, so it lives here
    in one place.
    """

    DEAD = -1

    def __init__(self, token_map, primes, offsets, multipliers, pad_id, max_ngram_size, n_heads):
        self.token_map = token_map            # [vocab] int64
        self.primes = primes                  # [n_layers, max_ngram-1, n_heads] int64
        self.offsets = offsets                # [n_layers, (max_ngram-1)*n_heads] int64
        self.multipliers = multipliers        # [n_layers, max_ngram] int64
        self.pad_id = pad_id
        self.max_ngram_size = max_ngram_size
        self.n_heads = n_heads
        self.history: list[int] = []          # compressed ids, DEAD for image positions

    def reset(self):
        self.history = []

    def push(self, compressed_ids: list[int]) -> torch.Tensor:
        """Extend the history and return the hash ids of the new positions,
        shaped [n_new, n_engram_layers, (max_ngram_size - 1) * n_heads]."""
        start = len(self.history)
        self.history.extend(compressed_ids)
        rows = []
        for pos in range(start, len(self.history)):
            tokens, blocked = [], False
            for shift in range(self.max_ngram_size):
                source = self.history[pos - shift] if pos - shift >= 0 else self.pad_id
                blocked = blocked or pos < shift or source == self.DEAD
                tokens.append(self.pad_id if blocked else source)
            tokens = torch.tensor(tokens, dtype=torch.int64)
            products = tokens[None, :] * self.multipliers          # [layers, max_ngram]
            rolling, hashes = products[:, 0], []
            for i in range(1, self.max_ngram_size):
                rolling = torch.bitwise_xor(rolling, products[:, i])
                hashes.append(rolling[:, None] % self.primes[:, i - 1])
            rows.append(torch.cat(hashes, dim=-1) + self.offsets)
        return torch.stack(rows) if rows else torch.zeros(0, self.primes.shape[0], 0, dtype=torch.int64)


# ------------------------------------------------------------------- the model


class RefModel:
    """One-batch reference. `w` holds every tensor already dequantized to fp32, keyed by
    the checkpoint's own names, so this class never sees a scale."""

    def __init__(self, cfg: dict, w: dict, ngram: NgramHash | None = None):
        self.c = cfg
        self.w = w
        self.ngram = ngram
        c = cfg
        self.hc = c["hc_mult"]
        self.dim = c["dim"]
        self.rd = c["rope_head_dim"]
        self.nlayers = c["n_layers"]
        self.max_seq = c["max_seq_len"]
        # RoPE tables: a compressing layer rotates at compress_rope_theta with YaRN, a
        # window-only layer at rope_theta with none (model.py Attention.__init__).
        self.freqs_compress = precompute_freqs_cis(
            self.rd, self.max_seq, c["original_seq_len"], c["compress_rope_theta"],
            c["rope_factor"], c["beta_fast"], c["beta_slow"])
        self.freqs_window = precompute_freqs_cis(
            self.rd, self.max_seq, 0, c["rope_theta"], c["rope_factor"], c["beta_fast"], c["beta_slow"])
        self.reset()

    def freqs(self, layer: int) -> torch.Tensor:
        return self.freqs_compress if self.c["compress_ratios"][layer] else self.freqs_window

    def reset(self):
        c, hd = self.c, self.c["head_dim"]
        self.pos = 0
        self.window = [torch.zeros(c["window_size"], hd) for _ in range(self.nlayers)]
        self.compress_kv = {}
        self.index_k = {}
        self.comp_state = {}
        for layer in c["kv_source_layers"]:
            ratio = c["compress_ratios"][layer]
            self.compress_kv[layer] = torch.zeros(self.max_seq // max(ratio, 1), hd)
            self.index_k[layer] = torch.zeros(self.max_seq // max(ratio, 1), c["index_head_dim"])
            self.comp_state[layer] = (torch.zeros(ratio, hd), torch.full((ratio, hd), -float("inf")))
        if self.ngram is not None:
            self.ngram.reset()
        # model.py's `shared_attn` is a module-level object: what a source layer
        # publishes stays published ACROSS forwards, which is what a decode step reads
        # when its own compression group is still filling (latent is None there, and the
        # indexer still needs the key cache the source layer wrote two tokens ago).
        self.shared = {"compress_kv": None, "index_k": None, "topk_idxs": None, "candidates": None}
        self.main_hidden = None
        self.spec_window = [torch.zeros(c["window_size"], hd)
                            for _ in range(c.get("n_mtp_layers") or 0)]

    # -- attention -----------------------------------------------------------

    def compressor(self, layer: int, x: torch.Tensor, start_pos: int):
        """model.py Compressor.forward. Returns the pre-RoPE latents produced by this
        chunk, or None while a group is still filling."""
        c = self.c
        ratio = c["compress_ratios"][layer]
        p = f"layers.{layer}.attn.compressor."
        kv = x @ self.w[p + "wkv.weight"].T
        if ratio == 1:
            return rms_norm(kv, self.w[p + "norm.weight"], c["norm_eps"])
        score = x @ self.w[p + "wgate.weight"].T
        state_kv, state_score = self.comp_state[layer]
        if start_pos == 0:
            seqlen = x.shape[0]
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            if remainder:
                state_kv[:remainder] = kv[cutoff:]
                state_score[:remainder] = score[cutoff:]
                kv, score = kv[:cutoff], score[:cutoff]
            if cutoff == 0:
                return None
            kv = kv.reshape(-1, ratio, kv.shape[-1])
            score = score.reshape(-1, ratio, score.shape[-1])
            pooled = (kv * score.softmax(dim=1)).sum(dim=1)
        else:
            slot = start_pos % ratio
            state_kv[slot] = kv[0]
            state_score[slot] = score[0]
            if (start_pos + 1) % ratio != 0:
                return None
            pooled = (state_kv * state_score.softmax(dim=0)).sum(dim=0, keepdim=True)
        return rms_norm(pooled, self.w[p + "norm.weight"], c["norm_eps"])

    def indexer(self, layer: int, x, qr, latent, start_pos, offset, compress_len, shared):
        """model.py Indexer.forward, minus the fp4 activation quantization."""
        c = self.c
        p = f"layers.{layer}.attn.indexer."
        ratio = c["compress_ratios"][layer]
        seqlen = x.shape[0]
        end_pos = start_pos + seqlen
        if layer in c["kv_source_layers"] and latent is not None:
            freqs = (self.freqs(layer)[: seqlen - seqlen % ratio : ratio] if start_pos == 0
                     else self.freqs(layer)[start_pos + 1 - ratio].unsqueeze(0))
            k = rms_norm(latent @ self.w[p + "wk.weight"].T, self.w[p + "k_norm.weight"], c["norm_eps"])
            k = torch.cat([k[..., : -self.rd], apply_rotary(k[..., -self.rd :], freqs)], dim=-1)
            base = start_pos // ratio
            self.index_k[layer][base : base + k.shape[0]] = k
            shared["index_k"] = self.index_k[layer]
        nh, hd = c["index_n_heads"], c["index_head_dim"]
        q = (qr @ self.w[p + "wq_b.weight"].T).reshape(seqlen, nh, hd)
        q = torch.cat([q[..., : -self.rd], apply_rotary(q[..., -self.rd :], self.freqs(layer)[start_pos:end_pos])], dim=-1)
        index_k = shared["index_k"][: end_pos // ratio]
        weights = (x @ self.w[p + "weights_proj.weight"].T) * (hd**-0.5 * nh**-0.5)
        score = torch.einsum("shd,td->sht", q, index_k)
        score = (score.relu() * weights[..., None]).sum(dim=1)          # [s, t]
        if start_pos == 0:
            lens = (torch.arange(1, seqlen + 1) // ratio).unsqueeze(-1)
            score = score.masked_fill(torch.arange(score.shape[-1]) >= lens, -float("inf"))
        else:
            lens = torch.tensor(end_pos // ratio)
        if layer == c["candidate_source_layer"]:
            shared["candidates"] = select_candidate_blocks(
                score, lens, c["candidate_topk_blocks"], c["candidate_block_size"])
        elif 0 <= c["candidate_source_layer"] < layer:
            score = score.masked_fill(~shared["candidates"], -float("inf"))
        topk = min(c["index_topk"], end_pos // ratio)
        idxs = score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        return torch.where(idxs < lens, idxs + offset, torch.tensor(-1)).int()

    def attention(self, layer: int, x: torch.Tensor, start_pos: int, shared: dict) -> torch.Tensor:
        c = self.c
        p = f"layers.{layer}.attn."
        seqlen, hd, rd = x.shape[0], c["head_dim"], self.rd
        nh, win, ratio = c["n_heads"], c["window_size"], c["compress_ratios"][layer]
        freqs_cis = self.freqs(layer)[start_pos : start_pos + seqlen]

        qr = rms_norm(x @ self.w[p + "wq_a.weight"].T, self.w[p + "q_norm.weight"], c["norm_eps"])
        q = (qr @ self.w[p + "wq_b.weight"].T).reshape(seqlen, nh, hd)
        q = torch.cat([q[..., :-rd], apply_rotary(q[..., -rd:], freqs_cis)], dim=-1)

        kv = rms_norm(x @ self.w[p + "wkv.weight"].T, self.w[p + "kv_norm.weight"], c["norm_eps"])
        kv = torch.cat([kv[..., :-rd], apply_rotary(kv[..., -rd:], freqs_cis)], dim=-1)
        ring = self.window[layer]
        if start_pos == 0:
            if seqlen <= win:
                ring[:seqlen] = kv
            else:
                cut = seqlen % win
                ring[cut:win], ring[:cut] = kv[-win:][: win - cut], kv[-win:][win - cut :]
            window_kv, window_idxs = kv, window_topk_idxs(win, seqlen, 0)
        else:
            ring[start_pos % win] = kv[0]
            window_kv, window_idxs = ring, window_topk_idxs(win, seqlen, start_pos)

        kv_all, idxs = window_kv, window_idxs
        if ratio:
            compress_len = (start_pos + seqlen) // ratio
            latent = None
            if layer in c["kv_source_layers"]:
                latent = self.compressor(layer, x, start_pos)
                shared["compress_kv"] = self.compress_kv[layer]
            if layer in c["index_source_layers"]:
                shared["topk_idxs"] = (
                    torch.zeros(seqlen, 0, dtype=torch.int32) if compress_len == 0
                    else self.indexer(layer, x, qr, latent, start_pos, window_kv.shape[0], compress_len, shared))
            if latent is not None:
                freqs = (self.freqs(layer)[: seqlen - seqlen % ratio : ratio] if start_pos == 0
                         else self.freqs(layer)[start_pos + 1 - ratio].unsqueeze(0))
                latent = torch.cat([latent[..., :-rd], apply_rotary(latent[..., -rd:], freqs)], dim=-1)
                base = start_pos // ratio
                self.compress_kv[layer][base : base + latent.shape[0]] = latent
            kv_all = torch.cat([window_kv, shared["compress_kv"][:compress_len]], dim=0)
            idxs = torch.cat([window_idxs, shared["topk_idxs"]], dim=-1)

        o = sparse_attn(q, kv_all, self.w[p + "attn_sink"], idxs, hd**-0.5)
        o = torch.cat([o[..., :-rd], apply_rotary(o[..., -rd:], freqs_cis, inverse=True)], dim=-1)
        groups, olora = c["o_groups"], c["o_lora_rank"]
        o = o.reshape(seqlen, groups, -1)
        wo_a = self.w[p + "wo_a.weight"].reshape(groups, olora, -1)
        o = torch.einsum("sgd,grd->sgr", o, wo_a).reshape(seqlen, groups * olora)
        return o @ self.w[p + "wo_b.weight"].T

    # -- MoE -----------------------------------------------------------------

    def expert(self, prefix: str, x: torch.Tensor) -> torch.Tensor:
        limit = self.c["swiglu_limit"]
        gate = x @ self.w[prefix + "w1.weight"].T
        up = x @ self.w[prefix + "w3.weight"].T
        if limit > 0:
            up = up.clamp(-limit, limit)
            gate = gate.clamp(max=limit)
        return (F.silu(gate) * up) @ self.w[prefix + "w2.weight"].T

    def moe(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        return self.moe_at(f"layers.{layer}.ffn.", x, self.c["n_activated_experts"])

    def moe_at(self, p: str, x: torch.Tensor, n_activated: int) -> torch.Tensor:
        """model.py MoE.forward. `p` is the ffn prefix, so a DSpark stage -- which
        routes over its own, smaller expert set -- runs the same code."""
        c = self.c
        scores = F.softplus(x @ self.w[p + "gate.weight"].T).sqrt()
        idx = (scores + self.w[p + "gate.bias"]).topk(n_activated, dim=-1).indices
        weights = scores.gather(1, idx)
        if c["norm_topk_prob"] and n_activated > 1:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * c["route_scale"]
        y = torch.zeros_like(x)
        for token in range(x.shape[0]):
            for k in range(idx.shape[1]):
                e = int(idx[token, k])
                y[token] += weights[token, k] * self.expert(p + f"experts.{e}.", x[token : token + 1])[0]
        return y + self.expert(p + "shared_experts.", x)

    # -- block / forward -----------------------------------------------------

    def hc_mixes(self, x, which: str, layer: int):
        return self.hc_mixes_at(x, which, f"layers.{layer}.")

    def hc_mixes_at(self, x, which: str, p: str):
        c = self.c
        flat = x.reshape(x.shape[0], -1)
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + c["norm_eps"])
        mixes = (flat @ self.w[p + f"hc_{which}_fn"].T) * rsqrt
        return hc_split_sinkhorn(mixes, self.w[p + f"hc_{which}_scale"],
                                 self.w[p + f"hc_{which}_base"],
                                 self.hc, c["hc_sinkhorn_iters"], c["hc_eps"])

    def engram(self, layer: int, x: torch.Tensor, hash_ids: torch.Tensor) -> torch.Tensor:
        """model.py Engram.forward. `x` is [s, hc, dim]; hash_ids [s, n_hash_cols]."""
        c = self.c
        p = f"layers.{layer}.engram."
        rows = self.w[p + "embed.weight"][hash_ids.reshape(-1)].reshape(hash_ids.shape[0], -1)
        kv = rows @ self.w[p + "wkv.weight"].T
        key, value = kv.split([self.hc * self.dim, self.dim], dim=-1)
        key = key.reshape(-1, self.hc, self.dim)
        weight = self.w[p + "q_weight"] * self.w[p + "k_weight"]
        eps = c["norm_eps"]
        rstd = torch.rsqrt(x.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
        dot = (x * weight * key).sum(-1) * rstd * self.dim**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        return x + gate[..., None] * value[:, None, :]

    def forward(self, ids: list[int], compressed: list[int] | None = None) -> torch.Tensor:
        """Run one chunk (prefill when the model is fresh, one token per call after) and
        return the logits of its last position."""
        c = self.c
        start_pos = self.pos
        hashes = None
        if self.ngram is not None:
            hashes = self.ngram.push(compressed if compressed is not None else ids)
        h = self.w["embed.weight"][torch.tensor(ids)]
        h = h[:, None, :].repeat(1, self.hc, 1)
        pre_mix = torch.zeros(h.shape[0], self.hc)
        pre_mix[:, 0] = 1.0
        shared = self.shared
        targets, mains = c.get("dspark_target_layer_ids") or [], []
        for layer in range(self.nlayers):
            if layer in c["engram_layer_ids"]:
                which = c["engram_layer_ids"].index(layer)
                h = self.engram(layer, h, hashes[:, which, :])
            # model.py: "the MTP head reads the attention input of its target layers,
            # not their output" -- the hc copies averaged, before the block runs
            if layer in targets:
                mains.append(h.mean(dim=1))
            residual = h
            attn_pre, attn_post, attn_comb = self.hc_mixes(h, "attn", layer)
            y = (pre_mix[..., None] * h).sum(dim=1)
            y = rms_norm(y, self.w[f"layers.{layer}.attn_norm.weight"], c["norm_eps"])
            y = self.attention(layer, y, start_pos, shared)
            h = attn_post[..., None] * y[:, None, :] + hc_combine(attn_comb, residual)

            residual = h
            ffn_pre, ffn_post, ffn_comb = self.hc_mixes(h, "ffn", layer)
            y = (attn_pre[..., None] * h).sum(dim=1)
            y = rms_norm(y, self.w[f"layers.{layer}.ffn_norm.weight"], c["norm_eps"])
            y = self.moe(layer, y)
            h = ffn_post[..., None] * y[:, None, :] + hc_combine(ffn_comb, residual)
            pre_mix = ffn_pre
        h = (pre_mix[..., None] * h).sum(dim=1)
        h = rms_norm(h, self.w["norm.weight"], c["norm_eps"])
        # what forward_spec needs from this forward, kept beside the logits rather than
        # returned, so every existing caller keeps its one-value signature
        self.main_hidden = torch.cat(mains, dim=-1) if mains else None
        self.pos += len(ids)
        return (h[-1:] @ self.w["head.weight"].T)[0]


    # -- DSpark (the MTP draft head) -----------------------------------------

    def spec_attention(self, stage: int, x: torch.Tensor, start_pos: int,
                       main_x: torch.Tensor) -> torch.Tensor:
        """model.py DSparkAttention.forward.

        A DSpark stage never compresses (`assert self.compress_ratio == 0`): it is
        window-only, and the window holds the MAIN stream's keys. The drafts add their
        own keys on top for the length of one block, and every draft row reads every
        other -- `get_dspark_topk_idxs` hands the same index list to all of them, so
        inside the block attention is not causal. That is the vendor's, not a
        simplification: the block is one guess, not a sequence the model committed to.
        """
        c, w = self.c, self.w
        p = f"mtp.{stage}.attn."
        hd, rd, nh, win = c["head_dim"], self.rd, c["n_heads"], c["window_size"]
        eps = c["norm_eps"]
        freqs = self.freqs_window                       # compress_ratio 0: window rope
        seqlen = main_x.shape[0]
        main_kv = rms_norm(main_x @ w[p + "wkv.weight"].T, w[p + "kv_norm.weight"], eps)
        main_kv = torch.cat(
            [main_kv[..., :-rd],
             apply_rotary(main_kv[..., -rd:], freqs[start_pos : start_pos + seqlen])], dim=-1)
        ring = self.spec_window[stage]
        if start_pos == 0:
            # prefill seeds the ring and nothing else: the block never runs
            if seqlen <= win:
                ring[:seqlen] = main_kv
            else:
                cut = seqlen % win
                ring[cut:win], ring[:cut] = main_kv[-win:][: win - cut], main_kv[-win:][win - cut :]
            return x

        block = x.shape[0]
        # the drafts sit at the positions AFTER the one the main model just produced
        bfreqs = freqs[start_pos + seqlen : start_pos + seqlen + block]
        qr = rms_norm(x @ w[p + "wq_a.weight"].T, w[p + "q_norm.weight"], eps)
        q = (qr @ w[p + "wq_b.weight"].T).reshape(block, nh, hd)
        q = torch.cat([q[..., :-rd], apply_rotary(q[..., -rd:], bfreqs)], dim=-1)
        kv = rms_norm(x @ w[p + "wkv.weight"].T, w[p + "kv_norm.weight"], eps)
        kv = torch.cat([kv[..., :-rd], apply_rotary(kv[..., -rd:], bfreqs)], dim=-1)

        ring[start_pos % win] = main_kv[0]
        kv_all = torch.cat([ring, kv], dim=0)
        reach = min(win, start_pos + 1)
        idxs = torch.cat([torch.arange(reach), win + torch.arange(block)])
        idxs = idxs.int().unsqueeze(0).expand(block, -1)
        o = sparse_attn(q, kv_all, w[p + "attn_sink"], idxs, hd**-0.5)
        o = torch.cat([o[..., :-rd], apply_rotary(o[..., -rd:], bfreqs, inverse=True)], dim=-1)
        groups, olora = c["o_groups"], c["o_lora_rank"]
        o = o.reshape(block, groups, -1)
        wo_a = w[p + "wo_a.weight"].reshape(groups, olora, -1)
        o = torch.einsum("sgd,grd->sgr", o, wo_a).reshape(block, groups * olora)
        return o @ w[p + "wo_b.weight"].T

    def spec_forward(self, token_id: int, main_hidden: torch.Tensor, start_pos: int):
        """model.py Transformer.forward_spec.

        `token_id` is what the main model just produced and `main_hidden` the hidden
        states its target layers handed over. Returns (draft_ids, confidence): the
        token itself followed by `dspark_block_size` guesses, and one confidence score
        per guess. During prefill (start_pos 0) it returns None -- that call exists
        only to seed the stages' windows from the whole prompt.
        """
        c, w = self.c, self.w
        stages, block = c["n_mtp_layers"], c["dspark_block_size"]
        hc, eps = self.hc, c["norm_eps"]
        main_x = rms_norm(main_hidden @ w["mtp.0.main_proj.weight"].T,
                          w["mtp.0.main_norm.weight"], eps)
        # every draft position but the first carries the noise id: the block is
        # generated in one shot, so there is nothing to put there
        ids = [token_id] + [c["dspark_noise_token_id"]] * (block - 1)
        x = w["embed.weight"][torch.tensor(ids)][:, None, :].repeat(1, hc, 1)
        pre_mix = torch.zeros(block, hc)
        pre_mix[:, 0] = 1.0
        for stage in range(stages):
            p = f"mtp.{stage}."
            if start_pos == 0:
                self.spec_attention(stage, x, 0, main_x)
                continue
            residual = x
            attn_pre, attn_post, attn_comb = self.hc_mixes_at(x, "attn", p)
            y = (pre_mix[..., None] * x).sum(dim=1)
            y = rms_norm(y, w[p + "attn_norm.weight"], eps)
            y = self.spec_attention(stage, y, start_pos, main_x)
            x = attn_post[..., None] * y[:, None, :] + hc_combine(attn_comb, residual)

            residual = x
            ffn_pre, ffn_post, ffn_comb = self.hc_mixes_at(x, "ffn", p)
            y = (attn_pre[..., None] * x).sum(dim=1)
            y = rms_norm(y, w[p + "ffn_norm.weight"], eps)
            y = self.moe_at(p + "ffn.", y, c["dspark_n_activated_experts"])
            x = ffn_post[..., None] * y[:, None, :] + hc_combine(ffn_comb, residual)
            pre_mix = ffn_pre
        if start_pos == 0:
            return None

        last = f"mtp.{stages - 1}."
        x = (pre_mix[..., None] * x).sum(dim=1)                    # [block, dim]
        logits = rms_norm(x, w[last + "norm.weight"], eps) @ w["head.weight"].T
        draft = [token_id]
        embeds = []
        for i in range(block):
            row = w[last + "markov_head.embed.weight"][draft[i]]
            logits[i] = logits[i] + row @ w[last + "markov_head.head.weight"].T
            embeds.append(row)
            draft.append(int(logits[i].argmax()))       # greedy: temperature 0
        markov = torch.stack(embeds, dim=0)
        confidence = torch.cat([x, markov], dim=-1) @ w[last + "confidence_head.proj.weight"].T
        return draft, confidence.reshape(-1)


def hc_combine(comb: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """out[j] = sum_i comb[i, j] * residual[i], the mix model.py Block.hc_post applies.

    The summed axis is comb's FIRST index. Written out rather than left to broadcasting:
    with the batch axis dropped, `comb[..., None] * residual[:, None]` lines comb's
    SECOND index up with the residual copies and sums the transpose -- which is what
    this reference did until the C engine disagreed with it, and the C engine was right.
    """
    return torch.einsum("nij,nid->njd", comb, residual)


def window_topk_idxs(window_size: int, seqlen: int, start_pos: int) -> torch.Tensor:
    """model.py get_window_topk_idxs, one batch."""
    if start_pos == 0:
        end = torch.arange(seqlen).unsqueeze(1)
        idxs = (end - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        return torch.where(idxs > end, torch.tensor(-1), idxs).int()
    oldest = start_pos % window_size + 1
    idxs = torch.cat([torch.arange(oldest, window_size), torch.arange(oldest)])
    return torch.where(idxs > start_pos, torch.tensor(-1), idxs).int().unsqueeze(0)


def select_candidate_blocks(logits, compress_lens, topk_blocks, block_size):
    """model.py select_candidate_blocks, one batch."""
    width = logits.shape[-1]
    pad = (-width) % block_size
    scores = F.pad(logits, (0, pad), value=-float("inf"))
    scores = scores.reshape(*scores.shape[:-1], -1, block_size).amax(dim=-1)
    num_blocks = scores.shape[-1]
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks) == last, float("inf"))
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -float("inf"))
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


# -------------------------------------------------------------------- vision


def vision_cos_sin(n_h: int, n_w: int, dim: int, theta: float):
    """vision.py get_vision_cos_sin: 2D RoPE over the patch grid, one row per patch."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return freqs.cos(), freqs.sin()


def vision_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """vision.py apply_rotary: the SPLIT-HALF convention, not the interleaved pairs the
    language model uses. x is [n, heads, head_dim] and cos/sin are [n, head_dim/2]."""
    x1, x2 = x.float().chunk(2, dim=-1)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class RefVision:
    """vision.py ViT + Aligner: a full-attention tower over one image, then a 3x3
    downsample into the language model's width."""

    def __init__(self, cfg: dict, w: dict):
        self.c = cfg
        self.w = w

    def forward(self, patches: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        c = self.c
        dim, heads = c["vision_dim"], c["vision_n_heads"]
        head_dim = dim // heads
        w = self.w
        x = patches.reshape(patches.shape[0], -1) @ w["vision.patch_embed.proj.weight"].T \
            + w["vision.patch_embed.proj.bias"]
        cos, sin = vision_cos_sin(n_h, n_w, head_dim // 2, c["vision_rope_theta"])
        for layer in range(c["vision_n_layers"]):
            p = f"vision.blocks.{layer}."
            h = rms_norm(x, w[p + "norm1.weight"], 1e-6)
            qkv = h @ w[p + "attn.wqkv.weight"].T + w[p + "attn.wqkv.bias"]
            q, k, v = qkv.chunk(3, dim=-1)
            n = x.shape[0]
            q = vision_rotary(q.reshape(n, heads, head_dim), cos, sin)
            k = vision_rotary(k.reshape(n, heads, head_dim), cos, sin)
            v = v.reshape(n, heads, head_dim).float()
            scores = torch.einsum("qhd,khd->hqk", q, k) / head_dim**0.5
            attended = torch.einsum("hqk,khd->qhd", scores.softmax(dim=-1), v)
            x = x + (attended.reshape(n, dim) @ w[p + "attn.wo.weight"].T + w[p + "attn.wo.bias"])
            h = rms_norm(x, w[p + "norm2.weight"], 1e-6)
            gate, up = (h @ w[p + "mlp.w1.weight"].T).chunk(2, dim=-1)
            x = x + (F.silu(gate) * up) @ w[p + "mlp.w2.weight"].T
        x = rms_norm(x, w["vision.norm.weight"], 1e-6)
        return self.align(x, n_h, n_w)

    def align(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        """vision.py Aligner.forward: pad the grid to a multiple of the ratio, take
        ratio x ratio blocks, and project. F.unfold orders a block channel-major, which
        is the detail the C side has to get right."""
        c = self.c
        ratio = c["vision_downsample_ratio"]
        grid = x.reshape(n_h, n_w, -1).permute(2, 0, 1)
        grid = F.pad(grid, (0, -n_w % ratio, 0, -n_h % ratio))
        blocks = F.unfold(grid.unsqueeze(0), ratio, stride=ratio).squeeze(0).transpose(0, 1)
        hidden = blocks @ self.w["aligner.w1.weight"].T + self.w["aligner.w1.bias"]
        return F.gelu(hidden) @ self.w["aligner.w2.weight"].T + self.w["aligner.w2.bias"]
