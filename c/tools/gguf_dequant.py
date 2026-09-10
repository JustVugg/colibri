#!/usr/bin/env python3
"""Dequantization of GGUF/ggml tensor blocks to float32, pure numpy.

Ported block-for-block from ggml-org/ggml src/ggml-quants.c (dequantize_row_*),
using the block layouts in src/ggml-common.h. This module owns the numerical
decode used by the converter profiles; gguf_reader.py owns structure only.

Only the types listed in _DEQUANTIZERS are implemented. dequantize() raises
GGUFError for any other type, so a profile never silently mis-decodes bytes.
Correctness is pinned by tests/test_gguf_dequant.py, which compares against
llama.cpp's gguf-py reference on random blocks and against committed vectors.
"""

import numpy as np

from gguf_reader import (
    GGML_TYPE_BF16,
    GGML_TYPE_F16,
    GGML_TYPE_F32,
    GGML_TYPE_Q2_K,
    GGML_TYPE_Q3_K,
    GGML_TYPE_Q6_K,
    GGUFError,
    ggml_block_spec,
    ggml_type_name,
)

QK_K = 256
_MASK_KMASK1 = 0x03030303
_MASK_KMASK2 = 0x0F0F0F0F


def _f16_to_f32(pair_bytes):
    """(n, 2) uint8 or (n,) uint16 -> float32."""
    u16 = pair_bytes.view("<u2").reshape(-1)
    return u16.view(np.float16).astype(np.float32)


def _bf16_to_f32(word_bytes):
    u16 = word_bytes.view("<u2").reshape(-1).astype(np.uint32)
    return (u16 << 16).view(np.float32)


def dequantize_q2_K(blocks):
    # Q2_K (ggml-common.h block_q2_K, 84 bytes/256 elems):
    #   scales[16] (per-16 low nibble=scale, high nibble=min, 4-bit),
    #   qs[64] (2-bit packed quants), then fp16 block scale (d) and min (dmin).
    num_blocks = blocks.shape[0]
    scales = blocks[:, 0:16].astype(np.int32)
    packed_q = blocks[:, 16:80]
    block_scale = _f16_to_f32(blocks[:, 80:82])
    block_min = _f16_to_f32(blocks[:, 82:84])
    out = np.empty((num_blocks, QK_K), dtype=np.float32)
    is_ = 0
    for half in range(2):
        base = half * 32
        for j in range(4):
            shift = j * 2
            for sub in range(2):
                sc = scales[:, is_]
                is_ += 1
                group_scale = block_scale * (sc & 0xF)
                group_min = block_min * (sc >> 4)
                cols = base + sub * 16 + np.arange(16)
                qv = ((packed_q[:, cols] >> shift) & 3).astype(np.float32)
                lo = half * 128 + j * 32 + sub * 16
                out[:, lo:lo + 16] = group_scale[:, None] * qv - group_min[:, None]
    return out


def dequantize_q3_K(blocks):
    # Q3_K (ggml-common.h block_q3_K, 110 bytes/256 elems):
    #   hmask[32] high bits, qs[64] low 2 bits, scales[12] (sixteen 6-bit scales
    #   packed into 12 bytes, reassembled by the kmask shuffle below), fp16 scale.
    num_blocks = blocks.shape[0]
    hmask = blocks[:, 0:32]
    packed_q = blocks[:, 32:96]
    scale_bytes = blocks[:, 96:108]
    block_scale = _f16_to_f32(blocks[:, 108:110])

    # Sixteen 6-bit scales live packed in 12 bytes: three 32-bit words, then a
    # bit-shuffle reassembles them (kmask1/kmask2 from ggml-quants.c).
    a0 = (scale_bytes[:, 0].astype(np.uint32) | (scale_bytes[:, 1].astype(np.uint32) << 8)
          | (scale_bytes[:, 2].astype(np.uint32) << 16) | (scale_bytes[:, 3].astype(np.uint32) << 24))
    a1 = (scale_bytes[:, 4].astype(np.uint32) | (scale_bytes[:, 5].astype(np.uint32) << 8)
          | (scale_bytes[:, 6].astype(np.uint32) << 16) | (scale_bytes[:, 7].astype(np.uint32) << 24))
    a2 = (scale_bytes[:, 8].astype(np.uint32) | (scale_bytes[:, 9].astype(np.uint32) << 8)
          | (scale_bytes[:, 10].astype(np.uint32) << 16) | (scale_bytes[:, 11].astype(np.uint32) << 24))
    tmp = a2
    a2n = ((a0 >> 4) & _MASK_KMASK2) | (((tmp >> 4) & _MASK_KMASK1) << 4)
    a3n = ((a1 >> 4) & _MASK_KMASK2) | (((tmp >> 6) & _MASK_KMASK1) << 4)
    a0n = (a0 & _MASK_KMASK2) | (((tmp >> 0) & _MASK_KMASK1) << 4)
    a1n = (a1 & _MASK_KMASK2) | (((tmp >> 2) & _MASK_KMASK1) << 4)

    row_scales = np.empty((num_blocks, 16), dtype=np.int8)
    for i, word in enumerate((a0n, a1n, a2n, a3n)):
        row_scales[:, i * 4:(i + 1) * 4] = word.astype("<u4").view(np.uint8).reshape(num_blocks, 4)

    out = np.empty((num_blocks, QK_K), dtype=np.float32)
    is_ = 0
    for half in range(2):
        base = half * 32
        for j in range(4):
            shift = j * 2
            m = 1 << (half * 4 + j)
            for sub in range(2):
                group_scale = block_scale * (row_scales[:, is_] - 32).astype(np.float32)
                is_ += 1
                qcols = base + sub * 16 + np.arange(16)
                hcols = sub * 16 + np.arange(16)
                qv = ((packed_q[:, qcols] >> shift) & 3).astype(np.float32)
                high_bit = ((hmask[:, hcols] & m) != 0).astype(np.float32)
                # low 2-bit code, minus 4 when the high bit of the mask is clear
                centered_q = qv - 4.0 * (1.0 - high_bit)
                lo = half * 128 + j * 32 + sub * 16
                out[:, lo:lo + 16] = group_scale[:, None] * centered_q
    return out


def dequantize_q6_K(blocks):
    # Q6_K (ggml-common.h block_q6_K, 210 bytes/256 elems):
    #   ql[128] low 4 bits, qh[64] upper 2 bits, scales[16] int8 per-16 scales,
    #   fp16 block scale. Two 128-elem halves; codes are (low | high<<4) - 32.
    num_blocks = blocks.shape[0]
    packed_lo = blocks[:, 0:128]
    packed_hi = blocks[:, 128:192]
    row_scales = blocks[:, 192:208].view(np.int8)
    block_scale = _f16_to_f32(blocks[:, 208:210])
    out = np.empty((num_blocks, QK_K), dtype=np.float32)
    for half in range(2):
        lo_base = half * 64
        hi_base = half * 32
        scale_base = half * 8
        l = np.arange(32)
        group = l // 16
        code0 = ((packed_lo[:, lo_base + l] & 0xF).astype(np.int32)
                 | (((packed_hi[:, hi_base + l] >> 0) & 3).astype(np.int32) << 4)) - 32
        code1 = ((packed_lo[:, lo_base + l + 32] & 0xF).astype(np.int32)
                 | (((packed_hi[:, hi_base + l] >> 2) & 3).astype(np.int32) << 4)) - 32
        code2 = ((packed_lo[:, lo_base + l] >> 4).astype(np.int32)
                 | (((packed_hi[:, hi_base + l] >> 4) & 3).astype(np.int32) << 4)) - 32
        code3 = ((packed_lo[:, lo_base + l + 32] >> 4).astype(np.int32)
                 | (((packed_hi[:, hi_base + l] >> 6) & 3).astype(np.int32) << 4)) - 32
        scale0 = row_scales[:, scale_base + group + 0].astype(np.float32)
        scale1 = row_scales[:, scale_base + group + 2].astype(np.float32)
        scale2 = row_scales[:, scale_base + group + 4].astype(np.float32)
        scale3 = row_scales[:, scale_base + group + 6].astype(np.float32)
        base = half * 128
        out[:, base + l] = block_scale[:, None] * scale0 * code0.astype(np.float32)
        out[:, base + l + 32] = block_scale[:, None] * scale1 * code1.astype(np.float32)
        out[:, base + l + 64] = block_scale[:, None] * scale2 * code2.astype(np.float32)
        out[:, base + l + 96] = block_scale[:, None] * scale3 * code3.astype(np.float32)
    return out


_DEQUANTIZERS = {
    GGML_TYPE_Q2_K: dequantize_q2_K,
    GGML_TYPE_Q3_K: dequantize_q3_K,
    GGML_TYPE_Q6_K: dequantize_q6_K,
}


def dequantize(raw, ggml_type, numel):
    """Decode a tensor's stored bytes to a flat float32 array (GGUF element order)."""
    if ggml_type in _DEQUANTIZERS:
        block_elems, block_bytes = ggml_block_spec(ggml_type)
        if numel % block_elems != 0:
            raise GGUFError(
                "dequant %s: numel %d is not a multiple of block %d"
                % (ggml_type_name(ggml_type), numel, block_elems))
        nblocks = numel // block_elems
        data = np.frombuffer(raw, dtype=np.uint8)
        if data.size != nblocks * block_bytes:
            raise GGUFError(
                "dequant %s: %d bytes for %d blocks of %d, expected %d"
                % (ggml_type_name(ggml_type), data.size, nblocks, block_bytes,
                   nblocks * block_bytes))
        blocks = data.reshape(nblocks, block_bytes)
        return _DEQUANTIZERS[ggml_type](blocks).reshape(-1)

    element_size = {GGML_TYPE_F32: 4, GGML_TYPE_F16: 2, GGML_TYPE_BF16: 2}.get(ggml_type)
    if element_size is not None:
        expected = numel * element_size
        if len(raw) != expected:
            raise GGUFError(
                "dequant %s: %d bytes for %d elements, expected %d"
                % (ggml_type_name(ggml_type), len(raw), numel, expected))
        if ggml_type == GGML_TYPE_F32:
            return np.frombuffer(raw, dtype="<f4").astype(np.float32)
        if ggml_type == GGML_TYPE_F16:
            return np.frombuffer(raw, dtype="<f2").astype(np.float32)
        return _bf16_to_f32(np.frombuffer(raw, dtype=np.uint8))

    raise GGUFError("dequant: unsupported ggml type %d (%s)"
                    % (ggml_type, ggml_type_name(ggml_type)))
