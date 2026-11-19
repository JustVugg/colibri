#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Repack native MXFP4 experts into aligned WEX1, WEX2 or WEX3 containers.

WEX2 packs UE8M0 scale tiles without changing FP4 weights. Its header padding
contains a random build ID at bytes 40..55 (identity metadata, not a checksum).
Primary and mirror publication use temporary siblings and atomic replacement.
WEX3 adds exact deduplication and explicitly opt-in lossy palette3 weights.
"""

import argparse
import hashlib
import importlib.util
from contextlib import ExitStack
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import struct
import tempfile
import uuid

WEX_MAGIC = WEX_MAGIC_V1 = 0x57455831
WEX_MAGIC_V2 = 0x57455832
WEX_MAGIC_V3 = 0x57455833
WEX_ALIGN = 4096
WEX_SCALE_TILES = 1
WEX_PALETTE3 = 2
WEX_DEDUP = 4
WEX_NEURAL_RANS = 3
WEX_HEADER_NEURAL_RANS = 8
WEX_SCALE_TILE_MAGIC = 0x31534357
WEX_TILE_SIZE = 256
HEADER = struct.Struct("<IIIIQQII")
ENTRY = struct.Struct("<QIII6H6H3I3II")
SCALE_PREFIX = struct.Struct("<III")
UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1
_DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E4M3FN": 1,
    "F8_E5M2": 1, "F8_E8M0": 1, "F8_E8M0FNU": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}
_PALETTE_SPEC = importlib.util.spec_from_file_location("kaira_wex_palette", Path(__file__).with_name("wex_palette.py"))
palette3 = importlib.util.module_from_spec(_PALETTE_SPEC)
_PALETTE_SPEC.loader.exec_module(palette3)
_NEURAL_SPEC = importlib.util.spec_from_file_location("kaira_wex_neural_rans", Path(__file__).with_name("wex_neural_rans.py"))
neural_rans = importlib.util.module_from_spec(_NEURAL_SPEC)
_NEURAL_SPEC.loader.exec_module(neural_rans)


def align_up(n, align=WEX_ALIGN):
    return (n + align - 1) // align * align


def pad_to_align(n, align=WEX_ALIGN):
    return align_up(n, align) - n


def _integer(value, name, minimum=0, maximum=UINT64_MAX):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def pack_scale_tiles(scales):
    """Encode tiles as base, bit width, then LSB-first unsigned differences."""
    encoded = bytearray()
    for start in range(0, len(scales), WEX_TILE_SIZE):
        tile = scales[start:start + WEX_TILE_SIZE]
        base = min(tile)
        bits = (max(tile) - base).bit_length()
        encoded.extend((base, bits))
        if bits:
            accumulator = available = 0
            for value in tile:
                accumulator |= (value - base) << available
                available += bits
                while available >= 8:
                    encoded.append(accumulator & 255)
                    accumulator >>= 8
                    available -= 8
            if available:
                encoded.append(accumulator)
    return bytes(encoded)


def unpack_scale_tiles(encoded, decoded_bytes):
    """Strict reference decoder, also used by format verification/tests."""
    _integer(decoded_bytes, "decoded scale bytes", 0, UINT32_MAX)
    decoded = bytearray()
    cursor = 0
    while len(decoded) < decoded_bytes:
        count = min(WEX_TILE_SIZE, decoded_bytes - len(decoded))
        if cursor + 2 > len(encoded):
            raise ValueError("truncated scale tile header")
        base, bits = encoded[cursor:cursor + 2]
        cursor += 2
        if bits > 8:
            raise ValueError("invalid scale tile bit width")
        payload_bytes = (count * bits + 7) // 8
        if cursor + payload_bytes > len(encoded):
            raise ValueError("truncated scale tile payload")
        if bits == 0:
            decoded.extend(bytes((base,)) * count)
        else:
            accumulator = available = 0
            payload_cursor = cursor
            mask = (1 << bits) - 1
            for _ in range(count):
                while available < bits:
                    accumulator |= encoded[payload_cursor] << available
                    available += 8
                    payload_cursor += 1
                value = base + (accumulator & mask)
                if value > 255:
                    raise ValueError("scale tile value overflows U8")
                decoded.append(value)
                accumulator >>= bits
                available -= bits
            if accumulator:
                raise ValueError("nonzero scale tile padding bits")
        cursor += payload_bytes
    if cursor != len(encoded):
        raise ValueError("trailing bytes in scale tile stream")
    return bytes(decoded)


@dataclass(frozen=True)
class TensorRef:
    path: Path
    offset: int
    length: int
    shape: tuple
    dtype: str


def index_safetensors(model_dir):
    tensor_map = {}
    shards = sorted(Path(model_dir).glob("*.safetensors"))
    if not shards:
        raise ValueError(f"no .safetensors shards in {model_dir}")
    for path in shards:
        file_bytes = path.stat().st_size
        with path.open("rb") as source:
            prefix = source.read(8)
            if len(prefix) != 8:
                raise ValueError(f"truncated safetensors header: {path}")
            header_bytes = struct.unpack("<Q", prefix)[0]
            if not 2 <= header_bytes <= min(file_bytes - 8, 100_000_000):
                raise ValueError(f"invalid safetensors header length: {path}")
            header_raw = source.read(header_bytes)
            if len(header_raw) != header_bytes:
                raise ValueError(f"short safetensors header read: {path}")
        header = json.loads(header_raw.decode("utf-8"), object_pairs_hook=_unique_object)
        if not isinstance(header, dict):
            raise ValueError(f"safetensors header must be an object: {path}")
        data_start = 8 + header_bytes
        ranges = []
        for key, info in header.items():
            if key == "__metadata__":
                continue
            if key in tensor_map:
                raise ValueError(f"duplicate tensor across shards: {key}")
            if not isinstance(info, dict):
                raise ValueError(f"invalid tensor header: {key}")
            shape, offsets, dtype = info.get("shape"), info.get("data_offsets"), info.get("dtype")
            if not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
                raise ValueError(f"invalid tensor shape/offsets: {key}")
            for dimension in shape:
                _integer(dimension, f"shape of {key}")
            start = _integer(offsets[0], f"start of {key}")
            end = _integer(offsets[1], f"end of {key}", start)
            if end > file_bytes - data_start:
                raise ValueError(f"tensor extends beyond shard: {key}")
            if not isinstance(dtype, str) or dtype not in _DTYPE_BYTES:
                raise ValueError(f"unsupported safetensors dtype for {key}: {dtype}")
            if math.prod(shape) * _DTYPE_BYTES[dtype] != end - start:
                raise ValueError(f"tensor shape/byte length mismatch: {key}")
            tensor_map[key] = TensorRef(path, data_start + start, end - start, tuple(shape), dtype)
            if end > start:
                ranges.append((start, end, key))
        ranges.sort()
        for previous, current in zip(ranges, ranges[1:]):
            if current[0] < previous[1]:
                raise ValueError(f"overlapping safetensors tensors: {previous[2]}, {current[2]}")
    return tensor_map


def _model_dims(model_dir, config):
    """Routed-expert geometry: V4 top level, V4.1 text_config, then vendor inference/config.json."""
    text = config.get("text_config", config)
    if not isinstance(text, dict):
        raise ValueError("text_config must be an object")
    keys = (("num_hidden_layers", "n_layers"), ("n_routed_experts", "n_routed_experts"),
            ("hidden_size", "dim"), ("moe_intermediate_size", "moe_inter_dim"))
    vendor = {}
    path = model_dir / "inference" / "config.json"
    if any(text.get(key) is None for key, _ in keys) and path.is_file():
        with path.open("r", encoding="utf-8") as source:
            vendor = json.load(source, object_pairs_hook=_unique_object)
        if not isinstance(vendor, dict):
            raise ValueError("inference/config.json must be an object")
    values = [text.get(key, vendor.get(alias)) for key, alias in keys]
    layers = _integer(43 if values[0] is None else values[0], "num_hidden_layers", 1, UINT32_MAX)
    experts = _integer(256 if values[1] is None else values[1], "n_routed_experts", 1, UINT32_MAX)
    dims = None if None in values[2:] else (_integer(values[2], "hidden_size", 1),
                                             _integer(values[3], "moe_intermediate_size", 1))
    return layers, experts, dims


def _expert_plan(tensors, layers, experts, dims=None):
    # Only layers.N experts are routed-store records; mtp.N.ffn.experts stay out of WEX.
    expected = None if dims is None else {"w1": (dims[1], dims[0] // 2), "w2": (dims[0], dims[1] // 2),
                                          "w3": (dims[1], dims[0] // 2)}
    plan = []
    for layer in range(layers):
        prefix = f"layers.{layer}.ffn.experts."
        if not any(key.startswith(prefix) for key in tensors):
            plan.extend([None] * experts)
            continue
        for expert in range(experts):
            matrices = []
            for matrix in ("w1", "w2", "w3"):
                name = f"{prefix}{expert}.{matrix}"
                try:
                    scale, weight = tensors[name + ".scale"], tensors[name + ".weight"]
                except KeyError as error:
                    raise ValueError(f"missing native expert tensor: {error.args[0]}") from error
                if scale.dtype not in ("F8_E8M0", "F8_E8M0FNU") or weight.dtype not in ("I8", "U8"):
                    raise ValueError(f"invalid native MXFP4 dtype: {name}")
                for ref in (scale, weight):
                    if len(ref.shape) != 2 or any(not 1 <= size <= 65535 for size in ref.shape):
                        raise ValueError(f"native expert shape must be positive 2D uint16: {name}")
                if scale.shape[0] != weight.shape[0] or scale.shape[1] * 32 != weight.shape[1] * 2:
                    raise ValueError(f"native expert scale/weight shapes disagree: {name}")
                if expected is not None and weight.shape != expected[matrix]:
                    raise ValueError(f"native expert shape disagrees with config dims: {name}")
                matrices.append((scale, weight))
            raw_bytes = sum(ref.length for pair in matrices for ref in pair)
            if align_up(raw_bytes) > UINT32_MAX:
                raise ValueError(f"expert record exceeds WEX uint32 size: layer={layer} expert={expert}")
            plan.append(matrices)
    if not any(plan):
        raise ValueError("no routed native MXFP4 experts found")
    return plan


def _read_tensor(ref, handles):
    source = handles[ref.path]
    source.seek(ref.offset)
    data = source.read(ref.length)
    if len(data) != ref.length:
        raise ValueError(f"short tensor read in {ref.path} at {ref.offset}")
    return data


def _scale_storage(scales, weight_bytes, format_version):
    raw_record_bytes = align_up(len(scales) + weight_bytes)
    if format_version >= 2 and len(scales) % WEX_ALIGN == 0:
        stream = pack_scale_tiles(scales)
        disk_scale_bytes = align_up(SCALE_PREFIX.size + len(stream))
        disk_record_bytes = align_up(disk_scale_bytes + weight_bytes)
        if disk_scale_bytes < len(scales) and disk_record_bytes < raw_record_bytes:
            pages = disk_scale_bytes // WEX_ALIGN
            if pages > 0xFFFFFF:
                raise ValueError("scale tile prefix exceeds WEX flag capacity")
            prefix = SCALE_PREFIX.pack(WEX_SCALE_TILE_MAGIC, len(scales), len(stream))
            return prefix + stream, disk_scale_bytes, disk_record_bytes, (pages << 8) | WEX_SCALE_TILES
    return scales, len(scales), raw_record_bytes, 0


def _entry_metadata(matrices):
    scale_shapes = tuple(size for scale, _ in matrices for size in scale.shape)
    weight_shapes = tuple(size for _, weight in matrices for size in weight.shape)
    scale_bytes = sum(scale.length for scale, _ in matrices)
    weight_bytes = sum(weight.length for _, weight in matrices)
    scale_offsets, weight_offsets = [], []
    scale_cursor, weight_cursor = 0, scale_bytes
    for scale, weight in matrices:
        scale_offsets.append(scale_cursor)
        weight_offsets.append(weight_cursor)
        scale_cursor += scale.length
        weight_cursor += weight.length
    return scale_bytes, weight_bytes, scale_shapes, weight_shapes, scale_offsets, weight_offsets


def validate_container(path, verify_payload=False):
    """Validate v1/v2/v3 metadata and optionally decode compressed payloads."""
    path = Path(path)
    file_bytes = path.stat().st_size
    with path.open("rb") as source:
        raw_header = source.read(WEX_ALIGN)
        if len(raw_header) != WEX_ALIGN:
            raise ValueError("truncated WEX header")
        magic, version, layers, experts, record_bytes, table_offset, flags, reserved = HEADER.unpack_from(raw_header)
        if (magic, version) not in ((WEX_MAGIC_V1, 1), (WEX_MAGIC_V2, 2), (WEX_MAGIC_V3, 3)):
            raise ValueError("unsupported WEX magic/version")
        allowed_flags = 15 if version == 3 else WEX_SCALE_TILES if version == 2 else 0
        if flags & ~allowed_flags or reserved:
            raise ValueError("unsupported WEX header flags/reserved")
        if not layers or not experts or not record_bytes or record_bytes % WEX_ALIGN:
            raise ValueError("invalid WEX dimensions/record size")
        table_bytes = layers * experts * ENTRY.size
        data_start = table_offset + align_up(table_bytes)
        if table_offset < WEX_ALIGN or table_offset % WEX_ALIGN or data_start > file_bytes:
            raise ValueError("invalid WEX table bounds")
        source.seek(table_offset)
        raw_table = source.read(table_bytes)
        if len(raw_table) != table_bytes:
            raise ValueError("truncated WEX table")
        expected_offset = data_start
        actual_max = 0
        entries = []
        physical_entries = {}
        for index in range(layers * experts):
            entry = ENTRY.unpack_from(raw_table, index * ENTRY.size)
            entries.append(entry)
            offset, total, scale_bytes, weight_bytes = entry[:4]
            scale_shapes, weight_shapes = entry[4:10], entry[10:16]
            scale_offsets, weight_offsets, entry_flags = entry[16:19], entry[19:22], entry[22]
            if not total:
                if any(entry):
                    raise ValueError("nonempty metadata for absent WEX expert")
                continue
            if (version == 3 and flags & WEX_DEDUP and offset in physical_entries and
                    entry == physical_entries[offset]):
                continue  # exact alias of a fully validated earlier record
            if offset != expected_offset or total % WEX_ALIGN or offset + total > file_bytes:
                raise ValueError("invalid WEX expert bounds/layout")
            scale_cursor, weight_cursor = 0, scale_bytes
            for matrix in range(3):
                sr, sc = scale_shapes[matrix * 2:matrix * 2 + 2]
                wr, wc = weight_shapes[matrix * 2:matrix * 2 + 2]
                if not sr or not sc or sr != wr or sc * 32 != wc * 2:
                    raise ValueError("invalid WEX expert shapes")
                if scale_offsets[matrix] != scale_cursor or weight_offsets[matrix] != weight_cursor:
                    raise ValueError("noncanonical WEX decoded offsets")
                scale_cursor += sr * sc
                weight_cursor += wr * wc
            if scale_cursor != scale_bytes or weight_cursor != scale_bytes + weight_bytes:
                raise ValueError("invalid WEX payload byte counts")
            decoded_record_bytes = align_up(scale_bytes + weight_bytes)
            if decoded_record_bytes > record_bytes:
                raise ValueError("decoded WEX record exceeds header capacity")
            actual_max = max(actual_max, decoded_record_bytes)
            if entry_flags == WEX_NEURAL_RANS:
                if version != 3 or not flags & WEX_HEADER_NEURAL_RANS or total >= decoded_record_bytes:
                    raise ValueError("invalid neural rANS entry flags/size")
                source.seek(offset)
                model, offsets, scale_stream, payload, chunk = neural_rans.parse_record(source.read(total), weight_bytes)
                if verify_payload:
                    scales = unpack_scale_tiles(scale_stream, scale_bytes)
                    neural_rans.decode_weights(model, offsets, payload, scales, weight_bytes, chunk)
            elif entry_flags == WEX_PALETTE3:
                if version != 3 or not flags & WEX_PALETTE3 or total >= decoded_record_bytes:
                    raise ValueError("invalid palette3 WEX entry flags/size")
                if decoded_record_bytes - total + palette3.PREFIX_BYTES < scale_bytes:
                    raise ValueError("unsafe palette3 scale expansion overlap")
                source.seek(offset)
                prefix = source.read(palette3.PREFIX.size)
                if len(prefix) != palette3.PREFIX.size:
                    raise ValueError("truncated palette3 prefix")
                p_magic, scale_encoded, weight_encoded, p_reserved = palette3.PREFIX.unpack(prefix)
                disk_bytes = palette3.PREFIX_BYTES + scale_encoded + weight_encoded
                if (p_magic != palette3.MAGIC or p_reserved or not scale_encoded or
                        weight_encoded != weight_bytes // 4 * 3 or align_up(disk_bytes) != total or disk_bytes > total):
                    raise ValueError("invalid palette3 WEX prefix")
                if verify_payload:
                    tables = source.read(palette3.TABLE_BYTES)
                    scale_stream = source.read(scale_encoded)
                    scales = unpack_scale_tiles(scale_stream, scale_bytes)
                    cursor = 0
                    for matrix in range(3):
                        groups = scale_shapes[2 * matrix] * scale_shapes[2 * matrix + 1]
                        palette3.unpack_matrix(scales[cursor:cursor + groups], tables[matrix * 1024:(matrix + 1) * 1024],
                                               source.read(groups * 12))
                        cursor += groups
            elif entry_flags:
                pages, codec = entry_flags >> 8, entry_flags & 255
                disk_scale_bytes = pages * WEX_ALIGN
                if version < 2 or not (flags & WEX_SCALE_TILES) or codec != WEX_SCALE_TILES or not pages or scale_bytes % WEX_ALIGN:
                    raise ValueError("invalid compressed WEX entry flags")
                if disk_scale_bytes >= scale_bytes or align_up(disk_scale_bytes + weight_bytes) != total or total >= decoded_record_bytes:
                    raise ValueError("invalid compressed WEX entry size")
                source.seek(offset)
                raw_prefix = source.read(SCALE_PREFIX.size)
                if len(raw_prefix) != SCALE_PREFIX.size:
                    raise ValueError("truncated WEX scale prefix")
                tile_magic, decoded_bytes, encoded_bytes = SCALE_PREFIX.unpack(raw_prefix)
                if tile_magic != WEX_SCALE_TILE_MAGIC or decoded_bytes != scale_bytes or not encoded_bytes or SCALE_PREFIX.size + encoded_bytes > disk_scale_bytes:
                    raise ValueError("invalid WEX scale prefix")
                if verify_payload:
                    stream = source.read(encoded_bytes)
                    if len(stream) != encoded_bytes:
                        raise ValueError("truncated WEX scale stream")
                    unpack_scale_tiles(stream, decoded_bytes)
            elif total != decoded_record_bytes:
                raise ValueError("invalid raw WEX record size")
            physical_entries[offset] = entry
            expected_offset += total
        if expected_offset != file_bytes or actual_max != record_bytes:
            raise ValueError("WEX file size/record maximum disagrees with table")
    return {"version": version, "layers": layers, "experts": experts,
            "record_bytes": record_bytes, "file_bytes": file_bytes, "entries": entries}


def _new_temp(destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
    return os.fdopen(descriptor, "w+b"), Path(name)


def _atomic_mirror(source, destination):
    if source.resolve() == destination.resolve():
        return
    stream, temporary = _new_temp(destination)
    try:
        with stream, source.open("rb") as input_stream:
            shutil.copyfileobj(input_stream, stream, length=1024 * 1024)
            stream.flush()
            os.fsync(stream.fileno())
        if temporary.stat().st_size != source.stat().st_size:
            raise ValueError("mirror size differs from primary")
        validate_container(temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def repack(model_dir, output=None, format_version=1, mirror_dir=None, dry_run=False,
           weight_codec="native", max_weight_relative_error=None, neural_cdf=None, neural_encoder_dll=None,
           neural_chunk_symbols=512):
    """V1/V2 dry-run reads scales only. V3 also reads weights for exact dedup/quantization."""
    if format_version not in (1, 2, 3):
        raise ValueError("format_version must be 1, 2 or 3")
    if weight_codec not in ("native", "palette3", "neural-rans") or (weight_codec != "native" and format_version != 3):
        raise ValueError("compressed weights require format_version=3")
    encoder = None
    if weight_codec == "neural-rans":
        if not neural_cdf or not neural_encoder_dll:
            raise ValueError("neural-rans requires neural_cdf and neural_encoder_dll")
        encoder = neural_rans.Encoder(neural_cdf, neural_encoder_dll, neural_chunk_symbols)
    if weight_codec == "palette3" and (isinstance(max_weight_relative_error, bool) or
            not isinstance(max_weight_relative_error, (int, float)) or
            not math.isfinite(max_weight_relative_error) or not 0 <= max_weight_relative_error <= 1):
        raise ValueError("palette3 requires an explicit finite max_weight_relative_error in [0,1]")
    model_dir = Path(model_dir).resolve()
    with (model_dir / "config.json").open("r", encoding="utf-8") as source:
        config = json.load(source, object_pairs_hook=_unique_object)
    if not isinstance(config, dict):
        raise ValueError("config.json must be an object")
    layers, experts, dims = _model_dims(model_dir, config)
    table_bytes = layers * experts * ENTRY.size
    if align_up(table_bytes) > UINT32_MAX:
        raise ValueError("WEX table exceeds single-read uint32 limit")
    tensors = index_safetensors(model_dir)
    plan = _expert_plan(tensors, layers, experts, dims)
    default_name = ("v4_flash_experts.wex" if format_version == 1 else
                    "v4_flash_experts.v2.wex" if format_version == 2 else
                    f"v4_flash_experts.v3.{weight_codec}.wex" if weight_codec != "native" else "v4_flash_experts.v3.wex")
    destination = Path(output).resolve() if output is not None else model_dir / default_name
    protected_inputs = {ref.path.resolve() for ref in tensors.values()}
    protected_inputs.update((model_dir / name).resolve()
                            for name in ("config.json", "model.safetensors.index.json", "inference/config.json"))
    if encoder is not None:
        protected_inputs.update((Path(neural_cdf).resolve(), Path(neural_encoder_dll).resolve()))
    if destination in protected_inputs or (mirror_dir is not None and
            (Path(mirror_dir).resolve() / destination.name) in protected_inputs):
        raise ValueError("output must not replace a model input")
    data_start = WEX_ALIGN + align_up(table_bytes)
    entries = []
    current_offset = data_start
    max_record_bytes = raw_payload_bytes = compressed_experts = 0
    palette_experts = neural_experts = dedup_experts = dedup_bytes = 0
    max_accepted_error = max_candidate_error = 0.
    identities = {}
    stream = temporary = None
    try:
        if not dry_run:
            stream, temporary = _new_temp(destination)
            stream.write(bytes(data_start))
        with ExitStack() as stack:
            handles = {path: stack.enter_context(path.open("rb")) for path in sorted({ref.path for ref in tensors.values()})}
            for matrices in plan:
                if matrices is None:
                    entries.append(bytes(ENTRY.size))
                    continue
                scale_bytes, weight_bytes, scale_shapes, weight_shapes, scale_offsets, weight_offsets = _entry_metadata(matrices)
                scales = b"".join(_read_tensor(scale, handles) for scale, _ in matrices)
                decoded_bytes = align_up(scale_bytes + weight_bytes)
                max_record_bytes = max(max_record_bytes, decoded_bytes)
                raw_payload_bytes += decoded_bytes
                weights = tuple(_read_tensor(weight, handles) for _, weight in matrices) if format_version == 3 else None
                identity = None
                if weights is not None:
                    metadata = ENTRY.pack(0, 0, scale_bytes, weight_bytes, *scale_shapes, *weight_shapes,
                                          *scale_offsets, *weight_offsets, 0)
                    digest = hashlib.sha256(metadata + scales)
                    for part in weights:
                        digest.update(part)
                    identity = digest.digest()
                    previous = identities.get(identity)
                    # Hash narrows candidates; actual bytes establish equality.
                    if previous is not None:
                        previous_entry, previous_matrices, previous_error, previous_metadata = previous
                        same_scales = metadata == previous_metadata and all(_read_tensor(old[0], handles) == scales[scale_offsets[m]:scale_offsets[m] + old[0].length]
                                          for m, old in enumerate(previous_matrices))
                        if same_scales and all(_read_tensor(old[1], handles) == weights[m]
                                               for m, old in enumerate(previous_matrices)):
                            entries.append(previous_entry)
                            reused = ENTRY.unpack(previous_entry)
                            compressed_experts += bool(reused[22])
                            palette_experts += reused[22] == WEX_PALETTE3
                            neural_experts += reused[22] == WEX_NEURAL_RANS
                            max_accepted_error = max(max_accepted_error, previous_error)
                            dedup_experts += 1
                            dedup_bytes += reused[1]
                            continue
                scale_data, disk_scale_bytes, total_bytes, entry_flags = _scale_storage(scales, weight_bytes, format_version)
                stored_weights = weights
                accepted_error = 0.
                if weight_codec == "palette3" and 255 not in scales:
                    results = [palette3.pack_matrix(scales[scale_offsets[m]:scale_offsets[m] + matrices[m][0].length], weights[m])
                               for m in range(3)]
                    candidate_error = max(result[2] for result in results)
                    max_candidate_error = max(max_candidate_error, candidate_error)
                    stream_scales = pack_scale_tiles(scales)
                    tables = b"".join(result[0] for result in results)
                    candidate_prefix = palette3.PREFIX.pack(palette3.MAGIC, len(stream_scales), weight_bytes // 4 * 3, 0)
                    candidate_data = candidate_prefix + tables + stream_scales
                    candidate_total = align_up(len(candidate_data) + weight_bytes // 4 * 3)
                    if (candidate_error <= max_weight_relative_error and candidate_total < total_bytes and
                            decoded_bytes - candidate_total + palette3.PREFIX_BYTES >= scale_bytes):
                        scale_data, disk_scale_bytes = candidate_data, len(candidate_data)
                        total_bytes, entry_flags = candidate_total, WEX_PALETTE3
                        stored_weights = tuple(result[1] for result in results)
                        palette_experts += 1
                        accepted_error = candidate_error
                        max_accepted_error = max(max_accepted_error, accepted_error)
                if encoder is not None and all(encoder.mapping[s] != 65535 for s in set(scales)):
                    neural_record = encoder.encode(scales, b"".join(weights), pack_scale_tiles(scales))
                    if align_up(len(neural_record)) < total_bytes:
                        scale_data, disk_scale_bytes = neural_record, len(neural_record)
                        total_bytes, entry_flags = align_up(len(neural_record)), WEX_NEURAL_RANS
                        stored_weights = (b"", b"", b"")
                        neural_experts += 1
                compressed_experts += bool(entry_flags)
                entries.append(ENTRY.pack(current_offset, total_bytes, scale_bytes, weight_bytes,
                                          *scale_shapes, *weight_shapes, *scale_offsets, *weight_offsets, entry_flags))
                if identity is not None:
                    identities[identity] = entries[-1], matrices, accepted_error, metadata
                if stream is not None:
                    stream.write(scale_data)
                    stream.write(bytes(disk_scale_bytes - len(scale_data)))
                    for m, (_, weight) in enumerate(matrices):
                        stream.write(stored_weights[m] if stored_weights is not None else _read_tensor(weight, handles))
                    stored_weight_bytes = (0 if entry_flags == WEX_NEURAL_RANS else
                                           weight_bytes // 4 * 3 if entry_flags == WEX_PALETTE3 else weight_bytes)
                    stream.write(bytes(total_bytes - disk_scale_bytes - stored_weight_bytes))
                current_offset += total_bytes
        header = HEADER.pack((WEX_MAGIC_V1, WEX_MAGIC_V2, WEX_MAGIC_V3)[format_version - 1],
                             format_version, layers, experts, max_record_bytes, WEX_ALIGN,
                             (WEX_SCALE_TILES | WEX_DEDUP | (WEX_PALETTE3 if weight_codec == "palette3" else 0) |
                              (WEX_HEADER_NEURAL_RANS if encoder is not None else 0))
                             if format_version == 3 else WEX_SCALE_TILES if format_version == 2 else 0, 0)
        if stream is not None:
            header_page = header + (uuid.uuid4().bytes if format_version >= 2 else b"")
            stream.seek(0)
            stream.write(header_page + bytes(WEX_ALIGN - len(header_page)))
            stream.seek(WEX_ALIGN)
            for entry in entries:
                stream.write(entry)
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            stream = None
            validated = validate_container(temporary)
            if validated["file_bytes"] != current_offset:
                raise ValueError("written WEX size differs from packing plan")
            os.replace(temporary, destination)
            temporary = None
            if mirror_dir is not None:
                # Readers resolve mirrors using the primary container basename.
                _atomic_mirror(destination, Path(mirror_dir).resolve() / destination.name)
        return {"output": str(destination), "format_version": format_version, "dry_run": dry_run,
                "layers": layers, "experts_per_layer": experts, "record_bytes": max_record_bytes,
                "experts": sum(item is not None for item in plan), "compressed_experts": compressed_experts,
                "weight_codec": weight_codec, "palette3_experts": palette_experts, "lossy": bool(palette_experts),
                "neural_rans_experts": neural_experts,
                "max_weight_relative_error": max_accepted_error, "max_candidate_weight_relative_error": max_candidate_error,
                "deduplicated_experts": dedup_experts, "deduplicated_bytes": dedup_bytes,
                "file_bytes": current_offset, "raw_file_bytes": data_start + raw_payload_bytes,
                "saved_bytes": data_start + raw_payload_bytes - current_offset}
    finally:
        if stream is not None:
            stream.close()
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=r"C:\models\DeepSeek-V4-Flash-0731")
    parser.add_argument("--output", help="Output path; default v1/v2 filenames are distinct")
    parser.add_argument("--mirror-dir", help="Optional second SSD directory (published atomically)")
    parser.add_argument("--format-version", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--weight-codec", choices=("native", "palette3", "neural-rans"), default="native",
                        help="neural-rans is LOSSLESS with CUDA encoding; palette3 is LOSSY; V3 only")
    parser.add_argument("--neural-cdf", help="Compiled integer neural CDF model, required for neural-rans")
    parser.add_argument("--neural-encoder-dll", help="CUDA encoder DLL, required for neural-rans")
    parser.add_argument("--neural-chunk-symbols", type=int, choices=(256, 512, 1024), default=512,
                        help="Neural rANS parallelism/size tradeoff; default 512")
    parser.add_argument("--max-weight-relative-error", type=float,
                        help="Required for palette3: maximum relative RMS weight error per matrix; not a model accuracy guarantee")
    parser.add_argument("--dry-run", action="store_true", help="Write no files; V3 also reads weights for dedup/quantization")
    args = parser.parse_args(argv)
    try:
        result = repack(args.model_dir, args.output, args.format_version, args.mirror_dir, args.dry_run,
                        args.weight_codec, args.max_weight_relative_error, args.neural_cdf, args.neural_encoder_dll,
                        args.neural_chunk_symbols)
    except (OSError, ValueError, KeyError, struct.error) as error:
        parser.exit(1, f"ERROR: {error}\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
