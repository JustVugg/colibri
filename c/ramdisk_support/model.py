"""Safetensors model discovery and immutable model identity helpers."""

from __future__ import print_function

import hashlib
import json
import os
import re
import struct

from .common import MIB, RamdiskError


MAX_ST_HEADER = 512 * MIB

EXPERT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight(\.qs)?$"
)

def _read_safetensors_header(path):
    size = os.path.getsize(path)
    with open(path, "rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise RamdiskError("truncated safetensors file: %s" % path)
        header_size = struct.unpack("<Q", raw)[0]
        if header_size <= 1 or header_size > MAX_ST_HEADER or header_size > size - 8:
            raise RamdiskError("invalid safetensors header length in %s" % path)
        raw_header = stream.read(header_size)
        if len(raw_header) != header_size:
            raise RamdiskError("truncated safetensors header: %s" % path)
    try:
        header = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise RamdiskError("invalid safetensors header in %s: %s" % (path, exc))
    if not isinstance(header, dict):
        raise RamdiskError("safetensors header is not an object: %s" % path)
    data_start = 8 + header_size
    tensors = {}
    for name, record in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(record, dict) or "data_offsets" not in record:
            raise RamdiskError("invalid tensor record %r in %s" % (name, path))
        offsets = record["data_offsets"]
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) for value in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
            or data_start + offsets[1] > size
        ):
            raise RamdiskError("invalid tensor offsets for %s in %s" % (name, path))
        tensors[name] = {
            "dtype": record.get("dtype"),
            "shape": record.get("shape"),
            "offset": data_start + offsets[0],
            "bytes": offsets[1] - offsets[0],
        }
    return raw_header, tensors

def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(8 * MIB)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()

def _shape_numel(shape):
    if not isinstance(shape, list) or not shape or not all(isinstance(value, int) and value >= 0 for value in shape):
        return None
    result = 1
    for value in shape:
        result *= value
    return result

def _resolve_direct_format(rows, columns, weight_bytes, scale_bytes):
    """Mirror ``qt_resolve_fmt`` for an unstamped routed-expert tensor.

    Return ``(fmt, group_size)`` only when the engine can decode the exact
    weight/scale geometry. Ambiguous E8 layouts and recognized-but-unsupported
    FP8 UE8M0 sidecars fail closed, while the established unstamped
    int8-versus-FP8 collision rule continues to select int8.
    """
    int8_bytes = rows * columns
    int4_bytes = rows * ((columns + 1) // 2)
    int2_bytes = rows * ((columns + 3) // 4)
    int3_groups = (columns + 63) // 64
    int3_bytes = rows * int3_groups * 24
    e8_bytes = rows * ((columns + 255) // 256) * 98
    fp8_blocks = ((rows + 127) // 128) * ((columns + 127) // 128)

    # qt_resolve_fmt's SECOND DESIGN LANDMINE: an unstamped I=98 tensor can
    # satisfy E8 and one or more raw-byte formats simultaneously.
    if scale_bytes == 4 and weight_bytes == e8_bytes:
        raw_bytes_also = weight_bytes == int8_bytes
        if raw_bytes_also and (fp8_blocks in (1, 4) or rows == 1):
            return None
        return 6, 0

    # Keep the engine's row-format precedence for small-shape byte collisions.
    if weight_bytes == int8_bytes:
        fmt, group_size = 1, 0
    elif weight_bytes == int4_bytes:
        fmt, group_size = 2, 0
        if scale_bytes > rows * 4:
            for candidate in (16, 32, 48, 64, 96, 128, 192, 256):
                if candidate > columns:
                    break
                if scale_bytes == rows * ((columns + candidate - 1) // candidate) * 4:
                    fmt, group_size = 4, candidate
                    break
    elif weight_bytes == int2_bytes:
        fmt, group_size = 3, 0
    elif weight_bytes == int3_bytes:
        fmt, group_size = 5, 0
    else:
        return None

    if fmt == 1:
        is_row = scale_bytes == rows * 4
        is_fp8_f32 = scale_bytes == fp8_blocks * 4
        is_fp8_ue8m0 = scale_bytes == fp8_blocks
        if is_row and is_fp8_f32:
            pass  # Unstamped collision: qt_resolve_fmt selects incumbent int8.
        elif is_fp8_ue8m0:
            return None
        elif is_fp8_f32 and not is_row:
            fmt = 8

    if fmt == 4:
        expected_scales = rows * ((columns + group_size - 1) // group_size)
    elif fmt == 5:
        expected_scales = rows * int3_groups
    elif fmt == 8:
        expected_scales = fp8_blocks
    else:
        expected_scales = rows
    if scale_bytes != expected_scales * 4:
        return None
    return fmt, group_size

def _direct_tensor_set_eligible(entry, config):
    hidden = int(config["hidden_size"])
    intermediate = int(config["moe_intermediate_size"])
    prefix = "model.layers.%d.mlp.experts.%d." % (entry["layer"], entry["expert"])
    for projection, rows, columns in (
        ("gate_proj", intermediate, hidden),
        ("up_proj", intermediate, hidden),
        ("down_proj", hidden, intermediate),
    ):
        weight = entry["tensors"][prefix + projection + ".weight"]
        scale = entry["tensors"][prefix + projection + ".weight.qs"]
        if (
            weight["dtype"] not in ("U8", "I8")
            or scale["dtype"] != "F32"
            or weight["offset"] % 4
            or scale["offset"] % 4
            or _shape_numel(weight["shape"]) != weight["bytes"]
            or _shape_numel(scale["shape"]) != scale["bytes"] // 4
            or scale["bytes"] % 4
        ):
            return False
        weight_bytes = weight["bytes"]
        if _resolve_direct_format(rows, columns, weight_bytes, scale["bytes"]) is None:
            return False
    return True

def scan_model(model_dir):
    """Index shards and each expert's complete six-tensor direct-map closure."""
    model_dir = os.path.realpath(os.path.abspath(os.path.expanduser(model_dir)))
    if not os.path.isdir(model_dir):
        raise RamdiskError("model directory not found: %s" % model_dir)
    names = sorted(name for name in os.listdir(model_dir) if name.endswith(".safetensors"))
    if not names:
        raise RamdiskError("no .safetensors shards found in %s" % model_dir)
    fingerprint = hashlib.sha256()
    identity_files = {}
    required_metadata = ("config.json", "tokenizer.json")
    optional_metadata = ("generation_config.json", "tokenizer_config.json")
    for name in required_metadata + optional_metadata:
        path = os.path.join(model_dir, name)
        if not os.path.isfile(path):
            if name in required_metadata:
                raise RamdiskError("required model metadata is missing: %s" % path)
            continue
        digest = _sha256_file(path)
        size = os.path.getsize(path)
        identity_files[name] = {"size_bytes": size, "sha256": digest}
        fingerprint.update(("metadata\0%s\0%d\0%s\n" % (name, size, digest)).encode("utf-8"))
    config_path = os.path.join(model_dir, "config.json")
    try:
        with open(config_path, "r", encoding="utf-8") as stream:
            config = json.load(stream)
    except (OSError, ValueError) as exc:
        raise RamdiskError("cannot parse %s: %s" % (config_path, exc))
    if not isinstance(config, dict):
        raise RamdiskError("config.json must contain a JSON object")
    required_positive = (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "n_routed_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "intermediate_size",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "n_shared_experts",
        "vocab_size",
    )
    missing = [name for name in required_positive if not isinstance(config.get(name), int) or config[name] <= 0]
    if missing:
        raise RamdiskError("config.json is missing positive engine fields: %s" % ", ".join(missing))
    shards = []
    experts = {}
    tensor_bytes = 0
    expert_tensor_bytes = 0
    for name in names:
        path = os.path.join(model_dir, name)
        if not os.path.isfile(path):
            raise RamdiskError("shard is not a regular file: %s" % path)
        st = os.stat(path, follow_symlinks=True)
        raw_header, tensors = _read_safetensors_header(path)
        header_digest = hashlib.sha256(raw_header).hexdigest()
        identity = "%s\0%d\0%d\0%d\0%s\n" % (
            name,
            st.st_size,
            st.st_mtime_ns,
            st.st_ino,
            header_digest,
        )
        fingerprint.update(identity.encode("utf-8"))
        shards.append(
            {
                "name": name,
                "path": path,
                "size_bytes": st.st_size,
                "device": st.st_dev,
                "inode": st.st_ino,
                "mtime_ns": st.st_mtime_ns,
                "header_sha256": header_digest,
                "tensor_count": len(tensors),
            }
        )
        for tensor_name, tensor in tensors.items():
            tensor_bytes += tensor["bytes"]
            match = EXPERT_RE.match(tensor_name)
            if not match:
                continue
            layer, expert = int(match.group(1)), int(match.group(2))
            key = "%d:%d" % (layer, expert)
            entry = experts.setdefault(
                key,
                {
                    "layer": layer,
                    "expert": expert,
                    "tensors": {},
                    "shards": set(),
                    "tensor_bytes": 0,
                },
            )
            entry["tensors"][tensor_name] = {
                "shard": name,
                "bytes": tensor["bytes"],
                "dtype": tensor["dtype"],
                "shape": tensor["shape"],
                "offset": tensor["offset"],
            }
            entry["shards"].add(name)
            entry["tensor_bytes"] += tensor["bytes"]
            expert_tensor_bytes += tensor["bytes"]
    complete = {}
    for key, entry in experts.items():
        prefix = "model.layers.%d.mlp.experts.%d." % (entry["layer"], entry["expert"])
        expected = set()
        for projection in ("gate_proj", "up_proj", "down_proj"):
            weight = prefix + projection + ".weight"
            expected.add(weight)
            expected.add(weight + ".qs")
        if expected == set(entry["tensors"]):
            entry["shards"] = sorted(entry["shards"])
            entry["direct_map_eligible"] = _direct_tensor_set_eligible(entry, config)
            complete[key] = entry
    total_bytes = sum(shard["size_bytes"] for shard in shards)
    return {
        "path": model_dir,
        "fingerprint": "sha256:" + fingerprint.hexdigest(),
        "fingerprint_algorithm": "metadata content plus sorted shard name,size,mtime,inode,header-sha256",
        "identity_files": identity_files,
        "shards": shards,
        "shard_names": names,
        "total_shard_bytes": total_bytes,
        "tensor_bytes": tensor_bytes,
        "dense_tensor_bytes": max(0, tensor_bytes - expert_tensor_bytes),
        "experts": complete,
        "complete_experts": len(complete),
        "config": config,
    }
