#!/usr/bin/env python3
"""Read-only installation diagnostics for colibri."""

import os
import sys
import json
import re
import subprocess
from pathlib import Path

from family_registry import (FamilyConfigError, PlannerUnsupportedError, UnknownFamilyError,
                             public_metadata, resolve_model)
from resource_plan import (GB, SSD_PROBE_PENDING, build_plan, discover_gpus, format_plan,
                           memory_available)

SAFETENSORS_MAX_HEADER = 512 << 20
MODEL_INDEX_MAX_BYTES = SAFETENSORS_MAX_HEADER
MAX_SAFETENSORS_SHARDS = 512
SAFETENSORS_DTYPES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "U8": 1,
    "I8": 1,
    # dtypes st.h's st_dtype_code accepts beyond the classic set (DeepSeek V4 /
    # Kimi-style checkpoints); sizes mirror st_dtype_esz exactly.
    "I64": 8,
    "U64": 8,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "float8_e4m3fn": 1,
    "F8_E8M0": 1,
    "F8_E8M0FNU": 1,
}
REQUIRED_CORE_TENSORS = (
    "model.embed_tokens.weight",
    "model.norm.weight",
    "lm_head.weight",
)


def _check(identifier, status, summary, **details):
    item = {"id": identifier, "status": status, "summary": summary}
    if details:
        item["details"] = details
    return item


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _safetensors_header(path):
    """Read one bounded safetensors header without touching tensor payloads."""
    path = Path(path)
    with path.open("rb") as stream:
        file_size = os.fstat(stream.fileno()).st_size
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise ValueError("short safetensors header")
        header_length = int.from_bytes(raw_length, "little")
        if (header_length < 2 or header_length > SAFETENSORS_MAX_HEADER or
                header_length > file_size - 8):
            raise ValueError(f"invalid safetensors header length: {header_length}")
        raw_header = stream.read(header_length)
        if len(raw_header) != header_length:
            raise ValueError("short safetensors header body")
    try:
        header = json.loads(raw_header, object_pairs_hook=_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid safetensors JSON: {error}") from error
    if not isinstance(header, dict):
        raise ValueError("safetensors header is not a JSON object")
    return file_size, raw_header, header


def _tensor_layout(meta, payload_size):
    if not isinstance(meta, dict):
        raise ValueError("tensor metadata is not an object")
    dtype = meta.get("dtype")
    offsets = meta.get("data_offsets")
    shape = meta.get("shape")
    if dtype not in SAFETENSORS_DTYPES:
        raise ValueError(f"unsupported dtype: {dtype!r}")
    if (not isinstance(offsets, list) or len(offsets) != 2 or
            any(isinstance(value, bool) or not isinstance(value, int) for value in offsets)):
        raise ValueError("data_offsets must contain exactly two integers")
    start, end = offsets
    if start < 0 or end < start or end > payload_size:
        raise ValueError(f"data_offsets [{start}, {end}] exceed payload size {payload_size}")
    if (not isinstance(shape, list) or
            any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in shape)):
        raise ValueError("shape must contain only non-negative integers")
    elements = 1
    for dimension in shape:
        elements *= dimension
        if elements > (1 << 63) - 1:
            raise ValueError("shape element count exceeds int64")
    if dtype not in ("U8", "I8") and end - start != elements * SAFETENSORS_DTYPES[dtype]:
        raise ValueError("shape and dtype disagree with the tensor byte span")
    return start, end


def _shard_sequence_report(shards):
    hf_shards = []
    out_shards = []
    for shard in shards:
        hf_match = re.fullmatch(r"model-(\d+)-of-(\d+)\.safetensors", shard.name)
        out_match = re.fullmatch(r"out-(\d+)\.safetensors", shard.name)
        if hf_match:
            hf_shards.append(tuple(map(int, hf_match.groups())))
        elif out_match:
            out_shards.append(int(out_match.group(1)))
    if hf_shards and out_shards:
        return {
            "status": "fail",
            "summary": "model mixes filename-declared shard schemes",
            "details": {
                "huggingface_shards": len(hf_shards),
                "converter_shards": len(out_shards),
            },
        }
    if hf_shards:
        declared = {total for _, total in hf_shards}
        if len(declared) != 1:
            return {"status": "fail", "summary": "shard filenames declare different totals"}
        total = declared.pop()
        found = {index for index, _ in hf_shards}
        in_range = {index for index in found if 1 <= index <= total}
        missing = max(total - len(in_range), 0)
        unexpected = len(found - in_range)
        duplicates = len(hf_shards) - len(found)
        if missing or unexpected or duplicates:
            return {
                "status": "fail",
                "summary": "declared shard sequence is incomplete or inconsistent",
                "details": {
                    "declared_shards": total,
                    "found_shards": len(found),
                    "missing_shards": missing,
                    "unexpected_shards": unexpected,
                    "duplicate_shards": duplicates,
                },
            }
        return {
            "status": "pass",
            "summary": "all filename-declared shards are present",
            "details": {"declared_shards": total, "found_shards": len(found)},
        }
    if out_shards:
        found = set(out_shards)
        first = min(found)
        last = max(found)
        missing = last + 1 - len(found)
        duplicates = len(out_shards) - len(found)
        if missing or duplicates:
            return {
                "status": "fail",
                "summary": "converter shard numbering contains gaps or duplicates",
                "details": {
                    "first_shard": first,
                    "last_shard": last,
                    "found_shards": len(found),
                    "missing_shards": missing,
                    "duplicate_shards": duplicates,
                    "tail_completeness_declared": False,
                },
            }
        return {
            "status": "pass",
            "summary": "converter shard numbering is contiguous",
            "details": {
                "first_shard": min(found),
                "last_shard": max(found),
                "found_shards": len(found),
                "tail_completeness_declared": False,
            },
        }
    return {"status": "skip", "summary": "shard filenames do not declare a sequence"}


def deep_container_report(model, mirror_dir=None):
    """Validate all tensor headers/layouts and runtime-equivalent mirror admission."""
    model = Path(model).expanduser().resolve()
    shards = sorted(model.glob("*.safetensors"))
    if not shards:
        raise ValueError("no safetensors shards found")
    if len(shards) > MAX_SAFETENSORS_SHARDS:
        raise ValueError(
            f"more than {MAX_SAFETENSORS_SHARDS} safetensors shards "
            "are not supported by the runtime"
        )

    tensor_sources = {}
    shard_headers = {}
    tensor_count = 0
    header_bytes = 0
    payload_bytes = 0
    for shard in shards:
        try:
            file_size, raw_header, header = _safetensors_header(shard)
        except (OSError, ValueError) as error:
            raise ValueError(f"{shard.name}: {error}") from error
        payload_size = file_size - 8 - len(raw_header)
        ranges = []
        for name, meta in header.items():
            if name == "__metadata__":
                if not isinstance(meta, dict):
                    raise ValueError(f"{shard.name}: __metadata__ is not an object")
                continue
            if name in tensor_sources:
                raise ValueError(
                    f"duplicate tensor {name!r} in {tensor_sources[name]} and {shard.name}"
                )
            try:
                start, end = _tensor_layout(meta, payload_size)
            except ValueError as error:
                raise ValueError(f"{shard.name}: tensor {name!r}: {error}") from error
            tensor_sources[name] = shard.name
            tensor_count += 1
            ranges.append((start, end, name))
        ranges = [item for item in ranges if item[0] != item[1]]
        ranges.sort()
        for previous, current in zip(ranges, ranges[1:]):
            if current[0] < previous[1]:
                raise ValueError(
                    f"{shard.name}: tensors {previous[2]!r} and {current[2]!r} overlap"
                )
        shard_headers[shard.name] = (file_size, raw_header)
        header_bytes += len(raw_header)
        payload_bytes += payload_size

    index_path = model / "model.safetensors.index.json"
    index = {"status": "skip", "summary": "model index is not present"}
    if index_path.is_file():
        try:
            with index_path.open("rb") as stream:
                index_size = os.fstat(stream.fileno()).st_size
                if index_size > MODEL_INDEX_MAX_BYTES:
                    raise ValueError(
                        f"model index exceeds {MODEL_INDEX_MAX_BYTES} bytes"
                    )
                raw_index = stream.read(index_size + 1)
                if len(raw_index) != index_size:
                    raise ValueError("model index changed while reading")
            document = json.loads(raw_index, object_pairs_hook=_json_object)
            if not isinstance(document, dict):
                raise ValueError("model index is not a JSON object")
            weight_map = document.get("weight_map")
            if not isinstance(weight_map, dict):
                raise ValueError("weight_map is not an object")
            if any(not isinstance(name, str) or not isinstance(shard, str)
                   for name, shard in weight_map.items()):
                raise ValueError("weight_map keys and values must be strings")
            missing_tensors = sorted(set(weight_map) - set(tensor_sources))
            unindexed_tensors = sorted(set(tensor_sources) - set(weight_map))
            misplaced_tensors = sorted(
                name for name in set(weight_map) & set(tensor_sources)
                if weight_map[name] != tensor_sources[name]
            )
            unknown_shards = sorted(
                {name for name in weight_map.values() if name not in shard_headers}
            )
            if missing_tensors or unindexed_tensors or misplaced_tensors or unknown_shards:
                raise ValueError(
                    "index disagrees with scanned tensors "
                    f"(missing={len(missing_tensors)}, unindexed={len(unindexed_tensors)}, "
                    f"misplaced={len(misplaced_tensors)}, unknown_shards={len(unknown_shards)})"
                )
            index = {
                "status": "pass",
                "summary": "model index matches every scanned tensor",
                "details": {"indexed_tensors": len(weight_map)},
            }
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            index = {"status": "fail", "summary": f"model index is invalid: {error}"}

    missing_core = [name for name in REQUIRED_CORE_TENSORS if name not in tensor_sources]
    required = {
        "status": "fail" if missing_core else "pass",
        "summary": (
            f"{len(missing_core)} required core tensor(s) are missing"
            if missing_core else "required core tensors are present"
        ),
        "details": {
            "required_tensors": len(REQUIRED_CORE_TENSORS),
            "missing_tensors": missing_core,
        },
    }

    mirror = {"status": "skip", "summary": "no mirror directory is configured"}
    if mirror_dir:
        mirror_path = Path(mirror_dir).expanduser().resolve()
        accepted = 0
        missing = 0
        divergent = 0
        if not mirror_path.is_dir():
            mirror = {
                "status": "warn",
                "summary": "configured mirror directory is unavailable",
                "details": {"path": str(mirror_path), "accepted_shards": 0},
            }
        else:
            for name, (primary_size, primary_header) in shard_headers.items():
                candidate = mirror_path / name
                if not candidate.is_file():
                    missing += 1
                    continue
                try:
                    mirror_size, mirror_header, _ = _safetensors_header(candidate)
                except (OSError, ValueError):
                    divergent += 1
                    continue
                if mirror_size == primary_size and mirror_header == primary_header:
                    accepted += 1
                else:
                    divergent += 1
            if divergent:
                status = "warn"
                summary = "one or more mirror shards would be rejected by the runtime"
            elif accepted:
                status = "pass"
                summary = "mirror shards satisfy runtime size and header admission"
            else:
                status = "warn"
                summary = "configured mirror contains no admissible primary shards"
            mirror = {
                "status": status,
                "summary": summary,
                "details": {
                    "path": str(mirror_path),
                    "accepted_shards": accepted,
                    "missing_shards": missing,
                    "divergent_shards": divergent,
                    "partial_mirror_allowed": True,
                },
            }

    return {
        "container": {
            "shards": len(shards),
            "tensors": tensor_count,
            "header_bytes": header_bytes,
            "payload_bytes": payload_bytes,
            "payload_hashing": False,
        },
        "sequence": _shard_sequence_report(shards),
        "required": required,
        "index": index,
        "mirror": mirror,
    }


def cuda_linkage(engine_path):
    """Return CUDA linkage state without loading the executable or CUDA runtime."""
    engine = Path(engine_path)
    if not engine.is_file():
        return {"linked": False, "missing": False}
    # `sys.platform` alone selects the branch, so a test can exercise the
    # Windows probe on a POSIX host by faking it. Faking `os.name` instead
    # would repoint pathlib at Windows semantics and turn the POSIX fixture
    # path into a WindowsPath that no longer resolves, so `is_file()` above
    # would return early. On every real host the two agree and this reads
    # exactly as `os.name == "posix"` did.
    if os.name == "posix" and sys.platform != "win32":
        try:
            result = subprocess.run(["ldd", str(engine)], capture_output=True, text=True,
                                    timeout=3, check=False)
        except (OSError, subprocess.SubprocessError):
            return {"linked": False, "missing": False}
        # A HIP/ROCm build links libamdhip64 (never libcudart), so match both
        # vendors here or a working AMD engine is reported CPU-only (#663). Mirrors
        # the vendor-aware probe cuda_binary() already uses in c/coli.
        lines = [line for line in result.stdout.splitlines()
                 if "libcudart" in line or "libamdhip64" in line]
        return {"linked": any("not found" not in line for line in lines),
                "missing": any("not found" in line for line in lines)}
    if sys.platform == "win32":
        # Windows DLL-split builds never link the GPU runtime directly: the host
        # LoadLibrary's its backend at runtime (backend_loader.c), so there's no
        # import-table entry for ldd/dumpbin to see. Detect the GPU build via a
        # marker string baked into the engine's #ifdef COLI_CUDA block, then
        # require the backend artifact to sit next to the executable.
        #
        # WHICH artifact is not a guess. backend_loader.c compiles exactly one
        # basename into the host -- COLI_BACKEND_DLL is "coli_hip.dll" under
        # COLI_HIP_DLL and "coli_cuda.dll" otherwise -- so the binary states
        # what it will load and we check for that. Asking for coli_cuda.dll
        # unconditionally failed a working HIP host (a hard error, not a
        # warning), and accepting either name would have passed a HIP host that
        # only had a stray CUDA backend beside it.
        try:
            image = engine.read_bytes()
        except OSError:
            return {"linked": False, "missing": False}
        # The DeepSeek V4 engine has its own loader (backend_loader_dsv4.c):
        # it tries coli_cuda_dsv4_dg.dll then coli_cuda_dsv4.dll, so either
        # next to the engine means the tier can start.
        if b"[DSV4 CUDA]" in image:
            present = any((engine.parent / name).is_file()
                          for name in ("coli_cuda_dsv4_dg.dll", "coli_cuda_dsv4.dll"))
            return {"linked": present, "missing": not present}
        if b"[CUDA] mode: routed experts" not in image:
            return {"linked": False, "missing": False}
        expected = next((name for name in ("coli_hip.dll", "coli_cuda.dll")
                         if name.encode() in image), None)
        if expected is None:
            return {"linked": False, "missing": False}
        dll_present = (engine.parent / expected).is_file()
        return {"linked": dll_present, "missing": not dll_present}
    return {"linked": False, "missing": False}


def missing_shared_libraries(engine_path):
    """Shared libraries the engine needs but the loader cannot resolve.

    A binary built elsewhere (a prebuilt release, a copied build, a disk moved to a
    fresh host) can be present and executable yet still fail to start. The engine
    exits before printing anything and the caller only sees "engine exited
    unexpectedly", so name the unresolved libraries instead. Typical case: a minimal
    image without libgomp1, where every OpenMP build reports
    "libgomp.so.1 => not found".
    """
    engine = Path(engine_path)
    if os.name != "posix" or not engine.is_file():
        return []
    try:
        result = subprocess.run(["ldd", str(engine)], capture_output=True, text=True,
                                timeout=3, check=False)
    except (OSError, subprocess.SubprocessError):
        return []          # no ldd (musl, macOS): cannot tell, so claim nothing
    return sorted({line.split("=>")[0].strip()
                   for line in result.stdout.splitlines() if "not found" in line})


def run_doctor(model, ram_gb=0, context=4096, gpu_indices=None, vram_gb=0, *,
               engine_path, available_memory=None, available_disk=None, gpus=None,
               linkage=None, deep=False, mirror_dir=None, kv_slots=1,
               engine_error=None):
    """Collect a complete report. No model payload, engine, or CUDA context is loaded."""
    model = Path(model).expanduser().resolve()
    checks = []
    plan = None
    resolved = None

    if model.is_dir() and os.access(model, os.R_OK):
        checks.append(_check("model.path", "pass", "model directory is readable", path=str(model)))
    elif model.is_dir():
        checks.append(_check("model.path", "fail", "model directory is not readable", path=str(model)))
    else:
        checks.append(_check("model.path", "fail", "model directory does not exist", path=str(model)))

    config = model / "config.json"
    try:
        valid_config = isinstance(json.loads(config.read_text(encoding="utf-8")), dict)
    except (OSError, ValueError):
        valid_config = False
    checks.append(_check("model.config", "pass" if valid_config else "fail",
                         "config.json is valid" if valid_config else "config.json is missing or invalid"))
    if valid_config:
        try:
            resolved = resolve_model(model)
            checks.append(_check("model.family", "pass",
                                 f"{resolved.descriptor.display_name} family is registered",
                                 family_id=resolved.descriptor.id,
                                 model_type=resolved.model_type,
                                 descriptor=public_metadata(resolved.descriptor)))
        except (FamilyConfigError, UnknownFamilyError) as error:
            checks.append(_check("model.family", "fail", str(error)))
    else:
        checks.append(_check("model.family", "skip", "family detection requires a valid config"))
    tokenizer = model / "tokenizer.json"
    checks.append(_check("model.tokenizer", "pass" if tokenizer.is_file() else "fail",
                         "tokenizer.json found" if tokenizer.is_file() else "tokenizer.json is missing"))
    if model.is_dir() and os.access(model, os.W_OK):
        checks.append(_check("storage.persistence", "pass", "model directory can store usage and KV state"))
    elif model.is_dir():
        checks.append(_check("storage.persistence", "warn", "model directory is read-only; disable persistence or change permissions"))
    else:
        checks.append(_check("storage.persistence", "skip", "persistence requires a model directory"))

    engine = Path(engine_path)
    # On Windows, os.access(X_OK) always returns True for any existing file
    # (NTFS has no execute bit; executability is governed by file extension).
    # So a chmod(0o644) "non-executable" scenario can't be detected via X_OK
    # on Windows. Use a platform-aware check: on POSIX, honor the mode bits;
    # on Windows, any existing file is treated as executable. (#141)
    if sys.platform == "win32":
        engine_ok = engine.is_file()
    else:
        engine_ok = engine.is_file() and os.access(engine, os.X_OK)
    if engine_error:
        checks.append(_check("engine.binary", "fail", str(engine_error), path=str(engine)))
    elif engine_ok:
        unresolved = missing_shared_libraries(engine)
        if unresolved:
            checks.append(_check("engine.binary", "fail",
                                 "engine cannot load: " + ", ".join(unresolved) +
                                 " (install the runtime package, e.g. libgomp1, and retry)",
                                 path=str(engine), missing=unresolved))
        else:
            checks.append(_check("engine.binary", "pass", "engine executable is ready", path=str(engine)))
    elif engine.is_file():
        checks.append(_check("engine.binary", "fail", "engine exists but is not executable", path=str(engine)))
    else:
        checks.append(_check("engine.binary", "fail", "engine is not built", path=str(engine)))

    available_memory = memory_available() if available_memory is None else available_memory
    detected_gpus = discover_gpus() if gpus is None else list(gpus)
    linkage = cuda_linkage(engine) if linkage is None else linkage
    selected_gpus = detected_gpus
    if gpu_indices is not None:
        wanted = set(gpu_indices)
        selected_gpus = [gpu for gpu in detected_gpus if gpu["index"] in wanted]

    if gpu_indices == []:
        checks.append(_check("accelerator.gpu", "skip", "GPU use was explicitly disabled"))
    elif gpu_indices is not None and len(selected_gpus) != len(set(gpu_indices)):
        checks.append(_check("accelerator.gpu", "fail", "one or more requested GPUs were not detected",
                             requested=gpu_indices, detected=[gpu["index"] for gpu in detected_gpus]))
    elif selected_gpus and linkage.get("missing"):
        checks.append(_check("accelerator.gpu", "fail", "GPU runtime library is missing"))
    elif selected_gpus and linkage.get("linked"):
        checks.append(_check("accelerator.gpu", "pass", "GPU engine and devices are available",
                             devices=[gpu["index"] for gpu in selected_gpus],
                             unified=any(gpu.get("unified_memory", False)
                                         for gpu in selected_gpus)))
    elif selected_gpus:
        checks.append(_check("accelerator.gpu", "warn", "GPU detected but the engine is CPU-only",
                             devices=[gpu["index"] for gpu in selected_gpus]))
    else:
        checks.append(_check("accelerator.gpu", "skip", "no supported GPU detected; CPU path is available"))

    try:
        if resolved is None:
            raise ValueError("placement requires a registered model family")
        plan = build_plan(model, ram_gb, context, gpu_indices, vram_gb,
                          available_memory=available_memory, available_disk=available_disk,
                          gpus=detected_gpus, kv_slots=kv_slots)
        model_info = plan["model"]
        checks.append(_check("model.shards", "pass", "safetensors headers are valid",
                             shards=model_info["shards"], model_bytes=model_info["model_bytes"]))
        disk = plan["tiers"]["disk"]
        disk_status = "warn" if disk["available_bytes"] < GB else "pass"
        disk_summary = ("less than 1 GB is free for runtime state" if disk_status == "warn" else
                        "model backing store is available")
        checks.append(_check("storage.disk", disk_status, disk_summary,
                             available_bytes=disk["available_bytes"], model_bytes=disk["model_bytes"]))
        ram = plan["tiers"]["ram"]
        if not available_memory:
            ram_status, ram_summary = "warn", "available RAM could not be measured"
        elif ram["budget_bytes"] > available_memory:
            ram_status, ram_summary = "fail", "planned RAM budget exceeds available memory"
        elif ram["cache_slots_per_layer"] < 1:
            ram_status, ram_summary = "fail", "RAM budget cannot hold one expert slot per sparse layer"
        else:
            ram_status, ram_summary = "pass", "RAM budget is viable"
        checks.append(_check("memory.ram", ram_status, ram_summary,
                             available_bytes=available_memory, budget_bytes=ram["budget_bytes"],
                             cache_slots_per_layer=ram["cache_slots_per_layer"]))
        if plan["warnings"]:
            checks.append(_check("placement.plan", "warn", "; ".join(plan["warnings"])))
        else:
            checks.append(_check("placement.plan", "pass", "tier placement has no warnings"))
        # #379: read-and-display only -- the cached value colibri.c already measured
        # (F_NOCACHE probe) on a Metal+darwin startup, never re-probed here. A cache
        # that exists but is not trusted says WHY (#386 r2, F10) -- "no cached probe
        # yet" would be a lie with a file sitting right there.
        ssd_gbs = plan.get("ssd_probe_gbs")
        ssd_state = plan.get("ssd_probe_state")
        if ssd_gbs is not None:
            checks.append(_check("storage.ssd_probe", "pass",
                                 f"F_NOCACHE probe: {ssd_gbs:.1f} GB/s (cached, .coli_ssd)", gbs=ssd_gbs))
        elif ssd_state in SSD_PROBE_PENDING:
            checks.append(_check("storage.ssd_probe", "skip",
                                 SSD_PROBE_PENDING[ssd_state], state=ssd_state))
        else:
            checks.append(_check("storage.ssd_probe", "skip",
                                 "no cached probe yet; measured on the first Metal+darwin engine start"))
    except PlannerUnsupportedError as error:
        checks.append(_check("model.shards", "pass", "safetensors headers are readable",
                             shards=len(list(model.glob("*.safetensors")))))
        checks.append(_check("storage.disk", "skip",
                             "storage projection requires a family planner"))
        checks.append(_check("memory.ram", "skip",
                             "RAM projection requires a family planner"))
        checks.append(_check("placement.plan", "skip", str(error)))
        checks.append(_check("storage.ssd_probe", "skip",
                             "probe surfacing requires a family planner"))
    except (OSError, ValueError, KeyError, TypeError) as error:
        checks.append(_check("model.shards", "fail", str(error)))
        checks.append(_check("storage.disk", "skip", "storage check requires a valid model"))
        checks.append(_check("memory.ram", "skip", "RAM projection requires a valid model"))
        checks.append(_check("placement.plan", "skip", "placement requires a valid model"))
        checks.append(_check("storage.ssd_probe", "skip", "probe surfacing requires a valid model"))

    if deep:
        try:
            deep_report = deep_container_report(model, mirror_dir=mirror_dir)
            details = deep_report["container"]
            checks.append(_check(
                "model.container", "pass",
                "all tensor headers and layouts are internally consistent",
                **details,
            ))
            sequence = deep_report["sequence"]
            checks.append(_check(
                "model.shard_sequence", sequence["status"], sequence["summary"],
                **sequence.get("details", {}),
            ))
            required = deep_report["required"]
            checks.append(_check(
                "model.required", required["status"], required["summary"],
                **required.get("details", {}),
            ))
            index = deep_report["index"]
            checks.append(_check("model.index", index["status"], index["summary"],
                                 **index.get("details", {})))
            mirror = deep_report["mirror"]
            checks.append(_check("storage.mirror", mirror["status"], mirror["summary"],
                                 **mirror.get("details", {})))
        except (OSError, ValueError) as error:
            checks.append(_check("model.container", "fail", str(error)))
            checks.append(_check(
                "model.shard_sequence", "skip",
                "shard sequence check requires a valid container",
            ))
            checks.append(_check(
                "model.required", "skip",
                "required-tensor check requires a valid container",
            ))
            checks.append(_check("model.index", "skip", "index check requires a valid container"))
            checks.append(_check("storage.mirror", "skip", "mirror check requires a valid container"))

    statuses = {item["status"] for item in checks}
    status = "error" if "fail" in statuses else "warning" if "warn" in statuses else "ok"
    return {"schema_version": 1, "status": status, "model": str(model),
            "mode": "deep" if deep else "standard", "checks": checks, "plan": plan}


def format_doctor(report):
    icons = {"pass": "ok", "warn": "warn", "fail": "fail", "skip": "skip"}
    # model is null in the JSON when none was given (#724); say that rather than "None"
    lines = [f"colibri doctor · {report['model'] or '(no model given)'}"]
    for check in report["checks"]:
        lines.append(f"[{icons[check['status']]:>4}] {check['id']:<18} {check['summary']}")
    if report["plan"]:
        lines.extend(["", format_plan(report["plan"])])
    lines.extend(["", f"result {report['status']}"])
    return "\n".join(lines)


def exit_code(report):
    return 1 if report["status"] == "error" else 0
