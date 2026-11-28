"""Read-only recipient preflight. Parses existing GGUF, never loads inference."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "c/tools"))
from gguf_reader import GGUFReader, ggml_type_name
sys.path.insert(0, str(ROOT / "c"))
from family_registry import family_for_config


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory(model_dir, partial_dir=None):
    started = time.perf_counter()
    config = json.loads((model_dir / "config.json").read_text())
    link = json.loads((model_dir / "gguf.json").read_text())
    weights = Path(link["gguf"])
    before = weights.stat()
    with GGUFReader(str(weights)) as reader:
        tensors = reader.tensors
        metadata = reader.metadata
        index = [{k: t[k] for k in ("name", "dims", "ggml_type", "abs_offset", "nbytes")}
                 for t in tensors]
        counts = Counter(ggml_type_name(t["ggml_type"]) for t in tensors)
        tensor_end = max(t["abs_offset"] + t["nbytes"] for t in tensors)
        selected = {k: metadata[k] for k in
                    ("general.architecture", "general.file_type", "general.quantization_version",
                     "qwen35.block_count", "qwen35.embedding_length", "qwen35.context_length")
                    if k in metadata}
    weight_hash = sha(weights)
    after = weights.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("model changed during inventory")
    result = {
        "id": "CASCADE-Q36-RECIPIENT-001", "version": 1,
        "scope": "read-only metadata, complete tensor bounds and full file hash; no inference",
        "model": "Qwen3.5-9B", "engine_family": family_for_config(config).id,
        "not_literal_qwen36_35b": True,
        "route_evidence": "c/family_registry.py qwen36.model_types includes qwen3_5_text; engine_artifact=qwen36",
        "model_type": config["model_type"], "weight_basename": weights.name,
        "weight_bytes": after.st_size, "weight_sha256": weight_hash,
        "tensor_count": len(index), "tensor_types": dict(counts),
        "tensor_index_sha256": hashlib.sha256(json.dumps(index, sort_keys=True).encode()).hexdigest(),
        "largest_tensor_end": tensor_end, "all_tensor_ranges_present": tensor_end <= after.st_size,
        "quantisation": "Q6_K with other tensor types as inventoried; unchanged file",
        "metadata": selected,
        "geometry": {k: config[k] for k in ("hidden_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size",
            "max_position_embeddings")},
        "layer_type_counts": dict(Counter(config["layer_types"])),
        "local_files": {name: {"bytes": (model_dir / name).stat().st_size,
                              "sha256": sha(model_dir / name)}
                        for name in ("config.json", "gguf.json", "tokenizer.json",
                                     "tokenizer_config.json", "generation_config.json")},
        "execution_ready": False,
        "pending": ["parent GPU/build admission", "same-build private engine/backend hashes and load gate",
                    "frozen prompt token counts", "sampler fraction and full-model metrics"],
        "gpu_jobs": 0, "model_modifications": 0,
    }
    if partial_dir:
        result["literal_qwen36_availability"] = {
            "status": "incomplete_not_eligible",
            "partial_files": [{"basename": p.name, "bytes": p.stat().st_size}
                              for p in sorted(partial_dir.glob("*.partial"))],
            "complete_gguf_files": len(list(partial_dir.glob("*.gguf")))}
    result["inventory_wall_seconds"] = time.perf_counter() - started
    return result


def local_candidates(root):
    candidates = []
    for path in sorted(root.rglob("*.gguf")):
        if "qwen" not in str(path.relative_to(root)).lower() or path.name.startswith("mmproj"):
            continue
        item = {"relative_locator": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size, "full_file_sha256": None}
        try:
            with GGUFReader(str(path)) as reader:
                arch = reader.metadata.get("general.architecture")
                item.update(architecture=arch, tensor_count=len(reader.tensors),
                            tensor_types=dict(Counter(ggml_type_name(t["ggml_type"]) for t in reader.tensors)),
                            tensor_bounds_present=True,
                            engine_family="qwen36" if arch in ("qwen35", "qwen35moe", "qwen3", "qwen3moe") else None,
                            qualification="source-compatible only; not loaded or benchmarked")
        except (ValueError, OSError) as exc:
            item.update(qualification="metadata inspection failed", error_type=type(exc).__name__)
        candidates.append(item)
    return {"scope": "read-only Qwen GGUF metadata inventory under user-provided root; no full hashes or inference",
            "candidates": candidates}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--partial-dir", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--models-root", type=Path)
    p.add_argument("--catalog-only", action="store_true")
    args = p.parse_args()
    if args.catalog_only:
        if not args.models_root:
            p.error("--catalog-only needs --models-root")
        result = local_candidates(args.models_root)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return
    if not args.model_dir:
        p.error("--model-dir required unless --catalog-only")
    result = inventory(args.model_dir, args.partial_dir)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.models_root:
        landscape = local_candidates(args.models_root)
        args.output.with_name("local-model-candidates.json").write_text(json.dumps(landscape, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
