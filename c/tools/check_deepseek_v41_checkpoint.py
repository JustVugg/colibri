#!/usr/bin/env python3
"""Validate a DeepSeek V4.1-Flash checkpoint against the engine's contract.

The point is to answer "is this checkpoint the model the V4.1 engine implements?"
*before* any of it is loaded -- and without the checkpoint on disk, since the
headers alone carry every name, shape and dtype:

    # offline against a saved headers dump (what CI can use)
    python tools/check_deepseek_v41_checkpoint.py --config config.json \
        --index model.safetensors.index.json --headers-json headers.json

    # straight from the Hub, reading only the safetensors headers (a few MB)
    python tools/check_deepseek_v41_checkpoint.py --from-hf deepseek-ai/DeepSeek-V4.1-Flash

    # against a local checkpoint
    python tools/check_deepseek_v41_checkpoint.py --model /models/DeepSeek-V4.1-Flash

Fail-closed by construction: every tensor name must be classified by a rule in
FAMILIES, and every rule that applies must match on shape and dtype. A name
nobody classified is a failure, not a warning -- an unclassified tensor is
exactly how a wrong-model run starts. Same discipline as
tools/check_glm53_checkpoint.py.

V4.1 specifics this asserts (docs/deepseek-v41-delta.md):
  * no token-id hash routing: zero `tid2eid` tensors and no `num_hash_layers`
    (V4's early-layer hash router is gone, replaced by engram tables)
  * shared KV/index: only `kv_source_layer_ids` own a compressor and the index
    keys; `index_source_layer_ids` are the layers that run an indexer, and the
    layers between two of them read the published top-k
  * `compressor.wgate` exists exactly where the compress ratio is > 1
  * engram tables are per-layer [rows, head_dim] fp8 with per-32 E8M0 scales
  * DSpark stages are heterogeneous: stage 0 owns the target-state projection,
    the last stage owns the markov head + confidence head
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import struct
import urllib.request
from pathlib import Path

DTYPE_BYTES = {"F8_E4M3": 1, "F8_E8M0": 1, "I8": 1, "U8": 1, "F16": 2, "BF16": 2,
               "F32": 4, "I32": 4, "I64": 8}


# ---------------------------------------------------------------- header access

def local_headers(directory: Path) -> dict[str, dict]:
    tensors: dict[str, dict] = {}
    for shard in sorted(directory.glob("*.safetensors")):
        with shard.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(size))
        tensors.update({name: spec for name, spec in header.items()
                        if name != "__metadata__"})
    return tensors


def hub_headers(repo: str, shard_names: list[str]) -> dict[str, dict]:
    """Read every shard header over HTTP Range: 8-byte length, then that many
    bytes of JSON. No weight byte is ever transferred."""
    def fetch(shard: str, start: int, end: int) -> bytes:
        url = f"https://huggingface.co/{repo}/resolve/main/{shard}"
        request = urllib.request.Request(
            url, headers={"Range": f"bytes={start}-{end}",
                          "User-Agent": "colibri-check-deepseek-v41"})
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()

    tensors: dict[str, dict] = {}
    for shard in shard_names:
        length = struct.unpack("<Q", fetch(shard, 0, 7))[0]
        header = json.loads(fetch(shard, 8, 8 + length - 1).decode("utf-8"))
        tensors.update({name: spec for name, spec in header.items()
                        if name != "__metadata__"})
    return tensors


# -------------------------------------------------------------------- contract

def classify(name: str) -> str:
    """Map a tensor name onto its family key, with layer indices as `N`."""
    key = re.sub(r"^layers\.\d+\.", "layers.N.", name)
    key = re.sub(r"^mtp\.\d+\.", "mtp.N.", key)
    key = re.sub(r"\.experts\.\d+\.", ".experts.E.", key)
    key = re.sub(r"^vision\.blocks\.\d+\.", "vision.blocks.N.", key)
    return key


# family key -> (dtype, shape builder) where the builder takes the config dict
def build_contract(text: dict) -> dict[str, tuple[list[str], object]]:
    hidden = text["hidden_size"]
    layers = text["num_hidden_layers"]
    heads = text["num_attention_heads"]
    head_dim = text["head_dim"]
    q_rank = text["q_lora_rank"]
    o_groups = text["o_groups"]
    o_rank = text["o_lora_rank"]
    moe = text["moe_intermediate_size"]
    experts = text["n_routed_experts"]
    index_heads = text["index_n_heads"]
    index_dim = text["index_head_dim"]
    rope = text["qk_rope_head_dim"]
    hc = text["hc_mult"]
    hc_rows = hc * (hc + 2)
    ds_experts = text["dspark_n_routed_experts"]

    fp8 = lambda rows, cols: ([rows, cols], "F8_E4M3",
                              [rows // 32, cols // 32], "F8_E8M0")
    bf16 = lambda rows, cols: ([rows, cols], "BF16", None, None)
    bf16_1 = lambda n: ([n], "BF16", None, None)
    f32_1 = lambda n: ([n], "F32", None, None)

    contract: dict[str, tuple] = {}
    contract["attn_norm.weight"] = bf16_1(hidden)
    contract["ffn_norm.weight"] = bf16_1(hidden)
    contract["attn.attn_sink"] = f32_1(heads)
    contract["attn.q_norm.weight"] = bf16_1(q_rank)
    contract["attn.kv_norm.weight"] = bf16_1(head_dim)
    contract["attn.wq_a"] = fp8(q_rank, hidden)
    contract["attn.wq_b"] = fp8(heads * head_dim, q_rank)
    contract["attn.wkv"] = fp8(head_dim, hidden)
    contract["attn.wo_a"] = fp8(o_groups * o_rank, heads * head_dim // o_groups)
    contract["attn.wo_b"] = fp8(hidden, o_groups * o_rank)
    contract["ffn.gate.weight"] = bf16(experts, hidden)
    contract["ffn.gate.bias"] = f32_1(experts)
    contract["ffn.gate.bias_vl"] = f32_1(experts)
    contract["ffn.shared_experts.w1"] = fp8(moe, hidden)
    contract["ffn.shared_experts.w2"] = fp8(hidden, moe)
    contract["ffn.shared_experts.w3"] = fp8(moe, hidden)
    # packed fp4 experts: I8 nibble pairs, per-32 E8M0 scales
    contract["ffn.experts.E.w1"] = ([moe, hidden // 2], "I8", [moe, hidden // 32], "F8_E8M0")
    contract["ffn.experts.E.w2"] = ([hidden, moe // 2], "I8", [hidden, moe // 32], "F8_E8M0")
    contract["ffn.experts.E.w3"] = ([moe, hidden // 2], "I8", [moe, hidden // 32], "F8_E8M0")
    contract["hc_attn_base"] = f32_1(hc_rows)
    contract["hc_attn_fn"] = ([hc_rows, hidden * hc], "F32", None, None)
    contract["hc_attn_scale"] = f32_1(3)
    contract["hc_ffn_base"] = f32_1(hc_rows)
    contract["hc_ffn_fn"] = ([hc_rows, hidden * hc], "F32", None, None)
    contract["hc_ffn_scale"] = f32_1(3)
    # indexer / compressor: only the source layers carry them, see SOURCES
    contract["attn.indexer.wq_b"] = fp8(index_heads * index_dim, q_rank)
    contract["attn.indexer.weights_proj.weight"] = bf16(index_heads, hidden)
    contract["attn.indexer.wk.weight"] = bf16(index_dim, head_dim)
    contract["attn.indexer.k_norm.weight"] = bf16_1(index_dim)
    contract["attn.compressor.wkv.weight"] = bf16(head_dim, hidden)
    contract["attn.compressor.wgate.weight"] = bf16(head_dim, hidden)
    contract["attn.compressor.norm.weight"] = bf16_1(head_dim)
    # engram: [rows, head_dim] fp8 + one E8M0 scale per 32 columns
    contract["engram.q_weight"] = bf16(text["engram_max_ngram_size"], hidden)
    contract["engram.k_weight"] = bf16(text["engram_max_ngram_size"], hidden)
    # Shape taken from the released checkpoint: [5 * hidden, hidden + o_lora_rank].
    # The 5 over hidden is consistent with the n-gram lookbacks folded into one
    # projection; the exact semantics come from the reference Engram class and are
    # confirmed when the engram path lands (docs/deepseek-v41-delta.md).
    contract["engram.wkv"] = fp8(5 * hidden, hidden + o_rank)
    # top level
    contract["embed.weight"] = bf16(text["vocab_size"], hidden)
    contract["head.weight"] = bf16(text["vocab_size"], hidden)
    contract["norm.weight"] = bf16_1(hidden)
    # dspark / mtp stages
    contract["mtp.N.attn_norm.weight"] = contract["attn_norm.weight"]
    contract["mtp.N.ffn_norm.weight"] = contract["ffn_norm.weight"]
    contract["mtp.N.attn.attn_sink"] = contract["attn.attn_sink"]
    contract["mtp.N.attn.q_norm.weight"] = contract["attn.q_norm.weight"]
    contract["mtp.N.attn.kv_norm.weight"] = contract["attn.kv_norm.weight"]
    contract["mtp.N.attn.wq_a"] = contract["attn.wq_a"]
    contract["mtp.N.attn.wq_b"] = contract["attn.wq_b"]
    contract["mtp.N.attn.wkv"] = contract["attn.wkv"]
    contract["mtp.N.attn.wo_a"] = contract["attn.wo_a"]
    contract["mtp.N.attn.wo_b"] = contract["attn.wo_b"]
    contract["mtp.N.ffn.gate.weight"] = bf16(ds_experts, hidden)
    contract["mtp.N.ffn.gate.bias"] = f32_1(ds_experts)
    contract["mtp.N.ffn.gate.bias_vl"] = f32_1(ds_experts)
    contract["mtp.N.ffn.shared_experts.w1"] = contract["ffn.shared_experts.w1"]
    contract["mtp.N.ffn.shared_experts.w2"] = contract["ffn.shared_experts.w2"]
    contract["mtp.N.ffn.shared_experts.w3"] = contract["ffn.shared_experts.w3"]
    contract["mtp.N.ffn.experts.E.w1"] = ([moe, hidden // 2], "I8", [moe, hidden // 32], "F8_E8M0")
    contract["mtp.N.ffn.experts.E.w2"] = ([hidden, moe // 2], "I8", [hidden, moe // 32], "F8_E8M0")
    contract["mtp.N.ffn.experts.E.w3"] = ([moe, hidden // 2], "I8", [moe, hidden // 32], "F8_E8M0")
    contract["mtp.N.hc_attn_base"] = contract["hc_attn_base"]
    contract["mtp.N.hc_attn_fn"] = contract["hc_attn_fn"]
    contract["mtp.N.hc_attn_scale"] = contract["hc_attn_scale"]
    contract["mtp.N.hc_ffn_base"] = contract["hc_ffn_base"]
    contract["mtp.N.hc_ffn_fn"] = contract["hc_ffn_fn"]
    contract["mtp.N.hc_ffn_scale"] = contract["hc_ffn_scale"]
    # DSpark conditioning and head tensors: they live on *different* stages (see
    # DSPARK_STAGE_TENSORS below), which is the part a port gets wrong by assuming
    # three identical draft layers.
    markov_rank = text["dspark_markov_rank"]
    contract["mtp.N.main_norm.weight"] = bf16_1(hidden)
    contract["mtp.N.main_proj"] = fp8(hidden, 3 * hidden)
    contract["mtp.N.norm.weight"] = bf16_1(hidden)
    contract["mtp.N.markov_head.embed.weight"] = bf16(text["vocab_size"], markov_rank)
    contract["mtp.N.markov_head.head.weight"] = bf16(text["vocab_size"], markov_rank)
    contract["mtp.N.confidence_head.proj.weight"] = bf16(1, hidden + markov_rank)
    # vision tower (text-only runs still require the tensors to be present)
    vision = text.get("_vision")
    if vision:
        vhidden, vlayers = vision["hidden_size"], vision["num_hidden_layers"]
        contract["vision.patch_embed.proj.weight"] = ([vhidden, vision["patch_size"] ** 2 * 3], "BF16", None, None)
        contract["vision.patch_embed.proj.bias"] = bf16_1(vhidden)
        contract["vision.norm.weight"] = bf16_1(vhidden)
        contract["vision.blocks.N.norm1.weight"] = bf16_1(vhidden)
        contract["vision.blocks.N.norm2.weight"] = bf16_1(vhidden)
        contract["vision.blocks.N.attn.wqkv.weight"] = bf16(3 * vhidden, vhidden)
        contract["vision.blocks.N.attn.wqkv.bias"] = bf16_1(3 * vhidden)
        contract["vision.blocks.N.attn.wo.weight"] = bf16(vhidden, vhidden)
        contract["vision.blocks.N.attn.wo.bias"] = bf16_1(vhidden)
        contract["vision.blocks.N.mlp.w1.weight"] = bf16(2 * vision["intermediate_size"], vhidden)
        contract["vision.blocks.N.mlp.w2.weight"] = bf16(vhidden, vision["intermediate_size"])
        contract["aligner.w1.weight"] = bf16(hidden, vhidden * vision["downsample_ratio"] ** 2)
        contract["aligner.w1.bias"] = bf16_1(hidden)
        contract["aligner.w2.weight"] = bf16(hidden, hidden)
        contract["aligner.w2.bias"] = bf16_1(hidden)
        for name in ("image_start", "image_end", "image_newline"):
            contract[name] = ([hidden], "BF16", None, None)

    # classify() keeps the layer prefix, so prefix every layer-local key here and
    # leave only the genuinely global tensors alone.
    globals_ = {"embed.weight", "head.weight", "norm.weight", "aligner.w1.weight",
                "aligner.w1.bias", "aligner.w2.weight", "aligner.w2.bias",
                "image_start", "image_end", "image_newline"}
    prefixed: dict[str, tuple] = {}
    for key, expectation in contract.items():
        if key in globals_ or key.startswith(("mtp.", "vision.")):
            prefixed[key] = expectation
        else:
            prefixed["layers.N." + key] = expectation
    return prefixed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, help="checkpoint directory")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--index", type=Path, help="*.safetensors.index.json")
    parser.add_argument("--headers-json", type=Path)
    parser.add_argument("--from-hf", metavar="REPO", help="read shard headers only, over HTTP Range")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not args.model and not (args.config and (args.headers_json or args.from_hf)):
        parser.error("provide --model, or --config with --headers-json/--from-hf")

    config_path = args.config or args.model / "config.json"
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    text = dict(raw.get("text_config") or raw)
    if "vision_config" in raw:
        text["_vision"] = raw["vision_config"]
    index_map = {}
    if args.index:
        index_map = json.loads(args.index.read_text(encoding="utf-8"))["weight_map"]
    elif args.model and (args.model / "model.safetensors.index.json").exists():
        index_map = json.loads((args.model / "model.safetensors.index.json")
                               .read_text(encoding="utf-8"))["weight_map"]

    if args.headers_json:
        tensors = json.loads(args.headers_json.read_text(encoding="utf-8"))
        if all(isinstance(v, dict) and "dtype" not in v for v in tensors.values()):
            tensors = {name: spec for shard in tensors.values() for name, spec in shard.items()}
    elif args.from_hf:
        if index_map:
            shards = sorted(set(index_map.values()))
        else:
            shards = [f"model-{i:05d}-of-00048.safetensors" for i in range(1, 49)]
        tensors = hub_headers(args.from_hf, shards)
    else:
        tensors = local_headers(args.model)

    failures: list[str] = []
    notes: list[str] = []
    contract = build_contract(text)
    layers = text["num_hidden_layers"]
    kv_sources = list(text["kv_source_layer_ids"])
    index_sources = list(text["index_source_layer_ids"])
    ratios = text["compress_ratios"]
    engram_layers = list(text["engram_layer_ids"])
    engram_rows = list(text["engram_num_embeddings"])
    nextn = text["num_nextn_predict_layers"]

    # --- config shape ------------------------------------------------------
    if raw.get("model_type") != "deepseek_v41" and "text_config" not in raw:
        failures.append(f"model_type is not deepseek_v41: {raw.get('model_type')}")
    if "num_hash_layers" in text:
        failures.append("num_hash_layers present: V4.1 dropped token-id hash routing")
    if text.get("scoring_func") != "sqrtsoftplus":
        failures.append(f"scoring_func is not sqrtsoftplus: {text.get('scoring_func')!r}")
    if text.get("topk_method") != "noaux_tc":
        failures.append(f"topk_method is not noaux_tc: {text.get('topk_method')!r}")
    if str((text.get("rope_scaling") or {}).get("rope_type")
           or (text.get("rope_scaling") or {}).get("type")) != "yarn":
        failures.append("rope_scaling is not yarn")
    if len(ratios) < layers + nextn:
        failures.append(f"compress_ratios has {len(ratios)} entries, need >= "
                        f"num_hidden_layers + num_nextn_predict_layers = {layers + nextn}")

    # --- engram tables -----------------------------------------------------
    for position, layer in enumerate(engram_layers):
        rows = engram_rows[position]
        head_dim = text["engram_head_dim"]
        want = {f"layers.{layer}.engram.embed.weight": ([rows, head_dim], "F8_E4M3", None, None),
                f"layers.{layer}.engram.embed.scale": ([rows, head_dim // 32], "F8_E8M0", None, None)}
        for name, spec in want.items():
            actual = tensors.get(name)
            if not actual:
                failures.append(f"missing {name}")
            elif actual["dtype"] != spec[1] or actual["shape"] != spec[0]:
                failures.append(f"invalid {name}: {actual['dtype']} {actual['shape']}, "
                                f"expected {spec[1]} {spec[0]}")
        size = rows * head_dim + rows * (head_dim // 32)
        notes.append(f"engram table layer {layer}: {rows} rows x {head_dim} fp8 "
                     f"= {size / 2**30:.1f} GiB (mmap, never resident)")
    for layer in engram_layers:
        for suffix in ("q_weight", "k_weight"):
            name = f"layers.{layer}.engram.{suffix}"
            want = contract[f"layers.N.engram.{suffix}"]
            actual = tensors.get(name)
            if not actual:
                failures.append(f"missing {name}")
            elif actual["shape"] != want[0]:
                failures.append(f"invalid {name}: {actual['shape']} != {want[0]}")
    for name in sorted(k for k in tensors if "engram" in k):
        actual = tensors[name]
        if classify(name).startswith("layers.N.engram."):
            if int(re.search(r"layers\.(\d+)\.", name).group(1)) not in engram_layers:
                failures.append(f"engram tensor on a layer that is not an engram layer: {name}")

    # --- ownership sets ----------------------------------------------------
    def layers_with(fragment: str) -> list[int]:
        return sorted(int(re.search(r"layers\.(\d+)\.", name).group(1))
                      for name in tensors if fragment in name and name.startswith("layers."))

    if layers_with("attn.compressor.wkv") != sorted(kv_sources):
        failures.append(f"compressor.wkv layers {layers_with('attn.compressor.wkv')} "
                        f"!= kv_source_layer_ids {sorted(kv_sources)}")
    if layers_with("attn.indexer.wk.weight") != sorted(kv_sources):
        failures.append("indexer.wk is not exactly on kv_source_layer_ids: "
                        f"{layers_with('attn.indexer.wk.weight')}")
    if layers_with("attn.indexer.wq_b.weight") != sorted(index_sources):
        failures.append("indexer.wq_b is not exactly on index_source_layer_ids: "
                        f"{layers_with('attn.indexer.wq_b.weight')}")
    expected_gate = sorted(L for L in kv_sources if ratios[L] > 1)
    if layers_with("attn.compressor.wgate") != expected_gate:
        failures.append(f"compressor.wgate layers {layers_with('attn.compressor.wgate')} "
                        f"!= source layers with ratio > 1 {expected_gate}")
    for layer in range(layers):
        for suffix in ("attn.attn_sink", "attn.q_norm.weight", "attn.kv_norm.weight",
                       "attn.wq_a.weight", "attn.wkv.weight", "ffn.gate.bias_vl"):
            if f"layers.{layer}.{suffix}" not in tensors:
                failures.append(f"missing layers.{layer}.{suffix}")
    if ratios[0] or ratios[1]:
        notes.append(f"note: compress_ratios[0..1] = {ratios[0]}, {ratios[1]} "
                     "(pure sliding-window layers)")

    # --- DSpark stage ownership -------------------------------------------
    stage_specials: dict[int, set[str]] = {}
    for name in tensors:
        match = re.match(r"mtp\.(\d+)\.(main_proj|main_norm|norm|markov_head|confidence_head)", name)
        if match:
            stage_specials.setdefault(int(match.group(1)), set()).add(match.group(2))
    if nextn:
        conditioning = {s for s in (0,) if s in stage_specials}
        head_stage = nextn - 1
        if not conditioning:
            failures.append("no DSpark stage carries the target-state conditioning "
                            "(main_proj/main_norm)")
        if "main_proj" not in stage_specials.get(0, set()):
            failures.append("DSpark stage 0 does not carry main_proj/main_norm")
        missing_head = {"markov_head", "confidence_head", "norm"} - stage_specials.get(head_stage, set())
        if missing_head:
            failures.append(f"DSpark stage {head_stage} is missing {sorted(missing_head)}")
        for stage, specials in sorted(stage_specials.items()):
            notes.append(f"DSpark stage {stage}: {sorted(specials)}")
        if len({frozenset(v) for v in stage_specials.values()}) < 2:
            failures.append("DSpark stages look homogeneous: expected stage-specific tensors")

    # --- every tensor classified and matching ------------------------------
    families: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for name in sorted(tensors):
        # two naming conventions meet here: the contract names a quantized family by
        # its base (attn.wq_a) and a plain tensor by its full name (embed.weight), so
        # try the exact key first and only then the weight/scale base.
        if ".engram.embed" in name:
            continue      # validated per engram layer, rows come from the config
        key, actual = classify(name), tensors[name]
        size = actual["data_offsets"][1] - actual["data_offsets"][0]
        expected = contract.get(key)
        if expected is None and key.endswith(".weight") and key[:-len(".weight")] in contract:
            key, expected = key[:-len(".weight")], contract[key[:-len(".weight")]]
        if expected is None and key.endswith(".scale") and key[:-len(".scale")] in contract:
            continue          # validated together with the weight it belongs to
        if expected is None:
            failures.append(f"unclassified tensor: {name}")
            continue
        shape, dtype, scale_shape, scale_dtype = expected
        if actual["shape"] != shape or actual["dtype"] != dtype:
            failures.append(f"invalid {name}: {actual['dtype']} {actual['shape']}, "
                            f"expected {dtype} {shape}")
        if scale_shape is not None:
            base = name[:-len(".weight")] if name.endswith(".weight") else name
            scale = tensors.get(base + ".scale")
            if not scale:
                failures.append(f"missing {base}.scale")
            elif scale["shape"] != scale_shape or scale["dtype"] != scale_dtype:
                failures.append(f"invalid {base}.scale: {scale['dtype']} {scale['shape']}, "
                                f"expected {scale_dtype} {scale_shape}")
        families[key][0] += 1
        families[key][1] += size

    # --- tid2eid must not exist -------------------------------------------
    hash_routing = [name for name in tensors if "tid2eid" in name]
    if hash_routing:
        failures.append(f"token-id hash routing tensors present ({len(hash_routing)}), "
                        "V4.1 has none")

    # --- report ------------------------------------------------------------
    total = sum(spec["data_offsets"][1] - spec["data_offsets"][0] for spec in tensors.values())
    if not args.quiet:
        print(f"V4.1 checkpoint: {len(tensors)} tensors, {total / 2**30:.1f} GiB of weights")
        if index_map and len(index_map) != len(tensors):
            print(f"  index lists {len(index_map)} tensors, headers carry {len(tensors)}")
        by_family = sorted(families.items(), key=lambda item: -item[1][1])[:12]
        for key, (count, size) in by_family:
            print(f"  {size / 2**30:10.2f} GiB  x{count:<6d} {key}")
        for note in notes:
            print(f"  {note}")
        print(f"  layers: {layers}, experts {text['n_routed_experts']} top-{text['num_experts_per_tok']}"
              f", kv sources {kv_sources}, index sources {index_sources}")
        print(f"  engram: layers {engram_layers}, DSpark: {nextn} stages, "
              f"{text['dspark_n_routed_experts']} experts top-{text['dspark_num_experts_per_tok']}")

    if failures:
        print("\nV4.1 checkpoint contract FAILED:")
        for failure in failures[:50]:
            print(f"  - {failure}")
        if len(failures) > 50:
            print(f"  ... and {len(failures) - 50} more")
        return 1
    print("\nV4.1 checkpoint contract OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
