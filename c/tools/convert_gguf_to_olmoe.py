#!/usr/bin/env python3
"""Convert a GGUF OLMoE checkpoint into a colibri container for c/olmoe.c.

Flow (one user-facing entry point):
    wrapper (this file) -> gguf_reader (structure) -> gguf_dequant (numbers)
                        -> gguf_olmoe_profile (OLMoE mapping/layout/quant)
                        -> safetensors shards + config.json + tokenizer.json

Output, per tensor, matches what c/olmoe.c loads:
  * dense tensors dequantized to F16 (1-D norms kept F32), names remapped;
  * routed experts dequantized and re-quantized to int8 row-wise, merged into
    model.layers.N.mlp.experts.E.merged_weight (I8) + .qs (F32);
  * config.json reconstructed from the GGUF olmoe.* metadata. GGUF does not
    carry norm_topk_prob; the default is false (matching the official OLMoE
    checkpoints) and --norm-topk-prob overrides it. q/k are taken as stored
    (HuggingFace layout); --qk-permuted undoes llama.cpp's RoPE permutation for
    GGUFs that carry it;
  * tokenizer.json rebuilt from tokenizer.ggml.* (byte-level GPT-2), validated
    token-for-token against the official OLMoE tokenizer, so the container is
    usable for chat/serve, not only loadable for validation.

Safety:
  * refuses a non-OLMoE GGUF, unknown tensor names, and missing required tensors;
  * refuses to touch an --output that already contains a container unless the user
    passes --overwrite (rebuild from scratch); it never silently skips work;
  * writes shards atomically (tmp + rename), so a crash leaves no half shard;
  * --remove-source-file deletes the source GGUF ONLY after a fully successful
    conversion (never on a partial run);
  * --dry-run prints the plan without writing anything or creating the output
    directory.

Usage:
  python tools/convert_gguf_to_olmoe.py \\
      --input  /path/to/model.gguf      # or a dir holding exactly one .gguf
      --output /path/to/olmoe-colibri   # created if missing
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from gguf_reader import GGUFReader
from gguf_dequant import dequantize
import gguf_olmoe_profile as profile

try:
    from safetensors.numpy import save_file
except ImportError as exc:
    sys.exit("Missing dependency: %s. Install: pip install numpy safetensors" % exc)


class ConversionError(Exception):
    pass


def resolve_sources(model):
    path = Path(model).expanduser()
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix == ".gguf")
        if not files:
            raise ConversionError("no .gguf files in %s" % path)
        if len(files) > 1:
            raise ConversionError(
                "%s holds %d .gguf files; OLMoE is converted from a single file "
                "for now, pass one explicitly" % (path, len(files)))
        return [files[0]]
    if not path.is_file():
        raise ConversionError("model not found: %s" % path)
    return [path]


def required_names(config):
    names = {
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.norm.weight",
    }
    for layer in range(config["num_hidden_layers"]):
        base = "model.layers.%d" % layer
        names.update({
            base + ".input_layernorm.weight",
            base + ".post_attention_layernorm.weight",
            base + ".self_attn.q_proj.weight",
            base + ".self_attn.k_proj.weight",
            base + ".self_attn.v_proj.weight",
            base + ".self_attn.o_proj.weight",
            base + ".self_attn.q_norm.weight",
            base + ".self_attn.k_norm.weight",
            base + ".mlp.gate.weight",
        })
    return names


def plan_tensors(reader):
    """Return (dense, experts, unmapped) planned from the GGUF tensor list."""
    dense = []
    experts = {}
    unmapped = []
    for tensor in reader.tensors:
        name = tensor["name"]
        parsed = profile.parse_expert(name)
        if parsed is not None:
            layer, kind = parsed
            experts.setdefault(layer, {})[kind] = tensor
            continue
        target = profile.dense_target(name)
        if target is None:
            unmapped.append(name)
            continue
        dense.append((target, tensor))
    return dense, experts, unmapped


class OutputWriter:
    def __init__(self, out_dir, flush_every):
        self.out_dir = out_dir
        self.flush_every = flush_every
        self.buf = {}
        self.shard_idx = 0

    def add_many(self, items):
        """Buffer related tensors and flush once the buffer reaches flush_every.

        Contract: every tensor of one logical unit -- e.g. an expert's int8
        merged_weight AND its f32 .qs -- is passed in a single call, so a flush
        never splits the pair across shards during a normal run.
        """
        for name, array in items:
            self.buf[name] = np.ascontiguousarray(array)
        if len(self.buf) >= self.flush_every:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        destination = self.out_dir / ("model-%05d.safetensors" % self.shard_idx)
        temporary = Path(str(destination) + ".tmp")
        temporary.unlink(missing_ok=True)
        try:
            save_file(self.buf, str(temporary))
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        print("  wrote %s (%d tensors)" % (destination.name, len(self.buf)), flush=True)
        self.buf = {}
        self.shard_idx += 1


def _write_json_atomic(path, payload):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def _dense_to_storage(logical):
    if logical.ndim == 1:
        return logical.astype(np.float32)
    return logical.astype(np.float16)


def convert_dense(reader, dense, writer, config, qk_permuted):
    for target, tensor in dense:
        raw = reader.read_tensor(tensor)
        flat = dequantize(raw, tensor["ggml_type"], tensor["numel"])
        logical = profile.to_logical(flat, tensor["dims"])
        if qk_permuted:
            # Some GGUF producers (recent llama.cpp olmo.py) RoPE-permute q/k;
            # c/olmoe.c uses the plain HuggingFace layout, so undo it.
            name = tensor["name"]
            if name.endswith("attn_q.weight"):
                logical = profile.restore_rope_layout(logical, config["num_attention_heads"],
                                                      config["num_attention_heads"])
            elif name.endswith("attn_k.weight"):
                logical = profile.restore_rope_layout(logical, config["num_attention_heads"],
                                                      config["num_key_value_heads"])
        writer.add_many(((target, _dense_to_storage(logical)),))


def convert_experts(reader, experts, writer, config):
    layer_count = config["num_hidden_layers"]
    expert_count = config["num_experts"]
    hidden = config["hidden_size"]
    inter = config["intermediate_size"]
    want_w = inter * hidden + inter * hidden + hidden * inter
    want_s = inter + inter + hidden
    total = layer_count * expert_count
    done_experts = 0

    for layer in range(layer_count):
        kinds = experts.get(layer)
        if not kinds or set(kinds) != {"gate", "up", "down"}:
            raise ConversionError("layer %d: incomplete expert trio: %s"
                                  % (layer, sorted(kinds) if kinds else None))
        raw_bytes, expert_layout = {}, {}
        for kind in ("gate", "up", "down"):
            tensor = kinds[kind]
            raw_bytes[kind] = reader.read_tensor(tensor)
            expert_layout[kind] = (profile.expert_matrix_shape(tensor["dims"]),
                                   tensor["ggml_type"],
                                   profile.expert_bytes_per_slice(tensor["dims"], tensor["ggml_type"]))
        for expert in range(expert_count):
            merged_name = "model.layers.%d.mlp.experts.%d.merged_weight" % (layer, expert)
            qs_name = "model.layers.%d.mlp.experts.%d.qs" % (layer, expert)
            per_kind = {}
            for kind in ("gate", "up", "down"):
                expert_shape, ggml_type, chunk = expert_layout[kind]
                start = expert * chunk
                flat = dequantize(raw_bytes[kind][start:start + chunk], ggml_type,
                                  expert_shape[0] * expert_shape[1])
                per_kind[kind] = flat.reshape(expert_shape)
            merged, scales = profile.merge_expert(per_kind["gate"], per_kind["up"], per_kind["down"])
            if merged.size != want_w or scales.size != want_s:
                raise ConversionError(
                    "expert [%d,%d] produced %d bytes / %d scales, expected %d / %d"
                    % (layer, expert, merged.size, scales.size, want_w, want_s))
            # The int8 weights and their f32 scales are ONE logical unit: one
            # add_many call keeps them in the same shard (see OutputWriter).
            writer.add_many(((merged_name, merged), (qs_name, scales)))
            done_experts += 1
        print("  layer %d/%d experts done (%d/%d)"
              % (layer + 1, layer_count, done_experts, total), flush=True)


def _format_duration(seconds):
    seconds = int(round(seconds))
    if seconds < 60:
        return "%ds" % seconds
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return "%dm %02ds" % (minutes, secs)
    hours, minutes = divmod(minutes, 60)
    return "%dh %02dm %02ds" % (hours, minutes, secs)


def run(args):
    started = time.perf_counter()
    sources = resolve_sources(args.input)
    source = sources[0]
    out_dir = Path(args.output).expanduser()

    print("colibri: converting a GGUF OLMoE checkpoint into a colibri container", flush=True)
    print("  flow:   %s" % source.name, flush=True)
    print("          -> gguf_reader (header/metadata/tensor index)", flush=True)
    print("          -> gguf_dequant + gguf_olmoe_profile (dequant, layout, int8 experts)", flush=True)
    print("          -> safetensors shards + config.json + tokenizer.json", flush=True)
    print("  input:  %s" % source, flush=True)
    print("  output: %s" % out_dir, flush=True)

    reader = GGUFReader(str(source))
    try:
        arch = reader.metadata.get("general.architecture")
        if arch != "olmoe":
            raise ConversionError("general.architecture is %r, expected 'olmoe'" % arch)
        config = profile.build_config(reader.metadata,
                                      norm_topk_prob=args.norm_topk_prob)
        dense, experts, unmapped = plan_tensors(reader)
        if unmapped:
            raise ConversionError("unrecognized tensors: %s" % ", ".join(sorted(unmapped)[:8]))

        present = {target for target, _ in dense}
        for layer, kinds in experts.items():
            for expert in range(config["num_experts"]):
                present.add("model.layers.%d.mlp.experts.%d.merged_weight" % (layer, expert))
        missing = sorted(required_names(config) - present)
        if missing:
            raise ConversionError("missing required tensors: %s" % ", ".join(missing[:8]))

        experts_total = config["num_hidden_layers"] * config["num_experts"]
        if args.dry_run:
            print("dry-run: %s" % source)
            print("  architecture: olmoe, %d layers, %d experts (top-%d), hidden %d, inter %d, vocab %d"
                  % (config["num_hidden_layers"], config["num_experts"],
                     config["num_experts_per_tok"], config["hidden_size"],
                     config["intermediate_size"], config["vocab_size"]))
            print("  dense tensors: %d" % len(dense))
            print("  expert tensors: %d (%d merged_weight + %d qs)"
                  % (experts_total * 2, experts_total, experts_total))
            if out_dir.exists() and any(out_dir.glob("model-*.safetensors")):
                print("  note: output already contains a container; --overwrite rebuilds it")
            return 0

        existing_shards = sorted(out_dir.glob("model-*.safetensors"))
        if existing_shards and not args.overwrite:
            raise ConversionError(
                "output already contains a converted container (%d safetensors shard(s)):\n"
                "  %s\n"
                "refusing to overwrite it. Pass --overwrite to rebuild it from scratch."
                % (len(existing_shards), out_dir))

        out_dir.mkdir(parents=True, exist_ok=True)
        if args.overwrite:
            for path in list(out_dir.glob("model-*.safetensors")):
                path.unlink()
            for name in ("config.json", "tokenizer.json"):
                sidecar = out_dir / name
                if sidecar.is_file():
                    sidecar.unlink()
            print("  --overwrite: cleared previous container outputs", flush=True)
        writer = OutputWriter(out_dir, args.flush_every)
        _write_json_atomic(out_dir / "config.json", config)
        print("  wrote config.json", flush=True)
        tokenizer = profile.build_tokenizer(reader.metadata)
        _write_json_atomic(out_dir / "tokenizer.json", tokenizer)
        print("  wrote tokenizer.json (%d vocab, %d merges, %d added)"
              % (len(tokenizer["model"]["vocab"]),
                 len(tokenizer["model"].get("merges", [])),
                 len(tokenizer["added_tokens"])), flush=True)
        convert_dense(reader, dense, writer, config, args.qk_permuted)
        convert_experts(reader, experts, writer, config)
        writer.flush()
    finally:
        reader.close()

    if args.remove_source_file:
        os.remove(source)
        print("  removed source %s" % source, flush=True)
    print("Done:")
    print("- path: %s" % out_dir)
    print("- time: %s" % _format_duration(time.perf_counter() - started))
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="convert_gguf_to_olmoe.py",
        description="Convert an OLMoE GGUF checkpoint into a colibri container "
                    "readable by the c/olmoe.c engine (safetensors shards + "
                    "config.json + tokenizer.json).",
        epilog=(
            "examples:\n"
            "  convert_gguf_to_olmoe.py --input model.gguf --output ./olmoe-colibri\n"
            "  convert_gguf_to_olmoe.py --input ./gguf-dir --output ~/llm/olmoe --dry-run\n"
            "  convert_gguf_to_olmoe.py --input model.gguf --output ./olmoe --remove-source-file\n"
            "\n"
            "notes:\n"
            "  --input may be a single .gguf file or a directory holding exactly one\n"
            "    .gguf (more than one is refused: OLMoE is converted from one file).\n"
            "  --output is created (with any missing parent) if it does not exist.\n"
            "  If --output already holds a container, the run REFUSES with a message;\n"
            "    pass --overwrite to rebuild it from scratch.\n"
            "  --remove-source-file deletes the input .gguf only after the whole\n"
            "    conversion finished successfully; it never removes on a partial run.\n"
            "  --dry-run prints the plan, writes nothing and creates no directory.\n"
            "  The engine can load the result for validation; chat/serve work because\n"
            "    tokenizer.json is reconstructed from the GGUF tokenizer metadata.\n"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, metavar="PATH",
                        help="source GGUF file, or a directory holding exactly one .gguf")
    parser.add_argument("--output", required=True, metavar="DIR",
                        help="destination directory for the colibri container "
                             "(created, including missing parents, if absent)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the conversion plan (layers, experts, output dir) "
                             "and exit without writing anything or creating --output")
    parser.add_argument("--remove-source-file", action="store_true",
                        help="after a fully successful conversion, delete the source "
                             ".gguf (never on a partial or failed run)")
    parser.add_argument("--overwrite", action="store_true",
                        help="rebuild an existing container in --output from scratch. "
                             "Without it, the run refuses when --output already holds "
                             "model-*.safetensors")
    parser.add_argument("--norm-topk-prob", action="store_true",
                        help="set config.json norm_topk_prob=true (renormalize top-k router "
                             "weights). GGUF does not carry this field; the default is false, "
                             "which matches the official OLMoE checkpoints")
    parser.add_argument("--qk-permuted", action="store_true",
                        help="undo llama.cpp's RoPE permutation of q_proj/k_proj. Off by "
                             "default because the official OLMoE GGUF is already in the "
                             "HuggingFace layout; use it only for GGUFs that were permuted")
    parser.add_argument("--flush-every", type=int, default=512, metavar="N",
                        help="flush an output safetensors shard every N tensors "
                             "(default: 512; smaller = more, smaller shards)")
    args = parser.parse_args()
    try:
        return run(args)
    except ConversionError as exc:
        sys.stderr.write("ERROR: %s\n" % exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
