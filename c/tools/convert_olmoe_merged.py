#!/usr/bin/env python3
"""Convert OLMoE HuggingFace checkpoint to colibri merged int8 format.

Consolidates gate_proj, up_proj, and down_proj into a single merged tensor per expert.
This allows olmoe.c to load an expert in a single disk read call instead of 3.

DISK/RAM-SAFE: with --repo, source shards are downloaded and processed ONE AT A
TIME (deleted immediately after extraction) instead of pulling the whole
checkpoint via snapshot_download and loading every shard fully into RAM at
once. Peak extra disk usage is one source shard, not the full checkpoint;
peak extra RAM is bounded by in-flight partial experts (an expert whose three
projections happen to land in different shards), not the whole state dict.
Resumable: existing output shards are scanned (header only) and already-done
tensors are skipped on a rerun.

Usage:
  python tools/convert_olmoe_merged.py --repo allenai/OLMoE-1B-7B-0125-Instruct --out ./olmoe_merged
  python tools/convert_olmoe_merged.py --model ./OLMoE-1B-7B-0125-Instruct --out ./olmoe_merged
"""

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

try:
    import numpy  # noqa: F401  -- safetensors.torch.save_file imports it internally
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
except ImportError as exc:
    sys.exit(f"Missing dependencies: {exc}. "
             f"Install: pip install numpy torch safetensors huggingface_hub")

EXPERT_KEY_RE = re.compile(
    r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight"
)
SKIP_KEY_RE = re.compile(r"\.e_score_correction_bias$|^model\.rotary_emb\.")

TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "vocab.json", "merges.txt", "generation_config.json",
)


def quantize_row(w: "torch.Tensor") -> tuple:
    """Row-wise int8 quantization, identical math to olmoe.c's quantize_rows()."""
    w_f32 = w.float()
    row_max = w_f32.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scales = row_max / 127.0
    q = (w_f32 / scales).round().clamp(-128, 127).to(torch.int8)
    return q, scales.squeeze(1)


def free_gb(path) -> float:
    return shutil.disk_usage(path).free / 1e9


def scan_existing_output(out_dir: Path) -> set:
    """Tensor names already written to a *.safetensors file in out_dir --
    header-only scan, cheap even against many GB of already-converted shards."""
    done = set()
    for shard in sorted(out_dir.glob("model-*.safetensors")):
        with safe_open(str(shard), framework="pt") as f:
            done.update(f.keys())
    return done


class OutputWriter:
    """Buffers converted tensors and flushes into new model-NNNNN.safetensors
    shard files once the buffer gets large -- st.h indexes every *.safetensors
    file in a directory by scanning each one's own header, so any number of
    arbitrarily-named shards works with no index.json needed."""

    def __init__(self, out_dir: Path, flush_every: int = 512):
        self.out_dir = out_dir
        self.flush_every = flush_every
        self.buf = {}
        self.shard_idx = len(list(out_dir.glob("model-*.safetensors")))

    def add(self, name: str, tensor: "torch.Tensor"):
        self.add_many(((name, tensor),))

    def add_many(self, tensors):
        """Add a group of tensors before deciding whether to flush.

        Companion tensors such as an expert's merged weights and row scales
        must be published together so a resumed conversion never mistakes a
        half-written expert for a completed one.
        """
        for name, tensor in tensors:
            self.buf[name] = tensor.contiguous()
        if len(self.buf) >= self.flush_every:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        fn = self.out_dir / f"model-{self.shard_idx:05d}.safetensors"
        tmp = Path(str(fn) + ".tmp")
        tmp.unlink(missing_ok=True)
        try:
            save_file(self.buf, str(tmp))
            os.replace(tmp, fn)
        finally:
            tmp.unlink(missing_ok=True)
        print(f"  wrote {fn.name} ({len(self.buf)} tensors)", flush=True)
        self.buf = {}
        self.shard_idx += 1


def convert_expert(gate, up, down):
    q_gate, s_gate = quantize_row(gate)
    q_up, s_up = quantize_row(up)
    q_down, s_down = quantize_row(down)
    merged_q = torch.cat([q_gate.flatten(), q_up.flatten(), q_down.flatten()])
    merged_s = torch.cat([s_gate, s_up, s_down])
    return merged_q, merged_s


def copy_metadata(src_dir: Path, out_dir: Path):
    shutil.copy2(src_dir / "config.json", out_dir / "config.json")
    for fn in TOKENIZER_FILES:
        p = src_dir / fn
        if p.is_file():
            shutil.copy2(p, out_dir / fn)


# ---------- local / pre-downloaded path ----------

def convert_local(src_dir: Path, out_dir: Path, flush_every: int):
    copy_metadata(src_dir, out_dir)
    shards = sorted(src_dir.glob("*.safetensors"))
    if not shards:
        sys.exit(f"No safetensors found in {src_dir}")

    name_to_shard = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt") as f:
            for name in f.keys():
                name_to_shard[name] = shard

    done = scan_existing_output(out_dir)
    writer = OutputWriter(out_dir, flush_every)
    handles = {}

    def get(name):
        shard = name_to_shard[name]
        h = handles.get(shard)
        if h is None:
            h = safe_open(str(shard), framework="pt")
            handles[shard] = h
        return h.get_tensor(name)

    experts = {}
    dense_names = []
    for name in name_to_shard:
        if SKIP_KEY_RE.search(name):
            continue
        m = EXPERT_KEY_RE.match(name)
        if m:
            layer, expert, proj = int(m.group(1)), int(m.group(2)), m.group(3)
            experts.setdefault((layer, expert), {})[proj] = name
        else:
            dense_names.append(name)

    for name in dense_names:
        if name in done:
            continue
        writer.add(name, get(name))
    writer.flush()

    n_total = len(experts)
    n_done = 0
    for (layer, expert), projs in sorted(experts.items()):
        mw = f"model.layers.{layer}.mlp.experts.{expert}.merged_weight"
        if mw in done:
            n_done += 1
            continue
        if not all(k in projs for k in ("gate_proj", "up_proj", "down_proj")):
            sys.exit(f"Missing projection for layer {layer} expert {expert}: have {list(projs)}")
        gate, up, down = get(projs["gate_proj"]), get(projs["up_proj"]), get(projs["down_proj"])
        merged_q, merged_s = convert_expert(gate, up, down)
        writer.add_many((
            (mw, merged_q),
            (f"model.layers.{layer}.mlp.experts.{expert}.qs", merged_s),
        ))
        n_done += 1
        if n_done % 100 == 0 or n_done == n_total:
            print(f"  {n_done}/{n_total} experts converted...", flush=True)
    writer.flush()
    print(f"Done: {n_total} experts, {len(dense_names)} dense tensors converted.")


# ---------- streaming --repo path: one source shard downloaded/converted/deleted at a time ----------

def convert_streaming(repo: str, out_dir: Path, flush_every: int, min_free_gb: float):
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi().repo_info(repo, files_metadata=True)
    shard_files = sorted(s.rfilename for s in info.siblings if s.rfilename.endswith(".safetensors"))
    if not shard_files:
        sys.exit(f"No .safetensors shards found in {repo}")

    for fn in ("config.json",) + TOKENIZER_FILES:
        try:
            p = hf_hub_download(repo, fn, local_dir=str(out_dir) + "/_meta")
            shutil.copy2(p, out_dir / fn)
        except Exception:
            pass
    if not (out_dir / "config.json").is_file():
        sys.exit(f"config.json could not be fetched from {repo}")

    done = scan_existing_output(out_dir)
    writer = OutputWriter(out_dir, flush_every)
    pending_experts = {}   # (layer, expert) -> {proj: tensor}, freed once a triple completes

    tmp = out_dir / "_inflight"
    tmp.mkdir(exist_ok=True)

    for si, shard_name in enumerate(shard_files):
        if free_gb(out_dir) < min_free_gb:
            print(f"STOP: free space below {min_free_gb} GB. Free space and rerun to resume.")
            break
        print(f"[{si+1}/{len(shard_files)}] downloading {shard_name} "
              f"({free_gb(out_dir):.0f} GB free)...", flush=True)
        local_path = hf_hub_download(repo, shard_name, local_dir=str(tmp))

        with safe_open(local_path, framework="pt") as f:
            for name in f.keys():
                if SKIP_KEY_RE.search(name) or name in done:
                    continue
                m = EXPERT_KEY_RE.match(name)
                if not m:
                    writer.add(name, f.get_tensor(name))
                    continue
                layer, expert, proj = int(m.group(1)), int(m.group(2)), m.group(3)
                key = (layer, expert)
                mw = f"model.layers.{layer}.mlp.experts.{expert}.merged_weight"
                if mw in done:
                    continue
                slot = pending_experts.setdefault(key, {})
                slot[proj] = f.get_tensor(name)
                if all(k in slot for k in ("gate_proj", "up_proj", "down_proj")):
                    merged_q, merged_s = convert_expert(slot["gate_proj"], slot["up_proj"], slot["down_proj"])
                    writer.add_many((
                        (mw, merged_q),
                        (f"model.layers.{layer}.mlp.experts.{expert}.qs", merged_s),
                    ))
                    del pending_experts[key]

        Path(local_path).unlink(missing_ok=True)  # delete the source shard now -- disk-safe
        print(f"    -> extracted+deleted {shard_name} "
              f"({len(pending_experts)} experts still incomplete)", flush=True)

    writer.flush()
    shutil.rmtree(tmp, ignore_errors=True)
    if pending_experts:
        print(f"WARNING: {len(pending_experts)} experts never completed all 3 projections -- rerun to resume.")
    else:
        print("DONE.")


def main():
    ap = argparse.ArgumentParser(description="Convert OLMoE HF checkpoint -> colibri merged int8")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--repo", help="HuggingFace repo ID (streams+deletes one shard at a time)")
    src.add_argument("--model", help="Local HF checkpoint directory (already fully downloaded)")
    ap.add_argument("--out", required=True, help="Output directory for merged model")
    ap.add_argument("--flush-every", type=int, default=512,
                     help="flush an output shard every N converted tensors")
    ap.add_argument("--min-free-gb", type=float, default=5.0,
                     help="--repo only: stop (resumably) if free space drops below this")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model:
        convert_local(Path(args.model), out_dir, args.flush_every)
    else:
        convert_streaming(args.repo, out_dir, args.flush_every, args.min_free_gb)


if __name__ == "__main__":
    main()
