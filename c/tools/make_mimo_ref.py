#!/usr/bin/env python3
"""Produce ref.json for the tiny MiMo-V2.6 fixture from Xiaomi's own code.

Maintainer tool, run offline once per fixture change; CI never executes it
(tests/mimo_tiny_harness.py only reads the committed ref.json).  The reference
tokens come from the vendor's published implementation (modeling_mimo_v2.py on
the MiMo-V2.6-Flash-MOPD Hugging Face repo, pinned below by SHA-256), so the
engine and its oracle never share an author.

Parity is by construction:
- the engine is tested with its dense weights at f32 (ref.json engine_env) and
  float expert matmuls, and this script dequantizes THE SAME BYTES into the
  vendor model's f32 Linear weights: BF16 by widening, FP8 e4m3 through torch's
  own float8_e4m3fn type times the 128x128 block scale, MXFP4 as
  e2m1 * 2^(scale-127) with the low nibble on the even column;
- the fused qkv_proj is stored pre-sharded ([Q_0|K_0|V_0|Q_1|...], scale grid
  tiled per chunk) as on the release, while the vendor module wants [Q|K|V]:
  this script de-interleaves it with the layout vLLM documents for the release
  (vllm/model_executor/models/mimo_v2.py, _shard_fp8_qkv_proj), so a reader
  that gets the layout wrong disagrees with the oracle;
- token-level argmax must then agree wherever the logit margin exceeds float
  noise; the generator engineers wide margins and this script VERIFIES them per
  emitted position (--min-margin, refuse on violation).

Requires: torch (CPU is fine), transformers >= 5.3, numpy.  Network only if the
vendor files are not already in --vendor-dir.
"""

import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np

VENDOR_REPO = "XiaomiMiMo/MiMo-V2.6-Flash-MOPD"
VENDOR_REVISION = "2479e2d0029eca9a34cc7e7f55a121925f81908e"
VENDOR_FILES = {
    "configuration_mimo_v2.py":
        "773062ac9850b908eb54751b3e4dbe00e653c0e80595599409e97d0c1af2ce3e",
    "modeling_mimo_v2.py":
        "a8c3cb3aae473bcc15f023010547c919f15eba6546e6ed7efb61a8937b12f3ad",
}
BLOCK = 128

MX4_LUT = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                   dtype=np.float32)


def read_fixture(fixture):
    """{name: (dtype string, raw numpy array)} across the indexed shards."""
    index = json.loads((fixture / "model.safetensors.index.json").read_text())
    raw_dtypes = {"F32": np.float32, "U8": np.uint8, "BF16": np.uint16,
                  "F8_E4M3": np.uint8}
    out = {}
    for shard in sorted(set(index["weight_map"].values())):
        with open(fixture / shard, "rb") as fh:
            header_len = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(header_len))
            raw = fh.read()
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            lo, hi = meta["data_offsets"]
            dtype = raw_dtypes[meta["dtype"]]
            arr = np.frombuffer(raw, dtype=dtype,
                                count=(hi - lo) // np.dtype(dtype).itemsize,
                                offset=lo).reshape(meta["shape"]).copy()
            out[name] = (meta["dtype"], arr)
    return out


def bf16_to_f32(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def fp8_to_f32(codes):
    """e4m3fn decode by torch's own float8 type (not colibri's decoder)."""
    import torch
    return torch.from_numpy(codes).view(torch.float8_e4m3fn).float().numpy()


def mxfp4_dequant(packed, scale):
    rows, half = packed.shape
    cols = half * 2
    codes = np.empty((rows, cols), dtype=np.uint8)
    codes[:, 0::2] = packed & 0x0F
    codes[:, 1::2] = packed >> 4
    values = MX4_LUT[codes]
    exponents = np.ldexp(np.float32(1.0), scale.astype(np.int32) - 127)
    return (values.reshape(rows, cols // 32, 32) *
            exponents[:, :, None]).reshape(rows, cols).astype(np.float32)


def fp8_block_dequant(codes, scale_inv, row_scale_index=None):
    """w[r, c] = fp8(codes[r, c]) * scale_inv[row_block(r), c // 128]."""
    w = fp8_to_f32(codes)
    rows, cols = w.shape
    if row_scale_index is None:
        row_scale_index = np.arange(rows) // BLOCK
    col_block = np.arange(cols) // BLOCK
    return (w * scale_inv[row_scale_index][:, col_block]).astype(np.float32)


def qkv_contiguous(codes, scale_inv, heads, kv_heads, head_dim, v_head_dim,
                   chunks):
    """Pre-sharded fused qkv (vLLM's documented layout) -> [Q|K|V] f32."""
    rows = codes.shape[0]
    per_chunk = rows // chunks
    q_rows = (heads // chunks) * head_dim
    k_rows = (kv_heads // chunks) * head_dim
    v_rows = (kv_heads // chunks) * v_head_dim
    assert q_rows + k_rows + v_rows == per_chunk, (rows, chunks)
    chunk_scale_rows = -(-per_chunk // BLOCK)
    r = np.arange(rows)
    if scale_inv.shape[0] == chunks * chunk_scale_rows:
        index = (r // per_chunk) * chunk_scale_rows + (r % per_chunk) // BLOCK
    elif scale_inv.shape[0] == -(-rows // BLOCK):
        index = r // BLOCK
    else:
        raise SystemExit(f"qkv scale grid {scale_inv.shape} fits neither layout")
    w = fp8_block_dequant(codes, scale_inv, index)
    parts = [w[c * per_chunk:(c + 1) * per_chunk] for c in range(chunks)]
    q = np.concatenate([p[:q_rows] for p in parts])
    k = np.concatenate([p[q_rows:q_rows + k_rows] for p in parts])
    v = np.concatenate([p[q_rows + k_rows:] for p in parts])
    return np.concatenate([q, k, v]).astype(np.float32)


def vision_state(tensors, model):
    """visual.* widened to f32, plus the zeros the release does not store: the
    merger's two Linear biases and its LayerNorm bias (vLLM builds that merger
    with bias=False; the HF module declares them and would otherwise keep its
    random init)."""
    import torch
    state = {}
    for name, (dtype, arr) in tensors.items():
        if name.startswith("visual."):
            state[name] = torch.from_numpy(bf16_to_f32(arr) if dtype == "BF16"
                                           else arr.astype(np.float32))
    for name, param in model.named_parameters():
        if name.startswith("visual.merger.") and name.endswith(".bias") \
                and name not in state:
            state[name] = torch.zeros_like(param)
    return state


def to_state_dict(tensors, config):
    import torch
    chunks = config["num_key_value_heads"]
    state = {}
    consumed = set()
    for name, (dtype, arr) in tensors.items():
        if name in consumed:
            continue
        if name.endswith(".weight_scale_inv") or name.endswith(".weight_scale"):
            continue
        if name.startswith("visual."):
            continue
        if dtype == "BF16":
            value = bf16_to_f32(arr)
        elif dtype == "F32":
            value = arr.astype(np.float32)
        elif name.endswith("qkv_proj.weight"):
            layer = int(name.split(".")[2])
            swa = config["hybrid_layer_pattern"][layer] == 1
            heads = config["swa_num_attention_heads" if swa else "num_attention_heads"]
            kv = config["swa_num_key_value_heads" if swa else "num_key_value_heads"]
            value = qkv_contiguous(arr, tensors[name + "_scale_inv"][1], heads,
                                   kv, config["head_dim"], config["v_head_dim"],
                                   chunks)
        elif dtype == "F8_E4M3":
            value = fp8_block_dequant(arr, tensors[name + "_scale_inv"][1])
        elif dtype == "U8" and ".experts." in name:
            value = mxfp4_dequant(arr, tensors[name + "_scale"][1])
        else:
            raise SystemExit(f"{name}: unexpected {dtype}")
        state[name] = torch.from_numpy(np.ascontiguousarray(value))
    return state


def fetch_vendor(vendor_dir):
    vendor_dir.mkdir(parents=True, exist_ok=True)
    for name, expected in VENDOR_FILES.items():
        path = vendor_dir / name
        if not path.exists():
            url = (f"https://huggingface.co/{VENDOR_REPO}/raw/"
                   f"{VENDOR_REVISION}/{name}")
            print(f"[ref] downloading {url}", file=sys.stderr)
            with urllib.request.urlopen(url, timeout=60) as resp:
                path.write_bytes(resp.read())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise SystemExit(
                f"{name}: sha256 {digest} does not match the pinned {expected}; "
                f"the vendor file changed. Review it, then update VENDOR_FILES.")


def load_vendor(vendor_dir):
    """The vendor files as a synthetic package, so the relative import of the
    configuration module works whatever the directory is called."""
    pkg = "mimo_v2_vendor"
    spec = importlib.machinery.ModuleSpec(pkg, None, is_package=True)
    package = importlib.util.module_from_spec(spec)
    package.__path__ = [str(vendor_dir)]
    sys.modules[pkg] = package

    def load(stem):
        file_spec = importlib.util.spec_from_file_location(
            f"{pkg}.{stem}", vendor_dir / f"{stem}.py")
        module = importlib.util.module_from_spec(file_spec)
        sys.modules[f"{pkg}.{stem}"] = module
        file_spec.loader.exec_module(module)
        return module

    config_module = load("configuration_mimo_v2")
    modeling = load("modeling_mimo_v2")
    shim_mask_builders(modeling)
    return config_module, modeling


def shim_mask_builders(modeling):
    """The vendor file targets transformers 5.3, whose mask builders take
    `input_embeds` and `cache_position`; later releases renamed the first
    (`inputs_embeds`, 5.6 dropped the old name) and derive the second
    themselves. Rename, drop what the installed builder no longer takes, and
    nothing else: the masks, and every number, stay the library's own. The
    reference runs without a cache from position 0, where the derived positions
    are the passed ones; the committed ref.json is identical under 5.3 and 5.12."""
    import inspect
    for name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        builder = getattr(modeling, name)
        params = inspect.signature(builder).parameters
        if "input_embeds" in params:
            continue

        def adapted(*args, _builder=builder, _params=params, **kwargs):
            if "input_embeds" in kwargs:
                kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
            if not any(p.kind == p.VAR_KEYWORD for p in _params.values()):
                kwargs = {k: v for k, v in kwargs.items() if k in _params}
            return _builder(*args, **kwargs)
        setattr(modeling, name, adapted)


def build_model(config_module, modeling, config):
    import torch
    cfg = config_module.MiMoV2Config(**dict(config, use_cache=False))
    cfg._attn_implementation = "eager"   # the sink path is eager-only
    torch.manual_seed(0)                 # every weight is overwritten below
    model = modeling.MiMoV2ForCausalLM(cfg).float().eval()
    return model


def check_rope(model, config):
    """The two rotary tables must be what the config says (theta 1e7 on full
    layers, 1e4 on SWA layers, over int(head_dim * 0.334) dims): refuse a
    transformers version that silently standardizes them away."""
    dim = int(config["head_dim"] * config["partial_rotary_factor"])
    for module, theta in ((model.model.rotary_emb, config["rope_theta"]),
                          (model.model.swa_rotary_emb, config["swa_rope_theta"])):
        want = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
        got = module.inv_freq.double().numpy()
        if got.shape != want.shape or not np.allclose(got, want, rtol=1e-6):
            raise SystemExit(f"rotary table {got} is not theta={theta} over "
                             f"{dim} dims: this transformers version changes "
                             f"the vendor's RoPE")


class RouterWatch:
    """Records, for every token at every MoE layer, the gap between the k-th
    and (k+1)-th biased router score: a near-tie there lets float noise swap an
    expert, which no logit margin downstream can protect against."""

    def __init__(self, model):
        self.gaps = []
        for layer in model.model.layers:
            gate = getattr(layer.mlp, "gate", None)
            if gate is not None:
                gate.register_forward_hook(self.hook)

    def hook(self, gate, inputs, _output):
        import torch
        x = inputs[0].reshape(-1, inputs[0].shape[-1]).float()
        scores = torch.sigmoid(x @ gate.weight.float().T)
        biased = scores + gate.e_score_correction_bias.float()
        top = torch.topk(biased, gate.top_k + 1, dim=-1).values
        self.gaps.extend((top[:, -2] - top[:, -1]).tolist())


def logits_for(model, ids):
    import torch
    with torch.no_grad():
        return model(input_ids=torch.tensor([ids], dtype=torch.long),
                     use_cache=False).logits[0]


def case_reference(model, prompt, max_new, min_margin):
    import torch
    sequence = list(prompt)
    margins = []
    for _ in range(max_new):
        logits = logits_for(model, sequence)[-1]
        top2 = torch.topk(logits, 2).values
        margins.append(float(top2[0] - top2[1]))
        sequence.append(int(logits.argmax()))
    full = logits_for(model, sequence)
    teacher = full.argmax(-1).tolist()
    # context-free control: the head applied to the embedding alone
    with torch.no_grad():
        emb = model.model.embed_tokens(torch.tensor(sequence))
        normed = model.model.norm(emb)
        alone = model.lm_head(normed).argmax(-1).tolist()
    context_free = sum(a == b for a, b in zip(alone, teacher)) / len(teacher)
    tf_top2 = torch.topk(full, 2, dim=-1).values
    tf_margins = (tf_top2[:, 0] - tf_top2[:, 1]).tolist()
    worst = min(margins + tf_margins)
    if worst < min_margin:
        raise SystemExit(
            f"logit margin {worst:.4f} below --min-margin {min_margin}: the "
            f"fixture rides a near-tie; re-tune the generator instead of "
            f"lowering the margin.")
    return {
        "prompt_ids": list(prompt),
        "teacher_forcing_ids": teacher,
        "greedy_full_ids": sequence,
        "greedy_new_ids": sequence[len(prompt):],
        "max_new_tokens": max_new,
        "min_logit_margin": round(worst, 4),
        "context_free_agreement": round(context_free, 3),
    }


def vision_reference(model, fixture, config, max_new, min_margin):
    """The official Qwen2-VL processor on the fixture's picture must give the
    fixture's patches bit for bit (the gateway's preprocessing); then Xiaomi's
    ViT and language model answer a prompt that carries the picture."""
    import torch
    from PIL import Image
    from transformers import Qwen2VLImageProcessor
    processor = Qwen2VLImageProcessor.from_pretrained(str(fixture))
    image = Image.open(fixture / "image.png").convert("RGB")
    batch = processor(images=[image], return_tensors="pt")
    official = batch["pixel_values"].float().numpy()
    thw = batch["image_grid_thw"]
    grid = json.loads((fixture / "image.json").read_text())
    ours = np.fromfile(fixture / "patches.f32", dtype=np.float32).reshape(official.shape)
    if thw.tolist() != [[1, grid["grid_h"], grid["grid_w"]]]:
        raise SystemExit(f"grid {thw.tolist()} != fixture {grid}")
    diff = float(np.abs(official - ours).max())
    if diff > 1e-6:
        raise SystemExit(f"gateway patches differ from the official processor by {diff}")
    n_img = grid["tokens"]
    prompt = ([5, 7, config["vision_start_token_id"]] + [config["image_token_id"]] * n_img
              + [config["vision_end_token_id"], 11, 13, 17, 19])
    pixels = torch.from_numpy(official)

    def logits_img(ids, pv):
        with torch.no_grad():
            return model(input_ids=torch.tensor([ids]), pixel_values=pv,
                         image_grid_thw=thw, use_cache=False).logits[0]

    def greedy(pv):
        seq, margins = list(prompt), []
        for _ in range(max_new):
            lg = logits_img(seq, pv)[-1]
            top2 = torch.topk(lg, 2).values
            margins.append(float(top2[0] - top2[1]))
            seq.append(int(lg.argmax()))
        return seq, margins

    seq, margins = greedy(pixels)
    full = logits_img(seq, pixels)
    tf_top2 = torch.topk(full, 2, dim=-1).values
    worst = min(margins + (tf_top2[:, 0] - tf_top2[:, 1]).tolist())
    if worst < min_margin:
        raise SystemExit(f"vision case: logit margin {worst:.4f} below {min_margin}")
    other, _ = greedy(-pixels)
    if other[len(prompt):] == seq[len(prompt):]:
        raise SystemExit("the answer does not depend on the picture: re-tune the tower")
    with torch.no_grad():
        rows = model.visual(pixel_values=pixels, grid_thw=thw).float().numpy()
    return {
        "prompt_ids": prompt,
        "grid_h": grid["grid_h"], "grid_w": grid["grid_w"],
        "patches_sha256": hashlib.sha256(ours.tobytes()).hexdigest(),
        "patch_max_diff_vs_official_processor": diff,
        "vision_rows_sum": float(rows.sum()),
        "vision_rows_abs_sum": float(np.abs(rows).sum()),
        "teacher_forcing_ids": full.argmax(-1).tolist(),
        "greedy_full_ids": seq,
        "greedy_new_ids": seq[len(prompt):],
        "max_new_tokens": max_new,
        "min_logit_margin": round(worst, 4),
        "negated_image_new_ids": other[len(prompt):],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fixture", default="./mimo_tiny")
    parser.add_argument("--vendor-dir", default=None,
                        help="default: <fixture>/.vendor_mimo_v2")
    parser.add_argument("--output", default=None,
                        help="default: <fixture>/ref.json")
    parser.add_argument("--min-margin", type=float, default=0.05)
    parser.add_argument(
        "--min-router-gap", type=float, default=1e-5,
        help="router scores are f32 dot products over the hidden size; engine "
             "and torch sum them in different orders, about 1e-6 apart")
    parser.add_argument("--max-context-free", type=float, default=0.5)
    args = parser.parse_args()

    fixture = Path(args.fixture)
    vendor_dir = (Path(args.vendor_dir) if args.vendor_dir
                  else fixture / ".vendor_mimo_v2")
    output = Path(args.output) if args.output else fixture / "ref.json"
    config = json.loads((fixture / "config.json").read_text())

    fetch_vendor(vendor_dir)
    config_module, modeling = load_vendor(vendor_dir)
    model = build_model(config_module, modeling, config)
    check_rope(model, config)
    state = to_state_dict(read_fixture(fixture), config)
    if "vision_config" in config:
        state.update(vision_state(read_fixture(fixture), model))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"state dict mismatch:\n  missing: {missing}\n"
                         f"  unexpected: {unexpected}")

    eos = config["eos_token_id"]
    vocab = config["vocab_size"]
    cases = {
        # every case outgrows the 8-token window; "long" crosses a 64-token
        # prefill chunk too
        "short": ([5, 7, 9, 11, 13, 17, 19, 23], 12),
        "window": ([(5 + i * 7) % vocab for i in range(40)], 6),
        "long": ([(5 + i * 11) % vocab for i in range(72)], 6),
    }
    watch = RouterWatch(model)
    reference = {}
    for name, (prompt, max_new) in cases.items():
        prompt = [t if t != eos else t + 1 for t in prompt]
        reference[name] = case_reference(model, prompt, max_new,
                                         args.min_margin)
        if eos in reference[name]["greedy_new_ids"]:
            raise SystemExit(f"case {name}: greedy emitted EOS")
        print(f"[ref] {name}: prompt {len(prompt)} -> +{max_new}, "
              f"min margin {reference[name]['min_logit_margin']}, "
              f"context-free agreement "
              f"{reference[name]['context_free_agreement']}", file=sys.stderr)
        if reference[name]["context_free_agreement"] > args.max_context_free:
            raise SystemExit(
                f"case {name}: the embedding alone predicts "
                f"{reference[name]['context_free_agreement']:.0%} of the "
                f"teacher-forced argmax; the layers barely matter and an engine "
                f"that skipped them would pass. Re-tune the generator.")
    router_gap = min(watch.gaps)
    print(f"[ref] router: {len(watch.gaps)} top-k boundaries, min gap "
          f"{router_gap:.3g}", file=sys.stderr)
    if router_gap < args.min_router_gap:
        raise SystemExit(f"router top-k boundary gap {router_gap:.3g} below "
                         f"--min-router-gap {args.min_router_gap}: float noise "
                         f"could swap an expert. Re-tune the generator.")

    image = None
    if "vision_config" in config:
        image = vision_reference(model, fixture, config, 8, args.min_margin)
        print(f"[ref] image: {image['grid_h']}x{image['grid_w']} patches, margin "
              f"{image['min_logit_margin']}, patches vs official processor "
              f"{image['patch_max_diff_vs_official_processor']}", file=sys.stderr)

    import torch
    import transformers
    output.write_text(json.dumps({
        "schema_version": 1,
        "source": "xiaomi-vendor",
        "vendor_repo": VENDOR_REPO,
        "vendor_revision": VENDOR_REVISION,
        "vendor_files": VENDOR_FILES,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "engine_env": {
            # exact-math configuration: dense weights widened to f32 instead of
            # the default int8 at load, float expert matmuls instead of int8
            # activations (a speed approximation, not an oracle target)
            "MIMO_DENSE_BITS": "32", "MIMO_IDOT": "0", "COLI_TEMP": "0"},
        "cases": reference,
        **({"image": image} if image else {}),
    }, indent=1) + "\n")
    print(f"[ref] wrote {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
