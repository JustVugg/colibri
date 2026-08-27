#!/usr/bin/env python3
"""Produce ref.json for the tiny Kimi K3 fixture from Moonshot's own code.

Maintainer tool, run offline once per fixture change; CI never executes it
(tests/test_kimi_k3_tiny.py only reads the committed ref.json).  The point is
independence: the reference tokens come from the vendor's published
implementation (modeling_kimi_linear.py on the Kimi-K3 Hugging Face repo,
pinned below by SHA-256), so the engine and its oracle never share an author.
A shared misunderstanding cannot pass both.

Parity is by construction, the same discipline as make_deepseek_v4_tiny.py:
- dense weights are stored f32 in the fixture and the engine is tested with
  K3_BITS=32 / K3_MLA_BITS=32 / K3_HEAD_BITS=32, so both sides compute from
  identical bytes;
- routed experts are stored packed MXFP4 g32; the engine decodes them
  natively while this script dequantizes THE SAME BYTES (low nibble = even
  column, w = e2m1 * 2^(scale-127)) into the vendor model's f32 Linear
  weights.  Either way the mathematical weights are equal, so token-level
  argmax must agree wherever the logit margin exceeds float noise -- and the
  generator engineers wide margins, which this script VERIFIES per emitted
  position instead of assuming (--min-margin, refuse on violation).

Requires: torch (CPU is fine), einops, numpy.  Network only if the vendor
files are not already cached next to --vendor-dir.
"""

import argparse
import hashlib
import importlib
import json
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np

VENDOR_REPO = "moonshotai/Kimi-K3"
# Pinned provenance: refuse to run against silently changed vendor code.
VENDOR_FILES = {
    "configuration_kimi_k3.py":
        "735eb9ebe593e17d231e08e1df7f7be9b5ee0e079f511aa201f9572077b416ae",
    "modeling_kimi_linear.py":
        "9e3564c70ac21854ce5a090cc946c5dc76b70d1050ef50840449181a20fff44a",
}

MX4_LUT = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                   dtype=np.float32)


def read_safetensors(path):
    with open(path, "rb") as fh:
        header_len = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(header_len))
        base = 8 + header_len
        raw = fh.read()
    dtypes = {"F32": np.float32, "U8": np.uint8}
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        lo, hi = meta["data_offsets"]
        dtype = dtypes[meta["dtype"]]
        arr = np.frombuffer(raw, dtype=dtype,
                            count=(hi - lo) // np.dtype(dtype).itemsize,
                            offset=lo)
        out[name] = arr.reshape(meta["shape"]).copy()
    return out


def mxfp4_dequant(packed, scale):
    """Exact peer of quant.h:matmul_mxfp4's decode: packed [O, I/2] with the
    LOW nibble on the even column, scale [O, I/32] as 2^(s-127) per group."""
    rows, half = packed.shape
    cols = half * 2
    codes = np.empty((rows, cols), dtype=np.uint8)
    codes[:, 0::2] = packed & 0x0F
    codes[:, 1::2] = packed >> 4
    values = MX4_LUT[codes]
    exponents = np.ldexp(np.float32(1.0), scale.astype(np.int32) - 127)
    return (values.reshape(rows, cols // 32, 32) *
            exponents[:, :, None]).reshape(rows, cols).astype(np.float32)


def fetch_vendor(vendor_dir):
    vendor_dir.mkdir(parents=True, exist_ok=True)
    for name, expected in VENDOR_FILES.items():
        path = vendor_dir / name
        if not path.exists():
            url = f"https://huggingface.co/{VENDOR_REPO}/raw/main/{name}"
            print(f"[ref] downloading {url}", file=sys.stderr)
            with urllib.request.urlopen(url, timeout=60) as resp:
                path.write_bytes(resp.read())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise SystemExit(
                f"{name}: sha256 {digest} does not match the pinned "
                f"{expected}; the vendor file changed upstream. Review it, "
                f"then update VENDOR_FILES deliberately.")
    init = vendor_dir / "__init__.py"
    if not init.exists():
        init.write_text("")


def load_vendor(vendor_dir):
    """Load the vendor files as a synthetic package, so their relative
    imports work regardless of the directory's (dotted) name."""
    import importlib.machinery
    import importlib.util
    pkg_name = "kimi_k3_vendor"
    spec = importlib.machinery.ModuleSpec(pkg_name, None, is_package=True)
    package = importlib.util.module_from_spec(spec)
    package.__path__ = [str(vendor_dir)]
    sys.modules[pkg_name] = package

    def load(stem):
        file_spec = importlib.util.spec_from_file_location(
            f"{pkg_name}.{stem}", vendor_dir / f"{stem}.py")
        module = importlib.util.module_from_spec(file_spec)
        sys.modules[f"{pkg_name}.{stem}"] = module
        file_spec.loader.exec_module(module)
        return module

    load("configuration_kimi_k3")
    module = load("modeling_kimi_linear")
    patch_kda_cpu(module)
    return module


def patch_kda_cpu(module):
    """The vendor calls fla's chunk_kda/fused_recurrent_kda, which are Triton
    kernels and need a GPU.  Reroute both to fla's OWN naive torch
    implementation (fla/ops/kda/naive.py) plus fla's OWN naive gate
    (fla/ops/kda/gate.py) -- the reference implementations fla tests its
    kernels against.  No colibri-authored math enters the oracle: the
    preprocessing below mirrors exactly the use_*_in_kernel flags the vendor
    passes, using fla's published formulas."""
    import torch
    import torch.nn.functional as F
    from fla.ops.kda.naive import naive_recurrent_kda
    from fla.ops.kda.gate import naive_kda_gate, naive_kda_lowerbound_gate

    def kda_cpu(q, k, v, g, beta, A_log=None, dt_bias=None, scale=None,
                initial_state=None, output_final_state=False,
                use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
                use_beta_sigmoid_in_kernel=False, safe_gate=False,
                lower_bound=None, transpose_state_layout=False,
                cu_seqlens=None, **_ignored):
        # our reference forward is single-sequence, cache-free
        assert cu_seqlens is None, "packed sequences not expected here"
        assert initial_state is None, "cache-free reference only"
        if use_qk_l2norm_in_kernel:
            q = F.normalize(q.float(), p=2, dim=-1)
            k = F.normalize(k.float(), p=2, dim=-1)
        if use_gate_in_kernel:
            if safe_gate or lower_bound is not None:
                g = naive_kda_lowerbound_gate(g, A_log, dt_bias,
                                              lower_bound
                                              if lower_bound is not None
                                              else -5.0)
            else:
                g = naive_kda_gate(g, A_log, dt_bias)
        if use_beta_sigmoid_in_kernel:
            beta = torch.sigmoid(beta.float())
        o, state = naive_recurrent_kda(
            q=q, k=k, v=v, g=g, beta=beta, scale=scale,
            initial_state=None, output_final_state=output_final_state)
        return o, state

    module.chunk_kda = kda_cpu
    module.fused_recurrent_kda = kda_cpu

    # fla's ShortConvolution supports only the triton/cuda backends.  Its
    # documented semantics are a causal depthwise conv (nn.Conv1d with
    # groups=D, padding=K-1, output truncated to T) plus SiLU.  Re-express
    # exactly that in stock torch.
    def shortconv_cpu(self, x, residual=None, mask=None, cache=None,
                      output_final_state=False, cu_seqlens=None,
                      chunk_indices=None, **_ignored):
        assert mask is None and cache is None and cu_seqlens is None
        assert not output_final_state, "cache-free reference only"
        length = x.shape[1]
        y = F.conv1d(x.transpose(1, 2).float(), self.weight.float(),
                     None if self.bias is None else self.bias.float(),
                     groups=self.groups,
                     padding=self.kernel_size[0] - 1)[..., :length]
        if self.activation in ("silu", "swish"):
            y = F.silu(y)
        y = y.transpose(1, 2).to(x.dtype)
        if residual is not None:
            y = y + residual
        return y, None

    module.ShortConvolution.forward = shortconv_cpu

    # FusedRMSNormGated is triton-only too.  Its kernel computes
    # y = rmsnorm(x) * weight * act(g) with act(g) = sigmoid(g) (or
    # g*sigmoid(g) for swish), per fused_norm_gate.py's kernel body.
    def rmsnorm_gated_cpu(self, x, g, residual=None, prenorm=False,
                          residual_in_fp32=False):
        assert residual is None and not prenorm
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            y = y * self.weight.float()
        gf = g.float()
        if self.activation in ("swish", "silu"):
            y = y * gf * torch.sigmoid(gf)
        else:
            y = y * torch.sigmoid(gf)
        return y.to(x.dtype)

    module.FusedRMSNormGated.forward = rmsnorm_gated_cpu


def build_model(module, config_dict):
    import torch
    config_cls = module.KimiLinearConfig
    config = config_cls(**dict(config_dict, attn_implementation="eager",
                               use_cache=False))
    # transformers 4.56 resolves the attention backend from the private
    # attribute; the constructor kwarg alone is not authoritative there.
    config._attn_implementation = "eager"
    torch.manual_seed(0)  # init values are all overwritten by the load below
    model = module.KimiLinearForCausalLM(config)
    # KimiLinearModel.__init__ force-overwrites any requested attention
    # implementation with flash_attention_2 (modeling_kimi_linear.py:1110).
    # The MLA forward re-reads self.config._attn_implementation at call time
    # and has a full eager path, so restoring it after construction is
    # effective and touches no vendor math.
    model.config._attn_implementation = "eager"
    model.eval()
    return model


def to_state_dict(tensors, config_dict):
    """Fixture names -> vendor parameter names, with MXFP4 dequantized."""
    import torch
    kda_heads = config_dict["linear_attn_config"]["num_heads"]
    state, packed = {}, {}
    for name, arr in tensors.items():
        if name.endswith(".weight_packed"):
            packed[name[:-len(".weight_packed")]] = arr
            continue
        if name.endswith(".weight_scale"):
            continue
        state[name] = torch.from_numpy(np.ascontiguousarray(arr))
    for base, pk in packed.items():
        scale = tensors[base + ".weight_scale"]
        state[base + ".weight"] = torch.from_numpy(mxfp4_dequant(pk, scale))
    # The checkpoint stores A_log zero-padded to head_dim; the model wants
    # one entry per KDA head (the engine reads the same first slice).
    for name in list(state):
        if name.endswith("self_attn.A_log"):
            state[name] = state[name][:kda_heads].clone()
    return state


def forward_argmax(model, ids):
    import torch
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids], dtype=torch.long),
                       use_cache=False).logits[0]
    return logits


def case_reference(model, prompt, max_new, min_margin):
    import torch
    sequence = list(prompt)
    margins = []
    for _ in range(max_new):
        logits = forward_argmax(model, sequence)[-1]
        top2 = torch.topk(logits, 2).values
        margins.append(float(top2[0] - top2[1]))
        sequence.append(int(logits.argmax()))
    full_logits = forward_argmax(model, sequence)
    teacher = full_logits.argmax(-1).tolist()
    tf_top2 = torch.topk(full_logits, 2, dim=-1).values
    tf_margins = (tf_top2[:, 0] - tf_top2[:, 1]).tolist()
    worst = min(margins + tf_margins)
    if worst < min_margin:
        raise SystemExit(
            f"logit margin {worst:.4f} below --min-margin {min_margin}: the "
            f"fixture rides a near-tie and token-exactness would be luck, "
            f"not correctness. Re-tune the generator instead of lowering "
            f"the margin.")
    return {
        "prompt_ids": list(prompt),
        "teacher_forcing_ids": teacher,
        "greedy_full_ids": sequence,
        "greedy_new_ids": sequence[len(prompt):],
        "max_new_tokens": max_new,
        "min_logit_margin": round(worst, 4),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fixture", default="./kimi_k3_tiny")
    parser.add_argument("--vendor-dir", default=None,
                        help="dir with the pinned vendor .py files "
                             "(default: <fixture>/.vendor_kimi_k3)")
    parser.add_argument("--output", default=None,
                        help="default: <fixture>/ref.json")
    parser.add_argument("--min-margin", type=float, default=0.05)
    args = parser.parse_args()

    fixture = Path(args.fixture)
    vendor_dir = (Path(args.vendor_dir) if args.vendor_dir
                  else fixture / ".vendor_kimi_k3")
    output = Path(args.output) if args.output else fixture / "ref.json"

    config_dict = json.loads((fixture / "config.json").read_text())
    tensors = read_safetensors(fixture / "model.safetensors")

    fetch_vendor(vendor_dir)
    module = load_vendor(vendor_dir)
    model = build_model(module, config_dict)
    state = to_state_dict(tensors, config_dict)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Refuse silence: every fixture tensor must land, and every model weight
    # must be fed. Anything else is a drifted name map, not a detail.
    if missing or unexpected:
        raise SystemExit(f"state dict mismatch:\n  missing: {missing}\n"
                         f"  unexpected: {unexpected}")

    eos = config_dict.get("eos_token_id", 1)
    vocab = config_dict["vocab_size"]
    cases = {
        # mirrors make_deepseek_v4_tiny.py: short, chunk-crossing, long
        "short": ([5, 7, 9, 11, 13, 17, 19, 23], 8),
        "chunk": ([(5 + i * 7) % vocab for i in range(40)], 4),
        "long": ([(5 + i * 11) % vocab for i in range(72)], 4),
    }
    reference = {}
    for name, (prompt, max_new) in cases.items():
        prompt = [t if t != eos else t + 1 for t in prompt]
        reference[name] = case_reference(model, prompt, max_new,
                                         args.min_margin)
        if eos in reference[name]["greedy_new_ids"]:
            raise SystemExit(f"case {name}: greedy emitted EOS; the fixture "
                             f"must keep EOS unattractive")
        print(f"[ref] {name}: prompt {len(prompt)} -> +{max_new}, "
              f"min margin {reference[name]['min_logit_margin']}",
              file=sys.stderr)

    import torch
    import transformers
    output.write_text(json.dumps({
        "schema_version": 1,
        "source": "moonshot-vendor",
        "vendor_repo": VENDOR_REPO,
        "vendor_files": VENDOR_FILES,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "engine_env": {
            # exact-math configuration: dense weights at f32 (the fixture
            # stores f32; default K3_BITS=4 would requantize at load) and
            # float expert matmuls (default K3_IDOT=1 quantizes activations
            # to int8 per group of 32, a deliberate ~0.4%/group speed
            # approximation that is NOT an oracle target)
            "K3_BITS": "32", "K3_MLA_BITS": "32", "K3_HEAD_BITS": "32",
            "K3_IDOT": "0", "COLI_TEMP": "0"},
        "cases": reference,
    }, indent=1) + "\n")
    print(f"[ref] wrote {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
