"""
A/B any quantization scheme against fp16 — WITHOUT converting a 370 GB model first.

Why this exists
---------------
Measuring "what does int4 cost us?" by comparing colibri's score to a published model-card
number does not work: the harness scores 0-shot log-likelihood, published numbers are
few-shot/CoT, and that protocol gap can swamp the quantization effect (see #108).

This tool removes the confound by construction. It takes an fp16 model, pushes its weights
through colibri's OWN quantizer (quantize -> dequantize, in place), and scores both with the
SAME harness, on the SAME questions, on the SAME machine. The only variable is the quantizer,
so the delta IS the quantization cost.

It runs on a small model (OLMoE) in minutes, so a scheme can be ranked BEFORE committing to
a multi-hour GLM conversion. The quantizer math is replicated from tools/convert_fp8_to_int4.py
(symmetric absmax, per-row scales) and generalised with an optional group size.

Measured with this tool on OLMoE-1B-7B, n=200/task (issue #108):

    scheme            hellaswag  arc_c   mmlu   mean   delta
    fp16                  77.0%  47.0%  47.0%  57.0%      --
    int4       (shipped)  74.0%  41.0%  31.5%  48.8%   -8.2pp
    int4-nohead           73.5%  40.5%  37.5%  50.5%   -6.5pp
    int4-g128             78.5%  45.5%  38.0%  54.0%   -3.0pp
    int4-g128-nohead      78.5%  46.5%  38.0%  54.3%   -2.7pp

  -> the per-row int4 container costs ~8pp, concentrated on the HARD task (MMLU falls to
     31.5% against a 25% random baseline, while easy HellaSwag barely moves): per-row scales
     eat the small logit margins hard questions depend on.
  -> group=128 recovers ~63% of that loss for ~+0.25 bits/weight.
  -> leaving lm_head/embed in fp16 is NOT the fix (+1.7pp alone, +0.3pp on top of grouping).

Usage
-----
    pip install torch transformers            # dev-only; the engine stays dependency-free
    python tools/fetch_benchmarks.py --out ./bench --tasks hellaswag,arc_challenge,mmlu --limit 200
    python tools/quant_ablation.py --model allenai/OLMoE-1B-7B-0924 --data ./bench \
        --tasks hellaswag,arc_challenge,mmlu --limit 200 \
        --schemes fp16,int4,int4-g64,int4-g128

Scheme grammar: fp16 | int{2,4,8}[-g<N>][-nohead]
    int4          per-row absmax int4 -- what the converter ships today
    int4-g64      one scale per 64 input weights instead of per row
    int4-nohead   as int4, but lm_head/embed kept in fp16
"""
import argparse
import json
import random
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# --------------------------------------------------------------------------------------
# colibri's quantizer (tools/convert_fp8_to_int4.py:32-52), generalised with a group size.
#
# LAYOUT NOTE, and it is a trap: transformers fuses MoE experts into 3D tensors
# (mlp.experts.gate_up_proj = [n_experts, in, out]) even when the checkpoint stores one 2D
# matrix per expert. A `p.ndim == 2` filter therefore skips EVERY expert and silently leaves
# ~85% of an MoE in fp16 while appearing to work. Both layouts must be handled, and the
# coverage assert below exists to make that failure loud instead of plausible.
# --------------------------------------------------------------------------------------
def _quant_last_dim(x, bits, group):
    """Symmetric absmax quantize->dequantize along the last (input) dim."""
    qmax = (1 << (bits - 1)) - 1          # int4 -> 7, int8 -> 127, int2 -> 1
    qmin = -(qmax + 1)                    # int4 -> -8  (nibble [-8,7], as the converter does)
    if group:
        if x.shape[-1] % group:
            raise SystemExit(f"group {group} does not divide input dim {x.shape[-1]}")
        x = x.reshape(*x.shape[:-1], x.shape[-1] // group, group)
    amax = x.abs().amax(dim=-1, keepdim=True)
    s = torch.clamp(amax / qmax, min=1e-8)
    q = torch.clamp(torch.round(x / s), qmin, qmax)
    out = q * s
    return out.reshape(*out.shape[:-2], -1) if group else out


# --------------------------------------------------------------------------------------
# Rotation preconditioning (QuaRot / QuIP# family, #81): multiply the input dimension by
# an orthogonal Q = diag(signs) @ H/sqrt(n) BEFORE quantizing, and by Q^T after — the
# round-trip Q4(W@Q)@Q.T measures exactly the weight error of a deployed scheme that
# stores W@Q quantized and rotates activations at runtime (W'@x' = W@x since Q@Q.T = I;
# the runtime cost is one O(D log D) transform per matmul INPUT, not per weight).
# Spreading outliers across the block is the point: absmax scales stop being hostage to
# one heavy coordinate, which is the failure mode #108 measured (margin erosion on MMLU).
# --------------------------------------------------------------------------------------
_ROT_CACHE = {}

def rotation(dim, device, seed=417):
    key = (dim, str(device))
    if key in _ROT_CACHE:
        return _ROT_CACHE[key]
    if dim & (dim - 1):
        raise SystemExit(f"-rot needs power-of-2 input dims (got {dim}); OLMoE dims are 2048/1024")
    h = torch.ones(1, 1, device=device, dtype=torch.float32)
    while h.shape[0] < dim:                       # Sylvester recursion
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    h /= h.shape[0] ** 0.5                        # orthonormal
    g = torch.Generator().manual_seed(seed + dim)
    signs = (torch.randint(0, 2, (dim,), generator=g).float() * 2 - 1).to(device)
    q = signs[:, None] * h                        # Q = D @ H/sqrt(n), orthogonal
    _ROT_CACHE[key] = q
    return q


def quantize_param(w, bits, group, rot=False, e8=""):
    if w.ndim == 3:                        # fused experts [E, in, out] -> move input last
        x = w.transpose(1, 2).contiguous()
        x = _rot_quant(x, bits, group, e8) if rot else _grid_or_e8(x, bits, group, e8)
        return x.transpose(1, 2).contiguous()
    if rot:
        return _rot_quant(w, bits, group, e8)
    return _grid_or_e8(w, bits, group, e8)  # nn.Linear [out, in] -- input already last


def _grid_or_e8(x, bits, group, e8):
    if e8:
        return _quant_e8(x.float(), group, bits, ball=(e8 == "-e8"))
    return _quant_last_dim(x, bits, group)


def _rot_quant(x, bits, group, e8=""):
    """W -> Qn(W@Q) @ Q^T along the last (input) dim — see rotation() above."""
    q = rotation(x.shape[-1], x.device)
    return (_grid_or_e8(x.float() @ q, bits, group, e8) @ q.T).contiguous()


# --------------------------------------------------------------------------------------
# E8 lattice quantization (#81 follow-up): the -rot schemes above are QuaRot (rotation +
# uniform grid). QuIP#'s 2-bit result needs the second ingredient — an E8 lattice codebook
# instead of the grid. E8 = D8 ∪ (D8 + 1/2), nearest point via Conway–Sloane: round every
# coordinate, and if the sum is odd re-round the worst coordinate the other way; repeat on
# the half-shifted copy and keep the closer of the two. `-e8` clamps points to |p|^2 <= 10,
# the E8P ball QuIP# builds its 2^16 codebook from (2 bits/weight for 8-dim blocks);
# `-e8u` leaves the lattice unbounded — an ideal-codebook upper bound, not a deployable rate.
# Scale: per group, a small MSE search over multiples of the block RMS (absmax is the wrong
# statistic for a lattice — the ball wants energy matched, not the peak).
# --------------------------------------------------------------------------------------
def _d8_nearest(y):
    f = torch.round(y)
    d = y - f
    odd = (f.sum(-1).long() & 1).bool()
    idx = d.abs().argmax(-1, keepdim=True)
    step = torch.where(d.gather(-1, idx) >= 0, 1.0, -1.0)
    flipped = f.gather(-1, idx) + step
    return f.scatter(-1, idx, torch.where(odd[..., None], flipped, f.gather(-1, idx)))


def _e8_nearest(y):
    a = _d8_nearest(y)
    b = _d8_nearest(y - 0.5) + 0.5
    da = ((y - a) ** 2).sum(-1, keepdim=True)
    db = ((y - b) ** 2).sum(-1, keepdim=True)
    return torch.where(da <= db, a, b)


def _e8_ball(y, r2=10.0):
    p = _e8_nearest(y)
    for _ in range(8):                                # shrink-and-requantize until inside
        n2 = (p ** 2).sum(-1, keepdim=True)
        over = n2 > r2 + 1e-6
        if not over.any():
            break
        y = torch.where(over, y * torch.sqrt(r2 / torch.clamp(n2, min=r2)) * 0.98, y)
        p = torch.where(over, _e8_nearest(y), p)
    return p


_E8_R2_REPORTED = set()
def _e8_radius(bits):
    # E8 lattice: points within |p|^2<=r2 grow ~r2^4, so +1 bit (x256 codebook) needs r2 x4.
    # Anchor: r2=10 is the ~2^16 E8P ball (2 bits over 8 dims). Scale from there.
    return 10.0 * (4.0 ** (bits - 2))

def _quant_e8(x, group, bits, ball):
    """Blocks of 8 along the input dim; per-group scale by MSE search over RMS multiples.
    ball=True clamps to the rate-scaled E8 ball for `bits`; ball=False is the unbounded ideal."""
    if x.shape[-1] % 8:
        raise SystemExit(f"-e8 needs input dim divisible by 8 (got {x.shape[-1]})")
    g = group or x.shape[-1]
    if g % 8:
        raise SystemExit(f"-e8 group {g} must be a multiple of 8")
    shp = x.shape
    xg = x.reshape(-1, g)                             # [G, g]
    rms = torch.clamp(xg.pow(2).mean(-1, keepdim=True).sqrt(), min=1e-8)
    best_out, best_err = None, None
    for k in (0.5, 0.7, 0.9, 1.1, 1.4, 1.8, 2.4):
        s = rms * k
        yb = (xg / s).reshape(-1, g // 8, 8)
        p = _e8_ball(yb, _e8_radius(bits)) if ball else _e8_nearest(yb)
        if ball and bits not in _E8_R2_REPORTED:
            _E8_R2_REPORTED.add(bits)
            import sys as _sys
            _sys.stderr.write(f"[e8] bits={bits}: ball r2={_e8_radius(bits):.1f}\n")
        out = (p.reshape(-1, g) * s)
        err = (out - xg).pow(2).sum(-1, keepdim=True)
        if best_err is None:
            best_out, best_err = out, err
        else:
            take = err < best_err
            best_out = torch.where(take, out, best_out)
            best_err = torch.where(take, err, best_err)
    return best_out.reshape(shp)


SCHEME_RE = re.compile(r"^int(2|3|4|8)(?:-g(\d+))?(-e8u?)?(-rot)?(-nohead)?$")


def parse_scheme(name):
    """'int4-g128-nohead' -> (bits, group, e8, skip_head...). 'fp16' -> None."""
    if name == "fp16":
        return None
    m = SCHEME_RE.match(name)
    if not m:
        raise SystemExit(f"bad scheme '{name}' (expected fp16 | int{{2,3,4,8}}[-g<N>][-e8|-e8u][-rot][-nohead])")
    return int(m.group(1)), int(m.group(2) or 0), m.group(3) or "", bool(m.group(4)), bool(m.group(5))


def is_router(name):
    # The router (mlp.gate.weight) stays f32 in the converter -- convert_fp8_to_int4.py:14.
    # Careful: expert weights are gate_proj/up_proj/down_proj and DO get quantized.
    return name.endswith("mlp.gate.weight")


def is_head_or_embed(name):
    return "embed_tokens" in name or "lm_head" in name


def apply_scheme(model, scheme):
    """Quantize the tensor classes the converter hits (attn/mlp/expert/embed/lm_head);
    norms, router and biases stay float. Returns (n_tensors, quantized_params, total)."""
    total = sum(p.numel() for p in model.parameters())
    spec = parse_scheme(scheme)
    if spec is None:
        return 0, 0, total
    bits, group, e8, rot, skip_head = spec
    n = qp = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim < 2 or is_router(name):
                continue
            if skip_head and is_head_or_embed(name):
                continue
            p.data.copy_(quantize_param(p.data.float(), bits, group, rot, e8).to(p.dtype))
            n += 1
            qp += p.numel()
    return n, qp, total


# --------------------------------------------------------------------------------------
# Scoring — mirrors tools/eval_glm.py exactly:
#   acc      = argmax over options of sum(logprob of continuation tokens)
#   acc_norm = argmax over options of sum(logprob) / len(continuation string in CHARACTERS)
#
# THE PREFIX IS NOT OPTIONAL (issue #108, credit @bokiko). GLM-5.2 sees "[gMASK]<sop>" at the
# start of every training sequence. Score it without that prefix and the model runs
# out-of-distribution: measured here, perplexity on plain English prose goes 9.4 -> 29.2, and
# on markdown/code 24.5 -> 131.0. It does not merely depress scores, it distorts SENSITIVITY:
# an int4-vs-exact kernel A/B measured without the prefix reported a penalty that halved and
# flipped sign on one corpus once the prefix was restored (#153). Any quantization delta
# measured OOD is therefore suspect.
#
# The GLM tokenizer does NOT add it for you — add_special_tokens=True is a no-op there — so it
# has to be prepended explicitly. Auto-detected from the vocab so it cannot be lost by
# omission; models with no such prefix (e.g. OLMoE, which has no BOS at all) are unaffected.
# --------------------------------------------------------------------------------------
def detect_prefix(tk):
    """'[gMASK]<sop>' for GLM snapshots, '' otherwise. --prefix overrides."""
    vocab = tk.get_vocab()
    if "[gMASK]" in vocab and "<sop>" in vocab:
        return "[gMASK]<sop>"
    return ""


def load_docs(task, data_dir, limit, seed, prefix=""):
    path = f"{data_dir}/{task}.jsonl"
    try:
        docs = [json.loads(l) for l in open(path) if l.strip()]
    except FileNotFoundError:
        raise SystemExit(f"missing {path} — run: python tools/fetch_benchmarks.py --out {data_dir} --tasks {task}")
    random.Random(seed).shuffle(docs)      # same seed/shuffle convention as eval_glm.py
    docs = docs[:limit] if limit else docs
    if prefix:                             # condition every context in-distribution
        docs = [dict(d, ctx=prefix + d["ctx"]) for d in docs]
    return docs


@torch.no_grad()
def score(model, tk, docs, device):
    acc = accn = 0
    for d in docs:
        ctx, choices, gold = d["ctx"], d["choices"], int(d["gold"])
        ctx_ids = tk(ctx, add_special_tokens=False).input_ids
        lps, norms = [], []
        for cont in choices:
            full = tk(ctx + cont, add_special_tokens=False).input_ids
            cl = len(ctx_ids)
            while cl > 0 and (cl > len(full) or full[:cl] != ctx_ids[:cl]):
                cl -= 1
            if not full[cl:]:
                full = ctx_ids + tk(cont, add_special_tokens=False).input_ids
                cl = len(ctx_ids)
            cl = max(1, cl)
            ids = torch.tensor([full], device=device)
            logprobs = torch.log_softmax(model(ids).logits.float()[0, :-1], dim=-1)
            tgt = ids[0, 1:]
            lps.append(logprobs[torch.arange(cl - 1, len(full) - 1), tgt[cl - 1:]].sum().item())
            norms.append(max(1, len(cont)))          # CHARACTER length, like eval_glm.py
        acc += max(range(len(lps)), key=lambda i: lps[i]) == gold
        accn += max(range(len(lps)), key=lambda i: lps[i] / norms[i]) == gold
    n = len(docs)
    return 100 * acc / n, 100 * accn / n


def main():
    ap = argparse.ArgumentParser(description="A/B a quantization scheme against fp16, engine-free")
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924", help="HF repo id or local dir")
    ap.add_argument("--data", default="./bench")
    ap.add_argument("--tasks", default="hellaswag,arc_challenge,mmlu")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--schemes", default="fp16,int4,int4-g128",
                    help="comma list: fp16 | int{2,4,8}[-g<N>][-nohead]")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--min-coverage", type=float, default=95.0,
                    help="fail if a scheme quantized less than this %% of params (catches the "
                         "3D-fused-expert trap, where a ndim==2 filter skips every expert)")
    ap.add_argument("--prefix", default=None,
                    help="context prefix (default: auto — '[gMASK]<sop>' for GLM, '' otherwise). "
                         "Scoring GLM without it runs the model out-of-distribution: see #108.")
    a = ap.parse_args()

    tasks = a.tasks.split(",")
    schemes = a.schemes.split(",")
    for s in schemes:
        parse_scheme(s)                              # fail fast on a typo

    tk = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    prefix = detect_prefix(tk) if a.prefix is None else a.prefix
    print(f"[prefix] {prefix!r}" + ("  (auto-detected: GLM snapshot)" if prefix and a.prefix is None
                                    else "  (no prefix for this model)" if not prefix else "  (--prefix)"),
          flush=True)
    docs = {t: load_docs(t, a.data, a.limit, a.seed, prefix) for t in tasks}

    means, rows = {}, {}
    for scheme in schemes:
        model = AutoModelForCausalLM.from_pretrained(
            a.model, dtype=torch.float16, low_cpu_mem_usage=True,
            device_map={"": 0} if a.device == "cuda" else None, trust_remote_code=True)
        model.eval()
        if a.device != "cuda":
            model.to(a.device)

        n, qp, tp = apply_scheme(model, scheme)
        cov = 100 * qp / tp if tp else 0.0
        print(f"[{scheme}] {n} tensors · {qp/1e9:.2f}B/{tp/1e9:.2f}B params ({cov:.1f}% coverage)",
              flush=True)
        if scheme != "fp16" and cov < a.min_coverage:
            raise SystemExit(
                f"ERROR: {scheme} quantized only {cov:.1f}% of parameters (< {a.min_coverage}%).\n"
                f"       The experts are probably being skipped: transformers fuses MoE experts\n"
                f"       into 3D tensors, so a ndim==2 filter silently leaves them in fp16.")

        rows[scheme] = {t: score(model, tk, docs[t], a.device) for t in tasks}
        means[scheme] = sum(v[1] for v in rows[scheme].values()) / len(tasks)
        for t in tasks:
            print(f"  {t:<16} n={len(docs[t]):<4} acc {rows[scheme][t][0]:5.1f}%"
                  f"  acc_norm {rows[scheme][t][1]:5.1f}%", flush=True)
        print(f"  {'MEAN acc_norm':<16} {means[scheme]:5.1f}%\n", flush=True)
        del model
        torch.cuda.empty_cache()

    base = means.get("fp16")
    print(f"{'scheme':<20}{'mean acc_norm':>14}{'delta vs fp16':>16}")
    for scheme in schemes:
        d = f"{means[scheme]-base:+.1f}pp" if base is not None and scheme != "fp16" else "--"
        print(f"{scheme:<20}{means[scheme]:>13.1f}%{d:>16}")
    if base is None:
        print("\n(no fp16 baseline in --schemes, so no deltas)", file=sys.stderr)


if __name__ == "__main__":
    main()
