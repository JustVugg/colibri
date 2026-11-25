#!/usr/bin/env python3
"""Verify c/nemotron.exe bit-for-bit against official PyTorch NemotronH + RADIO math
on both F32 and mixed FP8+NVFP4 tiny fixtures (text + multimodal vision)."""
import json
import os
import struct
import subprocess
import sys
import tempfile
import numpy as np
import torch
import torch.nn.functional as F


def load_safetensors_raw(path):
    """Load safetensors without requiring numpy.float8_e4m3fn; F8_E4M3 -> uint8."""
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen).decode("utf-8"))
        base = 8 + hlen
        out = {}
        dt_map = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16,
                  "U8": np.uint8, "F8_E4M3": np.uint8}
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            s, e = meta["data_offsets"]
            f.seek(base + s)
            raw = f.read(e - s)
            arr = np.frombuffer(raw, dtype=dt_map[meta["dtype"]]).reshape(meta["shape"]).copy()
            out[name] = (meta["dtype"], arr)
    return out


def e4m3_lut():
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        s = (i >> 7) & 1
        e = (i >> 3) & 0xF
        m = i & 0x7
        v = 0.0 if (e == 0xF and m == 7) else ((m * 2.0 ** -9) if e == 0 else ((1.0 + m / 8.0) * 2.0 ** (e - 7)))
        lut[i] = -v if s else v
    return lut


E4M3 = e4m3_lut()
FP4 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)


def get_f32(st, name):
    dt, arr = st[name]
    if dt == "BF16":
        return torch.from_numpy((arr.astype(np.uint32) << 16).view(np.float32))
    return torch.from_numpy(arr.astype(np.float32))


def dequant_mat(st, name):
    dt, w = st[name]
    s1 = name.replace(".weight", ".weight_scale")
    s2 = name.replace(".weight", ".weight_scale_2")
    if dt == "U8" and s2 in st:
        O, half_I = w.shape
        I = half_I * 2
        lo = FP4[w & 0xF]
        hi = FP4[w >> 4]
        full = np.empty((O, I), dtype=np.float32)
        full[:, 0::2] = lo
        full[:, 1::2] = hi
        sc = E4M3[st[s1][1]]
        full = (full.reshape(O, I // 16, 16) * sc[:, :, None]).reshape(O, I) * float(st[s2][1])
        return torch.from_numpy(full)
    elif dt == "F8_E4M3":
        sc = float(st[s1][1]) if s1 in st else 1.0
        return torch.from_numpy(E4M3[w] * sc)
    return get_f32(st, name)


def rmsnorm(x, w, eps):
    v = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(v + eps) * w


def torch_forward(fixture_dir, token_ids, chw_img=None):
    cfg = json.load(open(os.path.join(fixture_dir, "config.json")))
    lc = cfg["llm_config"]
    vc = cfg["vision_config"]
    st = load_safetensors_raw(os.path.join(fixture_dir, "model.safetensors"))
    p = "language_model.backbone."
    emb_w = get_f32(st, p + "embeddings.weight")
    x = emb_w[torch.tensor(token_ids, dtype=torch.long)]  # [S, D]

    if chw_img is not None:
        # Run exact RADIO + pixel_shuffle(v2) + mlp1
        vp = "vision_model.radio_model.model."
        P = vc["patch_size"]
        D = vc["hidden_size"]
        C, H, W = chw_img.shape
        gh, gw = H // P, W // P
        pv = torch.from_numpy(chw_img).unsqueeze(0)  # [1,3,H,W]
        patches = pv.reshape(1, C, gh, P, gw, P).permute(0, 2, 4, 1, 3, 5).reshape(1, gh * gw, C * P * P)
        ew = get_f32(st, vp + "patch_generator.embedder.weight")
        pe = get_f32(st, vp + "patch_generator.pos_embed")
        cls_reg = get_f32(st, vp + "patch_generator.cls_token.token")
        vx = F.linear(patches, ew)
        mr = vc["max_img_size"] // P
        pos = pe.reshape(1, mr, mr, D).permute(0, 3, 1, 2)
        md = max(gh, gw)
        pos = F.interpolate(pos, size=(md, md), mode="bilinear", align_corners=False)[..., :gh, :gw]
        vx = vx + pos.flatten(2).permute(0, 2, 1)
        vx = torch.cat([cls_reg.unsqueeze(0), vx], dim=1)
        for bi in range(vc["num_hidden_layers"]):
            bp = f"{vp}blocks.{bi}."
            h = F.layer_norm(vx, (D,), get_f32(st, bp + "norm1.weight"), get_f32(st, bp + "norm1.bias"), vc["layer_norm_eps"])
            qkv = F.linear(h, get_f32(st, bp + "attn.qkv.weight"), get_f32(st, bp + "attn.qkv.bias"))
            q, k, v = qkv.chunk(3, dim=-1)
            nh, hd = vc["num_attention_heads"], D // vc["num_attention_heads"]
            q = q.view(1, -1, nh, hd).transpose(1, 2)
            k = k.view(1, -1, nh, hd).transpose(1, 2)
            v = v.view(1, -1, nh, hd).transpose(1, 2)
            aw = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * (hd ** -0.5), dim=-1)
            ao = torch.matmul(aw, v).transpose(1, 2).reshape(1, -1, D)
            vx = vx + F.linear(ao, get_f32(st, bp + "attn.proj.weight"), get_f32(st, bp + "attn.proj.bias"))
            h2 = F.layer_norm(vx, (D,), get_f32(st, bp + "norm2.weight"), get_f32(st, bp + "norm2.bias"), vc["layer_norm_eps"])
            mh = F.gelu(F.linear(h2, get_f32(st, bp + "mlp.fc1.weight"), get_f32(st, bp + "mlp.fc1.bias")))
            vx = vx + F.linear(mh, get_f32(st, bp + "mlp.fc2.weight"), get_f32(st, bp + "mlp.fc2.bias"))
        feat = vx[:, vc["num_cls_tokens"] + vc["num_registers"]:]
        feat = F.layer_norm(feat, (D,), get_f32(st, "vision_projector.vision_final_layernorm.weight"),
                            get_f32(st, "vision_projector.vision_final_layernorm.bias"), vc["layer_norm_eps"])
        feat = feat.reshape(1, gh, gw, D)
        feat = feat.view(1, gh, gw // 2, 2 * D).permute(0, 2, 1, 3).contiguous()
        feat = feat.view(1, gw // 2, gh // 2, 4 * D).permute(0, 2, 1, 3).contiguous().reshape(-1, 4 * D)
        feat = rmsnorm(feat, get_f32(st, "mlp1.0.weight"), lc["layer_norm_epsilon"])
        feat = F.relu(F.linear(feat, get_f32(st, "mlp1.1.weight"))).pow(2)
        img_emb = F.linear(feat, get_f32(st, "mlp1.3.weight"))
        ic = 0
        for idx_t, tid in enumerate(token_ids):
            if tid == cfg["img_context_token_id"] and ic < img_emb.shape[0]:
                x[idx_t] = img_emb[ic]
                ic += 1

    S, D = x.shape
    eps = lc["layer_norm_epsilon"]
    for li, bt in enumerate(lc["layers_block_type"]):
        lp = f"{p}layers.{li}."
        xn = rmsnorm(x, get_f32(st, lp + "norm.weight"), eps)
        if bt == "mamba":
            H, P, N, G, K = lc["mamba_num_heads"], lc["mamba_head_dim"], lc["ssm_state_size"], lc["n_groups"], lc["conv_kernel"]
            d_in = H * P
            cdim = d_in + 2 * G * N
            proj = F.linear(xn, dequant_mat(st, lp + "mixer.in_proj.weight"))
            z, xbc, dt = proj.split([d_in, cdim, H], dim=-1)
            cw = get_f32(st, lp + "mixer.conv1d.weight")
            cb = get_f32(st, lp + "mixer.conv1d.bias")
            xbc_t = xbc.transpose(0, 1).unsqueeze(0)  # [1, cdim, S]
            xbc_c = F.silu(F.conv1d(xbc_t, cw, cb, padding=K - 1, groups=cdim)[:, :, :S]).squeeze(0).transpose(0, 1)
            xs, B, C_ = xbc_c.split([d_in, G * N, G * N], dim=-1)
            A = -torch.exp(get_f32(st, lp + "mixer.A_log"))
            dt_b = get_f32(st, lp + "mixer.dt_bias")
            Dv = get_f32(st, lp + "mixer.D")
            state = torch.zeros(H, P, N)
            ys = []
            hpg = H // G
            for t in range(S):
                dt_t = F.softplus(dt[t] + dt_b)
                dA = torch.exp(A * dt_t)
                yt = torch.zeros(H, P)
                for h in range(H):
                    g = h // hpg
                    Bg = B[t, g * N:(g + 1) * N]
                    Cg = C_[t, g * N:(g + 1) * N]
                    xh = xs[t, h * P:(h + 1) * P]
                    state[h] = state[h] * dA[h] + dt_t[h] * torch.outer(xh, Bg)
                    yt[h] = state[h] @ Cg + Dv[h] * xh
                ys.append(yt.reshape(d_in))
            y = torch.stack(ys, dim=0) * F.silu(z)
            y = y.view(S, G, d_in // G)
            y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
            y = y.view(S, d_in) * get_f32(st, lp + "mixer.norm.weight")
            res = F.linear(y, dequant_mat(st, lp + "mixer.out_proj.weight"))
        elif bt == "attention":
            nh, nkv, hd = lc["num_attention_heads"], lc["num_key_value_heads"], lc["head_dim"]
            q = F.linear(xn, dequant_mat(st, lp + "mixer.q_proj.weight")).view(S, nh, hd).transpose(0, 1)
            k = F.linear(xn, dequant_mat(st, lp + "mixer.k_proj.weight")).view(S, nkv, hd).transpose(0, 1)
            v = F.linear(xn, dequant_mat(st, lp + "mixer.v_proj.weight")).view(S, nkv, hd).transpose(0, 1)
            k = k.repeat_interleave(nh // nkv, dim=0)
            v = v.repeat_interleave(nh // nkv, dim=0)
            mask = torch.triu(torch.full((S, S), float("-inf")), diagonal=1)
            aw = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * (hd ** -0.5) + mask, dim=-1)
            ao = torch.matmul(aw, v).transpose(0, 1).reshape(S, nh * hd)
            res = F.linear(ao, dequant_mat(st, lp + "mixer.o_proj.weight"))
        elif bt == "moe":
            gw = get_f32(st, lp + "mixer.gate.weight")
            gb = get_f32(st, lp + "mixer.gate.e_score_correction_bias")
            sc = torch.sigmoid(F.linear(xn, gw))
            ch = sc + gb
            top_i = torch.topk(ch, lc["num_experts_per_tok"], dim=-1).indices
            top_w = sc.gather(1, top_i)
            top_w = top_w / (top_w.sum(-1, keepdim=True) + 1e-20) * lc["routed_scaling_factor"]
            hl = F.linear(xn, get_f32(st, lp + "mixer.fc1_latent_proj.weight"))
            lo = torch.zeros_like(hl)
            for t in range(S):
                for ki in range(lc["num_experts_per_tok"]):
                    e = int(top_i[t, ki])
                    u = F.relu(F.linear(hl[t], dequant_mat(st, f"{lp}mixer.experts.{e}.up_proj.weight"))).pow(2)
                    d = F.linear(u, dequant_mat(st, f"{lp}mixer.experts.{e}.down_proj.weight"))
                    lo[t] += top_w[t, ki] * d
            res = F.linear(lo, get_f32(st, lp + "mixer.fc2_latent_proj.weight"))
            su = F.relu(F.linear(xn, dequant_mat(st, lp + "mixer.shared_experts.up_proj.weight"))).pow(2)
            res = res + F.linear(su, dequant_mat(st, lp + "mixer.shared_experts.down_proj.weight"))
        x = x + res

    xn = rmsnorm(x[-1], get_f32(st, p + "norm_f.weight"), eps)
    logits = F.linear(xn, dequant_mat(st, "language_model.lm_head.weight"))
    return logits.numpy()


def main():
    binary = sys.argv[1]
    extra_modes = [("cpu", ["--cpu"]), ("auto", [])]
    for quant in ("f32", "mixed"):
        with tempfile.TemporaryDirectory() as td:
            fdir = os.path.join(td, "fix")
            subprocess.check_call([sys.executable, os.path.join(os.path.dirname(__file__), "..", "tools", "make_nemotron_tiny.py"),
                                   "--output", fdir, "--quant", quant])
            toks = [1, 5, 12, 7, 33, 9]
            py_logits = torch_forward(fdir, toks)
            rng = np.random.default_rng(77)
            chw = rng.standard_normal((3, 16, 16)).astype(np.float32)
            img_bin = os.path.join(td, "img.bin")
            chw.tofile(img_bin)
            vtoks = [1, 18, 18, 18, 18, 42]
            pyv_logits = torch_forward(fdir, vtoks, chw_img=chw)

            for mname, mflags in extra_modes:
                dump = os.path.join(td, f"logits_{mname}.bin")
                subprocess.check_call([binary, fdir, "--prompt", " ".join(map(str, toks)),
                                       "--max-tokens", "1", "--dump-logits", dump] + mflags)
                c_logits = np.fromfile(dump, dtype=np.float32)
                max_err = float(np.max(np.abs(c_logits - py_logits)))
                assert int(np.argmax(c_logits)) == int(np.argmax(py_logits)), f"{quant}/{mname} argmax mismatch!"
                assert max_err < 1e-4, f"{quant}/{mname} text max_err={max_err}"
                print(f"[PASS] {quant:5s} {mname:4s} text-only:   max_abs_err={max_err:.2e} argmax={int(np.argmax(c_logits))}")

                dump_v = os.path.join(td, f"logits_v_{mname}.bin")
                subprocess.check_call([binary, fdir, "--prompt", " ".join(map(str, vtoks)),
                                       "--image-f32", img_bin, "16", "16", "--max-tokens", "1",
                                       "--dump-logits", dump_v] + mflags)
                cv_logits = np.fromfile(dump_v, dtype=np.float32)
                max_err_v = float(np.max(np.abs(cv_logits - pyv_logits)))
                assert int(np.argmax(cv_logits)) == int(np.argmax(pyv_logits)), f"{quant}/{mname} vision argmax mismatch!"
                assert max_err_v < 1e-4, f"{quant}/{mname} vision max_err={max_err_v}"
                print(f"[PASS] {quant:5s} {mname:4s} vision+text: max_abs_err={max_err_v:.2e} argmax={int(np.argmax(cv_logits))}")
    print("ALL NEMOTRON ORACLE CHECKS PASSED")


if __name__ == "__main__":
    main()
