#!/usr/bin/env python3
"""Oracle for the Qwen-Image-2.1 VAE decoder (c/qwenimage_vae.h).

    make_qwenimage_vae_tiny.py OUTDIR
        builds a tiny random AutoencoderKLQwenImage21 with the real depth
        (five up blocks, three resnets each, the mid attention) and narrow
        channels, saves it to OUTDIR/vae, and writes the diffusers decode of
        random latents to OUTDIR/vae_ref/ref.safetensors.

    make_qwenimage_vae_tiny.py OUTDIR --vae VAE_DIR --shapes 32x32,27x48
        decodes random latents with an existing VAE (the real checkpoint) and
        writes only OUTDIR/vae_ref/ref.safetensors.

Every case k stores three tensors:
    case{k}.z     f32  [h][w][z_dim]  normalized latents, token-major: exactly
                                      what the DiT hands to the decoder
    case{k}.out   f32  [4][16h][16w]  vae.decode() after the clamp to [-1, 1]
    case{k}.rgba  u8   [16h][16w][4]  VaeImageProcessor.postprocess(..., "pil")
The decode path is the pipeline's own: unpack to [1, C, 1, h, w], multiply by
latents_std and add latents_mean from the VAE config, decode, take frame 0,
postprocess. Needs torch and diffusers (the reference venv), not the engine.
"""
import argparse
import os
import time

import numpy as np
import torch
from diffusers import AutoencoderKLQwenImage21
from diffusers.image_processor import VaeImageProcessor
from safetensors.torch import save_file


def tiny_vae(seed):
    torch.manual_seed(seed)
    z_dim = 16  # not 64: the engine must take z_dim from config.json
    g = torch.Generator().manual_seed(seed + 1)
    vae = AutoencoderKLQwenImage21(
        base_dim=8,
        decoder_base_dim=8,
        z_dim=z_dim,
        latents_mean=(torch.randn(z_dim, generator=g) * 1.5).tolist(),
        latents_std=(2.5 + torch.rand(z_dim, generator=g) * 2).tolist(),
    )
    with torch.no_grad():
        for name, p in vae.named_parameters():
            if name.endswith("gamma"):
                # the default is all ones, which would hide a gamma indexed per the wrong axis
                p.copy_(1.0 + 0.3 * torch.randn(p.shape, generator=g))
            elif name.endswith("bias"):
                p.copy_(0.1 * torch.randn(p.shape, generator=g))
        # left at its default init, conv_out spans the whole [-1, 1] range and
        # saturates about 3% of the values: the clamp is exercised, yet the
        # uint8 comparison still sees almost every pixel unclamped
    return vae.eval()


def decode_case(vae, z_tok, h, w):
    """z_tok [h*w][z_dim] normalized latents -> (float [4][H][W], uint8 [H][W][4], seconds)."""
    cfg = vae.config
    z_dim = cfg.z_dim
    # pipeline._unpack_latents, without its rounding of height and width to 32 pixels
    lat = z_tok.unsqueeze(0).transpose(1, 2).reshape(1, z_dim, 1, h, w)
    mean = torch.tensor(cfg.latents_mean).view(1, z_dim, 1, 1, 1).to(lat.dtype)
    std = torch.tensor(cfg.latents_std).view(1, z_dim, 1, 1, 1).to(lat.dtype)
    lat = lat * std + mean
    t0 = time.time()
    with torch.no_grad():
        image = vae.decode(lat, return_dict=False)[0][:, :, 0]
    dt = time.time() - t0
    proc = VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=z_dim)
    pil = proc.postprocess(image, output_type="pil")[0]
    rgba = np.array(pil)
    return image[0].contiguous(), torch.from_numpy(rgba.copy()), dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--vae", help="existing VAE dir (default: build a tiny random one)")
    ap.add_argument("--shapes", default="5x7,4x6", help="latent HxW list, comma separated")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    if a.vae:
        vae = AutoencoderKLQwenImage21.from_pretrained(a.vae, torch_dtype=torch.float32).eval()
    else:
        vae = tiny_vae(a.seed)
        vae.save_pretrained(os.path.join(a.outdir, "vae"))
    g = torch.Generator().manual_seed(a.seed + 7)
    tensors = {}
    for k, s in enumerate(a.shapes.split(",")):
        h, w = (int(v) for v in s.lower().split("x"))
        z_tok = torch.randn(h * w, vae.config.z_dim, generator=g)
        out, rgba, dt = decode_case(vae, z_tok, h, w)
        sat = (out.abs() >= 1.0).float().mean().item()
        print(f"case{k}: latent {h}x{w} -> image {16 * h}x{16 * w}, diffusers decode {dt:.2f} s, "
              f"out range [{out.min().item():.3f}, {out.max().item():.3f}], saturated {100 * sat:.1f}%")
        tensors[f"case{k}.z"] = z_tok.reshape(h, w, -1).contiguous()
        tensors[f"case{k}.out"] = out
        tensors[f"case{k}.rgba"] = rgba
    os.makedirs(os.path.join(a.outdir, "vae_ref"), exist_ok=True)
    save_file(tensors, os.path.join(a.outdir, "vae_ref", "ref.safetensors"))


if __name__ == "__main__":
    main()
