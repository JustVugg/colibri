#!/usr/bin/env python3
"""Generate CPU reference inputs and latents with the pinned Diffusers VAE.

Install requirements-qwenimage-tiny.txt, then run this script with an output
directory. The default uses the decoder fixture's tiny random VAE, without
downloading model weights. Pass a local checkpoint with --vae to generate
references for trained-weight validation.
"""
import argparse
import json
from pathlib import Path

import torch
from diffusers import AutoencoderKLQwenImage21
from safetensors.torch import save_file

from make_qwenimage_vae_tiny import tiny_vae


def generate(outdir, vae_dir=None, shapes=((16, 16), (32, 48), (48, 32), (80, 112))):
    torch.set_num_threads(2)
    root = Path(outdir)
    root.mkdir(parents=True, exist_ok=True)
    vae = (AutoencoderKLQwenImage21.from_pretrained(vae_dir, torch_dtype=torch.float32)
           if vae_dir else tiny_vae(1234)).eval()
    if not vae_dir:
        vae.save_pretrained(root / "vae")
    g = torch.Generator().manual_seed(1235)
    tensors = {}
    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
    cases = []
    inputs = []
    for k, (h, w) in enumerate(shapes):
        channels = 3 if k % 2 == 0 else 4
        pixels = torch.randint(0, 256, (h, w, channels), generator=g, dtype=torch.uint8)
        inputs.append(pixels)
    # Same RGB in three inputs: automatic opaque alpha, explicit opaque alpha,
    # and explicit transparent alpha. No compositing over black or white.
    rgb = inputs[0][:, :, :3]
    for alpha in (255, 0):
        inputs.append(torch.cat((rgb, torch.full((*rgb.shape[:2], 1), alpha, dtype=torch.uint8)), -1))
    for k, pixels in enumerate(inputs):
        h, w, channels = pixels.shape
        if channels == 3:
            rgba = torch.cat((pixels, torch.full((h, w, 1), 255, dtype=torch.uint8)), dim=-1)
        else:
            rgba = pixels
        x = (rgba.float() / 127.5 - 1).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
        with torch.no_grad():
            mode = vae.encode(x).latent_dist.mode()
            z = (mode - mean) / std
        for name, value in (("pixels", pixels), ("input", x[0, :, 0].permute(1, 2, 0)),
                            ("mode", mode[0, :, 0].permute(1, 2, 0)),
                            ("z", z[0, :, 0].permute(1, 2, 0))):
            tensors[f"case{k}.{name}"] = value.contiguous()
        cases.append(dict(height=h, width=w, channels=channels))
    first, opaque, transparent = (tensors[f"case{k}.z"] for k in (0, len(inputs) - 2, len(inputs) - 1))
    torch.testing.assert_close(first, opaque, rtol=0, atol=0)
    assert (opaque - transparent).abs().max().item() > 1e-5, "fixture does not exercise alpha"
    ref = root / "encoder_ref"
    ref.mkdir(exist_ok=True)
    save_file(tensors, str(ref / "ref.safetensors"))
    manifest = {"oracle": "diffusers.AutoencoderKLQwenImage21.encode().latent_dist.mode()",
                "weights": "existing" if vae_dir else "tiny-random", "cases": cases}
    (root / "encoder_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir")
    parser.add_argument("--vae", help="existing VAE directory; no downloads")
    args = parser.parse_args()
    generate(args.outdir, args.vae)
