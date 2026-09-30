"""Fit the latent -> RGB map the engine uses for its live previews.

Qwen-Image-2.1's 64 normalized latent channels at one latent pixel say roughly
what colour its 16x16 pixel tile will be. A least-squares linear map from those
64 values to the tile's mean RGB is enough to show the picture forming while the
DiT runs, at a cost of 64x3 multiply-adds per latent pixel.

The fit uses the real VAE encoder on a few varied images (pipeline encoding:
RGBA in [-1, 1], latent_dist mode, (z - mean) / std), holds one image out, and
prints the C table. Run with the diffusers venv:
  python fit_preview.py VAE_DIR img1 img2 ... """
import sys
import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKLQwenImage21

vae_dir, paths = sys.argv[1], sys.argv[2:]
vae = AutoencoderKLQwenImage21.from_pretrained(vae_dir, torch_dtype=torch.float32).eval()
mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)


def samples(path, side=512):
    img = Image.open(path).convert("RGBA").resize((side, side), Image.BICUBIC)
    a = np.asarray(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(a * 2 - 1).permute(2, 0, 1)[None, :, None]          # [1,4,1,H,W]
    with torch.no_grad():
        z = vae.encode(x).latent_dist.mode()
    z = ((z - mean) / std)[0, :, 0]                                          # [64,h,w]
    h, w = z.shape[1:]
    X = z.permute(1, 2, 0).reshape(h * w, -1).numpy()
    tiles = a[..., :3].reshape(h, 16, w, 16, 3).mean(axis=(1, 3)).reshape(h * w, 3)
    return X, tiles


data = [samples(p) for p in paths]
for held in range(len(data)):
    Xs = np.concatenate([d[0] for i, d in enumerate(data) if i != held])
    Ys = np.concatenate([d[1] for i, d in enumerate(data) if i != held])
    A = np.hstack([Xs, np.ones((len(Xs), 1))])
    W = np.linalg.solve(A.T @ A + 1e-2 * np.eye(A.shape[1]), A.T @ Ys)
    X, Y = data[held]
    P = np.hstack([X, np.ones((len(X), 1))]) @ W
    err = np.sqrt(((np.clip(P, 0, 1) - Y) ** 2).mean()) * 255
    print(f"held out {paths[held]}: RMS error {err:.1f} / 255 per channel")
Xs = np.concatenate([d[0] for d in data]); Ys = np.concatenate([d[1] for d in data])
A = np.hstack([Xs, np.ones((len(Xs), 1))])
W = np.linalg.solve(A.T @ A + 1e-2 * np.eye(A.shape[1]), A.T @ Ys)
print("static const float QI_PREVIEW_RGB[65][3] = {")
for r in W:
    print("    {%.6ff, %.6ff, %.6ff}," % tuple(r))
print("};")
