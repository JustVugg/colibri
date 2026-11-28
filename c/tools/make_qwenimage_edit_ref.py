#!/usr/bin/env python3
"""Write single-reference edit oracles from the tiny Qwen-Image-2.1 model."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import QwenImage21Pipeline
from PIL import Image
from safetensors.torch import save_file

import qwenimage_ref as qref


def reference_image(pattern):
    """Two RGBA images of the same non-square shape, including soft alpha."""
    y, x = np.indices((192, 384))
    pixels = np.stack([
        ((x + pattern * 41) * 13) % 256,
        ((y + pattern * 7) * 17) % 256,
        ((x + y + pattern * 29) * 19) % 256,
        ((x * 5 + y * 11 + pattern * 67) % 224) + 32,
    ], axis=-1).astype(np.uint8)
    return Image.fromarray(pixels)


@torch.no_grad()
def make_oracle(pipe, model, out, pattern, width, height, steps, initial):
    prompt = "Replace the background with a beach, keeping the foreground."
    image = reference_image(pattern)
    # The area snaps to 384x192, above the fixture processor's minimum area,
    # so the VLM and VAE see exactly the same spatial dimensions.
    area_side = 272
    ref_width, ref_height = image.size
    embeds, _, mask = pipe.encode_prompt(prompt=prompt, image=[image], device=torch.device("cpu"))
    slots = mask[0].nonzero(as_tuple=True)[0]
    assert slots.numel() * 4 == ref_width * ref_height // 256
    assert torch.equal(slots, torch.arange(slots[0], slots[0] + slots.numel()))

    pixels = pipe.image_processor.preprocess(image, width=ref_width, height=ref_height).unsqueeze(2)
    ref_latents = pipe._encode_vae_image(pixels, generator=None)
    packed_ref = ref_latents[0, :, 0].permute(1, 2, 0).reshape(-1, ref_latents.shape[1])
    predictions = []
    final = []

    def capture(_transformer, args, kwargs, output):
        if not predictions:
            torch.testing.assert_close(kwargs["hidden_states"][0, :len(packed_ref)], packed_ref)
            torch.testing.assert_close(kwargs["encoder_hidden_states"], embeds)
            assert kwargs["img_shapes"] == [[(1, ref_height // 16, ref_width // 16),
                                              (1, height // 16, width // 16)]]
        predictions.append(output[0][0, -len(initial):].clone())

    def after_step(_pipeline, _step, _time, kwargs):
        final[:] = [kwargs["latents"][0].clone()]
        return {}

    hook = pipe.transformer.register_forward_hook(capture, with_kwargs=True)
    try:
        result = pipe(prompt=prompt, image=image, latents=initial[None].clone(), width=width, height=height,
                      output_resolution=area_side, num_inference_steps=steps, true_cfg_scale=1.0,
                      sigmas=qref.sample_sigmas(model, steps), callback_on_step_end=after_step).images[0]
    finally:
        hook.remove()

    out.mkdir(parents=True, exist_ok=True)
    image.save(out / "reference.png")
    (out / "ref.json").write_text(json.dumps({
        "prompt": prompt, "width": width, "height": height, "steps": steps,
        "reference_width": ref_width, "reference_height": ref_height,
        "image_slot_start": int(slots[0]), "image_slot_count": slots.numel(),
    }, indent=2) + "\n")
    save_file({k: v.contiguous() for k, v in {
        "reference_pixels": torch.from_numpy(np.array(image)),
        "reference_latents": packed_ref,
        "prompt_embeds": embeds[0],
        "latents_init": initial.clone(),
        "noise_pred_000": predictions[0],
        "latents_final": final[0],
        "rgba": torch.from_numpy(np.array(result.convert("RGBA"))),
    }.items()}, out / "ref.safetensors")
    return predictions[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    pipe = QwenImage21Pipeline.from_pretrained(str(args.model), torch_dtype=torch.float32,
                                              processor=qref.load_processor(args.model))
    pipe.set_progress_bar_config(disable=True)
    width, height, steps = 96, 64, 3
    initial = qref.pack_initial_latents(8128, pipe.transformer.config.in_channels, width, height)
    a = make_oracle(pipe, args.model, args.out / "a", 0, width, height, steps, initial)
    b = make_oracle(pipe, args.model, args.out / "b", 1, width, height, steps, initial)
    if torch.allclose(a, b, atol=1e-5, rtol=1e-5):
        raise SystemExit("changing the reference image did not affect the first denoising step")
    print("wrote two reference edits with identical noise and distinct predictions")


if __name__ == "__main__":
    main()
