#!/usr/bin/env python3
"""Compare the CPU encoder and decoder with the pinned Diffusers reference.

Install requirements-qwenimage-tiny.txt first. Run via make
qwenimage-encoder-check PYTHON=/path/to/venv/bin/python, or directly with
--report RESULT.json. The runner uses random tiny weights without model
downloads and removes temporary fixtures on success or failure.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time


DIFFUSERS_COMMIT = "80c7ed262aeffbeb43ef13ae04baeb9b84515a69"
SOURCE = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="also save structured results here")
    parser.add_argument("--sanitizers", action="store_true", help="also run encoder checks with ASan/UBSan (clang or gcc)")
    args = parser.parse_args()
    result = {"status": "failed", "weights": "tiny-random", "device": "cpu",
              "diffusers_commit": DIFFUSERS_COMMIT, "checks": []}
    env = dict(os.environ, OMP_NUM_THREADS="2", HF_HUB_OFFLINE="1",
               ASAN_OPTIONS="detect_leaks=0", UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")

    def run(name, command, timeout=120):
        start = time.monotonic()
        check = {"name": name, "command": [str(x) for x in command]}
        result["checks"].append(check)
        try:
            process = subprocess.run(command, cwd=SOURCE, env=env, text=True,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
            check.update(returncode=process.returncode, output=process.stdout)
            if process.returncode:
                raise RuntimeError(f"{name} failed ({process.returncode})")
        finally:
            check["seconds"] = round(time.monotonic() - start, 3)

    try:
        direct = json.loads(importlib.metadata.distribution("diffusers").read_text("direct_url.json") or "{}")
        if direct.get("vcs_info", {}).get("commit_id") != DIFFUSERS_COMMIT:
            raise RuntimeError("install tools/requirements-qwenimage-tiny.txt: pinned Diffusers oracle required")
        result["versions"] = {name: importlib.metadata.version(name) for name in ("torch", "diffusers", "safetensors")}
        suffix = ".exe" if os.name == "nt" else ""
        encoder = SOURCE / "tests" / ("test_qwenimage_encoder" + suffix)
        decoder = SOURCE / "tests" / ("test_qwenimage_vae" + suffix)
        run("build", ["make", str(encoder.relative_to(SOURCE)), str(decoder.relative_to(SOURCE))])
        run("unit", [encoder])
        with tempfile.TemporaryDirectory(prefix="colibri-qwen-encoder-") as directory:
            root = Path(directory)
            if args.sanitizers:
                encoder = root / ("encoder-asan" + suffix)
                run("build_sanitizers", shlex.split(os.environ.get("CC", "cc")) + [
                    "-O1", "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
                    "-Wno-unknown-pragmas", "tests/test_qwenimage_encoder.c", "-lm", "-o", encoder])
                run("unit_sanitizers", [encoder])
            run("official_encoder_fixture", [sys.executable, "tools/make_qwenimage_encoder_tiny.py", root])
            run("encoder_oracle", [encoder, root])
            # Decoder shares the kernels but must retain its existing behavior.
            run("official_decoder_fixture", [sys.executable, "tools/make_qwenimage_vae_tiny.py", root,
                "--vae", root / "vae", "--threads", "2"])
            run("decoder_regression", [decoder, root])

            from safetensors.torch import load_file, save_file
            cfg = json.loads((root / "vae/config.json").read_text())
            weights = load_file(root / "vae/diffusion_pytorch_model.safetensors")
            # Decoder and temporal conv weights are not needed for a single image.
            encoder_weights = {k: v for k, v in weights.items()
                               if k.startswith(("encoder.", "quant_conv.")) and ".time_conv." not in k}
            isolated = root / "encoder_only"
            isolated.mkdir()
            (isolated / "config.json").write_text(json.dumps(cfg))
            save_file(encoder_weights, str(isolated / "model.safetensors"))
            run("encoder_only_checkpoint", [encoder, "--vae", isolated, "--ref", root / "encoder_ref"])
            mutations = {
                "non_residual": {"is_residual": False},
                "patchified": {"patch_size": 2},
                "zero_std": {"latents_std": [0] * cfg["z_dim"]},
                "wrong_normalization_length": {"latents_mean": [0]},
                "fractional_dimension": {"base_dim": 8.5},
                "wrong_temporal_length": {"temperal_downsample": []},
                "wrong_channel_groups": {"dim_mult": [3, 2, 4, 8, 8]},
            }
            for name, changes in mutations.items():
                (isolated / "config.json").write_text(json.dumps(dict(cfg, **changes)))
                run("reject_" + name, [encoder, "--reject", isolated])
            (isolated / "config.json").write_text(json.dumps(cfg))
            del encoder_weights["encoder.down_blocks.2.resnets.1.conv2.weight"]
            save_file(encoder_weights, str(isolated / "model.safetensors"))
            run("reject_missing_weight", [encoder, "--reject", isolated])
        result["status"] = "passed"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    output = json.dumps(result, indent=2)
    if args.report:
        args.report.write_text(output + "\n")
    print(output)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
