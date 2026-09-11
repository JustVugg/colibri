#!/usr/bin/env python3
"""Cross-check Colibri's complete Gemma 4 text graph against llama.cpp."""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import tempfile

import numpy as np

from validate_gemma4_llama_layer0 import metric


def run(command, environment=None):
    result = subprocess.run(
        [str(part) for part in command], env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode:
        detail = "\n".join((result.stdout, result.stderr)).strip()
        raise RuntimeError(
            f"command failed ({result.returncode}): "
            f"{' '.join(map(str, command))}\n{detail[-4000:]}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=pathlib.Path)
    parser.add_argument("oracle", type=pathlib.Path)
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("packed", type=pathlib.Path)
    parser.add_argument("--llama-bin", type=pathlib.Path,
                        help="directory containing llama.cpp runtime DLLs")
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--chat-file", type=pathlib.Path,
                        help="UTF-8 user message to wrap in the canonical chat frame")
    arguments = parser.parse_args()

    executable = arguments.executable.resolve()
    oracle = arguments.oracle.resolve()
    model = arguments.model.resolve()
    packed = arguments.packed.resolve()
    environment = os.environ.copy()
    if arguments.llama_bin:
        environment["PATH"] = (str(arguments.llama_bin.resolve()) +
                               os.pathsep + environment.get("PATH", ""))

    with tempfile.TemporaryDirectory(prefix="gemma4-full-") as directory:
        temporary = pathlib.Path(directory)
        if arguments.chat_file:
            chat_file = arguments.chat_file.resolve()
            user = chat_file.read_bytes().strip(b" \t\r\n")
            rendered = (b"<bos><|turn>user\n" + user +
                        b"<turn|>\n<|turn>model\n"
                        b"<|channel>thought\n<channel|>")
            rendered_path = temporary / "rendered-chat.txt"
            rendered_path.write_bytes(rendered)
            oracle_command = [oracle, "--prompt-special-file", model,
                              rendered_path, temporary]
            native_command = [executable, "next-token-file", model, packed,
                              chat_file, "--chat"]
        else:
            oracle_command = [oracle, model, arguments.prompt, temporary]
            native_command = [executable, "next-token", model, packed,
                              arguments.prompt]

        oracle_result = run(oracle_command, environment=environment)
        for line in oracle_result.stdout.splitlines():
            if line.startswith(("captured ", "tokens:")):
                print(line)

        native_residual = temporary / "colibri-final-residual.f32"
        native_logits = temporary / "colibri-logits.f32"
        native_command += ["--top", "10", "--residual-f32", native_residual,
                           "--logits-f32", native_logits]
        native_result = run(native_command)
        print(native_result.stdout.rstrip())

        actual_residual = np.fromfile(native_residual, dtype="<f4")
        actual_logits = np.fromfile(native_logits, dtype="<f4")
        reference_residual = np.fromfile(
            temporary / "llama-final-residual.f32", dtype="<f4")
        reference_logits = np.fromfile(
            temporary / "llama-logits.f32", dtype="<f4")
        results = {
            "final_residual": metric(actual_residual, reference_residual),
            "logits": metric(actual_logits, reference_logits),
        }
        actual_top = np.argsort(actual_logits)[-10:][::-1]
        reference_top = np.argsort(reference_logits)[-10:][::-1]

        print("\nboundary                 max_abs          rms       cosine")
        for name, values in results.items():
            maximum, rms, cosine = values
            print(f"{name:22s} {maximum:12.6g} {rms:12.6g} {cosine:12.9f}")
        print("Colibri top-10:", " ".join(map(str, actual_top)))
        print("llama.cpp top-10:", " ".join(map(str, reference_top)))

        if results["final_residual"][2] < 0.999 or results["logits"][2] < 0.999:
            raise SystemExit("full-model comparison exceeded cosine tolerance")
        if actual_top[0] != reference_top[0]:
            raise SystemExit("Colibri and llama.cpp selected different top-1 tokens")
        print("llama.cpp full-model oracle: PASS")


if __name__ == "__main__":
    main()
