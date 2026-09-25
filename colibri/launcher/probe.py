#!/usr/bin/env python3
"""Isolated, standard-library metadata bridge for an installed Colibri.

This file is executed by the user's external Python. It must remain free of
imports from the launcher package and print exactly one bounded JSON object.
"""

import argparse
import inspect
import json
import runpy
import subprocess
import sys
from pathlib import Path


def _nvidia_devices():
    try:
        from resource_plan import discover_gpus
        found = discover_gpus()
    except (ImportError, OSError, ValueError, TypeError):
        return []
    devices = []
    for value in found if isinstance(found, list) else []:
        if not isinstance(value, dict):
            continue
        name = value.get("name")
        index = value.get("index")
        if not isinstance(name, str) or isinstance(index, bool) or not isinstance(index, int):
            continue
        lowered = name.lower()
        markers = ("nvidia", "geforce", "quadro", "tesla", " rtx", "a100", "h100", "h200")
        if not any(marker in lowered for marker in markers):
            continue
        total = value.get("total_bytes", 0)
        if isinstance(total, bool) or not isinstance(total, (int, float)) or total < 0:
            total = 0
        devices.append({"index": index, "name": name, "memory_gb": round(total / (1024 ** 3), 2)})
    return devices


def _nvidia_runtime(engine, family_id):
    """Disambiguate the installed CLI's CUDA-or-HIP capability predicate."""
    if sys.platform == "linux":
        try:
            linked = subprocess.run(["ldd", str(engine)], capture_output=True, text=True, timeout=3)
        except (OSError, subprocess.SubprocessError):
            return False, "The selected engine's NVIDIA CUDA linkage could not be verified."
        if "libamdhip64" in linked.stdout:
            return False, "The selected engine uses AMD HIP, which this launcher does not support."
        supported = linked.returncode == 0 and any(
            "libcudart" in line and "not found" not in line for line in linked.stdout.splitlines())
    elif sys.platform == "win32":
        try:
            image = engine.read_bytes()
        except OSError:
            return False, "The selected engine could not be inspected for NVIDIA CUDA support."
        if b"coli_hip.dll" in image:
            return False, "The selected engine uses AMD HIP, which this launcher does not support."
        if family_id == "deepseek_v4":
            built = b"[DSV4 CUDA]" in image
            runtime_names = ("coli_cuda_dsv4_dg.dll", "coli_cuda_dsv4.dll")
        else:
            built = b"[CUDA] mode: routed experts" in image and b"coli_cuda.dll" in image
            runtime_names = ("coli_cuda.dll",)
        if not built:
            return False, "The selected engine has no verified NVIDIA CUDA support."
        if not any((engine.parent / name).is_file() for name in runtime_names):
            names = " or ".join(runtime_names)
            return False, f"A required CUDA DLL is missing beside the selected engine: {names}."
        supported = True
    else:
        supported = False
    return supported, ("The selected engine has verified NVIDIA CUDA runtime evidence." if supported else
                       "The selected engine's NVIDIA CUDA runtime could not be verified.")


def _cuda_support(namespace, engine, family_id, model=None):
    dedicated = family_id == "deepseek_v4"
    function = namespace.get("dsv4_cuda_available" if dedicated else "cuda_binary")
    if not callable(function):
        return False, "This Colibri version cannot verify CUDA for the selected engine."
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return False, "This Colibri version exposes an unsupported CUDA probe."
    try:
        argument = str(model) if dedicated else str(engine)
        signature.bind(argument)
    except TypeError:
        if dedicated:
            return False, "This Colibri version can only verify CUDA for its default engine."
        try:
            signature.bind()
        except TypeError:
            return False, "This Colibri version exposes an unsupported CUDA probe."
        if family_id != "glm":
            # Older CLIs inspect their default GLM binary even when launching
            # another family. Both the selected engine and that CLI gate must
            # pass; a GPU diagnostic alone cannot prove the CLI will launch.
            runtime_supported, runtime_reason = _nvidia_runtime(engine, family_id)
            if not runtime_supported:
                return False, runtime_reason
            try:
                legacy_supported = bool(function())
            except (OSError, ValueError, TypeError):
                legacy_supported = False
            if not legacy_supported:
                return False, ("This older Colibri CLI cannot launch CUDA for the selected engine. "
                               "Update Colibri to a version with model-specific CUDA checks.")
            return True, runtime_reason
        try:
            supported = bool(function())
        except (OSError, ValueError, TypeError):
            supported = False
    else:
        try:
            supported = bool(function(argument))
        except (OSError, ValueError, TypeError):
            supported = False
    if not supported:
        runtime_supported, runtime_reason = _nvidia_runtime(engine, family_id)
        if not runtime_supported:
            return False, runtime_reason
        return False, "The selected engine is CPU-only or its CUDA runtime is unavailable."
    return _nvidia_runtime(engine, family_id)


def _resource_environment(descriptor):
    """Inputs consumed by resource_request and the selected family's launcher."""
    names = [getattr(descriptor.limits, "context_env", "")]
    names.extend({"glm53": ("GLM53_EXPERT_GB",),
                  "kimi": ("K3_EXPERT_GB", "K3_MMAP"),
                  "deepseek_v4": ("V4_MTP_GB",)}.get(descriptor.id, ()))
    return [name for name in names if name]


def collect(support_dir, launcher, model):
    support = Path(support_dir).resolve()
    launcher_path = Path(launcher).resolve()
    model_path = Path(model).resolve()
    sys.path.insert(0, str(support))
    namespace = runpy.run_path(str(launcher_path), run_name="_colibri_desktop_probe")
    try:
        import family_registry
        resolve_model = family_registry.resolve_model
    except (ImportError, AttributeError) as error:
        raise ValueError("the installed Colibri family registry could not be loaded") from error

    resolved = resolve_model(model_path)
    descriptor = resolved.descriptor
    engine_for = namespace.get("engine_for")
    if not callable(engine_for):
        raise ValueError("this Colibri version cannot resolve a model-specific engine")
    engine = Path(engine_for(str(model_path))).resolve()
    limits = descriptor.limits
    name = descriptor.display_name
    display_for = getattr(family_registry, "display_for", None)
    try:
        displayed = display_for(resolved) if callable(display_for) else None
        if isinstance(displayed, tuple) and displayed and isinstance(displayed[0], str):
            name = displayed[0]
    except (AttributeError, KeyError, TypeError, ValueError):
        pass

    cuda_binary, cuda_reason = _cuda_support(namespace, engine, descriptor.id, model_path)
    family_accelerator = getattr(descriptor, "supports_accelerator", False)
    if family_accelerator is not True:
        cuda_binary = False
        cuda_reason = f"{name} does not support an accelerator in this Colibri installation."
    gateway = getattr(descriptor, "has_gateway_adapter", False) is True
    try:
        launcher_text = launcher_path.read_text(encoding="utf-8", errors="ignore")[:2_000_000]
    except OSError:
        launcher_text = ""
    return {
        "schema_version": 1,
        "family": descriptor.id,
        "name": name,
        "model_id": descriptor.default_model_id,
        "engine": str(engine),
        "limits": {
            "default_context": limits.default_context,
            "max_context": limits.max_context,
            "default_output": getattr(limits, "interactive_max_output", limits.default_max_output),
            "max_output": getattr(limits, "interactive_max_output", limits.default_max_output),
        },
        "gateway": gateway,
        "family_accelerator": family_accelerator,
        "cuda_binary": cuda_binary,
        "cuda_reason": cuda_reason,
        "gpus": _nvidia_devices(),
        "resource_env": _resource_environment(descriptor),
        "auto_tier": "--auto-tier" in launcher_text,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--support-dir", required=True)
    parser.add_argument("--launcher", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    try:
        payload = collect(args.support_dir, args.launcher, args.model)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        payload = {"schema_version": 1, "error": str(error)}
    encoded = json.dumps(payload, ensure_ascii=False)
    if len(encoded.encode("utf-8")) > 262_144:
        encoded = json.dumps({"schema_version": 1, "error": "metadata response was too large"})
    print(encoded)


if __name__ == "__main__":
    main()
