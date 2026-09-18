"""Family-aware preflight and launch construction for installed Colibri."""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Mapping

from .domain import (
    Check,
    GpuDevice,
    Installation,
    LauncherError,
    LaunchOptions,
    LaunchSpec,
    ModelInfo,
    Preflight,
)
from .installation import external_environment, run_external


_MAX_JSON_BYTES = 1_048_576
_CHECK_LEVELS = {"pass", "warn", "fail", "skip"}
_SINGLE_GPU_ENV = {"deepseek_v4": "DSV4_CUDA_DEVICE", "inkling": "GPU_DEV"}
_CLEAR_ENV = {
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "COLI_API_KEY",
    "COLI_MODEL", "COLI_MODEL_ID", "COLI_ENGINE", "SNAP",
    "COLI_MODEL_MIRROR", "COLI_MODEL_DIRS", "COLI_ALLOWED_HOSTS",
    "CLUSTER_WORKERS", "CLUSTER_COORDINATOR", "CLUSTER_WORKER_PORT",
    "COLI_CORS_ORIGIN", "COLI_HIP_RUNTIME_DIR", "HIP_PATH",
    "RAM_GB", "CAP", "COLI_PLAN_CAP", "COLI_PROFILE_CAP", "PIN_GB",
}
_CPU_DISABLE = {
    "COLI_CUDA": "0",
    "COLI_CUDA_MTP": "0",
    "COLI_CUDA_PIPE": "0",
    "COLI_METAL": "0",
    "COLI_VULKAN": "0",
    "COLI_VK_DENSE": "0",
    "COLI_VK_ATTN": "0",
    "COLI_VK_EXPERTS": "0",
    "K3_CUDA": "0",
    "K3_VK": "0",
    "K3_VK_UP": "0",
    "K3_VULKAN": "0",
    "CUDA_DENSE": "0",
    "CUDA_EXPERT_GB": "0",
}
_ACCELERATOR_PREFIXES = (
    "COLI_CUDA", "COLI_GPU", "COLI_METAL", "COLI_VK", "COLI_VULKAN",
    "DSV4_CUDA", "K3_CUDA", "K3_METAL", "K3_VK",
)
_ACCELERATOR_EXACT = {
    "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL", "GPU_DEV", "NOGPU", "CUDA_DENSE",
    "CUDA_EXPERT_GB", "CUDA_EXPERT_LOAD_BALANCE", "CUDA_RAW_EXPERTS",
    "CUDA_RELEASE_HOST", "CUDA_RESERVE_GB", "K3_EXPERT_GB",
    "Q38_TRUNK_GPU", "V4_MTP_GPU_MIRRORS", "INK_METAL_MIN_S",
    "INK_METAL_SHARED", "VK_PROF",
}


def _controlled_environment(environ: Mapping[str, str] | None = None,
                            resource_env: tuple[str, ...] = ()) -> dict[str, str]:
    result = external_environment(environ)
    for name in tuple(result):
        if (name in _CLEAR_ENV or name in resource_env or name in _ACCELERATOR_EXACT or
                name.startswith(_ACCELERATOR_PREFIXES)):
            result.pop(name, None)
    return result


def _json_output(result: subprocess.CompletedProcess[str], label: str) -> dict[str, Any]:
    encoded = result.stdout.encode("utf-8", errors="replace")
    if len(encoded) > _MAX_JSON_BYTES:
        raise LauncherError(f"Colibri {label} returned too much data.")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        detail = result.stderr.strip() or "output was not JSON"
        raise LauncherError(f"Colibri {label} could not be read: {detail}") from error
    if not isinstance(payload, dict):
        raise LauncherError(f"Colibri {label} must be a JSON object.")
    return payload


def _probe_metadata(installation: Installation, model_path: Path) -> dict[str, Any]:
    probe_path = Path(__file__).with_name("probe.py")
    if not probe_path.is_file():
        raise LauncherError("The desktop launcher's Colibri metadata bridge is missing.")
    try:
        result = run_external(
            [installation.python, probe_path,
             "--support-dir", installation.support_dir,
             "--launcher", installation.launcher,
             "--model", model_path],
            cwd=installation.support_dir,
            environ=_controlled_environment(),
            timeout=20,
        )
    except OSError as error:
        raise LauncherError(f"Colibri metadata could not be inspected: {error}") from error
    payload = _json_output(result, "metadata")
    if payload.get("schema_version") != 1:
        raise LauncherError("This Colibri metadata schema is not supported by the desktop launcher.")
    if isinstance(payload.get("error"), str):
        raise LauncherError(f"Colibri could not inspect this model: {payload['error']}")
    required_strings = ("family", "name", "model_id", "engine", "cuda_reason")
    if any(not isinstance(payload.get(key), str) or not payload[key] for key in required_strings):
        raise LauncherError("Colibri returned incomplete model metadata.")
    limits = payload.get("limits")
    if not isinstance(limits, dict):
        raise LauncherError("Colibri returned incomplete model limits.")
    for key in ("default_context", "max_context", "default_output", "max_output"):
        value = limits.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise LauncherError("Colibri returned invalid model limits.")
    if limits["default_context"] > limits["max_context"]:
        raise LauncherError("Colibri returned invalid context limits.")
    if not all(isinstance(payload.get(key), bool)
               for key in ("gateway", "family_accelerator", "cuda_binary", "auto_tier")):
        raise LauncherError("Colibri returned invalid capability metadata.")
    resource_env = payload.get("resource_env")
    if (not isinstance(resource_env, list) or len(resource_env) > 32 or
            any(not isinstance(name, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", name)
                for name in resource_env)):
        raise LauncherError("Colibri returned invalid family resource variables.")
    gpus = payload.get("gpus")
    if not isinstance(gpus, list) or len(gpus) > 64:
        raise LauncherError("Colibri returned invalid GPU metadata.")
    for gpu in gpus:
        if (not isinstance(gpu, dict) or isinstance(gpu.get("index"), bool) or
                not isinstance(gpu.get("index"), int) or gpu["index"] < 0 or
                not isinstance(gpu.get("name"), str) or not gpu["name"] or
                isinstance(gpu.get("memory_gb"), bool) or
                not isinstance(gpu.get("memory_gb"), (int, float)) or
                not math.isfinite(gpu["memory_gb"]) or gpu["memory_gb"] < 0):
            raise LauncherError("Colibri returned invalid GPU metadata.")
    return payload


def _validate_options(options: LaunchOptions, metadata: dict[str, Any] | None = None) -> None:
    if options.mode not in {"web", "serve"}:
        raise LauncherError("App mode must be Web Chat or API Server.")
    if options.compute not in {"auto", "cpu", "cuda"}:
        raise LauncherError("Compute mode must be Automatic, CPU only, or NVIDIA CUDA.")
    if not isinstance(options.gpu_ids, tuple) or len(options.gpu_ids) > 64:
        raise LauncherError("GPU selection is invalid.")
    if (any(isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in options.gpu_ids) or len(set(options.gpu_ids)) != len(options.gpu_ids)):
        raise LauncherError("GPU selections must be unique non-negative device numbers.")
    for name, value in (("RAM", options.ram_gb), ("context", options.context),
                        ("maximum output", options.max_tokens)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise LauncherError(f"The {name} setting must be zero (automatic) or a positive number.")
    if options.ram_gb > 1_048_576:
        raise LauncherError("The RAM setting is outside the supported range.")
    if (isinstance(options.vram_gb, bool) or not isinstance(options.vram_gb, (int, float)) or
            not math.isfinite(options.vram_gb) or options.vram_gb < 0 or options.vram_gb > 1_048_576):
        raise LauncherError("The VRAM setting must be zero (automatic) or a valid positive number.")
    if isinstance(options.port, bool) or not isinstance(options.port, int) or not 1 <= options.port <= 65535:
        raise LauncherError("The server port must be between 1 and 65535.")
    if metadata is not None:
        limits = metadata["limits"]
        if options.context > limits["max_context"]:
            raise LauncherError(
                f"The context setting exceeds this model's limit of {limits['max_context']}."
            )
        if options.max_tokens > limits["max_output"]:
            raise LauncherError(
                f"The maximum output setting exceeds this model's limit of {limits['max_output']}."
            )


def _diagnostic_environment(compute: str, metadata: dict[str, Any],
                            gpu_ids: tuple[int, ...] = ()) -> dict[str, str]:
    return _launch_environment(None, compute, gpu_ids, metadata["family"], tuple(metadata["resource_env"]))


def _run_doctor(
    installation: Installation,
    model_path: Path,
    options: LaunchOptions,
    gpu_value: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    argv: list[str | os.PathLike[str]] = [
        installation.python, installation.launcher, "doctor", "--json",
        "--model", model_path, "--gpu", gpu_value,
    ]
    if options.ram_gb:
        argv.extend(("--ram", str(options.ram_gb)))
    if options.context:
        argv.extend(("--ctx", str(options.context)))
    if options.vram_gb:
        argv.extend(("--vram", str(options.vram_gb)))
    gpu_ids = (() if gpu_value in {"none", "auto"} else
               tuple(int(value) for value in gpu_value.split(",")))
    try:
        result = run_external(
            argv,
            environ=_diagnostic_environment("cpu" if gpu_value == "none" else "cuda", metadata, gpu_ids),
            cwd=installation.support_dir,
            timeout=45,
        )
    except OSError as error:
        raise LauncherError(f"Colibri diagnostics could not be started: {error}") from error
    report = _json_output(result, "diagnostics")
    if report.get("schema_version") != 1:
        raise LauncherError("This Colibri diagnostic schema is not supported by the desktop launcher.")
    if report.get("status") not in {"ok", "warning", "error"}:
        raise LauncherError("Colibri diagnostics returned an invalid status.")
    raw_checks = report.get("checks")
    if not isinstance(raw_checks, list) or len(raw_checks) > 512:
        raise LauncherError("Colibri diagnostics returned an invalid checks list.")
    for item in raw_checks:
        if (not isinstance(item, dict) or not isinstance(item.get("id"), str) or
                item.get("status") not in _CHECK_LEVELS or
                not isinstance(item.get("summary"), str)):
            raise LauncherError("Colibri diagnostics returned an invalid check.")
    if report.get("plan") is not None and not isinstance(report.get("plan"), dict):
        raise LauncherError("Colibri diagnostics returned an invalid placement plan.")
    return report


def _checks(report: dict[str, Any]) -> tuple[Check, ...]:
    return tuple(Check(item["id"], item["status"], item["summary"])
                 for item in report["checks"])


def _accelerator_failure(report: dict[str, Any]) -> str | None:
    checks = [item for item in report["checks"] if item["id"] == "accelerator.gpu"]
    if not checks:
        return "Colibri did not report a passing accelerator diagnostic."
    return next((item["summary"] for item in checks if item["status"] != "pass"), None)


def _device_reason(family: str, requested: tuple[int, ...], known_ids: set[int]) -> str | None:
    if not known_ids:
        return "No NVIDIA CUDA devices were detected."
    if requested and not set(requested).issubset(known_ids):
        return "One or more requested NVIDIA GPUs were not detected."
    selected = requested or tuple(sorted(known_ids))
    if family == "kimi" and selected != (0,):
        return "This Kimi engine supports only CUDA device 0; select that device explicitly."
    if family in _SINGLE_GPU_ENV and len(selected) != 1:
        return "This engine supports only one GPU; select a single device."
    return None


def inspect_model(
    installation: Installation,
    model_path: str | Path,
    options: LaunchOptions = LaunchOptions(),
) -> Preflight:
    """Resolve the installed family and obtain a fresh, versioned preflight."""

    path = Path(model_path).expanduser().resolve()
    metadata = _probe_metadata(installation, path)
    _validate_options(options, metadata)
    limits = metadata["limits"]
    model = ModelInfo(path, metadata["name"], metadata["family"], metadata["model_id"],
                      Path(metadata["engine"]), limits["default_context"],
                      limits["max_context"], limits["default_output"])
    gpus = tuple(GpuDevice(item["index"], item["name"], float(item["memory_gb"]))
                 for item in metadata["gpus"])
    known_ids = {gpu.index for gpu in gpus}
    cuda_capable = bool(metadata["family_accelerator"] and metadata["cuda_binary"] and gpus)
    cuda_verified = False
    cuda_reason = metadata["cuda_reason"]
    extra: list[Check] = []
    device_reason = _device_reason(model.family, options.gpu_ids, known_ids)
    if device_reason:
        cuda_capable = False
        cuda_reason = device_reason
    gpu_ids = options.gpu_ids
    if model.family in _SINGLE_GPU_ENV:
        gpu_ids = gpu_ids or tuple(sorted(known_ids))
    gpu_value = ",".join(str(value) for value in gpu_ids) or "auto"

    if not metadata["gateway"]:
        extra.append(Check("launcher.gateway", "fail",
                           "This model family is not available through Colibri's local server."))
    if options.mode == "web" and installation.web_root is None:
        extra.append(Check("launcher.web", "fail",
                           "Web Chat assets are missing from this Colibri installation; choose API Server."))

    if options.compute == "cpu":
        backend = "cpu"
        report = _run_doctor(installation, path, options, "none", metadata)
        cuda_reason = ("CPU only was selected. Choose NVIDIA CUDA to check GPU readiness."
                       if cuda_capable else f"CPU only was selected. {cuda_reason}")
    elif options.compute == "cuda":
        backend = "cuda"
        if not metadata["family_accelerator"]:
            extra.append(Check("launcher.cuda", "fail",
                               f"{model.name} does not support NVIDIA CUDA in this Colibri installation."))
        elif not metadata["cuda_binary"]:
            extra.append(Check("launcher.cuda", "fail", metadata["cuda_reason"]))
        if device_reason:
            extra.append(Check("launcher.cuda.devices", "fail", device_reason))
        report = _run_doctor(installation, path, options, gpu_value, metadata)
        failure = _accelerator_failure(report)
        if failure:
            cuda_reason = failure
            extra.append(Check("launcher.cuda.verified", "fail", failure))
        else:
            cuda_verified = cuda_capable
    else:
        if cuda_capable:
            gpu_report = _run_doctor(installation, path, options, gpu_value, metadata)
            failure = _accelerator_failure(gpu_report)
            if failure is None:
                backend = "cuda"
                report = gpu_report
                cuda_verified = True
                cuda_reason = "The selected engine and NVIDIA CUDA devices passed Colibri diagnostics."
            else:
                backend = "cpu"
                report = _run_doctor(installation, path, options, "none", metadata)
                cuda_reason = failure
                extra.append(Check("launcher.compute", "warn",
                                   f"Automatic selected CPU because NVIDIA CUDA was not verified: {failure}"))
        else:
            backend = "cpu"
            report = _run_doctor(installation, path, options, "none", metadata)
            reason = device_reason or (metadata["cuda_reason"] if metadata["family_accelerator"] else
                      f"{model.name} does not support NVIDIA CUDA in this Colibri installation.")
            cuda_reason = reason
            extra.append(Check("launcher.compute", "warn",
                               f"Automatic selected CPU because NVIDIA CUDA was not verified: {reason}"))

    plan = dict(report.get("plan") or {})
    plan.update({
        "backend": backend,
        "cuda_capable": cuda_capable,
        "auto_tier": metadata["auto_tier"],
        "max_output": limits["max_output"],
        "resource_env": tuple(metadata["resource_env"]),
    })
    return Preflight(model, (*_checks(report), *extra), gpus, cuda_verified,
                     cuda_reason, plan)


def _launch_environment(environ: Mapping[str, str] | None, backend: str,
                        gpu_ids: tuple[int, ...], family: str,
                        resource_env: tuple[str, ...]) -> dict[str, str]:
    result = _controlled_environment(environ, resource_env)
    if backend == "cpu":
        result.update(_CPU_DISABLE)
        result["COLI_GPU"] = "none"
        result.pop("COLI_GPUS", None)
        # These engines do not read COLI_CUDA: V4 defaults to enabled,
        # and Inkling disables CUDA only when NOGPU is present.
        if family == "deepseek_v4":
            result["DSV4_CUDA"] = "0"
        elif family == "inkling":
            result["NOGPU"] = "1"
    else:
        for name in ("COLI_METAL", "COLI_VULKAN", "K3_VK", "K3_VULKAN"):
            result[name] = "0"
        result["COLI_CUDA"] = "1"
        if family == "kimi":
            result["K3_CUDA"] = "1"
        elif family == "deepseek_v4":
            result["DSV4_CUDA"] = "1"
        if family in _SINGLE_GPU_ENV and len(gpu_ids) == 1:
            result[_SINGLE_GPU_ENV[family]] = str(gpu_ids[0])
        if gpu_ids:
            result["COLI_GPUS"] = ",".join(str(value) for value in gpu_ids)
            result.pop("COLI_GPU", None)
        else:
            result.pop("COLI_GPUS", None)
            result.pop("COLI_GPU", None)
    return result


def build_launch(
    installation: Installation,
    result: Preflight,
    options: LaunchOptions,
    environ: dict | None = None,
) -> LaunchSpec:
    """Build an explicit local argv/environment pair without executing it."""

    metadata = {"limits": {"max_context": result.model.max_context,
                            "max_output": result.plan.get("max_output", result.model.default_output)}}
    _validate_options(options, metadata)
    if not result.can_start:
        raise LauncherError("Resolve the failed preflight checks before starting Colibri.")
    backend = result.plan.get("backend")
    if backend not in {"cpu", "cuda"}:
        raise LauncherError("The preflight does not contain a usable compute plan.")
    if options.compute == "cpu" and backend != "cpu":
        raise LauncherError("The preflight is stale for the selected CPU compute mode.")
    if options.compute == "cuda" and backend != "cuda":
        raise LauncherError("The preflight is stale for the selected CUDA compute mode.")
    if backend == "cuda" and not result.cuda_available:
        raise LauncherError("NVIDIA CUDA must pass preflight checks before starting.")
    gpu_ids = options.gpu_ids
    if backend == "cuda" and (result.model.family in _SINGLE_GPU_ENV or result.model.family == "kimi"):
        known_ids = {gpu.index for gpu in result.gpus}
        reason = _device_reason(result.model.family, gpu_ids, known_ids)
        if reason:
            raise LauncherError(reason)
        if result.model.family in _SINGLE_GPU_ENV:
            gpu_ids = gpu_ids or tuple(sorted(known_ids))

    argv = [str(installation.python), str(installation.launcher), options.mode,
            "--model", str(result.model.path),
            "--host", "127.0.0.1", "--port", str(options.port),
            "--model-id", result.model.model_id,
            "--gpu", "none" if backend == "cpu" else
            (",".join(str(value) for value in gpu_ids) or "auto")]
    if options.mode == "web":
        argv.append("--no-browser")
    if options.ram_gb:
        argv.extend(("--ram", str(options.ram_gb)))
    if options.vram_gb:
        argv.extend(("--vram", str(options.vram_gb)))
    if options.context:
        argv.extend(("--ctx", str(options.context)))
    if options.max_tokens:
        argv.extend(("--ngen", str(options.max_tokens)))
    if result.plan.get("auto_tier") is True:
        argv.append("--auto-tier")
    environment = _launch_environment(environ, backend, gpu_ids, result.model.family,
                                      tuple(result.plan.get("resource_env", ())))
    return LaunchSpec(tuple(argv), environment, installation.support_dir, options.port,
                      result.model.model_id, options.mode, backend)


def command_preview(spec: LaunchSpec) -> str:
    """Return a copyable command containing no environment values or secrets."""

    if os.name == "nt":
        return "& " + " ".join("'" + argument.replace("'", "''") + "'" for argument in spec.argv)
    return shlex.join(spec.argv)
