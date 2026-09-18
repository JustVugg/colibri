"""Versioned, atomic settings storage for the desktop launcher."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from .domain import LauncherError


_OPTION_KEYS = {
    "mode", "compute", "gpu_ids", "ram_gb", "vram_gb",
    "context", "max_tokens", "port",
}


def _empty_settings() -> dict[str, Any]:
    return {"version": 1, "installation": "", "python": "",
            "models": [], "selected_model": ""}


def default_settings_path() -> Path:
    """Return the current user's platform-appropriate settings path."""

    if sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return root / "Colibri" / "launcher.json"
    root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return root / "colibri" / "launcher.json"


def _string(value: Any, label: str, maximum: int) -> None:
    if not isinstance(value, str) or len(value) > maximum:
        raise LauncherError(f"Invalid launcher settings: {label} must be a string.")


def _integer(value: Any, label: str, minimum: int, maximum: int) -> None:
    if (isinstance(value, bool) or not isinstance(value, int) or
            not minimum <= value <= maximum):
        raise LauncherError(f"Invalid launcher settings: {label} is outside its supported range.")


def _validate_options(options: Any) -> None:
    if not isinstance(options, dict) or set(options) != _OPTION_KEYS:
        raise LauncherError("Invalid launcher settings: a model options object is incomplete.")
    if not isinstance(options["mode"], str) or options["mode"] not in {"web", "serve"}:
        raise LauncherError("Invalid launcher settings: app mode is not recognized.")
    if (not isinstance(options["compute"], str) or
            options["compute"] not in {"auto", "cpu", "cuda"}):
        raise LauncherError("Invalid launcher settings: compute mode is not recognized.")
    gpu_ids = options["gpu_ids"]
    if (not isinstance(gpu_ids, (list, tuple)) or len(gpu_ids) > 64 or
            any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids)):
        raise LauncherError("Invalid launcher settings: GPU selections are invalid.")
    _integer(options["ram_gb"], "RAM", 0, 1_048_576)
    _integer(options["context"], "context", 0, 1_048_576)
    _integer(options["max_tokens"], "maximum output", 0, 1_048_576)
    _integer(options["port"], "port", 1, 65535)
    vram = options["vram_gb"]
    if (isinstance(vram, bool) or not isinstance(vram, (int, float)) or
            not math.isfinite(vram) or not 0 <= vram <= 1_048_576):
        raise LauncherError("Invalid launcher settings: VRAM is outside its supported range.")


def _validate(data: Any) -> None:
    if not isinstance(data, dict) or set(data) - {"theme"} != {
            "version", "installation", "python", "models", "selected_model"}:
        raise LauncherError("Invalid launcher settings: the version 1 document is incomplete.")
    if isinstance(data["version"], bool) or data["version"] != 1:
        raise LauncherError("Invalid launcher settings: unsupported settings version.")
    if "theme" in data and (not isinstance(data["theme"], str) or
                            data["theme"] not in {"light", "dark"}):
        raise LauncherError("Invalid launcher settings: theme is not recognized.")
    _string(data["installation"], "installation", 32768)
    _string(data["python"], "Python", 32768)
    _string(data["selected_model"], "selected model", 32768)
    models = data["models"]
    if not isinstance(models, list) or len(models) > 1000:
        raise LauncherError("Invalid launcher settings: model library is invalid or too large.")
    seen = set()
    for model in models:
        if not isinstance(model, dict) or set(model) != {"path", "name", "options"}:
            raise LauncherError("Invalid launcher settings: a model entry is incomplete.")
        _string(model["path"], "model path", 32768)
        _string(model["name"], "model name", 512)
        if not model["path"] or model["path"] in seen:
            raise LauncherError("Invalid launcher settings: model paths must be non-empty and unique.")
        seen.add(model["path"])
        _validate_options(model["options"])
    if data["selected_model"] and data["selected_model"] not in seen:
        raise LauncherError("Invalid launcher settings: selected model is not in the library.")


def _preserve_corrupt(path: Path) -> Path | None:
    suffix = f".corrupt-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    destination = path.with_name(path.name + suffix)
    try:
        os.replace(path, destination)
    except OSError:
        return None
    return destination


def load_settings(path: Path | None = None) -> tuple[dict, str | None]:
    """Load validated settings, preserving malformed input for recovery."""

    target = default_settings_path() if path is None else Path(path)
    if not target.exists():
        return _empty_settings(), None
    try:
        if target.stat().st_size > 4 * 1024 * 1024:
            raise LauncherError("Invalid launcher settings: file is too large.")
        data = json.loads(target.read_text(encoding="utf-8"))
        _validate(data)
        return data, None
    except (OSError, UnicodeError, json.JSONDecodeError, LauncherError) as error:
        preserved = _preserve_corrupt(target)
        if preserved is not None:
            warning = (f"Saved launcher settings were invalid and were preserved as "
                       f"{preserved.name}. Defaults were restored.")
        else:
            warning = (f"Saved launcher settings were invalid ({error}) and could not be moved. "
                       "Defaults were restored.")
        return _empty_settings(), warning


def save_settings(data: dict, path: Path | None = None) -> None:
    """Validate and atomically replace the settings document."""

    _validate(data)
    target = default_settings_path() if path is None else Path(path)
    temporary_name = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n", prefix=target.name + ".",
                suffix=".tmp", dir=target.parent, delete=False) as stream:
            temporary_name = stream.name
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
    except OSError as error:
        if temporary_name:
            try:
                Path(temporary_name).unlink()
            except OSError:
                pass
        raise LauncherError(f"Launcher settings could not be saved: {error}") from error
