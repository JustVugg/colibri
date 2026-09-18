"""Discover an existing Colibri and a separate Python used to run it."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Iterator, Mapping, Sequence

from .domain import Installation, LauncherError


_DLL_DIRECTORY_LOCK = threading.RLock()
_PYTHON_PROBE = (
    "import json,sys; print(json.dumps({'executable':sys.executable,"
    "'version':list(sys.version_info[:3])}))"
)


def external_environment(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return an environment safe for launching non-bundled Python/processes.

    Frozen applications commonly carry Python search paths and, on Linux,
    replace ``LD_LIBRARY_PATH``. Those settings can make the user's Python load
    the launcher's bundled runtime instead of its own.
    """

    result = dict(os.environ if environ is None else environ)
    bundled = getattr(sys, "frozen", False) or "_MEIPASS2" in result
    for name in ("PYTHONHOME", "PYTHONPATH", "_MEIPASS2"):
        result.pop(name, None)
    # Keep the original-value marker: the adapter and subprocess boundary may
    # both sanitize this environment while sys.frozen remains true. Consuming
    # it would make a second pass mistake the restored user path for the bundle.
    original_library_path = result.get("LD_LIBRARY_PATH_ORIG")
    if original_library_path is not None:
        if original_library_path:
            result["LD_LIBRARY_PATH"] = original_library_path
        else:
            result.pop("LD_LIBRARY_PATH", None)
    elif bundled:
        result.pop("LD_LIBRARY_PATH", None)
    return {str(key): str(value) for key, value in result.items()}


@contextlib.contextmanager
def external_process_context() -> Iterator[None]:
    """Temporarily remove PyInstaller's Windows DLL search directory.

    The lock spans process creation, preventing another launcher worker from
    observing the temporary process-wide change. Long-running callers should
    create ``Popen`` inside this context and wait after leaving it.
    """

    with _DLL_DIRECTORY_LOCK:
        if os.name != "nt":
            yield
            return
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_directory = kernel32.GetDllDirectoryW
        get_directory.argtypes = [ctypes.c_uint32, ctypes.c_wchar_p]
        get_directory.restype = ctypes.c_uint32
        set_directory = kernel32.SetDllDirectoryW
        set_directory.argtypes = [ctypes.c_wchar_p]
        set_directory.restype = ctypes.c_int

        buffer = ctypes.create_unicode_buffer(32768)
        length = get_directory(len(buffer), buffer)
        previous = buffer.value if length else None
        if not set_directory(None):
            raise OSError(ctypes.get_last_error(), "could not clear the DLL search directory")
        try:
            yield
        finally:
            if not set_directory(previous):
                raise OSError(ctypes.get_last_error(), "could not restore the DLL search directory")


def run_external(
    argv: Sequence[str | os.PathLike[str]],
    *,
    environ: Mapping[str, str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
    timeout: float = 15.0,
) -> subprocess.CompletedProcess[str]:
    """Run a bounded external command without inheriting frozen-Python state."""

    command = [os.fspath(value) for value in argv]
    startupinfo = None
    creationflags = 0
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with external_process_context():
        process = subprocess.Popen(
            command,
            cwd=os.fspath(cwd) if cwd is not None else None,
            env=external_environment(environ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            startupinfo=startupinfo,
            creationflags=creationflags,
        )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        raise LauncherError(f"External command timed out after {timeout:g} seconds") from None
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _launcher_named(directory: Path) -> Path | None:
    for name in ("coli", "coli.py"):
        candidate = directory / name
        if candidate.is_file():
            return candidate.resolve()
    return None


def _layout(candidate: Path) -> tuple[Path, Path, Path] | None:
    candidate = candidate.expanduser().resolve()
    if candidate.is_file():
        launcher = candidate
        parent = launcher.parent
        if parent.name.lower() == "bin" and (parent.parent / "libexec" / "colibri").is_dir():
            return parent.parent, launcher, (parent.parent / "libexec" / "colibri").resolve()
        if parent.name.lower() == "c":
            return parent.parent, launcher, parent
        return parent, launcher, parent

    source_launcher = _launcher_named(candidate / "c")
    if source_launcher is not None:
        return candidate, source_launcher, (candidate / "c").resolve()
    installed_launcher = _launcher_named(candidate / "bin")
    installed_support = candidate / "libexec" / "colibri"
    if installed_launcher is not None and installed_support.is_dir():
        return candidate, installed_launcher, installed_support.resolve()
    direct_launcher = _launcher_named(candidate)
    if direct_launcher is not None:
        if candidate.name.lower() == "bin" and (candidate.parent / "libexec" / "colibri").is_dir():
            return candidate.parent, direct_launcher, (candidate.parent / "libexec" / "colibri").resolve()
        if candidate.name.lower() == "c":
            return candidate.parent, direct_launcher, candidate
        return candidate, direct_launcher, candidate
    return None


def _installation_candidate(folder: str | Path | None) -> tuple[Path, Path, Path]:
    if folder is not None:
        layout = _layout(Path(folder))
        if layout is None:
            raise LauncherError(
                f"This location is missing the Colibri launcher: {Path(folder).expanduser()}"
            )
        return layout

    if getattr(sys, "frozen", False):
        launcher_folder = Path(sys.executable).resolve().parent
        candidates = (launcher_folder, launcher_folder.parent)
    else:
        candidates = (Path.cwd(),)
    for candidate in candidates:
        layout = _layout(candidate)
        if layout is not None:
            return layout
    raise LauncherError(
        "Colibri was not found beside the launcher. Place the launcher folder inside "
        "your Colibri folder, or run the source launcher from the Colibri folder."
    )


def _python_command(value: str | Path) -> list[str]:
    text = os.fspath(value)
    if text.strip().lower() == "py -3":
        executable = shutil.which("py")
        if executable:
            return [executable, "-3"]
    path = Path(text).expanduser()
    if path.is_file():
        return [str(path.resolve())]
    found = shutil.which(text)
    return [found] if found else [str(path)]


def _resolve_python(python: str | Path | None) -> Path:
    commands: list[list[str]] = []
    if python is not None:
        commands.append(_python_command(python))
    else:
        virtual_env = os.environ.get("VIRTUAL_ENV")
        if virtual_env:
            relative = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
            commands.append([str(Path(virtual_env) / relative)])
        if not getattr(sys, "frozen", False):
            commands.append([sys.executable])
        if os.name == "nt" and shutil.which("py"):
            commands.append([shutil.which("py") or "py", "-3"])
        for name in ("python3", "python"):
            found = shutil.which(name)
            if found:
                commands.append([found])

    errors = []
    for command in commands:
        try:
            result = run_external([*command, "-c", _PYTHON_PROBE], timeout=8)
        except (LauncherError, OSError) as error:
            errors.append(str(error))
            continue
        try:
            payload = json.loads(result.stdout.strip())
            version = payload["version"]
            executable = Path(payload["executable"]).resolve()
            if (not isinstance(version, list) or len(version) < 2 or
                    any(isinstance(item, bool) or not isinstance(item, int) for item in version[:2])):
                raise ValueError("invalid version response")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            errors.append(f"{command[0]} did not identify itself as Python")
            continue
        if result.returncode != 0:
            errors.append(result.stderr.strip() or f"{command[0]} exited with an error")
            continue
        if tuple(version[:2]) < (3, 10):
            errors.append(f"{executable} is Python {version[0]}.{version[1]}; Python 3.10+ is required")
            continue
        return executable

    detail = f" ({'; '.join(errors[:2])})" if errors else ""
    raise LauncherError(f"A working external Python 3.10 or newer was not found{detail}.")


def find_installation(
    folder: str | Path | None = None,
    python: str | Path | None = None,
) -> Installation:
    """Validate a release, checkout, or installed-prefix Colibri layout."""

    root, launcher, support_dir = _installation_candidate(folder)
    if not (support_dir / "family_registry.py").is_file():
        raise LauncherError(
            f"Colibri support files are missing beside the launcher: {support_dir}"
        )
    interpreter = _resolve_python(python)
    version = "unknown"
    try:
        result = run_external([interpreter, launcher, "--version"], cwd=support_dir, timeout=8)
    except (LauncherError, OSError) as error:
        raise LauncherError(f"Colibri could not be started with the selected Python: {error}") from error
    if result.returncode != 0:
        reason = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise LauncherError(f"Colibri could not be started with the selected Python: {reason}")
    text = result.stdout.strip()
    if text.lower().startswith("colibri "):
        version = text.split(None, 1)[1].lstrip("v")

    web_root = next(
        (path.resolve() for path in (support_dir / "web" / "dist", root / "web" / "dist")
         if (path / "index.html").is_file()),
        None,
    )
    return Installation(root.resolve(), launcher.resolve(), interpreter, support_dir.resolve(),
                        version, web_root)
