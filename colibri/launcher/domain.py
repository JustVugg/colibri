"""Small immutable values shared by the launcher services and interface."""

from dataclasses import dataclass
from pathlib import Path


class LauncherError(ValueError):
    """A configuration problem that can be shown directly to the user."""


@dataclass(frozen=True)
class Installation:
    root: Path
    launcher: Path
    python: Path
    support_dir: Path
    version: str = "unknown"
    web_root: Path | None = None


@dataclass(frozen=True)
class LaunchOptions:
    mode: str = "web"
    compute: str = "auto"
    gpu_ids: tuple[int, ...] = ()
    ram_gb: int = 0
    vram_gb: float = 0.0
    context: int = 0
    max_tokens: int = 0
    port: int = 8000


@dataclass(frozen=True)
class ModelInfo:
    path: Path
    name: str
    family: str
    model_id: str
    engine: Path
    default_context: int
    max_context: int
    default_output: int


@dataclass(frozen=True)
class Check:
    id: str
    level: str
    message: str


@dataclass(frozen=True)
class GpuDevice:
    index: int
    name: str
    memory_gb: float = 0.0


@dataclass(frozen=True)
class Preflight:
    model: ModelInfo
    checks: tuple[Check, ...]
    gpus: tuple[GpuDevice, ...]
    cuda_available: bool
    cuda_reason: str
    plan: dict

    @property
    def can_start(self):
        return all(c.level != "fail" for c in self.checks)


@dataclass(frozen=True)
class LaunchSpec:
    argv: tuple[str, ...]
    env: dict[str, str]
    cwd: Path
    port: int
    model_id: str
    mode: str
    backend: str


@dataclass(frozen=True)
class ProcessEvent:
    kind: str
    text: str
