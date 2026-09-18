import json
from pathlib import Path
from types import SimpleNamespace


class FamilyConfigError(ValueError):
    pass


class UnknownFamilyError(ValueError):
    pass


def _descriptor(kind):
    cpu_only = kind == "fixture_cpu"
    return SimpleNamespace(
        id=kind,
        display_name="Fixture CPU" if cpu_only else "Fixture CUDA",
        engine_artifact=("fixture-cpu" if cpu_only else
                         "fixture-v4" if kind == "deepseek_v4" else "fixture-cuda"),
        default_model_id=f"{kind}-model",
        has_gateway_adapter=True,
        supports_accelerator=not cpu_only,
        limits=SimpleNamespace(default_context=4096, max_context=8192,
                               context_env={"kimi": "K3_MAXT", "glm53": "GLM53_MAXT"}.get(kind, "CTX"),
                               default_max_output=1024, interactive_max_output=2048),
    )


def resolve_model(model_dir):
    path = Path(model_dir)
    try:
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise FamilyConfigError(f"cannot read config.json: {path}") from error
    kind = config.get("model_type")
    if kind not in {"fixture", "fixture_cpu", "kimi", "deepseek_v4", "glm53"}:
        raise UnknownFamilyError(f"unsupported model_type: {kind}")
    return SimpleNamespace(descriptor=_descriptor(kind), model_type=kind,
                           config=config, family_config=config, model_dir=str(path))


def display_for(resolved):
    return resolved.config.get("display_name", resolved.descriptor.display_name), ""
