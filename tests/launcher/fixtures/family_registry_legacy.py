import json
from pathlib import Path
from types import SimpleNamespace


class FamilyConfigError(ValueError):
    pass


class UnknownFamilyError(ValueError):
    pass


def resolve_model(model_dir):
    path = Path(model_dir)
    try:
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise FamilyConfigError(f"cannot read config.json: {path}") from error
    kind = config.get("model_type")
    if kind != "fixture":
        raise UnknownFamilyError(f"unsupported model_type: {kind}")
    descriptor = SimpleNamespace(
        id="fixture", display_name="Legacy Fixture", engine_artifact="fixture-cuda",
        default_model_id="fixture-model", has_gateway_adapter=True,
        supports_accelerator=True,
        limits=SimpleNamespace(default_context=4096, max_context=8192,
                               default_max_output=1024, interactive_max_output=2048),
    )
    return SimpleNamespace(descriptor=descriptor, model_type=kind, config=config,
                           family_config=config, model_dir=str(path))
