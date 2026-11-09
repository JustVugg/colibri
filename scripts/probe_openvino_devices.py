"""Compile and run a tiny conv model on every OpenVINO device present.

Confirms which accelerators (Intel NPU / Arc iGPU / NVIDIA dGPU) can really
execute work, not just enumerate.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import openvino as ov

OUT = Path(r"D:\Colibri\results\openvino_devices.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

DEVICES = ["CPU", "GPU.0", "GPU.1", "NPU"]


def build_model() -> ov.Model:
    """Small conv net: 1x3x64x64 -> 16 channels."""
    x = ov.opset13.parameter([1, 3, 64, 64], ov.Type.f32, name="x")
    w = ov.opset13.constant(np.random.RandomState(0).randn(16, 3, 3, 3).astype(np.float32) * 0.1)
    conv = ov.opset13.convolution(x, w, [1, 1], [0, 0], [0, 0], [1, 1], "explicit")
    relu = ov.opset13.relu(conv)
    return ov.Model([relu], [x], "probe_conv")


def main() -> None:
    core = ov.Core()
    report: dict[str, object] = {
        "openvino": ov.__version__,
        "available_devices": core.available_devices,
        "results": {},
    }
    print("openvino", ov.__version__)
    print("available_devices", core.available_devices)

    model = build_model()
    data = {"x": np.random.RandomState(1).randn(1, 3, 64, 64).astype(np.float32)}

    for dev in DEVICES:
        entry: dict[str, object] = {}
        if dev not in core.available_devices:
            entry["status"] = "not-present"
            report["results"][dev] = entry
            print(f"{dev:6s} not-present")
            continue
        try:
            name = core.get_property(dev, "FULL_DEVICE_NAME")
            entry["name"] = name
        except Exception as exc:  # noqa: BLE001
            entry["name"] = f"<{exc}>"

        try:
            t0 = time.time()
            compiled = core.compile_model(model, dev)
            entry["compile_s"] = round(time.time() - t0, 3)
            t1 = time.time()
            res = compiled(data)
            entry["first_infer_s"] = round(time.time() - t1, 3)
            out = np.asarray(res[compiled.output(0)])
            entry["output_shape"] = list(out.shape)
            entry["output_mean"] = float(out.mean())
            entry["status"] = "ok"
            print(f"{dev:6s} OK  compile={entry['compile_s']}s infer={entry['first_infer_s']}s "
                  f"shape={out.shape} mean={out.mean():.4f}  ({entry['name']})")
        except Exception as exc:  # noqa: BLE001
            entry["status"] = "fail"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            print(f"{dev:6s} FAIL {type(exc).__name__}: {str(exc)[:200]}")

        report["results"][dev] = entry

    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    ok = [d for d, v in report["results"].items() if v.get("status") == "ok"]
    print("working devices:", ok)


if __name__ == "__main__":
    main()
