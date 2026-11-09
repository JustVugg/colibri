"""Report the real memory budget OpenVINO reports for each accelerator.

Answers the question "how much memory can the NPU / Arc iGPU / RTX 4060
actually address?" instead of guessing from marketing numbers.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import openvino as ov

OUT = Path(r"D:\Colibri\results\device_memory.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

# Property names vary by plugin; try the ones that exist.
CANDIDATES = [
    "FULL_DEVICE_NAME",
    "DEVICE_TYPE",
    "DEVICE_ARCHITECTURE",
    "DEVICE_ID",
    "DEVICE_LUID",
    "DEVICE_MEMORY_SIZE",
    "DEVICE_LOCAL_MEM_SIZE",
    "DEVICE_TOTAL_MEM_SIZE",
    "DEVICE_ALLOC_MEM_SIZE",
    "DEVICE_MAX_MEM_ALLOC_SIZE",
    "NPU_DEVICE_MEMORY_SIZE",
    "NPU_DEVICE_MEMORY_SIZE_FREE",
    "GPU_DEVICE_TOTAL_MEM_SIZE",
    "GPU_DEVICE_FREE_MEM_SIZE",
    "OPTIMIZATION_CAPABILITIES",
    "RANGE_FOR_ASYNC_INFER_REQUESTS",
    "SUPPORTED_PROPERTIES",
]

DEVICES = ["CPU", "GPU.0", "GPU.1", "NPU"]


def main() -> None:
    core = ov.Core()
    report: dict[str, object] = {"available_devices": core.available_devices, "devices": {}}

    print("=" * 78)
    print("OpenVINO device memory report")
    print("=" * 78)

    for dev in DEVICES:
        if dev not in core.available_devices:
            print(f"\n{dev}: NOT PRESENT")
            continue
        info: dict[str, object] = {}
        print(f"\n{dev}")
        for prop in CANDIDATES:
            try:
                val = core.get_property(dev, prop)
            except Exception:
                continue
            if prop == "SUPPORTED_PROPERTIES":
                info[prop] = list(map(str, val))
                continue
            if isinstance(val, int) and ("MEM" in prop):
                info[prop] = val
                print(f"  {prop:34s} {val:>14,} bytes  ({val / 1e9:.2f} GB)")
            else:
                info[prop] = str(val)
                print(f"  {prop:34s} {val}")
        report["devices"][dev] = info

    du = shutil.disk_usage("D:/")
    os_ = shutil.disk_usage("C:/")
    import psutil  # optional

    report["system"] = {
        "D_free_gb": round(du.free / 1e9, 2),
        "C_free_gb": round(os_.free / 1e9, 2),
    }
    print(f"\nD: free {du.free / 1e9:.1f} GB   C: free {os_.free / 1e9:.1f} GB")

    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
