import os, sys, subprocess, shutil
from pathlib import Path

BOOTSTRAP = Path(r"D:\Colibri\.venv\Scripts\python.exe")
env = os.environ.copy()
env["PYTHONDONTWRITEBYTECODE"] = "1"
env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
if BOOTSTRAP.exists():
    env["PATH"] = str(BOOTSTRAP.parent) + os.pathsep + env.get("PATH", "")

def run(cmd, timeout=30):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        return (p.stdout or "") + (p.stderr or ""), p.returncode
    except subprocess.TimeoutExpired:
        try:
            p.kill()
        except Exception:
            pass
        try:
            out, err = p.communicate(timeout=5)
        except Exception:
            out, err = "", ""
        return (out or "") + (err or "") + "\n[PROBE_KILLED_AFTER_TIMEOUT]", -1
    except Exception as e:
        return f"[{type(e).__name__}: {e}]", -1

lines = []

lines.append("=== NVIDIA RTX 4060 (dedicated + reserved) ===")
out, _ = run(["nvidia-smi","--query-gpu=name,driver_version,memory.total,memory.free,memory.used,memory.reserved",
              "--format=csv,noheader"], timeout=20)
lines.append(out.strip() or "(no nvidia-smi)")

lines.append("\n=== NVIDIA compute processes ===")
out, _ = run(["nvidia-smi","--query-compute-apps=name,pid,gpu_name,used_gpu_memory",
              "--format=csv,noheader"], timeout=20)
lines.append(out.strip() or "(no processes or empty)")

lines.append("\n=== Vulkan per-GPU summary ===")
for candidate in (r"C:\Windows\System32\vulkaninfo.exe",
                  r"C:\Program Files\VulkanRT\bin32\vulkaninfo.exe",
                  r"C:\Program Files\VulkanRT\bin64\vulkaninfo.exe"):
    if os.path.exists(candidate):
        out, _ = run([candidate, "--summary"], timeout=30)
        for ln in out.splitlines():
            s = ln.strip()
            if any(k in s for k in ("driverName","driverVersion","driverID","VK_LAYER_","PhysicalDevice","GPU ",
                                     "memoryHeap","dedicated","deviceLocal","hostVisible","hostCoherent")):
                lines.append("   " + s)
        break

lines.append("\n=== Intel NPU / IAI accelerator enumeration ===")
out, _ = run(["powershell","-NoProfile","-Command",
             "(Get-ChildItem 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Class' -ErrorAction SilentlyContinue | "
             "ForEach-Object { Get-ItemProperty $_.PsPath -ErrorAction SilentlyContinue } | "
             "Where-Object { $_.DriverDesc -match 'Intel.*NPU|Intel AI Boost|Intel AI Accelerator' } | "
             "Select-Object DriverDesc, ProviderName, Class, MatchingDeviceId | Format-Table -AutoSize)"],
            timeout=30)
lines.append(out.strip() or "(no NPU enum found via registry)")

lines.append("\n=== Windows video adapters (dedicated AdapterRAM) ===")
out, _ = run(["powershell","-NoProfile","-Command",
              "Get-CimInstance Win32_VideoController | ForEach-Object { Write-Host $_.Name; "
              "Write-Host ('  AdapterRAM GB = ' + [math]::Round($_.AdapterRAM/1GB,2)); "
              "Write-Host '  DriverVersion = ' $_.DriverVersion; "
              "Write-Host '  PNPDeviceID = ' $_.PNPDeviceID }"],
             timeout=20)
lines.append(out.strip() or "(no win32 adapters)")

lines.append("\n=== Windows system RAM ===")
try:
    import psutil
    vm = psutil.virtual_memory()
    lines.append(f"total RAM: {vm.total/1e9:.2f} GB")
    lines.append(f"available RAM: {vm.available/1e9:.2f} GB")
    lines.append(f"used RAM: {vm.used/1e9:.2f} GB")
except Exception as e:
    lines.append(f"psutil unavailable: {e}")

lines.append("\n=== platform ===")
lines.append("sys.platform: " + sys.platform)
lines.append("python: " + sys.version.split()[0])
if shutil.which("nvidia-smi"):
    lines.append("nvidia-smi: " + shutil.which("nvidia-smi"))
if shutil.which("vulkaninfo"):
    lines.append("vulkaninfo: " + shutil.which("vulkaninfo"))

print("\n".join(lines))
