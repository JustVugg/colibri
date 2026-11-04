colibri v2.0.0: the CUDA backend for Windows

Unpack these files into the folder of colibri-v2.0.0-windows-x86_64.zip, next to
coli.cmd, replacing colibri.exe, qwen36.exe and kimi_k3.exe. They are the same
engines with CUDA added; coli setup then picks CUDA for an NVIDIA card.

coli_cuda.dll runs on compute capability 8.0 and newer (RTX 30, 40, 50 series,
A100, H100 and later) with an NVIDIA driver for CUDA 12. Older cards keep
Vulkan. cudart64_12.dll is NVIDIA's CUDA runtime, redistributed with it.
