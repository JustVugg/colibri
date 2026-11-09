# Scripts in this repo

This dir has the working scripts for Qwen3-Coder-30B-A3B on this laptop.

- `serve.py` — local serve script. Tries Colibri first if it exists, else llama.cpp, else notes Ollama.
- `weight_space.py` — weight-space analysis + small targeted edits. Read-only analyze first, patch only after you have a hypothesis.
- `download_shards.py` — download bf16 safetensors shards from the official Qwen repo (no GGUF in the official repo; GGUF came from unsloth).
- `run_weight_space_check.py` — quick smoke test for the weight-space module import and whether shards are present.

What this machine actually has right now
----------------------------------------
- Python 3.12 from `C:\Users\Liso1\AppData\Local\Programs\Python\Python312\python.exe`
- Packages: huggingface_hub, safetensors, numpy, tqdm, PyYAML, etc.
- No torch / no CUDA toolkit / no llama.cpp / no Colibri binary found yet.
- GPU: RTX 4060 Laptop 8 GB, driver 621.14.
- Disk: D: 722 GB total, ~159 GB used, ~563 GB free.
- Model: Qwen3-Coder-30B-A3B-Instruct Q4_K_M GGUF present (~18.5 GB).

What I changed compared to the old broken setup
------------------------------------------------
- Fixed `download.py` to use unsloth's GGUF repo (official repo has no GGUF).
- Rewrote `evaluate.py`.
- Added a real `weight_space.py` instead of the stub that used to be there.
- Added `serve.py` with a Colibri-first policy.
