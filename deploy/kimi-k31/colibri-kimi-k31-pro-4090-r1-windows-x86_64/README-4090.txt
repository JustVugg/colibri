colibri on one RTX 4090 + 192 GB RAM + two NVMe drives, Windows x64: Kimi K3.1 Pro and MiMo-V2.6-Pro

Unzip anywhere (e.g. C:\colibri) and start one model at a time:

  START-KIMI-K31-PRO.cmd   Kimi K3.1 Pro (tools\start_kimi_k31_pro_4090.ps1)  README-KIMI-K31-PRO-4090.md
                           weights: tools\download_kimi_k31_pro_4090.ps1 (moonshotai/Kimi-K3.1-Pro)
  START-MIMO.cmd           MiMo-V2.6-Pro-MOPD (tools\start_mimo_cuda_4090.ps1)  README-MIMO-4090.md
  START-MIMO-HERETIC.cmd   MiMo-V2.6-Pro-MOPD-heretic (the same launcher, -Model heretic)

API: http://127.0.0.1:18080/v1 (OpenAI and Anthropic). Arguments after the .cmd go to its launcher
(model folders, threads, context): see each README. The other engines and `coli` are colibri's.

Needs: an AVX-512 CPU (built for Zen 4), an RTX 40xx with a CUDA 12/13 driver (the runtime is in
the DLLs), Python 3.10+ for the API server, about 192 GB of RAM for these profiles.

BUILD-INFO.txt: the commit and SHA-256 of every engine binary.
