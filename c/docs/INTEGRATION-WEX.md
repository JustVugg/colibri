# Integrazione WEX in Colibri

Questa guida descrive come abilitare il backend WEX (Windows Expert Container) in Colibri e come costruire container WEX4 con gli strumenti di repack forniti.

## 1. Prerequisiti

* Colibri compilato con supporto `expert_store_registry`.
* Python 3.10+ con `numpy` per i tool di repack.
* Opzionale: CUDA per `wex_cuda_bridge.cu` e build DLL.

## 2. File aggiunti

* `c/expert_store_wex.h` / `c/expert_store_wex.c` — backend WEX che registra `wex` nel registry.
* `c/wex_format.h`, `c/wex_neural_rans.h`, `c/wex_palette.h`, `c/wex4.h` — format e codec.
* `c/wex_decode.h`, `c/wex_decode.c` — decoder CPU e helper di caricamento.
* `c/qwen38_wex.h` — store Qwen3.8 WEX.
* `c/tools/repack_v4_wex.py`, `c/tools/repack_q38_mxfp4_wex.py`, `c/tools/build_nr3.py` — generazione container.

## 3. Abilitazione

Impostare le variabili d’ambiente prima di avviare Colibri:

```bash
export COLI_EXPERT_STORE=wex
export COLI_WEX_PATH=/percorso/verso/wex/container
export COLI_WEX_ROOT=/radice/wex
export COLI_WEX_VERIFY=1
export COLI_WEX_GPU_DIRECT=0
export V4_WEX_BATCH=1
export V4_WEX_SUBREADS=0
```

Variabili aggiuntive utili:
`COLI_WEX_EXTRA_MIRRORS`, `COLI_WEX_WEIGHTS`, `COLI_WEX_STRIPE`, `COLI_V4_TIER_PREWARM_AHEAD`, `V4_PREWARM_BATCH`, `COLI_WEX_RAM_CACHE_GB`, `WEX_READ_RETRY`, `COLI_WEX_CUDA_LIB`, `Q38_MXFP4_DIR`, `Q38_ASYNC_MISS`.

## 4. Build

Aggiungere `expert_store_wex.o` e `wex_decode.o` alla regola `deepseek-v4` e a `colibri`:

```make
OBJS += expert_store_wex.o wex_decode.o
```

Compilare:
```bash
make deepseek-v4
make colibri
```

## 5. Costruzione container WEX4

Esempio V4:
```bash
python c/tools/repack_v4_wex.py --quark-dir <quark> --fp8-dir <fp8> --output <out>.wex
```

Esempio Qwen3.8 MXFP4:
```bash
python c/tools/repack_q38_mxfp4_wex.py --quark-dir <quark> --fp8-dir <fp8> --output <out>.wex
```

Verifica:
```bash
python c/tools/wex4_verify.py <out>.wex
```

## 6. Patch engine

Applicare i tre patch dopo l’integrazione:
* `engine-wex4-loader-75932c1.patch`
* `engine-wex4-first-read-4ba5568.patch`
* `engine-q38-wex4-loader-de17198.patch`

Questi patch allineano `deepseek_v4.c` e `qwen38.c` all’uso del backend WEX.

## 7. Note di portabilità

WEX usa Direct I/O e API Windows in `wex_decode.c`. La build Linux/macOS richiede adattamento delle chiamate di file. I tool Python sono cross-platform.

## 8. Sicurezza

`COLI_WEX_VERIFY` attiva la verifica SHA-256 del container. `COLI_WEX_ROOT` pinna la radice del store per evitare sostituzioni non autorizzate.
