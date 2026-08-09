# DeepSeek V4 Flash — CUDA/VRAM expert tiering (colibri)

**Branch**: `feat/ds4-cuda-tier` — fork `rafpigna/colibri` (base: `JustVugg/colibri` `dev` @ `47a749d`)
**Stato**: implementato e verificato su WSL2 (RTX 3080 sm_86) — build Linux ok, tier VRAM attivo, numerics identici a CPU.
**Scopo del documento**: riepilogo progettuale per collaboratori, PR futura e handoff tra sessioni.

---

## 1. Contesto

**colibri** è un motore di inferenza locale per LLM MoE giganti (GLM-5.2, Kimi K3, DeepSeek V4) su hardware consumer. La sua filosofia (mantainer **JustVugg**): *la mancanza di memoria veloce deve costare tok/s, mai la capacità di eseguire il modello*. Ogni backend GPU (CUDA/Metal/Vulkan) è un **tier**: tiene la parte "hot" in VRAM e fa fallback sul resto. GLM-5.2 incarna il design: dense in VRAM/RAM, esperti hot in VRAM, warm in RAM, cold streammati da disco, con apprendimento via `.coli_usage` (run dopo run gli esperti più usati vengono promossi).

## 2. Il problema (DeepSeek V4 Flash)

Al punto di partenza, DS4 Flash non aveva una via di mezzo:

| Stato | Cosa succedeva |
|---|---|
| `dev` (main, PR #165) | **solo CPU + streaming disco senza cache** → ~0.05 tok/s (ttft ~146s), inutilizzabile ma dimostrava che il modello gira |
| fork `ZacharyZcR` (branch long-context, PR #772/#773) | kernel **CUDA funzionanti ma all-residente**: preload di TUTTI gli esperti (256/layer × 43 ≈ 148GB) → **OOM in VRAM al layer 15** (10.7GB) o in RAM (56GB) con `COLI_DSV4_HYBRID=1`. Nessuna cache/eviction (0 occorrenze di ecache/npin/evict/LRU) |
| **Obiettivo** | colmare il gap: GPU + streaming + apprendimento, esattamente come GLM-5.2 |

Il requisito bloccante posto dal mantainer sulla PR #772 (kernel come *tier*, con **offload path** per gli esperti) non era mai stato soddisfatto. Questo lavoro lo soddisfa.

## 3. Strategia e porting

- **Base = `JustVugg/colibri` (`dev`), pulita** — niente fork-di-fork (il long-context è un fork di un fork). Usato solo come riferimento per il wiring dei kernel.
- **Portati dal fork `ZacharyZcR/colibri`** (credito a **ZacharyZcR**), file auto-contenuti e arch-agnostici (sm_80→sm_120):
  - `c/backend_cuda_dsv4.cu` — kernel esperti fp4/fp8 (`run_mv<>`, `dsv4_cuda_expert_group`, `dsv4_cuda_upload_fp4`, …), runtime CUDA + cublasLt
  - `c/backend_cuda_dsv4.h` — API (opaque types, `extern "C"`)
  - `c/dsv4_mhc.h`, `c/dsv4_quant.h` — formati dati di riferimento
- **Esclusi** i path `*_vllm`, `*_flashinfer`, `*_sm120*` (dipendono da FLASHINFER/cutlass; sm120 è Blackwell — inutile su sm_86).
- **Nessuna conversione dei pesi**: il checkpoint HF nativo è già "routed experts fp4 + dense fp8-e4m3", layout che il motore consuma direttamente (verificato su shard reali e in `build_record`/`validate_matrix`).

## 4. Implementazione

**Scoperta chiave**: il base `dev` ha **già** il tiering RAM completo nell'engine DS4 (`V4HotPolicy`: pin hot per-layer ranked per usage, cache LRU, repin periodico, history `.coli_usage`, letture O_DIRECT). Mancava **solo il tier VRAM/CUDA** (0 riferimenti CUDA nel base).

Aggiunto (in `deepseek_v4.c`, tutto sotto `#ifdef COLI_DSV4_CUDA`, + Makefile):

- **Tier VRAM = top-M dei pin**: `vram_per_layer` esperti per layer (rank per usage tra i pin attivi) restano residenti in VRAM via `dsv4_cuda_upload_fp4`. Budget da env `CUDA_EXPERT_GB` (default 4GB), `COLI_DSV4_CUDA=0` per disabilitare, `COLI_GPU(S)` per il device.
- **Sync a lookup** (`lookup_hot`): upload all'entrata nella window, drop alla demozione, sempre sui **bytes FP4 nativi** (prima del packing rows16 — pitfall: `COLI_FP4_ROWS16_KERNEL` è attivo di default e riscriverebbe lo slab).
- **Dispatch tiered**: `coli_v4_expert_forward_tiered` → GPU `dsv4_cuda_expert_group` se lo slot ha handle VRAM, con **fallback CPU silenzioso**; altrimenti reference CPU. I 3 call-site MoE passano dal tiered.
- **Init/shutdown CUDA** con refcount (multi-store), cleanup in `destroy_hot`.
- **Makefile**: `Makefile.deepseek-v4` — `CUDA=1` compila `backend_cuda_dsv4.o` (nvcc, `CUDA_ARCH=sm_86`/`portable`) e linka `-lcudart -lcublasLt`; `c/Makefile` propaga `CUDA`/`CUDA_ARCH` al sub-make.
- **Env**: `RAM_GB` (cache esperti, default 4GiB nel CLI), `CUDA_EXPERT_GB`, `COLI_DSV4_CUDA`, `COLI_GPU(S)`.

**Scope v1**: esperti routed su GPU; **dense/attention restano su CPU** (fp8 nativo della base). Estensione futura: dense+attention su GPU (API già nel backend) — decisione rimandata.
**Numerics**: il path GPU applica simulazione fp8 sulle attivazioni (come l'oracle vLLM), il CPU bf16-round. Verificato **identico** sui test (A/B "Hello! How can" == "Hello! How can"; output 8 token coerente).

## 5. Evidenza empirica (RTX 3080 10.7GB, WSL2 Ubuntu 26.04, modello 156GB)

Build: `make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 CUDA_ARCH=sm_86` → binario ~1.7MB, linka `libcudart.so.13`+`libcublasLt.so.13`, ~70 simboli `dsv4_cuda_*`.

```
[DSV4 CUDA] device 0: NVIDIA GeForce RTX 3080 10.7 GB sm_86
v4_cuda_tier vram_per_layer=5 budget_gb=3.00 expert_mb=12.8 vram_experts_gb=2.87
v4_tokens prompt=5 generated=8 total=13 expert_requests=3096 hits=1571 misses=1525 hit_rate=50.743
TUNE decode: 8 tokens in 37.762s   TTFT=29.558s after_first=8.196s
generated_text=Hello! How can I help you today
v4_cuda_tier uploads=235 drops=235
```

- Tier VRAM attivo e misurato: **VRAM 3323→3795 MiB** durante decode (upload=235), **OOM-free** per costruzione (window M×layer×12.8MB ≤ budget).
- **Learning**: hit_rate 29→32→40→44→**51%** su run consecutivi (`.coli_usage` nel model dir).
- Clean shutdown: VRAM riportata a 794 MiB.
- A/B numerics CPU vs CUDA: token identici; 8 token coerenti.

## 6. Build e uso

```bash
# Linux (WSL2 o nativo)
source ~/cuda_env.sh          # CUDA_HOME=/usr/local/cuda
make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 CUDA_ARCH=sm_86    # o: make deepseek-v4 CUDA=1

# Esecuzione test
CUDA_EXPERT_GB=3 ./deepseek_v4 /path/DeepSeek-V4-Flash "Hello" --max-tokens 8 --memory-gb 16
COLI_DSV4_CUDA=0 ./deepseek_v4 ...   # variante CPU-only (baseline)
```

**Windows nativo**: prossima feature (stesso branch, commit successivo) — pattern in-repo `CUDA_DLL=1` + loader per MinGW; stima 1.5-2 gg.

## 7. Follow-up (ordine di priorità attuale)

1. **Build Windows nativa** (evita dual-boot/WSL-VHDX per il tier disco): loader `backend_cuda_dsv4`→DLL + Makefile Windows.
2. **Benchmark Linux nativo** (NVMe pieno ~3GB/s vs VHDX ~1GB/s): stress a RAM reale, convergenza hit-rate, tok/s, confronto CPU-only. Script pronti: `setup_ds4_linux.sh`, `stress_ds4_linux.sh`.
3. **Dense+attention su GPU** (schema GLM completo) — rimandato.
4. **PR verso `JustVugg/colibri` (base `dev`)** quando stabilizzato.

## 8. Referenze

- Base: `https://github.com/JustVugg/colibri` (dev)
- Fork sorgente kernel: `https://github.com/ZacharyZcR/colibri` (branch `feat/deepseek-v4-long-context`)
- Nostro fork: `https://github.com/rafpigna/colibri` — branch `feat/ds4-cuda-tier`
- Modello: `deepseek-ai/DeepSeek-V4-Flash-0731` (pesi originali HF, niente conversione)
- PR storiche: #165 (CPU engine in dev), #772 (kernel CUDA, respinta: offload mancante), #773 (closed)