<p align="center">
  <img src="assets/colibri.svg" width="500" alt="colibrì — motore piccolo, modello immenso">
</p>

<p align="center">
  <a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.zh-TW.md">繁體中文</a> · Italiano
</p>

**Motore piccolo, modello immenso.** Esegui **GLM-5.2 (744 miliardi di parametri, MoE)** su un computer consumer con ~25 GB di RAM — in C puro, zero dipendenze, caricando gli expert dal disco in streaming.

Colibrì è un runtime MoE leggero e che preserva la qualità: tratta VRAM, RAM e
disco come un'unica gerarchia di memoria gestita. Se la memoria veloce non basta
il modello rallenta, ma la policy predefinita **non cambia mai silenziosamente la
precisione del modello né la semantica del router**.

```
$ ./coli chat
  🐦 colibrì v1.0 — GLM-5.2 · 744B MoE · int4 · streaming CPU
  ✓ ready in 32s · resident 9.9 GB
  › ciao!
  ◆ Ciao! 😊 Come posso aiutarti oggi?
```

## Guardalo in azione

<p align="center">
  <img src="docs/media/colibri-dashboard.png" width="900" alt="dashboard web di colibrì — metriche live, pannello hardware, livelli degli expert">
</p>
<p align="center"><em>La dashboard web (<code>./coli web</code>): un modello da 744B a <strong>4 tok/s, TTFT 1.6 s, disco 0</strong> —
residenza completa degli expert su 6× RTX 5090, con metriche token in tempo reale, breakdown dei tempi per turno,
la barra dei livelli VRAM/RAM/disco e il mini-cervello live nell'angolo.</em></p>

<p align="center">
  <img src="docs/media/colibri-brain.png" width="900" alt="la pagina Brain — 19.456 expert come una corteccia vivente">
</p>
<p align="center"><em>La pagina <strong>Brain</strong>: tutti i 19.456 expert come una corteccia vivente — il colore indica
il livello di archiviazione, la luminosità il calore di routing, e ogni expert instradato in un turno
lampeggia bianco. Passando il cursore si vede l'<a href="https://github.com/JustVugg/colibri/issues/175">affinità
tematica misurata</a> dell'expert.</em></p>

<p align="center">
  <img src="docs/media/colibri-atlas.png" width="900" alt="la pagina Atlas — l'atlante misurato degli expert come una galassia 3D">
</p>
<p align="center"><em>La pagina <strong>Atlas</strong>: l'<a href="https://github.com/JustVugg/colibri/issues/175">atlante
misurato degli expert</a> come una galassia 3D — 13.260 expert caratterizzati, 1.041 specialisti
replicabili che si raggruppano per argomento (poesia, legge, cinese, SQL…). La posizione deriva
dall'affinità di routing misurata, non da un embedding appreso. Trascinare per ruotare.</em></p>

## L'idea

Un modello Mixture-of-Experts da 744B attiva solo ~40B parametri per token — e
solo ~11 GB di quelli cambiano da un token all'altro (gli expert instradati):

<p align="center">
  <img src="docs/media/sparse.png" width="880" alt="solo ~5.4% dei parametri è attivo per token">
</p>

Il modello non ha bisogno di *stare* in memoria veloce — ha bisogno di essere
**piazzato**:

- la **parte densa** (attenzione, expert condivisi, embedding — ~17B parametri)
  resta **residente in RAM a int4** (~9.9 GB);
- i **19.456 expert instradati** (75 layer MoE × 256 + la testa MTP, ~19 MB
  ciascuno a int4) stanno **su disco** (~370 GB) e vengono **caricati on demand
  in streaming**, con una cache LRU per layer, un hot-store pinnato che impara,
  e un livello VRAM opzionale.

Il motore è un singolo file C (`c/colibri.c`) più header piccoli. Niente BLAS,
niente Python a runtime, niente GPU obbligatoria.

## Come funziona

### Il percorso di ogni token

<p align="center">
  <img src="docs/media/token-path.png" width="880" alt="instrada → unione → piazza → sovrapponi → impara">
</p>

Ogni layer di ogni token percorre gli stessi cinque passi. L'obiettivo
progettuale è che **il piazzamento decide solo la velocità** — le decisioni
del router e la precisione dei pesi sono identiche sia che un expert risponda
dalla VRAM sia dal disco.

### Una gerarchia di memoria, non un requisito di memoria

<p align="center">
  <img src="docs/media/tiers.png" width="880" alt="residenza expert a tre livelli: VRAM / RAM / NVMe">
</p>

Lo stesso motore copre l'intero spettro: su un portatile da 25 GB tutto viene
caricato dal disco in streaming (lento, ma corretto); su un host grande l'intero
set di expert diventa residente (`CUDA_EXPERT_GB=auto PIN_GB=all`) e il disco
esce completamente dal percorso di decode. Tra i livelli c'è una **cache che
impara**: il motore registra quali expert il *tuo* carico di lavoro instrada
(`.coli_usage`, aggiornato a ogni turno) e fissa automaticamente i più caldi —
colibrì diventa letteralmente più veloce man mano che lo usi. Sugli host
multi-socket, `COLI_NUMA=1` interlaccia i pesi residenti tra i controller di
memoria ([#82](https://github.com/JustVugg/colibri/issues/82)).

### Mai aspettare il disco due volte

I miss nella cache costano caro, quindi il motore investe la maggior parte
della sua astuzia per evitarli e sovrapporli: le tre matrici di ogni expert sono
memorizzate contigue e lette con un unico `pread`; un pool I/O asincrono
limitato (`PIPE=1`, attivo per default) carica gli expert mancanti mentre quelli
residenti calcolano; le posizioni in batch leggono ogni expert unico una sola
volta (**batch-union**); un thread di lookahead del router (`PILOT=1`) fa il
prefetch degli expert del layer successivo — il routing è misurabilmente
**prevedibile al 71.6% un layer in anticipo**. Sulle GPU, la pipeline residente
(`COLI_CUDA_PIPE=2`) mantiene il flusso residuo on-device tra i layer, così il
loop CPU degli expert procede senza interruzioni; su Apple Silicon un backend
[Metal](docs/metal.md) sperimentale esegue la matmul batch degli expert sulla
GPU a memoria unificata.

### Modello fedele, stato compresso

Il forward pass è validato **token-esatto contro un oracle `transformers`**
(teacher-forcing 32/32). L'attenzione MLA memorizza uno stato KV compresso — 576
float/token invece di 32.768 (**57× più piccolo**) — e lo persiste tra i
riavvii (`.coli_kv`): le conversazioni riaprono "calde", senza alcun re-prefill,
byte-identiche a una sessione ininterrotta. L'attenzione sparsa DSA (il
lightning indexer di GLM-5.2) è implementata fedelmente e validata forzando la
selezione di tutte le chiavi per riprodurre esattamente l'attenzione densa.

### Decodifica speculativa, onestamente

La testa MTP nativa di GLM-5.2 propone token che il modello principale verifica
in un unico forward batch — 2.2–2.8 token/forward quando conviene. Due regole
conquistate a caro prezzo sono i default: la testa MTP deve essere **int8** (le
teste int4 crollano al 0–4% di accettazione,
[#8](https://github.com/JustVugg/colibri/issues/8)), e draft e verifica devono
calcolare **la stessa funzione** — `SPEC_PIN=1` fissa entrambi sulla stessa
famiglia di kernel ([#163](https://github.com/JustVugg/colibri/issues/163)
contiene l'intera indagine forense). I draft forzati da grammatica
([`GRAMMAR=file.gbnf`](docs/grammar-draft.md)) aggiungono accettazione quasi
gratuita sull'output JSON vincolato. Se la speculazione conviene dipende dalla
temperatura della cache — misura, e usa `DRAFT=0` quando non paga.

## Cosa ottiene

<p align="center">
  <img src="docs/media/ladder.png" width="880" alt="velocità di decode misurata per classe hardware">
</p>

Stesso motore, stesso container int4 — cambia solo dove risiedono gli expert.
Punti salienti dalle [tabelle benchmark complete](docs/benchmarks.md):

- **6× RTX 5090, residenza completa:** 5.8–6.8 tok/s in decode, TTFT ~13 s
  ([log dell'esperimento](docs/experiments/glm52-6x5090-2026-07-12.md));
- **desktop solo-CPU da 128 GB:** ~1.8 tok/s a cache calda
  ([#200](https://github.com/JustVugg/colibri/issues/200));
- **singola RTX 5070 Ti, classe laptop:** 1.07 tok/s tramite la pipeline
  GPU-residente ([#273](https://github.com/JustVugg/colibri/issues/273));
- **macchina di sviluppo da 25 GB:** 0.05–0.1 tok/s a freddo — il punto di
  partenza dimostrato da cui è nato il progetto, e ancora oggi la baseline onesta.

La qualità è misurata, non presunta: il costo di quantizzazione del container
int4 e le ablazioni su granularità delle scale e rotazione sono in
[docs/benchmarks.md](docs/benchmarks.md#quality-benchmark) e
[#108](https://github.com/JustVugg/colibri/issues/108)/[#81](https://github.com/JustVugg/colibri/issues/81).

## Per iniziare

### 1. Scarica il modello

Un container **GLM-5.2 int4** pre-convertito è su Hugging Face — **usa la
versione con le teste MTP int8**:

**https://huggingface.co/mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp**

> ⚠️ Il mirror originale contiene teste MTP int4 → accettazione dei draft allo 0%
> ([#8](https://github.com/JustVugg/colibri/issues/8)). Verifica la tua versione:
> `ls -l <modello>/out-mtp-*` — int8 (corretto) è `3527131672 / 5366238584 / 1065950496`.

Oppure converti tu stesso dalla sorgente FP8 — un unico comando riprendibile che
non richiede mai i 756 GB completi su disco contemporaneamente:

```bash
cd c && ./setup.sh                        # verifica gcc/OpenMP, compila, autotest
./coli convert --model /nvme/glm52_i4     # scarica e converti shard per shard (python, una tantum)
```

### 2. Esegui

```bash
COLI_MODEL=/nvme/glm52_i4 ./coli chat     # budget RAM, cache e MTP rilevati automaticamente
COLI_MODEL=/nvme/glm52_i4 ./coli plan     # mostra il piazzamento pianificato VRAM/RAM/disco
COLI_MODEL=/nvme/glm52_i4 ./coli doctor   # controllo di idoneità (sola lettura)
./coli web  --model /nvme/glm52_i4        # API + dashboard web sulla stessa porta
./coli serve --model /nvme/glm52_i4       # solo API compatibile OpenAI
```

Il motore a runtime è puro C — python si usa solo per il convertitore (una tantum)
e per il gateway API opzionale.

### 3. Approfondisci

| argomento | documento |
|---|---|
| Benchmark, dati dalla comunità, misurazioni di qualità | [docs/benchmarks.md](docs/benchmarks.md) |
| Parametri di tuning, policy, cache che impara, prefetch | [docs/tuning.md](docs/tuning.md) |
| Build nativa su Windows 11 (con CUDA DLL) | [docs/windows.md](docs/windows.md) |
| Backend CUDA, livello expert in VRAM, residenza completa | [docs/cuda.md](docs/cuda.md) |
| Backend Metal per Apple Silicon | [docs/metal.md](docs/metal.md) |
| API compatibile OpenAI, KV slot, dashboard web | [docs/api.md](docs/api.md) |
| Draft forzati da grammatica (output strutturato) | [docs/grammar-draft.md](docs/grammar-draft.md) |
| Inventario delle variabili d'ambiente | [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) |

## Sostenere il progetto

colibrì è nato come progetto di una sola persona su un portatile con 12 core
e 25 GB di RAM; oggi i suoi numeri arrivano da una comunità di macchine reali.
Se ti è utile:

- ⭐ metti una stella al repository e condividilo;
- 🐛 apri issue con i numeri di benchmark del tuo hardware — i datapoint
  fanno avanzare questo progetto più di qualsiasi altra cosa;
- 💬 contattaci via GitHub issues per sponsorizzare lo sviluppo o donare hardware.

## Struttura del repository

```
Makefile                  punto d'ingresso root per build/check
c/
├── colibri.c             motore principale
├── quant.h               kernel matmul quantizzati (SIMD multi-architettura)
├── sample.h              campionamento, RNG, set di stop
├── kv_persist.h          persistenza KV su disco (.coli_kv)
├── telemetry.h           protocollo dashboard, statistiche, usage
├── st.h, tok.h, json.h   header di runtime
├── backend_cuda.*        livello CUDA opzionale
├── Makefile              build e check locali
├── coli                  CLI utente
├── openai_server.py      gateway HTTP compatibile OpenAI
├── setup.sh              setup locale in un solo comando
├── tools/                conversione offline, fixture e benchmark
├── scripts/              helper per conversioni lunghe
└── tests/                test C e Python senza dipendenze
web/                      UI browser (puro client API OpenAI)
desktop/                  shell desktop Tauri v2 che racchiude la web UI
docs/                     documentazione di riferimento, esperimenti, media
```

Il percorso a runtime resta intenzionalmente piatto e leggibile: `colibri.c`
più i suoi header. Dalla radice del repository, `make`, `make check` e
`make clean` delegano al Makefile del motore.

## Perché "colibrì"

Il colibrì pesa pochi grammi, sta sospeso nel vuoto e visita un migliaio di
fiori al giorno. Questo motore tiene in vita un gigante da 744 miliardi di
parametri con le razioni di un colibrì: 25 GB di RAM, dodici core CPU e
tanta pazienza col disco.

Il nome è rimasto in italiano perché questa è la lingua in cui è stato scritto
il primo prototipo — i commenti nel codice lo testimoniano ancora.

## Licenza

Apache 2.0. I pesi di GLM-5.2 sono rilasciati da Z.ai sotto licenza MIT.
