<p align="center">
  <img src="assets/colibri.svg" width="500" alt="colibrì — motore piccolo, modello immenso">
</p>

<p align="center">
  <a href="https://discord.gg/MAaKtQRc"><b>Discord</b></a> ·
  <a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.zh-TW.md">繁體中文</a> · Italiano
</p>

**Motore piccolo, modello immenso.** Esegui **modelli MoE di frontiera — da 744
miliardi a 2,8 mila miliardi di parametri** — su hardware consumer ed eterogeneo,
in C puro e senza dipendenze del motore, trattando storage, RAM e VRAM come
un'unica gerarchia di inferenza.

Oggi girano quattro famiglie: **GLM-5.2** (744B), **Inkling** (975B), **Kimi K3**
(2,8T) e **OLMoE** (7B) — un file C ciascuna, la stessa interfaccia `coli chat` /
`coli serve` / `coli web`. [Elenco completo](README.md#other-supported-models)

> **Colibrì è un motore di inferenza che puoi usare oggi, e una piattaforma di
> ricerca aperta.** Il suo obiettivo principale è migliorare le prestazioni di
> inferenza lungo l'intero confine software/hardware — formati dei modelli,
> gerarchia di memoria, I/O dello storage, piazzamento, scheduling, kernel,
> speculazione e sovrapposizione CPU/GPU — affinché i grandi modelli dipendano
> meno da hardware raro e costino meno.

Colibrì è intenzionalmente un luogo dove verificare idee di sistema aggressive —
quindi **nessuno SLA sulla velocità, e una garanzia dura sulla semantica**: gli
esperimenti devono dimostrare il proprio valore con misure end-to-end riproducibili;
la policy predefinita **non cambia mai silenziosamente la precisione del modello né
la semantica del router**. Una memoria veloce insufficiente può ridurre la velocità,
ma non ridefinire il modello di nascosto.

```
$ ./coli chat
  🐦 colibri v1.4.0 — GLM-5.2 · 744B MoE · int4 · streaming CPU
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

## La missione di ricerca

L'inferenza di frontiera non dovrebbe richiedere per forza hardware da datacenter.
L'obiettivo di Colibrì è semplice: **ridurre la dipendenza dall'hardware e il costo
totale dell'inferenza, ottimizzando ogni parte del percorso che le misure indicano
come limitante**.

Questo comprende cambiare il modo in cui i pesi sono rappresentati e spostati,
decidere cosa risiede in VRAM, RAM o storage, sovrapporre calcolo eterogeneo,
ridurre i costi di avvio e sincronizzazione, sfruttare sparsità e riuso e verificare
nuovi algoritmi di decoding. La convenzione non protegge una tecnica; un microbenchmark
veloce non basta ad adottarla. Decide l'inferenza end-to-end su macchine reali,
misurando correttezza e qualità insieme a throughput, latenza, memoria e costo.

Il risultato pratico è l'accessibilità: eseguire un modello da 744B sull'hardware
che già possiedi, osservare ogni expert in tempo reale e modificare il codice che
lo rende possibile. Non noleggiare intelligenza dietro un'API, ma possederla,
analizzarla, misurarla e migliorarla. Il motore resta volutamente abbastanza piccolo
perché la prossima ottimizzazione utile possa arrivare da chiunque sia disposto a misurarla.

## Tecniche fondamentali e risultati misurati

- **Una gerarchia, non una soglia di memoria.** VRAM, RAM e NVMe sono livelli di
  piazzamento degli stessi pesi; poca memoria veloce cambia la velocità, non il modello.
- **Un JIT per i pesi.** Il calore di routing misurato alimenta una LRU per layer,
  un hot-store appreso e il prefetch del layer successivo senza caricare tutti gli expert.
  Aiuta sui carichi ripetibili, ma la cronologia può sovradattarsi e il prefetch può
  perdere su alcuni host: sono policy da misurare, non promesse.
- **L'I/O fa parte del motore.** Unione degli expert per batch, letture sovrapposte
  al calcolo, `O_DIRECT` e striping pesato su due SSD ottimizzano direttamente lo
  streaming. `O_DIRECT` dipende dal disco e il doppio SSD richiede più A/B end-to-end.
- **Esecuzione eterogenea.** CPU, CUDA, Metal, memoria NUMA e residenza parziale o
  completa degli expert condividono un runtime; la combinazione utile dipende da
  calcolo, banda, residenza e carico.
- **Stato compresso senza cambiare modello.** Validazione token-exact, stato MLA KV
  57× più piccolo, conversazioni persistenti e DSA fedele vincolano l'ottimizzazione
  alla correttezza. Sono proprietà di memoria, latenza e correttezza, non una
  promessa generale di throughput.
- **La speculazione deve meritarsi il costo.** MTP nativo e draft vincolati da
  grammatica sono misurati end-to-end e si disattivano quando l'accettazione non
  ripaga la verifica.

## Ipotesi aperte, esperimenti e partecipazione

Colibrì considera ogni ottimizzazione un'ipotesi finché un A/B end-to-end
controllato non dimostra il contrario. Le domande principali sono:

| ipotesi | evidenza attuale | esperimento ancora necessario |
|---|---|---|
| La cronologia di routing può piazzare gli expert meglio di una semplice LRU | i pin appresi migliorano carichi ripetuti, ma possono sovradattarsi al prompt | A/B cross-session su set esclusi: codice, chat, multilingua e contesti lunghi |
| Più SSD possono trasformare banda indipendente in velocità di decode | routing pesato mirror/split implementato e validato; il modello di banda è solido | GLM-5.2 a cache fredda, uno contro due dischi su controller indipendenti reali |
| Un planner hardware-aware può avvicinarsi automaticamente alla configurazione migliore | oggi rileva budget RAM/VRAM e diversi backend | confrontare il piano generato con sweep controllati su laptop, workstation, NUMA e multi-GPU |
| Rappresentazioni lossless o a qualità limitata possono ridurre abbastanza il movimento dei pesi | esistono ablation di formato e quantizzazione con gate di qualità | riprodurre insieme qualità, byte mossi, latenza e costo per token utile, non solo il rapporto di compressione |
| La speculazione routing-aware può convenire prima della residenza quasi completa | MTP e draft grammaticali funzionano, ma MTP ha anche perso il 32% intorno all'85% di expert hit | mappare il pareggio tra accettazione, hit rate, batch union e profondità del draft |
| La sovrapposizione CPU/GPU può nascondere trasferimenti e sincronizzazione | esistono risultati positivi CUDA e Metal, ma CPU veloci e bassa residenza possono annullarli | profili per fase e A/B a variabile singola su PCIe, memoria unificata e piena residenza |

Per contribuire, scegli una riga e pubblica anche i risultati negativi. Registra
hardware, commit, container del modello, comando esatto, prompt, stato cache,
throughput, TTFT, expert hit, byte letti e controllo qualità; cambia una sola
variabile, ripeti e allega i log grezzi. Parti da
[CONTRIBUTING.md](CONTRIBUTING.md), confronta il
[protocollo di benchmark](docs/benchmarks.md), quindi
[apri una issue di esperimento](https://github.com/JustVugg/colibri/issues/new).
Un fallimento controllato vale più di un numero veloce senza spiegazione.

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

Ti servono due cose: **il programma** (poche centinaia di KB) e **il modello**
(372 GB). Guida passo passo per tutte le piattaforme nella
[Quick Start](docs/quickstart.md).

### 1. Procurati colibri

**Scarica una release già compilata** — Linux, macOS e Windows, nessun
compilatore necessario. Prendi l'archivio della tua piattaforma dalla pagina
[Releases](https://github.com/JustVugg/colibri/releases) e scompattalo:

```bash
mkdir colibri && tar xzf colibri-v1.1.0-linux-x86_64.tar.gz -C colibri && cd colibri
python3 coli info                         # engine ready ✓
```

Dentro trovi il motore (`colibri`, `colibri.exe` su Windows), il launcher `coli`
e i suoi script Python di supporto. Niente da rinominare o configurare: `coli`
trova il motore accanto a sé. Serve solo avere
[Python 3](https://www.python.org/downloads/) installato — il launcher e il
gateway API sono script Python, mentre il motore è C puro senza dipendenze.

**Oppure compila dai sorgenti** — servono `gcc` (o clang) con OpenMP:

```bash
git clone https://github.com/JustVugg/colibri && cd colibri/c
./setup.sh                                # verifica gcc/OpenMP, compila, autotest
```

Vuoi `coli` nel PATH? Da un checkout, `pip install -e .` lo registra (il motore
resta in `c/` — è un'installazione editabile dal clone, non un wheel).

### 2. Scarica il modello

Un container **GLM-5.2 int4** pre-convertito è su Hugging Face — usa la build
**group-scaled (gs64) con la testa MTP int8**. Pesa circa **372 GB**, quindi mettilo su un
disco che abbia lo spazio, meglio se veloce:

**https://huggingface.co/mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp**

> ⚠️ Usa il container **gs64** qui sopra, non i vecchi mirror int4 per-row
> (`mateogrgic/…`, `jlnsrk/…`): misurano circa 9 punti percentuali in meno sulla
> qualità e causavano i loop in think-mode e le generazioni senza termine originali
> di [#455](https://github.com/JustVugg/colibri/issues/455). Il container gs64 ha
> corretto quegli A/B per-row controllati, ma non è una protezione generale contro
> ripetizioni o EOS starvation. Anche la testa MTP deve essere **int8, non int4**
> (int4 → 0% di accettazione dei draft,
> [#8](https://github.com/JustVugg/colibri/issues/8)):
> `ls -l <modello>/out-mtp-*` — int8 (corretto) è `3527131672 / 5366238584 / 1065950496`.

Oppure converti tu stesso dalla sorgente FP8 — un unico comando riprendibile che
non richiede mai i 756 GB completi su disco contemporaneamente:

```bash
./coli convert --model /nvme/glm52_i4     # scarica e converti shard per shard (python, una tantum)
```

### 3. Esegui

```bash
COLI_MODEL=/nvme/glm52_i4 ./coli chat     # budget RAM, cache e MTP rilevati automaticamente
COLI_MODEL=/nvme/glm52_i4 ./coli plan     # mostra il piazzamento pianificato VRAM/RAM/disco
COLI_MODEL=/nvme/glm52_i4 ./coli doctor   # controllo di idoneità (sola lettura)
./coli web  --model /nvme/glm52_i4        # API + dashboard web sulla stessa porta
./coli serve --model /nvme/glm52_i4       # solo API compatibile OpenAI
```

Su Windows gli stessi comandi funzionano con `python coli chat --model D:\glm52_i4`.
Il motore a runtime è puro C — python si usa solo per il convertitore (una tantum)
e per il gateway API opzionale.

### 4. Approfondisci

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

## Prossimi passi

- **La ricerca sui sistemi di inferenza è il prodotto.** La gerarchia attuale usa
  LRU e un insieme appreso di expert fissati; il lavoro attivo copre formati,
  compressione, piazzamento, scheduling, I/O, kernel CPU/GPU, sovrapposizione
  eterogenea, stato KV e speculazione consapevole del routing. L'obiettivo è
  ridurre i requisiti hardware e il costo per token utile, con risultati misurati
  end-to-end, revisionati e sviluppati apertamente.
- **Più modelli aperti.** L'algoritmo di tiering è indipendente dal modello:
  qualsiasi MoE con expert instradati può essere organizzato allo stesso modo.
  GLM-5.2 e OLMoE funzionano già; **Kimi K2**, **Qwen3 MoE** e **MiniMax** sono
  nella roadmap.

## Sostenere il progetto

colibrì è nato come progetto di una sola persona su un portatile con 12 core
e 25 GB di RAM; oggi i suoi numeri arrivano da una comunità di macchine reali.
Se ti è utile:

- ⭐ metti una stella al repository e condividilo;
- 🐛 apri issue con i numeri di benchmark del tuo hardware — i datapoint
  fanno avanzare questo progetto più di qualsiasi altra cosa;
- 💬 entra nella [comunità Discord](https://discord.gg/MAaKtQRc) per discutere
  esperimenti, risultati hardware e direzioni di ricerca;
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
