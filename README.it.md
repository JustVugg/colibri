<p align="center">
  <img src="assets/colibri-logo.svg" width="560" alt="colibrì — motore piccolo, modello immenso">
</p>

<p align="center">
  <a href="https://justvugg.github.io/colibri"><img src="https://img.shields.io/badge/website-justvugg.github.io%2Fcolibri-1f6feb" alt="Website"></a>
  <a href="https://github.com/JustVugg/colibri/releases"><img src="https://img.shields.io/github/v/release/JustVugg/colibri?color=2ea043" alt="Latest release"></a>
</p>

<p align="center">
  <a href="https://justvugg.github.io/colibri"><b>Sito web</b></a> ·
  <a href="https://discord.gg/RXV83nSZdk"><b>Discord</b></a> ·
  <a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.zh-TW.md">繁體中文</a> · Italiano · <a href="README.ja.md">日本語</a>
</p>

**Motore piccolo, modello immenso.** Esegui **modelli MoE di frontiera — da 744
miliardi a 2,8 mila miliardi di parametri** — su hardware consumer ed eterogeneo,
in C puro e senza dipendenze del motore, trattando storage, RAM e VRAM come
un'unica gerarchia di inferenza (multitiering della memoria per l'IA).

Oggi girano dieci famiglie di modelli linguistici. Contiamo i motori, non i
checkpoint: un file C ciascuna, la stessa interfaccia `coli chat` / `coli serve` /
`coli web`, e alcuni motori eseguono più di un modello. **GLM-5.2/5.3** (744B),
**GLM-5.3-Flash** (321B, con visione), **Inkling** (975B), **Kimi K3** (2,8T),
**DeepSeek V4 Flash** (284B), **DeepSeek V4.1 Flash** (552B, con visione),
**MiMo-V2.6 Flash** (309B, con visione; lo stesso motore esegue **MiMo-V2.6 Pro**,
1,02T), **Qwen3.8-Flash-Next** (125B + 51B n-gram), **Qwen3.6** (35B-A3B; lo stesso
motore esegue **Qwen3-Coder-30B-A3B** e il denso **Qwen3.8-27B**, con visione) e
**OLMoE** (7B).
Anche immagini: un undicesimo motore esegue **Qwen-Image-2.1**, che genera figure
da un testo, mostrate dentro al terminale da `coli chat` e servite su
`POST /v1/images/generations` ([qwen-image.md](docs/qwen-image.md)).
[Elenco completo ↓](#other-supported-models)

> **Colibrì è un motore di inferenza che puoi usare oggi, e una piattaforma di
> ricerca aperta.** Il suo obiettivo principale è migliorare le prestazioni di
> inferenza lungo l'intero confine software/hardware — formati dei modelli,
> gerarchia di memoria, I/O dello storage, piazzamento, scheduling, kernel,
> speculazione e sovrapposizione CPU/GPU — affinché i grandi modelli dipendano
> meno da hardware raro e costino meno.

Colibrì tratta VRAM, RAM e storage come un'unica gerarchia multilivello ed è
intenzionalmente un luogo dove verificare idee di sistema aggressive: quindi
**nessuno SLA sulla velocità, e una garanzia dura sulla semantica**. Gli
esperimenti devono dimostrare il proprio valore con misure end-to-end riproducibili;
la policy predefinita **non cambia mai silenziosamente la precisione del modello né
la semantica del router**. Una memoria veloce insufficiente può ridurre la velocità,
ma non ridefinire il modello di nascosto.

```
$ ./coli chat
  🐦 colibri v1.12.1 — GLM-5.2 · 744B MoE · int4 · streaming CPU
  ✓ ready in 32s · resident 9.9 GB
  › ciao!
  ◆ Ciao! 😊 Come posso aiutarti oggi?
```

## Guardalo in azione

<p align="center">
  <img src="docs/media/colibri-dashboard.png" width="900" alt="dashboard web di colibrì — metriche live, pannello hardware, livelli degli expert">
</p>
<p align="center"><em>La dashboard web (<code>./coli web</code>), ridisegnata nella 1.12.0: uno spazio di lavoro con un dock per la chat,
la modalità System One, la pagina Brain e il Profiling, in tema chiaro o scuro. Qui Qwen3.6 che risponde su una macchina
solo CPU, con gli expert letti dal disco.</em></p>

<p align="center">
  <img src="docs/media/colibri-brio.png" width="900" alt="la pagina System One: un documento letto una volta, una probabilità per ogni risposta ammessa, e un'entropia">
</p>
<p align="center"><em><strong>Modalità System One</strong>: lo stesso modello, a cui si dice di non scrivere. Gli dai un documento e le sole risposte
che può scegliere; legge la probabilità di ciascuna, non genera niente, e riporta un'entropia che dice quando non è
sicuro. Qui: <strong>request changes al 99.9%</strong>, entropia 0.005, 4 token letti, 0 generati.</em></p>

<p align="center">
  <img src="docs/media/colibri-brain.png" width="900" alt="la pagina Brain: l'atlante misurato degli expert di GLM-5.2 disegnato come una corteccia, dieci regioni da esplorare">
</p>
<p align="center"><em>La pagina <strong>Brain</strong>, <strong>Explore</strong>: l'<a href="https://github.com/JustVugg/colibri/issues/175">atlante misurato degli expert</a> di GLM-5.2
disegnato come una corteccia. 13.260 expert caratterizzati in dieci regioni (Python, SQL, matematica, poesia, legge, cinese…);
la posizione deriva dall'affinità di routing misurata, non da un embedding appreso. Si sceglie una regione e ci si entra.
<strong>Live routing</strong> passa al modello in esecuzione: una cella per expert, il colore è il livello di archiviazione, e ogni
expert instradato in un turno lampeggia bianco.</em></p>

<p align="center">
  <img src="docs/media/colibri-brain-region.png" width="900" alt="dentro la regione Python: 1.142 expert, uno selezionato con le sue affinità misurate">
</p>
<p align="center"><em>Dentro la regione <strong>Python</strong>: 1.142 expert come una costellazione, ciascuno etichettato per layer e indice. Il pannello
ne mostra uno, layer 17 expert 178: un generalista con entropia 3,13, la cui affinità misurata è 20,2% Python, 14,6% JSON,
14,2% conversazione, 13,3% SQL.</em></p>

<p align="center">
  <img src="docs/media/colibri-profiling.png" width="900" alt="la pagina Profiling: dove il motore spende ogni turno">
</p>
<p align="center"><em>La pagina <strong>Profiling</strong>: dove il motore spende ogni turno, per fase, con gli ultimi 30 turni come tendenza.
Qui Qwen3.6 su una macchina CPU: 19,0 s di tempo totale per 36 token di prompt e 55 generati, 2,9 tok/s, 11,4 s di
servizio disco sovrapposti al calcolo.</em></p>

## La missione di ricerca

Con Colibrì, l'accesso privato ai modelli di frontiera non è limitato dalla
disponibilità di hardware da hyperscaler.

Con le sue funzioni di multitiering, Colibrì **rimuove le dipendenze da hardware
proprietario ottimizzando in modo aggressivo le pipeline funzionali del motore di
inferenza**.

La nostra missione operativa comprende cambiare il modo in cui i pesi sono
rappresentati e spostati, decidere cosa risiede in VRAM, RAM o storage, sovrapporre
calcolo eterogeneo, ridurre i costi di avvio e sincronizzazione, sfruttare sparsità
e riuso e verificare nuovi algoritmi di decoding. La convenzione non protegge una
tecnica; un microbenchmark veloce non basta ad adottarla. Decide l'inferenza
end-to-end su macchine reali, misurando correttezza e qualità insieme a throughput,
latenza, memoria e costo.

Il risultato pratico è l'**accessibilità**: eseguire un modello da 744B
sull'hardware che già possiedi, osservare ogni expert attivarsi in tempo reale e
modificare il codice che lo fa. Non noleggiare intelligenza dietro un'API, ma
*possederla*: analizzarla, misurarla, migliorarla. Il motore resta volutamente
abbastanza piccolo perché la prossima ottimizzazione utile possa arrivare da
chiunque sia disposto a misurarla.

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
| Più SSD possono trasformare banda indipendente in velocità di decode | due unità NVMe indipendenti hanno misurato +37,5% in decode; una terza unità più lenta è risultata neutra dopo lo striping pesato ([misure](docs/multidisk.md#what-has-been-measured)) | riprodurre con dischi di velocità diverse, diverse disposizioni dei controller e diversi stati della cache |
| Un planner hardware-aware può avvicinarsi automaticamente alla configurazione migliore | oggi rileva budget RAM/VRAM e diversi backend | confrontare il piano generato con sweep controllati su laptop, workstation, NUMA e multi-GPU |
| Rappresentazioni lossless o a qualità limitata possono ridurre abbastanza il movimento dei pesi | esistono ablation di formato e quantizzazione con gate di qualità | riprodurre insieme qualità, byte mossi, latenza e costo per token utile, non solo il rapporto di compressione |
| La speculazione routing-aware può convenire prima della residenza quasi completa | MTP e draft grammaticali funzionano, ma MTP ha anche perso il 32% intorno all'85% di expert hit | mappare il pareggio tra accettazione, hit rate, batch union e profondità del draft |
| La sovrapposizione CPU/GPU può nascondere trasferimenti e sincronizzazione | esistono risultati positivi CUDA e Metal, ma CPU veloci e bassa residenza possono annullarli | profili per fase e A/B a variabile singola su PCIe, memoria unificata e piena residenza |

Per contribuire, scegli una riga e pubblica anche i risultati negativi. Registra
hardware, commit, container del modello, comando esatto, prompt, stato cache,
throughput, TTFT, expert hit, byte letti e controllo qualità; cambia una sola
variabile, ripeti e allega i log grezzi. Parti da
[CONTRIBUTING.md](CONTRIBUTING.md), confronta il
[protocollo di benchmark](docs/benchmarking.md), quindi
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

Pensa all'algoritmo di base come a **un JIT, ma per i pesi**. Un compilatore JIT
non compila mai l'intero programma: osserva cosa viene davvero eseguito e compila i
percorsi caldi, appena in tempo. colibrì fa la stessa scommessa su uno spazio di
744B parametri: i parametri non sono uno stato residente da custodire, sono **dati
da collocare** lungo una gerarchia di storage eterogenea (VRAM / RAM / NVMe),
esattamente quando il router dimostra che servono. Il calore di routing misurato
decide quale livello si guadagna ogni expert, il router lavora un layer in anticipo
così il prefetch nasconde la latenza di caricamento e, come un JIT, il motore impara
il tuo carico di lavoro: più lo usi, più si scaldano gli expert giusti. Funziona
perché il routing ha una struttura misurabile (vedi
l'[atlante degli expert](https://github.com/JustVugg/colibri/issues/175)), e una
struttura si può mettere in cache.

Il motore è un singolo file C (`c/colibri.c`) più header piccoli. Niente BLAS,
niente Python a runtime, niente GPU obbligatoria.

### Modalità cluster locale

Il coordinatore tiene in locale la generazione dei token, il routing e lo stato KV,
mentre worker di expert basati su disco eseguono le FFN instradate su altri Mac. La
batch-union instradata di un layer viene inviata come un'unica richiesta TCP
persistente, così un token non paga un round trip per ogni expert.

Avvia il servizio di registrazione opzionale:

```bash
./coli cluster coordinator --host 0.0.0.0 --port 8765
```

Su ogni worker, con lo stesso modello convertito disponibile in locale:

```bash
./coli cluster worker --model /nvme/glm52_i4 --port 9100 \
  --coordinator http://COORDINATOR:8765 --advertise-host WORKER_IP
```

Avvia il coordinatore con la discovery, oppure passa `--cluster-workers
HOST:PORT,...` per una configurazione statica:

```bash
./coli serve --model /nvme/glm52_i4 \
  --cluster-coordinator http://127.0.0.1:8765
```

Il trasporto resta disattivato finché non si configurano dei worker, quindi il
percorso esistente su una sola macchina non cambia. Lo sharding dei layer densi e i
worker browser/WebGPU sono passi successivi separati.

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

<a id="dual-ssd-two-copies-of-the-model-twice-the-read-bandwidth"></a>

### Più SSD: leggere copie del modello da più di un disco

Quando il decode è limitato dal disco, un **secondo SSD** può aiutare: mettici una
copia del modello e lascia che il motore legga da entrambi i dischi. Per GLM-5.2, da
`c/` in un checkout dei sorgenti (o da una release scompattata):

```bash
COLI_MODEL_MIRROR=/second/glm52_i4 python3 ./coli chat --model /fast/glm52_i4
```

Il motore misura i dischi all'avvio per pesare la ripartizione delle letture. Le
letture bufferizzate usano un instradamento deterministico degli expert; le letture
dirette idonee possono distribuire un expert tra le repliche. Dischi indipendenti
danno margine di banda, non un moltiplicatore garantito della velocità in token:
controller condivisi, hit in cache e calcolo possono limitare il guadagno. Vedi la
[guida multi-disco](docs/multidisk.md) per esempi in Bash e PowerShell, guadagni e
limiti misurati e un confronto con un solo disco. Dettagli utili:

- il mirror è **validato all'avvio** (dimensione di ogni file e header safetensors devono essere identici byte per byte al primario); i file divergenti o mancanti restano sul primario, quindi un **mirror parziale va bene**: un secondo SSD più piccolo può servire gli shard che contiene;
- il mirror **non viene mai scritto**: `.coli_usage`, `.coli_kv` e tutti i sidecar restano sul primario;
- un errore di lettura sul mirror ripiega sul primario (un avviso, nessun crash), quindi scollegare il secondo disco durante l'esecuzione degrada le prestazioni invece di far cadere il server;
- il routing non cambia mai i token, perché le due copie sono identiche byte per byte; attiva `PROF=1` per i contatori di profilo `MIRROR:` che mostrano i GB serviti da ciascun disco.

Lo stesso motore copre l'intero spettro: su un portatile da 25 GB tutto viene
caricato dal disco in streaming (lento, ma corretto); su un host grande l'intero
set di expert diventa residente (`CUDA_EXPERT_GB=auto PIN_GB=all`) e il disco
esce completamente dal percorso di decode. Tra i livelli c'è una **cache che
impara**: il motore registra quali expert il *tuo* carico di lavoro instrada
(`.coli_usage`, aggiornato a ogni turno) e fissa automaticamente i più caldi —
colibrì diventa letteralmente più veloce man mano che lo usi. Sugli host
multi-socket, `COLI_NUMA=1` interlaccia i pesi residenti tra i controller di
memoria ([#82](https://github.com/JustVugg/colibri/issues/82)).

Per un secondo disco che non può contenere l'intero modello, Colibri può
classificare un mirror parziale in base alla cronologia degli expert che già
impara. Esegui prima qualche prompt rappresentativo, così `.coli_usage` riflette il
carico di lavoro, poi pianifica, prepara e verifica il mirror:

```bash
./c/coli mirror plan  --model /fast/glm52_i4 --mirror /second/glm52_i4 \
  --budget-gib 200 --reserve-gib 20
./c/coli mirror stage --model /fast/glm52_i4 --mirror /second/glm52_i4 \
  --budget-gib 200 --reserve-gib 20
./c/coli mirror verify --model /fast/glm52_i4 --mirror /second/glm52_i4
```

Il planner legge direttamente gli header safetensors, segue le directory di modelli
divisi indicate in `COLI_MODEL_DIRS` e dà priorità agli shard che possono servire
gli expert instradati più caldi. La preparazione non modifica mai il modello
primario: copia passando per file temporanei, rispetta la riserva di spazio libero
richiesta, verifica ogni shard con SHA-256, non cancella mai uno shard di mirror
esistente e pubblica in modo atomico una ricevuta solo quando il mirror selezionato
è pronto.

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
GPU a memoria unificata; e un [backend Vulkan](docs/vulkan.md) porta il livello
degli expert, le proiezioni dense e il nucleo dell'attenzione MLA su qualsiasi GPU
con un driver Vulkan 1.2, comprese le schede AMD tramite Mesa/RADV (l'unico backend
per le schede che gli stack dei produttori non supportano più, come la RX 580, e
competitivo con ROCm su RDNA4: vedi [le note di benchmark](docs/vulkan.md)).
Ogni altro motore ora apre lo stesso backend per le sue matrici residenti con
`COLI_VULKAN=1` in una build `VK=1`; in CI è verificato contro i token della CPU, ma
sulla prima GPU reale su cui è stato misurato, una Radeon 780M integrata, oggi è più
lento della CPU ([gli altri motori](docs/vulkan.md#the-other-engines)).

> **Su NVMe reali, misura `DIRECT=1`.** O_DIRECT scavalca la page cache e spesso è
> un grande vantaggio sui dischi con cache DRAM e margine di banda (+34% in decode
> misurato con `PIPE=1` su una macchina Blackwell/Windows; da 4,25 a 9,69 GB/s in
> iobench su un GB10), ma dipende dal disco: unità QLC, senza DRAM o virtualizzate
> possono essere neutre o peggiorare. Provalo prima; tieni ciò che il tuo hardware
> premia.

### Modello fedele, stato compresso

Il forward pass è validato contro un oracle `transformers` (teacher-forcing di
norma 30-32/32; due posizioni dell'oracle tiny sono quasi-pareggi in virgola mobile
e dipendono dalla toolchain). L'attenzione MLA memorizza uno stato KV compresso
(576 float/token invece di 32.768, **57× più piccolo**) e lo persiste tra i
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

I batch di verifica possono anche attivare un **nucleo di attenzione esatto** con
`COLI_EXACT_VERIFY=1` ([#689](https://github.com/JustVugg/colibri/issues/689)): i
prodotti scalari di score e di contesto dell'MLA-absorb su CPU accumulano prodotti
interi e arrotondano una sola volta, così un quasi-pareggio in una riga di verifica
si risolve allo stesso modo su ogni host, a circa 0,6x tok/s su un oracle tiny (il
prodotto scalare in sé costa circa 5-7x il ciclo in float). Due limiti da
conoscere: con una cache KV quantizzata (`tq1`, TQ o KV int8) il prodotto di
contesto resta sul percorso float, quindi lì l'esattezza non è garantita; e un vero
ribaltamento da quasi-pareggio è stato solo argomentato, non ancora osservato su
GLM-5.2 a n=64.

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

### In un solo passo

**Windows:** scarica il repository (**Code**, poi **Download ZIP**, oppure
`git clone`), scompattalo e fai doppio clic su **`START-HERE.bat`**.
**Linux e macOS:**

```bash
git clone https://github.com/JustVugg/colibri && cd colibri
./start-here.sh
```

Rileva RAM, disco e GPU, consiglia un modello adatto a questa macchina (Invio
lo accetta), compila il motore con Vulkan o CUDA quando la tua GPU può usarli
(oppure scarica quello già compilato), scarica il modello con ripresa
(interrompilo quando vuoi, rilancialo per continuare) e apre la dashboard nel
browser. Stampa anche gli URL base OpenAI e Anthropic per le altre app.
Rilancialo più tardi e colibri parte subito; `c/coli stop` lo ferma. Cosa fa
ogni passo: [quickstart.md](docs/quickstart.md#the-one-step-way).

Usi un assistente di programmazione AI? Chiedigli di installare colibri seguendo [docs/AI_SETUP.md](docs/AI_SETUP.md).
Gli assistenti che parlano il Model Context Protocol possono usare `coli mcp`
([MCP_SERVER.md](docs/MCP_SERVER.md)).

Segue il percorso manuale.

### 1. Procurati colibri

**Scarica una release già compilata** — Linux, macOS e Windows, nessun
compilatore necessario. Prendi l'archivio della tua piattaforma dalla pagina
[Releases](https://github.com/JustVugg/colibri/releases) e scompattalo:

```bash
mkdir colibri && tar xzf colibri-v1.8.0-linux-x86_64.tar.gz -C colibri && cd colibri
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

**GLM-5.3** è la stessa famiglia e si carica con lo stesso motore. Ha il suo
container group-scaled (gs64), circa **419 GB**, e arriva **senza** la testa MTP,
quindi la decodifica speculativa resta disattivata:

**https://huggingface.co/Justvugg/GLM-5.3-colibri-int4-g64**

> ⚠️ Usa il container **gs64** qui sopra, non i vecchi mirror int4 per-row
> (`mateogrgic/…`, `jlnsrk/…`): misurano circa 9 punti percentuali in meno sulla
> qualità e causavano i loop in think-mode e le generazioni senza termine originali
> di [#455](https://github.com/JustVugg/colibri/issues/455). Il container gs64 ha
> corretto quegli A/B per-row controllati, ma non è una protezione generale contro
> ripetizioni o EOS starvation. Anche la testa MTP deve essere **int8, non int4**
> (int4 → 0% di accettazione dei draft,
> [#8](https://github.com/JustVugg/colibri/issues/8)):
> `ls -l <modello>/out-mtp-*`: int8 (corretto) è `3527131672 / 5366238584 / 1065950496`
> in tre file, oppure un solo `out-mtp-00000.safetensors` da `9959321520` byte
> (il caricamento attuale del container consigliato lo distribuisce come un unico
> file: stessi tensori int8, 777, a un byte per elemento).

Oppure converti tu stesso dalla sorgente FP8 — un unico comando riprendibile che
non richiede mai i 756 GB completi su disco contemporaneamente:

```bash
./coli convert --model /nvme/glm52_i4     # scarica e converti shard per shard (python, una tantum)
```

<a id="other-supported-models"></a>
#### Altri modelli supportati

GLM-5.2 è il modello di riferimento, ma lo stesso approccio in streaming fa girare
altre nove famiglie di modelli linguistici, e un motore genera immagini. Ognuna è un
**motore fratello**: un file C, la sua architettura, la stessa interfaccia
`coli chat` / `coli serve` / `coli web` (il launcher sceglie il binario dal
`config.json` del modello, o da `model_index.json` per il modello di immagini):

> **Cosa serve a ciascuno.** Sono molto diversi, e leggerne due insieme ha fatto
> credere ad alcuni che i requisiti si contraddicano
> ([#191](https://github.com/JustVugg/colibri/issues/191)). Non è così: sono
> modelli diversi. **Nessuno di loro richiede una GPU.**
>
> | Modello | Disco per i pesi | RAM | GPU |
> |---|---|---|---|
> | **OLMoE** | ~7 GB (container int8) | 8 GB | non serve; Vulkan opzionale |
> | **GLM-5.2/5.3** | ~372 GB (5.2) / ~419 GB (5.3) | 16 GB minimo, 24 GB comodi | non serve; Vulkan opzionale |
> | **GLM-5.3-Flash** | ~195 GB dopo la conversione | 25 GB (12 GB di pesi a int4 + cache degli expert) | non serve; Vulkan opzionale |
> | **Inkling** | ~469 GB | 25 GB con il container denso int4, ~120 GB senza | non serve; Vulkan opzionale |
> | **Kimi K3** | ~1,6 TB | 32 GB+ | non serve; Vulkan opzionale |
> | **DeepSeek V4 Flash** | ~167 GB (REAP 150B: ~85 GB) | 16 GB minimo, 32 GB comodi | opzionale; qualsiasi scheda NVIDIA dalla serie GTX 10 in su (Pascal/Turing con `CUDA_ARCH=portable-pre-ampere NO_TC=1`, al meglio su RTX 50) rende il prefill 5-10x e il decode ~2,5x più veloci; Vulkan opzionale |
> | **DeepSeek V4.1 Flash** | ~510 GB (checkpoint ufficiale; 203 GB sono una memoria n-gram letta poche centinaia di byte alla volta) | ~18 GB residenti (parte densa, embedding, visione) più la cache degli expert dimensionata da `--ram`; 24,8 GB di RSS di picco misurati a cap 8 | non serve; Vulkan opzionale |
> | **MiMo-V2.6 Flash** | ~178 GB (checkpoint ufficiale) | 30,1 GB residenti misurati con 32 expert in cache per layer, 49,8 GB con 64; `--ram` dimensiona la cache | non serve; Vulkan opzionale |
> | **MiMo-V2.6 Pro** | ~574 GB (~564 GB senza i tre file che il motore non carica mai) | parte densa 30,2 GiB come rilasciata, 21,7 GiB in int8; 48,0 GB residenti misurati con 12 expert in cache per layer (parte densa come rilasciata), 50,7 GB con 20 (parte densa in int8) | non serve; Vulkan opzionale |
> | **Qwen3.8-Flash-Next** | ~185,5 GB (checkpoint FP8 ufficiale), più 68,0 GB per il sidecar opzionale degli expert int4-g64 | 24 GB comodi al contesto predefinito con gli expert FP8 (cap 32; 16 GB sono sotto il minimo); 11,6 GB di RSS misurati a cap 32 con il sidecar int4-g64 | opzionale; il livello expert CUDA in VRAM (solo expert FP8) con il tronco denso quantizzato a int8 in VRAM; Vulkan opzionale |
> | **Qwen3.8-27B** (denso, testo e immagini) | ~51 GB dopo la conversione (f16) | 20 GB con pesi densi int4, 30 GB in int8 | non serve; ancora nessun livello CUDA; Vulkan opzionale |
> | **Qwen3.6-35B-A3B** | ~20 GB (container int4-gs64) | 24 GB (richiede la residenza completa in RAM) | opzionale; il livello expert CUDA in VRAM ha misurato **1,44 -> 10,05 tok/s (7,0x)** su due schede da 8 GB, output identico bit per bit alla CPU; Vulkan opzionale |
> | **Qwen3-Coder-30B-A3B** | ~19 GB (container int4-gs64; 30 GB in int8) | 6,5 GB residenti misurati con 32 expert in cache per layer, 15,2 GB con tutti i 128 | non serve; Vulkan opzionale |
> | **Qwen-Image-2.1** (da testo a immagine) | ~33 GB (checkpoint diffusers ufficiale) | 16,0 GB con tutto residente, 8,5 GB di picco con il text encoder caricato a ogni prompt, più i buffer di lavoro (9,0 GB di picco misurati per un'immagine 768x512) | non serve; Vulkan opzionale, non ancora cronometrato su GPU |
>
> Una GPU non cambia mai ciò che il modello risponde, solo dove si svolge il
> lavoro. La velocità la decide il disco, perché gli expert vengono letti in
> streaming da lì: aspettati una frazione di token al secondo su un disco lento e
> qualche token al secondo su uno veloce con la cache calda.
>
> **Vulkan opzionale** significa una build `VK=1`. Avviato con `COLI_VULKAN=1`, il
> motore mette le sue matrici residenti su qualsiasi GPU con un driver Vulkan 1.2
> (GLM-5.2 ha lì un percorso di decode completo), e la CI verifica quei motori
> contro i token della CPU su un driver
> software ([vulkan.md](docs/vulkan.md#the-other-engines)). Corretto non vuol dire
> ancora più veloce. Sulla prima GPU reale misurata per questi motori, una Radeon
> 780M integrata (Ryzen 7 PRO 8700GE, stessi binari, page cache fredda), oggi è più
> lento della CPU: Qwen3.6-35B-A3B ha fatto 3,06 tok/s in decode contro 5,97 sulla
> CPU, con output identico, e Qwen3.8-Flash-Next con expert int4 1,53 contro 3,55.
> Ogni matmul è una submit sincrona, circa 726 per token, e una GPU integrata legge
> la stessa RAM della CPU.

| Famiglia | Totali / attivi | Pesi | Build | Documentazione |
|---|---|---|---|---|
| **GLM-5.2/5.3** | 744B / 40B | [`mastouri/…-int4-g64-with-int8-mtp`](https://huggingface.co/mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp) (372 GB) o [`Justvugg/GLM-5.3-colibri-int4-g64`](https://huggingface.co/Justvugg/GLM-5.3-colibri-int4-g64) (419 GB) | `make -C c glm` | questa pagina |
| **Inkling** (Thinking Machines) | 975B / 41B | [`nbeerbower/Inkling-colibri-int4`](https://huggingface.co/nbeerbower/Inkling-colibri-int4) (469 GB) | `make -C c inkling` | [inkling.md](docs/inkling.md) |
| **GLM-5.3-Flash** (Z.ai) | 321B / 18B | [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash), convertito con gli expert instradati in **int4-gs64**; la parte densa resta BF16 e la sua precisione si sceglie al caricamento; visione inclusa | `make -C c glm53` | [glm53-flash.md](docs/glm53-flash.md) |
| **Kimi K3** (Moonshot) | 2,8T / 104B | [`moonshotai/Kimi-K3`](https://huggingface.co/moonshotai/Kimi-K3): checkpoint originale, gli expert instradati restano **MXFP4 nativi** | `make -C c kimi_k3` | [kimi_k3.md](docs/kimi_k3.md) |
| **DeepSeek V4 Flash** | 284B / 13B | checkpoint ufficiale in shard: gli expert instradati restano **fp4 nativi**, la parte densa resta fp8-e4m3; il **150B potato con REAP** ([`puwaer/DeepSeek-V4-Flash-0731-reap-150b`](https://huggingface.co/puwaer/DeepSeek-V4-Flash-0731-reap-150b), 85 GB, 132 expert su 256) si carica con lo stesso motore e senza conversione | `make -C c deepseek-v4` | [deepseek-v4.md](docs/deepseek-v4.md) |
| **DeepSeek V4.1 Flash** | 552B / 16B | checkpoint ufficiale, **nessuna conversione**: gli expert sono già fp4, la parte densa è fp8-e4m3. 203 GB sono una memoria n-gram letta dal disco poche centinaia di byte alla volta, e gli expert instradati costano **4,5 GB per token** contro i 12,7 di GLM-5.2. Visione, tool calling e la testa di draft DSpark sono tutti attivi | `make -C c deepseek_v41` | [deepseek-v41.md](docs/deepseek-v41.md) |
| **MiMo-V2.6 Flash** (Xiaomi) | 309B / 15B | [`XiaomiMiMo/MiMo-V2.6-Flash-MOPD`](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD) (178 GB), checkpoint ufficiale, **nessuna conversione**: gli expert instradati restano **MXFP4 nativi**, la parte densa resta FP8/BF16. 39 dei suoi 48 layer guardano una finestra di 128 token, quindi un contesto lungo costa il KV di 9. Visione e tool calling attivi | `make -C c mimo` | [mimo.md](docs/mimo.md) |
| **MiMo-V2.6 Pro** (Xiaomi) | 1,02T / 42B | [`XiaomiMiMo/MiMo-V2.6-Pro-MOPD`](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD) (573,5 GB), checkpoint ufficiale, **nessuna conversione**, sul motore MiMo: la stessa architettura a 70 layer e 384 expert. Verificato contro il codice di modellazione di Xiaomi sui primi 32 dei suoi 70 layer. Visione e tool calling attivi | `make -C c mimo` | [mimo.md](docs/mimo.md#pro) |
| **Qwen3.8-Flash-Next** (Alibaba) | 125B + 51B n-gram / 6B | [`Qwen/Qwen3.8-Flash-Next-FP8`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8), checkpoint originale; la PLE resta paginabile e gli expert restano **block-FP8 nativi**, oppure vengono letti come **int4-g64** da un sidecar opzionale (vedi sotto). Draft MTP opt-in (`Q38_MTP=1`) | `make -C c qwen38` (`CUDA=1` per il livello expert in VRAM) | [qwen38.md](docs/qwen38.md) |
| **Qwen3.6** (Alibaba) | 35B / 3B | [`Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64`](https://huggingface.co/Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64) (~20 GB, **consigliato**): ibrido Gated Attention + Gated DeltaNet | `make -C c qwen36` (`CUDA=1` per il livello expert in VRAM) | [qwen36.md](docs/qwen36.md) |
| **Qwen3.8-27B** (Alibaba) | 27B, denso | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) convertito con `c/tools/convert_qwen36.py` in un container f16 (51 GB); il motore lo quantizza a int8, o a int4 con `COLI_DENSE_BITS=4`, durante il caricamento. Un MLP per layer e nessun router, sul motore Qwen3.6. Testo e immagini | `make -C c qwen36` | [qwen36.md](docs/qwen36.md#the-dense-27b) |
| **Qwen3-Coder-30B-A3B** (Alibaba) | 30B / 3B | [`Justvugg/Qwen3-Coder-30B-A3B-colibri-int4`](https://huggingface.co/Justvugg/Qwen3-Coder-30B-A3B-colibri-int4) (19 GB, int4-gs64), convertito da [`Qwen/Qwen3-Coder-30B-A3B-Instruct`](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct). MoE Qwen3 di sola attenzione sul motore Qwen3.6, 128 expert top-8, una sua forma XML per le chiamate ai tool e nessun thinking; in teacher forcing il container int4 sceglie il token top-1 della release bf16 nel 96,9% delle posizioni | `make -C c qwen36` | [qwen36.md](docs/qwen36.md#qwen3-coder-30b-a3b) |
| **OLMoE** (AI2) | 7B / 1B | convertito con `c/tools/convert_olmoe_merged.py`: container **int8**, ~7 GB | `make -C c olmoe` | - |
| **Qwen-Image-2.1** (Alibaba) | modello di immagini | [`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1) (~33 GB), checkpoint diffusers ufficiale, **nessuna conversione**: il text encoder e il diffusion transformer vengono quantizzati a int8 durante il caricamento. Immagini dentro `coli chat`, `POST /v1/images/generations` su `coli serve`. Qwen Research License: solo uso non commerciale | `make -C c qwenimage` | [qwen-image.md](docs/qwen-image.md) |
| **Laya** (Convai Innovations) | modello di decisione, 421M | [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) (842 MB), checkpoint ufficiale, **nessuna conversione**: un encoder ModernBERT con una testa di decisione che risponde a domande tipizzate (choice, score, noul) con probabilità calibrate invece di generare. Servito su `POST /v1/systemone` da `coli serve`. Apache-2.0 | `make -C c laya` | [laya.md](docs/laya.md) |
| **GLiNER2.5-Decide** (fastino) | modello di decisione, 340M | [`fastino/GLiNER2.5-Decide`](https://huggingface.co/fastino/GLiNER2.5-Decide) (1,9 GB), checkpoint ufficiale, **nessuna conversione**: un encoder DeBERTa-v3 con la testa di classificazione di GLiNER2; le domande di una richiesta e lo stato si leggono in un solo passaggio e ogni opzione riceve una probabilità. Servito su `POST /v1/systemone` da `coli serve`. Apache-2.0 | `make -C c gliner_decide` | [gliner_decide.md](docs/gliner_decide.md) |

Qwen3.6 offre tre container pre-convertiti: **int4-gs64** (consigliato: coseno
misurato rispetto all'ancora int8 da 0,98777 a 0,99313 e KL da 0,109 a 0,080
rispetto al per-row, cioè ~44% di errore di quantizzazione in meno),
[int4 per-row](https://huggingface.co/Kreuzzelg/qwen36-35b-a3b-colibri-i4) come
baseline per gli A/B, e
[KAT-Coder v2.5](https://huggingface.co/Kreuzzelg/kat-coder-v2.5-dev-colibri-i4-gs64),
che lo stesso motore esegue senza modifiche: qualsiasi checkpoint con architettura
identica funziona senza un percorso di codice dedicato. Con `CUDA=1` il livello
expert in VRAM ha misurato **da 1,44 a 10,05 tok/s (7,0×) su due schede da 8 GB**,
con output identico bit per bit al percorso CPU.

Qwen3.8-Flash-Next legge i suoi expert instradati come rilasciati, in block-FP8. Un
**sidecar int4-g64** opzionale (`c/tools/convert_qwen38_experts_int4.py`, 68,0 GB
scritti accanto agli shard FP8) fa leggere a ogni miss il 56% dei byte. Misurato su
un Ryzen 7 PRO 8700GE (16 thread, 61 GiB, NVMe), il decode è 1,4-1,5x più veloce a
parità di cache e 1,56x più veloce a parità di RAM, per +0,017 nats per token di
perplessità in media ([qwen38.md](docs/qwen38.md#routed-experts-as-int4-g64)). La
decodifica speculativa con la testa MTP del checkpoint è opt-in (`Q38_MTP=1`), e
l'output è identico alla decodifica normale. Sulla stessa macchina con gli expert
int4, il 94-96% dei draft è stato accettato, 1,94 token per forward, per +12-14% di
tok/s (da 3,57 a 4,01 a cap 96, da 4,15 a 4,74 a cap 170). Lì il guadagno è piccolo
perché le letture degli expert dal disco non si riducono
([qwen38.md](docs/qwen38.md)).

Kimi K3 non richiede conversione: i suoi expert MXFP4 addestrati con QAT vengono
letti in streaming direttamente dagli shard originali di Hugging Face, e la parte
densa bf16 viene quantizzata al caricamento. Le sessioni lunghe degli agenti
possono attivare i checkpoint dello stato ricorrente (`COLI_K3_CKPT=N` slot in RAM,
o parcheggiati su disco con `COLI_K3_CKPT_DIR`): un prompt modificato o di
follow-up ripristina il checkpoint più profondo ancora disponibile e rifà il
prefill solo della coda, invece di ripercorrere l'intera conversazione attraverso i
layer SSM. Sugli host Vulkan (`COLI_VULKAN=1`) i suoi expert instradati entrano nel
livello di expert condiviso, riempito dalla storia degli expert e aggiornato man mano
che il routing cambia. I percorsi KDA e MLA del motore sono
validati token-esatti in CI contro l'implementazione del vendor.

Inkling distribuisce expert int4 ma **pesi densi bf16** (49,4 GB residenti); su un
host che non può tenerli, [inkling.md](docs/inkling.md) ha uno strumento a passata
singola che porta la parte densa a 15,3 GB e fa girare il 975B su una macchina da
25 GB, con il compromesso messo onestamente per iscritto.

### 3. Esegui

```bash
COLI_MODEL=/nvme/glm52_i4 ./coli chat     # budget RAM, cache e MTP rilevati automaticamente
COLI_MODEL=/nvme/glm52_i4 ./coli plan     # mostra il piazzamento pianificato VRAM/RAM/disco
COLI_MODEL=/nvme/glm52_i4 ./coli doctor   # controllo di idoneità (sola lettura)
COLI_MODEL=/nvme/glm52_i4 ./coli doctor --deep  # controllo preliminare rigoroso di tensori/shard/indice/mirror
COLI_MODEL=/nvme/glm52_i4 ./coli tune     # misura e salva il profilo di esecuzione sicuro più veloce di questa macchina
./coli web  --model /nvme/glm52_i4        # API + dashboard, e apre un browser
./coli serve --model /nvme/glm52_i4       # API + dashboard, senza browser (headless)
```

#### Modalità System One: una domanda a risposta chiusa

Gran parte di ciò che si chiede a un modello è una scelta, non un paragrafo:
quale coda, quale verdetto, quale dei quattro valori può prendere un campo. La
modalità System One passa al motore le opzioni e legge la probabilità di ciascuna
invece di generare: non si genera nulla, nessuna risposta può uscire dalla tua
lista, e ogni risposta arriva con una confidenza, così "il modello non è sicuro"
è un numero su cui mettere una soglia. Funziona su tutte e dieci le famiglie,
sullo stesso server, ed è opzionale per richiesta: la chat resta identica byte
per byte per chi non la chiede.

```bash
# nella TUI: lo stesso modello, a cui si dice di non scrivere
./coli chat --model /nvme/qwen36_i4_gs64
> /decide merge | request changes | close
> 340 lines, 8 files, no tests. CI is green but nothing covers that path.

# da qualunque programma: una richiesta JSON al server in esecuzione
curl -s http://127.0.0.1:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "340 lines, 8 files, no tests. CI is green but nothing covers that path.",
  "questions": {"review": {"type": "choice", "instructions": "What should the reviewer do?",
                           "criteria": {"merge": null, "request changes": null, "close": null}}}}'
```

`POST /v1/systemone` parla la richiesta e la risposta dell'API Jev di TypeSafe:
un client Jev passa a colibri cambiando solo il base URL. Più domande sullo
stesso documento lo leggono una volta sola: misurato su Qwen3.6 contro la
generazione delle stesse risposte sulla stessa macchina CPU, 5,7x su quattro
domande sullo stesso documento. Tutta la modalità, la forma di richiesta e
risposta, e dove non serve: [docs/systemone.md](docs/systemone.md). Anche la
dashboard ha una pagina System One.


Su Windows un archivio di release include `coli.cmd`: fai doppio clic per l'avvio
rapido, oppure esegui `coli.cmd chat --model D:\glm52_i4` da cmd o PowerShell. Da
un checkout dei sorgenti gli stessi comandi funzionano con `python coli chat --model
D:\glm52_i4`. I file `.exe` sono i motori, non il launcher: avviati da soli non
hanno un modello da caricare ed escono subito.
Il motore a runtime è puro C: python si usa solo per il convertitore (una tantum)
e per il gateway API opzionale.

#### Gli stessi comandi per tutti i modelli

`coli` legge il `config.json` del modello, sceglie il binario del motore
corrispondente e applica il chat template di quella famiglia: quindi **la riga di
comando non cambia tra un modello e l'altro**. Compila una volta il motore che ti
serve, poi punta `COLI_MODEL` alla directory giusta:

```bash
make -C c glm                                     # GLM-5.2
make -C c inkling                                 # Inkling
make -C c kimi_k3                                 # Kimi K3

COLI_MODEL=/nvme/glm52_i4      ./coli chat        # TUI
COLI_MODEL=/nvme/inkling_i4    ./coli chat
COLI_MODEL=/nvme/kimi_k3       ./coli chat

./coli web --model /nvme/inkling_i4               # API + dashboard, opens a browser
./coli web --model /nvme/kimi_k3
./coli serve --model /nvme/inkling_i4             # API + dashboard, no browser
```

Per i motori diversi da GLM, `coli chat` avvia il gateway in locale e ci collega
la TUI, così TUI, API e dashboard passano tutti per lo stesso chat template che
conosce l'architettura: non devi mai passare il template a mano.

Due cose cambiano da un modello all'altro, entrambe documentate nella pagina del
modello:

- **Inkling su un host con poca RAM** richiede il container denso int4 e una cache
  degli expert piccola: `./coli chat --model /nvme/inkling_i4 --cap 2`
  (vedi [inkling.md](docs/inkling.md): il `--cap 8` predefinito vuole ~14 GB di
  cache oltre al set residente).
- **Kimi K3** legge in streaming i suoi expert MXFP4 dal checkpoint originale,
  quindi non c'è niente da convertire, ma lo snapshot è di ~1,6 TB
  (vedi [kimi_k3.md](docs/kimi_k3.md)).

### 4. Approfondisci

| argomento | documento |
|---|---|
| Benchmark, dati dalla comunità, misurazioni di qualità | [docs/benchmarks.md](docs/benchmarks.md) |
| Protocollo di benchmark riproducibile e report minimo | [docs/benchmarking.md](docs/benchmarking.md) |
| Parametri di tuning, policy, cache che impara, prefetch | [docs/tuning.md](docs/tuning.md) |
| Build nativa su Windows 11 (con CUDA DLL) | [docs/windows.md](docs/windows.md) |
| Backend CUDA, livello expert in VRAM, residenza completa | [docs/cuda.md](docs/cuda.md) |
| Backend Vulkan (qualsiasi GPU: AMD tramite RADV, comprese le schede non più supportate da ROCm) | [docs/vulkan.md](docs/vulkan.md) |
| Backend Metal per Apple Silicon | [docs/metal.md](docs/metal.md) |
| API compatibile OpenAI, KV slot, dashboard web | [docs/api.md](docs/api.md) |
| Modalità System One: punteggiare un insieme chiuso di opzioni invece di generare | [docs/systemone.md](docs/systemone.md) |
| ABI sperimentale di embedding per segmenti di layer | [docs/segment-runtime.md](docs/segment-runtime.md) |
| ABI Edge sperimentale per tokenizer/embedding/head | [docs/edge-runtime.md](docs/edge-runtime.md) |
| Draft forzati da grammatica (output strutturato) | [docs/grammar-draft.md](docs/grammar-draft.md) |
| Inventario delle variabili d'ambiente | [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) |

## DeepSeek V4

**DeepSeek V4 Flash** legge in streaming il checkpoint ufficiale senza conversione:
gli expert instradati restano **fp4 nativi**, la parte densa resta **fp8-e4m3** con
scale di blocco UE8M0. Attenzione sparsa MLA + DSA, 43 layer, 256 expert instradati
più uno condiviso, top-6. Supportato su Linux x86-64/aarch64 e su Windows/MSYS2
(CPU), con un livello CUDA opzionale (DLL di runtime su Windows; link diretto
`CUDA=1` su Linux, verificato sotto WSL2) che mantiene ogni fase canonica rispetto
alla CPU e ripiega fase per fase.

```bash
cd c
make deepseek-v4
python ./coli chat --model /path/to/DeepSeek-V4-Flash --ram 32
# also: coli run / coli serve / coli web
# Windows CUDA tier: make cuda-dsv4-dll CUDA_ARCH=portable  (+ make cuda-dsv4-dg-dll on RTX 50)
```

Due leve GPU opt-in sono nuove e cercano numeri dalla comunità, entrambe disattivate
per default e identiche byte per byte quando non impostate: `DSV4_HYBRID=1` divide i
miss del livello VRAM tra il ramo di riempimento GPU e il ramo CPU usando le bande
misurate a runtime, e `COLI_CUDA_MOE_DOUBLE=1` (insieme a `COLI_CUDA_MOE_BATCH=1`)
precarica l'intero set di expert del layer successivo in un secondo banco di VRAM
mentre il layer corrente calcola, e ripiega sul banco singolo quando la VRAM non
basta. Il livello CUDA ora gira anche sulle schede Pascal e Turing (serie GTX 10 /
RTX 20): compila con `CUDA_ARCH=portable-pre-ampere NO_TC=1`.

Decode greedy e un solo slot KV. Il tool calling passa per il gateway HTTP con il
prompt nativo di V4 e i blocchi di chiamata DSML; la grammatica non è supportata.
Vedi la [matrice API per motore](docs/api.md#tool-calling-support). I checkpoint di
prefisso (in memoria e su disco) fanno partire le sessioni degli agenti e i turni
successivi in pochi secondi dopo il primo prefill di un system prompt. Misurato su
una RTX 5080 + 2 NVMe: prefill di 3324 token in 90 s, primo turno di 8,3k token in
~4 min una volta sola, sessioni e turni successivi in 6-9 s, decode ~1,6 tok/s a 3k
di contesto: vedi [docs/deepseek-v4.md](docs/deepseek-v4.md).

**Dagli RAM.** 43 × 256 expert instradati occupano ~137 GiB su disco e un token ne
tocca 301, quindi è il tasso di hit della cache degli expert a decidere i tok/s:
`--ram` è la manopola più preziosa in assoluto, e cambia solo la velocità, mai
l'output.

**Il drafting speculativo esiste ed è spento.** Il drafter markoviano di DSpark e
l'MTP completo sono entrambi implementati e verificati: un draft può risparmiare
forward pass ma non cambiare mai un token, perché ogni token accettato è comunque
l'argmax del modello target. Misurati su vere chat a più turni, hanno accettato 1
su 15 e 10 su 24, e il replay del suffisso rifiutato sullo stato di attenzione
ricorrente di questo motore è costato più di quanto i draft abbiano risparmiato:
una risposta di 14 token ha richiesto 495 secondi. Per questo `V4_DRAFT` e `V4_MTP`
sono `0` per default e il codice resta, con i numeri accanto, per chi vorrà
riprovarci su uno storage più veloce.

Vedi [docs/deepseek-v4.md](docs/deepseek-v4.md) per il livello CUDA (build, scelta
della DLL, copertura GPU), il riferimento delle variabili d'ambiente, i numeri di
prestazione, la validazione del checkpoint e l'oracle tiny indipendente generato.

## Prossimi passi

- **La ricerca sui sistemi di inferenza è il prodotto.** La gerarchia attuale usa
  LRU e un insieme appreso di expert fissati; il lavoro attivo copre formati,
  compressione, piazzamento, scheduling, I/O, kernel CPU/GPU, sovrapposizione
  eterogenea, stato KV e speculazione consapevole del routing. L'obiettivo è
  ridurre i requisiti hardware e il costo per token utile, con risultati misurati
  end-to-end, revisionati e sviluppati apertamente.
- **Più modelli aperti.** L'algoritmo di tiering è indipendente dal modello:
  qualsiasi MoE con expert instradati può essere organizzato allo stesso modo.
  Dieci famiglie di modelli linguistici funzionano già (GLM-5.2/5.3, GLM-5.3-Flash
  con la visione, Inkling, Kimi K3, DeepSeek V4 Flash, DeepSeek V4.1 Flash,
  MiMo-V2.6, Qwen3.8-Flash-Next, Qwen3.6, OLMoE), più Qwen-Image-2.1 per le
  immagini; altre famiglie open-weight, **MiniMax** tra le candidate, si guadagnano
  un engine come queste: quando qualcuno le misura end-to-end.

## Sostenere il progetto

colibrì è nato come progetto di una sola persona su un portatile con 12 core
e 25 GB di RAM; oggi i suoi numeri arrivano da una comunità di macchine reali.
Se ti è utile:

- ⭐ metti una stella al repository e condividilo;
- 🐛 apri issue con i numeri di benchmark del tuo hardware — i datapoint
  fanno avanzare questo progetto più di qualsiasi altra cosa;
- 💬 entra nella [comunità Discord](https://discord.gg/RXV83nSZdk) per discutere
  esperimenti, risultati hardware e direzioni di ricerca;
- 💬 contattaci via GitHub issues per sponsorizzare lo sviluppo o donare hardware.

## Struttura del repository

```
Makefile                  punto d'ingresso root per build/check
c/
├── colibri.c             motore GLM-5.2  (make glm)
├── inkling.c             motore Inkling  (make inkling)
├── kimi_k3.c             motore Kimi K3  (make kimi_k3)
├── glm53.c               motore GLM-5.3-Flash  (make glm53)
├── deepseek_v4.c         motore DeepSeek V4 Flash  (make deepseek-v4)
├── deepseek_v41.c        motore DeepSeek V4.1 Flash  (make deepseek_v41)
├── mimo.c                motore MiMo-V2.6 Flash e Pro  (make mimo)
├── qwen38.c              motore Qwen3.8-Flash-Next  (make qwen38)
├── qwen36.c              motore Qwen3.6, Qwen3-Coder, Qwen3.8-27B  (make qwen36)
├── olmoe.c               motore OLMoE  (make olmoe)
├── qwenimage.c           motore Qwen-Image-2.1  (make qwenimage)
│
├── st.h                  indice safetensors e letture a intervalli
├── quant.h               decoder canonici dei container
├── expert_ffn.h          kernel FFN degli expert instradati condiviso dai motori MoE (int4 planare, runner di layer)
├── tok.h, json.h         tokenizer e parser JSON
├── compat.h              shim per Windows/macOS (nomi POSIX, in un solo posto)
├── expert_store.h        cache degli expert in streaming
├── route_trace.h         telemetria di routing e .coli_usage, indipendente dal motore
├── kv_prefix.h           riuso del prefisso KV tra i turni
│
├── backend_cuda.*        livello CUDA opzionale   (CUDA=1)
├── backend_metal.*       livello Metal opzionale  (METAL=1)
├── backend_vulkan.*      livello Vulkan opzionale (VK=1)
│
├── Makefile              build e check locali
├── coli                  CLI utente
├── openai_server.py      gateway HTTP compatibile OpenAI
├── resource_plan.py      planner RAM/VRAM dietro `coli plan` e `coli doctor`
├── tools/                conversione offline, fixture e benchmark
├── scripts/              helper per conversioni lunghe
└── tests/                test C e Python senza dipendenze
web/                      UI browser (puro client API OpenAI)
desktop/                  shell desktop Tauri v2 che racchiude la web UI
docker/                   immagini container
docs/                     documentazione di riferimento, esperimenti, media
```

**Un `.c` per famiglia di modelli, sopra header singoli condivisi.** Un motore
possiede la sua architettura e nient'altro; tutto ciò che serve a due motori (il
lettore safetensors, i decoder dei container, il tokenizer, la cache degli expert)
vive in un header che entrambi includono, così una correzione li raggiunge tutti
insieme. Questa regola non è decorativa: i difetti che continuano a ripresentarsi
qui sono quelli in cui un meccanismo è arrivato in un motore e non ha mai raggiunto
i suoi fratelli.

Dalla radice del repository, `make`, `make check` e `make clean` delegano al
Makefile del motore.

## Perché "colibrì"

Il colibrì pesa pochi grammi, sta sospeso nel vuoto e visita un migliaio di
fiori al giorno. Questo motore tiene in vita un gigante da 744 miliardi di
parametri con le razioni di un colibrì: 25 GB di RAM, dodici core CPU e
tanta pazienza col disco.

Il nome è rimasto in italiano perché questa è la lingua in cui è stato scritto
il primo prototipo — i commenti nel codice lo testimoniano ancora.

## Ringraziamenti

colibrì è un motore; le menti che fa girare sono un dono. Grazie ai team che
rilasciano apertamente pesi di classe frontiera, **Z.ai** (GLM), **Moonshot AI**
(Kimi), **Alibaba Qwen**, **MiniMax** e **Allen AI** (OLMoE), e a ogni
contributore che ha fatto benchmark, bisect, replicato un'esecuzione dell'atlante o
mandato una patch. Questo progetto è la prova di ciò che i pesi aperti rendono
possibile.

Gli esperimenti del progetto su piazzamento, compressione e routing degli expert si
basano anche su idee ed evidenze dei seguenti lavori aperti di ricerca e di sistemi:

- [REAP](https://github.com/CerebrasResearch/reap) e
  [EASY-EP](https://github.com/RUCAIBox/EASYEP) per l'importanza degli expert
  consapevole dell'output e specifica del dominio.
- [SERE](https://github.com/JL-Cheng/SERE) per il re-routing degli expert basato
  sulla similarità, e [ReMoE](https://github.com/BUAA-OSCAR/ReMoE) per il
  fine-tuning del router consapevole della località della cache.
- [MC-SMoE](https://github.com/UNITES-Lab/MC-SMoE) per il merging e la compressione
  degli expert guidati dal routing.
- [MoBE](https://github.com/inclusionAI/MoBE) e
  [D²-MoE](https://github.com/lliai/D2MoE) per le basi di expert condivise e i
  delta di expert a basso rango.
- [HybriMoE](https://github.com/PKU-SEC-Lab/HybriMoE) per lo scheduling ibrido
  CPU/GPU degli expert, [ScMoE](https://arxiv.org/abs/2404.05019) per la
  sovrapposizione tra comunicazione degli expert e calcolo, e
  [OD-MoE](https://arxiv.org/abs/2512.03927) per il caricamento distribuito degli
  expert su richiesta.
- [vLLM](https://github.com/vllm-project/vllm),
  [llama.cpp](https://github.com/ggml-org/llama.cpp) e
  [kTransformers](https://github.com/kvcache-ai/ktransformers) per i sistemi di
  inferenza aperti e il lavoro sull'offload degli expert che rendono riproducibili i
  confronti.

Il motore poggia anche su lavoro di ingegneria concreto, non solo su idee. Ognuno
di questi è usato o reimplementato oggi nel repository:

- [safetensors](https://github.com/huggingface/safetensors): il container che ogni
  motore legge (`c/st.h`), compresi i suoi dtype fp8 e I64.
- [tiktoken](https://github.com/openai/tiktoken): `c/tok.h` reimplementa
  esattamente il suo `byte_pair_encode`, unendo la coppia adiacente la cui
  concatenazione ha l'id di vocabolario più basso, così un vocabolario derivato da
  tiktoken non ha bisogno di una lista di merge.
- [llama.cpp](https://github.com/ggml-org/llama.cpp): il sottoinsieme della
  grammatica GBNF in `c/grammar.h` segue la sua sintassi e il suo PDA a insieme di
  stack, e il percorso Metal prende in prestito il suo trucco di residenza
  `newBufferWithBytesNoCopy`.
- [vLLM](https://github.com/vllm-project/vllm): il riferimento per la semantica
  dell'output che il motore riproduce posizione per posizione (per esempio dove cade
  la norma finale rispetto alla LM head).
- [transformers](https://github.com/huggingface/transformers): l'oracle; la CI
  riproduce token per token contro di esso un modello inizializzato a caso.
- [DietGPU](https://github.com/facebookresearch/dietgpu): il codec ANS su GPU dietro
  il livello sperimentale di expert compressi (`COLI_ANS`).
- [rocWMMA](https://github.com/ROCm/rocWMMA): il backend HIP mappa su di esso l'API
  di frammenti e mma_sync `nvcuda::wmma` di CUDA (`c/backend_gpu_compat.h`), ed è
  ciò che permette a un unico sorgente .cu di compilare per entrambi i produttori.

## Licenza

Apache 2.0, Copyright 2026 Vincenzo Fornaro. Vedi [LICENSE](LICENSE) e [NOTICE](NOTICE). I pesi di GLM-5.2 sono rilasciati da Z.ai sotto licenza MIT.
