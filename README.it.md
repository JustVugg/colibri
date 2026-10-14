<p align="center">
  <img src="assets/colibri-logo.svg" width="560" alt="colibrì: motore piccolo, modello immenso">
</p>

<p align="center">
  <a href="https://justvugg.github.io/colibri"><img src="https://img.shields.io/badge/website-justvugg.github.io%2Fcolibri-1f6feb" alt="Sito web"></a>
  <a href="https://github.com/JustVugg/colibri/releases"><img src="https://img.shields.io/github/v/release/JustVugg/colibri?color=2ea043" alt="Ultima release"></a>
</p>

<p align="center">
  <a href="https://justvugg.github.io/colibri"><b>Sito web</b></a> ·
  <a href="https://discord.gg/RXV83nSZdk"><b>Discord</b></a> ·
  <a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.zh-TW.md">繁體中文</a> · Italiano · <a href="README.ja.md">日本語</a> · <a href="README.id.md">Bahasa Indonesia</a>
</p>

**Motore piccolo, modello immenso.** colibri fa girare modelli aperti molto
grandi sulla macchina che hai già. Un modello mixture-of-experts da centinaia di
miliardi di parametri usa solo una piccola parte di sé per ogni token: colibri
tiene quella parte in RAM e legge il resto, gli expert, dal disco quando il
modello li chiede. C puro, un file per famiglia di modelli, nessuna GPU
necessaria.

Oggi girano tredici motori. Dieci sono per modelli linguistici: **GLM-5.2/5.3**,
**GLM-5.3-Flash**, **Inkling**, **Kimi K3**, **DeepSeek V4 Flash**,
**DeepSeek V4.1 Flash**, **MiMo-V2.6 Flash** (e Pro), **Qwen3.8-Flash-Next**,
**Qwen3.6** (che esegue anche Qwen3-Coder e il denso Qwen3.8-27B) e
**OLMoE**. Uno disegna immagini: **Qwen-Image-2.1**. Due prendono decisioni:
**Laya** e **GLiNER2.5-Decide**, con un terzo modello di decisione, **Clef**, sul
motore Qwen3.6. [Quale scegliere per la mia macchina](#which-model-for-my-machine)

```
$ ./coli chat
  colibri v2.0.0 · GLM-5.2 · 744B MoE · int4 · streaming CPU
  ✓ ready in 32s · resident 9.9 GB
  › ciao!
  ◆ Ciao! Come posso aiutarti oggi?
```

<a id="get-started"></a>
<a id="the-one-step-way"></a>

## Inizia in un solo passo

Ti serve un computer con **8 GB di RAM** come minimo assoluto (meglio 16 GB o
più), **22 GB liberi sul disco** per il modello più piccolo e una connessione a
internet. La scheda grafica è facoltativa.

**Windows**

1. In questa pagina fai clic su **Code**, poi su **Download ZIP**, ed estrai
   l'archivio.
2. Fai doppio clic su **`START-HERE.bat`** nella cartella estratta. Se manca
   Python, si offre di installarlo per te.

**Linux** (Ubuntu e Debian; le altre distribuzioni hanno gli stessi pacchetti
con i loro nomi)

```bash
sudo apt install git python3 build-essential
git clone https://github.com/JustVugg/colibri
cd colibri
./start-here.sh
```

**macOS** (con [Homebrew](https://brew.sh))

```bash
xcode-select --install
brew install libomp git python
git clone https://github.com/JustVugg/colibri
cd colibri
./start-here.sh
```

Rispondi a una sola domanda, quale modello, e Invio accetta il consiglio. Poi
l'installazione:

1. **esamina la tua macchina**: RAM, disco libero, CPU e GPU;
2. **consiglia un modello che ci sta**: la parte del modello che resta sempre in
   RAM, più una cache minima di expert, deve entrare nella tua RAM, e il
   download sul tuo disco;
3. **prepara il motore**: lo compila per la tua macchina se c'è un compilatore,
   altrimenti scarica quello già compilato, che gira sulla CPU e, su Linux e
   Windows, anche su una GPU Vulkan. Lo compila per
   la tua GPU quando conviene: CUDA per una scheda NVIDIA su Linux se è
   installato il CUDA toolkit, altrimenti Vulkan. Su una GPU dedicata lo fa
   sempre; su una GPU integrata, che condivide la RAM della CPU, solo per i
   modelli che lì sono risultati più veloci nelle misure (Qwen3.6, Qwen3-Coder
   e Qwen3.8-Flash-Next). Se manca un pacchetto, stampa il comando esatto per
   installarlo e prosegue con la CPU; rilancia poi l'installazione e il motore
   viene ricompilato per la GPU;
4. **scarica il modello** mostrando l'avanzamento e con la ripresa: interrompilo
   quando vuoi, rilancialo e riparte da dove si era fermato;
5. **avvia colibri e apre la dashboard** nel browser, e stampa gli indirizzi
   che le altre app possono usare:

```
Starting colibri
  Browser:             http://127.0.0.1:8000/
  OpenAI base URL:     http://127.0.0.1:8000/v1
  Anthropic base URL:  http://127.0.0.1:8000
  stop: press Ctrl+C here (or close this window)
```

**La volta successiva**, rilancia `START-HERE.bat` o `./start-here.sh`: colibri
parte subito, senza download e senza compilazione. `c/coli status` mostra cosa è
installato e se è in esecuzione, `c/coli stop` lo ferma (`c\coli.cmd status` e
`c\coli.cmd stop` su Windows).

Le opzioni vanno dopo `./start-here.sh` o `START-HERE.bat`:

| Opzione | Cosa fa |
|---|---|
| `--list` | ogni modello confrontato con questa macchina, e il motivo per cui uno non ci sta |
| `--model ID` | installa quel modello (gli id sono nelle [tabelle qui sotto](#which-model-for-my-machine)) |
| `--yes` | nessuna domanda: accetta il consiglio |
| `--dir DIR` | mette i modelli su un altro disco (predefinito `~/colibri-models`) |
| `--backend vulkan`, `cuda` o `cpu` | scegli tu la build del motore; `--no-gpu` equivale a `--backend cpu` |
| `--model-dir DIR` | usa un modello che hai già scaricato |
| `--reconfigure` | per scegliere un altro modello |

Cosa fa ogni passo, nel dettaglio: [docs/quickstart.md](docs/quickstart.md#the-one-step-way).

### Se qualcosa va storto

| Cosa vedi | Cosa fare |
|---|---|
| si è fermato durante il download | rilancia lo stesso comando: riprende dai byte già presenti sul disco |
| `to use the GPU through ..., first run: <command>` | esegui quel comando, poi di nuovo l'installazione: ricompila per la GPU e non riscarica nulla |
| `the ... build failed`, per esempio `Unsupported gpu architecture` quando il CUDA toolkit installato non supporta più la scheda | l'installazione controlla prima il toolkit rispetto alla scheda e sceglie Vulkan da sé, spiegando perché; se una compilazione fallisce ancora, offre la prossima (Vulkan, poi la CPU). `./start-here.sh --backend vulkan` forza Vulkan; `--no-gpu` resta sulla CPU |
| `needs N GB free for the download` | `--dir` con una cartella su un disco più grande |
| su WSL, la cartella del modello è sotto `/mnt/c` | tienila sul disco Linux (il predefinito, `~/colibri-models`): `/mnt/c` è molte volte più lento |
| hai aggiornato il checkout (`git pull`) | rilancia `./start-here.sh`: ricompila il motore se i sorgenti sono cambiati, poi lo avvia |
| qualsiasi altra cosa | `c/coli logs -n 50` mostra il log del server avviato in background (quello avviato in primo piano scrive nel suo terminale) e `c/coli logs --install` quello dell'installazione; apri una [issue](https://github.com/JustVugg/colibri/issues) con le ultime righe stampate dall'installazione |

### Lascia che sia il tuo assistente AI a installarlo

Se usi un assistente AI per programmare, può fare tutto questo al posto tuo.
Chiedigli:

> Installa colibri su questa macchina seguendo docs/AI_SETUP.md da https://github.com/JustVugg/colibri

[docs/AI_SETUP.md](docs/AI_SETUP.md) dà all'assistente ogni passo sotto forma di
comando con un risultato leggibile da una macchina, e gli dice di chiederti
conferma prima di scaricare un modello o di installare un pacchetto di sistema.
Gli assistenti che parlano il Model Context Protocol possono invece usare il
server MCP di colibri: `coli mcp` offre strumenti per rilevare l'hardware,
consigliare un modello, installarlo, avviarlo, fermarlo e controllarlo
([docs/MCP_SERVER.md](docs/MCP_SERVER.md)).

### Oppure a mano

Per scegliere tu ogni passo (una release già compilata o una build dai sorgenti,
qualsiasi modello dalle tabelle qui sotto, poi `coli chat`, `coli web` o
`coli serve`), vedi [Installazione a mano](#install-by-hand), oppure la
[guida Quick Start](docs/quickstart.md), passo passo per ogni piattaforma.

## Cos'è colibri, e perché

Un modello mixture-of-experts è enorme sul disco e piccolo per token. GLM-5.2 ha
744B parametri, ne usa circa 40B per ogni token, e solo circa 11 GB di questi
cambiano da un token all'altro: gli expert instradati.

<p align="center">
  <img src="docs/media/sparse.png" width="880" alt="solo circa il 5.4% dei parametri è attivo per ogni token">
</p>

Quindi il modello non deve stare in memoria veloce; deve essere **piazzato**. La
parte densa (attenzione, expert condivisi, embedding) resta in RAM. Gli expert
instradati restano sul disco e vengono letti quando il router li chiede,
attraverso una cache che impara quali expert usa il tuo lavoro. Una GPU, quando
c'è, tiene gli expert più caldi e i layer densi. Dove si trova un peso cambia la
velocità con cui arriva la risposta, non quali pesi o quali decisioni del router
la producono.

Perché: per far girare modelli di queste dimensioni sull'hardware che le persone
già possiedono, per guardarli lavorare (la dashboard mostra ogni expert nel
momento in cui si attiva) e per tenere il motore abbastanza piccolo da permettere
a chiunque di misurarlo e renderlo più veloce. colibri è anche una piattaforma di
ricerca aperta: un'ottimizzazione si guadagna il suo posto con una misura
end-to-end riproducibile, e la policy predefinita non cambia mai silenziosamente
la precisione del modello né la semantica del router. Meno memoria veloce può
costare velocità; non deve ridefinire il modello di nascosto. I dettagli sono in
[Come funziona](#how-it-works).

<a id="which-model-for-my-machine"></a>

## Quale modello per la mia macchina

L'installazione consiglia il modello più capace che gira dalla RAM sulla tua
macchina, e subito sotto elenca quelli più grandi, che vengono letti dal disco in
streaming. `./start-here.sh --list` li mostra tutti confrontati con la tua
macchina. Le tabelle seguono il catalogo dell'installazione stessa
([`c/setup_catalog.py`](c/setup_catalog.py)); i download sono le dimensioni che
Hugging Face indica per ciascun repository.

- **RAM** sono due numeri: sotto il primo il modello non parte, dal secondo in
  su gira come nelle misure.
- **GPU** è ciò che l'installazione può compilare per quel motore ([GPU](#gpus)).
  *Anche integrata* significa che usa anche una GPU integrata, dove questi
  motori sono risultati più veloci nelle misure; gli altri usano solo una GPU
  dedicata.
- **Misurato** è ciò che è stato cronometrato sulla macchina indicata: per i
  modelli di chat, la velocità di decode. Le lettere sono le macchine elencate
  sotto le tabelle. Una cella vuota significa che nessuno l'ha ancora misurato.

**Modelli piccoli, che girano dalla RAM**

| Modello | `--model` | Download | RAM | GPU | Misurato |
|---|---|---|---|---|---|
| **Qwen3.6-35B-A3B**: chat con thinking e tool | `qwen36-35b` | 23 GB | 10 / 20 GB | CUDA, Vulkan (anche integrata) | 6.0 tok/s sulla CPU, 9.9 con Vulkan sulla GPU integrata (A); 30.0 con CUDA (C) |
| Qwen3-Coder-30B-A3B: codice e chiamate ai tool, senza thinking | `qwen3-coder-30b` | 19 GB | 8 / 18 GB | CUDA, Vulkan (anche integrata) | 8.5-9.6 tok/s con tutti gli expert in RAM, 5.1 con 32 per layer (A, CPU) |
| **Qwen-Image-2.1**: da testo a immagine, licenza non commerciale | `qwen-image-2.1` | 33 GB | 12 / 18 GB | Vulkan | un'immagine 768x512 in 2 min 40 s (8 core Zen 4, CPU) |

**Modelli grandi, i cui expert vengono letti dal disco in streaming** (la
velocità la decide il disco: un NVMe veloce è ciò che aiuta di più)

| Modello | `--model` | Download | RAM | GPU | Misurato |
|---|---|---|---|---|---|
| DeepSeek V4 Flash REAP 150B: 132 dei 256 expert | `deepseek-v4-flash-reap` | 85 GB | 16 / 32 GB | CUDA, Vulkan | |
| **DeepSeek V4 Flash** (284B): tool | `deepseek-v4-flash` | 167 GB | 16 / 32 GB | CUDA, Vulkan | 0.93 tok/s con 32 GB (Ryzen 7 5800X), 1.24 con 63 GB (Ryzen 9 5950X), solo CPU; 1.5-1.6 con CUDA (RTX 5080, 32 GB, due NVMe) |
| **MiMo-V2.6 Flash** (309B): visione e tool | `mimo-v2.6-flash` | 172 GB | 32 / 52 GB | Vulkan | 2.34-3.37 tok/s (A, CPU) |
| **Qwen3.8-Flash-Next** (125B + 51B n-gram): visione e tool | `qwen38-flash-next` | 186 GB | 24 / 32 GB | CUDA, Vulkan (anche integrata) | 1.91-2.56 tok/s con 32-96 expert per layer; 3.99 con gli expert int4 opzionali (A, CPU) |
| **GLM-5.2** (744B): il modello di riferimento, con la testa MTP | `glm-5.2` | 429 GB | 16 / 24 GB | CUDA, Vulkan | 0.05-0.1 tok/s a freddo su un portatile da 25 GB; 1.83 su un Ryzen AI Max+ 395 da 128 GB; 9.0-9.2 su 6x RTX 5090 |
| GLM-5.3 (744B): lo stesso motore, senza testa MTP | `glm-5.3` | 419 GB | 16 / 24 GB | CUDA, Vulkan | |
| **Inkling** (975B): expert int4, pesi densi bf16 | `inkling` | 514 GB | 120 / 128 GB così come scaricato; 25 GB dopo una [conversione della parte densa](docs/inkling.md) | CUDA, Vulkan | 0.25 tok/s (Ryzen 9 7900, 187 GB, RTX A6000) |
| MiMo-V2.6 Pro (1.02T): visione e tool | `mimo-v2.6-pro` | 564 GB | 54 / 64 GB | Vulkan | 0.66-0.79 tok/s (A, CPU) |
| **Kimi K3** (2.8T): il più grande | `kimi-k3` | 1.56 TB | 32 / 64 GB | CUDA, Vulkan | circa 9.4 s per token, expert letti a 6.3 GB/s |

<a id="other-supported-models"></a>

**A mano: un passo di conversione o di preparazione dopo il download**

| Modello | Download, poi su disco | RAM | GPU | Misurato |
|---|---|---|---|---|
| **OLMoE** (7B): piccolo, per prendere confidenza con gli strumenti | 14 GB, 7 GB dopo la conversione in int8 | 8 GB | Vulkan | 22-23 tok/s (A, CPU) |
| Qwen3.8-27B (denso): testo e immagini | 56 GB, 51 GB dopo la conversione | 20 GB in int4, 30 GB in int8 | Vulkan | 3.45 tok/s in int4, 2.1 in int8 (server CPU a 16 thread) |
| **GLM-5.3-Flash** (321B): visione e tool | 328 GB, convertiti shard per shard in 195 GB | 25 GB | CUDA, Vulkan | circa 20 s per token a caldo, 44 s a freddo (6 core, 25 GB, disco normale) |
| **DeepSeek V4.1 Flash** (552B): visione e tool, nessuna conversione ma una preparazione una tantum | 510 GB | circa 18 GB più la cache degli expert (24.8 GB di picco con 8 per layer) | Vulkan | 0.21-0.24 tok/s (server CPU a 16 thread che tiene il 68% degli expert) |

**Modelli di decisione** (rispondono alle domande di [System One](#system-one-a-decision-with-a-probability), non chattano)

| Modello | Download, poi su disco | RAM | GPU | Misurato |
|---|---|---|---|---|
| **Laya** (Convai Innovations), inglese | 0.85 GB | 1.7 GB | CPU | 219 ms per una domanda, 882 ms per tre (B) |
| **GLiNER2.5-Decide** (fastino), inglese | 1.95 GB | 1.9 GB | CPU | 294 ms per una domanda, 897 ms per tre (B, sotto carico) |
| **Clef** (Cloudflare): Qwen3.8-27B con una testa di decisione, sa anche chattare | 55 GB, 52 GB dopo la conversione | da 19 GB in int4 a 55 GB in f16 | CPU | 20.4 s per richiesta in int8 (A) |

Le macchine:
**A** un desktop Ryzen 7 PRO 8700GE (8 core, 61-64 GB DDR5, NVMe, Radeon 780M
integrata);
**B** un portatile i7-1355U;
**C** una RTX 3070 da 8 GB in una macchina Threadripper 3945WX, con i layer densi
e i layer DeltaNet sulla scheda (container int4 per-row).
Ogni numero viene dalla pagina del suo modello in [docs/](docs/) o dalle
[tabelle dei benchmark](docs/benchmarks.md), con le impostazioni esatte.

Ogni famiglia ha la sua pagina: [qwen36.md](docs/qwen36.md) (Qwen3.6, Qwen3-Coder,
Qwen3.8-27B), [qwen38.md](docs/qwen38.md), [deepseek-v4.md](docs/deepseek-v4.md),
[deepseek-v41.md](docs/deepseek-v41.md), [mimo.md](docs/mimo.md),
[glm53-flash.md](docs/glm53-flash.md), [inkling.md](docs/inkling.md),
[kimi_k3.md](docs/kimi_k3.md), [qwen-image.md](docs/qwen-image.md),
[laya.md](docs/laya.md), [gliner_decide.md](docs/gliner_decide.md),
[clef.md](docs/clef.md), e GLM-5.2 nella [Quick Start](docs/quickstart.md#3-get-the-model).
I checkpoint con la stessa architettura di uno supportato, come KAT-Coder v2.5
sul motore Qwen3.6, girano senza modifiche.

<a id="gpus"></a>

## GPU

### Nessuna GPU necessaria

Ogni motore gira sulla CPU senza installare nient'altro. Una GPU è un posto più
veloce dove tenere i pesi, non un requisito: per i modelli grandi la velocità la
decide il disco, per quelli piccoli la RAM.

### Vulkan: qualsiasi GPU

Ogni motore MoE può usare qualsiasi GPU con un driver Vulkan 1.2 (AMD, Intel,
NVIDIA, integrata o dedicata) in due modi:

- **il livello degli expert**: una cache di expert instradati nella memoria
  della GPU, riempita all'avvio con gli expert usati dalle tue conversazioni
  passate e che si adatta mentre chatti. La GPU calcola gli expert che contiene
  mentre la CPU calcola il resto;
- **la catena densa**: un intero layer registrato come un unico invio alla GPU,
  con lo stato corrente del modello tenuto sulla GPU da un layer al successivo.

Misurato sulla Radeon 780M integrata della macchina A, con i file del modello
tolti dalla page cache prima di ogni esecuzione, 100 token in decode
([vulkan.md](docs/vulkan.md#the-chain-on-a-radeon-780m)):

| | CPU | Vulkan, livello degli expert | Vulkan, livello e catena densa |
|---|---|---|---|
| Qwen3.6-35B-A3B, decode | 6.0 tok/s | 8.0 tok/s | **9.9 tok/s** |
| Qwen3.6-35B-A3B, un prompt di 512 token | 35.7 s | 12.2 s | **9.5 s** |
| Qwen3.8-Flash-Next (expert int4), decode | 3.5 tok/s | **3.8 tok/s** | 3.2 tok/s |
| Qwen3.8-Flash-Next, un prompt di 512 token | 43.6 s | 38.7 s | **30.1 s** |
| OLMoE, decode (a caldo) | **23.1 tok/s** | 12.8 tok/s | 17.3 tok/s |

Una GPU integrata condivide la RAM della CPU. Quello che fa risparmiare è il
lavoro e le letture dal disco degli expert che contiene: per questo conviene su
un modello come Qwen3.6, mentre un modello piccolo i cui expert stanno già in
RAM, come OLMoE, può perderci. Ecco perché l'installazione attiva Vulkan su una
GPU integrata solo per Qwen3.6, Qwen3-Coder e Qwen3.8-Flash-Next, e perché ogni
motore decide da sé se usare lì la catena densa (Qwen3.6 sì, Qwen3.8 no).
`--backend vulkan` chiede Vulkan comunque.

**Accendere o spegnere la GPU.** `coli setup --backend vulkan` usa la GPU per
qualunque modello, e `coli setup --backend cpu` (o `--no-gpu`) tiene tutto sulla
CPU. Un motore compilato con Vulkan usa la GPU solo con `COLI_VULKAN=1`
nell'ambiente di `coli chat`, `serve` o `web` (l'installazione lo imposta quando ha
scelto Vulkan); senza, il motore gira sulla CPU. Con la GPU accesa, `COLI_VK_CHAIN=0`
tiene il tier degli expert e fa girare gli strati densi sulla CPU. Su una GPU
integrata prova entrambe le strade: su un portatile con una Intel Iris Xe (Core
i7-1355U) Qwen3.6 ha decodificato a 2,1 tok/s sulla CPU, da 1,7 a 1,9 con Vulkan
e 2,1 con la catena densa spenta.

Su una GPU dedicata l'installazione compila Vulkan per ogni motore (prima CUDA,
dove il motore ce l'ha e il toolkit è installato), con i layer densi sulla
scheda. È il caso per cui il progetto è pensato. **Non abbiamo ancora misurato
una GPU dedicata noi stessi.** Il primo numero viene da un utente: Qwen3.6 da
17 a 19 tok/s su una Tesla V100 16 GB, con il livello degli expert e la catena
densa ([#1852](https://github.com/JustVugg/colibri/issues/1852)). (Prima di
questi, il precedente percorso Vulkan di GLM-5.2 faceva 1.7-1.8 tok/s in decode
su una RX 9070 dedicata.) I numeri della tua scheda sono benvenuti.

**Le schede senza Resizable BAR** ora funzionano. Una scheda così (tutte le
Turing, le Ampere con il firmware di lancio, le AMD più vecchie con l'opzione
disattivata) lascia scrivere direttamente alla CPU solo circa 256 MB della sua
memoria; ora colibri copia i pesi passando per un buffer di staging, da solo. Il
percorso è testato forzandolo ed emulando la finestra piccola su tre
dispositivi, e non costa nulla di misurabile sulla 780M; non è stato misurato su
una scheda senza Resizable BAR
([vulkan.md](docs/vulkan.md#memory-placement-without-resizable-bar)).

La CI verifica il percorso Vulkan di ogni motore contro i token della CPU su un
driver software. La GPU somma i numeri in un ordine diverso e tiene alcune
attivazioni in f32 dove la CPU le arrotonda, quindi una risposta lunga può
discostarsi da quella della CPU di una parola
([vulkan.md](docs/vulkan.md#the-other-engines)).

### CUDA: schede NVIDIA

L'installazione compila CUDA su Linux quando è installato il CUDA toolkit, per i
motori che hanno un percorso CUDA: GLM-5.2/5.3, GLM-5.3-Flash, Inkling, Kimi K3,
DeepSeek V4 Flash, Qwen3.8-Flash-Next, e Qwen3.6 con Qwen3-Coder. Su Windows il
motore CUDA è una DLL separata ([windows.md](docs/windows.md)), e ogni release la
porta già compilata: `colibri-<versione>-windows-x86_64-cuda.zip` contiene
`coli_cuda.dll` (schede con compute capability 8.0 o superiore) e i motori
colibri, qwen36 e kimi_k3 che la caricano. Estrailo sopra l'archivio principale e
l'installazione sceglie CUDA.

- **Il livello degli expert in VRAM** tiene sulla scheda gli expert più caldi,
  scelti in base al routing misurato; i miss vengono calcolati sulla CPU nello
  stesso momento. Qwen3.6 su due schede da 8 GB (RTX 3070 e Quadro RTX 4000) ha
  fatto 11.3 tok/s in decode con una cronologia calda
  ([qwen36-cuda-tier.md](docs/qwen36-cuda-tier.md)); GLM-5.2 su sei RTX 5090
  con tutti gli expert residenti, 9.0-9.2 tok/s
  ([benchmarks.md](docs/benchmarks.md)); DeepSeek V4 Flash su una RTX 5080,
  1.5-1.6 tok/s e un prompt di 3,324 token in 90 s
  ([deepseek-v4.md](docs/deepseek-v4.md)).
- **Novità per Qwen3.6: i layer DeltaNet sulla scheda** (`Q36_DN_GPU=1`,
  opt-in). Ognuno dei 30 layer DeltaNet di Qwen3.6 copiava i suoi dati tra la
  scheda e la CPU quattro volte per token; ora un token di decode esegue l'intero
  layer sulla scheda, con il suo stato ricorrente tenuto in VRAM. Su una RTX
  3070 con i layer densi in VRAM: da 25.4 a 30.0 tok/s
  ([qwen36-cuda-tier.md](docs/qwen36-cuda-tier.md#the-deltanet-layer-on-the-card-q36_dn_gpu1)).
- **Schede più vecchie.** Se il CUDA toolkit non compila più per la tua scheda
  (CUDA 13 e una V100, in [#1852](https://github.com/JustVugg/colibri/issues/1852)),
  l'installazione lo vede prima di compilare e usa la scheda tramite Vulkan,
  spiegando perché; un CUDA toolkit 12.x riporta il percorso CUDA. Il livello
  CUDA di DeepSeek V4 si compila anche per Pascal e Turing
  (`CUDA_ARCH=portable-pre-ampere NO_TC=1`).

Tutti i dettagli: [docs/cuda.md](docs/cuda.md).

### Apple Silicon

Un backend Metal esegue i calcoli degli expert sulla GPU a memoria unificata per
diversi motori ([docs/metal.md](docs/metal.md)). Nell'archivio macOS della release
`colibri`, `inkling` e `kimi_k3` sono compilati con Metal: `COLI_METAL=1`
(`K3_METAL=1` per Kimi K3) lo accende, e senza girano sulla CPU. Dai sorgenti
compila con `METAL=1`; l'installazione in un passo compila per la CPU.

<a id="system-one-mode-ask-a-closed-question"></a>
<a id="system-one-a-decision-with-a-probability"></a>

## System One: una decisione con una probabilità

Gran parte di ciò che si chiede a un modello è una scelta, non un paragrafo:
quale coda, quale verdetto, sì o no. `POST /v1/systemone` riceve uno stato (testo
o JSON) e domande tipizzate, e risponde a ciascuna con la probabilità di ogni
opzione ammessa e una confidenza. Non si genera nulla, quindi nessuna risposta
può uscire dalla tua lista, e "il modello non è sicuro" è un numero su cui puoi
mettere una soglia.

```bash
curl -s http://127.0.0.1:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "340 lines, 8 files, no tests. CI is green but nothing covers that path.",
  "questions": {
    "review": {"type": "choice", "instructions": "What should the reviewer do?",
               "criteria": {"merge": null, "request changes": null, "close": null}},
    "risky":  {"type": "noul", "instructions": "Is this change risky?"}}}'
```

Un `choice` torna con l'etichetta scelta, una probabilità per ogni etichetta e
una `confidence` da 0 (piatta) a 1 (certa); un `noul` con la probabilità del sì;
uno `score` con il livello atteso.

Chi risponde:

- **Qualsiasi modello di chat che colibri fa girare**, dando un punteggio: legge
  la probabilità di ogni opzione invece di scrivere una risposta. Più domande
  sullo stesso documento leggono il documento una volta sola: su Qwen3.6,
  quattro domande su un documento sono tornate 5.7x più velocemente che
  generando le stesse risposte sulla stessa macchina CPU.
- **Tre modelli di decisione**, in modo nativo, in un solo forward pass con la
  calibrazione stimata dai loro autori: [Laya](docs/laya.md) (Convai
  Innovations), [GLiNER2.5-Decide](docs/gliner_decide.md) (fastino) e
  [Clef](docs/clef.md) (Cloudflare; sa anche chattare). Dimensioni e velocità
  sono nella [tabella dei modelli di decisione](#which-model-for-my-machine).

**Passare da Jev.** La richiesta e la risposta sono quelle dell'API Jev di
TypeSafe, quindi un client Jev passa a colibri cambiando il base URL e
nient'altro: `TYPESAFE_BASE_URL=http://127.0.0.1:8000` (la chiave che già invia
viene accettata da un server avviato senza `COLI_API_KEY`). I due SDK ufficiali,
senza modifiche, sono testati contro `coli serve`.

La stessa modalità c'è nel terminale (`/decide merge | request changes | close`
in `coli chat`) e nella pagina System One della dashboard. Richiesta e risposta
complete, le regole di scoring e dove non aiuta:
[docs/systemone.md](docs/systemone.md).

## La dashboard

La apre `coli web`, e la apre anche l'installazione in un passo: la chat, la
pagina System One, il Brain e la pagina Profiling, in tema chiaro o scuro.

<p align="center">
  <img src="docs/media/colibri-dashboard.png" width="900" alt="la dashboard web di colibri: chat, metriche live, pannello hardware, livelli degli expert">
</p>
<p align="center"><em>Qwen3.6 che risponde su una macchina con sola CPU, con gli expert letti dal disco in streaming.</em></p>

<p align="center">
  <img src="docs/media/colibri-brio.png" width="900" alt="la pagina System One: un documento letto una volta, una probabilità per ogni risposta ammessa, e un'entropia">
</p>
<p align="center"><em><strong>System One</strong>: dai al modello un documento e le sole risposte che può scegliere. Qui:
<strong>request changes al 99.9%</strong>, entropia 0.005, 4 token letti, 0 generati.</em></p>

<p align="center">
  <img src="docs/media/colibri-brain.png" width="900" alt="la pagina Brain: l'atlante misurato degli expert di GLM-5.2 disegnato come una corteccia, dieci regioni in cui entrare">
</p>
<p align="center"><em>Il <strong>Brain</strong>: l'<a href="https://github.com/JustVugg/colibri/issues/175">atlante misurato degli expert</a> di GLM-5.2,
13,260 expert caratterizzati in dieci regioni (Python, SQL, matematica, poesia, legge, cinese...), posizionati in base
all'affinità di routing misurata. <strong>Live routing</strong> mostra il modello in esecuzione: una cella per expert, colorata
secondo il livello di archiviazione, e ogni expert instradato in un turno lampeggia.</em></p>

La pagina **Profiling** mostra dove ogni turno spende il suo tempo, fase per
fase, con gli ultimi 30 turni come tendenza.

## Usarlo da altre app

`coli serve` (che l'installazione avvia per te) è un unico server con diverse
API:

- **compatibile OpenAI**: `/v1/chat/completions`, `/v1/completions` e
  `/v1/models`, con streaming, risposte JSON, sequenze di stop e logprobs;
- **compatibile Anthropic**: `/v1/messages`, quindi Claude Code e gli SDK di
  Anthropic funzionano con esso;
- **tool calling** su ogni motore di chat tranne Inkling e OLMoE, ciascuno nel
  formato nativo del suo modello ([la tabella per motore](docs/api.md#tool-calling-support));
- **immagini in ingresso** su GLM-5.3-Flash, DeepSeek V4.1 Flash, MiMo-V2.6,
  Qwen3.8-Flash-Next e Qwen3.8-27B: un percorso in un messaggio di `coli chat`,
  un allegato in `coli web` o una parte `image_url`;
- **immagini in uscita** con Qwen-Image-2.1 su `POST /v1/images/generations`,
  disegnate anche dentro il terminale da `coli chat`
  ([qwen-image.md](docs/qwen-image.md));
- **decisioni** su `POST /v1/systemone` ([sopra](#system-one-a-decision-with-a-probability));
- **più conversazioni insieme** su ogni motore di testo: `coli serve --kv-slots N`
  ne tiene fino a 16, ognuna con la sua cache, e decodifica insieme i loro token
  successivi ([api.md](docs/api.md#isolated-kv-contexts)).

Le CLI di programmazione e gli editor si collegano come a qualsiasi provider
compatibile OpenAI: base URL `http://127.0.0.1:8000/v1`, l'id del modello che
stampa `coli status`, una chiave qualsiasi purché non vuota
([docs/api.md](docs/api.md#connect-a-coding-cli-or-editor)).

<a id="how-it-works"></a>

## Come funziona

<p align="center">
  <img src="docs/media/token-path.png" width="880" alt="instrada, unisci, piazza, sovrapponi, impara">
</p>

Ogni layer di ogni token percorre gli stessi cinque passi: instrada, unisci,
piazza, sovrapponi, impara. L'obiettivo progettuale è che **il piazzamento decida
soltanto la velocità**: le decisioni del router e la precisione dei pesi sono le
stesse sia che un expert risponda dalla VRAM, dalla RAM o dal disco.

<p align="center">
  <img src="docs/media/tiers.png" width="880" alt="VRAM, RAM e NVMe come tre livelli di residenza degli expert">
</p>

- **Un JIT per i pesi.** Un compilatore JIT non compila mai l'intero programma:
  osserva cosa viene eseguito e compila i percorsi caldi. colibri fa la stessa
  scommessa sui pesi. Il calore di routing misurato decide quali expert si
  guadagnano la VRAM, la RAM o il disco: una cache LRU per layer, più un insieme
  caldo fissato in memoria, appreso dalle tue conversazioni (`.coli_usage`,
  aggiornato a ogni turno). colibri diventa più veloce più lo usi. Funziona
  perché il routing ha una struttura misurabile
  (l'[atlante degli expert](https://github.com/JustVugg/colibri/issues/175)).
- **Mai aspettare il disco due volte.** Le tre matrici di un expert si leggono
  con un solo `pread`; un pool di loader legge gli expert mancanti mentre quelli
  residenti calcolano; un batch di posizioni legge ogni expert una volta sola; un
  thread di lookahead del router può fare il prefetch del layer successivo (il
  routing di GLM-5.2 è prevedibile al 71.6% un layer in anticipo). `DIRECT=1`
  (O_DIRECT) spesso è un grande vantaggio sui dischi NVMe veloci e neutro o
  peggiore sugli altri: misuralo sul tuo ([tuning.md](docs/tuning.md)).
- **Più di un SSD.** `COLI_MODEL_MIRROR=/second/glm52_i4 ./coli chat --model /fast/glm52_i4`
  legge da una copia su un secondo disco. Due NVMe su controller indipendenti
  hanno misurato +37.5% in decode; funziona anche un mirror parziale su un disco
  più piccolo ([multidisk.md](docs/multidisk.md)).
- **Dal portatile al rack.** Su un portatile da 25 GB ogni expert viene letto
  dal disco in streaming, lentamente e correttamente; su un host grande ogni
  expert è residente (`CUDA_EXPERT_GB=auto PIN_GB=all`) e il disco esce dal
  decode. `COLI_NUMA=1` distribuisce i pesi residenti tra i controller di
  memoria di un host multi-socket, e una modalità cluster locale esegue gli
  expert instradati su altre macchine ([cluster.md](docs/cluster.md)).
- **Un modello fedele.** Ogni motore è verificato in CI contro l'implementazione
  di riferimento del suo modello su una fixture minuscola. L'attenzione MLA di
  GLM-5.2 tiene uno stato KV compresso (576 float per token invece di 32,768,
  57x più piccolo) che sopravvive ai riavvii, quindi una conversazione si riapre
  senza un prompt da rileggere.
- **Speculazione che si ripaga.** La testa MTP int8 di GLM-5.2 propone 2.2-2.8
  token per forward quando conviene; la testa MTP di Qwen3.8-Flash-Next,
  accesa di default, aggiunge il 16-20% con lo stesso output, e il lookup sul
  prompt il 6-7% sulle modifiche al codice. Dove il drafting costa più di
  quanto fa risparmiare (DeepSeek V4) resta spento
  ([tuning.md](docs/tuning.md#speculation-and-reproducibility)).

Il motore è un file C per famiglia di modelli (`c/colibri.c` per GLM-5.2) sopra
header condivisi, senza BLAS e senza Python a runtime: Python esegue solo
l'installazione, il launcher, i convertitori e il gateway API.

<a id="what-it-achieves"></a>

## Benchmark

<p align="center">
  <img src="docs/media/ladder.png" width="880" alt="velocità di decode di GLM-5.2 misurata per classe di hardware">
</p>

Stesso motore e stesso container int4: l'hardware cambia solo dove risiedono gli
expert. Decode di GLM-5.2, dalle [tabelle complete](docs/benchmarks.md):

- **6x RTX 5090, tutti gli expert residenti:** 5.8-6.8 tok/s, 9.0-9.2 con
  l'interleave NUMA selettivo
  ([log dell'esperimento](docs/experiments/glm52-6x5090-2026-07-12.md));
- **128 GB, solo CPU** (Ryzen AI Max+ 395): 1.83 tok/s a caldo
  ([#200](https://github.com/JustVugg/colibri/issues/200));
- **una singola RTX 5070 Ti, macchina di classe laptop:** 1.07 tok/s
  ([#273](https://github.com/JustVugg/colibri/issues/273));
- **il portatile da 25 GB da cui tutto è cominciato:** 0.05-0.1 tok/s a freddo,
  il minimo onesto.

La qualità è misurata, non presunta: il costo del container int4 e le ablazioni
sulla quantizzazione sono in
[benchmarks.md](docs/benchmarks.md#quality-benchmark). Per aggiungere la tua
macchina, segui il [protocollo di benchmark](docs/benchmarking.md) e apri una
issue con i numeri.

<a id="install-by-hand"></a>

## Installazione a mano

<a id="1-get-colibri"></a>

**1. Il programma.** Prendi l'archivio per la tua piattaforma da
[Releases](https://github.com/JustVugg/colibri/releases) (Linux x86_64, macOS,
Windows; non serve un compilatore, solo [Python 3](https://www.python.org/downloads/)
per il launcher e l'API) ed estrailo, poi `python3 coli info`. I motori Linux e
Windows hanno Vulkan dentro, con le loro `shaders/` accanto, quelli macOS Metal;
per una scheda NVIDIA su Windows aggiungi l'archivio CUDA. Oppure compila dai
sorgenti con `gcc` (o clang) e OpenMP:

```bash
git clone https://github.com/JustVugg/colibri && cd colibri/c
./setup.sh                                # checks gcc/OpenMP, builds, self-tests
make qwen36 VK=1                          # one engine, here with Vulkan (CUDA=1 for CUDA)
```

<a id="2-get-the-model"></a>

**2. Un modello.** Qualsiasi download delle [tabelle qui sopra](#which-model-for-my-machine):
gli id dell'installazione corrispondono a repository Hugging Face, e la pagina di
ogni modello riporta il comando di download e di conversione. Per GLM-5.2 usa il
container group-scaled (gs64) con la testa MTP int8,
[`mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp`](https://huggingface.co/mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp)
(429 GB), oppure [`Justvugg/GLM-5.3-colibri-int4-g64`](https://huggingface.co/Justvugg/GLM-5.3-colibri-int4-g64)
per GLM-5.3 (419 GB, senza testa MTP). Non i vecchi mirror int4 per-row:
misurano circa 9 punti in meno sulla qualità e hanno causato le risposte in loop
di [#455](https://github.com/JustVugg/colibri/issues/455).
`./coli convert --model /nvme/glm52_i4` costruisce lo stesso container dalla
release FP8, shard per shard, senza mai aver bisogno di tutti i suoi 756 GB su
disco nello stesso momento. Come controllare la testa MTP e il resto:
[quickstart.md](docs/quickstart.md#3-get-the-model).

<a id="3-run-it"></a>

**3. Avvialo.** Da `c/` in un checkout dei sorgenti, o dalla release estratta.
Il launcher legge il `config.json` del modello e sceglie il motore e il suo chat
template, quindi i comandi sono gli stessi per ogni modello:

```bash
./coli chat  --model /nvme/qwen36          # chat in the terminal
./coli web   --model /nvme/qwen36          # API + dashboard, opens a browser
./coli serve --model /nvme/qwen36          # API + dashboard, no browser
./coli plan  --model /nvme/qwen36          # where the model will live: VRAM, RAM, disk
./coli doctor --model /nvme/qwen36         # read-only check: is everything ready?
./coli tune  --model /nvme/qwen36          # measure and save this machine's fastest safe settings
```

Su Windows un archivio di release include `coli.cmd` (`coli.cmd chat --model
D:\qwen36`); da un checkout dei sorgenti usa `py -3 c\coli`. I file `.exe` sono
i motori, non il launcher. Tutte le opzioni e le variabili:
[SETTINGS.md](docs/SETTINGS.md), [ENVIRONMENT.md](docs/ENVIRONMENT.md).

## Ricerca, e come aiutare

colibri vuole che i modelli di frontiera dipendano meno da hardware scarso e
costino meno da far girare. Questo significa cambiare il modo in cui i pesi sono
memorizzati e spostati, decidere cosa sta in VRAM, in RAM o nello storage,
sovrapporre il lavoro di CPU e GPU, e provare nuovi modi di fare decode. Niente
si tiene perché è convenzionale, e niente si adotta perché un microbenchmark
sembra veloce: il risultato che decide è l'inferenza end-to-end su macchine
reali, con la qualità misurata insieme alla velocità. Le domande aperte:

| ipotesi | evidenza finora | esperimento ancora necessario |
|---|---|---|
| La cronologia di routing può piazzare gli expert meglio di una semplice LRU | i pin appresi migliorano i carichi ripetuti, ma possono sovradattarsi a un prompt | A/B cross-session su dati esclusi dall'apprendimento, con carichi di codice, chat, multilingua e contesto lungo |
| Più SSD possono trasformare banda indipendente in velocità di decode | due NVMe indipendenti hanno misurato +37.5% in decode; un terzo disco più lento è risultato neutro dopo lo striping pesato ([misure](docs/multidisk.md#what-has-been-measured)) | riprodurre con dischi di velocità diverse, disposizioni diverse dei controller e diversi stati della cache |
| Un planner consapevole dell'hardware può avvicinarsi automaticamente alla configurazione migliore di ogni macchina | oggi si rilevano i budget di RAM/VRAM e diversi backend, e l'installazione sceglie la build | confrontare il piano generato con uno sweep controllato dei parametri su portatili, workstation, host NUMA e sistemi multi-GPU |
| Rappresentazioni lossless o a qualità limitata possono ridurre il movimento dei pesi abbastanza da fare la differenza | esistono ablazioni di formato e di quantizzazione, con gate di correttezza/qualità | riprodurre insieme qualità, byte spostati, latenza e costo per token utile, non solo il rapporto di compressione |
| La speculazione consapevole del routing può convenire prima della residenza quasi completa | MTP e draft da grammatica funzionano, ma MTP ha anche misurato una perdita del 32% intorno all'85% di expert hit | mappare la superficie di pareggio tra accettazione, hit rate degli expert, batch union e profondità del draft |
| La sovrapposizione CPU/GPU può nascondere trasferimenti e sincronizzazione invece di spostare soltanto il collo di bottiglia | esistono vantaggi con CUDA, Metal e Vulkan, ma CPU veloci, GPU integrate e bassa residenza possono annullarli | profili per fase e A/B a una variabile su macchine PCIe, a memoria unificata e a residenza completa, e i primi numeri su GPU dedicata per il livello e la catena Vulkan |

Vuoi aiutare? Scegli una riga e pubblica anche i risultati negativi. Registra
hardware, commit, modello, comando esatto, prompt, stato della cache,
throughput, tempo al primo token, hit rate degli expert, byte letti e un
controllo di qualità; cambia una sola variabile, ripeti e allega i log grezzi.
Parti da [CONTRIBUTING.md](CONTRIBUTING.md) e dal [protocollo di benchmark](docs/benchmarking.md),
poi [apri una issue](https://github.com/JustVugg/colibri/issues/new). Qui un
fallimento ben controllato vale più di un numero veloce senza spiegazione.

## Documentazione

| argomento | documento |
|---|---|
| L'installazione in un passo e quella manuale, per ogni piattaforma | [quickstart.md](docs/quickstart.md) |
| Installare tramite un assistente AI, e il server MCP | [AI_SETUP.md](docs/AI_SETUP.md), [MCP_SERVER.md](docs/MCP_SERVER.md) |
| L'API: OpenAI, Anthropic, tool, slot KV, dashboard | [api.md](docs/api.md) |
| System One e i modelli di decisione | [systemone.md](docs/systemone.md), [laya.md](docs/laya.md), [gliner_decide.md](docs/gliner_decide.md), [clef.md](docs/clef.md) |
| Vulkan: il livello degli expert, la catena densa, le schede senza Resizable BAR | [vulkan.md](docs/vulkan.md) |
| CUDA, e il livello CUDA di Qwen3.6 | [cuda.md](docs/cuda.md), [qwen36-cuda-tier.md](docs/qwen36-cuda-tier.md) |
| Apple Silicon, Windows | [metal.md](docs/metal.md), [windows.md](docs/windows.md) |
| Tuning, la cache che impara, prefetch, speculazione | [tuning.md](docs/tuning.md) |
| Più SSD, più macchine | [multidisk.md](docs/multidisk.md), [cluster.md](docs/cluster.md) |
| Benchmark e come misurare | [benchmarks.md](docs/benchmarks.md), [benchmarking.md](docs/benchmarking.md) |
| Ogni opzione e variabile d'ambiente | [SETTINGS.md](docs/SETTINGS.md), [ENVIRONMENT.md](docs/ENVIRONMENT.md) |
| Draft forzati da grammatica, e le ABI sperimentali di embedding | [grammar-draft.md](docs/grammar-draft.md), [segment-runtime.md](docs/segment-runtime.md), [edge-runtime.md](docs/edge-runtime.md) |

## Struttura del repository

```
start-here.sh, START-HERE.bat   the one-step setup (Linux and macOS, Windows)
Makefile                        root build/check entry point
c/
├── colibri.c             GLM-5.2/5.3 engine  (make glm)
├── glm53.c               GLM-5.3-Flash  (make glm53)
├── inkling.c             Inkling  (make inkling)
├── kimi_k3.c             Kimi K3  (make kimi_k3)
├── deepseek_v4.c         DeepSeek V4 Flash  (make deepseek-v4)
├── deepseek_v41.c        DeepSeek V4.1 Flash  (make deepseek_v41)
├── mimo.c                MiMo-V2.6 Flash and Pro  (make mimo)
├── qwen38.c              Qwen3.8-Flash-Next  (make qwen38)
├── qwen36.c              Qwen3.6, Qwen3-Coder, Qwen3.8-27B, Clef  (make qwen36)
├── olmoe.c               OLMoE  (make olmoe)
├── qwenimage.c           Qwen-Image-2.1  (make qwenimage)
├── laya.c                Laya  (make laya)
├── gliner_decide.c       GLiNER2.5-Decide  (make gliner_decide)
│
├── st.h, quant.h, idot.h        safetensors reads, container decoders, integer kernels
├── expert_ffn.h, expert_store.h routed-expert kernel and streaming expert cache
├── tok.h, json.h, compat.h      tokenizer, JSON, Windows/macOS shims
├── route_trace.h, kv_prefix.h   routing telemetry (.coli_usage), KV prefix reuse
├── decide_serve.h               the decision engines' side of System One
│
├── backend_cuda.*        optional CUDA tier   (CUDA=1)
├── backend_metal.*       optional Metal tier  (METAL=1)
├── backend_vulkan.*, vk_tier.c, vk_chain.c   optional Vulkan tier and dense chain (VK=1)
│
├── coli                  user-facing CLI
├── setup_*.py            the one-step setup: hardware, catalog, downloads, flow
├── mcp_server.py         the MCP server (coli mcp)
├── openai_server.py      OpenAI- and Anthropic-compatible HTTP gateway
├── resource_plan.py      RAM/VRAM planner behind coli plan and coli doctor
├── tools/                conversion, fixtures and benchmarks
└── tests/                dependency-free C and Python tests
web/                      the dashboard (a pure API client)
desktop/                  Tauri desktop shell around the dashboard
docker/                   container images
docs/                     reference docs, experiments, media
site/                     the website
```

**Un `.c` per famiglia di modelli, sopra header singoli condivisi.** Un motore
possiede la sua architettura e nient'altro; tutto ciò che serve a due motori vive
in un header che entrambi includono, così una correzione li raggiunge tutti
insieme. Dalla radice del repository, `make`, `make check` e `make clean`
delegano al Makefile del motore.

## Sostenere il progetto

colibri è nato come progetto di una sola persona su un portatile con 12 core e
25 GB di RAM; oggi i suoi numeri arrivano da una comunità di macchine reali. Se
ti è utile:

- metti una stella al repository e condividilo;
- apri issue con i numeri di benchmark del tuo hardware: i datapoint fanno
  avanzare questo progetto più di qualsiasi altra cosa;
- entra nella [comunità Discord](https://discord.gg/RXV83nSZdk) per discutere
  esperimenti, risultati hardware e direzioni di ricerca;
- contattaci tramite le issue di GitHub per sponsorizzare lo sviluppo o donare
  hardware.

## Perché "colibrì"

Il colibrì pesa pochi grammi, resta sospeso in volo e visita un migliaio di fiori
al giorno. Questo motore tiene in vita un gigante da 744 miliardi di parametri
con le razioni di un colibrì: 25 GB di RAM, dodici core CPU e tanta pazienza col
disco.

## Ringraziamenti

colibri è un motore; le menti che fa girare sono un dono. Grazie ai team che
rilasciano apertamente i loro pesi: **Z.ai** (GLM), **Moonshot AI** (Kimi),
**Alibaba Qwen**, **DeepSeek**, **Xiaomi** (MiMo), **Thinking Machines**
(Inkling), **Allen AI** (OLMoE), **Convai Innovations** (Laya), **fastino**
(GLiNER2.5-Decide) e **Cloudflare** (Clef); a chi pubblica i container
convertiti; e a ogni contributore che ha fatto benchmark, bisect, replicato
un'esecuzione dell'atlante o mandato una patch. Il codice di terze parti in
questo repository e le sue licenze: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

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

Apache 2.0, Copyright 2026 Vincenzo Fornaro. Vedi [LICENSE](LICENSE) e [NOTICE](NOTICE). Ogni modello mantiene la licenza che gli hanno dato i suoi autori (i pesi di GLM-5.2 sono rilasciati da Z.ai sotto licenza MIT; Qwen-Image-2.1 è solo per uso non commerciale).
