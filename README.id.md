<p align="center">
  <img src="assets/colibri-logo.svg" width="560" alt="colibrì: engine mungil, model raksasa">
</p>

<p align="center">
  <a href="https://justvugg.github.io/colibri"><img src="https://img.shields.io/badge/website-justvugg.github.io%2Fcolibri-1f6feb" alt="Situs web"></a>
  <a href="https://github.com/JustVugg/colibri/releases"><img src="https://img.shields.io/github/v/release/JustVugg/colibri?color=2ea043" alt="Rilis terbaru"></a>
</p>

<p align="center">
  <a href="https://justvugg.github.io/colibri"><b>Situs Web</b></a> ·
  <a href="https://discord.gg/RXV83nSZdk"><b>Discord</b></a> ·
  <a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.zh-TW.md">繁體中文</a> · <a href="README.it.md">Italiano</a> · <a href="README.ja.md">日本語</a> · Bahasa Indonesia · <a href="README.pt-BR.md">Português (Brasil)</a>
</p>

**Engine mungil, model raksasa.** colibri menjalankan model terbuka yang sangat
besar pada mesin yang sudah Anda miliki. Model mixture-of-experts dengan ratusan
miliar parameter hanya menggunakan sebagian kecil parameternya untuk setiap token,
sehingga colibri menyimpan bagian itu di RAM dan membaca sisanya, yaitu para
pakar, dari disk ketika model membutuhkannya. C murni, satu file per keluarga
model, tanpa memerlukan GPU.

Saat ini ada tiga belas engine yang dapat dijalankan. Sepuluh di antaranya untuk
model bahasa: **GLM-5.2/5.3**, **GLM-5.3-Flash**, **Inkling**, **Kimi K3**,
**DeepSeek V4 Flash**, **DeepSeek V4.1 Flash**, **MiMo-V2.6 Flash** (dan Pro),
**Qwen3.8-Flash-Next**, **Qwen3.6** (yang juga menjalankan Qwen3-Coder dan model
dense Qwen3.8-27B), serta **OLMoE**. Satu menghasilkan gambar:
**Qwen-Image-2.1**. Dua digunakan untuk menjawab pertanyaan pengambilan keputusan:
**Laya** dan **GLiNER2.5-Decide**, dengan model keputusan ketiga, **Clef**, pada
engine Qwen3.6. [Mana yang cocok untuk mesin saya](#which-model-for-my-machine)

```
$ ./coli chat
  colibri v2.0.0 · GLM-5.2 · 744B MoE · int4 · streaming CPU
  ✓ ready in 32s · resident 9.9 GB
  › ciao!
  ◆ Ciao! Come posso aiutarti oggi?
```

<a id="get-started"></a>
<a id="the-one-step-way"></a>

## Mulai dalam satu langkah

Anda setidaknya memerlukan komputer dengan **RAM 8 GB** (16 GB atau lebih disarankan), **ruang kosong 22 GB di disk** untuk model terkecil, dan koneksi
internet. Kartu grafis bersifat opsional.

**Windows**

1. Di halaman ini, klik **Code**, lalu **Download ZIP**, kemudian ekstrak file
   tersebut.
2. Klik dua kali **`START-HERE.bat`** di folder hasil ekstraksi. Jika Python
   belum tersedia, skrip akan menawarkan untuk menginstalnya.

**Linux** (Ubuntu dan Debian; distribusi lain memiliki paket yang sama dengan
nama masing-masing)

```bash
sudo apt install git python3 build-essential
git clone https://github.com/JustVugg/colibri
cd colibri
./start-here.sh
```

**macOS** (dengan [Homebrew](https://brew.sh))

```bash
xcode-select --install
brew install libomp git python
git clone https://github.com/JustVugg/colibri
cd colibri
./start-here.sh
```

Anda menjawab satu pertanyaan, yaitu model mana yang ingin digunakan, dan
menekan Enter akan menerima rekomendasi. Kemudian proses penyiapan:

1. **memeriksa mesin Anda**: RAM, ruang kosong di disk, CPU, dan GPU;
2. **merekomendasikan model yang muat**: bagian model yang selalu berada di
   RAM, ditambah cache minimum pakar, harus muat di RAM Anda, dan unduhannya
   harus muat di disk;
3. **menyiapkan engine**: engine akan di-build untuk mesin Anda jika compiler
   tersedia; jika tidak, versi prebuilt akan diunduh, yang berjalan di CPU dan,
   di Linux dan Windows, juga di GPU Vulkan.
   Build untuk GPU dibuat ketika memberikan keuntungan: CUDA untuk kartu NVIDIA
   di Linux jika CUDA toolkit sudah terinstal, jika tidak maka Vulkan. Pada GPU
   diskret, build GPU selalu digunakan; pada GPU terintegrasi yang berbagi RAM
   dengan CPU, build GPU hanya digunakan untuk model yang terbukti lebih cepat
   di sana (Qwen3.6, Qwen3-Coder, dan Qwen3.8-Flash-Next). Jika ada paket yang
   belum tersedia, proses penyiapan menampilkan perintah yang tepat untuk
   menginstalnya lalu melanjutkan menggunakan CPU; jalankan kembali proses
   penyiapan setelahnya dan engine akan di-build ulang untuk GPU;
4. **mengunduh model** dengan indikator progres dan dukungan resume: hentikan
   kapan saja, jalankan kembali, dan unduhan akan dilanjutkan dari tempat
   sebelumnya berhenti;
5. **menjalankan colibri dan membuka dashboard** di browser Anda, serta
   menampilkan alamat yang dapat digunakan aplikasi lain:

```
Starting colibri
  Browser:             http://127.0.0.1:8000/
  OpenAI base URL:     http://127.0.0.1:8000/v1
  Anthropic base URL:  http://127.0.0.1:8000
  stop: press Ctrl+C here (or close this window)
```

**Pada penggunaan berikutnya**, jalankan kembali `START-HERE.bat` atau
`./start-here.sh`: colibri akan langsung berjalan tanpa mengunduh atau melakukan
build lagi. `c/coli status` menunjukkan apa yang sudah terinstal dan apakah
sedang berjalan, sedangkan `c/coli stop` menghentikannya (`c\coli.cmd status`
dan `c\coli.cmd stop` di Windows).

Opsi ditempatkan setelah `./start-here.sh` atau `START-HERE.bat`:

| Opsi | Fungsinya |
|---|---|
| `--list` | menampilkan setiap model untuk mesin ini, serta alasan jika suatu model tidak muat |
| `--model ID` | menginstal model tersebut (id tersedia di [tabel di bawah](#which-model-for-my-machine)) |
| `--yes` | tanpa pertanyaan: gunakan rekomendasi |
| `--dir DIR` | menyimpan model di disk lain (default `~/colibri-models`) |
| `--backend vulkan`, `cuda` atau `cpu` | memilih sendiri build engine; `--no-gpu` sama dengan `--backend cpu` |
| `--model-dir DIR` | menggunakan model yang sudah Anda unduh |
| `--reconfigure` | memilih model lain |

Penjelasan mendetail mengenai setiap langkah:
[docs/quickstart.md](docs/quickstart.md#the-one-step-way).

### Jika terjadi masalah

| Yang Anda lihat | Yang harus dilakukan |
|---|---|
| proses berhenti saat mengunduh | jalankan kembali perintah yang sama: unduhan dilanjutkan dari byte yang sudah tersimpan di disk |
| `to use the GPU through ..., first run: <command>` | jalankan perintah tersebut, lalu jalankan kembali proses penyiapan: engine akan di-build ulang untuk GPU tanpa mengunduh ulang |
| `the ... build failed`, misalnya `Unsupported gpu architecture` ketika CUDA toolkit yang terinstal tidak lagi mendukung kartu tersebut | proses penyiapan terlebih dahulu memeriksa toolkit terhadap kartu dan memilih Vulkan secara otomatis sambil menjelaskan alasannya; jika build masih gagal, proses tersebut menawarkan pilihan berikutnya (Vulkan, lalu CPU). `./start-here.sh --backend vulkan` memaksa penggunaan Vulkan; `--no-gpu` tetap menggunakan CPU |
| `needs N GB free for the download` | gunakan `--dir` dengan folder pada disk yang lebih besar |
| di WSL, folder model berada di bawah `/mnt/c` | simpan di disk Linux (default, `~/colibri-models`): `/mnt/c` berkali-kali lebih lambat |
| hal lainnya | `c/coli logs -n 50` menampilkan log server dan `c/coli logs --install` menampilkan log proses penyiapan; buka [issue](https://github.com/JustVugg/colibri/issues) dengan menyertakan baris terakhir yang ditampilkan proses penyiapan |

### Biarkan asisten AI Anda menyiapkannya

Jika Anda menggunakan asisten coding AI, asisten tersebut dapat melakukan semua
ini untuk Anda. Minta asisten tersebut:

> Siapkan colibri di mesin ini dengan mengikuti docs/AI_SETUP.md dari https://github.com/JustVugg/colibri

[docs/AI_SETUP.md](docs/AI_SETUP.md) memberikan setiap langkah kepada asisten
sebagai perintah dengan hasil yang dapat dibaca mesin, dan memerintahkannya untuk
meminta persetujuan Anda sebelum mengunduh model atau menginstal paket sistem.
Asisten yang mendukung Model Context Protocol dapat menggunakan server MCP
colibri sebagai gantinya: `coli mcp` menyediakan tools untuk mendeteksi hardware,
merekomendasikan model, menginstal, menjalankan, menghentikan, dan memeriksanya
([docs/MCP_SERVER.md](docs/MCP_SERVER.md)).

### Atau secara manual

Untuk memilih sendiri setiap langkah (rilis prebuilt atau build dari source,
model apa pun dari tabel di bawah, lalu `coli chat`, `coli web`, atau
`coli serve`), lihat [Instal secara manual](#install-by-hand), atau
[panduan Quick Start](docs/quickstart.md) untuk langkah demi langkah pada setiap
platform.

## Apa itu colibri, dan mengapa

Model mixture-of-experts berukuran sangat besar di disk tetapi kecil untuk setiap
token. GLM-5.2 memiliki 744B parameter, menggunakan sekitar 40B untuk setiap
token, dan hanya sekitar 11 GB dari bagian tersebut yang berubah dari satu token
ke token berikutnya: pakar yang dirutekan.

<p align="center">
  <img src="docs/media/sparse.png" width="880" alt="hanya sekitar 5.4% parameter yang aktif per token">
</p>

Jadi, model tidak harus muat di memori cepat; model harus **ditempatkan**.
Bagian dense (attention, shared experts, embeddings) tetap berada di RAM. Pakar
yang dirutekan tetap berada di disk dan dibaca ketika router memintanya, melalui
cache yang mempelajari pakar mana yang digunakan oleh pekerjaan Anda. GPU, jika
tersedia, menyimpan pakar yang paling sering digunakan dan layer dense. Lokasi
sebuah bobot memengaruhi seberapa cepat jawaban diberikan, bukan bobot atau
keputusan router mana yang menghasilkannya.

Mengapa: agar model sebesar ini dapat dijalankan pada hardware yang sudah
dimiliki orang, agar cara kerjanya dapat diamati (dashboard menampilkan setiap
pakar saat aktif), dan agar engine tetap cukup kecil sehingga siapa pun dapat
mengukurnya dan membuatnya lebih cepat. colibri juga merupakan platform riset
terbuka: sebuah optimisasi hanya layak digunakan jika didukung pengukuran
end-to-end yang dapat direproduksi, dan kebijakan default tidak pernah diam-diam
mengubah presisi model atau semantik router. Memori cepat yang lebih sedikit
dapat mengurangi kecepatan; hal itu tidak boleh diam-diam mendefinisikan ulang
model. [Cara kerjanya](#how-it-works) menjelaskan detailnya.

<a id="which-model-for-my-machine"></a>

## Model mana yang cocok untuk mesin saya

Proses penyiapan merekomendasikan model paling kapabel yang dapat berjalan dari
RAM pada mesin Anda, lalu mencantumkan tepat di bawahnya model yang lebih besar
yang melakukan streaming dari disk. `./start-here.sh --list` menampilkan
semuanya untuk mesin Anda. Tabel berikut mengikuti katalog milik proses
penyiapan ([`c/setup_catalog.py`](c/setup_catalog.py)); ukuran unduhan adalah
ukuran yang dicantumkan Hugging Face untuk setiap repository.

- **RAM** terdiri dari dua angka: di bawah angka pertama model tidak dapat
  dijalankan, sedangkan mulai dari angka kedua model berjalan seperti yang
  telah diukur.
- **GPU** menunjukkan apa yang dapat di-build oleh proses penyiapan untuk engine
  tersebut ([GPU](#gpus)). *termasuk terintegrasi* berarti engine juga menggunakan GPU
  terintegrasi, karena engine tersebut terukur lebih cepat di sana; engine lain
  hanya menggunakan GPU diskret.
- **Hasil pengukuran** adalah hasil yang diukur pada mesin yang disebutkan,
  berupa kecepatan decode untuk model chat. Huruf-hurufnya merujuk pada mesin
  yang dicantumkan di bawah tabel. Kolom kosong berarti belum ada yang
  mengukurnya.

**Model kecil, yang berjalan dari RAM**

| Model | `--model` | Unduhan | RAM | GPU | Hasil pengukuran |
|---|---|---|---|---|---|
| **Qwen3.6-35B-A3B**: chat dengan thinking dan tools | `qwen36-35b` | 23 GB | 10 / 20 GB | CUDA, Vulkan (termasuk terintegrasi) | 6.0 tok/s pada CPU, 9.9 dengan Vulkan pada GPU terintegrasi (A); 30.0 dengan CUDA (C) |
| Qwen3-Coder-30B-A3B: kode dan tool calls, tanpa thinking | `qwen3-coder-30b` | 19 GB | 8 / 18 GB | CUDA, Vulkan (termasuk terintegrasi) | 8.5-9.6 tok/s dengan semua pakar di RAM, 5.1 dengan 32 per layer (A, CPU) |
| **Qwen-Image-2.1**: teks ke gambar, lisensi nonkomersial | `qwen-image-2.1` | 33 GB | 12 / 18 GB | Vulkan | satu gambar 768x512 dalam 2 min 40 s (8 core Zen 4, CPU) |

**Model besar, yang pakarnya melakukan streaming dari disk** (disk menentukan
kecepatan: drive NVMe yang cepat paling membantu)

| Model | `--model` | Unduhan | RAM | GPU | Hasil pengukuran |
|---|---|---|---|---|---|
| DeepSeek V4 Flash REAP 150B: 132 dari 256 pakar | `deepseek-v4-flash-reap` | 85 GB | 16 / 32 GB | CUDA, Vulkan | |
| **DeepSeek V4 Flash** (284B): tools | `deepseek-v4-flash` | 167 GB | 16 / 32 GB | CUDA, Vulkan | 0.93 tok/s dengan 32 GB (Ryzen 7 5800X), 1.24 dengan 63 GB (Ryzen 9 5950X), hanya CPU; 1.5-1.6 dengan CUDA (RTX 5080, 32 GB, dua NVMe) |
| **MiMo-V2.6 Flash** (309B): vision dan tools | `mimo-v2.6-flash` | 172 GB | 32 / 52 GB | Vulkan | 2.34-3.37 tok/s (A, CPU) |
| **Qwen3.8-Flash-Next** (125B + 51B n-gram): vision dan tools | `qwen38-flash-next` | 186 GB | 24 / 32 GB | CUDA, Vulkan (termasuk terintegrasi) | 1.91-2.56 tok/s dengan 32-96 pakar per layer; 3.99 dengan pakar int4 opsional (A, CPU) |
| **GLM-5.2** (744B): model referensi, dengan MTP head | `glm-5.2` | 429 GB | 16 / 24 GB | CUDA, Vulkan | 0.05-0.1 tok/s cold pada laptop 25 GB; 1.83 pada Ryzen AI Max+ 395 dengan 128 GB; 9.0-9.2 pada 6x RTX 5090 |
| GLM-5.3 (744B): engine yang sama, tanpa MTP head | `glm-5.3` | 419 GB | 16 / 24 GB | CUDA, Vulkan | |
| **Inkling** (975B): pakar int4, bobot dense bf16 | `inkling` | 514 GB | 120 / 128 GB sebagaimana diunduh; 25 GB setelah [konversi dense](docs/inkling.md) | CUDA, Vulkan | 0.25 tok/s (Ryzen 9 7900, 187 GB, RTX A6000) |
| MiMo-V2.6 Pro (1.02T): vision dan tools | `mimo-v2.6-pro` | 564 GB | 54 / 64 GB | Vulkan | 0.66-0.79 tok/s (A, CPU) |
| **Kimi K3** (2.8T): yang terbesar | `kimi-k3` | 1.56 TB | 32 / 64 GB | CUDA, Vulkan | sekitar 9.4 s per token, pakar dibaca pada 6.3 GB/s |

<a id="other-supported-models"></a>

**Secara manual: langkah konversi atau persiapan setelah unduhan**

| Model | Unduhan, lalu di disk | RAM | GPU | Hasil pengukuran |
|---|---|---|---|---|
| **OLMoE** (7B): kecil, untuk mempelajari tools | 14 GB, 7 GB setelah konversi ke int8 | 8 GB | Vulkan | 22-23 tok/s (A, CPU) |
| Qwen3.8-27B (dense): teks dan gambar | 56 GB, 51 GB setelah konversi | 20 GB dalam int4, 30 GB dalam int8 | Vulkan | 3.45 tok/s dalam int4, 2.1 dalam int8 (server CPU 16-thread) |
| **GLM-5.3-Flash** (321B): vision dan tools | 328 GB, dikonversi shard demi shard menjadi 195 GB | 25 GB | CUDA, Vulkan | sekitar 20 s per token saat warm, 44 s saat cold (6 core, 25 GB, disk biasa) |
| **DeepSeek V4.1 Flash** (552B): vision dan tools, tanpa konversi tetapi memerlukan persiapan satu kali | 510 GB | sekitar 18 GB ditambah cache pakar (peak 24.8 GB dengan 8 per layer) | Vulkan | 0.21-0.24 tok/s (server CPU 16-thread yang menyimpan 68% pakar) |

**Model keputusan** (menjawab pertanyaan [System One](#system-one-a-decision-with-a-probability), bukan untuk chat)

| Model | Unduhan, lalu di disk | RAM | GPU | Hasil pengukuran |
|---|---|---|---|---|
| **Laya** (Convai Innovations), bahasa Inggris | 0.85 GB | 1.7 GB | CPU | 219 ms untuk satu pertanyaan, 882 ms untuk tiga (B) |
| **GLiNER2.5-Decide** (fastino), bahasa Inggris | 1.95 GB | 1.9 GB | CPU | 294 ms untuk satu pertanyaan, 897 ms untuk tiga (B, under load) |
| **Clef** (Cloudflare): Qwen3.8-27B dengan decision head, juga dapat chat | 55 GB, 52 GB setelah konversi | 19 GB dalam int4 hingga 55 GB dalam f16 | CPU | 20.4 s per request dalam int8 (A) |

Mesin yang digunakan:
**A** desktop Ryzen 7 PRO 8700GE (8 core, 61-64 GB DDR5, NVMe, Radeon 780M
terintegrasi);
**B** laptop i7-1355U;
**C** RTX 3070 8 GB dalam mesin Threadripper 3945WX, dengan layer dense dan
layer DeltaNet pada kartu (container int4 per-row).
Setiap angka berasal dari halaman model masing-masing di [docs/](docs/) atau
dari [tabel benchmark](docs/benchmarks.md), beserta pengaturan yang tepat.

Setiap keluarga memiliki halamannya sendiri: [qwen36.md](docs/qwen36.md)
(Qwen3.6, Qwen3-Coder, Qwen3.8-27B), [qwen38.md](docs/qwen38.md),
[deepseek-v4.md](docs/deepseek-v4.md), [deepseek-v41.md](docs/deepseek-v41.md),
[mimo.md](docs/mimo.md), [glm53-flash.md](docs/glm53-flash.md),
[inkling.md](docs/inkling.md), [kimi_k3.md](docs/kimi_k3.md),
[qwen-image.md](docs/qwen-image.md), [laya.md](docs/laya.md),
[gliner_decide.md](docs/gliner_decide.md), [clef.md](docs/clef.md), dan GLM-5.2
di [Quick Start](docs/quickstart.md#3-get-the-model).
Checkpoint dengan arsitektur yang sama seperti yang didukung, misalnya
KAT-Coder v2.5 pada engine Qwen3.6, dapat dijalankan tanpa perubahan.

<a id="gpus"></a>

## GPU

### Tidak memerlukan GPU

Setiap engine berjalan di CPU tanpa perlu menginstal apa pun lagi. GPU adalah
tempat yang lebih cepat untuk menyimpan bobot, bukan suatu keharusan: untuk
model besar, disk menentukan kecepatannya; untuk model kecil, RAM.

### Vulkan: GPU apa pun

Setiap engine MoE dapat menggunakan GPU apa pun dengan driver Vulkan 1.2 (AMD,
Intel, NVIDIA, terintegrasi maupun diskret) dalam dua cara:

- **expert tier**: cache pakar yang dirutekan di memori GPU, diisi saat startup
  dari pakar yang digunakan percakapan Anda sebelumnya dan terus beradaptasi
  ketika Anda chat. GPU menghitung pakar yang disimpannya sementara CPU
  menghitung sisanya;
- **dense chain**: satu layer utuh direkam sebagai satu submission GPU, dengan
  running state model tetap berada di GPU dari satu layer ke layer berikutnya.

Diukur pada Radeon 780M terintegrasi milik mesin A, dengan file model dikeluarkan
dari page cache sebelum setiap run, dan 100 token di-decode
([vulkan.md](docs/vulkan.md#the-chain-on-a-radeon-780m)):

| | CPU | Vulkan, expert tier | Vulkan, tier dan dense chain |
|---|---|---|---|
| Qwen3.6-35B-A3B, decode | 6.0 tok/s | 8.0 tok/s | **9.9 tok/s** |
| Qwen3.6-35B-A3B, prompt 512 token | 35.7 s | 12.2 s | **9.5 s** |
| Qwen3.8-Flash-Next (pakar int4), decode | 3.5 tok/s | **3.8 tok/s** | 3.2 tok/s |
| Qwen3.8-Flash-Next, prompt 512 token | 43.6 s | 38.7 s | **30.1 s** |
| OLMoE, decode (warm) | **23.1 tok/s** | 12.8 tok/s | 17.3 tok/s |

GPU terintegrasi berbagi RAM dengan CPU. Keuntungan yang diberikannya adalah
mengurangi pekerjaan dan pembacaan disk untuk pakar yang disimpannya, sehingga
bermanfaat pada model seperti Qwen3.6; sebaliknya, model kecil yang pakarnya
sudah berada di RAM, seperti OLMoE, dapat menjadi lebih lambat. Karena itu,
proses penyiapan hanya mengaktifkan Vulkan pada GPU terintegrasi untuk Qwen3.6,
Qwen3-Coder, dan Qwen3.8-Flash-Next, dan setiap engine memutuskan sendiri apakah
dense chain dijalankan di sana (Qwen3.6 ya, Qwen3.8 tidak). `--backend vulkan`
tetap meminta Vulkan secara eksplisit.

**Menyalakan atau mematikan GPU.** `coli setup --backend vulkan` memakai GPU untuk
model apa pun, dan `coli setup --backend cpu` (atau `--no-gpu`) menjalankan
semuanya di CPU. Engine yang di-build dengan Vulkan memakai GPU hanya jika ada
`COLI_VULKAN=1` di environment `coli chat`, `serve` atau `web` (proses penyiapan
mengaturnya ketika memilih Vulkan); tanpanya, engine berjalan di CPU. Dengan GPU
menyala, `COLI_VK_CHAIN=0` mempertahankan expert tier dan menjalankan layer dense
di CPU. Pada GPU terintegrasi, coba keduanya: pada laptop dengan Intel Iris Xe
(Core i7-1355U), Qwen3.6 mendekode 2,1 tok/s di CPU, 1,7 sampai 1,9 dengan Vulkan,
dan 2,1 dengan dense chain dimatikan.

Pada GPU diskret, proses penyiapan membuat build Vulkan untuk setiap engine
(CUDA terlebih dahulu jika engine memilikinya dan toolkit sudah terinstal),
dengan layer dense berada di kartu. Inilah skenario yang menjadi sasaran desain
tersebut. **Kami belum mengukur GPU diskret sendiri.** Angka pertama berasal
dari seorang pengguna: Qwen3.6 pada Tesla V100 16 GB mencapai 17 hingga 19
tok/s, dengan expert tier dan dense chain
([#1852](https://github.com/JustVugg/colibri/issues/1852)). (Sebelumnya, jalur
Vulkan lama milik GLM-5.2 menghasilkan 1.7-1.8 tok/s pada RX 9070 diskret.)
Kami menyambut hasil pengukuran dari kartu Anda.

**Kartu tanpa Resizable BAR** kini dapat digunakan. Kartu seperti itu (semua
kartu Turing, kartu Ampere dengan firmware peluncurannya, kartu AMD lama dengan
opsi tersebut dimatikan) hanya memungkinkan CPU menulis langsung ke sekitar 256 MB memori kartu
tersebut; colibri kini menyalin bobot melalui staging buffer
secara mandiri. Jalur ini diuji dengan memaksanya dan dengan mengemulasikan
window kecil pada tiga perangkat, dan tidak menimbulkan biaya yang terukur pada
780M; jalur ini belum diukur pada kartu tanpa Resizable BAR
([vulkan.md](docs/vulkan.md#memory-placement-without-resizable-bar)).

CI memeriksa jalur Vulkan setiap engine terhadap token CPU pada software driver.
GPU menjumlahkan angka dalam urutan yang berbeda, dan mempertahankan beberapa
aktivasi dalam f32 ketika CPU membulatkannya, sehingga jawaban panjang dapat
menyimpang satu kata dari hasil CPU
([vulkan.md](docs/vulkan.md#the-other-engines)).

### CUDA: kartu NVIDIA

Proses penyiapan membuat build CUDA di Linux ketika CUDA toolkit terinstal,
untuk engine yang memiliki jalur CUDA: GLM-5.2/5.3, GLM-5.3-Flash, Inkling,
Kimi K3, DeepSeek V4 Flash, Qwen3.8-Flash-Next, dan Qwen3.6 dengan Qwen3-Coder.
Di Windows, engine CUDA merupakan DLL terpisah
([windows.md](docs/windows.md)), dan setiap rilis menyertakannya dalam keadaan
sudah di-build: `colibri-<versi>-windows-x86_64-cuda.zip` berisi `coli_cuda.dll`
(kartu dengan compute capability 8.0 ke atas) serta engine colibri, qwen36 dan
kimi_k3 yang memuatnya. Ekstrak di atas archive utama, dan proses penyiapan
memilih CUDA.

- **VRAM expert tier** menyimpan pakar yang paling sering digunakan di kartu,
  dipilih berdasarkan routing yang terukur; cache miss dihitung di CPU pada saat
  yang sama. Qwen3.6 pada dua kartu 8 GB (RTX 3070 dan Quadro RTX 4000)
  menghasilkan 11.3 tok/s dengan riwayat yang warm
  ([qwen36-cuda-tier.md](docs/qwen36-cuda-tier.md)); GLM-5.2 pada enam RTX 5090
  dengan semua pakar resident menghasilkan 9.0-9.2 tok/s
  ([benchmarks.md](docs/benchmarks.md)); DeepSeek V4 Flash pada RTX 5080
  menghasilkan 1.5-1.6 tok/s dan prompt 3,324 token dalam 90 s
  ([deepseek-v4.md](docs/deepseek-v4.md)).
- **Baru untuk Qwen3.6: layer DeltaNet di kartu** (`Q36_DN_GPU=1`, opt-in).
  Masing-masing dari 30 layer DeltaNet Qwen3.6 sebelumnya menyalin datanya antara
  kartu dan CPU empat kali per token; sekarang satu token decode menjalankan
  seluruh layer di kartu, dengan recurrent state tetap berada di VRAM. Pada RTX
  3070 dengan layer dense di VRAM: 25.4 hingga 30.0 tok/s
  ([qwen36-cuda-tier.md](docs/qwen36-cuda-tier.md#the-deltanet-layer-on-the-card-q36_dn_gpu1)).
- **Kartu lama.** Jika CUDA toolkit tidak lagi dapat mengompilasi untuk kartu
  Anda (CUDA 13 dan V100, dalam
  [#1852](https://github.com/JustVugg/colibri/issues/1852)), proses penyiapan
  mendeteksinya sebelum build dan menggunakan kartu melalui Vulkan sambil
  menjelaskan alasannya; CUDA 12.x toolkit mengaktifkan kembali jalur CUDA.
  CUDA tier DeepSeek V4 juga dapat di-build untuk Pascal dan Turing
  (`CUDA_ARCH=portable-pre-ampere NO_TC=1`).

Selengkapnya: [docs/cuda.md](docs/cuda.md).

### Apple Silicon

Backend Metal menjalankan perhitungan pakar pada GPU unified-memory untuk
beberapa engine ([docs/metal.md](docs/metal.md)). Archive macOS dari rilis berisi
`colibri`, `inkling` dan `kimi_k3` yang di-build dengan Metal: `COLI_METAL=1`
(`K3_METAL=1` untuk Kimi K3) menyalakannya, dan tanpanya engine berjalan di CPU.
Dari source, build dengan `METAL=1`; proses penyiapan satu langkah membuat build
untuk CPU.

<a id="system-one-mode-ask-a-closed-question"></a>

<a id="system-one-a-decision-with-a-probability"></a>

## System One: keputusan dengan probabilitas

Sebagian besar hal yang diminta orang dari sebuah model adalah pilihan, bukan
paragraf: antrean mana, keputusan mana, ya atau tidak. `POST /v1/systemone`
menerima sebuah state (teks atau JSON) dan pertanyaan bertipe, lalu menjawab
setiap pertanyaan dengan probabilitas untuk setiap opsi yang diizinkan beserta
confidence. Tidak ada teks yang dihasilkan, sehingga jawaban tidak mungkin
berada di luar daftar Anda, dan "model tidak yakin" menjadi angka yang dapat
Anda beri threshold.

```bash
curl -s http://127.0.0.1:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "340 lines, 8 files, no tests. CI is green but nothing covers that path.",
  "questions": {
    "review": {"type": "choice", "instructions": "What should the reviewer do?",
               "criteria": {"merge": null, "request changes": null, "close": null}},
    "risky":  {"type": "noul", "instructions": "Is this change risky?"}}}'
```

Sebuah `choice` mengembalikan label yang dipilih, probabilitas untuk setiap
label, dan `confidence` dari 0 (merata) hingga 1 (pasti); sebuah `noul`
mengembalikan probabilitas untuk yes; sebuah `score` mengembalikan level yang
diharapkan.

Yang menjawab:

- **Model chat apa pun yang dapat dijalankan colibri**, melalui scoring: model
  membaca probabilitas setiap opsi alih-alih menulis jawaban. Banyak pertanyaan
  mengenai satu dokumen membaca dokumen tersebut satu kali: pada Qwen3.6,
  empat pertanyaan mengenai satu dokumen selesai 5.7x lebih cepat daripada
  menghasilkan jawaban yang sama pada mesin CPU yang sama.
- **Tiga model keputusan**, secara native, dalam satu forward pass dengan
  kalibrasi yang disesuaikan oleh pembuatnya: [Laya](docs/laya.md)
  (Convai Innovations), [GLiNER2.5-Decide](docs/gliner_decide.md) (fastino),
  dan [Clef](docs/clef.md) (Cloudflare; model ini juga dapat chat). Ukuran dan
  kecepatannya tersedia di
  [tabel model keputusan](#which-model-for-my-machine).

**Beralih dari Jev.** Request dan reply menggunakan format API Jev dari TypeSafe,
sehingga client Jev dapat beralih ke colibri hanya dengan mengganti base URL,
tanpa perubahan lain:
`TYPESAFE_BASE_URL=http://127.0.0.1:8000` (key yang sudah dikirim client tetap
diterima oleh server yang dijalankan tanpa `COLI_API_KEY`). Kedua SDK resmi,
tanpa modifikasi, diuji terhadap `coli serve`.

Mode yang sama juga tersedia di terminal
(`/decide merge | request changes | close` dalam `coli chat`) dan pada halaman
System One di dashboard. Request dan reply lengkap, aturan scoring, serta kapan
mode ini tidak membantu dijelaskan di
[docs/systemone.md](docs/systemone.md).

## Dashboard

`coli web` membukanya, begitu juga proses penyiapan satu langkah: chat, halaman
System One, Brain, dan halaman Profiling, dalam tema terang atau gelap.

<p align="center">
  <img src="docs/media/colibri-dashboard.png" width="900" alt="dashboard web colibri: chat, metrik langsung, panel hardware, tier pakar">
</p>
<p align="center"><em>Qwen3.6 menjawab pada mesin CPU, dengan pakar yang di-streaming dari disk.</em></p>

<p align="center">
  <img src="docs/media/colibri-brio.png" width="900" alt="halaman System One: sebuah dokumen dibaca sekali, probabilitas untuk setiap jawaban yang diizinkan, dan entropy">
</p>
<p align="center"><em><strong>System One</strong>: berikan model sebuah dokumen dan hanya jawaban yang boleh dipilihnya. Di sini:
<strong>request changes pada 99.9%</strong>, entropy 0.005, 4 token dibaca, 0 dihasilkan.</em></p>

<p align="center">
  <img src="docs/media/colibri-brain.png" width="900" alt="halaman Brain: atlas pakar GLM-5.2 yang terukur digambar sebagai korteks, sepuluh wilayah yang dapat dimasuki">
</p>
<p align="center"><em><strong>Brain</strong>: <a href="https://github.com/JustVugg/colibri/issues/175">atlas pakar yang terukur</a> dari GLM-5.2,
13,260 pakar yang telah dikarakterisasi dalam sepuluh wilayah (Python, SQL, matematika, puisi, hukum, bahasa Tionghoa...), ditempatkan berdasarkan
afinitas routing yang terukur. <strong>Live routing</strong> menampilkan model yang sedang berjalan: satu sel per pakar, diberi warna berdasarkan
storage tier, dan setiap pakar yang dirutekan dalam satu turn akan berkedip.</em></p>

Halaman **Profiling** menunjukkan di mana waktu setiap turn digunakan, fase demi
fase, dengan 30 turn terakhir sebagai tren.

## Gunakan dari aplikasi lain

`coli serve` (yang dijalankan proses penyiapan untuk Anda) adalah satu server
dengan beberapa API:

- **kompatibel dengan OpenAI**: `/v1/chat/completions`, `/v1/completions`, dan
  `/v1/models`, dengan streaming, reply JSON, stop sequences, dan logprobs;
- **kompatibel dengan Anthropic**: `/v1/messages`, sehingga Claude Code dan SDK
  Anthropic dapat digunakan dengannya;
- **tool calling** pada setiap engine chat kecuali Inkling dan OLMoE, masing-masing
  dalam format native modelnya
  ([tabel per-engine](docs/api.md#tool-calling-support));
- **input gambar** pada GLM-5.3-Flash, DeepSeek V4.1 Flash, MiMo-V2.6,
  Qwen3.8-Flash-Next, dan Qwen3.8-27B: sebuah path dalam pesan `coli chat`,
  attachment di `coli web`, atau bagian `image_url`;
- **output gambar** dengan Qwen-Image-2.1 pada `POST /v1/images/generations`,
  dan digambar langsung di terminal oleh `coli chat`
  ([qwen-image.md](docs/qwen-image.md));
- **keputusan** pada `POST /v1/systemone`
  ([di atas](#system-one-a-decision-with-a-probability));
- **beberapa percakapan sekaligus** di setiap engine teks: `coli serve
  --kv-slots N` menyimpan hingga 16, masing-masing dengan cache sendiri, dan
  mendekode token berikutnya bersama-sama ([api.md](docs/api.md#isolated-kv-contexts)).

CLI coding dan editor terhubung seperti ke provider lain yang kompatibel dengan
OpenAI: base URL `http://127.0.0.1:8000/v1`, model id yang ditampilkan
`coli status`, dan key apa pun yang tidak kosong
([docs/api.md](docs/api.md#connect-a-coding-cli-or-editor)).

<a id="how-it-works"></a>

## Cara kerjanya

<p align="center">
  <img src="docs/media/token-path.png" width="880" alt="route, union, place, overlap, learn">
</p>

Setiap layer dari setiap token melewati lima langkah yang sama: route, union,
place, overlap, learn. Tujuan desainnya adalah agar **placement hanya menentukan
kecepatan**: keputusan router dan presisi bobot tetap sama, baik pakar menjawab
dari VRAM, RAM, maupun disk.

<p align="center">
  <img src="docs/media/tiers.png" width="880" alt="VRAM, RAM and NVMe as three tiers of expert residency">
</p>

- **JIT untuk bobot.** Compiler JIT tidak pernah mengompilasi seluruh program:
  ia mengamati bagian yang berjalan dan mengompilasi hot path. colibri membuat
  taruhan yang sama pada bobot. Routing heat yang terukur menentukan pakar mana
  yang layak berada di VRAM, RAM, atau disk: cache LRU per-layer, ditambah hot
  set yang di-pin dan dipelajari dari percakapan Anda sendiri (`.coli_usage`,
  diperbarui setiap turn). colibri menjadi lebih cepat semakin sering digunakan.
  Hal ini bekerja karena routing memiliki struktur yang dapat diukur
  ([expert atlas](https://github.com/JustVugg/colibri/issues/175)).
- **Jangan menunggu disk dua kali.** Tiga matriks milik satu pakar dibaca dalam
  satu `pread`; sekumpulan loader membaca pakar yang belum tersedia sementara
  pakar resident melakukan komputasi; satu batch posisi membaca setiap pakar
  sekali; thread router-lookahead dapat melakukan prefetch untuk layer berikutnya
  (routing GLM-5.2 dapat diprediksi 71.6% satu layer ke depan). `DIRECT=1`
  (O_DIRECT) sering memberikan peningkatan besar pada drive NVMe cepat dan netral
  atau lebih buruk pada drive lain: ukur pada perangkat Anda
  ([tuning.md](docs/tuning.md)).
- **Lebih dari satu SSD.** `COLI_MODEL_MIRROR=/second/glm52_i4 ./coli chat --model /fast/glm52_i4`
  membaca dari salinan pada drive kedua. Dua drive NVMe pada controller terpisah
  menghasilkan peningkatan decode sebesar +37.5%; mirror parsial pada drive yang
  lebih kecil juga dapat digunakan ([multidisk.md](docs/multidisk.md)).
- **Dari laptop hingga rack.** Pada laptop 25 GB, setiap pakar di-streaming dari
  disk, lambat tetapi tetap benar; pada host besar, setiap pakar resident
  (`CUDA_EXPERT_GB=auto PIN_GB=all`) dan disk tidak lagi terlibat dalam decoding.
  `COLI_NUMA=1` menyebarkan bobot resident ke memory controller pada host
  multi-socket, dan mode cluster lokal menjalankan pakar yang dirutekan pada
  mesin lain ([cluster.md](docs/cluster.md)).
- **Model yang setia pada referensi.** Setiap engine diperiksa di CI terhadap
  implementasi referensi modelnya menggunakan fixture kecil. MLA attention
  GLM-5.2 menyimpan compressed KV state (576 float per token, bukan 32,768,
  57x lebih kecil) yang bertahan setelah restart, sehingga percakapan dapat
  dibuka kembali tanpa membaca ulang prompt.
- **Spekulasi yang sepadan dengan biayanya.** Int8 MTP head GLM-5.2 membuat draft
  2.2-2.8 token per forward ketika menguntungkan; MTP head Qwen3.8-Flash-Next,
  opt-in, menambah 12-14% dengan output yang sama. Jika drafting membutuhkan
  biaya lebih besar daripada penghematannya (DeepSeek V4), fitur tersebut tetap
  dimatikan ([tuning.md](docs/tuning.md#speculation-and-reproducibility)).

Engine menggunakan satu file C per keluarga model (`c/colibri.c` untuk GLM-5.2)
di atas header bersama, tanpa BLAS dan tanpa Python saat runtime: Python hanya
menjalankan proses penyiapan, launcher, converter, dan API gateway.

<a id="what-it-achieves"></a>

## Benchmark

<p align="center">
  <img src="docs/media/ladder.png" width="880" alt="kecepatan decode GLM-5.2 yang terukur berdasarkan kelas hardware">
</p>

Engine yang sama dan container int4 yang sama: hardware hanya mengubah tempat
pakar berada. Decode GLM-5.2, dari [tabel lengkap](docs/benchmarks.md):

- **6x RTX 5090, semua pakar resident:** 5.8-6.8 tok/s, 9.0-9.2 dengan selective
  NUMA interleave
  ([log eksperimen](docs/experiments/glm52-6x5090-2026-07-12.md));
- **128 GB, hanya CPU** (Ryzen AI Max+ 395): 1.83 tok/s warm
  ([#200](https://github.com/JustVugg/colibri/issues/200));
- **satu mesin kelas laptop RTX 5070 Ti:** 1.07 tok/s
  ([#273](https://github.com/JustVugg/colibri/issues/273));
- **laptop 25 GB tempat proyek ini bermula:** 0.05-0.1 tok/s cold, batas bawah
  yang realistis.

Kualitas diukur, bukan diasumsikan: biaya container int4 dan ablation
quantization tersedia di
[benchmarks.md](docs/benchmarks.md#quality-benchmark). Untuk menambahkan mesin
Anda, ikuti [protokol benchmark](docs/benchmarking.md) dan buka issue dengan
hasil pengukurannya.

<a id="install-by-hand"></a>

## Instal secara manual

<a id="1-get-colibri"></a>

**1. Programnya.** Ambil archive untuk platform Anda dari
[Releases](https://github.com/JustVugg/colibri/releases) (Linux x86_64, macOS,
Windows; tidak memerlukan compiler, hanya [Python 3](https://www.python.org/downloads/)
untuk launcher dan API), ekstrak, lalu jalankan `python3 coli info`. Engine
Linux dan Windows sudah berisi Vulkan, dengan `shaders/` di sampingnya, dan engine
macOS berisi Metal; untuk kartu NVIDIA di Windows tambahkan archive CUDA. Atau build
dari source menggunakan `gcc` (atau clang) dan OpenMP:

```bash
git clone https://github.com/JustVugg/colibri && cd colibri/c
./setup.sh                                # checks gcc/OpenMP, builds, self-tests
make qwen36 VK=1                          # one engine, here with Vulkan (CUDA=1 for CUDA)
```

<a id="2-get-the-model"></a>

**2. Model.** Gunakan unduhan apa pun dari
[tabel di atas](#which-model-for-my-machine): id milik proses penyiapan dipetakan
ke repository Hugging Face, dan halaman setiap model memiliki perintah unduhan
dan konversinya. Untuk GLM-5.2 gunakan container group-scaled (gs64) dengan int8
MTP head,
[`mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp`](https://huggingface.co/mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp)
(429 GB), atau [`Justvugg/GLM-5.3-colibri-int4-g64`](https://huggingface.co/Justvugg/GLM-5.3-colibri-int4-g64)
untuk GLM-5.3 (419 GB, tanpa MTP head). Jangan gunakan mirror int4 per-row yang
lebih lama: hasil pengukurannya sekitar 9 poin lebih buruk dalam kualitas dan
menyebabkan jawaban berulang pada
[#455](https://github.com/JustVugg/colibri/issues/455). `./coli convert --model
/nvme/glm52_i4` membuat container yang sama dari rilis FP8, shard demi shard,
tanpa pernah memerlukan seluruh 756 GB berada di disk sekaligus. Cara memeriksa
MTP head dan bagian lainnya:
[quickstart.md](docs/quickstart.md#3-get-the-model).

<a id="3-run-it"></a>

**3. Jalankan.** Dari `c/` pada source checkout, atau dari rilis yang telah
diekstrak. Launcher membaca `config.json` milik model dan memilih engine serta
chat template-nya, sehingga perintahnya sama untuk setiap model:

```bash
./coli chat  --model /nvme/qwen36          # chat in the terminal
./coli web   --model /nvme/qwen36          # API + dashboard, opens a browser
./coli serve --model /nvme/qwen36          # API + dashboard, no browser
./coli plan  --model /nvme/qwen36          # where the model will live: VRAM, RAM, disk
./coli doctor --model /nvme/qwen36         # read-only check: is everything ready?
./coli tune  --model /nvme/qwen36          # measure and save this machine's fastest safe settings
```

Di Windows, archive rilis menyertakan `coli.cmd` (`coli.cmd chat --model
D:\qwen36`); dari source checkout gunakan `py -3 c\coli`. File `.exe` adalah
engine, bukan launcher. Semua opsi dan variabel:
[SETTINGS.md](docs/SETTINGS.md), [ENVIRONMENT.md](docs/ENVIRONMENT.md).

## Riset dan cara membantu

colibri ingin agar model frontier tidak terlalu bergantung pada hardware yang
langka dan lebih murah untuk dijalankan. Itu berarti mengubah cara bobot
disimpan dan dipindahkan, menentukan apa yang berada di VRAM, RAM, atau storage,
menumpangtindihkan pekerjaan CPU dan GPU, serta menguji cara baru untuk decode.
Tidak ada sesuatu yang dipertahankan hanya karena sudah menjadi kebiasaan, dan
tidak ada sesuatu yang diadopsi hanya karena microbenchmark terlihat cepat:
hasil yang menentukan adalah inference end-to-end pada mesin nyata, dengan
kualitas diukur bersama kecepatan. Pertanyaan terbukanya:

| hipotesis | bukti sejauh ini | eksperimen yang masih diperlukan |
|---|---|---|
| Riwayat routing dapat menempatkan pakar lebih baik daripada LRU biasa | learned pins meningkatkan workload berulang, tetapi dapat overfit pada sebuah prompt | A/B held-out lintas sesi pada workload coding, chat, multilingual, dan long-context |
| Beberapa SSD dapat mengubah bandwidth independen menjadi kecepatan decode | dua drive NVMe independen menghasilkan peningkatan decode +37.5%; drive ketiga yang lebih lambat netral setelah weighted striping ([pengukuran](docs/multidisk.md#what-has-been-measured)) | reproduksi pada berbagai kecepatan drive, layout controller, dan state cache |
| Planner yang memahami hardware dapat mendekati konfigurasi terbaik setiap mesin secara otomatis | budget RAM/VRAM dan beberapa backend sudah terdeteksi saat ini, dan proses penyiapan memilih build | bandingkan plan yang dihasilkan dengan controlled parameter sweep pada laptop, workstation, host NUMA, dan sistem multi-GPU |
| Representasi lossless atau dengan batas kualitas dapat mengurangi perpindahan bobot sampai cukup berarti | ablation format dan quantization sudah tersedia, dengan gate correctness/kualitas | reproduksi kualitas, byte yang dipindahkan, latency, dan biaya per token berguna secara bersamaan, bukan hanya compression ratio |
| Spekulasi yang mempertimbangkan routing dapat menguntungkan sebelum residensi hampir penuh | MTP dan grammar drafts bekerja, tetapi MTP juga pernah terukur mengalami penurunan 32% di sekitar expert hit 85% | petakan permukaan break-even terhadap acceptance, expert hit rate, batch union, dan draft depth |
| Overlap CPU/GPU dapat menyembunyikan transfer dan sinkronisasi alih-alih sekadar memindahkan bottleneck | peningkatan dari CUDA, Metal, dan Vulkan sudah ada, tetapi CPU cepat, GPU terintegrasi, dan residensi rendah dapat menghapusnya | profil per-stage dan A/B satu variabel pada PCIe, unified-memory, dan mesin full-resident, serta angka discrete-GPU pertama untuk Vulkan tier dan chain |

Ingin membantu? Pilih satu baris dan publikasikan juga hasil negatif. Catat
hardware, commit, model, perintah persis, prompt, state cache, throughput, time
to first token, expert hit rate, byte yang dibaca, dan pemeriksaan kualitas;
ubah satu variabel, ulangi, lalu lampirkan raw log. Mulailah dengan
[CONTRIBUTING.md](CONTRIBUTING.md) dan
[protokol benchmark](docs/benchmarking.md), lalu
[buka issue](https://github.com/JustVugg/colibri/issues/new). Kegagalan yang
dikontrol dengan baik lebih bernilai di sini daripada angka cepat tanpa
penjelasan.

## Dokumentasi

| topik | dokumen |
|---|---|
| Penyiapan satu langkah dan instalasi manual, untuk setiap platform | [quickstart.md](docs/quickstart.md) |
| Penyiapan melalui asisten AI, dan server MCP | [AI_SETUP.md](docs/AI_SETUP.md), [MCP_SERVER.md](docs/MCP_SERVER.md) |
| API: OpenAI, Anthropic, tools, KV slots, dashboard | [api.md](docs/api.md) |
| System One dan model keputusan | [systemone.md](docs/systemone.md), [laya.md](docs/laya.md), [gliner_decide.md](docs/gliner_decide.md), [clef.md](docs/clef.md) |
| Vulkan: expert tier, dense chain, kartu tanpa Resizable BAR | [vulkan.md](docs/vulkan.md) |
| CUDA, dan CUDA tier Qwen3.6 | [cuda.md](docs/cuda.md), [qwen36-cuda-tier.md](docs/qwen36-cuda-tier.md) |
| Apple Silicon, Windows | [metal.md](docs/metal.md), [windows.md](docs/windows.md) |
| Tuning, learning cache, prefetch, speculation | [tuning.md](docs/tuning.md) |
| Beberapa SSD, beberapa mesin | [multidisk.md](docs/multidisk.md), [cluster.md](docs/cluster.md) |
| Benchmark dan cara melakukan pengukuran | [benchmarks.md](docs/benchmarks.md), [benchmarking.md](docs/benchmarking.md) |
| Setiap opsi dan environment variable | [SETTINGS.md](docs/SETTINGS.md), [ENVIRONMENT.md](docs/ENVIRONMENT.md) |
| Draft yang dipaksa grammar, dan ABI embedding eksperimental | [grammar-draft.md](docs/grammar-draft.md), [segment-runtime.md](docs/segment-runtime.md), [edge-runtime.md](docs/edge-runtime.md) |

## Struktur repository

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

**Satu `.c` per keluarga model, menggunakan header tunggal bersama.** Sebuah
engine hanya memiliki arsitekturnya sendiri; apa pun yang dibutuhkan oleh dua
engine ditempatkan dalam header yang sama-sama mereka include, sehingga satu
perbaikan langsung menjangkau semuanya. Dari root repository, `make`,
`make check`, dan `make clean` meneruskan pekerjaan ke Makefile engine.

## Mendukung proyek

colibri bermula sebagai proyek satu orang pada laptop 12-core dengan RAM 25 GB;
sekarang angka-angkanya berasal dari komunitas dengan mesin nyata. Jika proyek
ini berguna bagi Anda:

- beri star pada repository dan bagikan;
- buka issue dengan angka benchmark dari hardware Anda: datapoint lebih
  mendorong proyek ini daripada hal lainnya;
- bergabunglah dengan [komunitas Discord](https://discord.gg/RXV83nSZdk) untuk
  membahas eksperimen, hasil hardware, dan arah riset;
- hubungi melalui GitHub issues untuk mensponsori pengembangan atau
  menyumbangkan hardware.

## Mengapa "colibrì"

Burung kolibri hanya berbobot beberapa gram, dapat melayang di tempat, dan
mengunjungi seribu bunga dalam sehari. Engine ini menjaga raksasa dengan 744
miliar parameter tetap hidup dengan jatah seekor kolibri: RAM 25 GB, dua belas
core CPU, dan banyak kesabaran menghadapi disk.

## Ucapan terima kasih

colibri adalah sebuah engine; kecerdasan yang dijalankannya merupakan hasil
karya banyak pihak. Terima kasih kepada tim yang merilis bobot model mereka
secara terbuka: **Z.ai** (GLM), **Moonshot AI** (Kimi), **Alibaba Qwen**,
**DeepSeek**, **Xiaomi** (MiMo), **Thinking Machines** (Inkling),
**Allen AI** (OLMoE), **Convai Innovations** (Laya), **fastino**
(GLiNER2.5-Decide), dan **Cloudflare** (Clef); kepada mereka yang menerbitkan
container hasil konversi; serta kepada setiap kontributor yang melakukan
benchmark, bisect, mereplikasi proses atlas, atau mengirim patch. Kode pihak
ketiga dalam repository ini beserta lisensinya:
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Eksperimen penempatan pakar, kompresi, dan routing dalam proyek ini juga
dibangun di atas gagasan dan bukti dari riset terbuka serta karya sistem berikut:

- [REAP](https://github.com/CerebrasResearch/reap) dan
  [EASY-EP](https://github.com/RUCAIBox/EASYEP) untuk tingkat kepentingan pakar
  yang mempertimbangkan output dan domain.
- [SERE](https://github.com/JL-Cheng/SERE) untuk rerouting pakar berbasis
  kemiripan, dan [ReMoE](https://github.com/BUAA-OSCAR/ReMoE) untuk fine-tuning
  router yang mempertimbangkan cache locality.
- [MC-SMoE](https://github.com/UNITES-Lab/MC-SMoE) untuk penggabungan dan
  kompresi pakar yang dipandu routing.
- [MoBE](https://github.com/inclusionAI/MoBE) dan
  [D²-MoE](https://github.com/lliai/D2MoE) untuk basis pakar bersama dan delta
  pakar low-rank.
- [HybriMoE](https://github.com/PKU-SEC-Lab/HybriMoE) untuk penjadwalan pakar
  hybrid CPU/GPU, [ScMoE](https://arxiv.org/abs/2404.05019) untuk
  menumpangtindihkan komunikasi pakar dengan komputasi, dan
  [OD-MoE](https://arxiv.org/abs/2512.03927) untuk pemuatan pakar on-demand
  terdistribusi.
- [vLLM](https://github.com/vllm-project/vllm),
  [llama.cpp](https://github.com/ggml-org/llama.cpp), dan
  [kTransformers](https://github.com/kvcache-ai/ktransformers) untuk sistem
  inference terbuka dan karya expert-offload yang membuat perbandingan dapat
  direproduksi.

Engine ini juga berdiri di atas karya engineering konkret, bukan hanya gagasan.
Setiap hal berikut digunakan atau diimplementasikan ulang dalam tree saat ini:

- [safetensors](https://github.com/huggingface/safetensors): container yang
  dibaca setiap engine (`c/st.h`), termasuk dtype fp8 dan I64.
- [tiktoken](https://github.com/openai/tiktoken): `c/tok.h` mengimplementasikan
  ulang `byte_pair_encode` secara persis, dengan menggabungkan pasangan
  bersebelahan yang hasil konkatenasinya memiliki vocab id terendah, sehingga
  vocabulary turunan tiktoken tidak memerlukan daftar merges.
- [llama.cpp](https://github.com/ggml-org/llama.cpp): subset grammar GBNF dalam
  `c/grammar.h` mengikuti sintaks dan set-of-stacks PDA miliknya, dan jalur
  Metal meminjam teknik residensi `newBufferWithBytesNoCopy`.
- [vLLM](https://github.com/vllm-project/vllm): referensi untuk semantik output
  yang dicocokkan engine posisi demi posisi (misalnya posisi final norm relatif
  terhadap LM head).
- [transformers](https://github.com/huggingface/transformers): oracle:
  CI mereproduksi model random-init token demi token terhadapnya.
- [DietGPU](https://github.com/facebookresearch/dietgpu): codec ANS GPU di balik
  experimental compressed expert tier (`COLI_ANS`).
- [rocWMMA](https://github.com/ROCm/rocWMMA): backend HIP memetakan API fragment/
  mma_sync `nvcuda::wmma` milik CUDA ke dalamnya
  (`c/backend_gpu_compat.h`), yang memungkinkan satu source .cu dikompilasi
  untuk kedua vendor.

## Lisensi

Apache 2.0, Copyright 2026 Vincenzo Fornaro. Lihat [LICENSE](LICENSE) dan
[NOTICE](NOTICE). Setiap model tetap menggunakan lisensi yang diberikan oleh
pembuatnya (bobot GLM-5.2 dirilis oleh Z.ai di bawah MIT; Qwen-Image-2.1 hanya
untuk penggunaan nonkomersial).
