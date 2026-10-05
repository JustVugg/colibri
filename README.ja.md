<p align="center">
  <img src="assets/colibri-logo.svg" width="560" alt="colibrì：小さなエンジン、巨大なモデル">
</p>

<p align="center">
  <a href="https://justvugg.github.io/colibri"><img src="https://img.shields.io/badge/website-justvugg.github.io%2Fcolibri-1f6feb" alt="ウェブサイト"></a>
  <a href="https://github.com/JustVugg/colibri/releases"><img src="https://img.shields.io/github/v/release/JustVugg/colibri?color=2ea043" alt="最新リリース"></a>
</p>

<p align="center">
  <a href="https://justvugg.github.io/colibri"><b>Website</b></a> ·
  <a href="https://discord.gg/RXV83nSZdk"><b>Discord</b></a> ·
  <a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.zh-TW.md">繁體中文</a> · <a href="README.it.md">Italiano</a> · 日本語
</p>

**小さなエンジン、巨大なモデル。** colibri は、とても大きなオープンモデルを、あなたがすでに持っているマシンで動かします。数千億パラメータの Mixture-of-Experts モデルは、1 トークンごとに自分自身のごく一部しか使いません。そこで colibri はその部分を RAM に置き、残りの部分、つまりエキスパートは、モデルが必要としたときにディスクから読み込みます。純粋な C で書かれ、モデルファミリーごとにファイルは 1 つ、GPU は不要です。

現在動作するエンジンは 13 個です。そのうち 10 個は言語モデル用です：**GLM-5.2/5.3**、**GLM-5.3-Flash**、**Inkling**、**Kimi K3**、**DeepSeek V4 Flash**、**DeepSeek V4.1 Flash**、**MiMo-V2.6 Flash**（と Pro）、**Qwen3.8-Flash-Next**、**Qwen3.6**（Qwen3-Coder と密モデルの Qwen3.8-27B も動かします）、そして **OLMoE**。1 個は画像を描きます：**Qwen-Image-2.1**。2 個は判定に答えます：**Laya** と **GLiNER2.5-Decide** で、さらに 3 つ目の判定モデル **Clef** が Qwen3.6 エンジンで動きます。[自分のマシンに合うのはどれか](#which-model-for-my-machine)

```
$ ./coli chat
  colibri v1.12.1 · GLM-5.2 · 744B MoE · int4 · streaming CPU
  ✓ ready in 32s · resident 9.9 GB
  › ciao!
  ◆ Ciao! Come posso aiutarti oggi?
```

<a id="get-started"></a>
<a id="the-one-step-way"></a>

## ワンステップで始める

必要なのは、最低でも **8 GB の RAM**（16 GB 以上あればなお良い）、最も小さなモデルのための **22 GB のディスク空き容量**、そしてインターネット接続のあるコンピューターです。グラフィックカードはあってもなくてもかまいません。

**Windows**

1. このページで **Code** をクリックし、次に **Download ZIP** をクリックして、ZIP を展開します。
2. 展開したフォルダーの中の **`START-HERE.bat`** をダブルクリックします。Python が入っていない場合は、その場でインストールすることを提案してくれます。

**Linux**（Ubuntu と Debian の場合。ほかのディストリビューションにも、同じパッケージがそれぞれの名前で用意されています）

```bash
sudo apt install git python3 build-essential
git clone https://github.com/JustVugg/colibri
cd colibri
./start-here.sh
```

**macOS**（[Homebrew](https://brew.sh) を使用）

```bash
xcode-select --install
brew install libomp git python
git clone https://github.com/JustVugg/colibri
cd colibri
./start-here.sh
```

聞かれる質問は「どのモデルにするか」の 1 つだけで、Enter を押せば推奨のモデルが選ばれます。そのあとセットアップは次のことを行います：

1. **マシンを調べます**：RAM、ディスクの空き容量、CPU、GPU。
2. **収まるモデルを推奨します**：モデルのうち常に RAM に置かれる部分と、最小限のエキスパートキャッシュが RAM に収まり、ダウンロードしたファイルがディスクに収まる必要があります。
3. **エンジンを用意します**：コンパイラがあればこのマシン向けにビルドし、なければ CPU で動くビルド済みのものをダウンロードします。GPU 向けのビルドは、それで速くなる場合に行います：Linux の NVIDIA カードで CUDA ツールキットがインストールされていれば CUDA、それ以外は Vulkan です。ディスクリート GPU では常に GPU 向けにビルドします。CPU と RAM を共有する内蔵 GPU では、そこで速くなることが計測されたモデル（Qwen3.6、Qwen3-Coder、Qwen3.8-Flash-Next）の場合だけです。足りないパッケージがある場合は、それをインストールするための正確なコマンドを表示して、CPU で続行します。あとでセットアップをもう一度実行すれば、GPU 向けにビルドし直します。
4. **モデルをダウンロードします**。進捗を表示し、途中から再開できます：いつ止めてもかまいません。もう一度実行すれば、止まったところから続きます。
5. **colibri を起動し、ブラウザでダッシュボードを開きます**。また、ほかのアプリが使えるアドレスを表示します：

```
Starting colibri
  Browser:             http://127.0.0.1:8000/
  OpenAI base URL:     http://127.0.0.1:8000/v1
  Anthropic base URL:  http://127.0.0.1:8000
  stop: press Ctrl+C here (or close this window)
```

**次回からは**、`START-HERE.bat` か `./start-here.sh` をもう一度実行するだけです：ダウンロードもビルドもなしに、colibri がすぐに起動します。`c/coli status` は何がインストールされていて、動いているかどうかを表示し、`c/coli stop` で停止します（Windows では `c\coli.cmd status` と `c\coli.cmd stop`）。

オプションは `./start-here.sh` または `START-HERE.bat` のあとに付けます：

| オプション | 何をするか |
|---|---|
| `--list` | このマシンに対する全モデルの一覧と、収まらないモデルについてはその理由 |
| `--model ID` | そのモデルをインストールする（ID は[下の表](#which-model-for-my-machine)にあります） |
| `--yes` | 質問なしで推奨を選ぶ |
| `--dir DIR` | モデルを別のディスクに置く（デフォルトは `~/colibri-models`） |
| `--backend vulkan`、`cuda` または `cpu` | エンジンのビルドを自分で選ぶ。`--no-gpu` は `--backend cpu` と同じ |
| `--model-dir DIR` | すでにダウンロードしたモデルを使う |
| `--reconfigure` | 別のモデルを選び直す |

各ステップの詳しい内容：[docs/quickstart.md](docs/quickstart.md#the-one-step-way)。

### うまくいかないとき

| 表示された内容 | 対処 |
|---|---|
| ダウンロード中に止まった | 同じコマンドをもう一度実行する：ディスクにすでにあるバイトから再開します |
| `to use the GPU through ..., first run: <command>` | そのコマンドを実行してから、もう一度セットアップを実行する：GPU 向けにビルドし直し、ダウンロードはやり直しません |
| `the ... build failed`。たとえば、インストールされている CUDA ツールキットがそのカードをもうサポートしていないときの `Unsupported gpu architecture` | セットアップはまずツールキットをカードと照らし合わせてチェックし、自分で Vulkan を選びます（理由も表示します）。それでもビルドが失敗する場合は次の選択肢（Vulkan、その次に CPU）を提示します。`./start-here.sh --backend vulkan` は Vulkan を強制します。`--no-gpu` なら CPU のままです |
| `needs N GB free for the download` | `--dir` で、より大きなディスク上のフォルダーを指定する |
| WSL で、モデルのフォルダーが `/mnt/c` の下にある | Linux 側のディスク（デフォルトの `~/colibri-models`）に置く：`/mnt/c` は何倍も遅くなります |
| チェックアウトを更新した（`git pull`） | `./start-here.sh` をもう一度実行してください。ソースが変わっていればエンジンを再ビルドしてから起動します |
| それ以外 | `c/coli logs -n 50` でバックグラウンドで起動したサーバーのログを（フォアグラウンドで起動したサーバーは自分のターミナルに出力します）、`c/coli logs --install` でセットアップのログを表示できます。セットアップが最後に表示した数行を添えて [issue](https://github.com/JustVugg/colibri/issues) を作成してください |

### AI アシスタントにセットアップしてもらう

AI コーディングアシスタントを使っているなら、これをすべて任せられます。次のように頼んでください：

> https://github.com/JustVugg/colibri の docs/AI_SETUP.md に従って、このマシンに colibri をセットアップしてください

[docs/AI_SETUP.md](docs/AI_SETUP.md) は、各ステップを機械可読な結果を返すコマンドとしてアシスタントに渡し、モデルをダウンロードしたりシステムパッケージをインストールしたりする前に、あなたに確認するよう指示しています。Model Context Protocol に対応したアシスタントは、代わりに colibri の MCP サーバーを使えます：`coli mcp` は、ハードウェアの検出、モデルの推奨、インストール、起動、停止、状態確認のためのツールを提供します（[docs/MCP_SERVER.md](docs/MCP_SERVER.md)）。

### または手動で

各ステップを自分で選びたい場合（ビルド済みリリースかソースからのビルドか、下の表のどのモデルか、そして `coli chat`、`coli web`、`coli serve` のどれか）は、[手動でのインストール](#install-by-hand)を、全プラットフォームの手順を一つずつ追いたい場合は[クイックスタートガイド](docs/quickstart.md)を参照してください。

## colibri とは何か、そしてなぜ作るのか

Mixture-of-Experts モデルは、ディスク上では巨大でも、1 トークンあたりでは小さなものです。GLM-5.2 は 744B のパラメータを持ち、1 トークンごとに使うのは約 40B、そのうちトークンごとに入れ替わるのは約 11 GB だけです：ルーティングされるエキスパートです。

<p align="center">
  <img src="docs/media/sparse.png" width="880" alt="1 トークンあたり活性化するのはパラメータの約 5.4% のみ">
</p>

つまり、モデルは高速メモリに収まる必要はなく、**配置** されればよいのです。密な部分（アテンション、共有エキスパート、埋め込み）は RAM に置かれます。ルーティングされるエキスパートはディスクに置かれ、ルーターが必要としたときに読み込まれます。読み込みは、あなたの作業がどのエキスパートを使うかを学習するキャッシュを通して行われます。GPU があれば、最もホットなエキスパートと密レイヤーを GPU が保持します。重みがどこにあるかで変わるのは答えが返ってくる速さであり、答えを生み出す重みやルーターの判断は変わりません。

なぜ作るのか：このサイズのモデルを、人々がすでに持っているハードウェアで動かすため。モデルが動く様子を眺めるため（ダッシュボードは、発火するエキスパートをすべて表示します）。そして、誰もが計測して速くできるほど、エンジンを小さく保つためです。colibri はオープンな研究プラットフォームでもあります：最適化は、再現可能なエンドツーエンドの計測によって採用に値することを示さなければならず、デフォルトのポリシーがモデルの精度やルーターのセマンティクスを黙って変えることは決してありません。高速メモリが少なければ速度は落ちるかもしれませんが、それによってモデルが密かに別物になってはなりません。詳しくは[仕組み](#how-it-works)を参照してください。

<a id="which-model-for-my-machine"></a>

## 自分のマシンに合うモデル

セットアップは、あなたのマシンで RAM から動かせるモデルのうち最も高性能なものを推奨し、そのすぐ下に、ディスクからストリーミングする、より大きなモデルを並べます。`./start-here.sh --list` を実行すると、すべてのモデルをこのマシンと照らし合わせて表示します。以下の表はセットアップ自身のカタログ（[`c/setup_catalog.py`](c/setup_catalog.py)）に沿っています。ダウンロードのサイズは、Hugging Face が各リポジトリについて表示しているサイズです。

- **RAM** は 2 つの数値です：1 つ目を下回るとモデルは起動せず、2 つ目以上あれば計測値どおりに動きます。
- **GPU** は、セットアップがそのエンジン向けにビルドできるものです（[GPU](#gpus)）。*内蔵 GPU も* とあるものは内蔵 GPU も使います。これらのエンジンは内蔵 GPU で速くなることが計測されています。それ以外のエンジンはディスクリート GPU だけを使います。
- **計測値** は、記載されたマシンで計時した結果で、チャットモデルではデコード速度です。アルファベットは表の下に挙げたマシンを指します。空欄は、まだ誰も計測していないという意味です。

**小さなモデル：RAM から動作します**

| モデル | `--model` | ダウンロード | RAM | GPU | 計測値 |
|---|---|---|---|---|---|
| **Qwen3.6-35B-A3B**：思考モードとツールに対応したチャット | `qwen36-35b` | 23 GB | 10 / 20 GB | CUDA、Vulkan（内蔵 GPU も） | CPU で 6.0 tok/s、内蔵 GPU の Vulkan で 9.9（A）。CUDA で 30.0（C） |
| Qwen3-Coder-30B-A3B：コードとツール呼び出し、思考モードなし | `qwen3-coder-30b` | 19 GB | 8 / 18 GB | CUDA、Vulkan（内蔵 GPU も） | 全エキスパートを RAM に置いて 8.5-9.6 tok/s、レイヤーあたり 32 個で 5.1（A、CPU） |
| **Qwen-Image-2.1**：テキストから画像、非商用ライセンス | `qwen-image-2.1` | 33 GB | 12 / 18 GB | Vulkan | 768x512 の画像 1 枚に 2 分 40 秒（Zen 4 の 8 コア、CPU） |

**大きなモデル：エキスパートをディスクからストリーミングします**（速度を決めるのはディスクで、高速な NVMe ドライブが最も効果的です）

| モデル | `--model` | ダウンロード | RAM | GPU | 計測値 |
|---|---|---|---|---|---|
| DeepSeek V4 Flash REAP 150B：256 個中 132 個のエキスパート | `deepseek-v4-flash-reap` | 85 GB | 16 / 32 GB | CUDA、Vulkan | |
| **DeepSeek V4 Flash**（284B）：ツール対応 | `deepseek-v4-flash` | 167 GB | 16 / 32 GB | CUDA、Vulkan | CPU のみで、32 GB（Ryzen 7 5800X）なら 0.93 tok/s、63 GB（Ryzen 9 5950X）なら 1.24。CUDA で 1.5-1.6（RTX 5080、32 GB、NVMe 2 台） |
| **MiMo-V2.6 Flash**（309B）：ビジョンとツール | `mimo-v2.6-flash` | 172 GB | 32 / 52 GB | Vulkan | 2.34-3.37 tok/s（A、CPU） |
| **Qwen3.8-Flash-Next**（125B + 51B n-gram）：ビジョンとツール | `qwen38-flash-next` | 186 GB | 24 / 32 GB | CUDA、Vulkan（内蔵 GPU も） | レイヤーあたり 32-96 エキスパートで 1.91-2.56 tok/s。オプションの int4 エキスパートで 3.99（A、CPU） |
| **GLM-5.2**（744B）：リファレンスモデル、MTP ヘッド付き | `glm-5.2` | 429 GB | 16 / 24 GB | CUDA、Vulkan | 25 GB のラップトップでコールド時 0.05-0.1 tok/s。128 GB の Ryzen AI Max+ 395 で 1.83。6x RTX 5090 で 9.0-9.2 |
| GLM-5.3（744B）：同じエンジン、MTP ヘッドなし | `glm-5.3` | 419 GB | 16 / 24 GB | CUDA、Vulkan | |
| **Inkling**（975B）：int4 エキスパート、bf16 の密な重み | `inkling` | 514 GB | ダウンロードしたままなら 120 / 128 GB。[密な重みの変換](docs/inkling.md)後は 25 GB | CUDA、Vulkan | 0.25 tok/s（Ryzen 9 7900、187 GB、RTX A6000） |
| MiMo-V2.6 Pro（1.02T）：ビジョンとツール | `mimo-v2.6-pro` | 564 GB | 54 / 64 GB | Vulkan | 0.66-0.79 tok/s（A、CPU） |
| **Kimi K3**（2.8T）：最大のモデル | `kimi-k3` | 1.56 TB | 32 / 64 GB | CUDA、Vulkan | 1 トークンあたり約 9.4 秒、エキスパートの読み込みは 6.3 GB/s |

<a id="other-supported-models"></a>

**手動：ダウンロード後に変換または準備の手順が必要なもの**

| モデル | ダウンロード、その後のディスク上のサイズ | RAM | GPU | 計測値 |
|---|---|---|---|---|
| **OLMoE**（7B）：小さく、ツールの使い方を覚えるのに向く | 14 GB、int8 への変換後は 7 GB | 8 GB | Vulkan | 22-23 tok/s（A、CPU） |
| Qwen3.8-27B（密モデル）：テキストと画像 | 56 GB、変換後は 51 GB | int4 で 20 GB、int8 で 30 GB | Vulkan | int4 で 3.45 tok/s、int8 で 2.1（16 スレッドの CPU サーバー） |
| **GLM-5.3-Flash**（321B）：ビジョンとツール | 328 GB、シャードごとに変換して 195 GB | 25 GB | CUDA、Vulkan | 1 トークンあたりウォーム時約 20 秒、コールド時 44 秒（6 コア、25 GB、普通のディスク） |
| **DeepSeek V4.1 Flash**（552B）：ビジョンとツール、変換は不要だが一度だけ準備が必要 | 510 GB | 約 18 GB とエキスパートキャッシュ（レイヤーあたり 8 個でピーク 24.8 GB） | Vulkan | 0.21-0.24 tok/s（エキスパートの 68% を保持する 16 スレッドの CPU サーバー） |

**判定モデル**（[System One](#system-one-a-decision-with-a-probability) の質問に答えるモデルで、チャットはしません）

| モデル | ダウンロード、その後のディスク上のサイズ | RAM | GPU | 計測値 |
|---|---|---|---|---|
| **Laya**（Convai Innovations）、英語 | 0.85 GB | 1.7 GB | CPU | 質問 1 つで 219 ms、3 つで 882 ms（B） |
| **GLiNER2.5-Decide**（fastino）、英語 | 1.95 GB | 1.9 GB | CPU | 質問 1 つで 294 ms、3 つで 897 ms（B、負荷あり） |
| **Clef**（Cloudflare）：判定ヘッド付きの Qwen3.8-27B、チャットも可能 | 55 GB、変換後は 52 GB | int4 の 19 GB から f16 の 55 GB まで | CPU | int8 で 1 リクエストあたり 20.4 秒（A） |

マシン：
**A** Ryzen 7 PRO 8700GE のデスクトップ（8 コア、61-64 GB DDR5、NVMe、内蔵 Radeon 780M）。
**B** i7-1355U のラップトップ。
**C** Threadripper 3945WX のマシンに載せた RTX 3070 8 GB。密レイヤーと DeltaNet レイヤーはカード上に置いています（行単位 int4 コンテナ）。
各数値は、[docs/](docs/) にある各モデルのページか[ベンチマーク表](docs/benchmarks.md)に、正確な設定とともに記載されています。

ファミリーごとにページがあります：[qwen36.md](docs/qwen36.md)（Qwen3.6、Qwen3-Coder、Qwen3.8-27B）、[qwen38.md](docs/qwen38.md)、[deepseek-v4.md](docs/deepseek-v4.md)、[deepseek-v41.md](docs/deepseek-v41.md)、[mimo.md](docs/mimo.md)、[glm53-flash.md](docs/glm53-flash.md)、[inkling.md](docs/inkling.md)、[kimi_k3.md](docs/kimi_k3.md)、[qwen-image.md](docs/qwen-image.md)、[laya.md](docs/laya.md)、[gliner_decide.md](docs/gliner_decide.md)、[clef.md](docs/clef.md)、そして GLM-5.2 については[クイックスタート](docs/quickstart.md#3-get-the-model)にあります。対応モデルと同じアーキテクチャのチェックポイントは、Qwen3.6 エンジン上の KAT-Coder v2.5 のように、変更なしでそのまま動きます。

<a id="gpus"></a>

## GPU

### GPU は不要

どのエンジンも、ほかに何もインストールせずに CPU で動きます。GPU は重みを置くためのより速い場所であって、必須ではありません：大きなモデルでは速度を決めるのはディスク、小さなモデルでは RAM です。

### Vulkan：あらゆる GPU

すべての MoE エンジンは、Vulkan 1.2 ドライバを持つあらゆる GPU（AMD、Intel、NVIDIA、内蔵でもディスクリートでも）を、2 通りの方法で使えます：

- **エキスパートティア**：GPU メモリ上の、ルーティングされるエキスパートのキャッシュです。起動時に過去の会話で使われたエキスパートで埋められ、チャット中にも適応していきます。GPU は自分が保持するエキスパートを計算し、その間に CPU が残りを計算します。
- **密チェーン**：レイヤー 1 つ全体を 1 回の GPU サブミッションとして記録し、モデルの実行中の状態をレイヤーからレイヤーへと GPU 上に保持したままにします。

マシン A の内蔵 Radeon 780M で計測しました。各実行の前にモデルファイルをページキャッシュから追い出し、100 トークンをデコードしています（[vulkan.md](docs/vulkan.md#the-chain-on-a-radeon-780m)）：

| | CPU | Vulkan、エキスパートティア | Vulkan、ティアと密チェーン |
|---|---|---|---|
| Qwen3.6-35B-A3B、デコード | 6.0 tok/s | 8.0 tok/s | **9.9 tok/s** |
| Qwen3.6-35B-A3B、512 トークンのプロンプト | 35.7 秒 | 12.2 秒 | **9.5 秒** |
| Qwen3.8-Flash-Next（int4 エキスパート）、デコード | 3.5 tok/s | **3.8 tok/s** | 3.2 tok/s |
| Qwen3.8-Flash-Next、512 トークンのプロンプト | 43.6 秒 | 38.7 秒 | **30.1 秒** |
| OLMoE、デコード（ウォーム） | **23.1 tok/s** | 12.8 tok/s | 17.3 tok/s |

内蔵 GPU は CPU と RAM を共有しています。内蔵 GPU が節約するのは、保持しているエキスパートの計算とディスク読み込みです。そのため Qwen3.6 のようなモデルでは効果がありますが、OLMoE のようにエキスパートがすでに RAM にある小さなモデルでは、かえって遅くなることがあります。だからセットアップは、内蔵 GPU では Qwen3.6、Qwen3-Coder、Qwen3.8-Flash-Next の場合にだけ Vulkan を有効にし、各エンジンもそこで密チェーンを動かすかどうかを自分で判断します（Qwen3.6 は動かし、Qwen3.8 は動かしません）。それでも Vulkan を使いたい場合は `--backend vulkan` を指定します。

ディスクリート GPU では、セットアップはすべてのエンジンを Vulkan 向けにビルドし（エンジンに CUDA の経路があり、ツールキットがインストールされている場合は CUDA が優先）、密レイヤーをカード上に置きます。これこそこの設計が想定しているケースです。**私たちはまだディスクリート GPU 自体を計測していません。** 最初の数値はあるユーザーから届いたもので、Tesla V100 16 GB 上の Qwen3.6 がエキスパートティアと密チェーンを使って 17 から 19 tok/s でした（[#1852](https://github.com/JustVugg/colibri/issues/1852)）。（これらより前の、GLM-5.2 の以前の Vulkan 経路は、ディスクリートの RX 9070 で 1.7-1.8 tok/s でデコードしました。）あなたのカードでの数値を歓迎します。

**Resizable BAR のないカード** でも動くようになりました。そうしたカード（すべての Turing カード、発売時のファームウェアのままの Ampere カード、このオプションを無効にした古い AMD カード）では、CPU がカードのメモリに直接書き込めるのは約 256 MB だけです。colibri は現在、ステージングバッファを通して重みを自動でコピーします。この経路は、強制的に使わせる方法と、3 つのデバイスで小さなウィンドウをエミュレートする方法でテストされており、780M では計測できるほどのコストはありません。ただし、Resizable BAR のないカードではまだ計測されていません（[vulkan.md](docs/vulkan.md#memory-placement-without-resizable-bar)）。

CI は、ソフトウェアドライバ上で、すべてのエンジンの Vulkan 経路を CPU のトークンと照合しています。GPU は数値を異なる順序で足し合わせ、CPU が丸める一部の活性値を f32 のまま保持するため、長い回答では CPU の回答と 1 単語ずれることがあります（[vulkan.md](docs/vulkan.md#the-other-engines)）。

### CUDA：NVIDIA カード

Linux で CUDA ツールキットがインストールされている場合、セットアップは CUDA の経路を持つエンジンを CUDA 向けにビルドします：GLM-5.2/5.3、GLM-5.3-Flash、Inkling、Kimi K3、DeepSeek V4 Flash、Qwen3.8-Flash-Next、そして Qwen3.6 と Qwen3-Coder です。Windows では CUDA エンジンは別の DLL です（[windows.md](docs/windows.md)）。

- **VRAM エキスパートティア** は、計測されたルーティングから選んだ最もホットなエキスパートをカード上に置きます。ミスしたエキスパートは同時に CPU で計算されます。8 GB のカード 2 枚（RTX 3070 と Quadro RTX 4000）での Qwen3.6 は、履歴が温まった状態で 11.3 tok/s でデコードしました（[qwen36-cuda-tier.md](docs/qwen36-cuda-tier.md)）。全エキスパートを常駐させた RTX 5090 6 枚での GLM-5.2 は 9.0-9.2 tok/s（[benchmarks.md](docs/benchmarks.md)）。RTX 5080 での DeepSeek V4 Flash は 1.5-1.6 tok/s で、3,324 トークンのプロンプトを 90 秒で処理しました（[deepseek-v4.md](docs/deepseek-v4.md)）。
- **Qwen3.6 の新機能：DeltaNet レイヤーをカード上で実行**（`Q36_DN_GPU=1`、オプトイン）。これまで Qwen3.6 の 30 個の DeltaNet レイヤーはそれぞれ、1 トークンごとに 4 回、カードと CPU の間でデータをコピーしていました。現在はデコード時の 1 トークンがレイヤー全体をカード上で実行し、そのリカレント状態は VRAM に保持されます。密レイヤーを VRAM に置いた RTX 3070 で：25.4 から 30.0 tok/s に向上しました（[qwen36-cuda-tier.md](docs/qwen36-cuda-tier.md#the-deltanet-layer-on-the-card-q36_dn_gpu1)）。
- **古いカード。** CUDA ツールキットがもうあなたのカード向けにコンパイルできない場合（[#1852](https://github.com/JustVugg/colibri/issues/1852) の CUDA 13 と V100 の例）、セットアップはビルドの前にそれを見抜き、理由を示してカードを Vulkan 経由で使います。CUDA 12.x ツールキットがあれば CUDA の経路が戻ります。DeepSeek V4 の CUDA ティアは Pascal と Turing 向けにもビルドできます（`CUDA_ARCH=portable-pre-ampere NO_TC=1`）。

すべての詳細：[docs/cuda.md](docs/cuda.md)。

### Apple Silicon

いくつかのエンジンでは、Metal バックエンドがユニファイドメモリ GPU 上でエキスパートの演算を行います。`METAL=1` でビルドしてください（[docs/metal.md](docs/metal.md)）。macOS では、ワンステップのセットアップは CPU 向けにビルドします。

<a id="system-one-mode-ask-a-closed-question"></a>
<a id="system-one-a-decision-with-a-probability"></a>

## System One：確率付きの判定

人がモデルに求めることの多くは、段落ではなく選択です：どのキューか、どの判定か、イエスかノーか。`POST /v1/systemone` は状態（テキストまたは JSON）と型付きの質問を受け取り、それぞれの質問に、許可された各選択肢の確率と確信度 (confidence) を付けて答えます。何も生成しないので、どの回答もリストの外に出ることはなく、「モデルに確信がない」ことが、閾値を設定できる数値になります。

```bash
curl -s http://127.0.0.1:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "340 lines, 8 files, no tests. CI is green but nothing covers that path.",
  "questions": {
    "review": {"type": "choice", "instructions": "What should the reviewer do?",
               "criteria": {"merge": null, "request changes": null, "close": null}},
    "risky":  {"type": "noul", "instructions": "Is this change risky?"}}}'
```

`choice` は、選ばれたラベル、すべてのラベルの確率、そして 0（平坦）から 1（確実）までの `confidence` を返します。`noul` はイエスの確率を、`score` は期待されるレベルを返します。

答えるのは：

- **colibri が動かすすべてのチャットモデル**。スコアリングによって答えます：回答を書く代わりに、各選択肢の確率を読み取ります。1 つの文書についての複数の質問は、文書を 1 回だけ読みます：Qwen3.6 では、1 つの文書についての 4 つの質問が、同じ CPU マシンで同じ回答を生成する場合より 5.7 倍速く返ってきました。
- **3 つの判定モデル**。作者が調整した較正を使い、1 回のフォワードパスでネイティブに答えます：[Laya](docs/laya.md)（Convai Innovations）、[GLiNER2.5-Decide](docs/gliner_decide.md)（fastino）、[Clef](docs/clef.md)（Cloudflare。チャットも可能）。サイズと速度は[判定モデルの表](#which-model-for-my-machine)にあります。

**Jev からの切り替え。** リクエストとレスポンスは TypeSafe の Jev API と同じなので、Jev のクライアントは base URL を変えるだけで、ほかは何も変えずに colibri に切り替えられます：`TYPESAFE_BASE_URL=http://127.0.0.1:8000`（クライアントがすでに送っているキーは、`COLI_API_KEY` なしで起動したサーバーなら受け入れられます）。2 つの公式 SDK は、改変なしのまま `coli serve` に対してテストされています。

同じモードはターミナル（`coli chat` での `/decide merge | request changes | close`）と、ダッシュボードの System One ページにもあります。リクエストとレスポンスの全体、スコアリングのルール、そして役に立たない場面：[docs/systemone.md](docs/systemone.md)。

## ダッシュボード

`coli web` で開きます。ワンステップのセットアップでも開きます：チャット、System One ページ、Brain、Profiling ページがあり、ライトテーマとダークテーマに対応しています。

<p align="center">
  <img src="docs/media/colibri-dashboard.png" width="900" alt="colibri の Web ダッシュボード：チャット、ライブメトリクス、ハードウェアパネル、エキスパートのティア">
</p>
<p align="center"><em>CPU マシン上で、ディスクからストリーミングされるエキスパートを使って応答する Qwen3.6。</em></p>

<p align="center">
  <img src="docs/media/colibri-brio.png" width="900" alt="System One ページ：1 回だけ読まれる文書、許可された各回答の確率、そしてエントロピー">
</p>
<p align="center"><em><strong>System One</strong>：モデルに文書と、選んでよい回答だけを渡します。ここでは
<strong>request changes が 99.9%</strong>、エントロピー 0.005、読み取り 4 トークン、生成 0 トークンです。</em></p>

<p align="center">
  <img src="docs/media/colibri-brain.png" width="900" alt="Brain ページ：皮質として描かれた GLM-5.2 の計測されたエキスパートアトラスと、中に入れる 10 の領域">
</p>
<p align="center"><em><strong>Brain</strong>：GLM-5.2 の<a href="https://github.com/JustVugg/colibri/issues/175">計測されたエキスパートアトラス</a>。
特性が明らかになった 13,260 個のエキスパートが 10 の領域（Python、SQL、数学、詩、法律、中国語...）に分かれ、計測された
ルーティング親和性に従って配置されています。<strong>Live routing</strong> は実行中のモデルを表示します：エキスパートごとに 1 セル、
色はストレージのティアを表し、1 ターンでルーティングされたエキスパートはすべて光ります。</em></p>

**Profiling** ページは、各ターンがどこで時間を使っているかをフェーズごとに示し、直近 30 ターンを推移として表示します。

## ほかのアプリから使う

`coli serve`（セットアップが起動してくれるもの）は、複数の API を備えた 1 つのサーバーです：

- **OpenAI 互換**：`/v1/chat/completions`、`/v1/completions`、`/v1/models`。ストリーミング、JSON での応答、停止シーケンス、logprobs に対応しています。
- **Anthropic 互換**：`/v1/messages`。Claude Code や Anthropic の SDK をそのまま接続できます。
- **ツール呼び出し**：Inkling と OLMoE を除くすべてのチャットエンジンで、それぞれのモデルのネイティブ形式で行います（[エンジンごとの表](docs/api.md#tool-calling-support)）。
- **画像入力**：GLM-5.3-Flash、DeepSeek V4.1 Flash、MiMo-V2.6、Qwen3.8-Flash-Next、Qwen3.8-27B で使えます。`coli chat` のメッセージ内のパス、`coli web` での添付、または `image_url` パートで渡します。
- **画像出力**：Qwen-Image-2.1 で `POST /v1/images/generations` から生成します。`coli chat` ではターミナル内に描画されます（[qwen-image.md](docs/qwen-image.md)）。
- **判定**：`POST /v1/systemone` で行います（[前述](#system-one-a-decision-with-a-probability)）。

コーディング用の CLI やエディタは、ほかの OpenAI 互換プロバイダーと同じように接続できます：base URL は `http://127.0.0.1:8000/v1`、モデル ID は `coli status` が表示するもの、キーは空でなければ何でもかまいません（[docs/api.md](docs/api.md#connect-a-coding-cli-or-editor)）。

<a id="how-it-works"></a>

## 仕組み

<p align="center">
  <img src="docs/media/token-path.png" width="880" alt="ルーティング、和集合、配置、オーバーラップ、学習">
</p>

すべてのトークンのすべてのレイヤーが、同じ 5 つのステップをたどります：ルーティング、和集合、配置、オーバーラップ、学習。設計上の目標は、**配置が決めるのは常に速度だけ** ということです：エキスパートが VRAM から答えても、RAM から答えても、ディスクから答えても、ルーターの判断と重みの精度は同じです。

<p align="center">
  <img src="docs/media/tiers.png" width="880" alt="エキスパート常駐の 3 つのティアとしての VRAM、RAM、NVMe">
</p>

- **重みのための JIT。** コンパイラの JIT はプログラム全体をコンパイルすることはなく、実際に実行される部分を観察して、ホットパスをコンパイルします。colibri は重みについて同じ賭けをします。計測されたルーティングの熱量が、どのエキスパートに VRAM、RAM、ディスクのどれを割り当てるかを決めます：レイヤーごとの LRU キャッシュに加えて、あなた自身の会話から学習したピン留めのホットセット（`.coli_usage`、ターンごとに更新）があります。colibri は使えば使うほど速くなります。これが機能するのは、ルーティングに計測可能な構造があるからです（[エキスパートアトラス](https://github.com/JustVugg/colibri/issues/175)）。
- **ディスクを二度待たない。** エキスパートの 3 つの行列は 1 回の `pread` で読み込まれます。ローダーのプールが、常駐エキスパートが計算している間に、欠けているエキスパートを読み込みます。複数の位置をまとめたバッチでは、各エキスパートを 1 回だけ読みます。ルーター先読みスレッドが次のレイヤーをプリフェッチできます（GLM-5.2 のルーティングは 1 レイヤー先を 71.6% 予測できます）。`DIRECT=1`（O_DIRECT）は高速な NVMe ドライブでは大きな改善になることが多い一方、ほかのドライブでは効果がないか逆効果になります：自分のドライブで計測してください（[tuning.md](docs/tuning.md)）。
- **複数の SSD。** `COLI_MODEL_MIRROR=/second/glm52_i4 ./coli chat --model /fast/glm52_i4` は、2 台目のドライブにあるコピーから読み込みます。独立したコントローラに接続された NVMe ドライブ 2 台で、デコード +37.5% を計測しました。より小さなドライブ上の部分ミラーでも動作します（[multidisk.md](docs/multidisk.md)）。
- **ラップトップからラックまで。** 25 GB のラップトップでは、すべてのエキスパートがディスクからストリーミングされ、遅いながらも正しく動きます。大きなホストではすべてのエキスパートが常駐し（`CUDA_EXPERT_GB=auto PIN_GB=all`）、ディスクはデコードから外れます。`COLI_NUMA=1` はマルチソケットのホストで、常駐する重みを複数のメモリコントローラに分散させます。ローカルクラスタモードは、ルーティングされるエキスパートをほかのマシンで実行します（[cluster.md](docs/cluster.md)）。
- **忠実なモデル。** すべてのエンジンは、CI で小さなフィクスチャを使い、そのモデルのリファレンス実装と照合されています。GLM-5.2 の MLA アテンションは圧縮された KV 状態（1 トークンあたり 32,768 個ではなく 576 個の浮動小数点数で、57 分の 1）を保持し、それは再起動をまたいで残るため、会話はプロンプトを読み直すことなく再開できます。
- **元が取れるときだけの投機的デコード。** GLM-5.2 の int8 MTP ヘッドは、効果がある場合、フォワードあたり 2.2-2.8 トークンをドラフトします。Qwen3.8-Flash-Next の MTP ヘッドはデフォルトで有効で、同じ出力のまま速度を 16-20% 上げます。プロンプトルックアップはコード編集で 6-7% 上げます。ドラフトのコストが節約を上回る場合（DeepSeek V4）はオフのままです（[tuning.md](docs/tuning.md#speculation-and-reproducibility)）。

エンジンは、共有ヘッダの上に、モデルファミリーごとに C ファイルを 1 つ置いた構成です（GLM-5.2 なら `c/colibri.c`）。BLAS は使わず、実行時に Python も使いません：Python が動かすのは、セットアップ、ランチャー、変換ツール、API ゲートウェイだけです。

<a id="what-it-achieves"></a>

## ベンチマーク

<p align="center">
  <img src="docs/media/ladder.png" width="880" alt="ハードウェアクラス別に計測した GLM-5.2 のデコード速度">
</p>

同じエンジン、同じ int4 コンテナです：ハードウェアが変えるのはエキスパートの置き場所だけです。GLM-5.2 のデコード速度を[完全な表](docs/benchmarks.md)から抜粋します：

- **6x RTX 5090、全エキスパート常駐：** 5.8-6.8 tok/s、選択的な NUMA インターリーブで 9.0-9.2（[実験ログ](docs/experiments/glm52-6x5090-2026-07-12.md)）。
- **128 GB、CPU のみ**（Ryzen AI Max+ 395）：ウォーム時 1.83 tok/s（[#200](https://github.com/JustVugg/colibri/issues/200)）。
- **RTX 5070 Ti 1 枚のラップトップ級マシン：** 1.07 tok/s（[#273](https://github.com/JustVugg/colibri/issues/273)）。
- **このプロジェクトが始まった 25 GB のラップトップ：** コールド時 0.05-0.1 tok/s。偽りのない下限です。

品質は仮定ではなく計測されています：int4 コンテナのコストと量子化のアブレーションは [benchmarks.md](docs/benchmarks.md#quality-benchmark) にあります。あなたのマシンを追加するには、[ベンチマークプロトコル](docs/benchmarking.md)に従い、数値を添えて issue を作成してください。

<a id="install-by-hand"></a>

## 手動でのインストール

<a id="1-get-colibri"></a>

**1. プログラム。** [Releases](https://github.com/JustVugg/colibri/releases) から自分のプラットフォーム用のアーカイブ（Linux x86_64、macOS、Windows。コンパイラは不要で、必要なのはランチャーと API のための [Python 3](https://www.python.org/downloads/) だけ）を取得して展開し、`python3 coli info` を実行します。または、`gcc`（または clang）と OpenMP を使ってソースからビルドします：

```bash
git clone https://github.com/JustVugg/colibri && cd colibri/c
./setup.sh                                # checks gcc/OpenMP, builds, self-tests
make qwen36 VK=1                          # one engine, here with Vulkan (CUDA=1 for CUDA)
```

<a id="2-get-the-model"></a>

**2. モデル。** [上の表](#which-model-for-my-machine)にあるどのダウンロードでもかまいません：セットアップの ID は Hugging Face のリポジトリに対応しており、各モデルのページにダウンロードと変換のコマンドがあります。GLM-5.2 には、int8 MTP ヘッド付きのグループスケール（gs64）コンテナ [`mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp`](https://huggingface.co/mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp)（429 GB）を、GLM-5.3 には [`Justvugg/GLM-5.3-colibri-int4-g64`](https://huggingface.co/Justvugg/GLM-5.3-colibri-int4-g64)（419 GB、MTP ヘッドなし）を使ってください。古い行単位 int4 のミラーは使わないでください：品質が約 9 ポイント劣ることが計測されており、[#455](https://github.com/JustVugg/colibri/issues/455) のループする回答の原因でした。`./coli convert --model /nvme/glm52_i4` は、FP8 リリースから同じコンテナをシャードごとに作成するので、756 GB 全体を一度にディスクに置く必要はありません。MTP ヘッドなどの確認方法：[quickstart.md](docs/quickstart.md#3-get-the-model)。

<a id="3-run-it"></a>

**3. 実行する。** ソースのチェックアウトなら `c/` から、リリースなら展開したフォルダーから実行します。ランチャーはモデルの `config.json` を読んでエンジンとそのチャットテンプレートを選ぶので、コマンドはどのモデルでも同じです：

```bash
./coli chat  --model /nvme/qwen36          # chat in the terminal
./coli web   --model /nvme/qwen36          # API + dashboard, opens a browser
./coli serve --model /nvme/qwen36          # API + dashboard, no browser
./coli plan  --model /nvme/qwen36          # where the model will live: VRAM, RAM, disk
./coli doctor --model /nvme/qwen36         # read-only check: is everything ready?
./coli tune  --model /nvme/qwen36          # measure and save this machine's fastest safe settings
```

Windows では、リリースアーカイブに `coli.cmd` が同梱されています（`coli.cmd chat --model D:\qwen36`）。ソースのチェックアウトからは `py -3 c\coli` を使ってください。`.exe` ファイルはエンジンであり、ランチャーではありません。すべてのオプションと環境変数：[SETTINGS.md](docs/SETTINGS.md)、[ENVIRONMENT.md](docs/ENVIRONMENT.md)。

## 研究、そして協力の方法

colibri は、フロンティアモデルが希少なハードウェアに頼る度合いを減らし、より安く動かせるようにしたいと考えています。そのためには、重みの格納方法と移動方法を変え、何を VRAM、RAM、ストレージに置くかを決め、CPU と GPU の処理をオーバーラップさせ、新しいデコード方法を試すことになります。慣習的だからという理由で残されるものはなく、マイクロベンチマークで速く見えるという理由で採用されるものもありません：決め手となるのは実機でのエンドツーエンドの推論であり、速度と並んで品質も計測されます。未解決の問いは次のとおりです：

| 仮説 | これまでのエビデンス | まだ必要な実験 |
|---|---|---|
| ルーティング履歴は単純な LRU よりもうまくエキスパートを配置できる | 学習されたピンは繰り返しのワークロードを改善するが、プロンプトに過学習し得る | コーディング、チャット、多言語、長コンテキストのワークロードにわたる、ホールドアウトかつセッション横断の A/B |
| 複数の SSD は独立した帯域幅をデコード速度に変えられる | 独立した NVMe ドライブ 2 台でデコード +37.5% を計測。より遅い 3 台目のドライブは、重み付きストライピングの後では効果が中立だった（[計測結果](docs/multidisk.md#what-has-been-measured)） | ドライブ速度、コントローラ構成、キャッシュ状態を変えて再現する |
| ハードウェアを考慮したプランナーは、各マシンの最適構成に自動で近づける | RAM/VRAM の予算といくつかのバックエンドは現在すでに検出され、セットアップがビルドを選ぶ | 生成されたプランを、ラップトップ、ワークステーション、NUMA ホスト、マルチ GPU システムにわたる制御されたパラメータスイープと比較する |
| ロスレスまたは品質上限付きの表現で、重みの移動を意味のあるほど減らせる | 正しさ/品質ゲート付きのフォーマットと量子化のアブレーションが存在する | 圧縮率だけでなく、品質、移動バイト数、レイテンシ、有用トークンあたりのコストを同時に再現する |
| ルーティングを考慮した投機的デコードは、ほぼ完全常駐に達する前でも元が取れる | MTP と文法ドラフトは動作するが、MTP はエキスパートヒット率約 85% 付近で 32% の損失も計測されている | 受理率、エキスパートヒット率、バッチの和集合、ドラフト深さにわたる損益分岐面をマッピングする |
| CPU/GPU のオーバーラップは、ボトルネックを移すだけでなく転送と同期を隠蔽できる | CUDA、Metal、Vulkan での改善はあるが、高速な CPU、内蔵 GPU、低い常駐率ではそれが打ち消され得る | PCIe、ユニファイドメモリ、完全常駐のマシンにわたる、ステージごとのプロファイルと 1 変数ずつの A/B、そして Vulkan のティアとチェーンについての最初のディスクリート GPU での数値 |

協力したいですか？いずれかの行を選び、ネガティブな結果も公開してください。ハードウェア、コミット、モデル、正確なコマンド、プロンプト、キャッシュ状態、スループット、最初のトークンまでの時間、エキスパートヒット率、読み込みバイト数、品質チェックを記録し、1 つの変数だけを変えて繰り返し、生のログを添付してください。まずは [CONTRIBUTING.md](CONTRIBUTING.md) と[ベンチマークプロトコル](docs/benchmarking.md)から始め、それから [issue を作成](https://github.com/JustVugg/colibri/issues/new)してください。ここでは、説明のつかない速い数値よりも、よく制御された失敗のほうが価値があります。

## ドキュメント

| トピック | ドキュメント |
|---|---|
| ワンステップのセットアップと手動インストール、全プラットフォーム | [quickstart.md](docs/quickstart.md) |
| AI アシスタントによるセットアップと MCP サーバー | [AI_SETUP.md](docs/AI_SETUP.md)、[MCP_SERVER.md](docs/MCP_SERVER.md) |
| API：OpenAI、Anthropic、ツール、KV スロット、ダッシュボード | [api.md](docs/api.md) |
| System One と判定モデル | [systemone.md](docs/systemone.md)、[laya.md](docs/laya.md)、[gliner_decide.md](docs/gliner_decide.md)、[clef.md](docs/clef.md) |
| Vulkan：エキスパートティア、密チェーン、Resizable BAR のないカード | [vulkan.md](docs/vulkan.md) |
| CUDA と、Qwen3.6 の CUDA ティア | [cuda.md](docs/cuda.md)、[qwen36-cuda-tier.md](docs/qwen36-cuda-tier.md) |
| Apple Silicon、Windows | [metal.md](docs/metal.md)、[windows.md](docs/windows.md) |
| チューニング、学習キャッシュ、プリフェッチ、投機的デコード | [tuning.md](docs/tuning.md) |
| 複数の SSD、複数のマシン | [multidisk.md](docs/multidisk.md)、[cluster.md](docs/cluster.md) |
| ベンチマークと計測方法 | [benchmarks.md](docs/benchmarks.md)、[benchmarking.md](docs/benchmarking.md) |
| すべてのオプションと環境変数 | [SETTINGS.md](docs/SETTINGS.md)、[ENVIRONMENT.md](docs/ENVIRONMENT.md) |
| 文法強制ドラフトと、実験的な組み込み用 ABI | [grammar-draft.md](docs/grammar-draft.md)、[segment-runtime.md](docs/segment-runtime.md)、[edge-runtime.md](docs/edge-runtime.md) |

## リポジトリ構成

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

**モデルファミリーごとに `.c` を 1 つ、共有の単一ヘッダの上に。** エンジンが持つのは自身のアーキテクチャだけで、それ以外は持ちません。2 つのエンジンが共に必要とするものは、両者がインクルードするヘッダに置かれるため、修正は一度にすべてのエンジンに届きます。リポジトリのルートからは、`make`、`make check`、`make clean` がエンジンの Makefile に委譲されます。

## プロジェクトへの支援

colibri は、RAM 25 GB の 12 コアのラップトップ上で、1 人のプロジェクトとして始まりました。今ではその数値は、実機を持つコミュニティから集まっています。役に立ったと感じたら：

- リポジトリにスターを付けて、共有してください。
- あなたのハードウェアでのベンチマーク数値を添えて issue を作成してください：データポイントは何よりもこのプロジェクトを前進させます。
- [Discord コミュニティ](https://discord.gg/RXV83nSZdk)に参加して、実験、ハードウェアでの結果、研究の方向性について議論してください。
- 開発のスポンサーやハードウェアの寄贈については、GitHub の issue からご連絡ください。

## なぜ「colibrì」なのか

ハチドリは体重わずか数グラムで、空中の一点にとどまり、1 日に千もの花を訪れます。このエンジンは、744B パラメータの巨人をハチドリの食事量で生かし続けます：RAM 25 GB、CPU 12 コア、そしてディスクへのたっぷりの忍耐です。

## 謝辞

colibri はエンジンにすぎず、それが動かす知性は贈り物です。重みをオープンに公開しているチーム、**Z.ai**（GLM）、**Moonshot AI**（Kimi）、**Alibaba Qwen**、**DeepSeek**、**Xiaomi**（MiMo）、**Thinking Machines**（Inkling）、**Allen AI**（OLMoE）、**Convai Innovations**（Laya）、**fastino**（GLiNER2.5-Decide）、**Cloudflare**（Clef）に、変換済みコンテナを公開してくれる人々に、そしてベンチマークを取り、バイセクトし、アトラスの実行を再現し、パッチを送ってくれたすべてのコントリビューターに感謝します。このリポジトリに含まれるサードパーティのコードとそのライセンス：[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

このプロジェクトのエキスパート配置、圧縮、ルーティングの実験は、以下のオープンな研究と
システムのアイデアやエビデンスにも基づいています:

- 出力を考慮した、ドメイン固有のエキスパート重要度については
  [REAP](https://github.com/CerebrasResearch/reap) と
  [EASY-EP](https://github.com/RUCAIBox/EASYEP)。
- 類似度に基づくエキスパートの再ルーティングについては [SERE](https://github.com/JL-Cheng/SERE)、
  キャッシュ局所性を考慮したルーターのファインチューニングについては
  [ReMoE](https://github.com/BUAA-OSCAR/ReMoE)。
- ルーティングに導かれたエキスパートのマージと圧縮については
  [MC-SMoE](https://github.com/UNITES-Lab/MC-SMoE)。
- 共有エキスパート基底と低ランクのエキスパート差分については
  [MoBE](https://github.com/inclusionAI/MoBE) と
  [D²-MoE](https://github.com/lliai/D2MoE)。
- CPU/GPU ハイブリッドのエキスパートスケジューリングについては
  [HybriMoE](https://github.com/PKU-SEC-Lab/HybriMoE)、エキスパート通信と計算のオーバーラップについては
  [ScMoE](https://arxiv.org/abs/2404.05019)、分散オンデマンドのエキスパートロードについては
  [OD-MoE](https://arxiv.org/abs/2512.03927)。
- 比較を再現可能にしているオープンな推論システムとエキスパートオフロードの取り組みについては
  [vLLM](https://github.com/vllm-project/vllm)、
  [llama.cpp](https://github.com/ggml-org/llama.cpp)、
  [kTransformers](https://github.com/kvcache-ai/ktransformers)。

エンジンはアイデアだけでなく、具体的なエンジニアリングの成果の上にも成り立っています。以下はいずれも
現在ツリー内で使われているか、再実装されています:

- [safetensors](https://github.com/huggingface/safetensors)：すべてのエンジンが読むコンテナ
  （`c/st.h`）。fp8 と I64 の dtype を含みます。
- [tiktoken](https://github.com/openai/tiktoken)：`c/tok.h` はその `byte_pair_encode` を
  正確に再実装しており、連結した結果の語彙 ID が最も小さい隣接ペアをマージするため、
  tiktoken 由来の語彙にはマージリストが不要です。
- [llama.cpp](https://github.com/ggml-org/llama.cpp)：`c/grammar.h` の GBNF 文法サブセットは
  その構文とスタック集合による PDA に従っており、Metal の経路はその
  `newBufferWithBytesNoCopy` による常駐テクニックを借用しています。
- [vLLM](https://github.com/vllm-project/vllm)：エンジンが位置ごとに一致させている出力
  セマンティクスのリファレンス（例: 最終ノルムが LM ヘッドに対してどこに入るか）。
- [transformers](https://github.com/huggingface/transformers)：オラクル:
  CI はランダム初期化モデルをこれに対してトークン単位で再現します。
- [DietGPU](https://github.com/facebookresearch/dietgpu)：実験的な圧縮エキスパートティア
  （`COLI_ANS`）の背後にある GPU ANS コーデック。
- [rocWMMA](https://github.com/ROCm/rocWMMA)：HIP バックエンドは CUDA の
  `nvcuda::wmma` の fragment/mma_sync API をこれにマッピングしており（`c/backend_gpu_compat.h`）、
  それによって 1 つの .cu ソースを両ベンダー向けにコンパイルできます。

## ライセンス

Apache 2.0、Copyright 2026 Vincenzo Fornaro。[LICENSE](LICENSE) と [NOTICE](NOTICE) を参照してください。各モデルには、その作者が付けたライセンスがそのまま適用されます（GLM-5.2 の重みは Z.ai により MIT ライセンスで公開されています。Qwen-Image-2.1 は非商用利用に限られます）。
