<p align="center">
  <img src="assets/colibri-logo.svg" width="560" alt="colibrì——小巧引擎，庞大模型">
</p>

<p align="center">
  <a href="https://discord.gg/RXV83nSZdk"><b>Discord</b></a> ·
  <a href="README.md">English</a> · 简体中文 · <a href="README.zh-TW.md">繁體中文</a> · <a href="README.it.md">Italiano</a> · <a href="README.ja.md">日本語</a>
</p>

**小巧引擎，庞大模型。**在消费级与异构硬件上运行**前沿 MoE 模型——从 744B 到
2.8T 参数**——以引擎零依赖的纯 C 实现，将存储、RAM 与 VRAM 视为统一的推理层级。

目前可运行九个模型家族：**GLM-5.2/5.3**（744B）、**GLM-5.3-Flash**（321B，含视觉）、
**Inkling**（975B）、**Kimi K3**（2.8T）、**DeepSeek V4 Flash**（284B）、**DeepSeek V4.1 Flash**（552B，含视觉）、
**Qwen3.8-Flash-Next**（125B + 51B n-gram）、**Qwen3.6**（35B-A3B）与 **OLMoE**（7B）
——各自一个 C 文件，共用同一套 `coli chat` / `coli serve` / `coli web` 前端。[完整列表](README.md#other-supported-models)

> **Colibrì 既是今天就能运行的推理引擎，也是一个开放的研究平台**。它的首要目标是在
> 完整的软硬件边界上追求推理侧性能——模型格式、内存层级、存储 I/O、放置、调度、内核、
> 推测解码以及 CPU/GPU 重叠执行——让大模型减少对稀缺硬件的依赖，并降低运行成本。

Colibrì 刻意用于验证激进的系统思路——因此**对速度不作 SLA 承诺，对语义则给出硬性保证**：
实验必须通过可复现的端到端测量证明价值；默认策略**绝不会在未告知的情况下改变模型精度或
路由语义**。高速内存不足可以降低速度，但不能悄悄重新定义模型。

```
$ ./coli chat
  🐦 colibri v1.12.0 — GLM-5.2 · 744B MoE · int4 · streaming CPU
  ✓ ready in 32s · resident 9.9 GB
  › ciao!
  ◆ Ciao! 😊 Come posso aiutarti oggi?
```

## 实际运行效果

<p align="center">
  <img src="docs/media/colibri-dashboard.png" width="900" alt="colibrì 网页仪表盘——实时指标、硬件面板与专家存储层级">
</p>
<p align="center"><em>网页仪表盘（<code>./coli web</code>），1.12.0 重新设计：一个工作区，底部停靠栏切换聊天、Brio 模式、
Brain 页面和性能分析，支持浅色与深色主题。图中是 Qwen3.6 在纯 CPU 机器上作答，专家从磁盘流式读取。</em></p>

<p align="center">
  <img src="docs/media/colibri-brio.png" width="900" alt="Brio 页面：文档只读一次，每个允许的答案各有一个概率，并给出熵">
</p>
<p align="center"><em><strong>Brio 模式</strong>：同一个模型，只是不再让它写。给它一段文档和唯一允许的几个答案，
它读出每个答案的概率，不生成任何 token，并给出一个熵，说明它何时没有把握。图中：<strong>request changes，99.9%</strong>，
熵 0.005，读取 4 个 token，生成 0 个。</em></p>

<p align="center">
  <img src="docs/media/colibri-brain.png" width="900" alt="大脑页面：GLM-5.2 的实测专家图谱绘成一块皮层，十个可进入的区域">
</p>
<p align="center"><em><strong>大脑（Brain）</strong>页面的 <strong>Explore</strong> 视图：将 GLM-5.2 的<a href="https://github.com/JustVugg/colibri/issues/175">实测专家图谱</a>绘成一块皮层。
13,260 个已分析专家分为十个区域（Python、SQL、数学、诗歌、法律、中文……）；位置取自实测路由亲和度，而非学习出的嵌入向量。
选择一个区域即可进入。<strong>Live routing</strong> 视图切换到正在运行的模型：每个专家一格，颜色代表存储层级，每轮被路由到的专家都会闪白。</em></p>

<p align="center">
  <img src="docs/media/colibri-brain-region.png" width="900" alt="Python 区域内部：1,142 个专家，其中一个被选中并显示其实测亲和度">
</p>
<p align="center"><em><strong>Python</strong> 区域内部：1,142 个专家组成的星座，每个都标注了层号和序号。面板显示其中一个：第 17 层第 178 号专家，
一个熵为 3.13 的通才，其实测亲和度为 Python 20.2%、JSON 14.6%、对话 14.2%、SQL 13.3%。</em></p>

<p align="center">
  <img src="docs/media/colibri-profiling.png" width="900" alt="性能剖析页面：引擎在每一轮中的时间去向">
</p>
<p align="center"><em><strong>性能剖析（Profiling）</strong>页面：引擎在每一轮中的时间去向，按阶段划分，并以最近 30 轮作为趋势。
此处为 CPU 机器上的 Qwen3.6：36 个提示词元与 55 个生成词元共用时 19.0 秒，2.9 tok/s，其中 11.4 秒的磁盘服务与计算重叠。</em></p>

## 研究使命

前沿模型推理不该默认要求数据中心级硬件。Colibrì 的研究目标很简单：
**优化证据表明受限的每一段推理路径，降低推理的硬件依赖与总成本**。

这包括改变权重的表示与移动方式，决定哪些内容常驻 VRAM、RAM 或存储，重叠异构计算，
降低启动与同步开销，利用稀疏性与复用，并验证新的解码算法。传统做法不是免责理由，
微基准快也不是采用理由；最终依据是在真实机器上的端到端推理，同时测量正确性、质量、
吞吐、延迟、内存与成本。

它最终带来的是可及性：在已有硬件上运行 744B 模型，实时观察每个专家，并直接修改实现。
不是从 API 租用智能，而是持有、探测、测量和改进它。引擎刻意保持足够小，让任何愿意测量
的人都可能贡献下一项有效优化。

## 核心技术与实测结论

- **统一层级，而非单一内存门槛**。VRAM、RAM 与 NVMe 是同一份权重的不同放置层级；
  高速内存不足只影响速度，不改变模型语义。
- **权重的 JIT**。实测路由热度驱动逐层 LRU、学习型热门专家固定区和提前一层的预取，
  无需加载所有专家。它在可重复负载上有收益，但历史可能过拟合，预取在部分主机上也可能
  负优化，因此它们是需要测量的策略，不是性能承诺。
- **I/O 本身就是引擎的一部分**。专家批次并集、读算重叠、`O_DIRECT` 与加权双 SSD
  分流直接优化流式路径，而不是假装存储延迟不存在。`O_DIRECT` 取决于磁盘，双 SSD
  仍需要更多社区端到端 A/B。
- **异构执行**。CPU、CUDA、Metal、NUMA 内存以及专家的部分或全部常驻共用一个运行时，
  可按机器条件组合；最佳组合取决于算力、带宽、驻留率与负载。
- **压缩状态，不篡改模型**。逐 token 精确的前向验证、缩小 57 倍的 MLA KV 状态、
  持久化热会话与忠实 DSA，让优化始终受正确性约束。这些是内存、延迟和正确性属性，
  不是笼统的吞吐承诺。
- **必须证明收益的推测解码**。原生 MTP 与语法强制草稿均接受端到端测量；
  接受率无法覆盖验证成本时可以关闭。

## 开放猜想、实验与参与方式

在受控的端到端 A/B 证明之前，Colibrì 将每项优化都视为猜想。当前主要问题如下：

| 猜想 | 当前证据 | 仍需完成的实验 |
|---|---|---|
| 路由历史能比普通 LRU 更好地放置专家 | 学习型固定区能改善重复负载，但也会对 prompt 过拟合 | 在代码、对话、多语言和长上下文负载上做留出集、跨会话 A/B |
| 多块 SSD 能将独立带宽转化为解码速度 | 加权镜像／分片路由已实现并通过校验，带宽模型成立 | 在独立控制器的真实磁盘上做冷缓存、单盘与双盘 GLM-5.2 对照 |
| 硬件感知规划器能自动接近每台机器的最优配置 | 当前已检测 RAM/VRAM 预算与多个后端 | 将自动方案与参数扫描对比，覆盖笔记本、工作站、NUMA 和多 GPU 主机 |
| 无损或质量受控的表示能充分减少权重搬运 | 已有格式与量化消融，并设置正确性／质量门槛 | 同时复现质量、搬运字节、延迟和每个有效 token 成本，而非只看压缩率 |
| 路由感知推测能在接近全驻留前盈利 | MTP 与语法草稿可用，但 MTP 在约 85% expert hit 时也实测过 -32% | 绘制接受率、命中率、批次并集与草稿深度的盈亏边界 |
| CPU/GPU 重叠能隐藏传输与同步，而非仅转移瓶颈 | CUDA 与 Metal 有成功数据，但强 CPU 和低驻留率会抹平收益 | 在 PCIe、统一内存与全驻留机器上做逐阶段 profile 和单变量 A/B |

想参与就任选一行，负结果也请公开。请记录硬件、commit、模型容器、完整命令、prompt、
缓存状态、吞吐、TTFT、expert hit、读取字节数与质量检查；每次只改一个变量，重复运行并附上
原始日志。先阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 和
[benchmark 协议](docs/benchmarks.md)，然后
[创建实验 issue](https://github.com/JustVugg/colibri/issues/new)。
在这里，一个受控的失败比一个无法解释的高数字更有价值。

## 核心概念

744B 的专家混合（Mixture-of-Experts）模型，每个 token 只会激活约 40B 参数——
其中每个 token 之间会变动的只有约 11 GB（被路由到的专家）：

<p align="center">
  <img src="docs/media/sparse.png" width="880" alt="每个 token 只会激活约 5.4% 的参数">
</p>

所以模型不必完整**装进**高速内存，而是需要正确**放置**：

- **稠密部分**（注意力、共享专家、嵌入——约 17B 参数）以 int4
  **常驻 RAM**（约 9.9 GB）；
- **19,456 个路由专家**（75 个 MoE 层 × 256，加上 MTP head；每个在 int4 下约 19 MB）
  **存放在磁盘**（约 370 GB），并**按需流式加载**，配合逐层 LRU 缓存、
  会学习的热门专家固定存储区，以及可选的 VRAM 层级。

引擎是一个 C 主文件（`c/colibri.c`）加上若干头文件。不需要 BLAS，
运行时不需要 Python，也不需要 GPU。

## 工作原理

### 每个 token 的处理路径

<p align="center">
  <img src="docs/media/token-path.png" width="880" alt="路由 → 并集 → 放置 → 重叠执行 → 学习">
</p>

每个 token 的每一层都会经过相同的五个步骤。设计目标是让
**放置只决定速度**——无论专家是从 VRAM 还是磁盘响应，路由器的决策与权重精度都完全相同。

### 统一内存层级，取代单一内存门槛

<p align="center">
  <img src="docs/media/tiers.png" width="880" alt="VRAM／RAM／NVMe 三层专家常驻架构">
</p>

同一套引擎覆盖完整硬件范围：在 25 GB 笔记本上，一切都从磁盘流式加载
（慢，但结果正确）；在大内存主机上，则可让整组专家常驻
（`CUDA_EXPERT_GB=auto PIN_GB=all`），让磁盘完全退出解码路径。
两端之间有一层**学习型缓存**：引擎会记录*你的*工作负载路由到哪些专家
（`.coli_usage`，每轮更新），并自动固定最热门的专家——colibrì 确实会越用越快。
在多路主机上，`COLI_NUMA=1` 会将常驻权重交错分配到各内存控制器
（[#82](https://github.com/JustVugg/colibri/issues/82)）。

### 绝不为同一次磁盘读取等待两遍

缓存未命中的代价很高，因此引擎大部分的巧思都用来避免或重叠这些读取：
每个专家的三个矩阵相邻存储，并以一次 `pread` 读取；有界异步 I/O 池
（`PIPE=1`，默认启用）会在常驻专家计算时加载缺失的专家；批量位置只读取每个
不重复专家一次（**批量并集**）；路由前瞻线程（`PILOT=1`）则预取下一层专家——
实测显示，路由结果提前一层时有 **71.6% 的可预测性**。
在 GPU 上，常驻管线（`COLI_CUDA_PIPE=2`）让残差流跨层保留在设备端，
使 CPU 专家循环不中断；在 Apple Silicon 上，实验性的
[Metal 后端](docs/metal.md)会用统一内存 GPU 执行批量专家运算。

### 忠实模型，压缩状态

前向传播已通过 `transformers` oracle 验证为**逐 token 完全一致**
（teacher-forcing 32/32）。MLA 注意力存储压缩后的 KV 状态——每个 token 为 576 个
浮点数，而非 32,768 个（**缩小 57×**）——并跨重启持久保存
（`.coli_kv`）：对话可暖启恢复，不需重新 prefill，结果与不中断的会话
逐字节相同。DSA 稀疏注意力（GLM-5.2 的 lightning indexer）已忠实实现，
并通过强制选取所有 key，验证可精确复现稠密注意力。

### 诚实的推测解码

GLM-5.2 原生 MTP head 会起草 token，再由主模型以一次批量前向传播验证——
条件合适时每次 forward 可产生 2.2–2.8 个 token。两条来之不易的规则已成为默认值：
MTP head 必须是 **int8**（int4 head 的接受率会崩塌到 0–4%，见
[#8](https://github.com/JustVugg/colibri/issues/8)），且草稿与验证必须计算
**相同函数**——`SPEC_PIN=1` 会把两者固定在同一 kernel family
（完整取证过程见 [#163](https://github.com/JustVugg/colibri/issues/163)）。
语法强制草稿（[`GRAMMAR=file.gbnf`](docs/grammar-draft.md)）可在受限 JSON 输出中，
以近乎免费的代价提高接受率。推测解码是否带来净收益取决于缓存热度——请实测，
若不划算就使用 `DRAFT=0`。

## 实际成果

<p align="center">
  <img src="docs/media/ladder.png" width="880" alt="各硬件级别的实测解码速度">
</p>

同一套引擎、同一个 int4 容器——硬件只会改变专家的存放位置。
[完整 benchmark 表格](docs/benchmarks.md)中的重点如下：

- **6× RTX 5090，全部常驻**：解码 5.8–6.8 tok/s，TTFT 约 13 秒
  （[实验记录](docs/experiments/glm52-6x5090-2026-07-12.md)）；
- **128 GB、仅使用 CPU 的台式机**：热缓存后约 1.8 tok/s
  （[#200](https://github.com/JustVugg/colibri/issues/200)）；
- **单张 RTX 5070 Ti 的笔记本级主机**：通过 GPU 常驻管线达到 1.07 tok/s
  （[#273](https://github.com/JustVugg/colibri/issues/273)）；
- **25 GB 开发机**：冷启动 0.05–0.1 tok/s——这是项目起步时已证实的下限，
  也仍是诚实的基准。

质量来自测量，而非假设：int4 容器的量化损失，以及 scale granularity／rotation
消融实验，收录于 [docs/benchmarks.md](docs/benchmarks.md#quality-benchmark)、
[#108](https://github.com/JustVugg/colibri/issues/108) 与
[#81](https://github.com/JustVugg/colibri/issues/81)。

## 开始使用

你需要两样东西：**程序本体**（几百 KB）和**模型**（372 GB）。各平台的分步
指引见 [Quick Start 指南](docs/quickstart.md)。

### 1. 获取 colibri

**下载预编译版本**——Linux、macOS 与 Windows 均已提供，无需编译器。从
[Releases](https://github.com/JustVugg/colibri/releases) 下载对应平台的压缩包并解压：

```bash
mkdir colibri && tar xzf colibri-v1.8.0-linux-x86_64.tar.gz -C colibri && cd colibri
python3 coli info                         # engine ready ✓
```

包内含引擎（`colibri`，Windows 上为 `colibri.exe`）、`coli` 启动器及其 Python
辅助脚本。无需重命名或配置：`coli` 会自动找到同目录下的引擎。你只需安装
[Python 3](https://www.python.org/downloads/)——启动器和 API gateway 是 Python
脚本，而引擎本身是零依赖的纯 C 程序。

**或者从源码构建**——需要带 OpenMP 的 `gcc`（或 clang）：

```bash
git clone https://github.com/JustVugg/colibri && cd colibri/c
./setup.sh                                # 检查 gcc/OpenMP、构建并运行自测
```

想把 `coli` 加入 PATH？在 checkout 中执行 `pip install -e .` 即可注册（引擎仍位于
`c/` 目录——这是从克隆目录做的可编辑安装，而非独立 wheel）。

### 2. 获取模型

Hugging Face 上已有预转换的 **GLM-5.2 int4** 容器——请务必使用
**含 int8 MTP head 的 group-scaled（gs64）版本**。它约为 **372 GB**，请放在空间足够的磁盘上，最好是快盘：

**https://huggingface.co/mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp**

**GLM-5.3** 属于同一家族,使用同一引擎加载。它有自己的 group-scaled(gs64)容器,
约 **419 GB**,且**不含** MTP head,因此推测解码保持关闭:

**https://huggingface.co/Justvugg/GLM-5.3-colibri-int4-g64**

> ⚠️ 请使用上面的 **gs64** 容器，不要使用较旧的 per-row int4 镜像
>（`mateogrgic/…`、`jlnsrk/…`）：后者质量实测低约 9 个百分点，也是
> [#455](https://github.com/JustVugg/colibri/issues/455) 最初 think-mode 循环与生成不终止的根因。
> gs64 修复了受控的 per-row A/B 问题，但不是通用的重复或 EOS starvation 防护。
> MTP head 也必须是 **int8，而非 int4**（int4 的草稿接受率为 0%，
> [#8](https://github.com/JustVugg/colibri/issues/8)）：
> `ls -l <model>/out-mtp-*`——正确的 int8 大小为 `3527131672 / 5366238584 / 1065950496`。

你也可以自行从 FP8 源转换——只需一条可断点续传的命令，且任何时候都不需要
在磁盘上同时存放完整的 756 GB：

```bash
./coli convert --model /nvme/glm52_i4     # 逐 shard 下载并转换（仅此一次需要 python）
```

### 3. 运行

```bash
COLI_MODEL=/nvme/glm52_i4 ./coli chat     # 自动检测 RAM 预算、缓存与 MTP
COLI_MODEL=/nvme/glm52_i4 ./coli plan     # 查看规划的 VRAM／RAM／磁盘配置
COLI_MODEL=/nvme/glm52_i4 ./coli doctor   # 只读就绪检查
./coli web  --model /nvme/glm52_i4        # 在同一端口提供 API 与网页仪表盘
./coli serve --model /nvme/glm52_i4       # 仅提供 OpenAI 兼容 API
```

#### Brio 模式：问一个封闭式问题

人们向模型提出的大多数请求是一次选择，而不是一段文字：哪个队列、哪个结论、某个字段应取四个值中的哪一个。
Brio 模式把允许的选项交给引擎，读出每个选项的概率，而不是生成文本：`completion_tokens` 为 0，
答案不可能落在你的列表之外，并且每个答案都附带一个熵，"模型没有把握"因此成为一个可以设阈值的数字。
它在全部九个模型家族上可用，运行在同一个服务器上，且按请求可选：不请求它的聊天，输出逐字节保持不变。

```bash
# 在 TUI 中：同一个模型，只是不再让它写
./coli chat --model /nvme/qwen36_i4_gs64
> /brio merge | request changes | close
> 340 lines, 8 files, no tests. CI is green but nothing covers that path.

# 从任何程序：向运行中的服务器发送一个 JSON 请求
curl -s http://127.0.0.1:8000/v1/brio -H 'Content-Type: application/json' -d '{
  "model": "qwen36",
  "state": "340 lines, 8 files, no tests. CI is green but nothing covers that path.",
  "question": "What should the reviewer do?",
  "options": ["merge", "request changes", "close"]}'
```

`questions` 可以对只读一次的文档提出多个问题；`schema` 逐字段填充一个 JSON 对象，结构上必然合法。
在 Qwen3.6 上与在同一台 CPU 机器上生成同样答案相比的实测：四字段 schema 快 2.4 倍，
对同一文档的四个问题快 5.7 倍。完整说明、请求与回复格式、以及它不适用的情形见 [docs/brio.md](docs/brio.md)。
仪表盘中也有 Brio 页面。


在 Windows 上同样使用这些命令，写作 `python coli chat --model D:\glm52_i4`。
引擎运行时是纯 C——python 只供一次性转换工具与可选的 API gateway 使用。

### 4. 深入了解

| 主题 | 文档 |
|---|---|
| Benchmark、社区实测数据、质量测量 | [docs/benchmarks.md](docs/benchmarks.md) |
| 调优选项、策略、学习型缓存、预取 | [docs/tuning.md](docs/tuning.md) |
| Windows 11 原生构建（含 CUDA DLL） | [docs/windows.md](docs/windows.md) |
| CUDA 后端、VRAM 专家层级、全部常驻 | [docs/cuda.md](docs/cuda.md) |
| Apple Silicon Metal 后端 | [docs/metal.md](docs/metal.md) |
| OpenAI 兼容 API、KV slots、网页仪表盘 | [docs/api.md](docs/api.md) |
| Brio 模式：对封闭的选项集打分而不是生成 | [docs/brio.md](docs/brio.md) |
| 语法强制草稿（结构化输出） | [docs/grammar-draft.md](docs/grammar-draft.md) |
| 环境变量完整清单 | [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) |

## DeepSeek V4

**DeepSeek V4 Flash** 直接流式读取官方 checkpoint，无需转换：路由专家
保持原生 FP4，稠密部分保持 fp8-e4m3。支持 x86-64／aarch64 Linux 与
Windows／MSYS2（CPU），并提供可选的 CUDA 层（Windows 运行时 DLL；Linux
`CUDA=1` 直接链接）——GTX 10 系及更新的 NVIDIA 显卡均可使用（Pascal/Turing
需 `CUDA_ARCH=portable-pre-ampere NO_TC=1` 构建），每个阶段都保持 CPU 语义一致并可
逐阶段回退。

```bash
cd c
make deepseek-v4
python ./coli run --model /path/to/DeepSeek-V4-Flash --ram 32 \
  "法国的首都是哪里？"
# 同一模型也支持：coli chat / coli serve / coli web
```

状态、checkpoint 验证、统一 CLI／server 用法和动态生成的 tiny 独立
oracle 说明，请参阅[中文版 DeepSeek V4 文档](docs/deepseek-v4.zh-CN.md)；
英文原文见 [docs/deepseek-v4.md](docs/deepseek-v4.md)。

## 下一步

- **推理系统研究就是产品**。当前层级采用 LRU 与学习型固定集；正在研究模型格式、压缩、
  放置、调度、I/O、CPU/GPU 内核、异构重叠、KV 状态与路由感知推测。目标是降低硬件要求
  和每个有效 token 的成本，所有成果都以端到端测量为准、经审查并公开开发。
- **支持更多开放模型**。层级算法与模型无关，任何带路由专家的 MoE 都能用相同方式分层。
  目前已有九个模型家族可用（GLM-5.2、GLM-5.3-Flash、Inkling、Kimi K3、DeepSeek V4 Flash、DeepSeek V4.1 Flash、
  Qwen3.8-Flash-Next、Qwen3.6、OLMoE）；更多开放权重家族（候选包括 **MiniMax**）将沿用同样的
  规则获得引擎支持：有人完成端到端实测之后。

## 支持项目

colibrì 最初由一人使用 12 核心、25 GB RAM 的笔记本开发；
如今它的数据来自社区中各种真实机器。如果这个项目对你有用：

- ⭐ 为仓库加星并分享；
- 🐛 以 issue 提交你的硬件 benchmark 数据——实测数据比任何其他事都更能推动项目；
- 💬 加入 [Discord 社区](https://discord.gg/RXV83nSZdk)，讨论实验、硬件数据与研究方向；
- 💬 若想赞助开发或捐赠硬件，请通过 GitHub issues 联系。

## 仓库结构

```
Makefile                  根目录构建／检查入口
c/
├── colibri.c             引擎主文件
├── quant.h               量化 matmul 内核（SIMD 多架构）
├── sample.h              采样与 stop-set 管理
├── kv_persist.h          .coli_kv 磁盘持久化
├── telemetry.h           仪表盘协议、统计与用量持久化
├── st.h, tok.h, json.h   运行时头文件
├── backend_cuda.*        可选的 CUDA 层级
├── Makefile              构建与本地检查
├── coli                  用户界面 CLI
├── openai_server.py      OpenAI 兼容 HTTP gateway
├── setup.sh              一条命令完成本地设置
├── tools/                离线转换、fixtures 与 benchmarks
├── scripts/              长时间转换辅助工具
└── tests/                零依赖的 C 与 Python 测试
web/                      浏览器 UI（纯 OpenAI API client）
desktop/                  封装网页 UI 的 Tauri v2 桌面 shell
docs/                     参考文档、实验、媒体文件与 DeepSeek V4 说明
```

运行时路径刻意保持扁平、易读：`colibri.c` 加上若干头文件。
在仓库根目录执行 `make`、`make check` 与 `make clean`，
都会转发给引擎的 Makefile。

## 为什么叫"colibrì"

蜂鸟只有几克重，能在原地悬停，并在一天内造访上千朵花。
这套引擎只用蜂鸟般的配给，就能让 744B 参数的巨人运转：
25 GB RAM、十二个 CPU 核心，以及对磁盘的大量耐心。

## 许可证

Apache 2.0，Copyright 2026 Vincenzo Fornaro。详见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。GLM-5.2 权重由 Z.ai 以 MIT 许可发布。
