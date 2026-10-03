<p align="center">
  <img src="assets/colibri-logo.svg" width="560" alt="colibrì——小巧引擎，庞大模型">
</p>

<p align="center">
  <a href="https://justvugg.github.io/colibri"><img src="https://img.shields.io/badge/website-justvugg.github.io%2Fcolibri-1f6feb" alt="Website"></a>
  <a href="https://github.com/JustVugg/colibri/releases"><img src="https://img.shields.io/github/v/release/JustVugg/colibri?color=2ea043" alt="Latest release"></a>
</p>

<p align="center">
  <a href="https://justvugg.github.io/colibri"><b>网站</b></a> ·
  <a href="https://discord.gg/RXV83nSZdk"><b>Discord</b></a> ·
  <a href="README.md">English</a> · 简体中文 · <a href="README.zh-TW.md">繁體中文</a> · <a href="README.it.md">Italiano</a> · <a href="README.ja.md">日本語</a>
</p>

**小巧引擎，庞大模型。**在消费级与异构硬件上运行**前沿 MoE 模型——从 744B 到
2.8T 参数**——以引擎零依赖的纯 C 实现，将存储、RAM 与 VRAM 视为统一的推理层级。

目前可运行十个语言模型家族。这里按引擎计数，而不是按 checkpoint：各自一个 C 文件，
共用同一套 `coli chat` / `coli serve` / `coli web` 前端，部分引擎可运行不止一个模型。
**GLM-5.2/5.3**（744B）、**GLM-5.3-Flash**（321B，含视觉）、**Inkling**（975B）、
**Kimi K3**（2.8T）、**DeepSeek V4 Flash**（284B）、**DeepSeek V4.1 Flash**（552B，含视觉）、
**MiMo-V2.6 Flash**（309B，含视觉；同一引擎也运行 **MiMo-V2.6 Pro**，1.02T）、
**Qwen3.8-Flash-Next**（125B + 51B n-gram）、**Qwen3.6**（35B-A3B；同一引擎也运行
**Qwen3-Coder-30B-A3B** 与稠密的 **Qwen3.8-27B**，含视觉）以及 **OLMoE**（7B）。
也能生成图像：第十一个引擎运行 **Qwen-Image-2.1**，根据文字生成图片，`coli chat` 直接在终端中显示，
`coli serve` 通过 `POST /v1/images/generations` 提供（[qwen-image.md](docs/qwen-image.md)）。
[完整列表 ↓](#other-supported-models)

> **Colibrì 既是今天就能运行的推理引擎，也是一个开放的研究平台**。它的首要目标是在
> 完整的软硬件边界上追求推理侧性能——模型格式、内存层级、存储 I/O、放置、调度、内核、
> 推测解码以及 CPU/GPU 重叠执行——让大模型减少对稀缺硬件的依赖，并降低运行成本。

Colibrì 将 VRAM、RAM 与存储视为统一的多层级结构，并刻意用于验证激进的系统思路，因此
**对速度不作 SLA 承诺，对语义则给出硬性保证**：实验必须通过可复现的端到端测量证明价值；
默认策略**绝不会在未告知的情况下改变模型精度或路由语义**。高速内存不足可以降低速度，
但不能悄悄重新定义模型。

```
$ ./coli chat
  🐦 colibri v1.12.1 — GLM-5.2 · 744B MoE · int4 · streaming CPU
  ✓ ready in 32s · resident 9.9 GB
  › ciao!
  ◆ Ciao! 😊 Come posso aiutarti oggi?
```

## 实际运行效果

<p align="center">
  <img src="docs/media/colibri-dashboard.png" width="900" alt="colibrì 网页仪表盘——实时指标、硬件面板与专家存储层级">
</p>
<p align="center"><em>网页仪表盘（<code>./coli web</code>），1.12.0 重新设计：一个工作区，底部停靠栏切换聊天、System One 模式、
Brain 页面和性能分析，支持浅色与深色主题。图中是 Qwen3.6 在纯 CPU 机器上作答，专家从磁盘流式读取。</em></p>

<p align="center">
  <img src="docs/media/colibri-brio.png" width="900" alt="System One 页面：文档只读一次，每个允许的答案各有一个概率，并给出熵">
</p>
<p align="center"><em><strong>System One 模式</strong>：同一个模型，只是不再让它写。给它一段文档和唯一允许的几个答案，
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

有了 Colibrì，私有部署的前沿模型不再受限于能否获得超大规模云厂商级别的硬件。

凭借多层级（multitiering）特性，Colibrì **通过积极优化推理引擎的各条功能流水线，消除对专有硬件的依赖**。

我们的具体使命包括：改变权重的表示与移动方式，决定哪些内容常驻 VRAM、RAM 或存储，重叠异构计算，
降低启动与同步开销，利用稀疏性与复用，并验证新的解码算法。传统做法不是免责理由，
微基准快也不是采用理由；最终依据是在真实机器上的端到端推理，同时测量正确性、质量、
吞吐、延迟、内存与成本。

它带来的实际结果是**可及性**：在已有硬件上运行 744B 模型，实时观察每个专家，并直接修改实现。
不是从 API 租用智能，而是*持有*它：探测、测量和改进它。引擎刻意保持足够小，让任何愿意测量
的人都可能贡献下一项有效优化。

## 核心技术与实测结论

- **统一层级，不受单一层级容量限制**。VRAM、RAM 与 NVMe 是同一份权重的不同放置层级；
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
| 多块 SSD 能将独立带宽转化为解码速度 | 两块独立 NVMe 实测解码 +37.5%；经加权分流后，较慢的第三块盘影响持平（[测量数据](docs/multidisk.md#what-has-been-measured)） | 在不同磁盘速度、控制器布局与缓存状态下复现 |
| 硬件感知规划器能自动接近每台机器的最优配置 | 当前已检测 RAM/VRAM 预算与多个后端 | 将自动方案与参数扫描对比，覆盖笔记本、工作站、NUMA 和多 GPU 主机 |
| 无损或质量受控的表示能充分减少权重搬运 | 已有格式与量化消融，并设置正确性／质量门槛 | 同时复现质量、搬运字节、延迟和每个有效 token 成本，而非只看压缩率 |
| 路由感知推测能在接近全驻留前盈利 | MTP 与语法草稿可用，但 MTP 在约 85% expert hit 时也实测过 -32% | 绘制接受率、命中率、批次并集与草稿深度的盈亏边界 |
| CPU/GPU 重叠能隐藏传输与同步，而非仅转移瓶颈 | CUDA 与 Metal 有成功数据，但强 CPU 和低驻留率会抹平收益 | 在 PCIe、统一内存与全驻留机器上做逐阶段 profile 和单变量 A/B |

想参与就任选一行，负结果也请公开。请记录硬件、commit、模型容器、完整命令、prompt、
缓存状态、吞吐、TTFT、expert hit、读取字节数与质量检查；每次只改一个变量，重复运行并附上
原始日志。先阅读 [CONTRIBUTING.md](CONTRIBUTING.md)，对照
[benchmark 协议](docs/benchmarking.md)，然后
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

可以把核心算法理解为**权重的 JIT**。编译器的 JIT 从不编译整个程序：它观察实际运行的部分，
及时编译热点路径。colibrì 对 744B 的参数空间下了同样的赌注：参数不是需要一直持有的常驻状态，
而是**需要分级调度的数据**，在路由器证明需要它们的那一刻，于异构存储层级（VRAM / RAM / NVMe）
之间就位。实测路由热度决定哪些专家进入哪一层级，路由器提前一层运行，让预取隐藏调度延迟；
而且像 JIT 一样，引擎会学习你的工作负载：运行得越多，正确的专家就越热。之所以可行，是因为
路由具有可测量的结构（见[专家图谱](https://github.com/JustVugg/colibri/issues/175)），
而结构是可以缓存的。

引擎是一个 C 主文件（`c/colibri.c`）加上若干头文件。不需要 BLAS，
运行时不需要 Python，也不需要 GPU。

### 本地集群模式

协调节点（coordinator）在本地保留 token 生成、路由与 KV 状态，而以磁盘为后端的专家 worker
在其他 Mac 上执行被路由的 FFN。一层的路由批量并集作为一个持久 TCP 请求发送，因此一个 token
不会为每个专家各付出一次往返。

启动可选的注册服务：

```bash
./coli cluster coordinator --host 0.0.0.0 --port 8765
```

在每个 worker 上，本地需有同一个已转换的模型：

```bash
./coli cluster worker --model /nvme/glm52_i4 --port 9100 \
  --coordinator http://COORDINATOR:8765 --advertise-host WORKER_IP
```

以发现模式运行协调节点，或为静态部署提供 `--cluster-workers
HOST:PORT,...`：

```bash
./coli serve --model /nvme/glm52_i4 \
  --cluster-coordinator http://127.0.0.1:8765
```

除非配置了 worker，否则传输层保持关闭，现有的单机路径不受影响。稠密层分片与
浏览器／WebGPU worker 是另行跟进的扩展点。

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

<a id="dual-ssd-two-copies-of-the-model-twice-the-read-bandwidth"></a>

### 多块 SSD：从不止一块盘流式读取模型副本

当解码受磁盘限制时，**第二块 SSD** 会有帮助：在上面放一份模型副本，让引擎同时从两块盘读取。
以 GLM-5.2 为例，在源码 checkout 的 `c/` 目录下（或在解压后的发布包中）：

```bash
COLI_MODEL_MIRROR=/second/glm52_i4 python3 ./coli chat --model /fast/glm52_i4
```

引擎会在启动时测量各块盘，以确定读取分配的权重。缓冲读取采用确定性的专家路由；符合条件的
直接读取可以把一个专家条带化分布到多个副本上。独立的磁盘提供的是带宽余量，而不是有保证的
token 速率倍数：共享控制器、缓存命中与计算都可能限制收益。Bash 与 PowerShell 示例、实测收益
与限制，以及与单盘的对比，见[多磁盘指南](docs/multidisk.md)。值得了解的细节：

- 镜像会在**启动时校验**（每个文件的大小与 safetensors 头必须与主副本逐字节一致）；不一致或缺失的文件仍由主副本提供，因此**部分镜像也可以**：较小的第二块 SSD 可以提供它所持有的 shard；
- 镜像**从不写入**：`.coli_usage`、`.coli_kv` 与所有 sidecar 都保留在主副本上；
- 镜像读取出错时会回退到主副本（一条警告，不会崩溃），因此运行中拔掉第二块盘只会降级，不会让服务器退出；
- 路由从不改变 token：两份副本逐字节相同；启用 `PROF=1` 可看到 `MIRROR:` 性能计数器，显示每块盘提供了多少 GB。

同一套引擎覆盖完整硬件范围：在 25 GB 笔记本上，一切都从磁盘流式加载
（慢，但结果正确）；在大内存主机上，则可让整组专家常驻
（`CUDA_EXPERT_GB=auto PIN_GB=all`），让磁盘完全退出解码路径。
两端之间有一层**学习型缓存**：引擎会记录*你的*工作负载路由到哪些专家
（`.coli_usage`，每轮更新），并自动固定最热门的专家——colibrì 确实会越用越快。
在多路主机上，`COLI_NUMA=1` 会将常驻权重交错分配到各内存控制器
（[#82](https://github.com/JustVugg/colibri/issues/82)）。

对于装不下整个模型的第二块盘，Colibri 可以根据它已经学到的专家历史，为部分镜像排序。
先运行几个有代表性的 prompt，让 `.coli_usage` 反映实际负载，然后规划、暂存并校验镜像：

```bash
./c/coli mirror plan  --model /fast/glm52_i4 --mirror /second/glm52_i4 \
  --budget-gib 200 --reserve-gib 20
./c/coli mirror stage --model /fast/glm52_i4 --mirror /second/glm52_i4 \
  --budget-gib 200 --reserve-gib 20
./c/coli mirror verify --model /fast/glm52_i4 --mirror /second/glm52_i4
```

规划器直接读取 safetensors 头，跟随 `COLI_MODEL_DIRS` 中的分割模型目录，并优先选择能服务
最热路由专家的 shard。暂存从不改动主模型：它经由临时文件复制，保留所要求的剩余空间，
用 SHA-256 校验每个 shard，从不删除已有的镜像 shard，并且只在所选镜像就绪后才以原子方式
发布回执。

### 绝不为同一次磁盘读取等待两遍

缓存未命中的代价很高，因此引擎大部分的巧思都用来避免或重叠这些读取：
每个专家的三个矩阵相邻存储，并以一次 `pread` 读取；有界异步 I/O 池
（`PIPE=1`，默认启用）会在常驻专家计算时加载缺失的专家；批量位置只读取每个
不重复专家一次（**批量并集**）；路由前瞻线程（`PILOT=1`）则预取下一层专家——
实测显示，路由结果提前一层时有 **71.6% 的可预测性**。
在 GPU 上，常驻管线（`COLI_CUDA_PIPE=2`）让残差流跨层保留在设备端，
使 CPU 专家循环不中断；在 Apple Silicon 上，实验性的
[Metal 后端](docs/metal.md)会用统一内存 GPU 执行批量专家运算；
[Vulkan 后端](docs/vulkan.md)则把专家层级、稠密投影与 MLA 注意力核心带到任何具有
Vulkan 1.2 驱动的 GPU 上，包括通过 Mesa/RADV 的 AMD 显卡（对于厂商软件栈已不再支持的显卡，
例如 RX 580，它是唯一的后端；在 RDNA4 上与 ROCm 不相上下，见[基准测试说明](docs/vulkan.md)）。
其余所有引擎现在也会在 `VK=1` 构建中通过 `COLI_VULKAN=1` 使用同一后端放置常驻矩阵；
CI 会将其与 CPU 的 token 对照检查，但在它们首次实测的真实 GPU（集成显卡 Radeon 780M）上，
目前比 CPU 慢（[其他引擎](docs/vulkan.md#the-other-engines)）。

> **在真实 NVMe 上，请实测 `DIRECT=1`。** O_DIRECT 绕过页缓存，在带 DRAM 缓存且带宽有余量
> 的磁盘上往往有显著收益（在一台 Blackwell/Windows 机器上配合 `PIPE=1` 实测解码 +34%；
> 在 GB10 上 iobench 为 4.25→9.69 GB/s），但它取决于磁盘：QLC／无 DRAM 或虚拟化的磁盘
> 可能没有收益甚至变慢。先试一试，保留你的硬件真正受益的设置。

### 忠实模型，压缩状态

前向传播已通过 `transformers` oracle 验证（teacher-forcing 通常为 30-32/32；
tiny oracle 中有两个位置是浮点数近似平局，结果取决于工具链）。MLA 注意力存储压缩后的
KV 状态（每个 token 为 576 个浮点数，而非 32,768 个，**缩小 57×**），并跨重启持久保存
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

验证批次还可以通过 `COLI_EXACT_VERIFY=1` 选用**精确注意力核心**
（[#689](https://github.com/JustVugg/colibri/issues/689)）：CPU 上 MLA-absorb 的 score 与
context 点积改为累加整数乘积并只舍入一次，因此验证行中的近似平局在每台主机上都会得到相同的
结果；在 tiny oracle 上速度约为 0.6x tok/s（点积本身约为浮点循环的 5–7x）。需要了解两点
限制：使用量化 KV 缓存（`tq1`、TQ 或 int8 KV）时，context 点积仍走浮点路径，因此那里不提供
精确性；而真正的近似平局翻转目前只是推论，尚未在 GLM-5.2 的 n=64 上实际捕捉到。

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

### 一步完成

**Windows：** 下载仓库（**Code** 中的 **Download ZIP**，或 `git clone`），解压后
双击 **`START-HERE.bat`**。
**Linux 和 macOS：**

```bash
git clone https://github.com/JustVugg/colibri && cd colibri
./start-here.sh
```

它会检测内存、磁盘和 GPU，推荐一个适合这台机器的模型（按回车即采用推荐），在 GPU
可用时用 Vulkan 或 CUDA 编译引擎（或获取预编译版本），以可续传的方式下载模型（随时
可以中断，再次运行即从断点继续），并在浏览器中打开仪表盘。它还会打印供其他应用使用的
OpenAI 和 Anthropic 基础 URL。之后再次运行，colibri 会直接启动；`c/coli stop` 可以
停止它。每一步做什么：[quickstart.md](docs/quickstart.md#the-one-step-way)。

在用 AI 编程助手？让它按照 [docs/AI_SETUP.md](docs/AI_SETUP.md) 来安装 colibri。
支持 Model Context Protocol 的助手可以使用 `coli mcp`
（[MCP_SERVER.md](docs/MCP_SERVER.md)）。

下面是手动安装的步骤。

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

**GLM-5.3** 属于同一家族，使用同一引擎加载。它有自己的 group-scaled（gs64）容器，
约 **419 GB**，且**不含** MTP head，因此推测解码保持关闭：

**https://huggingface.co/Justvugg/GLM-5.3-colibri-int4-g64**

> ⚠️ 请使用上面的 **gs64** 容器，不要使用较旧的 per-row int4 镜像
>（`mateogrgic/…`、`jlnsrk/…`）：后者质量实测低约 9 个百分点，也是
> [#455](https://github.com/JustVugg/colibri/issues/455) 最初 think-mode 循环与生成不终止的根因。
> gs64 修复了受控的 per-row A/B 问题，但不是通用的重复或 EOS starvation 防护。
> MTP head 也必须是 **int8，而非 int4**（int4 的草稿接受率为 0%，
> [#8](https://github.com/JustVugg/colibri/issues/8)）：
> `ls -l <model>/out-mtp-*`：正确的 int8 是三个文件，大小为 `3527131672 / 5366238584 / 1065950496`，
> 或单个 `out-mtp-00000.safetensors`，大小为 `9959321520` 字节（推荐容器当前的上传版本就是单个文件：
> 同样的 int8 张量，共 777 个，每个元素一个字节）。

你也可以自行从 FP8 源转换——只需一条可断点续传的命令，且任何时候都不需要
在磁盘上同时存放完整的 756 GB：

```bash
./coli convert --model /nvme/glm52_i4     # 逐 shard 下载并转换（仅此一次需要 python）
```

<a id="other-supported-models"></a>
#### 其他支持的模型

GLM-5.2 是参考模型，但同样的流式方法还能运行另外九个语言模型家族，另有一个引擎用于生成图像。
每个都是一个**同级引擎**：一个 C 文件、自己的架构、同一套 `coli chat` / `coli serve` /
`coli web` 前端（启动器根据模型的 `config.json` 选择二进制文件，图像模型则根据
`model_index.json`）：

> **各自需要什么。** 这些模型差别很大，有人把其中两个放在一起读，误以为要求互相矛盾
> （[#191](https://github.com/JustVugg/colibri/issues/191)）。它们并不矛盾，只是不同的模型。
> **它们都不需要 GPU。**
>
> | 模型 | 权重所需磁盘 | RAM | GPU |
> |---|---|---|---|
> | **OLMoE** | 约 7 GB（int8 容器） | 8 GB | 不需要；可选 Vulkan |
> | **GLM-5.2/5.3** | 约 372 GB（5.2）／约 419 GB（5.3） | 最低 16 GB，舒适 24 GB | 不需要；可选 Vulkan |
> | **GLM-5.3-Flash** | 转换后约 195 GB | 25 GB（int4 权重 12 GB + 专家缓存） | 不需要；可选 Vulkan |
> | **Inkling** | 约 469 GB | 使用 int4 稠密容器时 25 GB，不使用时约 120 GB | 不需要；可选 Vulkan |
> | **Kimi K3** | 约 1.6 TB | 32 GB 以上 | 不需要；可选 Vulkan |
> | **DeepSeek V4 Flash** | 约 167 GB（REAP 150B：约 85 GB） | 最低 16 GB，舒适 32 GB | 可选；GTX 10 系列及以上的任意 NVIDIA 显卡（Pascal/Turing 通过 `CUDA_ARCH=portable-pre-ampere NO_TC=1`，RTX 50 上效果最佳）可使 prefill 快 5-10 倍、解码快约 2.5 倍；可选 Vulkan |
> | **DeepSeek V4.1 Flash** | 约 510 GB（官方 checkpoint；其中 203 GB 是每次只读取几百字节的 n-gram 记忆） | 约 18 GB 常驻（稠密部分、嵌入、视觉），加上由 `--ram` 决定大小的专家缓存；cap 8 时实测峰值 RSS 24.8 GB | 不需要；可选 Vulkan |
> | **MiMo-V2.6 Flash** | 约 178 GB（官方 checkpoint） | 每层缓存 32 个专家时实测常驻 30.1 GB，64 个时 49.8 GB；缓存大小由 `--ram` 决定 | 不需要；可选 Vulkan |
> | **MiMo-V2.6 Pro** | 约 574 GB（不含引擎从不加载的三个文件时约 564 GB） | 稠密部分按发布格式为 30.2 GiB，int8 下为 21.7 GiB；每层缓存 12 个专家时实测常驻 48.0 GB（稠密部分按发布格式），20 个时 50.7 GB（稠密部分为 int8） | 不需要；可选 Vulkan |
> | **Qwen3.8-Flash-Next** | 约 185.5 GB（官方 FP8 checkpoint），可选的 int4-g64 专家 sidecar 另需 68.0 GB | 使用 FP8 专家、默认上下文时舒适为 24 GB（cap 32；16 GB 低于下限）；使用 int4-g64 sidecar 时 cap 32 实测 RSS 11.6 GB | 可选；CUDA VRAM 专家层级（仅限 FP8 专家），稠密主干在 VRAM 中量化为 int8；可选 Vulkan |
> | **Qwen3.8-27B**（稠密，文本与图像） | 转换后约 51 GB（f16） | 稠密权重为 int4 时 20 GB，int8 时 30 GB | 不需要；暂无 CUDA 层级；可选 Vulkan |
> | **Qwen3.6-35B-A3B** | 约 20 GB（int4-gs64 容器） | 24 GB（需要完全常驻 RAM） | 可选；CUDA VRAM 专家层级在两张 8 GB 显卡上实测 **1.44 -> 10.05 tok/s（7.0x）**，输出与 CPU 逐位一致；可选 Vulkan |
> | **Qwen3-Coder-30B-A3B** | 约 19 GB（int4-gs64 容器；int8 为 30 GB） | 每层缓存 32 个专家时实测常驻 6.5 GB，全部 128 个时 15.2 GB | 不需要；可选 Vulkan |
> | **Qwen-Image-2.1**（文生图） | 约 33 GB（官方 diffusers checkpoint） | 全部常驻 16.0 GB；每个 prompt 加载文本编码器时峰值 8.5 GB，另加工作缓冲区（生成一张 768x512 图像实测峰值 9.0 GB） | 不需要；可选 Vulkan，尚未在 GPU 上计时 |
>
> GPU 从不改变模型的回答，只改变计算在哪里进行。速度由你的磁盘决定，因为专家是从磁盘
> 流式读取的：慢盘上每秒不到一个 token，快盘在缓存预热后每秒几个 token。
>
> **可选 Vulkan** 指的是 `VK=1` 构建。以 `COLI_VULKAN=1` 运行时，引擎会把常驻矩阵放到任何
> 具有 Vulkan 1.2 驱动的 GPU 上（GLM-5.2 在那里有完整的解码路径），CI 在软件驱动上将这些
> 引擎与 CPU 的 token 对照检查
> （[vulkan.md](docs/vulkan.md#the-other-engines)）。正确还不等于更快。在这些引擎首次实测的
> 真实 GPU，即集成显卡 Radeon 780M（Ryzen 7 PRO 8700GE，同样的二进制文件，冷页缓存）上，
> 目前比 CPU 慢：Qwen3.6-35B-A3B 解码 3.06 tok/s，CPU 为 5.97，输出完全相同；使用 int4 专家的
> Qwen3.8-Flash-Next 为 1.53，CPU 为 3.55。每次矩阵乘法都是一次同步提交，每个 token 约 726 次，
> 而集成 GPU 读取的是与 CPU 相同的 RAM。

| 家族 | 总参数 / 激活参数 | 权重 | 构建 | 文档 |
|---|---|---|---|---|
| **GLM-5.2/5.3** | 744B / 40B | [`mastouri/…-int4-g64-with-int8-mtp`](https://huggingface.co/mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp)（372 GB）或 [`Justvugg/GLM-5.3-colibri-int4-g64`](https://huggingface.co/Justvugg/GLM-5.3-colibri-int4-g64)（419 GB） | `make -C c glm` | 本页 |
| **Inkling**（Thinking Machines） | 975B / 41B | [`nbeerbower/Inkling-colibri-int4`](https://huggingface.co/nbeerbower/Inkling-colibri-int4)（469 GB） | `make -C c inkling` | [inkling.md](docs/inkling.md) |
| **GLM-5.3-Flash**（Z.ai） | 321B / 18B | [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash)，路由专家转换为 **int4-gs64**，稠密部分保持 BF16，精度在加载时选择；含视觉 | `make -C c glm53` | [glm53-flash.md](docs/glm53-flash.md) |
| **Kimi K3**（Moonshot） | 2.8T / 104B | [`moonshotai/Kimi-K3`](https://huggingface.co/moonshotai/Kimi-K3)，原始 checkpoint，路由专家保持**原生 MXFP4** | `make -C c kimi_k3` | [kimi_k3.md](docs/kimi_k3.md) |
| **DeepSeek V4 Flash** | 284B / 13B | 官方分片 checkpoint，路由专家保持**原生 fp4**，稠密部分保持 fp8-e4m3；**REAP 剪枝的 150B**（[`puwaer/DeepSeek-V4-Flash-0731-reap-150b`](https://huggingface.co/puwaer/DeepSeek-V4-Flash-0731-reap-150b)，85 GB，256 个专家中保留 132 个）用同一引擎加载，无需转换 | `make -C c deepseek-v4` | [deepseek-v4.md](docs/deepseek-v4.md) |
| **DeepSeek V4.1 Flash** | 552B / 16B | 官方 checkpoint，**无需转换**：专家本身已是 fp4，稠密部分为 fp8-e4m3。其中 203 GB 是每次只从磁盘读取几百字节的 n-gram 记忆，路由专家**每个 token 4.5 GB**，而 GLM-5.2 为 12.7 GB。视觉、工具调用与 DSpark 草稿 head 全部启用 | `make -C c deepseek_v41` | [deepseek-v41.md](docs/deepseek-v41.md) |
| **MiMo-V2.6 Flash**（Xiaomi） | 309B / 15B | [`XiaomiMiMo/MiMo-V2.6-Flash-MOPD`](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD)（178 GB），官方 checkpoint，**无需转换**：路由专家保持**原生 MXFP4**，稠密部分保持 FP8/BF16。48 层中有 39 层只关注 128 个 token 的窗口，因此长上下文只需 9 层的 KV。支持视觉与工具调用 | `make -C c mimo` | [mimo.md](docs/mimo.md) |
| **MiMo-V2.6 Pro**（Xiaomi） | 1.02T / 42B | [`XiaomiMiMo/MiMo-V2.6-Pro-MOPD`](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD)（573.5 GB），官方 checkpoint，**无需转换**，运行在 MiMo 引擎上：同一架构，70 层、384 个专家。已在 70 层中的前 32 层上与 Xiaomi 自己的建模代码对照验证。支持视觉与工具调用 | `make -C c mimo` | [mimo.md](docs/mimo.md#pro) |
| **Qwen3.8-Flash-Next**（Alibaba） | 125B + 51B n-gram / 6B | [`Qwen/Qwen3.8-Flash-Next-FP8`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)，原始 checkpoint；PLE 保持可分页，专家保持**原生 block-FP8**，或从可选的 sidecar 以 **int4-g64** 读取（见下文）。可选 MTP 草稿（`Q38_MTP=1`） | `make -C c qwen38`（VRAM 专家层级需 `CUDA=1`） | [qwen38.md](docs/qwen38.md) |
| **Qwen3.6**（Alibaba） | 35B / 3B | [`Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64`](https://huggingface.co/Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64)（约 20 GB，**推荐**），Gated Attention + Gated DeltaNet 混合架构 | `make -C c qwen36`（VRAM 专家层级需 `CUDA=1`） | [qwen36.md](docs/qwen36.md) |
| **Qwen3.8-27B**（Alibaba） | 27B，稠密 | 用 `c/tools/convert_qwen36.py` 将 [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) 转换为 f16 容器（51 GB）；引擎在加载时将其量化为 int8，设置 `COLI_DENSE_BITS=4` 时量化为 int4。每层一个 MLP，没有路由器，运行在 Qwen3.6 引擎上。支持文本与图像 | `make -C c qwen36` | [qwen36.md](docs/qwen36.md#the-dense-27b) |
| **Qwen3-Coder-30B-A3B**（Alibaba） | 30B / 3B | [`Justvugg/Qwen3-Coder-30B-A3B-colibri-int4`](https://huggingface.co/Justvugg/Qwen3-Coder-30B-A3B-colibri-int4)（19 GB，int4-gs64），由 [`Qwen/Qwen3-Coder-30B-A3B-Instruct`](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) 转换而来。运行在 Qwen3.6 引擎上的全注意力 Qwen3 MoE，128 个专家 top-8，有自己的 XML 工具调用格式，不带思考；在 teacher forcing 下，int4 容器在 96.9% 的位置上选出与 bf16 发布版相同的 top-1 token | `make -C c qwen36` | [qwen36.md](docs/qwen36.md#qwen3-coder-30b-a3b) |
| **OLMoE**（AI2） | 7B / 1B | 用 `c/tools/convert_olmoe_merged.py` 转换，**int8** 容器，约 7 GB | `make -C c olmoe` | 无 |
| **Qwen-Image-2.1**（Alibaba） | 图像模型 | [`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1)（约 33 GB），官方 diffusers checkpoint，**无需转换**：文本编码器与扩散 transformer 在加载时量化为 int8。`coli chat` 中直接显示图片，`coli serve` 提供 `POST /v1/images/generations`。Qwen Research License：仅限非商业用途 | `make -C c qwenimage` | [qwen-image.md](docs/qwen-image.md) |
| **Laya**（Convai Innovations） | 决策模型，421M | [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya)（842 MB），官方 checkpoint，**无需转换**：ModernBERT 编码器加决策头，对类型化问题（choice、score、noul）给出校准后的概率，而不是生成文本。由 `coli serve` 在 `POST /v1/systemone` 上提供。Apache-2.0 | `make -C c laya` | [laya.md](docs/laya.md) |
| **GLiNER2.5-Decide**（fastino） | 决策模型，340M | [`fastino/GLiNER2.5-Decide`](https://huggingface.co/fastino/GLiNER2.5-Decide)（1.9 GB），官方 checkpoint，**无需转换**：DeBERTa-v3 编码器加 GLiNER2 的分类头，一次读完请求中的所有问题和状态，为每个选项给出概率。由 `coli serve` 在 `POST /v1/systemone` 上提供。Apache-2.0 | `make -C c gliner_decide` | [gliner_decide.md](docs/gliner_decide.md) |

Qwen3.6 提供三个预转换容器：**int4-gs64**（推荐：与 per-row 相比，对 int8 基准的余弦相似度
实测从 0.98777 提升到 0.99313，KL 从 0.109 降到 0.080，即量化误差减少约 44%）、作为 A/B 基线的
[int4 per-row](https://huggingface.co/Kreuzzelg/qwen36-35b-a3b-colibri-i4)，以及
[KAT-Coder v2.5](https://huggingface.co/Kreuzzelg/kat-coder-v2.5-dev-colibri-i4-gs64)，同一引擎可以
直接运行它：任何架构相同的 checkpoint 都无需专门的代码路径。使用 `CUDA=1` 时，VRAM 专家层级
**在两张 8 GB 显卡上实测 1.44 → 10.05 tok/s（7.0×）**，输出与 CPU 路径逐位一致。

Qwen3.8-Flash-Next 按发布格式以 block-FP8 读取路由专家。可选的 **int4-g64 sidecar**
（`c/tools/convert_qwen38_experts_int4.py`，在 FP8 shard 旁写入 68.0 GB）让每次未命中只读取
56% 的字节。在 Ryzen 7 PRO 8700GE（16 线程，61 GiB，NVMe）上实测，相同缓存大小下解码快
1.4-1.5 倍，相同 RAM 下快 1.56 倍，困惑度平均每个 token 增加 +0.017 nats
（[qwen38.md](docs/qwen38.md#routed-experts-as-int4-g64)）。使用 checkpoint 自带 MTP head 的
推测解码为可选功能（`Q38_MTP=1`），输出与普通解码完全相同。在同一台机器上使用 int4 专家时，
94-96% 的草稿被接受，每次前向传播 1.94 个 token，tok/s 提升 +12-14%（cap 96 时从 3.57 到 4.01，
cap 170 时从 4.15 到 4.74）。在那台机器上收益不大，因为从磁盘读取专家的量并没有减少
（[qwen38.md](docs/qwen38.md)）。

Kimi K3 无需转换：其 QAT 训练的 MXFP4 专家直接从原始 Hugging Face shard 流式读取，bf16 稠密部分
在加载时量化。长时间的 agent 会话可以选用循环状态检查点（RAM 中 `COLI_K3_CKPT=N` 个 slot，或用
`COLI_K3_CKPT_DIR` 存到磁盘）：编辑过的或后续的 prompt 会恢复仍然保留的最深检查点，只对尾部重新
prefill，而不是让整个对话重新经过 SSM 层回放。在 Vulkan 主机上（`COLI_VULKAN=1`），路由专家进入
共享的专家层级：按专家历史预先填充，并随路由变化逐出。引擎的 KDA 与 MLA 路径在 CI 中与厂商实现逐
token 对照验证。

Inkling 提供 int4 专家，但稠密权重为 **bf16**（常驻 49.4 GB）；对于放不下这些权重的主机，
[inkling.md](docs/inkling.md) 提供一个单次处理工具，把稠密部分降到 15.3 GB，让 975B 能在
25 GB 的机器上运行，并如实写明了其中的取舍。

### 3. 运行

```bash
COLI_MODEL=/nvme/glm52_i4 ./coli chat     # 自动检测 RAM 预算、缓存与 MTP
COLI_MODEL=/nvme/glm52_i4 ./coli plan     # 查看规划的 VRAM／RAM／磁盘配置
COLI_MODEL=/nvme/glm52_i4 ./coli doctor   # 只读就绪检查
COLI_MODEL=/nvme/glm52_i4 ./coli doctor --deep  # 严格的张量／shard／索引／镜像预检
COLI_MODEL=/nvme/glm52_i4 ./coli tune     # 测量并保存本机最快且安全的执行配置
./coli web  --model /nvme/glm52_i4        # API + 仪表盘，并打开浏览器
./coli serve --model /nvme/glm52_i4       # API + 仪表盘，不打开浏览器（headless）
```

#### System One 模式：问一个封闭式问题

人们向模型提出的大多数请求是一次选择，而不是一段文字：哪个队列、哪个结论、某个字段应取四个值中的哪一个。
System One 模式把允许的选项交给引擎，读出每个选项的概率，而不是生成文本：不生成任何内容，
答案不可能落在你的列表之外，并且每个答案都附带一个置信度（confidence），"模型没有把握"因此成为一个可以设阈值的数字。
它在全部十个模型家族上可用，运行在同一个服务器上，且按请求可选：不请求它的聊天，输出逐字节保持不变。

```bash
# 在 TUI 中：同一个模型，只是不再让它写
./coli chat --model /nvme/qwen36_i4_gs64
> /decide merge | request changes | close
> 340 lines, 8 files, no tests. CI is green but nothing covers that path.

# 从任何程序：向运行中的服务器发送一个 JSON 请求
curl -s http://127.0.0.1:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "340 lines, 8 files, no tests. CI is green but nothing covers that path.",
  "questions": {"review": {"type": "choice", "instructions": "What should the reviewer do?",
                           "criteria": {"merge": null, "request changes": null, "close": null}}}}'
```

`POST /v1/systemone` 使用与 TypeSafe 的 Jev API 相同的请求和回复：Jev 客户端只需更改 base URL 即可切换到 colibri。
对同一文档的多个问题只读取文档一次：在 Qwen3.6 上与在同一台 CPU 机器上生成同样答案相比的实测，
对同一文档的四个问题快 5.7 倍。完整说明、请求与回复格式、以及它不适用的情形见 [docs/systemone.md](docs/systemone.md)。
仪表盘中也有 System One 页面。


在 Windows 上，发布包附带 `coli.cmd`：双击即可快速开始，或在 cmd 或 PowerShell 中运行
`coli.cmd chat --model D:\glm52_i4`。在源码 checkout 中，同样的命令写作 `python coli chat --model
D:\glm52_i4`。`.exe` 文件是引擎，不是启动器：单独启动时没有可加载的模型，会立即退出。
引擎运行时是纯 C，python 只供一次性转换工具与可选的 API gateway 使用。

#### 同样的命令可运行任意模型

`coli` 会读取模型的 `config.json`，选出对应的引擎二进制文件，并渲染该家族的聊天模板，因此
**换模型时命令行无需任何改变**。只需构建一次想用的引擎，然后把 `COLI_MODEL` 指向正确的目录：

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

对于非 GLM 引擎，`coli chat` 会在本地启动 gateway 并把 TUI 连接上去，因此 TUI、API 与仪表盘
都经过同一个感知架构的聊天模板，你无需自己传入模板。

有两点因模型而异，都记录在各模型的页面中：

- **RAM 紧张主机上的 Inkling** 需要 int4 稠密容器和较小的专家缓存：
  `./coli chat --model /nvme/inkling_i4 --cap 2`（见 [inkling.md](docs/inkling.md)：默认的
  `--cap 8` 在常驻集之外还需要约 14 GB 缓存）。
- **Kimi K3** 直接从原始 checkpoint 流式读取其 MXFP4 专家，因此无需转换，但快照约为 1.6 TB
  （见 [kimi_k3.md](docs/kimi_k3.md)）。

### 4. 深入了解

| 主题 | 文档 |
|---|---|
| Benchmark、社区实测数据、质量测量 | [docs/benchmarks.md](docs/benchmarks.md) |
| 可复现的 benchmark 协议与最低报告要求 | [docs/benchmarking.md](docs/benchmarking.md) |
| 调优选项、策略、学习型缓存、预取 | [docs/tuning.md](docs/tuning.md) |
| Windows 11 原生构建（含 CUDA DLL） | [docs/windows.md](docs/windows.md) |
| CUDA 后端、VRAM 专家层级、全部常驻 | [docs/cuda.md](docs/cuda.md) |
| Vulkan 后端（任意 GPU：通过 RADV 支持 AMD，包括 ROCm 已放弃的显卡） | [docs/vulkan.md](docs/vulkan.md) |
| Apple Silicon Metal 后端 | [docs/metal.md](docs/metal.md) |
| OpenAI 兼容 API、KV slots、网页仪表盘 | [docs/api.md](docs/api.md) |
| System One 模式：对封闭的选项集打分而不是生成 | [docs/systemone.md](docs/systemone.md) |
| 实验性的层分段嵌入 ABI | [docs/segment-runtime.md](docs/segment-runtime.md) |
| 实验性的 tokenizer／嵌入／head Edge ABI | [docs/edge-runtime.md](docs/edge-runtime.md) |
| 语法强制草稿（结构化输出） | [docs/grammar-draft.md](docs/grammar-draft.md) |
| 环境变量完整清单 | [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) |

## DeepSeek V4

**DeepSeek V4 Flash** 直接流式读取官方 checkpoint，无需转换：路由专家保持**原生 fp4**，
稠密部分保持带 UE8M0 块缩放的 **fp8-e4m3**。MLA + DSA 稀疏注意力，43 层，256 个路由专家
加 1 个共享专家，top-6。支持 x86-64／aarch64 Linux 与 Windows／MSYS2（CPU），并提供可选的
CUDA 层级（Windows 运行时 DLL；Linux 通过 `CUDA=1` 直接链接，已在 WSL2 下验证），每个阶段都以
CPU 结果为准，并可逐阶段回退。

```bash
cd c
make deepseek-v4
python ./coli chat --model /path/to/DeepSeek-V4-Flash --ram 32
# 同一模型也支持：coli run / coli serve / coli web
# Windows CUDA 层级：make cuda-dsv4-dll CUDA_ARCH=portable  （RTX 50 上再加 make cuda-dsv4-dg-dll）
```

两个可选的 GPU 调节项是新增的，正在征集社区实测数据，二者默认关闭，未设置时输出逐字节不变：
`DSV4_HYBRID=1` 根据运行时测得的带宽，把 VRAM 层级的未命中分配给 GPU 填充分支和 CPU 分支；
`COLI_CUDA_MOE_DOUBLE=1`（在 `COLI_CUDA_MOE_BATCH=1` 之上）在当前层计算时，把下一层的完整专家集
预取到第二个 VRAM 存储区，VRAM 不足时回退到单个存储区。CUDA 层级现在也能在 Pascal 与 Turing
显卡（GTX 10／RTX 20 系列）上运行：使用 `CUDA_ARCH=portable-pre-ampere NO_TC=1` 构建。

贪心解码，一个 KV slot。工具调用通过 HTTP gateway 接入，使用 V4 原生的 prompt 与 DSML 调用块；
不支持语法约束。见[各引擎 API 矩阵](docs/api.md#tool-calling-support)。前缀检查点（内存中与
磁盘上）让 agent 会话和后续轮次在系统 prompt 首次 prefill 之后几秒内即可开始。在 RTX 5080 +
2 块 NVMe 上实测：3324 个 token 的 prefill 90 秒，8.3k token 的首轮约 4 分钟（仅一次），之后的
会话／轮次 6-9 秒，3k 上下文时解码约 1.6 tok/s，见 [docs/deepseek-v4.md](docs/deepseek-v4.md)。

**给它 RAM。** 43 × 256 个路由专家在磁盘上约 137 GiB，一个 token 会触及其中 301 个，因此专家缓存
命中率决定了 tok/s：`--ram` 是最有价值的单个调节项，而且它只改变速度，从不改变输出。

**推测草稿已实现，但默认关闭。** DSpark 的 markov 草稿器与完整 MTP 都已实现并验证：草稿可以节省
前向传播，但绝不会改变 token，因为每个被接受的 token 仍是目标模型自己的 argmax。在真实的多轮
对话中实测，它们分别只接受了 15 个中的 1 个和 24 个中的 10 个，而本引擎为被拒绝的后缀回放循环
注意力状态所花的代价，超过了草稿节省的时间：一个 14 个 token 的回答用了 495 秒。因此 `V4_DRAFT`
与 `V4_MTP` 默认为 `0`，代码连同这些数据一起保留，留给在更快的存储上重试的人。

CUDA 层级（构建、DLL 选择、GPU 覆盖范围）、环境变量参考、性能数据、checkpoint 验证以及动态生成的
tiny 独立 oracle，见 [docs/deepseek-v4.md](docs/deepseek-v4.md)；中文说明见
[中文版 DeepSeek V4 文档](docs/deepseek-v4.zh-CN.md)。

## 下一步

- **推理系统研究就是产品**。当前层级采用 LRU 与学习型固定集；正在研究模型格式、压缩、
  放置、调度、I/O、CPU/GPU 内核、异构重叠、KV 状态与路由感知推测。目标是降低硬件要求
  和每个有效 token 的成本，所有成果都以端到端测量为准、经审查并公开开发。
- **支持更多开放模型**。层级算法与模型无关，任何带路由专家的 MoE 都能用相同方式分层。
  目前已有十个语言模型家族可用（GLM-5.2/5.3、GLM-5.3-Flash、Inkling、Kimi K3、DeepSeek V4 Flash、
  DeepSeek V4.1 Flash、MiMo-V2.6、Qwen3.8-Flash-Next、Qwen3.6、OLMoE），另有用于图像的
  Qwen-Image-2.1；更多开放权重家族（候选包括 **MiniMax**）将沿用同样的规则获得引擎支持：
  有人完成端到端实测之后。

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
├── colibri.c             GLM-5.2 引擎  (make glm)
├── inkling.c             Inkling 引擎  (make inkling)
├── kimi_k3.c             Kimi K3 引擎  (make kimi_k3)
├── glm53.c               GLM-5.3-Flash 引擎  (make glm53)
├── deepseek_v4.c         DeepSeek V4 Flash 引擎  (make deepseek-v4)
├── deepseek_v41.c        DeepSeek V4.1 Flash 引擎  (make deepseek_v41)
├── mimo.c                MiMo-V2.6 Flash 与 Pro 引擎  (make mimo)
├── qwen38.c              Qwen3.8-Flash-Next 引擎  (make qwen38)
├── qwen36.c              Qwen3.6、Qwen3-Coder、Qwen3.8-27B 引擎  (make qwen36)
├── olmoe.c               OLMoE 引擎  (make olmoe)
├── qwenimage.c           Qwen-Image-2.1 引擎  (make qwenimage)
│
├── st.h                  safetensors 索引与范围读取
├── quant.h               规范的容器解码器
├── expert_ffn.h          各 MoE 引擎共用的路由专家 FFN 内核（planar int4、层运行器）
├── tok.h, json.h         tokenizer 与 JSON 解析器
├── compat.h              Windows/macOS 兼容层（POSIX 名称集中在一处）
├── expert_store.h        流式专家缓存
├── route_trace.h         路由遥测与 .coli_usage，与引擎无关
├── kv_prefix.h           跨轮次的 KV 前缀复用
│
├── backend_cuda.*        可选的 CUDA 层级   (CUDA=1)
├── backend_metal.*       可选的 Metal 层级  (METAL=1)
├── backend_vulkan.*      可选的 Vulkan 层级 (VK=1)
│
├── Makefile              构建与本地检查
├── coli                  用户界面 CLI
├── openai_server.py      OpenAI 兼容 HTTP gateway
├── resource_plan.py      `coli plan` 与 `coli doctor` 背后的 RAM/VRAM 规划器
├── tools/                离线转换、fixtures 与 benchmarks
├── scripts/              长时间转换辅助工具
└── tests/                零依赖的 C 与 Python 测试
web/                      浏览器 UI（纯 OpenAI API client）
desktop/                  封装网页 UI 的 Tauri v2 桌面 shell
docker/                   容器镜像
docs/                     参考文档、实验与媒体文件
```

**每个模型家族一个 `.c`，建立在共享的单头文件之上。** 一个引擎只负责自己的架构；两个引擎都需要
的东西（safetensors 读取器、容器解码器、tokenizer、专家缓存）都放在它们共同包含的头文件里，
这样一个修复能同时覆盖所有引擎。这条规则不是装饰：这里反复出现的缺陷，正是某个机制只落在一个
引擎里、从未传到同级引擎的那些。

在仓库根目录执行 `make`、`make check` 与 `make clean`，
都会转发给引擎的 Makefile。

## 为什么叫"colibrì"

蜂鸟只有几克重，能在原地悬停，并在一天内造访上千朵花。
这套引擎只用蜂鸟般的配给，就能让 744B 参数的巨人运转：
25 GB RAM、十二个 CPU 核心，以及对磁盘的大量耐心。

## 致谢

colibrì 是一个引擎；它运行的智慧是一份馈赠。感谢以开放方式发布前沿级权重的团队：**Z.ai**
（GLM）、**Moonshot AI**（Kimi）、**Alibaba Qwen**、**MiniMax** 与 **Allen AI**（OLMoE），也感谢
每一位做过 benchmark、二分定位问题、复现图谱运行或提交补丁的贡献者。这个项目证明了开放权重
能够带来什么。

本项目在专家放置、压缩与路由方面的实验，也建立在以下开放研究与系统工作的思路和证据之上：

- [REAP](https://github.com/CerebrasResearch/reap) 与
  [EASY-EP](https://github.com/RUCAIBox/EASYEP)：输出感知的与特定领域的专家重要性。
- [SERE](https://github.com/JL-Cheng/SERE)：基于相似度的专家重路由；
  [ReMoE](https://github.com/BUAA-OSCAR/ReMoE)：感知缓存局部性的路由器微调。
- [MC-SMoE](https://github.com/UNITES-Lab/MC-SMoE)：路由引导的专家合并与压缩。
- [MoBE](https://github.com/inclusionAI/MoBE) 与
  [D²-MoE](https://github.com/lliai/D2MoE)：共享专家基与低秩专家增量。
- [HybriMoE](https://github.com/PKU-SEC-Lab/HybriMoE)：CPU/GPU 混合专家调度；
  [ScMoE](https://arxiv.org/abs/2404.05019)：专家通信与计算的重叠；
  [OD-MoE](https://arxiv.org/abs/2512.03927)：分布式按需专家加载。
- [vLLM](https://github.com/vllm-project/vllm)、
  [llama.cpp](https://github.com/ggml-org/llama.cpp) 与
  [kTransformers](https://github.com/kvcache-ai/ktransformers)：开放的推理系统与专家卸载工作，
  使对比得以复现。

引擎也建立在具体的工程成果之上，而不只是思路。以下每一项如今都在代码树中被使用或重新实现：

- [safetensors](https://github.com/huggingface/safetensors)：每个引擎读取的容器格式
  （`c/st.h`），包括其 fp8 与 I64 数据类型。
- [tiktoken](https://github.com/openai/tiktoken)：`c/tok.h` 精确地重新实现了它的
  `byte_pair_encode`，合并拼接后词表 id 最小的相邻对，因此源自 tiktoken 的词表不需要 merges 列表。
- [llama.cpp](https://github.com/ggml-org/llama.cpp)：`c/grammar.h` 中的 GBNF 语法子集遵循它的
  语法与 set-of-stacks PDA，Metal 路径也借用了它的 `newBufferWithBytesNoCopy` 常驻技巧。
- [vLLM](https://github.com/vllm-project/vllm)：引擎逐位置对齐的输出语义参考（例如最终 norm
  相对于 LM head 的位置）。
- [transformers](https://github.com/huggingface/transformers)：oracle；CI 以它为基准逐 token
  复现一个随机初始化的模型。
- [DietGPU](https://github.com/facebookresearch/dietgpu)：实验性压缩专家层级（`COLI_ANS`）背后的
  GPU ANS 编解码器。
- [rocWMMA](https://github.com/ROCm/rocWMMA)：HIP 后端把 CUDA 的 `nvcuda::wmma`
  fragment/mma_sync API 映射到它之上（`c/backend_gpu_compat.h`），这让同一份 .cu 源码可以为
  两家厂商编译。

## 许可证

Apache 2.0，Copyright 2026 Vincenzo Fornaro。详见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。GLM-5.2 权重由 Z.ai 以 MIT 许可发布。
