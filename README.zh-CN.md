<p align="center">
  <img src="assets/colibri.svg" width="500" alt="colibrì——小巧引擎，庞大模型">
</p>

<p align="center">
  <a href="README.md">English</a> · 简体中文 · <a href="README.zh-TW.md">繁體中文</a> · <a href="README.it.md">Italiano</a>
</p>

**小巧引擎，庞大模型。**只需约 25 GB 内存，就能在消费级电脑上运行 **GLM-5.2（744B 参数的 MoE）**——以零依赖的纯 C 实现，从磁盘流式加载专家。

Colibrì 是一套轻量、保持模型质量的 MoE 运行时，将 VRAM、RAM
与存储设备视为统一管理的内存层级。高速内存不足可能降低速度，
但默认策略**绝不会在未告知的情况下改变模型精度或路由语义**。

```
$ ./coli chat
  🐦 colibrì v1.0 — GLM-5.2 · 744B MoE · int4 · streaming CPU
  ✓ ready in 32s · resident 9.9 GB
  › ciao!
  ◆ Ciao! 😊 Come posso aiutarti oggi?
```

## 实际运行效果

<p align="center">
  <img src="docs/media/colibri-dashboard.png" width="900" alt="colibrì 网页仪表盘——实时指标、硬件面板与专家存储层级">
</p>
<p align="center"><em>网页仪表盘（<code>./coli web</code>）：744B 模型达到 <strong>4 tok/s、TTFT 1.6 秒、磁盘读取 0</strong>——
在 6× RTX 5090 上让所有专家常驻，并实时显示 token 指标、每轮耗时明细、
VRAM／RAM／磁盘层级条，以及角落的实时迷你大脑。</em></p>

<p align="center">
  <img src="docs/media/colibri-brain.png" width="900" alt="大脑页面——以实时皮层呈现 19,456 个专家">
</p>
<p align="center"><em><strong>大脑（Brain）</strong>页面：将全部 19,456 个专家呈现为活的皮层——颜色代表存储层级，
亮度代表路由热度，每轮被路由到的专家都会闪白。将光标停在专家上，即可查看其
<a href="https://github.com/JustVugg/colibri/issues/175">实测主题亲和度</a>。</em></p>

<p align="center">
  <img src="docs/media/colibri-atlas.png" width="900" alt="图谱页面——以 3D 星系呈现实测专家图谱">
</p>
<p align="center"><em><strong>图谱（Atlas）</strong>页面：将<a href="https://github.com/JustVugg/colibri/issues/175">实测专家图谱</a>
呈现为 3D 星系——共 13,260 个已分析专家，其中 1,041 个可复现的专门专家会按主题聚集
（诗歌、法律、中文、SQL……）。位置取自实测路由亲和度，而非学习出的嵌入向量。拖拽即可旋转。</em></p>

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

- **6× RTX 5090，全部常驻：**解码 5.8–6.8 tok/s，TTFT 约 13 秒
  （[实验记录](docs/experiments/glm52-6x5090-2026-07-12.md)）；
- **128 GB、仅使用 CPU 的台式机：**热缓存后约 1.8 tok/s
  （[#200](https://github.com/JustVugg/colibri/issues/200)）；
- **单张 RTX 5070 Ti 的笔记本级主机：**通过 GPU 常驻管线达到 1.07 tok/s
  （[#273](https://github.com/JustVugg/colibri/issues/273)）；
- **25 GB 开发机：**冷启动 0.05–0.1 tok/s——这是项目起步时已证实的下限，
  也仍是诚实的基准。

质量来自测量，而非假设：int4 容器的量化损失，以及 scale granularity／rotation
消融实验，收录于 [docs/benchmarks.md](docs/benchmarks.md#quality-benchmark)、
[#108](https://github.com/JustVugg/colibri/issues/108) 与
[#81](https://github.com/JustVugg/colibri/issues/81)。

## 开始使用

### 1. 获取模型

Hugging Face 上已有预转换的 **GLM-5.2 int4** 容器——请务必使用
**含 int8 MTP head 的版本**：

**https://huggingface.co/mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp**

> ⚠️ 原始镜像使用 int4 MTP head → 草稿接受率为 0%
>（[#8](https://github.com/JustVugg/colibri/issues/8)）。请检查你的版本：
> `ls -l <model>/out-mtp-*`——正确的 int8 大小为 `3527131672 / 5366238584 / 1065950496`。

你也可以自行从 FP8 源转换——只需一条可断点续传的命令，且任何时候都不需要
在磁盘上同时存放完整的 756 GB：

```bash
cd c && ./setup.sh                        # 检查 gcc/OpenMP、构建并运行自测
./coli convert --model /nvme/glm52_i4     # 逐 shard 下载并转换（仅此一次需要 python）
```

### 2. 运行

```bash
COLI_MODEL=/nvme/glm52_i4 ./coli chat     # 自动检测 RAM 预算、缓存与 MTP
COLI_MODEL=/nvme/glm52_i4 ./coli plan     # 查看规划的 VRAM／RAM／磁盘配置
COLI_MODEL=/nvme/glm52_i4 ./coli doctor   # 只读就绪检查
./coli web  --model /nvme/glm52_i4        # 在同一端口提供 API 与网页仪表盘
./coli serve --model /nvme/glm52_i4       # 仅提供 OpenAI 兼容 API
```

引擎运行时是纯 C——python 只供一次性转换工具与可选的 API gateway 使用。

### 3. 深入了解

| 主题 | 文档 |
|---|---|
| Benchmark、社区实测数据、质量测量 | [docs/benchmarks.md](docs/benchmarks.md) |
| 调优选项、策略、学习型缓存、预取 | [docs/tuning.md](docs/tuning.md) |
| Windows 11 原生构建（含 CUDA DLL） | [docs/windows.md](docs/windows.md) |
| CUDA 后端、VRAM 专家层级、全部常驻 | [docs/cuda.md](docs/cuda.md) |
| Apple Silicon Metal 后端 | [docs/metal.md](docs/metal.md) |
| OpenAI 兼容 API、KV slots、网页仪表盘 | [docs/api.md](docs/api.md) |
| 语法强制草稿（结构化输出） | [docs/grammar-draft.md](docs/grammar-draft.md) |
| 环境变量完整清单 | [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) |

## 支持项目

colibrì 最初由一人使用 12 核心、25 GB RAM 的笔记本开发；
如今它的数据来自社区中各种真实机器。如果这个项目对你有用：

- ⭐ 为仓库加星并分享；
- 🐛 以 issue 提交你的硬件 benchmark 数据——实测数据比任何其他事都更能推动项目；
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
docs/                     参考文档、实验与媒体文件
```

运行时路径刻意保持扁平、易读：`colibri.c` 加上若干头文件。
在仓库根目录执行 `make`、`make check` 与 `make clean`，
都会转发给引擎的 Makefile。

## 为什么叫"colibrì"

蜂鸟只有几克重，能在原地悬停，并在一天内造访上千朵花。
这套引擎只用蜂鸟般的配给，就能让 744B 参数的巨人运转：
25 GB RAM、十二个 CPU 核心，以及对磁盘的大量耐心。

## 许可证

Apache 2.0。GLM-5.2 权重由 Z.ai 以 MIT 许可发布。
