# int4 / int8(+GPU) / int8(CPU) 冷启动对比 — Qwen3.6-35B-A3B

**测试时间**：2026-07-25
**引擎**：`c/qwen36.exe`（纯 C，长问题 `c/prompt_emerge.txt` *"详细介绍一下LLM智能涌现的原理，不少于1000字。"*）
**运行方式**：命令行冷启动（非 serve），串行各跑一次，`N_NEW=256`
**硬件**：本机 16GB RAM，AMD 780M 核显（Vulkan，**共享系统内存、无独立显存**）
**命令**：`./qwen36.exe 8 <bits> C:/.../prompt_emerge.txt`，`SNAP`/`TOK` 指向对应精度目录；int8-CPU 额外 `COLIBRI_GPU=0`
**GPU 路径**：int4 = in-shader unpack+float GEMV（480MB 上传）；int8 = float path（960MB 上传，因 AMD 驱动 0x800184 的 OpSDotKHR bug 禁用 int8 dot-product）

## 对比表

| 指标 | int4 +GPU | int8 +GPU | int8 CPU-only |
|---|---|---|---|
| **首字延迟 TTFT** | 50.18 s | 28.68 s | **13.83 s** |
| **每秒速度** | 0.29 tok/s | 0.48 tok/s | **1.08 tok/s** |
| **PEAK RSS（内存）** | 11.30 GB | 11.27 GB | **10.25 GB** |
| 加载后 RSS | 9.25 GB | 9.25 GB | 9.24 GB |
| 专家缓存命中率 | 30.4% | 30.4% | 30.4% |
| 256 token 总耗时 | 873.2 s | 532.0 s | **238.0 s** |
| 生成内容 | 正确 | 正确 | 正确（细节措辞略异）|

## 关键发现

1. **内存**：加载后三者均 ~9.25GB；峰值 int8-CPU 最小（10.25GB，无 GPU 上传副本），int4/int8+GPU 因核显共享内存上传权重副本各 ~11.3GB。int8 未 OOM（16GB 充裕 + `cap=8` LRU 未填满）。

2. **速度排序：int8-CPU > int8+GPU > int4+GPU**。
   - int4+GPU 最慢（0.29）：int4 每专家需在 CPU 侧 unpack→int8 再传 GPU（日志 `"unpacking to int8 in slot"`），核显又无带宽优势，双重拖累。
   - int8+GPU 快于 int4+GPU（0.48）：走 GPU float path 免解包，但仍受共享内存上传开销。
   - int8-CPU 最快（1.08）：直接 CPU 算 int8，无 unpack、无 GPU 上传，反而最干净。

3. **TTFT**：int8-CPU 13.83s 最低（prefill 直接 CPU 算，无上传）；int4+GPU 50.18s 最高（prefill 也受 unpack 拖累）。

4. **生成内容**：三者主题一致、质量相当，均正确阐述 LLM 涌现的规模效应/相变理论。int8-CPU 细节措辞与 GPU 版略有差异（如"发生了非线性的跃迁" vs "呈现出非线性的跳跃"），到 256 token 截断点不同，但无质量退化。

## 结论

本机**核显（共享内存、无独显）冷启动**下真实排序：**int8 CPU-only 最优**（速度 3.7× int4-GPU、内存最小），其次 int8+GPU，最后 int4+GPU。GPU 在核显上不仅没加速，反而因 int4 解包 + 共享内存上传变慢。

**int4 唯一硬优势是磁盘占用小**（19.7GB vs 34.6GB）——仅当磁盘吃紧才选 int4。
**独显 + 常驻显存**机器上 GPU 优势才会真正显现（独立高带宽显存），该场景未测。

> 注：为测首字延迟，给 `qwen36.c` 非 OpenAI 文本模式加了 TTFT stderr 打印；bash 下 prompt 参数须用 `C:/...` 大写盘符（MinGW fopen 不认 `/c/`）。
