# Naruto 对照：原生量化、并行路由与 Indexer 权重驻留

## 对照范围与来源

Colibri 基线为 `96d2000e60fbd140742fd768f0b501335a282913`。
Naruto 本地最初停在 `66347724ee453adda94fdff8e595a3d48fd0a62f`；本轮只读
fetch 后核对远端 `d0a00474a781a623197f24ae875b46d14d943704`（9 月 27 日），
没有切换其工作树，也没有操作 Naruto 的常驻主机。

固定版本的设计证据：

- [量化转换](https://github.com/GeetoRinku/naruto/blob/d0a00474a781a623197f24ae875b46d14d943704/backend/cuda/cuda_deepseek_quantization.cuh)：
  E4M3 硬件转换及旧架构位解码、E2M1 位解码；软件编码不穷举 255 个值。
- [Attention 执行程序](https://github.com/GeetoRinku/naruto/blob/d0a00474a781a623197f24ae875b46d14d943704/core/model/deepseek_v4_attention.cc)及
  [执行图](https://github.com/GeetoRinku/naruto/blob/d0a00474a781a623197f24ae875b46d14d943704/core/model/deepseek_v4_attention_graph.cc)：
  投影、归一化、RoPE、稀疏 Attention 和输出投影使用设备缓冲；满足准入条件时复用执行图。
- [输入执行岛](https://github.com/GeetoRinku/naruto/blob/d0a00474a781a623197f24ae875b46d14d943704/docs/reports/deepseek-v4-attention-input-island-2026-09-26.md)：
  压缩器部分状态所有权与辅助执行流，而非仅更换一个矩阵核。
- [小核融合验收](https://github.com/GeetoRinku/naruto/blob/d0a00474a781a623197f24ae875b46d14d943704/docs/reports/deepseek-v4-attention-kernel-fusion-2026-09-27.md)：
  bf16 影子覆盖所有生产者，KV/压缩器保留归一化与 RoPE 融合；q 头融合因重复计算角度表退回分步。
  报告中的最新融合收益在噪声内，不应把它单独当成巨大提速来源。
- [TP2 合同](https://github.com/GeetoRinku/naruto/blob/d0a00474a781a623197f24ae875b46d14d943704/core/model/deepseek_v4_tp.cc)：
  真正的两 rank 切分和集合通信。不能直接替换六张 5090 的连续分层放置。

Naruto 报告包含 GB10 双机、DSpark 每轮接受多个 token、C16 聚合吞吐等口径；
本轮是六张 RTX 5090、单请求、无推测的逐 token 解码。承认其执行架构优势，
但不把不同硬件、提示词、推测模式和并发数的速度直接相除。

## 差距与本轮边界

| 路径 | Colibri 基线 | Naruto 参考方式 | 本轮处理 |
| --- | --- | --- | --- |
| FP8 舍入 | 每元素遍历 255 个编码 | 直接编码 | 原生 RNE 转换，保留 Colibri 原数值合同 |
| FP8/FP4 解码 | 内循环 `ldexpf` / 局部表 | 硬件转换 / 位解码 | 已接入所有对应 generic 核，包括 Indexer 精确核 |
| 驻留 MoE 路由 | 单线程重复算概率、选 Top-6 | 设备路由与并行计算 | 接入 Colibri 已有 warp Top-6；原串行入口保留作 oracle |
| Indexer 查询权重 | 每次调用上传 8 MiB | 权重随模型驻留 | 随层上传一次，保留原布局、浮点 scale 和归约顺序 |
| Attention 中间激活 | 多次 H2D/D2H、CPU 归一化/RoPE | 设备执行程序与执行图 | 尚未闭合；不能称 Attention 已完整迁移 |
| 压缩器、Indexer 状态 | CPU 是权威状态，GPU 做部分计算 | 设备状态平面与执行流 | 后续需一起处理前缀、回滚、窗口/压缩边界 |
| 多 GPU | 43 层依次经过六卡 | TP2/EP2 与通信 | 未改变，无六路并行计算声明 |
| 推测与批处理 | 本轮 target-only | DSpark、批量验证 | 未开启；不混入此次收益 |

本轮默认启用原生转换、驻留路径的 warp 路由。在 `COLI_CUDA_ATTN_BATCH=1`
这条既有 CUDA Attention 路径中，Indexer 驻留也默认开启，
`DSV4_CUDA_RESIDENT_INDEXER=0` 可恢复按次上传。没有改变该既有 Attention 开关的默认值。

Indexer 镜像由所属层持有，使用该层设备，在专家全驻留预算检查前上传并计入
`uploaded_bytes`，引擎关闭随层释放。rows8 与普通行布局分别使用原精确核；
不替换为会改变求和顺序的普通 GEMM。新 DLL 符号是可选的，旧 DLL 或镜像不可用时
走原路径。专用镜像不能被误当成 BF16 或普通 FP8 matvec。

量化只借鉴 Naruto 的转换方式：它的软件编码在中点处取较低编码，而 Colibri
原路径取偶数编码。因此没有直接复制其编码规则，也没有改变激活尺度计算、
舍入位置、专家归约顺序或 Indexer Top-k 顺序。

## CUDA 时间线证据

基线 `nsys` 采样延迟 82 秒、窗口 12 秒；该窗口含约 73 个 token 的层调用，
边界不一定恰好是完整 token。kernel 时间占比不是整步墙钟占比。

- `route_top6_serial`：3,148 次，平均 297.4 µs、中位 319.9 µs，
  累计 0.936 秒，占采样 kernel 总时间 19.9%。
- 8 MiB H2D：**1,537 次**，累计 **12.0078 GiB**、设备复制耗时 **1.233 秒**。
  源码定位到 `dsv4_cuda_fp8_ref_matmul` 每次重新上传 Indexer 查询矩阵。
- `cudaMemcpyAsync` 的主机 API 累计耗时 6.50 秒，包含 pageable 内存及等待；
  不将它误报为纯 PCIe 传输时间。
- FP8 `mv<8>` 占 kernel 时间 17.1%，Indexer 精确投影占 10.1%，
  `fp8_sim` 占 2.6%。量化并不是唯一瓶颈。

候选使用相同 82 秒延迟、12 秒窗口进行结构复核：采到 2,322 次层路由
（相当于 54 个 token × 43 层）和 1,134 次 Indexer 精确投影，**8 MiB H2D 为零**，
最大的 H2D 为 128 KiB 激活。warp Top-6 平均 4.17 µs，`fp8_sim` 平均 1.04 µs。
候选提前完成生成，窗口还包含关闭阶段的 `cudaFree`；不直接比较两份 API 百分比，
也不把窗口内不同 token 数当成正式吞吐结果。

候选额外开启 `DSV4_DECODE_PROF=1` 的 128-token 样本得到 126 个完整 43 层记录：
Attention 中位 **61 ms**、MoE **16 ms**、mHC **7 ms**。这些值不含所有步间与
输出头开销，不能相加当成完整 token 延迟。Attention 仍是后续迁移重点。

## 验证与测量

主机 `yuesheng-gpu`，六张 RTX 5090，驱动 610.43.02，CUDA 13.3.73，
Debian GCC/G++ 14.2。模型 `/data/models/DeepSeek-V4-Flash-0731`。
构建使用 `make -C c -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 CUDA_ARCH=sm_120 -j8`。

已完成的检查：

- `test_dsv4_quant_cuda.cu`：GPU 0/5 各 35,769 个编码/舍入输入无差异。
  覆盖全部 E4M3 与 E2M1 编码、正负零、NaN 分类、全部相邻可表示数的中点和
  `nextafter` 两侧，以及范围内所有有限 BF16 值。对有限解码检查位模式，
  对编码与舍入检查原穷举实现。SM120 与 compute_80 PTX 两种编译都通过；
  后者仍运行于 5090，仅验证旧架构分支，不代表真实 SM80 硬件验收。
- `test_dsv4_resident_route.cpp`：GPU 0/5，普通/hash 路由、带/不带 bias、
  全同分 logits，对原串行 router + MoE 路径输出精确一致。
- `test_dsv4_resident_indexer.cpp`：GPU 0/5，row-major/rows8，batch 1/3/33，
  对原按次上传路径输出精确一致；修改源权重后设备副本仍保持原结果；
  无效维度、超限批次和误用 generic matvec 被拒绝。
  `compute-sanitizer --tool memcheck --error-exitcode=99 ./test_indexer` 报告 0 errors。
- CPU 构建、Windows loader 交叉编译、GPU placement mock 及其 ASan/UBSan 检查通过。

CUDA 小测试可在上述构建后运行：

```sh
/usr/local/cuda/bin/nvcc -O3 -arch=sm_120 c/tests/test_dsv4_quant_cuda.cu -o test_quant
./test_quant
g++ -O2 c/tests/test_dsv4_resident_indexer.cpp c/backend_cuda_dsv4.o \
  -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs \
  -Wl,-rpath,/usr/local/cuda/lib64 -lcudart -lcuda -lcublasLt -o test_indexer
./test_indexer
```

测试使用设备 0 和 5，需要至少六张可见 CUDA GPU。

吞吐 fixture：提示词 `用中文简短解释为什么天空是蓝色的。`，13 prompt tokens、
64 generated tokens，每次新进程；decode = `(64 - 1) / after_first_seconds`，
不包含预载和首 token。基线/量化路由候选交替三组，然后完整候选三次；
不是多提示词质量评估，也不是长上下文、并发或常驻服务验收。

环境固定为 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5`、`DSV4_CUDA=1`、
`DSV4_CUDA_DEVICES=0,1,2,3,4,5`、`OMP_NUM_THREADS=12`、`CTX=512`、
`COLI_CUDA_ATTN_BATCH=1`、`COLI_CUDA_MOE_BATCH=1`、
`DSV4_CUDA_EXPERT_MIRRORS=2048`、`V4_LOADER_LANES=3`、
`COLI_V4_SAVE_USAGE=0`、`V4_PREFIX_CKPT_DISK=0`、`COLI_V4_PREWARM=0`。
调用 `deepseek_v4 MODEL PROMPT --max-tokens 64 --memory-gb 48`，
未设置任何新的驻留开关；性能样本不启用 profiler。

| 版本 | 三次 decode token/s | 中位 token/s | 加载后 TTFT 中位 | 进程总时长中位 |
| --- | --- | --- | --- | --- |
| 基线 `96d2000e` | 6.230 / 6.019 / 6.089 | **6.089** | 0.942 s | 90.162 s |
| 原生量化 + warp 路由 | 7.480 / 7.719 / 7.702 | **7.702** | 0.637 s | 87.651 s |
| 再加 Indexer 驻留 | 8.872 / 9.277 / 8.906 | **8.906** | 0.621 s | 86.319 s |

完整候选比基线中位快 **46.27%**；单独量化/路由组合快 **26.49%**。
九次均退出 0、输出逐字一致，专家请求均为预载的 11,008 次。
完整候选记录 21 个 Indexer 镜像，每个 8,390,656 字节，共约 168 MiB。
预载仍约 72 秒；进程总时长中位只降低约 4.3%，不能把 decode 提升解释成
短请求冷启动同幅度提速。正式样本与日志摘要见[原始结果](dsv4-naruto-gap-2026-09-28.json)。

制品身份（基于基线的 dirty source 构建，最终提交仅增加报告，不改这些源文件）：

| 项目 | SHA-256 |
| --- | --- |
| 基线 binary | `417a4b3fc0ca4d293292c996730e3d79f579a48507252a3ce6914b987b644aa5` |
| 量化/路由 binary | `75079184e220019f12dff84630ccb18256719c587eac54f1bf14551d974be536` |
| 完整候选 binary | `4a981f7de70d730fb07ca370d33bf9fa05d274dbabf94a808ce765dd5ea8ec1f` |
| 完整候选 `deepseek_v4.c` | `454735d3229f938318399b869d73d05f64026eb189c15e1b72acbb5256342e9d` |
| 完整候选 `backend_cuda_dsv4.cu` | `fca02fe9fe993d7cd21bd3634482c36a2f96a9b4a8473e6d6586b39b758472f0` |
| 量化 header | `52c0d2523c96ed105bde6747de50dba3577ca4cc86e11064dccf7d926edf93a3` |

远端制品与日志分别位于 `/data/test/colibri-device-route`（基线 profiler）、
`/data/test/colibri-native-quant`、`/data/test/colibri-resident-indexer`。

## 下一段迁移的完整范围

下一段应闭合 Attention 的设备执行链：固定设备工作区，q/kv 投影和归一化/RoPE，
压缩器及 Indexer 状态推进、Top-k 和 KV 写入，稀疏 Attention、逆 RoPE、两段输出投影。
先保持层边界的输入/输出合同，仅在完整链条具备数值和状态 oracle 后减少往返并复用图。
需覆盖普通层、ratio-4/128、hash/learned route、前缀恢复、窗口环回、长上下文 Top-k 边界，
不能只靠短文本一致认定状态迁移完成。

本轮没有更改主机服务部署，vLLM 保持停止。
