# DeepSeek V4：保留投机前缀与 CUDA 批量验证

本轮从 PR #1766 的 `dbe79f77fbaf056f6e5807f21425ba58115bd1fb` 独立开发。
目标是定位 Colibri 与 Naruto 的单流差距，并在 Colibri 内实现可验证的改进。
后续结果见[延迟 indexer 与 CUDA drafter attention](dsv4-device-attention-2026-10-01.md)，本文保留本轮原始数据。
**尚未达到 Naruto 历史记录的 77.97 token/s。**

## 比较口径

- 主机：Yuesheng-gpu，6 × RTX 5090 32 GiB，CUDA 13.3，驱动 610.43.02，GCC 14.2。
- 同一模型目录 `/data/models/DeepSeek-V4-Flash-0731`，43 层、256 experts。
- Naruto 历史 long6 的六条原始模板化 prompt，SHA-256：
  `a977dec01cfd310d37b47fe16b884f9cacc56c00c9534a81a741e7cf8dcad87e`。
- C1 顺序生成，每条 256 token，每轮 1536 token；CTX=512，OMP_NUM_THREADS=12。
- 指标为整轮生成 token 数除以六次 session 总时间，含 prefill，排除模型加载。
  同次权重驻留，一轮预热、三轮正式，正式轮交替 target-only / MTP 顺序。
- Naruto 的 77.9747 是已有 `pipeline-batch-11` 日志的三轮中位数，本轮没有重跑 Naruto。
  对应 `2f63c38b + adapter11.patch` / `7763006c`，binary SHA-256：
  `4f29fb0c2e607b5b18e9c557729ad9cccba4b06621a672945da32511475e3f72`。
  两引擎并非相同数值执行路径，不能将速度比较解释为跨引擎输出等价。

## 慢在哪里

基线 target-only 12.3070 token/s；原默认 CPU-MTP 11.5801 token/s。
长文本下开启旧 MTP 并没有收益，且旧单 token / batch 路径有数值差异，六题仅两题输出相同。

一次独立的 warmed 64-token Nsight profile 显示：

- 74,208 次 `cudaMemcpyAsync`、25,305 次同步、81,797 次 kernel launch。
- copy API 累计 3.52 秒包含等待；实际 GPU 传输约 84 ms，不能诊断成纯 PCIe 带宽问题。
- GPU kernel 总时间约 3.11 秒；FP4 grouped 29.7%，compressor 投影 24.5%，
  FP8 matvec 18.3%，indexer head 8.3%。
- resident experts 原先直接拒绝 batch-union 接口，导致批量验证内部退回逐 token MoE。
- 拒绝草稿后恢复 attention 快照，并重新执行已接受前缀的整个 target。
- MTP dense/head 留在 CPU，expert GPU 缓存命中仍可能触发 CPU 缓存装载；
  三个 stage 争用同一 GPU 缓存。单次五 token 草稿约 0.4 秒。
- batch scratch 只缓存一个 device，层切换到另一张卡时反复释放和分配。

Naruto 的参考价值是执行粒度和状态寿命：设备工作区复用、保留已接受前缀、
真正的多行验证、统一单行/多行数学路径。其 TP2×PP3 与 Colibri 的连续层放置不同；
后者虽驻留六张卡，同一层仍只有一张卡计算。这里独立适配相关思路，没有引入 Naruto 运行时。

## 本轮实现

1. **投机前缀 journal**：记录验证实际消费的 KV 和 compressor/indexer 投影。
   回滚到原快照后仅重建已接受前缀的递归 attention 状态，不重跑 Transformer。
   journal 随验证 batch 有界增长；缺少投影或记录不完整时保留原 replay 回退。
   DSpark 只失效被拒绝的后缀。原快照继续负责故障回滚。
2. **resident MoE 真正批量化**：复用原 resident expert table，一次上传多行，
   每行保持相同路由排序、量化和归约次序，一次回传。单行接口复用同一实现。
3. **GPU drafter**：不可变 dense FP8 镜像归 engine 所有，五行 head 复用驻留 CUDA head；
   stage expert 缓存分布到三个设备，命中直接 grouped 计算，不先重载 CPU expert。
   GPU proposal 与 CPU drafter 不保证数值等价；所有输出仍由 target 验证，未将 draft 直接当作结果。
4. **统一 decode 路径**：单 token 也进入 batch=1 的同一 target 路径，
   避免 verifier 与 greedy 因不同数学路径产生分歧。这会改变部分旧路径输出，仍为显式开关。
5. **设备各自复用 scratch**：FP8/WO/mHC grow-only 工作区按 device 保存，engine close 统一释放。
   沿用当前单生成线程合同，不宣称新缓存支持多 engine 并发。
6. **精确投影预取**：compressor/indexer 提前读取 32 列，但保持原 FMA 或 mul/add 次序。
   配对小测降低延迟，整模型收益以 long6 为准。

## 开关与回退

新执行路径默认关闭；scratch 生命周期修复和逐位一致的投影预取直接生效。
实验组合需要以下设置，不能仅开 `V4_MTP_PARTIAL_KEEP`：

```sh
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export DSV4_CUDA=1 DSV4_CUDA_DEVICES=0,1,2,3,4,5
export OMP_NUM_THREADS=12 CTX=512
export COLI_CUDA_ATTN_BATCH=1 COLI_CUDA_MOE_BATCH=1
export DSV4_CUDA_EXPERT_MIRRORS=2048 V4_LOADER_LANES=3
export COLI_V4_PREWARM=0 DSV4_CUDA_PIN_HOST=0
export V4_MTP=1 V4_MTP_GPU=1 V4_MTP_GPU_DENSE=1
export V4_MTP_GPU_MIRRORS=768 V4_MTP_GB=4 V4_MTP_CONF=0 V4_MTP_DRAFT=5
export V4_NGRAM=0 V4_DRAFT=5 V4_MTP_PARTIAL_KEEP=1
export V4_SPEC_RETAIN=1 V4_RESIDENT_MOE_BATCH=1 V4_UNIFIED_DECODE=1
```

`V4_MTP_GB` 现有实现最大为 4，设置 12 不会得到 12 GiB。
本配置需要额外 drafter GPU 权重空间；并未承诺所有 GPU/模型布局都可驻留。
Windows loader 将新 batch 符号视为可选，旧 DLL 回退原路径。

## 验证

- CPU release 构建、25 项相关 Python 测试通过；Windows loader 交叉编译通过。
- attention journal 使用真实 compressor/indexer：ratio 0/4/128，窗口覆盖，
  分段记录，保留 0..6 行，缺失投影回退，序列化状态与直接执行一致；ASan/UBSan/LSan 通过。
- GPU placement mock：dense 镜像重排、scale 编码、上传失败、stage 缓存归属，
  多 device scratch 复用、扩容、释放；ASan/UBSan/LSan 通过。
- CUDA resident MoE：GPU 0/5，hidden/intermediate=128/128 及 4096/2048，
  batch=1/2/5/17/128，普通/固定路由，与逐行输出逐位一致；非法 expert ID 被拒绝。
- compressor/indexer 预取覆盖模型尺寸、非 32 对齐列、两张 GPU，精确 oracle 通过；默认路径 sky 64 token 与改动前逐 token 一致，重复四次。

## 完整 long6 结果

| 版本 / 模式 | 正式轮 1 | 正式轮 2 | 正式轮 3 | 中位 token/s |
| --- | ---: | ---: | ---: | ---: |
| 新版 unified target-only | 11.9546 | 11.9497 | 11.7676 | **11.9497** |
| 新版 retained GPU-MTP | 13.6656 | 13.6030 | 13.3954 | **13.6030** |

相对原始 target-only 12.3070 提升 **10.53%**；相对新版同次驻留 unified target-only 提升 **13.84%**。
统一路径自身未提速，甚至比原路径稍慢；它解决的是单行/多行 verifier 一致性。
不能用新 target-only 的较低基线把整个改动宣传为 13.84% 提升。
Naruto 的历史 77.9747 仍约为当前 5.73 倍，追平目标尚未完成。

- 四轮 × 两模式 × 六题 = 48 个样本，每个均生成 256 token，无提前 EOS、运行错误或重复差异。
- 24 个 MTP 样本全部与同版本 unified target-only 的逐 token 输出一致。
- 三轮正式 MTP 均提议 1744 个、接受 1167 个 token；其余输出仍经 target 纠正。
- 纯 decode 中位数 14.5651 token/s，不能将它和含 prefill 的 77.9747 混为同一计时口径。
- [机器可读结果](dsv4-retained-verification-2026-10-01-results.json) 保存三轮时间和计数。

## 原始材料与未完成边界

本地证据目录：`/home/Kei/colibri/result/dsv4-opt-20261001`。
基线：`/home/Kei/colibri/result/dsv4-pr1766-20261001`。
最终 CUDA 隔离目录：`/data/test/colibri-opt-pool-20261001`。
所有模型实验均使用 `/data/test`，没有修改远端脏工作树 `/data/src/colibri`。

仓库已有部分 TP2/设备 attention 原语，但当前生产 C 路径没有调用这些整段接口；
`dsv4_cuda_attention_sparse_batch` 仍是返回 0 的 stub。不能把已有函数名或环境开关当成已完成的整模型实现。

尚未实现：完整 attention/递归状态的 GPU 驻留、整层设备执行、TP2 协同计算，
以及与 Naruto 相同粒度的设备批处理。还没有长上下文、并发、CPU 全矩阵、
Windows DLL 实机验收。六题 greedy 一致只是这组回归证据，不是全模型质量保证。

后续应先闭合完整执行路径，再继续 kernel 调参：

| 缺口 | 当前边界 | 下一项可验证交付 |
| --- | --- | --- |
| 层内往返 | norm/RoPE/递归状态仍由 CPU 推进 | 同一设备完成 attention 与 MoE，并验证单行/多行、ratio-4/128、窗口覆盖、回滚一致性 |
| tensor parallel | 连续层分卡，每层单 GPU | TP2 权重分片、collective 顺序和故障结算；不能只复制权重后切一半计算 |
| 多行设备执行 | resident MoE 已批量，完整 sparse attention batch 仍缺失 | 一次设备事务覆盖整个 verifier block，保留已接受前缀 |
| 数值合同 | 当前六题 greedy 一致 | 扩大上下文与 token/logit 对照后，才能启用更快的 Tensor Core 数学路径 |

这些是尚未完成的实现与验收项，不是已经获得的性能收益，也不保证单项完成就达到 70 token/s。

用户授权停止 vLLM 且不要恢复；实验结束保持容器退出、watchdog 停用。
