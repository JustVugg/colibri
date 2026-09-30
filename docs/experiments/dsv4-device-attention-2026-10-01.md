# DeepSeek V4：延迟 indexer 评分与 CUDA drafter attention

本轮基于 `094030b87436a1c6b92f7edb17e75f429d823c99`，继续缩小与 Naruto 的执行路径差距。
比较口径沿用[上一轮](dsv4-retained-verification-2026-10-01.md)：Yuesheng-gpu 六张 RTX 5090，
官方 DeepSeek-V4-Flash-0731，同一 long6、C1、每题 256 token，含 prefill、不含模型加载。
Naruto 的 77.9747 token/s 仍为历史记录，本轮没有重跑，也不宣称两引擎输出等价。

## 实现与语义

1. **延迟 indexer selection**：所有可见压缩行都在 top-k 内时，现有 CUDA attention
   直接按 cache 顺序消费完整集合，不读取 indexer 排名。仍推进 compressor/indexer 状态，
   仅跳过无用的评分投影、点积与排序。超过 top-k 时保持原算法；CUDA cached/slab
   两条路径都失败时，在 CPU attention 前补做原 selection，恢复原浮点累加顺序。
   `DSV4_CUDA_INDEXER_LAZY` 默认开启，设置 `0` 可禁用；CPU-only 路径不变。
2. **CUDA DSpark attention**：将 drafter 的 attention 点积、含 sink 的 softmax、value
   加权和移到 GPU。按绝对位置排除无效与未来的历史槽，保留整个 proposal block 的双向可见性。
   概率保持 FP32，输出 BF16；不能复用 target 中会将概率舍入为 BF16 的 kernel。
   复用已有 per-device scratch，失败后清空临时输出并回退 CPU。
   仅在已有 `V4_MTP_GPU_DENSE=1` 路径启用，`V4_MTP_GPU_ATTN=0` 可单独关闭。
   Windows DLL 符号可选，旧 DLL 回退 CPU。

GPU dot 的并行归约与 CPU 顺序归约不逐位等价，草稿可以变化，最终 token 仍须经过 target 验证。
RoPE、norm、mHC 和 Markov 状态仍有 CPU 工作，不称为完整 GPU drafter。
整层 GPU 执行、TP2、CUDA Graph 和 FP4 Tensor Core 替换不在这次实现内。

## 单项对照

在上一版 binary 上，仅用已有实验开关 `V4_IDX_IDENTITY=1` 跳过 full-set 评分，
一轮预热加一轮正式，共 24 个样本，全部与上一版 token 数组相同。
target-only 为 13.2321 token/s，MTP 为 14.6439 token/s。
该单轮实验用于定位收益，不作为三轮正式结果，也没有将这个较宽的实验开关作为新默认值。
最终版本使用 `V4_IDX_IDENTITY=0`，由上面的安全延迟逻辑处理回退。

## 验证

- indexer 回归测试使用真实 compressor/indexer 与 CPU attention，注入 CUDA cached/slab
  成功或失败：batch=1/6、top-k=2/3/4，共 18 组，关闭/开启延迟选择的输出逐位相同。
  同时断言 selection 调用次数及存在非 identity 的 full-set 排序，覆盖需要恢复原顺序的场景。
- 新 CUDA attention 对照顺序 CPU oracle：GPU 0/5，head dim=32/128/512，heads=1/64，
  history=0/8/128，block=5；无效/未来槽填 NaN，覆盖 sink、窗口掩码和双向块。
  最大相对 L2 为 `4.45329707e-05`；compute-sanitizer memcheck 为 0 错误。
- CPU release、CUDA CLI、CUDA benchmark 与新增测试构建通过；25 项相关 Python 测试、
  attention trial 和 Windows loader 交叉编译通过。

## 完整 long6 结果

一次驻留，一轮预热加三轮正式，两模式交替顺序：

| 模式 | 正式轮 1 | 正式轮 2 | 正式轮 3 | 中位 token/s |
| --- | ---: | ---: | ---: | ---: |
| unified target-only | 13.1373 | 13.1962 | 13.1743 | **13.1743** |
| retained GPU-MTP | 16.0158 | 16.0957 | 16.4020 | **16.0957** |

MTP 比上一版 13.6030 提升 **18.33%**，比最初 target-only 基线 12.3070 提升 **30.79%**。
纯 decode 中位数为 17.3372 token/s，主指标仍使用含 prefill 的 16.0957。
Naruto 历史 77.9747 约为本版的 **4.84 倍**，尚未达到 70+。

四轮 × 两模式 × 六题共 48 个样本，每个生成 256 token，无提前 EOS 或错误。
同模式重复、MTP 对同版本 target-only、所有样本对上一版 target-only，token 数组全部相同。
三轮正式 MTP 均 drafted=1734、accepted=1169，上一版为 1744/1167；接受率变化很小。
[机器可读结果](dsv4-device-attention-2026-10-01-results.json) 保存逐轮计时和计数。

## 分项归因

取前后版本的第一轮正式 MTP，同样生成 1536 token：

| 累计计时 | 上一版 | 本版 |
| --- | ---: | ---: |
| target block + head | 83.962 s | 76.863 s |
| draft total | 25.587 s | 16.017 s |
| draft 三个 stage | 20.803 s | 11.172 s |
| draft head | 4.463 s | 4.468 s |
| draft 调用数 | 350 | 348 |

草稿耗时降低约 37.4%，head 时间基本不变，收益主要落在 stage 内 attention。
indexer 评分省略也降低了 target 时间。计数器是同步分段日志，draft 时间输出到毫秒，
不能将这些累计值解释为 Nsight 的纯 GPU kernel 时间。
target 仍是主要开销；仅把 drafter 变快不足以达到 70+ token/s。

## 证据与边界

本地原始材料：`/home/Kei/colibri/result/dsv4-device-attention-20261001`。
远端隔离构建：`/data/test/colibri-device-attention-20261001`。
`experiment.json` 保存完整环境，long6 prompt SHA-256 为
`a977dec01cfd310d37b47fe16b884f9cacc56c00c9534a81a741e7cf8dcad87e`。
未改动 `/data/src/colibri`，也未恢复 vLLM 或 watchdog。

full-set selection 优化主要惠及候选数不超过 top-k 的上下文。长上下文整模型、并发、
Windows DLL 实机尚未验收；小规模数值测试和六题 greedy 对照不等于通用质量保证。
