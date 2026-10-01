# DeepSeek V4：decode 输出投影在设备上串联

基线：`7019d0af017a33d3f69b805ac59659843215c064`，PR #1766。
参考 Naruto `d0a00474a781a623197f24ae875b46d14d943704` 的
`core/model/deepseek_v4_attention.cc::ProjectOutput`：正常执行路径将 grouped
`wo_a` 和 `wo_b` 作为一个设备执行单元，中间结果留在 workspace。

## 改动与数值约定

Colibri 原 decode 路径：上传 attended → grouped wo_a → 下载 OA → CPU BF16
舍入 → 上传 OA → wo_b → 下载 output → CPU BF16 舍入。

新增 `dsv4_cuda_wo_decode` 在同一 stream 上完成两次原有 `run_mv` 和中间、
最终的 BF16 舍入，只上传 attended、下载 output。支持 wo_a 的 fmt8 / fmt9，
wo_b 的 fmt8。没有改变矩阵乘法 kernel 或归约顺序，没有额外量化 activation。
现有 `dsv4_cuda_wo` 会对 OA 额外做 FP8 round-trip，因此不能直接复用；其原合同不变。

对于 OA=8192 的形状，每次命中减少 32 KiB D2H、32 KiB H2D 和一次 stream
同步。这是调用链上的减少量，尚不是实测端到端收益。Attended 和最终输出仍经过
CPU，输入端 Q/KV、RoPE、Compressor 状态等也没有在本轮完全迁入设备。

三个单 token Attention 实现副本均接入，batched prefill 不变。两份权重必须位于
同一设备；缺少 mirror、不支持的形状/格式、后端拒绝或旧 Windows DLL 缺少可选
导出时，继续执行原路径。任何已提交的复制都在返回前排空，避免回退与 host buffer
写入竞争。

默认启用；`DSV4_CUDA_WO_DECODE=0` 回退原路径。无需启用 `COLI_CUDA_ATTN_BATCH`。

## 本轮验证

初始提交完成以下编译和短单测；后续真实模型 A/B 记录见下文。所有测试使用独立 CLI 进程，未启动模型服务。

- Linux CPU 完整 engine 构建通过。
- yuesheng-gpu 上 CUDA `sm_120` 完整 engine 和新测试构建通过。
- Windows loader 使用 MinGW 交叉编译通过。
- GPU 0、5：groups=1 / 8，wo_a fmt8 / fmt9，每种组合 3 组随机输入，合计
  24 次与旧的两次 matvec + host BF16 路径逐位相同。
  小形状 width=256、rank=128、hidden=256；实际尺寸 width=4096、rank=1024、hidden=4096。
- 测试覆盖缺失 tensor、空输出、无效 groups、维度/格式拒绝、跨设备权重。
  fmt9 测试还确认输入能区分旧 `dsv4_cuda_wo` 的额外 FP8 量化，防止错误复用。
- GPU placement mock 覆盖生产 wrapper 的默认开启、关闭、后端失败及两种 mirror 缺失。
- GPU placement mock 和 CUDA 架构标志测试共 8 项通过。

GPU 单测命令（需要 GPU 0 和 5）：

```sh
cd c
make -f Makefile.deepseek-v4 deepseek-v4 tests/test_dsv4_wo_decode_cuda CUDA=1 CUDA_ARCH=sm_120 -j16
timeout 45s ./tests/test_dsv4_wo_decode_cuda
```

主机隔离目录：`/data/test/colibri-wo-decode`。`baseline` 保留上一版本，
`c/deepseek_v4` 为本轮构建；`test.log` 保留 8 组 `exact=3/3` 输出。
CUDA binary SHA256：
`c5c0711ff2027466b2025009f85dd732b8415ba5bff29bbb5962a91171442995`。
CUDA source SHA256：
`85e05e110d9f4356e8fa763bc40da152e9209b5018777638c96ef091ad05af1d`。

## 六卡真实模型 A/B

模型 DeepSeek-V4-Flash-0731，六张 RTX 5090，基线 `7019d0af` 对比候选 `371ed4e3`。
两份二进制分别为 `baseline` 和 `c/deepseek_v4`，SHA256 记录在同名 JSON。
提示词：`用中文简短解释为什么天空是蓝色的。`，13 prompt tokens。

短测交替执行 A1、B1、A2、B2、A3、B3，每轮重新启动进程并预载专家，最多生成
64 tokens。decode 定义为 `(generated - 1) / after_first_seconds`，不含加载和首 token。

| 版本 | 三次 decode token/s | 中位数 |
| --- | --- | --- |
| 基线 | 9.6081 / 9.7644 / 9.5137 | 9.6081 |
| 输出投影串联 | 9.7297 / 9.8545 / 9.8100 | 9.8100 |

中位数增加 **2.10%**；六次均成功生成 64 tokens，文本及生成数量完全一致。
完整进程 wall 中位数从 86.873 秒变为 85.760 秒，包含预载波动，不应等同于 decode 增益。
三次短测只支持该 fixture 的小幅改善，不足以证明普遍收益，更不能解释与 Naruto 的主要差距。

环境：

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
DSV4_CUDA=1 DSV4_CUDA_DEVICES=0,1,2,3,4,5
OMP_NUM_THREADS=12 CTX=512
COLI_CUDA_ATTN_BATCH=1 COLI_CUDA_MOE_BATCH=1
DSV4_CUDA_EXPERT_MIRRORS=2048 V4_LOADER_LANES=3
COLI_V4_SAVE_USAGE=0 V4_PREFIX_CKPT_DISK=0 COLI_V4_PREWARM=0
```

命令参数为 `MODEL PROMPT --max-tokens 64 --memory-gb 48`。新旧版都使用相同配置；
常规候选运行不设置 `DSV4_CUDA_WO_DECODE`，测试默认启用行为。此处沿用已有的
`COLI_CUDA_ATTN_BATCH=1` 测速配置，并非所有选项都采用默认值。


补充验收：

- 上限 192 tokens 的完整回答：基线与候选均在 130 tokens 正常结束，文本和数量相同，
  覆盖 position=128 压缩边界。单次 decode 分别为 9.6535、9.8767 token/s；不并入短测中位数。
- 同一候选二进制设置 `DSV4_CUDA_WO_DECODE=0`：64-token 文本和数量与六次短测相同，
  decode 为 9.5483 token/s。关闭开关保留旧路径。
- 九次运行均退出 0，均完整驻留 43 层专家，`v4_direct fallbacks=0`。
- 结束时六张 GPU 均回到 2 MiB；vLLM 容器保持停止，watchdog timer inactive。

[原始结果、环境和二进制指纹](dsv4-attention-output-2026-09-28.json)。远端
`/data/test/colibri-wo-decode/wo-bench.py`、`results-base1.json`、
`results-base-long.json` 与各 case 的 `.out` / `.err` 保留复验记录。
