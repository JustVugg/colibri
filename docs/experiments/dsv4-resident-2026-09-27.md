# 六张 RTX 5090：DeepSeek V4 驻留专家路径测试

测试日期：2026-09-27。主机：yuesheng-gpu。模型：
`/data/models/DeepSeek-V4-Flash-0731`。原始结果见
[dsv4-resident-2026-09-27.json](dsv4-resident-2026-09-27.json)。

## 版本与口径

- base：`bfaee5b` 六卡层分配版，保留原二进制；其源码哈希已核对。
- new：`460f21cb`，CUDA-enabled C 单元在目标机重新编译链接；复用源码未变化的 generic CUDA backend 对象。
- fused：从 new 构建的诊断对照，仅把路由后的 resident 快路径条件改为恒假；保留 shared expert 的 w1/w3/w2 正确绑定。只重新编译 block 单元并链接相同其他对象。
- 六卡、单请求、OMP_NUM_THREADS=12、CTX=512、48 GiB memory budget；
  DSV4_CUDA_EXPERT_MIRRORS=2048、V4_LOADER_LANES=3；
  COLI_CUDA_ATTN_BATCH=1、COLI_CUDA_MOE_BATCH=1，未强制 batch 最低 token 数。
- 提示词：`用中文简短解释为什么天空是蓝色的。`；13 prompt tokens，生成上限 64，所有长文本样本实际生成 64 tokens。
- 解码速度按 `(generated - 1) / after_first` 计算；首 token 时间采用引擎日志，不含完整进程初始化。
- 每次启动独立进程，按 base1、new1、fused1、new2、new3、fused2、fused3 顺序串行测试；未清除系统 page cache，也未固定 CPU affinity。

## 结果

| 版本 | 次数 | 首 token 中位数（秒） | 解码中位数（token/s） | 各轮解码速度 |
|---|---:|---:|---:|---|
| base | 1 | 12.660 | 2.086 | 2.086 |
| fused | 3 | 13.536 | 2.080 | 2.021, 2.080, 2.088 |
| new | 3 | 12.414 | 2.158 | 2.142, 2.158, 2.190 |

在相同融合路径、相同输出下，new 相对 fused 的解码中位数提升 **3.77%**。
CPU expert requests 从 17,995 降至 11,299（约 -37.2%）；读取字节中位数从
57,849,151,488 降至 56,071,028,736（约 -3.1%）。跳过的多为 CPU cache hits，
因此请求数降幅不能当作磁盘 I/O 或吞吐改善幅度。

## 输出与边界

- new 三轮和 fused 三轮输出完全一致。
- base 与 new 文本不同：base 为“散射”，new/fused 为“散射得最厉害”，后文也不同。
  本样本表明差异来自启用正确绑定后的融合路径；不能称为对旧版逐 token 等价。
- new 算术冒烟 `What is 2 + 2? Answer with only the number.` 输出 `4` 后 EOS，正常退出。
- 所有进程正常退出，无运行失败。测试后六卡均回到 2 MiB、0% utilization，vLLM 仍停止，watchdog timer inactive。
- 仅一个短文本提示词的三轮比较及算术冒烟；没有验证 logits 数值误差、长上下文、并发、模型质量或生产稳定性。
- 约 4% 的小样本收益不能外推为普遍加速；当前仍为逐层执行，未实现 TP/EP。

远端完整 stdout/stderr、构建日志和诊断脚本保留在
`/data/test/colibri-resident-460f21cb/`，没有替换现有服务。
